"""
Tool: cve_enrich

Lightweight, non-agent CVE enrichment that fetches data from multiple public
sources **in parallel** and returns a single unified snapshot.  No LLM required.

Sources queried (all public, no API key required for basic operation):

  1. **NVD** — CVSS scores, CWE, affected CPE, references, published/modified dates
  2. **EPSS** — current exploitation probability score + percentile
  3. **CISA KEV** — active exploitation flag + remediation deadline
  4. **OSV.dev** — affected packages with version ranges + first-fixed versions

Optional (requires VULNCHECK_API_KEY):
  5. **VulnCheck KEV** — multi-source exploitation signal

The output is a flat, structured dict suitable for JSON serialization, CLI
display, or programmatic consumption by other tools/agents.

CLI: ``manus-agent enrich CVE-XXXX-YYYY``
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
from strands import tool

__all__ = ["cve_enrich"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)
_REQUEST_TIMEOUT = 15  # seconds per HTTP call
_NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_EPSS_API = "https://api.first.org/data/v1/epss"
_CISA_KEV_API = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
_OSV_API = "https://api.osv.dev/v1/vulns"
_VULNCHECK_KEV_API = "https://api.vulncheck.com/v3/index/vulncheck-kev"


# ---------------------------------------------------------------------------
# Individual source fetchers
# ---------------------------------------------------------------------------


def _fetch_nvd(cve_id: str) -> dict[str, Any]:
    """Fetch NVD record for a CVE. Returns structured data or error."""
    url = f"{_NVD_API}?cveId={cve_id}"
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY", "").strip()
    if api_key:
        headers["apiKey"] = api_key

    try:
        resp = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return {"error": f"No NVD record for {cve_id}"}

        cve_item = vulns[0].get("cve", {})

        # Extract CVSS v3.1 or v3.0
        metrics = cve_item.get("metrics", {})
        cvss_v31 = metrics.get("cvssMetricV31", [])
        cvss_v30 = metrics.get("cvssMetricV30", [])
        cvss_data = (cvss_v31[0] if cvss_v31 else cvss_v30[0] if cvss_v30 else {}).get("cvssData", {})

        # Extract CWE
        weaknesses = cve_item.get("weaknesses", [])
        cwe_ids = []
        for w in weaknesses:
            for desc in w.get("description", []):
                val = desc.get("value", "")
                if val.startswith("CWE-"):
                    cwe_ids.append(val)

        # Extract description
        descriptions = cve_item.get("descriptions", [])
        description = ""
        for d in descriptions:
            if d.get("lang") == "en":
                description = d.get("value", "")
                break
        if not description and descriptions:
            description = descriptions[0].get("value", "")

        # Extract references
        references = [
            {"url": ref.get("url", ""), "source": ref.get("source", ""), "tags": ref.get("tags", [])}
            for ref in cve_item.get("references", [])
        ]

        return {
            "source": "nvd",
            "cve_id": cve_id,
            "published": cve_item.get("published"),
            "last_modified": cve_item.get("lastModified"),
            "status": cve_item.get("vulnStatus"),
            "description": description,
            "cvss_score": cvss_data.get("baseScore"),
            "cvss_severity": cvss_data.get("baseSeverity"),
            "cvss_vector": cvss_data.get("vectorString"),
            "cwe_ids": cwe_ids,
            "reference_count": len(references),
            "references": references[:10],  # Cap to avoid huge output
        }
    except requests.exceptions.RequestException as exc:
        return {"source": "nvd", "error": str(exc)}


def _fetch_epss(cve_id: str) -> dict[str, Any]:
    """Fetch current EPSS score for a CVE."""
    url = f"{_EPSS_API}?cve={cve_id}"
    try:
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("data", [])
        if not results:
            return {"source": "epss", "score": None, "percentile": None}

        entry = results[0]
        return {
            "source": "epss",
            "score": float(entry.get("epss", 0)),
            "percentile": float(entry.get("percentile", 0)),
            "date": entry.get("date"),
        }
    except requests.exceptions.RequestException as exc:
        return {"source": "epss", "error": str(exc)}


def _fetch_cisa_kev(cve_id: str) -> dict[str, Any]:
    """Check if CVE is in CISA KEV catalog."""
    try:
        resp = requests.get(_CISA_KEV_API, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        for vuln in data.get("vulnerabilities", []):
            if vuln.get("cveID", "").upper() == cve_id.upper():
                return {
                    "source": "cisa_kev",
                    "in_kev": True,
                    "vendor": vuln.get("vendorProject"),
                    "product": vuln.get("product"),
                    "date_added": vuln.get("dateAdded"),
                    "due_date": vuln.get("dueDate"),
                    "required_action": vuln.get("requiredAction"),
                    "known_ransomware_use": vuln.get("knownRansomwareCampaignUse", "Unknown"),
                }
        return {"source": "cisa_kev", "in_kev": False}
    except requests.exceptions.RequestException as exc:
        return {"source": "cisa_kev", "error": str(exc)}


def _fetch_osv(cve_id: str) -> dict[str, Any]:
    """Fetch OSV.dev record for affected packages."""
    url = f"{_OSV_API}/{cve_id}"
    try:
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 404:
            return {"source": "osv", "found": False, "affected_packages": []}
        resp.raise_for_status()
        data = resp.json()

        packages = []
        for affected in data.get("affected", []):
            pkg = affected.get("package", {})
            ranges = affected.get("ranges", [])
            fixed_versions = []
            for r in ranges:
                for event in r.get("events", []):
                    if "fixed" in event:
                        fixed_versions.append(event["fixed"])

            packages.append(
                {
                    "ecosystem": pkg.get("ecosystem", ""),
                    "name": pkg.get("name", ""),
                    "fixed_versions": fixed_versions,
                }
            )

        aliases = data.get("aliases", [])
        return {
            "source": "osv",
            "found": True,
            "aliases": aliases,
            "affected_packages": packages,
        }
    except requests.exceptions.RequestException as exc:
        return {"source": "osv", "error": str(exc)}


def _fetch_vulncheck_kev(cve_id: str) -> dict[str, Any]:
    """Fetch VulnCheck KEV data (requires API key)."""
    api_key = os.environ.get("VULNCHECK_API_KEY", "").strip()
    if not api_key:
        return {"source": "vulncheck_kev", "available": False, "reason": "no_api_key"}

    url = f"{_VULNCHECK_KEV_API}?cve={cve_id}"
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    try:
        resp = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        entries = data.get("data", [])
        if not entries:
            return {"source": "vulncheck_kev", "available": True, "in_kev": False}

        entry = entries[0]
        return {
            "source": "vulncheck_kev",
            "available": True,
            "in_kev": True,
            "date_added": entry.get("date_added"),
            "exploit_maturity": entry.get("exploit_maturity"),
            "reporting_source": entry.get("reporting_source"),
        }
    except requests.exceptions.RequestException as exc:
        return {"source": "vulncheck_kev", "error": str(exc)}


# ---------------------------------------------------------------------------
# Enrichment orchestrator
# ---------------------------------------------------------------------------


def _compute_risk_level(
    cvss_score: float | None,
    epss_score: float | None,
    in_cisa_kev: bool,
    in_vulncheck_kev: bool,
) -> dict[str, Any]:
    """Compute a composite risk level from enrichment data.

    Returns a risk level (critical/high/medium/low/unknown) with rationale.
    """
    signals: list[str] = []
    score = 0.0

    if in_cisa_kev:
        score += 40
        signals.append("CISA KEV (active exploitation)")
    if in_vulncheck_kev:
        score += 20
        signals.append("VulnCheck KEV (multi-source exploitation signal)")

    if cvss_score is not None:
        if cvss_score >= 9.0:
            score += 25
            signals.append(f"CVSS {cvss_score} (critical)")
        elif cvss_score >= 7.0:
            score += 15
            signals.append(f"CVSS {cvss_score} (high)")
        elif cvss_score >= 4.0:
            score += 8
            signals.append(f"CVSS {cvss_score} (medium)")
        else:
            score += 3
            signals.append(f"CVSS {cvss_score} (low)")

    if epss_score is not None:
        if epss_score >= 0.5:
            score += 25
            signals.append(f"EPSS {epss_score:.4f} (very high exploitation probability)")
        elif epss_score >= 0.1:
            score += 15
            signals.append(f"EPSS {epss_score:.4f} (elevated exploitation probability)")
        elif epss_score >= 0.01:
            score += 5
            signals.append(f"EPSS {epss_score:.4f} (moderate)")
        else:
            score += 1
            signals.append(f"EPSS {epss_score:.4f} (low)")

    if score >= 60:
        level = "critical"
    elif score >= 35:
        level = "high"
    elif score >= 15:
        level = "medium"
    elif score > 0:
        level = "low"
    else:
        level = "unknown"

    return {"level": level, "score": round(score, 1), "signals": signals}


def enrich_cve(cve_id: str, *, include_vulncheck: bool = True) -> dict[str, Any]:
    """Fetch enrichment data from all sources in parallel and return unified snapshot.

    Args:
        cve_id: CVE identifier (e.g. 'CVE-2024-3094').
        include_vulncheck: Whether to query VulnCheck (requires API key).

    Returns:
        Structured dict with data from each source plus a composite risk assessment.
    """
    cve_id = cve_id.strip().upper()
    if not _CVE_RE.match(cve_id):
        return {"error": f"Invalid CVE ID format: {cve_id!r}. Expected CVE-YYYY-NNNNN."}

    # Dispatch all fetchers in parallel
    fetchers: dict[str, Any] = {
        "nvd": (_fetch_nvd, cve_id),
        "epss": (_fetch_epss, cve_id),
        "cisa_kev": (_fetch_cisa_kev, cve_id),
        "osv": (_fetch_osv, cve_id),
    }
    if include_vulncheck:
        fetchers["vulncheck_kev"] = (_fetch_vulncheck_kev, cve_id)

    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fn, arg): key for key, (fn, arg) in fetchers.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                results[key] = {"source": key, "error": str(exc)}

    # Extract key fields for risk computation
    nvd = results.get("nvd", {})
    epss = results.get("epss", {})
    cisa = results.get("cisa_kev", {})
    vc = results.get("vulncheck_kev", {})

    cvss_score = nvd.get("cvss_score")
    epss_score = epss.get("score")
    in_cisa_kev = cisa.get("in_kev", False)
    in_vulncheck_kev = vc.get("in_kev", False)

    risk = _compute_risk_level(cvss_score, epss_score, in_cisa_kev, in_vulncheck_kev)

    return {
        "cve_id": cve_id,
        "risk_assessment": risk,
        "nvd": results.get("nvd", {}),
        "epss": results.get("epss", {}),
        "cisa_kev": results.get("cisa_kev", {}),
        "osv": results.get("osv", {}),
        "vulncheck_kev": results.get("vulncheck_kev", {}),
    }


# ---------------------------------------------------------------------------
# Strands @tool interface
# ---------------------------------------------------------------------------


@tool
def cve_enrich(cve_id: str, include_vulncheck: str = "true") -> dict[str, Any]:
    """Enrich a CVE with data from NVD, EPSS, CISA KEV, OSV, and VulnCheck in parallel.

    Returns a unified snapshot with risk assessment, CVSS score, EPSS probability,
    KEV status, affected packages, and version-range data — all from a single call.

    Args:
        cve_id: The CVE identifier to enrich (e.g. 'CVE-2024-3094').
        include_vulncheck: Whether to include VulnCheck KEV lookup ('true'/'false').
            Requires VULNCHECK_API_KEY env var. Default: 'true'.

    Returns:
        Structured enrichment result with risk_assessment, nvd, epss, cisa_kev,
        osv, and vulncheck_kev sections.
    """
    vc = include_vulncheck.lower().strip() in ("true", "1", "yes")
    return enrich_cve(cve_id, include_vulncheck=vc)
