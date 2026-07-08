"""
Tool: generate_cve_report

Produces a structured Markdown narrative report for a CVE by aggregating data
from multiple sources in parallel:

  1. NVD CVE 2.0 API          — description, CVSS scores, CWE, references
  2. FIRST.org EPSS API        — current exploit-prediction score + percentile
  3. CISA KEV catalog          — active exploitation confirmation (cached)
  4. OSV.dev                   — affected packages + version ranges
  5. GitHub Advisory Database  — GHSA patches + affected ecosystems

The report contains the following sections:

    ## Summary
    ## Technical Details
    ## Affected Packages & Version Ranges
    ## Exploitation Status
    ## Recommendations
    ## References\n
All network calls degrade gracefully — a failed source never crashes the report.

CLI: ``manus-agent report CVE-2021-44228``
     ``manus-agent report CVE-2024-3094 --output json``
     ``manus-agent report CVE-2024-3094 --output markdown --save report.md``
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from strands import tool

__all__ = ["generate_cve_report"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"^CVE-\d{4}-\d+$", re.IGNORECASE)
_TIMEOUT = 15

_NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_EPSS_URL = "https://api.first.org/data/v1/epss"
_CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
_OSV_QUERY_URL = "https://api.osv.dev/v1/query"
_GHSA_URL = "https://api.github.com/advisories"

_CISA_CACHE_FILE = Path(__file__).parent / ".cisa_kev_cache.json"
_CISA_CACHE_TTL = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Data-fetch helpers — each returns a dict; never raises
# ---------------------------------------------------------------------------


def _fetch_nvd(cve_id: str) -> dict[str, Any]:
    """Fetch NVD CVE record.  Returns parsed fields or ``{"available": False}``."""
    try:
        r = requests.get(_NVD_URL, params={"cveId": cve_id.upper()}, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return {"available": False, "reason": "CVE not found in NVD"}
        cve = vulns[0]["cve"]

        # Description — prefer English
        description = ""
        for d in cve.get("descriptions", []):
            if d.get("lang") == "en":
                description = d.get("value", "")
                break

        # CVSS scores — try v3.1 then v3.0 then v2.0
        cvss_score: float | None = None
        cvss_severity: str = ""
        cvss_vector: str = ""
        cvss_version: str = ""
        for metric_key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            metrics = cve.get("metrics", {}).get(metric_key, [])
            if metrics:
                ms = metrics[0].get("cvssData", {})
                cvss_score = ms.get("baseScore")
                cvss_severity = ms.get("baseSeverity") or metrics[0].get("baseSeverity", "")
                cvss_vector = ms.get("vectorString", "")
                cvss_version = ms.get("version", "")
                break

        # CWE
        cwes: list[str] = []
        for weakness in cve.get("weaknesses", []):
            for wd in weakness.get("description", []):
                val = wd.get("value", "")
                if val and val not in cwes:
                    cwes.append(val)

        # References
        refs: list[str] = [ref.get("url", "") for ref in cve.get("references", []) if ref.get("url")]

        # Published / modified dates
        published = cve.get("published", "")
        last_modified = cve.get("lastModified", "")

        # Affected CPE products (top-5 for brevity)
        cpe_products: list[str] = []
        for config in cve.get("configurations", [])[:5]:
            for node in config.get("nodes", [])[:5]:
                for match in node.get("cpeMatch", [])[:5]:
                    criteria = match.get("criteria", "")
                    if criteria:
                        cpe_products.append(criteria)

        return {
            "available": True,
            "description": description,
            "cvss_score": cvss_score,
            "cvss_severity": cvss_severity,
            "cvss_vector": cvss_vector,
            "cvss_version": cvss_version,
            "cwes": cwes,
            "references": refs[:20],
            "published": published,
            "last_modified": last_modified,
            "cpe_products": cpe_products[:20],
            "vuln_status": cve.get("vulnStatus", ""),
        }
    except Exception as exc:
        logger.debug("NVD fetch failed for %s: %s", cve_id, exc)
        return {"available": False, "reason": str(exc)}


def _fetch_epss(cve_id: str) -> dict[str, Any]:
    """Fetch current EPSS score from FIRST.org."""
    try:
        r = requests.get(_EPSS_URL, params={"cve": cve_id.upper()}, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        entries = data.get("data", [])
        if not entries:
            return {"available": False}
        entry = entries[0]
        return {
            "available": True,
            "epss": float(entry.get("epss", 0)),
            "percentile": float(entry.get("percentile", 0)),
            "date": entry.get("date", ""),
        }
    except Exception as exc:
        logger.debug("EPSS fetch failed for %s: %s", cve_id, exc)
        return {"available": False, "reason": str(exc)}


def _fetch_cisa_kev(cve_id: str) -> dict[str, Any]:
    """Check CISA KEV catalog with 1-hour disk cache."""
    try:
        kev_data: dict[str, Any] = {}
        if _CISA_CACHE_FILE.exists():
            try:
                cached = json.loads(_CISA_CACHE_FILE.read_text())
                if time.time() - cached.get("timestamp", 0) < _CISA_CACHE_TTL:
                    kev_data = cached.get("data", {})
            except Exception:
                pass

        if not kev_data:
            r = requests.get(_CISA_KEV_URL, timeout=_TIMEOUT)
            r.raise_for_status()
            kev_data = r.json()
            try:
                _CISA_CACHE_FILE.write_text(json.dumps({"timestamp": time.time(), "data": kev_data}))
            except Exception:
                pass

        cve_upper = cve_id.upper()
        for vuln in kev_data.get("vulnerabilities", []):
            if vuln.get("cveID") == cve_upper:
                return {
                    "available": True,
                    "exploited": True,
                    "vendor_project": vuln.get("vendorProject", ""),
                    "product": vuln.get("product", ""),
                    "vulnerability_name": vuln.get("vulnerabilityName", ""),
                    "date_added": vuln.get("dateAdded", ""),
                    "short_description": vuln.get("shortDescription", ""),
                    "required_action": vuln.get("requiredAction", ""),
                    "due_date": vuln.get("dueDate", ""),
                    "notes": vuln.get("notes", ""),
                }
        return {"available": True, "exploited": False}
    except Exception as exc:
        logger.debug("CISA KEV fetch failed for %s: %s", cve_id, exc)
        return {"available": False, "reason": str(exc)}


def _fetch_osv(cve_id: str) -> dict[str, Any]:
    """Query OSV.dev for affected packages."""
    try:
        payload = {"query": {"cve_id": cve_id.upper()}}
        r = requests.post(_OSV_QUERY_URL, json=payload, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        vulns = data.get("vulns", [])
        if not vulns:
            return {"available": True, "packages": []}

        packages: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for v in vulns[:10]:  # cap at 10 OSV records
            for affected in v.get("affected", [])[:5]:
                pkg = affected.get("package", {})
                name = pkg.get("name", "")
                ecosystem = pkg.get("ecosystem", "")
                key = (name.lower(), ecosystem.lower())
                if key in seen:
                    continue
                seen.add(key)

                ranges: list[str] = []
                for r_block in affected.get("ranges", [])[:3]:
                    events = r_block.get("events", [])
                    introduced = ""
                    fixed = ""
                    for ev in events:
                        if "introduced" in ev:
                            introduced = ev["introduced"]
                        if "fixed" in ev:
                            fixed = ev["fixed"]
                    if introduced or fixed:
                        ranges.append(f">={introduced}" + (f", <{fixed}" if fixed else " (unfixed)"))

                packages.append(
                    {
                        "name": name,
                        "ecosystem": ecosystem,
                        "version_ranges": ranges,
                        "osv_id": v.get("id", ""),
                    }
                )

        return {"available": True, "packages": packages}
    except Exception as exc:
        logger.debug("OSV fetch failed for %s: %s", cve_id, exc)
        return {"available": False, "reason": str(exc)}


def _fetch_ghsa(cve_id: str) -> dict[str, Any]:
    """Fetch GitHub Security Advisory data for the CVE."""
    try:
        import os

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"token {token}"

        r = requests.get(_GHSA_URL, params={"cve_id": cve_id.upper()}, headers=headers, timeout=_TIMEOUT)
        r.raise_for_status()
        advisories = r.json()
        if not advisories:
            return {"available": True, "advisories": []}

        results: list[dict[str, Any]] = []
        for adv in advisories[:5]:
            vuln_pkgs: list[dict[str, Any]] = []
            for vuln in adv.get("vulnerabilities", [])[:10]:
                pkg = vuln.get("package", {})
                vuln_pkgs.append(
                    {
                        "name": pkg.get("name", ""),
                        "ecosystem": pkg.get("ecosystem", ""),
                        "vulnerable_version_range": vuln.get("vulnerable_version_range", ""),
                        "patched_versions": vuln.get("patched_versions", ""),
                        "first_patched": vuln.get("first_patched_version", {}).get("identifier", "") if isinstance(vuln.get("first_patched_version"), dict) else str(vuln.get("first_patched_version") or ""),
                    }
                )
            results.append(
                {
                    "ghsa_id": adv.get("ghsa_id", ""),
                    "summary": adv.get("summary", ""),
                    "severity": adv.get("severity", ""),
                    "published_at": adv.get("published_at", ""),
                    "updated_at": adv.get("updated_at", ""),
                    "html_url": adv.get("html_url", ""),
                    "packages": vuln_pkgs,
                    "cwe_ids": [c.get("cwe_id", "") for c in adv.get("cwes", [])],
                    "cvss_score": adv.get("cvss", {}).get("score"),
                    "cvss_vector": adv.get("cvss", {}).get("vector_string", ""),
                }
            )
        return {"available": True, "advisories": results}
    except Exception as exc:
        logger.debug("GHSA fetch failed for %s: %s", cve_id, exc)
        return {"available": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

_SEVERITY_EMOJI: dict[str, str] = {
    "CRITICAL": "🔴",
    "HIGH": "🟠",
    "MEDIUM": "🟡",
    "LOW": "🟢",
    "NONE": "⚪",
}


def _severity_badge(score: float | None, severity: str) -> str:
    if score is None:
        return "Unknown"
    emoji = _SEVERITY_EMOJI.get(severity.upper(), "⚪")
    return f"{emoji} {severity} ({score:.1f})"


def _epss_label(epss: float, percentile: float) -> str:
    pct = percentile * 100
    if epss >= 0.9:
        tier = "🚨 Very High"
    elif epss >= 0.5:
        tier = "🔴 High"
    elif epss >= 0.2:
        tier = "🟠 Medium"
    elif epss >= 0.05:
        tier = "🟡 Low"
    else:
        tier = "🟢 Very Low"
    return f"{tier} ({epss:.4f} / {pct:.1f}th percentile)"


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def _build_report(cve_id: str, nvd: dict, epss: dict, kev: dict, osv: dict, ghsa: dict) -> dict[str, Any]:
    """Assemble all fetched data into a structured report dict."""

    # --- Summary block ---
    published = nvd.get("published", "")[:10] if nvd.get("available") else ""
    severity = nvd.get("cvss_severity", "") if nvd.get("available") else ""
    score = nvd.get("cvss_score") if nvd.get("available") else None
    description = nvd.get("description", "") if nvd.get("available") else ""

    # Exploitation status
    exploited_in_wild = kev.get("exploited", False) if kev.get("available") else False
    kev_date = kev.get("date_added", "") if exploited_in_wild else ""
    kev_action = kev.get("required_action", "") if exploited_in_wild else ""
    kev_due = kev.get("due_date", "") if exploited_in_wild else ""

    # EPSS
    epss_score = epss.get("epss") if epss.get("available") else None
    epss_pct = epss.get("percentile") if epss.get("available") else None

    # Affected packages — merge OSV and GHSA
    pkg_rows: list[dict[str, Any]] = []
    if osv.get("available"):
        for p in osv.get("packages", []):
            pkg_rows.append(
                {
                    "source": "OSV",
                    "name": p.get("name", ""),
                    "ecosystem": p.get("ecosystem", ""),
                    "version_ranges": p.get("version_ranges", []),
                    "patched_versions": "",
                    "first_patched": "",
                }
            )
    if ghsa.get("available"):
        for adv in ghsa.get("advisories", []):
            for p in adv.get("packages", []):
                pkg_rows.append(
                    {
                        "source": f"GHSA ({adv.get('ghsa_id', '')})",
                        "name": p.get("name", ""),
                        "ecosystem": p.get("ecosystem", ""),
                        "version_ranges": [p.get("vulnerable_version_range", "")],
                        "patched_versions": p.get("patched_versions", ""),
                        "first_patched": p.get("first_patched", ""),
                    }
                )

    # References — NVD + GHSA
    refs: list[str] = []
    if nvd.get("available"):
        refs.extend(nvd.get("references", []))
    if ghsa.get("available"):
        for adv in ghsa.get("advisories", []):
            url = adv.get("html_url", "")
            if url and url not in refs:
                refs.append(url)

    # CWEs
    cwes: list[str] = nvd.get("cwes", []) if nvd.get("available") else []

    return {
        "cve_id": cve_id.upper(),
        "published": published,
        "cvss_severity": severity,
        "cvss_score": score,
        "cvss_vector": nvd.get("cvss_vector", "") if nvd.get("available") else "",
        "cvss_version": nvd.get("cvss_version", "") if nvd.get("available") else "",
        "description": description,
        "cwes": cwes,
        "epss_score": epss_score,
        "epss_percentile": epss_pct,
        "exploited_in_wild": exploited_in_wild,
        "kev_date_added": kev_date,
        "kev_required_action": kev_action,
        "kev_due_date": kev_due,
        "kev_vulnerability_name": kev.get("vulnerability_name", "") if exploited_in_wild else "",
        "affected_packages": pkg_rows,
        "references": refs[:20],
        "sources": {
            "nvd": nvd.get("available", False),
            "epss": epss.get("available", False),
            "cisa_kev": kev.get("available", False),
            "osv": osv.get("available", False),
            "ghsa": ghsa.get("available", False),
        },
    }


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def _render_markdown(report: dict[str, Any]) -> str:
    """Convert a report dict to a well-structured Markdown document."""
    lines: list[str] = []

    cve_id = report["cve_id"]
    sev = report.get("cvss_severity", "")
    score = report.get("cvss_score")
    badge = _severity_badge(score, sev)
    published = report.get("published", "N/A")

    lines.append(f"# CVE Report: {cve_id}")
    lines.append("")
    lines.append(f"> **Severity:** {badge}  ")
    lines.append(f"> **Published:** {published}  ")

    epss = report.get("epss_score")
    epss_pct = report.get("epss_percentile")
    if epss is not None:
        lines.append(f"> **EPSS:** {_epss_label(epss, epss_pct or 0.0)}  ")

    if report.get("exploited_in_wild"):
        lines.append("> **⚠️ ACTIVELY EXPLOITED** — Listed in CISA KEV Catalog  ")

    lines.append("")

    # --- Summary ---
    lines.append("## Summary")
    lines.append("")
    desc = report.get("description", "")
    if desc:
        lines.append(desc)
    else:
        lines.append("_No description available from NVD._")
    lines.append("")

    # --- Technical Details ---
    lines.append("## Technical Details")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| CVE ID | `{cve_id}` |")
    lines.append(f"| CVSS Score | {score if score is not None else 'N/A'} |")
    lines.append(f"| CVSS Severity | {sev or 'N/A'} |")
    cvss_vec = report.get("cvss_vector", "")
    cvss_ver = report.get("cvss_version", "")
    if cvss_vec:
        lines.append(f"| CVSS Vector | `{cvss_vec}` (v{cvss_ver}) |")
    cwes = report.get("cwes", [])
    if cwes:
        lines.append(f"| CWE(s) | {', '.join(cwes)} |")
    if epss is not None:
        lines.append(f"| EPSS Score | {epss:.4f} ({(epss_pct or 0.0) * 100:.1f}th percentile) |")
    lines.append(f"| Published | {published} |")
    lines.append("")

    # --- Affected Packages ---
    lines.append("## Affected Packages & Version Ranges")
    lines.append("")
    pkgs = report.get("affected_packages", [])
    if pkgs:
        lines.append("| Package | Ecosystem | Vulnerable Versions | Fixed In | Source |")
        lines.append("|---------|-----------|---------------------|----------|--------|")
        for p in pkgs:
            name = p.get("name", "")
            eco = p.get("ecosystem", "")
            vranges = "; ".join(r for r in p.get("version_ranges", []) if r) or "N/A"
            fixed = p.get("first_patched", "") or p.get("patched_versions", "") or "N/A"
            src = p.get("source", "")
            lines.append(f"| `{name}` | {eco} | {vranges} | {fixed} | {src} |")
    else:
        lines.append("_No affected package records found in OSV or GitHub Advisory Database._")
    lines.append("")

    # --- Exploitation Status ---
    lines.append("## Exploitation Status")
    lines.append("")
    if report.get("exploited_in_wild"):
        vuln_name = report.get("kev_vulnerability_name", "")
        kev_date = report.get("kev_date_added", "")
        kev_action = report.get("kev_required_action", "")
        kev_due = report.get("kev_due_date", "")
        lines.append("### ⚠️ Actively Exploited in the Wild")
        lines.append("")
        lines.append("This CVE is listed in the **CISA Known Exploited Vulnerabilities (KEV) Catalog**,")
        lines.append("confirming evidence of active exploitation in the wild.")
        lines.append("")
        if vuln_name:
            lines.append(f"- **Vulnerability Name:** {vuln_name}")
        if kev_date:
            lines.append(f"- **Date Added to KEV:** {kev_date}")
        if kev_action:
            lines.append(f"- **Required Action:** {kev_action}")
        if kev_due:
            lines.append(f"- **Federal Remediation Due Date:** {kev_due}")
    else:
        lines.append("### No Known Active Exploitation")
        lines.append("")
        lines.append("This CVE is **not** currently listed in the CISA KEV Catalog.")
        lines.append("Monitor EPSS trends and threat intelligence feeds for emerging exploitation activity.")
    lines.append("")

    if epss is not None:
        lines.append("### EPSS Exploitation Probability")
        lines.append("")
        lines.append(
            f"The current EPSS score is **{epss:.4f}** "
            f"({(epss_pct or 0.0) * 100:.1f}th percentile), "
            f"indicating that {epss * 100:.1f}% of similarly-scored CVEs are exploited "
            f"within the next 30 days."
        )
        lines.append("")

    # --- Recommendations ---
    lines.append("## Recommendations")
    lines.append("")
    has_patches = any(p.get("first_patched") or p.get("patched_versions") for p in pkgs)
    urgency = "immediately" if report.get("exploited_in_wild") else "as soon as possible"

    if score is not None and score >= 9.0:
        priority = "**P0 — Critical**: Patch " + urgency + "."
    elif score is not None and score >= 7.0:
        priority = "**P1 — High**: Patch " + urgency + "."
    elif score is not None and score >= 4.0:
        priority = "**P2 — Medium**: Schedule patching in the next maintenance window."
    else:
        priority = "**P3 — Low**: Monitor and patch during normal maintenance cycles."

    lines.append(f"1. **Priority**: {priority}")
    lines.append("")

    if has_patches:
        lines.append("2. **Upgrade** affected packages to the patched versions listed in the table above.")
    else:
        lines.append("2. **Check vendor advisories** — patched versions may not yet be catalogued here.")

    lines.append("3. **Verify** your environment's exposure using `manus-agent blast-radius " + cve_id + "`.")

    if report.get("exploited_in_wild"):
        lines.append("4. **Treat as an active incident** — follow CISA KEV remediation guidance.")
        lines.append("5. **Monitor threat intelligence feeds** for Indicators of Compromise (IoCs).")
    else:
        lines.append("4. **Monitor** EPSS trends using `manus-agent epss-trend " + cve_id + "`.")
        lines.append(
            "5. **Check PoC availability** using `manus-agent poc-search " + cve_id + "` to gauge exploitation risk."
        )
    lines.append("")

    # --- References ---
    lines.append("## References")
    lines.append("")
    refs = report.get("references", [])
    if refs:
        for ref in refs[:15]:
            lines.append(f"- <{ref}>")
    else:
        lines.append(f"- <https://nvd.nist.gov/vuln/detail/{cve_id}>")
    lines.append("")

    # --- Data Sources ---
    lines.append("## Data Sources")
    lines.append("")
    sources = report.get("sources", {})
    source_labels = {
        "nvd": "NVD CVE 2.0 API",
        "epss": "FIRST.org EPSS API",
        "cisa_kev": "CISA KEV Catalog",
        "osv": "OSV.dev",
        "ghsa": "GitHub Advisory Database",
    }
    for key, label in source_labels.items():
        status = "✅ Available" if sources.get(key) else "❌ Unavailable"
        lines.append(f"- **{label}**: {status}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strands @tool entry point
# ---------------------------------------------------------------------------


@tool
def generate_cve_report(cve_id: str) -> dict[str, Any]:
    """
    Generate a comprehensive, structured Markdown narrative report for a CVE.

    Aggregates data from NVD, EPSS, CISA KEV, OSV, and GitHub Advisory Database
    in parallel and produces a complete analyst report with sections:
    Summary, Technical Details, Affected Packages, Exploitation Status,
    Recommendations, and References.

    All sources degrade gracefully — a failed source never crashes the report.
    The report can be rendered as Markdown or consumed as structured JSON.

    Args:
        cve_id: CVE identifier to report on (e.g. "CVE-2021-44228").

    Returns:
        A dict containing:
        - ``markdown``: full Markdown text of the report
        - ``report``: structured data dict (cvss, epss, packages, refs, …)
        - ``cve_id``: normalised CVE ID
    """
    cve_id = cve_id.strip().upper()
    if not _CVE_RE.match(cve_id):
        return {
            "error": f"Invalid CVE ID {cve_id!r}. Expected format: CVE-YYYY-NNNNN",
            "cve_id": cve_id,
        }

    # Fetch all sources concurrently
    fetch_tasks = {
        "nvd": lambda: _fetch_nvd(cve_id),
        "epss": lambda: _fetch_epss(cve_id),
        "kev": lambda: _fetch_cisa_kev(cve_id),
        "osv": lambda: _fetch_osv(cve_id),
        "ghsa": lambda: _fetch_ghsa(cve_id),
    }

    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fn): key for key, fn in fetch_tasks.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                logger.warning("Source %s failed unexpectedly: %s", key, exc)
                results[key] = {"available": False, "reason": str(exc)}

    report = _build_report(
        cve_id,
        nvd=results.get("nvd", {"available": False}),
        epss=results.get("epss", {"available": False}),
        kev=results.get("kev", {"available": False}),
        osv=results.get("osv", {"available": False}),
        ghsa=results.get("ghsa", {"available": False}),
    )

    markdown = _render_markdown(report)

    return {
        "cve_id": cve_id,
        "report": report,
        "markdown": markdown,
    }
