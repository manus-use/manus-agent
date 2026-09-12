"""
Tool for generating structured vulnerability intelligence reports.

Given one or more CVE identifiers, this tool aggregates data from multiple
public sources and produces a comprehensive Markdown or JSON report suitable
for security teams, management briefings, or compliance documentation.

Data sources queried per CVE:
- **NVD** — CVSS score, severity, CWE, affected products, references
- **EPSS** — exploitation probability score
- **CISA KEV** — active exploitation status and remediation deadlines
- **VulnCheck KEV** — broader exploitation intel (100+ sources)
- **OSV.dev** — affected packages and fix versions

The report includes:
- Executive summary with risk-ranked CVE table
- Per-CVE detail sections (severity, exploitation status, affected software,
  remediation guidance)
- Aggregate statistics (severity distribution, KEV count, mean EPSS)
- Generated timestamp and data-source attribution

All HTTP calls use retry/back-off for resilience.
"""

from __future__ import annotations

import json as _json
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# Retry / back-off
# ---------------------------------------------------------------------------
_MAX_RETRIES: int = int(os.environ.get("REPORT_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY: float = float(os.environ.get("REPORT_RETRY_BASE_DELAY", "1.0"))
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_TIMEOUT: int = 20

TOOL_SPEC = {
    "name": "generate_vuln_report",
    "description": (
        "Generates a structured vulnerability intelligence report for one or more CVE IDs. "
        "Aggregates NVD, EPSS, CISA KEV, VulnCheck KEV, and OSV.dev data into a comprehensive "
        "Markdown or JSON report with executive summary, per-CVE details, severity distribution, "
        "and remediation guidance. Use for security briefings, compliance documentation, or "
        "triage summaries."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of CVE identifiers to include in the report (e.g., ['CVE-2024-3094', 'CVE-2021-44228'])."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": ("Optional report title. Defaults to 'Vulnerability Intelligence Report'."),
                },
                "output_format": {
                    "type": "string",
                    "enum": ["markdown", "json"],
                    "description": "Output format: 'markdown' (default) or 'json'.",
                },
            },
            "required": ["cve_ids"],
        }
    },
}


# ---------------------------------------------------------------------------
# HTTP helper with retry
# ---------------------------------------------------------------------------


def _get_with_retry(
    url: str,
    *,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = _TIMEOUT,
) -> requests.Response:
    """GET *url* with exponential back-off on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code not in _RETRYABLE_STATUSES:
                return resp
            last_exc = requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
        except requests.RequestException as exc:
            last_exc = exc
        if attempt < _MAX_RETRIES - 1:
            time.sleep(_RETRY_BASE_DELAY * (2**attempt))
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Per-source fetch helpers
# ---------------------------------------------------------------------------


def _fetch_nvd(cve_id: str) -> dict[str, Any]:
    """Fetch NVD record for a single CVE."""
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id.upper()}"
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY", "")
    if api_key:
        headers["apiKey"] = api_key
    try:
        resp = _get_with_retry(url, headers=headers if headers else None)
        resp.raise_for_status()
        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return {"error": f"No NVD record for {cve_id}"}
        return vulns[0].get("cve", {})
    except Exception as exc:
        return {"error": f"NVD fetch failed: {exc}"}


def _fetch_epss(cve_id: str) -> dict[str, Any]:
    """Fetch current EPSS score for a single CVE."""
    try:
        resp = _get_with_retry(
            "https://api.first.org/data/v1/epss",
            params={"cve": cve_id.upper()},
        )
        resp.raise_for_status()
        entries = resp.json().get("data", [])
        if not entries:
            return {"error": f"No EPSS data for {cve_id}"}
        return entries[0]
    except Exception as exc:
        return {"error": f"EPSS fetch failed: {exc}"}


def _fetch_kev(cve_id: str, catalog: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Check CISA KEV catalog for a CVE.

    If *catalog* is provided (pre-fetched), look up directly; otherwise
    fetch the full catalog.
    """
    cid = cve_id.upper()

    if catalog is None:
        try:
            resp = _get_with_retry(
                "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
            )
            resp.raise_for_status()
            catalog = resp.json().get("vulnerabilities", [])
        except Exception as exc:
            return {"in_kev": False, "error": f"KEV fetch failed: {exc}"}

    for entry in catalog:
        if entry.get("cveID", "").upper() == cid:
            return {
                "in_kev": True,
                "date_added": entry.get("dateAdded"),
                "due_date": entry.get("dueDate"),
                "vendor_project": entry.get("vendorProject"),
                "product": entry.get("product"),
                "required_action": entry.get("requiredAction"),
            }
    return {"in_kev": False}


def _fetch_vulncheck_kev(cve_id: str) -> dict[str, Any]:
    """Fetch VulnCheck KEV data if API key is available."""
    api_key = os.environ.get("VULNCHECK_API_KEY", "")
    if not api_key:
        return {"available": False}
    try:
        resp = _get_with_retry(
            "https://api.vulncheck.com/v3/index/vulncheck-kev",
            params={"cve": cve_id.upper()},
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return {"available": True, "in_kev": False}
        entry = data[0]
        return {
            "available": True,
            "in_kev": True,
            "ransomware_use": entry.get("knownRansomwareCampaignUse", "Unknown") == "Known",
            "date_added": entry.get("dateAdded"),
            "due_date": entry.get("dueDate"),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _fetch_osv(cve_id: str) -> dict[str, Any]:
    """Fetch OSV.dev affected-package data."""
    try:
        resp = requests.post(
            "https://api.osv.dev/v1/query",
            json={"package": {}, "version": "", "aliases": [cve_id.upper()]},
            timeout=_TIMEOUT,
        )
        # OSV uses POST /v1/query — a 200 with empty vulns means no match
        if resp.status_code != 200:
            return {"error": f"OSV HTTP {resp.status_code}"}
        data = resp.json()
        vulns = data.get("vulns", [])
        if not vulns:
            return {"affected_packages": []}

        packages: list[dict[str, Any]] = []
        for vuln in vulns:
            for affected in vuln.get("affected", []):
                pkg = affected.get("package", {})
                ranges = affected.get("ranges", [])
                fixed_versions: list[str] = []
                for r in ranges:
                    for ev in r.get("events", []):
                        if "fixed" in ev:
                            fixed_versions.append(ev["fixed"])
                packages.append(
                    {
                        "ecosystem": pkg.get("ecosystem", "unknown"),
                        "name": pkg.get("name", "unknown"),
                        "fixed_versions": fixed_versions,
                    }
                )
        return {"affected_packages": packages}
    except Exception as exc:
        return {"error": f"OSV fetch failed: {exc}"}


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------


def _extract_cvss(nvd: dict[str, Any]) -> dict[str, Any]:
    """Extract CVSS score, severity, and vector from NVD record."""
    metrics = nvd.get("metrics", {})

    # Try CVSS 3.1 first, then 3.0, then 2.0
    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key, [])
        if entries:
            data = entries[0].get("cvssData", {})
            return {
                "version": data.get("version", "3.x"),
                "score": data.get("baseScore", 0.0),
                "severity": data.get("baseSeverity", "UNKNOWN"),
                "vector": data.get("vectorString", ""),
            }

    entries = metrics.get("cvssMetricV2", [])
    if entries:
        data = entries[0].get("cvssData", {})
        return {
            "version": data.get("version", "2.0"),
            "score": data.get("baseScore", 0.0),
            "severity": entries[0].get("baseSeverity", "UNKNOWN"),
            "vector": data.get("vectorString", ""),
        }

    return {"version": "N/A", "score": 0.0, "severity": "UNKNOWN", "vector": ""}


def _extract_cwe(nvd: dict[str, Any]) -> list[str]:
    """Extract CWE IDs from NVD record."""
    cwes: list[str] = []
    for weakness in nvd.get("weaknesses", []):
        for desc in weakness.get("description", []):
            val = desc.get("value", "")
            if val.startswith("CWE-"):
                cwes.append(val)
    return cwes


def _extract_description(nvd: dict[str, Any]) -> str:
    """Extract English description from NVD record."""
    for desc in nvd.get("descriptions", []):
        if desc.get("lang") == "en":
            return desc.get("value", "")
    descs = nvd.get("descriptions", [])
    if descs:
        return descs[0].get("value", "")
    return ""


def _extract_references(nvd: dict[str, Any]) -> list[dict[str, str]]:
    """Extract reference URLs from NVD record."""
    refs: list[dict[str, str]] = []
    for ref in nvd.get("references", []):
        refs.append(
            {
                "url": ref.get("url", ""),
                "source": ref.get("source", ""),
            }
        )
    return refs[:10]  # Cap at 10 to keep report manageable


def _extract_published(nvd: dict[str, Any]) -> str:
    """Extract published date from NVD record."""
    return nvd.get("published", "")


def _extract_affected_products(nvd: dict[str, Any]) -> list[str]:
    """Extract affected vendor:product pairs from NVD configurations."""
    products: list[str] = []
    for config in nvd.get("configurations", []):
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                if match.get("vulnerable"):
                    criteria = match.get("criteria", "")
                    parts = criteria.split(":")
                    if len(parts) >= 5:
                        vendor = parts[3]
                        product = parts[4]
                        products.append(f"{vendor}:{product}")
    return list(dict.fromkeys(products))[:10]  # Deduplicate, cap at 10


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4, "N/A": 5}
_SEVERITY_EMOJI = {
    "CRITICAL": "🔴",
    "HIGH": "🟠",
    "MEDIUM": "🟡",
    "LOW": "🟢",
    "UNKNOWN": "⚪",
    "N/A": "⚪",
}


def _severity_sort_key(finding: dict[str, Any]) -> tuple[int, float, float]:
    """Sort key: KEV first, then severity, then EPSS descending."""
    sev = finding.get("cvss", {}).get("severity", "UNKNOWN")
    epss = finding.get("epss_score", 0.0)
    kev_rank = 0 if finding.get("in_cisa_kev") else 1
    return (kev_rank, _SEVERITY_ORDER.get(sev, 5), -epss)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _gather_cve_data(
    cve_id: str,
    kev_catalog: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Gather all intelligence for a single CVE."""
    cve_id = cve_id.strip().upper()

    nvd = _fetch_nvd(cve_id)
    epss = _fetch_epss(cve_id)
    kev = _fetch_kev(cve_id, catalog=kev_catalog)
    vc_kev = _fetch_vulncheck_kev(cve_id)
    osv = _fetch_osv(cve_id)

    cvss = (
        _extract_cvss(nvd)
        if "error" not in nvd
        else {
            "version": "N/A",
            "score": 0.0,
            "severity": "UNKNOWN",
            "vector": "",
        }
    )

    return {
        "cve_id": cve_id,
        "description": _extract_description(nvd) if "error" not in nvd else "",
        "published": _extract_published(nvd) if "error" not in nvd else "",
        "cvss": cvss,
        "cwes": _extract_cwe(nvd) if "error" not in nvd else [],
        "affected_products": _extract_affected_products(nvd) if "error" not in nvd else [],
        "references": _extract_references(nvd) if "error" not in nvd else [],
        "epss_score": float(epss.get("epss", 0.0)) if "error" not in epss else 0.0,
        "epss_percentile": float(epss.get("percentile", 0.0)) if "error" not in epss else 0.0,
        "in_cisa_kev": kev.get("in_kev", False),
        "kev_details": kev if kev.get("in_kev") else None,
        "vulncheck_kev": vc_kev if vc_kev.get("available") and vc_kev.get("in_kev") else None,
        "osv_packages": osv.get("affected_packages", []),
        "data_errors": [
            src for src, data in [("nvd", nvd), ("epss", epss), ("kev", kev), ("osv", osv)] if "error" in data
        ],
    }


def _prefetch_kev_catalog() -> list[dict[str, Any]] | None:
    """Fetch the full KEV catalog once to avoid N fetches."""
    try:
        resp = _get_with_retry(
            "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
        )
        resp.raise_for_status()
        return resp.json().get("vulnerabilities", [])
    except Exception:
        return None


def _compute_statistics(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistics across all findings."""
    total = len(findings)
    if total == 0:
        return {
            "total_cves": 0,
            "severity_distribution": {},
            "kev_count": 0,
            "mean_epss": 0.0,
            "max_cvss": 0.0,
        }

    severity_dist: dict[str, int] = {}
    kev_count = 0
    epss_sum = 0.0
    max_cvss = 0.0

    for f in findings:
        sev = f.get("cvss", {}).get("severity", "UNKNOWN")
        severity_dist[sev] = severity_dist.get(sev, 0) + 1
        if f.get("in_cisa_kev"):
            kev_count += 1
        epss_sum += f.get("epss_score", 0.0)
        score = f.get("cvss", {}).get("score", 0.0)
        if score > max_cvss:
            max_cvss = score

    return {
        "total_cves": total,
        "severity_distribution": severity_dist,
        "kev_count": kev_count,
        "mean_epss": round(epss_sum / total, 5) if total else 0.0,
        "max_cvss": max_cvss,
    }


def generate_report(
    cve_ids: list[str],
    *,
    title: str = "Vulnerability Intelligence Report",
    output_format: str = "markdown",
) -> dict[str, Any]:
    """Generate a vulnerability report for the given CVE IDs.

    Returns a dict with ``report`` (formatted string) and ``data`` (structured
    findings + statistics).
    """
    if not cve_ids:
        return {
            "error": "No CVE IDs provided",
            "report": "",
            "data": {"findings": [], "statistics": {}},
        }

    # Deduplicate and normalise
    seen: set[str] = set()
    unique_ids: list[str] = []
    for cid in cve_ids:
        norm = cid.strip().upper()
        if norm and norm not in seen:
            seen.add(norm)
            unique_ids.append(norm)

    # Pre-fetch KEV catalog once
    kev_catalog = _prefetch_kev_catalog()

    # Gather data for each CVE
    findings: list[dict[str, Any]] = []
    for cve_id in unique_ids:
        finding = _gather_cve_data(cve_id, kev_catalog=kev_catalog)
        findings.append(finding)

    # Sort by risk: KEV first, then severity, then EPSS
    findings.sort(key=_severity_sort_key)

    stats = _compute_statistics(findings)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    result: dict[str, Any] = {
        "title": title,
        "generated_at": generated_at,
        "data": {
            "findings": findings,
            "statistics": stats,
        },
    }

    if output_format == "json":
        result["report"] = _json.dumps(result["data"], indent=2)
    else:
        result["report"] = _render_markdown(title, generated_at, findings, stats)

    return result


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def _render_markdown(
    title: str,
    generated_at: str,
    findings: list[dict[str, Any]],
    stats: dict[str, Any],
) -> str:
    """Render the full Markdown report."""
    lines: list[str] = []

    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Generated:** {generated_at}")
    lines.append(f"**CVEs analysed:** {stats['total_cves']}")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    lines.append("")

    kev_count = stats["kev_count"]
    if kev_count:
        lines.append(
            f"🚨 **{kev_count} CVE(s) are in the CISA Known Exploited Vulnerabilities catalog** "
            f"and require immediate attention."
        )
        lines.append("")

    # Severity breakdown
    dist = stats.get("severity_distribution", {})
    if dist:
        parts: list[str] = []
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            count = dist.get(sev, 0)
            if count:
                emoji = _SEVERITY_EMOJI.get(sev, "⚪")
                parts.append(f"{emoji} {count} {sev}")
        if parts:
            lines.append("**Severity breakdown:** " + " · ".join(parts))
            lines.append("")

    if stats["mean_epss"] > 0:
        lines.append(f"**Mean EPSS score:** {stats['mean_epss']:.4f} (exploitation probability within 30 days)")
        lines.append("")

    # Summary table
    lines.append("### Risk-Ranked CVE Overview")
    lines.append("")
    lines.append("| CVE ID | Severity | CVSS | EPSS | KEV | Description |")
    lines.append("|--------|----------|------|------|-----|-------------|")
    for f in findings:
        sev = f.get("cvss", {}).get("severity", "UNKNOWN")
        emoji = _SEVERITY_EMOJI.get(sev, "⚪")
        score = f.get("cvss", {}).get("score", 0.0)
        epss = f.get("epss_score", 0.0)
        kev_flag = "✅ Yes" if f.get("in_cisa_kev") else "No"
        desc = f.get("description", "")[:80]
        if len(f.get("description", "")) > 80:
            desc += "…"
        lines.append(f"| {f['cve_id']} | {emoji} {sev} | {score} | {epss:.4f} | {kev_flag} | {desc} |")
    lines.append("")

    # Per-CVE detail sections
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Findings")
    lines.append("")
    for f in findings:
        lines.extend(_render_finding(f))
        lines.append("")

    # Statistics footer
    lines.append("---")
    lines.append("")
    lines.append("## Report Metadata")
    lines.append("")
    lines.append(f"- **Generated:** {generated_at}")
    lines.append(f"- **Total CVEs:** {stats['total_cves']}")
    lines.append(f"- **Highest CVSS:** {stats['max_cvss']}")
    lines.append(f"- **Mean EPSS:** {stats['mean_epss']:.4f}")
    lines.append(f"- **In CISA KEV:** {kev_count}")
    lines.append("- **Data sources:** NVD, EPSS (FIRST.org), CISA KEV, VulnCheck KEV, OSV.dev")
    lines.append("")

    return "\n".join(lines)


def _render_finding(f: dict[str, Any]) -> list[str]:
    """Render a single CVE finding as Markdown."""
    lines: list[str] = []
    sev = f.get("cvss", {}).get("severity", "UNKNOWN")
    emoji = _SEVERITY_EMOJI.get(sev, "⚪")

    lines.append(f"### {emoji} {f['cve_id']}")
    lines.append("")

    if f.get("description"):
        lines.append(f"> {f['description']}")
        lines.append("")

    # CVSS
    cvss = f.get("cvss", {})
    lines.append(f"**CVSS:** {cvss.get('score', 0.0)} ({sev}) — v{cvss.get('version', 'N/A')}")
    if cvss.get("vector"):
        lines.append(f"**Vector:** `{cvss['vector']}`")
    lines.append("")

    # EPSS
    epss = f.get("epss_score", 0.0)
    pctl = f.get("epss_percentile", 0.0)
    lines.append(
        f"**EPSS:** {epss:.4f} (percentile: {pctl:.2f}) — "
        f"{'high' if epss >= 0.1 else 'moderate' if epss >= 0.01 else 'low'} exploitation probability"
    )
    lines.append("")

    # Published
    if f.get("published"):
        lines.append(f"**Published:** {f['published']}")
        lines.append("")

    # CWE
    if f.get("cwes"):
        lines.append(f"**Weaknesses:** {', '.join(f['cwes'])}")
        lines.append("")

    # KEV status
    if f.get("in_cisa_kev"):
        lines.append("🚨 **CISA KEV: ACTIVELY EXPLOITED**")
        kev = f.get("kev_details", {})
        if kev:
            if kev.get("date_added"):
                lines.append(f"- Added: {kev['date_added']}")
            if kev.get("due_date"):
                lines.append(f"- Remediation due: {kev['due_date']}")
            if kev.get("required_action"):
                lines.append(f"- Required action: {kev['required_action']}")
        lines.append("")

    # VulnCheck KEV
    vc = f.get("vulncheck_kev")
    if vc:
        ransomware = vc.get("ransomware_use", False)
        if ransomware:
            lines.append("⚠️ **VulnCheck: RANSOMWARE ASSOCIATED**")
        else:
            lines.append("🔍 **VulnCheck KEV: confirmed exploitation**")
        lines.append("")

    # Affected products (NVD CPE)
    products = f.get("affected_products", [])
    if products:
        lines.append("**Affected products (NVD):**")
        for p in products[:5]:
            lines.append(f"- `{p}`")
        if len(products) > 5:
            lines.append(f"- … and {len(products) - 5} more")
        lines.append("")

    # OSV packages
    osv_pkgs = f.get("osv_packages", [])
    if osv_pkgs:
        lines.append("**Affected packages (OSV.dev):**")
        for pkg in osv_pkgs[:5]:
            eco = pkg.get("ecosystem", "?")
            name = pkg.get("name", "?")
            fixed = pkg.get("fixed_versions", [])
            fix_str = f" → fix in {', '.join(fixed)}" if fixed else ""
            lines.append(f"- [{eco}] `{name}`{fix_str}")
        if len(osv_pkgs) > 5:
            lines.append(f"- … and {len(osv_pkgs) - 5} more")
        lines.append("")

    # References
    refs = f.get("references", [])
    if refs:
        lines.append("**Key references:**")
        for ref in refs[:5]:
            url = ref.get("url", "")
            lines.append(f"- {url}")
        if len(refs) > 5:
            lines.append(f"- … and {len(refs) - 5} more")
        lines.append("")

    # Data warnings
    if f.get("data_errors"):
        lines.append(f"⚠️ *Data unavailable from: {', '.join(f['data_errors'])}*")
        lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Strands tool handler
# ---------------------------------------------------------------------------


def handler(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands tool handler for generate_vuln_report."""
    inp = tool["input"]
    cve_ids = inp.get("cve_ids", [])
    title = inp.get("title", "Vulnerability Intelligence Report")
    output_format = inp.get("output_format", "markdown")

    if not cve_ids:
        return {
            "toolUseId": tool["toolUseId"],
            "status": "error",
            "content": [{"text": "Error: cve_ids list is required and must not be empty."}],
        }

    # Validate CVE ID format
    import re

    invalid = [c for c in cve_ids if not re.match(r"^CVE-\d{4}-\d+$", c.strip(), re.IGNORECASE)]
    if invalid:
        return {
            "toolUseId": tool["toolUseId"],
            "status": "error",
            "content": [{"text": f"Invalid CVE ID(s): {', '.join(invalid)}. Expected format: CVE-YYYY-NNNNN"}],
        }

    result = generate_report(cve_ids, title=title, output_format=output_format)

    output_text = result.get("report", "")
    log_tool_output_size("generate_vuln_report", output_text)

    return {
        "toolUseId": tool["toolUseId"],
        "status": "success",
        "content": [{"text": output_text}],
    }
