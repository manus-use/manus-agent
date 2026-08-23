"""
Batch CVE triage tool — accepts multiple CVE IDs and produces a prioritized
ranking based on CVSS severity, EPSS exploitation probability, and CISA KEV
membership.

Designed for ad-hoc triage of vulnerability lists (from scanners, advisories,
or manual collections). Unlike the watchlist tool (persistent tracking) or
compare tool (exactly 2 CVEs), this handles arbitrary-length lists statelessly
and outputs a ranked triage table.

Data sources:
- NVD (CVSS v3.1/v3.0/v2.0 base score)
- FIRST.org EPSS API (batch endpoint, up to 100 CVEs per call)
- CISA KEV catalogue (known exploited vulnerabilities)
- VulnCheck KEV (optional, when VULNCHECK_API_KEY is set)
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)

_NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_EPSS_API_URL = "https://api.first.org/data/v1/epss"
_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
_VULNCHECK_KEV_URL = "https://api.vulncheck.com/v3/index/vulncheck-kev"

# Retry settings
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.5

# Batch sizes
_EPSS_BATCH_SIZE = 100  # FIRST.org supports up to 100 CVEs per request
_NVD_DELAY = 0.7  # NVD rate limit: ~10 requests per minute without API key

# Triage scoring weights
_WEIGHT_CVSS = 0.30
_WEIGHT_EPSS = 0.40
_WEIGHT_KEV = 0.30

# ---------------------------------------------------------------------------
# TOOL_SPEC (Strands agent interface)
# ---------------------------------------------------------------------------

TOOL_SPEC = {
    "name": "triage_cves",
    "description": (
        "Batch CVE triage: accepts a list of CVE IDs, fetches NVD CVSS + EPSS + "
        "CISA KEV status for each, computes a composite triage score (0-100), and "
        "returns the list ranked by urgency. Use this when you have multiple CVEs "
        "to prioritize (e.g., from a scanner report, advisory list, or SBOM scan). "
        "Supports up to 50 CVEs per call. Returns both structured data and a "
        "human-readable triage table."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of CVE identifiers to triage (e.g., ['CVE-2024-3094', 'CVE-2021-44228']). Maximum 50."
                    ),
                },
                "output_format": {
                    "type": "string",
                    "enum": ["text", "json"],
                    "description": "Output format: 'text' for a ranked table, 'json' for structured data.",
                    "default": "text",
                },
            },
            "required": ["cve_ids"],
        }
    },
}


# ---------------------------------------------------------------------------
# HTTP helpers with retry/back-off
# ---------------------------------------------------------------------------


def _get_with_retry(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 20,
) -> requests.Response:
    """HTTP GET with exponential back-off on 429/5xx."""
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < _MAX_RETRIES - 1:
                    wait = _BACKOFF_BASE ** (attempt + 1)
                    time.sleep(wait)
                    continue
            return resp
        except requests.RequestException:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE ** (attempt + 1))
                continue
            raise
    # Should not reach here, but satisfy type checker
    return requests.get(url, params=params, headers=headers, timeout=timeout)  # pragma: no cover


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------


def _fetch_nvd_cvss(cve_id: str) -> dict[str, Any]:
    """Fetch CVSS score and vector from NVD for a single CVE."""
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY")
    if api_key:
        headers["apiKey"] = api_key

    try:
        resp = _get_with_retry(
            _NVD_API_URL,
            params={"cveId": cve_id.upper()},
            headers=headers if headers else None,
            timeout=20,
        )
        if resp.status_code != 200:
            return {"cvss_score": None, "cvss_vector": None, "severity": None}

        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return {"cvss_score": None, "cvss_vector": None, "severity": None}

        cve_data = vulns[0].get("cve", {})
        metrics = cve_data.get("metrics", {})

        # Try CVSS v3.1, then v3.0, then v2.0
        for key in ("cvssMetricV31", "cvssMetricV30"):
            entries = metrics.get(key, [])
            if entries:
                cvss_data = entries[0].get("cvssData", {})
                return {
                    "cvss_score": cvss_data.get("baseScore"),
                    "cvss_vector": cvss_data.get("vectorString"),
                    "severity": cvss_data.get("baseSeverity", "").upper(),
                }

        # Fallback to v2
        v2_entries = metrics.get("cvssMetricV2", [])
        if v2_entries:
            cvss_data = v2_entries[0].get("cvssData", {})
            score = cvss_data.get("baseScore")
            # Map v2 score to severity label
            severity = "LOW"
            if score and score >= 7.0:
                severity = "HIGH"
            elif score and score >= 4.0:
                severity = "MEDIUM"
            return {
                "cvss_score": score,
                "cvss_vector": cvss_data.get("vectorString"),
                "severity": severity,
            }

        return {"cvss_score": None, "cvss_vector": None, "severity": None}

    except Exception:
        return {"cvss_score": None, "cvss_vector": None, "severity": None}


def _fetch_epss_batch(cve_ids: list[str]) -> dict[str, dict[str, float | None]]:
    """Fetch EPSS scores for a batch of CVEs (up to 100 per API call)."""
    result: dict[str, dict[str, float | None]] = {}

    for i in range(0, len(cve_ids), _EPSS_BATCH_SIZE):
        batch = cve_ids[i : i + _EPSS_BATCH_SIZE]
        cve_param = ",".join(c.upper() for c in batch)

        try:
            resp = _get_with_retry(
                _EPSS_API_URL,
                params={"cve": cve_param},
                timeout=20,
            )
            if resp.status_code != 200:
                for cve_id in batch:
                    result[cve_id.upper()] = {"epss": None, "percentile": None}
                continue

            data = resp.json()
            epss_data = data.get("data", [])
            found_ids = set()
            for entry in epss_data:
                cid = entry.get("cve", "").upper()
                found_ids.add(cid)
                result[cid] = {
                    "epss": float(entry["epss"]) if entry.get("epss") else None,
                    "percentile": float(entry["percentile"]) if entry.get("percentile") else None,
                }

            # Mark missing CVEs
            for cve_id in batch:
                if cve_id.upper() not in found_ids:
                    result[cve_id.upper()] = {"epss": None, "percentile": None}

        except Exception:
            for cve_id in batch:
                result[cve_id.upper()] = {"epss": None, "percentile": None}

    return result


def _fetch_kev_catalogue() -> set[str]:
    """Fetch the CISA KEV catalogue and return a set of CVE IDs."""
    try:
        resp = _get_with_retry(_KEV_URL, timeout=30)
        if resp.status_code != 200:
            return set()
        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        return {v.get("cveID", "").upper() for v in vulns if v.get("cveID")}
    except Exception:
        return set()


def _fetch_vulncheck_kev(cve_ids: list[str]) -> set[str]:
    """Check VulnCheck KEV for CVE IDs. Returns set of CVEs found in VulnCheck KEV."""
    api_key = os.environ.get("VULNCHECK_API_KEY")
    if not api_key:
        return set()

    found: set[str] = set()
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

    for cve_id in cve_ids:
        try:
            resp = _get_with_retry(
                _VULNCHECK_KEV_URL,
                params={"cve": cve_id.upper()},
                headers=headers,
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("data"):
                    found.add(cve_id.upper())
        except Exception:
            continue

    return found


# ---------------------------------------------------------------------------
# Triage scoring
# ---------------------------------------------------------------------------


def _compute_triage_score(
    *,
    cvss_score: float | None,
    epss_score: float | None,
    in_kev: bool,
) -> float:
    """Compute composite triage score (0-100).

    Formula:
        score = (CVSS_norm * W_cvss) + (EPSS * W_epss) + (KEV * W_kev) * 100

    Where:
        - CVSS_norm = cvss_score / 10.0 (normalized to 0-1)
        - EPSS = epss_score (already 0-1)
        - KEV = 1.0 if in KEV catalogue, 0.0 otherwise
    """
    cvss_norm = (cvss_score / 10.0) if cvss_score is not None else 0.0
    epss_val = epss_score if epss_score is not None else 0.0
    kev_val = 1.0 if in_kev else 0.0

    raw = (cvss_norm * _WEIGHT_CVSS) + (epss_val * _WEIGHT_EPSS) + (kev_val * _WEIGHT_KEV)
    return round(raw * 100, 1)


def _severity_label(score: float) -> str:
    """Map triage score to urgency label."""
    if score >= 80:
        return "CRITICAL"
    if score >= 60:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    if score >= 20:
        return "LOW"
    return "INFO"


# ---------------------------------------------------------------------------
# Core triage logic
# ---------------------------------------------------------------------------


def triage_cves(cve_ids: list[str], output_format: str = "text") -> dict[str, Any]:
    """Run batch triage on a list of CVE IDs.

    Returns a dict with:
        - results: list of per-CVE triage records, sorted by score descending
        - summary: aggregate stats
        - formatted: human-readable text (when output_format='text')
    """
    # Validate inputs
    if not cve_ids:
        return {"error": "No CVE IDs provided.", "results": [], "summary": {}}

    if len(cve_ids) > 50:
        return {"error": "Maximum 50 CVE IDs per triage call.", "results": [], "summary": {}}

    # Normalize and validate
    normalized: list[str] = []
    invalid: list[str] = []
    for cid in cve_ids:
        cid_stripped = cid.strip()
        if _CVE_RE.match(cid_stripped):
            normalized.append(cid_stripped.upper())
        else:
            invalid.append(cid_stripped)

    if not normalized:
        return {
            "error": f"No valid CVE IDs found. Invalid: {invalid}",
            "results": [],
            "summary": {},
        }

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_cves: list[str] = []
    for cid in normalized:
        if cid not in seen:
            seen.add(cid)
            unique_cves.append(cid)

    # Fetch data in parallel-friendly batches
    # 1. EPSS (batch API — most efficient)
    epss_data = _fetch_epss_batch(unique_cves)

    # 2. CISA KEV (single fetch, then set lookup)
    kev_set = _fetch_kev_catalogue()

    # 3. VulnCheck KEV (optional enrichment)
    vc_kev_set = _fetch_vulncheck_kev(unique_cves)

    # 4. NVD CVSS (per-CVE, with rate limiting)
    nvd_data: dict[str, dict[str, Any]] = {}
    api_key = os.environ.get("NVD_API_KEY")
    for i, cve_id in enumerate(unique_cves):
        nvd_data[cve_id] = _fetch_nvd_cvss(cve_id)
        # Rate limit: wait between requests (faster with API key)
        if i < len(unique_cves) - 1:
            if api_key:
                time.sleep(0.15)
            else:
                time.sleep(_NVD_DELAY)

    # Build triage records
    records: list[dict[str, Any]] = []
    for cve_id in unique_cves:
        nvd = nvd_data.get(cve_id, {})
        epss = epss_data.get(cve_id, {})
        in_cisa_kev = cve_id in kev_set
        in_vc_kev = cve_id in vc_kev_set
        in_any_kev = in_cisa_kev or in_vc_kev

        cvss_score = nvd.get("cvss_score")
        epss_score = epss.get("epss")

        triage_score = _compute_triage_score(
            cvss_score=cvss_score,
            epss_score=epss_score,
            in_kev=in_any_kev,
        )

        records.append(
            {
                "cve_id": cve_id,
                "cvss_score": cvss_score,
                "cvss_severity": nvd.get("severity"),
                "epss_score": epss_score,
                "epss_percentile": epss.get("percentile"),
                "in_cisa_kev": in_cisa_kev,
                "in_vulncheck_kev": in_vc_kev,
                "triage_score": triage_score,
                "urgency": _severity_label(triage_score),
            }
        )

    # Sort by triage score descending
    records.sort(key=lambda r: r["triage_score"], reverse=True)

    # Summary stats
    summary = {
        "total_cves": len(records),
        "critical_count": sum(1 for r in records if r["urgency"] == "CRITICAL"),
        "high_count": sum(1 for r in records if r["urgency"] == "HIGH"),
        "medium_count": sum(1 for r in records if r["urgency"] == "MEDIUM"),
        "low_count": sum(1 for r in records if r["urgency"] == "LOW"),
        "info_count": sum(1 for r in records if r["urgency"] == "INFO"),
        "kev_count": sum(1 for r in records if r["in_cisa_kev"] or r["in_vulncheck_kev"]),
        "invalid_ids": invalid if invalid else None,
    }

    result: dict[str, Any] = {
        "results": records,
        "summary": summary,
    }

    if output_format == "text":
        result["formatted"] = _format_text(records, summary, invalid)

    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _format_text(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    invalid: list[str],
) -> str:
    """Format triage results as a human-readable text table."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  CVE TRIAGE REPORT — Prioritized by Composite Score")
    lines.append("=" * 72)
    lines.append("")

    # Table header
    lines.append(f"{'#':<4} {'CVE ID':<18} {'Score':<7} {'Urgency':<10} {'CVSS':<6} {'EPSS':<7} {'KEV':<5}")
    lines.append("-" * 72)

    for i, rec in enumerate(records, 1):
        cvss_str = f"{rec['cvss_score']:.1f}" if rec["cvss_score"] is not None else "N/A"
        epss_str = f"{rec['epss_score']:.4f}" if rec["epss_score"] is not None else "N/A"
        kev_str = "YES" if (rec["in_cisa_kev"] or rec["in_vulncheck_kev"]) else "no"

        lines.append(
            f"{i:<4} {rec['cve_id']:<18} {rec['triage_score']:<7.1f} "
            f"{rec['urgency']:<10} {cvss_str:<6} {epss_str:<7} {kev_str:<5}"
        )

    lines.append("-" * 72)
    lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append(f"  Total CVEs triaged: {summary['total_cves']}")
    lines.append(
        f"  CRITICAL: {summary['critical_count']}  |  HIGH: {summary['high_count']}  |  "
        f"MEDIUM: {summary['medium_count']}  |  LOW: {summary['low_count']}  |  "
        f"INFO: {summary['info_count']}"
    )
    lines.append(f"  In KEV catalogue: {summary['kev_count']}")

    if invalid:
        lines.append(f"  Skipped (invalid format): {', '.join(invalid)}")

    lines.append("")
    lines.append("SCORING METHODOLOGY")
    lines.append(
        f"  Composite = (CVSS/10 × {_WEIGHT_CVSS:.0%}) + (EPSS × {_WEIGHT_EPSS:.0%}) + (KEV × {_WEIGHT_KEV:.0%})"
    )
    lines.append("  Sources: NVD (CVSS), FIRST.org (EPSS), CISA KEV, VulnCheck KEV")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strands tool handler
# ---------------------------------------------------------------------------


def triage_cves_handler(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands-compatible tool handler for triage_cves."""
    tool_input = tool.get("input", {})
    cve_ids = tool_input.get("cve_ids", [])
    output_format = tool_input.get("output_format", "text")

    result = triage_cves(cve_ids, output_format=output_format)

    if "error" in result and not result["results"]:
        content_text = f"Error: {result['error']}"
    elif output_format == "json":
        import json

        content_text = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        content_text = result.get("formatted", str(result))

    log_tool_output_size("triage_cves", content_text)

    return {
        "toolUseId": tool["toolUseId"],
        "status": "error" if ("error" in result and not result["results"]) else "success",
        "content": [{"text": content_text}],
    }
