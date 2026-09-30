"""Cross-reference CVE correlation tool.

Given a seed CVE, discovers related vulnerabilities that share the same
affected component (NVD CPE vendor:product) or the same root-cause weakness
(CWE). This answers the question security teams ask after triaging one CVE:
"What else should we check?"

Data sources:
- **NVD 2.0 API** — CPE configurations (vendor:product extraction) and CWE
  weakness IDs from the seed CVE, then keyword search for correlated CVEs.
- **FIRST.org EPSS API** — bulk EPSS scores for all correlated CVEs to
  surface which related CVEs have highest exploitation probability.
- **CISA KEV catalog** — flags correlated CVEs that are actively exploited.

All HTTP calls use exponential back-off retry via ``_nvd_get_with_retry``
(for NVD endpoints) and plain ``requests`` with timeout for EPSS/KEV.
"""

from __future__ import annotations

import re
import time
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

__all__ = ["correlate_cves", "TOOL_SPEC"]

# ---------------------------------------------------------------------------
# Strands tool definition
# ---------------------------------------------------------------------------

TOOL_SPEC = {
    "name": "correlate_cves",
    "description": (
        "Cross-reference correlation for a CVE: finds related vulnerabilities "
        "that share the same affected software component (NVD CPE vendor:product) "
        "or the same root-cause weakness class (CWE). Returns correlated CVEs "
        "enriched with EPSS scores and CISA KEV status, ranked by exploitation "
        "probability. Use after get_nvd_data to discover the full attack surface "
        "around a known vulnerability."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The seed CVE identifier (e.g. 'CVE-2024-3094').",
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        "Maximum number of correlated CVEs to return per "
                        "dimension (CPE, CWE). Defaults to 20. Range: 1-50."
                    ),
                    "default": 20,
                },
            },
            "required": ["cve_id"],
        }
    },
}

# ---------------------------------------------------------------------------
# CVE-ID validation
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)

_NVD_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_EPSS_BASE = "https://api.first.org/data/v1/epss"
_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

# Rate-limit courteous delay between NVD calls (ms).
_NVD_DELAY_S = 0.8


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nvd_headers() -> dict[str, str]:
    """Return NVD request headers, injecting NVD_API_KEY when available."""
    import os

    headers: dict[str, str] = {"User-Agent": "manus-agent/correlate-cves"}
    api_key = os.environ.get("NVD_API_KEY", "").strip()
    if api_key:
        headers["apiKey"] = api_key
    return headers


def _nvd_get(url: str, params: dict[str, Any] | None = None, timeout: int = 30) -> requests.Response:
    """GET from NVD with exponential back-off retry (3 attempts)."""
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=_nvd_headers(), timeout=timeout)
            if resp.status_code == 403:
                # Rate-limited — back off harder
                time.sleep(2 ** (attempt + 1))
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            time.sleep(2**attempt)
    raise last_exc  # type: ignore[misc]


def _fetch_seed_cve(cve_id: str) -> dict[str, Any] | None:
    """Fetch the seed CVE record from NVD 2.0 API."""
    resp = _nvd_get(_NVD_BASE, params={"cveId": cve_id})
    data = resp.json()
    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return None
    return vulns[0].get("cve", {})


def _extract_cpe_products(cve_record: dict[str, Any]) -> list[dict[str, str]]:
    """Extract unique vendor:product pairs from NVD CPE configurations.

    Returns a list of dicts with keys 'vendor', 'product', 'cpe_prefix'.
    """
    products: dict[str, dict[str, str]] = {}
    for config in cve_record.get("configurations", []):
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                if not match.get("vulnerable", False):
                    continue
                criteria = match.get("criteria", "")
                # CPE 2.3 format: cpe:2.3:a:vendor:product:version:...
                parts = criteria.split(":")
                if len(parts) >= 5:
                    vendor = parts[3]
                    product = parts[4]
                    key = f"{vendor}:{product}"
                    if key not in products and vendor != "*" and product != "*":
                        products[key] = {
                            "vendor": vendor,
                            "product": product,
                            "cpe_prefix": f"cpe:2.3:*:{vendor}:{product}",
                        }
    return list(products.values())


def _extract_cwes(cve_record: dict[str, Any]) -> list[str]:
    """Extract CWE IDs from the CVE record."""
    cwes: list[str] = []
    for weakness in cve_record.get("weaknesses", []):
        for desc in weakness.get("description", []):
            val = desc.get("value", "")
            if val.startswith("CWE-") and val not in cwes and val != "CWE-noinfo":
                cwes.append(val)
    return cwes


def _search_by_cpe(vendor: str, product: str, seed_cve_id: str, max_results: int) -> list[dict[str, Any]]:
    """Search NVD for other CVEs affecting the same vendor:product."""
    # Use keywordSearch — NVD 2.0 supports cpeName filtering but it requires
    # an exact CPE string. Keyword search on "vendor product" is more practical.
    params: dict[str, Any] = {
        "keywordSearch": f"{vendor} {product}",
        "keywordExactMatch": "",
        "resultsPerPage": min(max_results + 5, 50),  # over-fetch to allow dedup
    }
    try:
        resp = _nvd_get(_NVD_BASE, params=params)
        data = resp.json()
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    for vuln in data.get("vulnerabilities", []):
        cve = vuln.get("cve", {})
        cve_id = cve.get("id", "")
        if cve_id == seed_cve_id.upper():
            continue
        # Verify this CVE actually has a CPE matching vendor:product
        has_matching_cpe = False
        for config in cve.get("configurations", []):
            for node in config.get("nodes", []):
                for match in node.get("cpeMatch", []):
                    criteria = match.get("criteria", "")
                    parts = criteria.split(":")
                    if len(parts) >= 5 and parts[3] == vendor and parts[4] == product:
                        has_matching_cpe = True
                        break
                if has_matching_cpe:
                    break
            if has_matching_cpe:
                break
        if has_matching_cpe:
            results.append(_summarize_cve(cve))
    return results[:max_results]


def _search_by_cwe(cwe_id: str, seed_cve_id: str, max_results: int) -> list[dict[str, Any]]:
    """Search NVD for other CVEs with the same CWE weakness."""
    params: dict[str, Any] = {
        "cweId": cwe_id,
        "resultsPerPage": min(max_results + 5, 50),
    }
    try:
        resp = _nvd_get(_NVD_BASE, params=params)
        data = resp.json()
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    for vuln in data.get("vulnerabilities", []):
        cve = vuln.get("cve", {})
        cve_id = cve.get("id", "")
        if cve_id == seed_cve_id.upper():
            continue
        results.append(_summarize_cve(cve))
    return results[:max_results]


def _summarize_cve(cve: dict[str, Any]) -> dict[str, Any]:
    """Extract a compact summary dict from an NVD CVE record."""
    cve_id = cve.get("id", "")
    description = ""
    for desc in cve.get("descriptions", []):
        if desc.get("lang") == "en":
            description = desc.get("value", "")
            break

    # CVSS score — prefer v3.1, fallback to v3.0, then v2.0
    cvss_score: float | None = None
    cvss_severity: str = "UNKNOWN"
    for metric_key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        metrics = cve.get("metrics", {}).get(metric_key, [])
        if metrics:
            cvss_data = metrics[0].get("cvssData", {})
            cvss_score = cvss_data.get("baseScore")
            cvss_severity = cvss_data.get("baseSeverity", "UNKNOWN")
            break

    # CWEs
    cwes: list[str] = []
    for weakness in cve.get("weaknesses", []):
        for desc in weakness.get("description", []):
            val = desc.get("value", "")
            if val.startswith("CWE-") and val != "CWE-noinfo":
                cwes.append(val)

    published = cve.get("published", "")

    return {
        "cve_id": cve_id,
        "description": description[:200] + ("…" if len(description) > 200 else ""),
        "cvss_score": cvss_score,
        "cvss_severity": cvss_severity,
        "cwes": cwes,
        "published": published[:10] if published else "",
    }


def _bulk_epss(cve_ids: list[str]) -> dict[str, dict[str, float]]:
    """Fetch EPSS scores for a batch of CVE IDs (max 100 per call)."""
    if not cve_ids:
        return {}
    result: dict[str, dict[str, float]] = {}
    # EPSS API accepts up to ~100 CVEs per request
    for i in range(0, len(cve_ids), 100):
        chunk = cve_ids[i : i + 100]
        try:
            resp = requests.get(
                _EPSS_BASE,
                params={"cve": ",".join(chunk)},
                timeout=20,
            )
            resp.raise_for_status()
            for item in resp.json().get("data", []):
                cve_id = item.get("cve", "")
                result[cve_id] = {
                    "epss": float(item.get("epss", 0)),
                    "percentile": float(item.get("percentile", 0)),
                }
        except Exception:
            pass
    return result


def _fetch_kev_set() -> set[str]:
    """Fetch the CISA KEV catalog and return a set of CVE IDs."""
    try:
        resp = requests.get(_KEV_URL, timeout=20)
        resp.raise_for_status()
        return {v.get("cveID", "") for v in resp.json().get("vulnerabilities", [])}
    except Exception:
        return set()


def _enrich_results(
    results: list[dict[str, Any]],
    epss_map: dict[str, dict[str, float]],
    kev_set: set[str],
) -> list[dict[str, Any]]:
    """Add EPSS and KEV data to each result, then sort by EPSS descending."""
    for item in results:
        cve_id = item["cve_id"]
        epss_data = epss_map.get(cve_id, {})
        item["epss_score"] = epss_data.get("epss")
        item["epss_percentile"] = epss_data.get("percentile")
        item["in_kev"] = cve_id in kev_set
    # Sort: KEV first, then by EPSS descending, then by CVSS descending
    results.sort(
        key=lambda x: (
            not x.get("in_kev", False),
            -(x.get("epss_score") or 0),
            -(x.get("cvss_score") or 0),
        )
    )
    return results


# ---------------------------------------------------------------------------
# Main tool function
# ---------------------------------------------------------------------------


def correlate_cves(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Find CVEs correlated to the seed CVE by shared CPE and/or CWE."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id", "")
    max_results = min(max(int(tool_input.get("max_results", 20)), 1), 50)

    # Validate CVE ID
    if not isinstance(cve_id, str) or not _CVE_RE.match(cve_id.strip()):
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid CVE ID format. Expected pattern: CVE-YYYY-NNNN (≥ 4 digits)."}],
        }
        log_tool_output_size("correlate_cves", result)
        return result

    cve_id = cve_id.strip().upper()

    # 1. Fetch the seed CVE
    try:
        seed = _fetch_seed_cve(cve_id)
    except Exception as exc:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Failed to fetch seed CVE from NVD: {exc}"}],
        }
        log_tool_output_size("correlate_cves", result)
        return result

    if seed is None:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"CVE {cve_id} not found in NVD."}],
        }
        log_tool_output_size("correlate_cves", result)
        return result

    # 2. Extract correlation dimensions
    cpe_products = _extract_cpe_products(seed)
    cwes = _extract_cwes(seed)

    if not cpe_products and not cwes:
        result = {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [
                {
                    "text": (
                        f"No correlation dimensions found for {cve_id}. "
                        "The CVE has no CPE configurations or CWE assignments in NVD."
                    )
                },
                {
                    "json": {
                        "seed_cve": cve_id,
                        "cpe_products": [],
                        "cwes": [],
                        "correlations": {"by_component": [], "by_weakness": []},
                        "total_unique_correlated": 0,
                    }
                },
            ],
        }
        log_tool_output_size("correlate_cves", result)
        return result

    # 3. Search for correlated CVEs
    by_component: list[dict[str, Any]] = []
    seen_ids: set[str] = {cve_id}

    for prod in cpe_products[:3]:  # Limit to top 3 products to avoid API overload
        time.sleep(_NVD_DELAY_S)
        matches = _search_by_cpe(prod["vendor"], prod["product"], cve_id, max_results)
        for m in matches:
            if m["cve_id"] not in seen_ids:
                m["correlation_source"] = f"cpe:{prod['vendor']}:{prod['product']}"
                by_component.append(m)
                seen_ids.add(m["cve_id"])

    by_weakness: list[dict[str, Any]] = []
    for cwe in cwes[:2]:  # Limit to top 2 CWEs
        time.sleep(_NVD_DELAY_S)
        matches = _search_by_cwe(cwe, cve_id, max_results)
        for m in matches:
            if m["cve_id"] not in seen_ids:
                m["correlation_source"] = cwe
                by_weakness.append(m)
                seen_ids.add(m["cve_id"])

    # 4. Enrich with EPSS + KEV
    all_cve_ids = [c["cve_id"] for c in by_component + by_weakness]
    epss_map = _bulk_epss(all_cve_ids)
    kev_set = _fetch_kev_set()

    by_component = _enrich_results(by_component, epss_map, kev_set)
    by_weakness = _enrich_results(by_weakness, epss_map, kev_set)

    # Trim to max_results per dimension
    by_component = by_component[:max_results]
    by_weakness = by_weakness[:max_results]

    total_unique = len({c["cve_id"] for c in by_component + by_weakness})

    # Count high-priority
    kev_count = sum(1 for c in by_component + by_weakness if c.get("in_kev"))
    high_epss_count = sum(1 for c in by_component + by_weakness if (c.get("epss_score") or 0) >= 0.1)

    # 5. Build human-readable summary
    lines: list[str] = [
        f"CVE Correlation Report for {cve_id}",
        f"{'=' * 50}",
        "",
        f"Seed CVE: {cve_id}",
        f"CPE products found: {len(cpe_products)} ({', '.join(p['vendor'] + ':' + p['product'] for p in cpe_products[:3])})",
        f"CWE weaknesses found: {len(cwes)} ({', '.join(cwes[:3])})",
        "",
        f"Total unique correlated CVEs: {total_unique}",
        f"  In CISA KEV (actively exploited): {kev_count}",
        f"  High EPSS (≥ 0.1): {high_epss_count}",
    ]

    if by_component:
        lines.append("")
        lines.append(f"By Component ({len(by_component)} CVEs):")
        lines.append("-" * 40)
        for c in by_component[:10]:
            kev_flag = " 🚨KEV" if c.get("in_kev") else ""
            epss_str = f"EPSS={c['epss_score']:.4f}" if c.get("epss_score") is not None else "EPSS=N/A"
            cvss_str = f"CVSS={c['cvss_score']}" if c.get("cvss_score") is not None else "CVSS=N/A"
            lines.append(f"  {c['cve_id']}  {cvss_str}  {epss_str}{kev_flag}")
            lines.append(f"    via {c['correlation_source']}")
            lines.append(f"    {c['description']}")

    if by_weakness:
        lines.append("")
        lines.append(f"By Weakness ({len(by_weakness)} CVEs):")
        lines.append("-" * 40)
        for c in by_weakness[:10]:
            kev_flag = " 🚨KEV" if c.get("in_kev") else ""
            epss_str = f"EPSS={c['epss_score']:.4f}" if c.get("epss_score") is not None else "EPSS=N/A"
            cvss_str = f"CVSS={c['cvss_score']}" if c.get("cvss_score") is not None else "CVSS=N/A"
            lines.append(f"  {c['cve_id']}  {cvss_str}  {epss_str}{kev_flag}")
            lines.append(f"    via {c['correlation_source']}")
            lines.append(f"    {c['description']}")

    summary = "\n".join(lines)

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [
            {"text": summary},
            {
                "json": {
                    "seed_cve": cve_id,
                    "cpe_products": [{"vendor": p["vendor"], "product": p["product"]} for p in cpe_products],
                    "cwes": cwes,
                    "correlations": {
                        "by_component": by_component,
                        "by_weakness": by_weakness,
                    },
                    "total_unique_correlated": total_unique,
                    "kev_correlated_count": kev_count,
                    "high_epss_count": high_epss_count,
                }
            },
        ],
    }
    log_tool_output_size("correlate_cves", result)
    return result
