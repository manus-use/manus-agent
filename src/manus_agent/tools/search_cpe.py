#!/usr/bin/env python3
"""
Search NVD for CPE (Common Platform Enumeration) matches and retrieve CVEs
affecting a given software product/version.

Answers the question: *"What CVEs affect <product> <version>?"*

Two-stage pipeline
------------------
1. **CPE search** — Query the NVD CPE API (``/cpes/2.0``) by keyword to
   discover matching CPE 2.3 URIs (e.g. ``cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*``).
2. **CVE lookup** — For each matched CPE, query the NVD CVE API
   (``/cves/2.0?cpeName=<cpe>``) to retrieve associated CVEs with CVSS
   scores and severity labels.

Rate-limit resilience
---------------------
NVD public API: 5 req/30 s (unauthenticated), 50 req/30 s (with key).
Exponential back-off retry on 429/5xx, inter-request delay to stay
under rate limits.

Optional API key
----------------
Set ``NVD_API_KEY`` in the environment to get higher rate limits.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

TOOL_SPEC = {
    "name": "search_cpe",
    "description": (
        "Search NVD for CPE (Common Platform Enumeration) matches by product "
        "keyword and optionally retrieve all CVEs affecting the matched CPEs. "
        "Use this to answer 'what CVEs affect <product> <version>?' — e.g. "
        "search_cpe(keyword='apache log4j', version='2.14.1'). "
        "Returns matched CPE URIs, vendor/product info, and optionally a "
        "ranked list of CVEs with CVSS scores."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": (
                        "Product search keyword(s). Examples: 'apache log4j', "
                        "'openssl', 'linux kernel', 'microsoft exchange'."
                    ),
                },
                "version": {
                    "type": "string",
                    "description": (
                        "Optional version string to filter CPE matches. Example: '2.14.1', '3.0.7', '5.15'."
                    ),
                },
                "cpe_type": {
                    "type": "string",
                    "description": (
                        "CPE part filter: 'a' (application), 'o' (operating system), 'h' (hardware). Default: 'a'."
                    ),
                },
                "fetch_cves": {
                    "type": "boolean",
                    "description": (
                        "If true (default), also fetch CVEs for the top matched CPEs. Set to false for CPE-only search."
                    ),
                },
                "max_cpes": {
                    "type": "integer",
                    "description": ("Maximum number of CPE matches to return (1-50, default 10)."),
                },
                "max_cves_per_cpe": {
                    "type": "integer",
                    "description": ("Maximum CVEs to fetch per CPE (1-100, default 20)."),
                },
            },
            "required": ["keyword"],
        }
    },
}

# ---------------------------------------------------------------------------
# Retry / back-off constants
# ---------------------------------------------------------------------------
_MAX_RETRIES = int(os.environ.get("NVD_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY = float(os.environ.get("NVD_RETRY_BASE_DELAY", "2"))
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Inter-request delay to respect NVD rate limits
_DELAY_WITH_KEY = float(os.environ.get("NVD_DELAY_WITH_KEY", "0.15"))
_DELAY_WITHOUT_KEY = float(os.environ.get("NVD_DELAY_WITHOUT_KEY", "0.7"))

# NVD API endpoints
_NVD_CPE_URL = "https://services.nvd.nist.gov/rest/json/cpes/2.0"
_NVD_CVE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# Valid CPE part values
_VALID_CPE_TYPES = {"a", "o", "h"}


def _build_nvd_headers() -> dict[str, str]:
    """Return request headers, injecting NVD_API_KEY when available."""
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY", "").strip()
    if api_key:
        headers["apiKey"] = api_key
    return headers


def _has_api_key() -> bool:
    return bool(os.environ.get("NVD_API_KEY", "").strip())


def _inter_request_delay() -> None:
    """Sleep between NVD API calls to respect rate limits."""
    delay = _DELAY_WITH_KEY if _has_api_key() else _DELAY_WITHOUT_KEY
    time.sleep(delay)


def _nvd_get_with_retry(url: str, params: dict[str, Any] | None = None, *, timeout: int = 15) -> requests.Response:
    """GET with exponential back-off retry on 429 / transient errors."""
    headers = _build_nvd_headers()
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        if attempt > 0:
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            time.sleep(delay)
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code in _RETRYABLE_STATUS:
                last_exc = requests.exceptions.HTTPError(f"HTTP {resp.status_code}", response=resp)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            continue
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            continue
        except requests.exceptions.HTTPError as exc:
            if hasattr(exc, "response") and exc.response is not None:
                if exc.response.status_code in _RETRYABLE_STATUS:
                    last_exc = exc
                    continue
            raise

    raise last_exc or RuntimeError("All retries exhausted")


# ---------------------------------------------------------------------------
# CPE search
# ---------------------------------------------------------------------------


def _parse_cpe_uri(cpe_uri: str) -> dict[str, str]:
    """Parse a CPE 2.3 URI into its component parts.

    Example: cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*
    """
    parts = cpe_uri.split(":")
    result: dict[str, str] = {"cpe_uri": cpe_uri}
    labels = [
        "cpe_version",
        "part",
        "vendor",
        "product",
        "version",
        "update",
        "edition",
        "language",
        "sw_edition",
        "target_sw",
        "target_hw",
        "other",
    ]
    for i, label in enumerate(labels):
        idx = i + 1  # skip 'cpe' prefix
        if idx < len(parts):
            val = parts[idx]
            if val not in ("*", "-"):
                result[label] = val
    return result


def search_cpes(keyword: str, *, version: str = "", cpe_type: str = "a", max_results: int = 10) -> dict[str, Any]:
    """Search NVD for CPE matches by keyword.

    Returns:
        dict with keys: cpes (list), total_results (int), keyword, version,
        cpe_type.
    """
    keyword = keyword.strip()
    if not keyword:
        return {"error": "keyword is required", "cpes": [], "total_results": 0}

    cpe_type = cpe_type.strip().lower()
    if cpe_type not in _VALID_CPE_TYPES:
        cpe_type = "a"

    max_results = max(1, min(50, max_results))

    params: dict[str, Any] = {
        "keywordSearch": keyword,
        "resultsPerPage": max_results,
    }

    try:
        resp = _nvd_get_with_retry(_NVD_CPE_URL, params=params)
    except Exception as exc:
        return {
            "error": f"NVD CPE API request failed: {exc}",
            "cpes": [],
            "total_results": 0,
            "keyword": keyword,
            "version": version,
        }

    data = resp.json()
    products = data.get("products", [])
    total = data.get("totalResults", 0)

    cpes: list[dict[str, Any]] = []
    for product_entry in products:
        cpe_data = product_entry.get("cpe", {})
        cpe_uri = cpe_data.get("cpeName", "")
        if not cpe_uri:
            continue

        parsed = _parse_cpe_uri(cpe_uri)

        # Filter by CPE type
        if parsed.get("part", "") != cpe_type:
            continue

        # Filter by version if specified
        if version:
            cpe_version = parsed.get("version", "")
            if cpe_version and version.lower() not in cpe_version.lower():
                continue

        entry: dict[str, Any] = {
            "cpe_uri": cpe_uri,
            "vendor": parsed.get("vendor", ""),
            "product": parsed.get("product", ""),
            "version": parsed.get("version", ""),
            "title": _first_title(cpe_data.get("titles", [])),
            "deprecated": cpe_data.get("deprecated", False),
            "last_modified": cpe_data.get("lastModified", ""),
        }
        cpes.append(entry)

        if len(cpes) >= max_results:
            break

    return {
        "cpes": cpes,
        "total_results": total,
        "returned": len(cpes),
        "keyword": keyword,
        "version": version,
        "cpe_type": cpe_type,
    }


def _first_title(titles: list[dict[str, str]]) -> str:
    """Extract the first English title from NVD CPE titles list."""
    for t in titles:
        if t.get("lang", "") in ("en", "en-US"):
            return t.get("title", "")
    if titles:
        return titles[0].get("title", "")
    return ""


# ---------------------------------------------------------------------------
# CVE lookup by CPE
# ---------------------------------------------------------------------------


def fetch_cves_for_cpe(cpe_uri: str, *, max_results: int = 20) -> dict[str, Any]:
    """Fetch CVEs from NVD that match a given CPE URI.

    Returns:
        dict with keys: cves (list), total_results (int), cpe_uri.
    """
    cpe_uri = cpe_uri.strip()
    if not cpe_uri:
        return {"error": "cpe_uri is required", "cves": [], "total_results": 0}

    max_results = max(1, min(100, max_results))

    params: dict[str, Any] = {
        "cpeName": cpe_uri,
        "resultsPerPage": max_results,
    }

    try:
        resp = _nvd_get_with_retry(_NVD_CVE_URL, params=params)
    except Exception as exc:
        return {
            "error": f"NVD CVE API request failed: {exc}",
            "cves": [],
            "total_results": 0,
            "cpe_uri": cpe_uri,
        }

    data = resp.json()
    vulnerabilities = data.get("vulnerabilities", [])
    total = data.get("totalResults", 0)

    cves: list[dict[str, Any]] = []
    for vuln_entry in vulnerabilities:
        cve_data = vuln_entry.get("cve", {})
        cve_id = cve_data.get("id", "")
        if not cve_id:
            continue

        metrics = cve_data.get("metrics", {})
        cvss_info = _extract_cvss(metrics)

        descriptions = cve_data.get("descriptions", [])
        description = _first_en_description(descriptions)

        cves.append(
            {
                "cve_id": cve_id,
                "description": description[:300] if description else "",
                "published": cve_data.get("published", ""),
                "last_modified": cve_data.get("lastModified", ""),
                **cvss_info,
            }
        )

    # Sort by CVSS score descending (most severe first)
    cves.sort(key=lambda c: c.get("cvss_score", 0.0), reverse=True)

    return {
        "cves": cves,
        "total_results": total,
        "returned": len(cves),
        "cpe_uri": cpe_uri,
    }


def _extract_cvss(metrics: dict[str, Any]) -> dict[str, Any]:
    """Extract the best available CVSS score from NVD metrics."""
    # Prefer v3.1 > v3.0 > v2.0
    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key, [])
        if entries:
            first = entries[0]
            cvss_data = first.get("cvssData", {})
            return {
                "cvss_version": cvss_data.get("version", ""),
                "cvss_score": cvss_data.get("baseScore", 0.0),
                "cvss_severity": cvss_data.get("baseSeverity", ""),
                "cvss_vector": cvss_data.get("vectorString", ""),
                "exploitability_score": first.get("exploitabilityScore", 0.0),
                "impact_score": first.get("impactScore", 0.0),
            }

    entries_v2 = metrics.get("cvssMetricV2", [])
    if entries_v2:
        first = entries_v2[0]
        cvss_data = first.get("cvssData", {})
        score = cvss_data.get("baseScore", 0.0)
        severity = "LOW"
        if score >= 7.0:
            severity = "HIGH"
        elif score >= 4.0:
            severity = "MEDIUM"
        return {
            "cvss_version": "2.0",
            "cvss_score": score,
            "cvss_severity": severity,
            "cvss_vector": cvss_data.get("vectorString", ""),
            "exploitability_score": first.get("exploitabilityScore", 0.0),
            "impact_score": first.get("impactScore", 0.0),
        }

    return {
        "cvss_version": "",
        "cvss_score": 0.0,
        "cvss_severity": "",
        "cvss_vector": "",
        "exploitability_score": 0.0,
        "impact_score": 0.0,
    }


def _first_en_description(descriptions: list[dict[str, str]]) -> str:
    """Extract the first English description from NVD CVE descriptions."""
    for d in descriptions:
        if d.get("lang", "") in ("en", "en-US"):
            return d.get("value", "")
    if descriptions:
        return descriptions[0].get("value", "")
    return ""


# ---------------------------------------------------------------------------
# Combined search: CPE discovery + CVE lookup
# ---------------------------------------------------------------------------


def search_cpe_and_cves(
    keyword: str,
    *,
    version: str = "",
    cpe_type: str = "a",
    fetch_cves: bool = True,
    max_cpes: int = 10,
    max_cves_per_cpe: int = 20,
) -> dict[str, Any]:
    """Two-stage pipeline: discover CPEs, then fetch CVEs for each.

    Returns:
        dict with keys: keyword, version, cpe_type, cpe_results,
        cve_results (if fetch_cves=True), summary.
    """
    max_cpes = max(1, min(50, max_cpes))
    max_cves_per_cpe = max(1, min(100, max_cves_per_cpe))

    cpe_result = search_cpes(
        keyword,
        version=version,
        cpe_type=cpe_type,
        max_results=max_cpes,
    )

    if cpe_result.get("error"):
        return {
            "keyword": keyword,
            "version": version,
            "cpe_type": cpe_type,
            "error": cpe_result["error"],
            "cpe_results": cpe_result,
            "cve_results": [],
            "summary": {
                "cpes_found": 0,
                "total_unique_cves": 0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
            },
        }

    if not fetch_cves or not cpe_result["cpes"]:
        return {
            "keyword": keyword,
            "version": version,
            "cpe_type": cpe_type,
            "cpe_results": cpe_result,
            "cve_results": [],
            "summary": {
                "cpes_found": len(cpe_result["cpes"]),
                "total_unique_cves": 0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
            },
        }

    # Fetch CVEs for each matched CPE
    all_cve_results: list[dict[str, Any]] = []
    seen_cve_ids: set[str] = set()
    unique_cves: list[dict[str, Any]] = []

    for cpe in cpe_result["cpes"]:
        _inter_request_delay()
        cve_result = fetch_cves_for_cpe(
            cpe["cpe_uri"],
            max_results=max_cves_per_cpe,
        )
        all_cve_results.append(
            {
                "cpe_uri": cpe["cpe_uri"],
                "vendor": cpe.get("vendor", ""),
                "product": cpe.get("product", ""),
                "version": cpe.get("version", ""),
                **cve_result,
            }
        )

        for cve in cve_result.get("cves", []):
            cve_id = cve.get("cve_id", "")
            if cve_id and cve_id not in seen_cve_ids:
                seen_cve_ids.add(cve_id)
                unique_cves.append(cve)

    # Sort unique CVEs by CVSS score descending
    unique_cves.sort(key=lambda c: c.get("cvss_score", 0.0), reverse=True)

    # Compute severity counts
    critical = sum(1 for c in unique_cves if c.get("cvss_severity", "").upper() == "CRITICAL")
    high = sum(1 for c in unique_cves if c.get("cvss_severity", "").upper() == "HIGH")
    medium = sum(1 for c in unique_cves if c.get("cvss_severity", "").upper() == "MEDIUM")
    low = sum(
        1 for c in unique_cves if c.get("cvss_severity", "").upper() in ("LOW", "") and c.get("cvss_score", 0.0) > 0.0
    )

    return {
        "keyword": keyword,
        "version": version,
        "cpe_type": cpe_type,
        "cpe_results": cpe_result,
        "cve_results": all_cve_results,
        "unique_cves": unique_cves,
        "summary": {
            "cpes_found": len(cpe_result["cpes"]),
            "total_unique_cves": len(unique_cves),
            "critical_count": critical,
            "high_count": high,
            "medium_count": medium,
            "low_count": low,
        },
    }


# ---------------------------------------------------------------------------
# Strands tool handler
# ---------------------------------------------------------------------------


def search_cpe_handler(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands-compatible handler for the search_cpe tool."""
    tool_input = tool.get("input", {})
    keyword = tool_input.get("keyword", "").strip()

    if not keyword:
        return {
            "status": "error",
            "content": [{"text": "Error: 'keyword' is required."}],
        }

    version = tool_input.get("version", "").strip()
    cpe_type = tool_input.get("cpe_type", "a").strip().lower()
    fetch_cves_flag = tool_input.get("fetch_cves", True)
    max_cpes = tool_input.get("max_cpes", 10)
    max_cves_per_cpe = tool_input.get("max_cves_per_cpe", 20)

    result = search_cpe_and_cves(
        keyword,
        version=version,
        cpe_type=cpe_type,
        fetch_cves=fetch_cves_flag,
        max_cpes=max_cpes,
        max_cves_per_cpe=max_cves_per_cpe,
    )

    text = json.dumps(result, indent=2, default=str)
    log_tool_output_size("search_cpe", text)

    return {
        "status": "success" if not result.get("error") else "error",
        "content": [{"text": text}],
    }
