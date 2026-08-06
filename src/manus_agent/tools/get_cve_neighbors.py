"""Tool: get_cve_neighbors

Given a CVE identifier, discovers other CVEs that affect the **same product
or package** — the "neighborhood" of vulnerabilities you should patch together.

Strategy:
1. Fetch the target CVE from NVD to extract CPE (Common Platform Enumeration)
   match criteria — specifically the vendor:product pair.
2. Query NVD for other CVEs matching the same CPE product keyword.
3. Enrich each neighbor with EPSS score for prioritisation.
4. Return a ranked list (highest EPSS first) with CVSS, publication date,
   and description snippet.

This is useful for batch-patching: if you're already fixing one CVE in a
library, you should fix its neighbors at the same time.

All HTTP calls are mockable — no side effects in unit tests.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

__all__ = ["get_cve_neighbors", "TOOL_SPEC"]

TOOL_SPEC = {
    "name": "get_cve_neighbors",
    "description": (
        "Given a CVE, finds other CVEs affecting the same product or package. "
        "Useful for batch patching — if you are fixing one vulnerability in a library, "
        "you should fix its neighbors at the same time. Returns a prioritised list "
        "of neighboring CVEs ranked by EPSS score, with CVSS severity, publication "
        "date, and description for each."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier to find neighbors for (e.g., 'CVE-2021-44228').",
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        "Maximum number of neighbor CVEs to return. Defaults to 10. "
                        "Maximum allowed is 50."
                    ),
                    "default": 10,
                },
            },
            "required": ["cve_id"],
        }
    },
}

# ---------------------------------------------------------------------------
# NVD retry/back-off (mirrors get_nvd_data conventions)
# ---------------------------------------------------------------------------
_NVD_MAX_RETRIES = int(os.environ.get("NVD_MAX_RETRIES", "3"))
_NVD_RETRY_BASE_DELAY = float(os.environ.get("NVD_RETRY_BASE_DELAY", "2"))
_NVD_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _build_nvd_headers() -> dict[str, str]:
    """Return request headers, injecting NVD_API_KEY when available."""
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY", "").strip()
    if api_key:
        headers["apiKey"] = api_key
    return headers


def _nvd_get(url: str, *, timeout: int = 15) -> requests.Response:
    """GET *url* with exponential back-off retry on 429/transient errors."""
    headers = _build_nvd_headers()
    last_exc: Exception | None = None

    for attempt in range(_NVD_MAX_RETRIES + 1):
        if attempt > 0:
            delay = _NVD_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            time.sleep(delay)
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code in _NVD_RETRYABLE_STATUS:
                last_exc = requests.exceptions.HTTPError(
                    f"HTTP {response.status_code}", response=response
                )
                if attempt < _NVD_MAX_RETRIES:
                    continue
                raise last_exc
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as exc:
            if hasattr(exc, "response") and exc.response is not None:
                if exc.response.status_code in _NVD_RETRYABLE_STATUS and attempt < _NVD_MAX_RETRIES:
                    last_exc = exc
                    continue
            raise
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            if attempt < _NVD_MAX_RETRIES:
                continue
            raise

    # Should not reach here, but satisfy type checkers
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------


def _extract_cpe_products(cve_record: dict[str, Any]) -> list[dict[str, str]]:
    """Extract unique vendor:product pairs from NVD CPE match criteria.

    Returns a list of dicts like:
        [{"vendor": "apache", "product": "log4j"}, ...]
    """
    products: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    configurations = cve_record.get("configurations", [])
    for config in configurations:
        for node in config.get("nodes", []):
            for cpe_match in node.get("cpeMatch", []):
                criteria = cpe_match.get("criteria", "")
                # CPE 2.3 format: cpe:2.3:part:vendor:product:version:...
                parts = criteria.split(":")
                if len(parts) >= 5:
                    vendor = parts[3]
                    product = parts[4]
                    if vendor and product and (vendor, product) not in seen:
                        seen.add((vendor, product))
                        products.append({"vendor": vendor, "product": product})

    return products


def _extract_cvss(cve_item: dict[str, Any]) -> dict[str, Any]:
    """Extract the best CVSS score/severity from a CVE metrics block."""
    metrics = cve_item.get("metrics", {})

    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key, [])
        if entries:
            cv = entries[0].get("cvssData", {})
            return {
                "score": cv.get("baseScore"),
                "severity": cv.get("baseSeverity"),
                "version": cv.get("version", key.replace("cvssMetric", "").replace("V", "v")),
            }

    entries = metrics.get("cvssMetricV2", [])
    if entries:
        cv = entries[0].get("cvssData", {})
        return {
            "score": cv.get("baseScore"),
            "severity": cv.get("baseSeverity", "N/A"),
            "version": "2.0",
        }

    return {"score": None, "severity": "UNKNOWN", "version": None}


def _extract_description(cve_item: dict[str, Any]) -> str:
    """Extract the English description from a CVE record."""
    descriptions = cve_item.get("descriptions", [])
    for desc in descriptions:
        if desc.get("lang") == "en":
            text = desc.get("value", "")
            # Truncate long descriptions
            if len(text) > 200:
                return text[:197] + "..."
            return text
    return "No description available."


def _fetch_epss_scores(cve_ids: list[str]) -> dict[str, dict[str, float]]:
    """Fetch EPSS scores for a batch of CVE IDs.

    Returns a dict mapping CVE-ID → {"epss": float, "percentile": float}.
    Gracefully returns empty dict on failure.
    """
    if not cve_ids:
        return {}

    # FIRST.org EPSS API supports up to 100 CVE IDs in a single request
    url = f"https://api.first.org/data/v1/epss?cve={','.join(cve_ids[:100])}"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
        return {
            item["cve"]: {
                "epss": float(item.get("epss", 0)),
                "percentile": float(item.get("percentile", 0)),
            }
            for item in data.get("data", [])
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main tool function
# ---------------------------------------------------------------------------


def get_cve_neighbors(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Find CVEs that affect the same product as the given CVE."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id", "").strip().upper()
    max_results = min(int(tool_input.get("max_results", 10)), 50)

    if not cve_id or not cve_id.startswith("CVE-"):
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid CVE ID format. Must be a string like 'CVE-YYYY-NNNN'."}],
        }
        log_tool_output_size("get_cve_neighbors", result)
        return result

    # Step 1: Fetch the target CVE from NVD to get CPE products
    try:
        nvd_url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
        resp = _nvd_get(nvd_url)
        data = resp.json()
    except requests.exceptions.RequestException as exc:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Failed to fetch NVD data for {cve_id}: {exc}"}],
        }
        log_tool_output_size("get_cve_neighbors", result)
        return result

    vulnerabilities = data.get("vulnerabilities", [])
    if not vulnerabilities:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"No NVD record found for {cve_id}."}],
        }
        log_tool_output_size("get_cve_neighbors", result)
        return result

    target_cve = vulnerabilities[0].get("cve", {})
    products = _extract_cpe_products(target_cve)

    if not products:
        result = {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [
                {
                    "text": (
                        f"No CPE product data found for {cve_id} in NVD. "
                        "Cannot determine affected product to search for neighbors."
                    )
                }
            ],
        }
        log_tool_output_size("get_cve_neighbors", result)
        return result

    # Step 2: Search NVD for other CVEs with the same product keyword
    # Use the first (most specific) product for the search
    primary_product = products[0]
    keyword = primary_product["product"]
    vendor = primary_product["vendor"]

    try:
        # Use cpeName-based search for precision: keywordSearch is broad but
        # covers cases where CPE configs aren't perfectly structured
        search_url = (
            f"https://services.nvd.nist.gov/rest/json/cves/2.0"
            f"?keywordSearch={vendor}+{keyword}&resultsPerPage=50"
        )
        resp = _nvd_get(search_url)
        search_data = resp.json()
    except requests.exceptions.RequestException as exc:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Failed to search NVD for neighbors: {exc}"}],
        }
        log_tool_output_size("get_cve_neighbors", result)
        return result

    neighbor_vulns = search_data.get("vulnerabilities", [])

    # Filter out the target CVE itself and collect neighbor data
    neighbors: list[dict[str, Any]] = []
    neighbor_cve_ids: list[str] = []

    for vuln in neighbor_vulns:
        cve_item = vuln.get("cve", {})
        neighbor_id = cve_item.get("id", "")
        if neighbor_id == cve_id:
            continue  # skip the input CVE itself

        cvss = _extract_cvss(cve_item)
        description = _extract_description(cve_item)
        published = cve_item.get("published", "")[:10]  # YYYY-MM-DD

        neighbors.append(
            {
                "cve_id": neighbor_id,
                "published": published,
                "cvss_score": cvss["score"],
                "cvss_severity": cvss["severity"],
                "description": description,
                "epss": 0.0,
                "epss_percentile": 0.0,
            }
        )
        neighbor_cve_ids.append(neighbor_id)

    # Step 3: Enrich with EPSS scores for prioritisation
    if neighbor_cve_ids:
        epss_data = _fetch_epss_scores(neighbor_cve_ids)
        for n in neighbors:
            epss_entry = epss_data.get(n["cve_id"], {})
            n["epss"] = epss_entry.get("epss", 0.0)
            n["epss_percentile"] = epss_entry.get("percentile", 0.0)

    # Step 4: Sort by EPSS (descending) then CVSS (descending)
    neighbors.sort(
        key=lambda x: (x["epss"], x["cvss_score"] or 0),
        reverse=True,
    )

    # Limit results
    neighbors = neighbors[:max_results]

    # Build summary
    product_label = f"{vendor}:{keyword}"
    if not neighbors:
        summary = (
            f"No neighboring CVEs found for product '{product_label}' "
            f"(searched from {cve_id})."
        )
        result = {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": summary}],
        }
        log_tool_output_size("get_cve_neighbors", result)
        return result

    summary_lines = [
        f"Found {len(neighbors)} neighboring CVE(s) for product '{product_label}' (from {cve_id}):",
        f"  Target product: {product_label}",
        f"  Ranked by EPSS exploitation probability (highest first).",
        "",
    ]

    for i, n in enumerate(neighbors, 1):
        epss_str = f"{n['epss']:.4f}" if n["epss"] else "N/A"
        cvss_str = f"{n['cvss_score']:.1f}" if n["cvss_score"] else "N/A"
        summary_lines.append(
            f"  {i:2d}. {n['cve_id']}  "
            f"EPSS={epss_str}  CVSS={cvss_str} ({n['cvss_severity']})  "
            f"Published={n['published']}"
        )
        summary_lines.append(f"      {n['description']}")
        summary_lines.append("")

    summary = "\n".join(summary_lines)

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [
            {"text": summary},
            {
                "json": {
                    "cve_id": cve_id,
                    "product": product_label,
                    "affected_products": products,
                    "neighbor_count": len(neighbors),
                    "neighbors": neighbors,
                }
            },
        ],
    }
    log_tool_output_size("get_cve_neighbors", result)
    return result
