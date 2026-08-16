#!/usr/bin/env python3
"""
CVE variant clustering tool.

Groups CVEs related to an input CVE across three cluster dimensions:
1. Same component/vendor (via NVD CPE configurations)
2. Same CWE weakness class (via NVD weakness data)
3. Same researcher/disclosure domain (via NVD reference URLs)

Useful for finding the full attack surface when one CVE is confirmed exploited.
"""

import json
import os
import re
import time
from typing import Any
from urllib.parse import urlparse

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

TOOL_SPEC = {
    "name": "cluster_variants",
    "description": (
        "Groups CVEs related to an input CVE across three cluster dimensions: "
        "same component/vendor, same CWE weakness class, and same researcher/"
        "disclosure domain. Useful for finding the full attack surface when one "
        "CVE is confirmed exploited."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier to cluster around (e.g., 'CVE-2021-44228').",
                }
            },
            "required": ["cve_id"],
        }
    },
}

# ---------------------------------------------------------------------------
# Configuration (overridable via env for tests)
# ---------------------------------------------------------------------------
_MAX_RETRIES = int(os.environ.get("NVD_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY = float(os.environ.get("NVD_RETRY_BASE_DELAY", "2"))
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_CLUSTER_RESULTS = int(os.environ.get("CLUSTER_MAX_RESULTS", "20"))
_NVD_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# Reference domains to ignore (too generic to be useful for clustering)
_IGNORED_DOMAINS = frozenset(
    {
        "nvd.nist.gov",
        "cve.mitre.org",
        "cve.org",
        "www.cve.org",
        "github.com",
        "web.nvd.nist.gov",
        "lists.apache.org",
        "lists.debian.org",
        "lists.fedoraproject.org",
        "security-tracker.debian.org",
        "bugzilla.redhat.com",
        "access.redhat.com",
        "ubuntu.com",
        "www.ubuntu.com",
        "usn.ubuntu.com",
    }
)

# CVE ID regex
_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# HTTP helper with retry/back-off
# ---------------------------------------------------------------------------


def _nvd_get(url: str, params: dict | None = None) -> requests.Response:
    """GET with retry/back-off and optional NVD API key."""
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY", "").strip()
    if api_key:
        headers["apiKey"] = api_key

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            time.sleep(delay)
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code in _RETRYABLE_STATUS:
                last_exc = requests.exceptions.HTTPError(response=resp)
                if attempt < _MAX_RETRIES:
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code not in _RETRYABLE_STATUS:
                raise
            last_exc = exc
            if attempt < _MAX_RETRIES:
                continue
            raise
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                continue
            raise
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# NVD data extraction helpers
# ---------------------------------------------------------------------------


def _extract_cpe_vendors_products(cve_data: dict) -> list[tuple[str, str]]:
    """Extract (vendor, product) pairs from NVD CPE configurations."""
    pairs: list[tuple[str, str]] = []
    configurations = cve_data.get("configurations", [])
    for config in configurations:
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                criteria = match.get("criteria", "")
                # CPE 2.3: cpe:2.3:part:vendor:product:version:...
                parts = criteria.split(":")
                if len(parts) >= 5:
                    vendor = parts[3]
                    product = parts[4]
                    if vendor != "*" and product != "*":
                        pairs.append((vendor, product))
    # Deduplicate preserving order
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)
    return unique


def _extract_cwes(cve_data: dict) -> list[str]:
    """Extract CWE IDs from NVD weakness data."""
    cwes: list[str] = []
    weaknesses = cve_data.get("weaknesses", [])
    for weakness in weaknesses:
        for desc in weakness.get("description", []):
            value = desc.get("value", "")
            if value.upper().startswith("CWE-") and value.upper() != "CWE-NOINFO":
                cwes.append(value.upper())
    return list(dict.fromkeys(cwes))  # deduplicate preserving order


def _extract_reference_domains(cve_data: dict) -> list[str]:
    """Extract meaningful reference domains from NVD references."""
    domains: list[str] = []
    references = cve_data.get("references", [])
    for ref in references:
        url = ref.get("url", "")
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Strip www. prefix for normalisation
            if domain.startswith("www."):
                domain = domain[4:]
            if domain and domain not in _IGNORED_DOMAINS:
                domains.append(domain)
        except Exception:
            continue
    return list(dict.fromkeys(domains))  # deduplicate preserving order


# ---------------------------------------------------------------------------
# Cluster search functions
# ---------------------------------------------------------------------------


def _search_by_cpe(vendor: str, product: str, exclude_cve: str) -> list[dict]:
    """Search NVD for CVEs affecting the same vendor:product."""
    # NVD API v2 cpeName filter: cpe:2.3:*:vendor:product:*:*:*:*:*:*:*
    cpe_name = f"cpe:2.3:*:{vendor}:{product}:*:*:*:*:*:*:*"
    params = {"cpeName": cpe_name, "resultsPerPage": str(_MAX_CLUSTER_RESULTS + 5)}
    try:
        resp = _nvd_get(_NVD_BASE_URL, params=params)
        data = resp.json()
        results: list[dict] = []
        for vuln in data.get("vulnerabilities", []):
            cve = vuln.get("cve", {})
            cve_id = cve.get("id", "")
            if cve_id.upper() == exclude_cve.upper():
                continue
            results.append(_summarize_cve(cve))
            if len(results) >= _MAX_CLUSTER_RESULTS:
                break
        return results
    except Exception:
        return []


def _search_by_cwe(cwe_id: str, exclude_cve: str) -> list[dict]:
    """Search NVD for CVEs with the same CWE weakness."""
    params = {"cweId": cwe_id, "resultsPerPage": str(_MAX_CLUSTER_RESULTS + 5)}
    try:
        resp = _nvd_get(_NVD_BASE_URL, params=params)
        data = resp.json()
        results: list[dict] = []
        for vuln in data.get("vulnerabilities", []):
            cve = vuln.get("cve", {})
            cve_id_found = cve.get("id", "")
            if cve_id_found.upper() == exclude_cve.upper():
                continue
            results.append(_summarize_cve(cve))
            if len(results) >= _MAX_CLUSTER_RESULTS:
                break
        return results
    except Exception:
        return []


def _search_by_reference_domain(domain: str, exclude_cve: str) -> list[dict]:
    """
    Search NVD for CVEs sharing a reference domain.

    NVD API doesn't directly support domain search, so we use keyword search
    with the domain as a keyword in the description/references.
    """
    params = {"keywordSearch": domain, "resultsPerPage": str(_MAX_CLUSTER_RESULTS + 5)}
    try:
        resp = _nvd_get(_NVD_BASE_URL, params=params)
        data = resp.json()
        results: list[dict] = []
        for vuln in data.get("vulnerabilities", []):
            cve = vuln.get("cve", {})
            cve_id_found = cve.get("id", "")
            if cve_id_found.upper() == exclude_cve.upper():
                continue
            # Verify the domain actually appears in references
            refs = cve.get("references", [])
            domain_match = False
            for ref in refs:
                ref_url = ref.get("url", "")
                try:
                    parsed = urlparse(ref_url)
                    ref_domain = parsed.netloc.lower()
                    if ref_domain.startswith("www."):
                        ref_domain = ref_domain[4:]
                    if ref_domain == domain:
                        domain_match = True
                        break
                except Exception:
                    continue
            if domain_match:
                results.append(_summarize_cve(cve))
                if len(results) >= _MAX_CLUSTER_RESULTS:
                    break
        return results
    except Exception:
        return []


def _summarize_cve(cve: dict) -> dict:
    """Create a compact summary of a CVE from NVD data."""
    cve_id = cve.get("id", "unknown")
    descriptions = cve.get("descriptions", [])
    desc = ""
    for d in descriptions:
        if d.get("lang") == "en":
            desc = d.get("value", "")
            break
    if not desc and descriptions:
        desc = descriptions[0].get("value", "")
    # Truncate long descriptions
    if len(desc) > 200:
        desc = desc[:197] + "..."

    # Extract CVSS score
    cvss_score = None
    metrics = cve.get("metrics", {})
    for version_key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        metric_list = metrics.get(version_key, [])
        if metric_list:
            cvss_data = metric_list[0].get("cvssData", {})
            cvss_score = cvss_data.get("baseScore")
            break

    # Published date
    published = cve.get("published", "")[:10]  # YYYY-MM-DD

    return {
        "cve_id": cve_id,
        "description": desc,
        "cvss_score": cvss_score,
        "published": published,
    }


# ---------------------------------------------------------------------------
# Main clustering logic
# ---------------------------------------------------------------------------


def _build_clusters(
    cve_id: str,
    cve_data: dict,
) -> dict:
    """Build variant clusters from the seed CVE's metadata."""
    vendor_products = _extract_cpe_vendors_products(cve_data)
    cwes = _extract_cwes(cve_data)
    ref_domains = _extract_reference_domains(cve_data)

    clusters: dict[str, Any] = {
        "seed_cve": cve_id.upper(),
        "dimensions": {
            "component": {
                "search_keys": [f"{v}:{p}" for v, p in vendor_products],
                "variants": [],
            },
            "weakness": {
                "search_keys": cwes,
                "variants": [],
            },
            "researcher": {
                "search_keys": ref_domains[:5],  # Cap to avoid excessive queries
                "variants": [],
            },
        },
        "total_unique_variants": 0,
    }

    seen_cves: set[str] = {cve_id.upper()}

    # Dimension 1: Same component/vendor (use first vendor:product, cap at 2)
    for vendor, product in vendor_products[:2]:
        results = _search_by_cpe(vendor, product, cve_id)
        for r in results:
            if r["cve_id"].upper() not in seen_cves:
                seen_cves.add(r["cve_id"].upper())
                clusters["dimensions"]["component"]["variants"].append(r)

    # Dimension 2: Same CWE weakness class (use first CWE, cap at 2)
    for cwe in cwes[:2]:
        results = _search_by_cwe(cwe, cve_id)
        for r in results:
            if r["cve_id"].upper() not in seen_cves:
                seen_cves.add(r["cve_id"].upper())
                clusters["dimensions"]["weakness"]["variants"].append(r)

    # Dimension 3: Same researcher/disclosure domain (use first domain, cap at 2)
    for domain in ref_domains[:2]:
        results = _search_by_reference_domain(domain, cve_id)
        for r in results:
            if r["cve_id"].upper() not in seen_cves:
                seen_cves.add(r["cve_id"].upper())
                clusters["dimensions"]["researcher"]["variants"].append(r)

    # Count total unique variants (excluding seed)
    clusters["total_unique_variants"] = (
        len(clusters["dimensions"]["component"]["variants"])
        + len(clusters["dimensions"]["weakness"]["variants"])
        + len(clusters["dimensions"]["researcher"]["variants"])
    )

    return clusters


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------


def _format_text(clusters: dict) -> str:
    """Format clusters as human-readable text."""
    lines: list[str] = []
    lines.append(f"CVE Variant Clusters for {clusters['seed_cve']}")
    lines.append("=" * 60)
    lines.append("")

    for dim_name, dim_data in clusters["dimensions"].items():
        label = {
            "component": "Same Component/Vendor",
            "weakness": "Same CWE Weakness Class",
            "researcher": "Same Researcher/Disclosure Domain",
        }.get(dim_name, dim_name)

        lines.append(f"── {label} ──")
        search_keys = dim_data.get("search_keys", [])
        if search_keys:
            lines.append(f"   Search keys: {', '.join(search_keys)}")

        variants = dim_data.get("variants", [])
        if not variants:
            lines.append("   (no variants found)")
        else:
            lines.append(f"   Found {len(variants)} variant(s):")
            for v in variants:
                cvss_str = f" [CVSS {v['cvss_score']}]" if v.get("cvss_score") else ""
                date_str = f" ({v['published']})" if v.get("published") else ""
                lines.append(f"   • {v['cve_id']}{cvss_str}{date_str}")
                if v.get("description"):
                    lines.append(f"     {v['description']}")
        lines.append("")

    lines.append(f"Total unique variants: {clusters['total_unique_variants']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strands tool handler
# ---------------------------------------------------------------------------


def cluster_variants(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands tool entry point for cluster_variants."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id", "")

    if not isinstance(cve_id, str) or not _CVE_RE.match(cve_id.strip()):
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid CVE ID format. Must be like 'CVE-2021-44228'."}],
        }
        log_tool_output_size("cluster_variants", result)
        return result

    cve_id = cve_id.strip().upper()

    # Fetch seed CVE data from NVD
    try:
        resp = _nvd_get(_NVD_BASE_URL, params={"cveId": cve_id})
        data = resp.json()
    except requests.exceptions.RequestException as exc:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Failed to fetch CVE data from NVD: {exc}"}],
        }
        log_tool_output_size("cluster_variants", result)
        return result
    except (json.JSONDecodeError, ValueError):
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Failed to parse NVD response."}],
        }
        log_tool_output_size("cluster_variants", result)
        return result

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"No vulnerability data found for {cve_id}."}],
        }
        log_tool_output_size("cluster_variants", result)
        return result

    cve_data = vulns[0].get("cve", {})
    clusters = _build_clusters(cve_id, cve_data)

    text_output = _format_text(clusters)
    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": text_output}, {"json": clusters}],
    }
    log_tool_output_size("cluster_variants", result)
    return result
