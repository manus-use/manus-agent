#!/usr/bin/env python3
"""
CVE variant clustering tool.

Groups CVEs related to an input CVE across three cluster dimensions:
1. Same component/vendor (CPE match)
2. Same CWE weakness class
3. Same researcher/disclosure source

Useful for finding the full attack surface when one CVE is confirmed exploited.
"""

from __future__ import annotations

import re
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.get_nvd_data import _nvd_get_with_retry
from manus_agent.tools.tool_output_logger import log_tool_output_size

TOOL_SPEC = {
    "name": "cluster_variants",
    "description": (
        "Groups CVEs related to an input CVE across three cluster dimensions: "
        "same component/vendor (CPE match), same CWE weakness class, and same "
        "researcher/disclosure source. Useful for finding the full attack surface "
        "when one CVE is confirmed exploited."
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
# Constants
# ---------------------------------------------------------------------------
_NVD_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_MAX_RESULTS_PER_CLUSTER = 20
_REQUEST_TIMEOUT = 15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_cpe_info(vuln_data: dict) -> list[dict[str, str]]:
    """Extract vendor/product pairs from CPE configurations."""
    cpe_pairs: list[dict[str, str]] = []
    configurations = vuln_data.get("cve", {}).get("configurations", [])
    for config in configurations:
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                cpe_uri = match.get("criteria", "")
                # CPE 2.3 format: cpe:2.3:part:vendor:product:version:...
                parts = cpe_uri.split(":")
                if len(parts) >= 5:
                    vendor = parts[3]
                    product = parts[4]
                    if vendor != "*" and product != "*":
                        pair = {"vendor": vendor, "product": product}
                        if pair not in cpe_pairs:
                            cpe_pairs.append(pair)
    return cpe_pairs


def _extract_cwes(vuln_data: dict) -> list[str]:
    """Extract CWE IDs from the vulnerability data."""
    cwes: list[str] = []
    weaknesses = vuln_data.get("cve", {}).get("weaknesses", [])
    for weakness in weaknesses:
        for desc in weakness.get("description", []):
            cwe_id = desc.get("value", "")
            if cwe_id.startswith("CWE-") and cwe_id != "CWE-noinfo" and cwe_id not in cwes:
                cwes.append(cwe_id)
    return cwes


def _extract_sources(vuln_data: dict) -> list[str]:
    """Extract researcher/disclosure source organizations from references."""
    sources: list[str] = []
    references = vuln_data.get("cve", {}).get("references", [])
    for ref in references:
        source = ref.get("source", "")
        if source and source not in sources:
            sources.append(source)
    return sources


def _extract_source_domains(vuln_data: dict) -> list[str]:
    """Extract unique domains from reference URLs as disclosure sources."""
    domains: list[str] = []
    references = vuln_data.get("cve", {}).get("references", [])
    for ref in references:
        url = ref.get("url", "")
        match = re.match(r"https?://([^/]+)", url)
        if match:
            domain = match.group(1).lower()
            # Skip generic domains
            if (
                domain
                not in (
                    "nvd.nist.gov",
                    "cve.org",
                    "cve.mitre.org",
                    "www.cve.org",
                    "web.nvd.nist.gov",
                )
                and domain not in domains
            ):
                domains.append(domain)
    return domains


def _extract_cvss_score(vuln_data: dict) -> float | None:
    """Extract the best available CVSS base score."""
    metrics = vuln_data.get("cve", {}).get("metrics", {})
    # Try CVSS 3.1 first, then 3.0, then 2.0
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        metric_list = metrics.get(key, [])
        if metric_list:
            return metric_list[0].get("cvssData", {}).get("baseScore")
    return None


def _extract_description(vuln_data: dict) -> str:
    """Extract the English description."""
    descriptions = vuln_data.get("cve", {}).get("descriptions", [])
    for desc in descriptions:
        if desc.get("lang") == "en":
            return desc.get("value", "")
    if descriptions:
        return descriptions[0].get("value", "")
    return ""


def _fetch_cves_by_cpe(vendor: str, product: str, exclude_cve: str) -> list[dict]:
    """Fetch CVEs affecting the same vendor/product via NVD keyword search."""
    # Use cpeName parameter for precise matching
    cpe_name = f"cpe:2.3:*:{vendor}:{product}:*:*:*:*:*:*:*:*"
    url = f"{_NVD_BASE}?cpeName={cpe_name}&resultsPerPage={_MAX_RESULTS_PER_CLUSTER}"
    try:
        resp = _nvd_get_with_retry(url, timeout=_REQUEST_TIMEOUT)
        data = resp.json()
        results = []
        for vuln in data.get("vulnerabilities", []):
            cve_id = vuln.get("cve", {}).get("id", "")
            if cve_id and cve_id.upper() != exclude_cve.upper():
                results.append(vuln)
        return results[:_MAX_RESULTS_PER_CLUSTER]
    except (requests.exceptions.RequestException, ValueError):
        return []


def _fetch_cves_by_keyword(keyword: str, exclude_cve: str) -> list[dict]:
    """Fetch CVEs by keyword search (fallback for component matching)."""
    url = f"{_NVD_BASE}?keywordSearch={keyword}&resultsPerPage={_MAX_RESULTS_PER_CLUSTER}"
    try:
        resp = _nvd_get_with_retry(url, timeout=_REQUEST_TIMEOUT)
        data = resp.json()
        results = []
        for vuln in data.get("vulnerabilities", []):
            cve_id = vuln.get("cve", {}).get("id", "")
            if cve_id and cve_id.upper() != exclude_cve.upper():
                results.append(vuln)
        return results[:_MAX_RESULTS_PER_CLUSTER]
    except (requests.exceptions.RequestException, ValueError):
        return []


def _fetch_cves_by_cwe(cwe_id: str, exclude_cve: str) -> list[dict]:
    """Fetch CVEs sharing the same CWE weakness."""
    url = f"{_NVD_BASE}?cweId={cwe_id}&resultsPerPage={_MAX_RESULTS_PER_CLUSTER}"
    try:
        resp = _nvd_get_with_retry(url, timeout=_REQUEST_TIMEOUT)
        data = resp.json()
        results = []
        for vuln in data.get("vulnerabilities", []):
            cve_id_str = vuln.get("cve", {}).get("id", "")
            if cve_id_str and cve_id_str.upper() != exclude_cve.upper():
                results.append(vuln)
        return results[:_MAX_RESULTS_PER_CLUSTER]
    except (requests.exceptions.RequestException, ValueError):
        return []


def _fetch_cves_by_source(source_domain: str, exclude_cve: str) -> list[dict]:
    """Fetch CVEs sharing the same disclosure source via keyword search on source domain."""
    url = f"{_NVD_BASE}?sourceIdentifier={source_domain}&resultsPerPage={_MAX_RESULTS_PER_CLUSTER}"
    try:
        resp = _nvd_get_with_retry(url, timeout=_REQUEST_TIMEOUT)
        data = resp.json()
        results = []
        for vuln in data.get("vulnerabilities", []):
            cve_id_str = vuln.get("cve", {}).get("id", "")
            if cve_id_str and cve_id_str.upper() != exclude_cve.upper():
                results.append(vuln)
        return results[:_MAX_RESULTS_PER_CLUSTER]
    except (requests.exceptions.RequestException, ValueError):
        return []


def _summarize_cve(vuln: dict) -> dict[str, Any]:
    """Create a compact summary of a CVE for cluster output."""
    cve_data = vuln.get("cve", {})
    cve_id = cve_data.get("id", "unknown")
    desc = _extract_description(vuln)
    score = _extract_cvss_score(vuln)
    cwes = _extract_cwes(vuln)
    published = cve_data.get("published", "")[:10]  # YYYY-MM-DD

    return {
        "cve_id": cve_id,
        "description": desc[:200] + ("..." if len(desc) > 200 else ""),
        "cvss_score": score,
        "cwes": cwes,
        "published": published,
    }


# ---------------------------------------------------------------------------
# Main clustering logic
# ---------------------------------------------------------------------------


def cluster_variants(cve_id: str) -> dict[str, Any]:
    """
    Cluster CVEs related to the given CVE across three dimensions.

    Returns a dict with:
      - input_cve: info about the queried CVE
      - clusters: dict with component, cwe, source keys
      - summary: counts per cluster
    """
    cve_id = cve_id.strip().upper()

    # Fetch the seed CVE
    url = f"{_NVD_BASE}?cveId={cve_id}"
    try:
        resp = _nvd_get_with_retry(url, timeout=_REQUEST_TIMEOUT)
        seed_data = resp.json()
    except requests.exceptions.RequestException as exc:
        return {"error": f"Failed to fetch NVD data for {cve_id}: {exc}"}

    vulns = seed_data.get("vulnerabilities", [])
    if not vulns:
        return {"error": f"No vulnerability data found for {cve_id}"}

    seed_vuln = vulns[0]
    cpe_pairs = _extract_cpe_info(seed_vuln)
    cwes = _extract_cwes(seed_vuln)
    sources = _extract_sources(seed_vuln)

    # Seed CVE summary
    input_cve_info = {
        "cve_id": cve_id,
        "description": _extract_description(seed_vuln),
        "cvss_score": _extract_cvss_score(seed_vuln),
        "cwes": cwes,
        "cpe_vendors": [f"{p['vendor']}/{p['product']}" for p in cpe_pairs],
        "sources": sources,
    }

    # --- Cluster 1: Same Component/Vendor ---
    component_cluster: list[dict] = []
    seen_cves: set[str] = set()
    for pair in cpe_pairs[:3]:  # Limit API calls
        related = _fetch_cves_by_cpe(pair["vendor"], pair["product"], cve_id)
        for vuln in related:
            vid = vuln.get("cve", {}).get("id", "")
            if vid not in seen_cves:
                seen_cves.add(vid)
                component_cluster.append(_summarize_cve(vuln))
    # Fallback: keyword search on product name if no CPE hits
    if not component_cluster and cpe_pairs:
        product_name = cpe_pairs[0]["product"].replace("_", " ")
        related = _fetch_cves_by_keyword(product_name, cve_id)
        for vuln in related:
            vid = vuln.get("cve", {}).get("id", "")
            if vid not in seen_cves:
                seen_cves.add(vid)
                component_cluster.append(_summarize_cve(vuln))

    # --- Cluster 2: Same CWE Weakness Class ---
    cwe_cluster: list[dict] = []
    seen_cves_cwe: set[str] = set()
    for cwe in cwes[:2]:  # Limit API calls
        related = _fetch_cves_by_cwe(cwe, cve_id)
        for vuln in related:
            vid = vuln.get("cve", {}).get("id", "")
            if vid not in seen_cves_cwe:
                seen_cves_cwe.add(vid)
                cwe_cluster.append(_summarize_cve(vuln))

    # --- Cluster 3: Same Source/Researcher ---
    source_cluster: list[dict] = []
    seen_cves_src: set[str] = set()
    # Use sourceIdentifier (the reporting CNA email/org)
    for source in sources[:2]:  # Limit API calls
        related = _fetch_cves_by_source(source, cve_id)
        for vuln in related:
            vid = vuln.get("cve", {}).get("id", "")
            if vid not in seen_cves_src:
                seen_cves_src.add(vid)
                source_cluster.append(_summarize_cve(vuln))

    clusters = {
        "component": {
            "dimension": "Same Component/Vendor",
            "criteria": [f"{p['vendor']}/{p['product']}" for p in cpe_pairs[:3]],
            "cves": component_cluster[:_MAX_RESULTS_PER_CLUSTER],
        },
        "cwe": {
            "dimension": "Same CWE Weakness Class",
            "criteria": cwes[:2],
            "cves": cwe_cluster[:_MAX_RESULTS_PER_CLUSTER],
        },
        "source": {
            "dimension": "Same Researcher/Disclosure Source",
            "criteria": sources[:2],
            "cves": source_cluster[:_MAX_RESULTS_PER_CLUSTER],
        },
    }

    summary = {
        "component_count": len(clusters["component"]["cves"]),
        "cwe_count": len(clusters["cwe"]["cves"]),
        "source_count": len(clusters["source"]["cves"]),
        "total_unique": len({c["cve_id"] for cluster in clusters.values() for c in cluster["cves"]}),
    }

    return {
        "input_cve": input_cve_info,
        "clusters": clusters,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _render_text(result: dict[str, Any]) -> str:
    """Render cluster result as human-readable text."""
    if "error" in result:
        return f"❌ {result['error']}"

    lines: list[str] = []
    input_cve = result["input_cve"]
    lines.append(f"╔══ CVE Variant Clustering: {input_cve['cve_id']} ══╗")
    lines.append("")

    if input_cve.get("description"):
        desc = input_cve["description"]
        if len(desc) > 120:
            desc = desc[:117] + "..."
        lines.append(f"  {desc}")
        lines.append("")

    if input_cve.get("cvss_score"):
        lines.append(f"  CVSS: {input_cve['cvss_score']}")
    if input_cve.get("cwes"):
        lines.append(f"  CWEs: {', '.join(input_cve['cwes'])}")
    if input_cve.get("cpe_vendors"):
        lines.append(f"  Components: {', '.join(input_cve['cpe_vendors'][:5])}")
    if input_cve.get("sources"):
        lines.append(f"  Sources: {', '.join(input_cve['sources'][:3])}")
    lines.append("")

    summary = result["summary"]
    lines.append(f"  📊 Found {summary['total_unique']} unique related CVEs across 3 dimensions")
    lines.append("")

    clusters = result["clusters"]

    # Component cluster
    comp = clusters["component"]
    lines.append(f"┌── Cluster 1: {comp['dimension']} ({len(comp['cves'])} CVEs) ──┐")
    if comp["criteria"]:
        lines.append(f"  Criteria: {', '.join(comp['criteria'])}")
    lines.append("")
    for cve in comp["cves"][:10]:
        score_str = f"CVSS {cve['cvss_score']}" if cve.get("cvss_score") else "no score"
        lines.append(f"  • {cve['cve_id']} [{score_str}] {cve.get('published', '')}")
        if cve.get("description"):
            lines.append(f"    {cve['description'][:100]}")
    if len(comp["cves"]) > 10:
        lines.append(f"  ... and {len(comp['cves']) - 10} more")
    lines.append("")

    # CWE cluster
    cwe = clusters["cwe"]
    lines.append(f"┌── Cluster 2: {cwe['dimension']} ({len(cwe['cves'])} CVEs) ──┐")
    if cwe["criteria"]:
        lines.append(f"  Criteria: {', '.join(cwe['criteria'])}")
    lines.append("")
    for cve in cwe["cves"][:10]:
        score_str = f"CVSS {cve['cvss_score']}" if cve.get("cvss_score") else "no score"
        lines.append(f"  • {cve['cve_id']} [{score_str}] {cve.get('published', '')}")
        if cve.get("description"):
            lines.append(f"    {cve['description'][:100]}")
    if len(cwe["cves"]) > 10:
        lines.append(f"  ... and {len(cwe['cves']) - 10} more")
    lines.append("")

    # Source cluster
    src = clusters["source"]
    lines.append(f"┌── Cluster 3: {src['dimension']} ({len(src['cves'])} CVEs) ──┐")
    if src["criteria"]:
        lines.append(f"  Criteria: {', '.join(src['criteria'])}")
    lines.append("")
    for cve in src["cves"][:10]:
        score_str = f"CVSS {cve['cvss_score']}" if cve.get("cvss_score") else "no score"
        lines.append(f"  • {cve['cve_id']} [{score_str}] {cve.get('published', '')}")
        if cve.get("description"):
            lines.append(f"    {cve['description'][:100]}")
    if len(src["cves"]) > 10:
        lines.append(f"  ... and {len(src['cves']) - 10} more")
    lines.append("")
    lines.append("╚══════════════════════════════════════════════════════════╝")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strands tool entry point
# ---------------------------------------------------------------------------


def cluster_variants_tool(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands SDK entry point for cluster_variants."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id", "")

    if not isinstance(cve_id, str) or not cve_id.upper().startswith("CVE-"):
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid CVE ID format. Must be like 'CVE-2021-44228'."}],
        }
        log_tool_output_size("cluster_variants", result)
        return result

    data = cluster_variants(cve_id)

    if "error" in data:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": data["error"]}],
        }
    else:
        result = {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"json": data}],
        }

    log_tool_output_size("cluster_variants", result)
    return result
