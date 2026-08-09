#!/usr/bin/env python3
"""
Tool for resolving a CVE to its affected version ranges.

Walks NVD CPE configurations and cross-references OSV.dev ecosystem data
(PyPI, npm, Maven, Go, RustSec, etc.) to produce structured vulnerable
version ranges, affected releases, and first-patched versions.

Strategy:
1. Query NVD for CPE match criteria (versionStart/versionEnd ranges).
2. Query OSV.dev for ecosystem-specific affected packages with precise
   introduced/fixed version events.
3. Merge and deduplicate, optionally filtering by ecosystem.

Both APIs are public; NVD_API_KEY is optional (higher rate limit).
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# Retry / back-off configuration
# ---------------------------------------------------------------------------
_MAX_RETRIES: int = int(os.environ.get("VERSION_RANGE_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY: float = float(os.environ.get("VERSION_RANGE_RETRY_BASE_DELAY", "2.0"))
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_HTTP_TIMEOUT: int = 15

# ---------------------------------------------------------------------------
# NVD API
# ---------------------------------------------------------------------------
_NVD_CVE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def _build_nvd_headers() -> dict[str, str]:
    """Return NVD request headers, injecting NVD_API_KEY when available."""
    headers: dict[str, str] = {"Accept": "application/json"}
    api_key = os.environ.get("NVD_API_KEY", "").strip()
    if api_key:
        headers["apiKey"] = api_key
    return headers


def _http_get_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = _HTTP_TIMEOUT,
) -> requests.Response:
    """GET with exponential back-off on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers or {}, timeout=timeout)
            if resp.status_code in _RETRYABLE_STATUSES and attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY * (2**attempt))
                continue
            return resp
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY * (2**attempt))
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("HTTP request failed without a specific exception")


# ---------------------------------------------------------------------------
# NVD CPE configuration parsing
# ---------------------------------------------------------------------------
def _parse_cpe_uri(cpe23_uri: str) -> dict[str, str]:
    """Parse a CPE 2.3 URI into its component parts.

    Format: cpe:2.3:part:vendor:product:version:update:edition:language:...
    """
    parts = cpe23_uri.split(":")
    if len(parts) < 6:
        return {"raw": cpe23_uri}
    return {
        "part": parts[2] if len(parts) > 2 else "*",
        "vendor": parts[3] if len(parts) > 3 else "*",
        "product": parts[4] if len(parts) > 4 else "*",
        "version": parts[5] if len(parts) > 5 else "*",
        "update": parts[6] if len(parts) > 6 else "*",
    }


def _extract_cpe_ranges(configurations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract version ranges from NVD CPE configurations.

    NVD configurations contain nested nodes with cpeMatch criteria that
    specify versionStartIncluding/Excluding and versionEndIncluding/Excluding.
    """
    ranges: list[dict[str, Any]] = []

    def _process_nodes(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            for match in node.get("cpeMatch", []):
                if not match.get("vulnerable", False):
                    continue

                cpe_uri = match.get("criteria", "")
                parsed = _parse_cpe_uri(cpe_uri)

                range_entry: dict[str, Any] = {
                    "vendor": parsed.get("vendor", "*"),
                    "product": parsed.get("product", "*"),
                    "cpe": cpe_uri,
                }

                # Extract version constraints
                if match.get("versionStartIncluding"):
                    range_entry["version_start"] = match["versionStartIncluding"]
                    range_entry["start_type"] = "including"
                elif match.get("versionStartExcluding"):
                    range_entry["version_start"] = match["versionStartExcluding"]
                    range_entry["start_type"] = "excluding"

                if match.get("versionEndIncluding"):
                    range_entry["version_end"] = match["versionEndIncluding"]
                    range_entry["end_type"] = "including"
                elif match.get("versionEndExcluding"):
                    range_entry["version_end"] = match["versionEndExcluding"]
                    range_entry["end_type"] = "excluding"

                # If the CPE has a specific version (not '*'), it's a single version match
                if parsed.get("version") and parsed["version"] != "*":
                    range_entry["exact_version"] = parsed["version"]

                ranges.append(range_entry)

            # Recurse into child nodes
            if node.get("nodes"):
                _process_nodes(node["nodes"])

    _process_nodes(configurations)
    return ranges


def fetch_nvd_cpe_ranges(cve_id: str) -> dict[str, Any]:
    """Fetch NVD CPE configurations and extract version ranges.

    Returns a dict with keys: success (bool), cpe_ranges (list),
    error (str | None).
    """
    url = f"{_NVD_CVE_URL}?cveId={cve_id.upper()}"
    headers = _build_nvd_headers()

    try:
        resp = _http_get_with_retry(url, headers=headers)
    except Exception as exc:
        return {"success": False, "cpe_ranges": [], "error": f"NVD request failed: {exc}"}

    if resp.status_code == 404:
        return {"success": False, "cpe_ranges": [], "error": f"No NVD record for {cve_id}"}

    try:
        resp.raise_for_status()
        data = resp.json()
    except (requests.exceptions.HTTPError, ValueError) as exc:
        return {"success": False, "cpe_ranges": [], "error": f"NVD response error: {exc}"}

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return {"success": False, "cpe_ranges": [], "error": f"No vulnerability data for {cve_id}"}

    cve_item = vulns[0].get("cve", {})
    configurations = cve_item.get("configurations", [])

    cpe_ranges = _extract_cpe_ranges(configurations)

    return {"success": True, "cpe_ranges": cpe_ranges, "error": None}


# ---------------------------------------------------------------------------
# OSV.dev version range fetching
# ---------------------------------------------------------------------------
_OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{osv_id}"
_OSV_MAX_ALIAS_FOLLOWS = 8


def _fetch_osv_record(osv_id: str) -> dict[str, Any] | None:
    """Fetch a single OSV record. Returns None on 404 or failure."""
    url = _OSV_VULN_URL.format(osv_id=osv_id)
    try:
        resp = _http_get_with_retry(url, headers={"Accept": "application/json"})
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _parse_osv_affected(affected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse OSV affected entries into structured version range data."""
    packages: list[dict[str, Any]] = []
    for entry in affected or []:
        if not isinstance(entry, dict):
            continue
        pkg = entry.get("package") or {}
        ecosystem = pkg.get("ecosystem") if isinstance(pkg, dict) else None
        name = pkg.get("name") if isinstance(pkg, dict) else None
        if ecosystem is None and name is None:
            continue

        ranges_data: list[dict[str, Any]] = []
        for rng in entry.get("ranges", []) or []:
            if not isinstance(rng, dict):
                continue
            range_type = rng.get("type", "ECOSYSTEM")
            events = rng.get("events", []) or []

            introduced: list[str] = []
            fixed: list[str] = []
            last_affected: list[str] = []
            limit: list[str] = []

            for ev in events:
                if not isinstance(ev, dict):
                    continue
                if "introduced" in ev:
                    introduced.append(str(ev["introduced"]))
                if "fixed" in ev:
                    fixed.append(str(ev["fixed"]))
                if "last_affected" in ev:
                    last_affected.append(str(ev["last_affected"]))
                if "limit" in ev:
                    limit.append(str(ev["limit"]))

            ranges_data.append(
                {
                    "type": range_type,
                    "introduced": introduced,
                    "fixed": fixed,
                    "last_affected": last_affected,
                    "limit": limit,
                }
            )

        versions = [str(v) for v in (entry.get("versions", []) or []) if v is not None]

        # Derive first_patched from the fixed events
        all_fixed = []
        for r in ranges_data:
            all_fixed.extend(r["fixed"])

        packages.append(
            {
                "ecosystem": ecosystem,
                "package": name,
                "ranges": ranges_data,
                "first_patched": all_fixed[0] if all_fixed else None,
                "all_patched_versions": all_fixed,
                "affected_version_count": len(versions),
                "affected_versions": versions[:20],  # Cap for output size
            }
        )
    return packages


def fetch_osv_version_ranges(cve_id: str) -> dict[str, Any]:
    """Fetch OSV.dev version ranges for a CVE.

    Follows GHSA aliases when the primary CVE record lacks per-package data.
    Returns a dict with: success (bool), packages (list), error (str | None).
    """
    record = _fetch_osv_record(cve_id)
    if record is None:
        return {"success": False, "packages": [], "error": f"No OSV record for {cve_id}"}

    packages = _parse_osv_affected(record.get("affected", []))
    seen_ids: set[str] = {record.get("id", "")}

    # If no packages found, follow GHSA aliases
    if not packages:
        aliases = record.get("aliases", []) or []
        ghsa_aliases = [a for a in aliases if isinstance(a, str) and a.startswith("GHSA-")]
        for alias in ghsa_aliases[:_OSV_MAX_ALIAS_FOLLOWS]:
            if alias in seen_ids:
                continue
            alias_record = _fetch_osv_record(alias)
            if alias_record:
                seen_ids.add(alias_record.get("id", ""))
                alias_pkgs = _parse_osv_affected(alias_record.get("affected", []))
                packages.extend(alias_pkgs)

    return {"success": True, "packages": packages, "error": None}


# ---------------------------------------------------------------------------
# Ecosystem mapping: CPE vendor/product → ecosystem
# ---------------------------------------------------------------------------
_CPE_TO_ECOSYSTEM: dict[str, str] = {
    "python": "PyPI",
    "pip": "PyPI",
    "pypi": "PyPI",
    "django": "PyPI",
    "flask": "PyPI",
    "numpy": "PyPI",
    "pandas": "PyPI",
    "requests": "PyPI",
    "node.js": "npm",
    "nodejs": "npm",
    "node": "npm",
    "npm": "npm",
    "express": "npm",
    "lodash": "npm",
    "webpack": "npm",
    "react": "npm",
    "maven": "Maven",
    "apache": "Maven",
    "spring": "Maven",
    "log4j": "Maven",
    "jackson": "Maven",
    "go": "Go",
    "golang": "Go",
    "rust": "crates.io",
    "cargo": "crates.io",
    "rubygems": "RubyGems",
    "ruby": "RubyGems",
    "nuget": "NuGet",
    ".net": "NuGet",
    "dotnet": "NuGet",
    "packagist": "Packagist",
    "php": "Packagist",
    "composer": "Packagist",
    "linux": "Linux",
    "kernel": "Linux",
}


def _infer_ecosystem_from_cpe(vendor: str, product: str) -> str | None:
    """Attempt to infer ecosystem from CPE vendor/product names."""
    for token in (vendor.lower(), product.lower()):
        if token in _CPE_TO_ECOSYSTEM:
            return _CPE_TO_ECOSYSTEM[token]
    return None


# ---------------------------------------------------------------------------
# Core logic: merge NVD + OSV data
# ---------------------------------------------------------------------------
def fetch_version_range(cve_id: str, ecosystem_filter: str | None = None) -> dict[str, Any]:
    """Resolve a CVE to its affected version ranges from NVD + OSV.

    Args:
        cve_id: CVE identifier (e.g., CVE-2021-44228).
        ecosystem_filter: Optional ecosystem to restrict results to.
            Accepted values: auto (default), pypi, npm, maven, go, crates.io,
            rubygems, nuget, packagist, or any OSV ecosystem string.

    Returns a comprehensive dict with:
        - cve_id: normalised CVE identifier
        - nvd_ranges: CPE-based version ranges from NVD
        - osv_packages: ecosystem-specific packages from OSV.dev
        - summary: human-readable summary
        - first_patched: first patched version (from OSV, if available)
    """
    cve_id = (cve_id or "").strip().upper()
    if not cve_id or not cve_id.startswith("CVE-"):
        return {
            "cve_id": cve_id,
            "nvd_ranges": [],
            "osv_packages": [],
            "summary": "Invalid CVE ID. Must match CVE-YYYY-NNNN format.",
            "error": "Invalid CVE ID format.",
        }

    # Normalise ecosystem filter
    eco_filter: str | None = None
    if ecosystem_filter and ecosystem_filter.lower() != "auto":
        eco_filter = ecosystem_filter

    # 1. Fetch NVD CPE ranges
    nvd_result = fetch_nvd_cpe_ranges(cve_id)
    nvd_ranges = nvd_result.get("cpe_ranges", [])

    # 2. Fetch OSV.dev ecosystem-level version ranges
    osv_result = fetch_osv_version_ranges(cve_id)
    osv_packages = osv_result.get("packages", [])

    # 3. Apply ecosystem filter
    if eco_filter:
        eco_lower = eco_filter.lower()
        osv_packages = [p for p in osv_packages if p.get("ecosystem", "").lower() == eco_lower]
        nvd_ranges = [
            r
            for r in nvd_ranges
            if _infer_ecosystem_from_cpe(r.get("vendor", ""), r.get("product", "")) is None
            or (_infer_ecosystem_from_cpe(r.get("vendor", ""), r.get("product", "")) or "").lower() == eco_lower
        ]

    # 4. Enrich NVD ranges with inferred ecosystems
    for r in nvd_ranges:
        inferred = _infer_ecosystem_from_cpe(r.get("vendor", ""), r.get("product", ""))
        if inferred:
            r["inferred_ecosystem"] = inferred

    # 5. Derive first_patched from OSV data
    first_patched: str | None = None
    for pkg in osv_packages:
        if pkg.get("first_patched"):
            first_patched = pkg["first_patched"]
            break

    # 6. Build summary
    total_nvd = len(nvd_ranges)
    total_osv = len(osv_packages)
    ecosystems = sorted({p["ecosystem"] for p in osv_packages if p.get("ecosystem")})

    parts: list[str] = []
    if total_nvd:
        parts.append(f"{total_nvd} NVD CPE range(s)")
    if total_osv:
        eco_str = ", ".join(ecosystems) if ecosystems else "unknown"
        parts.append(f"{total_osv} OSV package(s) [{eco_str}]")
    if first_patched:
        parts.append(f"first patched: {first_patched}")

    if parts:
        summary = f"{cve_id}: {'; '.join(parts)}."
    elif nvd_result.get("error") and osv_result.get("error"):
        summary = f"{cve_id}: No version range data available (NVD: {nvd_result['error']}; OSV: {osv_result['error']})."
    elif not total_nvd and not total_osv:
        summary = f"{cve_id}: No affected version ranges found in NVD or OSV.dev."
    else:
        summary = f"{cve_id}: Version range lookup completed."

    return {
        "cve_id": cve_id,
        "nvd_ranges": nvd_ranges,
        "osv_packages": osv_packages,
        "ecosystems": ecosystems,
        "first_patched": first_patched,
        "summary": summary,
        "nvd_error": nvd_result.get("error"),
        "osv_error": osv_result.get("error"),
    }


# ---------------------------------------------------------------------------
# TOOL_SPEC (Strands SDK)
# ---------------------------------------------------------------------------
TOOL_SPEC = {
    "name": "get_version_range",
    "description": (
        "Resolves a CVE to its affected version ranges by combining NVD CPE "
        "configurations (versionStart/End ranges) with OSV.dev ecosystem-specific "
        "package data (introduced/fixed version events). Returns structured vulnerable "
        "ranges, the list of affected ecosystems, and the first-patched version. "
        "Use this to answer 'what versions are vulnerable to this CVE and what "
        "version fixes it?'. Optionally filter by ecosystem (pypi, npm, maven, go)."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier (e.g., 'CVE-2021-44228').",
                },
                "ecosystem": {
                    "type": "string",
                    "description": (
                        "Optional ecosystem filter: auto (default), pypi, npm, maven, go, "
                        "crates.io, rubygems, nuget, packagist."
                    ),
                },
            },
            "required": ["cve_id"],
        }
    },
}


# ---------------------------------------------------------------------------
# Strands tool entry point
# ---------------------------------------------------------------------------
def get_version_range(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands tool handler for get_version_range."""
    import json

    tool_use_id = tool["toolUseId"]
    tool_input = tool.get("input", {}) or {}
    cve_id = tool_input.get("cve_id")
    ecosystem = tool_input.get("ecosystem")

    if not isinstance(cve_id, str) or not cve_id.strip():
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid CVE ID. Must be a non-empty string."}],
        }
        log_tool_output_size("get_version_range", result)
        return result

    payload = fetch_version_range(cve_id, ecosystem_filter=ecosystem)
    has_data = bool(payload.get("nvd_ranges") or payload.get("osv_packages"))
    status = "success" if has_data else "error"

    result = {
        "toolUseId": tool_use_id,
        "status": status,
        "content": [{"text": json.dumps(payload, indent=2)}],
    }
    log_tool_output_size("get_version_range", result)
    return result
