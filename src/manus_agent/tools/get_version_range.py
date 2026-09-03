#!/usr/bin/env python3
"""
Tool for resolving a CVE to its affected version ranges.

Combines two complementary data sources:

1. **NVD CPE configurations** — walks ``configurations[].nodes[].cpeMatch[]``
   to extract vendor/product identifiers and version constraints
   (``versionStartIncluding``, ``versionEndExcluding``, etc.).

2. **OSV.dev affected ranges** — fetches ecosystem-specific package +
   version-range tuples with ``introduced`` / ``fixed`` / ``last_affected``
   range events, which are far more actionable for dependency-level triage.

The tool merges both views into a unified response that answers:

- Which packages / ecosystems does this CVE affect?
- What are the exact vulnerable version ranges?
- What is the first patched (fixed) version per package?
- What are the NVD-level CPE version constraints?

Ecosystem filtering (``--ecosystem``) narrows the output to a single
ecosystem when the CVE spans multiple (e.g. pypi, npm, Maven).

Public APIs only, no key required (NVD_API_KEY supported for higher rate
limits on the NVD endpoint).
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
# Retry / back-off configuration
# ---------------------------------------------------------------------------
_MAX_RETRIES: int = int(os.environ.get("VERSION_RANGE_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY: float = float(os.environ.get("VERSION_RANGE_RETRY_BASE_DELAY", "1.0"))
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# API endpoints
_NVD_CVE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{osv_id}"

_NVD_TIMEOUT = 15
_OSV_TIMEOUT = 15

# Cap on GHSA alias follows (same as get_osv_data)
_MAX_ALIAS_FOLLOWS = 8

# ---------------------------------------------------------------------------
# Ecosystem mapping: CPE vendor:product → ecosystem hints
# ---------------------------------------------------------------------------
_CPE_ECOSYSTEM_MAP: dict[str, str] = {
    # Python / PyPI
    "python": "PyPI",
    "pypi": "PyPI",
    "django": "PyPI",
    "flask": "PyPI",
    "numpy": "PyPI",
    "pandas": "PyPI",
    "requests": "PyPI",
    "pillow": "PyPI",
    "tensorflow": "PyPI",
    "pytorch": "PyPI",
    "scipy": "PyPI",
    "celery": "PyPI",
    "gunicorn": "PyPI",
    "fastapi": "PyPI",
    "uvicorn": "PyPI",
    "aiohttp": "PyPI",
    "cryptography": "PyPI",
    "paramiko": "PyPI",
    "ansible": "PyPI",
    "salt": "PyPI",
    "twisted": "PyPI",
    # npm
    "node.js": "npm",
    "nodejs": "npm",
    "express": "npm",
    "lodash": "npm",
    "webpack": "npm",
    "next.js": "npm",
    "react": "npm",
    "angular": "npm",
    "vue.js": "npm",
    "axios": "npm",
    "minimist": "npm",
    # Maven / Java
    "apache": "Maven",
    "spring": "Maven",
    "log4j": "Maven",
    "tomcat": "Maven",
    "struts": "Maven",
    "jackson": "Maven",
    "hibernate": "Maven",
    "maven": "Maven",
    "gradle": "Maven",
    # Go
    "golang": "Go",
    "go": "Go",
    # Rust
    "rust": "crates.io",
    "cargo": "crates.io",
    # Ruby
    "ruby": "RubyGems",
    "rails": "RubyGems",
    "rubygems": "RubyGems",
    # PHP
    "php": "Packagist",
    "composer": "Packagist",
    "laravel": "Packagist",
    "symfony": "Packagist",
    "wordpress": "Packagist",
    "drupal": "Packagist",
}

# Normalise user-supplied ecosystem names to OSV ecosystem identifiers
_ECOSYSTEM_ALIASES: dict[str, str] = {
    "pypi": "PyPI",
    "python": "PyPI",
    "npm": "npm",
    "node": "npm",
    "maven": "Maven",
    "java": "Maven",
    "go": "Go",
    "golang": "Go",
    "crates.io": "crates.io",
    "rust": "crates.io",
    "rubygems": "RubyGems",
    "ruby": "RubyGems",
    "packagist": "Packagist",
    "php": "Packagist",
    "nuget": "NuGet",
    "csharp": "NuGet",
    "hex": "Hex",
    "elixir": "Hex",
    "pub": "Pub",
    "dart": "Pub",
    "auto": "",
}

TOOL_SPEC = {
    "name": "get_version_range",
    "description": (
        "Resolves a CVE to its affected version ranges by combining NVD CPE "
        "configurations (vendor/product version constraints) with OSV.dev "
        "ecosystem-specific package + version-range tuples. Returns, per "
        "affected package: ecosystem, package name, vulnerable version ranges, "
        "introduced/fixed versions, and NVD CPE constraints. Use this to answer "
        "'which versions of which packages does this CVE affect, and what version "
        "fixes it?'. Optionally filter by ecosystem (pypi, npm, maven, etc.)."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": ("The CVE identifier to look up (e.g., 'CVE-2021-44228')."),
                },
                "ecosystem": {
                    "type": "string",
                    "description": (
                        "Filter results to a specific ecosystem. "
                        "Supported: auto, pypi, npm, maven, go, crates.io, "
                        "rubygems, packagist, nuget, hex, pub. "
                        "Default: auto (all ecosystems)."
                    ),
                },
            },
            "required": ["cve_id"],
        }
    },
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _build_nvd_headers() -> dict[str, str]:
    """Return request headers, injecting NVD_API_KEY when available."""
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY", "").strip()
    if api_key:
        headers["apiKey"] = api_key
    return headers


def _get_with_retry(
    url: str,
    *,
    timeout: int = 15,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    """GET with exponential back-off retry on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            time.sleep(delay)
        try:
            resp = requests.get(url, headers=headers or {}, timeout=timeout)
            if resp.status_code in _RETRYABLE_STATUSES:
                last_exc = requests.exceptions.HTTPError(f"HTTP {resp.status_code}", response=resp)
                if attempt < _MAX_RETRIES:
                    continue
                raise last_exc
            return resp
        except requests.exceptions.HTTPError:
            raise
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Request failed without a specific exception")


# ---------------------------------------------------------------------------
# NVD CPE configuration parsing
# ---------------------------------------------------------------------------
def _parse_cpe_uri(cpe23: str) -> dict[str, str]:
    """Parse a CPE 2.3 URI into vendor, product, and version fields."""
    # cpe:2.3:a:vendor:product:version:update:edition:lang:sw_edition:target_sw:target_hw:other
    parts = cpe23.split(":")
    result: dict[str, str] = {}
    if len(parts) >= 5:
        result["vendor"] = parts[3] if parts[3] != "*" else ""
        result["product"] = parts[4] if parts[4] != "*" else ""
    if len(parts) >= 6:
        result["version"] = parts[5] if parts[5] != "*" else ""
    return result


def _extract_nvd_ranges(
    cve_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract version range constraints from NVD CVE configurations."""
    ranges: list[dict[str, Any]] = []
    configurations = cve_data.get("configurations", [])

    for config in configurations:
        nodes = config.get("nodes", [])
        for node in nodes:
            operator = node.get("operator", "OR")
            cpe_matches = node.get("cpeMatch", [])
            for match in cpe_matches:
                if not isinstance(match, dict):
                    continue
                if not match.get("vulnerable", False):
                    continue

                cpe23 = match.get("criteria", "")
                parsed = _parse_cpe_uri(cpe23)

                range_entry: dict[str, Any] = {
                    "cpe23": cpe23,
                    "vendor": parsed.get("vendor", ""),
                    "product": parsed.get("product", ""),
                    "exact_version": parsed.get("version", ""),
                    "operator": operator,
                }

                # Version range constraints
                for field in (
                    "versionStartIncluding",
                    "versionStartExcluding",
                    "versionEndIncluding",
                    "versionEndExcluding",
                ):
                    if field in match:
                        range_entry[field] = match[field]

                # Infer ecosystem from vendor/product
                product_lower = parsed.get("product", "").lower()
                vendor_lower = parsed.get("vendor", "").lower()
                eco = _CPE_ECOSYSTEM_MAP.get(product_lower) or _CPE_ECOSYSTEM_MAP.get(vendor_lower) or ""
                range_entry["inferred_ecosystem"] = eco

                ranges.append(range_entry)

    return ranges


def _format_nvd_constraint(r: dict[str, Any]) -> str:
    """Format a single NVD range entry as a human-readable constraint string."""
    parts: list[str] = []
    if r.get("versionStartIncluding"):
        parts.append(f">= {r['versionStartIncluding']}")
    if r.get("versionStartExcluding"):
        parts.append(f"> {r['versionStartExcluding']}")
    if r.get("versionEndIncluding"):
        parts.append(f"<= {r['versionEndIncluding']}")
    if r.get("versionEndExcluding"):
        parts.append(f"< {r['versionEndExcluding']}")
    if parts:
        return ", ".join(parts)
    exact = r.get("exact_version", "")
    if exact:
        return f"== {exact}"
    return "(all versions)"


# ---------------------------------------------------------------------------
# OSV.dev fetching
# ---------------------------------------------------------------------------
def _osv_get(osv_id: str) -> requests.Response:
    """Fetch a single OSV record by ID with retry."""
    url = _OSV_VULN_URL.format(osv_id=osv_id)
    return _get_with_retry(url, timeout=_OSV_TIMEOUT, headers={"Accept": "application/json"})


def _parse_osv_affected(affected: list[Any]) -> list[dict[str, Any]]:
    """Flatten OSV affected entries into package + version-range summaries."""
    packages: list[dict[str, Any]] = []
    for entry in affected or []:
        if not isinstance(entry, dict):
            continue
        pkg = entry.get("package") or {}
        ecosystem = pkg.get("ecosystem") if isinstance(pkg, dict) else None
        name = pkg.get("name") if isinstance(pkg, dict) else None
        if ecosystem is None and name is None:
            continue

        introduced: list[str] = []
        fixed: list[str] = []
        last_affected: list[str] = []
        range_type: str = ""

        for rng in entry.get("ranges", []) or []:
            if not isinstance(rng, dict):
                continue
            if rng.get("type"):
                range_type = rng["type"]
            for ev in rng.get("events", []) or []:
                if not isinstance(ev, dict):
                    continue
                if "introduced" in ev:
                    introduced.append(str(ev["introduced"]))
                if "fixed" in ev:
                    fixed.append(str(ev["fixed"]))
                if "last_affected" in ev:
                    last_affected.append(str(ev["last_affected"]))

        versions = [str(v) for v in (entry.get("versions", []) or []) if v is not None]

        # Build version range strings
        range_strings: list[str] = []
        for i, intro in enumerate(introduced):
            fix = fixed[i] if i < len(fixed) else None
            la = last_affected[i] if i < len(last_affected) else None
            if fix:
                range_strings.append(f">= {intro}, < {fix}")
            elif la:
                range_strings.append(f">= {intro}, <= {la}")
            else:
                range_strings.append(f">= {intro}")

        packages.append(
            {
                "ecosystem": ecosystem or "",
                "package": name or "",
                "introduced": introduced,
                "fixed": fixed,
                "last_affected": last_affected,
                "range_type": range_type,
                "range_strings": range_strings,
                "affected_version_count": len(versions),
                "affected_versions_sample": versions[:15],
                "first_patched_version": fixed[0] if fixed else None,
            }
        )
    return packages


def _fetch_osv_ranges(cve_id: str) -> dict[str, Any]:
    """Fetch OSV.dev data for a CVE and extract affected package ranges.

    Follows GHSA aliases when the CVE record itself lacks package-level
    ``affected`` data (the per-package version ranges typically live in
    the GHSA advisory rather than the CVE mirror).

    Returns a dict with ``found``, ``packages``, ``aliases``, ``severity``,
    and ``error`` keys.
    """
    try:
        resp = _osv_get(cve_id)
    except Exception as exc:
        return {
            "found": False,
            "packages": [],
            "aliases": [],
            "severity": [],
            "error": f"OSV request failed: {exc}",
        }

    if resp.status_code == 404:
        return {
            "found": False,
            "packages": [],
            "aliases": [],
            "severity": [],
        }

    try:
        resp.raise_for_status()
        data = resp.json()
    except (requests.exceptions.HTTPError, ValueError) as exc:
        return {
            "found": False,
            "packages": [],
            "aliases": [],
            "severity": [],
            "error": f"OSV response error: {exc}",
        }

    packages = _parse_osv_affected(data.get("affected", []))
    aliases = [a for a in (data.get("aliases", []) or []) if isinstance(a, str)]
    severity: list[dict[str, str]] = []
    for sev in data.get("severity", []) or []:
        if isinstance(sev, dict):
            severity.append(
                {
                    "type": str(sev.get("type", "")),
                    "score": str(sev.get("score", "")),
                }
            )

    # Follow GHSA aliases if no package-level data on primary record
    if not packages:
        ghsa_aliases = [a for a in aliases if a.startswith("GHSA-")]
        seen_ids: set[str] = {data.get("id", "")}
        for alias in ghsa_aliases[:_MAX_ALIAS_FOLLOWS]:
            try:
                a_resp = _osv_get(alias)
                if a_resp.status_code == 200:
                    a_data = a_resp.json()
                    a_id = a_data.get("id", "")
                    if a_id not in seen_ids:
                        seen_ids.add(a_id)
                        packages.extend(_parse_osv_affected(a_data.get("affected", [])))
                        # Merge severity from alias records
                        for sev in a_data.get("severity", []) or []:
                            if isinstance(sev, dict):
                                severity.append(
                                    {
                                        "type": str(sev.get("type", "")),
                                        "score": str(sev.get("score", "")),
                                    }
                                )
            except Exception:
                continue

    return {
        "found": True,
        "packages": packages,
        "aliases": aliases,
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# NVD fetching
# ---------------------------------------------------------------------------
def _fetch_nvd_ranges(cve_id: str) -> dict[str, Any]:
    """Fetch NVD data for a CVE and extract CPE version constraints.

    Returns a dict with ``found``, ``ranges``, ``description``, ``cvss``,
    ``published``, ``modified``, and ``error`` keys.
    """
    url = f"{_NVD_CVE_URL}?cveId={cve_id.upper()}"
    try:
        resp = _get_with_retry(url, timeout=_NVD_TIMEOUT, headers=_build_nvd_headers())
    except Exception as exc:
        return {
            "found": False,
            "ranges": [],
            "description": "",
            "cvss": {},
            "published": "",
            "modified": "",
            "error": f"NVD request failed: {exc}",
        }

    try:
        resp.raise_for_status()
        data = resp.json()
    except (requests.exceptions.HTTPError, ValueError) as exc:
        return {
            "found": False,
            "ranges": [],
            "description": "",
            "cvss": {},
            "published": "",
            "modified": "",
            "error": f"NVD response error: {exc}",
        }

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return {
            "found": False,
            "ranges": [],
            "description": "",
            "cvss": {},
            "published": "",
            "modified": "",
        }

    cve_item = vulns[0].get("cve", {})
    ranges = _extract_nvd_ranges(cve_item)

    # Extract description
    descriptions = cve_item.get("descriptions", [])
    desc = ""
    for d in descriptions:
        if isinstance(d, dict) and d.get("lang") == "en":
            desc = d.get("value", "")
            break
    if not desc and descriptions:
        desc = descriptions[0].get("value", "") if isinstance(descriptions[0], dict) else ""

    # Extract CVSS
    cvss: dict[str, Any] = {}
    metrics = cve_item.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        metric_list = metrics.get(key, [])
        if metric_list and isinstance(metric_list[0], dict):
            cvss_data = metric_list[0].get("cvssData", {})
            if cvss_data:
                cvss = {
                    "version": cvss_data.get("version", ""),
                    "baseScore": cvss_data.get("baseScore"),
                    "baseSeverity": cvss_data.get("baseSeverity", ""),
                    "vectorString": cvss_data.get("vectorString", ""),
                }
                break

    return {
        "found": True,
        "ranges": ranges,
        "description": desc,
        "cvss": cvss,
        "published": cve_item.get("published", ""),
        "modified": cve_item.get("lastModified", ""),
    }


# ---------------------------------------------------------------------------
# Ecosystem filter
# ---------------------------------------------------------------------------
def _normalise_ecosystem(eco: str) -> str:
    """Normalise a user-supplied ecosystem name to an OSV ecosystem ID."""
    if not eco:
        return ""
    return _ECOSYSTEM_ALIASES.get(eco.lower().strip(), eco)


def _filter_by_ecosystem(
    packages: list[dict[str, Any]],
    nvd_ranges: list[dict[str, Any]],
    ecosystem: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter OSV packages and NVD ranges by ecosystem."""
    if not ecosystem:
        return packages, nvd_ranges

    eco_lower = ecosystem.lower()
    filtered_pkgs = [p for p in packages if p.get("ecosystem", "").lower() == eco_lower]
    filtered_nvd = [r for r in nvd_ranges if r.get("inferred_ecosystem", "").lower() == eco_lower]
    return filtered_pkgs, filtered_nvd


# ---------------------------------------------------------------------------
# Public API: fetch_version_range (used by both tool and CLI)
# ---------------------------------------------------------------------------
def fetch_version_range(
    cve_id: str,
    ecosystem: str = "",
) -> dict[str, Any]:
    """Resolve a CVE to its affected version ranges.

    Combines NVD CPE configurations with OSV.dev ecosystem-specific
    package + version-range tuples.

    Parameters
    ----------
    cve_id : str
        CVE identifier (e.g. ``CVE-2021-44228``).
    ecosystem : str
        Optional ecosystem filter (e.g. ``pypi``, ``npm``, ``maven``).
        Empty string or ``auto`` returns all ecosystems.

    Returns
    -------
    dict
        Structured result with ``found``, ``cve_id``, ``description``,
        ``cvss``, ``osv_packages``, ``nvd_ranges``, ``summary``, etc.
    """
    cve_id = (cve_id or "").strip().upper()
    if not cve_id or not re.match(r"^CVE-\d{4}-\d{4,}$", cve_id):
        return {
            "found": False,
            "cve_id": cve_id,
            "description": "",
            "cvss": {},
            "osv_packages": [],
            "nvd_ranges": [],
            "aliases": [],
            "ecosystems": [],
            "first_patched_versions": [],
            "summary": {},
            "error": "Invalid CVE ID format. Expected CVE-YYYY-NNNN.",
            "message": "Invalid CVE ID format. Expected CVE-YYYY-NNNN.",
        }

    eco_filter = _normalise_ecosystem(ecosystem)

    # Fetch from both sources
    nvd_result = _fetch_nvd_ranges(cve_id)
    osv_result = _fetch_osv_ranges(cve_id)

    found = nvd_result.get("found", False) or osv_result.get("found", False)

    if not found:
        errors: list[str] = []
        if nvd_result.get("error"):
            errors.append(f"NVD: {nvd_result['error']}")
        if osv_result.get("error"):
            errors.append(f"OSV: {osv_result['error']}")
        error_msg = "; ".join(errors) if errors else ""
        return {
            "found": False,
            "cve_id": cve_id,
            "description": "",
            "cvss": {},
            "osv_packages": [],
            "nvd_ranges": [],
            "aliases": [],
            "ecosystems": [],
            "first_patched_versions": [],
            "summary": {},
            "error": error_msg or f"No data found for {cve_id}.",
            "message": f"No version range data found for {cve_id}." + (f" {error_msg}" if error_msg else ""),
        }

    osv_packages = osv_result.get("packages", [])
    nvd_ranges = nvd_result.get("ranges", [])

    # Apply ecosystem filter
    if eco_filter:
        osv_packages, nvd_ranges = _filter_by_ecosystem(osv_packages, nvd_ranges, eco_filter)

    # Collect ecosystems
    ecosystems = sorted(
        {p["ecosystem"] for p in osv_packages if p.get("ecosystem")}
        | {r["inferred_ecosystem"] for r in nvd_ranges if r.get("inferred_ecosystem")}
    )

    # Collect first patched versions
    first_patched: list[dict[str, str]] = []
    seen_patches: set[str] = set()
    for pkg in osv_packages:
        fpv = pkg.get("first_patched_version")
        if fpv:
            key = f"{pkg.get('ecosystem', '')}:{pkg.get('package', '')}:{fpv}"
            if key not in seen_patches:
                seen_patches.add(key)
                first_patched.append(
                    {
                        "ecosystem": pkg.get("ecosystem", ""),
                        "package": pkg.get("package", ""),
                        "fixed_version": fpv,
                    }
                )

    # Build summary
    total_osv = len(osv_packages)
    total_nvd = len(nvd_ranges)
    summary = {
        "osv_package_count": total_osv,
        "nvd_range_count": total_nvd,
        "ecosystem_count": len(ecosystems),
        "has_fix": len(first_patched) > 0,
        "fix_count": len(first_patched),
    }

    # Build human-readable message
    if total_osv > 0 and total_nvd > 0:
        msg = f"{cve_id}: {total_osv} affected package(s) from OSV.dev + {total_nvd} CPE range(s) from NVD"
    elif total_osv > 0:
        msg = f"{cve_id}: {total_osv} affected package(s) from OSV.dev"
    elif total_nvd > 0:
        msg = f"{cve_id}: {total_nvd} CPE range(s) from NVD (no OSV package data)"
    else:
        msg = f"{cve_id}: found in databases but no structured version ranges available"

    if ecosystems:
        msg += f" across {', '.join(ecosystems)}"

    if first_patched:
        fixes = [f"{f['package']}@{f['fixed_version']}" for f in first_patched[:5]]
        msg += f". First patched: {', '.join(fixes)}"
        if len(first_patched) > 5:
            msg += f" (+{len(first_patched) - 5} more)"

    eco_note = f" (filtered: {eco_filter})" if eco_filter else ""

    return {
        "found": True,
        "cve_id": cve_id,
        "description": nvd_result.get("description", ""),
        "cvss": nvd_result.get("cvss", {}),
        "published": nvd_result.get("published", ""),
        "modified": nvd_result.get("modified", ""),
        "ecosystem_filter": eco_filter,
        "ecosystems": ecosystems,
        "osv_packages": osv_packages,
        "nvd_ranges": [
            {
                "vendor": r.get("vendor", ""),
                "product": r.get("product", ""),
                "constraint": _format_nvd_constraint(r),
                "inferred_ecosystem": r.get("inferred_ecosystem", ""),
                "versionStartIncluding": r.get("versionStartIncluding"),
                "versionStartExcluding": r.get("versionStartExcluding"),
                "versionEndIncluding": r.get("versionEndIncluding"),
                "versionEndExcluding": r.get("versionEndExcluding"),
                "exact_version": r.get("exact_version", ""),
            }
            for r in nvd_ranges
        ],
        "aliases": osv_result.get("aliases", []),
        "severity": osv_result.get("severity", []),
        "first_patched_versions": first_patched,
        "summary": summary,
        "message": msg + eco_note,
    }


# ---------------------------------------------------------------------------
# Strands tool entry point
# ---------------------------------------------------------------------------
def get_version_range(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id", "")
    ecosystem = tool_input.get("ecosystem", "")

    result_data = fetch_version_range(cve_id, ecosystem)

    if not result_data.get("found"):
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": result_data.get("message", "No data found.")}],
        }
        log_tool_output_size("get_version_range", result)
        return result

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"json": result_data}],
    }
    log_tool_output_size("get_version_range", result)
    return result
