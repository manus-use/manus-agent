#!/usr/bin/env python3
"""
Tool for resolving CVE-affected version ranges across NVD CPE configurations
and OSV.dev ecosystem-specific advisories.

Walks NVD CPE match entries to extract structured vulnerable semver ranges,
then cross-references OSV.dev to produce concrete per-package version data
including affected releases and first-patched versions.

Strategy:
1. Fetch NVD CVE record → parse ``configurations[].nodes[].cpeMatch``
   entries marked ``vulnerable: true``, extracting version boundary fields
   (``versionStartIncluding``, ``versionEndExcluding``, etc.).
2. Fetch OSV.dev record → merge per-ecosystem ``affected[].ranges`` events
   (introduced/fixed/last_affected) and enumerated ``versions`` lists.
3. When ``--ecosystem`` is specified, filter results to that ecosystem only.
4. Produce a unified view: CPE-derived ranges + OSV package-level ranges +
   first-patched version + affected release count.

Public APIs, no key required (NVD key optional for higher rate limits).
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

import requests

from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# CVE validation
# ---------------------------------------------------------------------------
_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Retry / back-off configuration
# ---------------------------------------------------------------------------
_MAX_RETRIES: int = int(os.environ.get("VERSION_RANGE_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY: float = float(os.environ.get("VERSION_RANGE_RETRY_BASE_DELAY", "1.0"))
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_HTTP_TIMEOUT = 15

# ---------------------------------------------------------------------------
# API URLs
# ---------------------------------------------------------------------------
_NVD_CVE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{osv_id}"

# Maximum GHSA aliases to follow when the CVE record has no package data.
_MAX_ALIAS_FOLLOWS = 8

# ---------------------------------------------------------------------------
# Ecosystem mapping: CPE vendor:product → ecosystem hints
# ---------------------------------------------------------------------------
_CPE_ECOSYSTEM_MAP: dict[str, str] = {
    "python": "PyPI",
    "pypi": "PyPI",
    "pip": "PyPI",
    "django": "PyPI",
    "flask": "PyPI",
    "numpy": "PyPI",
    "pandas": "PyPI",
    "requests": "PyPI",
    "urllib3": "PyPI",
    "pillow": "PyPI",
    "node.js": "npm",
    "nodejs": "npm",
    "node": "npm",
    "npm": "npm",
    "express": "npm",
    "lodash": "npm",
    "webpack": "npm",
    "java": "Maven",
    "maven": "Maven",
    "spring": "Maven",
    "apache": "Maven",
    "log4j": "Maven",
    "tomcat": "Maven",
    "struts": "Maven",
    "golang": "Go",
    "go": "Go",
    "rust": "crates.io",
    "cargo": "crates.io",
    "rubygems": "RubyGems",
    "ruby": "RubyGems",
    "rails": "RubyGems",
    "nuget": "NuGet",
    ".net": "NuGet",
    "dotnet": "NuGet",
    "packagist": "Packagist",
    "php": "Packagist",
    "composer": "Packagist",
    "laravel": "Packagist",
    "wordpress": "Packagist",
    "linux": "Linux",
    "kernel": "Linux",
    "debian": "Debian",
    "ubuntu": "Debian",
    "alpine": "Alpine",
}

# Canonical ecosystem filter values
_ECOSYSTEM_FILTER_MAP: dict[str, str] = {
    "auto": "auto",
    "pypi": "PyPI",
    "npm": "npm",
    "maven": "Maven",
    "go": "Go",
    "crates.io": "crates.io",
    "rubygems": "RubyGems",
    "nuget": "NuGet",
    "packagist": "Packagist",
}


# ---------------------------------------------------------------------------
# HTTP helper with retry/back-off
# ---------------------------------------------------------------------------
def _get_with_retry(url: str, *, params: dict | None = None, headers: dict | None = None) -> requests.Response:
    """HTTP GET with exponential back-off on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers or {}, timeout=_HTTP_TIMEOUT)
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
# NVD CPE parsing
# ---------------------------------------------------------------------------
def _parse_cpe_uri(cpe_str: str) -> dict[str, str]:
    """Parse a CPE 2.3 URI into component fields.

    Format: cpe:2.3:part:vendor:product:version:update:edition:language:sw_edition:target_sw:target_hw:other
    """
    parts = cpe_str.split(":")
    if len(parts) < 5:
        return {}
    return {
        "part": parts[2] if len(parts) > 2 else "",
        "vendor": parts[3] if len(parts) > 3 else "",
        "product": parts[4] if len(parts) > 4 else "",
        "version": parts[5] if len(parts) > 5 else "",
        "update": parts[6] if len(parts) > 6 else "",
        "target_sw": parts[10] if len(parts) > 10 else "",
    }


def _guess_ecosystem(cpe: dict[str, str]) -> str:
    """Guess the ecosystem from CPE vendor/product fields."""
    for field in ("vendor", "product", "target_sw"):
        val = cpe.get(field, "").lower()
        if val in _CPE_ECOSYSTEM_MAP:
            return _CPE_ECOSYSTEM_MAP[val]
    return "Unknown"


def _extract_nvd_ranges(configurations: list[dict]) -> list[dict[str, Any]]:
    """Extract vulnerable version ranges from NVD CPE configurations.

    Returns a list of dicts, each with:
      - vendor, product, version (exact or wildcard)
      - version_start_including, version_start_excluding
      - version_end_including, version_end_excluding
      - ecosystem_hint (guessed from CPE fields)
      - cpe_uri (raw CPE 2.3 string)
      - range_display (human-readable range string)
    """
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for config in configurations:
        nodes = config.get("nodes", [])
        for node in nodes:
            for match in node.get("cpeMatch", []):
                if not match.get("vulnerable", False):
                    continue
                cpe_uri = match.get("criteria", "")
                cpe = _parse_cpe_uri(cpe_uri)
                if not cpe:
                    continue

                vsi = match.get("versionStartIncluding", "")
                vse = match.get("versionStartExcluding", "")
                vei = match.get("versionEndIncluding", "")
                vee = match.get("versionEndExcluding", "")
                exact_version = cpe.get("version", "")
                if exact_version in ("*", "-", ""):
                    exact_version = ""

                # Build dedup key
                dedup = f"{cpe.get('vendor')}:{cpe.get('product')}:{exact_version}:{vsi}:{vse}:{vei}:{vee}"
                if dedup in seen:
                    continue
                seen.add(dedup)

                # Build human-readable range
                range_display = _build_range_display(exact_version, vsi, vse, vei, vee)

                results.append(
                    {
                        "vendor": cpe.get("vendor", ""),
                        "product": cpe.get("product", ""),
                        "version": exact_version,
                        "version_start_including": vsi,
                        "version_start_excluding": vse,
                        "version_end_including": vei,
                        "version_end_excluding": vee,
                        "ecosystem_hint": _guess_ecosystem(cpe),
                        "cpe_uri": cpe_uri,
                        "range_display": range_display,
                    }
                )

    return results


def _build_range_display(exact: str, vsi: str, vse: str, vei: str, vee: str) -> str:
    """Build a human-readable version range string."""
    if exact:
        return f"= {exact}"

    parts: list[str] = []
    if vsi:
        parts.append(f">= {vsi}")
    elif vse:
        parts.append(f"> {vse}")

    if vee:
        parts.append(f"< {vee}")
    elif vei:
        parts.append(f"<= {vei}")

    if parts:
        return ", ".join(parts)
    return "all versions"


# ---------------------------------------------------------------------------
# NVD fetch
# ---------------------------------------------------------------------------
def _fetch_nvd_configurations(cve_id: str) -> dict[str, Any]:
    """Fetch NVD CVE record and extract CPE configurations.

    Returns dict with keys: configurations (raw), cpe_ranges (parsed),
    description, cvss_v31_score, cvss_v31_severity, references.
    """
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY", "").strip()
    if api_key:
        headers["apiKey"] = api_key

    try:
        resp = _get_with_retry(_NVD_CVE_URL, params={"cveId": cve_id}, headers=headers)
    except Exception as exc:
        return {"error": f"NVD request failed: {exc}"}

    if resp.status_code == 404:
        return {"error": f"CVE {cve_id} not found in NVD."}

    try:
        resp.raise_for_status()
        data = resp.json()
    except (requests.exceptions.HTTPError, ValueError) as exc:
        return {"error": f"NVD response error: {exc}"}

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return {"error": f"No vulnerability data returned for {cve_id}."}

    cve_data = vulns[0].get("cve", {})
    configurations = cve_data.get("configurations", [])
    cpe_ranges = _extract_nvd_ranges(configurations)

    # Extract CVSS v3.1 score if available
    metrics = cve_data.get("metrics", {})
    cvss_v31 = None
    cvss_v31_severity = None
    for metric_list in (metrics.get("cvssMetricV31", []), metrics.get("cvssMetricV30", [])):
        for m in metric_list:
            cvss_data = m.get("cvssData", {})
            if cvss_data.get("baseScore") is not None:
                cvss_v31 = cvss_data["baseScore"]
                cvss_v31_severity = cvss_data.get("baseSeverity", "")
                break
        if cvss_v31 is not None:
            break

    # Extract description
    descriptions = cve_data.get("descriptions", [])
    description = ""
    for d in descriptions:
        if d.get("lang") == "en":
            description = d.get("value", "")
            break
    if not description and descriptions:
        description = descriptions[0].get("value", "")

    # Extract references
    references = [r.get("url") for r in cve_data.get("references", []) if r.get("url")]

    return {
        "cpe_ranges": cpe_ranges,
        "cpe_count": len(cpe_ranges),
        "description": description,
        "cvss_v31_score": cvss_v31,
        "cvss_v31_severity": cvss_v31_severity,
        "references": references[:15],
    }


# ---------------------------------------------------------------------------
# OSV.dev fetch
# ---------------------------------------------------------------------------
def _osv_get(osv_id: str) -> requests.Response:
    """Fetch a single OSV record with retry."""
    url = _OSV_VULN_URL.format(osv_id=osv_id)
    return _get_with_retry(url, headers={"Accept": "application/json"})


def _parse_osv_affected(affected: list[dict]) -> list[dict[str, Any]]:
    """Parse OSV affected entries into structured package version data."""
    packages: list[dict[str, Any]] = []
    for entry in affected or []:
        if not isinstance(entry, dict):
            continue
        pkg = entry.get("package") or {}
        if not isinstance(pkg, dict):
            continue
        ecosystem = pkg.get("ecosystem")
        name = pkg.get("name")
        if not ecosystem and not name:
            continue

        introduced: list[str] = []
        fixed: list[str] = []
        last_affected: list[str] = []
        range_type = ""

        for rng in entry.get("ranges", []) or []:
            if not isinstance(rng, dict):
                continue
            range_type = rng.get("type", range_type)
            for ev in rng.get("events", []) or []:
                if not isinstance(ev, dict):
                    continue
                if "introduced" in ev:
                    introduced.append(str(ev["introduced"]))
                if "fixed" in ev:
                    fixed.append(str(ev["fixed"]))
                if "last_affected" in ev:
                    last_affected.append(str(ev["last_affected"]))

        versions = [str(v) for v in (entry.get("versions") or []) if v is not None]

        # Build human-readable ranges
        range_strings: list[str] = []
        for i, intro in enumerate(introduced):
            fix = fixed[i] if i < len(fixed) else None
            la = last_affected[i] if i < len(last_affected) else None
            if intro == "0":
                start = "all versions"
            else:
                start = f">= {intro}"
            if fix:
                range_strings.append(f"{start}, < {fix}")
            elif la:
                range_strings.append(f"{start}, <= {la}")
            else:
                range_strings.append(f"{start}")

        # Determine first patched version
        first_patched = fixed[0] if fixed else None

        packages.append(
            {
                "ecosystem": ecosystem,
                "package": name,
                "introduced": introduced,
                "fixed": fixed,
                "last_affected": last_affected,
                "first_patched": first_patched,
                "range_type": range_type,
                "range_strings": range_strings,
                "affected_version_count": len(versions),
                "affected_versions": versions[:30],
            }
        )
    return packages


def _fetch_osv_ranges(cve_id: str) -> dict[str, Any]:
    """Fetch OSV.dev data and extract per-package version ranges.

    Follows GHSA aliases when the CVE record has no package data.
    """
    try:
        resp = _osv_get(cve_id)
    except Exception as exc:
        return {"error": f"OSV request failed: {exc}", "packages": []}

    if resp.status_code == 404:
        return {"packages": [], "aliases": []}

    try:
        resp.raise_for_status()
        primary = resp.json()
    except (requests.exceptions.HTTPError, ValueError) as exc:
        return {"error": f"OSV response error: {exc}", "packages": []}

    packages = _parse_osv_affected(primary.get("affected", []))
    aliases = primary.get("aliases", []) or []
    seen_ids = {primary.get("id", cve_id)}

    # Follow GHSA aliases if primary has no package data
    if not packages:
        ghsa_aliases = [a for a in aliases if isinstance(a, str) and a.startswith("GHSA-")]
        for alias in ghsa_aliases[:_MAX_ALIAS_FOLLOWS]:
            try:
                a_resp = _osv_get(alias)
                if a_resp.status_code == 200:
                    a_data = a_resp.json()
                    a_id = a_data.get("id", "")
                    if a_id not in seen_ids:
                        seen_ids.add(a_id)
                        extra_pkgs = _parse_osv_affected(a_data.get("affected", []))
                        packages.extend(extra_pkgs)
                        for a in a_data.get("aliases", []) or []:
                            if a not in aliases:
                                aliases.append(a)
            except Exception:  # noqa: BLE001
                continue

    return {"packages": packages, "aliases": aliases}


# ---------------------------------------------------------------------------
# Unified version-range resolution
# ---------------------------------------------------------------------------
def fetch_version_range(cve_id: str, *, ecosystem: str = "auto") -> dict[str, Any]:
    """Resolve version ranges for a CVE from NVD + OSV.dev.

    Args:
        cve_id: CVE identifier (e.g. "CVE-2021-44228").
        ecosystem: Filter to a specific ecosystem ("auto", "pypi", "npm",
            "maven", "go", etc.). "auto" returns all.

    Returns:
        Structured dict with NVD CPE ranges, OSV package ranges,
        first-patched versions, and a human-readable summary.
    """
    cve_id = (cve_id or "").strip().upper()
    if not _CVE_RE.match(cve_id):
        return {
            "found": False,
            "cve_id": cve_id,
            "error": f"Invalid CVE ID format: {cve_id!r}. Expected CVE-YYYY-NNNNN.",
            "message": f"Invalid CVE ID format: {cve_id!r}. Expected CVE-YYYY-NNNNN.",
        }

    # Resolve ecosystem filter
    eco_filter = _ECOSYSTEM_FILTER_MAP.get(ecosystem.lower(), ecosystem) if ecosystem else "auto"

    # Fetch NVD CPE configurations
    nvd_data = _fetch_nvd_configurations(cve_id)
    nvd_error = nvd_data.get("error")
    cpe_ranges = nvd_data.get("cpe_ranges", [])

    # Filter CPE ranges by ecosystem if specified
    if eco_filter != "auto" and cpe_ranges:
        cpe_ranges = [r for r in cpe_ranges if r.get("ecosystem_hint", "").lower() == eco_filter.lower()]

    # Fetch OSV.dev package ranges
    osv_data = _fetch_osv_ranges(cve_id)
    osv_error = osv_data.get("error")
    osv_packages = osv_data.get("packages", [])

    # Filter OSV packages by ecosystem if specified
    if eco_filter != "auto" and osv_packages:
        osv_packages = [p for p in osv_packages if (p.get("ecosystem") or "").lower() == eco_filter.lower()]

    # Determine first patched versions across all sources
    first_patched: dict[str, str] = {}
    for pkg in osv_packages:
        eco = pkg.get("ecosystem", "")
        name = pkg.get("package", "")
        fp = pkg.get("first_patched")
        if fp and eco and name:
            key = f"{eco}/{name}"
            if key not in first_patched:
                first_patched[key] = fp

    # Collect all affected ecosystems
    nvd_ecosystems = sorted({r["ecosystem_hint"] for r in cpe_ranges if r.get("ecosystem_hint") != "Unknown"})
    osv_ecosystems = sorted({p["ecosystem"] for p in osv_packages if p.get("ecosystem")})
    all_ecosystems = sorted(set(nvd_ecosystems) | set(osv_ecosystems))

    # Count total affected releases from OSV enumerated versions
    total_affected_versions = sum(p.get("affected_version_count", 0) for p in osv_packages)

    found = bool(cpe_ranges or osv_packages)
    errors: list[str] = []
    if nvd_error:
        errors.append(f"NVD: {nvd_error}")
    if osv_error:
        errors.append(f"OSV: {osv_error}")

    # Build summary message
    if found:
        parts: list[str] = []
        if cpe_ranges:
            parts.append(f"{len(cpe_ranges)} CPE range(s) from NVD")
        if osv_packages:
            parts.append(f"{len(osv_packages)} package(s) from OSV.dev")
        if all_ecosystems:
            parts.append(f"ecosystems: {', '.join(all_ecosystems)}")
        if first_patched:
            fp_summary = "; ".join(f"{k} → {v}" for k, v in sorted(first_patched.items())[:5])
            parts.append(f"first patched: {fp_summary}")
        message = f"{cve_id}: {'; '.join(parts)}."
    else:
        message = f"No version range data found for {cve_id}."
        if errors:
            message += f" Errors: {'; '.join(errors)}"

    result: dict[str, Any] = {
        "found": found,
        "cve_id": cve_id,
        "ecosystem_filter": eco_filter,
        "nvd": {
            "cpe_ranges": cpe_ranges,
            "cpe_count": len(cpe_ranges),
            "description": nvd_data.get("description", ""),
            "cvss_v31_score": nvd_data.get("cvss_v31_score"),
            "cvss_v31_severity": nvd_data.get("cvss_v31_severity"),
        },
        "osv": {
            "packages": osv_packages,
            "package_count": len(osv_packages),
            "aliases": osv_data.get("aliases", []),
        },
        "summary": {
            "affected_ecosystems": all_ecosystems,
            "first_patched_versions": first_patched,
            "total_affected_version_count": total_affected_versions,
        },
        "message": message,
    }
    if errors:
        result["errors"] = errors
    return result


# ---------------------------------------------------------------------------
# Strands tool interface
# ---------------------------------------------------------------------------
try:
    from strands.types.tools import ToolResult, ToolUse

    TOOL_SPEC = {
        "name": "get_version_range",
        "description": (
            "Resolves affected version ranges for a CVE by walking NVD CPE configurations "
            "and cross-referencing OSV.dev ecosystem-specific advisories. Returns structured "
            "vulnerable semver ranges, affected releases, and first-patched versions per "
            "package ecosystem (PyPI, npm, Maven, Go, etc.). Use when you need to know "
            "exactly which versions of a package are affected by a CVE and what version "
            "fixes it."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "cve_id": {
                        "type": "string",
                        "description": "The CVE identifier (e.g. 'CVE-2021-44228').",
                    },
                    "ecosystem": {
                        "type": "string",
                        "description": (
                            "Filter to a specific ecosystem: 'auto' (all), 'pypi', 'npm', "
                            "'maven', 'go', 'crates.io', 'rubygems', 'nuget', 'packagist'. "
                            "Default: 'auto'."
                        ),
                        "default": "auto",
                    },
                },
                "required": ["cve_id"],
            }
        },
    }

    def get_version_range(tool: ToolUse, **kwargs: Any) -> ToolResult:
        """Strands tool entry point."""
        import json

        tool_use_id = tool["toolUseId"]
        tool_input = tool.get("input", {}) or {}
        cve_id = tool_input.get("cve_id")
        ecosystem = tool_input.get("ecosystem", "auto")

        if not isinstance(cve_id, str) or not cve_id.strip():
            result: ToolResult = {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": "Invalid CVE ID. Must be a non-empty string."}],
            }
            log_tool_output_size("get_version_range", result)
            return result

        payload = fetch_version_range(cve_id, ecosystem=ecosystem)
        status = "success" if payload.get("found") else "error"

        result = {
            "toolUseId": tool_use_id,
            "status": status,
            "content": [{"text": json.dumps(payload, indent=2)}],
        }
        log_tool_output_size("get_version_range", result)
        return result

except ImportError:
    # Strands not available — library-only mode
    TOOL_SPEC = None  # type: ignore[assignment]
