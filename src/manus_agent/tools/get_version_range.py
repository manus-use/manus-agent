#!/usr/bin/env python3
"""
Tool for resolving CVE-affected version ranges across NVD and OSV.dev.

Walks NVD CPE configurations to extract vendor-declared version boundaries
and cross-references OSV.dev ecosystem advisories (PyPI, npm, Maven, Go,
RustSec, etc.) to produce structured vulnerable semver ranges, a list of
affected releases, and the first patched release per package.

Strategy:
1. Fetch NVD CVE record → parse ``configurations[].nodes[].cpeMatch``
   to extract CPE-based version constraints (versionStartIncluding,
   versionEndExcluding, etc.).
2. Fetch OSV.dev record(s) → parse ``affected[].ranges[].events`` to
   extract ecosystem-specific introduced/fixed/last_affected markers,
   plus the concrete ``affected[].versions`` list when available.
3. Optionally filter by ``--ecosystem`` to narrow results.
4. Merge and deduplicate across sources, preferring OSV ecosystem data
   (more actionable) over NVD CPE data (more authoritative but coarser).

Public APIs, no key required:
- NVD: GET https://services.nvd.nist.gov/rest/json/cves/2.0?cveId=<id>
- OSV: GET https://api.osv.dev/v1/vulns/<id>
"""

from __future__ import annotations

import json
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
_HTTP_TIMEOUT: int = 15

# NVD
_NVD_CVE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# OSV
_OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{osv_id}"
_OSV_MAX_ALIAS_FOLLOWS = 8

# Ecosystem normalisation map (lowercase key → canonical label)
_ECOSYSTEM_ALIASES: dict[str, str] = {
    "pypi": "PyPI",
    "pip": "PyPI",
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
    "nuget": "NuGet",
    "packagist": "Packagist",
    "php": "Packagist",
    "hex": "Hex",
    "pub": "Pub",
    "hackage": "Hackage",
    "linux": "Linux",
    "alpine": "Alpine",
    "debian": "Debian",
    "auto": "auto",
}

# Valid ecosystem filter values for the CLI
VALID_ECOSYSTEMS = ("auto", "pypi", "npm", "maven", "go", "crates.io", "nuget", "rubygems")


TOOL_SPEC = {
    "name": "get_version_range",
    "description": (
        "Resolves a CVE to its affected version ranges by combining NVD CPE "
        "configuration data with OSV.dev ecosystem-specific advisories (PyPI, "
        "npm, Maven, Go, RustSec, etc.). Returns structured vulnerable version "
        "ranges, concrete affected version lists, and the first patched release "
        "per package. Use this to answer 'which versions are vulnerable to this "
        "CVE and what should I upgrade to?'"
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
                        "Filter results to a specific ecosystem. "
                        "One of: auto, pypi, npm, maven, go, crates.io, nuget, rubygems. "
                        "Default 'auto' returns all ecosystems."
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
def _get_with_retry(url: str, *, headers: dict[str, str] | None = None) -> requests.Response:
    """GET with exponential back-off on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=_HTTP_TIMEOUT, headers=headers or {})
            if resp.status_code in _RETRYABLE_STATUSES and attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY * (2**attempt))
                continue
            return resp
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
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
def _normalise_ecosystem(ecosystem: str | None) -> str:
    """Normalise an ecosystem string to its canonical label."""
    if not ecosystem:
        return ""
    return _ECOSYSTEM_ALIASES.get(ecosystem.lower().strip(), ecosystem)


def _cpe_to_ecosystem(cpe_uri: str) -> str:
    """Best-effort CPE 2.3 URI → ecosystem mapping.

    CPE format: cpe:2.3:a:<vendor>:<product>:...
    We use vendor/product heuristics to guess the ecosystem.
    """
    parts = cpe_uri.split(":")
    if len(parts) < 5:
        return ""
    vendor = parts[3].lower() if len(parts) > 3 else ""

    # Well-known vendor → ecosystem mappings
    eco_hints: dict[str, str] = {
        "python": "PyPI",
        "pypi": "PyPI",
        "djangoproject": "PyPI",
        "flask": "PyPI",
        "pallets": "PyPI",
        "npmjs": "npm",
        "nodejs": "npm",
        "node.js": "npm",
        "apache": "Maven",
        "oracle": "Maven",
        "spring": "Maven",
        "golang": "Go",
        "google": "",  # ambiguous
        "microsoft": "NuGet",
        "linux": "Linux",
        "linuxfoundation": "Linux",
        "debian": "Debian",
        "canonical": "Linux",
        "redhat": "Linux",
    }
    return eco_hints.get(vendor, "")


def _parse_nvd_configurations(configurations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract version-range records from NVD CPE match configurations.

    Returns a list of dicts with keys:
        cpe23Uri, vendor, product, ecosystem (best-effort),
        versionStartIncluding, versionStartExcluding,
        versionEndIncluding, versionEndExcluding,
        vulnerable (bool), range_summary (human-readable string).
    """
    results: list[dict[str, Any]] = []

    for config in configurations or []:
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                if not isinstance(match, dict):
                    continue
                cpe = match.get("criteria", "")
                parts = cpe.split(":")
                vendor = parts[3] if len(parts) > 3 else ""
                product = parts[4] if len(parts) > 4 else ""
                version_exact = parts[5] if len(parts) > 5 else "*"

                vsi = match.get("versionStartIncluding")
                vse = match.get("versionStartExcluding")
                vei = match.get("versionEndIncluding")
                vee = match.get("versionEndExcluding")

                # Build human-readable range summary
                range_parts: list[str] = []
                if vsi:
                    range_parts.append(f">= {vsi}")
                if vse:
                    range_parts.append(f"> {vse}")
                if vei:
                    range_parts.append(f"<= {vei}")
                if vee:
                    range_parts.append(f"< {vee}")

                if range_parts:
                    range_summary = " && ".join(range_parts)
                elif version_exact and version_exact != "*":
                    range_summary = f"= {version_exact}"
                else:
                    range_summary = "all versions"

                results.append(
                    {
                        "source": "NVD",
                        "cpe23Uri": cpe,
                        "vendor": vendor,
                        "product": product,
                        "ecosystem": _cpe_to_ecosystem(cpe),
                        "versionStartIncluding": vsi,
                        "versionStartExcluding": vse,
                        "versionEndIncluding": vei,
                        "versionEndExcluding": vee,
                        "version_exact": version_exact if version_exact != "*" else None,
                        "vulnerable": match.get("vulnerable", True),
                        "range_summary": range_summary,
                    }
                )

    return results


# ---------------------------------------------------------------------------
# OSV parsing
# ---------------------------------------------------------------------------
def _parse_osv_affected(affected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse OSV affected entries into structured version-range records."""
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
            rtype = rng.get("type", "")
            if rtype:
                range_type = rtype
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

        # Build range summaries
        ranges: list[str] = []
        for i, intro in enumerate(introduced):
            fix = fixed[i] if i < len(fixed) else None
            la = last_affected[i] if i < len(last_affected) else None
            if intro == "0" and fix:
                ranges.append(f"< {fix}")
            elif fix:
                ranges.append(f">= {intro}, < {fix}")
            elif la:
                ranges.append(f">= {intro}, <= {la}")
            else:
                ranges.append(f">= {intro}")

        # Determine first patched version
        first_patched = _pick_first_patched(fixed)

        packages.append(
            {
                "source": "OSV",
                "ecosystem": _normalise_ecosystem(ecosystem),
                "package": name,
                "introduced": introduced,
                "fixed": fixed,
                "last_affected": last_affected,
                "range_type": range_type,
                "ranges": ranges,
                "range_summary": " || ".join(ranges) if ranges else "unknown",
                "affected_versions": versions[:50],
                "affected_version_count": len(versions),
                "first_patched_version": first_patched,
            }
        )
    return packages


def _pick_first_patched(fixed_versions: list[str]) -> str | None:
    """Return the earliest 'fixed' version, or None."""
    if not fixed_versions:
        return None
    if len(fixed_versions) == 1:
        return fixed_versions[0]

    # Try semver-ish sorting (major.minor.patch)
    def _version_key(v: str) -> tuple:
        parts = re.split(r"[.\-+]", v)
        key: list[int | str] = []
        for p in parts:
            try:
                key.append(int(p))
            except ValueError:
                key.append(p)
        return tuple(key)

    try:
        return sorted(fixed_versions, key=_version_key)[0]
    except (TypeError, ValueError):
        return fixed_versions[0]


# ---------------------------------------------------------------------------
# Fetch functions (public, used by CLI)
# ---------------------------------------------------------------------------
def fetch_nvd_version_ranges(cve_id: str) -> dict[str, Any]:
    """Fetch NVD CVE record and extract CPE version ranges."""
    url = f"{_NVD_CVE_URL}?cveId={cve_id.upper()}"
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY", "").strip()
    if api_key:
        headers["apiKey"] = api_key

    try:
        resp = _get_with_retry(url, headers=headers)
    except Exception as exc:
        return {"found": False, "error": f"NVD request failed: {exc}", "ranges": []}

    if resp.status_code == 404:
        return {"found": False, "error": f"No NVD record for {cve_id}", "ranges": []}

    try:
        resp.raise_for_status()
        data = resp.json()
    except (requests.exceptions.HTTPError, ValueError) as exc:
        return {"found": False, "error": f"NVD response error: {exc}", "ranges": []}

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return {"found": False, "error": f"No vulnerability data for {cve_id}", "ranges": []}

    cve_data = vulns[0].get("cve", {})
    configurations = cve_data.get("configurations", [])
    ranges = _parse_nvd_configurations(configurations)

    # Extract description for context
    descriptions = cve_data.get("descriptions", [])
    desc_en = ""
    for d in descriptions:
        if isinstance(d, dict) and d.get("lang") == "en":
            desc_en = d.get("value", "")
            break

    return {
        "found": True,
        "cve_id": cve_id,
        "description": desc_en[:300] if desc_en else "",
        "ranges": ranges,
        "configuration_count": len(configurations),
    }


def fetch_osv_version_ranges(cve_id: str) -> dict[str, Any]:
    """Fetch OSV.dev record(s) and extract ecosystem version ranges."""
    url = _OSV_VULN_URL.format(osv_id=cve_id)

    try:
        resp = _get_with_retry(url, headers={"Accept": "application/json"})
    except Exception as exc:
        return {"found": False, "error": f"OSV request failed: {exc}", "packages": []}

    if resp.status_code == 404:
        return {"found": False, "error": f"No OSV record for {cve_id}", "packages": []}

    try:
        resp.raise_for_status()
        primary = resp.json()
    except (requests.exceptions.HTTPError, ValueError) as exc:
        return {"found": False, "error": f"OSV response error: {exc}", "packages": []}

    all_packages: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _process_record(record: dict[str, Any]) -> None:
        rid = record.get("id", "")
        if rid in seen_ids:
            return
        seen_ids.add(rid)
        pkgs = _parse_osv_affected(record.get("affected", []))
        all_packages.extend(pkgs)

    _process_record(primary)

    # Follow GHSA aliases if primary lacks package data
    if not all_packages:
        aliases = primary.get("aliases", []) or []
        ghsa_aliases = [a for a in aliases if isinstance(a, str) and a.startswith("GHSA-")]
        for alias in ghsa_aliases[:_OSV_MAX_ALIAS_FOLLOWS]:
            try:
                alias_url = _OSV_VULN_URL.format(osv_id=alias)
                a_resp = _get_with_retry(alias_url, headers={"Accept": "application/json"})
                if a_resp.status_code == 200:
                    _process_record(a_resp.json())
            except Exception:
                continue

    aliases = sorted({a for a in (primary.get("aliases", []) or []) if isinstance(a, str)})

    return {
        "found": True if all_packages else bool(primary),
        "cve_id": cve_id,
        "aliases": aliases,
        "packages": all_packages,
    }


def fetch_version_range(cve_id: str, ecosystem: str = "auto") -> dict[str, Any]:
    """Combine NVD and OSV data into a unified version-range report.

    This is the main entry point used by both the Strands tool and the CLI.
    """
    cve_id = (cve_id or "").strip().upper()
    if not cve_id or not re.match(r"^CVE-\d{4}-\d+$", cve_id):
        return {
            "cve_id": cve_id,
            "error": "Invalid CVE ID format. Expected CVE-YYYY-NNNNN.",
            "nvd": {"found": False, "ranges": []},
            "osv": {"found": False, "packages": []},
            "packages": [],
            "summary": {"total_packages": 0, "ecosystems": []},
        }

    eco_filter = _normalise_ecosystem(ecosystem) if ecosystem and ecosystem != "auto" else None

    nvd_data = fetch_nvd_version_ranges(cve_id)
    osv_data = fetch_osv_version_ranges(cve_id)

    # Merge into unified package list
    packages: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    # OSV packages are more actionable (concrete ecosystem + versions)
    for pkg in osv_data.get("packages", []):
        eco = pkg.get("ecosystem", "")
        name = pkg.get("package", "")
        if eco_filter and eco != eco_filter:
            continue
        key = (eco.lower(), name.lower())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        packages.append(
            {
                "ecosystem": eco,
                "package": name,
                "source": "OSV",
                "vulnerable_range": pkg.get("range_summary", "unknown"),
                "introduced": pkg.get("introduced", []),
                "fixed": pkg.get("fixed", []),
                "last_affected": pkg.get("last_affected", []),
                "first_patched_version": pkg.get("first_patched_version"),
                "affected_versions": pkg.get("affected_versions", []),
                "affected_version_count": pkg.get("affected_version_count", 0),
            }
        )

    # NVD CPE ranges fill gaps not covered by OSV
    for rng in nvd_data.get("ranges", []):
        if not rng.get("vulnerable", True):
            continue
        eco = rng.get("ecosystem", "")
        product = rng.get("product", "")
        if eco_filter and eco and eco != eco_filter:
            continue
        key = (eco.lower(), product.lower())
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Determine first patched from versionEndExcluding
        first_patched = rng.get("versionEndExcluding")

        packages.append(
            {
                "ecosystem": eco or "CPE",
                "package": f"{rng.get('vendor', '')}/{product}" if rng.get("vendor") else product,
                "source": "NVD",
                "vulnerable_range": rng.get("range_summary", "unknown"),
                "cpe23Uri": rng.get("cpe23Uri", ""),
                "first_patched_version": first_patched,
                "affected_versions": [],
                "affected_version_count": 0,
            }
        )

    ecosystems = sorted({p["ecosystem"] for p in packages if p.get("ecosystem")})

    # Global first_patched (from OSV preferably)
    first_patched_global = None
    for pkg in packages:
        fp = pkg.get("first_patched_version")
        if fp:
            first_patched_global = fp
            break

    return {
        "cve_id": cve_id,
        "description": nvd_data.get("description", ""),
        "nvd": {
            "found": nvd_data.get("found", False),
            "configuration_count": nvd_data.get("configuration_count", 0),
            "ranges": nvd_data.get("ranges", []),
        },
        "osv": {
            "found": osv_data.get("found", False),
            "aliases": osv_data.get("aliases", []),
            "packages": osv_data.get("packages", []),
        },
        "packages": packages,
        "first_patched_version": first_patched_global,
        "summary": {
            "total_packages": len(packages),
            "ecosystems": ecosystems,
            "has_osv_data": bool(osv_data.get("packages")),
            "has_nvd_cpe_data": bool(nvd_data.get("ranges")),
        },
    }


# ---------------------------------------------------------------------------
# Strands tool entry point
# ---------------------------------------------------------------------------
def get_version_range(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands SDK tool entry point."""
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
    status = "success" if payload.get("packages") else "error"
    if "error" in payload:
        status = "error"

    result = {
        "toolUseId": tool_use_id,
        "status": status,
        "content": [{"text": json.dumps(payload, indent=2)}],
    }
    log_tool_output_size("get_version_range", result)
    return result
