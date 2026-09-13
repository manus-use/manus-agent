#!/usr/bin/env python3
"""
Tool for resolving affected version ranges for a CVE.

Walks NVD CPE configurations and cross-references OSV.dev ecosystem-specific
advisories (GHSA, PyPA, npm, Go, RustSec, Maven, etc.) to produce structured
vulnerable version ranges, a list of affected packages, and first-patched
release information.

Sources:
  1. **NVD CPE match criteria** — ``configurations[].nodes[].cpeMatch[]``
     carries ``versionStartIncluding``, ``versionEndExcluding``, etc.
  2. **OSV.dev** — ``affected[].ranges[].events[]`` carries ``introduced``
     and ``fixed`` per ecosystem+package tuple, which is more actionable
     for dependency-level decisions.

The two are complementary: NVD is authoritative for CPE-centric analysis
(product/vendor), while OSV is authoritative for ecosystem-level package
management (pip/npm/go).

Public APIs — no key required (NVD_API_KEY accepted for higher rate limits).
"""

from __future__ import annotations

import json as _json
import os
import re
import time
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# Retry / back-off (mirrors project conventions)
# ---------------------------------------------------------------------------
_MAX_RETRIES: int = int(os.environ.get("VR_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY: float = float(os.environ.get("VR_RETRY_BASE_DELAY", "1.0"))
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_TIMEOUT = 15

# ---------------------------------------------------------------------------
# Ecosystem mapping: CPE vendor/product → ecosystem hint
# ---------------------------------------------------------------------------
_CPE_ECOSYSTEM_HINTS: dict[str, str] = {
    "python": "PyPI",
    "pypi": "PyPI",
    "pip": "PyPI",
    "django": "PyPI",
    "flask": "PyPI",
    "numpy": "PyPI",
    "requests": "PyPI",
    "node.js": "npm",
    "nodejs": "npm",
    "npm": "npm",
    "express": "npm",
    "lodash": "npm",
    "next.js": "npm",
    "webpack": "npm",
    "maven": "Maven",
    "apache": "Maven",
    "spring": "Maven",
    "log4j": "Maven",
    "golang": "Go",
    "go": "Go",
    "kubernetes": "Go",
    "rust": "crates.io",
    "cargo": "crates.io",
    "rubygems": "RubyGems",
    "ruby": "RubyGems",
    "nuget": "NuGet",
    ".net": "NuGet",
}

# Normalise user-facing ecosystem flags to OSV ecosystem names.
_ECOSYSTEM_ALIASES: dict[str, str] = {
    "auto": "auto",
    "pypi": "PyPI",
    "npm": "npm",
    "maven": "Maven",
    "go": "Go",
    "crates.io": "crates.io",
    "rubygems": "RubyGems",
    "nuget": "NuGet",
}

TOOL_SPEC = {
    "name": "get_version_range",
    "description": (
        "Resolves affected version ranges for a CVE by combining NVD CPE "
        "configurations with OSV.dev ecosystem-specific advisories. Returns "
        "structured vulnerable version ranges, affected packages, and "
        "first-patched release information per ecosystem. Supports filtering "
        "by ecosystem (auto/pypi/npm/maven/go/crates.io). Use this to answer "
        "'which versions are vulnerable?' and 'what version should I upgrade to?'."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "CVE identifier (e.g. 'CVE-2021-44228').",
                },
                "ecosystem": {
                    "type": "string",
                    "description": (
                        "Filter results to a specific ecosystem. "
                        "One of: auto, pypi, npm, maven, go, crates.io. "
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
def _get_with_retry(
    url: str,
    *,
    timeout: int = _TIMEOUT,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    """GET with exponential back-off on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            time.sleep(_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
        try:
            resp = requests.get(url, timeout=timeout, headers=headers or {})
            if resp.status_code in _RETRYABLE_STATUSES and attempt < _MAX_RETRIES:
                last_exc = requests.exceptions.HTTPError(f"HTTP {resp.status_code}", response=resp)
                continue
            return resp
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
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
def _build_nvd_headers() -> dict[str, str]:
    """Inject NVD_API_KEY when available."""
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY", "").strip()
    if api_key:
        headers["apiKey"] = api_key
    return headers


def _parse_cpe_uri(cpe23: str) -> dict[str, str]:
    """Extract vendor, product, version fields from a CPE 2.3 URI.

    A CPE 2.3 formatted string looks like:
    cpe:2.3:a:vendor:product:version:update:edition:language:sw_edition:target_sw:target_hw:other
    """
    parts = cpe23.split(":")
    return {
        "cpe": cpe23,
        "part": parts[2] if len(parts) > 2 else "*",
        "vendor": parts[3] if len(parts) > 3 else "*",
        "product": parts[4] if len(parts) > 4 else "*",
        "version": parts[5] if len(parts) > 5 else "*",
    }


def _extract_cpe_ranges(configurations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Walk NVD ``configurations[].nodes[].cpeMatch[]`` and extract version ranges.

    Returns a list of dicts, each describing one CPE match with its version
    constraints (versionStartIncluding/Excluding, versionEndIncluding/Excluding).
    """
    ranges: list[dict[str, Any]] = []
    for config in configurations or []:
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                if not match.get("vulnerable", False):
                    continue
                cpe_info = _parse_cpe_uri(match.get("criteria", ""))
                entry: dict[str, Any] = {
                    "vendor": cpe_info["vendor"],
                    "product": cpe_info["product"],
                    "exact_version": cpe_info["version"] if cpe_info["version"] not in ("*", "-") else None,
                }
                # Version range boundaries
                for key in (
                    "versionStartIncluding",
                    "versionStartExcluding",
                    "versionEndIncluding",
                    "versionEndExcluding",
                ):
                    if key in match:
                        entry[_camel_to_snake(key)] = match[key]

                # Build a human-readable range string
                entry["range_expression"] = _build_range_expression(entry)
                ranges.append(entry)
    return ranges


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    s1 = re.sub(r"([A-Z])", r"_\1", name)
    return s1.lower().lstrip("_")


def _build_range_expression(entry: dict[str, Any]) -> str:
    """Build a human-readable version range string from CPE match bounds."""
    parts: list[str] = []
    if entry.get("exact_version"):
        return f"== {entry['exact_version']}"
    start_inc = entry.get("version_start_including")
    start_exc = entry.get("version_start_excluding")
    end_inc = entry.get("version_end_including")
    end_exc = entry.get("version_end_excluding")

    if start_inc:
        parts.append(f">= {start_inc}")
    elif start_exc:
        parts.append(f"> {start_exc}")

    if end_inc:
        parts.append(f"<= {end_inc}")
    elif end_exc:
        parts.append(f"< {end_exc}")

    if not parts:
        return "all versions"
    return ", ".join(parts)


def _guess_ecosystem_from_cpe(vendor: str, product: str) -> str | None:
    """Heuristic: map CPE vendor/product to a likely ecosystem."""
    for key in (vendor.lower(), product.lower()):
        if key in _CPE_ECOSYSTEM_HINTS:
            return _CPE_ECOSYSTEM_HINTS[key]
    return None


def fetch_nvd_ranges(cve_id: str) -> dict[str, Any]:
    """Fetch NVD data and extract CPE version ranges.

    Returns a dict with ``found``, ``cpe_ranges`` (list), and ``descriptions``.
    """
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id.upper()}"
    try:
        resp = _get_with_retry(url, headers=_build_nvd_headers())
    except Exception as exc:
        return {
            "found": False,
            "cpe_ranges": [],
            "descriptions": [],
            "error": f"NVD request failed: {exc}",
        }

    if resp.status_code == 404:
        return {"found": False, "cpe_ranges": [], "descriptions": []}
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        return {
            "found": False,
            "cpe_ranges": [],
            "descriptions": [],
            "error": f"NVD HTTP error: {exc}",
        }

    try:
        data = resp.json()
    except ValueError:
        return {
            "found": False,
            "cpe_ranges": [],
            "descriptions": [],
            "error": "NVD returned invalid JSON",
        }

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return {"found": False, "cpe_ranges": [], "descriptions": []}

    cve_data = vulns[0].get("cve", {})
    configs = cve_data.get("configurations", [])
    cpe_ranges = _extract_cpe_ranges(configs)

    # Annotate with ecosystem hints
    for r in cpe_ranges:
        r["ecosystem_hint"] = _guess_ecosystem_from_cpe(r.get("vendor", ""), r.get("product", ""))

    # Extract English description
    descriptions = []
    for desc in cve_data.get("descriptions", []):
        if desc.get("lang") == "en":
            descriptions.append(desc.get("value", ""))

    return {
        "found": True,
        "cpe_ranges": cpe_ranges,
        "descriptions": descriptions,
    }


# ---------------------------------------------------------------------------
# OSV.dev ecosystem-specific version ranges
# ---------------------------------------------------------------------------
_OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{osv_id}"
_OSV_MAX_ALIAS_FOLLOWS = 8


def _osv_get(osv_id: str) -> requests.Response:
    """Fetch a single OSV record by ID with retry."""
    url = _OSV_VULN_URL.format(osv_id=osv_id)
    return _get_with_retry(url, headers={"Accept": "application/json"})


def _parse_osv_ranges(affected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse OSV ``affected[]`` into structured version-range records."""
    results: list[dict[str, Any]] = []
    for entry in affected or []:
        if not isinstance(entry, dict):
            continue
        pkg = entry.get("package") or {}
        ecosystem = pkg.get("ecosystem") if isinstance(pkg, dict) else None
        name = pkg.get("name") if isinstance(pkg, dict) else None
        if ecosystem is None and name is None:
            continue

        range_records: list[dict[str, Any]] = []
        for rng in entry.get("ranges", []) or []:
            if not isinstance(rng, dict):
                continue
            rng_type = rng.get("type", "UNSPECIFIED")
            introduced: list[str] = []
            fixed: list[str] = []
            last_affected: list[str] = []
            for ev in rng.get("events", []) or []:
                if not isinstance(ev, dict):
                    continue
                if "introduced" in ev:
                    introduced.append(str(ev["introduced"]))
                if "fixed" in ev:
                    fixed.append(str(ev["fixed"]))
                if "last_affected" in ev:
                    last_affected.append(str(ev["last_affected"]))
            if introduced or fixed or last_affected:
                range_records.append(
                    {
                        "type": rng_type,
                        "introduced": introduced,
                        "fixed": fixed,
                        "last_affected": last_affected,
                    }
                )

        # Enumerate explicit affected versions (sample)
        versions = [str(v) for v in (entry.get("versions", []) or []) if v is not None]

        # Determine first patched version across all ranges
        all_fixed = [f for rr in range_records for f in rr["fixed"]]
        first_patched = _pick_first_patched(all_fixed) if all_fixed else None

        results.append(
            {
                "ecosystem": ecosystem,
                "package": name,
                "ranges": range_records,
                "first_patched": first_patched,
                "affected_version_count": len(versions),
                "affected_versions_sample": versions[:20],
            }
        )
    return results


def _pick_first_patched(versions: list[str]) -> str:
    """Return the lowest version string from a list of fix versions.

    Uses a simple heuristic sort — splits on dots and compares segments
    numerically where possible. Falls back to lexicographic comparison.
    """
    if not versions:
        return ""

    def _sort_key(v: str) -> tuple[int | float, ...]:
        parts: list[int | float] = []
        for seg in re.split(r"[.\-+]", v):
            try:
                parts.append(int(seg))
            except ValueError:
                # Non-numeric segment → push to end
                parts.append(float("inf"))
        return tuple(parts)

    try:
        return sorted(versions, key=_sort_key)[0]
    except Exception:
        return sorted(versions)[0]


def fetch_osv_ranges(
    cve_id: str,
    ecosystem_filter: str | None = None,
) -> dict[str, Any]:
    """Fetch OSV.dev records for a CVE and extract version ranges.

    Returns a dict with ``found``, ``packages`` (list of per-package
    version-range records), and ``aliases``.
    """
    cve_id = (cve_id or "").strip().upper()
    if not cve_id:
        return {
            "found": False,
            "packages": [],
            "aliases": [],
            "error": "Invalid CVE ID. Must be a non-empty string.",
        }
    try:
        resp = _osv_get(cve_id)
    except Exception as exc:
        return {
            "found": False,
            "packages": [],
            "aliases": [],
            "error": f"OSV request failed: {exc}",
        }

    if resp.status_code == 404:
        return {
            "found": False,
            "packages": [],
            "aliases": [],
            "message": f"No OSV.dev record for {cve_id}.",
        }

    try:
        resp.raise_for_status()
        primary = resp.json()
    except (requests.exceptions.HTTPError, ValueError) as exc:
        return {
            "found": False,
            "packages": [],
            "aliases": [],
            "error": f"OSV response error: {exc}",
        }

    # Collect all records (primary + GHSA alias follow)
    all_affected: list[dict[str, Any]] = list(primary.get("affected", []) or [])
    aliases: list[str] = list(primary.get("aliases", []) or [])

    # If primary has no affected data, follow GHSA aliases
    if not all_affected:
        ghsa_aliases = [a for a in aliases if isinstance(a, str) and a.startswith("GHSA-")]
        seen: set[str] = {cve_id}
        for alias in ghsa_aliases[:_OSV_MAX_ALIAS_FOLLOWS]:
            if alias in seen:
                continue
            seen.add(alias)
            try:
                a_resp = _osv_get(alias)
                if a_resp.status_code == 200:
                    a_data = a_resp.json()
                    all_affected.extend(a_data.get("affected", []) or [])
                    for a in a_data.get("aliases", []) or []:
                        if a not in aliases:
                            aliases.append(a)
            except Exception:  # noqa: BLE001
                continue

    packages = _parse_osv_ranges(all_affected)

    # Apply ecosystem filter
    if ecosystem_filter and ecosystem_filter != "auto":
        norm = _ECOSYSTEM_ALIASES.get(ecosystem_filter.lower(), ecosystem_filter)
        packages = [p for p in packages if (p.get("ecosystem") or "").lower() == norm.lower()]

    return {
        "found": bool(packages),
        "packages": packages,
        "aliases": aliases,
    }


# ---------------------------------------------------------------------------
# Combined analysis
# ---------------------------------------------------------------------------
def fetch_version_range(
    cve_id: str,
    ecosystem: str = "auto",
) -> dict[str, Any]:
    """Combine NVD CPE ranges and OSV.dev ecosystem ranges into one result.

    This is the main entry point used by both the Strands tool and the CLI.
    """
    cve_id = (cve_id or "").strip().upper()
    if not cve_id or not cve_id.startswith("CVE-"):
        return {
            "cve_id": cve_id,
            "found": False,
            "error": "Invalid CVE ID format. Must start with 'CVE-'.",
        }

    # Fetch from both sources in sequence (public APIs, no parallelism needed)
    nvd_result = fetch_nvd_ranges(cve_id)
    osv_result = fetch_osv_ranges(cve_id, ecosystem_filter=ecosystem)

    cpe_ranges = nvd_result.get("cpe_ranges", [])
    osv_packages = osv_result.get("packages", [])
    descriptions = nvd_result.get("descriptions", [])

    found = nvd_result.get("found", False) or osv_result.get("found", False)

    # Collect unique ecosystems
    ecosystems_seen: set[str] = set()
    for pkg in osv_packages:
        eco = pkg.get("ecosystem")
        if eco:
            ecosystems_seen.add(eco)
    for r in cpe_ranges:
        hint = r.get("ecosystem_hint")
        if hint:
            ecosystems_seen.add(hint)

    # Determine first patched version (prefer OSV, more precise)
    first_patched: dict[str, str] = {}
    for pkg in osv_packages:
        fp = pkg.get("first_patched")
        if fp:
            key = f"{pkg.get('ecosystem', '?')}:{pkg.get('package', '?')}"
            if key not in first_patched:
                first_patched[key] = fp

    # Build summary
    total_osv_packages = len(osv_packages)

    if ecosystem != "auto":
        eco_label = _ECOSYSTEM_ALIASES.get(ecosystem.lower(), ecosystem)
    else:
        eco_label = None

    # Filter CPE ranges by ecosystem hint when a filter is active
    if eco_label and eco_label != "auto":
        cpe_ranges = [r for r in cpe_ranges if (r.get("ecosystem_hint") or "").lower() == eco_label.lower()]

    # Human-readable message
    parts: list[str] = []
    if cpe_ranges:
        parts.append(f"{len(cpe_ranges)} NVD CPE range(s)")
    if osv_packages:
        parts.append(f"{total_osv_packages} OSV package(s) across {len(ecosystems_seen)} ecosystem(s)")
    if not parts:
        msg = f"No version range data found for {cve_id}."
        if eco_label and eco_label != "auto":
            msg += f" (filtered to {eco_label})"
    else:
        msg = f"{cve_id}: {'; '.join(parts)}."
        if first_patched:
            fp_list = [f"{k} → {v}" for k, v in sorted(first_patched.items())]
            msg += f" First patched: {', '.join(fp_list[:5])}."

    # Collect errors from sub-sources (non-fatal)
    errors: list[str] = []
    if nvd_result.get("error"):
        errors.append(nvd_result["error"])
    if osv_result.get("error"):
        errors.append(osv_result["error"])

    return {
        "cve_id": cve_id,
        "found": found,
        "description": descriptions[0] if descriptions else None,
        "ecosystems": sorted(ecosystems_seen),
        "nvd_cpe_ranges": cpe_ranges,
        "osv_packages": osv_packages,
        "first_patched": first_patched,
        "message": msg,
        "errors": errors if errors else None,
    }


# ---------------------------------------------------------------------------
# Strands tool entry point
# ---------------------------------------------------------------------------
def get_version_range(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands SDK tool handler."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool.get("input", {}) or {}
    cve_id = tool_input.get("cve_id")
    ecosystem = tool_input.get("ecosystem", "auto") or "auto"

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
        "content": [{"text": _json.dumps(payload, indent=2)}],
    }
    log_tool_output_size("get_version_range", result)
    return result
