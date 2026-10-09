#!/usr/bin/env python3
"""
Tool for resolving affected version ranges for a CVE.

Walks NVD CPE configurations and cross-references OSV.dev ecosystem
databases (PyPI, npm, Maven, Go, RubyGems, crates.io, NuGet, Packagist,
etc.) to produce structured vulnerable version ranges, a list of known
affected releases, and the first patched release per package.

Data sources (in resolution order):
1. **NVD CVE 2.0 API** — CPE match criteria with versionStart/End bounds.
2. **OSV.dev** — ecosystem-native affected ranges, SEMVER/ECOSYSTEM events,
   and enumerated affected version lists per package.

The tool answers: *which packages and exact version ranges are affected,
and what is the first version that fixes the vulnerability?*
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
_RETRY_BASE_DELAY: float = float(os.environ.get("VERSION_RANGE_RETRY_DELAY", "1.0"))
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_HTTP_TIMEOUT: int = 15

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
_NVD_CVE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{osv_id}"
_OSV_TIMEOUT = 15

# Cap alias follows to bound fan-out on pathological CVEs.
_MAX_ALIAS_FOLLOWS = 8

# ---------------------------------------------------------------------------
# Ecosystem normalisation map (CPE vendor/product → ecosystem hint)
# ---------------------------------------------------------------------------
_CPE_ECOSYSTEM_HINTS: dict[str, str] = {
    "python": "PyPI",
    "pip": "PyPI",
    "pypi": "PyPI",
    "django": "PyPI",
    "flask": "PyPI",
    "numpy": "PyPI",
    "node.js": "npm",
    "nodejs": "npm",
    "npm": "npm",
    "express": "npm",
    "lodash": "npm",
    "maven": "Maven",
    "apache": "Maven",
    "spring": "Maven",
    "log4j": "Maven",
    "golang": "Go",
    "go": "Go",
    "rust": "crates.io",
    "cargo": "crates.io",
    "rubygems": "RubyGems",
    "ruby": "RubyGems",
    "nuget": "NuGet",
    ".net": "NuGet",
    "packagist": "Packagist",
    "php": "Packagist",
    "composer": "Packagist",
}

# Canonical ecosystem names accepted by --ecosystem filter.
_ECOSYSTEM_ALIASES: dict[str, str] = {
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
# Strands TOOL_SPEC
# ---------------------------------------------------------------------------
TOOL_SPEC = {
    "name": "get_version_range",
    "description": (
        "Resolves a CVE to its affected packages and version ranges by walking NVD CPE "
        "configurations and cross-referencing OSV.dev ecosystem databases (PyPI, npm, Maven, "
        "Go, RubyGems, crates.io, NuGet, Packagist). Returns per-package: ecosystem, package "
        "name, vulnerable version range, list of known affected versions, and the first patched "
        "version. Optionally filters results to a single ecosystem. Use this to answer 'which "
        "exact versions are affected and what version fixes it?'."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier to look up (e.g., 'CVE-2021-44228').",
                },
                "ecosystem": {
                    "type": "string",
                    "description": (
                        "Filter results to a specific ecosystem. "
                        "Accepted values: auto, pypi, npm, maven, go, crates.io, rubygems, nuget, packagist. "
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
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = _HTTP_TIMEOUT,
) -> requests.Response:
    """GET *url* with exponential back-off on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            time.sleep(_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code in _RETRYABLE_STATUSES:
                last_exc = requests.exceptions.HTTPError(f"HTTP {resp.status_code}", response=resp)
                if attempt < _MAX_RETRIES:
                    continue
                raise last_exc
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code not in _RETRYABLE_STATUSES:
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


def _nvd_headers() -> dict[str, str]:
    """Inject NVD_API_KEY if available."""
    headers: dict[str, str] = {}
    key = os.environ.get("NVD_API_KEY", "").strip()
    if key:
        headers["apiKey"] = key
    return headers


# ---------------------------------------------------------------------------
# Source 1: NVD CPE configurations → version ranges
# ---------------------------------------------------------------------------


def _parse_cpe_uri(cpe_uri: str) -> dict[str, str]:
    """Extract vendor, product, version from a CPE 2.3 URI."""
    parts = cpe_uri.split(":")
    return {
        "vendor": parts[3] if len(parts) > 3 else "",
        "product": parts[4] if len(parts) > 4 else "",
        "version": parts[5] if len(parts) > 5 else "*",
    }


def _guess_ecosystem_from_cpe(vendor: str, product: str) -> str:
    """Best-effort ecosystem guess from CPE vendor/product names."""
    for token in (vendor.lower(), product.lower()):
        for hint_key, eco in _CPE_ECOSYSTEM_HINTS.items():
            if hint_key in token:
                return eco
    return "unknown"


def _build_version_constraint(cpe_match: dict[str, Any]) -> str:
    """Build a human-readable version constraint string from CPE match."""
    ver_start_inc = cpe_match.get("versionStartIncluding", "")
    ver_start_exc = cpe_match.get("versionStartExcluding", "")
    ver_end_exc = cpe_match.get("versionEndExcluding", "")
    ver_end_inc = cpe_match.get("versionEndIncluding", "")

    parts: list[str] = []
    if ver_start_inc:
        parts.append(f">={ver_start_inc}")
    elif ver_start_exc:
        parts.append(f">{ver_start_exc}")

    if ver_end_exc:
        parts.append(f"<{ver_end_exc}")
    elif ver_end_inc:
        parts.append(f"<={ver_end_inc}")

    if parts:
        return ", ".join(parts)

    # Fallback: exact version from the CPE URI itself.
    cpe_uri = cpe_match.get("criteria", "")
    parsed = _parse_cpe_uri(cpe_uri)
    ver = parsed["version"]
    return ver if ver != "*" else "all versions"


def _fetch_nvd_version_ranges(cve_id: str) -> list[dict[str, Any]]:
    """Walk NVD CPE configurations and return per-product version ranges."""
    try:
        resp = _get_with_retry(
            _NVD_CVE_URL,
            params={"cveId": cve_id},
            headers=_nvd_headers(),
        )
        data = resp.json()
    except Exception:
        return []

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return []

    cve_data = vulns[0].get("cve", {})
    configs = cve_data.get("configurations", [])
    results: list[dict[str, Any]] = []

    for config in configs:
        operator = config.get("operator", "OR")
        for node in config.get("nodes", []):
            node_operator = node.get("operator", "OR")
            for cpe_match in node.get("cpeMatch", []):
                if not cpe_match.get("vulnerable", False):
                    continue
                cpe_uri = cpe_match.get("criteria", "")
                parsed = _parse_cpe_uri(cpe_uri)

                constraint = _build_version_constraint(cpe_match)
                ecosystem = _guess_ecosystem_from_cpe(parsed["vendor"], parsed["product"])

                # Determine first_patched from versionEndExcluding (common pattern).
                first_patched = cpe_match.get("versionEndExcluding", "")

                results.append(
                    {
                        "source": "nvd",
                        "vendor": parsed["vendor"],
                        "product": parsed["product"],
                        "ecosystem": ecosystem,
                        "vulnerable_range": constraint,
                        "first_patched": first_patched,
                        "affected_versions": [],
                        "cpe": cpe_uri,
                        "operator": f"{operator}/{node_operator}",
                    }
                )

    return results


# ---------------------------------------------------------------------------
# Source 2: OSV.dev — ecosystem-native ranges + enumerated versions
# ---------------------------------------------------------------------------


def _summarise_osv_range(rng: dict[str, Any]) -> dict[str, str]:
    """Extract introduced/fixed from a single OSV range object."""
    introduced = ""
    fixed = ""
    last_affected = ""
    for event in rng.get("events", []):
        if "introduced" in event:
            introduced = str(event["introduced"])
        if "fixed" in event:
            fixed = str(event["fixed"])
        if "last_affected" in event:
            last_affected = str(event["last_affected"])

    parts: list[str] = []
    if introduced and introduced != "0":
        parts.append(f">={introduced}")
    elif introduced == "0":
        parts.append(">=0")
    if fixed:
        parts.append(f"<{fixed}")
    elif last_affected:
        parts.append(f"<={last_affected}")

    return {
        "range": ", ".join(parts) if parts else "all versions",
        "introduced": introduced,
        "fixed": fixed,
        "last_affected": last_affected,
    }


def _fetch_osv_single(osv_id: str) -> dict[str, Any] | None:
    """Fetch a single OSV record by ID. Returns None on failure."""
    try:
        resp = _get_with_retry(
            _OSV_VULN_URL.format(osv_id=osv_id),
            timeout=_OSV_TIMEOUT,
        )
        return resp.json()
    except Exception:
        return None


def _fetch_osv_version_ranges(cve_id: str) -> list[dict[str, Any]]:
    """Query OSV.dev for the CVE, follow GHSA aliases, return per-package ranges."""
    # Fetch the CVE record itself.
    record = _fetch_osv_single(cve_id)
    if record is None:
        return []

    # Collect all records to process (CVE + GHSA aliases).
    all_records: dict[str, dict[str, Any]] = {record.get("id", cve_id): record}
    aliases = record.get("aliases", [])
    followed = 0
    for alias in aliases:
        if followed >= _MAX_ALIAS_FOLLOWS:
            break
        if alias.startswith("GHSA-"):
            alias_record = _fetch_osv_single(alias)
            if alias_record:
                all_records[alias] = alias_record
                followed += 1

    results: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for rec_id, rec in all_records.items():
        for affected in rec.get("affected", []):
            pkg = affected.get("package", {})
            name = pkg.get("name", "")
            ecosystem = pkg.get("ecosystem", "")
            if not name:
                continue

            # Deduplicate by (ecosystem, name).
            dedup_key = f"{ecosystem}:{name}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            # Collect ranges.
            ranges = affected.get("ranges", [])
            range_summaries: list[dict[str, str]] = []
            first_patched = ""
            last_affected_ver = ""

            for rng in ranges:
                summary = _summarise_osv_range(rng)
                range_summaries.append(summary)
                if summary["fixed"] and not first_patched:
                    first_patched = summary["fixed"]
                if summary["last_affected"] and not last_affected_ver:
                    last_affected_ver = summary["last_affected"]

            # Combine range strings.
            if range_summaries:
                combined_range = " || ".join(s["range"] for s in range_summaries if s["range"])
            else:
                combined_range = "all versions"

            # Enumerated affected versions (capped).
            affected_versions = affected.get("versions", [])
            if len(affected_versions) > 100:
                affected_versions = affected_versions[:100]

            # Database-specific severity.
            severity_vectors: list[str] = []
            for sev in affected.get("severity", []) or rec.get("severity", []):
                if sev.get("score"):
                    severity_vectors.append(sev["score"])

            results.append(
                {
                    "source": "osv",
                    "osv_id": rec_id,
                    "ecosystem": ecosystem,
                    "package": name,
                    "vulnerable_range": combined_range,
                    "first_patched": first_patched,
                    "last_affected": last_affected_ver,
                    "affected_versions": affected_versions,
                    "range_details": range_summaries,
                    "severity": severity_vectors,
                }
            )

    return results


# ---------------------------------------------------------------------------
# Merge NVD + OSV results
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)


def _normalise_ecosystem(raw: str) -> str:
    """Map raw ecosystem string to a canonical name."""
    return _ECOSYSTEM_ALIASES.get(raw.lower(), raw)


def _merge_results(
    nvd: list[dict[str, Any]],
    osv: list[dict[str, Any]],
    ecosystem_filter: str,
) -> list[dict[str, Any]]:
    """Merge NVD and OSV results, preferring OSV for richer data.

    Returns a list of per-package result dicts.
    """
    merged: list[dict[str, Any]] = []
    osv_ecosystems: set[str] = set()

    # OSV results are richer (enumerated versions, first-fixed), add them first.
    for entry in osv:
        eco = entry.get("ecosystem", "")
        if ecosystem_filter and ecosystem_filter != "auto":
            canon = _normalise_ecosystem(ecosystem_filter)
            if _normalise_ecosystem(eco) != canon:
                continue
        osv_ecosystems.add(f"{eco}:{entry.get('package', '')}")
        merged.append(
            {
                "source": "osv",
                "ecosystem": eco,
                "package": entry.get("package", ""),
                "vulnerable_range": entry.get("vulnerable_range", ""),
                "first_patched": entry.get("first_patched", ""),
                "last_affected": entry.get("last_affected", ""),
                "affected_versions": entry.get("affected_versions", []),
                "range_details": entry.get("range_details", []),
                "severity": entry.get("severity", []),
            }
        )

    # Add NVD entries that aren't covered by OSV (different product).
    for entry in nvd:
        eco = entry.get("ecosystem", "unknown")
        product = entry.get("product", "")
        if ecosystem_filter and ecosystem_filter != "auto":
            canon = _normalise_ecosystem(ecosystem_filter)
            if _normalise_ecosystem(eco) != canon and eco != "unknown":
                continue

        # Check if OSV already covers this product.
        covered = any(product.lower() in key.lower() for key in osv_ecosystems)
        if covered:
            continue

        merged.append(
            {
                "source": "nvd",
                "ecosystem": eco,
                "package": f"{entry.get('vendor', '')}/{product}" if entry.get("vendor") else product,
                "vulnerable_range": entry.get("vulnerable_range", ""),
                "first_patched": entry.get("first_patched", ""),
                "last_affected": "",
                "affected_versions": [],
                "range_details": [],
                "severity": [],
                "cpe": entry.get("cpe", ""),
            }
        )

    return merged


# ---------------------------------------------------------------------------
# Strands handler
# ---------------------------------------------------------------------------


def get_version_range(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands tool handler for get_version_range."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id", "")
    ecosystem = tool_input.get("ecosystem", "auto")

    if not isinstance(cve_id, str) or not _CVE_RE.match(cve_id.strip()):
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid CVE ID format. Must be like 'CVE-2021-44228'."}],
        }
        log_tool_output_size("get_version_range", result)
        return result

    cve_id = cve_id.strip().upper()

    # Fetch from both sources.
    nvd_ranges = _fetch_nvd_version_ranges(cve_id)
    osv_ranges = _fetch_osv_version_ranges(cve_id)

    merged = _merge_results(nvd_ranges, osv_ranges, ecosystem or "auto")

    if not merged:
        result = {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [
                {
                    "text": (
                        f"No affected version range data found for {cve_id}. "
                        "The CVE may not yet have CPE configurations in NVD "
                        "or ecosystem-level records in OSV.dev."
                    )
                }
            ],
        }
        log_tool_output_size("get_version_range", result)
        return result

    # Build structured output.
    output: dict[str, Any] = {
        "cve_id": cve_id,
        "packages_affected": len(merged),
        "ecosystem_filter": ecosystem or "auto",
        "packages": [],
    }

    for entry in merged:
        pkg: dict[str, Any] = {
            "source": entry["source"],
            "ecosystem": entry["ecosystem"],
            "package": entry["package"],
            "vulnerable_range": entry["vulnerable_range"],
            "first_patched": entry.get("first_patched", ""),
        }
        if entry.get("last_affected"):
            pkg["last_affected"] = entry["last_affected"]
        if entry.get("affected_versions"):
            pkg["affected_versions_count"] = len(entry["affected_versions"])
            pkg["affected_versions"] = entry["affected_versions"][:50]
        if entry.get("severity"):
            pkg["severity"] = entry["severity"]
        if entry.get("cpe"):
            pkg["cpe"] = entry["cpe"]
        output["packages"].append(pkg)

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"json": output}],
    }
    log_tool_output_size("get_version_range", result)
    return result
