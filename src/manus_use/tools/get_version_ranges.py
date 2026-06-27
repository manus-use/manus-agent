"""
Tool for resolving affected and patched version ranges for a CVE.

Given a CVE identifier, this tool:
1. Fetches NVD CPE match data to extract the declared vulnerable version ranges.
2. Cross-references with PyPI, npm, or Maven Central release dates to map each
   version range onto the concrete published package versions.
3. Returns a structured report showing:
   - Which package/ecosystem is affected
   - The raw CPE version constraints (versionStartIncluding, versionEndExcluding, etc.)
   - Which concrete released versions fall inside the vulnerable range
   - Which released version first introduced the fix (the first patched release)

Only public packages on PyPI, npm, and Maven Central are supported; non-public or
unknown ecosystems return the raw CPE range without release-date cross-referencing.
"""

from __future__ import annotations

import re
from typing import Any

import requests
from packaging.version import InvalidVersion, Version
from strands.types.tools import ToolResult, ToolUse

from manus_use.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# CPE URI component indices (CPE 2.3 URI format)
# ---------------------------------------------------------------------------
# Example: cpe:2.3:a:vendor:product:version:...
_CPE23_PART = 2  # "a" = application, "o" = OS, "h" = hardware
_CPE23_VENDOR = 3
_CPE23_PRODUCT = 4
_CPE23_VERSION = 5

# Sentinel for "all versions" in CPE
_ANY = "*"
_NA = "-"

# Max versions to list per range (prevents flooding output for packages with 1000+ releases)
_MAX_VERSIONS_PER_RANGE = 30

TOOL_SPEC = {
    "name": "get_version_ranges",
    "description": (
        "Resolves the exact affected and patched version ranges for a CVE by parsing NVD CPE "
        "match data and cross-referencing with PyPI, npm, or Maven Central release dates. "
        "Returns a structured report: the vulnerable semver ranges declared in NVD, the concrete "
        "package releases that fall inside those ranges, and the first patched release. "
        "Use this after get_nvd_data when you need to answer 'which exact versions are affected?' "
        "for patch management, dependency auditing, or blast-radius analysis."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier to look up (e.g., 'CVE-2024-3094').",
                },
                "ecosystem": {
                    "type": "string",
                    "description": (
                        "Package ecosystem to cross-reference: 'pypi', 'npm', 'maven', or 'auto'. "
                        "Use 'auto' (the default) to let the tool infer the ecosystem from CPE data."
                    ),
                    "enum": ["auto", "pypi", "npm", "maven"],
                    "default": "auto",
                },
            },
            "required": ["cve_id"],
        }
    },
}


# ---------------------------------------------------------------------------
# NVD CPE helpers
# ---------------------------------------------------------------------------


def _fetch_nvd_cpe_ranges(cve_id: str) -> dict[str, Any]:
    """
    Query NVD API and extract CPE match ranges.

    Returns a dict with keys:
      - "configurations": list of raw CPE node dicts
      - "error": set when the request failed or CVE is unknown
    """
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id.upper()}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return {"error": f"NVD request failed: {exc}"}

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return {"error": f"No NVD record for {cve_id}"}

    cve_record = vulns[0].get("cve", {})
    configs = cve_record.get("configurations", [])
    description = ""
    for d in cve_record.get("descriptions", []):
        if d.get("lang") == "en":
            description = d.get("value", "")
            break

    return {"configurations": configs, "description": description}


def _parse_cpe23(cpe_str: str) -> dict[str, str]:
    """Parse a CPE 2.3 URI into a dict of named components."""
    parts = cpe_str.split(":")
    result: dict[str, str] = {}
    if len(parts) < 6:
        return result
    result["part"] = parts[_CPE23_PART] if len(parts) > _CPE23_PART else _ANY
    result["vendor"] = parts[_CPE23_VENDOR] if len(parts) > _CPE23_VENDOR else _ANY
    result["product"] = parts[_CPE23_PRODUCT] if len(parts) > _CPE23_PRODUCT else _ANY
    result["version"] = parts[_CPE23_VERSION] if len(parts) > _CPE23_VERSION else _ANY
    return result


def _extract_ranges(configurations: list[dict]) -> list[dict[str, Any]]:
    """
    Walk the NVD configuration nodes and collect version range constraints.

    Each returned dict has:
      - vendor, product, version (from CPE)
      - version_start_including, version_start_excluding (optional)
      - version_end_including, version_end_excluding (optional)
    """
    ranges: list[dict[str, Any]] = []

    def _walk(nodes: list[dict]) -> None:
        for node in nodes:
            for match in node.get("cpeMatch", []):
                if not match.get("vulnerable", False):
                    continue
                cpe_str = match.get("criteria", "")
                parsed = _parse_cpe23(cpe_str)
                if not parsed:
                    continue
                entry: dict[str, Any] = {
                    "vendor": parsed.get("vendor", _ANY),
                    "product": parsed.get("product", _ANY),
                    "cpe_version": parsed.get("version", _ANY),
                    "cpe_uri": cpe_str,
                }
                for key in (
                    "versionStartIncluding",
                    "versionStartExcluding",
                    "versionEndIncluding",
                    "versionEndExcluding",
                ):
                    val = match.get(key)
                    if val:
                        entry[key] = val
                ranges.append(entry)
            # Recurse into child nodes
            _walk(node.get("children", []))

    _walk(configurations)
    return ranges


# ---------------------------------------------------------------------------
# Ecosystem inference
# ---------------------------------------------------------------------------

# Known Python packages (vendor patterns that map to PyPI)
_PYPI_VENDORS = {
    "python",
    "pypi",
    "pip",
    "djangoproject",
    "palletsprojects",
    "sqlalchemy",
    "flask",
    "requests",
    "pillow",
    "cryptography",
    "jinja2",
    "pyyaml",
    "urllib3",
    "setuptools",
    "certifi",
    "paramiko",
    "aiohttp",
    "werkzeug",
    "twisted",
}

# Known npm packages
_NPM_VENDORS = {
    "nodejs",
    "npmjs",
    "npm",
    "expressjs",
    "lodash_project",
    "nestjs",
    "angularjs",
    "jquery",
    "moment",
    "axios",
    "webpack",
    "babel",
    "react",
    "vue",
    "angular",
    "electron",
    "next.js",
}

# Maven groupId patterns
_MAVEN_VENDORS = {
    "apache",
    "springframework",
    "log4j",
    "netty",
    "jackson-databind",
    "struts",
    "tomcat",
    "shiro",
    "commons",
    "hibernate",
    "elastic",
    "kafka",
    "zookeeper",
}


def _infer_ecosystem(ranges: list[dict[str, Any]]) -> str:
    """Guess the package ecosystem from the CPE vendor/product fields."""
    for r in ranges:
        vendor = r.get("vendor", "").lower()
        product = r.get("product", "").lower()
        for v in _PYPI_VENDORS:
            if v in vendor or v in product:
                return "pypi"
        for v in _NPM_VENDORS:
            if v in vendor or v in product:
                return "npm"
        for v in _MAVEN_VENDORS:
            if v in vendor or v in product:
                return "maven"
    return "unknown"


# ---------------------------------------------------------------------------
# PyPI release cross-reference
# ---------------------------------------------------------------------------


def _fetch_pypi_releases(package_name: str) -> list[tuple[str, str]]:
    """
    Return a sorted list of (version_str, release_date) tuples from PyPI.
    release_date is the ISO-8601 date of the first uploaded file for that version.
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []

    releases: list[tuple[str, str]] = []
    for ver_str, files in data.get("releases", {}).items():
        if not files:
            continue
        # Use the upload_time of the first file for this version
        upload_time = files[0].get("upload_time", "")
        if upload_time:
            releases.append((ver_str, upload_time[:10]))

    # Sort by parsed Version for correct semver ordering
    def _sort_key(item: tuple[str, str]) -> tuple[Any, str]:
        try:
            return (Version(item[0]), item[1])
        except InvalidVersion:
            return (Version("0.0.0"), item[1])

    releases.sort(key=_sort_key)
    return releases


# ---------------------------------------------------------------------------
# npm release cross-reference
# ---------------------------------------------------------------------------


def _fetch_npm_releases(package_name: str) -> list[tuple[str, str]]:
    """
    Return a sorted list of (version_str, release_date) from the npm registry.
    """
    url = f"https://registry.npmjs.org/{package_name}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []

    time_map: dict[str, str] = data.get("time", {})
    releases: list[tuple[str, str]] = []
    for ver_str, ts in time_map.items():
        if ver_str in ("created", "modified"):
            continue
        releases.append((ver_str, ts[:10]))

    def _sort_key(item: tuple[str, str]) -> tuple[Any, str]:
        try:
            return (Version(item[0]), item[1])
        except InvalidVersion:
            return (Version("0.0.0"), item[1])

    releases.sort(key=_sort_key)
    return releases


# ---------------------------------------------------------------------------
# Maven release cross-reference
# ---------------------------------------------------------------------------


def _fetch_maven_releases(group_id: str, artifact_id: str) -> list[tuple[str, str]]:
    """
    Query Maven Central search API for all released versions of a given GA.
    Returns a sorted list of (version_str, release_date).
    release_date is derived from the 'timestamp' field (milliseconds since epoch).
    """
    url = "https://search.maven.org/solrsearch/select"
    params = {
        "q": f"g:{group_id} AND a:{artifact_id}",
        "core": "gav",
        "rows": "200",
        "wt": "json",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []

    import datetime

    releases: list[tuple[str, str]] = []
    for doc in data.get("response", {}).get("docs", []):
        ver_str = doc.get("v", "")
        ts_ms = doc.get("timestamp", 0)
        if ver_str and ts_ms:
            dt = datetime.datetime.utcfromtimestamp(ts_ms / 1000)
            releases.append((ver_str, dt.strftime("%Y-%m-%d")))

    def _sort_key(item: tuple[str, str]) -> tuple[Any, str]:
        try:
            return (Version(item[0]), item[1])
        except InvalidVersion:
            return (Version("0.0.0"), item[1])

    releases.sort(key=_sort_key)
    return releases


# ---------------------------------------------------------------------------
# Version-range membership check
# ---------------------------------------------------------------------------


def _version_in_range(
    ver_str: str,
    vsi: str | None,
    vse: str | None,
    vei: str | None,
    vee: str | None,
    cpe_version: str,
) -> bool:
    """
    Return True if ver_str falls inside the CPE version range.

    Parameters
    ----------
    ver_str  : the concrete release version to test
    vsi      : versionStartIncluding  (>= constraint, or None)
    vse      : versionStartExcluding  (>  constraint, or None)
    vei      : versionEndIncluding    (<= constraint, or None)
    vee      : versionEndExcluding    (<  constraint, or None)
    cpe_version: the literal version field from the CPE URI (used when all
                 four range fields are absent — exact single-version match)
    """
    try:
        v = Version(ver_str)
    except InvalidVersion:
        return False

    # Exact-version CPE (no range fields, specific version in URI)
    if cpe_version not in (_ANY, _NA, "") and not any([vsi, vse, vei, vee]):
        try:
            return v == Version(cpe_version)
        except InvalidVersion:
            return ver_str == cpe_version

    # Range checks
    if vsi:
        try:
            if v < Version(vsi):
                return False
        except InvalidVersion:
            pass
    if vse:
        try:
            if v <= Version(vse):
                return False
        except InvalidVersion:
            pass
    if vei:
        try:
            if v > Version(vei):
                return False
        except InvalidVersion:
            pass
    if vee:
        try:
            if v >= Version(vee):
                return False
        except InvalidVersion:
            pass

    # If no constraints at all and cpe_version is wildcard, all versions match
    if not any([vsi, vse, vei, vee]) and cpe_version in (_ANY, _NA, ""):
        return True

    return True


# ---------------------------------------------------------------------------
# Public resolve function (used by both the Strands tool and the CLI)
# ---------------------------------------------------------------------------


def resolve_version_ranges(cve_id: str, ecosystem: str = "auto") -> dict[str, Any]:
    """
    Core logic: fetch NVD CPE data, infer ecosystem, cross-reference releases.

    Returns a rich dict suitable for both text rendering and JSON output.
    """
    nvd = _fetch_nvd_cpe_ranges(cve_id)
    if "error" in nvd:
        return {"cve_id": cve_id, "error": nvd["error"], "ranges": []}

    configs = nvd.get("configurations", [])
    description = nvd.get("description", "")
    raw_ranges = _extract_ranges(configs)

    if not raw_ranges:
        return {
            "cve_id": cve_id,
            "description": description,
            "error": "No CPE configuration data found in NVD — version ranges cannot be determined.",
            "ranges": [],
        }

    # Deduplicate ranges (same product may appear in multiple nodes)
    seen: set[str] = set()
    unique_ranges: list[dict[str, Any]] = []
    for r in raw_ranges:
        key = (
            r["vendor"],
            r["product"],
            r.get("versionStartIncluding", ""),
            r.get("versionStartExcluding", ""),
            r.get("versionEndIncluding", ""),
            r.get("versionEndExcluding", ""),
            r.get("cpe_version", ""),
        )
        k = str(key)
        if k not in seen:
            seen.add(k)
            unique_ranges.append(r)

    # Infer or use supplied ecosystem
    active_ecosystem = ecosystem if ecosystem != "auto" else _infer_ecosystem(unique_ranges)

    results: list[dict[str, Any]] = []

    for r in unique_ranges:
        vendor = r["vendor"]
        product = r["product"]
        cpe_version = r.get("cpe_version", _ANY)
        vsi = r.get("versionStartIncluding")
        vse = r.get("versionStartExcluding")
        vei = r.get("versionEndIncluding")
        vee = r.get("versionEndExcluding")

        # Build human-readable range string
        range_parts: list[str] = []
        if vsi:
            range_parts.append(f">= {vsi}")
        elif vse:
            range_parts.append(f"> {vse}")
        if vei:
            range_parts.append(f"<= {vei}")
        elif vee:
            range_parts.append(f"< {vee}")
        if not range_parts and cpe_version not in (_ANY, _NA, ""):
            range_parts.append(f"== {cpe_version}")
        if not range_parts:
            range_parts.append("all versions")
        range_str = ", ".join(range_parts)

        entry: dict[str, Any] = {
            "vendor": vendor,
            "product": product,
            "range": range_str,
            "ecosystem": active_ecosystem,
        }

        # Cross-reference with package registry
        releases: list[tuple[str, str]] = []
        package_found = False

        if active_ecosystem == "pypi":
            releases = _fetch_pypi_releases(product)
            package_found = bool(releases)
        elif active_ecosystem == "npm":
            releases = _fetch_npm_releases(product)
            package_found = bool(releases)
        elif active_ecosystem == "maven":
            # Best-effort: map vendor→groupId, product→artifactId
            group_id = vendor.replace("_", ".").replace("-", ".")
            artifact_id = product
            releases = _fetch_maven_releases(group_id, artifact_id)
            package_found = bool(releases)

        if not package_found:
            # Return the raw CPE range without release cross-reference
            entry["note"] = (
                "Package not found in registry or ecosystem unknown — showing raw CPE version constraints only."
            )
            entry["affected_versions"] = []
            entry["first_patched_version"] = None
            results.append(entry)
            continue

        # Find affected and patched releases
        affected: list[dict[str, str]] = []
        first_patched: str | None = None

        for ver_str, release_date in releases:
            in_range = _version_in_range(ver_str, vsi, vse, vei, vee, cpe_version)
            if in_range:
                affected.append({"version": ver_str, "release_date": release_date})
            elif affected and first_patched is None:
                # First release *after* the vulnerable range ends — the fix release
                first_patched = ver_str

        # Trim to avoid flooding the output
        shown = affected
        truncated = False
        if len(affected) > _MAX_VERSIONS_PER_RANGE:
            shown = affected[:5] + [{"version": "...", "release_date": ""}] + affected[-5:]
            truncated = True

        entry["affected_versions"] = shown
        entry["total_affected"] = len(affected)
        entry["truncated"] = truncated
        entry["first_patched_version"] = first_patched
        results.append(entry)

    return {
        "cve_id": cve_id,
        "description": description,
        "ecosystem": active_ecosystem,
        "ranges": results,
    }


# ---------------------------------------------------------------------------
# Text renderer
# ---------------------------------------------------------------------------


def _render_text(report: dict[str, Any]) -> str:
    cve_id = report["cve_id"]
    lines: list[str] = [f"Version Range Analysis: {cve_id}", "=" * 60]

    if "error" in report and not report.get("ranges"):
        lines.append(f"Error: {report['error']}")
        return "\n".join(lines)

    desc = report.get("description", "")
    if desc:
        # Wrap to 80 chars
        import textwrap

        lines.append("")
        lines.append("Description:")
        lines.extend(textwrap.wrap(desc, width=80))

    ecosystem = report.get("ecosystem", "unknown")
    lines += ["", f"Ecosystem: {ecosystem}", ""]

    ranges = report.get("ranges", [])
    if not ranges:
        lines.append("No CPE configuration data found — version ranges unknown.")
        return "\n".join(lines)

    for i, r in enumerate(ranges, 1):
        lines.append(f"[{i}] {r['vendor']} / {r['product']}")
        lines.append(f"    Vulnerable range: {r['range']}")

        note = r.get("note")
        if note:
            lines.append(f"    Note: {note}")
            lines.append("")
            continue

        total = r.get("total_affected", 0)
        truncated = r.get("truncated", False)
        affected = r.get("affected_versions", [])
        patched = r.get("first_patched_version")

        if total == 0:
            lines.append("    No matching releases found in registry.")
        else:
            label = f"    Affected releases ({total}" + (" shown: first/last 5" if truncated else "") + "):"
            lines.append(label)
            for av in affected:
                if av["version"] == "...":
                    lines.append("        ...")
                else:
                    lines.append(f"        {av['version']:<20} released {av['release_date']}")

        if patched:
            lines.append(f"    First patched release: {patched}")
        else:
            lines.append("    First patched release: not yet released (or not found in registry)")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strands tool entry point
# ---------------------------------------------------------------------------


def get_version_ranges(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    cve_id = tool_input.get("cve_id", "")
    ecosystem = tool_input.get("ecosystem", "auto")

    if not isinstance(cve_id, str) or not re.match(r"^CVE-\d{4}-\d{4,}$", cve_id.upper()):
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid CVE ID format. Expected CVE-YYYY-NNNNN (e.g. CVE-2024-3094)."}],
        }
        log_tool_output_size("get_version_ranges", result)
        return result

    report = resolve_version_ranges(cve_id.upper(), ecosystem=ecosystem)

    if "error" in report and not report.get("ranges"):
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": report["error"]}],
        }
    else:
        result = {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"json": report}],
        }

    log_tool_output_size("get_version_ranges", result)
    return result
