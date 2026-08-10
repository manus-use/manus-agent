#!/usr/bin/env python3
"""
Tool for scanning SBOM (Software Bill of Materials) files against OSV.dev.

Parses CycloneDX (JSON) or SPDX (JSON) SBOM files, extracts all component
package identifiers (ecosystem + name + version), queries OSV.dev in batch
for known vulnerabilities, enriches each finding with EPSS score and CISA KEV
membership, and ranks results by KEV status then EPSS score descending.

Public APIs used (no keys required):
- OSV.dev batch query: POST https://api.osv.dev/v1/querybatch
- FIRST.org EPSS: GET https://api.first.org/data/v1/epss?cve=<ids>
- CISA KEV catalog: GET https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
_EPSS_URL = "https://api.first.org/data/v1/epss"
_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
_TIMEOUT = 20

# Retry settings (overridable via env for tests)
_MAX_RETRIES: int = int(os.environ.get("SBOM_SCAN_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY: float = float(os.environ.get("SBOM_SCAN_RETRY_BASE_DELAY", "1.0"))
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# OSV batch query limit per request
_OSV_BATCH_SIZE = 1000

# ---------------------------------------------------------------------------
# PURL ecosystem mapping
# ---------------------------------------------------------------------------

# Maps purl type prefixes to OSV ecosystem names.
_PURL_TO_ECOSYSTEM: dict[str, str] = {
    "npm": "npm",
    "pypi": "PyPI",
    "maven": "Maven",
    "golang": "Go",
    "cargo": "crates.io",
    "nuget": "NuGet",
    "gem": "RubyGems",
    "composer": "Packagist",
    "pub": "Pub",
    "hex": "Hex",
    "hackage": "Hackage",
    "cocoapods": "CocoaPods",
    "swift": "SwiftURL",
    "cran": "CRAN",
    "deb": "Debian",
    "apk": "Alpine",
    "rpm": "Rocky Linux",
}

# ---------------------------------------------------------------------------
# TOOL_SPEC (Strands tool interface)
# ---------------------------------------------------------------------------

TOOL_SPEC = {
    "name": "scan_sbom",
    "description": (
        "Scans a CycloneDX or SPDX SBOM file for known vulnerabilities using "
        "OSV.dev batch queries. Enriches each finding with EPSS score and CISA "
        "KEV membership. Returns results ranked by KEV status then EPSS score. "
        "Supports CycloneDX JSON and SPDX JSON formats."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Path to the SBOM file (CycloneDX JSON or SPDX JSON). Example: 'bom.json' or 'sbom.spdx.json'."
                    ),
                }
            },
            "required": ["file_path"],
        }
    },
}


# ---------------------------------------------------------------------------
# HTTP helper with retry/back-off
# ---------------------------------------------------------------------------


def _request_with_retry(
    method: str,
    url: str,
    *,
    json_body: Any | None = None,
    params: dict[str, Any] | None = None,
    timeout: int = _TIMEOUT,
) -> requests.Response:
    """Make an HTTP request with exponential back-off on retryable errors."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.request(method, url, json=json_body, params=params, timeout=timeout)
            if resp.status_code not in _RETRYABLE_STATUSES:
                return resp
            last_exc = requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
        except requests.RequestException as exc:
            last_exc = exc

        if attempt < _MAX_RETRIES - 1:
            delay = _RETRY_BASE_DELAY * (2**attempt)
            time.sleep(delay)

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SBOM Parsing
# ---------------------------------------------------------------------------


def _parse_purl(purl: str) -> tuple[str, str, str] | None:
    """Parse a Package URL into (ecosystem, name, version) or None."""
    # Format: pkg:<type>/<namespace>/<name>@<version>
    # or:     pkg:<type>/<name>@<version>
    if not purl.startswith("pkg:"):
        return None

    # Strip scheme
    remainder = purl[4:]

    # Split type from the rest
    slash_idx = remainder.find("/")
    if slash_idx < 0:
        return None
    purl_type = remainder[:slash_idx].lower()
    rest = remainder[slash_idx + 1 :]

    # Split version
    at_idx = rest.rfind("@")
    if at_idx < 0:
        return None
    name_part = rest[:at_idx]
    version = rest[at_idx + 1 :]

    # Strip qualifiers/subpath from version
    for sep in ("?", "#"):
        q_idx = version.find(sep)
        if q_idx >= 0:
            version = version[:q_idx]

    # URL-decode common characters
    name_part = name_part.replace("%2F", "/").replace("%40", "@")

    ecosystem = _PURL_TO_ECOSYSTEM.get(purl_type)
    if not ecosystem:
        # Use the purl type as ecosystem (best effort)
        ecosystem = purl_type

    # For Maven, the namespace/name is the full group:artifact
    if purl_type == "maven":
        name_part = name_part.replace("/", ":")

    return ecosystem, name_part, version


def _parse_cyclonedx(data: dict[str, Any]) -> list[dict[str, str]]:
    """Extract components from a CycloneDX SBOM."""
    components: list[dict[str, str]] = []
    for comp in data.get("components", []):
        purl = comp.get("purl", "")
        if purl:
            parsed = _parse_purl(purl)
            if parsed:
                ecosystem, name, version = parsed
                components.append({"ecosystem": ecosystem, "name": name, "version": version, "purl": purl})
        elif comp.get("name") and comp.get("version"):
            # Fallback: without purl we cannot reliably determine ecosystem
            name = comp["name"]
            version = comp["version"]
            # Without purl, we cannot reliably determine ecosystem
            # Use generic package query
            components.append({"ecosystem": "", "name": name, "version": version, "purl": ""})
    return components


def _parse_spdx(data: dict[str, Any]) -> list[dict[str, str]]:
    """Extract packages from an SPDX SBOM."""
    components: list[dict[str, str]] = []
    for pkg in data.get("packages", []):
        # SPDX uses externalRefs for purls
        for ref in pkg.get("externalRefs", []):
            if ref.get("referenceType") == "purl" or ref.get("referenceCategory") == "PACKAGE-MANAGER":
                locator = ref.get("referenceLocator", "")
                if locator.startswith("pkg:"):
                    parsed = _parse_purl(locator)
                    if parsed:
                        ecosystem, name, version = parsed
                        components.append({"ecosystem": ecosystem, "name": name, "version": version, "purl": locator})
                    break
        else:
            # Fallback: use name + version if available
            name = pkg.get("name", "")
            version = pkg.get("versionInfo", "")
            if name and version and name != "NOASSERTION" and version != "NOASSERTION":
                components.append({"ecosystem": "", "name": name, "version": version, "purl": ""})
    return components


def parse_sbom(file_path: str) -> tuple[str, list[dict[str, str]]]:
    """
    Parse an SBOM file and return (format_name, components).

    Raises ValueError on unrecognised format or parse errors.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"SBOM file not found: {file_path}")

    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in SBOM file: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("SBOM file must contain a JSON object at root level")

    # Detect format
    if "bomFormat" in data or "components" in data:
        # CycloneDX
        fmt = "CycloneDX"
        components = _parse_cyclonedx(data)
    elif "spdxVersion" in data or "packages" in data:
        # SPDX
        fmt = "SPDX"
        components = _parse_spdx(data)
    else:
        raise ValueError("Unrecognised SBOM format. Supported: CycloneDX JSON, SPDX JSON.")

    return fmt, components


# ---------------------------------------------------------------------------
# OSV.dev Batch Query
# ---------------------------------------------------------------------------


def _build_osv_queries(
    components: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Build OSV batch query payloads from parsed components."""
    queries: list[dict[str, Any]] = []
    for comp in components:
        q: dict[str, Any] = {"version": comp["version"]}
        if comp["ecosystem"]:
            q["package"] = {"name": comp["name"], "ecosystem": comp["ecosystem"]}
        else:
            # Without ecosystem, query by package name only
            q["package"] = {"name": comp["name"]}
        queries.append(q)
    return queries


def query_osv_batch(
    components: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """
    Query OSV.dev for vulnerabilities affecting the given components.

    Returns a list of finding dicts with keys:
      component, ecosystem, name, version, purl, vulns (list of osv records)
    """
    if not components:
        return []

    queries = _build_osv_queries(components)
    all_results: list[dict[str, Any]] = []

    # Process in batches
    for i in range(0, len(queries), _OSV_BATCH_SIZE):
        batch = queries[i : i + _OSV_BATCH_SIZE]
        batch_components = components[i : i + _OSV_BATCH_SIZE]

        resp = _request_with_retry("POST", _OSV_BATCH_URL, json_body={"queries": batch})
        resp.raise_for_status()
        data = resp.json()

        results_list = data.get("results", [])
        for idx, result in enumerate(results_list):
            vulns = result.get("vulns", [])
            if vulns and idx < len(batch_components):
                comp = batch_components[idx]
                all_results.append(
                    {
                        "ecosystem": comp["ecosystem"],
                        "name": comp["name"],
                        "version": comp["version"],
                        "purl": comp["purl"],
                        "vulns": vulns,
                    }
                )

    return all_results


# ---------------------------------------------------------------------------
# EPSS Enrichment
# ---------------------------------------------------------------------------


def _fetch_epss_scores(cve_ids: list[str]) -> dict[str, float]:
    """Fetch EPSS scores for a list of CVE IDs. Returns {cve_id: score}."""
    if not cve_ids:
        return {}

    scores: dict[str, float] = {}

    # EPSS API accepts comma-separated CVEs (limit batches to avoid URL length issues)
    batch_size = 100
    for i in range(0, len(cve_ids), batch_size):
        batch = cve_ids[i : i + batch_size]
        try:
            resp = _request_with_retry("GET", _EPSS_URL, params={"cve": ",".join(batch)})
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", []):
                    cve = item.get("cve", "")
                    score = float(item.get("epss", 0))
                    scores[cve] = score
        except Exception:
            # Graceful degradation — EPSS enrichment is optional
            pass

    return scores


# ---------------------------------------------------------------------------
# CISA KEV Enrichment
# ---------------------------------------------------------------------------


def _fetch_kev_cve_set() -> set[str]:
    """Fetch the set of CVE IDs in the CISA KEV catalog."""
    try:
        resp = _request_with_retry("GET", _KEV_URL)
        if resp.status_code == 200:
            data = resp.json()
            return {v.get("cveID", "") for v in data.get("vulnerabilities", []) if v.get("cveID")}
    except Exception:
        pass
    return set()


# ---------------------------------------------------------------------------
# Core scan logic
# ---------------------------------------------------------------------------


def scan_sbom_file(file_path: str) -> dict[str, Any]:
    """
    Scan an SBOM file for vulnerabilities.

    Returns a dict with:
      - format: SBOM format detected
      - total_components: number of components parsed
      - vulnerable_components: number with at least one vulnerability
      - total_vulnerabilities: unique CVE/advisory count
      - critical_count: findings with EPSS >= 0.5 or in KEV
      - findings: list of enriched findings sorted by severity
    """
    fmt, components = parse_sbom(file_path)

    if not components:
        return {
            "format": fmt,
            "total_components": 0,
            "vulnerable_components": 0,
            "total_vulnerabilities": 0,
            "critical_count": 0,
            "findings": [],
        }

    # Query OSV
    osv_results = query_osv_batch(components)

    if not osv_results:
        return {
            "format": fmt,
            "total_components": len(components),
            "vulnerable_components": 0,
            "total_vulnerabilities": 0,
            "critical_count": 0,
            "findings": [],
        }

    # Collect all unique CVE IDs for enrichment
    all_cve_ids: set[str] = set()
    for result in osv_results:
        for vuln in result["vulns"]:
            vuln_id = vuln.get("id", "")
            if vuln_id.startswith("CVE-"):
                all_cve_ids.add(vuln_id)
            # Also check aliases
            for alias in vuln.get("aliases", []):
                if alias.startswith("CVE-"):
                    all_cve_ids.add(alias)

    # Enrich with EPSS + KEV
    epss_scores = _fetch_epss_scores(sorted(all_cve_ids))
    kev_set = _fetch_kev_cve_set()

    # Build findings
    findings: list[dict[str, Any]] = []
    seen_vuln_ids: set[str] = set()

    for result in osv_results:
        for vuln in result["vulns"]:
            vuln_id = vuln.get("id", "")
            if vuln_id in seen_vuln_ids:
                continue
            seen_vuln_ids.add(vuln_id)

            # Determine CVE ID (may be the id itself or in aliases)
            cve_id = vuln_id if vuln_id.startswith("CVE-") else ""
            aliases = vuln.get("aliases", [])
            if not cve_id:
                for alias in aliases:
                    if alias.startswith("CVE-"):
                        cve_id = alias
                        break

            epss = epss_scores.get(cve_id, 0.0) if cve_id else 0.0
            in_kev = cve_id in kev_set if cve_id else False

            severity_list = vuln.get("severity", [])
            cvss_score = None
            for sev in severity_list:
                if sev.get("type") == "CVSS_V3":
                    # Try to extract base score from vector
                    vector = sev.get("score", "")
                    if vector:
                        cvss_score = vector

            finding: dict[str, Any] = {
                "vuln_id": vuln_id,
                "cve_id": cve_id,
                "aliases": aliases,
                "summary": vuln.get("summary", vuln.get("details", ""))[:200],
                "affected_package": result["name"],
                "affected_ecosystem": result["ecosystem"],
                "affected_version": result["version"],
                "epss": epss,
                "in_kev": in_kev,
                "cvss_vector": cvss_score,
            }
            findings.append(finding)

    # Sort: KEV first, then by EPSS descending
    findings.sort(key=lambda f: (not f["in_kev"], -f["epss"]))

    critical_count = sum(1 for f in findings if f["in_kev"] or f["epss"] >= 0.5)

    return {
        "format": fmt,
        "total_components": len(components),
        "vulnerable_components": len(osv_results),
        "total_vulnerabilities": len(findings),
        "critical_count": critical_count,
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# Strands tool entry point
# ---------------------------------------------------------------------------


def scan_sbom(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands tool entry point for SBOM scanning."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    file_path = tool_input.get("file_path", "")

    if not file_path:
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "The 'file_path' input is required."}],
        }
        log_tool_output_size("scan_sbom", result)
        return result

    try:
        scan_result = scan_sbom_file(file_path)
        result = {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"json": scan_result}],
        }
        log_tool_output_size("scan_sbom", result)
        return result
    except FileNotFoundError as exc:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": str(exc)}],
        }
        log_tool_output_size("scan_sbom", result)
        return result
    except ValueError as exc:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": str(exc)}],
        }
        log_tool_output_size("scan_sbom", result)
        return result
    except requests.RequestException as exc:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Network error during SBOM scan: {exc}"}],
        }
        log_tool_output_size("scan_sbom", result)
        return result
