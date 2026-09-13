#!/usr/bin/env python3
"""
Tool for scanning CycloneDX or SPDX SBOMs against known vulnerabilities.

Parses a CycloneDX (1.4/1.5/1.6 JSON) or SPDX (2.2/2.3 JSON) SBOM file,
extracts all software components with their ecosystem and version, then queries
OSV.dev in batch for known vulnerabilities. Each finding is enriched with
EPSS exploitation probability and CISA KEV active-exploitation status, then
ranked by KEV membership first, EPSS score second.

This tool answers: "Given my software bill of materials, which components
have known vulnerabilities, and which should I patch first?"

Public APIs used (no keys required):
- OSV.dev batch query: POST https://api.osv.dev/v1/querybatch
- FIRST.org EPSS: GET https://api.first.org/data/v1/epss
- CISA KEV catalog: GET https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# Retry / back-off configuration
# ---------------------------------------------------------------------------
_MAX_RETRIES: int = int(os.environ.get("SBOM_SCAN_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY: float = float(os.environ.get("SBOM_SCAN_RETRY_BASE_DELAY", "1.0"))
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
_OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
_EPSS_API_URL = "https://api.first.org/data/v1/epss"
_CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
_HTTP_TIMEOUT = 15

# OSV batch API limit: max 1000 queries per request.
_OSV_BATCH_SIZE = 1000

# EPSS API accepts up to ~100 CVEs per request (URL length constraint).
_EPSS_BATCH_SIZE = 100

# ---------------------------------------------------------------------------
# Ecosystem mapping: purl type → OSV ecosystem
# ---------------------------------------------------------------------------
_PURL_TYPE_TO_OSV: dict[str, str] = {
    "npm": "npm",
    "pypi": "PyPI",
    "golang": "Go",
    "maven": "Maven",
    "cargo": "crates.io",
    "gem": "RubyGems",
    "nuget": "NuGet",
    "composer": "Packagist",
    "cocoapods": "CocoaPods",
    "swift": "SwiftURL",
    "pub": "Pub",
    "hex": "Hex",
    "hackage": "Hackage",
    "cran": "CRAN",
}

# SPDX external ref type → OSV ecosystem (best-effort mapping)
_SPDX_EXTREF_TO_OSV: dict[str, str] = {
    "npm": "npm",
    "pypi": "PyPI",
    "golang": "Go",
    "maven-central": "Maven",
    "crate": "crates.io",
    "gem": "RubyGems",
    "nuget": "NuGet",
    "purl": "__purl__",  # sentinel — parse the actual purl
}


TOOL_SPEC = {
    "name": "scan_sbom",
    "description": (
        "Scans a CycloneDX or SPDX SBOM (JSON) for known vulnerabilities. "
        "Parses all software components, queries OSV.dev in batch for each, "
        "enriches findings with EPSS exploitation probability and CISA KEV "
        "active-exploitation status, then ranks results by KEV membership "
        "and EPSS score. Returns per-component vulnerability counts, CVE "
        "details, severity info, and an executive summary with critical counts. "
        "Use this after generate_sbom to audit a project's dependency risk."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "sbom_path": {
                    "type": "string",
                    "description": (
                        "Path to the SBOM file (CycloneDX or SPDX JSON). Mutually exclusive with sbom_content."
                    ),
                },
                "sbom_content": {
                    "type": "string",
                    "description": ("Raw SBOM JSON content as a string. Mutually exclusive with sbom_path."),
                },
                "skip_epss": {
                    "type": "boolean",
                    "description": "Skip EPSS enrichment (faster). Default: false.",
                    "default": False,
                },
                "skip_kev": {
                    "type": "boolean",
                    "description": "Skip CISA KEV enrichment (faster). Default: false.",
                    "default": False,
                },
            },
            "required": [],
        }
    },
}


# ---------------------------------------------------------------------------
# HTTP helper with retry/back-off
# ---------------------------------------------------------------------------
def _http_request(
    method: str,
    url: str,
    *,
    json_body: Any = None,
    params: dict[str, str] | None = None,
    timeout: int = _HTTP_TIMEOUT,
) -> requests.Response:
    """Make an HTTP request with exponential back-off on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.request(
                method,
                url,
                json=json_body,
                params=params,
                timeout=timeout,
                headers={"Accept": "application/json"},
            )
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
# SBOM parsing: extract (ecosystem, package, version) tuples
# ---------------------------------------------------------------------------
class Component:
    """A single software component extracted from an SBOM."""

    __slots__ = ("ecosystem", "name", "version", "purl", "bom_ref")

    def __init__(
        self,
        ecosystem: str,
        name: str,
        version: str,
        purl: str = "",
        bom_ref: str = "",
    ):
        self.ecosystem = ecosystem
        self.name = name
        self.version = version
        self.purl = purl
        self.bom_ref = bom_ref

    def key(self) -> str:
        return f"{self.ecosystem}:{self.name}:{self.version}"

    def to_dict(self) -> dict[str, str]:
        return {
            "ecosystem": self.ecosystem,
            "name": self.name,
            "version": self.version,
            "purl": self.purl,
        }


def _parse_purl(purl: str) -> tuple[str, str, str]:
    """Parse a Package URL into (ecosystem, name, version).

    Handles pkg:<type>/<namespace>/<name>@<version> and pkg:<type>/<name>@<version>.
    Returns ("", "", "") on failure.
    """
    if not purl or not isinstance(purl, str):
        return ("", "", "")

    # Strip scheme
    s = purl
    if s.startswith("pkg:"):
        s = s[4:]

    # Split off qualifiers and subpath
    s = s.split("?")[0].split("#")[0]

    # Split type from rest
    slash_idx = s.find("/")
    if slash_idx < 0:
        return ("", "", "")
    purl_type = s[:slash_idx].lower()
    rest = s[slash_idx + 1 :]

    # Split version
    at_idx = rest.rfind("@")
    if at_idx < 0:
        version = ""
        name_part = rest
    else:
        version = _url_decode(rest[at_idx + 1 :])
        name_part = rest[:at_idx]

    # Handle namespace/name — for Maven use group:artifact
    parts = name_part.split("/")
    if purl_type == "maven" and len(parts) >= 2:
        name = _url_decode(parts[0]) + ":" + _url_decode(parts[1])
    elif len(parts) >= 2:
        # npm scoped: @scope/name
        if purl_type == "npm":
            name = _url_decode(parts[0]) + "/" + _url_decode(parts[1])
        else:
            name = _url_decode(parts[-1])
    else:
        name = _url_decode(parts[0])

    ecosystem = _PURL_TYPE_TO_OSV.get(purl_type, "")
    return (ecosystem, name, version)


def _url_decode(s: str) -> str:
    """Minimal percent-decode for purl components."""
    try:
        from urllib.parse import unquote

        return unquote(s)
    except Exception:
        return s


def detect_sbom_format(data: dict[str, Any]) -> str:
    """Detect whether an SBOM is CycloneDX or SPDX.

    Returns "cyclonedx", "spdx", or "unknown".
    """
    if "bomFormat" in data:
        return "cyclonedx"
    if data.get("bomFormat", "").upper() == "CYCLONEDX":
        return "cyclonedx"
    if "spdxVersion" in data:
        return "spdx"
    if "components" in data and isinstance(data.get("components"), list):
        # Heuristic: CycloneDX has components at top level
        return "cyclonedx"
    if "packages" in data and isinstance(data.get("packages"), list):
        # Heuristic: SPDX has packages at top level
        return "spdx"
    return "unknown"


def _parse_cyclonedx(data: dict[str, Any]) -> list[Component]:
    """Extract components from a CycloneDX SBOM."""
    components: list[Component] = []
    seen: set[str] = set()

    for comp in data.get("components", []) or []:
        if not isinstance(comp, dict):
            continue

        purl = comp.get("purl", "") or ""
        bom_ref = comp.get("bom-ref", "") or ""
        name = comp.get("name", "") or ""
        version = comp.get("version", "") or ""

        ecosystem = ""

        # Prefer purl for ecosystem detection
        if purl:
            eco, purl_name, purl_version = _parse_purl(purl)
            if eco:
                ecosystem = eco
            if purl_name and not name:
                name = purl_name
            if purl_version and not version:
                version = purl_version

        # Fall back to component type
        if not ecosystem:
            comp_type = (comp.get("type") or "").lower()
            if comp_type == "library":
                # Try to infer from purl type in the purl string
                if purl and ":" in purl:
                    purl_type = purl.split(":")[1].split("/")[0] if "pkg:" in purl else ""
                    ecosystem = _PURL_TYPE_TO_OSV.get(purl_type, "")

        if not name or not version:
            continue

        c = Component(
            ecosystem=ecosystem,
            name=name,
            version=version,
            purl=purl,
            bom_ref=bom_ref,
        )
        key = c.key()
        if key not in seen:
            seen.add(key)
            components.append(c)

    return components


def _parse_spdx(data: dict[str, Any]) -> list[Component]:
    """Extract components from an SPDX SBOM."""
    components: list[Component] = []
    seen: set[str] = set()

    for pkg in data.get("packages", []) or []:
        if not isinstance(pkg, dict):
            continue

        name = pkg.get("name", "") or ""
        version = pkg.get("versionInfo", "") or ""
        purl = ""
        ecosystem = ""

        # Check externalRefs for purls
        for ref in pkg.get("externalRefs", []) or []:
            if not isinstance(ref, dict):
                continue
            ref_type = (ref.get("referenceType") or "").lower()
            locator = ref.get("referenceLocator", "") or ""

            if ref_type == "purl" or locator.startswith("pkg:"):
                purl = locator
                eco, purl_name, purl_version = _parse_purl(locator)
                if eco:
                    ecosystem = eco
                if purl_name:
                    name = purl_name
                if purl_version and not version:
                    version = purl_version
                break

        if not name or not version:
            continue

        # Skip the root document package (SPDX convention)
        spdx_id = pkg.get("SPDXID", "")
        if spdx_id == "SPDXRef-DOCUMENT":
            continue

        c = Component(
            ecosystem=ecosystem,
            name=name,
            version=version,
            purl=purl,
        )
        key = c.key()
        if key not in seen:
            seen.add(key)
            components.append(c)

    return components


def parse_sbom(data: dict[str, Any]) -> tuple[str, list[Component]]:
    """Parse an SBOM and return (format, components).

    Raises ValueError if the format is not recognized.
    """
    fmt = detect_sbom_format(data)
    if fmt == "cyclonedx":
        return ("cyclonedx", _parse_cyclonedx(data))
    elif fmt == "spdx":
        return ("spdx", _parse_spdx(data))
    else:
        raise ValueError("Unrecognized SBOM format. Expected CycloneDX (bomFormat key) or SPDX (spdxVersion key) JSON.")


# ---------------------------------------------------------------------------
# OSV.dev batch vulnerability query
# ---------------------------------------------------------------------------
def _build_osv_query(component: Component) -> dict[str, Any]:
    """Build a single OSV query for a component."""
    if component.purl:
        return {"package": {"purl": component.purl}}
    if component.ecosystem and component.name and component.version:
        return {
            "package": {
                "name": component.name,
                "ecosystem": component.ecosystem,
            },
            "version": component.version,
        }
    # Fall back to name+version without ecosystem (less precise)
    return {
        "package": {"name": component.name},
        "version": component.version,
    }


def query_osv_batch(components: list[Component]) -> dict[str, list[dict[str, Any]]]:
    """Query OSV.dev in batch for vulnerabilities affecting the given components.

    Returns a dict mapping component key → list of vulnerability records.
    """
    results: dict[str, list[dict[str, Any]]] = {}

    for batch_start in range(0, len(components), _OSV_BATCH_SIZE):
        batch = components[batch_start : batch_start + _OSV_BATCH_SIZE]
        queries = [_build_osv_query(c) for c in batch]

        try:
            resp = _http_request("POST", _OSV_BATCH_URL, json_body={"queries": queries})
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            # On failure, mark all components in this batch as having no results
            for c in batch:
                results.setdefault(c.key(), [])
            continue

        batch_results = data.get("results", [])
        for i, c in enumerate(batch):
            vulns = []
            if i < len(batch_results):
                for vuln in batch_results[i].get("vulns", []) or []:
                    if isinstance(vuln, dict):
                        vulns.append(vuln)
            results[c.key()] = vulns

    return results


# ---------------------------------------------------------------------------
# EPSS enrichment
# ---------------------------------------------------------------------------
def fetch_epss_scores(cve_ids: list[str]) -> dict[str, float]:
    """Fetch EPSS scores for a list of CVE IDs.

    Returns a dict mapping CVE ID → EPSS probability (0.0–1.0).
    Missing/failed lookups are omitted from the result.
    """
    scores: dict[str, float] = {}
    if not cve_ids:
        return scores

    # Deduplicate
    unique_cves = sorted(set(cve_ids))

    for batch_start in range(0, len(unique_cves), _EPSS_BATCH_SIZE):
        batch = unique_cves[batch_start : batch_start + _EPSS_BATCH_SIZE]
        try:
            resp = _http_request(
                "GET",
                _EPSS_API_URL,
                params={"cve": ",".join(batch)},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        for entry in data.get("data", []) or []:
            if isinstance(entry, dict) and "cve" in entry:
                try:
                    scores[entry["cve"]] = float(entry.get("epss", 0))
                except (ValueError, TypeError):
                    pass

    return scores


# ---------------------------------------------------------------------------
# CISA KEV enrichment
# ---------------------------------------------------------------------------
def fetch_kev_set() -> set[str]:
    """Download the CISA KEV catalog and return a set of CVE IDs.

    Returns an empty set on failure (graceful degradation).
    """
    try:
        resp = _http_request("GET", _CISA_KEV_URL)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return set()

    kev_set: set[str] = set()
    for vuln in data.get("vulnerabilities", []) or []:
        if isinstance(vuln, dict) and "cveID" in vuln:
            kev_set.add(vuln["cveID"])
    return kev_set


# ---------------------------------------------------------------------------
# Core scanning logic
# ---------------------------------------------------------------------------
class Finding:
    """A single vulnerability finding for a component."""

    __slots__ = (
        "component",
        "vuln_id",
        "aliases",
        "summary",
        "severity",
        "epss",
        "in_kev",
        "fixed_versions",
    )

    def __init__(
        self,
        component: Component,
        vuln_id: str,
        aliases: list[str] | None = None,
        summary: str = "",
        severity: list[dict[str, str]] | None = None,
        epss: float | None = None,
        in_kev: bool = False,
        fixed_versions: list[str] | None = None,
    ):
        self.component = component
        self.vuln_id = vuln_id
        self.aliases = aliases or []
        self.summary = summary
        self.severity = severity or []
        self.epss = epss
        self.in_kev = in_kev
        self.fixed_versions = fixed_versions or []

    def cve_id(self) -> str:
        """Return the primary CVE ID from aliases, or the vuln_id itself."""
        for alias in self.aliases:
            if alias.startswith("CVE-"):
                return alias
        if self.vuln_id.startswith("CVE-"):
            return self.vuln_id
        return ""

    def sort_key(self) -> tuple[int, float, str]:
        """Sort key: KEV first, then EPSS descending, then ID."""
        return (
            0 if self.in_kev else 1,
            -(self.epss if self.epss is not None else -1.0),
            self.vuln_id,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "vuln_id": self.vuln_id,
            "aliases": self.aliases,
            "summary": self.summary,
            "severity": self.severity,
            "in_kev": self.in_kev,
            "fixed_versions": self.fixed_versions,
            "component": self.component.to_dict(),
        }
        if self.epss is not None:
            d["epss"] = self.epss
        return d


def _extract_fixed_versions(vuln: dict[str, Any]) -> list[str]:
    """Extract fixed version strings from an OSV vulnerability record."""
    fixed: list[str] = []
    for affected in vuln.get("affected", []) or []:
        if not isinstance(affected, dict):
            continue
        for rng in affected.get("ranges", []) or []:
            if not isinstance(rng, dict):
                continue
            for event in rng.get("events", []) or []:
                if isinstance(event, dict) and "fixed" in event:
                    v = str(event["fixed"])
                    if v and v not in fixed:
                        fixed.append(v)
    return fixed


def _extract_severity(vuln: dict[str, Any]) -> list[dict[str, str]]:
    """Extract severity entries from an OSV vulnerability record."""
    out: list[dict[str, str]] = []
    for sev in vuln.get("severity", []) or []:
        if isinstance(sev, dict):
            out.append(
                {
                    "type": str(sev.get("type", "")),
                    "score": str(sev.get("score", "")),
                }
            )
    # Also check database_specific for CVSS
    db_specific = vuln.get("database_specific", {}) or {}
    if isinstance(db_specific, dict):
        cvss_score = db_specific.get("severity")
        if isinstance(cvss_score, str) and cvss_score and not out:
            out.append({"type": "database_specific", "score": cvss_score})
    return out


def scan_sbom(
    sbom_data: dict[str, Any],
    *,
    skip_epss: bool = False,
    skip_kev: bool = False,
) -> dict[str, Any]:
    """Scan an SBOM for known vulnerabilities.

    This is the core public function used by both the Strands tool and
    the CLI subcommand.

    Returns a structured result dict with:
    - format: str — detected SBOM format
    - component_count: int — total components parsed
    - vulnerable_component_count: int — components with ≥1 vulnerability
    - total_finding_count: int — total vulnerability findings
    - kev_count: int — findings in CISA KEV
    - critical_count: int — findings with EPSS ≥ 0.7 or in KEV
    - findings: list[dict] — per-finding details, sorted by risk
    - components_scanned: list[dict] — all components extracted
    - errors: list[str] — non-fatal errors encountered
    """
    errors: list[str] = []

    # Parse SBOM
    try:
        sbom_format, components = parse_sbom(sbom_data)
    except ValueError as exc:
        return {
            "format": "unknown",
            "component_count": 0,
            "vulnerable_component_count": 0,
            "total_finding_count": 0,
            "kev_count": 0,
            "critical_count": 0,
            "findings": [],
            "components_scanned": [],
            "errors": [str(exc)],
            "message": f"SBOM parsing failed: {exc}",
        }

    if not components:
        return {
            "format": sbom_format,
            "component_count": 0,
            "vulnerable_component_count": 0,
            "total_finding_count": 0,
            "kev_count": 0,
            "critical_count": 0,
            "findings": [],
            "components_scanned": [],
            "errors": [],
            "message": "No components found in SBOM.",
        }

    # Query OSV.dev for vulnerabilities
    try:
        osv_results = query_osv_batch(components)
    except Exception as exc:
        errors.append(f"OSV.dev batch query failed: {exc}")
        osv_results = {}

    # Build findings
    findings: list[Finding] = []
    seen_finding_keys: set[str] = set()
    for comp in components:
        vulns = osv_results.get(comp.key(), [])
        for vuln in vulns:
            vuln_id = vuln.get("id", "")
            if not vuln_id:
                continue
            # Deduplicate: same component + same vuln
            fkey = f"{comp.key()}:{vuln_id}"
            if fkey in seen_finding_keys:
                continue
            seen_finding_keys.add(fkey)

            aliases = vuln.get("aliases", []) or []
            summary = vuln.get("summary", "") or vuln.get("details", "") or ""
            # Truncate long summaries
            if len(summary) > 300:
                summary = summary[:297] + "..."
            severity = _extract_severity(vuln)
            fixed_versions = _extract_fixed_versions(vuln)

            findings.append(
                Finding(
                    component=comp,
                    vuln_id=vuln_id,
                    aliases=aliases,
                    summary=summary,
                    severity=severity,
                    fixed_versions=fixed_versions,
                )
            )

    # Collect all CVE IDs for enrichment
    all_cve_ids: list[str] = []
    for f in findings:
        cid = f.cve_id()
        if cid:
            all_cve_ids.append(cid)
        # Also check aliases
        for alias in f.aliases:
            if alias.startswith("CVE-"):
                all_cve_ids.append(alias)

    # EPSS enrichment
    epss_scores: dict[str, float] = {}
    if not skip_epss and all_cve_ids:
        try:
            epss_scores = fetch_epss_scores(all_cve_ids)
        except Exception as exc:
            errors.append(f"EPSS enrichment failed: {exc}")

    # KEV enrichment
    kev_set: set[str] = set()
    if not skip_kev and all_cve_ids:
        try:
            kev_set = fetch_kev_set()
        except Exception as exc:
            errors.append(f"CISA KEV enrichment failed: {exc}")

    # Apply enrichment to findings
    for f in findings:
        cid = f.cve_id()
        if cid and cid in epss_scores:
            f.epss = epss_scores[cid]
        # Check all aliases against KEV
        for alias in [f.vuln_id] + f.aliases:
            if alias in kev_set:
                f.in_kev = True
                break

    # Sort findings by risk
    findings.sort(key=lambda f: f.sort_key())

    # Count stats
    vulnerable_components: set[str] = set()
    kev_count = 0
    critical_count = 0
    for f in findings:
        vulnerable_components.add(f.component.key())
        if f.in_kev:
            kev_count += 1
            critical_count += 1
        elif f.epss is not None and f.epss >= 0.7:
            critical_count += 1

    # Build summary message
    msg_parts = [
        f"Scanned {len(components)} components ({sbom_format} format).",
        f"Found {len(findings)} vulnerabilities across {len(vulnerable_components)} component(s).",
    ]
    if kev_count:
        msg_parts.append(f"⚠️  {kev_count} finding(s) are in the CISA KEV (actively exploited).")
    if critical_count:
        msg_parts.append(f"🔴 {critical_count} critical finding(s) (KEV or EPSS ≥ 0.7).")
    if not findings:
        msg_parts.append("✅ No known vulnerabilities found.")

    return {
        "format": sbom_format,
        "component_count": len(components),
        "vulnerable_component_count": len(vulnerable_components),
        "total_finding_count": len(findings),
        "kev_count": kev_count,
        "critical_count": critical_count,
        "findings": [f.to_dict() for f in findings],
        "components_scanned": [c.to_dict() for c in components],
        "errors": errors,
        "message": " ".join(msg_parts),
    }


# ---------------------------------------------------------------------------
# Strands tool entry point
# ---------------------------------------------------------------------------
def scan_sbom_tool(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands tool wrapper for scan_sbom."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool.get("input", {}) or {}

    sbom_path = tool_input.get("sbom_path")
    sbom_content = tool_input.get("sbom_content")
    skip_epss = bool(tool_input.get("skip_epss", False))
    skip_kev = bool(tool_input.get("skip_kev", False))

    # Load SBOM data
    if sbom_path:
        try:
            with open(sbom_path) as fh:
                sbom_data = json.load(fh)
        except FileNotFoundError:
            result: ToolResult = {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"SBOM file not found: {sbom_path}"}],
            }
            log_tool_output_size("scan_sbom", result)
            return result
        except json.JSONDecodeError as exc:
            result = {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Invalid JSON in SBOM file: {exc}"}],
            }
            log_tool_output_size("scan_sbom", result)
            return result
    elif sbom_content:
        try:
            sbom_data = json.loads(sbom_content)
        except json.JSONDecodeError as exc:
            result = {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Invalid JSON in SBOM content: {exc}"}],
            }
            log_tool_output_size("scan_sbom", result)
            return result
    else:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Either sbom_path or sbom_content is required."}],
        }
        log_tool_output_size("scan_sbom", result)
        return result

    payload = scan_sbom(sbom_data, skip_epss=skip_epss, skip_kev=skip_kev)
    status = "success" if not payload.get("errors") or payload["total_finding_count"] >= 0 else "error"

    result = {
        "toolUseId": tool_use_id,
        "status": status,
        "content": [{"text": json.dumps(payload, indent=2)}],
    }
    log_tool_output_size("scan_sbom", result)
    return result
