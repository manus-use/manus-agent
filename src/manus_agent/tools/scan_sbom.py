#!/usr/bin/env python3
"""
Tool for scanning CycloneDX or SPDX SBOMs against OSV.dev for known
vulnerabilities, enriched with EPSS scores and CISA KEV membership.

Implements the README-documented ``manus-agent sbom-scan`` subcommand.

Strategy
--------
1. Parse the SBOM file (CycloneDX JSON, CycloneDX XML, or SPDX JSON).
2. Extract all components as (ecosystem, name, version) tuples.
3. Query OSV.dev ``/v1/querybatch`` in batches of up to 1 000 packages.
4. For each vulnerability found, fetch its current EPSS score from
   ``api.first.org`` and check CISA KEV membership.
5. Deduplicate results (one entry per vuln × component), rank by
   KEV membership (descending) then EPSS score (descending), and
   return a structured report.

All HTTP calls have retry/back-off for robustness.  No API keys
required (OSV.dev, EPSS, and CISA KEV are public).
"""

from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET  # noqa: N817
from pathlib import Path
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
_HTTP_TIMEOUT = 20

# OSV.dev batch query endpoint (public, no key)
_OSV_QUERYBATCH_URL = "https://api.osv.dev/v1/querybatch"
_OSV_BATCH_SIZE = 1000  # API limit per call

# EPSS endpoint (public, no key)
_EPSS_URL = "https://api.first.org/data/v1/epss"
_EPSS_BATCH_SIZE = 100  # fetch up to 100 CVEs at a time

# CISA KEV feed
_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

# ---------------------------------------------------------------------------
# PURL ecosystem → OSV ecosystem mapping
# ---------------------------------------------------------------------------
_PURL_TO_OSV: dict[str, str] = {
    "pypi": "PyPI",
    "npm": "npm",
    "maven": "Maven",
    "golang": "Go",
    "cargo": "crates.io",
    "gem": "RubyGems",
    "nuget": "NuGet",
    "composer": "Packagist",
    "pub": "Pub",
    "hex": "Hex",
    "swift": "SwiftURL",
    "cocoapods": "CocoaPods",
    "hackage": "Hackage",
    "cran": "CRAN",
    "bitnami": "Bitnami",
}

# ---------------------------------------------------------------------------
# TOOL_SPEC (Strands SDK)
# ---------------------------------------------------------------------------
TOOL_SPEC = {
    "name": "scan_sbom",
    "description": (
        "Scans a CycloneDX or SPDX SBOM file for known vulnerabilities. "
        "Parses components, queries OSV.dev in batch, enriches each finding "
        "with its current EPSS score and CISA KEV status, and ranks results "
        "by KEV membership then EPSS score (most urgent first). "
        "Accepts CycloneDX JSON/XML and SPDX JSON formats."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "sbom_path": {
                    "type": "string",
                    "description": ("Path to the SBOM file (CycloneDX JSON/XML or SPDX JSON)."),
                }
            },
            "required": ["sbom_path"],
        }
    },
}


# ── HTTP helper ──────────────────────────────────────────────────────────
def _http_get(url: str, params: dict[str, Any] | None = None) -> requests.Response:
    """GET with retry/back-off."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=_HTTP_TIMEOUT)
            if resp.status_code not in _RETRYABLE_STATUSES:
                return resp
            last_exc = requests.HTTPError(response=resp)
        except requests.RequestException as exc:
            last_exc = exc
        if attempt < _MAX_RETRIES - 1:
            time.sleep(_RETRY_BASE_DELAY * (2**attempt))
    raise last_exc  # type: ignore[misc]


def _http_post(url: str, json_body: Any) -> requests.Response:
    """POST with retry/back-off."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(url, json=json_body, timeout=_HTTP_TIMEOUT)
            if resp.status_code not in _RETRYABLE_STATUSES:
                return resp
            last_exc = requests.HTTPError(response=resp)
        except requests.RequestException as exc:
            last_exc = exc
        if attempt < _MAX_RETRIES - 1:
            time.sleep(_RETRY_BASE_DELAY * (2**attempt))
    raise last_exc  # type: ignore[misc]


# ── SBOM Parsing ─────────────────────────────────────────────────────────

# Component tuple: (ecosystem, name, version)
Component = tuple[str, str, str]


def _parse_purl(purl: str) -> Component | None:
    """Extract (ecosystem, name, version) from a package URL."""
    # pkg:type/namespace/name@version?qualifiers#subpath
    if not purl.startswith("pkg:"):
        return None
    rest = purl[4:]
    # Split off qualifiers/subpath
    rest = rest.split("?")[0].split("#")[0]
    parts = rest.split("/", 1)
    if len(parts) < 2:
        return None
    purl_type = parts[0].lower()
    remainder = parts[1]
    # Extract version
    if "@" not in remainder:
        return None
    name_part, version = remainder.rsplit("@", 1)
    if not version:
        return None
    ecosystem = _PURL_TO_OSV.get(purl_type, "")
    if not ecosystem:
        return None
    # For Maven, namespace/name → namespace:name
    if purl_type == "maven":
        name_part = name_part.replace("/", ":")
    elif "/" in name_part:
        # e.g. npm scoped: @scope/name → just use as-is
        pass
    return (ecosystem, name_part, version)


def _parse_cdx_json(data: dict[str, Any]) -> list[Component]:
    """Parse CycloneDX JSON SBOM."""
    components: list[Component] = []
    for comp in data.get("components", []):
        # Prefer purl
        purl = comp.get("purl", "")
        if purl:
            parsed = _parse_purl(purl)
            if parsed:
                components.append(parsed)
                continue
        # Fall back to type + name + version
        comp_type = (comp.get("type") or "").lower()
        name = comp.get("name", "")
        version = comp.get("version", "")
        if not name or not version:
            continue
        # Map CycloneDX component type to ecosystem heuristic
        eco = ""
        if comp_type == "library":
            # Can't determine ecosystem without purl; skip
            continue
        if eco and name and version:
            components.append((eco, name, version))
    return components


def _parse_cdx_xml(root: ET.Element) -> list[Component]:
    """Parse CycloneDX XML SBOM."""
    components: list[Component] = []
    # Handle namespace
    ns = ""
    tag = root.tag
    if "}" in tag:
        ns = tag.split("}")[0] + "}"

    for comp in root.iter(f"{ns}component"):
        purl_elem = comp.find(f"{ns}purl")
        if purl_elem is not None and purl_elem.text:
            parsed = _parse_purl(purl_elem.text)
            if parsed:
                components.append(parsed)
                continue
        name_elem = comp.find(f"{ns}name")
        version_elem = comp.find(f"{ns}version")
        if name_elem is not None and version_elem is not None:
            name = (name_elem.text or "").strip()
            version = (version_elem.text or "").strip()
            if name and version:
                # Without purl, ecosystem is unknown; skip
                continue
    return components


def _parse_spdx_json(data: dict[str, Any]) -> list[Component]:
    """Parse SPDX 2.x JSON SBOM."""
    components: list[Component] = []
    for pkg in data.get("packages", []):
        # Try externalRefs for purl
        for ref in pkg.get("externalRefs", []):
            if ref.get("referenceType") == "purl":
                parsed = _parse_purl(ref.get("referenceLocator", ""))
                if parsed:
                    components.append(parsed)
                    break
        # SPDX without purl: name + versionInfo, but no ecosystem
        # so we skip those (same as CycloneDX without purl)
    return components


def parse_sbom(path: str) -> list[Component]:
    """
    Detect format and parse an SBOM file into components.

    Returns a list of (ecosystem, name, version) tuples.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SBOM file not found: {path}")

    content = p.read_text(encoding="utf-8")

    # Try JSON first
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        data = None

    if data is not None:
        if isinstance(data, dict):
            # CycloneDX JSON: has "bomFormat" or "components" key
            if data.get("bomFormat") == "CycloneDX" or "components" in data:
                return _parse_cdx_json(data)
            # SPDX JSON: has "spdxVersion" key
            if "spdxVersion" in data or "packages" in data:
                return _parse_spdx_json(data)
            # Fallback: try CycloneDX
            if "components" in data:
                return _parse_cdx_json(data)
        raise ValueError("Unrecognised SBOM format: expected CycloneDX or SPDX JSON.")

    # Try XML (CycloneDX)
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError(f"Cannot parse SBOM file as JSON or XML: {exc}") from exc

    return _parse_cdx_xml(root)


# ── OSV batch query ──────────────────────────────────────────────────────


def _query_osv_batch(
    components: list[Component],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """
    Query OSV.dev ``/v1/querybatch`` for all components.

    Returns a mapping from (ecosystem, name, version) → list of OSV vulns.
    """
    results: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for batch_start in range(0, len(components), _OSV_BATCH_SIZE):
        batch = components[batch_start : batch_start + _OSV_BATCH_SIZE]
        queries = [
            {
                "package": {"name": name, "ecosystem": eco},
                "version": version,
            }
            for eco, name, version in batch
        ]

        resp = _http_post(_OSV_QUERYBATCH_URL, {"queries": queries})
        resp.raise_for_status()
        data = resp.json()

        for comp, result_entry in zip(batch, data.get("results", []), strict=False):
            vulns = result_entry.get("vulns", [])
            if vulns:
                results[comp] = vulns

    return results


# ── EPSS enrichment ──────────────────────────────────────────────────────


def _fetch_epss_scores(cve_ids: list[str]) -> dict[str, float]:
    """Fetch current EPSS scores for a batch of CVE IDs."""
    scores: dict[str, float] = {}
    if not cve_ids:
        return scores

    for batch_start in range(0, len(cve_ids), _EPSS_BATCH_SIZE):
        batch = cve_ids[batch_start : batch_start + _EPSS_BATCH_SIZE]
        try:
            resp = _http_get(_EPSS_URL, params={"cve": ",".join(batch)})
            resp.raise_for_status()
            data = resp.json()
            for entry in data.get("data", []):
                cve = entry.get("cve", "")
                epss = entry.get("epss")
                if cve and epss is not None:
                    scores[cve] = float(epss)
        except Exception:
            # Graceful degradation: if EPSS fails, continue without scores
            pass

    return scores


# ── CISA KEV enrichment ─────────────────────────────────────────────────


def _fetch_kev_cve_set() -> set[str]:
    """Fetch the set of CVE IDs in the CISA KEV catalogue."""
    try:
        resp = _http_get(_KEV_URL)
        resp.raise_for_status()
        data = resp.json()
        return {v.get("cveID", "") for v in data.get("vulnerabilities", []) if v.get("cveID")}
    except Exception:
        return set()


# ── Result building ──────────────────────────────────────────────────────


def build_scan_results(
    components: list[Component],
    osv_results: dict[tuple[str, str, str], list[dict[str, Any]]],
    epss_scores: dict[str, float],
    kev_set: set[str],
) -> dict[str, Any]:
    """
    Build the final structured scan report.

    Returns a dict with summary counts and a ranked list of findings.
    """
    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()  # (vuln_id, ecosystem, name)

    for comp, vulns in osv_results.items():
        eco, name, version = comp
        for vuln in vulns:
            vuln_id = vuln.get("id", "unknown")
            dedup_key = (vuln_id, eco, name)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Collect aliases (CVE IDs)
            aliases = vuln.get("aliases", [])
            cve_ids = [a for a in aliases if a.startswith("CVE-")]
            if vuln_id.startswith("CVE-") and vuln_id not in cve_ids:
                cve_ids.insert(0, vuln_id)

            # Severity
            severity_list = vuln.get("severity", [])
            cvss_score: float | None = None
            for sev in severity_list:
                score = sev.get("score")
                if score is not None:
                    try:
                        cvss_score = float(score)
                        break
                    except (ValueError, TypeError):
                        pass

            # EPSS: best score from any CVE alias
            epss_score: float | None = None
            for cve in cve_ids:
                s = epss_scores.get(cve)
                if s is not None and (epss_score is None or s > epss_score):
                    epss_score = s

            # KEV check
            in_kev = any(cve in kev_set for cve in cve_ids)

            # Affected ranges summary
            affected_text = ""
            for aff in vuln.get("affected", []):
                pkg = aff.get("package", {})
                if pkg.get("name") == name and pkg.get("ecosystem") == eco:
                    for rng in aff.get("ranges", []):
                        events = rng.get("events", [])
                        intro = ""
                        fixed = ""
                        for ev in events:
                            if "introduced" in ev:
                                intro = ev["introduced"]
                            if "fixed" in ev:
                                fixed = ev["fixed"]
                        if intro or fixed:
                            intro_str = intro or "0"
                            fixed_str = fixed or "(unfixed)"
                            affected_text = f">={intro_str}, <{fixed_str}"
                            break
                    break

            finding = {
                "vuln_id": vuln_id,
                "aliases": cve_ids,
                "ecosystem": eco,
                "package": name,
                "installed_version": version,
                "affected_range": affected_text,
                "cvss_score": cvss_score,
                "epss_score": epss_score,
                "in_kev": in_kev,
                "summary": vuln.get("summary", ""),
            }
            findings.append(finding)

    # Sort: KEV first, then by EPSS descending, then by CVSS descending
    findings.sort(
        key=lambda f: (
            not f["in_kev"],  # False sorts before True → KEV first
            -(f["epss_score"] or 0.0),
            -(f["cvss_score"] or 0.0),
        )
    )

    # Severity classification
    critical_count = sum(1 for f in findings if f.get("cvss_score") is not None and f["cvss_score"] >= 9.0)
    high_count = sum(1 for f in findings if f.get("cvss_score") is not None and 7.0 <= f["cvss_score"] < 9.0)
    medium_count = sum(1 for f in findings if f.get("cvss_score") is not None and 4.0 <= f["cvss_score"] < 7.0)
    low_count = sum(1 for f in findings if f.get("cvss_score") is not None and f["cvss_score"] < 4.0)
    unscored_count = sum(1 for f in findings if f.get("cvss_score") is None)

    return {
        "total_components": len(components),
        "vulnerable_components": len(osv_results),
        "total_findings": len(findings),
        "critical_count": critical_count,
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "unscored_count": unscored_count,
        "kev_count": sum(1 for f in findings if f["in_kev"]),
        "findings": findings,
    }


# ── Text formatting ──────────────────────────────────────────────────────


def format_text_report(report: dict[str, Any]) -> str:
    """Render the scan report as human-readable text."""
    lines: list[str] = []
    lines.append("═══ SBOM Vulnerability Scan ═══\n")
    lines.append(
        f"Components scanned : {report['total_components']}\n"
        f"Vulnerable packages: {report['vulnerable_components']}\n"
        f"Total findings     : {report['total_findings']}\n"
    )
    lines.append(
        f"  Critical (≥9.0)  : {report['critical_count']}\n"
        f"  High (7.0–8.9)   : {report['high_count']}\n"
        f"  Medium (4.0–6.9) : {report['medium_count']}\n"
        f"  Low (<4.0)       : {report['low_count']}\n"
        f"  Unscored         : {report['unscored_count']}\n"
        f"  In CISA KEV      : {report['kev_count']}\n"
    )

    if not report["findings"]:
        lines.append("✅ No known vulnerabilities found.\n")
        return "\n".join(lines)

    lines.append("─── Findings (ranked by urgency) ───\n")
    for i, f in enumerate(report["findings"], 1):
        kev_flag = " 🔴 KEV" if f["in_kev"] else ""
        epss_str = f"{f['epss_score']:.4f}" if f["epss_score"] is not None else "N/A"
        cvss_str = f"{f['cvss_score']:.1f}" if f["cvss_score"] is not None else "N/A"
        aliases_str = ", ".join(f["aliases"]) if f["aliases"] else "—"

        lines.append(
            f"{i}. {f['vuln_id']}{kev_flag}\n"
            f"   Package  : {f['ecosystem']}/{f['package']} @ {f['installed_version']}\n"
            f"   Aliases  : {aliases_str}\n"
            f"   CVSS     : {cvss_str}   EPSS: {epss_str}\n"
        )
        if f["affected_range"]:
            lines.append(f"   Affected : {f['affected_range']}\n")
        if f["summary"]:
            lines.append(f"   Summary  : {f['summary']}\n")
        lines.append("")

    return "\n".join(lines)


# ── Main Strands tool entry point ────────────────────────────────────────


def scan_sbom(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands tool entry point for SBOM scanning."""
    sbom_path = tool["input"].get("sbom_path", "")

    if not sbom_path:
        result = {
            "status": "error",
            "content": [{"text": "Error: sbom_path is required."}],
        }
        log_tool_output_size("scan_sbom", result)
        return result

    # 1. Parse SBOM
    try:
        components = parse_sbom(sbom_path)
    except (FileNotFoundError, ValueError) as exc:
        result = {
            "status": "error",
            "content": [{"text": f"Error parsing SBOM: {exc}"}],
        }
        log_tool_output_size("scan_sbom", result)
        return result

    if not components:
        result = {
            "status": "success",
            "content": [
                {
                    "text": (
                        "SBOM parsed but no components with recognised "
                        "ecosystems found. Ensure packages have purl identifiers."
                    )
                }
            ],
        }
        log_tool_output_size("scan_sbom", result)
        return result

    # 2. Query OSV.dev
    try:
        osv_results = _query_osv_batch(components)
    except Exception as exc:
        result = {
            "status": "error",
            "content": [{"text": f"Error querying OSV.dev: {exc}"}],
        }
        log_tool_output_size("scan_sbom", result)
        return result

    # 3. Collect all CVE IDs for EPSS enrichment
    all_cve_ids: set[str] = set()
    for vulns in osv_results.values():
        for vuln in vulns:
            vid = vuln.get("id", "")
            if vid.startswith("CVE-"):
                all_cve_ids.add(vid)
            for alias in vuln.get("aliases", []):
                if alias.startswith("CVE-"):
                    all_cve_ids.add(alias)

    # 4. Fetch EPSS scores (graceful degradation)
    epss_scores = _fetch_epss_scores(sorted(all_cve_ids))

    # 5. Fetch CISA KEV set (graceful degradation)
    kev_set = _fetch_kev_cve_set()

    # 6. Build report
    report = build_scan_results(components, osv_results, epss_scores, kev_set)

    text = format_text_report(report)
    result = {
        "status": "success",
        "content": [{"text": text}],
    }
    log_tool_output_size("scan_sbom", result)
    return result
