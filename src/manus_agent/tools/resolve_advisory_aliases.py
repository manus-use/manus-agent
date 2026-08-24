"""Resolve advisory aliases for a CVE across multiple vulnerability databases.

Maps a CVE identifier to all its cross-referenced vendor-specific advisory IDs
(GHSA, RHSA, DSA, USN, PYSEC, RUSTSEC, GO, etc.) using OSV.dev's alias graph
and NVD reference classification.

This is valuable for:
- Finding the GitHub Security Advisory (GHSA) for a CVE to get patch info
- Discovering distro-specific advisories (DSA, USN, RHSA, ALAS) for patching
- Identifying ecosystem-specific IDs (PYSEC, RUSTSEC, GO) for dependency tools
- Correlating advisories across databases for comprehensive coverage assessment

Data sources:
- OSV.dev /vulns/{id} API — primary alias graph
- NVD references — supplemental advisory URL extraction
- VulnCheck (optional) — additional cross-references when API key available
"""

import json
import os
import re
import time
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

TOOL_SPEC = {
    "name": "resolve_advisory_aliases",
    "description": (
        "Resolves all cross-referenced advisory identifiers for a given CVE. "
        "Maps a CVE to its aliases across vulnerability databases: GHSA (GitHub), "
        "RHSA/RHBA (Red Hat), DSA (Debian), USN (Ubuntu), ALAS (Amazon Linux), "
        "PYSEC (Python), RUSTSEC (Rust), GO (Go), HSEC (Haskell), MAL (malware), "
        "and more. Uses OSV.dev alias graph as the primary source, with NVD "
        "references and optional VulnCheck for additional coverage."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier to resolve aliases for (e.g., 'CVE-2021-44228').",
                },
                "include_urls": {
                    "type": "boolean",
                    "description": ("Whether to include advisory URLs for each alias. Default: true."),
                    "default": True,
                },
            },
            "required": ["cve_id"],
        }
    },
}

# ---------------------------------------------------------------------------
# Advisory prefix classification
# ---------------------------------------------------------------------------

ADVISORY_PREFIXES = {
    "GHSA": "GitHub Security Advisory",
    "RHSA": "Red Hat Security Advisory",
    "RHBA": "Red Hat Bug Advisory",
    "DSA": "Debian Security Advisory",
    "DLA": "Debian LTS Advisory",
    "USN": "Ubuntu Security Notice",
    "ALAS": "Amazon Linux Security Advisory",
    "PYSEC": "Python Security Advisory (OSV)",
    "RUSTSEC": "Rust Security Advisory",
    "GO": "Go Vulnerability",
    "HSEC": "Haskell Security Advisory",
    "MAL": "Malicious Package Advisory",
    "GSD": "Global Security Database",
    "OSV": "OSV.dev Advisory",
    "SNYK": "Snyk Vulnerability",
    "CVE": "Common Vulnerabilities and Exposures",
}

# URL templates for advisory databases
ADVISORY_URL_TEMPLATES = {
    "GHSA": "https://github.com/advisories/{id}",
    "RHSA": "https://access.redhat.com/errata/{id}",
    "RHBA": "https://access.redhat.com/errata/{id}",
    "DSA": "https://www.debian.org/security/{year}/{id_lower}",
    "DLA": "https://www.debian.org/lts/security/{year}/{id_lower}",
    "USN": "https://ubuntu.com/security/notices/{id}",
    "PYSEC": "https://osv.dev/vulnerability/{id}",
    "RUSTSEC": "https://rustsec.org/advisories/{id}.html",
    "GO": "https://pkg.go.dev/vuln/{id}",
    "HSEC": "https://osv.dev/vulnerability/{id}",
    "MAL": "https://osv.dev/vulnerability/{id}",
    "GSD": "https://osv.dev/vulnerability/{id}",
    "CVE": "https://nvd.nist.gov/vuln/detail/{id}",
}

# ---------------------------------------------------------------------------
# HTTP helpers with retry/back-off
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds


def _get_with_retry(url: str, headers: dict | None = None, timeout: int = 15) -> requests.Response:
    """HTTP GET with exponential back-off on transient failures."""
    resp = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers or {}, timeout=timeout)
            if resp.status_code == 429:
                wait = _BACKOFF_BASE ** (attempt + 1)
                time.sleep(wait)
                continue
            return resp
        except (requests.ConnectionError, requests.Timeout):
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_BACKOFF_BASE ** (attempt + 1))
    return resp  # type: ignore[return-value]


def _post_with_retry(url: str, json_data: dict, headers: dict | None = None, timeout: int = 15) -> requests.Response:
    """HTTP POST with exponential back-off on transient failures."""
    resp = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(url, json=json_data, headers=headers or {}, timeout=timeout)
            if resp.status_code == 429:
                wait = _BACKOFF_BASE ** (attempt + 1)
                time.sleep(wait)
                continue
            return resp
        except (requests.ConnectionError, requests.Timeout):
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_BACKOFF_BASE ** (attempt + 1))
    return resp  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Advisory ID classification
# ---------------------------------------------------------------------------


def _classify_advisory(advisory_id: str) -> dict[str, str]:
    """Classify an advisory ID by its prefix and generate URL."""
    for prefix, db_name in ADVISORY_PREFIXES.items():
        if advisory_id.startswith(prefix):
            url = _build_advisory_url(advisory_id, prefix)
            return {
                "id": advisory_id,
                "database": db_name,
                "prefix": prefix,
                "url": url,
            }

    # Unknown prefix
    return {
        "id": advisory_id,
        "database": "Unknown",
        "prefix": advisory_id.split("-")[0] if "-" in advisory_id else advisory_id,
        "url": f"https://osv.dev/vulnerability/{advisory_id}",
    }


def _build_advisory_url(advisory_id: str, prefix: str) -> str:
    """Build a URL for an advisory based on its prefix."""
    template = ADVISORY_URL_TEMPLATES.get(prefix)
    if not template:
        return f"https://osv.dev/vulnerability/{advisory_id}"

    # Special handling for Debian advisories (need year)
    if prefix in ("DSA", "DLA"):
        match = re.match(r"(DSA|DLA)-(\d+)", advisory_id)
        if match:
            num = int(match.group(2))
            # Approximate year from advisory number ranges
            if num >= 5000:
                year = "2023"
            elif num >= 4000:
                year = "2019"
            elif num >= 3000:
                year = "2016"
            else:
                year = "2014"
            return template.format(year=year, id_lower=advisory_id.lower())
        return f"https://osv.dev/vulnerability/{advisory_id}"

    return template.format(id=advisory_id)


# ---------------------------------------------------------------------------
# OSV.dev alias resolution
# ---------------------------------------------------------------------------


def _query_osv_aliases(cve_id: str) -> dict[str, Any]:
    """Query OSV.dev for advisory aliases of a CVE.

    Uses the OSV.dev vulnerability query endpoint which returns all
    advisories linked to a given CVE through the alias graph.
    """
    direct_url = f"https://api.osv.dev/v1/vulns/{cve_id}"
    aliases: set[str] = set()
    affected_packages: list[dict[str, str]] = []
    osv_entries: list[dict[str, str]] = []

    # Try direct lookup first
    try:
        resp = _get_with_retry(direct_url)
        if resp.status_code == 200:
            data = resp.json()
            if "aliases" in data:
                aliases.update(data["aliases"])
            if "affected" in data:
                for affected in data["affected"]:
                    pkg = affected.get("package", {})
                    if pkg:
                        affected_packages.append(
                            {
                                "ecosystem": pkg.get("ecosystem", ""),
                                "name": pkg.get("name", ""),
                            }
                        )
            osv_entries.append(
                {
                    "id": data.get("id", cve_id),
                    "summary": data.get("summary", ""),
                    "modified": data.get("modified", ""),
                }
            )
    except (requests.RequestException, ValueError):
        pass

    # Query for all OSV entries that reference this CVE as an alias
    query_url = "https://api.osv.dev/v1/query"
    try:
        resp = _post_with_retry(query_url, {"aliases": [cve_id]})
        if resp.status_code == 200:
            data = resp.json()
            vulns = data.get("vulns", [])
            for vuln in vulns:
                vuln_id = vuln.get("id", "")
                if vuln_id and vuln_id != cve_id:
                    aliases.add(vuln_id)
                # Also collect aliases listed within each vulnerability
                for alias in vuln.get("aliases", []):
                    if alias != cve_id:
                        aliases.add(alias)
                # Track affected packages
                for affected in vuln.get("affected", []):
                    pkg = affected.get("package", {})
                    if pkg:
                        pkg_entry = {
                            "ecosystem": pkg.get("ecosystem", ""),
                            "name": pkg.get("name", ""),
                        }
                        if pkg_entry not in affected_packages:
                            affected_packages.append(pkg_entry)
                if vuln_id not in [e["id"] for e in osv_entries]:
                    osv_entries.append(
                        {
                            "id": vuln_id,
                            "summary": vuln.get("summary", ""),
                            "modified": vuln.get("modified", ""),
                        }
                    )
    except (requests.RequestException, ValueError):
        pass

    return {
        "aliases": sorted(aliases),
        "affected_packages": affected_packages,
        "osv_entries": osv_entries,
    }


# ---------------------------------------------------------------------------
# NVD reference extraction
# ---------------------------------------------------------------------------


def _extract_nvd_advisory_refs(cve_id: str) -> list[dict[str, Any]]:
    """Extract advisory references from NVD data.

    Parses NVD reference URLs to identify vendor advisory IDs embedded in them.
    """
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY")
    if api_key:
        headers["apiKey"] = api_key

    refs: list[dict[str, Any]] = []
    try:
        resp = _get_with_retry(url, headers=headers, timeout=20)
        if resp.status_code != 200:
            return refs

        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return refs

        cve_data = vulns[0].get("cve", {})
        references = cve_data.get("references", [])

        for ref in references:
            ref_url = ref.get("url", "")
            ref_tags = ref.get("tags", [])

            # Extract advisory IDs from known URL patterns
            advisory_id = _extract_advisory_from_url(ref_url)
            if advisory_id:
                refs.append(
                    {
                        "id": advisory_id,
                        "url": ref_url,
                        "tags": ref_tags,
                    }
                )

    except (requests.RequestException, ValueError, KeyError):
        pass

    return refs


# Advisory URL patterns
_URL_PATTERNS: list[tuple[str, str | None]] = [
    # GitHub Security Advisory
    (r"github\.com/advisories/(GHSA-[\w-]+)", None),
    # Red Hat
    (r"access\.redhat\.com/errata/(RHSA-\d+:\d+)", None),
    (r"access\.redhat\.com/errata/(RHBA-\d+:\d+)", None),
    # Debian
    (r"debian\.org/security/\d+/(dsa-\d+)", "upper"),
    # Ubuntu
    (r"ubuntu\.com/security/notices/(USN-[\d.-]+)", None),
    # SUSE
    (r"suse\.com/.*/(SUSE-SU-[\d:-]+)", None),
]


def _extract_advisory_from_url(url: str) -> str | None:
    """Extract an advisory ID from a known advisory URL pattern."""
    for pattern, transform in _URL_PATTERNS:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            advisory_id = match.group(1)
            if transform == "upper":
                advisory_id = advisory_id.upper()
            return advisory_id
    return None


# ---------------------------------------------------------------------------
# VulnCheck cross-references (optional)
# ---------------------------------------------------------------------------


def _query_vulncheck_aliases(cve_id: str) -> list[str]:
    """Query VulnCheck for additional advisory cross-references.

    Requires VULNCHECK_API_KEY environment variable.
    Only queries vulncheck-kev (free tier).
    """
    api_key = os.environ.get("VULNCHECK_API_KEY")
    if not api_key:
        return []

    url = f"https://api.vulncheck.com/v3/index/vulncheck-kev?cve={cve_id}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        resp = _get_with_retry(url, headers=headers)
        if resp.status_code != 200:
            return []

        data = resp.json()
        items = data.get("data", [])
        aliases: list[str] = []
        for item in items:
            # VulnCheck KEV entries may reference other advisory IDs
            for ref in item.get("references", []):
                advisory_id = _extract_advisory_from_url(ref)
                if advisory_id and advisory_id not in aliases:
                    aliases.append(advisory_id)
        return aliases

    except (requests.RequestException, ValueError):
        return []


# ---------------------------------------------------------------------------
# Standalone function (used by CLI directly)
# ---------------------------------------------------------------------------


def fetch_advisory_aliases(cve_id: str, include_urls: bool = True) -> dict[str, Any]:
    """Resolve advisory aliases for a CVE. Returns structured dict.

    This is the standalone logic callable from both the Strands tool interface
    and the CLI subcommand.
    """
    all_aliases: dict[str, dict] = {}  # id -> metadata
    sources_queried: list[str] = []
    errors: list[str] = []
    osv_result: dict[str, Any] = {"affected_packages": []}

    # 1. OSV.dev (primary source)
    try:
        osv_result = _query_osv_aliases(cve_id)
        sources_queried.append("osv.dev")
        for alias_id in osv_result["aliases"]:
            if alias_id not in all_aliases:
                classified = _classify_advisory(alias_id)
                classified["source"] = "osv.dev"
                all_aliases[alias_id] = classified
    except Exception as exc:
        errors.append(f"OSV.dev: {exc}")

    # 2. NVD references
    try:
        nvd_refs = _extract_nvd_advisory_refs(cve_id)
        sources_queried.append("nvd")
        for ref in nvd_refs:
            ref_id = ref["id"]
            if ref_id not in all_aliases:
                classified = _classify_advisory(ref_id)
                classified["source"] = "nvd"
                if ref.get("url"):
                    classified["url"] = ref["url"]
                all_aliases[ref_id] = classified
    except Exception as exc:
        errors.append(f"NVD: {exc}")

    # 3. VulnCheck (optional, only if API key available)
    try:
        vc_aliases = _query_vulncheck_aliases(cve_id)
        if vc_aliases:
            sources_queried.append("vulncheck")
        for alias_id in vc_aliases:
            if alias_id not in all_aliases:
                classified = _classify_advisory(alias_id)
                classified["source"] = "vulncheck"
                all_aliases[alias_id] = classified
    except Exception as exc:
        errors.append(f"VulnCheck: {exc}")

    # Build structured result
    aliases_list = sorted(all_aliases.values(), key=lambda x: x["id"])

    # Group by database
    by_database: dict[str, list[dict]] = {}
    for alias in aliases_list:
        db = alias["database"]
        if db not in by_database:
            by_database[db] = []
        entry: dict[str, Any] = {"id": alias["id"]}
        if include_urls and alias.get("url"):
            entry["url"] = alias["url"]
        entry["source"] = alias.get("source", "unknown")
        by_database[db].append(entry)

    # Build affected packages summary
    affected_packages = osv_result.get("affected_packages", [])

    result: dict[str, Any] = {
        "cve_id": cve_id,
        "total_aliases": len(aliases_list),
        "databases_found": len(by_database),
        "sources_queried": sources_queried,
        "aliases_by_database": by_database,
        "affected_packages": affected_packages[:20],
    }
    if errors:
        result["errors"] = errors

    return result


# ---------------------------------------------------------------------------
# Main tool function (Strands SDK interface)
# ---------------------------------------------------------------------------


def resolve_advisory_aliases(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Resolve all advisory aliases for a CVE across vulnerability databases."""
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id", "").strip().upper()
    include_urls = tool_input.get("include_urls", True)

    if not cve_id:
        return {
            "toolUseId": tool["toolUseId"],
            "status": "error",
            "content": [{"text": "Error: cve_id is required."}],
        }

    if not re.match(r"^CVE-\d{4}-\d{4,}$", cve_id):
        return {
            "toolUseId": tool["toolUseId"],
            "status": "error",
            "content": [{"text": f"Error: Invalid CVE ID format: {cve_id}"}],
        }

    result = fetch_advisory_aliases(cve_id, include_urls=include_urls)

    output = json.dumps(result, indent=2)
    log_tool_output_size("resolve_advisory_aliases", output)

    return {
        "toolUseId": tool["toolUseId"],
        "status": "success",
        "content": [{"text": output}],
    }
