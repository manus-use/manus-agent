"""Tool: get_attack_map

Maps a CVE to MITRE ATT&CK techniques and tactics by resolving the
CVE → CWE → CAPEC → ATT&CK chain.

Data flow:
1. Fetch CVE from NVD to extract CWE IDs.
2. For each CWE, fetch the CAPEC (Common Attack Pattern Enumeration) entries
   from MITRE's CAPEC XML data (via capec.mitre.org).
3. Each CAPEC entry maps to ATT&CK techniques via the Related_Attack_Patterns
   and Taxonomy_Mappings fields.
4. Return structured ATT&CK technique/tactic mappings with kill-chain context.

This enables defenders to understand *how* a vulnerability might be exploited
in real-world attack campaigns and which defensive controls apply.
"""

from __future__ import annotations

import re
import time
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

__all__ = ["get_attack_map", "TOOL_SPEC"]

# ---------------------------------------------------------------------------
# TOOL_SPEC (Strands module-based tool interface)
# ---------------------------------------------------------------------------

TOOL_SPEC = {
    "name": "get_attack_map",
    "description": (
        "Maps a CVE to MITRE ATT&CK techniques and tactics by resolving the "
        "CVE → CWE → CAPEC → ATT&CK chain. Returns structured technique IDs, "
        "names, tactics (kill-chain phases), and the mapping path for each. "
        "Useful for understanding how a vulnerability fits into real-world "
        "attack campaigns and which defensive controls apply."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": (
                        "CVE identifier to map (e.g. 'CVE-2024-3094'). "
                        "The tool fetches the CVE's CWE weaknesses from NVD "
                        "and resolves them to ATT&CK techniques."
                    ),
                },
                "output_format": {
                    "type": "string",
                    "enum": ["text", "json"],
                    "description": "Output format: 'text' (human-readable) or 'json' (structured).",
                },
            },
            "required": ["cve_id"],
        }
    },
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_CAPEC_API_URL = "https://capec.mitre.org/data/xml/views/1000.xml.zip"
_CAPEC_DETAIL_URL = "https://capec.mitre.org/data/definitions/{capec_id}.html"

# CWE → CAPEC mapping endpoint (lightweight HTML scrape)
_CWE_CAPEC_URL = "https://cwe.mitre.org/data/definitions/{cwe_num}.html"

# ATT&CK technique URL
_ATTACK_URL = "https://attack.mitre.org/techniques/{technique_id}/"

_CVE_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)
_CWE_PATTERN = re.compile(r"CWE-(\d+)", re.IGNORECASE)
_CAPEC_PATTERN = re.compile(r"CAPEC-(\d+)", re.IGNORECASE)

# CAPEC detail URL for scraping ATT&CK mappings
_CAPEC_HTML_URL = "https://capec.mitre.org/data/definitions/{capec_num}.html"

# ATT&CK technique pattern (e.g., T1059, T1059.001)
_TECHNIQUE_PATTERN = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b")

# Common CWE → ATT&CK technique mappings (curated from MITRE's published data)
# This serves as a fast-path lookup when network access to CAPEC fails.
# Source: https://capec.mitre.org and https://attack.mitre.org
_CWE_ATTACK_FALLBACK: dict[int, list[dict[str, str]]] = {
    # Injection weaknesses
    79: [
        {"technique_id": "T1059.007", "technique_name": "JavaScript", "tactic": "Execution"},
        {"technique_id": "T1185", "technique_name": "Browser Session Hijacking", "tactic": "Collection"},
    ],
    89: [
        {"technique_id": "T1190", "technique_name": "Exploit Public-Facing Application", "tactic": "Initial Access"},
    ],
    78: [
        {"technique_id": "T1059", "technique_name": "Command and Scripting Interpreter", "tactic": "Execution"},
        {"technique_id": "T1190", "technique_name": "Exploit Public-Facing Application", "tactic": "Initial Access"},
    ],
    77: [
        {"technique_id": "T1059", "technique_name": "Command and Scripting Interpreter", "tactic": "Execution"},
    ],
    # Memory corruption
    120: [
        {"technique_id": "T1203", "technique_name": "Exploitation for Client Execution", "tactic": "Execution"},
    ],
    122: [
        {"technique_id": "T1203", "technique_name": "Exploitation for Client Execution", "tactic": "Execution"},
    ],
    125: [
        {"technique_id": "T1005", "technique_name": "Data from Local System", "tactic": "Collection"},
    ],
    416: [
        {"technique_id": "T1203", "technique_name": "Exploitation for Client Execution", "tactic": "Execution"},
        {
            "technique_id": "T1068",
            "technique_name": "Exploitation for Privilege Escalation",
            "tactic": "Privilege Escalation",
        },
    ],
    787: [
        {"technique_id": "T1203", "technique_name": "Exploitation for Client Execution", "tactic": "Execution"},
    ],
    # Auth/access control
    287: [
        {"technique_id": "T1078", "technique_name": "Valid Accounts", "tactic": "Defense Evasion"},
        {"technique_id": "T1110", "technique_name": "Brute Force", "tactic": "Credential Access"},
    ],
    306: [
        {"technique_id": "T1078", "technique_name": "Valid Accounts", "tactic": "Defense Evasion"},
    ],
    862: [
        {
            "technique_id": "T1548",
            "technique_name": "Abuse Elevation Control Mechanism",
            "tactic": "Privilege Escalation",
        },
    ],
    863: [
        {
            "technique_id": "T1548",
            "technique_name": "Abuse Elevation Control Mechanism",
            "tactic": "Privilege Escalation",
        },
    ],
    # Information exposure
    200: [
        {"technique_id": "T1005", "technique_name": "Data from Local System", "tactic": "Collection"},
        {"technique_id": "T1530", "technique_name": "Data from Cloud Storage", "tactic": "Collection"},
    ],
    # Deserialization
    502: [
        {"technique_id": "T1059", "technique_name": "Command and Scripting Interpreter", "tactic": "Execution"},
        {"technique_id": "T1190", "technique_name": "Exploit Public-Facing Application", "tactic": "Initial Access"},
    ],
    # Path traversal
    22: [
        {"technique_id": "T1083", "technique_name": "File and Directory Discovery", "tactic": "Discovery"},
        {"technique_id": "T1005", "technique_name": "Data from Local System", "tactic": "Collection"},
    ],
    # SSRF
    918: [
        {"technique_id": "T1090", "technique_name": "Proxy", "tactic": "Command and Control"},
        {"technique_id": "T1190", "technique_name": "Exploit Public-Facing Application", "tactic": "Initial Access"},
    ],
    # XXE
    611: [
        {"technique_id": "T1005", "technique_name": "Data from Local System", "tactic": "Collection"},
        {"technique_id": "T1190", "technique_name": "Exploit Public-Facing Application", "tactic": "Initial Access"},
    ],
    # Cryptographic issues
    327: [
        {"technique_id": "T1557", "technique_name": "Adversary-in-the-Middle", "tactic": "Credential Access"},
    ],
    # Supply chain
    829: [
        {"technique_id": "T1195", "technique_name": "Supply Chain Compromise", "tactic": "Initial Access"},
    ],
    # Privilege escalation
    269: [
        {
            "technique_id": "T1068",
            "technique_name": "Exploitation for Privilege Escalation",
            "tactic": "Privilege Escalation",
        },
    ],
}

# Tactic ordering for display
_TACTIC_ORDER = [
    "Reconnaissance",
    "Resource Development",
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command and Control",
    "Exfiltration",
    "Impact",
]

_MAX_RETRIES = 3
_RETRY_DELAY = 1.0
_TIMEOUT = 15


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _get_with_retry(
    url: str,
    *,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = _TIMEOUT,
    max_retries: int = _MAX_RETRIES,
) -> requests.Response:
    """GET with exponential back-off retry on transient failures."""
    default_headers = {"User-Agent": "manus-agent/attack-map (github.com/manus-use/manus-agent)"}
    if headers:
        default_headers.update(headers)

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=default_headers, timeout=timeout)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", _RETRY_DELAY * (2**attempt)))
                time.sleep(min(retry_after, 30))
                continue
            if resp.status_code >= 500:
                time.sleep(_RETRY_DELAY * (2**attempt))
                continue
            return resp
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(_RETRY_DELAY * (2**attempt))
    raise last_exc or requests.exceptions.ConnectionError("Max retries exceeded")


# ---------------------------------------------------------------------------
# NVD: CVE → CWE extraction
# ---------------------------------------------------------------------------


def _fetch_cwes_from_nvd(cve_id: str) -> list[str]:
    """Fetch CWE IDs associated with a CVE from NVD API v2.0."""
    import os

    params = {"cveId": cve_id.upper()}
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY")
    if api_key:
        headers["apiKey"] = api_key

    resp = _get_with_retry(_NVD_API_URL, params=params, headers=headers)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()

    data = resp.json()
    cwes: list[str] = []

    vulnerabilities = data.get("vulnerabilities", [])
    if not vulnerabilities:
        return []

    cve_item = vulnerabilities[0].get("cve", {})
    weaknesses = cve_item.get("weaknesses", [])
    for weakness in weaknesses:
        for desc in weakness.get("description", []):
            value = desc.get("value", "")
            match = _CWE_PATTERN.match(value)
            if match:
                cwes.append(f"CWE-{match.group(1)}")

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for cwe in cwes:
        if cwe not in seen:
            seen.add(cwe)
            unique.append(cwe)
    return unique


# ---------------------------------------------------------------------------
# CWE → CAPEC resolution (HTML scrape from cwe.mitre.org)
# ---------------------------------------------------------------------------


def _fetch_capecs_for_cwe(cwe_num: int) -> list[int]:
    """Scrape CAPEC IDs related to a CWE from cwe.mitre.org."""
    url = _CWE_CAPEC_URL.format(cwe_num=cwe_num)
    try:
        resp = _get_with_retry(url, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return []
    except (requests.exceptions.RequestException, OSError):
        return []

    # Look for CAPEC references in the HTML
    capec_ids: list[int] = []
    for match in _CAPEC_PATTERN.finditer(resp.text):
        capec_id = int(match.group(1))
        if capec_id not in capec_ids:
            capec_ids.append(capec_id)
    return capec_ids


# ---------------------------------------------------------------------------
# CAPEC → ATT&CK resolution (HTML scrape from capec.mitre.org)
# ---------------------------------------------------------------------------


def _fetch_attack_for_capec(capec_num: int) -> list[dict[str, str]]:
    """Scrape ATT&CK technique mappings from a CAPEC detail page."""
    url = _CAPEC_HTML_URL.format(capec_num=capec_num)
    try:
        resp = _get_with_retry(url, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return []
    except (requests.exceptions.RequestException, OSError):
        return []

    techniques: list[dict[str, str]] = []
    html = resp.text

    # Look for ATT&CK technique references (T1234 or T1234.001)
    # CAPEC pages have a "Taxonomy Mappings" section with ATT&CK refs
    technique_ids = _TECHNIQUE_PATTERN.findall(html)
    seen: set[str] = set()

    for tid in technique_ids:
        if tid in seen:
            continue
        seen.add(tid)

        # Try to extract technique name and tactic from context
        name = _extract_technique_name(html, tid)
        tactic = _extract_tactic(html, tid)

        techniques.append(
            {
                "technique_id": tid,
                "technique_name": name or tid,
                "tactic": tactic or "Unknown",
            }
        )

    return techniques


def _extract_technique_name(html: str, technique_id: str) -> str | None:
    """Try to extract the technique name from surrounding HTML context."""
    # Look for patterns like "T1059 - Command and Scripting Interpreter" or
    # "T1059</a>.*?Command and Scripting Interpreter"
    patterns = [
        re.compile(
            rf"{re.escape(technique_id)}\s*[-:]\s*([^<\n]+)",
            re.IGNORECASE,
        ),
        re.compile(
            rf"{re.escape(technique_id)}</a>\s*[-:]?\s*([^<\n]+)",
            re.IGNORECASE,
        ),
        re.compile(
            rf">\s*{re.escape(technique_id)}\s*</a>\s*</td>\s*<td[^>]*>\s*([^<]+)",
            re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(html)
        if match:
            name = match.group(1).strip().rstrip(".")
            if name and len(name) > 2 and not name.startswith("T1"):
                return name
    return None


def _extract_tactic(html: str, technique_id: str) -> str | None:
    """Try to extract the tactic (kill-chain phase) from surrounding context."""
    # Look for tactic names near the technique ID
    idx = html.find(technique_id)
    if idx == -1:
        return None

    # Search in a window around the technique reference
    window = html[max(0, idx - 500) : idx + 500]
    for tactic in _TACTIC_ORDER:
        if tactic.lower() in window.lower():
            return tactic
    return None


# ---------------------------------------------------------------------------
# Core mapping logic
# ---------------------------------------------------------------------------


def _map_cve_to_attack(cve_id: str) -> dict[str, Any]:
    """Map a CVE to ATT&CK techniques via the CWE → CAPEC → ATT&CK chain.

    Returns a structured result dict with:
    - cve_id: the input CVE
    - cwes: list of CWE IDs found
    - mappings: list of technique mappings with provenance
    - tactics_summary: ordered list of unique tactics involved
    """
    cve_id = cve_id.upper().strip()

    # Step 1: CVE → CWE (from NVD)
    cwes = _fetch_cwes_from_nvd(cve_id)
    if not cwes:
        return {
            "cve_id": cve_id,
            "cwes": [],
            "mappings": [],
            "tactics_summary": [],
            "error": f"No CWE mappings found for {cve_id} in NVD.",
        }

    # Step 2 & 3: CWE → CAPEC → ATT&CK
    all_mappings: list[dict[str, Any]] = []
    seen_techniques: set[str] = set()

    for cwe in cwes:
        cwe_num = int(_CWE_PATTERN.match(cwe).group(1))  # type: ignore[union-attr]

        # Try CAPEC-based resolution first
        capec_ids = _fetch_capecs_for_cwe(cwe_num)
        cwe_resolved = False

        for capec_id in capec_ids:
            techniques = _fetch_attack_for_capec(capec_id)
            for tech in techniques:
                key = tech["technique_id"]
                if key not in seen_techniques:
                    seen_techniques.add(key)
                    all_mappings.append(
                        {
                            "technique_id": tech["technique_id"],
                            "technique_name": tech["technique_name"],
                            "tactic": tech["tactic"],
                            "path": f"{cwe} → CAPEC-{capec_id} → {tech['technique_id']}",
                            "source": "capec",
                        }
                    )
                    cwe_resolved = True

        # Fallback: use curated CWE → ATT&CK mapping
        if not cwe_resolved and cwe_num in _CWE_ATTACK_FALLBACK:
            for tech in _CWE_ATTACK_FALLBACK[cwe_num]:
                key = tech["technique_id"]
                if key not in seen_techniques:
                    seen_techniques.add(key)
                    all_mappings.append(
                        {
                            "technique_id": tech["technique_id"],
                            "technique_name": tech["technique_name"],
                            "tactic": tech["tactic"],
                            "path": f"{cwe} → {tech['technique_id']} (curated mapping)",
                            "source": "curated",
                        }
                    )

    # Build tactics summary (ordered by kill-chain)
    tactics_seen: set[str] = set()
    for m in all_mappings:
        if m["tactic"] != "Unknown":
            tactics_seen.add(m["tactic"])
    tactics_summary = [t for t in _TACTIC_ORDER if t in tactics_seen]

    return {
        "cve_id": cve_id,
        "cwes": cwes,
        "mappings": all_mappings,
        "tactics_summary": tactics_summary,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _format_text(result: dict[str, Any]) -> str:
    """Format the mapping result as human-readable text."""
    lines: list[str] = []
    cve_id = result["cve_id"]

    lines.append(f"## ATT&CK Mapping: {cve_id}")
    lines.append("")

    if result.get("error"):
        lines.append(f"⚠ {result['error']}")
        return "\n".join(lines)

    # CWEs
    lines.append(f"**Weaknesses:** {', '.join(result['cwes'])}")
    lines.append("")

    # Tactics overview
    if result["tactics_summary"]:
        lines.append("**Kill-Chain Coverage:**")
        for tactic in result["tactics_summary"]:
            lines.append(f"  • {tactic}")
        lines.append("")

    # Technique details
    if result["mappings"]:
        lines.append("**ATT&CK Techniques:**")
        lines.append("")
        for m in result["mappings"]:
            lines.append(f"  [{m['technique_id']}] {m['technique_name']}")
            lines.append(f"    Tactic: {m['tactic']}")
            lines.append(f"    Path:   {m['path']}")
            url = _ATTACK_URL.format(technique_id=m["technique_id"].replace(".", "/"))
            lines.append(f"    URL:    {url}")
            lines.append("")
    else:
        lines.append("No ATT&CK technique mappings could be resolved.")

    # Summary
    lines.append("---")
    lines.append(f"Total: {len(result['mappings'])} technique(s) across {len(result['tactics_summary'])} tactic(s)")

    return "\n".join(lines)


def _format_json(result: dict[str, Any]) -> dict[str, Any]:
    """Format the mapping result as a structured JSON-serializable dict."""
    output: dict[str, Any] = {
        "cve_id": result["cve_id"],
        "cwes": result["cwes"],
        "technique_count": len(result["mappings"]),
        "tactic_count": len(result["tactics_summary"]),
        "tactics": result["tactics_summary"],
        "techniques": [],
    }

    if result.get("error"):
        output["error"] = result["error"]

    for m in result["mappings"]:
        output["techniques"].append(
            {
                "id": m["technique_id"],
                "name": m["technique_name"],
                "tactic": m["tactic"],
                "mapping_path": m["path"],
                "source": m["source"],
                "url": _ATTACK_URL.format(technique_id=m["technique_id"].replace(".", "/")),
            }
        )

    return output


# ---------------------------------------------------------------------------
# Strands tool handler
# ---------------------------------------------------------------------------


def get_attack_map(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands tool entry point for get_attack_map."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool.get("input", {})

    cve_id = tool_input.get("cve_id", "").strip()
    output_format = tool_input.get("output_format", "text")

    # Validate CVE ID
    if not cve_id or not _CVE_PATTERN.match(cve_id):
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Invalid CVE ID format: '{cve_id}'. Expected CVE-YYYY-NNNNN."}],
        }
        log_tool_output_size("get_attack_map", result)
        return result

    try:
        mapping = _map_cve_to_attack(cve_id)
    except requests.exceptions.RequestException as exc:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Network error mapping {cve_id} to ATT&CK: {exc}"}],
        }
        log_tool_output_size("get_attack_map", result)
        return result
    except Exception as exc:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Unexpected error mapping {cve_id}: {exc}"}],
        }
        log_tool_output_size("get_attack_map", result)
        return result

    if output_format == "json":
        result = {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"json": _format_json(mapping)}],
        }
    else:
        result = {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": _format_text(mapping)}],
        }

    log_tool_output_size("get_attack_map", result)
    return result


# ---------------------------------------------------------------------------
# CLI-facing function (called from cli.py)
# ---------------------------------------------------------------------------


def fetch_attack_map(cve_id: str, *, output_format: str = "text") -> str | dict[str, Any]:
    """Public API for CLI usage — returns formatted text or structured dict."""
    mapping = _map_cve_to_attack(cve_id)
    if output_format == "json":
        return _format_json(mapping)
    return _format_text(mapping)
