"""
Tool: decode_cvss_vector

Parses and explains a CVSS v3.0/v3.1 vector string, producing a structured
breakdown with human-readable descriptions for each metric component, the
computed base score, severity rating, and actionable security context.

Unlike the ``score_exploit_complexity`` tool (which scores PoC code difficulty),
this tool operates purely on the CVSS vector *specification* to answer:

  1. What does each metric mean in plain language?
  2. What is the computed base score and severity?
  3. Which metrics contribute most to the severity?
  4. What attack conditions does this vector describe?

Inputs:
  - A CVSS v3.0 or v3.1 vector string (e.g. "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
  - OR a CVE ID (fetches the vector from NVD automatically)

Output:
  - Structured dict with per-metric explanations, computed score, severity,
    attack summary, and remediation priority signal.

No external API calls when a vector string is provided directly.
NVD lookup only when a CVE ID is given instead of a vector.

CLI: ``manus-agent decode-cvss <VECTOR_OR_CVE>``
"""

from __future__ import annotations

import math
import re
from typing import Any

import requests
from strands import tool

__all__ = ["decode_cvss_vector"]

# ---------------------------------------------------------------------------
# CVSS v3.x metric definitions
# ---------------------------------------------------------------------------

# Metric abbreviation → full name
_METRIC_NAMES: dict[str, str] = {
    "AV": "Attack Vector",
    "AC": "Attack Complexity",
    "PR": "Privileges Required",
    "UI": "User Interaction",
    "S": "Scope",
    "C": "Confidentiality Impact",
    "I": "Integrity Impact",
    "A": "Availability Impact",
}

# Metric value abbreviation → full value name
_METRIC_VALUES: dict[str, dict[str, str]] = {
    "AV": {"N": "Network", "A": "Adjacent", "L": "Local", "P": "Physical"},
    "AC": {"L": "Low", "H": "High"},
    "PR": {"N": "None", "L": "Low", "H": "High"},
    "UI": {"N": "None", "R": "Required"},
    "S": {"U": "Unchanged", "C": "Changed"},
    "C": {"N": "None", "L": "Low", "H": "High"},
    "I": {"N": "None", "L": "Low", "H": "High"},
    "A": {"N": "None", "L": "Low", "H": "High"},
}

# Human-readable explanations for each metric/value combination
_EXPLANATIONS: dict[str, dict[str, str]] = {
    "AV": {
        "N": "Exploitable remotely over the network (e.g. via HTTP, email, or any network service). No physical or local access needed.",
        "A": "Exploitable from an adjacent network (e.g. same WiFi, Bluetooth, or local subnet). Not reachable from the internet.",
        "L": "Requires local access to the target system (e.g. local shell, physical console, or malicious local application).",
        "P": "Requires physical access to the hardware (e.g. USB port, JTAG, or direct hardware manipulation).",
    },
    "AC": {
        "L": "No specialized conditions needed. The attack can be reliably reproduced every time against the vulnerable component.",
        "H": "Attack requires specific conditions beyond the attacker's control (e.g. race condition, non-default configuration, or precise timing).",
    },
    "PR": {
        "N": "No authentication or privileges needed. Any anonymous attacker can exploit this.",
        "L": "Requires basic user-level privileges (e.g. a regular authenticated account).",
        "H": "Requires elevated/administrative privileges (e.g. root, admin, or highly privileged role).",
    },
    "UI": {
        "N": "No user interaction required. The attack can succeed without any victim action.",
        "R": "Requires a user to perform an action (e.g. clicking a link, opening a file, or visiting a page).",
    },
    "S": {
        "U": "Impact is limited to the vulnerable component only. No cascading effect on other systems.",
        "C": "Impact extends beyond the vulnerable component. Can affect other system resources or components (e.g. sandbox escape, cross-tenant breach).",
    },
    "C": {
        "N": "No confidentiality impact. No information disclosure.",
        "L": "Some restricted information is disclosed, but the attacker has limited control over what is obtained.",
        "H": "Total confidentiality loss. All information in the vulnerable component is disclosed to the attacker.",
    },
    "I": {
        "N": "No integrity impact. No data modification possible.",
        "L": "Some data modification is possible, but the attacker has limited control over scope or consequence.",
        "H": "Total integrity loss. The attacker can modify any/all data in the vulnerable component.",
    },
    "A": {
        "N": "No availability impact. The system remains operational.",
        "L": "Reduced performance or partial denial of service. Some functionality is degraded.",
        "H": "Total availability loss. The attacker can completely deny access to the vulnerable component (full DoS).",
    },
}

# Numeric weights for score calculation (CVSS v3.1 specification)
_AV_WEIGHTS: dict[str, float] = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.20}
_AC_WEIGHTS: dict[str, float] = {"L": 0.77, "H": 0.44}
_PR_WEIGHTS_UNCHANGED: dict[str, float] = {"N": 0.85, "L": 0.62, "H": 0.27}
_PR_WEIGHTS_CHANGED: dict[str, float] = {"N": 0.85, "L": 0.68, "H": 0.50}
_UI_WEIGHTS: dict[str, float] = {"N": 0.85, "R": 0.62}
_CIA_WEIGHTS: dict[str, float] = {"H": 0.56, "L": 0.22, "N": 0.0}

# Severity thresholds
_SEVERITY_THRESHOLDS: list[tuple[float, str]] = [
    (0.0, "None"),
    (0.1, "Low"),
    (4.0, "Medium"),
    (7.0, "High"),
    (9.0, "Critical"),
]

# Required base metric keys (order matters for canonical form)
_REQUIRED_METRICS = ("AV", "AC", "PR", "UI", "S", "C", "I", "A")

# Regex for a valid CVSS v3.x vector string
_VECTOR_RE = re.compile(r"^CVSS:3\.[01]/AV:[NALP]/AC:[LH]/PR:[NLH]/UI:[NR]/S:[UC]/C:[NLH]/I:[NLH]/A:[NLH]$")


# ---------------------------------------------------------------------------
# Score computation (pure CVSS v3.1 base score algorithm)
# ---------------------------------------------------------------------------


def _compute_base_score(metrics: dict[str, str]) -> float:
    """Compute the CVSS v3.1 base score from parsed metric values.

    Implements the exact formula from the CVSS v3.1 specification:
    https://www.first.org/cvss/v3.1/specification-document
    """
    scope_changed = metrics["S"] == "C"

    # Exploitability sub-score
    av = _AV_WEIGHTS[metrics["AV"]]
    ac = _AC_WEIGHTS[metrics["AC"]]
    pr_map = _PR_WEIGHTS_CHANGED if scope_changed else _PR_WEIGHTS_UNCHANGED
    pr = pr_map[metrics["PR"]]
    ui = _UI_WEIGHTS[metrics["UI"]]

    exploitability = 8.22 * av * ac * pr * ui

    # Impact sub-score
    conf_impact = 1 - _CIA_WEIGHTS[metrics["C"]]
    integ_impact = 1 - _CIA_WEIGHTS[metrics["I"]]
    avail_impact = 1 - _CIA_WEIGHTS[metrics["A"]]

    isc_base = 1 - (conf_impact * integ_impact * avail_impact)

    if scope_changed:
        impact = 7.52 * (isc_base - 0.029) - 3.25 * (isc_base - 0.02) ** 15
    else:
        impact = 6.42 * isc_base

    # If impact is <= 0, the base score is 0
    if impact <= 0:
        return 0.0

    # Final base score
    if scope_changed:
        base_score = min(1.08 * (impact + exploitability), 10.0)
    else:
        base_score = min(impact + exploitability, 10.0)

    # Round up to nearest tenth (CVSS spec uses ceiling at first decimal)
    return math.ceil(base_score * 10) / 10


def _severity_from_score(score: float) -> str:
    """Return the qualitative severity rating for a CVSS base score."""
    if score == 0.0:
        return "None"
    for threshold, label in reversed(_SEVERITY_THRESHOLDS):
        if score >= threshold:
            return label
    return "None"


# ---------------------------------------------------------------------------
# Vector string parsing
# ---------------------------------------------------------------------------


def _parse_vector(vector_string: str) -> dict[str, str] | None:
    """Parse a CVSS v3.x vector string into a dict of metric → value.

    Returns None if the vector is malformed.
    """
    vector_string = vector_string.strip()
    if not _VECTOR_RE.match(vector_string):
        return None

    # Skip the "CVSS:3.x/" prefix
    parts = vector_string.split("/")[1:]  # First element is "CVSS:3.x"
    metrics: dict[str, str] = {}
    for part in parts:
        if ":" not in part:
            return None
        key, value = part.split(":", 1)
        metrics[key] = value

    # Validate all required metrics are present
    for req in _REQUIRED_METRICS:
        if req not in metrics:
            return None

    return metrics


# ---------------------------------------------------------------------------
# Attack summary generation
# ---------------------------------------------------------------------------


def _generate_attack_summary(metrics: dict[str, str]) -> str:
    """Generate a concise natural-language attack summary from CVSS metrics."""
    parts: list[str] = []

    # Attack surface
    av_desc = {
        "N": "remotely over the network",
        "A": "from an adjacent network",
        "L": "with local system access",
        "P": "with physical hardware access",
    }
    parts.append(f"This vulnerability is exploitable {av_desc[metrics['AV']]}")

    # Complexity
    if metrics["AC"] == "L":
        parts.append("with low complexity (reliable exploitation)")
    else:
        parts.append("but requires specific conditions (unreliable exploitation)")

    # Authentication
    pr_desc = {"N": "no authentication", "L": "basic user privileges", "H": "administrative privileges"}
    parts.append(f"needing {pr_desc[metrics['PR']]}")

    # User interaction
    if metrics["UI"] == "R":
        parts.append("and victim interaction (e.g. clicking a link)")
    else:
        parts.append("with no victim interaction")

    summary = ", ".join(parts) + "."

    # Impact summary
    impacts: list[str] = []
    if metrics["C"] == "H":
        impacts.append("full data disclosure")
    elif metrics["C"] == "L":
        impacts.append("partial data disclosure")
    if metrics["I"] == "H":
        impacts.append("complete data modification")
    elif metrics["I"] == "L":
        impacts.append("limited data modification")
    if metrics["A"] == "H":
        impacts.append("total denial of service")
    elif metrics["A"] == "L":
        impacts.append("degraded availability")

    if impacts:
        summary += f" Successful exploitation leads to: {', '.join(impacts)}."

    if metrics["S"] == "C":
        summary += " Impact extends beyond the vulnerable component to other systems."

    return summary


def _remediation_priority(score: float, metrics: dict[str, str]) -> dict[str, Any]:
    """Generate remediation priority guidance based on the vector."""
    severity = _severity_from_score(score)

    if severity == "Critical":
        urgency = "IMMEDIATE"
        guidance = "Patch or mitigate within 24-48 hours. This vulnerability poses extreme risk."
    elif severity == "High":
        urgency = "HIGH"
        guidance = "Patch within 1-2 weeks. Prioritize if the system is internet-facing."
    elif severity == "Medium":
        urgency = "MODERATE"
        guidance = "Patch within 30 days as part of regular maintenance."
    elif severity == "Low":
        urgency = "LOW"
        guidance = "Address during next scheduled maintenance window."
    else:
        urgency = "INFORMATIONAL"
        guidance = "No immediate action required."

    # Adjust for specific attack conditions
    factors: list[str] = []
    if metrics["AV"] == "N" and metrics["PR"] == "N" and metrics["UI"] == "N":
        factors.append("Unauthenticated remote exploitation with no user interaction — wormable potential")
    if metrics["AV"] == "N" and metrics["AC"] == "L":
        factors.append("Low-complexity network attack — easily automated at scale")
    if metrics["S"] == "C":
        factors.append("Scope change — can pivot to other systems/components")
    if metrics["C"] == "H" and metrics["I"] == "H" and metrics["A"] == "H":
        factors.append("Full CIA triad impact — complete system compromise")

    return {
        "urgency": urgency,
        "guidance": guidance,
        "escalation_factors": factors,
    }


# ---------------------------------------------------------------------------
# NVD lookup (only when CVE ID is provided instead of vector)
# ---------------------------------------------------------------------------


def _fetch_vector_from_nvd(cve_id: str) -> str | None:
    """Attempt to fetch a CVSS v3.x vector string from NVD for a CVE ID.

    Returns None if the CVE is not found or has no CVSS v3 data.
    """
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id.upper()}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return None

    cve_data = vulns[0].get("cve", {})
    metrics = cve_data.get("metrics", {})

    # Try v3.1 first, then v3.0
    for key in ("cvssMetricV31", "cvssMetricV30"):
        metric_list = metrics.get(key, [])
        if metric_list:
            cvss_data = metric_list[0].get("cvssData", {})
            vector = cvss_data.get("vectorString")
            if vector:
                return vector

    return None


# ---------------------------------------------------------------------------
# Main tool function
# ---------------------------------------------------------------------------


@tool
def decode_cvss_vector(vector_or_cve: str) -> dict[str, Any]:
    """Decode and explain a CVSS v3.0/v3.1 vector string or fetch one for a CVE ID.

    Parses the vector into individual metrics, computes the base score using
    the official CVSS v3.1 formula, and produces human-readable explanations
    for each component along with an attack summary and remediation priority.

    Args:
        vector_or_cve: Either a full CVSS v3.x vector string
            (e.g. "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
            or a CVE ID (e.g. "CVE-2024-3094") to fetch from NVD.

    Returns:
        A dict containing:
        - vector: the canonical CVSS vector string
        - version: CVSS version (3.0 or 3.1)
        - base_score: computed base score (0.0-10.0)
        - severity: qualitative severity (None/Low/Medium/High/Critical)
        - metrics: list of per-metric breakdowns with name, value, and explanation
        - attack_summary: natural-language description of the attack
        - remediation_priority: urgency, guidance, and escalation factors
        - error: error message if parsing failed (only on failure)
    """
    input_str = (vector_or_cve or "").strip()

    if not input_str:
        return {"error": "Input required. Provide a CVSS v3.x vector string or CVE ID."}

    # Determine if input is a CVE ID or a vector string
    vector_string: str | None = None

    if input_str.upper().startswith("CVE-"):
        # Fetch vector from NVD
        vector_string = _fetch_vector_from_nvd(input_str)
        if not vector_string:
            return {
                "error": f"Could not retrieve a CVSS v3.x vector for {input_str.upper()} from NVD. "
                "The CVE may not exist, may lack CVSS v3 scoring, or NVD may be unreachable.",
                "cve_id": input_str.upper(),
            }
    elif input_str.upper().startswith("CVSS:"):
        vector_string = input_str.upper()
    else:
        return {
            "error": f"Invalid input: {input_str!r}. Provide a CVSS v3.x vector string "
            "(starting with 'CVSS:3.x/') or a CVE ID (starting with 'CVE-').",
        }

    # Parse the vector
    metrics = _parse_vector(vector_string)
    if metrics is None:
        return {
            "error": f"Malformed CVSS vector: {vector_string!r}. Expected format: "
            "CVSS:3.1/AV:[N|A|L|P]/AC:[L|H]/PR:[N|L|H]/UI:[N|R]/S:[U|C]/C:[N|L|H]/I:[N|L|H]/A:[N|L|H]",
            "input": vector_string,
        }

    # Compute score
    base_score = _compute_base_score(metrics)
    severity = _severity_from_score(base_score)

    # Extract version
    version = "3.1" if "3.1" in vector_string else "3.0"

    # Build per-metric breakdown
    metric_details: list[dict[str, str]] = []
    for abbr in _REQUIRED_METRICS:
        val = metrics[abbr]
        metric_details.append(
            {
                "abbreviation": abbr,
                "metric": _METRIC_NAMES[abbr],
                "value_code": val,
                "value": _METRIC_VALUES[abbr][val],
                "explanation": _EXPLANATIONS[abbr][val],
            }
        )

    # Generate summaries
    attack_summary = _generate_attack_summary(metrics)
    priority = _remediation_priority(base_score, metrics)

    result: dict[str, Any] = {
        "vector": vector_string,
        "version": version,
        "base_score": base_score,
        "severity": severity,
        "metrics": metric_details,
        "attack_summary": attack_summary,
        "remediation_priority": priority,
    }

    # Include CVE ID if one was provided
    if input_str.upper().startswith("CVE-"):
        result["cve_id"] = input_str.upper()

    return result
