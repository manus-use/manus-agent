"""
Tool for generating a rich Markdown diff-report comparing two CVEs.

Goes beyond the tabular ``compare_cves`` tool by producing a full narrative
document — suitable for tickets, briefings, or security advisories — that
covers CVSS delta analysis, EPSS trajectory divergence, KEV status, CWE
class differences, and a concrete prioritisation verdict with rationale.

Data sources (all degrade independently):
  - NVD          https://services.nvd.nist.gov/rest/json/cves/2.0
  - FIRST EPSS   https://api.first.org/data/v1/epss
  - CISA KEV     https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json

All HTTP calls are performed concurrently via ``ThreadPoolExecutor``.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import re
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

TOOL_SPEC = {
    "name": "diff_report_cves",
    "description": (
        "Generates a detailed Markdown diff-report comparing two CVEs side-by-side. "
        "The report covers CVSS score delta, severity comparison, EPSS exploitation-probability "
        "difference, CISA KEV membership, CWE weakness-class comparison, attack vector and "
        "privilege differences, affected components, and a clear prioritisation verdict "
        "with scoring rationale. "
        "Use when you need a shareable narrative document that explains *why* one CVE is more "
        "urgent than another — suitable for tickets, security briefings, or advisory drafts."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id_a": {
                    "type": "string",
                    "description": "First CVE identifier (e.g. 'CVE-2024-3094').",
                },
                "cve_id_b": {
                    "type": "string",
                    "description": "Second CVE identifier to compare against (e.g. 'CVE-2021-44228').",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["markdown", "json"],
                    "description": "Report format: 'markdown' (default) or 'json' for structured data.",
                },
            },
            "required": ["cve_id_a", "cve_id_b"],
        }
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_CVE_RE = re.compile(r"^CVE-\d{4}-\d+$", re.IGNORECASE)

_SEVERITY_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}

# ──────────────────────────────────────────────────────────────────────────────
# HTTP helpers  (self-contained; no imports from compare_cves)
# ──────────────────────────────────────────────────────────────────────────────


def _fetch_nvd(cve_id: str) -> dict[str, Any]:
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id.upper()}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        vulns = resp.json().get("vulnerabilities", [])
        return vulns[0].get("cve", {}) if vulns else {"error": f"No NVD record for {cve_id}"}
    except requests.RequestException as exc:
        return {"error": f"NVD request failed: {exc}"}


def _fetch_epss(cve_id: str) -> dict[str, Any]:
    try:
        resp = requests.get(
            "https://api.first.org/data/v1/epss",
            params={"cve": cve_id.upper()},
            timeout=15,
        )
        resp.raise_for_status()
        entries = resp.json().get("data", [])
        return entries[0] if entries else {"error": f"No EPSS data for {cve_id}"}
    except requests.RequestException as exc:
        return {"error": f"EPSS request failed: {exc}"}


def _fetch_kev(cve_id: str) -> dict[str, Any]:
    url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        catalog = resp.json()
    except requests.RequestException as exc:
        return {"in_kev": False, "error": f"KEV request failed: {exc}"}
    cid = cve_id.upper()
    for entry in catalog.get("vulnerabilities", []):
        if entry.get("cveID", "").upper() == cid:
            return {
                "in_kev": True,
                "date_added": entry.get("dateAdded"),
                "vendor_project": entry.get("vendorProject"),
                "product": entry.get("product"),
                "required_action": entry.get("requiredAction"),
                "due_date": entry.get("dueDate"),
            }
    return {"in_kev": False}


# ──────────────────────────────────────────────────────────────────────────────
# Data-extraction helpers
# ──────────────────────────────────────────────────────────────────────────────


def _extract_cvss(nvd: dict[str, Any]) -> dict[str, Any]:
    """Return the highest-version CVSS data available (3.1 -> 3.0 -> 2.0)."""
    metrics = nvd.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key, [])
        if entries:
            cv = entries[0].get("cvssData", {})
            return {
                "version": cv.get("version", key[-3:]),
                "score": cv.get("baseScore"),
                "severity": cv.get("baseSeverity"),
                "vector": cv.get("vectorString"),
                "attack_vector": cv.get("attackVector"),
                "privileges_required": cv.get("privilegesRequired"),
                "user_interaction": cv.get("userInteraction"),
                "scope": cv.get("scope"),
                "confidentiality_impact": cv.get("confidentialityImpact"),
                "integrity_impact": cv.get("integrityImpact"),
                "availability_impact": cv.get("availabilityImpact"),
            }
    entries = metrics.get("cvssMetricV2", [])
    if entries:
        cv = entries[0].get("cvssData", {})
        return {
            "version": "2.0",
            "score": cv.get("baseScore"),
            "severity": entries[0].get("baseSeverity"),
            "vector": cv.get("vectorString"),
            "attack_vector": cv.get("accessVector"),
            "privileges_required": None,
            "user_interaction": None,
            "scope": None,
            "confidentiality_impact": cv.get("confidentialityImpact"),
            "integrity_impact": cv.get("integrityImpact"),
            "availability_impact": cv.get("availabilityImpact"),
        }
    return _empty_cvss()


def _empty_cvss() -> dict[str, Any]:
    return {
        "version": None,
        "score": None,
        "severity": None,
        "vector": None,
        "attack_vector": None,
        "privileges_required": None,
        "user_interaction": None,
        "scope": None,
        "confidentiality_impact": None,
        "integrity_impact": None,
        "availability_impact": None,
    }


def _extract_cwe(nvd: dict[str, Any]) -> list[str]:
    cwes: list[str] = []
    for weakness in nvd.get("weaknesses", []):
        for desc in weakness.get("description", []):
            val = desc.get("value", "")
            if val and val not in ("NVD-CWE-Other", "NVD-CWE-noinfo"):
                cwes.append(val)
    return cwes


def _extract_affected(nvd: dict[str, Any]) -> str:
    for config in nvd.get("configurations", []):
        for node in config.get("nodes", []):
            for cpe_match in node.get("cpeMatch", []):
                uri = cpe_match.get("criteria", "")
                parts = uri.split(":")
                if len(parts) >= 5:
                    vendor = parts[3].replace("_", " ").title()
                    product = parts[4].replace("_", " ").title()
                    return f"{vendor} / {product}"
    return "Unknown"


def _extract_published(nvd: dict[str, Any]) -> str:
    raw = nvd.get("published", "")
    return raw[:10] if raw else "Unknown"


def _extract_description(nvd: dict[str, Any], max_chars: int = 400) -> str:
    for desc in nvd.get("descriptions", []):
        if desc.get("lang") == "en":
            val = desc.get("value", "")
            return val[:max_chars] + ("..." if len(val) > max_chars else "")
    return ""


def _extract_references(nvd: dict[str, Any], limit: int = 5) -> list[str]:
    refs: list[str] = []
    for r in nvd.get("references", []):
        url = r.get("url", "")
        if url:
            refs.append(url)
        if len(refs) >= limit:
            break
    return refs


# ──────────────────────────────────────────────────────────────────────────────
# Profile builder
# ──────────────────────────────────────────────────────────────────────────────


def _build_profile(cve_id: str) -> dict[str, Any]:
    """Fetch NVD + EPSS data for one CVE in parallel, return a normalised profile."""
    cid = cve_id.upper()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        nvd_f = ex.submit(_fetch_nvd, cid)
        epss_f = ex.submit(_fetch_epss, cid)
        nvd = nvd_f.result()
        epss = epss_f.result()

    has_nvd = "error" not in nvd
    return {
        "cve_id": cid,
        "nvd_error": nvd.get("error"),
        "epss_error": epss.get("error"),
        "cvss": _extract_cvss(nvd) if has_nvd else _empty_cvss(),
        "epss": {
            "score": float(epss["epss"]) if not epss.get("error") and epss.get("epss") else None,
            "percentile": float(epss["percentile"]) if not epss.get("error") and epss.get("percentile") else None,
        },
        "cwe": _extract_cwe(nvd) if has_nvd else [],
        "affected": _extract_affected(nvd) if has_nvd else "Unknown",
        "published": _extract_published(nvd) if has_nvd else "Unknown",
        "description": _extract_description(nvd) if has_nvd else "",
        "references": _extract_references(nvd) if has_nvd else [],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Scoring and prioritisation
# ──────────────────────────────────────────────────────────────────────────────


def _priority_score(profile: dict[str, Any]) -> tuple[float, list[str]]:
    """
    Composite priority score (higher = more urgent) with annotated reasons.

    Rubric:
      +10  in CISA KEV
      +8   CVSS >= 9.0 (Critical)
      +5   CVSS 7.0-8.9 (High)
      +2   CVSS 4.0-6.9 (Medium)
      +8   EPSS >= 0.70
      +5   EPSS 0.40-0.69
      +2   EPSS 0.10-0.39
      +3   Network attack vector
      +2   Privileges required = NONE
      +1   User interaction = NONE
    """
    score = 0.0
    reasons: list[str] = []

    if profile.get("kev", {}).get("in_kev"):
        score += 10
        reasons.append("in CISA KEV (actively exploited)")

    cvss_score = profile.get("cvss", {}).get("score")
    if cvss_score is not None:
        cs = float(cvss_score)
        if cs >= 9.0:
            score += 8
            reasons.append(f"CVSS {cs:.1f} -- Critical severity")
        elif cs >= 7.0:
            score += 5
            reasons.append(f"CVSS {cs:.1f} -- High severity")
        elif cs >= 4.0:
            score += 2
            reasons.append(f"CVSS {cs:.1f} -- Medium severity")

    epss = profile.get("epss", {}).get("score")
    if epss is not None:
        ev = float(epss)
        if ev >= 0.70:
            score += 8
            reasons.append(f"EPSS {ev:.3f} -- very high exploitation probability")
        elif ev >= 0.40:
            score += 5
            reasons.append(f"EPSS {ev:.3f} -- high exploitation probability")
        elif ev >= 0.10:
            score += 2
            reasons.append(f"EPSS {ev:.3f} -- elevated exploitation probability")

    av = (profile.get("cvss", {}).get("attack_vector") or "").upper()
    if av == "NETWORK":
        score += 3
        reasons.append("remotely exploitable (network attack vector)")

    pr = (profile.get("cvss", {}).get("privileges_required") or "").upper()
    if pr == "NONE":
        score += 2
        reasons.append("no privileges required")

    ui = (profile.get("cvss", {}).get("user_interaction") or "").upper()
    if ui == "NONE":
        score += 1
        reasons.append("no user interaction required")

    return score, reasons


# ──────────────────────────────────────────────────────────────────────────────
# Report formatting helpers
# ──────────────────────────────────────────────────────────────────────────────


def _severity_badge(severity: str | None) -> str:
    """Return an emoji badge for a CVSS severity string."""
    mapping = {
        "CRITICAL": "CRITICAL",
        "HIGH": "HIGH",
        "MEDIUM": "MEDIUM",
        "LOW": "LOW",
        "NONE": "NONE",
    }
    return mapping.get((severity or "").upper(), severity or "UNKNOWN")


def _epss_label(score: float | None) -> str:
    if score is None:
        return "N/A"
    if score >= 0.70:
        return f"{score:.3f} (very high)"
    if score >= 0.40:
        return f"{score:.3f} (high)"
    if score >= 0.10:
        return f"{score:.3f} (elevated)"
    return f"{score:.3f} (low)"


def _delta_arrow(val_a: float | None, val_b: float | None) -> str:
    """Return a delta string showing the numeric difference between two values."""
    if val_a is None or val_b is None:
        return "N/A"
    diff = val_a - val_b
    if abs(diff) < 0.0005:
        return "equal"
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.3f} (A-B)"


def _na(val: Any) -> str:
    return str(val) if val is not None else "N/A"


# ──────────────────────────────────────────────────────────────────────────────
# Report builder
# ──────────────────────────────────────────────────────────────────────────────


def _build_diff_report(
    profile_a: dict[str, Any],
    profile_b: dict[str, Any],
    kev_a: dict[str, Any],
    kev_b: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the full structured diff-report dict."""
    profile_a = dict(profile_a, kev=kev_a)
    profile_b = dict(profile_b, kev=kev_b)

    score_a, reasons_a = _priority_score(profile_a)
    score_b, reasons_b = _priority_score(profile_b)

    if score_a > score_b:
        winner = profile_a["cve_id"]
        loser = profile_b["cve_id"]
        winner_reasons = reasons_a
        confidence_margin = score_a - score_b
    elif score_b > score_a:
        winner = profile_b["cve_id"]
        loser = profile_a["cve_id"]
        winner_reasons = reasons_b
        confidence_margin = score_b - score_a
    else:
        winner = "tie"
        loser = "tie"
        winner_reasons = ["scores are equal"]
        confidence_margin = 0.0

    if confidence_margin >= 10:
        confidence = "strong"
    elif confidence_margin >= 5:
        confidence = "moderate"
    elif confidence_margin > 0:
        confidence = "weak"
    else:
        confidence = "tie"

    cvss_a = profile_a.get("cvss", {})
    cvss_b = profile_b.get("cvss", {})
    epss_a = profile_a.get("epss", {}).get("score")
    epss_b = profile_b.get("epss", {}).get("score")

    sev_rank_a = _SEVERITY_ORDER.get((cvss_a.get("severity") or "").upper(), 0)
    sev_rank_b = _SEVERITY_ORDER.get((cvss_b.get("severity") or "").upper(), 0)
    if sev_rank_a > sev_rank_b:
        higher_severity = profile_a["cve_id"]
    elif sev_rank_b > sev_rank_a:
        higher_severity = profile_b["cve_id"]
    else:
        higher_severity = "equal"

    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "cve_a": profile_a,
        "cve_b": profile_b,
        "priority_score_a": score_a,
        "priority_score_b": score_b,
        "priority_reasons_a": reasons_a,
        "priority_reasons_b": reasons_b,
        "higher_priority": winner,
        "lower_priority": loser,
        "confidence": confidence,
        "confidence_margin": confidence_margin,
        "winner_reasons": winner_reasons,
        "cvss_delta": _delta_arrow(
            float(cvss_a["score"]) if cvss_a.get("score") is not None else None,
            float(cvss_b["score"]) if cvss_b.get("score") is not None else None,
        ),
        "epss_delta": _delta_arrow(epss_a, epss_b),
        "severity_comparison": {
            "a": cvss_a.get("severity"),
            "b": cvss_b.get("severity"),
            "higher": higher_severity,
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    """Render the diff-report dict as a rich Markdown document."""
    pa = report["cve_a"]
    pb = report["cve_b"]
    cvss_a = pa.get("cvss", {})
    cvss_b = pb.get("cvss", {})
    epss_a = pa.get("epss", {})
    epss_b = pb.get("epss", {})
    kev_a = pa.get("kev", {})
    kev_b = pb.get("kev", {})

    id_a = pa["cve_id"]
    id_b = pb["cve_id"]
    winner = report["higher_priority"]
    loser = report["lower_priority"]
    confidence = report["confidence"]
    margin = report["confidence_margin"]
    score_a = report["priority_score_a"]
    score_b = report["priority_score_b"]

    lines: list[str] = []

    # Header
    lines.append(f"# CVE Diff Report: {id_a} vs {id_b}")
    lines.append("")
    lines.append(f"*Generated: {report['generated_at']}*")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    if winner == "tie":
        lines.append(
            f"**Verdict:** Both CVEs score equally ({score_a:.0f} points each). "
            "Manual review recommended -- consider operational context, patch availability, "
            "and asset exposure."
        )
    else:
        winner_pts = score_a if winner == id_a else score_b
        loser_pts = score_b if winner == id_a else score_a
        lines.append(
            f"**Verdict:** Prioritise **{winner}** over {loser} -- "
            f"{confidence} recommendation (score {winner_pts:.0f} vs {loser_pts:.0f}, "
            f"margin +{margin:.0f})."
        )
    lines.append("")
    if report["winner_reasons"] and winner != "tie":
        lines.append(f"**Key factors for {winner}:**")
        for reason in report["winner_reasons"][:4]:
            lines.append(f"- {reason}")
    lines.append("")

    # Side-by-Side Comparison Table
    lines.append("## Side-by-Side Comparison")
    lines.append("")
    lines.append(f"| Dimension | {id_a} | {id_b} | Delta / Notes |")
    lines.append("|-----------|--------|--------|---------------|")

    def row(dim: str, val_a: str, val_b: str, note: str = "") -> str:
        return f"| {dim} | {val_a} | {val_b} | {note} |"

    lines.append(row("Published", pa["published"], pb["published"]))
    lines.append(row("Affected", pa["affected"], pb["affected"]))
    lines.append(
        row(
            "CVSS Score",
            f"{_na(cvss_a.get('score'))} ({cvss_a.get('version') or '?'})",
            f"{_na(cvss_b.get('score'))} ({cvss_b.get('version') or '?'})",
            report["cvss_delta"],
        )
    )
    lines.append(
        row(
            "Severity",
            _severity_badge(cvss_a.get("severity")),
            _severity_badge(cvss_b.get("severity")),
            f"Higher: {report['severity_comparison']['higher']}",
        )
    )
    lines.append(
        row(
            "EPSS Score",
            _epss_label(epss_a.get("score")),
            _epss_label(epss_b.get("score")),
            report["epss_delta"],
        )
    )
    lines.append(
        row(
            "EPSS Percentile",
            f"{epss_a['percentile']:.1%}" if epss_a.get("percentile") is not None else "N/A",
            f"{epss_b['percentile']:.1%}" if epss_b.get("percentile") is not None else "N/A",
        )
    )
    lines.append(
        row(
            "CISA KEV",
            "YES" if kev_a.get("in_kev") else "No",
            "YES" if kev_b.get("in_kev") else "No",
        )
    )
    lines.append(
        row(
            "Attack Vector",
            _na(cvss_a.get("attack_vector")),
            _na(cvss_b.get("attack_vector")),
        )
    )
    lines.append(
        row(
            "Privileges Req.",
            _na(cvss_a.get("privileges_required")),
            _na(cvss_b.get("privileges_required")),
        )
    )
    lines.append(
        row(
            "User Interaction",
            _na(cvss_a.get("user_interaction")),
            _na(cvss_b.get("user_interaction")),
        )
    )
    lines.append(
        row(
            "CVSS Vector",
            cvss_a.get("vector") or "N/A",
            cvss_b.get("vector") or "N/A",
        )
    )
    lines.append(
        row(
            "CWE",
            ", ".join(pa.get("cwe", [])) or "N/A",
            ", ".join(pb.get("cwe", [])) or "N/A",
        )
    )
    lines.append(
        row(
            "Priority Score",
            f"**{score_a:.0f}**",
            f"**{score_b:.0f}**",
            f"Delta {score_a - score_b:+.0f}",
        )
    )
    lines.append("")

    # Vulnerability Summaries
    lines.append("## Vulnerability Summaries")
    lines.append("")
    for cve_id, profile, cvss, epss, kev in [
        (id_a, pa, cvss_a, epss_a, kev_a),
        (id_b, pb, cvss_b, epss_b, kev_b),
    ]:
        lines.append(f"### {cve_id}")
        lines.append("")
        if profile.get("nvd_error"):
            lines.append(f"> Source unavailable: {profile['nvd_error']}")
        elif profile.get("description"):
            lines.append(f"> {profile['description']}")
        lines.append("")
        lines.append(f"- **Published:** {profile['published']}")
        lines.append(f"- **Affected:** {profile['affected']}")
        score_str = (
            f"{cvss.get('score')} ({cvss.get('severity')}) -- `{cvss.get('vector') or 'N/A'}`"
            if cvss.get("score")
            else "N/A"
        )
        lines.append(f"- **CVSS:** {score_str}")
        lines.append(f"- **CWE:** {', '.join(profile.get('cwe', [])) or 'N/A'}")
        epss_str = _epss_label(epss.get("score"))
        pct = epss.get("percentile")
        pct_str = f" (top {(1 - pct):.1%} of all CVEs)" if pct is not None else ""
        if profile.get("epss_error"):
            lines.append("- **EPSS:** Source unavailable")
        else:
            lines.append(f"- **EPSS:** {epss_str}{pct_str}")
        if kev.get("in_kev"):
            lines.append(f"- **CISA KEV:** YES -- Added {kev.get('date_added', '?')}, due {kev.get('due_date', '?')}")
            if kev.get("required_action"):
                lines.append(f"  - Required action: {kev['required_action']}")
        else:
            lines.append("- **CISA KEV:** Not in catalog")
        lines.append("")

    # CVSS Delta Analysis
    lines.append("## CVSS Delta Analysis")
    lines.append("")
    cvss_score_a = cvss_a.get("score")
    cvss_score_b = cvss_b.get("score")
    if cvss_score_a is not None and cvss_score_b is not None:
        diff = float(cvss_score_a) - float(cvss_score_b)
        lines.append(f"CVSS score difference: **{diff:+.1f}** ({id_a}: {cvss_score_a} vs {id_b}: {cvss_score_b}).")
        lines.append("")
        lines.append("**Impact dimensions:**")
        lines.append("")
        lines.append(f"| Impact | {id_a} | {id_b} |")
        lines.append("|--------|--------|--------|")
        for dim in ("confidentiality_impact", "integrity_impact", "availability_impact"):
            label = dim.replace("_impact", "").capitalize()
            lines.append(f"| {label} | {_na(cvss_a.get(dim))} | {_na(cvss_b.get(dim))} |")
        lines.append("")
        lines.append("**Exploitability dimensions:**")
        lines.append("")
        lines.append(f"| Dimension | {id_a} | {id_b} |")
        lines.append("|-----------|--------|--------|")
        for dim, label in [
            ("attack_vector", "Attack Vector"),
            ("privileges_required", "Privileges Required"),
            ("user_interaction", "User Interaction"),
            ("scope", "Scope"),
        ]:
            lines.append(f"| {label} | {_na(cvss_a.get(dim))} | {_na(cvss_b.get(dim))} |")
    else:
        lines.append("CVSS data unavailable for one or both CVEs -- delta analysis skipped.")
    lines.append("")

    # EPSS Probability Divergence
    lines.append("## EPSS Exploitation Probability Divergence")
    lines.append("")
    epss_score_a = epss_a.get("score")
    epss_score_b = epss_b.get("score")
    if epss_score_a is not None and epss_score_b is not None:
        epss_diff = float(epss_score_a) - float(epss_score_b)
        lines.append(
            f"EPSS score difference: **{epss_diff:+.4f}** ({id_a}: {epss_score_a:.4f} vs {id_b}: {epss_score_b:.4f})."
        )
        lines.append("")
        if abs(epss_diff) < 0.01:
            lines.append("EPSS scores are nearly identical -- exploitation probability is comparable for both CVEs.")
        elif abs(epss_diff) < 0.10:
            higher_id = id_a if epss_diff > 0 else id_b
            lines.append(
                f"{higher_id} has a modestly higher exploitation probability. "
                "The difference is small enough that operational context (e.g., internet exposure, "
                "asset criticality) should drive final prioritisation."
            )
        else:
            higher_id = id_a if epss_diff > 0 else id_b
            lower_id = id_b if epss_diff > 0 else id_a
            higher_score = epss_score_a if epss_diff > 0 else epss_score_b
            lower_score = epss_score_b if epss_diff > 0 else epss_score_a
            lines.append(
                f"{higher_id} (EPSS {higher_score:.4f}) has a significantly higher exploitation "
                f"probability than {lower_id} (EPSS {lower_score:.4f}). "
                "This gap strongly differentiates the two CVEs when CVSS scores are similar."
            )
    else:
        lines.append("EPSS data unavailable for one or both CVEs -- divergence analysis skipped.")
    lines.append("")

    # KEV Exploitation Status
    lines.append("## CISA KEV Exploitation Status")
    lines.append("")
    both_kev = kev_a.get("in_kev") and kev_b.get("in_kev")
    either_kev = kev_a.get("in_kev") or kev_b.get("in_kev")
    if both_kev:
        lines.append(
            f"Both {id_a} and {id_b} are in CISA KEV -- both represent confirmed exploitation "
            "in the wild. Treat both as P0 and escalate immediately. Use CVSS, EPSS, and asset "
            "exposure to determine patching order."
        )
    elif either_kev:
        kev_id = id_a if kev_a.get("in_kev") else id_b
        no_kev_id = id_b if kev_a.get("in_kev") else id_a
        kev_entry = kev_a if kev_a.get("in_kev") else kev_b
        lines.append(
            f"{kev_id} is in CISA KEV -- confirmed exploitation in the wild. "
            f"{no_kev_id} is not in KEV. "
            f"Binding Operational Directive 22-01 requires federal agencies to remediate {kev_id} "
            f"by **{kev_entry.get('due_date', 'the stated due date')}**."
        )
        if kev_entry.get("required_action"):
            lines.append("")
            lines.append(f"**Required action for {kev_id}:** {kev_entry['required_action']}")
    else:
        lines.append(
            "Neither CVE is currently in CISA KEV. "
            "This does not rule out exploitation in the wild -- consult EPSS scores and threat feeds."
        )
    lines.append("")

    # CWE Class Comparison
    lines.append("## CWE Weakness Class Comparison")
    lines.append("")
    cwe_a = pa.get("cwe", [])
    cwe_b = pb.get("cwe", [])
    shared = set(cwe_a) & set(cwe_b)
    only_a = set(cwe_a) - set(cwe_b)
    only_b = set(cwe_b) - set(cwe_a)

    lines.append(f"| | {id_a} | {id_b} |")
    lines.append("|---|------|------|")
    lines.append(f"| CWEs | {', '.join(cwe_a) or 'N/A'} | {', '.join(cwe_b) or 'N/A'} |")
    lines.append("")
    if shared:
        lines.append(f"**Shared weakness classes:** {', '.join(sorted(shared))}")
        lines.append("")
        lines.append(
            "These CVEs share the same root weakness class(es), suggesting they may be related "
            "variants or that the same code pattern is exploited in both cases."
        )
    elif cwe_a and cwe_b:
        only_a_str = ", ".join(sorted(only_a)) if only_a else "none"
        only_b_str = ", ".join(sorted(only_b)) if only_b else "none"
        lines.append(
            f"**Distinct weakness classes** -- {id_a} ({only_a_str}) and {id_b} ({only_b_str}) "
            "affect different vulnerability categories. No pattern overlap detected."
        )
    else:
        lines.append("CWE data unavailable for one or both CVEs.")
    lines.append("")

    # Prioritisation Rationale
    lines.append("## Prioritisation Rationale")
    lines.append("")
    if winner == "tie":
        lines.append("### Tie -- Manual Review Required")
        lines.append("")
        lines.append(f"Both {id_a} and {id_b} score identically ({score_a:.0f} points). Use the following tiebreakers:")
        lines.append("")
        lines.append("1. **Asset exposure** -- which affected component is directly internet-accessible?")
        lines.append("2. **Patch availability** -- which has a confirmed fix or workaround?")
        lines.append("3. **Blast radius** -- how many downstream dependents are affected?")
        lines.append("4. **EPSS trajectory** -- is either score trending upward?")
    else:
        confidence_label = {
            "strong": "Strong Recommendation",
            "moderate": "Moderate Recommendation",
            "weak": "Weak Recommendation",
        }.get(confidence, "Recommendation")
        lines.append(f"### {confidence_label}: Prioritise {winner}")
        lines.append("")
        winner_pts = score_a if winner == id_a else score_b
        loser_pts = score_b if winner == id_a else score_a
        lines.append(
            f"**{winner}** scores {winner_pts:.0f} vs {loser_pts:.0f} for {loser} (margin: +{margin:.0f} points)."
        )
        lines.append("")
        lines.append(f"**Scoring factors for {winner}:**")
        for r in report["winner_reasons"]:
            lines.append(f"- {r}")
        lines.append("")
        loser_reasons = report["priority_reasons_b"] if winner == id_a else report["priority_reasons_a"]
        lines.append(f"**Scoring factors for {loser}:**")
        if loser_reasons:
            for r in loser_reasons:
                lines.append(f"- {r}")
        else:
            lines.append("- No significant scoring factors detected")
        lines.append("")
        if confidence == "weak":
            lines.append(
                "> **Weak confidence:** The score margin is small. Verify EPSS trends, "
                "active exploitation reports, and asset-specific context before committing resources."
            )
    lines.append("")

    # References
    lines.append("## References")
    lines.append("")
    for cve_id, profile in [(id_a, pa), (id_b, pb)]:
        lines.append(f"**{cve_id}**")
        lines.append(f"- NVD: https://nvd.nist.gov/vuln/detail/{cve_id}")
        for ref in profile.get("references", []):
            lines.append(f"- {ref}")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*This report was generated by manus-agent. Data sourced from NVD, FIRST EPSS, and CISA KEV. "
        "Scores reflect point-in-time data and may change as exploitation evidence evolves.*"
    )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Strands tool entry point
# ──────────────────────────────────────────────────────────────────────────────


def diff_report_cves(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Generate a rich Markdown diff-report comparing two CVEs."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    cve_id_a = str(tool_input.get("cve_id_a", "")).strip()
    cve_id_b = str(tool_input.get("cve_id_b", "")).strip()
    output_format = str(tool_input.get("output_format", "markdown")).lower()

    for cid in (cve_id_a, cve_id_b):
        if not _CVE_RE.match(cid):
            result: ToolResult = {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [
                    {"text": (f"Invalid CVE ID '{cid}'. Expected format: CVE-YYYY-NNNN (e.g. CVE-2021-44228).")}
                ],
            }
            log_tool_output_size("diff_report_cves", result)
            return result

    # Fetch all four data sources concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        f_a = ex.submit(_build_profile, cve_id_a)
        f_b = ex.submit(_build_profile, cve_id_b)
        f_kev_a = ex.submit(_fetch_kev, cve_id_a)
        f_kev_b = ex.submit(_fetch_kev, cve_id_b)
        profile_a = f_a.result()
        profile_b = f_b.result()
        kev_a = f_kev_a.result()
        kev_b = f_kev_b.result()

    report = _build_diff_report(profile_a, profile_b, kev_a, kev_b)

    if output_format == "json":
        result = {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"json": report}],
        }
    else:
        md = _render_markdown(report)
        result = {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [
                {"text": md},
                {"json": report},
            ],
        }

    log_tool_output_size("diff_report_cves", result)
    return result
