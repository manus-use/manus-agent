"""
Tool for comparing two CVEs side-by-side across multiple vulnerability intelligence dimensions.

Fetches NVD, EPSS, and CISA KEV data for both CVEs in parallel and produces a structured
comparison report useful for prioritisation decisions.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size
from manus_agent.utils.cve_utils import cve_validation_error

TOOL_SPEC = {
    "name": "compare_cves",
    "description": (
        "Compares two CVEs side-by-side across multiple vulnerability intelligence dimensions: "
        "CVSS base score and severity, EPSS exploitation-probability score, CISA KEV membership, "
        "CWE weakness class, affected vendor/product, and published date. "
        "Returns a structured comparison with a prioritisation recommendation — "
        "which CVE poses the greater immediate risk and why. "
        "Use this when you need to triage two vulnerabilities against each other."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id_a": {
                    "type": "string",
                    "description": "First CVE identifier to compare (e.g., 'CVE-2024-3094').",
                },
                "cve_id_b": {
                    "type": "string",
                    "description": "Second CVE identifier to compare (e.g., 'CVE-2021-44228').",
                },
            },
            "required": ["cve_id_a", "cve_id_b"],
        }
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Data-fetch helpers
# ──────────────────────────────────────────────────────────────────────────────


def _fetch_nvd(cve_id: str) -> dict[str, Any]:
    """Return the CVE sub-record from NVD, or an error dict."""
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id.upper()}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return {"error": f"No NVD record found for {cve_id}"}
        return vulns[0].get("cve", {})
    except requests.RequestException as exc:
        return {"error": f"NVD request failed: {exc}"}


def _fetch_epss(cve_id: str) -> dict[str, Any]:
    """Return the current EPSS score dict for a CVE, or an error dict."""
    url = "https://api.first.org/data/v1/epss"
    try:
        resp = requests.get(url, params={"cve": cve_id.upper()}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        entries = data.get("data", [])
        if not entries:
            return {"error": f"No EPSS data for {cve_id}"}
        return entries[0]
    except requests.RequestException as exc:
        return {"error": f"EPSS request failed: {exc}"}


def _fetch_kev(cve_id: str) -> dict[str, Any]:
    """Return the KEV catalog entry for a CVE, or {"in_kev": False} if absent."""
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


def _extract_cvss(nvd_record: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the highest-version available CVSS score from an NVD CVE record.
    Prefers CVSSv3.1 → CVSSv3.0 → CVSSv2.
    """
    metrics = nvd_record.get("metrics", {})

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
        }

    return {
        "version": None,
        "score": None,
        "severity": None,
        "vector": None,
        "attack_vector": None,
        "privileges_required": None,
        "user_interaction": None,
    }


def _extract_cwe(nvd_record: dict[str, Any]) -> list[str]:
    """Return a list of CWE IDs from the NVD record."""
    cwes: list[str] = []
    for weakness in nvd_record.get("weaknesses", []):
        for desc in weakness.get("description", []):
            val = desc.get("value", "")
            if val and val not in ("NVD-CWE-Other", "NVD-CWE-noinfo"):
                cwes.append(val)
    return cwes


def _extract_affected(nvd_record: dict[str, Any]) -> str:
    """Return a short 'Vendor / Product' string from the NVD record."""
    for config in nvd_record.get("configurations", []):
        for node in config.get("nodes", []):
            for cpe_match in node.get("cpeMatch", []):
                uri = cpe_match.get("criteria", "")
                parts = uri.split(":")
                if len(parts) >= 5:
                    vendor = parts[3].replace("_", " ").title()
                    product = parts[4].replace("_", " ").title()
                    return f"{vendor} / {product}"
    return "Unknown"


def _extract_published(nvd_record: dict[str, Any]) -> str:
    """Return the published date (YYYY-MM-DD) from the NVD record."""
    raw = nvd_record.get("published", "")
    return raw[:10] if raw else "Unknown"


def _extract_description(nvd_record: dict[str, Any]) -> str:
    """Return the English description from the NVD record, truncated to 200 chars."""
    for desc in nvd_record.get("descriptions", []):
        if desc.get("lang") == "en":
            val = desc.get("value", "")
            return val[:200] + ("…" if len(val) > 200 else "")
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Comparison and prioritisation logic
# ──────────────────────────────────────────────────────────────────────────────


def _score_cve(profile: dict[str, Any]) -> tuple[float, list[str]]:
    """
    Assign a composite priority score (higher = more urgent) and list the reasons.

    Scoring rubric:
      +10   in CISA KEV (actively exploited, confirmed)
      +8    CVSS score ≥ 9.0 (Critical)
      +5    CVSS score 7.0–8.9 (High)
      +2    CVSS score 4.0–6.9 (Medium)
      +8    EPSS ≥ 0.70 (very likely to be exploited)
      +5    EPSS 0.40–0.69
      +2    EPSS 0.10–0.39
      +3    Attack vector = NETWORK
      +2    Privileges required = NONE
      +1    User interaction = NONE
    """
    score = 0.0
    reasons: list[str] = []

    if profile.get("kev", {}).get("in_kev"):
        score += 10
        reasons.append("in CISA KEV (actively exploited in the wild)")

    cvss_score = profile.get("cvss", {}).get("score")
    if cvss_score is not None:
        cs = float(cvss_score)
        if cs >= 9.0:
            score += 8
            reasons.append(f"CVSS {cs:.1f} (Critical)")
        elif cs >= 7.0:
            score += 5
            reasons.append(f"CVSS {cs:.1f} (High)")
        elif cs >= 4.0:
            score += 2
            reasons.append(f"CVSS {cs:.1f} (Medium)")

    epss_val = profile.get("epss", {}).get("epss")
    if epss_val is not None:
        ev = float(epss_val)
        if ev >= 0.70:
            score += 8
            reasons.append(f"EPSS {ev:.3f} (very high exploitation probability)")
        elif ev >= 0.40:
            score += 5
            reasons.append(f"EPSS {ev:.3f} (high exploitation probability)")
        elif ev >= 0.10:
            score += 2
            reasons.append(f"EPSS {ev:.3f} (elevated exploitation probability)")

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


def _build_cve_profile(cve_id: str) -> dict[str, Any]:
    """Fetch NVD and EPSS data for a CVE and return a normalised profile dict."""
    cid = cve_id.upper()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        nvd_future = executor.submit(_fetch_nvd, cid)
        epss_future = executor.submit(_fetch_epss, cid)
        nvd_data = nvd_future.result()
        epss_data = epss_future.result()

    has_nvd_error = "error" in nvd_data
    cvss = (
        _extract_cvss(nvd_data)
        if not has_nvd_error
        else {
            "version": None,
            "score": None,
            "severity": None,
            "vector": None,
            "attack_vector": None,
            "privileges_required": None,
            "user_interaction": None,
        }
    )
    cwe = _extract_cwe(nvd_data) if not has_nvd_error else []
    affected = _extract_affected(nvd_data) if not has_nvd_error else "Unknown"
    published = _extract_published(nvd_data) if not has_nvd_error else "Unknown"
    description = _extract_description(nvd_data) if not has_nvd_error else ""

    return {
        "cve_id": cid,
        "nvd_error": nvd_data.get("error"),
        "epss_error": epss_data.get("error"),
        "cvss": cvss,
        "epss": {
            "epss": epss_data.get("epss"),
            "percentile": epss_data.get("percentile"),
        }
        if not epss_data.get("error")
        else {},
        "cwe": cwe,
        "affected": affected,
        "published": published,
        "description": description,
    }


def _build_comparison(
    profile_a: dict[str, Any],
    kev_a: dict[str, Any],
    profile_b: dict[str, Any],
    kev_b: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the final comparison dict with a prioritisation recommendation."""
    profile_a["kev"] = kev_a
    profile_b["kev"] = kev_b

    score_a, reasons_a = _score_cve(profile_a)
    score_b, reasons_b = _score_cve(profile_b)

    if score_a > score_b:
        winner = profile_a["cve_id"]
        winner_score = score_a
        loser_score = score_b
        winner_reasons = reasons_a
    elif score_b > score_a:
        winner = profile_b["cve_id"]
        winner_score = score_b
        loser_score = score_a
        winner_reasons = reasons_b
    else:
        winner = "tie"
        winner_score = score_a
        loser_score = score_b
        winner_reasons = ["scores are equal — review manually"]

    if winner != "tie":
        margin = winner_score - loser_score
        if margin >= 10:
            confidence = "strong"
        elif margin >= 5:
            confidence = "moderate"
        else:
            confidence = "weak"
        recommendation = (
            f"Prioritise {winner} ({confidence} recommendation, "
            f"composite score {winner_score:.0f} vs {loser_score:.0f}). "
            f"Key factors: {'; '.join(winner_reasons[:3])}."
        )
    else:
        confidence = "tie"
        recommendation = (
            "Both CVEs score equally. Review manually using CVSS vector, EPSS trajectory, and operational context."
        )

    return {
        "cve_a": profile_a,
        "cve_b": profile_b,
        "priority_score_a": score_a,
        "priority_score_b": score_b,
        "higher_priority": winner,
        "confidence": confidence,
        "recommendation": recommendation,
        "winner_reasons": winner_reasons,
    }


def _render_text(comparison: dict[str, Any]) -> str:
    """Render the comparison as a human-readable aligned table."""

    def _fmt_epss(e: dict[str, Any]) -> str:
        epss = e.get("epss")
        pct = e.get("percentile")
        if epss is None:
            return "N/A"
        pct_str = f"  ({float(pct):.1%} pct)" if pct is not None else ""
        return f"{float(epss):.4f}{pct_str}"

    def _fmt_cvss(c: dict[str, Any]) -> str:
        score = c.get("score")
        sev = c.get("severity") or ""
        ver = c.get("version") or ""
        if score is None:
            return "N/A"
        ver_str = f"  (v{ver})" if ver else ""
        return f"{score}  {sev}{ver_str}"

    def _fmt_kev(k: dict[str, Any]) -> str:
        if k.get("in_kev"):
            return f"YES — added {k.get('date_added', '?')} (due {k.get('due_date', '?')})"
        return "No"

    pa = comparison["cve_a"]
    pb = comparison["cve_b"]
    col_w = 40

    def row(label: str, val_a: str, val_b: str) -> str:
        return f"  {label:<24}  {val_a:<{col_w}}  {val_b}\n"

    lines: list[str] = []
    sep = "─" * 100
    lines.append(f"\n{sep}\n")
    lines.append(f"  {'DIMENSION':<24}  {pa['cve_id']:<{col_w}}  {pb['cve_id']}\n")
    lines.append(f"{sep}\n")
    lines.append(row("Published", pa["published"], pb["published"]))
    lines.append(row("Affected", pa["affected"][:col_w], pb["affected"]))
    lines.append(row("CVSS", _fmt_cvss(pa.get("cvss", {})), _fmt_cvss(pb.get("cvss", {}))))
    lines.append(row("EPSS score", _fmt_epss(pa.get("epss", {})), _fmt_epss(pb.get("epss", {}))))
    lines.append(row("CISA KEV", _fmt_kev(pa.get("kev", {})), _fmt_kev(pb.get("kev", {}))))
    lines.append(row("CWE", ", ".join(pa.get("cwe", [])) or "N/A", ", ".join(pb.get("cwe", [])) or "N/A"))
    lines.append(
        row(
            "Attack vector",
            pa.get("cvss", {}).get("attack_vector") or "N/A",
            pb.get("cvss", {}).get("attack_vector") or "N/A",
        )
    )
    lines.append(
        row(
            "Privileges req.",
            pa.get("cvss", {}).get("privileges_required") or "N/A",
            pb.get("cvss", {}).get("privileges_required") or "N/A",
        )
    )
    lines.append(
        row(
            "User interaction",
            pa.get("cvss", {}).get("user_interaction") or "N/A",
            pb.get("cvss", {}).get("user_interaction") or "N/A",
        )
    )
    lines.append(
        row("Priority score", str(int(comparison["priority_score_a"])), str(int(comparison["priority_score_b"])))
    )
    lines.append(f"{sep}\n")
    lines.append(f"\n  RECOMMENDATION\n  {comparison['recommendation']}\n")

    if pa.get("description"):
        lines.append(f"\n  {pa['cve_id']} — {pa['description']}\n")
    if pb.get("description"):
        lines.append(f"  {pb['cve_id']} — {pb['description']}\n")

    return "".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Strands tool entry point
# ──────────────────────────────────────────────────────────────────────────────


def compare_cves(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Fetch data for two CVEs in parallel and return a side-by-side comparison."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    cve_id_a = str(tool_input.get("cve_id_a", "")).strip()
    cve_id_b = str(tool_input.get("cve_id_b", "")).strip()

    for cid in (cve_id_a, cve_id_b):
        if (err := cve_validation_error(cid, tool_use_id)) is not None:
            log_tool_output_size("compare_cves", err)
            return err

    # Fetch both profiles and both KEV entries concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_a = executor.submit(_build_cve_profile, cve_id_a)
        future_b = executor.submit(_build_cve_profile, cve_id_b)
        future_kev_a = executor.submit(_fetch_kev, cve_id_a)
        future_kev_b = executor.submit(_fetch_kev, cve_id_b)

        profile_a = future_a.result()
        profile_b = future_b.result()
        kev_a = future_kev_a.result()
        kev_b = future_kev_b.result()

    comparison = _build_comparison(profile_a, kev_a, profile_b, kev_b)
    text_report = _render_text(comparison)

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [
            {"text": text_report},
            {"json": comparison},
        ],
    }
    log_tool_output_size("compare_cves", result)
    return result
