"""Tool for classifying and categorizing CVE reference URLs from NVD.

Given a CVE ID, fetches the NVD reference list and classifies each URL into
actionable categories:

  patch         — Fix commit, PR, or release containing the fix
  advisory      — Vendor or coordinator security advisory (GHSA, DSA, USN, etc.)
  exploit       — Public exploit or PoC code
  mailing_list  — Security mailing list post (oss-security, fulldisclosure, etc.)
  issue_tracker — Bug tracker issue or pull request discussion
  vendor_notice — Vendor blog, changelog, or release notes
  media         — News article, blog post, or analysis
  other         — Uncategorized reference

Each classified reference includes:
- url: the original reference URL
- category: one of the categories above
- source: extracted domain/org (e.g. "github.com", "debian.org")
- confidence: high | medium | low
- nvd_tags: original NVD-provided tags (Patch, Exploit, Third Party Advisory, etc.)

This tool is useful for triage workflows: quickly identify which references
contain actionable patch information vs. background reading.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.get_nvd_data import _nvd_get_with_retry
from manus_agent.tools.tool_output_logger import log_tool_output_size

__all__ = ["classify_references", "classify_url", "TOOL_SPEC"]

# ---------------------------------------------------------------------------
# Category constants
# ---------------------------------------------------------------------------

CATEGORY_PATCH = "patch"
CATEGORY_ADVISORY = "advisory"
CATEGORY_EXPLOIT = "exploit"
CATEGORY_MAILING_LIST = "mailing_list"
CATEGORY_ISSUE_TRACKER = "issue_tracker"
CATEGORY_VENDOR_NOTICE = "vendor_notice"
CATEGORY_MEDIA = "media"
CATEGORY_OTHER = "other"

# ---------------------------------------------------------------------------
# Classification rules
# ---------------------------------------------------------------------------

# Domain-based heuristics (domain substring → category)
_DOMAIN_RULES: list[tuple[str, str]] = [
    # Exploit sources
    ("exploit-db.com", CATEGORY_EXPLOIT),
    ("packetstormsecurity.com", CATEGORY_EXPLOIT),
    ("vuldb.com", CATEGORY_EXPLOIT),
    # Mailing lists
    ("seclists.org", CATEGORY_MAILING_LIST),
    ("openwall.com/lists", CATEGORY_MAILING_LIST),
    ("marc.info", CATEGORY_MAILING_LIST),
    ("lists.apache.org", CATEGORY_MAILING_LIST),
    ("lists.debian.org", CATEGORY_MAILING_LIST),
    ("lists.fedoraproject.org", CATEGORY_MAILING_LIST),
    # Advisories
    ("security.gentoo.org", CATEGORY_ADVISORY),
    ("usn.ubuntu.com", CATEGORY_ADVISORY),
    ("ubuntu.com/security", CATEGORY_ADVISORY),
    ("access.redhat.com/errata", CATEGORY_ADVISORY),
    ("access.redhat.com/security", CATEGORY_ADVISORY),
    ("security-tracker.debian.org", CATEGORY_ADVISORY),
    ("advisories.mageia.org", CATEGORY_ADVISORY),
    ("cert.org", CATEGORY_ADVISORY),
    ("jvn.jp", CATEGORY_ADVISORY),
    ("cisa.gov", CATEGORY_ADVISORY),
    # Media / analysis
    ("blog.qualys.com", CATEGORY_MEDIA),
    ("thehackernews.com", CATEGORY_MEDIA),
    ("bleepingcomputer.com", CATEGORY_MEDIA),
    ("securityweek.com", CATEGORY_MEDIA),
    ("krebs", CATEGORY_MEDIA),
]

# Path pattern rules (regex on full URL → category)
_PATH_RULES: list[tuple[re.Pattern[str], str]] = [
    # GitHub commits (high confidence patch)
    (re.compile(r"github\.com/[^/]+/[^/]+/commit/[0-9a-f]+", re.I), CATEGORY_PATCH),
    # GitHub PRs
    (re.compile(r"github\.com/[^/]+/[^/]+/pull/\d+", re.I), CATEGORY_PATCH),
    # GitHub releases / tags
    (re.compile(r"github\.com/[^/]+/[^/]+/releases/tag/", re.I), CATEGORY_PATCH),
    # GitLab commits
    (re.compile(r"gitlab\.\w+/[^/]+/[^/]+/-/commit/", re.I), CATEGORY_PATCH),
    # GitHub Security Advisories
    (re.compile(r"github\.com/advisories/GHSA-", re.I), CATEGORY_ADVISORY),
    (re.compile(r"github\.com/[^/]+/[^/]+/security/advisories/GHSA-", re.I), CATEGORY_ADVISORY),
    # GitHub issues
    (re.compile(r"github\.com/[^/]+/[^/]+/issues/\d+", re.I), CATEGORY_ISSUE_TRACKER),
    # Bugzilla
    (re.compile(r"bugzilla\.[^/]+/show_bug\.cgi", re.I), CATEGORY_ISSUE_TRACKER),
    # JIRA
    (re.compile(r"jira\.[^/]+/browse/", re.I), CATEGORY_ISSUE_TRACKER),
    # Debian DSA / DLA
    (re.compile(r"debian\.org/security/\d+/d[ls]a-", re.I), CATEGORY_ADVISORY),
    # Red Hat Bugzilla
    (re.compile(r"bugzilla\.redhat\.com", re.I), CATEGORY_ISSUE_TRACKER),
    # Exploit-DB by path
    (re.compile(r"exploit-db\.com/exploits/\d+", re.I), CATEGORY_EXPLOIT),
    # PacketStorm
    (re.compile(r"packetstormsecurity\.com/files/\d+", re.I), CATEGORY_EXPLOIT),
]

# NVD tag to category mapping
_TAG_MAP: dict[str, str] = {
    "Patch": CATEGORY_PATCH,
    "Exploit": CATEGORY_EXPLOIT,
    "Third Party Advisory": CATEGORY_ADVISORY,
    "Vendor Advisory": CATEGORY_ADVISORY,
    "US Government Resource": CATEGORY_ADVISORY,
    "Mailing List": CATEGORY_MAILING_LIST,
    "Issue Tracking": CATEGORY_ISSUE_TRACKER,
    "Release Notes": CATEGORY_VENDOR_NOTICE,
    "Product": CATEGORY_VENDOR_NOTICE,
    "Press/Media Coverage": CATEGORY_MEDIA,
    "Technical Description": CATEGORY_MEDIA,
    "Tool Signature": CATEGORY_OTHER,
    "VDB Entry": CATEGORY_OTHER,
    "Broken Link": CATEGORY_OTHER,
    "Not Applicable": CATEGORY_OTHER,
    "Permissions Required": CATEGORY_OTHER,
}

# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------


def classify_url(url: str, nvd_tags: list[str] | None = None) -> dict[str, Any]:
    """Classify a single reference URL into a category.

    Args:
        url: The reference URL to classify.
        nvd_tags: Optional NVD-provided tags for this reference.

    Returns:
        A dict with keys: url, category, source, confidence, nvd_tags.
    """
    nvd_tags = nvd_tags or []
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    full_url_lower = url.lower()

    # Extract human-readable source from domain
    source = domain.removeprefix("www.")

    # Step 1: Path-based pattern matching (highest specificity)
    for pattern, category in _PATH_RULES:
        if pattern.search(url):
            return {
                "url": url,
                "category": category,
                "source": source,
                "confidence": "high",
                "nvd_tags": nvd_tags,
            }

    # Step 2: Domain-based matching
    for domain_pattern, category in _DOMAIN_RULES:
        if domain_pattern in full_url_lower:
            return {
                "url": url,
                "category": category,
                "source": source,
                "confidence": "high",
                "nvd_tags": nvd_tags,
            }

    # Step 3: NVD tag-based classification (medium confidence)
    if nvd_tags:
        for tag in nvd_tags:
            if tag in _TAG_MAP:
                return {
                    "url": url,
                    "category": _TAG_MAP[tag],
                    "source": source,
                    "confidence": "medium",
                    "nvd_tags": nvd_tags,
                }

    # Step 4: Fallback heuristics (low confidence)
    # Check for common patch-related path keywords
    path_lower = parsed.path.lower()
    if any(kw in path_lower for kw in ("/commit/", "/releases/", "/tag/", "/changelog")):
        return {
            "url": url,
            "category": CATEGORY_PATCH,
            "source": source,
            "confidence": "low",
            "nvd_tags": nvd_tags,
        }

    if any(kw in path_lower for kw in ("/advisory", "/security", "/vuln", "/cve-")):
        return {
            "url": url,
            "category": CATEGORY_ADVISORY,
            "source": source,
            "confidence": "low",
            "nvd_tags": nvd_tags,
        }

    if any(kw in path_lower for kw in ("/issues/", "/bug/", "/ticket/")):
        return {
            "url": url,
            "category": CATEGORY_ISSUE_TRACKER,
            "source": source,
            "confidence": "low",
            "nvd_tags": nvd_tags,
        }

    # Default: other
    return {
        "url": url,
        "category": CATEGORY_OTHER,
        "source": source,
        "confidence": "low",
        "nvd_tags": nvd_tags,
    }


def classify_cve_references(cve_id: str) -> dict[str, Any]:
    """Fetch NVD references for *cve_id* and classify each one.

    Returns a dict with:
    - cve_id: the queried CVE
    - total: number of references found
    - references: list of classified reference dicts
    - summary: counts per category
    - actionable: list of high-confidence patch/advisory references
    """
    cve_upper = cve_id.strip().upper()

    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_upper}"
    try:
        response = _nvd_get_with_retry(url)
        data = response.json()
    except requests.exceptions.RequestException as exc:
        return {"error": f"NVD API request failed: {exc}", "cve_id": cve_upper}

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return {"error": f"No NVD data found for {cve_upper}", "cve_id": cve_upper}

    cve_obj = vulns[0].get("cve", {})
    raw_refs = cve_obj.get("references", [])

    classified: list[dict[str, Any]] = []
    for ref in raw_refs:
        ref_url = ref.get("url", "")
        ref_tags = ref.get("tags", [])
        if not ref_url:
            continue
        classified.append(classify_url(ref_url, nvd_tags=ref_tags))

    # Build summary counts
    summary: dict[str, int] = {}
    for item in classified:
        cat = item["category"]
        summary[cat] = summary.get(cat, 0) + 1

    # Extract actionable references (high/medium confidence patches + advisories)
    actionable = [
        r
        for r in classified
        if r["category"] in (CATEGORY_PATCH, CATEGORY_ADVISORY) and r["confidence"] in ("high", "medium")
    ]

    return {
        "cve_id": cve_upper,
        "total": len(classified),
        "references": classified,
        "summary": summary,
        "actionable": actionable,
    }


# ---------------------------------------------------------------------------
# Strands tool interface
# ---------------------------------------------------------------------------

TOOL_SPEC = {
    "name": "classify_references",
    "description": (
        "Fetches NVD reference URLs for a CVE and classifies each into actionable categories: "
        "patch (fix commits, PRs, releases), advisory (GHSA, vendor notices, coordinator alerts), "
        "exploit (PoC code, exploit-db entries), mailing_list, issue_tracker, vendor_notice, "
        "media (news articles), or other. Returns a ranked list with confidence levels and a "
        "summary count per category. Use this to quickly identify which references contain "
        "actionable patch information vs. background reading during triage."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier to classify references for (e.g., 'CVE-2024-3094').",
                }
            },
            "required": ["cve_id"],
        }
    },
}


def classify_references(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands tool entry point for classify_references."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id", "")

    if not isinstance(cve_id, str) or not cve_id.strip().upper().startswith("CVE-"):
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid CVE ID format. Must be a string like 'CVE-YYYY-NNNN'."}],
        }
        log_tool_output_size("classify_references", result)
        return result

    data = classify_cve_references(cve_id)

    if "error" in data:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": data["error"]}],
        }
        log_tool_output_size("classify_references", result)
        return result

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"json": data}],
    }
    log_tool_output_size("classify_references", result)
    return result
