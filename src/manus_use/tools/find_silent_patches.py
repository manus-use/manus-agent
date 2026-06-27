"""
Tool for detecting potential silent security patches in a GitHub repository.

A "silent patch" is a commit that:
  - Modifies code in a way that looks like a security fix (keywords in the
    commit message or diff: fix, sanitize, validate, escape, overflow, injection,
    auth, permission, …)
  - Has **no associated CVE ID** (no "CVE-XXXX-YYYY" in the message or in the
    linked PR/issue body)

Silent patches are a major blind spot in standard CVE-based vulnerability
management workflows.  Vendors sometimes quietly fix security bugs without
filing a CVE — either deliberately (to limit exposure) or because the fix
predates the CVE assignment.  Finding them requires inspecting commit history
directly.

Usage
-----
  manus-use silent-patches owner/repo
  manus-use silent-patches owner/repo --since 2024-01-01 --output json

The tool fetches the commit list from GitHub's REST API (no auth required for
public repos; set GITHUB_TOKEN env var to raise rate limits) and classifies
each commit using a two-stage heuristic:

  Stage 1 — message scan   : security keywords in the commit subject / body
  Stage 2 — diff scan      : security keywords in the unified diff (optional,
                              enabled unless --fast / fast=True is passed)

Results are scored (0-100) and returned in descending order.

Only public GitHub repositories are supported.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_use.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# Strands tool spec
# ---------------------------------------------------------------------------

TOOL_SPEC = {
    "name": "find_silent_patches",
    "description": (
        "Scans a GitHub repository's commit history for potential silent security fixes — "
        "commits that look like security patches (based on keywords in the commit message "
        "or diff) but have no associated CVE ID. "
        "Returns a list of candidate commits ranked by a suspicion score, together with "
        "the matched security keywords, the bug class inferred from the diff, and a direct "
        "link to the commit on GitHub. "
        "Useful for finding vulnerabilities that were quietly fixed before a CVE was assigned, "
        "or that the vendor chose not to disclose publicly."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": ("GitHub repository in 'owner/repo' format (e.g., 'django/django' or 'curl/curl')."),
                },
                "since": {
                    "type": "string",
                    "description": (
                        "ISO 8601 date string (YYYY-MM-DD). "
                        "Only commits after this date are scanned. "
                        "Defaults to 90 days ago."
                    ),
                },
                "until": {
                    "type": "string",
                    "description": (
                        "ISO 8601 date string (YYYY-MM-DD). "
                        "Only commits before this date are scanned. "
                        "Defaults to today."
                    ),
                },
                "max_commits": {
                    "type": "integer",
                    "description": ("Maximum number of commits to inspect. Defaults to 200. Maximum 500."),
                },
                "fast": {
                    "type": "boolean",
                    "description": (
                        "When true, skip the diff-content scan and rely only on commit "
                        "message keywords. Much faster but less accurate. Defaults to false."
                    ),
                },
            },
            "required": ["repo"],
        }
    },
}

# ---------------------------------------------------------------------------
# Keyword tables
# ---------------------------------------------------------------------------

# Commit-message keywords that suggest a security fix.
# Each tuple: (keyword_pattern, weight)
_MSG_KEYWORDS: list[tuple[str, int]] = [
    # High-signal terms
    (r"\bsecurity\b", 20),
    (r"\bvulnerabilit", 20),
    (r"\bexploit\b", 20),
    (r"\bcve-\d{4}-\d+", 25),  # would be CVE — included to *exclude* later
    (r"\bghsa-", 20),  # GitHub Security Advisory
    (r"\badvisory\b", 15),
    # Medium-signal
    (r"\bsanitize\b", 12),
    (r"\bsanitise\b", 12),
    (r"\bescape\b", 10),
    (r"\binjection\b", 15),
    (r"\boverflow\b", 15),
    (r"\buse[- ]after[- ]free\b", 20),
    (r"\bbuffer\b", 8),
    (r"\bauth(?:entication|oriz)?\b", 10),
    (r"\bpermission\b", 8),
    (r"\bprivilege\b", 10),
    (r"\baccess[- ]control\b", 12),
    (r"\bbypass\b", 12),
    (r"\bdenial[- ]of[- ]service\b", 15),
    (r"\bd(?:o|e)s\b", 8),
    (r"\brace[- ]condition\b", 12),
    (r"\bpath[- ]traversal\b", 15),
    (r"\bdirectory[- ]traversal\b", 15),
    (r"\bsql[- ]inject", 15),
    (r"\bxss\b", 15),
    (r"\bcross[- ]site\b", 12),
    (r"\bcsrf\b", 15),
    (r"\bssrf\b", 15),
    (r"\bremote[- ]code[- ]exec", 20),
    (r"\brce\b", 18),
    (r"\bnull[- ](?:pointer|deref)", 12),
    (r"\bheap\b", 8),
    (r"\bmemory[- ](?:corrupt|leak|safety)", 15),
    (r"\binteger[- ]overflow\b", 15),
    (r"\bformat[- ]string\b", 15),
    (r"\btype[- ]confusion\b", 15),
    (r"\bunintended\b", 6),
    (r"\bfix\b", 5),  # generic but common in security patches
    (r"\bpatch\b", 5),
    (r"\bmitigat", 8),
    (r"\bharden", 8),
    (r"\bprotect\b", 5),
    (r"\bvalidat", 7),
    (r"\bsecure\b", 8),
    (r"\bimproper\b", 8),
    (r"\bcritical\b", 10),
    (r"\bmalicious\b", 12),
    (r"\battack\b", 10),
]

# Diff-level keywords that suggest security-sensitive changes
_DIFF_KEYWORDS: list[tuple[str, int]] = [
    (r"\bsanitize\b", 10),
    (r"\bsanitise\b", 10),
    (r"\bhtml\.escape\b", 10),
    (r"\bescape\(", 8),
    (r"\bvalidate\b", 8),
    (r"\bcheck_permission\b", 12),
    (r"\bis_authenticated\b", 12),
    (r"\bhas_permission\b", 12),
    (r"\bpermission_required\b", 12),
    (r"\bauth(?:oriz|entic)", 10),
    (r"\boverflow\b", 12),
    (r"\bfree\(", 10),
    (r"\bkfree\(", 10),
    (r"\bnull[- ]check\b", 8),
    (r"\bassert\b", 5),
    (r"\braise\b.*[Ee]rror", 5),
    (r"\bif not\b", 4),
    (r"\bif\b.*is None", 5),
    (r"\bsql\b", 8),
    (r"\bexecute\(", 8),
    (r"\bprepare\(", 8),
    (r"\bparameterized\b", 10),
    (r"\bsubprocess\b", 8),
    (r"shell=False", 10),
    (r"\bos\.path\.realpath\b", 8),
    (r"\bos\.path\.abspath\b", 6),
    (r"\bpickle\b", 10),
    (r"\byaml\.safe_load\b", 10),
    (r"\bhmac\b", 10),
    (r"\bconstant_time_compare\b", 12),
    (r"\bsecrets\b", 8),
    (r"\bcryptography\b", 8),
    (r"\bssl_verify\b", 8),
    (r"\bcert(?:ificate)?\b", 6),
    (r"\bcsp\b", 8),
    (r"\bx-content-type\b", 8),
    (r"\bx-frame-options\b", 8),
    (r"\bstrict-transport\b", 8),
]

# Bug classes inferred from diff content
_BUG_CLASSES: list[tuple[str, list[str]]] = [
    ("sql_injection", [r"\bsql\b", r"\bexecute\(", r"\bparameterized\b", r"\bprepare\("]),
    ("command_injection", [r"\bsubprocess\b", r"shell=False", r"\bshlex\b"]),
    ("path_traversal", [r"realpath", r"abspath", r"\.\./", r"\.\.\\\\"]),
    ("buffer_overflow", [r"\boverflow\b", r"\bmemcpy\b", r"\bstrcpy\b"]),
    ("use_after_free", [r"\bfree\(", r"\bkfree\(", r"use.after.free"]),
    ("null_dereference", [r"null.check", r"is None", r"!= null"]),
    ("auth_bypass", [r"\bauth", r"\bpermission", r"\sis_authenticated\b", r"\bhas_permission\b"]),
    ("xss", [r"\bescape\b", r"html\.escape", r"\bsanitize\b", r"\bsanitise\b"]),
    ("deserialization", [r"\bpickle\b", r"yaml\.safe_load", r"\bunmarshal\b"]),
    ("cryptographic", [r"\bhmac\b", r"\bconstant_time_compare\b", r"\bcryptography\b"]),
    ("input_validation", [r"\bvalidate\b", r"\bassert\b", r"\braises?\b.*[Ee]rror"]),
    ("csrf_ssrf", [r"\bcsrf\b", r"\bssrf\b", r"\brequest_forgery\b"]),
    ("information_disclosure", [r"\bsecrets\b", r"\bcert\b", r"\bssl_verify\b"]),
    ("header_injection", [r"\bx-content-type\b", r"\bx-frame-options\b", r"\bcsp\b"]),
]

_CVE_RE = re.compile(r"\bCVE-\d{4}-\d+\b", re.IGNORECASE)
_GHSA_RE = re.compile(r"\bGHSA-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------


def _github_headers() -> dict[str, str]:
    """Return request headers for GitHub API, including auth token if present."""
    headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_json(url: str, headers: dict[str, str] | None = None, timeout: int = 15) -> Any:
    resp = requests.get(url, headers=headers or {}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _default_since() -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=90)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_date(date_str: str) -> str:
    """Accept YYYY-MM-DD or full ISO-8601 and normalise to GitHub format."""
    date_str = date_str.strip()
    if len(date_str) == 10:
        return f"{date_str}T00:00:00Z"
    return date_str


# ---------------------------------------------------------------------------
# Commit scoring
# ---------------------------------------------------------------------------


def _score_message(message: str) -> tuple[int, list[str]]:
    """Return (score, matched_keywords) for a commit message."""
    lower = message.lower()
    score = 0
    matched: list[str] = []
    for pattern, weight in _MSG_KEYWORDS:
        if re.search(pattern, lower):
            matched.append(pattern.strip(r"\b"))
            score += weight
    return score, matched


def _score_diff(diff_text: str) -> tuple[int, list[str]]:
    """Return (score, matched_keywords) based on diff content."""
    lower = diff_text.lower()
    score = 0
    matched: list[str] = []
    for pattern, weight in _DIFF_KEYWORDS:
        if re.search(pattern, lower):
            clean = re.sub(r"^\\b|\\b$|\\\\\(.*$", "", pattern)
            matched.append(clean)
            score += weight
    return score, matched


def _infer_bug_class(diff_text: str) -> str | None:
    """Return the most-likely bug class from the diff, or None."""
    lower = diff_text.lower()
    best_class: str | None = None
    best_hits = 0
    for bug_class, patterns in _BUG_CLASSES:
        hits = sum(1 for p in patterns if re.search(p, lower))
        if hits > best_hits:
            best_hits = hits
            best_class = bug_class
    return best_class if best_hits >= 2 else None


def _has_cve_reference(text: str) -> bool:
    return bool(_CVE_RE.search(text) or _GHSA_RE.search(text))


# ---------------------------------------------------------------------------
# GitHub API calls
# ---------------------------------------------------------------------------


def _list_commits(
    repo: str,
    since: str,
    until: str,
    max_commits: int,
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    """Paginate the GitHub commit list API and return up to max_commits items."""
    commits: list[dict[str, Any]] = []
    page = 1
    per_page = min(100, max_commits)
    while len(commits) < max_commits:
        url = f"https://api.github.com/repos/{repo}/commits?since={since}&until={until}&per_page={per_page}&page={page}"
        try:
            data = _fetch_json(url, headers=headers)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise ValueError(f"Repository '{repo}' not found on GitHub.") from exc
            raise
        if not data:
            break
        commits.extend(data)
        if len(data) < per_page:
            break
        page += 1
    return commits[:max_commits]


def _fetch_commit_diff(repo: str, sha: str, headers: dict[str, str]) -> str:
    """Fetch the unified diff for a single commit."""
    diff_headers = dict(headers)
    diff_headers["Accept"] = "application/vnd.github.diff"
    url = f"https://api.github.com/repos/{repo}/commits/{sha}"
    try:
        resp = requests.get(url, headers=diff_headers, timeout=20)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def find_silent_patches_impl(
    repo: str,
    since: str | None = None,
    until: str | None = None,
    max_commits: int = 200,
    fast: bool = False,
) -> dict[str, Any]:
    """
    Scan *repo* for commits that look like silent security fixes.

    Returns a dict with:
      - repo: str
      - since / until: str
      - total_scanned: int
      - candidates: list[dict]  — sorted by score descending
      - summary: str
    """
    repo = re.sub(r"^https?://github\.com/", "", repo.strip()).strip("/")
    if "/" not in repo:
        return {"error": f"Invalid repo format '{repo}'. Expected 'owner/repo'."}

    max_commits = min(max(1, max_commits), 500)

    since_str = _parse_date(since) if since else _default_since()
    until_str = _parse_date(until) if until else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    headers = _github_headers()

    try:
        commits = _list_commits(repo, since_str, until_str, max_commits, headers)
    except ValueError as exc:
        return {"error": str(exc)}
    except requests.HTTPError as exc:
        return {"error": f"GitHub API error: {exc}"}
    except requests.RequestException as exc:
        return {"error": f"Network error: {exc}"}

    candidates: list[dict[str, Any]] = []

    for commit_obj in commits:
        sha = commit_obj.get("sha", "")[:12]
        full_sha = commit_obj.get("sha", "")
        commit_data = commit_obj.get("commit", {})
        message = commit_data.get("message", "")
        author_info = commit_data.get("author", {})
        date_str = author_info.get("date", "")
        author_name = author_info.get("name", "unknown")
        url = commit_obj.get("html_url", f"https://github.com/{repo}/commit/{full_sha}")

        # Skip if the commit message itself contains a CVE/GHSA reference
        # (these are overt disclosures, not silent patches)
        if _has_cve_reference(message):
            continue

        msg_score, msg_keywords = _score_message(message)
        if msg_score < 5:
            # Not interesting based on message alone; skip diff scan
            continue

        diff_score = 0
        diff_keywords: list[str] = []
        bug_class: str | None = None

        if not fast:
            diff_text = _fetch_commit_diff(repo, full_sha, headers)
            if diff_text:
                diff_score, diff_keywords = _score_diff(diff_text)
                bug_class = _infer_bug_class(diff_text)

        total_score = min(100, msg_score + diff_score)

        if total_score >= 10:
            all_keywords = list(dict.fromkeys(msg_keywords + diff_keywords))  # dedup, preserve order
            candidates.append(
                {
                    "sha": sha,
                    "date": date_str[:10] if date_str else "",
                    "author": author_name,
                    "message": message.split("\n")[0][:120],
                    "score": total_score,
                    "msg_score": msg_score,
                    "diff_score": diff_score,
                    "keywords": all_keywords[:10],
                    "bug_class": bug_class,
                    "url": url,
                }
            )

    # Sort by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)

    summary_parts = [
        f"Scanned {len(commits)} commits in {repo} ({since_str[:10]} → {until_str[:10]}).",
        f"Found {len(candidates)} potential silent security patch(es) (no CVE reference).",
    ]
    if candidates:
        top = candidates[0]
        summary_parts.append(f"Highest-suspicion commit: {top['sha']} (score {top['score']}) — '{top['message'][:60]}'")

    return {
        "repo": repo,
        "since": since_str[:10],
        "until": until_str[:10],
        "total_scanned": len(commits),
        "candidates": candidates,
        "summary": " ".join(summary_parts),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_silent_patches_text(result: dict[str, Any]) -> str:
    """Render the find_silent_patches result as a human-readable report."""
    if "error" in result:
        return f"Error: {result['error']}"

    lines: list[str] = []
    lines.append(f"Silent Patch Scan: {result['repo']}")
    lines.append("=" * 60)
    lines.append(f"  Period  : {result['since']} → {result['until']}")
    lines.append(f"  Scanned : {result['total_scanned']} commits")
    lines.append(f"  Found   : {len(result['candidates'])} candidate(s) with no CVE reference")
    lines.append("")

    candidates = result.get("candidates", [])
    if not candidates:
        lines.append("  No suspicious commits found in this period.")
        return "\n".join(lines)

    lines.append("Candidates (sorted by suspicion score)")
    lines.append("-" * 60)

    for i, c in enumerate(candidates, 1):
        lines.append(f"  [{i}] {c['sha']}  score={c['score']:3d}  {c['date']}  by {c['author']}")
        lines.append(f"       {c['message']}")
        if c.get("bug_class"):
            lines.append(f"       Bug class : {c['bug_class']}")
        if c.get("keywords"):
            lines.append(f"       Keywords  : {', '.join(c['keywords'])}")
        lines.append(f"       URL       : {c['url']}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strands entry point
# ---------------------------------------------------------------------------


def find_silent_patches(tool: ToolUse, **kwargs: Any) -> ToolResult:  # noqa: ARG001
    """Strands tool entry point for finding silent security patches in a GitHub repo."""
    inp = tool.get("input", {})
    repo = inp.get("repo", "")
    since = inp.get("since")
    until = inp.get("until")
    max_commits = int(inp.get("max_commits", 200))
    fast = bool(inp.get("fast", False))

    if not repo:
        return {
            "toolUseId": tool["toolUseId"],
            "status": "error",
            "content": [{"text": "Missing required parameter: repo"}],
        }

    result = find_silent_patches_impl(
        repo=repo,
        since=since,
        until=until,
        max_commits=max_commits,
        fast=fast,
    )

    log_tool_output_size("find_silent_patches", result)

    return {
        "toolUseId": tool["toolUseId"],
        "status": "success",
        "content": [{"text": str(result)}],
    }
