"""Silent patch detector — find security fixes that were never assigned a CVE.

Scans a GitHub repository's commit history for commits whose messages or diffs
indicate a security fix but lack any associated CVE identifier. Uses two-stage
heuristic scoring:

1. **Message keywords** — commit message is checked against patterns typical
   of security fixes (e.g. "fix buffer overflow", "sanitize input").
2. **Diff keywords** — if enabled, the commit diff is fetched and scanned for
   security-relevant code patterns (e.g. bounds checks added, auth guards).

Each candidate commit is classified into one of 14 bug classes based on the
matching keywords.

Public API
----------
- ``detect_silent_patches(owner_repo, ...)`` — main entry point
- ``BUG_CLASSES`` — the 14 supported bug-class labels
"""

from __future__ import annotations

import datetime
import os
import re
from typing import Any
from urllib.parse import quote

import requests

# ---------------------------------------------------------------------------
# Bug classes
# ---------------------------------------------------------------------------

BUG_CLASSES: list[str] = [
    "auth_bypass",
    "buffer_overflow",
    "command_injection",
    "csrf",
    "directory_traversal",
    "information_disclosure",
    "integer_overflow",
    "memory_corruption",
    "null_dereference",
    "privilege_escalation",
    "race_condition",
    "sql_injection",
    "use_after_free",
    "xss",
]

# ---------------------------------------------------------------------------
# Keyword patterns — message stage
# ---------------------------------------------------------------------------

# Each entry maps a regex pattern (applied to the commit message, case-insensitive)
# to a list of bug-class labels it suggests.
_MESSAGE_PATTERNS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"auth(?:entication|orization)?\s+bypass", re.I), ["auth_bypass"]),
    (re.compile(r"bypass\s+(?:auth|access|permission)", re.I), ["auth_bypass"]),
    (re.compile(r"buffer\s+over(?:flow|run)", re.I), ["buffer_overflow"]),
    (re.compile(r"heap\s+(?:over(?:flow|run)|corrupt)", re.I), ["buffer_overflow", "memory_corruption"]),
    (re.compile(r"stack\s+(?:over(?:flow|run)|corrupt)", re.I), ["buffer_overflow", "memory_corruption"]),
    (re.compile(r"out[- ]of[- ]bounds?\s+(?:read|write|access)", re.I), ["buffer_overflow"]),
    (re.compile(r"command\s+injection", re.I), ["command_injection"]),
    (re.compile(r"(?:os|shell)\s+injection", re.I), ["command_injection"]),
    (re.compile(r"(?:unsanitized|unescaped)\s+(?:input|command|shell)", re.I), ["command_injection"]),
    (re.compile(r"csrf|cross[- ]site\s+request\s+forgery", re.I), ["csrf"]),
    (re.compile(r"directory\s+traversal", re.I), ["directory_traversal"]),
    (re.compile(r"path\s+traversal", re.I), ["directory_traversal"]),
    (re.compile(r"\.\./\s*(?:attack|vuln|exploit)", re.I), ["directory_traversal"]),
    (re.compile(r"information\s+(?:disclosure|leak(?:age)?)", re.I), ["information_disclosure"]),
    (re.compile(r"(?:sensitive|private)\s+(?:data|info)\s+(?:expos|leak)", re.I), ["information_disclosure"]),
    (re.compile(r"integer\s+over(?:flow|run)", re.I), ["integer_overflow"]),
    (re.compile(r"(?:signed|unsigned)\s+(?:integer|int)\s+(?:wrap|trunc)", re.I), ["integer_overflow"]),
    (re.compile(r"memory\s+corrupt", re.I), ["memory_corruption"]),
    (re.compile(r"double\s+free", re.I), ["memory_corruption", "use_after_free"]),
    (re.compile(r"null\s+(?:pointer\s+)?deref", re.I), ["null_dereference"]),
    (re.compile(r"(?:nullptr|NULL)\s+(?:access|deref)", re.I), ["null_dereference"]),
    (re.compile(r"privilege\s+escalat", re.I), ["privilege_escalation"]),
    (re.compile(r"(?:local|remote)\s+(?:root|admin)\s+(?:access|exploit)", re.I), ["privilege_escalation"]),
    (re.compile(r"race\s+condition", re.I), ["race_condition"]),
    (re.compile(r"(?:toctou|time[- ]of[- ]check)", re.I), ["race_condition"]),
    (re.compile(r"sql\s+injection", re.I), ["sql_injection"]),
    (re.compile(r"(?:unsanitized|unescaped)\s+(?:sql|query)", re.I), ["sql_injection"]),
    (re.compile(r"use[- ]after[- ]free", re.I), ["use_after_free"]),
    (re.compile(r"(?:dangling|stale)\s+pointer", re.I), ["use_after_free"]),
    (re.compile(r"xss|cross[- ]site\s+script", re.I), ["xss"]),
    (re.compile(r"(?:reflected|stored|dom)[- ]?xss", re.I), ["xss"]),
    # Generic security fix patterns (classify as information_disclosure fallback)
    (re.compile(r"(?:fix|patch|resolve)\s+(?:security|vuln)", re.I), ["information_disclosure"]),
    (re.compile(r"(?:security|vuln)\s+(?:fix|patch|update)", re.I), ["information_disclosure"]),
    (re.compile(r"sanitize\s+(?:input|user|data)", re.I), ["xss", "command_injection"]),
    (re.compile(r"(?:validate|check)\s+(?:bounds|size|length|limit)", re.I), ["buffer_overflow"]),
    (re.compile(r"(?:prevent|block|mitigate)\s+(?:inject|overflow|bypass)", re.I), ["command_injection"]),
]

# ---------------------------------------------------------------------------
# Keyword patterns — diff stage
# ---------------------------------------------------------------------------

_DIFF_PATTERNS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"\+.*\b(?:if|assert|check)\s*\(.*(?:len|size|count|bound)", re.I), ["buffer_overflow"]),
    (re.compile(r"\+.*\b(?:escape|sanitize|encode|quote)\s*\(", re.I), ["xss", "sql_injection", "command_injection"]),
    (re.compile(r"\+.*csrf[_-]?token", re.I), ["csrf"]),
    (re.compile(r"\+.*path\.(?:normalize|resolve|join)", re.I), ["directory_traversal"]),
    (re.compile(r"\+.*realpath", re.I), ["directory_traversal"]),
    (re.compile(r"\+.*(?:free|release|destroy)\s*\(.*(?:null|NULL)", re.I), ["use_after_free"]),
    (re.compile(r"\+.*=\s*NULL\s*;.*(?:after|before)\s+free", re.I), ["use_after_free"]),
    (re.compile(r"\+.*(?:mutex|lock|synchroniz|atomic)", re.I), ["race_condition"]),
    (re.compile(r"\+.*\b(?:parameterized|prepared)\s+(?:statement|query)", re.I), ["sql_injection"]),
    (
        re.compile(r"\+.*\b(?:is_?admin|has_?perm|check_?auth|require_?auth)", re.I),
        ["auth_bypass", "privilege_escalation"],
    ),
    (re.compile(r"\+.*\b(?:INT_MAX|SIZE_MAX|UINT_MAX|overflow)", re.I), ["integer_overflow"]),
]

# ---------------------------------------------------------------------------
# CVE reference regex (to exclude commits that already reference a CVE)
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.I)

# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

_GITHUB_API = "https://api.github.com"
_SESSION: requests.Session | None = None


def _get_session() -> requests.Session:
    """Return a reusable requests session with GitHub auth if available."""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            _SESSION.headers["Authorization"] = f"token {token}"
        _SESSION.headers["Accept"] = "application/vnd.github+json"
        _SESSION.headers["X-GitHub-Api-Version"] = "2022-11-28"
    return _SESSION


def _fetch_commits(
    owner_repo: str,
    since: str | None = None,
    until: str | None = None,
    max_commits: int = 500,
) -> list[dict[str, Any]]:
    """Fetch commits from GitHub REST API with pagination."""
    session = _get_session()
    params: dict[str, Any] = {"per_page": min(max_commits, 100)}
    if since:
        params["since"] = since + "T00:00:00Z" if "T" not in since else since
    if until:
        params["until"] = until + "T23:59:59Z" if "T" not in until else until

    url = f"{_GITHUB_API}/repos/{quote(owner_repo, safe='/')}/commits"
    commits: list[dict[str, Any]] = []

    while url and len(commits) < max_commits:
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        commits.extend(data)
        # Follow pagination
        link = resp.headers.get("Link", "")
        url = ""
        params = {}
        if 'rel="next"' in link:
            for part in link.split(","):
                if 'rel="next"' in part:
                    url = part.split(";")[0].strip(" <>")
                    break

    return commits[:max_commits]


def _fetch_diff(owner_repo: str, sha: str) -> str:
    """Fetch the diff for a specific commit."""
    session = _get_session()
    url = f"{_GITHUB_API}/repos/{quote(owner_repo, safe='/')}/commits/{sha}"
    headers = {"Accept": "application/vnd.github.diff"}
    resp = session.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------


def _score_message(message: str) -> tuple[float, list[str]]:
    """Score a commit message. Returns (score, matched_classes)."""
    classes: set[str] = set()
    hits = 0
    for pattern, labels in _MESSAGE_PATTERNS:
        if pattern.search(message):
            hits += 1
            classes.update(labels)

    # Normalize: more hits = higher confidence, capped at 1.0
    score = min(hits * 0.3, 1.0) if hits else 0.0
    return score, sorted(classes)


def _score_diff(diff_text: str) -> tuple[float, list[str]]:
    """Score a diff. Returns (score, matched_classes)."""
    classes: set[str] = set()
    hits = 0
    # Only look at added lines (limit scan to prevent huge diffs from dominating)
    lines = diff_text[:200_000].split("\n")
    for line in lines:
        if not line.startswith("+"):
            continue
        for pattern, labels in _DIFF_PATTERNS:
            if pattern.search(line):
                hits += 1
                classes.update(labels)
                break  # one match per line is enough

    score = min(hits * 0.15, 1.0) if hits else 0.0
    return score, sorted(classes)


def _classify(classes: list[str]) -> str:
    """Pick the most specific class from the matched set."""
    if not classes:
        return "unknown"
    # Prefer more specific classes over generic fallback
    for c in classes:
        if c != "information_disclosure":
            return c
    return classes[0]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def detect_silent_patches(
    owner_repo: str,
    *,
    since: str | None = None,
    until: str | None = None,
    max_commits: int = 500,
    fast: bool = False,
    min_score: float = 0.3,
) -> list[dict[str, Any]]:
    """Detect silently patched security fixes in a GitHub repository.

    Parameters
    ----------
    owner_repo : str
        GitHub repository in ``owner/repo`` format.
    since : str, optional
        ISO date string (YYYY-MM-DD). Defaults to 90 days ago.
    until : str, optional
        ISO date string (YYYY-MM-DD). Defaults to today.
    max_commits : int
        Maximum number of commits to fetch (default 500).
    fast : bool
        If True, skip diff scoring (message keywords only).
    min_score : float
        Minimum combined score to include a commit (default 0.3).

    Returns
    -------
    list[dict]
        List of candidate commits with scoring metadata.
    """
    if not since:
        since = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
    if not until:
        until = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    commits = _fetch_commits(owner_repo, since=since, until=until, max_commits=max_commits)

    candidates: list[dict[str, Any]] = []

    for commit_data in commits:
        sha = commit_data.get("sha", "")
        commit_obj = commit_data.get("commit", {})
        message = commit_obj.get("message", "")

        # Skip commits that already reference a CVE
        if _CVE_RE.search(message):
            continue

        # Stage 1: message scoring
        msg_score, msg_classes = _score_message(message)
        if msg_score == 0.0:
            continue

        # Stage 2: diff scoring (unless --fast)
        diff_score = 0.0
        diff_classes: list[str] = []
        if not fast and msg_score >= 0.3:
            try:
                diff_text = _fetch_diff(owner_repo, sha)
                diff_score, diff_classes = _score_diff(diff_text)
            except (requests.RequestException, ValueError):
                pass  # network failure — keep msg_score only

        # Combined score: if diff was fetched and scored positively, boost the
        # message score; otherwise fall back to message score alone
        if not fast and diff_score > 0:
            combined_score = msg_score + diff_score * 0.4
        else:
            combined_score = msg_score
        # Cap at 1.0
        combined_score = min(combined_score, 1.0)
        if combined_score < min_score:
            continue

        all_classes = sorted(set(msg_classes + diff_classes))
        classification = _classify(all_classes)

        author_info = commit_data.get("author") or {}
        commit_date = commit_obj.get("committer", {}).get("date", "")

        candidates.append(
            {
                "sha": sha,
                "short_sha": sha[:8],
                "message": message.split("\n")[0][:120],
                "full_message": message,
                "author": author_info.get("login", commit_obj.get("author", {}).get("name", "unknown")),
                "date": commit_date,
                "url": commit_data.get("html_url", f"https://github.com/{owner_repo}/commit/{sha}"),
                "score": round(combined_score, 3),
                "message_score": round(msg_score, 3),
                "diff_score": round(diff_score, 3),
                "classification": classification,
                "matched_classes": all_classes,
            }
        )

    # Sort by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates
