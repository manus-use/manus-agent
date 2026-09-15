"""
Tool for detecting silent security patches in a GitHub repository.

A "silent patch" is a commit that fixes a security vulnerability but was never
assigned a CVE identifier. This tool scans a repository's recent commit history
using a two-stage heuristic:

  1. **Message scoring** — commit message keywords that suggest security fixes
     (e.g. "fix buffer overflow", "sanitize input", "prevent injection").
  2. **Diff scoring** — unified diff keywords that indicate security-relevant
     code changes (e.g. added bounds checks, input validation, auth guards).

Each candidate commit is labelled with one of 14 bug classes (aligned with
``get_patch_diff``), scored 0–100, and classified as HIGH / MEDIUM / LOW
confidence.

Only public GitHub repositories are supported (unauthenticated, or via
``GITHUB_TOKEN`` for higher rate limits).
"""

from __future__ import annotations

import os
import re
import time
from datetime import date, timedelta
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

__all__ = ["detect_silent_patches"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT = "manus-agent/silent-patches (github.com/manus-use/manus-agent)"
_GITHUB_API = "https://api.github.com"
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.0  # seconds

# ---------------------------------------------------------------------------
# Bug-class keyword mapping (aligned with get_patch_diff.py's 14 categories)
# ---------------------------------------------------------------------------

_BUG_CLASSES: list[tuple[str, list[str]]] = [
    (
        "sql_injection",
        ["sql injection", "sqli", "parameterized query", "prepared statement", "bind variable"],
    ),
    (
        "command_injection",
        ["command injection", "shell injection", "os.system", "subprocess", "shell=true", "popen"],
    ),
    (
        "path_traversal",
        ["path traversal", "directory traversal", "../", "realpath", "canonical path", "sanitize path"],
    ),
    (
        "buffer_overflow",
        ["buffer overflow", "heap overflow", "stack overflow", "bounds check", "memcpy", "strcpy", "strncpy"],
    ),
    (
        "integer_overflow",
        ["integer overflow", "int overflow", "arithmetic overflow", "wraparound", "size_t"],
    ),
    (
        "use_after_free",
        ["use after free", "use-after-free", "uaf", "dangling pointer", "double free"],
    ),
    (
        "null_dereference",
        ["null pointer", "null dereference", "nullptr", "null check", "none check"],
    ),
    (
        "auth_bypass",
        [
            "auth bypass",
            "authentication bypass",
            "authorization bypass",
            "privilege escalation",
            "access control",
            "permission check",
        ],
    ),
    (
        "deserialization",
        ["deserialization", "insecure deserializ", "pickle", "yaml.load", "unmarshal", "object injection"],
    ),
    (
        "xss",
        [
            "cross-site scripting",
            "xss",
            "html injection",
            "script injection",
            "escape html",
            "sanitize html",
        ],
    ),
    (
        "ssrf",
        ["ssrf", "server-side request", "url validation", "allowlist", "internal url", "localhost"],
    ),
    (
        "input_validation",
        ["input validation", "validate input", "sanitize input", "allowlist", "whitelist", "regex filter"],
    ),
    (
        "race_condition",
        ["race condition", "toctou", "time-of-check", "mutex", "lock", "synchronize", "atomic"],
    ),
    (
        "cryptographic",
        [
            "cryptographic",
            "weak cipher",
            "weak hash",
            "random number",
            "entropy",
            "timing attack",
            "constant-time",
        ],
    ),
]

# Message-level keywords that suggest a security fix.
_SECURITY_MESSAGE_KEYWORDS: list[tuple[str, float]] = [
    # Strong signals (exact phrases)
    ("security fix", 25.0),
    ("security patch", 25.0),
    ("security vulnerability", 25.0),
    ("fixes vulnerability", 25.0),
    ("prevent injection", 20.0),
    ("prevent traversal", 20.0),
    ("prevent overflow", 20.0),
    ("prevent bypass", 20.0),
    ("fix vulnerability", 20.0),
    ("patch vulnerability", 20.0),
    ("address vulnerability", 20.0),
    ("mitigate vulnerability", 20.0),
    ("resolve vulnerability", 20.0),
    # Moderate signals
    ("sanitize", 12.0),
    ("sanitise", 12.0),
    ("escape input", 15.0),
    ("validate input", 12.0),
    ("bounds check", 15.0),
    ("access control", 12.0),
    ("permission check", 12.0),
    ("auth check", 12.0),
    ("fix crash", 8.0),
    ("null check", 8.0),
    ("null pointer", 10.0),
    ("buffer overflow", 20.0),
    ("use after free", 20.0),
    ("use-after-free", 20.0),
    ("double free", 18.0),
    ("heap overflow", 20.0),
    ("stack overflow", 18.0),
    ("integer overflow", 18.0),
    ("out of bounds", 15.0),
    ("out-of-bounds", 15.0),
    ("memory corruption", 18.0),
    ("arbitrary code", 20.0),
    ("remote code execution", 25.0),
    ("rce", 15.0),
    ("denial of service", 12.0),
    ("dos", 6.0),
    ("injection", 10.0),
    ("traversal", 10.0),
    ("overflow", 8.0),
    ("bypass", 8.0),
    ("privilege escalation", 18.0),
    ("privesc", 15.0),
]

# Diff-level keywords that indicate security-relevant code changes.
_SECURITY_DIFF_KEYWORDS: list[tuple[str, float]] = [
    # Validation / sanitization
    ("sanitize(", 8.0),
    ("sanitise(", 8.0),
    ("escape(", 6.0),
    ("html.escape", 8.0),
    ("htmlentities", 8.0),
    ("validate(", 5.0),
    ("is_valid(", 5.0),
    # Auth / access control
    ("check_permission", 8.0),
    ("is_authenticated", 8.0),
    ("require_auth", 8.0),
    ("has_permission", 8.0),
    ("access_denied", 6.0),
    ("forbidden", 4.0),
    # Bounds / memory safety
    ("bounds_check", 8.0),
    ("range_check", 6.0),
    ("size_check", 6.0),
    ("length_check", 6.0),
    ("MAX_SIZE", 5.0),
    ("MAX_LENGTH", 5.0),
    ("MIN_SIZE", 4.0),
    # Crypto
    ("constant_time", 8.0),
    ("timing_safe", 8.0),
    ("secure_random", 6.0),
    ("crypto_", 4.0),
    # Error handling / null safety
    ("if err != nil", 3.0),
    ("!= null", 3.0),
    ("!= nullptr", 4.0),
    ("is None", 2.0),
    # SQL / command safety
    ("parameterized", 6.0),
    ("prepared_statement", 8.0),
    ("bind_param", 6.0),
    ("shell=False", 8.0),
    ("shlex.quote", 8.0),
    ("subprocess.run(", 3.0),
]

# Commit-message patterns that indicate a CVE was already assigned
# (we want to exclude these — they are NOT silent patches).
_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
_GHSA_RE = re.compile(r"GHSA-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}", re.IGNORECASE)

# Owner/repo validation
_REPO_RE = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _github_headers() -> dict[str, str]:
    """Build GitHub API headers with optional GITHUB_TOKEN."""
    token = os.environ.get("GITHUB_TOKEN", "")
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "User-Agent": _USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_get(url: str, params: dict | None = None, timeout: int = 15) -> requests.Response:
    """GET from GitHub API with retry + back-off."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, headers=_github_headers(), params=params, timeout=timeout)
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                wait = _RETRY_BACKOFF * (2**attempt)
                time.sleep(wait)
                last_exc = requests.HTTPError(f"Rate limited (attempt {attempt + 1})")
                continue
            if resp.status_code == 502:
                wait = _RETRY_BACKOFF * (2**attempt)
                time.sleep(wait)
                last_exc = requests.HTTPError(f"502 Bad Gateway (attempt {attempt + 1})")
                continue
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BACKOFF * (2**attempt))
    raise last_exc or requests.RequestException("GitHub API request failed")


def _github_diff_get(url: str, timeout: int = 15) -> requests.Response:
    """GET a raw diff from GitHub with retry + back-off."""
    headers = _github_headers()
    headers["Accept"] = "application/vnd.github.diff"
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                wait = _RETRY_BACKOFF * (2**attempt)
                time.sleep(wait)
                last_exc = requests.HTTPError(f"Rate limited (attempt {attempt + 1})")
                continue
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BACKOFF * (2**attempt))
    raise last_exc or requests.RequestException("GitHub diff request failed")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_message(message: str) -> float:
    """Score a commit message for security-fix likelihood (0–50)."""
    msg_lower = message.lower()
    score = 0.0
    for keyword, weight in _SECURITY_MESSAGE_KEYWORDS:
        if keyword.lower() in msg_lower:
            score += weight
    return min(score, 50.0)


def _score_diff(diff_text: str) -> float:
    """Score a unified diff for security-relevant code changes (0–50)."""
    diff_lower = diff_text.lower()
    score = 0.0
    for keyword, weight in _SECURITY_DIFF_KEYWORDS:
        # Count occurrences but cap contribution per keyword.
        count = diff_lower.count(keyword.lower())
        if count > 0:
            score += weight * min(count, 3)
    return min(score, 50.0)


def _classify_bug_classes(text: str) -> list[str]:
    """Identify matching bug classes from combined message + diff text."""
    text_lower = text.lower()
    matches: list[str] = []
    for bug_class, keywords in _BUG_CLASSES:
        for kw in keywords:
            if kw.lower() in text_lower:
                matches.append(bug_class)
                break
    return matches


def _confidence_label(score: float) -> str:
    """Map a 0–100 score to a confidence label."""
    if score >= 60:
        return "HIGH"
    if score >= 30:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Core logic: fetch commits and score them
# ---------------------------------------------------------------------------


def fetch_commits(
    owner: str,
    repo: str,
    since: str | None = None,
    until: str | None = None,
    max_commits: int = 500,
) -> list[dict[str, Any]]:
    """Fetch commits from a GitHub repo within the given date range.

    Returns a list of commit dicts with keys: sha, message, date, url, author.
    """
    params: dict[str, Any] = {"per_page": min(max_commits, 100)}
    if since:
        params["since"] = f"{since}T00:00:00Z"
    if until:
        params["until"] = f"{until}T23:59:59Z"

    url = f"{_GITHUB_API}/repos/{owner}/{repo}/commits"
    all_commits: list[dict[str, Any]] = []
    page = 1

    while len(all_commits) < max_commits:
        params["page"] = page
        resp = _github_get(url, params=params)
        if resp.status_code == 404:
            raise ValueError(f"Repository {owner}/{repo} not found (404)")
        if resp.status_code == 409:
            # Empty repository
            return []
        if resp.status_code != 200:
            raise ValueError(f"GitHub API error {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        if not data:
            break

        for item in data:
            if len(all_commits) >= max_commits:
                break
            commit_data = item.get("commit", {})
            author_data = commit_data.get("author", {})
            all_commits.append(
                {
                    "sha": item.get("sha", ""),
                    "message": commit_data.get("message", ""),
                    "date": author_data.get("date", ""),
                    "url": item.get("html_url", ""),
                    "author": author_data.get("name", ""),
                }
            )

        if len(data) < params["per_page"]:
            break
        page += 1

    return all_commits


def fetch_commit_diff(owner: str, repo: str, sha: str) -> str:
    """Fetch the unified diff for a single commit."""
    url = f"{_GITHUB_API}/repos/{owner}/{repo}/commits/{sha}"
    resp = _github_diff_get(url)
    if resp.status_code != 200:
        return ""
    return resp.text[:50_000]  # Cap at 50 KB to avoid memory issues


def _has_cve_or_advisory(message: str) -> bool:
    """Return True if the commit message references a CVE or GHSA."""
    return bool(_CVE_RE.search(message)) or bool(_GHSA_RE.search(message))


def scan_repository(
    owner: str,
    repo: str,
    since: str | None = None,
    until: str | None = None,
    max_commits: int = 500,
    fast: bool = False,
    message_threshold: float = 10.0,
) -> dict[str, Any]:
    """Scan a repository for silent security patches.

    Parameters
    ----------
    owner, repo : str
        GitHub owner and repository name.
    since, until : str or None
        ISO date strings (YYYY-MM-DD) for the commit window.
    max_commits : int
        Maximum commits to scan.
    fast : bool
        If True, skip diff scoring (message keywords only).
    message_threshold : float
        Minimum message score to consider a commit as a candidate for
        diff analysis. Commits below this threshold are skipped entirely.

    Returns
    -------
    dict
        Result with keys: owner, repo, since, until, total_commits_scanned,
        candidates (list of scored commits), summary.
    """
    if not since:
        since = (date.today() - timedelta(days=90)).isoformat()
    if not until:
        until = date.today().isoformat()

    commits = fetch_commits(owner, repo, since=since, until=until, max_commits=max_commits)

    candidates: list[dict[str, Any]] = []
    errors: list[str] = []

    for commit in commits:
        msg = commit["message"]

        # Skip commits that already reference a CVE or advisory.
        if _has_cve_or_advisory(msg):
            continue

        msg_score = _score_message(msg)

        # Skip low-scoring messages entirely.
        if msg_score < message_threshold:
            continue

        diff_score = 0.0
        diff_text = ""

        if not fast:
            try:
                diff_text = fetch_commit_diff(owner, repo, commit["sha"])
                diff_score = _score_diff(diff_text)
            except Exception as exc:
                errors.append(f"Diff fetch error for {commit['sha'][:7]}: {exc}")

        total_score = min(msg_score + diff_score, 100.0)
        combined_text = msg + "\n" + diff_text
        bug_classes = _classify_bug_classes(combined_text)

        candidates.append(
            {
                "sha": commit["sha"],
                "short_sha": commit["sha"][:7],
                "message": msg.split("\n")[0][:120],  # First line, truncated
                "date": commit["date"],
                "url": commit["url"],
                "author": commit["author"],
                "message_score": round(msg_score, 1),
                "diff_score": round(diff_score, 1),
                "total_score": round(total_score, 1),
                "confidence": _confidence_label(total_score),
                "bug_classes": bug_classes,
            }
        )

    # Sort by total score descending.
    candidates.sort(key=lambda c: c["total_score"], reverse=True)

    # Summary stats.
    high = sum(1 for c in candidates if c["confidence"] == "HIGH")
    medium = sum(1 for c in candidates if c["confidence"] == "MEDIUM")
    low = sum(1 for c in candidates if c["confidence"] == "LOW")

    bug_class_counts: dict[str, int] = {}
    for c in candidates:
        for bc in c["bug_classes"]:
            bug_class_counts[bc] = bug_class_counts.get(bc, 0) + 1

    return {
        "owner": owner,
        "repo": repo,
        "since": since,
        "until": until,
        "total_commits_scanned": len(commits),
        "candidates": candidates,
        "summary": {
            "total_candidates": len(candidates),
            "high_confidence": high,
            "medium_confidence": medium,
            "low_confidence": low,
            "bug_class_distribution": dict(sorted(bug_class_counts.items(), key=lambda x: x[1], reverse=True)),
        },
        "errors": errors,
        "fast_mode": fast,
    }


# ---------------------------------------------------------------------------
# Strands tool interface
# ---------------------------------------------------------------------------


def detect_silent_patches(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Scan a GitHub repository for silent security patches (no CVE assigned).

    Accepts ``owner_repo`` (e.g. ``"torvalds/linux"``), optional ``since``
    (YYYY-MM-DD), ``until`` (YYYY-MM-DD), ``max_commits`` (int, default 500),
    and ``fast`` (bool, default false — skip diff analysis).
    """
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    owner_repo: str = tool_input.get("owner_repo", "")

    if not isinstance(owner_repo, str) or not _REPO_RE.match(owner_repo.strip()):
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": ("Invalid repository format. Must be 'owner/repo' (e.g. 'torvalds/linux').")}],
        }
        log_tool_output_size("detect_silent_patches", result)
        return result

    parts = owner_repo.strip().split("/", 1)
    owner, repo = parts[0], parts[1]

    since = tool_input.get("since") or None
    until = tool_input.get("until") or None
    max_commits = int(tool_input.get("max_commits", 500))
    fast = bool(tool_input.get("fast", False))

    try:
        data = scan_repository(
            owner=owner,
            repo=repo,
            since=since,
            until=until,
            max_commits=max_commits,
            fast=fast,
        )
    except ValueError as exc:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": str(exc)}],
        }
        log_tool_output_size("detect_silent_patches", result)
        return result
    except requests.RequestException as exc:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"GitHub API request failed: {exc}"}],
        }
        log_tool_output_size("detect_silent_patches", result)
        return result

    import json

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": json.dumps(data, indent=2)}],
    }
    log_tool_output_size("detect_silent_patches", result)
    return result
