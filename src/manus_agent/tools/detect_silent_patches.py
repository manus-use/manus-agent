"""
Tool for detecting silent security patches in a GitHub repository.

A "silent patch" is a commit that fixes a security vulnerability but was never
assigned a CVE identifier.  These are common: maintainers frequently fix bugs
without going through the CVE process, leaving downstream users unaware that a
security-relevant change was made.

Detection strategy (two-stage heuristic scoring):

  **Stage 1 — Message scoring:**
    Commit messages are scored against curated keyword lists for 14 security
    bug classes (e.g. ``auth_bypass``, ``buffer_overflow``, ``use_after_free``).
    Messages that reference a known CVE ID are excluded (they are *not* silent).

  **Stage 2 — Diff scoring (optional, skipped with ``--fast``):**
    For commits that pass stage 1, the unified diff is fetched and scored
    against diff-specific keywords (e.g. ``memcpy`` → buffer_overflow,
    ``sanitize`` → xss).  Diff scoring can promote or demote the initial
    message score.

Each candidate commit is labelled with its best-match bug class and an
overall confidence score (0–100).

Only public GitHub repositories are supported (unauthenticated, or via
``GITHUB_TOKEN``).
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# Bug-class keyword mappings
# ---------------------------------------------------------------------------

# Stage 1: commit message keywords (case-insensitive substring match)
_MSG_KEYWORDS: list[tuple[str, list[str]]] = [
    (
        "sql_injection",
        [
            "sql injection",
            "sqli",
            "parameterized query",
            "prepared statement",
            "sanitize sql",
            "escape sql",
        ],
    ),
    (
        "command_injection",
        [
            "command injection",
            "shell injection",
            "os.system",
            "subprocess",
            "shell=true",
            "code execution",
            "rce",
            "remote code",
        ],
    ),
    (
        "path_traversal",
        [
            "path traversal",
            "directory traversal",
            "lfi",
            "local file inclusion",
            "../ ",
            "dot-dot-slash",
            "realpath",
        ],
    ),
    (
        "buffer_overflow",
        [
            "buffer overflow",
            "heap overflow",
            "stack overflow",
            "out of bounds",
            "oob write",
            "oob read",
            "bounds check",
        ],
    ),
    (
        "integer_overflow",
        [
            "integer overflow",
            "int overflow",
            "wraparound",
            "wrap around",
            "truncation",
        ],
    ),
    (
        "use_after_free",
        [
            "use after free",
            "use-after-free",
            "uaf",
            "double free",
            "dangling pointer",
        ],
    ),
    (
        "null_dereference",
        [
            "null pointer",
            "null deref",
            "nullptr",
            "segfault",
            "null check",
        ],
    ),
    (
        "auth_bypass",
        [
            "auth bypass",
            "authentication bypass",
            "authorization bypass",
            "privilege escalation",
            "priv esc",
            "access control",
            "permission check",
        ],
    ),
    (
        "xss",
        [
            "cross-site scripting",
            "xss",
            "script injection",
            "html injection",
            "sanitize html",
            "escape html",
        ],
    ),
    (
        "csrf",
        [
            "csrf",
            "cross-site request forgery",
            "xsrf",
            "anti-forgery",
            "csrf token",
        ],
    ),
    (
        "ssrf",
        [
            "ssrf",
            "server-side request forgery",
            "url validation",
            "internal network",
        ],
    ),
    (
        "information_disclosure",
        [
            "information disclosure",
            "info leak",
            "data leak",
            "sensitive data",
            "credentials in",
            "password exposure",
            "secret exposure",
            "api key exposure",
        ],
    ),
    (
        "denial_of_service",
        [
            "denial of service",
            "dos",
            "resource exhaustion",
            "infinite loop",
            "memory exhaustion",
            "oom",
            "crash fix",
            "stack exhaustion",
            "regex dos",
            "redos",
        ],
    ),
    (
        "deserialization",
        [
            "deserialization",
            "insecure deseriali",
            "pickle",
            "yaml.load",
            "unsafe load",
            "object injection",
        ],
    ),
]

# Stage 2: diff-specific keywords (applied to unified diff text)
_DIFF_KEYWORDS: list[tuple[str, list[str]]] = [
    (
        "sql_injection",
        ["execute(", "cursor.", "parameterize", "bind_param", "%s", "?"],
    ),
    (
        "command_injection",
        [
            "subprocess.run",
            "subprocess.call",
            "subprocess.Popen",
            "os.system",
            "shell=True",
            "shlex.quote",
            "exec(",
            "eval(",
        ],
    ),
    (
        "path_traversal",
        [
            "os.path.join",
            "os.path.realpath",
            "os.path.abspath",
            "../",
            "..\\",
            "path.resolve",
            "normalize",
        ],
    ),
    (
        "buffer_overflow",
        [
            "memcpy",
            "strcpy",
            "strncpy",
            "strcat",
            "sprintf",
            "snprintf",
            "gets(",
            "sizeof(",
            "bounds",
        ],
    ),
    (
        "integer_overflow",
        [
            "size_t",
            "ssize_t",
            "uint32",
            "int32",
            "overflow",
            "UINT_MAX",
            "INT_MAX",
        ],
    ),
    (
        "use_after_free",
        ["free(", "kfree(", "g_free(", "delete ", "release(", "destroy("],
    ),
    (
        "null_dereference",
        [
            "!= NULL",
            "!= nullptr",
            "== NULL",
            "is None",
            "is not None",
            "if not ",
            "null_check",
        ],
    ),
    (
        "auth_bypass",
        [
            "is_authenticated",
            "check_permission",
            "has_perm",
            "require_login",
            "@login_required",
            "authorize",
            "is_admin",
        ],
    ),
    (
        "xss",
        [
            "escape(",
            "sanitize(",
            "innerHTML",
            "dangerouslySetInnerHTML",
            "bleach",
            "html.escape",
            "markupsafe",
            "DOMPurify",
        ],
    ),
    (
        "csrf",
        [
            "csrf_token",
            "csrfmiddleware",
            "X-CSRF",
            "anti_forgery",
            "_token",
        ],
    ),
    (
        "ssrf",
        [
            "urlparse",
            "is_internal",
            "private_ip",
            "127.0.0.1",
            "169.254",
            "localhost",
            "blocklist",
        ],
    ),
    (
        "information_disclosure",
        [
            "redact",
            "mask(",
            "[FILTERED]",
            "strip_credentials",
            "remove_sensitive",
        ],
    ),
    (
        "denial_of_service",
        [
            "timeout",
            "max_size",
            "limit(",
            "rate_limit",
            "throttle",
            "setrecursionlimit",
        ],
    ),
    (
        "deserialization",
        [
            "pickle.loads",
            "yaml.safe_load",
            "json.loads",
            "marshal.loads",
            "untrusted",
        ],
    ),
]

# Generic security signal words for message scoring (add a small bonus)
_SECURITY_SIGNALS: list[str] = [
    "security",
    "vulnerability",
    "exploit",
    "attack",
    "malicious",
    "unsafe",
    "insecure",
    "hardening",
    "sanitize",
    "validate input",
    "fix vuln",
    "patch vuln",
]

# CVE pattern — commits referencing an existing CVE are NOT silent patches.
_CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)

# GitHub API helpers
_GH_API = "https://api.github.com"
_DEFAULT_MAX_COMMITS = 500
_DEFAULT_SINCE_DAYS = 90
_DIFF_FETCH_TIMEOUT = 15  # seconds


def _gh_headers() -> dict[str, str]:
    """Build GitHub API request headers."""
    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _parse_repo(repo: str) -> tuple[str, str]:
    """Parse ``owner/repo`` string.  Returns ``(owner, repo)``."""
    repo = repo.strip().rstrip("/")
    # Handle full URLs
    for prefix in ("https://github.com/", "http://github.com/", "github.com/"):
        if repo.lower().startswith(prefix):
            repo = repo[len(prefix) :]
            break
    parts = repo.split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid repository format: {repo!r}. Expected 'owner/repo'.")
    return parts[0], parts[1].split("/")[0]  # strip trailing paths


def _fetch_commits(
    owner: str,
    repo: str,
    since: str,
    until: str,
    max_commits: int,
) -> list[dict[str, Any]]:
    """Fetch commits from the GitHub API with pagination.

    Returns a list of commit objects (each has ``sha``, ``commit.message``,
    ``commit.author.date``, ``html_url``).
    """
    commits: list[dict[str, Any]] = []
    page = 1
    per_page = min(max_commits, 100)

    while len(commits) < max_commits:
        params: dict[str, Any] = {
            "since": since,
            "until": until,
            "per_page": per_page,
            "page": page,
        }
        resp = requests.get(
            f"{_GH_API}/repos/{owner}/{repo}/commits",
            headers=_gh_headers(),
            params=params,
            timeout=20,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        commits.extend(batch)
        if len(batch) < per_page:
            break
        page += 1

    return commits[:max_commits]


def _fetch_diff(owner: str, repo: str, sha: str) -> str | None:
    """Fetch the unified diff for a single commit.  Returns ``None`` on error."""
    try:
        headers = _gh_headers()
        headers["Accept"] = "application/vnd.github.diff"
        resp = requests.get(
            f"{_GH_API}/repos/{owner}/{repo}/commits/{sha}",
            headers=headers,
            timeout=_DIFF_FETCH_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.text
    except (requests.RequestException, ValueError):
        pass
    return None


def _score_message(message: str) -> tuple[int, str, list[str]]:
    """Score a commit message against stage-1 keyword lists.

    Returns ``(score, best_class, matched_keywords)``.
    A score of 0 means no security signals detected.
    """
    lower = message.lower()
    class_scores: dict[str, int] = {}
    matched: list[str] = []

    for bug_class, keywords in _MSG_KEYWORDS:
        hits = 0
        for kw in keywords:
            if kw in lower:
                hits += 1
                matched.append(kw)
        if hits:
            # Base: 30 points for first hit, +10 per additional
            class_scores[bug_class] = 30 + (hits - 1) * 10

    # Bonus for generic security signals
    signal_bonus = 0
    for sig in _SECURITY_SIGNALS:
        if sig in lower:
            signal_bonus += 5
            matched.append(sig)

    if not class_scores:
        if signal_bonus >= 10:
            # Generic security fix, no specific class
            return min(signal_bonus, 25), "unknown_security", matched
        return 0, "", matched

    best_class = max(class_scores, key=class_scores.get)  # type: ignore[arg-type]
    score = min(class_scores[best_class] + signal_bonus, 70)
    return score, best_class, matched


def _score_diff(diff_text: str, initial_class: str) -> tuple[int, str, list[str]]:
    """Score a unified diff against stage-2 keyword lists.

    Returns ``(adjustment, refined_class, matched_keywords)``.
    The adjustment is added to the message score.
    """
    lower = diff_text.lower()
    class_scores: dict[str, int] = {}
    matched: list[str] = []

    for bug_class, keywords in _DIFF_KEYWORDS:
        hits = 0
        for kw in keywords:
            if kw.lower() in lower:
                hits += 1
                matched.append(kw)
        if hits:
            class_scores[bug_class] = 10 + (hits - 1) * 5

    if not class_scores:
        return 0, initial_class, matched

    # If the initial class is confirmed by diff analysis, boost more
    if initial_class in class_scores:
        return min(class_scores[initial_class] + 10, 30), initial_class, matched

    # If diff suggests a different class, use the stronger signal
    best_diff_class = max(class_scores, key=class_scores.get)  # type: ignore[arg-type]
    if class_scores[best_diff_class] > class_scores.get(initial_class, 0):
        return min(class_scores[best_diff_class], 25), best_diff_class, matched

    return min(class_scores.get(initial_class, 0), 20), initial_class, matched


def _confidence_label(score: int) -> str:
    """Map a 0-100 score to a human-readable label."""
    if score >= 75:
        return "high"
    if score >= 50:
        return "medium"
    if score >= 30:
        return "low"
    return "informational"


def detect_silent_patches(
    repo: str,
    since: str | None = None,
    until: str | None = None,
    max_commits: int = _DEFAULT_MAX_COMMITS,
    fast: bool = False,
) -> dict[str, Any]:
    """Scan a repo for commits that look like silent security patches.

    Parameters
    ----------
    repo:
        ``owner/repo`` string or full GitHub URL.
    since:
        ISO-8601 start date (default: 90 days ago).
    until:
        ISO-8601 end date (default: today).
    max_commits:
        Hard cap on commits fetched via the API.
    fast:
        If ``True``, skip stage-2 diff scoring (faster, less accurate).

    Returns
    -------
    dict with ``repo``, ``scan_window``, ``total_commits``, ``candidates``,
    and ``summary``.
    """
    owner, repo_name = _parse_repo(repo)

    now = datetime.now(tz=timezone.utc)
    if since is None:
        since_dt = now - timedelta(days=_DEFAULT_SINCE_DAYS)
        since = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if until is None:
        until = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    commits = _fetch_commits(owner, repo_name, since, until, max_commits)

    candidates: list[dict[str, Any]] = []

    for commit_obj in commits:
        sha = commit_obj.get("sha", "")
        commit_data = commit_obj.get("commit", {})
        message = commit_data.get("message", "")
        author_date = commit_data.get("author", {}).get("date", "")
        html_url = commit_obj.get("html_url", "")

        # Skip commits that reference a CVE — those are not silent
        if _CVE_PATTERN.search(message):
            continue

        msg_score, bug_class, msg_keywords = _score_message(message)
        if msg_score == 0:
            continue

        diff_adjustment = 0
        diff_keywords: list[str] = []
        refined_class = bug_class

        if not fast:
            diff_text = _fetch_diff(owner, repo_name, sha)
            if diff_text:
                diff_adjustment, refined_class, diff_keywords = _score_diff(diff_text, bug_class)

        final_score = min(msg_score + diff_adjustment, 100)

        candidates.append(
            {
                "sha": sha,
                "short_sha": sha[:8],
                "message": message.split("\n")[0][:200],
                "full_message": message[:500],
                "date": author_date,
                "url": html_url,
                "score": final_score,
                "confidence": _confidence_label(final_score),
                "classification": refined_class,
                "message_keywords": msg_keywords,
                "diff_keywords": diff_keywords,
                "stage2_applied": not fast and diff_adjustment != 0,
            }
        )

    # Sort by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Summary
    class_counts: dict[str, int] = {}
    for c in candidates:
        cls = c["classification"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    confidence_counts: dict[str, int] = {}
    for c in candidates:
        conf = c["confidence"]
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

    return {
        "repo": f"{owner}/{repo_name}",
        "scan_window": {"since": since, "until": until},
        "total_commits_scanned": len(commits),
        "candidates_found": len(candidates),
        "candidates": candidates,
        "summary": {
            "by_classification": class_counts,
            "by_confidence": confidence_counts,
            "highest_score": candidates[0]["score"] if candidates else 0,
            "fast_mode": fast,
        },
    }


# ---------------------------------------------------------------------------
# Strands tool interface
# ---------------------------------------------------------------------------

TOOL_SPEC = {
    "name": "detect_silent_patches",
    "description": (
        "Scans a GitHub repository's recent commit history for security fixes "
        "that were never assigned a CVE (silent patches).  Uses two-stage "
        "heuristic scoring: commit message keywords then diff keywords.  Each "
        "candidate is labelled with one of 14 bug classes (e.g. auth_bypass, "
        "buffer_overflow, use_after_free) and a confidence score (0-100).  "
        "Use for supply-chain risk assessment and identifying untracked fixes."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "GitHub repository (owner/repo or full URL).",
                },
                "since": {
                    "type": "string",
                    "description": ("Start date for commit scan (YYYY-MM-DD). Default: 90 days ago."),
                },
                "until": {
                    "type": "string",
                    "description": ("End date for commit scan (YYYY-MM-DD). Default: today."),
                },
                "max_commits": {
                    "type": "integer",
                    "description": "Hard limit on commits fetched. Default: 500.",
                },
                "fast": {
                    "type": "boolean",
                    "description": ("Skip diff scoring (message keywords only). Default: false."),
                },
            },
            "required": ["repo"],
        }
    },
}


def detect_silent_patches_tool(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands-compatible tool handler."""
    tool_input = tool.get("input", {}) if isinstance(tool, dict) else {}
    repo = tool_input.get("repo", "")
    since = tool_input.get("since")
    until_date = tool_input.get("until")
    max_commits = tool_input.get("max_commits", _DEFAULT_MAX_COMMITS)
    fast = tool_input.get("fast", False)

    if not repo:
        result: ToolResult = {
            "toolUseId": tool.get("toolUseId", ""),
            "status": "error",
            "content": [{"text": "Error: 'repo' parameter is required."}],
        }
        log_tool_output_size("detect_silent_patches", result)
        return result

    try:
        data = detect_silent_patches(
            repo=repo,
            since=since,
            until=until_date,
            max_commits=max_commits,
            fast=fast,
        )
        result = {
            "toolUseId": tool.get("toolUseId", ""),
            "status": "success",
            "content": [{"json": data}],
        }
        log_tool_output_size("detect_silent_patches", result)
        return result
    except ValueError as exc:
        result = {
            "toolUseId": tool.get("toolUseId", ""),
            "status": "error",
            "content": [{"text": f"Error: {exc}"}],
        }
        log_tool_output_size("detect_silent_patches", result)
        return result
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        result = {
            "toolUseId": tool.get("toolUseId", ""),
            "status": "error",
            "content": [
                {"text": (f"GitHub API error (HTTP {status}): {exc}. Check that the repository exists and is public.")}
            ],
        }
        log_tool_output_size("detect_silent_patches", result)
        return result
    except requests.RequestException as exc:
        result = {
            "toolUseId": tool.get("toolUseId", ""),
            "status": "error",
            "content": [{"text": f"Network error: {exc}"}],
        }
        log_tool_output_size("detect_silent_patches", result)
        return result
