"""
Tool for fetching and summarising the fixing commit diff for a CVE.

Given a CVE identifier, this tool:
1. Looks up the GitHub Security Advisory to find the patching repository / commit.
2. Falls back to searching for references in the NVD advisory (GitHub commit URLs).
3. Fetches the raw unified diff from GitHub's API.
4. Produces a structured summary: which function(s) changed, what class of bug
   was fixed, and the minimal condition required to trigger it.

Only public GitHub repositories (unauthenticated or via GITHUB_TOKEN) are
supported.  Private repositories and non-GitHub hosting return a descriptive
"not_found" result rather than an error.
"""

from __future__ import annotations

import os
import re
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size
from manus_agent.utils.cve_utils import cve_validation_error

# ---------------------------------------------------------------------------
# Bug-class keyword mapping (applied to the unified diff text)
# ---------------------------------------------------------------------------
_BUG_CLASSES: list[tuple[str, list[str]]] = [
    (
        "sql_injection",
        ["sql", "query", "execute", "cursor", "select ", "insert ", "update ", "delete ", "where "],
    ),
    (
        "command_injection",
        ["os.system", "subprocess", "shell=True", "exec(", "eval(", "popen", "execv"],
    ),
    (
        "path_traversal",
        ["../", "..\\", "os.path.join", "realpath", "abspath", "traverse", "chroot"],
    ),
    (
        "buffer_overflow",
        ["memcpy", "strcpy", "strcat", "sprintf", "gets(", "scanf", "overflow", "struct.pack", "ctypes"],
    ),
    (
        "integer_overflow",
        ["integer overflow", "int overflow", "wrap around", "wraparound", "size_t", "ssize_t", "uint"],
    ),
    (
        "use_after_free",
        ["use after free", "use-after-free", "uaf", "free(", "kfree(", "dangling"],
    ),
    (
        "null_dereference",
        ["null check", "nullptr", "null pointer", "is None", "if not ", "== null", "!= null"],
    ),
    (
        "auth_bypass",
        [
            "authentication",
            "authoriz",
            "auth",
            "permission",
            "privilege",
            "access control",
            "bypass",
            "check_permission",
            "is_authenticated",
        ],
    ),
    (
        "deserialization",
        ["deserializ", "pickle", "yaml.load", "json.loads", "unmarshal", "unserializ", "objectinputstream"],
    ),
    (
        "xss",
        ["escape", "sanitize", "sanitise", "html.escape", "encode", "htmlentities", "innerhtml", "xss"],
    ),
    (
        "ssrf",
        ["ssrf", "urlopen", "requests.get", "fetch(", "curl", "allowed_hosts", "internal", "localhost"],
    ),
    (
        "input_validation",
        ["validate", "validation", "sanitize", "sanitise", "allowlist", "whitelist", "blacklist", "regex", "pattern"],
    ),
    (
        "race_condition",
        ["race condition", "toctou", "time-of-check", "mutex", "lock(", "synchronized", "atomic"],
    ),
    (
        "cryptographic",
        ["encrypt", "decrypt", "cipher", "hash", "random", "entropy", "tls", "ssl", "hmac", "signature"],
    ),
    (
        "memory_leak",
        ["memory leak", "leak", "free(", "kfree(", "dealloc", "release("],
    ),
]

# ---------------------------------------------------------------------------
# Helpers for GitHub URL parsing
# ---------------------------------------------------------------------------

_GITHUB_COMMIT_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/commit/(?P<sha>[0-9a-f]{7,40})",
    re.IGNORECASE,
)

_GITHUB_PR_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<pr>\d+)",
    re.IGNORECASE,
)


def _github_headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN", "")
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _fetch_json(url: str, timeout: int = 15) -> dict[str, Any] | list[Any] | None:
    """GET *url* and return parsed JSON, or None on any error."""
    try:
        resp = requests.get(url, headers=_github_headers(), timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _fetch_text(url: str, timeout: int = 20) -> str | None:
    """GET *url* and return response text, or None on any error."""
    try:
        resp = requests.get(url, headers=_github_headers(), timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step 1: find commit candidates
# ---------------------------------------------------------------------------


def _commits_from_ghsa(cve_id: str) -> list[dict[str, str]]:
    """
    Query the GitHub Security Advisory REST API for *cve_id* and collect any
    commit/PR references embedded in the advisory text and references list.
    Returns a list of dicts: {"owner": ..., "repo": ..., "sha": ...}
    """
    url = f"https://api.github.com/advisories?cve_id={cve_id}"
    data = _fetch_json(url)
    if not data or not isinstance(data, list):
        return []

    candidates: list[dict[str, str]] = []
    text_blob = ""

    advisory = data[0]
    # Collect all text fields
    for field in ("description", "summary"):
        text_blob += advisory.get(field, "") + "\n"
    for ref in advisory.get("references", []):
        text_blob += ref + "\n"

    # Also check vulnerabilities → patched_versions (repos are in identifiers / cwes)
    for vuln in advisory.get("vulnerabilities", []):
        pkg = vuln.get("package", {})
        ecosystem = pkg.get("ecosystem", "")
        name = pkg.get("name", "")
        if "/" in name and ecosystem.lower() in ("github_actions", "other", ""):
            # name may be "owner/repo"
            parts = name.split("/", 1)
            if len(parts) == 2:
                text_blob += f"https://github.com/{parts[0]}/{parts[1]}/\n"

    candidates.extend(_extract_commits_from_text(text_blob))
    return candidates


def _commits_from_nvd(cve_id: str) -> list[dict[str, str]]:
    """
    Query the NVD CVE API for *cve_id* and mine reference URLs for GitHub
    commit/PR links.
    """
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id.upper()}"
    data = _fetch_json(url, timeout=15)
    if not data or not isinstance(data, dict):
        return []

    text_blob = ""
    for vuln in data.get("vulnerabilities", []):
        cve_obj = vuln.get("cve", {})
        for ref in cve_obj.get("references", []):
            text_blob += ref.get("url", "") + "\n"
        # descriptions
        for desc in cve_obj.get("descriptions", []):
            text_blob += desc.get("value", "") + "\n"

    return _extract_commits_from_text(text_blob)


def _extract_commits_from_text(text: str) -> list[dict[str, str]]:
    """Return unique commit dicts extracted from *text* (commit and PR URLs)."""
    seen: set[str] = set()
    results: list[dict[str, str]] = []

    for m in _GITHUB_COMMIT_RE.finditer(text):
        key = f"{m['owner']}/{m['repo']}@{m['sha']}"
        if key not in seen:
            seen.add(key)
            results.append({"owner": m["owner"], "repo": m["repo"], "sha": m["sha"]})

    # For PR refs, resolve to the merge commit
    for m in _GITHUB_PR_RE.finditer(text):
        pr_url = f"https://api.github.com/repos/{m['owner']}/{m['repo']}/pulls/{m['pr']}"
        pr_data = _fetch_json(pr_url)
        if pr_data and isinstance(pr_data, dict):
            sha = (pr_data.get("merge_commit_sha") or "")[:40]
            if sha:
                key = f"{m['owner']}/{m['repo']}@{sha}"
                if key not in seen:
                    seen.add(key)
                    results.append({"owner": m["owner"], "repo": m["repo"], "sha": sha})

    return results


# ---------------------------------------------------------------------------
# Step 2: fetch and analyse the diff
# ---------------------------------------------------------------------------


def _fetch_diff(owner: str, repo: str, sha: str) -> str | None:
    """
    Fetch the unified diff for *sha* using the GitHub commits API
    (Accept: application/vnd.github.diff).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    headers = _github_headers()
    headers["Accept"] = "application/vnd.github.diff"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def _summarise_diff(diff: str, owner: str, repo: str, sha: str) -> dict[str, Any]:
    """
    Analyse *diff* and return a structured summary dict.
    """
    lines = diff.splitlines()

    # --- Files changed ---
    files_changed: list[str] = []
    for line in lines:
        if line.startswith("diff --git "):
            # diff --git a/path b/path  →  strip "b/" prefix
            parts = line.split(" b/", 1)
            if len(parts) == 2:
                files_changed.append(parts[1].strip())

    # --- Functions / methods changed (hunk headers: @@ ... @@ funcname) ---
    func_re = re.compile(r"^@@[^@]+@@\s+(.+)$")
    functions_touched: list[str] = []
    seen_funcs: set[str] = set()
    for line in lines:
        m = func_re.match(line)
        if m:
            func_hint = m.group(1).strip()
            # Truncate to the first paren or brace
            func_hint = re.split(r"[({]", func_hint)[0].strip()
            if func_hint and func_hint not in seen_funcs:
                seen_funcs.add(func_hint)
                functions_touched.append(func_hint)

    # --- Added / removed line counts ---
    added_lines = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
    removed_lines = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))

    # --- Bug class detection (scan the diff text) ---
    diff_lower = diff.lower()
    matched_classes: list[str] = []
    for cls_name, keywords in _BUG_CLASSES:
        if any(kw in diff_lower for kw in keywords):
            matched_classes.append(cls_name)

    # --- Infer primary bug class ---
    # Prefer the class whose keywords have the most hits in the diff
    def _keyword_hits(keywords: list[str]) -> int:
        return sum(diff_lower.count(kw) for kw in keywords)

    if matched_classes:
        primary_bug_class = max(
            matched_classes,
            key=lambda c: _keyword_hits(dict(_BUG_CLASSES)[c]),
        )
    else:
        primary_bug_class = "unknown"

    # --- Extract a "minimal reproduction condition" hint ---
    # Look at added lines around bounds / condition checks
    repro_hint_lines: list[str] = []
    for line in lines:
        if not line.startswith("+") or line.startswith("+++"):
            continue
        stripped = line[1:].strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if any(
            kw in lower
            for kw in [
                "if ",
                "raise ",
                "throw ",
                "assert ",
                "validate",
                "check",
                "verify",
                "limit",
                "max",
                "min",
                "length",
                "size",
                "bound",
                "sanitize",
                "sanitise",
                "escape",
                "allow",
                "deny",
                "block",
            ]
        ):
            repro_hint_lines.append(stripped)

    # Keep top-5 most informative lines (longest, as a heuristic)
    repro_hints = sorted(set(repro_hint_lines), key=len, reverse=True)[:5]

    # --- Build commit URL ---
    commit_url = f"https://github.com/{owner}/{repo}/commit/{sha}"

    return {
        "owner": owner,
        "repo": repo,
        "sha": sha[:12],
        "commit_url": commit_url,
        "files_changed": files_changed[:20],  # cap at 20
        "functions_touched": functions_touched[:15],
        "added_lines": added_lines,
        "removed_lines": removed_lines,
        "matched_bug_classes": matched_classes,
        "primary_bug_class": primary_bug_class,
        "reproduction_condition_hints": repro_hints,
    }


# ---------------------------------------------------------------------------
# Public helper (used by CLI)
# ---------------------------------------------------------------------------


def fetch_and_summarise(cve_id: str) -> dict[str, Any]:
    """
    Top-level helper: find the fixing commit(s) for *cve_id* and return a
    list of diff summaries.  Returns a dict with keys:
        cve_id, commit_summaries, not_found (bool), message (str)
    """
    cve_upper = cve_id.upper()

    # Collect commit candidates from GHSA then NVD
    candidates = _commits_from_ghsa(cve_upper)
    if not candidates:
        candidates = _commits_from_nvd(cve_upper)

    if not candidates:
        return {
            "cve_id": cve_upper,
            "not_found": True,
            "message": (
                f"No GitHub fixing-commit references found for {cve_upper} "
                "in the GitHub Security Advisory database or NVD reference list. "
                "The fix may be in a private repository, a non-GitHub host, or not yet linked."
            ),
            "commit_summaries": [],
        }

    summaries: list[dict[str, Any]] = []
    for c in candidates[:3]:  # analyse at most 3 commits
        diff = _fetch_diff(c["owner"], c["repo"], c["sha"])
        if diff:
            summaries.append(_summarise_diff(diff, c["owner"], c["repo"], c["sha"]))

    if not summaries:
        return {
            "cve_id": cve_upper,
            "not_found": True,
            "message": (
                f"Found commit reference(s) for {cve_upper} but could not fetch the diff. "
                "The repository may be private or the commit may have been force-pushed away."
            ),
            "commit_summaries": [],
        }

    return {
        "cve_id": cve_upper,
        "not_found": False,
        "message": f"Found {len(summaries)} fixing commit(s) for {cve_upper}.",
        "commit_summaries": summaries,
    }


# ---------------------------------------------------------------------------
# Strands tool spec + handler
# ---------------------------------------------------------------------------

TOOL_SPEC = {
    "name": "get_patch_diff",
    "description": (
        "Finds the fixing commit(s) for a CVE on GitHub (via the GitHub Security Advisory "
        "database and NVD reference links), fetches the raw unified diff, and returns a "
        "structured summary: which files and functions were changed, how many lines were "
        "added/removed, what class of bug was fixed (e.g. sql_injection, auth_bypass, "
        "buffer_overflow), and hints about the minimal condition required to trigger the "
        "vulnerability. Use this after get_nvd_data to understand the mechanics of the fix "
        "without having to read the raw diff yourself."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier to look up (e.g., 'CVE-2024-3094').",
                },
            },
            "required": ["cve_id"],
        }
    },
}


def get_patch_diff(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Fetch the fixing-commit diff for a CVE and return a structured summary."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id", "")

    if (err := cve_validation_error(cve_id, tool_use_id)) is not None:
        log_tool_output_size("get_patch_diff", err)
        return err

    payload = fetch_and_summarise(cve_id)

    if payload["not_found"]:
        result = {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [
                {"text": payload["message"]},
                {"json": payload},
            ],
        }
        log_tool_output_size("get_patch_diff", result)
        return result

    # Build a human-readable summary for the first commit
    lines: list[str] = [payload["message"]]
    for idx, s in enumerate(payload["commit_summaries"], 1):
        lines.append(f"\nCommit {idx}: {s['commit_url']}")
        lines.append(f"  Files changed    : {', '.join(s['files_changed'][:5]) or 'N/A'}")
        if s["functions_touched"]:
            lines.append(f"  Functions touched: {', '.join(s['functions_touched'][:5])}")
        lines.append(f"  Lines +{s['added_lines']} / -{s['removed_lines']}")
        lines.append(f"  Primary bug class: {s['primary_bug_class']}")
        if s["matched_bug_classes"]:
            lines.append(f"  All bug classes  : {', '.join(s['matched_bug_classes'])}")
        if s["reproduction_condition_hints"]:
            lines.append("  Reproduction hints:")
            for hint in s["reproduction_condition_hints"]:
                lines.append(f"    • {hint}")

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [
            {"text": "\n".join(lines)},
            {"json": payload},
        ],
    }
    log_tool_output_size("get_patch_diff", result)
    return result
