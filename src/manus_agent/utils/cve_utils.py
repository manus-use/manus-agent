"""Shared CVE identifier utilities.

This module centralises CVE-ID validation, normalisation, and error-response
construction so that every tool in the codebase uses the same rules and
returns errors in a consistent format.

The canonical regex follows the MITRE CVE specification:
``CVE-YYYY-NNNN+`` where *YYYY* is a four-digit year ≥ 1999 and the
sequence number has **at least four digits** (no upper bound).
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "CVE_RE",
    "is_valid_cve_id",
    "normalize_cve_id",
    "validate_cve_id",
    "cve_validation_error",
    "extract_cve_year",
]

# ---------------------------------------------------------------------------
# Canonical CVE-ID pattern
# ---------------------------------------------------------------------------

#: Compiled regex that matches a well-formed CVE identifier.
#:
#: *  Year must be four digits (≥ 1999 is the MITRE minimum, but the regex
#:    intentionally accepts any ``\d{4}`` to avoid false negatives on
#:    futuristic test data).
#: *  Sequence number: at least four digits, no upper bound.
#: *  Case-insensitive (``cve-2024-1234`` is accepted and normalised).
CVE_RE: re.Pattern[str] = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)

_CVE_YEAR_RE: re.Pattern[str] = re.compile(r"^CVE-(\d{4})-\d{4,}$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def is_valid_cve_id(value: Any) -> bool:
    """Return ``True`` when *value* is a syntactically valid CVE identifier.

    >>> is_valid_cve_id("CVE-2024-3094")
    True
    >>> is_valid_cve_id("not-a-cve")
    False
    >>> is_valid_cve_id(None)
    False
    >>> is_valid_cve_id(12345)
    False
    """
    return isinstance(value, str) and CVE_RE.match(value) is not None


def normalize_cve_id(cve_id: str) -> str:
    """Return *cve_id* upper-cased and stripped.

    Raises :class:`ValueError` when the normalised value is not a valid CVE.

    >>> normalize_cve_id("  cve-2024-3094 ")
    'CVE-2024-3094'
    """
    cleaned = cve_id.strip().upper() if isinstance(cve_id, str) else ""
    if not CVE_RE.match(cleaned):
        raise ValueError(
            f"Invalid CVE ID: {cve_id!r}. Expected format: CVE-YYYY-NNNNN "
            f"(e.g. CVE-2024-3094)."
        )
    return cleaned


def extract_cve_year(cve_id: str) -> int | None:
    """Extract the four-digit year from a CVE identifier.

    Returns ``None`` when *cve_id* does not match the expected format.

    >>> extract_cve_year("CVE-2024-3094")
    2024
    >>> extract_cve_year("bad") is None
    True
    """
    m = _CVE_YEAR_RE.match(cve_id.strip() if isinstance(cve_id, str) else "")
    return int(m.group(1)) if m else None


def validate_cve_id(cve_id: Any) -> str | None:
    """Validate and normalise a CVE identifier, returning ``None`` on success.

    On failure returns a human-readable error message string describing the
    problem.  This is the preferred entry point for tool functions that need
    a quick guard clause::

        if (err := validate_cve_id(cve_id)) is not None:
            return _error_response(tool_use_id, err)

    Returns ``None`` when valid (the walrus-pattern idiom reads naturally:
    *"if there is an error …"*).
    """
    if not isinstance(cve_id, str):
        return "Invalid CVE ID. Must be a non-empty string."
    cleaned = cve_id.strip()
    if not cleaned:
        return "Invalid CVE ID. Must be a non-empty string."
    if not CVE_RE.match(cleaned):
        return f"Invalid CVE ID format: {cve_id!r}. Must match CVE-YYYY-NNNNN (e.g. CVE-2024-3094)."
    return None


# ---------------------------------------------------------------------------
# Error-response builders (Strands ToolResult format)
# ---------------------------------------------------------------------------


def cve_validation_error(
    cve_id: Any,
    tool_use_id: str,
) -> dict[str, Any]:
    """Build a Strands ``ToolResult`` error dict for an invalid CVE identifier.

    This is a convenience wrapper around :func:`validate_cve_id` that returns
    a ready-to-return error response in the Strands tool-result format::

        {
            "toolUseId": "<id>",
            "status": "error",
            "content": [{"text": "<message>"}],
        }

    Returns ``None`` when *cve_id* is valid — use the walrus pattern::

        if (err := cve_validation_error(cve_id, tool_use_id)) is not None:
            return err
    """
    msg = validate_cve_id(cve_id)
    if msg is None:
        return None  # type: ignore[return-value]  # intentional sentinel
    return {
        "toolUseId": tool_use_id,
        "status": "error",
        "content": [{"text": msg}],
    }
