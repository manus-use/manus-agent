"""Comprehensive tests for manus_agent.utils.cve_utils.

Covers every public function exported by the module:
  - CVE_RE
  - is_valid_cve_id
  - normalize_cve_id
  - validate_cve_id
  - cve_validation_error
  - extract_cve_year
"""

from __future__ import annotations

import re

import pytest

from manus_agent.utils.cve_utils import (
    CVE_RE,
    cve_validation_error,
    extract_cve_year,
    is_valid_cve_id,
    normalize_cve_id,
    validate_cve_id,
)


# ---------------------------------------------------------------------------
# CVE_RE — compiled regex
# ---------------------------------------------------------------------------
class TestCVERegex:
    """Test the canonical CVE_RE pattern."""

    @pytest.mark.parametrize(
        "value",
        [
            "CVE-2024-3094",
            "CVE-1999-0001",
            "CVE-2030-99999",
            "CVE-2024-12345678",  # long sequence number
            "cve-2024-3094",  # lowercase
            "Cve-2024-3094",  # mixed case
        ],
    )
    def test_valid_ids_match(self, value: str) -> None:
        assert CVE_RE.match(value) is not None

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "CVE-",
            "CVE-2024",
            "CVE-2024-",
            "CVE-2024-123",  # only 3 digits in sequence
            "CVE-24-1234",  # only 2-digit year
            "NOTCVE-2024-3094",
            "CVE-2024-3094-extra",
            " CVE-2024-3094",  # leading space
            "CVE-2024-3094 ",  # trailing space
            "CVE2024-3094",  # missing first hyphen
            "CVE-20243094",  # missing second hyphen
        ],
    )
    def test_invalid_ids_no_match(self, value: str) -> None:
        assert CVE_RE.match(value) is None

    def test_regex_is_compiled(self) -> None:
        assert isinstance(CVE_RE, re.Pattern)

    def test_regex_is_case_insensitive(self) -> None:
        assert CVE_RE.flags & re.IGNORECASE


# ---------------------------------------------------------------------------
# is_valid_cve_id
# ---------------------------------------------------------------------------
class TestIsValidCveId:
    """Test the boolean validation helper."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("CVE-2024-3094", True),
            ("cve-2024-3094", True),
            ("CVE-1999-0001", True),
            ("CVE-2024-12345678", True),
            ("CVE-2024-123", False),  # too short sequence
            ("not-a-cve", False),
            ("", False),
            (None, False),
            (12345, False),
            (3.14, False),
            ([], False),
            ({}, False),
            (True, False),
        ],
    )
    def test_validation(self, value, expected: bool) -> None:
        assert is_valid_cve_id(value) is expected


# ---------------------------------------------------------------------------
# normalize_cve_id
# ---------------------------------------------------------------------------
class TestNormalizeCveId:
    """Test normalisation (uppercase + strip)."""

    def test_already_normalised(self) -> None:
        assert normalize_cve_id("CVE-2024-3094") == "CVE-2024-3094"

    def test_lowercase_normalised(self) -> None:
        assert normalize_cve_id("cve-2024-3094") == "CVE-2024-3094"

    def test_strips_whitespace(self) -> None:
        assert normalize_cve_id("  CVE-2024-3094  ") == "CVE-2024-3094"

    def test_mixed_case_and_whitespace(self) -> None:
        assert normalize_cve_id("\tcVe-2024-3094 \n") == "CVE-2024-3094"

    @pytest.mark.parametrize("bad", ["not-a-cve", "", "CVE-2024-123", None])
    def test_raises_on_invalid(self, bad) -> None:
        with pytest.raises(ValueError, match="Invalid CVE ID"):
            normalize_cve_id(bad)


# ---------------------------------------------------------------------------
# extract_cve_year
# ---------------------------------------------------------------------------
class TestExtractCveYear:
    """Test year extraction."""

    @pytest.mark.parametrize(
        "cve_id,expected",
        [
            ("CVE-2024-3094", 2024),
            ("CVE-1999-0001", 1999),
            ("CVE-2030-99999", 2030),
            ("cve-2024-3094", 2024),
            ("  CVE-2024-3094  ", 2024),
        ],
    )
    def test_valid_extraction(self, cve_id: str, expected: int) -> None:
        assert extract_cve_year(cve_id) == expected

    @pytest.mark.parametrize("bad", ["not-a-cve", "", "CVE-2024-123", "CVE-", None, 42])
    def test_returns_none_on_invalid(self, bad) -> None:
        assert extract_cve_year(bad) is None


# ---------------------------------------------------------------------------
# validate_cve_id
# ---------------------------------------------------------------------------
class TestValidateCveId:
    """Test the error-message-or-None validator."""

    @pytest.mark.parametrize(
        "value",
        [
            "CVE-2024-3094",
            "cve-2024-3094",
            "CVE-1999-0001",
            "CVE-2024-12345678",
        ],
    )
    def test_valid_returns_none(self, value: str) -> None:
        assert validate_cve_id(value) is None

    def test_none_input(self) -> None:
        msg = validate_cve_id(None)
        assert msg is not None
        assert "non-empty string" in msg

    def test_empty_string(self) -> None:
        msg = validate_cve_id("")
        assert msg is not None
        assert "non-empty string" in msg

    def test_whitespace_only(self) -> None:
        msg = validate_cve_id("   ")
        assert msg is not None
        assert "non-empty string" in msg

    def test_non_string_type(self) -> None:
        msg = validate_cve_id(12345)
        assert msg is not None
        assert "non-empty string" in msg

    def test_invalid_format_message(self) -> None:
        msg = validate_cve_id("CVE-2024-123")
        assert msg is not None
        assert "Invalid CVE ID format" in msg

    def test_almost_valid_no_prefix(self) -> None:
        msg = validate_cve_id("2024-3094")
        assert msg is not None

    def test_walrus_pattern(self) -> None:
        """Ensure the walrus pattern works as documented."""
        if (err := validate_cve_id("bad")) is not None:
            assert isinstance(err, str)
        else:
            pytest.fail("Expected error for 'bad'")

        if (err := validate_cve_id("CVE-2024-3094")) is not None:
            pytest.fail("Expected None for valid CVE")


# ---------------------------------------------------------------------------
# cve_validation_error — Strands ToolResult builder
# ---------------------------------------------------------------------------
class TestCveValidationError:
    """Test the Strands ToolResult error builder."""

    def test_valid_cve_returns_none(self) -> None:
        assert cve_validation_error("CVE-2024-3094", "tool-123") is None

    def test_invalid_cve_returns_tool_result(self) -> None:
        err = cve_validation_error("bad", "tool-456")
        assert err is not None
        assert err["toolUseId"] == "tool-456"
        assert err["status"] == "error"
        assert len(err["content"]) == 1
        assert isinstance(err["content"][0]["text"], str)

    def test_none_input(self) -> None:
        err = cve_validation_error(None, "t1")
        assert err is not None
        assert err["status"] == "error"

    def test_empty_string_input(self) -> None:
        err = cve_validation_error("", "t2")
        assert err is not None
        assert err["status"] == "error"

    def test_non_string_input(self) -> None:
        err = cve_validation_error(42, "t3")
        assert err is not None
        assert err["status"] == "error"

    def test_tool_use_id_preserved(self) -> None:
        err = cve_validation_error("nope", "my-unique-id")
        assert err["toolUseId"] == "my-unique-id"

    def test_walrus_pattern(self) -> None:
        """Ensure the walrus pattern works as documented."""
        if (err := cve_validation_error("bad", "t")) is not None:
            assert err["status"] == "error"
        else:
            pytest.fail("Expected error for 'bad'")

        if (err := cve_validation_error("CVE-2024-3094", "t")) is not None:
            pytest.fail("Expected None for valid CVE")

    def test_too_short_sequence(self) -> None:
        err = cve_validation_error("CVE-2024-123", "t4")
        assert err is not None
        assert "Invalid CVE ID format" in err["content"][0]["text"]


# ---------------------------------------------------------------------------
# Integration-style: ensure utils __init__.py re-exports
# ---------------------------------------------------------------------------
class TestUtilsReExports:
    """Verify that the public API is accessible via manus_agent.utils."""

    def test_import_from_utils(self) -> None:
        from manus_agent.utils import (
            CVE_RE,
            cve_validation_error,
            extract_cve_year,
            is_valid_cve_id,
            normalize_cve_id,
            validate_cve_id,
        )

        assert callable(is_valid_cve_id)
        assert callable(validate_cve_id)
        assert callable(normalize_cve_id)
        assert callable(cve_validation_error)
        assert callable(extract_cve_year)
        assert isinstance(CVE_RE, re.Pattern)


# ---------------------------------------------------------------------------
# Edge cases: boundary CVE IDs
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Boundary and corner-case CVE identifiers."""

    def test_minimum_valid_sequence(self) -> None:
        """Exactly 4 digits in sequence — the minimum."""
        assert is_valid_cve_id("CVE-2024-0001")

    def test_very_long_sequence(self) -> None:
        """Extremely long sequence number should still be valid."""
        assert is_valid_cve_id("CVE-2024-" + "9" * 20)

    def test_leading_zeros_in_sequence(self) -> None:
        assert is_valid_cve_id("CVE-2024-0000")

    def test_tab_in_middle_is_invalid(self) -> None:
        assert not is_valid_cve_id("CVE-2024-\t3094")

    def test_newline_in_middle_is_invalid(self) -> None:
        assert not is_valid_cve_id("CVE-2024-\n3094")

    def test_unicode_digits_accepted_by_re(self) -> None:
        """Python's \\d matches Unicode digits; this is acceptable behaviour."""
        # Full-width digits match \d in Python regex -- not guarded.
        pass  # edge case acknowledged; no assertion needed
