"""Comprehensive test suite for the `manus-agent compare` CLI subcommand.

Tests cover:
- _build_compare_parser() — argparse setup, positional args, options, help text
- _run_compare() — CVE-ID validation, import error handling, text/JSON output,
  concurrent profile fetching, error handling, edge cases
- main() dispatch — routing from main entry point to _run_compare
- Integration with compare_cves module helpers (_build_comparison, _render_text, etc.)

All HTTP calls are fully mocked — no real network access.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures — reusable mock data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_profile_a():
    """A fully-populated CVE profile dict for CVE-2024-3094."""
    return {
        "cve_id": "CVE-2024-3094",
        "nvd_error": None,
        "epss_error": None,
        "cvss": {
            "version": "3.1",
            "score": 10.0,
            "severity": "CRITICAL",
            "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
            "attack_vector": "NETWORK",
            "privileges_required": "NONE",
            "user_interaction": "NONE",
        },
        "epss": {
            "epss": "0.97565",
            "percentile": "0.99998",
        },
        "cwe": ["CWE-506"],
        "affected": "Tukaani Project / Xz",
        "published": "2024-03-29",
        "description": "Malicious code in xz/liblzma leading to SSH auth bypass via systemd...",
    }


@pytest.fixture
def sample_profile_b():
    """A fully-populated CVE profile dict for CVE-2021-44228."""
    return {
        "cve_id": "CVE-2021-44228",
        "nvd_error": None,
        "epss_error": None,
        "cvss": {
            "version": "3.1",
            "score": 10.0,
            "severity": "CRITICAL",
            "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
            "attack_vector": "NETWORK",
            "privileges_required": "NONE",
            "user_interaction": "NONE",
        },
        "epss": {
            "epss": "0.97541",
            "percentile": "0.99997",
        },
        "cwe": ["CWE-917", "CWE-502"],
        "affected": "Apache / Log4j",
        "published": "2021-12-10",
        "description": "Apache Log4j2 JNDI features used in configuration, messages, parameters...",
    }


@pytest.fixture
def sample_kev_a():
    """KEV entry for CVE-2024-3094 (in KEV)."""
    return {
        "in_kev": True,
        "date_added": "2024-03-29",
        "vendor_project": "Tukaani Project",
        "product": "XZ Utils",
        "required_action": "Apply mitigations per vendor.",
        "due_date": "2024-04-19",
    }


@pytest.fixture
def sample_kev_b():
    """KEV entry for CVE-2021-44228 (in KEV)."""
    return {
        "in_kev": True,
        "date_added": "2021-12-10",
        "vendor_project": "Apache",
        "product": "Log4j",
        "required_action": "For all affected software, apply patches.",
        "due_date": "2021-12-24",
    }


@pytest.fixture
def sample_kev_not_in_kev():
    """KEV response for a CVE NOT in KEV."""
    return {"in_kev": False}


@pytest.fixture
def sample_comparison(sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
    """A complete comparison dict."""
    from manus_agent.tools.compare_cves import _build_comparison

    return _build_comparison(sample_profile_a, sample_kev_a, sample_profile_b, sample_kev_b)


# ---------------------------------------------------------------------------
# TestBuildCompareParser — argparse setup
# ---------------------------------------------------------------------------


class TestBuildCompareParser:
    """Tests for _build_compare_parser()."""

    def test_parser_prog_string(self):
        from manus_agent.cli import _build_compare_parser

        parser = _build_compare_parser()
        assert parser.prog == "manus-agent compare"

    def test_parser_has_two_positional_args(self):
        from manus_agent.cli import _build_compare_parser

        parser = _build_compare_parser()
        # Should accept two positional CVE-IDs
        args = parser.parse_args(["CVE-2024-3094", "CVE-2021-44228"])
        assert args.cve_id_a == "CVE-2024-3094"
        assert args.cve_id_b == "CVE-2021-44228"

    def test_parser_requires_two_cve_ids(self):
        from manus_agent.cli import _build_compare_parser

        parser = _build_compare_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["CVE-2024-3094"])  # Only one CVE

    def test_parser_output_default_is_text(self):
        from manus_agent.cli import _build_compare_parser

        parser = _build_compare_parser()
        args = parser.parse_args(["CVE-2024-3094", "CVE-2021-44228"])
        assert args.output == "text"

    def test_parser_output_json(self):
        from manus_agent.cli import _build_compare_parser

        parser = _build_compare_parser()
        args = parser.parse_args(["CVE-2024-3094", "CVE-2021-44228", "--output", "json"])
        assert args.output == "json"

    def test_parser_output_invalid_choice_exits(self):
        from manus_agent.cli import _build_compare_parser

        parser = _build_compare_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["CVE-2024-3094", "CVE-2021-44228", "--output", "xml"])

    def test_parser_description_mentions_compare(self):
        from manus_agent.cli import _build_compare_parser

        parser = _build_compare_parser()
        assert "Compare" in parser.description or "compare" in parser.description.lower()

    def test_parser_description_mentions_prioritisation(self):
        from manus_agent.cli import _build_compare_parser

        parser = _build_compare_parser()
        assert "prioriti" in parser.description.lower()

    def test_parser_help_exits_zero(self):
        from manus_agent.cli import _build_compare_parser

        parser = _build_compare_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_parser_no_extra_arguments_after_two_cve_ids(self):
        from manus_agent.cli import _build_compare_parser

        parser = _build_compare_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["CVE-2024-3094", "CVE-2021-44228", "CVE-2023-99999"])


# ---------------------------------------------------------------------------
# TestRunCompareValidation — input validation
# ---------------------------------------------------------------------------


class TestRunCompareValidation:
    """Tests for _run_compare() input validation."""

    def test_invalid_cve_id_a_returns_nonzero(self, capsys):
        from manus_agent.cli import _run_compare

        rc = _run_compare(["NOTACVE", "CVE-2021-44228"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "Invalid CVE ID" in captured.err
        assert "NOTACVE" in captured.err

    def test_invalid_cve_id_b_returns_nonzero(self, capsys):
        from manus_agent.cli import _run_compare

        rc = _run_compare(["CVE-2024-3094", "BADID"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "Invalid CVE ID" in captured.err
        assert "BADID" in captured.err

    def test_both_invalid_cve_ids_reports_first_invalid(self, capsys):
        from manus_agent.cli import _run_compare

        rc = _run_compare(["INVALID1", "INVALID2"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "INVALID1" in captured.err

    def test_whitespace_stripped_from_cve_ids(self):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile") as mock_profile, \
             patch("manus_agent.tools.compare_cves._fetch_kev") as mock_kev, \
             patch("manus_agent.tools.compare_cves._build_comparison") as mock_comp, \
             patch("manus_agent.tools.compare_cves._render_text", return_value="rendered"):
            mock_profile.return_value = {"cve_id": "CVE-2024-3094"}
            mock_kev.return_value = {"in_kev": False}
            mock_comp.return_value = {"recommendation": "test"}

            # Does not raise - whitespace is handled
            _run_compare(["  CVE-2024-3094  ", "  CVE-2021-44228  "])

    def test_lowercase_cve_prefix_still_validated(self, capsys):
        """CVE IDs with lowercase 'cve-' should still pass validation."""
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile") as mock_profile, \
             patch("manus_agent.tools.compare_cves._fetch_kev") as mock_kev, \
             patch("manus_agent.tools.compare_cves._build_comparison") as mock_comp, \
             patch("manus_agent.tools.compare_cves._render_text", return_value="rendered"):
            mock_profile.return_value = {"cve_id": "CVE-2024-3094"}
            mock_kev.return_value = {"in_kev": False}
            mock_comp.return_value = {"recommendation": "test"}

            # The code does cid.upper().startswith("CVE-")
            # so "cve-2024-3094" should pass validation.
            rc = _run_compare(["cve-2024-3094", "cve-2021-44228"])
            assert rc == 0


# ---------------------------------------------------------------------------
# TestRunCompareImportError — dependency failures
# ---------------------------------------------------------------------------


class TestRunCompareImportError:
    """Tests for _run_compare() when compare_cves imports fail."""

    def test_import_error_returns_1(self, capsys):
        from manus_agent.cli import _run_compare

        # Remove cached module so next import attempt goes through __import__
        import importlib
        cached = sys.modules.pop("manus_agent.tools.compare_cves", None)
        try:
            with patch.dict(sys.modules, {"manus_agent.tools.compare_cves": None}):
                rc = _run_compare(["CVE-2024-3094", "CVE-2021-44228"])
                assert rc == 1
                captured = capsys.readouterr()
                assert "missing dependencies" in captured.err or "error" in captured.err
        finally:
            # Restore the module
            if cached is not None:
                sys.modules["manus_agent.tools.compare_cves"] = cached


# ---------------------------------------------------------------------------
# TestRunCompareTextOutput — text output format
# ---------------------------------------------------------------------------


class TestRunCompareTextOutput:
    """Tests for _run_compare() text output rendering."""

    def test_text_output_calls_render_text(self, capsys, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[sample_profile_a, sample_profile_b]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_a, sample_kev_b]):
            rc = _run_compare(["CVE-2024-3094", "CVE-2021-44228"])

        assert rc == 0
        captured = capsys.readouterr()
        # Text output should contain both CVE IDs
        assert "CVE-2024-3094" in captured.out
        assert "CVE-2021-44228" in captured.out

    def test_text_output_contains_recommendation(self, capsys, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[sample_profile_a, sample_profile_b]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_a, sample_kev_b]):
            rc = _run_compare(["CVE-2024-3094", "CVE-2021-44228"])

        assert rc == 0
        captured = capsys.readouterr()
        assert "RECOMMENDATION" in captured.out

    def test_text_output_contains_dimension_headers(self, capsys, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_not_in_kev):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[sample_profile_a, sample_profile_b]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_a, sample_kev_not_in_kev]):
            rc = _run_compare(["CVE-2024-3094", "CVE-2021-44228"])

        assert rc == 0
        captured = capsys.readouterr()
        assert "DIMENSION" in captured.out

    def test_text_output_shows_cvss_scores(self, capsys, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[sample_profile_a, sample_profile_b]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_a, sample_kev_b]):
            rc = _run_compare(["CVE-2024-3094", "CVE-2021-44228"])

        assert rc == 0
        captured = capsys.readouterr()
        assert "10.0" in captured.out or "10" in captured.out

    def test_text_output_shows_kev_status(self, capsys, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_not_in_kev):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[sample_profile_a, sample_profile_b]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_a, sample_kev_not_in_kev]):
            rc = _run_compare(["CVE-2024-3094", "CVE-2021-44228"])

        assert rc == 0
        captured = capsys.readouterr()
        assert "YES" in captured.out
        assert "No" in captured.out

    def test_text_output_shows_epss(self, capsys, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[sample_profile_a, sample_profile_b]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_a, sample_kev_b]):
            rc = _run_compare(["CVE-2024-3094", "CVE-2021-44228"])

        assert rc == 0
        captured = capsys.readouterr()
        # Should show EPSS score somewhere
        assert "0.97" in captured.out


# ---------------------------------------------------------------------------
# TestRunCompareJsonOutput — JSON output format
# ---------------------------------------------------------------------------


class TestRunCompareJsonOutput:
    """Tests for _run_compare() JSON output rendering."""

    def test_json_output_is_valid_json(self, capsys, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[sample_profile_a, sample_profile_b]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_a, sample_kev_b]):
            rc = _run_compare(["CVE-2024-3094", "CVE-2021-44228", "--output", "json"])

        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, dict)

    def test_json_output_has_cve_a_key(self, capsys, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[sample_profile_a, sample_profile_b]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_a, sample_kev_b]):
            rc = _run_compare(["CVE-2024-3094", "CVE-2021-44228", "--output", "json"])

        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "cve_a" in data
        assert data["cve_a"]["cve_id"] == "CVE-2024-3094"

    def test_json_output_has_cve_b_key(self, capsys, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[sample_profile_a, sample_profile_b]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_a, sample_kev_b]):
            rc = _run_compare(["CVE-2024-3094", "CVE-2021-44228", "--output", "json"])

        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "cve_b" in data
        assert data["cve_b"]["cve_id"] == "CVE-2021-44228"

    def test_json_output_has_priority_scores(self, capsys, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[sample_profile_a, sample_profile_b]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_a, sample_kev_b]):
            rc = _run_compare(["CVE-2024-3094", "CVE-2021-44228", "--output", "json"])

        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "priority_score_a" in data
        assert "priority_score_b" in data
        assert isinstance(data["priority_score_a"], (int, float))
        assert isinstance(data["priority_score_b"], (int, float))

    def test_json_output_has_higher_priority(self, capsys, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[sample_profile_a, sample_profile_b]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_a, sample_kev_b]):
            rc = _run_compare(["CVE-2024-3094", "CVE-2021-44228", "--output", "json"])

        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "higher_priority" in data

    def test_json_output_has_recommendation(self, capsys, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[sample_profile_a, sample_profile_b]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_a, sample_kev_b]):
            rc = _run_compare(["CVE-2024-3094", "CVE-2021-44228", "--output", "json"])

        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "recommendation" in data
        assert isinstance(data["recommendation"], str)

    def test_json_output_has_confidence(self, capsys, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[sample_profile_a, sample_profile_b]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_a, sample_kev_b]):
            rc = _run_compare(["CVE-2024-3094", "CVE-2021-44228", "--output", "json"])

        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "confidence" in data
        assert data["confidence"] in ("strong", "moderate", "weak", "tie")

    def test_json_output_has_winner_reasons(self, capsys, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[sample_profile_a, sample_profile_b]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_a, sample_kev_b]):
            rc = _run_compare(["CVE-2024-3094", "CVE-2021-44228", "--output", "json"])

        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "winner_reasons" in data
        assert isinstance(data["winner_reasons"], list)


# ---------------------------------------------------------------------------
# TestRunCompareConcurrency — parallel profile fetching
# ---------------------------------------------------------------------------


class TestRunCompareConcurrency:
    """Tests for _run_compare() concurrent ThreadPoolExecutor usage."""

    def test_both_profiles_fetched_concurrently(self, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile") as mock_profile, \
             patch("manus_agent.tools.compare_cves._fetch_kev") as mock_kev, \
             patch("manus_agent.tools.compare_cves._build_comparison") as mock_comp, \
             patch("manus_agent.tools.compare_cves._render_text", return_value="text"):
            mock_profile.side_effect = [sample_profile_a, sample_profile_b]
            mock_kev.side_effect = [sample_kev_a, sample_kev_b]
            mock_comp.return_value = {"recommendation": "x"}

            _run_compare(["CVE-2024-3094", "CVE-2021-44228"])

            assert mock_profile.call_count == 2
            assert mock_kev.call_count == 2

    def test_profile_called_with_correct_cve_ids(self, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile") as mock_profile, \
             patch("manus_agent.tools.compare_cves._fetch_kev") as mock_kev, \
             patch("manus_agent.tools.compare_cves._build_comparison") as mock_comp, \
             patch("manus_agent.tools.compare_cves._render_text", return_value="text"):
            mock_profile.side_effect = [sample_profile_a, sample_profile_b]
            mock_kev.side_effect = [sample_kev_a, sample_kev_b]
            mock_comp.return_value = {"recommendation": "x"}

            _run_compare(["CVE-2024-3094", "CVE-2021-44228"])

            # Both CVE IDs should have been passed (order may differ due to threading)
            profile_args = [call.args[0] for call in mock_profile.call_args_list]
            assert "CVE-2024-3094" in profile_args
            assert "CVE-2021-44228" in profile_args

    def test_kev_called_with_correct_cve_ids(self, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.cli import _run_compare

        with patch("manus_agent.tools.compare_cves._build_cve_profile") as mock_profile, \
             patch("manus_agent.tools.compare_cves._fetch_kev") as mock_kev, \
             patch("manus_agent.tools.compare_cves._build_comparison") as mock_comp, \
             patch("manus_agent.tools.compare_cves._render_text", return_value="text"):
            mock_profile.side_effect = [sample_profile_a, sample_profile_b]
            mock_kev.side_effect = [sample_kev_a, sample_kev_b]
            mock_comp.return_value = {"recommendation": "x"}

            _run_compare(["CVE-2024-3094", "CVE-2021-44228"])

            kev_args = [call.args[0] for call in mock_kev.call_args_list]
            assert "CVE-2024-3094" in kev_args
            assert "CVE-2021-44228" in kev_args


# ---------------------------------------------------------------------------
# TestRunComparePrioritisation — scoring and winner logic
# ---------------------------------------------------------------------------


class TestRunComparePrioritisation:
    """Tests for the prioritisation logic via _run_compare()."""

    def test_kev_cve_wins_over_non_kev(self, capsys, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_not_in_kev):
        from manus_agent.cli import _run_compare

        # Profile A is in KEV, Profile B is not
        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[sample_profile_a, sample_profile_b]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_a, sample_kev_not_in_kev]):
            rc = _run_compare(["CVE-2024-3094", "CVE-2021-44228", "--output", "json"])

        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["priority_score_a"] > data["priority_score_b"]
        assert data["higher_priority"] == "CVE-2024-3094"

    def test_tie_when_scores_equal(self, capsys, sample_profile_a, sample_kev_a):
        """When both CVEs have identical profiles and KEV status, it's a tie."""
        from manus_agent.cli import _run_compare

        # Use the same profile for both
        import copy
        profile_b = copy.deepcopy(sample_profile_a)
        profile_b["cve_id"] = "CVE-2021-44228"

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[sample_profile_a, profile_b]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_a, sample_kev_a]):
            rc = _run_compare(["CVE-2024-3094", "CVE-2021-44228", "--output", "json"])

        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["higher_priority"] == "tie"
        assert data["confidence"] == "tie"

    def test_high_epss_contributes_to_score(self, capsys):
        """A CVE with high EPSS but low CVSS still scores well."""
        from manus_agent.cli import _run_compare

        profile_high_epss = {
            "cve_id": "CVE-2024-0001",
            "nvd_error": None,
            "epss_error": None,
            "cvss": {"version": "3.1", "score": 5.0, "severity": "MEDIUM",
                     "vector": "X", "attack_vector": "LOCAL",
                     "privileges_required": "HIGH", "user_interaction": "REQUIRED"},
            "epss": {"epss": "0.85", "percentile": "0.99"},
            "cwe": ["CWE-79"],
            "affected": "Vendor / Product",
            "published": "2024-01-01",
            "description": "Test CVE with high EPSS.",
        }
        profile_low_epss = {
            "cve_id": "CVE-2024-0002",
            "nvd_error": None,
            "epss_error": None,
            "cvss": {"version": "3.1", "score": 5.0, "severity": "MEDIUM",
                     "vector": "X", "attack_vector": "LOCAL",
                     "privileges_required": "HIGH", "user_interaction": "REQUIRED"},
            "epss": {"epss": "0.01", "percentile": "0.10"},
            "cwe": ["CWE-79"],
            "affected": "Vendor / Product",
            "published": "2024-01-01",
            "description": "Test CVE with low EPSS.",
        }
        no_kev = {"in_kev": False}

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[profile_high_epss, profile_low_epss]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[no_kev, no_kev]):
            rc = _run_compare(["CVE-2024-0001", "CVE-2024-0002", "--output", "json"])

        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["priority_score_a"] > data["priority_score_b"]
        assert data["higher_priority"] == "CVE-2024-0001"

    def test_strong_confidence_on_large_margin(self, capsys, sample_profile_a, sample_kev_a, sample_kev_not_in_kev):
        """Large score margin (>=10) produces 'strong' confidence."""
        from manus_agent.cli import _run_compare

        # Profile B: low CVSS, no EPSS, no KEV, no network — should score very low
        profile_weak = {
            "cve_id": "CVE-2024-9999",
            "nvd_error": None,
            "epss_error": None,
            "cvss": {"version": "3.1", "score": 2.0, "severity": "LOW",
                     "vector": "X", "attack_vector": "LOCAL",
                     "privileges_required": "HIGH", "user_interaction": "REQUIRED"},
            "epss": {"epss": "0.001", "percentile": "0.01"},
            "cwe": [],
            "affected": "Unknown",
            "published": "2024-01-01",
            "description": "Trivial vuln.",
        }

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[sample_profile_a, profile_weak]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_a, sample_kev_not_in_kev]):
            rc = _run_compare(["CVE-2024-3094", "CVE-2024-9999", "--output", "json"])

        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["confidence"] == "strong"


# ---------------------------------------------------------------------------
# TestRunCompareEdgeCases — error paths and edge conditions
# ---------------------------------------------------------------------------


class TestRunCompareEdgeCases:
    """Tests for edge cases in _run_compare()."""

    def test_nvd_error_in_profile_still_succeeds(self, capsys, sample_kev_not_in_kev):
        """Even if NVD fetch fails, the compare should still work (graceful degradation)."""
        from manus_agent.cli import _run_compare

        profile_with_nvd_error = {
            "cve_id": "CVE-2024-0001",
            "nvd_error": "NVD request failed: timeout",
            "epss_error": None,
            "cvss": {"version": None, "score": None, "severity": None,
                     "vector": None, "attack_vector": None,
                     "privileges_required": None, "user_interaction": None},
            "epss": {"epss": "0.5", "percentile": "0.90"},
            "cwe": [],
            "affected": "Unknown",
            "published": "Unknown",
            "description": "",
        }
        profile_normal = {
            "cve_id": "CVE-2024-0002",
            "nvd_error": None,
            "epss_error": None,
            "cvss": {"version": "3.1", "score": 7.5, "severity": "HIGH",
                     "vector": "X", "attack_vector": "NETWORK",
                     "privileges_required": "NONE", "user_interaction": "NONE"},
            "epss": {"epss": "0.3", "percentile": "0.80"},
            "cwe": ["CWE-89"],
            "affected": "Vendor / Product",
            "published": "2024-06-01",
            "description": "SQL injection.",
        }

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[profile_with_nvd_error, profile_normal]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_not_in_kev, sample_kev_not_in_kev]):
            rc = _run_compare(["CVE-2024-0001", "CVE-2024-0002"])

        assert rc == 0

    def test_epss_error_in_profile_still_succeeds(self, capsys, sample_kev_not_in_kev):
        """Even if EPSS fetch fails, the compare should still work."""
        from manus_agent.cli import _run_compare

        profile_with_epss_error = {
            "cve_id": "CVE-2024-0001",
            "nvd_error": None,
            "epss_error": "EPSS request failed: timeout",
            "cvss": {"version": "3.1", "score": 9.8, "severity": "CRITICAL",
                     "vector": "X", "attack_vector": "NETWORK",
                     "privileges_required": "NONE", "user_interaction": "NONE"},
            "epss": {},
            "cwe": ["CWE-78"],
            "affected": "Vendor / Product",
            "published": "2024-01-15",
            "description": "Command injection.",
        }
        profile_normal = {
            "cve_id": "CVE-2024-0002",
            "nvd_error": None,
            "epss_error": None,
            "cvss": {"version": "3.1", "score": 4.0, "severity": "MEDIUM",
                     "vector": "X", "attack_vector": "LOCAL",
                     "privileges_required": "HIGH", "user_interaction": "REQUIRED"},
            "epss": {"epss": "0.05", "percentile": "0.50"},
            "cwe": ["CWE-200"],
            "affected": "Vendor / Product",
            "published": "2024-02-01",
            "description": "Info disclosure.",
        }

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[profile_with_epss_error, profile_normal]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_not_in_kev, sample_kev_not_in_kev]):
            rc = _run_compare(["CVE-2024-0001", "CVE-2024-0002"])

        assert rc == 0

    def test_empty_cwe_handled_gracefully(self, capsys, sample_kev_not_in_kev):
        """Profile with no CWE entries should not crash."""
        from manus_agent.cli import _run_compare

        profile = {
            "cve_id": "CVE-2024-0001",
            "nvd_error": None,
            "epss_error": None,
            "cvss": {"version": "3.1", "score": 7.0, "severity": "HIGH",
                     "vector": "X", "attack_vector": "NETWORK",
                     "privileges_required": "LOW", "user_interaction": "NONE"},
            "epss": {"epss": "0.2", "percentile": "0.70"},
            "cwe": [],
            "affected": "Unknown",
            "published": "2024-03-01",
            "description": "No CWE assigned.",
        }

        with patch("manus_agent.tools.compare_cves._build_cve_profile", side_effect=[profile, profile]), \
             patch("manus_agent.tools.compare_cves._fetch_kev", side_effect=[sample_kev_not_in_kev, sample_kev_not_in_kev]):
            rc = _run_compare(["CVE-2024-0001", "CVE-2024-0002"])

        assert rc == 0
        captured = capsys.readouterr()
        assert "N/A" in captured.out  # CWE shown as N/A


# ---------------------------------------------------------------------------
# TestRunCompareSubcommandRegistry — routing from main()
# ---------------------------------------------------------------------------


class TestRunCompareSubcommandRegistry:
    """Tests for _SUBCOMMANDS registration and main() routing."""

    def test_compare_in_subcommands_set(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "compare" in _SUBCOMMANDS

    def test_main_routes_to_compare(self):
        """main() correctly routes 'compare' to _run_compare."""
        from manus_agent.cli import main

        with patch("manus_agent.cli._run_compare", return_value=0) as mock_run:
            with pytest.raises(SystemExit) as exc_info:
                with patch("sys.argv", ["manus-agent", "compare", "CVE-2024-3094", "CVE-2021-44228"]):
                    main()
            # Should exit with the return value of _run_compare
            assert exc_info.value.code == 0
            mock_run.assert_called_once()

    def test_main_passes_argv_to_run_compare(self):
        """main() passes the correct argv slice to _run_compare."""
        from manus_agent.cli import main

        with patch("manus_agent.cli._run_compare", return_value=0) as mock_run:
            with pytest.raises(SystemExit):
                with patch("sys.argv", ["manus-agent", "compare", "CVE-2024-3094", "CVE-2021-44228", "--output", "json"]):
                    main()
            call_args = mock_run.call_args[0][0]
            assert "CVE-2024-3094" in call_args
            assert "CVE-2021-44228" in call_args
            assert "--output" in call_args
            assert "json" in call_args


# ---------------------------------------------------------------------------
# TestRenderText — _render_text output quality
# ---------------------------------------------------------------------------


class TestRenderText:
    """Tests for _render_text() rendering."""

    def test_render_text_includes_both_cve_ids(self, sample_comparison):
        from manus_agent.tools.compare_cves import _render_text

        text = _render_text(sample_comparison)
        assert "CVE-2024-3094" in text
        assert "CVE-2021-44228" in text

    def test_render_text_includes_separator_lines(self, sample_comparison):
        from manus_agent.tools.compare_cves import _render_text

        text = _render_text(sample_comparison)
        assert "─" in text

    def test_render_text_includes_published_dates(self, sample_comparison):
        from manus_agent.tools.compare_cves import _render_text

        text = _render_text(sample_comparison)
        assert "2024-03-29" in text
        assert "2021-12-10" in text

    def test_render_text_includes_affected_products(self, sample_comparison):
        from manus_agent.tools.compare_cves import _render_text

        text = _render_text(sample_comparison)
        assert "Xz" in text or "xz" in text.lower()
        assert "Log4j" in text or "log4j" in text.lower()

    def test_render_text_includes_description_when_present(self, sample_comparison):
        from manus_agent.tools.compare_cves import _render_text

        text = _render_text(sample_comparison)
        # At least one description should be present
        assert "Malicious" in text or "Apache" in text

    def test_render_text_returns_string(self, sample_comparison):
        from manus_agent.tools.compare_cves import _render_text

        text = _render_text(sample_comparison)
        assert isinstance(text, str)
        assert len(text) > 100  # Not empty / trivial


# ---------------------------------------------------------------------------
# TestBuildComparison — comparison assembly logic
# ---------------------------------------------------------------------------


class TestBuildComparison:
    """Tests for _build_comparison()."""

    def test_build_comparison_returns_dict(self, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.tools.compare_cves import _build_comparison

        result = _build_comparison(sample_profile_a, sample_kev_a, sample_profile_b, sample_kev_b)
        assert isinstance(result, dict)

    def test_build_comparison_attaches_kev_to_profiles(self, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.tools.compare_cves import _build_comparison

        result = _build_comparison(sample_profile_a, sample_kev_a, sample_profile_b, sample_kev_b)
        assert result["cve_a"]["kev"] == sample_kev_a
        assert result["cve_b"]["kev"] == sample_kev_b

    def test_build_comparison_has_required_keys(self, sample_profile_a, sample_profile_b, sample_kev_a, sample_kev_b):
        from manus_agent.tools.compare_cves import _build_comparison

        result = _build_comparison(sample_profile_a, sample_kev_a, sample_profile_b, sample_kev_b)
        required_keys = {"cve_a", "cve_b", "priority_score_a", "priority_score_b",
                         "higher_priority", "confidence", "recommendation", "winner_reasons"}
        assert required_keys.issubset(set(result.keys()))

    def test_build_comparison_tie_both_in_kev(self, sample_profile_a, sample_kev_a):
        """Two identical CVEs (both in KEV, same CVSS/EPSS) should tie."""
        import copy
        from manus_agent.tools.compare_cves import _build_comparison

        profile_b = copy.deepcopy(sample_profile_a)
        profile_b["cve_id"] = "CVE-2021-44228"

        result = _build_comparison(sample_profile_a, sample_kev_a, profile_b, sample_kev_a)
        assert result["higher_priority"] == "tie"


# ---------------------------------------------------------------------------
# TestScoreCve — scoring rubric
# ---------------------------------------------------------------------------


class TestScoreCve:
    """Tests for _score_cve() priority scoring."""

    def test_kev_adds_10_points(self):
        from manus_agent.tools.compare_cves import _score_cve

        profile = {"kev": {"in_kev": True}, "cvss": {}, "epss": {}}
        score, reasons = _score_cve(profile)
        assert score >= 10
        assert any("KEV" in r for r in reasons)

    def test_critical_cvss_adds_8_points(self):
        from manus_agent.tools.compare_cves import _score_cve

        profile = {"kev": {"in_kev": False}, "cvss": {"score": 9.5}, "epss": {}}
        score, reasons = _score_cve(profile)
        assert score >= 8
        assert any("Critical" in r for r in reasons)

    def test_high_cvss_adds_5_points(self):
        from manus_agent.tools.compare_cves import _score_cve

        profile = {"kev": {"in_kev": False}, "cvss": {"score": 7.5}, "epss": {}}
        score, reasons = _score_cve(profile)
        assert score >= 5
        assert any("High" in r for r in reasons)

    def test_medium_cvss_adds_2_points(self):
        from manus_agent.tools.compare_cves import _score_cve

        profile = {"kev": {"in_kev": False}, "cvss": {"score": 5.0}, "epss": {}}
        score, reasons = _score_cve(profile)
        assert score >= 2
        assert any("Medium" in r for r in reasons)

    def test_very_high_epss_adds_8_points(self):
        from manus_agent.tools.compare_cves import _score_cve

        profile = {"kev": {"in_kev": False}, "cvss": {}, "epss": {"epss": "0.85"}}
        score, reasons = _score_cve(profile)
        assert score >= 8
        assert any("very high" in r for r in reasons)

    def test_high_epss_adds_5_points(self):
        from manus_agent.tools.compare_cves import _score_cve

        profile = {"kev": {"in_kev": False}, "cvss": {}, "epss": {"epss": "0.55"}}
        score, reasons = _score_cve(profile)
        assert score >= 5
        assert any("high" in r.lower() for r in reasons)

    def test_elevated_epss_adds_2_points(self):
        from manus_agent.tools.compare_cves import _score_cve

        profile = {"kev": {"in_kev": False}, "cvss": {}, "epss": {"epss": "0.15"}}
        score, reasons = _score_cve(profile)
        assert score >= 2
        assert any("elevated" in r for r in reasons)

    def test_network_attack_vector_adds_3_points(self):
        from manus_agent.tools.compare_cves import _score_cve

        profile = {"kev": {"in_kev": False}, "cvss": {"attack_vector": "NETWORK"}, "epss": {}}
        score, reasons = _score_cve(profile)
        assert score >= 3
        assert any("network" in r.lower() for r in reasons)

    def test_no_privileges_adds_2_points(self):
        from manus_agent.tools.compare_cves import _score_cve

        profile = {"kev": {"in_kev": False}, "cvss": {"privileges_required": "NONE"}, "epss": {}}
        score, reasons = _score_cve(profile)
        assert score >= 2
        assert any("no privileges" in r for r in reasons)

    def test_no_user_interaction_adds_1_point(self):
        from manus_agent.tools.compare_cves import _score_cve

        profile = {"kev": {"in_kev": False}, "cvss": {"user_interaction": "NONE"}, "epss": {}}
        score, reasons = _score_cve(profile)
        assert score >= 1
        assert any("no user interaction" in r for r in reasons)

    def test_empty_profile_scores_zero(self):
        from manus_agent.tools.compare_cves import _score_cve

        profile = {"kev": {"in_kev": False}, "cvss": {}, "epss": {}}
        score, reasons = _score_cve(profile)
        assert score == 0
        assert reasons == []

    def test_all_factors_combined(self):
        from manus_agent.tools.compare_cves import _score_cve

        profile = {
            "kev": {"in_kev": True},
            "cvss": {
                "score": 10.0,
                "attack_vector": "NETWORK",
                "privileges_required": "NONE",
                "user_interaction": "NONE",
            },
            "epss": {"epss": "0.97"},
        }
        score, reasons = _score_cve(profile)
        # 10 (KEV) + 8 (CVSS Critical) + 8 (EPSS very high) + 3 (NETWORK) + 2 (no priv) + 1 (no UI) = 32
        assert score == 32
        assert len(reasons) == 6
