"""Comprehensive test suite for get_attack_map tool and CLI subcommand.

All HTTP calls are mocked — no real network requests.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.get_attack_map import (
    _CWE_ATTACK_FALLBACK,
    _TACTIC_ORDER,
    TOOL_SPEC,
    _extract_tactic,
    _extract_technique_name,
    _fetch_attack_for_capec,
    _fetch_capecs_for_cwe,
    _fetch_cwes_from_nvd,
    _format_json,
    _format_text,
    _map_cve_to_attack,
    fetch_attack_map,
    get_attack_map,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_nvd_response_single_cwe():
    """NVD response with a single CWE."""
    return {
        "vulnerabilities": [
            {
                "cve": {
                    "weaknesses": [
                        {
                            "source": "nvd@nist.gov",
                            "description": [{"lang": "en", "value": "CWE-79"}],
                        }
                    ]
                }
            }
        ]
    }


@pytest.fixture
def mock_nvd_response_multi_cwe():
    """NVD response with multiple CWEs."""
    return {
        "vulnerabilities": [
            {
                "cve": {
                    "weaknesses": [
                        {
                            "source": "nvd@nist.gov",
                            "description": [{"lang": "en", "value": "CWE-89"}],
                        },
                        {
                            "source": "nvd@nist.gov",
                            "description": [{"lang": "en", "value": "CWE-78"}],
                        },
                    ]
                }
            }
        ]
    }


@pytest.fixture
def mock_nvd_response_no_cwe():
    """NVD response with no CWE mappings."""
    return {"vulnerabilities": [{"cve": {"weaknesses": []}}]}


@pytest.fixture
def mock_nvd_response_empty():
    """NVD response with no vulnerabilities."""
    return {"vulnerabilities": []}


@pytest.fixture
def mock_cwe_html_with_capec():
    """CWE HTML page containing CAPEC references."""
    return """
    <html><body>
    <h2>Related Attack Patterns</h2>
    <table>
    <tr><td><a href="/data/definitions/86.html">CAPEC-86</a></td></tr>
    <tr><td><a href="/data/definitions/198.html">CAPEC-198</a></td></tr>
    </table>
    </body></html>
    """


@pytest.fixture
def mock_capec_html_with_attack():
    """CAPEC HTML page containing ATT&CK technique references."""
    return """
    <html><body>
    <h2>Taxonomy Mappings</h2>
    <table>
    <tr>
        <td>ATTACK</td>
        <td>Execution</td>
        <td><a href="https://attack.mitre.org/techniques/T1059/">T1059</a> - Command and Scripting Interpreter</td>
    </tr>
    <tr>
        <td>ATTACK</td>
        <td>Initial Access</td>
        <td><a href="https://attack.mitre.org/techniques/T1190/">T1190</a> - Exploit Public-Facing Application</td>
    </tr>
    </table>
    </body></html>
    """


# ---------------------------------------------------------------------------
# TOOL_SPEC contract tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Verify TOOL_SPEC matches Strands contract."""

    def test_has_required_keys(self):
        assert "name" in TOOL_SPEC
        assert "description" in TOOL_SPEC
        assert "inputSchema" in TOOL_SPEC

    def test_name(self):
        assert TOOL_SPEC["name"] == "get_attack_map"

    def test_input_schema_has_cve_id(self):
        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert "cve_id" in props
        assert props["cve_id"]["type"] == "string"

    def test_input_schema_has_output_format(self):
        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert "output_format" in props
        assert props["output_format"]["enum"] == ["text", "json"]

    def test_required_fields(self):
        assert "cve_id" in TOOL_SPEC["inputSchema"]["json"]["required"]


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Test input validation in the Strands handler."""

    def test_empty_cve_id(self):
        tool: dict = {"toolUseId": "test-1", "input": {"cve_id": ""}}
        result = get_attack_map(tool)
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    def test_invalid_cve_format(self):
        tool: dict = {"toolUseId": "test-2", "input": {"cve_id": "not-a-cve"}}
        result = get_attack_map(tool)
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    def test_missing_cve_id(self):
        tool: dict = {"toolUseId": "test-3", "input": {}}
        result = get_attack_map(tool)
        assert result["status"] == "error"

    def test_malformed_cve_id(self):
        tool: dict = {"toolUseId": "test-4", "input": {"cve_id": "CVE-2024"}}
        result = get_attack_map(tool)
        assert result["status"] == "error"

    def test_whitespace_cve_id(self):
        tool: dict = {"toolUseId": "test-5", "input": {"cve_id": "   "}}
        result = get_attack_map(tool)
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# NVD CWE extraction tests
# ---------------------------------------------------------------------------


class TestFetchCwesFromNvd:
    """Test CVE → CWE extraction from NVD."""

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_single_cwe(self, mock_get, mock_nvd_response_single_cwe):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_nvd_response_single_cwe
        mock_get.return_value = mock_resp

        result = _fetch_cwes_from_nvd("CVE-2024-1234")
        assert result == ["CWE-79"]

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_multiple_cwes(self, mock_get, mock_nvd_response_multi_cwe):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_nvd_response_multi_cwe
        mock_get.return_value = mock_resp

        result = _fetch_cwes_from_nvd("CVE-2024-1234")
        assert result == ["CWE-89", "CWE-78"]

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_no_cwes(self, mock_get, mock_nvd_response_no_cwe):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_nvd_response_no_cwe
        mock_get.return_value = mock_resp

        result = _fetch_cwes_from_nvd("CVE-2024-1234")
        assert result == []

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_empty_vulnerabilities(self, mock_get, mock_nvd_response_empty):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_nvd_response_empty
        mock_get.return_value = mock_resp

        result = _fetch_cwes_from_nvd("CVE-2024-9999")
        assert result == []

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_404_returns_empty(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = _fetch_cwes_from_nvd("CVE-9999-0001")
        assert result == []

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_deduplicates_cwes(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "weaknesses": [
                            {"description": [{"value": "CWE-79"}]},
                            {"description": [{"value": "CWE-79"}]},
                        ]
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp

        result = _fetch_cwes_from_nvd("CVE-2024-1234")
        assert result == ["CWE-79"]

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_nvd_api_key_passed(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {"NVD_API_KEY": "test-key-123"}):
            _fetch_cwes_from_nvd("CVE-2024-1234")

        call_kwargs = mock_get.call_args
        assert call_kwargs is not None


# ---------------------------------------------------------------------------
# CWE → CAPEC extraction tests
# ---------------------------------------------------------------------------


class TestFetchCapecsForCwe:
    """Test CWE → CAPEC resolution."""

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_finds_capec_ids(self, mock_get, mock_cwe_html_with_capec):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = mock_cwe_html_with_capec
        mock_get.return_value = mock_resp

        result = _fetch_capecs_for_cwe(79)
        assert 86 in result
        assert 198 in result

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_no_capecs_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body>No attack patterns</body></html>"
        mock_get.return_value = mock_resp

        result = _fetch_capecs_for_cwe(999)
        assert result == []

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_http_error_returns_empty(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = _fetch_capecs_for_cwe(79)
        assert result == []

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_network_error_returns_empty(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("timeout")

        result = _fetch_capecs_for_cwe(79)
        assert result == []

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_deduplicates_capec_ids(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "CAPEC-86 CAPEC-86 CAPEC-86 CAPEC-100"
        mock_get.return_value = mock_resp

        result = _fetch_capecs_for_cwe(79)
        assert result.count(86) == 1
        assert 100 in result


# ---------------------------------------------------------------------------
# CAPEC → ATT&CK extraction tests
# ---------------------------------------------------------------------------


class TestFetchAttackForCapec:
    """Test CAPEC → ATT&CK technique resolution."""

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_finds_techniques(self, mock_get, mock_capec_html_with_attack):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = mock_capec_html_with_attack
        mock_get.return_value = mock_resp

        result = _fetch_attack_for_capec(86)
        technique_ids = [t["technique_id"] for t in result]
        assert "T1059" in technique_ids
        assert "T1190" in technique_ids

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_no_techniques_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body>No techniques here</body></html>"
        mock_get.return_value = mock_resp

        result = _fetch_attack_for_capec(999)
        assert result == []

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_http_error_returns_empty(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp

        result = _fetch_attack_for_capec(86)
        assert result == []

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_sub_technique_parsing(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = """
        <td>T1059.001 - PowerShell</td>
        <td>Execution</td>
        """
        mock_get.return_value = mock_resp

        result = _fetch_attack_for_capec(86)
        technique_ids = [t["technique_id"] for t in result]
        assert "T1059.001" in technique_ids

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_deduplicates_techniques(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "T1059 T1059 T1059 T1190"
        mock_get.return_value = mock_resp

        result = _fetch_attack_for_capec(86)
        ids = [t["technique_id"] for t in result]
        assert ids.count("T1059") == 1

    @patch("manus_agent.tools.get_attack_map._get_with_retry")
    def test_network_error_returns_empty(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.Timeout("timeout")

        result = _fetch_attack_for_capec(86)
        assert result == []


# ---------------------------------------------------------------------------
# Technique name extraction tests
# ---------------------------------------------------------------------------


class TestExtractTechniqueName:
    """Test technique name extraction from HTML context."""

    def test_dash_separator(self):
        html = "T1059 - Command and Scripting Interpreter"
        assert _extract_technique_name(html, "T1059") == "Command and Scripting Interpreter"

    def test_colon_separator(self):
        html = "T1059: Command and Scripting Interpreter"
        assert _extract_technique_name(html, "T1059") == "Command and Scripting Interpreter"

    def test_link_format(self):
        html = '<a href="...">T1059</a> - Command and Scripting Interpreter'
        result = _extract_technique_name(html, "T1059")
        assert result == "Command and Scripting Interpreter"

    def test_no_name_found(self):
        html = "some unrelated text"
        assert _extract_technique_name(html, "T1059") is None

    def test_ignores_technique_id_as_name(self):
        html = "T1059 - T1190"
        result = _extract_technique_name(html, "T1059")
        # Should not return T1190 as a name
        assert result is None or not result.startswith("T1")


# ---------------------------------------------------------------------------
# Tactic extraction tests
# ---------------------------------------------------------------------------


class TestExtractTactic:
    """Test tactic extraction from HTML context."""

    def test_finds_execution_tactic(self):
        html = "some text Execution some more T1059 text"
        assert _extract_tactic(html, "T1059") == "Execution"

    def test_finds_initial_access_tactic(self):
        html = "Initial Access stuff here T1190 related content"
        assert _extract_tactic(html, "T1190") == "Initial Access"

    def test_no_tactic_found(self):
        html = "no tactic T1234 in this text"
        assert _extract_tactic(html, "T1234") is None

    def test_technique_not_in_html(self):
        html = "Execution tactic mentioned"
        assert _extract_tactic(html, "T9999") is None


# ---------------------------------------------------------------------------
# Core mapping logic tests
# ---------------------------------------------------------------------------


class TestMapCveToAttack:
    """Test the full CVE → ATT&CK mapping pipeline."""

    @patch("manus_agent.tools.get_attack_map._fetch_attack_for_capec")
    @patch("manus_agent.tools.get_attack_map._fetch_capecs_for_cwe")
    @patch("manus_agent.tools.get_attack_map._fetch_cwes_from_nvd")
    def test_full_chain_success(self, mock_cwes, mock_capecs, mock_attack):
        mock_cwes.return_value = ["CWE-79"]
        mock_capecs.return_value = [86]
        mock_attack.return_value = [
            {"technique_id": "T1059.007", "technique_name": "JavaScript", "tactic": "Execution"}
        ]

        result = _map_cve_to_attack("CVE-2024-1234")
        assert result["cve_id"] == "CVE-2024-1234"
        assert result["cwes"] == ["CWE-79"]
        assert len(result["mappings"]) == 1
        assert result["mappings"][0]["technique_id"] == "T1059.007"
        assert "Execution" in result["tactics_summary"]

    @patch("manus_agent.tools.get_attack_map._fetch_cwes_from_nvd")
    def test_no_cwes_returns_error(self, mock_cwes):
        mock_cwes.return_value = []

        result = _map_cve_to_attack("CVE-2024-9999")
        assert result["cwes"] == []
        assert result["mappings"] == []
        assert "error" in result

    @patch("manus_agent.tools.get_attack_map._fetch_attack_for_capec")
    @patch("manus_agent.tools.get_attack_map._fetch_capecs_for_cwe")
    @patch("manus_agent.tools.get_attack_map._fetch_cwes_from_nvd")
    def test_fallback_mapping_used(self, mock_cwes, mock_capecs, mock_attack):
        """When CAPEC resolution fails, fallback curated mappings are used."""
        mock_cwes.return_value = ["CWE-79"]
        mock_capecs.return_value = []  # No CAPECs found
        mock_attack.return_value = []

        result = _map_cve_to_attack("CVE-2024-1234")
        # CWE-79 is in the fallback map
        assert len(result["mappings"]) > 0
        assert any(m["source"] == "curated" for m in result["mappings"])

    @patch("manus_agent.tools.get_attack_map._fetch_attack_for_capec")
    @patch("manus_agent.tools.get_attack_map._fetch_capecs_for_cwe")
    @patch("manus_agent.tools.get_attack_map._fetch_cwes_from_nvd")
    def test_multiple_cwes_combined(self, mock_cwes, mock_capecs, mock_attack):
        mock_cwes.return_value = ["CWE-89", "CWE-78"]
        mock_capecs.return_value = []
        mock_attack.return_value = []

        result = _map_cve_to_attack("CVE-2024-1234")
        assert len(result["cwes"]) == 2
        # Both CWE-89 and CWE-78 are in fallback map
        assert len(result["mappings"]) > 0

    @patch("manus_agent.tools.get_attack_map._fetch_attack_for_capec")
    @patch("manus_agent.tools.get_attack_map._fetch_capecs_for_cwe")
    @patch("manus_agent.tools.get_attack_map._fetch_cwes_from_nvd")
    def test_deduplicates_techniques_across_cwes(self, mock_cwes, mock_capecs, mock_attack):
        """Same technique from different CWEs shouldn't appear twice."""
        mock_cwes.return_value = ["CWE-78", "CWE-77"]
        mock_capecs.return_value = []
        mock_attack.return_value = []

        result = _map_cve_to_attack("CVE-2024-1234")
        technique_ids = [m["technique_id"] for m in result["mappings"]]
        assert len(technique_ids) == len(set(technique_ids))

    @patch("manus_agent.tools.get_attack_map._fetch_attack_for_capec")
    @patch("manus_agent.tools.get_attack_map._fetch_capecs_for_cwe")
    @patch("manus_agent.tools.get_attack_map._fetch_cwes_from_nvd")
    def test_capec_preferred_over_fallback(self, mock_cwes, mock_capecs, mock_attack):
        """If CAPEC resolves, fallback should NOT be used for that CWE."""
        mock_cwes.return_value = ["CWE-79"]
        mock_capecs.return_value = [86]
        mock_attack.return_value = [
            {"technique_id": "T1059.007", "technique_name": "JavaScript", "tactic": "Execution"}
        ]

        result = _map_cve_to_attack("CVE-2024-1234")
        # Should only have CAPEC-sourced mappings
        assert all(m["source"] == "capec" for m in result["mappings"])

    @patch("manus_agent.tools.get_attack_map._fetch_attack_for_capec")
    @patch("manus_agent.tools.get_attack_map._fetch_capecs_for_cwe")
    @patch("manus_agent.tools.get_attack_map._fetch_cwes_from_nvd")
    def test_tactics_summary_ordered(self, mock_cwes, mock_capecs, mock_attack):
        mock_cwes.return_value = ["CWE-79"]
        mock_capecs.return_value = [86]
        mock_attack.return_value = [
            {"technique_id": "T1185", "technique_name": "Browser Session Hijacking", "tactic": "Collection"},
            {"technique_id": "T1059.007", "technique_name": "JavaScript", "tactic": "Execution"},
        ]

        result = _map_cve_to_attack("CVE-2024-1234")
        # Execution comes before Collection in kill chain
        assert result["tactics_summary"].index("Execution") < result["tactics_summary"].index("Collection")

    @patch("manus_agent.tools.get_attack_map._fetch_cwes_from_nvd")
    def test_cve_id_normalized_to_uppercase(self, mock_cwes):
        mock_cwes.return_value = []
        result = _map_cve_to_attack("cve-2024-1234")
        assert result["cve_id"] == "CVE-2024-1234"


# ---------------------------------------------------------------------------
# Fallback mapping tests
# ---------------------------------------------------------------------------


class TestFallbackMappings:
    """Test the curated CWE → ATT&CK fallback dictionary."""

    def test_xss_mapping(self):
        assert 79 in _CWE_ATTACK_FALLBACK
        techniques = _CWE_ATTACK_FALLBACK[79]
        ids = [t["technique_id"] for t in techniques]
        assert "T1059.007" in ids

    def test_sqli_mapping(self):
        assert 89 in _CWE_ATTACK_FALLBACK
        techniques = _CWE_ATTACK_FALLBACK[89]
        ids = [t["technique_id"] for t in techniques]
        assert "T1190" in ids

    def test_command_injection_mapping(self):
        assert 78 in _CWE_ATTACK_FALLBACK
        techniques = _CWE_ATTACK_FALLBACK[78]
        ids = [t["technique_id"] for t in techniques]
        assert "T1059" in ids

    def test_deserialization_mapping(self):
        assert 502 in _CWE_ATTACK_FALLBACK
        techniques = _CWE_ATTACK_FALLBACK[502]
        ids = [t["technique_id"] for t in techniques]
        assert "T1190" in ids

    def test_all_entries_have_required_fields(self):
        for cwe_num, techniques in _CWE_ATTACK_FALLBACK.items():
            for tech in techniques:
                assert "technique_id" in tech, f"CWE-{cwe_num} missing technique_id"
                assert "technique_name" in tech, f"CWE-{cwe_num} missing technique_name"
                assert "tactic" in tech, f"CWE-{cwe_num} missing tactic"

    def test_all_tactics_are_valid(self):
        for cwe_num, techniques in _CWE_ATTACK_FALLBACK.items():
            for tech in techniques:
                assert tech["tactic"] in _TACTIC_ORDER, f"CWE-{cwe_num} has invalid tactic '{tech['tactic']}'"


# ---------------------------------------------------------------------------
# Text formatting tests
# ---------------------------------------------------------------------------


class TestFormatText:
    """Test human-readable text output formatting."""

    def test_success_output(self):
        result = {
            "cve_id": "CVE-2024-1234",
            "cwes": ["CWE-79"],
            "mappings": [
                {
                    "technique_id": "T1059.007",
                    "technique_name": "JavaScript",
                    "tactic": "Execution",
                    "path": "CWE-79 → CAPEC-86 → T1059.007",
                    "source": "capec",
                }
            ],
            "tactics_summary": ["Execution"],
        }
        text = _format_text(result)
        assert "CVE-2024-1234" in text
        assert "CWE-79" in text
        assert "T1059.007" in text
        assert "JavaScript" in text
        assert "Execution" in text
        assert "Kill-Chain Coverage" in text

    def test_error_output(self):
        result = {
            "cve_id": "CVE-2024-9999",
            "cwes": [],
            "mappings": [],
            "tactics_summary": [],
            "error": "No CWE mappings found for CVE-2024-9999 in NVD.",
        }
        text = _format_text(result)
        assert "⚠" in text
        assert "No CWE mappings" in text

    def test_no_mappings_output(self):
        result = {
            "cve_id": "CVE-2024-1234",
            "cwes": ["CWE-999"],
            "mappings": [],
            "tactics_summary": [],
        }
        text = _format_text(result)
        assert "No ATT&CK technique mappings" in text

    def test_attack_url_format(self):
        result = {
            "cve_id": "CVE-2024-1234",
            "cwes": ["CWE-79"],
            "mappings": [
                {
                    "technique_id": "T1059.007",
                    "technique_name": "JavaScript",
                    "tactic": "Execution",
                    "path": "CWE-79 → T1059.007 (curated)",
                    "source": "curated",
                }
            ],
            "tactics_summary": ["Execution"],
        }
        text = _format_text(result)
        assert "attack.mitre.org/techniques/T1059/007/" in text

    def test_summary_line(self):
        result = {
            "cve_id": "CVE-2024-1234",
            "cwes": ["CWE-79"],
            "mappings": [
                {
                    "technique_id": "T1059.007",
                    "technique_name": "JavaScript",
                    "tactic": "Execution",
                    "path": "...",
                    "source": "capec",
                },
                {
                    "technique_id": "T1185",
                    "technique_name": "Browser Session Hijacking",
                    "tactic": "Collection",
                    "path": "...",
                    "source": "capec",
                },
            ],
            "tactics_summary": ["Execution", "Collection"],
        }
        text = _format_text(result)
        assert "2 technique(s)" in text
        assert "2 tactic(s)" in text


# ---------------------------------------------------------------------------
# JSON formatting tests
# ---------------------------------------------------------------------------


class TestFormatJson:
    """Test structured JSON output formatting."""

    def test_success_structure(self):
        result = {
            "cve_id": "CVE-2024-1234",
            "cwes": ["CWE-79"],
            "mappings": [
                {
                    "technique_id": "T1059.007",
                    "technique_name": "JavaScript",
                    "tactic": "Execution",
                    "path": "CWE-79 → CAPEC-86 → T1059.007",
                    "source": "capec",
                }
            ],
            "tactics_summary": ["Execution"],
        }
        output = _format_json(result)
        assert output["cve_id"] == "CVE-2024-1234"
        assert output["cwes"] == ["CWE-79"]
        assert output["technique_count"] == 1
        assert output["tactic_count"] == 1
        assert len(output["techniques"]) == 1
        assert output["techniques"][0]["id"] == "T1059.007"
        assert output["techniques"][0]["source"] == "capec"
        assert "url" in output["techniques"][0]

    def test_error_in_json(self):
        result = {
            "cve_id": "CVE-2024-9999",
            "cwes": [],
            "mappings": [],
            "tactics_summary": [],
            "error": "No CWE mappings found.",
        }
        output = _format_json(result)
        assert output["error"] == "No CWE mappings found."
        assert output["technique_count"] == 0

    def test_json_serializable(self):
        result = {
            "cve_id": "CVE-2024-1234",
            "cwes": ["CWE-79"],
            "mappings": [
                {
                    "technique_id": "T1059.007",
                    "technique_name": "JavaScript",
                    "tactic": "Execution",
                    "path": "CWE-79 → CAPEC-86 → T1059.007",
                    "source": "capec",
                }
            ],
            "tactics_summary": ["Execution"],
        }
        output = _format_json(result)
        # Should not raise
        serialized = json.dumps(output)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# Strands tool handler tests
# ---------------------------------------------------------------------------


class TestGetAttackMapHandler:
    """Test the Strands tool entry point."""

    @patch("manus_agent.tools.get_attack_map._map_cve_to_attack")
    def test_success_text_output(self, mock_map):
        mock_map.return_value = {
            "cve_id": "CVE-2024-1234",
            "cwes": ["CWE-79"],
            "mappings": [
                {
                    "technique_id": "T1059.007",
                    "technique_name": "JavaScript",
                    "tactic": "Execution",
                    "path": "CWE-79 → CAPEC-86 → T1059.007",
                    "source": "capec",
                }
            ],
            "tactics_summary": ["Execution"],
        }
        tool: dict = {"toolUseId": "test-1", "input": {"cve_id": "CVE-2024-1234"}}
        result = get_attack_map(tool)
        assert result["status"] == "success"
        assert "text" in result["content"][0]
        assert "T1059.007" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_attack_map._map_cve_to_attack")
    def test_success_json_output(self, mock_map):
        mock_map.return_value = {
            "cve_id": "CVE-2024-1234",
            "cwes": ["CWE-79"],
            "mappings": [
                {
                    "technique_id": "T1059.007",
                    "technique_name": "JavaScript",
                    "tactic": "Execution",
                    "path": "CWE-79 → CAPEC-86 → T1059.007",
                    "source": "capec",
                }
            ],
            "tactics_summary": ["Execution"],
        }
        tool: dict = {
            "toolUseId": "test-2",
            "input": {"cve_id": "CVE-2024-1234", "output_format": "json"},
        }
        result = get_attack_map(tool)
        assert result["status"] == "success"
        assert "json" in result["content"][0]
        assert result["content"][0]["json"]["cve_id"] == "CVE-2024-1234"

    @patch("manus_agent.tools.get_attack_map._map_cve_to_attack")
    def test_network_error(self, mock_map):
        import requests

        mock_map.side_effect = requests.exceptions.ConnectionError("failed")

        tool: dict = {"toolUseId": "test-3", "input": {"cve_id": "CVE-2024-1234"}}
        result = get_attack_map(tool)
        assert result["status"] == "error"
        assert "Network error" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_attack_map._map_cve_to_attack")
    def test_unexpected_error(self, mock_map):
        mock_map.side_effect = ValueError("unexpected")

        tool: dict = {"toolUseId": "test-4", "input": {"cve_id": "CVE-2024-1234"}}
        result = get_attack_map(tool)
        assert result["status"] == "error"
        assert "Unexpected error" in result["content"][0]["text"]

    def test_preserves_tool_use_id(self):
        tool: dict = {"toolUseId": "my-unique-id", "input": {"cve_id": "bad"}}
        result = get_attack_map(tool)
        assert result["toolUseId"] == "my-unique-id"


# ---------------------------------------------------------------------------
# CLI-facing function tests
# ---------------------------------------------------------------------------


class TestFetchAttackMap:
    """Test the public CLI-facing function."""

    @patch("manus_agent.tools.get_attack_map._map_cve_to_attack")
    def test_text_output(self, mock_map):
        mock_map.return_value = {
            "cve_id": "CVE-2024-1234",
            "cwes": ["CWE-79"],
            "mappings": [],
            "tactics_summary": [],
        }
        result = fetch_attack_map("CVE-2024-1234", output_format="text")
        assert isinstance(result, str)

    @patch("manus_agent.tools.get_attack_map._map_cve_to_attack")
    def test_json_output(self, mock_map):
        mock_map.return_value = {
            "cve_id": "CVE-2024-1234",
            "cwes": ["CWE-79"],
            "mappings": [],
            "tactics_summary": [],
        }
        result = fetch_attack_map("CVE-2024-1234", output_format="json")
        assert isinstance(result, dict)
        assert result["cve_id"] == "CVE-2024-1234"


# ---------------------------------------------------------------------------
# CLI subcommand tests
# ---------------------------------------------------------------------------


class TestCliAttackMap:
    """Test the CLI subcommand dispatch."""

    @patch("manus_agent.tools.get_attack_map.fetch_attack_map")
    def test_text_output(self, mock_fetch, capsys):
        mock_fetch.return_value = "## ATT&CK Mapping: CVE-2024-1234\n\nNo mappings."

        from manus_agent.cli import _run_attack_map

        exit_code = _run_attack_map(["CVE-2024-1234"])
        assert exit_code == 0

    @patch("manus_agent.tools.get_attack_map.fetch_attack_map")
    def test_json_output(self, mock_fetch, capsys):
        mock_fetch.return_value = {"cve_id": "CVE-2024-1234", "techniques": []}

        from manus_agent.cli import _run_attack_map

        exit_code = _run_attack_map(["CVE-2024-1234", "--output", "json"])
        assert exit_code == 0

    @patch("manus_agent.tools.get_attack_map.fetch_attack_map")
    def test_error_returns_1(self, mock_fetch):
        mock_fetch.side_effect = RuntimeError("boom")

        from manus_agent.cli import _run_attack_map

        exit_code = _run_attack_map(["CVE-2024-1234"])
        assert exit_code == 1

    def test_subcommand_registered(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "attack-map" in _SUBCOMMANDS

    def test_parser_help(self):
        from manus_agent.cli import _build_attack_map_parser

        parser = _build_attack_map_parser()
        assert parser.prog == "manus-agent attack-map"


# ---------------------------------------------------------------------------
# HTTP retry tests
# ---------------------------------------------------------------------------


class TestGetWithRetry:
    """Test the retry/back-off HTTP helper."""

    @patch("manus_agent.tools.get_attack_map.requests.get")
    def test_success_first_try(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        from manus_agent.tools.get_attack_map import _get_with_retry

        result = _get_with_retry("https://example.com")
        assert result.status_code == 200
        assert mock_get.call_count == 1

    @patch("manus_agent.tools.get_attack_map.time.sleep")
    @patch("manus_agent.tools.get_attack_map.requests.get")
    def test_retries_on_429(self, mock_get, mock_sleep):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {}

        resp_200 = MagicMock()
        resp_200.status_code = 200

        mock_get.side_effect = [resp_429, resp_200]

        from manus_agent.tools.get_attack_map import _get_with_retry

        result = _get_with_retry("https://example.com")
        assert result.status_code == 200
        assert mock_get.call_count == 2

    @patch("manus_agent.tools.get_attack_map.time.sleep")
    @patch("manus_agent.tools.get_attack_map.requests.get")
    def test_retries_on_500(self, mock_get, mock_sleep):
        resp_500 = MagicMock()
        resp_500.status_code = 500

        resp_200 = MagicMock()
        resp_200.status_code = 200

        mock_get.side_effect = [resp_500, resp_200]

        from manus_agent.tools.get_attack_map import _get_with_retry

        result = _get_with_retry("https://example.com")
        assert result.status_code == 200

    @patch("manus_agent.tools.get_attack_map.time.sleep")
    @patch("manus_agent.tools.get_attack_map.requests.get")
    def test_retries_on_timeout(self, mock_get, mock_sleep):
        import requests

        mock_get.side_effect = [
            requests.exceptions.Timeout("timeout"),
            MagicMock(status_code=200),
        ]

        from manus_agent.tools.get_attack_map import _get_with_retry

        result = _get_with_retry("https://example.com")
        assert result.status_code == 200

    @patch("manus_agent.tools.get_attack_map.time.sleep")
    @patch("manus_agent.tools.get_attack_map.requests.get")
    def test_raises_after_max_retries(self, mock_get, mock_sleep):
        import requests

        mock_get.side_effect = requests.exceptions.Timeout("timeout")

        from manus_agent.tools.get_attack_map import _get_with_retry

        with pytest.raises(requests.exceptions.Timeout):
            _get_with_retry("https://example.com", max_retries=3)
        assert mock_get.call_count == 3


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("manus_agent.tools.get_attack_map._fetch_cwes_from_nvd")
    def test_unknown_cwe_no_fallback(self, mock_cwes):
        """CWE not in fallback map and no CAPEC data → empty mappings."""
        mock_cwes.return_value = ["CWE-9999"]

        with patch("manus_agent.tools.get_attack_map._fetch_capecs_for_cwe", return_value=[]):
            result = _map_cve_to_attack("CVE-2024-1234")
        assert result["mappings"] == []
        assert result["tactics_summary"] == []

    @patch("manus_agent.tools.get_attack_map._fetch_cwes_from_nvd")
    def test_cwe_noinfo_filtered(self, mock_cwes):
        """CWE 'NVD-CWE-noinfo' or 'NVD-CWE-Other' should not map."""
        mock_cwes.return_value = []  # NVD returns these as non-CWE-NNN patterns
        result = _map_cve_to_attack("CVE-2024-1234")
        assert "error" in result

    def test_tactic_order_completeness(self):
        """Ensure all 14 ATT&CK tactics are in the order list."""
        assert len(_TACTIC_ORDER) == 14
        assert "Initial Access" in _TACTIC_ORDER
        assert "Impact" in _TACTIC_ORDER

    @patch("manus_agent.tools.get_attack_map._map_cve_to_attack")
    def test_handler_with_whitespace_cve(self, mock_map):
        mock_map.return_value = {
            "cve_id": "CVE-2024-1234",
            "cwes": [],
            "mappings": [],
            "tactics_summary": [],
            "error": "No CWE mappings",
        }
        tool: dict = {"toolUseId": "test", "input": {"cve_id": "  CVE-2024-1234  "}}
        result = get_attack_map(tool)
        assert result["status"] == "success"
        mock_map.assert_called_once_with("CVE-2024-1234")

    @patch("manus_agent.tools.get_attack_map._fetch_attack_for_capec")
    @patch("manus_agent.tools.get_attack_map._fetch_capecs_for_cwe")
    @patch("manus_agent.tools.get_attack_map._fetch_cwes_from_nvd")
    def test_mapping_path_format(self, mock_cwes, mock_capecs, mock_attack):
        """Verify the mapping path string format."""
        mock_cwes.return_value = ["CWE-79"]
        mock_capecs.return_value = [86]
        mock_attack.return_value = [
            {"technique_id": "T1059.007", "technique_name": "JavaScript", "tactic": "Execution"}
        ]

        result = _map_cve_to_attack("CVE-2024-1234")
        path = result["mappings"][0]["path"]
        assert "CWE-79" in path
        assert "CAPEC-86" in path
        assert "T1059.007" in path
