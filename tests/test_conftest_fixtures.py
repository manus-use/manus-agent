"""Tests validating the conftest.py shared fixtures themselves.

These tests serve dual purposes:
1. Verify that all shared fixtures work correctly (regression guard)
2. Document usage patterns for future test authors

Run with: pytest tests/test_conftest_fixtures.py -v
"""

from __future__ import annotations

import json

import pytest
import requests.exceptions

# ===========================================================================
# make_tool_use fixture
# ===========================================================================


class TestMakeToolUse:
    """Verify the make_tool_use factory fixture."""

    def test_creates_valid_tool_use_dict(self, make_tool_use):
        tool_use = make_tool_use(cve_id="CVE-2024-3094")
        assert "toolUseId" in tool_use
        assert "input" in tool_use
        assert tool_use["input"]["cve_id"] == "CVE-2024-3094"

    def test_auto_generates_unique_ids(self, make_tool_use):
        t1 = make_tool_use(cve_id="CVE-2024-0001")
        t2 = make_tool_use(cve_id="CVE-2024-0002")
        assert t1["toolUseId"] != t2["toolUseId"]

    def test_accepts_custom_tool_use_id(self, make_tool_use):
        tool_use = make_tool_use(tool_use_id="my-custom-id", cve_id="CVE-2024-1234")
        assert tool_use["toolUseId"] == "my-custom-id"

    def test_accepts_tool_name(self, make_tool_use):
        tool_use = make_tool_use("get_nvd_data", cve_id="CVE-2024-1234")
        assert tool_use["name"] == "get_nvd_data"

    def test_multiple_input_kwargs(self, make_tool_use):
        tool_use = make_tool_use("compare_cves", cve_id_a="CVE-2021-44228", cve_id_b="CVE-2024-3094")
        assert tool_use["input"]["cve_id_a"] == "CVE-2021-44228"
        assert tool_use["input"]["cve_id_b"] == "CVE-2024-3094"

    def test_empty_input_valid(self, make_tool_use):
        tool_use = make_tool_use("check_cisa_kev")
        assert tool_use["input"] == {}


# ===========================================================================
# mock_http_response fixture
# ===========================================================================


class TestMockHttpResponse:
    """Verify the mock_http_response factory fixture."""

    def test_200_response_with_json(self, mock_http_response):
        resp = mock_http_response(200, {"data": [1, 2, 3]})
        assert resp.status_code == 200
        assert resp.json() == {"data": [1, 2, 3]}
        resp.raise_for_status()  # should not raise

    def test_429_raises_on_raise_for_status(self, mock_http_response):
        resp = mock_http_response(429)
        assert resp.status_code == 429
        with pytest.raises(requests.exceptions.HTTPError):
            resp.raise_for_status()

    def test_500_raises_on_raise_for_status(self, mock_http_response):
        resp = mock_http_response(500)
        with pytest.raises(requests.exceptions.HTTPError):
            resp.raise_for_status()

    def test_text_body_override(self, mock_http_response):
        resp = mock_http_response(200, text="plain text body")
        assert resp.text == "plain text body"

    def test_json_serialised_as_text_by_default(self, mock_http_response):
        payload = {"key": "value"}
        resp = mock_http_response(200, payload)
        assert json.loads(resp.text) == payload

    def test_no_json_raises_valueerror(self, mock_http_response):
        resp = mock_http_response(200, text="not json")
        with pytest.raises(ValueError):
            resp.json()

    def test_custom_headers(self, mock_http_response):
        resp = mock_http_response(200, headers={"X-Custom": "yes"})
        assert resp.headers["X-Custom"] == "yes"

    def test_links_for_pagination(self, mock_http_response):
        resp = mock_http_response(200, {"items": []}, links={"next": {"url": "http://example.com/page2"}})
        assert "next" in resp.links


# ===========================================================================
# nvd_cve_factory fixture
# ===========================================================================


class TestNvdCveFactory:
    """Verify the NVD CVE data factory."""

    def test_default_cve_has_required_fields(self, nvd_cve_factory):
        cve = nvd_cve_factory()
        assert cve["id"] == "CVE-2024-1234"
        assert cve["published"]
        assert len(cve["descriptions"]) >= 1
        assert "cvssMetricV31" in cve["metrics"]

    def test_custom_cve_id(self, nvd_cve_factory):
        cve = nvd_cve_factory("CVE-2021-44228")
        assert cve["id"] == "CVE-2021-44228"

    def test_custom_cvss_score(self, nvd_cve_factory):
        cve = nvd_cve_factory(base_score=7.5, severity="HIGH")
        cvss = cve["metrics"]["cvssMetricV31"][0]["cvssData"]
        assert cvss["baseScore"] == 7.5
        assert cvss["baseSeverity"] == "HIGH"

    def test_custom_attack_vector(self, nvd_cve_factory):
        cve = nvd_cve_factory(attack_vector="LOCAL", privileges_required="HIGH")
        cvss = cve["metrics"]["cvssMetricV31"][0]["cvssData"]
        assert cvss["attackVector"] == "LOCAL"
        assert cvss["privilegesRequired"] == "HIGH"

    def test_cpe_criteria_creates_configurations(self, nvd_cve_factory):
        cve = nvd_cve_factory(cpe_criteria="cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*")
        assert "configurations" in cve
        assert cve["configurations"][0]["nodes"][0]["cpeMatch"][0]["vulnerable"] is True

    def test_no_cpe_means_no_configurations(self, nvd_cve_factory):
        cve = nvd_cve_factory()
        assert "configurations" not in cve

    def test_custom_cwe(self, nvd_cve_factory):
        cve = nvd_cve_factory(cwe="CWE-502")
        assert cve["weaknesses"][0]["description"][0]["value"] == "CWE-502"


# ===========================================================================
# nvd_api_response fixture
# ===========================================================================


class TestNvdApiResponse:
    """Verify the full NVD API response wrapper."""

    def test_single_cve_response(self, nvd_api_response):
        payload = nvd_api_response("CVE-2024-3094")
        assert payload["totalResults"] == 1
        assert payload["vulnerabilities"][0]["cve"]["id"] == "CVE-2024-3094"

    def test_multiple_cves(self, nvd_api_response, nvd_cve_factory):
        cve1 = nvd_cve_factory("CVE-2024-0001")
        cve2 = nvd_cve_factory("CVE-2024-0002")
        payload = nvd_api_response(cve_entries=[cve1, cve2])
        assert payload["totalResults"] == 2

    def test_passes_kwargs_to_cve_factory(self, nvd_api_response):
        payload = nvd_api_response("CVE-2024-3094", base_score=10.0)
        cvss = payload["vulnerabilities"][0]["cve"]["metrics"]["cvssMetricV31"][0]["cvssData"]
        assert cvss["baseScore"] == 10.0


# ===========================================================================
# epss_data_factory fixture
# ===========================================================================


class TestEpssDataFactory:
    """Verify the EPSS data factory."""

    def test_default_payload_structure(self, epss_data_factory):
        payload = epss_data_factory()
        assert payload["status"] == "OK"
        assert len(payload["data"]) == 1
        assert payload["data"][0]["cve"] == "CVE-2024-1234"

    def test_custom_epss_values(self, epss_data_factory):
        payload = epss_data_factory("CVE-2021-44228", epss=0.975, percentile=0.999)
        entry = payload["data"][0]
        assert entry["epss"] == "0.975"
        assert entry["percentile"] == "0.999"

    def test_with_time_series(self, epss_data_factory):
        series = [
            {"date": "2024-06-01", "epss": "0.5", "percentile": "0.85"},
            {"date": "2024-06-02", "epss": "0.6", "percentile": "0.87"},
        ]
        payload = epss_data_factory(time_series=series)
        assert payload["data"][0]["time-series"] == series


# ===========================================================================
# kev_catalog_factory fixture
# ===========================================================================


class TestKevCatalogFactory:
    """Verify the KEV catalog factory."""

    def test_empty_catalog(self, kev_catalog_factory):
        catalog = kev_catalog_factory()
        assert catalog["vulnerabilities"] == []
        assert catalog["count"] == 0

    def test_catalog_with_cves(self, kev_catalog_factory):
        catalog = kev_catalog_factory(["CVE-2021-44228", "CVE-2024-3094"])
        assert catalog["count"] == 2
        cve_ids = [v["cveID"] for v in catalog["vulnerabilities"]]
        assert "CVE-2021-44228" in cve_ids
        assert "CVE-2024-3094" in cve_ids

    def test_catalog_entries_have_required_fields(self, kev_catalog_factory):
        catalog = kev_catalog_factory(["CVE-2024-1234"])
        entry = catalog["vulnerabilities"][0]
        assert "cveID" in entry
        assert "vendorProject" in entry
        assert "dateAdded" in entry
        assert "requiredAction" in entry


# ===========================================================================
# osv_record_factory fixture
# ===========================================================================


class TestOsvRecordFactory:
    """Verify the OSV record factory."""

    def test_default_record_structure(self, osv_record_factory):
        record = osv_record_factory()
        assert "id" in record
        assert "aliases" in record
        assert "affected" in record
        assert record["aliases"] == ["CVE-2024-1234"]

    def test_custom_ecosystem_and_package(self, osv_record_factory):
        record = osv_record_factory(
            "CVE-2021-44228",
            ecosystem="Maven",
            package="org.apache.logging.log4j:log4j-core",
        )
        affected = record["affected"][0]
        assert affected["package"]["ecosystem"] == "Maven"
        assert affected["package"]["name"] == "org.apache.logging.log4j:log4j-core"

    def test_custom_version_range(self, osv_record_factory):
        record = osv_record_factory(introduced="2.0-beta9", fixed="2.15.0")
        events = record["affected"][0]["ranges"][0]["events"]
        assert events[0]["introduced"] == "2.0-beta9"
        assert events[1]["fixed"] == "2.15.0"

    def test_custom_osv_id(self, osv_record_factory):
        record = osv_record_factory(osv_id="GHSA-jfh8-c2jp-5v3q")
        assert record["id"] == "GHSA-jfh8-c2jp-5v3q"


# ===========================================================================
# Config fixtures
# ===========================================================================


class TestConfigFixtures:
    """Verify pre-built Config fixtures."""

    def test_default_config(self, default_config):
        from manus_agent.config import Config

        assert isinstance(default_config, Config)

    def test_bedrock_config(self, bedrock_config):
        assert bedrock_config.llm.provider == "bedrock"

    def test_openai_config(self, openai_config):
        assert openai_config.llm.provider == "openai"
        assert openai_config.llm.model == "gpt-4o"

    def test_anthropic_config(self, anthropic_config):
        assert anthropic_config.llm.provider == "anthropic"

    def test_ollama_config(self, ollama_config):
        assert ollama_config.llm.provider == "ollama"
        assert ollama_config.llm.base_url == "http://localhost:11434"


# ===========================================================================
# Temporary file fixtures
# ===========================================================================


class TestTmpHistoryFile:
    """Verify the tmp_history_file fixture."""

    def test_creates_file_on_write(self, tmp_history_file):
        history_path, write = tmp_history_file
        write(task="test task")
        assert history_path.exists()

    def test_appends_valid_jsonl(self, tmp_history_file):
        history_path, write = tmp_history_file
        write(task="first task", success=True)
        write(task="second task", success=False)

        lines = history_path.read_text().strip().split("\n")
        assert len(lines) == 2
        rec1 = json.loads(lines[0])
        rec2 = json.loads(lines[1])
        assert rec1["task"] == "first task"
        assert rec2["success"] is False

    def test_records_have_expected_fields(self, tmp_history_file):
        _, write = tmp_history_file
        path = write(task="analyse CVE", agent="browser", mode="multi")
        record = json.loads(path.read_text().strip())
        assert record["agent"] == "browser"
        assert record["mode"] == "multi"
        assert "timestamp" in record


class TestTmpConfigFile:
    """Verify the tmp_config_file fixture."""

    def test_writes_valid_toml(self, tmp_config_file):
        import toml

        config_path = tmp_config_file({"llm": {"provider": "openai", "model": "gpt-4"}})
        assert config_path.exists()
        parsed = toml.loads(config_path.read_text())
        assert parsed["llm"]["provider"] == "openai"


# ===========================================================================
# env_override fixture
# ===========================================================================


class TestEnvOverride:
    """Verify the env_override fixture."""

    def test_sets_env_var(self, env_override):
        import os

        env_override(TEST_CONFTEST_VAR="hello")
        assert os.environ.get("TEST_CONFTEST_VAR") == "hello"

    def test_removes_env_var_with_none(self, env_override):
        import os

        os.environ["TEST_CONFTEST_REMOVE"] = "exists"
        env_override(TEST_CONFTEST_REMOVE=None)
        assert "TEST_CONFTEST_REMOVE" not in os.environ

    def test_multiple_vars(self, env_override):
        import os

        env_override(VAR_A="a", VAR_B="b")
        assert os.environ.get("VAR_A") == "a"
        assert os.environ.get("VAR_B") == "b"


# ===========================================================================
# Module-level constants
# ===========================================================================


class TestSampleCveConstants:
    """Verify the module-level CVE data constants."""

    def test_log4shell_has_id(self):
        from tests.conftest import SAMPLE_CVE_LOG4SHELL

        assert SAMPLE_CVE_LOG4SHELL["id"] == "CVE-2021-44228"

    def test_xz_has_id(self):
        from tests.conftest import SAMPLE_CVE_XZ

        assert SAMPLE_CVE_XZ["id"] == "CVE-2024-3094"

    def test_log4shell_has_cvss_score_10(self):
        from tests.conftest import SAMPLE_CVE_LOG4SHELL

        score = SAMPLE_CVE_LOG4SHELL["metrics"]["cvssMetricV31"][0]["cvssData"]["baseScore"]
        assert score == 10.0

    def test_xz_has_configurations(self):
        from tests.conftest import SAMPLE_CVE_XZ

        assert "configurations" in SAMPLE_CVE_XZ
        cpe = SAMPLE_CVE_XZ["configurations"][0]["nodes"][0]["cpeMatch"][0]
        assert "tukaani" in cpe["criteria"]
