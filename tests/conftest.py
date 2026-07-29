"""Shared pytest fixtures and factories for manus-agent tests.

This module provides reusable test infrastructure that eliminates common
boilerplate across test files:

- **Tool-use factories**: build valid Strands tool_use dicts with minimal args
- **HTTP response mocks**: construct ``requests.Response``-like mocks
- **CVE data factories**: realistic NVD, EPSS, KEV, and OSV payloads
- **Config fixtures**: pre-built Config objects for common provider scenarios
- **Temporary paths**: isolated tmp directories for history/config files

Usage
-----
Fixtures are auto-discovered by pytest. Import nothing — just declare the
fixture name as a test function parameter::

    def test_something(make_tool_use, mock_http_response):
        tool_use = make_tool_use("get_nvd_data", cve_id="CVE-2024-1234")
        response = mock_http_response(200, {"vulnerabilities": []})
        ...
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests.exceptions

# ===========================================================================
# Tool-use factories
# ===========================================================================


@pytest.fixture
def make_tool_use():
    """Factory fixture for creating Strands tool_use dicts.

    Returns a callable: ``make_tool_use(tool_name_or_id=None, **input_kwargs)``

    Examples::

        tool_use = make_tool_use(cve_id="CVE-2024-3094")
        tool_use = make_tool_use("get_nvd_data", cve_id="CVE-2024-3094")
        tool_use = make_tool_use(tool_use_id="custom-id", cve_id="CVE-2024-3094")
    """

    def _factory(tool_name: str | None = None, *, tool_use_id: str | None = None, **input_kwargs) -> dict[str, Any]:
        return {
            "toolUseId": tool_use_id or f"test-{uuid.uuid4().hex[:8]}",
            "name": tool_name or "test_tool",
            "input": input_kwargs,
        }

    return _factory


# ===========================================================================
# HTTP response mocks
# ===========================================================================


@pytest.fixture
def mock_http_response():
    """Factory fixture for creating mock ``requests.Response`` objects.

    Returns a callable: ``mock_http_response(status_code, json_payload=None, text=None)``

    The returned mock supports:
    - ``.status_code``
    - ``.json()`` → returns *json_payload*
    - ``.text`` → returns *text* or serialised JSON
    - ``.raise_for_status()`` → raises HTTPError for 4xx/5xx
    - ``.headers`` → empty dict (overridable)
    - ``.links`` → empty dict (overridable, useful for pagination mocks)

    Examples::

        resp = mock_http_response(200, {"data": [...]})
        resp = mock_http_response(429)  # triggers raise_for_status
        resp = mock_http_response(200, text="plain text body")
    """

    def _factory(
        status_code: int = 200,
        json_payload: Any = None,
        *,
        text: str | None = None,
        headers: dict[str, str] | None = None,
        links: dict | None = None,
    ) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        resp.headers = headers or {}
        resp.links = links or {}

        if json_payload is not None:
            resp.json.return_value = json_payload
            resp.text = text if text is not None else json.dumps(json_payload)
        else:
            resp.json.side_effect = ValueError("No JSON payload configured")
            resp.text = text or ""

        if status_code >= 400:
            http_error = requests.exceptions.HTTPError(
                f"{status_code} Error",
                response=resp,
            )
            resp.raise_for_status.side_effect = http_error
        else:
            resp.raise_for_status.return_value = None

        return resp

    return _factory


# ===========================================================================
# CVE data factories
# ===========================================================================


@pytest.fixture
def nvd_cve_factory():
    """Factory for building realistic NVD CVE response payloads.

    Returns a callable that produces a single NVD vulnerability entry::

        cve = nvd_cve_factory("CVE-2024-3094", base_score=10.0, severity="CRITICAL")
    """

    def _factory(
        cve_id: str = "CVE-2024-1234",
        *,
        base_score: float = 9.8,
        severity: str = "CRITICAL",
        attack_vector: str = "NETWORK",
        privileges_required: str = "NONE",
        user_interaction: str = "NONE",
        scope: str = "UNCHANGED",
        description: str = "A critical vulnerability.",
        cwe: str = "CWE-79",
        published: str = "2024-01-15T10:00:00.000",
        cpe_criteria: str | None = None,
    ) -> dict[str, Any]:
        cve_entry: dict[str, Any] = {
            "id": cve_id,
            "published": published,
            "descriptions": [{"lang": "en", "value": description}],
            "metrics": {
                "cvssMetricV31": [
                    {
                        "cvssData": {
                            "version": "3.1",
                            "baseScore": base_score,
                            "baseSeverity": severity,
                            "vectorString": (
                                f"CVSS:3.1/AV:{attack_vector[0]}/AC:L"
                                f"/PR:{privileges_required[0]}/UI:{user_interaction[0]}"
                                f"/S:{scope[0]}/C:H/I:H/A:H"
                            ),
                            "attackVector": attack_vector,
                            "privilegesRequired": privileges_required,
                            "userInteraction": user_interaction,
                            "scope": scope,
                        }
                    }
                ]
            },
            "weaknesses": [{"description": [{"lang": "en", "value": cwe}]}],
        }

        if cpe_criteria:
            cve_entry["configurations"] = [{"nodes": [{"cpeMatch": [{"criteria": cpe_criteria, "vulnerable": True}]}]}]

        return cve_entry

    return _factory


@pytest.fixture
def nvd_api_response(nvd_cve_factory):
    """Factory for building a full NVD API JSON response wrapping one or more CVEs.

    Returns a callable::

        payload = nvd_api_response("CVE-2024-3094")
        payload = nvd_api_response(cve_entries=[cve1, cve2])
    """

    def _factory(
        cve_id: str | None = None,
        *,
        cve_entries: list[dict] | None = None,
        **cve_kwargs,
    ) -> dict[str, Any]:
        if cve_entries is not None:
            vulns = [{"cve": entry} for entry in cve_entries]
        else:
            entry = nvd_cve_factory(cve_id or "CVE-2024-1234", **cve_kwargs)
            vulns = [{"cve": entry}]

        return {
            "resultsPerPage": len(vulns),
            "startIndex": 0,
            "totalResults": len(vulns),
            "vulnerabilities": vulns,
        }

    return _factory


@pytest.fixture
def epss_data_factory():
    """Factory for building EPSS API response payloads.

    Returns a callable::

        payload = epss_data_factory("CVE-2024-3094", epss=0.95, percentile=0.99)
    """

    def _factory(
        cve_id: str = "CVE-2024-1234",
        *,
        epss: float = 0.5,
        percentile: float = 0.85,
        date: str = "2024-06-01",
        time_series: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "cve": cve_id,
            "epss": str(epss),
            "percentile": str(percentile),
            "date": date,
        }
        if time_series is not None:
            entry["time-series"] = time_series

        return {
            "status": "OK",
            "status-code": 200,
            "version": "1.0",
            "total": 1,
            "offset": 0,
            "limit": 100,
            "data": [entry],
        }

    return _factory


@pytest.fixture
def kev_catalog_factory():
    """Factory for building a CISA KEV catalog payload.

    Returns a callable::

        catalog = kev_catalog_factory(["CVE-2021-44228", "CVE-2024-3094"])
        catalog = kev_catalog_factory()  # empty catalog
    """

    def _factory(
        cve_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        vulns = []
        for cve_id in cve_ids or []:
            vulns.append(
                {
                    "cveID": cve_id,
                    "vendorProject": "TestVendor",
                    "product": "TestProduct",
                    "vulnerabilityName": f"Test Vulnerability {cve_id}",
                    "dateAdded": "2024-01-01",
                    "dueDate": "2024-01-15",
                    "requiredAction": "Apply updates per vendor guidance.",
                    "knownRansomwareCampaignUse": "Unknown",
                }
            )

        return {
            "title": "CISA KEV Catalog",
            "catalogVersion": "2024.01.01",
            "dateReleased": "2024-01-01T00:00:00.000Z",
            "count": len(vulns),
            "vulnerabilities": vulns,
        }

    return _factory


@pytest.fixture
def osv_record_factory():
    """Factory for building OSV.dev vulnerability records.

    Returns a callable::

        record = osv_record_factory("CVE-2021-44228", ecosystem="Maven",
                                     package="org.apache.logging.log4j:log4j-core")
    """

    def _factory(
        cve_id: str = "CVE-2024-1234",
        *,
        osv_id: str | None = None,
        ecosystem: str = "PyPI",
        package: str = "example-package",
        introduced: str = "0",
        fixed: str = "2.0.0",
        summary: str = "A vulnerability.",
    ) -> dict[str, Any]:
        return {
            "id": osv_id or f"GHSA-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}",
            "aliases": [cve_id],
            "summary": summary,
            "modified": "2024-06-01T00:00:00Z",
            "published": "2024-01-15T00:00:00Z",
            "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}],
            "affected": [
                {
                    "package": {"ecosystem": ecosystem, "name": package},
                    "ranges": [
                        {
                            "type": "ECOSYSTEM",
                            "events": [{"introduced": introduced}, {"fixed": fixed}],
                        }
                    ],
                }
            ],
            "references": [{"type": "WEB", "url": f"https://nvd.nist.gov/vuln/detail/{cve_id}"}],
        }

    return _factory


# ===========================================================================
# Config fixtures
# ===========================================================================


@pytest.fixture
def default_config():
    """A default Config instance with no overrides (uses built-in defaults)."""
    from manus_agent.config import Config

    return Config()


@pytest.fixture
def bedrock_config():
    """A Config instance configured for AWS Bedrock."""
    from manus_agent.config import Config, LLMConfig

    return Config(llm=LLMConfig(provider="bedrock", model="us.anthropic.claude-sonnet-4-20250514-v1:0"))


@pytest.fixture
def openai_config():
    """A Config instance configured for OpenAI."""
    from manus_agent.config import Config, LLMConfig

    return Config(llm=LLMConfig(provider="openai", model="gpt-4o", api_key="sk-test-key"))


@pytest.fixture
def anthropic_config():
    """A Config instance configured for Anthropic."""
    from manus_agent.config import Config, LLMConfig

    return Config(llm=LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="test-key"))


@pytest.fixture
def ollama_config():
    """A Config instance configured for Ollama (local)."""
    from manus_agent.config import Config, LLMConfig

    return Config(llm=LLMConfig(provider="ollama", model="llama3", base_url="http://localhost:11434"))


# ===========================================================================
# Temporary file system fixtures
# ===========================================================================


@pytest.fixture
def tmp_history_file(tmp_path: Path):
    """Provides a temporary history.jsonl file path.

    Returns a tuple: ``(history_path, write_record_fn)``

    The *write_record_fn* appends a record and returns the path::

        history_path, write = tmp_history_file
        write(task="analyse CVE", success=True)
    """
    history_path = tmp_path / "history.jsonl"

    def _write_record(
        task: str = "test task",
        success: bool = True,
        agent: str = "manus",
        mode: str = "single",
        format: str = "text",
        result: str = "ok",
    ) -> Path:
        import datetime

        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "task": task,
            "agent": agent,
            "mode": mode,
            "format": format,
            "success": success,
            "result": result,
        }
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return history_path

    return history_path, _write_record


@pytest.fixture
def tmp_config_file(tmp_path: Path):
    """Provides a temporary config.toml file path.

    Returns a callable: ``tmp_config_file(content_dict)`` that writes TOML
    and returns the path::

        config_path = tmp_config_file({"llm": {"provider": "openai", "model": "gpt-4"}})
    """
    import toml

    config_path = tmp_path / "config.toml"

    def _write(content: dict[str, Any]) -> Path:
        config_path.write_text(toml.dumps(content), encoding="utf-8")
        return config_path

    return _write


# ===========================================================================
# Environment helpers
# ===========================================================================


@pytest.fixture
def env_override(monkeypatch):
    """Fixture for setting environment variables cleanly.

    Returns a callable: ``env_override(VAR_NAME="value", OTHER="value")``

    Variables are automatically restored after the test.

    Examples::

        env_override(NVD_API_KEY="test-key", VULNCHECK_API_KEY="vc-key")
    """

    def _set(**kwargs: str | None) -> None:
        for key, value in kwargs.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)

    return _set


# ===========================================================================
# Well-known CVE data constants (importable from conftest)
# ===========================================================================

# These are module-level constants that can be imported directly in tests
# that need pre-built CVE data without going through fixtures.

SAMPLE_CVE_LOG4SHELL = {
    "id": "CVE-2021-44228",
    "published": "2021-12-10T10:15:09.143",
    "descriptions": [{"lang": "en", "value": "Apache Log4j2 RCE vulnerability via JNDI lookup injection"}],
    "metrics": {
        "cvssMetricV31": [
            {
                "cvssData": {
                    "version": "3.1",
                    "baseScore": 10.0,
                    "baseSeverity": "CRITICAL",
                    "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                    "attackVector": "NETWORK",
                    "privilegesRequired": "NONE",
                    "userInteraction": "NONE",
                    "scope": "CHANGED",
                }
            }
        ]
    },
    "weaknesses": [{"description": [{"lang": "en", "value": "CWE-917"}]}],
    "configurations": [
        {
            "nodes": [
                {
                    "cpeMatch": [
                        {
                            "criteria": "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
                            "vulnerable": True,
                            "versionStartIncluding": "2.0-beta9",
                            "versionEndExcluding": "2.15.0",
                        }
                    ]
                }
            ]
        }
    ],
}

SAMPLE_CVE_XZ = {
    "id": "CVE-2024-3094",
    "published": "2024-03-29T17:15:21.940",
    "descriptions": [{"lang": "en", "value": "Malicious code in xz/liblzma leading to SSH auth bypass"}],
    "metrics": {
        "cvssMetricV31": [
            {
                "cvssData": {
                    "version": "3.1",
                    "baseScore": 10.0,
                    "baseSeverity": "CRITICAL",
                    "vectorString": "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:C/C:H/I:H/A:H",
                    "attackVector": "NETWORK",
                    "privilegesRequired": "NONE",
                    "userInteraction": "NONE",
                    "scope": "CHANGED",
                }
            }
        ]
    },
    "weaknesses": [{"description": [{"lang": "en", "value": "CWE-506"}]}],
    "configurations": [
        {
            "nodes": [
                {
                    "cpeMatch": [
                        {
                            "criteria": "cpe:2.3:a:tukaani:xz:*:*:*:*:*:*:*:*",
                            "vulnerable": True,
                            "versionStartIncluding": "5.6.0",
                            "versionEndIncluding": "5.6.1",
                        }
                    ]
                }
            ]
        }
    ],
}
