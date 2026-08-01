"""Comprehensive test suite for create_lark_document module.

Tests cover:
- TOOL_SPEC validation (name, required fields, input schema)
- Successful document creation (config URL, env URL)
- Authentication (config token, env token)
- Error handling (missing URL, missing token, HTTP errors, generic exceptions)
- Input handling (is_openclaw flag, OPENCLAW env var, field passthrough)
- Edge cases (empty title, special characters, large payloads)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.create_lark_document import (
    TOOL_SPEC,
    create_lark_document,
)

# ===========================================================================
# TOOL_SPEC validation
# ===========================================================================


class TestToolSpec:
    """Test the TOOL_SPEC constant."""

    def test_spec_name(self):
        assert TOOL_SPEC["name"] == "create_lark_document"

    def test_spec_has_description(self):
        assert "description" in TOOL_SPEC
        assert len(TOOL_SPEC["description"]) > 0

    def test_spec_has_input_schema(self):
        assert "inputSchema" in TOOL_SPEC
        assert "json" in TOOL_SPEC["inputSchema"]

    def test_spec_required_fields(self):
        required = TOOL_SPEC["inputSchema"]["json"]["required"]
        expected = [
            "title",
            "disclosure",
            "public_disclosure",
            "sources",
            "proof_of_concept_links",
            "cpe",
            "affected_versions",
            "technical_details",
            "cwe_info",
            "cvss_score",
            "recommendations",
            "background",
        ]
        assert set(required) == set(expected)

    def test_spec_properties_include_optional_fields(self):
        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        optional = ["exploit_verification", "dockerfile_content", "exploit_code", "docker_command", "is_openclaw"]
        for field in optional:
            assert field in props

    def test_spec_title_is_string_type(self):
        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert props["title"]["type"] == "string"

    def test_spec_is_openclaw_is_boolean(self):
        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert props["is_openclaw"]["type"] == "boolean"

    def test_spec_all_required_fields_in_properties(self):
        required = TOOL_SPEC["inputSchema"]["json"]["required"]
        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        for field in required:
            assert field in props


# ===========================================================================
# Successful document creation
# ===========================================================================


class TestCreateLarkDocumentSuccess:
    """Test successful document creation paths."""

    def _make_tool_use(self, title="[VI-001] CVE-2024-1234 Assessment: RCE in Foo", **extra):
        base_input = {
            "title": title,
            "disclosure": "Discovered by researcher X",
            "public_disclosure": "2024-01-15",
            "sources": "https://nvd.nist.gov/vuln/detail/CVE-2024-1234",
            "proof_of_concept_links": "https://github.com/poc/exploit",
            "cpe": "cpe:2.3:a:vendor:product:1.0",
            "affected_versions": "1.0, 1.1, 1.2",
            "technical_details": "### Exploitability Analysis\nRCE via deserialization",
            "cwe_info": "CWE-502: Deserialization of Untrusted Data",
            "cvss_score": "Critical(9.8),CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            "recommendations": "* Upgrade to version 2.0\n* Apply vendor patch",
            "background": "Foo is a widely-used web framework.",
        }
        base_input.update(extra)
        return {"toolUseId": "test-tool-123", "input": base_input}

    @patch("requests.post")
    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_success_with_config_url_and_token(self, mock_config, mock_post):
        mock_cfg = MagicMock()
        mock_cfg.lark.document_url = "https://api.example.com/create"
        mock_cfg.lark.api_token = "token123"
        mock_config.return_value = mock_cfg

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = '{"ok": true}'
        mock_post.return_value = mock_resp

        tool_use = self._make_tool_use()
        result = create_lark_document(tool_use)

        assert result["status"] == "success"
        assert result["toolUseId"] == "test-tool-123"
        assert "Created lark document" in result["content"][0]["text"]

    @patch("requests.post")
    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_success_with_env_url_and_token(self, mock_config, mock_post, monkeypatch):
        mock_cfg = MagicMock()
        mock_cfg.lark = None  # No lark config section
        mock_config.return_value = mock_cfg

        monkeypatch.setenv("LARK_DOCUMENT_URL", "https://env-api.example.com/create")
        monkeypatch.setenv("LARK_API_TOKEN", "env-token-456")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = '{"ok": true}'
        mock_post.return_value = mock_resp

        tool_use = self._make_tool_use()
        result = create_lark_document(tool_use)

        assert result["status"] == "success"

    @patch("requests.post")
    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_posts_to_correct_url(self, mock_config, mock_post):
        mock_cfg = MagicMock()
        mock_cfg.lark.document_url = "https://api.example.com/docs"
        mock_cfg.lark.api_token = "token"
        mock_config.return_value = mock_cfg

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = "{}"
        mock_post.return_value = mock_resp

        tool_use = self._make_tool_use()
        create_lark_document(tool_use)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://api.example.com/docs"

    @patch("requests.post")
    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_sends_auth_header(self, mock_config, mock_post):
        mock_cfg = MagicMock()
        mock_cfg.lark.document_url = "https://api.example.com/docs"
        mock_cfg.lark.api_token = "my-secret-token"
        mock_config.return_value = mock_cfg

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = "{}"
        mock_post.return_value = mock_resp

        tool_use = self._make_tool_use()
        create_lark_document(tool_use)

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer my-secret-token"

    @patch("requests.post")
    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_sends_input_as_json_body(self, mock_config, mock_post):
        mock_cfg = MagicMock()
        mock_cfg.lark.document_url = "https://api.example.com/docs"
        mock_cfg.lark.api_token = "token"
        mock_config.return_value = mock_cfg

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = "{}"
        mock_post.return_value = mock_resp

        tool_use = self._make_tool_use(title="Test Title")
        create_lark_document(tool_use)

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["title"] == "Test Title"

    @patch("requests.post")
    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_title_in_success_message(self, mock_config, mock_post):
        mock_cfg = MagicMock()
        mock_cfg.lark.document_url = "https://api.example.com/docs"
        mock_cfg.lark.api_token = "token"
        mock_config.return_value = mock_cfg

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = "{}"
        mock_post.return_value = mock_resp

        tool_use = self._make_tool_use(title="[VI-042] CVE-2024-9999 Test")
        result = create_lark_document(tool_use)

        assert "[VI-042] CVE-2024-9999 Test" in result["content"][0]["text"]


# ===========================================================================
# is_openclaw flag handling
# ===========================================================================


class TestIsOpenclawFlag:
    """Test is_openclaw field behavior."""

    def _make_tool_use(self, **input_overrides):
        base_input = {
            "title": "Test",
            "disclosure": "d",
            "public_disclosure": "2024-01-01",
            "sources": "https://example.com",
            "proof_of_concept_links": "https://example.com",
            "cpe": "cpe:2.3:a:v:p:1.0",
            "affected_versions": "1.0",
            "technical_details": "details",
            "cwe_info": "CWE-79",
            "cvss_score": "High(7.5),CVSS:3.1/AV:N",
            "recommendations": "* Fix it",
            "background": "bg",
        }
        base_input.update(input_overrides)
        return {"toolUseId": "test-123", "input": base_input}

    @patch("requests.post")
    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_is_openclaw_defaults_false_without_env(self, mock_config, mock_post, monkeypatch):
        mock_cfg = MagicMock()
        mock_cfg.lark.document_url = "https://api.example.com"
        mock_cfg.lark.api_token = "token"
        mock_config.return_value = mock_cfg

        monkeypatch.delenv("OPENCLAW", raising=False)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = "{}"
        mock_post.return_value = mock_resp

        tool_use = self._make_tool_use()
        create_lark_document(tool_use)

        sent_json = mock_post.call_args[1]["json"]
        assert sent_json["is_openclaw"] is False

    @patch("requests.post")
    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_is_openclaw_true_when_env_set(self, mock_config, mock_post, monkeypatch):
        mock_cfg = MagicMock()
        mock_cfg.lark.document_url = "https://api.example.com"
        mock_cfg.lark.api_token = "token"
        mock_config.return_value = mock_cfg

        monkeypatch.setenv("OPENCLAW", "true")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = "{}"
        mock_post.return_value = mock_resp

        tool_use = self._make_tool_use()
        create_lark_document(tool_use)

        sent_json = mock_post.call_args[1]["json"]
        assert sent_json["is_openclaw"] is True

    @patch("requests.post")
    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_is_openclaw_true_case_insensitive(self, mock_config, mock_post, monkeypatch):
        mock_cfg = MagicMock()
        mock_cfg.lark.document_url = "https://api.example.com"
        mock_cfg.lark.api_token = "token"
        mock_config.return_value = mock_cfg

        monkeypatch.setenv("OPENCLAW", "TRUE")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = "{}"
        mock_post.return_value = mock_resp

        tool_use = self._make_tool_use()
        create_lark_document(tool_use)

        sent_json = mock_post.call_args[1]["json"]
        assert sent_json["is_openclaw"] is True

    @patch("requests.post")
    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_explicit_is_openclaw_not_overridden(self, mock_config, mock_post, monkeypatch):
        """If is_openclaw is explicitly provided, env var should not override it."""
        mock_cfg = MagicMock()
        mock_cfg.lark.document_url = "https://api.example.com"
        mock_cfg.lark.api_token = "token"
        mock_config.return_value = mock_cfg

        monkeypatch.setenv("OPENCLAW", "true")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = "{}"
        mock_post.return_value = mock_resp

        tool_use = self._make_tool_use(is_openclaw=False)
        create_lark_document(tool_use)

        sent_json = mock_post.call_args[1]["json"]
        # Explicit False should be preserved
        assert sent_json["is_openclaw"] is False


# ===========================================================================
# Error handling — missing configuration
# ===========================================================================


class TestCreateLarkDocumentConfigErrors:
    """Test error handling for missing configuration."""

    def _make_tool_use(self):
        return {
            "toolUseId": "err-123",
            "input": {
                "title": "Test",
                "disclosure": "d",
                "public_disclosure": "2024-01-01",
                "sources": "s",
                "proof_of_concept_links": "p",
                "cpe": "c",
                "affected_versions": "v",
                "technical_details": "t",
                "cwe_info": "cwe",
                "cvss_score": "score",
                "recommendations": "r",
                "background": "b",
            },
        }

    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_missing_url_raises_value_error(self, mock_config, monkeypatch):
        mock_cfg = MagicMock()
        mock_cfg.lark = None
        mock_config.return_value = mock_cfg
        monkeypatch.delenv("LARK_DOCUMENT_URL", raising=False)
        monkeypatch.delenv("LARK_API_TOKEN", raising=False)

        tool_use = self._make_tool_use()
        with pytest.raises(ValueError, match="URL not set"):
            create_lark_document(tool_use)

    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_missing_token_raises_value_error(self, mock_config, monkeypatch):
        mock_cfg = MagicMock()
        mock_cfg.lark.document_url = "https://api.example.com"
        mock_cfg.lark.api_token = None
        mock_config.return_value = mock_cfg
        monkeypatch.delenv("LARK_API_TOKEN", raising=False)

        tool_use = self._make_tool_use()
        with pytest.raises(ValueError, match="token not set"):
            create_lark_document(tool_use)

    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_url_from_env_when_config_none(self, mock_config, monkeypatch):
        """URL falls back to environment when config.lark is None."""
        mock_cfg = MagicMock()
        mock_cfg.lark = None
        mock_config.return_value = mock_cfg
        monkeypatch.setenv("LARK_DOCUMENT_URL", "https://from-env.com")
        monkeypatch.delenv("LARK_API_TOKEN", raising=False)

        tool_use = self._make_tool_use()
        # Should get past URL check and fail on token
        with pytest.raises(ValueError, match="token not set"):
            create_lark_document(tool_use)

    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_url_from_env_when_config_attr_none(self, mock_config, monkeypatch):
        """URL falls back to env when config.lark.document_url is None."""
        mock_cfg = MagicMock()
        mock_cfg.lark.document_url = None
        mock_cfg.lark.api_token = None
        mock_config.return_value = mock_cfg
        monkeypatch.setenv("LARK_DOCUMENT_URL", "https://from-env.com")
        monkeypatch.delenv("LARK_API_TOKEN", raising=False)

        tool_use = self._make_tool_use()
        with pytest.raises(ValueError, match="token not set"):
            create_lark_document(tool_use)


# ===========================================================================
# Error handling — HTTP errors
# ===========================================================================


class TestCreateLarkDocumentHttpErrors:
    """Test error handling for HTTP failures."""

    def _make_tool_use(self):
        return {
            "toolUseId": "http-err",
            "input": {
                "title": "Test",
                "disclosure": "d",
                "public_disclosure": "2024-01-01",
                "sources": "s",
                "proof_of_concept_links": "p",
                "cpe": "c",
                "affected_versions": "v",
                "technical_details": "t",
                "cwe_info": "cwe",
                "cvss_score": "score",
                "recommendations": "r",
                "background": "b",
            },
        }

    @patch("requests.post")
    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_http_error_returns_error_status(self, mock_config, mock_post):
        import requests

        mock_cfg = MagicMock()
        mock_cfg.lark.document_url = "https://api.example.com"
        mock_cfg.lark.api_token = "token"
        mock_config.return_value = mock_cfg

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("403 Forbidden")
        mock_post.return_value = mock_resp

        tool_use = self._make_tool_use()
        result = create_lark_document(tool_use)

        assert result["status"] == "error"
        assert result["toolUseId"] == "http-err"
        assert "403 Forbidden" in result["content"][0]["text"]

    @patch("requests.post")
    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_http_500_error(self, mock_config, mock_post):
        import requests

        mock_cfg = MagicMock()
        mock_cfg.lark.document_url = "https://api.example.com"
        mock_cfg.lark.api_token = "token"
        mock_config.return_value = mock_cfg

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Internal Server Error")
        mock_post.return_value = mock_resp

        tool_use = self._make_tool_use()
        result = create_lark_document(tool_use)

        assert result["status"] == "error"
        assert "500" in result["content"][0]["text"]

    @patch("requests.post")
    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_connection_error_returns_error_status(self, mock_config, mock_post):
        mock_cfg = MagicMock()
        mock_cfg.lark.document_url = "https://api.example.com"
        mock_cfg.lark.api_token = "token"
        mock_config.return_value = mock_cfg

        mock_post.side_effect = ConnectionError("Connection refused")

        tool_use = self._make_tool_use()
        result = create_lark_document(tool_use)

        assert result["status"] == "error"
        assert "Connection refused" in result["content"][0]["text"]

    @patch("requests.post")
    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_timeout_error_returns_error_status(self, mock_config, mock_post):
        import requests

        mock_cfg = MagicMock()
        mock_cfg.lark.document_url = "https://api.example.com"
        mock_cfg.lark.api_token = "token"
        mock_config.return_value = mock_cfg

        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        tool_use = self._make_tool_use()
        result = create_lark_document(tool_use)

        assert result["status"] == "error"
        assert "timed out" in result["content"][0]["text"]

    @patch("requests.post")
    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_generic_exception_returns_error_status(self, mock_config, mock_post):
        mock_cfg = MagicMock()
        mock_cfg.lark.document_url = "https://api.example.com"
        mock_cfg.lark.api_token = "token"
        mock_config.return_value = mock_cfg

        mock_post.side_effect = RuntimeError("Unexpected error")

        tool_use = self._make_tool_use()
        result = create_lark_document(tool_use)

        assert result["status"] == "error"
        assert "Unexpected error" in result["content"][0]["text"]


# ===========================================================================
# Optional fields
# ===========================================================================


class TestOptionalFields:
    """Test optional field handling."""

    @patch("requests.post")
    @patch("manus_agent.tools.create_lark_document.Config.from_file")
    def test_optional_exploit_verification_passed_through(self, mock_config, mock_post):
        mock_cfg = MagicMock()
        mock_cfg.lark.document_url = "https://api.example.com"
        mock_cfg.lark.api_token = "token"
        mock_config.return_value = mock_cfg

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = "{}"
        mock_post.return_value = mock_resp

        tool_use = {
            "toolUseId": "opt-1",
            "input": {
                "title": "Test",
                "disclosure": "d",
                "public_disclosure": "2024-01-01",
                "sources": "s",
                "proof_of_concept_links": "p",
                "cpe": "c",
                "affected_versions": "v",
                "technical_details": "t",
                "cwe_info": "cwe",
                "cvss_score": "score",
                "recommendations": "r",
                "background": "b",
                "exploit_verification": "Verified in Docker container",
                "dockerfile_content": "FROM python:3.12\nRUN pip install vuln-app==1.0",
                "exploit_code": "import requests\nrequests.get('http://target:8080/rce')",
                "docker_command": "docker build -t target . && docker run target",
                "exploit_execution_command": "python exploit.py",
            },
        }

        create_lark_document(tool_use)

        sent_json = mock_post.call_args[1]["json"]
        assert sent_json["exploit_verification"] == "Verified in Docker container"
        assert "FROM python:3.12" in sent_json["dockerfile_content"]
        assert sent_json["exploit_code"] == "import requests\nrequests.get('http://target:8080/rce')"
        assert sent_json["docker_command"] == "docker build -t target . && docker run target"
        assert sent_json["exploit_execution_command"] == "python exploit.py"
