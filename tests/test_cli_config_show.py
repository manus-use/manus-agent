"""Comprehensive tests for the ``manus-agent config show`` subcommand.

Covers:
  - parser construction and defaults
  - secret redaction (``_redact``)
  - config-path resolution (``_resolve_config_path``)
  - section-to-dict conversion with and without redaction
  - full-config-to-dict conversion
  - source-label detection (config / env / default)
  - Rich text output (``_print_text``)
  - end-to-end ``_cmd_config_show`` in text and JSON modes
  - ``--section`` filtering (valid and invalid)
  - ``--reveal`` flag
  - integration through ``main()`` dispatch
  - error handling for invalid config files
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.cli import (
    _build_config_show_parser,
    _cmd_config_show,
    _config_to_dict,
    _redact,
    _resolve_config_path,
    _SECRET_FIELD_NAMES,
    _section_to_dict,
    _source_label,
)
from manus_agent.config import (
    AgentConfig,
    BrowserUseConfig,
    Config,
    GitHubConfig,
    LarkConfig,
    LLMConfig,
    MCPConfig,
    OTXConfig,
    SandboxConfig,
    ToolsConfig,
    WebhooksConfig,
)


# ===================================================================
# _build_config_show_parser
# ===================================================================


class TestBuildConfigShowParser:
    """Tests for ``_build_config_show_parser``."""

    def test_returns_parser(self) -> None:
        parser = _build_config_show_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_default_output_text(self) -> None:
        parser = _build_config_show_parser()
        args = parser.parse_args([])
        assert args.output == "text"

    def test_output_json(self) -> None:
        parser = _build_config_show_parser()
        args = parser.parse_args(["--output", "json"])
        assert args.output == "json"

    def test_reveal_default_false(self) -> None:
        parser = _build_config_show_parser()
        args = parser.parse_args([])
        assert args.reveal is False

    def test_reveal_flag(self) -> None:
        parser = _build_config_show_parser()
        args = parser.parse_args(["--reveal"])
        assert args.reveal is True

    def test_section_default_none(self) -> None:
        parser = _build_config_show_parser()
        args = parser.parse_args([])
        assert args.section is None

    def test_section_flag(self) -> None:
        parser = _build_config_show_parser()
        args = parser.parse_args(["--section", "llm"])
        assert args.section == "llm"

    def test_config_default_none(self) -> None:
        parser = _build_config_show_parser()
        args = parser.parse_args([])
        assert args.config is None

    def test_config_path(self, tmp_path: Path) -> None:
        parser = _build_config_show_parser()
        p = tmp_path / "config.toml"
        args = parser.parse_args(["--config", str(p)])
        assert args.config == p

    def test_invalid_output_rejected(self) -> None:
        parser = _build_config_show_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--output", "xml"])


# ===================================================================
# _redact
# ===================================================================


class TestRedact:
    """Tests for ``_redact``."""

    def test_short_value(self) -> None:
        assert _redact("abc") == "****"

    def test_exactly_four(self) -> None:
        assert _redact("abcd") == "****"

    def test_five_chars(self) -> None:
        assert _redact("abcde") == "abcd*"

    def test_long_value(self) -> None:
        result = _redact("sk-1234567890abcdef")
        assert result.startswith("sk-1")
        assert "*" in result
        # First 4 chars preserved, rest masked up to 12 stars
        assert result == "sk-1" + "*" * 12

    def test_medium_value(self) -> None:
        result = _redact("abcdefgh")
        assert result == "abcd****"

    def test_empty_string(self) -> None:
        assert _redact("") == "****"


# ===================================================================
# _resolve_config_path
# ===================================================================


class TestResolveConfigPath:
    """Tests for ``_resolve_config_path``."""

    def test_explicit_existing(self, tmp_path: Path) -> None:
        p = tmp_path / "config.toml"
        p.write_text("[llm]\nprovider = 'openai'\n")
        assert _resolve_config_path(p) == p

    def test_explicit_missing(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.toml"
        assert _resolve_config_path(p) is None

    def test_none_no_files(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir("/tmp")  # noqa: S108 — no config files here
        # Ensure none of the default paths exist from cwd
        result = _resolve_config_path(None)
        # May or may not find ~/.manus-agent/config.toml depending on env
        # Just verify it doesn't crash
        assert result is None or isinstance(result, Path)

    def test_none_with_local_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        p = tmp_path / "config.toml"
        p.write_text("[llm]\nprovider = 'bedrock'\n")
        monkeypatch.chdir(tmp_path)
        result = _resolve_config_path(None)
        assert result is not None
        assert result.resolve() == p.resolve()


# ===================================================================
# _section_to_dict
# ===================================================================


class TestSectionToDict:
    """Tests for ``_section_to_dict``."""

    def test_basic_section(self) -> None:
        section = LLMConfig(provider="openai", model="gpt-4o")
        result = _section_to_dict(section, reveal=True)
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-4o"
        assert "temperature" in result
        assert "max_tokens" in result

    def test_secret_redacted(self) -> None:
        section = LLMConfig(api_key="sk-1234567890abcdef")
        result = _section_to_dict(section, reveal=False)
        assert result["api_key"] != "sk-1234567890abcdef"
        assert result["api_key"].startswith("sk-1")
        assert "*" in result["api_key"]

    def test_secret_revealed(self) -> None:
        section = LLMConfig(api_key="sk-1234567890abcdef")
        result = _section_to_dict(section, reveal=True)
        assert result["api_key"] == "sk-1234567890abcdef"

    def test_none_secret_not_redacted(self) -> None:
        section = LLMConfig(api_key=None)
        result = _section_to_dict(section, reveal=False)
        assert result["api_key"] is None

    def test_non_secret_fields_unchanged(self) -> None:
        section = LLMConfig(provider="bedrock", model="claude-3")
        result = _section_to_dict(section, reveal=False)
        assert result["provider"] == "bedrock"
        assert result["model"] == "claude-3"

    def test_sandbox_section(self) -> None:
        section = SandboxConfig(enabled=False, timeout=600)
        result = _section_to_dict(section, reveal=False)
        assert result["enabled"] is False
        assert result["timeout"] == 600

    def test_otx_api_key_redacted(self) -> None:
        section = OTXConfig(api_key="mysecretkey123")
        result = _section_to_dict(section, reveal=False)
        assert "****" in result["api_key"]

    def test_github_token_redacted(self) -> None:
        section = GitHubConfig(api_token="ghp_supersecrettoken123")
        result = _section_to_dict(section, reveal=False)
        assert "****" in result["api_token"]

    def test_lark_fields_redacted(self) -> None:
        section = LarkConfig(
            api_token="lark-token-123",
            document_url="https://lark.example.com/doc/123",
        )
        result = _section_to_dict(section, reveal=False)
        assert "*" in result["api_token"]
        assert "*" in result["document_url"]

    def test_webhooks_url_redacted(self) -> None:
        section = WebhooksConfig(cve_submit_url="https://webhook.example.com/submit")
        result = _section_to_dict(section, reveal=False)
        assert "*" in result["cve_submit_url"]

    def test_mcp_server_url_redacted(self) -> None:
        section = MCPConfig(server_url="https://mcp.example.com")
        result = _section_to_dict(section, reveal=False)
        assert "*" in result["server_url"]


# ===================================================================
# _config_to_dict
# ===================================================================


class TestConfigToDict:
    """Tests for ``_config_to_dict``."""

    def test_all_sections_present(self) -> None:
        config = Config()
        data = _config_to_dict(config, reveal=True)
        for section_name in type(config).model_fields:
            assert section_name in data

    def test_returns_nested_dicts(self) -> None:
        config = Config()
        data = _config_to_dict(config, reveal=True)
        for _name, section_data in data.items():
            assert isinstance(section_data, dict)

    def test_redaction_applied(self) -> None:
        config = Config(llm=LLMConfig(api_key="sk-test123456"))
        data = _config_to_dict(config, reveal=False)
        assert data["llm"]["api_key"] != "sk-test123456"
        assert "*" in data["llm"]["api_key"]

    def test_reveal_applied(self) -> None:
        config = Config(llm=LLMConfig(api_key="sk-test123456"))
        data = _config_to_dict(config, reveal=True)
        assert data["llm"]["api_key"] == "sk-test123456"


# ===================================================================
# _source_label
# ===================================================================


class TestSourceLabel:
    """Tests for ``_source_label``."""

    def test_from_config_file(self) -> None:
        config = Config(llm=LLMConfig(provider="bedrock"))
        file_data: dict[str, dict[str, Any]] = {"llm": {"provider": "bedrock"}}
        assert _source_label("llm", "provider", file_data, config) == "config"

    def test_from_env(self) -> None:
        config = Config(llm=LLMConfig(provider="bedrock"))
        file_data: dict[str, dict[str, Any]] = {}
        assert _source_label("llm", "provider", file_data, config) == "env"

    def test_default(self) -> None:
        config = Config()
        file_data: dict[str, dict[str, Any]] = {}
        # provider defaults to "openai" → should show as default
        assert _source_label("llm", "provider", file_data, config) == "default"

    def test_default_with_empty_file_section(self) -> None:
        config = Config()
        file_data: dict[str, dict[str, Any]] = {"llm": {}}
        assert _source_label("llm", "temperature", file_data, config) == "default"

    def test_missing_section(self) -> None:
        config = Config()
        file_data: dict[str, dict[str, Any]] = {}
        # Section not in file data at all
        assert _source_label("sandbox", "enabled", file_data, config) == "default"

    def test_api_key_from_config(self) -> None:
        config = Config(otx=OTXConfig(api_key="mykey"))
        file_data: dict[str, dict[str, Any]] = {"otx": {"api_key": "mykey"}}
        assert _source_label("otx", "api_key", file_data, config) == "config"

    def test_api_key_from_env(self) -> None:
        config = Config(otx=OTXConfig(api_key="mykey"))
        file_data: dict[str, dict[str, Any]] = {}
        assert _source_label("otx", "api_key", file_data, config) == "env"


# ===================================================================
# _cmd_config_show — text output
# ===================================================================


class TestCmdConfigShowText:
    """Tests for ``_cmd_config_show`` with text output."""

    @patch("manus_agent.cli.Config.from_file")
    @patch("manus_agent.cli._resolve_config_path")
    def test_text_output_no_config_file(
        self,
        mock_resolve: MagicMock,
        mock_from_file: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_resolve.return_value = None
        mock_from_file.return_value = Config()
        args = argparse.Namespace(
            config=None, output="text", reveal=False, section=None,
        )
        rc = _cmd_config_show(args)
        assert rc == 0

    @patch("manus_agent.cli.Config.from_file")
    @patch("manus_agent.cli._resolve_config_path")
    def test_text_output_with_config_file(
        self,
        mock_resolve: MagicMock,
        mock_from_file: MagicMock,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "config.toml"
        p.write_text("[llm]\nprovider = 'bedrock'\n")
        mock_resolve.return_value = p
        mock_from_file.return_value = Config(llm=LLMConfig(provider="bedrock"))
        args = argparse.Namespace(
            config=p, output="text", reveal=False, section=None,
        )
        rc = _cmd_config_show(args)
        assert rc == 0


# ===================================================================
# _cmd_config_show — JSON output
# ===================================================================


class TestCmdConfigShowJson:
    """Tests for ``_cmd_config_show`` with JSON output."""

    @patch("manus_agent.cli.Config.from_file")
    @patch("manus_agent.cli._resolve_config_path")
    def test_json_output_valid(
        self,
        mock_resolve: MagicMock,
        mock_from_file: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_resolve.return_value = None
        mock_from_file.return_value = Config()
        args = argparse.Namespace(
            config=None, output="json", reveal=False, section=None,
        )
        rc = _cmd_config_show(args)
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "llm" in data
        assert "sandbox" in data
        assert "tools" in data

    @patch("manus_agent.cli.Config.from_file")
    @patch("manus_agent.cli._resolve_config_path")
    def test_json_output_all_sections(
        self,
        mock_resolve: MagicMock,
        mock_from_file: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_resolve.return_value = None
        mock_from_file.return_value = Config()
        args = argparse.Namespace(
            config=None, output="json", reveal=False, section=None,
        )
        _cmd_config_show(args)
        data = json.loads(capsys.readouterr().out)
        expected_sections = set(type(Config()).model_fields.keys())
        assert set(data.keys()) == expected_sections

    @patch("manus_agent.cli.Config.from_file")
    @patch("manus_agent.cli._resolve_config_path")
    def test_json_secrets_redacted(
        self,
        mock_resolve: MagicMock,
        mock_from_file: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_resolve.return_value = None
        mock_from_file.return_value = Config(
            llm=LLMConfig(api_key="sk-supersecretkey"),
        )
        args = argparse.Namespace(
            config=None, output="json", reveal=False, section=None,
        )
        _cmd_config_show(args)
        data = json.loads(capsys.readouterr().out)
        assert data["llm"]["api_key"] != "sk-supersecretkey"
        assert data["llm"]["api_key"].startswith("sk-s")

    @patch("manus_agent.cli.Config.from_file")
    @patch("manus_agent.cli._resolve_config_path")
    def test_json_secrets_revealed(
        self,
        mock_resolve: MagicMock,
        mock_from_file: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_resolve.return_value = None
        mock_from_file.return_value = Config(
            llm=LLMConfig(api_key="sk-supersecretkey"),
        )
        args = argparse.Namespace(
            config=None, output="json", reveal=True, section=None,
        )
        _cmd_config_show(args)
        data = json.loads(capsys.readouterr().out)
        assert data["llm"]["api_key"] == "sk-supersecretkey"


# ===================================================================
# _cmd_config_show — section filtering
# ===================================================================


class TestCmdConfigShowSection:
    """Tests for ``_cmd_config_show`` with ``--section``."""

    @patch("manus_agent.cli.Config.from_file")
    @patch("manus_agent.cli._resolve_config_path")
    def test_section_llm(
        self,
        mock_resolve: MagicMock,
        mock_from_file: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_resolve.return_value = None
        mock_from_file.return_value = Config()
        args = argparse.Namespace(
            config=None, output="json", reveal=False, section="llm",
        )
        rc = _cmd_config_show(args)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert set(data.keys()) == {"llm"}

    @patch("manus_agent.cli.Config.from_file")
    @patch("manus_agent.cli._resolve_config_path")
    def test_section_sandbox(
        self,
        mock_resolve: MagicMock,
        mock_from_file: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_resolve.return_value = None
        mock_from_file.return_value = Config()
        args = argparse.Namespace(
            config=None, output="json", reveal=False, section="sandbox",
        )
        rc = _cmd_config_show(args)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "sandbox" in data
        assert "enabled" in data["sandbox"]

    @patch("manus_agent.cli.Config.from_file")
    @patch("manus_agent.cli._resolve_config_path")
    def test_section_with_hyphen(
        self,
        mock_resolve: MagicMock,
        mock_from_file: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_resolve.return_value = None
        mock_from_file.return_value = Config()
        args = argparse.Namespace(
            config=None, output="json", reveal=False, section="browser-use",
        )
        rc = _cmd_config_show(args)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "browser_use" in data

    @patch("manus_agent.cli.Config.from_file")
    @patch("manus_agent.cli._resolve_config_path")
    def test_invalid_section(
        self,
        mock_resolve: MagicMock,
        mock_from_file: MagicMock,
    ) -> None:
        mock_resolve.return_value = None
        mock_from_file.return_value = Config()
        args = argparse.Namespace(
            config=None, output="text", reveal=False, section="nonexistent",
        )
        rc = _cmd_config_show(args)
        assert rc == 1


# ===================================================================
# _cmd_config_show — error handling
# ===================================================================


class TestCmdConfigShowErrors:
    """Tests for ``_cmd_config_show`` error handling."""

    @patch("manus_agent.cli.Config.from_file", side_effect=Exception("parse error"))
    @patch("manus_agent.cli._resolve_config_path")
    def test_config_load_error(
        self,
        mock_resolve: MagicMock,
        mock_from_file: MagicMock,
    ) -> None:
        mock_resolve.return_value = None
        args = argparse.Namespace(
            config=None, output="text", reveal=False, section=None,
        )
        rc = _cmd_config_show(args)
        assert rc == 1


# ===================================================================
# main() dispatch — integration
# ===================================================================


class TestMainDispatch:
    """Tests for ``config show`` dispatch through ``main()``."""

    @patch("manus_agent.cli.Config.from_file")
    @patch("manus_agent.cli._resolve_config_path")
    def test_config_show_dispatch(
        self,
        mock_resolve: MagicMock,
        mock_from_file: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_resolve.return_value = None
        mock_from_file.return_value = Config()
        from manus_agent.cli import main

        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["manus-agent", "config", "show", "--output", "json"]):
                main()
        assert exc_info.value.code == 0
        data = json.loads(capsys.readouterr().out)
        assert "llm" in data

    @patch("manus_agent.cli.Config.from_file")
    @patch("manus_agent.cli._resolve_config_path")
    def test_config_without_show_subcommand(
        self,
        mock_resolve: MagicMock,
        mock_from_file: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``manus-agent config`` should default to ``config show``."""
        mock_resolve.return_value = None
        mock_from_file.return_value = Config()
        from manus_agent.cli import main

        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["manus-agent", "config", "--output", "json"]):
                main()
        assert exc_info.value.code == 0
        data = json.loads(capsys.readouterr().out)
        assert "llm" in data

    @patch("manus_agent.cli.Config.from_file")
    @patch("manus_agent.cli._resolve_config_path")
    def test_config_show_with_section(
        self,
        mock_resolve: MagicMock,
        mock_from_file: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_resolve.return_value = None
        mock_from_file.return_value = Config()
        from manus_agent.cli import main

        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["manus-agent", "config", "show", "--section", "tools", "--output", "json"]):
                main()
        assert exc_info.value.code == 0
        data = json.loads(capsys.readouterr().out)
        assert set(data.keys()) == {"tools"}

    @patch("manus_agent.cli.Config.from_file")
    @patch("manus_agent.cli._resolve_config_path")
    def test_config_show_with_reveal(
        self,
        mock_resolve: MagicMock,
        mock_from_file: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_resolve.return_value = None
        mock_from_file.return_value = Config(
            github=GitHubConfig(api_token="ghp_test123456"),
        )
        from manus_agent.cli import main

        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["manus-agent", "config", "show", "--reveal", "--output", "json"]):
                main()
        assert exc_info.value.code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["github"]["api_token"] == "ghp_test123456"


# ===================================================================
# _SECRET_FIELD_NAMES — coverage
# ===================================================================


class TestSecretFieldNames:
    """Verify the secret field set is complete."""

    def test_expected_secrets(self) -> None:
        assert "api_key" in _SECRET_FIELD_NAMES
        assert "api_token" in _SECRET_FIELD_NAMES
        assert "cve_submit_url" in _SECRET_FIELD_NAMES
        assert "document_url" in _SECRET_FIELD_NAMES
        assert "server_url" in _SECRET_FIELD_NAMES

    def test_is_frozenset(self) -> None:
        assert isinstance(_SECRET_FIELD_NAMES, frozenset)


# ===================================================================
# End-to-end with real config file
# ===================================================================


class TestEndToEndWithConfigFile:
    """Integration tests using actual TOML config files."""

    def test_json_with_real_config(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "[llm]\n"
            'provider = "bedrock"\n'
            'model = "anthropic.claude-3-5-sonnet"\n'
            "\n"
            "[sandbox]\n"
            "timeout = 600\n"
            "\n"
            "[otx]\n"
            'api_key = "my-otx-key"\n'
        )
        args = argparse.Namespace(
            config=config_file, output="json", reveal=False, section=None,
        )
        rc = _cmd_config_show(args)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["llm"]["provider"] == "bedrock"
        assert data["llm"]["model"] == "anthropic.claude-3-5-sonnet"
        assert data["sandbox"]["timeout"] == 600
        # OTX key should be redacted
        assert data["otx"]["api_key"] != "my-otx-key"
        assert data["otx"]["api_key"].startswith("my-o")

    def test_json_with_real_config_revealed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "[github]\n"
            'api_token = "ghp_realtokenvalue"\n'
        )
        args = argparse.Namespace(
            config=config_file, output="json", reveal=True, section="github",
        )
        rc = _cmd_config_show(args)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["github"]["api_token"] == "ghp_realtokenvalue"

    def test_text_with_real_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "[llm]\n"
            'provider = "openai"\n'
            'model = "gpt-4o-mini"\n'
        )
        args = argparse.Namespace(
            config=config_file, output="text", reveal=False, section="llm",
        )
        rc = _cmd_config_show(args)
        assert rc == 0

    def test_malformed_toml_falls_back(
        self, tmp_path: Path,
    ) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text("this is not valid toml {{{{")
        args = argparse.Namespace(
            config=config_file, output="text", reveal=False, section=None,
        )
        rc = _cmd_config_show(args)
        # Should return 1 since Config.from_file will fail
        assert rc == 1


# ===================================================================
# Environment variable source detection
# ===================================================================


class TestEnvVarSourceDetection:
    """Tests for source labeling when env vars override defaults."""

    def test_env_var_overrides_show_env_source(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Set an env var that overrides a default
        monkeypatch.setenv("MANUS_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("MANUS_LLM_MODEL", "claude-3-5-sonnet")

        args = argparse.Namespace(
            config=None, output="json", reveal=False, section="llm",
        )
        rc = _cmd_config_show(args)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["llm"]["provider"] == "anthropic"
        assert data["llm"]["model"] == "claude-3-5-sonnet"


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_list_field_rendering(self) -> None:
        section = ToolsConfig(enabled=["web_search", "code_execute"])
        result = _section_to_dict(section, reveal=False)
        assert result["enabled"] == ["web_search", "code_execute"]

    def test_empty_list_rendering(self) -> None:
        section = ToolsConfig(enabled=[])
        result = _section_to_dict(section, reveal=False)
        assert result["enabled"] == []

    def test_boolean_field_rendering(self) -> None:
        section = SandboxConfig(enabled=True)
        result = _section_to_dict(section, reveal=False)
        assert result["enabled"] is True

    def test_float_field_rendering(self) -> None:
        section = LLMConfig(temperature=0.7)
        result = _section_to_dict(section, reveal=False)
        assert result["temperature"] == 0.7

    def test_browser_use_many_fields(self) -> None:
        """BrowserUseConfig has the most fields — verify all present."""
        section = BrowserUseConfig()
        result = _section_to_dict(section, reveal=False)
        expected_fields = set(type(section).model_fields.keys())
        assert set(result.keys()) == expected_fields

    def test_agent_config(self) -> None:
        section = AgentConfig(context_manager="agentic")
        result = _section_to_dict(section, reveal=False)
        assert result["context_manager"] == "agentic"
