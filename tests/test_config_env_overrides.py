"""Comprehensive tests for Config._apply_env_overrides() and Config.from_file().

Tests the environment-variable → config mapping pipeline, including:
- MANUS_LLM_* overrides (always-override semantics)
- Provider-specific API key resolution (fill-when-None semantics)
- AWS region resolution from multiple env var names
- Integration config env vars (OTX, GitHub, Lark, Webhooks, MCP)
- .env file loading and search-path resolution
- from_file() search-path discovery
- Priority: env vars > config.toml > pydantic defaults
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest
import toml

from manus_agent.config import Config, LLMConfig, _load_dotenv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**llm_kwargs) -> Config:
    """Build a Config with explicit LLM settings (skipping env override via direct construction)."""
    return Config.model_construct(
        llm=LLMConfig(**llm_kwargs),
        sandbox=Config.model_fields["sandbox"].default_factory(),
        tools=Config.model_fields["tools"].default_factory(),
        browser_use=Config.model_fields["browser_use"].default_factory(),
        otx=Config.model_fields["otx"].default_factory(),
        github=Config.model_fields["github"].default_factory(),
        mcp=Config.model_fields["mcp"].default_factory(),
        webhooks=Config.model_fields["webhooks"].default_factory(),
        lark=Config.model_fields["lark"].default_factory(),
        agent=Config.model_fields["agent"].default_factory(),
    )


# ---------------------------------------------------------------------------
# MANUS_LLM_PROVIDER override
# ---------------------------------------------------------------------------


class TestLLMProviderOverride:
    """MANUS_LLM_PROVIDER always overrides the configured provider."""

    def test_overrides_default_provider(self):
        with mock.patch.dict(os.environ, {"MANUS_LLM_PROVIDER": "anthropic"}, clear=False):
            cfg = Config()
        assert cfg.llm.provider == "anthropic"

    def test_overrides_explicit_provider(self):
        with mock.patch.dict(os.environ, {"MANUS_LLM_PROVIDER": "bedrock"}, clear=False):
            cfg = Config(llm=LLMConfig(provider="openai"))
        assert cfg.llm.provider == "bedrock"

    def test_no_override_when_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "MANUS_LLM_PROVIDER"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Config(llm=LLMConfig(provider="ollama"))
        assert cfg.llm.provider == "ollama"

    def test_empty_string_does_not_override(self):
        with mock.patch.dict(os.environ, {"MANUS_LLM_PROVIDER": ""}, clear=False):
            cfg = Config(llm=LLMConfig(provider="openai"))
        assert cfg.llm.provider == "openai"


# ---------------------------------------------------------------------------
# MANUS_LLM_MODEL override
# ---------------------------------------------------------------------------


class TestLLMModelOverride:
    """MANUS_LLM_MODEL always overrides the configured model."""

    def test_overrides_default_model(self):
        with mock.patch.dict(os.environ, {"MANUS_LLM_MODEL": "claude-3-opus"}, clear=False):
            cfg = Config()
        assert cfg.llm.model == "claude-3-opus"

    def test_overrides_explicit_model(self):
        with mock.patch.dict(os.environ, {"MANUS_LLM_MODEL": "gpt-4-turbo"}, clear=False):
            cfg = Config(llm=LLMConfig(model="gpt-3.5-turbo"))
        assert cfg.llm.model == "gpt-4-turbo"

    def test_no_override_when_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "MANUS_LLM_MODEL"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Config(llm=LLMConfig(model="my-model"))
        assert cfg.llm.model == "my-model"


# ---------------------------------------------------------------------------
# MANUS_LLM_BASE_URL override
# ---------------------------------------------------------------------------


class TestLLMBaseURLOverride:
    """MANUS_LLM_BASE_URL always overrides the configured base_url."""

    def test_overrides_base_url(self):
        with mock.patch.dict(os.environ, {"MANUS_LLM_BASE_URL": "http://custom:8080"}, clear=False):
            cfg = Config()
        assert cfg.llm.base_url == "http://custom:8080"

    def test_overrides_explicit_base_url(self):
        with mock.patch.dict(os.environ, {"MANUS_LLM_BASE_URL": "http://new:9090"}, clear=False):
            cfg = Config(llm=LLMConfig(base_url="http://old:1234"))
        assert cfg.llm.base_url == "http://new:9090"

    def test_no_override_when_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "MANUS_LLM_BASE_URL"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Config(llm=LLMConfig(base_url="http://keep-me"))
        assert cfg.llm.base_url == "http://keep-me"


# ---------------------------------------------------------------------------
# MANUS_LLM_TEMPERATURE override
# ---------------------------------------------------------------------------


class TestLLMTemperatureOverride:
    """MANUS_LLM_TEMPERATURE overrides the configured temperature."""

    def test_overrides_temperature(self):
        with mock.patch.dict(os.environ, {"MANUS_LLM_TEMPERATURE": "0.7"}, clear=False):
            cfg = Config()
        assert cfg.llm.temperature == 0.7

    def test_float_precision(self):
        with mock.patch.dict(os.environ, {"MANUS_LLM_TEMPERATURE": "1.5"}, clear=False):
            cfg = Config(llm=LLMConfig(temperature=0.0))
        assert cfg.llm.temperature == 1.5

    def test_no_override_when_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "MANUS_LLM_TEMPERATURE"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Config(llm=LLMConfig(temperature=0.3))
        assert cfg.llm.temperature == 0.3


# ---------------------------------------------------------------------------
# MANUS_LLM_MAX_TOKENS override
# ---------------------------------------------------------------------------


class TestLLMMaxTokensOverride:
    """MANUS_LLM_MAX_TOKENS overrides the configured max_tokens."""

    def test_overrides_max_tokens(self):
        with mock.patch.dict(os.environ, {"MANUS_LLM_MAX_TOKENS": "8192"}, clear=False):
            cfg = Config()
        assert cfg.llm.max_tokens == 8192

    def test_no_override_when_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "MANUS_LLM_MAX_TOKENS"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = Config(llm=LLMConfig(max_tokens=2048))
        assert cfg.llm.max_tokens == 2048


# ---------------------------------------------------------------------------
# API key resolution — fill-when-None semantics
# ---------------------------------------------------------------------------


class TestAPIKeyResolution:
    """API keys are only filled from env when not already set in config."""

    def test_openai_key_from_env(self):
        env = {"OPENAI_API_KEY": "sk-test-123"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config(llm=LLMConfig(provider="openai", api_key=None))
        assert cfg.llm.api_key == "sk-test-123"

    def test_openai_key_not_overwritten_by_env(self):
        env = {"OPENAI_API_KEY": "sk-env-key"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config(llm=LLMConfig(provider="openai", api_key="sk-config-key"))
        assert cfg.llm.api_key == "sk-config-key"

    def test_anthropic_key_from_env(self):
        env = {"ANTHROPIC_API_KEY": "ant-key-456"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config(llm=LLMConfig(provider="anthropic", api_key=None))
        assert cfg.llm.api_key == "ant-key-456"

    def test_anthropic_key_not_overwritten_by_env(self):
        env = {"ANTHROPIC_API_KEY": "ant-env-key"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config(llm=LLMConfig(provider="anthropic", api_key="ant-config-key"))
        assert cfg.llm.api_key == "ant-config-key"

    def test_bedrock_no_api_key_lookup(self):
        """Bedrock doesn't use API keys (uses AWS credentials), so no env lookup."""
        env = {"OPENAI_API_KEY": "sk-wrong", "ANTHROPIC_API_KEY": "ant-wrong"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config(llm=LLMConfig(provider="bedrock", api_key=None))
        assert cfg.llm.api_key is None

    def test_ollama_no_api_key_lookup(self):
        """Ollama doesn't use API keys."""
        env = {"OPENAI_API_KEY": "sk-wrong", "ANTHROPIC_API_KEY": "ant-wrong"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config(llm=LLMConfig(provider="ollama", api_key=None))
        assert cfg.llm.api_key is None

    def test_unknown_provider_no_api_key_lookup(self):
        """Unknown providers don't get env-based API keys."""
        env = {"OPENAI_API_KEY": "sk-wrong"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config(llm=LLMConfig(provider="custom", api_key=None))
        assert cfg.llm.api_key is None


# ---------------------------------------------------------------------------
# AWS region resolution
# ---------------------------------------------------------------------------


class TestAWSRegionResolution:
    """AWS region resolved from multiple env var names with priority."""

    def test_manus_aws_region_takes_priority(self):
        env = {
            "MANUS_AWS_REGION": "us-west-2",
            "AWS_DEFAULT_REGION": "eu-west-1",
            "AWS_REGION": "ap-east-1",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config(llm=LLMConfig(aws_region=None))
        assert cfg.llm.aws_region == "us-west-2"

    def test_aws_default_region_fallback(self):
        env = {"AWS_DEFAULT_REGION": "eu-central-1"}
        clean = {k: v for k, v in os.environ.items() if k not in ("MANUS_AWS_REGION", "AWS_REGION")}
        clean.update(env)
        with mock.patch.dict(os.environ, clean, clear=True):
            cfg = Config(llm=LLMConfig(aws_region=None))
        assert cfg.llm.aws_region == "eu-central-1"

    def test_aws_region_fallback(self):
        env = {"AWS_REGION": "ap-southeast-1"}
        clean = {k: v for k, v in os.environ.items() if k not in ("MANUS_AWS_REGION", "AWS_DEFAULT_REGION")}
        clean.update(env)
        with mock.patch.dict(os.environ, clean, clear=True):
            cfg = Config(llm=LLMConfig(aws_region=None))
        assert cfg.llm.aws_region == "ap-southeast-1"

    def test_no_region_env_vars_leaves_none(self):
        clean = {k: v for k, v in os.environ.items() if k not in ("MANUS_AWS_REGION", "AWS_DEFAULT_REGION", "AWS_REGION")}
        with mock.patch.dict(os.environ, clean, clear=True):
            cfg = Config(llm=LLMConfig(aws_region=None))
        assert cfg.llm.aws_region is None

    def test_config_region_not_overwritten(self):
        """When aws_region is already set in config, env vars don't overwrite."""
        env = {"MANUS_AWS_REGION": "us-west-1"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config(llm=LLMConfig(aws_region="eu-north-1"))
        assert cfg.llm.aws_region == "eu-north-1"


# ---------------------------------------------------------------------------
# Integration configs — OTX
# ---------------------------------------------------------------------------


class TestOTXConfig:
    """MANUS_OTX_API_KEY env var fills OTX config when None."""

    def test_otx_key_from_env(self):
        env = {"MANUS_OTX_API_KEY": "otx-key-abc"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config()
        assert cfg.otx.api_key == "otx-key-abc"

    def test_otx_key_not_overwritten(self):
        env = {"MANUS_OTX_API_KEY": "otx-env"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config(otx={"api_key": "otx-config"})
        assert cfg.otx.api_key == "otx-config"

    def test_otx_key_stays_none_when_no_env(self):
        clean = {k: v for k, v in os.environ.items() if k != "MANUS_OTX_API_KEY"}
        with mock.patch.dict(os.environ, clean, clear=True):
            cfg = Config()
        assert cfg.otx.api_key is None


# ---------------------------------------------------------------------------
# Integration configs — GitHub
# ---------------------------------------------------------------------------


class TestGitHubConfig:
    """MANUS_GITHUB_TOKEN / GITHUB_TOKEN env vars fill GitHub config."""

    def test_manus_github_token_takes_priority(self):
        env = {"MANUS_GITHUB_TOKEN": "ghp-manus", "GITHUB_TOKEN": "ghp-generic"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config()
        assert cfg.github.api_token == "ghp-manus"

    def test_github_token_fallback(self):
        env = {"GITHUB_TOKEN": "ghp-fallback"}
        clean = {k: v for k, v in os.environ.items() if k != "MANUS_GITHUB_TOKEN"}
        clean.update(env)
        with mock.patch.dict(os.environ, clean, clear=True):
            cfg = Config()
        assert cfg.github.api_token == "ghp-fallback"

    def test_not_overwritten_when_set(self):
        env = {"MANUS_GITHUB_TOKEN": "ghp-env"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config(github={"api_token": "ghp-config"})
        assert cfg.github.api_token == "ghp-config"

    def test_stays_none_when_no_env(self):
        clean = {k: v for k, v in os.environ.items() if k not in ("MANUS_GITHUB_TOKEN", "GITHUB_TOKEN")}
        with mock.patch.dict(os.environ, clean, clear=True):
            cfg = Config()
        assert cfg.github.api_token is None


# ---------------------------------------------------------------------------
# Integration configs — Lark
# ---------------------------------------------------------------------------


class TestLarkConfig:
    """MANUS_LARK_API_TOKEN and MANUS_LARK_DOCUMENT_URL env vars."""

    def test_lark_token_from_manus_env(self):
        env = {"MANUS_LARK_API_TOKEN": "lark-manus", "LARK_API_TOKEN": "lark-generic"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config()
        assert cfg.lark.api_token == "lark-manus"

    def test_lark_token_from_generic_env(self):
        env = {"LARK_API_TOKEN": "lark-generic"}
        clean = {k: v for k, v in os.environ.items() if k != "MANUS_LARK_API_TOKEN"}
        clean.update(env)
        with mock.patch.dict(os.environ, clean, clear=True):
            cfg = Config()
        assert cfg.lark.api_token == "lark-generic"

    def test_lark_doc_url_from_manus_env(self):
        env = {"MANUS_LARK_DOCUMENT_URL": "https://lark.example.com/doc/1"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config()
        assert cfg.lark.document_url == "https://lark.example.com/doc/1"

    def test_lark_doc_url_from_generic_env(self):
        env = {"LARK_DOCUMENT_URL": "https://generic.example.com/doc/2"}
        clean = {k: v for k, v in os.environ.items() if k != "MANUS_LARK_DOCUMENT_URL"}
        clean.update(env)
        with mock.patch.dict(os.environ, clean, clear=True):
            cfg = Config()
        assert cfg.lark.document_url == "https://generic.example.com/doc/2"

    def test_lark_not_overwritten(self):
        env = {"MANUS_LARK_API_TOKEN": "lark-env", "MANUS_LARK_DOCUMENT_URL": "https://env.url"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config(lark={"api_token": "lark-config", "document_url": "https://config.url"})
        assert cfg.lark.api_token == "lark-config"
        assert cfg.lark.document_url == "https://config.url"


# ---------------------------------------------------------------------------
# Integration configs — Webhooks
# ---------------------------------------------------------------------------


class TestWebhooksConfig:
    """MANUS_WEBHOOK_CVE_SUBMIT_URL env var."""

    def test_webhook_url_from_env(self):
        env = {"MANUS_WEBHOOK_CVE_SUBMIT_URL": "https://hooks.example.com/cve"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config()
        assert cfg.webhooks.cve_submit_url == "https://hooks.example.com/cve"

    def test_webhook_not_overwritten(self):
        env = {"MANUS_WEBHOOK_CVE_SUBMIT_URL": "https://env.url"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config(webhooks={"cve_submit_url": "https://config.url"})
        assert cfg.webhooks.cve_submit_url == "https://config.url"

    def test_webhook_stays_none(self):
        clean = {k: v for k, v in os.environ.items() if k != "MANUS_WEBHOOK_CVE_SUBMIT_URL"}
        with mock.patch.dict(os.environ, clean, clear=True):
            cfg = Config()
        assert cfg.webhooks.cve_submit_url is None


# ---------------------------------------------------------------------------
# Integration configs — MCP
# ---------------------------------------------------------------------------


class TestMCPConfig:
    """MANUS_MCP_SERVER_URL env var."""

    def test_mcp_url_from_env(self):
        env = {"MANUS_MCP_SERVER_URL": "http://mcp:3000"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config()
        assert cfg.mcp.server_url == "http://mcp:3000"

    def test_mcp_not_overwritten(self):
        env = {"MANUS_MCP_SERVER_URL": "http://env-mcp"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config(mcp={"server_url": "http://config-mcp"})
        assert cfg.mcp.server_url == "http://config-mcp"

    def test_mcp_stays_none(self):
        clean = {k: v for k, v in os.environ.items() if k != "MANUS_MCP_SERVER_URL"}
        with mock.patch.dict(os.environ, clean, clear=True):
            cfg = Config()
        assert cfg.mcp.server_url is None


# ---------------------------------------------------------------------------
# Config.from_file() — search path logic
# ---------------------------------------------------------------------------


class TestFromFileSearchPaths:
    """Config.from_file() discovers config.toml from standard locations."""

    def test_explicit_path(self, tmp_path):
        cfg_file = tmp_path / "custom.toml"
        cfg_file.write_text('[llm]\nprovider = "anthropic"\nmodel = "claude-3"\n')
        with mock.patch.dict(os.environ, {}, clear=False):
            cfg = Config.from_file(cfg_file)
        assert cfg.llm.provider == "anthropic"
        assert cfg.llm.model == "claude-3"

    def test_cwd_config_toml(self, tmp_path, monkeypatch):
        """config.toml in CWD is discovered first."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.toml").write_text('[llm]\nprovider = "ollama"\n')
        with mock.patch.dict(os.environ, {}, clear=False):
            cfg = Config.from_file()
        assert cfg.llm.provider == "ollama"

    def test_config_dir_config_toml(self, tmp_path, monkeypatch):
        """config/config.toml is discovered if CWD config.toml doesn't exist."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "config.toml").write_text('[llm]\nprovider = "bedrock"\n')
        with mock.patch.dict(os.environ, {}, clear=False):
            cfg = Config.from_file()
        assert cfg.llm.provider == "bedrock"

    def test_home_dir_config(self, tmp_path, monkeypatch):
        """~/.manus-agent/config.toml is discovered as last resort."""
        monkeypatch.chdir(tmp_path)
        home_cfg_dir = tmp_path / ".manus-agent"
        home_cfg_dir.mkdir()
        (home_cfg_dir / "config.toml").write_text('[llm]\nmodel = "home-model"\n')
        with mock.patch.object(Path, "home", return_value=tmp_path):
            with mock.patch.dict(os.environ, {}, clear=False):
                cfg = Config.from_file()
        assert cfg.llm.model == "home-model"

    def test_no_config_file_returns_defaults(self, tmp_path, monkeypatch):
        """When no config.toml is found, defaults are returned."""
        monkeypatch.chdir(tmp_path)
        with mock.patch.object(Path, "home", return_value=tmp_path):
            with mock.patch.dict(os.environ, {}, clear=False):
                cfg = Config.from_file()
        assert cfg.llm.provider == "openai"
        assert cfg.llm.model == "gpt-4o"

    def test_nonexistent_explicit_path_returns_defaults(self, tmp_path):
        """Explicit path that doesn't exist returns defaults."""
        with mock.patch.dict(os.environ, {}, clear=False):
            cfg = Config.from_file(tmp_path / "does_not_exist.toml")
        assert cfg.llm.provider == "openai"

    def test_from_file_loads_all_sections(self, tmp_path):
        """All config sections are loaded from TOML."""
        cfg_file = tmp_path / "full.toml"
        cfg_file.write_text(
            "[llm]\n"
            'provider = "anthropic"\n'
            'model = "claude-3-haiku"\n'
            "temperature = 0.5\n"
            "max_tokens = 2048\n\n"
            "[sandbox]\n"
            "enabled = false\n\n"
            "[github]\n"
            'api_token = "ghp-from-file"\n\n'
            "[otx]\n"
            'api_key = "otx-from-file"\n'
        )
        with mock.patch.dict(os.environ, {}, clear=False):
            cfg = Config.from_file(cfg_file)
        assert cfg.llm.provider == "anthropic"
        assert cfg.llm.model == "claude-3-haiku"
        assert cfg.llm.temperature == 0.5
        assert cfg.llm.max_tokens == 2048
        assert cfg.sandbox.enabled is False
        assert cfg.github.api_token == "ghp-from-file"
        assert cfg.otx.api_key == "otx-from-file"


# ---------------------------------------------------------------------------
# Config.from_file() — env overrides applied after TOML load
# ---------------------------------------------------------------------------


class TestFromFileEnvOverridePriority:
    """Env vars override config.toml values for MANUS_LLM_* (always-override semantics)."""

    def test_env_overrides_toml_provider(self, tmp_path):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('[llm]\nprovider = "openai"\nmodel = "gpt-4o"\n')
        env = {"MANUS_LLM_PROVIDER": "anthropic"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config.from_file(cfg_file)
        assert cfg.llm.provider == "anthropic"

    def test_env_overrides_toml_model(self, tmp_path):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('[llm]\nprovider = "openai"\nmodel = "gpt-4o"\n')
        env = {"MANUS_LLM_MODEL": "gpt-4-turbo"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config.from_file(cfg_file)
        assert cfg.llm.model == "gpt-4-turbo"

    def test_toml_api_key_not_overwritten_by_env(self, tmp_path):
        """API key set in TOML is preserved (fill-when-None semantics)."""
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('[llm]\nprovider = "openai"\napi_key = "sk-from-toml"\n')
        env = {"OPENAI_API_KEY": "sk-from-env"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config.from_file(cfg_file)
        assert cfg.llm.api_key == "sk-from-toml"

    def test_toml_no_api_key_filled_from_env(self, tmp_path):
        """API key absent in TOML is filled from env."""
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('[llm]\nprovider = "openai"\nmodel = "gpt-4o"\n')
        env = {"OPENAI_API_KEY": "sk-from-env"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config.from_file(cfg_file)
        assert cfg.llm.api_key == "sk-from-env"


# ---------------------------------------------------------------------------
# _load_dotenv — .env file search and loading
# ---------------------------------------------------------------------------


class TestLoadDotenv:
    """_load_dotenv() searches CWD, config/, and ~/.manus-agent/ for .env files."""

    def test_loads_cwd_dotenv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("TEST_DOTENV_VAR=cwd_value\n")
        # Remove the var if it happens to be set
        env = {k: v for k, v in os.environ.items() if k != "TEST_DOTENV_VAR"}
        with mock.patch.dict(os.environ, env, clear=True):
            _load_dotenv()
            assert os.environ.get("TEST_DOTENV_VAR") == "cwd_value"

    def test_loads_config_dir_dotenv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / ".env").write_text("TEST_DOTENV_VAR2=config_value\n")
        env = {k: v for k, v in os.environ.items() if k != "TEST_DOTENV_VAR2"}
        with mock.patch.dict(os.environ, env, clear=True):
            _load_dotenv()
            assert os.environ.get("TEST_DOTENV_VAR2") == "config_value"

    def test_loads_home_dir_dotenv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        home_dir = tmp_path / "fakehome"
        home_dir.mkdir()
        (home_dir / ".manus-agent").mkdir()
        (home_dir / ".manus-agent" / ".env").write_text("TEST_DOTENV_VAR3=home_value\n")
        env = {k: v for k, v in os.environ.items() if k != "TEST_DOTENV_VAR3"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(Path, "home", return_value=home_dir):
                _load_dotenv()
            assert os.environ.get("TEST_DOTENV_VAR3") == "home_value"

    def test_cwd_dotenv_takes_priority(self, tmp_path, monkeypatch):
        """CWD .env is loaded first — stops looking after first found."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("TEST_DOTENV_PRIO=cwd\n")
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / ".env").write_text("TEST_DOTENV_PRIO=config\n")
        env = {k: v for k, v in os.environ.items() if k != "TEST_DOTENV_PRIO"}
        with mock.patch.dict(os.environ, env, clear=True):
            _load_dotenv()
            assert os.environ.get("TEST_DOTENV_PRIO") == "cwd"

    def test_no_dotenv_file_no_error(self, tmp_path, monkeypatch):
        """No .env file anywhere doesn't raise."""
        monkeypatch.chdir(tmp_path)
        with mock.patch.object(Path, "home", return_value=tmp_path):
            _load_dotenv()  # Should not raise

    def test_dotenv_import_failure_graceful(self, monkeypatch):
        """If python-dotenv is not installed, _load_dotenv is a no-op."""
        import importlib
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "dotenv":
                raise ImportError("No module named 'dotenv'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        _load_dotenv()  # Should not raise

    def test_existing_env_vars_not_overwritten_by_dotenv(self, tmp_path, monkeypatch):
        """python-dotenv with override=False preserves existing env vars."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("TEST_ALREADY_SET=dotenv_value\n")
        env = dict(os.environ)
        env["TEST_ALREADY_SET"] = "shell_value"
        with mock.patch.dict(os.environ, env, clear=True):
            _load_dotenv()
            assert os.environ["TEST_ALREADY_SET"] == "shell_value"


# ---------------------------------------------------------------------------
# Combined env override scenarios
# ---------------------------------------------------------------------------


class TestCombinedOverrides:
    """Multiple env overrides applied together."""

    def test_full_env_only_config(self):
        """All LLM settings from env vars — no config file needed."""
        env = {
            "MANUS_LLM_PROVIDER": "anthropic",
            "MANUS_LLM_MODEL": "claude-3-5-sonnet",
            "MANUS_LLM_TEMPERATURE": "0.3",
            "MANUS_LLM_MAX_TOKENS": "16384",
            "ANTHROPIC_API_KEY": "ant-full-env-key",
            "MANUS_GITHUB_TOKEN": "ghp-full-env",
            "MANUS_OTX_API_KEY": "otx-full-env",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config()
        assert cfg.llm.provider == "anthropic"
        assert cfg.llm.model == "claude-3-5-sonnet"
        assert cfg.llm.temperature == 0.3
        assert cfg.llm.max_tokens == 16384
        assert cfg.llm.api_key == "ant-full-env-key"
        assert cfg.github.api_token == "ghp-full-env"
        assert cfg.otx.api_key == "otx-full-env"

    def test_partial_override_preserves_file_values(self, tmp_path):
        """Only override what env provides; keep the rest from file."""
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            "[llm]\n"
            'provider = "openai"\n'
            'model = "gpt-4o"\n'
            "temperature = 0.2\n"
            "max_tokens = 4096\n\n"
            "[github]\n"
            'api_token = "ghp-file-token"\n'
        )
        env = {"MANUS_LLM_MODEL": "gpt-4-turbo"}  # Only override model
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config.from_file(cfg_file)
        assert cfg.llm.provider == "openai"  # Unchanged
        assert cfg.llm.model == "gpt-4-turbo"  # Overridden
        assert cfg.llm.temperature == 0.2  # Unchanged
        assert cfg.github.api_token == "ghp-file-token"  # Unchanged

    def test_empty_env_values_treated_as_unset(self):
        """Empty strings in env vars are treated as not set (falsy check)."""
        env = {
            "MANUS_LLM_PROVIDER": "",
            "MANUS_LLM_MODEL": "",
            "MANUS_LLM_BASE_URL": "",
            "MANUS_OTX_API_KEY": "",
            "MANUS_GITHUB_TOKEN": "",
            "GITHUB_TOKEN": "",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = Config(llm=LLMConfig(provider="bedrock", model="my-model"))
        assert cfg.llm.provider == "bedrock"
        assert cfg.llm.model == "my-model"
        assert cfg.otx.api_key is None
        assert cfg.github.api_token is None
