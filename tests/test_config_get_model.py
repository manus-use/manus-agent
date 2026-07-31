"""Comprehensive test suite for Config.get_model() happy paths.

The existing test_config.py covers error cases (missing imports, unknown provider).
This file tests the SUCCESS paths: verifying that get_model() passes the correct
kwargs to each provider's model constructor (BedrockModel, OpenAIModel,
AnthropicModel, OllamaModel) under various configuration combinations.

All model constructors are mocked — no real API calls are made.
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest

from manus_agent.config import Config, LLMConfig

# ---------------------------------------------------------------------------
# Bedrock provider — happy paths
# ---------------------------------------------------------------------------


class TestGetModelBedrock:
    """get_model() with provider='bedrock' constructs BedrockModel correctly."""

    def test_bedrock_basic(self, monkeypatch):
        """BedrockModel is called with model_id, region, temperature, max_tokens."""
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        config = Config(llm=LLMConfig(provider="bedrock", model="us.anthropic.claude-3-5-sonnet-20241022-v2:0"))

        with mock.patch("strands.models.BedrockModel") as mock_bedrock:
            mock_bedrock.return_value = mock.sentinel.bedrock_model
            result = config.get_model()

        assert result is mock.sentinel.bedrock_model
        mock_bedrock.assert_called_once()
        call_kwargs = mock_bedrock.call_args[1]
        assert call_kwargs["model_id"] == "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["region"] == "us-west-2"
        assert call_kwargs["region_name"] == "us-west-2"

    def test_bedrock_custom_region_from_config(self, monkeypatch):
        """aws_region from LLMConfig is used when env var is absent."""
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        monkeypatch.delenv("MANUS_AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        config = Config(llm=LLMConfig(provider="bedrock", model="claude-3", aws_region="eu-west-1"))

        with mock.patch("strands.models.BedrockModel") as mock_bedrock:
            mock_bedrock.return_value = mock.sentinel.bedrock_model
            config.get_model()

        call_kwargs = mock_bedrock.call_args[1]
        assert call_kwargs["region"] == "eu-west-1"
        assert call_kwargs["region_name"] == "eu-west-1"

    def test_bedrock_defaults_to_us_west_2_when_no_region(self, monkeypatch):
        """Falls back to 'us-west-2' when no region is configured anywhere."""
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        monkeypatch.delenv("MANUS_AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        config = Config(llm=LLMConfig(provider="bedrock", model="claude-3"))

        with mock.patch("strands.models.BedrockModel") as mock_bedrock:
            mock_bedrock.return_value = mock.sentinel.bedrock_model
            config.get_model()

        call_kwargs = mock_bedrock.call_args[1]
        assert call_kwargs["region"] == "us-west-2"

    def test_bedrock_env_region_overrides_config(self, monkeypatch):
        """AWS_DEFAULT_REGION env var takes priority over config aws_region."""
        monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-1")
        config = Config(llm=LLMConfig(provider="bedrock", model="claude-3", aws_region="eu-west-1"))

        with mock.patch("strands.models.BedrockModel") as mock_bedrock:
            mock_bedrock.return_value = mock.sentinel.bedrock_model
            config.get_model()

        call_kwargs = mock_bedrock.call_args[1]
        assert call_kwargs["region"] == "ap-southeast-1"

    def test_bedrock_custom_temperature_and_max_tokens(self, monkeypatch):
        """Custom temperature and max_tokens are forwarded."""
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        config = Config(llm=LLMConfig(provider="bedrock", model="claude-3", temperature=0.7, max_tokens=8192))

        with mock.patch("strands.models.BedrockModel") as mock_bedrock:
            mock_bedrock.return_value = mock.sentinel.bedrock_model
            config.get_model()

        call_kwargs = mock_bedrock.call_args[1]
        assert call_kwargs["temperature"] == pytest.approx(0.7)
        assert call_kwargs["max_tokens"] == 8192

    def test_bedrock_provider_case_insensitive(self, monkeypatch):
        """Provider matching is case-insensitive."""
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        config = Config(llm=LLMConfig(provider="Bedrock", model="claude-3"))

        with mock.patch("strands.models.BedrockModel") as mock_bedrock:
            mock_bedrock.return_value = mock.sentinel.bedrock_model
            config.get_model()

        mock_bedrock.assert_called_once()


# ---------------------------------------------------------------------------
# OpenAI provider — happy paths
# ---------------------------------------------------------------------------


class TestGetModelOpenAI:
    """get_model() with provider='openai' constructs OpenAIModel correctly."""

    def test_openai_basic_no_key(self):
        """OpenAIModel is called without client_args when no api_key or base_url."""
        config = Config(llm=LLMConfig(provider="openai", model="gpt-4o"))

        with mock.patch("strands.models.openai.OpenAIModel") as mock_openai:
            mock_openai.return_value = mock.sentinel.openai_model
            result = config.get_model()

        assert result is mock.sentinel.openai_model
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["model_id"] == "gpt-4o"
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["client_args"] == {}

    def test_openai_with_api_key(self):
        """api_key is passed in client_args."""
        config = Config(llm=LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test-key-123"))

        with mock.patch("strands.models.openai.OpenAIModel") as mock_openai:
            mock_openai.return_value = mock.sentinel.openai_model
            config.get_model()

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["client_args"]["api_key"] == "sk-test-key-123"

    def test_openai_with_base_url(self):
        """base_url is passed in client_args."""
        config = Config(llm=LLMConfig(provider="openai", model="gpt-4", base_url="http://localhost:8000/v1"))

        with mock.patch("strands.models.openai.OpenAIModel") as mock_openai:
            mock_openai.return_value = mock.sentinel.openai_model
            config.get_model()

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["client_args"]["base_url"] == "http://localhost:8000/v1"

    def test_openai_with_key_and_base_url(self):
        """Both api_key and base_url are in client_args."""
        config = Config(
            llm=LLMConfig(
                provider="openai",
                model="gpt-4",
                api_key="sk-custom",
                base_url="https://proxy.example.com/v1",
            )
        )

        with mock.patch("strands.models.openai.OpenAIModel") as mock_openai:
            mock_openai.return_value = mock.sentinel.openai_model
            config.get_model()

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["client_args"] == {
            "api_key": "sk-custom",
            "base_url": "https://proxy.example.com/v1",
        }
        assert call_kwargs["model_id"] == "gpt-4"
        assert call_kwargs["max_tokens"] == 4096

    def test_openai_custom_max_tokens(self):
        """Custom max_tokens is forwarded to OpenAIModel."""
        config = Config(llm=LLMConfig(provider="openai", model="gpt-4o", max_tokens=16384))

        with mock.patch("strands.models.openai.OpenAIModel") as mock_openai:
            mock_openai.return_value = mock.sentinel.openai_model
            config.get_model()

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["max_tokens"] == 16384

    def test_openai_provider_case_insensitive(self):
        """Provider matching is case-insensitive for OpenAI."""
        config = Config(llm=LLMConfig(provider="OpenAI", model="gpt-4"))

        with mock.patch("strands.models.openai.OpenAIModel") as mock_openai:
            mock_openai.return_value = mock.sentinel.openai_model
            config.get_model()

        mock_openai.assert_called_once()

    def test_openai_returns_model_instance(self):
        """get_model() returns the constructed OpenAIModel instance."""
        config = Config(llm=LLMConfig(provider="openai", model="gpt-4o"))

        mock_instance = mock.MagicMock(name="OpenAIModelInstance")
        with mock.patch("strands.models.openai.OpenAIModel", return_value=mock_instance):
            result = config.get_model()

        assert result is mock_instance


# ---------------------------------------------------------------------------
# Anthropic provider — happy paths
# ---------------------------------------------------------------------------


class TestGetModelAnthropic:
    """get_model() with provider='anthropic' constructs AnthropicModel correctly."""

    def test_anthropic_basic_no_key(self):
        """AnthropicModel is called with model_id and max_tokens (no api_key)."""
        config = Config(llm=LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022"))

        with mock.patch("strands.models.anthropic.AnthropicModel") as mock_anthropic:
            mock_anthropic.return_value = mock.sentinel.anthropic_model
            result = config.get_model()

        assert result is mock.sentinel.anthropic_model
        mock_anthropic.assert_called_once()
        call_kwargs = mock_anthropic.call_args[1]
        assert call_kwargs["model_id"] == "claude-3-5-sonnet-20241022"
        assert call_kwargs["max_tokens"] == 4096
        assert "api_key" not in call_kwargs

    def test_anthropic_with_api_key(self):
        """api_key is passed when configured."""
        config = Config(
            llm=LLMConfig(
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                api_key="sk-ant-test-key",
            )
        )

        with mock.patch("strands.models.anthropic.AnthropicModel") as mock_anthropic:
            mock_anthropic.return_value = mock.sentinel.anthropic_model
            config.get_model()

        call_kwargs = mock_anthropic.call_args[1]
        assert call_kwargs["api_key"] == "sk-ant-test-key"
        assert call_kwargs["model_id"] == "claude-3-5-sonnet-20241022"
        assert call_kwargs["max_tokens"] == 4096

    def test_anthropic_custom_max_tokens(self):
        """Custom max_tokens is forwarded."""
        config = Config(llm=LLMConfig(provider="anthropic", model="claude-3-opus-20240229", max_tokens=2048))

        with mock.patch("strands.models.anthropic.AnthropicModel") as mock_anthropic:
            mock_anthropic.return_value = mock.sentinel.anthropic_model
            config.get_model()

        call_kwargs = mock_anthropic.call_args[1]
        assert call_kwargs["max_tokens"] == 2048

    def test_anthropic_no_base_url_forwarded(self):
        """base_url is NOT forwarded to AnthropicModel (not in its API)."""
        config = Config(
            llm=LLMConfig(
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                base_url="http://some-proxy.example.com",
            )
        )

        with mock.patch("strands.models.anthropic.AnthropicModel") as mock_anthropic:
            mock_anthropic.return_value = mock.sentinel.anthropic_model
            config.get_model()

        call_kwargs = mock_anthropic.call_args[1]
        assert "base_url" not in call_kwargs

    def test_anthropic_provider_case_insensitive(self):
        """Provider matching is case-insensitive for Anthropic."""
        config = Config(llm=LLMConfig(provider="ANTHROPIC", model="claude-3-haiku-20240307"))

        with mock.patch("strands.models.anthropic.AnthropicModel") as mock_anthropic:
            mock_anthropic.return_value = mock.sentinel.anthropic_model
            config.get_model()

        mock_anthropic.assert_called_once()

    def test_anthropic_returns_model_instance(self):
        """get_model() returns the constructed AnthropicModel instance."""
        config = Config(llm=LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022"))

        mock_instance = mock.MagicMock(name="AnthropicModelInstance")
        with mock.patch("strands.models.anthropic.AnthropicModel", return_value=mock_instance):
            result = config.get_model()

        assert result is mock_instance


# ---------------------------------------------------------------------------
# Ollama provider — happy paths
# ---------------------------------------------------------------------------


class TestGetModelOllama:
    """get_model() with provider='ollama' constructs OllamaModel correctly."""

    @staticmethod
    def _make_ollama_patcher():
        """Create a mock OllamaModel and patch it into strands.models.ollama."""
        mock_ollama_module = mock.MagicMock()
        mock_cls = mock.MagicMock(name="OllamaModel")
        mock_ollama_module.OllamaModel = mock_cls
        patcher = mock.patch.dict(sys.modules, {"strands.models.ollama": mock_ollama_module})
        return patcher, mock_cls

    def test_ollama_basic_default_host(self):
        """OllamaModel is called with model_id and default localhost host."""
        config = Config(llm=LLMConfig(provider="ollama", model="llama3"))

        patcher, mock_cls = self._make_ollama_patcher()
        mock_cls.return_value = mock.sentinel.ollama_model
        with patcher:
            result = config.get_model()

        assert result is mock.sentinel.ollama_model
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model_id"] == "llama3"
        assert call_kwargs["host"] == "http://localhost:11434"

    def test_ollama_custom_host(self):
        """Custom base_url is forwarded as host."""
        config = Config(llm=LLMConfig(provider="ollama", model="mistral", base_url="http://gpu-server:11434"))

        patcher, mock_cls = self._make_ollama_patcher()
        mock_cls.return_value = mock.sentinel.ollama_model
        with patcher:
            config.get_model()

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["host"] == "http://gpu-server:11434"
        assert call_kwargs["model_id"] == "mistral"

    def test_ollama_provider_case_insensitive(self):
        """Provider matching is case-insensitive for Ollama."""
        config = Config(llm=LLMConfig(provider="Ollama", model="codellama"))

        patcher, mock_cls = self._make_ollama_patcher()
        mock_cls.return_value = mock.sentinel.ollama_model
        with patcher:
            config.get_model()

        mock_cls.assert_called_once()

    def test_ollama_returns_model_instance(self):
        """get_model() returns the constructed OllamaModel instance."""
        config = Config(llm=LLMConfig(provider="ollama", model="phi3"))

        patcher, mock_cls = self._make_ollama_patcher()
        mock_instance = mock.MagicMock(name="OllamaModelInstance")
        mock_cls.return_value = mock_instance
        with patcher:
            result = config.get_model()

        assert result is mock_instance

    def test_ollama_constructor_receives_only_expected_keys(self):
        """OllamaModel gets exactly model_id and host."""
        config = Config(llm=LLMConfig(provider="ollama", model="phi3"))

        patcher, mock_cls = self._make_ollama_patcher()
        mock_cls.return_value = mock.sentinel.ollama_model
        with patcher:
            config.get_model()

        call_kwargs = mock_cls.call_args[1]
        assert set(call_kwargs.keys()) == {"model_id", "host"}


# ---------------------------------------------------------------------------
# Edge cases and integration scenarios
# ---------------------------------------------------------------------------


class TestGetModelEdgeCases:
    """Edge cases, None handling, and env-based configuration for get_model()."""

    def test_provider_none_raises_value_error(self):
        """provider=None (after lower()) raises ValueError for empty string."""
        config = Config(llm=LLMConfig(provider="", model="test"))
        with pytest.raises(ValueError, match="Unknown provider"):
            config.get_model()

    def test_provider_whitespace_raises_value_error(self):
        """Whitespace-only provider is not a valid provider."""
        config = Config(llm=LLMConfig(provider="  ", model="test"))
        with pytest.raises(ValueError, match="Unknown provider"):
            config.get_model()

    def test_bedrock_model_kwargs_property(self, monkeypatch):
        """model_kwargs property builds correct dict for bedrock."""
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        llm = LLMConfig(provider="bedrock", model="claude-3", temperature=0.5, max_tokens=2048)
        kwargs = llm.model_kwargs
        assert kwargs["model_id"] == "claude-3"
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 2048
        assert kwargs["region"] == "us-east-1"
        assert kwargs["region_name"] == "us-east-1"

    def test_openai_model_kwargs_property(self):
        """model_kwargs property builds correct dict for openai."""
        llm = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key="sk-k",
            base_url="http://x",
            temperature=0.3,
            max_tokens=1024,
        )
        kwargs = llm.model_kwargs
        assert kwargs["model_id"] == "gpt-4o"
        assert kwargs["api_key"] == "sk-k"
        assert kwargs["base_url"] == "http://x"
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_tokens"] == 1024

    def test_anthropic_model_kwargs_property(self):
        """model_kwargs property builds correct dict for anthropic."""
        llm = LLMConfig(provider="anthropic", model="claude-3-haiku", api_key="sk-ant")
        kwargs = llm.model_kwargs
        assert kwargs["model_id"] == "claude-3-haiku"
        assert kwargs["api_key"] == "sk-ant"
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_tokens"] == 4096

    def test_ollama_model_kwargs_property(self):
        """model_kwargs property builds correct dict for ollama."""
        llm = LLMConfig(provider="ollama", model="llama3", base_url="http://remote:11434")
        kwargs = llm.model_kwargs
        assert kwargs["model_id"] == "llama3"
        assert kwargs["host"] == "http://remote:11434"

    def test_ollama_model_kwargs_default_host(self):
        """model_kwargs uses default localhost when no base_url set."""
        llm = LLMConfig(provider="ollama", model="llama3")
        kwargs = llm.model_kwargs
        assert kwargs["host"] == "http://localhost:11434"

    def test_openai_model_kwargs_no_key_no_base_url(self):
        """model_kwargs does not include api_key or base_url when None."""
        llm = LLMConfig(provider="openai", model="gpt-4o")
        kwargs = llm.model_kwargs
        assert "api_key" not in kwargs
        assert "base_url" not in kwargs

    def test_anthropic_model_kwargs_no_key(self):
        """model_kwargs does not include api_key when None."""
        llm = LLMConfig(provider="anthropic", model="claude-3")
        kwargs = llm.model_kwargs
        assert "api_key" not in kwargs

    def test_get_model_from_env_configured_config(self, monkeypatch):
        """get_model() works end-to-end with env-only configuration."""
        monkeypatch.setenv("MANUS_LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("MANUS_LLM_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        config = Config()

        with mock.patch("strands.models.BedrockModel") as mock_bedrock:
            mock_bedrock.return_value = mock.sentinel.bedrock_model
            result = config.get_model()

        assert result is mock.sentinel.bedrock_model
        call_kwargs = mock_bedrock.call_args[1]
        assert call_kwargs["model_id"] == "us.anthropic.claude-sonnet-4-20250514-v1:0"
        assert call_kwargs["region"] == "us-east-1"

    def test_get_model_from_toml_file(self, tmp_path, monkeypatch):
        """get_model() works with config loaded from a TOML file."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("MANUS_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("MANUS_LLM_MODEL", raising=False)

        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            "[llm]\n"
            'provider = "anthropic"\n'
            'model = "claude-3-opus-20240229"\n'
            'api_key = "sk-ant-toml-key"\n'
            "max_tokens = 8192\n"
        )

        config = Config.from_file(cfg_file)

        with mock.patch("strands.models.anthropic.AnthropicModel") as mock_anthropic:
            mock_anthropic.return_value = mock.sentinel.anthropic_model
            result = config.get_model()

        assert result is mock.sentinel.anthropic_model
        call_kwargs = mock_anthropic.call_args[1]
        assert call_kwargs["model_id"] == "claude-3-opus-20240229"
        assert call_kwargs["api_key"] == "sk-ant-toml-key"
        assert call_kwargs["max_tokens"] == 8192

    def test_get_model_openai_env_key_backfill(self, monkeypatch):
        """OPENAI_API_KEY env var is backfilled and passed to constructor."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-backfill")
        monkeypatch.delenv("MANUS_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("MANUS_LLM_MODEL", raising=False)

        # Default provider is openai
        config = Config()

        with mock.patch("strands.models.openai.OpenAIModel") as mock_openai:
            mock_openai.return_value = mock.sentinel.openai_model
            config.get_model()

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["client_args"]["api_key"] == "sk-env-backfill"

    def test_get_model_bedrock_region_name_equals_region(self, monkeypatch):
        """Both 'region' and 'region_name' are set identically for compat."""
        monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-northeast-1")
        config = Config(llm=LLMConfig(provider="bedrock", model="claude-3"))

        with mock.patch("strands.models.BedrockModel") as mock_bedrock:
            mock_bedrock.return_value = mock.sentinel.bedrock_model
            config.get_model()

        call_kwargs = mock_bedrock.call_args[1]
        assert call_kwargs["region"] == call_kwargs["region_name"]
        assert call_kwargs["region"] == "ap-northeast-1"

    def test_bedrock_always_passes_temperature_and_max_tokens(self, monkeypatch):
        """temperature and max_tokens are always passed (even at defaults)."""
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        config = Config(llm=LLMConfig(provider="bedrock", model="claude-3"))

        with mock.patch("strands.models.BedrockModel") as mock_bedrock:
            mock_bedrock.return_value = mock.sentinel.bedrock_model
            config.get_model()

        call_kwargs = mock_bedrock.call_args[1]
        assert "temperature" in call_kwargs
        assert "max_tokens" in call_kwargs

    def test_openai_constructor_receives_only_expected_keys(self):
        """OpenAIModel gets exactly client_args, model_id, max_tokens."""
        config = Config(llm=LLMConfig(provider="openai", model="gpt-4o", api_key="sk-x", temperature=0.9))

        with mock.patch("strands.models.openai.OpenAIModel") as mock_openai:
            mock_openai.return_value = mock.sentinel.openai_model
            config.get_model()

        call_kwargs = mock_openai.call_args[1]
        assert set(call_kwargs.keys()) == {"client_args", "model_id", "max_tokens"}

    def test_anthropic_constructor_keys_without_api_key(self):
        """AnthropicModel gets model_id and max_tokens when no api_key."""
        config = Config(llm=LLMConfig(provider="anthropic", model="claude-3-haiku-20240307"))

        with mock.patch("strands.models.anthropic.AnthropicModel") as mock_anthropic:
            mock_anthropic.return_value = mock.sentinel.anthropic_model
            config.get_model()

        call_kwargs = mock_anthropic.call_args[1]
        assert set(call_kwargs.keys()) == {"model_id", "max_tokens"}

    def test_anthropic_constructor_keys_with_api_key(self):
        """AnthropicModel gets model_id, max_tokens, api_key when key present."""
        config = Config(llm=LLMConfig(provider="anthropic", model="claude-3", api_key="sk-ant-x"))

        with mock.patch("strands.models.anthropic.AnthropicModel") as mock_anthropic:
            mock_anthropic.return_value = mock.sentinel.anthropic_model
            config.get_model()

        call_kwargs = mock_anthropic.call_args[1]
        assert set(call_kwargs.keys()) == {"model_id", "max_tokens", "api_key"}
