from __future__ import annotations

from pathlib import Path

import pytest

from llmpxy.config import AppConfig, load_config


def test_load_multi_protocol_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
[server]
host = "127.0.0.1"
port = 8080

[route]
type = "group"
name = "default"

[network]
proxy = "http://127.0.0.1:7893"

[storage]
backend = "sqlite"
sqlite_path = "./data/test.db"

[[providers]]
name = "openai-a"
protocol = "oaichat"
base_url = "https://a.example/v1"
api_key_env = "A_KEY"

[[providers]]
name = "anthropic-a"
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key_env = "B_KEY"

[[provider_groups]]
name = "default"
strategy = "fallback"
members = ["openai-a", "anthropic-a"]
""",
        encoding="utf-8",
    )

    config = load_config(config_file)
    assert isinstance(config, AppConfig)
    assert config.route.type == "group"
    assert config.network.proxy == "http://127.0.0.1:7893"
    assert config.providers[0].protocol == "oaichat"
    assert config.providers[1].protocol == "anthropic"


def test_duplicate_provider_names_rejected() -> None:
    with pytest.raises(ValueError):
        AppConfig.model_validate(
            {
                "route": {"type": "provider", "name": "dup"},
                "providers": [
                    {
                        "name": "dup",
                        "protocol": "oaichat",
                        "base_url": "https://a",
                        "api_key_env": "A",
                    },
                    {
                        "name": "dup",
                        "protocol": "anthropic",
                        "base_url": "https://b",
                        "api_key_env": "B",
                    },
                ],
            }
        )


def test_group_cycle_rejected() -> None:
    with pytest.raises(ValueError):
        AppConfig.model_validate(
            {
                "route": {"type": "group", "name": "g1"},
                "providers": [
                    {
                        "name": "a",
                        "protocol": "oaichat",
                        "base_url": "https://a",
                        "api_key_env": "A",
                    },
                ],
                "provider_groups": [
                    {"name": "g1", "strategy": "fallback", "members": ["g2"]},
                    {"name": "g2", "strategy": "fallback", "members": ["g1"]},
                ],
            }
        )


def test_provider_model_whitelist_only_requires_explicit_mapping() -> None:
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "openai-a"},
            "providers": [
                {
                    "name": "openai-a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "model_whitelist_only": True,
                    "models": {"gpt-4.1": "a-model"},
                }
            ],
        }
    )

    provider = config.providers[0]
    assert provider.supports_model("gpt-4.1") is True
    assert provider.supports_model("gpt-4.1-mini") is False


def test_load_config_reads_dotenv_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DOTENV_TEST_KEY", raising=False)

    config_file = tmp_path / "config.toml"
    dotenv_file = tmp_path / ".env"

    dotenv_file.write_text('DOTENV_TEST_KEY="from-dotenv"\n', encoding="utf-8")
    config_file.write_text(
        """
[route]
type = "provider"
name = "openai-a"

[[providers]]
name = "openai-a"
protocol = "oaichat"
base_url = "https://a.example/v1"
api_key_env = "DOTENV_TEST_KEY"
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.providers[0].api_key() == "from-dotenv"


def test_api_key_limits_reference_known_provider_and_group() -> None:
    with pytest.raises(ValueError):
        AppConfig.model_validate(
            {
                "route": {"type": "provider", "name": "openai-a"},
                "api_keys": [
                    {
                        "uuid": "11111111-1111-1111-1111-111111111111",
                        "name": "client-a",
                        "key": "client-a-key",
                        "provider_limits_usd": {"missing": 1.0},
                    }
                ],
                "providers": [
                    {
                        "name": "openai-a",
                        "protocol": "oaichat",
                        "base_url": "https://a.example/v1",
                        "api_key_env": "A_KEY",
                    }
                ],
            }
        )

    with pytest.raises(ValueError):
        AppConfig.model_validate(
            {
                "route": {"type": "provider", "name": "openai-a"},
                "api_keys": [
                    {
                        "uuid": "11111111-1111-1111-1111-111111111111",
                        "name": "client-a",
                        "key": "client-a-key",
                        "group_limits_usd": {"missing-group": 1.0},
                    }
                ],
                "providers": [
                    {
                        "name": "openai-a",
                        "protocol": "oaichat",
                        "base_url": "https://a.example/v1",
                        "api_key_env": "A_KEY",
                    }
                ],
                "provider_groups": [
                    {"name": "default", "strategy": "fallback", "members": ["openai-a"]}
                ],
            }
        )


def test_provider_pricing_supports_default_and_model_override() -> None:
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "openai-a"},
            "providers": [
                {
                    "name": "openai-a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "pricing": {
                        "default": {
                            "input_per_million_tokens_usd": 1.0,
                            "cached_input_per_million_tokens_usd": 0.1,
                            "output_per_million_tokens_usd": 2.0,
                        },
                        "models": {
                            "gpt-4.1": {
                                "input_per_million_tokens_usd": 3.0,
                                "cached_input_per_million_tokens_usd": 0.3,
                                "output_per_million_tokens_usd": 4.0,
                            }
                        },
                    },
                }
            ],
        }
    )

    provider = config.providers[0]
    assert provider.resolve_pricing("gpt-4.1", "mapped-model") is not None
    assert provider.resolve_pricing("missing", "mapped-model") == provider.pricing.default
    assert provider.pricing.default is not None
    assert provider.pricing.default.cached_input_per_million_tokens_usd == 0.1


def test_provider_pricing_rejects_negative_cached_input_price() -> None:
    with pytest.raises(ValueError):
        AppConfig.model_validate(
            {
                "route": {"type": "provider", "name": "openai-a"},
                "providers": [
                    {
                        "name": "openai-a",
                        "protocol": "oaichat",
                        "base_url": "https://a.example/v1",
                        "api_key_env": "A_KEY",
                        "pricing": {
                            "default": {
                                "input_per_million_tokens_usd": 1.0,
                                "cached_input_per_million_tokens_usd": -0.1,
                                "output_per_million_tokens_usd": 2.0,
                            }
                        },
                    }
                ],
            }
        )


def test_api_key_uuid_must_be_unique() -> None:
    with pytest.raises(ValueError):
        AppConfig.model_validate(
            {
                "route": {"type": "provider", "name": "openai-a"},
                "api_keys": [
                    {
                        "uuid": "11111111-1111-1111-1111-111111111111",
                        "name": "client-a",
                        "key": "client-a-key",
                    },
                    {
                        "uuid": "11111111-1111-1111-1111-111111111111",
                        "name": "client-b",
                        "key": "client-b-key",
                    },
                ],
                "providers": [
                    {
                        "name": "openai-a",
                        "protocol": "oaichat",
                        "base_url": "https://a.example/v1",
                        "api_key_env": "A_KEY",
                    }
                ],
            }
        )


def test_provider_group_model_whitelist_requires_models() -> None:
    with pytest.raises(ValueError):
        AppConfig.model_validate(
            {
                "route": {"type": "group", "name": "default"},
                "providers": [
                    {
                        "name": "openai-a",
                        "protocol": "oaichat",
                        "base_url": "https://a.example/v1",
                        "api_key_env": "A_KEY",
                    }
                ],
                "provider_groups": [
                    {
                        "name": "default",
                        "strategy": "fallback",
                        "model_whitelist_only": True,
                        "members": ["openai-a"],
                    }
                ],
            }
        )
