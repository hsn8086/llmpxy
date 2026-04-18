from __future__ import annotations

from llmpxy.billing import calculate_usage_cost
from llmpxy.config import ProviderConfig
from llmpxy.models import CanonicalUsage
from llmpxy.protocols.oai_responses import OpenAIResponsesAdapter


def test_calculate_usage_cost_uses_cached_input_price() -> None:
    provider = ProviderConfig.model_validate(
        {
            "name": "provider-a",
            "protocol": "oaichat",
            "base_url": "https://a.example/v1",
            "api_key_env": "A_KEY",
            "pricing": {
                "default": {
                    "input_per_million_tokens_usd": 2.0,
                    "cached_input_per_million_tokens_usd": 0.2,
                    "output_per_million_tokens_usd": 8.0,
                }
            },
        }
    )

    cost = calculate_usage_cost(
        provider,
        requested_model="gpt-4.1",
        upstream_model="provider-model",
        usage=CanonicalUsage(
            input_tokens=1_000_000,
            cached_input_tokens=400_000,
            output_tokens=500_000,
            total_tokens=1_500_000,
        ),
    )

    assert cost.pricing_source == "default"
    assert cost.amount_usd == 5.28


def test_calculate_usage_cost_falls_back_to_input_price_without_cached_price() -> None:
    provider = ProviderConfig.model_validate(
        {
            "name": "provider-a",
            "protocol": "oaichat",
            "base_url": "https://a.example/v1",
            "api_key_env": "A_KEY",
            "pricing": {
                "default": {
                    "input_per_million_tokens_usd": 2.0,
                    "output_per_million_tokens_usd": 8.0,
                }
            },
        }
    )

    cost = calculate_usage_cost(
        provider,
        requested_model="gpt-4.1",
        upstream_model="provider-model",
        usage=CanonicalUsage(
            input_tokens=1_000_000,
            cached_input_tokens=400_000,
            output_tokens=500_000,
            total_tokens=1_500_000,
        ),
    )

    assert cost.amount_usd == 6.0


def test_calculate_usage_cost_inherits_default_cached_price_for_model_override() -> None:
    provider = ProviderConfig.model_validate(
        {
            "name": "provider-a",
            "protocol": "oaichat",
            "base_url": "https://a.example/v1",
            "api_key_env": "A_KEY",
            "pricing": {
                "default": {
                    "input_per_million_tokens_usd": 2.5,
                    "cached_input_per_million_tokens_usd": 0.25,
                    "output_per_million_tokens_usd": 15.0,
                },
                "models": {
                    "gpt-5.4": {
                        "input_per_million_tokens_usd": 2.5,
                        "output_per_million_tokens_usd": 15.0,
                    }
                },
            },
        }
    )

    cost = calculate_usage_cost(
        provider,
        requested_model="gpt-5.4",
        upstream_model="gpt-5.4",
        usage=CanonicalUsage(
            input_tokens=387_792,
            cached_input_tokens=387_456,
            output_tokens=1_013,
            total_tokens=388_805,
        ),
    )

    assert cost.pricing_source == "gpt-5.4"
    assert cost.amount_usd == 0.112899


def test_oairesp_parse_response_reads_cached_input_tokens() -> None:
    adapter = OpenAIResponsesAdapter()

    response = adapter.parse_response(
        {
            "id": "resp_1",
            "model": "a-model",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                }
            ],
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 40},
                "output_tokens": 25,
                "total_tokens": 125,
            },
        },
        "oairesp",
    )

    assert response.usage.input_tokens == 100
    assert response.usage.cached_input_tokens == 40
    assert response.usage.output_tokens == 25
