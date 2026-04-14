from __future__ import annotations

from dataclasses import dataclass

from llmpxy.config import ProviderConfig
from llmpxy.models import CanonicalUsage


@dataclass(frozen=True)
class UsageCost:
    amount_usd: float
    pricing_source: str | None


def calculate_usage_cost(
    provider: ProviderConfig,
    requested_model: str,
    upstream_model: str,
    usage: CanonicalUsage,
) -> UsageCost:
    pricing = provider.resolve_pricing(requested_model, upstream_model)
    if pricing is None:
        return UsageCost(amount_usd=0.0, pricing_source=None)

    input_cost = usage.input_tokens / 1_000_000 * pricing.input_per_million_tokens_usd
    output_cost = usage.output_tokens / 1_000_000 * pricing.output_per_million_tokens_usd
    pricing_source = (
        requested_model if requested_model in provider.pricing.models else upstream_model
    )
    if pricing_source not in provider.pricing.models:
        pricing_source = "default"
    return UsageCost(amount_usd=input_cost + output_cost, pricing_source=pricing_source)
