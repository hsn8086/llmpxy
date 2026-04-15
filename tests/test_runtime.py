from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import pytest
from fastapi import HTTPException

from llmpxy.auth import AuthenticatedApiKey
from llmpxy.models import ApiKeyUsageRecord, RequestEventRecord
from llmpxy.models import (
    CanonicalContentPart,
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalUsage,
)
from llmpxy.runtime import RuntimeManager


def _write_config(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    time.sleep(0.01)


def test_runtime_reloads_valid_config_and_keeps_previous_on_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPSTREAM_A_KEY", "upstream-a")

    config_file = tmp_path / "config.toml"
    _write_config(
        config_file,
        """
[route]
type = "provider"
name = "openai-a"

[[api_keys]]
uuid = "11111111-1111-1111-1111-111111111111"
name = "client-a"
key = "client-a"
limit_usd = 10.0

[[providers]]
name = "openai-a"
protocol = "oaichat"
base_url = "https://a.example/v1"
api_key_env = "UPSTREAM_A_KEY"
""",
    )

    runtime = RuntimeManager(config_file)
    assert runtime.config().api_keys[0].limit_usd == 10.0

    _write_config(
        config_file,
        """
[route]
type = "provider"
name = "openai-a"

[[api_keys]]
uuid = "11111111-1111-1111-1111-111111111111"
name = "client-a"
key = "client-a"
limit_usd = 20.0

[[providers]]
name = "openai-a"
protocol = "oaichat"
base_url = "https://a.example/v1"
api_key_env = "UPSTREAM_A_KEY"
""",
    )

    assert runtime.current().config.api_keys[0].limit_usd == 20.0

    _write_config(
        config_file,
        """
[route]
type = "provider"
name = "missing"

[[api_keys]]
uuid = "11111111-1111-1111-1111-111111111111"
name = "client-a"
key = "client-a"

[[providers]]
name = "openai-a"
protocol = "oaichat"
base_url = "https://a.example/v1"
api_key_env = "UPSTREAM_A_KEY"
""",
    )

    assert runtime.current().config.api_keys[0].limit_usd == 20.0


def test_runtime_selectable_providers_honor_provider_and_group_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPSTREAM_A_KEY", "upstream-a")
    monkeypatch.setenv("UPSTREAM_B_KEY", "upstream-b")

    config_file = tmp_path / "config.toml"
    _write_config(
        config_file,
        """
[route]
type = "group"
name = "default"

[[api_keys]]
uuid = "11111111-1111-1111-1111-111111111111"
name = "client-a"
key = "client-a"
provider_limits_usd = { provider-a = 1.0 }
group_limits_usd = { default = 5.0 }

[[providers]]
name = "provider-a"
protocol = "oaichat"
base_url = "https://a.example/v1"
api_key_env = "UPSTREAM_A_KEY"

[[providers]]
name = "provider-b"
protocol = "oaichat"
base_url = "https://b.example/v1"
api_key_env = "UPSTREAM_B_KEY"

[[provider_groups]]
name = "default"
strategy = "fallback"
members = ["provider-a", "provider-b"]
""",
    )

    runtime = RuntimeManager(config_file)
    api_key = runtime.authenticate("Bearer client-a")
    state = runtime.current()
    state.store.put_api_key_usage(
        ApiKeyUsageRecord(
            api_key_uuid="11111111-1111-1111-1111-111111111111",
            api_key_name="client-a",
            request_id="req-1",
            provider_name="provider-a",
            requested_model="gpt-4.1",
            upstream_model="gpt-4.1",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            cost_usd=1.0,
            created_at=int(time.time()),
        )
    )

    selectable = [provider.name for provider in runtime.selectable_provider_configs(api_key)]
    assert selectable == ["provider-b"]

    state.store.put_api_key_usage(
        ApiKeyUsageRecord(
            api_key_uuid="11111111-1111-1111-1111-111111111111",
            api_key_name="client-a",
            request_id="req-2",
            provider_name="provider-b",
            requested_model="gpt-4.1",
            upstream_model="gpt-4.1",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            cost_usd=4.0,
            created_at=int(time.time()),
        )
    )

    assert runtime.selectable_provider_configs(api_key) == []


def test_runtime_authentication_requires_valid_bearer_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPSTREAM_A_KEY", "upstream-a")

    config_file = tmp_path / "config.toml"
    _write_config(
        config_file,
        """
[route]
type = "provider"
name = "provider-a"

[[api_keys]]
uuid = "11111111-1111-1111-1111-111111111111"
name = "client-a"
key = "client-a"

[[providers]]
name = "provider-a"
protocol = "oaichat"
base_url = "https://a.example/v1"
api_key_env = "UPSTREAM_A_KEY"
""",
    )

    runtime = RuntimeManager(config_file)

    with pytest.raises(HTTPException):
        runtime.authenticate(None)

    with pytest.raises(HTTPException):
        runtime.authenticate("Bearer wrong")

    assert runtime.authenticate("Bearer client-a").name == "client-a"


def test_runtime_snapshot_includes_window_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPSTREAM_A_KEY", "upstream-a")

    config_file = tmp_path / "config.toml"
    _write_config(
        config_file,
        """
[route]
type = "provider"
name = "provider-a"

[[api_keys]]
uuid = "11111111-1111-1111-1111-111111111111"
name = "client-a"
key = "client-a"
limit_usd = 10.0

[[providers]]
name = "provider-a"
protocol = "oaichat"
base_url = "https://a.example/v1"
api_key_env = "UPSTREAM_A_KEY"
""",
    )

    runtime = RuntimeManager(config_file)
    now = int(time.time())
    state = runtime.current()
    state.store.put_request_event(
        RequestEventRecord(
            request_id="req-1",
            started_at=now - 10,
            finished_at=now - 10,
            latency_ms=500,
            protocol_in="oaichat",
            api_key_uuid="11111111-1111-1111-1111-111111111111",
            api_key_name="client-a",
            provider_name="provider-a",
            requested_model="gpt-4.1",
            upstream_model="gpt-4.1",
            status="success",
            http_status=200,
            input_tokens=100,
            cached_input_tokens=20,
            output_tokens=50,
            total_tokens=150,
            cost_usd=1.0,
        )
    )
    state.store.put_request_event(
        RequestEventRecord(
            request_id="req-2",
            started_at=now - 20,
            finished_at=now - 20,
            latency_ms=1500,
            protocol_in="oaichat",
            api_key_uuid="11111111-1111-1111-1111-111111111111",
            api_key_name="client-a",
            provider_name="provider-a",
            requested_model="gpt-4.1",
            upstream_model="gpt-4.1",
            status="provider_error",
            http_status=502,
            error_message="timeout",
            input_tokens=40,
            cached_input_tokens=10,
            output_tokens=0,
            total_tokens=40,
            cost_usd=0.0,
        )
    )
    state.store.put_request_event(
        RequestEventRecord(
            request_id="req-3",
            started_at=now - 120,
            finished_at=now - 120,
            latency_ms=250,
            protocol_in="oaichat",
            api_key_uuid="11111111-1111-1111-1111-111111111111",
            api_key_name="client-a",
            provider_name="provider-a",
            requested_model="gpt-4.1",
            upstream_model="gpt-4.1",
            status="success",
            http_status=200,
            input_tokens=30,
            cached_input_tokens=5,
            output_tokens=20,
            total_tokens=50,
            cost_usd=5.0,
        )
    )
    state.store.put_api_key_usage(
        ApiKeyUsageRecord(
            api_key_uuid="11111111-1111-1111-1111-111111111111",
            api_key_name="client-a",
            request_id="usage-1",
            provider_name="provider-a",
            requested_model="gpt-4.1",
            upstream_model="gpt-4.1",
            input_tokens=100,
            cached_input_tokens=25,
            output_tokens=50,
            total_tokens=150,
            cost_usd=8.0,
            created_at=now,
        )
    )

    snapshot = runtime.runtime_snapshot()
    window = cast(dict[str, object], snapshot["window"])
    windows = cast(dict[str, dict[str, object]], snapshot["windows"])
    providers = cast(list[dict[str, object]], snapshot["providers"])
    provider = providers[0]
    alerts = cast(list[dict[str, object]], snapshot["alerts"])

    assert window == {
        "seconds": 60,
        "request_count": 2,
        "success_count": 1,
        "error_count": 1,
        "error_rate": 0.5,
        "rps": 2 / 60,
        "success_rps": 1 / 60,
        "error_rps": 1 / 60,
        "tokens_per_second": 50 / 60,
        "cache_hit_rate": 30 / 140,
        "total_cost_usd": 1.0,
        "avg_latency_ms": 1000,
        "p95_latency_ms": 500,
        "total_input_tokens": 140,
        "total_cached_input_tokens": 30,
        "total_output_tokens": 50,
        "total_tokens": 190,
    }
    assert windows["10s"]["request_count"] == 1
    assert windows["60s"]["request_count"] == 2
    assert windows["5m"]["request_count"] == 3
    assert windows["5m"]["total_cost_usd"] == 6.0
    assert windows["5m"]["total_input_tokens"] == 170
    assert windows["5m"]["total_cached_input_tokens"] == 35
    assert windows["5m"]["total_output_tokens"] == 70
    assert provider["window"] == {
        "seconds": 60,
        "request_count": 2,
        "success_count": 1,
        "error_count": 1,
        "error_rate": 0.5,
        "rps": 2 / 60,
        "tokens_per_second": 50 / 60,
        "cache_hit_rate": 30 / 140,
        "avg_latency_ms": 1000,
        "total_input_tokens": 140,
        "total_cached_input_tokens": 30,
        "total_output_tokens": 50,
        "total_cost_usd": 1.0,
    }
    assert alerts == [
        {
            "kind": "api_key_budget",
            "name": "client-a",
            "severity": "warning",
            "message": "Budget usage 80%",
        }
    ]


def test_runtime_record_usage_preserves_millisecond_latency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPSTREAM_A_KEY", "upstream-a")

    config_file = tmp_path / "config.toml"
    _write_config(
        config_file,
        """
[route]
type = "provider"
name = "provider-a"

[[api_keys]]
uuid = "11111111-1111-1111-1111-111111111111"
name = "client-a"
key = "client-a"

[[providers]]
name = "provider-a"
protocol = "oaichat"
base_url = "https://a.example/v1"
api_key_env = "UPSTREAM_A_KEY"
""",
    )

    runtime = RuntimeManager(config_file)
    api_key = AuthenticatedApiKey(
        uuid="11111111-1111-1111-1111-111111111111",
        name="client-a",
        key="client-a",
        limit_usd=None,
        provider_limits_usd={},
        group_limits_usd={},
    )
    runtime.record_usage(
        api_key=api_key,
        request_id="req-latency",
        started_at=1234567890,
        request=CanonicalRequest(
            protocol_in="oaichat",
            requested_model="gpt-4.1",
            messages=[
                CanonicalMessage(
                    role="user",
                    content=[CanonicalContentPart(type="text", text="hi")],
                )
            ],
        ),
        response=CanonicalResponse(
            protocol_out="oaichat",
            response_id="resp-latency",
            created_at=int(time.time()),
            model="gpt-4.1",
            output_messages=[
                CanonicalMessage(
                    role="assistant",
                    content=[CanonicalContentPart(type="text", text="hello")],
                )
            ],
            usage=CanonicalUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
        provider_name="provider-a",
        latency_ms=123,
    )

    event = runtime.current().store.list_recent_request_events(1)[0]
    assert event.started_at == 1234567890
    assert event.latency_ms == 123


def test_runtime_snapshot_counts_provider_errors_by_provider_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPSTREAM_A_KEY", "upstream-a")

    config_file = tmp_path / "config.toml"
    _write_config(
        config_file,
        """
[route]
type = "provider"
name = "provider-a"

[[api_keys]]
uuid = "11111111-1111-1111-1111-111111111111"
name = "client-a"
key = "client-a"

[[providers]]
name = "provider-a"
protocol = "oaichat"
base_url = "https://a.example/v1"
api_key_env = "UPSTREAM_A_KEY"
""",
    )

    runtime = RuntimeManager(config_file)
    now = int(time.time())
    runtime.current().store.put_request_event(
        RequestEventRecord(
            request_id="req-provider-error",
            started_at=now,
            finished_at=now,
            latency_ms=42,
            protocol_in="oaichat",
            api_key_uuid="11111111-1111-1111-1111-111111111111",
            api_key_name="client-a",
            provider_name="provider-a",
            requested_model="gpt-4.1",
            status="provider_error",
            http_status=502,
            error_message="boom",
        )
    )

    snapshot = runtime.runtime_snapshot()
    providers = cast(list[dict[str, object]], snapshot["providers"])
    provider = providers[0]

    assert cast(dict[str, object], provider["window"])["request_count"] == 1
    assert cast(dict[str, object], provider["window"])["error_count"] == 1
    assert provider["recent_attempt_errors"] == 0


def test_runtime_snapshot_exposes_provider_attempt_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPSTREAM_A_KEY", "upstream-a")

    config_file = tmp_path / "config.toml"
    _write_config(
        config_file,
        """
[route]
type = "provider"
name = "provider-a"

[[api_keys]]
uuid = "11111111-1111-1111-1111-111111111111"
name = "client-a"
key = "client-a"

[[providers]]
name = "provider-a"
protocol = "oaichat"
base_url = "https://a.example/v1"
api_key_env = "UPSTREAM_A_KEY"
""",
    )

    runtime = RuntimeManager(config_file)
    runtime.stats().record_provider_error("provider-a", "transient boom")
    runtime.stats().record_provider_error("provider-a", "transient boom again")

    snapshot = runtime.runtime_snapshot()
    providers = cast(list[dict[str, object]], snapshot["providers"])
    provider = providers[0]

    assert provider["consecutive_errors"] == 2
    assert provider["recent_attempt_errors"] == 2
    assert provider["recent_errors"] == 2
