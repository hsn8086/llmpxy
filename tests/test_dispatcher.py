from __future__ import annotations

import httpx
import pytest

from llmpxy.config import AppConfig, ProtocolName
from llmpxy.dispatcher import AllProvidersFailedError, ProviderDispatcher
from llmpxy.models import CanonicalContentPart, CanonicalMessage, CanonicalRequest


def _request(protocol_in: ProtocolName = "oairesp", stream: bool = False) -> CanonicalRequest:
    return CanonicalRequest(
        protocol_in=protocol_in,
        requested_model="gpt-4.1" if protocol_in != "anthropic" else "claude-sonnet",
        stream=stream,
        messages=[
            CanonicalMessage(role="user", content=[CanonicalContentPart(type="text", text="hi")])
        ],
    )


def _config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "route": {"type": "group", "name": "default"},
            "retry": {
                "provider_error_threshold": 3,
                "base_backoff_seconds": 0.001,
                "max_backoff_seconds": 0.002,
                "max_rounds": 2,
            },
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                },
                {
                    "name": "b",
                    "protocol": "anthropic",
                    "base_url": "https://api.anthropic.com",
                    "api_key_env": "B_KEY",
                    "models": {"claude-sonnet": "claude-3-7-sonnet-latest"},
                },
            ],
            "provider_groups": [{"name": "default", "strategy": "fallback", "members": ["a", "b"]}],
        }
    )


def test_group_route_supports_group_model_whitelist() -> None:
    config = AppConfig.model_validate(
        {
            "route": {"type": "group", "name": "default"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                }
            ],
            "provider_groups": [
                {
                    "name": "default",
                    "strategy": "fallback",
                    "model_whitelist_only": True,
                    "models": ["gpt-4.1"],
                    "members": ["a"],
                }
            ],
        }
    )

    dispatcher = ProviderDispatcher(config)

    assert dispatcher.route_supports_model("gpt-4.1") is True
    assert dispatcher.route_supports_model("gpt-4.1-mini") is False


@pytest.mark.asyncio
async def test_dispatcher_switches_provider_immediately_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("A_KEY", "a")
    monkeypatch.setenv("B_KEY", "b")

    attempts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        if request.url.host == "a.example":
            return httpx.Response(500, text="upstream fail")
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-3-7-sonnet-latest",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    dispatcher = ProviderDispatcher(_config())

    response, provider = await dispatcher.dispatch(
        _request(protocol_in="anthropic"), request_id="req-1"
    )

    assert provider.name == "b"
    assert response.model == "claude-3-7-sonnet-latest"
    assert len([item for item in attempts if "a.example" in item]) == 1
    assert len([item for item in attempts if "api.anthropic.com" in item]) == 1
    monkeypatch.setattr(httpx, "AsyncClient", original)


@pytest.mark.asyncio
async def test_dispatcher_raises_after_all_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("A_KEY", "a")
    monkeypatch.setenv("B_KEY", "b")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="still failing")

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    dispatcher = ProviderDispatcher(_config())

    with pytest.raises(AllProvidersFailedError):
        await dispatcher.dispatch(_request(protocol_in="anthropic"), request_id="req-2")

    monkeypatch.setattr(httpx, "AsyncClient", original)


@pytest.mark.asyncio
async def test_dispatcher_load_balances_group_members(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("A_KEY", "a")
    monkeypatch.setenv("B_KEY", "b")

    config = AppConfig.model_validate(
        {
            "route": {"type": "group", "name": "balanced"},
            "retry": {
                "provider_error_threshold": 3,
                "base_backoff_seconds": 0.001,
                "max_backoff_seconds": 0.002,
                "max_rounds": 1,
            },
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                },
                {
                    "name": "b",
                    "protocol": "oaichat",
                    "base_url": "https://b.example/v1",
                    "api_key_env": "B_KEY",
                    "models": {"gpt-4.1": "b-model"},
                },
            ],
            "provider_groups": [
                {"name": "balanced", "strategy": "load_balance", "members": ["a", "b"]}
            ],
        }
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode("utf-8")
        model = "a-model" if "a-model" in body else "b-model"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": model,
                "choices": [{"message": {"role": "assistant", "content": model}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    thresholds = iter([1.5, 0.5])
    monkeypatch.setattr("llmpxy.dispatcher.random.uniform", lambda _start, _end: next(thresholds))
    dispatcher = ProviderDispatcher(config)

    response, provider = await dispatcher.dispatch(
        _request(protocol_in="oaichat"), request_id="req-3"
    )

    assert provider.name == "b"
    assert response.model == "b-model"
    monkeypatch.setattr(httpx, "AsyncClient", original)


@pytest.mark.asyncio
async def test_dispatcher_load_balance_order_stays_fixed_across_retry_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("A_KEY", "a")
    monkeypatch.setenv("B_KEY", "b")

    config = AppConfig.model_validate(
        {
            "route": {"type": "group", "name": "balanced"},
            "retry": {
                "provider_error_threshold": 1,
                "base_backoff_seconds": 0.001,
                "max_backoff_seconds": 0.002,
                "max_rounds": 2,
            },
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                },
                {
                    "name": "b",
                    "protocol": "oaichat",
                    "base_url": "https://b.example/v1",
                    "api_key_env": "B_KEY",
                    "models": {"gpt-4.1": "b-model"},
                },
            ],
            "provider_groups": [
                {"name": "balanced", "strategy": "load_balance", "members": ["a", "b"]}
            ],
        }
    )

    attempts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.host or "")
        return httpx.Response(500, text="retryable fail")

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    monkeypatch.setattr("llmpxy.dispatcher.random.uniform", lambda _start, _end: 1.5)
    dispatcher = ProviderDispatcher(config)

    with pytest.raises(AllProvidersFailedError):
        await dispatcher.dispatch(_request(protocol_in="oaichat"), request_id="req-4")

    assert attempts == ["b.example", "a.example", "b.example", "a.example"]
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_dispatcher_load_balances_nested_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AppConfig.model_validate(
        {
            "route": {"type": "group", "name": "global"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                },
                {
                    "name": "b",
                    "protocol": "oaichat",
                    "base_url": "https://b.example/v1",
                    "api_key_env": "B_KEY",
                },
                {
                    "name": "c",
                    "protocol": "oaichat",
                    "base_url": "https://c.example/v1",
                    "api_key_env": "C_KEY",
                },
            ],
            "provider_groups": [
                {"name": "inner", "strategy": "load_balance", "members": ["a", "b"]},
                {"name": "global", "strategy": "load_balance", "members": ["inner", "c"]},
            ],
        }
    )

    thresholds = iter([1.5, 0.5, 1.5, 0.5])
    monkeypatch.setattr("llmpxy.dispatcher.random.uniform", lambda _start, _end: next(thresholds))
    dispatcher = ProviderDispatcher(config)

    assert [provider.name for provider in dispatcher.resolve_route_provider_order()] == [
        "c",
        "b",
        "a",
    ]
    assert [provider.name for provider in dispatcher.resolve_route_provider_order_for_limits()] == [
        "a",
        "b",
        "c",
    ]


def test_dispatcher_load_balance_penalizes_unhealthy_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig.model_validate(
        {
            "route": {"type": "group", "name": "balanced"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                },
                {
                    "name": "b",
                    "protocol": "oaichat",
                    "base_url": "https://b.example/v1",
                    "api_key_env": "B_KEY",
                },
            ],
            "provider_groups": [
                {"name": "balanced", "strategy": "load_balance", "members": ["a", "b"]}
            ],
        }
    )

    dispatcher = ProviderDispatcher(config)
    dispatcher._states["a"].consecutive_errors = 5
    monkeypatch.setattr("llmpxy.dispatcher.random.uniform", lambda _start, _end: 0.2)

    assert [provider.name for provider in dispatcher.resolve_route_provider_order()] == ["b", "a"]


@pytest.mark.asyncio
async def test_dispatcher_skips_provider_when_model_whitelist_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("A_KEY", "a")
    monkeypatch.setenv("B_KEY", "b")

    config = AppConfig.model_validate(
        {
            "route": {"type": "group", "name": "default"},
            "retry": {
                "provider_error_threshold": 3,
                "base_backoff_seconds": 0.001,
                "max_backoff_seconds": 0.002,
                "max_rounds": 1,
            },
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "model_whitelist_only": True,
                    "models": {"gpt-4.1": "a-model"},
                },
                {
                    "name": "b",
                    "protocol": "oaichat",
                    "base_url": "https://b.example/v1",
                    "api_key_env": "B_KEY",
                    "models": {"gpt-4.1-mini": "b-mini"},
                },
            ],
            "provider_groups": [{"name": "default", "strategy": "fallback", "members": ["a", "b"]}],
        }
    )

    attempts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "gpt-4.1-mini",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    dispatcher = ProviderDispatcher(config)

    response, provider = await dispatcher.dispatch(
        CanonicalRequest(
            protocol_in="oaichat",
            requested_model="gpt-4.1-mini",
            messages=[
                CanonicalMessage(
                    role="user", content=[CanonicalContentPart(type="text", text="hi")]
                )
            ],
        ),
        request_id="req-5",
    )

    assert provider.name == "b"
    assert response.model == "gpt-4.1-mini"
    assert len([item for item in attempts if "a.example" in item]) == 0
    assert len([item for item in attempts if "b.example" in item]) == 1
    monkeypatch.setattr(httpx, "AsyncClient", original)


@pytest.mark.asyncio
async def test_dispatcher_tries_next_provider_for_model_not_found_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("A_KEY", "a")
    monkeypatch.setenv("B_KEY", "b")

    attempts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        return httpx.Response(
            503,
            json={
                "error": {
                    "code": "model_not_found",
                    "message": "No available channel for model gpt-5-nano",
                    "type": "new_api_error",
                }
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    dispatcher = ProviderDispatcher(_config())

    with pytest.raises(AllProvidersFailedError):
        await dispatcher.dispatch(
            CanonicalRequest(
                protocol_in="oairesp",
                requested_model="gpt-5-nano",
                messages=[
                    CanonicalMessage(
                        role="user", content=[CanonicalContentPart(type="text", text="hi")]
                    )
                ],
            ),
            request_id="req-6",
        )

    assert attempts == [
        "https://a.example/v1/chat/completions",
        "https://api.anthropic.com/v1/messages",
        "https://a.example/v1/chat/completions",
        "https://api.anthropic.com/v1/messages",
    ]
    monkeypatch.setattr(httpx, "AsyncClient", original)
