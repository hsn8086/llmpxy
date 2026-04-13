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


@pytest.mark.asyncio
async def test_dispatcher_switches_provider_after_three_failures(
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
    assert len([item for item in attempts if "a.example" in item]) == 3
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
    dispatcher = ProviderDispatcher(config)

    response1, provider1 = await dispatcher.dispatch(
        _request(protocol_in="oaichat"), request_id="req-3"
    )
    response2, provider2 = await dispatcher.dispatch(
        _request(protocol_in="oaichat"), request_id="req-4"
    )

    assert provider1.name == "a"
    assert provider2.name == "b"
    assert response1.model == "a-model"
    assert response2.model == "b-model"
    monkeypatch.setattr(httpx, "AsyncClient", original)


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
async def test_dispatcher_does_not_retry_model_not_found_errors(
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

    with pytest.raises(Exception):
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

    assert len(attempts) == 1
    monkeypatch.setattr(httpx, "AsyncClient", original)
