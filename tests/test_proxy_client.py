from __future__ import annotations

import httpx
import pytest

from llmpxy.config import AppConfig
from llmpxy.proxy_client import _build_target_url, post_json


@pytest.mark.asyncio
async def test_post_json_accepts_sse_completed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOKENFLUX_API_KEY", "test-key")
    sse_body = "\n".join(
        [
            "event: response.created",
            'data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}',
            "",
            "event: response.completed",
            (
                'data: {"type":"response.completed","response":'
                '{"id":"resp_1","status":"completed","model":"gpt-5.4-mini",'
                '"output":[{"id":"msg_1","type":"message","role":"assistant",'
                '"content":[{"type":"output_text","text":"ok"}]}],'
                '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}'
            ),
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=sse_body,
            headers={"content-type": "text/event-stream"},
        )

    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "tokenflux"},
            "providers": [
                {
                    "name": "tokenflux",
                    "protocol": "oairesp",
                    "base_url": "https://tokenflux.dev/v1",
                    "api_key_env": "TOKENFLUX_API_KEY",
                }
            ],
        }
    )
    provider = config.providers[0]

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        payload = await post_json(client, provider, "/responses", {"model": "gpt-5.4-mini"})

    assert payload["id"] == "resp_1"
    assert payload["status"] == "completed"
    assert payload["output"][0]["content"][0]["text"] == "ok"


@pytest.mark.asyncio
async def test_post_json_collects_output_items_from_sse_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOKENFLUX_API_KEY", "test-key")
    sse_body = "\n".join(
        [
            "event: response.output_item.done",
            (
                'data: {"type":"response.output_item.done","item":'
                '{"id":"msg_1","type":"message","role":"assistant",'
                '"content":[{"type":"output_text","text":"你好！"}]}}'
            ),
            "",
            "event: response.completed",
            (
                'data: {"type":"response.completed","response":'
                '{"id":"resp_1","status":"completed","model":"gpt-5.4-mini",'
                '"output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}'
            ),
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=sse_body,
            headers={"content-type": "text/event-stream"},
        )

    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "tokenflux"},
            "providers": [
                {
                    "name": "tokenflux",
                    "protocol": "oairesp",
                    "base_url": "https://tokenflux.dev/v1",
                    "api_key_env": "TOKENFLUX_API_KEY",
                }
            ],
        }
    )
    provider = config.providers[0]

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        payload = await post_json(client, provider, "/responses", {"model": "gpt-5.4-mini"})

    assert payload["output"][0]["content"][0]["text"] == "你好！"


def test_build_target_url_avoids_duplicate_v1_for_anthropic() -> None:
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "pneko"},
            "providers": [
                {
                    "name": "pneko",
                    "protocol": "anthropic",
                    "base_url": "https://bridge.pulseneko.com/v1",
                    "api_key_env": "PNEKO_API_KEY",
                }
            ],
        }
    )

    assert (
        _build_target_url(config.providers[0], "/v1/messages")
        == "https://bridge.pulseneko.com/v1/messages"
    )
