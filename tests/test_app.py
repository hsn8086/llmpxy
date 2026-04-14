from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from llmpxy.app import create_app
from llmpxy.config import AppConfig
from llmpxy.dispatcher import ProviderDispatcher
from llmpxy.models import CanonicalResponse, CanonicalUsage
from llmpxy.protocols.anthropic_messages import AnthropicAdapter
from llmpxy.protocols.oai_chat import OpenAIChatAdapter
from llmpxy.storage_sqlite import SQLiteConversationStore


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
            "storage": {"backend": "sqlite", "sqlite_path": "./data/test.db", "ttl_seconds": 60},
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


def _question_tool_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "header": {"type": "string"},
                        "multiple": {"type": "boolean"},
                    },
                    "required": None,
                },
            },
            "custom": {"type": "boolean"},
        },
        "required": None,
    }


def _assert_normalized_question_schema(schema: dict[str, Any]) -> None:
    assert schema["required"] == ["questions", "custom"]
    properties = cast(dict[str, Any], schema["properties"])
    questions_schema = cast(dict[str, Any], properties["questions"])
    items_schema = cast(dict[str, Any], questions_schema["items"])
    assert items_schema["required"] == ["header", "multiple"]


@pytest.mark.asyncio
async def test_oairesp_previous_response_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("A_KEY", "a")
    monkeypatch.setenv("B_KEY", "b")

    import httpx

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.example":
            return httpx.Response(500, text="provider a down")
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-3-7-sonnet-latest",
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)

    config = _config()
    store = SQLiteConversationStore(tmp_path / "app.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post("/v1/responses", json={"model": "claude-sonnet", "input": "hi"})
    assert response.status_code == 200
    first_id = response.json()["id"]

    second = client.post(
        "/v1/responses",
        json={"model": "claude-sonnet", "input": "again", "previous_response_id": first_id},
    )
    assert second.status_code == 200
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oaichat_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("A_KEY", "a")
    monkeypatch.setenv("B_KEY", "b")

    import httpx

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "a-model",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = _config()
    store = SQLiteConversationStore(tmp_path / "chat.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_group_model_whitelist_blocks_disallowed_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("A_KEY", "a")

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
    store = SQLiteConversationStore(tmp_path / "group-whitelist.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Requested model is not allowed by route"


def test_anthropic_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("A_KEY", "a")
    monkeypatch.setenv("B_KEY", "b")

    import httpx

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.example":
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_1",
                    "model": "a-model",
                    "choices": [{"message": {"role": "assistant", "content": "hello oaichat"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-3-7-sonnet-latest",
                "content": [{"type": "text", "text": "hello anthropic"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = _config()
    store = SQLiteConversationStore(tmp_path / "anthropic.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "max_tokens": 64,
        },
    )
    assert response.status_code == 200
    assert response.json()["type"] == "message"
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_stream(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("A_KEY", "a")
    monkeypatch.setenv("B_KEY", "b")

    import httpx

    async def stream_success():
        yield b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-7-sonnet-latest","content":[]}}\n\n'
        yield b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"he"}}\n\n'
        yield b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"llo"}}\n\n'
        yield b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":1,"output_tokens":1}}\n\n'
        yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.example":
            return httpx.Response(500, text="provider a down")
        return httpx.Response(200, content=stream_success())

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = _config()
    store = SQLiteConversationStore(tmp_path / "stream.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    with client.stream(
        "POST", "/v1/responses", json={"model": "claude-sonnet", "input": "hi", "stream": True}
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    event_types = [event["type"] for event in events]
    assert "response.created" in event_types
    assert "response.content_part.added" in event_types
    assert "response.output_text.delta" in event_types
    assert "response.output_text.done" in event_types
    assert "response.content_part.done" in event_types
    assert "response.completed" in event_types
    assert events[0]["response"]["id"]
    output_item_added = next(
        event for event in events if event["type"] == "response.output_item.added"
    )
    assert output_item_added["item"]["content"] == []
    content_part_added = next(
        event for event in events if event["type"] == "response.content_part.added"
    )
    assert content_part_added["part"]["type"] == "output_text"
    assert events[-1]["response"]["object"] == "response"
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_stream_exposes_reasoning_summary_from_oaichat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def stream_success():
        first = json.dumps(
            {
                "id": "chat1",
                "model": "a-model",
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "internal reasoning summary",
                            "content": "ok",
                        }
                    }
                ],
            }
        )
        second = json.dumps(
            {
                "id": "chat1",
                "model": "a-model",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
        yield f"data: {first}\n\n".encode()
        yield f"data: {second}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, content=stream_success())

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-5.4": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oairesp-stream-reasoning.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-5.4",
            "input": "hi",
            "stream": True,
            "reasoning": {"effort": "high"},
            "include": ["reasoning.encrypted_content"],
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "metadata" not in captured_body
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    reasoning_added = next(
        event
        for event in events
        if event["type"] == "response.output_item.added" and event["item"]["type"] == "reasoning"
    )
    assert reasoning_added["item"]["summary"][0]["text"] == "internal reasoning summary"
    assert reasoning_added["item"]["encrypted_content"] == "internal reasoning summary"
    completed = next(event for event in events if event["type"] == "response.completed")
    reasoning_item = next(
        item for item in completed["response"]["output"] if item["type"] == "reasoning"
    )
    assert reasoning_item["summary"][0]["text"] == "internal reasoning summary"
    assert reasoning_item["encrypted_content"] == "internal reasoning summary"
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_anthropic_stream_message_start_includes_usage() -> None:
    adapter = AnthropicAdapter()
    response = CanonicalResponse(
        response_id="msg_1",
        protocol_out="anthropic",
        model="claude-sonnet-4-6",
        created_at=1,
        usage=CanonicalUsage(input_tokens=3, output_tokens=5, total_tokens=8),
    )

    async def collect() -> list[str]:
        return [chunk async for chunk in adapter.format_stream(response, [])]

    chunks = asyncio.run(collect())
    assert chunks
    first_chunk = chunks[0]
    assert first_chunk.startswith("event: message_start\n")
    payload = json.loads(first_chunk.split("data: ", 1)[1])
    assert payload["type"] == "message_start"
    assert payload["message"]["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_oairesp_stream_exposes_function_call_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")
    monkeypatch.setenv("B_KEY", "b")

    import httpx

    async def stream_success():
        first = json.dumps(
            {
                "id": "chat1",
                "model": "a-model",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":',
                                    },
                                }
                            ]
                        }
                    }
                ],
            }
        )
        second = json.dumps(
            {
                "id": "chat1",
                "model": "a-model",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '"Paris"}',
                                    },
                                }
                            ]
                        }
                    }
                ],
            }
        )
        yield f"data: {first}\n\n".encode("utf-8")
        yield f"data: {second}\n\n".encode("utf-8")
        yield b'data: {"id":"chat1","model":"a-model","choices":[{"delta":{}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
        yield b"data: [DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream_success())

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "stream-tools.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": "weather?",
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather by city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    event_types = [event["type"] for event in events]
    assert "response.function_call_arguments.delta" in event_types
    assert "response.function_call_arguments.done" in event_types
    function_call_item = next(
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done" and event["item"]["type"] == "function_call"
    )
    assert function_call_item["name"] == "get_weather"
    assert function_call_item["arguments"] == '{"city":"Paris"}'
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_stream_parses_type_based_response_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    async def stream_success():
        yield (
            b'data: {"type":"response.created","response":{"id":"resp_1","model":"a-model"}}\n\n'
        )
        yield (b'data: {"type":"response.output_text.delta","delta":"He"}\n\n')
        yield (b'data: {"type":"response.output_text.delta","delta":"llo"}\n\n')
        yield (
            b'data: {"type":"response.completed","response":{"id":"resp_1","model":"a-model","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
        )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream_success())

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oairesp",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-5.4": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oairesp-type-stream.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/responses",
        json={"model": "gpt-5.4", "input": "hi", "stream": True},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    deltas = [
        event["delta"] for event in events if event.get("type") == "response.output_text.delta"
    ]
    completed = next(event for event in events if event["type"] == "response.completed")
    assert deltas == ["He", "llo"]
    output = cast(list[dict[str, Any]], completed["response"]["output"])
    assert output[0]["content"][0]["text"] == "Hello"
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_stream_preserves_function_call_events_from_type_based_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    async def stream_success():
        yield (
            b'data: {"type":"response.created","response":{"id":"resp_1","model":"a-model"}}\n\n'
        )
        yield (b'data: {"type":"response.output_text.delta","delta":"prep"}\n\n')
        yield (
            b'data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"read","arguments":"","status":"in_progress"}}\n\n'
        )
        yield (
            b'data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\\"filePath\\":\\"/tmp/x\\"}"}\n\n'
        )
        yield (
            b'data: {"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":"{\\"filePath\\":\\"/tmp/x\\"}"}\n\n'
        )
        yield (
            b'data: {"type":"response.output_item.done","item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"read","arguments":"{\\"filePath\\":\\"/tmp/x\\"}","status":"completed"}}\n\n'
        )
        yield (
            b'data: {"type":"response.completed","response":{"id":"resp_1","model":"a-model","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
        )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream_success())

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oairesp",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-5.4": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oairesp-type-stream-tools.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/responses",
        json={"model": "gpt-5.4", "input": "hi", "stream": True},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    function_call_done = next(
        event
        for event in events
        if event["type"] == "response.output_item.done" and event["item"]["type"] == "function_call"
    )
    completed = next(event for event in events if event["type"] == "response.completed")
    output = cast(list[dict[str, Any]], completed["response"]["output"])
    assert function_call_done["item"]["call_id"] == "call_1"
    assert function_call_done["item"]["arguments"] == '{"filePath":"/tmp/x"}'
    assert output[0]["type"] == "function_call"
    assert output[0]["call_id"] == "call_1"
    assert output[0]["name"] == "read"
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_stream_does_not_merge_distinct_tool_calls_same_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    async def stream_success():
        first = json.dumps(
            {
                "id": "chat1",
                "model": "a-model",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read",
                                        "arguments": '{"filePath":"/a"}',
                                    },
                                }
                            ]
                        }
                    }
                ],
            }
        )
        second = json.dumps(
            {
                "id": "chat1",
                "model": "a-model",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "read",
                                        "arguments": '{"filePath":"/b"}',
                                    },
                                }
                            ]
                        }
                    }
                ],
            }
        )
        yield f"data: {first}\n\n".encode("utf-8")
        yield f"data: {second}\n\n".encode("utf-8")
        yield b'data: {"id":"chat1","model":"a-model","choices":[{"delta":{}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
        yield b"data: [DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream_success())

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "stream-tool-index.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/responses",
        json={"model": "gpt-4.1", "input": "x", "stream": True},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    function_call_done_items = [
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done" and event["item"]["type"] == "function_call"
    ]
    assert len(function_call_done_items) == 2
    assert function_call_done_items[0]["arguments"] == '{"filePath":"/a"}'
    assert function_call_done_items[1]["arguments"] == '{"filePath":"/b"}'
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_preserves_developer_role_for_anthropic_upstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")
    monkeypatch.setenv("B_KEY", "b")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.example":
            return httpx.Response(500, text="provider a down")
        captured_body.update(json.loads(request.content.decode("utf-8")))
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
    config = _config()
    store = SQLiteConversationStore(tmp_path / "developer.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "claude-sonnet",
            "input": [
                {"role": "developer", "content": "You are a title generator."},
                {"role": "user", "content": [{"type": "input_text", "text": "test"}]},
            ],
        },
    )

    assert response.status_code == 200
    assert captured_body["system"] == "You are a title generator."
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_function_tools_are_converted_for_oaichat_upstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")
    monkeypatch.setenv("B_KEY", "b")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "a-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Paris"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "tools.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": "what is the weather?",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather by city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert isinstance(captured_body.get("tools"), list)
    tools_value = cast(list[dict[str, Any]], captured_body["tools"])
    first_tool = tools_value[0]
    assert first_tool["type"] == "function"
    function = cast(dict[str, Any], first_tool["function"])
    assert function["name"] == "get_weather"
    output = response.json()["output"]
    function_call_item = next(item for item in output if item["type"] == "function_call")
    assert function_call_item["name"] == "get_weather"
    assert function_call_item["arguments"] == '{"city":"Paris"}'
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_tool_parameters_are_normalized_before_oaichat_forwarding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "a-model",
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
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "tool-schema.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": "run diagnostics",
            "tools": [
                {
                    "type": "function",
                    "name": "diagnostics",
                    "description": "Run diagnostics",
                    "parameters": _question_tool_schema(),
                }
            ],
        },
    )

    assert response.status_code == 200
    tools_value = cast(list[dict[str, Any]], captured_body["tools"])
    function = cast(dict[str, Any], tools_value[0]["function"])
    parameters = cast(dict[str, Any], function["parameters"])
    _assert_normalized_question_schema(parameters)
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oaichat_input_tools_are_normalized_before_forwarding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "a-model",
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
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "chat-tool-schema.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "run diagnostics"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "diagnostics",
                        "description": "Run diagnostics",
                        "parameters": _question_tool_schema(),
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    tools_value = cast(list[dict[str, Any]], captured_body["tools"])
    function = cast(dict[str, Any], tools_value[0]["function"])
    parameters = cast(dict[str, Any], function["parameters"])
    _assert_normalized_question_schema(parameters)
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_anthropic_input_tools_are_normalized_before_forwarding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "a-model",
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
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "anthropic",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"claude-sonnet": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "anthropic-tool-schema.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "run diagnostics"}]}
            ],
            "max_tokens": 64,
            "tools": [
                {
                    "name": "diagnostics",
                    "description": "Run diagnostics",
                    "input_schema": _question_tool_schema(),
                }
            ],
        },
    )

    assert response.status_code == 200
    tools_value = cast(list[dict[str, Any]], captured_body["tools"])
    input_schema = cast(dict[str, Any], tools_value[0]["input_schema"])
    _assert_normalized_question_schema(input_schema)
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oaichat_to_oairesp_forwards_system_messages_as_instructions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "a-model",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oairesp",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oairesp-instructions.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "messages": [
                {"role": "system", "content": "Follow policy"},
                {"role": "developer", "content": "Ask one question at a time"},
                {"role": "user", "content": "hi"},
            ],
        },
    )

    assert response.status_code == 200
    assert captured_body["instructions"] == "Follow policy\n\nAsk one question at a time"
    input_items = cast(list[dict[str, Any]], captured_body["input"])
    assert [item["role"] for item in input_items] == ["system", "developer", "user"]
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oaichat_to_oairesp_maps_tool_messages_to_function_call_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "a-model",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oairesp",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oairesp-tool-output.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "glob", "arguments": '{"pattern":"**/*.md"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "README.md"},
                {"role": "user", "content": "继续"},
            ],
        },
    )

    assert response.status_code == 200
    input_items = cast(list[dict[str, Any]], captured_body["input"])
    function_call_item = input_items[0]
    assert function_call_item["type"] == "function_call"
    assert function_call_item["id"] == "fc_1"
    assert function_call_item["call_id"] == "call_1"
    assert function_call_item["name"] == "glob"
    assistant_item = input_items[1]
    assert assistant_item["role"] == "assistant"
    tool_output_item = input_items[2]
    assert tool_output_item["type"] == "function_call_output"
    assert tool_output_item["call_id"] == "call_1"
    assert tool_output_item["output"] == "README.md"
    assert input_items[3]["role"] == "user"
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_response_format_is_forwarded_as_text_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "a-model",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": '{"name":"Jane"}'}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oairesp",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oairesp-text-format.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Jane, 54 years old"}],
            "response_format": {
                "type": "json_schema",
                "name": "person",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        },
    )

    assert response.status_code == 200
    text = cast(dict[str, Any], captured_body["text"])
    assert text["format"]["name"] == "person"
    assert "response_format" not in captured_body
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_includes_empty_instructions_when_missing_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "a-model",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oairesp",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-5.4": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oairesp-empty-instructions.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        },
    )

    assert response.status_code == 200
    assert captured_body["instructions"] == ""
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_forwards_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "a-model",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oairesp",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-5.4": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oairesp-reasoning.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "reasoning": {"effort": "high"},
        },
    )

    assert response.status_code == 200
    assert captured_body["reasoning"] == {"effort": "high", "summary": "auto"}
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_omits_reasoning_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "a-model",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oairesp",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-5.4": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oairesp-no-reasoning.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        },
    )

    assert response.status_code == 200
    assert "reasoning" not in captured_body
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_maps_minimal_reasoning_effort_to_low(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "a-model",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oairesp",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-5.4": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oairesp-minimal-reasoning.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "reasoning": {"effort": "minimal"},
        },
    )

    assert response.status_code == 200
    assert captured_body["reasoning"] == {"effort": "low", "summary": "auto"}
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_preserves_explicit_reasoning_summary_setting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "a-model",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oairesp",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-5.4": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oairesp-explicit-reasoning-summary.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "reasoning": {"effort": "high", "summary": "detailed"},
        },
    )

    assert response.status_code == 200
    assert captured_body["reasoning"] == {"effort": "high", "summary": "detailed"}
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_to_oaichat_forwards_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "a-model",
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
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-5.4": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oairesp-to-oaichat-reasoning.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "reasoning": {"effort": "high"},
        },
    )

    assert response.status_code == 200
    assert captured_body["reasoning_effort"] == "high"
    assert "reasoning" not in captured_body
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oaichat_maps_minimal_reasoning_effort_to_low(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "a-model",
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
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-5.4": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oaichat-minimal-reasoning.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning": {"effort": "minimal"},
        },
    )

    assert response.status_code == 200
    assert captured_body["reasoning_effort"] == "low"
    assert "reasoning" not in captured_body
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oaichat_accepts_reasoning_effort_input_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "a-model",
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
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-5.4": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oaichat-reasoning-effort-input.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
        },
    )

    assert response.status_code == 200
    assert captured_body["reasoning_effort"] == "high"
    assert "reasoning" not in captured_body
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oaichat_preserves_reasoning_content_in_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "a-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                            "reasoning_content": "internal reasoning summary",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-5.4": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oaichat-reasoning-content.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert (
        response.json()["choices"][0]["message"]["reasoning_content"]
        == "internal reasoning summary"
    )
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_exposes_reasoning_summary_in_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "a-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                            "reasoning_content": "internal reasoning summary",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-5.4": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oairesp-reasoning-content.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4",
            "input": "hi",
            "reasoning": {"effort": "high", "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
        },
    )

    assert response.status_code == 200
    assert "metadata" not in captured_body
    reasoning_item = next(item for item in response.json()["output"] if item["type"] == "reasoning")
    assert reasoning_item["summary"][0]["text"] == "internal reasoning summary"
    assert reasoning_item["encrypted_content"] == "internal reasoning summary"
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_defaults_reasoning_summary_to_auto(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "a-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                            "reasoning_content": "internal reasoning summary",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-5.4": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oairesp-default-reasoning-summary.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4",
            "input": "hi",
            "reasoning": {"effort": "high"},
        },
    )

    assert response.status_code == 200
    assert captured_body["reasoning_effort"] == "high"
    assert "reasoning" not in captured_body
    reasoning_item = next(item for item in response.json()["output"] if item["type"] == "reasoning")
    assert reasoning_item["summary"][0]["text"] == "internal reasoning summary"
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_stream_history_function_calls_use_fc_item_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            text="\n\n".join(
                [
                    'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_1"}}',
                    (
                        "event: response.completed\n"
                        'data: {"type":"response.completed","response":{'
                        '"id":"resp_1","model":"a-model","status":"completed",'
                        '"output":[{"id":"msg_1","type":"message","role":"assistant",'
                        '"content":[{"type":"output_text","text":"ok"}]}],'
                        '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}'
                    ),
                ]
            ),
            headers={"content-type": "text/event-stream"},
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oairesp",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-5.4": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "oairesp-stream-fc-id.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4",
            "stream": True,
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "glob",
                    "arguments": '{"pattern":"**/*.md"}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "README.md"},
                {"role": "user", "content": [{"type": "input_text", "text": "continue"}]},
            ],
        },
    )

    assert response.status_code == 200
    input_items = cast(list[dict[str, Any]], captured_body["input"])
    function_call_item = next(item for item in input_items if item.get("type") == "function_call")
    assert function_call_item["id"] == "fc_1"
    assert function_call_item["call_id"] == "call_1"
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_accepts_function_call_output_items(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "a-model",
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "tool-output.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": [
                {"type": "function_call_output", "call_id": "call_1", "output": {"status": "ok"}}
            ],
        },
    )

    assert response.status_code == 200
    messages = cast(list[dict[str, Any]], captured_body["messages"])
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "call_1"
    assert messages[0]["content"] == '{"status": "ok"}'
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_accepts_assistant_message_tool_calls_with_null_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "a-model",
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "assistant-tool-calls-null-content.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "glob", "arguments": '{"pattern":"**/*.md"}'},
                        }
                    ],
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "README.md"},
            ],
        },
    )

    assert response.status_code == 200
    messages = cast(list[dict[str, Any]], captured_body["messages"])
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == ""
    tool_calls = cast(list[dict[str, Any]], messages[0]["tool_calls"])
    assert tool_calls[0]["id"] == "call_1"
    assert tool_calls[0]["function"]["name"] == "glob"
    assert messages[1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "README.md",
    }
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_merges_function_call_item_into_following_assistant_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "a-model",
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "merge-function-call-item.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "glob",
                    "arguments": '{"pattern":"**/*.md"}',
                },
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Looking up files"}],
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "README.md"},
            ],
        },
    )

    assert response.status_code == 200
    messages = cast(list[dict[str, Any]], captured_body["messages"])
    assert messages[0] == {
        "role": "assistant",
        "content": "Looking up files",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "glob", "arguments": '{"pattern":"**/*.md"}'},
            }
        ],
    }
    assert messages[1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "README.md",
    }
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_ignores_reasoning_items_between_function_call_and_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "a-model",
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "ignore-reasoning-between-tool-call.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "todowrite",
                    "arguments": '{"todos":[{"content":"check logs","status":"in_progress","priority":"high"}]}',
                },
                {"type": "reasoning", "summary": []},
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Checking logs now"}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "[Old tool result content cleared]",
                },
            ],
        },
    )

    assert response.status_code == 200
    messages = cast(list[dict[str, Any]], captured_body["messages"])
    assert messages[0] == {
        "role": "assistant",
        "content": "Checking logs now",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "todowrite",
                    "arguments": '{"todos":[{"content":"check logs","status":"in_progress","priority":"high"}]}',
                },
            }
        ],
    }
    assert messages[1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "[Old tool result content cleared]",
    }
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_previous_response_id_preserves_empty_assistant_tool_call_content_for_oaichat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_bodies: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content.decode("utf-8")))
        body = captured_bodies[-1]
        if len(captured_bodies) == 1:
            assert body["messages"] == [{"role": "user", "content": "hi"}]
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_1",
                    "model": "a-model",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "glob",
                                            "arguments": '{"pattern":"**/*.md"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        assert body["messages"][0]["role"] == "assistant"
        assert body["messages"][0]["tool_calls"][0]["id"] == "call_1"
        assert body["messages"][0]["content"] == ""
        assert body["messages"][1] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "README.md",
        }
        assert body["messages"][2] == {"role": "user", "content": "continue"}
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_2",
                "model": "a-model",
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "storage": {
                "backend": "sqlite",
                "sqlite_path": str(tmp_path / "history-tool-call.db"),
                "ttl_seconds": 60,
            },
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-5.4": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "history-tool-call.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    first = client.post("/v1/responses", json={"model": "gpt-5.4", "input": "hi"})
    assert first.status_code == 200
    first_id = first.json()["id"]

    second = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4",
            "previous_response_id": first_id,
            "input": [
                {"type": "function_call_output", "call_id": "call_1", "output": "README.md"},
                {"role": "user", "content": [{"type": "input_text", "text": "continue"}]},
            ],
        },
    )

    assert second.status_code == 200
    assert len(captured_bodies) == 2
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oaichat_serializes_empty_tool_message_content_as_empty_string() -> None:
    from llmpxy.models import CanonicalMessage
    from llmpxy.protocols.oai_chat import _message_to_oaichat

    payload = _message_to_oaichat(CanonicalMessage(role="tool"))

    assert payload == {"role": "tool", "content": ""}


@pytest.mark.asyncio
async def test_oaichat_parse_stream_preserves_tool_name_when_later_deltas_are_empty() -> None:
    adapter = OpenAIChatAdapter()

    async def lines() -> AsyncIterator[str]:
        chunks = [
            {
                "id": "chatcmpl_1",
                "model": "gpt-5.4",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "todowrite", "arguments": '{"todos":['},
                                }
                            ]
                        }
                    }
                ],
            },
            {
                "id": "chatcmpl_1",
                "model": "gpt-5.4",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"name": "", "arguments": '{"content":"check"}'},
                                }
                            ]
                        }
                    }
                ],
            },
            {
                "id": "chatcmpl_1",
                "model": "gpt-5.4",
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            },
        ]
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}"
        yield "data: [DONE]"

    response, events = await adapter.parse_stream(lines(), "oairesp")

    assert len(response.output_messages) == 1
    tool_calls = response.output_messages[0].tool_calls
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "todowrite"
    assert tool_calls[0].arguments == '{"todos":[{"content":"check"}'
    assert [event.event_type for event in events] == [
        "function_call_arguments_delta",
        "function_call_arguments_delta",
    ]


def test_oaichat_parse_response_reads_cached_prompt_tokens() -> None:
    adapter = OpenAIChatAdapter()

    response = adapter.parse_response(
        {
            "id": "chatcmpl_1",
            "model": "a-model",
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "usage": {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 40},
                "completion_tokens": 25,
                "total_tokens": 125,
            },
        },
        "oaichat",
    )

    assert response.usage.input_tokens == 100
    assert response.usage.cached_input_tokens == 40
    assert response.usage.output_tokens == 25


def test_oaichat_build_request_includes_stream_usage_option() -> None:
    adapter = OpenAIChatAdapter()

    request = adapter.parse_request(
        {
            "model": "gpt-5.4",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        strip_unsupported_fields=True,
    )

    _path, payload = adapter.build_request(request, "gpt-5.4")

    assert payload["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_oairesp_parse_stream_reads_cached_input_tokens() -> None:
    from llmpxy.protocols.oai_responses import OpenAIResponsesAdapter

    adapter = OpenAIResponsesAdapter()

    async def lines() -> AsyncIterator[str]:
        yield (
            'data: {"type":"response.completed","response":{"id":"resp_1","model":"a-model",'
            '"usage":{"input_tokens":100,"input_tokens_details":{"cached_tokens":40},'
            '"output_tokens":25,"total_tokens":125}}}\n\n'
        )
        yield "data: [DONE]"

    response, _events = await adapter.parse_stream(lines(), "oairesp")

    assert response.usage.input_tokens == 100
    assert response.usage.cached_input_tokens == 40
    assert response.usage.output_tokens == 25


def test_oairesp_stream_previous_response_id_preserves_tool_call_history_for_oaichat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_bodies: list[dict[str, Any]] = []

    async def first_stream() -> Any:
        chunks = [
            {
                "id": "chatcmpl_1",
                "model": "a-model",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "glob",
                                        "arguments": '{"pattern":"**/*.md"}',
                                    },
                                }
                            ]
                        }
                    }
                ],
            },
            {
                "id": "chatcmpl_1",
                "model": "a-model",
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ]
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_bodies.append(body)
        if len(captured_bodies) == 1:
            assert body["messages"] == [{"role": "user", "content": "hi"}]
            return httpx.Response(200, content=first_stream())

        assert body["messages"][0] == {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "glob", "arguments": '{"pattern":"**/*.md"}'},
                }
            ],
        }
        assert body["messages"][1] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "README.md",
        }
        assert body["messages"][2] == {"role": "user", "content": "continue"}
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_2",
                "model": "a-model",
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "storage": {
                "backend": "sqlite",
                "sqlite_path": str(tmp_path / "stream-history-tool-call.db"),
                "ttl_seconds": 60,
            },
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-5.4": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "stream-history-tool-call.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    with client.stream(
        "POST", "/v1/responses", json={"model": "gpt-5.4", "input": "hi", "stream": True}
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    first_id = next(event for event in events if event["type"] == "response.completed")["response"][
        "id"
    ]

    second = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4",
            "previous_response_id": first_id,
            "input": [
                {"type": "function_call_output", "call_id": "call_1", "output": "README.md"},
                {"role": "user", "content": [{"type": "input_text", "text": "continue"}]},
            ],
        },
    )

    assert second.status_code == 200
    assert len(captured_bodies) == 2
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_accepts_single_content_object_in_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "a-model",
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "single-content-object.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": {"type": "input_text", "text": "hi"},
                }
            ],
        },
    )

    assert response.status_code == 200
    messages = cast(list[dict[str, Any]], captured_body["messages"])
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "hi"
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_accepts_null_message_content_in_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "a-model",
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "null-message-content.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": [{"type": "message", "role": "assistant", "content": None}],
        },
    )

    assert response.status_code == 200
    messages = cast(list[dict[str, Any]], captured_body["messages"])
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == ""
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_oairesp_accepts_function_call_items_in_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("A_KEY", "a")

    import httpx

    captured_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "a-model",
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = httpx.AsyncClient

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)
    config = AppConfig.model_validate(
        {
            "route": {"type": "provider", "name": "a"},
            "providers": [
                {
                    "name": "a",
                    "protocol": "oaichat",
                    "base_url": "https://a.example/v1",
                    "api_key_env": "A_KEY",
                    "models": {"gpt-4.1": "a-model"},
                }
            ],
        }
    )
    store = SQLiteConversationStore(tmp_path / "tool-call-input.db")
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "glob",
                    "arguments": '{"pattern":"**/*.md"}',
                }
            ],
        },
    )

    assert response.status_code == 200
    messages = cast(list[dict[str, Any]], captured_body["messages"])
    first_message = messages[0]
    assert first_message["role"] == "assistant"
    tool_calls = cast(list[dict[str, Any]], first_message["tool_calls"])
    assert tool_calls[0]["function"]["name"] == "glob"
    assert tool_calls[0]["function"]["arguments"] == '{"pattern":"**/*.md"}'
    monkeypatch.setattr(httpx, "AsyncClient", original)
