from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient

from llmpxy.app import create_app
from llmpxy.config import load_config
import llmpxy.dispatcher as dispatcher_module
from llmpxy.dispatcher import ProviderDispatcher
from llmpxy.storage_sqlite import SQLiteConversationStore


def test_config_toml_route_to_local_test_responses_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLMPXY_ENABLE_TEST_ENDPOINTS", "1")
    monkeypatch.setenv("TEST_API_KEY", "test-key")

    config_file = tmp_path / "config.toml"
    sqlite_file = tmp_path / "integration.db"

    config_file.write_text(
        f"""
[server]
host = "127.0.0.1"
port = 18086

[route]
type = "provider"
name = "local-responses"

[network]
trust_env = false

[storage]
backend = "sqlite"
sqlite_path = "./{sqlite_file.name}"

[[providers]]
name = "local-responses"
protocol = "oairesp"
base_url = "http://testserver/_test/upstream"
api_key_env = "TEST_API_KEY"

[providers.models]
"gpt-5-nano" = "gpt-5.4-mini"
""",
        encoding="utf-8",
    )

    config = load_config(config_file)
    store = SQLiteConversationStore(config.resolve_sqlite_path(config_file))
    dispatcher = ProviderDispatcher(config)
    app = create_app(config, store, dispatcher)

    original_create_async_client = dispatcher_module.create_async_client

    def create_test_async_client(config: Any, provider: Any = None) -> httpx.AsyncClient:
        timeout = httpx.Timeout(
            connect=config.network.connect_timeout_seconds,
            read=config.network.read_timeout_seconds,
            write=config.network.write_timeout_seconds,
            pool=config.network.pool_timeout_seconds,
        )
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            timeout=timeout,
            trust_env=config.network.trust_env,
        )

    monkeypatch.setattr(dispatcher_module, "create_async_client", create_test_async_client)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5-nano",
            "messages": [
                {"role": "system", "content": "Follow policy"},
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
                {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(["README.md"])},
                {"role": "user", "content": "继续"},
            ],
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
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "ok"

    received = cast(dict[str, Any], app.state.test_upstream_last_payload)
    assert received["model"] == "gpt-5.4-mini"
    assert received["instructions"] == "Follow policy"
    assert received["text"]["format"]["name"] == "person"

    last_response = client.get("/_test/upstream/last")
    assert last_response.status_code == 200
    assert cast(dict[str, Any], last_response.json()["payload"])["model"] == "gpt-5.4-mini"

    input_items = cast(list[dict[str, Any]], received["input"])
    assert input_items[0]["role"] == "system"
    assert input_items[1]["type"] == "function_call"
    assert input_items[2]["role"] == "assistant"
    assert input_items[3]["type"] == "function_call_output"
    assert input_items[4]["role"] == "user"

    monkeypatch.setattr(dispatcher_module, "create_async_client", original_create_async_client)
