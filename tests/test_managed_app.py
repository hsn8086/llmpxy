from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from llmpxy.app import create_runtime_managed_app
from llmpxy.runtime import RuntimeManager


def _write_config(path: Path) -> None:
    path.write_text(
        """
[admin]
enabled = true
token = "admin-token"

[route]
type = "provider"
name = "provider-a"

[[api_keys]]
uuid = "11111111-1111-1111-1111-111111111111"
name = "client-a"
key = "client-a"
limit_usd = 0.000001

[[providers]]
name = "provider-a"
protocol = "oaichat"
base_url = "https://a.example/v1"
api_key_env = "UPSTREAM_A_KEY"

[providers.pricing.default]
input_per_million_tokens_usd = 1.0
output_per_million_tokens_usd = 1.0
""",
        encoding="utf-8",
    )


def test_managed_app_requires_api_key_and_enforces_budget(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_A_KEY", "upstream-a")

    config_file = tmp_path / "config.toml"
    _write_config(config_file)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "provider-model",
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
    runtime = RuntimeManager(config_file)
    client = TestClient(create_runtime_managed_app(runtime))

    unauthorized = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert unauthorized.status_code == 401

    first = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer client-a"},
        json={"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer client-a"},
        json={"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert second.status_code == 429

    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_managed_app_accepts_anthropic_x_api_key_header(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_A_KEY", "upstream-a")

    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
[admin]
enabled = true
token = "admin-token"

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
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key_env = "UPSTREAM_A_KEY"

[providers.pricing.default]
input_per_million_tokens_usd = 1.0
output_per_million_tokens_usd = 1.0
""",
        encoding="utf-8",
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "provider-model",
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
    runtime = RuntimeManager(config_file)
    client = TestClient(create_runtime_managed_app(runtime))

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "client-a", "anthropic-version": "2023-06-01"},
        json={"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    monkeypatch.setattr(httpx, "AsyncClient", original)


def test_managed_app_admin_endpoints_require_admin_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("UPSTREAM_A_KEY", "upstream-a")
    config_file = tmp_path / "config.toml"
    _write_config(config_file)

    runtime = RuntimeManager(config_file)
    client = TestClient(create_runtime_managed_app(runtime))

    unauthorized = client.get("/admin/status")
    assert unauthorized.status_code == 401

    authorized = client.get("/admin/status", headers={"Authorization": "Bearer admin-token"})
    assert authorized.status_code == 200
    assert "providers" in authorized.json()


def test_managed_app_reuses_request_random_order_for_load_balance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPSTREAM_A_KEY", "upstream-a")
    monkeypatch.setenv("UPSTREAM_B_KEY", "upstream-b")

    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
[admin]
enabled = true
token = "admin-token"

[route]
type = "group"
name = "balanced"

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

[providers.pricing.default]
input_per_million_tokens_usd = 1.0
output_per_million_tokens_usd = 1.0

[[providers]]
name = "provider-b"
protocol = "oaichat"
base_url = "https://b.example/v1"
api_key_env = "UPSTREAM_B_KEY"

[providers.pricing.default]
input_per_million_tokens_usd = 1.0
output_per_million_tokens_usd = 1.0

[[provider_groups]]
name = "balanced"
strategy = "load_balance"
members = ["provider-a", "provider-b"]
""",
        encoding="utf-8",
    )

    attempts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.host or "")
        model = "provider-b-model" if request.url.host == "b.example" else "provider-a-model"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": model,
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
    monkeypatch.setattr("llmpxy.dispatcher.random.uniform", lambda _start, _end: 1.5)

    runtime = RuntimeManager(config_file)
    client = TestClient(create_runtime_managed_app(runtime))

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer client-a"},
        json={"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["model"] == "provider-b-model"
    assert attempts == ["b.example"]

    monkeypatch.setattr(httpx, "AsyncClient", original)
