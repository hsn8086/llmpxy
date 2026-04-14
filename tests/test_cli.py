from __future__ import annotations

from pathlib import Path
import json

import httpx
import pytest
from typer.testing import CliRunner

from llmpxy.cli import cli
from llmpxy.config import load_config
from llmpxy.models import ApiKeyUsageRecord
from llmpxy.runtime import build_store


runner = CliRunner()


def _base_config() -> str:
    return """
[route]
type = "group"
name = "default"

[[providers]]
name = "openai-chat-a"
protocol = "oaichat"
base_url = "https://a.example/v1"
api_key_env = "A_KEY"

[[provider_groups]]
name = "default"
strategy = "fallback"
model_whitelist_only = true
models = ["gpt-4.1"]
members = ["openai-chat-a"]
"""


def test_config_validate_command(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(_base_config(), encoding="utf-8")

    result = runner.invoke(cli, ["config", "validate", "--config", str(config_file)])

    assert result.exit_code == 0
    assert "Config valid" in result.stdout


def test_add_api_key_command_updates_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(_base_config(), encoding="utf-8")

    result = runner.invoke(
        cli,
        [
            "config",
            "add-api-key",
            "--config",
            str(config_file),
            "--name",
            "client-a",
            "--limit-usd",
            "10",
            "--provider-limit",
            "openai-chat-a=5",
            "--group-limit",
            "default=8",
        ],
    )

    assert result.exit_code == 0
    assert "Added API key:" in result.stdout

    config = load_config(config_file)
    assert len(config.api_keys) == 1
    assert config.api_keys[0].name == "client-a"
    assert config.api_keys[0].limit_usd == 10.0
    assert config.api_keys[0].provider_limits_usd == {"openai-chat-a": 5.0}
    assert config.api_keys[0].group_limits_usd == {"default": 8.0}
    assert config.api_keys[0].uuid
    assert config.api_keys[0].key


def test_list_api_keys_command(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        _base_config()
        + """

[[api_keys]]
uuid = "11111111-1111-1111-1111-111111111111"
name = "client-a"
key = "secret-key"
limit_usd = 12.5
""",
        encoding="utf-8",
    )

    result = runner.invoke(cli, ["config", "list-api-keys", "--config", str(config_file)])

    assert result.exit_code == 0
    assert "11111111-1111-1111-1111-111111111111" in result.stdout
    assert "client-a" in result.stdout


def test_show_balance_command(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        _base_config()
        + """

[storage]
backend = "sqlite"
sqlite_path = "./data/test.db"

[[api_keys]]
uuid = "11111111-1111-1111-1111-111111111111"
name = "client-a"
key = "secret-key"
limit_usd = 12.5
provider_limits_usd = { openai-chat-a = 5.0 }
group_limits_usd = { default = 8.0 }
""",
        encoding="utf-8",
    )

    loaded = load_config(config_file)
    store = build_store(loaded, config_file.resolve())
    store.put_api_key_usage(
        ApiKeyUsageRecord(
            api_key_uuid="11111111-1111-1111-1111-111111111111",
            api_key_name="client-a",
            request_id="req-1",
            provider_name="openai-chat-a",
            requested_model="gpt-4.1",
            upstream_model="gpt-4.1",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=3.0,
            created_at=1,
        )
    )

    result = runner.invoke(
        cli,
        [
            "config",
            "show-balance",
            "--config",
            str(config_file),
            "--uuid",
            "11111111-1111-1111-1111-111111111111",
        ],
    )

    assert result.exit_code == 0
    assert "API Key: client-a (11111111-1111-1111-1111-111111111111)" in result.stdout
    assert "Total: used=$3.000000, limit=$12.500000, remaining=$9.500000" in result.stdout
    assert (
        "Provider openai-chat-a: used=$3.000000, limit=$5.000000, remaining=$2.000000"
        in result.stdout
    )
    assert "Group default: used=$3.000000, limit=$8.000000, remaining=$5.000000" in result.stdout


def test_remote_status_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer admin-token"
        return httpx.Response(200, json={"route": {"type": "group", "name": "default"}})

    class PatchedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", PatchedClient)
    result = runner.invoke(
        cli,
        [
            "remote",
            "status",
            "--base-url",
            "http://127.0.0.1:8080",
            "--admin-token",
            "admin-token",
        ],
    )

    assert result.exit_code == 0
    assert '"route"' in result.stdout


def test_remote_config_get_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/admin/config"
        return httpx.Response(200, json={"admin": {"enabled": True}, "api_keys": []})

    class PatchedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", PatchedClient)
    result = runner.invoke(
        cli,
        [
            "remote",
            "config-get",
            "--base-url",
            "http://127.0.0.1:8080",
            "--admin-token",
            "admin-token",
        ],
    )

    assert result.exit_code == 0
    assert '"admin"' in result.stdout


def test_remote_api_key_disable_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.url.path == "/admin/api-keys/111"
        assert json.loads(request.content.decode("utf-8")) == {"enabled": False}
        return httpx.Response(200, json={"ok": True})

    class PatchedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", PatchedClient)
    result = runner.invoke(
        cli,
        [
            "remote",
            "api-key-disable",
            "--base-url",
            "http://127.0.0.1:8080",
            "--admin-token",
            "admin-token",
            "--uuid",
            "111",
        ],
    )

    assert result.exit_code == 0
    assert '"ok"' in result.stdout


def test_remote_api_key_enable_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.url.path == "/admin/api-keys/111"
        assert json.loads(request.content.decode("utf-8")) == {"enabled": True}
        return httpx.Response(200, json={"ok": True})

    class PatchedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", PatchedClient)
    result = runner.invoke(
        cli,
        [
            "remote",
            "api-key-enable",
            "--base-url",
            "http://127.0.0.1:8080",
            "--admin-token",
            "admin-token",
            "--uuid",
            "111",
        ],
    )

    assert result.exit_code == 0
    assert '"ok"' in result.stdout


def test_remote_api_key_rotate_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.url.path == "/admin/api-keys/111"
        assert json.loads(request.content.decode("utf-8")) == {"key": "rotated-key"}
        return httpx.Response(200, json={"api_keys": []})

    class PatchedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", PatchedClient)
    result = runner.invoke(
        cli,
        [
            "remote",
            "api-key-rotate",
            "--base-url",
            "http://127.0.0.1:8080",
            "--admin-token",
            "admin-token",
            "--uuid",
            "111",
            "--key",
            "rotated-key",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["uuid"] == "111"
    assert payload["key"] == "rotated-key"


def test_remote_provider_list_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/admin/providers"
        return httpx.Response(200, json=[{"name": "openai-chat-a", "state": "healthy"}])

    class PatchedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", PatchedClient)
    result = runner.invoke(
        cli,
        [
            "remote",
            "provider-list",
            "--base-url",
            "http://127.0.0.1:8080",
            "--admin-token",
            "admin-token",
        ],
    )

    assert result.exit_code == 0
    assert '"openai-chat-a"' in result.stdout


def test_remote_group_list_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/admin/provider-groups"
        return httpx.Response(200, json=[{"name": "default", "strategy": "fallback"}])

    class PatchedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", PatchedClient)
    result = runner.invoke(
        cli,
        [
            "remote",
            "group-list",
            "--base-url",
            "http://127.0.0.1:8080",
            "--admin-token",
            "admin-token",
        ],
    )

    assert result.exit_code == 0
    assert '"default"' in result.stdout
