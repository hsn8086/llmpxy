from __future__ import annotations

from pathlib import Path

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
type = "group"
name = "default"

[storage]
backend = "sqlite"
sqlite_path = "./data/test.db"

[[providers]]
name = "openai-chat-a"
protocol = "oaichat"
base_url = "https://a.example/v1"
api_key_env = "A_KEY"

[[provider_groups]]
name = "default"
strategy = "fallback"
members = ["openai-chat-a"]
""",
        encoding="utf-8",
    )


def test_admin_api_can_add_and_update_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("A_KEY", "upstream-a")
    config_file = tmp_path / "config.toml"
    _write_config(config_file)

    runtime = RuntimeManager(config_file)
    client = TestClient(create_runtime_managed_app(runtime))
    headers = {"Authorization": "Bearer admin-token"}

    create_response = client.post(
        "/admin/api-keys",
        headers=headers,
        json={
            "name": "client-a",
            "limit_usd": 10.0,
            "provider_limits_usd": {"openai-chat-a": 6.0},
            "group_limits_usd": {"default": 8.0},
        },
    )
    assert create_response.status_code == 200
    created = create_response.json()["created"]

    update_response = client.patch(
        f"/admin/api-keys/{created['uuid']}",
        headers=headers,
        json={"enabled": False, "limit_usd": 7.5},
    )
    assert update_response.status_code == 200
    masked = update_response.json()
    assert masked["api_keys"][0]["limit_usd"] == 7.5


def test_admin_dashboard_stream_works(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("A_KEY", "upstream-a")
    config_file = tmp_path / "config.toml"
    _write_config(config_file)

    runtime = RuntimeManager(config_file)
    client = TestClient(create_runtime_managed_app(runtime))
    headers = {"Authorization": "Bearer admin-token"}

    snapshot = client.get("/admin/dashboard/snapshot", headers=headers)
    assert snapshot.status_code == 200
    assert "recent_requests" in snapshot.json()


def test_admin_list_endpoints_return_providers_and_groups(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("A_KEY", "upstream-a")
    config_file = tmp_path / "config.toml"
    _write_config(config_file)

    runtime = RuntimeManager(config_file)
    client = TestClient(create_runtime_managed_app(runtime))
    headers = {"Authorization": "Bearer admin-token"}

    providers_response = client.get("/admin/providers", headers=headers)
    assert providers_response.status_code == 200
    providers = providers_response.json()
    assert providers[0]["name"] == "openai-chat-a"
    assert providers[0]["protocol"] == "oaichat"

    groups_response = client.get("/admin/provider-groups", headers=headers)
    assert groups_response.status_code == 200
    groups = groups_response.json()
    assert groups[0]["name"] == "default"
    assert groups[0]["members"] == ["openai-chat-a"]
