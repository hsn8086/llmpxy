from __future__ import annotations

from typing import Any

import pytest
from textual.widgets import DataTable, Static

from llmpxy.remote_dashboard import DashboardClient, RemoteDashboardApp


class FakeDashboardClient(DashboardClient):
    def __init__(self, snapshot: dict[str, Any]) -> None:
        super().__init__(base_url="http://127.0.0.1:8080", admin_token="admin-token")
        self._snapshot = snapshot

    async def snapshot(self) -> dict[str, Any]:
        return self._snapshot

    async def stream(self):
        if False:
            yield "message", {}


@pytest.mark.asyncio
async def test_remote_dashboard_renders_snapshot() -> None:
    snapshot = {
        "route": {"type": "group", "name": "default"},
        "storage_backend": "sqlite",
        "window": {
            "seconds": 60,
            "request_count": 2,
            "success_count": 1,
            "error_count": 1,
            "error_rate": 0.5,
            "tps": 0.03,
            "total_cost_usd": 0.5,
            "avg_latency_ms": 1210,
        },
        "reload": {
            "current_revision": "abc123",
            "last_success_at": 1,
            "last_error": None,
        },
        "providers": [
            {
                "name": "openai-chat-b",
                "protocol": "oaichat",
                "state": "healthy",
                "consecutive_errors": 0,
                "recent_successes": 3,
                "recent_errors": 0,
                "in_flight": 0,
                "last_error_message": None,
            },
            {
                "name": "openai-chat-a",
                "protocol": "oaichat",
                "state": "failing",
                "consecutive_errors": 4,
                "recent_successes": 1,
                "recent_errors": 2,
                "in_flight": 1,
                "last_error_message": "upstream timeout after retry loop",
            },
        ],
        "api_keys": [
            {
                "uuid": "11111111-1111-1111-1111-111111111111",
                "name": "client-a",
                "enabled": True,
                "used_usd": 4.0,
                "remaining_usd": 1.0,
                "limit_usd": 5.0,
                "utilization_ratio": 0.8,
            },
            {
                "uuid": "22222222-2222-2222-2222-222222222222",
                "name": "client-b",
                "enabled": False,
                "used_usd": 0.0,
                "remaining_usd": None,
                "limit_usd": None,
                "utilization_ratio": None,
            },
        ],
        "recent_requests": [
            {
                "request_id": "req-1",
                "api_key_name": "client-a",
                "provider_name": "openai-chat-a",
                "requested_model": "gpt-4.1",
                "status": "provider_error",
                "http_status": 502,
                "latency_ms": 2100,
                "cost_usd": 0.0,
                "error_message": "upstream timeout after retry loop",
            },
            {
                "request_id": "req-2",
                "api_key_name": "client-a",
                "provider_name": "openai-chat-b",
                "requested_model": "gpt-4.1",
                "status": "success",
                "http_status": 200,
                "latency_ms": 320,
                "cost_usd": 0.5,
                "error_message": None,
            },
        ],
    }

    app = RemoteDashboardApp(FakeDashboardClient(snapshot), enable_stream=False)
    async with app.run_test() as pilot:
        await pilot.pause()

        summary = app.query_one("#summary", Static)
        rendered_summary = str(summary.render())
        assert "Window(60s): req=2 tps=0.03 err_rate=50.0% avg_latency=1210ms" in rendered_summary
        assert "Providers: total=2 healthy=1 failing=1" in rendered_summary
        assert "Recent Requests: total=2 success=1 errors=1 cost=$0.5000" in rendered_summary

        providers = app.query_one("#providers", DataTable)
        assert providers.row_count == 2
        assert providers.get_row_at(0)[0] == "openai-chat-a"
        assert providers.get_row_at(0)[7] == "upstream timeout after retry loop"

        api_keys = app.query_one("#api_keys", DataTable)
        assert api_keys.row_count == 2
        assert api_keys.get_row_at(0)[0] == "client-a"
        assert api_keys.get_row_at(0)[1] == "enabled"
        assert api_keys.get_row_at(0)[6] == "80%"
        assert api_keys.get_row_at(1)[1] == "disabled"

        recent = app.query_one("#recent", DataTable)
        assert recent.row_count == 2
        assert recent.get_row_at(0)[1] == "provider_error"
        assert recent.get_row_at(0)[6] == "2100ms"
        assert recent.get_row_at(0)[8] == "upstream timeout after retry loop"

        details = app.query_one("#details", Static)
        rendered_details = str(details.render())
        assert "Request: req-1" in rendered_details
        assert "Error: upstream timeout after retry loop" in rendered_details


@pytest.mark.asyncio
async def test_remote_dashboard_filter_and_sort_actions() -> None:
    snapshot = {
        "route": {"type": "group", "name": "default"},
        "storage_backend": "sqlite",
        "window": {
            "seconds": 60,
            "request_count": 3,
            "success_count": 2,
            "error_count": 1,
            "error_rate": 1 / 3,
            "tps": 0.05,
            "total_cost_usd": 1.5,
            "avg_latency_ms": 500,
        },
        "reload": {"current_revision": "abc123", "last_success_at": 1, "last_error": None},
        "providers": [
            {
                "name": "provider-z",
                "protocol": "oaichat",
                "state": "healthy",
                "consecutive_errors": 0,
                "recent_successes": 5,
                "recent_errors": 1,
                "in_flight": 0,
                "last_error_message": None,
            },
            {
                "name": "provider-a",
                "protocol": "oaichat",
                "state": "degraded",
                "consecutive_errors": 2,
                "recent_successes": 1,
                "recent_errors": 4,
                "in_flight": 0,
                "last_error_message": "rate limit",
            },
        ],
        "api_keys": [
            {
                "uuid": "22222222-2222-2222-2222-222222222222",
                "name": "z-client",
                "enabled": True,
                "used_usd": 1.0,
                "remaining_usd": 9.0,
                "limit_usd": 10.0,
                "utilization_ratio": 0.1,
            },
            {
                "uuid": "11111111-1111-1111-1111-111111111111",
                "name": "a-client",
                "enabled": True,
                "used_usd": 8.0,
                "remaining_usd": 2.0,
                "limit_usd": 10.0,
                "utilization_ratio": 0.8,
            },
        ],
        "recent_requests": [
            {
                "request_id": "req-err",
                "api_key_name": "a-client",
                "provider_name": "provider-a",
                "requested_model": "gpt-4.1",
                "status": "provider_error",
                "http_status": 429,
                "latency_ms": 900,
                "cost_usd": 0.0,
                "error_message": "rate limit",
            },
            {
                "request_id": "req-ok",
                "api_key_name": "z-client",
                "provider_name": "provider-z",
                "requested_model": "gpt-4.1-mini",
                "status": "success",
                "http_status": 200,
                "latency_ms": 100,
                "cost_usd": 1.0,
                "error_message": None,
            },
        ],
    }

    app = RemoteDashboardApp(FakeDashboardClient(snapshot), enable_stream=False)
    async with app.run_test() as pilot:
        await pilot.pause()

        app.action_sort_providers()
        providers = app.query_one("#providers", DataTable)
        assert providers.get_row_at(0)[0] == "provider-a"

        app.action_sort_keys()
        api_keys = app.query_one("#api_keys", DataTable)
        assert api_keys.get_row_at(0)[0] == "a-client"

        app.action_focus_errors()
        recent = app.query_one("#recent", DataTable)
        assert recent.row_count == 1
        assert recent.get_row_at(0)[0] == "req-err"

        app.action_clear_filter()
        recent = app.query_one("#recent", DataTable)
        assert recent.row_count == 2
