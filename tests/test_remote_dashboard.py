from __future__ import annotations

import asyncio
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


class CountingDashboardClient(FakeDashboardClient):
    def __init__(self, snapshot: dict[str, Any]) -> None:
        super().__init__(snapshot)
        self.snapshot_calls = 0

    async def snapshot(self) -> dict[str, Any]:
        self.snapshot_calls += 1
        return await super().snapshot()


class SlowCountingDashboardClient(CountingDashboardClient):
    def __init__(self, snapshot: dict[str, Any], delay_seconds: float) -> None:
        super().__init__(snapshot)
        self.delay_seconds = delay_seconds

    async def snapshot(self) -> dict[str, Any]:
        self.snapshot_calls += 1
        await asyncio.sleep(self.delay_seconds)
        return self._snapshot


class MutableDashboardClient(CountingDashboardClient):
    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._snapshot = snapshot


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
            "success_tps": 0.02,
            "error_tps": 0.01,
            "total_cost_usd": 0.5,
            "avg_latency_ms": 1210,
            "p95_latency_ms": 2100,
            "total_tokens": 3000,
        },
        "windows": {
            "10s": {
                "request_count": 1,
                "success_count": 0,
                "error_count": 1,
                "error_rate": 1.0,
                "tps": 0.10,
                "success_tps": 0.0,
                "error_tps": 0.10,
                "total_cost_usd": 0.0,
                "avg_latency_ms": 2100,
                "p95_latency_ms": 2100,
                "total_tokens": 1000,
            },
            "60s": {
                "request_count": 2,
                "success_count": 1,
                "error_count": 1,
                "error_rate": 0.5,
                "tps": 0.03,
                "success_tps": 0.02,
                "error_tps": 0.01,
                "total_cost_usd": 0.5,
                "avg_latency_ms": 1210,
                "p95_latency_ms": 2100,
                "total_tokens": 3000,
            },
            "5m": {
                "request_count": 6,
                "success_count": 5,
                "error_count": 1,
                "error_rate": 1 / 6,
                "tps": 0.02,
                "success_tps": 0.02,
                "error_tps": 0.00,
                "total_cost_usd": 2.5,
                "avg_latency_ms": 900,
                "p95_latency_ms": 2100,
                "total_tokens": 9000,
            },
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
                "window": {
                    "seconds": 60,
                    "request_count": 1,
                    "success_count": 1,
                    "error_count": 0,
                    "error_rate": 0.0,
                    "tps": 0.02,
                    "avg_latency_ms": 320,
                    "total_cost_usd": 0.5,
                },
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
                "window": {
                    "seconds": 60,
                    "request_count": 1,
                    "success_count": 0,
                    "error_count": 1,
                    "error_rate": 1.0,
                    "tps": 0.02,
                    "avg_latency_ms": 2100,
                    "total_cost_usd": 0.0,
                },
            },
        ],
        "alerts": [
            {
                "kind": "provider_health",
                "name": "openai-chat-a",
                "severity": "failing",
                "message": "upstream timeout after retry loop",
            },
            {
                "kind": "api_key_budget",
                "name": "client-a",
                "severity": "warning",
                "message": "Budget usage 80%",
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

        assert list(app.query("#filter")) == []

        summary = app.query_one("#summary", Static)
        rendered_summary = str(summary.render())
        assert "Filter: -  ProviderSort: health  KeySort: usage" in rendered_summary
        assert "Window(60s): req=2 tps=0.03 err_rate=50.0% avg_latency=1210ms" in rendered_summary
        assert "Window(10s): tps=0.10 success_tps=0.00 error_tps=0.10" in rendered_summary
        assert "Window(5m): req=6 p95=2100ms tokens=9000 cost=$2.50" in rendered_summary
        assert "Providers: total=2 healthy=1 failing=1" in rendered_summary
        assert "Recent Requests: total=2 success=1 errors=1 cost=$0.5000" in rendered_summary

        providers = app.query_one("#providers", DataTable)
        assert providers.row_count == 2
        assert providers.get_row_at(0)[0] == "openai-chat-a"
        assert providers.get_row_at(0)[7] == "upstream timeout after retry loop"

        provider_details = app.query_one("#provider_details", Static)
        rendered_provider_details = str(provider_details.render())
        assert "Provider: openai-chat-a" in rendered_provider_details
        assert "State: failing" in rendered_provider_details
        assert "Window(60s): req=1 tps=0.02 err_rate=100.0%" in rendered_provider_details
        assert "Window(60s): avg_latency=2100ms cost=$0.0000" in rendered_provider_details
        assert provider_details.has_class("state-bad")

        alerts = app.query_one("#alerts", Static)
        rendered_alerts = str(alerts.render())
        assert "FAILING openai-chat-a - upstream timeout after retry loop" in rendered_alerts
        assert "WARNING client-a - Budget usage 80%" in rendered_alerts
        assert alerts.has_class("state-bad")

        api_keys = app.query_one("#api_keys", DataTable)
        assert api_keys.row_count == 2
        assert api_keys.get_row_at(0)[0] == "client-a"
        assert api_keys.get_row_at(0)[1] == "enabled"
        assert api_keys.get_row_at(0)[6] == "80%"
        assert api_keys.get_row_at(1)[1] == "disabled"

        api_key_details = app.query_one("#api_key_details", Static)
        rendered_api_key_details = str(api_key_details.render())
        assert "API Key: client-a" in rendered_api_key_details
        assert "UUID: 11111111-1111-1111-1111-111111111111" in rendered_api_key_details

        recent = app.query_one("#recent", DataTable)
        assert recent.row_count == 2
        assert recent.get_row_at(0)[2] == "provider_error"
        assert recent.get_row_at(0)[7] == "2100ms"
        assert recent.get_row_at(0)[9] == "upstream timeout after retry loop"

        details = app.query_one("#details", Static)
        rendered_details = str(details.render())
        assert "Request: req-1" in rendered_details
        assert "Error: upstream timeout after retry loop" in rendered_details
        assert details.has_class("state-bad")


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
            "success_tps": 0.03,
            "error_tps": 0.02,
            "total_cost_usd": 1.5,
            "avg_latency_ms": 500,
            "p95_latency_ms": 900,
            "total_tokens": 6000,
        },
        "windows": {
            "10s": {
                "request_count": 1,
                "success_count": 0,
                "error_count": 1,
                "error_rate": 1.0,
                "tps": 0.10,
                "success_tps": 0.0,
                "error_tps": 0.10,
                "total_cost_usd": 0.0,
                "avg_latency_ms": 900,
                "p95_latency_ms": 900,
                "total_tokens": 1000,
            },
            "60s": {
                "request_count": 3,
                "success_count": 2,
                "error_count": 1,
                "error_rate": 1 / 3,
                "tps": 0.05,
                "success_tps": 0.03,
                "error_tps": 0.02,
                "total_cost_usd": 1.5,
                "avg_latency_ms": 500,
                "p95_latency_ms": 900,
                "total_tokens": 6000,
            },
            "5m": {
                "request_count": 9,
                "success_count": 8,
                "error_count": 1,
                "error_rate": 1 / 9,
                "tps": 0.03,
                "success_tps": 0.03,
                "error_tps": 0.00,
                "total_cost_usd": 4.0,
                "avg_latency_ms": 450,
                "p95_latency_ms": 900,
                "total_tokens": 18000,
            },
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
                "window": {
                    "seconds": 60,
                    "request_count": 1,
                    "success_count": 1,
                    "error_count": 0,
                    "error_rate": 0.0,
                    "tps": 0.02,
                    "avg_latency_ms": 100,
                    "total_cost_usd": 1.0,
                },
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
                "window": {
                    "seconds": 60,
                    "request_count": 2,
                    "success_count": 1,
                    "error_count": 1,
                    "error_rate": 0.5,
                    "tps": 0.03,
                    "avg_latency_ms": 500,
                    "total_cost_usd": 0.5,
                },
            },
        ],
        "alerts": [
            {
                "kind": "provider_health",
                "name": "provider-a",
                "severity": "degraded",
                "message": "rate limit",
            },
            {
                "kind": "api_key_budget",
                "name": "a-client",
                "severity": "warning",
                "message": "Budget usage 80%",
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


@pytest.mark.asyncio
async def test_remote_dashboard_updates_provider_and_api_key_details_when_rows_selected() -> None:
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
            "success_tps": 0.02,
            "error_tps": 0.01,
            "total_cost_usd": 0.5,
            "avg_latency_ms": 400,
            "p95_latency_ms": 700,
            "total_tokens": 2000,
        },
        "windows": {
            "10s": {
                "request_count": 1,
                "success_count": 1,
                "error_count": 0,
                "error_rate": 0.0,
                "tps": 0.10,
                "success_tps": 0.10,
                "error_tps": 0.0,
                "total_cost_usd": 0.5,
                "avg_latency_ms": 100,
                "p95_latency_ms": 100,
                "total_tokens": 1000,
            },
            "60s": {
                "request_count": 2,
                "success_count": 1,
                "error_count": 1,
                "error_rate": 0.5,
                "tps": 0.03,
                "success_tps": 0.02,
                "error_tps": 0.01,
                "total_cost_usd": 0.5,
                "avg_latency_ms": 400,
                "p95_latency_ms": 700,
                "total_tokens": 2000,
            },
            "5m": {
                "request_count": 4,
                "success_count": 3,
                "error_count": 1,
                "error_rate": 0.25,
                "tps": 0.01,
                "success_tps": 0.01,
                "error_tps": 0.00,
                "total_cost_usd": 1.0,
                "avg_latency_ms": 350,
                "p95_latency_ms": 700,
                "total_tokens": 4000,
            },
        },
        "reload": {"current_revision": "abc123", "last_success_at": 1, "last_error": None},
        "providers": [
            {
                "name": "provider-a",
                "protocol": "oaichat",
                "state": "failing",
                "consecutive_errors": 3,
                "recent_successes": 0,
                "recent_errors": 4,
                "in_flight": 1,
                "last_success_at": None,
                "last_error_message": "timeout",
                "window": {
                    "seconds": 60,
                    "request_count": 1,
                    "success_count": 0,
                    "error_count": 1,
                    "error_rate": 1.0,
                    "tps": 0.02,
                    "avg_latency_ms": 700,
                    "total_cost_usd": 0.0,
                },
            },
            {
                "name": "provider-b",
                "protocol": "anthropic",
                "state": "healthy",
                "consecutive_errors": 0,
                "recent_successes": 5,
                "recent_errors": 0,
                "in_flight": 0,
                "last_success_at": 1700000000,
                "last_error_message": None,
                "window": {
                    "seconds": 60,
                    "request_count": 1,
                    "success_count": 1,
                    "error_count": 0,
                    "error_rate": 0.0,
                    "tps": 0.02,
                    "avg_latency_ms": 100,
                    "total_cost_usd": 0.5,
                },
            },
        ],
        "alerts": [],
        "api_keys": [
            {
                "uuid": "11111111-1111-1111-1111-111111111111",
                "name": "client-a",
                "enabled": True,
                "used_usd": 8.0,
                "remaining_usd": 2.0,
                "limit_usd": 10.0,
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
        "recent_requests": [],
    }

    app = RemoteDashboardApp(FakeDashboardClient(snapshot), enable_stream=False)
    async with app.run_test() as pilot:
        await pilot.pause()

        providers = app.query_one("#providers", DataTable)
        providers.move_cursor(row=1)
        await pilot.pause()

        provider_details = app.query_one("#provider_details", Static)
        rendered_provider_details = str(provider_details.render())
        if "Provider: provider-b" not in rendered_provider_details:
            app._selected_provider_name = "provider-b"
            app._render_snapshot()
            rendered_provider_details = str(provider_details.render())

        assert "Provider: provider-b" in rendered_provider_details
        assert "Protocol: anthropic  State: healthy" in rendered_provider_details
        assert provider_details.has_class("state-good")

        api_keys = app.query_one("#api_keys", DataTable)
        api_keys.move_cursor(row=1)
        await pilot.pause()

        api_key_details = app.query_one("#api_key_details", Static)
        rendered_api_key_details = str(api_key_details.render())
        if "API Key: client-b" not in rendered_api_key_details:
            app._selected_api_key_uuid = "22222222-2222-2222-2222-222222222222"
            app._render_snapshot()
            rendered_api_key_details = str(api_key_details.render())

        assert "API Key: client-b" in rendered_api_key_details
        assert "State: disabled" in rendered_api_key_details
        assert api_key_details.has_class("state-warn")


@pytest.mark.asyncio
async def test_remote_dashboard_updates_details_when_recent_row_selected() -> None:
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
            "success_tps": 0.02,
            "error_tps": 0.01,
            "total_cost_usd": 0.5,
            "avg_latency_ms": 400,
            "p95_latency_ms": 700,
            "total_tokens": 2000,
        },
        "windows": {
            "10s": {
                "request_count": 1,
                "success_count": 1,
                "error_count": 0,
                "error_rate": 0.0,
                "tps": 0.10,
                "success_tps": 0.10,
                "error_tps": 0.0,
                "total_cost_usd": 0.5,
                "avg_latency_ms": 100,
                "p95_latency_ms": 100,
                "total_tokens": 1000,
            },
            "60s": {
                "request_count": 2,
                "success_count": 1,
                "error_count": 1,
                "error_rate": 0.5,
                "tps": 0.03,
                "success_tps": 0.02,
                "error_tps": 0.01,
                "total_cost_usd": 0.5,
                "avg_latency_ms": 400,
                "p95_latency_ms": 700,
                "total_tokens": 2000,
            },
            "5m": {
                "request_count": 4,
                "success_count": 3,
                "error_count": 1,
                "error_rate": 0.25,
                "tps": 0.01,
                "success_tps": 0.01,
                "error_tps": 0.00,
                "total_cost_usd": 1.0,
                "avg_latency_ms": 350,
                "p95_latency_ms": 700,
                "total_tokens": 4000,
            },
        },
        "reload": {"current_revision": "abc123", "last_success_at": 1, "last_error": None},
        "providers": [],
        "alerts": [],
        "api_keys": [],
        "recent_requests": [
            {
                "request_id": "req-1",
                "api_key_name": "client-a",
                "provider_name": "provider-a",
                "requested_model": "gpt-4.1",
                "status": "provider_error",
                "http_status": 502,
                "latency_ms": 700,
                "cost_usd": 0.0,
                "error_message": "timeout",
            },
            {
                "request_id": "req-2",
                "api_key_name": "client-b",
                "provider_name": "provider-b",
                "requested_model": "gpt-4.1-mini",
                "status": "success",
                "http_status": 200,
                "latency_ms": 100,
                "cost_usd": 0.5,
                "error_message": None,
            },
        ],
    }

    app = RemoteDashboardApp(FakeDashboardClient(snapshot), enable_stream=False)
    async with app.run_test() as pilot:
        await pilot.pause()

        recent = app.query_one("#recent", DataTable)
        recent.move_cursor(row=1)
        await pilot.pause()

        details = app.query_one("#details", Static)
        rendered_details = str(details.render())
        if "Request: req-2" not in rendered_details:
            app._selected_recent_request_id = "req-2"
            app._render_snapshot()
            rendered_details = str(details.render())

        assert "Request: req-2" in rendered_details
        assert "Provider: provider-b" in rendered_details
        assert "Cost: $0.5000" in rendered_details
        assert details.has_class("state-good")


@pytest.mark.asyncio
async def test_remote_dashboard_throttles_snapshot_refresh_requests() -> None:
    snapshot = {
        "route": {"type": "group", "name": "default"},
        "storage_backend": "sqlite",
        "window": {
            "seconds": 60,
            "request_count": 0,
            "success_count": 0,
            "error_count": 0,
            "error_rate": 0.0,
            "tps": 0.0,
            "total_cost_usd": 0.0,
            "avg_latency_ms": None,
        },
        "reload": {"current_revision": "abc123", "last_success_at": 1, "last_error": None},
        "providers": [],
        "alerts": [],
        "api_keys": [],
        "recent_requests": [],
    }

    client = CountingDashboardClient(snapshot)
    app = RemoteDashboardApp(client, enable_stream=False)
    app._refresh_interval_seconds = 1.0

    async with app.run_test() as pilot:
        await pilot.pause()
        assert client.snapshot_calls == 1

        app._request_refresh()
        app._request_refresh()
        app._request_refresh()
        await pilot.pause(0.05)

        assert client.snapshot_calls == 1

        await pilot.pause(1.05)
        assert client.snapshot_calls == 2


@pytest.mark.asyncio
async def test_remote_dashboard_keeps_single_in_flight_snapshot_request() -> None:
    snapshot = {
        "route": {"type": "group", "name": "default"},
        "storage_backend": "sqlite",
        "window": {
            "seconds": 60,
            "request_count": 0,
            "success_count": 0,
            "error_count": 0,
            "error_rate": 0.0,
            "tps": 0.0,
            "total_cost_usd": 0.0,
            "avg_latency_ms": None,
        },
        "reload": {"current_revision": "abc123", "last_success_at": 1, "last_error": None},
        "providers": [],
        "alerts": [],
        "api_keys": [],
        "recent_requests": [],
    }

    client = SlowCountingDashboardClient(snapshot, delay_seconds=0.2)
    app = RemoteDashboardApp(client, enable_stream=False)
    app._refresh_interval_seconds = 0.05

    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        assert client.snapshot_calls == 1

        app._request_refresh()
        app._request_refresh()
        app._request_refresh()
        await pilot.pause(0.1)

        assert client.snapshot_calls == 1

        await pilot.pause(0.25)
        assert client.snapshot_calls == 2


@pytest.mark.asyncio
async def test_remote_dashboard_skips_rerender_when_snapshot_unchanged() -> None:
    snapshot = {
        "route": {"type": "group", "name": "default"},
        "storage_backend": "sqlite",
        "window": {
            "seconds": 60,
            "request_count": 0,
            "success_count": 0,
            "error_count": 0,
            "error_rate": 0.0,
            "tps": 0.0,
            "total_cost_usd": 0.0,
            "avg_latency_ms": None,
        },
        "reload": {"current_revision": "abc123", "last_success_at": 1, "last_error": None},
        "providers": [],
        "alerts": [],
        "api_keys": [],
        "recent_requests": [],
    }

    client = MutableDashboardClient(snapshot)
    app = RemoteDashboardApp(client, enable_stream=False)
    app._refresh_interval_seconds = 0.01

    async with app.run_test() as pilot:
        await pilot.pause()
        summary = app.query_one("#summary", Static)
        first_render = str(summary.render())

        app._request_refresh(force=True)
        await pilot.pause(0.2)
        second_render = str(summary.render())

        assert client.snapshot_calls == 2
        assert first_render == second_render

        updated_snapshot = dict(snapshot)
        updated_snapshot["window"] = dict(snapshot["window"])
        updated_snapshot["window"]["request_count"] = 3
        client.set_snapshot(updated_snapshot)

        app._request_refresh(force=True)
        await pilot.pause(0.2)
        third_render = str(summary.render())

        assert "Window(60s): req=3" in third_render
