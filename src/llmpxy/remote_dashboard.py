from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input, Static


@dataclass
class DashboardClient:
    base_url: str
    admin_token: str

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.admin_token}"}

    async def snapshot(self) -> dict[str, Any]:
        async with httpx.AsyncClient(headers=self.headers()) as client:
            response = await client.get(f"{self.base_url.rstrip('/')}/admin/dashboard/snapshot")
            response.raise_for_status()
            return response.json()

    async def stream(self):
        async with httpx.AsyncClient(headers=self.headers(), timeout=None) as client:
            async with client.stream(
                "GET", f"{self.base_url.rstrip('/')}/admin/dashboard/stream"
            ) as response:
                response.raise_for_status()
                event_type = "message"
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("event: "):
                        event_type = line[7:]
                        continue
                    if line.startswith("data: "):
                        yield event_type, json.loads(line[6:])


class RemoteDashboardApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        layout: horizontal;
        height: 1fr;
    }

    #left, #right {
        width: 1fr;
        height: 1fr;
    }

    #recent {
        height: 14;
    }

    #controls {
        height: auto;
    }

    #filter {
        width: 1fr;
    }

    #details {
        height: 8;
    }

    .panel-title {
        height: auto;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("p", "sort_providers", "Sort Providers"),
        ("k", "sort_keys", "Sort Keys"),
        ("e", "focus_errors", "Errors Only"),
        ("a", "clear_filter", "Clear Filter"),
    ]

    def __init__(self, client: DashboardClient, *, enable_stream: bool = True) -> None:
        super().__init__()
        self._client = client
        self._enable_stream = enable_stream
        self._snapshot: dict[str, Any] = {}
        self._filter_text = ""
        self._provider_sort_mode = "health"
        self._api_key_sort_mode = "usage"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Horizontal(
            Input(placeholder="Filter providers, keys, models, errors", id="filter"),
            Static(
                "p=provider sort  k=key sort  e=errors  a=clear  r=refresh  q=quit", id="controls"
            ),
            id="controls",
        )
        yield Container(
            Horizontal(
                Vertical(
                    Static("Providers", classes="panel-title"),
                    Static("Loading...", id="summary"),
                    DataTable(id="providers"),
                    id="left",
                ),
                Vertical(
                    Static("API Keys", classes="panel-title"),
                    DataTable(id="api_keys"),
                    Static("Recent Requests", classes="panel-title"),
                    DataTable(id="recent"),
                    Static("No request selected", id="details"),
                    id="right",
                ),
            ),
            id="body",
        )
        yield Footer()

    async def on_mount(self) -> None:
        await self._load_snapshot()
        if self._enable_stream:
            self.run_worker(self._stream_events(), exclusive=True)

    async def action_refresh(self) -> None:
        await self._load_snapshot()

    def action_sort_providers(self) -> None:
        if self._provider_sort_mode == "health":
            self._provider_sort_mode = "errors"
        else:
            self._provider_sort_mode = "health"
        self._render_snapshot()

    def action_sort_keys(self) -> None:
        if self._api_key_sort_mode == "usage":
            self._api_key_sort_mode = "name"
        else:
            self._api_key_sort_mode = "usage"
        self._render_snapshot()

    def action_focus_errors(self) -> None:
        self._filter_text = "error"
        self.query_one("#filter", Input).value = self._filter_text
        self._render_snapshot()

    def action_clear_filter(self) -> None:
        self._filter_text = ""
        self.query_one("#filter", Input).value = ""
        self._render_snapshot()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "filter":
            return
        self._filter_text = event.value.strip().lower()
        self._render_snapshot()

    async def _load_snapshot(self) -> None:
        self._snapshot = await self._client.snapshot()
        self._render_snapshot()

    def _render_snapshot(self) -> None:
        summary = self.query_one("#summary", Static)
        details = self.query_one("#details", Static)
        route = self._snapshot.get("route", {})
        reload = self._snapshot.get("reload", {})
        window = self._snapshot.get("window", {})
        providers = list(self._snapshot.get("providers", []))
        api_keys = list(self._snapshot.get("api_keys", []))
        recent_requests = list(self._snapshot.get("recent_requests", []))
        healthy_provider_count = sum(
            1 for provider in providers if provider.get("state") == "healthy"
        )
        failing_provider_count = sum(
            1 for provider in providers if provider.get("state") == "failing"
        )
        success_request_count = sum(
            1 for item in recent_requests if item.get("status") == "success"
        )
        error_request_count = len(recent_requests) - success_request_count
        total_recent_cost = sum(float(item.get("cost_usd", 0.0) or 0.0) for item in recent_requests)
        enabled_key_count = sum(1 for api_key in api_keys if api_key.get("enabled", True))
        summary.update(
            "\n".join(
                [
                    f"Route: {route.get('type', '-')}/{route.get('name', '-')}",
                    f"Storage: {self._snapshot.get('storage_backend', '-')}",
                    f"Revision: {reload.get('current_revision', '-')}",
                    f"Window(60s): req={window.get('request_count', 0)} tps={self._format_float(window.get('tps'), 2)} err_rate={self._format_percent(window.get('error_rate'))} avg_latency={self._format_latency(window.get('avg_latency_ms'))}",
                    f"Providers: total={len(providers)} healthy={healthy_provider_count} failing={failing_provider_count}",
                    f"API Keys: total={len(api_keys)} enabled={enabled_key_count}",
                    f"Recent Requests: total={len(recent_requests)} success={success_request_count} errors={error_request_count} cost=${total_recent_cost:.4f}",
                    f"Last Reload OK: {reload.get('last_success_at', '-')}",
                    f"Last Reload Error: {reload.get('last_error', '-')}",
                ]
            )
        )

        providers = self.query_one("#providers", DataTable)
        providers.clear(columns=True)
        providers.add_columns(
            "Provider",
            "Protocol",
            "State",
            "Errors",
            "RecentOK",
            "RecentErr",
            "InFlight",
            "LastError",
        )
        provider_rows = [
            provider
            for provider in self._snapshot.get("providers", [])
            if self._matches_filter(provider)
        ]
        provider_rows.sort(key=self._provider_sort_key)
        for provider in provider_rows:
            providers.add_row(
                str(provider.get("name", "-")),
                str(provider.get("protocol", "-")),
                str(provider.get("state", "-")),
                str(provider.get("consecutive_errors", 0)),
                str(provider.get("recent_successes", 0)),
                str(provider.get("recent_errors", 0)),
                str(provider.get("in_flight", 0)),
                self._truncate(str(provider.get("last_error_message") or "-"), 36),
            )

        api_keys = self.query_one("#api_keys", DataTable)
        api_keys.clear(columns=True)
        api_keys.add_columns("API Key", "State", "UUID", "Used", "Remaining", "Limit", "Usage")
        api_key_rows = [
            api_key
            for api_key in self._snapshot.get("api_keys", [])
            if self._matches_filter(api_key)
        ]
        api_key_rows.sort(key=self._api_key_sort_key)
        for api_key in api_key_rows:
            utilization_ratio = api_key.get("utilization_ratio")
            if utilization_ratio is None:
                usage_display = "-"
            else:
                usage_display = f"{float(utilization_ratio) * 100:.0f}%"
            api_keys.add_row(
                str(api_key.get("name", "-")),
                "enabled" if api_key.get("enabled", True) else "disabled",
                str(api_key.get("uuid", "-"))[:8],
                f"{float(api_key.get('used_usd', 0.0)):.4f}",
                self._format_budget_value(api_key.get("remaining_usd")),
                self._format_budget_value(api_key.get("limit_usd")),
                usage_display,
            )

        recent = self.query_one("#recent", DataTable)
        recent.clear(columns=True)
        recent.add_columns(
            "Req", "Status", "HTTP", "Key", "Provider", "Model", "Latency", "Cost", "Error"
        )
        filtered_recent_requests = [item for item in recent_requests if self._matches_filter(item)]
        for item in filtered_recent_requests:
            recent.add_row(
                str(item.get("request_id", "-"))[:8],
                str(item.get("status", "-")),
                str(item.get("http_status", "-")),
                str(item.get("api_key_name", "-")),
                str(item.get("provider_name", "-")),
                str(item.get("requested_model", "-")),
                self._format_latency(item.get("latency_ms")),
                f"{float(item.get('cost_usd', 0.0)):.4f}",
                self._truncate(str(item.get("error_message") or item.get("error_code") or "-"), 40),
            )
        selected_request = filtered_recent_requests[0] if filtered_recent_requests else None
        details.update(self._render_request_details(selected_request))

    @staticmethod
    def _format_budget_value(value: object | None) -> str:
        if value is None:
            return "unlimited"
        if isinstance(value, int | float):
            return f"{float(value):.4f}"
        return str(value)

    @staticmethod
    def _format_latency(value: object | None) -> str:
        if value is None:
            return "-"
        if isinstance(value, int | float):
            return f"{int(value)}ms"
        return str(value)

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        if limit <= 3:
            return value[:limit]
        return f"{value[: limit - 3]}..."

    @staticmethod
    def _provider_sort_rank(state: str) -> int:
        if state == "failing":
            return 0
        if state == "degraded":
            return 1
        if state == "healthy":
            return 2
        return 3

    @staticmethod
    def _format_float(value: object | None, digits: int) -> str:
        if isinstance(value, int | float):
            return f"{float(value):.{digits}f}"
        return "-"

    @staticmethod
    def _format_percent(value: object | None) -> str:
        if isinstance(value, int | float):
            return f"{float(value) * 100:.1f}%"
        return "-"

    def _provider_sort_key(self, provider: dict[str, Any]) -> tuple[object, ...]:
        if self._provider_sort_mode == "errors":
            return (
                -int(provider.get("recent_errors", 0) or 0),
                -int(provider.get("consecutive_errors", 0) or 0),
                str(provider.get("name", "")),
            )
        return (
            self._provider_sort_rank(str(provider.get("state", "unknown"))),
            -int(provider.get("consecutive_errors", 0) or 0),
            str(provider.get("name", "")),
        )

    def _api_key_sort_key(self, api_key: dict[str, Any]) -> tuple[object, ...]:
        if self._api_key_sort_mode == "name":
            return (
                not bool(api_key.get("enabled", True)),
                str(api_key.get("name", "")),
            )
        return (
            not bool(api_key.get("enabled", True)),
            -float(api_key.get("utilization_ratio", -1.0) or -1.0),
            str(api_key.get("name", "")),
        )

    def _matches_filter(self, item: dict[str, Any]) -> bool:
        if not self._filter_text:
            return True
        haystacks = [
            str(item.get("name", "")),
            str(item.get("uuid", "")),
            str(item.get("api_key_name", "")),
            str(item.get("provider_name", "")),
            str(item.get("requested_model", "")),
            str(item.get("status", "")),
            str(item.get("state", "")),
            str(item.get("last_error_message", "")),
            str(item.get("error_message", "")),
            str(item.get("error_code", "")),
        ]
        return self._filter_text in " ".join(haystacks).lower()

    def _render_request_details(self, request: dict[str, Any] | None) -> str:
        if request is None:
            return "No request selected"
        return "\n".join(
            [
                f"Request: {request.get('request_id', '-')}",
                f"Status: {request.get('status', '-')} http={request.get('http_status', '-')}",
                f"Key: {request.get('api_key_name', '-')}  Provider: {request.get('provider_name', '-')}",
                f"Model: {request.get('requested_model', '-')}  Latency: {self._format_latency(request.get('latency_ms'))}",
                f"Cost: ${float(request.get('cost_usd', 0.0) or 0.0):.4f}",
                f"Error: {request.get('error_message') or request.get('error_code') or '-'}",
            ]
        )

    async def _stream_events(self) -> None:
        while True:
            try:
                async for _event_type, _payload in self._client.stream():
                    await self._load_snapshot()
            except Exception:
                await asyncio.sleep(1.0)


def run_remote_dashboard(base_url: str, admin_token: str) -> None:
    app = RemoteDashboardApp(DashboardClient(base_url=base_url, admin_token=admin_token))
    app.run()
