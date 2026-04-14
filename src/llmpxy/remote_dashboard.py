from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, cast

import httpx
from textual.css.query import NoMatches
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Static


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

    #details {
        height: 8;
    }

    #alerts {
        height: 8;
    }

    #provider_details, #api_key_details {
        height: 7;
    }

    .state-good {
        color: $success;
    }

    .state-warn {
        color: $warning;
    }

    .state-bad {
        color: $error;
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
        self._snapshot_signature: str | None = None
        self._filter_text = ""
        self._provider_sort_mode = "health"
        self._api_key_sort_mode = "usage"
        self._selected_provider_name: str | None = None
        self._selected_api_key_uuid: str | None = None
        self._selected_recent_request_id: str | None = None
        self._refresh_interval_seconds = 2.0
        self._refresh_in_flight = False
        self._snapshot_dirty = True
        self._last_refresh_completed_at: float | None = None
        self._force_refresh = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            "p=provider sort  k=key sort  e=errors  a=clear  r=refresh  q=quit", id="controls"
        )
        yield Container(
            Horizontal(
                Vertical(
                    Static("Providers", classes="panel-title"),
                    Static("Loading...", id="summary"),
                    Static("No active alerts", id="alerts", classes="state-warn"),
                    DataTable(id="providers"),
                    Static("No provider selected", id="provider_details"),
                    id="left",
                ),
                Vertical(
                    Static("API Keys", classes="panel-title"),
                    DataTable(id="api_keys"),
                    Static("No API key selected", id="api_key_details"),
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
        providers_table = self.query_one("#providers", DataTable)
        providers_table.cursor_type = "row"
        providers_table.add_columns(
            "Provider",
            "Protocol",
            "State",
            "Errors",
            "RecentOK",
            "RecentErr",
            "InFlight",
            "LastError",
        )

        api_keys_table = self.query_one("#api_keys", DataTable)
        api_keys_table.cursor_type = "row"
        api_keys_table.add_columns(
            "API Key", "State", "UUID", "Used", "Remaining", "Limit", "Usage"
        )

        recent_table = self.query_one("#recent", DataTable)
        recent_table.cursor_type = "row"
        recent_table.add_columns(
            "Req",
            "Started",
            "Status",
            "HTTP",
            "Key",
            "Provider",
            "Model",
            "In",
            "Cached",
            "Out",
            "Latency",
            "Cost",
            "Error",
        )
        self.run_worker(self._refresh_loop(), exclusive=True, group="dashboard-refresh")
        if self._enable_stream:
            self.run_worker(self._stream_events(), exclusive=True)

    async def action_refresh(self) -> None:
        self._request_refresh(force=True)

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
        self._render_snapshot()

    def action_clear_filter(self) -> None:
        self._filter_text = ""
        self._render_snapshot()

    async def _load_snapshot(self) -> None:
        self._refresh_in_flight = True
        self._snapshot_dirty = False
        try:
            snapshot = await self._client.snapshot()
        except Exception:
            self._snapshot_dirty = True
            raise
        finally:
            self._last_refresh_completed_at = time.monotonic()
            self._refresh_in_flight = False
        snapshot_signature = self._snapshot_signature_for(snapshot)
        if snapshot_signature == self._snapshot_signature:
            return
        self._snapshot = snapshot
        self._snapshot_signature = snapshot_signature
        self._render_snapshot()

    def _request_refresh(self, *, force: bool = False) -> None:
        self._snapshot_dirty = True
        if force:
            self._force_refresh = True

    @staticmethod
    def _snapshot_signature_for(snapshot: dict[str, Any]) -> str:
        return json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    async def _refresh_loop(self) -> None:
        while True:
            should_refresh = self._snapshot_dirty and not self._refresh_in_flight
            interval_elapsed = self._last_refresh_completed_at is None or (
                time.monotonic() - self._last_refresh_completed_at >= self._refresh_interval_seconds
            )
            if should_refresh and (self._force_refresh or interval_elapsed):
                try:
                    self._force_refresh = False
                    await self._load_snapshot()
                except Exception:
                    await asyncio.sleep(self._refresh_interval_seconds)
                    continue
            await asyncio.sleep(0.1)

    def _render_snapshot(self) -> None:
        summary = self.query_one("#summary", Static)
        alerts_widget = self.query_one("#alerts", Static)
        provider_details = self.query_one("#provider_details", Static)
        api_key_details = self.query_one("#api_key_details", Static)
        details = self.query_one("#details", Static)
        route = self._snapshot.get("route", {})
        reload = self._snapshot.get("reload", {})
        window = self._snapshot.get("window", {})
        windows = self._snapshot.get("windows", {})
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
                    f"Filter: {self._filter_text or '-'}  ProviderSort: {self._provider_sort_mode}  KeySort: {self._api_key_sort_mode}",
                    f"Window(60s): req={window.get('request_count', 0)} tps={self._format_float(window.get('tps'), 2)} err_rate={self._format_percent(window.get('error_rate'))} avg_latency={self._format_latency(window.get('avg_latency_ms'))}",
                    f"Window(10s): tps={self._window_value(windows, '10s', 'tps')} success_tps={self._window_value(windows, '10s', 'success_tps')} error_tps={self._window_value(windows, '10s', 'error_tps')}",
                    f"Window(5m): req={self._window_value(windows, '5m', 'request_count', integer=True)} p95={self._window_latency(windows, '5m', 'p95_latency_ms')} tokens={self._window_value(windows, '5m', 'total_tokens', integer=True)} cost=${self._window_value(windows, '5m', 'total_cost_usd')}",
                    f"Providers: total={len(providers)} healthy={healthy_provider_count} failing={failing_provider_count}",
                    f"API Keys: total={len(api_keys)} enabled={enabled_key_count}",
                    f"Recent Requests: total={len(recent_requests)} success={success_request_count} errors={error_request_count} cost=${total_recent_cost:.4f}",
                    f"Last Reload OK: {reload.get('last_success_at', '-')}",
                    f"Last Reload Error: {reload.get('last_error', '-')}",
                ]
            )
        )
        alerts = [
            alert for alert in self._snapshot.get("alerts", []) if self._matches_filter(alert)
        ]
        alerts_widget.update(self._render_alerts(alerts))
        alerts_widget.set_class(bool(alerts), "state-warn")
        alerts_widget.set_class(
            any(self._is_critical_alert(alert) for alert in alerts), "state-bad"
        )

        providers = self.query_one("#providers", DataTable)
        providers.clear()
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
                key=str(provider.get("name", "-")),
            )
        selected_provider = self._find_selected_item(
            provider_rows,
            self._selected_provider_name,
            lambda item: str(item.get("name", "-")),
        )
        if selected_provider is not None:
            self._selected_provider_name = str(selected_provider.get("name", "-"))
        else:
            self._selected_provider_name = None
        provider_details.update(self._render_provider_details(selected_provider))
        provider_details.set_class(
            bool(selected_provider and str(selected_provider.get("state", "")) == "healthy"),
            "state-good",
        )
        provider_details.set_class(
            bool(selected_provider and str(selected_provider.get("state", "")) == "degraded"),
            "state-warn",
        )
        provider_details.set_class(
            bool(selected_provider and str(selected_provider.get("state", "")) == "failing"),
            "state-bad",
        )

        api_keys = self.query_one("#api_keys", DataTable)
        api_keys.clear()
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
                key=str(api_key.get("uuid", "-")),
            )
        selected_api_key = self._find_selected_item(
            api_key_rows,
            self._selected_api_key_uuid,
            lambda item: str(item.get("uuid", "-")),
        )
        if selected_api_key is not None:
            self._selected_api_key_uuid = str(selected_api_key.get("uuid", "-"))
        else:
            self._selected_api_key_uuid = None
        api_key_details.update(self._render_api_key_details(selected_api_key))
        api_key_details.set_class(
            bool(selected_api_key and not bool(selected_api_key.get("enabled", True))),
            "state-warn",
        )
        api_key_details.set_class(
            bool(
                selected_api_key
                and isinstance(selected_api_key.get("utilization_ratio"), int | float)
                and float(selected_api_key.get("utilization_ratio") or 0.0) >= 1.0
            ),
            "state-bad",
        )

        recent = self.query_one("#recent", DataTable)
        recent.clear()
        filtered_recent_requests = [item for item in recent_requests if self._matches_filter(item)]
        for item in filtered_recent_requests:
            recent.add_row(
                str(item.get("request_id", "-"))[:8],
                self._format_timestamp(item.get("started_at")),
                str(item.get("status", "-")),
                str(item.get("http_status", "-")),
                str(item.get("api_key_name", "-")),
                str(item.get("provider_name", "-")),
                str(item.get("requested_model", "-")),
                self._format_integer(item.get("input_tokens")),
                self._format_integer(item.get("cached_input_tokens")),
                self._format_integer(item.get("output_tokens")),
                self._format_latency(item.get("latency_ms")),
                f"{float(item.get('cost_usd', 0.0)):.4f}",
                self._truncate(str(item.get("error_message") or item.get("error_code") or "-"), 40),
                key=str(item.get("request_id", "-")),
            )
        selected_request = self._find_selected_item(
            filtered_recent_requests,
            self._selected_recent_request_id,
            lambda item: str(item.get("request_id", "-")),
        )
        if selected_request is not None:
            self._selected_recent_request_id = str(selected_request.get("request_id", "-"))
        else:
            self._selected_recent_request_id = None
        details.update(self._render_request_details(selected_request))
        details.set_class(
            bool(selected_request and str(selected_request.get("status", "")) == "success"),
            "state-good",
        )
        details.set_class(
            bool(
                selected_request
                and str(selected_request.get("status", "")) in {"provider_error", "all_failed"}
            ),
            "state-bad",
        )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        row_key = event.row_key.value
        if not isinstance(row_key, str):
            return
        if event.data_table.id == "recent":
            self._selected_recent_request_id = row_key
            recent_requests = list(self._snapshot.get("recent_requests", []))
            filtered_recent_requests = [
                item for item in recent_requests if self._matches_filter(item)
            ]
            selected_request = self._find_selected_item(
                filtered_recent_requests,
                self._selected_recent_request_id,
                lambda item: str(item.get("request_id", "-")),
            )
            details = self.query_one("#details", Static)
            details.update(self._render_request_details(selected_request))
            details.set_class(
                bool(selected_request and str(selected_request.get("status", "")) == "success"),
                "state-good",
            )
            details.set_class(
                bool(
                    selected_request
                    and str(selected_request.get("status", "")) in {"provider_error", "all_failed"}
                ),
                "state-bad",
            )
        elif event.data_table.id == "providers":
            self._selected_provider_name = row_key
            provider_rows = [
                provider
                for provider in self._snapshot.get("providers", [])
                if self._matches_filter(provider)
            ]
            provider_rows.sort(key=self._provider_sort_key)
            selected_provider = self._find_selected_item(
                provider_rows,
                self._selected_provider_name,
                lambda item: str(item.get("name", "-")),
            )
            provider_details = self.query_one("#provider_details", Static)
            provider_details.update(self._render_provider_details(selected_provider))
            provider_details.set_class(
                bool(selected_provider and str(selected_provider.get("state", "")) == "healthy"),
                "state-good",
            )
            provider_details.set_class(
                bool(selected_provider and str(selected_provider.get("state", "")) == "degraded"),
                "state-warn",
            )
            provider_details.set_class(
                bool(selected_provider and str(selected_provider.get("state", "")) == "failing"),
                "state-bad",
            )
        elif event.data_table.id == "api_keys":
            self._selected_api_key_uuid = row_key
            api_key_rows = [
                api_key
                for api_key in self._snapshot.get("api_keys", [])
                if self._matches_filter(api_key)
            ]
            api_key_rows.sort(key=self._api_key_sort_key)
            selected_api_key = self._find_selected_item(
                api_key_rows,
                self._selected_api_key_uuid,
                lambda item: str(item.get("uuid", "-")),
            )
            api_key_details = self.query_one("#api_key_details", Static)
            api_key_details.update(self._render_api_key_details(selected_api_key))
            api_key_details.set_class(
                bool(selected_api_key and not bool(selected_api_key.get("enabled", True))),
                "state-warn",
            )
            api_key_details.set_class(
                bool(
                    selected_api_key
                    and isinstance(selected_api_key.get("utilization_ratio"), int | float)
                    and float(selected_api_key.get("utilization_ratio") or 0.0) >= 1.0
                ),
                "state-bad",
            )
        else:
            return
        try:
            self.refresh(layout=False)
        except NoMatches:
            return

    def _find_selected_item(
        self,
        items: list[dict[str, Any]],
        selected_key: str | None,
        get_key: Callable[[dict[str, Any]], str],
    ) -> dict[str, Any] | None:
        if not items:
            return None
        if selected_key is None:
            return items[0]
        for item in items:
            if get_key(item) == selected_key:
                return item
        return items[0]

    def _render_alerts(self, alerts: list[dict[str, Any]]) -> str:
        if not alerts:
            return "No active alerts"
        lines = []
        for alert in alerts[:5]:
            lines.append(
                f"{str(alert.get('severity', '-')).upper()} {alert.get('name', '-')} - {alert.get('message', '-')}"
            )
        if len(alerts) > 5:
            lines.append(f"... and {len(alerts) - 5} more alerts")
        return "\n".join(lines)

    @staticmethod
    def _is_critical_alert(alert: dict[str, Any]) -> bool:
        return str(alert.get("severity", "")).lower() in {"critical", "failing"}

    def _render_provider_details(self, provider: dict[str, Any] | None) -> str:
        if provider is None:
            return "No provider selected"
        return "\n".join(
            [
                f"Provider: {provider.get('name', '-')}",
                f"Protocol: {provider.get('protocol', '-')}  State: {provider.get('state', '-')}",
                f"Errors: consecutive={provider.get('consecutive_errors', 0)} recent={provider.get('recent_errors', 0)}",
                f"Success: recent={provider.get('recent_successes', 0)}  InFlight: {provider.get('in_flight', 0)}",
                f"Window(60s): req={provider.get('window', {}).get('request_count', 0)} tps={self._format_float(provider.get('window', {}).get('tps'), 2)} err_rate={self._format_percent(provider.get('window', {}).get('error_rate'))}",
                f"Window(60s): avg_latency={self._format_latency(provider.get('window', {}).get('avg_latency_ms'))} cost=${self._format_float(provider.get('window', {}).get('total_cost_usd'), 4)}",
                f"Window(60s): in={self._format_integer(provider.get('window', {}).get('total_input_tokens'))} cached={self._format_integer(provider.get('window', {}).get('total_cached_input_tokens'))} out={self._format_integer(provider.get('window', {}).get('total_output_tokens'))}",
                f"Last Success: {self._format_timestamp(provider.get('last_success_at'))}",
                f"Last Error: {provider.get('last_error_message') or '-'}",
            ]
        )

    def _render_api_key_details(self, api_key: dict[str, Any] | None) -> str:
        if api_key is None:
            return "No API key selected"
        utilization_ratio = api_key.get("utilization_ratio")
        usage = "-"
        if isinstance(utilization_ratio, int | float):
            usage = f"{float(utilization_ratio) * 100:.0f}%"
        return "\n".join(
            [
                f"API Key: {api_key.get('name', '-')}",
                f"UUID: {api_key.get('uuid', '-')}",
                f"State: {'enabled' if api_key.get('enabled', True) else 'disabled'}  Usage: {usage}",
                f"Used: ${float(api_key.get('used_usd', 0.0) or 0.0):.4f}",
                f"Remaining: {self._format_budget_value(api_key.get('remaining_usd'))}",
                f"Limit: {self._format_budget_value(api_key.get('limit_usd'))}",
            ]
        )

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

    def _window_value(
        self, windows: object, window_name: str, key: str, *, integer: bool = False
    ) -> str:
        if not isinstance(windows, dict):
            return "-"
        window_map = cast(dict[str, object], windows)
        window = window_map.get(window_name)
        if not isinstance(window, dict):
            return "-"
        value = cast(dict[str, object], window).get(key)
        if isinstance(value, int | float):
            if integer:
                return str(int(value))
            return self._format_float(value, 2)
        return "-"

    def _window_latency(self, windows: object, window_name: str, key: str) -> str:
        if not isinstance(windows, dict):
            return "-"
        window_map = cast(dict[str, object], windows)
        window = window_map.get(window_name)
        if not isinstance(window, dict):
            return "-"
        return self._format_latency(cast(dict[str, object], window).get(key))

    @staticmethod
    def _format_timestamp(value: object | None) -> str:
        if not isinstance(value, int | float):
            return "-"
        return time.strftime("%H:%M:%S", time.localtime(int(value)))

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
                f"Tokens: in={self._format_integer(request.get('input_tokens'))} cached={self._format_integer(request.get('cached_input_tokens'))} out={self._format_integer(request.get('output_tokens'))} total={self._format_integer(request.get('total_tokens'))}",
                f"Cost: ${float(request.get('cost_usd', 0.0) or 0.0):.4f}",
                f"Error: {request.get('error_message') or request.get('error_code') or '-'}",
            ]
        )

    @staticmethod
    def _format_integer(value: object | None) -> str:
        if isinstance(value, int | float):
            return str(int(value))
        return "-"

    async def _stream_events(self) -> None:
        while True:
            try:
                async for _event_type, _payload in self._client.stream():
                    self._request_refresh()
            except Exception:
                await asyncio.sleep(1.0)


def run_remote_dashboard(base_url: str, admin_token: str) -> None:
    app = RemoteDashboardApp(DashboardClient(base_url=base_url, admin_token=admin_token))
    app.run()
