from __future__ import annotations

from dataclasses import dataclass
import hashlib
import threading
import time
from pathlib import Path

from fastapi import HTTPException

from llmpxy.auth import ApiKeyRegistry, AuthenticatedApiKey
from llmpxy.billing import UsageCost, calculate_usage_cost
from llmpxy.config import AppConfig, load_config
from llmpxy.config import ProviderConfig
from llmpxy.dispatcher import ProviderDispatcher
from llmpxy.logging_utils import bind_logger, configure_logging
from llmpxy.models import ApiKeyUsageRecord, CanonicalRequest, CanonicalResponse, RequestEventRecord
from llmpxy.storage import ConversationStore
from llmpxy.storage_file import FileConversationStore
from llmpxy.storage_sqlite import SQLiteConversationStore
from llmpxy.runtime_stats import RuntimeStats


def build_store(config: AppConfig, config_path: Path) -> ConversationStore:
    if config.storage.backend == "sqlite":
        return SQLiteConversationStore(config.resolve_sqlite_path(config_path))
    return FileConversationStore(config.resolve_file_dir(config_path))


@dataclass
class RuntimeState:
    config_path: Path
    config: AppConfig
    store: ConversationStore
    dispatcher: ProviderDispatcher
    api_keys: ApiKeyRegistry
    config_mtime_ns: int


class RuntimeManager:
    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path.resolve()
        self._lock = threading.RLock()
        self._stats = RuntimeStats()
        self._state = self._load_state()

    def config(self) -> AppConfig:
        return self.current().config

    def stats(self) -> RuntimeStats:
        return self._stats

    def current(self) -> RuntimeState:
        with self._lock:
            self._reload_if_needed_locked()
            return self._state

    def force_reload(self) -> None:
        with self._lock:
            self._state = self._load_state()

    def authenticate(self, authorization: str | None) -> AuthenticatedApiKey:
        state = self.current()
        return state.api_keys.authenticate(authorization)

    def enforce_limit(self, api_key: AuthenticatedApiKey, provider_name: str) -> None:
        state = self.current()
        self._ensure_within_limits(api_key, provider_name, state)

    def selectable_providers(
        self,
        api_key: AuthenticatedApiKey,
        provider_order: list[ProviderConfig] | None = None,
    ) -> list[str]:
        state = self.current()
        selectable: list[str] = []
        routed_providers = (
            provider_order or state.dispatcher.resolve_route_provider_order_for_limits()
        )
        for provider in routed_providers:
            try:
                self._ensure_within_limits(api_key, provider.name, state)
            except HTTPException:
                continue
            selectable.append(provider.name)
        return selectable

    def selectable_provider_configs(
        self,
        api_key: AuthenticatedApiKey,
        provider_order: list[ProviderConfig] | None = None,
    ) -> list[ProviderConfig]:
        state = self.current()
        routed_providers = (
            provider_order or state.dispatcher.resolve_route_provider_order_for_limits()
        )
        selected_names = set(self.selectable_providers(api_key, routed_providers))
        return [provider for provider in routed_providers if provider.name in selected_names]

    def record_usage(
        self,
        *,
        api_key: AuthenticatedApiKey,
        request_id: str,
        request: CanonicalRequest,
        response: CanonicalResponse,
        provider_name: str,
    ) -> UsageCost:
        state = self.current()
        cost = calculate_usage_cost(
            state.config.provider_map()[provider_name],
            request.requested_model,
            response.model,
            response.usage,
        )
        state.store.put_api_key_usage(
            ApiKeyUsageRecord(
                api_key_uuid=api_key.uuid,
                api_key_name=api_key.name,
                request_id=request_id,
                provider_name=provider_name,
                requested_model=request.requested_model,
                upstream_model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
                cost_usd=cost.amount_usd,
                created_at=response.created_at,
            )
        )
        state.store.put_request_event(
            RequestEventRecord(
                request_id=request_id,
                started_at=response.created_at,
                finished_at=int(time.time()),
                latency_ms=max(int(time.time()) - response.created_at, 0) * 1000,
                protocol_in=request.protocol_in,
                stream=request.stream,
                api_key_uuid=api_key.uuid,
                api_key_name=api_key.name,
                provider_name=provider_name,
                requested_model=request.requested_model,
                upstream_model=response.model,
                status="success",
                http_status=200,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
                cost_usd=cost.amount_usd,
            )
        )
        self._log_post_request_limit_state(api_key, provider_name, state, request_id)
        return cost

    async def publish_request_event(self, payload: dict[str, object]) -> None:
        await self._stats.publish("request", payload)

    async def publish_config_event(self, payload: dict[str, object]) -> None:
        await self._stats.publish("config", payload)

    def runtime_snapshot(self) -> dict[str, object]:
        state = self.current()
        now = int(time.time())
        recent_requests = state.store.list_recent_request_events(20)
        recent_window_seconds = 60
        windows = {
            "10s": self._window_snapshot(state, now, 10),
            "60s": self._window_snapshot(state, now, 60),
            "5m": self._window_snapshot(state, now, 300),
        }
        window_snapshot = windows["60s"]
        provider_snapshots = []
        provider_alerts: list[dict[str, object]] = []
        for provider in state.config.providers:
            provider_stats = self._stats.ensure_provider(provider.name)
            provider_state = self._stats.provider_state(provider.name)
            provider_window = self._provider_window_snapshot(state, now, provider.name, 60)
            provider_snapshots.append(
                {
                    "name": provider.name,
                    "protocol": provider.protocol,
                    "state": provider_state,
                    "consecutive_errors": provider_stats.consecutive_errors,
                    "last_success_at": provider_stats.last_success_at,
                    "last_error_at": provider_stats.last_error_at,
                    "last_error_message": provider_stats.last_error_message,
                    "in_flight": provider_stats.in_flight,
                    "recent_successes": provider_stats.recent_successes,
                    "recent_errors": provider_stats.recent_errors,
                    "window": provider_window,
                }
            )
            if provider_state in {"failing", "degraded"}:
                provider_alerts.append(
                    {
                        "kind": "provider_health",
                        "name": provider.name,
                        "severity": provider_state,
                        "message": provider_stats.last_error_message or provider_state,
                    }
                )
        api_key_balances = []
        api_key_alerts: list[dict[str, object]] = []
        for api_key in state.config.api_keys:
            used_usd = state.store.get_api_key_total_cost(api_key.uuid)
            remaining_usd = None
            utilization_ratio = None
            if api_key.limit_usd is not None:
                remaining_usd = max(api_key.limit_usd - used_usd, 0.0)
                utilization_ratio = 0.0
                if api_key.limit_usd > 0:
                    utilization_ratio = min(used_usd / api_key.limit_usd, 1.0)
            api_key_balances.append(
                {
                    "uuid": api_key.uuid,
                    "name": api_key.name,
                    "enabled": api_key.enabled,
                    "used_usd": used_usd,
                    "limit_usd": api_key.limit_usd,
                    "remaining_usd": remaining_usd,
                    "utilization_ratio": utilization_ratio,
                }
            )
            if not api_key.enabled:
                api_key_alerts.append(
                    {
                        "kind": "api_key_disabled",
                        "name": api_key.name,
                        "severity": "info",
                        "message": "API key disabled",
                    }
                )
            elif utilization_ratio is not None and utilization_ratio >= 0.8:
                severity = "critical" if utilization_ratio >= 1.0 else "warning"
                api_key_alerts.append(
                    {
                        "kind": "api_key_budget",
                        "name": api_key.name,
                        "severity": severity,
                        "message": f"Budget usage {utilization_ratio * 100:.0f}%",
                    }
                )
        alerts = provider_alerts + api_key_alerts
        return {
            "route": state.config.route.model_dump(mode="json"),
            "storage_backend": state.config.storage.backend,
            "reload": {
                "current_revision": self._stats.reload.current_revision,
                "last_attempt_at": self._stats.reload.last_attempt_at,
                "last_success_at": self._stats.reload.last_success_at,
                "last_error_at": self._stats.reload.last_error_at,
                "last_error": self._stats.reload.last_error,
            },
            "window": {
                **window_snapshot,
                "seconds": recent_window_seconds,
            },
            "windows": windows,
            "providers": provider_snapshots,
            "api_keys": api_key_balances,
            "alerts": alerts,
            "recent_requests": [item.model_dump(mode="json") for item in recent_requests],
        }

    def _window_snapshot(self, state: RuntimeState, now: int, seconds: int) -> dict[str, object]:
        window_started_at = now - seconds
        window_requests = state.store.list_request_events_since(window_started_at)
        request_count = len(window_requests)
        success_count = sum(1 for item in window_requests if item.status == "success")
        error_count = request_count - success_count
        total_cost_usd = sum(item.cost_usd for item in window_requests)
        total_tokens = sum(item.total_tokens or 0 for item in window_requests)
        latency_values = [
            item.latency_ms for item in window_requests if item.latency_ms is not None
        ]
        avg_latency_ms = None
        p95_latency_ms = None
        if latency_values:
            avg_latency_ms = int(sum(latency_values) / len(latency_values))
            sorted_latencies = sorted(latency_values)
            index = max(0, int(len(sorted_latencies) * 0.95) - 1)
            p95_latency_ms = sorted_latencies[index]
        error_rate = 0.0
        if request_count > 0:
            error_rate = error_count / request_count
        return {
            "request_count": request_count,
            "success_count": success_count,
            "error_count": error_count,
            "error_rate": error_rate,
            "tps": request_count / seconds,
            "success_tps": success_count / seconds,
            "error_tps": error_count / seconds,
            "total_cost_usd": total_cost_usd,
            "avg_latency_ms": avg_latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "total_tokens": total_tokens,
        }

    def _provider_window_snapshot(
        self, state: RuntimeState, now: int, provider_name: str, seconds: int
    ) -> dict[str, object]:
        window_started_at = now - seconds
        requests = [
            item
            for item in state.store.list_request_events_since(window_started_at)
            if item.provider_name == provider_name
        ]
        request_count = len(requests)
        success_count = sum(1 for item in requests if item.status == "success")
        error_count = request_count - success_count
        avg_latency_ms = None
        latency_values = [item.latency_ms for item in requests if item.latency_ms is not None]
        if latency_values:
            avg_latency_ms = int(sum(latency_values) / len(latency_values))
        error_rate = 0.0
        if request_count > 0:
            error_rate = error_count / request_count
        return {
            "seconds": seconds,
            "request_count": request_count,
            "success_count": success_count,
            "error_count": error_count,
            "error_rate": error_rate,
            "tps": request_count / seconds,
            "avg_latency_ms": avg_latency_ms,
            "total_cost_usd": sum(item.cost_usd for item in requests),
        }

    def masked_config(self) -> dict[str, object]:
        state = self.current()
        payload = state.config.model_dump(mode="json")
        for api_key in payload.get("api_keys", []):
            key = api_key.get("key")
            if isinstance(key, str):
                api_key["key"] = _mask_secret(key)
        if isinstance(payload.get("admin"), dict):
            admin_payload = payload["admin"]
            token = admin_payload.get("token")
            if isinstance(token, str):
                admin_payload["token"] = _mask_secret(token)
        return payload

    def _ensure_within_limits(
        self,
        api_key: AuthenticatedApiKey,
        provider_name: str,
        state: RuntimeState,
    ) -> None:
        total_cost = state.store.get_api_key_total_cost(api_key.uuid)
        if api_key.limit_usd is not None and total_cost >= api_key.limit_usd:
            raise HTTPException(status_code=429, detail="API key budget exceeded")

        provider_limit = api_key.provider_limits_usd.get(provider_name)
        if provider_limit is not None:
            provider_cost = state.store.get_api_key_total_cost(api_key.uuid, [provider_name])
            if provider_cost >= provider_limit:
                raise HTTPException(status_code=429, detail="API key provider budget exceeded")

        group_names = state.dispatcher._group_names_for_provider(provider_name)
        for group_name in group_names:
            group_limit = api_key.group_limits_usd.get(group_name)
            if group_limit is None:
                continue
            group_provider_names = state.dispatcher.providers_for_group(group_name)
            group_cost = state.store.get_api_key_total_cost(api_key.uuid, group_provider_names)
            if group_cost >= group_limit:
                raise HTTPException(status_code=429, detail="API key group budget exceeded")

    def _log_post_request_limit_state(
        self,
        api_key: AuthenticatedApiKey,
        provider_name: str,
        state: RuntimeState,
        request_id: str,
    ) -> None:
        total_cost = state.store.get_api_key_total_cost(api_key.uuid)
        if api_key.limit_usd is not None and total_cost > api_key.limit_usd:
            bind_logger(request_id=request_id, provider=provider_name).warning(
                "api key exceeded total budget after request api_key_uuid={} api_key={} limit_usd={} total_cost_usd={}",
                api_key.uuid,
                api_key.name,
                api_key.limit_usd,
                total_cost,
            )
        provider_limit = api_key.provider_limits_usd.get(provider_name)
        if provider_limit is not None:
            provider_cost = state.store.get_api_key_total_cost(api_key.uuid, [provider_name])
            if provider_cost > provider_limit:
                bind_logger(request_id=request_id, provider=provider_name).warning(
                    "api key exceeded provider budget after request api_key_uuid={} api_key={} limit_usd={} provider_cost_usd={}",
                    api_key.uuid,
                    api_key.name,
                    provider_limit,
                    provider_cost,
                )
        for group_name in state.dispatcher._group_names_for_provider(provider_name):
            group_limit = api_key.group_limits_usd.get(group_name)
            if group_limit is None:
                continue
            group_provider_names = state.dispatcher.providers_for_group(group_name)
            group_cost = state.store.get_api_key_total_cost(api_key.uuid, group_provider_names)
            if group_cost > group_limit:
                bind_logger(request_id=request_id, provider=provider_name).warning(
                    "api key exceeded group budget after request api_key_uuid={} api_key={} group={} limit_usd={} group_cost_usd={}",
                    api_key.uuid,
                    api_key.name,
                    group_name,
                    group_limit,
                    group_cost,
                )

    def _reload_if_needed_locked(self) -> None:
        current_mtime_ns = self._config_path.stat().st_mtime_ns
        if current_mtime_ns == self._state.config_mtime_ns:
            return
        self._stats.reload.last_attempt_at = int(time.time())
        try:
            reloaded_state = self._load_state()
        except Exception as exc:
            self._stats.reload.last_error_at = int(time.time())
            self._stats.reload.last_error = str(exc)
            bind_logger().exception(
                "config reload rejected, keeping previous config path={} error={}",
                str(self._config_path),
                str(exc),
            )
            return
        self._state = reloaded_state
        self._stats.reload.last_success_at = int(time.time())
        self._stats.reload.last_error = None
        bind_logger().info("config reloaded path={}", str(self._config_path))

    def _load_state(self) -> RuntimeState:
        config = load_config(self._config_path)
        configure_logging(config, self._config_path)
        store = build_store(config, self._config_path)
        dispatcher = ProviderDispatcher(config, self._stats)
        api_keys = ApiKeyRegistry(config)
        self._stats.reload.current_revision = _config_revision(self._config_path)
        return RuntimeState(
            config_path=self._config_path,
            config=config,
            store=store,
            dispatcher=dispatcher,
            api_keys=api_keys,
            config_mtime_ns=self._config_path.stat().st_mtime_ns,
        )


def _config_revision(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:12]


def _mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"
