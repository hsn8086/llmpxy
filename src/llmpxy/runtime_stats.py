from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from llmpxy.models import AdminEvent


@dataclass
class ProviderRuntimeStats:
    consecutive_errors: int = 0
    last_success_at: int | None = None
    last_error_at: int | None = None
    last_error_message: str | None = None
    in_flight: int = 0
    recent_successes: int = 0
    recent_errors: int = 0


@dataclass
class ReloadRuntimeStats:
    current_revision: str = "unknown"
    last_attempt_at: int | None = None
    last_success_at: int | None = None
    last_error_at: int | None = None
    last_error: str | None = None


@dataclass
class RuntimeStats:
    provider_stats: dict[str, ProviderRuntimeStats] = field(default_factory=dict)
    reload: ReloadRuntimeStats = field(default_factory=ReloadRuntimeStats)
    events: deque[AdminEvent] = field(default_factory=lambda: deque(maxlen=200))
    _condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    def ensure_provider(self, provider_name: str) -> ProviderRuntimeStats:
        return self.provider_stats.setdefault(provider_name, ProviderRuntimeStats())

    def record_provider_attempt(self, provider_name: str) -> None:
        self.ensure_provider(provider_name).in_flight += 1

    def record_provider_success(self, provider_name: str) -> None:
        stats = self.ensure_provider(provider_name)
        stats.in_flight = max(stats.in_flight - 1, 0)
        stats.consecutive_errors = 0
        stats.last_success_at = int(time.time())
        stats.recent_successes += 1

    def record_provider_error(self, provider_name: str, message: str) -> None:
        stats = self.ensure_provider(provider_name)
        stats.in_flight = max(stats.in_flight - 1, 0)
        stats.consecutive_errors += 1
        stats.last_error_at = int(time.time())
        stats.last_error_message = message
        stats.recent_errors += 1

    def provider_state(self, provider_name: str) -> str:
        stats = self.ensure_provider(provider_name)
        if stats.last_success_at is None and stats.last_error_at is None:
            return "unknown"
        if stats.consecutive_errors == 0:
            return "healthy"
        if stats.consecutive_errors >= 3:
            return "failing"
        return "degraded"

    async def publish(
        self, event_type: Literal["request", "reload", "config"], payload: dict[str, Any]
    ) -> None:
        event = AdminEvent(event_type=event_type, created_at=int(time.time()), payload=payload)
        async with self._condition:
            self.events.append(event)
            self._condition.notify_all()

    async def wait_for_events(self, last_seen: int, timeout: float = 3.0) -> list[AdminEvent]:
        async with self._condition:
            if len(self.events) <= last_seen:
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=timeout)
                except TimeoutError:
                    return []
            return list(self.events)[last_seen:]
