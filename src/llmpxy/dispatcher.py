from __future__ import annotations

import asyncio
from dataclasses import dataclass

from llmpxy.config import AppConfig, ProviderConfig, ProviderGroupConfig
from llmpxy.logging_utils import bind_logger, sanitize_for_logging
from llmpxy.models import CanonicalRequest, CanonicalResponse, CanonicalStreamEvent
from llmpxy.protocols.registry import get_adapter
from llmpxy.proxy_client import (
    ProviderError,
    create_async_client,
    post_json,
    resolve_proxy,
    stream_lines,
)


@dataclass
class ProviderState:
    consecutive_errors: int = 0


class AllProvidersFailedError(Exception):
    def __init__(self, message: str, last_error: ProviderError | None = None) -> None:
        super().__init__(message)
        self.last_error = last_error


class ProviderDispatcher:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._provider_map = config.provider_map()
        self._group_map = config.provider_group_map()
        self._states = {provider.name: ProviderState() for provider in config.providers}
        self._group_offsets = {group.name: 0 for group in config.provider_groups}

    async def dispatch(
        self,
        request: CanonicalRequest,
        *,
        request_id: str,
    ) -> tuple[CanonicalResponse, ProviderConfig]:
        last_error: ProviderError | None = None
        for round_number in range(1, self._config.retry.max_rounds + 1):
            providers = self._resolve_route_provider_order()
            bind_logger(request_id=request_id, round_number=round_number).info(
                "starting provider round route_type={} route_name={} providers={}",
                self._config.route.type,
                self._config.route.name,
                [provider.name for provider in providers],
            )
            for provider in providers:
                result = await self._try_provider(
                    provider, request, request_id=request_id, round_number=round_number
                )
                if result is None:
                    state = self._states[provider.name]
                    if state.consecutive_errors > 0:
                        last_error = ProviderError(
                            f"Provider {provider.name} exhausted retry budget", retryable=True
                        )
                    continue
                return result, provider
            await self._backoff_if_needed(round_number, request_id)
        raise AllProvidersFailedError("all providers failed", last_error=last_error)

    async def dispatch_stream(
        self,
        request: CanonicalRequest,
        *,
        request_id: str,
    ) -> tuple[CanonicalResponse, list[CanonicalStreamEvent], ProviderConfig]:
        last_error: ProviderError | None = None
        for round_number in range(1, self._config.retry.max_rounds + 1):
            providers = self._resolve_route_provider_order()
            bind_logger(request_id=request_id, round_number=round_number).info(
                "starting provider streaming round route_type={} route_name={} providers={}",
                self._config.route.type,
                self._config.route.name,
                [provider.name for provider in providers],
            )
            for provider in providers:
                result = await self._try_provider_stream(
                    provider, request, request_id=request_id, round_number=round_number
                )
                if result is None:
                    state = self._states[provider.name]
                    if state.consecutive_errors > 0:
                        last_error = ProviderError(
                            f"Provider {provider.name} exhausted retry budget", retryable=True
                        )
                    continue
                response, events = result
                return response, events, provider
            await self._backoff_if_needed(round_number, request_id)
        raise AllProvidersFailedError("all providers failed", last_error=last_error)

    def _resolve_route_provider_order(self) -> list[ProviderConfig]:
        if self._config.route.type == "provider":
            return [self._provider_map[self._config.route.name]]
        return self._expand_group(self._group_map[self._config.route.name])

    def _expand_group(self, group: ProviderGroupConfig) -> list[ProviderConfig]:
        ordered_members = list(group.members)
        if group.strategy == "load_balance" and ordered_members:
            offset = self._group_offsets[group.name] % len(ordered_members)
            ordered_members = ordered_members[offset:] + ordered_members[:offset]
            self._group_offsets[group.name] = (offset + 1) % len(group.members)
        providers: list[ProviderConfig] = []
        for member in ordered_members:
            provider = self._provider_map.get(member)
            if provider is not None:
                providers.append(provider)
                continue
            providers.extend(self._expand_group(self._group_map[member]))
        return providers

    async def _try_provider(
        self,
        provider: ProviderConfig,
        request: CanonicalRequest,
        *,
        request_id: str,
        round_number: int,
    ) -> CanonicalResponse | None:
        state = self._states[provider.name]
        if not provider.supports_model(request.requested_model):
            bind_logger(
                request_id=request_id, provider=provider.name, round_number=round_number
            ).info(
                "provider skipped because requested model is not in whitelist requested_model={}",
                request.requested_model,
            )
            return None
        adapter = get_adapter(provider.protocol)
        mapped_model = provider.map_model(request.requested_model)
        path, payload = adapter.build_request(request, mapped_model)
        while state.consecutive_errors < self._config.retry.provider_error_threshold:
            log = bind_logger(
                request_id=request_id,
                provider=provider.name,
                round_number=round_number,
                attempt=state.consecutive_errors + 1,
            )
            log.debug(
                "sending request provider_protocol={} path={} mapped_model={} base_url={} proxy={}",
                provider.protocol,
                path,
                mapped_model,
                provider.base_url,
                resolve_proxy(self._config, provider),
            )
            log.debug(
                "request canonical summary={} upstream payload summary={}",
                sanitize_for_logging(
                    {
                        "protocol_in": request.protocol_in,
                        "requested_model": request.requested_model,
                        "reasoning": request.reasoning,
                        "message_count": len(request.messages),
                        "stream": request.stream,
                    }
                ),
                sanitize_for_logging(
                    {
                        "path": path,
                        "model": payload.get("model"),
                        "reasoning": payload.get("reasoning"),
                        "reasoning_effort": payload.get("reasoning_effort"),
                        "stream": payload.get("stream"),
                        "tool_choice": payload.get("tool_choice"),
                    }
                ),
            )
            try:
                async with create_async_client(self._config, provider) as client:
                    raw = await post_json(client, provider, path, payload)
                response = adapter.parse_response(raw, provider.protocol)
                reasoning = request.reasoning if isinstance(request.reasoning, dict) else None
                reasoning_summary = reasoning.get("summary") if reasoning is not None else None
                if isinstance(reasoning_summary, str):
                    response.metadata["reasoning_summary"] = reasoning_summary
            except ProviderError as exc:
                if not exc.retryable:
                    log.exception(
                        "fatal provider error status_code={} response_text={} base_url={} path={} proxy={} error_code={}",
                        exc.status_code,
                        exc.response_text,
                        exc.base_url,
                        exc.path,
                        exc.proxy,
                        exc.error_code,
                    )
                    raise
                state.consecutive_errors += 1
                log.exception(
                    "retryable provider error status_code={} consecutive_errors={} response_text={} base_url={} path={} proxy={} error_code={}",
                    exc.status_code,
                    state.consecutive_errors,
                    exc.response_text,
                    exc.base_url,
                    exc.path,
                    exc.proxy,
                    exc.error_code,
                )
                continue
            state.consecutive_errors = 0
            log.info("provider request succeeded")
            return response
        bind_logger(request_id=request_id, provider=provider.name, round_number=round_number).info(
            "provider reached error threshold, moving to next provider"
        )
        return None

    async def _try_provider_stream(
        self,
        provider: ProviderConfig,
        request: CanonicalRequest,
        *,
        request_id: str,
        round_number: int,
    ) -> tuple[CanonicalResponse, list[CanonicalStreamEvent]] | None:
        state = self._states[provider.name]
        if not provider.supports_model(request.requested_model):
            bind_logger(
                request_id=request_id, provider=provider.name, round_number=round_number
            ).info(
                "provider stream skipped because requested model is not in whitelist requested_model={}",
                request.requested_model,
            )
            return None
        adapter = get_adapter(provider.protocol)
        mapped_model = provider.map_model(request.requested_model)
        path, payload = adapter.build_request(request, mapped_model)
        while state.consecutive_errors < self._config.retry.provider_error_threshold:
            log = bind_logger(
                request_id=request_id,
                provider=provider.name,
                round_number=round_number,
                attempt=state.consecutive_errors + 1,
            )
            log.debug(
                "sending stream request provider_protocol={} path={} mapped_model={} base_url={} proxy={}",
                provider.protocol,
                path,
                mapped_model,
                provider.base_url,
                resolve_proxy(self._config, provider),
            )
            log.debug(
                "stream canonical summary={} upstream payload summary={}",
                sanitize_for_logging(
                    {
                        "protocol_in": request.protocol_in,
                        "requested_model": request.requested_model,
                        "reasoning": request.reasoning,
                        "message_count": len(request.messages),
                        "stream": request.stream,
                    }
                ),
                sanitize_for_logging(
                    {
                        "path": path,
                        "model": payload.get("model"),
                        "reasoning": payload.get("reasoning"),
                        "reasoning_effort": payload.get("reasoning_effort"),
                        "stream": payload.get("stream"),
                        "tool_choice": payload.get("tool_choice"),
                    }
                ),
            )
            try:
                async with create_async_client(self._config, provider) as client:
                    lines = stream_lines(client, provider, path, payload)
                    response, events = await adapter.parse_stream(lines, provider.protocol)
                reasoning = request.reasoning if isinstance(request.reasoning, dict) else None
                reasoning_summary = reasoning.get("summary") if reasoning is not None else None
                if isinstance(reasoning_summary, str):
                    response.metadata["reasoning_summary"] = reasoning_summary
            except ProviderError as exc:
                if not exc.retryable:
                    log.exception(
                        "fatal provider stream error status_code={} response_text={} base_url={} path={} proxy={} error_code={}",
                        exc.status_code,
                        exc.response_text,
                        exc.base_url,
                        exc.path,
                        exc.proxy,
                        exc.error_code,
                    )
                    raise
                state.consecutive_errors += 1
                log.exception(
                    "retryable provider stream error status_code={} consecutive_errors={} response_text={} base_url={} path={} proxy={} error_code={}",
                    exc.status_code,
                    state.consecutive_errors,
                    exc.response_text,
                    exc.base_url,
                    exc.path,
                    exc.proxy,
                    exc.error_code,
                )
                continue
            state.consecutive_errors = 0
            log.info("provider stream succeeded")
            return response, events
        bind_logger(request_id=request_id, provider=provider.name, round_number=round_number).info(
            "provider reached error threshold, moving to next provider"
        )
        return None

    async def _backoff_if_needed(self, round_number: int, request_id: str) -> None:
        if round_number >= self._config.retry.max_rounds:
            return
        delay = min(
            self._config.retry.base_backoff_seconds * (2 ** (round_number - 1)),
            self._config.retry.max_backoff_seconds,
        )
        bind_logger(request_id=request_id, round_number=round_number).error(
            "all providers failed in round, backing off {} seconds", delay
        )
        self._reset_thresholded_states()
        await asyncio.sleep(delay)

    def _reset_thresholded_states(self) -> None:
        for state in self._states.values():
            if state.consecutive_errors >= self._config.retry.provider_error_threshold:
                state.consecutive_errors = 0
