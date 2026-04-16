from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from json import JSONDecodeError
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.responses import Response

from llmpxy.auth import AuthenticatedApiKey
from llmpxy.admin_api import create_admin_router
from llmpxy.config import AppConfig, ProtocolName, ProviderConfig
from llmpxy.conversation import merge_history
from llmpxy.dispatcher import AllProvidersFailedError, ProviderDispatcher
from llmpxy.runtime import RuntimeManager
from llmpxy.logging_utils import bind_logger, sanitize_for_logging
from llmpxy.models import (
    CanonicalMessage,
    CanonicalResponse,
    CanonicalStreamEvent,
    RequestEventRecord,
    StoredConversation,
)
from llmpxy.protocols.registry import get_adapter
from llmpxy.proxy_client import ProviderError, create_async_client, open_stream, stream_lines
from llmpxy.protocols.oai_chat import (
    OpenAIChatAdapter,
    build_chat_stream_result,
    init_chat_stream_state,
    process_chat_stream_line,
)
from llmpxy.protocols.anthropic_messages import (
    build_anthropic_stream_result,
    init_anthropic_stream_state,
    process_anthropic_stream_line,
)
from llmpxy.protocols.oai_responses import (
    finalize_response_stream,
    init_response_stream_formatter_state,
    init_responses_stream_state,
    iter_response_stream_events,
    process_responses_stream_line,
    build_responses_stream_result,
)
from llmpxy.storage import ConversationStore


_BASE_URL_HINT = (
    "Oops! 这是base_url, 无法用浏览器访问哦, 如果你不知道怎么做的话...让codex/claude code教你吧!"
)


def create_app(
    config: AppConfig, store: ConversationStore, dispatcher: ProviderDispatcher
) -> FastAPI:
    app = FastAPI(title="llmpxy")

    @app.get("/")
    @app.get("/v1")
    async def index() -> PlainTextResponse:
        return PlainTextResponse(_BASE_URL_HINT)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/v1/responses")
    async def oai_responses(request: Request) -> Response:
        return await _handle_protocol_request(request, config, store, dispatcher, "oairesp")

    @app.post("/v1/chat/completions")
    async def oai_chat(request: Request) -> Response:
        return await _handle_protocol_request(request, config, store, dispatcher, "oaichat")

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:
        return await _handle_protocol_request(request, config, store, dispatcher, "anthropic")

    if os.getenv("LLMPXY_ENABLE_TEST_ENDPOINTS") == "1":
        app.state.test_upstream_last_payload = None

        @app.post("/_test/upstream/responses")
        async def test_upstream_responses(request: Request) -> dict[str, Any]:
            payload = await request.json()
            app.state.test_upstream_last_payload = payload
            return {
                "received": payload,
                "id": "resp_test",
                "object": "response",
                "created_at": int(time.time()),
                "status": "completed",
                "model": payload.get("model", "test-model"),
                "output": [
                    {
                        "id": "fc_test",
                        "type": "function_call",
                        "call_id": "fc_test",
                        "name": "glob",
                        "arguments": '{"pattern":"**/*.md"}',
                        "status": "completed",
                    },
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "ok",
                                "annotations": [],
                                "logprobs": [],
                            }
                        ],
                    },
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                "metadata": {},
            }

        @app.get("/_test/upstream/last")
        async def test_upstream_last() -> dict[str, Any]:
            return {"payload": app.state.test_upstream_last_payload}

    return app


def create_runtime_managed_app(runtime: RuntimeManager) -> FastAPI:
    app = FastAPI(title="llmpxy")
    app.include_router(create_admin_router(runtime, runtime.current().config_path))

    @app.get("/")
    @app.get("/v1")
    async def index() -> PlainTextResponse:
        return PlainTextResponse(_BASE_URL_HINT)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        runtime.current()
        return {"status": "ready"}

    @app.post("/v1/responses")
    async def oai_responses(request: Request) -> Response:
        return await _handle_runtime_protocol_request(request, runtime, "oairesp")

    @app.post("/v1/chat/completions")
    async def oai_chat(request: Request) -> Response:
        return await _handle_runtime_protocol_request(request, runtime, "oaichat")

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:
        return await _handle_runtime_protocol_request(request, runtime, "anthropic")

    if os.getenv("LLMPXY_ENABLE_TEST_ENDPOINTS") == "1":
        app.state.test_upstream_last_payload = None

        @app.post("/_test/upstream/responses")
        async def test_upstream_responses(request: Request) -> dict[str, Any]:
            payload = await request.json()
            app.state.test_upstream_last_payload = payload
            return {
                "received": payload,
                "id": "resp_test",
                "object": "response",
                "created_at": int(time.time()),
                "status": "completed",
                "model": payload.get("model", "test-model"),
                "output": [
                    {
                        "id": "fc_test",
                        "type": "function_call",
                        "call_id": "fc_test",
                        "name": "glob",
                        "arguments": '{"pattern":"**/*.md"}',
                        "status": "completed",
                    },
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "ok",
                                "annotations": [],
                                "logprobs": [],
                            }
                        ],
                    },
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                "metadata": {},
            }

        @app.get("/_test/upstream/last")
        async def test_upstream_last() -> dict[str, Any]:
            return {"payload": app.state.test_upstream_last_payload}

    return app


def _all_providers_failed_http_exception() -> HTTPException:
    return HTTPException(status_code=502, detail="all providers failed")


async def _handle_protocol_request(
    request: Request,
    config: AppConfig,
    store: ConversationStore,
    dispatcher: ProviderDispatcher,
    inbound_protocol: ProtocolName,
) -> Response:
    request_id = str(uuid.uuid4())
    started_at = int(time.time())
    started_perf = time.perf_counter()
    response: CanonicalResponse | None = None
    payload = await _parse_json_payload(request)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")

    adapter = get_adapter(inbound_protocol)
    log = bind_logger(request_id=request_id)
    log.info("received protocol request protocol={}", inbound_protocol)
    log.debug("request payload summary={}", sanitize_for_logging(payload))

    canonical_request = adapter.parse_request(
        payload, strip_unsupported_fields=config.proxy.strip_unsupported_fields
    )
    if not dispatcher.route_supports_model(canonical_request.requested_model):
        raise HTTPException(status_code=400, detail="Requested model is not allowed by route")
    history = _load_history(canonical_request.conversation_reference, store, inbound_protocol)
    merged_messages = merge_history(history, canonical_request)
    canonical_request.messages = merged_messages
    log.debug(
        "normalized request summary={}",
        sanitize_for_logging(
            {
                "protocol": inbound_protocol,
                "requested_model": canonical_request.requested_model,
                "stream": canonical_request.stream,
                "message_count": len(canonical_request.messages),
                "metadata": canonical_request.metadata,
                "has_previous_response_id": canonical_request.conversation_reference is not None,
            }
        ),
    )

    try:
        if canonical_request.stream:
            route_providers = _prioritize_protocol_compatible_providers(
                inbound_protocol, dispatcher.resolve_route_provider_order()
            )
            if _can_bridge_live_stream(inbound_protocol, route_providers):
                return StreamingResponse(
                    await _prepare_live_stream_bridge(
                        inbound_protocol=inbound_protocol,
                        request_id=request_id,
                        canonical_request=canonical_request,
                        providers=route_providers,
                        adapter=adapter,
                        runtime=None,
                        api_key=None,
                        store=store,
                        config=config,
                        started_at=started_at,
                        started_perf=started_perf,
                    ),
                    media_type=_stream_media_type(inbound_protocol),
                    headers=_stream_headers(),
                )
            if _can_passthrough_live_stream(inbound_protocol, route_providers):
                return StreamingResponse(
                    await _prepare_live_stream_passthrough(
                        inbound_protocol=inbound_protocol,
                        request_id=request_id,
                        canonical_request=canonical_request,
                        providers=route_providers,
                        adapter=adapter,
                        runtime=None,
                        api_key=None,
                        store=store,
                        config=config,
                        started_at=started_at,
                        started_perf=started_perf,
                    ),
                    media_type=_stream_media_type(inbound_protocol),
                    headers=_stream_headers(),
                )
            response, events, provider = await dispatcher.dispatch_stream(
                canonical_request,
                request_id=request_id,
                providers=route_providers,
            )
            if inbound_protocol == "oairesp":
                _save_conversation(
                    store,
                    adapter.format_response(response),
                    response.model,
                    response.output_messages,
                    canonical_request.metadata,
                    config.storage.ttl_seconds,
                )
            log.info(
                "stream request routed provider={} upstream_protocol={} mapped_model={}",
                provider.name,
                provider.protocol,
                response.model,
            )
            return StreamingResponse(
                adapter.format_stream(response, events),
                media_type=_stream_media_type(inbound_protocol),
                headers=_stream_headers(),
            )

        response, provider = await dispatcher.dispatch(
            canonical_request,
            request_id=request_id,
            providers=_prioritize_protocol_compatible_providers(
                inbound_protocol, dispatcher.resolve_route_provider_order()
            ),
        )
    except ProviderError as exc:
        log.exception(
            "provider fatal error status_code={} response_text={}",
            exc.status_code,
            exc.response_text,
        )
        raise _all_providers_failed_http_exception() from exc
    except AllProvidersFailedError as exc:
        log.exception("all providers failed")
        raise _all_providers_failed_http_exception() from exc

    formatted = adapter.format_response(response)
    if inbound_protocol == "oairesp":
        _save_conversation(
            store,
            formatted,
            response.model,
            response.output_messages,
            canonical_request.metadata,
            config.storage.ttl_seconds,
        )
    log.debug("response payload summary={}", sanitize_for_logging(formatted))
    log.info(
        "request completed provider={} upstream_protocol={} mapped_model={}",
        provider.name,
        provider.protocol,
        response.model,
    )
    return JSONResponse(formatted)


async def _handle_runtime_protocol_request(
    request: Request,
    runtime: RuntimeManager,
    inbound_protocol: ProtocolName,
) -> Response:
    state = runtime.current()
    api_key = runtime.authenticate(
        request.headers.get("Authorization"), request.headers.get("x-api-key")
    )
    provider_order = state.dispatcher.resolve_route_provider_order()
    selectable_providers = runtime.selectable_provider_configs(api_key, provider_order)
    if not selectable_providers:
        raise HTTPException(
            status_code=429, detail="API key budget exceeded for all routed providers"
        )
    return await _handle_protocol_request_with_api_key(
        request,
        state.config,
        state.store,
        state.dispatcher,
        inbound_protocol,
        runtime,
        api_key,
        selectable_providers,
    )


async def _handle_protocol_request_with_api_key(
    request: Request,
    config: AppConfig,
    store: ConversationStore,
    dispatcher: ProviderDispatcher,
    inbound_protocol: ProtocolName,
    runtime: RuntimeManager,
    api_key: AuthenticatedApiKey,
    selectable_providers: list[ProviderConfig],
) -> Response:
    request_id = str(uuid.uuid4())
    started_at = int(time.time())
    started_perf = time.perf_counter()
    payload = await _parse_json_payload(request)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")

    adapter = get_adapter(inbound_protocol)
    log = bind_logger(request_id=request_id)
    log.info("received protocol request protocol={} api_key={}", inbound_protocol, api_key.name)
    log.debug("request payload summary={}", sanitize_for_logging(payload))

    canonical_request = adapter.parse_request(
        payload, strip_unsupported_fields=config.proxy.strip_unsupported_fields
    )
    if not dispatcher.route_supports_model(canonical_request.requested_model):
        raise HTTPException(status_code=400, detail="Requested model is not allowed by route")
    history = _load_history(canonical_request.conversation_reference, store, inbound_protocol)
    merged_messages = merge_history(history, canonical_request)
    canonical_request.messages = merged_messages

    try:
        prioritized_providers = _prioritize_protocol_compatible_providers(
            inbound_protocol, selectable_providers
        )
        if canonical_request.stream:
            if _can_bridge_live_stream(inbound_protocol, prioritized_providers):
                return StreamingResponse(
                    await _prepare_live_stream_bridge(
                        inbound_protocol=inbound_protocol,
                        request_id=request_id,
                        canonical_request=canonical_request,
                        providers=prioritized_providers,
                        adapter=adapter,
                        runtime=runtime,
                        api_key=api_key,
                        store=store,
                        config=config,
                        started_at=started_at,
                        started_perf=started_perf,
                    ),
                    media_type=_stream_media_type(inbound_protocol),
                    headers=_stream_headers(),
                )
            if _can_passthrough_live_stream(inbound_protocol, prioritized_providers):
                return StreamingResponse(
                    await _prepare_live_stream_passthrough(
                        inbound_protocol=inbound_protocol,
                        request_id=request_id,
                        canonical_request=canonical_request,
                        providers=prioritized_providers,
                        adapter=adapter,
                        runtime=runtime,
                        api_key=api_key,
                        store=store,
                        config=config,
                        started_at=started_at,
                        started_perf=started_perf,
                    ),
                    media_type=_stream_media_type(inbound_protocol),
                    headers=_stream_headers(),
                )
            response, events, provider = await dispatcher.dispatch_stream(
                canonical_request,
                request_id=request_id,
                providers=prioritized_providers,
            )
            with runtime.budget_guard(api_key, provider.name):
                if inbound_protocol == "oairesp":
                    _save_conversation(
                        store,
                        adapter.format_response(response),
                        response.model,
                        response.output_messages,
                        canonical_request.metadata,
                        config.storage.ttl_seconds,
                    )
                runtime.record_usage(
                    api_key=api_key,
                    request_id=request_id,
                    started_at=started_at,
                    request=canonical_request,
                    response=response,
                    provider_name=provider.name,
                    latency_ms=int((time.perf_counter() - started_perf) * 1000),
                )
            await runtime.publish_request_event(
                {
                    "request_id": request_id,
                    "api_key": api_key.name,
                    "provider": provider.name,
                    "model": canonical_request.requested_model,
                    "status": "success",
                }
            )
            return StreamingResponse(
                adapter.format_stream(response, events),
                media_type=_stream_media_type(inbound_protocol),
                headers=_stream_headers(),
            )

        response, provider = await dispatcher.dispatch(
            canonical_request,
            request_id=request_id,
            providers=prioritized_providers,
        )
        with runtime.budget_guard(api_key, provider.name):
            runtime.record_usage(
                api_key=api_key,
                request_id=request_id,
                started_at=started_at,
                request=canonical_request,
                response=response,
                provider_name=provider.name,
                latency_ms=int((time.perf_counter() - started_perf) * 1000),
            )
        await runtime.publish_request_event(
            {
                "request_id": request_id,
                "api_key": api_key.name,
                "provider": provider.name,
                "model": canonical_request.requested_model,
                "status": "success",
                "duration_seconds": int(time.time()) - started_at,
            }
        )
    except ProviderError as exc:
        latency_ms = int((time.perf_counter() - started_perf) * 1000)
        runtime.current().store.put_request_event(
            RequestEventRecord(
                request_id=request_id,
                started_at=started_at,
                finished_at=int(time.time()),
                latency_ms=latency_ms,
                protocol_in=inbound_protocol,
                stream=canonical_request.stream,
                api_key_uuid=api_key.uuid,
                api_key_name=api_key.name,
                provider_name=exc.provider_name,
                requested_model=canonical_request.requested_model,
                status="provider_error",
                http_status=exc.status_code or 502,
                error_code=exc.error_code,
                error_message=str(exc),
            )
        )
        await runtime.publish_request_event(
            {
                "request_id": request_id,
                "api_key": api_key.name,
                "status": "provider_error",
                "error": str(exc),
            }
        )
        log.exception(
            "provider fatal error status_code={} response_text={}",
            exc.status_code,
            exc.response_text,
        )
        raise _all_providers_failed_http_exception() from exc
    except AllProvidersFailedError as exc:
        latency_ms = int((time.perf_counter() - started_perf) * 1000)
        runtime.current().store.put_request_event(
            RequestEventRecord(
                request_id=request_id,
                started_at=started_at,
                finished_at=int(time.time()),
                latency_ms=latency_ms,
                protocol_in=inbound_protocol,
                stream=canonical_request.stream,
                api_key_uuid=api_key.uuid,
                api_key_name=api_key.name,
                provider_name=exc.last_error.provider_name if exc.last_error is not None else None,
                requested_model=canonical_request.requested_model,
                status="all_failed",
                http_status=502,
                error_message=str(exc),
            )
        )
        await runtime.publish_request_event(
            {
                "request_id": request_id,
                "api_key": api_key.name,
                "status": "all_failed",
                "error": str(exc),
            }
        )
        log.exception("all providers failed")
        raise _all_providers_failed_http_exception() from exc

    if response is None:
        raise RuntimeError("dispatcher returned no response")

    formatted = adapter.format_response(response)
    if inbound_protocol == "oairesp":
        _save_conversation(
            store,
            formatted,
            response.model,
            response.output_messages,
            canonical_request.metadata,
            config.storage.ttl_seconds,
        )
    return JSONResponse(formatted)


async def _parse_json_payload(request: Request) -> Any:
    try:
        return await request.json()
    except JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc


async def _read_first_stream_line(
    line_iterator: AsyncIterator[str], provider: ProviderConfig
) -> str:
    try:
        async with asyncio.timeout(provider.timeout_seconds):
            async for line in line_iterator:
                if line:
                    return line
    except TimeoutError as exc:
        raise ProviderError(
            f"Provider {provider.name} timed out waiting for first token",
            provider_name=provider.name,
            retryable=True,
            base_url=provider.base_url,
        ) from exc
    return ""


def _load_history(
    response_id: str | None,
    store: ConversationStore,
    inbound_protocol: ProtocolName,
) -> StoredConversation | None:
    if response_id is None:
        return None
    if inbound_protocol != "oairesp":
        raise HTTPException(
            status_code=400, detail="previous_response_id is only supported for oairesp"
        )
    history = store.get(response_id)
    if history is None:
        raise HTTPException(status_code=404, detail="previous_response_id not found")
    return history


def _save_conversation(
    store: ConversationStore,
    response_payload: dict[str, Any],
    model: str,
    messages: list[CanonicalMessage],
    metadata: dict[str, Any],
    ttl_seconds: int,
) -> None:
    created_at = int(response_payload.get("created_at", int(time.time())))
    expires_at = int(time.time()) + ttl_seconds if ttl_seconds > 0 else None
    response_id = response_payload.get("id")
    if not isinstance(response_id, str):
        return
    conversation = StoredConversation(
        response_id=response_id,
        created_at=created_at,
        model=model,
        messages=[message.model_dump(mode="json") for message in messages],
        response_payload=response_payload,
        metadata=metadata,
        expires_at=expires_at,
    )
    store.put(conversation)


def _stream_media_type(inbound_protocol: ProtocolName) -> str:
    if inbound_protocol in {"oairesp", "oaichat", "anthropic"}:
        return "text/event-stream"
    return "application/json"


def _stream_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }


def _record_live_provider_attempt(runtime: RuntimeManager | None, provider_name: str) -> None:
    if runtime is None:
        return
    runtime.stats().record_provider_attempt(provider_name)


def _record_live_provider_success(runtime: RuntimeManager | None, provider_name: str) -> None:
    if runtime is None:
        return
    runtime.stats().record_provider_success(provider_name)


def _record_live_provider_error(
    runtime: RuntimeManager | None, provider_name: str, message: str
) -> None:
    if runtime is None:
        return
    runtime.stats().record_provider_error(provider_name, message)


def _can_passthrough_live_stream(
    inbound_protocol: ProtocolName, providers: list[ProviderConfig]
) -> bool:
    if inbound_protocol not in {"oaichat", "anthropic"}:
        return False
    if not providers:
        return False
    return all(provider.protocol == inbound_protocol for provider in providers)


def _can_bridge_live_stream(
    inbound_protocol: ProtocolName, providers: list[ProviderConfig]
) -> bool:
    if not providers:
        return False
    if not all(provider.protocol == providers[0].protocol for provider in providers):
        return False
    if inbound_protocol == "oairesp":
        return providers[0].protocol in {"oaichat", "anthropic"}
    if inbound_protocol == "oaichat":
        return providers[0].protocol in {"anthropic", "oairesp"}
    if inbound_protocol == "anthropic":
        return providers[0].protocol in {"oaichat", "oairesp"}
    return False


def _provider_protocol_priority(
    inbound_protocol: ProtocolName, provider_protocol: ProtocolName
) -> int:
    if provider_protocol == inbound_protocol:
        return 0
    if inbound_protocol == "oairesp" and provider_protocol in {"oaichat", "anthropic"}:
        return 1
    if inbound_protocol == "oaichat" and provider_protocol in {"anthropic", "oairesp"}:
        return 1
    if inbound_protocol == "anthropic" and provider_protocol in {"oaichat", "oairesp"}:
        return 1
    return 2


def _prioritize_protocol_compatible_providers(
    inbound_protocol: ProtocolName, providers: list[ProviderConfig]
) -> list[ProviderConfig]:
    return sorted(
        providers,
        key=lambda provider: _provider_protocol_priority(inbound_protocol, provider.protocol),
    )


async def _live_stream_passthrough(
    *,
    inbound_protocol: ProtocolName,
    request_id: str,
    canonical_request: Any,
    provider: ProviderConfig,
    adapter: Any,
    runtime: RuntimeManager | None,
    api_key: AuthenticatedApiKey | None,
    store: ConversationStore,
    config: AppConfig,
    started_at: int,
    started_perf: float,
    first_line: str,
    client: Any,
    response: Any,
    line_iterator: AsyncIterator[str],
) -> AsyncIterator[str]:
    buffered_lines: list[str] = []
    try:
        buffered_lines.append(first_line)
        yield f"{first_line}\n\n"
        async for line in line_iterator:
            if line:
                buffered_lines.append(line)
                yield f"{line}\n\n"
    finally:
        await response.aclose()
        await client.aclose()

    outbound_adapter = get_adapter(provider.protocol)

    async def replay_lines() -> AsyncIterator[str]:
        for line in buffered_lines:
            yield line

    response, _events = await outbound_adapter.parse_stream(replay_lines(), provider.protocol)
    _record_live_provider_success(runtime, provider.name)
    if runtime is not None and api_key is not None:
        runtime.record_usage(
            api_key=api_key,
            request_id=request_id,
            started_at=started_at,
            request=canonical_request,
            response=response,
            provider_name=provider.name,
            latency_ms=int((time.perf_counter() - started_perf) * 1000),
        )
        await runtime.publish_request_event(
            {
                "request_id": request_id,
                "api_key": api_key.name,
                "provider": provider.name,
                "model": canonical_request.requested_model,
                "status": "success",
            }
        )
    if inbound_protocol == "oairesp":
        _save_conversation(
            store,
            adapter.format_response(response),
            response.model,
            response.output_messages,
            canonical_request.metadata,
            config.storage.ttl_seconds,
        )


async def _prepare_live_stream_passthrough(
    *,
    inbound_protocol: ProtocolName,
    request_id: str,
    canonical_request: Any,
    providers: list[ProviderConfig],
    adapter: Any,
    runtime: RuntimeManager | None,
    api_key: AuthenticatedApiKey | None,
    store: ConversationStore,
    config: AppConfig,
    started_at: int,
    started_perf: float,
) -> AsyncIterator[str]:
    last_error: ProviderError | None = None
    for provider in providers:
        _record_live_provider_attempt(runtime, provider.name)
        outbound_adapter = get_adapter(provider.protocol)
        mapped_model = provider.map_model(canonical_request.requested_model)
        path, payload = outbound_adapter.build_request(canonical_request, mapped_model)
        client = create_async_client(config, provider)
        try:
            response = await open_stream(client, provider, path, payload)
            line_iterator = response.aiter_lines()
            first_line = await _read_first_stream_line(line_iterator, provider)
            if not first_line:
                await response.aclose()
                await client.aclose()
                raise ProviderError(
                    f"Provider {provider.name} returned an empty stream",
                    provider_name=provider.name,
                    retryable=False,
                    base_url=provider.base_url,
                    path=path,
                )
            return _live_stream_passthrough(
                inbound_protocol=inbound_protocol,
                request_id=request_id,
                canonical_request=canonical_request,
                provider=provider,
                adapter=adapter,
                runtime=runtime,
                api_key=api_key,
                store=store,
                config=config,
                started_at=started_at,
                started_perf=started_perf,
                first_line=first_line,
                client=client,
                response=response,
                line_iterator=line_iterator,
            )
        except ProviderError as exc:
            last_error = exc
            _record_live_provider_error(runtime, provider.name, str(exc))
            await client.aclose()
            continue
        except Exception:
            await client.aclose()
            raise
    if last_error is not None:
        raise AllProvidersFailedError("all providers failed", last_error=last_error)
    raise AllProvidersFailedError("all providers failed")


async def _live_stream_bridge_oaichat_to_oairesp(
    *,
    request_id: str,
    canonical_request: Any,
    provider: ProviderConfig,
    adapter: Any,
    runtime: RuntimeManager | None,
    api_key: AuthenticatedApiKey | None,
    store: ConversationStore,
    config: AppConfig,
    started_at: int,
    started_perf: float,
    request_lock: Any,
    first_line: str,
    client: Any,
    response: Any,
    line_iterator: AsyncIterator[str],
) -> AsyncIterator[str]:
    chat_state = init_chat_stream_state(provider.protocol)
    first_events = process_chat_stream_line(chat_state, first_line)
    formatter_state = init_response_stream_formatter_state(
        CanonicalResponse(
            response_id=chat_state.response_id,
            protocol_out="oairesp",
            model=chat_state.model,
            created_at=int(time.time()),
        )
    )
    try:
        for chunk in iter_response_stream_events(formatter_state, first_events):
            yield chunk
        async for line in line_iterator:
            if not line:
                continue
            events = process_chat_stream_line(chat_state, line)
            formatter_state.response.model = chat_state.model or formatter_state.response.model
            for chunk in iter_response_stream_events(formatter_state, events):
                yield chunk
    finally:
        await response.aclose()
        await client.aclose()

    final_response, _final_events = build_chat_stream_result(chat_state)
    final_response.metadata["_stream_events"] = list(chat_state.events)
    final_response.metadata.update(canonical_request.metadata)
    if final_response.output_messages and final_response.output_messages[0].reasoning_content:
        final_response.metadata["reasoning_summary"] = final_response.output_messages[
            0
        ].reasoning_content
    formatter_state.response = CanonicalResponse(
        response_id=final_response.response_id,
        protocol_out="oairesp",
        model=final_response.model,
        created_at=final_response.created_at,
        output_messages=final_response.output_messages,
        usage=final_response.usage,
        metadata=final_response.metadata,
    )
    for chunk in finalize_response_stream(formatter_state):
        yield chunk
    _record_live_provider_success(runtime, provider.name)
    if runtime is not None and api_key is not None:
        runtime.record_usage(
            api_key=api_key,
            request_id=request_id,
            started_at=started_at,
            request=canonical_request,
            response=final_response,
            provider_name=provider.name,
            latency_ms=int((time.perf_counter() - started_perf) * 1000),
        )
        await runtime.publish_request_event(
            {
                "request_id": request_id,
                "api_key": api_key.name,
                "provider": provider.name,
                "model": canonical_request.requested_model,
                "status": "success",
            }
        )
    _save_conversation(
        store,
        adapter.format_response(final_response),
        final_response.model,
        final_response.output_messages,
        canonical_request.metadata,
        config.storage.ttl_seconds,
    )
    _release_request_lock(request_lock)


async def _live_stream_bridge_anthropic_to_oairesp(
    *,
    request_id: str,
    canonical_request: Any,
    provider: ProviderConfig,
    adapter: Any,
    runtime: RuntimeManager | None,
    api_key: AuthenticatedApiKey | None,
    store: ConversationStore,
    config: AppConfig,
    started_at: int,
    started_perf: float,
    request_lock: Any,
    first_line: str,
    client: Any,
    response: Any,
    line_iterator: AsyncIterator[str],
) -> AsyncIterator[str]:
    anthropic_state = init_anthropic_stream_state(provider.protocol)
    first_events = process_anthropic_stream_line(anthropic_state, first_line)
    formatter_state = init_response_stream_formatter_state(
        CanonicalResponse(
            response_id=anthropic_state.response_id,
            protocol_out="oairesp",
            model=anthropic_state.model,
            created_at=int(time.time()),
        )
    )
    try:
        for chunk in iter_response_stream_events(formatter_state, first_events):
            yield chunk
        async for line in line_iterator:
            if not line:
                continue
            events = process_anthropic_stream_line(anthropic_state, line)
            formatter_state.response.model = anthropic_state.model or formatter_state.response.model
            for chunk in iter_response_stream_events(formatter_state, events):
                yield chunk
    finally:
        await response.aclose()
        await client.aclose()

    final_response, _final_events = build_anthropic_stream_result(anthropic_state)
    formatter_state.response = CanonicalResponse(
        response_id=final_response.response_id,
        protocol_out="oairesp",
        model=final_response.model,
        created_at=final_response.created_at,
        output_messages=final_response.output_messages,
        usage=final_response.usage,
        metadata=final_response.metadata,
    )
    for chunk in finalize_response_stream(formatter_state):
        yield chunk
    _record_live_provider_success(runtime, provider.name)
    if runtime is not None and api_key is not None:
        runtime.record_usage(
            api_key=api_key,
            request_id=request_id,
            started_at=started_at,
            request=canonical_request,
            response=final_response,
            provider_name=provider.name,
            latency_ms=int((time.perf_counter() - started_perf) * 1000),
        )
        await runtime.publish_request_event(
            {
                "request_id": request_id,
                "api_key": api_key.name,
                "provider": provider.name,
                "model": canonical_request.requested_model,
                "status": "success",
            }
        )
    _save_conversation(
        store,
        adapter.format_response(final_response),
        final_response.model,
        final_response.output_messages,
        canonical_request.metadata,
        config.storage.ttl_seconds,
    )
    _release_request_lock(request_lock)


async def _live_stream_bridge_anthropic_to_oaichat(
    *,
    request_id: str,
    canonical_request: Any,
    provider: ProviderConfig,
    adapter: Any,
    runtime: RuntimeManager | None,
    api_key: AuthenticatedApiKey | None,
    store: ConversationStore,
    config: AppConfig,
    started_at: int,
    started_perf: float,
    request_lock: Any,
    first_line: str,
    client: Any,
    response: Any,
    line_iterator: AsyncIterator[str],
) -> AsyncIterator[str]:
    anthropic_state = init_anthropic_stream_state(provider.protocol)
    first_events = process_anthropic_stream_line(anthropic_state, first_line)
    try:
        response_id = anthropic_state.response_id
        created_at = int(time.time())
        model = anthropic_state.model
        tool_indices: dict[str, int] = {}
        for event in first_events:
            if event.event_type == "text_delta" and event.delta is not None:
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_at,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"content": event.delta}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            elif event.event_type == "function_call_arguments_delta" and event.item_id is not None:
                if event.item_id not in tool_indices:
                    tool_indices[event.item_id] = len(tool_indices)
                tool_call = next(
                    (
                        call
                        for call in anthropic_state.tool_calls_in_order
                        if call.id == event.item_id
                    ),
                    None,
                )
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_at,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": tool_indices[event.item_id],
                                        "id": event.item_id,
                                        "type": "function",
                                        "function": {
                                            "name": _safe_tool_name(
                                                tool_call.name if tool_call is not None else None
                                            ),
                                            "arguments": event.delta,
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        async for line in line_iterator:
            if not line:
                continue
            events = process_anthropic_stream_line(anthropic_state, line)
            model = anthropic_state.model or model
            for event in events:
                if event.event_type == "text_delta" and event.delta is not None:
                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created_at,
                        "model": model,
                        "choices": [
                            {"index": 0, "delta": {"content": event.delta}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif (
                    event.event_type == "function_call_arguments_delta"
                    and event.item_id is not None
                ):
                    if event.item_id not in tool_indices:
                        tool_indices[event.item_id] = len(tool_indices)
                    tool_call = next(
                        (
                            call
                            for call in anthropic_state.tool_calls_in_order
                            if call.id == event.item_id
                        ),
                        None,
                    )
                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created_at,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": tool_indices[event.item_id],
                                            "id": event.item_id,
                                            "type": "function",
                                            "function": {
                                                "name": _safe_tool_name(
                                                    tool_call.name
                                                    if tool_call is not None
                                                    else None
                                                ),
                                                "arguments": event.delta,
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
    finally:
        await response.aclose()
        await client.aclose()

    final_response, final_events = build_anthropic_stream_result(anthropic_state)
    final_response.protocol_out = "oaichat"
    final_chunk = {
        "id": final_response.response_id,
        "object": "chat.completion.chunk",
        "created": final_response.created_at,
        "model": final_response.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": final_response.usage.input_tokens,
            "completion_tokens": final_response.usage.output_tokens,
            "total_tokens": final_response.usage.total_tokens,
        },
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
    _record_live_provider_success(runtime, provider.name)
    if runtime is not None and api_key is not None:
        runtime.record_usage(
            api_key=api_key,
            request_id=request_id,
            started_at=started_at,
            request=canonical_request,
            response=final_response,
            provider_name=provider.name,
            latency_ms=int((time.perf_counter() - started_perf) * 1000),
        )
        await runtime.publish_request_event(
            {
                "request_id": request_id,
                "api_key": api_key.name,
                "provider": provider.name,
                "model": canonical_request.requested_model,
                "status": "success",
            }
        )
    _release_request_lock(request_lock)


async def _live_stream_bridge_oairesp_to_oaichat(
    *,
    request_id: str,
    canonical_request: Any,
    provider: ProviderConfig,
    adapter: Any,
    runtime: RuntimeManager | None,
    api_key: AuthenticatedApiKey | None,
    store: ConversationStore,
    config: AppConfig,
    started_at: int,
    started_perf: float,
    request_lock: Any,
    first_line: str,
    client: Any,
    response: Any,
    line_iterator: AsyncIterator[str],
) -> AsyncIterator[str]:
    responses_state = init_responses_stream_state(provider.protocol)
    first_events = await process_responses_stream_line(
        responses_state, first_line, OpenAIChatAdapter(), provider.protocol
    )
    try:
        response_id = responses_state.response_id
        created_at = int(time.time())
        model = responses_state.model
        tool_indices: dict[str, int] = {}
        for event in first_events:
            if event.event_type == "text_delta" and event.delta is not None:
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_at,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"content": event.delta}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            elif event.event_type == "function_call_arguments_delta" and event.item_id is not None:
                if event.item_id not in tool_indices:
                    tool_indices[event.item_id] = len(tool_indices)
                tool_call = next(
                    (
                        call
                        for call in responses_state.tool_calls_in_order
                        if call.id == event.item_id
                    ),
                    None,
                )
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_at,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": tool_indices[event.item_id],
                                        "id": event.item_id,
                                        "type": "function",
                                        "function": {
                                            "name": _safe_tool_name(
                                                tool_call.name if tool_call is not None else None
                                            ),
                                            "arguments": event.delta,
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        async for line in line_iterator:
            if not line:
                continue
            events = await process_responses_stream_line(
                responses_state, line, OpenAIChatAdapter(), provider.protocol
            )
            model = responses_state.model or model
            for event in events:
                if event.event_type == "text_delta" and event.delta is not None:
                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created_at,
                        "model": model,
                        "choices": [
                            {"index": 0, "delta": {"content": event.delta}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif (
                    event.event_type == "function_call_arguments_delta"
                    and event.item_id is not None
                ):
                    if event.item_id not in tool_indices:
                        tool_indices[event.item_id] = len(tool_indices)
                    tool_call = next(
                        (
                            call
                            for call in responses_state.tool_calls_in_order
                            if call.id == event.item_id
                        ),
                        None,
                    )
                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created_at,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": tool_indices[event.item_id],
                                            "id": event.item_id,
                                            "type": "function",
                                            "function": {
                                                "name": _safe_tool_name(
                                                    tool_call.name
                                                    if tool_call is not None
                                                    else None
                                                ),
                                                "arguments": event.delta,
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
    finally:
        await response.aclose()
        await client.aclose()

    final_response, _final_events = build_responses_stream_result(responses_state)
    final_response.protocol_out = "oaichat"
    final_chunk = {
        "id": final_response.response_id,
        "object": "chat.completion.chunk",
        "created": final_response.created_at,
        "model": final_response.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": final_response.usage.input_tokens,
            "completion_tokens": final_response.usage.output_tokens,
            "total_tokens": final_response.usage.total_tokens,
        },
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
    _record_live_provider_success(runtime, provider.name)
    if runtime is not None and api_key is not None:
        runtime.record_usage(
            api_key=api_key,
            request_id=request_id,
            started_at=started_at,
            request=canonical_request,
            response=final_response,
            provider_name=provider.name,
            latency_ms=int((time.perf_counter() - started_perf) * 1000),
        )
        await runtime.publish_request_event(
            {
                "request_id": request_id,
                "api_key": api_key.name,
                "provider": provider.name,
                "model": canonical_request.requested_model,
                "status": "success",
            }
        )
    _release_request_lock(request_lock)


async def _live_stream_bridge_oaichat_to_anthropic(
    *,
    request_id: str,
    canonical_request: Any,
    provider: ProviderConfig,
    adapter: Any,
    runtime: RuntimeManager | None,
    api_key: AuthenticatedApiKey | None,
    store: ConversationStore,
    config: AppConfig,
    started_at: int,
    started_perf: float,
    request_lock: Any,
    first_line: str,
    client: Any,
    response: Any,
    line_iterator: AsyncIterator[str],
) -> AsyncIterator[str]:
    chat_state = init_chat_stream_state(provider.protocol)
    first_events = process_chat_stream_line(chat_state, first_line)
    response_id = chat_state.response_id
    model = chat_state.model
    tool_indices: dict[str, int] = {}
    try:
        start_event = {
            "type": "message_start",
            "message": {
                "id": response_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(start_event)}\n\n"
        yield (
            "event: content_block_start\n"
            f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        )
        for event in first_events:
            if event.event_type == "text_delta" and event.delta is not None:
                yield (
                    "event: content_block_delta\n"
                    f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': event.delta}})}\n\n"
                )
            elif event.event_type == "function_call_arguments_delta" and event.item_id is not None:
                if event.item_id not in tool_indices:
                    tool_indices[event.item_id] = len(tool_indices) + 1
                    tool_call = next(
                        (call for call in chat_state.tool_calls if call.id == event.item_id),
                        None,
                    )
                    yield (
                        "event: content_block_start\n"
                        f"data: {json.dumps({'type': 'content_block_start', 'index': tool_indices[event.item_id], 'content_block': {'type': 'tool_use', 'id': event.item_id, 'name': _safe_tool_name(tool_call.name if tool_call is not None else None), 'input': {}}})}\n\n"
                    )
                yield (
                    "event: content_block_delta\n"
                    f"data: {json.dumps({'type': 'content_block_delta', 'index': tool_indices[event.item_id], 'delta': {'type': 'input_json_delta', 'partial_json': event.delta}})}\n\n"
                )
        async for line in line_iterator:
            if not line:
                continue
            events = process_chat_stream_line(chat_state, line)
            model = chat_state.model or model
            for event in events:
                if event.event_type == "text_delta" and event.delta is not None:
                    yield (
                        "event: content_block_delta\n"
                        f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': event.delta}})}\n\n"
                    )
                elif (
                    event.event_type == "function_call_arguments_delta"
                    and event.item_id is not None
                ):
                    if event.item_id not in tool_indices:
                        tool_indices[event.item_id] = len(tool_indices) + 1
                        tool_call = next(
                            (call for call in chat_state.tool_calls if call.id == event.item_id),
                            None,
                        )
                        yield (
                            "event: content_block_start\n"
                            f"data: {json.dumps({'type': 'content_block_start', 'index': tool_indices[event.item_id], 'content_block': {'type': 'tool_use', 'id': event.item_id, 'name': _safe_tool_name(tool_call.name if tool_call is not None else None), 'input': {}}})}\n\n"
                        )
                    yield (
                        "event: content_block_delta\n"
                        f"data: {json.dumps({'type': 'content_block_delta', 'index': tool_indices[event.item_id], 'delta': {'type': 'input_json_delta', 'partial_json': event.delta}})}\n\n"
                    )
    finally:
        await response.aclose()
        await client.aclose()

    final_response, _final_events = build_chat_stream_result(chat_state)
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
    for index in tool_indices.values():
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': index})}\n\n"
    yield (
        "event: message_delta\n"
        f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'input_tokens': final_response.usage.input_tokens, 'output_tokens': final_response.usage.output_tokens}})}\n\n"
    )
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    _record_live_provider_success(runtime, provider.name)
    if runtime is not None and api_key is not None:
        runtime.record_usage(
            api_key=api_key,
            request_id=request_id,
            started_at=started_at,
            request=canonical_request,
            response=final_response,
            provider_name=provider.name,
            latency_ms=int((time.perf_counter() - started_perf) * 1000),
        )
        await runtime.publish_request_event(
            {
                "request_id": request_id,
                "api_key": api_key.name,
                "provider": provider.name,
                "model": canonical_request.requested_model,
                "status": "success",
            }
        )
    _release_request_lock(request_lock)


async def _live_stream_bridge_oairesp_to_anthropic(
    *,
    request_id: str,
    canonical_request: Any,
    provider: ProviderConfig,
    adapter: Any,
    runtime: RuntimeManager | None,
    api_key: AuthenticatedApiKey | None,
    store: ConversationStore,
    config: AppConfig,
    started_at: int,
    started_perf: float,
    request_lock: Any,
    first_line: str,
    client: Any,
    response: Any,
    line_iterator: AsyncIterator[str],
) -> AsyncIterator[str]:
    responses_state = init_responses_stream_state(provider.protocol)
    first_events = await process_responses_stream_line(
        responses_state, first_line, OpenAIChatAdapter(), provider.protocol
    )
    response_id = responses_state.response_id
    tool_indices: dict[str, int] = {}
    try:
        yield (
            "event: message_start\n"
            f"data: {json.dumps({'type': 'message_start', 'message': {'id': response_id, 'type': 'message', 'role': 'assistant', 'model': responses_state.model, 'content': [], 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
        )
        yield (
            "event: content_block_start\n"
            f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        )
        for event in first_events:
            if event.event_type == "text_delta" and event.delta is not None:
                yield (
                    "event: content_block_delta\n"
                    f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': event.delta}})}\n\n"
                )
            elif event.event_type == "function_call_arguments_delta" and event.item_id is not None:
                if event.item_id not in tool_indices:
                    tool_indices[event.item_id] = len(tool_indices) + 1
                    tool_call = next(
                        (
                            call
                            for call in responses_state.tool_calls_in_order
                            if call.id == event.item_id
                        ),
                        None,
                    )
                    yield (
                        "event: content_block_start\n"
                        f"data: {json.dumps({'type': 'content_block_start', 'index': tool_indices[event.item_id], 'content_block': {'type': 'tool_use', 'id': event.item_id, 'name': _safe_tool_name(tool_call.name if tool_call is not None else None), 'input': {}}})}\n\n"
                    )
                yield (
                    "event: content_block_delta\n"
                    f"data: {json.dumps({'type': 'content_block_delta', 'index': tool_indices[event.item_id], 'delta': {'type': 'input_json_delta', 'partial_json': event.delta}})}\n\n"
                )
        async for line in line_iterator:
            if not line:
                continue
            events = await process_responses_stream_line(
                responses_state, line, OpenAIChatAdapter(), provider.protocol
            )
            for event in events:
                if event.event_type == "text_delta" and event.delta is not None:
                    yield (
                        "event: content_block_delta\n"
                        f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': event.delta}})}\n\n"
                    )
                elif (
                    event.event_type == "function_call_arguments_delta"
                    and event.item_id is not None
                ):
                    if event.item_id not in tool_indices:
                        tool_indices[event.item_id] = len(tool_indices) + 1
                        tool_call = next(
                            (
                                call
                                for call in responses_state.tool_calls_in_order
                                if call.id == event.item_id
                            ),
                            None,
                        )
                        yield (
                            "event: content_block_start\n"
                            f"data: {json.dumps({'type': 'content_block_start', 'index': tool_indices[event.item_id], 'content_block': {'type': 'tool_use', 'id': event.item_id, 'name': _safe_tool_name(tool_call.name if tool_call is not None else None), 'input': {}}})}\n\n"
                        )
                    yield (
                        "event: content_block_delta\n"
                        f"data: {json.dumps({'type': 'content_block_delta', 'index': tool_indices[event.item_id], 'delta': {'type': 'input_json_delta', 'partial_json': event.delta}})}\n\n"
                    )
    finally:
        await response.aclose()
        await client.aclose()

    final_response, _final_events = build_responses_stream_result(responses_state)
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
    for index in tool_indices.values():
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': index})}\n\n"
    yield (
        "event: message_delta\n"
        f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'input_tokens': final_response.usage.input_tokens, 'output_tokens': final_response.usage.output_tokens}})}\n\n"
    )
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    _record_live_provider_success(runtime, provider.name)
    if runtime is not None and api_key is not None:
        runtime.record_usage(
            api_key=api_key,
            request_id=request_id,
            started_at=started_at,
            request=canonical_request,
            response=final_response,
            provider_name=provider.name,
            latency_ms=int((time.perf_counter() - started_perf) * 1000),
        )
        await runtime.publish_request_event(
            {
                "request_id": request_id,
                "api_key": api_key.name,
                "provider": provider.name,
                "model": canonical_request.requested_model,
                "status": "success",
            }
        )
    _release_request_lock(request_lock)


async def _prepare_live_stream_bridge(
    *,
    inbound_protocol: ProtocolName,
    request_id: str,
    canonical_request: Any,
    providers: list[ProviderConfig],
    adapter: Any,
    runtime: RuntimeManager | None,
    api_key: AuthenticatedApiKey | None,
    store: ConversationStore,
    config: AppConfig,
    started_at: int,
    started_perf: float,
    request_lock: Any = None,
) -> AsyncIterator[str]:
    last_error: ProviderError | None = None
    for provider in providers:
        _record_live_provider_attempt(runtime, provider.name)
        outbound_adapter = get_adapter(provider.protocol)
        mapped_model = provider.map_model(canonical_request.requested_model)
        path, payload = outbound_adapter.build_request(canonical_request, mapped_model)
        client = create_async_client(config, provider)
        try:
            response = await open_stream(client, provider, path, payload)
            line_iterator = response.aiter_lines()
            first_line = await _read_first_stream_line(line_iterator, provider)
            if not first_line:
                await response.aclose()
                await client.aclose()
                raise ProviderError(
                    f"Provider {provider.name} returned an empty stream",
                    provider_name=provider.name,
                    retryable=False,
                    base_url=provider.base_url,
                    path=path,
                )
            if inbound_protocol == "oairesp" and provider.protocol == "oaichat":
                return _live_stream_bridge_oaichat_to_oairesp(
                    request_id=request_id,
                    canonical_request=canonical_request,
                    provider=provider,
                    adapter=adapter,
                    runtime=runtime,
                    api_key=api_key,
                    store=store,
                    config=config,
                    started_at=started_at,
                    started_perf=started_perf,
                    request_lock=request_lock,
                    first_line=first_line,
                    client=client,
                    response=response,
                    line_iterator=line_iterator,
                )
            if inbound_protocol == "oairesp" and provider.protocol == "anthropic":
                return _live_stream_bridge_anthropic_to_oairesp(
                    request_id=request_id,
                    canonical_request=canonical_request,
                    provider=provider,
                    adapter=adapter,
                    runtime=runtime,
                    api_key=api_key,
                    store=store,
                    config=config,
                    started_at=started_at,
                    started_perf=started_perf,
                    request_lock=request_lock,
                    first_line=first_line,
                    client=client,
                    response=response,
                    line_iterator=line_iterator,
                )
            if inbound_protocol == "oaichat" and provider.protocol == "anthropic":
                return _live_stream_bridge_anthropic_to_oaichat(
                    request_id=request_id,
                    canonical_request=canonical_request,
                    provider=provider,
                    adapter=adapter,
                    runtime=runtime,
                    api_key=api_key,
                    store=store,
                    config=config,
                    started_at=started_at,
                    started_perf=started_perf,
                    request_lock=request_lock,
                    first_line=first_line,
                    client=client,
                    response=response,
                    line_iterator=line_iterator,
                )
            if inbound_protocol == "oaichat" and provider.protocol == "oairesp":
                return _live_stream_bridge_oairesp_to_oaichat(
                    request_id=request_id,
                    canonical_request=canonical_request,
                    provider=provider,
                    adapter=adapter,
                    runtime=runtime,
                    api_key=api_key,
                    store=store,
                    config=config,
                    started_at=started_at,
                    started_perf=started_perf,
                    request_lock=request_lock,
                    first_line=first_line,
                    client=client,
                    response=response,
                    line_iterator=line_iterator,
                )
            if inbound_protocol == "anthropic" and provider.protocol == "oaichat":
                return _live_stream_bridge_oaichat_to_anthropic(
                    request_id=request_id,
                    canonical_request=canonical_request,
                    provider=provider,
                    adapter=adapter,
                    runtime=runtime,
                    api_key=api_key,
                    store=store,
                    config=config,
                    started_at=started_at,
                    started_perf=started_perf,
                    request_lock=request_lock,
                    first_line=first_line,
                    client=client,
                    response=response,
                    line_iterator=line_iterator,
                )
            if inbound_protocol == "anthropic" and provider.protocol == "oairesp":
                return _live_stream_bridge_oairesp_to_anthropic(
                    request_id=request_id,
                    canonical_request=canonical_request,
                    provider=provider,
                    adapter=adapter,
                    runtime=runtime,
                    api_key=api_key,
                    store=store,
                    config=config,
                    started_at=started_at,
                    started_perf=started_perf,
                    request_lock=request_lock,
                    first_line=first_line,
                    client=client,
                    response=response,
                    line_iterator=line_iterator,
                )
            await response.aclose()
            await client.aclose()
            raise ProviderError(
                f"Unsupported live stream bridge {provider.protocol}->{inbound_protocol}",
                provider_name=provider.name,
                retryable=False,
                base_url=provider.base_url,
                path=path,
            )
        except ProviderError as exc:
            last_error = exc
            _record_live_provider_error(runtime, provider.name, str(exc))
            await client.aclose()
            continue
        except Exception:
            await client.aclose()
            _release_request_lock(request_lock)
            raise
    if last_error is not None:
        _release_request_lock(request_lock)
        raise AllProvidersFailedError("all providers failed", last_error=last_error)
    _release_request_lock(request_lock)
    raise AllProvidersFailedError("all providers failed")


def _safe_tool_name(name: str | None) -> str:
    if isinstance(name, str) and name:
        return name
    return "tool"


def _release_request_lock(request_lock: Any) -> None:
    if request_lock is None:
        return
    release = getattr(request_lock, "release", None)
    if callable(release):
        release()
