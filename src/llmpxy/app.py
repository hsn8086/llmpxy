from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
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
    iter_response_stream_events,
)
from llmpxy.storage import ConversationStore


def create_app(
    config: AppConfig, store: ConversationStore, dispatcher: ProviderDispatcher
) -> FastAPI:
    app = FastAPI(title="llmpxy")

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


async def _handle_protocol_request(
    request: Request,
    config: AppConfig,
    store: ConversationStore,
    dispatcher: ProviderDispatcher,
    inbound_protocol: ProtocolName,
) -> Response:
    request_id = str(uuid.uuid4())
    started_perf = time.perf_counter()
    payload = await request.json()
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
            route_providers = dispatcher.resolve_route_provider_order()
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
                        started_perf=started_perf,
                    ),
                    media_type=_stream_media_type(inbound_protocol),
                    headers=_stream_headers(),
                )
            response, events, provider = await dispatcher.dispatch_stream(
                canonical_request, request_id=request_id
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

        response, provider = await dispatcher.dispatch(canonical_request, request_id=request_id)
    except ProviderError as exc:
        log.exception(
            "provider fatal error status_code={} response_text={}",
            exc.status_code,
            exc.response_text,
        )
        raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc
    except AllProvidersFailedError as exc:
        log.exception("all providers failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

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
    api_key = runtime.authenticate(request.headers.get("Authorization"))
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
    payload = await request.json()
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
        if canonical_request.stream:
            if _can_bridge_live_stream(inbound_protocol, selectable_providers):
                return StreamingResponse(
                    await _prepare_live_stream_bridge(
                        inbound_protocol=inbound_protocol,
                        request_id=request_id,
                        canonical_request=canonical_request,
                        providers=selectable_providers,
                        adapter=adapter,
                        runtime=runtime,
                        api_key=api_key,
                        store=store,
                        config=config,
                        started_perf=started_perf,
                    ),
                    media_type=_stream_media_type(inbound_protocol),
                    headers=_stream_headers(),
                )
            if _can_passthrough_live_stream(inbound_protocol, selectable_providers):
                return StreamingResponse(
                    await _prepare_live_stream_passthrough(
                        inbound_protocol=inbound_protocol,
                        request_id=request_id,
                        canonical_request=canonical_request,
                        providers=selectable_providers,
                        adapter=adapter,
                        runtime=runtime,
                        api_key=api_key,
                        store=store,
                        config=config,
                        started_perf=started_perf,
                    ),
                    media_type=_stream_media_type(inbound_protocol),
                    headers=_stream_headers(),
                )
            response, events, provider = await dispatcher.dispatch_stream(
                canonical_request,
                request_id=request_id,
                providers=selectable_providers,
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
            runtime.record_usage(
                api_key=api_key,
                request_id=request_id,
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
            providers=selectable_providers,
        )
        runtime.record_usage(
            api_key=api_key,
            request_id=request_id,
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
        raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc
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
        raise HTTPException(status_code=502, detail=str(exc)) from exc

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
    return (
        inbound_protocol == "oairesp"
        and all(provider.protocol == providers[0].protocol for provider in providers)
        and providers[0].protocol in {"oaichat", "anthropic"}
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
    if runtime is not None and api_key is not None:
        runtime.record_usage(
            api_key=api_key,
            request_id=request_id,
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
    started_perf: float,
) -> AsyncIterator[str]:
    last_error: ProviderError | None = None
    for provider in providers:
        outbound_adapter = get_adapter(provider.protocol)
        mapped_model = provider.map_model(canonical_request.requested_model)
        path, payload = outbound_adapter.build_request(canonical_request, mapped_model)
        client = create_async_client(config, provider)
        try:
            response = await open_stream(client, provider, path, payload)
            line_iterator = response.aiter_lines()
            first_line = ""
            async for line in line_iterator:
                if not line:
                    continue
                first_line = line
                break
            if not first_line:
                await response.aclose()
                await client.aclose()
                raise ProviderError(
                    f"Provider {provider.name} returned an empty stream",
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
                started_perf=started_perf,
                first_line=first_line,
                client=client,
                response=response,
                line_iterator=line_iterator,
            )
        except ProviderError as exc:
            last_error = exc
            await client.aclose()
            continue
        except Exception:
            await client.aclose()
            raise
    if last_error is not None:
        raise last_error
    raise ProviderError("No providers available for live stream", retryable=False)


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
    started_perf: float,
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
    if runtime is not None and api_key is not None:
        runtime.record_usage(
            api_key=api_key,
            request_id=request_id,
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
    started_perf: float,
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
    if runtime is not None and api_key is not None:
        runtime.record_usage(
            api_key=api_key,
            request_id=request_id,
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
    started_perf: float,
) -> AsyncIterator[str]:
    last_error: ProviderError | None = None
    for provider in providers:
        outbound_adapter = get_adapter(provider.protocol)
        mapped_model = provider.map_model(canonical_request.requested_model)
        path, payload = outbound_adapter.build_request(canonical_request, mapped_model)
        client = create_async_client(config, provider)
        try:
            response = await open_stream(client, provider, path, payload)
            line_iterator = response.aiter_lines()
            first_line = ""
            async for line in line_iterator:
                if not line:
                    continue
                first_line = line
                break
            if not first_line:
                await response.aclose()
                await client.aclose()
                raise ProviderError(
                    f"Provider {provider.name} returned an empty stream",
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
                    started_perf=started_perf,
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
                    started_perf=started_perf,
                    first_line=first_line,
                    client=client,
                    response=response,
                    line_iterator=line_iterator,
                )
            await response.aclose()
            await client.aclose()
            raise ProviderError(
                f"Unsupported live stream bridge {provider.protocol}->{inbound_protocol}",
                retryable=False,
                base_url=provider.base_url,
                path=path,
            )
        except ProviderError as exc:
            last_error = exc
            await client.aclose()
            continue
        except Exception:
            await client.aclose()
            raise
    if last_error is not None:
        raise last_error
    raise ProviderError("No providers available for live stream bridge", retryable=False)
