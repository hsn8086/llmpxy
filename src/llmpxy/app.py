from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from llmpxy.config import AppConfig, ProtocolName
from llmpxy.conversation import merge_history
from llmpxy.dispatcher import AllProvidersFailedError, ProviderDispatcher
from llmpxy.logging_utils import bind_logger, sanitize_for_logging
from llmpxy.models import CanonicalMessage, CanonicalResponse, StoredConversation
from llmpxy.protocols.registry import get_adapter
from llmpxy.proxy_client import ProviderError
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


async def _handle_protocol_request(
    request: Request,
    config: AppConfig,
    store: ConversationStore,
    dispatcher: ProviderDispatcher,
    inbound_protocol: ProtocolName,
) -> Response:
    request_id = str(uuid.uuid4())
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
            response, events, provider = await dispatcher.dispatch_stream(
                canonical_request, request_id=request_id
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
