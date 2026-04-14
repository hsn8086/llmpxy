from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import HTTPException

from llmpxy.config import ProtocolName
from llmpxy.models import (
    CanonicalContentPart,
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStreamEvent,
    CanonicalToolCall,
    CanonicalUsage,
)


class AnthropicAdapter:
    protocol_name = "anthropic"

    def parse_request(
        self, payload: dict[str, Any], *, strip_unsupported_fields: bool
    ) -> CanonicalRequest:
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise HTTPException(status_code=400, detail="model is required")
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="messages must be a list")

        canonical_messages: list[CanonicalMessage] = []
        for message in messages:
            if not isinstance(message, dict):
                raise HTTPException(status_code=400, detail="each message must be an object")
            role = message.get("role")
            if role not in {"user", "assistant"}:
                raise HTTPException(
                    status_code=400, detail="Anthropic messages only support user and assistant"
                )
            canonical_messages.append(
                CanonicalMessage(
                    role=role, content=_normalize_anthropic_content(message.get("content"))
                )
            )

        system_prompt = payload.get("system") if isinstance(payload.get("system"), str) else None
        return CanonicalRequest(
            protocol_in="anthropic",
            requested_model=model,
            stream=bool(payload.get("stream", False)),
            system_prompt=system_prompt,
            messages=canonical_messages,
            temperature=_optional_float(payload.get("temperature")),
            top_p=_optional_float(payload.get("top_p")),
            max_output_tokens=_optional_int(payload.get("max_tokens")),
            tools=_normalize_anthropic_tools(payload.get("tools")),
            tool_choice=payload.get("tool_choice"),
            metadata=_normalize_metadata(payload.get("metadata")),
            raw_request=payload,
        )

    def build_request(
        self, request: CanonicalRequest, mapped_model: str
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": mapped_model,
            "messages": [
                _message_to_anthropic(message)
                for message in request.messages
                if message.role not in {"system", "developer"}
            ],
            "stream": request.stream,
            "max_tokens": request.max_output_tokens or 1024,
        }
        system_text = request.system_prompt or _extract_system_text(request.messages)
        if system_text is not None:
            payload["system"] = system_text
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.tools is not None:
            payload["tools"] = _build_anthropic_tools(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.metadata:
            payload["metadata"] = request.metadata
        return "/v1/messages", payload

    def parse_response(self, payload: dict[str, Any], protocol_out: str) -> CanonicalResponse:
        content = _normalize_anthropic_content(payload.get("content"))
        tool_calls = _normalize_anthropic_tool_calls(payload.get("content"))
        usage_payload = dict(payload["usage"]) if isinstance(payload.get("usage"), dict) else {}
        response_id = (
            payload["id"] if isinstance(payload.get("id"), str) else f"msg_{uuid.uuid4().hex}"
        )
        model = payload["model"] if isinstance(payload.get("model"), str) else ""
        return CanonicalResponse(
            response_id=response_id,
            protocol_out=_normalize_protocol(protocol_out, "anthropic"),
            model=model,
            created_at=int(time.time()),
            output_messages=[
                CanonicalMessage(role="assistant", content=content, tool_calls=tool_calls)
            ],
            usage=CanonicalUsage(
                input_tokens=int(usage_payload.get("input_tokens", 0)),
                output_tokens=int(usage_payload.get("output_tokens", 0)),
                total_tokens=int(usage_payload.get("input_tokens", 0))
                + int(usage_payload.get("output_tokens", 0)),
            ),
            raw_response=payload,
        )

    async def parse_stream(
        self,
        lines: AsyncIterator[str],
        protocol_out: str,
    ) -> tuple[CanonicalResponse, list[CanonicalStreamEvent]]:
        response_id = f"msg_{uuid.uuid4().hex}"
        text_parts: list[str] = []
        model = ""
        usage = CanonicalUsage()
        events: list[CanonicalStreamEvent] = []
        for line in [item async for item in lines]:
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            if payload.get("type") == "message_start":
                message = payload.get("message")
                if isinstance(message, dict) and isinstance(message.get("model"), str):
                    model = message["model"]
            elif payload.get("type") == "content_block_delta":
                delta = payload.get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("text"), str):
                    text_parts.append(delta["text"])
                    events.append(
                        CanonicalStreamEvent(
                            event_type="text_delta", response_id=response_id, delta=delta["text"]
                        )
                    )
            elif payload.get("type") == "message_delta":
                usage_payload = payload.get("usage")
                if isinstance(usage_payload, dict):
                    usage = CanonicalUsage(
                        input_tokens=int(usage_payload.get("input_tokens", usage.input_tokens)),
                        output_tokens=int(usage_payload.get("output_tokens", usage.output_tokens)),
                        total_tokens=int(usage_payload.get("input_tokens", usage.input_tokens))
                        + int(usage_payload.get("output_tokens", usage.output_tokens)),
                    )

        response = CanonicalResponse(
            response_id=response_id,
            protocol_out=_normalize_protocol(protocol_out, "anthropic"),
            model=model,
            created_at=int(time.time()),
            output_messages=[
                CanonicalMessage(
                    role="assistant",
                    content=[CanonicalContentPart(type="text", text="".join(text_parts))],
                )
            ],
            usage=usage,
        )
        return response, events

    def format_response(self, response: CanonicalResponse) -> dict[str, Any]:
        message = (
            response.output_messages[0]
            if response.output_messages
            else CanonicalMessage(role="assistant")
        )
        return {
            "id": response.response_id,
            "type": "message",
            "role": "assistant",
            "model": response.model,
            "content": [
                {"type": "text", "text": part.text}
                for part in message.content
                if part.type == "text" and part.text is not None
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    async def format_stream(
        self,
        response: CanonicalResponse,
        events: list[CanonicalStreamEvent],
    ) -> AsyncIterator[str]:
        start_event = {
            "type": "message_start",
            "message": {
                "id": response.response_id,
                "type": "message",
                "role": "assistant",
                "model": response.model,
                "content": [],
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            },
        }
        yield f"event: message_start\ndata: {json.dumps(start_event)}\n\n"
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        for event in events:
            if event.event_type != "text_delta" or event.delta is None:
                continue
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': event.delta}})}\n\n"
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'input_tokens': response.usage.input_tokens, 'output_tokens': response.usage.output_tokens}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise HTTPException(status_code=400, detail="Expected numeric value")
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise HTTPException(status_code=400, detail="Expected integer value")
    return value


def _normalize_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _normalize_protocol(value: str, default: ProtocolName) -> ProtocolName:
    if value == "oairesp":
        return "oairesp"
    if value == "oaichat":
        return "oaichat"
    if value == "anthropic":
        return "anthropic"
    return default


def _normalize_anthropic_content(content: Any) -> list[CanonicalContentPart]:
    if isinstance(content, str):
        return [CanonicalContentPart(type="text", text=content)]
    if isinstance(content, list):
        parts: list[CanonicalContentPart] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(CanonicalContentPart(type="text", text=item["text"]))
        return parts
    return []


def _normalize_anthropic_tool_calls(content: Any) -> list[CanonicalToolCall]:
    if not isinstance(content, list):
        return []
    calls: list[CanonicalToolCall] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "tool_use":
            continue
        if not isinstance(item.get("name"), str):
            continue
        calls.append(
            CanonicalToolCall(
                id=item.get("id")
                if isinstance(item.get("id"), str)
                else f"call_{uuid.uuid4().hex}",
                type="function",
                name=item["name"],
                arguments=json.dumps(item.get("input", {}), ensure_ascii=True),
            )
        )
    return calls


def _normalize_anthropic_tools(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise HTTPException(status_code=400, detail="tools must be a list")

    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="tool must be an object")
        if item.get("type") == "function" and isinstance(item.get("name"), str):
            tool = _strip_none_values(item)
            if not isinstance(tool, dict):
                raise HTTPException(status_code=400, detail="tool must be an object")
            tool["input_schema"] = _normalize_tool_parameters(item.get("input_schema"))
            normalized.append(tool)
            continue
        if isinstance(item.get("name"), str):
            tool = _strip_none_values(item)
            if not isinstance(tool, dict):
                raise HTTPException(status_code=400, detail="tool must be an object")
            tool["input_schema"] = _normalize_tool_parameters(item.get("input_schema"))
            normalized.append(tool)
            continue
        raise HTTPException(status_code=400, detail="tool must include a name")
    return normalized


def _build_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in tools:
        if item.get("type") == "function":
            function = item.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                normalized.append(
                    {
                        "name": function["name"],
                        "description": function.get("description", ""),
                        "input_schema": _normalize_tool_parameters(function.get("parameters")),
                    }
                )
                continue
            if isinstance(item.get("name"), str):
                normalized.append(
                    {
                        "name": item["name"],
                        "description": item.get("description", ""),
                        "input_schema": _normalize_tool_parameters(item.get("parameters")),
                    }
                )
                continue
        normalized.append(item)
    return normalized


def _normalize_tool_parameters(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": "object", "properties": {}, "required": []}

    normalized = _strip_none_values(value)
    if not isinstance(normalized, dict):
        return {"type": "object", "properties": {}, "required": []}

    normalized = _normalize_json_schema(normalized)
    if not isinstance(normalized.get("type"), str):
        normalized["type"] = "object"
    if not isinstance(normalized.get("properties"), dict):
        normalized["properties"] = {}
    normalized["required"] = list(normalized["properties"].keys())
    return normalized


def _normalize_json_schema(value: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(value)

    properties = normalized.get("properties")
    if isinstance(properties, dict):
        normalized["properties"] = {
            key: _normalize_json_schema(item) if isinstance(item, dict) else item
            for key, item in properties.items()
        }
        if not isinstance(normalized.get("type"), str):
            normalized["type"] = "object"
        normalized["required"] = list(normalized["properties"].keys())

    items = normalized.get("items")
    if isinstance(items, dict):
        normalized["items"] = _normalize_json_schema(items)
        if not isinstance(normalized.get("type"), str):
            normalized["type"] = "array"

    for key in ("anyOf", "oneOf", "allOf"):
        schemas = normalized.get(key)
        if isinstance(schemas, list):
            normalized[key] = [
                _normalize_json_schema(item) if isinstance(item, dict) else item for item in schemas
            ]

    return normalized


def _strip_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if item is None:
                continue
            result[key] = _strip_none_values(item)
        return result
    if isinstance(value, list):
        return [_strip_none_values(item) for item in value if item is not None]
    return value


def _message_to_anthropic(message: CanonicalMessage) -> dict[str, Any]:
    return {
        "role": "assistant" if message.role == "assistant" else "user",
        "content": [
            {"type": "text", "text": part.text}
            for part in message.content
            if part.type == "text" and part.text is not None
        ],
    }


def _extract_system_text(messages: list[CanonicalMessage]) -> str | None:
    system_parts: list[str] = []
    for message in messages:
        if message.role not in {"system", "developer"}:
            continue
        for part in message.content:
            if part.type == "text" and part.text:
                system_parts.append(part.text)
    if not system_parts:
        return None
    return "\n\n".join(system_parts)
