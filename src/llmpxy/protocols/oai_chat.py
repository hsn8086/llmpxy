from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import HTTPException
from loguru import logger

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


class OpenAIChatAdapter:
    protocol_name = "oaichat"

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
            if role not in {"system", "developer", "user", "assistant", "tool"}:
                raise HTTPException(status_code=400, detail="unsupported message role")
            canonical_messages.append(
                CanonicalMessage(
                    role=role,
                    content=_normalize_content(message.get("content")),
                    tool_calls=_normalize_tool_calls(message.get("tool_calls")),
                    tool_call_id=message.get("tool_call_id")
                    if isinstance(message.get("tool_call_id"), str)
                    else None,
                )
            )

        return CanonicalRequest(
            protocol_in="oaichat",
            requested_model=model,
            stream=bool(payload.get("stream", False)),
            messages=canonical_messages,
            temperature=_optional_float(payload.get("temperature")),
            top_p=_optional_float(payload.get("top_p")),
            max_output_tokens=_optional_int(payload.get("max_tokens")),
            reasoning=_normalize_reasoning(
                payload.get("reasoning"), payload.get("reasoning_effort")
            ),
            tools=_normalize_tools(payload.get("tools")),
            tool_choice=payload.get("tool_choice"),
            response_format=payload.get("response_format")
            if isinstance(payload.get("response_format"), dict)
            else None,
            metadata=_normalize_metadata(payload.get("metadata")),
            raw_request=payload,
        )

    def build_request(
        self, request: CanonicalRequest, mapped_model: str
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": mapped_model,
            "messages": [_message_to_oaichat(message) for message in request.messages],
            "stream": request.stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        reasoning = _build_reasoning_payload(request.reasoning)
        if reasoning is not None:
            payload["reasoning_effort"] = reasoning["effort"]
        if request.tools is not None:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        logger.bind(request_id="-", provider="-", round=0, attempt=0).debug(
            "oaichat build_request mapped_model={} message_count={} reasoning={} max_tokens={}",
            mapped_model,
            len(payload["messages"]),
            payload.get("reasoning_effort"),
            payload.get("max_tokens"),
        )
        return "/chat/completions", payload

    def parse_response(self, payload: dict[str, Any], protocol_out: str) -> CanonicalResponse:
        choice = _first_choice(payload)
        message = choice.get("message", {})
        content = _normalize_content(message.get("content"))
        tool_calls = _normalize_tool_calls(message.get("tool_calls"))
        reasoning_content = (
            message.get("reasoning_content")
            if isinstance(message.get("reasoning_content"), str)
            else None
        )
        response_id = (
            payload["id"] if isinstance(payload.get("id"), str) else f"chatcmpl_{uuid.uuid4().hex}"
        )
        model = payload["model"] if isinstance(payload.get("model"), str) else ""
        protocol = _normalize_protocol(protocol_out, "oaichat")
        return CanonicalResponse(
            response_id=response_id,
            protocol_out=protocol,
            model=model,
            created_at=int(payload.get("created", int(time.time()))),
            output_messages=[
                CanonicalMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_content,
                )
            ],
            usage=CanonicalUsage(
                input_tokens=int(payload.get("usage", {}).get("prompt_tokens", 0)),
                output_tokens=int(payload.get("usage", {}).get("completion_tokens", 0)),
                total_tokens=int(payload.get("usage", {}).get("total_tokens", 0)),
            ),
            raw_response=payload,
        )

    async def parse_stream(
        self,
        lines: AsyncIterator[str],
        protocol_out: str,
    ) -> tuple[CanonicalResponse, list[CanonicalStreamEvent]]:
        response_id = f"chatcmpl_{uuid.uuid4().hex}"
        item_id = f"msg_{uuid.uuid4().hex}"
        model = ""
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        active_tool_calls_by_index: dict[int, CanonicalToolCall] = {}
        tool_calls: list[CanonicalToolCall] = []
        usage = CanonicalUsage()
        events: list[CanonicalStreamEvent] = []

        async for line in lines:
            if not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                break
            payload = json.loads(raw)
            if isinstance(payload.get("model"), str):
                model = payload["model"]
            if isinstance(payload.get("usage"), dict):
                usage = CanonicalUsage(
                    input_tokens=int(payload["usage"].get("prompt_tokens", usage.input_tokens)),
                    output_tokens=int(
                        payload["usage"].get("completion_tokens", usage.output_tokens)
                    ),
                    total_tokens=int(payload["usage"].get("total_tokens", usage.total_tokens)),
                )
            choice = (payload.get("choices") or [{}])[0]
            delta = choice.get("delta", {})
            for text in _extract_delta_texts(delta.get("content")):
                text_parts.append(text)
                events.append(
                    CanonicalStreamEvent(
                        event_type="text_delta",
                        response_id=response_id,
                        item_id=item_id,
                        delta=text,
                    )
                )
            for text in _extract_reasoning_texts(delta):
                reasoning_parts.append(text)
            for event in _extract_tool_call_events(
                delta.get("tool_calls"),
                active_tool_calls_by_index,
                tool_calls,
            ):
                events.append(event)

            message = choice.get("message")
            if isinstance(message, dict):
                for text in _extract_reasoning_texts(message):
                    reasoning_parts.append(text)

        response = CanonicalResponse(
            response_id=response_id,
            protocol_out=_normalize_protocol(protocol_out, "oaichat"),
            model=model,
            created_at=int(time.time()),
            output_messages=[
                CanonicalMessage(
                    role="assistant",
                    content=[CanonicalContentPart(type="text", text="".join(text_parts))],
                    tool_calls=tool_calls,
                    reasoning_content="".join(reasoning_parts) or None,
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
        payload: dict[str, Any] = {
            "id": response.response_id,
            "object": "chat.completion",
            "created": response.created_at,
            "model": response.model,
            "choices": [
                {"index": 0, "message": _message_to_oaichat(message), "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
        return payload

    async def format_stream(
        self,
        response: CanonicalResponse,
        events: list[CanonicalStreamEvent],
    ) -> AsyncIterator[str]:
        for event in events:
            if event.event_type != "text_delta" or event.delta is None:
                continue
            chunk = {
                "id": response.response_id,
                "object": "chat.completion.chunk",
                "created": response.created_at,
                "model": response.model,
                "choices": [{"index": 0, "delta": {"content": event.delta}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        final_chunk = {
            "id": response.response_id,
            "object": "chat.completion.chunk",
            "created": response.created_at,
            "model": response.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"


def _first_choice(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise HTTPException(status_code=502, detail="Invalid chat completion response")
    return choices[0]


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


def _normalize_tools(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise HTTPException(status_code=400, detail="tools must be a list")

    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="tool must be an object")
        if item.get("type") != "function":
            normalized.append(item)
            continue

        function = item.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            normalized.append(
                {
                    **item,
                    "function": {
                        **function,
                        "parameters": _normalize_tool_parameters(function.get("parameters")),
                    },
                }
            )
            continue

        if isinstance(item.get("name"), str):
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "description": item.get("description", ""),
                        "parameters": _normalize_tool_parameters(item.get("parameters")),
                        "strict": item.get("strict", True),
                    },
                }
            )
            continue

        raise HTTPException(status_code=400, detail="function tool must include a name")

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


def _normalize_protocol(value: str, default: ProtocolName) -> ProtocolName:
    if value == "oairesp":
        return "oairesp"
    if value == "oaichat":
        return "oaichat"
    if value == "anthropic":
        return "anthropic"
    return default


def _normalize_reasoning(value: Any, effort_value: Any = None) -> dict[str, Any] | None:
    reasoning: dict[str, Any] = {}
    if isinstance(value, dict) and isinstance(value.get("effort"), str):
        reasoning["effort"] = value["effort"]
    elif isinstance(effort_value, str):
        reasoning["effort"] = effort_value
    return reasoning or None


def _build_reasoning_payload(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not value:
        return None
    reasoning = dict(value)
    effort = reasoning.get("effort")
    if effort == "minimal":
        reasoning["effort"] = "low"
    return reasoning or None


def _normalize_content(content: Any) -> list[CanonicalContentPart]:
    if content is None:
        return []
    if isinstance(content, str):
        return [CanonicalContentPart(type="text", text=content)]
    if isinstance(content, list):
        parts: list[CanonicalContentPart] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(CanonicalContentPart(type="text", text=item["text"]))
            elif item.get("type") == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                    parts.append(CanonicalContentPart(type="image_url", url=image_url["url"]))
        return parts
    raise HTTPException(status_code=400, detail="invalid content")


def _normalize_tool_calls(value: Any) -> list[CanonicalToolCall]:
    if not isinstance(value, list):
        return []
    calls: list[CanonicalToolCall] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        if not isinstance(function.get("name"), str):
            continue
        calls.append(
            CanonicalToolCall(
                id=item.get("id")
                if isinstance(item.get("id"), str)
                else f"call_{uuid.uuid4().hex}",
                type=item.get("type") if isinstance(item.get("type"), str) else "function",
                name=function["name"],
                arguments=function.get("arguments")
                if isinstance(function.get("arguments"), str)
                else "",
            )
        )
    return calls


def _message_to_oaichat(message: CanonicalMessage) -> dict[str, Any]:
    role = "system" if message.role == "developer" else message.role
    payload: dict[str, Any] = {"role": role}
    if not message.content:
        payload["content"] = ""
    elif (
        len(message.content) == 1
        and message.content[0].type == "text"
        and message.content[0].text is not None
    ):
        payload["content"] = message.content[0].text
    else:
        payload["content"] = [
            {"type": "text", "text": item.text}
            if item.type == "text"
            else {"type": "image_url", "image_url": {"url": item.url}}
            for item in message.content
        ]
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {"name": tool_call.name, "arguments": tool_call.arguments},
            }
            for tool_call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.reasoning_content is not None:
        payload["reasoning_content"] = message.reasoning_content
    return payload


def _extract_delta_texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content] if content else []
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
        return texts
    return []


def _extract_reasoning_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(_extract_reasoning_texts(item))
        return texts
    if not isinstance(value, dict):
        return []

    direct_keys = (
        "reasoning_content",
        "reasoning",
        "text",
        "summary",
        "summary_text",
    )
    texts: list[str] = []
    for key in direct_keys:
        item = value.get(key)
        if isinstance(item, str) and item:
            texts.append(item)
        elif isinstance(item, list | dict):
            texts.extend(_extract_reasoning_texts(item))

    item_type = value.get("type")
    if item_type in {"summary_text", "reasoning_text", "reasoning_content"} and isinstance(
        value.get("text"), str
    ):
        text = value["text"]
        if text and text not in texts:
            texts.append(text)
    return texts


def _extract_tool_call_events(
    value: Any,
    active_tool_calls_by_index: dict[int, CanonicalToolCall],
    tool_calls: list[CanonicalToolCall],
) -> list[CanonicalStreamEvent]:
    if not isinstance(value, list):
        return []

    events: list[CanonicalStreamEvent] = []
    for fallback_index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        raw_item = dict(item)
        raw_index = raw_item.get("index")
        index = raw_index if isinstance(raw_index, int) else fallback_index
        existing = active_tool_calls_by_index.get(index)
        raw_id = raw_item.get("id")
        function = raw_item.get("function")
        raw_type = raw_item.get("type")

        if existing is None or (isinstance(raw_id, str) and raw_id != existing.id):
            tool_call_id = raw_id if isinstance(raw_id, str) else f"call_{uuid.uuid4().hex}"
            tool_call_name = (
                function.get("name")
                if isinstance(function, dict) and isinstance(function.get("name"), str)
                else ""
            )
            existing = CanonicalToolCall(
                id=tool_call_id,
                type=raw_type if isinstance(raw_type, str) else "function",
                name=tool_call_name,
                arguments="",
            )
            active_tool_calls_by_index[index] = existing
            tool_calls.append(existing)

        if isinstance(raw_id, str):
            existing.id = raw_id
        if isinstance(raw_type, str):
            existing.type = raw_type

        if isinstance(function, dict):
            if isinstance(function.get("name"), str):
                existing.name = function["name"]
            arguments_delta = function.get("arguments")
            if isinstance(arguments_delta, str) and arguments_delta:
                existing.arguments += arguments_delta
                events.append(
                    CanonicalStreamEvent(
                        event_type="function_call_arguments_delta",
                        response_id="",
                        item_id=existing.id,
                        delta=arguments_delta,
                        payload={"name": existing.name, "output_index": index},
                    )
                )

    return events
