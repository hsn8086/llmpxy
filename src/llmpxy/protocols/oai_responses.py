from __future__ import annotations

from dataclasses import dataclass, field
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
from llmpxy.protocols.oai_chat import OpenAIChatAdapter


class OpenAIResponsesAdapter:
    protocol_name = "oairesp"

    def __init__(self) -> None:
        self._chat_adapter = OpenAIChatAdapter()

    def parse_request(
        self, payload: dict[str, Any], *, strip_unsupported_fields: bool
    ) -> CanonicalRequest:
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise HTTPException(status_code=400, detail="model is required")
        metadata = _normalize_metadata(payload.get("metadata"))
        include = _normalize_include(payload.get("include"))
        if include:
            metadata["_request_include"] = include

        return CanonicalRequest(
            protocol_in="oairesp",
            requested_model=model,
            stream=bool(payload.get("stream", False)),
            system_prompt=payload.get("instructions")
            if isinstance(payload.get("instructions"), str)
            else None,
            messages=_normalize_input(payload.get("input")),
            temperature=_optional_float(payload.get("temperature")),
            top_p=_optional_float(payload.get("top_p")),
            max_output_tokens=_optional_int(payload.get("max_output_tokens")),
            reasoning=_normalize_reasoning(payload.get("reasoning")),
            tools=_normalize_response_tools(payload.get("tools")),
            tool_choice=payload.get("tool_choice"),
            response_format=payload.get("response_format")
            if isinstance(payload.get("response_format"), dict)
            else None,
            metadata=metadata,
            conversation_reference=payload.get("previous_response_id")
            if isinstance(payload.get("previous_response_id"), str)
            else None,
            raw_request=payload,
        )

    def build_request(
        self, request: CanonicalRequest, mapped_model: str
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": mapped_model,
            "input": [
                item for message in request.messages for item in _message_to_response_items(message)
            ],
            "stream": request.stream,
        }
        instructions = request.system_prompt or _extract_system_text(request.messages) or ""
        payload["instructions"] = instructions
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_output_tokens is not None:
            payload["max_output_tokens"] = request.max_output_tokens
        reasoning = _build_reasoning_payload(request.reasoning)
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if request.tools is not None:
            payload["tools"] = _build_response_tools(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.response_format is not None:
            payload["text"] = {"format": request.response_format}
        metadata = _build_upstream_metadata(request.metadata)
        if metadata:
            payload["metadata"] = metadata
        logger.bind(request_id="-", provider="-", round=0, attempt=0).debug(
            "oairesp build_request mapped_model={} instructions_len={} reasoning={}",
            mapped_model,
            len(payload["instructions"]),
            payload.get("reasoning"),
        )
        return "/responses", payload

    def parse_response(self, payload: dict[str, Any], protocol_out: str) -> CanonicalResponse:
        if isinstance(payload.get("output"), list):
            output = payload["output"]
            messages: list[CanonicalMessage] = []
            pending_tool_calls: list[CanonicalToolCall] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "function_call":
                    pending_tool_calls.extend(_normalize_output_tool_calls(item))
                    continue
                if item.get("type") != "message":
                    continue
                messages.append(
                    CanonicalMessage(
                        role=item.get("role")
                        if item.get("role") in {"assistant", "user", "system", "tool"}
                        else "assistant",
                        content=_normalize_output_content(item.get("content")),
                        tool_calls=pending_tool_calls if item.get("role") == "assistant" else [],
                    )
                )
                pending_tool_calls = []
            if pending_tool_calls:
                if messages and messages[-1].role == "assistant":
                    messages[-1].tool_calls.extend(pending_tool_calls)
                else:
                    messages.append(
                        CanonicalMessage(role="assistant", tool_calls=pending_tool_calls)
                    )
            usage_payload = dict(payload["usage"]) if isinstance(payload.get("usage"), dict) else {}
            response_id = (
                payload["id"] if isinstance(payload.get("id"), str) else f"resp_{uuid.uuid4().hex}"
            )
            model = payload["model"] if isinstance(payload.get("model"), str) else ""
            return CanonicalResponse(
                response_id=response_id,
                protocol_out=_normalize_protocol(protocol_out, "oairesp"),
                model=model,
                created_at=int(payload.get("created_at", int(time.time()))),
                output_messages=messages,
                usage=CanonicalUsage(
                    input_tokens=int(usage_payload.get("input_tokens", 0)),
                    cached_input_tokens=int(
                        usage_payload.get("input_tokens_details", {}).get("cached_tokens", 0)
                    ),
                    output_tokens=int(usage_payload.get("output_tokens", 0)),
                    total_tokens=int(usage_payload.get("total_tokens", 0)),
                ),
                metadata={
                    **_normalize_metadata(payload.get("metadata")),
                    **(
                        {"_response_reasoning": dict(payload["reasoning"])}
                        if isinstance(payload.get("reasoning"), dict)
                        else {}
                    ),
                },
                raw_response=payload,
            )
        return self._chat_adapter.parse_response(payload, protocol_out)

    async def parse_stream(
        self,
        lines: AsyncIterator[str],
        protocol_out: str,
    ) -> tuple[CanonicalResponse, list[CanonicalStreamEvent]]:
        state = init_responses_stream_state(protocol_out)
        async for line in lines:
            await process_responses_stream_line(state, line, self._chat_adapter, protocol_out)
        return build_responses_stream_result(state)

    def format_response(self, response: CanonicalResponse) -> dict[str, Any]:
        message = (
            response.output_messages[0]
            if response.output_messages
            else CanonicalMessage(role="assistant")
        )
        output_items: list[dict[str, Any]] = []
        for tool_call in message.tool_calls:
            output_items.append(
                {
                    "id": tool_call.id,
                    "type": "function_call",
                    "call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "status": "completed",
                }
            )

        output_item: dict[str, Any] = {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": message.role,
            "status": "completed",
            "content": [
                {"type": "output_text", "text": part.text, "annotations": [], "logprobs": []}
                for part in message.content
                if part.type == "text" and part.text is not None
            ],
            "phase": "final_answer",
        }
        if (
            message.reasoning_content is not None
            and response.metadata.get("reasoning_summary") is not None
        ):
            output_items.append(
                _build_reasoning_item(
                    message.reasoning_content,
                    include_encrypted_content=_wants_reasoning_encrypted_content(response),
                )
            )
        output_items.append(output_item)

        public_metadata = _public_response_metadata(response.metadata)
        return {
            "id": response.response_id,
            "object": "response",
            "created_at": response.created_at,
            "status": response.status,
            "model": response.model,
            **(
                {"reasoning": response.metadata["_response_reasoning"]}
                if isinstance(response.metadata.get("_response_reasoning"), dict)
                else {}
            ),
            "output": output_items,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "metadata": public_metadata,
        }

    async def format_stream(
        self,
        response: CanonicalResponse,
        events: list[CanonicalStreamEvent],
    ) -> AsyncIterator[str]:
        state = init_response_stream_formatter_state(response)
        for event in iter_response_stream_events(state, events):
            yield event
        for event in finalize_response_stream(state):
            yield event


@dataclass
class ResponseStreamFormatterState:
    response: CanonicalResponse
    created: int
    public_metadata: dict[str, Any]
    created_event_chunk: str = ""
    final_text_parts: list[str] = field(default_factory=list)
    sequence_number: int = 0
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    message_output_index: int = 0
    message_opened: bool = False
    completed_item: dict[str, Any] | None = None
    tool_call_output_index_by_id: dict[str, int] = field(default_factory=dict)


def init_response_stream_formatter_state(
    response: CanonicalResponse,
) -> ResponseStreamFormatterState:
    state = ResponseStreamFormatterState(
        response=response,
        created=response.created_at,
        public_metadata=_public_response_metadata(response.metadata),
    )
    created_event = {
        "type": "response.created",
        "sequence_number": 0,
        "response": {
            "id": response.response_id,
            "object": "response",
            "created_at": response.created_at,
            "status": "in_progress",
            "model": response.model,
            **(
                {"reasoning": response.metadata["_response_reasoning"]}
                if isinstance(response.metadata.get("_response_reasoning"), dict)
                else {}
            ),
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "metadata": state.public_metadata,
        },
    }
    _log_stream_event(created_event)
    state.completed_item = {
        "id": state.message_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": "",
                "annotations": [],
                "logprobs": [],
            }
        ],
        "phase": "final_answer",
    }
    state.created_event_chunk = f"data: {json.dumps(created_event)}\n\n"
    return state


@dataclass
class ResponsesStreamState:
    protocol_out: str
    response_id: str = field(default_factory=lambda: f"resp_{uuid.uuid4().hex}")
    item_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    model: str = ""
    text_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    response_reasoning: dict[str, Any] | None = None
    usage: CanonicalUsage = field(default_factory=CanonicalUsage)
    events: list[CanonicalStreamEvent] = field(default_factory=list)
    tool_calls_by_item_id: dict[str, CanonicalToolCall] = field(default_factory=dict)
    tool_calls_in_order: list[CanonicalToolCall] = field(default_factory=list)


def init_responses_stream_state(protocol_out: str) -> ResponsesStreamState:
    return ResponsesStreamState(protocol_out=protocol_out)


async def process_responses_stream_line(
    state: ResponsesStreamState,
    line: str,
    chat_adapter: OpenAIChatAdapter,
    protocol_out: str,
) -> list[CanonicalStreamEvent]:
    if not line.startswith("data: "):
        return []
    raw = line[6:]
    if raw == "[DONE]":
        return []
    payload = json.loads(raw)
    payload_type = payload.get("type") if isinstance(payload.get("type"), str) else None
    payload_object = payload.get("object") if isinstance(payload.get("object"), str) else None
    response_event_type = payload_type or payload_object
    emitted: list[CanonicalStreamEvent] = []
    if isinstance(response_event_type, str) and response_event_type.startswith("response."):
        if isinstance(payload.get("model"), str):
            state.model = payload["model"]
        response_payload = payload.get("response")
        if (
            not state.model
            and isinstance(response_payload, dict)
            and isinstance(response_payload.get("model"), str)
        ):
            state.model = response_payload["model"]
        if isinstance(response_payload, dict) and isinstance(
            response_payload.get("reasoning"), dict
        ):
            state.response_reasoning = dict(response_payload["reasoning"])
        if response_event_type == "response.output_text.delta" and isinstance(
            payload.get("delta"), str
        ):
            state.text_parts.append(payload["delta"])
            event = CanonicalStreamEvent(
                event_type="text_delta",
                response_id=state.response_id,
                item_id=state.item_id,
                delta=payload["delta"],
            )
            state.events.append(event)
            emitted.append(event)
        elif response_event_type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        } and isinstance(payload.get("delta"), str):
            state.reasoning_parts.append(payload["delta"])
        elif response_event_type in {
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
        }:
            pass
        elif response_event_type == "response.function_call_arguments.delta" and isinstance(
            payload.get("item_id"), str
        ):
            item_id_value = payload["item_id"]
            tool_call = state.tool_calls_by_item_id.get(item_id_value)
            if tool_call is None:
                tool_call = CanonicalToolCall(id=item_id_value, name="", arguments="")
                state.tool_calls_by_item_id[item_id_value] = tool_call
                state.tool_calls_in_order.append(tool_call)
            if isinstance(payload.get("delta"), str):
                tool_call.arguments += payload["delta"]
                event = CanonicalStreamEvent(
                    event_type="function_call_arguments_delta",
                    response_id=state.response_id,
                    item_id=item_id_value,
                    delta=payload["delta"],
                )
                state.events.append(event)
                emitted.append(event)
        elif response_event_type == "response.output_item.done" and isinstance(
            payload.get("item"), dict
        ):
            item_payload = payload["item"]
            if item_payload.get("type") == "function_call" and isinstance(
                item_payload.get("id"), str
            ):
                item_id_value = item_payload["id"]
                call_id = item_payload.get("call_id")
                tool_call = state.tool_calls_by_item_id.get(item_id_value)
                if tool_call is None:
                    tool_call = CanonicalToolCall(
                        id=call_id if isinstance(call_id, str) and call_id else item_id_value,
                        name=item_payload.get("name")
                        if isinstance(item_payload.get("name"), str)
                        else "",
                        arguments=item_payload.get("arguments")
                        if isinstance(item_payload.get("arguments"), str)
                        else "",
                    )
                    state.tool_calls_by_item_id[item_id_value] = tool_call
                    state.tool_calls_in_order.append(tool_call)
                else:
                    if isinstance(call_id, str) and call_id:
                        tool_call.id = call_id
                    if isinstance(item_payload.get("name"), str):
                        tool_call.name = item_payload["name"]
                    if isinstance(item_payload.get("arguments"), str):
                        tool_call.arguments = item_payload["arguments"]
            elif item_payload.get("type") == "reasoning":
                for part in item_payload.get("summary", []):
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        state.reasoning_parts.append(part["text"])
        completed = response_payload
        if isinstance(completed, dict) and isinstance(completed.get("usage"), dict):
            state.usage = CanonicalUsage(
                input_tokens=int(completed["usage"].get("input_tokens", state.usage.input_tokens)),
                cached_input_tokens=int(
                    completed["usage"]
                    .get("input_tokens_details", {})
                    .get("cached_tokens", state.usage.cached_input_tokens)
                ),
                output_tokens=int(
                    completed["usage"].get("output_tokens", state.usage.output_tokens)
                ),
                total_tokens=int(completed["usage"].get("total_tokens", state.usage.total_tokens)),
            )
        return emitted

    chat_response, chat_events = await chat_adapter.parse_stream(
        _single_line_stream(payload), protocol_out
    )
    state.response_id = chat_response.response_id
    state.model = chat_response.model
    state.usage = chat_response.usage
    if chat_response.output_messages:
        state.text_parts = [part.text or "" for part in chat_response.output_messages[0].content]
    if chat_response.output_messages:
        reasoning_content = chat_response.output_messages[0].reasoning_content
        if isinstance(reasoning_content, str) and reasoning_content:
            state.reasoning_parts.append(reasoning_content)
    state.events.extend(chat_events)
    emitted.extend(chat_events)
    return emitted


def build_responses_stream_result(
    state: ResponsesStreamState,
) -> tuple[CanonicalResponse, list[CanonicalStreamEvent]]:
    response = CanonicalResponse(
        response_id=state.response_id,
        protocol_out=_normalize_protocol(state.protocol_out, "oairesp"),
        model=state.model,
        created_at=int(time.time()),
        output_messages=[
            CanonicalMessage(
                role="assistant",
                content=[CanonicalContentPart(type="text", text="".join(state.text_parts))],
                tool_calls=state.tool_calls_in_order,
                reasoning_content="".join(state.reasoning_parts) or None,
            )
        ],
        usage=state.usage,
        metadata=(
            {"_response_reasoning": state.response_reasoning}
            if state.response_reasoning is not None
            else {}
        ),
    )
    return response, state.events


def iter_response_stream_events(
    state: ResponseStreamFormatterState,
    events: list[CanonicalStreamEvent],
) -> list[str]:
    emitted: list[str] = []
    if state.created_event_chunk:
        emitted.append(state.created_event_chunk)
        state.created_event_chunk = ""
    for stream_event in events:
        if (
            stream_event.event_type == "function_call_arguments_delta"
            and stream_event.item_id is not None
        ):
            output_index = state.tool_call_output_index_by_id.get(stream_event.item_id)
            if output_index is None:
                output_index = len(state.tool_call_output_index_by_id)
                state.tool_call_output_index_by_id[stream_event.item_id] = output_index
                tool_call = (
                    next(
                        (
                            call
                            for call in state.response.output_messages[0].tool_calls
                            if call.id == stream_event.item_id
                        ),
                        None,
                    )
                    if state.response.output_messages
                    else None
                )
                state.sequence_number += 1
                event = {
                    "type": "response.output_item.added",
                    "sequence_number": state.sequence_number,
                    "output_index": output_index,
                    "item": {
                        "id": stream_event.item_id,
                        "type": "function_call",
                        "call_id": stream_event.item_id,
                        "name": tool_call.name if tool_call is not None else "",
                        "arguments": "",
                        "status": "in_progress",
                    },
                }
                _log_stream_event(event)
                emitted.append(f"data: {json.dumps(event)}\n\n")
            state.sequence_number += 1
            event = {
                "type": "response.function_call_arguments.delta",
                "sequence_number": state.sequence_number,
                "output_index": output_index,
                "item_id": stream_event.item_id,
                "delta": stream_event.delta or "",
            }
            _log_stream_event(event)
            emitted.append(f"data: {json.dumps(event)}\n\n")
        elif stream_event.event_type == "text_delta" and stream_event.delta is not None:
            if not state.message_opened:
                state.message_opened = True
                state.sequence_number += 1
                event = {
                    "type": "response.output_item.added",
                    "sequence_number": state.sequence_number,
                    "output_index": state.message_output_index,
                    "item": {
                        "id": state.message_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                        "phase": "final_answer",
                    },
                }
                _log_stream_event(event)
                emitted.append(f"data: {json.dumps(event)}\n\n")
                state.sequence_number += 1
                event = {
                    "type": "response.content_part.added",
                    "sequence_number": state.sequence_number,
                    "output_index": state.message_output_index,
                    "item_id": state.message_id,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": "",
                        "annotations": [],
                        "logprobs": [],
                    },
                }
                _log_stream_event(event)
                emitted.append(f"data: {json.dumps(event)}\n\n")
            state.final_text_parts.append(stream_event.delta)
            state.sequence_number += 1
            event = {
                "type": "response.output_text.delta",
                "sequence_number": state.sequence_number,
                "output_index": state.message_output_index,
                "item_id": state.message_id,
                "content_index": 0,
                "delta": stream_event.delta,
                "logprobs": [],
            }
            _log_stream_event(event)
            emitted.append(f"data: {json.dumps(event)}\n\n")
    return emitted


def finalize_response_stream(state: ResponseStreamFormatterState) -> list[str]:
    emitted: list[str] = []
    final_text = "".join(state.final_text_parts)
    if state.completed_item is None:
        return emitted
    state.completed_item["content"][0]["text"] = final_text
    message = (
        state.response.output_messages[0]
        if state.response.output_messages
        else CanonicalMessage(role="assistant")
    )
    reasoning_text = (
        message.reasoning_content
        if state.response.metadata.get("reasoning_summary") is not None
        else None
    )
    reasoning_item = None
    if reasoning_text is not None:
        reasoning_item = _build_reasoning_item(
            reasoning_text,
            include_encrypted_content=_wants_reasoning_encrypted_content(state.response),
        )
    tool_calls = _extract_response_tool_calls(state.response)
    if tool_calls:
        state.message_output_index = len(tool_calls) + (1 if reasoning_item is not None else 0)
    if not state.message_opened:
        state.message_opened = True
        state.sequence_number += 1
        event = {
            "type": "response.output_item.added",
            "sequence_number": state.sequence_number,
            "output_index": state.message_output_index,
            "item": {
                "id": state.message_id,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
                "phase": "final_answer",
            },
        }
        _log_stream_event(event)
        emitted.append(f"data: {json.dumps(event)}\n\n")
    for output_index, tool_call in enumerate(tool_calls):
        state.sequence_number += 1
        event = {
            "type": "response.output_item.added",
            "sequence_number": state.sequence_number,
            "output_index": output_index,
            "item": {
                "id": tool_call.id,
                "type": "function_call",
                "call_id": tool_call.id,
                "name": tool_call.name,
                "arguments": "",
                "status": "in_progress",
            },
        }
        _log_stream_event(event)
        emitted.append(f"data: {json.dumps(event)}\n\n")
        for stream_event in state.response.metadata.get("_stream_events", []):
            if stream_event.event_type != "function_call_arguments_delta":
                continue
            if stream_event.item_id != tool_call.id:
                continue
            state.sequence_number += 1
            event = {
                "type": "response.function_call_arguments.delta",
                "sequence_number": state.sequence_number,
                "output_index": output_index,
                "item_id": tool_call.id,
                "delta": stream_event.delta or "",
            }
            _log_stream_event(event)
            emitted.append(f"data: {json.dumps(event)}\n\n")
        state.sequence_number += 1
        event = {
            "type": "response.function_call_arguments.done",
            "sequence_number": state.sequence_number,
            "output_index": output_index,
            "item_id": tool_call.id,
            "arguments": tool_call.arguments,
            "name": tool_call.name,
        }
        _log_stream_event(event)
        emitted.append(f"data: {json.dumps(event)}\n\n")
        state.sequence_number += 1
        event = {
            "type": "response.output_item.done",
            "sequence_number": state.sequence_number,
            "output_index": output_index,
            "item": {
                "id": tool_call.id,
                "type": "function_call",
                "call_id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "status": "completed",
            },
        }
        _log_stream_event(event)
        emitted.append(f"data: {json.dumps(event)}\n\n")
    if reasoning_item is not None:
        state.sequence_number += 1
        event = {
            "type": "response.output_item.added",
            "sequence_number": state.sequence_number,
            "output_index": len(tool_calls),
            "item": reasoning_item,
        }
        _log_stream_event(event)
        emitted.append(f"data: {json.dumps(event)}\n\n")
        state.sequence_number += 1
        event = {
            "type": "response.output_item.done",
            "sequence_number": state.sequence_number,
            "output_index": len(tool_calls),
            "item": reasoning_item,
        }
        _log_stream_event(event)
        emitted.append(f"data: {json.dumps(event)}\n\n")
        state.sequence_number += 1
        event = {
            "type": "response.content_part.added",
            "sequence_number": state.sequence_number,
            "output_index": state.message_output_index,
            "item_id": state.message_id,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "",
                "annotations": [],
                "logprobs": [],
            },
        }
        _log_stream_event(event)
        emitted.append(f"data: {json.dumps(event)}\n\n")
    state.sequence_number += 1
    event = {
        "type": "response.output_text.done",
        "sequence_number": state.sequence_number,
        "output_index": state.message_output_index,
        "item_id": state.message_id,
        "content_index": 0,
        "text": final_text,
        "logprobs": [],
    }
    _log_stream_event(event)
    emitted.append(f"data: {json.dumps(event)}\n\n")
    state.sequence_number += 1
    event = {
        "type": "response.content_part.done",
        "sequence_number": state.sequence_number,
        "output_index": state.message_output_index,
        "item_id": state.message_id,
        "content_index": 0,
        "part": state.completed_item["content"][0],
    }
    _log_stream_event(event)
    emitted.append(f"data: {json.dumps(event)}\n\n")
    state.sequence_number += 1
    event = {
        "type": "response.output_item.done",
        "sequence_number": state.sequence_number,
        "output_index": state.message_output_index,
        "item": state.completed_item,
    }
    _log_stream_event(event)
    emitted.append(f"data: {json.dumps(event)}\n\n")
    payload = {
        "id": state.response.response_id,
        "object": "response",
        "created_at": state.response.created_at,
        "status": "completed",
        "model": state.response.model,
        **(
            {"reasoning": state.response.metadata["_response_reasoning"]}
            if isinstance(state.response.metadata.get("_response_reasoning"), dict)
            else {}
        ),
        "output": [
            *[
                {
                    "id": tool_call.id,
                    "type": "function_call",
                    "call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "status": "completed",
                }
                for tool_call in tool_calls
            ],
            *([reasoning_item] if reasoning_item is not None else []),
            state.completed_item,
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": state.response.usage.input_tokens,
            "output_tokens": state.response.usage.output_tokens,
            "total_tokens": state.response.usage.total_tokens,
        },
        "metadata": state.public_metadata,
    }
    state.sequence_number += 1
    event = {
        "type": "response.completed",
        "sequence_number": state.sequence_number,
        "response": payload,
    }
    _log_stream_event(event)
    emitted.append(f"data: {json.dumps(event)}\n\n")
    return emitted


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


def _build_upstream_metadata(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not key.startswith("_")}


def _public_response_metadata(value: dict[str, Any]) -> dict[str, Any]:
    return _build_upstream_metadata(value)


def _normalize_reasoning(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    reasoning: dict[str, Any] = {}
    if isinstance(value.get("effort"), str):
        reasoning["effort"] = value["effort"]
    if isinstance(value.get("summary"), str):
        reasoning["summary"] = value["summary"]
    if "summary" not in reasoning and "effort" in reasoning:
        reasoning["summary"] = "auto"
    return reasoning or None


def _normalize_include(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _wants_reasoning_encrypted_content(response: CanonicalResponse) -> bool:
    include = response.metadata.get("_request_include")
    return isinstance(include, list) and "reasoning.encrypted_content" in include


def _build_reasoning_item(
    reasoning_text: str, *, include_encrypted_content: bool
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": f"rs_{uuid.uuid4().hex}",
        "type": "reasoning",
        "summary": [
            {
                "type": "summary_text",
                "text": reasoning_text,
            }
        ],
    }
    if include_encrypted_content:
        item["encrypted_content"] = reasoning_text
    return item


def _build_reasoning_payload(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    reasoning = dict(value)
    effort = reasoning.get("effort")
    if effort == "minimal":
        reasoning["effort"] = "low"
    return reasoning or None


def _normalize_protocol(value: str, default: ProtocolName) -> ProtocolName:
    if value == "oairesp":
        return "oairesp"
    if value == "oaichat":
        return "oaichat"
    if value == "anthropic":
        return "anthropic"
    return default


def _log_stream_event(event: dict[str, Any]) -> None:
    logger.bind(request_id="-", provider="-", round=0, attempt=0).debug(
        "oairesp stream event={} ", event
    )


def _extract_response_text(response: CanonicalResponse) -> str:
    texts: list[str] = []
    for message in response.output_messages:
        for part in message.content:
            if part.type == "text" and part.text:
                texts.append(part.text)
    return "".join(texts)


def _extract_response_tool_calls(response: CanonicalResponse) -> list[CanonicalToolCall]:
    calls: list[CanonicalToolCall] = []
    for message in response.output_messages:
        calls.extend(message.tool_calls)
    return calls


def _normalize_input(value: Any) -> list[CanonicalMessage]:
    if value is None:
        raise HTTPException(status_code=400, detail="input is required")
    if isinstance(value, str):
        return [
            CanonicalMessage(role="user", content=[CanonicalContentPart(type="text", text=value)])
        ]
    if not isinstance(value, list):
        raise HTTPException(status_code=400, detail="input must be a string or list")
    messages: list[CanonicalMessage] = []

    def flush_pending_tool_calls() -> None:
        nonlocal pending_tool_calls
        if not pending_tool_calls:
            return
        messages.append(
            CanonicalMessage(role="assistant", content=[], tool_calls=pending_tool_calls)
        )
        pending_tool_calls = []

    pending_tool_calls: list[CanonicalToolCall] = []
    for item in value:
        if isinstance(item, str):
            flush_pending_tool_calls()
            messages.append(
                CanonicalMessage(
                    role="user", content=[CanonicalContentPart(type="text", text=item)]
                )
            )
            continue
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="input items must be strings or objects")

        item_type = item.get("type")
        if item_type == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            if not isinstance(call_id, str) or not call_id:
                raise HTTPException(
                    status_code=400,
                    detail="function_call.call_id is required",
                )
            if not isinstance(name, str) or not name:
                raise HTTPException(
                    status_code=400,
                    detail="function_call.name is required",
                )
            arguments = item.get("arguments")
            pending_tool_calls.append(
                CanonicalToolCall(
                    id=call_id,
                    type="function",
                    name=name,
                    arguments=arguments if isinstance(arguments, str) else "",
                )
            )
            continue

        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise HTTPException(
                    status_code=400,
                    detail="function_call_output.call_id is required",
                )
            flush_pending_tool_calls()
            messages.append(
                CanonicalMessage(
                    role="tool",
                    content=[
                        CanonicalContentPart(
                            type="text",
                            text=_normalize_function_call_output(item.get("output")),
                        )
                    ],
                    tool_call_id=call_id,
                )
            )
            continue

        if item_type == "reasoning":
            continue

        if item_type is not None and item_type != "message":
            continue

        role = (
            item.get("role")
            if item.get("role") in {"system", "developer", "user", "assistant", "tool"}
            else "user"
        )
        content = item.get("content")
        tool_calls = [*pending_tool_calls, *_normalize_output_tool_calls(item)]
        pending_tool_calls = []
        if content is None:
            messages.append(CanonicalMessage(role=role, content=[], tool_calls=tool_calls))
            continue
        if isinstance(content, str):
            messages.append(
                CanonicalMessage(
                    role=role,
                    content=[CanonicalContentPart(type="text", text=content)],
                    tool_calls=tool_calls,
                )
            )
            continue
        if isinstance(content, dict):
            messages.append(
                CanonicalMessage(
                    role=role,
                    content=_normalize_output_content([content]),
                    tool_calls=tool_calls,
                )
            )
            continue
        if isinstance(content, list):
            messages.append(
                CanonicalMessage(
                    role=role,
                    content=_normalize_output_content(content),
                    tool_calls=tool_calls,
                )
            )
            continue
        raise HTTPException(status_code=400, detail="input content must be a string or list")
    flush_pending_tool_calls()
    return messages


def _normalize_function_call_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = _normalize_output_content(value)
        return "".join(part.text or "" for part in parts if part.type == "text")
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True)
    if value is None:
        return ""
    return str(value)


def _normalize_response_tools(value: Any) -> list[dict[str, Any]] | None:
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

        if isinstance(item.get("name"), str):
            parameters = _normalize_tool_parameters(item.get("parameters"))
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "description": item.get("description", ""),
                        "parameters": parameters,
                        "strict": item.get("strict", True),
                    },
                }
            )
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

        raise HTTPException(status_code=400, detail="function tool must include a name")

    return normalized


def _build_response_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in tools:
        if item.get("type") != "function":
            normalized.append(item)
            continue

        function = item.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            normalized.append(
                {
                    "type": "function",
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "parameters": _normalize_tool_parameters(function.get("parameters")),
                    "strict": function.get("strict", item.get("strict", True)),
                }
            )
            continue

        if isinstance(item.get("name"), str):
            normalized.append(
                {
                    "type": "function",
                    "name": item["name"],
                    "description": item.get("description", ""),
                    "parameters": _normalize_tool_parameters(item.get("parameters")),
                    "strict": item.get("strict", True),
                }
            )
            continue

        normalized.append(item)
    return normalized


def _normalize_output_content(content: Any) -> list[CanonicalContentPart]:
    if not isinstance(content, list):
        return []
    parts: list[CanonicalContentPart] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"input_text", "output_text", "text"} and isinstance(
            item.get("text"), str
        ):
            parts.append(CanonicalContentPart(type="text", text=item["text"]))
        elif item.get("type") == "input_image" and isinstance(item.get("image_url"), str):
            parts.append(CanonicalContentPart(type="image_url", url=item["image_url"]))
    return parts


def _normalize_output_tool_calls(item: dict[str, Any]) -> list[CanonicalToolCall]:
    tool_calls = item.get("tool_calls")
    if isinstance(tool_calls, list):
        return _normalize_chat_style_tool_calls(tool_calls)

    if item.get("type") == "function_call":
        if isinstance(item.get("call_id"), str) and isinstance(item.get("name"), str):
            arguments = item.get("arguments")
            return [
                CanonicalToolCall(
                    id=item["call_id"],
                    type="function",
                    name=item["name"],
                    arguments=arguments if isinstance(arguments, str) else "",
                )
            ]

    return []


def _normalize_chat_style_tool_calls(value: Any) -> list[CanonicalToolCall]:
    if not isinstance(value, list):
        return []
    calls: list[CanonicalToolCall] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
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


def _message_to_response_items(message: CanonicalMessage) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    if message.role == "assistant":
        items.extend(
            {
                "id": _response_function_call_item_id(tool_call.id),
                "type": "function_call",
                "call_id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "status": "completed",
            }
            for tool_call in message.tool_calls
        )

    if message.role == "tool":
        items.append(
            {
                "type": "function_call_output",
                "call_id": message.tool_call_id or f"call_{uuid.uuid4().hex}",
                "output": "".join(
                    part.text or ""
                    for part in message.content
                    if part.type == "text" and part.text is not None
                ),
            }
        )
        return items

    items.append(
        {
            "role": message.role,
            "content": [
                {
                    "type": "input_text" if message.role != "assistant" else "output_text",
                    "text": part.text,
                }
                if part.type == "text"
                else {"type": "input_image", "image_url": part.url}
                for part in message.content
            ],
        }
    )
    return items


def _response_function_call_item_id(call_id: str) -> str:
    if call_id.startswith("fc"):
        return call_id
    suffix = call_id[5:] if call_id.startswith("call_") else call_id
    return f"fc_{suffix}"


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


async def _single_line_stream(payload: dict[str, Any]) -> AsyncIterator[str]:
    yield f"data: {json.dumps(payload)}"
