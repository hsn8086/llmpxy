from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from llmpxy.config import ProtocolName


class CanonicalContentPart(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    url: str | None = None


class CanonicalToolCall(BaseModel):
    id: str
    type: str = "function"
    name: str
    arguments: str = ""


class CanonicalMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: list[CanonicalContentPart] = Field(default_factory=list)
    tool_calls: list[CanonicalToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    reasoning_content: str | None = None


class CanonicalUsage(BaseModel):
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class CanonicalRequest(BaseModel):
    protocol_in: ProtocolName
    requested_model: str
    stream: bool = False
    system_prompt: str | None = None
    messages: list[CanonicalMessage] = Field(default_factory=list)
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    reasoning: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    conversation_reference: str | None = None
    raw_request: dict[str, Any] = Field(default_factory=dict)


class CanonicalResponse(BaseModel):
    response_id: str
    protocol_out: ProtocolName
    model: str
    created_at: int
    status: str = "completed"
    output_messages: list[CanonicalMessage] = Field(default_factory=list)
    usage: CanonicalUsage = Field(default_factory=CanonicalUsage)
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] = Field(default_factory=dict)


class CanonicalStreamEvent(BaseModel):
    event_type: str
    response_id: str
    item_id: str | None = None
    delta: str | None = None
    usage: CanonicalUsage | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class StoredConversation(BaseModel):
    response_id: str
    created_at: int
    model: str
    messages: list[dict[str, Any]]
    response_payload: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    expires_at: int | None = None


class ApiKeyUsageRecord(BaseModel):
    api_key_uuid: str
    api_key_name: str
    request_id: str
    provider_name: str
    requested_model: str
    upstream_model: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    created_at: int


class RequestEventRecord(BaseModel):
    request_id: str
    started_at: int
    finished_at: int | None = None
    latency_ms: int | None = None
    protocol_in: ProtocolName
    stream: bool = False
    api_key_uuid: str | None = None
    api_key_name: str | None = None
    provider_name: str | None = None
    requested_model: str | None = None
    upstream_model: str | None = None
    status: Literal[
        "success",
        "provider_error",
        "all_failed",
        "auth_error",
        "budget_rejected",
        "bad_request",
    ]
    http_status: int
    error_code: str | None = None
    error_message: str | None = None
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class AdminEvent(BaseModel):
    event_type: Literal["request", "reload", "config"]
    created_at: int
    payload: dict[str, Any] = Field(default_factory=dict)
