from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from llmpxy.models import CanonicalRequest, CanonicalResponse, CanonicalStreamEvent


class InboundAdapter(Protocol):
    protocol_name: str

    def parse_request(
        self, payload: dict[str, Any], *, strip_unsupported_fields: bool
    ) -> CanonicalRequest: ...

    def format_response(self, response: CanonicalResponse) -> dict[str, Any]: ...

    def format_stream(
        self,
        response: CanonicalResponse,
        events: list[CanonicalStreamEvent],
    ) -> AsyncIterator[str]: ...


class OutboundAdapter(Protocol):
    protocol_name: str

    def build_request(
        self, request: CanonicalRequest, mapped_model: str
    ) -> tuple[str, dict[str, Any]]: ...

    def parse_response(self, payload: dict[str, Any], protocol_out: str) -> CanonicalResponse: ...

    async def parse_stream(
        self,
        lines: AsyncIterator[str],
        protocol_out: str,
    ) -> tuple[CanonicalResponse, list[CanonicalStreamEvent]]: ...
