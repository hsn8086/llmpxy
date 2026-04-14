from __future__ import annotations

from typing import Protocol

from llmpxy.models import ApiKeyUsageRecord, RequestEventRecord, StoredConversation


class ConversationStore(Protocol):
    def get(self, response_id: str) -> StoredConversation | None: ...

    def put(self, conversation: StoredConversation) -> None: ...

    def delete_expired(self) -> int: ...

    def get_api_key_total_cost(
        self, api_key_uuid: str, provider_names: list[str] | None = None
    ) -> float: ...

    def put_api_key_usage(self, record: ApiKeyUsageRecord) -> None: ...

    def put_request_event(self, record: RequestEventRecord) -> None: ...

    def list_recent_request_events(self, limit: int) -> list[RequestEventRecord]: ...

    def list_request_events_since(
        self, started_at: int, limit: int | None = None
    ) -> list[RequestEventRecord]: ...
