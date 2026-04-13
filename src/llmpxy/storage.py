from __future__ import annotations

from typing import Protocol

from llmpxy.models import StoredConversation


class ConversationStore(Protocol):
    def get(self, response_id: str) -> StoredConversation | None: ...

    def put(self, conversation: StoredConversation) -> None: ...

    def delete_expired(self) -> int: ...
