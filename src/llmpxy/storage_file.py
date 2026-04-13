from __future__ import annotations

import json
import time
from pathlib import Path

from llmpxy.models import StoredConversation


class FileConversationStore:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    def _path_for(self, response_id: str) -> Path:
        return self._directory / f"{response_id}.json"

    def get(self, response_id: str) -> StoredConversation | None:
        self.delete_expired()
        path = self._path_for(response_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        conversation = StoredConversation.model_validate(payload)
        if conversation.expires_at is not None and conversation.expires_at < int(time.time()):
            path.unlink(missing_ok=True)
            return None
        return conversation

    def put(self, conversation: StoredConversation) -> None:
        path = self._path_for(conversation.response_id)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(conversation.model_dump(mode="json"), handle, ensure_ascii=True)

    def delete_expired(self) -> int:
        deleted = 0
        now = int(time.time())
        for path in self._directory.glob("*.json"):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            expires_at = payload.get("expires_at")
            if expires_at is not None and expires_at < now:
                path.unlink(missing_ok=True)
                deleted += 1
        return deleted
