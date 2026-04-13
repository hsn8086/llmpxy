from __future__ import annotations

import time
from pathlib import Path

from llmpxy.models import StoredConversation
from llmpxy.storage_file import FileConversationStore
from llmpxy.storage_sqlite import SQLiteConversationStore


def _conversation(response_id: str, expires_at: int | None = None) -> StoredConversation:
    return StoredConversation(
        response_id=response_id,
        created_at=int(time.time()),
        model="test-model",
        messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        response_payload={"id": response_id},
        metadata={"k": "v"},
        expires_at=expires_at,
    )


def test_sqlite_store_roundtrip(tmp_path: Path) -> None:
    store = SQLiteConversationStore(tmp_path / "conv.db")
    conversation = _conversation("resp_1")
    store.put(conversation)

    loaded = store.get("resp_1")
    assert loaded is not None
    assert loaded.response_id == "resp_1"
    assert loaded.metadata == {"k": "v"}


def test_file_store_roundtrip(tmp_path: Path) -> None:
    store = FileConversationStore(tmp_path / "conversations")
    conversation = _conversation("resp_2")
    store.put(conversation)

    loaded = store.get("resp_2")
    assert loaded is not None
    assert loaded.response_id == "resp_2"


def test_expired_file_store_entry_deleted(tmp_path: Path) -> None:
    store = FileConversationStore(tmp_path / "conversations")
    conversation = _conversation("resp_3", expires_at=int(time.time()) - 1)
    store.put(conversation)

    assert store.get("resp_3") is None
