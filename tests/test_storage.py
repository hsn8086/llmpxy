from __future__ import annotations

import time
from pathlib import Path

from llmpxy.models import ApiKeyUsageRecord, RequestEventRecord, StoredConversation
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


def test_sqlite_store_tracks_api_key_usage_costs(tmp_path: Path) -> None:
    store = SQLiteConversationStore(tmp_path / "usage.db")
    store.put_api_key_usage(
        ApiKeyUsageRecord(
            api_key_uuid="key-uuid-a",
            api_key_name="client-a",
            request_id="req-1",
            provider_name="provider-a",
            requested_model="gpt-4.1",
            upstream_model="gpt-4.1",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=1.25,
            created_at=int(time.time()),
        )
    )
    store.put_api_key_usage(
        ApiKeyUsageRecord(
            api_key_uuid="key-uuid-a",
            api_key_name="client-a",
            request_id="req-2",
            provider_name="provider-b",
            requested_model="claude-sonnet",
            upstream_model="claude-sonnet-4-6",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=2.75,
            created_at=int(time.time()),
        )
    )

    assert store.get_api_key_total_cost("key-uuid-a") == 4.0
    assert store.get_api_key_total_cost("key-uuid-a", ["provider-a"]) == 1.25


def test_file_store_tracks_api_key_usage_costs(tmp_path: Path) -> None:
    store = FileConversationStore(tmp_path / "conversations")
    store.put_api_key_usage(
        ApiKeyUsageRecord(
            api_key_uuid="key-uuid-a",
            api_key_name="client-a",
            request_id="req-1",
            provider_name="provider-a",
            requested_model="gpt-4.1",
            upstream_model="gpt-4.1",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=1.0,
            created_at=int(time.time()),
        )
    )
    store.put_api_key_usage(
        ApiKeyUsageRecord(
            api_key_uuid="key-uuid-a",
            api_key_name="client-a",
            request_id="req-2",
            provider_name="provider-b",
            requested_model="gpt-4.1",
            upstream_model="gpt-4.1-mini",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=3.0,
            created_at=int(time.time()),
        )
    )

    assert store.get_api_key_total_cost("key-uuid-a") == 4.0
    assert store.get_api_key_total_cost("key-uuid-a", ["provider-b"]) == 3.0


def test_stores_can_list_request_events_since(tmp_path: Path) -> None:
    stores = [
        SQLiteConversationStore(tmp_path / "events.db"),
        FileConversationStore(tmp_path / "events-files"),
    ]
    now = int(time.time())

    for store in stores:
        store.put_request_event(
            RequestEventRecord(
                request_id="req-old",
                started_at=now - 120,
                finished_at=now - 120,
                latency_ms=100,
                protocol_in="oaichat",
                status="success",
                http_status=200,
                cost_usd=1.0,
            )
        )
        store.put_request_event(
            RequestEventRecord(
                request_id="req-new",
                started_at=now - 5,
                finished_at=now - 5,
                latency_ms=200,
                protocol_in="oaichat",
                status="provider_error",
                http_status=502,
                error_message="timeout",
                cost_usd=0.0,
            )
        )

        recent = store.list_request_events_since(now - 60)
        assert [item.request_id for item in recent] == ["req-new"]
