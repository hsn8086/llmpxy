from __future__ import annotations

import time
from pathlib import Path

import json

from llmpxy.models import ApiKeyUsageRecord, RequestEventRecord, StoredConversation


class FileConversationStore:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)
        self._usage_directory = self._directory / "api_key_usage"
        self._usage_directory.mkdir(parents=True, exist_ok=True)
        self._events_directory = self._directory / "request_events"
        self._events_directory.mkdir(parents=True, exist_ok=True)

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

    def get_api_key_total_cost(
        self, api_key_uuid: str, provider_names: list[str] | None = None
    ) -> float:
        total_cost = 0.0
        for path in self._usage_directory.glob("*.json"):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            record = ApiKeyUsageRecord.model_validate(payload)
            if record.api_key_uuid != api_key_uuid:
                continue
            if provider_names is not None and record.provider_name not in provider_names:
                continue
            total_cost += record.cost_usd
        return total_cost

    def put_api_key_usage(self, record: ApiKeyUsageRecord) -> None:
        path = self._usage_directory / f"{record.request_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(record.model_dump(mode="json"), handle, ensure_ascii=True)

    def put_request_event(self, record: RequestEventRecord) -> None:
        path = self._events_directory / f"{record.request_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(record.model_dump(mode="json"), handle, ensure_ascii=True)

    def list_recent_request_events(self, limit: int) -> list[RequestEventRecord]:
        records: list[RequestEventRecord] = []
        for path in self._events_directory.glob("*.json"):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            records.append(RequestEventRecord.model_validate(payload))
        records.sort(key=lambda item: item.started_at, reverse=True)
        return records[:limit]

    def list_request_events_since(
        self, started_at: int, limit: int | None = None
    ) -> list[RequestEventRecord]:
        records: list[RequestEventRecord] = []
        for path in self._events_directory.glob("*.json"):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            record = RequestEventRecord.model_validate(payload)
            if record.started_at < started_at:
                continue
            records.append(record)
        records.sort(key=lambda item: item.started_at, reverse=True)
        if limit is None:
            return records
        return records[:limit]
