from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import json

from llmpxy.models import ApiKeyUsageRecord, RequestEventRecord, StoredConversation


class SQLiteConversationStore:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    response_id TEXT PRIMARY KEY,
                    created_at INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    expires_at INTEGER
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS api_key_usage (
                    request_id TEXT PRIMARY KEY,
                    api_key_uuid TEXT NOT NULL,
                    api_key_name TEXT NOT NULL,
                    provider_name TEXT NOT NULL,
                    requested_model TEXT NOT NULL,
                    upstream_model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS request_events (
                    request_id TEXT PRIMARY KEY,
                    started_at INTEGER NOT NULL,
                    finished_at INTEGER,
                    latency_ms INTEGER,
                    protocol_in TEXT NOT NULL,
                    stream INTEGER NOT NULL,
                    api_key_uuid TEXT,
                    api_key_name TEXT,
                    provider_name TEXT,
                    requested_model TEXT,
                    upstream_model TEXT,
                    status TEXT NOT NULL,
                    http_status INTEGER NOT NULL,
                    error_code TEXT,
                    error_message TEXT,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL
                )
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(api_key_usage)").fetchall()
            }
            if "api_key_uuid" not in columns:
                connection.execute(
                    "ALTER TABLE api_key_usage ADD COLUMN api_key_uuid TEXT NOT NULL DEFAULT ''"
                )

    def get(self, response_id: str) -> StoredConversation | None:
        self.delete_expired()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM responses WHERE response_id = ?",
                (response_id,),
            ).fetchone()
        if row is None:
            return None
        return StoredConversation(
            response_id=row["response_id"],
            created_at=row["created_at"],
            model=row["model"],
            messages=json.loads(row["messages_json"]),
            response_payload=json.loads(row["response_json"]),
            metadata=json.loads(row["metadata_json"]),
            expires_at=row["expires_at"],
        )

    def put(self, conversation: StoredConversation) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO responses (
                    response_id,
                    created_at,
                    model,
                    messages_json,
                    response_json,
                    metadata_json,
                    expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation.response_id,
                    conversation.created_at,
                    conversation.model,
                    json.dumps(conversation.messages),
                    json.dumps(conversation.response_payload),
                    json.dumps(conversation.metadata),
                    conversation.expires_at,
                ),
            )

    def delete_expired(self) -> int:
        now = int(time.time())
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM responses WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            )
            return cursor.rowcount

    def get_api_key_total_cost(
        self, api_key_uuid: str, provider_names: list[str] | None = None
    ) -> float:
        sql = "SELECT COALESCE(SUM(cost_usd), 0) AS total_cost FROM api_key_usage WHERE api_key_uuid = ?"
        params: list[str] = [api_key_uuid]
        if provider_names:
            placeholders = ", ".join("?" for _ in provider_names)
            sql += f" AND provider_name IN ({placeholders})"
            params.extend(provider_names)
        with self._connect() as connection:
            row = connection.execute(
                sql,
                tuple(params),
            ).fetchone()
        if row is None:
            return 0.0
        return float(row["total_cost"])

    def put_api_key_usage(self, record: ApiKeyUsageRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO api_key_usage (
                    request_id,
                    api_key_uuid,
                    api_key_name,
                    provider_name,
                    requested_model,
                    upstream_model,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cost_usd,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.request_id,
                    record.api_key_uuid,
                    record.api_key_name,
                    record.provider_name,
                    record.requested_model,
                    record.upstream_model,
                    record.input_tokens,
                    record.output_tokens,
                    record.total_tokens,
                    record.cost_usd,
                    record.created_at,
                ),
            )

    def put_request_event(self, record: RequestEventRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO request_events (
                    request_id,
                    started_at,
                    finished_at,
                    latency_ms,
                    protocol_in,
                    stream,
                    api_key_uuid,
                    api_key_name,
                    provider_name,
                    requested_model,
                    upstream_model,
                    status,
                    http_status,
                    error_code,
                    error_message,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.request_id,
                    record.started_at,
                    record.finished_at,
                    record.latency_ms,
                    record.protocol_in,
                    int(record.stream),
                    record.api_key_uuid,
                    record.api_key_name,
                    record.provider_name,
                    record.requested_model,
                    record.upstream_model,
                    record.status,
                    record.http_status,
                    record.error_code,
                    record.error_message,
                    record.input_tokens,
                    record.output_tokens,
                    record.total_tokens,
                    record.cost_usd,
                ),
            )

    def list_recent_request_events(self, limit: int) -> list[RequestEventRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM request_events ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_request_event_from_row(row) for row in rows]

    def list_request_events_since(
        self, started_at: int, limit: int | None = None
    ) -> list[RequestEventRecord]:
        sql = "SELECT * FROM request_events WHERE started_at >= ? ORDER BY started_at DESC"
        params: tuple[int, ...] | tuple[int, int]
        params = (started_at,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (started_at, limit)
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_request_event_from_row(row) for row in rows]


def _request_event_from_row(row: sqlite3.Row) -> RequestEventRecord:
    return RequestEventRecord(
        request_id=row["request_id"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        latency_ms=row["latency_ms"],
        protocol_in=row["protocol_in"],
        stream=bool(row["stream"]),
        api_key_uuid=row["api_key_uuid"],
        api_key_name=row["api_key_name"],
        provider_name=row["provider_name"],
        requested_model=row["requested_model"],
        upstream_model=row["upstream_model"],
        status=row["status"],
        http_status=row["http_status"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        total_tokens=row["total_tokens"],
        cost_usd=float(row["cost_usd"]),
    )
