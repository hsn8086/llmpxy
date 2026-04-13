from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from llmpxy.models import StoredConversation


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
