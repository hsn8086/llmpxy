from __future__ import annotations

from llmpxy.logging_utils import sanitize_for_logging


def test_sanitize_for_logging_redacts_secrets_and_truncates() -> None:
    payload = {
        "Authorization": "Bearer secret-token",
        "nested": {"api_key": "abc123", "note": "x" * 250},
        "list": [{"password": "pwd"}],
    }

    sanitized = sanitize_for_logging(payload)

    assert sanitized["Authorization"] == "***REDACTED***"
    assert sanitized["nested"]["api_key"] == "***REDACTED***"
    assert sanitized["list"][0]["password"] == "***REDACTED***"
    assert "truncated" in sanitized["nested"]["note"]
