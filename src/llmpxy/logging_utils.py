from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from llmpxy.config import AppConfig

if TYPE_CHECKING:
    from loguru import Record


REDACTED_KEYS = {"authorization", "api_key", "apikey", "token", "secret", "password"}


def configure_logging(config: AppConfig, config_path: Path) -> None:
    log_dir = config.resolve_log_dir(config_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level="INFO", enqueue=True, backtrace=True, diagnose=True)
    logger.add(
        log_dir / config.logging.info_file,
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        filter=_is_info_record,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra[request_id]} | "
            "provider={extra[provider]} | round={extra[round]} | attempt={extra[attempt]} | {message}"
        ),
    )
    logger.add(
        log_dir / config.logging.debug_file,
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        filter=_is_debug_record,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra[request_id]} | "
            "provider={extra[provider]} | round={extra[round]} | attempt={extra[attempt]} | {message}"
        ),
    )


def bind_logger(
    request_id: str = "-",
    provider: str = "-",
    round_number: int = 0,
    attempt: int = 0,
):
    return logger.bind(
        request_id=request_id,
        provider=provider,
        round=round_number,
        attempt=attempt,
    )


def sanitize_for_logging(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = key.lower().replace("-", "_")
            if normalized_key in REDACTED_KEYS or normalized_key.endswith("_key"):
                sanitized[key] = "***REDACTED***"
                continue
            sanitized[key] = sanitize_for_logging(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_for_logging(item) for item in value]
    if isinstance(value, str):
        if len(value) <= 200:
            return value
        return f"{value[:200]}...(truncated {len(value) - 200} chars)"
    return value


def _is_info_record(record: "Record") -> bool:
    return record["level"].name != "DEBUG"


def _is_debug_record(record: "Record") -> bool:
    return record["level"].name == "DEBUG"
