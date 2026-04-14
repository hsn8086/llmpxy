from __future__ import annotations

import secrets
import uuid
from pathlib import Path
from typing import Any

import tomlkit

from llmpxy.config import AppConfig, load_config


def update_admin_config(config_path: Path, patch: dict[str, Any]) -> AppConfig:
    document = _load_document(config_path)
    admin = document.get("admin")
    if admin is None:
        admin = tomlkit.table()
        document["admin"] = admin
    for key, value in patch.items():
        admin[key] = value
    return _write_and_validate(config_path, document)


def update_route_config(config_path: Path, patch: dict[str, Any]) -> AppConfig:
    document = _load_document(config_path)
    route = document["route"]
    for key, value in patch.items():
        route[key] = value
    return _write_and_validate(config_path, document)


def update_retry_config(config_path: Path, patch: dict[str, Any]) -> AppConfig:
    document = _load_document(config_path)
    retry = document.get("retry")
    if retry is None:
        retry = tomlkit.table()
        document["retry"] = retry
    for key, value in patch.items():
        retry[key] = value
    return _write_and_validate(config_path, document)


def update_provider_group(config_path: Path, group_name: str, patch: dict[str, Any]) -> AppConfig:
    document = _load_document(config_path)
    for group in document.get("provider_groups", []):
        if group.get("name") != group_name:
            continue
        for key, value in patch.items():
            group[key] = value
        return _write_and_validate(config_path, document)
    raise ValueError(f"Unknown provider group: {group_name}")


def update_provider(config_path: Path, provider_name: str, patch: dict[str, Any]) -> AppConfig:
    document = _load_document(config_path)
    for provider in document.get("providers", []):
        if provider.get("name") != provider_name:
            continue
        for key, value in patch.items():
            provider[key] = value
        return _write_and_validate(config_path, document)
    raise ValueError(f"Unknown provider: {provider_name}")


def add_api_key(config_path: Path, payload: dict[str, Any]) -> tuple[AppConfig, dict[str, str]]:
    document = _load_document(config_path)
    api_keys = document.get("api_keys")
    if api_keys is None:
        api_keys = tomlkit.aot()
        document["api_keys"] = api_keys

    generated_uuid = str(uuid.uuid4())
    generated_key = payload.get("key") or secrets.token_urlsafe(32)
    item = tomlkit.table()
    item["uuid"] = generated_uuid
    item["name"] = payload["name"]
    item["key"] = generated_key
    if payload.get("limit_usd") is not None:
        item["limit_usd"] = payload["limit_usd"]
    if payload.get("enabled") is not None:
        item["enabled"] = payload["enabled"]
    _assign_inline_table(item, "provider_limits_usd", payload.get("provider_limits_usd"))
    _assign_inline_table(item, "group_limits_usd", payload.get("group_limits_usd"))
    api_keys.append(item)
    config = _write_and_validate(config_path, document)
    return config, {"uuid": generated_uuid, "key": generated_key}


def update_api_key(config_path: Path, api_key_uuid: str, patch: dict[str, Any]) -> AppConfig:
    document = _load_document(config_path)
    for item in document.get("api_keys", []):
        if item.get("uuid") != api_key_uuid:
            continue
        for key, value in patch.items():
            if key in {"provider_limits_usd", "group_limits_usd"}:
                _assign_inline_table(item, key, value)
            else:
                item[key] = value
        return _write_and_validate(config_path, document)
    raise ValueError(f"Unknown API key: {api_key_uuid}")


def mask_config(config: AppConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json")
    for api_key in payload.get("api_keys", []):
        key = api_key.get("key")
        if isinstance(key, str):
            api_key["key"] = _mask_secret(key)
    admin = payload.get("admin")
    if isinstance(admin, dict):
        token = admin.get("token")
        if isinstance(token, str):
            admin["token"] = _mask_secret(token)
    return payload


def _assign_inline_table(item: Any, key: str, values: dict[str, float] | None) -> None:
    if values is None:
        return
    inline_table = tomlkit.inline_table()
    for name, value in values.items():
        inline_table[name] = value
    item[key] = inline_table


def _load_document(config_path: Path):
    return tomlkit.parse(config_path.read_text(encoding="utf-8"))


def _write_and_validate(config_path: Path, document: Any) -> AppConfig:
    original = config_path.read_text(encoding="utf-8")
    rendered = tomlkit.dumps(document)
    config_path.write_text(rendered, encoding="utf-8")
    try:
        return load_config(config_path)
    except Exception:
        config_path.write_text(original, encoding="utf-8")
        raise


def _mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"
