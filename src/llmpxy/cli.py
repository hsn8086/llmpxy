from __future__ import annotations

import secrets
import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
import typer
import tomlkit
import uvicorn

from llmpxy.app import create_app, create_runtime_managed_app
from llmpxy.config import AppConfig, load_config
from llmpxy.dispatcher import ProviderDispatcher
from llmpxy.logging_utils import configure_logging
from llmpxy.remote_dashboard import run_remote_dashboard
from llmpxy.runtime import RuntimeManager, build_store

cli = typer.Typer(no_args_is_help=True)
config_cli = typer.Typer(help="Manage llmpxy config")
remote_cli = typer.Typer(help="Remote admin operations")
_RUNTIME_CONFIG_ENV = "LLMPXY_CONFIG_PATH"
cli.add_typer(config_cli, name="config")
cli.add_typer(remote_cli, name="remote")


def create_runtime_app():
    config_path = os.environ.get(_RUNTIME_CONFIG_ENV)
    if not config_path:
        raise RuntimeError(f"{_RUNTIME_CONFIG_ENV} is not set")
    resolved_config = Path(config_path).resolve()
    loaded = load_config(resolved_config)
    configure_logging(loaded, resolved_config)
    store = build_store(loaded, resolved_config)
    dispatcher = ProviderDispatcher(loaded)
    return create_app(loaded, store, dispatcher)


def create_managed_runtime_app():
    config_path = os.environ.get(_RUNTIME_CONFIG_ENV)
    if not config_path:
        raise RuntimeError(f"{_RUNTIME_CONFIG_ENV} is not set")
    runtime = RuntimeManager(Path(config_path))
    return create_runtime_managed_app(runtime)


@cli.command()
def serve(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    reload: bool = typer.Option(False, "--reload/--no-reload"),
) -> None:
    loaded = load_config(config)
    resolved_config = config.resolve()
    os.environ[_RUNTIME_CONFIG_ENV] = str(resolved_config)
    uvicorn.run(
        "llmpxy.cli:create_managed_runtime_app",
        factory=True,
        host=loaded.server.host,
        port=loaded.server.port,
        log_level=loaded.proxy.log_level,
        reload=reload,
    )


@config_cli.command("validate")
def validate_config(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
) -> None:
    loaded = load_config(config)
    typer.echo(
        f"Config valid: providers={len(loaded.providers)} groups={len(loaded.provider_groups)} api_keys={len(loaded.api_keys)}"
    )


@config_cli.command("list-api-keys")
def list_api_keys(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
) -> None:
    loaded = load_config(config)
    if not loaded.api_keys:
        typer.echo("No API keys configured")
        return
    for api_key in loaded.api_keys:
        typer.echo(
            f"{api_key.uuid}\t{api_key.name}\tlimit_usd={api_key.limit_usd if api_key.limit_usd is not None else 'unlimited'}"
        )


@config_cli.command("add-api-key")
def add_api_key(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    name: str = typer.Option(..., "--name"),
    limit_usd: float | None = typer.Option(None, "--limit-usd"),
    provider_limit: list[str] = typer.Option(None, "--provider-limit"),
    group_limit: list[str] = typer.Option(None, "--group-limit"),
    key: str | None = typer.Option(None, "--key"),
) -> None:
    config_path = config.resolve()
    document = tomlkit.parse(config_path.read_text(encoding="utf-8"))

    api_keys = document.get("api_keys")
    if api_keys is None:
        api_keys = tomlkit.aot()
        document["api_keys"] = api_keys

    generated_uuid = str(uuid.uuid4())
    generated_key = key or secrets.token_urlsafe(32)
    api_key_item = tomlkit.table()
    api_key_item["uuid"] = generated_uuid
    api_key_item["name"] = name
    api_key_item["key"] = generated_key
    if limit_usd is not None:
        api_key_item["limit_usd"] = limit_usd

    provider_limits_table = _parse_named_float_options(provider_limit)
    if provider_limits_table:
        provider_limits_item = tomlkit.inline_table()
        for provider_name, value in provider_limits_table.items():
            provider_limits_item[provider_name] = value
        api_key_item["provider_limits_usd"] = provider_limits_item

    group_limits_table = _parse_named_float_options(group_limit)
    if group_limits_table:
        group_limits_item = tomlkit.inline_table()
        for group_name, value in group_limits_table.items():
            group_limits_item[group_name] = value
        api_key_item["group_limits_usd"] = group_limits_item

    api_keys.append(api_key_item)
    config_path.write_text(tomlkit.dumps(document), encoding="utf-8")
    load_config(config_path)
    typer.echo(f"Added API key: uuid={generated_uuid} name={name} key={generated_key}")


@config_cli.command("show-balance")
def show_balance(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    api_key_uuid: str | None = typer.Option(None, "--uuid"),
    name: str | None = typer.Option(None, "--name"),
) -> None:
    loaded = load_config(config)
    store = build_store(loaded, config.resolve())
    dispatcher = ProviderDispatcher(loaded)

    api_keys = loaded.api_keys
    if api_key_uuid is not None:
        api_keys = [api_key for api_key in api_keys if api_key.uuid == api_key_uuid]
    if name is not None:
        api_keys = [api_key for api_key in api_keys if api_key.name == name]
    if not api_keys:
        raise typer.BadParameter("No matching API keys found")

    for index, api_key in enumerate(api_keys):
        if index > 0:
            typer.echo("")
        total_used = store.get_api_key_total_cost(api_key.uuid)
        typer.echo(f"API Key: {api_key.name} ({api_key.uuid})")
        typer.echo(f"  Total: {_format_budget_line(total_used, api_key.limit_usd)}")

        for provider_name, limit_usd in sorted(api_key.provider_limits_usd.items()):
            used = store.get_api_key_total_cost(api_key.uuid, [provider_name])
            typer.echo(f"  Provider {provider_name}: {_format_budget_line(used, limit_usd)}")

        for group_name, limit_usd in sorted(api_key.group_limits_usd.items()):
            used = store.get_api_key_total_cost(
                api_key.uuid,
                dispatcher.providers_for_group(group_name),
            )
            typer.echo(f"  Group {group_name}: {_format_budget_line(used, limit_usd)}")


def _parse_named_float_options(items: list[str] | None) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for item in items or []:
        name, separator, raw_value = item.partition("=")
        if not separator or not name.strip() or not raw_value.strip():
            raise typer.BadParameter(f"Invalid limit option: {item!r}, expected name=value")
        parsed[name.strip()] = float(raw_value)
    return parsed


def _format_budget_line(used_usd: float, limit_usd: float | None) -> str:
    if limit_usd is None:
        return f"used=${used_usd:.6f}, remaining=unlimited"
    remaining_usd = max(limit_usd - used_usd, 0.0)
    return f"used=${used_usd:.6f}, limit=${limit_usd:.6f}, remaining=${remaining_usd:.6f}"


@remote_cli.command("status")
def remote_status(
    base_url: str = typer.Option(..., "--base-url"),
    admin_token: str = typer.Option(..., "--admin-token"),
) -> None:
    payload = _remote_get(base_url, admin_token, "/admin/status")
    typer.echo(json_dumps(payload))


@remote_cli.command("config-get")
def remote_config_get(
    base_url: str = typer.Option(..., "--base-url"),
    admin_token: str = typer.Option(..., "--admin-token"),
) -> None:
    payload = _remote_get(base_url, admin_token, "/admin/config")
    typer.echo(json_dumps(payload))


@remote_cli.command("dashboard")
def remote_dashboard(
    base_url: str = typer.Option(..., "--base-url"),
    admin_token: str = typer.Option(..., "--admin-token"),
) -> None:
    run_remote_dashboard(base_url, admin_token)


@remote_cli.command("reload")
def remote_reload(
    base_url: str = typer.Option(..., "--base-url"),
    admin_token: str = typer.Option(..., "--admin-token"),
) -> None:
    payload = _remote_post(base_url, admin_token, "/admin/config/reload", {})
    typer.echo(json_dumps(payload))


@remote_cli.command("retry-update")
def remote_retry_update(
    base_url: str = typer.Option(..., "--base-url"),
    admin_token: str = typer.Option(..., "--admin-token"),
    provider_error_threshold: int | None = typer.Option(None, "--provider-error-threshold"),
    base_backoff_seconds: float | None = typer.Option(None, "--base-backoff-seconds"),
    max_backoff_seconds: float | None = typer.Option(None, "--max-backoff-seconds"),
    max_rounds: int | None = typer.Option(None, "--max-rounds"),
) -> None:
    payload = {
        "provider_error_threshold": provider_error_threshold,
        "base_backoff_seconds": base_backoff_seconds,
        "max_backoff_seconds": max_backoff_seconds,
        "max_rounds": max_rounds,
    }
    result = _remote_patch(base_url, admin_token, "/admin/config/retry", payload)
    typer.echo(json_dumps(result))


@remote_cli.command("route-update")
def remote_route_update(
    base_url: str = typer.Option(..., "--base-url"),
    admin_token: str = typer.Option(..., "--admin-token"),
    route_type: str | None = typer.Option(None, "--type"),
    name: str | None = typer.Option(None, "--name"),
) -> None:
    result = _remote_patch(
        base_url,
        admin_token,
        "/admin/config/route",
        {"type": route_type, "name": name},
    )
    typer.echo(json_dumps(result))


@remote_cli.command("api-key-add")
def remote_api_key_add(
    base_url: str = typer.Option(..., "--base-url"),
    admin_token: str = typer.Option(..., "--admin-token"),
    name: str = typer.Option(..., "--name"),
    limit_usd: float | None = typer.Option(None, "--limit-usd"),
    provider_limit: list[str] = typer.Option(None, "--provider-limit"),
    group_limit: list[str] = typer.Option(None, "--group-limit"),
    key: str | None = typer.Option(None, "--key"),
) -> None:
    payload: dict[str, Any] = {
        "name": name,
        "key": key,
        "limit_usd": limit_usd,
        "provider_limits_usd": _parse_named_float_options(provider_limit),
        "group_limits_usd": _parse_named_float_options(group_limit),
    }
    result = _remote_post(base_url, admin_token, "/admin/api-keys", payload)
    typer.echo(json_dumps(result))


@remote_cli.command("api-key-list")
def remote_api_key_list(
    base_url: str = typer.Option(..., "--base-url"),
    admin_token: str = typer.Option(..., "--admin-token"),
) -> None:
    result = _remote_get(base_url, admin_token, "/admin/api-keys")
    typer.echo(json_dumps(result))


@remote_cli.command("api-key-enable")
def remote_api_key_enable(
    base_url: str = typer.Option(..., "--base-url"),
    admin_token: str = typer.Option(..., "--admin-token"),
    api_key_uuid: str = typer.Option(..., "--uuid"),
) -> None:
    result = _remote_patch(
        base_url,
        admin_token,
        f"/admin/api-keys/{api_key_uuid}",
        {"enabled": True},
    )
    typer.echo(json_dumps(result))


@remote_cli.command("api-key-rotate")
def remote_api_key_rotate(
    base_url: str = typer.Option(..., "--base-url"),
    admin_token: str = typer.Option(..., "--admin-token"),
    api_key_uuid: str = typer.Option(..., "--uuid"),
    key: str | None = typer.Option(None, "--key"),
) -> None:
    next_key = key or secrets.token_urlsafe(32)
    result = _remote_patch(
        base_url,
        admin_token,
        f"/admin/api-keys/{api_key_uuid}",
        {"key": next_key},
    )
    payload = {
        "uuid": api_key_uuid,
        "key": next_key,
        "config": result,
    }
    typer.echo(json_dumps(payload))


@remote_cli.command("api-key-update")
def remote_api_key_update(
    base_url: str = typer.Option(..., "--base-url"),
    admin_token: str = typer.Option(..., "--admin-token"),
    api_key_uuid: str = typer.Option(..., "--uuid"),
    name: str | None = typer.Option(None, "--name"),
    key: str | None = typer.Option(None, "--key"),
    limit_usd: float | None = typer.Option(None, "--limit-usd"),
    enabled: bool | None = typer.Option(None, "--enabled/--disabled"),
    provider_limit: list[str] = typer.Option(None, "--provider-limit"),
    group_limit: list[str] = typer.Option(None, "--group-limit"),
) -> None:
    payload = {
        "name": name,
        "key": key,
        "limit_usd": limit_usd,
        "enabled": enabled,
        "provider_limits_usd": _parse_named_float_options(provider_limit)
        if provider_limit
        else None,
        "group_limits_usd": _parse_named_float_options(group_limit) if group_limit else None,
    }
    result = _remote_patch(base_url, admin_token, f"/admin/api-keys/{api_key_uuid}", payload)
    typer.echo(json_dumps(result))


@remote_cli.command("api-key-disable")
def remote_api_key_disable(
    base_url: str = typer.Option(..., "--base-url"),
    admin_token: str = typer.Option(..., "--admin-token"),
    api_key_uuid: str = typer.Option(..., "--uuid"),
) -> None:
    result = _remote_patch(
        base_url,
        admin_token,
        f"/admin/api-keys/{api_key_uuid}",
        {"enabled": False},
    )
    typer.echo(json_dumps(result))


@remote_cli.command("provider-list")
def remote_provider_list(
    base_url: str = typer.Option(..., "--base-url"),
    admin_token: str = typer.Option(..., "--admin-token"),
) -> None:
    result = _remote_get(base_url, admin_token, "/admin/providers")
    typer.echo(json_dumps(result))


@remote_cli.command("provider-update")
def remote_provider_update(
    base_url: str = typer.Option(..., "--base-url"),
    admin_token: str = typer.Option(..., "--admin-token"),
    provider_name: str = typer.Option(..., "--name"),
    base_url_value: str | None = typer.Option(None, "--provider-base-url"),
    timeout_seconds: float | None = typer.Option(None, "--timeout-seconds"),
    proxy: str | None = typer.Option(None, "--proxy"),
    model_whitelist_only: bool | None = typer.Option(
        None, "--model-whitelist-only/--no-model-whitelist-only"
    ),
) -> None:
    payload = {
        "base_url": base_url_value,
        "timeout_seconds": timeout_seconds,
        "proxy": proxy,
        "model_whitelist_only": model_whitelist_only,
    }
    result = _remote_patch(base_url, admin_token, f"/admin/providers/{provider_name}", payload)
    typer.echo(json_dumps(result))


@remote_cli.command("group-update")
def remote_group_update(
    base_url: str = typer.Option(..., "--base-url"),
    admin_token: str = typer.Option(..., "--admin-token"),
    group_name: str = typer.Option(..., "--name"),
    strategy: str | None = typer.Option(None, "--strategy"),
    model_whitelist_only: bool | None = typer.Option(
        None, "--model-whitelist-only/--no-model-whitelist-only"
    ),
    model: list[str] = typer.Option(None, "--model"),
    member: list[str] = typer.Option(None, "--member"),
) -> None:
    payload = {
        "strategy": strategy,
        "model_whitelist_only": model_whitelist_only,
        "models": model or None,
        "members": member or None,
    }
    result = _remote_patch(
        base_url,
        admin_token,
        f"/admin/provider-groups/{group_name}",
        payload,
    )
    typer.echo(json_dumps(result))


@remote_cli.command("group-list")
def remote_group_list(
    base_url: str = typer.Option(..., "--base-url"),
    admin_token: str = typer.Option(..., "--admin-token"),
) -> None:
    result = _remote_get(base_url, admin_token, "/admin/provider-groups")
    typer.echo(json_dumps(result))


def _remote_get(base_url: str, admin_token: str, path: str) -> dict[str, object]:
    with httpx.Client(headers=_admin_headers(admin_token)) as client:
        response = client.get(f"{base_url.rstrip('/')}{path}")
        response.raise_for_status()
        return response.json()


def _remote_post(
    base_url: str, admin_token: str, path: str, payload: dict[str, Any]
) -> dict[str, object]:
    with httpx.Client(headers=_admin_headers(admin_token)) as client:
        response = client.post(f"{base_url.rstrip('/')}{path}", json=payload)
        response.raise_for_status()
        return response.json()


def _remote_patch(
    base_url: str, admin_token: str, path: str, payload: dict[str, Any]
) -> dict[str, object]:
    with httpx.Client(headers=_admin_headers(admin_token)) as client:
        response = client.patch(f"{base_url.rstrip('/')}{path}", json=payload)
        response.raise_for_status()
        return response.json()


def _admin_headers(admin_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {admin_token}"}


def json_dumps(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2)


def main() -> None:
    cli()
