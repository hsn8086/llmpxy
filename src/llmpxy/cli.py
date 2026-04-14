from __future__ import annotations

import os
from pathlib import Path

import typer
import uvicorn

from llmpxy.app import create_app
from llmpxy.config import AppConfig, load_config
from llmpxy.dispatcher import ProviderDispatcher
from llmpxy.logging_utils import configure_logging
from llmpxy.storage_file import FileConversationStore
from llmpxy.storage_sqlite import SQLiteConversationStore

cli = typer.Typer(no_args_is_help=True)
_RUNTIME_CONFIG_ENV = "LLMPXY_CONFIG_PATH"


def _build_store(config: AppConfig, config_path: Path):
    if config.storage.backend == "sqlite":
        return SQLiteConversationStore(config.resolve_sqlite_path(config_path))
    return FileConversationStore(config.resolve_file_dir(config_path))


def create_runtime_app():
    config_path = os.environ.get(_RUNTIME_CONFIG_ENV)
    if not config_path:
        raise RuntimeError(f"{_RUNTIME_CONFIG_ENV} is not set")
    resolved_config = Path(config_path).resolve()
    loaded = load_config(resolved_config)
    configure_logging(loaded, resolved_config)
    store = _build_store(loaded, resolved_config)
    dispatcher = ProviderDispatcher(loaded)
    return create_app(loaded, store, dispatcher)


@cli.command()
def serve(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
    reload: bool = typer.Option(False, "--reload/--no-reload"),
) -> None:
    loaded = load_config(config)
    resolved_config = config.resolve()
    os.environ[_RUNTIME_CONFIG_ENV] = str(resolved_config)
    uvicorn.run(
        "llmpxy.cli:create_runtime_app",
        factory=True,
        host=loaded.server.host,
        port=loaded.server.port,
        log_level=loaded.proxy.log_level,
        reload=reload,
    )


def main() -> None:
    cli()
