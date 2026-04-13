from __future__ import annotations

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


def _build_store(config: AppConfig, config_path: Path):
    if config.storage.backend == "sqlite":
        return SQLiteConversationStore(config.resolve_sqlite_path(config_path))
    return FileConversationStore(config.resolve_file_dir(config_path))


@cli.command()
def serve(
    config: Path = typer.Option(..., "--config", exists=True, file_okay=True, dir_okay=False),
) -> None:
    loaded = load_config(config)
    configure_logging(loaded, config.resolve())
    store = _build_store(loaded, config.resolve())
    dispatcher = ProviderDispatcher(loaded)
    app = create_app(loaded, store, dispatcher)
    uvicorn.run(
        app, host=loaded.server.host, port=loaded.server.port, log_level=loaded.proxy.log_level
    )


def main() -> None:
    cli()
