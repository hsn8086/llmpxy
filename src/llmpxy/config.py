from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import tomllib
from pydantic import BaseModel, ConfigDict, Field, model_validator


ProtocolName = Literal["oairesp", "oaichat", "anthropic"]


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080


class NetworkConfig(BaseModel):
    proxy: str | None = None
    trust_env: bool = False
    connect_timeout_seconds: float = 30.0
    read_timeout_seconds: float = 120.0
    write_timeout_seconds: float = 120.0
    pool_timeout_seconds: float = 30.0


class ApiKeyConfig(BaseModel):
    uuid: str
    name: str
    key: str
    enabled: bool = True
    limit_usd: float | None = None
    provider_limits_usd: dict[str, float] = Field(default_factory=dict)
    group_limits_usd: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_limit(self) -> "ApiKeyConfig":
        if self.limit_usd is not None and self.limit_usd < 0:
            raise ValueError("api_keys.limit_usd must be >= 0")
        for provider_name, limit in self.provider_limits_usd.items():
            if limit < 0:
                raise ValueError(f"api_keys.provider_limits_usd[{provider_name!r}] must be >= 0")
        for group_name, limit in self.group_limits_usd.items():
            if limit < 0:
                raise ValueError(f"api_keys.group_limits_usd[{group_name!r}] must be >= 0")
        return self


class PricingConfig(BaseModel):
    input_per_million_tokens_usd: float
    cached_input_per_million_tokens_usd: float | None = None
    output_per_million_tokens_usd: float

    @model_validator(mode="after")
    def validate_pricing(self) -> "PricingConfig":
        if self.input_per_million_tokens_usd < 0:
            raise ValueError("pricing.input_per_million_tokens_usd must be >= 0")
        if (
            self.cached_input_per_million_tokens_usd is not None
            and self.cached_input_per_million_tokens_usd < 0
        ):
            raise ValueError("pricing.cached_input_per_million_tokens_usd must be >= 0")
        if self.output_per_million_tokens_usd < 0:
            raise ValueError("pricing.output_per_million_tokens_usd must be >= 0")
        return self


class ProviderPricingConfig(BaseModel):
    default: PricingConfig | None = None
    models: dict[str, PricingConfig] = Field(default_factory=dict)


class ProviderConfig(BaseModel):
    name: str
    protocol: ProtocolName
    base_url: str
    api_key_env: str
    timeout_seconds: float = 120.0
    proxy: str | None = None
    anthropic_version: str = "2023-06-01"
    model_whitelist_only: bool = False
    models: dict[str, str] = Field(default_factory=dict)
    pricing: ProviderPricingConfig = Field(default_factory=ProviderPricingConfig)

    def api_key(self) -> str:
        value = os.getenv(self.api_key_env)
        if not value:
            raise ValueError(f"Environment variable {self.api_key_env!r} is not set")
        return value

    def map_model(self, requested_model: str) -> str:
        return self.models.get(requested_model, requested_model)

    def supports_model(self, requested_model: str) -> bool:
        if not self.model_whitelist_only:
            return True
        return requested_model in self.models

    def resolve_pricing(self, *model_names: str | None) -> PricingConfig | None:
        for model_name in model_names:
            if model_name is None:
                continue
            pricing = self.pricing.models.get(model_name)
            if pricing is not None:
                return pricing
        return self.pricing.default


class ProviderGroupConfig(BaseModel):
    name: str
    strategy: Literal["fallback", "load_balance"] = "fallback"
    members: list[str] = Field(default_factory=list)
    model_whitelist_only: bool = False
    models: list[str] = Field(default_factory=list)

    def supports_model(self, requested_model: str) -> bool:
        if not self.model_whitelist_only:
            return True
        return requested_model in self.models


class RouteConfig(BaseModel):
    type: Literal["provider", "group"] = "provider"
    name: str


class ProxyConfig(BaseModel):
    strip_unsupported_fields: bool = True
    log_level: str = "info"


class RetryConfig(BaseModel):
    provider_error_threshold: int = 3
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    max_rounds: int = 5


class LoggingConfig(BaseModel):
    dir: str = "./logs"
    info_file: str = "info.log"
    debug_file: str = "debug.log"
    rotation: str = "100 MB"
    retention: str = "10 days"


class AdminConfig(BaseModel):
    enabled: bool = False
    token: str | None = None

    @model_validator(mode="after")
    def validate_admin(self) -> "AdminConfig":
        if self.enabled and not self.token:
            raise ValueError("admin.token is required when admin.enabled=true")
        return self


class StorageConfig(BaseModel):
    backend: str = "sqlite"
    sqlite_path: str = "./data/llmpxy.db"
    file_dir: str = "./data/conversations"
    ttl_seconds: int = 604800


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    route: RouteConfig
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    api_keys: list[ApiKeyConfig] = Field(default_factory=list)
    providers: list[ProviderConfig]
    provider_groups: list[ProviderGroupConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_all(self) -> "AppConfig":
        if self.storage.backend not in {"sqlite", "file"}:
            raise ValueError("storage.backend must be either 'sqlite' or 'file'")
        if self.storage.backend == "sqlite" and not self.storage.sqlite_path:
            raise ValueError("storage.sqlite_path is required for sqlite backend")
        if self.storage.backend == "file" and not self.storage.file_dir:
            raise ValueError("storage.file_dir is required for file backend")
        if not self.providers:
            raise ValueError("At least one provider must be configured")

        api_key_names = [api_key.name for api_key in self.api_keys]
        if len(set(api_key_names)) != len(api_key_names):
            raise ValueError("API key names must be unique")
        api_key_uuids = [api_key.uuid for api_key in self.api_keys]
        if len(set(api_key_uuids)) != len(api_key_uuids):
            raise ValueError("API key UUIDs must be unique")

        provider_names = [provider.name for provider in self.providers]
        if len(set(provider_names)) != len(provider_names):
            raise ValueError("Provider names must be unique")

        group_names = [group.name for group in self.provider_groups]
        if len(set(group_names)) != len(group_names):
            raise ValueError("Provider group names must be unique")

        name_overlap = set(provider_names) & set(group_names)
        if name_overlap:
            raise ValueError(
                f"Names cannot be reused by provider and group: {sorted(name_overlap)}"
            )

        known_names = set(provider_names) | set(group_names)
        for group in self.provider_groups:
            if not group.members:
                raise ValueError(f"Provider group {group.name!r} must include at least one member")
            missing = [member for member in group.members if member not in known_names]
            if missing:
                raise ValueError(
                    f"Provider group {group.name!r} references unknown members: {missing}"
                )
            if group.model_whitelist_only and not group.models:
                raise ValueError(
                    f"Provider group {group.name!r} must declare models when model_whitelist_only=true"
                )

        if self.route.type == "provider" and self.route.name not in set(provider_names):
            raise ValueError(f"route.name {self.route.name!r} does not match a provider")
        if self.route.type == "group" and self.route.name not in set(group_names):
            raise ValueError(f"route.name {self.route.name!r} does not match a provider group")

        provider_name_set = set(provider_names)
        group_name_set = set(group_names)
        for api_key in self.api_keys:
            unknown_providers = sorted(set(api_key.provider_limits_usd) - provider_name_set)
            if unknown_providers:
                raise ValueError(
                    f"API key {api_key.name!r} references unknown providers: {unknown_providers}"
                )
            unknown_groups = sorted(set(api_key.group_limits_usd) - group_name_set)
            if unknown_groups:
                raise ValueError(
                    f"API key {api_key.name!r} references unknown groups: {unknown_groups}"
                )

        if self.retry.provider_error_threshold < 1:
            raise ValueError("retry.provider_error_threshold must be >= 1")
        if self.retry.base_backoff_seconds <= 0:
            raise ValueError("retry.base_backoff_seconds must be > 0")
        if self.retry.max_backoff_seconds < 0:
            raise ValueError("retry.max_backoff_seconds must be >= 0")
        if self.retry.max_rounds < 1:
            raise ValueError("retry.max_rounds must be >= 1")

        self._validate_group_cycles()
        return self

    def _validate_group_cycles(self) -> None:
        group_map = {group.name: group for group in self.provider_groups}
        visited: set[str] = set()
        active: set[str] = set()

        def walk(group_name: str) -> None:
            if group_name in active:
                raise ValueError(f"Provider groups contain a cycle at {group_name!r}")
            if group_name in visited:
                return
            visited.add(group_name)
            active.add(group_name)
            for member in group_map[group_name].members:
                if member in group_map:
                    walk(member)
            active.remove(group_name)

        for group in self.provider_groups:
            walk(group.name)

    def provider_map(self) -> dict[str, ProviderConfig]:
        return {provider.name: provider for provider in self.providers}

    def provider_group_map(self) -> dict[str, ProviderGroupConfig]:
        return {group.name: group for group in self.provider_groups}

    def resolve_sqlite_path(self, config_path: Path) -> Path:
        return (config_path.parent / self.storage.sqlite_path).resolve()

    def resolve_file_dir(self, config_path: Path) -> Path:
        return (config_path.parent / self.storage.file_dir).resolve()

    def resolve_log_dir(self, config_path: Path) -> Path:
        return (config_path.parent / self.logging.dir).resolve()


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    _load_default_env_files(config_path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    return AppConfig.model_validate(raw)


def _load_default_env_files(config_path: Path) -> None:
    cwd_env = Path.cwd() / ".env"
    config_env = config_path.parent / ".env"

    if cwd_env.exists():
        _load_env_file(cwd_env)

    if config_env != cwd_env and config_env.exists():
        _load_env_file(config_env)


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _strip_env_quotes(value)


def _strip_env_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
