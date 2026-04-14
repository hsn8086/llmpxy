from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException

from llmpxy.config import ApiKeyConfig, AppConfig


@dataclass(frozen=True)
class AuthenticatedApiKey:
    uuid: str
    name: str
    key: str
    limit_usd: float | None
    provider_limits_usd: dict[str, float]
    group_limits_usd: dict[str, float]


class ApiKeyRegistry:
    def __init__(self, config: AppConfig) -> None:
        self._api_keys_by_value: dict[str, AuthenticatedApiKey] = {}
        for api_key in config.api_keys:
            self._register(api_key)

    def _register(self, api_key: ApiKeyConfig) -> None:
        if not api_key.enabled:
            return
        key_value = api_key.key
        if key_value in self._api_keys_by_value:
            raise ValueError("API key values must be unique")
        self._api_keys_by_value[key_value] = AuthenticatedApiKey(
            uuid=api_key.uuid,
            name=api_key.name,
            key=key_value,
            limit_usd=api_key.limit_usd,
            provider_limits_usd=dict(api_key.provider_limits_usd),
            group_limits_usd=dict(api_key.group_limits_usd),
        )

    def authenticate(self, authorization: str | None) -> AuthenticatedApiKey:
        if not self._api_keys_by_value:
            raise HTTPException(status_code=503, detail="No inbound API keys configured")
        token = _extract_bearer_token(authorization)
        if token is None:
            raise HTTPException(status_code=401, detail="Missing Authorization bearer token")
        api_key = self._api_keys_by_value.get(token)
        if api_key is None:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return api_key


def _extract_bearer_token(authorization: str | None) -> str | None:
    if authorization is None:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()
