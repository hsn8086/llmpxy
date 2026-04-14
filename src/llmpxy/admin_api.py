from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llmpxy.config import AppConfig
from llmpxy.config_mutation import (
    add_api_key,
    mask_config,
    update_api_key,
    update_provider,
    update_provider_group,
    update_retry_config,
    update_route_config,
)
from llmpxy.runtime import RuntimeManager


class RetryPatch(BaseModel):
    provider_error_threshold: int | None = None
    base_backoff_seconds: float | None = None
    max_backoff_seconds: float | None = None
    max_rounds: int | None = None


class RoutePatch(BaseModel):
    type: str | None = None
    name: str | None = None


class ProviderPatch(BaseModel):
    base_url: str | None = None
    timeout_seconds: float | None = None
    proxy: str | None = None
    model_whitelist_only: bool | None = None
    models: dict[str, str] | None = None


class ProviderGroupPatch(BaseModel):
    strategy: str | None = None
    model_whitelist_only: bool | None = None
    models: list[str] | None = None
    members: list[str] | None = None


class ApiKeyCreateRequest(BaseModel):
    name: str
    key: str | None = None
    enabled: bool = True
    limit_usd: float | None = None
    provider_limits_usd: dict[str, float] = Field(default_factory=dict)
    group_limits_usd: dict[str, float] = Field(default_factory=dict)


class ApiKeyUpdateRequest(BaseModel):
    name: str | None = None
    key: str | None = None
    enabled: bool | None = None
    limit_usd: float | None = None
    provider_limits_usd: dict[str, float] | None = None
    group_limits_usd: dict[str, float] | None = None


def create_admin_router(runtime: RuntimeManager, config_path: Path) -> APIRouter:
    router = APIRouter(prefix="/admin", tags=["admin"])

    def require_admin(authorization: str | None = Header(default=None)) -> None:
        config = runtime.current().config
        if not config.admin.enabled:
            raise HTTPException(status_code=403, detail="Admin API disabled")
        expected = config.admin.token
        if not expected:
            raise HTTPException(status_code=503, detail="Admin token not configured")
        if authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="Invalid admin token")

    @router.get("/status", dependencies=[Depends(require_admin)])
    async def admin_status() -> dict[str, object]:
        return runtime.runtime_snapshot()

    @router.get("/config", dependencies=[Depends(require_admin)])
    async def admin_config() -> dict[str, object]:
        return mask_config(runtime.current().config)

    @router.post("/config/reload", dependencies=[Depends(require_admin)])
    async def admin_reload() -> dict[str, object]:
        runtime.force_reload()
        await runtime.publish_config_event({"action": "reload"})
        return {"status": "reloaded", "reload": runtime.runtime_snapshot()["reload"]}

    @router.patch("/config/retry", dependencies=[Depends(require_admin)])
    async def admin_update_retry(patch: RetryPatch) -> dict[str, object]:
        config = update_retry_config(config_path, patch.model_dump(exclude_none=True))
        runtime.force_reload()
        await runtime.publish_config_event({"action": "update_retry"})
        return mask_config(config)

    @router.patch("/config/route", dependencies=[Depends(require_admin)])
    async def admin_update_route(patch: RoutePatch) -> dict[str, object]:
        config = update_route_config(config_path, patch.model_dump(exclude_none=True))
        runtime.force_reload()
        await runtime.publish_config_event({"action": "update_route"})
        return mask_config(config)

    @router.patch("/providers/{provider_name}", dependencies=[Depends(require_admin)])
    async def admin_update_provider(provider_name: str, patch: ProviderPatch) -> dict[str, object]:
        config = update_provider(config_path, provider_name, patch.model_dump(exclude_none=True))
        runtime.force_reload()
        await runtime.publish_config_event({"action": "update_provider", "provider": provider_name})
        return mask_config(config)

    @router.patch("/provider-groups/{group_name}", dependencies=[Depends(require_admin)])
    async def admin_update_group(group_name: str, patch: ProviderGroupPatch) -> dict[str, object]:
        config = update_provider_group(config_path, group_name, patch.model_dump(exclude_none=True))
        runtime.force_reload()
        await runtime.publish_config_event({"action": "update_group", "group": group_name})
        return mask_config(config)

    @router.get("/api-keys", dependencies=[Depends(require_admin)])
    async def admin_api_keys() -> list[dict[str, object]]:
        snapshot = runtime.runtime_snapshot()
        return cast(list[dict[str, object]], snapshot["api_keys"])

    @router.get("/providers", dependencies=[Depends(require_admin)])
    async def admin_providers() -> list[dict[str, object]]:
        snapshot = runtime.runtime_snapshot()
        return cast(list[dict[str, object]], snapshot["providers"])

    @router.get("/provider-groups", dependencies=[Depends(require_admin)])
    async def admin_provider_groups() -> list[dict[str, object]]:
        config = runtime.current().config
        return [group.model_dump(mode="json") for group in config.provider_groups]

    @router.post("/api-keys", dependencies=[Depends(require_admin)])
    async def admin_add_api_key(payload: ApiKeyCreateRequest) -> dict[str, object]:
        config, created = add_api_key(config_path, payload.model_dump(mode="python"))
        runtime.force_reload()
        await runtime.publish_config_event({"action": "add_api_key", **created})
        return {"config": mask_config(config), "created": created}

    @router.patch("/api-keys/{api_key_uuid}", dependencies=[Depends(require_admin)])
    async def admin_update_api_key(
        api_key_uuid: str, payload: ApiKeyUpdateRequest
    ) -> dict[str, object]:
        config = update_api_key(config_path, api_key_uuid, payload.model_dump(exclude_none=True))
        runtime.force_reload()
        await runtime.publish_config_event({"action": "update_api_key", "uuid": api_key_uuid})
        return mask_config(config)

    @router.get("/dashboard/snapshot", dependencies=[Depends(require_admin)])
    async def admin_dashboard_snapshot() -> dict[str, object]:
        return runtime.runtime_snapshot()

    @router.get("/dashboard/stream", dependencies=[Depends(require_admin)])
    async def admin_dashboard_stream() -> StreamingResponse:
        async def event_stream() -> AsyncIterator[str]:
            cursor = 0
            while True:
                events = await runtime.stats().wait_for_events(cursor)
                if not events:
                    yield ": keepalive\n\n"
                    continue
                cursor += len(events)
                for event in events:
                    yield f"event: {event.event_type}\n"
                    yield f"data: {json.dumps(event.model_dump(mode='json'), ensure_ascii=True)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return router
