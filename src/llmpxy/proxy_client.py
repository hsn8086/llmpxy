from __future__ import annotations

from collections.abc import AsyncIterator
import json
from urllib.parse import urlparse
from typing import Any

import httpx
from loguru import logger

from llmpxy.config import AppConfig, ProviderConfig


class ProviderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        provider_name: str | None = None,
        retryable: bool,
        status_code: int | None = None,
        response_text: str | None = None,
        error_code: str | None = None,
        base_url: str | None = None,
        proxy: str | None = None,
        path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider_name = provider_name
        self.retryable = retryable
        self.status_code = status_code
        self.response_text = response_text
        self.error_code = error_code
        self.base_url = base_url
        self.proxy = proxy
        self.path = path


def create_async_client(
    config: AppConfig, provider: ProviderConfig | None = None
) -> httpx.AsyncClient:
    proxy = provider.proxy if provider and provider.proxy else config.network.proxy
    timeout = httpx.Timeout(
        connect=config.network.connect_timeout_seconds,
        read=config.network.read_timeout_seconds,
        write=config.network.write_timeout_seconds,
        pool=config.network.pool_timeout_seconds,
    )
    return httpx.AsyncClient(proxy=proxy, timeout=timeout, trust_env=config.network.trust_env)


def resolve_proxy(config: AppConfig, provider: ProviderConfig) -> str | None:
    return provider.proxy if provider.proxy else config.network.proxy


def build_headers(provider: ProviderConfig) -> dict[str, str]:
    if provider.protocol == "anthropic":
        return {
            "x-api-key": provider.api_key(),
            "anthropic-version": provider.anthropic_version,
            "Content-Type": "application/json",
        }
    return {
        "Authorization": f"Bearer {provider.api_key()}",
        "Content-Type": "application/json",
    }


def _build_target_url(provider: ProviderConfig, path: str) -> str:
    base_url = provider.base_url.rstrip("/")
    normalized_path = path if path.startswith("/") else f"/{path}"
    if (
        provider.protocol == "anthropic"
        and base_url.endswith("/v1")
        and normalized_path.startswith("/v1/")
    ):
        normalized_path = normalized_path[3:]
    return f"{base_url}{normalized_path}"


async def open_stream(
    client: httpx.AsyncClient,
    provider: ProviderConfig,
    path: str,
    payload: dict[str, Any],
) -> httpx.Response:
    target_url = _build_target_url(provider, path)
    proxy = getattr(client, "_proxy", None)
    try:
        response = await client.send(
            client.build_request(
                "POST",
                target_url,
                headers=build_headers(provider),
                json=payload,
            ),
            stream=True,
        )
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        raise ProviderError(
            str(exc),
            provider_name=provider.name,
            retryable=True,
            base_url=provider.base_url,
            proxy=str(proxy) if proxy is not None else None,
            path=path,
        ) from exc
    if response.status_code >= 400:
        body = await response.aread()
        await response.aclose()
        response_text = body.decode("utf-8")
        retryable, error_code = _classify_error_response(response.status_code, response_text)
        raise ProviderError(
            f"Provider {provider.name} returned HTTP {response.status_code}",
            provider_name=provider.name,
            retryable=retryable,
            status_code=response.status_code,
            response_text=response_text,
            error_code=error_code,
            base_url=provider.base_url,
            proxy=str(proxy) if proxy is not None else None,
            path=path,
        )
    return response


async def post_json(
    client: httpx.AsyncClient,
    provider: ProviderConfig,
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    proxy = getattr(client, "_proxy", None)
    try:
        response = await client.post(
            _build_target_url(provider, path),
            headers=build_headers(provider),
            json=payload,
            timeout=provider.timeout_seconds,
        )
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        raise ProviderError(
            str(exc),
            provider_name=provider.name,
            retryable=True,
            base_url=provider.base_url,
            proxy=str(proxy) if proxy is not None else None,
            path=path,
        ) from exc

    if response.status_code >= 400:
        retryable, error_code = _classify_error_response(response.status_code, response.text)
        raise ProviderError(
            f"Provider {provider.name} returned HTTP {response.status_code}",
            provider_name=provider.name,
            retryable=retryable,
            status_code=response.status_code,
            response_text=response.text,
            error_code=error_code,
            base_url=provider.base_url,
            proxy=str(proxy) if proxy is not None else None,
            path=path,
        )
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        sse_response = _parse_sse_json_response(response.text)
        if sse_response is not None:
            return sse_response
        raise ProviderError(
            f"Provider {provider.name} returned non-JSON response",
            provider_name=provider.name,
            retryable=False,
            status_code=response.status_code,
            response_text=response.text,
            base_url=provider.base_url,
            proxy=str(proxy) if proxy is not None else None,
            path=path,
        ) from exc


async def stream_lines(
    client: httpx.AsyncClient,
    provider: ProviderConfig,
    path: str,
    payload: dict[str, Any],
) -> AsyncIterator[str]:
    target_url = _build_target_url(provider, path)
    response = await open_stream(client, provider, path, payload)
    try:
        async for line in response.aiter_lines():
            if line:
                logger.bind(request_id="-", provider=provider.name, round=0, attempt=0).debug(
                    "upstream stream line host={} path={} line={}",
                    urlparse(target_url).netloc,
                    path,
                    line,
                )
                yield line
    finally:
        await response.aclose()


def _is_retryable_status(status_code: int) -> bool:
    if status_code in {408, 429}:
        return True
    return 500 <= status_code <= 599


def _classify_error_response(status_code: int, response_text: str) -> tuple[bool, str | None]:
    error_code: str | None = None
    message = response_text

    try:
        payload = json.loads(response_text)
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                if isinstance(error.get("code"), str):
                    error_code = error["code"]
                if isinstance(error.get("message"), str):
                    message = error["message"]
    except json.JSONDecodeError:
        pass

    normalized_message = message.lower()
    if error_code in {"model_not_found", "invalid_model", "unsupported_model"}:
        return False, error_code
    if "no available channel for model" in normalized_message:
        return False, error_code
    return _is_retryable_status(status_code), error_code


def _parse_sse_json_response(response_text: str) -> dict[str, Any] | None:
    if "event:" not in response_text or "data:" not in response_text:
        return None

    last_response: dict[str, Any] | None = None
    output_items: list[dict[str, Any]] = []
    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        try:
            payload = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        item = payload.get("item")
        if payload.get("type") == "response.output_item.done" and isinstance(item, dict):
            output_items.append(item)
        response = payload.get("response")
        if isinstance(response, dict):
            last_response = response
        elif payload.get("type") == "response.completed":
            return payload

    if last_response is not None and output_items and not last_response.get("output"):
        last_response = {**last_response, "output": output_items}
    return last_response
