from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from urllib.parse import urlparse
from typing import Any, AsyncIterator

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


CONFIG_PATH = os.environ.get("MINI_FALLBACK_PROXY_CONFIG")
DEFAULT_TIMEOUT = 60.0
DEFAULT_STICKY_TTL_SECONDS = 1800


def _normalize_upstream_model(model_name: str) -> str:
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def _coerce_optional_timeout(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def _coerce_headers(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(header_value) for key, header_value in value.items()}


def _safe_json_loads(body: bytes) -> dict[str, Any] | None:
    try:
        loaded = json.loads(body)
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def _iter_sse_events(body_text: str) -> list[tuple[str | None, str]]:
    events: list[tuple[str | None, str]] = []
    for block in re.split(r"\r?\n\r?\n", body_text):
        block = block.strip()
        if not block:
            continue

        event_name: str | None = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip() or None
                continue
            if line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())

        if data_lines:
            events.append((event_name, "\n".join(data_lines)))

    return events


def _extract_responses_json_from_sse(body: bytes) -> dict[str, Any] | None:
    body_text = body.decode("utf-8", errors="replace")
    completed_response: dict[str, Any] | None = None

    for event_name, data in _iter_sse_events(body_text):
        if data == "[DONE]":
            continue

        try:
            payload = json.loads(data)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        if event_name == "response.completed" or payload.get("type") == "response.completed":
            response = payload.get("response")
            if isinstance(response, dict):
                completed_response = response

    return completed_response


def _extract_error_code(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, str):
            return code
    code = payload.get("code")
    if isinstance(code, str):
        return code
    return None


def _extract_error_message(payload: dict[str, Any] | None, fallback: str) -> str:
    if not payload:
        return fallback
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message
    if isinstance(error, str):
        return error
    message = payload.get("message")
    if isinstance(message, str):
        return message
    return fallback


def _extract_session_key(request: Request, payload: dict[str, Any]) -> str | None:
    header_value = request.headers.get("x-fallback-session")
    if header_value:
        return f"header|{header_value}"

    candidates = (
        ("conversation_id", payload.get("conversation_id")),
        ("thread_id", payload.get("thread_id")),
        ("previous_response_id", payload.get("previous_response_id")),
        ("user", payload.get("user")),
    )
    for name, value in candidates:
        if isinstance(value, str) and value.strip():
            return f"{name}|{value.strip()}"

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for name in ("conversation_id", "thread_id", "session_id", "user"):
            value = metadata.get(name)
            if isinstance(value, str) and value.strip():
                return f"metadata:{name}|{value.strip()}"

    return None


def classify_transport_error(exc: Exception) -> FailureDecision:
    if isinstance(
        exc,
        (
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
            httpx.NetworkError,
            httpx.ProtocolError,
        ),
    ):
        return FailureDecision(
            failure_class=FailureClass.AVAILABILITY,
            should_fallback=True,
            count_failure=True,
        )

    return FailureDecision(
        failure_class=FailureClass.AVAILABILITY,
        should_fallback=True,
        count_failure=True,
    )


def classify_http_error(status_code: int, body: bytes, endpoint: str) -> FailureDecision:
    payload = _safe_json_loads(body)
    error_code = (_extract_error_code(payload) or "").lower()
    error_message = _extract_error_message(
        payload, body[:500].decode("utf-8", errors="replace")
    ).lower()

    if status_code in {401, 403}:
        return FailureDecision(
            failure_class=FailureClass.AUTH_OR_BALANCE,
            should_fallback=True,
            count_failure=True,
            cooldown_multiplier=3.0,
        )

    if status_code in {408, 429} or status_code >= 500:
        capability_codes = {
            "model_not_found",
            "unsupported_parameter",
            "unsupported_model",
            "not_supported",
            "endpoint_not_found",
            "unknown_model",
        }
        if error_code in capability_codes:
            return FailureDecision(
                failure_class=FailureClass.CAPABILITY_MISMATCH,
                should_fallback=True,
                count_failure=False,
            )
        return FailureDecision(
            failure_class=FailureClass.AVAILABILITY,
            should_fallback=True,
            count_failure=True,
        )

    if status_code == 404:
        return FailureDecision(
            failure_class=FailureClass.CAPABILITY_MISMATCH,
            should_fallback=True,
            count_failure=False,
        )

    if status_code in {400, 422}:
        capability_markers = (
            "model_not_found",
            "unsupported",
            "not support",
            "not_supported",
            "unknown model",
            "endpoint",
        )
        if any(marker in error_code or marker in error_message for marker in capability_markers):
            return FailureDecision(
                failure_class=FailureClass.CAPABILITY_MISMATCH,
                should_fallback=True,
                count_failure=False,
            )

        request_invalid_markers = (
            "context",
            "context_length",
            "maximum context length",
            "content policy",
            "safety",
            "invalid_request_error",
            "invalid request",
            "validation",
            "schema",
            "tool",
        )
        if any(marker in error_code or marker in error_message for marker in request_invalid_markers):
            return FailureDecision(
                failure_class=FailureClass.REQUEST_INVALID,
                should_fallback=False,
                count_failure=False,
            )

        return FailureDecision(
            failure_class=FailureClass.REQUEST_INVALID,
            should_fallback=False,
            count_failure=False,
        )

    return FailureDecision(
        failure_class=FailureClass.REQUEST_INVALID,
        should_fallback=False,
        count_failure=False,
    )


class FailureClass(str, Enum):
    AVAILABILITY = "availability"
    AUTH_OR_BALANCE = "auth_or_balance"
    CAPABILITY_MISMATCH = "capability_mismatch"
    REQUEST_INVALID = "request_invalid"
    MIDSTREAM_FAILURE = "midstream_failure"


@dataclass(frozen=True)
class FailureDecision:
    failure_class: FailureClass
    should_fallback: bool
    count_failure: bool
    cooldown_multiplier: float = 1.0


@dataclass(frozen=True)
class StickyBinding:
    provider_id: str
    expires_at: float


@dataclass(frozen=True)
class Provider:
    model_name: str
    configured_model: str
    upstream_model: str
    api_base: str
    api_key: str
    order: int
    timeout: float | None
    extra_headers: dict[str, str]

    @property
    def provider_id(self) -> str:
        return f"{self.model_name}|{self.order}|{self.api_base}|{self.upstream_model}"


def build_failure_key(provider: Provider, endpoint: str) -> str:
    return f"{provider.provider_id}|{endpoint}|{provider.model_name}"


def build_sticky_key(session_key: str, endpoint: str, model_name: str) -> str:
    return f"{session_key}|{endpoint}|{model_name}"


class RouterState:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.cooldown_until: dict[str, float] = {}
        self.fail_counts: dict[str, int] = {}
        self.last_error: dict[str, str] = {}
        self.session_bindings: dict[str, StickyBinding] = {}
        self._lock = asyncio.Lock()
        self.allowed_fails = 0
        self.cooldown_time = 300
        self.sticky_ttl_seconds = DEFAULT_STICKY_TTL_SECONDS
        self.default_timeout = DEFAULT_TIMEOUT
        self.host = "127.0.0.1"
        self.port = 8099
        self.log_level = "info"
        self.normalize_upstream_model = True
        self.providers_by_model: dict[str, list[Provider]] = {}
        self.reload()

    def reload(self) -> None:
        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        app_settings = raw.get("app_settings") or {}
        self.host = str(app_settings.get("host", "127.0.0.1"))
        self.port = int(app_settings.get("port", 8099))
        self.log_level = str(app_settings.get("log_level", "info"))
        self.default_timeout = float(app_settings.get("default_timeout", DEFAULT_TIMEOUT))
        self.sticky_ttl_seconds = int(
            app_settings.get("sticky_ttl_seconds", DEFAULT_STICKY_TTL_SECONDS)
        )
        self.normalize_upstream_model = bool(
            app_settings.get("normalize_upstream_model", True)
        )

        router_settings = raw.get("router_settings") or {}
        self.allowed_fails = int(router_settings.get("allowed_fails", 0))
        self.cooldown_time = int(router_settings.get("cooldown_time", 300))

        grouped: dict[str, list[Provider]] = {}
        for item in raw.get("model_list", []):
            model_name = item["model_name"]
            params = item["litellm_params"]
            configured_model = str(params["model"])
            upstream_model = configured_model
            if self.normalize_upstream_model:
                upstream_model = _normalize_upstream_model(configured_model)
            provider = Provider(
                model_name=model_name,
                configured_model=configured_model,
                upstream_model=upstream_model,
                api_base=str(params["api_base"]).rstrip("/"),
                api_key=params["api_key"],
                order=int(params.get("order", 100)),
                timeout=_coerce_optional_timeout(params.get("timeout")),
                extra_headers=_coerce_headers(params.get("headers")),
            )
            grouped.setdefault(model_name, []).append(provider)

        self.providers_by_model = {
            model: sorted(providers, key=lambda p: (p.order, p.api_base, p.upstream_model))
            for model, providers in grouped.items()
        }

    async def get_candidate_providers(
        self, model_name: str, endpoint: str, sticky_key: str | None
    ) -> list[Provider]:
        async with self._lock:
            providers = self.providers_by_model.get(model_name)
            if not providers:
                raise KeyError(model_name)

            now = time.time()
            healthy: list[Provider] = []
            cooling: list[Provider] = []
            for provider in providers:
                cooldown_until = self.cooldown_until.get(build_failure_key(provider, endpoint), 0)
                if cooldown_until <= now:
                    healthy.append(provider)
                else:
                    cooling.append(provider)

            # If all providers are cooling down, still return in order so the caller
            # gets a deterministic error chain instead of random selection.
            candidates = healthy or cooling
            return self._apply_sticky_preference(candidates, sticky_key)

    def _apply_sticky_preference(
        self, providers: list[Provider], sticky_key: str | None
    ) -> list[Provider]:
        if not sticky_key or len(providers) <= 1:
            return providers

        binding = self.session_bindings.get(sticky_key)
        now = time.time()
        if binding is None:
            return providers
        if binding.expires_at <= now:
            self.session_bindings.pop(sticky_key, None)
            return providers

        for index, provider in enumerate(providers):
            if provider.provider_id == binding.provider_id:
                if index == 0:
                    return providers
                return [provider, *providers[:index], *providers[index + 1 :]]

        return providers

    async def record_success(self, provider: Provider, endpoint: str) -> None:
        async with self._lock:
            failure_key = build_failure_key(provider, endpoint)
            self.fail_counts[failure_key] = 0
            self.cooldown_until.pop(failure_key, None)
            self.last_error.pop(failure_key, None)

    async def bind_session(self, sticky_key: str | None, provider: Provider) -> None:
        if not sticky_key:
            return
        async with self._lock:
            self.session_bindings[sticky_key] = StickyBinding(
                provider_id=provider.provider_id,
                expires_at=time.time() + self.sticky_ttl_seconds,
            )

    async def clear_session_binding(self, sticky_key: str | None, provider: Provider) -> None:
        if not sticky_key:
            return
        async with self._lock:
            binding = self.session_bindings.get(sticky_key)
            if binding and binding.provider_id == provider.provider_id:
                self.session_bindings.pop(sticky_key, None)

    async def record_failure(
        self,
        provider: Provider,
        endpoint: str,
        error_message: str,
        decision: FailureDecision,
    ) -> None:
        async with self._lock:
            failure_key = build_failure_key(provider, endpoint)
            self.last_error[failure_key] = error_message
            if not decision.count_failure:
                return

            count = self.fail_counts.get(failure_key, 0) + 1
            self.fail_counts[failure_key] = count
            if count > self.allowed_fails:
                cooldown_seconds = int(self.cooldown_time * decision.cooldown_multiplier)
                self.cooldown_until[failure_key] = time.time() + cooldown_seconds
                self.fail_counts[failure_key] = 0

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            now = time.time()
            providers: list[dict[str, Any]] = []
            for model_name, model_providers in self.providers_by_model.items():
                for provider in model_providers:
                    provider_id = provider.provider_id
                    responses_key = build_failure_key(provider, "/responses")
                    chat_key = build_failure_key(provider, "/chat/completions")
                    responses_cooldown = self.cooldown_until.get(responses_key)
                    chat_cooldown = self.cooldown_until.get(chat_key)
                    providers.append(
                        {
                            "model_name": model_name,
                            "provider_id": provider_id,
                            "upstream_model": provider.upstream_model,
                            "configured_model": provider.configured_model,
                            "api_base": provider.api_base,
                            "order": provider.order,
                            "timeout": provider.timeout,
                            "extra_headers": provider.extra_headers,
                            "cooldown_remaining_seconds": {
                                "/responses": max(0, int(responses_cooldown - now))
                                if responses_cooldown
                                else 0,
                                "/chat/completions": max(0, int(chat_cooldown - now))
                                if chat_cooldown
                                else 0,
                            },
                            "last_error": {
                                "/responses": self.last_error.get(responses_key),
                                "/chat/completions": self.last_error.get(chat_key),
                            },
                        }
                    )
            return {
                "config_path": self.config_path,
                "app_settings": {
                    "host": self.host,
                    "port": self.port,
                    "log_level": self.log_level,
                    "default_timeout": self.default_timeout,
                    "sticky_ttl_seconds": self.sticky_ttl_seconds,
                    "normalize_upstream_model": self.normalize_upstream_model,
                },
                "allowed_fails": self.allowed_fails,
                "cooldown_time": self.cooldown_time,
                "session_bindings": [
                    {
                        "session_key": session_key,
                        "provider_id": binding.provider_id,
                        "expires_in_seconds": max(0, int(binding.expires_at - now)),
                    }
                    for session_key, binding in self.session_bindings.items()
                    if binding.expires_at > now
                ],
                "providers": providers,
            }

    async def list_models(self) -> list[dict[str, Any]]:
        async with self._lock:
            data: list[dict[str, Any]] = []
            for model_name, providers in sorted(self.providers_by_model.items()):
                primary = providers[0]
                data.append(
                    {
                        "id": model_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": "mini-fallback-proxy",
                        "value": primary.configured_model,
                        "root": primary.configured_model,
                        "parent": None,
                        "providers": [
                            {
                                "order": provider.order,
                                "api_base": provider.api_base,
                                "configured_model": provider.configured_model,
                                "upstream_model": provider.upstream_model,
                            }
                            for provider in providers
                        ],
                    }
                )
            return data

    async def get_model(self, model_name: str) -> dict[str, Any]:
        async with self._lock:
            providers = self.providers_by_model.get(model_name)
            if not providers:
                raise KeyError(model_name)

            primary = providers[0]
            return {
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "mini-fallback-proxy",
                "value": primary.configured_model,
                "root": primary.configured_model,
                "parent": None,
                "providers": [
                    {
                        "order": provider.order,
                        "api_base": provider.api_base,
                        "configured_model": provider.configured_model,
                        "upstream_model": provider.upstream_model,
                    }
                    for provider in providers
                ],
            }


if not CONFIG_PATH:
    raise RuntimeError(
        "MINI_FALLBACK_PROXY_CONFIG is not set. Start the server with "
        "./start.sh --config /path/to/config.yaml"
    )

router_state = RouterState(os.path.expanduser(CONFIG_PATH))
app = FastAPI(title="Mini Fallback Proxy")
logger = logging.getLogger("uvicorn.error")


def log_provider_event(
    event: str,
    provider: Provider,
    detail: str = "",
) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    provider_name = urlparse(provider.api_base).netloc or provider.api_base
    message = (
        f"{timestamp} provider={provider_name} "
        f"model={provider.model_name} event={event}"
    )
    if detail:
        message = f"{message} {detail}"
    logger.info(message)


async def stream_upstream(
    response: httpx.Response,
    provider: Provider,
    endpoint: str,
    client: httpx.AsyncClient,
) -> AsyncIterator[bytes]:
    completed = False
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
        completed = True
    except Exception as exc:
        decision = FailureDecision(
            failure_class=FailureClass.MIDSTREAM_FAILURE,
            should_fallback=False,
            count_failure=True,
        )
        await router_state.record_failure(
            provider=provider,
            endpoint=endpoint,
            error_message=f"Midstream failure: {exc.__class__.__name__}: {exc}",
            decision=decision,
        )
        raise
    finally:
        if completed:
            await router_state.record_success(provider, endpoint)
            log_provider_event("success", provider, f"endpoint={endpoint} stream=true")
        if not response.is_closed:
            await response.aclose()
        await client.aclose()


def pick_timeout(payload: dict[str, Any], provider: Provider) -> float:
    timeout = payload.get("timeout")
    if isinstance(timeout, (int, float)) and timeout > 0:
        return float(timeout)
    if provider.timeout is not None:
        return provider.timeout
    return router_state.default_timeout


async def forward_request(request: Request, endpoint: str) -> Any:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

    model_name = payload.get("model")
    if not model_name:
        raise HTTPException(status_code=400, detail="Request body must include 'model'")

    session_key = _extract_session_key(request, payload)
    sticky_key = (
        build_sticky_key(session_key, endpoint, model_name) if session_key is not None else None
    )

    try:
        candidates = await router_state.get_candidate_providers(model_name, endpoint, sticky_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'") from exc

    stream = bool(payload.get("stream"))
    attempts: list[dict[str, Any]] = []

    client = httpx.AsyncClient(follow_redirects=True)
    streaming_response = False
    try:
        for provider in candidates:
            timeout = pick_timeout(payload, provider)
            upstream_payload = dict(payload)
            upstream_payload["model"] = provider.upstream_model
            url = f"{provider.api_base}{endpoint}"
            headers = {
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
                **provider.extra_headers,
            }
            log_provider_event(
                "attempt",
                provider,
                f"endpoint={endpoint} stream={str(stream).lower()}",
            )

            try:
                if stream:
                    request = client.build_request(
                        "POST",
                        url,
                        headers=headers,
                        json=upstream_payload,
                        timeout=timeout,
                    )
                    response = await client.send(
                        request,
                        stream=True,
                    )
                else:
                    response = await client.post(
                        url,
                        headers=headers,
                        json=upstream_payload,
                        timeout=timeout,
                    )
            except Exception as exc:
                error_message = f"{exc.__class__.__name__}: {exc}"
                decision = classify_transport_error(exc)
                log_provider_event(
                    "failure",
                    provider,
                    f"endpoint={endpoint} error={error_message}",
                )
                await router_state.record_failure(
                    provider=provider,
                    endpoint=endpoint,
                    error_message=error_message,
                    decision=decision,
                )
                if decision.count_failure:
                    await router_state.clear_session_binding(sticky_key, provider)
                attempts.append(
                    {
                        "provider_id": provider.provider_id,
                        "api_base": provider.api_base,
                        "timeout": timeout,
                        "status": "transport_error",
                        "failure_class": decision.failure_class,
                        "counted": decision.count_failure,
                        "error": error_message,
                    }
                )
                if decision.should_fallback:
                    continue
                raise HTTPException(status_code=503, detail={"message": error_message})

            if response.status_code >= 400:
                body = await response.aread()
                decision = classify_http_error(
                    status_code=response.status_code,
                    body=body,
                    endpoint=endpoint,
                )
                log_provider_event(
                    "failure",
                    provider,
                    f"endpoint={endpoint} status={response.status_code}",
                )
                error_message = (
                    f"HTTP {response.status_code}: "
                    f"{body[:500].decode('utf-8', errors='replace')}"
                )
                await router_state.record_failure(
                    provider=provider,
                    endpoint=endpoint,
                    error_message=error_message,
                    decision=decision,
                )
                if decision.count_failure:
                    await router_state.clear_session_binding(sticky_key, provider)
                attempts.append(
                    {
                        "provider_id": provider.provider_id,
                        "api_base": provider.api_base,
                        "timeout": timeout,
                        "status": "http_error",
                        "http_status": response.status_code,
                        "failure_class": decision.failure_class,
                        "counted": decision.count_failure,
                        "should_fallback": decision.should_fallback,
                        "body_preview": body[:500].decode("utf-8", errors="replace"),
                    }
                )
                await response.aclose()
                if decision.should_fallback:
                    continue
                parsed_detail: Any
                try:
                    parsed_detail = json.loads(body)
                except Exception:
                    parsed_detail = {"message": body.decode("utf-8", errors="replace")}
                return JSONResponse(content=parsed_detail, status_code=response.status_code)

            if stream:
                await router_state.bind_session(sticky_key, provider)
                headers = {
                    "x-fallback-provider-id": provider.provider_id,
                    "x-fallback-api-base": provider.api_base,
                }
                content_type = response.headers.get("content-type", "text/event-stream")
                streaming_response = True
                return StreamingResponse(
                    stream_upstream(response, provider, endpoint, client),
                    status_code=response.status_code,
                    media_type=content_type,
                    headers=headers,
                )

            body = await response.aread()
            await response.aclose()
            await router_state.record_success(provider, endpoint)
            log_provider_event("success", provider, f"endpoint={endpoint} stream=false")
            await router_state.bind_session(sticky_key, provider)

            parsed: Any
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                parsed = json.loads(body)
            elif endpoint == "/responses" and "text/event-stream" in content_type:
                parsed = _extract_responses_json_from_sse(body)
                if parsed is None:
                    parsed = {"raw_text": body.decode("utf-8", errors="replace")}
            else:
                parsed = {"raw_text": body.decode("utf-8", errors="replace")}

            return JSONResponse(
                content=parsed,
                status_code=response.status_code,
                headers={
                    "x-fallback-provider-id": provider.provider_id,
                    "x-fallback-api-base": provider.api_base,
                    "x-fallback-order": str(provider.order),
                },
            )

        raise HTTPException(
            status_code=503,
            detail={
                "message": "All providers failed",
                "model": model_name,
                "attempts": attempts,
            },
        )
    finally:
        if not streaming_response:
            await client.aclose()


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name": "mini-fallback-proxy",
        "config_path": router_state.config_path,
        "endpoints": ["/v1/responses", "/v1/chat/completions", "/healthz", "/debug/state"],
    }


@app.get("/debug/state")
async def debug_state() -> dict[str, Any]:
    return await router_state.snapshot()


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": await router_state.list_models(),
    }


@app.get("/v1/models/{model_name}")
async def get_model(model_name: str) -> dict[str, Any]:
    try:
        return await router_state.get_model(model_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'") from exc


@app.post("/admin/reload")
async def reload_config() -> dict[str, Any]:
    router_state.reload()
    return {"status": "reloaded"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    return await forward_request(request, "/chat/completions")


@app.post("/v1/responses")
async def responses_api(request: Request) -> Any:
    return await forward_request(request, "/responses")
