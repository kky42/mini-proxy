from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager, suppress
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
DEFAULT_HOT_RELOAD_INTERVAL_SECONDS = 1.0
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_ROLE_MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}
ANTHROPIC_ONE_M_SUFFIX_RE = re.compile(r"\[1m\]\s*$", re.IGNORECASE)
ENDPOINT_TYPES = {"responses", "openai-compatible", "anthropic"}


def _normalize_upstream_model(model_name: str) -> str:
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def _strip_anthropic_context_marker(model_name: str) -> str:
    stripped = model_name.strip()
    return ANTHROPIC_ONE_M_SUFFIX_RE.sub("", stripped).strip()


def _normalize_requested_model(model_name: str) -> str:
    return _strip_anthropic_context_marker(model_name)


def _infer_anthropic_role_from_model_name(model_name: str) -> str | None:
    normalized = _normalize_requested_model(model_name).lower()
    for role in ANTHROPIC_ROLE_MODEL_ALIASES:
        if role in normalized:
            return role
    return None


def _coerce_anthropic_role(value: Any, *, context: str) -> str:
    role = str(value).strip().lower()
    if role not in ANTHROPIC_ROLE_MODEL_ALIASES:
        allowed = ", ".join(sorted(ANTHROPIC_ROLE_MODEL_ALIASES))
        raise ValueError(f"{context} must be one of: {allowed}")
    return role


def _split_model_role_suffix(model_name: str) -> tuple[str, str | None]:
    stripped = model_name.strip()
    if ":" not in stripped:
        return stripped, None

    base, suffix = stripped.rsplit(":", 1)
    role = suffix.strip().lower()
    if role not in ANTHROPIC_ROLE_MODEL_ALIASES:
        return stripped, None

    base = base.strip()
    if not base:
        return stripped, None
    return base, role


def _split_model_mapping_suffix(model_name: str) -> tuple[str, str | None, str | None]:
    configured_model, role = _split_model_role_suffix(model_name)
    if role:
        return configured_model, ANTHROPIC_ROLE_MODEL_ALIASES[role], role

    stripped = model_name.strip()
    if ":" not in stripped:
        return stripped, None, None

    base, suffix = stripped.rsplit(":", 1)
    base = base.strip()
    alias = _normalize_requested_model(suffix)
    if base and alias:
        return base, alias, None
    return stripped, None, None


def _coerce_optional_timeout(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _coerce_positive_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _coerce_headers(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(header_value) for key, header_value in value.items()}


def _coerce_endpoint_type(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    endpoint_type = str(value).strip().lower()
    aliases = {
        "openai": "openai-compatible",
        "openai_compatible": "openai-compatible",
        "chat": "openai-compatible",
        "chat-completions": "openai-compatible",
        "chat_completions": "openai-compatible",
        "responses-api": "responses",
        "response": "responses",
        "messages": "anthropic",
    }
    endpoint_type = aliases.get(endpoint_type, endpoint_type)
    if endpoint_type not in ENDPOINT_TYPES:
        allowed = ", ".join(sorted(ENDPOINT_TYPES))
        raise ValueError(f"{context} must be one of: {allowed}")
    return endpoint_type


def _endpoint_type_supports_endpoint(endpoint_type: str | None, endpoint: str) -> bool:
    if endpoint_type is None:
        return True
    if endpoint_type == "anthropic":
        return endpoint == "/messages"
    if endpoint_type == "responses":
        return endpoint == "/responses"
    if endpoint_type == "openai-compatible":
        return endpoint == "/chat/completions"
    return False


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

    if status_code in {401, 402, 403}:
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
    provider_name: str | None
    model_name: str
    configured_model: str
    upstream_model: str
    anthropic_role: str | None
    endpoint_type: str | None
    api_base: str
    api_key: str
    order: int
    timeout: float | None
    extra_headers: dict[str, str]

    @property
    def provider_id(self) -> str:
        return f"{self.model_name}|{self.order}|{self.api_base}|{self.upstream_model}"


@dataclass(frozen=True)
class RouterConfig:
    host: str
    port: int
    log_level: str
    default_timeout: float
    sticky_ttl_seconds: int
    normalize_upstream_model: bool
    hot_reload: bool
    hot_reload_interval_seconds: float
    allowed_fails: int
    cooldown_time: int
    providers_by_model: dict[str, list[Provider]]


@dataclass(frozen=True)
class ReloadResult:
    status: str
    reloaded: bool
    error: str | None = None


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
        self.hot_reload = True
        self.hot_reload_interval_seconds = DEFAULT_HOT_RELOAD_INTERVAL_SECONDS
        self.providers_by_model: dict[str, list[Provider]] = {}
        self.last_reload_at: float | None = None
        self.last_config_mtime: float | None = None
        self.last_observed_config_mtime: float | None = None
        self.last_reload_error: str | None = None
        self._apply_config(self._load_config())

    def _build_provider(
        self,
        *,
        provider_name: str | None,
        model_name: str,
        configured_model: str,
        anthropic_role: str | None,
        endpoint_type: str | None,
        provider_params: dict[str, Any],
        normalize_upstream_model: bool,
    ) -> Provider:
        upstream_model = configured_model
        if normalize_upstream_model:
            upstream_model = _normalize_upstream_model(configured_model)
        return Provider(
            provider_name=provider_name,
            model_name=model_name,
            configured_model=configured_model,
            upstream_model=upstream_model,
            anthropic_role=anthropic_role,
            endpoint_type=endpoint_type,
            api_base=str(provider_params["api_base"]).rstrip("/"),
            api_key=provider_params["api_key"],
            order=int(provider_params.get("order", 100)),
            timeout=_coerce_optional_timeout(provider_params.get("timeout")),
            extra_headers=_coerce_headers(provider_params.get("headers")),
        )

    def _coerce_model_entry(
        self,
        model_entry: Any,
        *,
        provider_index: int,
        model_index: int,
    ) -> tuple[str, str, str | None]:
        if isinstance(model_entry, str):
            configured_model, alias_model_name, anthropic_role = _split_model_mapping_suffix(
                model_entry
            )
            if not configured_model:
                raise ValueError(
                    f"providers[{provider_index}].models[{model_index}] must not be empty"
                )
            model_name = alias_model_name if alias_model_name else configured_model
            return model_name, configured_model, anthropic_role

        if not isinstance(model_entry, dict):
            raise ValueError(
                f"providers[{provider_index}].models[{model_index}] must be a string or mapping"
            )

        configured_model = model_entry.get("model", model_entry.get("upstream_model"))
        model_name = model_entry.get("model_name", model_entry.get("name"))
        role_value = model_entry.get(
            "anthropic_role",
            model_entry.get("role", model_entry.get("map_to", model_entry.get("maps_to"))),
        )
        anthropic_role = (
            _coerce_anthropic_role(
                role_value,
                context=f"providers[{provider_index}].models[{model_index}].anthropic_role",
            )
            if role_value is not None
            else None
        )
        if configured_model is None and model_name is None:
            raise ValueError(
                f"providers[{provider_index}].models[{model_index}] must include "
                "'model_name' or 'model'"
            )
        if configured_model is None:
            configured_model = model_name
        if model_name is None:
            model_name = configured_model

        configured_model, inferred_model_name, inferred_role = _split_model_mapping_suffix(
            str(configured_model)
        )
        if inferred_role and anthropic_role is None:
            anthropic_role = inferred_role
        if (
            inferred_model_name
            and "model_name" not in model_entry
            and "name" not in model_entry
        ):
            model_name = inferred_model_name
        if anthropic_role and "model_name" not in model_entry and "name" not in model_entry:
            model_name = ANTHROPIC_ROLE_MODEL_ALIASES[anthropic_role]

        model_name = str(model_name).strip()
        configured_model = configured_model.strip()
        if not model_name or not configured_model:
            raise ValueError(
                f"providers[{provider_index}].models[{model_index}] has an empty model value"
            )
        return model_name, configured_model, anthropic_role

    def _models_url(self, provider_params: dict[str, Any]) -> str:
        configured_url = provider_params.get("models_url")
        if configured_url is not None:
            return str(configured_url)

        api_base = str(provider_params["api_base"]).rstrip("/")
        if api_base.endswith("/v1"):
            return f"{api_base}/models"
        return f"{api_base}/v1/models"

    def _load_auto_models(
        self,
        provider_params: dict[str, Any],
        *,
        provider_index: int,
        endpoint_type: str | None,
    ) -> list[str]:
        headers = {"Accept": "application/json"}
        if endpoint_type == "anthropic":
            headers["x-api-key"] = str(provider_params["api_key"])
            headers["anthropic-version"] = ANTHROPIC_VERSION
        else:
            headers["Authorization"] = f"Bearer {provider_params['api_key']}"

        response = httpx.get(
            self._models_url(provider_params),
            headers=headers,
            timeout=_coerce_optional_timeout(provider_params.get("timeout")) or DEFAULT_TIMEOUT,
        )
        if response.status_code >= 400:
            raise ValueError(
                f"providers[{provider_index}].models auto discovery failed with "
                f"HTTP {response.status_code}: {response.text[:500]}"
            )

        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ValueError(
                f"providers[{provider_index}].models auto discovery response must include "
                "a data list"
            )

        discovered: list[str] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                continue

            supported_types = item.get("supported_endpoint_types")
            if endpoint_type and isinstance(supported_types, list):
                normalized_supported_types = {
                    _coerce_endpoint_type(
                        supported_type,
                        context=(
                            f"providers[{provider_index}].models auto "
                            "supported_endpoint_types"
                        ),
                    )
                    for supported_type in supported_types
                }
                if endpoint_type not in normalized_supported_types:
                    continue

            discovered.append(model_id.strip())

        if not discovered:
            raise ValueError(f"providers[{provider_index}].models auto discovered no models")
        return discovered

    def _load_providers(
        self,
        raw: dict[str, Any],
        *,
        normalize_upstream_model: bool,
    ) -> dict[str, list[Provider]]:
        if "model_list" in raw:
            raise ValueError("model_list is no longer supported; use providers[*].models")

        providers_config = raw.get("providers")
        if not isinstance(providers_config, list):
            raise ValueError("providers must be a list")

        grouped: dict[str, list[Provider]] = {}
        for provider_index, provider_params in enumerate(providers_config):
            if not isinstance(provider_params, dict):
                raise ValueError(f"providers[{provider_index}] must be a mapping")

            for required_key in ("api_base", "api_key"):
                if required_key not in provider_params:
                    raise ValueError(
                        f"providers[{provider_index}].{required_key} is required"
                    )

            endpoint_type = _coerce_endpoint_type(
                provider_params.get("endpoint_type"),
                context=f"providers[{provider_index}].endpoint_type",
            )
            models = provider_params.get("models")
            if isinstance(models, str) and models.strip().lower() == "auto":
                models = self._load_auto_models(
                    provider_params,
                    provider_index=provider_index,
                    endpoint_type=endpoint_type,
                )
            if not isinstance(models, list) or not models:
                raise ValueError(
                    f"providers[{provider_index}].models must be a non-empty list or 'auto'"
                )

            provider_name_value = provider_params.get("name")
            provider_name = (
                str(provider_name_value).strip()
                if provider_name_value is not None
                else None
            )
            if provider_name == "":
                provider_name = None

            for model_index, model_entry in enumerate(models):
                model_name, configured_model, anthropic_role = self._coerce_model_entry(
                    model_entry,
                    provider_index=provider_index,
                    model_index=model_index,
                )
                provider = self._build_provider(
                    provider_name=provider_name,
                    model_name=model_name,
                    configured_model=configured_model,
                    anthropic_role=anthropic_role,
                    endpoint_type=endpoint_type,
                    provider_params=provider_params,
                    normalize_upstream_model=normalize_upstream_model,
                )
                grouped.setdefault(model_name, []).append(provider)

        return {
            model: sorted(providers, key=lambda p: (p.order, p.api_base, p.upstream_model))
            for model, providers in grouped.items()
        }

    def _load_config(self) -> RouterConfig:
        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("Config root must be a mapping")

        app_settings = raw.get("app_settings")
        if app_settings is None:
            app_settings = {}
        if not isinstance(app_settings, dict):
            raise ValueError("app_settings must be a mapping")

        host = str(app_settings.get("host", "127.0.0.1"))
        port = int(app_settings.get("port", 8099))
        log_level = str(app_settings.get("log_level", "info"))
        default_timeout = float(app_settings.get("default_timeout", DEFAULT_TIMEOUT))
        sticky_ttl_seconds = int(
            app_settings.get("sticky_ttl_seconds", DEFAULT_STICKY_TTL_SECONDS)
        )
        normalize_upstream_model = _coerce_bool(
            app_settings.get("normalize_upstream_model"), True
        )
        hot_reload = _coerce_bool(app_settings.get("hot_reload"), True)
        hot_reload_interval_seconds = _coerce_positive_float(
            app_settings.get("hot_reload_interval_seconds"),
            DEFAULT_HOT_RELOAD_INTERVAL_SECONDS,
        )

        router_settings = raw.get("router_settings")
        if router_settings is None:
            router_settings = {}
        if not isinstance(router_settings, dict):
            raise ValueError("router_settings must be a mapping")
        allowed_fails = int(router_settings.get("allowed_fails", 0))
        cooldown_time = int(router_settings.get("cooldown_time", 300))

        providers_by_model = self._load_providers(
            raw,
            normalize_upstream_model=normalize_upstream_model,
        )
        return RouterConfig(
            host=host,
            port=port,
            log_level=log_level,
            default_timeout=default_timeout,
            sticky_ttl_seconds=sticky_ttl_seconds,
            normalize_upstream_model=normalize_upstream_model,
            hot_reload=hot_reload,
            hot_reload_interval_seconds=hot_reload_interval_seconds,
            allowed_fails=allowed_fails,
            cooldown_time=cooldown_time,
            providers_by_model=providers_by_model,
        )

    def _apply_config(self, config: RouterConfig) -> None:
        self.host = config.host
        self.port = config.port
        self.log_level = config.log_level
        self.default_timeout = config.default_timeout
        self.sticky_ttl_seconds = config.sticky_ttl_seconds
        self.normalize_upstream_model = config.normalize_upstream_model
        self.hot_reload = config.hot_reload
        self.hot_reload_interval_seconds = config.hot_reload_interval_seconds
        self.allowed_fails = config.allowed_fails
        self.cooldown_time = config.cooldown_time
        self.providers_by_model = config.providers_by_model
        self.last_reload_at = time.time()
        self.last_reload_error = None
        try:
            config_mtime = os.path.getmtime(self.config_path)
        except OSError:
            config_mtime = None
        self.last_config_mtime = config_mtime
        self.last_observed_config_mtime = config_mtime

    async def reload(self) -> ReloadResult:
        try:
            config = self._load_config()
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {exc}"
            async with self._lock:
                self.last_reload_error = error
            return ReloadResult(status="rejected", reloaded=False, error=error)

        async with self._lock:
            self._apply_config(config)
        return ReloadResult(status="reloaded", reloaded=True)

    async def reload_if_changed(self) -> ReloadResult:
        try:
            current_mtime = os.path.getmtime(self.config_path)
        except OSError as exc:
            error = f"{exc.__class__.__name__}: {exc}"
            async with self._lock:
                self.last_reload_error = error
            return ReloadResult(status="rejected", reloaded=False, error=error)

        async with self._lock:
            previous_mtime = self.last_observed_config_mtime
        if previous_mtime is not None and current_mtime == previous_mtime:
            return ReloadResult(status="unchanged", reloaded=False)
        async with self._lock:
            self.last_observed_config_mtime = current_mtime
        return await self.reload()

    async def get_candidate_providers(
        self, model_name: str, endpoint: str, sticky_key: str | None
    ) -> list[Provider]:
        model_name = _normalize_requested_model(model_name)
        async with self._lock:
            providers = list(self.providers_by_model.get(model_name, []))
            if endpoint == "/messages":
                role = _infer_anthropic_role_from_model_name(model_name)
                if role:
                    role_providers = [
                        provider
                        for model_providers in self.providers_by_model.values()
                        for provider in model_providers
                        if provider.anthropic_role == role
                    ]
                    seen_provider_ids = {provider.provider_id for provider in providers}
                    providers.extend(
                        provider
                        for provider in role_providers
                        if provider.provider_id not in seen_provider_ids
                    )
            providers = sorted(
                providers,
                key=lambda provider: (provider.order, provider.api_base, provider.upstream_model),
            )
            if not providers:
                raise KeyError(model_name)

            now = time.time()
            healthy: list[Provider] = []
            cooling: list[Provider] = []
            for provider in providers:
                if not _endpoint_type_supports_endpoint(provider.endpoint_type, endpoint):
                    continue
                cooldown_until = self.cooldown_until.get(build_failure_key(provider, endpoint), 0)
                if cooldown_until <= now:
                    healthy.append(provider)
                else:
                    cooling.append(provider)

            if not healthy and not cooling:
                raise KeyError(model_name)

            # Prefer providers outside cooldown, but keep cooling providers as a
            # last-resort chain for cases where the only healthy upstream fails.
            if healthy:
                return [
                    *self._apply_sticky_preference(healthy, sticky_key),
                    *cooling,
                ]
            return self._apply_sticky_preference(cooling, sticky_key)

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
                    messages_key = build_failure_key(provider, "/messages")
                    responses_cooldown = self.cooldown_until.get(responses_key)
                    chat_cooldown = self.cooldown_until.get(chat_key)
                    messages_cooldown = self.cooldown_until.get(messages_key)
                    providers.append(
                        {
                            "model_name": model_name,
                            "provider_name": provider.provider_name,
                            "provider_id": provider_id,
                            "upstream_model": provider.upstream_model,
                            "configured_model": provider.configured_model,
                            "anthropic_role": provider.anthropic_role,
                            "endpoint_type": provider.endpoint_type,
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
                                "/messages": max(0, int(messages_cooldown - now))
                                if messages_cooldown
                                else 0,
                            },
                            "last_error": {
                                "/responses": self.last_error.get(responses_key),
                                "/chat/completions": self.last_error.get(chat_key),
                                "/messages": self.last_error.get(messages_key),
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
                    "hot_reload": self.hot_reload,
                    "hot_reload_interval_seconds": self.hot_reload_interval_seconds,
                },
                "hot_reload": {
                    "enabled": self.hot_reload,
                    "interval_seconds": self.hot_reload_interval_seconds,
                    "last_success_at": self.last_reload_at,
                    "last_success_at_iso": datetime.fromtimestamp(
                        self.last_reload_at
                    ).isoformat()
                    if self.last_reload_at
                    else None,
                    "last_config_mtime": self.last_config_mtime,
                    "last_observed_config_mtime": self.last_observed_config_mtime,
                    "last_error": self.last_reload_error,
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
                                "provider_name": provider.provider_name,
                                "order": provider.order,
                                "api_base": provider.api_base,
                                "configured_model": provider.configured_model,
                                "upstream_model": provider.upstream_model,
                                "anthropic_role": provider.anthropic_role,
                                "endpoint_type": provider.endpoint_type,
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
                        "provider_name": provider.provider_name,
                        "order": provider.order,
                        "api_base": provider.api_base,
                        "configured_model": provider.configured_model,
                        "upstream_model": provider.upstream_model,
                        "anthropic_role": provider.anthropic_role,
                        "endpoint_type": provider.endpoint_type,
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
logger = logging.getLogger("uvicorn.error")


async def watch_config_changes() -> None:
    while True:
        await asyncio.sleep(router_state.hot_reload_interval_seconds)
        if not router_state.hot_reload:
            continue

        result = await router_state.reload_if_changed()
        if result.status == "reloaded":
            logger.info("Config hot-reloaded from %s", router_state.config_path)
        elif result.status == "rejected":
            logger.error(
                "Config hot reload rejected for %s: %s",
                router_state.config_path,
                result.error,
            )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    task = asyncio.create_task(watch_config_changes())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="Mini Fallback Proxy", lifespan=lifespan)


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


def build_upstream_timeout(
    payload: dict[str, Any],
    provider: Provider,
    *,
    stream: bool,
) -> httpx.Timeout | float:
    timeout = pick_timeout(payload, provider)
    if not stream:
        return timeout

    # SSE responses can sit idle for a long time while the model reasons.
    # Keep connect/write/pool bounded, but do not abort the stream for idle reads.
    return httpx.Timeout(timeout, read=None)


def build_upstream_headers(request: Request, provider: Provider, endpoint: str) -> dict[str, str]:
    if endpoint == "/messages":
        headers = {
            "x-api-key": provider.api_key,
            "anthropic-version": request.headers.get(
                "anthropic-version",
                ANTHROPIC_VERSION,
            ),
            "Content-Type": "application/json",
        }
        anthropic_beta = request.headers.get("anthropic-beta")
        if anthropic_beta:
            headers["anthropic-beta"] = anthropic_beta
        return {**headers, **provider.extra_headers}

    return {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
        **provider.extra_headers,
    }


def build_upstream_url(provider: Provider, endpoint: str) -> str:
    api_base = provider.api_base.rstrip("/")
    parsed_base = urlparse(api_base)
    base_path = parsed_base.path.rstrip("/")
    if endpoint in {"/responses", "/chat/completions"} and not api_base.endswith("/v1"):
        api_base = f"{api_base}/v1"
    elif endpoint == "/messages" and not base_path:
        api_base = f"{api_base}/v1"
    return f"{api_base}{endpoint}"


def is_valid_stream_success_content_type(content_type: str) -> bool:
    return "text/event-stream" in content_type.lower()


def parse_upstream_success_body(
    body: bytes,
    *,
    content_type: str,
    endpoint: str,
) -> tuple[Any | None, str | None]:
    normalized_content_type = content_type.lower()
    stripped = body.lstrip()
    if (
        "application/json" in normalized_content_type
        or stripped.startswith(b"{")
        or stripped.startswith(b"[")
    ):
        try:
            return json.loads(body), None
        except Exception as exc:
            return None, f"Invalid JSON success response: {exc}"

    if endpoint == "/responses" and "text/event-stream" in normalized_content_type:
        parsed = _extract_responses_json_from_sse(body)
        if parsed is not None:
            return parsed, None
        return None, "Invalid Responses SSE success response"

    return None, f"Unexpected success content-type: {content_type or '<missing>'}"


async def forward_request(request: Request, endpoint: str) -> Any:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

    requested_model = payload.get("model")
    if not isinstance(requested_model, str) or not requested_model.strip():
        raise HTTPException(status_code=400, detail="Request body must include string 'model'")
    model_name = _normalize_requested_model(requested_model)

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
    total_provider_count = len(candidates)

    client = httpx.AsyncClient(follow_redirects=True)
    streaming_response = False
    try:
        for provider in candidates:
            timeout = build_upstream_timeout(payload, provider, stream=stream)
            upstream_payload = dict(payload)
            upstream_payload["model"] = provider.upstream_model
            url = build_upstream_url(provider, endpoint)
            headers = build_upstream_headers(request, provider, endpoint)
            log_provider_event(
                "attempt",
                provider,
                f"endpoint={endpoint} stream={str(stream).lower()}",
            )

            try:
                if stream:
                    upstream_request = client.build_request(
                        "POST",
                        url,
                        headers=headers,
                        json=upstream_payload,
                        timeout=timeout,
                    )
                    response = await client.send(
                        upstream_request,
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
                        "timeout": str(timeout),
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
                        "timeout": str(timeout),
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
                content_type = response.headers.get("content-type", "")
                if not is_valid_stream_success_content_type(content_type):
                    body = await response.aread()
                    await response.aclose()
                    decision = FailureDecision(
                        failure_class=FailureClass.CAPABILITY_MISMATCH,
                        should_fallback=True,
                        count_failure=False,
                    )
                    error_message = (
                        f"Unexpected stream content-type: "
                        f"{content_type or '<missing>'}: "
                        f"{body[:500].decode('utf-8', errors='replace')}"
                    )
                    log_provider_event(
                        "failure",
                        provider,
                        f"endpoint={endpoint} error=Unexpected stream content-type: "
                        f"{content_type or '<missing>'}",
                    )
                    await router_state.record_failure(
                        provider=provider,
                        endpoint=endpoint,
                        error_message=error_message,
                        decision=decision,
                    )
                    attempts.append(
                        {
                            "provider_id": provider.provider_id,
                            "api_base": provider.api_base,
                            "timeout": str(timeout),
                            "status": "invalid_stream_response",
                            "failure_class": decision.failure_class,
                            "counted": decision.count_failure,
                            "should_fallback": decision.should_fallback,
                            "content_type": content_type,
                            "body_preview": body[:500].decode("utf-8", errors="replace"),
                        }
                    )
                    continue

                await router_state.bind_session(sticky_key, provider)
                headers = {
                    "x-fallback-provider-id": provider.provider_id,
                    "x-fallback-api-base": provider.api_base,
                }
                streaming_response = True
                return StreamingResponse(
                    stream_upstream(response, provider, endpoint, client),
                    status_code=response.status_code,
                    media_type=content_type,
                    headers=headers,
                )

            body = await response.aread()
            await response.aclose()
            content_type = response.headers.get("content-type", "")
            parsed, parse_error = parse_upstream_success_body(
                body,
                content_type=content_type,
                endpoint=endpoint,
            )
            if parse_error is not None:
                decision = FailureDecision(
                    failure_class=FailureClass.CAPABILITY_MISMATCH,
                    should_fallback=True,
                    count_failure=False,
                )
                error_message = (
                    f"{parse_error}: {body[:500].decode('utf-8', errors='replace')}"
                )
                log_provider_event(
                    "failure",
                    provider,
                    f"endpoint={endpoint} error={parse_error}",
                )
                await router_state.record_failure(
                    provider=provider,
                    endpoint=endpoint,
                    error_message=error_message,
                    decision=decision,
                )
                attempts.append(
                    {
                        "provider_id": provider.provider_id,
                        "api_base": provider.api_base,
                        "timeout": str(timeout),
                        "status": "invalid_success_response",
                        "failure_class": decision.failure_class,
                        "counted": decision.count_failure,
                        "should_fallback": decision.should_fallback,
                        "content_type": content_type,
                        "body_preview": body[:500].decode("utf-8", errors="replace"),
                    }
                )
                continue

            await router_state.record_success(provider, endpoint)
            log_provider_event("success", provider, f"endpoint={endpoint} stream=false")
            await router_state.bind_session(sticky_key, provider)

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
                "candidate_provider_count": total_provider_count,
                "attempted_provider_count": len(attempts),
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
        "endpoints": [
            "/v1/messages",
            "/v1/responses",
            "/v1/chat/completions",
            "/healthz",
            "/debug/state",
        ],
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
    result = await router_state.reload()
    response: dict[str, Any] = {"status": result.status, "reloaded": result.reloaded}
    if result.error:
        response["error"] = result.error
    return response


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    return await forward_request(request, "/chat/completions")


@app.post("/chat/completions")
async def chat_completions_root(request: Request) -> Any:
    return await forward_request(request, "/chat/completions")


@app.post("/v1/responses")
async def responses_api(request: Request) -> Any:
    return await forward_request(request, "/responses")


@app.post("/responses")
async def responses_api_root(request: Request) -> Any:
    return await forward_request(request, "/responses")


@app.post("/v1/messages")
async def messages_api(request: Request) -> Any:
    return await forward_request(request, "/messages")


@app.post("/messages")
async def messages_api_root(request: Request) -> Any:
    return await forward_request(request, "/messages")


@app.get("/models")
async def list_models_root() -> dict[str, Any]:
    return await list_models()


@app.get("/models/{model_name}")
async def get_model_root(model_name: str) -> dict[str, Any]:
    return await get_model(model_name)
