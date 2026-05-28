from __future__ import annotations

import json
import re
from typing import Any

from .constants import (
    ANTHROPIC_ONE_M_SUFFIX_RE,
    ANTHROPIC_ROLE_MODEL_ALIASES,
    ANTHROPIC_ENDPOINTS,
    ENDPOINT_TYPES,
)
from .types import Provider


# ---------------------------------------------------------------------------
# Model name normalization
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Model suffix splitting
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------


def _coerce_anthropic_role(value: Any, *, context: str) -> str:
    role = str(value).strip().lower()
    if role not in ANTHROPIC_ROLE_MODEL_ALIASES:
        allowed = ", ".join(sorted(ANTHROPIC_ROLE_MODEL_ALIASES))
        raise ValueError(f"{context} must be one of: {allowed}")
    return role


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


def _coerce_optional_url(value: Any) -> str | None:
    if value is None:
        return None
    url = str(value).strip()
    return url or None


def _coerce_optional_api_base(value: Any) -> str | None:
    url = _coerce_optional_url(value)
    if url is None:
        return None
    return url.rstrip("/")


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
        return endpoint in ANTHROPIC_ENDPOINTS
    if endpoint_type == "responses":
        return endpoint == "/responses" or endpoint == "/responses/compact"
    if endpoint_type == "openai-compatible":
        return endpoint == "/chat/completions"
    return False


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def _append_v1_endpoint(api_base: str, endpoint: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}{endpoint}"
    return f"{base}/v1{endpoint}"


def _infer_models_url(api_base: str) -> str:
    return _append_v1_endpoint(api_base, "/models")


def build_upstream_url(provider: Provider, endpoint: str) -> str:
    if provider.api_url is not None:
        return provider.api_url
    if provider.api_base is None:
        raise ValueError("Provider must include api_base or api_url")
    return _append_v1_endpoint(provider.api_base, endpoint)


# ---------------------------------------------------------------------------
# JSON / SSE parsing
# ---------------------------------------------------------------------------


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


def _extract_responses_json_from_sse_text(body_text: str) -> dict[str, Any] | None:
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


def _extract_responses_json_from_sse(body: bytes) -> dict[str, Any] | None:
    body_text = body.decode("utf-8", errors="replace")
    return _extract_responses_json_from_sse_text(body_text)


def _extract_sse_error_message(body_text: str) -> str | None:
    for event_name, data in _iter_sse_events(body_text):
        try:
            payload = json.loads(data)
        except Exception:
            if event_name == "error":
                return data[:500]
            continue

        if event_name != "error" and not (
            isinstance(payload, dict) and payload.get("error") is not None
        ):
            continue

        message = _extract_error_message(payload, data)
        return message[:500]
    return None


# ---------------------------------------------------------------------------
# Error extraction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Sticky / failure key builders (pure)
# ---------------------------------------------------------------------------


def build_failure_key(provider: Provider, endpoint: str) -> str:
    return f"{provider.provider_id}|{endpoint}|{provider.model_name}|{provider.upstream_model}"


def build_sticky_key(session_key: str, endpoint: str, model_name: str) -> str:
    return f"{session_key}|{endpoint}|{model_name}"
