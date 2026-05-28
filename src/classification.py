from __future__ import annotations

import httpx

from .types import FailureClass, FailureDecision
from .utils import _safe_json_loads, _extract_error_code, _extract_error_message


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
