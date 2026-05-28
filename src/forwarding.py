from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import globals as _g
from .classification import classify_http_error, classify_transport_error
from .constants import ANTHROPIC_ENDPOINTS, ANTHROPIC_VERSION
from .routing import (
    provider_attempt_dict,
    provider_response_headers,
)
from .session import _extract_session_key
from .types import FailureClass, FailureDecision, Provider
from .utils import (
    _append_v1_endpoint,
    _extract_responses_json_from_sse,
    _extract_responses_json_from_sse_text,
    _extract_sse_error_message,
    _normalize_requested_model,
    build_sticky_key,
    build_upstream_url,
)


def log_provider_event(
    event: str,
    provider: Provider,
    detail: str = "",
) -> None:
    from .constants import logger

    timestamp = datetime.now().strftime("%H:%M:%S")
    provider_url = provider.sort_url
    provider_host = urlparse(provider_url).netloc or provider_url
    message = (
        f"{timestamp} provider={provider.provider_name} host={provider_host} "
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
    *,
    byte_iter: AsyncIterator[bytes] | None = None,
    initial_chunk: bytes | None = None,
) -> AsyncIterator[bytes]:
    completed = False
    try:
        if initial_chunk:
            yield initial_chunk
        if byte_iter is None:
            byte_iter = response.aiter_bytes()
        async for chunk in byte_iter:
            yield chunk
        completed = True
    except Exception as exc:
        decision = FailureDecision(
            failure_class=FailureClass.MIDSTREAM_FAILURE,
            should_fallback=False,
            count_failure=True,
        )
        await _g.router_state.record_failure(
            provider=provider,
            endpoint=endpoint,
            error_message=f"Midstream failure: {exc.__class__.__name__}: {exc}",
            decision=decision,
        )
        raise
    finally:
        if completed:
            await _g.router_state.record_success(provider, endpoint)
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
    return _g.router_state.default_timeout


def build_upstream_timeout(
    payload: dict[str, Any],
    provider: Provider,
    *,
    stream: bool,
) -> httpx.Timeout | float:
    timeout = pick_timeout(payload, provider)
    if not stream:
        return timeout
    return httpx.Timeout(timeout, read=None)


def build_upstream_headers(request: Request, provider: Provider, endpoint: str) -> dict[str, str]:
    if endpoint in ANTHROPIC_ENDPOINTS:
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


def is_valid_stream_success_content_type(content_type: str) -> bool:
    return "text/event-stream" in content_type.lower()


async def validate_responses_stream_start(
    byte_iter: AsyncIterator[bytes],
    *,
    timeout: float,
) -> tuple[bytes | None, str | None]:
    first_chunk: bytes | None = None
    try:
        async with asyncio.timeout(timeout):
            async for chunk in byte_iter:
                if chunk:
                    first_chunk = chunk
                    break
    except TimeoutError:
        return None, f"Responses stream did not start within {timeout:g}s"
    if first_chunk is None:
        return None, "Responses stream ended before first event"

    chunk_text = first_chunk.decode("utf-8", errors="replace")
    error_message = _extract_sse_error_message(chunk_text)
    if error_message is not None:
        return first_chunk, f"Responses stream error event: {error_message}"

    if _extract_responses_json_from_sse_text(chunk_text) is not None:
        return first_chunk, None

    return first_chunk, None


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

    session_key = _extract_session_key(request, payload, endpoint=endpoint)
    sticky_key = (
        build_sticky_key(session_key, endpoint, model_name) if session_key is not None else None
    )

    try:
        candidates = await _g.router_state.get_candidate_providers(model_name, endpoint, sticky_key)
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
                    async with asyncio.timeout(_g.router_state.stream_start_timeout):
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
                await _g.router_state.record_failure(
                    provider=provider,
                    endpoint=endpoint,
                    error_message=error_message,
                    decision=decision,
                )
                if decision.count_failure:
                    await _g.router_state.clear_session_binding(sticky_key, provider)
                attempts.append(
                    {
                        **provider_attempt_dict(provider, url),
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
                await _g.router_state.record_failure(
                    provider=provider,
                    endpoint=endpoint,
                    error_message=error_message,
                    decision=decision,
                )
                if decision.count_failure:
                    await _g.router_state.clear_session_binding(sticky_key, provider)
                attempts.append(
                    {
                        **provider_attempt_dict(provider, url),
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
                    await _g.router_state.record_failure(
                        provider=provider,
                        endpoint=endpoint,
                        error_message=error_message,
                        decision=decision,
                    )
                    attempts.append(
                        {
                            **provider_attempt_dict(provider, url),
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

                initial_chunk: bytes | None = None
                byte_iter: AsyncIterator[bytes] | None = None
                if endpoint == "/responses":
                    byte_iter = response.aiter_bytes()
                    initial_chunk, stream_error = (
                        await validate_responses_stream_start(
                            byte_iter,
                            timeout=_g.router_state.stream_start_timeout,
                        )
                    )
                    if stream_error is not None:
                        await response.aclose()
                        decision = FailureDecision(
                            failure_class=FailureClass.AVAILABILITY,
                            should_fallback=True,
                            count_failure=True,
                        )
                        log_provider_event(
                            "failure",
                            provider,
                            f"endpoint={endpoint} error={stream_error}",
                        )
                        await _g.router_state.record_failure(
                            provider=provider,
                            endpoint=endpoint,
                            error_message=stream_error,
                            decision=decision,
                        )
                        await _g.router_state.clear_session_binding(sticky_key, provider)
                        attempts.append(
                            {
                                **provider_attempt_dict(provider, url),
                                "timeout": str(timeout),
                                "status": "stream_error_event",
                                "failure_class": decision.failure_class,
                                "counted": decision.count_failure,
                                "should_fallback": decision.should_fallback,
                                "content_type": content_type,
                                "body_preview": (initial_chunk or b"")[:500].decode(
                                    "utf-8",
                                    errors="replace",
                                ),
                            }
                        )
                        continue

                await _g.router_state.bind_session(sticky_key, provider)
                streaming_response = True
                return StreamingResponse(
                    stream_upstream(
                        response,
                        provider,
                        endpoint,
                        client,
                        byte_iter=byte_iter,
                        initial_chunk=initial_chunk,
                    ),
                    status_code=response.status_code,
                    media_type=content_type,
                    headers=provider_response_headers(provider, url),
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
                await _g.router_state.record_failure(
                    provider=provider,
                    endpoint=endpoint,
                    error_message=error_message,
                    decision=decision,
                )
                attempts.append(
                    {
                        **provider_attempt_dict(provider, url),
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

            await _g.router_state.record_success(provider, endpoint)
            log_provider_event("success", provider, f"endpoint={endpoint} stream=false")
            await _g.router_state.bind_session(sticky_key, provider)

            return JSONResponse(
                content=parsed,
                status_code=response.status_code,
                headers={
                    **provider_response_headers(provider, url),
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
