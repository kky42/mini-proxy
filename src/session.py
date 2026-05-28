from __future__ import annotations

import hashlib
import json
from typing import Any

from fastapi import Request


def _extract_openai_chat_fingerprint(
    request: Request, payload: dict[str, Any]
) -> str | None:
    """Derive a stable session key from request content for /chat/completions.

    When no explicit session identifier is present the first user message
    combined with the client IP produces a fingerprint that is stable
    across turns of the same conversation.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    first_user_content: str | None = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            first_user_content = content.strip()
            break

    if not first_user_content:
        return None

    client_ip = ""
    if request.client is not None and request.client.host:
        client_ip = request.client.host

    raw = f"{client_ip}|{first_user_content}"
    fp = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"content|{fp}"


def _extract_session_key(
    request: Request,
    payload: dict[str, Any],
    *,
    endpoint: str | None = None,
) -> str | None:
    # -- Tier 1: x-fallback-session header --
    header_value = request.headers.get("x-fallback-session")
    if header_value:
        return f"header|{header_value}"

    # -- Tier 2: top-level body fields --
    candidates = (
        ("conversation_id", payload.get("conversation_id")),
        ("thread_id", payload.get("thread_id")),
        ("previous_response_id", payload.get("previous_response_id")),
        ("prompt_cache_key", payload.get("prompt_cache_key")),
        ("user", payload.get("user")),
    )
    for name, value in candidates:
        if isinstance(value, str) and value.strip():
            return f"{name}|{value.strip()}"

    # -- Tier 3: nested metadata --
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for name in ("conversation_id", "thread_id", "session_id"):
            value = metadata.get(name)
            if isinstance(value, str) and value.strip():
                return f"metadata:{name}|{value.strip()}"

        user_id = metadata.get("user_id")
        if isinstance(user_id, str) and user_id.strip():
            try:
                parsed_user_id = json.loads(user_id)
            except Exception:
                parsed_user_id = None
            if isinstance(parsed_user_id, dict):
                for name in ("conversation_id", "thread_id", "session_id"):
                    value = parsed_user_id.get(name)
                    if isinstance(value, str) and value.strip():
                        return f"metadata:user_id:{name}|{value.strip()}"

        value = metadata.get("user")
        if isinstance(value, str) and value.strip():
            return f"metadata:user|{value.strip()}"

    # -- Tier 4: OpenAI /chat/completions fingerprint fallback --
    if endpoint == "/chat/completions":
        return _extract_openai_chat_fingerprint(request, payload)

    return None
