"""Shared helpers for end-to-end proxy tests.

Provides fake upstream responses, config builders, and state management
so individual test files can focus on behaviour, not boilerplate.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx
import yaml


# ---------------------------------------------------------------------------
# Fake upstream responses
# ---------------------------------------------------------------------------


class FakeResponse:
    """Simulates a successful upstream JSON response."""

    status_code = 200
    headers: dict[str, str]
    is_closed = False

    def __init__(self, json_body: dict[str, Any], content_type: str = "application/json"):
        self._json = json_body
        self.headers = {"content-type": content_type}

    def json(self) -> dict[str, Any]:
        return self._json

    async def aread(self) -> bytes:
        return json.dumps(self._json).encode()

    async def aclose(self) -> None:
        self.is_closed = True


class FakeStreamResponse:
    """Simulates a successful SSE stream response."""

    status_code = 200
    headers = {"content-type": "text/event-stream"}
    is_closed = False

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def aread(self) -> bytes:
        return b"".join(self._chunks)

    async def aclose(self) -> None:
        self.is_closed = True

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


# ---------------------------------------------------------------------------
# Fake httpx clients
# ---------------------------------------------------------------------------


class _BaseFakeClient:
    """Mixin with the httpx.AsyncClient shape tests expect."""

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def aclose(self):
        pass


def make_uniform_client(response: FakeResponse | FakeStreamResponse):
    """Every upstream URL returns the same response."""

    class UniformClient(_BaseFakeClient):
        def build_request(self, method, url, headers=None, json=None, timeout=None):
            return httpx.Request(method, url, headers=headers or {})

        async def send(self, request, stream=False):
            return response

        async def post(self, url, headers=None, json=None, timeout=None):
            return response

    return UniformClient


def make_selective_client(
    fail_url_contains: str,
    ok_url_contains: str,
    *,
    fail_status: int = 503,
    ok_body: dict[str, Any] | None = None,
):
    """URLs containing fail_url_contains get an error; ok_url_contains get 200."""

    if ok_body is None:
        ok_body = {
            "id": "ok",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}}],
        }

    error_resp = FakeResponse(
        {"error": {"message": "Service unavailable", "code": "server_error"}},
    )
    error_resp.status_code = fail_status
    ok_resp = FakeResponse(ok_body)

    class SelectiveClient(_BaseFakeClient):
        def build_request(self, method, url, headers=None, json=None, timeout=None):
            return httpx.Request(method, url, headers=headers or {})

        async def send(self, request, stream=False):
            url = str(request.url)
            if fail_url_contains in url:
                return error_resp
            return ok_resp

        async def post(self, url, headers=None, json=None, timeout=None):
            if fail_url_contains in url:
                return error_resp
            return ok_resp

    return SelectiveClient


def make_mapped_client(url_to_response: dict[str, FakeResponse | FakeStreamResponse]):
    """Exact URL → response mapping."""

    class MappedClient(_BaseFakeClient):
        def build_request(self, method, url, headers=None, json=None, timeout=None):
            return httpx.Request(method, url, headers=headers or {})

        async def send(self, request, stream=False):
            url = str(request.url)
            for key, resp in url_to_response.items():
                if key in url:
                    return resp
            return FakeResponse({"error": "no mock for " + url})

        async def post(self, url, headers=None, json=None, timeout=None):
            for key, resp in url_to_response.items():
                if key in url:
                    return resp
            return FakeResponse({"error": "no mock for " + url})

    return MappedClient


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def build_config(
    *,
    openai_providers: int = 2,
    anthropic_providers: int = 2,
    extra_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a config dict with the requested number of providers per type."""
    providers: list[dict[str, Any]] = []

    for i in range(openai_providers):
        providers.append({
            "name": f"openai-provider-{chr(ord('a') + i)}",
            "api_base": f"https://api.{chr(ord('a') + i)}.example",
            "api_key": f"sk-{chr(ord('a') + i)}",
            "order": i + 1,
            "endpoint_type": "openai-compatible",
            "models": ["gpt-4"],
        })

    for i in range(anthropic_providers):
        providers.append({
            "name": f"anthropic-provider-{chr(ord('a') + i)}",
            "api_base": f"https://api.anthropic.{chr(ord('a') + i)}.example",
            "api_key": f"sk-ant-{chr(ord('a') + i)}",
            "order": i + 1,
            "endpoint_type": "anthropic",
            "models": ["claude-sonnet-4-6"],
        })

    config: dict[str, Any] = {
        "app_settings": {"hot_reload": False, "sticky_ttl_seconds": 300},
        "providers": providers,
    }

    if extra_settings:
        config.update(extra_settings)

    return config


def write_temp_config(config: dict[str, Any], tmpdir: str, name: str = "config.yaml") -> Path:
    """Write a config dict to a temp YAML file. Returns the Path."""
    path = Path(tmpdir) / name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# App module loader
# ---------------------------------------------------------------------------


def load_app_module(config_path: str | Path):
    """Import the app module with MINI_FALLBACK_PROXY_CONFIG set to config_path."""
    os.environ["MINI_FALLBACK_PROXY_CONFIG"] = str(config_path)

    import importlib
    import app

    # Re-import so the module picks up the env var
    return importlib.import_module("app")


# ---------------------------------------------------------------------------
# Anthropic SSE helpers
# ---------------------------------------------------------------------------

ANTHROPIC_SSE_OK = (
    b'event: message_start\n'
    b'data: {"type":"message_start","message":{"id":"msg_1"}}\n\n'
    b'event: content_block_delta\n'
    b'data: {"type":"content_block_delta","delta":{"text":"Hello"}}\n\n'
    b'event: message_delta\n'
    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n'
    b'event: message_stop\n'
    b'data: {"type":"message_stop"}\n\n'
)


def make_claude_session_id(session_id: str) -> str:
    """Build a metadata.user_id JSON string with the given session_id."""
    return json.dumps({
        "device_id": "device-test",
        "account_uuid": "",
        "session_id": session_id,
    })
