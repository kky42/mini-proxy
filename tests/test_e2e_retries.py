"""Provider retry e2e tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import yaml
from fastapi.testclient import TestClient

from tests.e2e_base import E2EBase
from tests.e2e_helpers import FakeResponse, build_config


def _chat_ok(content: str = "OK") -> FakeResponse:
    return FakeResponse(
        {
            "id": "ok",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}},
            ],
        }
    )


def _error(status_code: int) -> FakeResponse:
    response = FakeResponse(
        {"error": {"message": "temporary failure", "code": "server_error"}},
    )
    response.status_code = status_code
    return response


class RetryTests(E2EBase):
    config = build_config(
        extra_settings={
            "router_settings": {
                "allowed_fails": 1,
                "cooldown_time": 300,
                "allowed_retries": 2,
                "retry_backoff_seconds": 0,
            },
        },
    )

    def test_retry_success_stays_on_original_provider(self):
        """A retryable upstream failure is retried before fallback."""
        calls: list[str] = []

        class FakeClient:
            def __init__(self, **_: Any) -> None:
                pass

            async def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, Any],
                timeout: Any,
            ) -> FakeResponse:
                calls.append(url)
                if "api.a.example" in url and calls.count(url) <= 2:
                    return _error(503)
                return _chat_ok("recovered")

            async def aclose(self) -> None:
                pass

        client = TestClient(self.app_module.app)
        orig_client, orig_state = self.swap_state_and_client(self.state, FakeClient)
        try:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        finally:
            self.restore_state_and_client(orig_client, orig_state)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-fallback-provider-id"], "openai-provider-a")
        self.assertEqual(len([url for url in calls if "api.a.example" in url]), 3)
        self.assertFalse(any("api.b.example" in url for url in calls))

    def test_exhausted_retries_count_as_one_failure_then_fallback(self):
        """Each exhausted retry cycle increments allowed_fails only once."""
        calls: list[str] = []

        class FakeClient:
            def __init__(self, **_: Any) -> None:
                pass

            async def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, Any],
                timeout: Any,
            ) -> FakeResponse:
                calls.append(url)
                if "api.a.example" in url:
                    return _error(503)
                return _chat_ok("fallback")

            async def aclose(self) -> None:
                pass

        client = TestClient(self.app_module.app)
        orig_client, orig_state = self.swap_state_and_client(self.state, FakeClient)
        try:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        finally:
            self.restore_state_and_client(orig_client, orig_state)

        provider_a = self.state.providers_by_model["gpt-4"][0]
        failure_key = self.app_module.build_failure_key(provider_a, "/chat/completions")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-fallback-provider-id"], "openai-provider-b")
        self.assertEqual(len([url for url in calls if "api.a.example" in url]), 3)
        self.assertEqual(len([url for url in calls if "api.b.example" in url]), 1)
        self.assertEqual(self.state.fail_counts[failure_key], 1)
        self.assertNotIn(failure_key, self.state.cooldown_until)

    def test_auth_error_is_not_retried(self):
        """401 falls back immediately even when retries are configured."""
        calls: list[str] = []

        class FakeClient:
            def __init__(self, **_: Any) -> None:
                pass

            async def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, Any],
                timeout: Any,
            ) -> FakeResponse:
                calls.append(url)
                if "api.a.example" in url:
                    return _error(401)
                return _chat_ok("fallback")

            async def aclose(self) -> None:
                pass

        client = TestClient(self.app_module.app)
        orig_client, orig_state = self.swap_state_and_client(self.state, FakeClient)
        try:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        finally:
            self.restore_state_and_client(orig_client, orig_state)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-fallback-provider-id"], "openai-provider-b")
        self.assertEqual(len([url for url in calls if "api.a.example" in url]), 1)
        self.assertEqual(len([url for url in calls if "api.b.example" in url]), 1)

    def test_responses_stream_start_error_is_retried_before_fallback(self):
        cfg = build_config(
            anthropic_providers=0,
            extra_settings={
                "app_settings": {"hot_reload": False, "stream_start_timeout": 1},
                "router_settings": {
                    "allowed_fails": 1,
                    "cooldown_time": 300,
                    "allowed_retries": 1,
                    "retry_backoff_seconds": 0,
                },
            },
        )
        for index, provider in enumerate(cfg["providers"]):
            provider["endpoint_type"] = "responses"
            provider["models"] = ["gpt-test"]
            provider["name"] = f"responses-{index}"
        cfg_path = self.config_path.parent / "responses_retry.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        retry_state = self.app_module.RouterState.create_sync(str(cfg_path))
        calls: list[str] = []

        class FakeStreamResponse:
            def __init__(self, chunks: list[bytes]) -> None:
                self.status_code = 200
                self.headers = {"content-type": "text/event-stream"}
                self._chunks = chunks
                self.is_closed = False
                self.is_stream_consumed = False

            async def aiter_bytes(self) -> AsyncIterator[bytes]:
                if self.is_stream_consumed:
                    raise httpx.StreamConsumed()
                self.is_stream_consumed = True
                for chunk in self._chunks:
                    yield chunk

            async def aread(self) -> bytes:
                return b"".join(self._chunks)

            async def aclose(self) -> None:
                self.is_closed = True

        class FakeClient:
            def __init__(self, **_: Any) -> None:
                pass

            def build_request(
                self,
                method: str,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, Any],
                timeout: Any,
            ) -> dict[str, Any]:
                return {"method": method, "url": url, "headers": headers, "json": json}

            async def send(
                self,
                request: dict[str, Any],
                *,
                stream: bool,
            ) -> FakeStreamResponse:
                calls.append(request["url"])
                if "api.a.example" in request["url"] and calls.count(request["url"]) == 1:
                    return FakeStreamResponse(
                        [
                            (
                                "event: error\n"
                                'data: {"error":{"message":"temporarily overloaded"}}\n\n'
                            ).encode()
                        ]
                    )
                return FakeStreamResponse(
                    [
                        (
                            "event: response.completed\n"
                            'data: {"type":"response.completed","response":'
                            '{"id":"resp-test","status":"completed"}}\n\n'
                        ).encode()
                    ]
                )

            async def aclose(self) -> None:
                pass

        client = TestClient(self.app_module.app)
        orig_client, orig_state = self.swap_state_and_client(retry_state, FakeClient)
        try:
            with client.stream(
                "POST",
                "/v1/responses",
                json={"model": "gpt-test", "input": "hello", "stream": True},
            ) as response:
                body = response.read().decode()
        finally:
            self.restore_state_and_client(orig_client, orig_state)

        self.assertEqual(response.status_code, 200)
        self.assertIn("response.completed", body)
        self.assertEqual(response.headers["x-fallback-provider-id"], "responses-0")
        self.assertEqual(len([url for url in calls if "api.a.example" in url]), 2)
        self.assertFalse(any("api.b.example" in url for url in calls))
