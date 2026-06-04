from __future__ import annotations

import unittest
from types import SimpleNamespace

from starlette.datastructures import Headers

from src.forwarding import build_upstream_headers
from src.types import Provider


def make_provider(*, extra_headers: dict[str, str] | None = None) -> Provider:
    return Provider(
        provider_name="provider",
        model_name="model",
        configured_model="model",
        upstream_model="model",
        anthropic_role=None,
        endpoint_type=None,
        api_base="https://provider.example",
        api_url=None,
        models_url=None,
        api_key="sk-provider",
        order=1,
        timeout=None,
        extra_headers=extra_headers or {},
    )


def make_request(headers: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(headers=Headers(headers))


class UpstreamHeaderForwardingTests(unittest.TestCase):
    def test_responses_relay_uses_codex_allowlist(self) -> None:
        headers = build_upstream_headers(
            make_request(
                {
                    "Authorization": "Bearer client",
                    "Cookie": "sid=secret",
                    "Origin": "https://client.example",
                    "Referer": "https://client.example/page",
                    "x-fallback-session": "internal-session",
                    "x-stainless-lang": "js",
                    "User-Agent": "codex_exec/0.136.0",
                    "originator": "codex_cli_rs",
                    "session-id": "session-1",
                    "thread-id": "thread-1",
                    "x-client-request-id": "request-1",
                    "x-codex-beta-features": "terminal_resize_reflow",
                    "x-codex-turn-metadata": "{}",
                    "x-codex-window-id": "window-1",
                }
            ),
            make_provider(),
            "/responses",
        )

        lowered = {key.lower() for key in headers}
        self.assertEqual(headers["Authorization"], "Bearer sk-provider")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertIn("user-agent", lowered)
        self.assertIn("originator", lowered)
        self.assertIn("session-id", lowered)
        self.assertIn("thread-id", lowered)
        self.assertIn("x-client-request-id", lowered)
        self.assertIn("x-codex-beta-features", lowered)
        self.assertIn("x-codex-turn-metadata", lowered)
        self.assertIn("x-codex-window-id", lowered)
        self.assertNotIn("cookie", lowered)
        self.assertNotIn("origin", lowered)
        self.assertNotIn("referer", lowered)
        self.assertNotIn("x-fallback-session", lowered)
        self.assertNotIn("x-stainless-lang", lowered)

    def test_anthropic_relay_uses_claude_code_allowlist(self) -> None:
        headers = build_upstream_headers(
            make_request(
                {
                    "Authorization": "Bearer client",
                    "Cookie": "sid=secret",
                    "x-fallback-session": "internal-session",
                    "session-id": "codex-session",
                    "x-codex-window-id": "codex-window",
                    "User-Agent": "claude-cli/2.1.154",
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "claude-code-20250219",
                    "anthropic-dangerous-direct-browser-access": "true",
                    "x-app": "cli",
                    "x-claude-code-session-id": "claude-session",
                    "x-stainless-lang": "js",
                    "x-stainless-timeout": "600",
                }
            ),
            make_provider(),
            "/messages",
        )

        lowered = {key.lower() for key in headers}
        self.assertEqual(headers["x-api-key"], "sk-provider")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertIn("user-agent", lowered)
        self.assertIn("anthropic-version", lowered)
        self.assertIn("anthropic-beta", lowered)
        self.assertIn("anthropic-dangerous-direct-browser-access", lowered)
        self.assertIn("x-app", lowered)
        self.assertIn("x-claude-code-session-id", lowered)
        self.assertIn("x-stainless-lang", lowered)
        self.assertIn("x-stainless-timeout", lowered)
        self.assertNotIn("authorization", lowered)
        self.assertNotIn("cookie", lowered)
        self.assertNotIn("x-fallback-session", lowered)
        self.assertNotIn("session-id", lowered)
        self.assertNotIn("x-codex-window-id", lowered)

    def test_connection_scoped_headers_are_not_relayed_even_when_allowlisted(self) -> None:
        headers = build_upstream_headers(
            make_request(
                {
                    "Connection": "session-id, x-codex-turn-metadata, x-debug",
                    "User-Agent": "codex_exec/0.136.0",
                    "session-id": "session-1",
                    "x-codex-turn-metadata": "{}",
                    "x-debug": "must-drop",
                }
            ),
            make_provider(),
            "/responses",
        )

        lowered = {key.lower() for key in headers}
        self.assertIn("user-agent", lowered)
        self.assertNotIn("connection", lowered)
        self.assertNotIn("session-id", lowered)
        self.assertNotIn("x-codex-turn-metadata", lowered)
        self.assertNotIn("x-debug", lowered)

    def test_provider_extra_headers_can_override_relayed_headers(self) -> None:
        headers = build_upstream_headers(
            make_request({"User-Agent": "client", "x-codex-window-id": "client-window"}),
            make_provider(extra_headers={"User-Agent": "provider", "X-Provider": "configured"}),
            "/responses",
        )

        self.assertEqual(headers["User-Agent"], "provider")
        self.assertEqual(headers["X-Provider"], "configured")


if __name__ == "__main__":
    unittest.main()
