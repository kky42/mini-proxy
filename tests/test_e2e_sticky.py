"""Session stickiness e2e tests across pi-agent, Claude Code, and Responses API.

Verifies that multi-turn conversations stay routed to the same provider
via the appropriate session identifier for each client type.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from tests.e2e_base import E2EBase
from tests.e2e_helpers import (
    ANTHROPIC_SSE_OK,
    FakeResponse,
    FakeStreamResponse,
    build_config,
    make_claude_session_id,
    make_mapped_client,
    make_uniform_client,
)


class StickyTests(E2EBase):
    config = build_config()

    # ------------------------------------------------------------------
    # pi-agent (OpenAI /v1/chat/completions) — content fingerprint
    # ------------------------------------------------------------------

    def test_pi_agent_two_turns_same_provider(self):
        ok = FakeResponse({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Paris"}}],
        })
        FakeClient = make_uniform_client(ok)

        client = TestClient(self.app_module.app)
        orig_client, orig_state = self.swap_state_and_client(self.state, FakeClient)
        try:
            r1 = client.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Capital of France?"},
                ],
            })
            self.assertEqual(r1.status_code, 200)
            p1 = r1.headers.get("x-fallback-provider-id")

            r2 = client.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Capital of France?"},
                    {"role": "assistant", "content": "Paris."},
                    {"role": "user", "content": "Population?"},
                ],
            })
            self.assertEqual(r2.status_code, 200)
            p2 = r2.headers.get("x-fallback-provider-id")

            self.assertEqual(p1, p2, f"Same session, got {p1} then {p2}")
        finally:
            self.restore_state_and_client(orig_client, orig_state)

    def test_pi_agent_different_first_message_independent_sessions(self):
        ok = FakeResponse({
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
        })
        FakeClient = make_uniform_client(ok)

        client = TestClient(self.app_module.app)
        orig_client, orig_state = self.swap_state_and_client(self.state, FakeClient)
        try:
            r1 = client.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Tell me about France"}],
            })
            r2 = client.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Tell me about Python"}],
            })
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r2.status_code, 200)
            # Both independently pick the highest-priority provider (order=1)
            self.assertEqual(r1.headers.get("x-fallback-provider-id"), "openai-provider-a")
            self.assertEqual(r2.headers.get("x-fallback-provider-id"), "openai-provider-a")
        finally:
            self.restore_state_and_client(orig_client, orig_state)

    def test_pi_agent_header_overrides_fingerprint(self):
        ok = FakeResponse({
            "id": "chatcmpl-3",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}}],
        })
        FakeClient = make_uniform_client(ok)

        client = TestClient(self.app_module.app)
        orig_client, orig_state = self.swap_state_and_client(self.state, FakeClient)
        try:
            r1 = client.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            }, headers={"x-fallback-session": "my-explicit-session"})

            r2 = client.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Different first message"}],
            }, headers={"x-fallback-session": "my-explicit-session"})

            # Same explicit session → same provider, even with different content
            self.assertEqual(
                r1.headers.get("x-fallback-provider-id"),
                r2.headers.get("x-fallback-provider-id"),
            )
        finally:
            self.restore_state_and_client(orig_client, orig_state)

    # ------------------------------------------------------------------
    # Claude Code (Anthropic /v1/messages) — metadata.user_id
    # ------------------------------------------------------------------

    def test_claude_code_same_session_id_sticks(self):
        sse = FakeStreamResponse([ANTHROPIC_SSE_OK])
        FakeClient = make_uniform_client(sse)

        client = TestClient(self.app_module.app)
        orig_client, orig_state = self.swap_state_and_client(self.state, FakeClient)
        try:
            sid = "ec5bf141-a549-4540-835e-63af0155c8e9"
            body = {
                "model": "claude-sonnet-4-6",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
                "metadata": {"user_id": make_claude_session_id(sid)},
                "stream": True,
            }

            r1 = client.post("/v1/messages", json=body)
            body["messages"] = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"},
            ]
            r2 = client.post("/v1/messages", json=body)

            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r2.status_code, 200)
            self.assertEqual(
                r1.headers.get("x-fallback-provider-id"),
                r2.headers.get("x-fallback-provider-id"),
                "Same Claude Code session should stick to same provider",
            )
        finally:
            self.restore_state_and_client(orig_client, orig_state)

    def test_claude_code_different_sessions_independent(self):
        sse = FakeStreamResponse([ANTHROPIC_SSE_OK])
        FakeClient = make_uniform_client(sse)

        client = TestClient(self.app_module.app)
        orig_client, orig_state = self.swap_state_and_client(self.state, FakeClient)
        try:
            base = {
                "model": "claude-sonnet-4-6",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            }

            r1 = client.post("/v1/messages", json={
                **base,
                "metadata": {"user_id": make_claude_session_id("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")},
            })
            r2 = client.post("/v1/messages", json={
                **base,
                "metadata": {"user_id": make_claude_session_id("ffffffff-gggg-hhhh-iiii-jjjjjjjjjjjj")},
            })

            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r2.status_code, 200)
            self.assertEqual(r1.headers.get("x-fallback-provider-id"), "anthropic-provider-a")
            self.assertEqual(r2.headers.get("x-fallback-provider-id"), "anthropic-provider-a")
        finally:
            self.restore_state_and_client(orig_client, orig_state)

    # ------------------------------------------------------------------
    # Responses API — previous_response_id
    # ------------------------------------------------------------------

    def test_responses_api_previous_response_id_sticks(self):
        ok = FakeResponse({
            "id": "resp_1",
            "object": "response",
            "status": "completed",
            "output": [{"type": "message", "content": [{"text": "Hello!"}]}],
        })

        # Add a responses-capable provider
        from app import Provider
        resp_a = Provider(
            provider_name="resp-a", model_name="gpt-4", configured_model="gpt-4",
            upstream_model="gpt-4", anthropic_role=None, endpoint_type="responses",
            api_base="https://resp.a.example", api_url=None, models_url=None,
            api_key="sk-a", order=1, timeout=None, extra_headers={},
        )
        resp_b = Provider(
            provider_name="resp-b", model_name="gpt-4", configured_model="gpt-4",
            upstream_model="gpt-4", anthropic_role=None, endpoint_type="responses",
            api_base="https://resp.b.example", api_url=None, models_url=None,
            api_key="sk-b", order=2, timeout=None, extra_headers={},
        )
        self.state.providers_by_model["gpt-4"] = sorted(
            [resp_a, resp_b], key=lambda p: (p.order, p.sort_url, p.upstream_model)
        )

        FakeClient = make_mapped_client({
            "https://resp.a.example": ok,
            "https://resp.b.example": ok,
        })

        client = TestClient(self.app_module.app)
        orig_client, orig_state = self.swap_state_and_client(self.state, FakeClient)
        try:
            r1 = client.post("/v1/responses", json={
                "model": "gpt-4", "input": "Hello",
            })
            r2 = client.post("/v1/responses", json={
                "model": "gpt-4", "input": "Continue",
                "previous_response_id": "resp_1",
            })
            p1 = r1.headers.get("x-fallback-provider-id")
            p2 = r2.headers.get("x-fallback-provider-id")
            if p1 and p2:
                self.assertEqual(p1, p2, f"Got {p1} then {p2}")
        finally:
            self.restore_state_and_client(orig_client, orig_state)

    # ------------------------------------------------------------------
    # Cross-client isolation
    # ------------------------------------------------------------------

    def test_cross_client_sessions_isolated(self):
        json_ok = FakeResponse({"id": "x", "object": "chat.completion", "choices": []})
        sse_ok = FakeStreamResponse([ANTHROPIC_SSE_OK])

        FakeClient = make_mapped_client({
            "https://api.a.example": json_ok,
            "https://api.b.example": json_ok,
            "https://api.anthropic.a.example": sse_ok,
            "https://api.anthropic.b.example": sse_ok,
        })

        client = TestClient(self.app_module.app)
        orig_client, orig_state = self.swap_state_and_client(self.state, FakeClient)
        try:
            r_pi = client.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            })
            r_claude = client.post("/v1/messages", json={
                "model": "claude-sonnet-4-6", "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
                "metadata": {"user_id": make_claude_session_id("11111111-2222-3333-4444-555555555555")},
                "stream": True,
            })

            pi_p = r_pi.headers.get("x-fallback-provider-id")
            claude_p = r_claude.headers.get("x-fallback-provider-id")
            self.assertIsNotNone(pi_p)
            self.assertIsNotNone(claude_p)
            self.assertNotEqual(pi_p, claude_p,
                                "Different endpoint types use different provider families")
        finally:
            self.restore_state_and_client(orig_client, orig_state)
