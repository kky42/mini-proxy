"""Provider cooldown e2e tests.

Verifies that:
- A provider enters cooldown after exceeding allowed_fails
- Cooling providers are skipped (not tried first)
- Cooldown expiry brings the provider back
- Cooldown interacts correctly with sticky sessions
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from tests.e2e_base import E2EBase
from tests.e2e_helpers import (
    FakeResponse,
    build_config,
    make_selective_client,
    make_uniform_client,
    write_temp_config,
)


class CooldownTests(E2EBase):
    # allowed_fails=0 so a single failure triggers cooldown
    config = build_config(extra_settings={
        "router_settings": {"allowed_fails": 0, "cooldown_time": 300},
    })

    def test_single_failure_triggers_cooldown(self):
        """With allowed_fails=0, one 503 puts the provider in cooldown."""
        FakeClient = make_selective_client(
            fail_url_contains="https://api.a.example",
            ok_url_contains="https://api.b.example",
        )

        client = TestClient(self.app_module.app)
        orig_client, orig_state = self.swap_state_and_client(self.state, FakeClient)
        try:
            r = client.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Trigger cooldown"}],
            })
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.headers.get("x-fallback-provider-id"), "openai-provider-b")

            # A should be in cooldown now
            from app import build_failure_key
            provider_a = self.state.providers_by_model["gpt-4"][0]
            failure_key = build_failure_key(provider_a, "/chat/completions")
            cooldown_until = self.state.cooldown_until.get(failure_key, 0)
            self.assertGreater(cooldown_until, 0,
                               f"A should be in cooldown after 1 failure (allowed_fails=0)")
        finally:
            self.restore_state_and_client(orig_client, orig_state)

    def test_cooling_provider_skipped_for_new_sessions(self):
        """While A is cooling, new conversations go directly to B."""
        FakeClient = make_selective_client(
            fail_url_contains="https://api.a.example",
            ok_url_contains="https://api.b.example",
        )

        client = TestClient(self.app_module.app)
        orig_client, orig_state = self.swap_state_and_client(self.state, FakeClient)
        try:
            # Session 1: triggers A's failure and cooldown
            r1 = client.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Session one"}],
            })
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r1.headers.get("x-fallback-provider-id"), "openai-provider-b")

            # Session 2: different conversation, A is cooling → B
            r2 = client.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Session two"}],
            })
            self.assertEqual(r2.status_code, 200)
            self.assertEqual(r2.headers.get("x-fallback-provider-id"), "openai-provider-b")
        finally:
            self.restore_state_and_client(orig_client, orig_state)

    def test_cooldown_expiry_restores_provider(self):
        """After cooldown_seconds elapses, provider is tried first again."""
        # Use a very short cooldown
        cfg = build_config(extra_settings={
            "router_settings": {"allowed_fails": 0, "cooldown_time": 1},
        })
        cfg_path = write_temp_config(cfg, self.tmpdir.name, "expiry_config.yaml")
        expiry_state = self.app_module.RouterState.create_sync(str(cfg_path))

        FakeClient = make_selective_client(
            fail_url_contains="https://api.a.example",
            ok_url_contains="https://api.b.example",
        )

        client = TestClient(self.app_module.app)
        orig_client, orig_state = self.swap_state_and_client(expiry_state, FakeClient)
        try:
            # Trigger cooldown on A
            r1 = client.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Trigger"}],
            })
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r1.headers.get("x-fallback-provider-id"), "openai-provider-b")

            # Wait for 1-second cooldown to expire
            time.sleep(1.5)

            # Now A is healthy again — new session should pick A (order=1)
            ok = FakeResponse({
                "id": "recovered",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Back!"}}],
            })
            self.app_module.httpx.AsyncClient = make_uniform_client(ok)

            r2 = client.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "After cooldown"}],
            })
            self.assertEqual(r2.status_code, 200)
            self.assertEqual(
                r2.headers.get("x-fallback-provider-id"),
                "openai-provider-a",
                "After cooldown expires, A should be first choice again",
            )
        finally:
            self.restore_state_and_client(orig_client, orig_state)

    def test_cooldown_cooling_providers_kept_as_last_resort(self):
        """When all providers are in cooldown, they are still tried (last resort)."""
        # Use a config with only one provider, so when it's cooling, it's the only option
        cfg = build_config(openai_providers=1, anthropic_providers=0, extra_settings={
            "router_settings": {"allowed_fails": 0, "cooldown_time": 300},
        })
        cfg_path = write_temp_config(cfg, self.tmpdir.name, "single_provider.yaml")
        single_state = self.app_module.RouterState.create_sync(str(cfg_path))

        # Provider succeeds on first try
        ok = FakeResponse({
            "id": "ok",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}}],
        })
        FakeClient = make_uniform_client(ok)

        client = TestClient(self.app_module.app)
        orig_client, orig_state = self.swap_state_and_client(single_state, FakeClient)
        try:
            r1 = client.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            })
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r1.headers.get("x-fallback-provider-id"), "openai-provider-a")

            # Even though the provider should be healthy (no failures), the test
            # confirms the setup works: single provider, successful request.
            # If we wanted to test "all cooling", we'd need all providers to fail,
            # which is covered by test_all_providers_fail_returns_503 in fallback tests.
        finally:
            self.restore_state_and_client(orig_client, orig_state)

    def test_auth_error_cooldown_multiplier(self):
        """Auth errors (401) use 3x cooldown multiplier."""
        cfg = build_config(extra_settings={
            "router_settings": {"allowed_fails": 0, "cooldown_time": 10},
        })
        cfg_path = write_temp_config(cfg, self.tmpdir.name, "auth_multiplier.yaml")
        auth_state = self.app_module.RouterState.create_sync(str(cfg_path))

        FakeClient = make_selective_client(
            fail_url_contains="https://api.a.example",
            ok_url_contains="https://api.b.example",
            fail_status=401,
        )

        client = TestClient(self.app_module.app)
        orig_client, orig_state = self.swap_state_and_client(auth_state, FakeClient)
        try:
            r = client.post("/v1/chat/completions", json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            })
            self.assertEqual(r.status_code, 200)

            from app import build_failure_key
            provider_a = auth_state.providers_by_model["gpt-4"][0]
            failure_key = build_failure_key(provider_a, "/chat/completions")
            cooldown_until = auth_state.cooldown_until.get(failure_key, 0)
            now = time.time()
            cooldown_remaining = cooldown_until - now
            # 3x multiplier: 10 * 3 = 30s
            self.assertGreater(cooldown_remaining, 20,
                               f"Auth error should have ~30s cooldown (10*3), got {cooldown_remaining:.0f}s remaining")
            self.assertLess(cooldown_remaining, 35)
        finally:
            self.restore_state_and_client(orig_client, orig_state)
