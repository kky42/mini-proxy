from __future__ import annotations

import asyncio
import inspect
import importlib
import os
import tempfile
import time
import unittest
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import yaml
from fastapi.testclient import TestClient


def write_config(
    path: Path,
    *,
    api_base: str,
    order: int = 1,
    extra_providers: list[tuple[str, int]] | None = None,
) -> None:
    providers = [
        {
            "name": "primary",
            "api_base": api_base,
            "api_key": "sk-test",
            "order": order,
            "models": ["gpt-test"],
        }
    ]
    for index, (extra_api_base, extra_order) in enumerate(extra_providers or [], start=2):
        providers.append(
            {
                "name": f"provider-{index}",
                "api_base": extra_api_base,
                "api_key": "sk-test",
                "order": extra_order,
                "models": ["gpt-test"],
            }
        )

    data: dict[str, Any] = {
        "app_settings": {
            "hot_reload": True,
            "hot_reload_interval_seconds": 0.1,
            "default_timeout": 60,
        },
        "router_settings": {
            "allowed_fails": 1,
            "cooldown_time": 300,
        },
        "providers": providers,
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def touch_newer(path: Path, previous_mtime: float) -> None:
    new_mtime = previous_mtime + 2
    os.utime(path, (new_mtime, new_mtime))


with tempfile.TemporaryDirectory() as import_config_dir:
    import_config_path = Path(import_config_dir) / "config.yaml"
    write_config(import_config_path, api_base="https://initial.example/v1")
    os.environ["MINI_FALLBACK_PROXY_CONFIG"] = str(import_config_path)
    app_module = importlib.import_module("app")


def install_fake_async_get(handler: Any) -> Any:
    class FakeAsyncClient:
        def __init__(self, **_: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_: Any) -> None:
            pass

        async def get(self, url: str, **kwargs: Any) -> httpx.Response:
            result = handler(url, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result

    original_client = app_module.httpx.AsyncClient
    app_module.httpx.AsyncClient = FakeAsyncClient
    return original_client


class HotReloadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "config.yaml"
        write_config(self.config_path, api_base="https://one.example/v1")
        self.state = app_module.RouterState.create_sync(str(self.config_path))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_reload_if_changed_applies_valid_config(self) -> None:
        previous_mtime = os.path.getmtime(self.config_path)
        write_config(self.config_path, api_base="https://two.example/v1", order=2)
        touch_newer(self.config_path, previous_mtime)

        result = await self.state.reload_if_changed()
        models = await self.state.list_models()

        self.assertEqual(result.status, "reloaded")
        self.assertTrue(result.reloaded)
        self.assertEqual(models[0]["providers"][0]["api_base"], "https://two.example/v1")
        self.assertEqual(models[0]["providers"][0]["order"], 2)

    async def test_invalid_hot_reload_keeps_last_good_config(self) -> None:
        previous_mtime = os.path.getmtime(self.config_path)
        self.config_path.write_text("providers: not-a-list\n", encoding="utf-8")
        touch_newer(self.config_path, previous_mtime)

        result = await self.state.reload_if_changed()
        models = await self.state.list_models()
        snapshot = await self.state.snapshot()

        self.assertEqual(result.status, "rejected")
        self.assertFalse(result.reloaded)
        self.assertEqual(models[0]["providers"][0]["api_base"], "https://one.example/v1")
        self.assertIn("providers must be a list", snapshot["hot_reload"]["last_error"])

    async def test_manual_reload_rejects_invalid_config(self) -> None:
        self.config_path.write_text("app_settings: []\n", encoding="utf-8")

        result = await self.state.reload()
        models = await self.state.list_models()

        self.assertEqual(result.status, "rejected")
        self.assertFalse(result.reloaded)
        self.assertEqual(models[0]["providers"][0]["api_base"], "https://one.example/v1")

    def test_extract_session_key_uses_responses_prompt_cache_key(self) -> None:
        request = SimpleNamespace(headers={})

        session_key = app_module._extract_session_key(
            request,
            {
                "model": "gpt-test",
                "prompt_cache_key": " 019e6d5d-27c5-7244-a97c-71b41d81f3b9 ",
            },
        )

        self.assertEqual(
            session_key,
            "prompt_cache_key|019e6d5d-27c5-7244-a97c-71b41d81f3b9",
        )

    def test_extract_session_key_uses_claude_metadata_user_id_session_id(self) -> None:
        request = SimpleNamespace(headers={})

        session_key = app_module._extract_session_key(
            request,
            {
                "model": "claude-sonnet-4-6",
                "metadata": {
                    "user_id": (
                        '{"device_id":"device-test",'
                        '"account_uuid":"",'
                        '"session_id":"ec5bf141-a549-4540-835e-63af0155c8e9"}'
                    ),
                },
            },
        )

        self.assertEqual(
            session_key,
            "metadata:user_id:session_id|ec5bf141-a549-4540-835e-63af0155c8e9",
        )

    def test_extract_session_key_openai_chat_fingerprint(self) -> None:
        request = SimpleNamespace(
            headers={},
            client=SimpleNamespace(host="127.0.0.1"),
        )

        session_key = app_module._extract_session_key(
            request,
            {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hello world"},
                ],
            },
            endpoint="/chat/completions",
        )

        self.assertIsNotNone(session_key)
        self.assertTrue(session_key.startswith("content|"))
        self.assertEqual(len(session_key), 24)  # "content|" + 16 hex chars

    def test_extract_session_key_openai_chat_fingerprint_stable(self) -> None:
        request = SimpleNamespace(
            headers={},
            client=SimpleNamespace(host="127.0.0.1"),
        )

        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "hello world"},
            ],
        }

        key1 = app_module._extract_session_key(
            request, payload, endpoint="/chat/completions"
        )
        key2 = app_module._extract_session_key(
            request, payload, endpoint="/chat/completions"
        )

        self.assertEqual(key1, key2)

    def test_extract_session_key_openai_chat_different_messages_different_fingerprints(self) -> None:
        request = SimpleNamespace(
            headers={},
            client=SimpleNamespace(host="127.0.0.1"),
        )

        key1 = app_module._extract_session_key(
            request,
            {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
            endpoint="/chat/completions",
        )
        key2 = app_module._extract_session_key(
            request,
            {"model": "gpt-4", "messages": [{"role": "user", "content": "world"}]},
            endpoint="/chat/completions",
        )

        self.assertNotEqual(key1, key2)

    def test_extract_session_key_openai_chat_no_messages_returns_none(self) -> None:
        request = SimpleNamespace(
            headers={},
            client=SimpleNamespace(host="127.0.0.1"),
        )

        session_key = app_module._extract_session_key(
            request,
            {"model": "gpt-4", "messages": []},
            endpoint="/chat/completions",
        )

        self.assertIsNone(session_key)

    def test_extract_session_key_openai_chat_no_user_message_returns_none(self) -> None:
        request = SimpleNamespace(
            headers={},
            client=SimpleNamespace(host="127.0.0.1"),
        )

        session_key = app_module._extract_session_key(
            request,
            {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                ],
            },
            endpoint="/chat/completions",
        )

        self.assertIsNone(session_key)

    def test_extract_session_key_openai_chat_header_takes_priority(self) -> None:
        request = SimpleNamespace(
            headers={"x-fallback-session": "my-session-id"},
            client=SimpleNamespace(host="127.0.0.1"),
        )

        session_key = app_module._extract_session_key(
            request,
            {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hello"}],
            },
            endpoint="/chat/completions",
        )

        self.assertEqual(session_key, "header|my-session-id")

    def test_extract_session_key_openai_chat_streaming_preserves_sticky(self) -> None:
        """Fingerprint works when stream=True (same as non-streaming)."""
        request = SimpleNamespace(
            headers={},
            client=SimpleNamespace(host="10.0.0.1"),
        )

        payload = {
            "model": "gpt-4",
            "stream": True,
            "messages": [{"role": "user", "content": "explain quantum computing"}],
        }

        session_key = app_module._extract_session_key(
            request, payload, endpoint="/chat/completions"
        )
        self.assertIsNotNone(session_key)
        self.assertTrue(session_key.startswith("content|"))

    async def test_reload_auto_model_discovery_does_not_block_event_loop(self) -> None:
        self.config_path.write_text(
            yaml.safe_dump(
                {
                    "providers": [
                        {
                            "name": "auto",
                            "api_base": "https://auto.example",
                            "api_key": "sk-auto",
                            "endpoint_type": "openai-compatible",
                            "models": "auto",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        entered = asyncio.Event()
        release = asyncio.Event()

        async def fake_get(url: str, **kwargs: Any) -> httpx.Response:
            entered.set()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(release.wait(), timeout=0.25)
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "gpt-auto", "object": "model"}]},
            )

        original_client = install_fake_async_get(fake_get)
        reload_task = asyncio.create_task(self.state.reload())
        try:
            start = time.perf_counter()
            while not entered.is_set() and time.perf_counter() - start < 0.5:
                await asyncio.sleep(0.005)

            elapsed = time.perf_counter() - start
            self.assertTrue(entered.is_set())
            self.assertLess(elapsed, 0.1)
            release.set()
            result = await reload_task
        finally:
            release.set()
            app_module.httpx.AsyncClient = original_client
            if not reload_task.done():
                reload_task.cancel()

        self.assertEqual(result.status, "reloaded")
        models = await self.state.list_models()
        self.assertEqual([model["id"] for model in models], ["gpt-auto"])

    async def test_reload_if_changed_ignores_unchanged_file(self) -> None:
        await asyncio.sleep(0)

        result = await self.state.reload_if_changed()

        self.assertEqual(result.status, "unchanged")
        self.assertFalse(result.reloaded)

    async def test_payment_required_falls_back_and_counts_for_cooldown(self) -> None:
        decision = app_module.classify_http_error(
            402,
            b'{"error":{"message":"Insufficient balance"}}',
            "/responses",
        )

        self.assertEqual(decision.failure_class, app_module.FailureClass.AUTH_OR_BALANCE)
        self.assertTrue(decision.should_fallback)
        self.assertTrue(decision.count_failure)
        self.assertEqual(decision.cooldown_multiplier, 3.0)

    async def test_cooling_providers_are_kept_as_last_resort(self) -> None:
        write_config(
            self.config_path,
            api_base="https://one.example/v1",
            extra_providers=[("https://two.example/v1", 2)],
        )
        self.state = app_module.RouterState.create_sync(str(self.config_path))
        providers = self.state.providers_by_model["gpt-test"]
        cooling_provider = providers[1]
        self.state.cooldown_until[
            app_module.build_failure_key(cooling_provider, "/responses")
        ] = 9999999999

        candidates = await self.state.get_candidate_providers(
            "gpt-test",
            "/responses",
            sticky_key=None,
        )

        self.assertEqual([provider.api_base for provider in candidates], [
            "https://one.example/v1",
            "https://two.example/v1",
        ])

    async def test_provider_level_models_expand_to_model_routes(self) -> None:
        data: dict[str, Any] = {
            "providers": [
                {
                    "name": "first",
                    "api_base": "https://first.example/v1",
                    "api_key": "sk-first",
                    "order": 1,
                    "models": [
                        "gpt-a",
                        {"model_name": "gpt-b", "model": "openai/provider-gpt-b"},
                    ],
                },
                {
                    "name": "second",
                    "api_base": "https://second.example/v1",
                    "api_key": "sk-second",
                    "order": 2,
                    "models": ["gpt-a"],
                },
                {
                    "name": "third",
                    "api_base": "https://third.example/v1",
                    "api_key": "sk-third",
                    "order": 3,
                    "models": ["gpt-c"],
                },
            ]
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.state = app_module.RouterState.create_sync(str(self.config_path))

        gpt_a_candidates = await self.state.get_candidate_providers(
            "gpt-a",
            "/responses",
            sticky_key=None,
        )
        gpt_b_candidates = await self.state.get_candidate_providers(
            "gpt-b",
            "/responses",
            sticky_key=None,
        )
        models = await self.state.list_models()

        self.assertEqual(
            [provider.api_base for provider in gpt_a_candidates],
            ["https://first.example/v1", "https://second.example/v1"],
        )
        self.assertEqual(
            [provider.api_base for provider in gpt_b_candidates],
            ["https://first.example/v1"],
        )
        self.assertEqual(gpt_b_candidates[0].configured_model, "openai/provider-gpt-b")
        self.assertEqual(gpt_b_candidates[0].upstream_model, "provider-gpt-b")
        self.assertEqual([model["id"] for model in models], ["gpt-a", "gpt-b", "gpt-c"])

    async def test_provider_name_is_required_and_unique(self) -> None:
        self.config_path.write_text(
            yaml.safe_dump(
                {
                    "providers": [
                        {
                            "api_base": "https://missing-name.example/v1",
                            "api_key": "sk-one",
                            "models": ["gpt-test"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "providers\\[0\\]\\.name is required"):
            app_module.RouterState.create_sync(str(self.config_path))

        self.config_path.write_text(
            yaml.safe_dump(
                {
                    "providers": [
                        {
                            "name": "duplicate",
                            "api_base": "https://one.example/v1",
                            "api_key": "sk-one",
                            "models": ["gpt-test"],
                        },
                        {
                            "name": "duplicate",
                            "api_base": "https://two.example/v1",
                            "api_key": "sk-two",
                            "models": ["gpt-test"],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "providers\\[1\\]\\.name must be unique: duplicate",
        ):
            app_module.RouterState.create_sync(str(self.config_path))

    async def test_provider_name_is_provider_id_for_same_route_url(self) -> None:
        data: dict[str, Any] = {
            "providers": [
                {
                    "name": "same-url-one",
                    "api_base": "https://same.example/v1",
                    "api_key": "sk-one",
                    "order": 1,
                    "models": ["gpt-test"],
                },
                {
                    "name": "same-url-two",
                    "api_base": "https://same.example/v1",
                    "api_key": "sk-two",
                    "order": 1,
                    "models": ["gpt-test"],
                },
            ]
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.state = app_module.RouterState.create_sync(str(self.config_path))

        candidates = await self.state.get_candidate_providers(
            "gpt-test",
            "/responses",
            sticky_key=None,
        )
        await self.state.record_failure(
            candidates[0],
            "/responses",
            "synthetic failure",
            app_module.FailureDecision(
                failure_class=app_module.FailureClass.AVAILABILITY,
                should_fallback=True,
                count_failure=True,
            ),
        )
        snapshot = await self.state.snapshot()

        self.assertEqual(
            [provider.provider_id for provider in candidates],
            ["same-url-one", "same-url-two"],
        )
        self.assertEqual(
            {
                provider["provider_name"]: provider["provider_id"]
                for provider in snapshot["providers"]
            },
            {
                "same-url-one": "same-url-one",
                "same-url-two": "same-url-two",
            },
        )
        self.assertEqual(
            {
                provider["provider_name"]: provider["last_error"]["/responses"]
                for provider in snapshot["providers"]
            },
            {
                "same-url-one": "synthetic failure",
                "same-url-two": None,
            },
        )

    async def test_config_load_logs_provider_model_summary_without_api_keys(self) -> None:
        data: dict[str, Any] = {
            "providers": [
                {
                    "name": "responses",
                    "api_base": "https://responses.example",
                    "api_key": "sk-secret",
                    "order": 1,
                    "endpoint_type": "responses",
                    "models": ["gpt-a", "gpt-b"],
                }
            ]
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            self.state = app_module.RouterState.create_sync(str(self.config_path))

        output = "\n".join(logs.output)
        self.assertIn("Loaded config: 1 providers, 2 model routes", output)
        self.assertIn(
            "Provider responses endpoint=responses order=1 models=explicit "
            "count=2 ids=gpt-a,gpt-b",
            output,
        )
        self.assertNotIn("sk-secret", output)

    async def test_string_mapping_suffix_exposes_alias_and_forwards_upstream_model(
        self,
    ) -> None:
        data: dict[str, Any] = {
            "providers": [
                {
                    "name": "responses",
                    "api_base": "https://responses.example",
                    "api_key": "sk-responses",
                    "order": 1,
                    "endpoint_type": "responses",
                    "models": ["gpt-5.4-mini:gpt-5.5"],
                }
            ]
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.state = app_module.RouterState.create_sync(str(self.config_path))

        candidates = await self.state.get_candidate_providers(
            "gpt-5.5",
            "/responses",
            sticky_key=None,
        )
        models = await self.state.list_models()

        self.assertEqual([model["id"] for model in models], ["gpt-5.5"])
        self.assertEqual(candidates[0].configured_model, "gpt-5.4-mini")
        self.assertEqual(candidates[0].upstream_model, "gpt-5.4-mini")
        with self.assertRaises(KeyError):
            await self.state.get_candidate_providers(
                "gpt-5.4-mini",
                "/responses",
                sticky_key=None,
            )

    async def test_auto_models_are_loaded_from_provider_models_endpoint(self) -> None:
        data: dict[str, Any] = {
            "providers": [
                {
                    "name": "auto",
                    "api_base": "https://auto.example",
                    "api_key": "sk-auto",
                    "order": 1,
                    "endpoint_type": "openai-compatible",
                    "models": "auto",
                }
            ]
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        captured: list[dict[str, Any]] = []

        def fake_get(url: str, **kwargs: Any) -> httpx.Response:
            captured.append({"url": url, **kwargs})
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "gpt-auto-a", "object": "model"},
                        {"id": "gpt-auto-b", "object": "model"},
                    ],
                },
            )

        original_client = install_fake_async_get(fake_get)
        try:
            self.state = app_module.RouterState.create_sync(str(self.config_path))
        finally:
            app_module.httpx.AsyncClient = original_client

        models = await self.state.list_models()
        candidates = await self.state.get_candidate_providers(
            "gpt-auto-a",
            "/chat/completions",
            sticky_key=None,
        )

        self.assertEqual(captured[0]["url"], "https://auto.example/v1/models")
        self.assertEqual(
            captured[0]["headers"]["Authorization"],
            "Bearer sk-auto",
        )
        self.assertEqual([model["id"] for model in models], ["gpt-auto-a", "gpt-auto-b"])
        self.assertEqual(candidates[0].upstream_model, "gpt-auto-a")

    async def test_auto_models_are_filtered_by_endpoint_type_when_some_match(self) -> None:
        data: dict[str, Any] = {
            "providers": [
                {
                    "name": "songsong",
                    "api_base": "https://ai.songsongcard.shop",
                    "api_key": "sk-provider",
                    "order": 1,
                    "endpoint_type": "anthropic",
                    "models": "auto",
                }
            ]
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

        def fake_get(url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "claude-opus-4-7",
                            "object": "model",
                            "supported_endpoint_types": ["anthropic"],
                        },
                        {
                            "id": "gpt-5.5",
                            "object": "model",
                            "supported_endpoint_types": ["openai"],
                        },
                    ],
                },
            )

        original_client = install_fake_async_get(fake_get)
        try:
            self.state = app_module.RouterState.create_sync(str(self.config_path))
        finally:
            app_module.httpx.AsyncClient = original_client

        models = await self.state.list_models()
        message_candidates = await self.state.get_candidate_providers(
            "claude-opus-4-7",
            "/messages",
            sticky_key=None,
        )

        self.assertEqual([model["id"] for model in models], ["claude-opus-4-7"])
        self.assertEqual(message_candidates[0].upstream_model, "claude-opus-4-7")
        with self.assertRaises(KeyError):
            await self.state.get_candidate_providers(
                "claude-opus-4-7",
                "/chat/completions",
                sticky_key=None,
            )

    async def test_auto_models_keep_discovered_ids_when_metadata_all_conflicts(
        self,
    ) -> None:
        data: dict[str, Any] = {
            "providers": [
                {
                    "name": "responses",
                    "api_base": "https://responses.example",
                    "api_key": "sk-provider",
                    "order": 1,
                    "endpoint_type": "responses",
                    "models": "auto",
                }
            ]
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

        def fake_get(url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "gpt-5.4",
                            "object": "model",
                            "supported_endpoint_types": ["openai"],
                        },
                        {
                            "id": "gpt-5.4-mini",
                            "object": "model",
                            "supported_endpoint_types": ["openai"],
                        },
                    ],
                },
            )

        original_client = install_fake_async_get(fake_get)
        try:
            with self.assertLogs("uvicorn.error", level="WARNING") as logs:
                self.state = app_module.RouterState.create_sync(str(self.config_path))
        finally:
            app_module.httpx.AsyncClient = original_client

        models = await self.state.list_models()
        candidates = await self.state.get_candidate_providers(
            "gpt-5.4-mini",
            "/responses",
            sticky_key=None,
        )
        snapshot = await self.state.snapshot()

        self.assertEqual([model["id"] for model in models], ["gpt-5.4", "gpt-5.4-mini"])
        self.assertEqual(candidates[0].upstream_model, "gpt-5.4-mini")
        self.assertIn("keeping discovered IDs", "\n".join(logs.output))
        self.assertEqual(snapshot["providers"][0]["model_source"], "auto")
        self.assertEqual(
            snapshot["providers"][0]["discovered_model_ids"],
            ["gpt-5.4", "gpt-5.4-mini"],
        )
        self.assertEqual(
            snapshot["providers"][0]["filtered_model_ids"],
            ["gpt-5.4", "gpt-5.4-mini"],
        )
        self.assertIn(
            "keeping discovered IDs",
            snapshot["providers"][0]["discovery_warnings"][0],
        )

    async def test_anthropic_role_suffix_maps_claude_alias_to_provider_model(self) -> None:
        data: dict[str, Any] = {
            "providers": [
                {
                    "name": "fast",
                    "api_base": "https://fast.example/v1",
                    "api_key": "sk-fast",
                    "order": 1,
                    "models": ["deepseek-v4-flash:haiku", "deepseek-v4-pro:opus"],
                },
                {
                    "name": "backup",
                    "api_base": "https://backup.example/v1",
                    "api_key": "sk-backup",
                    "order": 2,
                    "models": [{"model": "kimi-k2", "anthropic_role": "opus"}],
                },
                {
                    "name": "direct",
                    "api_base": "https://direct.example/v1",
                    "api_key": "sk-direct",
                    "order": 3,
                    "models": ["deepseek-v4-pro"],
                },
                {
                    "name": "exact-haiku",
                    "api_base": "https://exact.example/v1",
                    "api_key": "sk-exact",
                    "order": 4,
                    "models": ["qwen-haiku:claude-haiku-4-5-20251001[1M]"],
                },
            ]
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.state = app_module.RouterState.create_sync(str(self.config_path))

        opus_candidates = await self.state.get_candidate_providers(
            "claude-opus-4-7[1M]",
            "/messages",
            sticky_key=None,
        )
        haiku_candidates = await self.state.get_candidate_providers(
            "claude-haiku-4-5-20251001",
            "/messages",
            sticky_key=None,
        )
        direct_candidates = await self.state.get_candidate_providers(
            "deepseek-v4-pro",
            "/messages",
            sticky_key=None,
        )
        models = await self.state.list_models()

        self.assertEqual(
            [provider.upstream_model for provider in opus_candidates],
            ["deepseek-v4-pro", "kimi-k2"],
        )
        self.assertEqual(
            [provider.upstream_model for provider in haiku_candidates],
            ["deepseek-v4-flash", "qwen-haiku"],
        )
        self.assertEqual([provider.upstream_model for provider in direct_candidates], [
            "deepseek-v4-pro",
        ])
        self.assertEqual(opus_candidates[0].anthropic_role, "opus")
        self.assertIn("claude-haiku-4-5-20251001", [model["id"] for model in models])
        self.assertIn("claude-opus-4-7", [model["id"] for model in models])

    async def test_anthropic_role_suffix_matches_requested_model_by_role_name(self) -> None:
        data: dict[str, Any] = {
            "providers": [
                {
                    "name": "future-opus",
                    "api_base": "https://future.example/v1",
                    "api_key": "sk-future",
                    "order": 1,
                    "endpoint_type": "anthropic",
                    "models": ["model-a:opus"],
                }
            ]
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.state = app_module.RouterState.create_sync(str(self.config_path))

        candidates = await self.state.get_candidate_providers(
            "claude-opus-4-99",
            "/messages",
            sticky_key=None,
        )

        self.assertEqual(candidates[0].upstream_model, "model-a")

    async def test_upstream_openai_style_urls_add_v1_when_provider_base_omits_it(
        self,
    ) -> None:
        responses_provider = app_module.Provider(
            provider_name="responses",
            model_name="gpt-5.5",
            configured_model="gpt-5.5",
            upstream_model="gpt-5.5",
            anthropic_role=None,
            endpoint_type="responses",
            api_base="https://responses.example",
            api_url=None,
            models_url=None,
            api_key="sk-responses",
            order=1,
            timeout=None,
            extra_headers={},
        )
        chat_provider = app_module.Provider(
            provider_name="chat",
            model_name="deepseek-v4-flash",
            configured_model="deepseek-v4-flash",
            upstream_model="deepseek-v4-flash",
            anthropic_role=None,
            endpoint_type="openai-compatible",
            api_base="https://chat.example",
            api_url=None,
            models_url=None,
            api_key="sk-chat",
            order=1,
            timeout=None,
            extra_headers={},
        )
        anthropic_provider = app_module.Provider(
            provider_name="anthropic",
            model_name="claude-opus-4-7",
            configured_model="deepseek-v4-pro",
            upstream_model="deepseek-v4-pro",
            anthropic_role="opus",
            endpoint_type="anthropic",
            api_base="https://anthropic.example/anthropic",
            api_url=None,
            models_url=None,
            api_key="sk-anthropic",
            order=1,
            timeout=None,
            extra_headers={},
        )
        anthropic_host_provider = app_module.Provider(
            provider_name="anthropic-host",
            model_name="claude-opus-4-7",
            configured_model="deepseek-v4-pro",
            upstream_model="deepseek-v4-pro",
            anthropic_role="opus",
            endpoint_type="anthropic",
            api_base="https://anthropic-host.example",
            api_url=None,
            models_url=None,
            api_key="sk-anthropic",
            order=1,
            timeout=None,
            extra_headers={},
        )

        self.assertEqual(
            app_module.build_upstream_url(responses_provider, "/responses"),
            "https://responses.example/v1/responses",
        )
        self.assertEqual(
            app_module.build_upstream_url(chat_provider, "/chat/completions"),
            "https://chat.example/v1/chat/completions",
        )
        self.assertEqual(
            app_module.build_upstream_url(anthropic_provider, "/messages"),
            "https://anthropic.example/anthropic/v1/messages",
        )
        self.assertEqual(
            app_module.build_upstream_url(anthropic_host_provider, "/messages"),
            "https://anthropic-host.example/v1/messages",
        )

    async def test_api_url_exactly_overrides_inferred_request_url(self) -> None:
        provider = app_module.Provider(
            provider_name="custom",
            model_name="gpt-5.5",
            configured_model="gpt-5.4-mini",
            upstream_model="gpt-5.4-mini",
            anthropic_role=None,
            endpoint_type="responses",
            api_base="https://base.example",
            api_url="https://custom.example/respond",
            models_url=None,
            api_key="sk-custom",
            order=1,
            timeout=None,
            extra_headers={},
        )

        self.assertEqual(
            app_module.build_upstream_url(provider, "/responses"),
            "https://custom.example/respond",
        )

    async def test_api_url_preserves_exact_configured_url(self) -> None:
        provider = app_module.Provider(
            provider_name="custom",
            model_name="gpt-5.5",
            configured_model="gpt-5.4-mini",
            upstream_model="gpt-5.4-mini",
            anthropic_role=None,
            endpoint_type="responses",
            api_base=None,
            api_url="https://custom.example/respond/",
            models_url=None,
            api_key="sk-custom",
            order=1,
            timeout=None,
            extra_headers={},
        )

        self.assertEqual(
            app_module.build_upstream_url(provider, "/responses"),
            "https://custom.example/respond/",
        )

    async def test_auto_models_with_api_base_and_api_url_discovers_from_api_base(
        self,
    ) -> None:
        data: dict[str, Any] = {
            "providers": [
                {
                    "name": "custom-request",
                    "api_base": "https://models.example",
                    "api_url": "https://custom.example/respond",
                    "api_key": "sk-custom",
                    "order": 1,
                    "endpoint_type": "responses",
                    "models": "auto",
                }
            ]
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        captured: list[dict[str, Any]] = []

        def fake_get(url: str, **kwargs: Any) -> httpx.Response:
            captured.append({"url": url, **kwargs})
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "gpt-auto", "object": "model"}]},
            )

        original_client = install_fake_async_get(fake_get)
        try:
            self.state = app_module.RouterState.create_sync(str(self.config_path))
        finally:
            app_module.httpx.AsyncClient = original_client

        candidates = await self.state.get_candidate_providers(
            "gpt-auto",
            "/responses",
            sticky_key=None,
        )

        self.assertEqual(captured[0]["url"], "https://models.example/v1/models")
        self.assertEqual(
            app_module.build_upstream_url(candidates[0], "/responses"),
            "https://custom.example/respond",
        )

    async def test_auto_models_with_api_url_and_models_url_uses_models_url(self) -> None:
        data: dict[str, Any] = {
            "providers": [
                {
                    "name": "custom-only",
                    "api_url": "https://custom.example/respond",
                    "models_url": "https://custom.example/list-models/",
                    "api_key": "sk-custom",
                    "order": 1,
                    "endpoint_type": "responses",
                    "models": "auto",
                }
            ]
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        captured: list[dict[str, Any]] = []

        def fake_get(url: str, **kwargs: Any) -> httpx.Response:
            captured.append({"url": url, **kwargs})
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "gpt-auto", "object": "model"}]},
            )

        original_client = install_fake_async_get(fake_get)
        try:
            self.state = app_module.RouterState.create_sync(str(self.config_path))
        finally:
            app_module.httpx.AsyncClient = original_client

        self.assertEqual(captured[0]["url"], "https://custom.example/list-models/")

    async def test_auto_models_with_api_url_only_is_rejected(self) -> None:
        self.config_path.write_text(
            yaml.safe_dump(
                {
                    "providers": [
                        {
                            "name": "custom-only",
                            "api_url": "https://custom.example/respond",
                            "api_key": "sk-custom",
                            "order": 1,
                            "endpoint_type": "responses",
                            "models": "auto",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "models auto discovery requires api_base or models_url",
        ):
            app_module.RouterState.create_sync(str(self.config_path))

    async def test_api_url_requires_endpoint_type(self) -> None:
        self.config_path.write_text(
            yaml.safe_dump(
                {
                    "providers": [
                        {
                            "name": "custom-only",
                            "api_url": "https://custom.example/respond",
                            "api_key": "sk-custom",
                            "order": 1,
                            "models": ["gpt-custom"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            r"providers\[0\]\.endpoint_type is required when api_url is set",
        ):
            app_module.RouterState.create_sync(str(self.config_path))

    async def test_anthropic_exact_model_and_role_suffix_share_candidates(self) -> None:
        data: dict[str, Any] = {
            "providers": [
                {
                    "name": "exact",
                    "api_base": "https://exact.example/v1",
                    "api_key": "sk-exact",
                    "order": 1,
                    "endpoint_type": "anthropic",
                    "models": ["claude-opus-4-7"],
                },
                {
                    "name": "mapped",
                    "api_base": "https://mapped.example/v1",
                    "api_key": "sk-mapped",
                    "order": 2,
                    "endpoint_type": "anthropic",
                    "models": ["model-a:opus"],
                },
            ]
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.state = app_module.RouterState.create_sync(str(self.config_path))

        candidates = await self.state.get_candidate_providers(
            "claude-opus-4-7",
            "/messages",
            sticky_key=None,
        )

        self.assertEqual(
            [candidate.upstream_model for candidate in candidates],
            ["claude-opus-4-7", "model-a"],
        )

    async def test_messages_endpoint_rewrites_alias_to_upstream_model(self) -> None:
        data: dict[str, Any] = {
            "app_settings": {"hot_reload": False},
            "providers": [
                {
                    "name": "anthropic-compatible",
                    "api_base": "https://anthropic.example/v1",
                    "api_key": "sk-anthropic",
                    "order": 1,
                    "models": ["deepseek-v4-pro:opus"],
                }
            ],
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.state = app_module.RouterState.create_sync(str(self.config_path))

        captured: list[dict[str, Any]] = []

        class FakeAsyncClient:
            def __init__(self, **_: Any) -> None:
                pass

            async def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, Any],
                timeout: Any,
            ) -> httpx.Response:
                captured.append(
                    {
                        "url": url,
                        "headers": headers,
                        "json": json,
                        "timeout": timeout,
                    }
                )
                return httpx.Response(
                    200,
                    json={"id": "msg-test", "type": "message"},
                    headers={"content-type": "application/json"},
                )

            async def aclose(self) -> None:
                pass

        original_state = app_module.router_state
        original_client = app_module.httpx.AsyncClient
        app_module.router_state = self.state
        app_module.httpx.AsyncClient = FakeAsyncClient
        try:
            client = TestClient(app_module.app)
            response = client.post(
                "/v1/messages",
                headers={"anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-opus-4-7[1M]",
                    "max_tokens": 256,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        finally:
            app_module.router_state = original_state
            app_module.httpx.AsyncClient = original_client

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured[0]["url"], "https://anthropic.example/v1/messages")
        self.assertEqual(captured[0]["json"]["model"], "deepseek-v4-pro")
        self.assertEqual(captured[0]["headers"]["x-api-key"], "sk-anthropic")
        self.assertEqual(captured[0]["headers"]["anthropic-version"], "2023-06-01")
        self.assertNotIn("Authorization", captured[0]["headers"])

    async def test_messages_count_tokens_endpoint_rewrites_alias_to_upstream_model(
        self,
    ) -> None:
        data: dict[str, Any] = {
            "app_settings": {"hot_reload": False},
            "providers": [
                {
                    "name": "anthropic-compatible",
                    "api_base": "https://anthropic.example/v1",
                    "api_key": "sk-anthropic",
                    "order": 1,
                    "endpoint_type": "anthropic",
                    "models": ["deepseek-v4-pro:opus"],
                }
            ],
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.state = app_module.RouterState.create_sync(str(self.config_path))

        captured: list[dict[str, Any]] = []

        class FakeAsyncClient:
            def __init__(self, **_: Any) -> None:
                pass

            async def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, Any],
                timeout: Any,
            ) -> httpx.Response:
                captured.append(
                    {
                        "url": url,
                        "headers": headers,
                        "json": json,
                        "timeout": timeout,
                    }
                )
                return httpx.Response(
                    200,
                    json={"input_tokens": 42},
                    headers={"content-type": "application/json"},
                )

            async def aclose(self) -> None:
                pass

        original_state = app_module.router_state
        original_client = app_module.httpx.AsyncClient
        app_module.router_state = self.state
        app_module.httpx.AsyncClient = FakeAsyncClient
        try:
            client = TestClient(app_module.app)
            response = client.post(
                "/v1/messages/count_tokens?beta=true",
                headers={"anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-opus-4-99[1M]",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        finally:
            app_module.router_state = original_state
            app_module.httpx.AsyncClient = original_client

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"input_tokens": 42})
        self.assertEqual(
            captured[0]["url"],
            "https://anthropic.example/v1/messages/count_tokens",
        )
        self.assertEqual(captured[0]["json"]["model"], "deepseek-v4-pro")
        self.assertEqual(captured[0]["headers"]["x-api-key"], "sk-anthropic")
        self.assertEqual(captured[0]["headers"]["anthropic-version"], "2023-06-01")
        self.assertNotIn("Authorization", captured[0]["headers"])

    async def test_invalid_success_response_falls_back_to_next_provider(self) -> None:
        data: dict[str, Any] = {
            "app_settings": {"hot_reload": False},
            "providers": [
                {
                    "name": "bad-html",
                    "api_base": "https://bad.example",
                    "api_key": "sk-bad",
                    "order": 1,
                    "endpoint_type": "anthropic",
                    "models": ["bad-model:opus"],
                },
                {
                    "name": "good-json",
                    "api_base": "https://good.example/v1",
                    "api_key": "sk-good",
                    "order": 2,
                    "endpoint_type": "anthropic",
                    "models": ["good-model:opus"],
                },
            ],
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.state = app_module.RouterState.create_sync(str(self.config_path))

        captured: list[dict[str, Any]] = []

        class FakeAsyncClient:
            def __init__(self, **_: Any) -> None:
                pass

            async def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, Any],
                timeout: Any,
            ) -> httpx.Response:
                captured.append({"url": url, "json": json})
                if "bad.example" in url:
                    return httpx.Response(
                        200,
                        text="<!doctype html><title>Gateway</title>",
                        headers={"content-type": "text/html; charset=utf-8"},
                    )
                return httpx.Response(
                    200,
                    json={"id": "msg-good", "type": "message"},
                    headers={"content-type": "application/json"},
                )

            async def aclose(self) -> None:
                pass

        original_state = app_module.router_state
        original_client = app_module.httpx.AsyncClient
        app_module.router_state = self.state
        app_module.httpx.AsyncClient = FakeAsyncClient
        try:
            client = TestClient(app_module.app)
            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude-opus-4-7",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        finally:
            app_module.router_state = original_state
            app_module.httpx.AsyncClient = original_client

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "msg-good")
        self.assertEqual(
            [attempt["json"]["model"] for attempt in captured],
            ["bad-model", "good-model"],
        )

    async def test_responses_stream_error_event_falls_back_before_streaming(self) -> None:
        data: dict[str, Any] = {
            "app_settings": {"hot_reload": False},
            "providers": [
                {
                    "name": "stream-error",
                    "api_base": "https://stream-error.example",
                    "api_key": "sk-bad",
                    "order": 1,
                    "endpoint_type": "responses",
                    "models": ["gpt-test"],
                },
                {
                    "name": "stream-good",
                    "api_base": "https://stream-good.example",
                    "api_key": "sk-good",
                    "order": 2,
                    "endpoint_type": "responses",
                    "models": ["gpt-test"],
                },
            ],
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.state = app_module.RouterState.create_sync(str(self.config_path))
        captured: list[dict[str, Any]] = []

        class FakeStreamResponse:
            def __init__(self, *, headers: dict[str, str], chunks: list[bytes]) -> None:
                self.status_code = 200
                self.headers = headers
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

        class FakeAsyncClient:
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
                return {
                    "method": method,
                    "url": url,
                    "headers": headers,
                    "json": json,
                    "timeout": timeout,
                }

            async def send(
                self,
                request: dict[str, Any],
                *,
                stream: bool,
            ) -> FakeStreamResponse:
                captured.append(request)
                if "stream-error.example" in request["url"]:
                    return FakeStreamResponse(
                        headers={"content-type": "text/event-stream"},
                        chunks=[
                            (
                                "event:\n"
                                'data: {"error":{"type":"rate_limit_error",'
                                '"message":"Concurrency limit exceeded"}}\n\n'
                            ).encode(),
                        ],
                    )
                return FakeStreamResponse(
                    headers={"content-type": "text/event-stream"},
                    chunks=[
                        (
                            "event: response.created\n"
                            'data: {"type":"response.created","response":'
                            '{"id":"resp-test","status":"in_progress"}}\n\n'
                        ).encode(),
                        (
                            "event: response.completed\n"
                            'data: {"type":"response.completed","response":'
                            '{"id":"resp-test","status":"completed"}}\n\n'
                        ).encode(),
                    ],
                )

            async def aclose(self) -> None:
                pass

        original_state = app_module.router_state
        original_client = app_module.httpx.AsyncClient
        app_module.router_state = self.state
        app_module.httpx.AsyncClient = FakeAsyncClient
        try:
            client = TestClient(app_module.app)
            with client.stream(
                "POST",
                "/v1/responses",
                json={"model": "gpt-test", "input": "hello", "stream": True},
            ) as response:
                body = response.read().decode()
        finally:
            app_module.router_state = original_state
            app_module.httpx.AsyncClient = original_client

        snapshot = await self.state.snapshot()

        self.assertEqual(response.status_code, 200)
        self.assertIn("response.completed", body)
        self.assertEqual(
            [attempt["json"]["model"] for attempt in captured],
            ["gpt-test", "gpt-test"],
        )
        self.assertEqual(
            response.headers["x-fallback-provider-id"],
            "stream-good",
        )
        self.assertIn(
            "Responses stream error event",
            {
                provider["provider_name"]: provider["last_error"]["/responses"]
                for provider in snapshot["providers"]
            }["stream-error"],
        )

    async def test_responses_stream_start_timeout_falls_back_before_streaming(
        self,
    ) -> None:
        data: dict[str, Any] = {
            "app_settings": {"hot_reload": False, "stream_start_timeout": 0.01},
            "providers": [
                {
                    "name": "stream-stalled",
                    "api_base": "https://stream-stalled.example",
                    "api_key": "sk-stalled",
                    "order": 1,
                    "endpoint_type": "responses",
                    "models": ["gpt-test"],
                },
                {
                    "name": "stream-good",
                    "api_base": "https://stream-good.example",
                    "api_key": "sk-good",
                    "order": 2,
                    "endpoint_type": "responses",
                    "models": ["gpt-test"],
                },
            ],
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.state = app_module.RouterState.create_sync(str(self.config_path))
        captured: list[dict[str, Any]] = []

        class FakeStreamResponse:
            def __init__(
                self,
                *,
                headers: dict[str, str],
                chunks: list[bytes],
                first_chunk_delay: float = 0,
            ) -> None:
                self.status_code = 200
                self.headers = headers
                self._chunks = chunks
                self._first_chunk_delay = first_chunk_delay
                self.is_closed = False
                self.is_stream_consumed = False

            async def aiter_bytes(self) -> AsyncIterator[bytes]:
                if self.is_stream_consumed:
                    raise httpx.StreamConsumed()
                self.is_stream_consumed = True
                if self._first_chunk_delay:
                    await asyncio.sleep(self._first_chunk_delay)
                for chunk in self._chunks:
                    yield chunk

            async def aread(self) -> bytes:
                return b"".join(self._chunks)

            async def aclose(self) -> None:
                self.is_closed = True

        class FakeAsyncClient:
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
                return {
                    "method": method,
                    "url": url,
                    "headers": headers,
                    "json": json,
                    "timeout": timeout,
                }

            async def send(
                self,
                request: dict[str, Any],
                *,
                stream: bool,
            ) -> FakeStreamResponse:
                captured.append(request)
                if "stream-stalled.example" in request["url"]:
                    return FakeStreamResponse(
                        headers={"content-type": "text/event-stream"},
                        chunks=[],
                        first_chunk_delay=10,
                    )
                return FakeStreamResponse(
                    headers={"content-type": "text/event-stream"},
                    chunks=[
                        (
                            "event: response.completed\n"
                            'data: {"type":"response.completed","response":'
                            '{"id":"resp-test","status":"completed"}}\n\n'
                        ).encode(),
                    ],
                )

            async def aclose(self) -> None:
                pass

        original_state = app_module.router_state
        original_client = app_module.httpx.AsyncClient
        app_module.router_state = self.state
        app_module.httpx.AsyncClient = FakeAsyncClient
        try:
            client = TestClient(app_module.app)
            with client.stream(
                "POST",
                "/v1/responses",
                json={"model": "gpt-test", "input": "hello", "stream": True},
            ) as response:
                body = response.read().decode()
        finally:
            app_module.router_state = original_state
            app_module.httpx.AsyncClient = original_client

        snapshot = await self.state.snapshot()

        self.assertEqual(response.status_code, 200)
        self.assertIn("response.completed", body)
        self.assertEqual(
            [attempt["json"]["model"] for attempt in captured],
            ["gpt-test", "gpt-test"],
        )
        self.assertEqual(response.headers["x-fallback-provider-id"], "stream-good")
        self.assertIn(
            "Responses stream did not start within",
            {
                provider["provider_name"]: provider["last_error"]["/responses"]
                for provider in snapshot["providers"]
            }["stream-stalled"],
        )

    async def test_responses_stream_send_timeout_falls_back_before_streaming(
        self,
    ) -> None:
        data: dict[str, Any] = {
            "app_settings": {"hot_reload": False, "stream_start_timeout": 0.01},
            "providers": [
                {
                    "name": "send-stalled",
                    "api_base": "https://send-stalled.example",
                    "api_key": "sk-stalled",
                    "order": 1,
                    "endpoint_type": "responses",
                    "models": ["gpt-test"],
                },
                {
                    "name": "stream-good",
                    "api_base": "https://stream-good.example",
                    "api_key": "sk-good",
                    "order": 2,
                    "endpoint_type": "responses",
                    "models": ["gpt-test"],
                },
            ],
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.state = app_module.RouterState.create_sync(str(self.config_path))
        captured: list[dict[str, Any]] = []

        class FakeStreamResponse:
            status_code = 200
            headers = {"content-type": "text/event-stream"}
            is_closed = False
            is_stream_consumed = False

            async def aiter_bytes(self) -> AsyncIterator[bytes]:
                if self.is_stream_consumed:
                    raise httpx.StreamConsumed()
                self.is_stream_consumed = True
                yield (
                    "event: response.completed\n"
                    'data: {"type":"response.completed","response":'
                    '{"id":"resp-test","status":"completed"}}\n\n'
                ).encode()

            async def aread(self) -> bytes:
                return b""

            async def aclose(self) -> None:
                self.is_closed = True

        class FakeAsyncClient:
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
                return {
                    "method": method,
                    "url": url,
                    "headers": headers,
                    "json": json,
                    "timeout": timeout,
                }

            async def send(
                self,
                request: dict[str, Any],
                *,
                stream: bool,
            ) -> FakeStreamResponse:
                captured.append(request)
                if "send-stalled.example" in request["url"]:
                    await asyncio.sleep(10)
                return FakeStreamResponse()

            async def aclose(self) -> None:
                pass

        original_state = app_module.router_state
        original_client = app_module.httpx.AsyncClient
        app_module.router_state = self.state
        app_module.httpx.AsyncClient = FakeAsyncClient
        try:
            client = TestClient(app_module.app)
            with client.stream(
                "POST",
                "/v1/responses",
                json={"model": "gpt-test", "input": "hello", "stream": True},
            ) as response:
                body = response.read().decode()
        finally:
            app_module.router_state = original_state
            app_module.httpx.AsyncClient = original_client

        snapshot = await self.state.snapshot()

        self.assertEqual(response.status_code, 200)
        self.assertIn("response.completed", body)
        self.assertEqual(
            [attempt["json"]["model"] for attempt in captured],
            ["gpt-test", "gpt-test"],
        )
        self.assertEqual(response.headers["x-fallback-provider-id"], "stream-good")
        self.assertIn(
            "TimeoutError",
            {
                provider["provider_name"]: provider["last_error"]["/responses"]
                for provider in snapshot["providers"]
            }["send-stalled"],
        )

    async def test_root_models_endpoint_matches_v1_models_endpoint(self) -> None:
        data: dict[str, Any] = {
            "app_settings": {"hot_reload": False},
            "providers": [
                {
                    "name": "responses",
                    "api_base": "https://responses.example",
                    "api_key": "sk-responses",
                    "order": 1,
                    "endpoint_type": "responses",
                    "models": ["gpt-5.4-mini:gpt-5.5"],
                }
            ],
        }
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.state = app_module.RouterState.create_sync(str(self.config_path))

        original_state = app_module.router_state
        app_module.router_state = self.state
        try:
            client = TestClient(app_module.app)
            root_response = client.get("/models")
            v1_response = client.get("/v1/models")
        finally:
            app_module.router_state = original_state

        self.assertEqual(root_response.status_code, 200)
        self.assertEqual(root_response.json(), v1_response.json())
        self.assertEqual(root_response.json()["data"][0]["id"], "gpt-5.5")

    async def test_root_head_returns_ok(self) -> None:
        client = TestClient(app_module.app)
        response = client.head("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"")

    async def test_legacy_model_list_is_rejected(self) -> None:
        self.config_path.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": "gpt-test",
                            "litellm_params": {
                                "model": "openai/gpt-test",
                                "api_base": "https://old.example/v1",
                                "api_key": "sk-test",
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "model_list is no longer supported"):
            app_module.RouterState.create_sync(str(self.config_path))


if __name__ == "__main__":
    unittest.main()
