from __future__ import annotations

import asyncio
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml


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


class HotReloadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "config.yaml"
        write_config(self.config_path, api_base="https://one.example/v1")
        self.state = app_module.RouterState(str(self.config_path))

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
        self.state = app_module.RouterState(str(self.config_path))
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
        self.state = app_module.RouterState(str(self.config_path))

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
            app_module.RouterState(str(self.config_path))


if __name__ == "__main__":
    unittest.main()
