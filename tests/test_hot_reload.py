from __future__ import annotations

import asyncio
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml


def write_config(path: Path, *, api_base: str, order: int = 1) -> None:
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
        "model_list": [
            {
                "model_name": "gpt-test",
                "litellm_params": {
                    "model": "openai/gpt-test",
                    "api_base": api_base,
                    "api_key": "sk-test",
                    "order": order,
                },
            }
        ],
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
        self.config_path.write_text("model_list: not-a-list\n", encoding="utf-8")
        touch_newer(self.config_path, previous_mtime)

        result = await self.state.reload_if_changed()
        models = await self.state.list_models()
        snapshot = await self.state.snapshot()

        self.assertEqual(result.status, "rejected")
        self.assertFalse(result.reloaded)
        self.assertEqual(models[0]["providers"][0]["api_base"], "https://one.example/v1")
        self.assertIn("model_list must be a list", snapshot["hot_reload"]["last_error"])

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


if __name__ == "__main__":
    unittest.main()
