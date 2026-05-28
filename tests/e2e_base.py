"""Base class for e2e tests. Sets up a temp config + fresh RouterState per test."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.e2e_helpers import build_config, load_app_module, write_temp_config


class E2EBase(unittest.IsolatedAsyncioTestCase):
    """Each test class gets one config; each test method gets a fresh RouterState."""

    config: dict | None = None  # override in subclass
    tmpdir: tempfile.TemporaryDirectory | None = None
    config_path: Path | None = None
    app_module = None

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cfg = cls.config if cls.config is not None else build_config()
        cls.config_path = write_temp_config(cfg, cls.tmpdir.name)
        cls.app_module = load_app_module(cls.config_path)

    @classmethod
    def tearDownClass(cls):
        if cls.tmpdir is not None:
            cls.tmpdir.cleanup()

    def setUp(self):
        self.state = self.app_module.RouterState.create_sync(str(self.config_path))

    def swap_state_and_client(self, state, FakeClient):
        """Install a fake httpx client and RouterState, return (original_client, original_state)."""
        original_client = self.app_module.httpx.AsyncClient
        original_state = self.app_module.router_state
        self.app_module.httpx.AsyncClient = FakeClient
        self.app_module.router_state = state
        return original_client, original_state

    def restore_state_and_client(self, original_client, original_state):
        self.app_module.httpx.AsyncClient = original_client
        self.app_module.router_state = original_state
