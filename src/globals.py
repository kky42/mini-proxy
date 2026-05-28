from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .routing import RouterState

CONFIG_PATH = os.environ.get("MINI_FALLBACK_PROXY_CONFIG")

# Set by endpoints.py at startup. Modules that need the live reference
# should import this module and access globals.router_state as an
# attribute (NOT via `from src.globals import router_state`).
router_state: RouterState | None = None
