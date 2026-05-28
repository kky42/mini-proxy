from __future__ import annotations

import logging
import re

logger = logging.getLogger("uvicorn.error")
DEFAULT_TIMEOUT = 60.0
DEFAULT_STICKY_TTL_SECONDS = 1800
DEFAULT_HOT_RELOAD_INTERVAL_SECONDS = 1.0
DEFAULT_STREAM_START_TIMEOUT = 30.0
DEFAULT_ALLOWED_RETRIES = 0
DEFAULT_RETRY_BACKOFF_SECONDS = 0.25
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_ROLE_MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}
ANTHROPIC_ONE_M_SUFFIX_RE = re.compile(r"\[1m\]\s*$", re.IGNORECASE)
ENDPOINT_TYPES = {"responses", "openai-compatible", "anthropic"}
ANTHROPIC_ENDPOINTS = {"/messages", "/messages/count_tokens"}
