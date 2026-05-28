from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class FailureClass(str, Enum):
    AVAILABILITY = "availability"
    AUTH_OR_BALANCE = "auth_or_balance"
    CAPABILITY_MISMATCH = "capability_mismatch"
    REQUEST_INVALID = "request_invalid"
    MIDSTREAM_FAILURE = "midstream_failure"


@dataclass(frozen=True)
class FailureDecision:
    failure_class: FailureClass
    should_fallback: bool
    count_failure: bool
    cooldown_multiplier: float = 1.0


@dataclass(frozen=True)
class StickyBinding:
    provider_id: str
    expires_at: float


@dataclass(frozen=True)
class Provider:
    provider_name: str
    model_name: str
    configured_model: str
    upstream_model: str
    anthropic_role: str | None
    endpoint_type: str | None
    api_base: str | None
    api_url: str | None
    models_url: str | None
    api_key: str
    order: int
    timeout: float | None
    extra_headers: dict[str, str]
    provider_index: int = -1
    model_source: str = "explicit"
    discovered_model_ids: tuple[str, ...] = ()
    filtered_model_ids: tuple[str, ...] = ()
    discovery_warnings: tuple[str, ...] = ()

    @property
    def provider_id(self) -> str:
        return self.provider_name

    @property
    def sort_url(self) -> str:
        return self.api_url or self.api_base or ""


@dataclass(frozen=True)
class RouterConfig:
    host: str
    port: int
    log_level: str
    default_timeout: float
    stream_start_timeout: float
    sticky_ttl_seconds: int
    normalize_upstream_model: bool
    hot_reload: bool
    hot_reload_interval_seconds: float
    allowed_fails: int
    cooldown_time: int
    allowed_retries: int
    retry_backoff_seconds: float
    providers_by_model: dict[str, list[Provider]]


@dataclass(frozen=True)
class AutoModelDiscovery:
    models: list[str]
    discovered_models: tuple[str, ...]
    filtered_models: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReloadResult:
    status: str
    reloaded: bool
    error: str | None = None
