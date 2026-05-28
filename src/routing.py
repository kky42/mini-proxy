from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from typing import Any

import httpx
import yaml

from .constants import (
    ANTHROPIC_ENDPOINTS,
    ANTHROPIC_ROLE_MODEL_ALIASES,
    ANTHROPIC_VERSION,
    DEFAULT_HOT_RELOAD_INTERVAL_SECONDS,
    DEFAULT_STICKY_TTL_SECONDS,
    DEFAULT_STREAM_START_TIMEOUT,
    DEFAULT_TIMEOUT,
    logger,
)
from .types import (
    AutoModelDiscovery,
    FailureDecision,
    Provider,
    ReloadResult,
    RouterConfig,
    StickyBinding,
)
from .utils import (
    _append_v1_endpoint,
    _coerce_anthropic_role,
    _coerce_bool,
    _coerce_endpoint_type,
    _coerce_headers,
    _coerce_optional_api_base,
    _coerce_optional_timeout,
    _coerce_optional_url,
    _coerce_positive_float,
    _endpoint_type_supports_endpoint,
    _infer_anthropic_role_from_model_name,
    _infer_models_url,
    _normalize_requested_model,
    _normalize_upstream_model,
    _split_model_mapping_suffix,
    _split_model_role_suffix,
    build_failure_key,
    build_upstream_url,
)


# ---------------------------------------------------------------------------
# Standalone helpers used by RouterState
# ---------------------------------------------------------------------------


def _format_model_ids(model_ids: list[str] | tuple[str, ...]) -> str:
    return ",".join(model_ids) if model_ids else "(none)"


def provider_debug_dict(provider: Provider) -> dict[str, Any]:
    return {
        "provider_name": provider.provider_name,
        "order": provider.order,
        "api_base": provider.api_base,
        "api_url": provider.api_url,
        "models_url": provider.models_url,
        "configured_model": provider.configured_model,
        "upstream_model": provider.upstream_model,
        "anthropic_role": provider.anthropic_role,
        "endpoint_type": provider.endpoint_type,
        "model_source": provider.model_source,
        "discovered_model_ids": list(provider.discovered_model_ids),
        "filtered_model_ids": list(provider.filtered_model_ids),
        "discovery_warnings": list(provider.discovery_warnings),
        "upstream_urls": {
            "/responses": build_upstream_url(provider, "/responses"),
            "/chat/completions": build_upstream_url(provider, "/chat/completions"),
            "/messages": build_upstream_url(provider, "/messages"),
            "/messages/count_tokens": build_upstream_url(
                provider,
                "/messages/count_tokens",
            ),
        },
    }


def provider_attempt_dict(provider: Provider, url: str) -> dict[str, Any]:
    return {
        "provider_id": provider.provider_id,
        "api_base": provider.api_base,
        "api_url": provider.api_url,
        "upstream_url": url,
    }


def provider_response_headers(provider: Provider, url: str) -> dict[str, str]:
    headers = {
        "x-fallback-provider-id": provider.provider_id,
        "x-fallback-upstream-url": url,
    }
    if provider.api_base is not None:
        headers["x-fallback-api-base"] = provider.api_base
    if provider.api_url is not None:
        headers["x-fallback-api-url"] = provider.api_url
    headers["x-fallback-provider-name"] = provider.provider_name
    return headers


# ---------------------------------------------------------------------------
# RouterState
# ---------------------------------------------------------------------------


class RouterState:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.cooldown_until: dict[str, float] = {}
        self.fail_counts: dict[str, int] = {}
        self.last_error: dict[str, str] = {}
        self.session_bindings: dict[str, StickyBinding] = {}
        self._lock = asyncio.Lock()
        self.allowed_fails = 0
        self.cooldown_time = 300
        self.sticky_ttl_seconds = DEFAULT_STICKY_TTL_SECONDS
        self.default_timeout = DEFAULT_TIMEOUT
        self.stream_start_timeout = DEFAULT_STREAM_START_TIMEOUT
        self.host = "127.0.0.1"
        self.port = 8099
        self.log_level = "info"
        self.normalize_upstream_model = True
        self.hot_reload = True
        self.hot_reload_interval_seconds = DEFAULT_HOT_RELOAD_INTERVAL_SECONDS
        self.providers_by_model: dict[str, list[Provider]] = {}
        self.last_reload_at: float | None = None
        self.last_config_mtime: float | None = None
        self.last_observed_config_mtime: float | None = None
        self.last_reload_error: str | None = None
        self._reload_lock = asyncio.Lock()

    async def initialize(self) -> None:
        config = await self._load_config()
        self._apply_config(config)

    @classmethod
    def create_sync(cls, config_path: str) -> "RouterState":
        instance = cls(config_path)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(instance.initialize())
            return instance

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(instance.initialize()))
            future.result()
        return instance

    # ------------------------------------------------------------------
    # Provider building
    # ------------------------------------------------------------------

    def _build_provider(
        self,
        *,
        provider_index: int,
        provider_name: str,
        model_name: str,
        configured_model: str,
        anthropic_role: str | None,
        endpoint_type: str | None,
        provider_params: dict[str, Any],
        normalize_upstream_model: bool,
        model_source: str,
        discovered_model_ids: tuple[str, ...],
        filtered_model_ids: tuple[str, ...],
        discovery_warnings: tuple[str, ...],
    ) -> Provider:
        upstream_model = configured_model
        if normalize_upstream_model:
            upstream_model = _normalize_upstream_model(configured_model)
        return Provider(
            provider_name=provider_name,
            model_name=model_name,
            configured_model=configured_model,
            upstream_model=upstream_model,
            anthropic_role=anthropic_role,
            endpoint_type=endpoint_type,
            api_base=_coerce_optional_api_base(provider_params.get("api_base")),
            api_url=_coerce_optional_url(provider_params.get("api_url")),
            models_url=_coerce_optional_url(provider_params.get("models_url")),
            api_key=provider_params["api_key"],
            order=int(provider_params.get("order", 100)),
            timeout=_coerce_optional_timeout(provider_params.get("timeout")),
            extra_headers=_coerce_headers(provider_params.get("headers")),
            provider_index=provider_index,
            model_source=model_source,
            discovered_model_ids=discovered_model_ids,
            filtered_model_ids=filtered_model_ids,
            discovery_warnings=discovery_warnings,
        )

    def _coerce_model_entry(
        self,
        model_entry: Any,
        *,
        provider_index: int,
        model_index: int,
    ) -> tuple[str, str, str | None]:
        if isinstance(model_entry, str):
            configured_model, alias_model_name, anthropic_role = _split_model_mapping_suffix(
                model_entry
            )
            if not configured_model:
                raise ValueError(
                    f"providers[{provider_index}].models[{model_index}] must not be empty"
                )
            model_name = alias_model_name if alias_model_name else configured_model
            return model_name, configured_model, anthropic_role

        if not isinstance(model_entry, dict):
            raise ValueError(
                f"providers[{provider_index}].models[{model_index}] must be a string or mapping"
            )

        configured_model = model_entry.get("model", model_entry.get("upstream_model"))
        model_name = model_entry.get("model_name", model_entry.get("name"))
        role_value = model_entry.get(
            "anthropic_role",
            model_entry.get("role", model_entry.get("map_to", model_entry.get("maps_to"))),
        )
        anthropic_role = (
            _coerce_anthropic_role(
                role_value,
                context=f"providers[{provider_index}].models[{model_index}].anthropic_role",
            )
            if role_value is not None
            else None
        )
        if configured_model is None and model_name is None:
            raise ValueError(
                f"providers[{provider_index}].models[{model_index}] must include "
                "'model_name' or 'model'"
            )
        if configured_model is None:
            configured_model = model_name
        if model_name is None:
            model_name = configured_model

        configured_model, inferred_model_name, inferred_role = _split_model_mapping_suffix(
            str(configured_model)
        )
        if inferred_role and anthropic_role is None:
            anthropic_role = inferred_role
        if (
            inferred_model_name
            and "model_name" not in model_entry
            and "name" not in model_entry
        ):
            model_name = inferred_model_name
        if anthropic_role and "model_name" not in model_entry and "name" not in model_entry:
            model_name = ANTHROPIC_ROLE_MODEL_ALIASES[anthropic_role]

        model_name = str(model_name).strip()
        configured_model = configured_model.strip()
        if not model_name or not configured_model:
            raise ValueError(
                f"providers[{provider_index}].models[{model_index}] has an empty model value"
            )
        return model_name, configured_model, anthropic_role

    # ------------------------------------------------------------------
    # Auto model discovery
    # ------------------------------------------------------------------

    def _models_url(self, provider_params: dict[str, Any]) -> str:
        configured_url = _coerce_optional_url(provider_params.get("models_url"))
        if configured_url is not None:
            return configured_url

        api_base = _coerce_optional_api_base(provider_params.get("api_base"))
        if api_base is None:
            raise ValueError("models auto discovery requires api_base or models_url")
        return _infer_models_url(api_base)

    async def _load_auto_models(
        self,
        provider_params: dict[str, Any],
        *,
        provider_index: int,
        endpoint_type: str | None,
    ) -> AutoModelDiscovery:
        headers = {"Accept": "application/json"}
        if endpoint_type == "anthropic":
            headers["x-api-key"] = str(provider_params["api_key"])
            headers["anthropic-version"] = ANTHROPIC_VERSION
        else:
            headers["Authorization"] = f"Bearer {provider_params['api_key']}"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                self._models_url(provider_params),
                headers=headers,
                timeout=_coerce_optional_timeout(provider_params.get("timeout")) or DEFAULT_TIMEOUT,
            )
        if response.status_code >= 400:
            raise ValueError(
                f"providers[{provider_index}].models auto discovery failed with "
                f"HTTP {response.status_code}: {response.text[:500]}"
            )

        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ValueError(
                f"providers[{provider_index}].models auto discovery response must include "
                "a data list"
            )

        discovered: list[str] = []
        filtered: list[str] = []
        advertised_supported_types: set[str] = set()
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                continue

            normalized_model_id = model_id.strip()
            discovered.append(normalized_model_id)
            supported_types = item.get("supported_endpoint_types")
            if endpoint_type and isinstance(supported_types, list):
                normalized_supported_types = {
                    _coerce_endpoint_type(
                        supported_type,
                        context=(
                            f"providers[{provider_index}].models auto "
                            "supported_endpoint_types"
                        ),
                    )
                    for supported_type in supported_types
                }
                advertised_supported_types.update(
                    supported_type
                    for supported_type in normalized_supported_types
                    if supported_type is not None
                )
                if endpoint_type not in normalized_supported_types:
                    continue

            filtered.append(normalized_model_id)

        if not discovered:
            raise ValueError(f"providers[{provider_index}].models auto discovered no models")

        warnings: list[str] = []
        if endpoint_type and advertised_supported_types and not filtered:
            advertised = ", ".join(sorted(advertised_supported_types))
            warnings.append(
                "discovered models advertise supported_endpoint_types="
                f"{advertised}, configured endpoint_type={endpoint_type}; "
                "keeping discovered IDs"
            )
            filtered = list(discovered)

        return AutoModelDiscovery(
            models=filtered,
            discovered_models=tuple(discovered),
            filtered_models=tuple(filtered),
            warnings=tuple(warnings),
        )

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    async def _load_providers(
        self,
        raw: dict[str, Any],
        *,
        normalize_upstream_model: bool,
    ) -> dict[str, list[Provider]]:
        if "model_list" in raw:
            raise ValueError("model_list is no longer supported; use providers[*].models")

        providers_config = raw.get("providers")
        if not isinstance(providers_config, list):
            raise ValueError("providers must be a list")

        validated_providers: list[tuple[int, dict[str, Any], str, str | None]] = []
        auto_discovery_tasks: list[tuple[int, dict[str, Any], str | None]] = []
        seen_provider_names: set[str] = set()

        for provider_index, provider_params in enumerate(providers_config):
            if not isinstance(provider_params, dict):
                raise ValueError(f"providers[{provider_index}] must be a mapping")

            provider_name_value = provider_params.get("name")
            provider_name = (
                str(provider_name_value).strip()
                if provider_name_value is not None
                else ""
            )
            if not provider_name:
                raise ValueError(f"providers[{provider_index}].name is required")
            if provider_name in seen_provider_names:
                raise ValueError(
                    f"providers[{provider_index}].name must be unique: {provider_name}"
                )
            seen_provider_names.add(provider_name)

            if "api_key" not in provider_params:
                raise ValueError(f"providers[{provider_index}].api_key is required")
            api_base = _coerce_optional_api_base(provider_params.get("api_base"))
            api_url = _coerce_optional_url(provider_params.get("api_url"))
            if api_base is None and api_url is None:
                raise ValueError(
                    f"providers[{provider_index}] must include api_base or api_url"
                )

            endpoint_type = _coerce_endpoint_type(
                provider_params.get("endpoint_type"),
                context=f"providers[{provider_index}].endpoint_type",
            )
            if api_url is not None and endpoint_type is None:
                raise ValueError(
                    f"providers[{provider_index}].endpoint_type is required when "
                    "api_url is set"
                )

            models = provider_params.get("models")
            if isinstance(models, str) and models.strip().lower() == "auto":
                auto_discovery_tasks.append((provider_index, provider_params, endpoint_type))

            validated_providers.append((provider_index, provider_params, provider_name, endpoint_type))

        auto_discovery_results: dict[int, AutoModelDiscovery] = {}
        if auto_discovery_tasks:
            discovery_coros = [
                self._load_auto_models(params, provider_index=idx, endpoint_type=ep_type)
                for idx, params, ep_type in auto_discovery_tasks
            ]
            results = await asyncio.gather(*discovery_coros)
            auto_discovery_results = {
                auto_discovery_tasks[i][0]: result
                for i, result in enumerate(results)
            }

        grouped: dict[str, list[Provider]] = {}
        for provider_index, provider_params, provider_name, endpoint_type in validated_providers:
            models = provider_params.get("models")
            model_source = "explicit"
            discovered_model_ids: tuple[str, ...] = ()
            filtered_model_ids: tuple[str, ...] = ()
            discovery_warnings: tuple[str, ...] = ()

            if isinstance(models, str) and models.strip().lower() == "auto":
                auto_discovery = auto_discovery_results[provider_index]
                models = auto_discovery.models
                model_source = "auto"
                discovered_model_ids = auto_discovery.discovered_models
                filtered_model_ids = auto_discovery.filtered_models
                discovery_warnings = auto_discovery.warnings

            if not isinstance(models, list) or not models:
                raise ValueError(
                    f"providers[{provider_index}].models must be a non-empty list or 'auto'"
                )

            for model_index, model_entry in enumerate(models):
                model_name, configured_model, anthropic_role = self._coerce_model_entry(
                    model_entry,
                    provider_index=provider_index,
                    model_index=model_index,
                )
                provider = self._build_provider(
                    provider_name=provider_name,
                    model_name=model_name,
                    configured_model=configured_model,
                    anthropic_role=anthropic_role,
                    endpoint_type=endpoint_type,
                    provider_params=provider_params,
                    normalize_upstream_model=normalize_upstream_model,
                    provider_index=provider_index,
                    model_source=model_source,
                    discovered_model_ids=discovered_model_ids,
                    filtered_model_ids=filtered_model_ids,
                    discovery_warnings=discovery_warnings,
                )
                grouped.setdefault(model_name, []).append(provider)

        return {
            model: sorted(providers, key=lambda p: (p.order, p.sort_url, p.upstream_model))
            for model, providers in grouped.items()
        }

    async def _load_config(self) -> RouterConfig:
        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("Config root must be a mapping")

        app_settings = raw.get("app_settings")
        if app_settings is None:
            app_settings = {}
        if not isinstance(app_settings, dict):
            raise ValueError("app_settings must be a mapping")

        host = str(app_settings.get("host", "127.0.0.1"))
        port = int(app_settings.get("port", 8099))
        log_level = str(app_settings.get("log_level", "info"))
        default_timeout = float(app_settings.get("default_timeout", DEFAULT_TIMEOUT))
        stream_start_timeout = _coerce_positive_float(
            app_settings.get("stream_start_timeout"),
            min(default_timeout, DEFAULT_STREAM_START_TIMEOUT),
        )
        sticky_ttl_seconds = int(
            app_settings.get("sticky_ttl_seconds", DEFAULT_STICKY_TTL_SECONDS)
        )
        normalize_upstream_model = _coerce_bool(
            app_settings.get("normalize_upstream_model"), True
        )
        hot_reload = _coerce_bool(app_settings.get("hot_reload"), True)
        hot_reload_interval_seconds = _coerce_positive_float(
            app_settings.get("hot_reload_interval_seconds"),
            DEFAULT_HOT_RELOAD_INTERVAL_SECONDS,
        )

        router_settings = raw.get("router_settings")
        if router_settings is None:
            router_settings = {}
        if not isinstance(router_settings, dict):
            raise ValueError("router_settings must be a mapping")
        allowed_fails = int(router_settings.get("allowed_fails", 0))
        cooldown_time = int(router_settings.get("cooldown_time", 300))

        providers_by_model = await self._load_providers(
            raw,
            normalize_upstream_model=normalize_upstream_model,
        )
        return RouterConfig(
            host=host,
            port=port,
            log_level=log_level,
            default_timeout=default_timeout,
            stream_start_timeout=stream_start_timeout,
            sticky_ttl_seconds=sticky_ttl_seconds,
            normalize_upstream_model=normalize_upstream_model,
            hot_reload=hot_reload,
            hot_reload_interval_seconds=hot_reload_interval_seconds,
            allowed_fails=allowed_fails,
            cooldown_time=cooldown_time,
            providers_by_model=providers_by_model,
        )

    def _apply_config(self, config: RouterConfig) -> None:
        self.host = config.host
        self.port = config.port
        self.log_level = config.log_level
        self.default_timeout = config.default_timeout
        self.stream_start_timeout = config.stream_start_timeout
        self.sticky_ttl_seconds = config.sticky_ttl_seconds
        self.normalize_upstream_model = config.normalize_upstream_model
        self.hot_reload = config.hot_reload
        self.hot_reload_interval_seconds = config.hot_reload_interval_seconds
        self.allowed_fails = config.allowed_fails
        self.cooldown_time = config.cooldown_time
        self.providers_by_model = config.providers_by_model
        self.last_reload_at = time.time()
        self.last_reload_error = None
        try:
            config_mtime = os.path.getmtime(self.config_path)
        except OSError:
            config_mtime = None
        self.last_config_mtime = config_mtime
        self.last_observed_config_mtime = config_mtime
        self._log_config_loaded()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _provider_log_summaries(self) -> list[dict[str, Any]]:
        summaries: dict[tuple[str | None, str | None, int, str], dict[str, Any]] = {}
        for providers in self.providers_by_model.values():
            for provider in providers:
                route_url = provider.api_url or provider.api_base or ""
                key = (
                    provider.provider_name,
                    provider.endpoint_type,
                    provider.order,
                    route_url,
                )
                summary = summaries.setdefault(
                    key,
                    {
                        "name": provider.provider_name or "(unnamed)",
                        "endpoint_type": provider.endpoint_type or "(any)",
                        "order": provider.order,
                        "model_source": provider.model_source,
                        "models": set(),
                        "discovered_model_ids": provider.discovered_model_ids,
                        "filtered_model_ids": provider.filtered_model_ids,
                        "discovery_warnings": provider.discovery_warnings,
                    },
                )
                summary["models"].add(provider.model_name)
        return sorted(
            summaries.values(),
            key=lambda item: (item["order"], item["name"], item["endpoint_type"]),
        )

    def _log_config_loaded(self) -> None:
        provider_summaries = self._provider_log_summaries()
        model_route_count = sum(
            len(providers) for providers in self.providers_by_model.values()
        )
        logger.info(
            "Loaded config: %s providers, %s model routes",
            len(provider_summaries),
            model_route_count,
        )
        for summary in provider_summaries:
            models = sorted(summary["models"])
            logger.info(
                "Provider %s endpoint=%s order=%s models=%s count=%s ids=%s",
                summary["name"],
                summary["endpoint_type"],
                summary["order"],
                summary["model_source"],
                len(models),
                _format_model_ids(models),
            )
            if (
                summary["model_source"] == "auto"
                and set(summary["discovered_model_ids"]) != set(models)
            ):
                logger.info(
                    "Provider %s auto-discovered count=%s ids=%s",
                    summary["name"],
                    len(summary["discovered_model_ids"]),
                    _format_model_ids(summary["discovered_model_ids"]),
                )
            for warning in summary["discovery_warnings"]:
                logger.warning("Provider %s %s", summary["name"], warning)

    # ------------------------------------------------------------------
    # Hot reload
    # ------------------------------------------------------------------

    async def reload(self) -> ReloadResult:
        async with self._reload_lock:
            try:
                config = await self._load_config()
            except Exception as exc:
                error = f"{exc.__class__.__name__}: {exc}"
                async with self._lock:
                    self.last_reload_error = error
                return ReloadResult(status="rejected", reloaded=False, error=error)

            async with self._lock:
                self._apply_config(config)
            return ReloadResult(status="reloaded", reloaded=True)

    async def reload_if_changed(self) -> ReloadResult:
        try:
            current_mtime = os.path.getmtime(self.config_path)
        except OSError as exc:
            error = f"{exc.__class__.__name__}: {exc}"
            async with self._lock:
                self.last_reload_error = error
            return ReloadResult(status="rejected", reloaded=False, error=error)

        async with self._lock:
            previous_mtime = self.last_observed_config_mtime
        if previous_mtime is not None and current_mtime == previous_mtime:
            return ReloadResult(status="unchanged", reloaded=False)
        async with self._lock:
            self.last_observed_config_mtime = current_mtime
        return await self.reload()

    # ------------------------------------------------------------------
    # Provider selection and sticky routing
    # ------------------------------------------------------------------

    async def get_candidate_providers(
        self, model_name: str, endpoint: str, sticky_key: str | None
    ) -> list[Provider]:
        model_name = _normalize_requested_model(model_name)
        async with self._lock:
            providers = list(self.providers_by_model.get(model_name, []))
            if endpoint in ANTHROPIC_ENDPOINTS:
                role = _infer_anthropic_role_from_model_name(model_name)
                if role:
                    role_providers = [
                        provider
                        for model_providers in self.providers_by_model.values()
                        for provider in model_providers
                        if provider.anthropic_role == role
                    ]
                    seen_provider_ids = {provider.provider_id for provider in providers}
                    providers.extend(
                        provider
                        for provider in role_providers
                        if provider.provider_id not in seen_provider_ids
                    )
            providers = sorted(
                providers,
                key=lambda provider: (provider.order, provider.sort_url, provider.upstream_model),
            )
            if not providers:
                raise KeyError(model_name)

            now = time.time()
            healthy: list[Provider] = []
            cooling: list[Provider] = []
            for provider in providers:
                if not _endpoint_type_supports_endpoint(provider.endpoint_type, endpoint):
                    continue
                cooldown_until = self.cooldown_until.get(build_failure_key(provider, endpoint), 0)
                if cooldown_until <= now:
                    healthy.append(provider)
                else:
                    cooling.append(provider)

            if not healthy and not cooling:
                raise KeyError(model_name)

            if healthy:
                return [
                    *self._apply_sticky_preference(healthy, sticky_key),
                    *cooling,
                ]
            return self._apply_sticky_preference(cooling, sticky_key)

    def _apply_sticky_preference(
        self, providers: list[Provider], sticky_key: str | None
    ) -> list[Provider]:
        if not sticky_key or len(providers) <= 1:
            return providers

        binding = self.session_bindings.get(sticky_key)
        now = time.time()
        if binding is None:
            return providers
        if binding.expires_at <= now:
            self.session_bindings.pop(sticky_key, None)
            return providers

        for index, provider in enumerate(providers):
            if provider.provider_id == binding.provider_id:
                if index == 0:
                    return providers
                return [provider, *providers[:index], *providers[index + 1 :]]

        return providers

    # ------------------------------------------------------------------
    # Session binding
    # ------------------------------------------------------------------

    async def record_success(self, provider: Provider, endpoint: str) -> None:
        async with self._lock:
            failure_key = build_failure_key(provider, endpoint)
            self.fail_counts[failure_key] = 0
            self.cooldown_until.pop(failure_key, None)
            self.last_error.pop(failure_key, None)

    async def bind_session(self, sticky_key: str | None, provider: Provider) -> None:
        if not sticky_key:
            return
        async with self._lock:
            self.session_bindings[sticky_key] = StickyBinding(
                provider_id=provider.provider_id,
                expires_at=time.time() + self.sticky_ttl_seconds,
            )

    async def clear_session_binding(self, sticky_key: str | None, provider: Provider) -> None:
        if not sticky_key:
            return
        async with self._lock:
            binding = self.session_bindings.get(sticky_key)
            if binding and binding.provider_id == provider.provider_id:
                self.session_bindings.pop(sticky_key, None)

    async def record_failure(
        self,
        provider: Provider,
        endpoint: str,
        error_message: str,
        decision: FailureDecision,
    ) -> None:
        async with self._lock:
            failure_key = build_failure_key(provider, endpoint)
            self.last_error[failure_key] = error_message
            if not decision.count_failure:
                return

            count = self.fail_counts.get(failure_key, 0) + 1
            self.fail_counts[failure_key] = count
            if count > self.allowed_fails:
                cooldown_seconds = int(self.cooldown_time * decision.cooldown_multiplier)
                self.cooldown_until[failure_key] = time.time() + cooldown_seconds
                self.fail_counts[failure_key] = 0

    # ------------------------------------------------------------------
    # Snapshot / introspection
    # ------------------------------------------------------------------

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            now = time.time()
            providers: list[dict[str, Any]] = []
            for model_name, model_providers in self.providers_by_model.items():
                for provider in model_providers:
                    provider_id = provider.provider_id
                    responses_key = build_failure_key(provider, "/responses")
                    chat_key = build_failure_key(provider, "/chat/completions")
                    messages_key = build_failure_key(provider, "/messages")
                    count_tokens_key = build_failure_key(
                        provider,
                        "/messages/count_tokens",
                    )
                    responses_cooldown = self.cooldown_until.get(responses_key)
                    chat_cooldown = self.cooldown_until.get(chat_key)
                    messages_cooldown = self.cooldown_until.get(messages_key)
                    count_tokens_cooldown = self.cooldown_until.get(count_tokens_key)
                    providers.append(
                        {
                            "model_name": model_name,
                            "provider_name": provider.provider_name,
                            "provider_id": provider_id,
                            "upstream_model": provider.upstream_model,
                            "configured_model": provider.configured_model,
                            "anthropic_role": provider.anthropic_role,
                            "endpoint_type": provider.endpoint_type,
                            "model_source": provider.model_source,
                            "discovered_model_ids": list(provider.discovered_model_ids),
                            "filtered_model_ids": list(provider.filtered_model_ids),
                            "discovery_warnings": list(provider.discovery_warnings),
                            "api_base": provider.api_base,
                            "api_url": provider.api_url,
                            "models_url": provider.models_url,
                            "upstream_urls": {
                                "/responses": build_upstream_url(provider, "/responses"),
                                "/chat/completions": build_upstream_url(
                                    provider,
                                    "/chat/completions",
                                ),
                                "/messages": build_upstream_url(provider, "/messages"),
                                "/messages/count_tokens": build_upstream_url(
                                    provider,
                                    "/messages/count_tokens",
                                ),
                            },
                            "order": provider.order,
                            "timeout": provider.timeout,
                            "extra_headers": provider.extra_headers,
                            "cooldown_remaining_seconds": {
                                "/responses": max(0, int(responses_cooldown - now))
                                if responses_cooldown
                                else 0,
                                "/chat/completions": max(0, int(chat_cooldown - now))
                                if chat_cooldown
                                else 0,
                                "/messages": max(0, int(messages_cooldown - now))
                                if messages_cooldown
                                else 0,
                                "/messages/count_tokens": max(
                                    0,
                                    int(count_tokens_cooldown - now),
                                )
                                if count_tokens_cooldown
                                else 0,
                            },
                            "last_error": {
                                "/responses": self.last_error.get(responses_key),
                                "/chat/completions": self.last_error.get(chat_key),
                                "/messages": self.last_error.get(messages_key),
                                "/messages/count_tokens": self.last_error.get(
                                    count_tokens_key,
                                ),
                            },
                        }
                    )
            return {
                "config_path": self.config_path,
                "app_settings": {
                    "host": self.host,
                    "port": self.port,
                    "log_level": self.log_level,
                    "default_timeout": self.default_timeout,
                    "stream_start_timeout": self.stream_start_timeout,
                    "sticky_ttl_seconds": self.sticky_ttl_seconds,
                    "normalize_upstream_model": self.normalize_upstream_model,
                    "hot_reload": self.hot_reload,
                    "hot_reload_interval_seconds": self.hot_reload_interval_seconds,
                },
                "hot_reload": {
                    "enabled": self.hot_reload,
                    "interval_seconds": self.hot_reload_interval_seconds,
                    "last_success_at": self.last_reload_at,
                    "last_success_at_iso": datetime.fromtimestamp(
                        self.last_reload_at
                    ).isoformat()
                    if self.last_reload_at
                    else None,
                    "last_config_mtime": self.last_config_mtime,
                    "last_observed_config_mtime": self.last_observed_config_mtime,
                    "last_error": self.last_reload_error,
                },
                "allowed_fails": self.allowed_fails,
                "cooldown_time": self.cooldown_time,
                "session_bindings": [
                    {
                        "session_key": session_key,
                        "provider_id": binding.provider_id,
                        "expires_in_seconds": max(0, int(binding.expires_at - now)),
                    }
                    for session_key, binding in self.session_bindings.items()
                    if binding.expires_at > now
                ],
                "providers": providers,
            }

    async def list_models(self) -> list[dict[str, Any]]:
        async with self._lock:
            data: list[dict[str, Any]] = []
            for model_name, providers in sorted(self.providers_by_model.items()):
                primary = providers[0]
                data.append(
                    {
                        "id": model_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": "mini-fallback-proxy",
                        "value": primary.configured_model,
                        "root": primary.configured_model,
                        "parent": None,
                        "providers": [
                            provider_debug_dict(provider)
                            for provider in providers
                        ],
                    }
                )
            return data

    async def get_model(self, model_name: str) -> dict[str, Any]:
        async with self._lock:
            providers = self.providers_by_model.get(model_name)
            if not providers:
                raise KeyError(model_name)

            primary = providers[0]
            return {
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "mini-fallback-proxy",
                "value": primary.configured_model,
                "root": primary.configured_model,
                "parent": None,
                "providers": [
                    provider_debug_dict(provider)
                    for provider in providers
                ],
            }
