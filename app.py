"""Thin re-export facade. All logic lives in src/."""

from __future__ import annotations

import asyncio
import httpx
import sys
import types

# Re-export everything tests and external callers expect.
from src.constants import (  # noqa: F401
    ANTHROPIC_ENDPOINTS,
    ANTHROPIC_ONE_M_SUFFIX_RE,
    ANTHROPIC_ROLE_MODEL_ALIASES,
    ANTHROPIC_VERSION,
    DEFAULT_HOT_RELOAD_INTERVAL_SECONDS,
    DEFAULT_STICKY_TTL_SECONDS,
    DEFAULT_STREAM_START_TIMEOUT,
    DEFAULT_TIMEOUT,
    ENDPOINT_TYPES,
    logger,
)
from src.types import (  # noqa: F401
    AutoModelDiscovery,
    FailureClass,
    FailureDecision,
    Provider,
    ReloadResult,
    RouterConfig,
    StickyBinding,
)
from src.utils import (  # noqa: F401
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
    _extract_error_code,
    _extract_error_message,
    _extract_responses_json_from_sse,
    _extract_responses_json_from_sse_text,
    _extract_sse_error_message,
    _infer_anthropic_role_from_model_name,
    _infer_models_url,
    _iter_sse_events,
    _normalize_requested_model,
    _normalize_upstream_model,
    _safe_json_loads,
    _split_model_mapping_suffix,
    _split_model_role_suffix,
    _strip_anthropic_context_marker,
    build_failure_key,
    build_sticky_key,
    build_upstream_url,
)
from src.session import _extract_session_key  # noqa: F401
from src.classification import (  # noqa: F401
    classify_http_error,
    classify_transport_error,
)
from src.routing import (  # noqa: F401
    RouterState,
    _format_model_ids,
    provider_attempt_dict,
    provider_debug_dict,
    provider_response_headers,
)
from src.forwarding import (  # noqa: F401
    build_upstream_headers,
    build_upstream_timeout,
    forward_request,
    is_valid_stream_success_content_type,
    log_provider_event,
    parse_upstream_success_body,
    pick_timeout,
    stream_upstream,
    validate_responses_stream_start,
)
from src.endpoints import (  # noqa: F401
    app,
    lifespan,
    watch_config_changes,
)
from src import globals as _globals


# ---- Module-class hack for test monkeypatching ----
# Tests do `app_module.router_state = custom_state`.  With this,
# assignment goes through a property setter that updates
# src.globals.router_state, which forwarding.py reads via attribute
# access on the globals module.  This keeps the singleton in sync
# across the split modules without changing the test API at all.
class _AppModule(types.ModuleType):
    @property
    def router_state(self):
        return _globals.router_state

    @router_state.setter
    def router_state(self, value):
        _globals.router_state = value


sys.modules[__name__].__class__ = _AppModule
