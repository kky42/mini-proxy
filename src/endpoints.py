from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from . import globals as _g
from .constants import logger
from .forwarding import forward_request
from .routing import RouterState


# ---------------------------------------------------------------------------
# Startup guard
# ---------------------------------------------------------------------------

if not _g.CONFIG_PATH:
    raise RuntimeError(
        "MINI_FALLBACK_PROXY_CONFIG is not set. Start the server with "
        "./start.sh --config /path/to/config.yaml"
    )

router_state = RouterState.create_sync(os.path.expanduser(_g.CONFIG_PATH))
_g.router_state = router_state


# ---------------------------------------------------------------------------
# Hot-reload background task
# ---------------------------------------------------------------------------


async def watch_config_changes() -> None:
    while True:
        await asyncio.sleep(_g.router_state.hot_reload_interval_seconds)
        if not _g.router_state.hot_reload:
            continue

        result = await _g.router_state.reload_if_changed()
        if result.status == "reloaded":
            logger.info("Config hot-reloaded from %s", _g.router_state.config_path)
        elif result.status == "rejected":
            logger.error(
                "Config hot reload rejected for %s: %s",
                _g.router_state.config_path,
                result.error,
            )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    task = asyncio.create_task(watch_config_changes())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="Mini Fallback Proxy", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name": "mini-fallback-proxy",
        "config_path": _g.router_state.config_path,
        "endpoints": [
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/v1/responses",
            "/v1/chat/completions",
            "/healthz",
            "/debug/state",
        ],
    }


@app.head("/")
async def root_head() -> Response:
    return Response(status_code=200)


@app.get("/debug/state")
async def debug_state() -> dict[str, Any]:
    return await _g.router_state.snapshot()


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": await _g.router_state.list_models(),
    }


@app.get("/v1/models/{model_name}")
async def get_model(model_name: str) -> dict[str, Any]:
    try:
        return await _g.router_state.get_model(model_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'") from exc


@app.post("/admin/reload")
async def reload_config() -> dict[str, Any]:
    result = await _g.router_state.reload()
    response: dict[str, Any] = {"status": result.status, "reloaded": result.reloaded}
    if result.error:
        response["error"] = result.error
    return response


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    return await forward_request(request, "/chat/completions")


@app.post("/chat/completions")
async def chat_completions_root(request: Request) -> Any:
    return await forward_request(request, "/chat/completions")


@app.post("/v1/responses")
async def responses_api(request: Request) -> Any:
    return await forward_request(request, "/responses")


@app.post("/responses")
async def responses_api_root(request: Request) -> Any:
    return await forward_request(request, "/responses")


@app.post("/v1/responses/compact")
async def responses_compact_api(request: Request) -> Any:
    return await forward_request(request, "/responses/compact")


@app.post("/v1/messages")
async def messages_api(request: Request) -> Any:
    return await forward_request(request, "/messages")


@app.post("/v1/messages/count_tokens")
async def messages_count_tokens_api(request: Request) -> Any:
    return await forward_request(request, "/messages/count_tokens")


@app.post("/messages")
async def messages_api_root(request: Request) -> Any:
    return await forward_request(request, "/messages")


@app.post("/messages/count_tokens")
async def messages_count_tokens_api_root(request: Request) -> Any:
    return await forward_request(request, "/messages/count_tokens")


@app.get("/models")
async def list_models_root() -> dict[str, Any]:
    return await list_models()


@app.get("/models/{model_name}")
async def get_model_root(model_name: str) -> dict[str, Any]:
    return await get_model(model_name)
