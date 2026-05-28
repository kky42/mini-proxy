#!/usr/bin/env python3
"""Real end-to-end provider cooldown verification.

Requires a config with allowed_fails=0, cooldown_time set to a short
value, and a broken provider at order=0 (unreachable host / invalid key)
so that cooldown behaviour is observable.

Usage:
  MINI_FALLBACK_PROXY_CONFIG=tests/e2e_config.yaml uv run python tests/e2e_real_cooldown.py

Environment variables:
  MINI_FALLBACK_PROXY_CONFIG   Path to proxy config (default: tests/e2e_config.yaml)
  E2E_HOST                     Proxy host (default: 127.0.0.1)
  E2E_PORT                     Proxy port (default: 8199)
  E2E_MODEL                    Model to test (default: auto-detected, favours openai-compatible)
  E2E_ENDPOINT_TYPE            Endpoint type to test (default: openai-compatible)
  E2E_COOLDOWN_SECONDS         Expected cooldown (default: from config, fallback 5)
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = os.environ.get(
    "MINI_FALLBACK_PROXY_CONFIG",
    os.path.join(os.path.dirname(__file__), "e2e_config.yaml"),
)
HOST = os.environ.get("E2E_HOST", "127.0.0.1")
PORT = int(os.environ.get("E2E_PORT", "8199"))
BASE_URL = f"http://{HOST}:{PORT}"
ENDPOINT_TYPE = os.environ.get("E2E_ENDPOINT_TYPE", "openai-compatible")

# Read cooldown_time from config
_cooldown_from_config = 5
try:
    with open(CONFIG_PATH) as f:
        _cfg = yaml.safe_load(f) or {}
    _cooldown_from_config = int(
        (_cfg.get("router_settings") or {}).get("cooldown_time", 5)
    )
except Exception:
    pass
COOLDOWN_SECONDS = int(os.environ.get("E2E_COOLDOWN_SECONDS", str(_cooldown_from_config)))

# Endpoint to use for the request
ENDPOINT_MAP = {
    "openai-compatible": "/chat/completions",
    "anthropic": "/messages",
    "responses": "/responses",
}
ENDPOINT = ENDPOINT_MAP.get(ENDPOINT_TYPE, "/chat/completions")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def start_proxy() -> subprocess.Popen:
    env = os.environ.copy()
    env["MINI_FALLBACK_PROXY_CONFIG"] = CONFIG_PATH
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "app:app",
            "--host", HOST,
            "--port", str(PORT),
            "--log-level", "warning",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            r = httpx.get(f"{BASE_URL}/healthz", timeout=2)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            out, err = proc.communicate()
            print(f"Proxy exited early.\nstdout:\n{out.decode()}\nstderr:\n{err.decode()}")
            sys.exit(1)
        time.sleep(0.3)
    proc.kill()
    print("ERROR: proxy did not start within 15s", file=sys.stderr)
    sys.exit(1)


def stop_proxy(proc: subprocess.Popen):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def discover_model(endpoint_type: str) -> str | None:
    """Find a model with >=2 providers for the given endpoint type."""
    r = httpx.get(f"{BASE_URL}/v1/models", timeout=10)
    if r.status_code != 200:
        return None
    for model in r.json().get("data", []):
        matching = [
            p for p in model.get("providers", [])
            if p.get("endpoint_type") == endpoint_type
        ]
        if len(matching) >= 2:
            return model["id"]
    return None


def get_debug_state() -> dict[str, Any]:
    r = httpx.get(f"{BASE_URL}/debug/state", timeout=10)
    if r.status_code != 200:
        return {}
    return r.json()


def check(label: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def find_broken_provider_name(state: dict, model: str) -> str | None:
    """Find a provider name that looks like the broken/injected one."""
    for p in state.get("providers", []):
        if p["model_name"] == model and "broken" in p["provider_name"].lower():
            return p["provider_name"]
    return None


def get_provider_cooldown(state: dict, provider_name: str) -> int:
    for p in state.get("providers", []):
        if p["provider_name"] == provider_name:
            return p["cooldown_remaining_seconds"].get(ENDPOINT, 0)
    return 0


# ---------------------------------------------------------------------------
# Request builder (endpoint-specific payload)
# ---------------------------------------------------------------------------


def make_request(model: str, message: str, **extra) -> httpx.Response:
    """Send a request to the correct endpoint for ENDPOINT_TYPE."""
    if ENDPOINT_TYPE == "anthropic":
        return httpx.post(f"{BASE_URL}{ENDPOINT}", json={
            "model": model,
            "max_tokens": 50,
            "messages": [{"role": "user", "content": message}],
            **extra,
        }, timeout=60)
    elif ENDPOINT_TYPE == "responses":
        return httpx.post(f"{BASE_URL}{ENDPOINT}", json={
            "model": model,
            "input": message,
            **extra,
        }, timeout=60)
    else:
        return httpx.post(f"{BASE_URL}{ENDPOINT}", json={
            "model": model,
            "messages": [{"role": "user", "content": message}],
            **extra,
        }, timeout=60)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_broken_provider_enters_cooldown(model: str) -> bool:
    """First request: broken provider fails → falls back → enters cooldown."""
    print("\n--- Broken provider enters cooldown ---")

    r = make_request(model, "Say hi")
    if r.status_code != 200:
        return check("request succeeds via fallback", False,
                     f"HTTP {r.status_code}: {r.text[:200]}")

    provider = r.headers.get("x-fallback-provider-id")
    print(f"  Response from: {provider}")

    if "broken" in (provider or "").lower():
        return check("fell back past broken provider", False,
                     "broken provider unexpectedly succeeded")

    # Verify the broken provider is now in cooldown
    state = get_debug_state()
    broken_name = find_broken_provider_name(state, model)
    if not broken_name:
        return check("broken provider in state", False, f"providers: {[p['provider_name'] for p in state.get('providers', []) if p['model_name'] == model]}")

    remaining = get_provider_cooldown(state, broken_name)
    return check(
        f"{broken_name} entered cooldown",
        remaining > 0,
        f"cooldown remaining: {remaining}s",
    )


def test_cooling_provider_skipped(model: str) -> bool:
    """While broken is cooling, new requests skip it."""
    print("\n--- Cooling provider skipped ---")

    r = make_request(model, "Another request while cooling")
    if r.status_code != 200:
        return check("request succeeds", False, f"HTTP {r.status_code}")

    provider = r.headers.get("x-fallback-provider-id")
    state = get_debug_state()
    broken_name = find_broken_provider_name(state, model)

    if not broken_name:
        return check("broken provider found", False)

    if "broken" in (provider or "").lower():
        return check("cooling provider skipped", False,
                     "cooling provider was still tried first")

    remaining = get_provider_cooldown(state, broken_name)
    return check(
        "cooling provider skipped, still in cooldown",
        remaining > 0,
        f"{broken_name} remaining={remaining}s, response from {provider}",
    )


def test_cooldown_expiry_restores_provider(model: str) -> bool:
    """After cooldown expires, broken provider is tried first, fails again, re-enters cooldown."""
    print(f"\n--- Cooldown expiry (waiting {COOLDOWN_SECONDS + 2}s) ---")

    state = get_debug_state()
    broken_name = find_broken_provider_name(state, model)
    if not broken_name:
        return check("broken provider found", False)

    max_remaining = get_provider_cooldown(state, broken_name)
    print(f"  {broken_name} cooldown remaining: {max_remaining}s")

    wait = max(max_remaining, COOLDOWN_SECONDS) + 2
    print(f"  Waiting {wait}s ...")
    time.sleep(wait)

    # Verify cooldown expired
    state = get_debug_state()
    remaining = get_provider_cooldown(state, broken_name)
    check("cooldown expired", remaining == 0, f"remaining={remaining}s")

    # Send request — broken is tried first (order=0), fails, falls back
    r = make_request(model, "After cooldown expiry")
    if r.status_code != 200:
        return check("request succeeds after cooldown", False,
                     f"HTTP {r.status_code}: {r.text[:200]}")

    provider = r.headers.get("x-fallback-provider-id")
    if "broken" in (provider or "").lower():
        return check("broken tried, failed, fell back", False,
                     "broken unexpectedly succeeded after cooldown")

    # Should have re-entered cooldown
    state = get_debug_state()
    new_remaining = get_provider_cooldown(state, broken_name)
    return check(
        "broken tried → failed → re-entered cooldown",
        new_remaining > 0,
        f"re-cooldown: {new_remaining}s, response from {provider}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"Config:       {CONFIG_PATH}")
    print(f"Proxy:        {BASE_URL}")
    print(f"Endpoint:     {ENDPOINT} ({ENDPOINT_TYPE})")
    print(f"Cooldown:     {COOLDOWN_SECONDS}s")

    proc = start_proxy()
    try:
        time.sleep(1)

        model = os.environ.get("E2E_MODEL") or discover_model(ENDPOINT_TYPE)
        if not model:
            print(f"ERROR: no model with >=2 {ENDPOINT_TYPE} providers", file=sys.stderr)
            r = httpx.get(f"{BASE_URL}/v1/models", timeout=10)
            print(f"Available: {r.text[:1000]}")
            sys.exit(1)

        print(f"Model:        {model}")
        print(f"Broken first: order=0, timeout=2s, unreachable host")

        results = [
            test_broken_provider_enters_cooldown(model),
            test_cooling_provider_skipped(model),
            test_cooldown_expiry_restores_provider(model),
        ]

        passed = sum(1 for r in results if r)
        failed = len(results) - passed
        print(f"\n{'='*50}")
        print(f"Cooldown tests: {passed} passed, {failed} failed out of {len(results)}")
        if failed:
            sys.exit(1)
    finally:
        stop_proxy(proc)


if __name__ == "__main__":
    main()
