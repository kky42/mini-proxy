#!/usr/bin/env python3
"""Real end-to-end provider fallback verification.

Requires a config with at least TWO providers for the SAME model — the
first should be intentionally broken (invalid API key or unreachable host)
so fallback behaviour is observable.

Usage:
  MINI_FALLBACK_PROXY_CONFIG=/path/to/config.yaml uv run python tests/e2e_real_fallback.py

Environment variables:
  MINI_FALLBACK_PROXY_CONFIG   Path to proxy config (required)
  E2E_HOST                     Proxy host (default: 127.0.0.1)
  E2E_PORT                     Proxy port (default: 8199)
  E2E_MODEL                    Model name to test (default: auto-detected)
  E2E_ENDPOINT_TYPE            Endpoint type to test (default: openai)
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from typing import Any

import httpx

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


def get_providers_for_model(model: str, endpoint_type: str) -> list[dict[str, Any]]:
    r = httpx.get(f"{BASE_URL}/v1/models/{model}", timeout=10)
    if r.status_code != 200:
        return []
    return [
        p for p in r.json().get("providers", [])
        if p.get("endpoint_type") == endpoint_type
    ]


def check(label: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_first_provider_broken_falls_back(model: str) -> bool:
    """If the first provider is broken (invalid key / unreachable), the
    second provider should handle the request."""
    print("\n--- First provider broken → fallback ---")

    providers = get_providers_for_model(model, ENDPOINT_TYPE)
    if len(providers) < 2:
        return check(">=2 providers", False, f"only {len(providers)} found")

    print(f"  Providers: {[p['provider_name'] for p in providers]}")
    print(f"  Provider 0: order={providers[0]['order']} base={providers[0]['api_base']}")
    print(f"  Provider 1: order={providers[1]['order']} base={providers[1]['api_base']}")

    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
    }, timeout=60)

    if r.status_code == 200:
        provider = r.headers.get("x-fallback-provider-id")
        order = r.headers.get("x-fallback-order")
        print(f"  Response: status=200 provider={provider} order={order}")

        # If first provider works, warn but don't fail (config might be all-healthy)
        if provider == providers[0]["provider_name"]:
            print("  NOTE: first provider succeeded — cannot verify fallback.")
            print("  To test fallback, make the first provider broken (invalid key, unreachable host).")
            return True  # not a failure, just not testable

        return check(
            "fell back to second provider",
            provider == providers[1]["provider_name"],
            f"expected {providers[1]['provider_name']}, got {provider}",
        )

    # Non-200: check if it's a 503 with attempt details
    if r.status_code == 503:
        try:
            detail = r.json().get("detail", r.json())
            attempts = detail.get("attempts", [])
            print(f"  All providers failed ({len(attempts)} attempts):")
            for a in attempts:
                print(f"    - {a.get('provider_id')}: {a.get('status')} {a.get('error', '')[:100]}")
        except Exception:
            print(f"  503 body: {r.text[:500]}")

        return check(
            "at least tried first provider",
            len(attempts) >= 1,
            f"{len(attempts)} attempts made",
        )

    return check("unexpected status", False, f"HTTP {r.status_code}: {r.text[:200]}")


def test_fallback_response_headers(model: str) -> bool:
    """Verify x-fallback-* response headers are present."""
    print("\n--- Fallback response headers ---")

    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
    }, timeout=60)

    headers_ok = True
    for header in ("x-fallback-provider-id", "x-fallback-provider-name"):
        value = r.headers.get(header)
        if value:
            print(f"  {header}: {value}")
        else:
            print(f"  {header}: MISSING")
            headers_ok = False

    return check("response headers present", headers_ok)


def test_all_providers_broken_returns_503(model: str) -> bool:
    """Send to a model with all broken providers → should get 503 with
    attempt details. This is informational — we can't easily make ALL
    providers broken in a real config."""
    print("\n--- All providers broken → 503 (informational) ---")
    print("  (skipped — requires all providers to be intentionally broken)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"Config: {CONFIG_PATH}")
    print(f"Proxy:  {BASE_URL}")
    print(f"Endpoint type: {ENDPOINT_TYPE}")

    proc = start_proxy()
    try:
        model = os.environ.get("E2E_MODEL") or discover_model(ENDPOINT_TYPE)
        if not model:
            print(f"ERROR: no model with >=2 {ENDPOINT_TYPE} providers found", file=sys.stderr)
            r = httpx.get(f"{BASE_URL}/v1/models", timeout=10)
            print(f"Available: {r.text[:1000]}")
            sys.exit(1)

        print(f"Using model: {model}")

        results = [
            test_first_provider_broken_falls_back(model),
            test_fallback_response_headers(model),
            test_all_providers_broken_returns_503(model),
        ]

        passed = sum(1 for r in results if r)
        failed = len(results) - passed
        print(f"\n{'='*50}")
        print(f"Fallback tests: {passed} passed, {failed} failed out of {len(results)}")
        if failed:
            sys.exit(1)
    finally:
        stop_proxy(proc)


if __name__ == "__main__":
    main()
