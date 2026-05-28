#!/usr/bin/env python3
"""Real end-to-end session stickiness verification.

Requires a config with at least TWO providers for the SAME model on the
SAME endpoint type so sticky routing is observable.

Usage:
  MINI_FALLBACK_PROXY_CONFIG=/path/to/config.yaml uv run python tests/e2e_real_sticky.py

The script starts the proxy server, sends multi-turn requests mimicking
pi-agent, Claude Code, and Responses API clients, then verifies that the
same session uses the same upstream provider (x-fallback-provider-id).

Environment variables:
  MINI_FALLBACK_PROXY_CONFIG   Path to proxy config (required)
  E2E_HOST                     Proxy host (default: 127.0.0.1)
  E2E_PORT                     Proxy port (default: 8199)
  E2E_MODEL                    Model name to test (default: auto-detected)
  E2E_ENDPOINT_TYPE            Endpoint type: openai, anthropic, or all (default: all)
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
ENDPOINT_TYPE = os.environ.get("E2E_ENDPOINT_TYPE", "all")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def start_proxy() -> subprocess.Popen:
    """Start the proxy server in a subprocess, wait until it's ready."""
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
    # Wait for server to be ready
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
    """Find a model that has >=2 providers for the given endpoint type."""
    r = httpx.get(f"{BASE_URL}/v1/models", timeout=10)
    if r.status_code != 200:
        return None
    data = r.json().get("data", [])
    for model in data:
        providers = model.get("providers", [])
        matching = [
            p for p in providers
            if p.get("endpoint_type") == endpoint_type or endpoint_type == "all"
        ]
        if len(matching) >= 2:
            return model["id"]
    return None


def check(label: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


# ---------------------------------------------------------------------------
# Test: pi-agent (OpenAI /v1/chat/completions) — content fingerprint
# ---------------------------------------------------------------------------


def test_pi_agent_sticky(model: str) -> bool:
    print("\n--- pi-agent (/v1/chat/completions) fingerprint stickiness ---")

    session_messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    # Turn 1
    r1 = httpx.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": model,
        "messages": session_messages,
    }, timeout=60)
    if r1.status_code != 200:
        return check("turn 1", False, f"HTTP {r1.status_code}: {r1.text[:200]}")
    p1 = r1.headers.get("x-fallback-provider-id")
    check("turn 1 succeeds", True, f"provider={p1}")

    if not p1:
        return check("turn 1 provider header", False, "missing x-fallback-provider-id")

    # Turn 2 — same session (expanded messages, same first user message)
    r2 = httpx.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "What is 3+3?"},
        ],
    }, timeout=60)
    p2 = r2.headers.get("x-fallback-provider-id")
    ok = p1 == p2
    return check("same session → same provider", ok, f"{p1} → {p2}")


# ---------------------------------------------------------------------------
# Test: Claude Code (Anthropic /v1/messages) — metadata.user_id
# ---------------------------------------------------------------------------


def test_claude_code_sticky(model: str) -> bool:
    print("\n--- Claude Code (/v1/messages) metadata.user_id stickiness ---")

    session_id = "e2e-real-test-session-0001"
    user_id = json.dumps({
        "device_id": "e2e-device",
        "account_uuid": "",
        "session_id": session_id,
    })

    # Turn 1
    r1 = httpx.post(f"{BASE_URL}/v1/messages", json={
        "model": model,
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "metadata": {"user_id": user_id},
    }, timeout=60)
    if r1.status_code != 200:
        return check("turn 1", False, f"HTTP {r1.status_code}: {r1.text[:200]}")
    p1 = r1.headers.get("x-fallback-provider-id")
    check("turn 1 succeeds", True, f"provider={p1}")

    if not p1:
        return check("turn 1 provider header", False, "missing x-fallback-provider-id")

    # Turn 2 — same session
    r2 = httpx.post(f"{BASE_URL}/v1/messages", json={
        "model": model,
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": "Say hello in one word."},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Now say goodbye."},
        ],
        "metadata": {"user_id": user_id},
    }, timeout=60)
    p2 = r2.headers.get("x-fallback-provider-id")
    ok = p1 == p2
    return check("same session → same provider", ok, f"{p1} → {p2}")


# ---------------------------------------------------------------------------
# Test: Explicit header (x-fallback-session)
# ---------------------------------------------------------------------------


def test_explicit_header_sticky(model: str) -> bool:
    print("\n--- Explicit header (x-fallback-session) stickiness ---")

    headers = {"x-fallback-session": "e2e-explicit-session-0001"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    }

    r1 = httpx.post(
        f"{BASE_URL}/v1/chat/completions", json=body, headers=headers, timeout=60,
    )
    if r1.status_code != 200:
        return check("turn 1", False, f"HTTP {r1.status_code}: {r1.text[:200]}")
    p1 = r1.headers.get("x-fallback-provider-id")

    r2 = httpx.post(
        f"{BASE_URL}/v1/chat/completions", json=body, headers=headers, timeout=60,
    )
    p2 = r2.headers.get("x-fallback-provider-id")

    ok = p1 is not None and p1 == p2
    return check("explicit header → same provider", ok, f"{p1} → {p2}")


# ---------------------------------------------------------------------------
# Test: Cross-client isolation
# ---------------------------------------------------------------------------


def test_cross_client_isolation(openai_model: str, anthropic_model: str) -> bool:
    print("\n--- Cross-client isolation ---")

    r1 = httpx.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": openai_model,
        "messages": [{"role": "user", "content": "Hello"}],
    }, timeout=60)
    p1 = r1.headers.get("x-fallback-provider-id")

    r2 = httpx.post(f"{BASE_URL}/v1/messages", json={
        "model": anthropic_model,
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {"user_id": json.dumps({"session_id": "e2e-isolation-test"})},
    }, timeout=60)
    p2 = r2.headers.get("x-fallback-provider-id")

    if not p1 or not p2:
        return check("both endpoints respond", False, f"pi={p1}, claude={p2}")

    # They should use different provider families
    ok = p1 != p2
    return check("different endpoint types → different providers", ok, f"pi={p1}, claude={p2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"Config: {CONFIG_PATH}")
    print(f"Proxy:  {BASE_URL}")

    proc = start_proxy()
    try:
        # Discover models
        openai_model = os.environ.get("E2E_MODEL") or discover_model("openai-compatible")
        anthropic_model = os.environ.get("E2E_MODEL") or discover_model("anthropic")

        if not openai_model and not anthropic_model:
            print("ERROR: no models with >=2 providers found", file=sys.stderr)
            r = httpx.get(f"{BASE_URL}/v1/models", timeout=10)
            print(f"Available models: {r.text[:1000]}")
            sys.exit(1)

        results: list[bool] = []

        if ENDPOINT_TYPE in ("all", "openai") and openai_model:
            print(f"\nUsing OpenAI model: {openai_model}")
            results.append(test_pi_agent_sticky(openai_model))
            results.append(test_explicit_header_sticky(openai_model))

        if ENDPOINT_TYPE in ("all", "anthropic") and anthropic_model:
            print(f"\nUsing Anthropic model: {anthropic_model}")
            results.append(test_claude_code_sticky(anthropic_model))

        if openai_model and anthropic_model:
            results.append(test_cross_client_isolation(openai_model, anthropic_model))

        passed = sum(results)
        failed = len(results) - passed
        print(f"\n{'='*50}")
        print(f"Sticky tests: {passed} passed, {failed} failed out of {len(results)}")
        if failed:
            sys.exit(1)
    finally:
        stop_proxy(proc)


if __name__ == "__main__":
    main()
