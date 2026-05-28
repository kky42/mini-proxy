# Repository Guidelines

## Project Structure & Module Organization

This repository is a compact Python proxy service. The FastAPI application and routing logic live in `app.py`. Configuration examples live in `config.example.yaml`; keep sample values safe and non-secret. `start.sh` is the supported local launcher and reads host, port, and log level from YAML before starting uvicorn. Tests live under `tests/`, currently centered on hot reload and routing state in `tests/test_hot_reload.py`. Static assets are stored under `assets/`.

## Build, Test, and Development Commands

- `uv sync`: create or update the local `.venv` from `pyproject.toml` and `uv.lock`.
- `./start.sh --config /path/to/config.yaml`: install `uv` if missing, sync dependencies, set `MINI_FALLBACK_PROXY_CONFIG`, and run the service.
- `MINI_FALLBACK_PROXY_CONFIG=config.example.yaml uv run uvicorn app:app --host 127.0.0.1 --port 8099`: run uvicorn directly for debugging.
- `uv run python -m unittest discover -s tests`: run the test suite.
- `uv run python -m unittest tests.test_hot_reload`: run the current focused test module.

## Coding Style & Naming Conventions

Use Python 3.11+ syntax and type hints. Follow PEP 8 layout with 4-space indentation. Use `snake_case` for functions, variables, and helpers; use `PascalCase` for dataclasses, enums, and test classes. Internal helpers generally use a leading underscore, for example `_normalize_requested_model`. Prefer small pure helpers for config coercion, URL inference, and routing decisions. No formatter or linter is configured, so match nearby code and avoid unrelated style churn.

## Testing Guidelines

Tests use the standard library `unittest`, including `unittest.IsolatedAsyncioTestCase` for async behavior. Name test files `test_*.py`, test classes after the feature under test, and test methods `test_<behavior>`. When changing reload, routing, provider selection, model aliases, timeouts, or streaming behavior, add regression coverage in `tests/`. Keep network behavior mocked with local fakes rather than real upstream calls.

## Commit & Pull Request Guidelines

Recent commits use concise imperative subjects, such as `Support sticky sessions for client session keys` and `Fix responses stream fallback handling`. Keep the first line focused on the behavioral change. Pull requests should include a short problem statement, the implemented change, test results, and any config or endpoint compatibility notes. Link issues when available and include screenshots only for user-facing asset or documentation changes.

## Security & Configuration Tips

Do not commit real provider API keys, private config files, or credential-bearing logs. Keep `config.example.yaml` illustrative only. API keys should remain in local YAML files outside source control.
