# mini-fallback-proxy

![mini-fallback-proxy logo](assets/logo/mini-fallback-proxy-logo-v1.png)

A small local proxy with:

- strict ordered fallback
- in-memory cooldown after failure
- automatic return to higher-priority providers after cooldown expires
- OpenAI-compatible `responses` and `chat.completions` endpoints

It is designed to be a thin local layer, not a full gateway.

## Start

```bash
cd /path/to/mini-fallback-proxy
chmod +x ./start.sh
./start.sh --config ~/.config/litellm.yaml
```

`--config` is required.

`start.sh` will:

- install `uv` locally if it is missing
- create or update the project `.venv` with `uv sync`
- read bind settings from the config file
- launch the server with that config

## Config Example

Example file: [config.example.yaml](/Users/kky/Desktop/mini-fallback-proxy/config.example.yaml)

```yaml
app_settings:
  host: 0.0.0.0
  port: 8099
  log_level: info
  default_timeout: 60
  normalize_upstream_model: true

router_settings:
  allowed_fails: 1
  cooldown_time: 300

providers:
  - name: jucode
    api_base: https://cf.jucode.top/v1
    api_key: sk-your-jucode-key
    order: 1
    models:
      - gpt-5.4
      - gpt-5.4-mini
      - gpt-5.3-codex

  - name: rightcode
    api_base: https://right.codes/codex/v1
    api_key: sk-your-rightcode-key
    order: 2
    models:
      - gpt-5.4
      - gpt-5.4-mini
      - gpt-5.3-codex

  - name: yescode
    api_base: https://co.yes.vg/v1
    api_key: sk-your-yescode-key
    order: 3
    models:
      - gpt-5.4
      - gpt-5.4-mini
      - gpt-5.3-codex
```

## Supported Settings

`app_settings`

- `host`: bind host for the local server. Default `127.0.0.1`.
  Use `0.0.0.0` if you want other machines on your LAN to access it.
- `port`: bind port for the local server. Default `8099`.
- `log_level`: uvicorn log level. Default `info`.
- `default_timeout`: request timeout in seconds when neither request body nor provider sets a timeout. Default `60`.
  For streaming requests, this proxy keeps `connect`/`write`/`pool` bounded but disables the upstream `read` timeout so long-thinking SSE streams are not cut off mid-response.
- `sticky_ttl_seconds`: optional session stickiness TTL in seconds. Default `1800`, so you do not need to set it unless you want to override it.
- `normalize_upstream_model`: if `true`, `openai/gpt-5.4` becomes `gpt-5.4` before forwarding upstream. Default `true`.
- `hot_reload`: if `true`, the proxy polls the config file and applies valid edits without restart. Default `true`.
- `hot_reload_interval_seconds`: config file polling interval. Default `1`.

`host`, `port`, and `log_level` are read by `start.sh` before uvicorn starts. Editing them is reflected in `/debug/state` after reload, but changing the actual listening socket or uvicorn log level still requires restarting the process.

`router_settings`

- `allowed_fails`: allowed failures before cooldown. Cooldown starts only when `fail_count > allowed_fails`. Default `0`.
- `cooldown_time`: cooldown length in seconds. Default `300`.

`providers[*]`

- `name`: optional provider label shown in debug/model output.
- `api_base`: upstream base URL.
- `api_key`: upstream API key.
- `order`: provider priority. Lower number means higher priority.
- `timeout`: optional provider-specific timeout in seconds.
- `headers`: optional extra headers sent on every request to that upstream.
- `models`: non-empty list of models this provider supports.

`providers[*].models[*]`

- String form, for same local and upstream model name: `gpt-5.4`.
- Mapping form, for aliases or provider-specific upstream names:

```yaml
models:
  - model_name: gpt-5.4
    model: openai/gpt-5.4
```

`model_name` is the alias exposed by this local proxy. `model` is the upstream model value sent to that provider.

`general_settings`

- ignored by this project. It can stay in the file for compatibility with your existing LiteLLM config.

## Endpoints

- `POST /v1/responses`
- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /`
- `GET /healthz`
- `GET /debug/state`
- `POST /admin/reload`

## Behavior Notes

- For a requested model, the proxy tries only providers whose `models` list includes that model, ordered by lowest `order` first.
- Session stickiness is enabled by default for 30 minutes.
- Session key extraction order is: `x-fallback-session` header, then request body `conversation_id`, `thread_id`, `previous_response_id`, `user`, then the same keys under `metadata`.
- Stickiness is applied per `session + endpoint + model alias`.
- Cooldown state is tracked per `provider + endpoint + model alias`, not globally.
- A provider enters cooldown only when its counted failure count becomes greater than `allowed_fails`.
- After cooldown expires, new requests automatically go back to the higher-priority provider.
- Request body `timeout` overrides provider timeout, which overrides `app_settings.default_timeout`.
- For streaming requests, the selected timeout still applies to connect/write/pool, but upstream idle reads are left unbounded to avoid false reconnect loops.
- Streaming fallback is only supported before the upstream stream starts.
- Failures and cooldown state live in memory only.
- Config hot reload is enabled by default. Valid changes to routes, providers, timeouts, cooldown settings, stickiness TTL, and model normalization are applied automatically.
- If a changed config is invalid, the proxy keeps serving with the last good config and records the error in `/debug/state`.
- For your current `jucode` setup, `responses` works with bare model names such as `gpt-5.4`. This proxy normalizes `openai/gpt-5.4` to `gpt-5.4` by default.

Built-in default failure policy:

- `401`, `402`, `403`, timeouts, transport errors, `408`, `429`, `5xx`: fallback and count toward cooldown
- model/endpoint/parameter capability mismatch: fallback but do not count toward cooldown
- request-invalid errors such as generic `400/422`, context issues, content policy: do not count; default is no fallback
- mid-stream disconnects: count toward cooldown but do not transparently replay on another provider

## Debug

```bash
curl http://127.0.0.1:8099/
curl http://127.0.0.1:8099/healthz
curl http://127.0.0.1:8099/debug/state
curl -X POST http://127.0.0.1:8099/admin/reload
```
