#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH=""

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --config /path/to/config.yaml

Options:
  --config PATH   Config file to load
  -h, --help      Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --config" >&2; exit 1; }
      CONFIG_PATH="$1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CONFIG_PATH" ]]; then
  echo "--config is required" >&2
  usage >&2
  exit 1
fi

CONFIG_PATH="${CONFIG_PATH/#\~/$HOME}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  echo "uv not found, installing locally..." >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

  if ! command -v uv >/dev/null 2>&1; then
    echo "Failed to install uv" >&2
    exit 1
  fi
}

ensure_uv

cd "$SCRIPT_DIR"
uv sync --quiet

read_config_value() {
  local key="$1"
  uv run python - "$CONFIG_PATH" "$key" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1]).expanduser()
key = sys.argv[2]
data = yaml.safe_load(config_path.read_text()) or {}
app = data.get("app_settings") or {}
defaults = {
    "host": "127.0.0.1",
    "port": 8099,
    "log_level": "info",
}
print(app.get(key, defaults[key]))
PY
}

HOST="$(read_config_value host)"
PORT="$(read_config_value port)"
LOG_LEVEL="$(read_config_value log_level)"

export MINI_FALLBACK_PROXY_CONFIG="$CONFIG_PATH"

echo "Starting mini-fallback-proxy" >&2
echo "Config: $CONFIG_PATH" >&2
echo "Listen: http://$HOST:$PORT" >&2

exec uv run uvicorn app:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
