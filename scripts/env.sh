#!/usr/bin/env bash
set -euo pipefail
if [ -n "${BASH_VERSION:-}" ]; then
  DEID_ENV_FILE="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  DEID_ENV_FILE="${(%):-%x}"
else
  echo 'Use bash or zsh to load scripts/env.sh' >&2
  return 1 2>/dev/null || exit 1
fi
DEID_ROOT=$(cd "$(dirname "$DEID_ENV_FILE")/.." && pwd)
if [ -f "$DEID_ROOT/.deid.env" ]; then
  source "$DEID_ROOT/.deid.env"
fi
: "${DEID_STORAGE_ROOT:?Copy .deid.env.example to .deid.env and configure the mounted storage volume}"
export DEID_STORAGE_ROOT
export DEID_REQUIRE_EXTERNAL="${DEID_REQUIRE_EXTERNAL:-1}"
export DEID_WORK="$DEID_STORAGE_ROOT/medical-deid-rs"
python3 "$DEID_ROOT/scripts/storage.py" check
export CARGO_HOME="$DEID_WORK/cargo-home"
export CARGO_TARGET_DIR="$DEID_WORK/target"
export TMPDIR="$DEID_WORK/tmp"
export HF_HOME="$DEID_WORK/cache/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export UV_CACHE_DIR="$DEID_WORK/cache/uv"
export UV_PYTHON_INSTALL_DIR="$DEID_WORK/python"
export UV_PROJECT_ENVIRONMENT="$DEID_WORK/venv"
export PIP_CACHE_DIR="$DEID_WORK/cache/pip"
export PYTHONPYCACHEPREFIX="$DEID_WORK/cache/pycache"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$CARGO_HOME" "$CARGO_TARGET_DIR" "$TMPDIR" "$HF_HOME" "$UV_CACHE_DIR"
