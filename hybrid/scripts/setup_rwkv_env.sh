#!/usr/bin/env bash
# shellcheck disable=SC1091
set -euo pipefail

# Resolve repository root relative to this script
ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR=${VENV_DIR:-"$ROOT_DIR/venv"}
PYTHON_BIN=${PYTHON_BIN:-python3}
PIP_FLAGS=${PIP_FLAGS:-}

# Configure reproducible git clones (URL|destination folder)
RWKV_REPOS=(
  "https://github.com/BlinkDL/RWKV-LM.git|$ROOT_DIR/RWKV-LM"
  "https://github.com/BlinkDL/RWKV-CUDA.git|$ROOT_DIR/RWKV-CUDA"
  "https://github.com/RWKV/RWKV-infctx-trainer.git|$ROOT_DIR/RWKV-infctx-trainer"
  "https://github.com/JL-er/RWKV-PEFT.git|$ROOT_DIR/RWKV-PEFT"
  "https://github.com/josStorer/RWKV-Runner.git|$ROOT_DIR/RWKV-Runner"
  "https://github.com/RWKV/rwkv.cpp.git|$ROOT_DIR/rwkv.cpp"
  "https://github.com/BlinkDL/ChatRWKV.git|$ROOT_DIR/ChatRWKV"
  "https://github.com/johanwind/wind_rwkv.git|$ROOT_DIR/wind_rwkv"
  "https://github.com/fla-org/flash-linear-attention.git|$ROOT_DIR/flash-linear-attention"
)

create_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[setup] Creating virtualenv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
}

install_requirements() {
  local pip_bin="$VENV_DIR/bin/pip"
  if [[ ! -x "$pip_bin" ]]; then
    echo "[setup] pip not found in venv ($pip_bin)" >&2
    exit 1
  fi
  echo "[setup] Upgrading pip tooling"
  "$pip_bin" install --upgrade pip wheel $PIP_FLAGS
  if [[ -f "$ROOT_DIR/requirements.txt" ]]; then
    echo "[setup] Installing project requirements"
    "$pip_bin" install $PIP_FLAGS -r "$ROOT_DIR/requirements.txt"
  fi
  if [[ -f "$ROOT_DIR/requirements-dev.txt" ]]; then
    echo "[setup] Installing developer requirements"
    "$pip_bin" install $PIP_FLAGS -r "$ROOT_DIR/requirements-dev.txt"
  fi
}

sync_repo() {
  local repo_url="$1"
  local dest_dir="$2"
  if [[ -d "$dest_dir/.git" ]]; then
    echo "[setup] Updating $(basename "$dest_dir")"
    git -C "$dest_dir" fetch --depth 1 origin
    git -C "$dest_dir" reset --hard FETCH_HEAD
  else
    echo "[setup] Cloning $(basename "$dest_dir")"
    git clone --depth 1 "$repo_url" "$dest_dir"
  fi
}

install_local_repos() {
  local pip_bin="$VENV_DIR/bin/pip"
  if [[ ! -x "$pip_bin" ]]; then
    echo "[setup] pip not found in venv ($pip_bin)" >&2
    exit 1
  fi
  if [[ -d "$ROOT_DIR/wind_rwkv" ]]; then
    echo "[setup] Installing wind_rwkv (editable)"
    "$pip_bin" install $PIP_FLAGS -e "$ROOT_DIR/wind_rwkv"
  fi
  if [[ -d "$ROOT_DIR/flash-linear-attention" ]]; then
    echo "[setup] Installing flash-linear-attention (editable)"
    "$pip_bin" install $PIP_FLAGS -e "$ROOT_DIR/flash-linear-attention"
  fi
}

main() {
  create_venv
  install_requirements
  for entry in "${RWKV_REPOS[@]}"; do
    IFS="|" read -r url dest <<<"$entry"
    sync_repo "$url" "$dest"
  done
  install_local_repos
  echo "[setup] RWKV environment ready"
}

main "$@"
