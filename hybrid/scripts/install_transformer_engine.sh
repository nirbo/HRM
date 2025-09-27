#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the target virtualenv before running this script" >&2
  exit 1
fi

CUDNN_ROOT="${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cudnn"
if [[ ! -d "${CUDNN_ROOT}" ]]; then
  echo "error: cuDNN package not found at ${CUDNN_ROOT}. Install nvidia-cudnn-cu12 via pip first." >&2
  exit 1
fi

if [[ -d "${CUDNN_ROOT}/include" ]]; then
  export CUDNN_INCLUDE_DIR="${CUDNN_ROOT}/include"
else
  echo "error: ${CUDNN_ROOT}/include not found (expected cudnn headers)." >&2
  exit 1
fi

if [[ -d "${CUDNN_ROOT}/lib64" ]]; then
  export CUDNN_LIBRARY_DIR="${CUDNN_ROOT}/lib64"
elif [[ -d "${CUDNN_ROOT}/lib" ]]; then
  export CUDNN_LIBRARY_DIR="${CUDNN_ROOT}/lib"
else
  echo "error: neither lib/ nor lib64/ found under ${CUDNN_ROOT}." >&2
  exit 1
fi

export CPATH="${CUDNN_INCLUDE_DIR}:${CPATH:-}"
export LIBRARY_PATH="${CUDNN_LIBRARY_DIR}:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDNN_LIBRARY_DIR}:${LD_LIBRARY_PATH:-}"

TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" MAX_JOBS="${MAX_JOBS:-16}" \
  pip install --force-reinstall \
  --extra-index-url https://pypi.nvidia.com \
  "transformer_engine[pytorch]==2.7.0"
