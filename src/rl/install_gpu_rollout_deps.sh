#!/bin/bash
set -euo pipefail

# Install only when GPU rollout is needed. This can download large wheels and may
# need a torch/vLLM/CUDA version match, so the CPU data-prep environment does not
# run it automatically.

INSTALL_VLLM="${INSTALL_VLLM:-1}"
INSTALL_SGLANG="${INSTALL_SGLANG:-0}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
INSTALL_FLASHINFER="${INSTALL_FLASHINFER:-0}"

if [ "$INSTALL_VLLM" = "1" ]; then
  python -m pip install "vllm"
fi

if [ "$INSTALL_SGLANG" = "1" ]; then
  python -m pip install "sglang[all]"
fi

if [ "$INSTALL_FLASH_ATTN" = "1" ]; then
  python -m pip install "flash-attn" --no-build-isolation
fi

if [ "$INSTALL_FLASHINFER" = "1" ]; then
  python -m pip install "flashinfer-python"
fi

python -m pip check
