#!/bin/bash
set -euo pipefail

# Lightweight setup for the current server environment.
# It uses the active Python/pip config, does not create a venv, and does not
# decompress documents, build indexes, or start the search service.

python -m pip install -r requirements.txt

if [ "${SHOPPINGBENCH_INSTALL_RL_LIGHT:-0}" = "1" ]; then
  python -m pip install \
    accelerate codetiming datasets dill hydra-core pyarrow pybind11 \
    "ray[default]" tensordict wandb peft torchdata
fi

echo "Lightweight environment setup complete."
echo "Skipped documents, index build, and search service startup."
