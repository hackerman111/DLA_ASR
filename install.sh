#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
ENV_NAME="${ENV_NAME:-project_env}"

cd "$(dirname "$0")"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"

    if ! conda env list --json | grep -q "\"$ENV_NAME\""; then
        conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
    fi

    conda activate "$ENV_NAME"
    PYTHON="$(command -v python)"
else
    PYTHON="$(command -v python3)"
fi

"$PYTHON" -m pip install -r requirements.txt

if [[ -d .git && "${INSTALL_HOOKS:-0}" == "1" ]]; then
    "$PYTHON" -m pre_commit install
fi

"$PYTHON" - <<'PY'
import sys
print("Python:", sys.version)
print("Executable:", sys.executable)

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
except ImportError:
    print("PyTorch not installed")
PY

