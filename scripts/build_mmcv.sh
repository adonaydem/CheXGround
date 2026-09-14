#!/usr/bin/env bash
set -euo pipefail

: "${CONDA_PREFIX:?Activate the chexground conda environment first.}"
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CPATH="$(python - <<'PY'
from pathlib import Path
import sysconfig

headers = (Path(sysconfig.get_paths()["purelib"]) / "nvidia").glob("*/include")
print(":".join(str(path) for path in sorted(headers)))
PY
)${CPATH:+:$CPATH}"

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "No CUDA device is visible. Set TORCH_CUDA_ARCH_LIST to the target "
        "CUDA architecture before building."
    )
PY
fi

cd -- "$REPO_ROOT/mmcv"
FORCE_CUDA=1 MMCV_WITH_OPS=1 MAX_JOBS="${MAX_JOBS:-4}" \
  python setup.py build_ext --inplace
