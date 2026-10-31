#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Create the base environment and install the binary/runtime dependencies.
conda env create -f "${script_dir}/environment.yml"

# Make conda activation available in this non-interactive shell.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate searchdiff

# Build the CUDA extensions only after the matching PyTorch build is installed.
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/searchdiff-build.XXXXXX")"
cleanup() {
  rm -rf -- "${build_dir}"
}
trap cleanup EXIT

git clone --branch v1.5.0 --depth 1 \
  https://github.com/Dao-AILab/causal-conv1d.git \
  "${build_dir}/causal-conv1d"
MAX_JOBS=2 CAUSAL_CONV1D_FORCE_BUILD=TRUE CUDA_HOME="${CONDA_PREFIX}" \
  python -m pip install "${build_dir}/causal-conv1d" \
  --no-build-isolation --no-cache-dir

git clone --branch v1.2.0.post1 --depth 1 \
  https://github.com/state-spaces/mamba.git \
  "${build_dir}/mamba"
MAX_JOBS=2 MAMBA_FORCE_BUILD=TRUE CUDA_HOME="${CONDA_PREFIX}" \
  python -m pip install "${build_dir}/mamba" \
  --no-build-isolation --no-cache-dir

python -m pip check
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

echo "Environment created. Activate it with: conda activate searchdiff"
