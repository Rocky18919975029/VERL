#!/bin/bash
# Build flash-attn from a local source archive on an ACD H100 compute node.
#
# Compute nodes have no network, so first download the source archive on a
# network-enabled login node:
#   mkdir -p /data/user/zhongal/packages
#   pip download --no-binary flash-attn --no-deps -d /data/user/zhongal/packages flash-attn==2.8.3.post1
#
# Then submit:
#   sbatch examples/grpo_trainer/build_flash_attn_h100.slurm

#SBATCH --job-name=build-flash-attn
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
##SBATCH --account=your-account

set -xeuo pipefail

VERL_WORKDIR=${VERL_WORKDIR:-${SLURM_SUBMIT_DIR}}
CONDA_SH=${CONDA_SH:-/share/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV_PATH=${CONDA_ENV_PATH:-/data/user/zhongal/.conda/envs/verl}
FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION:-2.8.3.post1}
FLASH_ATTN_SOURCE=${FLASH_ATTN_SOURCE:-/data/user/zhongal/packages/flash_attn-${FLASH_ATTN_VERSION}.tar.gz}

# Avoid inheriting a glibc LD_LIBRARY_PATH from the submit shell.
LD_LIBRARY_PATH=":${LD_LIBRARY_PATH:-}:"
LD_LIBRARY_PATH=${LD_LIBRARY_PATH//:\/data\/apps\/glibc\/2.32\/lib:/:}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH//:\/data\/apps\/glibc\/2.32\/lib64:/:}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH#:}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH%:}
export LD_LIBRARY_PATH

module purge
module load gcc/13.3
module load cuda/12.8
module list

if [ -f "${CONDA_SH}" ]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
    conda deactivate || true
    conda activate "${CONDA_ENV_PATH}"
    hash -r
else
    echo "Conda init script not found: ${CONDA_SH}" >&2
    exit 1
fi

cd "${VERL_WORKDIR}"

if [ ! -f "${FLASH_ATTN_SOURCE}" ]; then
    echo "flash-attn source archive not found: ${FLASH_ATTN_SOURCE}" >&2
    echo "Download it on the login node first:" >&2
    echo "  mkdir -p /data/user/zhongal/packages" >&2
    echo "  pip download --no-binary flash-attn --no-deps -d /data/user/zhongal/packages flash-attn==${FLASH_ATTN_VERSION}" >&2
    exit 1
fi

export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$(which g++)
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS:-90}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export MAX_JOBS=${MAX_JOBS:-4}
export NVCC_THREADS=${NVCC_THREADS:-1}
export TMPDIR=${TMPDIR:-/data/user/zhongal/tmp}
mkdir -p "${TMPDIR}"

python - <<'PY'
import os
import sys

import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("FLASH_ATTN_CUDA_ARCHS:", os.environ.get("FLASH_ATTN_CUDA_ARCHS"))
PY

pip uninstall -y flash-attn flash_attn || true
pip install --no-build-isolation --no-cache-dir --no-index "${FLASH_ATTN_SOURCE}"

python - <<'PY'
import importlib.util
import subprocess

spec = importlib.util.find_spec("flash_attn_2_cuda")
print("flash_attn_2_cuda:", None if spec is None else spec.origin)
if spec is not None and spec.origin is not None:
    subprocess.run(["ldd", spec.origin], check=False)
import flash_attn
print("flash_attn ok", flash_attn.__version__)
PY
