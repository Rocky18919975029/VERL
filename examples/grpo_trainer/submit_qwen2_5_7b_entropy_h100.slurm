#!/bin/bash
# Slurm smoke test for the Entropy paper's Qwen2.5-7B GRPO/KL-Cov setup.
#
# Submit from the repository root with:
#   sbatch examples/grpo_trainer/submit_qwen2_5_7b_entropy_h100.slurm
#
# Common overrides:
#   sbatch --export=ALL,METHOD=baseline examples/grpo_trainer/submit_qwen2_5_7b_entropy_h100.slurm
#   sbatch --export=ALL,ENTROPY_REPRO_FULL=1,METHOD=kl_cov examples/grpo_trainer/submit_qwen2_5_7b_entropy_h100.slurm

#SBATCH --job-name=verl-entropy-7b
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
##SBATCH --account=your-account

set -xeuo pipefail

VERL_WORKDIR=${VERL_WORKDIR:-${SLURM_SUBMIT_DIR}}

# Do not run conda/Python by globally overriding glibc with LD_LIBRARY_PATH.
# The ACD glibc module only adjusts PATH; adding its lib directory can segfault
# mixed conda/system processes. Build flash-attn against the host glibc instead.
LD_LIBRARY_PATH=":${LD_LIBRARY_PATH:-}:"
LD_LIBRARY_PATH=${LD_LIBRARY_PATH//:\/data\/apps\/glibc\/2.32\/lib:/:}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH//:\/data\/apps\/glibc\/2.32\/lib64:/:}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH#:}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH%:}
export LD_LIBRARY_PATH

module purge
module load cuda/12.8
module list

echo "LD_LIBRARY_PATH after module load: ${LD_LIBRARY_PATH:-unset}"
ldd --version | head -n 1 || true

CONDA_SH=${CONDA_SH:-/share/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV_PATH=${CONDA_ENV_PATH:-/data/user/zhongal/.conda/envs/verl}
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

python - <<'PY'
import importlib.util
import os
import subprocess
import sys

print("python:", sys.version)
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "unset"))
try:
    import torch
except Exception as exc:
    raise SystemExit(f"failed to import torch: {exc}")

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
spec = importlib.util.find_spec("flash_attn_2_cuda")
print("flash_attn_2_cuda:", None if spec is None else spec.origin)
if spec is not None and spec.origin is not None:
    subprocess.run(["ldd", spec.origin], check=False)
try:
    import flash_attn
except Exception as exc:
    raise SystemExit(f"failed to import flash_attn: {exc}")
print("flash_attn import ok")
PY

METHOD=${METHOD:-kl_cov}
MODEL_PATH=${MODEL_PATH:-/data/user/zhongal/models/Qwen2.5-7B}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/entropy/dapo-math-17k.parquet}
VAL_FILES=${VAL_FILES:-"['$HOME/data/entropy/aime-2024.parquet']"}

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Local model directory not found: ${MODEL_PATH}" >&2
    echo "Compute nodes have no network; download Qwen/Qwen2.5-7B to this path first." >&2
    exit 1
fi

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Training parquet not found: ${TRAIN_FILE}" >&2
    echo "Download/prepare DAPO-MATH parquet first. The paper repo points to a Google Drive data folder." >&2
    exit 1
fi

for val_file in $(python - <<PY
import ast
for path in ast.literal_eval("""${VAL_FILES}"""):
    print(path)
PY
); do
    if [ ! -f "${val_file}" ]; then
        echo "Validation parquet not found: ${val_file}" >&2
        exit 1
    fi
done

# ACD is an NVIDIA GPU cluster, but some images/modules may leave ROCm/HIP
# visibility variables in the environment. verl refuses to start workers when
# ROCR_VISIBLE_DEVICES coexists with CUDA_VISIBLE_DEVICES.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# Let Slurm set CUDA_VISIBLE_DEVICES for the allocated GPUs. If Slurm did not
# set it, fall back to all eight GPUs on this node.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-${VERL_WORKDIR}/wandb}
export WANDB_SILENT=${WANDB_SILENT:-true}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
mkdir -p "${WANDB_DIR}"

PROJECT_NAME=${PROJECT_NAME:-entropy_qwen2_5_7b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_7b_${METHOD}_smoke_${SLURM_JOB_ID}}

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Working directory: ${VERL_WORKDIR}"
echo "Method: ${METHOD}"
echo "Model: ${MODEL_PATH}"
echo "Train file: ${TRAIN_FILE}"
echo "Val files: ${VAL_FILES}"

METHOD="${METHOD}" \
MODEL_PATH="${MODEL_PATH}" \
TRAIN_FILE="${TRAIN_FILE}" \
VAL_FILES="${VAL_FILES}" \
PROJECT_NAME="${PROJECT_NAME}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
NNODES=1 \
NGPUS_PER_NODE=8 \
bash examples/grpo_trainer/run_qwen2_5_7b_entropy_fsdp.sh "$@"
