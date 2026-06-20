#!/bin/bash
# Slurm job for reproducing the Re-Schedule paper's GRPO baseline on 8xH100.
#
# Submit from the repository root with:
#   sbatch --export=ALL,MODEL_PATH=/data/user/zhongal/.cache/qwen2.5-math-7b-local,DATA_DIR=/data/user/zhongal/data/reschedule,TOTAL_TRAINING_STEPS=2 examples/grpo_trainer/submit_qwen2_5_math_7b_reschedule_baseline_h100.slurm
#
# If an override value contains commas, export it in the shell first and submit
# with --export=ALL. Slurm splits --export key-value pairs on commas.

#SBATCH --job-name=verl-resched-grpo
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
##SBATCH --account=your-account

set -xeuo pipefail

VERL_WORKDIR=${VERL_WORKDIR:-${SLURM_SUBMIT_DIR}}

# Do not globally preload alternate glibc libraries into conda/Python.
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
import os
import sys

print("python:", sys.version)
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
else:
    raise SystemExit("No CUDA device visible inside the Slurm allocation.")
PY

MODEL_PATH=${MODEL_PATH:-/data/user/zhongal/.cache/qwen2.5-math-7b-local}
DATA_DIR=${DATA_DIR:-/data/user/zhongal/data/reschedule}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/DAPO-Math-17k.parquet}

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Local model directory not found: ${MODEL_PATH}" >&2
    exit 1
fi

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Training parquet not found: ${TRAIN_FILE}" >&2
    exit 1
fi

for val_file in \
    "${DATA_DIR}/aime24.parquet" \
    "${DATA_DIR}/aime25.parquet" \
    "${DATA_DIR}/amc23.parquet" \
    "${DATA_DIR}/math500.parquet" \
    "${DATA_DIR}/minerva_math.parquet" \
    "${DATA_DIR}/olympiadbench.parquet"
do
    if [ ! -f "${val_file}" ]; then
        echo "Validation parquet not found: ${val_file}" >&2
        exit 1
    fi
done

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export RAY_TMP_DIR=${RAY_TMP_DIR:-${VERL_WORKDIR}/ray_tmp/${SLURM_JOB_ID}}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-${VERL_WORKDIR}/wandb}
export WANDB_SILENT=${WANDB_SILENT:-true}
export WANDB_START_METHOD=${WANDB_START_METHOD:-thread}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export CC=${CC:-$(which gcc)}
export CXX=${CXX:-$(which g++)}
export CUDAHOSTCXX=${CUDAHOSTCXX:-$(which g++)}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}

mkdir -p "${WANDB_DIR}" "${RAY_TMP_DIR}"

PROJECT_NAME=${PROJECT_NAME:-grpo_dapo_math17k_reschedule_baseline}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_math_7b_grpo_baseline_${SLURM_JOB_ID}}

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Working directory: ${VERL_WORKDIR}"
echo "Ray tmp dir: ${RAY_TMP_DIR}"
echo "Project: ${PROJECT_NAME}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Data dir: ${DATA_DIR}"
echo "Total training steps: ${TOTAL_TRAINING_STEPS:-150}"

MODEL_PATH="${MODEL_PATH}" \
DATA_DIR="${DATA_DIR}" \
PROJECT_NAME="${PROJECT_NAME}" \
RUN_NAME="${EXPERIMENT_NAME}" \
MODE="${MODE:-train}" \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}" \
NNODES=1 \
NGPUS_PER_NODE=8 \
bash examples/grpo_trainer/run_qwen2_5_math_7b_grpo_reschedule_baseline.sh "$@"
