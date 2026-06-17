#!/bin/bash
# Slurm job for Qwen3-8B GRPO RLVR on the MATH train split.
#
# Submit from the repository root with:
#   sbatch examples/grpo_trainer/submit_qwen3_8b_math_h100.slurm
#
# Cluster-specific fields:
# - This version is tuned for the ACD Slurm cluster shown by `sinfo`.
# - If your allocation/account is required by Slurm, uncomment and set --account.

#SBATCH --job-name=verl-grpo-qwen3-math
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
##SBATCH --account=your-account

set -xeuo pipefail

########################### cluster environment ###########################

# Submit this script from the VERL repo root, or set VERL_WORKDIR explicitly:
#   sbatch --export=ALL,VERL_WORKDIR=/path/to/VERL examples/grpo_trainer/submit_qwen3_8b_math_h100.slurm
VERL_WORKDIR=${VERL_WORKDIR:-${SLURM_SUBMIT_DIR}}

# ACD modules.
module purge
module load cuda/12.8

# Activate the known working conda environment. Override CONDA_ENV_PATH when
# submitting if you move/recreate the environment.
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
mkdir -p logs

echo "Python executable: $(which python || true)"
echo "Conda prefix: ${CONDA_PREFIX:-unset}"
python - <<'PY'
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
PY

########################### data and model ###########################

MODEL_PATH=${MODEL_PATH:-/data/user/zhongal/models/Qwen3-8B}
MATH_TRAIN_FILE=${MATH_TRAIN_FILE:-$HOME/data/math/train.parquet}
MATH_TEST_FILE=${MATH_TEST_FILE:-$HOME/data/math/test.parquet}

if [ ! -f "${MATH_TRAIN_FILE}" ] || [ ! -f "${MATH_TEST_FILE}" ]; then
    echo "MATH parquet files not found." >&2
    echo "Expected:" >&2
    echo "  ${MATH_TRAIN_FILE}" >&2
    echo "  ${MATH_TEST_FILE}" >&2
    echo "Prepare them with:" >&2
    echo "  python examples/data_preprocess/math_dataset.py --local_save_dir ~/data/math" >&2
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Local model directory not found: ${MODEL_PATH}" >&2
    echo "Compute nodes have no network, so MODEL_PATH must point to a local directory." >&2
    echo "Download on a network-enabled node, for example:" >&2
    echo "  huggingface-cli download Qwen/Qwen3-8B --local-dir /data/user/zhongal/models/Qwen3-8B" >&2
    exit 1
fi

########################### training knobs ###########################

# Conservative 8xH100 smoke-test defaults. Increase after the first run is stable.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
ROLLOUT_N=${ROLLOUT_N:-5}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_math_h100_${SLURM_JOB_ID}}

########################### runtime environment ###########################

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

# Compute nodes have no network on ACD, so all models/tokenizers must already
# exist on the shared filesystem.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

mkdir -p "${WANDB_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Working directory: ${VERL_WORKDIR}"
echo "Model: ${MODEL_PATH}"
echo "Train file: ${MATH_TRAIN_FILE}"
echo "Val file: ${MATH_TEST_FILE}"

########################### launch ###########################

MODEL_PATH="${MODEL_PATH}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" \
ROLLOUT_N="${ROLLOUT_N}" \
ROLLOUT_TP="${ROLLOUT_TP}" \
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
TOTAL_EPOCHS="${TOTAL_EPOCHS}" \
SAVE_FREQ="${SAVE_FREQ}" \
TEST_FREQ="${TEST_FREQ}" \
PROJECT_NAME="${PROJECT_NAME}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
NNODES=1 \
NGPUS_PER_NODE=8 \
DEVICE=gpu \
INFER_BACKEND=vllm \
bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh \
    data.train_files="['${MATH_TRAIN_FILE}']" \
    data.val_files="['${MATH_TEST_FILE}']" \
    trainer.val_before_train=True \
    "$@"
