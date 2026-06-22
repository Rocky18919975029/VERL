#!/bin/bash
# Generate 16 responses per Math-500 problem with Qwen2.5-7B at temp=0.25,
# then compute response-length-normalized model log likelihoods.
#
# Submit from the repository root:
#   sbatch examples/generation/submit_math500_qwen25_7b_sample_score_h100.slurm

#SBATCH --job-name=math500-qwen25-score
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

VERL_WORKDIR=${VERL_WORKDIR:-${SLURM_SUBMIT_DIR}}

# Avoid globally preloading alternate glibc into conda/Python.
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

MODEL_PATH=${MODEL_PATH:-/data/user/zhongal/.cache/qwen2.5-7b-local}
MATH500_FILE=${MATH500_FILE:-/data/user/zhongal/data/reschedule/math500.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-${VERL_WORKDIR}/outputs/math500_qwen25_7b_temp025_n16_${SLURM_JOB_ID}}

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Local model directory not found: ${MODEL_PATH}" >&2
    exit 1
fi

if [ ! -f "${MATH500_FILE}" ]; then
    echo "Math-500 parquet not found: ${MATH500_FILE}" >&2
    exit 1
fi

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

mkdir -p "${OUTPUT_DIR}"

python - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("No CUDA device visible inside the Slurm allocation.")
PY

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Working directory: ${VERL_WORKDIR}"
echo "Model: ${MODEL_PATH}"
echo "Math-500 file: ${MATH500_FILE}"
echo "Output dir: ${OUTPUT_DIR}"

ARGS=(
    --model "${MODEL_PATH}"
    --data "${MATH500_FILE}"
    --output-dir "${OUTPUT_DIR}"
    --temperature "${TEMPERATURE:-0.25}"
    --top-p "${TOP_P:-1.0}"
    --n "${N_RESPONSES:-16}"
    --max-tokens "${MAX_TOKENS:-3072}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-8}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}"
    --dtype "${DTYPE:-bfloat16}"
    --seed "${SEED:-42}"
    --score-batch-size "${SCORE_BATCH_SIZE:-64}"
)
if [ -n "${LIMIT:-}" ]; then
    ARGS+=(--limit "${LIMIT}")
fi

python examples/generation/math500_qwen25_7b_sample_and_score.py "${ARGS[@]}"
