#!/bin/bash
# Run offline CoT verification with a local DeepSeek-R1-0528-Qwen3-8B judge.
#
# Submit from the repository root:
#   sbatch --export=ALL,INPUT_DIR=/path/to/trajectory_dir,JUDGE_MODEL=/path/to/judge \
#     examples/generation/submit_cot_deepseek_judge_h100.slurm

#SBATCH --job-name=cot-ds-judge
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
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

INPUT_DIR=${INPUT_DIR:?Set INPUT_DIR to a trajectory output directory.}
JUDGE_MODEL=${JUDGE_MODEL:?Set JUDGE_MODEL to local DeepSeek-R1-0528-Qwen3-8B path.}
OUTPUT_DIR=${OUTPUT_DIR:-${INPUT_DIR}/cot_judge_ds8b_${SLURM_JOB_ID}}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
DTYPE=${DTYPE:-bfloat16}
JUDGE_ATTEMPTS=${JUDGE_ATTEMPTS:-3}
JUDGE_TEMPERATURE=${JUDGE_TEMPERATURE:-0.6}
JUDGE_TOP_P=${JUDGE_TOP_P:-0.95}
MAX_JUDGE_TOKENS=${MAX_JUDGE_TOKENS:-16384}
JUDGE_BATCH_SIZE=${JUDGE_BATCH_SIZE:-32}
COT_LABEL_AGGREGATION=${COT_LABEL_AGGREGATION:-majority}
JUDGE_SCOPE=${JUDGE_SCOPE:-answer-correct}
SEED=${SEED:-42}
TOP_K=${TOP_K:-1,2,4,8,16,32,64,128,256,512,1024}
LIMIT_PROBLEMS=${LIMIT_PROBLEMS:-}
LIMIT_ROWS=${LIMIT_ROWS:-}

if [ ! -d "${INPUT_DIR}" ]; then
    echo "Input directory not found: ${INPUT_DIR}" >&2
    exit 1
fi
if [ ! -d "${JUDGE_MODEL}" ]; then
    echo "Judge model directory not found: ${JUDGE_MODEL}" >&2
    exit 1
fi

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TENSOR_PARALLEL_SIZE - 1)))
else
    export CUDA_VISIBLE_DEVICES
fi
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
echo "Input dir: ${INPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Judge model: ${JUDGE_MODEL}"
echo "Judge attempts: ${JUDGE_ATTEMPTS}"
echo "Judge scope: ${JUDGE_SCOPE}"
echo "Limit problems: ${LIMIT_PROBLEMS:-none}"
echo "Limit rows: ${LIMIT_ROWS:-none}"

ARGS=(
    --input-dir "${INPUT_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --judge-model "${JUDGE_MODEL}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --dtype "${DTYPE}"
    --seed "${SEED}"
    --judge-attempts "${JUDGE_ATTEMPTS}"
    --judge-temperature "${JUDGE_TEMPERATURE}"
    --judge-top-p "${JUDGE_TOP_P}"
    --max-judge-tokens "${MAX_JUDGE_TOKENS}"
    --judge-batch-size "${JUDGE_BATCH_SIZE}"
    --cot-label-aggregation "${COT_LABEL_AGGREGATION}"
    --judge-scope "${JUDGE_SCOPE}"
    --top-k "${TOP_K}"
    --enforce-eager
)
if [ -n "${LIMIT_PROBLEMS}" ]; then
    ARGS+=(--limit-problems "${LIMIT_PROBLEMS}")
fi
if [ -n "${LIMIT_ROWS}" ]; then
    ARGS+=(--limit-rows "${LIMIT_ROWS}")
fi

python examples/generation/verify_cot_deepseek_judge.py "${ARGS[@]}"
