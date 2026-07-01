#!/bin/bash
# Run offline CoT verification with multiple independent vLLM judge shards.
#
# Submit from the repository root:
#   sbatch --export=ALL,INPUT_DIR=/path/to/trajectory_dir,JUDGE_MODEL=/path/to/judge \
#     examples/generation/submit_cot_deepseek_judge_sharded_h100.slurm

#SBATCH --job-name=cot-ds-shard
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

INPUT_DIR=${INPUT_DIR:?Set INPUT_DIR to a trajectory output directory.}
JUDGE_MODEL=${JUDGE_MODEL:?Set JUDGE_MODEL to local DeepSeek-R1-0528-Qwen3-8B path.}
OUTPUT_DIR=${OUTPUT_DIR:-${INPUT_DIR}/cot_judge_ds8b_sharded_${SLURM_JOB_ID}}
NUM_SHARDS=${NUM_SHARDS:-8}
GPUS_PER_SHARD=${GPUS_PER_SHARD:-1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-${GPUS_PER_SHARD}}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
DTYPE=${DTYPE:-bfloat16}
JUDGE_ATTEMPTS=${JUDGE_ATTEMPTS:-3}
JUDGE_TEMPERATURE=${JUDGE_TEMPERATURE:-0.6}
JUDGE_TOP_P=${JUDGE_TOP_P:-0.95}
MAX_JUDGE_TOKENS=${MAX_JUDGE_TOKENS:-16384}
JUDGE_BATCH_SIZE=${JUDGE_BATCH_SIZE:-16}
JUDGE_SCOPE=${JUDGE_SCOPE:-answer-correct}
SEED=${SEED:-42}
TOP_K=${TOP_K:-1,2,4,8,16,32,64,128,256,512,1024}
LIMIT_PROBLEMS=${LIMIT_PROBLEMS:-}
LIMIT_ROWS=${LIMIT_ROWS:-}

if [ "${TENSOR_PARALLEL_SIZE}" -ne "${GPUS_PER_SHARD}" ]; then
    echo "TENSOR_PARALLEL_SIZE must match GPUS_PER_SHARD for this launcher." >&2
    echo "Got TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}, GPUS_PER_SHARD=${GPUS_PER_SHARD}" >&2
    exit 1
fi
if [ $((NUM_SHARDS * GPUS_PER_SHARD)) -gt 8 ]; then
    echo "Requested more than 8 GPUs on one node: NUM_SHARDS=${NUM_SHARDS}, GPUS_PER_SHARD=${GPUS_PER_SHARD}" >&2
    exit 1
fi
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
echo "Num shards: ${NUM_SHARDS}"
echo "GPUs per shard: ${GPUS_PER_SHARD}"
echo "Tensor parallel size per shard: ${TENSOR_PARALLEL_SIZE}"
echo "Judge attempts: ${JUDGE_ATTEMPTS}"
echo "Judge scope: ${JUDGE_SCOPE}"
echo "Limit problems: ${LIMIT_PROBLEMS:-none}"
echo "Limit rows: ${LIMIT_ROWS:-none}"

COMMON_ARGS=(
    --input-dir "${INPUT_DIR}"
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
    --judge-scope "${JUDGE_SCOPE}"
    --top-k "${TOP_K}"
    --num-shards "${NUM_SHARDS}"
    --enforce-eager
)
if [ -n "${LIMIT_PROBLEMS}" ]; then
    COMMON_ARGS+=(--limit-problems "${LIMIT_PROBLEMS}")
fi
if [ -n "${LIMIT_ROWS}" ]; then
    COMMON_ARGS+=(--limit-rows "${LIMIT_ROWS}")
fi

pids=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    first_gpu=$((shard * GPUS_PER_SHARD))
    last_gpu=$((first_gpu + GPUS_PER_SHARD - 1))
    visible_devices=$(seq -s, "${first_gpu}" "${last_gpu}")
    shard_output_dir="${OUTPUT_DIR}/shard_${shard}"
    shard_cache_dir="${OUTPUT_DIR}/cache/shard_${shard}"
    mkdir -p "${shard_output_dir}" "${shard_cache_dir}/vllm" "${shard_cache_dir}/torchinductor" "${shard_cache_dir}/triton"
    echo "Launching judge shard ${shard}/${NUM_SHARDS} on CUDA_VISIBLE_DEVICES=${visible_devices}"
    CUDA_VISIBLE_DEVICES="${visible_devices}" \
        XDG_CACHE_HOME="${shard_cache_dir}" \
        VLLM_CACHE_ROOT="${shard_cache_dir}/vllm" \
        TORCHINDUCTOR_CACHE_DIR="${shard_cache_dir}/torchinductor" \
        TRITON_CACHE_DIR="${shard_cache_dir}/triton" \
        python examples/generation/verify_cot_deepseek_judge.py \
            "${COMMON_ARGS[@]}" \
            --shard-index "${shard}" \
            --output-dir "${shard_output_dir}" &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "${pid}"
done

python examples/generation/aggregate_cot_judge_results.py \
    --input-dir "${OUTPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}/aggregate" \
    --top-k "${TOP_K}"

echo "All judge shards completed. Aggregate outputs are under ${OUTPUT_DIR}/aggregate"

