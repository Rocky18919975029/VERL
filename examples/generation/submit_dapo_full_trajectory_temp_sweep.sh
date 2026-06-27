#!/bin/bash
# Submit full-response DAPO-MATH trajectory generation jobs for a temperature sweep.
#
# This does not split responses into prefix/suffix and does not train. Each job
# samples complete responses at one temperature, verifies answer correctness, and
# saves the trajectories.
#
# Usage from the repository root:
#   bash examples/generation/submit_dapo_full_trajectory_temp_sweep.sh

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/data/user/zhongal/.cache/qwen2.5-math-7b-local}
DATA_FILE=${DATA_FILE:-/data/user/zhongal/data/reschedule/DAPO-Math-17k.parquet}
DATASET_NAME=${DATASET_NAME:-dapo_math_17k_full}

LIMIT=${LIMIT:-1000}
N_RESPONSES=${N_RESPONSES:-8}
MAX_TOKENS=${MAX_TOKENS:-3072}
TOP_P=${TOP_P:-1.0}
SEED=${SEED:-42}

NUM_SHARDS=${NUM_SHARDS:-4}
GPUS_PER_SHARD=${GPUS_PER_SHARD:-1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
DTYPE=${DTYPE:-bfloat16}
GPUS_PER_JOB=$((NUM_SHARDS * GPUS_PER_SHARD))
CPUS_PER_JOB=${CPUS_PER_JOB:-$((GPUS_PER_JOB * 12))}

SWEEP_NAME=${SWEEP_NAME:-dapo_full_trajectory_temp_sweep_$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/zhongal/VERL/outputs/${SWEEP_NAME}}
TEMPS=(${TEMPS:-1.2 1.0 0.8 0.6})

echo "Submitting DAPO full-trajectory temperature sweep"
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_FILE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Temps: ${TEMPS[*]}"
echo "Limit: ${LIMIT}"
echo "Responses per problem: ${N_RESPONSES}"
echo "Top-p: ${TOP_P}"
echo "GPUs per job: ${GPUS_PER_JOB}"
echo "CPUs per job: ${CPUS_PER_JOB}"

mkdir -p "${OUTPUT_ROOT}"

for temp in "${TEMPS[@]}"; do
    temp_tag=${temp/./p}
    output_dir="${OUTPUT_ROOT}/temp_${temp_tag}"
    job_id=$(
        sbatch --parsable \
            --gres="gpu:${GPUS_PER_JOB}" \
            --cpus-per-task="${CPUS_PER_JOB}" \
            --export=ALL,MODEL_PATH="${MODEL_PATH}",DATA_FILE="${DATA_FILE}",DATASET_NAME="${DATASET_NAME}",TEMPERATURE="${temp}",TOP_P="${TOP_P}",N_RESPONSES="${N_RESPONSES}",MAX_TOKENS="${MAX_TOKENS}",LIMIT="${LIMIT}",SEED="${SEED}",NUM_SHARDS="${NUM_SHARDS}",GPUS_PER_SHARD="${GPUS_PER_SHARD}",TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}",GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}",DTYPE="${DTYPE}",SKIP_LOGLIK_SCORING=1,OUTPUT_DIR="${output_dir}" \
            examples/generation/submit_math500_qwen25_7b_sample_score_h100.slurm
    )
    echo "temperature=${temp} job_id=${job_id} output_dir=${output_dir}"
done

