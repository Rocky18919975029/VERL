#!/bin/bash
# Submit HPF tree-rollout-only jobs for a leader-temperature sweep.
#
# Usage from the repository root:
#   bash examples/generation/submit_hpf_dapo_leader_temp_sweep.sh
#
# The jobs do not train. Each job samples one round of prefix/suffix tree
# rollouts and saves trajectory parquet/jsonl files for later analysis.

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/data/user/zhongal/.cache/qwen2.5-math-7b-local}
DATA_FILE=${DATA_FILE:-/data/user/zhongal/data/reschedule/DAPO-Math-17k.parquet}
DATASET_NAME=${DATASET_NAME:-dapo_math_17k}

ROUND_INDEX=${ROUND_INDEX:-1}
PROGRESSIVE_BLOCK_SIZE=${PROGRESSIVE_BLOCK_SIZE:-192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
NUM_PREFIXES=${NUM_PREFIXES:-4}
NUM_SUFFIXES=${NUM_SUFFIXES:-2}
SUFFIX_TEMPERATURE=${SUFFIX_TEMPERATURE:-0.25}
SUFFIX_TOP_P=${SUFFIX_TOP_P:-1.0}
PREFIX_TOP_P=${PREFIX_TOP_P:-1.0}
LIMIT=${LIMIT:-1000}
SEED=${SEED:-42}

NUM_SHARDS=${NUM_SHARDS:-8}
GPUS_PER_SHARD=${GPUS_PER_SHARD:-1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
DTYPE=${DTYPE:-bfloat16}

SWEEP_NAME=${SWEEP_NAME:-hpf_dapo_leader_temp_sweep_$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/zhongal/VERL/outputs/${SWEEP_NAME}}
TEMPS=(${LEADER_TEMPS:-1.2 1.0 0.8 0.6})

echo "Submitting HPF leader-temperature rollout sweep"
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_FILE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Leader temps: ${TEMPS[*]}"
echo "Follower temp: ${SUFFIX_TEMPERATURE}"
echo "Limit: ${LIMIT}"
echo "Prefixes x suffixes: ${NUM_PREFIXES} x ${NUM_SUFFIXES}"

mkdir -p "${OUTPUT_ROOT}"

for temp in "${TEMPS[@]}"; do
    temp_tag=${temp/./p}
    output_dir="${OUTPUT_ROOT}/leader_temp_${temp_tag}"
    job_id=$(
        sbatch --parsable \
            --export=ALL,MODEL_PATH="${MODEL_PATH}",DATA_FILE="${DATA_FILE}",DATASET_NAME="${DATASET_NAME}",ROUND_INDEX="${ROUND_INDEX}",PROGRESSIVE_BLOCK_SIZE="${PROGRESSIVE_BLOCK_SIZE}",MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}",NUM_PREFIXES="${NUM_PREFIXES}",NUM_SUFFIXES="${NUM_SUFFIXES}",PREFIX_TEMPERATURE="${temp}",PREFIX_TOP_P="${PREFIX_TOP_P}",SUFFIX_TEMPERATURE="${SUFFIX_TEMPERATURE}",SUFFIX_TOP_P="${SUFFIX_TOP_P}",LIMIT="${LIMIT}",SEED="${SEED}",NUM_SHARDS="${NUM_SHARDS}",GPUS_PER_SHARD="${GPUS_PER_SHARD}",TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}",GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}",DTYPE="${DTYPE}",OUTPUT_DIR="${output_dir}" \
            examples/generation/submit_hpf_progressive_dapo_rollout_h100.slurm
    )
    echo "leader_temp=${temp} job_id=${job_id} output_dir=${output_dir}"
done

