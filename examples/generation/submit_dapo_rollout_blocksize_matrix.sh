#!/bin/bash
# Submit DAPO-MATH rollout jobs for a block-size/seed matrix.
#
# The matrix contains:
# - tree rollouts for every SEEDS x BLOCK_SIZES setting
# - one full-trajectory rollout baseline for every seed
#
# Example:
#   SEEDS="42 43 44" BLOCK_SIZES="64 128 192" \
#     bash examples/generation/submit_dapo_rollout_blocksize_matrix.sh
#
# To add more tree-only block-size points to an existing OUTPUT_ROOT:
#   RUN_FULL=0 OUTPUT_ROOT=/path/to/matrix BLOCK_SIZES="32 96 160" \
#     bash examples/generation/submit_dapo_rollout_blocksize_matrix.sh

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/data/user/zhongal/.cache/qwen2.5-math-7b-local}
DATA_FILE=${DATA_FILE:-/data/user/zhongal/data/reschedule/DAPO-Math-17k.parquet}
DATASET_NAME=${DATASET_NAME:-dapo_math_17k}

SEEDS=(${SEEDS:-42})
BLOCK_SIZES=(${BLOCK_SIZES:-64 128 192})
LEADER_TEMPERATURE=${LEADER_TEMPERATURE:-1.0}
FOLLOWER_TEMPERATURE=${FOLLOWER_TEMPERATURE:-0.25}
TOP_P=${TOP_P:-1.0}
LIMIT=${LIMIT:-1000}
NUM_PREFIXES=${NUM_PREFIXES:-4}
NUM_SUFFIXES=${NUM_SUFFIXES:-2}
N_RESPONSES=${N_RESPONSES:-$((NUM_PREFIXES * NUM_SUFFIXES))}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
RUN_TREE=${RUN_TREE:-1}
RUN_FULL=${RUN_FULL:-1}

NUM_SHARDS=${NUM_SHARDS:-8}
GPUS_PER_SHARD=${GPUS_PER_SHARD:-1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
DTYPE=${DTYPE:-bfloat16}
GPUS_PER_JOB=$((NUM_SHARDS * GPUS_PER_SHARD))
CPUS_PER_JOB=${CPUS_PER_JOB:-$((GPUS_PER_JOB * 12))}

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/zhongal/VERL/outputs/dapo_rollout_blocksize_matrix_${STAMP}}
MANIFEST=${OUTPUT_ROOT}/submitted_jobs.tsv

temp_tag=${LEADER_TEMPERATURE/./p}

echo "Submitting DAPO rollout block-size matrix"
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_FILE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Seeds: ${SEEDS[*]}"
echo "Block sizes: ${BLOCK_SIZES[*]}"
echo "Leader/full temperature: ${LEADER_TEMPERATURE}"
echo "Follower temperature: ${FOLLOWER_TEMPERATURE}"
echo "Top-p: ${TOP_P}"
echo "Limit: ${LIMIT}"
echo "Tree prefixes x suffixes: ${NUM_PREFIXES} x ${NUM_SUFFIXES}"
echo "Full responses per problem: ${N_RESPONSES}"
echo "Run tree jobs: ${RUN_TREE}"
echo "Run full jobs: ${RUN_FULL}"
echo "GPUs per job: ${GPUS_PER_JOB}"
echo "CPUs per job: ${CPUS_PER_JOB}"

mkdir -p "${OUTPUT_ROOT}"
if [ ! -f "${MANIFEST}" ]; then
    printf "kind\tseed\tblock_size\ttemperature\tjob_id\toutput_dir\n" > "${MANIFEST}"
fi

for seed in "${SEEDS[@]}"; do
    if [ "${RUN_TREE}" = "1" ]; then
        for block_size in "${BLOCK_SIZES[@]}"; do
            output_dir="${OUTPUT_ROOT}/tree/seed_${seed}/block_${block_size}/leader_temp_${temp_tag}"
            job_id=$(
                sbatch --parsable \
                    --gres="gpu:${GPUS_PER_JOB}" \
                    --cpus-per-task="${CPUS_PER_JOB}" \
                    --export=ALL,MODEL_PATH="${MODEL_PATH}",DATA_FILE="${DATA_FILE}",DATASET_NAME="${DATASET_NAME}",ROUND_INDEX=1,PROGRESSIVE_BLOCK_SIZE="${block_size}",MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}",NUM_PREFIXES="${NUM_PREFIXES}",NUM_SUFFIXES="${NUM_SUFFIXES}",PREFIX_TEMPERATURE="${LEADER_TEMPERATURE}",PREFIX_TOP_P="${TOP_P}",SUFFIX_TEMPERATURE="${FOLLOWER_TEMPERATURE}",SUFFIX_TOP_P="${TOP_P}",LIMIT="${LIMIT}",SEED="${seed}",NUM_SHARDS="${NUM_SHARDS}",GPUS_PER_SHARD="${GPUS_PER_SHARD}",TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}",GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}",DTYPE="${DTYPE}",OUTPUT_DIR="${output_dir}" \
                    examples/generation/submit_hpf_progressive_dapo_rollout_h100.slurm
            )
            printf "tree\t%s\t%s\t%s\t%s\t%s\n" "${seed}" "${block_size}" "${LEADER_TEMPERATURE}" "${job_id}" "${output_dir}" | tee -a "${MANIFEST}"
        done
    fi

    if [ "${RUN_FULL}" = "1" ]; then
        output_dir="${OUTPUT_ROOT}/full/seed_${seed}/temp_${temp_tag}"
        job_id=$(
            sbatch --parsable \
                --gres="gpu:${GPUS_PER_JOB}" \
                --cpus-per-task="${CPUS_PER_JOB}" \
                --export=ALL,MODEL_PATH="${MODEL_PATH}",DATA_FILE="${DATA_FILE}",DATASET_NAME="${DATASET_NAME}_full",TEMPERATURE="${LEADER_TEMPERATURE}",TOP_P="${TOP_P}",N_RESPONSES="${N_RESPONSES}",MAX_TOKENS="${MAX_RESPONSE_LENGTH}",LIMIT="${LIMIT}",SEED="${seed}",NUM_SHARDS="${NUM_SHARDS}",GPUS_PER_SHARD="${GPUS_PER_SHARD}",TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}",GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}",DTYPE="${DTYPE}",SKIP_LOGLIK_SCORING=1,OUTPUT_DIR="${output_dir}" \
                examples/generation/submit_math500_qwen25_7b_sample_score_h100.slurm
        )
        printf "full\t%s\tNA\t%s\t%s\t%s\n" "${seed}" "${LEADER_TEMPERATURE}" "${job_id}" "${output_dir}" | tee -a "${MANIFEST}"
    fi
done

echo "Submitted job manifest: ${MANIFEST}"
