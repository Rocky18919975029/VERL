#!/bin/bash
# Submit official RPD jobs for selected DAPO rollout settings.
#
# Target matrix by default:
#   - tree rollout, seed=42, leader blocks: 32, 64, 128, 256
#   - full trajectory rollout, seed=42, temperatures: 0.25, 1.0
#
# This script assumes the rollout parquet outputs already exist. It only submits
# official RPD computation jobs.
#
# Example:
#   bash examples/generation/submit_selected_dapo_rpd_jobs.sh
#
# If the full temp=0.25 rollout lives outside the default matrix root, pass it:
#   FULL_TEMP_0P25_DIR=/path/to/temp_0p25 bash examples/generation/submit_selected_dapo_rpd_jobs.sh

set -euo pipefail

SEED=${SEED:-42}
TREE_BLOCK_SIZES=(${TREE_BLOCK_SIZES:-32 64 128 256})
FULL_TEMPS=(${FULL_TEMPS:-0.25 1.0})
RUN_TREE=${RUN_TREE:-1}
RUN_FULL=${RUN_FULL:-1}

ROLLOUT_MATRIX_ROOT=${ROLLOUT_MATRIX_ROOT:-/data/user/zhongal/VERL/outputs/dapo_rollout_blocksize_matrix_20260701_203658}
RPD_REPO=${RPD_REPO:-/data/user/zhongal/external/Reasoning-Path-Divergence}
MODEL_PATH_INSTRUCT=${MODEL_PATH_INSTRUCT:-/data/user/zhongal/.cache/Qwen3-14B}
MODEL_PATH_EMBEDDING=${MODEL_PATH_EMBEDDING:-/data/user/zhongal/.cache/Qwen3-Embedding-8B}
TREE_LEADER_TEMP=${TREE_LEADER_TEMP:-1.0}
TREE_FOLLOWER_TEMP=${TREE_FOLLOWER_TEMP:-0.25}
FULL_TEMP_0P25_DIR=${FULL_TEMP_0P25_DIR:-}
FULL_TEMP_1P0_DIR=${FULL_TEMP_1P0_DIR:-}

TEST_LIMIT=${TEST_LIMIT:-None}
RESPONSES_PER_PROBLEM=${RESPONSES_PER_PROBLEM:-}
MIN_RESPONSES=${MIN_RESPONSES:-2}
MAX_ROWS_PER_FILE=${MAX_ROWS_PER_FILE:-10000}
OVERWRITE=${OVERWRITE:-0}
RPD_NUM_SHARDS=${RPD_NUM_SHARDS:-8}
RPD_GPUS_PER_SHARD=${RPD_GPUS_PER_SHARD:-1}
RPD_GPUS_PER_JOB=${RPD_GPUS_PER_JOB:-$((RPD_NUM_SHARDS * RPD_GPUS_PER_SHARD))}
RPD_CPUS_PER_JOB=${RPD_CPUS_PER_JOB:-96}
RPD_MEM_PER_JOB=${RPD_MEM_PER_JOB:-0}
RPD_OUTPUT_ROOT=${RPD_OUTPUT_ROOT:-outputs/rpd_official_runs/dapo_seed${SEED}_tree32_64_128_256_full025_10_sharded${RPD_NUM_SHARDS}}

MANIFEST=${RPD_OUTPUT_ROOT}/submitted_rpd_jobs.tsv

temp_tag() {
    local value="$1"
    echo "${value/./p}"
}

tree_input_dir() {
    local block="$1"
    local leader_tag
    leader_tag=$(temp_tag "${TREE_LEADER_TEMP}")
    echo "${ROLLOUT_MATRIX_ROOT}/tree/seed_${SEED}/block_${block}/leader_temp_${leader_tag}"
}

full_input_dir() {
    local temp="$1"
    local tag
    tag=$(temp_tag "${temp}")
    case "${tag}" in
        0p25)
            if [ -n "${FULL_TEMP_0P25_DIR}" ]; then
                echo "${FULL_TEMP_0P25_DIR}"
            else
                echo "${ROLLOUT_MATRIX_ROOT}/full/seed_${SEED}/temp_${tag}"
            fi
            ;;
        1p0)
            if [ -n "${FULL_TEMP_1P0_DIR}" ]; then
                echo "${FULL_TEMP_1P0_DIR}"
            else
                echo "${ROLLOUT_MATRIX_ROOT}/full/seed_${SEED}/temp_${tag}"
            fi
            ;;
        *)
            echo "${ROLLOUT_MATRIX_ROOT}/full/seed_${SEED}/temp_${tag}"
            ;;
    esac
}

submit_rpd() {
    local kind="$1"
    local setting="$2"
    local input_dir="$3"
    local run_name="$4"

    if [ ! -d "${input_dir}" ]; then
        echo "ERROR: missing rollout directory for ${setting}: ${input_dir}" >&2
        return 1
    fi

    local job_id
    job_id=$(
        sbatch --parsable \
            --gres="gpu:${RPD_GPUS_PER_JOB}" \
            --cpus-per-task="${RPD_CPUS_PER_JOB}" \
            --mem="${RPD_MEM_PER_JOB}" \
            --export=ALL,INPUT_DIR="${input_dir}",RPD_REPO="${RPD_REPO}",MODEL_PATH_INSTRUCT="${MODEL_PATH_INSTRUCT}",MODEL_PATH_EMBEDDING="${MODEL_PATH_EMBEDDING}",OUTPUT_ROOT="${RPD_OUTPUT_ROOT}",RUN_NAME="${run_name}",TEST_LIMIT="${TEST_LIMIT}",RESPONSES_PER_PROBLEM="${RESPONSES_PER_PROBLEM}",MIN_RESPONSES="${MIN_RESPONSES}",MAX_ROWS_PER_FILE="${MAX_ROWS_PER_FILE}",RPD_NUM_SHARDS="${RPD_NUM_SHARDS}",RPD_GPUS_PER_SHARD="${RPD_GPUS_PER_SHARD}",OVERWRITE="${OVERWRITE}" \
            examples/generation/submit_official_rpd_pipeline_sharded_h100.slurm
    )
    local shard
    for shard in $(seq 0 $((RPD_NUM_SHARDS - 1))); do
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${kind}" "${setting}" "${shard}" "${RPD_NUM_SHARDS}" "${job_id}" "${run_name}_shard${shard}of${RPD_NUM_SHARDS}" "${input_dir}" | tee -a "${MANIFEST}"
    done
}

echo "Submitting selected official RPD jobs"
echo "Seed: ${SEED}"
echo "Tree block sizes: ${TREE_BLOCK_SIZES[*]}"
echo "Tree leader temp: ${TREE_LEADER_TEMP}"
echo "Tree follower temp: ${TREE_FOLLOWER_TEMP}"
echo "Full temps: ${FULL_TEMPS[*]}"
echo "Run tree jobs: ${RUN_TREE}"
echo "Run full jobs: ${RUN_FULL}"
echo "RPD shards per setting: ${RPD_NUM_SHARDS}"
echo "RPD GPUs per shard: ${RPD_GPUS_PER_SHARD}"
echo "RPD GPUs per setting job: ${RPD_GPUS_PER_JOB}"
echo "RPD CPUs per setting job: ${RPD_CPUS_PER_JOB}"
echo "RPD memory per setting job: ${RPD_MEM_PER_JOB}"
echo "Rollout matrix root: ${ROLLOUT_MATRIX_ROOT}"
echo "RPD output root: ${RPD_OUTPUT_ROOT}"
echo "TEST_LIMIT: ${TEST_LIMIT}"

mkdir -p "${RPD_OUTPUT_ROOT}"
if [ ! -f "${MANIFEST}" ] || [ "${OVERWRITE_MANIFEST:-0}" = "1" ]; then
    printf "kind\tsetting\tshard_index\tnum_shards\tjob_id\trun_name\tinput_dir\n" > "${MANIFEST}"
fi

if [ "${RUN_TREE}" = "1" ]; then
    for block in "${TREE_BLOCK_SIZES[@]}"; do
        setting="tree_seed${SEED}_block${block}_t${TREE_LEADER_TEMP}_f${TREE_FOLLOWER_TEMP}"
        safe_setting="tree_seed${SEED}_block${block}_t$(temp_tag "${TREE_LEADER_TEMP}")_f$(temp_tag "${TREE_FOLLOWER_TEMP}")"
        submit_rpd "tree" "${setting}" "$(tree_input_dir "${block}")" "rpd_${safe_setting}"
    done
fi

if [ "${RUN_FULL}" = "1" ]; then
    for temp in "${FULL_TEMPS[@]}"; do
        setting="full_seed${SEED}_temp${temp}"
        safe_setting="full_seed${SEED}_temp$(temp_tag "${temp}")"
        submit_rpd "full" "${setting}" "$(full_input_dir "${temp}")" "rpd_${safe_setting}"
    done
fi

echo "Submitted RPD job manifest: ${MANIFEST}"
echo "After all jobs finish, summarize with:"
echo "  python examples/generation/summarize_official_rpd_runs.py --manifest ${MANIFEST} --output-dir ${RPD_OUTPUT_ROOT}/analysis"
