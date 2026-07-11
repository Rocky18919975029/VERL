#!/usr/bin/env bash
# Submit HPF training to the debug partition as a chain of one-step jobs.
#
# Each Slurm job advances the same run by exactly one global step by setting
# TOTAL_TRAINING_STEPS to the next target step and using resume=auto after the
# first job. This is useful for debug partitions with short wall-time limits.
#
# Example:
#   TRAIN_FILE=/data/user/zhongal/data/reschedule/DAPO-Math-17k.filtered.seed42.sample1536.parquet \
#   RUN_NAME=hpf_alg3_localwin64_mini1536_boxed_debug_seed42_$(date +%Y%m%d_%H%M%S) \
#   NUM_DEBUG_STEPS=3 \
#   bash examples/grpo_trainer/submit_hpf_debug_step_chain.sh

set -euo pipefail

DEBUG_PARTITION=${DEBUG_PARTITION:-debug}
DEBUG_NODE=${DEBUG_NODE:-ACD1-63}
DEBUG_TIME=${DEBUG_TIME:-00:30:00}
DEBUG_CPUS_PER_TASK=${DEBUG_CPUS_PER_TASK:-64}

NUM_DEBUG_STEPS=${NUM_DEBUG_STEPS:-3}
START_STEP=${START_STEP:-1}

MODEL_PATH=${MODEL_PATH:-/data/user/zhongal/.cache/qwen2.5-math-7b-local}
DATA_DIR=${DATA_DIR:-/data/user/zhongal/data/reschedule}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/DAPO-Math-17k.filtered.seed42.sample1536.parquet}

PROJECT_NAME=${PROJECT_NAME:-hpf_alg3_localwin64_mini1536_boxed_debug_seed42}
RUN_NAME=${RUN_NAME:-hpf_alg3_localwin64_4x4_b512_mini1536_boxed_debug_seed42_$(date +%Y%m%d_%H%M%S)}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
SAVE_FREQ=${SAVE_FREQ:-1}
TEST_FREQ=${TEST_FREQ:-1}
REWARD_STRICT_BOX_VERIFY=${REWARD_STRICT_BOX_VERIFY:-True}
PYTHONHASHSEED=${PYTHONHASHSEED:-42}
PYTORCH_SEED=${PYTORCH_SEED:-42}
ROLLOUT_SEED=${ROLLOUT_SEED:-42}

HPF_TREE_ROLLOUT=${HPF_TREE_ROLLOUT:-True}
HPF_TREE_NUM_PREFIXES=${HPF_TREE_NUM_PREFIXES:-4}
HPF_TREE_NUM_SUFFIXES=${HPF_TREE_NUM_SUFFIXES:-4}
HPF_FRESH_TREE_NUM_PREFIXES=${HPF_FRESH_TREE_NUM_PREFIXES:-4}
HPF_FRESH_TREE_NUM_SUFFIXES=${HPF_FRESH_TREE_NUM_SUFFIXES:-4}
HPF_FRESH_LEADER_TREE=${HPF_FRESH_LEADER_TREE:-True}
HPF_PROGRESSIVE_BLOCK_SIZE=${HPF_PROGRESSIVE_BLOCK_SIZE:-64}
HPF_LOCAL_UPDATE_WINDOW=${HPF_LOCAL_UPDATE_WINDOW:-True}
HPF_LOCAL_UPDATE_WINDOW_SIZE=${HPF_LOCAL_UPDATE_WINDOW_SIZE:-64}
HPF_HORIZON_SCHEDULE=${HPF_HORIZON_SCHEDULE:-epoch}
HPF_HORIZON_UPDATE_INTERVAL_STEPS=${HPF_HORIZON_UPDATE_INTERVAL_STEPS:-1}

if [ "${NUM_DEBUG_STEPS}" -lt 1 ]; then
    echo "NUM_DEBUG_STEPS must be >= 1, got ${NUM_DEBUG_STEPS}" >&2
    exit 1
fi

if [ "${START_STEP}" -lt 1 ]; then
    echo "START_STEP must be >= 1, got ${START_STEP}" >&2
    exit 1
fi

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Training parquet not found: ${TRAIN_FILE}" >&2
    exit 1
fi

submit_script=examples/grpo_trainer/submit_qwen2_5_math_7b_hpf_masked_grpo_h100.slurm
if [ ! -f "${submit_script}" ]; then
    echo "Submit script not found: ${submit_script}" >&2
    exit 1
fi

echo "Submitting HPF debug one-step chain"
echo "Partition: ${DEBUG_PARTITION}"
echo "Node: ${DEBUG_NODE:-any}"
echo "Time per job: ${DEBUG_TIME}"
echo "CPUs per job: ${DEBUG_CPUS_PER_TASK}"
echo "Project: ${PROJECT_NAME}"
echo "Run name: ${RUN_NAME}"
echo "Train file: ${TRAIN_FILE}"
echo "Steps: ${START_STEP}..$((START_STEP + NUM_DEBUG_STEPS - 1))"
echo "Extra Hydra overrides: $*"

previous_job=""
for offset in $(seq 0 $((NUM_DEBUG_STEPS - 1))); do
    target_step=$((START_STEP + offset))
    if [ "${target_step}" -eq 1 ] && [ -z "${previous_job}" ]; then
        resume_mode=${RESUME_FIRST_MODE:-disable}
    else
        resume_mode=${RESUME_NEXT_MODE:-auto}
    fi

    export_arg="ALL"
    export_arg+=",MODEL_PATH=${MODEL_PATH}"
    export_arg+=",DATA_DIR=${DATA_DIR}"
    export_arg+=",TRAIN_FILE=${TRAIN_FILE}"
    export_arg+=",PROJECT_NAME=${PROJECT_NAME}"
    export_arg+=",RUN_NAME=${RUN_NAME}"
    export_arg+=",TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
    export_arg+=",TOTAL_TRAINING_STEPS=${target_step}"
    export_arg+=",SAVE_FREQ=${SAVE_FREQ}"
    export_arg+=",TEST_FREQ=${TEST_FREQ}"
    export_arg+=",RESUME_MODE=${resume_mode}"
    export_arg+=",REWARD_STRICT_BOX_VERIFY=${REWARD_STRICT_BOX_VERIFY}"
    export_arg+=",PYTHONHASHSEED=${PYTHONHASHSEED}"
    export_arg+=",PYTORCH_SEED=${PYTORCH_SEED}"
    export_arg+=",ROLLOUT_SEED=${ROLLOUT_SEED}"
    export_arg+=",HPF_TREE_ROLLOUT=${HPF_TREE_ROLLOUT}"
    export_arg+=",HPF_TREE_NUM_PREFIXES=${HPF_TREE_NUM_PREFIXES}"
    export_arg+=",HPF_TREE_NUM_SUFFIXES=${HPF_TREE_NUM_SUFFIXES}"
    export_arg+=",HPF_FRESH_TREE_NUM_PREFIXES=${HPF_FRESH_TREE_NUM_PREFIXES}"
    export_arg+=",HPF_FRESH_TREE_NUM_SUFFIXES=${HPF_FRESH_TREE_NUM_SUFFIXES}"
    export_arg+=",HPF_FRESH_LEADER_TREE=${HPF_FRESH_LEADER_TREE}"
    export_arg+=",HPF_PROGRESSIVE_BLOCK_SIZE=${HPF_PROGRESSIVE_BLOCK_SIZE}"
    export_arg+=",HPF_LOCAL_UPDATE_WINDOW=${HPF_LOCAL_UPDATE_WINDOW}"
    export_arg+=",HPF_LOCAL_UPDATE_WINDOW_SIZE=${HPF_LOCAL_UPDATE_WINDOW_SIZE}"
    export_arg+=",HPF_HORIZON_SCHEDULE=${HPF_HORIZON_SCHEDULE}"
    export_arg+=",HPF_HORIZON_UPDATE_INTERVAL_STEPS=${HPF_HORIZON_UPDATE_INTERVAL_STEPS}"

    sbatch_args=(
        --parsable
        --partition="${DEBUG_PARTITION}"
        --time="${DEBUG_TIME}"
        --cpus-per-task="${DEBUG_CPUS_PER_TASK}"
        --export="${export_arg}"
    )
    if [ -n "${DEBUG_NODE}" ]; then
        sbatch_args+=(--nodelist="${DEBUG_NODE}")
    fi
    if [ -n "${previous_job}" ]; then
        sbatch_args+=(--dependency="afterok:${previous_job}")
    fi

    job_id=$(sbatch "${sbatch_args[@]}" "${submit_script}" data.shuffle=True data.seed=42 "$@")
    echo -e "step\t${target_step}\tjob\t${job_id}\tresume\t${resume_mode}\tdependency\t${previous_job:-none}"
    previous_job="${job_id}"
done

echo "Submitted chain ending at job ${previous_job}"
echo "Monitor with:"
echo "  squeue -j ${previous_job}"
echo "  squeue --name=verl-hpf-mask"
