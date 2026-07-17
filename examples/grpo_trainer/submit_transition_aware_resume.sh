#!/usr/bin/env bash

# Fail-closed launcher for resuming transition-aware mixed-policy training.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  submit_transition_aware_resume.sh \
    --project PROJECT \
    --run RUN_NAME \
    --lambda LAMBDA \
    --from-step STEP \
    --to-step STEP \
    [--dry-run]

The launcher verifies the exact checkpoint, optimizer state, project, run,
lambda, and target step before submitting. The Slurm job repeats the same
checks before loading Python or the model.
EOF
}

PROJECT=
RUN_NAME=
LAMBDA_TRANS=
FROM_STEP=
TO_STEP=
DRY_RUN=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --project)
            PROJECT=${2:?missing value for --project}
            shift 2
            ;;
        --run)
            RUN_NAME=${2:?missing value for --run}
            shift 2
            ;;
        --lambda)
            LAMBDA_TRANS=${2:?missing value for --lambda}
            shift 2
            ;;
        --from-step)
            FROM_STEP=${2:?missing value for --from-step}
            shift 2
            ;;
        --to-step)
            TO_STEP=${2:?missing value for --to-step}
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for entry in \
    "project:${PROJECT}" \
    "run:${RUN_NAME}" \
    "lambda:${LAMBDA_TRANS}" \
    "from-step:${FROM_STEP}" \
    "to-step:${TO_STEP}"
do
    name=${entry%%:*}
    value=${entry#*:}
    if [ -z "${value}" ]; then
        echo "ERROR: --${name} is required" >&2
        usage >&2
        exit 2
    fi
    if [[ "${value}" == *,* ]]; then
        echo "ERROR: --${name} cannot contain a comma: ${value}" >&2
        exit 2
    fi
done

if ! [[ "${FROM_STEP}" =~ ^[0-9]+$ && "${TO_STEP}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --from-step and --to-step must be nonnegative integers" >&2
    exit 2
fi
if [ "${TO_STEP}" -le "${FROM_STEP}" ]; then
    echo "ERROR: --to-step (${TO_STEP}) must exceed --from-step (${FROM_STEP})" >&2
    exit 2
fi
if ! [[ "${LAMBDA_TRANS}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]; then
    echo "ERROR: --lambda must be a nonnegative number: ${LAMBDA_TRANS}" >&2
    exit 2
fi

CANONICAL_LAMBDA=$(awk -v value="${LAMBDA_TRANS}" 'BEGIN { printf "%.12g", value + 0 }')
LAMBDA_MARKER=lambda$(printf '%s' "${CANONICAL_LAMBDA}" | sed -e 's/-/m/g' -e 's/\./p/g' -e 's/+//g')
DECLARED_LAMBDA_MARKERS=$(
    printf '%s\n%s\n' "${PROJECT}" "${RUN_NAME}" \
        | grep -oE 'lambda[0-9]+(p[0-9]+)?' \
        | sort -u \
        | tr '\n' ' ' \
        || true
)
for marker in ${DECLARED_LAMBDA_MARKERS}; do
    if [ "${marker}" != "${LAMBDA_MARKER}" ]; then
        echo "ERROR: checkpoint lineage declares ${marker}, but requested lambda is ${CANONICAL_LAMBDA} (${LAMBDA_MARKER})" >&2
        exit 2
    fi
done
if [ "${CANONICAL_LAMBDA}" != "1" ] && [ -z "${DECLARED_LAMBDA_MARKERS}" ]; then
    echo "ERROR: non-default lambda ${CANONICAL_LAMBDA} requires ${LAMBDA_MARKER} in the project or run name" >&2
    echo "ERROR: refusing to resume an unlabelled checkpoint with a different lambda" >&2
    exit 2
fi

VERL_WORKDIR=${VERL_WORKDIR:-$(pwd)}
MODEL_PATH=${MODEL_PATH:-/data/user/zhongal/.cache/qwen2.5-math-7b-local}
DATA_DIR=${DATA_DIR:-/data/user/zhongal/data/reschedule}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/DAPO-Math-17k.filtered.seed42.sample1536.parquet}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${VERL_WORKDIR}/checkpoints/${PROJECT}/${RUN_NAME}}
LATEST_FILE=${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt
STEP_DIR=${CHECKPOINT_ROOT}/global_step_${FROM_STEP}

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: model directory does not exist: ${MODEL_PATH}" >&2
    exit 1
fi
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: training parquet does not exist: ${TRAIN_FILE}" >&2
    exit 1
fi
if [ ! -f "${LATEST_FILE}" ]; then
    echo "ERROR: checkpoint pointer does not exist: ${LATEST_FILE}" >&2
    exit 1
fi

LATEST=$(tr -d '[:space:]' < "${LATEST_FILE}")
if [ "${LATEST}" != "${FROM_STEP}" ]; then
    echo "ERROR: latest checkpoint is ${LATEST}, expected ${FROM_STEP}" >&2
    exit 1
fi
if [ ! -d "${STEP_DIR}/actor" ]; then
    echo "ERROR: actor checkpoint directory is missing: ${STEP_DIR}/actor" >&2
    exit 1
fi
if [ ! -f "${STEP_DIR}/data.pt" ]; then
    echo "ERROR: dataloader/extra state is missing: ${STEP_DIR}/data.pt" >&2
    exit 1
fi
if ! compgen -G "${STEP_DIR}/actor/optim_world_size_*" >/dev/null; then
    echo "ERROR: optimizer shards are missing from ${STEP_DIR}/actor" >&2
    exit 1
fi

export VERL_WORKDIR MODEL_PATH DATA_DIR TRAIN_FILE CHECKPOINT_ROOT
export PROJECT_NAME=${PROJECT}
export RUN_NAME=${RUN_NAME}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
export ROLLOUT_N=${ROLLOUT_N:-16}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
export TOTAL_TRAINING_STEPS=${TO_STEP}
export SAVE_FREQ=${SAVE_FREQ:-1}
export TEST_FREQ=${TEST_FREQ:-1}
export RESUME_MODE=auto
export SAVE_CONTENTS=${SAVE_CONTENTS:-"['model','optimizer','extra','hf_model']"}
export REWARD_STRICT_BOX_VERIFY=${REWARD_STRICT_BOX_VERIFY:-True}
export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export PYTORCH_SEED=${PYTORCH_SEED:-42}
export ROLLOUT_SEED=${ROLLOUT_SEED:-42}
export ROLLOUT_LOGPROB_REUSE=${ROLLOUT_LOGPROB_REUSE:-True}

export HPF_TREE_ROLLOUT=True
export HPF_TREE_NUM_PREFIXES=${HPF_TREE_NUM_PREFIXES:-16}
export HPF_TREE_NUM_SUFFIXES=${HPF_TREE_NUM_SUFFIXES:-1}
export HPF_TREE_PREFIX_TEMPERATURE=${HPF_TREE_PREFIX_TEMPERATURE:-1.0}
export HPF_TREE_PREFIX_TOP_P=${HPF_TREE_PREFIX_TOP_P:-1.0}
export HPF_TREE_SUFFIX_TEMPERATURE=${HPF_TREE_SUFFIX_TEMPERATURE:-0.25}
export HPF_TREE_SUFFIX_TOP_P=${HPF_TREE_SUFFIX_TOP_P:-1.0}
export HPF_FRESH_LEADER_TREE=False
export HPF_LOCAL_UPDATE_WINDOW=False
export HPF_ROLE_PHASED_TRAINING=False
export HPF_PROGRESSIVE_BLOCK_SIZE=${HPF_PROGRESSIVE_BLOCK_SIZE:-64}
export HPF_MAX_RESPONSE_LENGTH=${HPF_MAX_RESPONSE_LENGTH:-3072}
export HPF_HORIZON_SCHEDULE=${HPF_HORIZON_SCHEDULE:-epoch}
export HPF_HORIZON_UPDATE_INTERVAL_STEPS=${HPF_HORIZON_UPDATE_INTERVAL_STEPS:-1}
export HPF_MIXED_POLICY_GRPO=True
export HPF_MIXED_POLICY_SUFFIX_WINDOW_SIZE=null
export HPF_MIXED_POLICY_TRANSITION_AWARE_ROLLOUT=True
export HPF_TRANSITION_AWARE_MIXED_POLICY_OPTIMIZATION=True
export HPF_TRANSITION_LAMBDA=${LAMBDA_TRANS}
export HPF_TRANSITION_DIAGNOSTICS=False

export EXPECTED_PROJECT_NAME=${PROJECT}
export EXPECTED_RUN_NAME=${RUN_NAME}
export EXPECTED_TRANSITION_LAMBDA=${LAMBDA_TRANS}
export EXPECTED_RESUME_STEP=${FROM_STEP}

echo "[RESUME-LAUNCHER] verified checkpoint and optimizer state" >&2
echo "[RESUME-LAUNCHER] project=${PROJECT_NAME}" >&2
echo "[RESUME-LAUNCHER] run=${RUN_NAME}" >&2
echo "[RESUME-LAUNCHER] lambda=${HPF_TRANSITION_LAMBDA}" >&2
echo "[RESUME-LAUNCHER] steps=${EXPECTED_RESUME_STEP}->${TOTAL_TRAINING_STEPS}" >&2
echo "[RESUME-LAUNCHER] reuse_vllm_rollout_log_probs=${ROLLOUT_LOGPROB_REUSE}" >&2
echo "[RESUME-LAUNCHER] checkpoint=${CHECKPOINT_ROOT}" >&2

EXPORTS="ALL,PROJECT_NAME=${PROJECT_NAME},RUN_NAME=${RUN_NAME},HPF_TRANSITION_LAMBDA=${HPF_TRANSITION_LAMBDA},ROLLOUT_LOGPROB_REUSE=${ROLLOUT_LOGPROB_REUSE},RESUME_MODE=${RESUME_MODE},EXPECTED_PROJECT_NAME=${EXPECTED_PROJECT_NAME},EXPECTED_RUN_NAME=${EXPECTED_RUN_NAME},EXPECTED_TRANSITION_LAMBDA=${EXPECTED_TRANSITION_LAMBDA},EXPECTED_RESUME_STEP=${EXPECTED_RESUME_STEP},TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}"
SLURM_SCRIPT=examples/grpo_trainer/submit_qwen2_5_math_7b_hpf_masked_grpo_h100.slurm

if [ "${DRY_RUN}" = true ]; then
    printf 'sbatch --parsable --export=%q %q data.shuffle=True data.seed=42\n' \
        "${EXPORTS}" "${SLURM_SCRIPT}"
    exit 0
fi

JOB_ID=$(sbatch --parsable \
    --export="${EXPORTS}" \
    "${SLURM_SCRIPT}" \
    data.shuffle=True \
    data.seed=42)

echo "[RESUME-LAUNCHER] submitted job=${JOB_ID}" >&2
printf '%s\n' "${JOB_ID}"
