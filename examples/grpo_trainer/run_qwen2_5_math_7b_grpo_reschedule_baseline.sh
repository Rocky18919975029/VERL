#!/usr/bin/env bash
# Reproduce the GRPO baseline from "Scheduling Your LLM Reinforcement Learning
# with Reasoning Trees" using the released Re-Schedule hyperparameters.
#
# This script intentionally does not enable any Re-Schedule dynamic weighting:
# no data.use_dynamic_weights, no metric_column, no alpha_mode, and no
# dynamic_weights_* overrides.
#
# Modes:
#   MODE=train  Train the GRPO baseline for the paper's 150 rollout steps.
#   MODE=eval   Run final validation on the paper's six reported math sets.
#
# Example:
#   MODEL_PATH=/path/to/Qwen2.5-Math-7B \
#   DATA_DIR=/path/to/Re-Schedule/datasets \
#   MODE=train \
#   bash examples/grpo_trainer/run_qwen2_5_math_7b_grpo_reschedule_baseline.sh

set -xeuo pipefail

MODE=${MODE:-train}

PROJECT_NAME=${PROJECT_NAME:-grpo_dapo_math17k_reschedule_baseline}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Math-7B}
DATA_DIR=${DATA_DIR:-./datasets}
RUN_NAME=${RUN_NAME:-qwen2_5_math_7b_grpo_baseline_$(date +%Y%m%d_%H%M)}

TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/DAPO-Math-17k.parquet}
TRAIN_VAL_FILES=${TRAIN_VAL_FILES:-"['${DATA_DIR}/aime24.parquet']"}
PAPER_EVAL_FILES=${PAPER_EVAL_FILES:-"['${DATA_DIR}/aime24.parquet','${DATA_DIR}/aime25.parquet','${DATA_DIR}/amc23.parquet','${DATA_DIR}/math500.parquet','${DATA_DIR}/minerva_math.parquet','${DATA_DIR}/olympiadbench.parquet']"}
REPO_EXTRA_EVAL_FILES=${REPO_EXTRA_EVAL_FILES:-"['${DATA_DIR}/gsm8k_test.parquet','${DATA_DIR}/omni.parquet']"}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
ROLLOUT_N=${ROLLOUT_N:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-50}

ACTOR_LR=${ACTOR_LR:-1e-6}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
ROLLOUT_TRAIN_TEMPERATURE=${ROLLOUT_TRAIN_TEMPERATURE:-1.0}
ROLLOUT_TRAIN_TOP_P=${ROLLOUT_TRAIN_TOP_P:-1.0}
ROLLOUT_VAL_TEMPERATURE=${ROLLOUT_VAL_TEMPERATURE:-1.0}
ROLLOUT_VAL_TOP_P=${ROLLOUT_VAL_TOP_P:-0.7}
ROLLOUT_VAL_N=${ROLLOUT_VAL_N:-1}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-False}

TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_MODE
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}
export WANDB_TIMEOUT=${WANDB_TIMEOUT:-300}
export WANDB_RETRY_DELAY=${WANDB_RETRY_DELAY:-60}
export WANDB_MAX_RETRIES=${WANDB_MAX_RETRIES:-10}

export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export PYTORCH_SEED=${PYTORCH_SEED:-42}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

SAVE_CONTENTS=${SAVE_CONTENTS:-${ACTOR_CHECKPOINT_SAVE_CONTENTS:-"['hf_model']"}}

COMMON_DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=left
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    +actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}
    actor_rollout_ref.actor.checkpoint.save_contents="${SAVE_CONTENTS}"
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TRAIN_TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TRAIN_TOP_P}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.val_kwargs.n=${ROLLOUT_VAL_N}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=${ROLLOUT_VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${ROLLOUT_VAL_TOP_P}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD}
)

TRAINER_COMMON=(
    trainer.logger="${TRAINER_LOGGER}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${RUN_NAME}"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.critic_warmup=0
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ:-10}
    trainer.test_freq=${TEST_FREQ:-1}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.resume_mode=${RESUME_MODE:-disable}
    +trainer.save_best_only=${SAVE_BEST_ONLY:-False}
    +trainer.delete_old_best_checkpoint=${DELETE_OLD_BEST_CHECKPOINT:-False}
    +trainer.save_after=${SAVE_AFTER:-60}
    +trainer.best_metric_key=${BEST_METRIC_KEY:-val-core/math_dapo/acc/mean@32}
)

case "${MODE}" in
    train)
        DATA=( "${COMMON_DATA[@]}" data.val_files="${TRAIN_VAL_FILES}" )
        TRAINER=(
            "${TRAINER_COMMON[@]}"
            trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
            trainer.rollout_data_dir="${ROLLOUT_DATA_DIR:-./rollout_data/${PROJECT_NAME}/${RUN_NAME}}"
        )
        ;;
    eval)
        if [ "${INCLUDE_REPO_EXTRA_EVAL:-0}" = "1" ]; then
            VAL_FILES=${VAL_FILES:-"['${DATA_DIR}/aime24.parquet','${DATA_DIR}/aime25.parquet','${DATA_DIR}/amc23.parquet','${DATA_DIR}/math500.parquet','${DATA_DIR}/minerva_math.parquet','${DATA_DIR}/olympiadbench.parquet','${DATA_DIR}/gsm8k_test.parquet','${DATA_DIR}/omni.parquet']"}
        else
            VAL_FILES=${VAL_FILES:-${PAPER_EVAL_FILES}}
        fi
        DATA=( "${COMMON_DATA[@]}" data.val_files="${VAL_FILES}" )
        TRAINER=(
            "${TRAINER_COMMON[@]}"
            trainer.val_only=True
            trainer.rollout_data_dir="${ROLLOUT_DATA_DIR:-./rollout_data/${PROJECT_NAME}/${RUN_NAME}_eval}"
        )
        ;;
    *)
        echo "Unknown MODE=${MODE}; expected train or eval" >&2
        exit 1
        ;;
esac

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"
