#!/usr/bin/env bash
# Entropy paper reproduction smoke/run script | Qwen2.5-7B | FSDP | vLLM.
#
# METHOD choices:
#   baseline    : vanilla GRPO, clip low/high = 0.2/0.2
#   cliphigher  : vanilla GRPO, clip low/high = 0.2/0.28
#   clip_cov    : CLIP-Cov, clip_cov_ratio = 2e-4, bounds = [1, 5]
#   kl_cov      : KL-Cov, kl_cov_ratio = 2e-3 for Qwen2.5-7B, ppo_kl_coef = 1
#
# Default values are conservative enough for an 8xH100 smoke test. Set
# ENTROPY_REPRO_FULL=1 to switch to the paper-like 7B settings from
# PRIME-RL/Entropy-Mechanism-of-RL.

set -xeuo pipefail

METHOD=${METHOD:-kl_cov}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

TRAIN_FILE=${TRAIN_FILE:-$HOME/data/entropy/dapo-math-17k.parquet}
VAL_FILES=${VAL_FILES:-"['$HOME/data/entropy/aime-2024.parquet']"}

if [ "${ENTROPY_REPRO_FULL:-0}" = "1" ]; then
    train_batch_size=${TRAIN_BATCH_SIZE:-256}
    ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
    rollout_n=${ROLLOUT_N:-8}
    rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.35}
    max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
    max_response_length=${MAX_RESPONSE_LENGTH:-8192}
    rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}
    rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${rollout_max_model_len}}
    rollout_enforce_eager=${ROLLOUT_ENFORCE_EAGER:-True}
    ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-30720}
    actor_param_offload=${ACTOR_PARAM_OFFLOAD:-True}
    actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-True}
    ref_param_offload=${REF_PARAM_OFFLOAD:-True}
    total_epochs=${TOTAL_EPOCHS:-15}
    total_training_steps=${TOTAL_TRAINING_STEPS:-null}
    test_freq=${TEST_FREQ:-4}
    save_freq=${SAVE_FREQ:-32}
    resume_mode=${RESUME_MODE:-auto}
    max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-3}
else
    train_batch_size=${TRAIN_BATCH_SIZE:-16}
    ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-8}
    rollout_n=${ROLLOUT_N:-2}
    rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.55}
    max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
    max_response_length=${MAX_RESPONSE_LENGTH:-1024}
    ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
    rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}
    rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${ppo_max_token_len_per_gpu}}
    rollout_enforce_eager=${ROLLOUT_ENFORCE_EAGER:-False}
    actor_param_offload=${ACTOR_PARAM_OFFLOAD:-False}
    actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-False}
    ref_param_offload=${REF_PARAM_OFFLOAD:-False}
    total_epochs=${TOTAL_EPOCHS:-1}
    total_training_steps=${TOTAL_TRAINING_STEPS:-2}
    test_freq=${TEST_FREQ:--1}
    save_freq=${SAVE_FREQ:-1}
    resume_mode=${RESUME_MODE:-auto}
    max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-2}
fi

if [ -n "${RESUME_FROM_PATH:-}" ] && [ -z "${RESUME_MODE:-}" ]; then
    resume_mode=resume_path
fi

actor_lr=${ACTOR_LR:-5e-7}
rollout_tp=${ROLLOUT_TP:-2}
max_num_gen_batches=${MAX_NUM_GEN_BATCHES:-10}
filter_groups=${FILTER_GROUPS:-True}
attn_implementation=${ATTN_IMPLEMENTATION:-sdpa}
dataloader_num_workers=${DATALOADER_NUM_WORKERS:-8}

case "${METHOD}" in
    baseline)
        loss_mode=vanilla
        clip_ratio_low=0.2
        clip_ratio_high=0.2
        ;;
    cliphigher | clip_higher)
        loss_mode=vanilla
        clip_ratio_low=0.2
        clip_ratio_high=0.28
        ;;
    clip_cov)
        loss_mode=clip_cov
        clip_ratio_low=1.0
        clip_ratio_high=1.0
        ;;
    kl_cov)
        loss_mode=kl_cov
        clip_ratio_low=0.2
        clip_ratio_high=0.2
        ;;
    *)
        echo "Unknown METHOD=${METHOD}; expected baseline, cliphigher, clip_cov, or kl_cov" >&2
        exit 1
        ;;
esac

PROJECT_NAME=${PROJECT_NAME:-entropy_qwen2_5_7b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_7b_${METHOD}_$(date +%Y%m%d_%H%M)}
trainer_logger=${TRAINER_LOGGER:-'["console","wandb"]'}

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    "+algorithm.filter_groups={enable:${filter_groups},metric:acc,max_num_gen_batches:${max_num_gen_batches}}"
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="${VAL_FILES}"
    data.prompt_key=prompt
    data.truncation=left
    data.filter_overlong_prompts=False
    data.return_raw_chat=True
    data.train_batch_size=${train_batch_size}
    data.dataloader_num_workers=${dataloader_num_workers}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    +actor_rollout_ref.model.override_config.attn_implementation=${attn_implementation}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.weight_decay=0
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode}
    actor_rollout_ref.actor.policy_loss.clip_cov_ratio=0.0002
    actor_rollout_ref.actor.policy_loss.clip_cov_lb=1.0
    actor_rollout_ref.actor.policy_loss.clip_cov_ub=5.0
    actor_rollout_ref.actor.policy_loss.kl_cov_ratio=0.002
    actor_rollout_ref.actor.policy_loss.ppo_kl_coef=1.0
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}
    actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager}
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.val_kwargs.temperature=0
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.n=1
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_param_offload}
)

REWARD=(
    reward.reward_manager.name=dapo
)

RAY=()
if [ -n "${RAY_TMP_DIR:-}" ]; then
    RAY+=("+ray_kwargs.ray_init._temp_dir=${RAY_TMP_DIR}")
fi
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    RAY+=("+ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH=${LD_LIBRARY_PATH}")
fi

RESUME=(
    trainer.resume_mode=${resume_mode}
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep}
)
if [ -n "${RESUME_FROM_PATH:-}" ]; then
    RESUME+=(trainer.resume_from_path="${RESUME_FROM_PATH}")
fi

TRAINER=(
    trainer.balance_batch=True
    trainer.logger="${trainer_logger}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.test_freq=${test_freq}
    trainer.save_freq=${save_freq}
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_training_steps}
)

echo "Effective entropy training config:"
echo "  mode: $([ "${ENTROPY_REPRO_FULL:-0}" = "1" ] && echo full || echo smoke)"
echo "  method: ${METHOD}"
echo "  experiment: ${EXPERIMENT_NAME}"
echo "  train_batch_size: ${train_batch_size}"
echo "  ppo_mini_batch_size: ${ppo_mini_batch_size}"
echo "  rollout_n: ${rollout_n}"
echo "  rollout_gpu_memory_utilization: ${rollout_gpu_mem_util}"
echo "  rollout_max_model_len: ${rollout_max_model_len}"
echo "  rollout_max_num_batched_tokens: ${rollout_max_num_batched_tokens}"
echo "  rollout_enforce_eager: ${rollout_enforce_eager}"
echo "  max_prompt_length: ${max_prompt_length}"
echo "  max_response_length: ${max_response_length}"
echo "  ppo_max_token_len_per_gpu: ${ppo_max_token_len_per_gpu}"
echo "  total_training_steps: ${total_training_steps}"
echo "  save_freq: ${save_freq}"
echo "  test_freq: ${test_freq}"
echo "  resume_mode: ${resume_mode}"

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${RAY[@]}" \
    "${RESUME[@]}" \
    "${TRAINER[@]}" \
    "$@"
