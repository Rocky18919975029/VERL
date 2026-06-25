#!/usr/bin/env bash
# Experimental HPF masked-GRPO smoke path.
#
# This wrapper reuses the already validated Re-Schedule GRPO baseline script and
# only appends HPF-specific switches. It intentionally does not modify the
# baseline script so baseline reproduction remains unchanged.

set -xeuo pipefail

PROJECT_NAME=${PROJECT_NAME:-hpf_masked_grpo_dapo_math17k}
RUN_NAME=${RUN_NAME:-qwen2_5_math_7b_hpf_masked_grpo_$(date +%Y%m%d_%H%M)}

HPF_PROGRESSIVE_BLOCK_SIZE=${HPF_PROGRESSIVE_BLOCK_SIZE:-256}
HPF_MAX_RESPONSE_LENGTH=${HPF_MAX_RESPONSE_LENGTH:-${MAX_RESPONSE_LENGTH:-3072}}
HPF_EPSILON=${HPF_EPSILON:-1e-6}
HPF_STD_NORMALIZE=${HPF_STD_NORMALIZE:-True}
HPF_HORIZON_SCHEDULE=${HPF_HORIZON_SCHEDULE:-epoch}
HPF_HORIZON_UPDATE_INTERVAL_STEPS=${HPF_HORIZON_UPDATE_INTERVAL_STEPS:-1}
HPF_PREFIX_KL_COEF=${HPF_PREFIX_KL_COEF:-0.001}
HPF_SUFFIX_KL_COEF=${HPF_SUFFIX_KL_COEF:-0.001}
HPF_CORRECTION_CLIP=${HPF_CORRECTION_CLIP:-5.0}
HPF_PROGRESS_LOG_INTERVAL=${HPF_PROGRESS_LOG_INTERVAL:-1}
HPF_TREE_ROLLOUT=${HPF_TREE_ROLLOUT:-False}
HPF_TREE_NUM_PREFIXES=${HPF_TREE_NUM_PREFIXES:-4}
HPF_TREE_NUM_SUFFIXES=${HPF_TREE_NUM_SUFFIXES:-2}
HPF_TREE_PREFIX_TEMPERATURE=${HPF_TREE_PREFIX_TEMPERATURE:-1.0}
HPF_TREE_PREFIX_TOP_P=${HPF_TREE_PREFIX_TOP_P:-1.0}
HPF_TREE_SUFFIX_TEMPERATURE=${HPF_TREE_SUFFIX_TEMPERATURE:-0.25}
HPF_TREE_SUFFIX_TOP_P=${HPF_TREE_SUFFIX_TOP_P:-1.0}
HPF_LOSS_AGG_MODE=${HPF_LOSS_AGG_MODE:-token-mean}

HPF_ARGS=(
    algorithm.hpf_rlvr.enable=True
    algorithm.hpf_rlvr.progressive_block_size="${HPF_PROGRESSIVE_BLOCK_SIZE}"
    algorithm.hpf_rlvr.max_response_length="${HPF_MAX_RESPONSE_LENGTH}"
    algorithm.hpf_rlvr.epsilon="${HPF_EPSILON}"
    algorithm.hpf_rlvr.std_normalize="${HPF_STD_NORMALIZE}"
    algorithm.hpf_rlvr.horizon_schedule="${HPF_HORIZON_SCHEDULE}"
    algorithm.hpf_rlvr.horizon_update_interval_steps="${HPF_HORIZON_UPDATE_INTERVAL_STEPS}"
    algorithm.hpf_rlvr.prefix_kl_coef="${HPF_PREFIX_KL_COEF}"
    algorithm.hpf_rlvr.suffix_kl_coef="${HPF_SUFFIX_KL_COEF}"
    algorithm.hpf_rlvr.correction_clip="${HPF_CORRECTION_CLIP}"
    algorithm.hpf_rlvr.progress_log_interval="${HPF_PROGRESS_LOG_INTERVAL}"
    actor_rollout_ref.actor.loss_agg_mode="${HPF_LOSS_AGG_MODE}"
)

if [ "${HPF_TREE_ROLLOUT}" = "True" ] || [ "${HPF_TREE_ROLLOUT}" = "true" ] || [ "${HPF_TREE_ROLLOUT}" = "1" ]; then
    ROLLOUT_N=${ROLLOUT_N:-$((HPF_TREE_NUM_PREFIXES * HPF_TREE_NUM_SUFFIXES))}
    export ROLLOUT_N
    HPF_ARGS+=(
        algorithm.hpf_rlvr.tree_rollout.enable=True
        algorithm.hpf_rlvr.tree_rollout.num_prefixes="${HPF_TREE_NUM_PREFIXES}"
        algorithm.hpf_rlvr.tree_rollout.num_suffixes="${HPF_TREE_NUM_SUFFIXES}"
        algorithm.hpf_rlvr.tree_rollout.prefix_temperature="${HPF_TREE_PREFIX_TEMPERATURE}"
        algorithm.hpf_rlvr.tree_rollout.prefix_top_p="${HPF_TREE_PREFIX_TOP_P}"
        algorithm.hpf_rlvr.tree_rollout.suffix_temperature="${HPF_TREE_SUFFIX_TEMPERATURE}"
        algorithm.hpf_rlvr.tree_rollout.suffix_top_p="${HPF_TREE_SUFFIX_TOP_P}"
    )
fi

PROJECT_NAME="${PROJECT_NAME}" \
RUN_NAME="${RUN_NAME}" \
bash examples/grpo_trainer/run_qwen2_5_math_7b_grpo_reschedule_baseline.sh \
    "${HPF_ARGS[@]}" \
    "$@"
