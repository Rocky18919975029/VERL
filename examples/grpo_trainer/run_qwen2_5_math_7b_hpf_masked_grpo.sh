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
HPF_FRESH_LEADER_TREE=${HPF_FRESH_LEADER_TREE:-False}
HPF_LOCAL_UPDATE_WINDOW=${HPF_LOCAL_UPDATE_WINDOW:-False}
HPF_LOCAL_UPDATE_WINDOW_SIZE=${HPF_LOCAL_UPDATE_WINDOW_SIZE:-null}
HPF_ROLE_PHASED_TRAINING=${HPF_ROLE_PHASED_TRAINING:-False}
HPF_MIXED_POLICY_GRPO=${HPF_MIXED_POLICY_GRPO:-False}
HPF_FOLLOWER_PHASE_EPOCHS=${HPF_FOLLOWER_PHASE_EPOCHS:-1}
HPF_LEADER_PHASE_EPOCHS=${HPF_LEADER_PHASE_EPOCHS:-1}
HPF_PROGRESS_LOG_INTERVAL=${HPF_PROGRESS_LOG_INTERVAL:-1}
HPF_TREE_ROLLOUT=${HPF_TREE_ROLLOUT:-False}
HPF_TREE_NUM_PREFIXES=${HPF_TREE_NUM_PREFIXES:-4}
HPF_TREE_NUM_SUFFIXES=${HPF_TREE_NUM_SUFFIXES:-2}
HPF_TREE_PREFIX_TEMPERATURE=${HPF_TREE_PREFIX_TEMPERATURE:-1.0}
HPF_TREE_PREFIX_TOP_P=${HPF_TREE_PREFIX_TOP_P:-1.0}
HPF_TREE_SUFFIX_TEMPERATURE=${HPF_TREE_SUFFIX_TEMPERATURE:-0.25}
HPF_TREE_SUFFIX_TOP_P=${HPF_TREE_SUFFIX_TOP_P:-1.0}
HPF_FRESH_TREE_NUM_PREFIXES=${HPF_FRESH_TREE_NUM_PREFIXES:-${HPF_TREE_NUM_PREFIXES}}
HPF_FRESH_TREE_NUM_SUFFIXES=${HPF_FRESH_TREE_NUM_SUFFIXES:-${HPF_TREE_NUM_SUFFIXES}}
HPF_LOSS_AGG_MODE=${HPF_LOSS_AGG_MODE:-token-mean}

if [ "${HPF_ROLE_PHASED_TRAINING}" = "True" ] || [ "${HPF_ROLE_PHASED_TRAINING}" = "true" ] \
    || [ "${HPF_ROLE_PHASED_TRAINING}" = "1" ]; then
    if [ "${HPF_FRESH_LEADER_TREE}" != "True" ] && [ "${HPF_FRESH_LEADER_TREE}" != "true" ] \
        && [ "${HPF_FRESH_LEADER_TREE}" != "1" ]; then
        echo "HPF role-phased training requires HPF_FRESH_LEADER_TREE=True." >&2
        exit 1
    fi
    if [ "${HPF_TREE_ROLLOUT}" != "True" ] && [ "${HPF_TREE_ROLLOUT}" != "true" ] \
        && [ "${HPF_TREE_ROLLOUT}" != "1" ]; then
        echo "HPF role-phased training requires HPF_TREE_ROLLOUT=True." >&2
        exit 1
    fi
    if [ "${HPF_HORIZON_SCHEDULE}" != "epoch" ]; then
        echo "HPF role-phased training requires HPF_HORIZON_SCHEDULE=epoch." >&2
        exit 1
    fi
    if [ "${HPF_FOLLOWER_PHASE_EPOCHS}" -le 0 ] || [ "${HPF_LEADER_PHASE_EPOCHS}" -le 0 ]; then
        echo "HPF follower and leader phase epochs must both be positive." >&2
        exit 1
    fi
fi

if [ "${HPF_MIXED_POLICY_GRPO}" = "True" ] || [ "${HPF_MIXED_POLICY_GRPO}" = "true" ] \
    || [ "${HPF_MIXED_POLICY_GRPO}" = "1" ]; then
    if [ "${HPF_TREE_ROLLOUT}" != "True" ] && [ "${HPF_TREE_ROLLOUT}" != "true" ] \
        && [ "${HPF_TREE_ROLLOUT}" != "1" ]; then
        echo "HPF mixed-policy GRPO requires HPF_TREE_ROLLOUT=True." >&2
        exit 1
    fi
    if [ "${HPF_TREE_NUM_SUFFIXES}" -ne 1 ]; then
        echo "HPF mixed-policy GRPO requires HPF_TREE_NUM_SUFFIXES=1." >&2
        exit 1
    fi
    if [ "${HPF_TREE_PREFIX_TOP_P}" != "1.0" ] && [ "${HPF_TREE_PREFIX_TOP_P}" != "1" ]; then
        echo "HPF mixed-policy GRPO requires HPF_TREE_PREFIX_TOP_P=1.0." >&2
        exit 1
    fi
    if [ "${HPF_TREE_SUFFIX_TOP_P}" != "1.0" ] && [ "${HPF_TREE_SUFFIX_TOP_P}" != "1" ]; then
        echo "HPF mixed-policy GRPO requires HPF_TREE_SUFFIX_TOP_P=1.0." >&2
        exit 1
    fi
    if [ "${HPF_FRESH_LEADER_TREE}" = "True" ] || [ "${HPF_FRESH_LEADER_TREE}" = "true" ] \
        || [ "${HPF_FRESH_LEADER_TREE}" = "1" ]; then
        echo "HPF mixed-policy GRPO requires HPF_FRESH_LEADER_TREE=False." >&2
        exit 1
    fi
    if [ "${HPF_LOCAL_UPDATE_WINDOW}" = "True" ] || [ "${HPF_LOCAL_UPDATE_WINDOW}" = "true" ] \
        || [ "${HPF_LOCAL_UPDATE_WINDOW}" = "1" ]; then
        echo "HPF mixed-policy GRPO defines its own update window; set HPF_LOCAL_UPDATE_WINDOW=False." >&2
        exit 1
    fi
    if [ "${HPF_ROLE_PHASED_TRAINING}" = "True" ] || [ "${HPF_ROLE_PHASED_TRAINING}" = "true" ] \
        || [ "${HPF_ROLE_PHASED_TRAINING}" = "1" ]; then
        echo "HPF mixed-policy GRPO cannot be combined with role-phased training." >&2
        exit 1
    fi
fi

# A suffix KL to theta_F is exactly zero on the first leader optimizer step.
# Keep at least two mini-batches per HPF phase by default so the second and
# later steps evaluate the combined PG+KL loss after the policy has moved.
HPF_TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
HPF_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
if [ -z "${PPO_MINI_BATCH_SIZE+x}" ]; then
    HPF_PPO_MINI_TARGET=$((HPF_TRAIN_BATCH_SIZE / 2))
    if [ "${HPF_PPO_MINI_TARGET}" -gt 32 ]; then
        HPF_PPO_MINI_TARGET=32
    fi
    PPO_MINI_BATCH_SIZE=$(((HPF_PPO_MINI_TARGET / HPF_MICRO_BATCH_SIZE) * HPF_MICRO_BATCH_SIZE))
    if [ "${PPO_MINI_BATCH_SIZE}" -lt "${HPF_MICRO_BATCH_SIZE}" ]; then
        PPO_MINI_BATCH_SIZE=${HPF_MICRO_BATCH_SIZE}
    fi
fi
HPF_LEADER_MINI_BATCHES=$(((HPF_TRAIN_BATCH_SIZE + PPO_MINI_BATCH_SIZE - 1) / PPO_MINI_BATCH_SIZE))
if [ "${HPF_SUFFIX_KL_COEF}" != "0" ] && [ "${HPF_SUFFIX_KL_COEF}" != "0.0" ] \
    && [ "${HPF_LEADER_MINI_BATCHES}" -lt 2 ]; then
    echo "HPF suffix KL needs at least two leader mini-batches per phase; set PPO_MINI_BATCH_SIZE below TRAIN_BATCH_SIZE." >&2
    exit 1
fi
export PPO_MINI_BATCH_SIZE
# HPF smoke runs are used to validate exact continuation. Keep the native
# FSDP model, optimizer, and extra state in addition to the HF export.
SAVE_CONTENTS=${SAVE_CONTENTS:-${ACTOR_CHECKPOINT_SAVE_CONTENTS:-"['model','optimizer','extra','hf_model']"}}
export SAVE_CONTENTS

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
    +algorithm.hpf_rlvr.fresh_leader_tree="${HPF_FRESH_LEADER_TREE}"
    algorithm.hpf_rlvr.local_update_window.enable="${HPF_LOCAL_UPDATE_WINDOW}"
    algorithm.hpf_rlvr.local_update_window.size="${HPF_LOCAL_UPDATE_WINDOW_SIZE}"
    algorithm.hpf_rlvr.role_phased_training.enable="${HPF_ROLE_PHASED_TRAINING}"
    algorithm.hpf_rlvr.role_phased_training.follower_epochs="${HPF_FOLLOWER_PHASE_EPOCHS}"
    algorithm.hpf_rlvr.role_phased_training.leader_epochs="${HPF_LEADER_PHASE_EPOCHS}"
    algorithm.hpf_rlvr.mixed_policy_grpo.enable="${HPF_MIXED_POLICY_GRPO}"
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
        algorithm.hpf_rlvr.fresh_tree_rollout.num_prefixes="${HPF_FRESH_TREE_NUM_PREFIXES}"
        algorithm.hpf_rlvr.fresh_tree_rollout.num_suffixes="${HPF_FRESH_TREE_NUM_SUFFIXES}"
    )
fi

PROJECT_NAME="${PROJECT_NAME}" \
RUN_NAME="${RUN_NAME}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" \
bash examples/grpo_trainer/run_qwen2_5_math_7b_grpo_reschedule_baseline.sh \
    "${HPF_ARGS[@]}" \
    "$@"
