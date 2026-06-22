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

PROJECT_NAME="${PROJECT_NAME}" \
RUN_NAME="${RUN_NAME}" \
bash examples/grpo_trainer/run_qwen2_5_math_7b_grpo_reschedule_baseline.sh \
    algorithm.hpf_rlvr.enable=True \
    algorithm.hpf_rlvr.progressive_block_size="${HPF_PROGRESSIVE_BLOCK_SIZE}" \
    algorithm.hpf_rlvr.max_response_length="${HPF_MAX_RESPONSE_LENGTH}" \
    algorithm.hpf_rlvr.epsilon="${HPF_EPSILON}" \
    algorithm.hpf_rlvr.std_normalize="${HPF_STD_NORMALIZE}" \
    "$@"
