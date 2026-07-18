#!/bin/bash
# Auto-discover lambda=0.5 checkpoints and evaluate each at its training cut.

set -euo pipefail

VERL_WORKDIR=${VERL_WORKDIR:-/data/user/zhongal/VERL}
PROJECT_NAME=${PROJECT_NAME:-hpf_transition_aware_fulltail_k16_lambda0p5_mini1536_boxed_seed42}
RUN_NAME=${RUN_NAME:-transition_aware_fulltail_k16_lambda0p5_b512_mini1536_boxed_2epoch_seed42_20260717_053640}
CKPT_ROOT=${CKPT_ROOT:-${VERL_WORKDIR}/checkpoints/${PROJECT_NAME}/${RUN_NAME}}
DATA_FILE=${DATA_FILE:-/data/user/zhongal/data/reschedule/aime24.parquet}
OUTPUT_ROOT=${OUTPUT_ROOT:-${VERL_WORKDIR}/outputs/lambda0p5_aime_mixed_policy_eval_$(date +%Y%m%d_%H%M%S)}

STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-3}
BLOCK_SIZE=${BLOCK_SIZE:-64}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
PREFIX_TEMPERATURE=${PREFIX_TEMPERATURE:-1.0}
PREFIX_TOP_P=${PREFIX_TOP_P:-1.0}
SUFFIX_TEMPERATURE=${SUFFIX_TEMPERATURE:-0.25}
SUFFIX_TOP_P=${SUFFIX_TOP_P:-1.0}
SEED=${SEED:-42}
NUM_SHARDS=${NUM_SHARDS:-8}
GPUS_PER_SHARD=${GPUS_PER_SHARD:-1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
DTYPE=${DTYPE:-bfloat16}
DRY_RUN=${DRY_RUN:-0}

if [ ! -f "${DATA_FILE}" ]; then
    echo "AIME parquet does not exist: ${DATA_FILE}" >&2
    exit 1
fi
if [ ! -d "${CKPT_ROOT}" ]; then
    echo "Checkpoint root does not exist: ${CKPT_ROOT}" >&2
    exit 1
fi

if [ -n "${STEPS:-}" ]; then
    read -r -a discovered_steps <<< "${STEPS}"
else
    discovered_steps=()
    while IFS= read -r step; do
        discovered_steps+=("${step}")
    done < <(
        find "${CKPT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' -print \
            | sed 's#.*global_step_##' \
            | sort -n
    )
fi
if [ "${#discovered_steps[@]}" -eq 0 ]; then
    echo "No global_step_* checkpoints found under ${CKPT_ROOT}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
manifest="${OUTPUT_ROOT}/submitted_mixed_policy_eval_jobs.tsv"
printf "step\tround\thorizon\tjob_id\tcheckpoint_hf\toutput_dir\tprefix_temperature\tprefix_top_p\tsuffix_temperature\tsuffix_top_p\tseed\n" > "${manifest}"

echo "Lambda=0.5 AIME mixed-policy checkpoint evaluation"
echo "Checkpoint root: ${CKPT_ROOT}"
echo "Steps: ${discovered_steps[*]}"
echo "Cut schedule: steps_per_epoch=${STEPS_PER_EPOCH}, block_size=${BLOCK_SIZE}"
echo "Policy: prefix T=${PREFIX_TEMPERATURE} top_p=${PREFIX_TOP_P}; suffix T=${SUFFIX_TEMPERATURE} top_p=${SUFFIX_TOP_P}"
echo "Verifier: math_dapo strict_box_verify=True"
echo "Seed mapping: vLLM-native base_seed+replica_rank (${SEED}..$((SEED + NUM_SHARDS - 1)))"
echo "AIME rows are preserved; expected aggregate is 30 problems x 32 samples"
echo "Output root: ${OUTPUT_ROOT}"

for step in "${discovered_steps[@]}"; do
    if ! [[ "${step}" =~ ^[0-9]+$ ]] || [ "${step}" -lt 1 ]; then
        echo "Invalid step: ${step}" >&2
        exit 1
    fi
    round=$(( (step - 1) / STEPS_PER_EPOCH + 1 ))
    horizon=$(( round * BLOCK_SIZE ))
    if [ "${horizon}" -gt "${MAX_RESPONSE_LENGTH}" ]; then
        horizon=${MAX_RESPONSE_LENGTH}
    fi
    model_path="${CKPT_ROOT}/global_step_${step}/actor/huggingface"
    output_dir="${OUTPUT_ROOT}/lambda0p5/step_${step}"
    if [ ! -d "${model_path}" ]; then
        echo "Missing Hugging Face checkpoint: ${model_path}" >&2
        exit 1
    fi

    if [ "${DRY_RUN}" = "1" ]; then
        job_id="DRY_RUN"
    else
        job_id=$(
            sbatch --parsable \
                --job-name="aime-mix-s${step}" \
                --export=ALL,VERL_WORKDIR="${VERL_WORKDIR}",MODEL_PATH="${model_path}",DATA_FILE="${DATA_FILE}",OUTPUT_DIR="${output_dir}",CHECKPOINT_STEP="${step}",HORIZON="${horizon}",MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}",PREFIX_TEMPERATURE="${PREFIX_TEMPERATURE}",PREFIX_TOP_P="${PREFIX_TOP_P}",SUFFIX_TEMPERATURE="${SUFFIX_TEMPERATURE}",SUFFIX_TOP_P="${SUFFIX_TOP_P}",SEED="${SEED}",NUM_SHARDS="${NUM_SHARDS}",GPUS_PER_SHARD="${GPUS_PER_SHARD}",TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}",GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}",DTYPE="${DTYPE}" \
                examples/generation/submit_aime_mixed_policy_checkpoint_eval.slurm
        )
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${step}" "${round}" "${horizon}" "${job_id}" "${model_path}" "${output_dir}" \
        "${PREFIX_TEMPERATURE}" "${PREFIX_TOP_P}" "${SUFFIX_TEMPERATURE}" "${SUFFIX_TOP_P}" "${SEED}" \
        | tee -a "${manifest}"
done

echo "Manifest: ${manifest}"
echo "After all jobs finish:"
echo "  python examples/generation/plot_aime_mixed_policy_checkpoint_eval.py --root \"${OUTPUT_ROOT}\""
