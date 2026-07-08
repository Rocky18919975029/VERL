#!/bin/bash
# Submit offline AIME pass@k eval jobs for GRPO and HPF checkpoints.
#
# The default configuration evaluates the aligned mini2048 seed42 GRPO/HPF
# checkpoints at steps 5, 6, and 7. Each checkpoint samples 16 responses per
# AIME problem at temperature=1.0, top_p=0.7, with the same seed.
#
# Usage from the repository root:
#   bash examples/generation/submit_aime_checkpoint_passk_eval.sh
#
# After all jobs finish:
#   python examples/generation/plot_aime_checkpoint_passk_eval.py --root "$OUTPUT_ROOT"

set -euo pipefail

VERL_WORKDIR=${VERL_WORKDIR:-/data/user/zhongal/VERL}
DATA_FILE=${DATA_FILE:-/data/user/zhongal/data/reschedule/aime24.parquet}
DATASET_NAME=${DATASET_NAME:-aime24_ckpt_eval}

GRPO_CKPT_ROOT=${GRPO_CKPT_ROOT:-${VERL_WORKDIR}/checkpoints/grpo_dapo_math17k_mini2048_seed42/grpo_n16_b512_mini2048_1epoch_seed42_20260707_223720}
HPF_CKPT_ROOT=${HPF_CKPT_ROOT:-${VERL_WORKDIR}/checkpoints/hpf_alg3_fresh_tree_mini2048_seed42/hpf_alg3_fresh_tree_4x4_b512_mini2048_1epoch_seed42_20260707_223720}

STEPS=(${STEPS:-5 6 7})
METHODS=(${METHODS:-grpo hpf_alg3})

TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.7}
N_RESPONSES=${N_RESPONSES:-16}
MAX_TOKENS=${MAX_TOKENS:-3072}
SEED=${SEED:-42}

NUM_SHARDS=${NUM_SHARDS:-8}
GPUS_PER_SHARD=${GPUS_PER_SHARD:-1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
DTYPE=${DTYPE:-bfloat16}
GPUS_PER_JOB=${GPUS_PER_JOB:-$((NUM_SHARDS * GPUS_PER_SHARD))}
CPUS_PER_JOB=${CPUS_PER_JOB:-$((GPUS_PER_JOB * 12))}
OUTPUT_ROOT=${OUTPUT_ROOT:-${VERL_WORKDIR}/outputs/aime_ckpt_passk_eval_$(date +%Y%m%d_%H%M%S)}

if [ "${TENSOR_PARALLEL_SIZE}" -ne "${GPUS_PER_SHARD}" ]; then
    echo "TENSOR_PARALLEL_SIZE must match GPUS_PER_SHARD for this launcher." >&2
    echo "Got TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}, GPUS_PER_SHARD=${GPUS_PER_SHARD}" >&2
    exit 1
fi

if [ "${GPUS_PER_JOB}" -ne $((NUM_SHARDS * GPUS_PER_SHARD)) ]; then
    echo "GPUS_PER_JOB must equal NUM_SHARDS * GPUS_PER_SHARD for this launcher." >&2
    echo "Got GPUS_PER_JOB=${GPUS_PER_JOB}, NUM_SHARDS=${NUM_SHARDS}, GPUS_PER_SHARD=${GPUS_PER_SHARD}" >&2
    exit 1
fi

if [ ! -f "${DATA_FILE}" ]; then
    echo "AIME parquet not found: ${DATA_FILE}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
MANIFEST="${OUTPUT_ROOT}/submitted_aime_ckpt_eval_jobs.tsv"
printf "method\tstep\tjob_id\tcheckpoint_hf\toutput_dir\n" > "${MANIFEST}"

echo "Submitting AIME checkpoint pass@k eval"
echo "AIME data: ${DATA_FILE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Methods: ${METHODS[*]}"
echo "Steps: ${STEPS[*]}"
echo "temperature=${TEMPERATURE} top_p=${TOP_P} n=${N_RESPONSES} seed=${SEED}"
echo "GPUs per job: ${GPUS_PER_JOB}"
echo "CPUs per job: ${CPUS_PER_JOB}"

ckpt_root_for_method() {
    case "$1" in
        grpo) echo "${GRPO_CKPT_ROOT}" ;;
        hpf|hpf_alg3) echo "${HPF_CKPT_ROOT}" ;;
        *)
            echo "Unknown method: $1" >&2
            return 1
            ;;
    esac
}

for method in "${METHODS[@]}"; do
    ckpt_root=$(ckpt_root_for_method "${method}")
    for step in "${STEPS[@]}"; do
        model_path="${ckpt_root}/global_step_${step}/actor/huggingface"
        output_dir="${OUTPUT_ROOT}/${method}/step_${step}"
        if [ ! -d "${model_path}" ]; then
            echo "Missing checkpoint HF model for ${method} step ${step}: ${model_path}" >&2
            exit 1
        fi
        job_id=$(
            sbatch --parsable \
                --job-name="aime-${method}-s${step}" \
                --gres="gpu:${GPUS_PER_JOB}" \
                --cpus-per-task="${CPUS_PER_JOB}" \
                --export=ALL,VERL_WORKDIR="${VERL_WORKDIR}",MODEL_PATH="${model_path}",DATA_FILE="${DATA_FILE}",DATASET_NAME="${DATASET_NAME}_${method}_step${step}",TEMPERATURE="${TEMPERATURE}",TOP_P="${TOP_P}",N_RESPONSES="${N_RESPONSES}",MAX_TOKENS="${MAX_TOKENS}",SEED="${SEED}",NUM_SHARDS="${NUM_SHARDS}",GPUS_PER_SHARD="${GPUS_PER_SHARD}",TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}",GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}",DTYPE="${DTYPE}",SKIP_LOGLIK_SCORING=1,OUTPUT_DIR="${output_dir}" \
                examples/generation/submit_math500_qwen25_7b_sample_score_h100.slurm
        )
        printf "%s\t%s\t%s\t%s\t%s\n" "${method}" "${step}" "${job_id}" "${model_path}" "${output_dir}" | tee -a "${MANIFEST}"
    done
done

echo "Submitted manifest: ${MANIFEST}"
echo "After jobs finish, run:"
echo "  python examples/generation/plot_aime_checkpoint_passk_eval.py --root \"${OUTPUT_ROOT}\""
