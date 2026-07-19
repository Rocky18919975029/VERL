#!/bin/bash
# Evaluate GRPO checkpoints on AIME at fixed temperatures with 32 samples/problem.
#
# The AIME parquet already contains 32 copies of each of 30 problems. Each
# source row therefore generates exactly one trajectory. The existing mixed-
# policy evaluator is reused with horizon=max_response_length, which makes the
# whole response use one fixed temperature and avoids an unnecessary suffix
# generation call.

set -euo pipefail

VERL_WORKDIR=${VERL_WORKDIR:-/data/user/zhongal/VERL}
CKPT_PROJECT_DIR=${CKPT_PROJECT_DIR:-${VERL_WORKDIR}/checkpoints/grpo_vllm_logprob_n32_mini1536_boxed_seed42}
CKPT_ROOT=${CKPT_ROOT:-}
DATA_FILE=${DATA_FILE:-/data/user/zhongal/data/reschedule/aime24.parquet}
OUTPUT_ROOT=${OUTPUT_ROOT:-${VERL_WORKDIR}/outputs/grpo_n32_aime_temperature_eval_$(date +%Y%m%d_%H%M%S)}

read -r -a EVAL_STEPS <<< "${STEPS:-1 2 3 4 5 6}"
read -r -a EVAL_TEMPERATURES <<< "${TEMPERATURES:-1.0 0.25}"
TOP_P=${TOP_P:-1.0}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
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
if [ "${#EVAL_STEPS[@]}" -eq 0 ] || [ "${#EVAL_TEMPERATURES[@]}" -eq 0 ]; then
    echo "STEPS and TEMPERATURES must both be non-empty." >&2
    exit 1
fi
if [ "${TENSOR_PARALLEL_SIZE}" -ne "${GPUS_PER_SHARD}" ]; then
    echo "TENSOR_PARALLEL_SIZE must equal GPUS_PER_SHARD." >&2
    exit 1
fi
if [ $((NUM_SHARDS * GPUS_PER_SHARD)) -gt 8 ]; then
    echo "NUM_SHARDS * GPUS_PER_SHARD cannot exceed 8." >&2
    exit 1
fi

if [ -z "${CKPT_ROOT}" ]; then
    if [ ! -d "${CKPT_PROJECT_DIR}" ]; then
        echo "Checkpoint project directory does not exist: ${CKPT_PROJECT_DIR}" >&2
        exit 1
    fi
    CKPT_ROOT=$(
        python - "${CKPT_PROJECT_DIR}" "${EVAL_STEPS[@]}" <<'PY'
from pathlib import Path
import sys

project = Path(sys.argv[1])
steps = [int(value) for value in sys.argv[2:]]
candidates = []
for run in project.iterdir():
    if not run.is_dir():
        continue
    complete = all(
        (run / f"global_step_{step}" / "actor" / "huggingface").is_dir()
        for step in steps
    )
    if complete:
        candidates.append(run)
if not candidates:
    raise SystemExit(
        f"No run under {project} contains Hugging Face checkpoints for steps {steps}."
    )
print(max(candidates, key=lambda path: path.stat().st_mtime))
PY
    )
fi
if [ ! -d "${CKPT_ROOT}" ]; then
    echo "Checkpoint root does not exist: ${CKPT_ROOT}" >&2
    exit 1
fi

row_count=$(python - "${DATA_FILE}" <<'PY'
import sys
import pyarrow.parquet as pq

print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)
PY
)
if [ "${row_count}" -ne 960 ]; then
    echo "Expected the repeated AIME parquet to contain 960 rows, found ${row_count}." >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
manifest="${OUTPUT_ROOT}/submitted_grpo_aime_temperature_eval_jobs.tsv"
printf "temperature\ttemperature_tag\tstep\tjob_id\tcheckpoint_hf\toutput_dir\ttop_p\tseed\n" > "${manifest}"

echo "GRPO AIME fixed-temperature checkpoint evaluation"
echo "Checkpoint root: ${CKPT_ROOT}"
echo "Steps: ${EVAL_STEPS[*]}"
echo "Temperatures: ${EVAL_TEMPERATURES[*]}"
echo "top_p=${TOP_P} max_response_length=${MAX_RESPONSE_LENGTH} seed=${SEED}"
echo "AIME: 30 problems x 32 source repetitions = 960 trajectories/setting"
echo "Verifier: math_dapo strict_box_verify=True"
echo "Shards: ${NUM_SHARDS}; GPUs/shard: ${GPUS_PER_SHARD}"
echo "Output root: ${OUTPUT_ROOT}"

for temperature in "${EVAL_TEMPERATURES[@]}"; do
    temperature_tag=$(echo "${temperature}" | tr '.' 'p')
    for step in "${EVAL_STEPS[@]}"; do
        if ! [[ "${step}" =~ ^[0-9]+$ ]] || [ "${step}" -lt 1 ]; then
            echo "Invalid checkpoint step: ${step}" >&2
            exit 1
        fi
        model_path="${CKPT_ROOT}/global_step_${step}/actor/huggingface"
        output_dir="${OUTPUT_ROOT}/temperature_${temperature_tag}/step_${step}"
        if [ ! -d "${model_path}" ]; then
            echo "Missing Hugging Face checkpoint: ${model_path}" >&2
            exit 1
        fi

        if [ "${DRY_RUN}" = "1" ]; then
            job_id="DRY_RUN"
        else
            job_id=$(
                sbatch --parsable \
                    --job-name="aime-g-t${temperature_tag}-s${step}" \
                    --gres="gpu:$((NUM_SHARDS * GPUS_PER_SHARD))" \
                    --export=ALL,VERL_WORKDIR="${VERL_WORKDIR}",MODEL_PATH="${model_path}",DATA_FILE="${DATA_FILE}",OUTPUT_DIR="${output_dir}",CHECKPOINT_STEP="${step}",HORIZON="${MAX_RESPONSE_LENGTH}",MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}",PREFIX_TEMPERATURE="${temperature}",PREFIX_TOP_P="${TOP_P}",SUFFIX_TEMPERATURE="${temperature}",SUFFIX_TOP_P="${TOP_P}",SEED="${SEED}",NUM_SHARDS="${NUM_SHARDS}",GPUS_PER_SHARD="${GPUS_PER_SHARD}",TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}",GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}",DTYPE="${DTYPE}" \
                    examples/generation/submit_aime_mixed_policy_checkpoint_eval.slurm
            )
        fi
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${temperature}" "${temperature_tag}" "${step}" "${job_id}" \
            "${model_path}" "${output_dir}" "${TOP_P}" "${SEED}" \
            | tee -a "${manifest}"
    done
done

echo "Manifest: ${manifest}"
echo "After all jobs finish:"
echo "  python examples/generation/plot_grpo_aime_temperature_eval.py --root \"${OUTPUT_ROOT}\" --data \"${DATA_FILE}\""
