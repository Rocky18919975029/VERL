#!/bin/bash
# Evaluate one checkpoint on AIME with the training mixed-temperature policy.

#SBATCH --job-name=aime-mixed
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -xeuo pipefail

VERL_WORKDIR=${VERL_WORKDIR:-${SLURM_SUBMIT_DIR}}
CONDA_SH=${CONDA_SH:-/share/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV_PATH=${CONDA_ENV_PATH:-/data/user/zhongal/.conda/envs/verl}

module purge
module load gcc/13.3
module load cuda/12.8
source "${CONDA_SH}"
conda deactivate || true
conda activate "${CONDA_ENV_PATH}"
hash -r
cd "${VERL_WORKDIR}"

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${DATA_FILE:?DATA_FILE is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${CHECKPOINT_STEP:?CHECKPOINT_STEP is required}"
: "${HORIZON:?HORIZON is required}"

NUM_SHARDS=${NUM_SHARDS:-8}
GPUS_PER_SHARD=${GPUS_PER_SHARD:-1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
if [ $((NUM_SHARDS * GPUS_PER_SHARD)) -gt "${SLURM_GPUS_ON_NODE:-8}" ]; then
    echo "Requested more GPUs than allocated" >&2
    exit 1
fi
if [ "${TENSOR_PARALLEL_SIZE}" -ne "${GPUS_PER_SHARD}" ]; then
    echo "TENSOR_PARALLEL_SIZE must equal GPUS_PER_SHARD" >&2
    exit 1
fi

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
mkdir -p "${OUTPUT_DIR}"

echo "Mixed-policy AIME checkpoint evaluation"
echo "checkpoint_step=${CHECKPOINT_STEP} horizon=${HORIZON}"
echo "prefix: temperature=${PREFIX_TEMPERATURE:-1.0} top_p=${PREFIX_TOP_P:-1.0}"
echo "suffix: temperature=${SUFFIX_TEMPERATURE:-0.25} top_p=${SUFFIX_TOP_P:-1.0}"
echo "strict_box_verify=True base_seed=${SEED:-42} engine_seed=base_seed+shard_index"

pids=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    first_gpu=$((shard * GPUS_PER_SHARD))
    last_gpu=$((first_gpu + GPUS_PER_SHARD - 1))
    visible_devices=$(seq -s, "${first_gpu}" "${last_gpu}")
    shard_dir="${OUTPUT_DIR}/shard_${shard}"
    cache_dir="${OUTPUT_DIR}/cache/shard_${shard}"
    mkdir -p "${shard_dir}" "${cache_dir}/vllm" "${cache_dir}/torchinductor" "${cache_dir}/triton"
    CUDA_VISIBLE_DEVICES="${visible_devices}" \
        XDG_CACHE_HOME="${cache_dir}" \
        VLLM_CACHE_ROOT="${cache_dir}/vllm" \
        TORCHINDUCTOR_CACHE_DIR="${cache_dir}/torchinductor" \
        TRITON_CACHE_DIR="${cache_dir}/triton" \
        python examples/generation/aime_mixed_policy_checkpoint_eval.py \
            --model "${MODEL_PATH}" \
            --data "${DATA_FILE}" \
            --output-dir "${shard_dir}" \
            --checkpoint-step "${CHECKPOINT_STEP}" \
            --horizon "${HORIZON}" \
            --max-response-length "${MAX_RESPONSE_LENGTH:-3072}" \
            --prefix-temperature "${PREFIX_TEMPERATURE:-1.0}" \
            --prefix-top-p "${PREFIX_TOP_P:-1.0}" \
            --suffix-temperature "${SUFFIX_TEMPERATURE:-0.25}" \
            --suffix-top-p "${SUFFIX_TOP_P:-1.0}" \
            --seed "${SEED:-42}" \
            --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
            --dtype "${DTYPE:-bfloat16}" \
            --num-shards "${NUM_SHARDS}" \
            --shard-index "${shard}" \
            --enforce-eager &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
if [ "${status}" -ne 0 ]; then
    echo "At least one mixed-policy eval shard failed" >&2
    exit "${status}"
fi
echo "All shards completed: ${OUTPUT_DIR}"
