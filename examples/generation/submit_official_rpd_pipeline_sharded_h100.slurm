#!/bin/bash
# Slurm launcher for one official RPD setting split across multiple in-job shards.
#
# This requests one multi-GPU allocation and starts one official RPD pipeline per
# shard. Each shard gets an isolated copy of the official RPD repository because
# the official scripts read parameters from config.py.

#SBATCH --job-name=rpd-official
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
##SBATCH --account=your-account

set -xeuo pipefail

VERL_WORKDIR=${VERL_WORKDIR:-${SLURM_SUBMIT_DIR}}

# Do not globally preload alternate glibc libraries into conda/Python.
LD_LIBRARY_PATH=":${LD_LIBRARY_PATH:-}:"
LD_LIBRARY_PATH=${LD_LIBRARY_PATH//:\/data\/apps\/glibc\/2.32\/lib:/:}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH//:\/data\/apps\/glibc\/2.32\/lib64:/:}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH#:}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH%:}
export LD_LIBRARY_PATH

module purge
module load gcc/13.3
module load cuda/12.8
module list

CONDA_SH=${CONDA_SH:-/share/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV_PATH=${CONDA_ENV_PATH:-/data/user/zhongal/.conda/envs/verl}
if [ -f "${CONDA_SH}" ]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
    conda deactivate || true
    conda activate "${CONDA_ENV_PATH}"
    hash -r
else
    echo "Conda init script not found: ${CONDA_SH}" >&2
    exit 1
fi

cd "${VERL_WORKDIR}"

python - <<'PY'
import sys
print("python:", sys.version)
try:
    import torch
except Exception as exc:
    raise SystemExit(f"failed to import torch: {exc}")
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("No CUDA device visible inside the Slurm allocation.")
print("gpu:", torch.cuda.get_device_name(0))
try:
    import vllm
    print("vllm:", getattr(vllm, "__version__", "unknown"))
except Exception as exc:
    raise SystemExit(f"failed to import vllm: {exc}")
try:
    import sentence_transformers
    print("sentence_transformers:", getattr(sentence_transformers, "__version__", "unknown"))
except Exception as exc:
    raise SystemExit(f"failed to import sentence_transformers: {exc}")
PY

INPUT_DIR=${INPUT_DIR:-}
RPD_REPO=${RPD_REPO:-/data/user/zhongal/external/Reasoning-Path-Divergence}
MODEL_PATH_INSTRUCT=${MODEL_PATH_INSTRUCT:-/data/user/zhongal/.cache/Qwen3-14B}
MODEL_PATH_EMBEDDING=${MODEL_PATH_EMBEDDING:-/data/user/zhongal/.cache/Qwen3-Embedding-8B}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/rpd_official_runs}
RUN_NAME=${RUN_NAME:-rpd_$(basename "${INPUT_DIR:-rollouts}")_${SLURM_JOB_ID}}
RPD_NUM_SHARDS=${RPD_NUM_SHARDS:-8}
RPD_GPUS_PER_SHARD=${RPD_GPUS_PER_SHARD:-1}
RPD_KEEP_WORK_REPOS=${RPD_KEEP_WORK_REPOS:-0}
RPD_WORK_ROOT=${RPD_WORK_ROOT:-${SLURM_TMPDIR:-${OUTPUT_ROOT}/_rpd_repo_copies/${RUN_NAME}_${SLURM_JOB_ID}}}

if [ -z "${INPUT_DIR}" ]; then
    echo "INPUT_DIR is required." >&2
    exit 2
fi
if [ ! -d "${INPUT_DIR}" ]; then
    echo "INPUT_DIR does not exist: ${INPUT_DIR}" >&2
    exit 2
fi
if [ ! -d "${RPD_REPO}" ]; then
    echo "RPD_REPO does not exist: ${RPD_REPO}" >&2
    exit 2
fi
if [ ! -d "${MODEL_PATH_INSTRUCT}" ]; then
    echo "MODEL_PATH_INSTRUCT does not exist: ${MODEL_PATH_INSTRUCT}" >&2
    exit 2
fi
if [ ! -d "${MODEL_PATH_EMBEDDING}" ]; then
    echo "MODEL_PATH_EMBEDDING does not exist: ${MODEL_PATH_EMBEDDING}" >&2
    exit 2
fi

VISIBLE_GPU_COUNT=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
if [ "${VISIBLE_GPU_COUNT}" -lt $((RPD_NUM_SHARDS * RPD_GPUS_PER_SHARD)) ]; then
    echo "Need $((RPD_NUM_SHARDS * RPD_GPUS_PER_SHARD)) visible GPUs, got ${VISIBLE_GPU_COUNT}." >&2
    exit 2
fi

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export CC=${CC:-$(which gcc)}
export CXX=${CXX:-$(which g++)}
export CUDAHOSTCXX=${CUDAHOSTCXX:-$(which g++)}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Working directory: ${VERL_WORKDIR}"
echo "Input dir: ${INPUT_DIR}"
echo "RPD repo: ${RPD_REPO}"
echo "Summary model: ${MODEL_PATH_INSTRUCT}"
echo "Embedding model: ${MODEL_PATH_EMBEDDING}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run name: ${RUN_NAME}"
echo "RPD_NUM_SHARDS: ${RPD_NUM_SHARDS}"
echo "RPD_GPUS_PER_SHARD: ${RPD_GPUS_PER_SHARD}"
echo "RPD_WORK_ROOT: ${RPD_WORK_ROOT}"
echo "TEST_LIMIT: ${TEST_LIMIT:-None}"
echo "RESPONSES_PER_PROBLEM: ${RESPONSES_PER_PROBLEM:-unset}"
echo "RUN_SUMMARY: ${RUN_SUMMARY:-1}"
echo "RUN_MATRIX: ${RUN_MATRIX:-1}"

mkdir -p "${RPD_WORK_ROOT}" "${OUTPUT_ROOT}/_sharded_job_logs/${RUN_NAME}"

pids=()
for shard in $(seq 0 $((RPD_NUM_SHARDS - 1))); do
    shard_run_name="${RUN_NAME}_shard${shard}of${RPD_NUM_SHARDS}"
    shard_repo="${RPD_WORK_ROOT}/rpd_repo_shard_${shard}"
    shard_log="${OUTPUT_ROOT}/_sharded_job_logs/${RUN_NAME}/shard_${shard}.log"
    first_gpu=$((shard * RPD_GPUS_PER_SHARD))
    last_gpu=$((first_gpu + RPD_GPUS_PER_SHARD - 1))
    gpu_list=$(seq -s, "${first_gpu}" "${last_gpu}")

    rm -rf "${shard_repo}"
    cp -a "${RPD_REPO}" "${shard_repo}"

    (
        echo "[rpd-shard] start shard=${shard}/${RPD_NUM_SHARDS} gpus=${gpu_list} run=${shard_run_name}"
        CUDA_VISIBLE_DEVICES="${gpu_list}" \
        INPUT_DIR="${INPUT_DIR}" \
        RPD_REPO="${shard_repo}" \
        MODEL_PATH_INSTRUCT="${MODEL_PATH_INSTRUCT}" \
        MODEL_PATH_EMBEDDING="${MODEL_PATH_EMBEDDING}" \
        OUTPUT_ROOT="${OUTPUT_ROOT}" \
        RUN_NAME="${shard_run_name}" \
        RESPONSES_PER_PROBLEM="${RESPONSES_PER_PROBLEM:-}" \
        MIN_RESPONSES="${MIN_RESPONSES:-2}" \
        MAX_ROWS_PER_FILE="${MAX_ROWS_PER_FILE:-10000}" \
        EXPORT_NUM_SHARDS="${RPD_NUM_SHARDS}" \
        EXPORT_SHARD_INDEX="${shard}" \
        TEST_LIMIT="${TEST_LIMIT:-None}" \
        RUN_SUMMARY="${RUN_SUMMARY:-1}" \
        RUN_MATRIX="${RUN_MATRIX:-1}" \
        SKIP_EXPORT="${SKIP_EXPORT:-0}" \
        OVERWRITE="${OVERWRITE:-0}" \
        bash examples/generation/run_official_rpd_pipeline.sh
        echo "[rpd-shard] done shard=${shard}/${RPD_NUM_SHARDS} run=${shard_run_name}"
    ) > "${shard_log}" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if [ "${RPD_KEEP_WORK_REPOS}" != "1" ]; then
    rm -rf "${RPD_WORK_ROOT}"
fi

exit "${status}"
