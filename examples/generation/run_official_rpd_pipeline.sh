#!/usr/bin/env bash
# Run the official Reasoning-Path-Divergence Step 4/5 pipeline on VERL rollouts.
#
# This script keeps the official RPD implementation intact. It only:
#   1. exports our rollout parquet files to the official processed parquet format;
#   2. temporarily points the official repo's config.py at this run directory;
#   3. runs official 04_generate_summary.py and 05_compute_matrix.py;
#   4. records logs, config, manifest, and the official repo commit.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  INPUT_DIR=/path/to/verl/rollout_dir \
  RPD_REPO=/data/user/zhongal/external/Reasoning-Path-Divergence \
  MODEL_PATH_INSTRUCT=/data/user/zhongal/.cache/Qwen3-14B \
  MODEL_PATH_EMBEDDING=/data/user/zhongal/.cache/Qwen3-Embedding-8B \
  bash examples/generation/run_official_rpd_pipeline.sh

Required:
  INPUT_DIR                 VERL rollout/loglik output directory.

Common optional env vars:
  RPD_REPO                  Official RPD repository path.
                            Default: /data/user/zhongal/external/Reasoning-Path-Divergence
  MODEL_PATH_INSTRUCT       Summary model path. Default: /data/user/zhongal/.cache/Qwen3-14B
  MODEL_PATH_EMBEDDING      Embedding model path. Default: /data/user/zhongal/.cache/Qwen3-Embedding-8B
  OUTPUT_ROOT               Parent output dir. Default: outputs/rpd_official_runs
  RUN_NAME                  Run folder name. Default: rpd_<input-basename>_<timestamp>
  RESPONSES_PER_PROBLEM     Optional cap passed to the adapter. Default: unset
  MIN_RESPONSES             Drop problems with fewer exported responses. Default: 2
  TEST_LIMIT                Official config TEST_LIMIT. Default: None
  RUN_SUMMARY               Run official 04_generate_summary.py. Default: 1
  RUN_MATRIX                Run official 05_compute_matrix.py. Default: 1
  SKIP_EXPORT               Reuse existing processed parquet files. Default: 0
  OVERWRITE                 Allow reusing an existing RUN_DIR. Default: 0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

VERL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${VERL_ROOT}"

INPUT_DIR="${INPUT_DIR:-}"
RPD_REPO="${RPD_REPO:-/data/user/zhongal/external/Reasoning-Path-Divergence}"
MODEL_PATH_INSTRUCT="${MODEL_PATH_INSTRUCT:-/data/user/zhongal/.cache/Qwen3-14B}"
MODEL_PATH_EMBEDDING="${MODEL_PATH_EMBEDDING:-/data/user/zhongal/.cache/Qwen3-Embedding-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rpd_official_runs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-rpd_$(basename "${INPUT_DIR:-rollouts}")_${STAMP}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
PROCESSED_DIR="${RUN_DIR}/processed"
LOG_DIR="${RUN_DIR}/logs"
RESPONSES_PER_PROBLEM="${RESPONSES_PER_PROBLEM:-}"
MIN_RESPONSES="${MIN_RESPONSES:-2}"
MAX_ROWS_PER_FILE="${MAX_ROWS_PER_FILE:-10000}"
TEST_LIMIT="${TEST_LIMIT:-None}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
RUN_MATRIX="${RUN_MATRIX:-1}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
OVERWRITE="${OVERWRITE:-0}"

if [[ -z "${INPUT_DIR}" ]]; then
  echo "ERROR: INPUT_DIR is required." >&2
  usage >&2
  exit 2
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "ERROR: INPUT_DIR does not exist: ${INPUT_DIR}" >&2
  exit 2
fi
if [[ ! -d "${RPD_REPO}" ]]; then
  echo "ERROR: RPD_REPO does not exist: ${RPD_REPO}" >&2
  echo "Clone it first, for example:" >&2
  echo "  git clone https://github.com/fengjujf/Reasoning-Path-Divergence.git ${RPD_REPO}" >&2
  exit 2
fi
if [[ ! -f "${RPD_REPO}/04_generate_summary.py" || ! -f "${RPD_REPO}/05_compute_matrix.py" ]]; then
  echo "ERROR: RPD_REPO does not look like the official RPD repository: ${RPD_REPO}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH_INSTRUCT}" ]]; then
  echo "ERROR: summary model path does not exist: ${MODEL_PATH_INSTRUCT}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH_EMBEDDING}" ]]; then
  echo "ERROR: embedding model path does not exist: ${MODEL_PATH_EMBEDDING}" >&2
  exit 2
fi

if [[ -e "${RUN_DIR}" && "${OVERWRITE}" != "1" ]]; then
  echo "ERROR: RUN_DIR already exists: ${RUN_DIR}" >&2
  echo "Set OVERWRITE=1 or choose a new RUN_NAME/RUN_DIR." >&2
  exit 2
fi

mkdir -p "${PROCESSED_DIR}" "${LOG_DIR}"

RPD_COMMIT="$(
  cd "${RPD_REPO}"
  git rev-parse HEAD 2>/dev/null || echo unknown
)"
VERL_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

cat > "${RUN_DIR}/rpd_pipeline_config.json" <<EOF
{
  "input_dir": "${INPUT_DIR}",
  "run_dir": "${RUN_DIR}",
  "processed_dir": "${PROCESSED_DIR}",
  "rpd_repo": "${RPD_REPO}",
  "rpd_commit": "${RPD_COMMIT}",
  "verl_root": "${VERL_ROOT}",
  "verl_commit": "${VERL_COMMIT}",
  "model_path_instruct": "${MODEL_PATH_INSTRUCT}",
  "model_path_embedding": "${MODEL_PATH_EMBEDDING}",
  "responses_per_problem": "${RESPONSES_PER_PROBLEM}",
  "min_responses": ${MIN_RESPONSES},
  "max_rows_per_file": ${MAX_ROWS_PER_FILE},
  "test_limit": "${TEST_LIMIT}",
  "run_summary": "${RUN_SUMMARY}",
  "run_matrix": "${RUN_MATRIX}",
  "skip_export": "${SKIP_EXPORT}"
}
EOF

echo "============================================================"
echo "Official RPD pipeline"
echo "Input:       ${INPUT_DIR}"
echo "Run dir:     ${RUN_DIR}"
echo "RPD repo:    ${RPD_REPO}"
echo "RPD commit:  ${RPD_COMMIT}"
echo "VERL commit: ${VERL_COMMIT}"
echo "============================================================"

if [[ "${SKIP_EXPORT}" != "1" ]]; then
  ADAPTER_ARGS=(
    examples/generation/export_rollouts_to_rpd_format.py
    --input-dir "${INPUT_DIR}"
    --output-dir "${PROCESSED_DIR}"
    --min-responses "${MIN_RESPONSES}"
    --max-rows-per-file "${MAX_ROWS_PER_FILE}"
  )
  if [[ -n "${RESPONSES_PER_PROBLEM}" ]]; then
    ADAPTER_ARGS+=(--responses-per-problem "${RESPONSES_PER_PROBLEM}")
  fi

  echo "[1/4] Exporting VERL rollouts to official RPD input format..."
  python "${ADAPTER_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/01_export_rollouts.log"
else
  echo "[1/4] SKIP_EXPORT=1, reusing processed parquet files in ${PROCESSED_DIR}"
fi

CONFIG_PATH="${RPD_REPO}/config.py"
CONFIG_BACKUP="${LOG_DIR}/official_config.py.before"
cp "${CONFIG_PATH}" "${CONFIG_BACKUP}"

restore_config() {
  if [[ -f "${CONFIG_BACKUP}" ]]; then
    cp "${CONFIG_BACKUP}" "${CONFIG_PATH}"
  fi
}
trap restore_config EXIT

cat > "${CONFIG_PATH}" <<EOF
import os

SOURCE_DATASET_REPO = "open-thoughts/OpenThoughts3-1.2M"
MODEL_PATH_INSTRUCT = "${MODEL_PATH_INSTRUCT}"
MODEL_PATH_EMBEDDING = "${MODEL_PATH_EMBEDDING}"
SOURCE_FILENAME_FMT = "data/train-{index:05d}-of-00120.parquet"

START_FILE_INDEX = 25
END_FILE_INDEX = 109

BASE_DATA_DIR = "${RUN_DIR}"
RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, "raw_source")
PROCESSED_DATA_DIR = "${PROCESSED_DIR}"

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

TARGET_ROWS_PER_FILE = 10000
MAX_AVG_TOKENS = 14000
MIN_HIGH_QUALITY_ANSWERS = 10

DIR_STEP4_SUMMARY = os.path.join(PROCESSED_DATA_DIR, "summaries")
os.makedirs(DIR_STEP4_SUMMARY, exist_ok=True)

OUTPUT_DIR = PROCESSED_DATA_DIR
FILE_STEP5_MATRIX = "05_distance_matrix.npz"
FILE_STEP6_SELECTED_Q = "06_selected_questions.json"
FILE_STEP7_FINAL = "07_final_dataset.json"

FINAL_SELECTION_COUNT = 100
ANSWERS_PER_QUESTION = 3

TEST_LIMIT = ${TEST_LIMIT}
EOF
cp "${CONFIG_PATH}" "${RUN_DIR}/official_config.py.used"

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  echo "[2/4] Running official 04_generate_summary.py..."
  (
    cd "${RPD_REPO}"
    python 04_generate_summary.py
  ) 2>&1 | tee "${LOG_DIR}/04_generate_summary.log"
else
  echo "[2/4] RUN_SUMMARY=0, skipping official summary generation."
fi

if [[ "${RUN_MATRIX}" == "1" ]]; then
  echo "[3/4] Running official 05_compute_matrix.py..."
  (
    cd "${RPD_REPO}"
    python 05_compute_matrix.py
  ) 2>&1 | tee "${LOG_DIR}/05_compute_matrix.log"
else
  echo "[3/4] RUN_MATRIX=0, skipping official distance matrix computation."
fi

echo "[4/4] Recording output manifest..."
python - <<PY
import json
from pathlib import Path

run_dir = Path("${RUN_DIR}")
processed_dir = Path("${PROCESSED_DIR}")
summary_dir = processed_dir / "summaries"
matrix_path = processed_dir / "05_distance_matrix.npz"

record = {
    "run_dir": str(run_dir),
    "processed_dir": str(processed_dir),
    "processed_parquet_files": sorted(str(p) for p in processed_dir.glob("quality_filtered_data_*.parquet")),
    "num_summary_json": len(list(summary_dir.glob("*.json"))) if summary_dir.exists() else 0,
    "distance_matrix": str(matrix_path) if matrix_path.exists() else None,
    "logs": sorted(str(p) for p in (run_dir / "logs").glob("*.log")),
    "config": str(run_dir / "rpd_pipeline_config.json"),
    "official_config_used": str(run_dir / "official_config.py.used"),
}
(run_dir / "rpd_pipeline_outputs.json").write_text(
    json.dumps(record, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(record, indent=2, ensure_ascii=False))
PY

echo "Done. Results are under: ${RUN_DIR}"
