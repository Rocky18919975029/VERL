# Using the Official RPD Repository on VERL Rollouts

This note keeps the RPD computation as close as possible to the official
`fengjujf/Reasoning-Path-Divergence` repository. VERL only provides an adapter
that reshapes rollout parquet files into the official Step 4 input format.

## One-command official pipeline on an H100 node

The preferred path is to submit the Slurm launcher below. It requests one H100
node, exports VERL rollout parquet files, temporarily points the official RPD
repository's `config.py` at a run-specific output directory, runs the official
Step 4 and Step 5 scripts, and then restores the original official `config.py`.

```bash
cd /data/user/zhongal/VERL

INPUT_DIR=outputs/dapo_rollout_blocksize_matrix_20260701_203658/tree/seed_42/block_64/leader_temp_1p0 \
RPD_REPO=/data/user/zhongal/external/Reasoning-Path-Divergence \
MODEL_PATH_INSTRUCT=/data/user/zhongal/.cache/Qwen3-14B \
MODEL_PATH_EMBEDDING=/data/user/zhongal/.cache/Qwen3-Embedding-8B \
RUN_NAME=rpd_tree_seed42_block64 \
sbatch examples/generation/submit_official_rpd_pipeline_h100.slurm
```

For a very small smoke test:

```bash
INPUT_DIR=outputs/dapo_rollout_blocksize_matrix_20260701_203658/tree/seed_42/block_64/leader_temp_1p0 \
RUN_NAME=rpd_smoke_tree_seed42_block64 \
TEST_LIMIT=2 \
sbatch examples/generation/submit_official_rpd_pipeline_h100.slurm
```

If already inside an interactive GPU allocation, the underlying wrapper can be
run directly:

```bash
INPUT_DIR=outputs/dapo_rollout_blocksize_matrix_20260701_203658/tree/seed_42/block_64/leader_temp_1p0 \
RPD_REPO=/data/user/zhongal/external/Reasoning-Path-Divergence \
MODEL_PATH_INSTRUCT=/data/user/zhongal/.cache/Qwen3-14B \
MODEL_PATH_EMBEDDING=/data/user/zhongal/.cache/Qwen3-Embedding-8B \
RUN_NAME=rpd_tree_seed42_block64 \
bash examples/generation/run_official_rpd_pipeline.sh
```

The run writes a self-contained record under:

```text
outputs/rpd_official_runs/rpd_tree_seed42_block64/
```

Important files:

```text
processed/quality_filtered_data_00000.parquet
processed/rpd_rollout_manifest.json
processed/rpd_export_summary.json
processed/summaries/*.json
processed/05_distance_matrix.npz
logs/01_export_rollouts.log
logs/04_generate_summary.log
logs/05_compute_matrix.log
rpd_pipeline_config.json
rpd_pipeline_outputs.json
official_config.py.used
```

Optional controls:

```bash
RESPONSES_PER_PROBLEM=8      # cap responses per problem before RPD
MIN_RESPONSES=2              # drop problems with too few responses
TEST_LIMIT=10                # pass through to official config.TEST_LIMIT
RUN_SUMMARY=0                # skip official Step 4 if summaries already exist
RUN_MATRIX=0                 # skip official Step 5
SKIP_EXPORT=1                # reuse existing processed parquet files
OVERWRITE=1                  # reuse an existing RUN_DIR
```

## Manual path

### 1. Export VERL rollouts

```bash
cd /data/user/zhongal/VERL

python examples/generation/export_rollouts_to_rpd_format.py \
  --input-dir outputs/dapo_rollout_blocksize_matrix_20260701_203658/tree/seed_42/block_64/leader_temp_1p0 \
  --output-dir /path/to/Reasoning-Path-Divergence/data/processed
```

The adapter writes:

```text
quality_filtered_data_00000.parquet
rpd_rollout_manifest.json
rpd_export_summary.json
```

The parquet has one row per problem and columns:

```text
question, problem_index, ground_truth, answer_0, answer_1, ...
```

This is the format consumed by the official `04_generate_summary.py`.

## 2. Point the official RPD repo at local models

In the official RPD repository, edit `config.py`:

```python
MODEL_PATH_INSTRUCT = "/path/to/Qwen3-14B"
MODEL_PATH_EMBEDDING = "/path/to/Qwen3-Embedding-8B"
PROCESSED_DATA_DIR = "/path/to/Reasoning-Path-Divergence/data/processed"
DIR_STEP4_SUMMARY = os.path.join(PROCESSED_DATA_DIR, "summaries")
OUTPUT_DIR = PROCESSED_DATA_DIR
TEST_LIMIT = None
```

## 3. Run the official RPD stages

From the official RPD repository:

```bash
python 04_generate_summary.py
python 05_compute_matrix.py
```

The official Step 5 output is:

```text
data/processed/05_distance_matrix.npz
```

It stores one pairwise RPD matrix per exported problem:

```text
q_<rpd_global_idx>_matrix
q_<rpd_global_idx>_ids
```

Use `rpd_rollout_manifest.json` to map `rpd_global_idx` back to VERL
`problem_index`.
