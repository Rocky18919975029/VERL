# Using the Official RPD Repository on VERL Rollouts

This note keeps the RPD computation as close as possible to the official
`fengjujf/Reasoning-Path-Divergence` repository. VERL only provides an adapter
that reshapes rollout parquet files into the official Step 4 input format.

## 1. Export VERL rollouts

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

