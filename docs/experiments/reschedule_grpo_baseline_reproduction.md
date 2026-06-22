# Re-Schedule Paper GRPO Baseline Reproduction

This note records how to reproduce the GRPO baseline from the paper
**"Scheduling Your LLM Reinforcement Learning with Reasoning Trees"** in this
repository, and records the paper's reported baseline results.

The target here is the **plain GRPO baseline only**. It does **not** enable the
paper's Re-Schedule method.

## Scope

Paper and code sources:

- Paper: `/Users/zeshenghong/Downloads/1388_Scheduling_Your_LLM_Reinf.pdf`
- Upstream experiment repository: `https://github.com/zz-haooo/Re-Schedule`
- Local training script:
  `examples/grpo_trainer/run_qwen2_5_math_7b_grpo_reschedule_baseline.sh`
- Local ACD Slurm launcher:
  `examples/grpo_trainer/submit_qwen2_5_math_7b_reschedule_baseline_h100.slurm`

The upstream Re-Schedule repository provides dynamic-weight scripts such as
`run/Re_Schedule_linear.sh` and `run/Re_Schedule_sigmoid.sh`. For the GRPO
baseline we intentionally remove these Re-Schedule-specific overrides:

```bash
+data.use_dynamic_weights
+data.metric_column
+data.alpha_mode
+data.dynamic_weights_*
```

## Paper Baseline Setup

The paper describes the baseline as standard GRPO implemented with VeRL. The
main math experiments use:

| Item | Value |
| --- | --- |
| Base models | `Qwen2.5-Math-7B`, `Qwen2.5-7B` |
| Training data | DAPO-Math-17k |
| Training data size in released repo | 17,398 rows |
| Rollout batch | 512 questions |
| Answers per question | 8 |
| Responses per rollout step | 4,096 |
| Policy mini-batches per rollout step | 16 |
| Policy gradient updates per rollout step | 16 |
| Learning rate | `1e-6` |
| Warmup | none |
| Max prompt length | 1024 |
| Max response length | 3072 |
| Training temperature | 1.0 |
| Training top-p | 1.0 |
| KL in reward | disabled |
| Actor KL loss | disabled |
| Entropy loss | disabled (`entropy_coeff=0`) |
| Max rollout steps | 150 |
| Validation temperature | 1.0 |
| Validation top-p | 0.7 |
| Reported evaluation metric | avg@32 with Math-Verify |

The upstream scripts set `trainer.total_epochs=50`, but the paper appendix says
models are trained for at most 150 rollout steps. For paper reproduction, this
repository explicitly sets:

```bash
trainer.total_training_steps=150
```

Thus the outer `Training Progress` bar should be `0/150 ... 150/150`. The
internal actor optimizer update count is `150 * 16 = 2400`.

## Paper-Reported Results

### Qwen2.5-Math-7B

| Method | AIME24 | AIME25 | AMC23 | MATH-500 | MinervaMath | OlympiadBench | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base model | 13.8 | 5.3 | 44.6 | 39.6 | 9.9 | 13.8 | 21.2 |
| GRPO baseline | 28.0 | 14.3 | 66.2 | 78.6 | 37.5 | 40.9 | 44.3 |

### Qwen2.5-7B

| Method | AIME24 | AIME25 | AMC23 | MATH-500 | MinervaMath | OlympiadBench | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base model | 5.1 | 2.5 | 27.8 | 34.4 | 5.9 | 13.5 | 14.9 |
| GRPO baseline | 15.6 | 8.8 | 62.5 | 78.2 | 38.6 | 40.4 | 40.7 |

The primary target for reproducing the paper's main Table 1 baseline is:

```text
Qwen2.5-Math-7B + DAPO-Math-17k + GRPO
```

## Required Files on ACD

Use local paths because ACD compute nodes do not have network access.

Expected model:

```text
/data/user/zhongal/.cache/qwen2.5-math-7b-local
```

Expected data directory:

```text
/data/user/zhongal/data/reschedule
```

Required data files:

```text
DAPO-Math-17k.parquet
aime24.parquet
aime25.parquet
amc23.parquet
math500.parquet
minerva_math.parquet
olympiadbench.parquet
```

The released Re-Schedule repository also contains `gsm8k_test.parquet` and
`omni.parquet`; they are not part of the paper's main six-dataset math table.

## Preparing Data and Model

Run on a network-enabled login node:

```bash
cd /data/user/zhongal

mkdir -p /data/user/zhongal/data
mkdir -p /data/user/zhongal/.cache

if [ ! -d /data/user/zhongal/Re-Schedule ]; then
  git clone https://github.com/zz-haooo/Re-Schedule.git /data/user/zhongal/Re-Schedule
fi

mkdir -p /data/user/zhongal/data/reschedule
cp -v /data/user/zhongal/Re-Schedule/datasets/*.parquet /data/user/zhongal/data/reschedule/

source /share/anaconda3/etc/profile.d/conda.sh
conda activate /data/user/zhongal/.conda/envs/verl
hash -r

huggingface-cli download Qwen/Qwen2.5-Math-7B \
  --local-dir /data/user/zhongal/.cache/qwen2.5-math-7b-local \
  --local-dir-use-symlinks False
```

Check:

```bash
ls -lh /data/user/zhongal/data/reschedule/DAPO-Math-17k.parquet
ls -lh /data/user/zhongal/data/reschedule/aime24.parquet
ls -lh /data/user/zhongal/.cache/qwen2.5-math-7b-local/config.json
ls -lh /data/user/zhongal/.cache/qwen2.5-math-7b-local/model*.safetensors
```

## Smoke Test

Use the Slurm launcher, not direct `bash`, from the ACD login node:

```bash
cd /data/user/zhongal/VERL
git pull --ff-only origin main

sbatch --export=ALL,MODEL_PATH=/data/user/zhongal/.cache/qwen2.5-math-7b-local,DATA_DIR=/data/user/zhongal/data/reschedule,TOTAL_TRAINING_STEPS=2 \
  examples/grpo_trainer/submit_qwen2_5_math_7b_reschedule_baseline_h100.slurm
```

Monitor:

```bash
squeue
tail -F slurm-verl-resched-grpo-<JOBID>.err
tail -F slurm-verl-resched-grpo-<JOBID>.out
```

Expected smoke behavior:

- `trainer.val_before_train=False`, so no pre-train validation.
- `trainer.test_freq=1`, so each of the 2 rollout steps runs AIME24 validation.
- The smoke test should finish at `Training Progress: 100%|...| 2/2`.
- Final validation metrics should include an AIME24 key similar to
  `val-core/aime_2024_dapo_boxed/acc/mean@1`.

Observed successful smoke run:

| Item | Value |
| --- | --- |
| Experiment | `qwen2_5_math_7b_grpo_baseline_361094` |
| Steps | 2 |
| Progress | `2/2` |
| Training time | `28:53` |
| Average step time | about `866.61s` / `14.44 min` |
| Checkpoint | `checkpoints/grpo_dapo_math17k_reschedule_baseline/qwen2_5_math_7b_grpo_baseline_361094/global_step_2/actor/huggingface` |
| Final AIME24 smoke accuracy | `val-core/aime_2024_dapo_boxed/acc/mean@1 = 0.0010416667` |

The smoke accuracy is expected to be poor because it uses only two rollout
steps. The purpose of the smoke test is to verify model loading, rollout,
reward, actor update, vLLM weight sync, checkpointing, validation, and wandb
offline logging.

## Full 150-Step Run

Submit the full baseline:

```bash
cd /data/user/zhongal/VERL
git pull --ff-only origin main

sbatch --export=ALL,MODEL_PATH=/data/user/zhongal/.cache/qwen2.5-math-7b-local,DATA_DIR=/data/user/zhongal/data/reschedule,TOTAL_TRAINING_STEPS=150 \
  examples/grpo_trainer/submit_qwen2_5_math_7b_reschedule_baseline_h100.slurm
```

Expected outer progress:

```text
Training Progress: 0/150 ... 150/150
```

Estimated runtime from the successful 2-step smoke:

```text
150 * 866.61s = about 36.1 hours
```

Plan for about 36-40 hours on the current 8xH100 ACD setup.

## Final Evaluation

After training, evaluate the final checkpoint on the six paper-reported math
sets through Slurm:

```bash
cd /data/user/zhongal/VERL

sbatch --export=ALL,MODE=eval,MODEL_PATH=/data/user/zhongal/VERL/checkpoints/grpo_dapo_math17k_reschedule_baseline/<EXPERIMENT_NAME>/global_step_150/actor/huggingface,DATA_DIR=/data/user/zhongal/data/reschedule \
  examples/grpo_trainer/submit_qwen2_5_math_7b_reschedule_baseline_h100.slurm
```

Only use direct `bash` if already inside an interactive GPU allocation:

```bash
MODEL_PATH=/data/user/zhongal/VERL/checkpoints/grpo_dapo_math17k_reschedule_baseline/<EXPERIMENT_NAME>/global_step_150/actor/huggingface \
DATA_DIR=/data/user/zhongal/data/reschedule \
MODE=eval \
bash examples/grpo_trainer/run_qwen2_5_math_7b_grpo_reschedule_baseline.sh
```

By default, `MODE=eval` evaluates the six paper-reported sets:

```text
aime24, aime25, amc23, math500, minerva_math, olympiadbench
```

To include the two extra datasets from the upstream eval script:

```bash
INCLUDE_REPO_EXTRA_EVAL=1
```

## Wandb Offline Logging

The launcher defaults to:

```bash
TRAINER_LOGGER='["console","wandb"]'
WANDB_MODE=offline
WANDB_DIR=/data/user/zhongal/VERL/wandb
```

After a run, local offline runs appear under:

```text
/data/user/zhongal/VERL/wandb/wandb/offline-run-*
```

Sync from a network-enabled node:

```bash
wandb sync /data/user/zhongal/VERL/wandb/wandb/offline-run-*
```

If file upload gets HTTP 403 errors, sync scalar curves only by excluding
problematic files:

```bash
wandb sync \
  --skip-console \
  --exclude-globs '**/requirements.txt,**/wandb-metadata.json,**/wandb-summary.json,**/config.yaml,**/output.log' \
  /data/user/zhongal/VERL/wandb/wandb/offline-run-<RUN_ID>
```

## ACD/H100 Engineering Notes

The paper/upstream script assumes the released placement:

```bash
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
actor_rollout_ref.ref.fsdp_config.param_offload=False
actor_rollout_ref.rollout.gpu_memory_utilization=0.6
```

On our ACD 8xH100 colocated setup, that placement OOMed at vLLM wake-up during
weight synchronization:

```text
update_weights -> rollout.resume(tags=["weights"]) -> vLLM wake_up
CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:62
```

The Slurm launcher therefore uses safer defaults:

```bash
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIMIZER_OFFLOAD=True
REF_PARAM_OFFLOAD=True
ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
```

This changes memory placement, not the GRPO algorithm, data, objective, or
training-step count.

The launcher also defaults to:

```bash
ATTN_IMPLEMENTATION=sdpa
```

because the installed `flash_attn` wheel on ACD requires `GLIBC_2.32` and fails
to import against the system runtime:

```text
ImportError: /lib64/libc.so.6: version `GLIBC_2.32' not found
```

Do not globally prepend `/data/apps/glibc/2.32/lib` to `LD_LIBRARY_PATH` for
this conda environment; it previously caused segmentation faults in mixed
conda/system processes.

## Useful Checks

Check the latest job:

```bash
ls -lt slurm-verl-resched-grpo-*.err | head
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList%30
```

Check checkpoint:

```bash
ls -R checkpoints/grpo_dapo_math17k_reschedule_baseline/<EXPERIMENT_NAME> | head -n 80
cat checkpoints/grpo_dapo_math17k_reschedule_baseline/<EXPERIMENT_NAME>/latest_checkpointed_iteration.txt
```

Watch GPU every 2 seconds from an active allocation/node:

```bash
watch -n 2 nvidia-smi
```
