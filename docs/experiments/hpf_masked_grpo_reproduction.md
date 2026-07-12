# HPF Masked-GRPO Reproduction Runbook

This document records how to run the HPF masked-GRPO experiments used in our
internal reproduction and smoke tests. The HPF entrypoint intentionally wraps
the already validated Re-Schedule GRPO baseline script and appends only
HPF-specific overrides, so the baseline reproduction path remains unchanged.

## Scope

The relevant scripts are:

```text
examples/grpo_trainer/run_qwen2_5_math_7b_hpf_masked_grpo.sh
examples/grpo_trainer/submit_qwen2_5_math_7b_hpf_masked_grpo_h100.slurm
examples/grpo_trainer/run_qwen2_5_math_7b_grpo_reschedule_baseline.sh
examples/generation/diagnose_hpf_wandb_run.py
```

The Slurm launcher is the recommended entrypoint on the ACD H100 cluster. It
runs on one node with eight H100 GPUs.

## Environment

Submit all jobs from the VERL repository root:

```bash
cd /data/user/zhongal/VERL
```

The Slurm launcher loads the expected compiler and CUDA modules, activates the
`verl` conda environment, and removes the problematic `glibc/2.32` library path
from `LD_LIBRARY_PATH`:

```text
module purge
module load gcc/13.3
module load cuda/12.8
conda activate /data/user/zhongal/.conda/envs/verl
```

Expected local paths:

```text
MODEL_PATH=/data/user/zhongal/.cache/qwen2.5-math-7b-local
DATA_DIR=/data/user/zhongal/data/reschedule
TRAIN_FILE=/data/user/zhongal/data/reschedule/DAPO-Math-17k.parquet
```

The launcher also expects the validation parquet files below to exist:

```text
aime24.parquet
aime25.parquet
amc23.parquet
math500.parquet
minerva_math.parquet
olympiadbench.parquet
```

W&B is run offline by default:

```text
WANDB_MODE=offline
WANDB_DIR=/data/user/zhongal/VERL/wandb
```

## Default HPF Configuration

The HPF wrapper inherits the Re-Schedule GRPO baseline settings and appends:

```text
algorithm.hpf_rlvr.enable=True
algorithm.hpf_rlvr.progressive_block_size=256
algorithm.hpf_rlvr.max_response_length=3072
algorithm.hpf_rlvr.std_normalize=True
algorithm.hpf_rlvr.horizon_schedule=epoch
algorithm.hpf_rlvr.horizon_update_interval_steps=1
algorithm.hpf_rlvr.prefix_kl_coef=0.001
algorithm.hpf_rlvr.suffix_kl_coef=0.001
algorithm.hpf_rlvr.correction_clip=5.0
algorithm.hpf_rlvr.progress_log_interval=1
actor_rollout_ref.actor.loss_agg_mode=token-mean
```

Tree rollout is optional and should be enabled for the HPF method:

```text
HPF_TREE_ROLLOUT=True
HPF_TREE_NUM_PREFIXES=4
HPF_TREE_NUM_SUFFIXES=2
HPF_TREE_PREFIX_TEMPERATURE=1.0
HPF_TREE_PREFIX_TOP_P=1.0
HPF_TREE_SUFFIX_TEMPERATURE=0.25
HPF_TREE_SUFFIX_TOP_P=1.0
```

The wrapper sets `ROLLOUT_N = HPF_TREE_NUM_PREFIXES * HPF_TREE_NUM_SUFFIXES`
when tree rollout is enabled.

The training launcher uses memory-safe defaults for one 8xH100 node:

```text
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIMIZER_OFFLOAD=True
REF_PARAM_OFFLOAD=True
ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
ROLLOUT_TP=4
TRAIN_BATCH_SIZE=512
PPO_MINI_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE_PER_GPU=8
```

For exact resume tests, actor checkpoints save FSDP model shards, optimizer
state, extra state, and Hugging Face export:

```text
SAVE_CONTENTS=['model','optimizer','extra','hf_model']
```

## Smoke Test: One Step With Checkpoint State

Use this to validate that rollout, follower update, leader update, W&B logging,
and checkpoint writing all work.

```bash
RUN_NAME=hpf_one_step_ckpt_$(date +%Y%m%d_%H%M%S)

JOB=$(sbatch --parsable \
  --export=ALL,\
MODEL_PATH=/data/user/zhongal/.cache/qwen2.5-math-7b-local,\
DATA_DIR=/data/user/zhongal/data/reschedule,\
RUN_NAME=${RUN_NAME},\
TOTAL_TRAINING_STEPS=1,\
HPF_TREE_ROLLOUT=True,\
HPF_TREE_NUM_PREFIXES=4,\
HPF_TREE_NUM_SUFFIXES=2,\
HPF_TREE_PREFIX_TEMPERATURE=1.0,\
HPF_TREE_SUFFIX_TEMPERATURE=0.25,\
SAVE_FREQ=1,\
SAVE_AFTER=0,\
RESUME_MODE=disable \
  examples/grpo_trainer/submit_qwen2_5_math_7b_hpf_masked_grpo_h100.slurm)

echo "JOB=${JOB}"
echo "RUN_NAME=${RUN_NAME}"
```

Monitor:

```bash
tail -F "slurm-verl-hpf-mask-${JOB}.out" "slurm-verl-hpf-mask-${JOB}.err"
```

Expected completion:

```bash
sacct -j "${JOB}" --format=JobID,JobName%30,State,ExitCode,Elapsed,NodeList%30
```

The job should show `COMPLETED` with `ExitCode 0:0`.

Confirm checkpoint contents:

```bash
CKPT="checkpoints/hpf_masked_grpo_dapo_math17k/${RUN_NAME}"
cat "${CKPT}/latest_checkpointed_iteration.txt"

find "${CKPT}/global_step_1/actor" -maxdepth 1 -type f \
  \( -name 'model_world_size_*' -o -name 'optim_world_size_*' -o -name 'extra_state_world_size_*' \) \
  | sort
```

The actor directory should include eight model shards, eight optimizer shards,
and eight extra state shards.

## Strict Resume Test

To test resume with optimizer state, resume from a saved checkpoint and extend
`TOTAL_TRAINING_STEPS`. For example, resume the one-step smoke run above to
step 2:

```bash
SOURCE_RUN=<previous_RUN_NAME>
SOURCE_CKPT="/data/user/zhongal/VERL/checkpoints/hpf_masked_grpo_dapo_math17k/${SOURCE_RUN}/global_step_1"
RESUME_RUN=hpf_resume_from1_$(date +%Y%m%d_%H%M%S)

JOB=$(sbatch --parsable \
  --export=ALL,\
MODEL_PATH=/data/user/zhongal/.cache/qwen2.5-math-7b-local,\
DATA_DIR=/data/user/zhongal/data/reschedule,\
RUN_NAME=${RESUME_RUN},\
TOTAL_TRAINING_STEPS=2,\
HPF_TREE_ROLLOUT=True,\
HPF_TREE_NUM_PREFIXES=4,\
HPF_TREE_NUM_SUFFIXES=2,\
HPF_TREE_PREFIX_TEMPERATURE=1.0,\
HPF_TREE_SUFFIX_TEMPERATURE=0.25,\
SAVE_FREQ=1,\
SAVE_AFTER=0,\
RESUME_MODE=resume_path \
  examples/grpo_trainer/submit_qwen2_5_math_7b_hpf_masked_grpo_h100.slurm \
  trainer.resume_from_path="${SOURCE_CKPT}")

echo "JOB=${JOB}"
echo "RUN_NAME=${RESUME_RUN}"
```

Check that the job really resumed:

```bash
grep -nE \
  'Load from checkpoint folder|Setting global step|Resuming from|Training from scratch|Skipping dataloader state restore|Training Progress' \
  "slurm-verl-hpf-mask-${JOB}.out" "slurm-verl-hpf-mask-${JOB}.err"
```

Expected resume signals:

```text
Load from checkpoint folder: .../global_step_1
Setting global step to 1
Resuming from .../global_step_1
Training Progress:  50%|...| 1/2
```

For mid-epoch resume, use a non-boundary checkpoint such as `global_step_7`.
The previous validation run showed correct mid-step resume behavior:

```text
Load from checkpoint folder: .../global_step_7
Setting global step to 7
Resuming from .../global_step_7
Training Progress:  39%|...| 7/18
```

## Small Resume-Validation Run

For a short end-to-end resume validation, run 12 global steps with 32 prompts
per global step, horizon window 192, and tree rollout enabled. Then resume to
18 global steps. If using a 200-problem subset, point `DATA_DIR` to a directory
whose `DAPO-Math-17k.parquet` is that subset. Otherwise the same command reads
the full DAPO-MATH-17k parquet while still limiting the run by
`TOTAL_TRAINING_STEPS`.

Initial 12-step run:

```bash
RUN_NAME=hpf_smoke_b32_h192_strict_$(date +%Y%m%d_%H%M%S)

JOB=$(sbatch --parsable \
  --export=ALL,\
MODEL_PATH=/data/user/zhongal/.cache/qwen2.5-math-7b-local,\
DATA_DIR=/data/user/zhongal/data/reschedule,\
RUN_NAME=${RUN_NAME},\
TOTAL_TRAINING_STEPS=12,\
TRAIN_BATCH_SIZE=32,\
PPO_MINI_BATCH_SIZE=16,\
PPO_MICRO_BATCH_SIZE_PER_GPU=8,\
HPF_PROGRESSIVE_BLOCK_SIZE=192,\
HPF_HORIZON_SCHEDULE=epoch,\
HPF_TREE_ROLLOUT=True,\
HPF_TREE_NUM_PREFIXES=4,\
HPF_TREE_NUM_SUFFIXES=2,\
HPF_TREE_PREFIX_TEMPERATURE=1.0,\
HPF_TREE_SUFFIX_TEMPERATURE=0.25,\
SAVE_FREQ=1,\
SAVE_AFTER=0,\
RESUME_MODE=disable \
  examples/grpo_trainer/submit_qwen2_5_math_7b_hpf_masked_grpo_h100.slurm)
```

Resume from step 12 to step 18:

```bash
SOURCE_RUN=<RUN_NAME_FROM_INITIAL_RUN>
SOURCE_CKPT="/data/user/zhongal/VERL/checkpoints/hpf_masked_grpo_dapo_math17k/${SOURCE_RUN}/global_step_12"

JOB=$(sbatch --parsable \
  --export=ALL,\
MODEL_PATH=/data/user/zhongal/.cache/qwen2.5-math-7b-local,\
DATA_DIR=/data/user/zhongal/data/reschedule,\
RUN_NAME=${SOURCE_RUN},\
TOTAL_TRAINING_STEPS=18,\
TRAIN_BATCH_SIZE=32,\
PPO_MINI_BATCH_SIZE=16,\
PPO_MICRO_BATCH_SIZE_PER_GPU=8,\
HPF_PROGRESSIVE_BLOCK_SIZE=192,\
HPF_HORIZON_SCHEDULE=epoch,\
HPF_TREE_ROLLOUT=True,\
HPF_TREE_NUM_PREFIXES=4,\
HPF_TREE_NUM_SUFFIXES=2,\
HPF_TREE_PREFIX_TEMPERATURE=1.0,\
HPF_TREE_SUFFIX_TEMPERATURE=0.25,\
SAVE_FREQ=10,\
SAVE_AFTER=0,\
RESUME_MODE=resume_path \
  examples/grpo_trainer/submit_qwen2_5_math_7b_hpf_masked_grpo_h100.slurm \
  trainer.resume_from_path="${SOURCE_CKPT}")
```

The previously validated run reached:

```text
latest_checkpointed_iteration.txt = 18
Training Progress: 100%|...| 18/18
```

## Offline W&B Sync and Diagnostics

Offline W&B runs are stored under:

```text
wandb/wandb/offline-run-*
```

Sync an offline run:

```bash
wandb sync \
  --id hpf_0to18_<job_or_run_id> \
  --entity zhongal-hkust \
  --project hpf_masked_grpo_dapo_math17k \
  --skip-console \
  wandb/wandb/<offline-run-dir>
```

Generate the terminal diagnostic report:

```bash
python examples/generation/diagnose_hpf_wandb_run.py \
  --entity zhongal-hkust \
  --project hpf_masked_grpo_dapo_math17k \
  --run-id hpf_0to18_<job_or_run_id>
```

The diagnostic report records:

```text
hpf/tree_horizon_tokens
hpf/horizon_round_index
hpf/tree_prefix_tokens_mean
hpf/tree_prefix_stopped_frac
hpf/follower/actor/pg_loss
hpf/leader/actor/pg_loss
hpf/follower/actor/hpf_kl_loss
hpf/leader/actor/hpf_kl_loss
hpf/correction_log_ratio_mean
hpf/correction_clip_frac
timing_s/hpf/tree_rollout_total_wall
timing_s/hpf/follower_update_actor
timing_s/hpf/leader_update_actor
timing_s/hpf/update_actor_total
```

For the validated 18-step smoke run, the horizon schedule was:

```text
steps 1-6:   192 tokens
steps 7-12:  384 tokens
steps 13-18: 576 tokens
```

## Progress and Log Signals

The HPF training path emits progress lines for actor mini-batches and
micro-batches when `HPF_PROGRESS_LOG_INTERVAL=1`:

```text
[HPF] actor mini-batch start label=hpf/follower/step-...
[HPF] actor micro-batch progress label=hpf/follower/step-... micro=...
[HPF] actor mini-batch done label=hpf/follower/step-...
```

Use these logs to distinguish rollout time from actor update time. In W&B, the
primary timing keys are:

```text
timing_s/hpf/tree_rollout_total_wall
timing_s/hpf/prefix_rollout_wall
timing_s/hpf/suffix_rollout_wall
timing_s/hpf/role_old_log_prob
timing_s/hpf/follower_update_actor
timing_s/hpf/suffix_correction_log_prob
timing_s/hpf/leader_update_actor
timing_s/hpf/update_actor_total
```

## Expected Artifacts

Training outputs are written under:

```text
checkpoints/hpf_masked_grpo_dapo_math17k/<RUN_NAME>/
rollout_data/hpf_masked_grpo_dapo_math17k/<RUN_NAME>/
wandb/wandb/offline-run-*
slurm-verl-hpf-mask-<JOB_ID>.out
slurm-verl-hpf-mask-<JOB_ID>.err
```

For resume-friendly checkpoints, expect:

```text
global_step_<N>/actor/model_world_size_8_rank_*.pt
global_step_<N>/actor/optim_world_size_8_rank_*.pt
global_step_<N>/actor/extra_state_world_size_8_rank_*.pt
global_step_<N>/actor/huggingface/
latest_checkpointed_iteration.txt
```

## Optional Full-Dataset Role Phases

The default Alg3 schedule updates follower and leader back-to-back for every
prompt batch. Set `HPF_ROLE_PHASED_TRAINING=True` to keep a fixed horizon while
the follower and leader each make complete train-set passes:

```text
for each horizon round:
    train follower over the full train set for m epochs
    train leader over the full train set for n epochs
    advance the horizon
```

Configure `m` and `n` with:

```bash
HPF_FOLLOWER_PHASE_EPOCHS=1
HPF_LEADER_PHASE_EPOCHS=1
```

Both default to one. This mode requires tree rollout, fresh Alg3 leader rollout,
an epoch-based horizon schedule, and a filtered train-set size divisible by the
dataloader batch size. These constraints are checked before GPU training starts.

`TOTAL_TRAINING_STEPS` counts role-specific prompt batches. For `R` horizon
rounds, use:

```text
TOTAL_TRAINING_STEPS = R * (m + n) * batches_per_train_set_pass
```

For example, 1536 filtered prompts with batch size 512 produce three batches
per pass. With `m=n=1`, one horizon round is six global steps: follower steps
1-3 followed by leader steps 4-6. The next round starts at step 7 with the next
horizon.

Checkpoint resume uses the existing stateful dataloader. The role and horizon
are reconstructed from the completed global steps, so a checkpoint at step 3
resumes at the first leader batch, while a checkpoint inside a role pass resumes
at the next unconsumed batch. Phase identity is logged through:

```text
hpf/role_phased_training_enabled
hpf/role_phase_follower
hpf/role_phase_leader
hpf/role_phase_epoch
hpf/role_phase_round
```

## Notes

- This HPF path should not modify the GRPO baseline reproduction script beyond
  passing extra Hydra overrides from the HPF wrapper.
- `trainer.val_before_train=False` is used to avoid an initial validation pass.
- Validation during training uses the baseline script's default
  `TRAIN_VAL_FILES`, currently `aime24.parquet`.
- The default launcher uses offline W&B to avoid network-dependent training
  failures; sync after the job completes.
- If using `RESUME_MODE=resume_path`, always pass the exact `global_step_*`
  directory as a Hydra override:
  `trainer.resume_from_path=/path/to/global_step_N`.
