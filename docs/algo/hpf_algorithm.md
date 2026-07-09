# HPF Fresh-Tree RLVR

This note summarizes the experimental HPF implementation used by our DAPO-Math
experiments. The implementation lives on the `hpf-fresh-tree-alg3` branch and is
enabled through the `algorithm.hpf_rlvr` config block.

## Goal

HPF separates a response into two roles:

- **Leader / prefix policy**: learns to choose promising early reasoning paths.
- **Follower / suffix policy**: learns to complete a sampled prefix into a final
  answer.

The intended effect is to expose richer advantage signal than plain full
trajectory GRPO. For each problem, HPF samples multiple prefixes and multiple
suffixes under each prefix, so the learner can compare both:

- suffix quality under the same prefix, and
- prefix quality under the same problem.

## Tree Rollout

With tree rollout enabled, each training prompt produces:

```text
num_prefixes K x num_suffixes M trajectories
```

In our main setting this is `K=4`, `M=4`, so a batch of 512 prompts produces
8192 trajectories.

The rollout is generated in two stages:

1. Sample `K` prefixes using the leader temperature, usually higher, e.g. `1.0`.
2. For each non-finished prefix, sample `M` suffixes using the follower
   temperature, usually lower, e.g. `0.25`.

If a prefix already reaches a stop condition, the suffix is empty. These rows are
still represented in the tree, but the follower suffix mask is empty.

## Follower Update

The follower is trained only on suffix tokens. The prefix tokens are treated as
context and are masked out of the policy-gradient loss.

For each sampled prefix, HPF normalizes rewards over the suffixes under that
prefix. This gives the follower a within-prefix comparison:

```text
A_follower(prefix, suffix) = normalized reward among suffixes sharing the prefix
```

The follower update can also include a prefix-side KL reference through the HPF
KL mask and reference log-prob fields.

## Fresh-Tree Leader Update

The fresh-tree branch implements the Algorithm-3 style leader update. After the
follower update, the trainer refreshes rollout weights and samples a fresh tree
from the updated policy. The leader advantage is then computed at prefix level.

For each prefix `k` under problem `b`, the implementation first averages the
rewards of its suffixes:

```text
Q[b, k] = mean_m reward[b, k, m]
```

Then it compares prefixes within the same problem:

```text
A_leader[b, k] = Q[b, k] - mean_k Q[b, k]
```

When `std_normalize=True`, this centered value is divided by the prefix-level
standard deviation. This fresh-tree leader path is not the old any-correct
leader objective; it uses the mean suffix reward per prefix.

## Prefix-Level Deduplication

The leader objective is prefix-level, but the raw tree batch contains one row per
suffix. Without deduplication, each prefix appears `M` times and the leader would
repeat the same prefix update once per suffix.

The optimized branch computes prefix-level values and advantages on the full
trajectory tree first, then keeps one row per prefix for the leader actor update:

```text
512 prompts x 4 prefixes x 4 suffixes = 8192 trajectory rows
512 prompts x 4 prefixes             = 2048 leader rows
```

The retained row carries the already-aggregated prefix advantage, so the suffix
reward information is not lost. The leader batch, leader mask, leader advantage,
and leader old log-probs are all indexed with the same dedup indices.

## Prefix Truncation

The leader loss only applies to prefix tokens up to the progressive horizon. The
optimized branch therefore truncates leader update batches to:

```text
full prompt + first horizon response tokens
```

This avoids paying forward/backward cost for suffix tokens that are fully masked
out of the leader loss. The implementation also computes leader old log-probs on
the truncated prefix batch, then pads them back only where full-length tensors are
needed for intermediate bookkeeping.

Because a prefix-truncated leader batch has no suffix tokens left in its leader
KL mask, the fresh-tree path skips the follower-side suffix KL reference log-prob
computation when that mask is empty. This removes a full-length forward pass that
would otherwise have zero contribution to the loss.

## Progressive Horizon

The progressive horizon is:

```text
horizon = min(round_index * progressive_block_size, max_response_length)
```

The round index can be advanced by epoch or by fixed step intervals. Our main
runs used epoch-based progression.

## Main Switches

Common runtime settings are exposed in
`examples/grpo_trainer/run_qwen2_5_math_7b_hpf_masked_grpo.sh`:

```bash
HPF_TREE_ROLLOUT=True
HPF_TREE_NUM_PREFIXES=4
HPF_TREE_NUM_SUFFIXES=4
HPF_TREE_PREFIX_TEMPERATURE=1.0
HPF_TREE_SUFFIX_TEMPERATURE=0.25
HPF_PROGRESSIVE_BLOCK_SIZE=64
HPF_HORIZON_SCHEDULE=epoch
HPF_FRESH_LEADER_TREE=True
```

The underlying config block is `algorithm.hpf_rlvr` in
`verl/trainer/config/ppo_trainer.yaml`.

## Important Metrics

The implementation logs metrics that are useful for correctness and performance
checks:

- `hpf/tree_horizon_tokens`
- `hpf/tree_prefix_stopped_frac`
- `hpf/suffix_empty_frac`
- `hpf/leader_prefix_dedup_original_rows`
- `hpf/leader_prefix_dedup_rows`
- `hpf/leader_prefix_dedup_factor`
- `hpf/leader_prefix_truncation_response_len`
- `timing_s/hpf/tree_rollout_total_wall`
- `timing_s/hpf/follower_update_actor`
- `timing_s/hpf/leader_update_actor`
- `timing_s/hpf/update_actor_total`

In the 1-step smoke test, leader deduplication reduced the leader update batch
from 8192 rows to 2048 rows and reduced leader update time from about 1038s to
about 35s.

## Files

The core implementation is concentrated in:

- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/ppo/hpf_utils.py`
- `verl/trainer/config/ppo_trainer.yaml`
- `examples/grpo_trainer/run_qwen2_5_math_7b_hpf_masked_grpo.sh`
- `examples/grpo_trainer/submit_qwen2_5_math_7b_hpf_masked_grpo_h100.slurm`
