#!/usr/bin/env python3
"""Verify a transition-aware mixed-policy smoke run from saved artifacts."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


TOLERANCE = 2e-5


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing diagnostic artifact: {path}")
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing diagnostic artifact: {path}")
    return [json.loads(line) for line in path.open() if line.strip()]


def close(lhs: float, rhs: float, tolerance: float = TOLERANCE) -> bool:
    return math.isfinite(float(lhs)) and math.isfinite(float(rhs)) and abs(float(lhs) - float(rhs)) <= tolerance


def index_rollout(rows: list[dict], expected_cut: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        pair_uid = str(row.get("hpf_transition_pair_uid", ""))
        if not pair_uid:
            raise ValueError(f"Missing pair UID in {expected_cut} rollout dump.")
        if row.get("hpf_transition_cut") != expected_cut:
            raise ValueError(f"Pair {pair_uid} has the wrong cut label in {expected_cut} dump.")
        if pair_uid in indexed:
            raise ValueError(f"Duplicate pair UID in {expected_cut} rollout dump: {pair_uid}")
        indexed[pair_uid] = row
    return indexed


def expected_advantages(rows: list[dict]) -> dict[tuple[str, str], float]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["cut"], row["group_uid"])].append(float(row["score"]))

    expected = {}
    for group_key, scores in groups.items():
        if len(scores) == 1:
            mean, std = 0.0, 1.0
        else:
            mean = statistics.fmean(scores)
            std = statistics.stdev(scores)
        for score in scores:
            expected[(group_key[0], group_key[1], score)] = (score - mean) / (std + 1e-6)
    return expected


def aggregate_behavior_objective(rows: list[dict], mode: str, loss_scale_factor: float | None) -> float:
    nonempty = [row for row in rows if int(row["pg_tokens"]) > 0]
    if mode == "token-mean":
        numerator = sum(float(row["advantage"]) * int(row["pg_tokens"]) for row in nonempty)
        denominator = sum(int(row["pg_tokens"]) for row in nonempty)
        return numerator / denominator
    if mode == "seq-mean-token-mean":
        return statistics.fmean(float(row["advantage"]) for row in nonempty)
    if mode in {"seq-mean-token-sum", "seq-mean-token-sum-norm"}:
        value = statistics.fmean(
            float(row["advantage"]) * int(row["pg_tokens"]) for row in nonempty
        )
        if mode == "seq-mean-token-sum-norm":
            if loss_scale_factor is None:
                raise ValueError("Cannot audit seq-mean-token-sum-norm without its effective scale factor.")
            value /= float(loss_scale_factor)
        return value
    raise ValueError(f"Unsupported loss aggregation mode in diagnostic bundle: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()

    step = args.step
    diagnostics_dir = args.run_dir / "transition_diagnostics"
    summary = load_json(diagnostics_dir / f"step_{step}_summary.json")
    diagnostic_rows = load_jsonl(diagnostics_dir / f"step_{step}_rows.jsonl")
    samples = load_jsonl(diagnostics_dir / f"step_{step}_samples.jsonl")
    current_rollout_rows = load_jsonl(args.run_dir / f"{step}.jsonl")
    next_rollout_rows = load_jsonl(args.run_dir / "transition_next" / f"{step}.jsonl")

    checks = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    current_rollout = index_rollout(current_rollout_rows, "current")
    next_rollout = index_rollout(next_rollout_rows, "next")
    pair_sets_equal = current_rollout.keys() == next_rollout.keys()
    record("paired_rollout_uid_sets", pair_sets_equal, f"pairs={len(current_rollout)}")
    if not pair_sets_equal:
        raise ValueError("Current and next rollout pair UID sets differ.")

    if args.expected_rows is not None:
        record(
            "expected_rows_per_cut",
            len(current_rollout_rows) == args.expected_rows == len(next_rollout_rows),
            f"current={len(current_rollout_rows)} next={len(next_rollout_rows)} expected={args.expected_rows}",
        )

    prefix_failures = 0
    score_failures = 0
    diagnostic_by_key = {(row["cut"], row["pair_uid"]): row for row in diagnostic_rows}
    for pair_uid, current_row in current_rollout.items():
        next_row = next_rollout[pair_uid]
        current_prefix = [int(value) for value in current_row["hpf_prefix_ids"]]
        next_prefix = [int(value) for value in next_row["hpf_prefix_ids"]]
        if current_prefix != next_prefix[: len(current_prefix)]:
            prefix_failures += 1
        for cut, rollout_row in (("current", current_row), ("next", next_row)):
            diagnostic_row = diagnostic_by_key.get((cut, pair_uid))
            if diagnostic_row is None or not close(
                float(rollout_row["score"]), float(diagnostic_row["score"]), args.tolerance
            ):
                score_failures += 1
    record("current_prefix_is_next_prefix_truncation", prefix_failures == 0, f"failures={prefix_failures}")
    record("dumped_rewards_match_training_rewards", score_failures == 0, f"failures={score_failures}")

    expected = expected_advantages(diagnostic_rows)
    advantage_failures = 0
    max_advantage_error = 0.0
    max_advantage_span = 0.0
    for row in diagnostic_rows:
        if int(row["pg_tokens"]) == 0:
            continue
        key = (row["cut"], row["group_uid"], float(row["score"]))
        error = abs(float(row["advantage"]) - expected[key])
        max_advantage_error = max(max_advantage_error, error)
        max_advantage_span = max(max_advantage_span, abs(float(row["advantage_span"])))
        if error > args.tolerance:
            advantage_failures += 1
    record(
        "grpo_advantages_recomputed_from_complete_rewards",
        advantage_failures == 0,
        f"failures={advantage_failures} max_abs_error={max_advantage_error:.3e}",
    )
    record(
        "trajectory_advantage_constant_on_pg_tokens",
        max_advantage_span <= args.tolerance,
        f"max_span={max_advantage_span:.3e}",
    )

    rollout = summary["rollout"]
    behavior = summary["behavior"]
    identities = summary["identities"]
    dynamic = summary["dynamic"]
    record(
        "transition_return_is_expected_return_difference",
        abs(float(rollout["transition_return_residual"])) <= args.tolerance,
        f"residual={float(rollout['transition_return_residual']):.3e}",
    )
    record(
        "static_B_formula",
        abs(float(behavior["static_offset_residual"])) <= args.tolerance,
        f"residual={float(behavior['static_offset_residual']):.3e}",
    )
    loss_mode = summary["loss_agg_mode"]
    loss_scale_factor = summary.get("loss_scale_factor")
    if loss_mode == "seq-mean-token-sum-norm" and loss_scale_factor is None:
        loss_scale_factor = summary["batch"]["response_len_before"]
    for cut, expected_value in (
        ("current", float(behavior["current_objective"])),
        ("next", float(behavior["next_objective"])),
    ):
        cut_rows = [row for row in diagnostic_rows if row["cut"] == cut]
        recomputed = aggregate_behavior_objective(cut_rows, loss_mode, loss_scale_factor)
        record(
            f"{cut}_behavior_GRPO_objective",
            close(recomputed, expected_value, args.tolerance),
            f"saved={expected_value:.8f} recomputed={recomputed:.8f}",
        )

    preparation_checks = {
        "old_logprob_prefix_stitching": "hpf/mixed_policy_grpo_prefix_old_logprob_max_abs_error",
        "old_logprob_suffix_stitching": "hpf/mixed_policy_grpo_suffix_old_logprob_max_abs_error",
        "prefix_temperature_assignment": "hpf/mixed_policy_grpo_prefix_temperature_max_abs_error",
        "suffix_temperature_assignment": "hpf/mixed_policy_grpo_suffix_temperature_max_abs_error",
        "pg_mask_within_response": "hpf/mixed_policy_grpo_pg_tokens_outside_response",
    }
    for cut_name, preparation in (
        ("current", summary["current_cut_preparation"]),
        ("next", summary["next_cut_preparation"]),
    ):
        for check_name, metric_name in preparation_checks.items():
            value = float(preparation[metric_name])
            record(
                f"{cut_name}_{check_name}",
                abs(value) <= args.tolerance,
                f"value={value:.3e}",
            )

    sample_failures = 0
    nonfinite_sample_values = 0
    prefix_temperature = float(summary["current_cut_preparation"]["hpf/mixed_policy_grpo_prefix_temperature"])
    suffix_temperature = float(summary["current_cut_preparation"]["hpf/mixed_policy_grpo_suffix_temperature"])
    suffix_window = int(summary["current_cut_preparation"]["hpf/mixed_policy_grpo_suffix_window_size"])
    for sample in samples:
        horizon = summary["current_horizon"] if sample["cut"] == "current" else summary["next_horizon"]
        length = len(sample["response_ids"])
        expected_end = length if suffix_window < 0 else min(length, int(horizon) + suffix_window)
        for position, (pg, temperature) in enumerate(zip(sample["pg_mask"], sample["temperature"], strict=True)):
            expected_pg = position < expected_end
            expected_temperature = prefix_temperature if position < int(horizon) else suffix_temperature
            if bool(pg) != expected_pg or (pg and not close(temperature, expected_temperature, args.tolerance)):
                sample_failures += 1
                break
        nonfinite_sample_values += sum(
            not math.isfinite(float(value))
            for key in ("old_log_probs", "temperature", "advantages")
            for value in sample[key]
        )
    record(
        "token_samples_match_cut_window_and_temperature",
        sample_failures == 0 and bool(samples),
        f"samples={len(samples)} failures={sample_failures}",
    )
    record(
        "token_sample_values_are_finite",
        nonfinite_sample_values == 0,
        f"nonfinite_values={nonfinite_sample_values}",
    )

    surrogate_residual = identities.get("surrogate_residual")
    objective_residual = identities.get("objective_loss_residual")
    record(
        "dynamic_surrogate_formula",
        surrogate_residual is not None and abs(float(surrogate_residual)) <= args.tolerance,
        f"residual={surrogate_residual}",
    )
    record(
        "dynamic_objective_formula",
        objective_residual is not None and abs(float(objective_residual)) <= args.tolerance,
        f"residual={objective_residual}",
    )
    penalty = float(dynamic.get("penalty", float("nan")))
    active = float(dynamic.get("active", float("nan")))
    record("transition_hinge_penalty_nonnegative", math.isfinite(penalty) and penalty >= 0, f"penalty={penalty}")
    record("transition_hinge_active_fraction_valid", math.isfinite(active) and 0 <= active <= 1, f"active={active}")
    grad_norm = float(summary["actor_metrics"].get("actor/grad_norm", float("nan")))
    record(
        "actor_backward_produced_finite_gradient",
        math.isfinite(grad_norm) and grad_norm > 0,
        f"grad_norm={grad_norm}",
    )
    record(
        "joint_batch_has_no_padding",
        int(summary["batch"]["joint_pad_rows"]) == 0
        and int(summary["batch"]["joint_rows"]) == 2 * int(summary["num_pairs"]),
        f"joint={summary['batch']['joint_rows']} pairs={summary['num_pairs']} pad={summary['batch']['joint_pad_rows']}",
    )

    passed = all(check["passed"] for check in checks)
    report = {
        "schema_version": 1,
        "run_dir": str(args.run_dir),
        "step": step,
        "passed": passed,
        "checks": checks,
    }
    report_path = diagnostics_dir / f"step_{step}_verification.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print("Transition-aware mixed policy verification")
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['name']}: {check['detail']}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {report_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
