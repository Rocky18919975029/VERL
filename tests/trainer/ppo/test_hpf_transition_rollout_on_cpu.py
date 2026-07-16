# Copyright 2026 The verl team.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import numpy as np
import pytest

from verl.trainer.ppo.hpf_utils import build_hpf_transition_prefix_plan


def test_transition_prefix_plan_pairs_current_and_next_cut_requests():
    plan = build_hpf_transition_prefix_plan(
        [
            list(range(8)),
            list(range(5)),
            list(range(4)),
            list(range(2)),
        ],
        current_horizon=4,
        next_horizon=6,
        max_response_length=10,
    )

    assert plan.current_prefix_ids == [list(range(4)), list(range(4)), list(range(4)), list(range(2))]
    assert plan.next_prefix_ids == [list(range(6)), list(range(5)), list(range(4)), list(range(2))]
    np.testing.assert_array_equal(plan.current_needs_suffix, [True, True, False, False])
    np.testing.assert_array_equal(plan.next_needs_suffix, [True, False, False, False])
    # Requests are source-major so the two cuts from one sampled high-temp
    # prefix stay adjacent in the combined low-temperature batch.
    np.testing.assert_array_equal(plan.request_source_rows, [0, 0, 1])
    np.testing.assert_array_equal(plan.request_cut_indices, [0, 1, 0])


def test_transition_prefix_plan_omits_suffix_requests_at_final_cut():
    plan = build_hpf_transition_prefix_plan(
        [list(range(6))],
        current_horizon=6,
        next_horizon=6,
        max_response_length=6,
    )

    assert not plan.current_needs_suffix.any()
    assert not plan.next_needs_suffix.any()
    assert len(plan.request_source_rows) == 0


def test_transition_prefix_plan_rejects_reversed_cuts():
    with pytest.raises(ValueError, match="next_horizon must be at least current_horizon"):
        build_hpf_transition_prefix_plan(
            [[1, 2, 3]],
            current_horizon=4,
            next_horizon=2,
            max_response_length=8,
        )
