# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from verl.trainer.ppo.hpf_schedule import get_hpf_role_phase


def test_role_phase_schedule_default_epochs():
    assert [get_hpf_role_phase(i, 1, 1) for i in range(4)] == [
        ("follower", 1, 1),
        ("leader", 1, 1),
        ("follower", 1, 2),
        ("leader", 1, 2),
    ]


def test_role_phase_schedule_multiple_role_epochs():
    assert [get_hpf_role_phase(i, 2, 3) for i in range(6)] == [
        ("follower", 1, 1),
        ("follower", 2, 1),
        ("leader", 1, 1),
        ("leader", 2, 1),
        ("leader", 3, 1),
        ("follower", 1, 2),
    ]


@pytest.mark.parametrize("physical_epoch,follower_epochs,leader_epochs", [(-1, 1, 1), (0, 0, 1), (0, 1, 0)])
def test_role_phase_schedule_rejects_invalid_values(physical_epoch, follower_epochs, leader_epochs):
    with pytest.raises(ValueError):
        get_hpf_role_phase(physical_epoch, follower_epochs, leader_epochs)
