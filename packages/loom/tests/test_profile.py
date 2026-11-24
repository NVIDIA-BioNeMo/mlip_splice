# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from loom._src.primitive.profile import capture_compile_profile, phase, record

pytestmark = pytest.mark.capped_autotune


def test_compile_profile_accumulates_phases():
    with capture_compile_profile() as profile:
        with phase("jax_trace_ms"):
            pass
        record({"plan_ms": 2.5, "ptxas_ms": 3.0})
        record({"plan_ms": 1.5})
    assert profile["jax_trace_ms"] >= 0
    assert profile["plan_ms"] == 4.0
    assert profile["ptxas_ms"] == 3.0


def test_compile_profile_states_machine_facts_once():
    """A lowering runs many compiles, and each describes the same machine.

    Durations are per-compile and add up; the size of the compile pool and of
    the host do not. Were they summed, a lowering of twenty compiles would
    report twenty hosts and twenty times the workers it had.
    """
    with capture_compile_profile() as profile:
        for _ in range(3):
            record(
                {
                    "render_ptx_ms": 100.0,
                    "compile_workers": 16.0,
                    "host_memory_bytes": 3.2e10,
                }
            )
    assert profile["render_ptx_ms"] == 300.0
    assert profile["compile_workers"] == 16.0
    assert profile["host_memory_bytes"] == 3.2e10
