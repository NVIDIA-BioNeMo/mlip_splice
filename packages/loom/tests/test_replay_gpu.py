# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The public capture seam stops safely and replays through real CUDA lowering."""

import jax
import jax.numpy as jnp
import loom
import numpy.testing as npt
import pytest
from loom.compilation import RecipeLoadFinished, SourceCompilation

from _support import _call, _sds


@pytest.mark.capped_autotune
def test_capture_abort_then_replay(device, tmp_path):
    if device.platform != "gpu":
        pytest.skip("CUDA source compilation contract")
    x = jnp.arange(32, dtype=jnp.float32).reshape(4, 1, 8)
    options = loom.Compile(search_budget_ms=25, shortlist_budget_ms=25, scheduling_time_limit_ms=10)

    def program(x):
        return _call(
            lambda value: value * value + 2, x, _sds(x.shape, x.dtype), compile_options=options
        )

    class Stop(Exception):
        pass

    def capture(event):
        if isinstance(event, SourceCompilation):
            event.save(tmp_path / "source.json")
            raise Stop
        pytest.fail("realization started before capture stopped")

    with pytest.raises(Stop), loom.observe(capture):
        jax.jit(program).lower(x).compile()
    artifact = SourceCompilation.load(tmp_path / "source.json")
    events = []
    with loom.replay([artifact]), loom.observe(events.append):
        executable = jax.jit(program).lower(x).compile()
    npt.assert_allclose(executable(x), x * x + 2)
    assert sum(isinstance(event, SourceCompilation) for event in events) == 1
    assert sum(isinstance(event, RecipeLoadFinished) for event in events) == 1
    with pytest.raises(loom.ReplayError, match="unused"), loom.replay([artifact]):
        executable(x)
