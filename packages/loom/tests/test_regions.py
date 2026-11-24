# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operations stating a computation as a region applied a fixed number of times.

A sort of a fixed-length axis is a sorting network, and a loop whose count is a
constant is its iterations written out. Both are circuits, so Loom compiles them
by unrolling the region. A loop whose count is read at run time is not a circuit,
and is refused.

Each program is traced per lane, so the axis a program sorts is the middle axis
of the array handed to `_call`: `(receivers, sorted, lanes)`.
"""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax import lax

from _support import _call, _sds
from marks import gpu_only

pytestmark = pytest.mark.capped_autotune

_VALUES = np.asarray(
    [
        [[3.0, -1.0, 2.0], [1.0, 7.0, 2.0], [4.0, 0.5, 2.0], [1.5, 3.0, -8.0]],
        [[-2.0, 6.0, 1.0], [7.0, 6.0, 0.0], [0.0, -3.0, 5.0], [5.0, 6.0, 4.0]],
    ],
    np.float32,
)


def _values():
    return jnp.asarray(_VALUES)


def test_a_sort_orders_the_axis_it_is_given(device):
    x = _values()
    y = _call(lambda value: jnp.sort(value), x, _sds(x.shape, x.dtype))
    npt.assert_array_equal(np.asarray(y), np.sort(_VALUES, axis=1))


def test_a_sort_of_equal_keys_keeps_the_original_order(device):
    x = _values()

    def program(value):
        payload = jnp.asarray(np.arange(value.shape[0], dtype=np.float32), value.dtype)
        _, moved = lax.sort((value, payload), dimension=0, is_stable=True, num_keys=1)
        return moved

    y = np.asarray(_call(program, x, _sds(x.shape, x.dtype)))
    # Lane 1 of the second receiver holds 6, 6, -3, 6: the -3 leads, and the
    # three equal keys keep the order they were written in.
    npt.assert_array_equal(y[1, :, 1], [2.0, 0.0, 1.0, 3.0])
    # A lane whose keys are all distinct is the ordering permutation itself.
    npt.assert_array_equal(y[0, :, 0], np.argsort(_VALUES[0, :, 0], kind="stable"))


def test_a_loop_with_a_constant_count_runs_its_iterations(device):
    x = _values()
    y = _call(
        lambda value: lax.fori_loop(0, 3, lambda _, carried: carried * 2.0, value),
        x,
        _sds(x.shape, x.dtype),
    )
    npt.assert_array_equal(np.asarray(y), _VALUES * 8.0)


def test_a_loop_accumulating_over_its_index_runs_its_iterations(device):
    x = _values()
    y = _call(
        lambda value: lax.fori_loop(
            1, 4, lambda step, carried: carried + step.astype(value.dtype), value
        ),
        x,
        _sds(x.shape, x.dtype),
    )
    npt.assert_array_equal(np.asarray(y), _VALUES + 6.0)


@gpu_only
def test_a_loop_whose_count_is_read_at_run_time_is_refused(device):
    """The refusal that remains: a count no constant states cannot be unrolled."""
    x = _values()

    def program(value):
        limit = jnp.sum(value).astype(jnp.int32)
        return lax.fori_loop(0, limit, lambda _, carried: carried * 2.0, value)

    with pytest.raises(RuntimeError, match="while"):
        _call(program, x, _sds(x.shape, x.dtype))


def test_a_sorted_value_feeds_the_arithmetic_after_it(device):
    x = _values()

    def program(value):
        weights = jnp.asarray(np.arange(1.0, 1.0 + value.shape[0], dtype=np.float32), value.dtype)
        return jnp.sort(value) * weights

    y = _call(program, x, _sds(x.shape, x.dtype))
    expected = np.sort(_VALUES, axis=1) * np.arange(1.0, 5.0, dtype=np.float32).reshape(1, 4, 1)
    npt.assert_array_equal(np.asarray(y), expected)
