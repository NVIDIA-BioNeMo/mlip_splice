# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import e3
import e3.jax
import jax
import jax.numpy as jnp
import numpy.testing as npt


def test_periodic_unit_displacement():
    sender = jnp.array([0.0, 0.0, 0.0])
    receiver = jnp.array([0.5, 0.0, 0.0])
    shift = jnp.array([1, -1, 0], dtype=jnp.int8)
    lattice = jnp.diag(jnp.array([2.0, 3.0, 4.0]))

    direction, distance = e3.jax.unit_displacement_pbc(sender, receiver, shift, lattice)

    displacement = jnp.array([2.5, -3.0, 0.0])
    expected_distance = jnp.linalg.norm(displacement)
    npt.assert_allclose(direction, displacement / expected_distance)
    npt.assert_allclose(distance, expected_distance, rtol=2e-7)


def test_free_knot_radial_has_exact_cutoff():
    evaluate = jax.jit(e3.jax.free_knot_radial)
    inside = evaluate(jnp.array(0.25), jnp.array(1.0), jnp.array(3.0))
    outside = evaluate(jnp.array(1.25), jnp.array(1.0), jnp.array(3.0))

    npt.assert_allclose(inside, jnp.array(0.984375))
    npt.assert_array_equal(outside, jnp.array(0.0))


def test_free_knot_radial_can_have_one_shape_per_path():
    result = e3.jax.free_knot_radial(
        jnp.array(0.25),
        jnp.array(1.0),
        jnp.array([3.0, 5.0]),
    )
    npt.assert_allclose(result, jnp.array([0.984375, 1.265625]))
