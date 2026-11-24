# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import e3
import e3.jax
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from e3 import O3, SO3, Irreps


def test_add_and_scale():
    scalar = O3("0e")
    vector = O3("1o")
    x = {scalar: jnp.array([2.0]), vector: jnp.array([1.0, 2.0, 3.0])}
    y = {vector: jnp.array([4.0, 5.0, 6.0])}

    result = e3.jax.scale(jnp.array(3.0), e3.jax.add(x, y))

    npt.assert_array_equal(result[scalar], jnp.array([6.0]))
    npt.assert_array_equal(result[vector], jnp.array([15.0, 21.0, 27.0]))


def test_tensor_product_paths_are_composed_explicitly():
    scalar = O3("0e")
    vector = O3("1o")
    left = {scalar: jnp.array([2.0]), vector: jnp.array([1.0, 2.0, 3.0])}
    right = {scalar: jnp.array([3.0]), vector: jnp.array([4.0, 5.0, 6.0])}

    paths = e3.jax.tensor_product(left, right, (scalar, vector))
    assert paths[scalar].shape == (1, 2)
    assert paths[vector].shape == (3, 2)
    result = e3.jax.linear_combination(
        paths,
        {
            scalar: jnp.array([2.0, 3.0]),
            vector: jnp.array([4.0, 5.0]),
        },
    )

    npt.assert_allclose(result[scalar], 12 + 96 / jnp.sqrt(3.0), rtol=1e-6)
    npt.assert_allclose(result[vector], jnp.array([47.0, 70.0, 93.0]))


def test_tensor_product_is_jittable_and_differentiable():
    vector = O3("1o")
    scalar = O3("0e")

    @jax.jit
    def squared_norm(value):
        paths = e3.jax.tensor_product({vector: value}, {vector: value}, (scalar,))
        return paths[scalar][0, 0]

    value = jnp.array([1.0, 2.0, 3.0])
    npt.assert_allclose(squared_norm(value), 14 / jnp.sqrt(3.0), rtol=1e-6)
    npt.assert_allclose(jax.grad(squared_norm)(value), 2 * value / jnp.sqrt(3.0), rtol=1e-6)


def test_reduced_tensor_product_is_nary():
    scalar = O3("0e")
    vector = O3("1o")
    values = {scalar: jnp.array([2.0]), vector: jnp.array([1.0, 2.0, 3.0])}

    result = e3.jax.reduced_tensor_product(values, values, values, keep=(scalar,))

    assert result[scalar].shape[0] == scalar.dim
    assert result[scalar].shape[1] > 0
    assert jnp.all(jnp.isfinite(result[scalar]))


def test_symmetric_contraction_of_a_scalar_is_an_ordinary_polynomial():
    scalar = O3("0e")
    value = {scalar: jnp.array([2.0])}
    weights = {
        1: {scalar: jnp.array([3.0])},
        2: {scalar: jnp.array([5.0])},
    }

    degree_two = e3.jax.symmetric_power(value, 2)
    result = e3.jax.symmetric_contraction(value, weights)

    npt.assert_array_equal(degree_two[scalar], jnp.array([[4.0]]))
    npt.assert_array_equal(result[scalar], jnp.array([26.0]))


def test_symmetric_contraction_is_jittable_and_differentiable():
    scalar = O3("0e")
    vector = O3("1o")
    weights = {
        1: {scalar: jnp.array([2.0]), vector: jnp.array([3.0])},
        2: {scalar: jnp.array([5.0, 7.0]), vector: jnp.array([11.0])},
    }

    @jax.jit
    def loss(scalar_value, vector_value):
        result = e3.jax.symmetric_contraction({scalar: scalar_value, vector: vector_value}, weights)
        return sum(jnp.sum(value**2) for value in result.values())

    values = (jnp.array([0.5]), jnp.array([0.2, -0.3, 0.7]))
    assert jnp.isfinite(loss(*values))
    assert all(jnp.all(jnp.isfinite(value)) for value in jax.grad(loss, (0, 1))(*values))


def test_spherical_harmonics_have_exact_low_degree_values():
    vector = O3("1o")
    result = jax.jit(e3.jax.spherical_harmonics, static_argnums=1)(
        {vector: jnp.array([1.0, 2.0, 3.0])}, 3
    )

    assert list(result) == [O3("0e"), O3("1o"), O3("2e"), O3("3o")]
    assert {ir: value.shape for ir, value in result.items()} == {
        O3("0e"): (1,),
        O3("1o"): (3,),
        O3("2e"): (5,),
        O3("3o"): (7,),
    }
    npt.assert_array_equal(result[O3("0e")], jnp.ones(1))
    npt.assert_allclose(result[vector], jnp.sqrt(3.0) * jnp.array([1.0, 2.0, 3.0]), rtol=1e-6)


def test_eval_shape_returns_path_axis_shapes():
    irreps = Irreps(O3, "0e + 1o")
    keep = tuple(ir for _, ir in irreps)
    values = {ir: jnp.arange(ir.dim, dtype=jnp.float32) + 1 for ir in keep}
    abstract = jax.tree.map(lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype), values)

    operations = (
        lambda operands: e3.jax.tensor_product(operands, operands, keep),
        *(
            lambda operands, degree=degree: e3.jax.symmetric_power(operands, degree, keep)
            for degree in (1, 2, 3)
        ),
    )
    for operation in operations:
        abstract_paths = jax.eval_shape(operation, abstract)
        paths = operation(values)
        assert jax.tree.map(lambda value: value.shape, abstract_paths) == jax.tree.map(
            lambda value: value.shape, paths
        )


def test_shapes_and_groups_are_explicit():
    with pytest.raises(ValueError, match=r"expected \(3,\)"):
        e3.jax.tensor_product(
            {O3("1o"): jnp.ones((3, 1))},
            {O3("1o"): jnp.ones((3,))},
        )
    with pytest.raises(ValueError, match="different groups"):
        e3.jax.add(
            {O3("0e"): jnp.ones((1,))},
            {SO3("0"): jnp.ones((1,))},
        )
    assert (
        e3.jax.tensor_product(
            {O3("0e"): jnp.ones((1,))},
            {O3("0e"): jnp.ones((1,))},
            keep=(),
        )
        == {}
    )


def test_tensor_product_dtype_is_symmetric():
    scalar = O3("0e")
    low = {scalar: jnp.array([2.0], dtype=jnp.bfloat16)}
    high = {scalar: jnp.array([3.0], dtype=jnp.float32)}

    left = e3.jax.tensor_product(low, high)
    right = e3.jax.tensor_product(high, low)

    assert left[scalar].dtype == right[scalar].dtype == jnp.float32
    npt.assert_array_equal(left[scalar], right[scalar])
