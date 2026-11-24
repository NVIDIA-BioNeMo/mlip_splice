# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-JAX evaluation of exact equivariant bases."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from fractions import Fraction
from typing import Any

import jax.numpy as jnp
import numpy as np
from algebraic_real import pow
from algebraic_real.jax import exact

from e3._bases import (
    clebsch_gordan,
    reduced_symmetric_tensor_product,
    spherical_harmonic_basis,
)
from e3._bases import (
    reduced_tensor_product as reduced_tensor_product_basis,
)
from e3._groups import Group
from e3._types import Irrep, Irreps

IrrepValues = Mapping[Irrep, Any]


def _canonical(
    name: str,
    *operands: IrrepValues,
    keep: Iterable[Irrep] | None = None,
) -> tuple[Group, list[list[tuple[Irrep, Any]]], list[Irrep] | None]:
    """Validate and order irrep-valued operands and an optional output filter."""
    if not operands:
        raise ValueError(f"{name} needs at least one operand")

    ordered = []
    group = None
    for operand_index, operand in enumerate(operands):
        if not operand:
            raise ValueError(f"{name} operand {operand_index} must not be empty")
        items = sorted(operand.items())
        for ir, value in items:
            if value.shape != (ir.dim,):
                raise ValueError(
                    f"{name} operand {operand_index}[{ir!r}] has shape {value.shape}, "
                    f"expected ({ir.dim},)"
                )
            if group is None:
                group = ir.group
            elif ir.group is not group:
                raise ValueError(f"{name} operands use different groups")
        ordered.append(items)

    selected = None if keep is None else sorted(set(keep))
    if selected is not None and any(ir.group is not group for ir in selected):
        raise ValueError(f"{name} output filter uses a different group")
    return group, ordered, selected


def _evaluate_basis(basis: Any, operands: list[Any]) -> jnp.ndarray:
    """Contract operands with a sparse exact basis without materializing zeros."""
    input_rank = len(operands)
    input_shape = tuple(operand.shape[0] for operand in operands)
    if tuple(basis.shape[:input_rank]) != input_shape:
        raise ValueError(
            f"basis input shape {tuple(basis.shape[:input_rank])}, expected {input_shape}"
        )
    output_shape = tuple(basis.shape[input_rank:])
    if not output_shape:
        raise ValueError("basis must have at least one output axis")

    entries = list(basis)
    coordinates = np.asarray([coordinates for coordinates, _ in entries], dtype=np.int32)
    dtype = jnp.result_type(*(operand.dtype for operand in operands))
    terms = exact([coefficient for _, coefficient in entries], dtype=dtype)
    for axis, operand in enumerate(operands):
        selection = np.zeros((len(entries), operand.shape[0]), dtype=np.int8)
        selection[np.arange(len(entries)), coordinates[:, axis]] = 1
        selected = jnp.asarray(selection, dtype=dtype) * operand[None, :]
        terms = terms * jnp.sum(selected, axis=1)

    output_coordinates = tuple(coordinates[:, axis] for axis in range(input_rank, basis.ndim))
    output_indices = np.ravel_multi_index(output_coordinates, output_shape)
    projection = np.zeros((np.prod(output_shape), len(entries)), dtype=np.int8)
    projection[output_indices, np.arange(len(entries))] = 1
    return jnp.sum(jnp.asarray(projection, dtype=dtype) * terms[None, :], axis=1).reshape(
        output_shape
    )


def add(*operands: IrrepValues) -> dict[Irrep, jnp.ndarray]:
    """Add equivariant values, treating absent irreps as zero."""
    _, ordered, _ = _canonical("add", *operands)
    output = dict(ordered[0])
    for operand in ordered[1:]:
        for ir, value in operand:
            output[ir] = output[ir] + value if ir in output else value
    return {ir: output[ir] for ir in sorted(output)}


def scale(scalar: Any, features: IrrepValues) -> dict[Irrep, jnp.ndarray]:
    """Multiply every irrep by a scalar JAX array."""
    if scalar.shape != ():
        raise ValueError(f"scalar has shape {scalar.shape}, expected ()")
    _, (items,), _ = _canonical("scale", features)
    return {ir: scalar * value for ir, value in items}


def tensor_product(
    left: IrrepValues,
    right: IrrepValues,
    keep: Iterable[Irrep] | None = None,
) -> dict[Irrep, jnp.ndarray]:
    """Return every Clebsch–Gordan coupling as an output-irrep path column."""
    _, (left_items, right_items), selected = _canonical("tensor_product", left, right, keep=keep)
    selected_set = None if selected is None else set(selected)
    paths: dict[Irrep, list[jnp.ndarray]] = {}
    for left_ir, left_value in left_items:
        for right_ir, right_value in right_items:
            for out_ir in left_ir.selection_rule(right_ir):
                if selected_set is not None and out_ir not in selected_set:
                    continue
                for basis in clebsch_gordan(left_ir, right_ir, out_ir):
                    paths.setdefault(out_ir, []).append(
                        _evaluate_basis(basis, [left_value, right_value])
                    )
    return {ir: jnp.stack(paths[ir], axis=1) for ir in sorted(paths)}


def reduced_tensor_product(
    *operands: IrrepValues,
    keep: Iterable[Irrep] | None = None,
) -> dict[Irrep, jnp.ndarray]:
    """Evaluate an exact orthonormal basis of a general tensor product."""
    group, ordered, selected = _canonical("reduced_tensor_product", *operands, keep=keep)
    structures = [Irreps(group, [(1, ir) for ir, _ in items]) for items in ordered]
    flat = [jnp.concatenate([value for _, value in items]) for items in ordered]
    paths: dict[Irrep, list[jnp.ndarray]] = {}
    for out_ir, basis in reduced_tensor_product_basis(structures, keep=selected):
        paths.setdefault(out_ir, []).append(_evaluate_basis(basis, flat).T)
    return {ir: jnp.concatenate(paths[ir], axis=1) for ir in sorted(paths)}


def symmetric_power(
    features: IrrepValues,
    degree: int,
    keep: Iterable[Irrep] | None = None,
) -> dict[Irrep, jnp.ndarray]:
    """Evaluate an exact basis of equivariant homogeneous polynomials."""
    if degree < 1:
        raise ValueError("degree must be positive")
    group, (items,), selected = _canonical("symmetric_power", features, keep=keep)
    structure = Irreps(group, [(1, ir) for ir, _ in items])
    flat = jnp.concatenate([value for _, value in items])
    paths: dict[Irrep, list[jnp.ndarray]] = {}
    for out_ir, basis in reduced_symmetric_tensor_product(structure, degree, selected):
        paths.setdefault(out_ir, []).append(_evaluate_basis(basis, [flat] * degree).T)
    return {ir: jnp.concatenate(paths[ir], axis=1) for ir in sorted(paths)}


def unflatten_by_irrep(values: Any, sizes: Mapping[Irrep, int]) -> dict[Irrep, jnp.ndarray]:
    """Split a vector into irrep-keyed slices in canonical order."""
    if values.ndim != 1:
        raise ValueError(f"values has shape {values.shape}, expected a vector")
    if any(size < 1 for size in sizes.values()):
        raise ValueError("sizes must be positive")
    if values.shape[0] != sum(sizes.values()):
        raise ValueError(f"values has length {values.shape[0]}, expected {sum(sizes.values())}")

    output = {}
    start = 0
    for ir in sorted(sizes):
        stop = start + sizes[ir]
        output[ir] = values[start:stop]
        start = stop
    return output


def linear_combination(
    paths: Mapping[Irrep, Any],
    weights: Mapping[Irrep, Any],
    *,
    normalize: bool = False,
) -> dict[Irrep, jnp.ndarray]:
    """Pair path columns with one scalar weight per path."""
    if set(paths) != set(weights):
        raise ValueError("paths and weights must have the same irreps")

    output = {}
    for ir, value in sorted(paths.items()):
        if value.ndim != 2 or value.shape[0] != ir.dim:
            raise ValueError(f"paths[{ir!r}] has shape {value.shape}, expected ({ir.dim}, P)")
        if weights[ir].shape != (value.shape[1],):
            raise ValueError(
                f"weights[{ir!r}] has shape {weights[ir].shape}, expected ({value.shape[1]},)"
            )
        result = jnp.sum(value * weights[ir][None, :], axis=1)
        if normalize:
            result = result * exact(
                pow(Fraction(1, value.shape[1]), Fraction(1, 2)), dtype=result.dtype
            )
        output[ir] = result
    return output


def symmetric_contraction(
    features: IrrepValues,
    weights_by_degree: Mapping[int, Mapping[Irrep, Any]],
) -> dict[Irrep, jnp.ndarray]:
    """Contract symmetric powers with explicit path weights."""
    if not weights_by_degree:
        raise ValueError("weights_by_degree must not be empty")
    return add(
        *(
            linear_combination(symmetric_power(features, degree, keep=weights.keys()), weights)
            for degree, weights in sorted(weights_by_degree.items())
        )
    )


def spherical_harmonics(vector: IrrepValues, max_degree: int) -> dict[Irrep, jnp.ndarray]:
    """Return solid harmonics in the group's canonical normalization and phase."""
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")
    group, (items,), _ = _canonical("spherical_harmonics", vector)
    if len(items) != 1:
        raise ValueError("vector must contain exactly one irrep")
    vector_ir, value = items[0]
    output = {group.trivial(): exact([group.harmonic_normalization(0)], dtype=value.dtype)}
    for degree in range(1, max_degree + 1):
        out_ir, basis = spherical_harmonic_basis(vector_ir, degree)
        output[out_ir] = _evaluate_basis(basis, [value] * degree)
    return output


__all__ = [
    "add",
    "linear_combination",
    "reduced_tensor_product",
    "scale",
    "spherical_harmonics",
    "symmetric_contraction",
    "symmetric_power",
    "tensor_product",
    "unflatten_by_irrep",
]
