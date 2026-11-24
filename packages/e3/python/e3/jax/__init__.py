# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate equivariant functions on JAX arrays."""

from e3.jax._equivariant import (
    add,
    linear_combination,
    reduced_tensor_product,
    scale,
    spherical_harmonics,
    symmetric_contraction,
    symmetric_power,
    tensor_product,
    unflatten_by_irrep,
)
from e3.jax._geometry import free_knot_radial, unit_displacement, unit_displacement_pbc

__all__ = [
    "add",
    "linear_combination",
    "scale",
    "spherical_harmonics",
    "symmetric_contraction",
    "symmetric_power",
    "tensor_product",
    "unflatten_by_irrep",
    "free_knot_radial",
    "unit_displacement",
    "unit_displacement_pbc",
    "reduced_tensor_product",
]
