# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The mathematics of E(3): rotations, reflections, translations, and the
equivariant functions of 3D points built from them.

``Irrep`` and ``Irreps`` name representations of a group — ``O3`` for the model,
``SO3`` and ``SU2`` for the cases that need them; the group is always explicit,
as in ``Irreps(O3, "0e+1o")``. Coefficients are exact ``AlgebraicReal`` values, so
tensor products and contractions are built from proved identities rather than
rounded constants.
"""

from e3._bases import (
    SparseTensor,
    clebsch_gordan,
    reduced_symmetric_tensor_product,
    reduced_tensor_product,
    spherical_harmonic_basis,
)
from e3._groups import O3, SO3, SU2, Group
from e3._types import Irrep, Irreps

__all__ = [
    "O3",
    "SO3",
    "SU2",
    "Group",
    "Irrep",
    "Irreps",
    "SparseTensor",
    "clebsch_gordan",
    "reduced_symmetric_tensor_product",
    "reduced_tensor_product",
    "spherical_harmonic_basis",
]
