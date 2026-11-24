# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact bases of the equivariant tensor products.

Every basis is a ``SparseTensor`` of exact ``AlgebraicReal`` coefficients, so the
identities among them hold exactly rather than to within rounding, and the
compiler can prove the cancellations they contain.
"""

from __future__ import annotations

from fractions import Fraction

from algebraic_real import AlgebraicReal, pow

from e3._native import SparseTensor, reduced_symmetric_tensor_product
from e3._native import reduced_tensor_product as _reduce_product
from e3._types import Irrep, Irreps, common_group


def clebsch_gordan(a: Irrep, b: Irrep, c: Irrep) -> list[SparseTensor]:
    """The couplings of ``a (x) b`` into ``c``, one basis element per path."""
    if not (a.group == b.group == c.group):
        raise ValueError("clebsch_gordan requires three irreps of the same group")
    return a.group.clebsch_gordan(a, b, c)


def reduced_tensor_product(
    per_index: list[Irreps], keep=None, formula=None
) -> list[tuple[Irrep, SparseTensor]]:
    """An orthonormal basis of the tensor product of the given index spaces.

    ``formula`` states an index symmetry, as in ``"ijk=jik"``, and the basis is
    reduced accordingly; ``keep`` restricts the output irreps.
    """
    common_group(per_index)
    return _reduce_product(per_index, keep, formula)


def spherical_harmonic_basis(vector: Irrep, degree: int) -> tuple[Irrep, SparseTensor]:
    """The solid harmonic of ``degree`` as an exact symmetric tensor.

    The harmonic of degree ``l`` is the unique symmetric traceless part of
    ``vector**l``, so it is the ``l``-fold symmetric tensor product reduced onto
    the Cartan irrep ``vector**l``. That basis is unique up to scale and sign;
    the two lines below fix both, giving the same convention as the standard
    real solid harmonics:

    - scale, so that the basis has norm ``sqrt(out.dim)`` before the group's own
      harmonic normalization, which makes ``Y_l`` orthonormal on the sphere;
    - sign, so that the highest-weight coefficient is positive.
    """
    if degree < 1:
        raise ValueError("degree must be positive")
    group = vector.group
    target = group.trivial()
    for _ in range(degree):
        target = target * vector
    ((out, basis),) = reduced_symmetric_tensor_product(
        Irreps(group, [(1, vector)]), degree, [target]
    )
    entries = list(basis)

    norm_squared = sum((c * c for _, c in entries), AlgebraicReal())
    scale = group.harmonic_normalization(degree) * pow(
        Fraction(out.dim, 1) / norm_squared.as_fraction(), Fraction(1, 2)
    )
    highest = tuple(size - 1 for size in basis.shape[:degree])
    highest_coefficient = next(
        c
        for coordinates, c in entries
        if tuple(coordinates[:degree]) == highest and coordinates[-1] == out.dim - 1
    )
    if highest_coefficient.sign() < 0:
        scale = -scale

    # The reduction returns one path axis, always of length 1 here because the
    # Cartan irrep appears once; drop it so the basis is (3,) * degree + (dim,).
    return out, SparseTensor(
        (*basis.shape[:degree], out.dim),
        [((*coordinates[:degree], coordinates[-1]), c * scale) for coordinates, c in entries],
    )
