# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fractions import Fraction

import pytest
from algebraic_real import AlgebraicReal
from e3 import (
    O3,
    SO3,
    SU2,
    Irreps,
    clebsch_gordan,
    reduced_symmetric_tensor_product,
    reduced_tensor_product,
    spherical_harmonic_basis,
)


def test_irrep_construction_and_attributes():
    assert O3("1o").label == (1, -1)
    assert O3("2e").label == (2, 1)
    assert O3("2e").dim == 5
    assert O3.irrep((1, -1)) == O3("1o")
    assert repr(O3("0e")) == "0e"
    assert O3.trivial() == O3("0e")
    assert SO3("2").label == 2 and SO3("2").dim == 5
    assert SU2("1/2").label == 1 and SU2("1/2").dim == 2
    assert repr(SU2.irrep(3)) == "3/2"
    with pytest.raises(ValueError):
        O3("nope")
    with pytest.raises(ValueError):
        SU2("2/2")  # an even numerator is not a reduced half-integer


def test_irrep_ordering_and_hash():
    assert O3("0e") < O3("1o")
    assert sorted([O3("2e"), O3("0e"), O3("1o")]) == [O3("0e"), O3("1o"), O3("2e")]
    assert len({O3("1o"), O3.irrep((1, -1)), O3("2e")}) == 2
    assert SO3("1") != SU2("1")


def test_cartan_product_and_selection_rule():
    assert O3("1o") * O3("1o") == O3("2e")
    assert [repr(x) for x in O3("1o").selection_rule(O3("1o"))] == ["0e", "1e", "2e"]
    assert [repr(x) for x in O3("1o").selection_rule(O3("2e"))] == ["1o", "2o", "3o"]
    assert SU2("1/2").selection_rule(SU2("1/2")) == [SU2("0"), SU2("1")]


def test_irreps_parse_pairs_regroup():
    irs = Irreps(O3, "4x0e+4x1o")
    assert irs.dim == 4 * 1 + 4 * 3
    assert irs.num_irreps == 8
    assert len(irs) == 2
    assert [(m, repr(ir)) for m, ir in irs.pairs()] == [(4, "0e"), (4, "1o")]
    built = Irreps(O3, [(4, O3("0e")), (4, O3("1o"))])
    assert built.pairs() == irs.pairs()
    merged = Irreps(O3, "1x2e+2x1o+1x2e").regroup()
    assert [(m, repr(ir)) for m, ir in merged.pairs()] == [(2, "1o"), (2, "2e")]
    with pytest.raises(ValueError):
        Irreps(O3, [(1, SO3("1"))])  # an irrep of the wrong group


def test_irreps_equality_and_hash():
    a = Irreps(O3, "4x0e+4x1o")
    assert a == Irreps(O3, "4x0e+4x1o")
    assert hash(a) == hash(Irreps(O3, "4x0e+4x1o"))
    assert a != Irreps(O3, "4x0e+2x1o")
    assert a != Irreps(O3, "4x1o+4x0e")  # by entries, not regrouped
    assert a != Irreps(SO3, "4x0+4x1")
    assert a != "4x0e+4x1o"


def test_clebsch_gordan_is_exact():
    (cg,) = clebsch_gordan(O3("1o"), O3("1o"), O3("0e"))
    assert cg.shape == (3, 3, 1)
    assert len(cg) == 3  # the three diagonal components
    for (i, j, k), coef in cg:
        assert i == j and k == 0
        assert isinstance(coef, AlgebraicReal)
        assert coef * coef == Fraction(1, 3)


def test_clebsch_gordan_forbidden_is_empty():
    assert clebsch_gordan(O3("1o"), O3("1o"), O3("1o")) == []


def test_reduced_symmetric_tensor_product_is_sparse_and_exact():
    blocks = reduced_symmetric_tensor_product(Irreps(O3, "0e+1o"), 2, [O3("0e"), O3("1o")])
    muls = {repr(ir): tensor.shape[-2] for ir, tensor in blocks}
    assert muls == {"0e": 2, "1o": 1}
    for ir, tensor in blocks:
        assert tensor.shape[:-2] == (4, 4)
        assert tensor.shape[-1] == ir.dim
    coeffs = [c for _, tensor in blocks for _, c in tensor]
    assert coeffs and all(isinstance(c, AlgebraicReal) for c in coeffs)
    for c in coeffs:
        (c * c).as_fraction()  # single surd ⇒ rational square; raises otherwise


def test_reduced_tensor_product_qk_scalar_routes():
    q = Irreps(O3, "0e+1o+2e")
    sph = Irreps(O3, "0e+1o+2e+3o")
    blocks = reduced_tensor_product([q, q, sph], keep=[O3.trivial()])
    assert len(blocks) == 1
    ir, tensor = blocks[0]
    assert ir == O3.trivial()
    assert tensor.shape[:-2] == (9, 9, 16)
    assert tensor.shape[-2] == 13
    for _, c in tensor:
        (c * c).as_fraction()  # single surd ⇒ rational square


def test_reduced_tensor_product_formula_symmetry():
    v = Irreps(O3, "1o")
    full = reduced_tensor_product([v, v, v])
    sym = reduced_tensor_product([v, v, v], formula="ijk=jik")

    def total(blocks):
        return sum(tensor.shape[-2] * ir.dim for ir, tensor in blocks)

    assert total(sym) < total(full)


def test_reduced_tensor_product_rejects_mismatch():
    q = Irreps(O3, "0e+1o")
    with pytest.raises(ValueError, match="indices"):
        reduced_tensor_product([q, q], formula="ijk=jik")
    with pytest.raises(ValueError, match="mixes groups"):
        reduced_tensor_product([q, Irreps(SO3, "1")])


def test_spherical_harmonic_basis_has_canonical_normalization_and_phase():
    vector = O3("1o")
    for degree in range(1, 5):
        out, basis = spherical_harmonic_basis(vector, degree)
        entries = list(basis)
        norm_squared = sum(
            (coefficient * coefficient for _, coefficient in entries), AlgebraicReal()
        )
        expected = O3.harmonic_normalization(degree)
        assert norm_squared == out.dim * expected * expected
        highest = (2,) * degree + (out.dim - 1,)
        assert basis.get(highest).sign() > 0
