# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import pickle
from fractions import Fraction

import algebraic_real
import numpy as np
import pytest


def test_constructor_reduces_a_repeated_reducible_polynomial():
    value = algebraic_real.AlgebraicReal.from_polynomial([12, -4, -12, 4, 3, -1], 0)
    assert value.minimal_polynomial() == [-2, 0, 1]
    assert value == -algebraic_real.pow(2, Fraction(1, 2))


def test_constructor_accepts_any_coefficient_iterable():
    coefficients = (coefficient for coefficient in (-2, 0, 1))
    assert algebraic_real.AlgebraicReal.from_polynomial(coefficients, 1) == algebraic_real.pow(
        2, Fraction(1, 2)
    )


def test_cubic_arithmetic_and_rational_powers_are_exact():
    cube_root = algebraic_real.AlgebraicReal.from_polynomial([-2, 0, 0, 1], 0)
    assert cube_root**3 == 2
    assert cube_root.trace() == 0
    assert cube_root.norm() == 2
    assert algebraic_real.pow(2, Fraction(1, 3)) == cube_root
    assert algebraic_real.pow(-8, Fraction(1, 3)) == -2
    assert algebraic_real.pow(-8, Fraction(2, 3)) == 4
    assert algebraic_real.pow(2, Fraction(2, 3)) == cube_root * cube_root
    assert algebraic_real.AlgebraicReal(4) ** Fraction(1, 2) == 2
    with pytest.raises(ValueError):
        algebraic_real.pow(-2, Fraction(1, 2))
    with pytest.raises(TypeError):
        cube_root**0.5
    with pytest.raises(TypeError):
        algebraic_real.pow(2, 0.5)


def test_fractional_base_and_exponent_stay_exact():
    value = algebraic_real.pow(base=Fraction(3, 4), exponent=Fraction(5, 7))
    assert value.minimal_polynomial() == [-243, 0, 0, 0, 0, 0, 0, 1024]


def test_power_edge_cases_are_exact_or_refused():
    assert algebraic_real.pow(np.int64(4), np.int64(2)) == 16
    assert algebraic_real.pow(7, 0) == 1
    assert algebraic_real.pow(4, -1) == Fraction(1, 4)
    assert algebraic_real.pow(0, Fraction(1, 2)) == 0
    with pytest.raises(ZeroDivisionError):
        algebraic_real.pow(0, -1)
    with pytest.raises(TypeError):
        algebraic_real.pow(2, algebraic_real.AlgebraicReal(1))
    with pytest.raises(ValueError):
        algebraic_real.pow(2, Fraction(1, 4_097))
    with pytest.raises(OverflowError):
        algebraic_real.pow(2, Fraction(2**40, 3))


def test_square_power_keeps_multiquadratic_encoding():
    square_root = algebraic_real.pow(2, Fraction(1, 2))
    reduced_root = algebraic_real.pow(8, Fraction(1, 6))
    assert square_root._algebraic_real_data()[0] == "m"
    assert reduced_root == square_root
    assert reduced_root._algebraic_real_data()[0] == "m"
    assert algebraic_real.pow(2, Fraction(1, 4)) > 0


def test_float_is_exact_only_in_comparison():
    value = algebraic_real.AlgebraicReal(Fraction(1, 10))
    assert value != 0.1
    assert algebraic_real.AlgebraicReal(1) == 1.0
    with pytest.raises(TypeError):
        _ = value + 0.1


def test_rounding_and_pickle_cover_both_representations():
    for value in (algebraic_real.pow(2, Fraction(1, 2)), algebraic_real.pow(2, Fraction(1, 3))):
        assert math.floor(value) == 1
        assert float(value) > 1
        assert pickle.loads(pickle.dumps(value)) == value
        assert eval(repr(value), {"AlgebraicReal": algebraic_real.AlgebraicReal}) == value
