# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency and Python type identity contracts in fresh interpreters."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["algebraic_real", "e3", "loom"])
def test_import_boundaries(module):
    forbidden = {
        "algebraic_real": ("jax", "e3", "loom", "splice"),
        "e3": ("jax", "loom", "splice"),
        "loom": ("e3", "splice"),
    }[module]
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {module}; import sys; assert not set({forbidden!r}) & sys.modules.keys()",
        ],
        check=True,
    )


def test_native_coefficients_share_one_exact_type():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import pickle
from fractions import Fraction
from algebraic_real import AlgebraicReal, pow
import e3
value = pow(2, Fraction(1, 2))
assert type(value) is AlgebraicReal
assert pickle.loads(pickle.dumps(value)) == value
basis = e3.clebsch_gordan(e3.O3("1o"), e3.O3("1o"), e3.O3("0e"))[0]
assert list(basis)
assert all(type(x) is AlgebraicReal for _, x in basis)
assert all(pickle.loads(pickle.dumps(x)) == x for _, x in basis)
""",
        ],
        check=True,
    )


def test_no_legacy_modules_or_reexports():
    import importlib.util

    import loom
    import splice

    for module in ("splice.loom", "splice.e3"):
        assert importlib.util.find_spec(module) is None
    for name in ("pow", "AlgebraicReal", "exact"):
        assert not hasattr(loom, name)
        assert not hasattr(splice, name)


def test_public_exact_api_has_one_path():
    import algebraic_real
    import loom
    import splice

    assert algebraic_real.AlgebraicReal.__module__ == "algebraic_real"
    assert not hasattr(loom, "AlgebraicReal")
    assert not hasattr(splice, "sqrt")
    assert not hasattr(splice, "root")
    assert not hasattr(splice, "pow_rational")
