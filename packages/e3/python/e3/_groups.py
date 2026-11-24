# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The groups: O(3), SO(3) and SU(2), each stated as its own mathematics.

A group answers four questions about its irreducible representations — their
dimension, which ones a tensor product can produce, how it couples them, and how
they are ordered — plus how it writes them down. Clebsch-Gordan coefficients and
harmonic normalizations are exact ``AlgebraicReal`` values, never floating point.
"""

from __future__ import annotations

import math
from fractions import Fraction

from algebraic_real import AlgebraicReal, pow

from e3._native import (
    _so3_clebsch_gordan,
    _su2_clebsch_gordan,
)
from e3._types import Irrep


class Group:
    def irrep(self, label) -> Irrep:
        return Irrep(self, label)

    def __call__(self, spec) -> Irrep:
        if isinstance(spec, Irrep):
            if spec.group is not self:
                raise ValueError(f"{spec!r} is not an irrep of {self}")
            return spec
        if isinstance(spec, str):
            return self.parse(spec)
        raise TypeError(
            f"call a group with a string spec, not {type(spec).__name__}; "
            "use group.irrep(label) for a raw label"
        )

    def trivial(self) -> Irrep:
        raise NotImplementedError

    def dim(self, ir: Irrep) -> int:
        raise NotImplementedError

    def selection_rule(self, a: Irrep, b: Irrep) -> list[Irrep]:
        raise NotImplementedError

    def cartan(self, a: Irrep, b: Irrep) -> Irrep:
        raise NotImplementedError

    def clebsch_gordan(self, a: Irrep, b: Irrep, c: Irrep):
        raise NotImplementedError

    def less(self, a: Irrep, b: Irrep) -> bool:
        raise NotImplementedError

    def parse(self, s: str) -> Irrep:
        raise NotImplementedError

    def format(self, ir: Irrep) -> str:
        raise NotImplementedError

    def harmonic_normalization(self, degree: int) -> AlgebraicReal:
        raise NotImplementedError

    def __repr__(self) -> str:
        return type(self).__name__.lstrip("_")


def _cue_harmonic_normalization(degree: int) -> AlgebraicReal:
    return pow(
        Fraction(math.prod(range(1, 2 * degree + 2, 2)), math.factorial(degree)),
        Fraction(1, 2),
    )


class _O3(Group):
    def trivial(self):
        return self.irrep((0, 1))

    def dim(self, ir):
        ell, _ = ir.label
        return 2 * ell + 1

    def selection_rule(self, a, b):
        (la, pa), (lb, pb) = a.label, b.label
        return [self.irrep((ell, pa * pb)) for ell in range(abs(la - lb), la + lb + 1)]

    def cartan(self, a, b):
        (la, pa), (lb, pb) = a.label, b.label
        return self.irrep((la + lb, pa * pb))

    def clebsch_gordan(self, a, b, c):
        (la, pa), (lb, pb), (lc, pc) = a.label, b.label, c.label
        if pc != pa * pb:
            return []
        return _so3_clebsch_gordan(la, lb, lc)

    def less(self, a, b):
        def key(ir):
            ell, p = ir.label
            return (ell, -p * (-1) ** ell)  # natural-parity ordering within each degree

        return key(a) < key(b)

    def parse(self, s):
        s = s.strip()
        return self.irrep((int(s[:-1]), {"e": 1, "o": -1}[s[-1]]))

    def format(self, ir):
        ell, p = ir.label
        return f"{ell}{'e' if p > 0 else 'o'}"

    def harmonic_normalization(self, degree):
        return _cue_harmonic_normalization(degree)


class _SO3(Group):
    def trivial(self):
        return self.irrep(0)

    def dim(self, ir):
        return 2 * ir.label + 1

    def selection_rule(self, a, b):
        return [self.irrep(ell) for ell in range(abs(a.label - b.label), a.label + b.label + 1)]

    def cartan(self, a, b):
        return self.irrep(a.label + b.label)

    def clebsch_gordan(self, a, b, c):
        return _so3_clebsch_gordan(a.label, b.label, c.label)

    def less(self, a, b):
        return a.label < b.label

    def parse(self, s):
        return self.irrep(int(s.strip()))

    def format(self, ir):
        return str(ir.label)

    def harmonic_normalization(self, degree):
        return _cue_harmonic_normalization(degree)


class _SU2(Group):
    def trivial(self):
        return self.irrep(0)

    def dim(self, ir):
        return ir.label + 1

    def selection_rule(self, a, b):
        lo, hi = abs(a.label - b.label), a.label + b.label
        return [self.irrep(t) for t in range(lo, hi + 1, 2)]

    def cartan(self, a, b):
        return self.irrep(a.label + b.label)

    def clebsch_gordan(self, a, b, c):
        return _su2_clebsch_gordan(a.label, b.label, c.label)

    def less(self, a, b):
        return a.label < b.label

    def parse(self, s):
        s = s.strip()
        if "/" in s:
            num, den = s.split("/")
            twice_j = int(num)
            if den != "2" or twice_j % 2 == 0:
                raise ValueError(f"not a reduced half-integer spin: {s!r}")
            return self.irrep(twice_j)
        return self.irrep(2 * int(s))

    def format(self, ir):
        t = ir.label
        return str(t // 2) if t % 2 == 0 else f"{t}/2"

    def harmonic_normalization(self, degree):
        return AlgebraicReal(1)  # the SU(2) reduced symmetric basis is orthonormal


O3 = _O3()
SO3 = _SO3()
SU2 = _SU2()
