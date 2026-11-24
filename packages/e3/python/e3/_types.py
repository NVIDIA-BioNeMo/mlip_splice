# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Irrep`` and ``Irreps``: the objects the mathematics is written in terms of.

Both delegate every group-theoretic question to their ``Group``. What lives here
is only what Python needs — equality, hashing, ordering, parsing and printing —
so that :mod:`e3._groups` states the mathematics and nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from e3._groups import Group


class Irrep:
    __slots__ = ("group", "label")

    def __init__(self, group: Group, label):
        self.group = group
        self.label = label

    @property
    def dim(self) -> int:
        return self.group.dim(self)

    def is_scalar(self) -> bool:
        """True for the group's trivial representation, the only one an
        elementwise nonlinearity may touch without breaking equivariance."""
        return self == self.group.trivial()

    def selection_rule(self, other: Irrep) -> list[Irrep]:
        return self.group.selection_rule(self, other)

    def clebsch_gordan(self, other: Irrep, out: Irrep):
        return self.group.clebsch_gordan(self, other, out)

    def __mul__(self, other: Irrep) -> Irrep:
        return self.group.cartan(self, other)

    def __lt__(self, other: Irrep) -> bool:
        return self.group.less(self, other)

    def __eq__(self, other) -> bool:
        return isinstance(other, Irrep) and self.group == other.group and self.label == other.label

    def __hash__(self) -> int:
        return hash((self.group, self.label))

    def __repr__(self) -> str:
        return self.group.format(self)


class Irreps:
    __slots__ = ("_entries", "group")

    def __init__(self, group: Group, spec):
        self.group = group
        if isinstance(spec, str):
            self._entries = _parse_irreps(group, spec)
        else:
            entries = [(int(mul), ir) for mul, ir in spec]
            for _, ir in entries:
                if ir.group is not group:
                    raise ValueError(f"{ir!r} is not an irrep of {group}")
            self._entries = entries

    def pairs(self) -> list[tuple[int, Irrep]]:
        return list(self._entries)

    @property
    def dim(self) -> int:
        return sum(mul * ir.dim for mul, ir in self._entries)

    @property
    def num_irreps(self) -> int:
        return sum(mul for mul, _ in self._entries)

    def regroup(self) -> Irreps:
        pairs = sorted(((m, ir) for m, ir in self._entries if m > 0), key=lambda mi: mi[1])
        merged: list[tuple[int, Irrep]] = []
        for mul, ir in pairs:
            if merged and merged[-1][1] == ir:
                merged[-1] = (merged[-1][0] + mul, ir)
            else:
                merged.append((mul, ir))
        return Irreps(self.group, merged)

    def set_mul(self, mul: int) -> Irreps:
        return Irreps(self.group, [(mul, ir) for _, ir in self._entries])

    def filter(self, keep: list[Irrep]) -> Irreps:
        return Irreps(self.group, [(m, ir) for m, ir in self._entries if ir in keep])

    def __mul__(self, n: int) -> Irreps:
        return Irreps(self.group, [(n * m, ir) for m, ir in self._entries])

    __rmul__ = __mul__

    def __add__(self, other: Irreps) -> Irreps:
        if self.group is not other.group:
            raise ValueError(f"cannot add Irreps of {self.group} and {other.group}")
        return Irreps(self.group, self._entries + other._entries)

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Irreps):
            return NotImplemented
        return self.group is other.group and self._entries == other._entries

    def __hash__(self) -> int:
        return hash((self.group, tuple(self._entries)))

    def __repr__(self) -> str:
        return "+".join(f"{mul}x{ir}" for mul, ir in self._entries)


def common_group(per_index: list[Irreps]) -> Group:
    """The group shared by every entry, or an error naming the two that differ."""
    if not per_index:
        raise ValueError("per_index must be non-empty")
    group = per_index[0].group
    for irreps in per_index[1:]:
        if irreps.group is not group:
            raise ValueError(f"mixes groups: {group} and {irreps.group}")
    return group


def _parse_irreps(group: Group, s: str) -> list[tuple[int, Irrep]]:
    s = s.strip()
    if not s:
        return []
    entries = []
    for term in s.split("+"):
        mul, _, label = term.rpartition("x")
        entries.append((int(mul) if mul else 1, group.parse(label.strip())))
    return entries
