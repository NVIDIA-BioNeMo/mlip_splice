# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How one Loom program is compiled."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, replace
from types import EllipsisType

from loom._native import compile_defaults

_SOURCE_DEFAULTS = compile_defaults()
_DEFAULTS = {**_SOURCE_DEFAULTS, "scheduling_time_limit_ms": None}


class _Budgets(Mapping[int, float | None]):
    """Immutable rule-indexed budgets suitable for a JAX primitive parameter.

    ``Compile`` is part of Loom's static compilation key. Plain dictionaries
    therefore cannot be retained: JAX requires primitive parameters to be
    hashable. The sorted tuple provides a canonical hash while this wrapper
    preserves the useful read-only ``Mapping`` interface and dictionary-like
    representation of the public input.
    """

    __slots__ = ("_items",)

    def __init__(self, items: tuple[tuple[int, float | None], ...]):
        object.__setattr__(self, "_items", items)

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name):
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __reduce__(self):
        # Reconstruct through __init__: blocked slot assignment would otherwise
        # prevent JAX/recipe cache pickle round-trips.
        return type(self), (self._items,)

    def __getitem__(self, key: int) -> float | None:
        for index, value in self._items:
            if index == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[int]:
        return (index for index, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))


type Budget = float | Mapping[int, float | None] | None | EllipsisType


@dataclass(frozen=True, slots=True)
class Compile:
    """Shared options for `loom.call`, `loom.compile`, and `loom.aot`.

    Omitted fields (``...``) inherit field by field: per-call options, then
    the innermost AOT context, then outer contexts, then compiler defaults.
    Explicit ``None`` disables a nullable limit (or selects automatic kernel
    splitting). The default shortlist costs at most 600,000 predicted ms,
    except that the first candidate and its support are always retained.

    Loom always rewrites arithmetic exactly over the algebraic numbers the
    program's coefficients live in. Declared ``algebraic_real.jax.exact`` values take
    precedence; other finite floating constants denote the exact rationals
    encoded by their bits, not nearby simple fractions. NaNs and infinities
    are not rational coefficients. A result can differ from a fixed IEEE-754
    evaluation order in the last bit, and differs deliberately where
    the special values disagree with the algebra: ``x * 0`` is ``0`` even for an
    infinity or a NaN, and a signed zero does not survive a reassociated sum.
    There is no option to turn this off.
    Exact arithmetic and domain indexing share a typed rewrite graph; sums
    remain symbolic. Extraction keeps alternative placements of work before
    and after accumulation. The rewrite space does not set kernel boundaries.

    Loom first builds one complete legal plan with bounded local rewrites and
    required materialization cuts. ``search_budget_ms`` gives optional source
    search one shared wall-clock horizon, measured from compilation start, and
    overrides the ambient budget set by `aot`. Time expiry does not prevent the
    first plan; the search-memory estimate budget below can stop before it.
    Optional frontier extraction is one operation; subsequent graph lowering
    and schedule passes are admitted in sequence. Checks are cooperative: a
    started operation can overrun the horizon, and timing noise can change the
    admitted prefix. Source search finishes before GPU tuning begins.
    ``shortlist_budget_ms`` limits predicted realization time when selecting a
    source-only shortlist after source search finishes. The estimate is the
    maximum of unique assembly cost divided by assembly workers and serial
    per-candidate tuning cost. Work per resident warp and a fixed-seed
    Boltzmann permutation at temperature 1 determine its order. Shared native
    compilations, guards and entry witnesses are charged once. The first ranked
    candidate and its support are retained even if they exceed the budget.
    Kiln assembles and measures only this frozen shortlist; measured times never
    feed back into selection. Overestimates can leave budget unused. This budget
    also overrides `aot`. Either budget may be a number or a mapping from the number
    of transpose rules already applied to the budget, for example
    ``{0: 2000, 1: 1600, 2: 200}``. Rule zero applies to the original program;
    rules one and two apply after one and two transposes. A depth takes the
    entry of the greatest rule not exceeding it, so a missing next-rule entry
    retains the current budget and ``{2: 200}`` is unset until two transposes
    have occurred. A ``None`` entry explicitly disables the budget until another
    numeric entry appears. A missing applicable entry or an empty mapping
    inherits from the next configuration layer. The same form is accepted
    in the `Compile` object passed to `aot` for the ambient budget, indexed
    by the same depth. It is the number of
    actual Loom transpose-rule applications that matters, not derivative order
    or internal AD depth: JVP derivations do not advance these budgets, while
    reverse mode and `jax.linear_transpose` select the next order when their
    transpose rule plans a new Loom program. Mixed forward/reverse
    differentiation therefore advances only for programs derived by transpose.
    Kiln retains the fastest measured complete plan among the candidates it
    finishes. Search elapsed time and predicted realization time are distinct
    budgets; the latter is not a runtime timeout.
    ``search_memory_budget_bytes`` defaults to 16 GiB of estimated search volume.
    The counter charges twice the reported PTX bytes plus 16 KiB per complete
    candidate, allowing roughly for compiler overhead and plan metadata.
    Repeated source units are charged again; this is not a live-allocation
    census. ``None`` disables the limit. No process RAM is measured. Charges
    follow source attempts, so a large attempt can overrun; complete plans are
    saved before stopping. Stopping without any plan raises an error. This is
    a coarse search limit, not a hard RAM bound or a limit on later GPU tuning.
    ``max_workspace_bytes`` rejects candidates whose private scratch exceeds it.
    ``max_kernel_statements`` bounds the statements one generated kernel may
    hold, splitting it further when it would exceed that.
    ``max_algebraic_work`` bounds exact work across one XIR execution.
    ``max_algebraic_expression_work`` bounds one exact expression; exhausted
    expressions stay unfolded while independent expressions continue.

    ``scheduling_time_limit_ms`` limits opening new runtime tuning candidates,
    not completion of in-flight work. ``None`` measures all candidates; zero
    selects a feasible fallback without measurement. This is a runtime policy,
    never an input to source search or shortlist selection.

    ``transpose_depth`` is the number of transpose rules already applied to this
    program; it selects the rule-indexed budget entries above and rises only
    through `advance_transpose`. Every field participates in the JAX static
    compilation key. The AOT recipe cache uses resolved source settings and
    runtime policy, so equivalent configuration layers can share a recipe.
    """

    search_budget_ms: Budget = ...
    shortlist_budget_ms: Budget = ...
    max_workspace_bytes: int | None | EllipsisType = ...
    max_kernel_statements: int | None | EllipsisType = ...
    max_algebraic_work: int | EllipsisType = ...
    max_algebraic_expression_work: int | EllipsisType = ...
    transpose_depth: int = 0
    search_memory_budget_bytes: int | None | EllipsisType = ...
    scheduling_time_limit_ms: float | None | EllipsisType = ...

    def __post_init__(self):
        for name in ("search_budget_ms", "shortlist_budget_ms"):
            value = getattr(self, name)
            object.__setattr__(self, name, _budget(value, name.removesuffix("_budget_ms")))
        for name in (
            "max_workspace_bytes",
            "search_memory_budget_bytes",
            "max_kernel_statements",
            "max_algebraic_work",
            "max_algebraic_expression_work",
        ):
            value = getattr(self, name)
            nullable = name not in {"max_algebraic_work", "max_algebraic_expression_work"}
            if value is ... or (nullable and value is None):
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"loom {name} must be an integer" + (" or None" if nullable else "")
                )
            minimum = 0 if name.endswith("_bytes") else 1
            if not minimum <= value < 1 << 64:
                bound = "nonnegative" if minimum == 0 else "positive"
                raise ValueError(f"loom {name} must be {bound} and fit an unsigned 64-bit integer")
        value = self.scheduling_time_limit_ms
        if value is not ... and value is not None:
            if isinstance(value, bool):
                raise TypeError("loom scheduling_time_limit_ms must be a number or None")
            value = float(value)
            if not math.isfinite(value) or value < 0 or value / 1000 >= 1 << 64:
                raise ValueError(
                    "loom scheduling_time_limit_ms must be finite, nonnegative, and duration-representable"
                )
            object.__setattr__(self, "scheduling_time_limit_ms", value)
        if isinstance(self.transpose_depth, bool) or not isinstance(self.transpose_depth, int):
            raise TypeError("loom transpose_depth must be an integer")
        if self.transpose_depth < 0:
            raise ValueError("loom transpose_depth must be nonnegative")

    @property
    def current_search_budget_ms(self) -> float | None | EllipsisType:
        """Search budget at this depth; ``...`` means inherit."""
        return budget_at(self.search_budget_ms, self.transpose_depth)

    @property
    def current_shortlist_budget_ms(self) -> float | None | EllipsisType:
        """Shortlist budget at this depth; ``...`` means inherit."""
        return budget_at(self.shortlist_budget_ms, self.transpose_depth)

    def resolve(self, *fallbacks: Compile) -> Compile:
        """Resolve each field at this program's depth against ordered fallbacks."""
        if any(not isinstance(layer, Compile) for layer in fallbacks):
            raise TypeError("loom configuration must be a loom.Compile")
        values = {}
        layers = (self, *fallbacks)
        for field in fields(self):
            name = field.name
            if name == "transpose_depth":
                continue
            for layer in layers:
                value = getattr(layer, name)
                if name in {"search_budget_ms", "shortlist_budget_ms"}:
                    value = budget_at(value, self.transpose_depth)
                if value is not ...:
                    break
            values[name] = _DEFAULTS[name] if value is ... else value
        return Compile(**values, transpose_depth=self.transpose_depth)

    def source_kwargs(self) -> dict:
        """Native source-compiler inputs; call on resolved options."""
        values = {name: getattr(self, name) for name in _SOURCE_DEFAULTS}
        if any(value is ... or isinstance(value, Mapping) for value in values.values()):
            raise ValueError("resolve Compile before reading source inputs")
        return values

    def advance_transpose(self) -> Compile:
        """Return the same configuration at the next transpose-rule depth."""
        return replace(self, transpose_depth=self.transpose_depth + 1)


def _budget(value: Budget, name: str) -> Budget:
    if value is ...:
        return ...
    if value is None or not isinstance(value, Mapping):
        return _positive_ms(value, name)
    items = []
    for index, budget in value.items():
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"loom {name}_budget_ms rule indices must be integers")
        if index < 0:
            raise ValueError(f"loom {name}_budget_ms rule indices must be nonnegative")
        items.append((index, _positive_ms(budget, name)))
    return _Budgets(tuple(sorted(items))) if items else ...


def budget_at(value: Budget, depth: int) -> float | None | EllipsisType:
    """Resolve a budget at a transpose depth.

    A scalar (or ``None``) ignores the depth. A mapping takes the entry of the
    greatest rule not exceeding ``depth`` — the last budget set at or before this
    many transposes — and inherits (``...``) when no such entry exists. An entry of
    ``None`` explicitly disables the limit. A call's own budget and the ambient
    `aot` budget use this same indexing rule.
    """
    if not isinstance(value, Mapping):
        return value
    applicable = [rule for rule in value if rule <= depth]
    return value[max(applicable)] if applicable else ...


def _positive_ms(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"loom {name}_budget_ms must be a number")
    budget = float(value)
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError(f"loom {name}_budget_ms must be finite and positive")
    return budget
