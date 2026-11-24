# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Instantiation, binding, and execution.

Ops are instances declaring `INPUTS` and `OUTPUTS` as pytrees of slot names; each string leaf is a
slot, serving as the slot's default context name. `INPUTS` is inferred from `__call__` when
absent. The registry is any mapping `name -> op class`.
"""

import inspect
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any, NamedTuple

import equinox as eqx
import jax
from jax.tree_util import PyTreeDef


class FrozenMap(Mapping):
    """An immutable, hashable mapping — usable as static pytree metadata.

    `StepSpec` is stored in the chain's static treedef, so its fields must be hashable; a plain
    dict would break `jit`. Values must themselves be hashable (they are hashed lazily on first
    `hash()`). Reads use the ordinary mapping interface (`m["k"]`, `m.get("k")`).
    """

    __slots__ = ("_d", "_h")

    def __init__(self, d: Mapping[str, Any] = ()):
        self._d = dict(d)
        self._h: int | None = None

    def __getitem__(self, k: str) -> Any:
        return self._d[k]

    def __iter__(self):
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __hash__(self) -> int:
        if self._h is None:
            self._h = hash(frozenset(self._d.items()))
        return self._h

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Mapping) and dict(other) == self._d

    def __repr__(self) -> str:
        return f"FrozenMap({self._d!r})"


class StepSpec(NamedTuple):
    op: str  # registry name
    ins: tuple[tuple[str, PyTreeDef, tuple[str, ...]], ...]  # (kwarg, structure, context names)
    outs: tuple[tuple[str, tuple[str, ...]], ...]  # (slot, context names) — a write may fan out
    out_structure: PyTreeDef  # the return contract: the result must match it exactly
    annotations: FrozenMap = FrozenMap()  # opaque per-step metadata for consumers


def _slots(tree: Any, where: str) -> tuple[tuple[str, ...], PyTreeDef]:
    """Flatten an interface pytree to its slots — the string leaves — and its structure.

    A slot is identified by its name: names must be pairwise distinct.
    """
    leaves, structure = jax.tree.flatten(tree)
    if bad := [x for x in leaves if not isinstance(x, str)]:
        raise TypeError(f"{where}: interface leaves must be slot names, got {bad}")
    if len(set(leaves)) != len(leaves):
        raise ValueError(f"{where}: slot names must be pairwise distinct, got {leaves}")
    return tuple(leaves), structure


def _gather(interface: Any, where: str) -> tuple[tuple[str, PyTreeDef, tuple[str, ...]], ...]:
    """Normalize INPUTS to (call keyword, structure, slot names) triples.

    Flat forms — a name or a sequence of names — make each slot its own call keyword. The mapping
    form gathers: each keyword's argument is assembled as its pytree of slots, keys sorted.
    Slot names must be pairwise distinct across the whole interface.
    """
    if isinstance(interface, str):
        interface = (interface,)
    if isinstance(interface, Mapping):
        if bad := [k for k in interface if not isinstance(k, str)]:
            raise TypeError(f"{where}: INPUTS keywords must be strings, got {bad}")
        items = [(k, interface[k]) for k in sorted(interface)]
    elif isinstance(interface, Sequence):
        if bad := [s for s in interface if not isinstance(s, str)]:
            raise TypeError(f"{where}: sequence INPUTS elements must be names, got {bad}")
        items = [(s, s) for s in interface]
    else:
        raise TypeError(
            f"{where}: INPUTS must be str, sequence, or mapping, got {type(interface).__name__}"
        )
    triples = []
    for kwarg, tree in items:
        slots, structure = _slots(tree, where)
        triples.append((kwarg, structure, slots))
    names = [s for _, _, slots in triples for s in slots]
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: slot names must be pairwise distinct, got {names}")
    return tuple(triples)


def _inferred_inputs(op: Any, where: str) -> tuple[str, ...]:
    """The named parameters of `__call__`, in declaration order."""
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    params = inspect.signature(op.__call__).parameters.values()
    if bad := [p.name for p in params if p.kind not in kinds]:
        raise TypeError(
            f"{where}: cannot infer INPUTS — __call__ parameters {bad} are not "
            "keyword-bindable; declare INPUTS on the op"
        )
    return tuple(p.name for p in params)


def _bind(spec: Any, slots: tuple[str, ...], where: str) -> tuple[tuple[str, str], ...]:
    """Resolve an `in:`/`out:` binding to one context name per slot.

    Two forms: absent → declared names; string → the single slot; mapping → by slot
    name, unmentioned slots → declared names.
    """
    if spec is None:
        return tuple((s, s) for s in slots)
    if isinstance(spec, str):
        if len(slots) != 1:
            raise ValueError(
                f"{where}: string binding but interface has {len(slots)} slots {slots}"
            )
        return ((slots[0], spec),)
    if isinstance(spec, Mapping):
        if unknown := spec.keys() - set(slots):
            raise ValueError(f"{where}: unknown slots {sorted(unknown)}; interface is {slots}")
        return tuple((s, spec.get(s, s)) for s in slots)
    raise TypeError(
        f"{where}: binding must be absent, a name, or a mapping, got {type(spec).__name__}"
    )


class Chain(eqx.Module):
    """The built program: op instances (pytree subtrees) with static slot bindings.

    Calling the chain executes it: the context starts as a shallow copy of the host mapping, steps
    run in sequence order — gather arguments, call, scatter the result — reads observe the most
    recent write, writes replace. Returns the final context.
    """

    ops: tuple[Any, ...]
    specs: tuple[StepSpec, ...] = eqx.field(static=True)

    @property
    def inputs(self) -> frozenset[str]:
        """The chain's free names: read by some step and written by none before it."""
        written: set[str] = set()
        free: set[str] = set()
        for spec in self.specs:
            free |= {name for _, _, names in spec.ins for name in names} - written
            written |= {name for _, names in spec.outs for name in names}
        return frozenset(free)

    def __call__(self, context: Mapping[str, Any]) -> dict[str, Any]:
        ctx = dict(context)
        if missing := self.inputs - ctx.keys():
            raise KeyError(f"chain inputs missing from context: {sorted(missing)}")
        for i, (op, spec) in enumerate(zip(self.ops, self.specs, strict=True)):
            args = {
                kwarg: jax.tree.unflatten(structure, [ctx[name] for name in names])
                for kwarg, structure, names in spec.ins
            }
            result = op(**args)
            try:
                values = spec.out_structure.flatten_up_to(result)
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"step {i} ({spec.op}): result does not match OUTPUTS structure "
                    f"{spec.out_structure}: {e}"
                ) from e
            for (_, names), value in zip(spec.outs, values, strict=True):
                for name in names:
                    ctx[name] = value
        return ctx


def build_chain(
    config: Sequence[Mapping[str, Any]], registry: Mapping[str, type], key: jax.Array
) -> Chain:
    """Instantiate each step through the registry and resolve its slot bindings.

    One PRNG key per step is derived from `key` and passed as `key` when the constructor accepts
    that parameter and `params` omits it. Interfaces are read from the instance — an op may make
    them config-dependent — and INPUTS is inferred from `__call__` when absent. Output bindings
    within a step must be pairwise distinct.
    """
    keys = jax.random.split(key, len(config)) if config else ()
    ops: list[Any] = []
    specs: list[StepSpec] = []
    for i, (step, k) in enumerate(zip(config, keys, strict=True)):
        name = step["op"]
        cls = registry[name]
        params = dict(step.get("params") or {})
        if "key" in inspect.signature(cls.__init__).parameters:
            params.setdefault("key", k)
        op = cls(**params)
        interface = getattr(op, "INPUTS", None)
        if interface is None:
            interface = _inferred_inputs(op, f"step {i} ({name})")
        gather = _gather(interface, f"step {i} ({name}) in")
        in_slots = tuple(s for _, _, slots in gather for s in slots)
        bound = iter(_bind(step.get("in"), in_slots, f"step {i} ({name}) in"))
        ins = tuple(
            (kwarg, structure, tuple(n for _, n in islice(bound, len(slots))))
            for kwarg, structure, slots in gather
        )
        if (out_interface := getattr(op, "OUTPUTS", None)) is None:
            raise TypeError(f"step {i} ({name}): op declares no OUTPUTS")
        out_slots, out_structure = _slots(out_interface, f"step {i} ({name}) OUTPUTS")
        outs = _bind(step.get("out"), out_slots, f"step {i} ({name}) out")
        out_names = [n for _, n in outs]
        if len(set(out_names)) != len(out_names):
            raise ValueError(f"step {i} ({name}) out: bindings not pairwise distinct: {out_names}")
        ops.append(op)
        annotations = FrozenMap(step.get("annotations") or {})
        specs.append(
            StepSpec(name, ins, tuple((s, (n,)) for s, n in outs), out_structure, annotations)
        )
    return Chain(ops=tuple(ops), specs=tuple(specs))
