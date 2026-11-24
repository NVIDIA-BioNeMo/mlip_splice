# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static registry dump.

`dump_registry` serialises the metadata external tooling needs — constructor parameters and the
input/output interfaces — without instantiating ops. Interfaces are read from the class, exactly
as `build_chain` reads them from the instance; an interface the class does not carry (set in `__init__`, or
`__call__` not inferable) is omitted — dynamic, resolvable only at build.
"""

import inspect
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

import yaml

from .chain import _gather, _inferred_inputs, _slots

_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def _doc(cls: type) -> str | None:
    """The class docstring, ignoring the dataclass-generated signature stub."""
    doc = inspect.getdoc(cls)
    try:
        stub = cls.__name__ + str(inspect.signature(cls)).replace(" -> None", "")
    except (TypeError, ValueError):
        stub = None
    return doc if doc and doc != stub else None


def _params(cls: type) -> list[str]:
    """Named constructor parameters in signature order; `'...'` marks a `**kwargs` constructor."""
    if cls.__init__ is object.__init__:
        return []
    params = inspect.signature(cls.__init__).parameters.values()
    named = [p.name for p in params if p.kind in _KINDS and p.name != "self"]
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        named.append("...")
    return named


def _defaults(cls: type) -> dict[str, Any]:
    """Constructor defaults that have a YAML form; the rest stay listed in `params` only."""
    if cls.__init__ is object.__init__:
        return {}
    out: dict[str, Any] = {}
    for p in inspect.signature(cls.__init__).parameters.values():
        if p.kind in _KINDS and p.name != "self" and p.default is not inspect.Parameter.empty:
            try:
                yaml.safe_dump(p.default)
            except yaml.YAMLError:
                continue
            out[p.name] = p.default
    return out


def _inputs(cls: type, where: str) -> list[str]:
    """The INPUT slot names — the pytree structure is Python's calling convention, not protocol."""
    interface = getattr(cls, "INPUTS", None)
    if interface is None:
        return [n for n in _inferred_inputs(cls, where) if n != "self"]
    return [s for _, _, slots in _gather(interface, where) for s in slots]


def _outputs(cls: type, where: str) -> list[str]:
    """The OUTPUT slot names."""
    interface = getattr(cls, "OUTPUTS", None)
    if interface is None:
        raise TypeError(f"{where}: class declares no OUTPUTS")
    return list(_slots(interface, where)[0])


def dump_registry(registry: Mapping[str, type]) -> str:
    """Serialise a registry's static op metadata to YAML text.

    Emitted per op: `doc`, `params` (named constructor parameters, `'...'` marking `**kwargs`),
    `defaults` (the YAML-representable ones), `inputs`/`outputs` (flat slot-name lists). The
    interface lists appear only when the class carries them.
    """
    ops: dict[str, dict[str, Any]] = {}
    for name, cls in registry.items():
        entry: dict[str, Any] = {}
        if doc := _doc(cls):
            entry["doc"] = doc
        entry["params"] = _params(cls)
        if defaults := _defaults(cls):
            entry["defaults"] = defaults
        for key, derive in (("inputs", _inputs), ("outputs", _outputs)):
            with suppress(TypeError, ValueError):
                entry[key] = derive(cls, name)
        ops[name] = entry
    document = {"draupnir_registry": 1, "ops": ops}
    return yaml.safe_dump(document, default_flow_style=None, sort_keys=False, width=10_000)
