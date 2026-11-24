# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config loading and dumping.

`load_config` turns a YAML document into the canonical form — a list of step mappings with fully
resolved `params`. `dump_config` inverts it: parameters equal to `defaults` are elided and each
step is written as one flow-style line.
"""

import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

_STEP_KEYS = {"op", "params", "in", "out", "annotations"}


def _ctor_params(cls: type) -> frozenset[str]:
    """Named constructor parameters; a `**kwargs` constructor attracts no defaults."""
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    sig = inspect.signature(cls.__init__)
    return frozenset(p.name for p in sig.parameters.values() if p.kind in kinds) - {"self"}


def load_config(
    document: str | Path | list | Mapping,
    registry: Mapping[str, type],
    overrides: Mapping[str, Any] | None = None,
    decode: Callable[[type, dict], dict] | None = None,
) -> list[dict[str, Any]]:
    """Load a document into canonical form: `[{"op", "params", "in"?, "out"?, "annotations"?}, ...]`.

    `document` is YAML text (`str`), a file (`Path`), or the parsed object (`list | Mapping`).
    Each step's `params` passes through three stages: `defaults` keys are copied in when the
    constructor accepts them and `params` omits them; `overrides` entries `"<op>.<field>": value`
    overwrite the field in every matching step (unmatched ops are skipped, a field outside a
    matched constructor's signature is an error); `decode(op_cls, params)` maps serialised values
    to application objects.
    """
    if isinstance(document, Path):
        document = yaml.safe_load(document.read_text())
    elif isinstance(document, str):
        document = yaml.safe_load(document)
    if isinstance(document, list):
        defaults: Mapping[str, Any] = {}
        steps = document
    elif isinstance(document, Mapping):
        if unknown := document.keys() - {"defaults", "steps"}:
            raise ValueError(f"top-level keys must be 'defaults'/'steps', got {sorted(unknown)}")
        if "steps" not in document:
            raise ValueError("top-level mapping must have 'steps'")
        defaults = document.get("defaults") or {}
        steps = document["steps"]
    else:
        raise ValueError(
            f"document must be a sequence of steps or a mapping with 'steps', "
            f"got {type(document).__name__}"
        )

    by_op: dict[str, dict[str, Any]] = {}
    for k, v in (overrides or {}).items():
        op_name, dot, field = k.partition(".")
        if not dot:
            raise ValueError(f"override key {k!r} must be '<op>.<field>'")
        by_op.setdefault(op_name, {})[field] = v

    config: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        if "op" not in step:
            raise ValueError(f"step {i}: missing 'op'")
        name = step["op"]
        if unknown := step.keys() - _STEP_KEYS:
            raise ValueError(f"step {i} ({name}): unknown keys {sorted(unknown)}")
        cls = registry[name]
        sig = _ctor_params(cls)
        params = dict(step.get("params") or {})
        for k, v in defaults.items():
            if k in sig and k not in params:
                params[k] = v
        for field, v in by_op.get(name, {}).items():
            if field not in sig:
                raise ValueError(
                    f"override {name}.{field}: not a parameter of {cls.__name__}.__init__"
                )
            params[field] = v
        if decode is not None:
            params = decode(cls, params)
        out: dict[str, Any] = {"op": name, "params": params}
        for k in ("in", "out", "annotations"):
            if k in step:
                out[k] = step[k]
        config.append(out)
    return config


def _flow(obj: Any) -> str:
    return yaml.safe_dump(obj, default_flow_style=True, sort_keys=False, width=10_000).strip()


def dump_config(
    config: list[Mapping[str, Any]],
    registry: Mapping[str, type],
    defaults: Mapping[str, Any] | None = None,
    encode: Callable[[type, dict], dict] | None = None,
) -> str:
    """Serialise a canonical config back to YAML text, inverting `load_config`.

    `encode(op_cls, params)` is `decode`'s inverse — same shape, receiving the class from the
    registry. `defaults`, given in serialised form, are elided from each step (stage 1 restores
    them on reload). Each step is written as one flow-style line.
    """
    defaults = dict(defaults or {})
    lines: list[str] = []
    if defaults:
        lines.append("defaults: " + _flow(defaults))
        lines.append("steps:")
    for step in config:
        params = dict(step.get("params") or {})
        if encode is not None:
            params = encode(registry[step["op"]], params)
        params = {k: v for k, v in params.items() if not (k in defaults and defaults[k] == v)}
        out: dict[str, Any] = {"op": step["op"]}
        if params:
            out["params"] = params
        for k in ("in", "out", "annotations"):
            if k in step:
                out[k] = step[k]
        lines.append("- " + _flow(out))
    return "\n".join(lines) + "\n"
