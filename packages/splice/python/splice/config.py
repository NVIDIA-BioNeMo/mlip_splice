# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The draupnir seam: SPLICE's config→chain layer, delegating to draupnir.

draupnir (vendored in `splice/draupnir`) is the sole config/chain engine. This module supplies the two application extension points draupnir needs —
a `registry` (SPLICE's global op registry) and a `decode`/`encode` hook (the
string↔object resolution for irreps / dtype / precision / activation) — and wraps
draupnir's four functions so existing call sites keep their signatures.

Ops are pure draupnir ops: `INPUTS` is inferred from `__call__` (parameters named
as the canonical context slots — `node_feats`, `edge_vectors`, `edge_sph`, `graph`,
…), `OUTPUTS` is a bare slot name, wiring is nominal (a step chains to the next
wherever an output name and the next input name match). The old primary-input /
`last`-threading loader is gone.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from e3 import O3, Irreps

import splice.ops  # noqa: F401 — import for op-registration side effects
from splice import draupnir, precision
from splice.registry import _REGISTRY

# ---------------------------------------------------------------- resolvers ----
# String → object (decode) and object → string (encode) for the conventional
# param keys. Content is identical to the pre-draupnir yaml_io resolvers.

# Storage dtypes for `param_dtype` (weight storage) and `dtype` (the cast op's
# activation target). Short aliases (f32/bf16/…) are the canonical form and share
# the base names of the precision vocabulary (splice.precision); the long jnp
# names are accepted on input for convenience.
_DTYPES: dict[str, Any] = {
    "f16": jnp.float16,
    "f32": jnp.float32,
    "f64": jnp.float64,
    "bf16": jnp.bfloat16,
    "float16": jnp.float16,
    "float32": jnp.float32,
    "float64": jnp.float64,
    "bfloat16": jnp.bfloat16,
}
_DTYPES_INV = {  # encode -> short canonical name
    jnp.dtype(jnp.float16): "f16",
    jnp.dtype(jnp.float32): "f32",
    jnp.dtype(jnp.float64): "f64",
    jnp.dtype(jnp.bfloat16): "bf16",
}

# YAML ``defaults.activation`` is copied into every op that takes ``activation``
# (``scalar_activation``, ``radial_mlp``). That is a per-layer choice, not a
# global, and ops whose nonlinearity is structural (``scalar_gated_activation``)
# omit the parameter so they are not swept along.
# TODO: move ``activation`` to per-step params and drop it from ``defaults``.
_ACTIVATIONS: dict[str, Any] = {
    "silu": jax.nn.silu,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "tanh": jnp.tanh,
}
_ACTIVATIONS_INV = {v: k for k, v in _ACTIVATIONS.items()}

_IRREPS_KEYS = frozenset(
    {
        "irreps_in",
        "irreps_in1",
        "irreps_in2",
        "irreps_out",
        "irreps_sh",
        "irreps_q",
        "irreps_k",
        "irreps_v",
        "filter_out",
        "target_irreps",
    }
)


_DTYPE_KEYS = frozenset({"dtype", "param_dtype", "out_dtype"})


def _decode_value(key: str, v: Any) -> Any:
    if key in _IRREPS_KEYS:
        return v if isinstance(v, Irreps) else Irreps(O3, str(v))
    if key in _DTYPE_KEYS:
        return _DTYPES[v] if isinstance(v, str) else v
    if key == "precision":
        return precision.parse(v)
    if key == "irrep_precision":
        if not isinstance(v, Mapping):
            raise ValueError("irrep_precision must be a mapping of irrep → preset")
        return {str(k): precision.parse(val) for k, val in v.items()}
    if key == "activation":
        return _ACTIVATIONS[v] if isinstance(v, str) else v
    return v


def _encode_value(key: str, v: Any) -> Any:
    if key in _IRREPS_KEYS:
        return str(v).replace(" ", "") if isinstance(v, Irreps) else v
    if key in _DTYPE_KEYS:
        dt = jnp.dtype(v)
        if dt in _DTYPES_INV:
            return _DTYPES_INV[dt]
        raise ValueError(f"can't encode dtype {v!r}")
    if key == "precision":
        return precision.encode(v)
    if key == "irrep_precision":
        if not isinstance(v, Mapping):
            raise ValueError("irrep_precision must be a mapping of irrep → preset")
        return {str(k): precision.encode(val) for k, val in v.items()}
    if key == "activation":
        if v in _ACTIVATIONS_INV:
            return _ACTIVATIONS_INV[v]
        raise ValueError(f"can't encode activation {v!r}; add it to config._ACTIVATIONS")
    return v


def decode(op_cls: type, params: dict) -> dict:
    """draupnir `decode` hook: resolve conventional string params to objects.

    Resolution is by key name (op-independent), so `op_cls` is unused.
    """
    return {k: _decode_value(k, v) for k, v in params.items()}


def encode(op_cls: type, params: dict) -> dict:
    """draupnir `encode` hook: the inverse of `decode`, for `dump_config`."""
    return {k: _encode_value(k, v) for k, v in params.items()}


def registry() -> Mapping[str, type]:
    """The op registry as a draupnir registry (name → op class)."""
    return _REGISTRY


# ----------------------------------------------------------- main entry pts ----


def load_config(path: str | Path, *, overrides: dict[str, Any] | None = None) -> list[dict]:
    """Load a YAML arch config to canonical form via draupnir (decode applied).

    SPLICE "resolved" cards — the format the converters write and
    ``SpliceCalculator`` reads — carry card-level keys (``species``, ``cutoff``,
    ``offsets``, ...) at the top level alongside the arch ``defaults``/``steps``.
    draupnir (a generic engine) only accepts ``defaults``/``steps``, so drop any
    extra top-level keys here; callers read the card-level ones from the raw YAML.
    """
    import yaml

    document = yaml.safe_load(Path(path).read_text())
    if isinstance(document, dict):
        document = {k: document[k] for k in ("defaults", "steps") if k in document}
    return draupnir.load_config(document, _REGISTRY, overrides=overrides or {}, decode=decode)


def build_chain(config: list[dict], *, key: jax.Array) -> draupnir.Chain:
    """Instantiate and wire the chain via draupnir."""
    return draupnir.build_chain(config, _REGISTRY, key)


def dump_config(
    config: list[dict],
    path: str | Path,
    *,
    defaults: dict[str, Any] | None = None,
    header: str | None = None,
) -> None:
    """Serialise a canonical config to YAML (encode applied), inverting load_config."""
    text = draupnir.dump_config(config, _REGISTRY, defaults=defaults or {}, encode=encode)
    if header:
        text = header.rstrip() + "\n" + text
    Path(path).write_text(text)


# ---- legacy-shape helpers: chain reconstruction + output-name lookup --------
# draupnir's Chain exposes `.ops` (tuple of op instances) and static `.specs`.
# Subsystems that used to reach into Chain's `.modules`/`.wirings`
# (µP, calibration, species-offset seeding, probe) go through these.

Chain = draupnir.Chain
Chain = draupnir.Chain  # legacy alias


def rebuild(chain, ops):
    """A copy of `chain` with its op instances replaced (wiring/specs preserved)."""
    return draupnir.Chain(ops=tuple(ops), specs=chain.specs)


def step_out_name(chain, i: int) -> str:
    """The primary output context name written by step `i`."""
    return chain.specs[i].outs[0][1][0]


def chain_out_names(chain) -> set[str]:
    """Every context name written by any step."""
    return {n for spec in chain.specs for _, names in spec.outs for n in names}
