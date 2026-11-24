# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-op learning-rate multipliers, read from the config.

Each config step may carry ``annotations: {lr_mult: <float>}``. The multiplier
travels with the config, next to the op it scales, so it survives arch edits.
Absent -> 1.0.

``lr_mult_transform`` turns those per-op factors into an optax
``GradientTransformation`` that multiplies each op's updates by its factor.
Compose it *after* a plain optimizer::

    optax.chain(optax.adam(lr), lr_mult_transform(chain, annotation_lr_mults(chain)))
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from splice.config import Chain, rebuild

# The step-level `annotations` keys this package interprets. The config engine
# carries `annotations` opaquely; the trainer owns their meaning + validation.
_KNOWN_ANNOTATIONS = frozenset({"lr_mult"})


def _replace_arrays(module: Any, arr_to_scalar: Callable[[jax.Array], float]) -> Any:
    """Return a copy of `module` with each inexact-array leaf replaced by a
    0-d jnp.array carrying `arr_to_scalar(arr)`.

    Bypasses eqx.Module's frozen attributes and custom `__init__` (the same
    pattern as `splice.ops._common.replace_fields`).
    """
    new_module = object.__new__(type(module))
    for f in dataclasses.fields(module):
        val = getattr(module, f.name)
        new_val = jax.tree.map(
            lambda x: jnp.asarray(arr_to_scalar(x), x.dtype) if eqx.is_inexact_array(x) else x,
            val,
        )
        object.__setattr__(new_module, f.name, new_val)
    return new_module


def annotation_lr_mults(chain: Chain) -> tuple[float, ...]:
    """Per-op LR multipliers read from each step's draupnir ``annotations``. Absent → 1.0. This is the clean replacement for the old
    op-index ``LR_OP_MULT`` env hack: the multiplier travels with the config, next
    to the op it scales, so it survives arch edits.

    SPLICE owns validation of the annotation keys it defines: an unknown key or a
    non-positive ``lr_mult`` is an error (fail loud rather than silently no-op).
    """
    mults: list[float] = []
    for i, spec in enumerate(chain.specs):
        ann = spec.annotations
        if unknown := set(ann) - _KNOWN_ANNOTATIONS:
            raise ValueError(
                f"step {i} ({spec.op}): unknown annotations {sorted(unknown)}; "
                f"SPLICE understands {sorted(_KNOWN_ANNOTATIONS)}"
            )
        m = ann.get("lr_mult", 1.0)
        if not isinstance(m, (int, float)) or isinstance(m, bool) or m <= 0:
            raise ValueError(
                f"step {i} ({spec.op}): annotations.lr_mult must be a positive number, got {m!r}"
            )
        mults.append(float(m))
    return tuple(mults)


def _const_lr_mult_tree(chain: Chain, lr_mults: tuple[float, ...]) -> Chain:
    """Chain-shaped tree with every learnable array replaced by its op's ``lr_mult``
    constant (0-d). For the no-µP path: composed via :func:`_scale_updates_by_tree`."""
    new_modules = tuple(
        _replace_arrays(mod, lambda x, m=float(m): m)
        for mod, m in zip(chain.ops, lr_mults, strict=True)
    )
    return rebuild(chain, new_modules)


def lr_mult_transform(chain: Chain, lr_mults: tuple[float, ...]) -> optax.GradientTransformation:
    """A no-µP GradientTransformation applying per-op LR multipliers: it multiplies
    each op's updates by its ``lr_mult``. Compose *after* a plain optax optimizer
    (`optax.chain(optax.adam(lr), lr_mult_transform(chain, lr_mults))`)."""
    lr_tree = _const_lr_mult_tree(chain, lr_mults)
    lr_params, _ = eqx.partition(lr_tree, eqx.is_inexact_array)
    return _scale_updates_by_tree(lr_params)


def _scale_updates_by_tree(lr_tree: Any) -> optax.GradientTransformation:
    """optax transformation: pre-multiply updates by `lr_tree` element-wise.

    `lr_tree` must have the same PyTree structure as the updates (which come
    from `eqx.partition(chain, eqx.is_inexact_array)`'s params partition).
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        new_updates = jax.tree.map(lambda u, lr: u * lr, updates, lr_tree)
        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)
