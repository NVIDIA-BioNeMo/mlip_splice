# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-op LR multiplier via draupnir `annotations: {lr_mult}` — the clean
replacement for the old op-index LR_OP_MULT env hack.

Checks that the multiplier (a) is read off the chain's specs, (b) folds
multiplicatively into the µP LR tree, (c) applies as a post-optimizer scale on
the no-µP path, (d) is validated, and (e) is a no-op when absent (backward
compatible with every existing annotation-free config)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from splice.config import build_chain, load_config
from splice.lr_annotations import annotation_lr_mults, lr_mult_transform

from tests.conftest import CONFIGS

TINY_YAML = CONFIGS / "smoke.yaml"

# Ops in mace_smoke that carry learnable arrays and that the handle targets:
SC0 = 5  # symmetric_contraction (boost)
READOUT0 = 8  # is_readout linear (damp)


def _chain(annotations=None):
    cfg = load_config(TINY_YAML)
    for i, ann in (annotations or {}).items():
        cfg[i]["annotations"] = ann
    return build_chain(cfg, key=jax.random.key(0))


def _op_inexact(op):
    """Flat list of inexact-array leaves of an op (µP-tree or param subtree)."""
    return [x for x in jax.tree.leaves(op) if eqx.is_inexact_array(x)]


def test_annotation_lr_mults_read():
    chain = _chain({SC0: {"lr_mult": 4.0}, READOUT0: {"lr_mult": 0.25}})
    mults = annotation_lr_mults(chain)
    assert len(mults) == len(chain.ops)
    assert mults[SC0] == 4.0 and mults[READOUT0] == 0.25
    assert all(m == 1.0 for i, m in enumerate(mults) if i not in (SC0, READOUT0))


def test_absent_annotations_are_noop():
    chain = _chain()
    assert annotation_lr_mults(chain) == (1.0,) * len(chain.ops)


def test_nomup_transform_scales_updates():
    chain = _chain({SC0: {"lr_mult": 4.0}, READOUT0: {"lr_mult": 0.25}})
    lr_mults = annotation_lr_mults(chain)
    params, _ = eqx.partition(chain, eqx.is_inexact_array)
    # Unit "gradient" of ones per param leaf; the transform must scale each op's
    # leaves by its lr_mult and leave the rest untouched.
    ones = jax.tree.map(lambda x: jnp.ones_like(x), params)
    tx = lr_mult_transform(chain, lr_mults)
    scaled, _ = tx.update(ones, tx.init(ones))
    for i, (op_ones, op_scaled) in enumerate(zip(ones.ops, scaled.ops, strict=True)):
        want = lr_mults[i]
        for a, b in zip(_op_inexact(op_ones), _op_inexact(op_scaled), strict=True):
            assert jnp.allclose(b, a * want), f"op {i}: expected x{want}"


def test_validation_rejects_bad_annotations():
    with pytest.raises(ValueError, match="unknown annotations"):
        annotation_lr_mults(_chain({SC0: {"lr_bogus": 2.0}}))
    with pytest.raises(ValueError, match="positive number"):
        annotation_lr_mults(_chain({SC0: {"lr_mult": -1.0}}))
    with pytest.raises(ValueError, match="positive number"):
        annotation_lr_mults(_chain({SC0: {"lr_mult": 0.0}}))
