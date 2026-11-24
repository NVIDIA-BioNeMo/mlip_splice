# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""YAML round-trip + resolver tests."""

import jax
import jax.numpy as jnp
import pytest
from e3 import Irreps
from splice.config import build_chain, dump_config, load_config
from splice.ops.embedding import SpeciesEmbedding
from splice.ops.head import SpeciesOffsets
from splice.ops.linear import IrrepsLinear
from splice.optim import ADAM, MUON, muon_label_tree

from tests.conftest import CONFIGS


def test_smoke_yaml_loads_and_resolves():
    cfg = load_config(CONFIGS / "smoke.yaml")
    assert len(cfg) == 11
    # Defaults should have been merged + resolved on a step that takes param_dtype.
    species_emb = next(s for s in cfg if s["op"] == "species_embedding")
    assert species_emb["params"]["param_dtype"] is jnp.float32
    # Irreps strings should have become Irreps.
    a_linear = next(s for s in cfg if s["op"] == "irreps_linear")
    assert isinstance(a_linear["params"]["irreps_in"], Irreps)
    assert isinstance(a_linear["params"]["irreps_out"], Irreps)
    # Precision default should resolve to a real splice Precision.
    assert a_linear["params"]["precision"].alias == "f32"


def test_smoke_yaml_builds_chain():
    cfg = load_config(CONFIGS / "smoke.yaml")
    chain = build_chain(cfg, key=jax.random.key(0))
    assert len(chain.ops) == 11
    emb = next(op for op in chain.ops if isinstance(op, SpeciesEmbedding))
    offs = next(op for op in chain.ops if isinstance(op, SpeciesOffsets))
    emb_spec = chain.specs[chain.ops.index(emb)]
    offs_spec = chain.specs[chain.ops.index(offs)]
    assert [kw for kw, _, _ in emb_spec.ins] == ["species"]
    assert [kw for kw, _, _ in offs_spec.ins] == ["atom_energies", "species"]
    linears = [op for op in chain.ops if isinstance(op, IrrepsLinear)]
    assert all(op.optimizer == ("adam" if op.is_readout else "muon") for op in linears)
    labels = muon_label_tree(chain)
    for index, op in enumerate(chain.ops):
        if isinstance(op, IrrepsLinear):
            expected = ADAM if op.is_readout else MUON
            assert set(labels.ops[index].w.values()) == {expected}


def test_round_trip_preserves_semantics(tmp_path):
    cfg_in = load_config(CONFIGS / "smoke.yaml")
    out_path = tmp_path / "round_trip.yaml"
    dump_config(
        cfg_in,
        out_path,
        defaults={"param_dtype": "f32", "precision": "f32", "activation": "silu"},
    )
    cfg_out = load_config(out_path)
    # Op-level structure must be identical.
    assert [s["op"] for s in cfg_in] == [s["op"] for s in cfg_out]
    # Spot-check: irreps_in on the first irreps_linear matches.
    a_in = next(s for s in cfg_in if s["op"] == "irreps_linear")
    a_out = next(s for s in cfg_out if s["op"] == "irreps_linear")
    assert a_in["params"]["irreps_in"] == a_out["params"]["irreps_in"]
    assert a_in["params"]["precision"] == a_out["params"]["precision"]
    assert a_in["params"]["param_dtype"] is a_out["params"]["param_dtype"]


def test_defaults_dont_leak_into_ops_that_dont_take_them(tmp_path):
    """Defaults must not reach ops whose __init__ does not accept them."""
    # `irreps_add` is weightless: it would error if param_dtype were passed.
    cfg = load_config(CONFIGS / "smoke.yaml")
    add_step = next(s for s in cfg if s["op"] == "irreps_add")
    assert "param_dtype" not in add_step["params"]
    assert "precision" not in add_step["params"]


def test_flat_list_yaml_no_defaults(tmp_path):
    """Loader must also accept a bare list at the top level (no defaults block)."""
    yaml_text = "- {op: irreps_linear, params: {irreps_in: 8x0e, irreps_out: 8x0e}}\n"
    p = tmp_path / "flat.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p)
    assert len(cfg) == 1
    assert isinstance(cfg[0]["params"]["irreps_in"], Irreps)


def test_overrides_apply_to_all_matching_ops():
    """A card-style override `<op>.<field>: v` replaces that field in every matching step."""
    cfg_no_override = load_config(CONFIGS / "smoke.yaml")
    cfg_overridden = load_config(
        CONFIGS / "smoke.yaml",
        overrides={"message_passing.epsilon": 0.0596},
    )
    key = "message_passing"
    cw_orig = [s for s in cfg_no_override if s["op"] == key]
    cw_new = [s for s in cfg_overridden if s["op"] == key]
    assert len(cw_new) == 1
    assert all(s["params"]["epsilon"] == 0.0596 for s in cw_new)
    # And the un-overridden config should still hold the original.
    assert all(s["params"]["epsilon"] == 0.25 for s in cw_orig)


def test_overrides_reject_unknown_field():
    """Override that references a non-existent kwarg should error early."""
    with pytest.raises(ValueError, match="not a parameter of"):
        load_config(
            CONFIGS / "smoke.yaml",
            overrides={"message_passing.this_does_not_exist": 42},
        )
