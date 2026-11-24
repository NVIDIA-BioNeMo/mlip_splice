# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-irrep XLA-dot overrides on IrrepsLinear."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from e3 import O3, Irreps
from splice.config import dump_config, load_config
from splice.ops.linear import IrrepsLinear
from splice.precision import parse as parse_precision

IRREPS = Irreps(O3, "8x0e+8x1o+8x2e")
OUT_IRREPS = Irreps(O3, "16x0e+6x1o+6x2e")
_GPU = any(getattr(device, "platform", "") == "gpu" for device in jax.devices())
requires_gpu = pytest.mark.skipif(not _GPU, reason="split dot algorithms require a GPU")


def _features(irreps=IRREPS, n=7, seed=1):
    keys = jax.random.split(jax.random.key(seed), len(irreps))
    return {
        ir: jax.random.normal(key, (n, ir.dim, mul), jnp.float32)
        for key, (mul, ir) in zip(keys, irreps, strict=True)
    }


def _linear(irrep_precision=None, **kwargs) -> IrrepsLinear:
    params = dict(
        irreps_in=IRREPS,
        irreps_out=OUT_IRREPS,
        param_dtype=jnp.float32,
        key=jax.random.key(0),
        precision="f32",
        irrep_precision=irrep_precision,
    )
    params.update(kwargs)
    return IrrepsLinear(**params)


def test_omitted_irreps_keep_global_precision():
    op = _linear({"1o": "tf32x3"})
    assert op._mapped_precision
    assert op.precision.alias == "f32"
    assert op._dot_precision["0e"].alias == "f32"
    assert op._dot_precision["1o"].alias == "tf32x3"
    assert op._dot_precision["2e"].alias == "f32"
    assert all(w.dtype == jnp.float32 for w in op.w.values())


def test_precision_dict_is_rejected():
    with pytest.raises(ValueError, match="irrep_precision"):
        _linear(precision={"1o": "tf32x3"}, irrep_precision=None)


def test_null_override_is_rejected():
    with pytest.raises(ValueError, match="cannot be null"):
        _linear({"0e": None, "1o": "tf32x3"})


def test_unknown_irrep_override_is_rejected():
    with pytest.raises(ValueError, match="matching irrep"):
        _linear({"3o": "tf32x3"})


@requires_gpu
def test_mapped_forward_keeps_fp32_boundaries():
    op = _linear({"1o": "tf32x3"})
    y = jax.jit(lambda model, feats: model(feats))(op, _features())
    assert all(value.dtype == jnp.float32 for value in y.values())
    assert all(np.all(np.isfinite(np.asarray(value))) for value in y.values())


def test_mapping_mode_rejects_bf16_inputs():
    op = _linear({"0e": "bf16"})
    mixed = _features()
    mixed[O3.parse("0e")] = mixed[O3.parse("0e")].astype(jnp.bfloat16)
    with pytest.raises(ValueError, match="f32"):
        op(mixed)


@requires_gpu
@pytest.mark.parametrize(
    ("mode", "passes"),
    [("tf32x3", 3), ("bf16x3", 3), ("bf16x6", 6), ("bf16x9", 9)],
)
def test_split_compute_modes_lower_explicitly(mode, passes):
    op = _linear({"1o": mode})
    y = jax.jit(lambda model, feats: model(feats))(op, _features())
    text = jax.jit(lambda model, feats: model(feats)).lower(op, _features()).as_text()
    assert all(value.dtype == jnp.float32 for value in y.values())
    assert f"num_primitive_operations = {passes}" in text


@requires_gpu
def test_mapping_gradients_stay_on_fp32_masters():
    op = _linear({"0e": "bf16", "1o": "tf32x3"})
    x = _features()

    def loss(packed):
        model, feats = packed
        return sum(jnp.sum(value**2) for value in model(feats).values())

    op_grad, x_grad = eqx.filter_grad(loss)((op, x))
    assert all(w.dtype == jnp.float32 for w in op_grad.w.values())
    assert all(value.dtype == jnp.float32 for value in x_grad.values())


@requires_gpu
def test_channel_mixing_commutes_with_rotation():
    op = _linear({"0e": "bf16", "1o": "tf32x3"})
    x = _features()
    keys = iter(jax.random.split(jax.random.key(2), len(IRREPS)))
    actions = {}
    for _, ir in IRREPS:
        key = next(keys)
        if ir.is_scalar():
            actions[ir] = jnp.ones((1, 1), jnp.float32)
            continue
        q, _ = jnp.linalg.qr(jax.random.normal(key, (ir.dim, ir.dim)))
        parity = -1.0 if str(ir).endswith("o") else 1.0
        actions[ir] = parity * q

    def transform(feats):
        return {ir: jnp.einsum("ij,nju->niu", actions[ir], value) for ir, value in feats.items()}

    left = op(transform(x))
    right = {ir: jnp.einsum("ij,nju->niu", actions[ir], value) for ir, value in op(x).items()}
    for ir in left:
        actual, expected = np.asarray(left[ir]), np.asarray(right[ir])
        relative = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        assert relative < 5e-4, f"{ir} equivariance relative error {relative:.2e}"


def test_irrep_precision_round_trips_through_yaml(tmp_path):
    yaml_text = """
defaults:
  param_dtype: f32
  precision: f32
steps:
  - op: irreps_linear
    params:
      irreps_in: 8x0e+8x1o+8x2e
      irreps_out: 8x0e+8x1o+8x2e
      irrep_precision:
        1o: tf32x3
"""
    path = tmp_path / "map.yaml"
    path.write_text(yaml_text)
    cfg = load_config(path)
    step = next(s for s in cfg if s["op"] == "irreps_linear")
    assert step["params"]["precision"] == parse_precision("f32")
    assert step["params"]["irrep_precision"] == {"1o": parse_precision("tf32x3")}
    out = tmp_path / "round.yaml"
    dump_config(cfg, out, defaults={"param_dtype": "f32", "precision": "f32"})
    again = load_config(out)
    step2 = next(s for s in again if s["op"] == "irreps_linear")
    assert step2["params"]["precision"].alias == "f32"
    assert step2["params"]["irrep_precision"]["1o"].alias == "tf32x3"
    assert "0e" not in step2["params"]["irrep_precision"]
    assert "2e" not in step2["params"]["irrep_precision"]
