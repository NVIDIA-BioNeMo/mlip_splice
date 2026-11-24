# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused inference backend of ``radial_mlp`` against the XLA path.

The kernels are an optimization, so the reference is the op itself: same weights,
same input, compare. Covered here are the agreement of the forward and of the
input gradient (the force path), the aligned-row contract the kernels require,
the exactly-zero weight gradient
that makes this backend inference only, and the configurations the op must
refuse rather than quietly run.

Skipped unless Pallas can target the current GPU: its Triton lowering supports a
fixed set of device kinds, so a perfectly good card can be unsupported.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from splice.ops._pallas_mlp import BLOCK_ROWS, HAVE_PALLAS
from splice.ops.radial import PALLAS_INFERENCE, RadialMLP

SIZES = (8, 64, 64, 64, 512)  # the standard-MACE radial shape


def _pallas_targets_this_gpu() -> bool:
    if not HAVE_PALLAS or jax.default_backend() != "gpu":
        return False
    from jax.experimental.pallas import triton as plgpu

    probe = getattr(plgpu, "get_gpu_info", None)
    if probe is None:
        return True  # older JAX cannot be asked; try, and let a real failure show
    try:
        probe()
    except ValueError:
        return False
    return True


needs_pallas = pytest.mark.skipif(
    not _pallas_targets_this_gpu(), reason="Pallas cannot target this device"
)


def _mlp(
    backend,
    *,
    sizes=SIZES,
    precision="f32",
    normalize_output=False,
    forward_fusion="auto",
    backward_fusion="auto",
):
    return RadialMLP(
        list(sizes),
        normalize_output=normalize_output,
        backend=backend,
        forward_fusion=forward_fusion,
        backward_fusion=backward_fusion,
        precision=precision,
        param_dtype=jnp.float32,
        key=jax.random.key(0),
    )


def _bessel(num_edges, width=SIZES[0], seed=1):
    return jax.random.normal(jax.random.key(seed), (num_edges, width), jnp.float32)


@needs_pallas
@pytest.mark.parametrize("num_edges", [BLOCK_ROWS, 1024, 4096])
def test_forward_matches_xla(num_edges):
    x = _bessel(num_edges)
    want = jax.jit(_mlp("xla").__call__)(x)
    got = jax.jit(_mlp(PALLAS_INFERENCE).__call__)(x)
    assert got.shape == want.shape
    err = jnp.abs(got - want).max() / jnp.abs(want).max()
    assert float(err) < 1e-5


@needs_pallas
def test_input_gradient_matches_xla():
    """The input gradient is the force path, and the reason the backend exists."""
    x = _bessel(1024)
    grad = jax.jit(jax.grad(lambda z, m: (m(z) ** 2).sum()))
    want = grad(x, _mlp("xla"))
    got = grad(x, _mlp(PALLAS_INFERENCE))
    err = jnp.abs(got - want).max() / jnp.abs(want).max()
    assert float(err) < 1e-5


@needs_pallas
@pytest.mark.parametrize("forward_fusion", ["xla", "per_layer", "chain", "full", "auto"])
@pytest.mark.parametrize("backward_fusion", ["xla", "per_layer", "chain", "full", "auto"])
def test_fusion_strategies_match_xla(forward_fusion, backward_fusion):
    x = _bessel(1024)
    reference = _mlp("xla")
    candidate = _mlp(
        PALLAS_INFERENCE,
        forward_fusion=forward_fusion,
        backward_fusion=backward_fusion,
    )
    want_y = jax.jit(reference.__call__)(x)
    got_y = jax.jit(candidate.__call__)(x)
    fwd_err = jnp.abs(got_y - want_y).max() / jnp.abs(want_y).max()
    assert float(fwd_err) < 1e-5

    grad = jax.jit(jax.grad(lambda z, m: (m(z) ** 2).sum()))
    want_dx = grad(x, reference)
    got_dx = grad(x, candidate)
    bwd_err = jnp.abs(got_dx - want_dx).max() / jnp.abs(want_dx).max()
    assert float(bwd_err) < 1e-5


def test_auto_fusion_heuristic_uses_only_measured_shapes():
    from splice.ops._pallas_mlp import resolve_fusions

    standard = [
        np.empty((16, 64)),
        np.empty((64, 64)),
        np.empty((64, 64)),
        np.empty((64, 512)),
    ]
    assert resolve_fusions(standard, 4096, "auto", "auto") == (
        "chain",
        "chain",
    )

    wider = [
        np.empty((16, 128)),
        np.empty((128, 128)),
        np.empty((128, 128)),
        np.empty((128, 512)),
    ]
    assert resolve_fusions(
        wider, 32768, "auto", "auto", device_kind="NVIDIA H200", original_input_width=8
    ) == (
        "chain",
        "chain",
    )


def test_chain_tile_heuristic_only_retiles_the_4k_bucket():
    from splice.ops._pallas_mlp import _chain_config

    assert _chain_config(4096) == (16, 4)
    assert _chain_config(8192) == (64, 8)
    assert _chain_config(102400) == (64, 8)


def test_tf32_chain_halves_the_warp_count_above_the_small_bucket():
    """The tensor-core dot and the FMA dot want different warp counts, and tuning
    one value for both made tf32 slower than f32 (110 us against 85 us at 102k
    edges). The small tile is already at 4 warps and does not split."""
    from splice.ops._pallas_mlp import _chain_config, _layer_warps

    assert _chain_config(102400, allow_tf32=True) == (64, 4)
    assert _chain_config(102400, allow_tf32=False) == (64, 8)
    assert _chain_config(4096, allow_tf32=True) == _chain_config(4096, allow_tf32=False)
    assert (_layer_warps(True), _layer_warps(False)) == (4, 8)


def test_chain_fusion_rejects_unmeasured_hidden_shapes():
    from splice.ops._pallas_mlp import resolve_fusions

    weights = [np.empty((16, 64)), np.empty((64, 64)), np.empty((64, 512))]
    with pytest.raises(ValueError, match="exactly three 64-wide"):
        resolve_fusions(weights, 32768, "chain", "chain")


@needs_pallas
@pytest.mark.parametrize("num_edges", [BLOCK_ROWS, 4 * BLOCK_ROWS, 1600 * BLOCK_ROWS])
def test_output_has_one_row_per_edge(num_edges):
    got = jax.jit(_mlp(PALLAS_INFERENCE).__call__)(_bessel(num_edges))
    assert got.shape[0] == num_edges


@needs_pallas
def test_unaligned_batch_is_refused_not_worked_around():
    """Padding up to a whole block and slicing back was measured slower than XLA, so
    the op refuses the batch instead of silently doing something worse. The batching
    is ours to size, and the message says which knob to turn."""
    with pytest.raises(ValueError, match="base_edges"):
        jax.jit(_mlp(PALLAS_INFERENCE).__call__)(_bessel(4 * BLOCK_ROWS - 1))


@needs_pallas
def test_kernel_refuses_unaligned_rows_directly():
    """Both layers refuse: the op for the friendly message, the kernel entry so that a
    direct caller cannot slip past it either."""
    from splice.ops._pallas_mlp import fused_dense_mlp

    op = _mlp(PALLAS_INFERENCE)
    with pytest.raises(ValueError, match="does not pad"):
        fused_dense_mlp(
            _bessel(BLOCK_ROWS - 1),
            list(op.linears),
            [(1.0 / n) ** 0.5 for n in SIZES[:-1]],
            op.silu_k,
        )


@needs_pallas
def test_weight_gradient_is_zero():
    """Pinned, not aspirational: this backend is inference only and reports zero
    weight cotangents. A future change that starts producing real ones should have
    to delete this test on purpose."""
    x = _bessel(256)
    grads = jax.jit(jax.grad(lambda m, z: (m(z) ** 2).sum()))(_mlp(PALLAS_INFERENCE), x)
    assert all(bool((w == 0).all()) for w in grads.linears)
    xla_grads = jax.jit(jax.grad(lambda m, z: (m(z) ** 2).sum()))(_mlp("xla"), x)
    assert any(bool((w != 0).any()) for w in xla_grads.linears)


@needs_pallas
def test_normalize_output_matches_xla():
    """The RMS normalization remains an XLA operation after the kernels."""
    x = _bessel(1024)
    want = jax.jit(_mlp("xla", normalize_output=True).__call__)(x)
    got = jax.jit(_mlp(PALLAS_INFERENCE, normalize_output=True).__call__)(x)
    err = jnp.abs(got - want).max() / jnp.abs(want).max()
    assert float(err) < 1e-5


@needs_pallas
def test_tf32_is_not_silently_enabled():
    """``precision`` reaches the Triton dot. If it did not, both would be identical
    and the op would be quietly less accurate than the XLA path it replaces."""
    x = _bessel(1024)
    exact = jax.jit(_mlp(PALLAS_INFERENCE, precision="f32").__call__)(x)
    tf32 = jax.jit(_mlp(PALLAS_INFERENCE, precision="tf32").__call__)(x)
    reference = jax.jit(_mlp("xla").__call__)(x)
    assert float(jnp.abs(tf32 - exact).max()) > 0.0
    assert float(jnp.abs(exact - reference).max()) < float(jnp.abs(tf32 - reference).max())


def test_rejects_widths_the_kernels_cannot_tile():
    with pytest.raises(ValueError, match="multiple of"):
        _mlp(PALLAS_INFERENCE, sizes=(8, 64, 96, 512))


def test_rejects_precisions_it_does_not_implement():
    with pytest.raises(ValueError, match="only implements"):
        _mlp(PALLAS_INFERENCE, precision="bf16")
    # The multi-pass presets are measured and deliberately not offered; a config
    # asking for one must fail rather than silently get plain tf32 or fp32.
    with pytest.raises(ValueError, match="only implements"):
        _mlp(PALLAS_INFERENCE, precision="tf32x3")
    with pytest.raises(ValueError, match="only implements"):
        _mlp(PALLAS_INFERENCE, precision="bf16x3")


def test_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend must be one of"):
        _mlp("pallas")


def test_rejects_unknown_fusion_policy():
    with pytest.raises(ValueError, match="forward_fusion must be one of"):
        _mlp(PALLAS_INFERENCE, forward_fusion="everything")


def test_op_rejects_chain_fusion_for_unmeasured_shape():
    with pytest.raises(ValueError, match="exactly three 64-wide"):
        _mlp(
            PALLAS_INFERENCE,
            sizes=(8, 64, 64, 512),
            forward_fusion="chain",
        )


@needs_pallas
def test_backend_is_static_and_does_not_perturb_the_weights():
    """Swapping the backend is a deployment choice, not a model change: the two ops
    hold the same parameters, so a checkpoint moves between them untouched."""
    a, b = _mlp("xla"), _mlp(PALLAS_INFERENCE)
    for wa, wb in zip(a.linears, b.linears, strict=True):
        assert bool((wa == wb).all())
    assert eqx.partition(a, eqx.is_array)[1].backend == "xla"


@needs_pallas
@pytest.mark.parametrize(
    "sizes",
    [
        (8, 128, 128, 128, 256),
        (16, 128, 128, 128, 768),
        (24, 128, 128, 128, 1280),
        (40, 64, 64, 64, 64),
    ],
)
@pytest.mark.parametrize("num_edges", [64, 8192])
@pytest.mark.parametrize("precision", ["f32", "tf32"])
@pytest.mark.parametrize("fusion", ["chain", "full"])
def test_chain_shapes_and_signed_vjp(sizes, num_edges, precision, fusion):
    """Independent signed cotangents cover cancellation and padded entry widths."""
    reference = _mlp("xla", sizes=sizes, precision=precision)
    fused = _mlp(
        PALLAS_INFERENCE,
        sizes=sizes,
        precision=precision,
        forward_fusion=fusion,
        backward_fusion=fusion,
    )
    x = _bessel(num_edges, width=sizes[0], seed=101)
    dy = _bessel(num_edges, width=sizes[-1], seed=202)

    def run(model):
        @jax.jit
        def call(x, dy):
            y, pullback = jax.vjp(model, x)
            return y, pullback(dy)[0]

        return call(x, dy)

    want, got = run(reference), run(fused)
    tolerance = 1e-5 if precision == "f32" else 5e-3
    for expected, actual in zip(want, got, strict=True):
        assert bool(jnp.isfinite(actual).all())
        error = jnp.max(jnp.abs(actual - expected)) / jnp.maximum(jnp.max(jnp.abs(expected)), 1e-12)
        assert float(error) < tolerance


@needs_pallas
@pytest.mark.parametrize("allow_tf32", [False, True])
@pytest.mark.parametrize("fusion", ["chain", "full"])
@pytest.mark.parametrize("hidden,num_edges", [(128, 128), (64, 100032)])
def test_chain_last_activation_and_zero_weight_cotangents(allow_tf32, fusion, hidden, num_edges):
    """Direct entry supports last-layer SiLU and preserves inference-only semantics."""
    from splice.ops._pallas_mlp import fused_dense_mlp

    sizes = (8, hidden, hidden, hidden, 256)
    weights = tuple(_mlp("xla", sizes=sizes).linears)
    scales = tuple(n**-0.5 for n in sizes[:-1])
    dot = jax.lax.Precision.HIGH if allow_tf32 else jax.lax.Precision.HIGHEST
    x = _bessel(num_edges)
    dy = _bessel(num_edges, width=256, seed=303)

    def call(x, weights, forward, backward):
        return fused_dense_mlp(
            x,
            list(weights),
            list(scales),
            1.5,
            act_last=True,
            allow_tf32=allow_tf32,
            xla_dot=dot,
            forward_fusion=forward,
            backward_fusion=backward,
        )

    def evaluate(forward, backward):
        @jax.jit
        def run(x, weights, dy):
            y, pull = jax.vjp(lambda z, w: call(z, w, forward, backward), x, weights)
            return y, pull(dy)

        return run(x, weights, dy)

    want_y, (want_dx, _) = evaluate("xla", "xla")
    got_y, (got_dx, got_dw) = evaluate(fusion, fusion)
    for expected, actual in [(want_y, got_y), (want_dx, got_dx)]:
        error = jnp.max(jnp.abs(actual - expected)) / jnp.maximum(jnp.max(jnp.abs(expected)), 1e-12)
        assert float(error) < (5e-3 if allow_tf32 else 1e-5)
    assert all(bool((w == 0).all()) for w in got_dw)


def test_auto_policy_uses_rows_and_preserves_unmeasured_devices():
    from splice.ops._pallas_mlp import resolve_fusions

    for width, threshold in [(64, 64), (128, 64)]:
        weights = [
            np.empty((16, width)),
            np.empty((width, width)),
            np.empty((width, width)),
            np.empty((width, 512)),
        ]
        assert resolve_fusions(
            weights, threshold, "auto", "auto", device_kind="NVIDIA H200", original_input_width=8
        ) == (
            "full",
            "full",
        )
        assert resolve_fusions(
            weights,
            threshold + 64,
            "auto",
            "auto",
            device_kind="NVIDIA H200",
            original_input_width=8,
        ) == ("chain", "chain")
        assert resolve_fusions(
            weights,
            threshold,
            "auto",
            "auto",
            device_kind="NVIDIA H200",
            allow_tf32=True,
            original_input_width=8,
        ) == ("chain", "chain")
        for input_width in (16, 40):
            assert resolve_fusions(
                weights,
                64,
                "auto",
                "auto",
                device_kind="NVIDIA H200",
                original_input_width=input_width,
            ) == ("chain", "chain")
        narrow_output = [*weights[:-1], np.empty((width, 256))]
        assert resolve_fusions(
            narrow_output,
            64,
            "auto",
            "auto",
            device_kind="NVIDIA H200",
            original_input_width=8,
        ) == ("chain", "chain")
        expected = ("chain", "chain") if width == 64 else ("xla", "per_layer")
        assert (
            resolve_fusions(weights, threshold, "auto", "auto", device_kind="NVIDIA B200")
            == expected
        )


def test_full_tiles_cover_valid_rows_and_non_power_of_two_outputs():
    from splice.ops._pallas_mlp_full import _full_config

    for rows in (64, 192, 1024):
        for hidden in (64, 128):
            for output in (64, 256, 512, 768, 1280):
                for tf32 in (False, True):
                    bm, bn, _, _ = _full_config(rows, hidden, output, tf32)
                    assert rows % bm == output % bn == 0


@needs_pallas
@pytest.mark.parametrize(
    "forward,backward", [("chain", "chain"), ("xla", "chain"), ("chain", "xla")]
)
def test_large_edge_tf32_weight_layout_preserves_signed_vjp(forward, backward):
    """Exercise the large-edge tensor-dot layout and mixed residual producers."""
    sizes = (8, 64, 64, 64, 256)
    reference = _mlp("xla", sizes=sizes, precision="tf32")
    fused = _mlp(
        PALLAS_INFERENCE,
        sizes=sizes,
        precision="tf32",
        forward_fusion=forward,
        backward_fusion=backward,
    )
    x = _bessel(100032, seed=401)
    dy = _bessel(100032, width=256, seed=402)

    def run(model):
        @jax.jit
        def call(x, dy):
            y, pullback = jax.vjp(model, x)
            return y, pullback(dy)[0]

        return call(x, dy)

    for expected, actual in zip(run(reference), run(fused), strict=True):
        assert bool(jnp.isfinite(actual).all())
        error = jnp.max(jnp.abs(actual - expected)) / jnp.maximum(jnp.max(jnp.abs(expected)), 1e-12)
        assert float(error) < 5e-3


@pytest.mark.parametrize("allow_tf32", [False, True])
def test_auto_policy_measured_ada_large_edges(allow_tf32):
    from splice.ops._pallas_mlp import resolve_fusions

    def policy(
        rows=100032,
        output=512,
        input_width=8,
        device="NVIDIA RTX 6000 Ada Generation",
        forward="auto",
        backward="auto",
    ):
        weights = [
            np.empty((16, 128)),
            np.empty((128, 128)),
            np.empty((128, 128)),
            np.empty((128, output)),
        ]
        return resolve_fusions(
            weights,
            rows,
            forward,
            backward,
            device_kind=device,
            original_input_width=input_width,
            allow_tf32=allow_tf32,
        )

    for rows in (100032, 250048, 500032, 1000000, 2000000):
        for output in (256, 512, 768, 1280):
            assert policy(rows=rows, output=output) == ("chain", "chain")
    for rows in (128, 8192, 65536, 99968):
        assert policy(rows=rows) == ("xla", "per_layer")
    for width in (16, 40):
        assert policy(input_width=width) == ("xla", "per_layer")
    for output in (128, 1536):
        assert policy(output=output) == ("xla", "per_layer")
    for device in ("NVIDIA B200", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"):
        assert policy(device=device) == ("xla", "per_layer")
    assert policy(forward="xla") == ("xla", "chain")
    assert policy(backward="per_layer") == ("chain", "per_layer")
