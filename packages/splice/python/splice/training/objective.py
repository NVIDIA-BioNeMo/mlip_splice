# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimization schedules, loss reductions, train steps, and evaluation."""

import os
import time
from functools import partial

# Best-effort: keep XLA from preallocating huge buffers, helps small-model multi-run.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from splice.channel_sort import radial_support_penalty
from splice.data import batched_iter
from splice.forward import model_call
from splice.lr_annotations import annotation_lr_mults, lr_mult_transform
from splice.padding import real_atoms_mask
from splice.parallel import (
    DATA_AXIS,
    shard_map,
)

# unchanged (target_virial = stress*V); we divide the per-graph virial diff by
# |det(cell)| so the loss/eval compare stress = (1/V) dE/dε (MACE paper Eq 26).
_STRESS_LOSS = bool(os.environ.get("SPLICE_STRESS_LOSS"))


def _build_schedule(
    base_lr: float,
    schedule: str,
    *,
    total_steps: int,
    warmup_frac: float = 0.05,
    quench_frac: float = 0.2,
    quench_alpha: float = 0.0,
    hold_steps: int = 100_000,
    quench_steps: int = 200_000,
):
    """Return a float or a callable(step)->float for the requested LR schedule.

    Schedules supported:
      "constant"             : returns `base_lr` (float)
      "cosine"               : cosine decay from base_lr → 0 over total_steps
      "warmup_cosine"        : linear warmup from 0 → base_lr over warmup_frac
                               of total_steps, then cosine decay → 0 over the rest
      "constant_then_quench" : constant `base_lr` for the first (1 - quench_frac),
                               then cosine decay base_lr → 0 over the final
                               quench_frac of steps. Lets the model train at full
                               LR and only freezes once near the end.

    `total_steps` is the global optimizer step count (epochs × steps_per_epoch).
    Schedules are deterministic functions of the step index, so width-transfer
    of µP HPs is preserved when both widths use the same schedule.
    """
    import optax

    if schedule == "constant":
        return float(base_lr)
    if schedule == "cosine":
        return optax.cosine_decay_schedule(init_value=base_lr, decay_steps=total_steps)
    if schedule == "warmup_cosine":
        warmup_steps = max(1, int(warmup_frac * total_steps))
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=0.0,
        )
    if schedule == "constant_then_quench":
        quench_steps = max(1, int(quench_frac * total_steps))
        constant_steps = total_steps - quench_steps
        return optax.join_schedules(
            schedules=[
                optax.constant_schedule(base_lr),
                optax.cosine_decay_schedule(
                    init_value=base_lr, decay_steps=quench_steps, alpha=quench_alpha
                ),
            ],
            boundaries=[constant_steps],
        )
    if schedule == "warmup_constant":
        warmup_steps = max(1, int(warmup_frac * total_steps))
        return optax.join_schedules(
            schedules=[
                optax.linear_schedule(0.0, base_lr, transition_steps=warmup_steps),
                optax.constant_schedule(base_lr),
            ],
            boundaries=[warmup_steps],
        )
    if schedule == "warmup_constant_quench":
        # gentle warmup -> high constant LR -> cosine quench at the very end. The
        # multi-head finetune shape: a long warmup protects the already-correct
        # (grafted) heads from the untrained heads' large initial gradients, then
        # full LR, then freeze near the end. Uses warmup_frac + quench_frac of total.
        warmup_steps = max(1, int(warmup_frac * total_steps))
        quench_steps = max(1, int(quench_frac * total_steps))
        constant_steps = max(1, total_steps - warmup_steps - quench_steps)
        return optax.join_schedules(
            schedules=[
                optax.linear_schedule(0.0, base_lr, transition_steps=warmup_steps),
                optax.constant_schedule(base_lr),
                optax.cosine_decay_schedule(
                    init_value=base_lr, decay_steps=quench_steps, alpha=quench_alpha
                ),
            ],
            boundaries=[warmup_steps, warmup_steps + constant_steps],
        )
    if schedule == "hold_quench_hold":
        # Absolute-step 3-phase "high-LR then settle low and HOLD" shape:
        #   [0, warmup)                        : linear 0 -> base_lr (skipped if warmup_frac==0)
        #   [warmup, hold_steps)               : constant base_lr
        #   [hold_steps, hold_steps+quench)    : cosine base_lr -> base_lr*quench_alpha
        #   [hold_steps+quench, end)           : constant floor base_lr*quench_alpha
        # Unlike constant_then_quench (fraction-based, quench-to-zero at the very end),
        # this quenches MID-RUN to a NONZERO floor and HOLDS it. Boundaries are ABSOLUTE
        # optimizer steps (hold_steps + quench_steps length), independent of total_steps.
        warmup_steps = max(0, int(warmup_frac * total_steps))
        floor = base_lr * quench_alpha
        scheds = []
        bounds = []
        if warmup_steps > 0:
            scheds.append(optax.linear_schedule(0.0, base_lr, transition_steps=warmup_steps))
            bounds.append(warmup_steps)
        scheds.append(optax.constant_schedule(base_lr))
        bounds.append(hold_steps)
        scheds.append(
            optax.cosine_decay_schedule(
                init_value=base_lr, decay_steps=quench_steps, alpha=quench_alpha
            )
        )
        bounds.append(hold_steps + quench_steps)
        scheds.append(optax.constant_schedule(floor))
        return optax.join_schedules(schedules=scheds, boundaries=bounds)
    raise ValueError(f"unknown LR schedule {schedule!r}")


def _build_optimizer(
    chain,
    base_lr_or_schedule,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
    optimizer: str = "adam",
    muon_lr_scale: float = 1.0,
):
    # Per-op LR multipliers travel with the config as draupnir `annotations: {lr_mult}`
    # (the clean replacement for the old op-index LR_OP_MULT env hack). Absent → 1.0.
    lr_mults = annotation_lr_mults(chain)
    handle = [(i, chain.specs[i].op, m) for i, m in enumerate(lr_mults) if m != 1.0]
    if handle:
        print(
            f"[annotations] per-op lr_mult on {len(handle)} ops: "
            + ", ".join(f"{op}#{i}x{m:g}" for i, op, m in handle)
        )
    if optimizer == "muon":
        # Muon (Newton-Schulz orthogonalized momentum) on the hidden equivariant
        # linear blocks; Adam(W) on everything else. µP does not apply to the Muon
        # matrices (Muon carries its own width scaling); the annotation handle still
        # rides on top, hitting the Adam-governed layers (SC / readouts / radial).
        import optax

        from splice.optim import make_muon_optimizer, muon_leaf_counts

        nm, na = muon_leaf_counts(chain)
        print(
            f"[muon] {nm} muon leaves (hidden IrrepsLinear.w) / {na} adam leaves; "
            f"lr_scale={muon_lr_scale}"
        )
        opt = make_muon_optimizer(
            chain,
            base_lr_or_schedule,
            muon_lr_scale=muon_lr_scale,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        return optax.chain(opt, lr_mult_transform(chain, lr_mults)) if handle else opt
    import optax

    # no-µP path: plain optimizer, then a per-op lr_mult scaling transform on top.
    if weight_decay > 0.0:
        base = optax.adamw(base_lr_or_schedule, weight_decay=weight_decay)
    elif amsgrad:
        base = optax.amsgrad(base_lr_or_schedule)
    else:
        base = optax.adam(base_lr_or_schedule)
    if handle:
        return optax.chain(base, lr_mult_transform(chain, lr_mults))
    return base


# Keys in the batch dict that are loss-side labels / metadata — NOT chain inputs.
# Strips them before passing the batch into ``model_call``.
_LOSS_NON_BATCH_KEYS = frozenset(
    {"dummy_graph_id", "target_E", "target_F", "target_virial", "virial_weight"}
)
_STAT_NAMES = (
    "loss",
    "E_mse",
    "F_mse",
    "V_mse",
    "E_mae",
    "F_mae",
    "V_mae",
    "cutoff_reg",
)
_STAT = {name: i for i, name in enumerate(_STAT_NAMES)}


def _huber_elementwise(diff, delta):
    """Elementwise Huber: 0.5·diff² where |diff|≤δ, else δ(|diff|−0.5δ).

    ``delta`` may broadcast against ``diff`` (e.g. per-atom δ for forces).
    Matches ``torch.nn.functional.huber_loss(reduction='none')``.
    """
    a = jnp.abs(diff)
    return jnp.where(a <= delta, 0.5 * diff * diff, delta * (a - 0.5 * delta))


def _conditional_huber_forces_sum(F_pred, F_ref, delta, atom_mask):
    """Sum of MACE's conditional-Huber force loss (paper arxiv 2510.25380 Eq.26).

    The Huber δ is scaled per atom by the *reference* force magnitude — stepwise
    1.0 / 0.7 / 0.4 / 0.1 for ``|F_ref|`` in <100 / <200 / <300 / ≥300 eV/Å — so
    high-force atoms (which a from-scratch model gets wildly wrong) cannot
    dominate the gradient. Padding atoms are masked out. Returns a plain sum;
    the per-element count is ``n_real_atoms*3`` (same as the MSE branch), so the
    cross-device mean in :func:`_assemble_loss` stays exact.
    """
    norm = jnp.linalg.norm(F_ref, axis=-1)  # [N]
    factor = jnp.where(
        norm < 100.0, 1.0, jnp.where(norm < 200.0, 0.7, jnp.where(norm < 300.0, 0.4, 0.1))
    )
    d = (delta * factor)[:, None]  # [N,1] -> broadcasts over the 3 components
    he = _huber_elementwise(F_pred - F_ref, d) * atom_mask[:, None]
    return jnp.sum(he)


def _assemble_loss(
    components: dict,
    weights: dict,
    axis_name: str | None = None,
    *,
    local_mean: bool = False,
):
    """Reduce per-term ``(sse, sae, n)`` into ``(loss, mse_dict, mae_dict)``.

    The reduction is done **after** aggregating counts, so it is exact under
    any split of real content across devices (the key property for data
    parallelism with padded, possibly-uneven sub-batches):

    - With ``axis_name`` set (inside ``shard_map``) and ``local_mean=False``,
      counts are summed across that mesh axis via ``psum``, so each device
      contributes ``sse_local / count_global`` — its share of the global mean.
      ``psum``'ing the returned loss/metrics **and** the gradients then gives
      the exact global quantities (``Σ_d sse_d / Σ_d count_d``), regardless of
      how many real atoms/graphs each device's sub-batch held.
    - With ``local_mean=True``, counts stay local even inside ``shard_map``
      (no count AllReduce). Each rank optimizes ``sse_local / n_local`` and
      the grad all-reduce averages those locally-normalized gradients — standard
      DDP, slightly biased when ranks hold unequal real content.
    - With ``axis_name=None`` (single device) counts are local → the plain
      mean, numerically identical to a one-device run.
    """

    loss = jnp.array(0.0)
    mse: dict = {}
    mae: dict = {}
    items = tuple(components.items())
    counts = jnp.stack([jnp.asarray(values[3]) for _, values in items])
    if axis_name is not None and not local_mean:
        # One compact normalization collective. Issuing a psum from inside the
        # Python loop creates one latency-bound collective per loss term.
        counts = jax.lax.psum(counts, axis_name)
    # Each component is ``(loss_sum, sse, sae, n)``: ``loss_sum`` is the term's
    # contribution to the optimized loss (== ``sse`` for MSE, or a Huber sum),
    # while ``mse``/``mae`` are always derived from ``sse``/``sae`` so the logged
    # E_mse/F_mse/E_mae/F_mae columns stay comparable across loss types.
    for i, (term, (loss_sum, sse, sae, _)) in enumerate(items):
        ng = jnp.maximum(counts[i], 1.0)
        mse[term] = sse / ng
        mae[term] = sae / ng
        loss = loss + weights.get(term, 0.0) * (loss_sum / ng)
    return loss, mse, mae


def _train_step_factory(
    static,
    num_species: int,
    tx,
    *,
    energy_weight: float,
    force_weight: float,
    virial_weight: float = 0.0,
    ema_decay: float = 0.0,
    loss_type: str = "mse",
    huber_delta: float = 0.01,
    huber_delta_energy: float | None = None,
    cutoff_weight: float = 0.0,
    mesh=None,
    local_loss_mean: bool = False,
):
    """Build a single-JIT train step.

    EMA is folded inside the same JIT boundary as forward+backward+optimizer
    because a second compiled function introduces a host-side dispatch tail
    that leaves the GPU idle (2026-05-29 nsys: ~40 ms/iter on A100). One JIT
    keeps the GPU fed.

    Use ``jax.jit``, not ``eqx.filter_jit``. Equinox's wrapper returns a dummy
    ``marker`` and ``block_until_ready``s it on every call, which makes dispatch
    equal wall time and prevents run-ahead. ``donate_argnums`` aliases
    params/opt-state/EMA/stats_acc so those buffers can be reused rather than
    copied. The batch is not donated: the next step's H2D reuses that slot.

    Per-step metrics are stacked into one vector and added onto a donated
    ``stats_acc`` inside the compiled step so the host loop does not issue a
    tree of scalar ``add`` kernels after every dispatch.

    When ``mesh`` is given, the step is data-parallel via ``shard_map`` over
    that mesh's ``"data"`` axis: params/opt-state/EMA replicated, the batch
    sharded along a leading device axis, gradients all-reduced. By default the
    loss is assembled from per-term ``(sse, count)`` so the cross-device
    reduction is exact regardless of how real content splits across devices
    (see :func:`_assemble_loss`). ``local_loss_mean=True`` skips the count
    AllReduce and averages locally-normalized gradients instead. ``mesh=None``
    is the original single-device step.
    """
    use_virial = virial_weight > 0.0
    use_ema = ema_decay > 0.0
    # Energy Huber δ defaults to the shared --huber-delta. Decoupling it (larger
    # δ_E) keeps from-scratch per-atom energy errors in the quadratic regime so
    # the energy term actually gets gradient, while forces keep the tight paper δ.
    delta_E = huber_delta if huber_delta_energy is None else huber_delta_energy
    # Radial-cutoff regularizer: a batch-independent penalty on per-channel radial
    # support (pushes cutoffs shorter → more warp zero-skip; see channel_sort). It is
    # a pure function of `params`, replicated on every data-parallel device, so it must
    # be divided by the device count before the `_reduce` psum re-multiplies it (the
    # data-loss terms are genuine per-device partials; this one is not). Local-mean
    # mode all-reduces gradients with a mean, so the penalty is left undivided.
    use_cutoff_reg = cutoff_weight > 0.0
    n_data = 1.0 if mesh is None else float(mesh.shape[DATA_AXIS])
    reg_div = 1.0 if (mesh is None or local_loss_mean) else n_data
    weights = {"E": energy_weight, "F": force_weight}
    if use_virial:
        weights["V"] = virial_weight

    def _components(params, batch):
        """Per-term ``(loss_sum, sse, sae, count)`` on one (sub-)batch.

        ``loss_sum`` is the term's contribution to the optimized loss: ``sse``
        for ``loss_type='mse'``, or a Huber sum for ``loss_type='huber'``
        (per-atom energy + conditional-Huber forces — the MACE-MH-1 recipe,
        arxiv 2510.25380 Eq.26). ``sse``/``sae`` are always returned for the
        logged MSE/MAE metrics. Reduction (and the cross-device count sum)
        happens in :func:`_assemble_loss`, so the same components reduce
        correctly on one device or sharded across many. ``local_mean`` only
        changes that reduction (skip the count psum).
        """
        model = eqx.combine(params, static)
        b = {k: v for k, v in batch.items() if k not in _LOSS_NON_BATCH_KEYS}
        out = model_call(
            model,
            b,
            num_species=num_species,
            compute_virial=use_virial,
        )
        # The padding contract always places the dummy graph last. Distributed
        # placement omits the redundant scalar metadata from the device batch.
        dummy_graph_id = batch.get("dummy_graph_id", batch["nats"].shape[-1] - 1)
        atom_mask = real_atoms_mask(batch["inde"], dummy_graph_id)
        n_real_atoms = jnp.sum(atom_mask)
        use_huber = loss_type == "huber"

        comps: dict = {}
        # E (per-graph, drop dummy graph at end); count = number of real graphs.
        E_diff = out["E"][:-1] - batch["target_E"][:-1]
        E_sse = jnp.sum(E_diff**2)
        E_sae = jnp.sum(jnp.abs(E_diff))
        if use_huber:
            # Huber on PER-ATOM energy (Ê/N_a): bounds the gradient for the large
            # total-energy residuals a from-scratch model produces on big molecules.
            nat = jax.ops.segment_sum(
                atom_mask.astype(E_diff.dtype), batch["inde"], num_segments=out["E"].shape[0]
            )
            E_loss = jnp.sum(_huber_elementwise(E_diff / jnp.maximum(nat[:-1], 1.0), delta_E))
        else:
            E_loss = E_sse
        comps["E"] = (E_loss, E_sse, E_sae, jnp.asarray(E_diff.shape[0], E_diff.dtype))
        # F (per-atom, mask padding atoms); count = N_real_atoms * 3 components.
        F_diff_masked = (out["F"] - batch["target_F"]) * atom_mask[:, None]
        F_sse = jnp.sum(F_diff_masked**2)
        F_sae = jnp.sum(jnp.abs(F_diff_masked))
        if use_huber:
            F_loss = _conditional_huber_forces_sum(
                out["F"], batch["target_F"], huber_delta, atom_mask
            )
        else:
            F_loss = F_sse
        comps["F"] = (F_loss, F_sse, F_sae, n_real_atoms * 3.0)
        # V (per-graph, drop dummy graph at end); count = N_real_graphs * 9.
        # Multi-head: `virial_weight` (per-graph) gates stress to periodic heads
        # only — molecular batches (no stress label) get weight 0, so the loss
        # and the count both ignore them. Absent ⇒ all-ones (single-head path
        # unchanged: count == V_diff.size as before).
        if use_virial:
            vw = batch.get("virial_weight")
            if vw is None:
                vw = jnp.ones((out["virial"].shape[0],), out["virial"].dtype)
            vwg = vw[:-1, None, None]
            V_diff = (out["virial"][:-1] - batch["target_virial"][:-1]) * vwg
            if _STRESS_LOSS:
                # /per-graph volume |det(cell)| -> compare STRESS, not virial (paper Eq 26).
                _vol = jnp.abs(jnp.linalg.det(batch["cell"][:-1].astype(V_diff.dtype)))
                V_diff = V_diff * jnp.where(_vol > 1e-6, 1.0 / _vol, 0.0)[:, None, None]
            V_sse = jnp.sum(V_diff**2)
            V_sae = jnp.sum(jnp.abs(V_diff))
            V_loss = jnp.sum(_huber_elementwise(V_diff, huber_delta)) if use_huber else V_sse
            V_n = jnp.maximum(jnp.sum(vw[:-1]) * 9.0, 1.0)
            comps["V"] = (V_loss, V_sse, V_sae, V_n)
        return comps

    def _packed_loss(params, batch, axis_name):
        loss, mse, mae = _assemble_loss(
            _components(params, batch),
            weights,
            axis_name,
            local_mean=local_loss_mean,
        )
        zero = jnp.array(0.0)
        # Radial-cutoff penalty (pure fn of params). Divide by device count so the
        # `_reduce` psum re-multiplies it back to one global copy; single-device
        # reg_div=1 (no psum). Logged as `cutoff_reg` (÷reg_div for the same reason).
        cutoff_reg = zero
        if use_cutoff_reg:
            cutoff_reg = radial_support_penalty(eqx.combine(params, static)) / reg_div
            loss = loss + cutoff_weight * cutoff_reg
        stats = jnp.stack(
            tuple(
                {
                    "loss": loss,
                    "E_mse": mse["E"],
                    "F_mse": mse["F"],
                    "V_mse": mse.get("V", zero),
                    "E_mae": mae["E"],
                    "F_mae": mae["F"],
                    "V_mae": mae.get("V", zero),
                    "cutoff_reg": cutoff_reg,
                }[name]
                for name in _STAT_NAMES
            )
        )
        return loss, stats

    def _loss(params, batch):
        return _packed_loss(params, batch, None)

    if mesh is None:
        if use_ema:

            @partial(jax.jit, donate_argnums=(0, 1, 2, 3))
            def train_step(params, opt_state, ema_params, stats_acc, batch):
                (_, stats), grads = eqx.filter_value_and_grad(_loss, has_aux=True)(params, batch)
                updates, opt_state = tx.update(grads, opt_state, params)
                new_params = eqx.apply_updates(params, updates)
                new_ema = jax.tree.map(
                    lambda e, p: ema_decay * e + (1.0 - ema_decay) * p,
                    ema_params,
                    new_params,
                )
                return new_params, opt_state, new_ema, stats_acc + stats, stats

        else:

            @partial(jax.jit, donate_argnums=(0, 1, 2))
            def train_step(params, opt_state, stats_acc, batch):
                (_, stats), grads = eqx.filter_value_and_grad(_loss, has_aux=True)(params, batch)
                updates, opt_state = tx.update(grads, opt_state, params)
                params = eqx.apply_updates(params, updates)
                return params, opt_state, stats_acc + stats, stats

        return train_step

    # ---- data-parallel (shard_map) variant ----
    from jax.sharding import PartitionSpec as P

    repl, shard = P(), P(DATA_AXIS)

    def _sharded_loss(params, sub_batch):
        return _packed_loss(params, sub_batch, DATA_AXIS)

    def _reduce_grads(grads_c):
        """All-reduce gradients over the data axis.

        Default (exact global mean): each rank's loss is ``sse_d / n_global``,
        so the local grad is already a *share* of the global gradient. A
        **sum** ``psum`` reconstructs ``∇(Σ sse / Σ n)``.

        ``local_loss_mean`` (DDP): each rank's loss is ``sse_d / n_local``, so
        the local grad is a complete mini-batch mean. **Average** those with
        ``pmean`` — the same allreduce-mean PyTorch DDP and the JAX pmap
        examples use. Equal local counts ⇒ identical to the exact path;
        unequal counts ⇒ unweighted average of per-rank means (DDP's known
        bias vs size-weighted global mean).
        """
        reduce = jax.lax.pmean if local_loss_mean else jax.lax.psum
        return jax.tree.map(lambda g: reduce(g, DATA_AXIS), grads_c)

    if use_ema:

        def _core(params, opt_state, ema_params, stats_acc, batch):
            sub = jax.tree.map(lambda x: x[0], batch)  # drop the device axis
            local_acc = stats_acc[0]
            ((_, stats_c), grads_c) = eqx.filter_value_and_grad(_sharded_loss, has_aux=True)(
                params, sub
            )
            grads = _reduce_grads(grads_c)
            updates, opt_state = tx.update(grads, opt_state, params)
            new_params = eqx.apply_updates(params, updates)
            new_ema = jax.tree.map(
                lambda e, p: ema_decay * e + (1.0 - ema_decay) * p,
                ema_params,
                new_params,
            )
            # Local normalized contributions are intentionally data-sharded.
            # Exact mode sums them at an epoch/log boundary (each rank holds
            # sse_d / n_global); local-mean mode averages them (each rank
            # holds sse_d / n_local).
            return (
                new_params,
                opt_state,
                new_ema,
                (local_acc + stats_c)[None, :],
                stats_c[None, :],
            )

        sm = shard_map(
            _core,
            mesh=mesh,
            in_specs=(repl, repl, repl, shard, shard),
            out_specs=(repl, repl, repl, shard, shard),
        )
    else:

        def _core(params, opt_state, stats_acc, batch):
            sub = jax.tree.map(lambda x: x[0], batch)
            local_acc = stats_acc[0]
            ((_, stats_c), grads_c) = eqx.filter_value_and_grad(_sharded_loss, has_aux=True)(
                params, sub
            )
            grads = _reduce_grads(grads_c)
            updates, opt_state = tx.update(grads, opt_state, params)
            return (
                eqx.apply_updates(params, updates),
                opt_state,
                (local_acc + stats_c)[None, :],
                stats_c[None, :],
            )

        sm = shard_map(
            _core,
            mesh=mesh,
            in_specs=(repl, repl, shard, shard),
            out_specs=(repl, repl, shard, shard),
        )

    return jax.jit(sm, donate_argnums=(0, 1, 2, 3) if use_ema else (0, 1, 2))


def _make_eval_one():
    """Build a JIT-cached single-batch eval kernel."""

    @eqx.filter_jit
    def _eval_one(chain, batch, num_species: int, with_virial: bool):
        b = {k: v for k, v in batch.items() if k not in _LOSS_NON_BATCH_KEYS}
        out = model_call(
            chain,
            b,
            num_species=num_species,
            compute_virial=with_virial,
        )
        E_diff = out["E"][:-1] - batch["target_E"][:-1]
        atom_mask = real_atoms_mask(batch["inde"], batch["dummy_graph_id"])
        F_diff = (out["F"] - batch["target_F"]) * atom_mask[:, None]
        n_real_graphs = jnp.array(E_diff.shape[0] - sum(jnp.isclose(batch["nats"][:-1], 0)))
        if with_virial:
            V_diff = out["virial"][:-1] - batch["target_virial"][:-1]
            if _STRESS_LOSS:  # report STRESS-units virial MAE (coherent with the loss)
                _vol = jnp.abs(jnp.linalg.det(batch["cell"][:-1].astype(V_diff.dtype)))
                V_diff = V_diff * jnp.where(_vol > 1e-6, 1.0 / _vol, 0.0)[:, None, None]
            sum_V_abs = jnp.sum(jnp.abs(V_diff))
        else:
            sum_V_abs = jnp.array(0.0, dtype=E_diff.dtype)
        return (
            jnp.sum(jnp.abs(E_diff)),
            jnp.sum(jnp.abs(F_diff)),
            jnp.sum(atom_mask),
            n_real_graphs,
            sum_V_abs,
        )

    return _eval_one


def _eval_subset(frames, frac):
    """Subsample an eval set for PERIODIC (monitoring) evals.

    Full-set validation is expensive enough to dominate a run: on SPICE2 the
    87,764-frame val set costs ~87s against a ~105s epoch, so evaluating it
    every epoch spent more wall time validating than training. A strided
    subsample is an equally good monitoring signal at a fraction of the cost;
    the final epoch still runs the full set, so the reported number is exact.

    Strided, NOT first-N: packed val sets can be ordered by source (OMat24 is),
    so ``frames[:int(frac*N)]`` samples ONE source — biased and optimistic.
    Striding spans all sources at the same cost.
    """
    if frac >= 1.0:
        return frames
    return frames[:: max(1, round(1.0 / frac))]


def _eval_loop(
    chain,
    frames,
    *,
    num_species: int,
    batch_size: int,
    offsets,
    base_atoms: int,
    base_edges: int,
    with_virial: bool = False,
    eval_one=None,
    rank: int = 0,
    world_size: int = 1,
    head_id: int | None = None,
):
    """Compute E/F (and optionally virial) MAE across the given frames.

    ``head_id`` (multi-head finetune): tag every batch's graphs with this head so
    the chain's ``multihead_energy`` selects the right readout column. Each eval
    set is one dataset/head, so per-set metrics ARE the per-head MAE.

    Also times the eval pass to give an inference throughput estimate.
    Eval here = forward + autograd-for-F (+ optional cell-grad for virial),
    matching the work an MLIP MD step does (per-step the SR cutoff +
    optional stress). The first eval call after a JIT cache miss pays the
    compile cost; later calls amortise it. `eval_step_ms_mean` averages
    over all batches in this call, so the *first* eval per run reads
    high — subsequent evals on the same shapes give a clean number.

    **Distributed (``world_size>1``):** each rank evaluates a disjoint slice
    (``frames[rank::world_size]``) on its own GPU, then the unnormalized
    per-rank ``(Σ|err|, count)`` accumulators are summed across ranks via a
    ``process_allgather`` collective and normalized — the same exact
    post-aggregation reduction used for training. All ranks must call this
    together; every rank returns identical global metrics. ``chain`` here is a
    device-LOCAL copy (so each rank's per-slice eval is purely local).
    """
    if eval_one is None:
        # Caller didn't pre-build one. Doesn't share the JIT cache with the
        # main training-side ``eval_one``; just a fallback for ad-hoc use.
        eval_one = _make_eval_one()
    my_frames = frames[rank::world_size] if world_size > 1 else frames
    sum_E_abs, sum_F_abs, sum_V_abs = 0.0, 0.0, 0.0
    n_atoms_total, n_graphs_total, n_batches = 0, 0, 0
    eval_start = time.time()
    for batch in batched_iter(
        my_frames,
        batch_size=batch_size,
        base_atoms=base_atoms,
        base_edges=base_edges,
        energy_offsets=offsets,
        shuffle=False,
        drop_last=False,
        # Pin the graph axis to a single shape (batch_size + 1 dummy) so the
        # remainder batch doesn't trigger a mid-run XLA recompile that stalls a
        # rank past the JAX coordination heartbeat and aborts the job.
        fixed_graphs=batch_size + 1,
    ):
        if head_id is not None:
            batch["head_id"] = np.full((batch["nats"].shape[0],), head_id, dtype=np.int32)
        eA, fA, n_at, n_g, vA = eval_one(chain, batch, num_species, with_virial)
        sum_E_abs += float(eA)  # block-syncs the JAX graph each batch
        sum_F_abs += float(fA)
        n_atoms_total += int(n_at)
        n_graphs_total += int(n_g)
        sum_V_abs += float(vA)
        n_batches += 1
    eval_total_s = time.time() - eval_start

    if world_size > 1:
        # Sum the unnormalized accumulators across ranks (exact global metrics,
        # independent of how frames split). eval_total_s -> max (wall is bounded
        # by the slowest rank). One collective for the whole eval pass.
        vec = [sum_E_abs, sum_F_abs, float(n_atoms_total), float(n_graphs_total), sum_V_abs]
        vec.append(eval_total_s)
        g = np.asarray(multihost_utils.process_allgather(jnp.asarray(vec)))
        s = g.sum(axis=0)
        sum_E_abs, sum_F_abs = float(s[0]), float(s[1])
        n_atoms_total, n_graphs_total = round(float(s[2])), round(float(s[3]))
        sum_V_abs = float(s[4])
        eval_total_s = float(g[:, -1].max())

    result = {
        "E_mae_per_graph": sum_E_abs / max(n_graphs_total, 1),
        "E_mae_per_atom": sum_E_abs / max(n_atoms_total, 1),
        "F_mae": sum_F_abs / max(n_atoms_total * 3, 1),
        "n_graphs": n_graphs_total,
        "n_atoms": n_atoms_total,
        "eval_step_ms_mean": (eval_total_s / max(n_batches, 1)) * 1000,
        "eval_atoms_per_sec": n_atoms_total / max(eval_total_s, 1e-9),
    }
    if with_virial:
        # Per-graph virial MAE (units eV); divide by 9 to average over tensor
        # components. Convert to per-volume stress MAE in the loss-side log if
        # needed — that requires per-graph volumes that the eval doesn't
        # currently track.
        result["virial_mae_per_graph"] = sum_V_abs / max(n_graphs_total * 9, 1)
    return result
