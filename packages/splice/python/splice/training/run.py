# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare data, model, optimizer, and distributed training state."""

import json
import os
from pathlib import Path

# Best-effort: keep XLA from preallocating huge buffers, helps small-model multi-run.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from splice.config import build_chain
from splice.data import batched_iter, load_extxyz
from splice.model_card import card_offsets, load_card, resolve_arch
from splice.multihead_data import PackedMultiHeadDataset, multihead_batched_iter
from splice.packed_data import PackedDataset, PackedPrefetch
from splice.parallel import (
    build_mesh,
    commit_tree,
    init_distributed,
    is_primary,
    replicate_tree,
)
from splice.training.iteration import train
from splice.training.objective import (
    _build_optimizer,
    _build_schedule,
    _make_eval_one,
    _train_step_factory,
)
from splice.training.options import parse_options
from splice.training.report import run_provenance


def main():
    p, args, card_path = parse_options()
    # ---------------- distributed init (one process per GPU) ----------------
    # Must run before any heavy JAX work. No-op (single process) when the
    # launcher didn't request >1 process, so the single-GPU path is unchanged.
    dist = init_distributed()
    distributed = dist["distributed"]
    n_proc = dist["num_processes"]
    mesh = build_mesh()
    primary = is_primary()

    def log0(*a, **k):
        """Print on process 0 only (avoids N-way console spam)."""
        if primary:
            print(*a, **k)

    # Grid bucketing in DP: when distributed and NO single fixed bucket is given,
    # pad each step to the smallest --base-atoms/--base-edges grid multiple that
    # fits the GLOBAL batch (max atoms/edges across ranks, agreed via a tiny
    # per-step allgather). Reuses the same bucketing as the single-process path;
    # the only extra is the cross-rank shape agreement.
    balanced = args.atom_budget is not None
    if balanced:
        if args.data_format != "packed":
            p.error("--atom-budget (balanced binning) requires --data-format packed")
        if args.edge_budget is None:
            p.error("--atom-budget needs --edge-budget (the fixed padded edge count)")
    # Balanced binning carries its own single fixed shape -> no grid allgather.
    use_grid = (
        distributed and not balanced and args.fixed_atoms is None and args.fixed_edges is None
    )
    if distributed:
        if args.data_format != "packed":
            p.error("multi-process training requires --data-format packed")
        if (args.fixed_atoms is None) != (args.fixed_edges is None):
            p.error("--fixed-atoms and --fixed-edges must be set together")
        if balanced:
            log0(
                f"DP batching: BALANCED (atom_budget={args.atom_budget} "
                f"edge_budget={args.edge_budget} graph_budget={args.graph_budget})"
            )
        elif use_grid:
            log0(
                f"DP batching: GRID (base {args.base_atoms}/{args.base_edges} + per-step allgather)"
            )
        else:
            log0(f"DP batching: FIXED bucket ({args.fixed_atoms} atoms / {args.fixed_edges} edges)")
        gb = (
            f"~{args.atom_budget}atoms x {n_proc}"
            if balanced
            else (f"{args.batch_size} x {n_proc} = {args.batch_size * n_proc}")
        )
        log0(f"distributed: {n_proc} processes (1 GPU each), global_batch = {gb}")
        if args.local_loss_mean:
            log0(
                "DP loss: local mean / DDP (no count AllReduce; "
                "grads allreduce-mean'd across ranks)"
            )
    elif args.local_loss_mean:
        print("WARNING: --local-loss-mean has no effect on a single process")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the inputs so the results dir is self-describing (process 0 only).
    # arch.yaml + card.yaml are saved verbatim; CLI args land in run_config.json;
    # run_meta.json records WHICH CODE produced the run (commit, host, argv).
    if primary:
        (out_dir / "arch.yaml").write_text(Path(args.arch).read_text())
        (out_dir / "card.yaml").write_text(Path(args.card).read_text())
        if card_path is not None:
            (out_dir / "training_card.yaml").write_text(Path(card_path).read_text())
        (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))
        (out_dir / "run_meta.json").write_text(json.dumps(run_provenance(), indent=2))

    # ---------------- load card + arch ----------------
    card = load_card(args.card)
    if args.cutoff is not None:
        # Same dataset, different Rc: override the card cutoff and pick the
        # matching measured avg_num_neighbors so normalisation stays correct.
        # r_max + epsilon then auto-follow inside resolve_arch.
        from splice.model_card import avg_neighbors_for_cutoff

        card["cutoff"] = float(args.cutoff)
        ann = avg_neighbors_for_cutoff(card, args.cutoff)
        if ann is not None:
            card["avg_num_neighbors"] = ann
            print(f"--cutoff {args.cutoff}: measured avg_num_neighbors={ann} (from card table)")
        elif args.allow_unmeasured_cutoff:
            print(
                f"WARNING: --cutoff {args.cutoff} not in card neighbors_by_cutoff "
                f"table; keeping avg_num_neighbors={card['avg_num_neighbors']} "
                f"(epsilon may be miscalibrated)."
            )
        else:
            avail = sorted((card.get("neighbors_by_cutoff") or {}).keys())
            raise SystemExit(
                f"--cutoff {args.cutoff} has no measured avg_num_neighbors in the "
                f"card's neighbors_by_cutoff table (measured: {avail}). Add a "
                f"measured entry, or pass --allow-unmeasured-cutoff to proceed "
                f"with the base value."
            )
    species_map = card["species"]
    cutoff = float(card["cutoff"])
    num_species = len(species_map)

    # Multi-head Stage-2 finetune: parse the spec -> per-head packed datasets
    # (with their E0 rows + sampling weights) + per-head eval metadata.
    mh = None
    if args.multihead_spec:
        import yaml as _yaml

        if args.atom_budget is None:
            p.error("--multihead-spec requires balanced binning (--atom-budget / --edge-budget)")
        with open(os.path.expanduser(args.multihead_spec)) as _fh:
            _spec = _yaml.safe_load(_fh)
        mh_index = os.path.expanduser(_spec["index_dir"])
        if not os.path.exists(os.path.join(mh_index, "meta.json")):
            p.error(f"multihead index missing at {mh_index} (run build_multihead_index.py)")
        _heads = list(_spec["heads"])
        # per-head E0 rows the index used (training), reused for eval-target shifting
        _e0_table = np.load(os.path.join(mh_index, "e0_table.npy"))  # [num_heads, num_species]
        mh_eval = {}
        for d in _spec.get("datasets", []):
            if not d.get("val"):
                continue
            hid = _heads.index(d["head"])
            name = d.get("name", d["head"])
            mh_eval[name] = {
                "path": os.path.expanduser(d["val"]),
                "head_id": hid,
                "offsets": _e0_table[hid],
                # virial: train stress? defaults to periodic (OC20 = periodic but False)
                "virial": bool(d.get("virial", d.get("periodic", False))),
            }
        mh = {"heads": _heads, "index_dir": mh_index, "eval": mh_eval}
        print(
            f"MULTI-HEAD (heterogeneous): index {mh_index}; "
            f"{len(_heads)} heads {_heads}; {len(mh_eval)} eval sets"
        )

    cfg = resolve_arch(args.arch, card)
    chain = build_chain(cfg, key=jax.random.key(args.seed))
    # Note: species_offsets stay at zero in the chain. The canonical training
    # pattern subtracts per-species offsets from targets (via batched_iter's
    # energy_offsets) so the loss runs in the small interaction-energy range
    # (precision-friendly at fp32). Seed `species_offsets` only at inference.

    # Resume from a checkpoint if asked. Loads only the chain weights — the
    # optimizer state starts fresh (Adam moments rebuild after a few steps).
    start_epoch = 0
    start_global_step = 0
    resume_stem = None  # stem of the resume ckpt, for locating its _opt/_ema siblings
    resume_is_step = False
    if args.resume_from is not None:
        ckpt = Path(args.resume_from)
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        chain = eqx.tree_deserialise_leaves(ckpt, chain)
        resume_stem = ckpt.stem
        if ckpt.stem.startswith("ckpt_step_"):
            # Step ckpt (ckpt_step_<global_step>.eqx): restore the global step;
            # the epoch index is cosmetic (the shuffle seed is global_step-based,
            # so resume always draws a fresh order — no data re-seen).
            resume_is_step = True
            try:
                start_global_step = int(ckpt.stem.rsplit("_", 1)[-1])
            except ValueError:
                start_global_step = 0
            print(f"resumed chain weights from {ckpt} (step ckpt, global_step={start_global_step})")
        else:
            # Epoch ckpt (ckpt_epoch_NNNN.eqx): parse the epoch index.
            try:
                start_epoch = int(ckpt.stem.split("_")[-1])
            except ValueError:
                start_epoch = 0
            print(f"resumed chain weights from {ckpt} (start_epoch={start_epoch})")

    n_params = sum(int(x.size) for x in jax.tree.leaves(eqx.filter(chain, eqx.is_array)))
    print(f"arch={Path(args.arch).name} card={Path(args.card).name}")
    print(f"num_species={num_species} cutoff={cutoff} #params={n_params:,}")

    # ---------------- load data ----------------
    print(f"loading data (format={args.data_format})...")
    train_frames: list[dict] = []
    train_spec: dict | None = None
    eval_sets: dict[str, list[dict]] = {}
    if mh is not None:
        # Train data = the unified multi-head index (heterogeneous mixed batches,
        # streamed lazily). Materialise each head's val set for per-head eval.
        mh_ds_main = PackedMultiHeadDataset(mh["index_dir"])
        n_train = mh_ds_main.n_struct
        print(f"  {n_train} train configs across {len(mh['heads'])} heads (heterogeneous mix)")
        for name, em in mh["eval"].items():
            eds = PackedDataset(em["path"], species_map, cutoff, neighbour=args.neighbour)
            eval_sets[name] = [eds.make_frame(i) for i in range(eds.n_struct)]
            print(f"  {len(eval_sets[name])} eval frames in {name} (head {em['head_id']})")
    elif args.data_format == "packed":
        # Streaming: train stays on disk (memmap), built lazily per batch.
        if len(args.train) != 1:
            p.error("--data-format packed expects exactly one packed dir in --train")
        train_spec = dict(
            path=args.train[0], species_map=species_map, cutoff=cutoff, neighbour=args.neighbour
        )
        _n = PackedDataset(**train_spec).n_struct
        n_train = _n if args.max_frames is None else min(_n, args.max_frames)
        print(f"  {n_train} train structs (packed: {args.train[0]}, neighbour={args.neighbour})")
        # Eval sets are small -> materialise as frame lists and reuse the
        # existing in-memory eval path (`_eval_loop`) unchanged.
        for path in args.eval:
            eds = PackedDataset(path, species_map, cutoff, neighbour=args.neighbour)
            eval_sets[Path(path).stem] = [eds.make_frame(i) for i in range(eds.n_struct)]
            print(f"  {len(eval_sets[Path(path).stem])} eval frames in {Path(path).stem}")
    else:
        for path in args.train:
            train_frames += load_extxyz(path, species_map, cutoff=cutoff)
        if args.max_frames is not None:
            train_frames = train_frames[: args.max_frames]
        n_train = len(train_frames)
        print(f"  {n_train} train frames")
        for path in args.eval:
            eval_sets[Path(path).stem] = load_extxyz(path, species_map, cutoff=cutoff)
            print(f"  {len(eval_sets[Path(path).stem])} eval frames in {Path(path).stem}")

    offsets = card_offsets(card)

    packed_loader: PackedPrefetch | None = None
    if mh is not None:

        def make_train_iter(seed):
            # Heterogeneous mixed batches over the unified index; one shared padded
            # shape; balanced-binning DP sharding (same as the single-pack path).
            return multihead_batched_iter(
                mh["index_dir"],
                atom_budget=args.atom_budget,
                edge_budget=args.edge_budget,
                graph_budget=args.graph_budget,
                seed=seed,
                rank=dist["process_id"],
                world_size=n_proc,
                num_workers=args.num_workers,
                prefetch=args.prefetch,
                mega_batches=args.mega_batches,
                base_atoms=args.base_atoms,
                base_edges=args.base_edges,
            )
    elif args.data_format == "packed":
        packed_loader = PackedPrefetch(
            train_spec,
            batch_size=args.batch_size,
            base_atoms=args.base_atoms,
            base_edges=args.base_edges,
            energy_offsets=offsets,
            num_workers=args.num_workers,
            prefetch=args.prefetch,
            max_struct=args.max_frames,
            rank=dist["process_id"],
            world_size=n_proc,
            fixed_atoms=args.fixed_atoms,
            fixed_edges=args.fixed_edges,
            grid=use_grid,
            atom_budget=args.atom_budget,
            edge_budget=args.edge_budget,
            graph_budget=args.graph_budget,
            mega_batches=args.mega_batches,
        )
        # Hide the first epoch's host planning behind optimizer init / AOT load.
        packed_loader.prefetch_plan(start_epoch + 1)

        def make_train_iter(seed):
            return packed_loader.iter_epoch(seed)
    else:

        def make_train_iter(seed):
            return batched_iter(
                train_frames,
                batch_size=args.batch_size,
                base_atoms=args.base_atoms,
                base_edges=args.base_edges,
                energy_offsets=offsets,
                seed=seed,
            )

    # ---------------- optimizer ----------------
    params, static = eqx.partition(chain, eqx.is_inexact_array)
    # LR schedule: compute total steps from full training span. When resuming
    # with `_opt.eqx` present, the saved schedule step counter inside opt_state
    # ensures the LR resumes at the right phase; if only chain weights are
    # loaded (legacy resume), the schedule starts at step 0.
    # Global batch spans all processes -> steps/epoch counts global batches.
    if mh is not None:
        # Multi-head: one "epoch" = one pass over the combined (subsampled) index.
        total_atoms = int(mh_ds_main.natoms.sum())
        steps_per_epoch = max(1, int(total_atoms / (0.96 * args.atom_budget * n_proc)))
        global_batch = f"~{args.atom_budget}atoms x {n_proc} (multihead x{len(mh['heads'])})"
    elif balanced:
        # Atom-budget packing: structs/batch varies, so estimate steps/epoch from
        # total atoms / (per-GPU atom_budget x fill x n_proc). Only sizes the LR
        # warmup (warmup_constant), so an estimate within ~10% is fine. ~0.96 =
        # measured atom fill of the balanced packer.
        total_atoms = int(PackedDataset(**train_spec).natoms[:n_train].sum())
        steps_per_epoch = max(1, int(total_atoms / (0.96 * args.atom_budget * n_proc)))
        global_batch = f"~{args.atom_budget}atoms x {n_proc}"
    else:
        global_batch = args.batch_size * n_proc
        steps_per_epoch = max(1, n_train // global_batch)
    total_steps = args.n_epochs * steps_per_epoch
    base_lr = args.base_lr * (n_proc if args.lr_scale_by_devices else 1)
    lr_or_schedule = _build_schedule(
        base_lr,
        args.lr_schedule,
        total_steps=total_steps,
        warmup_frac=args.warmup_frac,
        quench_frac=args.quench_frac,
        quench_alpha=args.quench_final_frac,
        hold_steps=args.hold_steps,
        quench_steps=args.quench_steps,
    )
    _lr_note = (
        f" (={args.base_lr}x{n_proc} devices)" if args.lr_scale_by_devices and distributed else ""
    )
    log0(
        f"LR: schedule={args.lr_schedule} base_lr={base_lr}{_lr_note} "
        f"total_steps={total_steps} (steps/epoch={steps_per_epoch})"
    )

    tx = _build_optimizer(
        chain,
        lr_or_schedule,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
        optimizer=args.optimizer,
        muon_lr_scale=args.muon_lr_scale,
    )
    if args.clip_grad_norm > 0.0:
        import optax

        # Clip the raw gradient *before* Adam moments update. Matches the
        # MACE-MP / Allegro convention; addresses Adam's tendency to take
        # outsized early-training steps when v_hat is still small.
        tx = optax.chain(optax.clip_by_global_norm(args.clip_grad_norm), tx)
        print(f"grad clipping: global L2 norm capped at {args.clip_grad_norm}")
    if args.skip_nonfinite_updates:
        import optax

        # Outermost wrap: if the raw grads are non-finite, skip clip+Adam+EMA for
        # that batch (params/opt-state intact) instead of poisoning them with NaN.
        tx = optax.apply_if_finite(tx, max_consecutive_errors=args.max_nonfinite)
        print(f"skip-nonfinite-updates: ON (raise after {args.max_nonfinite} consecutive)")
    opt_state = tx.init(params)

    # If resuming, look for `_opt.eqx` next to the chain checkpoint and load
    # Adam moments + schedule step into opt_state.
    if args.resume_from is not None:
        # Sibling next to the resume ckpt (works for both ckpt_epoch_* and
        # ckpt_step_* — same stem + "_opt").
        opt_ckpt = Path(args.resume_from).with_name(f"{resume_stem}_opt.eqx")
        if opt_ckpt.exists():
            opt_state = eqx.tree_deserialise_leaves(opt_ckpt, opt_state)
            print(f"resumed optimizer state from {opt_ckpt}")
        else:
            print(f"  (no opt_state at {opt_ckpt}; Adam moments start fresh)")
    # EMA on params (eval-only). Tracks `params` with `ema_decay`. ema_params is
    # initialised to a copy of the current params; eval + checkpoints use it.
    # Now folded into the JIT'd train_step (see _train_step_factory) to keep the
    # GPU continuously fed — separate eqx.filter_jit caused a 2-3× host-side
    # dispatch tail per iter on A100 (nsys profile 2026-05-29).
    use_ema = args.ema_decay > 0.0
    ema_decay = args.ema_decay
    # EMA is independently updated and both trees are donated to the compiled
    # step. A shallow tree copy aliases every parameter buffer, which makes the
    # first donated call attempt to donate each buffer twice.
    ema_params = jax.tree.map(lambda x: jnp.array(x, copy=True), params) if use_ema else None

    # If resuming with EMA, load the saved EMA params from the sibling
    # `_ema.eqx` file (the EMA chain on disk) and re-extract its params.
    if args.resume_from is not None and use_ema:
        ema_ckpt = Path(args.resume_from).with_name(f"{resume_stem}_ema.eqx")
        if ema_ckpt.exists():
            ema_chain = eqx.tree_deserialise_leaves(ema_ckpt, chain)
            ema_params, _ = eqx.partition(ema_chain, eqx.is_inexact_array)
            print(f"resumed EMA params from {ema_ckpt}")
        else:
            print(f"  (no EMA at {ema_ckpt}; EMA starts from current params)")

    # Replicate trainable state on every device. Each process builds identical
    # params/opt/EMA (same seed + deterministic init), so this just wraps them
    # as globally-replicated arrays; the shard_map step keeps them in sync via
    # the gradient all-reduce.
    if distributed:
        params = replicate_tree(params, mesh)
        opt_state = replicate_tree(opt_state, mesh)
        if use_ema:
            ema_params = replicate_tree(ema_params, mesh)
    else:
        # Commit to the local device so step 1 presents the same input sharding
        # as step 2 does (see commit_tree) -- otherwise the step is compiled
        # twice, which costs a second full compile on every process start.
        params = commit_tree(params)
        opt_state = commit_tree(opt_state)
        if use_ema:
            ema_params = commit_tree(ema_params)

    train_step = _train_step_factory(
        static,
        num_species,
        tx,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        virial_weight=args.virial_weight,
        ema_decay=ema_decay,
        loss_type=args.loss,
        huber_delta=args.huber_delta,
        huber_delta_energy=args.huber_delta_energy,
        cutoff_weight=args.radial_cutoff_weight,
        mesh=mesh if distributed else None,
        local_loss_mean=bool(args.local_loss_mean) and distributed,
    )
    # JIT-cached eval-one closure; shared across all eval_sets so the cache
    # survives between training epochs and ID/OOD eval passes.
    eval_one_fn = _make_eval_one()
    use_virial = args.virial_weight > 0.0
    _delta_e = args.huber_delta if args.huber_delta_energy is None else args.huber_delta_energy
    huber_note = (
        f" (huber_delta={args.huber_delta}, delta_E={_delta_e}, per-atom E + conditional F)"
        if args.loss == "huber"
        else ""
    )
    print(
        f"loss={args.loss}{huber_note}"
        + f" weights: E={args.energy_weight} F={args.force_weight}"
        + (f" V={args.virial_weight}" if use_virial else "")
    )
    if args.log_every:
        print(f"intra-epoch logging: every {args.log_every} steps")

    if use_ema:
        print(f"EMA: decay={ema_decay} (folded into train_step JIT, used for eval + ckpt)")

    # ---------------- train loop ----------------
    history: list[dict] = []
    log_path = out_dir / "log.jsonl"
    if args.resume_from is None and primary:
        log_path.write_text("")  # fresh run (resume -> append-only, preserve history)
    train(
        parser=p,
        args=args,
        card=card,
        dist=dist,
        distributed=distributed,
        ema_params=ema_params,
        eval_one_fn=eval_one_fn,
        eval_sets=eval_sets,
        history=history,
        log0=log0,
        log_path=log_path,
        make_train_iter=make_train_iter,
        mesh=mesh,
        mh=mh,
        n_proc=n_proc,
        num_species=num_species,
        offsets=offsets,
        opt_state=opt_state,
        out_dir=out_dir,
        packed_loader=packed_loader,
        params=params,
        primary=primary,
        resume_is_step=resume_is_step,
        start_epoch=start_epoch,
        start_global_step=start_global_step,
        static=static,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        train_step=train_step,
        use_ema=use_ema,
        use_grid=use_grid,
        use_virial=use_virial,
    )
