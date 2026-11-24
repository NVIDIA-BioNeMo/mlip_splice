# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Iterate batches, maintain checkpoints, and record training metrics."""

import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Best-effort: keep XLA from preallocating huge buffers, helps small-model multi-run.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from splice.data import pad_to
from splice.padding import choose_bucket_sizes
from splice.parallel import (
    localize_replicated_tree,
    place_local_batch,
    shard_batch,
)
from splice.training.objective import (
    _STAT,
    _STAT_NAMES,
    _eval_loop,
    _eval_subset,
)
from splice.training.report import _kiln_timings, _save_chain, _write_compile_profile


def train(
    *,
    parser,
    args,
    card,
    dist,
    distributed,
    ema_params,
    eval_one_fn,
    eval_sets,
    history,
    log0,
    log_path,
    make_train_iter,
    mesh,
    mh,
    n_proc,
    num_species,
    offsets,
    opt_state,
    out_dir,
    packed_loader,
    params,
    primary,
    resume_is_step,
    start_epoch,
    start_global_step,
    static,
    steps_per_epoch,
    total_steps,
    train_step,
    use_ema,
    use_grid,
    use_virial,
):
    train_start = time.time()

    # Step-based ckpt/eval (mid-epoch) — for epochs >> a job's walltime (OMat).
    # global_step is the master progress counter (persists across resume); in
    # step-mode the per-iterator shuffle seed is global_step-based so every
    # segment draws a FRESH order (no data re-seen on sub-epoch resume), and
    # termination is global_step >= total_steps (the epoch label is cosmetic).
    step_mode = args.checkpoint_every_steps > 0 or args.eval_every_steps > 0
    if start_global_step == 0 and not resume_is_step and args.resume_from is not None:
        start_global_step = start_epoch * steps_per_epoch  # epoch ckpt -> derive
    global_step = start_global_step
    if step_mode:
        log0(
            f"STEP-MODE: ckpt every {args.checkpoint_every_steps} steps, "
            f"eval every {args.eval_every_steps} steps (frac={args.eval_frac}); "
            f"total_steps={total_steps}, start_global_step={global_step}"
        )

    def _step_eval_ckpt(
        g_step, epoch, params, opt_state, ema_params, do_eval, do_ckpt, full_eval=False
    ):
        """Mid-epoch step-based eval (subset) + checkpoint. Collective-safe:
        eval runs _eval_loop on ALL ranks (all-reduce inside); ckpt writes on the
        primary only (device_get is a local transfer); a barrier resyncs ranks
        after so a slow primary-only write never trips the next step's NCCL
        all-reduce timeout."""
        params_eval = localize_replicated_tree(params) if distributed else params
        ema_eval = localize_replicated_tree(ema_params) if (distributed and use_ema) else ema_params
        chain_now = eqx.combine(params_eval, static)
        eval_chain = eqx.combine(ema_eval, static) if use_ema else chain_now
        entry = {"global_step": g_step, "epoch": epoch, "wall_s": time.time() - train_start}
        if do_eval and eval_sets:
            for name, frames in eval_sets.items():
                fr = frames if full_eval else _eval_subset(frames, args.eval_frac)
                em = mh["eval"].get(name) if mh else None
                entry[f"eval_{name}"] = _eval_loop(
                    eval_chain,
                    fr,
                    num_species=num_species,
                    batch_size=args.eval_batch_size or args.batch_size,
                    offsets=em["offsets"] if em else offsets,
                    base_atoms=args.base_atoms,
                    base_edges=args.base_edges,
                    with_virial=(use_virial and em["virial"]) if em else use_virial,
                    eval_one=eval_one_fn,
                    rank=dist["process_id"],
                    world_size=n_proc,
                    head_id=em["head_id"] if em else None,
                )
        if do_ckpt and primary:
            stem = f"ckpt_step_{g_step:09d}"
            _save_chain(chain_now, out_dir / f"{stem}.eqx")
            eqx.tree_serialise_leaves(out_dir / f"{stem}_opt.eqx", opt_state)
            if use_ema:
                _save_chain(eval_chain, out_dir / f"{stem}_ema.eqx")
            entry["checkpoint"] = f"{stem}.eqx"
            _persist_aot_cache()
        if primary:
            history.append(entry)
            with log_path.open("a") as fh:
                fh.write(json.dumps(entry) + "\n")
            msg = f"  [step {g_step} ep{epoch}]"
            if "checkpoint" in entry:
                msg += f" saved {entry['checkpoint']}"
            for k, v in entry.items():
                if k.startswith("eval_"):
                    msg += f"  {k}: E_mae/atom={v['E_mae_per_atom']:.5f} F_mae={v['F_mae']:.5f}"
                    if "virial_mae_per_graph" in v:
                        msg += f" V_mae/graph={v['virial_mae_per_graph']:.4g}"
            print(msg, flush=True)
        if distributed:
            multihost_utils.sync_global_devices(f"step_{g_step}")

    # In step-mode the loop is bounded by global_step (each segment restarts the
    # epoch label at 1 but global_step persists), so make the range effectively
    # unbounded and break on total_steps. Epoch-mode keeps the exact old range.
    epoch_range_end = (start_epoch + 10**9) if step_mode else (start_epoch + args.n_epochs)
    # Persist tuned recipes so a resubmitted process skips the multi-hour lowering,
    # planning and autotuning (cubins are disk-cached separately). Entered manually
    # (not `with`) so the settings stay active across the loop without re-indenting
    # it; the process ends after the loop, so the unclosed context manager is
    # harmless. --search-budget-ms bounds recipe search by elapsed wall-clock
    # time; --shortlist-budget-ms bounds predicted realization cost, taming the long
    # cold compile of heavy graphs (full MACE-L took 44 min on GB200 uncapped).
    #
    # A reusable cache requires a workload identity. Tuning depends on the input
    # distribution as well as the program, so the identity digests both: the
    # architecture and card files, and the batch-shape budgets that set the shapes
    # Loom tunes for. Stable across resubmissions of this run, and changed by any
    # of those, so stale recipes are never reused.
    import hashlib
    import pickle

    import loom

    # Predicted realization cost is max(assembly cost / workers, serial tuning
    # cost); the shortlist budget is not an elapsed-time deadline for realization.
    _aot_kw = {
        "compile_options": loom.Compile(
            search_budget_ms=args.search_budget_ms,
            shortlist_budget_ms=args.shortlist_budget_ms,
        ),
    }
    if args.profile_compile:
        # kiln writes one tune report per realized recipe into this directory,
        # each stating what assembly and on-device tuning cost. Loom's own
        # phases arrive through capture_compile_profile below.
        _aot_kw["debug_dump_dir"] = str(Path(args.profile_compile))

    def _aot_workload_identity():
        h = hashlib.sha256()
        for _p in (args.arch, args.card):
            try:
                h.update(Path(_p).read_bytes())
            except OSError:
                h.update(str(_p).encode())
        h.update(
            repr(
                (
                    card.get("cutoff"),
                    getattr(args, "atom_budget", None),
                    getattr(args, "edge_budget", None),
                    getattr(args, "graph_budget", None),
                    n_proc,
                )
            ).encode()
        )
        return f"splice-train-{h.hexdigest()[:32]}"

    _aot_identity = _aot_workload_identity()
    _aot_cache_path = out_dir / "loom_aot_cache.pkl"
    _aot_cache = {}
    if _aot_cache_path.exists():
        try:
            with open(_aot_cache_path, "rb") as _f:
                _blob = pickle.load(_f)
            # reject a cache tuned for a different workload rather than replaying
            # recipes that were not tuned for these shapes
            if isinstance(_blob, dict) and _blob.get("identity") == _aot_identity:
                _aot_cache = _blob.get("cache") or {}
                log0(f"[aot] loaded loom cache: {len(_aot_cache)} entries from {_aot_cache_path}")
            else:
                log0("[aot] loom cache is for a different workload; compiling fresh")
        except Exception as _e:
            _aot_cache = {}
            log0(f"[aot] loom cache load failed ({_e!r}); compiling fresh")

    _aot_cm = loom.aot(cache=_aot_cache, workload_identity=_aot_identity, **_aot_kw)
    _aot_cm.__enter__()
    log0(
        f"[aot] loom AOT cache active ({len(_aot_cache)} preloaded, "
        f"id={_aot_identity[-8:]}) "
        f"search_budget_ms={args.search_budget_ms} "
        f"shortlist_budget_ms={args.shortlist_budget_ms}"
    )

    def _persist_aot_cache():
        """primary-only; exit+reenter aot() so pending tuned recipes flush into
        the dict, then atomically pickle it. Never raises into training."""
        nonlocal _aot_cm
        if not primary:
            return
        try:
            _aot_cm.__exit__(None, None, None)
            _tmp = _aot_cache_path.with_suffix(".pkl.tmp")
            with open(_tmp, "wb") as _f:
                pickle.dump({"identity": _aot_identity, "cache": _aot_cache}, _f)
            os.replace(_tmp, _aot_cache_path)
            log0(f"[aot] persisted loom cache: {len(_aot_cache)} entries")
        except Exception as _e:
            log0(f"[aot] loom cache persist failed ({_e!r}); continuing")
        finally:
            _aot_cm = loom.aot(cache=_aot_cache, workload_identity=_aot_identity, **_aot_kw)
            _aot_cm.__enter__()

    _profile_first_step = bool(args.profile_compile)
    _loop_prof = args.profile_loop
    _loop_steps = int(args.profile_loop_steps)
    _loop_warmup = int(args.profile_warmup)
    _prof_cm = None
    _prof_t0 = None
    _prof_seen = 0
    _replay_mod = None
    if _loop_prof:
        if _loop_steps < 1:
            parser.error("--profile-loop-steps must be >= 1")
        if _loop_warmup < 0:
            parser.error("--profile-warmup must be >= 0")
        # Each rank traces its own GPU; sharing one directory races CUPTI/Perfetto.
        if distributed:
            _loop_prof = str(Path(_loop_prof) / f"rank{dist['process_id']}")
        log0(f"[profile-loop] warmup={_loop_warmup} traced={_loop_steps} -> {_loop_prof}")

    def _profile_compile_step(step_args):
        """Measure one first step as three disjoint spans, and report them.

        `eqx.filter_jit` exposes `lower`, and lowering is where Loom's compiler
        runs, so the spans can be separated instead of inferred: lower, then
        `compile` for XLA's own work, then the first call, where kiln assembles
        and tunes. Their sum is the wall clock, which is what makes the table a
        decomposition rather than three numbers that happen to be related.
        """
        from loom._src.primitive.profile import capture_compile_profile

        directory = Path(args.profile_compile)
        directory.mkdir(parents=True, exist_ok=True)
        batch = step_args[-1]

        def _extent(name, axis):
            # Under a mesh the batch carries a leading device axis; the padded
            # extent is the last axis of the species/edge vectors regardless.
            return int(np.asarray(jax.device_get(batch[name])).shape[axis])

        stages = {}
        started = time.perf_counter()
        with capture_compile_profile() as phases:
            lowered = train_step.lower(*step_args)
        stages["lower_ms"] = (time.perf_counter() - started) * 1_000.0

        mark = time.perf_counter()
        compiled = lowered.compile()
        stages["xla_compile_ms"] = (time.perf_counter() - mark) * 1_000.0

        mark = time.perf_counter()
        outputs = jax.block_until_ready(compiled(*step_args))
        stages["first_execute_ms"] = (time.perf_counter() - mark) * 1_000.0

        _write_compile_profile(
            directory,
            {
                "arch": args.arch,
                "wall_ms": (time.perf_counter() - started) * 1_000.0,
                "stages": stages,
                "loom": dict(phases),
                "kiln": _kiln_timings(directory),
                "shapes": {
                    "atoms": _extent("species", -1),
                    "edges": _extent("inda", -1),
                    "graphs": _extent("nats", -1),
                },
            },
        )
        return outputs

    def _prepare_local(local_sub):
        if not use_grid:
            return local_sub
        # Rank agreement determines this batch's pad shape. The allgather must
        # stay lockstep with the previous step's gradient all-reduce.
        ra, re = local_sub["real_atoms"], local_sub["real_edges"]
        if distributed:
            g = np.asarray(multihost_utils.process_allgather(np.array([ra, re], np.int32)))
            ra, re = int(g[:, 0].max()), int(g[:, 1].max())
        A, E = choose_bucket_sizes(ra, re, base_atoms=args.base_atoms, base_edges=args.base_edges)
        return pad_to(
            local_sub["raw"],
            local_sub["targets"],
            local_sub["n_frames"],
            n_atoms_padded=A,
            n_edges_padded=E,
        )

    def _stage_batch(host):
        started = time.perf_counter()
        with jax.profiler.TraceAnnotation("splice.h2d"):
            batch = shard_batch(host, mesh) if distributed else place_local_batch(host)
        return batch, time.perf_counter() - started

    def _next_staged(iterator):
        """Fetch, pad, and place one batch without blocking the dispatch thread."""
        item = next(iterator, None)
        if item is None:
            return None
        return _stage_batch(_prepare_local(item))

    io_profile = {
        "h2d_s": [],
        "stage_wait_s": [],
        "step_dispatch_s": [],
    }
    try:
        for epoch in range(start_epoch + 1, epoch_range_end + 1):
            epoch_start = time.time()
            stats_zero = np.zeros((len(_STAT_NAMES),), dtype=np.dtype(jnp.result_type(float)))
            if distributed:
                epoch_stats_device = shard_batch({"stats": stats_zero}, mesh)["stats"]
            else:
                epoch_stats_device = jax.device_put(stats_zero, jax.local_devices()[0])
            n_batches = 0
            hit_max_steps = False
            # Step-mode: seed the iterator with global_step (unique per segment/epoch
            # -> fresh shuffle on resume). Epoch-mode: seed with the epoch index
            # (unchanged behaviour).
            iter_seed = (args.seed + global_step) if step_mode else epoch
            epoch_iter = iter(make_train_iter(iter_seed))
            # Consume this epoch's plan first, then hide the next epoch's
            # ~0.6s SPICE2 pack behind this epoch's GPU work.
            if packed_loader is not None and not step_mode:
                packed_loader.prefetch_plan(epoch + 1)
            # Keep CPU assemble + H2D off the JAX dispatch thread so they overlap
            # the previous compiled step. Staging keeps batch construction and
            # device_put off the dispatch thread while the GPU executes.
            #
            # GRID's _prepare_local contains a cross-rank shape collective. Keep
            # its original host ordering rather than enqueueing batch N+1's
            # allgather ahead of batch N's gradient all-reduce.
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="splice-h2d") as staging:
                if use_grid:
                    item = next(epoch_iter, None)
                    staged = (
                        staging.submit(_stage_batch, _prepare_local(item))
                        if item is not None
                        else None
                    )
                else:
                    staged = staging.submit(_next_staged, epoch_iter)
                while staged is not None:
                    wait_started = time.perf_counter()
                    result = staged.result()
                    stage_wait_s = time.perf_counter() - wait_started
                    if result is None:
                        break
                    batch, h2d_s = result
                    if use_grid:
                        item = next(epoch_iter, None)
                        next_staged = (
                            staging.submit(_stage_batch, _prepare_local(item))
                            if item is not None
                            else None
                        )
                    else:
                        next_staged = staging.submit(_next_staged, epoch_iter)
                    if args.profile_io:
                        io_profile["h2d_s"].append(h2d_s)
                        io_profile["stage_wait_s"].append(stage_wait_s)
                    if _profile_first_step:
                        step_args = (
                            (params, opt_state, ema_params, epoch_stats_device, batch)
                            if use_ema
                            else (params, opt_state, epoch_stats_device, batch)
                        )
                        _profile_compile_step(step_args)
                        if primary:
                            _persist_aot_cache()
                        return
                    if _loop_prof and _prof_seen == 0:
                        sp = batch["species"].shape
                        ed = batch["inda"].shape
                        gr = batch["nats"].shape
                        log0(
                            f"[profile-loop] batch species={tuple(sp)} inda={tuple(ed)} "
                            f"nats={tuple(gr)}",
                            flush=True,
                        )
                    if _loop_prof and _prof_seen == _loop_warmup:
                        drain = (
                            (params, opt_state, ema_params, epoch_stats_device)
                            if use_ema
                            else (params, opt_state, epoch_stats_device)
                        )
                        jax.block_until_ready(drain)
                        if primary:
                            _persist_aot_cache()
                        try:
                            import loom._native as _replay_mod

                            # Replay counters are optional: the training host
                            # improvements also work with the mainline runtime.
                            if not all(
                                hasattr(_replay_mod, name)
                                for name in ("reset_replay_stats", "replay_stats")
                            ):
                                _replay_mod = None
                        except Exception:
                            _replay_mod = None
                        if _replay_mod is not None:
                            _replay_mod.reset_replay_stats()
                        Path(_loop_prof).mkdir(parents=True, exist_ok=True)
                        _prof_cm = jax.profiler.trace(_loop_prof, create_perfetto_trace=True)
                        _prof_cm.__enter__()
                        _prof_t0 = time.perf_counter()
                        log0(
                            f"[profile-loop] tracing {_loop_steps} steps -> {_loop_prof}",
                            flush=True,
                        )
                    t_step = time.perf_counter()
                    with jax.profiler.StepTraceAnnotation("splice.train", step_num=global_step):
                        if use_ema:
                            (
                                params,
                                opt_state,
                                ema_params,
                                epoch_stats_device,
                                stats,
                            ) = train_step(
                                params,
                                opt_state,
                                ema_params,
                                epoch_stats_device,
                                batch,
                            )
                        else:
                            params, opt_state, epoch_stats_device, stats = train_step(
                                params, opt_state, epoch_stats_device, batch
                            )
                    if args.profile_io:
                        io_profile["step_dispatch_s"].append(time.perf_counter() - t_step)
                    n_batches += 1
                    global_step += 1
                    _prof_seen += 1
                    if _loop_prof and _prof_seen == _loop_warmup + _loop_steps:
                        t_disp = time.perf_counter()
                        drain = (
                            (params, opt_state, ema_params, epoch_stats_device)
                            if use_ema
                            else (params, opt_state, epoch_stats_device)
                        )
                        jax.block_until_ready(drain)
                        t_end = time.perf_counter()
                        if _prof_cm is not None:
                            _prof_cm.__exit__(None, None, None)
                            _prof_cm = None
                        n = _loop_steps
                        log0(
                            f"[profile-loop] dispatch {(t_disp - _prof_t0) / n * 1e3:.3f} ms/step "
                            f"(host return; GPU may still be running)",
                            flush=True,
                        )
                        log0(
                            f"[profile-loop] wall     {(t_end - _prof_t0) / n * 1e3:.3f} ms/step "
                            f"(including drain)",
                            flush=True,
                        )
                        if _replay_mod is not None:
                            st = _replay_mod.replay_stats()
                            log0(
                                "[profile-loop] loom per step "
                                f"replayed={st['replayed'] / n:.2f} "
                                f"captured={st['captured'] / n:.2f}",
                                flush=True,
                            )
                        if primary:
                            _persist_aot_cache()
                        log0(f"[profile-loop] trace written to {_loop_prof}", flush=True)
                        # Ranks must leave the process group together; a primary-only
                        # return tears down the coordinator and the others log a
                        # WatchTasks connection-refused on the way out.
                        if distributed:
                            multihost_utils.sync_global_devices("profile_loop_end")
                        return
                    if args.log_every and n_batches % args.log_every == 0 and _prof_cm is None:
                        log_stats = (
                            jnp.mean(stats, axis=0)
                            if distributed and args.local_loss_mean
                            else (jnp.sum(stats, axis=0) if distributed else stats)
                        )
                        log_stats_h = np.asarray(jax.device_get(log_stats))
                        _creg = (
                            f" cutoff_reg={log_stats_h[_STAT['cutoff_reg']]:.4f}"
                            if args.radial_cutoff_weight > 0.0
                            else ""
                        )
                        if primary:
                            print(
                                f"  [ep{epoch} step {n_batches} g{global_step}] "
                                f"loss={log_stats_h[_STAT['loss']]:.6f} "
                                f"E_mse={log_stats_h[_STAT['E_mse']]:.6f} "
                                f"F_mse={log_stats_h[_STAT['F_mse']]:.6f} "
                                f"E_mae={log_stats_h[_STAT['E_mae']]:.6f} "
                                f"F_mae={log_stats_h[_STAT['F_mae']]:.6f}{_creg}",
                                flush=True,
                            )
                    if step_mode:
                        do_eval = args.eval_every_steps and global_step % args.eval_every_steps == 0
                        do_ckpt = (
                            args.checkpoint_every_steps
                            and global_step % args.checkpoint_every_steps == 0
                        )
                        if do_eval or do_ckpt:
                            _step_eval_ckpt(
                                global_step,
                                epoch,
                                params,
                                opt_state,
                                ema_params,
                                do_eval,
                                do_ckpt,
                            )
                        if global_step >= total_steps:
                            break
                    if args.max_steps is not None and global_step >= args.max_steps:
                        hit_max_steps = True
                        break
                    staged = next_staged

            # Step-mode: stop the whole run once total_steps is reached, after a
            # final ckpt+eval (the epoch-end block below is for completed epochs).
            # full_eval: a step-mode run ends mid-epoch, so the epoch-end block's
            # is_last_epoch full eval never fires -- this is the ONLY place the
            # headline number gets computed on the whole val set.
            if step_mode and global_step >= total_steps:
                _step_eval_ckpt(
                    global_step,
                    epoch,
                    params,
                    opt_state,
                    ema_params,
                    do_eval=bool(eval_sets),
                    do_ckpt=True,
                    full_eval=True,
                )
                break

            # One device reduction and one host sync for all train statistics.
            with jax.profiler.TraceAnnotation("splice.epoch_reduce", epoch=epoch):
                global_epoch_stats = (
                    jnp.mean(epoch_stats_device, axis=0)
                    if distributed and args.local_loss_mean
                    else (
                        jnp.sum(epoch_stats_device, axis=0) if distributed else epoch_stats_device
                    )
                )
                epoch_stats_h = np.asarray(jax.device_get(global_epoch_stats))
            train_epoch_time = time.time() - epoch_start
            step_ms_mean = (train_epoch_time / max(n_batches, 1)) * 1000
            n = max(n_batches, 1)
            epoch_loss_h = float(epoch_stats_h[_STAT["loss"]])
            epoch_E_mse_h = float(epoch_stats_h[_STAT["E_mse"]])
            epoch_F_mse_h = float(epoch_stats_h[_STAT["F_mse"]])
            epoch_V_mse_h = float(epoch_stats_h[_STAT["V_mse"]])
            epoch_E_mae_h = float(epoch_stats_h[_STAT["E_mae"]])
            epoch_F_mae_h = float(epoch_stats_h[_STAT["F_mae"]])
            epoch_V_mae_h = float(epoch_stats_h[_STAT["V_mae"]])

            params_eval = localize_replicated_tree(params) if distributed else params
            ema_eval = (
                localize_replicated_tree(ema_params) if (distributed and use_ema) else ema_params
            )
            chain_now = eqx.combine(params_eval, static)
            eval_chain = eqx.combine(ema_eval, static) if use_ema else chain_now
            epoch_time = time.time() - epoch_start
            entry = {
                "epoch": epoch,
                "train_loss": epoch_loss_h / n,
                "train_E_mse": epoch_E_mse_h / n,
                "train_F_mse": epoch_F_mse_h / n,
                "train_E_mae": epoch_E_mae_h / n,
                "train_F_mae": epoch_F_mae_h / n,
                "epoch_s": epoch_time,
                "step_ms_mean": step_ms_mean,
                "n_batches": n_batches,
                "wall_s": time.time() - train_start,
            }
            if use_virial:
                entry["train_V_mse"] = epoch_V_mse_h / n
                entry["train_V_mae"] = epoch_V_mae_h / n

            is_last_epoch = epoch == start_epoch + args.n_epochs
            eval_due = bool(
                eval_sets
                and args.eval_every > 0
                and (epoch % args.eval_every == 0 or (is_last_epoch and not hit_max_steps))
            )
            if eval_due:
                for name, frames in eval_sets.items():
                    fr = frames if is_last_epoch else _eval_subset(frames, args.eval_frac)
                    em = mh["eval"].get(name) if mh else None
                    metrics = _eval_loop(
                        eval_chain,
                        fr,
                        num_species=num_species,
                        batch_size=args.eval_batch_size or args.batch_size,
                        offsets=em["offsets"] if em else offsets,
                        base_atoms=args.base_atoms,
                        base_edges=args.base_edges,
                        with_virial=(use_virial and em["virial"]) if em else use_virial,
                        eval_one=eval_one_fn,
                        rank=dist["process_id"],
                        world_size=n_proc,
                        head_id=em["head_id"] if em else None,
                    )
                    entry[f"eval_{name}"] = metrics

            last_epoch = start_epoch + args.n_epochs
            checkpoint_due = not hit_max_steps and (
                epoch % args.checkpoint_every == 0 or epoch == last_epoch
            )
            if primary and checkpoint_due:
                ckpt_path = out_dir / f"ckpt_epoch_{epoch:04d}.eqx"
                _save_chain(chain_now, ckpt_path)
                entry["checkpoint"] = ckpt_path.name
                _persist_aot_cache()
                opt_ckpt_path = out_dir / f"ckpt_epoch_{epoch:04d}_opt.eqx"
                eqx.tree_serialise_leaves(opt_ckpt_path, opt_state)
                if use_ema:
                    ema_path = out_dir / f"ckpt_epoch_{epoch:04d}_ema.eqx"
                    _save_chain(eval_chain, ema_path)
                    entry["checkpoint_ema"] = ema_path.name

            history.append(entry)
            if primary:
                with log_path.open("a") as fh:
                    fh.write(json.dumps(entry) + "\n")

            msg = (
                f"epoch {epoch:4d}/{start_epoch + args.n_epochs} "
                f"loss={entry['train_loss']:.6f} "
                f"E_mse={entry['train_E_mse']:.6f} "
                f"F_mse={entry['train_F_mse']:.6f} "
                f"E_mae={entry['train_E_mae']:.6f} "
                f"F_mae={entry['train_F_mae']:.6f} "
            )
            if use_virial:
                msg += f"V_mse={entry['train_V_mse']:.6f} V_mae={entry['train_V_mae']:.6f} "
            msg += f"epoch={epoch_time:.1f}s step={step_ms_mean:.1f}ms"
            if "checkpoint" in entry:
                msg += f" [saved {entry['checkpoint']}]"
            if any(k.startswith("eval_") for k in entry):
                for k, v in entry.items():
                    if k.startswith("eval_"):
                        msg += (
                            f"  {k}: E_mae/graph={v['E_mae_per_graph']:.4f} "
                            f"E_mae/atom={v['E_mae_per_atom']:.5f} F_mae={v['F_mae']:.5f}"
                        )
                        if "virial_mae_per_graph" in v:
                            msg += f" V_mae/graph={v['virial_mae_per_graph']:.4g}"
                        if "eval_step_ms_mean" in v:
                            msg += (
                                f" inf={v['eval_step_ms_mean']:.1f}ms/batch "
                                f"({v['eval_atoms_per_sec']:.0f} atoms/s)"
                            )
            if primary:
                print(msg, flush=True)

            if distributed and (eval_due or checkpoint_due):
                multihost_utils.sync_global_devices(f"epoch_{epoch}")
            if hit_max_steps:
                if primary:
                    _persist_aot_cache()
                break

        if primary:
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        log0(f"\ntotal wall time: {time.time() - train_start:.1f}s")
        step_ms_post_warmup = [e["step_ms_mean"] for e in history[2:] if "step_ms_mean" in e]
        if step_ms_post_warmup:
            med = statistics.median(step_ms_post_warmup)
            n_batches_typ = history[min(5, len(history) - 1)].get("n_batches", "?")
            log0(
                f"median post-warmup step time: {med:.2f} ms "
                f"({len(step_ms_post_warmup)} epochs, ~{n_batches_typ} batches/epoch)"
            )
        log0(f"results: {out_dir}")
        if distributed:
            multihost_utils.sync_global_devices("train_end")

    finally:
        if packed_loader is not None:
            packed_loader.close()
        if args.profile_io and primary:

            def _stats(xs):
                if not xs:
                    return None
                values = list(xs)
                tail = values[1:] or values
                return {
                    "n": len(values),
                    "first_s": float(values[0]),
                    "median_s": float(np.median(tail)),
                    "p90_s": float(np.percentile(tail, 90)),
                    "max_s": float(np.max(tail)),
                    "argmax_step": int(np.argmax(tail)) + 1,
                    "series_s": [round(float(x), 6) for x in values],
                }

            waits = packed_loader.wait_s if packed_loader is not None else []
            report = {
                "loader_wait": _stats(waits),
                "h2d": _stats(io_profile["h2d_s"]),
                "stage_wait": _stats(io_profile["stage_wait_s"]),
                "step_dispatch": _stats(io_profile["step_dispatch_s"]),
            }
            (out_dir / "io_profile.json").write_text(json.dumps(report, indent=2))
            for k, v in report.items():
                if v is None or not isinstance(v, dict):
                    continue
                log0(
                    f"[profile-io] {k:<13} n={v['n']:<4d} first={v['first_s'] * 1e3:9.1f}ms "
                    f"median={v['median_s'] * 1e3:7.2f}ms p90={v['p90_s'] * 1e3:7.2f}ms "
                    f"max={v['max_s'] * 1e3:8.2f}ms @dispatch{v['argmax_step']}"
                )
