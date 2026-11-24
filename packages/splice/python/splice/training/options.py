# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse training cards and command-line options."""

import argparse
import os

# Best-effort: keep XLA from preallocating huge buffers, helps small-model multi-run.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _load_training_card(path):
    """Read a training-card YAML and return a dict suitable for
    ``parser.set_defaults(**card)``. Keys are snake_case argparse dest names.
    Unknown keys are silently ignored (filtered in main against the parser's
    actual dests) so cards can include forward-compatible fields.
    """
    from pathlib import Path

    import yaml

    with open(Path(path)) as fh:
        card = yaml.safe_load(fh) or {}
    return card


def _peek_training_card_path(argv):
    """Pre-parse argv just to find `--training-card`; returns the path or None."""
    for i, tok in enumerate(argv):
        if tok == "--training-card" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--training-card="):
            return tok.split("=", 1)[1]
    return None


# Loom's own phases, in the order the compile performs them. These are *work*
# and not elapsed time: the Rust compiler renders and assembles across
# `compile_workers` threads, so the phases sum to more than the span that
# contains them. Only `stage_ms` below is wall clock.
def _parse_aot_budget(raw):
    """Number, or comma-separated per-transpose-depth schedule (0,1,2,...)."""
    raw = raw.strip()
    if "," in raw:
        return {i: float(x) for i, x in enumerate(raw.split(","))}
    return float(raw)


def parse_options():
    # Pre-parse to grab --training-card before building the full parser. The
    # card's values become argparse defaults so explicit CLI args still win.
    import sys

    card_path = _peek_training_card_path(sys.argv[1:])

    p = argparse.ArgumentParser()
    p.add_argument(
        "--training-card",
        default=None,
        help="YAML bundling training hyperparameters (batch_size, n_epochs, "
        "base_lr, lr_schedule, ema_decay, etc.). Values become defaults; "
        "explicit CLI args override.",
    )
    p.add_argument("--arch", required=True, help="architecture YAML")
    p.add_argument("--card", required=True, help="model card YAML")
    p.add_argument(
        "--train",
        nargs="+",
        help="train data: .xyz file(s), or one packed dir if --data-format packed",
    )
    p.add_argument("--eval", nargs="*", default=[], help="eval data: .xyz file(s) or packed dir(s)")
    p.add_argument(
        "--multihead-spec",
        default=None,
        help="YAML for multi-head Stage-2 finetuning: lists per-head packed datasets "
        "(head, train, val, periodic, weight) + e0_json + head order. Overrides "
        "--train/--eval; uses a multihead arch (multihead_energy / "
        "multihead_species_offsets). 10%% OMat replay = give the omat head weight 0.1.",
    )
    p.add_argument(
        "--data-format",
        default="extxyz",
        choices=["extxyz", "packed"],
        help="extxyz: in-memory ASE read (small sets). packed: streaming memmap "
        "loader over a packed dir (splice.packed_data) for large datasets.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="packed loader: process-pool workers for batch building (0 = inline).",
    )
    p.add_argument(
        "--prefetch",
        type=int,
        default=8,
        help="packed loader: batches kept in flight across the pool.",
    )
    p.add_argument(
        "--neighbour",
        default="free",
        choices=["free", "matscipy"],
        help="packed loader neighbour-list backend (free=non-periodic brute force).",
    )
    p.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help=(
            "Override the card's cutoff (Å) for a same-dataset Rc sweep. "
            "avg_num_neighbors is then taken from the card's "
            "`neighbors_by_cutoff` table (errors if the cutoff is not measured "
            "there, unless --allow-unmeasured-cutoff); the radial basis r_max "
            "and channelwise epsilon follow automatically."
        ),
    )
    p.add_argument(
        "--allow-unmeasured-cutoff",
        action="store_true",
        help=(
            "With --cutoff: proceed even if the cutoff has no measured entry in "
            "the card's neighbors_by_cutoff table, keeping the card's base "
            "avg_num_neighbors (epsilon may be miscalibrated). Off by default."
        ),
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--search-budget-ms",
        type=_parse_aot_budget,
        default=None,
        help="Recipe-search budget in elapsed wall-clock milliseconds: a scalar or "
        "comma-separated per-transpose-depth schedule (0,1,2,...). Default: unlimited.",
    )
    p.add_argument(
        "--shortlist-budget-ms",
        type=_parse_aot_budget,
        default=None,
        help="Source-shortlist budget in predicted realization milliseconds: "
        "max(assembly cost / workers, serial tuning cost). A scalar or "
        "comma-separated per-transpose-depth schedule (0,1,2,...). Default: unlimited.",
    )
    p.add_argument(
        "--profile-compile",
        default=None,
        metavar="DIR",
        help=(
            "Decompose the first training step's compile into its phases and "
            "write them, with kiln's tune reports, into DIR. Training stops "
            "after that step: the point is the compile, not the fit."
        ),
    )
    p.add_argument(
        "--profile-loop",
        default=None,
        metavar="DIR",
        help=(
            "After --profile-warmup steps, jax.profiler.trace the next "
            "--profile-loop-steps training steps (the real loop: loader, "
            "device_put prefetch, jit train_step, device-side metric add) "
            "and stop. Writes a Perfetto trace under DIR and prints dispatch "
            "vs drained wall ms/step plus Loom capture counts when available."
        ),
    )
    p.add_argument(
        "--profile-loop-steps",
        type=int,
        default=20,
        help="Steps captured by --profile-loop (default 20).",
    )
    p.add_argument(
        "--profile-warmup",
        type=int,
        default=5,
        help="Untraced steps before --profile-loop so compile/first-exec "
        "are not in the window (default 5).",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help=(
            "Batch size used for eval passes. Defaults to --batch-size. Larger "
            "is usually fine for eval (no backprop activations) and substantially "
            "reduces per-batch host-sync overhead."
        ),
    )
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--base-lr", type=float, default=1e-3)
    p.add_argument("--base-atoms", type=int, default=1024)
    p.add_argument("--base-edges", type=int, default=8192)
    # Multi-process data parallelism (one process per GPU). The launcher sets
    # SPLICE_NUM_PROCESSES/PROCESS_ID/COORDINATOR; train.py auto-detects.
    # `--batch-size` is PER-DEVICE; global batch = batch_size * num_processes.
    p.add_argument(
        "--fixed-atoms",
        type=int,
        default=None,
        help=(
            "Pad every batch to exactly this many atoms (REQUIRED for "
            "multi-process: all ranks must emit identically-shaped sub-batches). "
            "Must exceed the largest per-batch real atom count."
        ),
    )
    p.add_argument("--fixed-edges", type=int, default=None, help="As --fixed-atoms, for edges.")
    # Balanced binning (packed data only): pack each batch to a near-constant
    # total atom count instead of a fixed struct count, so the SINGLE compiled
    # pad shape is ~95% full for wide size-distribution data (OMat: 1-236
    # atoms/struct). Supersedes --fixed-atoms/-edges when set. See
    # splice.packed_data.plan_balanced_batches.
    p.add_argument(
        "--atom-budget",
        type=int,
        default=None,
        help=(
            "Enable balanced binning: max REAL atoms per batch (pad shape = "
            "atom_budget+1). ~ effective per-GPU batch (paper-256 -> ~1280). "
            "Packed data only; ignores --batch-size/--fixed-atoms when set."
        ),
    )
    p.add_argument(
        "--edge-budget",
        type=int,
        default=None,
        help="Balanced binning: fixed padded edge count (size to ~p99.9 of "
        "real edges/batch so the drop-guard rarely fires; OMat ~atom_budget*58).",
    )
    p.add_argument(
        "--graph-budget",
        type=int,
        default=256,
        help="Balanced binning: fixed padded graph count (structs/batch capped "
        "at graph_budget-1 by early-flush; graphs are cheap so set generously).",
    )
    p.add_argument(
        "--mega-batches",
        type=int,
        default=32,
        help="Balanced binning: mega-window size in units of batches (sort+pack "
        "within each window; larger = tighter fill + milder size-locality).",
    )
    p.add_argument(
        "--lr-scale-by-devices",
        action="store_true",
        help="Linearly scale base_lr by the number of processes (global-batch rule).",
    )
    p.add_argument(
        "--local-loss-mean",
        action="store_true",
        help=(
            "Data-parallel: DDP-style local mean. Skip the per-step count "
            "AllReduce and normalize each rank by its local real-atom/graph "
            "counts. Gradients are allreduce-mean'd across ranks (jax.lax.pmean), "
            "matching PyTorch DDP. Equal local counts ⇒ same as the default "
            "exact global mean; unequal counts ⇒ unweighted average of per-rank "
            "means. No effect on a single process."
        ),
    )
    p.add_argument("--eval-every", type=int, default=10, help="epochs between evals")
    p.add_argument("--checkpoint-every", type=int, default=50, help="epochs between checkpoints")
    # Step-based ckpt/eval (for epochs >> a job's walltime, e.g. an epoch of
    # ~368k steps against a 5h scheduler cap). When set (>0), they fire mid-epoch on a
    # global_step counter that persists across resume; the per-epoch ckpt/eval
    # above still run at epoch boundaries (full eval at epoch end).
    p.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=0,
        help="Steps between mid-epoch checkpoints (ckpt_step_<g>.eqx + _opt/_ema). "
        "0 = off (epoch-based only).",
    )
    p.add_argument(
        "--eval-every-steps",
        type=int,
        default=0,
        help="Steps between mid-epoch SUBSET evals (fine-grained learning curve). 0 = off.",
    )
    p.add_argument(
        "--eval-frac",
        type=float,
        default=1.0,
        help="Fraction of each eval set used by PERIODIC evals (both mid-epoch "
        "--eval-every-steps and epoch-end). The FINAL epoch always uses the full "
        "set, so the reported end-of-run number is never a subsample.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--lr-schedule",
        default="constant",
        choices=[
            "constant",
            "cosine",
            "warmup_cosine",
            "constant_then_quench",
            "warmup_constant",
            "warmup_constant_quench",
            "hold_quench_hold",
        ],
        help=(
            "LR schedule. `constant` keeps base_lr fixed. `cosine` decays "
            "base_lr → 0 over the whole training run. `warmup_cosine` linearly "
            "ramps from 0 → base_lr over `warmup_frac` of total steps, then "
            "cosine-decays to 0. `constant_then_quench` runs at base_lr for the "
            "first (1 - quench_frac), then cosine-decays to 0 over the last "
            "quench_frac of steps."
        ),
    )
    p.add_argument(
        "--warmup-frac",
        type=float,
        default=0.05,
        help="Fraction of total steps used for warmup in `warmup_cosine` schedule.",
    )
    p.add_argument(
        "--quench-frac",
        type=float,
        default=0.2,
        help="Fraction of total steps used for the LR quench at the end "
        "(in `constant_then_quench` schedule).",
    )
    p.add_argument(
        "--quench-final-frac",
        type=float,
        default=0.0,
        help="Quench floor as a fraction of base_lr: the end-of-quench LR is "
        "base_lr * quench_final_frac (cosine `alpha`). Default 0.0 = decay to zero "
        "(prior behavior). E.g. 0.1 with base_lr 0.01 quenches down to a 0.001 floor.",
    )
    p.add_argument(
        "--hold-steps",
        type=int,
        default=100_000,
        help="hold_quench_hold: absolute optimizer step at which to STOP holding "
        "base_lr and begin the quench.",
    )
    p.add_argument(
        "--quench-steps",
        type=int,
        default=200_000,
        help="hold_quench_hold: LENGTH (optimizer steps) of the cosine quench window "
        "after --hold-steps; LR then stays flat at base_lr*quench_final_frac to the end.",
    )
    p.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="EMA decay for params. 0 disables EMA. Typical: 0.99. EMA params "
        "are tracked alongside training and used for eval + saved as "
        "`ckpt_epoch_NNNN_ema.eqx` next to the regular checkpoint.",
    )
    p.add_argument(
        "--optimizer",
        choices=["adam", "muon"],
        default="adam",
        help="'muon' = Newton-Schulz orthogonalized momentum on the hidden equivariant "
        "IrrepsLinear blocks + Adam(W) on the rest (embeddings/SC/uvu/radial/readouts/biases).",
    )
    p.add_argument(
        "--muon-lr-scale",
        type=float,
        default=1.0,
        help="LR multiplier for the Muon group relative to --base-lr (Muon often likes higher LR).",
    )
    p.add_argument(
        "--amsgrad",
        action="store_true",
        help="use AMSGrad (caps the 2nd-moment estimate at its running max). "
        "Tames the sharp-landscape gradient spikes that NaN corr=3 from-scratch.",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW-style weight decay. 0 = plain Adam. µP-aware: decay is "
        "scaled by the same per-leaf µP multiplier as the gradient.",
    )
    p.add_argument(
        "--energy-weight", type=float, default=1.0, help="Weight on the E-MSE term of the loss."
    )
    p.add_argument(
        "--force-weight",
        type=float,
        default=100.0,
        help=(
            "Weight on the F-MSE term of the loss. MACE uses 1000 "
            "in the standard E+F stage; 100 is a milder default."
        ),
    )
    p.add_argument(
        "--radial-cutoff-weight",
        type=float,
        default=0.0,
        help=(
            "If > 0, add a regularizer that pushes each fused channelwise-MP radial "
            "channel toward a SHORTER cutoff: weight * mean((support_radius/r_max)^2) "
            "over all CW channels. Shorter supports make more edges skip whole warps "
            "(anvil zero_skip; the same lever channel-sort exploits at deploy), trading "
            "a little accuracy for inference speed. The squash floor bounds supports at "
            "extent_floor*r_max, so channels shorten but never die. Logged as cutoff_reg."
        ),
    )
    p.add_argument(
        "--clip-grad-norm",
        type=float,
        default=0.0,
        help=(
            "If > 0, clip the global L2 norm of the gradient PyTree to this "
            "value before applying the optimizer (optax.clip_by_global_norm). "
            "Stabilises training for sharp losses (e.g. corr=3 or large V "
            "gradients during early training). 0 disables clipping."
        ),
    )
    p.add_argument(
        "--skip-nonfinite-updates",
        action="store_true",
        help=(
            "Skip the optimizer + EMA update on any step whose gradient is NaN/Inf "
            "(optax.apply_if_finite), keeping params/opt-state unchanged for that "
            "batch. Fresh-from-scratch MH-1 can transiently overflow fp32 in the "
            "cubic symmetric-contraction on dense/high-coordination batches; this "
            "lets the run shrug those off instead of dying. Raises only after "
            "--max-nonfinite consecutive bad steps (a genuinely stuck run)."
        ),
    )
    p.add_argument(
        "--max-nonfinite",
        type=int,
        default=1000,
        help="Max CONSECUTIVE non-finite-gradient steps tolerated by "
        "--skip-nonfinite-updates before raising (default 1000).",
    )
    p.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "huber"],
        help=(
            "Loss type. 'mse' (default): per-term squared error. 'huber': the "
            "MACE-MH-1 recipe (arxiv 2510.25380 Eq.26) — Huber on PER-ATOM energy "
            "(E/N_a) + conditional-Huber forces (delta scaled 1.0/0.7/0.4/0.1 by "
            "|F| bins 100/200/300 eV/A) + Huber virial. Bounds gradients for the "
            "huge residuals a from-scratch model produces; required for stable "
            "corr=3 from-scratch training. Logged E_mse/F_mse stay squared-error "
            "(comparable across loss types)."
        ),
    )
    p.add_argument(
        "--huber-delta",
        type=float,
        default=0.01,
        help="Huber transition point delta (used when --loss huber). Paper: 0.01.",
    )
    p.add_argument(
        "--huber-delta-energy",
        type=float,
        default=None,
        help=(
            "Energy-specific Huber delta (used when --loss huber). Defaults to "
            "--huber-delta. The paper's δ_E=0.01 is a finetuning value; from "
            "scratch the per-atom energy error sits far above it (all in Huber's "
            "linear regime → starved gradient → energy plateaus). Set larger "
            "(e.g. 0.1) to keep early energy errors in the quadratic regime. "
            "Forces keep --huber-delta (conditional)."
        ),
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=0,
        help=(
            "If > 0, print running train loss/MSE/MAE every N steps WITHIN each "
            "epoch (primary rank only). Essential for from-scratch debugging: an "
            "epoch is ~67k steps, so per-epoch logging can't show where/when a "
            "divergence (NaN) begins. 0 = per-epoch logging only."
        ),
    )
    p.add_argument(
        "--virial-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on the virial-MSE term. 0 disables virial training "
            "(force-grad path still hits, just no virial loss). For materials "
            "datasets with stress labels (MPtraj, MatPES) a value of 1 is "
            "a reasonable starting point — virial is in eV like E, so use "
            "a weight comparable to --energy-weight."
        ),
    )
    p.add_argument("--max-frames", type=int, default=None, help="cap train frames (debug)")
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many optimizer steps (profiling / short benches). "
        "Does not change the LR schedule's total_steps.",
    )
    p.add_argument(
        "--profile-io",
        action="store_true",
        help="Log loader wait / H2D / step-dispatch timings to stdout and "
        "<output-dir>/io_profile.json (does not change numerics).",
    )
    p.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "Path to a ckpt_epoch_NNNN.eqx file to load chain weights from. "
            "Training resumes at epoch NNNN+1. Sibling `_opt.eqx` and "
            "`_ema.eqx` files (if present) restore the optimizer state "
            "(Adam moments, schedule step) and EMA params — full warm "
            "resume. Missing sibling files fall back to fresh Adam / EMA. "
            "log.jsonl is appended to, not overwritten."
        ),
    )
    # Apply training-card values as defaults (CLI args still win).
    if card_path is not None:
        tcard = _load_training_card(card_path)
        known_dests = {a.dest for a in p._actions}
        applicable = {k: v for k, v in tcard.items() if k in known_dests}
        unknown = set(tcard) - known_dests
        if unknown:
            print(f"WARNING: training-card has unknown keys (ignored): {sorted(unknown)}")
        p.set_defaults(**applicable)
    args = p.parse_args()

    # `train` is required either via CLI or training-card; check after merge.
    if not args.train and not args.multihead_spec:
        p.error("--train is required (provide via CLI or in --training-card)")

    return p, args, card_path
