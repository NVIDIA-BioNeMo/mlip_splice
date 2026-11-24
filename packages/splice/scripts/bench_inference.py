# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone inference-throughput benchmark for a trained SPLICE chain.

Sweep batch sizes (doubling from --batch-start) until OOM or --batch-max;
report µs/atom + atoms/sec + peak GB at each size. Designed to give a
best-case inference-speed estimate on realistic data with minimal padding.

Typical use:
    python packages/splice/scripts/bench_inference.py \\
        --arch examples/configs/fk_cw64_tr256.yaml \\
        --card cards/matpes_100k.yaml \\
        --ckpt /path/to/ckpt_epoch_0200_ema.eqx \\
        --data /path/to/test_id.xyz \\
        --batch-start 8 --batch-max 1024 \\
        --output bench_results.json

The "inference call" times one ``model_call`` with full autograd for forces
(and optional cell-grad for stress). That matches the work an MLIP MD step
does. Padding uses small bucket bases by default (``--base-atoms 64``,
``--base-edges 512``) so wasted compute stays low; the JIT caches one
specialisation per unique padded shape and the first warmup pass covers
the compile cost.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Push JAX's BFC allocator to use ~95% of device memory instead of the ~80%
# default. The bench is about finding the real OOM boundary, so leaving 16 GB
# of an 80 GB A100 on the table makes the headline batch size artificially
# small. Must be set BEFORE jax is imported below.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import numpy as np
from splice.config import build_chain
from splice.data import batched_iter, load_extxyz
from splice.forward import model_call
from splice.model_card import card_offsets, load_card, resolve_arch

# When this script is run under ``tee`` (or any pipe) python defaults to
# block-buffering stdout, so per-batch-size progress lines sit in a 4KB
# buffer for minutes while JIT compiles run. Force line buffering so the
# log file updates in real time. (After imports to satisfy E402; runs
# before any print.)
sys.stdout.reconfigure(line_buffering=True)

_OOM_TOKENS = ("resource_exhausted", "out of memory", "oom")


def _is_oom(exc: BaseException) -> bool:
    s = str(exc).lower()
    return any(t in s for t in _OOM_TOKENS)


def _real_atom_count(batch: dict) -> int:
    inde = np.asarray(batch["inde"])
    return int(np.sum(inde != batch["dummy_graph_id"]))


def _strip_targets(batch: dict) -> dict:
    """Drop target/mask/aux keys that aren't model inputs.

    Trims the PyTree the JIT sees so cache keys stay tight.
    """
    keep = {}
    for k, v in batch.items():
        if k.startswith("target_") or k.startswith("mask_") or k == "dummy_graph_id":
            continue
        keep[k] = v
    return keep


def bench_one_size(
    chain,
    frames,
    *,
    num_species: int,
    batch_size: int,
    base_atoms: int,
    base_edges: int,
    energy_offsets,
    compute_virial: bool,
    n_warmup_calls: int,
    n_measure_calls: int,
) -> dict | None:
    """Time inference at a fixed batch size + fixed shape.

    Builds **one** padded batch from the first ``batch_size`` frames, then
    calls the model on it ``n_warmup_calls + n_measure_calls`` times. Single
    shape → one JIT compile per batch_size → fast warmup. This measures the
    steady-state kernel throughput at that batch size, which is what the
    Pareto-speed comparison cares about.
    """
    if batch_size > len(frames):
        return None
    # Take the first batch_size frames and assemble one padded batch.
    # Use batched_iter with that subset so we reuse the existing padding logic
    # (which routes targets, picks bucket sizes, etc.).
    batches = list(
        batched_iter(
            frames[:batch_size],
            batch_size=batch_size,
            base_atoms=base_atoms,
            base_edges=base_edges,
            shuffle=False,
            drop_last=True,
            energy_offsets=energy_offsets,
        )
    )
    if not batches:
        return None
    assert len(batches) == 1, "expected exactly one batch from a fixed-batch bench"
    batch = _strip_targets(batches[0])
    real_atoms = _real_atom_count(batches[0])

    @eqx.filter_jit
    def call(chain, batch):
        return model_call(chain, batch, num_species=num_species, compute_virial=compute_virial)

    # Warmup: triggers the single JIT compile and warms the kernels. Block
    # afterwards so the compile time falls entirely outside the measurement.
    for _ in range(n_warmup_calls):
        out = call(chain, batch)
    jax.block_until_ready(out)

    # Measure
    start = time.perf_counter()
    for _ in range(n_measure_calls):
        out = call(chain, batch)
    jax.block_until_ready(out)
    elapsed = time.perf_counter() - start

    total_real_atoms = real_atoms * n_measure_calls

    mem = jax.devices()[0].memory_stats()
    peak_gb = mem.get("peak_bytes_in_use", 0) / 1024**3

    return {
        "batch_size": batch_size,
        "n_calls": n_measure_calls,
        "real_atoms_per_batch": real_atoms,
        "ms_per_batch": elapsed / n_measure_calls * 1000,
        "us_per_atom": elapsed / total_real_atoms * 1e6,
        "atoms_per_sec": total_real_atoms / elapsed,
        "peak_gb": peak_gb,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arch", required=True, help="arch yaml")
    p.add_argument("--card", required=True, help="card yaml")
    p.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Override the card cutoff (Å) for a same-dataset Rc sweep; picks "
        "avg_num_neighbors from the card's neighbors_by_cutoff table. Mirrors "
        "train.py --cutoff. Used to bench the Rc4 free-stack candidate.",
    )
    p.add_argument(
        "--ckpt",
        default=None,
        help="chain checkpoint (.eqx). If omitted, the chain is initialised from "
        "``build_chain`` (random weights) — fine when only inference speed is "
        "wanted; accuracy will obviously be garbage.",
    )
    p.add_argument("--data", required=True, help="extxyz file of representative frames")
    p.add_argument("--n-frames", type=int, default=128, help="frames to load from --data")
    p.add_argument(
        "--batch-start",
        type=int,
        default=8,
        help="smallest batch size in the sweep (pick based on dataset/model: small "
        "models can start at 1, big models should start higher to skip irrelevant points)",
    )
    p.add_argument("--batch-max", type=int, default=2048)
    p.add_argument(
        "--base-atoms",
        type=int,
        default=64,
        help="bucket base for atoms (small=less padding waste, more JIT compiles)",
    )
    p.add_argument("--base-edges", type=int, default=512, help="bucket base for edges")
    p.add_argument(
        "--n-warmup-calls",
        type=int,
        default=3,
        help="model calls before timing starts (covers JIT compile + kernel warmup)",
    )
    p.add_argument(
        "--n-measure-calls",
        type=int,
        default=10,
        help="model calls timed for the throughput average",
    )
    p.add_argument("--no-stress", action="store_true", help="skip virial backward")
    p.add_argument(
        "--output",
        default=None,
        help="optional .json sidecar to write results (for downstream aggregation)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    compute_virial = not args.no_stress

    card = load_card(args.card)
    if args.cutoff is not None:
        from splice.model_card import avg_neighbors_for_cutoff

        card["cutoff"] = float(args.cutoff)
        ann = avg_neighbors_for_cutoff(card, args.cutoff)
        if ann is not None:
            card["avg_num_neighbors"] = ann
            print(f"--cutoff {args.cutoff}: avg_num_neighbors={ann} (card table)")
        else:
            print(f"WARNING: --cutoff {args.cutoff} not in card neighbors_by_cutoff table")
    species_map = card["species"]
    cutoff = float(card["cutoff"])
    num_species = len(species_map)
    cfg = resolve_arch(args.arch, card)
    chain = build_chain(cfg, key=jax.random.key(args.seed))
    if args.ckpt is not None:
        chain = eqx.tree_deserialise_leaves(args.ckpt, chain)
    n_params = sum(int(x.size) for x in jax.tree.leaves(eqx.filter(chain, eqx.is_array)))

    print(f"arch  : {Path(args.arch).name}")
    print(f"card  : {Path(args.card).name}")
    print(f"ckpt  : {Path(args.ckpt).name if args.ckpt else '<random init>'}")
    print(f"params: {n_params:,}")
    print(f"stress: {compute_virial}")

    loaded = load_extxyz(args.data, species_map, cutoff=cutoff)
    # Cycle frames if --n-frames > available, so a smaller dataset can still
    # drive arbitrarily large batches for OOM-probing. Speed bench cares about
    # realistic molecule sizes, not specific frames — duplicates are fine.
    if args.n_frames <= len(loaded):
        frames = loaded[: args.n_frames]
        cycle_note = ""
    else:
        repeats = (args.n_frames + len(loaded) - 1) // len(loaded)
        frames = (loaded * repeats)[: args.n_frames]
        cycle_note = f" (cycled {repeats}x from {len(loaded)} unique)"
    print(f"frames: {len(frames)} from {Path(args.data).name}{cycle_note}")

    offsets = card_offsets(card)

    results: list[dict] = []
    B = args.batch_start
    while args.batch_max >= B and len(frames) >= B:
        print(f"\n=== batch_size={B} ===")
        try:
            r = bench_one_size(
                chain,
                frames,
                num_species=num_species,
                batch_size=B,
                base_atoms=args.base_atoms,
                base_edges=args.base_edges,
                energy_offsets=offsets,
                compute_virial=compute_virial,
                n_warmup_calls=args.n_warmup_calls,
                n_measure_calls=args.n_measure_calls,
            )
            if r is None:
                print("  no full batches at this size — skip")
                break
            print(
                f"  n_calls={r['n_calls']:>3}  "
                f"real_atoms={r['real_atoms_per_batch']:>5}  "
                f"ms/batch={r['ms_per_batch']:>7.2f}  "
                f"µs/atom={r['us_per_atom']:>7.3f}  "
                f"atoms/s={r['atoms_per_sec']:.3e}  "
                f"peak_GB={r['peak_gb']:.2f}"
            )
            results.append(r)
        except Exception as e:
            if _is_oom(e):
                print(f"  OOM at batch_size={B} — stopping sweep")
                results.append({"batch_size": B, "status": "OOM"})
                break
            raise
        B *= 2

    print("\n" + "=" * 72)
    print(f"{'batch':>6} {'ms/batch':>10} {'µs/atom':>10} {'atoms/s':>14} {'peak_GB':>9}")
    for r in results:
        if "status" in r:
            print(f"{r['batch_size']:>6}  {r['status']}")
        else:
            print(
                f"{r['batch_size']:>6} {r['ms_per_batch']:>10.2f} "
                f"{r['us_per_atom']:>10.3f} {r['atoms_per_sec']:>14.3e} "
                f"{r['peak_gb']:>9.2f}"
            )

    ok = [r for r in results if "status" not in r]
    if ok:
        best = min(ok, key=lambda r: r["us_per_atom"])
        print(
            f"\nbest: batch_size={best['batch_size']}  "
            f"µs/atom={best['us_per_atom']:.3f}  "
            f"atoms/s={best['atoms_per_sec']:.3e}  "
            f"peak_GB={best['peak_gb']:.2f}"
        )

    if args.output:
        out = {
            "arch": str(args.arch),
            "card": str(args.card),
            "ckpt": str(args.ckpt),
            "data": str(args.data),
            "n_frames": len(frames),
            "n_params": n_params,
            "compute_virial": compute_virial,
            "base_atoms": args.base_atoms,
            "base_edges": args.base_edges,
            "sweep": results,
            "best": min(ok, key=lambda r: r["us_per_atom"]) if ok else None,
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
