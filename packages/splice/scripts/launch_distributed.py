# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch data-parallel training: one process per GPU on a single node.

Spawns ``--nproc`` copies of ``train.py``, each pinned to one GPU via
``CUDA_VISIBLE_DEVICES`` (so each sees its GPU as device 0) and wired into one
``jax.distributed`` group via a shared coordinator. All arguments after ``--``
are forwarded verbatim to ``train.py`` (use ``--fixed-atoms``/``--fixed-edges``;
``--batch-size`` is per-GPU).

Example (2 GPUs):
    python packages/splice/scripts/launch_distributed.py --nproc 2 -- \
        --data-format packed --train data/packed_4M --eval data/packed_val \
        --arch examples/configs/fk_cw64_tr256.yaml --card examples/card.yaml \
        --batch-size 64 --fixed-atoms 4096 --fixed-edges 131072 \
        --lr-scale-by-devices --output-dir runs/omol_dp

Process 0's output goes to the console; other ranks write ``<out>/rank_N.log``
(``--log-dir`` to override). On a multi-node cluster use ``srun`` instead (it
sets SLURM_NTASKS/SLURM_PROCID; export SPLICE_COORDINATOR=<rank0_host>:port
and pin one GPU per task) — train.py's init reads the SLURM_* vars directly.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nproc", type=int, default=None, help="processes/GPUs (default: all visible)")
    ap.add_argument("--gpus", type=str, default=None, help="comma-sep GPU ids (default 0..n-1)")
    ap.add_argument("--coordinator-port", type=int, default=12355)
    ap.add_argument("--log-dir", type=str, default=None, help="dir for non-primary rank logs")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- then train.py args")
    args = ap.parse_args()

    train_args = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    if not train_args:
        ap.error("no train.py arguments given (put them after `--`)")

    if args.gpus:
        gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    else:
        n = args.nproc or _visible_gpu_count()
        gpu_ids = [str(i) for i in range(n)]
    nproc = len(gpu_ids)
    if nproc < 1:
        ap.error("need at least one GPU")

    log_dir = Path(args.log_dir) if args.log_dir else _infer_out_dir(train_args) or Path(".")
    log_dir.mkdir(parents=True, exist_ok=True)
    train_py = str(Path(__file__).with_name("train.py"))
    coord = f"localhost:{args.coordinator_port}"
    print(f"launching {nproc} processes on GPUs {gpu_ids}, coordinator {coord}", flush=True)

    procs = []
    logs = {}
    for pid, gpu in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["SPLICE_NUM_PROCESSES"] = str(nproc)
        env["SPLICE_PROCESS_ID"] = str(pid)
        env["SPLICE_COORDINATOR"] = coord
        cmd = [sys.executable, train_py, *train_args]
        if pid == 0:
            procs.append(subprocess.Popen(cmd, env=env))  # inherit stdout
        else:
            fh = (log_dir / f"rank_{pid}.log").open("w")
            logs[pid] = fh
            procs.append(subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT))

    # Wait for all; if any rank dies, kill the rest so we don't hang on a
    # half-collective and surface the failing rank's log tail.
    failed = None
    try:
        while procs:
            for i, pr in enumerate(list(procs)):
                rc = pr.poll()
                if rc is None:
                    continue
                procs.remove(pr)
                if rc != 0 and failed is None:
                    failed = (i, rc)
                    raise RuntimeError(f"a rank exited with code {rc}")
            _wait_a_bit(procs)
    except (RuntimeError, KeyboardInterrupt) as e:
        print(f"\n{e}; terminating remaining ranks", flush=True)
        for pr in procs:
            pr.terminate()
        for pr in procs:
            try:
                pr.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pr.kill()
    finally:
        for fh in logs.values():
            fh.close()

    if failed is not None:
        rank, rc = failed
        tail = ""
        if rank in logs:
            tail = "\n".join((log_dir / f"rank_{rank}.log").read_text().splitlines()[-25:])
        print(f"\n=== rank tail (exit {rc}) ===\n{tail}", flush=True)
        sys.exit(rc)
    print("all ranks finished cleanly", flush=True)


def _visible_gpu_count() -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return len([x for x in cvd.split(",") if x.strip() != ""])
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except Exception:
        return 1


def _infer_out_dir(train_args: list[str]) -> Path | None:
    for i, a in enumerate(train_args):
        if a == "--output-dir" and i + 1 < len(train_args):
            return Path(train_args[i + 1])
        if a.startswith("--output-dir="):
            return Path(a.split("=", 1)[1])
    return None


def _wait_a_bit(procs) -> None:
    """Block briefly on the first live process so the poll loop isn't a spin."""
    if procs:
        with contextlib.suppress(subprocess.TimeoutExpired):
            procs[0].wait(timeout=2)


if __name__ == "__main__":
    main()
