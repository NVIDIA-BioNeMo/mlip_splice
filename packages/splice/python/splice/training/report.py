# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training provenance, checkpoints, and compile reports."""

import json
import os
import time
from pathlib import Path

# Best-effort: keep XLA from preallocating huge buffers, helps small-model multi-run.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx

_LOOM_PHASES = (
    "jax_trace_ms",
    "stablehlo_ms",
    "xir_lowering_ms",
    "simplify_ms",
    "plan_ms",
    "render_ptx_ms",
    "state_requests_ms",
    "build_recipe_ms",
)

# The three disjoint spans one first step passes through. Loom compiles while
# JAX lowers; XLA compiles what Loom left; kiln assembles and tunes on first
# execution. Measured separately they partition the step's wall clock, which no
# sum of per-phase work can do.
_STAGES = ("lower_ms", "xla_compile_ms", "first_execute_ms")


def run_provenance() -> dict:
    """Record which code produced a run, so a results dir can be traced back later.

    Written once to ``run_meta.json`` next to the other input snapshots. Entirely
    best-effort: a missing git, a tarball checkout or a detached HEAD degrade to
    ``None`` fields rather than failing a training job.
    """
    import socket
    import subprocess
    import sys

    repo = next(
        (p for p in Path(__file__).resolve().parents if (p / ".git").exists()),
        Path(__file__).resolve().parent,
    )

    def git(*a):
        try:
            r = subprocess.run(
                ["git", "-C", str(repo), *a], capture_output=True, text=True, timeout=10
            )
            return r.stdout.strip() or None if r.returncode == 0 else None
        except Exception:
            return None

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_describe": git("describe", "--always", "--dirty", "--tags"),
        # uncommitted edits at launch -> the commit alone does not reproduce the run
        "git_dirty": bool(git("status", "--porcelain")),
        "repo": str(repo),
        "hostname": socket.gethostname(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "argv": sys.argv,
    }


def _save_chain(chain, path: Path):
    """Serialize the chain via eqx (binary)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, chain)


def _kiln_timings(directory: Path) -> dict:
    """Sum every kiln tune report in `directory`, keeping work apart from time.

    One report is written per realized recipe and a step realizes several, so
    the sums are what the step paid rather than what any one recipe cost.
    `total` and `tuning` are elapsed time inside a `realize` call and the calls
    do not overlap, so summing them stays wall clock. `compile_wall` and
    `compile_production_cost` are already summed over concurrent compile tasks,
    so they are work: `compile_production_cost` counts historical work returned
    by cache hits and may exceed this run entirely.
    """
    elapsed_keys = ("total", "tuning", "artifact", "artifact_cache", "compile_blocked")
    work_keys = ("compile_wall", "compile_production_cost", "resource")
    summed = dict.fromkeys(elapsed_keys + work_keys, 0.0)
    recipes = 0
    candidates = 0
    for path in sorted(directory.glob("tune_report_*.json")):
        try:
            with open(path) as handle:
                report = json.load(handle)
        except (OSError, ValueError):
            continue
        timing = (report.get("realization") or {}).get("timing_ms") or {}
        recipes += 1
        candidates += int(report.get("num_timed") or 0)
        for key in summed:
            summed[key] += float(timing.get(key) or 0.0)
    summed["recipes"] = recipes
    summed["candidates_timed"] = candidates
    return summed


def _write_compile_profile(directory: Path, report: dict) -> None:
    """Write the decomposition as JSON, and print the table it states.

    Two tables, because the numbers are two kinds. The first partitions the
    step's wall clock into spans that do not overlap and do sum to it. The
    second states, inside those spans, how much work each layer's phases did —
    work that exceeds its span exactly as far as its threads were busy.
    """
    loom = report.get("loom") or {}
    kiln = report.get("kiln") or {}
    stages = report.get("stages") or {}
    wall_ms = float(report.get("wall_ms") or 0.0)
    execute_ms = float(stages.get("first_execute_ms") or 0.0)

    # kiln realizes inside the first execution, so its elapsed total is a part
    # of that span and the rest of the span is the step's own device work.
    kiln_ms = float(kiln.get("total") or 0.0)
    spans = [(name, float(stages.get(name) or 0.0)) for name in _STAGES]
    spans.append(("  kiln_realize_ms", kiln_ms))
    spans.append(("  kiln_tuning_ms", float(kiln.get("tuning") or 0.0)))
    spans.append(("  step_device_ms", execute_ms - kiln_ms))
    spans.append(("wall_ms", wall_ms))

    workers = float(loom.get("compile_workers") or 0.0)
    loom_work_ms = sum(float(loom.get(name) or 0.0) for name in _LOOM_PHASES)
    work = [(name, float(loom.get(name) or 0.0)) for name in _LOOM_PHASES]
    work.append(("loom_work_ms", loom_work_ms))
    work.extend(
        (f"kiln_{name}_work_ms", float(kiln.get(name) or 0.0))
        for name in ("compile_wall", "compile_production_cost")
    )

    report["spans"] = dict(spans)
    report["work"] = dict(work)
    path = directory / "compile_profile.json"
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    shapes = report.get("shapes") or {}
    print(
        f"[profile] {shapes.get('atoms')} atoms / {shapes.get('edges')} edges, "
        f"{kiln.get('recipes')} recipes realized, "
        f"{kiln.get('candidates_timed')} candidates timed",
        flush=True,
    )
    print("[profile] wall clock, disjoint spans summing to the step:", flush=True)
    for name, value in spans:
        share = f"{100.0 * value / wall_ms:5.1f}%" if wall_ms > 0 else "    -"
        print(f"[profile]   {name:<26} {value:10.1f} ms  {share}", flush=True)
    lower_ms = float(stages.get("lower_ms") or 0.0)
    speedup = loom_work_ms / lower_ms if lower_ms > 0 else 0.0
    print(
        f"[profile] work inside those spans, across {workers:.0f} compile "
        f"workers (loom work / lower span = {speedup:.1f}x):",
        flush=True,
    )
    for name, value in work:
        share = f"{100.0 * value / loom_work_ms:5.1f}%" if loom_work_ms > 0 else "    -"
        print(f"[profile]   {name:<26} {value:10.1f} ms  {share}", flush=True)
    print(f"[profile] written to {path}", flush=True)
