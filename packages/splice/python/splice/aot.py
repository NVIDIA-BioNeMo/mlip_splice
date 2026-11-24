# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A bounded, cached compile context for inference and MD.

Loom autotunes without a budget unless it is given one. That is the right
default for training, where a single compile is amortised over days of steps,
and the wrong one everywhere else: `packages/splice/scripts/train.py` was the only caller that
ever set a budget, so every MD run and every ASE calculator autotuned uncapped.

Measured on four solvated-protein systems (35-42k atoms) on cold kernels, the
difference between uncapped and the defaults here was 98 minutes of pure CPU
compile against 6 -- with nothing written to the log in the meantime, so it
reads as a hang rather than as progress.

    from splice.aot import aot_session

    with aot_session(identity_files=["arch.yaml"], cache_path="aot.pkl"):
        trace, state = run_md(...)

`run_md` opens one of these itself, so the common path needs no change.

Budgets are explicit arguments: `search_budget_ms` bounds recipe search by
elapsed wall-clock milliseconds. `shortlist_budget_ms` bounds source shortlisting
by predicted realization milliseconds: max(assembly cost / workers, serial
tuning cost).
Omitting them uses the bounded defaults; passing None makes that stage unlimited.
Prefer the shortlist budget: assembly is where a first step spends nearly all
of its time, and a smaller recipe was measured to change compile time sixfold
and the resulting kernel not at all.
"""

import hashlib
import os
import pickle
from contextlib import contextmanager

# What we ship as defaults. Chosen because they were measured to give the same
# kernels as an uncapped search on this model family while bounding the compile:
# raising them buys nothing here, and lowering the shortlist budget starts to matter.
DEFAULT_SEARCH_BUDGET_MS = 250.0
DEFAULT_SHORTLIST_BUDGET_MS = 2000.0


def workload_identity(identity_files=(), extra=""):
    """A digest naming the compiled workload, for keying a persistent cache.

    Must cover everything that changes the program: the architecture, the card
    it resolves against, and any shape the caller pins. A cache reused under a
    stale identity would hand a changed model the previous model's recipes,
    which is why this hashes file *contents* rather than paths.
    """
    h = hashlib.sha256()
    for path in identity_files:
        if path is None:  # e.g. a resolved arch needs no separate card
            continue
        with open(path, "rb") as f:
            h.update(f.read())
    h.update(str(extra).encode())
    return h.hexdigest()


@contextmanager
def aot_session(
    *,
    identity_files=(),
    extra="",
    cache_path=None,
    search_budget_ms=DEFAULT_SEARCH_BUDGET_MS,
    shortlist_budget_ms=DEFAULT_SHORTLIST_BUDGET_MS,
    verbose=True,
):
    """Compile inside a bounded Loom context, optionally against a saved cache.

    Parameters
    ----------
    identity_files, extra : what the cache is keyed on — see `workload_identity`
    cache_path : str | None
        Where to persist realized recipes. None keeps them in-process, which
        still bounds the compile but re-pays it next run.
    search_budget_ms : float | None
        Recipe-search budget in elapsed wall-clock milliseconds, defaulting to
        250. Explicit None makes search unlimited.
    shortlist_budget_ms : float | None
        Source-shortlist budget in predicted realization milliseconds:
        max(assembly cost / workers, serial tuning cost), defaulting to 2000.
        Explicit None makes shortlisting unlimited.

    A corrupt or unreadable cache is ignored rather than fatal: it is an
    optimisation, and failing a run over it would trade minutes for hours.
    """
    import loom

    cache, identity = None, None
    if cache_path:
        cache = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                if verbose:
                    print(f"[aot] {len(cache)} cached recipes from {cache_path}", flush=True)
            except Exception as exc:  # a bad cache must not block a run
                print(f"[aot] ignoring unreadable cache ({exc})", flush=True)
                cache = {}
        identity = workload_identity(identity_files, extra)

    if verbose:
        search = "unlimited" if search_budget_ms is None else f"{search_budget_ms:g}ms"
        shortlist = "unlimited" if shortlist_budget_ms is None else f"{shortlist_budget_ms:g}ms"
        print(f"[aot] search={search} shortlist={shortlist}", flush=True)

    with loom.aot(
        loom.Compile(search_budget_ms=search_budget_ms, shortlist_budget_ms=shortlist_budget_ms),
        cache=cache,
        workload_identity=identity,
    ):
        yield

    if cache_path:
        try:
            tmp = cache_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(cache, f)
            os.replace(tmp, cache_path)  # atomic: never leave a half-written cache
            if verbose:
                print(f"[aot] saved {len(cache)} recipes -> {cache_path}", flush=True)
        except Exception as exc:
            print(f"[aot] could not save cache: {exc}", flush=True)
