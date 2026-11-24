# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context-local observation and strict substitution at compiler stage seams."""

from __future__ import annotations

import contextlib
import time
from collections import defaultdict, deque
from contextvars import ContextVar

from loom.compilation import (
    RecipeLoadFinished,
    RecipeLoadStarted,
    SourceCompilation,
    _json,
    _sha256,
    compiler_identity,
)

_observers: ContextVar[tuple] = ContextVar("loom_observers", default=())
_replay: ContextVar = ContextVar("loom_replay", default=None)


class ReplayError(ValueError):
    """A capture is incompatible, missing, or unused by the compiled workload."""


@contextlib.contextmanager
def observe(callback):
    """Observe source and recipe-registration events for compilations in this context.

    Callbacks run synchronously. Raising from a SourceCompilation callback stops
    before assembly, without caching a dummy executable. AOT recipe reuse is
    bypassed so every lowering is visible. Already compiled JAX executables are
    not recompiled: clear JAX caches or use a fresh process when capturing.
    Tuning reports can be enabled with ``loom.aot(debug_dump_dir=...)``.
    """
    if not callable(callback):
        raise TypeError("loom.observe expects a callback")
    token = _observers.set((*_observers.get(), callback))
    try:
        yield
    finally:
        _observers.reset(token)


@contextlib.contextmanager
def replay(artifacts):
    """Replay each source artifact once, matching the full semantic signature.

    Requires the same compiler build, JAX versions and GPU target. Source
    settings must match; realization policy may differ. Missing or unused
    captures fail explicitly. An already cached JAX program therefore fails
    rather than silently ignoring captures. Nested replay contexts are rejected.
    """
    if _replay.get() is not None:
        raise ReplayError("nested replay contexts are not supported")
    identity = compiler_identity()
    pending = defaultdict(deque)
    for artifact in artifacts:
        if not isinstance(artifact, SourceCompilation):
            raise TypeError("loom.replay expects SourceCompilation records")
        if artifact.identity != identity:
            raise ReplayError("capture requires the same compiler build and GPU target")
        pending[artifact.signature].append(artifact)
    token = _replay.set(pending)
    try:
        yield
        remaining = sum(map(len, pending.values()))
        if remaining:
            raise ReplayError(f"{remaining} unused source compilation capture(s)")
    finally:
        _replay.reset(token)


def active() -> bool:
    return bool(_observers.get()) or _replay.get() is not None


def _emit(event):
    for callback in _observers.get():
        callback(event)


def compile_sources(compiler, *args, **kwargs):
    if not active():
        return compiler(*args, **kwargs)
    # The native XIR object is represented by its text, never by its address.
    name, module, xir, *contract = args
    if module.text() != xir:
        raise ValueError("XIR object and source text disagree")
    semantic = {
        key: value
        for key, value in kwargs.items()
        if key not in {"diagnostics", "dump_ptx", "record_structure"}
    }
    signature = _sha256(_json((name, xir, contract, semantic)).encode())
    pending = _replay.get()
    if pending is None:
        diagnostics = kwargs.get("diagnostics")
        if diagnostics is None:
            diagnostics = {}
            kwargs["diagnostics"] = diagnostics
        identity = compiler_identity()
        result = compiler(*args, **kwargs)
        artifact = SourceCompilation._from_result(identity, signature, xir, result, diagnostics)
    else:
        matches = pending.get(signature)
        if not matches:
            raise ReplayError("no capture matches this source compilation signature")
        artifact = matches.popleft()
        if kwargs.get("diagnostics") is not None:
            kwargs["diagnostics"].update(artifact.diagnostics)
        result = artifact.recipe, artifact.slots, artifact.profile
    _emit(artifact)
    return result


def load_recipe(loader, recipe, **kwargs):
    if not _observers.get():
        return loader(recipe, **kwargs)
    digest = _sha256(bytes(recipe))
    report_path = kwargs.get("tune_report_to")
    _emit(RecipeLoadStarted(digest, report_path))
    started_unix_ns = time.time_ns()
    started = time.perf_counter()
    identity, workspace = loader(recipe, **kwargs)
    elapsed = time.perf_counter() - started
    _emit(
        RecipeLoadFinished(
            digest, report_path, elapsed, started_unix_ns, int(workspace), bytes(identity)
        )
    )
    return identity, workspace
