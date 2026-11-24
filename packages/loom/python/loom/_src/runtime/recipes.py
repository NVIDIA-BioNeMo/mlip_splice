# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import hashlib
import itertools
import os
import pathlib
from collections.abc import Iterator, Sequence
from contextvars import ContextVar
from typing import NamedTuple

import jax
import numpy as np
from jax import ffi

import loom._native as _engine
from loom._src.primitive.options import Compile
from loom._src.runtime import cuda_paths as _cuda_paths  # noqa: F401
from loom._src.runtime import instrumentation


class _AotState(NamedTuple):
    cache: dict
    workload_identity: bytes | None
    options: tuple[Compile, ...]
    pending: list
    debug_dump_dir: str | None
    debug_seq: Iterator[int]


class _Capture(NamedTuple):
    key: str
    companions: tuple


_aot: ContextVar = ContextVar("recipe_compilation", default=None)


def _with_executable(recipe, executable):
    inputs = getattr(recipe, "inputs", None)
    outputs = getattr(recipe, "outputs", None)
    if inputs is None or outputs is None:
        return executable
    return type(recipe)(executable, inputs, outputs)


@contextlib.contextmanager
def aot(
    compile_options: Compile | None = None,
    *,
    cache: dict | None = None,
    workload_identity: str | bytes | None = None,
    debug_dump_dir: str | None = None,
):
    """Compile and realize with one shared `Compile` configuration.

    ``with loom.aot(loom.Compile(shortlist_budget_ms=600_000), cache=cache,
    workload_identity="my-workload"): ...``

    Each omitted field inherits from the enclosing context, then compiler
    defaults. Per-call settings win; explicit ``None`` disables nullable limits.
    Budget schedules use the consuming program's transpose-rule depth.
    ``shortlist_budget_ms`` limits predicted realization cost, not runtime.
    ``scheduling_time_limit_ms`` applies only when Kiln opens tuning work.

    A reusable ``cache`` requires a stable ``workload_identity`` because tuning
    depends on the input distribution. Its key includes source settings and
    runtime policy, so a differently tuned result is not reused by mistake.
    Settings take effect on compilation, not on already-compiled JAX programs.
    """
    options = Compile() if compile_options is None else compile_options
    if not isinstance(options, Compile):
        raise TypeError("loom.aot expects a loom.Compile; pass cache by keyword")
    parent = _aot.get()
    persistent = cache is not None
    if persistent and workload_identity is None:
        raise ValueError("a reusable AOT cache requires workload_identity")
    if cache is None:
        cache = {}
    if isinstance(workload_identity, str):
        workload_identity = workload_identity.encode()
    elif workload_identity is not None and not isinstance(workload_identity, bytes):
        raise TypeError("workload_identity must be str or bytes")
    state = _AotState(
        cache,
        workload_identity,
        (options, *(parent.options if parent is not None else ())),
        [],
        debug_dump_dir,
        itertools.count(),
    )
    token = _aot.set(state)
    try:
        yield cache
    finally:
        try:
            for key, companions, identity, recipe in state.pending:
                tuned = _engine.tuned_recipe_bytes(identity)
                if tuned is not None:
                    cache[key] = (_with_executable(recipe, tuned), companions)
        finally:
            _aot.reset(token)


def debug_dump_path(prefix: str) -> str | None:
    state = _aot.get()
    if state is None or state.debug_dump_dir is None:
        return None
    os.makedirs(state.debug_dump_dir, exist_ok=True)
    return os.path.join(state.debug_dump_dir, f"{prefix}_{next(state.debug_seq):03d}.json")


def _device_fingerprint() -> tuple:
    device = jax.devices("gpu")[0]
    so = pathlib.Path(_engine.__file__).stat()
    return (device.device_kind, device.compute_capability, so.st_size, so.st_mtime_ns)


def resolve_compile_options(options: Compile | None = None) -> Compile:
    """Resolve per-call, inner-to-outer AOT, and default configuration layers."""
    options = Compile() if options is None else options
    if not isinstance(options, Compile):
        raise TypeError("loom configuration must be a loom.Compile")
    state = _aot.get()
    return options.resolve(*(state.options if state is not None else ()))


def compile_recipe(
    compile_to_bytes,
    /,
    *args,
    cache_status: dict | None = None,
    compile_options: Compile | None = None,
    **kwargs,
) -> tuple:
    def status(cache_hit, source):
        if cache_status is not None:
            cache_status.update(cache_hit=cache_hit, cache_source=source)

    options = resolve_compile_options(compile_options)
    source_kwargs = options.source_kwargs()
    state = _aot.get()
    if state is None or instrumentation.active():
        status(False, "instrumented" if instrumentation.active() else "disabled")
        result = instrumentation.compile_sources(compile_to_bytes, *args, **kwargs, **source_kwargs)
        recipe_bytes, *companions = result if isinstance(result, tuple) else (result,)
        return (recipe_bytes, *companions, None)

    semantic_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"diagnostics", "dump_ptx", "record_structure"}
    }
    material = repr(
        (
            compile_to_bytes.__module__,
            compile_to_bytes.__qualname__,
            args,
            sorted(semantic_kwargs.items()),
            sorted(source_kwargs.items()),
            options.scheduling_time_limit_ms,
            _device_fingerprint(),
            state.workload_identity,
        )
    )
    key = hashlib.sha256(material.encode()).hexdigest()[:32]

    hit = state.cache.get(key)
    if hit is not None:
        status(True, "cache")
        recipe_bytes, companions = hit
        return (recipe_bytes, *companions, None)

    # A re-lowering inside the same aot context (e.g. a second jit trace of the
    # same computation) reuses the pending recipe's tuned winner instead of
    # re-forging the full candidate set.
    for pending_key, companions, identity, recipe in state.pending:
        if pending_key == key:
            tuned = _engine.tuned_recipe_bytes(identity)
            if tuned is not None:
                tuned = _with_executable(recipe, tuned)
                status(True, "pending")
                state.cache[key] = (tuned, companions)
                return (tuned, *companions, None)
            break

    result = instrumentation.compile_sources(compile_to_bytes, *args, **kwargs, **source_kwargs)
    status(False, "compiled")
    recipe_bytes, *companions = result if isinstance(result, tuple) else (result,)
    return (recipe_bytes, *companions, _Capture(key, tuple(companions)))


def run_recipe(
    recipe_bytes: object,
    inputs: Sequence[jax.Array],
    out_specs: Sequence[jax.ShapeDtypeStruct],
    *,
    compile_options: Compile | None = None,
    capture: _Capture | None = None,
    vmap_method: str | None = None,
    aliases: dict[int, int] | None = None,
) -> list[jax.Array]:
    if aliases is None:
        aliases = getattr(recipe_bytes, "aliases", None)
    state = _aot.get()
    identity, workspace_nbytes = instrumentation.load_recipe(
        _engine.load_recipe,
        recipe_bytes,
        tune_report_to=debug_dump_path("tune_report"),
        scheduling_time_limit_ms=resolve_compile_options(compile_options).scheduling_time_limit_ms,
    )
    if capture is not None:
        if state is not None:
            state.pending.append((capture.key, capture.companions, identity, recipe_bytes))
    workspace = jax.ShapeDtypeStruct((int(workspace_nbytes),), np.uint8)
    output_aliases = {
        input_index: output_index + 1 for input_index, output_index in (aliases or {}).items()
    }
    outs = ffi.ffi_call(
        "run_recipe",
        [workspace, *out_specs],
        vmap_method=vmap_method,
        input_output_aliases=output_aliases,
    )(*inputs, recipe=np.frombuffer(identity, dtype=np.uint8))
    return list(outs[1:])
