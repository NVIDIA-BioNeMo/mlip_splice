# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from typing import Any

_profile: ContextVar[dict[str, Any] | None] = ContextVar("loom_compile_profile", default=None)

# A supervising benchmark process points this at a file and tails the tab-
# separated events emit_event appends, enforcing the per-phase watchdog.
_WATCHDOG_PATH = "LOOM_WATCHDOG_PATH"


def emit_event(*fields: str) -> None:
    path = os.environ.get(_WATCHDOG_PATH)
    if path is None:
        return
    with open(path, "a") as events:
        print(*fields, sep="\t", file=events, flush=True)


@contextlib.contextmanager
def capture_compile_profile() -> Iterator[dict[str, Any]]:
    profile: dict[str, Any] = {}
    token = _profile.set(profile)
    try:
        yield profile
    finally:
        _profile.reset(token)


@contextlib.contextmanager
def phase(name: str) -> Iterator[None]:
    profile = _profile.get()
    emit_event("start", name)
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        if profile is not None:
            profile[name] = profile.get(name, 0.0) + elapsed_ms
        emit_event("end", name)


def record(phases: Mapping[str, float]) -> None:
    """Add durations into the capture, and state everything else once.

    A compile records into whichever capture is open, and a lowering runs many
    compiles, so durations accumulate across them. Facts about the machine do
    not: `compile_workers` describes one pool and `host_memory_bytes` one host,
    so summing them over 22 compiles would report 22 hosts. Durations are named
    by their unit, which is what separates the two.
    """
    profile = _profile.get()
    if profile is None:
        return
    for name, value in phases.items():
        value = float(value)
        if name.endswith("_ms"):
            profile[name] = profile.get(name, 0.0) + value
        else:
            profile[name] = value


def metadata(**values: Any) -> None:
    profile = _profile.get()
    if profile is not None:
        profile.update(values)
