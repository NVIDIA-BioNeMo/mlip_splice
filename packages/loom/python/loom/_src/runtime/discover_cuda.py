# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sysconfig

_LIB_SUBDIRS = (
    "nvidia/cu13/lib",
    "nvidia/cu12/lib",
    "nvidia/nvjitlink/lib",
    "nvidia/cuda_runtime/lib",
    "cutensor/lib",
)


def autodiscover() -> None:
    import loom._native as _engine

    for site in _site_packages_roots():
        for sub in _LIB_SUBDIRS:
            p = os.path.join(site, sub)
            if os.path.isdir(p):
                _engine.add_lib_dir(p)


def _site_packages_roots() -> list[str]:
    paths = sysconfig.get_paths()
    candidates = [paths.get("purelib"), paths.get("platlib")]
    try:
        import site

        candidates.append(site.getusersitepackages())
    except Exception:
        pass
    out: list[str] = []
    seen: set[str] = set()
    for p in candidates:
        if p and p not in seen and os.path.isdir(p):
            seen.add(p)
            out.append(p)
    return out
