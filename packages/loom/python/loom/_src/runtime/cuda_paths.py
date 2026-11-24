# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from jax import ffi
from jax._src.lib import xla_client

import loom._native as _engine

# Opt the recipe handler into XLA command buffers. XLA records FFI custom calls
# by CUDA stream capture, so this is only sound once every replay launches the
# same device work with no capture-illegal driver calls on the path. Off by
# default until the handler's capture path lands; SPLICE_LOOM_COMMAND_BUFFER=1
# turns it on for bring-up.
_COMMAND_BUFFER = os.environ.get("SPLICE_LOOM_COMMAND_BUFFER", "0") == "1"

# Older PJRT plugins reject the traits API even for DEFAULT. Omitting the
# keyword preserves the pre-traits registration path when capture is disabled.
_registration = {}
if _COMMAND_BUFFER:
    _registration["traits"] = xla_client.CustomCallTargetTraits.COMMAND_BUFFER_COMPATIBLE

ffi.register_ffi_target(
    "run_recipe",
    _engine.run_recipe_capsule(),
    platform="CUDA",
    **_registration,
)


def lib_dirs() -> list[str]:
    return _engine.lib_dirs()


def add_lib_dir(path: str) -> None:
    _engine.add_lib_dir(path)
