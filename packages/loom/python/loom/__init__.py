# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile callable JAX lane programs into fused GPU kernels."""

from __future__ import annotations

from loom import recipe
from loom._src.runtime.discover_cuda import autodiscover as _autodiscover_cuda

__version__ = "0.1.0"
from loom import compilation
from loom._src.domain_index.graph import DomainGraph
from loom._src.domain_index.types import IndexMap, Preimage, domains
from loom._src.pipeline import (
    LoweredProgram,
    compile,
    execute,
    lower_domains,
    lower_to_stablehlo,
    lower_to_xir,
    trace,
)
from loom._src.primitive.api import call
from loom._src.primitive.options import Compile
from loom._src.runtime.instrumentation import ReplayError, observe, replay
from loom._src.runtime.recipes import aot

_autodiscover_cuda()

__all__ = [
    "Compile",
    "ReplayError",
    "compilation",
    "observe",
    "replay",
    "DomainGraph",
    "IndexMap",
    "LoweredProgram",
    "Preimage",
    "__version__",
    "aot",
    "call",
    "compile",
    "domains",
    "execute",
    "lower_domains",
    "lower_to_stablehlo",
    "lower_to_xir",
    "recipe",
    "trace",
]
