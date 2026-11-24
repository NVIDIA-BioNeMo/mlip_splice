# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""draupnir — a YAML dataflow format: steps invoking registered ops over a shared context.

The registry is any mapping `name -> op class`; value decoding is an application-supplied hook.

    config = load_config(document, registry, overrides={}, decode=None)  # YAML -> step mappings
    chain  = build_chain(config, registry, key)                          # instantiate and bind
    final  = chain(context)                                              # execute
    text   = dump_registry(registry)                                     # op metadata as YAML
"""

from .chain import Chain, StepSpec, build_chain
from .config import dump_config, load_config
from .registry import dump_registry
from .transforms import Graph, profile, record, render

__version__ = "0.1.0"

__all__ = [
    "Chain",
    "Graph",
    "StepSpec",
    "__version__",
    "build_chain",
    "dump_config",
    "dump_registry",
    "load_config",
    "profile",
    "record",
    "render",
]
