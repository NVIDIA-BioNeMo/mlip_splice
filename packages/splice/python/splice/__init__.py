# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPLine Interatomic Cluster Expansion (SPLICE): model, training and inference."""

__version__ = "0.1.0"

from splice import ops as ops  # registers ops via import side-effect
from splice.config import Chain, build_chain, dump_config, load_config
from splice.context import build_context
from splice.forward import model_call
from splice.registry import get_op, list_ops, register_op

__all__ = [
    "Chain",
    "__version__",
    "build_chain",
    "build_context",
    "dump_config",
    "get_op",
    "list_ops",
    "load_config",
    "model_call",
    "register_op",
]
