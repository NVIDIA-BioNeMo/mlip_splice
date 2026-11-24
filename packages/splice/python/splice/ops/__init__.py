# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Importing this package registers every op via @register_op side-effect."""

from splice.ops import (
    activation,
    embedding,
    head,
    linear,
    message_passing,
    radial,
    radial_free_knot,
    sph,
    sym_contr,
)

__all__ = [
    "activation",
    "embedding",
    "head",
    "linear",
    "message_passing",
    "radial",
    "radial_free_knot",
    "sph",
    "sym_contr",
]
