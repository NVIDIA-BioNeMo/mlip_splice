# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures shared by the SPLICE tests.

The tests are organised by layer:

- ``ops/``    — one model operation at a time, against a reference;
- ``model/``  — the assembled model: forward, gradients, padding;
- ``config/`` — YAML, the chain engine, per-op learning-rate annotations;
- ``data/``   — datasets and the packed format;
- ``md/``     — molecular dynamics, which needs a GPU and the built CUDA kernels.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = REPO_ROOT / "examples" / "configs"
