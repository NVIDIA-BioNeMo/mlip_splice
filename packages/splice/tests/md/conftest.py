# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MD layer needs a GPU and the separately-built CUDA kernels."""

import pytest

pytest.importorskip("cupy", reason="splice.md needs CuPy")
pytest.importorskip(
    "splice.md",
    reason="splice.md needs the MD CUDA extension — build it with "
    "`make -C packages/splice/python/splice/md/_ops_src`",
)
