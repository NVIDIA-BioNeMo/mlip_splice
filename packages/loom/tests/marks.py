# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest

f32_only = pytest.mark.parametrize("dtype", [jnp.float32], indirect=True)
gpu_only = pytest.mark.parametrize("device", ["gpu"], indirect=True)

try:
    _has_gpu = bool(jax.devices("gpu"))
except RuntimeError:
    _has_gpu = False
needs_gpu = pytest.mark.skipif(not _has_gpu, reason="No GPU")
