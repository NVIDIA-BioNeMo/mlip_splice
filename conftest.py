# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest setup shared across every package's tests.

Each package keeps its tests beside its sources, in ``packages/<name>/tests``;
``tests/`` holds only the contracts that span packages.
"""

import os

# Parts of the suite lower a program in a child interpreter, and a lowering needs
# the driver rather than device memory. Reserving most of the device in this
# process would leave a child nothing to build a context from, so this process
# takes what it uses and no more. An operator who wants the reservation back sets
# the variable before pytest starts.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
