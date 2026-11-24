# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail if a source file carries no SPDX licence header.

The OSRB requires an SPDX identifier on every NVIDIA-authored source file. This
ran as a one-off cleanup once; the hook exists so that the next new crate cannot
quietly reintroduce the gap. Vendored third-party files are excluded in
.pre-commit-config.yaml -- they keep their upstream notice instead.
"""

import sys

missing = [
    f
    for f in sys.argv[1:]
    if "SPDX-License-Identifier" not in open(f, encoding="utf-8", errors="replace").read()
]
if missing:
    print("Missing SPDX-License-Identifier header:")
    for f in missing:
        print(f"  {f}")
    print("\nAdd, with the comment syntax for the file:")
    print(
        "  SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION "
        "& AFFILIATES. All rights reserved."
    )
    print("  SPDX-License-Identifier: Apache-2.0")
    sys.exit(1)
