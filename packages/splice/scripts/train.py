# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train a SPLICE chain on a dataset with periodic eval + checkpointing.

CLI knobs are deliberately minimal — most architecture/init/scale knobs live
in the arch yaml and the card. The training-side controls (batch, epochs,
LR, eval cadence) are CLI flags.

Compilation is unlimited by default. Set --search-budget-ms and
--shortlist-budget-ms explicitly to bound recipe search and source shortlisting.
Search uses elapsed wall-clock milliseconds; shortlisting uses predicted
realization milliseconds: max(assembly cost / workers, serial tuning cost).
Each accepts a scalar or comma-separated transpose-depth schedule, for example
--search-budget-ms 250 --shortlist-budget-ms 2000,1600,200.

Example:
    python packages/splice/scripts/train.py \\
        --arch examples/configs/fk_cw64_tr256.yaml \\
        --card cards/3bpa.yaml \\
        --train ~/data/3bpa/train_300K.xyz ~/data/3bpa/train_mixedT.xyz \\
        --eval ~/data/3bpa/test_300K.xyz ~/data/3bpa/test_600K.xyz \\
        --output-dir results/3bpa_workhorse_mul16 \\
        --batch-size 32 --n-epochs 200 --base-lr 1e-3 \\
        --eval-every 10
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from splice.training.run import main

if __name__ == "__main__":
    main()
