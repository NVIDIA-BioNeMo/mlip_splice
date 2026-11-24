// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// How an operand axis reaches its domain from the iteration domain: already
/// there (`Direct`), absent (`Broadcast`), or one index map arrow
/// (`MapLookup` / `PreimageWalk`) or aligned view away.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AxisBridge {
    Direct,
    Broadcast,
    MapLookup { map_idx: usize },
    PreimageWalk { map_idx: usize },
    AlignedLookup { view_idx: usize },
}
