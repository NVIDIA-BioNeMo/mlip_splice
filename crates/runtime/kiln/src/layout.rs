// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_driver::CUdeviceptr;

use recipe::layout::checked_workspace_offsets;

use crate::Error;

/// Where each declared buffer sits in one allocation starting at `base`.
///
/// The sizes are a body's own declarations, so the region they carve can leave
/// the address space and there is no set of pointers to answer with. Every
/// caller here holds work whose layout was proved — a residency proves it as it
/// loads — so this returns the refusal rather than deciding on their behalf.
pub(crate) fn carve(
    base: CUdeviceptr,
    sizes: impl Iterator<Item = usize>,
) -> Result<Vec<CUdeviceptr>, Error> {
    let (offsets, _span) = checked_workspace_offsets(sizes).ok_or(Error::InvalidRecipe(
        recipe::RecipeError::WorkspaceLayoutOverflow,
    ))?;
    Ok(offsets
        .iter()
        .map(|&off| base + off as CUdeviceptr)
        .collect())
}
