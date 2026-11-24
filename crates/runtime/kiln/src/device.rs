// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_driver::{CUcontext, CUstream, StreamRef};

use crate::Error;

pub(crate) unsafe fn context_of_stream(stream: CUstream) -> Result<CUcontext, Error> {
    Ok(StreamRef::from_raw(stream)
        .context()
        .map_err(Error::Driver)?
        .raw())
}
