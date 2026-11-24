// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::{c_uint, c_void};
use std::ops::Deref;

use crate::module::Function;
use crate::{CUdeviceptr, CUstream, Context, CudaDriver, Result, check};

const CU_STREAM_CAPTURE_STATUS_NONE: c_uint = 0;

/// Grid/block/shared-memory shape for a kernel launch or graph kernel node.
#[derive(Clone, Copy)]
pub struct LaunchConfig {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_mem: u32,
}

/// A *borrowed* stream — every stream operation lives here. Externally-owned
/// streams (handed in by XLA/PJRT) are used through `StreamRef::from_raw`; only
/// streams this crate created own their handle (see [`Stream`]).
#[derive(Clone, Copy)]
pub struct StreamRef(pub(crate) CUstream);

unsafe impl Send for StreamRef {}

impl StreamRef {
    pub fn from_raw(raw: CUstream) -> Self {
        StreamRef(raw)
    }

    pub fn raw(&self) -> CUstream {
        self.0
    }

    pub fn synchronize(&self) -> Result<()> {
        let drv = CudaDriver::get();
        check(
            unsafe { (drv.cuStreamSynchronize)(self.0) },
            "cuStreamSynchronize",
        )
    }

    /// Whether this stream is mid-capture, i.e. whether work queued on it is
    /// being recorded into a CUDA graph rather than executed.
    ///
    /// Callers that replay pre-instantiated graphs have to ask: launching a
    /// `CUgraphExec` into a capturing stream is refused outright, and waiting
    /// on an event recorded into one invalidates the capture. Both are legal
    /// on a stream that is merely executing, so the two paths differ and this
    /// is how a caller tells which one it is on. The query itself is a host
    /// lookup (~20ns) and is safe to call during capture.
    ///
    /// A capture that some earlier illegal call already invalidated reports as
    /// capturing — it is still a capture, and the work queued here is still
    /// not going to run. Nothing this answers would be improved by pretending
    /// otherwise.
    pub fn is_capturing(&self) -> Result<bool> {
        let drv = CudaDriver::get();
        let mut status: c_uint = CU_STREAM_CAPTURE_STATUS_NONE;
        check(
            unsafe { (drv.cuStreamIsCapturing)(self.0, &mut status) },
            "cuStreamIsCapturing",
        )?;
        Ok(status != CU_STREAM_CAPTURE_STATUS_NONE)
    }

    /// The context this stream was created in.
    pub fn context(&self) -> Result<Context> {
        let drv = CudaDriver::get();
        let mut ctx = std::ptr::null_mut();
        check(
            unsafe { (drv.cuStreamGetCtx)(self.0, &mut ctx) },
            "cuStreamGetCtx",
        )?;
        Ok(Context(ctx))
    }

    /// `cuLaunchKernel`. `params` is the array of pointers-to-arguments CUDA
    /// expects; it only needs to outlive the (queued) call.
    pub fn launch(
        &self,
        func: Function,
        cfg: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        let drv = CudaDriver::get();
        let [gx, gy, gz] = cfg.grid;
        let [bx, by, bz] = cfg.block;
        let pp = if params.is_empty() {
            std::ptr::null_mut()
        } else {
            params.as_mut_ptr()
        };
        check(
            unsafe {
                (drv.cuLaunchKernel)(
                    func.raw(),
                    gx,
                    gy,
                    gz,
                    bx,
                    by,
                    bz,
                    cfg.shared_mem,
                    self.0,
                    pp,
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
    /// # Safety
    /// `src` must point to at least `nbytes` host bytes that stay valid until
    /// the copy completes on the stream.
    pub unsafe fn memcpy_h2d(
        &self,
        dst: CUdeviceptr,
        src: *const c_void,
        nbytes: usize,
    ) -> Result<()> {
        if nbytes == 0 {
            return Ok(());
        }
        let drv = CudaDriver::get();
        check(
            unsafe { (drv.cuMemcpyHtoDAsync_v2)(dst, src, nbytes, self.0) },
            "cuMemcpyHtoDAsync_v2",
        )
    }
    /// Device-to-device copy on this stream.
    ///
    /// Unlike the host-to-device form, this carries no host pointer, so it is
    /// the only copy that keeps its meaning under graph capture: a captured
    /// memcpy node dereferences its source at every replay, and a host address
    /// is not something a replay can be trusted to still own.
    pub fn memcpy_d2d(&self, dst: CUdeviceptr, src: CUdeviceptr, nbytes: usize) -> Result<()> {
        if nbytes == 0 {
            return Ok(());
        }
        let drv = CudaDriver::get();
        check(
            unsafe { (drv.cuMemcpyDtoDAsync_v2)(dst, src, nbytes, self.0) },
            "cuMemcpyDtoDAsync_v2",
        )
    }

    pub fn memset_d8(&self, dst: CUdeviceptr, value: u8, nbytes: usize) -> Result<()> {
        let drv = CudaDriver::get();
        check(
            unsafe { (drv.cuMemsetD8Async)(dst, value, nbytes, self.0) },
            "cuMemsetD8Async",
        )
    }
}

/// A stream this crate created and owns; destroyed on drop. Derefs to
/// [`StreamRef`] for the operations.
pub struct Stream(StreamRef);

unsafe impl Send for Stream {}

impl Stream {
    /// Create a stream in the current context. A context must already be
    /// current (e.g. via [`ensure_context`](crate::ensure_context)).
    pub fn new() -> Result<Self> {
        let drv = CudaDriver::get();
        let mut raw: CUstream = std::ptr::null_mut();
        check(
            unsafe { (drv.cuStreamCreate)(&mut raw, 0) },
            "cuStreamCreate",
        )?;
        Ok(Stream(StreamRef(raw)))
    }

    pub fn borrow(&self) -> StreamRef {
        self.0
    }
}

impl Deref for Stream {
    type Target = StreamRef;
    fn deref(&self) -> &StreamRef {
        &self.0
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let drv = CudaDriver::get();
        let _ = unsafe { (drv.cuStreamDestroy_v2)(self.0.0) };
    }
}
