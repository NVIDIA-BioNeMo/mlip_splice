// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use crate::{CUcontext, CudaDriver, Device, Result, check};

/// A CUDA context handle. Non-owning and `Copy`: the contexts this crate hands
/// out are long-lived (the process-wide default, or a device's primary
/// context), so dropping a `Context` value frees nothing.
#[derive(Clone, Copy)]
pub struct Context(pub(crate) CUcontext);

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    /// Create a fresh context on `device` (`cuCtxCreate`).
    pub fn create(device: Device) -> Result<Context> {
        init()?;
        let drv = CudaDriver::get();
        let mut ctx: CUcontext = std::ptr::null_mut();
        check(
            unsafe { (drv.cuCtxCreate_v2)(&mut ctx, 0, device.raw()) },
            "cuCtxCreate_v2",
        )?;
        Ok(Context(ctx))
    }

    /// Retain `device`'s primary context (`cuDevicePrimaryCtxRetain`) — the
    /// context the runtime API shares; preferred for multi-GPU.
    pub fn primary(device: Device) -> Result<Context> {
        init()?;
        let drv = CudaDriver::get();
        let mut ctx: CUcontext = std::ptr::null_mut();
        check(
            unsafe { (drv.cuDevicePrimaryCtxRetain)(&mut ctx, device.raw()) },
            "cuDevicePrimaryCtxRetain",
        )?;
        Ok(Context(ctx))
    }

    /// The context current on this thread (may be null).
    pub fn current() -> Context {
        let drv = CudaDriver::get();
        let mut ctx: CUcontext = std::ptr::null_mut();
        unsafe { (drv.cuCtxGetCurrent)(&mut ctx) };
        Context(ctx)
    }

    pub fn set_current(&self) -> Result<()> {
        let drv = CudaDriver::get();
        check(unsafe { (drv.cuCtxSetCurrent)(self.0) }, "cuCtxSetCurrent")
    }

    /// Wrap a raw context handle obtained elsewhere (an FFI boundary, a stream's
    /// owner, …). Lets callers holding a `CUcontext` use the safe API.
    pub fn from_raw(raw: CUcontext) -> Context {
        Context(raw)
    }

    pub fn raw(&self) -> CUcontext {
        self.0
    }

    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}

/// Makes a context current for its lifetime and restores the previously-current
/// one on drop. A no-op (nothing to restore) when the target is already current.
pub struct ContextGuard {
    prev: CUcontext,
    restore: bool,
}

impl ContextGuard {
    pub fn enter(ctx: Context) -> Result<Self> {
        let drv = CudaDriver::get();
        let mut prev: CUcontext = std::ptr::null_mut();
        check(
            unsafe { (drv.cuCtxGetCurrent)(&mut prev) },
            "cuCtxGetCurrent",
        )?;
        if prev == ctx.0 {
            return Ok(Self {
                prev,
                restore: false,
            });
        }
        ctx.set_current()?;
        Ok(Self {
            prev,
            restore: true,
        })
    }
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        if self.restore {
            let drv = CudaDriver::get();
            let _ = unsafe { (drv.cuCtxSetCurrent)(self.prev) };
        }
    }
}

/// Run `f` with `ctx` current, restoring the previous context afterward. A null
/// `ctx` runs `f` as-is.
pub(crate) fn with_current_context<R>(ctx: CUcontext, f: impl FnOnce() -> Result<R>) -> Result<R> {
    if ctx.is_null() {
        return f();
    }
    let _guard = ContextGuard::enter(Context(ctx))?;
    f()
}

/// Ensure a context exists and is current on this thread, creating one on
/// device 0 on first call (the single-GPU default). Idempotent and
/// thread-aware: the context is created once, then made current per thread.
pub fn ensure_context() -> Result<()> {
    static CONTEXT: OnceLock<Result<usize>> = OnceLock::new();
    let addr = CONTEXT
        .get_or_init(|| {
            let device = Device::get(0)?;
            Ok(Context::create(device)?.raw() as usize)
        })
        .clone()?;
    Context(addr as CUcontext).set_current()
}

/// `cuInit(0)`, exactly once.
pub(crate) fn init() -> Result<()> {
    static INIT: OnceLock<Result<()>> = OnceLock::new();
    INIT.get_or_init(|| {
        let drv = CudaDriver::get();
        check(unsafe { (drv.cuInit)(0) }, "cuInit")
    })
    .clone()
}
