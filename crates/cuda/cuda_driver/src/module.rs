// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::{CStr, CString, c_char};
use std::os::raw::c_int;
use std::sync::Arc;

use crate::{
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, CUfunction, CUkernel, CUlibrary, CUmodule,
    CudaDriver, Error, Result, check,
};

struct LibraryInner(CUlibrary);

impl Drop for LibraryInner {
    fn drop(&mut self) {
        let drv = CudaDriver::get();
        let _ = unsafe { (drv.cuLibraryUnload)(self.0) };
    }
}

unsafe impl Send for LibraryInner {}
unsafe impl Sync for LibraryInner {}

/// A loaded CUDA library (PTX/CUBIN image), unloaded when the last clone drops.
/// Shared via `Arc`, so any [`Kernel`] it produced keeps it loaded.
#[derive(Clone)]
pub struct Module(Arc<LibraryInner>);

impl Module {
    /// Load a library from a CUBIN/PTX image (`cuLibraryLoadData`). A context
    /// must already be current.
    pub fn load(image: &[u8]) -> Result<Module> {
        let drv = CudaDriver::get();
        let mut lib: CUlibrary = std::ptr::null_mut();
        check(
            unsafe {
                (drv.cuLibraryLoadData)(
                    &mut lib,
                    image.as_ptr() as *const _,
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                )
            },
            "cuLibraryLoadData",
        )?;
        Ok(Module(Arc::new(LibraryInner(lib))))
    }

    /// Look up `name` and return a [`Kernel`] that holds this module alive.
    pub fn kernel(&self, name: &str) -> Result<Kernel> {
        let drv = CudaDriver::get();
        let cname = CString::new(name)
            .map_err(|_| Error::other("kernel name", format!("name {name:?} contains a NUL")))?;
        let mut kernel: CUkernel = std::ptr::null_mut();
        check(
            unsafe { (drv.cuLibraryGetKernel)(&mut kernel, self.raw(), cname.as_ptr()) },
            "cuLibraryGetKernel",
        )?;
        let mut func: CUfunction = std::ptr::null_mut();
        check(
            unsafe { (drv.cuKernelGetFunction)(&mut func, kernel) },
            "cuKernelGetFunction",
        )?;
        Ok(Kernel {
            func,
            _module: self.clone(),
        })
    }

    pub fn raw(&self) -> CUlibrary {
        self.0.0
    }

    /// Query a `CU_FUNC_ATTRIBUTE_*` of `name` for `device` straight from the
    /// library handle (`cuKernelGetAttribute`) — needs no current context, so
    /// a compiler can read assembler feedback (registers, local bytes) from a
    /// cubin it just produced.
    pub fn kernel_attribute(&self, name: &str, attr: c_int, device: crate::Device) -> Result<i32> {
        let drv = CudaDriver::get();
        let cname = CString::new(name)
            .map_err(|_| Error::other("kernel name", format!("name {name:?} contains a NUL")))?;
        let mut kernel: CUkernel = std::ptr::null_mut();
        check(
            unsafe { (drv.cuLibraryGetKernel)(&mut kernel, self.raw(), cname.as_ptr()) },
            "cuLibraryGetKernel",
        )?;
        let mut value: c_int = 0;
        check(
            unsafe { (drv.cuKernelGetAttribute)(&mut value, attr, kernel, device.raw()) },
            "cuKernelGetAttribute",
        )?;
        Ok(value)
    }
}

/// A function looked up from a [`Module`]. Holds a clone of its module, so the
/// library stays loaded as long as the kernel lives.
#[derive(Clone)]
pub struct Kernel {
    func: CUfunction,
    _module: Module,
}

unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    pub fn raw(&self) -> CUfunction {
        self.func
    }

    pub fn function(&self) -> Function {
        Function(self.func)
    }

    pub fn set_max_dynamic_shared(&self, bytes: i32) -> Result<()> {
        self.function()
            .set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes)
    }
}

/// A borrowed `CUfunction` — the introspection/launch handle. Use it to inspect
/// a function obtained from a graph node, or a [`Kernel`]'s underlying function.
#[derive(Clone, Copy)]
pub struct Function(pub(crate) CUfunction);

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl Function {
    pub fn from_raw(func: CUfunction) -> Self {
        Function(func)
    }

    pub fn raw(&self) -> CUfunction {
        self.0
    }

    pub fn name(&self) -> Result<String> {
        let drv = CudaDriver::get();
        let mut s: *const c_char = std::ptr::null();
        check(
            unsafe { (drv.cuFuncGetName)(&mut s, self.0) },
            "cuFuncGetName",
        )?;
        if s.is_null() {
            return Ok(String::new());
        }
        Ok(unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned())
    }
    pub fn module(&self) -> Result<CUmodule> {
        let drv = CudaDriver::get();
        let mut module: CUmodule = std::ptr::null_mut();
        check(
            unsafe { (drv.cuFuncGetModule)(&mut module, self.0) },
            "cuFuncGetModule",
        )?;
        Ok(module)
    }

    pub fn set_attribute(&self, attr: c_int, value: c_int) -> Result<()> {
        let drv = CudaDriver::get();
        check(
            unsafe { (drv.cuFuncSetAttribute)(self.0, attr, value) },
            "cuFuncSetAttribute",
        )
    }
}
