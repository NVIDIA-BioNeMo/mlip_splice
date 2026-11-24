// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dlbind::load_sym;
use libloading::Library;
use std::ffi::{c_char, c_int, c_void};
use std::path::PathBuf;

pub(crate) type NvJitLinkHandle = *mut c_void;
pub(crate) type NvJitLinkResult = c_int;

pub(crate) const NVJITLINK_SUCCESS: NvJitLinkResult = 0;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum NvJitLinkInputType {
    Ptx = 2,
    Ltoir = 3,
}

#[allow(non_snake_case)]
pub(crate) struct NvJitLink {
    _lib: Library,
    loaded_path: Option<PathBuf>,
    version: Option<(u32, u32, u32)>,
    lib_dirs: Vec<PathBuf>,
    nvJitLinkCreate:
        unsafe extern "C" fn(*mut NvJitLinkHandle, u32, *const *const c_char) -> NvJitLinkResult,
    nvJitLinkDestroy: unsafe extern "C" fn(*mut NvJitLinkHandle) -> NvJitLinkResult,
    nvJitLinkAddData: unsafe extern "C" fn(
        NvJitLinkHandle,
        c_int,
        *const c_void,
        usize,
        *const c_char,
    ) -> NvJitLinkResult,
    nvJitLinkComplete: unsafe extern "C" fn(NvJitLinkHandle) -> NvJitLinkResult,
    nvJitLinkGetLinkedCubinSize:
        unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult,
    nvJitLinkGetLinkedCubin: unsafe extern "C" fn(NvJitLinkHandle, *mut c_void) -> NvJitLinkResult,
    nvJitLinkGetErrorLogSize: unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult,
    nvJitLinkGetErrorLog: unsafe extern "C" fn(NvJitLinkHandle, *mut c_char) -> NvJitLinkResult,
}

// SAFETY: the loaded function pointers refer to a library kept alive for the
// lifetime of this struct.
unsafe impl Send for NvJitLink {}
unsafe impl Sync for NvJitLink {}

const SONAMES: [&str; 2] = ["libnvJitLink.so.13", "libnvJitLink.so"];

fn cross_product(dirs: &[PathBuf], names: &[&str]) -> Vec<PathBuf> {
    let mut out = Vec::with_capacity(dirs.len() * names.len());
    for d in dirs {
        for n in names {
            out.push(d.join(n));
        }
    }
    out
}

impl NvJitLink {
    pub(crate) fn version(&self) -> Option<(u32, u32, u32)> {
        self.version
    }

    pub(crate) fn load(
        lib_dirs: &[PathBuf],
        companion_symbol: Option<*const c_void>,
    ) -> Option<Self> {
        let companion: Vec<PathBuf> = companion_symbol
            .and_then(|symbol| unsafe { dlfind::loaded_lib_dir(symbol) })
            .map(|dir| SONAMES.iter().map(|name| dir.join(name)).collect())
            .unwrap_or_default();
        // Reuse a resident libnvJitLink before opening an absolute path: two
        // separately mapped copies can corrupt their shared JIT-LTO state.
        let fallback: Vec<_> = [companion, SONAMES.iter().map(PathBuf::from).collect()].concat();
        let configured = cross_product(lib_dirs, &SONAMES);
        let lib = dlfind::open_resident(&SONAMES)
            .or_else(|| dlfind::open_first(&configured))
            .or_else(|| dlfind::open_first(&fallback))?;
        type VerFn = unsafe extern "C" fn(*mut u32, *mut u32) -> NvJitLinkResult;
        let version_fn: Option<VerFn> = unsafe { lib.get::<VerFn>(b"nvJitLinkVersion\0") }
            .ok()
            .map(|s| *s);
        let version = version_fn.and_then(|f| {
            let (mut major, mut minor) = (0u32, 0u32);
            (unsafe { f(&mut major, &mut minor) } == NVJITLINK_SUCCESS).then_some((major, minor, 0))
        });
        // SAFETY: `nvJitLinkCreate` is a freshly resolved symbol in the
        // loaded library. We don't call it — just hand its address to
        // dladdr.
        let create_sym: libloading::Symbol<unsafe extern "C" fn()> =
            unsafe { lib.get(b"nvJitLinkCreate\0") }.ok()?;
        let loaded_path = unsafe { dlfind::loaded_lib_path(*create_sym as *const c_void) };
        Some(Self {
            loaded_path,
            version,
            lib_dirs: lib_dirs.to_vec(),
            nvJitLinkCreate: load_sym!(
                &lib,
                "nvJitLinkCreate",
                unsafe extern "C" fn(
                    *mut NvJitLinkHandle,
                    u32,
                    *const *const c_char,
                ) -> NvJitLinkResult
            ),
            nvJitLinkDestroy: load_sym!(
                &lib,
                "nvJitLinkDestroy",
                unsafe extern "C" fn(*mut NvJitLinkHandle) -> NvJitLinkResult
            ),
            nvJitLinkAddData: load_sym!(
                &lib,
                "nvJitLinkAddData",
                unsafe extern "C" fn(
                    NvJitLinkHandle,
                    c_int,
                    *const c_void,
                    usize,
                    *const c_char,
                ) -> NvJitLinkResult
            ),
            nvJitLinkComplete: load_sym!(
                &lib,
                "nvJitLinkComplete",
                unsafe extern "C" fn(NvJitLinkHandle) -> NvJitLinkResult
            ),
            nvJitLinkGetLinkedCubinSize: load_sym!(
                &lib,
                "nvJitLinkGetLinkedCubinSize",
                unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult
            ),
            nvJitLinkGetLinkedCubin: load_sym!(
                &lib,
                "nvJitLinkGetLinkedCubin",
                unsafe extern "C" fn(NvJitLinkHandle, *mut c_void) -> NvJitLinkResult
            ),
            nvJitLinkGetErrorLogSize: load_sym!(
                &lib,
                "nvJitLinkGetErrorLogSize",
                unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult
            ),
            nvJitLinkGetErrorLog: load_sym!(
                &lib,
                "nvJitLinkGetErrorLog",
                unsafe extern "C" fn(NvJitLinkHandle, *mut c_char) -> NvJitLinkResult
            ),
            _lib: lib,
        })
    }

    fn provenance(&self) -> String {
        let path = self
            .loaded_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<unknown>".to_string());
        let version = self
            .version
            .map(|(a, b, c)| format!("{a}.{b}.{c}"))
            .unwrap_or_else(|| "<unknown>".to_string());
        let mut out = format!("loaded libnvJitLink: {path} (version {version})");
        let mut alternates: Vec<PathBuf> = cross_product(&self.lib_dirs, &SONAMES)
            .into_iter()
            .filter(|p| p.exists())
            .filter(|p| {
                self.loaded_path.as_deref().is_none_or(|loaded| {
                    std::fs::canonicalize(p).ok() != std::fs::canonicalize(loaded).ok()
                })
            })
            .collect();
        alternates.dedup();
        if !alternates.is_empty() {
            out.push_str("\nother libnvJitLink copies known to the engine:");
            for p in &alternates {
                out.push_str("\n  ");
                out.push_str(&p.display().to_string());
            }
            out.push_str(
                "\nIf the loaded copy is older than required: the engine reused \
                 the libnvJitLink already mapped into this process, to avoid a \
                 second, conflicting instance. Make the process load a newer one \
                 first — prepend it to $LD_LIBRARY_PATH or upgrade the system \
                 CUDA toolkit.",
            );
        }
        out
    }

    fn error_log(&self, handle: NvJitLinkHandle) -> String {
        let mut n: usize = 0;
        if unsafe { (self.nvJitLinkGetErrorLogSize)(handle, &mut n) } != NVJITLINK_SUCCESS || n == 0
        {
            return String::new();
        }
        let mut buf = vec![0u8; n];
        if unsafe { (self.nvJitLinkGetErrorLog)(handle, buf.as_mut_ptr() as *mut c_char) }
            != NVJITLINK_SUCCESS
        {
            return String::new();
        }
        if let Some(p) = buf.iter().position(|&b| b == 0) {
            buf.truncate(p);
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    pub(crate) fn link_cubin(
        &self,
        sm: i32,
        inputs: &[LinkInput],
        lto: bool,
        extra_opts: &[&[u8]],
    ) -> Result<Vec<u8>, String> {
        let arch = std::ffi::CString::new(format!("-arch=sm_{sm}")).unwrap();
        // `-lto` is only correct when the inputs are LTO-IR; a plain PTX/cubin
        // link (ptxas assemble) must omit it.
        let lto_opt = std::ffi::CString::new("-lto").unwrap();
        let mut opt_ptrs: Vec<*const c_char> = vec![arch.as_ptr()];
        if lto {
            opt_ptrs.push(lto_opt.as_ptr());
        }
        let extra_cstrs: Vec<std::ffi::CString> = extra_opts
            .iter()
            .map(|b| std::ffi::CString::new(*b).expect("nvJitLink opt contains NUL"))
            .collect();
        opt_ptrs.extend(extra_cstrs.iter().map(|c| c.as_ptr()));

        let mut handle: NvJitLinkHandle = std::ptr::null_mut();
        let r = unsafe {
            (self.nvJitLinkCreate)(&mut handle, opt_ptrs.len() as u32, opt_ptrs.as_ptr())
        };
        if r != NVJITLINK_SUCCESS {
            return Err(format!("nvJitLinkCreate failed: {r}"));
        }
        struct LinkGuard<'a> {
            h: NvJitLinkHandle,
            lib: &'a NvJitLink,
        }
        impl Drop for LinkGuard<'_> {
            fn drop(&mut self) {
                if !self.h.is_null() {
                    unsafe { (self.lib.nvJitLinkDestroy)(&mut self.h) };
                }
            }
        }
        let _g = LinkGuard {
            h: handle,
            lib: self,
        };

        for inp in inputs {
            let name = std::ffi::CString::new(inp.name).map_err(|e| e.to_string())?;
            // PTX is text: nvJitLink parses it as a C string, so the buffer
            // must be NUL-terminated (cuLinkAddData documents the same).
            let nul_owned;
            let data: &[u8] = if inp.kind == NvJitLinkInputType::Ptx && inp.data.last() != Some(&0)
            {
                nul_owned = [inp.data, &[0]].concat();
                &nul_owned
            } else {
                inp.data
            };
            let r = unsafe {
                (self.nvJitLinkAddData)(
                    handle,
                    inp.kind as c_int,
                    data.as_ptr() as *const c_void,
                    data.len(),
                    name.as_ptr(),
                )
            };
            if r != NVJITLINK_SUCCESS {
                return Err(format!(
                    "nvJitLinkAddData({}) failed: {r}: {}\n{}",
                    inp.name,
                    self.error_log(handle),
                    self.provenance(),
                ));
            }
        }

        let r = unsafe { (self.nvJitLinkComplete)(handle) };
        if r != NVJITLINK_SUCCESS {
            return Err(format!(
                "nvJitLinkComplete failed: {r}: {}\n{}",
                self.error_log(handle),
                self.provenance(),
            ));
        }

        let mut n: usize = 0;
        let r = unsafe { (self.nvJitLinkGetLinkedCubinSize)(handle, &mut n) };
        if r != NVJITLINK_SUCCESS {
            return Err(format!("nvJitLinkGetLinkedCubinSize failed: {r}"));
        }
        let mut cubin = vec![0u8; n];
        let r =
            unsafe { (self.nvJitLinkGetLinkedCubin)(handle, cubin.as_mut_ptr() as *mut c_void) };
        if r != NVJITLINK_SUCCESS {
            return Err(format!("nvJitLinkGetLinkedCubin failed: {r}"));
        }
        Ok(cubin)
    }
}

/// One input of a link: what the bytes are, the bytes, and the name the
/// linker's diagnostics refer to them by.
pub(crate) struct LinkInput<'a> {
    pub kind: NvJitLinkInputType,
    pub data: &'a [u8],
    pub name: &'a str,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaJitConfig;

    fn loaded() -> Option<crate::CudaJit> {
        crate::CudaJit::load(CudaJitConfig::default()).ok()
    }

    #[test]
    fn library_loads() {
        if loaded().is_none() {
            eprintln!("skip: CUDA JIT toolchain not loadable in this environment");
        }
    }

    #[test]
    fn link_with_bogus_input_returns_error() {
        let Some(jit) = loaded() else {
            eprintln!("skip: CUDA JIT toolchain not loadable in this environment");
            return;
        };
        let err = jit
            .nvjitlink
            .link_cubin(
                120,
                &[LinkInput {
                    kind: NvJitLinkInputType::Ltoir,
                    data: b"not actually LTOIR",
                    name: "bogus",
                }],
                true,
                &[],
            )
            .expect_err("bogus LTOIR must be rejected");
        assert!(!err.is_empty());
    }
}
