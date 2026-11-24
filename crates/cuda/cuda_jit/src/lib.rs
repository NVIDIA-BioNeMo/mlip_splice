// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod nvjitlink;
mod nvrtc;

use std::path::PathBuf;

use nvrtc::NvrtcCompiler;

#[derive(Clone, Debug, Default)]
pub struct CudaJitConfig {
    pub lib_dirs: Vec<PathBuf>,
    pub include_dirs: Vec<PathBuf>,
}

pub struct CudaJit {
    nvrtc: NvrtcCompiler,
    nvjitlink: nvjitlink::NvJitLink,
    include_dirs: Vec<PathBuf>,
}

pub struct PtxLinker {
    nvjitlink: nvjitlink::NvJitLink,
}

/// One named PTX module of a compile unit. The name is what the linker's
/// diagnostics point at, so a unit gives each of its inputs a distinct one.
pub struct PtxInput<'a> {
    pub name: &'a str,
    pub ptx: &'a [u8],
}

impl PtxLinker {
    pub fn load(lib_dirs: &[PathBuf]) -> Result<Self, String> {
        let nvjitlink = nvjitlink::NvJitLink::load(lib_dirs, None).ok_or_else(|| {
            "libnvJitLink.so.13 not found. Install the CUDA 13 toolkit or add its directory to the toolchain library paths.".to_string()
        })?;
        Ok(PtxLinker { nvjitlink })
    }

    pub fn version(&self) -> Option<(u32, u32, u32)> {
        self.nvjitlink.version()
    }

    /// Link every input of one compile unit into a cubin. The modules are
    /// handed to nvJitLink as the separate inputs they are — a `.func` one of
    /// them defines resolves a `call` another makes — and are never
    /// concatenated into one text.
    pub fn to_cubin(
        &self,
        inputs: &[PtxInput<'_>],
        sm: i32,
        opts: &[&[u8]],
    ) -> Result<Vec<u8>, String> {
        if inputs.is_empty() {
            return Err("a PTX compile unit must state at least one input".to_string());
        }
        let inputs: Vec<nvjitlink::LinkInput<'_>> = inputs
            .iter()
            .map(|input| nvjitlink::LinkInput {
                kind: nvjitlink::NvJitLinkInputType::Ptx,
                data: input.ptx,
                name: input.name,
            })
            .collect();
        self.nvjitlink.link_cubin(sm, &inputs, false, opts)
    }
}

impl CudaJit {
    pub fn load(config: CudaJitConfig) -> Result<CudaJit, String> {
        let nvrtc = NvrtcCompiler::load(&config.lib_dirs).ok_or_else(|| {
            "libnvrtc not found. Tried libnvrtc.so, libnvrtc.so.13, \
             libnvrtc.so.12, libnvrtc.so.11 in the configured lib dirs."
                .to_string()
        })?;
        // nvJitLink is companion-loaded from the directory of the libnvrtc
        // we just mapped (via dladdr) so the two stay version-aligned, with
        // the configured lib dirs as the explicit fallback.
        let nvjitlink = nvjitlink::NvJitLink::load(
            &config.lib_dirs,
            Some(nvrtc.nvrtcVersion as *const std::ffi::c_void),
        )
        .ok_or_else(|| {
            "libnvJitLink.so.13 not found. Install the CUDA 13 toolkit \
             or `uv pip install nvidia-nvjitlink-cu13`, or add its directory \
             to the toolchain lib dirs."
                .to_string()
        })?;
        Ok(CudaJit {
            nvrtc,
            nvjitlink,
            include_dirs: config.include_dirs,
        })
    }

    pub fn to_cubin(
        &self,
        src: &[u8],
        filename: Option<&str>,
        opts: &[String],
        ltoir_blobs: &[&[u8]],
        sm: i32,
    ) -> Result<Vec<u8>, String> {
        let filename = filename.unwrap_or("module");
        if ltoir_blobs.is_empty() {
            let src = std::str::from_utf8(src).map_err(|e| e.to_string())?;
            self.nvrtc
                .cubin_from_plaintext(src, filename, sm, opts, &self.include_dirs)
        } else {
            let nvrtc_lto = self.to_ltoir(src, Some(filename), opts, sm)?;
            self.link_with_ltoir(&nvrtc_lto, filename, sm, ltoir_blobs)
        }
    }

    pub fn to_ltoir(
        &self,
        src: &[u8],
        filename: Option<&str>,
        opts: &[String],
        sm: i32,
    ) -> Result<Vec<u8>, String> {
        let filename = filename.unwrap_or("module");
        let src = std::str::from_utf8(src).map_err(|e| e.to_string())?;
        self.nvrtc
            .ltoir_from_plaintext(src, filename, sm, opts, &self.include_dirs)
    }

    fn link_with_ltoir(
        &self,
        nvrtc_lto: &[u8],
        filename: &str,
        sm: i32,
        ltoir_blobs: &[&[u8]],
    ) -> Result<Vec<u8>, String> {
        let mut inputs: Vec<nvjitlink::LinkInput> = Vec::with_capacity(1 + ltoir_blobs.len());
        inputs.push(nvjitlink::LinkInput {
            kind: nvjitlink::NvJitLinkInputType::Ltoir,
            data: nvrtc_lto,
            name: filename,
        });
        for blob in ltoir_blobs.iter() {
            inputs.push(nvjitlink::LinkInput {
                kind: nvjitlink::NvJitLinkInputType::Ltoir,
                data: blob,
                name: "lib_lto",
            });
        }
        // Each link runs on its own freshly-created nvJitLink handle with no
        // shared mutable state, so candidate links proceed concurrently —
        // together with the NVRTC compiles feeding them — and the FFT
        // pipeline's per-candidate `into_par_iter` actually fans out across
        // cores instead of queuing on a lock.
        self.nvjitlink.link_cubin(sm, &inputs, true, &[])
    }
}
