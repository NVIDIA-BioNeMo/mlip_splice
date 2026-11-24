// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::OnceLock;

use cuda_jit::PtxLinker;
use parking_lot::Mutex;

pub use blob_cache::CacheSource;

mod compile_task;
mod environment;

pub use compile_task::CompileTask;

pub use environment::{
    CgroupCpuLimit, CompileEnvironment, compile_environment, in_compile_pool, spawn_compile,
};

struct State {
    lib_dirs: Vec<PathBuf>,
    frozen: bool,
}
static STATE: Mutex<State> = Mutex::new(State {
    lib_dirs: Vec::new(),
    frozen: false,
});

static PTX_LINKER: OnceLock<Result<PtxLinker, String>> = OnceLock::new();

static CACHE: OnceLock<blob_cache::Cache> = OnceLock::new();

pub fn cache() -> &'static blob_cache::Cache {
    CACHE.get_or_init(|| blob_cache::Cache::open(default_cache_dir(), compile_cache_max_bytes()))
}

pub fn ptx_linker() -> Result<&'static PtxLinker, String> {
    STATE.lock().frozen = true;
    PTX_LINKER
        .get_or_init(|| PtxLinker::load(&lib_dirs()))
        .as_ref()
        .map_err(Clone::clone)
}

fn default_cache_dir() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("LOOM_COMPILE_CACHE_DIR") {
        return (!path.is_empty()).then(|| PathBuf::from(path));
    }
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .map(|root| root.join("loom").join("compile"))
}

fn compile_cache_max_bytes() -> Option<u64> {
    std::env::var("LOOM_COMPILE_CACHE_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
}

pub fn add_lib_dir(dir: PathBuf) -> Result<(), String> {
    let mut state = STATE.lock();
    if state.frozen {
        return Err("cannot add a CUDA library directory after the linker is initialized".into());
    }
    push_unique(&mut state.lib_dirs, dir);
    Ok(())
}

pub fn lib_dirs() -> Vec<PathBuf> {
    STATE.lock().lib_dirs.clone()
}

fn push_unique(list: &mut Vec<PathBuf>, dir: PathBuf) {
    if !list.iter().any(|d| d == &dir) {
        list.push(dir);
    }
}

/// How a single self-contained PTX image becomes a cubin. nvJitLink assembles
/// relocatable (`-c`), whose ABI constraints inflate register allocation
/// (measured 254 vs 168 registers on the same loom kernel — a third of its
/// occupancy at one warp per block); an external `ptxas` binary assembles
/// whole-program. Neither dominates across workloads, so every compile states
/// its assembler explicitly.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum PtxAssembler {
    #[default]
    NvJitLink,
    Ptxas(PathBuf),
}

impl PtxAssembler {
    pub fn name(&self) -> &'static str {
        match self {
            PtxAssembler::NvJitLink => "nvjitlink",
            PtxAssembler::Ptxas(_) => "ptxas",
        }
    }

    pub fn version(&self) -> Result<String, String> {
        match self {
            PtxAssembler::NvJitLink => Ok(ptx_linker()?
                .version()
                .map(|(major, minor, patch)| format!("{major}.{minor}.{patch}"))
                .unwrap_or_else(|| "unknown".into())),
            PtxAssembler::Ptxas(path) => ptxas_version(path),
        }
    }
}

fn ptxas_version(path: &std::path::Path) -> Result<String, String> {
    use std::collections::HashMap;
    static VERSIONS: OnceLock<Mutex<HashMap<PathBuf, Result<String, String>>>> = OnceLock::new();
    let versions = VERSIONS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(version) = versions.lock().get(path).cloned() {
        return version;
    }
    let version = std::process::Command::new(path)
        .arg("--version")
        .output()
        .map_err(|error| format!("failed to run {}: {error}", path.display()))
        .and_then(|output| {
            if output.status.success() {
                Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
            } else {
                Err(format!("{} --version failed", path.display()))
            }
        });
    versions.lock().insert(path.to_owned(), version.clone());
    version
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PtxCompileOptions {
    pub max_registers: Option<u32>,
}

pub struct PtxCompileResult {
    pub cubin: Vec<u8>,
    pub cache_source: CacheSource,
    pub production_cost_ms: u64,
}

static PTX_ASSEMBLER: Mutex<PtxAssembler> = Mutex::new(PtxAssembler::NvJitLink);
pub fn ptx_assembler() -> PtxAssembler {
    PTX_ASSEMBLER.lock().clone()
}

pub fn ptxas_assembler() -> Option<PtxAssembler> {
    static PTXAS: OnceLock<Option<PathBuf>> = OnceLock::new();
    let found = PTXAS.get_or_init(|| {
        let beside = lib_dirs()
            .into_iter()
            .filter_map(|dir| Some(dir.parent()?.join("bin").join("ptxas")));
        let on_path = std::env::var_os("PATH")
            .into_iter()
            .flat_map(|paths| {
                std::env::split_paths(&paths)
                    .map(|dir| dir.join("ptxas"))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        beside.chain(on_path).find(|path| path.is_file())
    });
    found.clone().map(PtxAssembler::Ptxas)
}

/// Assemble one whole PTX compile unit into a cubin, cached by everything that
/// determines the result. An order of magnitude faster than compiling CUDA C.
///
/// `sources` is the unit in link order and is never empty. One source is
/// assembled on its own — whole-program `ptxas` can do that, and so can the
/// driver's own JIT above this call; several are handed to nvJitLink as the
/// separate inputs they are, so a `call` in one resolves against a `.func` in
/// another. The texts are never concatenated: two modules are two translation
/// units, and joining them would silently merge two `.version` and `.target`
/// headers into one program neither of them states.
pub fn compile_ptx_unit_reported(
    sources: &[&[u8]],
    assembler: &PtxAssembler,
    sm: i32,
    options: PtxCompileOptions,
) -> Result<PtxCompileResult, String> {
    if sources.is_empty() {
        return Err("a PTX compile unit states at least one source to assemble".into());
    }
    let capped = match options.max_registers {
        Some(max_registers) => Some(cap_unit_registers(sources, max_registers)?),
        None => None,
    };
    let assembled: Cow<'_, [&[u8]]> = match &capped {
        Some(capped) => Cow::Owned(capped.iter().map(Vec::as_slice).collect()),
        None => Cow::Borrowed(sources),
    };
    let cached = match assembler {
        PtxAssembler::NvJitLink => {
            let linker = ptx_linker()?;
            let version = linker.version().unwrap_or_default();
            let key = ptx_unit_cache_key(
                sources,
                b"nvjitlink",
                sm,
                options,
                &[
                    &version.0.to_le_bytes(),
                    &version.1.to_le_bytes(),
                    &version.2.to_le_bytes(),
                ],
            );
            cache().get_or_insert_reported(&key, "cubin", || {
                // The name is what a linker diagnostic points at, and what
                // identifies a source here is its place in the unit.
                let names: Vec<String> = (0..assembled.len())
                    .map(|input| format!("recipe.{input}.ptx"))
                    .collect();
                let inputs: Vec<cuda_jit::PtxInput<'_>> = names
                    .iter()
                    .zip(assembled.iter())
                    .map(|(name, ptx)| cuda_jit::PtxInput { name, ptx })
                    .collect();
                linker.to_cubin(&inputs, sm, &[])
            })
        }
        PtxAssembler::Ptxas(path) => {
            let [source] = assembled.as_ref() else {
                return Err(format!(
                    "whole-program ptxas assembles one PTX module; this compile unit has {} \
                     inputs and needs a linker",
                    assembled.len()
                ));
            };
            let mut identity = Vec::from(path.as_os_str().as_encoded_bytes());
            if let Ok(metadata) = std::fs::metadata(path) {
                identity.extend_from_slice(&metadata.len().to_le_bytes());
                if let Ok(modified) = metadata.modified()
                    && let Ok(since_epoch) = modified.duration_since(std::time::UNIX_EPOCH)
                {
                    identity.extend_from_slice(&since_epoch.as_nanos().to_le_bytes());
                }
            }
            if let Ok(version) = ptxas_version(path) {
                identity.extend_from_slice(version.as_bytes());
            }
            let key = ptx_unit_cache_key(sources, b"ptxas", sm, options, &[&identity]);
            cache().get_or_insert_reported(&key, "cubin", || ptxas_cubin(path, source, sm))
        }
    }?;
    Ok(PtxCompileResult {
        cubin: cached.bytes,
        cache_source: cached.source,
        production_cost_ms: cached.production_cost_ms,
    })
}

/// A register cap over a whole compile unit.
///
/// `.maxnreg` bounds an entry's allocation, so it is written into every input
/// that defines one and into no others: a module of `.func` callables has no
/// entry to bound and crosses unchanged, and rewriting it would put a
/// directive where PTX does not admit one. A unit that defines no entry
/// anywhere launches nothing, so its cap bounds nothing and the request is
/// refused rather than assembled into an image no bound was applied to.
fn cap_unit_registers(sources: &[&[u8]], max_registers: u32) -> Result<Vec<Vec<u8>>, String> {
    let mut capped = Vec::with_capacity(sources.len());
    let mut entries = 0usize;
    for source in sources {
        let (source, in_source) = cap_entry_registers(source, max_registers)?;
        entries += in_source;
        capped.push(source);
    }
    if entries == 0 {
        return Err("PTX register cap requested for a compile unit with no entry function".into());
    }
    Ok(capped)
}

/// One module with `.maxnreg` on each of its entries, and how many it had.
fn cap_entry_registers(ptx: &[u8], max_registers: u32) -> Result<(Vec<u8>, usize), String> {
    let ptx = std::str::from_utf8(ptx).map_err(|error| format!("invalid PTX: {error}"))?;
    let directive = format!(".maxnreg {max_registers}\n");
    let mut output = String::with_capacity(ptx.len() + directive.len());
    let mut inside_entry_declaration = false;
    let mut wrote_directive = false;
    let mut entries = 0usize;
    for line in ptx.split_inclusive('\n') {
        let trimmed = line.trim();
        if trimmed.starts_with(".visible .entry ") || trimmed.starts_with(".entry ") {
            inside_entry_declaration = true;
            wrote_directive = false;
            entries += 1;
        }
        if inside_entry_declaration && trimmed.starts_with(".maxnreg ") {
            output.push_str(&directive);
            wrote_directive = true;
            continue;
        }
        if inside_entry_declaration && trimmed == "{" {
            if !wrote_directive {
                output.push_str(&directive);
            }
            inside_entry_declaration = false;
        }
        output.push_str(line);
    }
    if inside_entry_declaration {
        return Err("PTX entry declaration is incomplete".into());
    }
    Ok((output.into_bytes(), entries))
}

/// What determines the cubin: the assembler and its version, the target, the
/// options, and the unit's sources in link order — length-framed, so no two
/// units agree by re-cutting one source into the next.
fn ptx_unit_cache_key(
    sources: &[&[u8]],
    assembler: &[u8],
    sm: i32,
    options: PtxCompileOptions,
    identity: &[&[u8]],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"loom-ptx-cubin-v1\0");
    hasher.update(assembler);
    hasher.update(&sm.to_le_bytes());
    match options.max_registers {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_le_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    for field in identity {
        hasher.update(&(field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    hasher.update(&(sources.len() as u64).to_le_bytes());
    for source in sources {
        hasher.update(&(source.len() as u64).to_le_bytes());
        hasher.update(source);
    }
    *hasher.finalize().as_bytes()
}

fn ptxas_cubin(ptxas: &std::path::Path, ptx: &[u8], sm: i32) -> Result<Vec<u8>, String> {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir();
    let src = dir.join(format!("loom-ptx-{}-{seq}.ptx", std::process::id()));
    let out = dir.join(format!("loom-ptx-{}-{seq}.cubin", std::process::id()));
    std::fs::write(&src, ptx).map_err(|e| e.to_string())?;
    let mut command = std::process::Command::new(ptxas);
    command.arg(format!("-arch=sm_{sm}"));
    let result = command.arg("-o").arg(&out).arg(&src).output();
    let _ = std::fs::remove_file(&src);
    let output = result.map_err(|e| format!("failed to run {}: {e}", ptxas.display()))?;
    if !output.status.success() {
        let _ = std::fs::remove_file(&out);
        return Err(format!(
            "ptxas failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let cubin = std::fs::read(&out).map_err(|e| e.to_string());
    let _ = std::fs::remove_file(&out);
    cubin
}

/// Assembler feedback read back from a compiled kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelResources {
    pub registers: u32,
    /// Per-thread local frame — the spill bytes for kernels without local
    /// arrays.
    pub local_bytes: u32,
}

pub fn kernel_image_resources(
    image: &[u8],
    entries: &[impl AsRef<str>],
) -> Option<Vec<KernelResources>> {
    let device = cuda_driver::Device::current();
    let module = cuda_driver::Module::load(image).ok()?;
    entries
        .iter()
        .map(|entry| {
            let entry = entry.as_ref();
            let registers = module
                .kernel_attribute(entry, cuda_driver::ffi::CU_FUNC_ATTRIBUTE_NUM_REGS, device)
                .ok()?;
            let local_bytes = module
                .kernel_attribute(
                    entry,
                    cuda_driver::ffi::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                    device,
                )
                .ok()?;
            Some(KernelResources {
                registers: registers as u32,
                local_bytes: local_bytes as u32,
            })
        })
        .collect()
}

#[doc(hidden)]
pub fn populate_from_python_for_tests() {
    use std::fs;
    let Ok(output) = std::process::Command::new("python")
        .args(["-c", "import sys; print(sys.prefix)"])
        .output()
    else {
        return;
    };
    if !output.status.success() {
        return;
    }
    let Ok(prefix) = std::str::from_utf8(&output.stdout) else {
        return;
    };
    let lib_dir = PathBuf::from(prefix.trim()).join("lib");
    let Ok(entries) = fs::read_dir(&lib_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let site = entry.path().join("site-packages");
        if !site.is_dir() {
            continue;
        }
        for sub in [
            "nvidia/cu13/lib",
            "nvidia/cu12/lib",
            "nvidia/cuda_nvrtc/lib",
            "nvidia/nvjitlink/lib",
            "nvidia/cuda_runtime/lib",
        ] {
            let p = site.join(sub);
            if p.is_dir() {
                let _ = add_lib_dir(p);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CALLABLE: &[u8] =
        b".version 8.7\n.target sm_90\n.visible .func scale(.param .b32 x)\n{\nret;\n}\n";

    #[test]
    fn register_cap_is_a_directive_on_every_entry() {
        let ptx = b".version 8.7\n.visible .entry first()\n.maxntid 32, 1, 1\n{\nret;\n}\n.entry second()\n.maxnreg 64\n{\nret;\n}\n";
        let (capped, entries) = cap_entry_registers(ptx, 128).unwrap();
        let capped = String::from_utf8(capped).unwrap();
        assert_eq!(entries, 2);
        assert_eq!(capped.matches(".maxnreg 128").count(), 2);
        assert!(!capped.contains(".maxnreg 64"));
        assert!(capped.contains(".maxntid 32, 1, 1\n.maxnreg 128\n{"));
    }

    /// The cap belongs to the unit, not to each of its inputs: it is written
    /// where an entry can carry it and nowhere else, and it needs one entry
    /// somewhere rather than one in every module.
    #[test]
    fn register_cap_is_a_unit_level_rule() {
        let entry = b".version 8.7\n.visible .entry first()\n{\nret;\n}\n";
        let capped = cap_unit_registers(&[CALLABLE, entry], 128).unwrap();
        assert_eq!(capped[0], CALLABLE, "a callable-only module is unchanged");
        assert_eq!(
            String::from_utf8(capped[1].clone())
                .unwrap()
                .matches(".maxnreg 128")
                .count(),
            1
        );
    }

    #[test]
    fn register_cap_requires_an_entry_somewhere_in_the_unit() {
        assert!(cap_unit_registers(&[b".version 8.7\n"], 128).is_err());
        assert!(cap_unit_registers(&[CALLABLE, CALLABLE], 128).is_err());
    }

    /// Two units differ in the key whenever they differ at all, including when
    /// one source's bytes are re-cut into the next.
    #[test]
    fn a_unit_key_frames_every_source() {
        let key = |sources: &[&[u8]]| {
            ptx_unit_cache_key(sources, b"nvjitlink", 90, PtxCompileOptions::default(), &[])
        };
        assert_eq!(key(&[b"ab", b"c"]), key(&[b"ab", b"c"]));
        assert_ne!(key(&[b"ab", b"c"]), key(&[b"a", b"bc"]));
        assert_ne!(key(&[b"ab", b"c"]), key(&[b"c", b"ab"]));
        assert_ne!(key(&[b"abc"]), key(&[b"ab", b"c"]));
    }

    #[test]
    fn a_unit_states_at_least_one_source() {
        assert!(
            compile_ptx_unit_reported(
                &[],
                &PtxAssembler::NvJitLink,
                90,
                PtxCompileOptions::default()
            )
            .is_err()
        );
    }
}
