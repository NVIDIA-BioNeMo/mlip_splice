// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One compile: what identifies it, what produces it, and whether what it
//! produced is admissible.

use std::collections::BTreeSet;
use std::time::Instant;

use cuda_toolchain::{KernelResources, PtxAssembler};
use recipe::algebra::Launches;
use recipe::{Blob, CompileRequest, ImagePersistence, PtxAssemblyMode, Recipe, ResourceBounds};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompileFailureKind {
    IncompatibleTarget,
    /// The assembly policy the request states cannot express the compile it
    /// asks for. A property of the request, not of this machine.
    IncompatibleAssembly,
    UnavailableToolchain,
    CompilerDiagnostic,
    ResourceBound,
    InvalidImage,
    /// Bytes that contradict the digest identifying them.
    CorruptSource,
    Interrupted,
}

impl CompileFailureKind {
    /// Whether the same request fails this way again. Only a deterministic
    /// failure may be remembered as a failure; an absent toolchain or an
    /// interrupted worker says nothing about the request.
    pub fn is_deterministic(self) -> bool {
        match self {
            CompileFailureKind::IncompatibleTarget
            | CompileFailureKind::IncompatibleAssembly
            | CompileFailureKind::CompilerDiagnostic
            | CompileFailureKind::ResourceBound
            | CompileFailureKind::InvalidImage
            | CompileFailureKind::CorruptSource => true,
            CompileFailureKind::UnavailableToolchain | CompileFailureKind::Interrupted => false,
        }
    }

    /// Whether the failure is a property of the program the recipe's identity
    /// names. A missing toolchain and an interrupted worker are properties of
    /// this machine; a source whose bytes contradict their digest is not the
    /// program that digest names. A selection made around any of them is not
    /// the selection this program has elsewhere, so it must not be published
    /// under the program's identity — while a compiler diagnostic or a blown
    /// resource bound is exactly such a property and may be.
    pub fn is_program_property(self) -> bool {
        match self {
            CompileFailureKind::IncompatibleTarget
            | CompileFailureKind::IncompatibleAssembly
            | CompileFailureKind::CompilerDiagnostic
            | CompileFailureKind::ResourceBound
            | CompileFailureKind::InvalidImage => true,
            CompileFailureKind::UnavailableToolchain
            | CompileFailureKind::CorruptSource
            | CompileFailureKind::Interrupted => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileFailure {
    pub kind: CompileFailureKind,
    pub message: String,
}

/// Which compiles are worth comparing: the whole compile unit in link order by
/// *claimed* digest, the register cap that rewrites it, the target, the
/// assembler that consumes it, and whether this request may fall back to the
/// driver's PTX JIT when that assembler is absent.
///
/// A bucket, not an identity. An encoded source carries the digest its
/// artifact states, and nothing has checked it until the bytes are expanded —
/// so an equal key says only that two requests *might* be one compile, and
/// [`CompileTask::unit_sources`] is what settles it wherever an image would
/// cross from one request to another without being compiled.
///
/// `assembly_mode` is deliberately absent — it is how a producer *spells* a
/// request, not what the request produces. An absent mode and an explicit
/// `Relocatable` that resolve to one assembler must share one compile.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RequestKey {
    unit: Vec<[u8; 32]>,
    pub(crate) max_registers: Option<u32>,
    sm: u16,
    assembler: PtxAssembler,
    driver_jit_fallback: bool,
}

impl RequestKey {
    /// Whether `base`'s image realizes this request too, once a measurement
    /// proves this request's register cap non-binding: one compile unit, one
    /// target, one assembler, and `base` itself uncapped.
    ///
    /// The comparison is on the *resolved* assembler rather than on how each
    /// request spelled its `assembly_mode`, so cap reuse follows compile identity
    /// exactly as image sharing does. `driver_jit_fallback` is deliberately
    /// not compared: it decides how a *fresh* compile may degrade, while reuse
    /// hands over an image that already exists and whose registers were
    /// measured on that very image.
    ///
    /// Like the key itself this narrows rather than decides: the caller proves
    /// the two units hold the same bytes before handing the image over.
    pub(crate) fn reuses_uncapped(&self, base: &RequestKey) -> bool {
        base.max_registers.is_none()
            && self.unit == base.unit
            && self.sm == base.sm
            && self.assembler == base.assembler
    }
}

/// One compile, complete in itself: no borrow of the recipe it came from and no
/// device state, so producing it needs only these bytes and this target.
///
/// That self-containment is the seam a compile could one day cross — another
/// process, another host — and it is why [`CompileTask::produce`] is a function
/// of the task alone rather than a method on the resolver that scheduled it.
/// What a compile produced, in the terms the recipe model states images in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProductForm {
    /// Assembled for the architecture the task named.
    Cubin,
    /// Loadable in this session only — the driver's own JIT of the source.
    SessionLoadable,
}

impl ProductForm {
    pub(crate) fn persistence(self) -> ImagePersistence {
        match self {
            ProductForm::Cubin => ImagePersistence::Persistable,
            ProductForm::SessionLoadable => ImagePersistence::SessionOnly,
        }
    }
}

pub(crate) struct CompileTask {
    /// The unit's inputs in link order, one per distinct source.
    sources: Vec<Blob>,
    unit: Vec<[u8; 32]>,
    max_registers: Option<u32>,
    sm: u16,
    assembler: PtxAssembler,
    driver_jit_fallback: bool,
}

pub(crate) struct CompileProduct {
    pub(crate) image: Blob,
    /// What the produced bytes are, so the image built from them states its
    /// own compatibility and persistence rather than being recognised later.
    pub(crate) form: ProductForm,
    pub(crate) assembler: String,
    pub(crate) assembler_version: String,
    pub(crate) production_cost_ms: u64,
    pub(crate) wall_ms: u64,
    pub(crate) cache_source: Option<String>,
    pub(crate) unit_digest: [u8; 32],
}

impl CompileTask {
    pub(crate) fn for_request(
        request: &CompileRequest,
        sm: u16,
    ) -> Result<CompileTask, CompileFailure> {
        if request.sources.is_empty() {
            return Err(CompileFailure {
                kind: CompileFailureKind::InvalidImage,
                message: "compile request states no source to assemble".into(),
            });
        }
        // Canonicalize before anything is decided from the unit's size: the
        // assembler, the driver-JIT fallback and the artifact's assembler
        // identity all turn on whether this compile *links*, and a request
        // that names one module twice does not.
        let inputs = linked_inputs(request)?;
        let links = inputs.len() > 1;
        let assembler = assembler_for(request.options.assembly_mode, links)?;
        let (sources, unit) = inputs.into_iter().unzip();
        Ok(CompileTask {
            sources,
            unit,
            max_registers: request.options.max_registers,
            sm,
            assembler,
            // A request that constrains neither assembly nor registers can be
            // realized by the driver's own PTX JIT; one that states either
            // cannot, so the two never share an image. A unit that links
            // several modules cannot either: `cuModuleLoadData` assembles one
            // text and has nothing to link it against.
            driver_jit_fallback: request.options.assembly_mode
                == PtxAssemblyMode::EnvironmentDefault
                && request.options.max_registers.is_none()
                && !links,
        })
    }

    pub(crate) fn key(&self) -> RequestKey {
        RequestKey {
            unit: self.unit.clone(),
            max_registers: self.max_registers,
            sm: self.sm,
            assembler: self.assembler.clone(),
            driver_jit_fallback: self.driver_jit_fallback,
        }
    }

    /// The modules this compile links, in link order: what a request has to
    /// hold, byte for byte, to be this compile rather than merely to look like
    /// it in a key.
    pub(crate) fn unit_sources(&self) -> &[Blob] {
        &self.sources
    }

    pub(crate) fn produce(self) -> Result<CompileProduct, CompileFailure> {
        let started = Instant::now();
        let (image, form, assembler, assembler_version, production_cost_ms, cache_source) =
            match self.assembler.version() {
                Ok(assembler_version) => {
                    let compiled = Blob::with_all_decoded(&self.sources, |sources| {
                        cuda_toolchain::compile_ptx_unit_reported(
                            sources,
                            &self.assembler,
                            i32::from(self.sm),
                            cuda_toolchain::PtxCompileOptions {
                                max_registers: self.max_registers,
                            },
                        )
                    })
                    .map_err(corrupt_source)?
                    .map_err(|message| CompileFailure {
                        kind: CompileFailureKind::CompilerDiagnostic,
                        message,
                    })?;
                    (
                        Blob::from_vec(compiled.cubin),
                        ProductForm::Cubin,
                        self.assembler.name().to_string(),
                        assembler_version,
                        compiled.production_cost_ms,
                        Some(compiled.cache_source.name().to_string()),
                    )
                }
                Err(_) if self.driver_jit_fallback => {
                    let [source] = self.sources.as_slice() else {
                        unreachable!("only a one-module unit falls back to the driver's JIT")
                    };
                    let image = source
                        .with_decoded(|ptx| Blob::spilled_with_suffix(ptx, &[0]))
                        .map_err(corrupt_source)?
                        .map_err(interrupted_io)?;
                    (
                        image,
                        // The driver's JIT hands back the source in another
                        // form: keeping it would duplicate the source blob and
                        // promise a portability it does not have.
                        ProductForm::SessionLoadable,
                        "cuda-driver-jit".into(),
                        "driver-managed".into(),
                        0,
                        None,
                    )
                }
                Err(message) => {
                    return Err(CompileFailure {
                        kind: CompileFailureKind::UnavailableToolchain,
                        message,
                    });
                }
            };
        Ok(CompileProduct {
            image,
            form,
            assembler,
            assembler_version,
            production_cost_ms,
            wall_ms: started.elapsed().as_millis() as u64,
            cache_source,
            unit_digest: unit_digest(&self.unit),
        })
    }
}

/// What an image was compiled from, as one value: BLAKE3 over the unit's
/// source digests in link order, length-framed.
///
/// A digest per source would leave provenance a variable-length field for the
/// sake of a fact nothing reads apart; one digest over the ordered sequence
/// still distinguishes two units that differ in a module or in the order of
/// two, which is what an image's record of its own origin has to do.
fn unit_digest(sources: &[[u8; 32]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"kiln-compile-unit-v1\0");
    hasher.update(&(sources.len() as u64).to_le_bytes());
    for digest in sources {
        hasher.update(digest);
    }
    *hasher.finalize().as_bytes()
}

pub(crate) fn corrupt_source(error: recipe::BlobDecodeError) -> CompileFailure {
    CompileFailure {
        kind: CompileFailureKind::CorruptSource,
        message: error.to_string(),
    }
}

fn interrupted_io(error: std::io::Error) -> CompileFailure {
    CompileFailure {
        kind: CompileFailureKind::Interrupted,
        message: error.to_string(),
    }
}

/// The inputs a unit actually links: one per distinct source, in link order,
/// each paired with the digest that identifies it.
///
/// A unit that names the same module twice links it once — a second copy would
/// define every symbol in it again — so this sequence, and never the request's
/// stated source count, is what a compile is. Everything decided from the
/// unit's size reads it here.
///
/// Distinctness is decided on decoded bytes; a digest only narrows which
/// inputs are worth comparing. An encoded blob carries the digest its artifact
/// *claims*, which nothing has checked until the bytes are expanded — so
/// dropping an input because two digests agreed would let a forged source
/// disappear instead of being refused, and would merge two modules that
/// merely collided.
fn linked_inputs(request: &CompileRequest) -> Result<Vec<(Blob, [u8; 32])>, CompileFailure> {
    let mut inputs: Vec<(Blob, [u8; 32])> = Vec::with_capacity(request.sources.len());
    for source in &request.sources {
        let digest = source.blob.claimed_digest().map_err(corrupt_source)?;
        let mut repeats = false;
        for (kept, claimed) in &inputs {
            // Two *different* modules that define one symbol both reach the
            // linker; whether that is a collision is the linker's to say, and
            // it is the only thing that can.
            if *claimed == digest && kept.same_content(&source.blob).map_err(corrupt_source)? {
                repeats = true;
                break;
            }
        }
        if !repeats {
            inputs.push((source.blob.clone(), digest));
        }
    }
    Ok(inputs)
}

/// Whether two units link the same modules, in order, byte for byte.
///
/// This is what an equal [`RequestKey`] does not establish. A key carries each
/// source's claimed digest, so it says two compiles are worth comparing; only
/// the bytes say they are one compile, and an image may not cross from one
/// request to another on anything less.
pub(crate) fn same_unit(left: &[Blob], right: &[Blob]) -> Result<bool, CompileFailure> {
    if left.len() != right.len() {
        return Ok(false);
    }
    for (left, right) in left.iter().zip(right) {
        if !left.same_content(right).map_err(corrupt_source)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether `request` links more than one module — the fact its assembly policy
/// resolves against.
///
/// Bytes that cannot be expanded cannot be shown to repeat, so they count as
/// stated. That compile fails as a corrupt source, which is not a property of
/// the program, so nothing is published under an identity this answered for.
fn links_several(request: &CompileRequest) -> bool {
    linked_inputs(request).map_or(request.sources.len() > 1, |inputs| inputs.len() > 1)
}

/// Which assembler realizes a unit spelled `mode`, where `links` says the unit
/// holds more than one module.
///
/// A unit of several modules is a link, and only nvJitLink performs one here:
/// whole-program `ptxas` assembles one translation unit and cannot express a
/// link at all, so a request that states that policy over several modules is
/// refused rather than quietly assembled from one of them. An unstated policy
/// asks for whatever this machine does with the unit, and for several modules
/// that is the linker.
fn assembler_for(mode: PtxAssemblyMode, links: bool) -> Result<PtxAssembler, CompileFailure> {
    match mode {
        PtxAssemblyMode::Relocatable => Ok(PtxAssembler::NvJitLink),
        PtxAssemblyMode::WholeProgram if links => Err(CompileFailure {
            kind: CompileFailureKind::IncompatibleAssembly,
            message: "whole-program PTX assembly states one translation unit; this compile \
                      links several modules and needs a relocatable link"
                .into(),
        }),
        PtxAssemblyMode::WholeProgram => {
            cuda_toolchain::ptxas_assembler().ok_or_else(|| CompileFailure {
                kind: CompileFailureKind::UnavailableToolchain,
                message: "whole-program PTX assembly requires ptxas".into(),
            })
        }
        PtxAssemblyMode::EnvironmentDefault if links => Ok(PtxAssembler::NvJitLink),
        PtxAssemblyMode::EnvironmentDefault => Ok(cuda_toolchain::ptx_assembler()),
    }
}

/// Every assembler the recipe's requests can reach, as `name@version` or the
/// reason it is unreachable — a machine without `ptxas` eliminates every
/// whole-program candidate, and so realizes a different artifact. Empty for a
/// recipe with no compile request, which is what keeps a cubin-only
/// realization independent of any installed toolchain.
///
/// A policy is resolved together with whether its unit links, because that is
/// what resolves it: the same `EnvironmentDefault` reaches `ptxas` for one
/// module and the linker for several. That question is asked of the canonical
/// unit, so this states the assembler each request will really use.
pub(crate) fn reachable_assemblers<F: Launches>(recipe: &Recipe<F>) -> Vec<String> {
    let policies: BTreeSet<(PtxAssemblyMode, bool)> = recipe
        .context()
        .as_slice()
        .iter()
        .map(|request| (request.options.assembly_mode, links_several(request)))
        .collect();
    policies
        .into_iter()
        .map(|(mode, links)| match assembler_for(mode, links) {
            Ok(assembler) => match assembler.version() {
                Ok(version) => format!("{}@{version}", assembler.name()),
                Err(error) => format!("{}@unavailable:{error}", assembler.name()),
            },
            Err(failure) => format!("unavailable:{}", failure.message),
        })
        .collect()
}

/// Artifacts whose sources claim an identity they do not hold, for the tests
/// that prove kiln never acts on such a claim.
///
/// A forged artifact is written rather than constructed: an encoded payload's
/// digest is checked when its bytes are expanded and at no earlier point, so
/// overwriting one in a real artifact is exactly what an attacker can do and
/// exactly what decoding lets through.
#[cfg(test)]
pub(crate) mod forgery {
    use recipe::{
        Blob, CompileRequest, PtxCompileOptions, Recipe, RecipeBody, RecipeSignature, SourceImage,
        SourceKind,
    };

    /// Two PTX modules of equal decoded length, so one can be made to claim
    /// the other's identity by overwriting nothing but a digest.
    pub(crate) const CALLER: &[u8] = b".visible .entry caller() { call scale; }\n";
    pub(crate) const CALLEE: &[u8] = b".visible .func  scale()  { ret;        }\n";

    pub(crate) fn source(ptx: &[u8]) -> SourceImage {
        SourceImage {
            kind: SourceKind::Ptx,
            target: recipe::PtxTarget {
                isa: [8, 8],
                minimum_sm: 100,
            },
            blob: Blob::zstd_spilled(ptx).expect("a compressed source"),
        }
    }

    pub(crate) fn recipe_over(units: Vec<Vec<SourceImage>>) -> Recipe {
        Recipe::stating_edges(
            RecipeSignature::empty(),
            recipe::CompileRequests::new(
                units
                    .into_iter()
                    .map(|sources| CompileRequest::unguarded(sources, PtxCompileOptions::default()))
                    .collect(),
            ),
            RecipeBody::default(),
        )
        .unwrap()
    }

    /// Where pool entry `index`'s 32 digest bytes sit in a serialized
    /// artifact: past the four-byte pool count, then one entry's encoding
    /// byte, decoded length and digest, with its payload length after them.
    fn pool_digest_at(artifact: &[u8], index: usize) -> std::ops::Range<usize> {
        let word =
            |at: usize| u32::from_le_bytes(artifact[at..at + 4].try_into().unwrap()) as usize;
        let sections = word(8);
        let blobs = (0..sections)
            .map(|entry| 12 + entry * 20)
            .find(|&at| word(at) == 1)
            .map(|at| u64::from_le_bytes(artifact[at + 4..at + 12].try_into().unwrap()) as usize)
            .expect("the BLOBS section");
        let mut entry = blobs + 4;
        for _ in 0..index {
            entry += 45 + word(entry + 41);
        }
        entry + 9..entry + 41
    }

    /// `recipe`, written out with pool entry `forged`'s digest replaced by
    /// entry `claimed`'s, and read back.
    pub(crate) fn claiming_another_digest(
        recipe: &Recipe,
        claimed: usize,
        forged: usize,
    ) -> Recipe {
        let mut artifact = recipe::serialize(recipe).expect("a writable recipe");
        let stolen = artifact[pool_digest_at(&artifact, claimed)].to_vec();
        let forged = pool_digest_at(&artifact, forged);
        artifact[forged].copy_from_slice(&stolen);
        recipe::deserialize(&artifact)
            .expect("an encoded blob's claimed digest is not checked on decode")
    }
}

/// Apply a bound to the entries an image is expected to expose, requiring a
/// measurement for each. A bound that silently passes because nothing was
/// measured is the same as no bound at all — and this bound is what removes a
/// spilling register-capped candidate, so it must never pass on absent data:
/// neither on an entry nobody measured, nor on an empty entry set, which is
/// how a bound over an image no launch names would otherwise pass vacuously.
pub(crate) fn check_measured_bounds(
    bounds: ResourceBounds,
    entries: &BTreeSet<String>,
    measured: &std::collections::HashMap<String, KernelResources>,
) -> Result<(), CompileFailure> {
    if bounds == ResourceBounds::default() {
        return Ok(());
    }
    if entries.is_empty() {
        return Err(CompileFailure {
            kind: CompileFailureKind::InvalidImage,
            message: "no entry is launched from this image, so its resource bound \
                      states nothing that could be measured"
                .into(),
        });
    }
    let mut resources = Vec::with_capacity(entries.len());
    for entry in entries {
        resources.push(*measured.get(entry).ok_or_else(|| CompileFailure {
            kind: CompileFailureKind::InvalidImage,
            message: format!("no resource measurement for {entry}, so its bound cannot be applied"),
        })?);
    }
    check_bounds(bounds, resources.iter())
}

pub(crate) fn check_bounds<'a>(
    bounds: ResourceBounds,
    resources: impl IntoIterator<Item = &'a KernelResources>,
) -> Result<(), CompileFailure> {
    let mut min_registers_actual = u32::MAX;
    let mut max_local_bytes_actual = 0;
    for resources in resources {
        min_registers_actual = min_registers_actual.min(resources.registers);
        max_local_bytes_actual = max_local_bytes_actual.max(resources.local_bytes);
    }
    if let Some(min_registers) = bounds.min_registers
        && min_registers_actual < min_registers
    {
        return Err(CompileFailure {
            kind: CompileFailureKind::ResourceBound,
            message: format!(
                "register allocation {min_registers_actual} is below required {min_registers} registers"
            ),
        });
    }
    if let Some(max_local_bytes) = bounds.max_local_bytes
        && max_local_bytes_actual > max_local_bytes
    {
        return Err(CompileFailure {
            kind: CompileFailureKind::ResourceBound,
            message: format!(
                "local memory {max_local_bytes_actual} bytes exceeds admitted {max_local_bytes} bytes"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use recipe::{PtxCompileOptions, SourceImage, SourceKind};
    use std::collections::HashMap;

    fn request(max_registers: Option<u32>, assembly_mode: PtxAssemblyMode) -> CompileRequest {
        CompileRequest {
            sources: vec![SourceImage {
                kind: SourceKind::Ptx,
                target: recipe::PtxTarget {
                    isa: [8, 8],
                    minimum_sm: 100,
                },
                blob: Blob::from_vec(b"ptx".to_vec()),
            }],
            options: PtxCompileOptions {
                max_registers,
                assembly_mode,
            },
            requires: Default::default(),
        }
    }

    fn key(max_registers: Option<u32>, assembly_mode: PtxAssemblyMode) -> RequestKey {
        CompileTask::for_request(&request(max_registers, assembly_mode), 100)
            .unwrap()
            .key()
    }

    #[test]
    fn cap_reuse_follows_the_resolved_assembler_not_the_spelling() {
        let relocatable = PtxAssemblyMode::Relocatable;
        assert!(key(Some(128), relocatable).reuses_uncapped(&key(None, relocatable)));
        assert!(!key(Some(128), relocatable).reuses_uncapped(&key(Some(64), relocatable)));
        assert!(
            key(Some(128), PtxAssemblyMode::EnvironmentDefault)
                .reuses_uncapped(&key(None, PtxAssemblyMode::EnvironmentDefault))
        );
        // The environment default resolves to the process default assembler,
        // so it shares cap reuse with exactly the mode that names it.
        let default_is_nvjitlink = cuda_toolchain::ptx_assembler() == PtxAssembler::NvJitLink;
        assert_eq!(
            key(Some(128), PtxAssemblyMode::EnvironmentDefault)
                .reuses_uncapped(&key(None, relocatable)),
            default_is_nvjitlink
        );
    }

    fn unit(sources: &[&[u8]], assembly_mode: PtxAssemblyMode) -> CompileRequest {
        CompileRequest::unguarded(
            sources
                .iter()
                .map(|ptx| SourceImage {
                    kind: SourceKind::Ptx,
                    target: recipe::PtxTarget {
                        isa: [8, 8],
                        minimum_sm: 100,
                    },
                    blob: Blob::from_vec(ptx.to_vec()),
                })
                .collect(),
            PtxCompileOptions {
                max_registers: None,
                assembly_mode,
            },
        )
    }

    /// Only a linker realizes a unit of several modules. A whole-program
    /// policy states one translation unit, so over several it is refused —
    /// deterministically, and as a property of the request rather than of what
    /// this machine happens to have installed.
    #[test]
    fn a_link_refuses_a_whole_program_policy_and_accepts_the_environment_s() {
        let two: &[&[u8]] = &[b"caller", b"callee"];
        let Err(refused) = CompileTask::for_request(&unit(two, PtxAssemblyMode::WholeProgram), 100)
        else {
            panic!("a link has no whole-program spelling")
        };
        assert_eq!(refused.kind, CompileFailureKind::IncompatibleAssembly);
        assert!(refused.kind.is_deterministic() && refused.kind.is_program_property());
        // One module still assembles whole-program wherever ptxas exists, so
        // the refusal is about the link and not about the mode.
        assert!(
            CompileTask::for_request(&unit(&[b"caller"], PtxAssemblyMode::WholeProgram), 100)
                .is_ok()
                == cuda_toolchain::ptxas_assembler().is_some()
        );
        for mode in [
            PtxAssemblyMode::Relocatable,
            PtxAssemblyMode::EnvironmentDefault,
        ] {
            let task = CompileTask::for_request(&unit(two, mode), 100).expect("a link is realized");
            assert_eq!(task.assembler, PtxAssembler::NvJitLink, "{mode:?}");
            assert!(
                !task.driver_jit_fallback,
                "the driver's JIT assembles one text and links nothing"
            );
        }
    }

    /// The same module named twice is one input — a second copy would define
    /// every symbol in it again — while two different modules both reach the
    /// linker, whose business it is to say whether their symbols collide.
    #[test]
    fn a_unit_links_each_distinct_source_once() {
        let mode = PtxAssemblyMode::Relocatable;
        let repeated =
            CompileTask::for_request(&unit(&[b"caller", b"callee", b"caller"], mode), 100).unwrap();
        assert_eq!(repeated.sources.len(), 2);
        assert_eq!(
            repeated.key(),
            CompileTask::for_request(&unit(&[b"caller", b"callee"], mode), 100)
                .unwrap()
                .key(),
            "dropping a repeat leaves the same compile"
        );
        assert_ne!(
            repeated.key(),
            CompileTask::for_request(&unit(&[b"callee", b"caller"], mode), 100)
                .unwrap()
                .key(),
            "link order is part of what is compiled"
        );
    }

    /// A digest a blob merely *claims* is not the bytes it holds.
    ///
    /// An encoded payload carries the digest its artifact states, and nothing
    /// checks it until the bytes are expanded. Deciding a repeat on that claim
    /// would let a forged source be dropped from the unit — and the compile
    /// would then succeed, silently linking one module fewer than the recipe
    /// states, instead of refusing bytes that are not what they say.
    #[test]
    fn a_forged_digest_cannot_remove_an_input() {
        use super::forgery;

        let honest = forgery::recipe_over(vec![vec![
            forgery::source(forgery::CALLER),
            forgery::source(forgery::CALLEE),
        ]]);
        let forged = forgery::claiming_another_digest(&honest, 0, 1);

        let request = &forged.context().as_slice()[0];
        assert_eq!(
            request.sources[0].blob.claimed_digest().unwrap(),
            request.sources[1].blob.claimed_digest().unwrap(),
            "the second source now claims the first one's identity"
        );
        let Err(refused) = CompileTask::for_request(request, 100) else {
            panic!("a forged repeat must be refused, never dropped")
        };
        assert_eq!(refused.kind, CompileFailureKind::CorruptSource);

        // An honest repeat still collapses: content is what decides, so
        // refusing a forgery costs nothing that really was a repeat.
        let twice = forgery::recipe_over(vec![vec![
            forgery::source(forgery::CALLER),
            forgery::source(forgery::CALLER),
        ]]);
        let repeated = CompileTask::for_request(&twice.context().as_slice()[0], 100).unwrap();
        assert_eq!(repeated.sources.len(), 1);
    }

    /// A unit that names one module twice does not link, so every answer that
    /// turns on linking has to be given about the canonical unit and not about
    /// the request's stated source count.
    #[test]
    fn a_repeated_module_is_not_a_link() {
        let twice: &[&[u8]] = &[b"caller", b"caller"];
        let once: &[&[u8]] = &[b"caller"];

        // Whole-program assembly states one translation unit, and this is one.
        assert_eq!(
            CompileTask::for_request(&unit(twice, PtxAssemblyMode::WholeProgram), 100)
                .map(|task| task.assembler)
                .map_err(|failure| failure.kind),
            CompileTask::for_request(&unit(once, PtxAssemblyMode::WholeProgram), 100)
                .map(|task| task.assembler)
                .map_err(|failure| failure.kind),
        );

        // The environment's own assembler still answers, and the driver's JIT
        // is still a fallback: there is one text to hand it.
        let default =
            CompileTask::for_request(&unit(twice, PtxAssemblyMode::EnvironmentDefault), 100)
                .expect("one module twice is one module");
        assert_eq!(default.assembler, cuda_toolchain::ptx_assembler());
        assert!(default.driver_jit_fallback);
        assert_eq!(
            default.key(),
            CompileTask::for_request(&unit(once, PtxAssemblyMode::EnvironmentDefault), 100)
                .unwrap()
                .key(),
            "and it is the very same compile"
        );

        // What the artifact records as reachable agrees with what will run.
        let reachable = |sources: &[&[u8]]| {
            reachable_assemblers(
                &<Recipe>::stating_edges(
                    recipe::RecipeSignature::empty(),
                    recipe::CompileRequests::new(vec![unit(
                        sources,
                        PtxAssemblyMode::EnvironmentDefault,
                    )]),
                    recipe::RecipeBody::default(),
                )
                .unwrap(),
            )
        };
        assert_eq!(reachable(twice), reachable(once));
    }

    #[test]
    fn a_unit_states_at_least_one_source() {
        let empty = CompileRequest::unguarded(Vec::new(), PtxCompileOptions::default());
        let Err(refused) = CompileTask::for_request(&empty, 100) else {
            panic!("a unit with no source has nothing to assemble")
        };
        assert_eq!(refused.kind, CompileFailureKind::InvalidImage);
    }

    #[test]
    fn a_capped_request_never_owns_an_uncapped_image() {
        assert_ne!(
            key(Some(128), PtxAssemblyMode::EnvironmentDefault),
            key(None, PtxAssemblyMode::EnvironmentDefault)
        );
        assert!(
            !key(None, PtxAssemblyMode::EnvironmentDefault)
                .reuses_uncapped(&key(Some(128), PtxAssemblyMode::EnvironmentDefault))
        );
    }

    #[test]
    fn resource_bounds_are_producer_data() {
        let resources = [KernelResources {
            registers: 64,
            local_bytes: 32,
        }];
        assert!(check_bounds(ResourceBounds::default(), resources.iter()).is_ok());
        assert!(matches!(
            check_bounds(
                ResourceBounds {
                    max_local_bytes: Some(16),
                    ..ResourceBounds::default()
                },
                resources.iter()
            ),
            Err(CompileFailure {
                kind: CompileFailureKind::ResourceBound,
                ..
            })
        ));
        assert!(
            check_bounds(
                ResourceBounds {
                    min_registers: Some(64),
                    ..ResourceBounds::default()
                },
                resources.iter()
            )
            .is_ok()
        );
        assert!(matches!(
            check_bounds(
                ResourceBounds {
                    min_registers: Some(65),
                    ..ResourceBounds::default()
                },
                resources.iter()
            ),
            Err(CompileFailure {
                kind: CompileFailureKind::ResourceBound,
                ..
            })
        ));
    }

    #[test]
    fn resource_bound_requires_every_launched_entry_to_be_measured() {
        let bounds = ResourceBounds {
            max_local_bytes: Some(0),
            ..ResourceBounds::default()
        };
        assert!(matches!(
            check_measured_bounds(
                bounds,
                &BTreeSet::from(["kernel".to_string()]),
                &HashMap::new()
            ),
            Err(CompileFailure {
                kind: CompileFailureKind::InvalidImage,
                ..
            })
        ));
    }

    /// The hole a raw `check_bounds` leaves: with nothing to measure, `max`
    /// over an empty set is zero and every bound passes. A bound whose entry
    /// set is empty is a bound over an image no launch names, so it cannot be
    /// applied and must not stand in for having been applied.
    #[test]
    fn a_bound_over_no_entry_at_all_is_not_a_satisfied_bound() {
        let bounds = ResourceBounds {
            max_local_bytes: Some(0),
            ..ResourceBounds::default()
        };
        assert!(
            check_bounds(bounds, std::iter::empty()).is_ok(),
            "the raw check has nothing to compare and says so"
        );
        assert!(matches!(
            check_measured_bounds(bounds, &BTreeSet::new(), &HashMap::new()),
            Err(CompileFailure {
                kind: CompileFailureKind::InvalidImage,
                ..
            })
        ));
        // A bound that states nothing needs no measurement, empty or not.
        assert!(
            check_measured_bounds(ResourceBounds::default(), &BTreeSet::new(), &HashMap::new())
                .is_ok()
        );
    }

    /// A register floor reaches this contract on its own, so a bound stating
    /// only a floor is applied rather than passed over for stating nothing
    /// about local memory.
    #[test]
    fn a_register_floor_alone_is_a_bound_that_must_be_measured() {
        let bounds = ResourceBounds {
            min_registers: Some(64),
            max_local_bytes: None,
        };
        assert_ne!(bounds, ResourceBounds::default());
        let entries = BTreeSet::from(["kernel".to_string()]);
        let measured = |registers| {
            HashMap::from([(
                "kernel".to_string(),
                KernelResources {
                    registers,
                    local_bytes: 0,
                },
            )])
        };
        assert!(check_measured_bounds(bounds, &entries, &measured(64)).is_ok());
        assert!(matches!(
            check_measured_bounds(bounds, &entries, &measured(32)),
            Err(CompileFailure {
                kind: CompileFailureKind::ResourceBound,
                ..
            })
        ));
        // The same floor over an image no launch names, and over one nobody
        // measured, is unevaluated rather than met.
        assert!(matches!(
            check_measured_bounds(bounds, &BTreeSet::new(), &measured(64)),
            Err(CompileFailure {
                kind: CompileFailureKind::InvalidImage,
                ..
            })
        ));
        assert!(matches!(
            check_measured_bounds(bounds, &entries, &HashMap::new()),
            Err(CompileFailure {
                kind: CompileFailureKind::InvalidImage,
                ..
            })
        ));
    }

    #[test]
    fn a_failure_this_machine_caused_is_not_the_program_s() {
        for kind in [
            CompileFailureKind::CompilerDiagnostic,
            CompileFailureKind::ResourceBound,
            CompileFailureKind::IncompatibleTarget,
            CompileFailureKind::InvalidImage,
        ] {
            assert!(kind.is_program_property(), "{kind:?}");
            assert!(kind.is_deterministic(), "{kind:?}");
        }
        for kind in [
            CompileFailureKind::UnavailableToolchain,
            CompileFailureKind::Interrupted,
        ] {
            assert!(!kind.is_program_property(), "{kind:?}");
            assert!(!kind.is_deterministic(), "{kind:?}");
        }
        // Bytes that contradict their digest fail the same way every time,
        // which makes the failure rememberable — but they are not the program
        // the digest identifies, so nothing may be published under it.
        assert!(CompileFailureKind::CorruptSource.is_deterministic());
        assert!(!CompileFailureKind::CorruptSource.is_program_property());
    }

    #[test]
    fn a_recipe_without_compile_requests_reaches_no_assembler() {
        assert!(
            reachable_assemblers(
                &<Recipe>::stating_edges(
                    recipe::RecipeSignature::empty(),
                    recipe::CompileRequests::default(),
                    recipe::RecipeBody::default()
                )
                .unwrap()
            )
            .is_empty()
        );
    }
}
