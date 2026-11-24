// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a module's machine code is, and where it comes from.
//!
//! A module either names a compile the root table states, or carries images
//! and names no compile — [`KernelCode`] is that choice, and having it be a
//! choice is what makes "a module with neither" unwritable rather than
//! something a validator has to look for.
//!
//! Where a module carries more than one image, their order is the priority the
//! runtime honours: the first one this machine can load wins, and compiling is
//! what happens when none can. An image-only module states its preferred image
//! apart from its alternatives, so that privilege is in the type rather than in
//! a convention about index zero.
//!
//! An image says exactly what it is. A cubin is assembled for one architecture
//! and names it; a CUDA-loadable image is whatever the loader accepts and says
//! instead whether its bytes outlive the session that produced them. Neither
//! answer is inferred from an assembler's name.

use std::collections::BTreeMap;

use crate::blob::Blob;

/// Position in the root compile table. The table is the recipe's, so an id is
/// meaningless outside the recipe that states it and is remapped when recipes
/// merge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompileRequestId(pub usize);

/// One PTX compile, and every resource fact that must already hold before it
/// is worth starting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileRequest {
    /// Every source this one compile links, in link order.
    ///
    /// A sequence rather than a set: the linker is handed these modules as the
    /// separate inputs they are, and which one comes first is a fact about the
    /// compile, so two orders are two compiles. It is never empty — a compile
    /// with nothing to assemble is not a compile — and
    /// [`crate::ExecutableRefusal::EmptyCompileUnit`] is what says so.
    pub sources: Vec<SourceImage>,
    pub options: PtxCompileOptions,
    /// A conjunction of guards, keyed by the request each one is about.
    ///
    /// This is a map because a prerequisite is a statement *about* another
    /// request, and two statements about the same request are one statement
    /// with two halves. Keying by request makes the second one an update
    /// rather than a duplicate the reader has to reconcile, and gives the
    /// wire one canonical order.
    pub requires: BTreeMap<CompileRequestId, ResourceBounds>,
}

impl CompileRequest {
    /// A compile nothing has to hold before it starts.
    pub fn unguarded(sources: Vec<SourceImage>, options: PtxCompileOptions) -> Self {
        CompileRequest {
            sources,
            options,
            requires: BTreeMap::new(),
        }
    }

    /// The lowest architecture that runs every source of this unit: a link is
    /// only as portable as its least portable input.
    pub fn minimum_sm(&self) -> u16 {
        self.sources
            .iter()
            .map(|source| source.target.minimum_sm)
            .max()
            .unwrap_or(0)
    }

    /// Guard this compile on `request` resolving within `bounds`, keeping
    /// whatever was already required of that request.
    ///
    /// Two guards about one request are one guard that says both, so this
    /// conjoins rather than replaces. Adding a guard by inserting into the map
    /// would drop the one already there — and after interning renumbers a
    /// table, two guards that named different requests can turn out to name
    /// the same one.
    pub fn require(&mut self, request: CompileRequestId, bounds: ResourceBounds) {
        let stated = self.requires.entry(request).or_default();
        *stated = stated.and(bounds);
    }
}

/// The guards `stated` amounts to, with two about one request kept as the one
/// guard that says both.
///
/// Every path that renumbers or re-collects prerequisites comes through here,
/// so none of them can lose a constraint by overwriting an entry it did not
/// know was already there.
pub fn conjoined_requires(
    stated: impl IntoIterator<Item = (CompileRequestId, ResourceBounds)>,
) -> BTreeMap<CompileRequestId, ResourceBounds> {
    let mut requires: BTreeMap<CompileRequestId, ResourceBounds> = BTreeMap::new();
    for (request, bounds) in stated {
        let entry = requires.entry(request).or_default();
        *entry = entry.and(bounds);
    }
    requires
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceImage {
    pub kind: SourceKind,
    pub target: PtxTarget,
    pub blob: Blob,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SourceKind {
    Ptx,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PtxTarget {
    pub isa: [u8; 2],
    pub minimum_sm: u16,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PtxCompileOptions {
    pub max_registers: Option<u32>,
    pub assembly_mode: PtxAssemblyMode,
}

/// How the PTX is to be assembled.
///
/// Stating nothing is its own mode rather than the absence of one: a producer
/// that does not care is asking for whatever this machine assembles with, and
/// that is a decision the recipe records, not a hole in it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PtxAssemblyMode {
    #[default]
    EnvironmentDefault,
    Relocatable,
    WholeProgram,
}

/// What a kernel launch is executed from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelCode {
    /// A compile in the root table, with images to try before running it.
    ///
    /// `images` may be empty — that is a module which must be compiled — and
    /// when it is not, the first image this machine accepts is used and the
    /// compile is the fallback.
    PtxBacked {
        compile_request: CompileRequestId,
        images: Vec<ModuleImage>,
    },
    /// Images and nothing else, so at least one must be present and one of
    /// them is preferred. A module of this shape has no compile to fall back
    /// on, which is why the preference is stated rather than implied.
    ImageOnly {
        preferred_image: ModuleImage,
        alternative_images: Vec<ModuleImage>,
    },
}

impl KernelCode {
    pub fn compile_request(&self) -> Option<CompileRequestId> {
        match self {
            KernelCode::PtxBacked {
                compile_request, ..
            } => Some(*compile_request),
            KernelCode::ImageOnly { .. } => None,
        }
    }

    /// Every image this code offers, most preferred first. Order is the
    /// contract: a caller walks it and takes the first it can use.
    pub fn images(&self) -> impl Iterator<Item = &ModuleImage> {
        let (first, rest) = match self {
            KernelCode::PtxBacked { images, .. } => (None, images.as_slice()),
            KernelCode::ImageOnly {
                preferred_image,
                alternative_images,
            } => (Some(preferred_image), alternative_images.as_slice()),
        };
        first.into_iter().chain(rest)
    }

    pub fn images_mut(&mut self) -> impl Iterator<Item = &mut ModuleImage> {
        let (first, rest) = match self {
            KernelCode::PtxBacked { images, .. } => (None, images.as_mut_slice()),
            KernelCode::ImageOnly {
                preferred_image,
                alternative_images,
            } => (Some(preferred_image), alternative_images.as_mut_slice()),
        };
        first.into_iter().chain(rest)
    }

    /// The one image this code is, when it is exactly one and there is no
    /// compile behind it — what a loader can act on without choosing.
    pub fn sole_image(&self) -> Option<&ModuleImage> {
        match self {
            KernelCode::ImageOnly {
                preferred_image,
                alternative_images,
            } if alternative_images.is_empty() => Some(preferred_image),
            KernelCode::PtxBacked { .. } | KernelCode::ImageOnly { .. } => None,
        }
    }
}

/// Loadable bytes, and the fact that decides whether this machine can use
/// them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModuleImage {
    /// Assembled for exactly one architecture. A cubin is not compatible with
    /// a device it does not name, so the architecture is a field and not a
    /// description to be matched loosely.
    Cubin {
        target_sm: u16,
        bytes: Blob,
        provenance: Option<CompileProvenance>,
    },
    /// Anything `cuModuleLoadData` accepts — a fatbin, a cubin the producer
    /// did not label, PTX the driver will assemble. Compatibility is the
    /// loader's to decide, so what the recipe states instead is whether the
    /// bytes are worth keeping.
    CudaLoadable {
        bytes: Blob,
        persistence: ImagePersistence,
        provenance: Option<CompileProvenance>,
    },
}

impl ModuleImage {
    pub fn bytes(&self) -> &Blob {
        match self {
            ModuleImage::Cubin { bytes, .. } | ModuleImage::CudaLoadable { bytes, .. } => bytes,
        }
    }

    pub fn bytes_mut(&mut self) -> &mut Blob {
        match self {
            ModuleImage::Cubin { bytes, .. } | ModuleImage::CudaLoadable { bytes, .. } => bytes,
        }
    }

    pub fn provenance(&self) -> Option<&CompileProvenance> {
        match self {
            ModuleImage::Cubin { provenance, .. }
            | ModuleImage::CudaLoadable { provenance, .. } => provenance.as_ref(),
        }
    }

    /// Whether these bytes are worth writing into an artifact another process
    /// will load. A cubin always is; a loadable image says so itself.
    pub fn is_persistable(&self) -> bool {
        match self {
            ModuleImage::Cubin { .. } => true,
            ModuleImage::CudaLoadable { persistence, .. } => {
                *persistence == ImagePersistence::Persistable
            }
        }
    }
}

/// Whether an image outlives the session that produced it.
///
/// The driver's own PTX JIT hands back something loadable, but it is the
/// source again in another form: storing it as a persistable image would
/// duplicate the source blob and promise a portability it does not have. That is a
/// property of how the bytes were made, so the value carries it and no
/// consumer has to recognise an assembler by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImagePersistence {
    Persistable,
    SessionOnly,
}

/// How an image was produced. Diagnostic and cache identity; compatibility is
/// decided by the image's own fields or by the loader, never by these strings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileProvenance {
    /// What was compiled: a digest over the whole compile unit, not over one
    /// of its sources, so an image records every module that was linked into
    /// it and the order they were linked in.
    pub unit_digest: [u8; 32],
    pub options: PtxCompileOptions,
    pub target_sm: u16,
    pub assembler: String,
    pub assembler_version: String,
    pub driver_version: Option<u32>,
    pub toolkit_version: Option<u32>,
}

/// What a kernel must fit for an image to be usable for it.
///
/// A `None` states no constraint on its coordinate, so the default bounds
/// admit everything.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ResourceBounds {
    pub min_registers: Option<u32>,
    pub max_local_bytes: Option<u32>,
}

impl ResourceBounds {
    /// Both bounds at once: a kernel satisfies this exactly when it satisfies
    /// each of the two.
    ///
    /// A floor conjoins by taking the higher and a ceiling by taking the
    /// lower. An absent bound constrains nothing, so it is the identity for
    /// its own coordinate rather than a floor of zero or a ceiling of none.
    pub fn and(self, other: ResourceBounds) -> ResourceBounds {
        ResourceBounds {
            min_registers: tighter(self.min_registers, other.min_registers, u32::max),
            max_local_bytes: tighter(self.max_local_bytes, other.max_local_bytes, u32::min),
        }
    }
}

fn tighter(stated: Option<u32>, other: Option<u32>, pick: fn(u32, u32) -> u32) -> Option<u32> {
    match (stated, other) {
        (Some(stated), Some(other)) => Some(pick(stated, other)),
        (bound, None) | (None, bound) => bound,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelModule {
    pub code: KernelCode,
    /// What the module's kernels must fit, whichever image or compile answers.
    pub resource_bounds: ResourceBounds,
}

impl KernelModule {
    /// A module that is one loadable image, kept across sessions.
    pub fn from_cuda_loadable(bytes: impl Into<Blob>) -> Self {
        KernelModule {
            code: KernelCode::ImageOnly {
                preferred_image: ModuleImage::CudaLoadable {
                    bytes: bytes.into(),
                    persistence: ImagePersistence::Persistable,
                    provenance: None,
                },
                alternative_images: Vec::new(),
            },
            resource_bounds: ResourceBounds::default(),
        }
    }

    /// A module compiled from `request`, with `images` to try first.
    pub fn from_request(
        compile_request: CompileRequestId,
        images: Vec<ModuleImage>,
        resource_bounds: ResourceBounds,
    ) -> Self {
        KernelModule {
            code: KernelCode::PtxBacked {
                compile_request,
                images,
            },
            resource_bounds,
        }
    }

    pub fn sole_image(&self) -> Option<&ModuleImage> {
        self.code.sole_image()
    }
}

#[cfg(test)]
#[path = "code_tests.rs"]
mod tests;
