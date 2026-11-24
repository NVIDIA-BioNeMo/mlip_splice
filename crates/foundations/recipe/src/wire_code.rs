// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How a module's code is encoded: compile requests, modules, and images.
//!
//! Everything in [`crate::code`], as bytes. Payloads are pool indices — the
//! bytes themselves live once in BLOBS — so what is here is the shape: which
//! compile a module names, which images it offers and in what order, what each
//! image is for, and what its kernels must fit.

use super::wire_io::{Cursor, Writer};
use super::{Pool, WireError, checked_u32};
use crate::Blob;
use crate::code::{
    CompileProvenance, CompileRequest, CompileRequestId, ImagePersistence, KernelCode,
    KernelModule, ModuleImage, PtxAssemblyMode, PtxCompileOptions, PtxTarget, ResourceBounds,
    SourceImage, SourceKind,
};
use crate::payloads::PayloadRole;

pub(super) fn write_resource_bounds(w: &mut Writer, bounds: ResourceBounds) {
    write_optional_u32(w, bounds.min_registers);
    write_optional_u32(w, bounds.max_local_bytes);
}

pub(super) fn read_resource_bounds(c: &mut Cursor) -> Result<ResourceBounds, WireError> {
    Ok(ResourceBounds {
        min_registers: read_optional_u32(c)?,
        max_local_bytes: read_optional_u32(c)?,
    })
}

fn write_provenance(w: &mut Writer, provenance: &CompileProvenance) -> Result<(), WireError> {
    w.bytes.extend_from_slice(&provenance.unit_digest);
    write_compile_options(w, provenance.options);
    w.u16(provenance.target_sm);
    w.str(&provenance.assembler)?;
    w.str(&provenance.assembler_version)?;
    write_optional_u32(w, provenance.driver_version);
    write_optional_u32(w, provenance.toolkit_version);
    Ok(())
}

fn read_provenance(c: &mut Cursor) -> Result<CompileProvenance, WireError> {
    let unit_digest = c.take(32)?.try_into().expect("32-byte digest");
    Ok(CompileProvenance {
        unit_digest,
        options: read_compile_options(c)?,
        target_sm: c.u16()?,
        assembler: c.str()?,
        assembler_version: c.str()?,
        driver_version: read_optional_u32(c)?,
        toolkit_version: read_optional_u32(c)?,
    })
}

fn write_optional_provenance(
    w: &mut Writer,
    provenance: Option<&CompileProvenance>,
) -> Result<(), WireError> {
    match provenance {
        Some(provenance) => {
            w.u8(1);
            write_provenance(w, provenance)
        }
        None => {
            w.u8(0);
            Ok(())
        }
    }
}

fn read_optional_provenance(c: &mut Cursor) -> Result<Option<CompileProvenance>, WireError> {
    match c.u8()? {
        0 => Ok(None),
        1 => Ok(Some(read_provenance(c)?)),
        tag => Err(WireError::BadTag {
            what: "compile provenance",
            tag,
        }),
    }
}

fn write_optional_u32(w: &mut Writer, value: Option<u32>) {
    match value {
        Some(value) => {
            w.u8(1);
            w.u32(value);
        }
        None => w.u8(0),
    }
}

fn read_optional_u32(c: &mut Cursor) -> Result<Option<u32>, WireError> {
    match c.u8()? {
        0 => Ok(None),
        1 => Ok(Some(c.u32()?)),
        tag => Err(WireError::BadTag {
            what: "option",
            tag,
        }),
    }
}

fn write_compile_options(w: &mut Writer, options: PtxCompileOptions) {
    write_optional_u32(w, options.max_registers);
    w.u8(match options.assembly_mode {
        PtxAssemblyMode::EnvironmentDefault => 0,
        PtxAssemblyMode::Relocatable => 1,
        PtxAssemblyMode::WholeProgram => 2,
    });
}

fn read_compile_options(c: &mut Cursor) -> Result<PtxCompileOptions, WireError> {
    let max_registers = read_optional_u32(c)?;
    let assembly_mode = match c.u8()? {
        0 => PtxAssemblyMode::EnvironmentDefault,
        1 => PtxAssemblyMode::Relocatable,
        2 => PtxAssemblyMode::WholeProgram,
        tag => {
            return Err(WireError::BadTag {
                what: "assembly mode",
                tag,
            });
        }
    };
    Ok(PtxCompileOptions {
        max_registers,
        assembly_mode,
    })
}

fn write_image<'a>(
    w: &mut Writer,
    pool: &mut Pool<'a>,
    image: &'a ModuleImage,
) -> Result<(), WireError> {
    match image {
        ModuleImage::Cubin {
            target_sm,
            bytes,
            provenance,
        } => {
            w.u8(0);
            w.u16(*target_sm);
            w.u32(pool.index_of(bytes, PayloadRole::ModuleImage)?);
            write_optional_provenance(w, provenance.as_ref())
        }
        ModuleImage::CudaLoadable {
            bytes,
            persistence,
            provenance,
        } => {
            w.u8(1);
            w.u8(match persistence {
                ImagePersistence::Persistable => 0,
                ImagePersistence::SessionOnly => 1,
            });
            w.u32(pool.index_of(bytes, PayloadRole::ModuleImage)?);
            write_optional_provenance(w, provenance.as_ref())
        }
    }
}

fn read_image(c: &mut Cursor, pool: &[Blob]) -> Result<ModuleImage, WireError> {
    match c.u8()? {
        0 => Ok(ModuleImage::Cubin {
            target_sm: c.u16()?,
            bytes: super::wire_body::pool_blob(c, pool)?,
            provenance: read_optional_provenance(c)?,
        }),
        1 => {
            let persistence = match c.u8()? {
                0 => ImagePersistence::Persistable,
                1 => ImagePersistence::SessionOnly,
                tag => {
                    return Err(WireError::BadTag {
                        what: "image persistence",
                        tag,
                    });
                }
            };
            Ok(ModuleImage::CudaLoadable {
                bytes: super::wire_body::pool_blob(c, pool)?,
                persistence,
                provenance: read_optional_provenance(c)?,
            })
        }
        tag => Err(WireError::BadTag {
            what: "module image",
            tag,
        }),
    }
}

fn write_images<'a>(
    w: &mut Writer,
    pool: &mut Pool<'a>,
    images: impl ExactSizeIterator<Item = &'a ModuleImage>,
) -> Result<(), WireError> {
    w.count(images.len())?;
    for image in images {
        write_image(w, pool, image)?;
    }
    Ok(())
}

fn read_images(c: &mut Cursor, pool: &[Blob]) -> Result<Vec<ModuleImage>, WireError> {
    (0..c.bounded_count(stride::IMAGE)?)
        .map(|_| read_image(c, pool))
        .collect()
}

pub(super) fn write_module<'a>(
    w: &mut Writer,
    pool: &mut Pool<'a>,
    module: &'a KernelModule,
) -> Result<(), WireError> {
    match &module.code {
        KernelCode::PtxBacked {
            compile_request,
            images,
        } => {
            w.u8(0);
            w.u32(checked_u32(compile_request.0, "compile request index")?);
            write_images(w, pool, images.iter())?;
        }
        KernelCode::ImageOnly {
            preferred_image,
            alternative_images,
        } => {
            w.u8(1);
            write_image(w, pool, preferred_image)?;
            write_images(w, pool, alternative_images.iter())?;
        }
    }
    write_resource_bounds(w, module.resource_bounds);
    Ok(())
}

pub(super) fn read_module(c: &mut Cursor, pool: &[Blob]) -> Result<KernelModule, WireError> {
    let code = match c.u8()? {
        0 => KernelCode::PtxBacked {
            compile_request: CompileRequestId(c.u32()? as usize),
            images: read_images(c, pool)?,
        },
        1 => KernelCode::ImageOnly {
            preferred_image: read_image(c, pool)?,
            alternative_images: read_images(c, pool)?,
        },
        tag => {
            return Err(WireError::BadTag {
                what: "kernel code",
                tag,
            });
        }
    };
    Ok(KernelModule {
        code,
        resource_bounds: read_resource_bounds(c)?,
    })
}

/// The unit a compile links, in link order: one entry per source, each stating
/// what it is, what it needs of the machine, and where its bytes are pooled.
fn write_sources<'a>(
    w: &mut Writer,
    pool: &mut Pool<'a>,
    sources: &'a [SourceImage],
) -> Result<(), WireError> {
    w.count(sources.len())?;
    for source in sources {
        match source.kind {
            SourceKind::Ptx => w.u8(0),
        }
        w.u8(source.target.isa[0]);
        w.u8(source.target.isa[1]);
        w.u16(source.target.minimum_sm);
        w.u32(pool.index_of(&source.blob, PayloadRole::CompileSource)?);
    }
    Ok(())
}

fn read_sources(c: &mut Cursor, pool: &[Blob]) -> Result<Vec<SourceImage>, WireError> {
    (0..c.bounded_count(stride::SOURCE)?)
        .map(|_| {
            let kind = match c.u8()? {
                0 => SourceKind::Ptx,
                tag => {
                    return Err(WireError::BadTag {
                        what: "source kind",
                        tag,
                    });
                }
            };
            let target = PtxTarget {
                isa: [c.u8()?, c.u8()?],
                minimum_sm: c.u16()?,
            };
            Ok(SourceImage {
                kind,
                target,
                blob: super::wire_body::pool_blob(c, pool)?,
            })
        })
        .collect()
}

pub(super) fn write_request<'a>(
    w: &mut Writer,
    pool: &mut Pool<'a>,
    request: &'a CompileRequest,
) -> Result<(), WireError> {
    write_sources(w, pool, &request.sources)?;
    write_compile_options(w, request.options);
    // The map's own order, which is by request: one entry per prerequisite and
    // the same bytes however the producer inserted them.
    w.count(request.requires.len())?;
    for (prerequisite, bounds) in &request.requires {
        w.u32(checked_u32(prerequisite.0, "compile request index")?);
        write_resource_bounds(w, *bounds);
    }
    Ok(())
}

pub(super) fn read_request(c: &mut Cursor, pool: &[Blob]) -> Result<CompileRequest, WireError> {
    let sources = read_sources(c, pool)?;
    let options = read_compile_options(c)?;
    let mut requires = std::collections::BTreeMap::new();
    let mut previous: Option<CompileRequestId> = None;
    for _ in 0..c.bounded_count(stride::PREREQUISITE)? {
        let prerequisite = CompileRequestId(c.u32()? as usize);
        // Ascending and without repeats is what the map is; bytes that say
        // otherwise are two statements about one request, and a reader that
        // took the last would disagree with one that took the first.
        if previous.is_some_and(|previous| previous >= prerequisite) {
            return Err(WireError::NonCanonical {
                what: "compile prerequisites",
            });
        }
        previous = Some(prerequisite);
        requires.insert(prerequisite, read_resource_bounds(c)?);
    }
    Ok(CompileRequest {
        sources,
        options,
        requires,
    })
}

/// The least a single element of each counted sequence can occupy. A count
/// comes from untrusted bytes and immediately sizes an allocation, so it is
/// refused unless the section still holds the bytes its elements need.
mod stride {
    /// An image tag, a blob index, and a provenance tag.
    pub(super) const IMAGE: usize = 6;
    /// A `u32` request plus two optional `u32`, a tag byte each when absent.
    pub(super) const PREREQUISITE: usize = 6;
    /// A source kind, a two-byte PTX ISA, a `u16` minimum SM, a blob index.
    pub(super) const SOURCE: usize = 9;
}
