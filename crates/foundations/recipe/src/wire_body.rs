// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The BODY and REQUESTS sections: how the structure of a recipe is encoded.
//!
//! Everything a recipe *is* — its nodes, the buffers they bind, the modules
//! they launch, and the compile requests those modules resolve from — as
//! opposed to the artifact that carries it, which is [`super`]. Payload bytes
//! never appear here: they live once in the blob pool and this codec moves
//! pool indices. A Choice candidate is another body, so the encoding recurses.

use std::collections::BTreeSet;

use super::wire_code::{read_module, write_module};
use super::wire_io::{Cursor, Writer};
use super::wire_signature::{read_aliases, write_aliases};
use super::{Pool, WireError, checked_u32, decode_nbytes, encode_nbytes};
use crate::algebra::{Geometry, KernelLaunch, Work};
use crate::payloads::PayloadRole;
use crate::{
    Address, Arg, Blob, BufRef, Node, RecipeBody, RecipeError, Segment, WorkspaceBuf, WritableBuf,
};

fn write_bufref(w: &mut Writer, buf: BufRef) -> Result<(), WireError> {
    let tag = match buf {
        BufRef::Input(_) => 0,
        BufRef::Output(_) => 1,
        BufRef::Workspace(_) => 2,
    };
    w.u8(tag);
    w.u32(checked_u32(buf.index(), "bufref index")?);
    Ok(())
}

fn read_bufref(c: &mut Cursor) -> Result<BufRef, WireError> {
    let tag = c.u8()?;
    let i = c.u32()? as usize;
    match tag {
        0 => Ok(BufRef::Input(i)),
        1 => Ok(BufRef::Output(i)),
        2 => Ok(BufRef::Workspace(i)),
        tag => Err(WireError::BadTag {
            what: "bufref",
            tag,
        }),
    }
}

fn write_writablebuf(w: &mut Writer, buf: WritableBuf) -> Result<(), WireError> {
    let tag = match buf {
        WritableBuf::Output(_) => 0,
        WritableBuf::Workspace(_) => 1,
    };
    w.u8(tag);
    w.u32(checked_u32(buf.index(), "writablebuf index")?);
    Ok(())
}

fn read_writablebuf(c: &mut Cursor) -> Result<WritableBuf, WireError> {
    let tag = c.u8()?;
    match tag {
        0 => Ok(WritableBuf::Output(c.u32()? as usize)),
        1 => Ok(WritableBuf::Workspace(c.u32()? as usize)),
        tag => Err(WireError::BadTag {
            what: "writablebuf",
            tag,
        }),
    }
}

/// An address is its buffer followed by a `u64` byte offset, whichever
/// capability the buffer carries.
fn write_address(w: &mut Writer, address: Address<BufRef>) -> Result<(), WireError> {
    write_bufref(w, address.buf)?;
    w.u64(encode_nbytes(address.byte_offset, "address byte offset")?);
    Ok(())
}

fn read_address(c: &mut Cursor) -> Result<Address<BufRef>, WireError> {
    let buf = read_bufref(c)?;
    Ok(buf.at(decode_nbytes(c.u64()?, "address byte offset")?))
}

fn write_writable_address(w: &mut Writer, address: Address<WritableBuf>) -> Result<(), WireError> {
    write_writablebuf(w, address.buf)?;
    w.u64(encode_nbytes(address.byte_offset, "address byte offset")?);
    Ok(())
}

fn read_writable_address(c: &mut Cursor) -> Result<Address<WritableBuf>, WireError> {
    let buf = read_writablebuf(c)?;
    Ok(buf.at(decode_nbytes(c.u64()?, "address byte offset")?))
}

fn write_arg(w: &mut Writer, arg: &Arg) -> Result<(), WireError> {
    w.count(arg.segments().len())?;
    for segment in arg.segments() {
        match segment {
            Segment::Bytes(bytes) => {
                w.u8(0);
                w.blob(bytes)?;
            }
            Segment::Pointer(address) => {
                w.u8(1);
                write_address(w, *address)?;
            }
        }
    }
    Ok(())
}

fn read_arg(c: &mut Cursor) -> Result<Arg, WireError> {
    let n = c.bounded_count(stride::SEGMENT)?;
    let segments = (0..n)
        .map(|_| match c.u8()? {
            0 => Ok(Segment::Bytes(c.blob()?.to_vec())),
            1 => Ok(Segment::Pointer(read_address(c)?)),
            tag => Err(WireError::BadTag {
                what: "segment",
                tag,
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    // An empty byte run, or two adjacent ones, is a second spelling of an
    // argument that already has one. Joining them here would map two byte
    // strings onto one value and break the round trip, so the reader refuses
    // instead — the same answer this format gives everywhere else it states a
    // canonical form.
    if !Arg::is_canonical(&segments) {
        return Err(WireError::NonCanonical {
            what: "argument segments",
        });
    }
    Ok(Arg::from_segments(segments))
}

/// A node's dependencies, in the set's own ascending order.
fn write_deps(w: &mut Writer, deps: &BTreeSet<usize>) -> Result<(), WireError> {
    w.count(deps.len())?;
    for &d in deps {
        w.u32(checked_u32(d, "deps entry")?);
    }
    Ok(())
}

fn read_deps(c: &mut Cursor) -> Result<BTreeSet<usize>, WireError> {
    let n = c.bounded_count(stride::DEP)?;
    let mut deps = BTreeSet::new();
    let mut previous: Option<usize> = None;
    for _ in 0..n {
        let dep = c.u32()? as usize;
        // Ascending and without repeats is what a set is. Bytes that say
        // otherwise are a second spelling of the same edges, which two readers
        // could disagree about the length of.
        if previous.is_some_and(|previous| previous >= dep) {
            return Err(WireError::NonCanonical {
                what: "node dependencies",
            });
        }
        previous = Some(dep);
        deps.insert(dep);
    }
    Ok(deps)
}

fn write_node<'a>(
    w: &mut Writer,
    pool: &mut Pool<'a>,
    index: usize,
    node: &'a Node,
) -> Result<(), WireError> {
    match node.work() {
        Work::Call {
            call:
                KernelLaunch {
                    module,
                    entry,
                    args,
                },
            geometry,
        } => {
            // A kernel launch is one launch, so the format states one shape.
            // A validated recipe of this family holds no other count.
            let [geometry] = geometry[..] else {
                return Err(WireError::Invalid(RecipeError::GeometryArity {
                    node: index,
                    stated: geometry.len(),
                    arity: 1,
                }));
            };
            w.u8(0);
            w.u32(checked_u32(*module, "module index")?);
            w.str(entry)?;
            for &g in &geometry.grid {
                w.u32(g);
            }
            for &b in &geometry.block {
                w.u32(b);
            }
            w.u32(geometry.dynamic_shmem);
            w.count(args.len())?;
            for arg in args {
                write_arg(w, arg)?;
            }
        }
        Work::Memset {
            destination,
            value,
            nbytes,
        } => {
            w.u8(1);
            write_writable_address(w, *destination)?;
            w.u8(*value);
            w.u64(encode_nbytes(*nbytes, "memset length")?);
        }
        Work::Memcpy {
            source,
            destination,
            nbytes,
        } => {
            w.u8(2);
            write_address(w, *source)?;
            write_writable_address(w, *destination)?;
            w.u64(encode_nbytes(*nbytes, "memcpy length")?);
        }
        Work::Choice {
            candidates,
            input_binding,
            output_binding,
        } => {
            w.u8(3);
            w.count(input_binding.len())?;
            for binding in input_binding {
                write_bufref(w, *binding)?;
            }
            w.count(output_binding.len())?;
            for binding in output_binding {
                write_bufref(w, *binding)?;
            }
            w.count(candidates.len())?;
            for candidate in candidates {
                write_aliases(w, candidate.aliases())?;
                write_body(w, pool, candidate)?;
            }
        }
    }
    write_deps(w, node.deps())
}

fn read_node(c: &mut Cursor, pool: &[Blob]) -> Result<Node, WireError> {
    let kind = c.u8()?;
    let work = match kind {
        0 => {
            let module = c.u32()? as usize;
            let entry = c.str()?;
            let grid = [c.u32()?, c.u32()?, c.u32()?];
            let block = [c.u32()?, c.u32()?, c.u32()?];
            let dynamic_shmem = c.u32()?;
            let n_args = c.bounded_count(stride::ARG)?;
            let args = (0..n_args).map(|_| read_arg(c)).collect::<Result<_, _>>()?;
            Work::Call {
                call: KernelLaunch {
                    module,
                    entry,
                    args,
                },
                geometry: vec![Geometry::new(grid, block, dynamic_shmem)],
            }
        }
        1 => {
            let destination = read_writable_address(c)?;
            let value = c.u8()?;
            let nbytes = decode_nbytes(c.u64()?, "memset length")?;
            Work::Memset {
                destination,
                value,
                nbytes,
            }
        }
        2 => {
            let source = read_address(c)?;
            let destination = read_writable_address(c)?;
            let nbytes = decode_nbytes(c.u64()?, "memcpy length")?;
            Work::Memcpy {
                source,
                destination,
                nbytes,
            }
        }
        3 => {
            let n_in = c.bounded_count(stride::BUFREF)?;
            let input_binding: Vec<BufRef> = (0..n_in)
                .map(|_| read_bufref(c))
                .collect::<Result<_, _>>()?;
            let n_out = c.bounded_count(stride::BUFREF)?;
            let output_binding: Vec<BufRef> = (0..n_out)
                .map(|_| read_bufref(c))
                .collect::<Result<_, _>>()?;
            let n_candidates = c.bounded_count(stride::BODY)?;
            // A candidate's arity is the binding that selects it, which is why
            // the format never states it twice: one number could disagree with
            // the other, and the binding is the one that decides.
            let candidates = (0..n_candidates)
                .map(|_| {
                    let aliases = read_aliases(c, input_binding.len(), output_binding.len())?;
                    let body = read_body(c, pool, input_binding.len(), output_binding.len())?
                        .with_port_aliases(aliases);
                    Ok(body)
                })
                .collect::<Result<_, WireError>>()?;
            Work::Choice {
                candidates,
                input_binding,
                output_binding,
            }
        }
        kind => {
            return Err(WireError::BadTag {
                what: "node kind",
                tag: kind,
            });
        }
    };
    Ok(Node::new(work, read_deps(c)?))
}

/// The least a single element of each counted sequence can occupy, read off
/// the writers above. A count comes from untrusted bytes and immediately sizes
/// an allocation, so it is refused here unless the section still holds the
/// bytes its elements need — the reader would otherwise reserve for `u32::MAX`
/// elements before the first short read failed.
mod stride {
    /// `Segment::Bytes` is a tag and a `u32` length; a pointer segment is
    /// larger.
    pub(super) const SEGMENT: usize = 5;
    /// One `usize` dep, written as a `u32`.
    pub(super) const DEP: usize = 4;
    /// An `Arg` is a counted list of segments, so at least its own count.
    pub(super) const ARG: usize = 4;
    /// A `BufRef` is a tag byte and a `u32` index.
    pub(super) const BUFREF: usize = 5;
    /// A body is three counts: modules, workspace, nodes.
    pub(super) const BODY: usize = 12;
    /// A code tag, and the module's resource bounds.
    pub(super) const MODULE: usize = 3;
    /// A `u64` size and an init tag.
    pub(super) const WORKSPACE: usize = 9;
    /// The smallest node is a `Choice`: an op tag, three counts, and its deps
    /// count.
    pub(super) const NODE: usize = 17;
}

pub(super) fn write_body<'a>(
    w: &mut Writer,
    pool: &mut Pool<'a>,
    recipe: &'a RecipeBody,
) -> Result<(), WireError> {
    w.count(recipe.context().len())?;
    for module in recipe.context() {
        write_module(w, pool, module)?;
    }
    w.count(recipe.workspace().len())?;
    for ws in recipe.workspace() {
        w.u64(encode_nbytes(ws.nbytes, "workspace byte extent")?);
        match &ws.init {
            Some(init) => {
                w.u8(1);
                w.u32(pool.index_of(init, PayloadRole::WorkspaceInit)?);
            }
            None => w.u8(0),
        }
    }
    w.count(recipe.nodes().len())?;
    for (index, node) in recipe.nodes().iter().enumerate() {
        write_node(w, pool, index, node)?;
    }
    Ok(())
}

pub(super) fn pool_blob(c: &mut Cursor, pool: &[Blob]) -> Result<Blob, WireError> {
    let index = c.u32()?;
    pool.get(index as usize)
        .cloned()
        .ok_or(WireError::BlobIndexOutOfRange { index })
}

/// The body these bytes state, bound through `inputs` and `outputs` formal
/// slots.
///
/// Arity arrives from outside rather than off the wire because it is never a
/// fact of its own: a root body's is its signature's, and a candidate's is the
/// binding that selects it. Deriving it is what keeps this format from carrying
/// a second number that could contradict the first.
pub(super) fn read_body(
    c: &mut Cursor,
    pool: &[Blob],
    inputs: usize,
    outputs: usize,
) -> Result<RecipeBody, WireError> {
    let n_modules = c.bounded_count(stride::MODULE)?;
    let modules = (0..n_modules)
        .map(|_| read_module(c, pool))
        .collect::<Result<_, _>>()?;
    let n_workspace = c.bounded_count(stride::WORKSPACE)?;
    let workspace = (0..n_workspace)
        .map(|_| {
            let nbytes = decode_nbytes(c.u64()?, "workspace byte extent")?;
            let init = match c.u8()? {
                0 => None,
                1 => Some(pool_blob(c, pool)?),
                tag => {
                    return Err(WireError::BadTag {
                        what: "workspace initializer",
                        tag,
                    });
                }
            };
            Ok(WorkspaceBuf { nbytes, init })
        })
        .collect::<Result<_, _>>()?;
    let n_nodes = c.bounded_count(stride::NODE)?;
    let nodes = (0..n_nodes)
        .map(|_| read_node(c, pool))
        .collect::<Result<_, _>>()?;
    Ok(RecipeBody::new(inputs, outputs, modules, workspace, nodes))
}
