// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Addressing, the forms a space admits, the alignment a wide access owes, and
//! the storage a produced value takes over.
//!
//! A multi-lane access renders as one vector instruction whose effective
//! address must be as wide as the vector. That obligation splits: the element
//! offset the body computes must be a whole multiple of the lane count, which
//! [`crate::Function::offset_divides`] decides here, and the base must be
//! aligned to it, which is what the binding owes the kernel.
//!
//! Taking over storage is the other half of this file. A result produced into
//! a place is one register under two names, and what must hold of that is
//! about storage rather than about meaning: a destination is a place with
//! storage to name, and never the index a running loop counts with, which the
//! loop writes where no step shows.

use super::structure::inner_blocks;
use super::{Checking, Result, Rule, Site};
use crate::arena::{BlockId, OpId, ValueId};
use crate::op::{OpKind, WriteMode};
use crate::ptx::vector_form;
use crate::{Evict, Scalar, Space, ValTy};

/// Every operation's account of what it does to memory, answered where it
/// stands.
pub(super) fn effects(c: &Checking<'_>) -> Result {
    for op in c.f.live_ops() {
        addresses(c, op)?;
    }
    Ok(())
}

pub(super) fn check(c: &Checking<'_>) -> Result {
    for op in c.f.live_ops() {
        takeover(c, op)?;
        access(c, op)?;
    }
    let mut running = fixedbitset::FixedBitSet::with_capacity(c.f.num_values());
    block(c, c.f.entry(), &mut running)
}

/// Every operand an operation says it touches memory through holds an address.
///
/// The rule is over the effects and not over the kind, so what it checks is that
/// an operation's own account of what it does is answerable at all — for each of
/// the operands it names, an external call naming several included.
fn addresses(c: &Checking<'_>, op: OpId) -> Result {
    for effect in c.f.op(op).kind.effects().each() {
        let Some(slot) = effect.address() else {
            continue;
        };
        let Some(&address) = c.f.op(op).operands.get(slot) else {
            return c.fail(
                Rule::Effect,
                Site::Op(op),
                format!("it touches memory through operand {slot} and has none"),
            );
        };
        c.require(
            matches!(c.f.ty(address), ValTy::Ptr(_)),
            Rule::Effect,
            Site::Op(op),
            || format!("it touches memory through operand {slot}, which holds a datum"),
        )?;
    }
    Ok(())
}

/// A destination names storage a result may be produced into.
///
/// It is a **request about allocation, never about meaning**: the value a
/// result holds is the same whether it runs in the storage it names or in
/// storage of its own, and the backend declines wherever running there would
/// change what something else reads. So what must hold here is only what makes
/// the request answerable — one destination per result, and each a place with
/// storage to name. A destination of the wrong type, or two results naming one,
/// is a request the backend cannot honour and does not; it is not a program
/// that means something impossible.
fn takeover(c: &Checking<'_>, op: OpId) -> Result {
    let into = &c.f.op(op).into;
    if into.is_empty() {
        return Ok(());
    }
    c.require(
        into.len() == c.f.op(op).results.len(),
        Rule::Aliasing,
        Site::Op(op),
        || {
            format!(
                "it names {} destinations for {} results",
                into.len(),
                c.f.op(op).results.len()
            )
        },
    )?;
    for &destination in into.iter().flatten() {
        c.require(
            c.f.is_materialized(destination),
            Rule::Aliasing,
            Site::Value(destination),
            || "it is an operand, so it has no storage to name".into(),
        )?;
    }
    Ok(())
}

/// What a load or a write asks of the space it reaches and of the address it
/// forms.
fn access(c: &Checking<'_>, op: OpId) -> Result {
    match c.f.op(op).kind {
        OpKind::Load { lanes, nc, evict } => {
            let address = c.f.op(op).operands[0];
            let ValTy::Ptr(ptr) = c.f.ty(address) else {
                return Ok(());
            };
            c.require(
                !nc || ptr.space == Space::Global,
                Rule::Memory,
                Site::Op(op),
                || "a non-coherent read outside global memory".into(),
            )?;
            hinted(c, op, evict, ptr.space)?;
            aligned(c, op, address, lanes, ptr.elem)
        }
        OpKind::Write { mode, evict } => {
            let address = c.f.op(op).operands[0];
            let ValTy::Ptr(ptr) = c.f.ty(address) else {
                return Ok(());
            };
            let ValTy::Data(ty) = c.f.ty(c.f.op(op).operands[1]) else {
                return Ok(());
            };
            c.require(ptr.elem == ty.scalar, Rule::Memory, Site::Op(op), || {
                format!(
                    "a write of {:?} through an address into {:?}",
                    ty.scalar, ptr.elem
                )
            })?;
            match mode {
                // One `red.global.add` per lane at consecutive addresses, so a
                // wide value asks nothing wider of the base than an element.
                WriteMode::Add => {
                    c.require(
                        ptr.space == Space::Global,
                        Rule::Memory,
                        Site::Op(op),
                        || "an atomic add outside global memory".into(),
                    )?;
                    c.require(evict == Evict::Normal, Rule::Memory, Site::Op(op), || {
                        "an eviction hint on an atomic add, which takes none".into()
                    })?;
                    c.require(
                        adds_atomically(ty.scalar),
                        Rule::Memory,
                        Site::Op(op),
                        || {
                            format!(
                                "an atomic add of {:?}, which has no reduction form",
                                ty.scalar
                            )
                        },
                    )
                }
                WriteMode::Set => {
                    hinted(c, op, evict, ptr.space)?;
                    aligned(c, op, address, ty.lanes, ptr.elem)
                }
            }
        }
        OpKind::FetchAdd => {
            let address = c.f.op(op).operands[0];
            let ValTy::Ptr(ptr) = c.f.ty(address) else {
                return Ok(());
            };
            let ValTy::Data(ty) = c.f.ty(c.f.op(op).operands[1]) else {
                return Ok(());
            };
            c.require(
                ptr.space == Space::Global,
                Rule::Memory,
                Site::Op(op),
                || "an atomic add outside global memory".into(),
            )?;
            c.require(
                adds_atomically(ty.scalar),
                Rule::Memory,
                Site::Op(op),
                || {
                    format!(
                        "an atomic add of {:?}, which has no reduction form",
                        ty.scalar
                    )
                },
            )
        }
        _ => Ok(()),
    }
}

fn hinted(c: &Checking<'_>, op: OpId, evict: Evict, space: Space) -> Result {
    c.require(
        evict == Evict::Normal || space == Space::Global,
        Rule::Memory,
        Site::Op(op),
        || "an eviction hint outside global memory".into(),
    )
}

/// Which scalars `red.global.add` has a form for.
fn adds_atomically(scalar: Scalar) -> bool {
    matches!(
        scalar,
        Scalar::F16
            | Scalar::BF16
            | Scalar::F32
            | Scalar::F64
            | Scalar::S32
            | Scalar::U32
            | Scalar::S64
            | Scalar::U64
    )
}

/// An access that renders as one vector instruction lands on a boundary as
/// wide as the vector, so the offset from the base must be a whole multiple of
/// the lane count. Widths with no vector form split into element accesses and
/// ask nothing beyond an element.
fn aligned(c: &Checking<'_>, op: OpId, address: ValueId, lanes: u8, elem: Scalar) -> Result {
    if !vector_form(crate::Ty::vec(elem, lanes)) {
        return Ok(());
    }
    c.require(
        c.f.offset_divides(address, lanes),
        Rule::Alignment,
        Site::Op(op),
        || {
            format!(
                "it moves {lanes} lanes at an offset proven only a multiple of {}, so the \
                 address it forms is not {lanes}-element aligned",
                c.f.address_stride(address)
            )
        },
    )
}

/// A binding may not take over the index a running loop counts with: the loop
/// writes that register at the bottom, where no step shows.
fn block(c: &Checking<'_>, at: BlockId, running: &mut fixedbitset::FixedBitSet) -> Result {
    for index in 0..c.f.block(at).ops.len() {
        let op = c.f.block(at).ops[index];
        if !c.f.op(op).placed {
            continue;
        }
        if matches!(c.f.op(op).kind, OpKind::Bind)
            && let Some(place) = c.f.op(op).destination(0)
            && running.contains(place.index())
        {
            return c.fail(
                Rule::Aliasing,
                Site::Op(op),
                "it takes over the index a running loop counts with, which the loop writes \
                 where no step shows",
            );
        }
        let owned = match c.f.op(op).kind {
            OpKind::Loop { .. } => {
                let inner = c.f.region_block(c.f.op(op).regions[0]);
                c.f.block(inner).params[..1].to_vec()
            }
            _ => Vec::new(),
        };
        for param in &owned {
            running.insert(param.index());
        }
        let walked = inner_blocks(c, op).try_for_each(|inner| block(c, inner, running));
        for param in owned {
            running.set(param.index(), false);
        }
        walked?;
    }
    Ok(())
}
