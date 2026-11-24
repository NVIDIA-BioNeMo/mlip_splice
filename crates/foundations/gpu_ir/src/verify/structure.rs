// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Who owns what, what a region hands back, and where a value may be read.
//!
//! Ownership is one definition per value and one home per operation: a value
//! is defined by the block that takes it or the operation that produces it, and
//! every operation occupies exactly one position in exactly one block.
//! A region belongs to the operation that lists it.
//!
//! Dominance, in a structured body, is the same statement as scope: a value is
//! readable where the region that defined it is still open. An operand is
//! evaluated where it is read, so what it reads must be readable *there*,
//! which is why the places under an operand are checked at the use and not at
//! the definition.

use std::collections::{BTreeMap, BTreeSet};

use super::{Checking, Result, Rule, Site};
use crate::arena::{BlockId, OpId, ValueId};
use crate::op::OpKind;

pub(super) fn check(c: &Checking<'_>) -> Result {
    let mut homes = BTreeMap::new();
    let mut visible = Vec::new();
    let mut nesting = Nesting::default();
    let entry = c.f.entry();
    c.require(
        c.f.block(entry).region.is_none(),
        Rule::Ownership,
        Site::Block(entry),
        || "the entry block belongs to the function, not to a region".into(),
    )?;
    visible.push(c.f.block(entry).params.iter().copied().collect());
    block(c, entry, &mut homes, &mut visible, &mut nesting)?;
    for op in c.f.live_ops() {
        c.require(
            homes.contains_key(&op.index()),
            Rule::Ownership,
            Site::Op(op),
            || "it belongs to no block".into(),
        )?;
    }
    for block in c.f.live_blocks() {
        c.require(
            nesting.blocks.contains(&block),
            Rule::Ownership,
            Site::Block(block),
            || "it is not reachable from the entry block".into(),
        )?;
    }
    for region in c.f.live_regions() {
        c.require(
            nesting.regions.contains(&region),
            Rule::Ownership,
            Site::Region(region),
            || "it is not reachable from the entry block".into(),
        )?;
    }
    Ok(())
}

/// What encloses the block being walked: the operations whose regions it sits
/// in, innermost last.
#[derive(Default)]
struct Nesting {
    enclosing: Vec<OpId>,
    blocks: BTreeSet<BlockId>,
    regions: BTreeSet<crate::RegionId>,
}

impl Nesting {
    /// The innermost loop this sits in, if any.
    fn innermost_loop(&self, c: &Checking<'_>) -> Option<OpId> {
        self.enclosing
            .iter()
            .rev()
            .copied()
            .find(|&op| matches!(c.f.op(op).kind, OpKind::Loop { .. }))
    }

    /// Whether every region between here and the function hands back nothing,
    /// which is what leaving the function outright needs: a result of a region
    /// jumped out of would never be produced.
    fn all_resultless(&self, c: &Checking<'_>) -> bool {
        self.enclosing
            .iter()
            .all(|&op| c.f.op(op).results.is_empty())
    }
}

fn block(
    c: &Checking<'_>,
    at: BlockId,
    homes: &mut BTreeMap<usize, BlockId>,
    visible: &mut Vec<BTreeSet<ValueId>>,
    nesting: &mut Nesting,
) -> Result {
    c.require(
        nesting.blocks.insert(at),
        Rule::Ownership,
        Site::Block(at),
        || "it is reached through more than one structural path".into(),
    )?;
    for index in 0..c.f.block(at).ops.len() {
        let op = c.f.block(at).ops[index];
        // One definition, one operation position. Standing twice would define
        // it produces, and what they hold the second time need not be what they
        // held the first — the same load between two writes is two values, not
        // one — so a caller that means to compute something twice builds it
        // twice.
        if let Some(&held) = homes.get(&op.index()) {
            return c.fail(
                Rule::Ownership,
                Site::Op(op),
                format!(
                    "it is already a step of block {}, so what it defines would be \
                     defined twice",
                    held.index()
                ),
            );
        }
        homes.insert(op.index(), at);
        step(c, op, homes, visible, nesting)?;
        visible
            .last_mut()
            .expect("a scope is always open")
            .extend(c.f.op(op).results.iter().copied());
    }
    for index in 0..c.f.block(at).terminator.yields.len() {
        readable(c, c.f.block(at).terminator.yields[index], visible)?;
    }
    Ok(())
}

/// Every place a value reads is defined in a scope that is still open.
///
/// A region's own bindings are the region's: what a region computes leaves it
/// only through what it yields, which the operation owning it produces. Naming
/// storage — an accumulator a fold is produced into — does not make a value
/// escape, because the storage outliving the region says nothing about whether
/// the region ran.
fn readable(c: &Checking<'_>, value: ValueId, visible: &[BTreeSet<ValueId>]) -> Result {
    for place in c.f.reads(value) {
        c.require(
            visible.iter().any(|scope| scope.contains(&place)),
            Rule::Dominance,
            Site::Value(value),
            || {
                format!(
                    "it reads %{}, which a region that has ended defines; a value leaves a \
                     region through what the region yields",
                    place.index()
                )
            },
        )?;
    }
    Ok(())
}

fn step(
    c: &Checking<'_>,
    op: OpId,
    homes: &mut BTreeMap<usize, BlockId>,
    visible: &mut Vec<BTreeSet<ValueId>>,
    nesting: &mut Nesting,
) -> Result {
    for index in 0..c.f.op(op).operands.len() {
        readable(c, c.f.op(op).operands[index], visible)?;
    }
    for index in 0..c.f.op(op).into.len() {
        if let Some(destination) = c.f.op(op).into[index] {
            readable(c, destination, visible)?;
        }
    }
    regions(c, op, nesting)?;
    match c.f.op(op).kind {
        OpKind::If => conditional(c, op, homes, visible, nesting),
        OpKind::Loop { .. } => counted_loop(c, op, homes, visible, nesting),
        OpKind::Scope => {
            let region = c.f.op(op).regions[0];
            let inner = c.f.region_block(region);
            c.require(
                c.f.block(inner).params.is_empty(),
                Rule::Region,
                Site::Region(region),
                || "a scope region takes no parameters".into(),
            )?;
            c.require(
                c.f.block(inner).terminator.yields.is_empty(),
                Rule::Region,
                Site::Op(op),
                || "a scope produces nothing, so its region yields nothing".into(),
            )?;
            under(c, op, inner, homes, visible, nesting)
        }
        OpKind::Break => {
            let Some(held) = nesting.innermost_loop(c) else {
                return c.fail(Rule::Region, Site::Op(op), "a break outside every loop");
            };
            c.require(
                c.f.op(held).results.is_empty(),
                Rule::Region,
                Site::Op(op),
                || "a loop that carries values has no value to leave it with".into(),
            )
        }
        OpKind::Ret => c.require(
            nesting.all_resultless(c),
            Rule::Region,
            Site::Op(op),
            || "a result of the region this leaves would never be produced".into(),
        ),
        _ => Ok(()),
    }
}

/// Who owns the regions a region-owning kind lists. How many there are and how
/// many blocks each holds is [`super::shape`]'s, decided before any walk looks.
fn regions(c: &Checking<'_>, op: OpId, nesting: &mut Nesting) -> Result {
    for index in 0..c.f.op(op).regions.len() {
        let region = c.f.op(op).regions[index];
        c.require(
            nesting.regions.insert(region),
            Rule::Ownership,
            Site::Region(region),
            || "it is reached through more than one structural path".into(),
        )?;
        c.require(
            c.f.region(region).owner == Some(op),
            Rule::Ownership,
            Site::Region(region),
            || format!("operation {} lists it and does not own it", op.index()),
        )?;
        let block = c.f.region(region).blocks[0];
        c.require(
            c.f.block(block).region == Some(region),
            Rule::Ownership,
            Site::Block(block),
            || {
                format!(
                    "region {} holds it and it belongs elsewhere",
                    region.index()
                )
            },
        )?;
    }
    Ok(())
}

fn under(
    c: &Checking<'_>,
    op: OpId,
    inner: BlockId,
    homes: &mut BTreeMap<usize, BlockId>,
    visible: &mut Vec<BTreeSet<ValueId>>,
    nesting: &mut Nesting,
) -> Result {
    nesting.enclosing.push(op);
    visible.push(c.f.block(inner).params.iter().copied().collect());
    let walked = block(c, inner, homes, visible, nesting);
    visible.pop();
    nesting.enclosing.pop();
    walked
}

fn conditional(
    c: &Checking<'_>,
    op: OpId,
    homes: &mut BTreeMap<usize, BlockId>,
    visible: &mut Vec<BTreeSet<ValueId>>,
    nesting: &mut Nesting,
) -> Result {
    let results = c.f.op(op).results.len();
    for (arm, index) in ["then", "else"].iter().zip(0..2) {
        let region = c.f.op(op).regions[index];
        let inner = c.f.region_block(region);
        c.require(
            c.f.block(inner).params.is_empty(),
            Rule::Region,
            Site::Region(region),
            || "an arm of an if takes nothing on the way in".into(),
        )?;
        let yields = c.f.block(inner).terminator.yields.len();
        c.require(yields == results, Rule::Region, Site::Op(op), || {
            format!(
                "an if {arm} arm yields {yields} of {results} results{}",
                if results > 0 && c.f.block(inner).ops.is_empty() && yields == 0 {
                    " — an absent else arm is legal only with no results"
                } else {
                    ""
                }
            )
        })?;
        for slot in 0..yields {
            let yielded = c.f.block(inner).terminator.yields[slot];
            let result = c.f.op(op).results[slot];
            c.require(
                c.f.ty(yielded) == c.f.ty(result),
                Rule::Type,
                Site::Value(yielded),
                || {
                    format!(
                        "an if {arm} arm yields {:?} where its result holds {:?}",
                        c.f.ty(yielded),
                        c.f.ty(result)
                    )
                },
            )?;
        }
        if results > 0 {
            c.require(!escapes(c, inner, true), Rule::Region, Site::Op(op), || {
                "a result of an if is undefined where its arm returns or breaks".into()
            })?;
        }
        under(c, op, inner, homes, visible, nesting)?;
    }
    Ok(())
}

fn counted_loop(
    c: &Checking<'_>,
    op: OpId,
    homes: &mut BTreeMap<usize, BlockId>,
    visible: &mut Vec<BTreeSet<ValueId>>,
    nesting: &mut Nesting,
) -> Result {
    let carried = c.f.op(op).results.len();
    let region = c.f.op(op).regions[0];
    let inner = c.f.region_block(region);
    for slot in 0..carried {
        let param = c.f.block(inner).params[1 + slot];
        let result = c.f.op(op).results[slot];
        let init = c.f.op(op).operands[3 + slot];
        let yielded = c.f.block(inner).terminator.yields[slot];
        for (held, what) in [(param, "takes"), (init, "starts from"), (yielded, "yields")] {
            c.require(
                c.f.ty(held) == c.f.ty(result),
                Rule::Type,
                Site::Value(held),
                || {
                    format!(
                        "a loop {what} {:?} where it carries {:?}",
                        c.f.ty(held),
                        c.f.ty(result)
                    )
                },
            )?;
        }
    }
    if carried > 0 {
        c.require(!leaves_loop(c, inner), Rule::Region, Site::Op(op), || {
            "a loop that carries values has no value to leave it with".into()
        })?;
    }
    under(c, op, inner, homes, visible, nesting)
}

/// Whether a block leaves the region it is in rather than reaching its end.
/// `breaks` says whether a `Break` leaves it — inside a loop it does not.
fn escapes(c: &Checking<'_>, at: BlockId, breaks: bool) -> bool {
    c.f.block_steps(at).any(|op| match c.f.op(op).kind {
        OpKind::Ret => true,
        OpKind::Break => breaks,
        OpKind::If | OpKind::Scope => inner_blocks(c, op).any(|inner| escapes(c, inner, breaks)),
        OpKind::Loop { .. } => inner_blocks(c, op).any(|inner| escapes(c, inner, false)),
        _ => false,
    })
}

/// Whether a block leaves the loop it is the body of, by breaking out of it or
/// by leaving the kernel. A break under a loop it opens itself is that loop's.
fn leaves_loop(c: &Checking<'_>, at: BlockId) -> bool {
    c.f.block_steps(at).any(|op| match c.f.op(op).kind {
        OpKind::Break | OpKind::Ret => true,
        OpKind::If | OpKind::Scope => inner_blocks(c, op).any(|inner| leaves_loop(c, inner)),
        OpKind::Loop { .. } => inner_blocks(c, op).any(|inner| escapes(c, inner, false)),
        _ => false,
    })
}

pub(super) fn inner_blocks<'a>(
    c: &'a Checking<'a>,
    op: OpId,
) -> impl Iterator<Item = BlockId> + 'a {
    c.f.op(op)
        .regions
        .iter()
        .flat_map(move |&region| c.f.region(region).blocks.iter().copied())
}
