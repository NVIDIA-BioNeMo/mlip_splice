// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What every rule after this one is entitled to assume.
//!
//! The other passes read a program the way the backend will: an `if` has a
//! predicate, a loop has bounds and a body that takes its index, a region holds
//! a block. Those are facts about *shape*, and a pass that met one broken would
//! index past the end rather than say so — which is why they are decided here,
//! first, over every operation in the arena rather than only the ones some walk
//! reaches.
//!
//! It is deliberately small. Anything a later pass can meet and answer for
//! itself stays where it is; what is here is exactly what would otherwise stop
//! the process.

use super::{Checking, Result, Rule, Site};
use crate::arena::Def;
use crate::op::OpKind;

pub(super) fn check(c: &Checking<'_>) -> Result {
    references(c)?;
    declarations(c)?;
    definitions(c)?;
    acyclic(c)?;
    for op_id in c.f.live_ops() {
        op(c, op_id)?;
    }
    Ok(())
}

fn references(c: &Checking<'_>) -> Result {
    c.require(
        c.f.holds_block(c.f.entry()),
        Rule::Ownership,
        Site::Function,
        || "its entry block does not exist".into(),
    )?;
    for op in c.f.live_ops() {
        for &value in c.f.op(op).operands.iter().chain(c.f.op(op).results.iter()) {
            c.require(
                c.f.holds_value(value),
                Rule::Ownership,
                Site::Op(op),
                || format!("it names value %{}, which does not exist", value.index()),
            )?;
        }
        for &value in c.f.op(op).into.iter().flatten() {
            c.require(
                c.f.holds_value(value),
                Rule::Ownership,
                Site::Op(op),
                || {
                    format!(
                        "it names value %{} as storage, and it does not exist",
                        value.index()
                    )
                },
            )?;
        }
        for &region in &c.f.op(op).regions {
            c.require(
                c.f.holds_region(region),
                Rule::Ownership,
                Site::Op(op),
                || format!("it names region {}, which does not exist", region.index()),
            )?;
        }
    }
    for block in c.f.live_blocks() {
        for &value in
            c.f.block(block)
                .params
                .iter()
                .chain(&c.f.block(block).terminator.yields)
        {
            c.require(
                c.f.holds_value(value),
                Rule::Ownership,
                Site::Block(block),
                || format!("it names value %{}, which does not exist", value.index()),
            )?;
        }
        for &op in &c.f.block(block).ops {
            c.require(
                c.f.holds_op(op),
                Rule::Ownership,
                Site::Block(block),
                || format!("it names operation {}, which does not exist", op.index()),
            )?;
        }
        if let Some(region) = c.f.block(block).region {
            c.require(
                c.f.holds_region(region),
                Rule::Ownership,
                Site::Block(block),
                || format!("it names region {}, which does not exist", region.index()),
            )?;
        }
    }
    for region in c.f.live_regions() {
        if let Some(owner) = c.f.region(region).owner {
            c.require(
                c.f.holds_op(owner),
                Rule::Ownership,
                Site::Region(region),
                || format!("its owner operation {} does not exist", owner.index()),
            )?;
        }
        for &block in &c.f.region(region).blocks {
            c.require(
                c.f.holds_block(block),
                Rule::Ownership,
                Site::Region(region),
                || format!("it names block {}, which does not exist", block.index()),
            )?;
        }
    }
    Ok(())
}

fn declarations(c: &Checking<'_>) -> Result {
    let entry_params = &c.f.block(c.f.entry()).params;
    c.require(
        entry_params.len() == c.f.declarations().len(),
        Rule::Abi,
        Site::Function,
        || {
            format!(
                "the entry takes {} values for {} declarations",
                entry_params.len(),
                c.f.declarations().len()
            )
        },
    )?;
    for (index, declared) in c.f.declarations().iter().enumerate() {
        let expected = match declared {
            crate::Declaration::Param(param) => param.ty.val_ty(),
            crate::Declaration::Shared(array) => crate::ValTy::Ptr(crate::PtrTy {
                elem: array.elem,
                space: crate::Space::Shared,
            }),
        };
        let held = c.f.ty(entry_params[index]);
        c.require(held == expected, Rule::Abi, Site::Function, || {
            format!("declaration {index} states {expected:?} and its entry value holds {held:?}")
        })?;
    }
    Ok(())
}

/// What `op` must be visited after: what it reads, what it produces into, and
/// everything the bodies of its regions hold.
fn depends_on(c: &Checking<'_>, op: crate::arena::OpId) -> Vec<crate::arena::OpId> {
    let mut dependencies =
        c.f.op(op)
            .operands
            .iter()
            .chain(c.f.op(op).into.iter().flatten())
            .filter_map(|&value| match c.f.value(value).def {
                Def::Result { op, .. } => Some(op),
                Def::Param { .. } => None,
            })
            .collect::<Vec<_>>();
    for &region in &c.f.op(op).regions {
        for &block in &c.f.region(region).blocks {
            dependencies.extend(c.f.block(block).ops.iter().copied());
            dependencies.extend(
                c.f.block(block)
                    .terminator
                    .yields
                    .iter()
                    .filter_map(|&value| match c.f.value(value).def {
                        Def::Result { op, .. } => Some(op),
                        Def::Param { .. } => None,
                    }),
            );
        }
    }
    dependencies
}

fn acyclic(c: &Checking<'_>) -> Result {
    let roots = c.f.live_ops().into_iter().map(|op| op.index());
    let found = weave::first_cycle(c.f.num_ops(), roots, |operation, leads_to| {
        let op = c.f.op_at(operation);
        for dependency in depends_on(c, op) {
            leads_to(dependency.index());
        }
    });
    match found {
        None => Ok(()),
        Some(operation) => c.fail(
            Rule::Dominance,
            Site::Op(c.f.op_at(operation)),
            format!("its use-def chain returns to operation {operation}"),
        ),
    }
}

/// Every value is defined exactly once, by the block or operation that claims
/// it, at the index it says.
fn definitions(c: &Checking<'_>) -> Result {
    let mut claimed = fixedbitset::FixedBitSet::with_capacity(c.f.num_values());
    for block in c.f.live_blocks() {
        for (index, &value) in c.f.block(block).params.iter().enumerate() {
            c.require(
                !claimed.put(value.index()),
                Rule::Ownership,
                Site::Value(value),
                || "it is claimed as more than one definition".into(),
            )?;
            c.require(
                c.f.value(value).def
                    == (Def::Param {
                        block,
                        index: index as u32,
                    }),
                Rule::Ownership,
                Site::Value(value),
                || {
                    format!(
                        "block {} claims it at {index}, but it points elsewhere",
                        block.index()
                    )
                },
            )?;
        }
    }
    for op in c.f.live_ops() {
        for (index, &value) in c.f.op(op).results.iter().enumerate() {
            c.require(
                !claimed.put(value.index()),
                Rule::Ownership,
                Site::Value(value),
                || "it is claimed as more than one definition".into(),
            )?;
            c.require(
                c.f.value(value).def
                    == (Def::Result {
                        op,
                        index: index as u32,
                    }),
                Rule::Ownership,
                Site::Value(value),
                || {
                    format!(
                        "operation {} claims it at {index}, but it points elsewhere",
                        op.index()
                    )
                },
            )?;
        }
    }
    for value in c.f.live_values() {
        match c.f.value(value).def {
            Def::Param { block, index } => {
                c.require(
                    c.f.holds_block(block),
                    Rule::Ownership,
                    Site::Value(value),
                    || format!("its defining block {} does not exist", block.index()),
                )?;
                let held = c.f.block(block).params.get(index as usize).copied();
                c.require(
                    held == Some(value),
                    Rule::Ownership,
                    Site::Value(value),
                    || format!("block {} does not take it at {index}", block.index()),
                )?;
            }
            Def::Result { op, index } => {
                c.require(
                    c.f.holds_op(op),
                    Rule::Ownership,
                    Site::Value(value),
                    || format!("its defining operation {} does not exist", op.index()),
                )?;
                let held = c.f.op(op).results.get(index as usize).copied();
                c.require(
                    held == Some(value),
                    Rule::Ownership,
                    Site::Value(value),
                    || format!("operation {} does not produce it at {index}", op.index()),
                )?;
            }
        }
    }
    Ok(())
}

fn op(c: &Checking<'_>, op: crate::arena::OpId) -> Result {
    let held = c.f.op(op).regions.len();
    let wanted = c.f.op(op).kind.regions();
    c.require(held == wanted, Rule::Region, Site::Op(op), || {
        format!("it owns {held} regions where its kind owns {wanted}")
    })?;
    for index in 0..held {
        let region = c.f.op(op).regions[index];
        let blocks = c.f.region(region).blocks.len();
        c.require(blocks == 1, Rule::Region, Site::Region(region), || {
            format!("a region holds one block and this holds {blocks}")
        })?;
    }
    let placed = c.f.op(op).placed;
    let kind = &c.f.op(op).kind;
    c.require(
        placed == kind.must_be_placed() || kind.may_be_either(),
        Rule::Region,
        Site::Op(op),
        || {
            format!(
                "it is {} and its kind is only ever {}",
                if placed {
                    "a step of a block"
                } else {
                    "an operand"
                },
                if placed {
                    "an operand"
                } else {
                    "a step of a block"
                }
            )
        },
    )?;
    match c.f.op(op).kind {
        OpKind::If => c.require(
            c.f.op(op).operands.len() == 1,
            Rule::Region,
            Site::Op(op),
            || {
                format!(
                    "an if takes one predicate and takes {}",
                    c.f.op(op).operands.len()
                )
            },
        ),
        OpKind::Loop { .. } => counted_loop(c, op),
        _ => Ok(()),
    }
}

/// A loop states its bounds and one starting value per result, and its body
/// takes the index and each carried value and yields one per result.
///
/// Everything after this reads a loop by those positions — the trip walk asks
/// what its bounds hold, the storage walk asks what its index is — so a loop
/// shaped otherwise is refused before any of them looks.
fn counted_loop(c: &Checking<'_>, op: crate::arena::OpId) -> Result {
    let carried = c.f.op(op).results.len();
    let operands = c.f.op(op).operands.len();
    c.require(operands == 3 + carried, Rule::Region, Site::Op(op), || {
        format!(
            "a loop carrying {carried} values starts {}",
            operands.saturating_sub(3)
        )
    })?;
    let region = c.f.op(op).regions[0];
    let inner = c.f.region_block(region);
    let params = c.f.block(inner).params.len();
    c.require(
        params == 1 + carried,
        Rule::Region,
        Site::Region(region),
        || "a loop's body takes its index and each carried value".into(),
    )?;
    let yields = c.f.block(inner).terminator.yields.len();
    c.require(yields == carried, Rule::Region, Site::Op(op), || {
        format!("a loop carrying {carried} values yields {yields}")
    })
}
