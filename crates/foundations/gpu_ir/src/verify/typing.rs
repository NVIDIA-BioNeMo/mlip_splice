// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What each operation's operands and results hold, and which arithmetic has a
//! meaning at a scalar.
//!
//! **Every operation names its contract.** `Fma` is one rounding and never two;
//! `Div` and `Sqrt` on floats are correctly rounded (`.rn`) and there is no
//! approximate spelling of either, so a producer cannot lose accuracy by
//! accident; the packed pair forms the backend selects compute the same
//! `.rn` result as the lane-split ones, which is why selection can be the
//! backend's. Where a scalar has no such operation — remainder on floats, a
//! sum of predicates — there is no contract to name and the operation is
//! refused here rather than lowered into something else.

use super::{Checking, Result, Rule, Site};
use crate::arena::OpId;
use crate::op::OpKind;
use crate::{BinOp, Scalar, Space, Ty, UnOp, ValTy};

pub(super) fn check(c: &Checking<'_>) -> Result {
    for held in c.f.live_ops() {
        op(c, held)?;
    }
    Ok(())
}

/// The one data type this operand holds, or a diagnostic saying it is an
/// address where a datum belongs.
fn data(c: &Checking<'_>, op: OpId, slot: usize) -> Result<Ty> {
    let value = c.f.op(op).operands[slot];
    match c.f.ty(value) {
        ValTy::Data(ty) => Ok(ty),
        ValTy::Ptr(ptr) => c.fail(
            Rule::Type,
            Site::Op(op),
            format!(
                "operand {slot} is an address into {:?} where a data value belongs",
                ptr.elem
            ),
        ),
    }
}

fn pointer(c: &Checking<'_>, op: OpId, slot: usize) -> Result<crate::PtrTy> {
    let value = c.f.op(op).operands[slot];
    match c.f.ty(value) {
        ValTy::Ptr(ptr) => Ok(ptr),
        ValTy::Data(ty) => c.fail(
            Rule::Type,
            Site::Op(op),
            format!("operand {slot} holds {ty:?} where an address belongs"),
        ),
    }
}

fn arity(c: &Checking<'_>, op: OpId, operands: usize, results: usize) -> Result {
    let held = (c.f.op(op).operands.len(), c.f.op(op).results.len());
    c.require(held == (operands, results), Rule::Type, Site::Op(op), || {
        format!(
            "{:?} takes {operands} operands and produces {results}; this takes {} and produces {}",
            c.f.op(op).kind,
            held.0,
            held.1
        )
    })
}

/// The type the single result holds.
fn result(c: &Checking<'_>, op: OpId) -> ValTy {
    c.f.ty(c.f.op(op).results[0])
}

fn produces(c: &Checking<'_>, op: OpId, ty: ValTy) -> Result {
    let held = result(c, op);
    c.require(held == ty, Rule::Type, Site::Op(op), || {
        format!("it produces {held:?} where its operands make {ty:?}")
    })
}

fn op(c: &Checking<'_>, op: OpId) -> Result {
    match c.f.op(op).kind.clone() {
        OpKind::IntConst { scalar, .. } => {
            arity(c, op, 0, 1)?;
            c.require(
                !scalar.is_float() && scalar != Scalar::Pred,
                Rule::Type,
                Site::Op(op),
                || format!("an integer constant of {scalar:?}"),
            )?;
            produces(c, op, ValTy::Data(Ty::scalar(scalar)))
        }
        OpKind::FloatConst { scalar, .. } => {
            arity(c, op, 0, 1)?;
            c.require(scalar.is_float(), Rule::Type, Site::Op(op), || {
                format!("a float constant of {scalar:?}")
            })?;
            produces(c, op, ValTy::Data(Ty::scalar(scalar)))
        }
        OpKind::Sreg(_) => {
            arity(c, op, 0, 1)?;
            produces(c, op, ValTy::Data(Ty::scalar(Scalar::U32)))
        }
        OpKind::Load { lanes, .. } => {
            arity(c, op, 1, 1)?;
            let ptr = pointer(c, op, 0)?;
            c.require(lanes > 0, Rule::Type, Site::Op(op), || {
                "a load of no lanes".into()
            })?;
            produces(c, op, ValTy::Data(Ty::vec(ptr.elem, lanes)))
        }
        OpKind::ConstLoad(array) => {
            arity(c, op, 1, 1)?;
            let index = data(c, op, 0)?;
            c.require(index.lanes == 1, Rule::Type, Site::Op(op), || {
                "a constant read at a vector index".into()
            })?;
            c.require(
                matches!(
                    index.scalar,
                    Scalar::S32 | Scalar::U32 | Scalar::S64 | Scalar::U64
                ),
                Rule::Type,
                Site::Op(op),
                || format!("a constant read at an index of {:?}", index.scalar),
            )?;
            let size = array.scalar.size() as usize;
            c.require(
                !array.data.is_empty() && array.data.len().is_multiple_of(size),
                Rule::Memory,
                Site::Op(op),
                || {
                    format!(
                        "constant {} holds {} bytes, which is not whole {:?} elements",
                        array.name,
                        array.data.len(),
                        array.scalar
                    )
                },
            )?;
            produces(c, op, ValTy::Data(Ty::scalar(array.scalar)))
        }
        OpKind::Offset => {
            arity(c, op, 2, 1)?;
            let ptr = pointer(c, op, 0)?;
            let elems = data(c, op, 1)?;
            c.require(elems.lanes == 1, Rule::Type, Site::Op(op), || {
                "a pointer offset by a vector".into()
            })?;
            c.require(!elems.scalar.is_float(), Rule::Type, Site::Op(op), || {
                format!("a pointer offset by {:?}", elems.scalar)
            })?;
            produces(c, op, ValTy::Ptr(ptr))
        }
        OpKind::Un(unop) => {
            arity(c, op, 1, 1)?;
            let ty = data(c, op, 0)?;
            match unop {
                UnOp::Neg => {
                    c.require(ty.scalar != Scalar::Pred, Rule::Type, Site::Op(op), || {
                        "negating a predicate".into()
                    })?
                }
                UnOp::Not => c.require(!ty.scalar.is_float(), Rule::Type, Site::Op(op), || {
                    format!("a complement of {:?}", ty.scalar)
                })?,
                UnOp::Sqrt => c.require(ty.scalar.is_float(), Rule::Type, Site::Op(op), || {
                    format!("a square root of {:?}", ty.scalar)
                })?,
                UnOp::Exp2 | UnOp::Log2 | UnOp::Sin | UnOp::Cos | UnOp::Tanh => c.require(
                    ty.scalar.is_float() && ty.scalar != Scalar::F64,
                    Rule::Type,
                    Site::Op(op),
                    || format!("an approximate {unop:?} of {:?}", ty.scalar),
                )?,
                UnOp::Floor | UnOp::Ceil | UnOp::RoundEven => {
                    c.require(ty.scalar.is_float(), Rule::Type, Site::Op(op), || {
                        format!("a {unop:?} of {:?}", ty.scalar)
                    })?
                }
                UnOp::LeadingZeros | UnOp::OneBits => c.require(
                    !ty.scalar.is_float() && ty.scalar != Scalar::Pred,
                    Rule::Type,
                    Site::Op(op),
                    || format!("a {unop:?} of {:?}", ty.scalar),
                )?,
            }
            produces(c, op, ValTy::Data(ty))
        }
        OpKind::Bin(binop) => binary(c, op, binop),
        OpKind::Fma => {
            arity(c, op, 3, 1)?;
            let ty = data(c, op, 0)?;
            for slot in 1..3 {
                let held = data(c, op, slot)?;
                c.require(held == ty, Rule::Type, Site::Op(op), || {
                    format!("a fused multiply-add over {ty:?} and {held:?}")
                })?;
            }
            c.require(ty.scalar.is_float(), Rule::Type, Site::Op(op), || {
                format!(
                    "a fused multiply-add on {:?}, which rounds once only for floats",
                    ty.scalar
                )
            })?;
            produces(c, op, ValTy::Data(ty))
        }
        OpKind::Cmp(_) => {
            arity(c, op, 2, 1)?;
            let (a, b) = (data(c, op, 0)?, data(c, op, 1)?);
            c.require(a == b, Rule::Type, Site::Op(op), || {
                format!("a comparison of {a:?} with {b:?}")
            })?;
            produces(c, op, ValTy::Data(Ty::vec(Scalar::Pred, a.lanes)))
        }
        OpKind::Select => {
            arity(c, op, 3, 1)?;
            let pred = data(c, op, 0)?;
            let (a, b) = (data(c, op, 1)?, data(c, op, 2)?);
            c.require(
                pred.scalar == Scalar::Pred,
                Rule::Type,
                Site::Op(op),
                || format!("a select on {:?}", pred.scalar),
            )?;
            c.require(a == b, Rule::Type, Site::Op(op), || {
                format!("a select between {a:?} and {b:?}")
            })?;
            c.require(
                pred.lanes == 1 || pred.lanes == a.lanes,
                Rule::Type,
                Site::Op(op),
                || format!("a {}-lane predicate selecting {a:?}", pred.lanes),
            )?;
            produces(c, op, ValTy::Data(a))
        }
        OpKind::Cast(scalar) => {
            arity(c, op, 1, 1)?;
            let ty = data(c, op, 0)?;
            c.require(
                (scalar == Scalar::Pred) == (ty.scalar == Scalar::Pred),
                Rule::Type,
                Site::Op(op),
                || format!("a conversion between {:?} and {scalar:?}", ty.scalar),
            )?;
            produces(c, op, ValTy::Data(Ty::vec(scalar, ty.lanes)))
        }
        OpKind::Splat(lanes) => {
            arity(c, op, 1, 1)?;
            let ty = data(c, op, 0)?;
            c.require(ty.lanes == 1, Rule::Type, Site::Op(op), || {
                format!("a broadcast of {ty:?}, which is already a vector")
            })?;
            c.require(lanes > 0, Rule::Type, Site::Op(op), || {
                "a broadcast to no lanes".into()
            })?;
            produces(c, op, ValTy::Data(Ty::vec(ty.scalar, lanes)))
        }
        OpKind::Extract(lane) => {
            arity(c, op, 1, 1)?;
            let ty = data(c, op, 0)?;
            c.require(lane < ty.lanes, Rule::Type, Site::Op(op), || {
                format!("lane {lane} of {ty:?}")
            })?;
            produces(c, op, ValTy::Data(Ty::scalar(ty.scalar)))
        }
        OpKind::Pack => {
            c.require(
                !c.f.op(op).operands.is_empty() && c.f.op(op).results.len() == 1,
                Rule::Type,
                Site::Op(op),
                || "a pack of nothing".into(),
            )?;
            let held = data(c, op, 0)?;
            let mut lanes: u8 = 0;
            for slot in 0..c.f.op(op).operands.len() {
                let part = data(c, op, slot)?;
                c.require(part.scalar == held.scalar, Rule::Type, Site::Op(op), || {
                    format!("a pack of {:?} with {:?}", held.scalar, part.scalar)
                })?;
                let Some(total) = lanes.checked_add(part.lanes) else {
                    return c.fail(Rule::Type, Site::Op(op), "a pack of more than 255 lanes");
                };
                lanes = total;
            }
            produces(c, op, ValTy::Data(Ty::vec(held.scalar, lanes)))
        }
        OpKind::ShflDown => {
            arity(c, op, 2, 1)?;
            let ty = data(c, op, 0)?;
            let delta = data(c, op, 1)?;
            c.require(ty.scalar != Scalar::Pred, Rule::Type, Site::Op(op), || {
                "a shuffle of predicates, which have no bit pattern to move".into()
            })?;
            c.require(delta.lanes == 1, Rule::Type, Site::Op(op), || {
                "a shuffle by a vector delta".into()
            })?;
            c.require(!delta.scalar.is_float(), Rule::Type, Site::Op(op), || {
                format!("a shuffle by {:?}", delta.scalar)
            })?;
            produces(c, op, ValTy::Data(ty))
        }
        OpKind::ActiveMask => {
            arity(c, op, 0, 1)?;
            produces(c, op, ValTy::Data(Ty::scalar(Scalar::U32)))
        }
        OpKind::ShflFrom => {
            arity(c, op, 3, 1)?;
            let ty = data(c, op, 0)?;
            let lane = data(c, op, 1)?;
            let participants = data(c, op, 2)?;
            c.require(ty.scalar != Scalar::Pred, Rule::Type, Site::Op(op), || {
                "a shuffle of predicates, which have no bit pattern to move".into()
            })?;
            c.require(lane.lanes == 1, Rule::Type, Site::Op(op), || {
                "a shuffle from a vector of source lanes".into()
            })?;
            // `shfl.sync` takes its lane in a 32-bit word whatever it moves, so
            // a narrower or wider one is a register the instruction cannot be
            // given — the same rule a shift amount answers to.
            c.require(
                matches!(lane.scalar, Scalar::S32 | Scalar::U32),
                Rule::Type,
                Site::Op(op),
                || format!("a shuffle from a source lane of {:?}", lane.scalar),
            )?;
            // The mask is a bit per lane, which is what `shfl.sync` takes and
            // what `activemask.b32` produces; a wider or narrower one names
            // lanes the instruction cannot be given.
            c.require(
                participants == Ty::scalar(Scalar::U32),
                Rule::Type,
                Site::Op(op),
                || format!("a shuffle over participants held as {participants:?}"),
            )?;
            produces(c, op, ValTy::Data(ty))
        }
        OpKind::WarpAny => {
            arity(c, op, 1, 1)?;
            let ty = data(c, op, 0)?;
            c.require(ty.scalar == Scalar::Pred, Rule::Type, Site::Op(op), || {
                format!("a warp vote over {:?}", ty.scalar)
            })?;
            produces(c, op, ValTy::Data(Ty::scalar(Scalar::Pred)))
        }
        OpKind::Bind => {
            arity(c, op, 1, 1)?;
            let held = c.f.ty(c.f.op(op).operands[0]);
            produces(c, op, held)?;
            // A binding runs in the storage it names, whatever that storage
            // holds — there is no arm to fall back to, as there is under a
            // branch — so the two must hold the same thing.
            if let Some(destination) = c.f.op(op).destination(0) {
                let there = c.f.ty(destination);
                c.require(there == held, Rule::Aliasing, Site::Op(op), || {
                    format!("a binding of {held:?} runs in storage holding {there:?}")
                })?;
            }
            Ok(())
        }
        OpKind::Write { .. } => {
            arity(c, op, 2, 0)?;
            pointer(c, op, 0)?;
            data(c, op, 1)?;
            Ok(())
        }
        OpKind::FetchAdd => {
            arity(c, op, 2, 1)?;
            let ptr = pointer(c, op, 0)?;
            let ty = data(c, op, 1)?;
            // One address is one cell: a wide value would be several additions
            // and several previous values, which is not one indivisible step.
            c.require(ty.lanes == 1, Rule::Type, Site::Op(op), || {
                format!("an atomic add of {ty:?}, which is more than one cell")
            })?;
            c.require(ptr.elem == ty.scalar, Rule::Memory, Site::Op(op), || {
                format!(
                    "an atomic add of {:?} through an address into {:?}",
                    ty.scalar, ptr.elem
                )
            })?;
            produces(c, op, ValTy::Data(ty))
        }
        OpKind::Call(callee) => call(c, op, &callee),
        // A guard is a step: it runs where it stands, and what it produces
        // becomes visible after the branch. A loop is either, being also the
        // value a gather-and-combine reduces to.
        OpKind::If => {
            arity(c, op, 1, c.f.op(op).results.len())?;
            let predicate = data(c, op, 0)?;
            c.require(
                predicate == Ty::scalar(Scalar::Pred),
                Rule::Type,
                Site::Op(op),
                || format!("an if on {predicate:?} rather than a scalar predicate"),
            )?;
            c.require(c.f.op(op).placed, Rule::Region, Site::Op(op), || {
                "an if standing for a value has no place to run".into()
            })?;
            data_results(c, op, "an if")
        }
        OpKind::Loop { index, .. } => {
            c.require(
                matches!(index, Scalar::S32 | Scalar::U32 | Scalar::S64 | Scalar::U64),
                Rule::Type,
                Site::Op(op),
                || format!("a loop indexed by {index:?}"),
            )?;
            let expected = Ty::scalar(index);
            for slot in 0..3 {
                let held = data(c, op, slot)?;
                c.require(held == expected, Rule::Type, Site::Op(op), || {
                    format!("loop header operand {slot} holds {held:?} rather than {expected:?}")
                })?;
            }
            let region = c.f.op(op).regions[0];
            let block = c.f.region_block(region);
            c.require(
                c.f.ty(c.f.block(block).params[0]) == ValTy::Data(expected),
                Rule::Type,
                Site::Region(region),
                || "a loop body index does not match its declared index type".into(),
            )?;
            data_results(c, op, "a loop")
        }
        OpKind::Scope => arity(c, op, 0, 0),
        OpKind::Break | OpKind::Ret | OpKind::Bar => arity(c, op, 0, 0),
    }
}

fn data_results(c: &Checking<'_>, op: OpId, what: &str) -> Result {
    for &value in &c.f.op(op).results {
        c.require(
            matches!(c.f.ty(value), ValTy::Data(_)),
            Rule::Type,
            Site::Op(op),
            || format!("{what} produces a pointer, which the PTX region ABI cannot carry"),
        )?;
    }
    Ok(())
}

/// Which binary operations have a meaning at a scalar.
///
/// Bit operations are the predicate's only arithmetic — there is no sum of two
/// predicates — and the remainder has no float form: `rem` is integral, and a
/// float remainder would be a library call this language does not contain.
fn binary(c: &Checking<'_>, op: OpId, binop: BinOp) -> Result {
    arity(c, op, 2, 1)?;
    let a = data(c, op, 0)?;
    let b = data(c, op, 1)?;
    let bitwise = matches!(binop, BinOp::And | BinOp::Or | BinOp::Xor);
    if matches!(binop, BinOp::Shl | BinOp::Shr) {
        c.require(b.lanes == 1, Rule::Type, Site::Op(op), || {
            "a shift by a vector amount".into()
        })?;
        c.require(
            matches!(b.scalar, Scalar::S32 | Scalar::U32),
            Rule::Type,
            Site::Op(op),
            || format!("a shift amount of {:?}", b.scalar),
        )?;
    } else {
        c.require(a == b, Rule::Type, Site::Op(op), || {
            format!("{binop:?} over {a:?} and {b:?}")
        })?;
    }
    if a.scalar == Scalar::Pred {
        c.require(bitwise, Rule::Type, Site::Op(op), || {
            format!("{binop:?} on predicates, which have no arithmetic but and/or/xor")
        })?;
    } else if a.scalar.is_float() {
        c.require(!bitwise, Rule::Type, Site::Op(op), || {
            format!("{binop:?} on {:?}, which has no bit operations", a.scalar)
        })?;
        c.require(
            !matches!(binop, BinOp::Shl | BinOp::Shr),
            Rule::Type,
            Site::Op(op),
            || format!("a shift of {:?}", a.scalar),
        )?;
        c.require(binop != BinOp::Rem, Rule::Type, Site::Op(op), || {
            format!(
                "a remainder of {:?}, which has no rounded contract",
                a.scalar
            )
        })?;
    }
    produces(c, op, ValTy::Data(a))
}

/// A call agrees with its callee at both ends, and the callee is a function a
/// call can run: it declares no shared memory, because shared memory is the
/// launched kernel's and nothing outside a body can name it.
///
/// One rule, both kinds of callee. What a body states through its arena and what
/// a declaration states in words are the same facts, so the call that runs
/// either answers here identically; which of the two it is decides only whether
/// the callee is run where it stands or left for the linker.
fn call(c: &Checking<'_>, op: OpId, callee: &crate::func::DeviceFunction) -> Result {
    super::verify(callee).map_err(|inner| super::Diagnostic {
        function: c.name.to_string(),
        rule: inner.rule,
        at: Site::Op(op),
        detail: format!("its callee {} is not a function: {}", callee.name, inner),
    })?;
    c.require(
        callee.shared_arrays().is_empty(),
        Rule::Abi,
        Site::Op(op),
        || {
            format!(
                "callee {} declares shared memory, which belongs to the kernel that is launched",
                callee.name
            )
        },
    )?;
    let taken = callee.parameter_tys();
    let params = taken.len();
    let arguments = c.f.op(op).operands.len();
    c.require(arguments == params, Rule::Type, Site::Op(op), || {
        format!(
            "callee {} takes {params} arguments and was given {arguments}",
            callee.name
        )
    })?;
    for (slot, &wanted) in taken.iter().enumerate() {
        let held = c.f.ty(c.f.op(op).operands[slot]);
        c.require(held == wanted, Rule::Type, Site::Op(op), || {
            format!(
                "argument {slot} holds {held:?} where callee {} takes {wanted:?}",
                callee.name
            )
        })?;
        if let ValTy::Ptr(ptr) = wanted {
            c.require(ptr.space == Space::Global, Rule::Abi, Site::Op(op), || {
                format!("callee {} takes a shared-memory pointer", callee.name)
            })?;
        }
    }
    let produced = callee.result_tys();
    c.require(
        c.f.op(op).results.len() == produced.len(),
        Rule::Type,
        Site::Op(op),
        || {
            format!(
                "callee {} hands back {} values and the call produces {}",
                callee.name,
                produced.len(),
                c.f.op(op).results.len()
            )
        },
    )?;
    for (slot, &wanted) in produced.iter().enumerate() {
        let held = c.f.ty(c.f.op(op).results[slot]);
        c.require(held == wanted, Rule::Type, Site::Op(op), || {
            format!(
                "result {slot} holds {held:?} where callee {} hands back {wanted:?}",
                callee.name
            )
        })?;
    }
    Ok(())
}
