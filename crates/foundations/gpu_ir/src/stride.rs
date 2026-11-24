// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What an integer value is provably a multiple of.
//!
//! A vector access renders as one instruction whose effective address must be
//! as wide as the vector, and that obligation splits in two: the element offset
//! the body computes must be a whole multiple of the lane count, and the base
//! must be aligned to it. This settles the first half, over the arena, for both
//! the renderer's own precondition and the binding contract a caller derives.
//!
//! The lattice is divisibility: `0` is every stride — nothing was added to the
//! base — and `1` is nothing proven. Meets are `gcd`, products scale, and a
//! product that does not fit keeps the larger factor rather than a saturated
//! number that would prove something nobody established.

use std::collections::BTreeMap;

use crate::BinOp;
use crate::arena::{BlockId, Def, Function, OpId, ValueId};
use crate::op::OpKind;

pub fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Whether an offset proven a multiple of `stride` is a multiple of `lanes`.
pub fn divides(stride: u64, lanes: u8) -> bool {
    stride == 0 || stride.is_multiple_of(u64::from(lanes))
}

/// A product is a multiple of the product of its factors' strides — and, when
/// that product is not representable, still a multiple of each factor's.
fn scale(a: u64, b: u64) -> u64 {
    a.checked_mul(b).unwrap_or(a.max(b))
}

/// What survives narrowing an integer to `bits`: only the power-of-two part of
/// a stride does, because `x ≡ 0 (mod 2^k)` still holds for the low bits and
/// nothing else has to.
fn narrowed(stride: u64, bits: u32) -> u64 {
    stride.isolate_lowest_one().min(1u64 << bits.min(63))
}

impl Function {
    /// The largest stride `value` is provably a multiple of: zero for one that
    /// is exactly zero, and one where nothing is proven.
    pub fn stride(&self, value: ValueId) -> u64 {
        let mut walking = BTreeMap::new();
        self.stride_under(value, &mut walking)
    }

    /// A carried value is defined in terms of itself, so a value already being
    /// walked proves nothing on this path; `1` is the unit that leaves the
    /// meet with whatever the rest of the cycle proves.
    fn stride_under(&self, value: ValueId, walking: &mut BTreeMap<usize, ()>) -> u64 {
        let index = value.index();
        if walking.insert(index, ()).is_some() {
            return 1;
        }
        let stride = self.stride_of(value, walking);
        walking.remove(&index);
        stride
    }

    fn stride_of(&self, value: ValueId, walking: &mut BTreeMap<usize, ()>) -> u64 {
        match self.value(value).def {
            Def::Param { block, index } => self.param_stride(block, index, walking),
            Def::Result { op, index } => {
                let operands = self.op(op).operands.clone();
                let at = |slot: usize, walking: &mut BTreeMap<usize, ()>| {
                    self.stride_under(operands[slot], walking)
                };
                match self.op(op).kind {
                    OpKind::IntConst { value, .. } => value.unsigned_abs(),
                    OpKind::Bind => at(0, walking),
                    OpKind::Bin(BinOp::Add | BinOp::Sub) => gcd(at(0, walking), at(1, walking)),
                    OpKind::Bin(BinOp::Mul) => scale(at(0, walking), at(1, walking)),
                    OpKind::Bin(BinOp::Shl) => match self.operand_op(operands[1]) {
                        Some(shift) => match self.op(shift).kind {
                            OpKind::IntConst { value, .. } if (0..64).contains(&value) => {
                                scale(at(0, walking), 1u64 << value)
                            }
                            _ => 1,
                        },
                        None => 1,
                    },
                    // A conversion that keeps or widens the value keeps its
                    // multiples; a narrowing one keeps only the power of two.
                    OpKind::Cast(scalar) => {
                        let held = self.data_ty(operands[0]).scalar;
                        let stride = at(0, walking);
                        if scalar.size() >= held.size() {
                            stride
                        } else {
                            narrowed(stride, 8 * scalar.size() as u32)
                        }
                    }
                    // Whichever way the branch went, the value came from one of
                    // the arms, so what both prove is what it proves.
                    OpKind::If => self.regions_stride(op, index, walking),
                    // A loop hands back what the body yielded, or what it
                    // started from when the body never ran.
                    OpKind::Loop { .. } => gcd(
                        self.stride_under(operands[3 + index as usize], walking),
                        self.regions_stride(op, index, walking),
                    ),
                    _ => 1,
                }
            }
        }
    }

    /// What the `index`-th value every region of `op` yields is a multiple of.
    fn regions_stride(&self, op: OpId, index: u32, walking: &mut BTreeMap<usize, ()>) -> u64 {
        let mut proven: Option<u64> = None;
        for slot in 0..self.op(op).regions.len() {
            let region = self.op(op).regions[slot];
            for inner in 0..self.region(region).blocks.len() {
                let block = self.region(region).blocks[inner];
                let Some(&yielded) = self.block(block).terminator.yields.get(index as usize) else {
                    return 1;
                };
                let stride = self.stride_under(yielded, walking);
                proven = Some(match proven {
                    None => stride,
                    Some(held) => gcd(held, stride),
                });
            }
        }
        proven.unwrap_or(1)
    }

    /// A block parameter: a kernel's own parameter proves nothing, a loop's
    /// index steps from its start by its step, and a carried value is what it
    /// started from meeting what the body yields.
    fn param_stride(&self, block: BlockId, index: u32, walking: &mut BTreeMap<usize, ()>) -> u64 {
        if block == self.entry() {
            return 1;
        }
        let Some(op) = self.block_owner(block) else {
            return 1;
        };
        if !matches!(self.op(op).kind, OpKind::Loop { .. }) {
            return 1;
        }
        let operands = self.op(op).operands.clone();
        if index == 0 {
            // `k = start + n·step`, so every value it takes is a multiple of
            // whatever both are.
            return gcd(
                self.stride_under(operands[0], walking),
                self.stride_under(operands[2], walking),
            );
        }
        let carried = index as usize - 1;
        gcd(
            self.stride_under(operands[3 + carried], walking),
            self.regions_stride(op, carried as u32, walking),
        )
    }

    /// Whether an access `lanes` wide through `address` is proven to land on a
    /// `lanes`-element boundary relative to the pointer it starts from.
    pub fn offset_divides(&self, address: ValueId, lanes: u8) -> bool {
        divides(self.address_stride(address), lanes)
    }

    /// What the element offset from a base pointer to `address` is a multiple
    /// of. A pointer nothing offsets is the base itself, at offset zero.
    pub fn address_stride(&self, address: ValueId) -> u64 {
        match self.operand_op(address) {
            Some(op) if matches!(self.op(op).kind, OpKind::Offset) => {
                let operands = &self.op(op).operands;
                gcd(self.address_stride(operands[0]), self.stride(operands[1]))
            }
            Some(op) if matches!(self.op(op).kind, OpKind::Bind) => {
                self.address_stride(self.op(op).operands[0])
            }
            _ => match self.value(address).def {
                Def::Result { op, .. } if matches!(self.op(op).kind, OpKind::Bind) => {
                    self.address_stride(self.op(op).operands[0])
                }
                _ => 0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::narrowed;

    #[test]
    fn narrowing_preserves_zero_and_caps_the_power_of_two_factor() {
        for bits in (0..=64).chain([u32::MAX]) {
            assert_eq!(narrowed(0, bits), 0);
            for shift in 0..64 {
                for odd in [1, 3, 5, u64::MAX] {
                    let stride = odd << shift;
                    assert_eq!(narrowed(stride, bits), 1 << shift.min(bits.min(63)));
                }
            }
        }
    }
}
