// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Where an operand element lives, as an index a target IR can compute.
//!
//! A [`DomainIndex`] says how each operand axis reaches its domain from the
//! iteration domain. Turning that into an address means gathering through
//! maps and aligned views, clamping the cells a bridge leaves unmapped, and
//! folding the per-axis indices into one flat element offset. That algebra is
//! the same whatever IR the address is built in, so it is written once here,
//! over an [`IndexBuilder`] that supplies the arithmetic and an
//! [`AddressSource`] that supplies the leaves.
//!
//! Nothing here names an IR: a backend brings its own [`IndexBuilder`], which is
//! also the only place a backend dependency could enter this crate.

use crate::{AxisBridge, DomainIndex};

/// The expression vocabulary the domain-index emitters need, over any target
/// IR. Every index value is 64-bit by contract: each impl widens where its IR
/// requires it (the C++ AST by promotion, `gpu_ir` by explicit casts), so leaves
/// handed in by an [`AddressSource`] — axis indices, `fiber_edge` — must
/// already be 64-bit values.
///
/// Construction goes through `&mut self` because a target that owns its values
/// needs somewhere to put them. No impl reorders, folds or reassociates what it
/// is handed: the index algebra a caller writes is the one that reaches the
/// target.
pub trait IndexBuilder {
    type Value: Clone;
    fn lit(&mut self, v: i64) -> Self::Value;
    /// `array[idx]` as a 64-bit index value.
    fn gather(&mut self, array: Self::Value, idx: Self::Value) -> Self::Value;
    /// Clamp an unmapped `-1` in bounds: `max(value, 0)`.
    fn max0(&mut self, value: Self::Value) -> Self::Value;
    /// Widen to 64-bit (identity when already wide).
    fn to_i64(&mut self, value: Self::Value) -> Self::Value;
    fn mul_lit(&mut self, value: Self::Value, k: i64) -> Self::Value;
    fn add(&mut self, a: Self::Value, b: Self::Value) -> Self::Value;
    /// The predicate `array[idx] >= 0`.
    fn nonneg(&mut self, array: Self::Value, idx: Self::Value) -> Self::Value;
    /// The predicate `value >= 0`.
    fn nonneg_value(&mut self, value: Self::Value) -> Self::Value;
    fn and(&mut self, a: Self::Value, b: Self::Value) -> Self::Value;
}

/// The leaves an address is built from, in the target `B` builds into: this
/// context's axis indices, and the reference arrays its launch bound.
///
/// The builder is a parameter rather than a member because the two are chosen
/// independently — the same placement is addressed in whichever IR is being
/// emitted, and a builder serves every context that emits into its IR.
pub trait AddressSource<B: IndexBuilder + ?Sized> {
    type Index: DomainIndex;
    fn index(&self) -> &Self::Index;
    fn axis_index(&self, b: &mut B, axis: usize) -> B::Value;
    fn target_indices_array(&self, b: &mut B, map_idx: usize) -> B::Value;
    fn aligned_view_array(&self, b: &mut B, view_idx: usize) -> B::Value;
    /// `view[_k]` pre-bound once for the fiber iteration, when the emitter
    /// provides it — otherwise every operand bridged through the view
    /// re-gathers it.
    fn view_element(&self, _b: &mut B, _view_idx: usize) -> Option<B::Value> {
        None
    }
    /// The index the fiber loop is on, for a view whose element this context
    /// did not pre-bind. Only a context iterating a fiber has one, and only a
    /// bridge through a view asks.
    fn fiber_index(&self) -> B::Value;
}

pub fn operand_flat_index<B, C>(
    ctx: &C,
    b: &mut B,
    operand: usize,
    fiber_edge: Option<B::Value>,
) -> B::Value
where
    B: IndexBuilder + ?Sized,
    C: AddressSource<B> + ?Sized,
{
    let axes: Vec<usize> = (0..ctx.index().num_axes()).collect();
    operand_flat_index_over(ctx, b, operand, fiber_edge, &axes)
}

pub fn operand_flat_index_over<B, C>(
    ctx: &C,
    b: &mut B,
    operand: usize,
    fiber_edge: Option<B::Value>,
    axes: &[usize],
) -> B::Value
where
    B: IndexBuilder + ?Sized,
    C: AddressSource<B> + ?Sized,
{
    let p = ctx.index();
    let mut idx_per_axis: Vec<B::Value> = Vec::with_capacity(axes.len());
    let mut extent_per_axis: Vec<i64> = Vec::with_capacity(axes.len());
    for &axis in axes {
        let axis_idx = ctx.axis_index(b, axis);
        let dom_id = p.operand_domain(operand, axis);
        let (idx, ext) = match p.axis_bridge(operand, axis) {
            AxisBridge::Broadcast => (b.lit(0), 1),
            AxisBridge::Direct => {
                let iter_dom = p.iter_domain(axis);
                (axis_idx.clone(), p.graph().domain_extent(iter_dom))
            }
            AxisBridge::MapLookup { map_idx } => {
                let target_indices = ctx.target_indices_array(b, map_idx);
                let raw = b.gather(target_indices, axis_idx.clone());
                // A `-1` (unmapped) index would point the base before the
                // buffer. Clamp it in bounds; `operand_alive` masks the value
                // / skips the write so the clamped slot never contributes.
                let idx = if p.graph().index_maps[map_idx].may_have_unmapped {
                    b.max0(raw)
                } else {
                    raw
                };
                (idx, p.graph().domain_extent(dom_id))
            }
            AxisBridge::PreimageWalk { .. } => {
                let edge = fiber_edge
                    .clone()
                    .expect("PreimageWalk operand encountered outside the fiber loop");
                (edge, p.graph().domain_extent(dom_id))
            }
            AxisBridge::AlignedLookup { view_idx } => {
                fiber_edge
                    .as_ref()
                    .expect("AlignedView operand encountered outside the fiber loop");
                let raw = match ctx.view_element(b, view_idx) {
                    Some(element) => element,
                    None => {
                        let view = ctx.aligned_view_array(b, view_idx);
                        let k = ctx.fiber_index();
                        b.gather(view, k)
                    }
                };
                let idx = if p.graph().aligned_view_other_may_have_unmapped(view_idx) {
                    b.max0(raw)
                } else {
                    raw
                };
                (idx, p.graph().domain_extent(dom_id))
            }
        };
        idx_per_axis.push(idx);
        extent_per_axis.push(ext);
    }

    // The flat index is a global element offset reaching anywhere in the array,
    // so it is built in 64-bit: the widened seed keeps every `acc * extent + idx`
    // step 64-bit even when an axis index is a 32-bit gather or fiber-edge load,
    // so neither the accumulation nor the `flat * stride` pointer base overflows.
    let Some(idx0) = idx_per_axis.first().cloned() else {
        return b.lit(0);
    };
    let mut acc = b.to_i64(idx0);
    for pos in 1..idx_per_axis.len() {
        let ext = extent_per_axis[pos];
        let idx = idx_per_axis[pos].clone();
        let scaled = b.mul_lit(acc, ext);
        acc = b.add(scaled, idx);
    }
    acc
}

pub fn operand_buffer_offset_over<B, C>(
    ctx: &C,
    b: &mut B,
    operand: usize,
    axes: &[usize],
) -> B::Value
where
    B: IndexBuilder + ?Sized,
    C: AddressSource<B> + ?Sized,
{
    let p = ctx.index();
    let mut terms: Vec<B::Value> = Vec::with_capacity(axes.len());
    for &axis in axes {
        let axis_idx = ctx.axis_index(b, axis);
        let idx = match p.axis_bridge(operand, axis) {
            AxisBridge::Broadcast => continue,
            AxisBridge::Direct => axis_idx,
            AxisBridge::MapLookup { map_idx } => {
                let target_indices = ctx.target_indices_array(b, map_idx);
                let raw = b.gather(target_indices, axis_idx);
                if p.graph().index_maps[map_idx].may_have_unmapped {
                    b.max0(raw)
                } else {
                    raw
                }
            }
            other => panic!("operand_buffer_offset_over: unsupported row bridge {other:?}"),
        };
        // Built in 64-bit (the offset reaches anywhere in the array).
        // The offset walk runs inside emission, where a refused stride has
        // nowhere to go; the schedule that built this operand multiplied the
        // same extents out first and would have refused there.
        let stride = p
            .operand_axis_stride(operand, axis)
            .expect("the schedule accepted this operand's strides");
        let wide = b.to_i64(idx);
        terms.push(if stride == 1 {
            wide
        } else {
            b.mul_lit(wide, stride)
        });
    }
    let mut it = terms.into_iter();
    let Some(first) = it.next() else {
        return b.lit(0);
    };
    it.fold(first, |acc, term| b.add(acc, term))
}

/// Whether [`operand_alive`] has a predicate to build for `operand`, answered
/// off the index alone and without building anything.
///
/// Both are `masks_operand` over every axis, so they cannot reach different
/// conclusions about the same index — which is what lets a caller that has no
/// context decide the shape of an expansion the predicate will land in. It
/// takes the index rather than the context precisely because such a caller has
/// only the index: an expansion that saw the context would depend on it.
pub fn operand_has_alive<I: DomainIndex>(index: &I, operand: usize) -> bool {
    (0..index.num_axes()).any(|axis| masks_operand(index, operand, axis))
}

/// The one rule: an axis masks its operand when it bridges through something
/// that leaves cells unmapped.
fn masks_operand<I: DomainIndex>(index: &I, operand: usize, axis: usize) -> bool {
    match index.axis_bridge(operand, axis) {
        AxisBridge::MapLookup { map_idx } => index.graph().index_maps[map_idx].may_have_unmapped,
        AxisBridge::AlignedLookup { view_idx } => {
            index.graph().aligned_view_other_may_have_unmapped(view_idx)
        }
        _ => false,
    }
}

pub fn operand_alive<B, C>(ctx: &C, b: &mut B, operand: usize) -> Option<B::Value>
where
    B: IndexBuilder + ?Sized,
    C: AddressSource<B> + ?Sized,
{
    let p = ctx.index();
    let mut preds: Vec<B::Value> = Vec::new();
    for axis in 0..p.num_axes() {
        if !masks_operand(p, operand, axis) {
            continue;
        }
        match p.axis_bridge(operand, axis) {
            AxisBridge::MapLookup { map_idx } => {
                let axis_idx = ctx.axis_index(b, axis);
                let target_indices = ctx.target_indices_array(b, map_idx);
                preds.push(b.nonneg(target_indices, axis_idx));
            }
            AxisBridge::AlignedLookup { view_idx } => {
                preds.push(match ctx.view_element(b, view_idx) {
                    Some(element) => b.nonneg_value(element),
                    None => {
                        let view = ctx.aligned_view_array(b, view_idx);
                        let k = ctx.fiber_index();
                        b.nonneg(view, k)
                    }
                });
            }
            _ => unreachable!("only a gather or a view masks an operand"),
        }
    }
    preds.into_iter().reduce(|acc, next| b.and(acc, next))
}

#[cfg(test)]
mod tests;
