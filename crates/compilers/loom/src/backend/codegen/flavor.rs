// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which tuning flavors a consumer launch can express.
//!
//! Every flavor is a candidate the autotuner may take, so declining one costs
//! nothing but the slot: these checks state what each arrangement assumes
//! about the program, and a launch that breaks an assumption simply does not
//! offer that slot.

use super::super::name::{Tuning, WarpAxis};
use super::{
    DType, DomainIndex, GuardedPlan, KernelPart, LiftedProgram, MaterializedValue, Plan, ScalarOp,
    WARP_SIZE,
};

/// The launch arrangement a consumer kernel is asked for, beyond the part and
/// block size that every kernel carries.
#[derive(Clone, Copy)]
pub(in crate::backend) struct Flavor<'a> {
    pub block_size: u32,
    pub part_index: usize,
    pub num_parts: usize,
    pub materialized: &'a [MaterializedValue],
    pub guard: Option<&'a GuardedPlan>,
    pub lane_hoist: bool,
    pub io_lanes: u8,
    pub warps_per_position: u8,
    pub warp_u: bool,
    pub fiber_lanes: bool,
}

impl Flavor<'_> {
    /// The plain arrangement: one warp per position, one lane per thread, no
    /// cut, guard, hoist, or split. Every other flavor is this with fields
    /// replaced.
    pub(in crate::backend) fn plain(block_size: u32, part_index: usize, num_parts: usize) -> Self {
        Self {
            block_size,
            part_index,
            num_parts,
            materialized: &[],
            guard: None,
            lane_hoist: false,
            io_lanes: 1,
            warps_per_position: 1,
            warp_u: false,
            fiber_lanes: false,
        }
    }

    /// The name tuning this arrangement carries. `fiber_lanes` is the lane
    /// count a lane-split fiber walk resolved, which depends on the hidden
    /// extent rather than on the arrangement alone.
    pub(crate) fn tuning(&self, fiber_lanes: Option<u8>) -> Tuning {
        Tuning {
            part: (self.num_parts > 1).then_some(self.part_index),
            cut: !self.materialized.is_empty(),
            guard: self.guard.is_some(),
            io_lanes: self.io_lanes,
            warp: (
                if self.warp_u {
                    WarpAxis::Hidden
                } else {
                    WarpAxis::Position
                },
                self.warps_per_position,
            ),
            fiber_lanes,
            lane_hoist: self.lane_hoist,
            ..Tuning::default()
        }
    }

    /// Whether this is the plain fiber walk a lane-split reduction builds on: one warp per position walking the whole hidden axis a lane at
    /// a time, over one part, with nothing — cut, guard, hoist, split,
    /// u-arrangement — layered on top.
    fn walks_a_plain_fiber(&self, program: &Plan, loom: &LiftedProgram, part: &KernelPart) -> bool {
        program.index.fiber_bridge().is_some()
            && self.materialized.is_empty()
            && self.guard.is_none()
            && !self.lane_hoist
            && self.io_lanes == 1
            && self.warps_per_position == 1
            && !self.warp_u
            && self.num_parts == 1
            && !part
                .elements
                .iter()
                .flatten()
                .any(|selection| selection.continuation)
            && !loom.outputs().iter().enumerate().any(|(output, binding)| {
                !part.elements[output].is_empty() && binding.dtype == DType::Bool
            })
    }
}

/// Reject the launch arrangements this program cannot express.
///
/// `fiber_scatter[output]` says the output is written straight from the fiber
/// walk rather than accumulated across it.
pub(super) fn check(
    flavor: Flavor,
    program: &Plan,
    loom: &LiftedProgram,
    part: &KernelPart,
    fiber_scatter: &[bool],
) -> Result<(), String> {
    let Flavor {
        block_size,
        num_parts: _,
        part_index: _,
        materialized,
        guard,
        lane_hoist,
        io_lanes,
        warps_per_position,
        warp_u,
        fiber_lanes,
    } = flavor;
    let has_wide_operand = loom
        .operands
        .iter()
        .any(|binding| binding.lane_extent != 1 && binding.lane_extent == loom.lane_count);
    let wide_storage_is_f32 = loom.operands.iter().all(|binding| {
        binding.lane_extent == 1
            || binding.lane_extent != loom.lane_count
            || binding.dtype == DType::F32
    });
    let wide_storage_is_half = loom.operands.iter().all(|binding| {
        binding.lane_extent == 1
            || binding.lane_extent != loom.lane_count
            || matches!(binding.dtype, DType::F16 | DType::Bf16)
    });
    // f32 lanes ride v2/v4 loads; half lanes ride packed-word v4/v8 loads. A
    // vector width is chosen for the dtype the buffers hold, so an output that
    // accumulates in a wider dtype than it stores rides neither: its writes are
    // the accumulator's width, not the storage width the lanes were sized for.
    let outputs_store_what_they_accumulate = (0..loom.outputs().len()).all(|output| {
        part.elements[output].is_empty() || program.folds.output(output).stores_its_accumulator()
    });
    let wide_f32_is_eligible = has_wide_operand && wide_storage_is_f32;
    let vector_width_is_eligible = outputs_store_what_they_accumulate
        && match io_lanes {
            2 => wide_f32_is_eligible,
            4 => wide_storage_is_half || wide_f32_is_eligible,
            8 => wide_storage_is_half,
            _ => false,
        };
    if lane_hoist && program.index.fiber_bridge().is_some() {
        return Err("loom lane hoisting does not support fiber iteration".into());
    }
    let plain_fiber = flavor.walks_a_plain_fiber(program, loom, part);
    if fiber_lanes
        && (!plain_fiber
            || loom.lane_count <= 0
            || loom.lane_count > i64::from(WARP_SIZE / 2)
            || fiber_scatter
                .iter()
                .enumerate()
                .any(|(output, &scatter)| scatter && !part.elements[output].is_empty()))
    {
        return Err("loom fiber-lane reduction is not eligible for this kernel".into());
    }
    if io_lanes > 1
        && (!materialized.is_empty()
            || !vector_width_is_eligible
            || loom.lane_count % (WARP_SIZE as i64 * io_lanes as i64) != 0
            || loom.module.entry_func().nodes.iter().any(|node| {
                node.op.as_elementwise().is_some_and(|op| {
                    matches!(
                        op.scalar(),
                        ScalarOp::ShiftLeft
                            | ScalarOp::ShiftRightArithmetic
                            | ScalarOp::ShiftRightLogical
                    )
                })
            }))
    {
        return Err(format!(
            "loom hidden-lane vec{io_lanes} is not eligible for this kernel"
        ));
    }
    if warp_u && (warps_per_position == 1 || program.index.fiber_bridge().is_none()) {
        return Err("loom warp u-split only applies to multi-warp fiber iteration".into());
    }
    if warps_per_position != 1 {
        let threads_per_position = WARP_SIZE * u32::from(warps_per_position);
        let plan_ok = matches!(warps_per_position, 2 | 4)
            && block_size.is_multiple_of(threads_per_position)
            && materialized.is_empty()
            && (guard.is_none()
                || (io_lanes == 1 && (program.index.fiber_bridge().is_none() || warp_u)))
            && !lane_hoist;
        // Splitting the fiber walk keeps every write exact: warp w visits
        // k = lo + w, lo + w + W, … — a partition of [lo, hi) — and every
        // warp runs the full hidden axis, so each (element, u) keeps a single
        // writer. Warp-uniform u passes let the wide combine barrier.
        let eligible = if fiber_split(program, warps_per_position, warp_u) {
            plan_ok && loom.lane_count % i64::from(WARP_SIZE * u32::from(io_lanes)) == 0
        } else {
            plan_ok
                && (!warp_u
                    || loom.lane_count
                        % i64::from(
                            WARP_SIZE * u32::from(io_lanes) * u32::from(warps_per_position),
                        )
                        == 0)
                && loom.lane_count >= i64::from(threads_per_position)
                && has_wide_operand
                && wide_storage_is_f32
                && !part
                    .elements
                    .iter()
                    .flatten()
                    .any(|selection| selection.continuation)
        };
        if !eligible {
            return Err(format!(
                "loom {warps_per_position}-warp position is not eligible for this kernel"
            ));
        }
    }
    Ok(())
}

/// A fiber-bridge launch has two warp arrangements: split the fiber walk
/// (each warp strides the edges, full hidden axis, partials combined), or
/// split the hidden axis (`warp_u` — each warp walks the whole fiber over
/// only its u slice, so every (element, u) keeps a single writer with no
/// combine at all). Kernels without a bridge already split the hidden
/// axis, so `warp_u` is only a distinct arrangement over a bridge.
pub(super) fn fiber_split(program: &Plan, warps_per_position: u8, warp_u: bool) -> bool {
    warps_per_position > 1 && program.index.fiber_bridge().is_some() && !warp_u
}

/// Whether the selected outputs whose width matches `width1` are written
/// straight from the fiber walk.
pub(super) fn scatter_selected(
    loom: &LiftedProgram,
    part: &KernelPart,
    fiber_scatter: &[bool],
    width1: bool,
) -> bool {
    fiber_scatter.iter().enumerate().any(|(output, &scatter)| {
        scatter
            && !part.elements[output].is_empty()
            && (loom.outputs()[output].lane_extent == 1) == width1
    })
}

/// Reject the arrangements whose writes a fiber-direct output would break.
/// These need `fiber_scatter`, so they sit after the walk's own checks.
pub(super) fn check_fiber_direct_writes(
    flavor: Flavor,
    loom: &LiftedProgram,
    part: &KernelPart,
    fiber_scatter: &[bool],
) -> Result<(), String> {
    let any_scatter = scatter_selected(loom, part, fiber_scatter, false)
        || scatter_selected(loom, part, fiber_scatter, true);
    if flavor.guard.is_some() && any_scatter {
        return Err("loom zero-skip does not cover fiber-direct output writes".into());
    }
    // A width-1 fiber-direct write reduces across the warp per finer element,
    // inside the fiber loop: every lane must run the same u passes.
    if scatter_selected(loom, part, fiber_scatter, true) {
        if loom.lane_count % i64::from(WARP_SIZE * u32::from(flavor.io_lanes)) != 0 {
            return Err("loom width-1 fiber-direct writes need warp-uniform u passes".into());
        }
        // Under a warp u-split every warp's per-edge tree yields a partial of
        // the same finer-element scalar: several writers per cell.
        if flavor.warp_u {
            return Err("loom warp u-split cannot write width-1 fiber-direct outputs".into());
        }
    }
    Ok(())
}
