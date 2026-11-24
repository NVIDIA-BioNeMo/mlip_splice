// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The canonical identity of an emitted Loom kernel. `KernelName`'s `Display`
//! is the CUDA `.visible .entry` symbol; derived projection symbols are
//! produced through this module, so
//! codegen (which emits the symbol) and the compiler (which reconstructs it to
//! bind the recipe) can never disagree.
//!
//! Grammar: `loom_<program>_<role>_bs<N>[_<knob>…]`. The role letters and knob
//! tags are compact by design; each knob renders nothing at its default value,
//! so a bare consumer is `loom_<program>_c_bs<N>`.

use std::fmt;

fn to_ptx_identifier(raw: &str) -> String {
    let mut identifier = String::with_capacity(raw.len().max(1));
    for character in raw.chars() {
        identifier.push(if character.is_ascii_alphanumeric() || character == '_' {
            character
        } else {
            '_'
        });
    }
    match identifier.chars().next() {
        None => identifier.push_str("kernel"),
        Some(character) if !character.is_ascii_alphabetic() && character != '_' => {
            identifier.insert(0, '_');
        }
        _ => {}
    }
    identifier
}

/// The kind of kernel — the role field after the program name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Role {
    /// The geometry worker: one pass over positions, carrying the full tuning
    /// block. Renders `c`.
    Consumer(Tuning),
    /// Materialization producer for the cut intermediate `output`. Renders
    /// `pr<output>`.
    Producer { output: usize },
    /// A single-block reduction (`br`).
    Reduction,
}

/// Whether warps beyond the first cover distinct positions (`wp`) or divide the
/// hidden `u` axis of one position (`wu`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WarpAxis {
    Position,
    Hidden,
}

/// The autotuning knobs that distinguish two consumer candidates. Every knob
/// renders one compact tag and renders nothing at its default (`1` for the
/// N-way splits, `None`/`false` for the rest), so adding a knob is one field
/// plus one line in `fmt`. The render order here is the canonical tag order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Tuning {
    /// Partition index, set only when the operation is split (`num_parts > 1`).
    pub part: Option<usize>,
    /// Reads materialized cut intermediates.
    pub cut: bool,
    /// Zero-skip certificate guard.
    pub guard: bool,
    /// Vector IO lanes per thread; `1` is scalar.
    pub io_lanes: u8,
    /// Warps per position and the axis they cover; count `1` is a single warp.
    pub warp: (WarpAxis, u8),
    /// Hidden-axis lanes per position (subwarp), `None` when the fiber is not
    /// lane-split.
    pub fiber_lanes: Option<u8>,
    /// Position prologue hoisted out of the lane loop.
    pub lane_hoist: bool,
    /// Whole-element output shards executed as sibling grid-y branches; `1` is
    /// unsharded.
    pub element_shards: u8,
    /// Whole-element output groups executed as sequential passes; `1` is a
    /// single pass.
    pub passes: u8,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            part: None,
            cut: false,
            guard: false,
            io_lanes: 1,
            warp: (WarpAxis::Position, 1),
            fiber_lanes: None,
            lane_hoist: false,
            element_shards: 1,
            passes: 1,
        }
    }
}

impl fmt::Display for Tuning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(part) = self.part {
            write!(f, "_p{part}")?;
        }
        if self.cut {
            f.write_str("_cut")?;
        }
        if self.guard {
            f.write_str("_zs")?;
        }
        if self.io_lanes > 1 {
            write!(f, "_v{}", self.io_lanes)?;
        }
        if self.warp.1 > 1 {
            let axis = match self.warp.0 {
                WarpAxis::Position => "wp",
                WarpAxis::Hidden => "wu",
            };
            write!(f, "_{axis}{}", self.warp.1)?;
        }
        if let Some(lanes) = self.fiber_lanes {
            write!(f, "_fl{lanes}")?;
        }
        if self.lane_hoist {
            f.write_str("_lh")?;
        }
        if self.element_shards > 1 {
            write!(f, "_sh{}", self.element_shards)?;
        }
        if self.passes > 1 {
            write!(f, "_pp{}", self.passes)?;
        }
        Ok(())
    }
}

/// A primary kernel's entry symbol, built structurally rather than by string
/// concatenation.
#[derive(Clone, Debug)]
pub(super) struct KernelName {
    program: String,
    role: Role,
    block_size: u32,
}

impl KernelName {
    pub(super) fn consumer(program: &str, block_size: u32, tuning: Tuning) -> Self {
        Self {
            program: program.to_owned(),
            role: Role::Consumer(tuning),
            block_size,
        }
    }

    pub(super) fn producer(program: &str, output: usize, block_size: u32) -> Self {
        Self {
            program: program.to_owned(),
            role: Role::Producer { output },
            block_size,
        }
    }

    pub(super) fn reduction(program: &str, block_size: u32) -> Self {
        Self {
            program: program.to_owned(),
            role: Role::Reduction,
            block_size,
        }
    }

    /// The entry symbol, as a PTX identifier.
    pub(super) fn symbol(&self) -> String {
        to_ptx_identifier(&self.to_string())
    }
}

impl fmt::Display for KernelName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "loom_{}", self.program)?;
        match &self.role {
            Role::Consumer(tuning) => write!(f, "_c_bs{}{tuning}", self.block_size),
            Role::Producer { output } => write!(f, "_pr{output}_bs{}", self.block_size),
            Role::Reduction => write!(f, "_br_bs{}", self.block_size),
        }
    }
}

/// The projection that converts an output's accumulator into its storage dtype.
/// One entry serves every consumer candidate, so its symbol depends only on the
/// program and output, never on a flavor.
pub(super) fn shared_convert(program: &str, output: usize) -> String {
    format!(
        "{}_cvt{output}",
        to_ptx_identifier(&format!("loom_{program}"))
    )
}

/// The projection that sums an output's bucket copies away. It takes its bucket
/// count at launch, so one entry serves every arm of the ladder.
pub(super) fn shared_bucket_fold(program: &str, output: usize) -> String {
    format!(
        "{}_fold{output}",
        to_ptx_identifier(&format!("loom_{program}"))
    )
}
