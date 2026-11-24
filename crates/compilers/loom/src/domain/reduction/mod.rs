// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Every operation output is a fold by key.
//!
//! A position's operand address is its key, addition is the reducer and zero
//! the identity, so an output is `out[k] = sum of the values whose key is k`.
//! A key no two positions share needs no combining and stores; a key they share
//! combines atomically. Buckets widen the key with a block-derived coordinate
//! that a projection pass sums away — the one degree of freedom the model has,
//! and the one the executor times.

use domain_index::DomainIndex;
use gpu_ir::{Evict, Scalar, Value};

mod project;

pub(crate) const FINALIZE_BLOCK_SIZE: u32 = 256;

const MAX_FOLD_TILE: u32 = FINALIZE_BLOCK_SIZE;

/// Ceiling on the transient the bucket copies may hold. It bounds memory, not
/// speed: which of the affordable counts is fastest is measured, never guessed.
const BUCKET_WORKSPACE_BYTES: usize = 64 << 20;

/// L2 sector granularity: the unit that serialises an address's atomic chain,
/// so every value sharing a sector shares the chain.
const ACCUMULATOR_LINE_BYTES: usize = 256;

/// Buckets buy their fold an extra kernel launch, so the chain they break has
/// to be at least a launch long — measured in the read-modify-writes one L2
/// sector serialises, each of which occupies its slice for a cycle. Below this
/// the ladder offers nothing to time; above it, measurement decides.
const MIN_CHAIN_DEPTH: i64 = 8192;

/// Number of independent L2-sector chains the center bucket count aims to
/// expose. Neighboring powers of two are measured around that center.
const TARGET_ACCUMULATOR_LINES: usize = 2048;

/// The run one bucket copy of one address is aimed at accumulating.
///
/// Every value in a run rounds into the running total, so the run length *is*
/// the accuracy a plan delivers: an address that receives `n` values carries
/// `n - 1` roundings, and spreading it over `b` copies leaves about `n / b` per
/// copy plus `b - 1` to sum the copies back. Where the first term dominates,
/// copies buy accuracy outright.
///
/// Measured on `symmetric_contraction`'s weight gradient — 1000 values per
/// address, u=192, 10000 receivers, RTX 5080 — the relative error against an
/// exact double-precision sum falls from 4.4 ppm at one copy to 0.53 ppm at
/// sixteen copies and 0.29 ppm at thirty-two, with no measurable change in time
/// and 6 MB of the 64 MB [`BUCKET_WORKSPACE_BYTES`] the ladder may spend. So
/// this is a floor the ladder must clear, not a preference: counts below it are
/// not offered, and speed is measured among the rest.
///
/// It is a target, not a guarantee, and [`OutputFold::longest_run`] is careful
/// not to report it as one. A copy is `blockIdx & (b - 1)`, so `n / b` per copy
/// assumes an address's positions spread evenly over those bits. A map is free
/// to correlate with them: with `species[r] = r % 10` and one position per
/// block, an address reaches only `b / gcd(10, b)` copies, so two copies shorten
/// nothing at all. Nothing the compiler reads can rule that out — an index map
/// carries extents and flags here, not target indices — so the spread is an
/// assumption about the data, held because it is what the measurement above
/// rests on.
const TARGET_ACCUMULATED_RUN: i64 = 64;

/// How many values one output address receives over the whole operation.
///
/// Both readings carry the count the extents give. It is what every address
/// receives when broadcast axes alone decide the fold; it is the mean over
/// addresses when a non-injective index map decides it, because that map's
/// target indices are runtime data and one address may take every position of
/// the iteration space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Collisions {
    /// Every address receives this many values.
    Uniform(i64),
    /// Addresses receive this many on average.
    Mean(i64),
}

impl Collisions {
    /// The count the extents give, under either reading.
    pub(crate) fn count(self) -> i64 {
        match self {
            Self::Uniform(count) | Self::Mean(count) => count,
        }
    }
}

/// Whether two positions of the operation can land on one output address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Key {
    /// Every address receives exactly one value, so the fold is a store.
    Unique,
    /// Positions collide, so combining must be atomic against other blocks.
    Colliding,
}

impl Key {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Unique => "unique",
            Self::Colliding => "colliding",
        }
    }
}

/// Why an output's bucket ladder stops where it does — what a diagnostic
/// answers when an output offers no further copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BucketCap {
    /// A unique key never collides, so a second bucket would cost a pass and
    /// shorten nothing.
    UniqueKey,
    /// Read-modify-writes one L2 sector serialises. Below [`MIN_CHAIN_DEPTH`]
    /// the chain is shorter than the launch that would split it.
    Chain(i64),
    /// Values per address. A bucket no position writes stays at the identity,
    /// so more buckets than this pay the projection and shorten no chain.
    Collisions(i64),
    /// The copies would exceed [`BUCKET_WORKSPACE_BYTES`].
    Workspace(usize),
    /// The output accumulates nothing.
    Empty,
}

impl BucketCap {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::UniqueKey => "unique_key",
            Self::Chain(_) => "chain",
            Self::Collisions(_) => "collisions",
            Self::Workspace(_) => "workspace",
            Self::Empty => "empty",
        }
    }

    /// The quantity that bound the ladder, in the unit its variant names.
    pub(crate) fn bound(self) -> i64 {
        match self {
            Self::UniqueKey | Self::Empty => 1,
            Self::Chain(depth) => depth,
            Self::Collisions(collisions) => collisions,
            Self::Workspace(bytes) => bytes as i64,
        }
    }
}

/// What projecting the bucket coordinate away takes at one bucket count.
/// `None` means the main kernel already left the output in its final form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Projection {
    /// One thread per element: the accumulator's dtype converted to storage.
    Convert,
    /// One block per element: the element's `buckets` copies summed, then
    /// converted.
    Fold { buckets: u32 },
}

impl Projection {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Convert => "convert",
            Self::Fold { .. } => "fold",
        }
    }

    /// Element-tile widths offered to a fold projection. The middle width
    /// targets 256 blocks; its neighbors target 512 and 128 blocks. Kiln
    /// measures which one fits the device and bucket count.
    pub(crate) fn tile_widths(self, elements: usize) -> Vec<u32> {
        if self == Self::Convert {
            return Vec::new();
        }
        let center = u32::try_from(elements.div_ceil(FINALIZE_BLOCK_SIZE as usize))
            .unwrap_or(u32::MAX)
            .clamp(1, MAX_FOLD_TILE)
            .next_power_of_two();
        let mut widths = vec![(center / 2).max(1), center, (center * 2).min(MAX_FOLD_TILE)];
        widths.dedup();
        widths
    }

    /// Grid and block this projection launches on for an output holding
    /// `elements` values. A fold's tile is the number of output elements one
    /// block produces.
    pub(crate) fn launch(
        self,
        elements: usize,
        tile_width: Option<u32>,
    ) -> Result<([u32; 3], [u32; 3]), String> {
        let count = u32::try_from(elements)
            .map_err(|_| "PTX fold projection element count exceeds u32".to_string())?;
        let grid = match self {
            Self::Convert => {
                if tile_width.is_some() {
                    return Err("PTX conversion projection does not take a tile width".into());
                }
                count.div_ceil(FINALIZE_BLOCK_SIZE)
            }
            Self::Fold { .. } => {
                let tile_width =
                    tile_width.ok_or("PTX fold projection needs an element-tile width")?;
                if !tile_width.is_power_of_two() || tile_width > MAX_FOLD_TILE {
                    return Err(format!(
                        "PTX fold projection tile width {tile_width} is not a power of two up to {MAX_FOLD_TILE}"
                    ));
                }
                count.div_ceil(tile_width)
            }
        };
        Ok(([grid.max(1), 1, 1], [FINALIZE_BLOCK_SIZE, 1, 1]))
    }
}

/// The entry symbols an output's projection kernels take.
#[derive(Clone, Debug)]
pub(crate) struct ProjectionEntries {
    pub convert: String,
    pub fold: String,
}

/// One output's fold by key: what combines, in which dtype, over how many
/// copies of the keyspace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OutputFold {
    key: Key,
    storage: Scalar,
    math: Scalar,
    /// Dtype the fold accumulates in — the output's own storage when the
    /// output buffer carries the accumulation, a wider one when a workspace
    /// does.
    accumulator: Scalar,
    /// Values one address receives over the whole operation, and how firmly the
    /// extents say so.
    collisions: Collisions,
    /// Bucket counts on offer, ascending, always starting at 1.
    buckets: Vec<u32>,
    cap: BucketCap,
}

impl OutputFold {
    /// `collisions` is how many values one address receives over the whole
    /// operation, `values_per_cell` how many the output holds per key cell, and
    /// `elements` how many it holds in all; together they say how deep the
    /// atomic chain runs and how many copies of the keyspace are affordable.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        key: Key,
        storage: Scalar,
        math: Scalar,
        defer_conversion: bool,
        collisions: Collisions,
        values_per_cell: i64,
        elements: usize,
    ) -> Result<Self, String> {
        let accumulator = match key {
            Key::Unique if defer_conversion && storage != math => math,
            Key::Unique => storage,
            Key::Colliding => match math {
                Scalar::S8 | Scalar::S16 => Scalar::S32,
                Scalar::U8 | Scalar::U16 => Scalar::U32,
                Scalar::Pred => {
                    return Err("boolean output reduction has no additive semantics".into());
                }
                scalar => scalar,
            },
        };
        let bytes = elements.saturating_mul(accumulator.size() as usize);
        let per_sector = (ACCUMULATOR_LINE_BYTES / accumulator.size().max(1) as usize) as i64;
        let (buckets, cap) = bucket_ladder(
            key,
            collisions.count(),
            collisions
                .count()
                .saturating_mul(values_per_cell.clamp(1, per_sector)),
            bytes,
        );
        Ok(Self {
            key,
            storage,
            math,
            accumulator,
            collisions,
            buckets,
            cap,
        })
    }

    pub(crate) fn key(&self) -> Key {
        self.key
    }

    pub(crate) fn collisions(&self) -> i64 {
        self.collisions.count()
    }

    /// An upper bound on the values one bucket copy of one address accumulates,
    /// where `positions_per_copy` counts the iteration positions a copy
    /// receives — its share of the blocks, times the positions a block covers.
    ///
    /// Nothing here divides by the number of copies. Dividing would assume that
    /// an address's positions spread evenly over `blockIdx & (b - 1)`, which is
    /// a property of the index map's contents, not of the program: see
    /// [`TARGET_ACCUMULATED_RUN`]. What holds whatever the map does is that a
    /// copy cannot accumulate more values than the positions it receives, and
    /// that a uniform fold's addresses each receive what the extents say. So
    /// copies shorten this figure only by shortening a copy's share of the
    /// launch — the sound half of the effect the measurement shows.
    pub(crate) fn longest_run(&self, positions_per_copy: i64) -> i64 {
        let positions = positions_per_copy.max(1);
        match (self.key, self.collisions) {
            (Key::Unique, _) => 1,
            (Key::Colliding, Collisions::Uniform(count)) => count.min(positions),
            (Key::Colliding, Collisions::Mean(_)) => positions,
        }
    }

    /// The dtype the kernel's output parameter points at.
    pub(crate) fn parameter_scalar(&self) -> Scalar {
        self.accumulator
    }

    /// Whether the accumulation buffer holds the output's own storage dtype, so
    /// a write through it is the width the output was declared at.
    pub(crate) fn stores_its_accumulator(&self) -> bool {
        self.storage == self.accumulator
    }

    /// A colliding key combines with `red.add`; a unique one stores.
    pub(crate) fn combines_atomically(&self) -> bool {
        self.key == Key::Colliding
    }

    pub(crate) fn bucket_counts(&self) -> &[u32] {
        &self.buckets
    }

    pub(crate) fn bucket_cap(&self) -> BucketCap {
        self.cap
    }

    /// Whether the kernel takes a bucket-mask parameter. A ladder of the lone
    /// count 1 needs no bucket arithmetic at all, so it pays none; any count
    /// above 1 does, even when the ladder offers nothing else to compare it
    /// with, or the copies it asked for would receive nothing.
    pub(crate) fn selects_buckets(&self) -> bool {
        self.buckets.iter().any(|&buckets| buckets > 1)
    }

    /// The largest count on offer — what sizes the workspace when one buffer
    /// must serve every arm.
    pub(crate) fn max_buckets(&self) -> u32 {
        *self.buckets.last().expect("a fold offers one bucket count")
    }

    /// What closes the accumulation at `buckets` copies: nothing when the main
    /// kernel already left the output in its storage form, a conversion when
    /// the accumulator's dtype is not the output's, a fold when the copies must
    /// be summed away.
    pub(crate) fn projection(&self, buckets: u32) -> Option<Projection> {
        if buckets > 1 {
            return Some(Projection::Fold { buckets });
        }
        (self.storage != self.accumulator).then_some(Projection::Convert)
    }

    pub(crate) fn accumulation_nbytes(
        &self,
        elements: usize,
        buckets: u32,
    ) -> Result<usize, String> {
        elements
            .checked_mul(buckets as usize)
            .and_then(|values| values.checked_mul(self.accumulator.size() as usize))
            .ok_or_else(|| "PTX fold accumulation size overflow".to_string())
    }

    /// Atomic arms reduce with `red.add`, which has no eviction qualifier; the
    /// hint applies to the plain store only.
    /// `value` arrives in the program's math dtype, so it converts only where
    /// the accumulator differs.
    pub(crate) fn write(&self, b: &mut gpu_ir::Builder, ptr: Value, value: Value, evict: Evict) {
        let value = self.in_accumulator(b, value);
        if self.combines_atomically() {
            b.red_add(ptr, value)
        } else {
            b.store_evict(ptr, value, evict)
        }
    }

    /// Adds `value` to the partial already at `ptr`, then writes it back. The
    /// accumulation buffer holds the accumulator's dtype, so the value joins it
    /// there rather than the partial being read at the dtype of the value.
    pub(crate) fn accumulate(
        &self,
        b: &mut gpu_ir::Builder,
        ptr: Value,
        value: Value,
        lanes: u8,
        evict: Evict,
    ) {
        let value = self.in_accumulator(b, value);
        let partial = b.load(ptr.clone(), lanes, false);
        let accumulated = b.add(partial, value);
        if self.combines_atomically() {
            b.red_add(ptr, accumulated)
        } else {
            b.store_evict(ptr, accumulated, evict)
        }
    }

    /// `value` in the dtype the accumulation buffer holds.
    fn in_accumulator(&self, b: &mut gpu_ir::Builder, value: Value) -> Value {
        if self.accumulator == self.math {
            value
        } else {
            b.cast(self.accumulator, value)
        }
    }

    /// This block's bucket, as an element offset into an accumulation buffer
    /// holding `elements` values per copy. `mask` names the `s32` parameter
    /// carrying `buckets - 1`, so one kernel serves every count on the ladder.
    pub(crate) fn bucket_offset(
        &self,
        b: &mut gpu_ir::Builder,
        mask: gpu_ir::Value,
        elements: i64,
    ) -> Option<gpu_ir::Value> {
        self.selects_buckets().then(|| {
            let block = b.ctaid_x();
            let block = b.cast(Scalar::S32, block);
            let bucket = b.and(block, mask);
            let bucket = b.cast(Scalar::S64, bucket);
            let elements = b.i64(elements);
            b.mul(bucket, elements)
        })
    }
}

/// The bucket counts an output offers and why the ladder stops. Counts are
/// powers of two so a block's bucket is a mask of its index.
///
/// Two rules meet here. [`TARGET_ACCUMULATED_RUN`] is a floor: it says how few
/// copies an address's run may be spread over, and counts below it are not
/// offered at all. [`TARGET_ACCUMULATOR_LINES`] is a guess at which of the rest
/// runs fastest, and the counts around it are offered so measurement can
/// decide. A floor the workspace cannot afford is approached as far as it
/// reaches; it never refuses the fold.
fn bucket_ladder(key: Key, collisions: i64, chain: i64, bytes: usize) -> (Vec<u32>, BucketCap) {
    if key == Key::Unique {
        return (vec![1], BucketCap::UniqueKey);
    }
    if bytes == 0 {
        return (vec![1], BucketCap::Empty);
    }
    let affordable = BUCKET_WORKSPACE_BYTES / bytes;
    let collisions_bound = usize::try_from(collisions).unwrap_or(usize::MAX);
    let (bound, cap) = if affordable <= collisions_bound {
        (affordable, BucketCap::Workspace(BUCKET_WORKSPACE_BYTES))
    } else {
        (collisions_bound, BucketCap::Collisions(collisions))
    };
    if bound < 2 {
        return (vec![1], cap);
    }
    let cap_count = bound.min(u32::MAX as usize).isolate_highest_one() as u32;
    let required = copies_for_target_run(collisions).min(cap_count).max(1);
    if required < 2 && chain < MIN_CHAIN_DEPTH {
        return (vec![1], BucketCap::Chain(chain));
    }
    let accumulator_lines = bytes.div_ceil(ACCUMULATOR_LINE_BYTES);
    let target = TARGET_ACCUMULATOR_LINES.div_ceil(accumulator_lines);
    let center_bound = target.min(bound).min(u32::MAX as usize);
    if required < 2 && center_bound < 2 {
        return (vec![1], cap);
    }
    let center = center_bound.isolate_highest_one() as u32;
    // The floor and its neighbour keep measurement a choice even when the floor
    // stands above the count that spreads sectors best.
    let mut counts = vec![
        required,
        required.saturating_mul(2),
        (center / 2).max(1),
        center,
        center.saturating_mul(2),
    ];
    counts.retain(|&count| count >= required && count <= cap_count);
    counts.sort_unstable();
    counts.dedup();
    if counts.is_empty() {
        counts.push(required.min(cap_count).max(1));
    }
    (counts, cap)
}

/// The fewest bucket copies — a power of two, so a block's copy is a mask of its
/// index — that leave one copy of one address accumulating at most
/// [`TARGET_ACCUMULATED_RUN`] values, on the assumption that the copies share an
/// address's values evenly.
fn copies_for_target_run(collisions: i64) -> u32 {
    let copies = collisions
        .saturating_add(TARGET_ACCUMULATED_RUN - 1)
        .div_euclid(TARGET_ACCUMULATED_RUN)
        .max(1);
    u32::try_from(copies)
        .unwrap_or(u32::MAX)
        .checked_next_power_of_two()
        .unwrap_or(1 << 31)
}

/// Every output of one operation as a fold by key.
pub(crate) struct Folds {
    num_inputs: usize,
    parameter_scalars: Vec<Scalar>,
    outputs: Vec<OutputFold>,
}

impl Folds {
    /// `output_elements[j]` counts the values output `j` holds; it sizes the
    /// accumulator and, against the iteration extent, says how many values one
    /// address receives.
    pub(crate) fn new<I: DomainIndex>(
        index: &I,
        num_inputs: usize,
        storage_scalars: &[Scalar],
        output_math_scalars: &[Scalar],
        defer_output_conversion: &[bool],
        output_elements: &[usize],
    ) -> Result<Self, String> {
        if storage_scalars.len() != index.num_operands() {
            return Err(format!(
                "got {} fold storage scalars for {} operands",
                storage_scalars.len(),
                index.num_operands()
            ));
        }
        if num_inputs > storage_scalars.len() {
            return Err(format!(
                "fold num_inputs={num_inputs} exceeds operands={}",
                storage_scalars.len()
            ));
        }
        if output_elements.len() != storage_scalars.len() - num_inputs {
            return Err(format!(
                "got {} fold element counts for {} outputs",
                output_elements.len(),
                storage_scalars.len() - num_inputs
            ));
        }
        if output_math_scalars.len() != output_elements.len() {
            return Err(format!(
                "got {} output math scalars for {} outputs",
                output_math_scalars.len(),
                output_elements.len()
            ));
        }
        if defer_output_conversion.len() != output_elements.len() {
            return Err(format!(
                "got {} output conversion modes for {} outputs",
                defer_output_conversion.len(),
                output_elements.len()
            ));
        }
        let outputs = storage_scalars[num_inputs..]
            .iter()
            .enumerate()
            .map(|(output, &storage)| {
                let operand = num_inputs + output;
                let cells = index.operand_batch_extent(operand)?.max(1);
                let count = index.batch_total()? / cells;
                OutputFold::new(
                    if index.needs_atomic(operand) {
                        Key::Colliding
                    } else {
                        Key::Unique
                    },
                    storage,
                    output_math_scalars[output],
                    defer_output_conversion[output],
                    if index.collisions_are_uniform(operand) {
                        Collisions::Uniform(count)
                    } else {
                        Collisions::Mean(count)
                    },
                    output_elements[output] as i64 / cells,
                    output_elements[output],
                )
            })
            .collect::<Result<Vec<_>, String>>()?;
        let parameter_scalars = storage_scalars[..num_inputs]
            .iter()
            .copied()
            .chain(outputs.iter().map(|fold| fold.parameter_scalar()))
            .collect();
        Ok(Self {
            num_inputs,
            parameter_scalars,
            outputs,
        })
    }

    pub(crate) fn parameter_scalars(&self) -> &[Scalar] {
        &self.parameter_scalars
    }

    pub(crate) fn output(&self, output: usize) -> &OutputFold {
        &self.outputs[output]
    }

    pub(crate) fn operand(&self, operand: usize) -> &OutputFold {
        self.output(operand - self.num_inputs)
    }

    /// The outputs whose kernels take a bucket-mask parameter, in output order.
    pub(crate) fn bucket_selectors(&self) -> Vec<usize> {
        (0..self.outputs.len())
            .filter(|&output| self.outputs[output].selects_buckets())
            .collect()
    }

    /// Every bucket assignment this operation offers, in ascending order. A
    /// rung has the same physical meaning for every output: direct, below the
    /// sector-spreading center, at the center, or above it.
    pub(crate) fn bucket_arms(&self) -> Vec<Vec<u32>> {
        let rungs = self
            .outputs
            .iter()
            .map(|fold| fold.bucket_counts().len())
            .max()
            .unwrap_or(0);
        let mut arms: Vec<Vec<u32>> = Vec::new();
        for rung in 0..rungs {
            let arm = self
                .outputs
                .iter()
                .map(|fold| {
                    fold.bucket_counts()
                        .get(rung)
                        .copied()
                        .unwrap_or_else(|| fold.max_buckets())
                })
                .collect::<Vec<_>>();
            if arms.last() != Some(&arm) {
                arms.push(arm);
            }
        }
        arms
    }
}

#[cfg(test)]
mod tests;
