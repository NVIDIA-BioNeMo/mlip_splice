// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use recipe::{Geometry, RecipeBody, Work};
use std::cmp::Ordering;

/// The GPU a compile renders and shortlists for: its compute capability and
/// the occupancy ceilings the work-per-resident-warp heuristic reads. These are
/// not register- or shared-memory-aware occupancy estimates.
#[derive(Clone, Copy, Debug)]
pub struct GpuTarget {
    /// Compute capability, e.g. `90` for sm_90.
    pub sm: u16,
    pub sm_count: u32,
    pub warp_size: u32,
    pub max_threads_per_sm: u32,
    pub max_blocks_per_sm: u32,
}

impl GpuTarget {
    pub(crate) fn validate(self) -> Result<Self, String> {
        if self.sm == 0 {
            return Err("loom target sm must be a real compute capability".into());
        }
        if [
            self.sm_count,
            self.warp_size,
            self.max_threads_per_sm,
            self.max_blocks_per_sm,
        ]
        .iter()
        .any(|&value| value == 0 || value > i32::MAX as u32)
            || self.warp_size > self.max_threads_per_sm
        {
            return Err("loom target requires positive CUDA device limits".into());
        }
        Ok(self)
    }

    /// The limits of device 0, the device the render target is read from.
    pub(crate) fn device() -> Result<Self, String> {
        use cuda_driver::Device;
        let query = || {
            let device = Device::get(0)?;
            Ok::<_, cuda_driver::Error>(Self {
                sm: device.sm_arch()? as u16,
                sm_count: device.sm_count()? as u32,
                warp_size: device.warp_size()? as u32,
                max_threads_per_sm: device.max_threads_per_sm()? as u32,
                max_blocks_per_sm: device.max_blocks_per_sm()? as u32,
            })
        };
        query()
            .map_err(|error| format!("loom shortlist device limits: {error}"))?
            .validate()
    }

    pub(super) fn resident_warps(self, geometry: &Geometry) -> Result<u64, String> {
        let threads = geometry
            .block
            .iter()
            .map(|&n| u128::from(n))
            .product::<u128>();
        let blocks = geometry
            .grid
            .iter()
            .map(|&n| u128::from(n))
            .product::<u128>();
        let warps = threads.div_ceil(u128::from(self.warp_size));
        if warps == 0 || blocks == 0 {
            return Err("loom shortlist launch geometry must be positive".into());
        }
        let slots = self.max_threads_per_sm / self.warp_size;
        let blocks_per_sm = u128::from(self.max_blocks_per_sm).min(u128::from(slots) / warps);
        if blocks_per_sm == 0 {
            return Err("loom shortlist block exceeds the target warp capacity".into());
        }
        Ok((warps * blocks.min(u128::from(self.sm_count) * blocks_per_sm)) as u64)
    }

    pub(super) fn bottleneck(self, body: &RecipeBody) -> Result<u64, String> {
        let mut minimum = None;
        self.visit_launches(body, &mut minimum)?;
        // A memory-only or empty program needs no resident warps.
        Ok(minimum.unwrap_or(1))
    }

    fn visit_launches(self, body: &RecipeBody, minimum: &mut Option<u64>) -> Result<(), String> {
        for node in body.nodes() {
            match node.work() {
                Work::Call { geometry, .. } => {
                    for launch in geometry {
                        let warps = self.resident_warps(launch)?;
                        *minimum = Some(minimum.map_or(warps, |old| old.min(warps)));
                    }
                }
                Work::Choice { candidates, .. } => {
                    // Source configurations fix every nested choice. Reject an
                    // unfixed configuration rather than score an arbitrary path.
                    if candidates.len() != 1 {
                        return Err("loom shortlist requires fixed source configurations".into());
                    }
                    self.visit_launches(&candidates[0], minimum)?;
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Score {
    pub work: u128,
    pub parallelism: u64,
}

impl Score {
    fn compare(self, other: Self) -> Ordering {
        // Work is the sum of two u64 counters; validated device limits bound
        // resident warps below 2^62. Both cross products therefore fit u128.
        (self.work * u128::from(other.parallelism))
            .cmp(&(other.work * u128::from(self.parallelism)))
    }

    fn value(self) -> f64 {
        self.work as f64 / self.parallelism as f64
    }
}

/// A fixed-seed Plackett–Luce permutation at tau=1, using median-normalized
/// work/parallelism energies and Gumbel keys. No runtime observations enter it.
pub(super) fn boltzmann_order(scores: &[Score]) -> Vec<usize> {
    let mut order = (0..scores.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| scores[a].compare(scores[b]).then(a.cmp(&b)));
    if order.is_empty() {
        return order;
    }
    let middle = order.len() / 2;
    let median = if order.len() % 2 == 0 {
        (scores[order[middle - 1]].value() + scores[order[middle]].value()) / 2.0
    } else {
        scores[order[middle]].value()
    };
    // When a majority do no arithmetic the median is zero. The smallest
    // positive score supplies a scale without breaking scale invariance;
    // an all-zero population is uniform, whatever scale it uses.
    let scale = if median > 0.0 {
        median
    } else {
        order
            .iter()
            .map(|&index| scores[index].value())
            .find(|&score| score > 0.0)
            .unwrap_or(1.0)
    };
    let mut random = 0x4c6f_6f6d_0000_0001u64;
    let keys = scores
        .iter()
        .map(|score| {
            // SplitMix64, then a midpoint in (0,1) from 52 random bits.
            random = random.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut bits = random;
            bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            bits ^= bits >> 31;
            let uniform = ((bits >> 12) as f64 + 0.5) / ((1u64 << 52) as f64);
            score.value() / scale + (-uniform.ln()).ln()
        })
        .collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        keys[a]
            .total_cmp(&keys[b])
            .then_with(|| scores[a].compare(scores[b]))
            .then(a.cmp(&b))
    });
    order
}

#[cfg(test)]
mod tests;
