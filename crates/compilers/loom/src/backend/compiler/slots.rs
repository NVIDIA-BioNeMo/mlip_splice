// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{Settings, WARP_SIZE};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum CandidateFlavor {
    Baseline,
    /// The baseline render assembled whole-program instead of relocatably.
    WholeProgram,
    Geometry,
    ZeroSkip,
    LaneHoist,
    RegCap,
}

impl CandidateFlavor {
    pub(super) fn name(self) -> &'static str {
        match self {
            CandidateFlavor::Baseline => "baseline",
            CandidateFlavor::WholeProgram => "whole_program",
            CandidateFlavor::Geometry => "geometry",
            CandidateFlavor::ZeroSkip => "zero_skip",
            CandidateFlavor::LaneHoist => "lane_hoist",
            CandidateFlavor::RegCap => "reg_cap",
        }
    }
}

/// Admission prior for a geometry cross, derived from 782 RTX5080+B200
/// scheduling sweeps and scaled to an impact
/// multiplier. wp4@128 keeps its historical weight — it is loom's most
/// frequent non-baseline winner on the current gallery.
fn geometry_impact_weight(
    fiber: bool,
    warps: u8,
    io_lanes: u8,
    block_size: u32,
    shards: u8,
    passes: u8,
    warp_u: bool,
) -> f64 {
    // A warp u-split is the arrangement u1d's `warps` tunings measure, so it
    // shares their win-order prior regardless of the fiber bridge.
    if warp_u {
        return geometry_impact_weight(false, warps, io_lanes, block_size, shards, passes, false);
    }
    // Fiber-iteration launches have their own winner history (2026-07-21
    // benzene/tensornet snd measurements): hidden-lane vec2 — plain at bs32
    // and with the shard axis — and the multi-warp walk split. The u1d table
    // below encodes u-split geometries fiber kernels do not share.
    if fiber {
        return match (warps, io_lanes, block_size, shards, passes) {
            (1, 2, 32, 1, 1) => 9.0,
            (1, 2, _, 2 | 3, 1) => 6.0,
            (2 | 4, 1 | 2, _, 1, 1) => 4.0,
            (1, 2, _, 1, 1) => 3.5,
            (2 | 4, _, _, 2 | 3, 1) => 2.0,
            _ => 1.0,
        };
    }
    // Passes only exist for register-fat bodies (the slot gate) and target
    // exactly the rows where caps and shards lose to the plain baseline, so
    // the plain pass crosses rank above the generic split prior; the wide
    // crosses stay speculative.
    if passes > 1 {
        return match (warps, io_lanes) {
            (1, 1) | (4, 1) => 3.0,
            _ => 1.5,
        };
    }
    match (warps, io_lanes, block_size, shards) {
        (4, 1, 128, 2 | 3) => 6.0,
        (4, 4, 128, 3) => 5.0,
        (4, 4, 128, 2) => 4.0,
        (2, 2, _, 3) | (1, 1, _, 3) | (1, 2, 32, 3) | (4, 2, 256, 3) => 2.0,
        (_, _, _, 2..) => 1.5,
        (1, 2, 32, _) => 9.0,
        (1, 4, 32, _) => 7.0,
        (4, 2, 128, _) => 4.0,
        (4, 1, 128, _) => 4.0,
        (2, 2, 256, _) => 3.5,
        (1, 2, 128, _) => 3.5,
        (2, 2, 128, _) => 3.0,
        (1, 4, 128, _) => 2.0,
        _ => 1.0,
    }
}

/// A slot's admission prior. A fiber-lane reduction is its own arrangement
/// with its own history; every other geometry reads the cross table.
pub(super) fn slot_weight(fiber: bool, spec: &SlotSpec) -> f64 {
    if spec.fiber_lanes {
        12.0
    } else {
        geometry_impact_weight(
            fiber,
            spec.warps,
            spec.io_lanes,
            spec.block_size,
            spec.shards,
            spec.passes,
            spec.warp_u,
        )
    }
}

/// One candidate slot of a kernel part: the launch geometry and flavor that
/// slot's kernel was rendered for. The slot index is a `PartTarget::Kernel`
/// block index; emission, admission, and launch lowering all read this table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct SlotSpec {
    pub(super) flavor: CandidateFlavor,
    pub(super) block_size: u32,
    pub(super) io_lanes: u8,
    pub(super) warps: u8,
    /// Warps split the hidden axis instead of the fiber walk — distinct
    /// arrangements only over a fiber bridge (`kernel::emit_geometry_kernel`).
    pub(super) warp_u: bool,
    pub(super) fiber_lanes: bool,
    pub(super) shards: u8,
    pub(super) passes: u8,
    pub(super) register_cap: Option<u32>,
}

impl SlotSpec {
    /// The plain arrangement of a flavor at one block size: no cap, no split,
    /// one warp per position, one lane per thread.
    pub(super) fn plain(flavor: CandidateFlavor, block_size: u32) -> Self {
        Self {
            flavor,
            block_size,
            io_lanes: 1,
            warps: 1,
            warp_u: false,
            fiber_lanes: false,
            shards: 1,
            passes: 1,
            register_cap: None,
        }
    }

    pub(super) fn positions_per_block(&self) -> u32 {
        self.block_size / (WARP_SIZE * u32::from(self.warps))
    }
}

/// One family of slots: the flavor, and the axes crossed to enumerate it. The
/// table is the concatenation of every family's cross product, taken in the
/// field order below and then over the block sizes, so a slot index is stable
/// as long as the families are.
struct SlotFamily {
    flavor: CandidateFlavor,
    fiber_lanes: bool,
    register_caps: Vec<Option<u32>>,
    /// `(shards, passes)` — the whole-element split, at most one axis at a time.
    splits: Vec<(u8, u8)>,
    /// `(warps, io_lanes)` — the launch geometry cross.
    crosses: Vec<(u8, u8)>,
    /// Whether the warps split the hidden axis instead of the fiber walk.
    warp_us: Vec<bool>,
}

impl SlotFamily {
    /// One slot per block size: no cap, no split, one warp, one lane.
    fn plain(flavor: CandidateFlavor) -> Self {
        Self {
            flavor,
            fiber_lanes: false,
            register_caps: vec![None],
            splits: vec![(1, 1)],
            crosses: vec![(1, 1)],
            warp_us: vec![false],
        }
    }
}

fn capped(caps: &[u32]) -> Vec<Option<u32>> {
    caps.iter().copied().map(Some).collect()
}

/// The zero-skip guard votes across one warp's lanes, so its wide slots stay
/// at one warp and its multi-warp slots at one lane.
const HIDDEN_LANE_WIDTHS: [(u8, u8); 3] = [(1, 2), (1, 4), (1, 8)];
const WARP_SPLITS: [(u8, u8); 2] = [(2, 1), (4, 1)];

fn slot_families(settings: &Settings) -> Vec<SlotFamily> {
    let geometry_crosses = |keep: fn(&(u8, u8)) -> bool| {
        settings
            .geometry_crosses
            .into_iter()
            .filter(keep)
            .collect::<Vec<_>>()
    };
    let mut families = [
        CandidateFlavor::Baseline,
        CandidateFlavor::WholeProgram,
        CandidateFlavor::ZeroSkip,
        CandidateFlavor::LaneHoist,
    ]
    .map(SlotFamily::plain)
    .into_iter()
    .collect::<Vec<_>>();
    families.push(SlotFamily {
        fiber_lanes: true,
        ..SlotFamily::plain(CandidateFlavor::Geometry)
    });
    families.push(SlotFamily {
        crosses: HIDDEN_LANE_WIDTHS.to_vec(),
        ..SlotFamily::plain(CandidateFlavor::ZeroSkip)
    });
    families.push(SlotFamily {
        crosses: WARP_SPLITS.to_vec(),
        warp_us: vec![false, true],
        ..SlotFamily::plain(CandidateFlavor::ZeroSkip)
    });
    families.push(SlotFamily {
        register_caps: capped(&settings.register_caps),
        crosses: std::iter::once((1, 1)).chain(HIDDEN_LANE_WIDTHS).collect(),
        ..SlotFamily::plain(CandidateFlavor::ZeroSkip)
    });
    families.push(SlotFamily {
        register_caps: capped(&settings.register_caps),
        ..SlotFamily::plain(CandidateFlavor::RegCap)
    });
    families.push(SlotFamily {
        crosses: settings.geometry_crosses.to_vec(),
        ..SlotFamily::plain(CandidateFlavor::Geometry)
    });
    // The warp u-split arrangement of the same crosses; only fiber-bridge
    // launches compile them (elsewhere they duplicate the plain cross).
    families.push(SlotFamily {
        crosses: geometry_crosses(|&(warps, _)| warps != 1),
        warp_us: vec![true],
        ..SlotFamily::plain(CandidateFlavor::Geometry)
    });
    // Register-capped clones of the vector crosses: vector lanes widen every
    // live range, and the driver JIT can allocate far more registers than an
    // offline assembly of the same PTX (218 vs 168 measured on a v2 kernel),
    // so occupancy relief is a measurable variant. Appended after the
    // uncapped crosses so equal-impact ties admit the uncapped sibling first.
    families.push(SlotFamily {
        register_caps: capped(&settings.register_caps[..1]),
        crosses: geometry_crosses(|&(_, io_lanes)| io_lanes != 1),
        ..SlotFamily::plain(CandidateFlavor::Geometry)
    });
    for split in [
        settings.split_counts.map(|shards| (shards, 1)),
        settings.split_counts.map(|passes| (1, passes)),
    ] {
        for (shards, passes) in split {
            families.push(SlotFamily {
                splits: vec![(shards, passes)],
                crosses: settings.split_crosses.to_vec(),
                ..SlotFamily::plain(CandidateFlavor::Geometry)
            });
        }
    }
    families.push(SlotFamily {
        register_caps: capped(&[160, 168, 176]),
        crosses: WARP_SPLITS.to_vec(),
        warp_us: vec![false, true],
        ..SlotFamily::plain(CandidateFlavor::ZeroSkip)
    });
    families.push(SlotFamily {
        register_caps: capped(&[160, 176]),
        crosses: geometry_crosses(|&(_, io_lanes)| io_lanes != 1),
        ..SlotFamily::plain(CandidateFlavor::Geometry)
    });
    families
}

/// Every slot a launch may offer, with the two a qualifying launch always
/// compiles whatever the admission budget allows: the leading register cap
/// and the leading zero-skip guard.
pub(super) struct SlotTable {
    pub(super) specs: Vec<SlotSpec>,
    pub(super) first_regcap: Option<usize>,
}

impl SlotTable {
    pub(super) fn new(block_sizes: &[u32], settings: &Settings) -> Self {
        let specs = slot_table(block_sizes, settings);
        let first = |flavor| specs.iter().position(|spec| spec.flavor == flavor);
        Self {
            first_regcap: first(CandidateFlavor::RegCap),
            specs,
        }
    }

    pub(super) fn indexed(block_sizes: &[u32]) -> Self {
        let mut specs = vec![
            SlotSpec::plain(CandidateFlavor::Baseline, block_sizes[0]),
            SlotSpec::plain(CandidateFlavor::ZeroSkip, block_sizes[0]),
        ];
        for warps in [2, 4] {
            let threads = WARP_SIZE * u32::from(warps);
            if let Some(block_size) = block_sizes
                .iter()
                .copied()
                .find(|block_size| *block_size >= threads && block_size.is_multiple_of(threads))
            {
                let mut spec = SlotSpec::plain(CandidateFlavor::Geometry, block_size);
                spec.warps = warps;
                specs.push(spec);
            }
        }
        Self {
            specs,
            first_regcap: None,
        }
    }
}

fn slot_table(block_sizes: &[u32], settings: &Settings) -> Vec<SlotSpec> {
    let mut table = Vec::new();
    for family in slot_families(settings) {
        for &register_cap in &family.register_caps {
            for &(shards, passes) in &family.splits {
                for &(warps, io_lanes) in &family.crosses {
                    for &block_size in block_sizes {
                        if !block_size.is_multiple_of(WARP_SIZE * u32::from(warps)) {
                            continue;
                        }
                        for &warp_u in &family.warp_us {
                            table.push(SlotSpec {
                                flavor: family.flavor,
                                block_size,
                                io_lanes,
                                warps,
                                warp_u,
                                fiber_lanes: family.fiber_lanes,
                                shards,
                                passes,
                                register_cap,
                            });
                        }
                    }
                }
            }
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_programs_offer_plain_zero_skip_and_multi_warp_slots() {
        let table = SlotTable::indexed(&[32, 64, 128]);
        assert_eq!(table.specs.len(), 4);
        assert_eq!(table.specs[0].flavor, CandidateFlavor::Baseline);
        assert_eq!(table.specs[0].block_size, 32);
        assert_eq!(table.specs[1].flavor, CandidateFlavor::ZeroSkip);
        assert_eq!(table.specs[1].block_size, 32);
        assert_eq!(table.specs[2].flavor, CandidateFlavor::Geometry);
        assert_eq!(table.specs[2].block_size, 64);
        assert_eq!(table.specs[2].warps, 2);
        assert_eq!(table.specs[3].block_size, 128);
        assert_eq!(table.specs[3].warps, 4);
        assert!(table.first_regcap.is_none());
    }
}
