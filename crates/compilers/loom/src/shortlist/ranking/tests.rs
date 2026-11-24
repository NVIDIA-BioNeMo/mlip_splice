// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn target() -> GpuTarget {
    GpuTarget {
        sm: 100,
        sm_count: 148,
        warp_size: 32,
        max_threads_per_sm: 2048,
        max_blocks_per_sm: 32,
    }
}

#[test]
fn occupancy_uses_grid_warp_rounding_and_both_sm_limits() {
    let target = target().validate().unwrap();
    let warps = |blocks, threads| {
        target
            .resident_warps(&Geometry::new([blocks, 1, 1], [threads, 1, 1], 0))
            .unwrap()
    };
    assert_eq!(warps(1, 33), 2);
    assert_eq!(warps(100_000, 32), 148 * 32);
    assert_eq!(warps(100_000, 256), 148 * 64);
    assert_eq!(warps(2, 1024), 64);
    assert_eq!(warps(100_000, 96), 148 * 21 * 3);
    let smaller = GpuTarget {
        sm_count: 1,
        max_blocks_per_sm: 4,
        ..target
    };
    assert_eq!(
        smaller
            .resident_warps(&Geometry::new([100, 1, 1], [32, 1, 1], 0))
            .unwrap(),
        4
    );
}

#[test]
fn invalid_limits_and_unrealizable_geometry_are_refused() {
    assert!(
        GpuTarget {
            warp_size: 0,
            ..target()
        }
        .validate()
        .is_err()
    );
    assert!(
        GpuTarget {
            sm_count: u32::MAX,
            ..target()
        }
        .validate()
        .is_err()
    );
    for (grid, block) in [
        ([0, 1, 1], [1, 1, 1]),
        ([1, 1, 1], [0, 1, 1]),
        ([1, 1, 1], [4096, 1, 1]),
    ] {
        assert!(
            target()
                .resident_warps(&Geometry::new(grid, block, 0))
                .is_err()
        );
    }
}

#[test]
fn ratios_compare_exactly_even_when_float_scores_tie() {
    let smaller = Score {
        work: u64::MAX as u128,
        parallelism: 1,
    };
    let larger = Score {
        work: smaller.work + 1,
        parallelism: 1,
    };
    assert_eq!(smaller.value(), larger.value());
    assert_eq!(smaller.compare(larger), Ordering::Less);
    let maximum = Score {
        work: 2 * u64::MAX as u128,
        parallelism: (i32::MAX as u64).pow(2),
    };
    assert_eq!(maximum.compare(maximum), Ordering::Equal);
    assert_eq!(
        Score {
            work: 2,
            parallelism: 3
        }
        .compare(Score {
            work: 4,
            parallelism: 6
        }),
        Ordering::Equal
    );
}

#[test]
fn boltzmann_is_a_reproducible_scale_invariant_permutation() {
    let scores = (1..=20)
        .map(|work| Score {
            work,
            parallelism: 32,
        })
        .collect::<Vec<_>>();
    let order = boltzmann_order(&scores);
    assert_eq!(order, boltzmann_order(&scores));
    let scaled = scores
        .iter()
        .map(|score| Score {
            work: score.work * 19,
            ..*score
        })
        .collect::<Vec<_>>();
    assert_eq!(order, boltzmann_order(&scaled));
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..scores.len()).collect::<Vec<_>>());
    assert_ne!(order, sorted);
    assert!(boltzmann_order(&[]).is_empty());
    assert_eq!(
        boltzmann_order(&[Score {
            work: 0,
            parallelism: 1
        }]),
        [0]
    );
}

#[test]
fn a_large_energy_gap_beats_gumbel_noise() {
    let scores = [
        Score {
            work: 1_000_000,
            parallelism: 1,
        },
        Score {
            work: 1,
            parallelism: 1,
        },
        Score {
            work: 1,
            parallelism: 1,
        },
    ];
    assert_eq!(boltzmann_order(&scores).last(), Some(&0));
}

#[test]
fn zero_median_keeps_scale_invariance_and_zero_work_is_valid() {
    let scores = [0, 0, 0, 1, 2].map(|work| Score {
        work,
        parallelism: 1,
    });
    let scaled = scores.map(|score| Score {
        work: score.work * 100,
        ..score
    });
    assert_eq!(boltzmann_order(&scores), boltzmann_order(&scaled));
    assert_eq!(boltzmann_order(&scores[..3]).len(), 3);
}
