// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::traffic_clusters::{InputTraffic, agglomerate_traffic};
use super::super::value_clusters::ValueCluster;
use super::super::*;

fn traffic_cluster(loads: &[usize], acc: (usize, usize)) -> ValueCluster {
    ValueCluster {
        atoms: Vec::new(),
        input_loads: loads.iter().copied().collect(),
        interior: BTreeSet::new(),
        accs: BTreeSet::from([acc]),
    }
}

/// The gathered-bytes budget is the only stop: pairs sharing the small
/// seam loads merge while their union's gathered set fits, and the big
/// per-group loads stay exactly partitioned.
#[test]
fn traffic_agglomeration_stops_at_the_gathered_budget() {
    // Loads 0..2 mimic per-path gathered fibers (40/30/20 bytes), load 3
    // the small shared gather, load 4 a large streamed operand.
    let traffic = InputTraffic {
        bytes: vec![40, 30, 20, 5, 100],
        gathered: vec![true, true, true, true, false],
    };
    let clusters = vec![
        traffic_cluster(&[0, 3, 4], (0, 0)),
        traffic_cluster(&[1, 3, 4], (0, 1)),
        traffic_cluster(&[2, 3, 4], (0, 2)),
    ];
    let acc_bytes = |_: &(usize, usize)| 10u64;
    let merged = agglomerate_traffic(clusters.clone(), &traffic, &acc_bytes, 50, 0);
    assert_eq!(merged.len(), 3);
    let merged = agglomerate_traffic(clusters, &traffic, &acc_bytes, 60, 0);
    assert_eq!(merged.len(), 2);
    let loads: Vec<_> = merged
        .iter()
        .map(|cluster| cluster.input_loads.iter().copied().collect::<Vec<_>>())
        .collect();
    assert!(loads.contains(&vec![0, 3, 4]));
    assert!(loads.contains(&vec![1, 2, 3, 4]));
}

/// A shared accumulator outweighs sharing nothing: the pair writing the
/// same output element merges first, and the budget then blocks the
/// rest.
#[test]
fn traffic_agglomeration_prefers_shared_accumulators() {
    let traffic = InputTraffic {
        bytes: vec![10, 10, 10],
        gathered: vec![true, true, true],
    };
    let clusters = vec![
        traffic_cluster(&[0], (0, 0)),
        traffic_cluster(&[1], (0, 0)),
        traffic_cluster(&[2], (0, 1)),
    ];
    let acc_bytes = |_: &(usize, usize)| 8u64;
    let merged = agglomerate_traffic(clusters, &traffic, &acc_bytes, 20, 0);
    assert_eq!(merged.len(), 2);
    let with_acc = merged
        .iter()
        .find(|cluster| cluster.input_loads.len() == 2)
        .unwrap();
    assert_eq!(with_acc.accs, BTreeSet::from([(0, 0)]));
}
