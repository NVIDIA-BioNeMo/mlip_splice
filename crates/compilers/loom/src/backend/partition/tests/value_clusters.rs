// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use gpu_ir::Value;

use super::super::value_clusters::value_cluster_parts;
use super::super::*;
use super::support::*;

/// A profile whose outputs are built by `build` over inputs the sketch names on
/// demand, then clustered under `cap`.
fn cluster_parts(
    build: impl FnOnce(&mut Sketch) -> Vec<Vec<Value>>,
    cap: usize,
) -> Option<Vec<KernelPart>> {
    let mut sketch = Sketch::new();
    let outputs = build(&mut sketch);
    let sketched = sketch.finish();
    let outputs = outputs
        .into_iter()
        .map(|values| {
            values
                .into_iter()
                .map(|value| value.id_in(sketched.function()))
                .collect::<Vec<_>>()
        })
        .collect();
    value_cluster_parts(&incidence(sketched, outputs), cap, &Settings::default()).unwrap()
}

/// `x{2i} * x{2i+1}` for each of `count` elements: disjoint operand loads.
fn disjoint_products(sketch: &mut Sketch, count: usize) -> Vec<Vec<Value>> {
    let mut elements = Vec::with_capacity(count);
    for index in 0..count {
        let left = sketch.input(&format!("x{}", 2 * index));
        let right = sketch.input(&format!("x{}", 2 * index + 1));
        elements.push(sketch.mul(left, right));
    }
    vec![elements]
}

fn part_elements(part: &KernelPart) -> Vec<usize> {
    part.elements[0]
        .iter()
        .map(|selection| selection.element)
        .collect()
}

/// Elements with disjoint operand loads split by the footprint cap
/// alone: two launches of two elements each, no accumulator continued.
#[test]
fn value_clusters_respect_the_footprint_cap() {
    let parts = cluster_parts(|sketch| disjoint_products(sketch, 4), 6).unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(part_elements(&parts[0]), [0, 1]);
    assert_eq!(part_elements(&parts[1]), [2, 3]);
    assert!(parts.iter().all(|part| !part.output_continues(0)));
}

/// Elements sharing an operand load cluster together even across
/// trace-order neighbors: affinity, not adjacency, decides the launches.
#[test]
fn value_clusters_group_load_sharing_atoms() {
    let parts = cluster_parts(
        |sketch| {
            let shared = ["y", "z", "y", "z"];
            let own = ["a0", "b0", "a1", "b1"];
            let mut elements = Vec::new();
            let mut named = std::collections::BTreeMap::new();
            for (left, right) in shared.into_iter().zip(own) {
                let left = named
                    .entry(left)
                    .or_insert_with(|| sketch.input(left))
                    .clone();
                let right = sketch.input(right);
                elements.push(sketch.mul(left, right));
            }
            vec![elements]
        },
        5,
    )
    .unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(part_elements(&parts[0]), [0, 2]);
    assert_eq!(part_elements(&parts[1]), [1, 3]);
}

/// One cluster (everything fits), a single atom, and a split whose
/// launches still hold most of the fused footprint all decline.
#[test]
fn value_clusters_decline_degenerate_partitions() {
    let cap = Settings::default().value_cluster_cap;
    assert!(
        cluster_parts(
            |sketch| {
                let x = sketch.input("x");
                let y = sketch.input("y");
                let z = sketch.input("z");
                let first = sketch.mul(x.clone(), y);
                let second = sketch.mul(x, z);
                vec![vec![first, second]]
            },
            cap,
        )
        .is_none()
    );
    assert!(
        cluster_parts(
            |sketch| {
                let x = sketch.input("x");
                let y = sketch.input("y");
                vec![vec![sketch.mul(x, y)]]
            },
            cap,
        )
        .is_none()
    );
    assert!(
        cluster_parts(
            |sketch| {
                let x = sketch.input("x");
                let y = sketch.input("y");
                let a = sketch.input("a");
                let b = sketch.input("b");
                let left = sketch.mul(y.clone(), a);
                let first = sketch.mul(x.clone(), left);
                let right = sketch.mul(y, b);
                let second = sketch.mul(x, right);
                vec![vec![first, second]]
            },
            4,
        )
        .is_none()
    );
}

/// Additive terms of one element that cannot share a launch become an
/// ordered continuation: the later launch adds into the written element.
#[test]
fn value_clusters_split_elements_into_continuations() {
    let parts = cluster_parts(
        |sketch| {
            let product = |sketch: &mut Sketch, prefix: &str| {
                let mut held: Option<Value> = None;
                for index in 0..8 {
                    let factor = sketch.input(&format!("{prefix}{index}"));
                    held = Some(match held {
                        None => factor,
                        Some(held) => sketch.mul(held, factor),
                    });
                }
                held.expect("eight factors")
            };
            let left = product(sketch, "a");
            let right = product(sketch, "b");
            vec![vec![sketch.add(left, right)]]
        },
        8,
    )
    .unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].elements[0][0].terms, [0]);
    assert_eq!(parts[1].elements[0][0].terms, [1]);
    assert!(!parts[0].output_continues(0));
    assert!(parts[1].output_continues(0));
}

/// Two output families whose per-path atoms share operand loads while
/// each family shares interior arithmetic: absolute gain merges within
/// the families first and re-reads every shared load; the normalized
/// priority merges each path's cross-output atoms into one launch.
#[test]
fn value_clusters_cross_output_families_on_shared_loads() {
    let parts = cluster_parts(
        |sketch| {
            let chain = |sketch: &mut Sketch, prefix: &str, root: &str| {
                let mut last = sketch.input(root);
                for index in 0..6 {
                    let scale = sketch.f32(2.0);
                    let step = sketch.mul(last, scale);
                    last = sketch.bind(&format!("{prefix}{index}"), step);
                }
                last
            };
            let first_tail = chain(sketch, "d0_", "c0");
            let second_tail = chain(sketch, "d1_", "c1");
            let element = |sketch: &mut Sketch, path: usize, tail: &Value| {
                let left = sketch.input(&format!("s{path}0"));
                let right = sketch.input(&format!("s{path}1"));
                let product = sketch.mul(left, right);
                sketch.mul(product, tail.clone())
            };
            let first = (0..4)
                .map(|path| element(sketch, path, &first_tail))
                .collect::<Vec<_>>();
            let second = (0..4)
                .map(|path| element(sketch, path, &second_tail))
                .collect::<Vec<_>>();
            vec![first, second]
        },
        8,
    )
    .unwrap();
    assert_eq!(parts.len(), 4);
    for (path, part) in parts.iter().enumerate() {
        assert_eq!(part_elements(part), [path]);
        let mixed = part.elements[1]
            .iter()
            .map(|selection| selection.element)
            .collect::<Vec<_>>();
        assert_eq!(mixed, [path]);
    }
}

/// The cap proposal comes from the part: a fused footprint that cannot
/// double the cap declines before clustering, while the same program
/// clusters under a cap it doubles.
#[test]
fn value_clusters_decline_caps_the_footprint_cannot_double() {
    // 8 loads + 4 accumulators: footprint 12.
    assert!(cluster_parts(|sketch| disjoint_products(sketch, 4), 7).is_none());
    assert!(cluster_parts(|sketch| disjoint_products(sketch, 4), 6).is_some());
}
