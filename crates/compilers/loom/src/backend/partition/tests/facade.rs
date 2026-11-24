// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use gpu_ir::{Param, ParamTy, Scalar, Value};

use super::super::*;
use super::support::*;

/// One `f32` array every sketch here reads its inputs through.
fn input_array(sketch: &mut Sketch) -> Value {
    sketch.builder().parameter(Param {
        name: "input".into(),
        ty: ParamTy::Ptr(Scalar::F32),
    })
}

fn gathered(sketch: &mut Sketch, array: &Value, element: i64) -> Value {
    let b = sketch.builder();
    let at = b.i64(element);
    let cell = b.offset(array.clone(), at);
    b.load(cell, 1, true)
}

#[test]
fn closures_intern_profile_inputs_after_definitions() {
    let mut sketch = Sketch::new();
    let x = sketch.input("x");
    let y = sketch.input("y");
    let sum = sketch.add(x.clone(), y);
    sketch.bind("a0", sum);
    let zero = sketch.get("a0");
    let product = sketch.mul(zero, x);
    sketch.bind("a1", product);
    let sketched = sketch.finish();
    let (root, ex, why) = (sketched.id("a1"), sketched.id("x"), sketched.id("y"));
    let profile = incidence(sketched, vec![vec![root]]);
    assert_eq!(profile.bindings.inputs(), 2);
    assert_eq!(profile.bindings.entry(ex), Some(2));
    assert_eq!(profile.bindings.entry(why), Some(3));
    let mut closure = BTreeSet::new();
    let mut inserted = Vec::new();
    profile.bindings.add_dependencies_with_inputs(
        profile.function(),
        root,
        &mut closure,
        &mut inserted,
    );
    assert_eq!(closure, BTreeSet::from([0, 1, 2, 3]));
}

/// Elements whose arithmetic is fully disjoint but whose operand loads
/// are shared attract each other: the interned inputs carry the seam at
/// the configured seam weight.
#[test]
fn shard_groups_elements_by_shared_input_loads() {
    let mut sketch = Sketch::new();
    let x = sketch.input("x");
    let y = sketch.input("y");
    for (name, source, factor) in [
        ("a0", &x, 2.0),
        ("a1", &x, 3.0),
        ("b0", &y, 2.0),
        ("b1", &y, 3.0),
    ] {
        let scale = sketch.f32(factor);
        let product = sketch.mul(source.clone(), scale);
        sketch.bind(name, product);
    }
    let sketched = sketch.finish();
    let outputs = vec![
        ["a0", "a1", "b0", "b1"]
            .map(|name| sketched.id(name))
            .to_vec(),
    ];
    let profile = incidence(sketched, outputs);
    let part = KernelPart {
        elements: vec![vec![
            selection(0, vec![0]),
            selection(1, vec![0]),
            selection(2, vec![0]),
            selection(3, vec![0]),
        ]],
    };
    let shards = part.shard(2, &profile, &Settings::default()).unwrap();
    assert_eq!(shards.len(), 2);
    let elements = |shard: &KernelPart| {
        let mut elements = shard.elements[0]
            .iter()
            .map(|selection| selection.element)
            .collect::<Vec<_>>();
        elements.sort_unstable();
        elements
    };
    assert_eq!(elements(&shards[0]), [0, 1]);
    assert_eq!(elements(&shards[1]), [2, 3]);
}

/// Two closure families feed two elements each; the split follows the
/// families instead of scattering siblings by term count.
#[test]
fn shard_groups_elements_by_closure_overlap() {
    let mut sketch = Sketch::new();
    let array = input_array(&mut sketch);
    for (base, tip, element, factor) in [("a0", "a1", 0, 2.0), ("b0", "b1", 1, 3.0)] {
        let loaded = gathered(&mut sketch, &array, element);
        sketch.bind(base, loaded);
        let held = sketch.get(base);
        let scale = sketch.f32(factor);
        let product = sketch.mul(held, scale);
        sketch.bind(tip, product);
    }
    let sketched = sketch.finish();
    let (a, b) = (sketched.id("a1"), sketched.id("b1"));
    let outputs = vec![vec![a, a, b, b]];
    let profile = incidence(sketched, outputs);
    let part = KernelPart {
        elements: vec![vec![
            selection(0, vec![0]),
            selection(1, vec![0]),
            selection(2, vec![0]),
            selection(3, vec![0]),
        ]],
    };
    let shards = part.shard(2, &profile, &Settings::default()).unwrap();
    assert_eq!(shards.len(), 2);
    let elements = |shard: &KernelPart| {
        let mut elements = shard.elements[0]
            .iter()
            .map(|selection| selection.element)
            .collect::<Vec<_>>();
        elements.sort_unstable();
        elements
    };
    assert_eq!(elements(&shards[0]), [0, 1]);
    assert_eq!(elements(&shards[1]), [2, 3]);
    assert_eq!(
        shards.iter().map(KernelPart::len).sum::<usize>(),
        part.len()
    );
    assert!(part.shard(5, &profile, &Settings::default()).is_none());
}

/// Every element reads the same long chain: no grouping narrows the
/// closure, so the split declines instead of tripling the work.
#[test]
fn shard_declines_when_elements_share_one_closure() {
    let mut sketch = Sketch::new();
    let array = input_array(&mut sketch);
    let loaded = gathered(&mut sketch, &array, 0);
    sketch.bind("chain_0", loaded);
    for step in 1..20 {
        let held = sketch.get(&format!("chain_{}", step - 1));
        let scale = sketch.f32(1.5);
        let product = sketch.mul(held, scale);
        sketch.bind(&format!("chain_{step}"), product);
    }
    let sketched = sketch.finish();
    let root = sketched.id("chain_19");
    let outputs = vec![vec![root, root, root, root]];
    let profile = incidence(sketched, outputs);
    let part = KernelPart {
        elements: vec![vec![
            selection(0, vec![0]),
            selection(1, vec![0]),
            selection(2, vec![0]),
            selection(3, vec![0]),
        ]],
    };
    assert!(part.shard(2, &profile, &Settings::default()).is_none());
}

#[test]
fn selection_key_ranks_footprint_then_duplication() {
    let cost = |max_weighted_footprint, duplicated_closure_statements, duplicated_input_loads| {
        PartitionCost {
            max_closure_statements: 0,
            max_weighted_footprint,
            duplicated_closure_statements,
            duplicated_input_loads,
            distinct_input_loads: 0,
        }
    };
    // A narrower max launch wins regardless of duplication.
    let settings = Settings::default();
    assert!(cost(30, 50, 50).selection_key(&settings) < cost(40, 0, 0).selection_key(&settings));
    // At equal footprint one duplicated load outweighs three duplicated
    // statements.
    assert!(cost(30, 3, 0).selection_key(&settings) < cost(30, 0, 1).selection_key(&settings));
    let settings = Settings {
        seam_load_weight: 2,
        ..settings
    };
    assert!(cost(30, 3, 0).selection_key(&settings) > cost(30, 0, 1).selection_key(&settings));
}
