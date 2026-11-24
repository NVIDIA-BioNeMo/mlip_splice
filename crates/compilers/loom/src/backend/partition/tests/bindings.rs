// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the closure index space is, and the order a closure comes out in.

use super::super::*;
use super::support::*;

/// `v10` sorts before `v9` and is defined after it, so name order and index
/// order disagree — and a closure follows the names. Every entry a definition
/// reaches is recorded once, at the point it is first reached.
#[test]
fn a_closure_is_walked_in_name_order() {
    let mut sketch = Sketch::new();
    let x = sketch.input("x");
    let y = sketch.input("y");
    sketch.bind("v9", x);
    sketch.bind("v10", y);
    let (nine, ten) = (sketch.get("v9"), sketch.get("v10"));
    let sum = sketch.add(nine, ten);
    sketch.bind("top", sum);
    let sketched = sketch.finish();
    let bindings = sketched.bindings();

    assert_eq!(bindings.entry(sketched.id("v9")), Some(0));
    assert_eq!(bindings.entry(sketched.id("v10")), Some(1));
    assert_eq!(bindings.entry(sketched.id("x")), Some(3));
    assert_eq!(bindings.entry(sketched.id("y")), Some(4));

    let mut required = BTreeSet::new();
    let mut inserted = Vec::new();
    bindings.add_dependencies_with_inputs(
        sketched.function(),
        sketched.id("top"),
        &mut required,
        &mut inserted,
    );
    assert_eq!(inserted, vec![2, 1, 4, 0, 3]);
}

/// Without inputs a closure stops at the profile's loads, and asking twice adds
/// nothing the first ask already added.
#[test]
fn a_closure_holds_input_loads_only_when_asked_for_them() {
    let mut sketch = Sketch::new();
    let x = sketch.input("x");
    let y = sketch.input("y");
    let product = sketch.mul(x.clone(), y);
    sketch.bind("v0", product);
    let zero = sketch.get("v0");
    let sum = sketch.add(zero, x);
    sketch.bind("v1", sum);
    let sketched = sketch.finish();
    let bindings = sketched.bindings();
    let (f, root) = (sketched.function(), sketched.id("v1"));

    let mut required = BTreeSet::new();
    let mut inserted = Vec::new();
    bindings.add_dependencies(f, root, &mut required, &mut inserted);
    assert_eq!(inserted, vec![1, 0]);

    let mut required = BTreeSet::new();
    let mut inserted = Vec::new();
    bindings.add_dependencies_with_inputs(f, root, &mut required, &mut inserted);
    assert_eq!(inserted, vec![1, 0, 2, 3]);
    bindings.add_dependencies_with_inputs(f, root, &mut required, &mut inserted);
    assert_eq!(inserted, vec![1, 0, 2, 3]);
}

/// An output naming a load no binding reaches interns after the ones they do, so
/// the closure index space covers every load the profile performs.
#[test]
fn outputs_intern_the_loads_no_binding_reaches() {
    let mut sketch = Sketch::new();
    let x = sketch.input("x");
    let bias = sketch.input("bias");
    sketch.bind("v0", x);
    let sketched = sketch.finish();
    let outputs = vec![vec![sketched.id("v0"), sketched.id("bias")]];
    let _ = bias;
    let mut bindings = sketched.bindings();
    assert_eq!(bindings.inputs(), 1);
    bindings.intern_outputs(sketched.function(), &outputs);
    assert_eq!(bindings.entry(sketched.id("x")), Some(1));
    assert_eq!(bindings.entry(sketched.id("bias")), Some(2));
    assert_eq!(bindings.inputs(), 2);
    assert_eq!(
        bindings.input_entries().collect::<Vec<_>>(),
        vec![(sketched.id("x"), 1), (sketched.id("bias"), 2)]
    );
}

/// One entry per binding is what makes the index space a partition. Definitions
/// are identified by value rather than diagnostic name, so two definitions may
/// share a name without aliasing one entry.
#[test]
fn a_second_binding_of_a_name_is_its_own_entry() {
    let mut sketch = Sketch::new();
    let x = sketch.input("x");
    let y = sketch.input("y");
    let first = sketch.bind("v0", x);
    let second = sketch.bind("v0", y);
    sketch.bind("v1", second.clone());
    let sketched = sketch.finish();
    let bindings = sketched.bindings();
    let (first, second) = (
        first.id_in(sketched.function()),
        second.id_in(sketched.function()),
    );
    assert_ne!(first, second);
    assert_eq!(bindings.entry(first), Some(0));
    assert_eq!(bindings.entry(second), Some(1));
    assert_eq!(bindings.definitions(), 3);
}

/// A body that is not a flat list of bindings is refused: the index space gives
/// each binding the position of its step, and a region-owning step holds more
/// than one.
#[test]
fn a_body_that_is_not_flat_bindings_is_refused() {
    let mut sketch = Sketch::new();
    let x = sketch.input("x");
    let zero = sketch.f32(0.0);
    let below = {
        let b = sketch.builder();
        b.lt(x, zero)
    };
    sketch.builder().if_(below, |arm| arm.ret());
    let sketched = sketch.finish();
    let Err(refused) = Bindings::of(sketched.function(), &sketched.ops()) else {
        panic!("a body with a region must not resolve");
    };
    assert!(refused.contains("flat bindings"), "{refused}");
}

/// The same closure re-derived independently: over the sketch's names, through
/// a map of binding names, recursing on each dependency's name rather than on
/// the interned entry the analysis walks.
fn closure_by_name(sketched: &Sketched, root: &str, with_inputs: bool) -> Vec<usize> {
    let f = sketched.function();
    let ops = sketched.ops();
    let named = |value: gpu_ir::ValueId| f.name(value).expect("every value is named").to_string();
    let dependencies = |index: usize| {
        let mut names = f
            .reads(f.op(ops[index]).operands[0])
            .into_iter()
            .map(named)
            .collect::<Vec<_>>();
        names.sort();
        names
    };
    let by_name = (0..ops.len())
        .map(|index| (named(f.op(ops[index]).results[0]), index))
        .collect::<HashMap<_, _>>();
    let mut inputs = HashMap::new();
    for index in 0..ops.len() {
        for name in dependencies(index) {
            if !by_name.contains_key(&name) && !inputs.contains_key(&name) {
                inputs.insert(name, ops.len() + inputs.len());
            }
        }
    }

    fn add(
        name: &str,
        dependencies: &impl Fn(usize) -> Vec<String>,
        by_name: &HashMap<String, usize>,
        inputs: &HashMap<String, usize>,
        required: &mut BTreeSet<usize>,
        inserted: &mut Vec<usize>,
        with_inputs: bool,
    ) {
        let Some(&index) = by_name.get(name) else {
            if with_inputs
                && let Some(&input) = inputs.get(name)
                && required.insert(input)
            {
                inserted.push(input);
            }
            return;
        };
        if !required.insert(index) {
            return;
        }
        inserted.push(index);
        for dependency in dependencies(index) {
            add(
                &dependency,
                dependencies,
                by_name,
                inputs,
                required,
                inserted,
                with_inputs,
            );
        }
    }

    let mut required = BTreeSet::new();
    let mut inserted = Vec::new();
    add(
        root,
        &dependencies,
        &by_name,
        &inputs,
        &mut required,
        &mut inserted,
        with_inputs,
    );
    inserted
}

/// The dense walk and the name walk agree entry for entry, in order, from every
/// binding of a profile whose names, definition order and sharing all disagree.
#[test]
fn the_dense_closure_matches_the_name_closure() {
    let mut sketch = Sketch::new();
    let x = sketch.input("x");
    let w = sketch.input("w");
    let y = sketch.input("y");
    let z = sketch.input("z");
    let product = sketch.mul(x.clone(), w);
    sketch.bind("v9", product);
    let nine = sketch.get("v9");
    let sum = sketch.add(nine.clone(), y);
    sketch.bind("v10", sum);
    let difference = sketch.sub(x, nine);
    sketch.bind("v2", difference);
    let (ten, two) = (sketch.get("v10"), sketch.get("v2"));
    let product = sketch.mul(ten.clone(), two);
    sketch.bind("b", product);
    let b = sketch.get("b");
    let sum = sketch.add(b, ten);
    sketch.bind("a", sum);
    let a = sketch.get("a");
    let difference = sketch.sub(a, z);
    sketch.bind("v100", difference);
    let sketched = sketch.finish();
    let bindings = sketched.bindings();
    let f = sketched.function();

    // The whole profile from its last binding, so neither walk can agree by both
    // coming back empty: bindings in the order the names lead to them, each
    // input load where the walk first reaches it.
    let mut required = BTreeSet::new();
    let mut inserted = Vec::new();
    bindings.add_dependencies_with_inputs(f, sketched.id("v100"), &mut required, &mut inserted);
    assert_eq!(inserted, vec![5, 4, 3, 1, 0, 6, 7, 8, 2, 9]);

    for name in ["v9", "v10", "v2", "b", "a", "v100"] {
        for with_inputs in [false, true] {
            let mut required = BTreeSet::new();
            let mut dense = Vec::new();
            if with_inputs {
                bindings.add_dependencies_with_inputs(
                    f,
                    sketched.id(name),
                    &mut required,
                    &mut dense,
                );
            } else {
                bindings.add_dependencies(f, sketched.id(name), &mut required, &mut dense);
            }
            assert_eq!(
                dense,
                closure_by_name(&sketched, name, with_inputs),
                "{name} with_inputs={with_inputs}"
            );
        }
    }
}
