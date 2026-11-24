// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::builder::ModuleBuilder;
use crate::dtype::DType;
use crate::ir::Module;
use crate::ops::{Concatenate, Maximum, Multiply, Slice, Tanh, Transpose};
use crate::shape::{Dims, Shape};
use smallvec::smallvec;

use super::*;

const MAX_TERMS: usize = 1 << 16;

fn scalar() -> Shape {
    Shape::scalar(DType::F32)
}

/// Runs the recovery the way the pipeline does. The pass replays every original
/// node and lets dead-code elimination drop whatever the recovered region
/// replaced, so counting operations is only meaningful after that runs.
fn recover(module: &mut Module) -> bool {
    let changed = recover_map_reduce_regions(module, MAX_TERMS).unwrap();
    if changed {
        crate::optimize::passes::dce(module).unwrap();
    }
    changed
}

/// A parameter vector read out as individual scalars, which is what
/// scalarization leaves for every operand a lane program touches.
fn sliced_elements(builder: &mut ModuleBuilder, count: i64) -> Vec<ValueId> {
    let source = builder.param(Shape::new(DType::F32, [count]));
    (0..count)
        .map(|index| {
            let sliced = builder.op(
                Slice {
                    start_indices: vec![index].into(),
                    limit_indices: vec![index + 1].into(),
                    strides: vec![1].into(),
                },
                &[source],
            )[0];
            builder.op(
                Reshape {
                    out_dims: Dims::from_slice(&[]),
                },
                &[sliced],
            )[0]
        })
        .collect()
}

/// Sums `terms` into one scalar with a right-leaning tree, which is the shape
/// scalarization and the e-graph leave behind.
fn sum(builder: &mut ModuleBuilder, terms: &[ValueId]) -> ValueId {
    let mut total = terms[0];
    for &term in &terms[1..] {
        total = builder.op(Add, &[total, term])[0];
    }
    total
}

fn scatters(func: &Func) -> Vec<&Scatter> {
    func.nodes
        .iter()
        .filter_map(|node| node.op.as_any().downcast_ref::<Scatter>())
        .collect()
}

fn count(func: &Func, mnemonic: &str) -> usize {
    func.nodes
        .iter()
        .filter(|node| node.op.mnemonic() == mnemonic)
        .count()
}

/// A sum of many products over one parameter vector: the canonical map/reduce.
/// The recovery must leave one Scatter that is *not* unique, because that is
/// what the backend can walk with a loop.
#[test]
fn a_sum_of_products_becomes_one_non_unique_scatter() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let source = builder.param(Shape::new(DType::F32, [16]));
    let element = |builder: &mut ModuleBuilder, index: i64| {
        builder.op(
            Slice {
                start_indices: vec![index].into(),
                limit_indices: vec![index + 1].into(),
                strides: vec![1].into(),
            },
            &[source],
        )[0]
    };
    let elements = (0..16)
        .map(|index| {
            let sliced = element(&mut builder, index);
            builder.op(
                Reshape {
                    out_dims: Dims::from_slice(&[]),
                },
                &[sliced],
            )[0]
        })
        .collect::<Vec<_>>();
    let terms = (0..8)
        .map(|i| builder.op(Multiply, &[elements[i], elements[i + 8]])[0])
        .collect::<Vec<_>>();
    let total = sum(&mut builder, &terms);
    let main = builder.end_func(vec![total]);
    let mut module = builder.finish(main).unwrap();

    assert!(recover(&mut module));
    let func = module.entry_func();
    let scatters = scatters(func);
    assert_eq!(scatters.len(), 1, "one batch of eight isomorphic products");
    assert!(
        !scatters[0].unique_indices,
        "a unique scatter is a permutation and resolves back to scalar work"
    );
    // Eight products collapse to one multiply over a lane axis.
    assert_eq!(count(func, "multiply"), 1);
}

/// The recognizer is structural, not polynomial. A sum of `tanh(a)*b` terms has
/// no monomial reading at all, and must batch exactly the same way.
#[test]
fn a_sum_of_transcendental_terms_is_recovered() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let source = builder.param(Shape::new(DType::F32, [16]));
    let elements = (0..16)
        .map(|index| {
            let sliced = builder.op(
                Slice {
                    start_indices: vec![index].into(),
                    limit_indices: vec![index + 1].into(),
                    strides: vec![1].into(),
                },
                &[source],
            )[0];
            builder.op(
                Reshape {
                    out_dims: Dims::from_slice(&[]),
                },
                &[sliced],
            )[0]
        })
        .collect::<Vec<_>>();
    let terms = (0..8)
        .map(|i| {
            let activated = builder.op(Tanh, &[elements[i]])[0];
            builder.op(Multiply, &[activated, elements[i + 8]])[0]
        })
        .collect::<Vec<_>>();
    let total = sum(&mut builder, &terms);
    let main = builder.end_func(vec![total]);
    let mut module = builder.finish(main).unwrap();

    assert!(
        recover(&mut module),
        "a transcendental map/reduce was not recovered"
    );
    let func = module.entry_func();
    assert_eq!(scatters(func).len(), 1);
    assert_eq!(count(func, "tanh"), 1, "eight tanh calls became one batch");
    assert_eq!(count(func, "multiply"), 1);
}

/// Terms of different structure must land in different batches, each with its
/// own scatter, rather than being forced into one.
#[test]
fn structurally_different_terms_take_separate_batches() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    // Every term reads its own elements, so nothing here is recovered by
    // replaying work the program already shares: the question is only whether
    // two shapes take two batches.
    let elements = sliced_elements(&mut builder, 40);
    let mut terms = (0..8)
        .map(|i| builder.op(Multiply, &[elements[i], elements[i + 8]])[0])
        .collect::<Vec<_>>();
    terms.extend((0..8).map(|i| {
        let inner = builder.op(Multiply, &[elements[i + 16], elements[i + 24]])[0];
        builder.op(Multiply, &[inner, elements[i + 32]])[0]
    }));
    let total = sum(&mut builder, &terms);
    let main = builder.end_func(vec![total]);
    let mut module = builder.finish(main).unwrap();

    assert!(recover(&mut module));
    let func = module.entry_func();
    let scatters = scatters(func);
    assert_eq!(scatters.len(), 2, "two shapes, two batches");
    assert!(scatters.iter().all(|scatter| !scatter.unique_indices));
}

/// A leaf that is not a parameter element or a literal cannot be laid out at a
/// known offset, so no gather can select it. The region is declined and the
/// program is left scalar — the pass must not fail the whole compilation.
#[test]
fn a_leaf_with_no_pool_declines_instead_of_failing() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let source = builder.param(Shape::new(DType::F32, [16]));
    // A transpose between the parameter and the slice: the elements are still
    // scalars, but none of them is at a statically known parameter offset.
    let moved = builder.op(
        Transpose {
            permutation: smallvec![0],
        },
        &[source],
    )[0];
    let elements = (0..16)
        .map(|index| {
            let sliced = builder.op(
                Slice {
                    start_indices: vec![index].into(),
                    limit_indices: vec![index + 1].into(),
                    strides: vec![1].into(),
                },
                &[moved],
            )[0];
            builder.op(
                Reshape {
                    out_dims: Dims::from_slice(&[]),
                },
                &[sliced],
            )[0]
        })
        .collect::<Vec<_>>();
    let terms = (0..8)
        .map(|i| builder.op(Multiply, &[elements[i], elements[i + 8]])[0])
        .collect::<Vec<_>>();
    let total = sum(&mut builder, &terms);
    let main = builder.end_func(vec![total]);
    let mut module = builder.finish(main).unwrap();

    assert!(
        !recover_map_reduce_regions(&mut module, MAX_TERMS).unwrap(),
        "an unpoolable leaf must decline the region, not fail the pass"
    );
}

/// Distribution is not free, and the extractor factored for a reason. Scaling a
/// sum by a factor costs one multiply; distributing that factor across the terms
/// costs one multiply *per term*. The recovered XIR still prints small — a few
/// gathers and a loop — so a size-based test calls this a win, and it is not: the
/// loop performs the extra multiplies once per iteration, in sequence.
///
/// So a region whose expansion costs more operations than the factored cone is
/// declined, and the factorization stands.
#[test]
fn an_expansion_that_costs_more_work_than_the_factored_form_is_declined() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let elements = sliced_elements(&mut builder, 34);
    let shared = builder.op(Add, &[elements[32], elements[33]])[0];
    let terms = (0..16)
        .map(|i| {
            let scaled = builder.op(Multiply, &[elements[i], shared])[0];
            builder.op(Multiply, &[elements[i + 16], scaled])[0]
        })
        .collect::<Vec<_>>();
    let total = sum(&mut builder, &terms);
    let main = builder.end_func(vec![total]);
    let mut module = builder.finish(main).unwrap();

    // Factored: one shared add, 32 multiplies, 15 adds. Distributed: 32 terms of
    // three factors, so 32 products and 32 reducer steps — twice the work.
    assert!(
        !recover(&mut module),
        "an expansion that doubles the executed work was accepted"
    );
}

/// A leaf that is not a parameter element or a literal cannot be laid out at a
/// known offset, so no gather can select it. The region is declined and the
/// program is left scalar — the pass must not fail the whole compilation.
#[test]
fn a_reducer_without_an_identity_is_left_scalar() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let source = builder.param(Shape::new(DType::F32, [16]));
    let elements = (0..16)
        .map(|index| {
            let sliced = builder.op(
                Slice {
                    start_indices: vec![index].into(),
                    limit_indices: vec![index + 1].into(),
                    strides: vec![1].into(),
                },
                &[source],
            )[0];
            builder.op(
                Reshape {
                    out_dims: Dims::from_slice(&[]),
                },
                &[sliced],
            )[0]
        })
        .collect::<Vec<_>>();
    let terms = (0..8)
        .map(|i| builder.op(Multiply, &[elements[i], elements[i + 8]])[0])
        .collect::<Vec<_>>();
    let mut total = terms[0];
    for &term in &terms[1..] {
        total = builder.op(Maximum, &[total, term])[0];
    }
    let main = builder.end_func(vec![total]);
    let mut module = builder.finish(main).unwrap();

    assert!(
        !recover(&mut module),
        "a reducer with no constructible identity must not be recovered"
    );
}

/// One term per output element is a permutation. Recovering it would only add
/// indexing, so the pass declines — this is the scalar fallback by actual work.
#[test]
fn a_permutation_is_not_recovered() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let left = builder.param(scalar());
    let right = builder.param(scalar());
    let total = builder.op(Add, &[left, right])[0];
    let main = builder.end_func(vec![total]);
    let mut module = builder.finish(main).unwrap();

    assert!(
        !recover(&mut module),
        "a single add is not a reduction worth indexing"
    );
}

/// Each maximal reducer root is its own region, so a packed output of two sums
/// yields two scatters — and every lane of each targets that root's one element.
#[test]
fn each_root_scatters_onto_its_own_element() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let elements = sliced_elements(&mut builder, 32);
    let outputs = (0..2)
        .map(|out| {
            let terms = (0..8)
                .map(|i| {
                    builder.op(
                        Multiply,
                        &[elements[out * 8 + i], elements[16 + out * 8 + i]],
                    )[0]
                })
                .collect::<Vec<_>>();
            let total = sum(&mut builder, &terms);
            builder.op(
                Reshape {
                    out_dims: Dims::from_slice(&[1]),
                },
                &[total],
            )[0]
        })
        .collect::<Vec<_>>();
    let packed = builder.op(Concatenate { dimension: 0 }, &outputs)[0];
    let main = builder.end_func(vec![packed]);
    let mut module = builder.finish(main).unwrap();

    assert!(recover(&mut module));
    let func = module.entry_func();
    let scatters = scatters(func);
    assert_eq!(scatters.len(), 2, "two maximal roots, two regions");
    assert!(scatters.iter().all(|scatter| !scatter.unique_indices));
    for node in func
        .nodes
        .iter()
        .filter(|node| node.op.as_any().is::<Scatter>())
    {
        let destinations = crate::analysis::constant_literal(func, node.inputs[1])
            .expect("scatter destinations are a constant index vector");
        let values = destinations
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| i32::from_le_bytes(*bytes))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![0; 8], "a scalar root has one destination");
    }
}

/// The reduction that matters is often not at an output. After the e-graph
/// refactors a program the largest sum sits under a scale or a convert, and a
/// pass that only inspected outputs would walk straight past it.
#[test]
fn a_reduction_buried_under_a_multiply_is_recovered() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let elements = sliced_elements(&mut builder, 17);
    let terms = (0..8)
        .map(|i| builder.op(Multiply, &[elements[i], elements[i + 8]])[0])
        .collect::<Vec<_>>();
    let total = sum(&mut builder, &terms);
    // The sum is not the output: a scale sits above it.
    let scaled = builder.op(Multiply, &[total, elements[16]])[0];
    let main = builder.end_func(vec![scaled]);
    let mut module = builder.finish(main).unwrap();

    assert!(
        recover(&mut module),
        "a reduction below the output was not recovered"
    );
    let func = module.entry_func();
    assert_eq!(scatters(func).len(), 1);
    assert!(!scatters(func)[0].unique_indices);
    // One batched product inside the region, and the surviving outer scale.
    assert_eq!(count(func, "multiply"), 2);

    // The consumer must actually read the recovered value: the output multiply
    // has to reach the scatter, or the alias never replaced anything.
    let result = func.results[0];
    let mut stack = vec![result];
    let mut reaches_scatter = false;
    let mut seen = vec![false; func.values.len()];
    while let Some(value) = stack.pop() {
        if std::mem::replace(&mut seen[value.index()], true) {
            continue;
        }
        let Some(node) = producer(func, value) else {
            continue;
        };
        if node.op.as_any().is::<Scatter>() {
            reaches_scatter = true;
            break;
        }
        stack.extend(node.inputs.iter().copied());
    }
    assert!(
        reaches_scatter,
        "the outer scale does not read the recovered reduction"
    );
}

/// A reducer node whose single consumer applies the same reducer belongs to that
/// consumer's tree; recovering it separately would cut one region into two.
#[test]
fn only_maximal_reducer_roots_become_regions() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let elements = sliced_elements(&mut builder, 16);
    let terms = (0..8)
        .map(|i| builder.op(Multiply, &[elements[i], elements[i + 8]])[0])
        .collect::<Vec<_>>();
    let total = sum(&mut builder, &terms);
    let main = builder.end_func(vec![total]);
    let mut module = builder.finish(main).unwrap();

    assert!(recover(&mut module));
    assert_eq!(
        scatters(module.entry_func()).len(),
        1,
        "the seven inner adds are part of one region, not seven"
    );
}

/// Work the cone shares with a consumer outside it is not work recovery
/// removes: that value still has to be computed for the other reader. Counting
/// the whole cone would credit the region with deleting it and let an expansion
/// that is really a loss look like a win.
///
/// Here every product feeding the sum is also read by a second output, so the
/// sum's cone deletes only its own adds — far less than the expansion costs.
#[test]
fn work_shared_with_an_outside_consumer_does_not_count_as_replaced() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let elements = sliced_elements(&mut builder, 34);
    let shared = builder.op(Add, &[elements[32], elements[33]])[0];
    let terms = (0..16)
        .map(|i| {
            let scaled = builder.op(Multiply, &[elements[i], shared])[0];
            builder.op(Multiply, &[elements[i + 16], scaled])[0]
        })
        .collect::<Vec<_>>();
    let total = sum(&mut builder, &terms);
    // A second output reads every term, so none of them dies with the sum.
    let escape = sum(&mut builder, &terms);
    let left = builder.op(
        Reshape {
            out_dims: Dims::from_slice(&[1]),
        },
        &[total],
    )[0];
    let right = builder.op(
        Reshape {
            out_dims: Dims::from_slice(&[1]),
        },
        &[escape],
    )[0];
    let packed = builder.op(Concatenate { dimension: 0 }, &[left, right])[0];
    let main = builder.end_func(vec![packed]);
    let mut module = builder.finish(main).unwrap();

    assert!(
        !recover(&mut module),
        "a region was credited with deleting work its outside consumer still needs"
    );
}

/// Retention is transitive, and a direct-use test misses that.
///
/// Here the escape sits on an intermediate: only `scaled` is read from outside,
/// so only `scaled` has a use the cone does not account for. But `scaled` must
/// still be computed, which means the product feeding it must be too — even
/// though *that* product has exactly one use and it is inside the cone. Counting
/// it as deleted overstates what recovery removes.
#[test]
fn retention_reaches_the_inputs_of_an_escaping_intermediate() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let elements = sliced_elements(&mut builder, 48);
    let mut terms = Vec::new();
    let mut escapes = Vec::new();
    for i in 0..16 {
        // `inner` is read once, by `scaled`; `scaled` is read by the sum and by
        // the second output.
        let inner = builder.op(Multiply, &[elements[i], elements[i + 16]])[0];
        let scaled = builder.op(Multiply, &[inner, elements[i + 32]])[0];
        terms.push(scaled);
        escapes.push(scaled);
    }
    let total = sum(&mut builder, &terms);
    let escape = sum(&mut builder, &escapes);
    let left = builder.op(
        Reshape {
            out_dims: Dims::from_slice(&[1]),
        },
        &[total],
    )[0];
    let right = builder.op(
        Reshape {
            out_dims: Dims::from_slice(&[1]),
        },
        &[escape],
    )[0];
    let packed = builder.op(Concatenate { dimension: 0 }, &[left, right])[0];
    let main = builder.end_func(vec![packed]);
    let module = builder.finish(main).unwrap();

    let func = module.entry_func();
    let scalar = scalar_region(func);
    let uses = use_counts(func);
    // The cone holds 15 adds, 16 `scaled`, and 16 `inner`. Every `scaled` is read
    // from outside, and each retains the `inner` it reads, so only the adds are
    // really deleted. A direct-use test would retain the 16 `scaled` alone and
    // report 31.
    assert_eq!(
        replaced_work(func, total, &scalar, &uses),
        15,
        "retention did not reach the inputs of the escaping intermediates"
    );
}
