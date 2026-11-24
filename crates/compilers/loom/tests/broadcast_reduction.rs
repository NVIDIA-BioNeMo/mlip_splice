// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An output that carries no iteration axis sums every position into the same
//! cells, so its fold offers extra buckets and a projection that sums them
//! away. Every bucket count must reach the same sum as the direct one.

use cuda_driver::DeviceBuffer;
use kiln::{Bindings, Realizer, RealizerOptions, ResidentRecipe, execute};
use xir::builder::ModuleBuilder;
use xir::dtype::DType;
use xir::shape::Shape;

mod common;

use common::{default_stream, gpu_available};
use domain_index::Domain;
use std::num::NonZeroUsize;

use loom::{CompileOptions, CompiledProgram, Lanes, Operand, Program, compile};

const LANE: i64 = 8;

/// `out[u] = sum over positions of x[position, u] * x[position, u]`.
fn square_sum_recipe(positions: i64, elements: i64) -> (recipe::Recipe, usize, usize) {
    assert_eq!(elements % LANE, 0);
    let element_shape = vec![elements / LANE];
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::new(DType::F32, element_shape.clone()));
    let squared = builder.multiply(x, x);
    let main = builder.end_func(vec![squared]);
    let module = builder.finish(main).unwrap();
    let operand = |axis_domain| Operand {
        dtype: DType::F32,
        element_shape: element_shape.clone(),
        lanes: Lanes::PerLane,
        axis_domains: vec![axis_domain],
    };
    let (
        CompiledProgram {
            recipe,
            input_count,
            output_count,
            index_arrays,
        },
        _,
    ) = compile(
        Program {
            name: "broadcast_reduction".into(),
            module,
            lane_count: NonZeroUsize::new(LANE as usize).unwrap(),
            graph: domain_index::Graph {
                domains: vec![Domain { extent: positions }],
                index_maps: Vec::new(),
                aligned_views: Vec::new(),
            },
            inputs: vec![operand(Some(0))],
            outputs: vec![operand(None)],
        },
        CompileOptions::default(),
    )
    .expect("compile broadcast reduction");
    assert!(
        index_arrays.is_empty(),
        "no index map carries an index array"
    );
    (recipe, input_count, output_count)
}

/// Whether any arm of `recipe` clears the accumulator in a workspace rather
/// than in the output itself — the signature of a bucketed fold.
fn accumulates_out_of_place(body: &recipe::RecipeBody) -> bool {
    body.nodes().iter().any(|node| match node.work() {
        recipe::Work::Memset {
            destination:
                recipe::Address {
                    buf: recipe::WritableBuf::Workspace(_),
                    ..
                },
            ..
        } => true,
        recipe::Work::Choice { candidates, .. } => candidates.iter().any(accumulates_out_of_place),
        _ => false,
    })
}

fn input_values(positions: i64, elements: i64) -> Vec<f32> {
    (0..positions * elements)
        .map(|i| ((i % 17) as f32 - 8.0) / 16.0)
        .collect()
}

/// Every arm of the recipe, run on its own — one output vector per covering
/// selection, so a bucket count that sums wrongly cannot hide behind a faster
/// sibling.
fn run_every_arm(positions: i64, elements: i64) -> Vec<Vec<f32>> {
    let (recipe, num_inputs, num_outputs) = square_sum_recipe(positions, elements);
    let ctx = cuda_driver::Context::current().raw();
    let input = DeviceBuffer::<f32>::from_host(ctx, &input_values(positions, elements)).unwrap();
    let output = DeviceBuffer::<f32>::alloc(ctx, elements as usize).unwrap();
    let workspace =
        DeviceBuffer::<u8>::alloc(ctx, recipe::algebra::workspace_upper_bound(&recipe)).unwrap();
    let inputs = [input.ptr()];
    let outputs = [output.ptr()];
    assert_eq!((inputs.len(), outputs.len()), (num_inputs, num_outputs));
    let bindings = Bindings {
        inputs: &inputs,
        outputs: &outputs,
        workspace: workspace.ptr(),
    };
    let mut sums = Vec::new();
    let realizer = Realizer::new(ctx, RealizerOptions::default()).expect("realizer");
    let unrealized = realizer
        .for_each_covering_recipe(&recipe, &(), |flat| {
            let mut loaded = unsafe { ResidentRecipe::new(ctx, flat) }.expect("load");
            unsafe { execute(&mut loaded, &bindings, default_stream()) }.expect("execute");
            sums.push(output.to_host().unwrap());
        })
        .expect("the recipe under test is valid");
    // Loom states assembler alternatives, and whether a machine has `ptxas`
    // decides which of them exist here — so an arm this machine cannot build is
    // not a wrong sum, it is an absent toolchain, and it must not fail a test
    // about arithmetic. An arm the *program* makes infeasible is another
    // matter: a compile diagnostic or a blown resource bound would be a bug in
    // what loom emitted, and no machine would realize it either.
    for skipped in &unrealized {
        assert!(
            !skipped.error.is_program_property(),
            "arm {} is infeasible because of the program, not this machine: {:?}",
            skipped.arm,
            skipped.error
        );
    }
    if !unrealized.is_empty() {
        eprintln!(
            "run_every_arm: {} of {} arms skipped — this machine cannot build them",
            unrealized.len(),
            unrealized.len() + sums.len()
        );
    }
    assert!(!sums.is_empty(), "the recipe covers at least one arm");
    sums
}

fn expected(positions: i64, elements: i64) -> Vec<f32> {
    let mut sums = vec![0.0f64; elements as usize];
    for position in 0..positions {
        for element in 0..elements {
            let value = (((position * elements + element) % 17) as f64 - 8.0) / 16.0;
            sums[element as usize] += value * value;
        }
    }
    sums.into_iter().map(|value| value as f32).collect()
}

#[test]
#[serial_test::serial(gpu)]
fn every_bucket_count_sums_every_position() {
    if !gpu_available() {
        eprintln!("every_bucket_count_sums_every_position: no GPU, skipping");
        return;
    }
    // Short enough a run that no copy shortens it usefully, then long enough
    // that the accuracy floor asks for copies: both must reach the same sum, on
    // every arm the recipe offers.
    for (positions, elements, bucketed) in [
        (64, 8, false),
        (512, 8, true),
        (1_024, 8, true),
        (128, 640, true),
        (128, 2_048, true),
    ] {
        assert_eq!(
            accumulates_out_of_place(square_sum_recipe(positions, elements).0.body()),
            bucketed,
            "positions={positions} elements={elements}"
        );
        let reference = expected(positions, elements);
        for (arm, got) in run_every_arm(positions, elements).into_iter().enumerate() {
            for (element, (&value, reference)) in got.iter().zip(&reference).enumerate() {
                let tolerance = 1e-4 * reference.abs().max(1.0);
                assert!(
                    (value - reference).abs() <= tolerance,
                    "positions={positions} elements={elements} arm={arm} element={element}: {value} vs {reference}"
                );
            }
        }
    }
}
