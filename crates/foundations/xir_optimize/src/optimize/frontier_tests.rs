// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use domain_index::FiniteMapGraph;
use xir::{builder::ModuleBuilder, dtype::DType, shape::Shape};

fn product_of_sums() -> Module {
    let mut graph = FiniteMapGraph::default();
    let fine = graph.add_set(8).unwrap();
    let coarse = graph.add_set(3).unwrap();
    let map = graph.add_map(fine, coarse).unwrap();
    let mut builder = ModuleBuilder::with_finite_map_graph(graph);
    builder.begin_func("product_of_sums");
    let x = builder.param(Shape::scalar(DType::F32).with_domain_axes([fine]));
    let y = builder.param(Shape::scalar(DType::F32).with_domain_axes([fine]));
    let left = builder.sum_along_map(x, map, 0);
    let right = builder.sum_along_map(y, map, 0);
    let product = builder.multiply(left, right);
    let main = builder.end_func(vec![product]);
    builder.finish(main).unwrap()
}

fn describe() -> (AlgebraicFrontier, PipelineReport) {
    describe_algebraic_frontier_with_limits(&product_of_sums(), 1 << 24, 1 << 20, &mut || Ok(()))
        .unwrap()
}

#[test]
fn cleanup_is_per_requested_module_and_does_not_spend_the_extraction_reserve() {
    let (mut frontier, report) = describe();
    assert!(frontier.len() > 1);
    assert_eq!(report.passes.len(), 5);
    assert!(!frontier.indexed_algebra_abandoned);
    let count = frontier.len();
    let consumed = frontier.work.consumed;
    let mut emitted = Vec::new();
    while let Some(result) = frontier.next(&mut || Ok(())) {
        let (alternative, report) = result.unwrap();
        assert_eq!(report.passes.len(), 2);
        assert_eq!(frontier.work.consumed, consumed);
        xir::verify::verify(&alternative.module).unwrap();
        emitted.push(alternative.module.to_string());
        assert_eq!(frontier.len(), count - emitted.len());
    }
    let (collected, _) =
        extract_algebraic_frontier_with_limits(&product_of_sums(), 1 << 24, 1 << 20).unwrap();
    assert_eq!(
        emitted,
        collected
            .into_iter()
            .map(|alternative| alternative.module.to_string())
            .collect::<Vec<_>>()
    );
    assert!(frontier.next(&mut || Ok(())).is_none());
}

#[test]
fn cancellation_before_reconstruction_preserves_the_next_description() {
    let (mut frontier, _) = describe();
    let count = frontier.len();
    let error = frontier
        .next(&mut || Err("stop".into()))
        .unwrap()
        .err()
        .unwrap();
    assert_eq!(error.pass, "source-checkpoint");
    assert_eq!(frontier.len(), count);
    let expected = describe().0.next(&mut || Ok(())).unwrap().unwrap().0.module;
    let actual = frontier.next(&mut || Ok(())).unwrap().unwrap().0.module;
    assert_eq!(actual.to_string(), expected.to_string());
}

#[test]
fn stopping_cleanup_does_not_reconstruct_later_modules() {
    let (mut frontier, _) = describe();
    let (mut control, _) = describe();
    let first = frontier.next(&mut || Ok(())).unwrap().unwrap().0.module;
    assert_eq!(
        first.to_string(),
        control
            .next(&mut || Ok(()))
            .unwrap()
            .unwrap()
            .0
            .module
            .to_string()
    );
    let count = frontier.len();
    assert!(count > 0);
    let mut checkpoints = 0;
    let error = frontier
        .next(&mut || {
            checkpoints += 1;
            if checkpoints == 4 {
                Err("stop at cleanup".into())
            } else {
                Ok(())
            }
        })
        .unwrap()
        .err()
        .unwrap();
    assert_eq!(error.pass, "source-checkpoint");
    assert_eq!(frontier.len(), count - 1);
    control.next(&mut || Ok(())).unwrap().unwrap();
    while let Some(result) = frontier.next(&mut || Ok(())) {
        assert_eq!(
            result.unwrap().0.module.to_string(),
            control
                .next(&mut || Ok(()))
                .unwrap()
                .unwrap()
                .0
                .module
                .to_string()
        );
    }
    assert!(control.is_empty());
}
