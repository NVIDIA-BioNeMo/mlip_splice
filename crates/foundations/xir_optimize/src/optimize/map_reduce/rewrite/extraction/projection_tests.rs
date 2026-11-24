// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use domain_index::FiniteMapGraph;

use super::*;
use crate::builder::ModuleBuilder;
use crate::dtype::DType;
use crate::shape::Shape;

fn projection_frontier(factored: bool) -> Vec<FiniteMapWork> {
    let mut graph = FiniteMapGraph::default();
    let fine = graph.add_set(8).unwrap();
    let coarse = graph.add_set(3).unwrap();
    let map = graph.add_map(fine, coarse).unwrap();
    let mut builder = ModuleBuilder::with_finite_map_graph(graph.clone());
    builder.begin_func("projection_frontier");
    let coefficient = builder.param(Shape::scalar(DType::F32).with_domain_axes([coarse]));
    let values = builder.param(Shape::scalar(DType::F32).with_domain_axes([fine]));
    let result = if factored {
        let sum = builder.sum_along_map(values, map, 0);
        builder.multiply(coefficient, sum)
    } else {
        let coefficient = builder.gather_along_map(coefficient, map, 0);
        let product = builder.multiply(coefficient, values);
        builder.sum_along_map(product, map, 0)
    };
    let entry = builder.end_func(vec![result]);
    let module = builder.finish(entry).unwrap();
    let func = module.entry_func();
    let original = super::super::expression(func, result, &mut 0).unwrap();

    let mut work = extract_roots(func, &graph, &[original])
        .unwrap()
        .into_iter()
        .map(|candidate| candidate.work)
        .collect::<Vec<_>>();
    work.sort_by_key(|work| (work.before_accumulation, work.after_accumulation));
    work
}

fn expected_projection_frontier() -> [FiniteMapWork; 2] {
    [
        FiniteMapWork {
            before_accumulation: 0,
            after_accumulation: 3,
        },
        FiniteMapWork {
            before_accumulation: 8,
            after_accumulation: 0,
        },
    ]
}

#[test]
fn egglog_projection_exposes_the_same_frontier() {
    assert_eq!(projection_frontier(false), expected_projection_frontier());
}

#[test]
fn factored_projection_recovers_work_before_accumulation() {
    assert_eq!(projection_frontier(true), expected_projection_frontier());
}
