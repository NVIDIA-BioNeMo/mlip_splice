// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use algebraic_real::AlgebraicReal;
use domain_index::FiniteMapGraph;
use xir::builder::ModuleBuilder;
use xir::dtype::DType;
use xir::ir::{Def, Func, ValueId};
use xir::ops::{ComparisonDirection, Constant, Literal, Select};
use xir::shape::Shape;
use xir_optimize::{
    DEFAULT_MAX_ALGEBRAIC_EXPRESSION_WORK, DEFAULT_MAX_ALGEBRAIC_WORK,
    extract_algebraic_frontier_with_limits,
};

fn product_leaves(func: &Func, value: ValueId, leaves: &mut Vec<ValueId>) {
    if let Def::NodeResult { node, slot: 0 } = func.values[value.index()].def
        && func.nodes[node].op.mnemonic() == "multiply"
    {
        for &input in &func.nodes[node].inputs {
            product_leaves(func, input, leaves);
        }
    } else {
        leaves.push(value);
    }
}

#[test]
fn indexed_extraction_preserves_the_materialized_guard_factor() {
    let mut graph = FiniteMapGraph::default();
    let fine = graph.add_set(8).unwrap();
    let coarse = graph.add_set(3).unwrap();
    let map = graph.add_map(fine, coarse).unwrap();
    let mut b = ModuleBuilder::with_finite_map_graph(graph);
    b.begin_func("guarded_sum");
    let shape = Shape::scalar(DType::F32).with_domain_axes([fine]);
    let r = b.param(shape.clone());
    let inv = b.param(shape.clone());
    let w = b.param(shape.clone());
    let one = b.constant(Literal::exact(shape.clone(), vec![AlgebraicReal::one()]).unwrap());
    let eight = b.constant(Literal::exact(shape.clone(), vec![AlgebraicReal::integer(8)]).unwrap());
    let zero = b.constant(Literal::zeros(shape).unwrap());
    let scaled = b.multiply(eight, r);
    let scaled = b.multiply(scaled, inv);
    let envelope = b.binary("subtract", one, scaled).unwrap();
    let predicate = b.compare(ComparisonDirection::Ge, None, envelope, zero);
    let gated = b.op1(Select, &[predicate, w, zero]);
    let square = b.multiply(envelope, envelope);
    let weighted = b.multiply(gated, square);
    let sum = b.sum_along_map(weighted, map, 0);
    let entry = b.end_func(vec![sum]);
    let module = b.finish(entry).unwrap();

    for alternative in extract_algebraic_frontier_with_limits(
        &module,
        DEFAULT_MAX_ALGEBRAIC_WORK,
        DEFAULT_MAX_ALGEBRAIC_EXPRESSION_WORK,
    )
    .unwrap()
    .0
    {
        let func = alternative.module.entry_func();
        let expanded = [AlgebraicReal::integer(64), -&AlgebraicReal::integer(16)];
        for node in &func.nodes {
            if let Some(constant) = node.op.as_any().downcast_ref::<Constant>() {
                assert!(
                    !constant
                        .literal
                        .algebraic_values()
                        .unwrap()
                        .iter()
                        .any(|value| expanded.contains(value))
                );
            }
        }
        let compare = func
            .nodes
            .iter()
            .find(|node| node.op.mnemonic() == "compare")
            .unwrap();
        let envelope = compare.inputs[0];
        let select = func
            .nodes
            .iter()
            .find(|node| node.op.mnemonic() == "select")
            .unwrap();
        let sum = func
            .nodes
            .iter()
            .find(|node| node.op.mnemonic() == "sum_along_map")
            .unwrap();
        let mut leaves = Vec::new();
        product_leaves(func, sum.inputs[0], &mut leaves);
        leaves.sort_unstable_by_key(|value| value.index());
        let mut expected = vec![envelope, envelope, select.results[0]];
        expected.sort_unstable_by_key(|value| value.index());
        assert_eq!(
            leaves, expected,
            "the sum lost its predicate's exact envelope factor"
        );
    }
}
