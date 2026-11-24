// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::builder::ModuleBuilder;
use crate::dtype::DType;
use crate::ops::Literal;
use crate::optimize::rebuild::Rebuilder;
use crate::shape::Shape;

#[test]
fn a_thousand_shared_levels_import_and_emit_once_without_unfolding() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("shared_dag");
    let mut value = builder.param(Shape::scalar(DType::F32));
    for _ in 0..1000 {
        value = builder.multiply(value, value);
    }
    let entry = builder.end_func(vec![value, value]);
    let module = builder.finish(entry).unwrap();
    let func = module.entry_func();
    let source = expression_dag::import(func).unwrap().unwrap();
    assert_eq!(source.dag.len(), 1001);
    assert_eq!(
        source.dag.quality(source.roots.iter().map(|(_, id)| *id)),
        (1001, 0, 0)
    );

    let mut graph = RewriteGraph::new(func, &module.finite_map_graph).unwrap();
    graph.import(&source.dag).unwrap();
    let (snapshot, classes) = graph.snapshot_dag(&source.dag).unwrap();
    assert_eq!(classes.len(), 1001);
    assert_eq!(snapshot.classes.iter().map(Vec::len).sum::<usize>(), 1001);
    assert_eq!(
        snapshot
            .classes
            .iter()
            .flatten()
            .filter(|node| matches!(node, ENode::Value(_)))
            .count(),
        1
    );

    let mut rebuilt = Rebuilder::new(func);
    let mut emitted = HashMap::new();
    for (value, id) in source.roots {
        let result = expression_dag::emit(
            &mut rebuilt,
            &module.finite_map_graph,
            &source.dag,
            id,
            &mut emitted,
        )
        .unwrap();
        rebuilt.alias(&[value], &[result]).unwrap();
    }
    let rebuilt = rebuilt.finish(func).unwrap();
    assert_eq!(rebuilt.nodes.len(), 1000);
    assert_eq!(rebuilt.results[0], rebuilt.results[1]);
    assert!(
        rebuilt
            .nodes
            .iter()
            .all(|node| node.inputs[0] == node.inputs[1])
    );
    let mut result = module.clone();
    result.funcs[entry.index()] = rebuilt;
    xir::verify::verify(&result).unwrap();
}

#[test]
fn base_graph_residency_does_not_spend_rewrite_admission() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("resident_values");
    let values = (0..17_000)
        .map(|_| builder.param(Shape::scalar(DType::F32)))
        .collect::<Vec<_>>();
    let zero = builder.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());
    let sum = builder.add(values[0], zero);
    let entry = builder.end_func(vec![sum]);
    let module = builder.finish(entry).unwrap();
    let mut dag = ExpressionDag::default();
    for &value in &values {
        dag.intern(ExpressionNode::Value(value));
    }
    let x = dag.intern(ExpressionNode::Value(values[0]));
    let zero = dag.intern(ExpressionNode::Coefficient {
        shape: Shape::scalar(DType::F32),
        values: vec![algebraic_real::AlgebraicReal::zero()].into(),
    });
    let sum = dag.intern(ExpressionNode::Add(x, zero));
    let mut graph = RewriteGraph::new(module.entry_func(), &module.finite_map_graph).unwrap();
    graph.import(&dag).unwrap();
    let (before, _) = graph.snapshot_dag(&dag).unwrap();
    assert!(before.classes.iter().map(Vec::len).sum::<usize>() > 16_384);
    graph
        .rewrite(
            &mut dag,
            0,
            256,
            &mut AlgebraicWorkBudget::new(1 << 20, 256),
        )
        .unwrap();
    let (_, classes) = graph.snapshot_dag(&dag).unwrap();
    assert_eq!(classes[x.0], classes[sum.0]);
}

#[test]
fn refinement_exhaustion_keeps_a_complete_improved_incumbent_and_original() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("bounded_complete_roots");
    let mut shared = builder.param(Shape::scalar(DType::F32));
    let x = builder.param(Shape::scalar(DType::F32));
    for _ in 0..1000 {
        shared = builder.multiply(shared, shared);
    }
    let zero = builder.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());
    let sum = builder.add(x, zero);
    let entry = builder.end_func(vec![shared, sum, shared]);
    let module = builder.finish(entry).unwrap();
    let func = module.entry_func();
    let source = expression_dag::import(func).unwrap().unwrap();
    let x_id = source.values[x.index()].unwrap();
    let roots = source.roots.iter().map(|(_, id)| *id).collect::<Vec<_>>();
    let frontier = extract_roots_with_limits(
        func,
        &module.finite_map_graph,
        source.dag,
        &roots,
        0,
        256,
        &mut AlgebraicWorkBudget::new(1 << 16, 256),
    )
    .unwrap();
    assert!(frontier.refinement_exhausted);
    assert!(
        frontier
            .candidates
            .iter()
            .any(|candidate| candidate.roots == roots)
    );
    assert!(
        frontier
            .candidates
            .iter()
            .any(|candidate| candidate.roots[1] == x_id)
    );
    assert!(
        frontier
            .candidates
            .iter()
            .all(|candidate| candidate.roots.len() == roots.len())
    );
    let candidates = super::super::extract_func_frontier(
        func,
        &module.finite_map_graph,
        0,
        256,
        &mut AlgebraicWorkBudget::new(1 << 16, 256),
    )
    .unwrap();
    assert!(candidates.iter().any(|(_, _, changed)| *changed));
    for (func, _, _) in candidates {
        assert!(func.nodes.len() <= module.entry_func().nodes.len());
        assert_eq!(func.results[0], func.results[2]);
        let mut rebuilt = module.clone();
        rebuilt.funcs[entry.index()] = func;
        xir::verify::verify(&rebuilt).unwrap();
    }
}

#[test]
fn maps_axes_and_tensor_shapes_remain_nominal_in_the_dag() {
    let mut maps = domain_index::FiniteMapGraph::default();
    let fine = maps.add_set(8).unwrap();
    let coarse = maps.add_set(3).unwrap();
    let first = maps.add_map(fine, coarse).unwrap();
    let second = maps.add_map(fine, coarse).unwrap();
    let mut builder = ModuleBuilder::with_finite_map_graph(maps);
    builder.begin_func("nominal_axes");
    let value = builder.param(Shape::new(DType::F32, [2, 3]).with_domain_axes([coarse, coarse]));
    let a = builder.gather_along_map(value, first, 0);
    let b = builder.gather_along_map(value, first, 1);
    let c = builder.gather_along_map(value, second, 0);
    let entry = builder.end_func(vec![a, b, c]);
    let module = builder.finish(entry).unwrap();
    let func = module.entry_func();
    let mut source = expression_dag::import(func).unwrap().unwrap();
    let mut graph = RewriteGraph::new(func, &module.finite_map_graph).unwrap();
    graph.import(&source.dag).unwrap();
    graph
        .rewrite(
            &mut source.dag,
            0,
            256,
            &mut AlgebraicWorkBudget::new(4096, 256),
        )
        .unwrap();
    let (_, classes) = graph.snapshot_dag(&source.dag).unwrap();
    let identities = source
        .roots
        .iter()
        .map(|(_, id)| classes[id.0])
        .collect::<HashSet<_>>();
    assert_eq!(identities.len(), 3);
    let mut rebuilt = Rebuilder::new(func);
    let mut emitted = HashMap::new();
    for &(value, id) in &source.roots {
        let result = expression_dag::emit(
            &mut rebuilt,
            &module.finite_map_graph,
            &source.dag,
            id,
            &mut emitted,
        )
        .unwrap();
        assert_eq!(rebuilt.func.value_shape(result), func.value_shape(value));
        rebuilt.alias(&[value], &[result]).unwrap();
    }
    assert_eq!(rebuilt.func.nodes.len(), 3);
    let mut output = module.clone();
    output.funcs[entry.index()] = rebuilt.finish(func).unwrap();
    xir::verify::verify(&output).unwrap();
}

#[test]
fn unsupported_definitions_stay_at_their_ssa_boundary() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("nominal_boundary");
    let value = builder.param(Shape::scalar(DType::F32));
    let sum = builder.add(value, value);
    let boundary = builder.unary("sine", sum).unwrap();
    let product = builder.multiply(boundary, boundary);
    let entry = builder.end_func(vec![product]);
    let module = builder.finish(entry).unwrap();
    let func = module.entry_func();
    let source = expression_dag::import(func).unwrap().unwrap();
    let boundary_id = source.values[boundary.index()].unwrap();
    assert_eq!(
        source.dag.node(boundary_id),
        &ExpressionNode::Value(boundary)
    );
    assert_eq!(source.roots.len(), 2);
    let latest = expression_dag::latest_definitions(&source.dag, func);
    let crate::ir::Def::NodeResult { node: earlier, .. } = func.values[sum.index()].def else {
        unreachable!()
    };
    assert!(latest[boundary_id.0] > earlier);
    let candidates = super::super::extract_func_frontier(
        func,
        &module.finite_map_graph,
        0,
        256,
        &mut AlgebraicWorkBudget::new(4096, 256),
    )
    .unwrap();
    for (func, _, _) in candidates {
        assert_eq!(
            func.nodes
                .iter()
                .filter(|node| node.op.mnemonic() == "sine")
                .count(),
            1
        );
        let mut output = module.clone();
        output.funcs[entry.index()] = func;
        xir::verify::verify(&output).unwrap();
    }
}

#[test]
fn zero_result_calls_do_not_hide_supported_roots_or_panic() {
    for call_first in [false, true] {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("empty_callee");
        let callee = builder.end_func(vec![]);
        builder.begin_func("zero_result_call");
        let value = builder.param(Shape::scalar(DType::F32));
        if call_first {
            assert!(builder.op(crate::ops::Call { callee }, &[]).is_empty());
        }
        let result = builder.add(value, value);
        if !call_first {
            assert!(builder.op(crate::ops::Call { callee }, &[]).is_empty());
        }
        let entry = builder.end_func(vec![result]);
        let module = builder.finish(entry).unwrap();
        xir::verify::verify(&module).unwrap();
        let source = expression_dag::import(module.entry_func())
            .unwrap()
            .unwrap();
        assert_eq!(source.roots.len(), 1);
        assert_eq!(source.roots[0].0, result);
        let candidates = super::super::extract_func_frontier(
            module.entry_func(),
            &module.finite_map_graph,
            entry.index(),
            256,
            &mut AlgebraicWorkBudget::new(4096, 256),
        )
        .unwrap();
        assert!(!candidates.is_empty());
        for (func, _, _) in candidates {
            let mut output = module.clone();
            output.funcs[entry.index()] = func;
            xir::verify::verify(&output).unwrap();
        }
    }
}

#[test]
fn completed_projection_tradeoff_survives_pareto_exhaustion() {
    let mut maps = domain_index::FiniteMapGraph::default();
    let fine = maps.add_set(8).unwrap();
    let coarse = maps.add_set(3).unwrap();
    let map = maps.add_map(fine, coarse).unwrap();
    let mut b = ModuleBuilder::with_finite_map_graph(maps);
    b.begin_func("projection_tradeoff_and_deep_shared");
    let mut shared = b.param(Shape::scalar(DType::F32));
    let c = b.param(Shape::scalar(DType::F32).with_domain_axes([coarse]));
    let x = b.param(Shape::scalar(DType::F32).with_domain_axes([fine]));
    for _ in 0..1000 {
        shared = b.multiply(shared, shared);
    }
    let summed = b.sum_along_map(x, map, 0);
    let projection = b.multiply(c, summed);
    let entry = b.end_func(vec![shared, projection]);
    let module = b.finish(entry).unwrap();
    let func = module.entry_func();
    let source = expression_dag::import(func).unwrap().unwrap();
    let roots = source.roots.iter().map(|(_, id)| *id).collect::<Vec<_>>();
    let mut checked_dag = source.dag.clone();
    let mut graph = RewriteGraph::new(func, &module.finite_map_graph).unwrap();
    graph.import(&checked_dag).unwrap();
    let proofs = graph
        .rewrite(
            &mut checked_dag,
            0,
            256,
            &mut AlgebraicWorkBudget::new(1 << 20, 256),
        )
        .unwrap();
    let (_, classes) = graph.snapshot_dag(&checked_dag).unwrap();
    assert!(
        proofs.iter().any(|id| classes[id.0] == classes[roots[1].0]
            && matches!(checked_dag.node(*id), ExpressionNode::Sum { .. })),
        "fixture did not admit before-sum proof"
    );
    let frontier = extract_roots_with_limits(
        func,
        &module.finite_map_graph,
        source.dag,
        &roots,
        0,
        256,
        &mut AlgebraicWorkBudget::new(1 << 20, 256),
    )
    .unwrap();
    assert!(frontier.refinement_exhausted);
    assert!(
        frontier
            .candidates
            .iter()
            .any(|c| c.work.before_accumulation == 8 && c.work.after_accumulation == 1000),
        "lost proved placement that wins when k>=3"
    );
}
