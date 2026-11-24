// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use xir::builder::ModuleBuilder;
use xir::shape::Shape;

/// A module whose first function is the two-argument add the reduction
/// reduces with, and whose entry is a one-parameter identity — enough to
/// stand an evaluator up.
fn evaluator_module() -> xir::ir::Module {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("add");
    let left = builder.param(Shape::scalar(DType::F32));
    let right = builder.param(Shape::scalar(DType::F32));
    let sum = builder.op(xir::ops::Add, &[left, right])[0];
    builder.end_func(vec![sum]);
    builder.begin_func("main");
    let value = builder.param(Shape::scalar(DType::F32));
    let main = builder.end_func(vec![value]);
    builder.finish(main).unwrap()
}

fn source(b: &mut gpu_ir::Builder, elements: usize) -> IndexedTensor {
    let base = b.parameter(gpu_ir::Param {
        name: format!("operand_{}", b.function().num_values()),
        ty: gpu_ir::ParamTy::Ptr(Scalar::F32),
    });
    IndexedTensor::pointer(
        base,
        Scalar::F32,
        1,
        1,
        1,
        false,
        None,
        Evict::Normal,
        elements,
    )
}

/// One reduction over `terms` inputs, all landing on a single output — the
/// shape a recovered non-unique scatter takes by the time it reaches here.
fn reduction(b: &mut gpu_ir::Builder, terms: usize) -> IndexedTensor {
    let init = IndexedTensor::dense(vec![b.float(Scalar::F32, 0.0)]);
    let source = source(b, terms);
    IndexedTensor::scatter(
        "recovered".into(),
        init,
        source,
        xir::ir::FuncId(0),
        &vec![Some(0); terms],
    )
    .unwrap()
}

/// Whether a reduction of `terms` inputs came out as a loop rather than as an
/// unrolled chain.
fn folds(terms: usize) -> bool {
    let module = evaluator_module();
    let mut evaluator = crate::eval::Evaluator::new(&module, 1);
    let x = evaluator.builder().parameter(gpu_ir::Param {
        name: "x".into(),
        ty: gpu_ir::ParamTy::Scalar(Scalar::F32),
    });
    let base = source(evaluator.builder(), 1);
    let reduction = reduction(evaluator.builder(), terms);
    evaluator
        .bind_params(vec![vec![x]], Some(vec![base]))
        .unwrap();
    let emitted = evaluator.emit_indexed_at(&reduction, 0).unwrap();
    let body = evaluator.builder().function();
    let held = emitted.id_in(body);
    // A loop standing for the value it reduces to is an operand of whatever
    // reads it, so it is found by walking what the emitted value is made of.
    fn contains_loop(f: &gpu_ir::Function, value: gpu_ir::ValueId) -> bool {
        let Some(op) = f.operand_op(value) else {
            return false;
        };
        matches!(f.op(op).kind, gpu_ir::OpKind::Loop { .. })
            || f.op(op)
                .operands
                .iter()
                .any(|&operand| contains_loop(f, operand))
    }
    contains_loop(body, held)
}

/// The whole point of recovering a wide reduction: past the unroll bound the
/// backend walks its terms with a loop, so the program stops growing with the
/// term count. Without this a recovered region is just as large as the scalar
/// code it replaced.
#[test]
fn a_wide_reduction_becomes_a_loop() {
    assert!(
        folds(MAX_UNROLLED_REDUCTION_TERMS * 4),
        "a reduction past the unroll bound did not become a loop"
    );
}

/// Below the bound, naming each term is worth its space: it exposes the terms
/// to common subexpression elimination and to fused multiply-add.
#[test]
fn a_narrow_reduction_stays_unrolled() {
    assert!(
        !folds(MAX_UNROLLED_REDUCTION_TERMS),
        "a reduction within the unroll bound was rolled anyway"
    );
}

#[test]
fn index_table_resolves_constant_indices_without_a_table_load() {
    let table = IndexTable::new("indices".into(), &[3, 5, 8]).unwrap();
    let mut b = gpu_ir::Builder::new();
    let at = b.i32(1);
    let resolved = table.load(&mut b, at);
    assert_eq!(b.as_int_const(&resolved), Some(5));
    let index = b.parameter(gpu_ir::Param {
        name: "index".into(),
        ty: gpu_ir::ParamTy::Scalar(Scalar::S32),
    });
    let loaded = table.load(&mut b, index);
    let f = b.function();
    assert!(matches!(
        f.operand_op(loaded.id_in(f)).map(|op| &f.op(op).kind),
        Some(gpu_ir::OpKind::ConstLoad(_))
    ));
}

/// Re-indexing denotes no value of its own, so reads through it belong to the
/// value underneath.
///
/// Counting the alias instead credits each wrapper with its own read and misses
/// that one leaf was read twice — the leaf then looks private, is not named, and
/// is rebuilt at every reach. The plan resolves aliases before hash-consing, so
/// both reads land on one node.
#[test]
fn reads_through_an_alias_belong_to_the_value_underneath() {
    let mut b = gpu_ir::Builder::new();
    let pool = source(&mut b, 8);
    // The same element of the same re-index, read twice.
    let alias = pool.reindex("alias".into(), vec![5, 6]).unwrap();
    let doubled = IndexedTensor::elementwise(
        ScalarOp::Add,
        DType::F32,
        vec![alias.clone(), alias.clone()],
        2,
    );
    let dag = schedule::plan(&[(doubled, 0)], 0).unwrap();

    let leaf = dag
        .nodes
        .iter()
        .position(|node| node.tensor.element_key(node.element) == pool.element_key(5))
        .expect("the alias resolved to the pool element");
    assert_eq!(
        dag.nodes[leaf].uses, 2,
        "reads through the alias were counted apart from the leaf"
    );
    // And two different aliases onto one element are still that one element.
    let other = pool.reindex("other".into(), vec![5]).unwrap();
    let mixed =
        IndexedTensor::elementwise(ScalarOp::Add, DType::F32, vec![alias.clone(), other], 1);
    let dag = schedule::plan(&[(mixed, 0)], 0).unwrap();
    let leaf = dag
        .nodes
        .iter()
        .position(|node| node.tensor.element_key(node.element) == pool.element_key(5))
        .expect("both aliases resolved to the pool element");
    assert_eq!(dag.nodes[leaf].uses, 2);
}

/// A reduction the backend walks with a loop reads its terms under a runtime
/// index. Those reads are not static, so the plan must not name the terms: it
/// would announce a live count the emission never creates.
#[test]
fn a_rolled_reduction_does_not_name_its_terms() {
    let terms = MAX_UNROLLED_REDUCTION_TERMS * 4;
    let mut b = gpu_ir::Builder::new();
    let rolled = reduction(&mut b, terms);
    let dag = schedule::plan(&[(rolled, 0)], 0).unwrap();
    // The reduction itself is a node; none of its terms are.
    assert_eq!(
        dag.nodes.len(),
        1,
        "a rolled reduction pre-named terms the fold reads dynamically"
    );

    // Within the bound the terms are static and do become nodes.
    let unrolled = reduction(&mut b, MAX_UNROLLED_REDUCTION_TERMS);
    let dag = schedule::plan(&[(unrolled, 0)], 0).unwrap();
    assert!(
        dag.nodes.len() > 1,
        "an unrolled reduction lost the terms it reads statically"
    );
}

/// Which branch a constant-index gather takes, proven rather than assumed.
///
/// `lower_gather` resolves the indices at lowering time whenever the start
/// operand is a constant, and the recovery's index vectors are exactly that. So
/// a gather becomes a re-indexing of the pool, not a runtime address
/// computation: two lanes reading the same pool element resolve to one named
/// value, and nothing downstream sees index arithmetic.
#[test]
fn a_constant_index_gather_reindexes_the_pool_instead_of_computing_addresses() {
    use xir::builder::ModuleBuilder;
    use xir::ops::{Constant, Literal};
    use xir::shape::Shape;

    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let pool = builder.param(Shape::new(DType::F32, [8]));
    // Lanes 0 and 1 both read pool element 3.
    let picks: [i32; 4] = [3, 3, 5, 7];
    let index_shape = Shape::new(DType::I32, [4, 1]);
    let indices = builder.op(
        Constant {
            literal: Literal::new(
                index_shape,
                picks.iter().flat_map(|value| value.to_le_bytes()).collect(),
            ),
        },
        &[],
    )[0];
    let gathered = builder.op(take_vector_for_test(), &[pool, indices])[0];
    let main = builder.end_func(vec![gathered]);
    let module = builder.finish(main).unwrap();

    let mut evaluator = crate::eval::Evaluator::new(&module, 1);
    let unused = (0..8)
        .map(|element| {
            evaluator.builder().parameter(gpu_ir::Param {
                name: format!("unused_{element}"),
                ty: gpu_ir::ParamTy::Scalar(Scalar::F32),
            })
        })
        .collect::<Vec<_>>();
    let pool = source(evaluator.builder(), 8);
    evaluator
        .bind_params(vec![unused], Some(vec![pool]))
        .unwrap();
    let evaluation = evaluator.run_with_provenance().unwrap();
    let lanes = &evaluation.outputs[0];
    let f = evaluation.body.function();

    assert_eq!(lanes.len(), 4);
    // Both lanes that read element 3 must be the very same named value. A runtime
    // address computation could not produce that.
    assert!(
        f.is_materialized(lanes[0]),
        "a constant-index gather did not resolve to a named pool read"
    );
    assert_eq!(
        lanes[0], lanes[1],
        "two lanes reading one pool element were not shared"
    );
    assert_ne!(lanes[0], lanes[2]);
}

/// The gather the recovery emits: one element of a flat pool per lane.
fn take_vector_for_test() -> xir::ops::Gather {
    let mut dnums = xir::dim_numbers::GatherDimensionNumbers::default();
    dnums.collapsed_slice_dims.push(0);
    dnums.start_index_map.push(0);
    dnums.index_vector_dim = 1;
    xir::ops::Gather {
        dnums,
        slice_sizes: xir::shape::Dims::from_slice(&[1]),
        indices_are_sorted: false,
    }
}
