// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::dtype::DType;

#[expect(
    clippy::mutable_key_type,
    reason = "AlgebraicReal caches preserve exact equality and hash"
)]
fn tree_quality<'a>(trees: impl IntoIterator<Item = &'a ExprTree>) -> (usize, usize, usize) {
    let mut seen = HashSet::new();
    let mut pending = trees.into_iter().collect::<Vec<_>>();
    let mut constants = 0usize;
    let mut selects = 0usize;
    while let Some(tree) = pending.pop() {
        if !seen.insert(tree) {
            continue;
        }
        match tree {
            ExprTree::Value(_) => {}
            ExprTree::Coefficient { values, .. } => {
                constants = constants.saturating_add(
                    values
                        .iter()
                        .map(|value| value.structural_size().max(1))
                        .sum::<usize>(),
                );
            }
            ExprTree::Negate(x)
            | ExprTree::Gather { value: x, .. }
            | ExprTree::Sum { value: x, .. } => pending.push(x),
            ExprTree::Add(x, y) | ExprTree::Multiply(x, y) => {
                pending.extend([x.as_ref(), y.as_ref()])
            }
            ExprTree::Select(p, x, y) => {
                selects += 1;
                pending.extend([p.as_ref(), x.as_ref(), y.as_ref()]);
            }
        }
    }
    (seen.len(), selects, constants)
}

fn expressions() -> Vec<ExprTree> {
    let mut trees = vec![ExprTree::Value(ValueId(0)), ExprTree::Value(ValueId(1))];
    for shape in [
        Shape::scalar(DType::F32),
        Shape::scalar(DType::F64),
        Shape::new(DType::F32, [1]),
        Shape::scalar(DType::F32).with_domain_axes([domain_index::FiniteSetId(0)]),
        Shape::scalar(DType::F32).with_domain_axes([domain_index::FiniteSetId(1)]),
    ] {
        for value in [AlgebraicReal::integer(2), AlgebraicReal::sqrt(2u32)] {
            trees.push(ExprTree::Coefficient {
                shape: shape.clone(),
                values: vec![value].into(),
            });
        }
    }
    for map in [domain_index::FiniteMapId(0), domain_index::FiniteMapId(1)] {
        for axis in [0, 1] {
            for value in [ValueId(0), ValueId(1)] {
                let value = Box::new(ExprTree::Value(value));
                trees.push(ExprTree::Gather {
                    map,
                    axis,
                    value: value.clone(),
                });
                trees.push(ExprTree::Sum { map, axis, value });
            }
        }
    }
    for i in 0..48 {
        let x = Box::new(trees[i % trees.len()].clone());
        let y = Box::new(trees[(i * 7 + 3) % trees.len()].clone());
        trees.push(match i % 4 {
            0 => ExprTree::Negate(x),
            1 => ExprTree::Add(x, y),
            2 => ExprTree::Multiply(x, y),
            _ => ExprTree::Select(Box::new(ExprTree::Value(ValueId(2))), x, y),
        });
    }
    trees
}

#[test]
fn identities_match_exact_tree_equality_and_preserve_nominal_fields() {
    let mut dag = ExpressionDag::default();
    let trees = expressions();
    let ids = trees
        .iter()
        .map(|tree| dag.import(tree))
        .collect::<Vec<_>>();
    for (tree, &id) in trees.iter().zip(&ids) {
        assert_eq!(dag.materialize(id), *tree);
        assert_eq!(dag.import(&tree.clone()), id);
        for (other, &other_id) in trees.iter().zip(&ids) {
            assert_eq!(id == other_id, tree == other);
        }
    }
}

#[test]
fn equal_algebraic_coefficients_have_one_identity() {
    let mut dag = ExpressionDag::default();
    let root = AlgebraicReal::sqrt(2u32);
    let squared = &root * &root;
    let coefficient = |value| ExprTree::Coefficient {
        shape: Shape::scalar(DType::F32),
        values: vec![value].into(),
    };
    let left = coefficient(squared);
    let right = coefficient(AlgebraicReal::integer(2));
    assert_eq!(left, right);
    assert_eq!(dag.import(&left), dag.import(&right));
}

#[test]
fn quality_matches_structural_tree_sets_across_shared_roots() {
    let mut dag = ExpressionDag::default();
    let trees = expressions();
    let ids = trees
        .iter()
        .map(|tree| dag.import(tree))
        .collect::<Vec<_>>();
    for (i, tree) in trees.iter().enumerate() {
        assert_eq!(dag.quality([ids[i]]), tree_quality([tree]));
        assert_eq!(dag.quality([ids[i], ids[i]]), tree_quality([tree, tree]));
        assert_eq!(
            dag.quality(ids[..=i].iter().copied()),
            tree_quality(&trees[..=i])
        );
    }
}

#[test]
fn shared_subtrees_need_one_node_per_level() {
    let mut dag = ExpressionDag::default();
    let mut id = dag.intern(ExpressionNode::Value(ValueId(0)));
    for _ in 0..256 {
        let node = ExpressionNode::Add(id, id);
        id = dag.intern(node.clone());
        assert_eq!(dag.intern(node), id);
    }
    assert_eq!(dag.nodes.len(), 257);
    assert_eq!(dag.quality([id, id]), (257, 0, 0));
}

#[test]
fn coefficient_identity_preserves_each_element_and_its_shape() {
    let mut dag = ExpressionDag::default();
    let radical = AlgebraicReal::sqrt(2u32);
    let two = AlgebraicReal::integer(2);
    let three = AlgebraicReal::integer(3);
    let coefficient = |shape, values: Vec<AlgebraicReal>| ExprTree::Coefficient {
        shape,
        values: values.into(),
    };
    let shape = Shape::new(DType::F32, [2]);
    let exact = coefficient(shape.clone(), vec![&radical * &radical, three.clone()]);
    let same = coefficient(shape.clone(), vec![two.clone(), three.clone()]);
    let swapped = coefficient(shape, vec![three.clone(), two.clone()]);
    let reshaped = coefficient(Shape::new(DType::F32, [1, 2]), vec![two, three]);
    let id = dag.import(&exact);
    assert_eq!(id, dag.import(&same));
    assert_ne!(id, dag.import(&swapped));
    assert_ne!(id, dag.import(&reshaped));
    assert_eq!(dag.materialize(id), same);
}

#[test]
fn materialization_preserves_unfolded_occurrences() {
    let mut dag = ExpressionDag::default();
    let mut id = dag.intern(ExpressionNode::Value(ValueId(0)));
    for _ in 0..11 {
        id = dag.intern(ExpressionNode::Add(id, id));
    }
    id = dag.intern(ExpressionNode::Negate(id));
    let tree = dag.materialize(id);
    let mut pending = vec![&tree];
    let mut nodes = 0;
    while let Some(tree) = pending.pop() {
        nodes += 1;
        match tree {
            ExprTree::Value(_) => {}
            ExprTree::Negate(value) => pending.push(value),
            ExprTree::Add(left, right) => pending.extend([left.as_ref(), right.as_ref()]),
            _ => unreachable!(),
        }
    }
    assert_eq!(nodes, 1 + (2usize.pow(12) - 1));
    assert_eq!(dag.nodes.len(), 13);
}
