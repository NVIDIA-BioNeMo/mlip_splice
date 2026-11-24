// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw polynomial arithmetic counts and their compiler and execution views.

use std::collections::HashMap;

use domain_index::FiniteMapGraph;

use crate::ir::Func;
use crate::ops::{Constant, ScalarOp};
use crate::shape::Shape;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PolynomialOperation {
    Add,
    Subtract,
    Negate,
    Multiply,
}

impl PolynomialOperation {
    fn from_scalar(scalar: ScalarOp) -> Option<Self> {
        match scalar {
            ScalarOp::Add => Some(Self::Add),
            ScalarOp::Subtract => Some(Self::Subtract),
            ScalarOp::Negate => Some(Self::Negate),
            ScalarOp::Multiply => Some(Self::Multiply),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PolynomialOperationCounts {
    additions: u128,
    subtractions: u128,
    negations: u128,
    multiplications: u128,
}

impl PolynomialOperationCounts {
    fn record(&mut self, operation: PolynomialOperation) {
        let count = match operation {
            PolynomialOperation::Add => &mut self.additions,
            PolynomialOperation::Subtract => &mut self.subtractions,
            PolynomialOperation::Negate => &mut self.negations,
            PolynomialOperation::Multiply => &mut self.multiplications,
        };
        *count += 1;
    }

    fn operation_count(self) -> Result<u128, String> {
        [
            self.additions,
            self.subtractions,
            self.negations,
            self.multiplications,
        ]
        .into_iter()
        .try_fold(0u128, |total, count| {
            total
                .checked_add(count)
                .ok_or_else(|| "polynomial operation count overflow".to_string())
        })
    }

    fn weighted_cost(self, weights: PolynomialOperationWeights) -> Result<u128, String> {
        let additive = self
            .additions
            .checked_add(self.subtractions)
            .and_then(|count| count.checked_add(self.negations))
            .ok_or_else(|| "polynomial additive operation count overflow".to_string())?;
        let additive = additive
            .checked_mul(u128::from(weights.additive))
            .ok_or_else(|| "weighted polynomial additive cost overflow".to_string())?;
        let multiplicative = self
            .multiplications
            .checked_mul(u128::from(weights.multiply))
            .ok_or_else(|| "weighted polynomial multiplication cost overflow".to_string())?;
        additive
            .checked_add(multiplicative)
            .ok_or_else(|| "weighted polynomial cost overflow".to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PolynomialOperationWeights {
    additive: u64,
    multiply: u64,
}

impl PolynomialOperationWeights {
    pub(super) const fn new(additive: u64, multiply: u64) -> Self {
        Self { additive, multiply }
    }

    pub(super) const fn cost(self, operation: PolynomialOperation) -> u64 {
        match operation {
            PolynomialOperation::Add
            | PolynomialOperation::Subtract
            | PolynomialOperation::Negate => self.additive,
            PolynomialOperation::Multiply => self.multiply,
        }
    }
}

pub(super) const POLYNOMIAL_EXECUTION_WEIGHTS: PolynomialOperationWeights =
    PolynomialOperationWeights::new(1, 4);

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct PolynomialCost {
    operations_by_shape: HashMap<Shape, PolynomialOperationCounts>,
}

impl PolynomialCost {
    pub(super) fn count_operations(func: &Func) -> Result<Self, String> {
        let mut live = vec![false; func.values.len()];
        for &result in &func.results {
            live[result.index()] = true;
        }
        let mut cost = Self::default();
        for node in func.nodes.iter().rev() {
            if !node.results.iter().any(|result| live[result.index()]) {
                continue;
            }
            for &input in &node.inputs {
                live[input.index()] = true;
            }
            if node.results.len() != 1 {
                continue;
            }
            let Some(operation) = node
                .op
                .as_elementwise()
                .and_then(|elementwise| PolynomialOperation::from_scalar(elementwise.scalar()))
            else {
                continue;
            };
            let shape = func.value_shape(node.results[0]);
            if !shape.dtype.is_float() {
                continue;
            }
            shape.element_count().map_err(|error| {
                format!(
                    "cannot count polynomial operation with shape {:?}: {error}",
                    shape.dims
                )
            })?;
            cost.operations_by_shape
                .entry(shape.clone())
                .or_default()
                .record(operation);
        }
        Ok(cost)
    }

    pub(super) fn max_exact_polynomial_operation_depth(func: &Func) -> usize {
        let mut depths = vec![0usize; func.values.len()];
        let mut exact = vec![false; func.values.len()];
        let mut maximum = 0usize;
        for node in &func.nodes {
            if let Some(constant) = node.op.as_any().downcast_ref::<Constant>() {
                if constant.literal.exact.is_some() {
                    for result in &node.results {
                        exact[result.index()] = true;
                    }
                }
                continue;
            }
            if node.results.len() != 1
                || node
                    .op
                    .as_elementwise()
                    .and_then(|elementwise| PolynomialOperation::from_scalar(elementwise.scalar()))
                    .is_none()
            {
                continue;
            }
            let result = node.results[0];
            if !func.value_shape(result).dtype.is_float() {
                continue;
            }
            let depth = node
                .inputs
                .iter()
                .map(|input| depths[input.index()])
                .max()
                .unwrap_or(0)
                .saturating_add(1);
            let depends_on_exact = node.inputs.iter().any(|input| exact[input.index()]);
            depths[result.index()] = depth;
            exact[result.index()] = depends_on_exact;
            if depends_on_exact {
                maximum = maximum.max(depth);
            }
        }
        maximum
    }

    pub(super) fn static_operation_count(&self) -> Result<u128, String> {
        self.operations_by_shape
            .values()
            .try_fold(0u128, |total, counts| {
                total
                    .checked_add(counts.operation_count()?)
                    .ok_or_else(|| "static polynomial operation count overflow".to_string())
            })
    }

    fn execution_cost(
        &self,
        graph: &FiniteMapGraph,
        weights: PolynomialOperationWeights,
    ) -> Result<PolynomialExecutionCost, String> {
        let mut by_shape = HashMap::with_capacity(self.operations_by_shape.len());
        let mut total = 0u128;
        for (shape, &counts) in &self.operations_by_shape {
            let mut executions = shape.element_count()? as u128;
            for &axis in &shape.domain_axes {
                let extent = graph.set(axis)?.extent;
                if extent < 1 {
                    return Err(format!(
                        "finite set {} extent must be positive, got {extent}",
                        axis.index()
                    ));
                }
                executions = executions
                    .checked_mul(u128::try_from(extent).expect("positive i64 fits u128"))
                    .ok_or_else(|| {
                        format!("execution count overflows for shape {:?}", shape.dims)
                    })?;
            }
            let weighted = counts
                .weighted_cost(weights)?
                .checked_mul(executions)
                .ok_or_else(|| format!("execution cost overflows for shape {:?}", shape.dims))?;
            total = total
                .checked_add(weighted)
                .ok_or_else(|| "total polynomial execution cost overflow".to_string())?;
            by_shape.insert(shape.clone(), weighted);
        }
        Ok(PolynomialExecutionCost { by_shape, total })
    }
}

struct PolynomialExecutionCost {
    by_shape: HashMap<Shape, u128>,
    total: u128,
}

pub(super) fn candidate_is_profitable(
    graph: &FiniteMapGraph,
    baseline: &PolynomialCost,
    candidate: &PolynomialCost,
    weights: PolynomialOperationWeights,
) -> Result<bool, String> {
    let baseline = baseline.execution_cost(graph, weights)?;
    let candidate = candidate.execution_cost(graph, weights)?;
    let non_worse_by_shape = candidate
        .by_shape
        .iter()
        .all(|(shape, &cost)| cost <= baseline.by_shape.get(shape).copied().unwrap_or(0));
    if !non_worse_by_shape {
        return Ok(false);
    }
    let three_candidate = candidate
        .total
        .checked_mul(3)
        .ok_or_else(|| "scaled candidate polynomial cost overflow".to_string())?;
    let two_baseline = baseline
        .total
        .checked_mul(2)
        .ok_or_else(|| "scaled baseline polynomial cost overflow".to_string())?;
    Ok(three_candidate < two_baseline)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ModuleBuilder;
    use crate::dtype::DType;
    use crate::ops::{Literal, Negate, Subtract};

    fn cost_with_shapes(
        entries: impl IntoIterator<Item = (Shape, PolynomialOperationCounts)>,
    ) -> PolynomialCost {
        PolynomialCost {
            operations_by_shape: entries.into_iter().collect(),
        }
    }

    #[test]
    fn exact_polynomial_depth_tracks_the_longest_arithmetic_chain() {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("depth");
        let shape = Shape::scalar(DType::F32);
        let input = builder.param(shape.clone());
        let other = builder.param(shape.clone());
        let coefficient = builder.constant(
            Literal::exact(shape, vec![algebraic_real::AlgebraicReal::sqrt(2u32)]).unwrap(),
        );
        let scaled = builder.multiply(input, coefficient);
        let chain = (0..7).fold(scaled, |value, _| builder.add(value, other));
        let independent = builder.multiply(input, other);
        let entry = builder.end_func(vec![chain, independent]);
        let module = builder.finish(entry).unwrap();

        assert_eq!(
            PolynomialCost::max_exact_polynomial_operation_depth(module.entry_func()),
            8
        );
    }

    #[test]
    fn records_each_polynomial_operation_separately() {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("operation_counts");
        let shape = Shape::scalar(DType::F32);
        let left = builder.param(shape.clone());
        let right = builder.param(shape);
        let add = builder.add(left, right);
        let subtract = builder.op1(Subtract, &[left, right]);
        let negate = builder.op1(Negate, &[left]);
        let multiply = builder.multiply(left, right);
        let entry = builder.end_func(vec![add, subtract, negate, multiply]);
        let module = builder.finish(entry).unwrap();

        let cost = PolynomialCost::count_operations(module.entry_func()).unwrap();
        assert_eq!(cost.static_operation_count().unwrap(), 4);
        assert_eq!(
            cost.operations_by_shape[&Shape::scalar(DType::F32)],
            PolynomialOperationCounts {
                additions: 1,
                subtractions: 1,
                negations: 1,
                multiplications: 1,
            }
        );
    }

    #[test]
    fn counts_a_shared_operation_once() {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("shared_operation");
        let shape = Shape::scalar(DType::F32);
        let left = builder.param(shape.clone());
        let right = builder.param(shape.clone());
        let first_factor = builder.param(shape.clone());
        let second_factor = builder.param(shape);
        let shared = builder.add(left, right);
        let first = builder.multiply(shared, first_factor);
        let second = builder.multiply(shared, second_factor);
        let entry = builder.end_func(vec![first, second]);
        let module = builder.finish(entry).unwrap();

        let cost = PolynomialCost::count_operations(module.entry_func()).unwrap();
        assert_eq!(cost.static_operation_count().unwrap(), 3);
    }

    #[test]
    fn static_view_omits_dead_operations() {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("dead_operation");
        let shape = Shape::scalar(DType::F32);
        let left = builder.param(shape.clone());
        let right = builder.param(shape);
        let _dead = builder.add(left, right);
        let live = builder.multiply(left, right);
        let entry = builder.end_func(vec![live]);
        let module = builder.finish(entry).unwrap();

        let cost = PolynomialCost::count_operations(module.entry_func()).unwrap();
        assert_eq!(cost.static_operation_count().unwrap(), 1);
    }

    #[test]
    fn execution_cost_scales_with_tensor_and_domain_extents() {
        let mut graph = FiniteMapGraph::default();
        let first_domain = graph.add_set(3).unwrap();
        let second_domain = graph.add_set(5).unwrap();
        let mut builder = ModuleBuilder::with_finite_map_graph(graph.clone());
        builder.begin_func("domain_scaling");
        let shape = Shape::new(DType::F32, [2]).with_domain_axes([first_domain, second_domain]);
        let left = builder.param(shape.clone());
        let right = builder.param(shape.clone());
        let add = builder.add(left, right);
        let multiply = builder.multiply(left, right);
        let entry = builder.end_func(vec![add, multiply]);
        let module = builder.finish(entry).unwrap();

        let cost = PolynomialCost::count_operations(module.entry_func()).unwrap();
        let execution = cost
            .execution_cost(&graph, POLYNOMIAL_EXECUTION_WEIGHTS)
            .unwrap();
        assert_eq!(execution.total, (1 + 4) * 2 * 3 * 5);
        assert_eq!(execution.by_shape[&shape], execution.total);
    }

    #[test]
    fn improvement_is_strict_at_the_one_and_a_half_boundary() {
        let shape = Shape::scalar(DType::F32);
        let baseline = cost_with_shapes([(
            shape.clone(),
            PolynomialOperationCounts {
                additions: 3,
                ..PolynomialOperationCounts::default()
            },
        )]);
        let boundary = cost_with_shapes([(
            shape.clone(),
            PolynomialOperationCounts {
                additions: 2,
                ..PolynomialOperationCounts::default()
            },
        )]);
        let improved = cost_with_shapes([(
            shape,
            PolynomialOperationCounts {
                additions: 1,
                ..PolynomialOperationCounts::default()
            },
        )]);
        let graph = FiniteMapGraph::default();

        assert!(
            !candidate_is_profitable(&graph, &baseline, &boundary, POLYNOMIAL_EXECUTION_WEIGHTS,)
                .unwrap()
        );
        assert!(
            candidate_is_profitable(&graph, &baseline, &improved, POLYNOMIAL_EXECUTION_WEIGHTS,)
                .unwrap()
        );
    }

    #[test]
    fn improvement_cannot_hide_a_regression_in_one_execution_bucket() {
        let mut graph = FiniteMapGraph::default();
        let first_domain = graph.add_set(1).unwrap();
        let second_domain = graph.add_set(1).unwrap();
        let first_shape = Shape::new(DType::F32, [1]).with_domain_axes([first_domain]);
        let second_shape = Shape::new(DType::F32, [1]).with_domain_axes([second_domain]);
        let baseline = cost_with_shapes([
            (
                first_shape.clone(),
                PolynomialOperationCounts {
                    multiplications: 100,
                    ..PolynomialOperationCounts::default()
                },
            ),
            (
                second_shape.clone(),
                PolynomialOperationCounts {
                    additions: 1,
                    ..PolynomialOperationCounts::default()
                },
            ),
        ]);
        let candidate = cost_with_shapes([
            (
                first_shape,
                PolynomialOperationCounts {
                    multiplications: 1,
                    ..PolynomialOperationCounts::default()
                },
            ),
            (
                second_shape,
                PolynomialOperationCounts {
                    additions: 2,
                    ..PolynomialOperationCounts::default()
                },
            ),
        ]);

        assert!(
            !candidate_is_profitable(&graph, &baseline, &candidate, POLYNOMIAL_EXECUTION_WEIGHTS,)
                .unwrap()
        );
    }

    #[test]
    fn uncountable_shape_is_not_silently_omitted() {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("uncountable");
        let shape = Shape::scalar(DType::F32);
        let left = builder.param(shape.clone());
        let right = builder.param(shape);
        let result = builder.add(left, right);
        let entry = builder.end_func(vec![result]);
        let mut module = builder.finish(entry).unwrap();
        module.funcs[entry.index()].values[result.index()]
            .shape
            .dims = crate::shape::Dims::from_slice(&[i64::MAX, 3]);

        let error = PolynomialCost::count_operations(module.entry_func()).unwrap_err();
        assert!(
            error.contains("cannot count polynomial operation"),
            "{error}"
        );
    }
}
