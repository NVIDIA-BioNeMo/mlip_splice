// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read-only structural analysis over an XIR [`Func`]. Analyses here *produce*
//! facts for consumers such as `xir_optimize`. Every fact is derived in a
//! bounded number of forward/backward sweeps
//! over the dataflow DAG, so the whole analysis is near-linear in the graph and
//! deterministic.
//!
//! The generic facts are structural — they need only the [`Func`]: per value a
//! use count, liveness, a provable-zero flag, and a scalar-statement estimate;
//! per node its purity and whether dead-code elimination retains it. Loom layers
//! its axis-role facts (hidden-lane dependence, lane uniformity, DomainIndex
//! dependence, register and memory-traffic estimates) on top of this table where
//! the operand layout is known.

use algebraic_real::AlgebraicReal;
use smallvec::{SmallVec, smallvec};

use crate::ir::{Def, Func, FuncId, Module, Node, ValueId};
use crate::op::Op;
use crate::ops::{
    BroadcastInDim, Concatenate, Constant, Convert, DotGeneral, Gather, GatherAlongMap, Literal,
    Pad, Reduce, Reshape, ScalarOp, Scatter, Slice, SumAlongMap, Transpose,
};
use crate::shape::Shape;

/// Whether an op is free of side effects, so a dead result lets the whole node
/// be removed and two identical nodes may be shared. The list is the closed set
/// of ops Loom lowers; an unknown op is conservatively impure.
pub fn is_pure(op: &dyn Op) -> bool {
    op.as_elementwise().is_some()
        || op.as_any().is::<BroadcastInDim>()
        || op.as_any().is::<Concatenate>()
        || op.as_any().is::<Constant>()
        || op.as_any().is::<Convert>()
        || op.as_any().is::<DotGeneral>()
        || op.as_any().is::<GatherAlongMap>()
        || op.as_any().is::<Pad>()
        || op.as_any().is::<Reduce>()
        || op.as_any().is::<Reshape>()
        || op.as_any().is::<Slice>()
        || op.as_any().is::<SumAlongMap>()
        || op.as_any().is::<Transpose>()
}

/// A structural identity key for a node: its op mnemonic, printed attributes,
/// and (for a constant) its literal. Two pure nodes with equal keys over equal
/// operands compute the same value, which is what common-subexpression
/// elimination deduplicates and what shape canonicalization uses to detect that
/// a rewrite actually changed the op.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OpKey {
    mnemonic: &'static str,
    attrs: Vec<(&'static str, String)>,
    literal: Option<(Shape, Vec<u8>, Option<Vec<AlgebraicReal>>)>,
}

pub fn op_key(op: &dyn Op) -> OpKey {
    OpKey {
        mnemonic: op.mnemonic(),
        attrs: op.attrs(),
        literal: op.as_any().downcast_ref::<Constant>().map(|constant| {
            (
                constant.literal.shape.clone(),
                constant.literal.data.clone(),
                constant.literal.exact.clone(),
            )
        }),
    }
}

/// The literal a value holds when a single-result `Constant` produces it, else
/// `None`. This is the constant-ness query constant folding and algebraic
/// canonicalization share.
pub fn constant_literal(func: &Func, value: ValueId) -> Option<&Literal> {
    let Def::NodeResult { node, slot: 0 } = func.values[value.index()].def else {
        return None;
    };
    let node = &func.nodes[node];
    (node.results.len() == 1)
        .then(|| node.op.as_any().downcast_ref::<Constant>())
        .flatten()
        .map(|constant| &constant.literal)
}

/// When `value` is the single result of a single-operand node whose op is `T`,
/// that op and its operand. This is the one-hop producer query the shape and
/// algebraic rewrites walk (e.g. `transpose ∘ transpose`, `neg ∘ neg`).
pub fn unary_producer<T: Clone + 'static>(func: &Func, value: ValueId) -> Option<(T, ValueId)> {
    let Def::NodeResult { node, slot: 0 } = func.values[value.index()].def else {
        return None;
    };
    let node = &func.nodes[node];
    if node.results.len() != 1 || node.inputs.len() != 1 {
        return None;
    }
    node.op
        .as_any()
        .downcast_ref::<T>()
        .cloned()
        .map(|op| (op, node.inputs[0]))
}

/// Structural facts about one value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueFacts {
    /// Number of consumers: appearances as a node operand plus as a function
    /// result. A value used twice by one node counts twice.
    pub uses: u32,
    /// Reaches a function result through retained nodes; dead-code elimination
    /// keeps this value's producer.
    pub live: bool,
    /// Provably the real-algebra zero in every element: a zero constant, a
    /// shape/convert/negate/abs of a zero, a product with a zero, a sum of
    /// zeros, a dot or reduction that a zero forces. `x * 0` is included even
    /// though IEEE `NaN * 0` differs — the fact is real-algebra structural; the
    /// rewrite that acts on it gates on the math policy.
    pub exact_zero: bool,
    /// Estimated scalar-VIR statements to materialize this value's producer:
    /// the result element count times the op's per-element scalar cost (1 for
    /// pointwise and shape ops, the contracted extent for a dot, the reduced
    /// extent for a reduction, 0 for a constant or a reshape that only relabels
    /// elements). Saturates at `u64::MAX`; coarse by design.
    pub statements: u64,
}

/// Structural facts about one node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeFacts {
    pub pure: bool,
    /// Retained by dead-code elimination: the op is impure or some result is
    /// live.
    pub live: bool,
}

/// A per-value and per-node fact table for one function, produced by three
/// linear sweeps of the DAG. It is a pure function of the function: analyzing
/// the same function twice yields the same table.
#[derive(Clone, Debug)]
pub struct FuncFacts {
    values: Vec<ValueFacts>,
    nodes: Vec<NodeFacts>,
}

impl FuncFacts {
    pub fn analyze(func: &Func) -> FuncFacts {
        let n = func.values.len();

        // Forward sweep (nodes are topologically ordered): provable zeros and
        // statement estimates flow from operands to results.
        let mut exact_zero = vec![false; n];
        let mut statements = vec![0u64; n];
        for node in &func.nodes {
            let zeros = result_exact_zero(node, &exact_zero);
            let cost = per_element_cost(func, node);
            for (slot, &result) in node.results.iter().enumerate() {
                exact_zero[result.index()] = zeros.get(slot).copied().unwrap_or(false);
                statements[result.index()] =
                    num_scalars(func.value_shape(result)).saturating_mul(cost);
            }
        }

        // Use counts: one shared CSR index plus the function results.
        let readers = weave::Uses::of_arena(func);
        let mut uses = (0..n)
            .map(|value| u32::try_from(readers.of_value(value).len()).unwrap_or(u32::MAX))
            .collect::<Vec<_>>();
        for &result in &func.results {
            uses[result.index()] = uses[result.index()].saturating_add(1);
        }

        // Backward sweep: liveness seeds at the results and flows to operands
        // through retained nodes.
        let mut live = vec![false; n];
        for &result in &func.results {
            live[result.index()] = true;
        }
        let mut nodes = vec![
            NodeFacts {
                pure: true,
                live: false
            };
            func.nodes.len()
        ];
        for (index, node) in func.nodes.iter().enumerate().rev() {
            let pure = is_pure(node.op.as_ref());
            let retained = !pure || node.results.iter().any(|&r| live[r.index()]);
            nodes[index] = NodeFacts {
                pure,
                live: retained,
            };
            if retained {
                for &input in &node.inputs {
                    live[input.index()] = true;
                }
            }
        }

        let values = (0..n)
            .map(|i| ValueFacts {
                uses: uses[i],
                live: live[i],
                exact_zero: exact_zero[i],
                statements: statements[i],
            })
            .collect();
        FuncFacts { values, nodes }
    }

    pub fn value(&self, value: ValueId) -> &ValueFacts {
        &self.values[value.index()]
    }

    pub fn node(&self, index: usize) -> &NodeFacts {
        &self.nodes[index]
    }

    pub fn values(&self) -> &[ValueFacts] {
        &self.values
    }

    pub fn nodes(&self) -> &[NodeFacts] {
        &self.nodes
    }
}

/// Provable-zero flag for each result of `node`, given the flags of its
/// operands. Conservative: an op not listed leaves its results non-zero.
fn result_exact_zero(node: &Node, zero: &[bool]) -> SmallVec<[bool; 1]> {
    let z = |value: ValueId| zero[value.index()];
    let op = node.op.as_ref();
    if let Some(constant) = op.as_any().downcast_ref::<Constant>() {
        smallvec![constant.literal.exact.as_ref().map_or_else(
            || {
                !constant.literal.data.is_empty() && constant.literal.data.iter().all(|&b| b == 0)
            },
            |values| !values.is_empty() && values.iter().all(AlgebraicReal::is_zero),
        )]
    } else if op.as_any().is::<Convert>()
        || op.as_any().is::<Reshape>()
        || op.as_any().is::<Transpose>()
        || op.as_any().is::<BroadcastInDim>()
        || op.as_any().is::<Slice>()
        || op.as_any().is::<GatherAlongMap>()
        || op.as_any().is::<SumAlongMap>()
    {
        smallvec![z(node.inputs[0])]
    } else if op.as_any().is::<Concatenate>() {
        smallvec![node.inputs.iter().all(|&v| z(v))]
    } else if op.as_any().is::<Pad>() {
        smallvec![z(node.inputs[0]) && z(node.inputs[1])]
    } else if op.as_any().is::<DotGeneral>() {
        smallvec![z(node.inputs[0]) || z(node.inputs[1])]
    } else if op.as_any().is::<Reduce>() {
        let k = node.inputs.len() / 2;
        (0..k)
            .map(|i| z(node.inputs[i]) && z(node.inputs[k + i]))
            .collect()
    } else if let Some(elementwise) = op.as_elementwise() {
        smallvec![match elementwise.scalar() {
            ScalarOp::Multiply => node.inputs.iter().any(|&v| z(v)),
            ScalarOp::Add | ScalarOp::Subtract => node.inputs.iter().all(|&v| z(v)),
            ScalarOp::Negate | ScalarOp::Abs => z(node.inputs[0]),
            _ => false,
        }]
    } else {
        smallvec![false; node.results.len()]
    }
}

/// Scalar statements to materialize one output element of `node`'s producer.
fn per_element_cost(func: &Func, node: &Node) -> u64 {
    let op = node.op.as_ref();
    if op.as_any().is::<Constant>() || op.as_any().is::<Reshape>() {
        0
    } else if let Some(reduce) = op.as_any().downcast_ref::<Reduce>() {
        let input = func.value_shape(node.inputs[0]);
        saturating_product(
            reduce
                .dimensions
                .iter()
                .map(|&axis| input.dims[axis as usize]),
        )
    } else if let Some(dot) = op.as_any().downcast_ref::<DotGeneral>() {
        let lhs = func.value_shape(node.inputs[0]);
        saturating_product(
            dot.dnums
                .lhs_contracting_dimensions
                .iter()
                .map(|&axis| lhs.dims[axis as usize]),
        )
    } else {
        1
    }
}

/// Floating-point operations one evaluation of `module` performs. Every result
/// element of an arithmetic op counts one, a reduction counts its reducer once
/// per reduced element, and a dot counts a multiply and an add per contracted
/// term. Data movement, comparison, selection, and bit work are free. An op the
/// classification does not cover is an error rather than a silent zero, so the
/// count stays a statement about the program instead of about the classifier.
pub fn flops(module: &Module) -> Result<u64, String> {
    let mut memo = vec![None; module.funcs.len()];
    func_flops(module, module.entry, &mut memo)
}

fn func_flops(module: &Module, id: FuncId, memo: &mut Vec<Option<u64>>) -> Result<u64, String> {
    if let Some(known) = memo[id.index()] {
        return Ok(known);
    }
    let func = module.func(id);
    let mut total = 0u64;
    for node in &func.nodes {
        total = total.saturating_add(node_flops(module, func, node, memo)?);
    }
    memo[id.index()] = Some(total);
    Ok(total)
}

fn node_flops(
    module: &Module,
    func: &Func,
    node: &Node,
    memo: &mut Vec<Option<u64>>,
) -> Result<u64, String> {
    let op = node.op.as_ref();
    let results = || num_scalars(func.value_shape(node.results[0]));
    if let Some(elementwise) = op.as_elementwise() {
        return Ok(if is_arithmetic(elementwise.scalar()) {
            results()
        } else {
            0
        });
    }
    if let Some(reduce) = op.as_any().downcast_ref::<Reduce>() {
        let input = func.value_shape(node.inputs[0]);
        let per_element = func_flops(module, reduce.reducer, memo)?;
        return Ok(num_scalars(input).saturating_mul(per_element));
    }
    if let Some(scatter) = op.as_any().downcast_ref::<Scatter>() {
        let updates = func.value_shape(node.inputs[2]);
        let per_element = func_flops(module, scatter.update, memo)?;
        return Ok(num_scalars(updates).saturating_mul(per_element));
    }
    if op.as_any().is::<SumAlongMap>() {
        return Ok(num_scalars(func.value_shape(node.inputs[0])));
    }
    if let Some(dot) = op.as_any().downcast_ref::<DotGeneral>() {
        let lhs = func.value_shape(node.inputs[0]);
        let contracted = saturating_product(
            dot.dnums
                .lhs_contracting_dimensions
                .iter()
                .map(|&axis| lhs.dims[axis as usize]),
        );
        return Ok(results().saturating_mul(contracted).saturating_mul(2));
    }
    if op.as_any().is::<BroadcastInDim>()
        || op.as_any().is::<Concatenate>()
        || op.as_any().is::<Constant>()
        || op.as_any().is::<Convert>()
        || op.as_any().is::<Gather>()
        || op.as_any().is::<GatherAlongMap>()
        || op.as_any().is::<Pad>()
        || op.as_any().is::<Reshape>()
        || op.as_any().is::<Slice>()
        || op.as_any().is::<Transpose>()
    {
        return Ok(0);
    }
    Err(format!(
        "xir flop count does not classify {:?}",
        op.mnemonic()
    ))
}

/// Whether a scalar op costs floating-point arithmetic. Sign, rounding, and
/// extremum ops are bit or comparison work, so they are free.
fn is_arithmetic(op: ScalarOp) -> bool {
    match op {
        ScalarOp::Exponential
        | ScalarOp::ExponentialMinusOne
        | ScalarOp::Log
        | ScalarOp::LogPlusOne
        | ScalarOp::Sqrt
        | ScalarOp::Rsqrt
        | ScalarOp::Cbrt
        | ScalarOp::Tanh
        | ScalarOp::Sine
        | ScalarOp::Cosine
        | ScalarOp::Tan
        | ScalarOp::Logistic
        | ScalarOp::Add
        | ScalarOp::Subtract
        | ScalarOp::Multiply
        | ScalarOp::Divide
        | ScalarOp::Power
        | ScalarOp::Remainder
        | ScalarOp::Atan2 => true,
        ScalarOp::Negate
        | ScalarOp::Abs
        | ScalarOp::Floor
        | ScalarOp::Ceil
        | ScalarOp::RoundNearestEven
        | ScalarOp::RoundNearestAfz
        | ScalarOp::Sign
        | ScalarOp::Not
        | ScalarOp::CountLeadingZeros
        | ScalarOp::Popcnt
        | ScalarOp::IsFinite
        | ScalarOp::Maximum
        | ScalarOp::Minimum
        | ScalarOp::And
        | ScalarOp::Or
        | ScalarOp::Xor
        | ScalarOp::ShiftLeft
        | ScalarOp::ShiftRightArithmetic
        | ScalarOp::ShiftRightLogical
        | ScalarOp::Compare(_)
        | ScalarOp::Select
        | ScalarOp::Clamp => false,
    }
}

fn num_scalars(shape: &Shape) -> u64 {
    saturating_product(shape.dims.iter().copied())
}

fn saturating_product(dims: impl IntoIterator<Item = i64>) -> u64 {
    dims.into_iter()
        .try_fold(1u64, |acc, dim| {
            u64::try_from(dim).ok().and_then(|dim| acc.checked_mul(dim))
        })
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ModuleBuilder;
    use crate::dim_numbers::DotDimensionNumbers;
    use crate::dtype::DType;
    use crate::ops::{DotAlgorithm, Literal};
    use crate::shape::Shape;

    fn analyze(
        build: impl FnOnce(&mut ModuleBuilder) -> Vec<ValueId>,
    ) -> (crate::ir::Module, FuncFacts) {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("main");
        let results = build(&mut builder);
        let main = builder.end_func(results);
        let module = builder.finish(main).unwrap();
        let facts = FuncFacts::analyze(module.entry_func());
        (module, facts)
    }

    #[test]
    fn use_counts_and_liveness_match_a_hand_computed_dag() {
        // y = x * x (x used twice), plus a dead z = x + x.
        let (module, facts) = analyze(|f| {
            let x = f.param(Shape::new(DType::F32, [4]));
            let y = f.multiply(x, x);
            let _z = f.add(x, x);
            vec![y]
        });
        let func = module.entry_func();
        let x = func.params[0];
        let y = func.results[0];
        // x feeds two operands of the product and two of the dead sum.
        assert_eq!(facts.value(x).uses, 4);
        // y is a function result: one use.
        assert_eq!(facts.value(y).uses, 1);
        assert!(facts.value(x).live);
        assert!(facts.value(y).live);
        // The multiply is node 0 (live), the add is node 1 (dead).
        assert!(facts.node(0).live);
        assert!(!facts.node(1).live);
    }

    #[test]
    fn zero_constant_propagates_through_product_and_shape_ops() {
        let (module, facts) = analyze(|f| {
            let x = f.param(Shape::new(DType::F32, [4]));
            let zero = f.constant(Literal::zeros(Shape::new(DType::F32, [4])).unwrap());
            let product = f.multiply(x, zero);
            let broadcast = f.broadcast_in_dim(product, &[2, 4], &[1]);
            let one = f.constant(Literal::new(
                Shape::new(DType::F32, [4]),
                1.0f32.to_le_bytes().repeat(4),
            ));
            let nonzero = f.multiply(x, one);
            vec![broadcast, nonzero]
        });
        let func = module.entry_func();
        let zero = func.nodes[0].results[0];
        let product = func.nodes[1].results[0];
        let broadcast = func.nodes[2].results[0];
        let one = func.nodes[3].results[0];
        let nonzero = func.nodes[4].results[0];
        assert!(facts.value(zero).exact_zero);
        assert!(facts.value(product).exact_zero, "x * 0 is exact zero");
        assert!(
            facts.value(broadcast).exact_zero,
            "broadcast preserves zero"
        );
        assert!(!facts.value(one).exact_zero, "1.0 constant is not zero");
        assert!(
            !facts.value(nonzero).exact_zero,
            "x * 1 is not provably zero"
        );
    }

    #[test]
    fn statement_estimate_grows_with_structural_size() {
        // A reduce over a 4x8 tensor costs 4*8 (output 4) scalar adds vs an
        // elementwise op's one statement per element; a reshape costs nothing.
        let (module, facts) = analyze(|f| {
            let x = f.param(Shape::new(DType::F32, [4, 8]));
            let doubled = f.multiply(x, x);
            let init = f.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());
            let reducer = f.func("add", |g| {
                let a = g.param(Shape::scalar(DType::F32));
                let b = g.param(Shape::scalar(DType::F32));
                vec![g.add(a, b)]
            });
            let summed = f.reduce(doubled, init, &[1], reducer);
            vec![summed]
        });
        let func = module.entry_func();
        let doubled = func.nodes[0].results[0];
        let summed = func.results[0];
        assert_eq!(
            facts.value(doubled).statements,
            32,
            "4*8 pointwise products"
        );
        assert_eq!(
            facts.value(summed).statements,
            32,
            "4 outputs times 8 reduced"
        );
    }

    #[test]
    fn dot_statement_estimate_counts_the_contracted_extent() {
        let (module, facts) = analyze(|f| {
            let a = f.param(Shape::new(DType::F32, [3, 5]));
            let b = f.param(Shape::new(DType::F32, [5, 7]));
            let c = f.dot_general(
                a,
                b,
                DotDimensionNumbers::matmul(),
                DotAlgorithm::tf32_tf32_f32(),
            );
            vec![c]
        });
        let func = module.entry_func();
        let c = func.results[0];
        // 3*7 outputs each summing over the contracted extent 5.
        assert_eq!(facts.value(c).statements, 3 * 7 * 5);
    }

    #[test]
    fn flops_count_arithmetic_elements_and_skip_data_movement() {
        let (module, _) = analyze(|f| {
            let x = f.param(Shape::new(DType::F32, [4]));
            let doubled = f.multiply(x, x);
            let moved = f.transpose(doubled, &[0]);
            let compared = f.maximum(moved, x);
            vec![f.add(compared, x)]
        });
        // Two arithmetic ops over 4 elements; transpose and maximum are free.
        assert_eq!(flops(&module).unwrap(), 8);
    }

    #[test]
    fn reduce_reducer_operates_on_plain_scalars() {
        use domain_index::FiniteMapGraph;

        let mut graph = FiniteMapGraph::default();
        let domain = graph.add_set(4).unwrap();
        let mut builder = ModuleBuilder::with_finite_map_graph(graph);
        builder.begin_func("reducer");
        let a = builder.param(Shape::scalar(DType::F32));
        let b = builder.param(Shape::scalar(DType::F32));
        let sum = builder.add(a, b);
        let reducer = builder.end_func(vec![sum]);
        builder.begin_func("main");
        let x = builder.param(Shape::new(DType::F32, [4]).with_domain_axes([domain]));
        let zero = builder.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());
        let reduced = builder.reduce(x, zero, &[0], reducer);
        let main = builder.end_func(vec![reduced]);
        builder
            .finish(main)
            .expect("domain annotations do not enter a scalar reducer");
    }

    #[test]
    fn flops_charge_a_reduction_its_reducer_per_reduced_element() {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("reducer");
        let a = builder.param(Shape::new(DType::F32, []));
        let b = builder.param(Shape::new(DType::F32, []));
        let sum = builder.add(a, b);
        let reducer = builder.end_func(vec![sum]);
        builder.begin_func("main");
        let x = builder.param(Shape::new(DType::F32, [4, 8]));
        let zero = builder.constant(Literal::zeros(Shape::new(DType::F32, [])).unwrap());
        let reduced = builder.reduce(x, zero, &[1], reducer);
        let main = builder.end_func(vec![reduced]);
        let module = builder.finish(main).unwrap();
        assert_eq!(flops(&module).unwrap(), 32);
    }

    #[test]
    fn flops_charge_a_dot_a_multiply_and_an_add_per_contracted_term() {
        let (module, _) = analyze(|f| {
            let a = f.param(Shape::new(DType::F32, [3, 5]));
            let b = f.param(Shape::new(DType::F32, [5, 7]));
            vec![f.dot_general(
                a,
                b,
                DotDimensionNumbers::matmul(),
                DotAlgorithm::tf32_tf32_f32(),
            )]
        });
        assert_eq!(flops(&module).unwrap(), 2 * 3 * 7 * 5);
    }

    #[test]
    fn analysis_is_a_pure_function() {
        let (module, first) = analyze(|f| {
            let x = f.param(Shape::new(DType::F32, [4]));
            let zero = f.constant(Literal::zeros(Shape::new(DType::F32, [4])).unwrap());
            let product = f.multiply(x, zero);
            vec![f.add(product, x)]
        });
        let second = FuncFacts::analyze(module.entry_func());
        assert_eq!(first.values(), second.values());
        assert_eq!(first.nodes(), second.nodes());
    }
}
