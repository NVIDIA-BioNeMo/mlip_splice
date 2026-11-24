// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded distribution of a product over a reducer, inside the recognizer.
//!
//! The extractor that chose this program's shape minimizes node count, so it
//! *factors*: it prefers `a*(b + c)` to `a*b + a*c`. A map/reduce region is the
//! opposite form. Both are the same e-class, but only the distributed one has a
//! reduction wide enough to walk with a loop, and only the factored one reached
//! XIR. Reading the DAG as written can therefore never find the region — the
//! rewrite that reveals it is exactly the one the extractor ran backwards.
//!
//! So the recognizer distributes for itself. It does not rewrite the scalar
//! program: the expansion lives here, is used to choose a batching, and is
//! discarded. What comes out is the distributed term list; what stays in XIR is
//! either the recovered map/reduce or the original factored DAG, untouched.
//!
//! The only algebra involved is a declared law — `⊗` distributes over `⊕` — and
//! the fact that `⊗` is associative and commutative, so a term is the multiset
//! of its factors. Nothing here knows about polynomials, degrees, or monomials;
//! swapping the law table changes which operations expand.

use std::collections::{HashMap, HashSet};

use crate::ir::{Func, ValueId};
use crate::ops::ScalarOp;

use super::infer_shapes::producer;
use super::scalar_op;

/// One distributed term: the factors combined by `⊗`. Factors are values the
/// expansion stopped at, so a term is a flat product of opaque atoms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Term {
    pub factors: Vec<ValueId>,
    /// Unary operations wrapping the whole term, innermost first. Each is
    /// declared to commute with the reducer, so pushing it down onto the terms
    /// is what lets an expansion continue through it instead of stopping.
    pub wrappers: Vec<ScalarOp>,
}

/// A unary operation that commutes with the reducer: `f(a ⊕ b) = f(a) ⊕ f(b)`.
/// Without this a negation seals everything below it into a single atom, which
/// in a gradient program truncates almost every expansion.
pub(super) fn commutes_with(unary: ScalarOp, reducer: ScalarOp) -> bool {
    matches!((unary, reducer), (ScalarOp::Negate, ScalarOp::Add))
}

/// The reducer an operation contributes to when its operands enter with opposite
/// signs. `a - b` is `a + (-b)`, which is the only way a subtraction can join an
/// additive expansion.
fn inverse_of(scalar: ScalarOp, reducer: ScalarOp) -> Option<ScalarOp> {
    matches!((scalar, reducer), (ScalarOp::Subtract, ScalarOp::Add)).then_some(ScalarOp::Negate)
}

/// The declared law: `product` distributes over `reducer`, and both are
/// associative and commutative. Widening this table is all another algebra
/// needs — the expansion below reads nothing else about the operations.
pub(super) fn distributes_over(product: ScalarOp, reducer: ScalarOp) -> bool {
    matches!(
        (product, reducer),
        (ScalarOp::Multiply, ScalarOp::Add)
            | (ScalarOp::And, ScalarOp::Or)
            | (ScalarOp::Or, ScalarOp::And)
            | (ScalarOp::Add, ScalarOp::Maximum)
            | (ScalarOp::Add, ScalarOp::Minimum)
    )
}

/// Which operations an expansion under `reducer` may descend through.
fn role(func: &Func, value: ValueId, region: &[bool], reducer: ScalarOp) -> Option<ScalarOp> {
    if !region[value.index()] {
        return None;
    }
    let scalar = producer(func, value)?.op.as_elementwise()?.scalar();
    (scalar == reducer
        || distributes_over(scalar, reducer)
        || commutes_with(scalar, reducer)
        || inverse_of(scalar, reducer).is_some())
    .then_some(scalar)
}

pub(super) struct Expansion {
    memo: HashMap<ValueId, Option<Vec<Term>>>,
    reducer: ScalarOp,
    limit: usize,
}

impl Expansion {
    pub(super) fn new(reducer: ScalarOp, limit: usize) -> Self {
        Self {
            memo: HashMap::new(),
            reducer,
            limit,
        }
    }

    /// The distributed term list of `value`, or `None` if it would exceed the
    /// limit. Memoized per value: a shared subexpression expands once, and every
    /// reader of it copies the same list rather than re-deriving it.
    ///
    /// Sharing is why the bound is on the *result* rather than the recursion.
    /// A DAG whose every level doubles has an exponential expansion, and the
    /// limit is what turns that into a refusal instead of an out-of-memory.
    pub(super) fn terms(
        &mut self,
        func: &Func,
        value: ValueId,
        region: &[bool],
    ) -> Option<Vec<Term>> {
        if let Some(cached) = self.memo.get(&value) {
            return cached.clone();
        }
        // Guard against a cyclic walk; XIR is acyclic, so this only bounds the
        // recursion while the entry is being computed.
        self.memo.insert(value, None);
        let expanded = self.expand(func, value, region);
        self.memo.insert(value, expanded.clone());
        expanded
    }

    fn expand(&mut self, func: &Func, value: ValueId, region: &[bool]) -> Option<Vec<Term>> {
        let Some(scalar) = role(func, value, region, self.reducer) else {
            return Some(vec![Term {
                factors: vec![value],
                wrappers: Vec::new(),
            }]);
        };
        let node = producer(func, value)?;
        let inputs = node.inputs.clone();
        if commutes_with(scalar, self.reducer) {
            let mut terms = self.terms(func, inputs[0], region)?;
            for term in &mut terms {
                term.wrappers.push(scalar);
            }
            return Some(terms);
        }
        if let Some(inverse) = inverse_of(scalar, self.reducer) {
            let mut terms = self.terms(func, inputs[0], region)?;
            let mut negated = self.terms(func, inputs[1], region)?;
            if terms.len().checked_add(negated.len())? > self.limit {
                return None;
            }
            for term in &mut negated {
                term.wrappers.push(inverse);
            }
            terms.append(&mut negated);
            return Some(terms);
        }
        if scalar == self.reducer {
            let mut terms = Vec::new();
            for input in inputs {
                let mut expanded = self.terms(func, input, region)?;
                if terms.len().checked_add(expanded.len())? > self.limit {
                    return None;
                }
                terms.append(&mut expanded);
            }
            return Some(terms);
        }
        // `scalar` distributes over the reducer: the product of two sums is the
        // sum of the pairwise products, and a term's factors simply concatenate.
        let mut terms = vec![Term {
            factors: Vec::new(),
            wrappers: Vec::new(),
        }];
        for input in inputs {
            let expanded = self.terms(func, input, region)?;
            if terms.len().checked_mul(expanded.len())? > self.limit {
                return None;
            }
            let mut product = Vec::with_capacity(terms.len() * expanded.len());
            for left in &terms {
                for right in &expanded {
                    let mut factors = left.factors.clone();
                    factors.extend(right.factors.iter().copied());
                    let mut wrappers = left.wrappers.clone();
                    wrappers.extend(right.wrappers.iter().copied());
                    product.push(Term { factors, wrappers });
                }
            }
            terms = product;
        }
        Some(terms)
    }
}

/// Whether a value tops a distributable region: nothing above it would have
/// absorbed it into a larger expansion, so it is where recovery should start.
pub(super) fn is_expansion_root(
    func: &Func,
    value: ValueId,
    region: &[bool],
    reducer: ScalarOp,
    uses: &[usize],
    consumer: &[usize],
) -> bool {
    if role(func, value, region, reducer).is_none() {
        return false;
    }
    if uses[value.index()] != 1 || consumer[value.index()] == usize::MAX {
        return true;
    }
    // A single consumer that would itself expand through this value makes it
    // interior, not a root.
    func.nodes[consumer[value.index()]]
        .op
        .as_elementwise()
        .map(|op| op.scalar())
        .is_none_or(|parent| parent != reducer && !distributes_over(parent, reducer))
}

#[cfg(test)]
mod tests;

/// The outermost reductions strictly inside `root`: where recovery should look
/// next when `root` itself could not be paid for. Descends only through the
/// operations an expansion would have descended through, so it hands back the
/// reductions that distribution dissolved rather than arbitrary sub-values.
pub(super) fn interior_reductions(
    func: &Func,
    root: ValueId,
    region: &[bool],
    reducer: ScalarOp,
) -> Vec<ValueId> {
    let mut found = Vec::new();
    let mut stack = producer(func, root)
        .map(|node| node.inputs.iter().copied().collect::<Vec<_>>())
        .unwrap_or_default();
    while let Some(value) = stack.pop() {
        let Some(scalar) = role(func, value, region, reducer) else {
            continue;
        };
        if scalar == reducer {
            found.push(value);
            continue;
        }
        if let Some(node) = producer(func, value) {
            stack.extend(node.inputs.iter().copied());
        }
    }
    found
}

/// The operation whose distribution built this region's terms. A root that only
/// applies the reducer still needs one to combine a term's atoms; any product
/// that distributes will do, and every term will then hold a single atom.
pub(super) fn distributed_product(
    func: &Func,
    root: ValueId,
    reducer: ScalarOp,
) -> Option<ScalarOp> {
    let scalar = producer(func, root)?.op.as_elementwise()?.scalar();
    if distributes_over(scalar, reducer) && scalar_op(scalar).is_some() {
        return Some(scalar);
    }
    if scalar != reducer {
        return None;
    }
    [
        ScalarOp::Multiply,
        ScalarOp::And,
        ScalarOp::Or,
        ScalarOp::Add,
    ]
    .into_iter()
    .find(|&product| distributes_over(product, reducer) && scalar_op(product).is_some())
}

pub(super) fn use_counts(func: &Func) -> Vec<usize> {
    let mut uses = vec![0usize; func.values.len()];
    for node in &func.nodes {
        for &input in &node.inputs {
            uses[input.index()] += 1;
        }
    }
    for &result in &func.results {
        uses[result.index()] += 1;
    }
    uses
}

pub(super) fn sole_consumers(func: &Func) -> Vec<usize> {
    let mut consumer = vec![usize::MAX; func.values.len()];
    for (index, node) in func.nodes.iter().enumerate() {
        for &input in &node.inputs {
            consumer[input.index()] = index;
        }
    }
    consumer
}

/// Every value an elementwise scalar operation produces. Anything outside it —
/// a parameter slice, a literal, a tensor operation — is a leaf a batch indexes.
pub(super) fn scalar_region(func: &Func) -> Vec<bool> {
    let mut region = vec![false; func.values.len()];
    for node in &func.nodes {
        let [result] = node.results.as_slice() else {
            continue;
        };
        if func.value_shape(*result).is_scalar()
            && node.op.as_elementwise().is_some()
            && node
                .inputs
                .iter()
                .all(|&input| func.value_shape(input).is_scalar())
        {
            region[result.index()] = true;
        }
    }
    region
}

pub(super) fn region_leaves(func: &Func, region: &[bool]) -> Vec<ValueId> {
    let mut leaves = HashSet::new();
    for node in &func.nodes {
        let [result] = node.results.as_slice() else {
            continue;
        };
        if region[result.index()] {
            leaves.extend(
                node.inputs
                    .iter()
                    .copied()
                    .filter(|input| !region[input.index()]),
            );
        }
    }
    let mut leaves = leaves.into_iter().collect::<Vec<_>>();
    leaves.sort_unstable_by_key(|value| value.index());
    leaves
}
