// SPDX-License-Identifier: Apache-2.0

//! Budgeted shared-DAG choice among exact Rust-built polynomial schedules.

use std::collections::HashMap;

use super::super::cost::PolynomialOperation;
use super::{Bounded, Request, ScheduleBudget, Settings, TermTree};
use crate::ir::ValueId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum DagNode {
    Factor(ValueId),
    Coefficient(usize),
    Mul(u32, u32),
    Add(u32, u32),
}

impl DagNode {
    fn cost(self, settings: &Settings) -> u64 {
        match self {
            Self::Factor(_) | Self::Coefficient(_) => 0,
            Self::Add(..) => settings.operation_weights.cost(PolynomialOperation::Add),
            Self::Mul(..) => settings
                .operation_weights
                .cost(PolynomialOperation::Multiply),
        }
    }

    fn push_children(self, children: &mut Vec<u32>) {
        match self {
            Self::Mul(left, right) | Self::Add(left, right) => children.extend([left, right]),
            Self::Factor(_) | Self::Coefficient(_) => {}
        }
    }
}

struct ExtractionBudget<'work, 'charge> {
    remaining: u64,
    work: &'work mut ScheduleBudget<'charge>,
}

impl ExtractionBudget<'_, '_> {
    fn reserve(&mut self, cost: usize) -> Bounded<()> {
        let stage_cost = u64::try_from(cost).map_err(|_| ())?;
        if stage_cost > self.remaining {
            return Err(());
        }
        self.work.reserve(cost)?;
        self.remaining -= stage_cost;
        Ok(())
    }
}

#[derive(Default)]
struct TreeDag {
    nodes: Vec<DagNode>,
    ids: HashMap<DagNode, u32>,
}

impl TreeDag {
    fn intern(&mut self, tree: &TermTree, budget: &mut ExtractionBudget<'_, '_>) -> Bounded<u32> {
        budget.reserve(1)?;
        let node = match tree {
            TermTree::Factor(value) => DagNode::Factor(*value),
            TermTree::Coefficient(id) => DagNode::Coefficient(*id),
            TermTree::Mul(left, right) => {
                DagNode::Mul(self.intern(left, budget)?, self.intern(right, budget)?)
            }
            TermTree::Add(left, right) => {
                DagNode::Add(self.intern(left, budget)?, self.intern(right, budget)?)
            }
        };
        if let Some(&id) = self.ids.get(&node) {
            return Ok(id);
        }
        let id = u32::try_from(self.nodes.len()).map_err(|_| ())?;
        self.nodes.push(node);
        self.ids.insert(node, id);
        Ok(id)
    }

    fn closure(
        &self,
        root: u32,
        budget: &mut ExtractionBudget<'_, '_>,
        mark: &mut [u32],
        epoch: u32,
    ) -> Bounded<Vec<u32>> {
        budget.reserve(1)?;
        let mut stack = vec![root];
        let mut closure = Vec::new();
        while let Some(node) = stack.pop() {
            budget.reserve(1)?;
            if mark[node as usize] == epoch {
                continue;
            }
            mark[node as usize] = epoch;
            budget.reserve(2)?;
            closure.push(node);
            self.nodes[node as usize].push_children(&mut stack);
        }
        Ok(closure)
    }
}

fn clone_tree_bounded(tree: &TermTree, budget: &mut ExtractionBudget<'_, '_>) -> Bounded<TermTree> {
    budget.reserve(1)?;
    Ok(match tree {
        TermTree::Factor(value) => TermTree::Factor(*value),
        TermTree::Coefficient(id) => TermTree::Coefficient(*id),
        TermTree::Mul(left, right) => TermTree::Mul(
            Box::new(clone_tree_bounded(left, budget)?),
            Box::new(clone_tree_bounded(right, budget)?),
        ),
        TermTree::Add(left, right) => TermTree::Add(
            Box::new(clone_tree_bounded(left, budget)?),
            Box::new(clone_tree_bounded(right, budget)?),
        ),
    })
}

fn balance_tree(tree: TermTree, budget: &mut ExtractionBudget<'_, '_>) -> Bounded<TermTree> {
    budget.reserve(1)?;
    Ok(match tree {
        TermTree::Add(left, right) => {
            let mut terms = Vec::new();
            collect_add_terms(*left, budget, &mut terms)?;
            collect_add_terms(*right, budget, &mut terms)?;
            balance_terms(terms, budget, |left, right| {
                TermTree::Add(Box::new(left), Box::new(right))
            })?
        }
        TermTree::Mul(left, right) => {
            let mut factors = Vec::new();
            collect_multiply_factors(*left, budget, &mut factors)?;
            collect_multiply_factors(*right, budget, &mut factors)?;
            balance_terms(factors, budget, |left, right| {
                TermTree::Mul(Box::new(left), Box::new(right))
            })?
        }
        leaf @ (TermTree::Factor(_) | TermTree::Coefficient(_)) => leaf,
    })
}

fn collect_add_terms(
    tree: TermTree,
    budget: &mut ExtractionBudget<'_, '_>,
    terms: &mut Vec<TermTree>,
) -> Bounded<()> {
    budget.reserve(1)?;
    match tree {
        TermTree::Add(left, right) => {
            collect_add_terms(*left, budget, terms)?;
            collect_add_terms(*right, budget, terms)
        }
        tree => {
            terms.push(balance_tree(tree, budget)?);
            Ok(())
        }
    }
}

fn collect_multiply_factors(
    tree: TermTree,
    budget: &mut ExtractionBudget<'_, '_>,
    factors: &mut Vec<TermTree>,
) -> Bounded<()> {
    budget.reserve(1)?;
    match tree {
        TermTree::Mul(left, right) => {
            collect_multiply_factors(*left, budget, factors)?;
            collect_multiply_factors(*right, budget, factors)
        }
        tree => {
            factors.push(balance_tree(tree, budget)?);
            Ok(())
        }
    }
}

fn balance_terms(
    mut terms: Vec<TermTree>,
    budget: &mut ExtractionBudget<'_, '_>,
    combine: impl Fn(TermTree, TermTree) -> TermTree + Copy,
) -> Bounded<TermTree> {
    budget.reserve(terms.len())?;
    while terms.len() > 1 {
        let mut next = Vec::with_capacity(terms.len().div_ceil(2));
        let mut items = terms.into_iter();
        while let Some(left) = items.next() {
            next.push(match items.next() {
                Some(right) => combine(left, right),
                None => left,
            });
        }
        budget.reserve(next.len())?;
        terms = next;
    }
    terms.pop().ok_or(())
}

/// Choose among exact Rust-built variants under a shared DAG cost. Every tree
/// walk spends a stage-local allowance before its result can affect a choice.
pub(super) fn extract_variants(
    requests: &[Request],
    settings: &Settings,
    balance_associative_trees: bool,
    work: &mut ScheduleBudget<'_>,
) -> Bounded<Vec<Option<TermTree>>> {
    let (dag, closures, extraction_remaining) = {
        let mut extraction = ExtractionBudget {
            remaining: settings.extract_step_budget,
            work,
        };
        let mut dag = TreeDag::default();
        let mut roots = Vec::with_capacity(requests.len());
        for request in requests {
            let mut candidates = Vec::with_capacity(request.variants.len() + 1);
            candidates.push(dag.intern(&request.flat, &mut extraction)?);
            for variant in &request.variants {
                candidates.push(dag.intern(variant, &mut extraction)?);
            }
            roots.push(candidates);
        }
        extraction.reserve(dag.nodes.len())?;
        let mut mark = vec![0u32; dag.nodes.len()];
        let mut epoch = 0u32;
        let mut closures = Vec::with_capacity(roots.len());
        for candidates in &roots {
            let mut request_closures = Vec::with_capacity(candidates.len());
            for &root in candidates {
                epoch = epoch.checked_add(1).ok_or(())?;
                request_closures.push(dag.closure(root, &mut extraction, &mut mark, epoch)?);
            }
            closures.push(request_closures);
        }
        extraction.reserve(closures.len())?;
        let candidate_count = closures.iter().fold(0usize, |total, candidates| {
            total.saturating_add(candidates.len())
        });
        extraction.reserve(candidate_count)?;
        let closure_nodes = closures
            .iter()
            .flatten()
            .fold(0usize, |total, closure| total.saturating_add(closure.len()));
        extraction.reserve(
            closure_nodes
                .saturating_mul(2)
                .saturating_add(dag.nodes.len()),
        )?;
        (dag, closures, extraction.remaining)
    };
    let closure_cost = |closure: &[u32]| {
        closure.iter().fold(0u64, |cost, &node| {
            cost.saturating_add(dag.nodes[node as usize].cost(settings))
        })
    };
    let mut chosen = closures
        .iter()
        .map(|candidates| {
            candidates
                .iter()
                .enumerate()
                .min_by_key(|(_, closure)| closure_cost(closure))
                .map(|(index, _)| index)
                .expect("a request has its flat candidate")
        })
        .collect::<Vec<_>>();
    let mut counts = vec![0u32; dag.nodes.len()];
    for (candidates, &choice) in closures.iter().zip(&chosen) {
        for &node in &candidates[choice] {
            counts[node as usize] += 1;
        }
    }

    {
        let mut refinement = ExtractionBudget {
            remaining: settings.refine_step_budget,
            work,
        };
        for _ in 0..settings.refine_rounds {
            refinement.reserve(1)?;
            let mut changed = false;
            for (request, candidates) in closures.iter().enumerate() {
                refinement.reserve(candidates[chosen[request]].len())?;
                for &node in &candidates[chosen[request]] {
                    counts[node as usize] -= 1;
                }
                let mut best = chosen[request];
                let mut best_cost = u64::MAX;
                for (candidate, closure) in candidates.iter().enumerate() {
                    refinement.reserve(closure.len().saturating_add(1))?;
                    let cost = closure.iter().fold(0u64, |cost, &node| {
                        if counts[node as usize] == 0 {
                            cost.saturating_add(dag.nodes[node as usize].cost(settings))
                        } else {
                            cost
                        }
                    });
                    if cost < best_cost {
                        best = candidate;
                        best_cost = cost;
                    }
                }
                changed |= best != chosen[request];
                chosen[request] = best;
                refinement.reserve(candidates[best].len())?;
                for &node in &candidates[best] {
                    counts[node as usize] += 1;
                }
            }
            if !changed {
                break;
            }
        }
    }
    let mut extraction = ExtractionBudget {
        remaining: extraction_remaining,
        work,
    };
    extraction.reserve(requests.len())?;
    requests
        .iter()
        .zip(chosen)
        .zip(&closures)
        .map(|((request, choice), _)| {
            if choice == 0 {
                Ok(None)
            } else {
                let tree = clone_tree_bounded(&request.variants[choice - 1], &mut extraction)?;
                if balance_associative_trees {
                    balance_tree(tree, &mut extraction).map(Some)
                } else {
                    Ok(Some(tree))
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn left_add_chain(count: u32) -> TermTree {
        (1..count).fold(TermTree::Factor(ValueId(0)), |tree, value| {
            TermTree::Add(Box::new(tree), Box::new(TermTree::Factor(ValueId(value))))
        })
    }

    fn depth(tree: &TermTree) -> usize {
        match tree {
            TermTree::Mul(left, right) | TermTree::Add(left, right) => {
                1 + depth(left).max(depth(right))
            }
            TermTree::Factor(_) | TermTree::Coefficient(_) => 1,
        }
    }

    #[test]
    fn balances_an_associative_chain_without_extra_operations() {
        let tree = left_add_chain(32);
        let mut work = ScheduleBudget {
            remaining: 1 << 16,
            shared: None,
        };
        let mut extraction = ExtractionBudget {
            remaining: 1 << 16,
            work: &mut work,
        };
        let balanced = balance_tree(tree, &mut extraction).unwrap();
        assert_eq!(depth(&balanced), 6);

        let mut nodes = Vec::new();
        let mut stack = vec![&balanced];
        while let Some(tree) = stack.pop() {
            match tree {
                TermTree::Add(left, right) => {
                    nodes.push(tree);
                    stack.extend([left.as_ref(), right.as_ref()]);
                }
                TermTree::Factor(_) => {}
                _ => panic!("an Add chain balances to Adds and its factors"),
            }
        }
        assert_eq!(nodes.len(), 31);
    }
}
