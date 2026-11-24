// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Budgeted exact-polynomial scheduling. Rust constructs factorizations and
//! chooses among them under a shared DAG cost.

use std::collections::HashMap;

use algebraic_real::AlgebraicReal;

use super::arithmetic::ExactMonomial;
use super::cost::PolynomialOperationWeights;
use crate::ir::ValueId;

mod select_dag;

#[derive(Clone, Copy)]
pub(super) struct Settings {
    pub(super) operation_weights: PolynomialOperationWeights,
    pub(super) balance_associative_trees: bool,
    pub(super) max_island_terms: usize,
    pub(super) max_greedy_variants: usize,
    pub(super) max_factors_per_term: usize,
    pub(super) max_tree_depth: usize,
    pub(super) extract_step_budget: u64,
    pub(super) refine_rounds: usize,
    pub(super) refine_step_budget: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum TermTree {
    Factor(ValueId),
    Coefficient(usize),
    Mul(Box<TermTree>, Box<TermTree>),
    Add(Box<TermTree>, Box<TermTree>),
}

/// One schedule per island, sharing one coefficient table. `None` keeps the
/// island's flat materialization.
pub(super) struct BatchSchedule {
    pub(super) trees: Vec<Option<TermTree>>,
    pub(super) coefficients: Vec<Vec<AlgebraicReal>>,
}

#[derive(Clone)]
struct IslandTerm {
    factors: Vec<ValueId>,
    coefficient: Option<usize>,
}

#[derive(Default)]
struct Context {
    coefficients: Vec<Vec<AlgebraicReal>>,
    ids: HashMap<Vec<AlgebraicReal>, usize>,
    ones: HashMap<usize, usize>,
}

impl Context {
    fn intern(&mut self, coefficient: Vec<AlgebraicReal>) -> usize {
        if let Some(&id) = self.ids.get(&coefficient) {
            return id;
        }
        let id = self.coefficients.len();
        self.coefficients.push(coefficient.clone());
        self.ids.insert(coefficient, id);
        id
    }

    fn ones_id(&mut self, elements: usize) -> usize {
        if let Some(&id) = self.ones.get(&elements) {
            return id;
        }
        let id = self.intern(vec![AlgebraicReal::one(); elements]);
        self.ones.insert(elements, id);
        id
    }
}

/// A bounded scheduling request destined for the shared e-graph.
pub(super) struct Request {
    elements: usize,
    terms: Vec<IslandTerm>,
    pub(super) flat: TermTree,
    pub(super) variants: Vec<TermTree>,
}

/// Rust-side shape of one island before the e-graph runs: flat, a single
/// e-graph request, or an oversized island peeled into factor groups whose
/// quotients are themselves plans.
enum Plan {
    Flat {
        elements: usize,
        terms: Vec<IslandTerm>,
    },
    Request(usize),
    Split {
        elements: usize,
        parts: Vec<(ValueId, Plan)>,
        leftovers: Vec<IslandTerm>,
    },
}

struct ScheduleBudget<'a> {
    remaining: usize,
    shared: Option<&'a mut dyn FnMut(usize) -> bool>,
}

impl ScheduleBudget<'_> {
    fn reserve(&mut self, cost: usize) -> Result<(), ()> {
        if cost > self.remaining || self.shared.as_mut().is_some_and(|reserve| !reserve(cost)) {
            return Err(());
        }
        self.remaining -= cost;
        Ok(())
    }
}

type Bounded<T> = Result<T, ()>;

#[cfg(test)]
pub(super) fn schedule_with_work_limit(
    islands: &[&[ExactMonomial]],
    settings: &Settings,
    work_limit: usize,
) -> Result<Option<BatchSchedule>, String> {
    schedule_impl(
        islands,
        settings,
        ScheduleBudget {
            remaining: work_limit,
            shared: None,
        },
    )
}

pub(super) fn schedule_with_shared_budget(
    islands: &[&[ExactMonomial]],
    settings: &Settings,
    reserve: &mut impl FnMut(usize) -> bool,
) -> Result<Option<BatchSchedule>, String> {
    schedule_impl(
        islands,
        settings,
        ScheduleBudget {
            remaining: usize::MAX,
            shared: Some(reserve),
        },
    )
}

fn schedule_impl(
    islands: &[&[ExactMonomial]],
    settings: &Settings,
    mut work: ScheduleBudget<'_>,
) -> Result<Option<BatchSchedule>, String> {
    if work.reserve(islands.len().saturating_mul(2)).is_err() {
        return Ok(None);
    }
    let mut batch = BatchSchedule {
        trees: islands.iter().map(|_| None).collect(),
        coefficients: Vec::new(),
    };
    let one = AlgebraicReal::one();
    let mut context = Context::default();
    let mut requests = Vec::new();
    let mut plans = Vec::with_capacity(islands.len());
    for &terms in islands {
        if terms.len() < 2 {
            plans.push(None);
            continue;
        }
        let mut too_deep = false;
        for term in terms {
            if work.reserve(1).is_err() {
                return Ok(None);
            }
            too_deep |= term.factors().len() > settings.max_factors_per_term;
        }
        if too_deep {
            plans.push(None);
            continue;
        }
        if work.reserve(terms.len()).is_err() {
            return Ok(None);
        }
        let elements = terms[0].coefficient().len();
        let mut island = Vec::with_capacity(terms.len());
        for term in terms {
            if work
                .reserve(
                    term.factors()
                        .len()
                        .saturating_add(term.coefficient().len())
                        .saturating_add(1),
                )
                .is_err()
            {
                return Ok(None);
            }
            let mut coefficient_work = 0usize;
            for coefficient in term.coefficient() {
                if work.reserve(1).is_err() {
                    return Ok(None);
                }
                coefficient_work = coefficient_work
                    .saturating_add(coefficient.structural_size().max(1).saturating_mul(2));
            }
            if work.reserve(coefficient_work).is_err() {
                return Ok(None);
            }
            let coefficient = (!term.coefficient().iter().all(|value| value == &one))
                .then(|| context.intern(term.coefficient().to_vec()));
            island.push(IslandTerm {
                factors: term.factors().to_vec(),
                coefficient,
            });
        }
        let Ok(plan) = plan_terms(
            &mut context,
            &mut requests,
            elements,
            island,
            settings,
            &mut work,
        ) else {
            return Ok(None);
        };
        plans.push(Some(plan));
    }
    let mut extracted = if requests.is_empty() {
        Vec::new()
    } else {
        let Ok(extracted) = select_dag::extract_variants(
            &requests,
            settings,
            settings.balance_associative_trees,
            &mut work,
        ) else {
            return Ok(None);
        };
        extracted
    };
    for (slot, plan) in batch.trees.iter_mut().zip(&plans) {
        if let Some(plan) = plan {
            let Ok(tree) =
                assemble_island(&mut context, plan, &requests, &mut extracted, &mut work)
            else {
                return Ok(None);
            };
            *slot = if let Some(tree) = tree {
                let Ok(within_depth) =
                    tree_depth_within_bounded(&tree, settings.max_tree_depth, &mut work)
                else {
                    return Ok(None);
                };
                within_depth.then_some(tree)
            } else {
                None
            };
        }
    }
    batch.coefficients = context.coefficients;
    Ok(Some(batch))
}

fn tree_depth_within_bounded(
    tree: &TermTree,
    limit: usize,
    work: &mut ScheduleBudget<'_>,
) -> Bounded<bool> {
    let mut stack = vec![(tree, 1usize)];
    while let Some((tree, depth)) = stack.pop() {
        work.reserve(1)?;
        if depth > limit {
            return Ok(false);
        }
        match tree {
            TermTree::Mul(left, right) | TermTree::Add(left, right) => {
                stack.push((left, depth + 1));
                stack.push((right, depth + 1));
            }
            TermTree::Factor(_) | TermTree::Coefficient(_) => {}
        }
    }
    Ok(true)
}

/// Plan one term set: islands within the e-graph bound become requests;
/// larger ones peel factor groups greedily — the terms sharing the
/// most-supported remaining factor become `factor · quotient`, with the
/// quotient planned recursively — leaving the leftover terms to a balanced
/// Add combine at assembly.
fn plan_terms(
    context: &mut Context,
    requests: &mut Vec<Request>,
    elements: usize,
    terms: Vec<IslandTerm>,
    settings: &Settings,
    work: &mut ScheduleBudget<'_>,
) -> Bounded<Plan> {
    work.reserve(terms.len().saturating_add(1))?;
    if terms.len() <= settings.max_island_terms {
        let candidates = factor_candidates(&terms, work)?;
        if candidates.is_empty() {
            return Ok(Plan::Flat { elements, terms });
        }
        let flat = flat_chain(context, elements, &terms, work)?;
        if !tree_depth_within_bounded(&flat, settings.max_tree_depth, work)? {
            return Ok(Plan::Flat { elements, terms });
        }
        let mut variants = Vec::new();
        for pick in 0..candidates.len().min(settings.max_greedy_variants) {
            let tree = greedy_tree(context, elements, &terms, pick, work)?;
            if tree_depth_within_bounded(&tree, settings.max_tree_depth, work)? {
                variants.push(tree);
            }
        }
        let grouped = grouped_chain(context, elements, &terms, work)?;
        if grouped != flat && tree_depth_within_bounded(&grouped, settings.max_tree_depth, work)? {
            variants.push(grouped);
        }
        requests.push(Request {
            elements,
            terms,
            flat,
            variants,
        });
        return Ok(Plan::Request(requests.len() - 1));
    }
    let mut remaining = terms;
    let mut parts = Vec::new();
    loop {
        let candidates = factor_candidates(&remaining, work)?;
        let Some(&factor) = candidates.first() else {
            break;
        };
        let (pulled, rest) = pull_factor(&remaining, factor, work)?;
        let quotient = plan_terms(context, requests, elements, pulled, settings, work)?;
        parts.push((factor, quotient));
        remaining = rest;
    }
    if parts.is_empty() {
        return Ok(Plan::Flat {
            elements,
            terms: remaining,
        });
    }
    Ok(Plan::Split {
        elements,
        parts,
        leftovers: remaining,
    })
}

/// Best tree for the island, or `None` to keep the flat materialization.
fn assemble_island(
    context: &mut Context,
    plan: &Plan,
    requests: &[Request],
    extracted: &mut [Option<TermTree>],
    work: &mut ScheduleBudget<'_>,
) -> Bounded<Option<TermTree>> {
    Ok(match plan {
        Plan::Flat { .. } => None,
        Plan::Request(id) => extracted[*id].take(),
        Plan::Split { .. } => Some(assemble_quotient(context, requests, plan, extracted, work)?),
    })
}

/// Inside a split, quotients that stay flat materialize as a balanced Add of
/// their monomials rather than falling back to the island's flat form.
fn assemble_quotient(
    context: &mut Context,
    requests: &[Request],
    plan: &Plan,
    extracted: &mut [Option<TermTree>],
    work: &mut ScheduleBudget<'_>,
) -> Bounded<TermTree> {
    match plan {
        Plan::Flat { elements, terms } => {
            let mut items = Vec::with_capacity(terms.len());
            for term in terms {
                items.push(monomial(context, *elements, term, work)?);
            }
            balanced_add(items, work)
        }
        Plan::Request(id) => {
            if let Some(tree) = extracted[*id].take() {
                return Ok(tree);
            }
            let request = &requests[*id];
            let mut items = Vec::with_capacity(request.terms.len());
            for term in &request.terms {
                items.push(monomial(context, request.elements, term, work)?);
            }
            balanced_add(items, work)
        }
        Plan::Split {
            elements,
            parts,
            leftovers,
        } => {
            work.reserve(parts.len().saturating_add(leftovers.len()))?;
            let mut items = Vec::with_capacity(parts.len().saturating_add(leftovers.len()));
            for (factor, plan) in parts {
                let quotient = assemble_quotient(context, requests, plan, extracted, work)?;
                items.push(TermTree::Mul(
                    Box::new(TermTree::Factor(*factor)),
                    Box::new(quotient),
                ));
            }
            for term in leftovers {
                items.push(monomial(context, *elements, term, work)?);
            }
            balanced_add(items, work)
        }
    }
}

/// Balanced Add combine, keeping tree depth logarithmic for wide islands.
fn balanced_add(mut parts: Vec<TermTree>, work: &mut ScheduleBudget<'_>) -> Bounded<TermTree> {
    work.reserve(parts.len().saturating_mul(2))?;
    while parts.len() > 1 {
        let mut next = Vec::with_capacity(parts.len().div_ceil(2));
        let mut items = parts.into_iter();
        while let Some(left) = items.next() {
            next.push(match items.next() {
                Some(right) => TermTree::Add(Box::new(left), Box::new(right)),
                None => left,
            });
        }
        parts = next;
    }
    Ok(parts.pop().expect("balanced_add over at least one part"))
}

/// Split terms into the quotients of those containing `factor` and the rest.
fn pull_factor(
    terms: &[IslandTerm],
    factor: ValueId,
    work: &mut ScheduleBudget<'_>,
) -> Bounded<(Vec<IslandTerm>, Vec<IslandTerm>)> {
    work.reserve(terms.len().saturating_mul(2))?;
    let mut pulled = Vec::new();
    let mut rest = Vec::new();
    for term in terms {
        if let Some(position) = term
            .factors
            .iter()
            .position(|&candidate| candidate == factor)
        {
            let mut factors = term.factors.clone();
            factors.remove(position);
            pulled.push(IslandTerm {
                factors,
                coefficient: term.coefficient,
            });
        } else {
            rest.push(term.clone());
        }
    }
    Ok((pulled, rest))
}

/// Factors appearing in at least two terms, most shared first.
fn factor_candidates(terms: &[IslandTerm], work: &mut ScheduleBudget<'_>) -> Bounded<Vec<ValueId>> {
    work.reserve(terms.len())?;
    let mut support: HashMap<ValueId, usize> = HashMap::new();
    for term in terms {
        let mut factors = term.factors.clone();
        factors.dedup();
        for factor in factors {
            *support.entry(factor).or_default() += 1;
        }
    }
    let mut candidates: Vec<(ValueId, usize)> = support
        .into_iter()
        .filter(|&(_, support)| support >= 2)
        .collect();
    let ordering = candidates.len().checked_ilog2().unwrap_or(0) as usize + 1;
    work.reserve(candidates.len().saturating_mul(ordering))?;
    candidates.sort_unstable_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then(left.0.index().cmp(&right.0.index()))
    });
    Ok(candidates.into_iter().map(|(factor, _)| factor).collect())
}

/// Recursively pull the ranked shared factor out of the terms containing it:
/// `factor·(quotient) + rest`. `pick` selects the top-level factor so each
/// variant explores a different first split; recursion always takes the best.
fn greedy_tree(
    context: &mut Context,
    elements: usize,
    terms: &[IslandTerm],
    pick: usize,
    work: &mut ScheduleBudget<'_>,
) -> Bounded<TermTree> {
    work.reserve(1)?;
    if terms.len() == 1 {
        return monomial(context, elements, &terms[0], work);
    }
    let candidates = factor_candidates(terms, work)?;
    let Some(&factor) = candidates.get(pick).or_else(|| candidates.first()) else {
        return flat_chain(context, elements, terms, work);
    };
    let (pulled, rest) = pull_factor(terms, factor, work)?;
    let quotient = greedy_tree(context, elements, &pulled, 0, work)?;
    let group = TermTree::Mul(Box::new(TermTree::Factor(factor)), Box::new(quotient));
    if rest.is_empty() {
        Ok(group)
    } else {
        Ok(TermTree::Add(
            Box::new(group),
            Box::new(greedy_tree(context, elements, &rest, 0, work)?),
        ))
    }
}

/// Directed head-factoring closed over the sorted monomials: each maximal
/// run sharing a leading factor becomes `factor · (grouped residual)`, the
/// residual grouped recursively, all combined in a right-leaning chain.
fn grouped_chain(
    context: &mut Context,
    elements: usize,
    terms: &[IslandTerm],
    work: &mut ScheduleBudget<'_>,
) -> Bounded<TermTree> {
    let ordering = terms.len().checked_ilog2().unwrap_or(0) as usize + 1;
    work.reserve(
        terms
            .len()
            .saturating_mul(ordering)
            .saturating_add(terms.len()),
    )?;
    let mut sorted: Vec<&IslandTerm> = terms.iter().collect();
    sorted.sort_by(|left, right| {
        left.factors
            .iter()
            .map(|value| value.index())
            .cmp(right.factors.iter().map(|value| value.index()))
    });
    let mut items = Vec::new();
    let mut start = 0;
    while start < sorted.len() {
        let head = sorted[start].factors.first().copied();
        let mut end = start + 1;
        while head.is_some() && end < sorted.len() && sorted[end].factors.first().copied() == head {
            end += 1;
        }
        match head {
            Some(head) if end - start >= 2 => {
                work.reserve(end - start)?;
                let residual: Vec<IslandTerm> = sorted[start..end]
                    .iter()
                    .map(|term| IslandTerm {
                        factors: term.factors[1..].to_vec(),
                        coefficient: term.coefficient,
                    })
                    .collect();
                items.push(TermTree::Mul(
                    Box::new(TermTree::Factor(head)),
                    Box::new(grouped_chain(context, elements, &residual, work)?),
                ));
            }
            _ => {
                for term in &sorted[start..end] {
                    items.push(monomial(context, elements, term, work)?);
                }
            }
        }
        start = end;
    }
    work.reserve(items.len())?;
    let mut tree = items.pop().expect("grouped chain over at least one term");
    while let Some(item) = items.pop() {
        tree = TermTree::Add(Box::new(item), Box::new(tree));
    }
    Ok(tree)
}

fn flat_chain(
    context: &mut Context,
    elements: usize,
    terms: &[IslandTerm],
    work: &mut ScheduleBudget<'_>,
) -> Bounded<TermTree> {
    work.reserve(terms.len())?;
    let mut monomials = Vec::with_capacity(terms.len());
    for term in terms {
        monomials.push(monomial(context, elements, term, work)?);
    }
    let mut tree = monomials.pop().expect("island has at least one term");
    while let Some(term) = monomials.pop() {
        tree = TermTree::Add(Box::new(term), Box::new(tree));
    }
    Ok(tree)
}

/// Right-leaning product chain with the coefficient at the tail, so terms
/// sharing a leading factor expose it to the head-factoring rewrites.
fn monomial(
    context: &mut Context,
    elements: usize,
    term: &IslandTerm,
    work: &mut ScheduleBudget<'_>,
) -> Bounded<TermTree> {
    work.reserve(1)?;
    let mut items: Vec<TermTree> = term
        .factors
        .iter()
        .map(|&factor| TermTree::Factor(factor))
        .collect();
    if let Some(id) = term.coefficient {
        items.push(TermTree::Coefficient(id));
    }
    let Some(mut tree) = items.pop() else {
        work.reserve(elements.saturating_mul(2))?;
        return Ok(TermTree::Coefficient(context.ones_id(elements)));
    };
    while let Some(item) = items.pop() {
        tree = TermTree::Mul(Box::new(item), Box::new(tree));
    }
    Ok(tree)
}

/// One e-graph for every request: seed the flat chains and greedy variants,
/// run the directed-factoring rewrites, then extract all roots at once.
/// `None` per request keeps its flat form.
#[cfg(test)]
mod tests;
