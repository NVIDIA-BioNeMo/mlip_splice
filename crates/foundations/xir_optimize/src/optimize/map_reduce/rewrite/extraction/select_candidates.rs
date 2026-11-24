// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded shared-work improvement of complete, immutable DAG witnesses.

use super::*;

#[path = "witnesses.rs"]
mod witnesses;
pub(super) use witnesses::Witnesses;

#[path = "cost.rs"]
mod cost;
#[path = "usage.rs"]
mod usage;
use cost::{Score, evaluate};
use usage::Usage;

fn score(
    dag: &ExpressionDag,
    id: ExpressionId,
    side: AccumulationSide,
    instances: &[u64],
    cache: &mut HashMap<(ExpressionId, AccumulationSide), Score>,
    budget: &mut ExtractionBudget,
) -> Option<Score> {
    if let Some(&score) = cache.get(&(id, side)) {
        return Some(score);
    }
    let score = evaluate(dag, [(id, side)], instances, Some(budget))?;
    cache.insert((id, side), score);
    Some(score)
}

fn build(
    node: ENode,
    side: AccumulationSide,
    best: &[[Option<ExpressionId>; 2]],
    coefficients: &[ExpressionId],
    dag: &mut ExpressionDag,
    instances: &mut Vec<u64>,
) -> Option<ExpressionId> {
    let child = |class: u32| best[class as usize][side.index()];
    let (node, count) = match node {
        ENode::Value(value) => (ExpressionNode::Value(value), 0),
        ENode::Coefficient(id) => return Some(coefficients[id]),
        ENode::Negate { instances, value } => (ExpressionNode::Negate(child(value)?), instances),
        ENode::Select {
            instances,
            predicate,
            on_true,
            on_false,
        } => (
            ExpressionNode::Select(child(predicate)?, child(on_true)?, child(on_false)?),
            instances,
        ),
        ENode::Add {
            instances,
            left,
            right,
        } => (ExpressionNode::Add(child(left)?, child(right)?), instances),
        ENode::Multiply {
            instances,
            left,
            right,
        } => (
            ExpressionNode::Multiply(child(left)?, child(right)?),
            instances,
        ),
        ENode::Gather { map, axis, value } => (
            ExpressionNode::Gather {
                map,
                axis,
                value: child(value)?,
            },
            0,
        ),
        ENode::Sum { map, axis, value } => (
            ExpressionNode::Sum {
                map,
                axis,
                value: best[value as usize][AccumulationSide::BeforeAccumulation.index()]?,
            },
            0,
        ),
    };
    let id = dag.intern(node);
    instances.resize(dag.len(), 0);
    instances[id.0] = count;
    Some(id)
}

pub(super) fn extract(
    classes: &Classes,
    roots: &[u32],
    originals: &[ExpressionId],
    witnesses: Witnesses<'_>,
    class_instances: &[u64],
    coefficients: &[ExpressionId],
    dag: &mut ExpressionDag,
) -> Vec<RootExtraction> {
    let source_classes = witnesses.classes;
    let mut instances = source_classes
        .iter()
        .map(|&class| class_instances[class as usize])
        .collect::<Vec<_>>();
    instances.resize(dag.len(), 0);
    let after = AccumulationSide::AfterAccumulation;
    let initial = Usage::new(
        dag,
        originals.iter().map(|&id| (id, after)),
        &instances,
        None,
    )
    .unwrap();
    let mut candidates = vec![initial.score().candidate(originals.to_vec())];
    let mut best = vec![[None; 2]; classes.len()];
    for (id, &class) in source_classes.iter().enumerate() {
        for side in &mut best[class as usize] {
            side.get_or_insert(ExpressionId(id));
        }
    }
    let mut budget = ExtractionBudget::new();
    witnesses.seed(
        originals,
        dag,
        &mut instances,
        &mut candidates,
        initial,
        &mut budget,
    );
    let dependents = class_dependents(classes);
    let mut pending = (0..classes.len() * 2).collect::<VecDeque<_>>();
    let mut queued = vec![true; pending.len()];
    let mut cache = HashMap::new();
    'refine: while let Some(state) = pending.pop_front() {
        queued[state] = false;
        let class = state / 2;
        let side = if state % 2 == 0 {
            AccumulationSide::BeforeAccumulation
        } else {
            after
        };
        let mut changed = false;
        for &node in &classes[class] {
            if budget.reserve(1).is_err() {
                break 'refine;
            }
            let Some(id) = build(node, side, &best, coefficients, dag, &mut instances) else {
                continue;
            };
            if best[class][side.index()] == Some(id) {
                continue;
            }
            let Some(candidate) = score(dag, id, side, &instances, &mut cache, &mut budget) else {
                break 'refine;
            };
            if let Some(existing) = best[class][side.index()] {
                let Some(existing) =
                    score(dag, existing, side, &instances, &mut cache, &mut budget)
                else {
                    break 'refine;
                };
                if candidate.key() >= existing.key() {
                    continue;
                }
            }
            best[class][side.index()] = Some(id);
            changed = true;
        }
        if changed && side == after && roots.contains(&(class as u32)) {
            let Some(candidate) = complete(&best, roots, dag, &instances, Some(&mut budget)) else {
                break 'refine;
            };
            insert(&mut candidates, candidate, originals, dag);
        }
        if changed {
            for &dependent in &dependents[state] {
                if !queued[dependent] {
                    queued[dependent] = true;
                    pending.push_back(dependent);
                }
            }
        }
    }
    if let Some(candidate) = complete(&best, roots, dag, &instances, Some(&mut budget)) {
        insert(&mut candidates, candidate, originals, dag);
    }
    candidates
}

fn complete(
    best: &[[Option<ExpressionId>; 2]],
    roots: &[u32],
    dag: &ExpressionDag,
    instances: &[u64],
    budget: Option<&mut ExtractionBudget>,
) -> Option<RootExtraction> {
    let after = AccumulationSide::AfterAccumulation;
    let chosen = roots
        .iter()
        .map(|&class| best[class as usize][after.index()])
        .collect::<Option<Vec<_>>>()?;
    let score = evaluate(dag, chosen.iter().map(|&id| (id, after)), instances, budget)?;
    Some(score.candidate(chosen))
}

pub(super) fn insert(
    frontier: &mut Vec<RootExtraction>,
    candidate: RootExtraction,
    originals: &[ExpressionId],
    _dag: &ExpressionDag,
) {
    if frontier
        .iter()
        .any(|existing| existing.roots == candidate.roots)
    {
        return;
    }
    if frontier
        .iter()
        .any(|existing| existing.dominates(&candidate))
    {
        return;
    }
    frontier.retain(|existing| existing.roots == originals || !candidate.dominates(existing));
    frontier.push(candidate);
    frontier.sort_by_key(|candidate| {
        (
            candidate.physical_work,
            candidate.quality,
            candidate.work.before_accumulation,
            candidate.work.after_accumulation,
        )
    });
    if frontier.len() > MAX_FRONTIER {
        let original = frontier
            .iter()
            .position(|candidate| candidate.roots == originals)
            .unwrap();
        let remove = if original == MAX_FRONTIER / 2 {
            MAX_FRONTIER / 2 + 1
        } else {
            MAX_FRONTIER / 2
        };
        frontier.remove(remove);
    }
}

pub(super) fn refine_snapshot_roots(
    mut classes: Classes,
    roots: Vec<u32>,
    coefficients: &[ExpressionId],
    dag: &mut ExpressionDag,
) -> Result<Vec<RootExtraction>, ExtractionLimit> {
    for nodes in &mut classes {
        if let Some(node) = nodes
            .iter()
            .copied()
            .find(|node| matches!(node, ENode::Coefficient(_)))
        {
            nodes.clear();
            nodes.push(node);
        }
    }
    let mut reachable = vec![false; classes.len()];
    let mut pending = roots.clone();
    while let Some(class) = pending.pop() {
        if std::mem::replace(&mut reachable[class as usize], true) {
            continue;
        }
        for node in &classes[class as usize] {
            match *node {
                ENode::Value(_) | ENode::Coefficient(_) => {}
                ENode::Negate { value, .. }
                | ENode::Gather { value, .. }
                | ENode::Sum { value, .. } => pending.push(value),
                ENode::Add { left, right, .. } | ENode::Multiply { left, right, .. } => {
                    pending.extend([left, right])
                }
                ENode::Select {
                    predicate,
                    on_true,
                    on_false,
                    ..
                } => pending.extend([predicate, on_true, on_false]),
            }
        }
    }
    for (nodes, reachable) in classes.iter_mut().zip(reachable) {
        if !reachable {
            nodes.clear();
        }
    }
    let mut extraction_budget = ExtractionBudget::new();
    let mut frontiers =
        pareto_frontiers_with_coefficients(&classes, coefficients, &mut extraction_budget, dag)?;
    if roots.iter().any(|&root| {
        frontiers[root as usize][AccumulationSide::AfterAccumulation.index()].is_empty()
    }) {
        return Err(ExtractionLimit::ExpressionSize);
    }
    let InternedRootOperations {
        candidates_by_root,
        work_by_operation,
        opposite_sides,
    } = intern_root_operations(
        &mut frontiers,
        roots.iter().copied(),
        &mut extraction_budget,
    )?;
    let mut combined = vec![CombinedRoots {
        extraction: RootExtraction {
            work: FiniteMapWork::default(),
            physical_work: 0,
            roots: Vec::new(),
            quality: (0, 0, 0),
        },
        operations: vec![0; work_by_operation.len().div_ceil(u64::BITS as usize)],
    }];
    let root_count = roots.len();
    for (position, root) in roots.into_iter().enumerate() {
        let mut next = Vec::new();
        'combine: for accumulated in &combined {
            for candidate in &candidates_by_root[&root] {
                let reservation = extraction_budget.reserve(
                    accumulated
                        .extraction
                        .roots
                        .len()
                        .saturating_add(1)
                        .saturating_add(accumulated.operations.len())
                        .saturating_add(candidate.operations.len())
                        .saturating_add(accumulated.extraction.quality.0)
                        .saturating_add(candidate.nodes),
                );
                if let Err(limit) = reservation {
                    if position + 1 == root_count && !next.is_empty() {
                        break 'combine;
                    }
                    return Err(limit);
                }
                let mut trees = accumulated.extraction.roots.clone();
                trees.push(candidate.tree);
                let mut operations = accumulated.operations.clone();
                let mut work = accumulated.extraction.work;
                let mut physical_work = accumulated.extraction.physical_work;
                for &operation in &candidate.operations {
                    let word = operation / u64::BITS as usize;
                    let bit = 1u64 << (operation % u64::BITS as usize);
                    if operations[word] & bit != 0 {
                        continue;
                    }
                    operations[word] |= bit;
                    let opposite_live = opposite_sides[operation].is_some_and(|opposite| {
                        operations[opposite / u64::BITS as usize]
                            & (1u64 << (opposite % u64::BITS as usize))
                            != 0
                    });
                    let CountedOperation {
                        side, instances, ..
                    } = work_by_operation[operation];
                    if !opposite_live {
                        physical_work += u128::from(instances);
                    }
                    match side {
                        AccumulationSide::BeforeAccumulation => {
                            work.before_accumulation =
                                work.before_accumulation.saturating_add(instances);
                        }
                        AccumulationSide::AfterAccumulation => {
                            work.after_accumulation =
                                work.after_accumulation.saturating_add(instances);
                        }
                    }
                }
                let extraction = CombinedRoots {
                    extraction: RootExtraction {
                        work,
                        physical_work,
                        quality: dag.quality(trees.iter().copied()),
                        roots: trees,
                    },
                    operations,
                };
                insert_combined_roots(&mut next, extraction, position + 1 == root_count, dag);
            }
        }
        combined = next;
    }
    Ok(combined.into_iter().map(|roots| roots.extraction).collect())
}

fn intern_root_operations(
    frontiers: &mut ClassFrontiers,
    roots: impl IntoIterator<Item = u32>,
    budget: &mut ExtractionBudget,
) -> Result<InternedRootOperations, ExtractionLimit> {
    let mut operation_ids = HashMap::new();
    let mut operation_work = Vec::new();
    let mut root_candidates = HashMap::new();
    for root in roots {
        budget.reserve(
            frontiers[root as usize][AccumulationSide::AfterAccumulation.index()]
                .iter()
                .fold(1usize, |work, candidate| {
                    work.saturating_add(candidate.quality.0)
                        .saturating_add(candidate.operations.len())
                }),
        )?;
        root_candidates.entry(root).or_insert_with(|| {
            std::mem::take(
                &mut frontiers[root as usize][AccumulationSide::AfterAccumulation.index()],
            )
            .into_iter()
            .map(|candidate| {
                let mut operations = candidate
                    .operations
                    .into_iter()
                    .map(|operation| match operation_ids.entry(operation) {
                        std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            let id = operation_work.len();
                            operation_work.push(entry.key().clone());
                            entry.insert(id);
                            id
                        }
                    })
                    .collect::<Vec<_>>();
                operations.sort_unstable();
                RootCandidate {
                    tree: candidate.tree,
                    operations,
                    nodes: candidate.quality.0,
                }
            })
            .collect()
        });
    }
    budget.reserve(operation_work.len())?;
    let opposite_sides = operation_work
        .iter()
        .map(|operation| {
            operation_ids
                .get(&CountedOperation {
                    side: operation.side.opposite(),
                    ..operation.clone()
                })
                .copied()
        })
        .collect();
    Ok(InternedRootOperations {
        candidates_by_root: root_candidates,
        work_by_operation: operation_work,
        opposite_sides,
    })
}

#[cfg(test)]
pub(super) fn pareto_frontiers(classes: &Classes) -> Result<ClassFrontiers, ExtractionLimit> {
    pareto_frontiers_with_coefficients(
        classes,
        &[],
        &mut ExtractionBudget::new(),
        &mut ExpressionDag::default(),
    )
}

pub(super) fn pareto_frontiers_with_coefficients(
    classes: &Classes,
    coefficients: &[ExpressionId],
    budget: &mut ExtractionBudget,
    dag: &mut ExpressionDag,
) -> Result<ClassFrontiers, ExtractionLimit> {
    let mut frontiers = vec![[Vec::new(), Vec::new()]; classes.len()];
    let state_count = classes.len().saturating_mul(2);
    let dependents = class_dependents(classes);

    let mut pending = (0..state_count).collect::<VecDeque<_>>();
    let mut queued = vec![true; state_count];
    while let Some(state) = pending.pop_front() {
        queued[state] = false;
        let class = state / 2;
        let side = if state % 2 == 0 {
            AccumulationSide::BeforeAccumulation
        } else {
            AccumulationSide::AfterAccumulation
        };
        let mut state_candidates = Vec::new();
        for &node in &classes[class] {
            budget.reserve(candidate_construction_work(node, side, &frontiers))?;
            state_candidates.extend(candidates(node, side, &frontiers, coefficients, dag));
        }
        let mut changed = false;
        for candidate in state_candidates {
            changed |= insert_candidate(&mut frontiers[class][side.index()], candidate, dag);
        }
        if !changed {
            continue;
        }
        for &dependent in &dependents[state] {
            if !queued[dependent] {
                pending.push_back(dependent);
                queued[dependent] = true;
            }
        }
    }
    Ok(frontiers)
}

fn candidate_construction_work(
    node: ENode,
    side: AccumulationSide,
    frontiers: &ClassFrontiers,
) -> usize {
    let child = |class: u32| &frontiers[class as usize][side.index()];
    match node {
        ENode::Value(_) | ENode::Coefficient(_) => 1,
        ENode::Negate { value, .. } | ENode::Gather { value, .. } => {
            cartesian_tree_nodes(&[child(value)])
        }
        ENode::Sum { value, .. } => cartesian_tree_nodes(&[
            &frontiers[value as usize][AccumulationSide::BeforeAccumulation.index()]
        ]),
        ENode::Add { left, right, .. } | ENode::Multiply { left, right, .. } => {
            cartesian_tree_nodes(&[child(left), child(right)])
        }
        ENode::Select {
            predicate,
            on_true,
            on_false,
            ..
        } => cartesian_tree_nodes(&[child(predicate), child(on_true), child(on_false)]),
    }
}

pub(super) fn cartesian_tree_nodes(children: &[&Vec<ExtractionCandidate>]) -> usize {
    children
        .iter()
        .fold((1usize, 1usize), |(count, nodes), child| {
            let child_nodes = child.iter().fold(0usize, |total, candidate| {
                total
                    .saturating_add(candidate.quality.0)
                    .saturating_add(candidate.operations.len())
            });
            (
                count.saturating_mul(child.len()),
                nodes
                    .saturating_mul(child.len())
                    .saturating_add(count.saturating_mul(child_nodes)),
            )
        })
        .1
}

fn candidates(
    node: ENode,
    side: AccumulationSide,
    frontiers: &ClassFrontiers,
    coefficients: &[ExpressionId],
    dag: &mut ExpressionDag,
) -> Vec<ExtractionCandidate> {
    match node {
        ENode::Value(value) => vec![ExtractionCandidate::value(value, dag)],
        ENode::Coefficient(id) => vec![ExtractionCandidate {
            work: FiniteMapWork::default(),
            tree: coefficients[id],
            operations: HashSet::new(),
            quality: dag.quality([coefficients[id]]),
        }],
        ENode::Negate { instances, value } => frontiers[value as usize][side.index()]
            .iter()
            .map(|value| {
                let tree = dag.intern(ExpressionNode::Negate(value.tree));
                let mut operations = value.operations.clone();
                operations.insert(CountedOperation {
                    side,
                    instances,
                    expression: tree,
                });
                ExtractionCandidate {
                    work: work_for_operations(&operations),
                    tree,
                    operations,
                    quality: dag.quality([tree]),
                }
            })
            .collect(),
        ENode::Select {
            instances,
            predicate,
            on_true,
            on_false,
        } => {
            let mut result = Vec::new();
            for p in &frontiers[predicate as usize][side.index()] {
                for t in &frontiers[on_true as usize][side.index()] {
                    for f in &frontiers[on_false as usize][side.index()] {
                        let tree = dag.intern(ExpressionNode::Select(p.tree, t.tree, f.tree));
                        let mut operations = p.operations.clone();
                        operations.extend(t.operations.iter().cloned());
                        operations.extend(f.operations.iter().cloned());
                        operations.insert(CountedOperation {
                            side,
                            instances,
                            expression: tree,
                        });
                        result.push(ExtractionCandidate {
                            work: work_for_operations(&operations),
                            tree,
                            operations,
                            quality: dag.quality([tree]),
                        });
                    }
                }
            }
            result
        }
        ENode::Add {
            instances,
            left,
            right,
        } => binary_candidates(
            &frontiers[left as usize][side.index()],
            &frontiers[right as usize][side.index()],
            side,
            instances,
            ExpressionNode::Add,
            dag,
        ),
        ENode::Multiply {
            instances,
            left,
            right,
        } => binary_candidates(
            &frontiers[left as usize][side.index()],
            &frontiers[right as usize][side.index()],
            side,
            instances,
            ExpressionNode::Multiply,
            dag,
        ),
        ENode::Gather { map, axis, value } => frontiers[value as usize][side.index()]
            .iter()
            .map(|candidate| {
                let tree = dag.intern(ExpressionNode::Gather {
                    map,
                    axis,
                    value: candidate.tree,
                });
                candidate.unary(tree, dag)
            })
            .collect(),
        ENode::Sum { map, axis, value } => frontiers[value as usize]
            [AccumulationSide::BeforeAccumulation.index()]
        .iter()
        .map(|candidate| {
            let tree = dag.intern(ExpressionNode::Sum {
                map,
                axis,
                value: candidate.tree,
            });
            candidate.unary(tree, dag)
        })
        .collect(),
    }
}

fn binary_candidates(
    left: &[ExtractionCandidate],
    right: &[ExtractionCandidate],
    side: AccumulationSide,
    instances: u64,
    make_node: impl Copy + FnOnce(ExpressionId, ExpressionId) -> ExpressionNode,
    dag: &mut ExpressionDag,
) -> Vec<ExtractionCandidate> {
    let mut result = Vec::new();
    for left in left {
        for right in right {
            result.push(ExtractionCandidate::binary(
                left, right, side, instances, make_node, dag,
            ));
        }
    }
    result
}

pub(super) fn insert_candidate(
    frontier: &mut Vec<ExtractionCandidate>,
    candidate: ExtractionCandidate,
    _dag: &ExpressionDag,
) -> bool {
    let dominates = |left: &ExtractionCandidate, right: &ExtractionCandidate| {
        left.work.dominates(right.work)
            && left.operations.is_subset(&right.operations)
            && (left.operations != right.operations || left.quality <= right.quality)
    };
    if frontier
        .iter()
        .any(|existing| dominates(existing, &candidate))
    {
        return false;
    }
    frontier.retain(|existing| !dominates(&candidate, existing));
    let tree = candidate.tree;
    frontier.push(candidate);
    frontier.sort_by_cached_key(|candidate| {
        (
            candidate.work.before_accumulation,
            candidate.work.after_accumulation,
            candidate.quality,
        )
    });
    bound_frontier(frontier);
    frontier.iter().any(|candidate| candidate.tree == tree)
}

fn operation_subset(left: &[u64], right: &[u64]) -> bool {
    left.iter()
        .zip(right)
        .all(|(left, right)| left & !right == 0)
}

fn insert_combined_roots(
    frontier: &mut Vec<CombinedRoots>,
    candidate: CombinedRoots,
    complete: bool,
    _dag: &ExpressionDag,
) {
    let dominates = |left: &CombinedRoots, right: &CombinedRoots| {
        left.extraction.dominates(&right.extraction)
            && (complete || operation_subset(&left.operations, &right.operations))
    };
    if frontier
        .iter()
        .any(|existing| dominates(existing, &candidate))
    {
        return;
    }
    frontier.retain(|existing| !dominates(&candidate, existing));
    frontier.push(candidate);
    frontier.sort_by_key(|candidate| {
        (
            candidate.extraction.work.before_accumulation,
            candidate.extraction.work.after_accumulation,
            candidate.extraction.quality,
        )
    });
    if frontier.len() > MAX_FRONTIER {
        let candidates = std::mem::take(frontier);
        let last = candidates.len() - 1;
        frontier.extend(
            (0..MAX_FRONTIER)
                .map(|position| candidates[position * last / (MAX_FRONTIER - 1)].clone()),
        );
    }
}

fn bound_frontier(frontier: &mut Vec<ExtractionCandidate>) {
    if frontier.len() <= MAX_FRONTIER {
        return;
    }
    let candidates = std::mem::take(frontier);
    let last = candidates.len() - 1;
    frontier.extend(
        (0..MAX_FRONTIER).map(|position| candidates[position * last / (MAX_FRONTIER - 1)].clone()),
    );
}

fn class_dependents(classes: &Classes) -> Vec<Vec<usize>> {
    let state_count = classes.len().saturating_mul(2);
    let mut dependents = vec![Vec::new(); state_count];
    for (class, nodes) in classes.iter().enumerate() {
        for &node in nodes {
            match node {
                ENode::Value(_) | ENode::Coefficient(_) => {}
                ENode::Negate { value, .. } => {
                    for side in 0..2 {
                        dependents[value as usize * 2 + side].push(class * 2 + side);
                    }
                }
                ENode::Select {
                    predicate,
                    on_true,
                    on_false,
                    ..
                } => {
                    for child in [predicate, on_true, on_false] {
                        for side in 0..2 {
                            dependents[child as usize * 2 + side].push(class * 2 + side);
                        }
                    }
                }
                ENode::Add { left, right, .. } | ENode::Multiply { left, right, .. } => {
                    for side in 0..2 {
                        dependents[left as usize * 2 + side].push(class * 2 + side);
                        dependents[right as usize * 2 + side].push(class * 2 + side);
                    }
                }
                ENode::Gather { value, .. } => {
                    for side in 0..2 {
                        dependents[value as usize * 2 + side].push(class * 2 + side);
                    }
                }
                ENode::Sum { value, .. } => {
                    dependents[value as usize * 2].push(class * 2);
                    dependents[value as usize * 2].push(class * 2 + 1);
                }
            }
        }
    }
    for states in &mut dependents {
        states.sort_unstable();
        states.dedup();
    }

    dependents
}

#[cfg(test)]
mod tests;
