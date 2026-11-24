// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, cost-directed substitution of immutable complete proofs.

use super::*;

pub(in crate::optimize::map_reduce::rewrite::extraction) struct Witnesses<'a> {
    pub(in crate::optimize::map_reduce::rewrite::extraction) source_len: usize,
    pub(in crate::optimize::map_reduce::rewrite::extraction) classes: &'a [u32],
    pub(in crate::optimize::map_reduce::rewrite::extraction) proofs: &'a [ExpressionId],
}

struct Substitutions {
    classes: Vec<u32>,
    members: HashMap<u32, Vec<ExpressionId>>,
    parents: Vec<Vec<ExpressionId>>,
    representatives: HashMap<u32, ExpressionId>,
}

impl Witnesses<'_> {
    pub(super) fn seed(
        &self,
        originals: &[ExpressionId],
        dag: &mut ExpressionDag,
        instances: &mut Vec<u64>,
        candidates: &mut Vec<RootExtraction>,
        mut usage: Usage,
        budget: &mut ExtractionBudget,
    ) {
        let Some(mut substitutions) = Substitutions::new(self.classes, dag, budget) else {
            return;
        };
        let mut proofs = HashMap::<u32, Vec<ExpressionId>>::new();
        for &proof in self.proofs {
            if budget.reserve(1).is_err() {
                return;
            }
            proofs.entry(self.classes[proof.0]).or_default().push(proof);
        }
        let mut roots = originals.to_vec();
        // Reverse concrete source order reaches consumers before their inputs;
        // class labels affect lookup only, never the scheduling order.
        for i in (0..self.source_len).rev() {
            if budget.reserve(1).is_err() {
                return;
            }
            let class = self.classes[i];
            let Some(proofs) = proofs.remove(&class) else {
                continue;
            };
            for proof in proofs {
                let Some((chosen, rebuilt)) = substitutions.replace(
                    proof,
                    class,
                    self.source_len,
                    &roots,
                    &usage,
                    dag,
                    instances,
                    budget,
                ) else {
                    return;
                };
                let changes = roots
                    .iter()
                    .copied()
                    .zip(chosen.iter().copied())
                    .collect::<Vec<_>>();
                let before = usage.score().key();
                let Some(score) = usage.improve(dag, &changes, instances, budget) else {
                    return;
                };
                insert(candidates, score.candidate(chosen.clone()), originals, dag);
                if score.key() >= before {
                    continue;
                }
                let replacements = rebuilt
                    .into_iter()
                    .filter_map(|(old, new)| {
                        let class = substitutions.classes[old.0];
                        (substitutions.representatives.get(&class) == Some(&old))
                            .then_some((class, new))
                    })
                    .collect::<Vec<_>>();
                substitutions.representatives.extend(replacements);
                roots = chosen;
            }
        }
    }
}

impl Substitutions {
    fn new(classes: &[u32], dag: &ExpressionDag, budget: &mut ExtractionBudget) -> Option<Self> {
        budget.reserve(classes.len()).ok()?;
        let mut result = Self {
            classes: classes.to_vec(),
            members: HashMap::new(),
            parents: vec![Vec::new(); dag.len()],
            representatives: HashMap::new(),
        };
        for (i, &class) in classes.iter().enumerate() {
            let id = ExpressionId(i);
            result.members.entry(class).or_default().push(id);
            result.representatives.entry(class).or_insert(id);
            for child in dag.node(id).children() {
                budget.reserve(1).ok()?;
                result.parents[child.0].push(id);
            }
        }
        Some(result)
    }

    fn intern(
        &mut self,
        old: ExpressionId,
        node: ExpressionNode,
        dag: &mut ExpressionDag,
        instances: &mut Vec<u64>,
        budget: &mut ExtractionBudget,
    ) -> Option<ExpressionId> {
        budget.reserve(1 + node.children().count()).ok()?;
        let id = dag.intern(node);
        if id.0 == self.classes.len() {
            let class = self.classes[old.0];
            self.classes.push(class);
            self.members.entry(class).or_default().push(id);
            self.parents.push(Vec::new());
            for child in dag.node(id).children() {
                self.parents[child.0].push(id);
            }
            instances.push(instances[old.0]);
        }
        Some(id)
    }

    /// Proof edges are concrete. Original leaves may refer to the current
    /// representative once; its own edges are never recursively substituted.
    fn rebase(
        &mut self,
        root: ExpressionId,
        class: u32,
        source_len: usize,
        dag: &mut ExpressionDag,
        instances: &mut Vec<u64>,
        budget: &mut ExtractionBudget,
    ) -> Option<ExpressionId> {
        let mut pending = vec![(root, false)];
        let mut rebuilt = HashMap::new();
        while let Some((id, visited)) = pending.pop() {
            budget.reserve(1).ok()?;
            if rebuilt.contains_key(&id) {
                continue;
            }
            let child_class = self.classes[id.0];
            if id.0 < source_len {
                let chosen = if child_class == class {
                    id
                } else {
                    self.representatives[&child_class]
                };
                rebuilt.insert(id, chosen);
            } else if visited {
                let node = dag.node(id).map_children(|child| rebuilt[&child]);
                let chosen = self.intern(id, node, dag, instances, budget)?;
                rebuilt.insert(id, chosen);
            } else {
                pending.push((id, true));
                pending.extend(dag.node(id).children().map(|child| (child, false)));
            }
        }
        Some(rebuilt[&root])
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "One bounded substitution has explicit DAG, proof, cost and budget inputs"
    )]
    fn replace(
        &mut self,
        proof: ExpressionId,
        class: u32,
        source_len: usize,
        roots: &[ExpressionId],
        usage: &Usage,
        dag: &mut ExpressionDag,
        instances: &mut Vec<u64>,
        budget: &mut ExtractionBudget,
    ) -> Option<(Vec<ExpressionId>, HashMap<ExpressionId, ExpressionId>)> {
        let mut targets = Vec::new();
        for &id in &self.members[&class] {
            budget.reserve(1).ok()?;
            if usage.live(id) {
                targets.push(id);
            }
        }
        if targets.is_empty() {
            return Some((roots.to_vec(), HashMap::new()));
        }
        let proof = self.rebase(proof, class, source_len, dag, instances, budget)?;
        let mut rebuilt = targets
            .iter()
            .map(|&id| (id, proof))
            .collect::<HashMap<_, _>>();
        let mut affected = HashSet::new();
        let mut pending = targets;
        while let Some(id) = pending.pop() {
            budget.reserve(1).ok()?;
            if !affected.insert(id) {
                continue;
            }
            for &parent in &self.parents[id.0] {
                budget.reserve(1).ok()?;
                if usage.live(parent) {
                    pending.push(parent);
                }
            }
        }
        let mut affected = affected.into_iter().collect::<Vec<_>>();
        affected.sort_unstable_by_key(|id| id.0);
        for id in affected {
            if rebuilt.contains_key(&id) {
                continue;
            }
            let node = dag
                .node(id)
                .map_children(|child| rebuilt.get(&child).copied().unwrap_or(child));
            let chosen = self.intern(id, node, dag, instances, budget)?;
            rebuilt.insert(id, chosen);
        }
        budget.reserve(roots.len()).ok()?;
        let chosen = roots
            .iter()
            .map(|id| rebuilt.get(id).copied().unwrap_or(*id))
            .collect();
        Some((chosen, rebuilt))
    }
}
