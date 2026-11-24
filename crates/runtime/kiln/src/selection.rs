// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use recipe::algebra::{ExecutableRefusal, Launches};
use recipe::{CompileRequests, NodeId, Recipe, RecipeBody, RecipeBuilder, RecipeError, Work};
#[cfg(test)]
use recipe::{Geometry, KernelLaunch};

pub(crate) fn choice_indices<F: Launches>(recipe: &Recipe<F>) -> Vec<usize> {
    body_choice_indices(recipe.body())
}

pub(crate) fn body_choice_indices<F: Launches>(body: &RecipeBody<F>) -> Vec<usize> {
    body.nodes()
        .iter()
        .enumerate()
        .filter_map(|(idx, node)| matches!(node.work(), Work::Choice { .. }).then_some(idx))
        .collect()
}

fn remap_deps(deps: &BTreeSet<usize>, sinks: &[Vec<usize>]) -> BTreeSet<usize> {
    deps.iter()
        .flat_map(|&dep| sinks[dep].iter().copied())
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    pub candidate: usize,
    pub nested: Vec<Selection>,
}

impl Selection {
    #[cfg(test)]
    pub fn leaf(candidate: usize) -> Self {
        Selection {
            candidate,
            nested: Vec::new(),
        }
    }
}

/// The first ordered path through a recipe: the first candidate of every
/// choice, descending into that candidate's own choices.
///
/// It is the path a caller wrote first, and so the one a baseline comes from
/// when nothing else states what the recipe should produce.
pub fn first_selection<F: Launches>(recipe: &Recipe<F>) -> Vec<Selection> {
    body_default_selection(recipe.body())
}

pub(crate) fn body_default_selection<F: Launches>(body: &RecipeBody<F>) -> Vec<Selection> {
    body_choice_indices(body)
        .iter()
        .map(|&ci| match body.nodes()[ci].work() {
            Work::Choice { candidates, .. } => Selection {
                candidate: 0,
                nested: body_default_selection(&candidates[0]),
            },
            _ => unreachable!("choice_indices points only at Choice nodes"),
        })
        .collect()
}

/// Inlining chooses which candidate runs; it does not change what the recipe is
/// bound through, so the ports and the compile table come across untouched.
///
/// The selection is proved against the recipe before anything is read from it:
/// one entry per choice, at every depth, each naming a candidate that choice
/// offers. A selection that is not this recipe's is a refusal rather than
/// whatever happens to sit at those indices.
pub fn inline<F: Launches>(
    recipe: &Recipe<F>,
    selection: &[Selection],
    authorization: &F::Authorization,
) -> Result<Recipe<F>, RecipeError<ExecutableRefusal>> {
    selects(recipe.body(), selection)?;
    let body = inline_body(recipe.context(), recipe.body(), selection, authorization)?;
    Recipe::stating_edges_authorized(
        recipe.contract().clone(),
        recipe.context().clone(),
        body,
        authorization,
    )
}

/// Whether `selection` names the choices of `body`: as many entries as it holds
/// choices, each naming a candidate it offers, and the same again inside
/// whichever candidate each entry names.
pub(crate) fn selects<F: Launches>(
    body: &RecipeBody<F>,
    selection: &[Selection],
) -> Result<(), RecipeError<ExecutableRefusal>> {
    let choices = body_choice_indices(body);
    if selection.len() != choices.len() {
        return Err(RecipeError::SelectionArity {
            stated: selection.len(),
            choices: choices.len(),
        });
    }
    for (held, &node) in selection.iter().zip(&choices) {
        let Work::Choice { candidates, .. } = body.nodes()[node].work() else {
            unreachable!("choice_indices points only at Choice nodes")
        };
        let Some(candidate) = candidates.get(held.candidate) else {
            return Err(RecipeError::NoSuchCandidate {
                node,
                candidate: held.candidate,
                offered: candidates.len(),
            });
        };
        selects(candidate, &held.nested)?;
    }
    Ok(())
}

/// Splice the selected candidate of every choice into one choice-free body.
///
/// A module's `request` names the root recipe's shared table, so inlining
/// neither rewrites those indices nor carries a copy of the table: the
/// resolver borrows the one the pending recipe owns.
pub(crate) fn inline_body<F: Launches>(
    requests: &CompileRequests,
    body: &RecipeBody<F>,
    selection: &[Selection],
    authorization: &F::Authorization,
) -> Result<RecipeBody<F>, RecipeError<ExecutableRefusal>> {
    selects(body, selection)?;
    let choices = body_choice_indices(body);

    let mut out = RecipeBuilder::new_authorized(body.inputs(), body.outputs(), authorization);
    // A candidate's modules name the root recipe's compile table, so resolving
    // one is only meaningful against that table. It is handed over rather than
    // copied: what comes back below is a body, and the table it names is the
    // one the caller already holds.
    *out.root_context_mut() = requests.clone();
    // The body's own data comes across whole rather than rebuilt from its
    // modules: what a family keeps beside them is the family's, and inlining
    // decides which candidate runs, not what the body holding it is.
    *out.body_context_mut() = body.context().clone();
    for ws in body.workspace() {
        match &ws.init {
            Some(bytes) => out.scratch_init(bytes.clone()),
            None => out.scratch(ws.nbytes),
        };
    }

    let mut sinks: Vec<Vec<usize>> = vec![Vec::new(); body.nodes().len()];
    let mut choice_of = vec![usize::MAX; body.nodes().len()];
    for (k, &ci) in choices.iter().enumerate() {
        choice_of[ci] = k;
    }

    for (orig_idx, node) in body.nodes().iter().enumerate() {
        let deps = remap_deps(node.deps(), &sinks);
        if let Work::Choice {
            candidates,
            input_binding,
            output_binding,
        } = node.work()
        {
            let sel = &selection[choice_of[orig_idx]];
            // Resolve the chosen candidate's own choices first, so what we
            // splice is a flat sub-body in the candidate's formal index space.
            // `selects` proved the candidate is one this choice offers.
            let chosen = candidates
                .get(sel.candidate)
                .ok_or(RecipeError::NoSuchCandidate {
                    node: orig_idx,
                    candidate: sel.candidate,
                    offered: candidates.len(),
                })?;
            let resolved = inline_body(requests, chosen, &sel.nested, authorization)?;
            let external: Vec<NodeId> = deps.iter().map(|&d| NodeId::from_index(d)).collect();
            let cand_sinks =
                out.splice_body(&resolved, input_binding, output_binding, &external)?;
            sinks[orig_idx] = cand_sinks.iter().map(|n| n.index()).collect();
        } else {
            let new = out.land(node.work().clone(), deps);
            sinks[orig_idx] = vec![new.index()];
        }
    }

    // A body has no ports of its own — whatever selects it does — so none are
    // stated here. The table that comes back is the one handed in: inlining
    // moves bodies whose module handles already name it and states no compile
    // of its own.
    Ok(out.finish_fragment().into_parts().1)
}

/// Where the body being tuned sits inside the whole body.
///
/// The tuner descends into one candidate at a time and measures it against the
/// buffers its parent bound, so a frame deep in the tree knows nothing about
/// the body as a whole. Validation needs exactly that: a statement about a
/// candidate is a statement about the program that contains it. A site is the
/// chain back to the root — at each level, which choice was descended through,
/// into which candidate, and what that level's other choices take meanwhile.
pub(crate) struct Site<'a> {
    pub(crate) parent: Option<&'a Site<'a>>,
    /// The choice descended through, by position among its body's choices.
    pub(crate) position: usize,
    /// The candidate of that choice this site is inside.
    pub(crate) candidate: usize,
    /// One selection per choice of the parent body: what the earlier choices
    /// have already won, and for the later ones a selection that is known to
    /// resolve. The entry at `position` is replaced on the way up.
    pub(crate) siblings: Vec<Selection>,
}

/// The whole body's selection with `local` in place at `site`.
///
/// Walking up replaces each level's descended choice with the selection built
/// below it, so the result names one candidate per choice of the root — the
/// program that contains the candidate under test, rather than the candidate
/// alone.
pub(crate) fn root_selection(site: Option<&Site<'_>>, local: Vec<Selection>) -> Vec<Selection> {
    let mut here = local;
    let mut site = site;
    while let Some(frame) = site {
        let mut up = frame.siblings.clone();
        up[frame.position] = Selection {
            candidate: frame.candidate,
            nested: here,
        };
        here = up;
        site = frame.parent;
    }
    here
}

/// A set of selections in which every candidate of every choice, at every
/// depth, appears at least once — the smallest cover of the candidate space,
/// not its product.
///
/// This is structure alone: the recipes [`inline`] builds from these selections
/// name whatever images their pending modules named, so a source-bearing one
/// still has to be realized before it can load. [`crate::Realizer::for_each_covering_recipe`]
/// is that path; this is for callers that only inspect or account for the
/// candidate space and never reach a device.
pub fn covering_selections<F: Launches>(body: &Recipe<F>) -> Vec<Vec<Selection>> {
    body_covering_selections(body.body())
}

fn body_covering_selections<F: Launches>(body: &RecipeBody<F>) -> Vec<Vec<Selection>> {
    let choices = body_choice_indices(body);
    if choices.is_empty() {
        return vec![Vec::new()];
    }
    let mut out: Vec<Vec<Selection>> = Vec::new();
    for (k, &ci) in choices.iter().enumerate() {
        let Work::Choice { candidates, .. } = body.nodes()[ci].work() else {
            unreachable!("choice_indices points only at Choice nodes")
        };
        for (c, cand) in candidates.iter().enumerate() {
            for nested in body_covering_selections(cand) {
                let mut tree = body_default_selection(body);
                tree[k] = Selection {
                    candidate: c,
                    nested,
                };
                if !out.contains(&tree) {
                    out.push(tree);
                }
            }
        }
    }
    out
}

#[cfg(test)]
#[path = "selection/site_tests.rs"]
mod site_tests;

#[cfg(test)]
#[path = "selection/inline_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "selection/binding_tests.rs"]
mod binding_tests;
