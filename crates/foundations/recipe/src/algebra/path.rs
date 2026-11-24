// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Naming one place in one recipe.
//!
//! A recipe is a tree of bodies: the root, and a body for each candidate of
//! each choice below it. A path is the descent — which node, which candidate —
//! and then which node in the body it arrives at.
//!
//! A path also names *whose* tree it descends. Indices alone would be a
//! coincidence waiting to happen: the same numbers name a node in every recipe
//! that is large enough, so a path from somewhere else, or from before an edit
//! moved what it named, would quietly resolve to whatever now sits there. So a
//! path carries the [`Shape`] it was taken from, and every navigation and every
//! transformation checks it. There is no way to write one down by hand — a path
//! comes from the recipe it names, and goes back to it.
//!
//! What a path survives is what leaves the shape alone. Compiling a recipe from
//! one representation to the next keeps every path that named it; moving work
//! does not, and paths taken before refuse afterwards.

use crate::algebra::computation::Computation;
use crate::algebra::data::{Node, Recipe, RecipeBody, Shape, Work};
use crate::algebra::error::RecipeError;

/// One step down into a choice: which node, and which of its candidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Descent {
    pub node: usize,
    pub candidate: usize,
}

/// Which body of which recipe. The empty descent is that recipe's root body.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BodyPath {
    shape: Shape,
    steps: Vec<Descent>,
}

/// Which node of which body.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodePath {
    body: BodyPath,
    node: usize,
}

impl BodyPath {
    /// The body of `candidate` of the choice at `node` of this body — still in
    /// the same recipe, so still checkable against it.
    pub fn descend(&self, node: usize, candidate: usize) -> BodyPath {
        let mut steps = self.steps.clone();
        steps.push(Descent { node, candidate });
        BodyPath {
            shape: self.shape.clone(),
            steps,
        }
    }

    pub fn steps(&self) -> &[Descent] {
        &self.steps
    }

    pub fn is_root(&self) -> bool {
        self.steps.is_empty()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The node at `node` of this body.
    pub fn node(&self, node: usize) -> NodePath {
        NodePath {
            body: self.clone(),
            node,
        }
    }

    pub(crate) fn rooted(shape: Shape) -> BodyPath {
        BodyPath {
            shape,
            steps: Vec::new(),
        }
    }
}

impl NodePath {
    pub fn body(&self) -> &BodyPath {
        &self.body
    }

    pub fn node(&self) -> usize {
        self.node
    }

    pub fn shape(&self) -> &Shape {
        &self.body.shape
    }
}

impl<F: Computation> Recipe<F> {
    /// The root body of this recipe, and the start of every path into it.
    pub fn root_path(&self) -> BodyPath {
        BodyPath::rooted(self.shape().clone())
    }

    /// Whether this path was taken from this recipe.
    fn owns<R>(&self, shape: &Shape) -> Result<(), RecipeError<R>> {
        if shape == self.shape() {
            Ok(())
        } else {
            Err(RecipeError::PathNotOwned)
        }
    }

    pub fn body_at(&self, path: &BodyPath) -> Result<&RecipeBody<F>, RecipeError<F::Refusal>> {
        self.owns(path.shape())?;
        descend(self.body(), path.steps())
    }

    pub fn node_at(&self, path: &NodePath) -> Result<&Node<F>, RecipeError<F::Refusal>> {
        self.body_at(path.body())?
            .nodes()
            .get(path.node())
            .ok_or_else(|| RecipeError::NoSuchNode { path: path.clone() })
    }

    /// The call at `path`, or the refusal saying what sits there instead.
    pub fn call_at(&self, path: &NodePath) -> Result<&F::Call, RecipeError<F::Refusal>> {
        match self.node_at(path)?.work() {
            Work::Call { call, .. } => Ok(call),
            _ => Err(RecipeError::NotACall { path: path.clone() }),
        }
    }

    /// Every node in this recipe, each with the path that names it: a body's
    /// own nodes in index order, and a choice's candidates after the node that
    /// offers them.
    pub fn nodes_with_paths(&self) -> Vec<(NodePath, &Node<F>)> {
        let mut found = Vec::new();
        collect_nodes(self.body(), &self.root_path(), &mut found);
        found
    }

    pub fn node_paths(&self) -> Vec<NodePath> {
        self.nodes_with_paths()
            .into_iter()
            .map(|(path, _)| path)
            .collect()
    }

    /// The same, restricted to the nodes that carry a computation.
    pub fn call_paths(&self) -> Vec<NodePath> {
        self.nodes_with_paths()
            .into_iter()
            .filter(|(_, node)| matches!(node.work(), Work::Call { .. }))
            .map(|(path, _)| path)
            .collect()
    }

    /// This recipe's body with `update` applied to the body `path` names.
    ///
    /// Every body from the root down to the one that changed is rebuilt, and
    /// this recipe is untouched: a recipe is a value, so an edit produces the
    /// recipe that has it rather than mutating the one that did not.
    pub(crate) fn update_body_at(
        &self,
        path: &BodyPath,
        update: &mut impl FnMut(&RecipeBody<F>) -> Result<RecipeBody<F>, RecipeError<F::Refusal>>,
    ) -> Result<RecipeBody<F>, RecipeError<F::Refusal>> {
        self.owns(path.shape())?;
        update_steps(self.body(), path.steps(), 0, update)
    }
}

fn descend<'a, F: Computation, R>(
    body: &'a RecipeBody<F>,
    steps: &[Descent],
) -> Result<&'a RecipeBody<F>, RecipeError<R>> {
    let mut body = body;
    for (depth, step) in steps.iter().enumerate() {
        let candidates = match body.nodes().get(step.node).map(|node| node.work()) {
            Some(Work::Choice { candidates, .. }) => candidates,
            _ => return Err(RecipeError::NoSuchBody { at: depth }),
        };
        body = candidates
            .get(step.candidate)
            .ok_or(RecipeError::NoSuchBody { at: depth })?;
    }
    Ok(body)
}

fn collect_nodes<'a, F: Computation>(
    body: &'a RecipeBody<F>,
    at: &BodyPath,
    found: &mut Vec<(NodePath, &'a Node<F>)>,
) {
    for (node, held) in body.nodes().iter().enumerate() {
        found.push((at.node(node), held));
        if let Work::Choice { candidates, .. } = held.work() {
            for (candidate, body) in candidates.iter().enumerate() {
                collect_nodes(body, &at.descend(node, candidate), found);
            }
        }
    }
}

fn update_steps<F: Computation>(
    body: &RecipeBody<F>,
    steps: &[Descent],
    depth: usize,
    update: &mut impl FnMut(&RecipeBody<F>) -> Result<RecipeBody<F>, RecipeError<F::Refusal>>,
) -> Result<RecipeBody<F>, RecipeError<F::Refusal>> {
    let Some((step, rest)) = steps.split_first() else {
        return update(body);
    };
    let mut nodes = body.nodes().to_vec();
    let held = nodes
        .get_mut(step.node)
        .ok_or(RecipeError::NoSuchBody { at: depth })?;
    let Work::Choice {
        candidates,
        input_binding,
        output_binding,
    } = held.work().clone()
    else {
        return Err(RecipeError::NoSuchBody { at: depth });
    };
    let mut candidates = candidates;
    let candidate = candidates
        .get(step.candidate)
        .ok_or(RecipeError::NoSuchBody { at: depth })?;
    candidates[step.candidate] = update_steps(candidate, rest, depth + 1, update)?;
    *held = Node::new(
        Work::Choice {
            candidates,
            input_binding,
            output_binding,
        },
        held.deps().iter().copied(),
    );
    Ok(RecipeBody::with_aliases(
        body.inputs(),
        body.outputs(),
        body.aliases().clone(),
        body.context().clone(),
        body.workspace().to_vec(),
        nodes,
    ))
}
