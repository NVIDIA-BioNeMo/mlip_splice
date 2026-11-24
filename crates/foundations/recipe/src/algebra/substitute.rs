// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Putting one recipe's work inside another's.
//!
//! An incoming body names buffers in its own formal slot space — `Input(i)`,
//! `Output(j)`, `Workspace(k)` — and its calls name its own context. Landing it
//! somewhere rewrites the formal slots through the host's binding, shifts
//! workspace onto the host's, and hands the family its two remaps so the
//! handles it owns follow. Byte offsets are inside the buffer the slot selects,
//! so they cross unchanged.
//!
//! Nothing is indexed into before it is proved: a fragment's shape is checked
//! first, so a body whose deps name nodes it does not have is a refusal rather
//! than a walk off the end of a table.
//!
//! Editing a *built* recipe is a stronger thing to ask than building one. A
//! builder derives edges from what each placement proves; a built recipe kept
//! the edges and not the proof. So substituting into a placement, or replacing
//! what one computes, is offered only to families that keep their calls'
//! effects ([`DeclaredEffects`]) — and each of them proves that what arrives
//! does no more than what the edges around it were derived from. Where that
//! holds, the surrounding graph is preserved exactly; where it does not, the
//! edit is refused rather than resting on an assumption.

use std::collections::BTreeSet;

use crate::algebra::computation::{Addresses, Computation, DeclaredEffects};
use crate::algebra::data::{Node, Recipe, RecipeBody, RecipeFragment, Work};
use crate::algebra::error::RecipeError;
use crate::algebra::path::{BodyPath, NodePath};
use crate::buffer::{Address, BufKind, BufRef, WorkspaceBuf, WritableBuf};
use crate::effects::Effects;

/// Where an incoming body lands in a host body: what its formal slots are bound
/// to, where the host's own tables end, and which host nodes the region
/// follows.
pub(crate) struct Landing<'a> {
    pub(crate) input_binding: &'a [BufRef],
    pub(crate) output_binding: &'a [BufRef],
    pub(crate) ws_base: usize,
    pub(crate) node_base: usize,
}

impl Landing<'_> {
    /// Whether this binding fills exactly the slots `body` has.
    pub(crate) fn binds<F: Computation, R>(
        &self,
        body: &RecipeBody<F>,
    ) -> Result<(), RecipeError<R>> {
        if self.input_binding.len() != body.inputs() {
            return Err(RecipeError::BindingArity {
                node: self.node_base,
                kind: BufKind::Input,
                bound: self.input_binding.len(),
                formal: body.inputs(),
            });
        }
        if self.output_binding.len() != body.outputs() {
            return Err(RecipeError::BindingArity {
                node: self.node_base,
                kind: BufKind::Output,
                bound: self.output_binding.len(),
                formal: body.outputs(),
            });
        }
        Ok(())
    }

    /// The same effects on the host's buffers. A declared store that the
    /// binding sends somewhere read-only is refused here, because the binding
    /// is what decided it — the incoming body was right about its own slots.
    pub(crate) fn rebind_effects<R>(
        &self,
        node: usize,
        effects: &Effects,
    ) -> Result<Effects, RecipeError<R>> {
        effects.rebind(&mut |buf| {
            let host = rebind_buf(node, buf, self)?;
            if effects.access(buf).is_some_and(|access| access.writes()) {
                rebind_store(node, buf, self)?;
            }
            Ok(host)
        })
    }
}

/// `is_sink[n]` iff no other node in `nodes` depends on node `n`.
///
/// A dep naming no node of its own body is not this function's to report — the
/// shape proof has already refused one — so it is skipped rather than indexed.
pub(crate) fn sink_mask<F: Computation>(nodes: &[Node<F>]) -> Vec<bool> {
    let mut is_sink = vec![true; nodes.len()];
    for node in nodes {
        for &dep in node.deps() {
            if let Some(sink) = is_sink.get_mut(dep) {
                *sink = false;
            }
        }
    }
    is_sink
}

fn rebind_buf<R>(node: usize, buf: BufRef, into: &Landing<'_>) -> Result<BufRef, RecipeError<R>> {
    let bound = |binding: &[BufRef], index: usize, kind: BufKind| {
        binding
            .get(index)
            .copied()
            .ok_or(RecipeError::BufOutOfRange { node, kind, index })
    };
    match buf {
        BufRef::Input(index) => bound(into.input_binding, index, BufKind::Input),
        BufRef::Output(index) => bound(into.output_binding, index, BufKind::Output),
        BufRef::Workspace(index) => Ok(BufRef::Workspace(index + into.ws_base)),
    }
}

fn rebind_address<R>(
    node: usize,
    address: Address<BufRef>,
    into: &Landing<'_>,
) -> Result<Address<BufRef>, RecipeError<R>> {
    Ok(rebind_buf(node, address.buf, into)?.at(address.byte_offset))
}

/// The host buffer an incoming store lands on, or the refusal saying the
/// binding put it somewhere read-only.
fn rebind_store<R>(
    node: usize,
    buf: BufRef,
    into: &Landing<'_>,
) -> Result<WritableBuf, RecipeError<R>> {
    let host = rebind_buf(node, buf, into)?;
    host.writable().ok_or(RecipeError::WriteToReadOnlyBuffer {
        node,
        kind: host.kind(),
        index: host.index(),
    })
}

fn rebind_writable_address<R>(
    node: usize,
    address: Address<WritableBuf>,
    into: &Landing<'_>,
) -> Result<Address<WritableBuf>, RecipeError<R>> {
    Ok(rebind_store(node, address.buf.readable(), into)?.at(address.byte_offset))
}

/// Rewrite one incoming node's work into the host's index space. `node` is the
/// node's index within the incoming body, which is what a refusal names.
///
/// What the landed work then waits for is the placing caller's to decide, which
/// is why this answers with work rather than a node: a builder derives it from
/// what the work does against the history it is landing in, and a substitution
/// inherits it from the placement it replaces.
///
/// A choice survives landing: only its own bindings are rewritten through the
/// host's; its candidates' internal nodes, workspace and contexts stay in the
/// candidate's private space, to be resolved when the choice is resolved.
pub(crate) fn rebind_work<F: Computation>(
    node: usize,
    incoming: &Work<F>,
    into: &Landing<'_>,
) -> Result<Work<F>, RecipeError<F::Refusal>> {
    Ok(match incoming {
        Work::Call { call, geometry } => {
            let mut call = call.clone();
            let mut refused = None;
            call.rebind(&mut |address| match rebind_address(node, address, into) {
                Ok(landed) => landed,
                Err(refusal) => {
                    refused.get_or_insert(refusal);
                    address
                }
            });
            if let Some(refusal) = refused {
                return Err(refusal);
            }
            Work::Call {
                call,
                geometry: geometry.clone(),
            }
        }
        Work::Memset {
            destination,
            value,
            nbytes,
        } => Work::Memset {
            destination: rebind_writable_address(node, *destination, into)?,
            value: *value,
            nbytes: *nbytes,
        },
        Work::Memcpy {
            source,
            destination,
            nbytes,
        } => Work::Memcpy {
            source: rebind_address(node, *source, into)?,
            destination: rebind_writable_address(node, *destination, into)?,
            nbytes: *nbytes,
        },
        Work::Choice {
            candidates,
            input_binding,
            output_binding,
        } => Work::Choice {
            candidates: candidates.clone(),
            input_binding: input_binding
                .iter()
                .map(|&buf| rebind_buf(node, buf, into))
                .collect::<Result<_, _>>()?,
            output_binding: output_binding
                .iter()
                .map(|&buf| rebind_buf(node, buf, into))
                .collect::<Result<_, _>>()?,
        },
    })
}

/// Move every root handle an incoming body states — its own, and every
/// candidate's below it — onto the merged root context.
pub(crate) fn rebase_root_handles<F: Computation>(body: &mut RecipeBody<F>, remap: &F::RootRemap) {
    let (context, _workspace, nodes) = body.parts_mut();
    F::rebase_body_context(context, remap);
    for node in nodes.iter_mut() {
        if let Work::Choice {
            candidates,
            input_binding,
            output_binding,
        } = node.work().clone()
        {
            let mut candidates = candidates;
            for candidate in &mut candidates {
                rebase_root_handles(candidate, remap);
            }
            *node = Node::new(
                Work::Choice {
                    candidates,
                    input_binding,
                    output_binding,
                },
                node.deps().iter().copied(),
            );
        }
    }
}

/// Rebase the calls of the one body whose context was merged into a host's.
/// Candidates keep their own contexts, so their calls keep their own handles.
pub(crate) fn rebase_calls<F: Computation>(nodes: &mut [Node<F>], remap: &F::BodyRemap) {
    for node in nodes.iter_mut() {
        if let Work::Call { call, geometry } = node.work().clone() {
            let mut call = call;
            F::rebase_call(&mut call, remap);
            *node = Node::new(Work::Call { call, geometry }, node.deps().iter().copied());
        }
    }
}

impl<F: DeclaredEffects> Recipe<F> {
    /// This recipe with the call at `path` replaced by `fragment`'s body, bound
    /// through the buffers the containing body names.
    ///
    /// The graph around it is preserved: whatever the call waited for, the
    /// incoming sources wait for; whatever waited on the call waits on the
    /// incoming sinks. That is sound exactly when what arrives does no more
    /// than the call it replaces did — otherwise those edges were derived from
    /// too little — so the incoming boundary is proved against the replaced
    /// call's own effects first.
    ///
    /// A fragment with no nodes at all removes the call and joins its
    /// dependents to what it waited for.
    ///
    /// The source recipe is untouched — this returns the recipe that has the
    /// substitution, and holds a value that never had it. Moving work is a new
    /// shape, so paths taken before it, at any depth, refuse against the
    /// result.
    pub fn substitute(
        &self,
        path: &NodePath,
        fragment: RecipeFragment<F>,
        input_binding: &[BufRef],
        output_binding: &[BufRef],
    ) -> Result<Recipe<F>, RecipeError<F::Refusal>> {
        let replaced = F::call_effects(self.call_at(path)?);
        // Prove the incoming work before reaching into it.
        fragment.validate_shape()?;
        let boundary = fragment
            .boundary()
            .ok_or(RecipeError::UnprovedFragment)?
            .clone();
        let (incoming_context, mut incoming) = fragment.into_parts();

        let mut context = self.context().clone();
        let root_remap = F::merge_root_context(&mut context, incoming_context);
        rebase_root_handles(&mut incoming, &root_remap);

        let at = path.node();
        let mut incoming = Some(incoming);
        let body = self.update_body_at(path.body(), &mut |body| {
            let incoming = incoming
                .take()
                .expect("a path names one body, so this lands once");
            substitute_into_body(
                body,
                at,
                incoming,
                Boundary {
                    arriving: &boundary,
                    answered_for: &replaced,
                },
                input_binding,
                output_binding,
            )
        })?;
        Recipe::restating(self.contract().clone(), context, body)
    }

    /// The same, for a family whose values anyone may make.
    pub fn replace_call(
        &self,
        path: &NodePath,
        call: F::Call,
    ) -> Result<Recipe<F>, RecipeError<F::Refusal>>
    where
        F: DeclaredEffects<Authorization = ()>,
    {
        self.replacing_call(path, call)
    }

    fn replacing_call(
        &self,
        path: &NodePath,
        call: F::Call,
    ) -> Result<Recipe<F>, RecipeError<F::Refusal>> {
        let answered_for = F::call_effects(self.call_at(path)?);
        answers_for(path.node(), &F::call_effects(&call), &answered_for)?;

        let at = path.node();
        let mut call = Some(call);
        let body = self.update_body_at(path.body(), &mut |body| {
            let mut nodes = body.nodes().to_vec();
            let node = nodes
                .get_mut(at)
                .ok_or_else(|| RecipeError::NoSuchNode { path: path.clone() })?;
            let Work::Call { geometry, .. } = node.work() else {
                return Err(RecipeError::NotACall { path: path.clone() });
            };
            let geometry = geometry.clone();
            *node = Node::new(
                Work::Call {
                    call: call.take().expect("a path names one node"),
                    geometry,
                },
                node.deps().iter().copied(),
            );
            Ok(RecipeBody::with_aliases(
                body.inputs(),
                body.outputs(),
                body.aliases().clone(),
                body.context().clone(),
                body.workspace().to_vec(),
                nodes,
            ))
        })?;
        let replaced = Recipe::holding(
            self.contract().clone(),
            self.context().clone(),
            body,
            self.shape().clone(),
        );
        replaced.validate()?;
        Ok(replaced)
    }
}

impl<F: Computation> Recipe<F> {
    /// This recipe with the workspace buffers `released` gone from the body at
    /// `path`, and everything naming the ones that stay renumbered.
    ///
    /// A body declares buffers for itself, and work that moves takes its own
    /// with it. When a placement leaves, the memory carved for it is memory
    /// nothing can reach — and left in place it would go on being allocated,
    /// because an allocation for a recipe covers every buffer its bodies
    /// declare. So releasing it is a structural edit like any other: whoever
    /// knows the memory was that placement's says which buffers those were, and
    /// this proves nothing still reaches them.
    ///
    /// Only the named body's own declarations move. A candidate below it
    /// carves its own, in its own slot space, and is untouched. The graph is
    /// unchanged, so paths into this recipe still name what they named.
    pub fn releasing_workspace(
        &self,
        path: &BodyPath,
        released: &BTreeSet<usize>,
    ) -> Result<Recipe<F>, RecipeError<F::Refusal>> {
        let body = self.update_body_at(path, &mut |body| {
            let kept: Vec<usize> = (0..body.workspace().len())
                .filter(|index| !released.contains(index))
                .collect();
            let mut moved_to = vec![None; body.workspace().len()];
            for (to, &from) in kept.iter().enumerate() {
                moved_to[from] = Some(to);
            }
            let workspace: Vec<WorkspaceBuf> = kept
                .iter()
                .map(|&index| body.workspace()[index].clone())
                .collect();
            let nodes = body
                .nodes()
                .iter()
                .enumerate()
                .map(|(node, held)| {
                    Ok(Node::new(
                        renumber_workspace(node, held.work(), &moved_to)?,
                        held.deps().iter().copied(),
                    ))
                })
                .collect::<Result<Vec<_>, RecipeError<F::Refusal>>>()?;
            Ok(RecipeBody::with_aliases(
                body.inputs(),
                body.outputs(),
                body.aliases().clone(),
                body.context().clone(),
                workspace,
                nodes,
            ))
        })?;
        let released = Recipe::holding(
            self.contract().clone(),
            self.context().clone(),
            body,
            self.shape().clone(),
        );
        released.validate()?;
        Ok(released)
    }
}

/// One node's work over the workspace that stayed. A node still reaching a
/// released buffer is refused: the memory is gone, and renumbering it to
/// whatever now sits there would be the quiet version of the same mistake.
fn renumber_workspace<F: Computation>(
    node: usize,
    work: &Work<F>,
    moved_to: &[Option<usize>],
) -> Result<Work<F>, RecipeError<F::Refusal>> {
    let moved = |buf: BufRef| -> Result<BufRef, RecipeError<F::Refusal>> {
        match buf {
            BufRef::Workspace(index) => match moved_to.get(index).copied().flatten() {
                Some(to) => Ok(BufRef::Workspace(to)),
                None => Err(RecipeError::WorkspaceStillReached { node, index }),
            },
            buf => Ok(buf),
        }
    };
    Ok(match work {
        Work::Call { call, geometry } => {
            let mut call = call.clone();
            let mut refused = None;
            call.rebind(&mut |address| match moved(address.buf) {
                Ok(landed) => landed.at(address.byte_offset),
                Err(refusal) => {
                    refused.get_or_insert(refusal);
                    address
                }
            });
            if let Some(refusal) = refused {
                return Err(refusal);
            }
            Work::Call {
                call,
                geometry: geometry.clone(),
            }
        }
        Work::Memset {
            destination,
            value,
            nbytes,
        } => Work::Memset {
            destination: writable(moved(destination.readable().buf)?, node)?
                .at(destination.byte_offset),
            value: *value,
            nbytes: *nbytes,
        },
        Work::Memcpy {
            source,
            destination,
            nbytes,
        } => Work::Memcpy {
            source: moved(source.buf)?.at(source.byte_offset),
            destination: writable(moved(destination.readable().buf)?, node)?
                .at(destination.byte_offset),
            nbytes: *nbytes,
        },
        Work::Choice {
            candidates,
            input_binding,
            output_binding,
        } => Work::Choice {
            candidates: candidates.clone(),
            input_binding: input_binding
                .iter()
                .map(|&buf| moved(buf))
                .collect::<Result<_, _>>()?,
            output_binding: output_binding
                .iter()
                .map(|&buf| moved(buf))
                .collect::<Result<_, _>>()?,
        },
    })
}

/// Workspace stays workspace and outputs stay outputs, so a destination that
/// was writable still is. Renumbering says so rather than assuming it.
fn writable<R>(buf: BufRef, node: usize) -> Result<WritableBuf, RecipeError<R>> {
    buf.writable().ok_or(RecipeError::WriteToReadOnlyBuffer {
        node,
        kind: buf.kind(),
        index: buf.index(),
    })
}

/// What arrives at a placement, and what the placement's own edges were derived
/// from.
struct Boundary<'a> {
    arriving: &'a Effects,
    answered_for: &'a Effects,
}

/// Whether the edges derived from `answered_for` also order everything
/// `arriving` does.
fn answers_for<R>(
    node: usize,
    arriving: &Effects,
    answered_for: &Effects,
) -> Result<(), RecipeError<R>> {
    for (buf, needed) in arriving.iter() {
        if !answered_for
            .access(buf)
            .is_some_and(|answered| answered.covers(needed))
        {
            return Err(RecipeError::UndeclaredEffect {
                node,
                kind: buf.kind(),
                index: buf.index(),
            });
        }
    }
    Ok(())
}

/// `body` with the node at `at` replaced by `incoming`'s work. `at` is a node
/// of `body`: the path that reached here proved it names a call.
fn substitute_into_body<F: Computation>(
    body: &RecipeBody<F>,
    at: usize,
    incoming: RecipeBody<F>,
    boundary: Boundary<'_>,
    input_binding: &[BufRef],
    output_binding: &[BufRef],
) -> Result<RecipeBody<F>, RecipeError<F::Refusal>> {
    let replaced = &body.nodes()[at];
    let waited_for: Vec<usize> = replaced.deps().iter().copied().collect();

    let landing = Landing {
        input_binding,
        output_binding,
        ws_base: body.workspace().len(),
        node_base: at,
    };
    landing.binds(&incoming)?;
    answers_for(
        at,
        &landing.rebind_effects(at, boundary.arriving)?,
        boundary.answered_for,
    )?;

    let mut landed = incoming
        .nodes()
        .iter()
        .enumerate()
        .map(|(node, held)| {
            let work = rebind_work(node, held.work(), &landing)?;
            let deps: std::collections::BTreeSet<usize> = if held.deps().is_empty() {
                waited_for.iter().copied().collect()
            } else {
                held.deps().iter().map(|dep| dep + at).collect()
            };
            Ok(Node::new(work, deps))
        })
        .collect::<Result<Vec<_>, RecipeError<F::Refusal>>>()?;

    let mut context = body.context().clone();
    let body_remap = F::merge_body_context(&mut context, incoming.context().clone());
    rebase_calls(&mut landed, &body_remap);

    let sinks: Vec<usize> = sink_mask(incoming.nodes())
        .iter()
        .enumerate()
        .filter(|&(_, &sink)| sink)
        .map(|(node, _)| at + node)
        .collect();
    let placed = landed.len();
    let follow = |dep: usize| -> Vec<usize> {
        match dep {
            dep if dep < at => vec![dep],
            dep if dep == at && placed == 0 => waited_for.clone(),
            dep if dep == at => sinks.clone(),
            dep => vec![dep - 1 + placed],
        }
    };

    let mut nodes = Vec::with_capacity(body.nodes().len() - 1 + placed);
    nodes.extend(body.nodes()[..at].iter().cloned());
    nodes.append(&mut landed);
    for held in &body.nodes()[at + 1..] {
        nodes.push(Node::new(
            held.work().clone(),
            held.deps().iter().copied().flat_map(follow),
        ));
    }

    let mut workspace = body.workspace().to_vec();
    workspace.extend(incoming.workspace().to_vec());
    Ok(RecipeBody::with_aliases(
        body.inputs(),
        body.outputs(),
        body.aliases().clone(),
        context,
        workspace,
        nodes,
    ))
}
