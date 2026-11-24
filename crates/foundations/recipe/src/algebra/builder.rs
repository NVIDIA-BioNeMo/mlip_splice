// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Assembling a recipe, one placement at a time.
//!
//! A builder owns the tables a body is made of — its workspace, its nodes, and
//! the family data those name — and owns them together, so nothing it hands out
//! can outlive the tables it indexes.
//!
//! What it adds on top of the value is ordering, and ordering comes from one
//! place: every placement states what it does to the buffers around it. The
//! statement is proved against what the placement is actually handed, spent on
//! the dependency edges it justifies, and then discharged — a recipe keeps the
//! edges. Nothing recovers the statement afterwards, which is why a family that
//! means to edit a built recipe by effect keeps its own copy
//! ([`crate::algebra::DeclaredEffects`]).
//!
//! Landing a whole region is the other thing a builder does, and it orders
//! differently on purpose. A region arrives with its own internal edges already
//! settled, and what it must follow *here* is what the caller says it follows.
//! So a landing states its prerequisites and the builder adds nothing: it does
//! not re-derive the region's edges from what its work does, and the region
//! does not join the history that orders later placements. A caller that wants
//! a placement to follow a region names its sinks. That is the whole of it, and
//! it is what keeps a recipe assembled from regions identical to the one the
//! same regions produced before.

use crate::PortAliases;
use crate::algebra::computation::{Addresses, BytePortContract, Computation, Geometry};
use crate::algebra::data::{Node, ProvedRecipe, Recipe, RecipeBody, RecipeFragment, Work};
use crate::algebra::error::RecipeError;
use crate::algebra::substitute::{
    Landing, rebase_calls, rebase_root_handles, rebind_work, sink_mask,
};
use crate::buffer::WorkspaceBuf;
use crate::buffer::{Address, BufRef, WritableBuf};
use crate::effects::{Access, Effects};
use crate::handles::{Buf, NodeId, WorkspaceId, Writable};
use crate::hazards::Hazards;

pub struct RecipeBuilder<F: Computation> {
    inputs: usize,
    outputs: usize,
    aliases: PortAliases,
    context: F::RootContext,
    body_context: F::BodyContext,
    workspace: Vec<WorkspaceBuf>,
    nodes: Vec<Node<F>>,
    /// What everything placed here does to the slots this body is bound
    /// through — its own workspace is private, so nothing about that. This is
    /// what a choice over the result is held to.
    boundary: Effects,
    hazards: Hazards,
}

impl<F: Computation> RecipeBuilder<F> {
    /// A builder for a body bound through `inputs` and `outputs` formal slots.
    ///
    /// The arity is stated up front because it is the interface: whatever binds
    /// the result has to fill exactly these slots, and a body that discovered
    /// its own arity from the slots it happened to name could not tell an
    /// unused slot from one that does not exist.
    /// A builder for a family whose values anyone may make.
    pub fn new(inputs: usize, outputs: usize) -> Self
    where
        F: Computation<Authorization = ()>,
    {
        RecipeBuilder::new_authorized(inputs, outputs, &())
    }

    /// A builder that may make one family's values, holding what says so from
    /// here until it finishes.
    ///
    /// The authorization is taken once, at the start, because everything this
    /// builder is handed afterwards — a call, a context, a landing — becomes
    /// part of the value it finishes. A builder nobody authorized cannot be
    /// opened, so there is nothing to check again later.
    pub fn new_authorized(inputs: usize, outputs: usize, authorization: &F::Authorization) -> Self {
        let _ = authorization;
        RecipeBuilder::opened(inputs, outputs)
    }

    fn opened(inputs: usize, outputs: usize) -> Self {
        RecipeBuilder {
            inputs,
            outputs,
            aliases: PortAliases::empty(),
            context: F::RootContext::default(),
            body_context: F::BodyContext::default(),
            workspace: Vec::new(),
            nodes: Vec::new(),
            boundary: Effects::none(),
            hazards: Hazards::default(),
        }
    }

    pub fn aliases(mut self, aliases: PortAliases) -> Self {
        self.aliases = aliases;
        self
    }

    /// The family data this recipe holds once, for every body below it.
    pub fn root_context(&self) -> &F::RootContext {
        &self.context
    }

    pub fn root_context_mut(&mut self) -> &mut F::RootContext {
        &mut self.context
    }

    /// The family data the body being assembled holds for its own calls.
    pub fn body_context(&self) -> &F::BodyContext {
        &self.body_context
    }

    pub fn body_context_mut(&mut self) -> &mut F::BodyContext {
        &mut self.body_context
    }

    /// How many nodes this builder holds, which is the index the next
    /// placement takes — what a producer that refuses one of its own says it
    /// refused.
    pub fn placed(&self) -> usize {
        self.nodes.len()
    }

    pub fn scratch(&mut self, nbytes: usize) -> WorkspaceId {
        self.workspace.push(WorkspaceBuf { nbytes, init: None });
        WorkspaceId::from_index(self.workspace.len() - 1)
    }

    pub fn scratch_init(&mut self, bytes: impl Into<crate::Blob>) -> WorkspaceId {
        let bytes = bytes.into();
        self.workspace.push(WorkspaceBuf {
            nbytes: bytes.len(),
            init: Some(bytes),
        });
        WorkspaceId::from_index(self.workspace.len() - 1)
    }

    /// A computation, to be placed once it has said what it does, in the shape
    /// each of its launches runs in.
    pub fn call(&mut self, call: F::Call, geometry: Vec<Geometry>) -> Placement<'_, F> {
        Placement {
            builder: self,
            call,
            geometry,
            effects: Vec::new(),
            extra_deps: Vec::new(),
        }
    }

    pub fn memset(&mut self, destination: Address<Writable>, value: u8, nbytes: usize) -> NodeId {
        self.place(Work::Memset {
            destination: WritableBuf::from(destination.buf).at(destination.byte_offset),
            value,
            nbytes,
        })
    }

    /// A copy reads its source and writes its destination, so an input is a
    /// legal source here and has no spelling as a destination.
    pub fn memcpy(
        &mut self,
        source: Address<Buf>,
        destination: Address<Writable>,
        nbytes: usize,
    ) -> NodeId {
        self.place(Work::Memcpy {
            source: source.readable(),
            destination: WritableBuf::from(destination.buf).at(destination.byte_offset),
            nbytes,
        })
    }

    /// A choice between fragments. A candidate is a [`RecipeFragment`] — work,
    /// the root data it names, and what it proved it does — because the binding
    /// below is its interface, not any ports the candidate might once have
    /// stated.
    pub fn choice(&mut self, candidates: Vec<RecipeFragment<F>>) -> Choice<'_, F> {
        Choice {
            builder: self,
            candidates,
            input_binding: Vec::new(),
            output_binding: Vec::new(),
            effects: Vec::new(),
            from_candidates: false,
            extra_deps: Vec::new(),
        }
    }

    /// Land `sub` in this builder, re-indexing it into the builder's slot
    /// space, and answer with the nodes nothing in `sub` consumed.
    ///
    /// The region keeps the order it already had, and follows `external` where
    /// it had nothing to follow. Nothing else is added: a region is work whose
    /// ordering was settled where it was built, and only the caller knows what
    /// it must come after here. Landing therefore asks the region for no proof
    /// of what it does — a store the binding sends somewhere read-only is still
    /// refused, because that is a fact about the addresses, not about a
    /// statement.
    ///
    /// Every table this builder owns moves in the same step or not at all, so a
    /// caller that reports a refusal and keeps building is holding the builder
    /// it had before it asked.
    pub fn splice(
        &mut self,
        sub: RecipeFragment<F>,
        input_binding: &[BufRef],
        output_binding: &[BufRef],
        external: &[NodeId],
    ) -> Result<Vec<NodeId>, RecipeError<F::Refusal>> {
        sub.validate_shape()?;
        let landed = self.rebind_into_place(&sub, input_binding, output_binding, external)?;
        let boundary = sub.boundary().cloned();
        let sources = sub.body().nodes().to_vec();
        let (context, body) = sub.into_parts();
        let root_remap = F::merge_root_context(&mut self.context, context);
        let mut incoming = RecipeBody::with_aliases(
            body.inputs(),
            body.outputs(),
            body.aliases().clone(),
            body.context().clone(),
            body.workspace().to_vec(),
            landed,
        );
        rebase_root_handles(&mut incoming, &root_remap);
        self.account_for(&boundary, input_binding, output_binding)?;
        Ok(self.absorb(incoming, &sources))
    }

    /// Land one piece of work whose ordering was settled elsewhere, following
    /// exactly the nodes stated.
    ///
    /// The single-node counterpart of [`RecipeBuilder::splice_body`], and it
    /// orders the same way: the edges come from the caller rather than from
    /// what the work does, because the caller is re-laying work whose edges
    /// were derived once already. Work assembled here for the first time goes
    /// through [`RecipeBuilder::call`], which derives them from what it proves.
    pub fn land(&mut self, work: Work<F>, follows: impl IntoIterator<Item = usize>) -> NodeId {
        for buf in reaches(&work) {
            self.answer_for(buf);
        }
        let at = self.nodes.len();
        self.nodes.push(Node::new(work, follows));
        NodeId::from_index(at)
    }

    /// Take a landed region into what this builder answers for at its own ports.
    ///
    /// A region that proved what it does is accounted for by exactly that,
    /// rebound onto the host's buffers. One that did not is accounted for by its
    /// whole binding: rebinding sends every slot it names to one of those
    /// buffers and its own workspace is private, so the buffers it was bound to
    /// are everything it can reach — which is the strongest thing a landing
    /// proves on its own.
    fn account_for(
        &mut self,
        boundary: &Option<Effects>,
        input_binding: &[BufRef],
        output_binding: &[BufRef],
    ) -> Result<(), RecipeError<F::Refusal>> {
        match boundary {
            Some(boundary) => {
                let landing = Landing {
                    input_binding,
                    output_binding,
                    ws_base: self.workspace.len(),
                    node_base: self.nodes.len(),
                };
                for (buf, access) in landing.rebind_effects(self.nodes.len(), boundary)?.iter() {
                    if !matches!(buf, BufRef::Workspace(_)) {
                        self.boundary.insert(buf, access);
                    }
                }
            }
            None => {
                for &buf in input_binding.iter().chain(output_binding) {
                    self.answer_for(buf);
                }
            }
        }
        Ok(())
    }

    /// Answer for whatever a buffer admits, which is what work that stated
    /// nothing finer might do to it. Workspace is this body's own, so nothing
    /// binding it needs to hear about it.
    fn answer_for(&mut self, buf: BufRef) {
        if matches!(buf, BufRef::Workspace(_)) {
            return;
        }
        let access = match buf.writable() {
            Some(_) => Access::ReadWrite,
            None => Access::Read,
        };
        self.boundary.insert(buf, access);
    }

    /// The same, for a body whose calls already name the root data this builder
    /// is assembling — a candidate being resolved inside the recipe that offered
    /// it.
    ///
    /// Nothing about the root context moves, because nothing about it has to: a
    /// candidate's handles were already rebased onto this recipe's when the
    /// choice took it in. Only the body's own data merges.
    pub fn splice_body(
        &mut self,
        sub: &RecipeBody<F>,
        input_binding: &[BufRef],
        output_binding: &[BufRef],
        external: &[NodeId],
    ) -> Result<Vec<NodeId>, RecipeError<F::Refusal>> {
        let fragment = RecipeFragment::unproved(self.context.clone(), sub.clone());
        crate::algebra::validate::validate_shape(sub, &self.context)?;
        let landed = self.rebind_into_place(&fragment, input_binding, output_binding, external)?;
        self.account_for(&None, input_binding, output_binding)?;
        let incoming = RecipeBody::with_aliases(
            sub.inputs(),
            sub.outputs(),
            sub.aliases().clone(),
            sub.context().clone(),
            sub.workspace().to_vec(),
            landed,
        );
        Ok(self.absorb(incoming, sub.nodes()))
    }

    /// `sub`'s nodes in this builder's index space, keeping the order they
    /// already had. Nothing is mutated, so a refusal here is one that happened
    /// before the landing began.
    fn rebind_into_place(
        &self,
        sub: &RecipeFragment<F>,
        input_binding: &[BufRef],
        output_binding: &[BufRef],
        external: &[NodeId],
    ) -> Result<Vec<Node<F>>, RecipeError<F::Refusal>> {
        let external: Vec<usize> = external.iter().map(|node| node.index()).collect();
        let node_base = self.nodes.len();
        let landing = Landing {
            input_binding,
            output_binding,
            ws_base: self.workspace.len(),
            node_base,
        };
        landing.binds(sub.body())?;

        sub.body()
            .nodes()
            .iter()
            .enumerate()
            .map(|(node, held)| {
                let work = rebind_work(node, held.work(), &landing)?;
                let follows: std::collections::BTreeSet<usize> = if held.deps().is_empty() {
                    external.iter().copied().collect()
                } else {
                    held.deps().iter().map(|dep| dep + node_base).collect()
                };
                Ok(Node::new(work, follows))
            })
            .collect()
    }

    /// Commit an already-rebound body. `sources` are the incoming body's own
    /// nodes, whose dependency edges say which of them nothing else consumes.
    fn absorb(&mut self, incoming: RecipeBody<F>, sources: &[Node<F>]) -> Vec<NodeId> {
        let node_base = self.nodes.len();
        let mut incoming = incoming;
        let body_remap = F::merge_body_context(&mut self.body_context, incoming.context().clone());
        let mut landed = incoming.nodes().to_vec();
        rebase_calls(&mut landed, &body_remap);
        incoming.set_nodes(Vec::new());

        self.workspace.extend(incoming.workspace().to_vec());
        self.nodes.extend(landed);
        sink_mask(sources)
            .iter()
            .enumerate()
            .filter(|&(_, &sink)| sink)
            .map(|(node, _)| NodeId::from_index(node_base + node))
            .collect()
    }

    /// Bring a fragment's root data into this builder's, so what lands here
    /// names this builder's context and not the one it was built against.
    fn import_fragment(&mut self, fragment: RecipeFragment<F>) -> RecipeBody<F> {
        let (context, mut body) = fragment.into_parts();
        let root_remap = F::merge_root_context(&mut self.context, context);
        rebase_root_handles(&mut body, &root_remap);
        body
    }

    /// Append work whose effects are what it is, and order it against
    /// everything before it.
    fn place(&mut self, work: Work<F>) -> NodeId {
        let effects = work
            .memory_effects()
            .expect("memory work states its own effects");
        self.commit(work, effects, &[])
    }

    fn commit(&mut self, work: Work<F>, effects: Effects, extra: &[NodeId]) -> NodeId {
        let extra: Vec<usize> = extra.iter().map(|node| node.index()).collect();
        let deps = self.hazards.deps(&effects, &extra);
        let at = self.nodes.len();
        self.hazards.commit(at, &effects);
        for (buf, access) in effects.iter() {
            if !matches!(buf, BufRef::Workspace(_)) {
                self.boundary.insert(buf, access);
            }
        }
        self.nodes.push(Node::new(work, deps));
        NodeId::from_index(at)
    }

    /// The recipe this builder assembled, over the ports it is bound through,
    /// proved against them.
    ///
    /// The contract arrives here rather than at `new` because a producer
    /// usually settles its buffer extents while laying the work down, and this
    /// is the point where all of them are known. The arity was stated at `new`,
    /// because it is the interface rather than a measurement.
    pub fn finish(self, contract: F::Contract) -> Result<ProvedRecipe<F>, RecipeError<F::Refusal>> {
        let fragment = self.aliases(contract.aliases()).finish_fragment();
        let boundary = fragment
            .boundary()
            .expect("a builder proves what it placed")
            .clone();
        let (context, body) = fragment.into_parts();
        Ok(ProvedRecipe::new(
            Recipe::restating(contract, context, body)?,
            boundary,
        ))
    }

    /// Just the work this builder assembled, with no ports stated, and the
    /// proof of what each node of it does.
    ///
    /// A body is bound through whatever selects it — a choice's bindings, or a
    /// landing into another builder — so it has no contract of its own to
    /// state. Callers that need one take it from here instead of finishing a
    /// recipe over invented ports and unwrapping it: a contract no one can
    /// state truthfully is one no one should have to write down.
    pub fn finish_fragment(self) -> RecipeFragment<F> {
        RecipeFragment::proved(
            self.context,
            RecipeBody::with_aliases(
                self.inputs,
                self.outputs,
                self.aliases,
                self.body_context,
                self.workspace,
                self.nodes,
            ),
            self.boundary,
        )
    }
}

/// Every buffer `work` can reach: the addresses a call is handed, the buffers
/// memory work names, and the binding a choice selects through.
fn reaches<F: Computation>(work: &Work<F>) -> Vec<BufRef> {
    let mut reached = Vec::new();
    match work {
        Work::Call { call, .. } => call.addresses(&mut |address| reached.push(address.buf)),
        Work::Memset { destination, .. } => reached.push(destination.readable().buf),
        Work::Memcpy {
            source,
            destination,
            ..
        } => {
            reached.push(source.buf);
            reached.push(destination.readable().buf);
        }
        Work::Choice {
            input_binding,
            output_binding,
            ..
        } => reached.extend(input_binding.iter().chain(output_binding).copied()),
    }
    reached
}

/// What every candidate does to the buffers a choice binds them to: their own
/// boundaries, taken through the binding and joined slot by slot.
///
/// A candidate's own workspace is its own — nothing outside can reach it — so
/// it contributes nothing here. A candidate nobody proved contributes nothing
/// either, which costs nothing: placing one is refused.
fn joined<F: Computation>(
    candidates: &[RecipeFragment<F>],
    input_binding: &[BufRef],
    output_binding: &[BufRef],
) -> Vec<(BufRef, Access)> {
    let mut joined = Vec::new();
    for candidate in candidates {
        let Some(boundary) = candidate.boundary() else {
            continue;
        };
        for (buf, access) in boundary.iter() {
            let landed = match buf {
                BufRef::Input(slot) => input_binding.get(slot).copied(),
                BufRef::Output(slot) => output_binding.get(slot).copied(),
                BufRef::Workspace(_) => None,
            };
            if let Some(landed) = landed {
                joined.push((landed, access));
            }
        }
    }
    joined
}

/// What a placement says it does, proved against what it is: every buffer it
/// names has to be one it is handed — a claim about memory it never receives
/// orders nothing real — and a store has to land somewhere that accepts one.
fn proves<R>(node: usize, stated: &Effects, handed: &[BufRef]) -> Result<(), RecipeError<R>> {
    for (buf, access) in stated.iter() {
        if !handed.contains(&buf) {
            return Err(RecipeError::EffectNotHanded {
                node,
                kind: buf.kind(),
                index: buf.index(),
            });
        }
        if access.writes() && buf.writable().is_none() {
            return Err(RecipeError::WriteToReadOnlyBuffer {
                node,
                kind: buf.kind(),
                index: buf.index(),
            });
        }
    }
    Ok(())
}

/// A computation waiting to be placed: what it is, how it is launched, and what
/// it does to the buffers around it.
#[must_use]
pub struct Placement<'a, F: Computation> {
    builder: &'a mut RecipeBuilder<F>,
    call: F::Call,
    geometry: Vec<Geometry>,
    effects: Vec<(BufRef, Access)>,
    extra_deps: Vec<NodeId>,
}

impl<F: Computation> Placement<'_, F> {
    pub fn reads(mut self, bufs: impl IntoIterator<Item = Buf>) -> Self {
        self.effects.extend(
            bufs.into_iter()
                .map(|buf| (BufRef::from(buf), Access::Read)),
        );
        self
    }

    pub fn writes(mut self, bufs: impl IntoIterator<Item = Buf>) -> Self {
        self.effects.extend(
            bufs.into_iter()
                .map(|buf| (BufRef::from(buf), Access::Write)),
        );
        self
    }

    pub fn read_writes(mut self, bufs: impl IntoIterator<Item = Buf>) -> Self {
        self.effects.extend(
            bufs.into_iter()
                .map(|buf| (BufRef::from(buf), Access::ReadWrite)),
        );
        self
    }

    pub fn after(mut self, deps: &[NodeId]) -> Self {
        self.extra_deps.extend_from_slice(deps);
        self
    }

    /// Place it, having proved its statement against the addresses it is
    /// handed.
    pub fn add(self) -> Result<NodeId, RecipeError<F::Refusal>> {
        let Placement {
            builder,
            call,
            geometry,
            effects,
            extra_deps,
        } = self;
        let effects = Effects::of(effects);
        let mut addresses = Vec::new();
        call.addresses(&mut |address| addresses.push(address.buf));
        proves(builder.nodes.len(), &effects, &addresses)?;
        Ok(builder.commit(Work::Call { call, geometry }, effects, &extra_deps))
    }
}

/// A choice under construction: what may run, which buffers it is bound
/// through, and what it does to them.
///
/// Binding and effects are two statements, because they are two facts. A
/// uniform call ABI passes every sibling slot's address whether or not the work
/// behind it touches that memory, so which buffer a slot selects says nothing
/// about what is done through it — and saying it twice is how a choice comes to
/// be ordered against buffers it never reaches.
#[must_use]
pub struct Choice<'a, F: Computation> {
    builder: &'a mut RecipeBuilder<F>,
    candidates: Vec<RecipeFragment<F>>,
    input_binding: Vec<BufRef>,
    output_binding: Vec<BufRef>,
    effects: Vec<(BufRef, Access)>,
    /// Whether the effects are the candidates' own, joined slot by slot.
    from_candidates: bool,
    extra_deps: Vec<NodeId>,
}

impl<F: Computation> Choice<'_, F> {
    /// The buffers the candidates' formal input slots select, in slot order.
    pub fn inputs(mut self, bufs: impl IntoIterator<Item = Buf>) -> Self {
        self.input_binding
            .extend(bufs.into_iter().map(BufRef::from));
        self
    }

    /// The buffers their formal output slots select, in slot order. A slot may
    /// select a buffer nothing may write — the address is still passed — and
    /// what may be stored through it is decided by what is done, not by what is
    /// bound.
    pub fn outputs(mut self, bufs: impl IntoIterator<Item = Buf>) -> Self {
        self.output_binding
            .extend(bufs.into_iter().map(BufRef::from));
        self
    }

    /// State that this choice does `access` to each of `bufs`.
    pub fn does(mut self, bufs: impl IntoIterator<Item = Buf>, access: Access) -> Self {
        self.effects
            .extend(bufs.into_iter().map(|buf| (BufRef::from(buf), access)));
        self
    }

    /// State what this choice does from what its candidates proved they do.
    ///
    /// Whichever candidate runs is what runs, so a choice does the join of what
    /// all of them do, slot by slot, through the binding: a slot no candidate
    /// touches is carried rather than read or written, one that is only read is
    /// read, and one a candidate reads while another writes is both. Nothing is
    /// over-declared, so nothing is ordered against memory the work never
    /// reaches — and nothing is under-declared, because the join is over every
    /// candidate.
    ///
    /// The binding decides where each slot lands, so this is taken at `add`
    /// rather than where it is asked for, and may be stated in any order.
    pub fn does_what_its_candidates_do(mut self) -> Self {
        self.from_candidates = true;
        self
    }

    pub fn after(mut self, deps: &[NodeId]) -> Self {
        self.extra_deps.extend_from_slice(deps);
        self
    }

    /// Place it, having proved that what it says it does answers for what every
    /// candidate does.
    pub fn add(self) -> Result<NodeId, RecipeError<F::Refusal>> {
        let Choice {
            builder,
            candidates,
            input_binding,
            output_binding,
            mut effects,
            from_candidates,
            extra_deps,
        } = self;
        let node = builder.nodes.len();
        if from_candidates {
            effects.extend(joined(&candidates, &input_binding, &output_binding));
        }
        let effects = Effects::of(effects);
        let bound: Vec<BufRef> = input_binding
            .iter()
            .chain(&output_binding)
            .copied()
            .collect();
        proves(node, &effects, &bound)?;

        for candidate in &candidates {
            let boundary = candidate.boundary().ok_or(RecipeError::UnprovedFragment)?;
            for (buf, needed) in boundary.iter() {
                let landed = match buf {
                    BufRef::Input(slot) => input_binding.get(slot).copied(),
                    BufRef::Output(slot) => output_binding.get(slot).copied(),
                    BufRef::Workspace(_) => continue,
                };
                let Some(landed) = landed else { continue };
                if !effects
                    .access(landed)
                    .is_some_and(|declared| declared.covers(needed))
                {
                    return Err(RecipeError::UndeclaredEffect {
                        node,
                        kind: landed.kind(),
                        index: landed.index(),
                    });
                }
            }
        }

        let aliases = candidates
            .first()
            .map(|candidate| candidate.aliases().clone())
            .unwrap_or_default();
        if candidates
            .iter()
            .any(|candidate| candidate.aliases() != &aliases)
        {
            return Err(RecipeError::CandidateAliasesDisagree { node });
        }
        let candidates = candidates
            .into_iter()
            .map(|candidate| builder.import_fragment(candidate))
            .collect();
        Ok(builder.commit(
            Work::Choice {
                candidates,
                input_binding,
                output_binding,
            },
            effects,
            &extra_deps,
        ))
    }
}
