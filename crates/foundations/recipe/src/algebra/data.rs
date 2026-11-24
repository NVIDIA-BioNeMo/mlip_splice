// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a recipe is: a contract, the family data it carries, and the work it
//! runs.
//!
//! A finished recipe has been proved. There is no way to hold one that was not:
//! every way in validates, every functional edit validates what it produces,
//! and nothing a caller can reach hands out the interior mutably. A body, a
//! node and a fragment are ordinary values a producer assembles freely — they
//! become a recipe only by passing through a proof.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::PortAliases;
use crate::algebra::computation::{Computation, Geometry};
use crate::algebra::error::RecipeError;
use crate::algebra::executable::Executable;
use crate::buffer::WorkspaceBuf;
use crate::buffer::{Address, BufRef, WritableBuf};
use crate::effects::{Access, Effects};

/// Which recipe a path belongs to.
///
/// A recipe mints one when its shape comes into being, and hands it to every
/// path it gives out. A transformation that leaves the shape alone carries it
/// forward, so a path survives compiling a recipe from one representation to
/// the next; one that moves work mints a new shape, so paths taken before it
/// refuse afterwards instead of quietly naming whatever moved into their place.
///
/// It is an identity, not a value: two recipes that are equal were still built
/// separately, and a path from one is not evidence about the other. So it takes
/// no part in equality or in bytes, and cloning a recipe keeps it — a clone is
/// the same recipe.
///
/// Identity is one allocation, and two of them are the same exactly when they
/// are the same allocation. A counter would be an identity only until it wrapped
/// and a number came round again; this cannot come round, because everything
/// that could compare two shapes holds one of them alive — the recipe, and every
/// path it handed out — so no address it could be confused with is free to be
/// reused.
#[derive(Clone)]
pub struct Shape(Arc<()>);

impl Shape {
    fn mint() -> Shape {
        Shape(Arc::new(()))
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Shape {}

impl std::hash::Hash for Shape {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.0) as usize).hash(state);
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Shape({:p})", Arc::as_ptr(&self.0))
    }
}

/// A recipe: the ports it is bound through, the family data its bodies draw
/// from, and the work it runs — proved against each other.
///
/// The parts are private because they only mean anything together. A body's
/// calls name the context beside them, and the work addresses buffers the
/// contract sizes, so a caller that walked off with one part would hold handles
/// and offsets naming things that did not come along. Reading any part is free;
/// separating them is [`Recipe::into_parts`], which hands over all three at
/// once, or [`Recipe::into_fragment`], which says in its name which one it
/// drops.
///
/// Taking a finished one apart does not compile, and the first of these is the
/// extraction the model exists to prevent — a body whose handles name a context
/// left behind:
///
/// ```compile_fail
/// # use recipe::{RecipeBody, RecipeSignature};
/// # use recipe::algebra::{CompileRequests, Executable, Recipe};
/// let recipe = Recipe::<Executable>::stating_edges(
///     RecipeSignature::empty(),
///     CompileRequests::default(),
///     RecipeBody::default(),
/// ).unwrap();
/// let body = recipe.body;
/// ```
///
/// ```compile_fail
/// # use recipe::{RecipeBody, RecipeSignature};
/// # use recipe::algebra::{CompileRequests, Executable, Recipe};
/// let recipe = Recipe::<Executable>::stating_edges(
///     RecipeSignature::empty(),
///     CompileRequests::default(),
///     RecipeBody::default(),
/// ).unwrap();
/// let context = recipe.context;
/// ```
///
/// ```compile_fail
/// # use recipe::{RecipeBody, RecipeSignature};
/// # use recipe::algebra::{CompileRequests, Executable, Recipe};
/// let recipe = Recipe::<Executable> {
///     contract: RecipeSignature::empty(),
///     context: CompileRequests::default(),
///     body: RecipeBody::default(),
/// };
/// ```
///
/// Nor does reaching into a finished recipe to change what it holds:
///
/// ```compile_fail
/// # use recipe::{RecipeBody, RecipeSignature};
/// # use recipe::algebra::{CompileRequests, Executable, Recipe};
/// let recipe = Recipe::<Executable>::stating_edges(
///     RecipeSignature::empty(),
///     CompileRequests::default(),
///     RecipeBody::default(),
/// ).unwrap();
/// recipe.body().nodes().clear();
/// ```
///
/// While building one, and taking the whole of it apart, do:
///
/// ```
/// # use recipe::RecipeSignature;
/// # use recipe::algebra::{Executable, RecipeBuilder};
/// let recipe = RecipeBuilder::<Executable>::new(0, 0)
///     .finish(RecipeSignature::empty())
///     .unwrap()
///     .into_recipe();
/// let (ports, context, body) = recipe.into_parts();
/// assert!(ports.inputs().is_empty() && context.is_empty() && body.nodes().is_empty());
/// ```
#[derive(Clone)]
pub struct Recipe<F: Computation = Executable> {
    contract: F::Contract,
    context: F::RootContext,
    body: RecipeBody<F>,
    shape: Shape,
}

/// Whether anyone stated what a fragment's work does at its own ports.
///
/// Absent evidence and evidence that says nothing are different facts, and a
/// choice needs to tell them apart: a fragment that proved it touches nothing
/// may be a candidate anywhere, and one nobody proved may not be a candidate at
/// all. So this is a state and not an empty set, it is private, and the only
/// ways to reach `Proved` are to have built the work or to have stated the
/// boundary of work acquired from somewhere else.
///
/// It is a boundary and not a per-node account because a boundary is all
/// anything asks for. Landing a region orders it by what the caller says it
/// follows, not by what its nodes do — see [`crate::algebra::RecipeBuilder`] —
/// so nothing needs to know which node inside does which part.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Proof {
    Proved(Effects),
    Unproved,
}

/// Work plus the root data it names: everything a recipe is except the ports it
/// is bound through.
///
/// This is what a candidate or a substituted sub-recipe actually is. A body's
/// context holds handles into root data, so a body cannot travel without it —
/// carrying them together is what keeps those handles meaning something after
/// the fragment lands somewhere else, where they are rebased onto the host's.
///
/// It states no contract, and that is the model rather than an omission: a
/// fragment's interface is whatever selects it — a choice's bindings, or a
/// substitution's — so ports of its own would be a second, disagreeable copy of
/// a fact the binding already fixes. What it does state is its body's *arity*,
/// which is how many slots that binding has to fill, and filling exactly that
/// many is what makes a binding complete.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecipeFragment<F: Computation = Executable> {
    context: F::RootContext,
    body: RecipeBody<F>,
    /// What this work does at its own ports, as proved where it was built — or
    /// that nobody proved it.
    ///
    /// Construction evidence, not recipe state: it travels with a fragment so
    /// that a choice can hold its candidates to what it declared, and it is
    /// spent there. A recipe keeps the edges and not this.
    proof: Proof,
}

/// One body of work: the formal slots it is bound through, the family data its
/// own calls name, the buffers it declares for itself, and the nodes it runs.
///
/// Its fields are read-only from outside because a body is reachable from a
/// finished recipe, and a recipe that was proved must stay the thing that was
/// proved. Building one states every part at once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecipeBody<F: Computation = Executable> {
    inputs: usize,
    outputs: usize,
    aliases: PortAliases,
    context: F::BodyContext,
    workspace: Vec<WorkspaceBuf>,
    // A node's deps reference strictly-earlier node indices, so the natural
    // index order is a valid topological order.
    nodes: Vec<Node<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node<F: Computation = Executable> {
    work: Work<F>,
    deps: BTreeSet<usize>,
}

/// What one node does.
///
/// Three of the four are pure structure over buffers, which is why they are
/// spelled out here rather than left to a family: moving and clearing bytes,
/// and choosing between bodies, mean the same thing whatever is being
/// orchestrated. The fourth is the family's, and the core sees the addresses it
/// is handed, the shape of each launch it is launched in, and the effects it
/// declares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Work<F: Computation = Executable> {
    Call {
        call: F::Call,
        /// One shape per launch the call states, in the call's own launch
        /// order. A kernel states one; a composite states one each.
        geometry: Vec<Geometry>,
    },
    Memset {
        destination: Address<WritableBuf>,
        value: u8,
        nbytes: usize,
    },
    /// Readability of the source and writability of the destination are what
    /// their types say, so an input is a legal source and never a destination.
    Memcpy {
        source: Address<BufRef>,
        destination: Address<WritableBuf>,
        nbytes: usize,
    },
    Choice {
        candidates: Vec<RecipeBody<F>>,
        input_binding: Vec<BufRef>,
        output_binding: Vec<BufRef>,
    },
}

impl<F: Computation> Work<F> {
    /// What this node does to the buffers around it, when the work itself is
    /// the whole statement of that.
    ///
    /// Moving and clearing bytes say what they do by being what they are. A
    /// call and a choice do not: what a computation does through an address it
    /// is handed is the producer's to state, at the placement, where it is
    /// proved and then spent on the dependency edges it justifies. A recipe
    /// keeps those edges, not the statement — so this answers `None` there
    /// rather than inventing one.
    pub fn memory_effects(&self) -> Option<Effects> {
        match self {
            Work::Call { .. } | Work::Choice { .. } => None,
            Work::Memset { destination, .. } => {
                Some(Effects::of([(destination.buf.readable(), Access::Write)]))
            }
            Work::Memcpy {
                source,
                destination,
                ..
            } => Some(Effects::of([
                (source.buf, Access::Read),
                (destination.buf.readable(), Access::Write),
            ])),
        }
    }
}

impl<F: Computation> Default for RecipeBody<F> {
    fn default() -> Self {
        RecipeBody {
            inputs: 0,
            outputs: 0,
            aliases: PortAliases::empty(),
            context: F::BodyContext::default(),
            workspace: Vec::new(),
            nodes: Vec::new(),
        }
    }
}

impl<F: Computation> RecipeBody<F> {
    /// The body `nodes` are, bound through `inputs` and `outputs` formal slots,
    /// over the family data its calls name.
    ///
    /// A body assembled this way states edges nobody derived, which is why it
    /// only becomes a recipe through [`Recipe::stating_edges`] — where that
    /// trust is written down — or through a builder that derived them.
    pub fn new(
        inputs: usize,
        outputs: usize,
        context: F::BodyContext,
        workspace: Vec<WorkspaceBuf>,
        nodes: Vec<Node<F>>,
    ) -> RecipeBody<F> {
        RecipeBody {
            inputs,
            outputs,
            aliases: PortAliases::empty(),
            context,
            workspace,
            nodes,
        }
    }

    pub fn with_aliases(
        inputs: usize,
        outputs: usize,
        aliases: PortAliases,
        context: F::BodyContext,
        workspace: Vec<WorkspaceBuf>,
        nodes: Vec<Node<F>>,
    ) -> RecipeBody<F> {
        RecipeBody {
            inputs,
            outputs,
            aliases,
            context,
            workspace,
            nodes,
        }
    }

    pub fn with_port_aliases(mut self, aliases: PortAliases) -> Self {
        self.aliases = aliases;
        self
    }

    pub fn inputs(&self) -> usize {
        self.inputs
    }

    pub fn outputs(&self) -> usize {
        self.outputs
    }

    pub fn aliases(&self) -> &PortAliases {
        &self.aliases
    }

    pub fn context(&self) -> &F::BodyContext {
        &self.context
    }

    pub fn workspace(&self) -> &[WorkspaceBuf] {
        &self.workspace
    }

    pub fn nodes(&self) -> &[Node<F>] {
        &self.nodes
    }

    /// Everything a body is, for a caller assembling the body it means from the
    /// one it has. What comes back states its own arity, so nothing about how
    /// many slots bind it is lost on the way through.
    pub fn into_parts(
        self,
    ) -> (
        usize,
        usize,
        F::BodyContext,
        Vec<WorkspaceBuf>,
        Vec<Node<F>>,
    ) {
        (
            self.inputs,
            self.outputs,
            self.context,
            self.workspace,
            self.nodes,
        )
    }

    /// The three tables a body owns, mutably, for this crate's own passes over
    /// a body it is already treating as one value.
    pub(crate) fn parts_mut(
        &mut self,
    ) -> (
        &mut F::BodyContext,
        &mut Vec<WorkspaceBuf>,
        &mut Vec<Node<F>>,
    ) {
        (&mut self.context, &mut self.workspace, &mut self.nodes)
    }

    pub(crate) fn set_nodes(&mut self, nodes: Vec<Node<F>>) {
        self.nodes = nodes;
    }
}

impl<F: Computation> Node<F> {
    pub fn new(work: Work<F>, deps: impl IntoIterator<Item = usize>) -> Node<F> {
        Node {
            work,
            deps: deps.into_iter().collect(),
        }
    }

    pub fn work(&self) -> &Work<F> {
        &self.work
    }

    pub(crate) fn work_mut(&mut self) -> &mut Work<F> {
        &mut self.work
    }

    /// The nodes this one runs after.
    ///
    /// A set, because depending on a node twice is depending on it, and the
    /// order two dependencies are written in says nothing — the graph is what
    /// orders the work. Being a set by construction is what makes one recipe
    /// one value: two builders that discovered the same edges in different
    /// orders produce the same recipe, the same bytes, and the same identity.
    pub fn deps(&self) -> &BTreeSet<usize> {
        &self.deps
    }
}

impl<F: Computation> Recipe<F> {
    /// A recipe over all three of its parts, checked against each other, with
    /// the dependency edges its body states taken on the caller's word.
    ///
    /// This is the whole of the trust boundary, and it is narrow on purpose.
    /// Checking a body proves its accesses are in range, its indices are
    /// ordered, and its family data is coherent; it cannot prove that the edges
    /// it states are *enough*, because what a computation does is not written
    /// down anywhere in it. Two conflicting stores with no edge between them are
    /// a well-formed body and a wrong program, and nothing here could tell.
    ///
    /// So a producer that assembles work does not come here: it states what each
    /// placement does, has that proved against the addresses the placement
    /// reaches, and gets its edges derived — see
    /// [`RecipeBuilder`](crate::algebra::RecipeBuilder). The callers here are the
    /// ones holding work from somewhere else, where there is no statement left
    /// to derive from: bytes being decoded, and a rewrite of a recipe whose
    /// edges were already proved and are being carried across.
    /// Which family this makes a recipe of decides who may: pairing a contract
    /// with work is stating what that work is bound through, and a family whose
    /// values mean something says who may state one. The authorization is
    /// handed over rather than named, because a name is not a capability.
    pub fn stating_edges_authorized(
        contract: F::Contract,
        context: F::RootContext,
        body: RecipeBody<F>,
        authorization: &F::Authorization,
    ) -> Result<Self, RecipeError<F::Refusal>> {
        let _ = authorization;
        Recipe::restating(contract, context, body)
    }

    /// The same, where the contract and the family data are already this
    /// recipe's: restating one over work derived from it introduces no family
    /// facts, so there is nothing here to authorize.
    pub(crate) fn restating(
        contract: F::Contract,
        context: F::RootContext,
        body: RecipeBody<F>,
    ) -> Result<Self, RecipeError<F::Refusal>> {
        let recipe = Recipe {
            contract,
            context,
            body,
            shape: Shape::mint(),
        };
        recipe.validate()?;
        Ok(recipe)
    }

    /// The same, for a family whose values anyone may make.
    pub fn stating_edges(
        contract: F::Contract,
        context: F::RootContext,
        body: RecipeBody<F>,
    ) -> Result<Self, RecipeError<F::Refusal>>
    where
        F: Computation<Authorization = ()>,
    {
        Recipe::stating_edges_authorized(contract, context, body, &())
    }

    /// The same recipe, already proved, over a shape that is the same shape.
    ///
    /// Crate-private and the only unproved way in: it exists for the
    /// transformations that preserve structure exactly — mapping computations,
    /// replacing what one node computes — where re-proving the graph would be
    /// proving the same graph twice. Each of those still proves what it
    /// changed.
    pub(crate) fn holding(
        contract: F::Contract,
        context: F::RootContext,
        body: RecipeBody<F>,
        shape: Shape,
    ) -> Self {
        Recipe {
            contract,
            context,
            body,
            shape,
        }
    }

    /// The same, made by a mapping, which states the target family's
    /// authorization and hands it over here.
    pub(crate) fn mapped(
        contract: F::Contract,
        context: F::RootContext,
        body: RecipeBody<F>,
        shape: Shape,
        authorization: F::Authorization,
    ) -> Self {
        let _ = authorization;
        Recipe::holding(contract, context, body, shape)
    }

    /// Which recipe this is, for the paths it hands out.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The ports this recipe is bound through.
    pub fn contract(&self) -> &F::Contract {
        &self.contract
    }

    /// The family data every body below this one names.
    pub fn context(&self) -> &F::RootContext {
        &self.context
    }

    pub fn body(&self) -> &RecipeBody<F> {
        &self.body
    }

    /// The parts, mutably, for this crate's own passes over a recipe it is
    /// already treating as one value — replacing every payload blob with a
    /// detached copy of the same bytes.
    ///
    /// Crate-private, and only for edits that leave the structure alone: a
    /// caller outside builds the recipe it means instead, so what it gets is a
    /// value it named rather than a field it reached into.
    pub(crate) fn parts_mut(&mut self) -> (&mut F::RootContext, &mut RecipeBody<F>) {
        (&mut self.context, &mut self.body)
    }

    /// The three things a recipe is: the ports it is bound through, the family
    /// data its bodies draw from, and the work it runs.
    ///
    /// Decomposition is this, in one place, so that a caller taking a recipe
    /// apart has to say what it is doing with each part rather than letting a
    /// conversion drop two of them silently.
    pub fn into_parts(self) -> (F::Contract, F::RootContext, RecipeBody<F>) {
        (self.contract, self.context, self.body)
    }

    /// This recipe as nested work, leaving its ports behind.
    ///
    /// A fragment is bound by whatever selects it, so the contract stated here
    /// stops being true the moment it is made a candidate or substituted. That
    /// is why this is a named method and not a conversion: dropping a root's
    /// external contract is a decision a caller makes in writing, at the line
    /// where it happens, rather than something an `into()` does on its behalf.
    /// The root context travels, so the handles into it survive.
    pub fn into_fragment(self) -> RecipeFragment<F> {
        let (_ports, context, body) = self.into_parts();
        RecipeFragment::unproved(context, body)
    }
}

impl<F: Computation> PartialEq for Recipe<F> {
    /// Two recipes are equal when they say the same thing. Which of them was
    /// built first — the whole of what [`Shape`] records — is not part of that.
    fn eq(&self, other: &Self) -> bool {
        self.contract == other.contract && self.context == other.context && self.body == other.body
    }
}

impl<F: Computation> Eq for Recipe<F> {}

impl<F: Computation> std::fmt::Debug for Recipe<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recipe")
            .field("contract", &self.contract)
            .field("context", &self.context)
            .field("body", &self.body)
            .finish()
    }
}

/// A recipe, and what its work does at the ports it is bound through.
///
/// The same graph as the recipe inside it — not a second one — with a property
/// established where it was built, which is the shape [`crate::FlatRecipe`]
/// already has in this crate. What the certificate buys is being a candidate: a
/// choice holds its candidates to what it declared, and only work someone
/// vouched for can be held to anything.
///
/// The certificate is about the ports and nothing else. It never reaches bytes,
/// equality or identity, because those are the recipe's and the recipe is what
/// is encoded and hashed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProvedRecipe<F: Computation = Executable> {
    recipe: Recipe<F>,
    boundary: Effects,
}

impl<F: Computation> ProvedRecipe<F> {
    pub(crate) fn new(recipe: Recipe<F>, boundary: Effects) -> Self {
        ProvedRecipe { recipe, boundary }
    }

    /// The same, for a family whose values anyone may make.
    pub fn stating_boundary(recipe: Recipe<F>, boundary: Effects) -> Self
    where
        F: Computation<Authorization = ()>,
    {
        ProvedRecipe::proving(recipe, boundary)
    }

    pub(crate) fn proving(recipe: Recipe<F>, boundary: Effects) -> Self {
        ProvedRecipe { recipe, boundary }
    }

    /// The recipe itself, to check, inspect or identify.
    pub fn recipe(&self) -> &Recipe<F> {
        &self.recipe
    }

    /// What this work does at its ports.
    pub fn boundary(&self) -> &Effects {
        &self.boundary
    }

    /// The recipe alone, giving up the certificate — so what comes back can no
    /// longer be made a candidate.
    pub fn into_recipe(self) -> Recipe<F> {
        self.recipe
    }

    /// This work as a candidate or a region, keeping what was proved about it.
    pub fn into_fragment(self) -> RecipeFragment<F> {
        let (_ports, context, body) = self.recipe.into_parts();
        RecipeFragment::proved(context, body, self.boundary)
    }
}

/// A recipe with no choice left to resolve.
///
/// The same graph as the recipe inside it, with a property established once so
/// that everything downstream of a resolution — loading, launching, replaying —
/// asks for the property rather than re-deriving it or discovering the choice
/// at the point where there is nothing to do about it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlatRecipe<F: Computation = Executable>(Recipe<F>);

impl<F: Computation> FlatRecipe<F> {
    /// The one way to have one: flatness is decided here, from the graph, and
    /// no caller states it instead.
    ///
    /// It is decidable, which is what separates this from a boundary — there is
    /// nothing to take a producer's word about. The root body is the whole of
    /// the question, because a nested choice lives in a candidate body and a
    /// candidate body exists only under a [`Work::Choice`] node here.
    pub fn new(recipe: Recipe<F>) -> Result<Self, RecipeError<F::Refusal>> {
        if let Some(node) = recipe
            .body()
            .nodes()
            .iter()
            .position(|node| matches!(node.work(), Work::Choice { .. }))
        {
            return Err(RecipeError::UnresolvedChoice { node });
        }
        Ok(FlatRecipe(recipe))
    }

    /// The recipe this wraps. Flatness is a property of its node graph, so
    /// reading it is reading a recipe — but the level has to be named, because
    /// a caller that meant the body says so.
    pub fn recipe(&self) -> &Recipe<F> {
        &self.0
    }

    pub fn into_recipe(self) -> Recipe<F> {
        self.0
    }

    pub(crate) fn recipe_mut(&mut self) -> &mut Recipe<F> {
        &mut self.0
    }
}

impl<F: Computation> RecipeFragment<F> {
    /// A fragment and the proof of what its work does at its own ports.
    ///
    /// Crate-private, because being proved is what a choice relies on: only the
    /// builder, which derived the evidence, and the transformations that carry
    /// it across unchanged, may say a fragment is.
    pub(crate) fn proved(context: F::RootContext, body: RecipeBody<F>, boundary: Effects) -> Self {
        RecipeFragment {
            context,
            body,
            proof: Proof::Proved(boundary),
        }
    }

    /// The same, made by a mapping, which states the target family's
    /// authorization and hands it over here.
    pub(crate) fn proving(
        context: F::RootContext,
        body: RecipeBody<F>,
        boundary: Effects,
        authorization: F::Authorization,
    ) -> Self {
        let _ = authorization;
        RecipeFragment::proved(context, body, boundary)
    }

    /// A fragment nobody proved anything about.
    ///
    /// What a computation does is not recoverable from a recipe — a recipe
    /// keeps the edges that were derived from it, not the statement — so a
    /// fragment made from one arrives here, and a placement that needs a proof
    /// refuses it rather than reading its silence as "touches nothing".
    pub fn unproved(context: F::RootContext, body: RecipeBody<F>) -> Self {
        RecipeFragment {
            context,
            body,
            proof: Proof::Unproved,
        }
    }

    /// A fragment whose bodies name no root data, and that nobody proved.
    pub fn from_body(body: RecipeBody<F>) -> Self {
        RecipeFragment::unproved(F::RootContext::default(), body)
    }

    /// Whether anyone proved what this fragment's work does.
    pub fn is_proved(&self) -> bool {
        matches!(self.proof, Proof::Proved(_))
    }

    /// What this fragment does to the slots it is bound through — what a choice
    /// over it must be prepared for. Its own workspace is private, so this says
    /// nothing about that.
    pub fn boundary(&self) -> Option<&Effects> {
        match &self.proof {
            Proof::Proved(boundary) => Some(boundary),
            Proof::Unproved => None,
        }
    }

    pub fn context(&self) -> &F::RootContext {
        &self.context
    }

    pub fn body(&self) -> &RecipeBody<F> {
        &self.body
    }

    pub fn aliases(&self) -> &PortAliases {
        self.body.aliases()
    }

    /// Both parts, for a caller that takes the root data somewhere the body's
    /// handles will still reach.
    /// The root data and the work, leaving the proof behind — which is what
    /// taking a fragment apart does to it, said in the name of what comes back.
    pub fn into_parts(self) -> (F::RootContext, RecipeBody<F>) {
        (self.context, self.body)
    }

    /// The work alone, for a caller that has already established the root data
    /// is not its to carry — a body substituted into a host whose calls name
    /// the host's context, or one selected by a choice inside the same recipe.
    ///
    /// Dropping a non-empty context here would strand the handles the bodies
    /// still name, so this refuses rather than silently discarding one.
    pub fn into_body(self) -> Result<RecipeBody<F>, RecipeError<F::Refusal>> {
        if self.context != F::RootContext::default() {
            return Err(RecipeError::FragmentLeavesRootContext);
        }
        Ok(self.body)
    }
}

#[cfg(test)]
#[path = "data_tests.rs"]
mod tests;
