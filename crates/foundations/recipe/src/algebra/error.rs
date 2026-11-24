// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a recipe refuses, structurally or on its family's behalf.

use crate::algebra::path::NodePath;
use crate::buffer::BufKind;

/// A refusal from a recipe. `R` is the computation family's own refusal, which
/// arrives here at the place it was refused and is otherwise untouched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecipeError<R> {
    /// A node depends on something that is not a strictly-earlier node, so the
    /// index order is not a topological order.
    DepNotEarlier {
        node: usize,
        dep: usize,
    },
    /// A node names a buffer the body it sits in does not have.
    BufOutOfRange {
        node: usize,
        kind: BufKind,
        index: usize,
    },
    /// A node would touch more bytes than the buffer it names holds. The
    /// buffer's extent comes from the root contract, or — inside a choice —
    /// from the parent buffer the binding selects.
    AccessPastEnd {
        node: usize,
        kind: BufKind,
        index: usize,
        wanted: usize,
        capacity: usize,
    },
    /// A node's start byte plus the bytes it touches leaves the address space,
    /// so there is no range to compare against the buffer at all.
    AddressOverflow {
        node: usize,
        kind: BufKind,
        index: usize,
        byte_offset: usize,
        nbytes: usize,
    },
    /// A node stores into a buffer it may only read. At the root that is
    /// impossible for its own outputs; inside a choice it means the formal
    /// output it writes is bound to a read-only parent buffer.
    WriteToReadOnlyBuffer {
        node: usize,
        kind: BufKind,
        index: usize,
    },
    /// A body states an arity its contract does not: the root's formal slots
    /// are its ports, and disagreeing about how many there are makes every
    /// index below it mean two things.
    RootArity {
        kind: BufKind,
        stated: usize,
        contract: usize,
    },
    AliasExtentMismatch {
        input: usize,
        output: usize,
        input_nbytes: usize,
        output_nbytes: usize,
    },
    /// A binding fills a different number of slots than the body it binds has.
    /// Both directions are refused: an unfilled slot has no buffer behind it,
    /// and a surplus binding names a slot nothing can select.
    BindingArity {
        node: usize,
        kind: BufKind,
        bound: usize,
        formal: usize,
    },
    CandidateAliasesDisagree {
        node: usize,
    },
    /// Something under a placement does more to a buffer than the placement
    /// answers for — a candidate writing where the choice promised only to
    /// read, or work substituted into a call reaching past what that call's
    /// edges were derived from. Either way nothing would have been ordered
    /// against that buffer's other users.
    UndeclaredEffect {
        node: usize,
        kind: BufKind,
        index: usize,
    },
    /// A placement claims an effect on a buffer it is never handed. A statement
    /// about memory a computation does not receive is not a statement about
    /// this placement, and the edges derived from it would order nothing real.
    EffectNotHanded {
        node: usize,
        kind: BufKind,
        index: usize,
    },
    /// A call states a different number of launch shapes than it has launches.
    /// A launch with no shape has nowhere to run, and a shape with no launch
    /// describes nothing.
    GeometryArity {
        node: usize,
        stated: usize,
        arity: usize,
    },
    /// A launch of no threads. A zero extent is an absent launch rather than a
    /// small one, so a recipe states work it does not have.
    InvalidGeometry {
        node: usize,
        launch: usize,
        grid: [u32; 3],
        block: [u32; 3],
    },
    /// Work still reaching a workspace buffer that was released. The memory is
    /// gone, so there is nothing for it to name.
    WorkspaceStillReached {
        node: usize,
        index: usize,
    },
    /// A workspace buffer seeded with more bytes than it declares.
    WorkspaceInitTooLarge {
        index: usize,
        init: usize,
        nbytes: usize,
    },
    /// The buffers a body declares, laid out end to end on their alignment,
    /// leave the address space — so there is no allocation that could hold
    /// them and no offset arithmetic that stays true.
    WorkspaceLayoutOverflow,
    EmptyChoice {
        node: usize,
    },
    UnresolvedChoice {
        node: usize,
    },
    /// A fragment asked for its body alone while still carrying root data its
    /// bodies name. Dropping that data would strand the handles into it.
    FragmentLeavesRootContext,
    /// A fragment nobody proved was asked to be placed by effect. Its silence
    /// is not a claim that it touches nothing, so there is nothing to derive
    /// edges from and nothing to hold it to.
    UnprovedFragment,
    /// A path from a different recipe, or from this one before an edit moved
    /// what it named.
    PathNotOwned,
    /// A path descends through a choice this recipe does not have, at the step
    /// that stopped resolving.
    NoSuchBody {
        at: usize,
    },
    NoSuchNode {
        path: NodePath,
    },
    /// A path names a node that is not a computation, where one was needed.
    NotACall {
        path: NodePath,
    },
    /// A selection states a different number of choices than the body it
    /// selects in holds. One entry per choice is what makes a selection name
    /// the choices rather than happen to line up with some of them.
    SelectionArity {
        stated: usize,
        choices: usize,
    },
    /// A selection names a candidate a choice does not offer.
    NoSuchCandidate {
        node: usize,
        candidate: usize,
        offered: usize,
    },
    /// The family refused the data a recipe or one of its bodies carries.
    Context(R),
    /// The family refused the call at `node`.
    Call {
        node: usize,
        refusal: R,
    },
}

impl<R: std::fmt::Display> std::fmt::Display for RecipeError<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecipeError::DepNotEarlier { node, dep } => {
                write!(
                    f,
                    "node {node} depends on {dep}, not a strictly-earlier node"
                )
            }
            RecipeError::BufOutOfRange { node, kind, index } => {
                write!(
                    f,
                    "node {node} references {} {index}, out of range",
                    kind.name()
                )
            }
            RecipeError::AccessPastEnd {
                node,
                kind,
                index,
                wanted,
                capacity,
            } => {
                write!(
                    f,
                    "node {node} touches {wanted} bytes of {} {index}, which holds {capacity}",
                    kind.name()
                )
            }
            RecipeError::AddressOverflow {
                node,
                kind,
                index,
                byte_offset,
                nbytes,
            } => {
                write!(
                    f,
                    "node {node} touches {nbytes} bytes of {} {index} from byte {byte_offset}, past the end of the address space",
                    kind.name()
                )
            }
            RecipeError::WriteToReadOnlyBuffer { node, kind, index } => {
                write!(
                    f,
                    "node {node} writes {} {index}, which is read-only here",
                    kind.name()
                )
            }
            RecipeError::RootArity {
                kind,
                stated,
                contract,
            } => {
                write!(
                    f,
                    "the root body states {stated} formal {} slot(s), but its contract states {contract}",
                    kind.name()
                )
            }
            RecipeError::AliasExtentMismatch {
                input,
                output,
                input_nbytes,
                output_nbytes,
            } => write!(
                f,
                "alias input {input} is {input_nbytes} bytes and output {output} is {output_nbytes}"
            ),
            RecipeError::BindingArity {
                node,
                kind,
                bound,
                formal,
            } => {
                write!(
                    f,
                    "node {node} binds {bound} {} slot(s) to a body that has {formal}",
                    kind.name()
                )
            }
            RecipeError::CandidateAliasesDisagree { node } => {
                write!(
                    f,
                    "choice at node {node} has candidates with different port aliases"
                )
            }
            RecipeError::UndeclaredEffect { node, kind, index } => {
                write!(
                    f,
                    "node {node} does more to {} {index} than it declares",
                    kind.name()
                )
            }
            RecipeError::EffectNotHanded { node, kind, index } => {
                write!(
                    f,
                    "node {node} claims an effect on {} {index}, which it is never handed",
                    kind.name()
                )
            }
            RecipeError::GeometryArity {
                node,
                stated,
                arity,
            } => {
                write!(
                    f,
                    "node {node} states {stated} launch shape(s) for a call that is {arity} launch(es)"
                )
            }
            RecipeError::InvalidGeometry {
                node,
                launch,
                grid,
                block,
            } => {
                write!(
                    f,
                    "node {node} launch {launch} runs grid {grid:?} of blocks {block:?}, which is no threads at all"
                )
            }
            RecipeError::WorkspaceStillReached { node, index } => {
                write!(
                    f,
                    "node {node} reaches workspace {index}, which this recipe no longer declares"
                )
            }
            RecipeError::WorkspaceInitTooLarge {
                index,
                init,
                nbytes,
            } => {
                write!(
                    f,
                    "workspace {index} is seeded with {init} bytes but declares {nbytes}"
                )
            }
            RecipeError::WorkspaceLayoutOverflow => {
                write!(
                    f,
                    "the declared workspace does not fit in the address space"
                )
            }
            RecipeError::EmptyChoice { node } => {
                write!(f, "node {node} is a choice with no candidates")
            }
            RecipeError::UnresolvedChoice { node } => {
                write!(f, "node {node} is an unresolved choice")
            }
            RecipeError::FragmentLeavesRootContext => {
                write!(
                    f,
                    "a fragment carrying root data has no body of its own to give"
                )
            }
            RecipeError::UnprovedFragment => {
                write!(
                    f,
                    "that fragment carries no proof of what its work does, so nothing can place it by effect"
                )
            }
            RecipeError::PathNotOwned => {
                write!(f, "that path names a place in another recipe's shape")
            }
            RecipeError::NoSuchBody { at } => {
                write!(f, "no such body: step {at} names no candidate here")
            }
            RecipeError::NoSuchNode { path } => {
                write!(f, "no such node: nothing holds node {}", path.node())
            }
            RecipeError::NotACall { path } => {
                write!(f, "node {} is not a computation", path.node())
            }
            RecipeError::SelectionArity { stated, choices } => write!(
                f,
                "a selection of {stated} names the choices of a body that holds {choices}"
            ),
            RecipeError::NoSuchCandidate {
                node,
                candidate,
                offered,
            } => write!(
                f,
                "the choice at node {node} offers {offered} candidate(s), and candidate \
                 {candidate} is not one of them"
            ),
            RecipeError::Context(refusal) => write!(f, "{refusal}"),
            RecipeError::Call { node, refusal } => write!(f, "node {node}: {refusal}"),
        }
    }
}

impl<R: std::fmt::Debug + std::fmt::Display> std::error::Error for RecipeError<R> {}
