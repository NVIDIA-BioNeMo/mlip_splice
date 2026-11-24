// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Turning one computation family into another, leaving the structure alone.
//!
//! A recipe's shape — the graph, the bindings, the geometry, the effects, the
//! choices, the workspace — is independent of what its calls compute. So
//! compiling a whole recipe from one representation to the next is a leaf
//! transformation: state what one contract, one context and one call become,
//! and the recipe that comes back has the same structure over the new family.
//!
//! What the structure does not change is why the two ends stay comparable:
//! candidate order, dependency edges, byte offsets, launch geometry, declared
//! effects and choice bindings are the same values in the result. The shape is
//! the same shape, so a path taken from the source names the same place in the
//! result — which is what lets a caller hold one across a compilation.
//!
//! The result is still proved: the new family has its own idea of what a
//! coherent call is, and a structure that was sound for one is not evidence
//! about the other.

use crate::algebra::computation::{Computation, DeclaredEffects};
use crate::algebra::data::{Node, ProvedRecipe, Recipe, RecipeBody, RecipeFragment, Work};
use crate::algebra::error::RecipeError;
use crate::buffer::BufRef;
use crate::effects::Effects;

/// What one family's values become in another's.
///
/// The traversal reaches a body's context before that body's calls and before
/// the candidates below it, so a mapping that needs to know where it is can
/// remember what it last saw rather than being told twice.
pub trait Mapping<F: Computation, G: Computation> {
    /// What this mapping refuses. A structural refusal about the recipe it
    /// produces arrives here too, so one error answers "what went wrong".
    type Error: From<RecipeError<G::Refusal>>;

    /// What lets this mapping produce the target family's values.
    ///
    /// It is a constant rather than something the mapping computes, because
    /// what a mapping is allowed to make is decided when it is written and not
    /// while it runs — so a mapping that cannot state one is refused where it
    /// is written down rather than where it is used.
    const AUTHORIZATION: G::Authorization;

    fn contract(&mut self, contract: &F::Contract) -> Result<G::Contract, Self::Error>;
    fn root_context(&mut self, context: &F::RootContext) -> Result<G::RootContext, Self::Error>;
    fn body_context(&mut self, context: &F::BodyContext) -> Result<G::BodyContext, Self::Error>;
    fn call(&mut self, call: &F::Call) -> Result<G::Call, Self::Error>;
}

impl<F: Computation> Recipe<F> {
    /// This recipe's structure over another family's computations.
    pub fn map<G: Computation, M: Mapping<F, G>>(
        &self,
        mapping: &mut M,
    ) -> Result<Recipe<G>, M::Error> {
        let contract = mapping.contract(self.contract())?;
        let context = mapping.root_context(self.context())?;
        let mapped = Recipe::mapped(
            contract,
            context,
            self.body().map(mapping)?,
            self.shape().clone(),
            M::AUTHORIZATION,
        );
        mapped.validate().map_err(M::Error::from)?;
        Ok(mapped)
    }
}

impl<F: Computation> ProvedRecipe<F> {
    /// The same, with the proof carried across and proved again.
    ///
    /// What a recipe does at its ports is a fact about its structure and its
    /// bindings, and mapping changes neither — so the boundary the source was
    /// proved against is a statement about the result as well. It is only a
    /// true one if the result does no more than the source did, which is not
    /// something the mapping is trusted for: the mapped work's own effects are
    /// read back and every one of them has to be covered by the boundary that
    /// came across.
    ///
    /// That is why the result's family has to answer for what its calls do. A
    /// family whose calls are opaque cannot be asked, so a proof does not cross
    /// into one — which is the honest outcome rather than a proof resting on an
    /// assumption.
    pub fn map<G: DeclaredEffects, M: Mapping<F, G>>(
        &self,
        mapping: &mut M,
    ) -> Result<ProvedRecipe<G>, M::Error> {
        let mapped = self.recipe().map(mapping)?;
        let boundary = self.boundary().clone();
        covers(mapped.body(), &boundary).map_err(M::Error::from)?;
        Ok(ProvedRecipe::proving(mapped, boundary))
    }
}

/// Whether `boundary` answers for everything the work in `body` does to the
/// buffers around it, node by node.
///
/// A body's own workspace is nobody else's, so it is not part of what a recipe
/// does at its ports and contributes nothing here.
fn covers<G: DeclaredEffects>(
    body: &RecipeBody<G>,
    boundary: &Effects,
) -> Result<(), RecipeError<G::Refusal>> {
    for (node, held) in body.nodes().iter().enumerate() {
        for (buf, needed) in does(held.work()).iter() {
            if matches!(buf, BufRef::Workspace(_)) {
                continue;
            }
            if !boundary
                .access(buf)
                .is_some_and(|declared| declared.covers(needed))
            {
                return Err(RecipeError::UndeclaredEffect {
                    node,
                    kind: buf.kind(),
                    index: buf.index(),
                });
            }
        }
    }
    Ok(())
}

/// What one node does to the buffers of the body that holds it.
///
/// A choice does what its candidates do, taken through the binding that places
/// them, because whichever one runs is what the choice runs.
fn does<G: DeclaredEffects>(work: &Work<G>) -> Effects {
    match work {
        Work::Call { call, .. } => G::call_effects(call),
        Work::Choice {
            candidates,
            input_binding,
            output_binding,
        } => {
            let mut joined = Vec::new();
            for candidate in candidates {
                for (buf, access) in reaches(candidate).iter() {
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
            Effects::of(joined)
        }
        held => held.memory_effects().unwrap_or_else(Effects::none),
    }
}

/// What a whole body does to the buffers it is bound through.
fn reaches<G: DeclaredEffects>(body: &RecipeBody<G>) -> Effects {
    let mut held = Vec::new();
    for node in body.nodes() {
        held.extend(does(node.work()).iter());
    }
    Effects::of(held)
}

impl<F: Computation> RecipeFragment<F> {
    /// This work over another family's computations, with whatever proof it
    /// carries proved again.
    ///
    /// The rule is the recipe's: mapping leaves the graph and the bindings
    /// alone, so a proof about this work is a proof about the work it becomes
    /// — as long as what arrives does no more than what it replaces, which is
    /// read back off the mapped calls rather than taken on trust. An absence
    /// of proof stays an absence, and it is the result's family that has to
    /// answer for what its calls do, so a proof does not cross into a family
    /// whose calls are opaque.
    pub fn map<G: DeclaredEffects, M: Mapping<F, G>>(
        &self,
        mapping: &mut M,
    ) -> Result<RecipeFragment<G>, M::Error> {
        let context = mapping.root_context(self.context())?;
        let body = self.body().map(mapping)?;
        Ok(match self.boundary() {
            Some(boundary) => {
                covers(&body, boundary).map_err(M::Error::from)?;
                RecipeFragment::proving(context, body, boundary.clone(), M::AUTHORIZATION)
            }
            None => RecipeFragment::unproved(context, body),
        })
    }
}

impl<F: Computation> RecipeBody<F> {
    pub fn map<G: Computation, M: Mapping<F, G>>(
        &self,
        mapping: &mut M,
    ) -> Result<RecipeBody<G>, M::Error> {
        let context = mapping.body_context(self.context())?;
        let nodes = self
            .nodes()
            .iter()
            .map(|node| {
                Ok(Node::new(
                    node.work().map(mapping)?,
                    node.deps().iter().copied(),
                ))
            })
            .collect::<Result<Vec<_>, M::Error>>()?;
        Ok(RecipeBody::with_aliases(
            self.inputs(),
            self.outputs(),
            self.aliases().clone(),
            context,
            self.workspace().to_vec(),
            nodes,
        ))
    }
}

impl<F: Computation> Work<F> {
    pub fn map<G: Computation, M: Mapping<F, G>>(
        &self,
        mapping: &mut M,
    ) -> Result<Work<G>, M::Error> {
        Ok(match self {
            Work::Call { call, geometry } => Work::Call {
                call: mapping.call(call)?,
                geometry: geometry.clone(),
            },
            Work::Memset {
                destination,
                value,
                nbytes,
            } => Work::Memset {
                destination: *destination,
                value: *value,
                nbytes: *nbytes,
            },
            Work::Memcpy {
                source,
                destination,
                nbytes,
            } => Work::Memcpy {
                source: *source,
                destination: *destination,
                nbytes: *nbytes,
            },
            Work::Choice {
                candidates,
                input_binding,
                output_binding,
            } => Work::Choice {
                candidates: candidates
                    .iter()
                    .map(|candidate| candidate.map(mapping))
                    .collect::<Result<Vec<_>, M::Error>>()?,
                input_binding: input_binding.clone(),
                output_binding: output_binding.clone(),
            },
        })
    }
}
