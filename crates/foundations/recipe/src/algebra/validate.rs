// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a recipe proves about itself before anything runs it.
//!
//! Two proofs, because they need different evidence. A body's *shape* — that
//! its nodes are ordered, its bindings fill exactly the slots they bind, its
//! launches launch something, its candidates do only what the choice above them
//! declared, its family data is coherent — is provable about a body wherever it
//! sits, and so is provable about a fragment nobody has bound yet. That is what
//! makes a fragment safe to index into before it lands anywhere.
//!
//! The other proof needs the buffers. A recipe states its ports once, at the
//! root; everything else — how far a memset writes, where an address a call is
//! handed starts, what a nested candidate touches — has to fit inside those
//! ports. The evidence is an environment: for the body being checked, how many
//! bytes each buffer it can name holds, and whether it may be written. At the
//! root that comes from the contract and the workspace declarations. Under a
//! choice it comes from the parent: a candidate's `Input(i)` *is* the parent
//! buffer `input_binding[i]`, so the candidate is checked against the buffer it
//! will really be handed rather than against ports it restates. That is why a
//! candidate needs no contract of its own — its interface is the binding, and a
//! second copy of it would be evidence that could disagree.
//!
//! Writability travels the same way. A candidate's formal `Output(j)` is
//! writable exactly when the parent buffer `output_binding[j]` is: binding a
//! formal output to a parent input is legal — a uniform call ABI passes that
//! address — but storing through it is not, and the binding is the only place
//! that fact exists.
//!
//! What a computation *is* is proved by its family, at the node or context that
//! holds it. Structure never asks what it is looking at.

use crate::algebra::computation::{Addresses, BytePortContract, Computation};
use crate::algebra::data::{Recipe, RecipeBody, RecipeFragment, Work};
use crate::algebra::error::RecipeError;
use crate::algebra::workspace::checked_body_workspace_upper_bound;
use crate::buffer::{Address, BufKind, BufRef, WritableBuf};

/// What a body may do with one buffer it is given: how many bytes it holds, and
/// whether it accepts writes.
#[derive(Clone, Copy)]
struct Port {
    nbytes: usize,
    writable: bool,
}

/// What each buffer a body can name holds, and what it may be used for.
///
/// `inputs` and `outputs` are the buffers the body is *given*; `workspace` is
/// what it declares for itself. Nothing here is derived from the body's own
/// accesses, which is what makes checking them against it meaningful.
struct Env {
    inputs: Vec<usize>,
    outputs: Vec<Port>,
    workspace: Vec<usize>,
}

impl Env {
    fn root<F: Computation>(recipe: &Recipe<F>) -> Self {
        let contract = recipe.contract();
        Env {
            inputs: (0..contract.num_inputs())
                .map(|port| contract.input_nbytes(port).unwrap_or(0))
                .collect(),
            outputs: (0..contract.num_outputs())
                .map(|port| Port {
                    nbytes: contract.output_nbytes(port).unwrap_or(0),
                    writable: true,
                })
                .collect(),
            workspace: workspace_extents(recipe.body()),
        }
    }

    /// The buffer's extent and write capability, or the error naming it as out
    /// of range. Workspace is the body's own, so it is always writable; an
    /// input never is.
    fn port<R>(&self, node: usize, buf: BufRef) -> Result<Port, RecipeError<R>> {
        let out_of_range = || RecipeError::BufOutOfRange {
            node,
            kind: buf.kind(),
            index: buf.index(),
        };
        match buf {
            BufRef::Input(index) => self
                .inputs
                .get(index)
                .map(|&nbytes| Port {
                    nbytes,
                    writable: false,
                })
                .ok_or_else(out_of_range),
            BufRef::Output(index) => self.outputs.get(index).copied().ok_or_else(out_of_range),
            BufRef::Workspace(index) => self
                .workspace
                .get(index)
                .map(|&nbytes| Port {
                    nbytes,
                    writable: true,
                })
                .ok_or_else(out_of_range),
        }
    }

    /// The environment a candidate runs in: its formal ports are the parent
    /// buffers the binding selects — extent and write capability together — and
    /// its workspace is its own.
    fn under<F: Computation, R>(
        &self,
        node: usize,
        input_binding: &[BufRef],
        output_binding: &[BufRef],
        body: &RecipeBody<F>,
    ) -> Result<Self, RecipeError<R>> {
        Ok(Env {
            inputs: input_binding
                .iter()
                .map(|&buf| Ok(self.port(node, buf)?.nbytes))
                .collect::<Result<_, RecipeError<R>>>()?,
            outputs: output_binding
                .iter()
                .map(|&buf| self.port(node, buf))
                .collect::<Result<_, RecipeError<R>>>()?,
            workspace: workspace_extents(body),
        })
    }

    /// An access of `nbytes` starting at `address`, proved to stay inside the
    /// buffer it names.
    fn reaches<R>(
        &self,
        node: usize,
        address: Address<BufRef>,
        nbytes: usize,
    ) -> Result<(), RecipeError<R>> {
        let port = self.port(node, address.buf)?;
        let wanted = address.end(nbytes).ok_or(RecipeError::AddressOverflow {
            node,
            kind: address.buf.kind(),
            index: address.buf.index(),
            byte_offset: address.byte_offset,
            nbytes,
        })?;
        if wanted > port.nbytes {
            return Err(RecipeError::AccessPastEnd {
                node,
                kind: address.buf.kind(),
                index: address.buf.index(),
                wanted,
                capacity: port.nbytes,
            });
        }
        Ok(())
    }

    fn writes<R>(
        &self,
        node: usize,
        address: Address<WritableBuf>,
        nbytes: usize,
    ) -> Result<(), RecipeError<R>> {
        let address = address.readable();
        self.accepts_writes(node, address.buf)?;
        self.reaches(node, address, nbytes)
    }

    /// A buffer a store may land on. What a node *does* is its effects, so this
    /// is where a declared write meets the binding that decided where it goes.
    fn accepts_writes<R>(&self, node: usize, buf: BufRef) -> Result<(), RecipeError<R>> {
        if !self.port(node, buf)?.writable {
            return Err(RecipeError::WriteToReadOnlyBuffer {
                node,
                kind: buf.kind(),
                index: buf.index(),
            });
        }
        Ok(())
    }
}

fn workspace_extents<F: Computation>(body: &RecipeBody<F>) -> Vec<usize> {
    body.workspace()
        .iter()
        .map(|buffer| buffer.nbytes)
        .collect()
}

impl<F: Computation> Recipe<F> {
    /// Checks the recipe against the ports it states. What a caller believes
    /// those ports to be is not consulted, so the same value validates the same
    /// way in every process that holds it.
    pub fn validate(&self) -> Result<(), RecipeError<F::Refusal>> {
        F::validate_root_context(self.context()).map_err(RecipeError::Context)?;
        let body = self.body();
        let contract = self.contract();
        if body.inputs() != contract.num_inputs() {
            return Err(RecipeError::RootArity {
                kind: BufKind::Input,
                stated: body.inputs(),
                contract: contract.num_inputs(),
            });
        }
        if body.outputs() != contract.num_outputs() {
            return Err(RecipeError::RootArity {
                kind: BufKind::Output,
                stated: body.outputs(),
                contract: contract.num_outputs(),
            });
        }
        for input in 0..contract.num_inputs() {
            if let Some(output) = contract.aliased_output(input) {
                let input_nbytes = contract.input_nbytes(input).unwrap_or(0);
                let output_nbytes = contract.output_nbytes(output).unwrap_or(0);
                if input_nbytes != output_nbytes {
                    return Err(RecipeError::AliasExtentMismatch {
                        input,
                        output,
                        input_nbytes,
                        output_nbytes,
                    });
                }
            }
        }
        validate_shape(body, self.context())?;
        validate_extents(body, &Env::root(self))
    }
}

impl<F: Computation> RecipeFragment<F> {
    /// What is provable about a fragment before anything binds it: its graph,
    /// its arity, its geometry, its declared effects, its family data.
    ///
    /// Everything that indexes into a fragment goes through this first, so a
    /// malformed one is refused where it arrives rather than reached into.
    pub fn validate_shape(&self) -> Result<(), RecipeError<F::Refusal>> {
        validate_shape(self.body(), self.context())
    }
}

impl<F: Computation> RecipeBody<F> {
    /// The same, for a body held on its own against the root data its calls
    /// name.
    ///
    /// Everything provable without the buffers: the graph, the arity, the
    /// geometry, the bindings, the family data. What it deliberately does not
    /// prove is extents, because those are the binding's — a body that is not a
    /// root has no ports of its own to check them against, and checking them
    /// against someone else's would be checking the wrong thing.
    pub fn validate_shape(&self, root: &F::RootContext) -> Result<(), RecipeError<F::Refusal>> {
        validate_shape(self, root)
    }
}

/// Everything provable about a body without knowing what its formal slots are
/// bound to.
pub(crate) fn validate_shape<F: Computation>(
    body: &RecipeBody<F>,
    root: &F::RootContext,
) -> Result<(), RecipeError<F::Refusal>> {
    F::validate_body_context(body.context(), root).map_err(RecipeError::Context)?;
    for (index, buf) in body.workspace().iter().enumerate() {
        if let Some(init) = &buf.init
            && init.len() > buf.nbytes
        {
            return Err(RecipeError::WorkspaceInitTooLarge {
                index,
                init: init.len(),
                nbytes: buf.nbytes,
            });
        }
    }
    if checked_body_workspace_upper_bound(body).is_none() {
        return Err(RecipeError::WorkspaceLayoutOverflow);
    }
    for (node, held) in body.nodes().iter().enumerate() {
        for &dep in held.deps() {
            if dep >= node {
                return Err(RecipeError::DepNotEarlier { node, dep });
            }
        }
        validate_node_shape(body, node, held.work(), root)?;
    }
    Ok(())
}

fn validate_node_shape<F: Computation>(
    body: &RecipeBody<F>,
    node: usize,
    work: &Work<F>,
    root: &F::RootContext,
) -> Result<(), RecipeError<F::Refusal>> {
    match work {
        Work::Call { call, geometry } => {
            F::validate_call(call, body.context())
                .map_err(|refusal| RecipeError::Call { node, refusal })?;
            let arity = F::launch_arity(call);
            if geometry.len() != arity {
                return Err(RecipeError::GeometryArity {
                    node,
                    stated: geometry.len(),
                    arity,
                });
            }
            for (launch, shape) in geometry.iter().enumerate() {
                if !shape.launches() {
                    return Err(RecipeError::InvalidGeometry {
                        node,
                        launch,
                        grid: shape.grid,
                        block: shape.block,
                    });
                }
                F::validate_geometry(call, launch, shape)
                    .map_err(|refusal| RecipeError::Call { node, refusal })?;
            }
        }
        Work::Memset { .. } | Work::Memcpy { .. } => {}
        Work::Choice {
            candidates,
            input_binding,
            output_binding,
        } => {
            if candidates.is_empty() {
                return Err(RecipeError::EmptyChoice { node });
            }
            let mut aliases = None;
            for candidate in candidates {
                if aliases
                    .as_ref()
                    .is_some_and(|held| held != candidate.aliases())
                {
                    return Err(RecipeError::CandidateAliasesDisagree { node });
                }
                aliases = Some(candidate.aliases().clone());
                // Alternatives share one interface, so the binding that selects
                // any of them fills exactly the slots each of them has: an
                // unfilled slot has no buffer behind it, and a surplus binding
                // names a slot nothing can select.
                if input_binding.len() != candidate.inputs() {
                    return Err(RecipeError::BindingArity {
                        node,
                        kind: BufKind::Input,
                        bound: input_binding.len(),
                        formal: candidate.inputs(),
                    });
                }
                if output_binding.len() != candidate.outputs() {
                    return Err(RecipeError::BindingArity {
                        node,
                        kind: BufKind::Output,
                        bound: output_binding.len(),
                        formal: candidate.outputs(),
                    });
                }
                validate_shape(candidate, root)?;
            }
        }
    }
    Ok(())
}

fn validate_extents<F: Computation>(
    body: &RecipeBody<F>,
    env: &Env,
) -> Result<(), RecipeError<F::Refusal>> {
    for (node, held) in body.nodes().iter().enumerate() {
        match held.work() {
            Work::Call { call, .. } => {
                // A computation forms its own addresses from the ones it is
                // handed, so how far it then reaches is its business; that the
                // address it is handed lies inside the buffer is not.
                let mut handed = Vec::new();
                call.addresses(&mut |address| handed.push(address));
                for address in handed {
                    env.reaches(node, address, 0)?;
                }
            }
            Work::Memset {
                destination,
                nbytes,
                value: _,
            } => env.writes(node, *destination, *nbytes)?,
            Work::Memcpy {
                source,
                destination,
                nbytes,
            } => {
                env.reaches(node, *source, *nbytes)?;
                env.writes(node, *destination, *nbytes)?;
            }
            Work::Choice {
                candidates,
                input_binding,
                output_binding,
            } => {
                for &binding in input_binding.iter().chain(output_binding) {
                    env.port(node, binding)?;
                }
                for candidate in candidates {
                    let inner = env.under(node, input_binding, output_binding, candidate)?;
                    validate_extents(candidate, &inner)?;
                }
            }
        }
    }
    Ok(())
}
