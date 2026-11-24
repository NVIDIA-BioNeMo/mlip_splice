// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The byte extent of every buffer a Loom recipe binds, in binding order.
//!
//! A launch takes the program's data operands first, then the reference arrays
//! its DomainIndex slots carry, then its outputs — the order
//! [`crate::backend::compiler::lowering`] reads and writes
//! them in. Each data extent comes from the same product the kernels address
//! with: the operand's cells on the iteration domain, times its visible shape,
//! times its hidden lane, times the width it is stored at. So a buffer that
//! passes a recipe's check is one every address the kernels form lands inside.

use ::recipe::{BufRef, Recipe, RecipeSignature};
use domain_index::{DomainIndex, Slot, reference_array_nbytes};

use super::super::iteration::{Plan, accumulated_elements, storage_scalar};
use super::bindings::input_bindings;
use crate::LiftedProgram;

/// The bytes operand `operand` occupies: every value it carries over the whole
/// operation, in the width it is stored at.
pub(super) fn operand_nbytes(
    program: &Plan,
    loom: &LiftedProgram,
    operand: usize,
) -> Result<usize, String> {
    let binding = loom.operands.get(operand).ok_or_else(|| {
        format!(
            "loom operand {operand} is past the {} the program binds",
            loom.operands.len()
        )
    })?;
    let elements = accumulated_elements(&program.index, operand, binding)?;
    elements
        .checked_mul(storage_scalar(binding.dtype)?.size() as usize)
        .ok_or_else(|| format!("loom operand {operand} byte extent overflow"))
}

/// The ports a whole Loom program binds: its data inputs, then every reference
/// array `slots` carry in slot then array order, then its outputs.
pub(super) fn program_signature(
    program: &Plan,
    loom: &LiftedProgram,
    slots: &[Slot],
) -> Result<RecipeSignature, String> {
    let mut inputs = (0..loom.num_inputs)
        .map(|operand| operand_nbytes(program, loom, operand))
        .collect::<Result<Vec<_>, String>>()?;
    inputs.extend(reference_array_nbytes(slots, program.index.graph())?);
    let outputs = (0..loom.outputs().len())
        .map(|output| operand_nbytes(program, loom, loom.num_inputs + output))
        .collect::<Result<Vec<_>, String>>()?;
    Ok(RecipeSignature::new(inputs, outputs))
}

/// The ports one launch of a candidate binds as a recipe of its own: the whole
/// program's inputs, then the workspace values this launch reads back, then the
/// buffers it accumulates into.
///
/// `materialized` and `accumulators` are byte extents of the buffers the launch
/// is spliced onto, so the launch states exactly what it will be handed.
pub(super) fn launch_signature(
    program: &RecipeSignature,
    materialized: &[usize],
    accumulators: &[usize],
) -> RecipeSignature {
    let mut inputs = program.inputs().to_vec();
    inputs.extend_from_slice(materialized);
    RecipeSignature::new(inputs, accumulators.to_vec())
}

/// One recipe about to be spliced onto shared ports: what it states, the slots
/// its reference-array inputs carry, and which shared output each of its own
/// outputs becomes.
pub(super) struct SplicedArm<'a> {
    pub recipe: &'a Recipe,
    pub slots: &'a [Slot],
    pub outputs: &'a [usize],
}

/// The ports a recipe takes once every arm is spliced onto a shared slot list.
///
/// An arm is placed by [`input_bindings`], the same map that splices it, so a
/// shared port takes the extent the arm already states for the port it binds
/// there. Two arms that disagree about one shared port is a failure rather than
/// a choice between extents: they are handed the same buffer.
pub(super) fn shared_signature(
    arms: &[SplicedArm<'_>],
    num_data_inputs: usize,
    all_slots: &[Slot],
    num_outputs: usize,
) -> Result<RecipeSignature, String> {
    let num_inputs = num_data_inputs + slot_array_count(all_slots);
    let mut inputs = vec![None; num_inputs];
    let mut outputs = vec![None; num_outputs];
    for arm in arms {
        for (port, binding) in input_bindings(num_data_inputs, arm.slots, all_slots)?
            .iter()
            .enumerate()
        {
            let BufRef::Input(shared) = *binding else {
                return Err("loom arm binds an input port to something else".into());
            };
            claim(
                &mut inputs,
                shared,
                arm.recipe.contract().input_nbytes(port),
            )?;
        }
        for (port, &shared) in arm.outputs.iter().enumerate() {
            claim(
                &mut outputs,
                shared,
                arm.recipe.contract().output_nbytes(port),
            )?;
        }
    }
    Ok(RecipeSignature::new(settled(inputs)?, settled(outputs)?))
}

/// How many reference arrays a slot list contributes to a recipe's inputs.
pub(super) fn slot_array_count(slots: &[Slot]) -> usize {
    slots.iter().map(|slot| slot.kind.array_count()).sum()
}

/// Record what one arm states for a shared port, refusing a port two arms
/// describe differently or one an arm binds without stating.
fn claim(ports: &mut [Option<usize>], port: usize, nbytes: Option<usize>) -> Result<(), String> {
    let nbytes = nbytes.ok_or_else(|| format!("loom arm binds port {port} without stating it"))?;
    let seen = ports
        .get_mut(port)
        .ok_or_else(|| format!("loom arm binds port {port}, past the shared ports"))?;
    match seen {
        Some(settled) if *settled != nbytes => Err(format!(
            "loom arms state {settled} and {nbytes} bytes for shared port {port}"
        )),
        _ => {
            *seen = Some(nbytes);
            Ok(())
        }
    }
}

/// The extents once every shared port has an arm that states it. A port no arm
/// binds has no extent to state, and a signature may not invent one.
fn settled(ports: Vec<Option<usize>>) -> Result<Vec<usize>, String> {
    ports
        .into_iter()
        .enumerate()
        .map(|(port, nbytes)| nbytes.ok_or_else(|| format!("no loom arm binds shared port {port}")))
        .collect()
}

#[cfg(test)]
mod tests;
