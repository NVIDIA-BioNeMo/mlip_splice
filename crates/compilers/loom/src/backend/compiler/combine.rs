// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::CompiledProgram;

pub(crate) fn combine_compiled(
    mut candidates: Vec<CompiledProgram>,
    num_data_inputs: usize,
) -> Result<CompiledProgram, String> {
    let _combining =
        chronicle::range(|| format!("loom.combine_candidates candidates={}", candidates.len()));
    if candidates.len() == 1 {
        return Ok(candidates.pop().expect("one candidate"));
    }
    let first = candidates
        .first()
        .ok_or("loom search produced no complete plan")?;
    let num_outputs = first.output_count;
    if candidates
        .iter()
        .any(|candidate| candidate.output_count != num_outputs)
    {
        return Err("loom search changed output arity".into());
    }
    let mut slots = Vec::new();
    for candidate in &candidates {
        for slot in &candidate.index_arrays {
            if !slots.iter().any(|other| super::same_slot(other, slot)) {
                slots.push(slot.clone());
            }
        }
    }
    let outputs = (0..num_outputs).collect::<Vec<_>>();
    let signature = super::shared_signature(
        &candidates
            .iter()
            .map(|candidate| super::SplicedArm {
                recipe: &candidate.recipe,
                slots: &candidate.index_arrays,
                outputs: &outputs,
            })
            .collect::<Vec<_>>(),
        num_data_inputs,
        &slots,
        num_outputs,
    )?;
    let arms = candidates
        .into_iter()
        .map(|candidate| {
            super::bind_recipe(
                &candidate.recipe,
                num_data_inputs,
                &candidate.index_arrays,
                &slots,
                &outputs,
                &signature,
            )
            .map(::recipe::ProvedRecipe::into_fragment)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let num_inputs = signature.num_inputs();
    let mut builder = ::recipe::RecipeBuilder::new(num_inputs, num_outputs);
    builder
        .whole_recipe_choice(arms, num_inputs, num_outputs)
        .map_err(|error| format!("{error:?}"))?;
    let recipe = builder
        .finish(signature)
        .map_err(|error| format!("{error:?}"))?
        .into_recipe();
    Ok(CompiledProgram {
        recipe,
        input_count: num_inputs,
        output_count: num_outputs,
        index_arrays: slots,
    })
}
