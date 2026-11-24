// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    CompileLimits, CompileProfile, CompiledProgram, FamilyPartition, SplicedArm,
    configurations::Configurations, input_bindings, materialize::SingleCandidates,
    product::Product, same_slot, shared_signature,
};

pub(super) struct PartitionCandidates {
    families: Vec<(SingleCandidates, Vec<usize>)>,
    selections: Product<Configurations>,
    num_inputs: usize,
    num_outputs: usize,
    max_workspace_bytes: Option<usize>,
}

impl PartitionCandidates {
    pub(super) fn new(
        partition: FamilyPartition,
        limits: CompileLimits,
        num_inputs: usize,
        num_outputs: usize,
        profile: &mut CompileProfile,
        checkpoint: &mut impl FnMut() -> Result<(), String>,
    ) -> Result<Self, String> {
        let mut families = Vec::new();
        for (program, outputs, placement) in partition {
            checkpoint()?;
            let planned = super::pipeline::plan_single_with_limits_profiled(
                program, limits, placement, profile,
            )?;
            checkpoint()?;
            families.push((
                SingleCandidates::new(planned, limits, placement, true, profile)?,
                outputs,
            ));
        }
        let selections = Product::new(
            families
                .iter()
                .map(|(family, _)| family.configurations())
                .collect(),
        );
        Ok(Self {
            families,
            selections,
            num_inputs,
            num_outputs,
            max_workspace_bytes: limits.max_workspace_bytes,
        })
    }

    pub(super) fn next(
        &mut self,
        profile: &mut CompileProfile,
        checkpoint: &mut impl FnMut() -> Result<(), String>,
    ) -> Option<Result<CompiledProgram, String>> {
        let selection = self.selections.next()?;
        Some((|| {
            let mut compiled = Vec::new();
            for ((family, outputs), configuration) in self.families.iter_mut().zip(selection) {
                checkpoint()?;
                let candidate = family.materialize(configuration, profile, checkpoint)?;
                if candidate.output_count != outputs.len() {
                    return Err("loom output-family recipe has inconsistent arity".into());
                }
                compiled.push((candidate, outputs));
            }
            checkpoint()?;
            let mut slots = Vec::new();
            for (candidate, _) in &compiled {
                for slot in &candidate.index_arrays {
                    if !slots.iter().any(|other| same_slot(slot, other)) {
                        slots.push(slot.clone());
                    }
                }
            }
            let signature = shared_signature(
                &compiled
                    .iter()
                    .map(|(candidate, outputs)| SplicedArm {
                        recipe: &candidate.recipe,
                        slots: &candidate.index_arrays,
                        outputs,
                    })
                    .collect::<Vec<_>>(),
                self.num_inputs,
                &slots,
                self.num_outputs,
            )?;
            let num_inputs = signature.num_inputs();
            let mut builder = ::recipe::RecipeBuilder::new(num_inputs, self.num_outputs);
            for (candidate, outputs) in compiled {
                builder
                    .splice(
                        candidate.recipe.into_fragment(),
                        &input_bindings(self.num_inputs, &candidate.index_arrays, &slots)?,
                        &outputs
                            .iter()
                            .copied()
                            .map(::recipe::BufRef::Output)
                            .collect::<Vec<_>>(),
                        &[],
                    )
                    .map_err(|error| format!("{error:?}"))?;
            }
            let recipe = builder
                .finish(signature)
                .map_err(|error| format!("{error:?}"))?;
            if !super::within_workspace_bound(&recipe, self.max_workspace_bytes) {
                return Err(super::workspace_limit_error(self.max_workspace_bytes));
            }
            Ok(CompiledProgram {
                recipe: recipe.into_recipe(),
                input_count: num_inputs,
                output_count: self.num_outputs,
                index_arrays: slots,
            })
        })())
    }
}
