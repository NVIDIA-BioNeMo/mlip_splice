// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use ::recipe::RecipeSignature;

use super::signature::launch_signature;
use super::{
    Candidate, CompiledParts, DomainIndex, Launch, LiftedProgram, ModuleSource, Plan, SlotSpec,
    WARP_SIZE, WriteMode, reduction,
};

/// Whole recipes taken as nested work: each states the ports it was compiled
/// with, and the `Choice` bindings below replace them, so the erasure is named
/// here rather than happening inside the call.
fn fragments(recipes: Vec<::recipe::ProvedRecipe>) -> Vec<::recipe::RecipeFragment> {
    recipes
        .into_iter()
        .map(::recipe::ProvedRecipe::into_fragment)
        .collect()
}

/// Every lowering ends the same way: a recipe is only worth returning once the
/// buffers it binds are the ones it states.
fn finish(
    builder: ::recipe::RecipeBuilder,
    signature: RecipeSignature,
) -> Result<::recipe::ProvedRecipe, String> {
    builder
        .finish(signature)
        .map_err(|error| format!("{error:?}"))
}

pub(super) fn lower_single_block_reduction_candidate(
    loom: &LiftedProgram,
    source: &ModuleSource,
    block_size: u32,
    signature: &RecipeSignature,
) -> Result<::recipe::ProvedRecipe, String> {
    let mut builder = ::recipe::RecipeBuilder::new(signature.num_inputs(), signature.num_outputs());
    let module = source.install(&mut builder);
    let mut launch = builder
        .kernel(module, &source.entry)
        .grid([1, 1, 1])
        .block([block_size, 1, 1]);
    for input in 0..signature.num_inputs() {
        launch = launch.read(::recipe::Buf::Input(input));
    }
    for output in 0..loom.outputs().len() {
        launch = launch.write(::recipe::Writable::Output(output));
    }
    launch.add();
    finish(builder, signature.clone())
}

/// One whole-program plan lowered at one bucket assignment: `buckets[j]` is how
/// many copies of output `j`'s keyspace the main kernels fold into, and the
/// projection sums them away.
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_candidate(
    program: &Plan,
    loom: &LiftedProgram,
    candidate: &Candidate,
    table: &[SlotSpec],
    signature: &RecipeSignature,
    part_modules: &CompiledParts,
    cut_modules: Option<Vec<Vec<(usize, ModuleSource)>>>,
    projections: &[Option<reduction::ProjectionModule>],
    buckets: &[u32],
) -> Result<::recipe::ProvedRecipe, String> {
    if candidate.guards.len() != candidate.launches.len()
        || candidate.hoists.len() != candidate.launches.len()
        || candidate.guards.iter().flatten().any(|plan| {
            plan.regions.is_empty() || plan.regions.iter().any(|region| region.outputs.is_empty())
        })
    {
        return Err("loom guarded region has invalid launch ownership".into());
    }
    for region in &candidate.reductions {
        if region.launches.first() != Some(&region.init_launch)
            || region
                .launches
                .iter()
                .any(|&launch| launch >= candidate.launches.len())
        {
            return Err("loom reduction region has invalid launch ownership".into());
        }
    }
    let mut builder = ::recipe::RecipeBuilder::new(signature.num_inputs(), signature.num_outputs());
    let output_buffers = reduction::allocate(&mut builder, program, loom, buckets)?;
    // What each launch is handed for the buffers it shares with the others: the
    // accumulation the parts write, and the workspace values a cut produces.
    let accumulators = output_buffers
        .iter()
        .map(|buffer| buffer.nbytes)
        .collect::<Vec<_>>();
    let materialized = candidate
        .materialized
        .iter()
        .map(|value| value.nbytes)
        .collect::<Vec<_>>();
    let masks = bucket_mask_args(program, buckets)?;
    let material_buffers = candidate
        .materialized
        .iter()
        .map(|value| builder.scratch(value.nbytes))
        .collect::<Vec<_>>();
    let mut cut_modules = cut_modules.map(Vec::into_iter);
    let mut produced = 0;

    // Launch ordering is not stated explicitly: every node declares what it
    // reads and which outputs its part writes, and the builder infers the
    // dependency edges. Continuation launches share their output buffer, so
    // writer-after-writer keeps them in plan order; launches with disjoint
    // outputs stay unordered and the executor's graph overlaps them.
    for (launch_index, launch) in candidate.launches.iter().enumerate() {
        match launch {
            Launch::Producer { output, inputs } => {
                if *output != produced {
                    return Err("loom materialized outputs are not produced in order".into());
                }
                let slots = cut_modules
                    .as_mut()
                    .and_then(Iterator::next)
                    .ok_or("loom cut producer has no kernel")?;
                let reads = inputs
                    .iter()
                    .map(|&value| materialized_nbytes(&materialized, value))
                    .collect::<Result<Vec<_>, String>>()?;
                let produces = materialized_nbytes(&materialized, *output)?;
                let producers = slots
                    .into_iter()
                    .map(|(slot, source)| {
                        lower_producer(
                            program,
                            &source,
                            table[slot].block_size,
                            signature,
                            &reads,
                            produces,
                        )
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let read: Vec<::recipe::Buf> = (0..signature.num_inputs())
                    .map(::recipe::Buf::Input)
                    .chain(
                        inputs
                            .iter()
                            .map(|&input| ::recipe::Buf::Workspace(material_buffers[input])),
                    )
                    .collect();
                let written = [::recipe::Buf::Workspace(material_buffers[*output])];
                builder
                    .choice(fragments(producers))
                    .inputs(read.iter().copied())
                    .does(read, ::recipe::Access::Read)
                    .outputs(written)
                    .does(written, ::recipe::Access::Write)
                    .add()
                    .map_err(|error| format!("{error:?}"))?;
                produced += 1;
            }
            Launch::Consumer { part, output_modes } => {
                // A part stores only into the outputs it has elements for; the
                // rest it is merely handed, which is what the choice below
                // carries rather than declares.
                let output_modes: Vec<Option<WriteMode>> = output_modes
                    .iter()
                    .enumerate()
                    .map(|(output, mode)| part.writes_output(output).then_some(*mode))
                    .collect();
                let candidates = if let Some(modules) = cut_modules.as_mut() {
                    let slots = modules.next().ok_or("loom cut consumer has no kernel")?;
                    slots
                        .into_iter()
                        .map(|(slot, source)| {
                            let spec = &table[slot];
                            lower_part(
                                program,
                                &source,
                                (
                                    spec.block_size,
                                    spec.positions_per_block(),
                                    u32::from(spec.shards),
                                ),
                                signature,
                                &materialized[..produced],
                                &accumulators,
                                &output_modes,
                                &masks,
                            )
                        })
                        .collect::<Result<Vec<_>, String>>()?
                } else {
                    let part_index = launch_index;
                    part_modules[part_index]
                        .iter()
                        .enumerate()
                        .filter_map(|(block_index, source)| {
                            let source = source.as_ref()?;
                            let spec = &table[block_index];
                            Some(lower_part(
                                program,
                                source,
                                (
                                    spec.block_size,
                                    spec.positions_per_block(),
                                    u32::from(spec.shards),
                                ),
                                signature,
                                &[],
                                &accumulators,
                                &output_modes,
                                &masks,
                            ))
                        })
                        .collect::<Result<Vec<_>, String>>()?
                };
                let read: Vec<::recipe::Buf> = (0..signature.num_inputs())
                    .map(::recipe::Buf::Input)
                    .chain(
                        material_buffers[..produced]
                            .iter()
                            .copied()
                            .map(::recipe::Buf::Workspace),
                    )
                    .collect();
                let mut choice = builder
                    .choice(fragments(candidates))
                    .inputs(read.iter().copied())
                    .does(read, ::recipe::Access::Read);
                for (output, buffer) in output_buffers.iter().enumerate() {
                    choice = choice.outputs([buffer.main]);
                    if part.writes_output(output) {
                        choice = choice.does([buffer.main], ::recipe::Access::Write);
                    }
                }
                choice.add().map_err(|error| format!("{error:?}"))?;
            }
        }
    }
    if let Some(mut remaining) = cut_modules
        && remaining.next().is_some()
    {
        return Err("loom cut plan has unused kernels".into());
    }
    reduction::add_projections(&mut builder, projections, program, loom, &output_buffers)?;
    finish(builder, signature.clone())
}

/// The extent of one workspace value, which a launch can bind only once the
/// candidate states it.
fn materialized_nbytes(materialized: &[usize], value: usize) -> Result<usize, String> {
    materialized.get(value).copied().ok_or_else(|| {
        format!("loom launch binds materialized value {value}, which the candidate does not state")
    })
}

/// The bucket-mask arguments this assignment binds, in the order the layout
/// declares the parameters.
fn bucket_mask_args(program: &Plan, buckets: &[u32]) -> Result<Vec<i32>, String> {
    program
        .layout
        .bucket_masks
        .iter()
        .map(|mask| {
            let count = *buckets
                .get(mask.output)
                .ok_or("loom bucket assignment is missing an output")?;
            if !count.is_power_of_two() {
                return Err(format!("loom bucket count {count} is not a power of two"));
            }
            i32::try_from(count - 1)
                .map_err(|_| format!("loom bucket count {count} exceeds the mask width"))
        })
        .collect()
}

/// One partition's launch at one block size as a recipe of its own: its formal
/// `Input`/`Output` slots map onto the shared recipe's buffers through the
/// partition's block-size Choice. `masks` carries one bucket mask per output
/// whose fold picks its bucket count at launch. This launch reads the whole
/// program's inputs, then the workspace values `materialized` states, and
/// accumulates into the `accumulators` the parts share.
///
/// `output_modes[j]` says how this part reaches output `j`: a plain store, a
/// read-modify-write (atomic scatter, or a continuation of an earlier
/// partition), or — `None` — that it stores nothing there at all. The ABI
/// passes every output's pointer whatever the answer, so a part with no
/// elements for an output is handed the address and touches nothing behind it.
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_part(
    program: &Plan,
    source: &ModuleSource,
    launch: (u32, u32, u32),
    signature: &RecipeSignature,
    materialized: &[usize],
    accumulators: &[usize],
    output_modes: &[Option<WriteMode>],
    masks: &[i32],
) -> Result<::recipe::ProvedRecipe, String> {
    if accumulators.len() != output_modes.len() {
        return Err("loom launch writes buffers it does not state".into());
    }
    let num_recipe_inputs = signature.num_inputs();
    let (block_size, positions_per_block, shards) = launch;
    // The ports this launch is bound through are its own, not the whole
    // program's: it takes the materialized workspace values as further inputs
    // and writes only the accumulators the parts share.
    let ports = launch_signature(signature, materialized, accumulators);
    let mut builder = ::recipe::RecipeBuilder::new(ports.num_inputs(), ports.num_outputs());
    let module = source.install(&mut builder);
    let grid = [
        ::recipe::count_covering_blocks(program.index.batch_total()?, positions_per_block as i32)?,
        shards,
        1,
    ];
    let mut launch = builder
        .kernel(module, &source.entry)
        .grid(grid)
        .block([block_size, 1, 1]);
    for input in 0..num_recipe_inputs {
        launch = launch.read(::recipe::Buf::Input(input));
    }
    for (output, mode) in output_modes.iter().enumerate() {
        launch = match mode {
            None => launch.pointer(::recipe::Buf::Output(output)),
            Some(WriteMode::Store) => launch.write(::recipe::Writable::Output(output)),
            Some(_) => launch.read_write(::recipe::Writable::Output(output)),
        };
    }
    for input in num_recipe_inputs..num_recipe_inputs + materialized.len() {
        launch = launch.read(::recipe::Buf::Input(input));
    }
    for &mask in masks {
        launch = launch.scalar_i32(mask);
    }
    launch.add();
    finish(builder, ports)
}

/// One cut producer as a recipe of its own: it reads the whole program's
/// inputs, then the workspace values `materialized` states, and writes the one
/// workspace value it produces.
pub(super) fn lower_producer(
    program: &Plan,
    source: &ModuleSource,
    block_size: u32,
    signature: &RecipeSignature,
    materialized: &[usize],
    produced: usize,
) -> Result<::recipe::ProvedRecipe, String> {
    // This producer takes the materialized values it reads as further inputs
    // and writes the one workspace value it produces.
    let ports = launch_signature(signature, materialized, &[produced]);
    let mut builder = ::recipe::RecipeBuilder::new(ports.num_inputs(), ports.num_outputs());
    let module = source.install(&mut builder);
    let positions_per_block = block_size / WARP_SIZE;
    let grid = [
        ::recipe::count_covering_blocks(program.index.batch_total()?, positions_per_block as i32)?,
        1,
        1,
    ];
    let mut launch = builder
        .kernel(module, &source.entry)
        .grid(grid)
        .block([block_size, 1, 1]);
    for input in 0..signature.num_inputs() + materialized.len() {
        launch = launch.read(::recipe::Buf::Input(input));
    }
    launch = launch.write(::recipe::Writable::Output(0));
    launch.add();
    finish(builder, ports)
}
