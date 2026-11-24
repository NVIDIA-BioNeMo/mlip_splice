// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use domain_index::DomainIndex;
use gpu_ir::{Evict, Launch, Value};

use super::compiler::ModuleSource;
use super::iteration::{Index, Plan};
use super::name::{shared_bucket_fold, shared_convert};
use crate::LiftedProgram;
use crate::domain::reduction::{OutputFold, Projection, ProjectionEntries};

/// Where one output's fold accumulates and what closes it: the buffer the main
/// kernels write, how many bytes it holds, and the projection that turns it
/// into the output.
pub(super) struct OutputBuffer {
    pub main: recipe::Buf,
    pub nbytes: usize,
    pub projection: Option<Projection>,
}

pub(super) fn validate(program: &Plan, loom: &LiftedProgram) -> Result<(), String> {
    let fiber = program.index.fiber_bridge().is_some();
    for (output, binding) in loom.outputs().iter().enumerate() {
        let func = loom.module.entry_func();
        if func.value_shape(func.results[output]).dtype != xir::dtype::DType::Bool {
            continue;
        }
        let operand = loom.num_inputs + output;
        let form = if binding.lane_extent == 1 {
            Some("hidden-lane")
        } else if fiber {
            Some("fiber")
        } else if program.index.needs_atomic(operand) {
            Some("DomainIndex scatter")
        } else {
            None
        };
        if let Some(form) = form {
            return Err(format!(
                "loom output {output}: boolean {form} reduction has no additive semantics"
            ));
        }
    }
    Ok(())
}

/// `lanes` is the hidden-lane width `value` carries: a continuation reads the
/// partial back to add to it, and the two sides must be the same vector.
#[allow(clippy::too_many_arguments)]
pub(super) fn write(
    b: &mut gpu_ir::Builder,
    program: &Plan,
    operand: usize,
    ptr: gpu_ir::Value,
    value: Value,
    lanes: u8,
    continuation: bool,
    evict: Evict,
) {
    let fold = program.folds.operand(operand);
    if continuation && !fold.combines_atomically() {
        fold.accumulate(b, ptr, value, lanes, evict)
    } else {
        fold.write(b, ptr, value, evict)
    }
}

/// One selected projection entry for one output. Tile width and bucket count
/// are launch arguments, not separate sources.
pub(super) fn projection_kernel(
    program: &Plan,
    loom: &LiftedProgram,
    output: usize,
    projection: Projection,
) -> Result<Launch, String> {
    output_fold(program, output).projection_kernel(
        projection,
        &entries(loom, output),
        output_elements(&program.index, loom, output)?,
    )
}

/// The buffers one bucket assignment needs: an output whose fold ends in the
/// output buffer writes it directly, one that projects accumulates in scratch.
pub(super) fn allocate(
    builder: &mut recipe::RecipeBuilder,
    program: &Plan,
    loom: &LiftedProgram,
    buckets: &[u32],
) -> Result<Vec<OutputBuffer>, String> {
    (0..loom.outputs().len())
        .map(|output| {
            let fold = program.folds.output(output);
            let buckets = buckets[output];
            let elements = output_elements(&program.index, loom, output)?;
            let nbytes = fold.accumulation_nbytes(elements, buckets)?;
            let projection = fold.projection(buckets);
            let main = match projection {
                Some(_) => {
                    let workspace = builder.scratch(nbytes);
                    builder.memset(recipe::Writable::Workspace(workspace).at(0), 0, nbytes);
                    recipe::Buf::Workspace(workspace)
                }
                None => {
                    if fold.combines_atomically() {
                        builder.memset(recipe::Writable::Output(output).at(0), 0, nbytes);
                    }
                    recipe::Buf::Output(output)
                }
            };
            Ok(OutputBuffer {
                main,
                nbytes,
                projection,
            })
        })
        .collect()
}

/// One selected projection entry and its launch geometry.
pub(super) struct ProjectionModule {
    pub source: ModuleSource,
    pub tile_width: Option<u32>,
}

pub(super) fn add_projections(
    builder: &mut recipe::RecipeBuilder,
    modules: &[Option<ProjectionModule>],
    program: &Plan,
    loom: &LiftedProgram,
    buffers: &[OutputBuffer],
) -> Result<(), String> {
    for (output, buffer) in buffers.iter().enumerate() {
        let Some(projection) = buffer.projection else {
            continue;
        };
        let selected = modules
            .get(output)
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("loom output {output} has no selected projection"))?;
        let elements = output_elements(&program.index, loom, output)?;
        let (grid, block) = projection.launch(elements, selected.tile_width)?;
        let module = selected.source.install(builder);
        let mut launch = builder
            .kernel(module, &selected.source.entry)
            .grid(grid)
            .block(block)
            .read(buffer.main)
            .write(recipe::Writable::Output(output));
        if let Projection::Fold { buckets } = projection {
            let buckets = i32::try_from(buckets).map_err(|_| "loom bucket count exceeds i32")?;
            let width = selected
                .tile_width
                .ok_or("loom fold has no selected tile width")?;
            launch = launch
                .scalar_i32(buckets)
                .scalar_i32(width.trailing_zeros() as i32);
        }
        launch.add();
    }
    Ok(())
}

pub(super) fn output_fold(program: &Plan, output: usize) -> &OutputFold {
    program.folds.output(output)
}

pub(super) fn output_elements(
    index: &Index,
    loom: &LiftedProgram,
    output: usize,
) -> Result<usize, String> {
    let operand = loom.num_inputs + output;
    super::iteration::accumulated_elements(index, operand, &loom.operands[operand])
}

fn entries(loom: &LiftedProgram, output: usize) -> ProjectionEntries {
    ProjectionEntries {
        convert: shared_convert(&loom.name, output),
        fold: shared_bucket_fold(&loom.name, output),
    }
}
