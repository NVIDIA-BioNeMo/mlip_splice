// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{HashMap, Kernel, LiftedProgram, Placement, SourceAlias};

/// One family of a split: the sub-program, the outputs it writes, and the
/// iteration its placement reaches.
pub(super) type OutputFamily = (LiftedProgram, Vec<usize>, Placement);

/// One complete way to split the outputs into families.
pub(super) type FamilyPartition = Vec<OutputFamily>;

/// What one candidate module asks the executor for: the PTX to assemble, the
/// options that assemble it, the resource bounds its image must fit to stand,
/// and the entry it launches.
///
/// A register cap is an option over the same source, never another render, so a
/// capped module and its base share one `SourceImage` and one entry. The cap is
/// worth assembling only when the base does not already spill, which is what
/// `prerequisite` states about that base.
#[derive(Clone, Debug)]
pub(in crate::backend) struct ModuleSource {
    source: ::recipe::SourceImage,
    options: ::recipe::PtxCompileOptions,
    prerequisite: Option<::recipe::ResourceBounds>,
    resource_bounds: ::recipe::ResourceBounds,
    pub(in crate::backend) entry: String,
}

impl ModuleSource {
    /// Install this module into `builder`, interning the request it resolves
    /// from — and, for a capped module, the uncapped request it presupposes.
    pub(in crate::backend) fn install(
        &self,
        builder: &mut ::recipe::RecipeBuilder,
    ) -> ::recipe::ModuleId {
        let requires = self
            .prerequisite
            .map(|bounds| {
                let uncapped = builder.compile_request(::recipe::CompileRequest::unguarded(
                    vec![self.source.clone()],
                    ::recipe::PtxCompileOptions {
                        max_registers: None,
                        ..self.options
                    },
                ));
                (uncapped, bounds)
            })
            .into_iter()
            .collect();
        let request = builder.compile_request(::recipe::CompileRequest {
            sources: vec![self.source.clone()],
            options: self.options,
            requires,
        });
        builder.requested_module(request, Vec::new(), self.resource_bounds)
    }
}

/// `side` names the kernels that stand outside the slot table — projections and
/// whole-launch reductions — as `(PTX, entry)`; the modules that serve them come
/// back in that order.
pub(super) fn request_modules<'a>(
    units: &'a [SourceAlias],
    side: &[(&'a [u8], &str)],
    spill_band_bytes: u32,
    target: ::recipe::PtxTarget,
) -> Result<(Vec<ModuleSource>, Vec<ModuleSource>), String> {
    let mut indices = HashMap::<*const u8, usize>::new();
    let mut sources = Vec::<&'a [u8]>::new();
    let mut intern = |source: &'a [u8]| {
        *indices.entry(source.as_ptr()).or_insert_with(|| {
            sources.push(source);
            sources.len() - 1
        })
    };
    let unit_sources = units
        .iter()
        .map(|unit| intern(unit.source.as_slice()))
        .collect::<Vec<_>>();
    let side_sources = side
        .iter()
        .map(|(source, _)| intern(source))
        .collect::<Vec<_>>();
    // The slices borrow renders that outlive this call; the map they land in
    // does not, so the blobs are built before anything else moves.
    let blobs = ::recipe::Blob::zstd_spilled_many(sources)
        .map_err(|error| format!("loom cannot store its rendered PTX: {error}"))?;
    let image = |source: usize| ::recipe::SourceImage {
        kind: ::recipe::SourceKind::Ptx,
        target,
        blob: blobs[source].clone(),
    };

    let mut aliases = Vec::new();
    for (unit, source) in units.iter().zip(unit_sources) {
        let module = ModuleSource {
            source: image(source),
            options: ::recipe::PtxCompileOptions {
                max_registers: unit.cap,
                assembly_mode: unit.assembly_mode,
            },
            prerequisite: unit.cap.map(|cap| ::recipe::ResourceBounds {
                min_registers: Some(cap.saturating_add(1)),
                max_local_bytes: Some(0),
            }),
            resource_bounds: ::recipe::ResourceBounds {
                min_registers: None,
                max_local_bytes: unit.cap.map(|_| spill_band_bytes),
            },
            entry: unit.entry.clone(),
        };
        aliases.push(module);
    }
    let side = side
        .iter()
        .zip(side_sources)
        .map(|((_, entry), source)| ModuleSource {
            source: image(source),
            options: ::recipe::PtxCompileOptions {
                max_registers: None,
                assembly_mode: ::recipe::PtxAssemblyMode::Relocatable,
            },
            prerequisite: None,
            resource_bounds: ::recipe::ResourceBounds::default(),
            entry: (*entry).to_owned(),
        })
        .collect();
    Ok((aliases, side))
}

/// One record per source unit the compiler produced. The arm's order and cap are stamped on by the caller that chose
/// them, so a record joins its arm by identity and never by parsing an entry
/// name.
pub(super) fn unit_records(
    units: &[SourceAlias],
    dump_ptx: bool,
    placement: Placement,
) -> Vec<Kernel> {
    let placement = placement.name();
    units
        .iter()
        .map(|unit| {
            let rank = unit.rank;
            let (flavor, part, slot) = (rank.flavor.name(), rank.part, rank.block);
            Kernel {
                graph: 0,
                // Stamped by the arm that asked for this unit.
                order: "",
                statement_cap: 0,
                placement: placement.clone(),
                entry: unit.entry.clone(),
                flavor,
                part,
                slot,
                register_cap: unit.cap,
                assembly_mode: assembly_mode_name(unit.assembly_mode),
                statements: unit.statements,
                addition_chain_within_thread: unit.addition_chain.map(|chain| chain.within_thread),
                addition_chain_folds_across_threads: unit
                    .addition_chain
                    .is_some_and(|chain| chain.folds_across_threads),
                ptx_bytes: unit.source.len(),
                ptx_hash: crate::digest(&unit.source),
                ptx: dump_ptx.then(|| String::from_utf8_lossy(&unit.source).into_owned()),
                admitted: true,
            }
        })
        .collect()
}

fn assembly_mode_name(mode: ::recipe::PtxAssemblyMode) -> &'static str {
    match mode {
        ::recipe::PtxAssemblyMode::EnvironmentDefault => "environment-default",
        ::recipe::PtxAssemblyMode::Relocatable => "relocatable",
        ::recipe::PtxAssemblyMode::WholeProgram => "whole-program",
    }
}
