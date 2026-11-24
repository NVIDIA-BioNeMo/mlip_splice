// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::OnceCell;
use std::collections::{BTreeSet, HashMap};

use domain_index::DomainIndex;
use gpu_ir::ValueId;

use super::BackendSettings;
use super::codegen::emit_producer_kernel;
use super::facts::LoomFacts;
use super::iteration::Plan as DomainPlan;
use super::partition::{
    IncidenceProfile, KernelPart, evaluation_profile_with_provenance, plan_shared_work,
    plan_traffic_clusters, plan_value_clusters, plan_with_settings, selected_term_values,
    slice_bindings, statement_count,
};
use crate::LiftedProgram;

mod boundary;
mod candidates;
mod cuts;
mod plan;
#[cfg(test)]
mod tests;

pub(super) use boundary::PartBoundary;
pub(super) use candidates::output_domain_groups;
use candidates::{candidate, consumer_launch};
use cuts::{
    CapProbe, candidate_fits, choose_cut, consumer_statement_count, cut_scalar, materialized_value,
    refine_parts,
};
pub(super) use plan::{
    Candidate, CandidateOrigin, GuardedPlan, GuardedRegion, Launch, MaterializedValue, PlanIr,
    ReductionProvenance, ReductionRegion, WriteMode,
};

#[derive(Clone, Copy, Debug)]
pub(super) struct Settings {
    pub default_register_cap: usize,
    pub default_cut_limit: usize,
    pub min_cut_closure: usize,
    pub cut_candidates: usize,
    pub value_cluster_pressure_factor: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Limits {
    pub statement_cap: usize,
    pub register_cap: usize,
    pub cut_limit: usize,
    pub plan_clusters: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            default_register_cap: 128,
            default_cut_limit: 16,
            min_cut_closure: 8,
            cut_candidates: 24,
            value_cluster_pressure_factor: 4,
        }
    }
}

impl PlanIr {
    #[cfg(test)]
    pub(super) fn build(
        loom: &LiftedProgram,
        domain: DomainPlan,
        parts: Vec<KernelPart>,
        register_cap: usize,
        cut_limit: usize,
    ) -> Result<Self, String> {
        Self::build_with_settings(
            loom,
            domain,
            parts,
            register_cap,
            cut_limit,
            &Settings::default(),
        )
    }

    pub(super) fn build_with_settings(
        loom: &LiftedProgram,
        domain: DomainPlan,
        parts: Vec<KernelPart>,
        register_cap: usize,
        cut_limit: usize,
        settings: &Settings,
    ) -> Result<Self, String> {
        let boundaries = parts
            .iter()
            .map(|part| PartBoundary::of(part, 0, false))
            .collect();
        let consumer = |part| consumer_launch(loom, &domain, part);
        let baseline_launches = parts.iter().cloned().map(consumer).collect::<Vec<_>>();
        let baseline = candidate(loom, &domain, Vec::new(), baseline_launches)?;
        let mut candidates = vec![baseline];
        if parts.len() > 1
            && cut_limit > 0
            && LoomFacts::analyze(loom).register_pressure() as usize > register_cap
            && domain.index.fiber_bridge().is_none()
            && let Some(value) = choose_cut(loom, &parts, &domain, settings)?
        {
            let launches = std::iter::once(Launch::Producer {
                output: 0,
                inputs: Vec::new(),
            })
            .chain(parts.into_iter().map(consumer))
            .collect::<Vec<_>>();
            candidates.push(candidate(loom, &domain, vec![value], launches)?);
        }
        Ok(Self {
            domain,
            candidates,
            boundaries,
        })
    }

    #[cfg(test)]
    pub(super) fn build_bounded(
        loom: &LiftedProgram,
        domain: DomainPlan,
        statement_cap: usize,
        register_cap: usize,
        cut_limit: usize,
        incidence: &OnceCell<Result<IncidenceProfile, String>>,
        plan_clusters: bool,
    ) -> Result<Self, String> {
        Self::build_bounded_with_settings(
            loom,
            domain,
            Limits {
                statement_cap,
                register_cap,
                cut_limit,
                plan_clusters,
            },
            incidence,
            &BackendSettings::default(),
        )
    }

    pub(super) fn build_bounded_with_settings(
        loom: &LiftedProgram,
        domain: DomainPlan,
        limits: Limits,
        incidence: &OnceCell<Result<IncidenceProfile, String>>,
        settings: &BackendSettings,
    ) -> Result<Self, String> {
        let Limits {
            statement_cap,
            register_cap,
            cut_limit,
            plan_clusters,
        } = limits;
        let mut parts = Vec::new();
        let probe_block_size = settings.compiler.block_sizes[0];
        let (initial, allow_partial_splits) =
            match plan_with_settings(loom, statement_cap, &settings.partition) {
                Ok(parts) => (parts, true),
                Err(_) => (
                    plan_with_settings(loom, usize::MAX, &settings.partition)?,
                    true,
                ),
            };
        let probe = CapProbe {
            domain: &domain,
            loom,
            statement_cap,
            allow_partial_splits,
            block_size: probe_block_size,
        };
        for part in initial {
            refine_parts(&probe, part, &mut parts, 0)?;
        }
        if parts.iter().all(|(_, oversized, _)| !oversized) {
            let boundaries = parts
                .iter()
                .map(|(part, oversized, bisections)| {
                    PartBoundary::of(part, *bisections, *oversized)
                })
                .collect::<Vec<_>>();
            let bounded = parts
                .into_iter()
                .map(|(part, _, _)| part)
                .collect::<Vec<_>>();
            let grouped = plan_shared_work(
                incidence
                    .get_or_init(|| IncidenceProfile::prepare(loom))
                    .as_ref()
                    .map_err(String::clone)?,
                statement_cap,
                &bounded,
                &settings.partition,
            )?;
            let value_clusters = if plan_clusters
                && LoomFacts::analyze(loom).register_pressure() as usize
                    > register_cap.saturating_mul(settings.schedule.value_cluster_pressure_factor)
            {
                let profile = incidence
                    .get_or_init(|| IncidenceProfile::prepare(loom))
                    .as_ref()
                    .map_err(String::clone)?;
                plan_value_clusters(
                    profile,
                    settings.partition.value_cluster_cap,
                    &bounded,
                    &settings.partition,
                )?
            } else {
                None
            };
            // A fiber walk streams finer-domain operands exactly once, but
            // coarse operands gathered through the walk (directly or via a
            // aligned view) are re-read once per fiber element — the same
            // L2 reuse a split protects on the plain scattered-gather pass.
            // Gather reuse is therefore judged against the walked finer
            // extent, and the traffic cut proposes on every arm; kiln
            // arbitrates it against the value cut and the fused form.
            let traffic_clusters = if plan_clusters {
                let profile = incidence
                    .get_or_init(|| IncidenceProfile::prepare(loom))
                    .as_ref()
                    .map_err(String::clone)?;
                // Every axis the index walks, not just the axes the program
                // declared: an ellipsis spec adds axes, and the fiber bridge
                // names one of the index's own.
                let iter_domains = (0..domain.index.num_axes())
                    .map(|axis| domain.index.iter_domain(axis))
                    .collect::<Vec<_>>();
                // An iteration domain is named by its identity in the graph,
                // which is what holds every domain the index can reach; the
                // program's own list covers only the ones it declared.
                let graph = domain.index.graph();
                let extent_of = |d: i32, unmapped: u64| match d {
                    d if d >= 0 => graph.domain_extent(d).max(0) as u64,
                    _ => unmapped,
                };
                let mut iter_positions: u64 =
                    iter_domains.iter().map(|&d| extent_of(d, 1)).product();
                if let Some((map_idx, axis)) = domain.index.fiber_bridge() {
                    let finer = graph
                        .domain_extent(graph.index_maps[map_idx].source_id)
                        .max(0) as u64;
                    let coarse = extent_of(iter_domains[axis], 1).max(1);
                    iter_positions = iter_positions / coarse * finer;
                }
                plan_traffic_clusters(
                    loom,
                    &iter_domains,
                    iter_positions,
                    profile,
                    &bounded,
                    &settings.partition,
                )?
            } else {
                None
            };
            // Sibling kernels per output-domain group, all iterating the
            // shared domain — the coarse-domain outputs keep their scatter
            // reduction instead of forcing a starved own-domain iteration.
            let output_split = {
                let groups = output_domain_groups(loom, &domain);
                let split = groups
                    .iter()
                    .flat_map(|group| {
                        let group = group.iter().copied().collect::<BTreeSet<_>>();
                        bounded
                            .iter()
                            .filter_map(|part| part.restrict_to_outputs(&group))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                (groups.len() >= 2 && split.len() > bounded.len()).then_some(split)
            };
            let mut planned = Self::build_with_settings(
                loom,
                domain,
                bounded,
                register_cap,
                cut_limit,
                &settings.schedule,
            )?;
            // The parts are the same ones, so this only adds what index map
            // knows and `build_with_settings` cannot see.
            planned.boundaries = boundaries;
            let domain = &planned.domain;
            planned.candidates.retain(|candidate| {
                candidate.materialized.is_empty()
                    || candidate_fits(domain, loom, candidate, statement_cap, probe_block_size)
                        .unwrap_or(false)
            });
            // Sibling plans of the same program: each competes with the fused
            // baseline and earns its compile slot only by fitting the cap. The
            // partition plans lead the list, so kiln times them first; the
            // cluster plans follow it.
            for (parts, origin, leading) in [
                (output_split, CandidateOrigin::Structural, true),
                (grouped, CandidateOrigin::Structural, true),
                (value_clusters, CandidateOrigin::Structural, false),
                (traffic_clusters, CandidateOrigin::Traffic, false),
            ] {
                let Some(parts) = parts else { continue };
                let launches = parts
                    .into_iter()
                    .map(|part| consumer_launch(loom, domain, part))
                    .collect();
                let mut sibling = candidate(loom, domain, Vec::new(), launches)?;
                sibling.origin = origin;
                if candidate_fits(domain, loom, &sibling, statement_cap, probe_block_size)? {
                    if leading {
                        planned.candidates.insert(1, sibling);
                    } else {
                        planned.candidates.push(sibling);
                    }
                }
            }
            return Ok(planned);
        }
        if domain.index.fiber_bridge().is_some() {
            return Err("loom cannot materialize an oversized fiber iteration".into());
        }

        let profile = evaluation_profile_with_provenance(loom)?;
        let (outputs, body) = (&profile.outputs, &profile.body);
        let boundaries = parts
            .iter()
            .map(|(part, oversized, bisections)| PartBoundary::of(part, *bisections, *oversized))
            .collect();
        let mut materialized = Vec::new();
        let mut launches = Vec::new();
        for (part, oversized, _) in parts {
            if oversized {
                loop {
                    let count = consumer_statement_count(
                        &domain,
                        loom,
                        &part,
                        &materialized,
                        probe_block_size,
                    )?;
                    if count <= statement_cap {
                        break;
                    }
                    if materialized.len() >= cut_limit {
                        return Err(format!(
                            "loom output DAG still needs {count} statements after {cut_limit} cuts, exceeding cap {statement_cap}"
                        ));
                    }
                    let f = body.function();
                    let (_, roots) = selected_term_values(f, &body.steps(), outputs, &part)?;
                    let mut reads = std::collections::BTreeSet::new();
                    for root in &roots {
                        root.add_reads(f, &mut reads);
                    }
                    let reads = reads.into_iter().collect::<Vec<_>>();
                    let sliced = slice_bindings(f, &body.steps(), &reads)?;
                    // Scoring a cut emits full kernels, so only the
                    // Only the configured number of bindings with the largest estimated
                    // closures are scored exactly; the estimate double-counts
                    // shared work, which is fine for ranking — planning stays
                    // bounded instead of quadratic in the program size.
                    let mut closure_est = HashMap::<ValueId, u64>::with_capacity(sliced.len());
                    let mut candidates = Vec::new();
                    for &op in &sliced {
                        let defined = f.op(op).results[0];
                        let est = 1 + f
                            .reads(f.op(op).operands[0])
                            .iter()
                            .filter_map(|read| closure_est.get(read))
                            .sum::<u64>();
                        closure_est.insert(defined, est.min(u64::MAX / 2));
                        let Some(scalar) = cut_scalar(f, defined) else {
                            continue;
                        };
                        let Some(name) = f.name(defined) else {
                            continue;
                        };
                        if materialized
                            .iter()
                            .any(|value: &MaterializedValue| value.name == name)
                        {
                            continue;
                        }
                        candidates.push((est, name.to_owned(), defined, scalar));
                    }
                    // The estimate overcounts shared work, so candidates
                    // whose estimate exceeds the cap may still fit: keep the
                    // largest that estimate under the cap and the smallest
                    // above it, half the budget each.
                    candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                    let cap = statement_cap as u64;
                    let split = candidates.partition_point(|(est, _, _, _)| *est > cap);
                    let above = split.min(settings.schedule.cut_candidates / 2);
                    let below =
                        (candidates.len() - split).min(settings.schedule.cut_candidates - above);
                    candidates = candidates[split - above..split + below].to_vec();
                    let mut best = None;
                    for (_, _, root, scalar) in candidates {
                        let value = materialized_value(loom, &domain, body, root, scalar, 1)?;
                        let mut trial = materialized.clone();
                        trial.push(value);
                        let output = trial.len() - 1;
                        let inputs = (0..output).collect::<Vec<_>>();
                        let producer = emit_producer_kernel(
                            &domain,
                            loom,
                            probe_block_size,
                            output,
                            &inputs,
                            &trial,
                        )?;
                        let producer_count = statement_count(producer.body());
                        if producer_count > statement_cap {
                            continue;
                        }
                        let consumer_count = consumer_statement_count(
                            &domain,
                            loom,
                            &part,
                            &trial,
                            probe_block_size,
                        )?;
                        if consumer_count >= count {
                            continue;
                        }
                        let score = (consumer_count.max(producer_count), consumer_count);
                        if best
                            .as_ref()
                            .is_none_or(|(best_score, _, _)| score < *best_score)
                        {
                            best = Some((score, trial.pop().unwrap(), inputs));
                        }
                    }
                    let Some((_, value, inputs)) = best else {
                        return Err(format!(
                            "loom output DAG needs {count} statements and has no bounded materialization cut under cap {statement_cap}"
                        ));
                    };
                    let output = materialized.len();
                    materialized.push(value);
                    launches.push(Launch::Producer { output, inputs });
                }
            }
            launches.push(consumer_launch(loom, &domain, part));
        }
        let planned = candidate(loom, &domain, materialized, launches)?;
        Ok(Self {
            domain,
            candidates: vec![planned],
            boundaries,
        })
    }
}
