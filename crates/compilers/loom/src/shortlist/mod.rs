// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod compilations;
mod ranking;
use crate::{CompileOptions, CompileProfile, CompiledProgram};
pub use ranking::GpuTarget;
use ranking::{Score, boltzmann_order};
use std::collections::BTreeSet;

/// Milliseconds of wall clock a realizer needs for a shortlist: it assembles on
/// `workers` threads while one tuner thread walks the candidates, so the two
/// lanes overlap and the slower one sets the time. The assembly lane is a floor,
/// since work arrives in bursts and a worker can idle; the realizer only waits
/// on it when a candidate needs an image nobody has built yet.
fn predicted_wall_ms(assembly_ms: f64, tuning_ms: f64, workers: usize) -> f64 {
    (assembly_ms / workers as f64).max(tuning_ms)
}

pub(crate) fn select(
    candidates: &[CompiledProgram],
    profile: &mut CompileProfile,
    options: CompileOptions,
) -> Result<Vec<usize>, String> {
    let _shortlisting = chronicle::range(|| {
        format!(
            "loom.shortlist candidates={} shortlist_budget={:?}",
            candidates.len(),
            options.shortlist_budget
        )
    });
    let Some(limit) = options
        .shortlist_budget
        .map(|budget| budget.as_secs_f64() * 1000.0)
    else {
        let indices = (0..candidates.len()).collect::<Vec<_>>();
        chronicle::mark(|| format!("loom.shortlist_unbounded selected={}", indices.len()));
        profile.shortlisted_candidates = indices.clone();
        return Ok(indices);
    };
    let workers = options.assumed_assembly_workers.get();
    let tuning_cost_ms = options.tuning_cost_per_candidate.as_secs_f64() * 1000.0;
    let target = match options.target {
        Some(target) => target.validate()?,
        None => GpuTarget::device()?,
    };
    if candidates.len() != profile.arms.len() {
        return Err("loom source candidate/profile arity mismatch".into());
    }
    let scores = candidates
        .iter()
        .zip(&profile.arms)
        .map(|(candidate, arm)| {
            let graph = profile
                .frontier_graphs
                .iter()
                .find(|graph| graph.graph == arm.graph)
                .ok_or("loom source candidate has no graph work record")?;
            Ok(Score {
                work: u128::from(graph.before_accumulation) + u128::from(graph.after_accumulation),
                parallelism: target.bottleneck(candidate.recipe.body())?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let compilations = compilations::Compilations::new(candidates);
    chronicle::mark(|| {
        format!(
            "loom.shortlist_scored candidates={} tuning_cost_ms={tuning_cost_ms} workers={workers}",
            scores.len(),
        )
    });
    let mut selected = vec![false; candidates.len()];
    let mut indices = Vec::new();
    let mut paid = BTreeSet::new();
    let mut covered = BTreeSet::new();
    let mut assembly_ms = 0.0;
    let mut tuning_ms = 0.0;
    let order = boltzmann_order(&scores);
    chronicle::mark(|| format!("loom.shortlist_ordered candidates={}", order.len()));
    for index in order {
        if selected[index] {
            continue;
        }
        let (closure, units) = compilations.closure(index, &selected, &covered);
        let assembly = compilations.additional_cost_ms(&units, &paid);
        let tuning = tuning_cost_ms * closure.len() as f64;
        let predicted = predicted_wall_ms(assembly_ms + assembly, tuning_ms + tuning, workers);
        if !indices.is_empty() && assembly + tuning > 0.0 && predicted > limit {
            chronicle::mark(|| {
                format!(
                    "loom.shortlist_refused candidate={index} closure={} \
                     marginal_assembly_ms={assembly} marginal_tuning_ms={tuning} \
                     predicted_ms={predicted} shortlist_budget_ms={limit}",
                    closure.len()
                )
            });
            continue;
        }
        assembly_ms += assembly;
        tuning_ms += tuning;
        // Exactly what `assembly` was priced on: a unit already paid for is not
        // added twice, so a reader summing this field reaches the recipe total.
        let (admitted, added) = (closure.len(), units.difference(&paid).count());
        paid.extend(units);
        for index in closure {
            selected[index] = true;
            compilations.cover_entries(index, &mut covered);
            indices.push(index);
        }
        chronicle::mark(|| {
            format!(
                "loom.shortlist_admitted candidate={index} closure={admitted} units={added} \
                 selected={} assembly_ms={assembly_ms} tuning_ms={tuning_ms}",
                indices.len()
            )
        });
    }
    let estimated = predicted_wall_ms(assembly_ms, tuning_ms, workers);
    chronicle::mark(|| {
        format!(
            "loom.shortlist_result selected={} units={} assembly_ms={assembly_ms} \
             tuning_ms={tuning_ms} predicted_ms={estimated} shortlist_budget_ms={limit}",
            indices.len(),
            paid.len()
        )
    });
    profile.shortlisted_candidates = indices.clone();
    profile.shortlist_estimated_ms = Some(estimated);
    profile.shortlist_compile_units = Some(paid.len());
    Ok(indices)
}

#[cfg(test)]
mod tests;
