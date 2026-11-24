// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::iteration::output_domain_groups;
use super::{
    Candidate, CandidateOrigin, DomainIndex, FamilyPartition, IncidenceProfile, KernelPart, Launch,
    LiftedProgram, Placement, Plan, Slot, partition_cost,
};

fn family_program(loom: &LiftedProgram, name: String, outputs: &[usize]) -> LiftedProgram {
    let mut family = loom.clone();
    family.keep_outputs(outputs);
    family.request_mut().name = name;
    family
}

/// The iteration a family reaches, and whether that iteration walks a fiber.
fn placement(family: &LiftedProgram) -> Result<(Placement, bool), String> {
    if let Some(program) = Plan::prepare_fiber_direct(family)? {
        return Ok((
            Placement::FiberDirect(0),
            program.index.fiber_bridge().is_some(),
        ));
    }
    Ok((
        Placement::Default,
        Plan::prepare(family)?.index.fiber_bridge().is_some(),
    ))
}

/// The partition, and for each of its families whether that family iterates
/// coarse — the caller's admission rule reads those flags.
fn build_partition(
    loom: &LiftedProgram,
    prefix: &str,
    groups: &[&Vec<usize>],
) -> Result<(FamilyPartition, Vec<bool>), String> {
    let mut families = Vec::with_capacity(groups.len());
    let mut fibers = Vec::with_capacity(groups.len());
    for (family_index, outputs) in groups.iter().enumerate() {
        let family = family_program(loom, format!("{prefix}d{family_index}"), outputs);
        let (placement, fiber) = placement(&family)?;
        fibers.push(fiber);
        families.push((family, (*outputs).clone(), placement));
    }
    Ok((families, fibers))
}

/// Every way the operation's outputs split into per-domain families. The
/// domain-forced groups always split; the free outputs either keep a family of
/// their own — one launch per cell of the finest domain — or ride along in a
/// coarse-iterating group, which reaches them through data that group already
/// holds instead of gathering the whole neighbourhood a second time. Which is
/// faster is a hardware question, so every placement becomes a recipe arm.
pub(super) fn output_domain_families(
    loom: &LiftedProgram,
    plan_clusters: bool,
) -> Result<Option<Vec<FamilyPartition>>, String> {
    let baseline = Plan::prepare(loom)?;
    let default = (0..loom.num_axes)
        .map(|axis| baseline.index.iter_domain(axis))
        .collect::<Vec<_>>();
    let groups = output_domain_groups(loom, &default);
    if groups.len() < 2 {
        return Ok(None);
    }

    let free = groups.get(&default);
    let anchored = groups
        .iter()
        .filter(|(key, _)| **key != default)
        .map(|(_, outputs)| outputs)
        .collect::<Vec<_>>();
    let prefix = format!("{}_", loom.name);
    let mut partitions = Vec::new();
    // The forced split earns its arm when any family walks a fiber; a hosted
    // split earns its arm only when the host itself does, since hosting the
    // free outputs is exactly a bet on the host's gathered data.
    let (split, fibers) = build_partition(loom, &prefix, &groups.values().collect::<Vec<_>>())?;
    if fibers.iter().any(|&fiber| fiber) {
        partitions.push(split);
    }
    // With one anchored group the hosted placement holds every output in one
    // family, which is the undivided partition below.
    if let Some(free) = free.filter(|_| anchored.len() > 1) {
        for host in 0..anchored.len() {
            let merged = anchored[host]
                .iter()
                .chain(free)
                .copied()
                .collect::<Vec<_>>();
            let mut placed = vec![&merged];
            placed.extend(
                anchored
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != host)
                    .map(|(_, outputs)| *outputs),
            );
            let (partition, fibers) = build_partition(loom, &format!("{prefix}h{host}"), &placed)?;
            if fibers[0] {
                partitions.push(partition);
            }
        }
    }
    // The undivided partitions: one family holding every output, iterating a
    // coarse anchor with the finer-domain outputs written fiber-direct inside
    // the fiber loop — no atomics, no zero-init, per-position operands
    // gathered once per position. One arm per promoted placement, because
    // which anchor to walk is a traffic question the hardware answers.
    if plan_clusters {
        let all = (0..loom.outputs().len()).collect::<Vec<_>>();
        for index in 0..Plan::fiber_direct_placements(loom)? {
            let placement = Placement::FiberDirect(index);
            let family = family_program(loom, format!("{prefix}snd{index}_d0"), &all);
            let Some(program) = Plan::prepare_for(&family, placement)? else {
                continue;
            };
            if program.index.fiber_bridge().is_some() {
                partitions.push(vec![(family, all.clone(), placement)]);
            }
        }
    }
    Ok((!partitions.is_empty()).then_some(partitions))
}

pub(super) fn same_slot(left: &Slot, right: &Slot) -> bool {
    left.kind == right.kind
        && left.source_id == right.source_id
        && left.target_id == right.target_id
        && left.other_target_id == right.other_target_id
}

pub(super) fn input_bindings(
    num_data_inputs: usize,
    local: &[Slot],
    all: &[Slot],
) -> Result<Vec<::recipe::BufRef>, String> {
    let mut bindings = (0..num_data_inputs)
        .map(::recipe::BufRef::Input)
        .collect::<Vec<_>>();
    for slot in local {
        let index = all
            .iter()
            .position(|candidate| same_slot(candidate, slot))
            .ok_or("loom output-family slot is absent from the shared inputs")?;
        let first = num_data_inputs
            + all[..index]
                .iter()
                .map(|slot| slot.kind.array_count())
                .sum::<usize>();
        bindings.extend((first..first + slot.kind.array_count()).map(::recipe::BufRef::Input));
    }
    Ok(bindings)
}

/// `signature` is the shared one every arm is bound onto, so the bound recipe
/// states the buffers the choice hands it rather than the ones it was compiled
/// against.
pub(super) fn bind_recipe(
    recipe: &::recipe::Recipe,
    num_data_inputs: usize,
    local_slots: &[Slot],
    all_slots: &[Slot],
    outputs: &[usize],
    signature: &::recipe::RecipeSignature,
) -> Result<::recipe::ProvedRecipe, String> {
    let mut builder = ::recipe::RecipeBuilder::new(signature.num_inputs(), signature.num_outputs());
    builder
        .splice(
            recipe.clone().into_fragment(),
            &input_bindings(num_data_inputs, local_slots, all_slots)?,
            &outputs
                .iter()
                .copied()
                .map(::recipe::BufRef::Output)
                .collect::<Vec<_>>(),
            &[],
        )
        .map_err(|error| format!("{error:?}"))?;
    builder
        .finish(signature.clone())
        .map_err(|error| format!("{error:?}"))
}

pub(super) fn consumer_parts(candidate: &Candidate) -> Vec<KernelPart> {
    candidate
        .launches
        .iter()
        .filter_map(|launch| match launch {
            Launch::Consumer { part, .. } => Some(part.clone()),
            Launch::Producer { .. } => None,
        })
        .collect()
}

/// Every non-baseline partitioning kiln can arbitrate, ordered by the
/// seam-metric partition cost — weighted max per-launch footprint, then
/// duplication (`PartitionCost::selection_key`). The metric ranks the
/// structural candidates, it does not eliminate them: which partition wins is
/// measured, and the ordering only decides who compiles first when the budget
/// runs short. Materialized cut candidates stay the fallback: the profile
/// prices their consumers as if they recomputed the workspace value. The
/// traffic candidate answers a question the seam metric cannot see — L2
/// residency of the gathered working set — so it joins as its own alternative.
pub(super) fn select_alternatives(
    loom: &LiftedProgram,
    candidates: &[Candidate],
    baseline: usize,
    incidence: &std::cell::OnceCell<Result<IncidenceProfile, String>>,
    settings: &super::super::partition::Settings,
) -> Result<Vec<usize>, String> {
    let (traffic, competitors): (Vec<usize>, Vec<usize>) = (0..candidates.len())
        .filter(|&index| index != baseline && candidates[index].materialized.is_empty())
        .partition(|&index| candidates[index].origin == CandidateOrigin::Traffic);
    let structural: Vec<usize> = if competitors.len() < 2 {
        competitors
            .first()
            .copied()
            .or_else(|| {
                (0..candidates.len())
                    .find(|&index| index != baseline && !traffic.contains(&index))
                    .or_else(|| (0..candidates.len()).find(|&index| index != baseline))
                    .filter(|index| !traffic.contains(index))
            })
            .into_iter()
            .collect()
    } else {
        let mut ranked = Vec::with_capacity(competitors.len());
        for index in competitors {
            let profile = incidence
                .get_or_init(|| IncidenceProfile::prepare(loom))
                .as_ref()
                .map_err(String::clone)?;
            let cost = partition_cost(profile, &consumer_parts(&candidates[index]), settings)?;
            ranked.push((cost.selection_key(settings), index));
        }
        ranked.sort();
        ranked.into_iter().map(|(_, index)| index).collect()
    };
    Ok(structural
        .into_iter()
        .chain(traffic.first().copied())
        .collect())
}
