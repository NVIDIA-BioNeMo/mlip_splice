// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use super::value_clusters::{ValueCluster, cluster_kernel_parts, coalesce_atoms, term_atoms};
use gpu_ir::{OpKind, ValueId};

use super::{
    AffinityCluster, BTreeSet, HashMap, IncidenceProfile, KernelPart, LiftedProgram, PartitionCost,
    Settings, selected_term_values, slice_bindings,
};

pub(super) struct InputTraffic {
    pub(super) bytes: Vec<u64>,
    pub(super) gathered: Vec<bool>,
}

fn operand_positions(loom: &LiftedProgram, operand: usize) -> u64 {
    loom.operands[operand]
        .axis_domains
        .iter()
        .map(|&domain| match domain {
            d if d >= 0 => loom.domains[d as usize].extent.max(0) as u64,
            _ => 1,
        })
        .product()
}

/// Whole-operand slice bytes, the per-entry weight. Entries are keyed per
/// (operand, element), so this overweights many-element operands relative to
/// their true per-element slice — measured to cut better than the honest
/// per-element weight: the winning cuts balance element counts within the
/// big gathered operands, and honest weights let a launch soak up most of
/// one operand's elements before the budget binds (mp0-S-bwd2 0.88 vs 1.12).
fn operand_slice_bytes(loom: &LiftedProgram, operand: usize) -> u64 {
    let binding = &loom.operands[operand];
    operand_positions(loom, operand)
        * binding.lane_extent.max(0) as u64
        * binding.dtype.size_bytes() as u64
}

fn input_traffic(
    loom: &LiftedProgram,
    profile: &IncidenceProfile,
    iter_domains: &[i32],
    iter_positions: u64,
) -> Option<InputTraffic> {
    let num_definitions = profile.bindings.definitions();
    let mut bytes = vec![0u64; profile.bindings.inputs()];
    let mut gathered = vec![false; profile.bindings.inputs()];
    for (value, intern) in profile.bindings.input_entries() {
        // Every entry has to be an element of an operand for the byte weights to
        // describe the whole gathered set; an input the profile reaches some
        // other way leaves a hole, and a plan built over a hole is not one.
        let operand = profile.input_operand(value)?;
        let binding = loom.operands.get(operand)?;
        let index = intern - num_definitions;
        bytes[index] = operand_slice_bytes(loom, operand);
        let off_iteration = binding
            .axis_domains
            .iter()
            .zip(iter_domains)
            .any(|(&domain, &iter)| domain >= 0 && domain != iter);
        gathered[index] = off_iteration && 2 * operand_positions(loom, operand) <= iter_positions;
    }
    Some(InputTraffic { bytes, gathered })
}

/// The fused work recut so each launch's gathered working set shrinks:
/// pairwise agglomeration by shared bytes — duplicating a load across
/// launches re-reads its slice, splitting an accumulator re-reads and
/// re-writes it — under a budget on each launch's gathered bytes. The
/// candidate emits only when the fused gathered set is large enough to
/// plausibly exceed an L2; whether the relief beats the seam on the actual
/// device is kiln's call. `iter_positions` counts the per-position visits a
/// launch performs — over a fiber bridge that is the walked finer extent,
/// where coarse operands gathered through the walk are re-read once per
/// fiber element.
pub(in super::super) fn plan_traffic_clusters(
    loom: &LiftedProgram,
    iter_domains: &[i32],
    iter_positions: u64,
    profile: &IncidenceProfile,
    baseline: &[KernelPart],
    settings: &Settings,
) -> Result<Option<Vec<KernelPart>>, String> {
    if baseline.is_empty() {
        return Ok(None);
    }
    let Some(traffic) = input_traffic(loom, profile, iter_domains, iter_positions) else {
        return Ok(None);
    };
    let fused_gathered: u64 = traffic
        .bytes
        .iter()
        .zip(&traffic.gathered)
        .filter(|&(_, &gathered)| gathered)
        .map(|(&bytes, _)| bytes)
        .sum();
    if fused_gathered < settings.traffic_cluster_min_gathered_bytes {
        return Ok(None);
    }
    let Some(atoms) = term_atoms(
        profile,
        settings.max_value_cluster_terms,
        settings.max_shared_work_closure_entries,
        BTreeSet::len,
    ) else {
        return Ok(None);
    };
    let num_definitions = profile.bindings.definitions();
    let clusters = coalesce_atoms(&atoms, num_definitions);
    // Seeds with identical gathered-entry sets are budget-equivalent — any
    // agglomeration order merges them for free — so they collapse up front,
    // keeping the pairwise loop over distinct gathered signatures only.
    let mut buckets: BTreeMap<Vec<usize>, ValueCluster> = BTreeMap::new();
    for cluster in clusters {
        let signature = cluster
            .input_loads
            .iter()
            .map(|&load| load - num_definitions)
            .filter(|&index| traffic.gathered[index])
            .collect::<Vec<_>>();
        match buckets.entry(signature) {
            std::collections::btree_map::Entry::Vacant(slot) => {
                slot.insert(cluster);
            }
            std::collections::btree_map::Entry::Occupied(mut slot) => {
                slot.get_mut().absorb(cluster);
            }
        }
    }
    let clusters = buckets.into_values().collect::<Vec<_>>();
    if clusters.len() < 2 || clusters.len() > settings.traffic_cluster_max_seeds {
        return Ok(None);
    }
    let num_inputs = loom.num_inputs;
    let acc_bytes = |&(output, _element): &(usize, usize)| -> u64 {
        operand_slice_bytes(loom, num_inputs + output)
    };
    let budget = fused_gathered / 2;
    let mut clusters = agglomerate_traffic(clusters, &traffic, &acc_bytes, budget, num_definitions);
    if clusters.len() < 2
        || clusters.len() > settings.traffic_cluster_max_launches
        || clusters.len() <= baseline.len()
    {
        return Ok(None);
    }
    Ok(Some(cluster_kernel_parts(
        &mut clusters,
        &atoms,
        profile.outputs.len(),
    )))
}

/// Pairwise agglomeration by shared bytes under the gathered budget: merge
/// the pair sharing the most bytes — a split accumulator costs its slice
/// twice (the extra launch re-reads and re-writes it) where a duplicated
/// load re-reads once; +1 merges share-free pairs last — while the union's
/// gathered bytes fit. Bottom-up pairing locks onto the small seams;
/// sequential growth packs the first cluster to the budget and mixes them.
pub(super) fn agglomerate_traffic(
    mut clusters: Vec<ValueCluster>,
    traffic: &InputTraffic,
    acc_bytes: &dyn Fn(&(usize, usize)) -> u64,
    budget: u64,
    num_definitions: usize,
) -> Vec<ValueCluster> {
    loop {
        let mut best: Option<(u64, usize, usize)> = None;
        for b in 1..clusters.len() {
            for a in 0..b {
                let union_gathered: u64 = clusters[a]
                    .input_loads
                    .union(&clusters[b].input_loads)
                    .map(|&load| load - num_definitions)
                    .filter(|&index| traffic.gathered[index])
                    .map(|index| traffic.bytes[index])
                    .sum();
                if union_gathered > budget {
                    continue;
                }
                let shared_loads: u64 = clusters[a]
                    .input_loads
                    .intersection(&clusters[b].input_loads)
                    .map(|&load| traffic.bytes[load - num_definitions])
                    .sum();
                let shared_accs: u64 = clusters[a]
                    .accs
                    .intersection(&clusters[b].accs)
                    .map(acc_bytes)
                    .sum();
                let gain = shared_loads + 2 * shared_accs + 1;
                if best.is_none_or(|(g, _, _)| gain > g) {
                    best = Some((gain, a, b));
                }
            }
        }
        let Some((_, a, b)) = best else {
            break;
        };
        let other = clusters.swap_remove(b);
        clusters[a].absorb(other);
    }
    clusters
}

pub(in super::super) fn partition_cost(
    profile: &IncidenceProfile,
    parts: &[KernelPart],
    settings: &Settings,
) -> Result<PartitionCost, String> {
    let f = profile.function();
    let mut uses = HashMap::<ValueId, usize>::new();
    let mut input_uses = HashMap::<ValueId, usize>::new();
    let mut max_closure_statements = 0;
    let mut max_weighted_footprint = 0;
    for part in parts {
        let (_, roots) = selected_term_values(f, profile.ops(), &profile.outputs, part)?;
        let mut free = BTreeSet::new();
        for root in &roots {
            root.add_reads(f, &mut free);
        }
        let root_reads = free.iter().copied().collect::<Vec<_>>();
        let sliced = slice_bindings(f, profile.ops(), &root_reads)?;
        max_closure_statements = max_closure_statements.max(sliced.len());
        let mut defined = BTreeSet::new();
        for &op in &sliced {
            if !matches!(f.op(op).kind, OpKind::Bind) {
                return Err("loom partition cost expects flat bindings".into());
            }
            f.add_reads(f.op(op).operands[0], &mut free);
            defined.insert(f.op(op).results[0]);
            *uses.entry(f.op(op).results[0]).or_default() += 1;
        }
        let mut part_inputs = 0;
        for &value in free.difference(&defined) {
            part_inputs += 1;
            *input_uses.entry(value).or_default() += 1;
        }
        max_weighted_footprint =
            max_weighted_footprint.max(sliced.len() + settings.seam_load_weight * part_inputs);
    }
    Ok(PartitionCost {
        max_closure_statements,
        max_weighted_footprint,
        duplicated_closure_statements: uses.values().map(|uses| uses - 1).sum(),
        duplicated_input_loads: input_uses.values().map(|uses| uses - 1).sum(),
        distinct_input_loads: input_uses.len(),
    })
}
