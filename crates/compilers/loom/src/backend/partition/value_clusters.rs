// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    AffinityCluster, BTreeSet, HashMap, IncidenceProfile, KernelPart, Settings, TermAtom,
    additive_term_guards, additive_term_spans, agglomerate, selected_term_values, slice_bindings,
};

#[derive(Clone)]
pub(super) struct ValueCluster {
    pub(super) atoms: Vec<usize>,
    pub(super) input_loads: BTreeSet<usize>,
    pub(super) interior: BTreeSet<usize>,
    pub(super) accs: BTreeSet<(usize, usize)>,
}

impl ValueCluster {
    pub(super) fn footprint(&self) -> usize {
        self.input_loads.len() + self.accs.len()
    }
}

/// Word-packed load and accumulator sets over the dense index space the
/// tasks actually touch: the growth loop is quadratic in tasks, so every
/// candidate evaluation must be a few word operations, not tree walks.
struct GrowMask {
    loads: Vec<u64>,
    accs: Vec<u64>,
}

impl GrowMask {
    fn footprint_with(&self, other: &GrowMask) -> usize {
        let union = |a: &[u64], b: &[u64]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x | y).count_ones() as usize)
                .sum::<usize>()
        };
        union(&self.loads, &other.loads) + union(&self.accs, &other.accs)
    }

    fn shared_loads(&self, other: &GrowMask) -> usize {
        self.loads
            .iter()
            .zip(&other.loads)
            .map(|(x, y)| (x & y).count_ones() as usize)
            .sum()
    }

    fn shared_accs(&self, other: &GrowMask) -> usize {
        self.accs
            .iter()
            .zip(&other.accs)
            .map(|(x, y)| (x & y).count_ones() as usize)
            .sum()
    }

    fn footprint(&self) -> usize {
        self.loads
            .iter()
            .chain(&self.accs)
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    fn extend(&mut self, other: &GrowMask) {
        for (word, extra) in self.loads.iter_mut().zip(&other.loads) {
            *word |= extra;
        }
        for (word, extra) in self.accs.iter_mut().zip(&other.accs) {
            *word |= extra;
        }
    }
}

/// Sequential best-fit growth under the live-value cap: seed a cluster with
/// the first unplaced atom, then repeatedly absorb the unplaced atom with
/// the most shared loads per unit of footprint the union adds, until nothing
/// fits; repeat from the next seed. Pairwise agglomeration maximizes local
/// shared work and locks into the per-output structure (whose atoms also
/// share interior arithmetic), re-reading every fiber the outputs share;
/// growing one cluster to saturation packs each fiber's readers together
/// regardless of which output they feed — measured on benzene-bwd, +3%
/// duplicated fibers per family and +22% across the whole mixed-domain
/// program, against +27%/+111% for the agglomeration.
fn grow_value_clusters(
    tasks: Vec<ValueCluster>,
    cap: usize,
    settings: &Settings,
) -> Vec<ValueCluster> {
    let mut load_bits = HashMap::new();
    let mut acc_bits = HashMap::new();
    for task in &tasks {
        for &load in &task.input_loads {
            let next = load_bits.len();
            load_bits.entry(load).or_insert(next);
        }
        for &acc in &task.accs {
            let next = acc_bits.len();
            acc_bits.entry(acc).or_insert(next);
        }
    }
    let load_words = load_bits.len().div_ceil(64);
    let acc_words = acc_bits.len().div_ceil(64);
    let budget =
        (tasks.len() as u64) * (tasks.len() as u64) * ((load_words + acc_words).max(1) as u64);
    if budget > settings.value_grow_word_budget {
        return agglomerate(tasks, cap, settings);
    }
    let masks = tasks
        .iter()
        .map(|task| {
            let mut mask = GrowMask {
                loads: vec![0; load_words],
                accs: vec![0; acc_words],
            };
            for load in &task.input_loads {
                let bit = load_bits[load];
                mask.loads[bit / 64] |= 1 << (bit % 64);
            }
            for acc in &task.accs {
                let bit = acc_bits[acc];
                mask.accs[bit / 64] |= 1 << (bit % 64);
            }
            mask
        })
        .collect::<Vec<_>>();
    let mut placed = vec![false; tasks.len()];
    let mut tasks = tasks.into_iter().map(Some).collect::<Vec<_>>();
    let mut clusters = Vec::new();
    for seed in 0..tasks.len() {
        if placed[seed] {
            continue;
        }
        placed[seed] = true;
        let mut cluster = tasks[seed].take().expect("unplaced task present");
        let mut mask = GrowMask {
            loads: masks[seed].loads.clone(),
            accs: masks[seed].accs.clone(),
        };
        let mut footprint = mask.footprint();
        loop {
            // Priority (shared + 1) / (1 + added footprint), compared by
            // cross-multiplication; ties resolve to the earliest atom. A
            // shared accumulator outweighs a shared load: splitting an
            // element's terms across launches turns one accumulator into
            // several atomic or read-modify-write streams of the whole
            // output fiber, where a duplicated load re-reads one fiber.
            let mut best: Option<(u64, u64, usize, usize)> = None;
            for candidate in seed + 1..tasks.len() {
                if placed[candidate] {
                    continue;
                }
                let union = mask.footprint_with(&masks[candidate]);
                if union > cap {
                    continue;
                }
                let shared = mask.shared_loads(&masks[candidate])
                    + settings.shared_load_weight * mask.shared_accs(&masks[candidate]);
                let num = (shared + 1) as u64;
                let den = (1 + union - footprint) as u64;
                if best.is_none_or(|(bn, bd, _, _)| num * bd > bn * den) {
                    best = Some((num, den, candidate, union));
                }
            }
            let Some((_, _, candidate, union)) = best else {
                break;
            };
            placed[candidate] = true;
            cluster.absorb(tasks[candidate].take().expect("unplaced task present"));
            mask.extend(&masks[candidate]);
            footprint = union;
        }
        clusters.push(cluster);
    }
    clusters
}

impl AffinityCluster for ValueCluster {
    fn fits_with(&self, other: &Self, cap: usize) -> bool {
        self.input_loads.union(&other.input_loads).count() + self.accs.union(&other.accs).count()
            <= cap
    }

    fn merge_gain(&self, other: &Self, settings: &Settings) -> usize {
        settings.affinity_merge_base
            + settings.shared_load_weight
                * self.input_loads.intersection(&other.input_loads).count()
            + self.interior.intersection(&other.interior).count()
    }

    fn absorb(&mut self, other: Self) {
        self.atoms.extend(other.atoms);
        self.input_loads.extend(other.input_loads);
        self.interior.extend(other.interior);
        self.accs.extend(other.accs);
    }
}

/// The fused work recut at additive-term granularity into launches whose
/// live-value footprint — distinct input loads plus `(output, element)`
/// accumulators — stays within `value_cap`, terms clustered by load and
/// arithmetic affinity so duplication is what the partition minimizes
/// subject to the cap. Within-element splits continue their accumulator in a
/// later launch. `None` when a baseline launch is already too small to be
/// register-bound, when the fused footprint cannot double the cap, or when the
/// clusters would not be finer than the baseline.
pub(in super::super) fn plan_value_clusters(
    profile: &IncidenceProfile,
    value_cap: usize,
    baseline: &[KernelPart],
    settings: &Settings,
) -> Result<Option<Vec<KernelPart>>, String> {
    if baseline.is_empty() {
        return Ok(None);
    }
    let f = profile.function();
    for part in baseline {
        let (_, roots) = selected_term_values(f, profile.ops(), &profile.outputs, part)?;
        let mut reads = BTreeSet::new();
        for root in &roots {
            root.add_reads(f, &mut reads);
        }
        let reads = reads.into_iter().collect::<Vec<_>>();
        if slice_bindings(f, profile.ops(), &reads)?.len() <= settings.register_cap_statement_floor
        {
            return Ok(None);
        }
    }
    let parts = value_cluster_parts(profile, value_cap, settings)?;
    Ok(parts.filter(|parts| parts.len() > baseline.len()))
}

pub(super) fn value_cluster_parts(
    profile: &IncidenceProfile,
    value_cap: usize,
    settings: &Settings,
) -> Result<Option<Vec<KernelPart>>, String> {
    // The cap proposal comes from the part: a fused footprint that cannot
    // double the cap predicts fewer than two narrowed clusters, so the split
    // declines before any atom is built.
    if profile.fused_footprint() < 2 * value_cap {
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
    let clusters = coalesce_atoms(&atoms, profile.bindings.definitions());
    if clusters.len() < 2 {
        return Ok(None);
    }
    let fused_footprint = clusters
        .iter()
        .flat_map(|cluster| cluster.input_loads.iter().copied())
        .collect::<BTreeSet<_>>()
        .len()
        + clusters
            .iter()
            .flat_map(|cluster| cluster.accs.iter().copied())
            .collect::<BTreeSet<_>>()
            .len();
    let mut clusters = grow_value_clusters(clusters, value_cap, settings);
    if clusters.len() < 2 {
        return Ok(None);
    }
    // A split that narrows nothing: the widest launch still holds most of
    // the fused footprint.
    let max_footprint = clusters
        .iter()
        .map(ValueCluster::footprint)
        .max()
        .unwrap_or(0);
    if max_footprint * 10 >= fused_footprint * 6 {
        return Ok(None);
    }
    Ok(Some(cluster_kernel_parts(
        &mut clusters,
        &atoms,
        profile.outputs.len(),
    )))
}

/// One atom per (output element, additive term), each with its dependency
/// closure over definitions and interned input loads. `entries` prices one
/// closure against `max_entries` — the two clusterings weigh a closure
/// differently, and the budget is what keeps atom building near-linear.
/// `None` when the program is too large to cluster at term granularity.
pub(super) fn term_atoms(
    profile: &IncidenceProfile,
    max_atoms: usize,
    max_entries: usize,
    entries: impl Fn(&BTreeSet<usize>) -> usize,
) -> Option<Vec<TermAtom>> {
    let mut atoms = Vec::new();
    let mut closure_entries = 0;
    let f = profile.function();
    for (output, values) in profile.outputs.iter().enumerate() {
        for (element, &value) in values.iter().enumerate() {
            let (terms, spans) = additive_term_spans(f, value, &profile.bindings);
            let term_guards = additive_term_guards(f, &terms, &spans, &profile.statement_guards);
            for (term, (held, term_guard)) in terms.iter().zip(term_guards).enumerate() {
                let mut closure = BTreeSet::new();
                let mut inserted = Vec::new();
                for read in held.reads(f) {
                    profile.bindings.add_dependencies_with_inputs(
                        f,
                        read,
                        &mut closure,
                        &mut inserted,
                    );
                }
                closure_entries += entries(&closure);
                if closure_entries > max_entries {
                    return None;
                }
                atoms.push(TermAtom {
                    output,
                    element,
                    term,
                    closure: closure.into_iter().collect(),
                    guard: profile.guards[output][element]
                        .as_ref()
                        .map(|guard| guard.key),
                    term_guard,
                });
                if atoms.len() > max_atoms {
                    return None;
                }
            }
        }
    }
    Some(atoms)
}

/// Atoms feeding the same accumulator through identical load sets can never
/// profitably separate: coalescing them sets the clustering granularity.
pub(super) fn coalesce_atoms(atoms: &[TermAtom], num_definitions: usize) -> Vec<ValueCluster> {
    use std::collections::hash_map::Entry;
    let mut clusters: Vec<ValueCluster> = Vec::new();
    let mut coalesced = HashMap::<_, usize>::new();
    for (index, atom) in atoms.iter().enumerate() {
        let (input_loads, interior): (BTreeSet<usize>, BTreeSet<usize>) = atom
            .closure
            .iter()
            .copied()
            .partition(|&entry| entry >= num_definitions);
        let key = (
            atom.output,
            atom.element,
            input_loads.iter().copied().collect::<Vec<_>>(),
        );
        match coalesced.entry(key) {
            Entry::Occupied(slot) => {
                let cluster = &mut clusters[*slot.get()];
                cluster.atoms.push(index);
                cluster.interior.extend(interior);
            }
            Entry::Vacant(vacancy) => {
                vacancy.insert(clusters.len());
                clusters.push(ValueCluster {
                    atoms: vec![index],
                    input_loads,
                    interior,
                    accs: BTreeSet::from([(atom.output, atom.element)]),
                });
            }
        }
    }
    clusters
}

/// Clusters as kernel parts: within-element splits continue their
/// accumulator in a later launch.
pub(super) fn cluster_kernel_parts(
    clusters: &mut [ValueCluster],
    atoms: &[TermAtom],
    num_outputs: usize,
) -> Vec<KernelPart> {
    for cluster in clusters.iter_mut() {
        cluster.atoms.sort_unstable();
    }
    clusters.sort_by_key(|cluster| cluster.atoms[0]);
    let mut written = BTreeSet::new();
    let mut parts = Vec::with_capacity(clusters.len());
    for cluster in clusters.iter() {
        let mut part = KernelPart {
            elements: vec![Vec::new(); num_outputs],
        };
        for &index in &cluster.atoms {
            let atom = &atoms[index];
            part.push_atom(atom, written.contains(&(atom.output, atom.element)));
        }
        for (output, selections) in part.elements.iter().enumerate() {
            for selection in selections {
                written.insert((output, selection.element));
            }
        }
        parts.push(part);
    }
    parts
}
