// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::CompiledProgram;
use recipe::algebra::Computation;
use recipe::{CompilationCostModel, CompileRequests, Executable, RecipeBody, Work};
use std::collections::{BTreeMap, BTreeSet};

/// Source-level request identities, their native compile units, and the source
/// candidates that witness every launched entry of each request. Selecting a
/// guarded request must not shrink the entry set on which its bounds depend.
pub(super) struct Compilations {
    requests: CompileRequests,
    candidate_requests: Vec<BTreeSet<usize>>,
    witnesses: Vec<BTreeMap<String, usize>>,
    candidate_entries: Vec<BTreeSet<(usize, String)>>,
    units: Vec<usize>,
    costs: Vec<f64>,
}

impl Compilations {
    pub(super) fn new(candidates: &[CompiledProgram]) -> Self {
        let mut requests = CompileRequests::default();
        let mut candidate_requests = Vec::new();
        let mut witnesses = Vec::new();
        let mut candidate_entries = Vec::new();
        for (index, candidate) in candidates.iter().enumerate() {
            let remap =
                Executable::merge_root_context(&mut requests, candidate.recipe.context().clone());
            witnesses.resize_with(requests.len(), BTreeMap::new);
            let mut used = BTreeSet::new();
            let mut entries = BTreeSet::new();
            collect_request_uses(
                candidate.recipe.body(),
                remap.as_slice(),
                index,
                &mut used,
                &mut witnesses,
                &mut entries,
            );
            candidate_entries.push(entries);
            candidate_requests.push(used);
        }
        let mut native_units = CompileRequests::default();
        let units = requests
            .as_slice()
            .iter()
            .map(|request| {
                let mut unit = request.clone();
                unit.requires.clear();
                native_units.intern(unit).0
            })
            .collect();
        let model = CompilationCostModel::default();
        let costs = native_units
            .as_slice()
            .iter()
            .map(|request| model.request_cost_ms(request))
            .collect();
        Self {
            requests,
            candidate_requests,
            witnesses,
            candidate_entries,
            units,
            costs,
        }
    }

    /// Close over guards and original entry sets, using the first source
    /// witness for each entry. Already selected candidates need no re-expansion.
    pub(super) fn closure(
        &self,
        candidate: usize,
        selected: &[bool],
        covered: &BTreeSet<(usize, String)>,
    ) -> (Vec<usize>, BTreeSet<usize>) {
        let mut pending = vec![candidate];
        let mut candidates = BTreeSet::new();
        let mut requests = BTreeSet::new();
        let mut entries = BTreeSet::new();
        while let Some(candidate) = pending.pop() {
            if selected[candidate] || !candidates.insert(candidate) {
                continue;
            }
            entries.extend(self.candidate_entries[candidate].iter().cloned());
            let mut required = self.candidate_requests[candidate]
                .iter()
                .copied()
                .collect::<Vec<_>>();
            while let Some(request) = required.pop() {
                if !requests.insert(request) {
                    continue;
                }
                required.extend(
                    self.requests.as_slice()[request]
                        .requires
                        .keys()
                        .map(|id| id.0),
                );
                for (entry, &witness) in &self.witnesses[request] {
                    let key = (request, entry.clone());
                    if !covered.contains(&key) && !entries.contains(&key) {
                        pending.push(witness);
                    }
                }
            }
        }
        let units = requests
            .into_iter()
            .map(|request| self.units[request])
            .collect();
        // The requested candidate leads its support, so Kiln tries the ranked
        // candidate first without turning an entry witness into a priority.
        let mut ordered = vec![candidate];
        ordered.extend(candidates.into_iter().filter(|&index| index != candidate));
        (ordered, units)
    }

    pub(super) fn cover_entries(&self, candidate: usize, covered: &mut BTreeSet<(usize, String)>) {
        covered.extend(self.candidate_entries[candidate].iter().cloned());
    }

    pub(super) fn additional_cost_ms(
        &self,
        units: &BTreeSet<usize>,
        paid: &BTreeSet<usize>,
    ) -> f64 {
        units.difference(paid).map(|&unit| self.costs[unit]).sum()
    }
}

fn collect_request_uses(
    body: &RecipeBody,
    remap: &[usize],
    candidate: usize,
    used: &mut BTreeSet<usize>,
    witnesses: &mut [BTreeMap<String, usize>],
    entries: &mut BTreeSet<(usize, String)>,
) {
    for module in body.context() {
        if let Some(request) = module.code.compile_request() {
            used.insert(remap[request.0]);
        }
    }
    for node in body.nodes() {
        match node.work() {
            Work::Call { call, .. } => {
                if let Some(request) = body.context()[call.module].code.compile_request() {
                    entries.insert((remap[request.0], call.entry.clone()));
                    witnesses[remap[request.0]]
                        .entry(call.entry.clone())
                        .or_insert(candidate);
                }
            }
            Work::Choice { candidates, .. } => {
                for child in candidates {
                    collect_request_uses(child, remap, candidate, used, witnesses, entries);
                }
            }
            _ => {}
        }
    }
}
