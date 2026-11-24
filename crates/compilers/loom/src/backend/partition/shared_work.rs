// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::value_clusters::term_atoms;
use super::{
    BTreeSet, ElementPart, GuardKey, IncidenceProfile, KernelPart, Reverse, Settings,
    partition_cost, seam_weight,
};

pub(super) struct TermAtom {
    pub(super) output: usize,
    pub(super) element: usize,
    pub(super) term: usize,
    pub(super) closure: Vec<usize>,
    pub(super) guard: Option<GuardKey>,
    pub(super) term_guard: Option<GuardKey>,
}

impl KernelPart {
    pub(super) fn push_atom(&mut self, atom: &TermAtom, continuation: bool) {
        if let Some(selection) = self.elements[atom.output]
            .iter_mut()
            .find(|selection| selection.element == atom.element)
        {
            selection.terms.push(atom.term);
            selection.term_guards.push(atom.term_guard);
        } else {
            self.elements[atom.output].push(ElementPart {
                element: atom.element,
                terms: vec![atom.term],
                continuation,
                read_later: false,
                reduction: None,
                guard: atom.guard,
                term_guards: vec![atom.term_guard],
            });
        }
    }
}

pub(in super::super) fn plan_shared_work(
    profile: &IncidenceProfile,
    cap: usize,
    baseline: &[KernelPart],
    settings: &Settings,
) -> Result<Option<Vec<KernelPart>>, String> {
    if cap <= settings.kernel_overhead + settings.output_overhead {
        return Ok(None);
    }
    if baseline.len() < 2 {
        return Ok(None);
    }

    let num_definitions = profile.bindings.definitions();
    // Grouping prices a closure by its definition statements alone: `work` and
    // the cap stay on statements, so which groupings fit — and thus which
    // candidates exist — is unchanged by the interned input loads.
    let statements_in = |closure: &BTreeSet<usize>| {
        closure
            .iter()
            .filter(|&&entry| entry < num_definitions)
            .count()
    };
    let Some(atoms) = term_atoms(
        profile,
        settings.max_shared_work_terms,
        settings.max_shared_work_closure_entries,
        statements_in,
    ) else {
        return Ok(None);
    };
    let mut atom_statements = Vec::with_capacity(atoms.len());
    let mut atom_weights = Vec::with_capacity(atoms.len());
    for atom in &atoms {
        let statements = atom
            .closure
            .iter()
            .filter(|&&entry| entry < num_definitions)
            .count();
        if settings.kernel_overhead + settings.output_overhead + statements + 1 > cap {
            return Ok(None);
        }
        atom_statements.push(statements);
        atom_weights
            .push(statements + settings.seam_load_weight * (atom.closure.len() - statements));
    }
    let outputs = &profile.outputs;

    let mut dependents = vec![Vec::new(); num_definitions + profile.bindings.inputs()];
    for (atom, value) in atoms.iter().enumerate() {
        for &dependency in &value.closure {
            dependents[dependency].push(atom);
        }
    }

    let mut grouped = Vec::new();
    let mut assigned = vec![false; atoms.len()];
    let mut remaining = atoms.len();
    let mut written = BTreeSet::new();
    while remaining > 0 {
        let seed = (0..atoms.len())
            .filter(|&atom| !assigned[atom])
            .max_by_key(|&atom| (atom_weights[atom], Reverse(atom)))
            .unwrap();
        let mut part = KernelPart {
            elements: vec![Vec::new(); outputs.len()],
        };
        let mut required = BTreeSet::new();
        // `overlap` counts shared definition statements and feeds `work`;
        // `affinity` adds shared input loads at the configured seam weight and only
        // orders the packing.
        let mut overlap = vec![0usize; atoms.len()];
        let mut affinity = vec![0usize; atoms.len()];
        let mut selected_elements = BTreeSet::new();
        let mut work = settings.kernel_overhead;
        let mut next = Some(seed);
        while let Some(atom_index) = next {
            let atom = &atoms[atom_index];
            let new_element = selected_elements.insert((atom.output, atom.element));
            work += atom_statements[atom_index] - overlap[atom_index]
                + 1
                + usize::from(new_element) * settings.output_overhead;
            assigned[atom_index] = true;
            remaining -= 1;
            for &dependency in &atom.closure {
                if required.insert(dependency) {
                    let weight = seam_weight(dependency, num_definitions, settings);
                    for &dependent in &dependents[dependency] {
                        affinity[dependent] += weight;
                        if dependency < num_definitions {
                            overlap[dependent] += 1;
                        }
                    }
                }
            }
            part.push_atom(atom, written.contains(&(atom.output, atom.element)));

            next = (0..atoms.len())
                .filter(|&candidate| !assigned[candidate])
                .filter_map(|candidate| {
                    let atom = &atoms[candidate];
                    let new_element = !selected_elements.contains(&(atom.output, atom.element));
                    let added = atom_statements[candidate] - overlap[candidate]
                        + 1
                        + usize::from(new_element) * settings.output_overhead;
                    (work + added <= cap).then_some((
                        affinity[candidate],
                        Reverse(added),
                        Reverse(candidate),
                        candidate,
                    ))
                })
                .max()
                .map(|(_, _, _, candidate)| candidate);
        }
        for (output, selections) in part.elements.iter_mut().enumerate() {
            selections.sort_by_key(|selection| selection.element);
            for selection in selections {
                written.insert((output, selection.element));
            }
        }
        grouped.push(part);
    }

    let cost = |parts| partition_cost(profile, parts, settings);
    let baseline_cost = cost(baseline)?;
    let grouped_cost = cost(&grouped)?;
    Ok(
        ((grouped_cost.duplicated_closure_statements
            < baseline_cost.duplicated_closure_statements)
            || (grouped_cost.duplicated_closure_statements
                == baseline_cost.duplicated_closure_statements
                && grouped_cost.max_closure_statements < baseline_cost.max_closure_statements))
            .then_some(grouped),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atoms_preserve_term_order_and_element_continuations() {
        let mut part = KernelPart {
            elements: vec![Vec::new(); 2],
        };
        let atom = |output, element, term| TermAtom {
            output,
            element,
            term,
            closure: Vec::new(),
            guard: None,
            term_guard: None,
        };
        part.push_atom(&atom(0, 3, 5), false);
        part.push_atom(&atom(0, 3, 2), true);
        part.push_atom(&atom(0, 1, 4), true);
        part.push_atom(&atom(1, 3, 6), false);

        assert_eq!(
            part.elements.iter().map(Vec::len).collect::<Vec<_>>(),
            [2, 1]
        );
        let first = &part.elements[0][0];
        assert_eq!(first.element, 3);
        assert_eq!(first.terms, [5, 2]);
        assert_eq!(first.term_guards, [None, None]);
        assert!(!first.continuation);
        assert!(!first.read_later);
        assert!(first.reduction.is_none());
        assert!(first.guard.is_none());
        let continuation = &part.elements[0][1];
        assert_eq!(continuation.element, 1);
        assert_eq!(continuation.terms, [4]);
        assert!(continuation.continuation);
        let other_output = &part.elements[1][0];
        assert_eq!(other_output.element, 3);
        assert_eq!(other_output.terms, [6]);
        assert!(!other_output.continuation);
    }
}
