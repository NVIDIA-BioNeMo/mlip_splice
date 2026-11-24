// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use gpu_ir::{BinOp, Body, Function, OpId, OpKind, Param, ParamTy, Scalar, Ty, ValTy, ValueId};
use xir::dtype::DType;

use super::super::LiftedProgram;
use super::super::eval::{
    Evaluation, Evaluator, GuardKey, GuardProvenance, ReductionKind, math_scalar, num_elements,
};

mod affinity;
mod bindings;
mod contraction;
mod order;
mod selection;
mod shared_work;
mod terms;
mod traffic_clusters;
mod value_clusters;

#[cfg(test)]
mod tests;

pub(super) use bindings::{slice_bindings, statement_count};
pub(super) use contraction::contract_fma_bindings;
pub use order::ElementOrder;
pub(super) use order::program_order_summaries;
pub(super) use selection::{group_bindings_by_element, selected_bindings, selected_term_values};
pub(super) use shared_work::plan_shared_work;
pub(super) use traffic_clusters::{partition_cost, plan_traffic_clusters};
pub(super) use value_clusters::plan_value_clusters;

use affinity::{AffinityCluster, agglomerate};
pub(super) use bindings::Bindings;
use shared_work::TermAtom;
pub(super) use terms::Term;
use terms::{additive_term_guards, additive_term_spans, additive_terms, cover_terms};

#[derive(Clone, Copy, Debug)]
pub(super) struct Settings {
    pub max_kernel_statements: usize,
    pub kernel_overhead: usize,
    pub output_overhead: usize,
    pub max_shared_work_terms: usize,
    pub max_shared_work_closure_entries: usize,
    pub max_value_cluster_terms: usize,
    pub value_cluster_cap: usize,
    pub register_cap_statement_floor: usize,
    pub seam_load_weight: usize,
    pub affinity_merge_base: usize,
    pub shared_load_weight: usize,
    pub max_affinity_tasks: usize,
    pub value_grow_word_budget: u64,
    pub traffic_cluster_min_gathered_bytes: u64,
    pub traffic_cluster_max_seeds: usize,
    pub traffic_cluster_max_launches: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_kernel_statements: DEFAULT_MAX_KERNEL_STATEMENTS,
            kernel_overhead: 64,
            output_overhead: 8,
            max_shared_work_terms: 4_096,
            max_shared_work_closure_entries: 1 << 24,
            max_value_cluster_terms: 65_536,
            value_cluster_cap: 192,
            register_cap_statement_floor: 512,
            seam_load_weight: 4,
            affinity_merge_base: 16,
            shared_load_weight: 4,
            max_affinity_tasks: 512,
            value_grow_word_budget: 1 << 30,
            traffic_cluster_min_gathered_bytes: 24 << 20,
            traffic_cluster_max_seeds: 160,
            traffic_cluster_max_launches: 6,
        }
    }
}

pub(super) const DEFAULT_MAX_KERNEL_STATEMENTS: usize = 16_384;

/// Closure entries below `num_definitions` are definition statements; the
/// rest are interned input loads.
fn seam_weight(entry: usize, num_definitions: usize, settings: &Settings) -> usize {
    if entry < num_definitions {
        1
    } else {
        settings.seam_load_weight
    }
}

pub(super) type SelectedOutputs = Vec<Vec<Option<gpu_ir::Value>>>;
pub(super) type SelectedTerms = Vec<Vec<Option<Vec<Term>>>>;

pub(super) fn evaluation_profile(
    loom: &LiftedProgram,
) -> Result<(Vec<Vec<ValueId>>, Body), String> {
    let evaluation = evaluation_profile_with_provenance(loom)?;
    Ok((evaluation.outputs, evaluation.body))
}

/// The program evaluated once with its inputs standing for themselves.
///
/// The inputs are parameters of the evaluation's own arena: a profile is a body
/// like any other, and what it reads has to be a value it holds rather than a
/// name resolved later.
pub(super) fn evaluation_profile_with_provenance(
    loom: &LiftedProgram,
) -> Result<Evaluation, String> {
    let func = loom.module.entry_func();
    let mut evaluator = Evaluator::new(&loom.module, 1);
    let mut params = Vec::with_capacity(func.params.len());
    for (input, &param) in func.params.iter().enumerate() {
        let shape = func.value_shape(param);
        let scalar = math_scalar(shape.dtype)?;
        let count = num_elements(&shape.dims)?;
        params.push(
            (0..count)
                .map(|element| {
                    evaluator.builder().parameter(Param {
                        name: format!("loom_profile_input_{input}_{element}"),
                        ty: ParamTy::Scalar(scalar),
                    })
                })
                .collect::<Vec<_>>(),
        );
    }
    let indexed = loom
        .has_indexed_ops()
        .then(|| {
            func.params
                .iter()
                .enumerate()
                .map(|(input, &param)| {
                    let shape = func.value_shape(param);
                    let scalar = math_scalar(shape.dtype)?;
                    let base = evaluator.builder().parameter(Param {
                        name: format!("loom_profile_input_base_{input}"),
                        ty: ParamTy::Ptr(scalar),
                    });
                    Ok(super::super::eval::IndexedTensor::pointer(
                        base,
                        scalar,
                        1,
                        1,
                        1,
                        shape.dtype == DType::Bool,
                        None,
                        gpu_ir::Evict::Normal,
                        num_elements(&shape.dims)?,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .transpose()?;
    evaluator.bind_params(params, indexed)?;
    evaluator.run_with_provenance()
}

#[derive(Clone, Debug)]
pub(super) struct ElementPart {
    pub element: usize,
    pub terms: Vec<usize>,
    pub continuation: bool,
    /// A later launch of the same candidate reads this element back
    /// (a continuation load); its store must stay cache-resident.
    /// Marked by [`super::schedule::Candidate`] construction.
    pub read_later: bool,
    /// Which kind of reduction this element is, where it is one. The values a
    /// reduction is made of belong to whichever evaluation found them, so a
    /// part carries only what it can carry between plans: the kind. A consumer
    /// asks its own profile for the rest.
    pub reduction: Option<ReductionKind>,
    /// The guard this element runs under, and the one each of its terms runs
    /// under — named in the program rather than in an arena, for the same
    /// reason.
    pub guard: Option<GuardKey>,
    pub term_guards: Vec<Option<GuardKey>>,
}

#[derive(Clone, Debug)]
pub(super) struct KernelPart {
    pub elements: Vec<Vec<ElementPart>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PartitionCost {
    pub max_closure_statements: usize,
    /// Max over launches of closure statements plus the configured weight of
    /// the launch's distinct input loads — the occupancy proxy.
    pub max_weighted_footprint: usize,
    pub duplicated_closure_statements: usize,
    pub duplicated_input_loads: usize,
    pub distinct_input_loads: usize,
}

impl PartitionCost {
    /// Ordering for choosing among alternative candidates: the weighted max
    /// per-launch footprint first, total seam-weighted duplication as the
    /// tie-break.
    pub(super) fn selection_key(&self, settings: &Settings) -> (usize, usize) {
        (
            self.max_weighted_footprint,
            settings.seam_load_weight * self.duplicated_input_loads
                + self.duplicated_closure_statements,
        )
    }
}

/// The symbolic profile's output values and its ordered `Bind` operations, from
/// which a part derives each output element's dependency closure for
/// closure-aware element grouping. The closure entries below the binding count
/// are those bindings; what the closure reaches and no binding defines interns
/// after them in the same space. Prepared once per compile and shared by element
/// grouping, value clustering, and alternative ranking.
pub(super) struct IncidenceProfile {
    outputs: Vec<Vec<ValueId>>,
    body: Body,
    ops: Vec<OpId>,
    bindings: Bindings,
    guards: Vec<Vec<Option<GuardProvenance>>>,
    statement_guards: BTreeMap<ValueId, GuardProvenance>,
    /// Which operand each per-element input value of this arena belongs to.
    ///
    /// An indexed program also reaches its inputs through a base pointer, which
    /// is an input of the closure without being an element of anything, so it is
    /// deliberately absent rather than mapped to its operand.
    input_operands: BTreeMap<ValueId, usize>,
}

impl IncidenceProfile {
    pub(super) fn prepare(loom: &LiftedProgram) -> Result<Self, String> {
        let evaluation = evaluation_profile_with_provenance(loom)?;
        let Evaluation {
            inputs,
            outputs,
            body,
            reductions: _,
            guards,
            statement_guards,
        } = evaluation;
        let ops = body.steps().to_vec();
        let mut bindings = Bindings::of(body.function(), &ops)?;
        bindings.intern_outputs(body.function(), &outputs);
        let input_operands = inputs
            .iter()
            .enumerate()
            .flat_map(|(operand, elements)| elements.iter().map(move |&value| (value, operand)))
            .collect();
        Ok(Self {
            outputs,
            body,
            ops,
            bindings,
            guards,
            statement_guards,
            input_operands,
        })
    }

    pub(super) fn function(&self) -> &Function {
        self.body.function()
    }

    pub(super) fn ops(&self) -> &[OpId] {
        &self.ops
    }

    /// The operand `value` is an element of, for a value this profile reads and
    /// does not define.
    pub(super) fn input_operand(&self, value: ValueId) -> Option<usize> {
        self.input_operands.get(&value).copied()
    }

    /// Live-value footprint of the fused program: every input load plus every
    /// output-element accumulator.
    pub(super) fn fused_footprint(&self) -> usize {
        self.bindings.inputs() + self.outputs.iter().map(Vec::len).sum::<usize>()
    }
}

impl KernelPart {
    pub(super) fn contains(&self, output: usize, element: usize) -> bool {
        self.elements[output]
            .iter()
            .any(|selection| selection.element == element)
    }

    pub(super) fn writes_output(&self, output: usize) -> bool {
        !self.elements[output].is_empty()
    }

    pub(super) fn output_continues(&self, output: usize) -> bool {
        self.elements[output]
            .iter()
            .any(|selection| selection.continuation)
    }

    pub(super) fn len(&self) -> usize {
        self.elements
            .iter()
            .flatten()
            .map(|selection| selection.terms.len())
            .sum()
    }

    /// The part restricted to one group of outputs, on the same iteration
    /// domain; `None` when the group contributes nothing to this part.
    pub(super) fn restrict_to_outputs(&self, outputs: &BTreeSet<usize>) -> Option<Self> {
        let elements = self
            .elements
            .iter()
            .enumerate()
            .map(|(output, selections)| {
                if outputs.contains(&output) {
                    selections.clone()
                } else {
                    Vec::new()
                }
            })
            .collect::<Vec<_>>();
        let part = Self { elements };
        (part.len() > 0).then_some(part)
    }

    /// The part split into `shards` whole-element groups along the program's
    /// incidence seams: each element group is placed where it duplicates the
    /// fewest closure entries — shared input loads weighing
    /// the configured seam weight over shared arithmetic — under a balance cap, so
    /// elements sharing operand loads or arithmetic stay together and
    /// near-disjoint closures split cleanly. Every selection of one element
    /// stays in one group — sibling groups never write the same address.
    /// `None` when the part has fewer distinct output elements than `shards`,
    /// when the selected terms cannot be re-derived from the profile, or when
    /// the largest group's weighted closure stays within 10% of the fused
    /// kernel — a split that narrows nothing.
    pub(super) fn shard(
        &self,
        shards: usize,
        profile: &IncidenceProfile,
        settings: &Settings,
    ) -> Option<Vec<Self>> {
        let mut groups: Vec<((usize, usize), Vec<&ElementPart>)> = Vec::new();
        for (output, selections) in self.elements.iter().enumerate() {
            for selection in selections {
                let key = (output, selection.element);
                match groups.iter_mut().find(|(existing, _)| *existing == key) {
                    Some((_, group)) => group.push(selection),
                    None => groups.push((key, vec![selection])),
                }
            }
        }
        if groups.len() < shards {
            return None;
        }
        let closures = groups
            .iter()
            .map(|((output, element), group)| {
                let f = profile.function();
                let terms = additive_terms(
                    f,
                    *profile.outputs.get(*output)?.get(*element)?,
                    &profile.bindings,
                );
                let mut closure = BTreeSet::new();
                let mut inserted = Vec::new();
                for selection in group {
                    for &term in &selection.terms {
                        for read in terms.get(term)?.reads(f) {
                            profile.bindings.add_dependencies_with_inputs(
                                f,
                                read,
                                &mut closure,
                                &mut inserted,
                            );
                        }
                    }
                }
                Some(closure)
            })
            .collect::<Option<Vec<_>>>()?;
        let num_definitions = profile.bindings.definitions();
        let weight_of = |closure: &BTreeSet<usize>| {
            closure
                .iter()
                .map(|&entry| seam_weight(entry, num_definitions, settings))
                .sum::<usize>()
        };
        let terms_of = |group: &[&ElementPart]| {
            group
                .iter()
                .map(|selection| selection.terms.len())
                .sum::<usize>()
                .max(1)
        };
        let total_terms = groups
            .iter()
            .map(|(_, group)| terms_of(group))
            .sum::<usize>();
        let fused =
            weight_of(&closures.iter().flatten().copied().collect::<BTreeSet<_>>()) + total_terms;
        let weights = closures.iter().map(weight_of).collect::<Vec<_>>();
        let mut order = (0..groups.len()).collect::<Vec<_>>();
        order.sort_by_key(|&group| {
            (
                Reverse(weights[group] + terms_of(&groups[group].1)),
                groups[group].0,
            )
        });
        let largest = weights[order[0]] + terms_of(&groups[order[0]].1);
        let cap = (fused * 5).div_ceil(shards * 4).max(largest);
        let mut shard_closures: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); shards];
        let mut loads = vec![0usize; shards];
        let mut counts = vec![0usize; shards];
        let mut parts = vec![vec![Vec::new(); self.elements.len()]; shards];
        for (placed, &group_index) in order.iter().enumerate() {
            let ((output, _), group) = &groups[group_index];
            let closure = &closures[group_index];
            let terms = terms_of(group);
            let added_to = |shard_closure: &BTreeSet<usize>| {
                closure
                    .difference(shard_closure)
                    .map(|&entry| seam_weight(entry, num_definitions, settings))
                    .sum::<usize>()
                    + terms
            };
            let empty = counts.iter().filter(|&&count| count == 0).count();
            let must_seed = groups.len() - placed <= empty;
            let choice = (0..shards)
                .filter(|&shard| !must_seed || counts[shard] == 0)
                .map(|shard| {
                    let added = added_to(&shard_closures[shard]);
                    let resulting = loads[shard] + added;
                    if resulting <= cap {
                        (false, added, resulting, shard)
                    } else {
                        (true, resulting, added, shard)
                    }
                })
                .min()
                .expect("at least one shard")
                .3;
            loads[choice] += added_to(&shard_closures[choice]);
            shard_closures[choice].extend(closure.iter().copied());
            counts[choice] += 1;
            parts[choice][*output].extend(group.iter().map(|&selection| selection.clone()));
        }
        if loads.iter().copied().max().unwrap_or(0) * 10 >= fused * 9 {
            return None;
        }
        Some(
            parts
                .into_iter()
                .map(|elements| Self { elements })
                .collect(),
        )
    }

    pub(super) fn split(&self) -> (Self, Self) {
        let pivot = self.len() / 2;
        let mut left = vec![Vec::new(); self.elements.len()];
        let mut right = vec![Vec::new(); self.elements.len()];
        let mut position = 0;
        for (output, selections) in self.elements.iter().enumerate() {
            for selection in selections {
                let mut left_terms = Vec::new();
                let mut right_terms = Vec::new();
                for &term in &selection.terms {
                    if position < pivot {
                        left_terms.push(term);
                    } else {
                        right_terms.push(term);
                    }
                    position += 1;
                }
                let split_element = !left_terms.is_empty() && !right_terms.is_empty();
                let left_len = left_terms.len();
                if !left_terms.is_empty() {
                    left[output].push(ElementPart {
                        element: selection.element,
                        terms: left_terms,
                        continuation: selection.continuation,
                        read_later: selection.read_later,
                        reduction: selection.reduction,
                        guard: selection.guard,
                        term_guards: selection.term_guards[..left_len].to_vec(),
                    });
                }
                if !right_terms.is_empty() {
                    right[output].push(ElementPart {
                        element: selection.element,
                        terms: right_terms,
                        continuation: selection.continuation || split_element,
                        read_later: selection.read_later,
                        reduction: selection.reduction,
                        guard: selection.guard,
                        term_guards: selection.term_guards[left_len..].to_vec(),
                    });
                }
            }
        }
        (Self { elements: left }, Self { elements: right })
    }
}

struct Definition {
    dependencies: Vec<usize>,
    /// What this binding was bound to.
    init: ValueId,
}

#[cfg(test)]
pub(super) fn plan(loom: &LiftedProgram, cap: usize) -> Result<Vec<KernelPart>, String> {
    plan_with_settings(loom, cap, &Settings::default())
}

/// What each binding of the evaluation profile reads, in binding order.
///
/// This is the closure index space [`Bindings`] works in: an entry below the
/// returned length is another binding, and one at or above it is an input load.
pub(super) fn definition_dependencies(loom: &LiftedProgram) -> Result<Vec<Vec<usize>>, String> {
    let evaluation = evaluation_profile_with_provenance(loom)?;
    let ops = evaluation.body.steps().to_vec();
    Ok(Bindings::of(evaluation.body.function(), &ops)?.dependency_sets())
}

pub(super) fn plan_with_settings(
    loom: &LiftedProgram,
    cap: usize,
    settings: &Settings,
) -> Result<Vec<KernelPart>, String> {
    if cap <= settings.kernel_overhead + settings.output_overhead {
        return Err(format!("loom max_kernel_statements {cap} is too small"));
    }
    let Evaluation {
        outputs,
        body,
        reductions,
        guards,
        statement_guards,
        ..
    } = evaluation_profile_with_provenance(loom)?;
    let f = body.function();
    let ops = body.steps().to_vec();
    let bindings = Bindings::of(f, &ops)?;

    let mut parts = Vec::new();
    let mut current = KernelPart {
        elements: vec![Vec::new(); outputs.len()],
    };
    let mut required = BTreeSet::new();
    let mut work = settings.kernel_overhead;
    let mut written = BTreeSet::new();
    for (output, values) in outputs.iter().enumerate() {
        for (element, value) in values.iter().enumerate() {
            let reduction = reductions[output][element];
            let (terms, spans) = additive_term_spans(f, *value, &bindings);
            let term_guards = additive_term_guards(f, &terms, &spans, &statement_guards);
            for (term_index, term) in terms.iter().enumerate() {
                let new_element = current.elements[output]
                    .last()
                    .is_none_or(|selection| selection.element != element);
                let mut inserted = Vec::new();
                for read in term.reads(f) {
                    bindings.add_dependencies(f, read, &mut required, &mut inserted);
                }
                let added =
                    inserted.len() + 1 + usize::from(new_element) * settings.output_overhead;
                if current.len() > 0 && work + added > cap {
                    for index in inserted {
                        required.remove(&index);
                    }
                    for (written_output, selections) in current.elements.iter().enumerate() {
                        for selection in selections {
                            written.insert((written_output, selection.element));
                        }
                    }
                    parts.push(current);
                    current = KernelPart {
                        elements: vec![Vec::new(); outputs.len()],
                    };
                    required.clear();
                    work = settings.kernel_overhead;
                    let mut first = Vec::new();
                    for read in term.reads(f) {
                        bindings.add_dependencies(f, read, &mut required, &mut first);
                    }
                    let term_work = first.len() + 1 + settings.output_overhead;
                    if work + term_work > cap {
                        return Err(format!(
                            "loom output {output} element {element} additive term {term_index} \
                             needs {} statements, exceeding kernel cap {cap}",
                            work + term_work,
                        ));
                    }
                    work += term_work;
                } else {
                    work += added;
                }
                let selections = &mut current.elements[output];
                let term_guard = term_guards[term_index];
                if let Some(selection) = selections
                    .last_mut()
                    .filter(|selection| selection.element == element)
                {
                    selection.terms.push(term_index);
                    selection.term_guards.push(term_guard);
                } else {
                    selections.push(ElementPart {
                        element,
                        terms: vec![term_index],
                        continuation: written.contains(&(output, element)),
                        read_later: false,
                        reduction,
                        guard: guards[output][element].as_ref().map(|guard| guard.key),
                        term_guards: vec![term_guard],
                    });
                }
            }
        }
    }
    if current.len() > 0 {
        parts.push(current);
    }
    if parts.is_empty() {
        return Err("loom entry function has no output elements".into());
    }
    Ok(parts)
}
