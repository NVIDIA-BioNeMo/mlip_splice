// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::facts::LoomFacts;
use super::super::partition::{ElementPart, KernelPart};
use super::super::{iteration, reduction};
use super::{
    Candidate, CandidateOrigin, DomainPlan, GuardedPlan, GuardedRegion, Launch, MaterializedValue,
    ReductionProvenance, ReductionRegion, WriteMode,
};
use crate::LiftedProgram;
use crate::eval::{ReductionKind, output_scalar};
use domain_index::DomainIndex;
use gpu_ir::Scalar;
use std::collections::{BTreeMap, BTreeSet};

pub(in crate::backend) fn output_domain_groups(
    loom: &LiftedProgram,
    domain: &DomainPlan,
) -> Vec<Vec<usize>> {
    let default = (0..loom.num_axes)
        .map(|axis| domain.index.iter_domain(axis))
        .collect::<Vec<_>>();
    iteration::output_domain_groups(loom, &default)
        .into_values()
        .collect()
}

pub(super) fn candidate(
    loom: &LiftedProgram,
    domain: &DomainPlan,
    materialized: Vec<MaterializedValue>,
    mut launches: Vec<Launch>,
) -> Result<Candidate, String> {
    mark_read_later(&mut launches);
    let reductions = reduction_regions(loom, domain, &launches)?;
    let guards = guarded_regions(domain, &launches);
    let can_hoist = domain.index.fiber_bridge().is_none()
        && LoomFacts::analyze(loom).hoistable_computed_values() > 0;
    let hoists = launches
        .iter()
        .map(|launch| can_hoist && matches!(launch, Launch::Consumer { .. }))
        .collect();
    Ok(Candidate {
        materialized,
        launches,
        reductions,
        guards,
        hoists,
        origin: CandidateOrigin::Structural,
    })
}

/// A later launch selecting the same (output, element) continues it — a
/// load-add-store that reads the earlier launch's bytes back. Every launch's
/// other output stores are written once per recipe, which is what licenses
/// their `evict_first` hint at emission.
fn mark_read_later(launches: &mut [Launch]) {
    let mut selected = BTreeSet::new();
    for launch in launches.iter_mut().rev() {
        let Launch::Consumer { part, .. } = launch else {
            continue;
        };
        for (output, selections) in part.elements.iter_mut().enumerate() {
            for selection in selections.iter_mut() {
                selection.read_later = selected.contains(&(output, selection.element));
            }
        }
        for (output, selections) in part.elements.iter().enumerate() {
            for selection in selections {
                selected.insert((output, selection.element));
            }
        }
    }
}

fn guarded_regions(domain: &DomainPlan, launches: &[Launch]) -> Vec<Option<GuardedPlan>> {
    launches
        .iter()
        .map(|launch| {
            let Launch::Consumer { part, .. } = launch else {
                return None;
            };
            let fiber = domain.index.fiber_bridge().is_some();
            let mut regions = BTreeMap::<(u32, usize, bool), GuardedRegion>::new();
            let mut remainder = KernelPart {
                elements: vec![Vec::new(); part.elements.len()],
            };
            for (output, selections) in part.elements.iter().enumerate() {
                for selection in selections {
                    let mut remainder_terms = Vec::new();
                    let mut remainder_guards = Vec::new();
                    for (&term, &guard) in selection.terms.iter().zip(&selection.term_guards) {
                        let Some(guard) = guard.filter(|_| fiber) else {
                            remainder_terms.push(term);
                            remainder_guards.push(guard);
                            continue;
                        };
                        let region = regions
                            .entry((guard.value.0, guard.element, guard.inverted))
                            .or_insert_with(|| GuardedRegion {
                                key: guard,
                                outputs: Vec::new(),
                                part: KernelPart {
                                    elements: vec![Vec::new(); part.elements.len()],
                                },
                            });
                        if region.outputs.last() != Some(&(output, selection.element)) {
                            region.outputs.push((output, selection.element));
                        }
                        if let Some(part) = region.part.elements[output]
                            .last_mut()
                            .filter(|part| part.element == selection.element)
                        {
                            part.terms.push(term);
                            part.term_guards.push(Some(guard));
                        } else {
                            region.part.elements[output].push(ElementPart {
                                element: selection.element,
                                terms: vec![term],
                                continuation: selection.continuation,
                                read_later: selection.read_later,
                                reduction: selection.reduction,
                                guard: Some(guard),
                                term_guards: vec![Some(guard)],
                            });
                        }
                    }
                    if !remainder_terms.is_empty() {
                        remainder.elements[output].push(ElementPart {
                            element: selection.element,
                            terms: remainder_terms,
                            continuation: selection.continuation,
                            read_later: selection.read_later,
                            reduction: selection.reduction,
                            guard: selection.guard,
                            term_guards: remainder_guards,
                        });
                    }
                }
            }
            if regions.is_empty() {
                let mut guard = None;
                let mut outputs = Vec::new();
                for (output, selections) in part.elements.iter().enumerate() {
                    for selection in selections {
                        let candidate = selection.guard?;
                        if guard.is_some_and(|guard| guard != candidate) {
                            return None;
                        }
                        guard = Some(candidate);
                        outputs.push((output, selection.element));
                    }
                }
                let guard = guard?;
                return Some(GuardedPlan {
                    regions: vec![GuardedRegion {
                        key: guard,
                        outputs,
                        part: part.clone(),
                    }],
                    remainder: KernelPart {
                        elements: vec![Vec::new(); part.elements.len()],
                    },
                });
            }
            (!regions.is_empty()).then_some(GuardedPlan {
                regions: regions.into_values().collect(),
                remainder,
            })
        })
        .collect()
}

fn reduction_regions(
    loom: &LiftedProgram,
    domain: &DomainPlan,
    launches: &[Launch],
) -> Result<Vec<ReductionRegion>, String> {
    let mut regions = BTreeMap::<(ReductionProvenance, usize, usize), (Scalar, Vec<usize>)>::new();
    for (launch_index, launch) in launches.iter().enumerate() {
        let Launch::Consumer { part, .. } = launch else {
            continue;
        };
        for (output, selections) in part.elements.iter().enumerate() {
            let scalar = output_scalar(loom, output)?;
            let binding = &loom.outputs()[output];
            for selection in selections {
                let mut add = |provenance| {
                    let launches = &mut regions
                        .entry((provenance, output, selection.element))
                        .or_insert_with(|| (scalar, Vec::new()))
                        .1;
                    if launches.last() != Some(&launch_index) {
                        launches.push(launch_index);
                    }
                };
                if let Some(kind) = selection.reduction {
                    add(match kind {
                        ReductionKind::Reduce => ReductionProvenance::Reduce,
                        ReductionKind::Dot => ReductionProvenance::Dot,
                    });
                }
                if binding.lane_extent == 1 {
                    add(ReductionProvenance::HiddenLane);
                }
                // A fiber-scatter output is written per finer element inside
                // the fiber loop — nothing reduces over the fiber for it.
                if domain.index.fiber_bridge().is_some()
                    && !domain.index.fiber_scatter(loom.num_inputs + output)
                {
                    add(ReductionProvenance::Fiber);
                }
            }
        }
    }
    Ok(regions
        .into_iter()
        .map(
            |((provenance, output, element), (scalar, launches))| ReductionRegion {
                provenance,
                output,
                element,
                scalar,
                init_launch: launches[0],
                launches,
            },
        )
        .collect())
}

pub(super) fn consumer_launch(
    loom: &LiftedProgram,
    domain: &DomainPlan,
    part: KernelPart,
) -> Launch {
    Launch::Consumer {
        output_modes: (0..loom.outputs().len())
            .map(|output| {
                if reduction::output_fold(domain, output).combines_atomically() {
                    WriteMode::AddAtomic
                } else if part.output_continues(output) {
                    WriteMode::AddOrdered
                } else {
                    WriteMode::Store
                }
            })
            .collect(),
        part,
    }
}
