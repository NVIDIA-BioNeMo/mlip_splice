// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{LiftedProgram, PartialContributionRatio};

pub(super) fn retain_graphs_supported_for_some_ratio(
    programs: Vec<LiftedProgram>,
) -> Vec<LiftedProgram> {
    let work = programs
        .iter()
        .map(LiftedProgram::finite_map_work)
        .collect::<Vec<_>>();
    programs
        .into_iter()
        .zip(&work)
        .filter(|(_, candidate)| work_is_supported_for_some_ratio(**candidate, &work))
        .map(|(program, _)| program)
        .collect()
}

fn work_is_supported_for_some_ratio(
    candidate: xir_optimize::FiniteMapWork,
    frontier: &[xir_optimize::FiniteMapWork],
) -> bool {
    let mut lower = (1u64, 1u64);
    let mut upper = None;
    for &other in frontier {
        if candidate.after_accumulation == other.after_accumulation {
            if candidate.before_accumulation > other.before_accumulation {
                return false;
            }
            continue;
        }
        if candidate.after_accumulation > other.after_accumulation {
            if candidate.before_accumulation > other.before_accumulation {
                return false;
            }
            let bound = (
                other.before_accumulation - candidate.before_accumulation,
                candidate.after_accumulation - other.after_accumulation,
            );
            upper = match upper {
                None => Some(bound),
                Some(existing) if ratio_le(bound, existing) => Some(bound),
                Some(existing) => Some(existing),
            };
        } else if candidate.before_accumulation > other.before_accumulation {
            let bound = (
                candidate.before_accumulation - other.before_accumulation,
                other.after_accumulation - candidate.after_accumulation,
            );
            if ratio_le(lower, bound) {
                lower = bound;
            }
        }
    }
    upper.is_none_or(|upper| ratio_le(lower, upper))
}

fn ratio_le(left: (u64, u64), right: (u64, u64)) -> bool {
    u128::from(left.0) * u128::from(right.1) <= u128::from(right.0) * u128::from(left.1)
}

pub(super) fn retain_graphs_supported_by_ratios<T>(
    graphs: &mut Vec<T>,
    describe: impl Fn(
        &T,
    ) -> (
        xir_optimize::FiniteMapWork,
        &Option<Vec<PartialContributionRatio>>,
    ),
) {
    let supported = graphs
        .iter()
        .enumerate()
        .map(|(index, graph)| {
            let (work, ratios) = describe(graph);
            ratios.as_ref().is_none_or(|ratios| {
                ratios.is_empty()
                    || ratios.iter().any(|&ratio| {
                        graphs.iter().enumerate().all(|(other_index, other)| {
                            let (other_work, other_ratios) = describe(other);
                            other_index == index
                                || !other_ratios
                                    .as_ref()
                                    .is_some_and(|ratios| ratios.contains(&ratio))
                                || ratio.scaled_work(work) <= ratio.scaled_work(other_work)
                        })
                    })
            })
        })
        .collect::<Vec<_>>();
    let mut index = 0;
    graphs.retain(|_| {
        let keep = supported[index];
        index += 1;
        keep
    });
}

pub(crate) fn select_graph(
    programs: Vec<LiftedProgram>,
    options: super::CompileOptions,
) -> Result<(LiftedProgram, usize), String> {
    let settings = super::BackendSettings::default();
    let programs = retain_graphs_supported_for_some_ratio(programs);
    let compare = programs.len() > 1;
    let mut described = programs
        .into_iter()
        .map(|program| {
            let ratios = compare
                .then(|| super::planned_partial_contribution_ratios(&program, options, &settings))
                .flatten();
            let work = program.finite_map_work();
            (program, work, ratios)
        })
        .collect::<Vec<_>>();
    retain_graphs_supported_by_ratios(&mut described, |(_, work, ratios)| (*work, ratios));
    let count = described.len();
    let graph = options
        .only_frontier_graph
        .expect("explicit graph selection");
    let (program, _, _) = described
        .into_iter()
        .nth(graph)
        .ok_or_else(|| format!("loom frontier_graph {graph} outside [0, {count})"))?;
    Ok((program, count))
}

pub(crate) fn graph_record(
    program: &LiftedProgram,
    graph: usize,
    frontier_size: usize,
) -> crate::profile::FrontierGraph {
    let work = program.finite_map_work();
    let facts = super::LoomFacts::analyze(program);
    crate::profile::FrontierGraph {
        graph,
        frontier_size,
        recipe_arm: None,
        before_accumulation: work.before_accumulation,
        after_accumulation: work.after_accumulation,
        estimated_statements: facts.estimated_statements(),
        register_pressure: facts.register_pressure(),
        hoistable_computed_values: facts.hoistable_computed_values(),
    }
}

#[cfg(test)]
mod graph_frontier_tests {
    use super::*;

    #[test]
    fn unsupported_work_line_is_removed_before_compilation() {
        let frontier = [
            xir_optimize::FiniteMapWork {
                before_accumulation: 0,
                after_accumulation: 10,
            },
            xir_optimize::FiniteMapWork {
                before_accumulation: 6,
                after_accumulation: 6,
            },
            xir_optimize::FiniteMapWork {
                before_accumulation: 10,
                after_accumulation: 0,
            },
        ];

        assert!(work_is_supported_for_some_ratio(frontier[0], &frontier));
        assert!(!work_is_supported_for_some_ratio(frontier[1], &frontier));
        assert!(work_is_supported_for_some_ratio(frontier[2], &frontier));
    }

    #[test]
    fn work_line_supported_only_at_one_contribution_is_retained() {
        let frontier = [
            xir_optimize::FiniteMapWork {
                before_accumulation: 0,
                after_accumulation: 10,
            },
            xir_optimize::FiniteMapWork {
                before_accumulation: 10,
                after_accumulation: 0,
            },
        ];

        assert!(work_is_supported_for_some_ratio(frontier[0], &frontier));
    }

    #[test]
    fn concrete_planner_ratio_removes_a_losing_graph() {
        let ratio = PartialContributionRatio::new(4, 1).unwrap();
        let mut graphs = vec![
            (
                xir_optimize::FiniteMapWork {
                    before_accumulation: 0,
                    after_accumulation: 3,
                },
                Some(vec![ratio]),
            ),
            (
                xir_optimize::FiniteMapWork {
                    before_accumulation: 8,
                    after_accumulation: 0,
                },
                Some(vec![ratio]),
            ),
        ];

        retain_graphs_supported_by_ratios(&mut graphs, |(work, ratios)| (*work, ratios));

        assert_eq!(graphs.len(), 1);
        assert_eq!(graphs[0].0.before_accumulation, 8);
    }

    #[test]
    fn unknown_planner_ratio_never_removes_a_graph() {
        let ratio = PartialContributionRatio::new(4, 1).unwrap();
        let mut graphs = vec![
            (
                xir_optimize::FiniteMapWork {
                    before_accumulation: 0,
                    after_accumulation: 3,
                },
                None,
            ),
            (
                xir_optimize::FiniteMapWork {
                    before_accumulation: 8,
                    after_accumulation: 0,
                },
                Some(vec![ratio]),
            ),
        ];

        retain_graphs_supported_by_ratios(&mut graphs, |(work, ratios)| (*work, ratios));

        assert_eq!(graphs.len(), 2);
    }
}
