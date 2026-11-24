// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    BackendSettings, IncidenceProfile, LiftedProgram, PlanIr, candidate_statement_counts,
    lane_hoist_statement_counts, select_alternatives, zero_skip_statement_counts,
};

/// How many statements each planned launch needs, per flavor, and where the
/// selected alternative plans sit in the flat alternative part index space.
/// A flavor whose count is `None` for a launch has no viable body there.
pub(super) struct LaunchEstimates {
    pub(super) baseline_index: Option<usize>,
    pub(super) alternative_indices: Vec<usize>,
    pub(super) baseline: Vec<usize>,
    pub(super) zero_skip: Vec<Option<usize>>,
    pub(super) hoist: Vec<Option<usize>>,
    pub(super) alternatives: Vec<Vec<usize>>,
}

impl LaunchEstimates {
    pub(super) fn measure(
        loom: &LiftedProgram,
        execution: &PlanIr,
        probe_block_size: u32,
        statement_cap: usize,
        incidence: &std::cell::OnceCell<Result<IncidenceProfile, String>>,
        settings: &BackendSettings,
    ) -> Result<Self, String> {
        let program = &execution.domain;
        let baseline_index = execution
            .candidates
            .iter()
            .position(|candidate| candidate.materialized.is_empty());
        let alternative_indices = if let Some(baseline) = baseline_index {
            select_alternatives(
                loom,
                &execution.candidates,
                baseline,
                incidence,
                &settings.partition,
            )?
        } else {
            (!execution.candidates.is_empty())
                .then_some(0)
                .into_iter()
                .collect()
        };
        let baseline = baseline_index
            .map(|index| {
                candidate_statement_counts(
                    program,
                    loom,
                    &execution.candidates[index],
                    probe_block_size,
                )
            })
            .transpose()?
            .unwrap_or_default();
        let mut zero_skip = baseline_index
            .map(|index| {
                let candidate = &execution.candidates[index];
                zero_skip_statement_counts(program, loom, candidate, probe_block_size)
                    .unwrap_or_else(|_| vec![None; candidate.launches.len()])
            })
            .unwrap_or_else(|| vec![None; baseline.len()]);
        let mut hoist = baseline_index
            .map(|index| {
                lane_hoist_statement_counts(
                    program,
                    loom,
                    &execution.candidates[index],
                    probe_block_size,
                )
            })
            .unwrap_or_else(|| vec![None; baseline.len()]);
        for statements in zero_skip.iter_mut().chain(hoist.iter_mut()) {
            if statements.is_some_and(|count| count > statement_cap) {
                *statements = None;
            }
        }
        let alternatives = alternative_indices
            .iter()
            .map(|&index| {
                candidate_statement_counts(
                    program,
                    loom,
                    &execution.candidates[index],
                    probe_block_size,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        for &statements in baseline.iter().chain(alternatives.iter().flatten()) {
            if statements > statement_cap {
                return Err(format!(
                    "loom planned launch needs {statements} statements, exceeding cap {statement_cap}"
                ));
            }
        }
        Ok(Self {
            baseline_index,
            alternative_indices,
            baseline,
            zero_skip,
            hoist,
            alternatives,
        })
    }
}
