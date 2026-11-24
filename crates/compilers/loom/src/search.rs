// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::VecDeque,
    ops::ControlFlow,
    time::{Duration, Instant},
};

use crate::profile::StopReason;
use crate::{CompileOptions, CompileProfile, CompiledProgram, FlatProgram, Program};

mod memory;
use memory::EstimatedMemoryBudget;

/// Compile a program into a recipe of candidate kernels, and report what the
/// compile did.
pub fn compile(
    program: Program,
    options: CompileOptions,
) -> Result<(CompiledProgram, CompileProfile), String> {
    compile_flat(program.try_into()?, options)
}

pub(crate) fn compile_flat(
    program: FlatProgram,
    options: CompileOptions,
) -> Result<(CompiledProgram, CompileProfile), String> {
    let num_inputs = program.num_inputs;
    let _building = chronicle::range(|| "loom.compile".into());
    let mut candidates = Vec::new();
    let mut profile = stream_flat_candidates(
        program,
        options,
        |candidate| {
            candidates.push(candidate.clone());
            Ok(())
        },
        |_| ControlFlow::Continue(()),
    )?;
    let selected = crate::shortlist::select(&candidates, &mut profile, options)?;
    let mut candidates = candidates.into_iter().map(Some).collect::<Vec<_>>();
    let selected = selected
        .into_iter()
        .map(|index| {
            candidates[index]
                .take()
                .expect("distinct shortlist indices")
        })
        .collect::<Vec<_>>();
    drop(candidates);
    let compiled = crate::backend::compiler::combine_compiled(selected, num_inputs)?;
    chronicle::mark(|| {
        format!(
            "loom.recipe_built candidates={} units={}",
            profile.shortlisted_candidates.len().max(1),
            compiled.recipe.context().len()
        )
    });
    Ok((compiled, profile))
}

/// Produce complete optimized source configurations, one at a time. `save` must
/// succeed before an emission is committed. `emitted` receives that candidate
/// and elapsed source time, including preparation, refusals, and saving it.
/// Its `Break` stops further work. The first viable configuration is required
/// even after time expiry; no clock bounds work before its publication. Memory
/// checks use a weighted search-volume estimate: twice reported PTX bytes plus
/// 16 KiB per complete candidate. No process RAM is measured. Charges follow
/// source attempts, which may cross the limit; complete candidates are saved
/// before stopping. Exhaustion returns the saved prefix, or an error if empty.
/// This is not a hard RAM bound. Subsequent
/// work stops at cooperative checkpoints. An error may follow successful saves;
/// it does not retract them. Source readiness does not imply assembly or
/// numerical validation.
pub fn stream_candidates(
    program: Program,
    options: CompileOptions,
    save: impl FnMut(&CompiledProgram) -> Result<(), String>,
    emitted: impl FnMut((&CompiledProgram, Duration)) -> ControlFlow<()>,
) -> Result<CompileProfile, String> {
    stream_flat_candidates(program.try_into()?, options, save, emitted)
}

pub(crate) fn stream_flat_candidates(
    request: FlatProgram,
    options: CompileOptions,
    mut save: impl FnMut(&CompiledProgram) -> Result<(), String>,
    mut emitted: impl FnMut((&CompiledProgram, Duration)) -> ControlFlow<()>,
) -> Result<CompileProfile, String> {
    let started = Instant::now();
    let options = options.validate()?;
    let _producing = chronicle::range(|| "loom.stream_candidates".into());
    let budget = SourceBudget {
        started,
        limit_ms: options
            .search_budget
            .map(|budget| budget.as_secs_f64() * 1000.0),
    };
    let mut memory = EstimatedMemoryBudget::new(options.search_size_estimate_limit);
    memory.checkpoint()?;
    let mut ledger = Publications::default();
    let mut profile = CompileProfile::default();
    let mut first_error = None;
    let mut active = VecDeque::new();
    let mut frontier = None;
    if let Some(graph) = options.only_frontier_graph {
        let (programs, work, abandoned) =
            crate::frontier::lower_program_frontier_with_options(request, options)?;
        memory.checkpoint()?;
        profile.exact_work += work;
        profile.abandoned_exact_expressions += abandoned;
        profile.timings.simplify = started.elapsed();
        let (program, count) = crate::backend::compiler::select_graph(programs, options)?;
        let opened = Instant::now();
        let passes = crate::backend::compiler::GraphPasses::new(program, options)?;
        profile
            .frontier_graphs
            .push(passes.graph_record(graph, count));
        active.push_back((graph, passes));
        profile.timings.plan += opened.elapsed();
    } else {
        let (extracted, report) =
            crate::frontier::ProgramFrontier::extract(request, options, &mut || {
                ledger.checkpoint_all(budget, &memory)
            })?;
        profile.exact_work += report.algebraic_work;
        profile.abandoned_exact_expressions += report.abandoned_algebraic_expressions;
        frontier = Some(extracted);
        profile.timings.simplify = started.elapsed();
    }
    let frontier_size = frontier
        .as_ref()
        .map_or(active.len(), |frontier| frontier.len());
    let mut next_graph = 0;
    loop {
        if ledger.checkpoint_all(budget, &memory).is_err() {
            break;
        }
        if let Some(frontier) = &mut frontier {
            let opened = Instant::now();
            let next = frontier.next(&mut || ledger.checkpoint_all(budget, &memory));
            profile.timings.simplify += opened.elapsed();
            if ledger.checkpoint_all(budget, &memory).is_err() {
                break;
            }
            if let Some(result) = next {
                let graph = next_graph;
                next_graph += 1;
                match result {
                    Ok((program, report)) => {
                        profile.exact_work += report.algebraic_work;
                        profile.abandoned_exact_expressions +=
                            report.abandoned_algebraic_expressions;
                        let opened = Instant::now();
                        match crate::backend::compiler::GraphPasses::new(program, options) {
                            Ok(passes) => {
                                profile
                                    .frontier_graphs
                                    .push(passes.graph_record(graph, frontier_size));
                                chronicle::mark(|| {
                                    format!(
                                        "loom.graph_planned graph={graph} frontier={frontier_size}"
                                    )
                                });
                                active.push_back((graph, passes));
                            }
                            Err(error) => {
                                profile.refused_attempts += 1;
                                chronicle::mark(|| format!("loom.graph_refused graph={graph}"));
                                first_error.get_or_insert(error);
                            }
                        }
                        profile.timings.plan += opened.elapsed();
                    }
                    Err(error) => {
                        profile.refused_attempts += 1;
                        first_error.get_or_insert(error);
                    }
                }
            }
        }
        if ledger.checkpoint_all(budget, &memory).is_err() {
            break;
        }
        let Some((graph, mut passes)) = active.pop_front() else {
            if frontier.as_ref().is_some_and(|frontier| frontier.len() > 0) {
                continue;
            }
            break;
        };
        if let Some((result, mut candidate_profile)) =
            passes.next(&mut || ledger.checkpoint_all(budget, &memory))
        {
            active.push_back((graph, passes));
            let ptx_bytes = candidate_profile.kernels.iter().fold(0u64, |bytes, unit| {
                bytes.saturating_add(unit.ptx_bytes as u64)
            });
            memory.charge(ptx_bytes, u64::from(result.is_ok()));
            candidate_profile.set_graph(graph);
            profile.merge(&candidate_profile);
            match result {
                Ok(candidate) => {
                    let elapsed = ledger.commit(&candidate, &mut save, || started.elapsed())?;
                    profile.candidate_elapsed.push(elapsed);
                    chronicle::mark(|| {
                        format!(
                            "loom.candidate_emitted candidate={} graph={graph} \
                             source_elapsed_ns={} units={}",
                            ledger.count - 1,
                            elapsed.as_nanos(),
                            candidate.recipe.context().len()
                        )
                    });
                    if let Some(record) = profile
                        .frontier_graphs
                        .iter_mut()
                        .find(|record| record.graph == graph)
                    {
                        record.recipe_arm.get_or_insert(ledger.count - 1);
                    }
                    let stopped = emitted((&candidate, elapsed)).is_break();
                    if memory.checkpoint().is_err() {
                        break;
                    }
                    if stopped {
                        profile.stop_reason = Some(StopReason::CallerStopped);
                        break;
                    }
                }
                Err(error) if ledger.checkpoint_all(budget, &memory).is_ok() => {
                    profile.refused_attempts += 1;
                    chronicle::mark(|| format!("loom.candidate_refused graph={graph}"));
                    first_error.get_or_insert(error);
                }
                Err(_) => {}
            }
        }
    }
    profile.search_size_estimate = memory.estimated_bytes;
    if memory.exhausted() {
        if ledger.count == 0 {
            memory.checkpoint()?;
        }
        profile.stop_reason = Some(StopReason::SearchSizeEstimateLimit);
    } else if profile.stop_reason.is_none() {
        profile.stop_reason = Some(if budget.expired() {
            StopReason::SearchBudget
        } else {
            StopReason::Exhausted
        });
    }
    chronicle::mark(|| {
        format!(
            "loom.search_stopped reason={} emitted={} refused={} graphs={}",
            profile.stop_reason.map_or("unset", StopReason::name),
            ledger.count,
            profile.refused_attempts,
            profile.frontier_graphs.len()
        )
    });
    if ledger.count == 0 {
        return Err(format!(
            "loom search found no viable optimized source configuration: {}",
            first_error.unwrap_or_else(|| "optimized frontier exhausted".into())
        ));
    }
    Ok(profile)
}

#[derive(Default)]
struct Publications {
    count: usize,
}

impl Publications {
    fn checkpoint_all(
        &self,
        budget: SourceBudget,
        memory: &EstimatedMemoryBudget,
    ) -> Result<(), String> {
        memory.checkpoint()?;
        self.checkpoint(budget)
    }

    fn checkpoint(&self, budget: SourceBudget) -> Result<(), String> {
        if self.count == 0 {
            Ok(())
        } else {
            budget.checkpoint()
        }
    }

    fn commit<T>(
        &mut self,
        candidate: &T,
        save: &mut impl FnMut(&T) -> Result<(), String>,
        elapsed: impl FnOnce() -> Duration,
    ) -> Result<Duration, String> {
        save(candidate)?;
        self.count += 1;
        Ok(elapsed())
    }
}

/// One wall-clock horizon. Non-cooperative operations may cross it, but no
/// subsequent optional stage is started after a checkpoint observes expiry.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SourceBudget {
    started: Instant,
    limit_ms: Option<f64>,
}

impl SourceBudget {
    pub(crate) fn remaining_ms(self) -> Option<f64> {
        self.limit_ms
            .map(|limit| (limit - self.started.elapsed().as_secs_f64() * 1000.0).max(0.0))
    }
    pub(crate) fn expired(self) -> bool {
        self.remaining_ms()
            .is_some_and(|remaining| remaining == 0.0)
    }
    pub(crate) fn checkpoint(self) -> Result<(), String> {
        if self.expired() {
            Err("loom optional source budget expired".into())
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests;
