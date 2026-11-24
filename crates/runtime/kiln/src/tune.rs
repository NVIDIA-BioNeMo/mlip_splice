// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::{Duration, Instant};

use cuda_driver::{CUcontext, CUdeviceptr, CUstream, Context, StreamRef};

use recipe::algebra::Launches;
use recipe::algebra::workspace_sizes;
use recipe::{BufRef, KernelLaunch, Recipe, RecipeBody, Work};

use crate::bind::Ports;
use crate::device::context_of_stream;
use crate::exec::run;
use crate::layout::carve;
use crate::progress::{
    CandidateFailure, CandidateOutcome, CandidateReport, ChoiceProgress, ChoiceReport, TuneReport,
    TuningProgress, Under, schedule_order,
};
use crate::resolve::{ResolvedRecipe, Resolver, prefetch_at};
use crate::selection::{
    Selection, Site, body_choice_indices, body_default_selection, choice_indices, inline_body,
    root_selection,
};
use crate::validate::{ValidationContract, ValidationStatus};
use crate::{Bindings, Error, ResidentRecipe};
use recipe::FlatRecipe;
use recipe::layout::checked_workspace_offsets;

mod measurement;
use measurement::{candidate_key, time_recipe};

const PROBE_ABORT_FACTOR: u64 = 10;
const BATCH_ABORT_FACTOR: u64 = 2;

/// How long the tuner may keep opening new work.
#[derive(Clone, Copy, Debug, Default)]
pub struct TuningPolicy {
    /// Wall-clock milliseconds from the start of autotuning after which no
    /// further candidate is scheduled. `None` measures every candidate; zero
    /// permits only the feasible fallback. Must be finite, nonnegative, and
    /// representable as a `Duration`.
    pub scheduling_time_limit_ms: Option<f64>,
}

impl TuningPolicy {
    /// Validate and convert the public millisecond limit without starting its clock.
    pub fn scheduling_time_limit(&self) -> Result<Option<Duration>, &'static str> {
        self.scheduling_time_limit_ms
            .map(|ms| {
                if !ms.is_finite() || ms < 0.0 {
                    return Err("scheduling_time_limit_ms must be finite and nonnegative");
                }
                Duration::try_from_secs_f64(ms / 1000.0)
                    .map_err(|_| "scheduling_time_limit_ms exceeds the supported duration")
            })
            .transpose()
    }
}

/// A limit on *scheduling*, and on nothing else.
///
/// When it expires the tuner stops opening candidates: none is compiled for
/// measurement, none is timed. It cannot stop an assembly already running in
/// the compile pool and it cannot stop a kernel already launched, so a
/// realization overruns its deadline by whatever was in flight — the deadline
/// bounds what is started, never what is finished. It never abandons the
/// recipe either: every choice still selects a feasible candidate, unmeasured
/// when the deadline expired before this choice measured anything.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Deadline {
    started: Instant,
    limit: Option<Duration>,
}

impl Deadline {
    pub(crate) fn start(limit: Option<Duration>) -> Self {
        let started = Instant::now();
        chronicle::mark(|| {
            format!(
                "kiln.deadline limit_ns={:?}",
                limit.map(|limit| limit.as_nanos())
            )
        });
        Deadline::since(started, limit)
    }

    pub(crate) fn since(started: Instant, limit: Option<Duration>) -> Self {
        Deadline { started, limit }
    }

    pub(crate) fn admits_scheduling(&self, now: Instant) -> bool {
        self.limit
            .is_none_or(|limit| now.duration_since(self.started) < limit)
    }
}

/// Every `Op::KernelLaunch` entry point in a flat recipe, in node order.
pub fn kernel_entries<F: Launches>(recipe: &Recipe<F>) -> Vec<String> {
    kernel_entries_of(recipe.body())
}

/// The same, over a body — a candidate is one, and has no ports to reach it
/// through.
fn kernel_entries_of<F: Launches>(body: &RecipeBody<F>) -> Vec<String> {
    body.nodes()
        .iter()
        .filter_map(|node| match node.work() {
            Work::Call {
                call: KernelLaunch { entry, .. },
                ..
            } => Some(entry.clone()),
            _ => None,
        })
        .collect()
}

fn parent_ptr(
    buf: &BufRef,
    inputs: &[CUdeviceptr],
    outputs: &[CUdeviceptr],
    parent_workspace: &[CUdeviceptr],
) -> std::result::Result<CUdeviceptr, Error> {
    match buf {
        BufRef::Input(i) => inputs.get(*i).copied().ok_or(Error::InputUnbound(*i)),
        BufRef::Output(i) => outputs.get(*i).copied().ok_or(Error::OutputUnbound(*i)),
        BufRef::Workspace(i) => parent_workspace
            .get(*i)
            .copied()
            .ok_or(Error::WorkspaceUnbound(*i)),
    }
}

struct CandidateRun {
    nested: Vec<Selection>,
    progress: Vec<ChoiceProgress>,
    ns: u64,
}

/// What the choice loop decided about one candidate before running it: the
/// time it must beat, whether it is timed at all, the ceilings that abandon a
/// hopeless measurement, and where a resumed run left this candidate's own
/// choices.
struct Attempt<'a> {
    bound: Option<u64>,
    measure: bool,
    probe_ceil: Option<u64>,
    batch_ceil: Option<u64>,
    resume: Option<&'a ChoiceProgress>,
}

#[derive(Default)]
struct Winners {
    selections: Vec<Selection>,
    progress: Vec<ChoiceProgress>,
}

/// The buffers one body is tuned against: the caller's ports at this level,
/// and the base of the workspace arena it may carve.
#[derive(Clone, Copy)]
struct Region<'a> {
    inputs: &'a [CUdeviceptr],
    outputs: &'a [CUdeviceptr],
    workspace_base: CUdeviceptr,
}

struct Tuner<'a, 'r, F: Launches> {
    ctx: CUcontext,
    stream: CUstream,
    deadline: Deadline,
    memo: HashMap<u64, u64>,
    /// What says this tuner may make recipes of the family it measures.
    authorization: &'a F::Authorization,
    resolver: &'a mut Resolver<'r, F>,
    report: Option<TuneReport>,
    unrepresentative_failures: usize,
    /// What a candidate is validated *inside*. A candidate proves nothing on
    /// its own, so a validation run is a run of this whole body with that one
    /// candidate selected.
    root: &'a RecipeBody<F>,
    /// The caller's ports for a validation run: their real inputs, the
    /// workspace arena the recipe was sized against, and — in place of the live
    /// outputs — the caller's candidate buffers.
    inputs: &'a [CUdeviceptr],
    workspace: CUdeviceptr,
    validation: Option<ValidationContract<'a>>,
    /// What the caller's output buffers are put back to before each timing.
    starting_state: Option<&'a dyn crate::validate::StartingState>,
    validated: usize,
    rejected: usize,
}

impl<F: Launches> Tuner<'_, '_, F> {
    fn note(&mut self, error: &Error) -> CandidateFailure {
        let of_the_program = error.is_program_property();
        if !of_the_program {
            self.unrepresentative_failures += 1;
        }
        CandidateFailure {
            message: format!("{error:?}"),
            of_the_program,
        }
    }

    /// The lowest-indexed candidate of every choice whose modules all resolve.
    /// A candidate that cannot be realized must not take its viable siblings
    /// with it, so a body's baseline is the first feasible selection, not the
    /// first one.
    fn feasible_selection(
        &mut self,
        body: &RecipeBody<F>,
    ) -> std::result::Result<Vec<Selection>, Error> {
        let mut selection = Vec::new();
        for &ci in &body_choice_indices(body) {
            let Work::Choice { candidates, .. } = body.nodes()[ci].work() else {
                unreachable!("choice_indices points only at Choice nodes")
            };
            let mut failures = Vec::new();
            let mut chosen = None;
            for (index, candidate) in candidates.iter().enumerate() {
                let feasible = self.feasible_selection(candidate).and_then(|nested| {
                    self.resolver.resolve_body(
                        inline_body(
                            self.resolver.requests(),
                            candidate,
                            &nested,
                            self.authorization,
                        )
                        .map_err(Error::InvalidRecipe)?,
                    )?;
                    Ok(nested)
                });
                match feasible {
                    Ok(nested) => {
                        chosen = Some(Selection {
                            candidate: index,
                            nested,
                        });
                        break;
                    }
                    Err(error) => {
                        self.note(&error);
                        failures.push(error);
                    }
                }
            }
            selection.push(chosen.ok_or(Error::ExhaustedChoice { node: ci, failures })?);
        }
        Ok(selection)
    }

    /// Run one feasible selection against the caller's buffers before timing
    /// anything, so every candidate is measured from the same device state.
    ///
    /// Returns the selection it ran. That selection is this body's *filler*:
    /// the one thing known to resolve here, so a validation run of some other
    /// choice's candidate can fill this body's undecided choices with it and
    /// fail only for reasons that belong to the candidate under test.
    unsafe fn warm(
        &mut self,
        body: &RecipeBody<F>,
        region: &Region<'_>,
        own_ws: &[CUdeviceptr],
        child_base: CUdeviceptr,
    ) -> std::result::Result<Vec<Selection>, Error> {
        let default = body_default_selection(body);
        let (filler, resolved) =
            match inline_body(self.resolver.requests(), body, &default, self.authorization)
                .map_err(Error::InvalidRecipe)
                .and_then(|body| self.resolver.resolve_body(body))
            {
                Ok(resolved) => (default, resolved),
                Err(error) => {
                    self.note(&error);
                    let feasible = self.feasible_selection(body)?;
                    let resolved = self.resolver.resolve_body(
                        inline_body(
                            self.resolver.requests(),
                            body,
                            &feasible,
                            self.authorization,
                        )
                        .map_err(Error::InvalidRecipe)?,
                    )?;
                    (feasible, resolved)
                }
            };
        let resolved_body = resolved.body;
        // The buffers resolution added past the ones the body declares.
        let private: Vec<usize> = workspace_sizes(&resolved_body)
            .skip(body.workspace().len())
            .collect();
        let mut loaded = unsafe { ResidentRecipe::over_body(self.ctx, resolved_body)? };
        let mut ws = own_ws.to_vec();
        ws.extend(carve(child_base, private.iter().copied())?);
        let ports = Ports {
            inputs: region.inputs,
            outputs: region.outputs,
            workspace: &ws,
        };
        unsafe { run(&mut loaded, &ports, self.stream)? };
        StreamRef::from_raw(self.stream)
            .synchronize()
            .map_err(Error::Driver)?;
        Ok(filler)
    }

    /// Run the whole recipe once with `local` selected at `site`, into the
    /// caller's candidate buffers, and ask the caller whether the result is the
    /// program's.
    ///
    /// The live outputs are not bound: this run writes only the caller's
    /// candidate storage and the workspace arena. It is one clean run from the
    /// state the caller establishes, which is what makes the comparison a
    /// statement about the candidate rather than about the accumulated history
    /// of every candidate timed before it.
    unsafe fn agrees(
        &mut self,
        site: Option<&Site<'_>>,
        local: &[Selection],
    ) -> std::result::Result<Option<usize>, Error> {
        let Some(contract) = self.validation else {
            return Ok(None);
        };
        let selection = root_selection(site, local.to_vec());
        let flat = self
            .resolver
            .resolve_body(
                inline_body(
                    self.resolver.requests(),
                    self.root,
                    &selection,
                    self.authorization,
                )
                .map_err(Error::InvalidRecipe)?,
            )?
            .body;
        let sizes: Vec<usize> = workspace_sizes(&flat).collect();
        let ws = carve(self.workspace, sizes.iter().copied())?;
        let outputs = contract.candidate_ports();
        let ports = Ports {
            inputs: self.inputs,
            outputs: &outputs,
            workspace: &ws,
        };
        let mut loaded = unsafe { ResidentRecipe::over_body(self.ctx, flat)? };
        contract.establish(self.stream)?;
        unsafe { run(&mut loaded, &ports, self.stream)? };
        StreamRef::from_raw(self.stream)
            .synchronize()
            .map_err(Error::Driver)?;
        self.validated += 1;
        let disagreement = contract.disagreement(self.stream);
        if disagreement.is_some() {
            self.rejected += 1;
        }
        Ok(disagreement)
    }

    /// Resolve one candidate, check it computes the program, and — unless
    /// `attempt.measure` is false — time it.
    unsafe fn run_candidate(
        &mut self,
        candidate: &RecipeBody<F>,
        region: &Region<'_>,
        depth: usize,
        attempt: Attempt<'_>,
        site: &Site<'_>,
    ) -> std::result::Result<Option<(CandidateRun, Vec<String>, bool)>, Error> {
        let Some(winners) = (unsafe {
            self.pick_winners(
                candidate,
                *region,
                attempt.bound,
                depth + 1,
                attempt.resume.map(|progress| progress.nested.as_slice()),
                Some(site),
            )?
        }) else {
            return Ok(None);
        };
        let flat = self
            .resolver
            .resolve_body(
                inline_body(
                    self.resolver.requests(),
                    candidate,
                    &winners.selections,
                    self.authorization,
                )
                .map_err(Error::InvalidRecipe)?,
            )?
            .body;
        let entries = kernel_entries_of(&flat);
        // A candidate earns its measurement. Validation happens after its own
        // nested choices are settled — the candidate being checked is the one
        // that would be selected — and before any timing, so a wrong candidate
        // never competes on speed.
        if let Some(output) = unsafe { self.agrees(Some(site), &winners.selections)? } {
            return Err(Error::CandidateDisagrees {
                candidate: site.candidate,
                output,
            });
        }
        let run = CandidateRun {
            nested: winners.selections,
            progress: winners.progress,
            ns: 0,
        };
        if !attempt.measure {
            return Ok(Some((run, entries, false)));
        }
        let sizes: Vec<usize> = workspace_sizes(&flat).collect();
        let ws = carve(region.workspace_base, sizes.iter().copied())?;
        let ports = Ports {
            inputs: region.inputs,
            outputs: region.outputs,
            workspace: &ws,
        };
        let key = candidate_key(&flat, &ports);
        if let Some(&ns) = key.and_then(|key| self.memo.get(&key)) {
            return Ok(Some((CandidateRun { ns, ..run }, entries, true)));
        }
        // Timing binds the caller's own outputs and replays the body, so a
        // program that reads what it writes must be put back first — otherwise
        // the second candidate measures the first one's leavings.
        self.restore(region.outputs)?;
        let mut loaded = unsafe { ResidentRecipe::over_body(self.ctx, flat)? };
        let (ns, complete) = unsafe {
            time_recipe(
                &mut loaded,
                &ports,
                self.stream,
                attempt.probe_ceil,
                attempt.batch_ceil,
            )?
        };
        if complete && let Some(key) = key {
            self.memo.insert(key, ns);
        }
        Ok(Some((CandidateRun { ns, ..run }, entries, false)))
    }

    /// Put every buffer this body writes back to what a run starts from, when
    /// the caller said what that is.
    ///
    /// The pointers are offered as they are bound here, which at a nested
    /// choice means the parent's ports in the child's order and slices of the
    /// parent's workspace among them. The caller answers by allocation, so a
    /// slice it never supplied is passed over rather than mistaken for state.
    fn restore(&self, outputs: &[CUdeviceptr]) -> std::result::Result<(), Error> {
        let Some(state) = self.starting_state else {
            return Ok(());
        };
        for &buffer in outputs {
            state
                .restore(buffer, self.stream)
                .map_err(Error::ValidationContract)?;
        }
        Ok(())
    }

    /// The chain of steps from the root body down to `site`'s body, outermost
    /// first. It is what makes one report's identity independent of every other
    /// report's, however deep or however repetitive the tree is.
    fn ancestry_of(site: Option<&Site<'_>>) -> Vec<Under> {
        let mut held = Vec::new();
        let mut walk = site;
        while let Some(step) = walk {
            held.push(Under {
                position: step.position,
                candidate: step.candidate,
            });
            walk = step.parent;
        }
        held.reverse();
        held
    }

    unsafe fn pick_winners(
        &mut self,
        body: &RecipeBody<F>,
        region: Region<'_>,
        bound: Option<u64>,
        depth: usize,
        resume: Option<&[ChoiceProgress]>,
        site: Option<&Site<'_>>,
    ) -> std::result::Result<Option<Winners>, Error> {
        let choices = body_choice_indices(body);
        if choices.is_empty() {
            return Ok(Some(Winners::default()));
        }
        // The warm-up below resolves this body's own modules and, through each
        // choice's first candidate, one whole default selection. Start that set
        // before blocking on any of it.
        self.resolver.prefetch_body(body);

        // Candidates are timed against the caller's real buffers: the recipe's
        // own workspace at the base of the bound region, each candidate's
        // private workspace carved right after it — the same arena, and for the
        // winner the same addresses, that execution will use. The region covers
        // every selection by `workspace_upper_bound`, which sizes the binding.
        let own_sizes: Vec<usize> = workspace_sizes(body).collect();
        let (_, own_total) = checked_workspace_offsets(own_sizes.iter().copied()).ok_or(
            Error::InvalidRecipe(recipe::RecipeError::WorkspaceLayoutOverflow),
        )?;
        let own_ws = carve(region.workspace_base, own_sizes.iter().copied())?;
        let child_base = region.workspace_base + own_total as CUdeviceptr;

        let filler = unsafe { self.warm(body, &region, &own_ws, child_base)? };

        let mut winners = Winners {
            selections: Vec::with_capacity(choices.len()),
            progress: Vec::with_capacity(choices.len()),
        };
        let mut committed = 0u64;
        for (index, &ci) in choices.iter().enumerate() {
            let Work::Choice {
                candidates,
                input_binding,
                output_binding,
            } = body.nodes()[ci].work()
            else {
                unreachable!("choice_indices points only at Choice nodes")
            };
            let inputs = input_binding
                .iter()
                .map(|b| parent_ptr(b, region.inputs, region.outputs, &own_ws))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let outputs = output_binding
                .iter()
                .map(|b| parent_ptr(b, region.inputs, region.outputs, &own_ws))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let child = Region {
                inputs: &inputs,
                outputs: &outputs,
                workspace_base: child_base,
            };
            let resumed = resume.and_then(|progress| progress.get(index));
            let order = schedule_order(resumed, candidates.len());
            // Candidates are compiled in the order they will be measured, a
            // bounded window ahead of the measurement — so the compile pool is
            // busy while a candidate is on the device, and no more sources are
            // outstanding than there are workers to assemble them with. The
            // window never runs past the deadline: what will not be scheduled
            // is not worth compiling.
            let window = self.resolver.prefetch_window();
            let prefetch = |resolver: &mut Resolver<'_, F>, position: usize| {
                for &ahead in prefetch_at(&order, position, window) {
                    let _prefetch = chronicle::range(|| {
                        format!("kiln.prefetch depth={depth} choice={index} candidate={ahead}")
                    });
                    resolver.prefetch_body(&candidates[ahead]);
                }
            };
            let mut unscheduled = resumed.map_or(0, |progress| progress.unscheduled);
            let mut best: Option<(usize, CandidateRun)> = None;
            let mut failures = Vec::new();
            let mut pruned_any = false;
            let mut choice_report = ChoiceReport {
                depth,
                position: index,
                path: Self::ancestry_of(site),
                winner: None,
                candidates: Vec::with_capacity(order.len()),
            };
            // A lone candidate needs no measurement to win; it still recurses
            // for its nested selections. Under a bound its time still feeds
            // `committed` — that is what strangles doomed sibling plans early —
            // so one probe run (a zero probe ceiling stops after it) estimates
            // it; without a bound the estimate would inform nothing and no run
            // happens.
            let sole = order.len() == 1;
            let residual = bound.map(|b| b.saturating_sub(committed));
            for (position, &c) in order.iter().enumerate() {
                let schedulable = self.deadline.admits_scheduling(Instant::now());
                if !schedulable && best.is_some() {
                    break;
                }
                if schedulable {
                    prefetch(self.resolver, position);
                }
                let measure = (!sole || bound.is_some()) && schedulable;
                let best_ns = best.as_ref().map(|(_, run)| run.ns);
                let attempt = Attempt {
                    bound: tighter(residual, best_ns),
                    measure,
                    probe_ceil: if sole {
                        Some(0)
                    } else {
                        best_ns.map(|b| b.saturating_mul(PROBE_ABORT_FACTOR))
                    },
                    batch_ceil: if sole {
                        None
                    } else {
                        tighter(
                            best_ns.map(|b| b.saturating_mul(BATCH_ABORT_FACTOR)),
                            residual,
                        )
                    },
                    resume: resumed.filter(|progress| progress.selected == c),
                };
                unscheduled = unscheduled.max(c + 1);
                // What the rest of the recipe takes while this candidate is
                // under test: the choices of this body that have already been
                // decided, and the warm-up's own selection for the ones that
                // have not.
                let mut siblings = filler.clone();
                siblings[..winners.selections.len()].clone_from_slice(&winners.selections);
                let here = Site {
                    parent: site,
                    position: index,
                    candidate: c,
                    siblings,
                };
                let _candidate = chronicle::range(|| {
                    format!(
                        "kiln.candidate depth={depth} choice={index} candidate={c} path={:?}",
                        Self::ancestry_of(site)
                    )
                });
                let timed =
                    unsafe { self.run_candidate(&candidates[c], &child, depth, attempt, &here) };
                chronicle::mark(|| {
                    let status = match &timed {
                        Ok(Some((_, _, true))) if measure => "cached",
                        Ok(Some(_)) if measure => "measured",
                        Ok(Some(_)) => "unmeasured",
                        Ok(None) => "pruned",
                        Err(_) => "infeasible",
                    };
                    format!(
                        "kiln.candidate_result depth={depth} choice={index} candidate={c} status={status}"
                    )
                });
                match timed {
                    Ok(Some((run, entries, cached))) => {
                        choice_report.candidates.push(CandidateReport {
                            candidate: c,
                            outcome: if measure {
                                CandidateOutcome::Measured { ns: run.ns, cached }
                            } else {
                                CandidateOutcome::Unmeasured
                            },
                            entries,
                        });
                        if best.as_ref().is_none_or(|(_, b)| run.ns < b.ns) {
                            best = Some((c, run));
                        }
                    }
                    Ok(None) => {
                        pruned_any = true;
                        choice_report.candidates.push(CandidateReport {
                            candidate: c,
                            outcome: CandidateOutcome::Pruned,
                            entries: Vec::new(),
                        });
                    }
                    Err(error) => {
                        let failure = self.note(&error);
                        choice_report.candidates.push(CandidateReport {
                            candidate: c,
                            outcome: CandidateOutcome::Infeasible(failure),
                            entries: Vec::new(),
                        });
                        failures.push(error);
                    }
                }
                // The deadline stops scheduling, but never leaves a choice
                // without a candidate: the first feasible one is selected
                // unmeasured.
                if !schedulable && best.is_some() {
                    break;
                }
            }
            let Some((winner, run)) = best else {
                if pruned_any {
                    return Ok(None);
                }
                if let Some(report) = self.report.as_mut() {
                    report.choices.push(choice_report);
                }
                return Err(Error::ExhaustedChoice { node: ci, failures });
            };
            choice_report.winner = Some(winner);
            if let Some(report) = self.report.as_mut() {
                report.choices.push(choice_report);
            }
            winners.progress.push(ChoiceProgress {
                selected: winner,
                candidates: candidates.len(),
                unscheduled,
                nested: run.progress,
            });
            winners.selections.push(Selection {
                candidate: winner,
                nested: run.nested,
            });
            committed = committed.saturating_add(run.ns);
            if bound.is_some_and(|b| committed >= b) {
                return Ok(None);
            }
        }
        Ok(Some(winners))
    }
}

fn tighter(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

/// One tuning run: which program, against which bindings, under which limit,
/// continuing from which interrupted run.
pub(crate) struct TuningRun<'a, F: Launches = recipe::Executable> {
    pub(crate) ctx: CUcontext,
    pub(crate) recipe: &'a Recipe<F>,
    /// What says this run may make recipes of the family it is tuning: every
    /// body it inlines to measure one becomes work of that family.
    pub(crate) authorization: &'a F::Authorization,
    pub(crate) bindings: &'a Bindings<'a>,
    pub(crate) stream: CUstream,
    pub(crate) deadline: Deadline,
    pub(crate) resume: Option<&'a TuningProgress>,
    pub(crate) report: Option<TuneReport>,
    pub(crate) validation: Option<ValidationContract<'a>>,
    pub(crate) starting_state: Option<&'a dyn crate::validate::StartingState>,
}

pub(crate) struct TuningResult<F: Launches = recipe::Executable> {
    pub(crate) resolved: ResolvedRecipe<F>,
    pub(crate) report: Option<TuneReport>,
    pub(crate) progress: TuningProgress,
    pub(crate) validation: ValidationStatus,
}

pub(crate) unsafe fn tune_with_resolver<F: Launches>(
    run: TuningRun<'_, F>,
    resolver: &mut Resolver<'_, F>,
) -> std::result::Result<TuningResult<F>, Error> {
    // Validation selects among candidates; a recipe that states one program
    // offers nothing to select, so there is nothing here to reject.
    let outputs = run.validation.map_or(0, |contract| contract.outputs.len());
    if choice_indices(run.recipe).is_empty() {
        let mut resolved = resolver.resolve_body(run.recipe.body().clone())?;
        resolved.detach_blobs();
        return Ok(TuningResult {
            resolved,
            report: run.report,
            progress: TuningProgress::default(),
            validation: match run.validation {
                None => ValidationStatus::NotRequested,
                Some(_) => ValidationStatus::Checked {
                    outputs,
                    validated: 0,
                    rejected: 0,
                },
            },
        });
    }

    Context::from_raw(run.ctx)
        .set_current()
        .map_err(Error::Driver)?;

    let mut tuner = Tuner {
        authorization: run.authorization,
        ctx: run.ctx,
        stream: run.stream,
        deadline: run.deadline,
        memo: HashMap::new(),
        resolver,
        report: run.report,
        unrepresentative_failures: 0,
        root: run.recipe.body(),
        inputs: run.bindings.inputs,
        workspace: run.bindings.workspace,
        validation: run.validation,
        starting_state: run.starting_state,
        validated: 0,
        rejected: 0,
    };
    let region = Region {
        inputs: run.bindings.inputs,
        outputs: run.bindings.outputs,
        workspace_base: run.bindings.workspace,
    };
    let winners = unsafe {
        tuner.pick_winners(
            run.recipe.body(),
            region,
            None,
            0,
            run.resume.map(|progress| progress.choices.as_slice()),
            None,
        )?
    }
    .expect("the top-level recipe has no bound, so it is never pruned");
    let progress = TuningProgress {
        choices: winners.progress,
        unrepresentative_failures: tuner.unrepresentative_failures,
    };
    let validation = match run.validation {
        None => ValidationStatus::NotRequested,
        Some(_) => ValidationStatus::Checked {
            outputs,
            validated: tuner.validated,
            rejected: tuner.rejected,
        },
    };
    let report = tuner.report;
    let mut resolved = resolver.resolve_body(
        inline_body(
            run.recipe.context(),
            run.recipe.body(),
            &winners.selections,
            run.authorization,
        )
        .map_err(Error::InvalidRecipe)?,
    )?;
    // The winner re-owns its payload bytes so the pending recipe's backing
    // buffer (every loser candidate) can be released.
    resolved.detach_blobs();
    Ok(TuningResult {
        resolved,
        report,
        progress,
        validation,
    })
}

/// Tune every `Choice` against the caller's real bindings — the buffers the
/// tuned recipe will execute with. Candidate runs write `bindings.outputs`
/// and the workspace; the caller executes the winner afterwards, so both are
/// scratch until then. The workspace must cover the recipe's
/// `workspace_upper_bound`.
pub unsafe fn tune(
    recipe: &Recipe,
    bindings: &Bindings,
    stream: CUstream,
) -> std::result::Result<FlatRecipe, Error> {
    unsafe { tune_run(recipe, bindings, stream, None) }?
        .resolved
        .into_flat(recipe.contract().clone(), &())
}

/// Same as `tune`, but also returns per-candidate timing telemetry: an empty
/// report when `recipe` has no `Choice` nodes.
pub unsafe fn tune_reported(
    recipe: &Recipe,
    bindings: &Bindings,
    stream: CUstream,
) -> std::result::Result<(FlatRecipe, TuneReport), Error> {
    let result = unsafe { tune_run(recipe, bindings, stream, Some(TuneReport::default())) }?;
    Ok((
        result.resolved.into_flat(recipe.contract().clone(), &())?,
        result.report.unwrap_or_default(),
    ))
}

unsafe fn tune_run(
    recipe: &Recipe,
    bindings: &Bindings,
    stream: CUstream,
    report: Option<TuneReport>,
) -> std::result::Result<TuningResult, Error> {
    let limit = TuningPolicy::default()
        .scheduling_time_limit()
        .map_err(Error::InvalidTuningPolicy)?;
    let ctx = unsafe { context_of_stream(stream)? };
    let realizer = crate::Realizer::new(ctx, crate::RealizerOptions::default())?;
    let mut resolver = realizer.resolver(recipe)?;
    unsafe {
        tune_with_resolver(
            TuningRun {
                authorization: &(),
                starting_state: None,
                ctx,
                recipe,
                bindings,
                stream,
                deadline: Deadline::start(limit),
                resume: None,
                report,
                validation: None,
            },
            &mut resolver,
        )
    }
}

#[cfg(test)]
#[path = "tune_tests.rs"]
mod tests;
