// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a tuning run reports and how far it got: per-candidate outcomes for a
//! caller that asked for them, and the position an interrupted run stopped at.

/// Per-candidate autotune timing, off by default (`tune`); opt in via
/// `tune_reported`.
#[derive(Default, Debug)]
pub struct TuneReport {
    pub choices: Vec<ChoiceReport>,
}

#[derive(Debug)]
pub struct ChoiceReport {
    pub depth: usize,
    /// Which choice of its own body this is, by position among that body's
    /// choices.
    pub position: usize,
    /// Where this choice sits, as the exact chain of steps from the root body
    /// down to the body that holds it — empty for a choice of the root itself.
    ///
    /// Reports are emitted innermost first, so depth cannot say which candidate
    /// a nested choice belonged to, and neither can one immediate parent: two
    /// branches three levels down can repeat the same local coordinates. The
    /// whole chain is what identifies a site, and `(path, position)` is unique
    /// across a run.
    pub path: Vec<Under>,
    pub winner: Option<usize>,
    pub candidates: Vec<CandidateReport>,
}

/// One step of a descent: the choice descended through, and the candidate of it
/// that was being explored.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Under {
    pub position: usize,
    pub candidate: usize,
}

#[derive(Debug)]
pub struct CandidateReport {
    /// Which candidate of the choice this is. Resuming and the deadline both
    /// leave gaps, so a report's position is not a candidate index.
    pub candidate: usize,
    pub outcome: CandidateOutcome,
    pub entries: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateOutcome {
    Measured {
        ns: u64,
        /// The measurement was reused from an identical earlier candidate.
        cached: bool,
    },
    /// Selected without being timed: it is the only candidate on offer, or the
    /// deadline expired before this choice measured anything.
    Unmeasured,
    /// The tuning bound proved this subtree cannot win.
    Pruned,
    Infeasible(CandidateFailure),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateFailure {
    pub message: String,
    /// Whether the candidate failed of its own accord. A candidate this
    /// machine could not compile — no toolchain, an interrupted worker, a
    /// source that contradicts its own digest — is not an infeasible
    /// candidate, and a selection made around it is not this program's
    /// selection everywhere.
    pub of_the_program: bool,
}

/// Where the tuner stopped, per choice on the selected path.
///
/// This is kiln's resumable tuning state, and it is deliberately not
/// persisted. Persisting it would need an invalidation identity it does not
/// have: a partial selection is only meaningful against the very bindings it
/// was measured with, and a resumed run compares fresh measurements against an
/// incumbent it re-measures — so what is stored is not a result but a position
/// in an interrupted process. Only a *completed* realization has an identity
/// worth writing down, which is what the realized-artifact cache stores.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TuningProgress {
    pub choices: Vec<ChoiceProgress>,
    /// Candidates eliminated by this machine rather than by the program.
    pub unrepresentative_failures: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChoiceProgress {
    pub selected: usize,
    pub candidates: usize,
    /// The first candidate never scheduled; equal to `candidates` when the
    /// choice was explored to the end.
    pub unscheduled: usize,
    /// Progress of the choices inside the selected candidate.
    pub nested: Vec<ChoiceProgress>,
}

impl TuningProgress {
    /// Every choice enumerated every candidate, so this selection is the
    /// tuner's answer rather than a deadline's.
    pub fn is_complete(&self) -> bool {
        self.choices.iter().all(ChoiceProgress::is_complete)
    }
}

impl ChoiceProgress {
    pub fn is_complete(&self) -> bool {
        self.unscheduled >= self.candidates && self.nested.iter().all(ChoiceProgress::is_complete)
    }
}

/// The candidates a resumed choice schedules: the incumbent first — a time
/// measured in another call, against another device state, is not comparable
/// with the ones about to be taken — then everything the deadline never
/// reached. The candidates in between are not re-measured: they already lost
/// to this incumbent under one set of conditions, and re-running them would
/// spend the new deadline re-deciding what is decided.
///
/// Progress that cannot describe this choice — another candidate count, an
/// incumbent this choice does not have, a position past its end — is not
/// progress on it, and the choice is opened whole. [`TuningProgress`] is the
/// caller's to hold and hand back, so those three are inputs like any other
/// rather than states the tuner alone could reach.
pub(crate) fn schedule_order(resume: Option<&ChoiceProgress>, candidates: usize) -> Vec<usize> {
    let describes_this_choice = |progress: &&ChoiceProgress| {
        progress.candidates == candidates
            && progress.selected < candidates
            && progress.unscheduled <= candidates
    };
    let Some(resume) = resume.filter(describes_this_choice) else {
        return (0..candidates).collect();
    };
    std::iter::once(resume.selected)
        .chain((resume.unscheduled..candidates).filter(|&candidate| candidate != resume.selected))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn progress(selected: usize, candidates: usize, unscheduled: usize) -> ChoiceProgress {
        ChoiceProgress {
            selected,
            candidates,
            unscheduled,
            nested: Vec::new(),
        }
    }

    #[test]
    fn a_fresh_choice_schedules_every_candidate_in_order() {
        assert_eq!(schedule_order(None, 3), vec![0, 1, 2]);
    }

    #[test]
    fn a_resumed_choice_remeasures_the_incumbent_then_continues() {
        assert_eq!(schedule_order(Some(&progress(1, 5, 3)), 5), vec![1, 3, 4]);
        assert_eq!(schedule_order(Some(&progress(0, 5, 5)), 5), vec![0]);
    }

    #[test]
    fn progress_from_another_program_is_ignored() {
        // A choice that no longer has the candidates the progress counted is
        // not the choice that progress describes.
        assert_eq!(schedule_order(Some(&progress(1, 5, 3)), 2), vec![0, 1]);
        // Nor is one whose incumbent it does not have, nor one it claims to
        // have opened past its own end.
        assert_eq!(
            schedule_order(Some(&progress(5, 5, 5)), 5),
            vec![0, 1, 2, 3, 4]
        );
        assert_eq!(
            schedule_order(Some(&progress(1, 5, 9)), 5),
            vec![0, 1, 2, 3, 4]
        );
    }

    /// What a resumed choice may open, stated as the three things the second
    /// deadline is spent on — for every progress a first run can leave.
    ///
    /// A run records `selected` as a candidate it opened and `unscheduled` as
    /// one past the highest it opened, so `selected < unscheduled <= candidates`
    /// covers every position a first deadline can stop at.
    #[test]
    fn a_resumed_choice_reopens_the_incumbent_and_nothing_else_already_decided() {
        for candidates in 1..6 {
            for unscheduled in 1..=candidates {
                for selected in 0..unscheduled {
                    let resume = progress(selected, candidates, unscheduled);
                    let order = schedule_order(Some(&resume), candidates);
                    let at = |c| order.iter().filter(|&&x| x == c).count();

                    assert_eq!(
                        order.first(),
                        Some(&selected),
                        "the incumbent is remeasured first: its time came from another \
                         call against another device state, so it is not comparable with \
                         the ones about to be taken"
                    );
                    assert_eq!(at(selected), 1, "and it is opened once, not twice");
                    for loser in (0..unscheduled).filter(|&c| c != selected) {
                        assert_eq!(
                            at(loser),
                            0,
                            "candidate {loser} already lost to {selected}; re-running it \
                             would spend the new deadline re-deciding what is decided"
                        );
                    }
                    for fresh in unscheduled..candidates {
                        assert_eq!(at(fresh), 1, "candidate {fresh} was never opened");
                    }
                    assert!(order.iter().all(|&c| c < candidates));
                }
            }
        }
    }

    /// The guarantee an expired deadline rests on: every choice that has a
    /// candidate is handed one to select. The tuner stops *scheduling* at its
    /// deadline and takes the first feasible candidate of the order unmeasured,
    /// so an empty order would be the one way a choice could come back
    /// undecided — resumed or not.
    #[test]
    fn every_choice_with_a_candidate_gets_something_to_select() {
        for candidates in 1..6 {
            assert!(!schedule_order(None, candidates).is_empty());
            for unscheduled in 0..=candidates + 1 {
                for selected in 0..=candidates {
                    let resume = progress(selected, candidates, unscheduled);
                    assert!(
                        !schedule_order(Some(&resume), candidates).is_empty(),
                        "resume({selected}, {candidates}, {unscheduled}) left nothing to select"
                    );
                }
            }
        }
        // A choice with no candidate is the one case with nothing to offer, and
        // it says so rather than naming a candidate that is not there.
        assert!(schedule_order(None, 0).is_empty());
        assert!(schedule_order(Some(&progress(0, 0, 0)), 0).is_empty());
    }

    /// `unscheduled` as the tuner keeps it: seeded from the run being resumed,
    /// then raised past every candidate opened. Replaying that fold over
    /// `schedule_order` is what ties a resumed run to
    /// [`TuningProgress::is_complete`], and so to whether the realization may
    /// be published under the program's identity.
    #[test]
    fn resuming_until_the_order_runs_out_is_what_completes_a_choice() {
        let advance = |resume: Option<&ChoiceProgress>, candidates, opened: usize| {
            let order = schedule_order(resume, candidates);
            let mut unscheduled = resume.map_or(0, |progress| progress.unscheduled);
            let mut selected = resume.map_or(0, |progress| progress.selected);
            for &candidate in order.iter().take(opened) {
                unscheduled = unscheduled.max(candidate + 1);
                selected = candidate;
            }
            ChoiceProgress {
                selected,
                candidates,
                unscheduled,
                nested: Vec::new(),
            }
        };

        // One deadline that opens one candidate of five, then a second that
        // runs its order out. Two calls decide the choice; neither reopens
        // what the other settled.
        let first = advance(None, 5, 1);
        assert_eq!(first.unscheduled, 1);
        assert!(!first.is_complete());
        let second = advance(Some(&first), 5, usize::MAX);
        assert_eq!(schedule_order(Some(&first), 5), vec![0, 1, 2, 3, 4]);
        assert!(
            second.is_complete(),
            "the order ran out, so the choice is the tuner's answer"
        );

        // A deadline that stops mid-resume carries the earlier run's position
        // rather than reporting the losers it skipped as still unopened.
        let stopped = advance(Some(&progress(1, 5, 3)), 5, 2);
        assert_eq!(schedule_order(Some(&progress(1, 5, 3)), 5), vec![1, 3, 4]);
        assert_eq!(
            stopped.unscheduled, 4,
            "candidates 0..3 were decided by the first run and 3 by this one"
        );
        assert!(!stopped.is_complete());
        assert!(advance(Some(&stopped), 5, usize::MAX).is_complete());
    }

    #[test]
    fn completeness_is_the_whole_selected_path() {
        let complete = TuningProgress {
            choices: vec![ChoiceProgress {
                selected: 0,
                candidates: 2,
                unscheduled: 2,
                nested: vec![progress(1, 3, 3)],
            }],
            unrepresentative_failures: 0,
        };
        assert!(complete.is_complete());
        assert!(TuningProgress::default().is_complete());
        let truncated = TuningProgress {
            choices: vec![ChoiceProgress {
                selected: 0,
                candidates: 2,
                unscheduled: 2,
                nested: vec![progress(1, 3, 2)],
            }],
            unrepresentative_failures: 0,
        };
        assert!(
            !truncated.is_complete(),
            "a choice inside the winner is part of the selection"
        );
    }
}
