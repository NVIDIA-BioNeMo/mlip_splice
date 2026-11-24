// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Announcing on stderr the compilation phases that are still running.
//!
//! A phase says nothing while it stays under the interval, so a compile whose
//! every phase is short prints nothing at all. A phase that crosses the
//! interval and is still open is announced then, and once per further interval
//! it remains open. The announcement is therefore a liveness signal, observed
//! by a clock outside the phase rather than reported by the phase itself.
//!
//! Occurrences of one phase name are announced as a group: several worker
//! threads may hold the same phase open at once, and the group's age is the
//! age of its oldest open occurrence. So the output is one line per phase name
//! per interval however wide the compile fans out. Only the announcing thread
//! writes, so the lines never interleave, and a compiling thread pays one
//! uncontended lock at each end of a phase.

use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

const ANNOUNCE_INTERVAL: Duration = Duration::from_secs(60);

/// Every open occurrence of one phase name, with the number of whole intervals
/// of its oldest occurrence already announced.
struct PhaseGroup {
    started: Vec<Instant>,
    announced: u64,
}

impl PhaseGroup {
    fn oldest(&self) -> Instant {
        self.started
            .iter()
            .copied()
            .min()
            .expect("a group is dropped when its last occurrence closes")
    }
}

/// The phases open right now, and whether a thread is already announcing them.
struct OpenPhases {
    groups: HashMap<&'static str, PhaseGroup>,
    announcing: bool,
}

pub(crate) struct OpenPhaseReporter {
    interval: Duration,
    announce: Box<dyn Fn(&str) + Send + Sync>,
    phases: Mutex<OpenPhases>,
}

/// One open occurrence of a phase, closing it where it drops.
pub(crate) struct OpenPhase {
    reporter: Arc<OpenPhaseReporter>,
    name: &'static str,
    started: Instant,
}

impl Drop for OpenPhase {
    fn drop(&mut self) {
        self.reporter.close(self.name, self.started);
    }
}

impl OpenPhaseReporter {
    fn new(interval: Duration, announce: Box<dyn Fn(&str) + Send + Sync>) -> Arc<Self> {
        Arc::new(Self {
            interval: interval.max(Duration::from_nanos(1)),
            announce,
            phases: Mutex::new(OpenPhases {
                groups: HashMap::new(),
                announcing: false,
            }),
        })
    }

    fn open(self: &Arc<Self>, name: &'static str) -> OpenPhase {
        let started = Instant::now();
        let mut phases = self.locked();
        let group = phases.groups.entry(name).or_insert(PhaseGroup {
            started: Vec::new(),
            announced: 0,
        });
        group.started.push(started);
        let start_announcing = !phases.announcing;
        phases.announcing = true;
        drop(phases);
        if start_announcing {
            let reporter = Arc::clone(self);
            std::thread::spawn(move || reporter.announce_until_idle());
        }
        OpenPhase {
            reporter: Arc::clone(self),
            name,
            started,
        }
    }

    fn close(&self, name: &'static str, started: Instant) {
        let mut phases = self.locked();
        let Some(group) = phases.groups.get_mut(name) else {
            return;
        };
        if let Some(index) = group.started.iter().position(|open| *open == started) {
            group.started.swap_remove(index);
        }
        if group.started.is_empty() {
            phases.groups.remove(name);
        }
    }

    /// Announce every group that has crossed another interval, and answer how
    /// long until the next group crosses one, or `None` once nothing is open.
    fn announce_crossings(&self) -> Option<Duration> {
        let mut crossed = Vec::new();
        let mut until_next = self.interval;
        {
            let mut phases = self.locked();
            if phases.groups.is_empty() {
                phases.announcing = false;
                return None;
            }
            let now = Instant::now();
            let interval = self.interval.as_nanos();
            for (name, group) in phases.groups.iter_mut() {
                let age = now.saturating_duration_since(group.oldest());
                let intervals = (age.as_nanos() / interval) as u64;
                if intervals > group.announced {
                    crossed.push((*name, age, group.started.len()));
                }
                group.announced = intervals;
                let into_interval = Duration::from_nanos((age.as_nanos() % interval) as u64);
                until_next = until_next.min(self.interval - into_interval);
            }
        }
        for (name, age, open) in crossed {
            (self.announce)(&still_running(name, age, open));
        }
        Some(until_next)
    }

    fn announce_until_idle(&self) {
        while let Some(until_next) = self.announce_crossings() {
            std::thread::sleep(until_next);
        }
    }

    fn locked(&self) -> std::sync::MutexGuard<'_, OpenPhases> {
        self.phases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn still_running(name: &str, age: Duration, open: usize) -> String {
    let seconds = age.as_secs();
    let in_flight = if open > 1 {
        format!(" ({open} in flight)")
    } else {
        String::new()
    };
    format!(
        "loom: {name} still running after {}m{:02}s{in_flight}",
        seconds / 60,
        seconds % 60
    )
}

static REPORTER: LazyLock<Arc<OpenPhaseReporter>> = LazyLock::new(|| {
    OpenPhaseReporter::new(
        ANNOUNCE_INTERVAL,
        Box::new(|message| {
            let mut stderr = std::io::stderr().lock();
            let _ = writeln!(stderr, "{message}");
        }),
    )
});

pub(crate) fn open_phase(name: &'static str) -> OpenPhase {
    REPORTER.open(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recording(interval: Duration) -> (Arc<OpenPhaseReporter>, Arc<Mutex<Vec<String>>>) {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&lines);
        let reporter = OpenPhaseReporter::new(
            interval,
            Box::new(move |message| sink.lock().unwrap().push(message.to_owned())),
        );
        (reporter, lines)
    }

    fn announced(lines: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
        lines.lock().unwrap().clone()
    }

    #[test]
    fn a_phase_under_the_interval_announces_nothing() {
        let (reporter, lines) = recording(Duration::from_millis(200));
        drop(reporter.open("plan_ms"));
        drop(reporter.open("render_ptx_ms"));
        std::thread::sleep(Duration::from_millis(600));
        assert_eq!(announced(&lines), Vec::<String>::new());
    }

    #[test]
    fn a_phase_over_the_interval_is_announced_once_per_interval() {
        let (reporter, lines) = recording(Duration::from_millis(100));
        let open = reporter.open("render_ptx_ms");
        std::thread::sleep(Duration::from_millis(40));
        assert_eq!(announced(&lines), Vec::<String>::new());
        std::thread::sleep(Duration::from_millis(120));
        let first = announced(&lines);
        assert_eq!(first.len(), 1, "{first:?}");
        std::thread::sleep(Duration::from_millis(300));
        let later = announced(&lines);
        drop(open);
        assert!((3..=6).contains(&later.len()), "{later:?}");
        assert!(
            later
                .iter()
                .all(|line| line.starts_with("loom: render_ptx_ms still running after ")),
            "{later:?}"
        );
    }

    #[test]
    fn concurrent_occurrences_of_one_phase_are_announced_as_a_group() {
        let (reporter, lines) = recording(Duration::from_millis(100));
        let first = reporter.open("render_ptx_ms");
        let second = reporter.open("render_ptx_ms");
        std::thread::sleep(Duration::from_millis(150));
        drop(second);
        let while_both_open = announced(&lines);
        std::thread::sleep(Duration::from_millis(100));
        drop(first);
        let after_one_closed = announced(&lines);
        assert_eq!(while_both_open.len(), 1, "{while_both_open:?}");
        assert!(
            while_both_open[0].ends_with("(2 in flight)"),
            "{while_both_open:?}"
        );
        assert!(
            after_one_closed.len() > while_both_open.len(),
            "{after_one_closed:?}"
        );
        assert!(
            !after_one_closed[1].contains("in flight"),
            "{after_one_closed:?}"
        );
    }

    #[test]
    fn an_age_is_announced_in_minutes_and_seconds() {
        assert_eq!(
            still_running("plan_ms", Duration::from_secs(62), 1),
            "loom: plan_ms still running after 1m02s"
        );
        assert_eq!(
            still_running("render_ptx_ms", Duration::from_secs(600), 4),
            "loom: render_ptx_ms still running after 10m00s (4 in flight)"
        );
    }
}
