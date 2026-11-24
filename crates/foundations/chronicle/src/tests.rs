// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::Arc;

/// A sink whose bytes a test can read back.
#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl Captured {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().expect("captured bytes").clone()).expect("utf-8")
    }
}

impl Write for Captured {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .expect("captured bytes")
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A recorder of its own, so a case reads a whole capture without the process
/// wide installation, which one test binary can perform only once.
fn recording() -> (Recorder, Captured) {
    let captured = Captured::default();
    (Recorder::new(Box::new(captured.clone())), captured)
}

fn rows(text: &str) -> Vec<Vec<String>> {
    text.lines()
        .map(|line| line.split('\t').map(str::to_owned).collect())
        .collect()
}

#[test]
fn a_line_keeps_its_fields_separate_and_ends_once() {
    let mut line = String::new();
    compose(
        &mut line,
        7,
        "ThreadId(3)",
        "begin",
        11,
        "loom.shortlist candidates=2",
    );
    assert_eq!(
        line,
        "loom.profile\t7\tThreadId(3)\tbegin\t11\tloom.shortlist candidates=2\n"
    );
    assert_eq!(line.split('\t').count(), 6);
    assert_eq!(line.matches('\n').count(), 1);
}

#[test]
fn an_end_carries_no_message() {
    let mut line = String::new();
    compose(&mut line, 0, "ThreadId(1)", "end", 0, "");
    assert_eq!(line, "loom.profile\t0\tThreadId(1)\tend\t0\t\n");
}

/// A message holding the separators must not become two lines or six fields.
#[test]
fn a_message_cannot_forge_a_field_or_a_line() {
    let mut line = String::new();
    compose(&mut line, 1, "ThreadId(1)", "mark", 2, "a\tb\nc\rd\\e");
    assert_eq!(
        line,
        "loom.profile\t1\tThreadId(1)\tmark\t2\ta\\tb\\nc\\rd\\\\e\n"
    );
    assert_eq!(line.matches('\n').count(), 1);
    assert_eq!(line.split('\t').count(), 6);
    assert_eq!(rows(&line)[0][5], "a\\tb\\nc\\rd\\\\e");
}

#[test]
fn a_capture_announces_its_format_and_states_its_length() {
    let (recorder, captured) = recording();
    recorder.announce();
    let id = recorder.open();
    recorder.emit("mark", id, "loom.shortlist_ordered candidates=2");
    recorder.conclude();
    let rows = rows(&captured.text());
    assert_eq!(rows.len(), 3);
    assert!(rows[0][5].starts_with("loom.profile_open format=1 fields=\"prefix unix_ns thread kind id message\" anchor_unix_ns="));
    assert_eq!(rows[2][5], "loom.profile_close events=3 lines=3");
    assert_eq!(
        rows.iter().map(|row| row[4].as_str()).collect::<Vec<_>>(),
        ["0", "1", "2"]
    );
}

/// A range spends one id on two lines, so the two counts differ and both must
/// let a reader check the capture it holds.
#[test]
fn the_closing_counts_separate_allocations_from_lines() {
    let (recorder, captured) = recording();
    recorder.announce();
    let held = recorder.open();
    recorder.emit("begin", held, "loom.shortlist candidates=2");
    recorder.emit("end", held, "");
    recorder.conclude();
    let rows = rows(&captured.text());
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[3][5], "loom.profile_close events=3 lines=4");
}

#[test]
fn a_written_stream_pairs_every_range_and_dates_it_forward() {
    let (recorder, captured) = recording();
    let outer = recorder.open();
    recorder.emit("begin", outer, "loom.shortlist candidates=2");
    let inner = recorder.open();
    recorder.emit("mark", inner, "loom.shortlist_ordered candidates=2");
    recorder.emit("end", outer, "");
    let rows = rows(&captured.text());
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows.iter()
            .map(|row| (row[3].as_str(), row[4].as_str()))
            .collect::<Vec<_>>(),
        [("begin", "0"), ("mark", "1"), ("end", "0")]
    );
    assert_eq!(rows[2][5], "");
    let stamps = rows
        .iter()
        .map(|row| row[1].parse::<u64>().expect("nanoseconds"))
        .collect::<Vec<_>>();
    assert!(stamps[0] >= recorder.unix_ns && stamps[0] <= stamps[1] && stamps[1] <= stamps[2]);
    assert!(rows.iter().all(|row| row[0] == PREFIX && row.len() == 6));
}

#[test]
fn ids_are_unique_and_dense_across_threads() {
    let (recorder, _captured) = recording();
    let recorder = &recorder;
    let allocated = std::thread::scope(|scope| {
        let handles = (0..4)
            .map(|_| scope.spawn(|| (0..250).map(|_| recorder.open()).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("joined"))
            .collect::<std::collections::BTreeSet<_>>()
    });
    assert_eq!(
        allocated.into_iter().collect::<Vec<_>>(),
        (0..1000).collect::<Vec<_>>()
    );
}

/// The one case that installs the process wide recorder, which also proves that
/// nothing is recorded before it starts or after it ends.
#[test]
fn an_installed_sink_records_only_while_its_guard_lives() {
    assert!(
        std::env::var_os(SWITCH).is_none(),
        "this case reads the off state, so it needs {SWITCH} unset"
    );
    let mut built = 0;
    mark(|| {
        built += 1;
        String::new()
    });
    assert!(
        range(|| {
            built += 1;
            String::new()
        })
        .is_none()
    );
    assert_eq!(built, 0, "nothing is formatted while the stream is off");

    let captured = Captured::default();
    let guard = start(Box::new(captured.clone())).expect("one installation");
    mark(|| "loom.recipe_built candidates=2".into());
    drop(guard);
    mark(|| "loom.recipe_built candidates=3".into());

    let rows = rows(&captured.text());
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[1][5], "loom.recipe_built candidates=2");
    assert_eq!(rows[2][5], "loom.profile_close events=3 lines=3");
    assert!(start(Box::new(Captured::default())).is_none());
}
