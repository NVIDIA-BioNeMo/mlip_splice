// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Timestamped compile- and run-time events, off unless asked for.
//!
//! One line per event, tab separated, so a reader needs no library:
//!
//! ```text
//! loom.profile<TAB>unix_ns<TAB>ThreadId(n)<TAB>begin|end|mark<TAB>id<TAB>message
//! ```
//!
//! A `begin` and its `end` share an id; an `end` carries no message. The first
//! five fields never hold a tab, and the message escapes `\`, tab, newline and
//! carriage return, so one event is exactly one line whatever a caller writes.
//!
//! One id is allocated per `begin` or `mark`, dense from zero, so a reader pairs
//! ranges and sees that nothing was lost: allocations are lines less ends. Ids
//! are taken just before a line is written, so
//! two threads can write them out of order: event order is the timestamp's, not
//! the id's. The clock is anchored once to Unix time and advanced by a monotonic
//! instant, which keeps order and differences exact while staying comparable to
//! timestamps taken elsewhere in the process.
//!
//! A stream opens with a `loom.profile_open` mark stating the format, the field
//! names, the clock anchor and the process, and closes with a
//! `loom.profile_close` mark stating both counts, so a reader checks `lines`
//! against its own count and `events` against the greatest id. Without that mark
//! a reader still has dense ids, which show a gap but not a missing tail.
//!
//! [`SWITCH`] turns the stream on: `1` writes to standard error, and any other
//! value is a file path. Neither buffers, because nothing runs at process exit
//! to flush one: a whole line per write costs a system call and loses nothing,
//! and an empty capture would be worse than a slow one. [`start`] installs a
//! sink in process instead, for a caller that profiles one piece of work rather
//! than a whole run; that caller may buffer, since its guard flushes.
//!
//! Nothing is formatted, no clock is read, and nothing is allocated while the
//! stream is off.

use std::cell::{OnceCell, RefCell};
use std::fmt::Write as _;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, Once, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// The environment variable that turns the stream on: `1` for standard error,
/// any other value as a file path.
pub const SWITCH: &str = "LOOM_PROFILE";

const PREFIX: &str = "loom.profile";

/// The line format a reader can rely on. Raise it when a field's meaning moves.
const FORMAT: u32 = 1;

const FIELDS: &str = "prefix unix_ns thread kind id message";

/// Where events go, and the clock and counter they share.
struct Recorder {
    started: Instant,
    unix_ns: u64,
    next: AtomicU64,
    written: AtomicU64,
    sink: Mutex<Box<dyn Write + Send>>,
}

thread_local! {
    /// One reused line buffer per thread, so an event allocates nothing.
    static LINE: RefCell<String> = const { RefCell::new(String::new()) };
    /// This thread's label, formatted once rather than once per event.
    static LABEL: OnceCell<String> = const { OnceCell::new() };
}

/// Append `message` with every separator escaped, so one event is one line.
fn write_escaped(line: &mut String, message: &str) {
    for character in message.chars() {
        match character {
            '\\' => line.push_str("\\\\"),
            '\t' => line.push_str("\\t"),
            '\n' => line.push_str("\\n"),
            '\r' => line.push_str("\\r"),
            _ => line.push(character),
        }
    }
}

/// Compose one complete event line, including its newline, into `line`.
fn compose(line: &mut String, unix_ns: u64, thread: &str, kind: &str, id: u64, message: &str) {
    line.clear();
    let _ = write!(line, "{PREFIX}\t{unix_ns}\t{thread}\t{kind}\t{id}\t");
    write_escaped(line, message);
    line.push('\n');
}

impl Recorder {
    fn new(sink: Box<dyn Write + Send>) -> Self {
        Self {
            started: Instant::now(),
            unix_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            next: AtomicU64::new(0),
            written: AtomicU64::new(0),
            sink: Mutex::new(sink),
        }
    }

    fn open(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed)
    }

    /// Write one event whole, so concurrent writers cannot interleave a line.
    fn emit(&self, kind: &str, id: u64, message: &str) {
        let unix_ns = self.unix_ns + self.started.elapsed().as_nanos() as u64;
        self.written.fetch_add(1, Ordering::Relaxed);
        LABEL.with(|label| {
            let label = label.get_or_init(|| format!("{:?}", std::thread::current().id()));
            LINE.with_borrow_mut(|line| {
                compose(line, unix_ns, label, kind, id, message);
                if let Ok(mut sink) = self.sink.lock() {
                    let _ = sink.write_all(line.as_bytes());
                }
            });
        });
    }

    /// State what a reader needs before reading anything else.
    fn announce(&self) {
        let id = self.open();
        self.emit(
            "mark",
            id,
            &format!(
                "{PREFIX}_open format={FORMAT} fields=\"{FIELDS}\" anchor_unix_ns={} pid={}",
                self.unix_ns,
                std::process::id()
            ),
        );
    }

    /// State both counts a reader verifies the capture with, then flush it.
    fn conclude(&self) {
        let id = self.open();
        let lines = self.written.load(Ordering::Relaxed) + 1;
        self.emit(
            "mark",
            id,
            &format!("{PREFIX}_close events={} lines={lines}", id + 1),
        );
        if let Ok(mut sink) = self.sink.lock() {
            let _ = sink.flush();
        }
    }
}

static RECORDER: OnceLock<Recorder> = OnceLock::new();
static ENABLED: AtomicBool = AtomicBool::new(false);
static FROM_ENVIRONMENT: Once = Once::new();

/// Install `sink` as the one recorder of this process, announcing the format.
/// False when a recorder is already installed, which drops `sink` unused.
fn install(sink: Box<dyn Write + Send>) -> bool {
    let mut sink = Some(sink);
    let recorder = RECORDER.get_or_init(|| Recorder::new(sink.take().expect("one sink")));
    if sink.is_some() {
        return false;
    }
    ENABLED.store(true, Ordering::Release);
    recorder.announce();
    true
}

fn install_from_environment() {
    let Some(value) = std::env::var_os(SWITCH) else {
        return;
    };
    let sink: Box<dyn Write + Send> = if value == *"1" {
        Box::new(std::io::stderr())
    } else {
        match std::fs::File::create(&value) {
            Ok(file) => Box::new(file),
            Err(_) => return,
        }
    };
    install(sink);
}

fn recorder() -> Option<&'static Recorder> {
    FROM_ENVIRONMENT.call_once(install_from_environment);
    if !ENABLED.load(Ordering::Acquire) {
        return None;
    }
    RECORDER.get()
}

/// A stream that closes when this value is dropped.
pub struct Guard;

impl Drop for Guard {
    fn drop(&mut self) {
        finish();
    }
}

/// Record into `sink` for as long as the guard lives. `None` when a recorder is
/// already installed, whether by an earlier call or by [`SWITCH`].
pub fn start(sink: Box<dyn Write + Send>) -> Option<Guard> {
    FROM_ENVIRONMENT.call_once(install_from_environment);
    install(sink).then_some(Guard)
}

/// Close the stream: state its event count, flush it, and record nothing more.
pub fn finish() {
    let Some(recorder) = RECORDER.get() else {
        return;
    };
    if ENABLED.swap(false, Ordering::AcqRel) {
        recorder.conclude();
    }
}

/// An open range, ended when it is dropped, including while unwinding.
pub struct Range {
    recorder: &'static Recorder,
    id: u64,
}

impl Drop for Range {
    fn drop(&mut self) {
        self.recorder.emit("end", self.id, "");
    }
}

/// Begin a range whose end this value's lifetime marks. `None` when off.
pub fn range(message: impl FnOnce() -> String) -> Option<Range> {
    recorder().map(|recorder| {
        let id = recorder.open();
        recorder.emit("begin", id, &message());
        Range { recorder, id }
    })
}

/// Record one instant.
pub fn mark(message: impl FnOnce() -> String) {
    if let Some(recorder) = recorder() {
        let id = recorder.open();
        recorder.emit("mark", id, &message());
    }
}

#[cfg(test)]
mod tests;
