// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! When the tuner may still open work, and what a candidate is put back to
//! before it is timed.

use std::cell::RefCell;

use super::*;

#[test]
fn rejects_invalid_milliseconds_before_scaling() {
    for ms in [
        -1.0,
        -f64::from_bits(1),
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
    ] {
        let policy = TuningPolicy {
            scheduling_time_limit_ms: Some(ms),
        };
        let error = policy.scheduling_time_limit().unwrap_err();
        assert!(error.contains("scheduling_time_limit_ms"), "{ms}: {error}");
    }
}

#[test]
fn converts_finite_nonnegative_milliseconds() {
    assert_eq!(TuningPolicy::default().scheduling_time_limit(), Ok(None));
    for (ms, duration) in [
        (0.0, Duration::ZERO),
        (-0.0, Duration::ZERO),
        (f64::from_bits(1), Duration::ZERO),
        (300_000.0, Duration::from_secs(300)),
        (0.001, Duration::from_micros(1)),
    ] {
        let policy = TuningPolicy {
            scheduling_time_limit_ms: Some(ms),
        };
        assert_eq!(policy.scheduling_time_limit(), Ok(Some(duration)));
    }
}

#[test]
fn a_zero_public_limit_admits_no_scheduling() {
    let policy = TuningPolicy {
        scheduling_time_limit_ms: Some(0.0),
    };
    let deadline = Deadline::start(policy.scheduling_time_limit().unwrap());
    assert!(!deadline.admits_scheduling(deadline.started));
}

#[test]
fn a_deadline_admits_scheduling_until_it_expires() {
    let started = Instant::now();
    let deadline = Deadline::since(started, Some(Duration::from_millis(10)));
    assert!(deadline.admits_scheduling(started));
    assert!(deadline.admits_scheduling(started + Duration::from_millis(9)));
    assert!(!deadline.admits_scheduling(started + Duration::from_millis(10)));
    assert!(!deadline.admits_scheduling(started + Duration::from_secs(1)));
}

#[test]
fn no_deadline_admits_scheduling_forever() {
    let started = Instant::now();
    let deadline = Deadline::since(started, None);
    assert!(deadline.admits_scheduling(started + Duration::from_secs(3600)));
}

/// What a caller who keeps state by allocation does: restore the addresses it
/// saved, and pass over every other one.
struct Saved {
    /// address -> the bytes that belong to *that* allocation.
    saved: Vec<(CUdeviceptr, usize)>,
    asked: RefCell<Vec<CUdeviceptr>>,
    restored: RefCell<Vec<(CUdeviceptr, usize)>>,
}

impl Saved {
    fn of(saved: &[(CUdeviceptr, usize)]) -> Saved {
        Saved {
            saved: saved.to_vec(),
            asked: RefCell::new(Vec::new()),
            restored: RefCell::new(Vec::new()),
        }
    }
}

impl crate::validate::StartingState for Saved {
    fn restore(&self, buffer: CUdeviceptr, _stream: CUstream) -> std::result::Result<(), String> {
        self.asked.borrow_mut().push(buffer);
        if let Some(held) = self.saved.iter().find(|(address, _)| *address == buffer) {
            self.restored.borrow_mut().push(*held);
        }
        Ok(())
    }
}

/// The child's ports, as `pick_winners` builds them for a nested choice: the
/// parent's ports in the child's order, with parent workspace slices among
/// them.
fn child_outputs(
    binding: &[BufRef],
    inputs: &[CUdeviceptr],
    outputs: &[CUdeviceptr],
    workspace: &[CUdeviceptr],
) -> Vec<CUdeviceptr> {
    binding
        .iter()
        .map(|buf| parent_ptr(buf, inputs, outputs, workspace).expect("bound"))
        .collect()
}

#[test]
fn a_nested_choice_restores_by_allocation_and_not_by_position() {
    let inputs = [0x1000u64];
    let outputs = [0x2000u64, 0x3000u64];
    let workspace = [0x4000u64, 0x5000u64];
    // Only the caller's outputs are state; the workspace is scratch every run
    // carves, and their extents differ so a mixed-up pointer copies the wrong
    // number of bytes as well as the wrong bytes.
    let state = Saved::of(&[(outputs[0], 64), (outputs[1], 256)]);

    // A child that takes the parent's outputs in the other order, with a
    // workspace slice in the middle — every way a position can lie at once.
    let binding = [BufRef::Output(1), BufRef::Workspace(0), BufRef::Output(0)];
    let bound = child_outputs(&binding, &inputs, &outputs, &workspace);
    assert_eq!(bound, vec![0x3000, 0x4000, 0x2000]);

    for buffer in &bound {
        crate::validate::StartingState::restore(&state, *buffer, std::ptr::null_mut())
            .expect("restoring is not a failure");
    }

    // Each allocation got its own bytes, and the workspace slice got none.
    assert_eq!(
        *state.restored.borrow(),
        vec![(outputs[1], 256), (outputs[0], 64)]
    );
    assert!(state.asked.borrow().contains(&workspace[0]));

    // Non-vacuous: reading the same array by position — what the child's own
    // index space would have said — names the wrong allocation at every one of
    // the three, and runs off the end of the caller's two.
    let by_position: Vec<_> = bound
        .iter()
        .enumerate()
        .map(|(position, buffer)| (position, *buffer))
        .collect();
    for (position, buffer) in by_position {
        let named = state.saved.get(position);
        assert_ne!(
            named.map(|(address, _)| *address),
            Some(buffer),
            "position {position} must not name the allocation bound there"
        );
    }
}

#[test]
fn a_child_binding_that_reaches_no_saved_allocation_restores_nothing() {
    let inputs = [0x1000u64];
    let outputs = [0x2000u64];
    let workspace = [0x4000u64, 0x5000u64];
    let state = Saved::of(&[(outputs[0], 64)]);

    // A choice whose outputs are entirely workspace-local — a real shape, and
    // the one an index scheme would have silently restored the caller's state
    // into.
    let binding = [BufRef::Workspace(1), BufRef::Workspace(0)];
    let bound = child_outputs(&binding, &inputs, &outputs, &workspace);
    for buffer in &bound {
        crate::validate::StartingState::restore(&state, *buffer, std::ptr::null_mut())
            .expect("restoring is not a failure");
    }
    assert_eq!(*state.asked.borrow(), bound);
    assert!(
        state.restored.borrow().is_empty(),
        "nothing here is the caller's state"
    );
}
