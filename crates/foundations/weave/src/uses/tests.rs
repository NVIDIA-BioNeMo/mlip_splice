// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the index answers, against an arena small enough to read.

use super::Uses;
use crate::Operations;

/// An arena stated outright: what each operation reads, what it defines, and
/// whether it is still there.
struct Stated {
    operands: Vec<Vec<usize>>,
    results: Vec<Vec<usize>>,
    live: Vec<bool>,
    values: usize,
}

impl Operations for Stated {
    fn operation_count(&self) -> usize {
        self.operands.len()
    }
    fn value_count(&self) -> usize {
        self.values
    }
    fn is_live(&self, operation: usize) -> bool {
        self.live[operation]
    }
    fn each_operand(&self, operation: usize, mut read: impl FnMut(usize)) {
        self.operands[operation].iter().copied().for_each(&mut read);
    }
    fn each_result(&self, operation: usize, mut defined: impl FnMut(usize)) {
        self.results[operation]
            .iter()
            .copied()
            .for_each(&mut defined);
    }
}

/// `v0 = op0`, `v1 = op1`, `v2 = op2(v0, v1)`, `v3 = op3(v0)`.
fn shared() -> Stated {
    Stated {
        operands: vec![vec![], vec![], vec![0, 1], vec![0]],
        results: vec![vec![0], vec![1], vec![2], vec![3]],
        live: vec![true; 4],
        values: 4,
    }
}

#[test]
fn a_value_names_every_operation_reading_it() {
    let uses = Uses::of_arena(&shared());
    assert_eq!(
        uses.of_value(0),
        [2, 3],
        "both readers of v0, in arena order"
    );
    assert_eq!(uses.of_value(1), [2]);
    assert_eq!(uses.of_value(2), [], "nothing reads what op2 produces");
}

#[test]
fn a_value_no_operation_reads_has_no_readers() {
    let uses = Uses::of_arena(&shared());
    assert_eq!(uses.of_value(3), []);
    assert_eq!(uses.of_value(9), [], "and neither has one out of range");
}

/// An operation is known by what it produces, so this is the question above
/// asked from the other end.
#[test]
fn an_operation_is_read_by_whoever_reads_its_results() {
    let arena = shared();
    let uses = Uses::of_arena(&arena);
    assert_eq!(uses.of(&arena, 0), [2, 3]);
    assert_eq!(uses.of(&arena, 2), []);
    assert!(uses.any_reader(&arena, 0));
    assert!(!uses.any_reader(&arena, 2));
}

/// Deleting an operation only ever removes a reader, so an index built before
/// a deletion answers correctly after it.
#[test]
fn a_deleted_reader_stops_counting_without_rebuilding() {
    let mut arena = shared();
    let uses = Uses::of_arena(&arena);
    arena.live[2] = false;
    assert_eq!(uses.of(&arena, 0), [3], "op2 no longer reads anything");
    assert_eq!(uses.of(&arena, 1), [], "and it was v1's only reader");
    assert!(uses.any_reader(&arena, 0));
    assert!(!uses.any_reader(&arena, 1));
}

/// An operation reading two of another's results names it once, not twice.
#[test]
fn an_operation_reading_two_results_is_named_once() {
    let arena = Stated {
        operands: vec![vec![], vec![0, 1]],
        results: vec![vec![0, 1], vec![2]],
        live: vec![true; 2],
        values: 3,
    };
    let uses = Uses::of_arena(&arena);
    assert_eq!(uses.of(&arena, 0), [1]);
}

/// An operation reading what it produces is not a reader of itself: deletion
/// asks whether anything *else* still needs it.
#[test]
fn an_operation_reading_itself_is_not_its_own_reader() {
    let arena = Stated {
        operands: vec![vec![0]],
        results: vec![vec![0]],
        live: vec![true],
        values: 1,
    };
    let uses = Uses::of_arena(&arena);
    assert_eq!(uses.of(&arena, 0), []);
    assert!(!uses.any_reader(&arena, 0));
}

/// The index is built from live operations, so a deleted one never becomes a
/// reader in the first place.
#[test]
fn a_deleted_operation_is_no_reader_when_the_index_is_built() {
    let mut arena = shared();
    arena.live[3] = false;
    let uses = Uses::of_arena(&arena);
    assert_eq!(uses.of_value(0), [2]);
}

/// An arena holding nothing is an ordinary arena, not a special case.
#[test]
fn an_empty_arena_answers_about_no_value() {
    let arena = Stated {
        operands: Vec::new(),
        results: Vec::new(),
        live: Vec::new(),
        values: 0,
    };
    let uses = Uses::of_arena(&arena);
    assert_eq!(uses.of_value(0), []);
}
