// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the walks answer, against relations small enough to read.

use super::{first_cycle, reachable};

/// A relation stated outright: `leads_to[n]` is what `n` leads to.
fn stated<'a>(leads_to: &'a [&'a [usize]]) -> impl Fn(usize, &mut dyn FnMut(usize)) + 'a {
    move |operation, found| leads_to[operation].iter().copied().for_each(found)
}

/// `0 -> 1 -> 2`, and `3` on its own.
const CHAIN: &[&[usize]] = &[&[1], &[2], &[], &[]];

#[test]
fn a_walk_reaches_what_the_relation_leads_to() {
    let reached = reachable(4, [0], stated(CHAIN));
    assert!(reached.contains(1) && reached.contains(2));
    assert_eq!(reached.iter().collect::<Vec<_>>(), [1, 2]);
    assert_eq!(reached.len(), 2);
}

/// The starting operations are not reached by being started from: a walk says
/// what they lead to, and one of them leading to another is what puts it in.
#[test]
fn a_starting_operation_is_reached_only_by_being_led_to() {
    assert!(!reachable(4, [0], stated(CHAIN)).contains(0));
    let both = reachable(4, [0, 1], stated(CHAIN));
    assert!(
        both.contains(1),
        "0 leads to 1, so starting at 1 too keeps it"
    );
    assert!(!both.contains(0));
}

#[test]
fn a_walk_from_nothing_reaches_nothing() {
    let reached = reachable(4, [], stated(CHAIN));
    assert!(reached.is_empty());
    assert_eq!(reached.len(), 0, "and its length agrees");
    assert!(!reached.contains(0));
}

/// Two paths to one operation visit it once, which is what makes a shared
/// operand cone cost its size rather than its number of paths.
#[test]
fn an_operation_two_paths_lead_to_is_reached_once() {
    let diamond: &[&[usize]] = &[&[1, 2], &[3], &[3], &[]];
    let reached = reachable(4, [0], stated(diamond));
    assert_eq!(reached.iter().collect::<Vec<_>>(), [1, 2, 3]);
    assert_eq!(reached.len(), 3);
}

/// A walk terminates on a cycle: an operation already reached is not walked
/// again.
#[test]
fn a_walk_around_a_cycle_stops() {
    let loops: &[&[usize]] = &[&[1], &[0]];
    let reached = reachable(2, [0], stated(loops));
    assert!(reached.contains(0) && reached.contains(1));
}

#[test]
fn an_acyclic_relation_has_no_cycle() {
    assert_eq!(first_cycle(4, 0..4, stated(CHAIN)), None);
    let diamond: &[&[usize]] = &[&[1, 2], &[3], &[3], &[]];
    assert_eq!(first_cycle(4, 0..4, stated(diamond)), None);
}

#[test]
fn an_operation_leading_back_to_itself_is_a_cycle() {
    let itself: &[&[usize]] = &[&[0]];
    assert_eq!(first_cycle(1, 0..1, stated(itself)), Some(0));
    let round: &[&[usize]] = &[&[1], &[2], &[0]];
    assert_eq!(first_cycle(3, 0..3, stated(round)), Some(0));
}

/// A cycle out of reach of the roots given is not found, and one reachable
/// from any of them is: the roots decide what is examined.
#[test]
fn a_cycle_is_found_from_whatever_root_reaches_it() {
    let apart: &[&[usize]] = &[&[], &[2], &[1]];
    assert_eq!(first_cycle(3, [0], stated(apart)), None);
    assert_eq!(first_cycle(3, [1], stated(apart)), Some(1));
    assert_eq!(first_cycle(3, 0..3, stated(apart)), Some(1));
}

/// Reaching an operation whose descent already finished is ordinary sharing,
/// not a cycle — the distinction the walk's three states exist for.
#[test]
fn reaching_a_finished_operation_again_is_not_a_cycle() {
    let shared: &[&[usize]] = &[&[1], &[2], &[], &[1]];
    assert_eq!(first_cycle(4, 0..4, stated(shared)), None);
}
