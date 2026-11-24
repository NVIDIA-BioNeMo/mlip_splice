// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One argument, one spelling.
//!
//! A producer emits argument bytes in whatever pieces it had. How it cut them
//! up is not a fact about the argument, and these are the laws that make two
//! producers' output one value — and therefore one byte string and one
//! identity, which is what a cache keyed on content depends on.

use super::*;

fn address() -> Address<BufRef> {
    BufRef::Input(0).at(0)
}

/// The ABI sees one byte string per argument, so how a producer cut it up is
/// not a fact about the argument.
#[test]
fn a_byte_run_is_the_bytes_and_not_how_they_were_cut() {
    let joined = Arg::from_segments([Segment::Bytes(vec![1, 2, 3, 4])]);
    for spelling in [
        vec![Segment::Bytes(vec![1, 2]), Segment::Bytes(vec![3, 4])],
        vec![
            Segment::Bytes(vec![1]),
            Segment::Bytes(vec![2, 3]),
            Segment::Bytes(vec![4]),
        ],
        vec![
            Segment::Bytes(Vec::new()),
            Segment::Bytes(vec![1, 2, 3, 4]),
            Segment::Bytes(Vec::new()),
        ],
    ] {
        assert_eq!(Arg::from_segments(spelling.clone()), joined, "{spelling:?}");
    }
    assert_eq!(joined.segments(), [Segment::Bytes(vec![1, 2, 3, 4])]);
}

/// A pointer is a field the runtime fills in with a device address, so it is
/// its own segment however the bytes around it are cut.
#[test]
fn a_pointer_never_joins_what_is_next_to_it() {
    let arg = Arg::from_segments([
        Segment::Bytes(vec![1]),
        Segment::Bytes(vec![2]),
        Segment::Pointer(address()),
        Segment::Pointer(address()),
        Segment::Bytes(vec![3]),
    ]);
    assert_eq!(
        arg.segments(),
        [
            Segment::Bytes(vec![1, 2]),
            Segment::Pointer(address()),
            Segment::Pointer(address()),
            Segment::Bytes(vec![3]),
        ]
    );
}

/// An argument of nothing has one spelling, not one per way of writing nothing.
#[test]
fn an_argument_carrying_nothing_has_no_segments() {
    assert_eq!(
        Arg::from_segments([Segment::Bytes(Vec::new()), Segment::Bytes(Vec::new())]),
        Arg::default()
    );
    assert!(Arg::default().segments().is_empty());
}

/// What the constructor builds is what the decoder accepts, and nothing else
/// is: the two agree on which sequences are canonical, which is what keeps
/// bytes and values in one-to-one correspondence.
#[test]
fn canonical_is_exactly_what_the_constructor_produces() {
    let spellings = [
        vec![],
        vec![Segment::Bytes(vec![1, 2])],
        vec![Segment::Bytes(vec![1]), Segment::Bytes(vec![2])],
        vec![Segment::Bytes(Vec::new())],
        vec![Segment::Pointer(address())],
        vec![Segment::Pointer(address()), Segment::Pointer(address())],
        vec![Segment::Bytes(vec![1]), Segment::Pointer(address())],
        vec![
            Segment::Pointer(address()),
            Segment::Bytes(Vec::new()),
            Segment::Bytes(vec![1]),
        ],
    ];
    for spelling in spellings {
        let canonical = Arg::from_segments(spelling.clone());
        assert_eq!(
            Arg::is_canonical(&spelling),
            spelling == canonical.segments(),
            "{spelling:?}"
        );
        assert!(
            Arg::is_canonical(canonical.segments()),
            "the constructor's own output is canonical: {spelling:?}"
        );
    }
}
