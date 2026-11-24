// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What no caller can state instead of proving.
//!
//! A property this crate decides is decided here and nowhere else. Flatness is
//! one: it is a fact about a node graph, so there is no door that takes a
//! producer's word for it, and no way to reach past the one that asks.
//!
//! The golden quotes the checked constructor's signature verbatim, so changing
//! that line regenerates this snapshot: `TRYBUILD=overwrite cargo test -p
//! recipe --test sealed`.

#[test]
fn a_decidable_property_cannot_be_asserted() {
    let held = trybuild::TestCases::new();
    held.compile_fail("tests/ui/*.rs");
}
