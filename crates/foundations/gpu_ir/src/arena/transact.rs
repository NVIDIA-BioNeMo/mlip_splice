// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Writing a function all at once, or not at all.
//!
//! A builder owns the arena it writes, so a write that stops partway — an error
//! its caller returns, a panic from a caller driving it from another language —
//! would take the function with it. The write therefore happens in a copy, and
//! the copy takes the function's place only once the write has returned.
//!
//! The copy carries the *same tag*, which is what lets every id taken before the
//! write go on naming what it named. That is only sound while exactly one arena
//! wears the tag, so the copy is made here, held in a local, and either installed
//! or dropped: it is never handed to anyone. [`Function::transact`] takes
//! `&mut self`, so the function it copies is exclusively borrowed for as long as
//! the copy exists and no caller can hold both. Nothing else in this crate can
//! make a same-tag copy — [`Function::clone`] re-tags, decoding and parsing mint
//! — and this module is the arena's own descendant, which is why the copy is
//! reachable from here and nowhere else.
//!
//! A write that unwinds drops the copy without installing it, so the function is
//! the one it was, byte for byte and handle for handle.

use crate::build::{Body, Builder, Value};

use super::{Function, OpId, ValueId};

/// What one transaction wrote, as ids of the function it wrote into.
#[derive(Clone, Debug, Default)]
pub struct Written {
    /// The values the write produced, in the order it produced them.
    pub values: Vec<ValueId>,
    /// The steps it appended to the entry block, in order.
    pub steps: Vec<OpId>,
}

impl Function {
    /// Writes into this function through a builder: all of what `write` did, or
    /// none of it.
    ///
    /// Every id this function had before still names what it named, and the ids
    /// handed back are the ones the write added.
    pub fn transact(&mut self, write: impl FnOnce(&mut Builder) -> Vec<Value>) -> Written {
        let before = self.block(self.entry).ops.len();
        let mut builder = Builder::resuming(Body::of(self.copied(self.tag)));
        let produced = write(&mut builder);
        let written = builder.into_function();
        let held = Written {
            values: produced.iter().map(|value| value.id_in(&written)).collect(),
            steps: written.block(written.entry).ops[before..].to_vec(),
        };
        *self = written;
        held
    }
}
