// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which operations read each value.

use crate::Operations;

#[cfg(test)]
mod tests;

/// Which operations read each value, read off an arena once.
///
/// Who reads one value is a question about every operation, so asking it one
/// value at a time reads the whole arena once per question. Reading the arena
/// once answers every such question.
///
/// A value is a dense index, so its readers are a *position* in a list rather
/// than a key to look up: the whole index is two allocations, and reading it
/// is an offset. A map of vectors costs a comparison-ordered descent and an
/// allocation per value read, which is what this exists instead of.
///
/// Deleting an operation only ever removes a reader, so an index built before
/// a round of deletion stays true through it once the deleted are filtered
/// out — which [`Uses::of`] does, against the arena as it stands.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Uses {
    /// Where each value's readers begin in `readers`, with one entry past the
    /// end, so a value's readers are `readers[starts[v]..starts[v + 1]]`.
    starts: Vec<u32>,
    readers: Vec<u32>,
}

impl Uses {
    /// Reads the arena once and records which operations read each value.
    ///
    /// Two passes: the first counts what each value's slice needs, the second
    /// fills it. Counting first is what makes the whole index one allocation
    /// rather than one per value.
    pub fn of_arena(arena: &impl Operations) -> Uses {
        let mut starts = vec![0u32; arena.value_count() + 1];
        for operation in 0..arena.operation_count() {
            if !arena.is_live(operation) {
                continue;
            }
            // Counted into the following entry, so that summing in place turns
            // the counts into the starts they describe.
            arena.each_operand(operation, |value| starts[value + 1] += 1);
        }
        for index in 1..starts.len() {
            starts[index] += starts[index - 1];
        }
        let mut filled = starts.clone();
        let mut readers = vec![0u32; starts[arena.value_count()] as usize];
        for operation in 0..arena.operation_count() {
            if !arena.is_live(operation) {
                continue;
            }
            arena.each_operand(operation, |value| {
                readers[filled[value] as usize] = operation as u32;
                filled[value] += 1;
            });
        }
        Uses { starts, readers }
    }

    /// The operations reading `value`, in arena order, deleted ones included.
    ///
    /// A caller wanting only the live ones passes the arena to [`Uses::of`],
    /// which is the same question asked about an operation rather than a
    /// value.
    pub fn of_value(&self, value: usize) -> &[u32] {
        // The last entry is the sentinel past the end, so it addresses no
        // value: a `get` on it succeeds and the entry after it does not exist.
        let Some(&end) = self.starts.get(value + 1) else {
            return &[];
        };
        &self.readers[self.starts[value] as usize..end as usize]
    }

    /// The live operations reading what `operation` produces, itself excluded,
    /// each named once however many of its results they read.
    pub fn of(&self, arena: &impl Operations, operation: usize) -> Vec<u32> {
        let mut found = Vec::new();
        arena.each_result(operation, |value| {
            for &reader in self.of_value(value) {
                if reader as usize != operation && arena.is_live(reader as usize) {
                    found.push(reader);
                }
            }
        });
        found.sort_unstable();
        found.dedup();
        found
    }

    /// Whether a selected live operation other than `operation` reads what it produces.
    pub fn any_of(
        &self,
        arena: &impl Operations,
        operation: usize,
        mut predicate: impl FnMut(u32) -> bool,
    ) -> bool {
        let mut read = false;
        arena.each_result(operation, |value| {
            read = read
                || self.of_value(value).iter().any(|&reader| {
                    reader as usize != operation
                        && arena.is_live(reader as usize)
                        && predicate(reader)
                });
        });
        read
    }

    /// Whether any live operation other than `operation` reads what it
    /// produces.
    ///
    /// An operation nothing reads computes nothing, which is the question
    /// deletion actually asks. Answering it directly stops at the first
    /// reader, where [`Uses::of`] would gather and sort them all.
    pub fn any_reader(&self, arena: &impl Operations, operation: usize) -> bool {
        self.any_of(arena, operation, |_| true)
    }
}
