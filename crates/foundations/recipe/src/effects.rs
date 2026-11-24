// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a node does to the buffers it names.
//!
//! Effects are the one authority on this. Dependencies are derived from them,
//! validation proves a store lands somewhere writable through them, landing
//! rebinds them with the work they belong to, and a choice's candidates are
//! held to them. Nothing anywhere else states what a node touches, so there is
//! no second answer to drift from.
//!
//! Passing an address is not touching the memory behind it: a computation that
//! is handed an address it never loads from declares no effect on that buffer,
//! and no edge forms against the buffer's writers. That is a statement the
//! producer makes and the recipe keeps.

use std::collections::BTreeMap;

use crate::buffer::BufRef;

/// What a node does to one buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

impl Access {
    pub fn reads(self) -> bool {
        matches!(self, Access::Read | Access::ReadWrite)
    }

    pub fn writes(self) -> bool {
        matches!(self, Access::Write | Access::ReadWrite)
    }

    /// Whether declaring this answers for `needed`.
    ///
    /// A declaration is a promise about ordering, and a writer's promise is the
    /// stronger one: it follows the last writer *and* every reader since, which
    /// is everything a reader would have followed. So a write covers a read, a
    /// read does not cover a write, and nothing covers anything unless it was
    /// declared at all — which is what makes a slot carried but undeclared a
    /// slot nothing may touch.
    pub fn covers(self, needed: Access) -> bool {
        !needed.writes() || self.writes()
    }

    /// Doing both: two statements about one buffer are one statement that says
    /// both, which is what keeps a set of effects independent of the order they
    /// were stated in.
    pub fn and(self, other: Access) -> Access {
        match (
            self.reads() || other.reads(),
            self.writes() || other.writes(),
        ) {
            (true, true) => Access::ReadWrite,
            (true, false) => Access::Read,
            (false, true) => Access::Write,
            (false, false) => Access::Read,
        }
    }
}

/// Every buffer a node touches, and what it does to each.
///
/// A map keyed by buffer, so naming a buffer twice names it once and the order
/// two effects were stated in never reaches the recipe: one set of effects is
/// one value, and two producers that discovered the same effects in different
/// orders build the same node.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Effects(BTreeMap<BufRef, Access>);

impl Effects {
    /// A node that touches nothing. Legal, and not the same as one whose
    /// effects were never stated — there is no such node.
    pub fn none() -> Effects {
        Effects::default()
    }

    /// The effects `stated` amount to, with two about one buffer kept as the
    /// one effect that says both.
    pub fn of(stated: impl IntoIterator<Item = (BufRef, Access)>) -> Effects {
        let mut effects: BTreeMap<BufRef, Access> = BTreeMap::new();
        for (buf, access) in stated {
            let entry = effects.entry(buf).or_insert(access);
            *entry = entry.and(access);
        }
        Effects(effects)
    }

    pub fn reading(bufs: impl IntoIterator<Item = BufRef>) -> Effects {
        Effects::of(bufs.into_iter().map(|buf| (buf, Access::Read)))
    }

    pub fn writing(bufs: impl IntoIterator<Item = BufRef>) -> Effects {
        Effects::of(bufs.into_iter().map(|buf| (buf, Access::Write)))
    }

    /// State one more effect. Two about one buffer become the one that says
    /// both, so a producer can discover them in any order.
    pub fn insert(&mut self, buf: BufRef, access: Access) {
        let entry = self.0.entry(buf).or_insert(access);
        *entry = entry.and(access);
    }

    pub fn with(&self, buf: BufRef, access: Access) -> Effects {
        let mut joined = self.clone();
        joined.insert(buf, access);
        joined
    }

    pub fn iter(&self) -> impl Iterator<Item = (BufRef, Access)> + '_ {
        self.0.iter().map(|(&buf, &access)| (buf, access))
    }

    pub fn access(&self, buf: BufRef) -> Option<Access> {
        self.0.get(&buf).copied()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn reads(&self) -> impl Iterator<Item = BufRef> + '_ {
        self.iter()
            .filter(|(_, access)| access.reads())
            .map(|(buf, _)| buf)
    }

    pub fn writes(&self) -> impl Iterator<Item = BufRef> + '_ {
        self.iter()
            .filter(|(_, access)| access.writes())
            .map(|(buf, _)| buf)
    }

    /// The same effects on the buffers `rebind` sends each of them to. Two
    /// buffers that land on one are one effect that says both.
    pub fn rebind<E>(
        &self,
        rebind: &mut impl FnMut(BufRef) -> Result<BufRef, E>,
    ) -> Result<Effects, E> {
        let mut landed = Vec::with_capacity(self.0.len());
        for (buf, access) in self.iter() {
            landed.push((rebind(buf)?, access));
        }
        Ok(Effects::of(landed))
    }
}
