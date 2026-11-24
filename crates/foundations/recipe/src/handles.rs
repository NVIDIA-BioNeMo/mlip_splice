// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a producer names while it assembles a recipe.
//!
//! A builder hands out a typed id for everything it holds a table of, so a
//! module index cannot be passed where a node index belongs. Buffers get two
//! names rather than one: [`Buf`] is anything a body can address, [`Writable`]
//! is the half it may store into. The distinction is the read-only rule, and
//! having it in the type is what makes a store through an input something that
//! cannot be written down rather than something a check has to catch.

use crate::buffer::{Address, BufRef, WritableBuf};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ModuleId(usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceId(usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeId(usize);

impl ModuleId {
    pub fn from_index(index: usize) -> ModuleId {
        ModuleId(index)
    }

    pub fn index(self) -> usize {
        self.0
    }
}

impl WorkspaceId {
    pub fn from_index(index: usize) -> WorkspaceId {
        WorkspaceId(index)
    }

    pub fn index(self) -> usize {
        self.0
    }
}

impl NodeId {
    pub fn from_index(index: usize) -> NodeId {
        NodeId(index)
    }

    pub fn index(self) -> usize {
        self.0
    }
}

/// A buffer this builder can name, with workspace held as the typed id the
/// builder handed out rather than a bare index.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Buf {
    Input(usize),
    Output(usize),
    Workspace(WorkspaceId),
}

/// The buffers this builder may write. Same shape as [`Buf`] minus inputs, so
/// a producer that hands a memset an input never compiles.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Writable {
    Output(usize),
    Workspace(WorkspaceId),
}

impl From<WorkspaceId> for Buf {
    fn from(w: WorkspaceId) -> Buf {
        Buf::Workspace(w)
    }
}

impl From<WorkspaceId> for Writable {
    fn from(w: WorkspaceId) -> Writable {
        Writable::Workspace(w)
    }
}

impl From<Writable> for Buf {
    fn from(buf: Writable) -> Buf {
        match buf {
            Writable::Output(i) => Buf::Output(i),
            Writable::Workspace(w) => Buf::Workspace(w),
        }
    }
}

impl From<BufRef> for Buf {
    fn from(buf: BufRef) -> Buf {
        match buf {
            BufRef::Input(i) => Buf::Input(i),
            BufRef::Output(i) => Buf::Output(i),
            BufRef::Workspace(k) => Buf::Workspace(WorkspaceId(k)),
        }
    }
}

impl Buf {
    pub fn at(self, byte_offset: usize) -> Address<Buf> {
        Address::new(self, byte_offset)
    }

    /// The buffer this builder may write here, or `None` for an input.
    pub fn writable(self) -> Option<Writable> {
        match self {
            Buf::Input(_) => None,
            Buf::Output(i) => Some(Writable::Output(i)),
            Buf::Workspace(w) => Some(Writable::Workspace(w)),
        }
    }
}

impl Writable {
    pub fn at(self, byte_offset: usize) -> Address<Writable> {
        Address::new(self, byte_offset)
    }
}

impl From<Buf> for BufRef {
    fn from(buf: Buf) -> BufRef {
        match buf {
            Buf::Input(i) => BufRef::Input(i),
            Buf::Output(i) => BufRef::Output(i),
            Buf::Workspace(w) => BufRef::Workspace(w.index()),
        }
    }
}

impl From<Writable> for WritableBuf {
    fn from(buf: Writable) -> WritableBuf {
        match buf {
            Writable::Output(i) => WritableBuf::Output(i),
            Writable::Workspace(w) => WritableBuf::Workspace(w.index()),
        }
    }
}

impl From<Writable> for BufRef {
    fn from(buf: Writable) -> BufRef {
        WritableBuf::from(buf).readable()
    }
}
