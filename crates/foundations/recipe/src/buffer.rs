// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which buffer an operation names, what it may do with it, and where inside
//! it the operation starts.
//!
//! Buffers form a capability hierarchy: every buffer a body can name is
//! readable, and only outputs and workspace are writable. [`WritableBuf`] is
//! the writable half spelled as its own type, so a write through an input is
//! not a value that can be built and later rejected — it cannot be written
//! down. The one direction that always holds, writable to readable, is the
//! `From` impl below.
//!
//! Every operation reaches memory through one [`Address`]: a buffer and a byte
//! offset into it. The capability rides in the type parameter, so a memcpy
//! source (`Address<BufRef>`) and its destination (`Address<WritableBuf>`) are
//! the same construction under two capabilities, and the arithmetic that says
//! how far an access reaches is written once.

/// A buffer a body declares for itself: how many bytes it spans, and the bytes
/// it starts life holding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceBuf {
    pub nbytes: usize,
    pub init: Option<crate::blob::Blob>,
}

/// Which buffer an access names. A port index means nothing without it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufKind {
    Input,
    Output,
    Workspace,
}

impl BufKind {
    pub fn name(self) -> &'static str {
        match self {
            BufKind::Input => "input",
            BufKind::Output => "output",
            BufKind::Workspace => "workspace",
        }
    }
}

/// A buffer a body can name. Position is the port index within its kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BufRef {
    Input(usize),
    Output(usize),
    Workspace(usize),
}

/// A buffer a body may write: the outputs and workspace it can name. Inputs
/// have no spelling here, which is the whole of the read-only rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WritableBuf {
    Output(usize),
    Workspace(usize),
}

impl BufRef {
    pub fn kind(self) -> BufKind {
        match self {
            BufRef::Input(_) => BufKind::Input,
            BufRef::Output(_) => BufKind::Output,
            BufRef::Workspace(_) => BufKind::Workspace,
        }
    }

    pub fn index(self) -> usize {
        match self {
            BufRef::Input(index) | BufRef::Output(index) | BufRef::Workspace(index) => index,
        }
    }

    /// The buffer a body may write here, or `None` for an input.
    pub fn writable(self) -> Option<WritableBuf> {
        match self {
            BufRef::Input(_) => None,
            BufRef::Output(index) => Some(WritableBuf::Output(index)),
            BufRef::Workspace(index) => Some(WritableBuf::Workspace(index)),
        }
    }

    pub fn at(self, byte_offset: usize) -> Address<BufRef> {
        Address::new(self, byte_offset)
    }
}

impl WritableBuf {
    /// The same buffer named without its write capability. Every writable
    /// buffer is readable, so this never fails.
    pub fn readable(self) -> BufRef {
        match self {
            WritableBuf::Output(index) => BufRef::Output(index),
            WritableBuf::Workspace(index) => BufRef::Workspace(index),
        }
    }

    pub fn kind(self) -> BufKind {
        self.readable().kind()
    }

    pub fn index(self) -> usize {
        self.readable().index()
    }

    pub fn at(self, byte_offset: usize) -> Address<WritableBuf> {
        Address::new(self, byte_offset)
    }
}

impl From<WritableBuf> for BufRef {
    fn from(buf: WritableBuf) -> BufRef {
        buf.readable()
    }
}

/// Where in a buffer an operation starts. `B` is the capability the operation
/// needs there: [`BufRef`] to read, [`WritableBuf`] to write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Address<B> {
    pub buf: B,
    pub byte_offset: usize,
}

impl<B: Copy + Into<BufRef>> Address<B> {
    pub fn new(buf: B, byte_offset: usize) -> Self {
        Address { buf, byte_offset }
    }

    /// The same address with the buffer named without its write capability —
    /// what a reader of this memory, such as a kernel pointer argument, needs.
    pub fn readable(self) -> Address<BufRef> {
        Address {
            buf: self.buf.into(),
            byte_offset: self.byte_offset,
        }
    }

    /// One byte past the last byte an `nbytes` access from here touches, or
    /// `None` when that sum leaves the address space. Every extent check in
    /// this crate goes through it, so no caller adds the two itself.
    pub fn end(self, nbytes: usize) -> Option<usize> {
        self.byte_offset.checked_add(nbytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_writable_buffer_is_readable_at_the_same_index() {
        for writable in [WritableBuf::Output(3), WritableBuf::Workspace(7)] {
            let readable: BufRef = writable.into();
            assert_eq!(readable.index(), writable.index());
            assert_eq!(readable.kind(), writable.kind());
            assert_eq!(readable.writable(), Some(writable));
        }
    }

    /// The read-only rule is a fact about the types, not a check: an input has
    /// no writable spelling at all.
    #[test]
    fn an_input_has_no_writable_buffer() {
        assert_eq!(BufRef::Input(0).writable(), None);
    }

    #[test]
    fn an_address_reaches_its_offset_plus_the_bytes_it_touches() {
        assert_eq!(BufRef::Input(0).at(8).end(4), Some(12));
        assert_eq!(BufRef::Input(0).at(8).end(0), Some(8));
        assert_eq!(
            WritableBuf::Output(0).at(0).end(usize::MAX),
            Some(usize::MAX)
        );
    }

    #[test]
    fn an_address_past_the_end_of_the_address_space_has_no_end() {
        assert_eq!(BufRef::Workspace(0).at(1).end(usize::MAX), None);
        assert_eq!(WritableBuf::Output(0).at(usize::MAX).end(1), None);
    }

    #[test]
    fn dropping_the_write_capability_keeps_the_offset() {
        let address = WritableBuf::Workspace(2).at(64);
        assert_eq!(address.readable(), BufRef::Workspace(2).at(64));
    }
}
