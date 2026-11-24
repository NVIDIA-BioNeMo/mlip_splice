// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The bytes a kernel launch hands its entry point.

use crate::buffer::{Address, BufRef};

/// One kernel argument: the bytes the ABI passes, as the segments they are
/// assembled from.
///
/// The segments are private because only some sequences of them denote a
/// distinct argument. Two adjacent byte runs are one byte run — the ABI sees
/// the concatenation either way — and an empty run is nothing at all. Building
/// through [`Arg::from_segments`] is what makes a producer that emits the
/// pieces separately and one that emits them joined produce the same argument,
/// and so the same recipe.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Arg(smallvec::SmallVec<[Segment; 2]>);

impl Arg {
    /// The argument `segments` denote, in canonical form: empty byte runs
    /// dropped, adjacent byte runs joined, pointers left exactly where they
    /// are. A pointer is a distinct 8-byte field the runtime fills in, so it
    /// never merges with anything.
    pub fn from_segments(segments: impl IntoIterator<Item = Segment>) -> Arg {
        let mut canonical = smallvec::SmallVec::<[Segment; 2]>::new();
        for segment in segments {
            match segment {
                Segment::Bytes(bytes) if bytes.is_empty() => {}
                Segment::Bytes(bytes) => match canonical.last_mut() {
                    Some(Segment::Bytes(previous)) => previous.extend(bytes),
                    Some(Segment::Pointer(_)) | None => canonical.push(Segment::Bytes(bytes)),
                },
                Segment::Pointer(address) => canonical.push(Segment::Pointer(address)),
            }
        }
        Arg(canonical)
    }

    pub fn segments(&self) -> &[Segment] {
        &self.0
    }

    /// Whether `segments` are already the form [`Arg::from_segments`] would
    /// build. A decoder asks this rather than normalizing, so that bytes and
    /// value stay in one-to-one correspondence.
    pub fn is_canonical(segments: &[Segment]) -> bool {
        segments
            .iter()
            .enumerate()
            .all(|(at, segment)| match segment {
                Segment::Pointer(_) => true,
                Segment::Bytes(bytes) => {
                    !bytes.is_empty()
                        && !matches!(segments.get(at.wrapping_sub(1)), Some(Segment::Bytes(_)))
                }
            })
    }

    pub fn pointer(address: Address<BufRef>) -> Arg {
        Arg::from_segments([Segment::Pointer(address)])
    }

    pub fn scalar(bytes: Vec<u8>) -> Arg {
        Arg::from_segments([Segment::Bytes(bytes)])
    }

    pub fn input(i: usize) -> Arg {
        Arg::pointer(BufRef::Input(i).at(0))
    }

    pub fn output(i: usize) -> Arg {
        Arg::pointer(BufRef::Output(i).at(0))
    }

    pub fn workspace(i: usize) -> Arg {
        Arg::pointer(BufRef::Workspace(i).at(0))
    }

    pub fn scalar_i32(v: i32) -> Arg {
        Arg::scalar(v.to_le_bytes().to_vec())
    }

    pub fn scalar_u32(v: u32) -> Arg {
        Arg::scalar(v.to_le_bytes().to_vec())
    }

    pub fn scalar_f32(v: f32) -> Arg {
        Arg::scalar(v.to_le_bytes().to_vec())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Segment {
    Bytes(Vec<u8>),
    /// The bound device address of the buffer, advanced by the byte offset.
    /// The kernel forms its own addresses from there.
    Pointer(Address<BufRef>),
}

#[cfg(test)]
#[path = "arg_tests.rs"]
mod tests;
