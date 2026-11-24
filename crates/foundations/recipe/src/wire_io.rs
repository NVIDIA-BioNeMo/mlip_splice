// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{WireError, checked_u32};

pub(super) struct Writer {
    pub(super) bytes: Vec<u8>,
}

impl Writer {
    pub(super) fn new() -> Self {
        Writer { bytes: Vec::new() }
    }

    pub(super) fn u8(&mut self, v: u8) {
        self.bytes.push(v);
    }

    pub(super) fn u16(&mut self, v: u16) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    pub(super) fn u32(&mut self, v: u32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    pub(super) fn u64(&mut self, v: u64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    pub(super) fn blob(&mut self, b: &[u8]) -> Result<(), WireError> {
        self.u32(checked_u32(b.len(), "blob length")?);
        self.bytes.extend_from_slice(b);
        Ok(())
    }

    pub(super) fn str(&mut self, s: &str) -> Result<(), WireError> {
        self.blob(s.as_bytes())
    }

    pub(super) fn count(&mut self, n: usize) -> Result<(), WireError> {
        self.u32(checked_u32(n, "count")?);
        Ok(())
    }
}

/// Reads within one section slice; `base` is the slice's absolute offset in
/// the source buffer, so zero-copy views can be addressed in source space.
pub(super) struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    base: usize,
}

impl<'a> Cursor<'a> {
    pub(super) fn new(bytes: &'a [u8], base: usize) -> Self {
        Cursor {
            bytes,
            pos: 0,
            base,
        }
    }

    pub(super) fn abs_pos(&self) -> usize {
        self.base + self.pos
    }

    pub(super) fn take(&mut self, n: usize) -> Result<&'a [u8], WireError> {
        let end = self.pos.checked_add(n).ok_or(WireError::Truncated)?;
        let slice = self.bytes.get(self.pos..end).ok_or(WireError::Truncated)?;
        self.pos = end;
        Ok(slice)
    }

    pub(super) fn u8(&mut self) -> Result<u8, WireError> {
        Ok(self.take(1)?[0])
    }

    pub(super) fn u16(&mut self) -> Result<u16, WireError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub(super) fn u32(&mut self) -> Result<u32, WireError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub(super) fn u64(&mut self) -> Result<u64, WireError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub(super) fn blob(&mut self) -> Result<&'a [u8], WireError> {
        let len = self.u32()? as usize;
        self.take(len)
    }

    pub(super) fn str(&mut self) -> Result<String, WireError> {
        let bytes = self.blob()?;
        String::from_utf8(bytes.to_vec()).map_err(|_| WireError::BadUtf8)
    }

    pub(super) fn count(&mut self) -> Result<usize, WireError> {
        Ok(self.u32()? as usize)
    }

    pub(super) fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    /// A count whose elements each occupy at least `stride` bytes, refused
    /// unless the section still holds them all.
    ///
    /// The count comes off untrusted bytes, so a malformed one must fail here
    /// rather than downstream: a caller that sized a `Vec` from it first would
    /// have committed `u32::MAX` elements of memory to an artifact that never
    /// held them. `stride` is the *smallest* an element can be, so this bounds
    /// the allocation without having to predict the exact encoding.
    pub(super) fn bounded_count(&mut self, stride: usize) -> Result<usize, WireError> {
        let n = self.count()?;
        let needed = n.checked_mul(stride).ok_or(WireError::Truncated)?;
        if needed > self.remaining() {
            return Err(WireError::Truncated);
        }
        Ok(n)
    }
}
