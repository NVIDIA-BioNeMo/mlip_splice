// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concatenation interpreted as UTF-8 text or its byte length.

use std::fmt;

pub(super) trait Text: fmt::Write {
    fn append(&mut self, other: &Self);
    fn len(&self) -> usize;
}

impl Text for String {
    fn append(&mut self, other: &Self) {
        self.push_str(other);
    }

    fn len(&self) -> usize {
        String::len(self)
    }
}

#[derive(Default)]
pub(super) struct ByteCount(pub(super) usize);

impl ByteCount {
    fn add(&mut self, bytes: usize) {
        self.0 = self.0.checked_add(bytes).expect("PTX text exceeds usize");
    }
}

impl fmt::Write for ByteCount {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.add(text.len());
        Ok(())
    }
}

impl Text for ByteCount {
    fn append(&mut self, other: &Self) {
        self.add(other.0);
    }

    fn len(&self) -> usize {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::{ByteCount, Text};
    use std::fmt::Write;

    #[test]
    fn formatting_and_concatenation_count_utf8_bytes() {
        let mut text = String::new();
        let mut count = ByteCount::default();
        writeln!(text, "é λ {:#x} {:.3}", 0xfeedu32, 1.25).unwrap();
        writeln!(count, "é λ {:#x} {:.3}", 0xfeedu32, 1.25).unwrap();
        let tail = "尾".to_owned();
        text.append(&tail);
        count.append(&ByteCount(tail.len()));
        assert_eq!(count.len(), text.len());
        assert!(count.len() > text.chars().count());
    }

    #[test]
    #[should_panic(expected = "PTX text exceeds usize")]
    fn an_unrepresentable_length_is_not_wrapped_or_saturated() {
        ByteCount(usize::MAX).append(&ByteCount(1));
    }
}
