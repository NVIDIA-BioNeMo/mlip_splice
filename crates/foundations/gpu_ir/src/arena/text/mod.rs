// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The arena written down as text, and that text read back as the same arena.
//!
//! This is the device half of one durable editing format. It prints and reads
//! **everything the arena states** — every value, operation, block, region,
//! declaration, constant and name — under one spelling each, so a rendered
//! arena and the arena it came from carry the same facts and no others. That is
//! what makes render → parse → render exact: nothing is inferred on the way out
//! and nothing is normalized on the way in.
//!
//! Numbers are written as their bits. A float constant is an `f64` in the
//! arena, an immediate is bits the ABI packs, and constant array data is bytes;
//! a decimal spelling of any of them is a conversion, and a conversion is where
//! a bit goes missing. Hexadecimal is exact and says the same thing on every
//! machine.
//!
//! A **function** is written here whole. [`render_device_function`] prints its
//! name and either the arena that is its body or the contract a declaration
//! states — every parameter, what it does through each, what it hands back and
//! the threads it needs — and [`parse_device_function`] reads either back. A
//! declaration therefore survives this text on its own: nothing about it is
//! recovered from a copy the reader was already holding.
//!
//! What is *not* written here is which function a call runs. That is a fact
//! about the *module* the body belongs to, so a call names a symbol its caller
//! resolves — [`RawFunction::calls`] on the way in, and the symbol a renderer is
//! handed on the way out.

mod parse;
mod render;
#[cfg(test)]
mod tests;

pub use parse::{
    At, RawBlock, RawDef, RawDeviceFunction, RawFunction, RawImplementation, RawKind, RawOp,
    RawRegion, RawValue, Ref, parse_device_function, parse_function, parse_signature,
};
pub use render::{render_device_function, render_function, render_signature};

use std::fmt;

use crate::effect::Sync;
use crate::op::{Trip, WriteMode};
use crate::{BinOp, CmpOp, Evict, Scalar, Space, Sreg, UnOp};

/// Why text is not what it claims to be, and where in it that was decided.
///
/// Line and column are one-based and name the token the reader was looking at,
/// which is the token to fix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextError {
    pub line: usize,
    pub column: usize,
    pub detail: String,
}

impl fmt::Display for TextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "line {}, column {}: {}",
            self.line, self.column, self.detail
        )
    }
}

impl std::error::Error for TextError {}

pub type Result<T> = std::result::Result<T, TextError>;

/// A reader over the whole document, shared by every grammar written in this
/// language: the device half here, and the module half above it.
///
/// Whitespace is nothing and `#` runs to the end of its line, so a caller may
/// lay the text out and annotate it however it likes; what comes back is the
/// canonical spelling either way.
pub struct Reader<'a> {
    text: &'a str,
    at: usize,
    /// Where the token last handed out began, which is the place a refusal is
    /// about.
    token: usize,
}

impl<'a> Reader<'a> {
    pub fn new(text: &'a str) -> Reader<'a> {
        Reader {
            text,
            at: 0,
            token: 0,
        }
    }

    /// The line and column one-based, of a byte offset.
    fn position(&self, at: usize) -> (usize, usize) {
        let before = &self.text[..at.min(self.text.len())];
        let line = before.matches('\n').count() + 1;
        let column = before.len() - before.rfind('\n').map_or(0, |held| held + 1) + 1;
        (line, column)
    }

    /// Where the reader stands, one-based, for a caller that has to point back
    /// here once it knows more than it does now.
    pub fn here(&mut self) -> (usize, usize) {
        self.skip();
        self.position(self.at)
    }

    pub fn fail<T>(&self, detail: impl Into<String>) -> Result<T> {
        let (line, column) = self.position(self.token);
        Err(TextError {
            line,
            column,
            detail: detail.into(),
        })
    }

    fn skip(&mut self) {
        let bytes = self.text.as_bytes();
        while self.at < bytes.len() {
            match bytes[self.at] {
                b' ' | b'\t' | b'\r' | b'\n' => self.at += 1,
                b'#' => {
                    while self.at < bytes.len() && bytes[self.at] != b'\n' {
                        self.at += 1;
                    }
                }
                _ => break,
            }
        }
    }

    /// What comes next without taking it: a word, a bracket, a quoted string, or
    /// the end.
    pub fn ahead(&mut self) -> Ahead<'a> {
        self.skip();
        let bytes = self.text.as_bytes();
        if self.at >= bytes.len() {
            return Ahead::End;
        }
        match bytes[self.at] {
            b'[' => Ahead::Open,
            b']' => Ahead::Close,
            b'"' => Ahead::Text,
            _ => Ahead::Word(self.scan()),
        }
    }

    /// The run of characters a word is, from where the reader stands.
    fn scan(&self) -> &'a str {
        let bytes = self.text.as_bytes();
        let mut end = self.at;
        while end < bytes.len()
            && !matches!(
                bytes[end],
                b' ' | b'\t' | b'\r' | b'\n' | b'[' | b']' | b'"' | b'#'
            )
        {
            end += 1;
        }
        &self.text[self.at..end]
    }

    pub fn at_end(&mut self) -> bool {
        matches!(self.ahead(), Ahead::End)
    }

    pub fn word(&mut self) -> Result<&'a str> {
        self.skip();
        self.token = self.at;
        match self.ahead() {
            Ahead::Word(held) => {
                self.at += held.len();
                Ok(held)
            }
            _ => self.fail("a word belongs here"),
        }
    }

    pub fn keyword(&mut self, expected: &str) -> Result<()> {
        let held = self.word()?;
        if held != expected {
            return self.fail(format!("{expected:?} belongs here and this says {held:?}"));
        }
        Ok(())
    }

    /// Whether the next word is `expected`, taking it when it is.
    pub fn taking(&mut self, expected: &str) -> bool {
        if matches!(self.ahead(), Ahead::Word(held) if held == expected) {
            self.at += expected.len();
            return true;
        }
        false
    }

    /// A quoted string, with `\\`, `\"`, `\n`, `\t`, `\r` and `\u{...}` read
    /// back as what [`escaped`] wrote.
    pub fn quoted(&mut self) -> Result<String> {
        self.skip();
        self.token = self.at;
        let bytes = self.text.as_bytes();
        if self.at >= bytes.len() || bytes[self.at] != b'"' {
            return self.fail("a quoted string belongs here");
        }
        let mut held = String::new();
        let mut chars = self.text[self.at + 1..].char_indices();
        loop {
            let Some((offset, held_char)) = chars.next() else {
                return self.fail("a quoted string that never ends");
            };
            match held_char {
                '"' => {
                    self.at += 1 + offset + 1;
                    return Ok(held);
                }
                '\\' => {
                    let Some((_, escape)) = chars.next() else {
                        return self.fail("an escape that never ends");
                    };
                    match escape {
                        '\\' => held.push('\\'),
                        '"' => held.push('"'),
                        'n' => held.push('\n'),
                        't' => held.push('\t'),
                        'r' => held.push('\r'),
                        'u' => {
                            let mut digits = String::new();
                            match chars.next() {
                                Some((_, '{')) => {}
                                _ => return self.fail("\\u is written \\u{...}"),
                            }
                            loop {
                                match chars.next() {
                                    Some((_, '}')) => break,
                                    Some((_, digit)) => digits.push(digit),
                                    None => return self.fail("an escape that never ends"),
                                }
                            }
                            let Ok(code) = u32::from_str_radix(&digits, 16) else {
                                return self.fail(format!("{digits:?} is no character number"));
                            };
                            let Some(held_char) = char::from_u32(code) else {
                                return self.fail(format!("{code} is no character"));
                            };
                            held.push(held_char);
                        }
                        other => return self.fail(format!("\\{other} is no escape")),
                    }
                }
                other => held.push(other),
            }
        }
    }

    pub fn open(&mut self) -> Result<()> {
        self.skip();
        self.token = self.at;
        match self.ahead() {
            Ahead::Open => {
                self.at += 1;
                Ok(())
            }
            _ => self.fail("`[` belongs here"),
        }
    }

    pub fn close(&mut self) -> Result<()> {
        self.skip();
        self.token = self.at;
        match self.ahead() {
            Ahead::Close => {
                self.at += 1;
                Ok(())
            }
            _ => self.fail("`]` belongs here"),
        }
    }

    pub fn at_close(&mut self) -> bool {
        matches!(self.ahead(), Ahead::Close)
    }

    /// A bracketed list of whatever `each` reads.
    pub fn list<T>(
        &mut self,
        mut each: impl FnMut(&mut Reader<'a>) -> Result<T>,
    ) -> Result<Vec<T>> {
        self.open()?;
        let mut held = Vec::new();
        while !self.at_close() {
            held.push(each(self)?);
        }
        self.close()?;
        Ok(held)
    }

    /// A word carrying an id under its sigil, as `%3` carries three.
    pub fn tagged(&mut self, sigil: char, what: &str) -> Result<usize> {
        let held = self.word()?;
        let Some(digits) = held.strip_prefix(sigil) else {
            return self.fail(format!(
                "a {what} is written {sigil}<number> and this says {held:?}"
            ));
        };
        match digits.parse() {
            Ok(index) => Ok(index),
            Err(_) => self.fail(format!("{digits:?} is no {what} number")),
        }
    }

    /// Where the token the reader last handed out began.
    ///
    /// A reference is refused once the whole body has been read — an id can
    /// only be out of range against a count that is not known yet — so what
    /// says where it was is carried with it.
    pub fn taken(&self) -> (usize, usize) {
        self.position(self.token)
    }

    pub fn number<T: std::str::FromStr>(&mut self, what: &str) -> Result<T> {
        let held = self.word()?;
        match held.parse() {
            Ok(value) => Ok(value),
            Err(_) => self.fail(format!("{held:?} is no {what}")),
        }
    }

    /// A `0x`-prefixed run of hexadecimal digits, exactly `digits` of them.
    pub fn bits(&mut self, digits: usize) -> Result<u64> {
        let held = self.word()?;
        let Some(hex) = held.strip_prefix("0x") else {
            return self.fail(format!("bits are written 0x… and this says {held:?}"));
        };
        if hex.len() != digits {
            return self.fail(format!(
                "{digits} hexadecimal digits belong here and this has {}",
                hex.len()
            ));
        }
        match u64::from_str_radix(hex, 16) {
            Ok(bits) => Ok(bits),
            Err(_) => self.fail(format!("{hex:?} is no hexadecimal number")),
        }
    }

    /// Raw bytes, as a quoted run of hexadecimal digit pairs.
    pub fn blob(&mut self) -> Result<Vec<u8>> {
        let held = self.quoted()?;
        if !held.len().is_multiple_of(2) {
            return self.fail("raw bytes are written two hexadecimal digits each");
        }
        let mut bytes = Vec::with_capacity(held.len() / 2);
        for pair in held.as_bytes().chunks(2) {
            let pair = std::str::from_utf8(pair).unwrap_or("");
            match u8::from_str_radix(pair, 16) {
                Ok(byte) => bytes.push(byte),
                Err(_) => return self.fail(format!("{pair:?} is no byte")),
            }
        }
        Ok(bytes)
    }
}

/// What stands next in the text.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Ahead<'a> {
    Word(&'a str),
    Text,
    Open,
    Close,
    End,
}

/// One string as the text says it, so that reading it back gives exactly the
/// string that was written — a quote, a backslash and every control character
/// included.
pub fn escaped(value: &str) -> String {
    let mut held = String::with_capacity(value.len() + 2);
    held.push('"');
    for character in value.chars() {
        match character {
            '"' => held.push_str("\\\""),
            '\\' => held.push_str("\\\\"),
            '\n' => held.push_str("\\n"),
            '\t' => held.push_str("\\t"),
            '\r' => held.push_str("\\r"),
            other if (other as u32) < 0x20 || other as u32 == 0x7f => {
                held.push_str(&format!("\\u{{{:x}}}", other as u32));
            }
            other => held.push(other),
        }
    }
    held.push('"');
    held
}

/// Raw bytes as hexadecimal digit pairs, which say the same thing whichever way
/// a machine orders them.
pub fn hex(bytes: &[u8]) -> String {
    let mut held = String::with_capacity(bytes.len() * 2 + 2);
    held.push('"');
    for byte in bytes {
        held.push_str(&format!("{byte:02x}"));
    }
    held.push('"');
    held
}

/// One spelling per variant, both ways. The match is exhaustive, so a variant
/// added to the language fails to compile until it is spelled here; the table
/// beside it is what reads that spelling back.
macro_rules! spelled {
    ($name:ident, $named:ident, $all:ident, $ty:ty, $what:literal, [$($variant:pat => $word:literal),+ $(,)?]) => {
        pub fn $name(value: $ty) -> &'static str {
            match value {
                $($variant => $word),+
            }
        }

        pub fn $named(word: &str) -> Option<$ty> {
            $all.iter().find(|held| $name(**held) == word).copied()
        }
    };
}

pub const ALL_SCALARS: &[Scalar] = &[
    Scalar::Pred,
    Scalar::S8,
    Scalar::U8,
    Scalar::S16,
    Scalar::U16,
    Scalar::S32,
    Scalar::U32,
    Scalar::S64,
    Scalar::U64,
    Scalar::F16,
    Scalar::BF16,
    Scalar::F32,
    Scalar::F64,
];

spelled!(scalar_name, scalar_named, ALL_SCALARS, Scalar, "scalar", [
    Scalar::Pred => "pred",
    Scalar::S8 => "s8",
    Scalar::U8 => "u8",
    Scalar::S16 => "s16",
    Scalar::U16 => "u16",
    Scalar::S32 => "s32",
    Scalar::U32 => "u32",
    Scalar::S64 => "s64",
    Scalar::U64 => "u64",
    Scalar::F16 => "f16",
    Scalar::BF16 => "bf16",
    Scalar::F32 => "f32",
    Scalar::F64 => "f64",
]);

pub const ALL_SPACES: &[Space] = &[Space::Global, Space::Shared];

spelled!(space_name, space_named, ALL_SPACES, Space, "space", [
    Space::Global => "global",
    Space::Shared => "shared",
]);

pub const ALL_EVICTS: &[Evict] = &[Evict::Normal, Evict::First, Evict::Last];

spelled!(evict_name, evict_named, ALL_EVICTS, Evict, "eviction priority", [
    Evict::Normal => "normal",
    Evict::First => "first",
    Evict::Last => "last",
]);

pub const ALL_SREGS: &[Sreg] = &[Sreg::TidX, Sreg::CtaidX, Sreg::CtaidY, Sreg::NtidX];

spelled!(sreg_name, sreg_named, ALL_SREGS, Sreg, "special register", [
    Sreg::TidX => "tid.x",
    Sreg::CtaidX => "ctaid.x",
    Sreg::CtaidY => "ctaid.y",
    Sreg::NtidX => "ntid.x",
]);

pub const ALL_UNARY: &[UnOp] = &[
    UnOp::Neg,
    UnOp::Not,
    UnOp::Sqrt,
    UnOp::Exp2,
    UnOp::Log2,
    UnOp::Sin,
    UnOp::Cos,
    UnOp::Tanh,
    UnOp::Floor,
    UnOp::Ceil,
    UnOp::RoundEven,
    UnOp::LeadingZeros,
    UnOp::OneBits,
];

spelled!(unary_name, unary_named, ALL_UNARY, UnOp, "unary operator", [
    UnOp::Neg => "neg",
    UnOp::Not => "not",
    UnOp::Sqrt => "sqrt",
    UnOp::Exp2 => "exp2",
    UnOp::Log2 => "log2",
    UnOp::Sin => "sin",
    UnOp::Cos => "cos",
    UnOp::Tanh => "tanh",
    UnOp::Floor => "floor",
    UnOp::Ceil => "ceil",
    UnOp::RoundEven => "round_even",
    UnOp::LeadingZeros => "leading_zeros",
    UnOp::OneBits => "one_bits",
]);

pub const ALL_BINARY: &[BinOp] = &[
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::Div,
    BinOp::Rem,
    BinOp::Min,
    BinOp::Max,
    BinOp::And,
    BinOp::Or,
    BinOp::Xor,
    BinOp::Shl,
    BinOp::Shr,
];

spelled!(binary_name, binary_named, ALL_BINARY, BinOp, "binary operator", [
    BinOp::Add => "add",
    BinOp::Sub => "sub",
    BinOp::Mul => "mul",
    BinOp::Div => "div",
    BinOp::Rem => "rem",
    BinOp::Min => "min",
    BinOp::Max => "max",
    BinOp::And => "and",
    BinOp::Or => "or",
    BinOp::Xor => "xor",
    BinOp::Shl => "shl",
    BinOp::Shr => "shr",
]);

pub const ALL_COMPARISONS: &[CmpOp] = &[
    CmpOp::Eq,
    CmpOp::Ne,
    CmpOp::Lt,
    CmpOp::Le,
    CmpOp::Gt,
    CmpOp::Ge,
];

spelled!(comparison_name, comparison_named, ALL_COMPARISONS, CmpOp, "comparison", [
    CmpOp::Eq => "eq",
    CmpOp::Ne => "ne",
    CmpOp::Lt => "lt",
    CmpOp::Le => "le",
    CmpOp::Gt => "gt",
    CmpOp::Ge => "ge",
]);

pub const ALL_WRITE_MODES: &[WriteMode] = &[WriteMode::Set, WriteMode::Add];

spelled!(write_mode_name, write_mode_named, ALL_WRITE_MODES, WriteMode, "write mode", [
    WriteMode::Set => "set",
    WriteMode::Add => "add",
]);

pub const ALL_TRIPS: &[Trip] = &[Trip::PerThread, Trip::BlockUniform];

spelled!(trip_name, trip_named, ALL_TRIPS, Trip, "trip count", [
    Trip::PerThread => "per-thread",
    Trip::BlockUniform => "block-uniform",
]);

pub const ALL_SYNCS: &[Sync] = &[Sync::Warp, Sync::Block];

spelled!(sync_name, sync_named, ALL_SYNCS, Sync, "synchronisation", [
    Sync::Warp => "warp",
    Sync::Block => "block",
]);
