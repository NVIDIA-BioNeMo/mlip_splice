// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::str::FromStr;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DialectName(String);

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OperationName {
    full: String,
    separator: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvalidName(pub String);

impl fmt::Display for InvalidName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid IR name `{}`", self.0)
    }
}
impl std::error::Error for InvalidName {}

fn valid_component(value: &str) -> bool {
    let mut chars = value.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

impl DialectName {
    pub fn new(value: impl Into<String>) -> Result<Self, InvalidName> {
        let value = value.into();
        if valid_component(&value) {
            Ok(Self(value))
        } else {
            Err(InvalidName(value))
        }
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for DialectName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl FromStr for DialectName {
    type Err = InvalidName;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl OperationName {
    pub fn new(value: impl Into<String>) -> Result<Self, InvalidName> {
        let full = value.into();
        let Some(separator) = full.find('.') else {
            return Err(InvalidName(full));
        };
        if full[separator + 1..].contains('.')
            || !valid_component(&full[..separator])
            || !valid_component(&full[separator + 1..])
        {
            return Err(InvalidName(full));
        }
        Ok(Self { full, separator })
    }
    pub fn dialect(&self) -> &str {
        &self.full[..self.separator]
    }
    pub fn mnemonic(&self) -> &str {
        &self.full[self.separator + 1..]
    }
    pub fn as_str(&self) -> &str {
        &self.full
    }
}
impl fmt::Display for OperationName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.full.fmt(f)
    }
}
impl FromStr for OperationName {
    type Err = InvalidName;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}
