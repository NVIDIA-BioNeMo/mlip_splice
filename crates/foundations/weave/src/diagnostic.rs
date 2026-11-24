// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Severity {
    Note,
    Warning,
    Error,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceSpan {
    pub offset: usize,
    pub length: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    pub span: Option<SourceSpan>,
    pub notes: Vec<String>,
}

impl Diagnostic {
    pub fn new(severity: Severity, message: impl Into<String>) -> Self {
        Self {
            severity,
            message: message.into(),
            span: None,
            notes: Vec::new(),
        }
    }
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(Severity::Error, message)
    }
    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, message)
    }
    pub fn at(mut self, offset: usize, length: usize) -> Self {
        self.span = Some(SourceSpan { offset, length });
        self
    }
    pub fn note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.severity, self.message)?;
        if let Some(span) = &self.span {
            write!(f, " at {}..{}", span.offset, span.offset + span.length)?;
        }
        for note in &self.notes {
            write!(f, "\n  note: {note}")?;
        }
        Ok(())
    }
}
