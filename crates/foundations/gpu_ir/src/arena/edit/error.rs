// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Errors produced by checked arena edits.

use std::fmt;

/// Why an edit was refused. Every one of these is a fact about the request, so
/// none of them can be answered by looking at what the program means.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EditError {
    /// An id of another arena, which numbers from zero as this one does.
    Foreign { what: &'static str, index: usize },
    /// An id that named something and no longer names anything.
    Deleted { what: &'static str, index: usize },
    /// A position past what the thing being indexed holds.
    OutOfRange {
        what: &'static str,
        index: usize,
        held: usize,
    },
    /// One thing named twice where each may be named once.
    Twice { what: &'static str, index: usize },
    /// An operand is a step of no block, so it cannot be one here.
    NotAStep { op: usize },
    /// An operation already runs somewhere, and running twice would define what
    /// it produces twice.
    AlreadyAStep { op: usize, block: usize },
    /// An operation still runs, so it is still part of the program.
    Scheduled { op: usize, block: usize },
    /// A block that still runs something, or still hands something back.
    NotEmpty { block: usize },
    /// A value something still reads cannot stop being defined.
    StillRead { value: usize, by: String },
    /// A kind owns a fixed number of regions and was given another.
    RegionCount {
        kind: &'static str,
        held: usize,
        wanted: usize,
    },
    /// A kind that is only ever a step, asked to be an operand, or the reverse.
    Placement { kind: &'static str, placed: bool },
    /// One destination per result, or none at all.
    DestinationCount { results: usize, into: usize },
    /// A block already belongs to a region, and a block belongs to one.
    BlockTaken { block: usize, region: usize },
    /// The entry block's parameters are the kernel's declarations, so it takes
    /// nothing that is not one.
    EntryTakesDeclarations,
    /// One name is one declaration.
    Redeclared { name: String },
    /// A vector of no lanes is no value.
    ZeroLanes,
    /// A trip count belongs to a loop and to nothing else.
    NotALoop { op: usize },
    /// A callee belongs to a call and to nothing else.
    NotACall { op: usize },
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EditError::Foreign { what, index } => {
                write!(f, "{what} {index} belongs to another function's body")
            }
            EditError::Deleted { what, index } => {
                write!(f, "{what} {index} was deleted and names nothing now")
            }
            EditError::OutOfRange { what, index, held } => {
                write!(f, "{what} {index} is named where there are {held}")
            }
            EditError::Twice { what, index } => {
                write!(f, "{what} {index} is named twice where it is named once")
            }
            EditError::NotAStep { op } => write!(
                f,
                "operation {op} is an operand, evaluated wherever it is read, so it is a step \
                 of no block"
            ),
            EditError::AlreadyAStep { op, block } => write!(
                f,
                "operation {op} is already a step of block {block}, and running twice would \
                 define what it produces twice"
            ),
            EditError::Scheduled { op, block } => write!(
                f,
                "operation {op} runs as a step of block {block}: take it out of the block \
                 before deleting it"
            ),
            EditError::NotEmpty { block } => write!(
                f,
                "block {block} still runs something or still hands something back"
            ),
            EditError::StillRead { value, by } => {
                write!(f, "%{value} is still read by {by}")
            }
            EditError::RegionCount { kind, held, wanted } => {
                write!(f, "{kind} owns {wanted} regions and was given {held}")
            }
            EditError::Placement { kind, placed } => write!(
                f,
                "{kind} cannot be {}",
                if *placed {
                    "a step of a block"
                } else {
                    "an operand"
                }
            ),
            EditError::DestinationCount { results, into } => write!(
                f,
                "{into} destinations for {results} results: an operation names one per result \
                 or none at all"
            ),
            EditError::BlockTaken { block, region } => {
                write!(f, "block {block} already belongs to region {region}")
            }
            EditError::EntryTakesDeclarations => f.write_str(
                "the entry block takes exactly what the kernel declares, so a parameter is added \
                 by declaring one",
            ),
            EditError::Redeclared { name } => write!(f, "{name} is already declared here"),
            EditError::ZeroLanes => f.write_str("a value of no lanes holds nothing"),
            EditError::NotALoop { op } => {
                write!(f, "operation {op} is no loop, so it counts no trips")
            }
            EditError::NotACall { op } => {
                write!(f, "operation {op} is no call, so it runs no callee")
            }
        }
    }
}

impl std::error::Error for EditError {}
