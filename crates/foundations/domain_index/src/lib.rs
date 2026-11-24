// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Domain index: an index map [`Graph`] of iteration domains plus a placement
//! of those domains onto iteration and operand axes.
//!
//! The crate is layered:
//! - the **graph** — [`Domain`]s and the [`IndexMap`] / [`AlignedView`]
//!   arrows between them, with the queries that walk that topology;
//! - the **bridge** — [`AxisBridge`], classifying how one axis reaches its
//!   domain from the iteration domain;
//! - the **index** — the [`DomainIndex`] trait, a placement over a graph and
//!   the problem-level queries it induces;
//! - the **slot layout** — [`OperandGraph`], which answers one of those
//!   queries: which reference arrays a launch must bind for a placement to be
//!   addressable, as an ordered list of [`Slot`]s, and how many bytes each of
//!   them occupies;
//! - the **address algebra** — [`operand_flat_index`] and its neighbours, which
//!   answer the other: where an operand element lives, built in whatever IR an
//!   [`IndexBuilder`] speaks.

mod address;
mod bridge;
pub mod extent;
mod finite;
mod graph;
mod index;
mod slot_extent;
mod slot_layout;

pub use address::{
    AddressSource, IndexBuilder, operand_alive, operand_buffer_offset_over, operand_flat_index,
    operand_flat_index_over, operand_has_alive,
};
pub use bridge::AxisBridge;
pub use finite::{FiniteMap, FiniteMapGraph, FiniteMapId, FiniteSet, FiniteSetId};
pub use graph::{AlignedView, Domain, Graph, IndexMap};
pub use index::DomainIndex;
pub use slot_extent::{
    INDEX_NBYTES, reference_array_nbytes, slot_array_lengths, slot_array_nbytes,
};
pub use slot_layout::{IterPlacement, OperandGraph, Slot, SlotKind};
