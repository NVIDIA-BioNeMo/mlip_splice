// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The boundary between a recipe's structure and the computations it runs.
//!
//! A few small contracts are everything the structural core asks of a
//! computation family: what the byte extents of a root recipe's ports are
//! ([`BytePortContract`]), how that contract states its own identity
//! ([`Identify`]), where a call reaches into the buffers around it
//! ([`Addresses`]), and how the family's own data merges when two recipes
//! become one ([`Computation`]). Nothing here can look at a representation,
//! because nothing here is given one to look at.
//!
//! What a call *does* to those buffers is not here: it is
//! [`crate::algebra::Effects`], stated at the placement, because it is a fact
//! about the recipe rather than about the representation. Launch geometry is
//! not here either, for the same reason — the recipe decides how work is
//! launched, and [`Geometry`] is where it says so.

use std::fmt::{Debug, Display};

use crate::buffer::{Address, BufRef};
use crate::effects::Effects;

/// The ports a root recipe is bound through, as execution sees them: how many
/// there are and how many bytes each holds.
///
/// A family may state far more than this — names, element types, shapes, roles
/// — and keeps all of it. This is the projection the structural core and the
/// runtime need, taken from the contract rather than stored a second time
/// beside it.
pub trait BytePortContract {
    fn num_inputs(&self) -> usize;
    fn num_outputs(&self) -> usize;
    fn input_nbytes(&self, port: usize) -> Option<usize>;
    fn output_nbytes(&self, port: usize) -> Option<usize>;
    fn aliased_output(&self, _input: usize) -> Option<usize> {
        None
    }
    fn aliases(&self) -> crate::PortAliases {
        let aliases = (0..self.num_inputs())
            .filter_map(|input| self.aliased_output(input).map(|output| (input, output)));
        crate::PortAliases::new(aliases, self.num_inputs(), self.num_outputs())
            .expect("a contract's aliases are valid")
    }
}

/// Where a stable identity walk writes: framed fields, in an order the value
/// fixes.
///
/// Each field arrives length- or width-framed, so no two different values can
/// agree by re-cutting one field into the next. Which hash the fields end up in
/// is the caller's; what a value *is* for identity purposes is the value's.
pub trait IdentityWalk {
    fn byte(&mut self, value: u8);
    fn count(&mut self, value: usize);
    fn number(&mut self, value: u64);
    fn bytes(&mut self, value: &[u8]);
}

/// A value that states its own identity, field by field.
///
/// A contract states everything that makes two of them different — extents, and
/// whatever else the family holds — so that two recipes with differently typed
/// contracts cannot share one identity merely because their byte extents agree.
pub trait Identify {
    fn identify(&self, walk: &mut dyn IdentityWalk);
}

/// The shape one launch runs in: how many blocks, how many threads each, and
/// how much memory a block asks for beyond what it declares statically.
///
/// This is the recipe's, not the family's. A recipe is the authority on how the
/// work it holds is launched, so the core keeps the geometry of every launch a
/// call states — one for a kernel, one each for a composite's — where it can
/// prove things about them, and every transformation carries them across
/// unchanged. How many there are is the call's, which the family answers with
/// [`Computation::launch_arity`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Geometry {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub dynamic_shmem: u32,
}

impl Geometry {
    /// One block of one thread, asking for nothing.
    pub fn single() -> Geometry {
        Geometry {
            grid: [1, 1, 1],
            block: [1, 1, 1],
            dynamic_shmem: 0,
        }
    }

    pub fn new(grid: [u32; 3], block: [u32; 3], dynamic_shmem: u32) -> Geometry {
        Geometry {
            grid,
            block,
            dynamic_shmem,
        }
    }

    /// Whether this launches anything. A zero extent in any dimension launches
    /// no threads at all, which is not a smaller launch but an absent one, so a
    /// recipe states it as work it does not have.
    pub fn launches(&self) -> bool {
        self.grid
            .iter()
            .chain(&self.block)
            .all(|&extent| extent > 0)
    }
}

/// How many blocks of `per_block` items it takes to cover `total`, as a
/// dimension of [`Geometry::grid`].
///
/// A launch grid is `u32`, so a count that does not fit is an error rather than
/// a truncation. An item count of zero or less still takes one block, because
/// the result is a grid dimension and a zero one is not a launch.
pub fn count_covering_blocks(total: i64, per_block: i32) -> Result<u32, String> {
    if per_block <= 0 {
        return Err(format!("grid divisor must be positive, got {per_block}"));
    }
    let blocks = (total.max(0) as u64).div_ceil(per_block as u64).max(1);
    u32::try_from(blocks).map_err(|_| format!("grid dimension {blocks} does not fit in u32"))
}

/// Where a call reaches into the buffers of the body that holds it.
///
/// The core proves each address lies inside the buffer it names, and rewrites
/// addresses when the body lands somewhere else. How far past an address the
/// computation then reaches is the family's business, and what it *does* there
/// is the placement's [`crate::algebra::Effects`] — never inferred from an
/// address, because being handed an address is not touching what it points at.
pub trait Addresses {
    /// Every address this call is handed, in the order it states them.
    fn addresses(&self, visit: &mut dyn FnMut(Address<BufRef>));

    /// The same addresses, each replaced by where it now points.
    fn rebind(&mut self, bind: &mut dyn FnMut(Address<BufRef>) -> Address<BufRef>);
}

/// One computation family: what a recipe carries on its behalf, and what the
/// family does when two recipes are merged into one.
///
/// The core stores these values and never reads inside them. What makes that
/// possible is one rule about where handles live: **a body context names root
/// data, and a call names only the data of the body that holds it.** Every
/// handle a merge renumbers is therefore reached by exactly one of
/// [`Computation::rebase_body_context`] and [`Computation::rebase_call`], and
/// the core knows which bodies each applies to without knowing what a handle
/// is. A candidate that keeps its own context keeps its own call handles too;
/// only the body whose context was merged has its calls rebased.
///
/// A merge is required to be *stable*: whatever was already in the context it
/// merges into still sits where it sat, and still means what it meant. Only the
/// incoming handles move, and the returned remap is where they moved to. The
/// core relies on this everywhere it merges into a value that other work
/// already names — a builder taking in one candidate after another, a
/// substitution landing in a recipe whose other calls are untouched — and it
/// has no way to check it, so it is stated here and each family is tested for
/// it.
///
/// The family itself is a tag rather than a value — it is never constructed —
/// so its `Clone`/`Eq` bounds cost nothing and let the recipe types derive
/// theirs.
pub trait Computation: Clone + Debug + PartialEq + Eq {
    /// What a mapping has to present to produce this family's values.
    ///
    /// Structure is anyone's to transform and computations are not: a family
    /// whose values mean something — a contract that names typed buffers, a
    /// call that was proved against them — decides who may make one. It says
    /// so with a type nobody else can hold a value of, and
    /// [`crate::algebra::Mapping`] must hand one over to map into it.
    ///
    /// A family whose values anyone may make says `()`, which every mapping
    /// already has. Nothing here bounds this type: a bound is a way to make
    /// one without being given one.
    type Authorization;

    /// The external contract a root recipe states.
    type Contract: BytePortContract + Identify + Clone + Debug + PartialEq + Eq;

    /// Data one recipe holds once, for every body below it.
    type RootContext: Clone + Debug + Default + PartialEq + Eq;

    /// Data one body holds, for its own calls. Every body has one: the root's
    /// and each candidate's are separate values.
    type BodyContext: Clone + Debug + Default + PartialEq + Eq;

    /// One call, with the addresses it is handed.
    type Call: Addresses + Clone + Debug + PartialEq + Eq;

    /// Where an incoming recipe's root handles land after a merge.
    type RootRemap;

    /// Where an incoming body's own handles land after a merge.
    type BodyRemap;

    /// What this family refuses. The core reports it at the node or context it
    /// was refused for, and states nothing about it itself.
    type Refusal: Clone + Debug + Display + PartialEq + Eq;

    /// Take `incoming`'s root data into `into`, answering where each incoming
    /// handle now lives.
    ///
    /// The family decides what merging means — appending, interning equal
    /// entries onto one — and the answer it returns is what the core applies to
    /// everything that came with `incoming`. Handles already in `into` do not
    /// move.
    fn merge_root_context(
        into: &mut Self::RootContext,
        incoming: Self::RootContext,
    ) -> Self::RootRemap;

    /// The same, for one body's own data. Handles already in `into` do not
    /// move.
    fn merge_body_context(
        into: &mut Self::BodyContext,
        incoming: Self::BodyContext,
    ) -> Self::BodyRemap;

    /// Move the root handles a body context states onto the merged root
    /// context.
    fn rebase_body_context(context: &mut Self::BodyContext, remap: &Self::RootRemap);

    /// Move the body handles a call states onto the merged body context.
    fn rebase_call(call: &mut Self::Call, remap: &Self::BodyRemap);

    /// What the family proves about a recipe's root data on its own.
    fn validate_root_context(context: &Self::RootContext) -> Result<(), Self::Refusal>;

    /// What the family proves about one body's data against the root data it
    /// names.
    fn validate_body_context(
        context: &Self::BodyContext,
        root: &Self::RootContext,
    ) -> Result<(), Self::Refusal>;

    /// What the family proves about one call against the body that holds it.
    fn validate_call(call: &Self::Call, context: &Self::BodyContext) -> Result<(), Self::Refusal>;

    /// How many times one call is launched, and so how many geometries the
    /// placement of it states.
    ///
    /// A kernel is one launch. A composite is as many as it holds, and each of
    /// them is launched in its own shape — which is why the count is the call's
    /// and the shapes are the recipe's.
    fn launch_arity(call: &Self::Call) -> usize;

    /// What the family proves about the shape one of a call's launches runs in.
    ///
    /// The core has already proved there are as many shapes as launches and
    /// that each of them launches something. What is left is what only the
    /// computation knows: the widths it was compiled under, the axes its body
    /// can read.
    fn validate_geometry(
        call: &Self::Call,
        launch: usize,
        geometry: &Geometry,
    ) -> Result<(), Self::Refusal>;
}

/// A family whose calls carry what they do, so a recipe holding them can still
/// be edited by effect after it was built.
///
/// Effects are construction evidence: a placement proves what it does, the
/// proof is spent on the dependency edges it justifies, and the recipe keeps
/// the edges. That is enough to build and to run, and it is all a family whose
/// calls are opaque bytes can offer — nothing recovers a statement from an
/// edge.
///
/// A family that means to *edit* a built recipe needs the statement back:
/// substituting work into a placement, or replacing one computation with
/// another, is only sound if what arrives does no more than what the
/// surrounding edges were derived from. So such a family keeps its calls'
/// effects in its own payload and says so by implementing this. A family that
/// does not implement it cannot be edited that way, which is the honest
/// outcome rather than an edit resting on an assumption.
pub trait DeclaredEffects: Computation {
    /// What this call does to the buffers of the body that holds it.
    fn call_effects(call: &Self::Call) -> Effects;
}

#[cfg(test)]
mod tests {
    use super::count_covering_blocks;

    /// A grid dimension covers the work and fits where it is stored: the last
    /// partial block counts, an empty problem is still one block, and a count
    /// past `u32` is refused rather than truncated into a launch that skips
    /// work.
    #[test]
    fn a_grid_dimension_covers_the_work_and_fits_in_a_launch() {
        assert_eq!(count_covering_blocks(0, 256), Ok(1));
        assert_eq!(count_covering_blocks(-7, 256), Ok(1));
        assert_eq!(count_covering_blocks(256, 256), Ok(1));
        assert_eq!(count_covering_blocks(257, 256), Ok(2));
        assert_eq!(count_covering_blocks(u32::MAX as i64, 1), Ok(u32::MAX));
        assert!(count_covering_blocks(u32::MAX as i64 + 1, 1).is_err());
        assert!(count_covering_blocks(1, 0).is_err());
        assert!(count_covering_blocks(1, -4).is_err());
    }
}
