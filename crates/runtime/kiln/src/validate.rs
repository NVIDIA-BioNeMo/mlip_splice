// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checking that a candidate computes what the caller says the program
//! computes, before it is allowed to compete on speed.
//!
//! The statement being checked is about the *whole program*, never about one
//! candidate in isolation. A choice may write a workspace slot, may accumulate
//! into an output the rest of the graph also writes, and may carry an output
//! slot it never touches at all — so "what this candidate produced" is not a
//! value a caller can hold a reference for. "What the recipe produces with this
//! candidate selected" is. Validation therefore runs the recipe once per
//! candidate, into caller-owned candidate buffers, and compares those against
//! caller-owned reference buffers.
//!
//! Kiln allocates no device memory and owns no notion of agreement: the
//! reference storage, the candidate storage, the initial state each run starts
//! from, and the comparison are all the caller's.

use cuda_driver::{CUdeviceptr, CUstream, Span};
use recipe::algebra::Launches;
use recipe::{BufRef, KernelLaunch, Recipe, RecipeBody, Segment, Work};

use crate::{Bindings, Error};

/// One output's storage for validation, at that output's own index.
///
/// A recipe declares which outputs it writes, never how large they are — the
/// shapes belong to the caller's program. Comparing buffers of unknown extent
/// would read whatever follows them, so every output states its own.
///
/// `reference` holds what the program is supposed to produce; `candidate` is
/// where one candidate's clean run writes. Both are the caller's, both are at
/// least `nbytes`, and neither is a buffer the caller is still using: a
/// candidate run overwrites `candidate` completely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidationOutput {
    pub nbytes: usize,
    pub reference: CUdeviceptr,
    pub candidate: CUdeviceptr,
}

/// The caller's answer to "do these agree?".
///
/// Kiln cannot own it: agreement is a property of the dtype and the algorithm.
/// Exact bytes settle an integer permutation; a float reduction reordered
/// across blocks needs a tolerance only the caller can justify.
pub trait OutputComparator {
    fn agrees(
        &self,
        output: usize,
        reference: CUdeviceptr,
        candidate: CUdeviceptr,
        nbytes: usize,
        stream: CUstream,
    ) -> bool;
}

/// The caller's answer to "what does a run start from?".
///
/// Kiln cannot own this either. A recipe that memsets its outputs establishes
/// its own initial state; one that accumulates into them does not, and running
/// a second candidate over the first one's result would compare a sum of two
/// candidates against one reference. Only the caller knows which, so the caller
/// puts the candidate buffers into the starting state before every run.
pub trait OutputInitializer {
    fn initialize(
        &self,
        output: usize,
        buffer: CUdeviceptr,
        nbytes: usize,
        stream: CUstream,
    ) -> Result<(), String>;
}

/// What every *measured* candidate starts from, when a program reads what it
/// writes.
///
/// Timing binds the caller's own output buffers and replays the body, so a
/// program that accumulates into one of them measures its own previous results
/// from the second run on — and two candidates timed in sequence do not start
/// from the same state. A caller whose program has such a buffer supplies this;
/// one whose outputs are written before they are read needs none, which is why
/// it is optional and there is no default.
pub trait StartingState {
    /// Put the storage at `buffer` back into the state a run begins from, when
    /// it is storage whose starting state the caller stated.
    ///
    /// *The key is the allocation, not a port index.* A choice binds its
    /// candidates against the parent's ports in its own order and against
    /// slices of the parent's workspace, so position `i` of a nested body's
    /// output array names a different allocation at every level — and asking
    /// for "output `i`" there would restore the wrong buffer, or a workspace
    /// slice that was never the caller's state at all. A pointer is the same
    /// pointer wherever it is bound, which is the only identity that survives
    /// the descent. How many bytes belong to it is the caller's too: it
    /// produced the allocation, and kiln binds it without knowing its extent.
    ///
    /// Storage the caller does not recognise has no state to restore, and
    /// leaving it alone is the answer rather than an error.
    fn restore(&self, buffer: CUdeviceptr, stream: CUstream) -> Result<(), String>;
}

/// Zeroing is not a default kiln may assume, so there is none. A caller whose
/// program does start from zeros says so with this.
pub struct ZeroInitialized;

impl OutputInitializer for ZeroInitialized {
    fn initialize(
        &self,
        _output: usize,
        buffer: CUdeviceptr,
        nbytes: usize,
        stream: CUstream,
    ) -> Result<(), String> {
        cuda_driver::StreamRef::from_raw(stream)
            .memset_d8(buffer, 0, nbytes)
            .map_err(|error| error.to_string())
    }
}

/// Caller-owned storage, initial state, and comparison for candidate
/// validation.
///
/// Both slices are total over the caller's bindings and indexed the way the
/// recipe indexes them — `outputs[i]` is output `i`, `input_extents[j]` is
/// input `j`. Totality is what makes the contract exact: a sparse one would
/// leave kiln binding the caller's live buffer for whatever it omitted, and
/// leave an extent it was never told about outside every disjointness check.
/// An extent kiln does not know is not an extent of zero — it is an incomplete
/// contract, and that fails by count.
#[derive(Clone, Copy)]
pub struct ValidationContract<'a> {
    pub outputs: &'a [ValidationOutput],
    /// How many bytes each bound input occupies, by input index. Kiln reads no
    /// input and writes none, but it must know where they are: a candidate
    /// buffer that lands on one is zeroed by the initializer before the first
    /// run, and the program then reads what validation destroyed.
    pub input_extents: &'a [usize],
    pub comparator: &'a dyn OutputComparator,
    pub initializer: &'a dyn OutputInitializer,
}

impl ValidationContract<'_> {
    /// Whether this contract can be honoured for `recipe` against `bindings`:
    /// it covers every binding the caller made, and its storage is nobody
    /// else's.
    ///
    /// Both halves are checked here rather than separately, because a caller
    /// who ran only one of them would get a contract that looks usable and
    /// silently is not.
    pub fn covers<F: Launches>(
        &self,
        recipe: &Recipe<F>,
        bindings: &Bindings<'_>,
    ) -> Result<(), Error> {
        self.covers_every_binding(recipe, bindings)?;
        self.owns_its_storage(recipe, bindings)
    }

    fn covers_every_binding<F: Launches>(
        &self,
        recipe: &Recipe<F>,
        bindings: &Bindings<'_>,
    ) -> Result<(), Error> {
        if self.outputs.len() != bindings.outputs.len() {
            return Err(Error::ValidationContract(format!(
                "{} validation outputs for {} bound outputs: validation binds every output, \
                 since the only pointer kiln has for an uncovered one is the caller's live buffer",
                self.outputs.len(),
                bindings.outputs.len()
            )));
        }
        if self.input_extents.len() != bindings.inputs.len() {
            return Err(Error::ValidationContract(format!(
                "{} input extents for {} bound inputs: an input whose extent is unstated cannot \
                 be kept clear of validation storage",
                self.input_extents.len(),
                bindings.inputs.len()
            )));
        }
        let mut touched: Vec<usize> = Vec::new();
        collect_touched_outputs(recipe.body(), &mut touched);
        for output in &touched {
            if *output >= self.outputs.len() {
                return Err(Error::ValidationContract(format!(
                    "output {output} is touched by the recipe but not covered"
                )));
            }
        }
        Ok(())
    }

    /// Validation storage belongs to validation alone.
    ///
    /// Two rules, from two reasons. *Nothing tuning writes* may hold validation
    /// storage: the live outputs are written by the warm-up run before any
    /// candidate and by every timed candidate after, and the workspace is
    /// scratch every run carves — so a reference kept in either is gone before
    /// the first comparison, and a candidate kept in either is the overwriting
    /// this protocol exists to avoid. And *nothing a run must not clobber* may
    /// hold a candidate: the caller's inputs, which the program reads, and every
    /// reference, which the comparison reads. A reference may sit on an input,
    /// because nothing writes a reference.
    ///
    /// Every comparison is between ranges: a buffer that starts one byte inside
    /// another is written just as thoroughly as one that starts on it. Extents
    /// come from the contract — the caller's live output `i` and validation's
    /// output `i` are the same output, so one extent describes both — and from
    /// `workspace_upper_bound`, which is what the tuner may carve.
    fn owns_its_storage<F: Launches>(
        &self,
        recipe: &Recipe<F>,
        bindings: &Bindings<'_>,
    ) -> Result<(), Error> {
        let mut candidates = Vec::with_capacity(self.outputs.len());
        let mut references = Vec::with_capacity(self.outputs.len());
        // What tuning writes, and so may hold neither candidate nor reference.
        let mut mutated = Vec::with_capacity(self.outputs.len() + 1);
        // What a candidate run must leave standing.
        let mut readable = Vec::with_capacity(self.input_extents.len());

        for (output, (storage, bound)) in self.outputs.iter().zip(bindings.outputs).enumerate() {
            candidates.push(span(
                storage.candidate,
                storage.nbytes,
                &format!("output {output}'s candidate buffer"),
            )?);
            references.push(span(
                storage.reference,
                storage.nbytes,
                &format!("output {output}'s reference buffer"),
            )?);
            mutated.push((
                format!("the caller's live output {output}"),
                span(
                    *bound,
                    storage.nbytes,
                    &format!("the caller's live output {output}"),
                )?,
            ));
        }
        for (input, (nbytes, bound)) in self.input_extents.iter().zip(bindings.inputs).enumerate() {
            readable.push((
                format!("the caller's input {input}"),
                span(*bound, *nbytes, &format!("the caller's input {input}"))?,
            ));
        }
        mutated.push((
            "the recipe's workspace".to_string(),
            span(
                bindings.workspace,
                recipe::algebra::workspace_upper_bound(recipe),
                "the recipe's workspace",
            )?,
        ));

        let clash = |what: &str, region: &str| {
            Err(Error::ValidationContract(format!(
                "{what} overlaps {region}, so validation would not be a statement about the \
                 program"
            )))
        };
        for (output, candidate) in candidates.iter().enumerate() {
            let what = format!("output {output}'s candidate storage");
            for (region, span) in mutated.iter().chain(&readable) {
                if candidate.overlaps(*span) {
                    return clash(&what, region);
                }
            }
            for (other, against) in references.iter().enumerate() {
                if candidate.overlaps(*against) {
                    return clash(&what, &format!("output {other}'s reference"));
                }
            }
            for (other, against) in candidates.iter().enumerate().skip(output + 1) {
                if candidate.overlaps(*against) {
                    return clash(&what, &format!("output {other}'s candidate storage"));
                }
            }
        }
        // References only have to survive; two of them sharing one buffer is
        // the caller's business, since nothing writes either.
        for (output, reference) in references.iter().enumerate() {
            let what = format!("output {output}'s reference storage");
            for (region, span) in &mutated {
                if reference.overlaps(*span) {
                    return clash(&what, region);
                }
            }
        }
        Ok(())
    }

    /// The candidate buffers, by output index — what a validation run binds in
    /// place of the caller's live outputs.
    pub(crate) fn candidate_ports(&self) -> Vec<CUdeviceptr> {
        self.outputs.iter().map(|output| output.candidate).collect()
    }

    /// Put every candidate buffer into the state a run starts from. A
    /// zero-byte output has no state to establish, so nothing is asked of the
    /// caller for it.
    pub(crate) fn establish(&self, stream: CUstream) -> Result<(), Error> {
        for (output, storage) in self.outputs.iter().enumerate() {
            if storage.nbytes == 0 {
                continue;
            }
            self.initializer
                .initialize(output, storage.candidate, storage.nbytes, stream)
                .map_err(Error::ValidationContract)?;
        }
        Ok(())
    }

    /// The first output whose candidate result the caller does not accept, or
    /// `None` when every one agrees. A zero-byte output holds no bytes to
    /// disagree about, so the caller is not asked.
    pub(crate) fn disagreement(&self, stream: CUstream) -> Option<usize> {
        self.outputs
            .iter()
            .enumerate()
            .filter(|(_, storage)| storage.nbytes > 0)
            .find_map(|(output, storage)| {
                let agrees = self.comparator.agrees(
                    output,
                    storage.reference,
                    storage.candidate,
                    storage.nbytes,
                    stream,
                );
                (!agrees).then_some(output)
            })
    }
}

/// The caller's phrasing of `Span::of`'s one failure: a length that runs past
/// the end of the address space is a contract error, not a wrap to zero.
fn span(start: CUdeviceptr, nbytes: usize, what: &str) -> Result<Span, Error> {
    Span::of(start, nbytes).ok_or_else(|| {
        Error::ValidationContract(format!(
            "{what} spans {nbytes} bytes from {start:#x}, which runs past the end of the \
             address space"
        ))
    })
}

/// What one realization did about validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationStatus {
    NotRequested,
    /// A contract was supplied, and this is what it decided. `rejected` counts
    /// candidates that ran cleanly and disagreed — each one a candidate the
    /// tuner was not allowed to time.
    Checked {
        outputs: usize,
        validated: usize,
        rejected: usize,
    },
}

/// Every output index the body reads or writes, including the ones a choice
/// only passes through to its candidates.
fn collect_touched_outputs<F: Launches>(body: &RecipeBody<F>, touched: &mut Vec<usize>) {
    let mut note = |output: usize| {
        if !touched.contains(&output) {
            touched.push(output);
        }
    };
    for node in body.nodes() {
        match node.work() {
            Work::Call {
                call: KernelLaunch { args, .. },
                ..
            } => {
                for segment in args.iter().flat_map(|arg| arg.segments().iter()) {
                    if let Segment::Pointer(address) = segment
                        && let BufRef::Output(output) = address.buf
                    {
                        note(output);
                    }
                }
            }
            Work::Memset { destination, .. } => {
                if let BufRef::Output(output) = destination.readable().buf {
                    note(output);
                }
            }
            Work::Memcpy {
                source,
                destination,
                ..
            } => {
                for buf in [source.buf, destination.readable().buf] {
                    if let BufRef::Output(output) = buf {
                        note(output);
                    }
                }
            }
            // A candidate writes in its own index space; the binding says which
            // of this body's outputs those writes reach, so the binding is the
            // whole answer and the candidate bodies need no walk.
            Work::Choice {
                input_binding,
                output_binding,
                ..
            } => {
                for buf in input_binding.iter().chain(output_binding) {
                    if let BufRef::Output(output) = buf {
                        note(*output);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "validate_tests.rs"]
mod tests;
