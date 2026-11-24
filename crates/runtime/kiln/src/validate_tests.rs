// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a validation contract must state before any candidate runs.

use super::*;
use recipe::{
    Arg, BufRef, CompileRequests, Geometry, KernelLaunch, KernelModule, Node, RecipeBody,
    RecipeSignature, Work, WorkspaceBuf,
};

struct Agrees;

impl OutputComparator for Agrees {
    fn agrees(&self, _: usize, _: CUdeviceptr, _: CUdeviceptr, _: usize, _: CUstream) -> bool {
        true
    }
}

struct Untouched;

impl OutputInitializer for Untouched {
    fn initialize(&self, _: usize, _: CUdeviceptr, _: usize, _: CUstream) -> Result<(), String> {
        Ok(())
    }
}

fn launch(entry: &str, args: Vec<Arg>) -> Node {
    Node::new(
        Work::Call {
            call: KernelLaunch {
                module: 0,
                entry: entry.into(),
                args,
            },
            geometry: vec![Geometry {
                grid: [1, 1, 1],
                block: [1, 1, 1],
                dynamic_shmem: 0,
            }],
        },
        [],
    )
}

/// One kernel reading input 0 and writing output 1: output 0 is bound but
/// never touched.
fn recipe() -> Recipe {
    Recipe::stating_edges(
        // One input and two outputs, each one extent wide.
        RecipeSignature::new(vec![EXTENT], vec![EXTENT; 2]),
        CompileRequests::default(),
        RecipeBody::new(
            1,
            2,
            vec![KernelModule::from_cuda_loadable(b"image".to_vec())],
            Vec::new(),
            vec![launch("kernel", vec![Arg::input(0), Arg::output(1)])],
        ),
    )
    .unwrap()
}

/// The same program with a workspace the tuner may carve.
fn recipe_with_workspace(nbytes: usize) -> Recipe {
    Recipe::stating_edges(
        RecipeSignature::new(vec![EXTENT], vec![EXTENT; 2]),
        CompileRequests::default(),
        RecipeBody::new(
            1,
            2,
            vec![KernelModule::from_cuda_loadable(b"image".to_vec())],
            vec![WorkspaceBuf { nbytes, init: None }],
            vec![launch("kernel", vec![Arg::input(0), Arg::output(1)])],
        ),
    )
    .unwrap()
}

const EXTENT: usize = 256;
/// The same extent as a device address, for arithmetic on spans.
const STRIDE: CUdeviceptr = EXTENT as CUdeviceptr;

/// Non-overlapping addresses, one page apart, so a test that wants an overlap
/// has to ask for it.
const LIVE_0: CUdeviceptr = 0x1_0000;
const LIVE_1: CUdeviceptr = 0x1_1000;
const REF_0: CUdeviceptr = 0x2_0000;
const REF_1: CUdeviceptr = 0x2_1000;
const CAND_0: CUdeviceptr = 0x3_0000;
const CAND_1: CUdeviceptr = 0x3_1000;
const INPUT_0: CUdeviceptr = 0x4_0000;
const WORKSPACE: CUdeviceptr = 0x5_0000;

fn storage(reference: CUdeviceptr, candidate: CUdeviceptr) -> ValidationOutput {
    ValidationOutput {
        nbytes: EXTENT,
        reference,
        candidate,
    }
}

const INPUT_EXTENTS: [usize; 1] = [EXTENT];

fn contract(outputs: &[ValidationOutput]) -> ValidationContract<'_> {
    ValidationContract {
        outputs,
        input_extents: &INPUT_EXTENTS,
        comparator: &Agrees,
        initializer: &Untouched,
    }
}

const LIVE: [CUdeviceptr; 2] = [LIVE_0, LIVE_1];
const INPUTS: [CUdeviceptr; 1] = [INPUT_0];

fn bindings() -> Bindings<'static> {
    Bindings {
        inputs: &INPUTS,
        outputs: &LIVE,
        workspace: WORKSPACE,
    }
}

/// Storage that is nobody else's: the contract this whole protocol rests on.
fn sound() -> [ValidationOutput; 2] {
    [storage(REF_0, CAND_0), storage(REF_1, CAND_1)]
}

fn rejects(outputs: &[ValidationOutput]) -> bool {
    matches!(
        contract(outputs).covers(&recipe(), &bindings()),
        Err(Error::ValidationContract(_))
    )
}

#[test]
fn disjoint_storage_is_accepted() {
    assert!(contract(&sound()).covers(&recipe(), &bindings()).is_ok());
}

#[test]
fn a_contract_binds_every_output_the_caller_bound() {
    // Covering only the written output leaves kiln nothing to bind for output
    // 0 but the caller's live buffer, which validation must never write.
    assert!(rejects(&[storage(REF_0, CAND_0)]));
}

/// An empty extent is a real extent — an empty array — not a stand-in for one
/// nobody stated. It occupies no address, so it collides with nothing, and
/// there is nothing to initialize or compare. An extent kiln was never told
/// about is a different thing entirely, and fails by count.
#[test]
fn an_empty_extent_is_a_real_extent() {
    let mut empty = sound();
    empty[1].nbytes = 0;
    // Even aimed straight at a live output, an empty buffer takes no byte.
    empty[1].candidate = LIVE_1;
    empty[1].reference = LIVE_0;
    assert!(contract(&empty).covers(&recipe(), &bindings()).is_ok());

    let stream = std::ptr::null_mut();
    assert!(contract(&empty).establish(stream).is_ok());
    assert_eq!(contract(&empty).disagreement(stream), None);
}

/// A comparator that refuses everything still says nothing about an output
/// with no bytes, because it is never asked.
#[test]
fn a_zero_byte_output_is_never_put_to_the_caller() {
    struct Refuses;
    impl OutputComparator for Refuses {
        fn agrees(&self, _: usize, _: CUdeviceptr, _: CUdeviceptr, _: usize, _: CUstream) -> bool {
            false
        }
    }
    struct Fails;
    impl OutputInitializer for Fails {
        fn initialize(
            &self,
            _: usize,
            _: CUdeviceptr,
            _: usize,
            _: CUstream,
        ) -> Result<(), String> {
            Err("no output should reach this".into())
        }
    }
    let empty = [ValidationOutput {
        nbytes: 0,
        reference: REF_0,
        candidate: CAND_0,
    }];
    let hostile = ValidationContract {
        outputs: &empty,
        input_extents: &INPUT_EXTENTS,
        comparator: &Refuses,
        initializer: &Fails,
    };
    let stream = std::ptr::null_mut();
    assert!(hostile.establish(stream).is_ok());
    assert_eq!(hostile.disagreement(stream), None);
}

/// A candidate on a live output is the overwriting this protocol exists to
/// avoid — this is what the caller most plausibly gets wrong.
#[test]
fn a_candidate_may_not_be_a_live_output() {
    let mut aimed = sound();
    aimed[0].candidate = LIVE_0;
    assert!(rejects(&aimed));
    // Any live output, not just this output's own.
    let mut crossed = sound();
    crossed[0].candidate = LIVE_1;
    assert!(rejects(&crossed));
}

/// Tuning writes the live outputs — the warm-up run before any candidate, and
/// every timed candidate after — so a reference kept there is gone before the
/// first comparison.
#[test]
fn a_reference_may_not_be_a_live_output() {
    let mut aimed = sound();
    aimed[1].reference = LIVE_1;
    assert!(rejects(&aimed));
    let mut crossed = sound();
    crossed[1].reference = LIVE_0;
    assert!(rejects(&crossed));
}

/// Overlap, not identity, is what makes storage unusable: a buffer that starts
/// one byte inside a live output is written just as thoroughly.
#[test]
fn partial_overlap_is_rejected_the_same_as_identity() {
    for offset in [1, STRIDE - 1] {
        let mut straddling = sound();
        straddling[0].candidate = LIVE_0 + offset;
        assert!(rejects(&straddling), "candidate at live + {offset}");

        let mut behind = sound();
        behind[0].candidate = LIVE_0 - offset;
        assert!(rejects(&behind), "candidate at live - {offset}");
    }
    // Exactly abutting is not overlapping: the ranges are half-open.
    let mut abutting = sound();
    abutting[0].candidate = LIVE_0 + STRIDE;
    assert!(
        contract(&abutting).covers(&recipe(), &bindings()).is_ok(),
        "a buffer that starts where another ends shares no byte"
    );
}

#[test]
fn a_candidate_may_not_reach_any_reference() {
    let mut own = sound();
    own[0].candidate = own[0].reference;
    assert!(
        rejects(&own),
        "a run compared against itself agrees for free"
    );

    // The other output's reference is just as destroyed.
    let mut crossed = sound();
    crossed[0].candidate = REF_1 + 8;
    assert!(rejects(&crossed));
}

#[test]
fn two_outputs_may_not_share_candidate_storage() {
    let mut shared = sound();
    shared[1].candidate = shared[0].candidate;
    assert!(rejects(&shared));

    let mut straddling = sound();
    straddling[1].candidate = CAND_0 + 16;
    assert!(rejects(&straddling));
}

/// Nothing writes a reference, so two outputs comparing against one buffer is
/// the caller's business.
#[test]
fn two_outputs_may_share_reference_storage() {
    let mut shared = sound();
    shared[1].reference = shared[0].reference;
    assert!(contract(&shared).covers(&recipe(), &bindings()).is_ok());
}

/// The initializer writes every candidate buffer before every run, so an input
/// underneath one is destroyed before the program reads it — whether or not any
/// kernel would have written through that pointer.
#[test]
fn a_candidate_may_not_reach_an_input() {
    let mut aimed = sound();
    aimed[0].candidate = INPUT_0;
    assert!(rejects(&aimed));

    // Partial overlap, which a base-pointer check would miss.
    for offset in [1, STRIDE - 1] {
        let mut straddling = sound();
        straddling[0].candidate = INPUT_0 + offset;
        assert!(rejects(&straddling), "candidate at input + {offset}");
        let mut behind = sound();
        behind[0].candidate = INPUT_0 - offset;
        assert!(rejects(&behind), "candidate at input - {offset}");
    }
    let mut abutting = sound();
    abutting[0].candidate = INPUT_0 + STRIDE;
    assert!(contract(&abutting).covers(&recipe(), &bindings()).is_ok());
}

/// Nothing writes a reference, and kiln reads no input, so a caller comparing
/// an output against an input buffer is stating an identity, not a hazard.
#[test]
fn a_reference_may_sit_on_an_input() {
    let mut identity = sound();
    identity[0].reference = INPUT_0;
    assert!(contract(&identity).covers(&recipe(), &bindings()).is_ok());
}

/// An input whose extent is unstated cannot be kept clear of anything, so a
/// contract that omits one is incomplete rather than permissive.
#[test]
fn every_bound_input_states_an_extent() {
    let outputs = sound();
    for extents in [&[][..], &[EXTENT, EXTENT][..]] {
        let miscounted = ValidationContract {
            outputs: &outputs,
            input_extents: extents,
            comparator: &Agrees,
            initializer: &Untouched,
        };
        assert!(matches!(
            miscounted.covers(&recipe(), &bindings()),
            Err(Error::ValidationContract(_))
        ));
    }
}

/// The tuner carves the workspace for every run — the warm-up, each validation
/// run, and each measurement — so neither a candidate nor a reference may live
/// inside what `workspace_upper_bound` says it may take.
#[test]
fn validation_storage_stays_out_of_the_workspace() {
    let carved = recipe_with_workspace(EXTENT * 4);
    let live =
        |outputs: &[ValidationOutput]| contract(outputs).covers(&carved, &bindings()).is_err();
    let mut candidate_inside = sound();
    candidate_inside[0].candidate = WORKSPACE + STRIDE;
    assert!(live(&candidate_inside));

    let mut reference_inside = sound();
    reference_inside[0].reference = WORKSPACE + STRIDE * 2;
    assert!(
        live(&reference_inside),
        "tuning would overwrite the reference"
    );

    // Straddling the far end still shares bytes with it.
    let mut straddling = sound();
    straddling[0].candidate = WORKSPACE + STRIDE * 4 - 8;
    assert!(live(&straddling));

    // Just past what the recipe may carve is clear.
    let mut past = sound();
    past[0].candidate = WORKSPACE + STRIDE * 4;
    assert!(contract(&past).covers(&carved, &bindings()).is_ok());

    // A recipe with no workspace claims no address at all.
    let mut at_base = sound();
    at_base[0].candidate = WORKSPACE;
    assert!(contract(&at_base).covers(&recipe(), &bindings()).is_ok());
    assert!(
        live(&at_base),
        "the same buffer collides once there is a workspace"
    );
}

/// An extent that runs off the address space would wrap to an end below its
/// start and compare as disjoint from everything.
#[test]
fn an_extent_past_the_end_of_the_address_space_is_rejected() {
    let mut overflowing = sound();
    overflowing[0].candidate = u64::MAX - 8;
    assert!(rejects(&overflowing));

    let mut reference = sound();
    reference[1].reference = u64::MAX - 8;
    assert!(rejects(&reference));

    // An input extent runs off the end just as impossibly.
    let outputs = sound();
    let huge = [usize::MAX];
    let unbounded_input = ValidationContract {
        outputs: &outputs,
        input_extents: &huge,
        comparator: &Agrees,
        initializer: &Untouched,
    };
    assert!(matches!(
        unbounded_input.covers(&recipe(), &bindings()),
        Err(Error::ValidationContract(_))
    ));

    // And so does the workspace, from its own base.
    let at_end = Bindings {
        inputs: &INPUTS,
        outputs: &LIVE,
        workspace: u64::MAX - 8,
    };
    assert!(matches!(
        contract(&sound()).covers(&recipe_with_workspace(EXTENT), &at_end),
        Err(Error::ValidationContract(_))
    ));
}

/// A validation run binds candidate storage in place of the live outputs, in
/// the recipe's own output order.
#[test]
fn candidate_ports_replace_the_live_outputs_in_order() {
    assert_eq!(contract(&sound()).candidate_ports(), vec![CAND_0, CAND_1]);
}

/// An output a choice only passes through to its candidates is still an output
/// the recipe touches, so it still needs covering — the candidate writes it.
#[test]
fn a_choice_binding_counts_as_touching_its_outputs() {
    let candidate = RecipeBody::new(
        0,
        0,
        vec![KernelModule::from_cuda_loadable(b"image".to_vec())],
        Vec::new(),
        vec![launch("nested", vec![Arg::output(0)])],
    );
    let body = RecipeBody::new(
        0,
        0,
        Vec::new(),
        vec![WorkspaceBuf {
            nbytes: 64,
            init: None,
        }],
        vec![Node::new(
            Work::Choice {
                candidates: vec![candidate],
                input_binding: Vec::new(),
                // The candidate's own output 0 reaches this body's output 2;
                // its output 1 goes to workspace and is nobody's reference.
                output_binding: vec![BufRef::Output(2), BufRef::Workspace(0)],
            },
            [],
        )],
    );
    let mut touched = Vec::new();
    collect_touched_outputs(&body, &mut touched);
    assert_eq!(touched, vec![2]);
}

#[test]
fn a_touched_output_beyond_the_contract_is_rejected() {
    let live = [LIVE_0];
    let one_output = Bindings {
        inputs: &INPUTS,
        outputs: &live,
        workspace: 0,
    };
    let outputs = [storage(REF_0, CAND_0)];
    assert!(matches!(
        contract(&outputs).covers(&recipe(), &one_output),
        Err(Error::ValidationContract(_))
    ));
}
