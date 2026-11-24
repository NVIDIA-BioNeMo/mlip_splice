// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a placement has to prove before it becomes a node.
//!
//! Every way into a recipe states what its work does to the buffers around it,
//! and every one of those statements is checked against what the work is: the
//! addresses it is handed, where a store may land, how many slots a binding
//! fills, whether a launch launches anything, and what the family says about
//! its own data.

use super::structure_fixture::*;
use super::*;
use crate::arg::Arg;
use crate::buffer::{BufKind, BufRef};
use crate::code::{KernelModule, ResourceBounds};
use crate::handles::{Buf, Writable};

/// A store is a store whoever states it: the core proves the buffer accepts one
/// from what the placement declared, without knowing what the computation is.
#[test]
fn a_store_through_a_read_only_buffer_is_refused() {
    assert!(
        storing(BufRef::Output(0).at(0)).is_ok(),
        "an output accepts stores"
    );

    // A store is refused where it is stated, which is where the statement is.
    let mut builder = RecipeBuilder::<Stores>::new(1, 1);
    assert_eq!(
        builder
            .call(
                Store {
                    at: BufRef::Input(0).at(0)
                },
                vec![Geometry::single()]
            )
            .writes([Buf::Input(0)])
            .add(),
        Err(RecipeError::WriteToReadOnlyBuffer {
            node: 0,
            kind: BufKind::Input,
            index: 0
        })
    );

    assert_eq!(
        storing(BufRef::Output(0).at(65)),
        Err(RecipeError::AccessPastEnd {
            node: 0,
            kind: BufKind::Output,
            index: 0,
            wanted: 65,
            capacity: 64
        })
    );
}

/// A launch that declares a store is held to it, exactly as any other family's
/// call is: what a computation does is never read off its argument list.
#[test]
fn an_executable_launch_that_declares_a_store_is_held_to_it() {
    let mut builder = RecipeBuilder::<Executable>::new(1, 1);
    builder
        .body_context_mut()
        .push(KernelModule::from_cuda_loadable(vec![7u8]));
    let builder = builder
        .call(
            launch(0, "loom_a", vec![Arg::input(0)]),
            vec![Geometry::single()],
        )
        .writes([Buf::Input(0)])
        .add();
    assert_eq!(
        builder,
        Err(RecipeError::WriteToReadOnlyBuffer {
            node: 0,
            kind: BufKind::Input,
            index: 0
        })
    );
}

/// A statement is about the placement it is made at, so it may only name
/// buffers that placement is handed. Claiming otherwise would order work
/// against memory the computation never receives.
#[test]
fn a_placement_may_only_claim_what_it_is_handed() {
    let mut builder = RecipeBuilder::<Executable>::new(1, 1);
    builder
        .body_context_mut()
        .push(KernelModule::from_cuda_loadable(vec![7u8]));
    assert_eq!(
        builder
            .call(
                launch(0, "loom_a", vec![Arg::input(0)]),
                vec![Geometry::single()]
            )
            .writes([Buf::Output(0)])
            .add(),
        Err(RecipeError::EffectNotHanded {
            node: 0,
            kind: BufKind::Output,
            index: 0
        })
    );
}

/// A binding decides where landed work writes, so a binding that sends a store
/// to a buffer the host only reads is refused where the landing happens.
///
/// It is the memory operations that carry this, because their destination is
/// writable by type. What a computation does through an address it is handed is
/// not written down in a recipe, so landing one asks nothing about it — the
/// same as it always has.
#[test]
fn a_binding_that_lands_a_store_on_an_input_is_refused() {
    let clearing = {
        let mut builder = RecipeBuilder::<Stores>::new(1, 1);
        builder.memset(Writable::Output(0).at(0), 0, 8);
        builder.finish_fragment()
    };
    let mut builder = RecipeBuilder::<Stores>::new(1, 1);
    let refusal = builder
        .splice(clearing, &[BufRef::Input(0)], &[BufRef::Input(0)], &[])
        .unwrap_err();
    assert_eq!(
        refusal,
        RecipeError::WriteToReadOnlyBuffer {
            node: 0,
            kind: BufKind::Input,
            index: 0
        }
    );

    let landed = builder.finish(ports(64, 64)).unwrap().into_recipe();
    assert!(
        landed.body().nodes().is_empty(),
        "a refused landing adds nothing to the host"
    );
}

/// A binding fills exactly the slots the body it binds has. An unfilled slot
/// has no buffer behind it; a surplus one names a slot nothing can select. A
/// slot the body never touches still has to be filled — it exists.
#[test]
fn a_binding_fills_exactly_the_slots_it_binds() {
    let two_slots = {
        let mut builder = RecipeBuilder::<Stores>::new(2, 2);
        builder
            .call(
                Store {
                    at: BufRef::Output(0).at(0),
                },
                vec![Geometry::single()],
            )
            .writes([Buf::Output(0)])
            .add()
            .unwrap();
        builder.finish_fragment()
    };

    let mut short = RecipeBuilder::<Stores>::new(1, 1);
    assert_eq!(
        short.splice(
            two_slots.clone(),
            &[BufRef::Input(0)],
            &[BufRef::Output(0), BufRef::Output(0)],
            &[]
        ),
        Err(RecipeError::BindingArity {
            node: 0,
            kind: BufKind::Input,
            bound: 1,
            formal: 2
        })
    );

    let mut surplus = RecipeBuilder::<Stores>::new(1, 1);
    assert_eq!(
        surplus.splice(
            two_slots.clone(),
            &[BufRef::Input(0), BufRef::Input(0)],
            &[BufRef::Output(0), BufRef::Output(0), BufRef::Output(0)],
            &[]
        ),
        Err(RecipeError::BindingArity {
            node: 0,
            kind: BufKind::Output,
            bound: 3,
            formal: 2
        })
    );

    // The second input slot is never touched, and still has to be bound.
    let mut exact = RecipeBuilder::<Stores>::new(1, 1);
    assert!(
        exact
            .splice(
                two_slots,
                &[BufRef::Input(0), BufRef::Input(0)],
                &[BufRef::Output(0), BufRef::Output(0)],
                &[]
            )
            .is_ok()
    );
}

/// A choice is one node in its parent's graph, so what its candidates do has to
/// be what it declared. A candidate storing through a slot the choice only
/// carries is refused, because nothing would have been ordered against that
/// buffer's other writers.
#[test]
fn a_candidate_may_not_do_what_the_choice_did_not_declare() {
    let mut builder = RecipeBuilder::<Stores>::new(1, 1);
    let refused = builder
        .choice(vec![storing_fragment(BufRef::Output(0).at(0))])
        .inputs([Buf::Input(0)])
        .does([Buf::Input(0)], Access::Read)
        .outputs([Buf::Output(0)])
        .add();
    assert_eq!(
        refused,
        Err(RecipeError::UndeclaredEffect {
            node: 0,
            kind: BufKind::Output,
            index: 0
        })
    );

    let mut declared = RecipeBuilder::<Stores>::new(1, 1);
    declared
        .choice(vec![storing_fragment(BufRef::Output(0).at(0))])
        .inputs([Buf::Input(0)])
        .does([Buf::Input(0)], Access::Read)
        .outputs([Buf::Output(0)])
        .does([Buf::Output(0)], Access::Write)
        .add()
        .unwrap();
    assert!(declared.finish(ports(64, 64)).is_ok());
}

/// A launch of no threads is work the recipe does not have.
#[test]
fn a_launch_of_no_threads_is_refused() {
    let flat = Work::<Stores>::Call {
        call: Store {
            at: BufRef::Output(0).at(0),
        },
        geometry: vec![Geometry::new([1, 0, 1], [32, 1, 1], 0)],
    };
    assert_eq!(
        Recipe::stating_edges(
            ports(64, 64),
            (),
            RecipeBody::new(1, 1, (), Vec::new(), vec![Node::new(flat, [])]),
        ),
        Err(RecipeError::InvalidGeometry {
            node: 0,
            launch: 0,
            grid: [1, 0, 1],
            block: [32, 1, 1]
        })
    );
}

/// A placement states the shape of each launch its call is, and the family
/// says how many that is. Either way of disagreeing leaves a launch with
/// nowhere to run or a shape with nothing to run in it.
#[test]
fn a_placement_states_one_shape_per_launch() {
    let stating = |geometry: Vec<Geometry>| {
        let work = Work::<Stores>::Call {
            call: Store {
                at: BufRef::Output(0).at(0),
            },
            geometry,
        };
        Recipe::stating_edges(
            ports(64, 64),
            (),
            RecipeBody::new(1, 1, (), Vec::new(), vec![Node::new(work, [])]),
        )
    };
    assert_eq!(
        stating(Vec::new()),
        Err(RecipeError::GeometryArity {
            node: 0,
            stated: 0,
            arity: 1
        })
    );
    assert_eq!(
        stating(vec![Geometry::single(), Geometry::single()]),
        Err(RecipeError::GeometryArity {
            node: 0,
            stated: 2,
            arity: 1
        })
    );
    assert!(stating(vec![Geometry::single()]).is_ok());
}

/// The root body's formal slots are its ports. Two answers to how many there
/// are would make every index below mean two things.
#[test]
fn the_root_body_states_the_arity_its_contract_states() {
    assert_eq!(
        Recipe::<Stores>::stating_edges(
            ports(64, 64),
            (),
            RecipeBody::new(2, 1, (), Vec::new(), vec![])
        ),
        Err(RecipeError::RootArity {
            kind: BufKind::Input,
            stated: 2,
            contract: 1
        })
    );
}

/// What the family owns, the family refuses — and the core says where.
#[test]
fn a_family_refusal_arrives_at_the_place_it_was_refused() {
    let mut builder = RecipeBuilder::<Executable>::new(1, 1);
    builder
        .call(
            launch(4, "loom_a", vec![Arg::output(0)]),
            vec![Geometry::single()],
        )
        .writes([Buf::Output(0)])
        .add()
        .unwrap();
    assert_eq!(
        builder.finish(ports(64, 64)),
        Err(RecipeError::Call {
            node: 0,
            refusal: ExecutableRefusal::ModuleOutOfRange { module: 4 }
        })
    );

    assert_eq!(
        Recipe::<Executable>::stating_edges(
            ports(64, 64),
            CompileRequests::default(),
            RecipeBody::new(
                1,
                1,
                vec![KernelModule::from_request(
                    crate::CompileRequestId(0),
                    Vec::new(),
                    ResourceBounds::default(),
                )],
                Vec::new(),
                Vec::new(),
            ),
        ),
        Err(RecipeError::Context(
            ExecutableRefusal::ModuleRequestOutOfRange {
                module: 0,
                referenced: 0
            }
        ))
    );
}

/// A candidate is proved against the buffer its binding selects, not against
/// ports it restates: it has none.
#[test]
fn a_candidate_is_proved_against_what_the_binding_hands_it() {
    let candidate = |nbytes: usize| {
        let mut builder = RecipeBuilder::<Stores>::new(1, 1);
        builder.memset(Writable::Output(0).at(0), 0, nbytes);
        builder.finish_fragment()
    };

    let build = |nbytes: usize| {
        let mut builder = RecipeBuilder::<Stores>::new(1, 1);
        builder
            .choice(vec![candidate(nbytes)])
            .inputs([Buf::Input(0)])
            .does([Buf::Input(0)], Access::Read)
            .outputs([Buf::Output(0)])
            .does([Buf::Output(0)], Access::Write)
            .add()
            .unwrap();
        builder.finish(ports(64, 32))
    };

    assert!(build(32).is_ok());
    assert_eq!(
        build(33),
        Err(RecipeError::AccessPastEnd {
            node: 0,
            kind: BufKind::Output,
            index: 0,
            wanted: 33,
            capacity: 32
        })
    );
}
