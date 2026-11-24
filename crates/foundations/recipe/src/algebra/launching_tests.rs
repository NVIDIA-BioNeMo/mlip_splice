// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a launch says it does, and the edges that follow from it.
//!
//! Every producer of device work states its effects through this surface, so
//! the edges below are the ones every recipe in this workspace is ordered by:
//! a read follows the last write, a write follows the last write and every
//! read since, an input orders nothing, and an address that is merely passed
//! orders nothing either.

use super::*;
use crate::{Arg, Buf, BufRef, RecipeSignature, WorkspaceBuf, Writable};

fn ports() -> RecipeSignature {
    RecipeSignature::new(vec![64], vec![64])
}

fn edges(recipe: &Recipe) -> Vec<Vec<usize>> {
    edges_of(recipe.body())
}

fn edges_of(body: &RecipeBody) -> Vec<Vec<usize>> {
    body.nodes()
        .iter()
        .map(|node| node.deps().iter().copied().collect())
        .collect()
}

/// A read follows the last write; a write follows that write and every read
/// since it. Nothing orders against an input, because nothing inside a body
/// writes one.
#[test]
fn a_launch_orders_against_what_it_says_it_touches() {
    let mut b = RecipeBuilder::<Executable>::new(1, 1);
    let m = b.module(vec![0xAA]);
    let ws = b.scratch(64);
    b.kernel(m, "write").write(ws).add();
    b.kernel(m, "read_a").read(ws).read(Buf::Input(0)).add();
    b.kernel(m, "read_b").read(ws).add();
    b.kernel(m, "rewrite").write(ws).add();
    b.kernel(m, "accumulate").read_write(ws).add();
    let recipe = b.finish(ports()).unwrap().into_recipe();

    assert_eq!(
        edges(&recipe),
        vec![vec![], vec![0], vec![0], vec![0, 1, 2], vec![3],]
    );
}

/// An address a launch is merely handed orders nothing: passing a pointer is
/// not touching the memory behind it, which is what a producer that proved its
/// kernel never reaches there is entitled to state.
#[test]
fn a_passed_address_orders_nothing() {
    let mut b = RecipeBuilder::<Executable>::new(0, 1);
    let m = b.module(vec![0xAA]);
    let ws = b.scratch(64);
    b.kernel(m, "write").write(ws).add();
    b.kernel(m, "carry")
        .pointer(ws)
        .write(Writable::Output(0))
        .add();
    let recipe = b
        .finish(RecipeSignature::new(Vec::new(), vec![64]))
        .unwrap()
        .into_recipe();

    assert_eq!(edges(&recipe), vec![vec![], vec![]]);
}

/// Arguments a producer assembled itself declare the most each address they
/// carry could be for, so work behind them is still ordered — and an input,
/// which nothing writes, still orders nothing.
#[test]
fn assembled_arguments_declare_what_their_addresses_admit() {
    let mut b = RecipeBuilder::<Executable>::new(1, 1);
    let m = b.module(vec![0xAA]);
    let ws = b.scratch(64);
    b.kernel(m, "seed").write(ws).add();
    b.kernel(m, "assembled")
        .args([Arg::workspace(ws.index()), Arg::input(0)])
        .add();
    b.kernel(m, "after").read(ws).add();
    let recipe = b.finish(ports()).unwrap().into_recipe();

    assert_eq!(edges(&recipe), vec![vec![], vec![0], vec![1]]);
}

/// Memory work states its effects by being what it is, and orders against
/// launches the same way.
#[test]
fn memory_work_and_launches_order_against_each_other() {
    let mut b = RecipeBuilder::<Executable>::new(1, 1);
    let m = b.module(vec![0xAA]);
    let ws = b.scratch(64);
    b.memset(Writable::Workspace(ws).at(0), 0, 64);
    b.kernel(m, "consume")
        .read(ws)
        .write(Writable::Output(0))
        .add();
    b.memcpy(Buf::Input(0).at(0), Writable::Workspace(ws).at(0), 64);
    let recipe = b.finish(ports()).unwrap().into_recipe();

    assert_eq!(edges(&recipe), vec![vec![], vec![0], vec![0, 1]]);
}

/// A caller can order otherwise independent work, and an edge discovered twice
/// is one edge.
#[test]
fn a_stated_prerequisite_orders_work_that_touches_nothing_shared() {
    let mut b = RecipeBuilder::<Executable>::new(0, 2);
    let m = b.module(vec![0xAA]);
    let first = b.kernel(m, "left").write(Writable::Output(0)).add();
    b.kernel(m, "right")
        .write(Writable::Output(1))
        .after(&[first])
        .add();
    b.kernel(m, "both")
        .write(Writable::Output(0))
        .after(&[first])
        .add();
    let recipe = b
        .finish(RecipeSignature::new(Vec::new(), vec![64, 64]))
        .unwrap()
        .into_recipe();

    assert_eq!(edges(&recipe), vec![vec![], vec![0], vec![0]]);
}

/// A region that was built states what it does, so the host answers for exactly
/// that, rebound onto its own buffers — and the region's private workspace
/// stays private.
#[test]
fn a_landed_region_is_accounted_for_by_what_it_proved() {
    let region = {
        let mut b = RecipeBuilder::<Executable>::new(1, 1);
        let m = b.module(vec![0xBB]);
        let ws = b.scratch(16);
        b.kernel(m, "inner")
            .read(Buf::Input(0))
            .write(ws)
            .write(Writable::Output(0))
            .add();
        b.finish_fragment()
    };

    let mut host = RecipeBuilder::<Executable>::new(2, 1);
    host.splice(region, &[BufRef::Input(1)], &[BufRef::Output(0)], &[])
        .unwrap();
    let fragment = host.finish_fragment();

    assert_eq!(
        fragment.boundary(),
        Some(&Effects::of([
            (BufRef::Input(1), Access::Read),
            (BufRef::Output(0), Access::Write),
        ])),
        "what the region proved, through the binding that landed it"
    );
}

/// A region nobody proved is accounted for by its whole binding: rebinding
/// sends every slot it names to one of those buffers and its own workspace is
/// private, so they are everything it can reach.
///
/// The landing itself is unchanged by that: the region keeps the order it had
/// and follows only what the caller states.
#[test]
fn an_unproved_landing_answers_for_the_buffers_it_was_bound_to() {
    let region = {
        let mut b = RecipeBuilder::<Executable>::new(1, 1);
        let m = b.module(vec![0xBB]);
        b.kernel(m, "first").read(Buf::Input(0)).add();
        b.kernel(m, "second").write(Writable::Output(0)).add();
        // A recipe keeps the edges and not the statement they came from, so a
        // fragment taken from one arrives unproved.
        b.finish(RecipeSignature::new(vec![64], vec![64]))
            .unwrap()
            .into_recipe()
            .into_fragment()
    };

    let mut host = RecipeBuilder::<Executable>::new(1, 1);
    let m = host.module(vec![0xAA]);
    let before = host.kernel(m, "before").write(Writable::Output(0)).add();
    host.splice(region, &[BufRef::Input(0)], &[BufRef::Output(0)], &[before])
        .unwrap();
    let fragment = host.finish_fragment();

    assert_eq!(
        fragment.boundary(),
        Some(&Effects::of([
            (BufRef::Input(0), Access::Read),
            (BufRef::Output(0), Access::ReadWrite),
        ]))
    );
    assert_eq!(
        edges_of(fragment.body()),
        vec![vec![], vec![0], vec![0]],
        "both of the region's sources follow the caller's prerequisite, and \
         nothing else was derived for them"
    );
}

/// A workspace buffer is the body's own, so it never reaches the boundary a
/// choice over that body would be held to.
#[test]
fn private_workspace_stays_out_of_the_boundary() {
    let mut b = RecipeBuilder::<Executable>::new(0, 0);
    let m = b.module(vec![0xAA]);
    let ws = b.scratch_init(vec![7u8; 8]);
    b.kernel(m, "scratch").read_write(ws).add();
    let fragment = b.finish_fragment();

    assert_eq!(fragment.boundary(), Some(&Effects::none()));
    assert_eq!(
        fragment.body().workspace(),
        &[WorkspaceBuf {
            nbytes: 8,
            init: Some(crate::Blob::from_vec(vec![7u8; 8])),
        }]
    );
}

/// A recipe this build assembled may be a candidate; the same recipe decoded
/// from bytes may not.
///
/// Decoding keeps the dependency edges and not the statement they came from, so
/// there is nothing to hold a decoded candidate to — and a choice that took one
/// anyway would be promising something nobody proved. Work acquired from
/// elsewhere states its boundary explicitly, which is how a captured candidate
/// still reaches the choice around it.
#[test]
fn only_work_someone_vouched_for_can_be_a_candidate() {
    let built = {
        let mut b = RecipeBuilder::<Executable>::new(1, 1);
        let m = b.module(vec![0xAA]);
        b.kernel(m, "candidate")
            .read(Buf::Input(0))
            .write(Writable::Output(0))
            .add();
        b.finish(ports()).unwrap()
    };

    let over = |candidate: RecipeFragment<Executable>| {
        let mut root = RecipeBuilder::<Executable>::new(1, 1);
        root.choice(vec![candidate])
            .inputs([Buf::Input(0)])
            .does([Buf::Input(0)], Access::Read)
            .outputs([Buf::Output(0)])
            .does([Buf::Output(0)], Access::Write)
            .add()
            .map(|_| ())
    };

    assert_eq!(over(built.clone().into_fragment()), Ok(()));

    let decoded = crate::deserialize(&crate::serialize(built.recipe()).unwrap()).unwrap();
    assert_eq!(
        over(decoded.clone().into_fragment()),
        Err(RecipeError::UnprovedFragment),
        "a decoded recipe carries no proof, and silence is not a claim"
    );

    assert_eq!(
        over(
            ProvedRecipe::stating_boundary(
                decoded,
                Effects::of([
                    (BufRef::Input(0), Access::Read),
                    (BufRef::Output(0), Access::Write),
                ]),
            )
            .into_fragment()
        ),
        Ok(()),
        "work whose boundary its acquirer states reaches the choice around it"
    );
}

/// And what it states is what it is held to: a boundary reaching past the
/// choice's own declaration is refused rather than taken on trust.
#[test]
fn a_stated_boundary_is_held_to_the_choice_that_selects_it() {
    let built = {
        let mut b = RecipeBuilder::<Executable>::new(1, 1);
        let m = b.module(vec![0xAA]);
        b.kernel(m, "candidate")
            .read(Buf::Input(0))
            .write(Writable::Output(0))
            .add();
        b.finish(ports()).unwrap()
    };
    let decoded = crate::deserialize(&crate::serialize(built.recipe()).unwrap()).unwrap();

    let mut root = RecipeBuilder::<Executable>::new(1, 1);
    assert_eq!(
        root.choice(vec![
            ProvedRecipe::stating_boundary(
                decoded,
                Effects::of([(BufRef::Output(0), Access::Write)]),
            )
            .into_fragment(),
        ])
        .inputs([Buf::Input(0)])
        .does([Buf::Input(0)], Access::Read)
        .outputs([Buf::Output(0)])
        .add(),
        Err(RecipeError::UndeclaredEffect {
            node: 0,
            kind: crate::BufKind::Output,
            index: 0,
        })
    );
}
