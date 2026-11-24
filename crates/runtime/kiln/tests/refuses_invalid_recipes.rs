// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A recipe whose body contradicts the ports it states never reaches a device.
//!
//! The program below says its one output is `PORT` bytes and then writes far
//! more into it. Every kernel address in a recipe is formed from the recipe,
//! never from the buffer, so if this ran there would be nothing on the device to
//! notice the buffer ended — the write would simply land somewhere else.
//!
//! There are two ways such a program could arrive, and both are closed here.
//! Assembling one is refused where a recipe comes into being, so no caller ever
//! holds the value; bytes that state it are refused at decode, so no artifact,
//! cache entry or replayed capture can hand one over. Neither refusal needs a
//! GPU, because neither is reached by touching one.

use recipe::{
    Recipe, RecipeBuilder, RecipeError, RecipeSignature, WireError, Writable, deserialize,
    deserialize_shared, serialize, serialize_spilled,
};

/// The output extent the valid program states, and the bytes it writes into it.
///
/// They differ so that the memset length appears exactly once in the artifact —
/// the test below finds it by value and overwrites it, which is the only way to
/// reach a decoder with a program no encoder of ours produces.
const PORT: usize = 4096;
const WRITTEN: usize = 2048;
const OVERRUN: usize = 1_000_000;

/// One output port of `PORT` bytes, and a memset within it: the valid recipe
/// every test here compares against.
fn writes_within_its_only_port() -> Recipe {
    let mut b = RecipeBuilder::new(0, 1);
    b.memset(Writable::Output(0).at(0), 0, WRITTEN);
    b.finish(RecipeSignature::new(Vec::new(), vec![PORT]))
        .expect("finish")
        .into_recipe()
}

/// The same artifact, with the memset length overwritten by one no buffer it
/// binds could hold.
fn bytes_that_write_past_the_port() -> Vec<u8> {
    let mut wire = serialize(&writes_within_its_only_port()).expect("encode");
    let written = (WRITTEN as u64).to_le_bytes();
    let at = wire
        .windows(8)
        .position(|window| window == written)
        .expect("the artifact states the memset length");
    assert!(
        wire[at + 1..].windows(8).all(|window| window != written),
        "the length must occur once for this to patch the field under test"
    );
    wire[at..at + 8].copy_from_slice(&(OVERRUN as u64).to_le_bytes());
    wire
}

fn is_the_overrun(error: &RecipeError) -> bool {
    matches!(
        error,
        RecipeError::AccessPastEnd {
            wanted: OVERRUN,
            capacity: PORT,
            ..
        }
    )
}

/// A program that writes past the port it states is not a recipe: it is refused
/// where one comes into being, so the value never exists to be handed anywhere.
#[test]
fn writing_past_its_own_port_is_not_a_recipe() {
    let mut b = <RecipeBuilder>::new(0, 1);
    b.memset(Writable::Output(0).at(0), 0, OVERRUN);
    let error = b
        .finish(RecipeSignature::new(Vec::new(), vec![PORT]))
        .expect_err("it writes past the port it states");
    assert!(is_the_overrun(&error), "{error}");
}

/// Bytes that state it are refused at decode — so no caller downstream ever
/// holds one. This is the path `kiln_py::load_recipe`, an on-disk artifact
/// cache, and a replayed capture all take.
#[test]
fn encoded_bytes_are_refused_at_decode() {
    match deserialize(&bytes_that_write_past_the_port()) {
        Err(WireError::Invalid(error)) => assert!(is_the_overrun(&error), "{error}"),
        other => panic!("decoding must refuse these bytes, got {other:?}"),
    }
}

/// The zero-copy door refuses them too: a caller cannot reach a `Recipe` by
/// choosing the other decode function.
#[test]
fn the_shared_decode_refuses_them_as_well() {
    let blob = recipe::Blob::from_vec(bytes_that_write_past_the_port());
    assert!(matches!(
        deserialize_shared(blob),
        Err(WireError::Invalid(_))
    ));
}

/// The valid recipe beside it decodes, so the refusal above is about this
/// program and not about the fixture shape or the encoder.
#[test]
fn the_same_program_within_its_port_decodes() {
    let ok = writes_within_its_only_port();
    let decoded = deserialize(&serialize(&ok).expect("encode")).expect("a valid recipe decodes");
    assert_eq!(decoded, ok);
    assert!(
        serialize_spilled(&ok).is_ok(),
        "and so does the spilled door"
    );
}

/// It cannot become pending or resident through the Python entry point either:
/// `load_recipe` decodes, and decoding is where it stops.
#[test]
fn it_cannot_become_a_pending_python_handle() {
    // `load_recipe` is `deserialize_shared` plus a workspace bound and a
    // handle; the decode is the whole of its checking, and it refuses.
    assert!(matches!(
        deserialize_shared(recipe::Blob::from_vec(bytes_that_write_past_the_port())),
        Err(WireError::Invalid(_))
    ));
}

/// Every way to hold a `Recipe` proves it first, so there is no such value to
/// hand a device.
///
/// This is what makes the device-side refusals unnecessary rather than missing.
/// Tuning, realization and the artifact cache all take a `Recipe`, and the only
/// ways to one are the three below — a builder that finishes, parts stated to
/// [`Recipe::stating_edges`], and bytes decoded. Each is given the same program
/// that writes past the port it states, and each refuses it, so nothing
/// downstream is ever handed one to refuse.
#[test]
fn every_way_to_a_recipe_proves_it_first() {
    let overrun = |b: &mut RecipeBuilder| b.memset(Writable::Output(0).at(0), 0, OVERRUN);
    let ports = || RecipeSignature::new(Vec::new(), vec![PORT]);

    // Built.
    let mut built = <RecipeBuilder>::new(0, 1);
    overrun(&mut built);
    assert!(is_the_overrun(&built.finish(ports()).expect_err("built")));

    // Stated. The same body, handed over as parts by a caller that assembled it
    // somewhere else.
    let mut parts = <RecipeBuilder>::new(0, 1);
    overrun(&mut parts);
    let body = parts.finish_fragment().into_parts().1;
    assert!(is_the_overrun(
        &recipe::Recipe::stating_edges(ports(), recipe::CompileRequests::default(), body)
            .expect_err("stated")
    ));

    // Decoded, from bytes no encoder of ours produces.
    match deserialize(&bytes_that_write_past_the_port()) {
        Err(WireError::Invalid(error)) => assert!(is_the_overrun(&error), "{error}"),
        other => panic!("decoding must refuse these bytes, got {other:?}"),
    }
}
