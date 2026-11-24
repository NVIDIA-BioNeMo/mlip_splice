// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Who owns a value: the builder that bound it, and the scope it was bound in.

use super::super::{Body, Builder, Carry};
use super::{kinds, pointer, scalar, steps};
use crate::op::OpKind;
use crate::{Scalar, Trip, Ty};

/// A value belongs to the arena that defined it, and handing it to another is
/// caught where it is read rather than becoming a real value of the wrong
/// program.
#[test]
#[should_panic(expected = "a value of arena")]
fn a_value_of_another_builder_is_refused() {
    let mut first = Builder::new();
    let mut second = Builder::new();
    let held = scalar(&mut first, "x");
    let other = scalar(&mut second, "y");
    let _ = second.add(held, other);
}

/// A fragment is built by one builder, and appending it elsewhere would place
/// operations of another arena.
#[test]
#[should_panic(expected = "a fragment of another builder")]
fn a_fragment_of_another_builder_is_refused() {
    let mut first = Builder::new();
    let x = scalar(&mut first, "x");
    let (fragment, ()) = first.build_fragment(|b| {
        let doubled = b.add(x.clone(), x.clone());
        let bound = b.bind_value(Ty::scalar(Scalar::F32), doubled);
        b.name_value(&bound, "doubled");
    });
    let mut second = Builder::new();
    second.append_fragment(fragment);
}

/// A fragment hands back what it built, in the order it built it, and appending
/// it is what makes those steps run.
#[test]
fn a_fragment_runs_only_once_it_is_appended() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let (fragment, ()) = b.build_fragment(|b| {
        let doubled = b.add(x.clone(), x.clone());
        let bound = b.bind_value(Ty::scalar(Scalar::F32), doubled);
        b.name_value(&bound, "doubled");
    });
    assert_eq!(fragment.len(), 2);
    assert!(b.is_empty(), "a fragment is not placed where it was built");
    b.append_fragment(fragment);
    assert_eq!(b.owned().len(), 2);
}

/// A fragment holds every operation it built, and runs only some of them: an
/// operand is evaluated wherever it is read, so it is owned without being a
/// step. A caller placing the fragment needs all of them, and a caller reading
/// it as a list of statements needs the steps alone.
#[test]
fn a_fragment_owns_more_than_it_runs() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let (fragment, ()) = b.build_fragment(|b| {
        let doubled = b.add(x.clone(), x.clone());
        let bound = b.bind_value(Ty::scalar(Scalar::F32), doubled);
        b.name_value(&bound, "doubled");
    });
    let steps = fragment.steps(b.function());
    assert_eq!(fragment.len(), 2, "the sum and the binding are both held");
    assert_eq!(steps.len(), 1, "only the binding runs where it stands");
    assert!(matches!(b.function().op(steps[0]).kind, OpKind::Bind));
    b.discard_fragment(fragment);
}

/// Sparing what a survivor reads is settled to a fixpoint, not in one pass.
///
/// Operations offered together are judged together, so one read only by others
/// of them goes with them. But sparing one spares what it reads, and that can
/// be a third of the set, read by nothing else — so a pass that judges each
/// operation once against the set as it stood keeps only the first link of the
/// chain and deletes the rest while something live still reads them.
///
/// Here a discarded fragment builds `deep`, `middle` and `held` in that order,
/// each reading the last, and a step outside the fragment reads `held`. All
/// three must survive: `held` because the step reads it, and the other two
/// because `held` does.
///
/// Two mechanisms enforce that and this fails only if both go: the fixpoint
/// that spares a survivor's operands before deleting anything, and the sweep
/// afterwards that refuses to forget an operation something live still reads.
/// The sweep is what makes the result correct; the fixpoint is what stops the
/// sweep from having to undo the deletions.
#[test]
fn sparing_what_a_survivor_reads_follows_the_whole_chain() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let mut chain = None;
    let (fragment, ()) = b.build_fragment(|b| {
        let deep = b.add(x.clone(), x.clone());
        let middle = b.mul(deep.clone(), x.clone());
        let held = b.add(middle.clone(), x.clone());
        chain = Some((deep, middle, held));
    });
    let (deep, middle, held) = chain.expect("the fragment built them");
    // Read from outside the fragment, so what the fragment holds is still owed
    // to this step whatever the discard says.
    b.bind_value(Ty::scalar(Scalar::F32), held.clone());
    let at = |value: &crate::build::Value, b: &Builder| {
        b.function()
            .operand_op(value.id_in(b.function()))
            .expect("an operand")
    };
    let (deep, middle, held) = (at(&deep, &b), at(&middle, &b), at(&held, &b));

    b.discard_fragment(fragment);

    let live = b.function().live_ops();
    assert!(live.contains(&held), "the step outside reads it");
    assert!(live.contains(&middle), "and what it reads in turn");
    assert!(
        live.contains(&deep),
        "and what that reads, to the end of the chain"
    );
}

/// A fragment holds operations no block lists yet, so one neither placed nor
/// discarded would strand them. Letting it go is refused rather than silent.
#[test]
#[should_panic(expected = "neither placed nor discarded")]
fn a_fragment_let_go_of_is_refused() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let (fragment, ()) = b.build_fragment(|b| {
        b.bind_value(Ty::scalar(Scalar::F32), x.clone());
    });
    drop(fragment);
}

/// Discarding one says what letting it go cannot: the operations it built are
/// no part of the program.
#[test]
fn a_discarded_fragment_leaves_nothing_behind() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let (fragment, ()) = b.build_fragment(|b| {
        let doubled = b.add(x.clone(), x.clone());
        b.bind_value(Ty::scalar(Scalar::F32), doubled);
    });
    let built = fragment.len();
    let live = b.function().live_ops().len();
    b.discard_fragment(fragment);
    assert_eq!(
        b.function().live_ops().len(),
        live - built,
        "everything it built is gone"
    );
}

/// Slicing a body states which steps survive. The operands stay owned: they are
/// evaluated where they are read, so a slice that dropped them would leave them
/// in no block at all, and the program would name values nothing holds.
#[test]
fn slicing_a_body_keeps_the_operands_it_no_longer_names() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let doubled = b.add(x.clone(), x.clone());
    let first = b.bind_value(Ty::scalar(Scalar::F32), doubled);
    let tripled = b.add(x.clone(), first.clone());
    let second = b.bind_value(Ty::scalar(Scalar::F32), tripled);
    let kept = b.defining_op(&second).expect("the second binding");
    b.replace_placed(vec![kept]);
    let held = b.owned().to_vec();
    assert!(held.contains(&kept), "the slice runs the step it named");
    // `first` is read by the expression the kept step binds, so it stays: a
    // slice states which steps run, and what they read runs with them.
    let earlier = b.defining_op(&first).expect("the first binding");
    assert!(
        held.contains(&earlier),
        "a step the kept one reads stays whatever the slice named"
    );
    let read = b.function().op(kept).operands[0];
    let defining = b
        .function()
        .operand_op(read)
        .expect("an operand defines it");
    assert!(
        held.contains(&defining),
        "the operand the surviving step reads is still held"
    );
}

/// A slice keeps the operands its steps reach, and only those: one read solely
/// by a step the slice dropped would go on reading what that step defined.
#[test]
fn slicing_a_body_drops_the_operands_nothing_reaches() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let doubled = b.add(x.clone(), x.clone());
    let first = b.bind_value(Ty::scalar(Scalar::F32), doubled);
    let scaled = b.add(first.clone(), first);
    let second = b.bind_value(Ty::scalar(Scalar::F32), scaled);
    let dropped = b.defining_op(&second).expect("the second binding");
    let unreached = b
        .function()
        .operand_op(b.function().op(dropped).operands[0])
        .expect("the sum it binds");
    let kept = b.owned()[0];
    b.replace_placed(vec![kept]);
    let held = b.owned().to_vec();
    assert!(!held.contains(&dropped), "the slice did not name that step");
    assert!(
        !held.contains(&unreached),
        "an operand only the dropped step read is dropped with it"
    );
    assert!(
        !b.function().live_ops().contains(&unreached),
        "and it stops being part of the program, rather than belonging to no \
         block"
    );
}

/// A step owning a region reaches what its body reads: an operand built out
/// here and read only inside the region is reached through the step that owns
/// it, exactly as one read by the step itself is.
#[test]
fn a_slice_reaches_through_the_regions_of_the_steps_it_keeps() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let doubled = b.add(x.clone(), x.clone());
    let read_inside = b
        .function()
        .operand_op(doubled.id_in(b.function()))
        .expect("the sum is an operand");
    let pred = b.lt(x.clone(), x);
    b.if_(pred, |arm| {
        arm.bind_value(Ty::scalar(Scalar::F32), doubled.clone());
    });
    let guard = *b.owned().last().expect("the guard");
    b.replace_placed(vec![guard]);
    assert!(
        b.owned().contains(&read_inside),
        "an operand the region reads is reached through the step owning it"
    );
}

/// Steps a slice drops go together, so a chain of them goes whole: one read
/// only by another of the dropped is read by nothing that survives, and
/// judging them one at a time would keep the lot, each held up by the next.
#[test]
fn slicing_a_body_drops_a_chain_of_steps_whole() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let keep = b.bind_value(Ty::scalar(Scalar::F32), x.clone());
    let first = b.bind_value(Ty::scalar(Scalar::F32), x);
    let chained = b.add(first.clone(), first.clone());
    let second = b.bind_value(Ty::scalar(Scalar::F32), chained);
    let head = b.defining_op(&first).expect("the first binding");
    let tail = b.defining_op(&second).expect("the second binding");
    let kept = b.defining_op(&keep).expect("the kept binding");
    b.replace_placed(vec![kept]);
    let live = b.function().live_ops();
    assert!(!live.contains(&tail), "the step nothing reads goes");
    assert!(
        !live.contains(&head),
        "and so does the one only that step read"
    );
}

/// An operand goes where its reader goes. Moving a step into a region and
/// leaving behind what it reads would define that operand outside the region
/// and read it within — across a boundary that closes.
#[test]
fn placing_a_step_carries_the_operands_only_it_reads() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let doubled = b.add(x.clone(), x.clone());
    let bound = b.bind_value(Ty::scalar(Scalar::F32), doubled.clone());
    let carried = b
        .function()
        .operand_op(doubled.id_in(b.function()))
        .expect("the sum is an operand");
    let step = b.defining_op(&bound).expect("the binding");
    let pred = b.lt(x.clone(), x);
    b.if_(pred, |arm| arm.place_all(vec![step]));
    assert!(
        !b.owned().contains(&carried),
        "the operand travelled with the step that reads it"
    );
    let inner = b
        .function()
        .region_block(b.function().op(*b.owned().last().unwrap()).regions[0]);
    assert!(
        b.function().block(inner).ops.contains(&carried),
        "and it is a member of the region now"
    );
}

/// The caller's order is also the tie-breaker among carried operands. The
/// operand DAG only says all three must precede the step; it does not choose
/// their relative order, and that order decides which register each gets.
///
/// This is what an emission that reads the travelling set's order off a rank
/// must preserve, and what sorting by `OpId` instead would silently change:
/// the rendered PTX stays byte-identical either way, because the register
/// allocator renumbers, so nothing but this test says which order ran.
#[test]
fn placing_uses_the_callers_order_for_carried_operands() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let first_value = b.add(x.clone(), x.clone());
    let second_value = b.mul(x.clone(), x);
    let sum = b.add(first_value.clone(), second_value.clone());
    let bound = b.bind_value(Ty::scalar(Scalar::F32), sum.clone());
    let first = b
        .function()
        .operand_op(first_value.id_in(b.function()))
        .unwrap();
    let second = b
        .function()
        .operand_op(second_value.id_in(b.function()))
        .unwrap();
    let sum = b.function().operand_op(sum.id_in(b.function())).unwrap();
    let step = b.defining_op(&bound).unwrap();

    b.place_all([step, second, first, sum]);

    assert_eq!(b.owned(), [second, first, sum, step, second, first, sum]);
}

/// A whole operand chain moves with its reader. Every operation in the chain
/// is read only by the next one, so no intermediate use keeps its input behind.
#[test]
fn placing_a_step_carries_its_whole_operand_chain() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let doubled = b.add(x.clone(), x.clone());
    let quadrupled = b.add(doubled.clone(), doubled.clone());
    let bound = b.bind_value(Ty::scalar(Scalar::F32), quadrupled.clone());
    let first = b
        .function()
        .operand_op(doubled.id_in(b.function()))
        .expect("the first sum is an operand");
    let second = b
        .function()
        .operand_op(quadrupled.id_in(b.function()))
        .expect("the second sum is an operand");
    let step = b.defining_op(&bound).expect("the binding");
    let pred = b.lt(x.clone(), x);
    b.if_(pred, |arm| arm.place_all(vec![step]));
    let inner = b
        .function()
        .region_block(b.function().op(*b.owned().last().unwrap()).regions[0]);
    assert!(b.function().block(inner).ops.contains(&first));
    assert!(b.function().block(inner).ops.contains(&second));
    assert!(!b.owned().contains(&first));
    assert!(!b.owned().contains(&second));
}

/// Placing a region-owning step does not move operands owned by its region a
/// second time. They already belong to the nested block, not to the step's
/// destination block.
#[test]
fn placing_a_guard_does_not_duplicate_its_arm_operands() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let pred = b.lt(x.clone(), x.clone());
    let (fragment, ()) = b.build_fragment(|f| {
        f.if_(pred, |arm| {
            let doubled = arm.add(x.clone(), x.clone());
            arm.bind_value(Ty::scalar(Scalar::F32), doubled);
        });
    });
    let guard = fragment.steps(b.function())[0];
    let block = b.function().region_block(b.function().op(guard).regions[0]);
    let inner = b.function().block(block).ops.to_vec();
    b.append_fragment(fragment);
    assert!(
        inner
            .iter()
            .all(|op| !b.function().block(b.function().entry()).ops.contains(op))
    );
}

/// A shared ancestor stays outside when one user moves into a region and
/// another remains beside it.
#[test]
fn placing_a_step_leaves_a_transitively_shared_operand() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let shared = b.add(x.clone(), x.clone());
    let inside = b.add(shared.clone(), x.clone());
    let moved = b.bind_value(Ty::scalar(Scalar::F32), inside);
    let outside = b.add(shared.clone(), x.clone());
    b.bind_value(Ty::scalar(Scalar::F32), outside);
    let shared = b
        .function()
        .operand_op(shared.id_in(b.function()))
        .expect("the shared sum");
    let step = b.defining_op(&moved).expect("the moved binding");
    let pred = b.lt(x.clone(), x);
    b.if_(pred, |arm| arm.place_all([step]));
    assert!(b.owned().contains(&shared));
    let guard = *b.steps().last().expect("the guard");
    let inner = b.function().region_block(b.function().op(guard).regions[0]);
    assert!(!b.function().block(inner).ops.contains(&shared));
}

/// Moving a step after its original frame has closed removes its old home.
#[test]
fn placing_a_closed_step_moves_it_out_of_its_old_block() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let pred = b.lt(x.clone(), x.clone());
    let mut step = None;
    b.if_(pred.clone(), |arm| {
        let bound = arm.bind_value(Ty::scalar(Scalar::F32), x.clone());
        step = arm.defining_op(&bound);
    });
    let step = step.expect("the binding");
    let first_guard = *b.steps().last().expect("the first guard");
    let old = b
        .function()
        .region_block(b.function().op(first_guard).regions[0]);
    b.if_(pred, |arm| arm.place_all([step]));
    assert!(!b.function().block(old).ops.contains(&step));
    let second_guard = *b.steps().last().expect("the second guard");
    let new = b
        .function()
        .region_block(b.function().op(second_guard).regions[0]);
    assert!(b.function().block(new).ops.contains(&step));
}

/// An operand read from outside stays: it is still owed to whatever reads it
/// there, so only what nothing else reads travels.
#[test]
fn placing_a_step_leaves_an_operand_something_else_reads() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let shared = b.add(x.clone(), x.clone());
    let kept = b.bind_value(Ty::scalar(Scalar::F32), shared.clone());
    let moved = b.bind_value(Ty::scalar(Scalar::F32), shared.clone());
    let operand = b
        .function()
        .operand_op(shared.id_in(b.function()))
        .expect("the sum is an operand");
    let step = b.defining_op(&moved).expect("the second binding");
    let pred = b.lt(x.clone(), x);
    b.if_(pred, |arm| arm.place_all(vec![step]));
    assert!(
        b.owned().contains(&operand),
        "the binding out here still reads it"
    );
    let _ = kept;
}

/// What the arena will not let go of stays held. An operation offered for
/// deletion and spared — because something still reads it — is put back in the
/// frame that offered it: taken out of every list and deleted from none, it
/// would belong to no block.
#[test]
fn a_slice_goes_on_holding_what_the_arena_spares() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let shared = b.bind_value(Ty::scalar(Scalar::F32), x.clone());
    let reader = b.add(shared.clone(), x);
    let kept = b.bind_value(Ty::scalar(Scalar::F32), reader);
    let step = b.defining_op(&kept).expect("the second binding");
    let spared = b.defining_op(&shared).expect("the first binding");
    b.replace_placed(vec![step]);
    assert!(
        b.owned().contains(&spared),
        "it is read by what the slice keeps, so the frame holds it still"
    );
    assert!(b.function().live_ops().contains(&spared));
}

/// The two cheapest questions a body answers about itself, asked of a body
/// built plainly: nothing is stranded, and nothing stands before what it reads.
#[test]
fn a_plainly_built_body_is_well_formed() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let doubled = b.add(x.clone(), x.clone());
    let bound = b.bind_value(Ty::scalar(Scalar::F32), doubled);
    let pred = b.lt(bound.clone(), x);
    b.if_(pred, |arm| {
        let inner = arm.add(bound.clone(), bound.clone());
        arm.bind_value(Ty::scalar(Scalar::F32), inner);
    });
    let f = b.into_function();
    assert_eq!(
        f.stranded(),
        Vec::new(),
        "every operation belongs to a block"
    );
    assert_eq!(
        f.misordered(),
        Vec::new(),
        "each operand stands before its reader"
    );
}

/// `misordered` says what it means: a block whose operand follows its reader
/// is reported, and the reader is named.
#[test]
fn misordering_is_reported_where_it_stands() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let doubled = b.add(x.clone(), x.clone());
    let bound = b.bind_value(Ty::scalar(Scalar::F32), doubled.clone());
    let step = b.defining_op(&bound).expect("the binding");
    let operand = b
        .function()
        .operand_op(doubled.id_in(b.function()))
        .expect("the sum");
    let mut f = b.into_function();
    let entry = f.entry();
    // Put the operand after the step that reads it, which is the shape a
    // careless slice produces.
    f.set_block_ops(entry, vec![step, operand]);
    let found = f.misordered();
    assert_eq!(found.len(), 1, "one block, one offending reader");
    assert_eq!(found[0].1, step, "the reader is named, not the operand");
}

/// A slice writes each operand before whatever reads it, and what that operand
/// reads before that again. A value is readable after the operation defining
/// it, so any other order is not a program — however few rules it happens to
/// break.
#[test]
fn a_slice_writes_an_operand_before_its_reader() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    // Two steps deep on each side, so a naive "steps first, operands after"
    // and a one-level walk both come out wrong.
    let inner = b.add(x.clone(), x.clone());
    let outer = b.mul(inner.clone(), x.clone());
    let first = b.bind_value(Ty::scalar(Scalar::F32), outer);
    let second_inner = b.add(first.clone(), x);
    let second = b.bind_value(Ty::scalar(Scalar::F32), second_inner);
    let steps = [
        b.defining_op(&first).expect("the first binding"),
        b.defining_op(&second).expect("the second binding"),
    ];
    b.replace_placed(steps.to_vec());
    let f = b.into_function();
    assert_eq!(
        f.misordered(),
        Vec::new(),
        "each operand stands before what reads it"
    );
    assert_eq!(
        f.stranded(),
        Vec::new(),
        "and everything belongs to a block"
    );
}

/// A region belongs to the operation listing it, so an operation that stops
/// being part of the program takes its regions with it. One left behind would
/// be owned by nothing, which is not a program.
#[test]
fn forgetting_a_guard_forgets_the_region_it_owned() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let kept = b.bind_value(Ty::scalar(Scalar::F32), x.clone());
    let pred = b.lt(x.clone(), x);
    b.if_(pred, |arm| {
        arm.barrier();
    });
    let guard = *b
        .owned()
        .iter()
        .find(|&&op| matches!(b.function().op(op).kind, OpKind::If))
        .expect("the guard");
    let inner = b.function().region_block(b.function().op(guard).regions[0]);
    let step = b.defining_op(&kept).expect("the binding");
    b.replace_placed(vec![step]);
    let f = b.into_function();
    assert!(
        !f.live_ops().contains(&guard),
        "the slice does not run the guard, and nothing reads it"
    );
    assert!(
        !f.live_blocks().contains(&inner),
        "the block it owned goes with it"
    );
}

/// Placing a fragment and reading back its steps is one act. Taking the two
/// apart is what strands an operand: it would be named by no step list and
/// held by no block.
#[test]
fn placing_a_fragment_reports_the_steps_it_placed() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let (fragment, ()) = b.build_fragment(|b| {
        let doubled = b.add(x.clone(), x.clone());
        let bound = b.bind_value(Ty::scalar(Scalar::F32), doubled);
        b.name_value(&bound, "doubled");
    });
    let steps = b.place_fragment(fragment);
    assert_eq!(steps.len(), 1, "the binding is the step");
    assert!(matches!(b.function().op(steps[0]).kind, OpKind::Bind));
    assert_eq!(b.owned().len(), 2, "the operand it reads is held here too");
}

/// An operation occupies one position in one block, so placing a step built
/// out here into a region moves it: were it listed both places, what it
/// produces would be defined twice.
#[test]
fn placing_a_step_into_a_region_moves_it_there() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let bound = b.bind_value(Ty::scalar(Scalar::F32), x.clone());
    let step = b.defining_op(&bound).expect("the binding");
    assert!(b.owned().contains(&step));
    let pred = b.lt(x.clone(), x);
    b.if_(pred, |arm| arm.place_all(vec![step]));
    assert!(
        !b.owned().contains(&step),
        "the step is a step of the region now, and of nowhere else"
    );
}

/// A value bound inside a region is gone once the region closes: reading it
/// afterwards would name storage the emitted code no longer has.
#[test]
#[should_panic(expected = "was read outside it")]
fn a_value_of_a_closed_region_is_refused() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let mut escaped = None;
    let pred = b.gt(x.clone(), x.clone());
    b.if_(pred, |b| {
        let inner = b.bind_value(Ty::scalar(Scalar::F32), x.clone());
        escaped = Some(inner);
    });
    let _ = b.bind_value(Ty::scalar(Scalar::F32), escaped.unwrap());
}

/// A region reads what encloses it: that is the whole point of nesting one.
#[test]
fn a_region_reads_what_encloses_it() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let outer = b.bind_value(Ty::scalar(Scalar::F32), x.clone());
    let pred = b.gt(x.clone(), outer.clone());
    b.if_(pred, |b| {
        let inner = b.add(outer.clone(), outer.clone());
        let bound = b.bind_value(Ty::scalar(Scalar::F32), inner);
        b.name_value(&bound, "inner");
    });
    let held = b.finish_owned();
    let f = held.function();
    let guard = *steps(&held).last().unwrap();
    let arm = f.region_block(f.op(guard).regions[0]);
    let inner = f.value_named("inner").unwrap();
    assert_eq!(f.block_steps(arm).count(), 1);
    assert!(
        f.reads(f.op(f.defining_op(inner).unwrap()).operands[0])
            .contains(&outer.id_in(f))
    );
}

/// A kernel declares what it takes, not a region of it. Writing a declaration
/// under a guard would still be the kernel's, which is not what writing it
/// there says.
#[test]
#[should_panic(expected = "declared inside a region")]
fn a_parameter_declared_inside_a_region_is_refused() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let pred = b.gt(x.clone(), x.clone());
    b.if_(pred, |b| {
        scalar(b, "late");
    });
}

/// One name is one declaration; a second saying the same thing is still a
/// second, and it is a mistake about what the kernel takes.
#[test]
#[should_panic(expected = "is already a parameter")]
fn a_repeated_declaration_is_refused() {
    let mut b = Builder::new();
    scalar(&mut b, "x");
    scalar(&mut b, "x");
}

/// Resuming a finished body keeps its steps and goes on writing the same
/// arena, so the values it already defined stay readable.
#[test]
fn resuming_a_body_continues_the_arena_it_was_given() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let first = b.bind_value(Ty::scalar(Scalar::F32), x.clone());
    b.name_value(&first, "first");
    let held = b.finish_owned();
    let tag = held.function().tag();

    let mut b = Builder::resuming(held);
    assert_eq!(b.owned().len(), 1, "the steps it was given are still there");
    let second = b.bind_value(Ty::scalar(Scalar::F32), first);
    b.name_value(&second, "second");
    let held = b.finish_owned();
    assert_eq!(held.function().tag(), tag, "the same arena, not a copy");
    assert_eq!(
        super::bound_names(held.function(), held.ops()),
        ["first", "second"]
    );
}

/// A body holds everything the kernel around it declares, so a caller that
/// finishes one knows what it takes without being told separately.
#[test]
fn a_body_holds_what_its_kernel_declares() {
    let mut b = Builder::new();
    pointer(&mut b, "source");
    b.shared_array(crate::SharedArr {
        name: "tile".into(),
        elem: Scalar::F32,
        len: 32,
    });
    scalar(&mut b, "count");
    let held: Body = b.finish_owned();
    let f = held.function();
    assert_eq!(
        f.parameters()
            .iter()
            .map(|param| param.name.clone())
            .collect::<Vec<_>>(),
        ["source", "count"]
    );
    assert_eq!(f.shared_arrays().len(), 1);
    // The declaration order is the order they were made in, shared arrays
    // included, because that is what the entry block's parameters record.
    assert_eq!(f.declarations().len(), 3);
}

/// Slicing the list being built is choosing which steps run; the arena keeps
/// every operation either way.
#[test]
fn replacing_what_is_placed_keeps_the_arena_whole() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let first = b.bind_value(Ty::scalar(Scalar::F32), x.clone());
    let second = b.bind_value(Ty::scalar(Scalar::F32), x);
    b.name_value(&first, "first");
    b.name_value(&second, "second");
    let ops = b.owned().to_vec();
    let values = b.function().num_values();
    b.replace_placed(vec![ops[1]]);
    assert_eq!(b.owned(), [ops[1]]);
    assert_eq!(b.function().num_values(), values);
}

/// A loop body is entered once and closed once; its index and what it carries
/// come from the body, so a body built to be entered one way cannot be placed
/// another.
#[test]
#[should_panic(expected = "a loop body of another builder")]
fn a_loop_body_of_another_builder_is_refused() {
    let mut first = Builder::new();
    let start = first.float(Scalar::F32, 0.0);
    let (body, ()) = first.build_loop_body(
        Scalar::S32,
        vec![Carry {
            ty: Ty::scalar(Scalar::F32),
            init: start,
            into: None,
        }],
        |b, _, given| {
            let one = b.float(Scalar::F32, 1.0);
            (vec![b.add(given[0].clone(), one)], ())
        },
    );
    let mut second = Builder::new();
    let (zero, stop, step) = (second.i32(0), second.i32(4), second.i32(1));
    second.place_counted_loop(Trip::PerThread, zero, stop, step, body);
}

/// A loop the body runs is a step; one standing for the value it reduces to is
/// an operand, evaluated where the expression it was written in is.
#[test]
fn a_loop_is_a_step_or_an_operand_by_how_it_was_placed() {
    let carried = |placed: bool| {
        let mut b = Builder::new();
        let start = b.float(Scalar::F32, 0.0);
        let (body, ()) = b.build_loop_body(
            Scalar::S32,
            vec![Carry {
                ty: Ty::scalar(Scalar::F32),
                init: start,
                into: None,
            }],
            |b, _, given| {
                let one = b.float(Scalar::F32, 1.0);
                (vec![b.add(given[0].clone(), one)], ())
            },
        );
        let (zero, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
        let produced = if placed {
            b.place_counted_loop(Trip::PerThread, zero, stop, step, body)
        } else {
            b.fold_loop(Trip::PerThread, zero, stop, step, body)
        };
        let bound = b.bind_value(
            Ty::scalar(Scalar::F32),
            produced.into_iter().next().unwrap(),
        );
        b.name_value(&bound, "total");
        b.finish_owned()
    };

    let held = carried(true);
    let f = held.function();
    assert!(matches!(
        kinds(f, &held.steps()).as_slice(),
        [OpKind::Loop { .. }, OpKind::Bind]
    ));

    let held = carried(false);
    let f = held.function();
    assert!(matches!(kinds(f, &held.steps()).as_slice(), [OpKind::Bind]));
    let total = f.value_named("total").unwrap();
    let fold = f.op(f.defining_op(total).unwrap()).operands[0];
    assert!(matches!(
        f.op(f.operand_op(fold).unwrap()).kind,
        OpKind::Loop { .. }
    ));
}

/// Leaving a region open is a mistake about which list a step belongs to, and
/// finishing says so rather than handing back half a body.
#[test]
#[should_panic(expected = "a region was left open")]
fn finishing_with_a_region_open_is_refused() {
    let mut b = Builder::new();
    b.open_arm();
    let _ = b.finish_owned();
}
