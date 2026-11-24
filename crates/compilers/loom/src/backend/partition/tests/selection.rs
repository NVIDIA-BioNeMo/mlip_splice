// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which steps a slice keeps, and how a sum enumerates into terms.

use gpu_ir::{OpKind, Scalar, Ty};

use super::super::terms::{additive_term_spans, cover_terms};
use super::super::*;
use super::support::*;

#[test]
fn slicing_keeps_only_transitive_dependencies() {
    let mut sketch = Sketch::new();
    let two = sketch.f32(2.0);
    sketch.bind("input", two);
    let nine = sketch.f32(9.0);
    sketch.bind("dead", nine);
    let (input, three) = (sketch.get("input"), sketch.f32(3.0));
    let product = sketch.mul(input, three);
    sketch.bind("result", product);
    let sketched = sketch.finish();

    let kept = slice_bindings(
        sketched.function(),
        &sketched.ops(),
        &[sketched.id("result")],
    )
    .unwrap();
    assert_eq!(kept.len(), 2);
    let named = |op: gpu_ir::OpId| {
        sketched
            .function()
            .name(sketched.function().op(op).results[0])
            .unwrap()
            .to_string()
    };
    assert_eq!(named(kept[0]), "input");
    assert_eq!(named(kept[1]), "result");
}

#[test]
fn partial_selections_refold_complete_factored_subtrees() {
    let mut sketch = Sketch::new();
    for name in ["a", "b", "c", "d", "f", "g"] {
        sketch.input(name);
    }
    let (a, b) = (sketch.get("a"), sketch.get("b"));
    let sum = sketch.add(a, b);
    sketch.bind("s1", sum);
    let (c, d) = (sketch.get("c"), sketch.get("d"));
    let sum = sketch.add(c, d);
    sketch.bind("s2", sum);
    let (s1, f) = (sketch.get("s1"), sketch.get("f"));
    let left = sketch.mul(s1, f);
    let (s2, g) = (sketch.get("s2"), sketch.get("g"));
    let right = sketch.mul(s2, g);
    let whole = sketch.add(left, right);
    sketch.bind("e", whole);
    let sketched = sketch.finish();
    let bindings = sketched.bindings();

    let (terms, spans) = additive_term_spans(sketched.function(), sketched.id("e"), &bindings);
    assert_eq!(terms.len(), 4);
    let chosen = cover_terms(&[0, 1, 2], &terms, &spans).unwrap();
    assert_eq!(chosen.len(), 2);
    let read = chosen[0].reads(sketched.function());
    assert!(read.contains(&sketched.id("s1")), "{:?}", chosen[0]);
    assert!(!read.contains(&sketched.id("a")), "{:?}", chosen[0]);
    assert!(cover_terms(&[0, 4], &terms, &spans).is_none());
}

#[test]
fn additive_slicing_distributes_one_sum_side_without_cartesian_expansion() {
    let mut sketch = Sketch::new();
    for name in ["a", "b", "c", "d"] {
        sketch.input(name);
    }
    let (a, b) = (sketch.get("a"), sketch.get("b"));
    let sum = sketch.add(a, b);
    sketch.bind("lhs_sum", sum);
    let (c, d) = (sketch.get("c"), sketch.get("d"));
    let sum = sketch.add(c, d);
    sketch.bind("rhs_sum", sum);
    let (lhs, rhs) = (sketch.get("lhs_sum"), sketch.get("rhs_sum"));
    let product = sketch.mul(lhs, rhs);
    sketch.bind("product", product);
    let sketched = sketch.finish();
    let bindings = sketched.bindings();
    let f = sketched.function();
    let root = sketched.id("product");

    let terms = additive_terms(f, root, &bindings);
    assert_eq!(terms.len(), 2);
    assert!(terms.iter().all(|term| matches!(term, Term::Product(..))));
}

#[test]
fn a_zero_product_folds_without_cancelling_self_division() {
    let mut sketch = Sketch::new();
    let zero = sketch.f32(0.0);
    sketch.bind("x", zero);
    let (x, zero) = (sketch.get("x"), sketch.f32(0.0));
    let product = sketch.mul(x.clone(), zero);
    sketch.bind("zero_product", product);
    let quotient = {
        let x = sketch.get("x");
        let b = sketch.builder();
        b.div(x.clone(), x)
    };
    sketch.bind("self_division", quotient);
    let sketched = sketch.finish();
    let bindings = sketched.bindings();
    let f = sketched.function();

    let zero = additive_terms(f, sketched.id("zero_product"), &bindings);
    let [Term::Held(held)] = &zero[..] else {
        panic!("a zero product folds to the zero it holds")
    };
    let op = f.operand_op(*held).expect("a constant");
    assert!(matches!(
        f.op(op).kind,
        OpKind::FloatConst {
            scalar: Scalar::F32,
            value
        } if value == 0.0
    ));

    let root = sketched.id("self_division");
    let quotient = additive_terms(f, root, &bindings);
    assert!(matches!(&quotient[..], [Term::Held(held)] if *held == root));
}

#[test]
fn cast_self_divisions_are_preserved() {
    for scalar in [
        Scalar::S32,
        Scalar::F16,
        Scalar::BF16,
        Scalar::F32,
        Scalar::F64,
    ] {
        let mut sketch = Sketch::new();
        let x = {
            let b = sketch.builder();
            let zero = if scalar.is_float() {
                b.float(scalar, 0.0)
            } else {
                b.int(scalar, 0)
            };
            b.bind_value_as("x", Ty::scalar(scalar), zero)
        };
        let division = {
            let b = sketch.builder();
            let quotient = b.div(x.clone(), x);
            b.bind_value_as("division", Ty::scalar(scalar), quotient)
        };
        let cast = {
            let b = sketch.builder();
            let converted = b.cast(Scalar::F32, division);
            b.bind_value_as("cast", Ty::scalar(Scalar::F32), converted)
        };
        let sketched = sketch.finish();
        let bindings = Bindings::of(sketched.function(), &sketched.ops()).unwrap();
        let root = cast.id_in(sketched.function());
        let terms = additive_terms(sketched.function(), root, &bindings);
        assert!(matches!(&terms[..], [Term::Held(held)] if *held == root));
    }
}
