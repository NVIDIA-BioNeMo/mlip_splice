// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use num_bigint::{BigInt, BigUint};
use num_rational::BigRational;
use num_traits::One;

use crate::AlgebraicError;
use crate::algebraic_real::*;
use crate::canonical;
use crate::float::round_rational_to_f64;
use crate::representation::Representation;

fn polynomial(values: &[i64], root_index: usize) -> AlgebraicReal {
    AlgebraicReal::from_polynomial(
        values
            .iter()
            .map(|&value| BigRational::from_integer(value.into()))
            .collect(),
        root_index,
    )
    .unwrap()
}

#[test]
fn constructs_every_real_root_in_order() {
    let negative = polynomial(&[0, -1, 0, 1], 0);
    let zero = polynomial(&[0, -1, 0, 1], 1);
    let positive = polynomial(&[0, -1, 0, 1], 2);
    assert!(negative < zero && zero < positive);
    assert_eq!(negative, AlgebraicReal::integer(-1));
    assert_eq!(zero, AlgebraicReal::zero());
    assert_eq!(positive, AlgebraicReal::one());
    assert_eq!(positive.as_fraction(), Some(BigRational::one()));
    assert!(matches!(
        positive.to_data(),
        crate::AlgebraicRealData::Multiquadratic(_)
    ));
}

#[test]
fn linear_roots_use_the_rational_fast_path() {
    let root = polynomial(&[-2, 1], 0);
    assert_eq!(root, AlgebraicReal::integer(2));
    assert!(matches!(
        root.to_data(),
        crate::AlgebraicRealData::Multiquadratic(_)
    ));
}

#[test]
fn reducible_repeated_input_becomes_minimal() {
    // (x² - 2)²(x - 3)
    let root = polynomial(&[12, -4, -12, 4, 3, -1], 0);
    assert_eq!(
        root.minimal_polynomial(),
        vec![(-2).into(), 0.into(), 1.into()]
    );
    assert_eq!(root, -AlgebraicReal::sqrt(2u32));
}

#[test]
fn canonical_selection_keeps_the_value_when_factor_root_index_changes() {
    // (x² - 2)(x + 10) has root indices 0, 1, 2, while x² - 2 has 0, 1.
    let value = polynomial(&[-20, -2, 10, 1], 2);
    let Representation::IsolatedRoot(root) = &value.representation else {
        panic!("a reducible cubic is stored as an isolated root");
    };
    assert_eq!(root.root_index(), 2);
    let canonical = canonical::selected_root(root);
    assert_eq!(
        canonical.polynomial(),
        &vec![(-2).into(), 0.into(), 1.into()]
    );
    assert_eq!(canonical.root_index(), 1);
    assert_eq!(&value * &value, AlgebraicReal::integer(2));
}

#[test]
fn equality_uses_the_selected_common_factor() {
    let root_two_or_three = [6, 0, -5, 0, 1];
    let root_two_or_five = [10, 0, -7, 0, 1];
    assert_eq!(
        polynomial(&root_two_or_three, 2),
        polynomial(&root_two_or_five, 2)
    );
    assert_ne!(
        polynomial(&root_two_or_three, 3),
        polynomial(&root_two_or_five, 3)
    );
    assert_ne!(
        polynomial(&root_two_or_three, 0),
        polynomial(&root_two_or_three, 1)
    );
}

#[test]
fn equality_counts_shared_roots_in_different_annihilators() {
    for radicand in [2, 3, 5, 8, 17] {
        let with_quadratic =
            |quadratic: i64| [radicand * quadratic, 0, -radicand, -quadratic, 0, 1];
        let left = polynomial(&with_quadratic(101), 1);
        let right = polynomial(&with_quadratic(103), 1);
        assert_eq!(left, right);
        assert_ne!(
            polynomial(&with_quadratic(101), 2),
            polynomial(&with_quadratic(103), 2)
        );
    }
}

#[test]
fn cubic_field_arithmetic_is_exact() {
    let root = polynomial(&[-2, 0, 0, 1], 0);
    assert_eq!(&(&root * &root) * &root, AlgebraicReal::integer(2));
    assert_eq!((&root + AlgebraicReal::one()) - AlgebraicReal::one(), root);
    let quotient = &root / &root;
    assert_eq!(quotient, AlgebraicReal::one());
    assert_eq!(quotient.as_fraction(), Some(BigRational::one()));
    assert!(matches!(
        quotient.to_data(),
        crate::AlgebraicRealData::Multiquadratic(_)
    ));
    assert!((&root - &root).is_zero());
    assert_eq!(root.degree(), 3);
}

#[test]
fn same_generator_chains_keep_bounded_degree() {
    let alpha = polynomial(&[-1, -1, 0, 0, 0, 1], 0);
    let mut power = alpha.clone();
    for _ in 2..=100 {
        power = &power * &alpha;
        assert!(power.structural_size() <= 11);
    }
    let fourth = alpha.pow_integer(&BigInt::from(4)).unwrap();
    assert_eq!(
        fourth.minimal_polynomial(),
        vec![
            (-1).into(),
            1.into(),
            (-4).into(),
            6.into(),
            (-4).into(),
            1.into()
        ]
    );
    let rebuilt = AlgebraicReal::from_data(fourth.to_data()).unwrap();
    assert_eq!(rebuilt, fourth);
    let hash = |value: &AlgebraicReal| {
        let mut state = DefaultHasher::new();
        value.hash(&mut state);
        state.finish()
    };
    assert_eq!(hash(&rebuilt), hash(&fourth));
    assert_eq!(
        alpha.pow_integer(&BigInt::from(5)).unwrap(),
        &alpha + AlgebraicReal::one()
    );
    assert_eq!(&alpha * alpha.recip().unwrap(), AlgebraicReal::one());

    let fourth_root_two = polynomial(&[-2, 0, 0, 0, 1], 1);
    let square = &fourth_root_two * &fourth_root_two;
    assert_eq!(square.degree(), 2);
    assert_eq!(square.trace(), BigRational::from_integer(0.into()));
    assert_eq!(square.norm(), BigRational::from_integer((-2).into()));
}

#[test]
fn unrelated_fields_still_compose_exactly() {
    let cube_root_two = polynomial(&[-2, 0, 0, 1], 0);
    let cube_root_three = polynomial(&[-3, 0, 0, 1], 0);
    let sum = &cube_root_two + &cube_root_three;
    assert_eq!(sum.degree(), 9);
    assert_eq!(&sum - &cube_root_two, cube_root_three);
}

#[test]
fn rational_power_leaves_multiquadratic_fields() {
    let cube_root = AlgebraicReal::integer(2)
        .pow_rational(&BigInt::one(), &BigUint::from(3u32))
        .unwrap();
    assert_eq!(
        &(&cube_root * &cube_root) * &cube_root,
        AlgebraicReal::integer(2)
    );
    assert_eq!(
        cube_root.minimal_polynomial(),
        vec![(-2).into(), 0.into(), 0.into(), 1.into()]
    );
    assert_eq!(cube_root.to_string(), "root(x^3 - 2, 0)");
}

#[test]
fn rational_power_handles_every_real_rational_exponent() {
    let one = BigInt::one();
    let three = BigUint::from(3u32);
    assert_eq!(
        AlgebraicReal::integer(-8)
            .pow_rational(&one, &three)
            .unwrap(),
        AlgebraicReal::integer(-2)
    );
    assert_eq!(
        AlgebraicReal::integer(-8)
            .pow_rational(&BigInt::from(2), &three)
            .unwrap(),
        AlgebraicReal::integer(4)
    );
    assert_eq!(
        AlgebraicReal::integer(-2).pow_rational(&one, &BigUint::from(2u32)),
        Err(AlgebraicError::EvenRootOfNegative)
    );
    assert_eq!(
        AlgebraicReal::integer(2).pow_rational(&one, &BigUint::from(4_097u32)),
        Err(AlgebraicError::RootDegreeTooLarge)
    );
    let reduced_root = AlgebraicReal::integer(8)
        .pow_rational(&one, &BigUint::from(6u32))
        .unwrap();
    assert_eq!(reduced_root, AlgebraicReal::sqrt(2u32));
    assert!(matches!(
        reduced_root.to_data(),
        crate::AlgebraicRealData::Multiquadratic(_)
    ));
}

#[test]
fn square_power_preserves_the_multiquadratic_fast_path() {
    let root = AlgebraicReal::integer(2)
        .pow_rational(&BigInt::one(), &BigUint::from(2u32))
        .unwrap();
    assert_eq!(root, AlgebraicReal::sqrt(2u32));
    assert!(matches!(
        root.to_data(),
        crate::AlgebraicRealData::Multiquadratic(_)
    ));
}

#[test]
fn nested_binomial_roots_compose_their_degrees() {
    let cube_root = AlgebraicReal::integer(2)
        .nth_root(3)
        .expect("positive cube root");
    let fifteenth_root = cube_root.nth_root(5).expect("nested positive root");
    assert_eq!(
        fifteenth_root.minimal_polynomial(),
        [BigInt::from(-2)]
            .into_iter()
            .chain(std::iter::repeat_n(BigInt::ZERO, 14))
            .chain([BigInt::one()])
            .collect::<Vec<_>>()
    );

    let negative_sixth_root = (-AlgebraicReal::sqrt(2u32))
        .nth_root(3)
        .expect("odd root of a negative value");
    assert_eq!(negative_sixth_root.signum(), -1);
    assert_eq!(
        negative_sixth_root.minimal_polynomial(),
        vec![
            (-2).into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            1.into()
        ]
    );
}

#[test]
fn rational_to_f64_is_ties_to_even() {
    let power = |bits: usize| BigInt::one() << bits;
    assert_eq!(
        round_rational_to_f64(&BigRational::new(1.into(), 10.into())).to_bits(),
        0.1f64.to_bits()
    );
    let halfway_above_one = BigRational::new(&power(53) + 1, power(53));
    assert_eq!(round_rational_to_f64(&halfway_above_one), 1.0);
    let carry_to_two = BigRational::new((&power(55) - 1) << 1usize, power(55));
    assert_eq!(round_rational_to_f64(&carry_to_two), 2.0);
    assert_eq!(round_rational_to_f64(&-&carry_to_two), -2.0);
    let half_min_subnormal = BigRational::new(BigInt::one(), power(1075));
    assert_eq!(round_rational_to_f64(&half_min_subnormal).to_bits(), 0);
    assert_eq!(
        round_rational_to_f64(&-&half_min_subnormal).to_bits(),
        (-0.0f64).to_bits()
    );
    let one_and_half_subnormal = BigRational::new(3.into(), power(1075));
    assert_eq!(round_rational_to_f64(&one_and_half_subnormal).to_bits(), 2);
    let overflow = BigRational::from_integer(power(2_000));
    assert_eq!(round_rational_to_f64(&overflow), f64::INFINITY);
    assert_eq!(round_rational_to_f64(&-overflow), f64::NEG_INFINITY);
}

#[test]
fn irrational_to_f64_is_correctly_rounded() {
    assert_eq!(AlgebraicReal::sqrt(2u32).to_f64(), 2.0f64.sqrt());
    assert_eq!((-AlgebraicReal::sqrt(2u32)).to_f64(), -2.0f64.sqrt());
}

#[test]
fn current_multiquadratic_encoding_rebuilds_large_canonical_radicands_directly() {
    let radicand = (BigUint::one() << 127usize) - 1u32;
    let data = crate::AlgebraicRealData::Multiquadratic(vec![(
        BigInt::one(),
        BigInt::one(),
        radicand.clone(),
    )]);
    let value = AlgebraicReal::from_data(data).unwrap();
    assert_eq!(
        value.to_data(),
        crate::AlgebraicRealData::Multiquadratic(vec![(BigInt::one(), BigInt::one(), radicand,)])
    );
}

#[test]
fn data_tags_and_layout_are_exact() {
    assert_eq!(
        AlgebraicReal::rational(BigRational::new(3.into(), 5.into())).to_data(),
        crate::AlgebraicRealData::Multiquadratic(vec![(
            BigInt::from(3),
            BigInt::from(5),
            BigUint::one(),
        )])
    );
    assert_eq!(
        (AlgebraicReal::one() + AlgebraicReal::sqrt(2u32)).to_data(),
        crate::AlgebraicRealData::Multiquadratic(vec![
            (BigInt::one(), BigInt::one(), BigUint::one()),
            (BigInt::one(), BigInt::one(), BigUint::from(2u32)),
        ])
    );
    assert_eq!(
        polynomial(&[-2, 0, 0, 1], 0).to_data(),
        crate::AlgebraicRealData::Polynomial {
            coefficients: vec![(-2).into(), 0.into(), 0.into(), 1.into()],
            root_index: 0,
        }
    );
}

#[test]
fn multiquadratic_decoding_combines_duplicates_and_drops_zero_terms() {
    let value = AlgebraicReal::from_data(crate::AlgebraicRealData::Multiquadratic(vec![
        (BigInt::one(), BigInt::one(), BigUint::from(2u32)),
        (BigInt::from(2), BigInt::one(), BigUint::from(2u32)),
        (BigInt::ZERO, BigInt::from(7), BigUint::from(3u32)),
    ]))
    .unwrap();
    assert_eq!(
        value.to_data(),
        crate::AlgebraicRealData::Multiquadratic(vec![(
            BigInt::from(3),
            BigInt::one(),
            BigUint::from(2u32),
        )])
    );
}

#[test]
fn multiquadratic_decoding_canonicalizes_radicands() {
    let value = AlgebraicReal::from_data(crate::AlgebraicRealData::Multiquadratic(vec![(
        BigInt::one(),
        BigInt::one(),
        BigUint::from(8u32),
    )]))
    .unwrap();
    assert_eq!(value, AlgebraicReal::integer(2) * AlgebraicReal::sqrt(2u32));
    assert_eq!(
        value.to_data(),
        crate::AlgebraicRealData::Multiquadratic(vec![(
            BigInt::from(2),
            BigInt::one(),
            BigUint::from(2u32),
        )])
    );
}

#[test]
fn equal_representations_hash_identically() {
    let fast = AlgebraicReal::sqrt(2u32);
    let root = polynomial(&[-2, 0, 1], 1);
    assert_eq!(fast, root);
    let hash = |value: &AlgebraicReal| {
        let mut state = DefaultHasher::new();
        value.hash(&mut state);
        state.finish()
    };
    assert_eq!(hash(&fast), hash(&root));

    let lazy_one = polynomial(&[0, -1, 0, 1], 2);
    assert_eq!(hash(&AlgebraicReal::one()), hash(&lazy_one));
}

#[test]
fn equal_separately_allocated_fields_share_arithmetic() {
    let left = polynomial(&[-2, 0, 0, 1], 0)
        .pow_integer(&BigInt::from(2))
        .unwrap();
    let right = polynomial(&[-2, 0, 0, 1], 0)
        .pow_integer(&BigInt::from(2))
        .unwrap();
    let sum = &left + &right;
    assert!(matches!(sum.representation, Representation::NumberField(_)));
    assert_eq!(sum, AlgebraicReal::integer(2) * left);
}

#[test]
fn public_representations_obey_field_and_data_identities() {
    let values = [
        AlgebraicReal::rational(BigRational::new(3.into(), 5.into())),
        AlgebraicReal::sqrt(2u32) + AlgebraicReal::sqrt(3u32),
        polynomial(&[-2, 0, 0, 1], 0)
            .pow_integer(&BigInt::from(2))
            .unwrap(),
    ];
    for value in &values {
        assert_eq!(AlgebraicReal::from_data(value.to_data()).unwrap(), *value);
        assert_eq!(value.abs().signum(), 1);
    }
    for left in &values {
        for right in &values {
            assert_eq!((&(left + right) - right), *left);
            assert_eq!((&(left * right) / right), *left);
        }
    }
}

#[test]
fn equal_separately_allocated_isolated_roots_share_arithmetic() {
    let left = polynomial(&[-2, 0, 0, 1], 0);
    let right = polynomial(&[-2, 0, 0, 1], 0);
    let sum = &left + &right;
    assert!(matches!(sum.representation, Representation::NumberField(_)));
    assert_eq!(sum, AlgebraicReal::integer(2) * left);
}

#[test]
fn isolated_root_reuses_its_selected_root() {
    let value = polynomial(&[-2, 0, 0, 1], 0);
    let Representation::IsolatedRoot(root) = &value.representation else {
        panic!("an irreducible cubic is stored as an isolated root");
    };
    let first = canonical::selected_root(root);
    let second = canonical::selected_root(root);
    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn minimum_small_integer_has_an_exact_reciprocal() {
    let minimum = AlgebraicReal::integer(BigInt::from(i128::MIN));
    assert_eq!(
        minimum.recip().unwrap().as_fraction(),
        Some(BigRational::new(BigInt::one(), BigInt::from(i128::MIN)))
    );
}

#[test]
fn invalid_inputs_have_the_documented_results() {
    assert_eq!(
        AlgebraicReal::from_polynomial(vec![BigRational::one()], 0),
        Err(AlgebraicError::ConstantPolynomial)
    );
    assert_eq!(
        AlgebraicReal::from_polynomial(
            vec![BigRational::from_integer((-2).into()), BigRational::one()],
            1,
        ),
        Err(AlgebraicError::RootIndex { index: 1, roots: 1 })
    );
    assert_eq!(
        AlgebraicReal::from_data(crate::AlgebraicRealData::Multiquadratic(vec![(
            BigInt::one(),
            BigInt::ZERO,
            BigUint::one(),
        )])),
        Err(AlgebraicError::ZeroDenominator)
    );
    assert_eq!(
        AlgebraicReal::from_data(crate::AlgebraicRealData::Multiquadratic(vec![(
            BigInt::one(),
            BigInt::one(),
            BigUint::ZERO,
        )])),
        Err(AlgebraicError::InvalidEncoding)
    );

    let zero = AlgebraicReal::zero();
    assert_eq!(AlgebraicReal::one().checked_div(&zero), None);
    assert_eq!(zero.recip(), None);
    assert_eq!(
        zero.pow_integer(&BigInt::from(-1)),
        Err(AlgebraicError::DivisionByZero)
    );
    assert!(std::panic::catch_unwind(|| AlgebraicReal::one() / zero).is_err());
}

#[test]
fn rounding_is_exact_across_rational_and_irrational_roots() {
    let positive = polynomial(&[-2, 0, 1], 1);
    let negative = -&positive;
    assert_eq!(positive.floor(), BigInt::one());
    assert_eq!(positive.ceil(), BigInt::from(2));
    assert_eq!(positive.trunc(), BigInt::one());
    assert_eq!(negative.floor(), BigInt::from(-2));
    assert_eq!(negative.ceil(), BigInt::from(-1));
    assert_eq!(negative.trunc(), BigInt::from(-1));

    let rational_root_of_reducible = polynomial(&[6, -2, -3, 1], 2);
    assert_eq!(rational_root_of_reducible.floor(), BigInt::from(3));
    assert_eq!(rational_root_of_reducible.to_f64(), 3.0);
}
