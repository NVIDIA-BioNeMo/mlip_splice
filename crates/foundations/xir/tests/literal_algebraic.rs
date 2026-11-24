// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use algebraic_real::AlgebraicReal;
use num_bigint::BigInt;
use num_rational::BigRational;

use xir::dtype::DType;
use xir::ops::Literal;
use xir::shape::Shape;

#[test]
fn finite_bits_decode_to_exact_values_in_every_float_format() {
    assert_eq!(
        values(scalar(DType::F16, 0xc100u16.to_le_bytes().to_vec())),
        vec![dyadic(-5, -1)]
    );
    assert_eq!(
        values(scalar(DType::Bf16, 0x3fc0u16.to_le_bytes().to_vec())),
        vec![dyadic(3, -1)]
    );
    assert_eq!(
        values(scalar(DType::F32, (-13.25f32).to_le_bytes().to_vec())),
        vec![dyadic(-53, -2)]
    );
    assert_eq!(
        values(scalar(DType::F64, 10.125f64.to_le_bytes().to_vec())),
        vec![dyadic(81, -3)]
    );
}

#[test]
fn subnormals_and_maximum_finite_values_decode_exactly() {
    assert_eq!(
        values(scalar(DType::F16, 0x0001u16.to_le_bytes().to_vec())),
        vec![dyadic(1, -24)]
    );
    assert_eq!(
        values(scalar(DType::F16, 0x7bffu16.to_le_bytes().to_vec())),
        vec![dyadic(2047, 5)]
    );

    assert_eq!(
        values(scalar(DType::Bf16, 0x0001u16.to_le_bytes().to_vec())),
        vec![dyadic(1, -133)]
    );
    assert_eq!(
        values(scalar(DType::Bf16, 0x7f7fu16.to_le_bytes().to_vec())),
        vec![dyadic(255, 120)]
    );

    assert_eq!(
        values(scalar(DType::F32, 0x0000_0001u32.to_le_bytes().to_vec())),
        vec![dyadic(1, -149)]
    );
    assert_eq!(
        values(scalar(DType::F32, 0x7f7f_ffffu32.to_le_bytes().to_vec())),
        vec![dyadic((BigInt::from(1u8) << 24usize) - 1u8, 104)]
    );

    assert_eq!(
        values(scalar(
            DType::F64,
            0x0000_0000_0000_0001u64.to_le_bytes().to_vec(),
        )),
        vec![dyadic(1, -1074)]
    );
    assert_eq!(
        values(scalar(
            DType::F64,
            0x7fef_ffff_ffff_ffffu64.to_le_bytes().to_vec(),
        )),
        vec![dyadic((BigInt::from(1u8) << 53usize) - 1u8, 971)]
    );
}

#[test]
fn finite_float_bits_round_trip_through_exact_literals() {
    let cases = [
        (
            DType::F16,
            [0x0001u16, 0x03ff, 0x0400, 0x7bff, 0x8001, 0xfbff]
                .map(u16::to_le_bytes)
                .concat(),
        ),
        (
            DType::Bf16,
            [0x0001u16, 0x007f, 0x0080, 0x7f7f, 0x8001, 0xff7f]
                .map(u16::to_le_bytes)
                .concat(),
        ),
        (
            DType::F32,
            [
                1u32,
                0x007f_ffff,
                0x0080_0000,
                f32::MAX.to_bits(),
                (-f32::MAX).to_bits(),
            ]
            .map(u32::to_le_bytes)
            .concat(),
        ),
        (
            DType::F64,
            [
                1u64,
                0x000f_ffff_ffff_ffff,
                0x0010_0000_0000_0000,
                f64::MAX.to_bits(),
                (-f64::MAX).to_bits(),
            ]
            .map(u64::to_le_bytes)
            .concat(),
        ),
    ];
    for (dtype, data) in cases {
        let elements = data.len() / dtype.size_bytes();
        let shape = Shape::new(dtype, [elements as i64]);
        let literal = Literal::new(shape.clone(), data.clone());
        let encoded = Literal::exact(shape.clone(), values(literal)).unwrap();
        assert_eq!(encoded.shape, shape);
        assert_eq!(encoded.data, data, "{dtype:?}");
    }
}

#[test]
fn both_float_zero_signs_are_algebraic_zero() {
    let cases = [
        (
            DType::F16,
            [0x0000u16.to_le_bytes(), 0x8000u16.to_le_bytes()].concat(),
        ),
        (
            DType::Bf16,
            [0x0000u16.to_le_bytes(), 0x8000u16.to_le_bytes()].concat(),
        ),
        (
            DType::F32,
            [0x0000_0000u32.to_le_bytes(), 0x8000_0000u32.to_le_bytes()].concat(),
        ),
        (
            DType::F64,
            [
                0x0000_0000_0000_0000u64.to_le_bytes(),
                0x8000_0000_0000_0000u64.to_le_bytes(),
            ]
            .concat(),
        ),
    ];

    for (dtype, data) in cases {
        let decoded = Literal::new(Shape::new(dtype, [2]), data)
            .algebraic_values()
            .unwrap();
        assert_eq!(decoded, vec![AlgebraicReal::zero(); 2]);
    }
}

#[test]
fn scalar_vector_matrix_and_empty_shapes_preserve_element_order() {
    let scalar_value = scalar(DType::F32, 1.5f32.to_le_bytes().to_vec());
    assert_eq!(values(scalar_value), vec![dyadic(3, -1)]);

    let vector = Literal::new(
        Shape::new(DType::F16, [3]),
        [0x3c00u16, 0xc000, 0x3800]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
    );
    assert_eq!(
        values(vector),
        vec![
            AlgebraicReal::one(),
            AlgebraicReal::integer(-2),
            dyadic(1, -1),
        ]
    );

    let matrix = Literal::new(
        Shape::new(DType::Bf16, [2, 2]),
        [0x3f80u16, 0x4000, 0x4040, 0x4080]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
    );
    assert_eq!(
        values(matrix),
        (1..=4).map(AlgebraicReal::integer).collect::<Vec<_>>()
    );

    let empty = Literal::new(Shape::new(DType::F64, [2, 0, 3]), Vec::new());
    assert_eq!(empty.algebraic_values(), Some(Vec::new()));
}

#[test]
fn nonfloat_or_malformed_literals_have_no_algebraic_values() {
    assert!(
        Literal::new(Shape::scalar(DType::I32), 7i32.to_le_bytes().to_vec())
            .algebraic_values()
            .is_none()
    );

    let malformed = [
        Literal::new(Shape::scalar(DType::F16), vec![0]),
        Literal::new(Shape::new(DType::F32, [2]), 1.0f32.to_le_bytes().to_vec()),
        Literal::new(Shape::scalar(DType::F64), vec![0; 9]),
    ];
    for literal in malformed {
        assert!(literal.algebraic_values().is_none());
    }

    let mut wrong_exact_count =
        Literal::exact(Shape::scalar(DType::F32), vec![AlgebraicReal::one()]).unwrap();
    wrong_exact_count.exact = Some(Vec::new());
    assert!(wrong_exact_count.algebraic_values().is_none());
}

#[test]
fn any_nan_or_infinity_rejects_the_whole_literal() {
    let cases = [
        (
            DType::F16,
            0x3c00u16.to_le_bytes().to_vec(),
            0x7c00u16.to_le_bytes().to_vec(),
            0x7e00u16.to_le_bytes().to_vec(),
        ),
        (
            DType::Bf16,
            0x3f80u16.to_le_bytes().to_vec(),
            0x7f80u16.to_le_bytes().to_vec(),
            0x7fc0u16.to_le_bytes().to_vec(),
        ),
        (
            DType::F32,
            1.0f32.to_le_bytes().to_vec(),
            f32::INFINITY.to_le_bytes().to_vec(),
            f32::NAN.to_le_bytes().to_vec(),
        ),
        (
            DType::F64,
            1.0f64.to_le_bytes().to_vec(),
            f64::NEG_INFINITY.to_le_bytes().to_vec(),
            f64::NAN.to_le_bytes().to_vec(),
        ),
    ];

    for (dtype, finite, infinity, nan) in cases {
        let mut with_infinity = finite.clone();
        with_infinity.extend(infinity);
        assert!(
            Literal::new(Shape::new(dtype, [2]), with_infinity)
                .algebraic_values()
                .is_none()
        );
        assert!(
            Literal::new(Shape::scalar(dtype), nan)
                .algebraic_values()
                .is_none()
        );
    }
}

#[test]
fn explicit_exact_values_override_runtime_bits_without_mutating_provenance() {
    let root = AlgebraicReal::sqrt(2u32);
    let exact = Literal::new(Shape::scalar(DType::F32), f32::NAN.to_le_bytes().to_vec())
        .with_exact(vec![root.clone()])
        .unwrap();
    let before = exact.clone();

    assert_eq!(exact.algebraic_values(), Some(vec![root.clone()]));
    assert_eq!(exact, before);

    let mut missing_runtime_bytes = exact.clone();
    missing_runtime_bytes.data.clear();
    assert_eq!(missing_runtime_bytes.algebraic_values(), Some(vec![root]));

    let ordinary = scalar(DType::F32, 0.25f32.to_le_bytes().to_vec());
    let before = ordinary.clone();
    assert_eq!(ordinary.algebraic_values(), Some(vec![dyadic(1, -2)]));
    assert_eq!(ordinary, before);
    assert!(ordinary.exact.is_none());
}

#[test]
fn f32_point_one_is_its_dyadic_value_not_one_tenth() {
    let decoded = values(scalar(DType::F32, 0.1f32.to_le_bytes().to_vec()));
    let dyadic_point_one = rational(13_421_773, 134_217_728);
    let one_tenth = rational(1, 10);

    assert_eq!(decoded, vec![dyadic_point_one]);
    assert_ne!(decoded, vec![one_tenth]);
}

fn scalar(dtype: DType, data: Vec<u8>) -> Literal {
    Literal::new(Shape::scalar(dtype), data)
}

fn values(literal: Literal) -> Vec<AlgebraicReal> {
    literal.algebraic_values().unwrap()
}

fn rational(numerator: impl Into<BigInt>, denominator: impl Into<BigInt>) -> AlgebraicReal {
    AlgebraicReal::rational(BigRational::new(numerator.into(), denominator.into()))
}

fn dyadic(significand: impl Into<BigInt>, exponent: i32) -> AlgebraicReal {
    let significand = significand.into();
    if exponent < 0 {
        rational(
            significand,
            BigInt::from(1u8) << usize::try_from(-exponent).unwrap(),
        )
    } else {
        AlgebraicReal::integer(significand << usize::try_from(exponent).unwrap())
    }
}
