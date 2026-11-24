// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod algebraic_real;
#[cfg(test)]
mod algebraic_real_tests;
mod arithmetic;
mod canonical;
mod float;
mod integer_polynomial_factorization;
mod isolated_root;
mod multiquadratic;
mod number_field;
mod polynomial;
mod prime_field_polynomial;
mod rational_polynomial;
mod rational_root;
mod representation;
mod root_composition;
mod selected_root;

use std::fmt;

use num_bigint::{BigInt, BigUint};

pub use algebraic_real::AlgebraicReal;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AlgebraicRealData {
    Multiquadratic(Vec<(BigInt, BigInt, BigUint)>),
    Polynomial {
        coefficients: Vec<BigInt>,
        root_index: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AlgebraicError {
    ConstantPolynomial,
    RootIndex { index: usize, roots: usize },
    DivisionByZero,
    ZeroDenominator,
    InvalidEncoding,
    ZeroRootDegree,
    RootDegreeTooLarge,
    EvenRootOfNegative,
    IsolationFailed,
}

impl fmt::Display for AlgebraicError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConstantPolynomial => write!(formatter, "a root needs a nonconstant polynomial"),
            Self::RootIndex { index, roots } => write!(
                formatter,
                "root index {index} is outside {roots} distinct real roots"
            ),
            Self::DivisionByZero => write!(formatter, "division by zero"),
            Self::ZeroDenominator => write!(formatter, "a rational denominator must be nonzero"),
            Self::InvalidEncoding => write!(formatter, "invalid AlgebraicReal encoding"),
            Self::ZeroRootDegree => write!(formatter, "root degree must be positive"),
            Self::RootDegreeTooLarge => write!(formatter, "root degree is too large"),
            Self::EvenRootOfNegative => write!(
                formatter,
                "an even real root of a negative number does not exist"
            ),
            Self::IsolationFailed => write!(formatter, "root isolation precision overflowed"),
        }
    }
}

impl std::error::Error for AlgebraicError {}
