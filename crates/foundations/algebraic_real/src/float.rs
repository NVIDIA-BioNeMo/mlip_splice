// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use num_bigint::BigUint;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};

fn rounded_quotient(numerator: BigUint, denominator: &BigUint) -> BigUint {
    let (quotient, remainder) = numerator.div_rem(denominator);
    let twice_remainder = &remainder << 1usize;
    if twice_remainder > *denominator || (twice_remainder == *denominator && quotient.bit(0)) {
        quotient + 1u32
    } else {
        quotient
    }
}

pub(crate) fn round_rational_to_f64(value: &BigRational) -> f64 {
    if value.is_zero() {
        return 0.0;
    }
    let negative = value.is_negative();
    let numerator = value.numer().magnitude();
    let denominator = value.denom().magnitude();
    let mut exponent = numerator.bits() as i64 - denominator.bits() as i64;
    if exponent >= 0 {
        if numerator < &(denominator << exponent as usize) {
            exponent -= 1;
        }
    } else if (numerator << (-exponent) as usize) < *denominator {
        exponent -= 1;
    }
    let sign = if negative { 1u64 << 63 } else { 0 };
    if exponent < -1022 {
        let significand = rounded_quotient(numerator << 1074usize, denominator);
        if significand.is_zero() {
            return f64::from_bits(sign);
        }
        let bits = significand.to_u64().unwrap_or(1u64 << 52);
        return f64::from_bits(sign | bits);
    }
    if exponent > 1023 {
        return f64::from_bits(sign | (0x7ffu64 << 52));
    }
    let shift = 52 - exponent;
    let significand = if shift >= 0 {
        rounded_quotient(numerator << shift as usize, denominator)
    } else {
        rounded_quotient(numerator.clone(), &(denominator << (-shift) as usize))
    };
    if significand.bits() > 53 {
        exponent += 1;
        if exponent > 1023 {
            return f64::from_bits(sign | (0x7ffu64 << 52));
        }
        let mantissa = (significand >> 1usize).to_u64().unwrap() & ((1u64 << 52) - 1);
        return f64::from_bits(sign | ((exponent + 1023) as u64) << 52 | mantissa);
    }
    let mantissa = significand.to_u64().unwrap() & ((1u64 << 52) - 1);
    f64::from_bits(sign | ((exponent + 1023) as u64) << 52 | mantissa)
}
