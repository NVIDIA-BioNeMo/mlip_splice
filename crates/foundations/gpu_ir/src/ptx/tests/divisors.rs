// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Division by a constant: the multiply-shift sequence the backend emits,
//! checked against exact division over the whole width.

use super::super::divide::{magic_signed, magic_unsigned};
use super::{kernel, ptr, render};
use crate::{Scalar, Ty};

/// Mirror of the emitted signed quotient sequence at `bits` width:
/// mulhi/add/shifts for magic divisors, bias-and-shift for powers of two.
fn sdiv_sequence(n: i128, d: i64, bits: u32) -> i128 {
    let wrap = |x: i128| (x << (128 - bits)) >> (128 - bits);
    let lshr = |x: i128, k: u32| ((x as u128 & ((1u128 << bits) - 1)) >> k) as i128;
    if d == 1 {
        return n;
    }
    if d == -1 {
        return wrap(-n);
    }
    if d == (-1i64) << (bits - 1) {
        return i128::from(n == i128::from(d));
    }
    if d.unsigned_abs().is_power_of_two() {
        let k = d.unsigned_abs().trailing_zeros();
        let sign = n >> (bits - 1);
        let q = wrap(n + lshr(sign, bits - k)) >> k;
        return if d < 0 { wrap(-q) } else { q };
    }
    let (m, sh) = magic_signed(d, bits);
    let mut q = wrap((i128::from(m) * n) >> bits);
    if d > 0 && m < 0 {
        q = wrap(q + n);
    } else if d < 0 && m > 0 {
        q = wrap(q - n);
    }
    q >>= sh;
    wrap(q + lshr(q, bits - 1))
}

/// Mirror of the emitted unsigned quotient sequence at `bits` width.
fn udiv_sequence(n: u128, d: u64, bits: u32) -> u128 {
    let mask = (1u128 << bits) - 1;
    if d == 1 {
        return n;
    }
    if d.is_power_of_two() {
        return n >> d.trailing_zeros();
    }
    let (m, add, sh) = magic_unsigned(d, bits);
    let q0 = (u128::from(m) * n) >> bits;
    if add {
        let t = ((n - q0) & mask) >> 1;
        ((t + q0) & mask) >> (sh - 1)
    } else {
        q0 >> sh
    }
}

#[test]
fn div_rem_by_constant_matches_exact_division() {
    let mut rng = 0x9e3779b97f4a7c15u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for bits in [32u32, 64] {
        let (min, max) = (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1);
        let mut divisors = vec![
            1,
            -1,
            2,
            -2,
            3,
            -3,
            5,
            6,
            7,
            -7,
            9,
            10,
            11,
            12,
            25,
            60,
            125,
            641,
            3520,
            4352,
            10000,
            -10000,
            1 << 20,
            (1i64 << (bits - 2)) - 1,
        ];
        divisors.push(min as i64); // MIN at this width
        divisors.push(max as i64);
        for &d in &divisors {
            let mut dividends = vec![0, 1, -1, 2, -2, min, min + 1, max, max - 1];
            for _ in 0..500 {
                let n = (next() as i128) << (bits / 2) ^ next() as i128;
                dividends.push((n << (128 - bits)) >> (128 - bits));
            }
            for &n in &dividends {
                let q = sdiv_sequence(n, d, bits);
                let wrap = |x: i128| (x << (128 - bits)) >> (128 - bits);
                assert_eq!(q, wrap(n / i128::from(d)), "{n} / {d} at {bits} bits");
                let r = wrap(n - q * i128::from(d));
                assert_eq!(r, wrap(n % i128::from(d)), "{n} % {d} at {bits} bits");
            }
        }
        let mask = if bits == 64 {
            u128::from(u64::MAX)
        } else {
            (1u128 << bits) - 1
        };
        let mut udivisors: Vec<u64> = vec![
            1,
            2,
            3,
            5,
            6,
            7,
            9,
            10,
            11,
            12,
            25,
            60,
            125,
            641,
            3520,
            4352,
            10000,
            1 << 20,
            7 << 20,
        ];
        udivisors.push(((mask >> 1) + 1) as u64); // 2^(bits-1)
        udivisors.push(((mask >> 1) + 3) as u64); // just above it
        udivisors.push(mask as u64); // 2^bits - 1
        for &d in &udivisors {
            let mut dividends: Vec<u128> = vec![0, 1, 2, mask, mask - 1, mask >> 1];
            for _ in 0..500 {
                dividends.push(((u128::from(next()) << 64) | u128::from(next())) & mask);
            }
            for &n in &dividends {
                let q = udiv_sequence(n, d, bits);
                assert_eq!(q, n / u128::from(d), "{n} / {d} at {bits} bits");
                let r = (n - q * u128::from(d)) & mask;
                assert_eq!(r, n % u128::from(d), "{n} % {d} at {bits} bits");
            }
        }
    }
}

#[test]
fn constant_divisors_lower_to_multiply_shift() {
    let k = kernel(|b| {
        let out = ptr(b, "out", Scalar::S64);
        let read = b.load(out.clone(), 1, false);
        let n = b.bind_value_as("n", Ty::scalar(Scalar::S64), read);
        let divisor = b.i64(4352);
        let quotient = b.div(n, divisor);
        let modulus = b.i64(10000);
        let value = b.rem(quotient, modulus);
        b.store(out, value);
    });
    let ptx = render(&k, 120);
    assert!(!ptx.contains("div.s64"), "{ptx}");
    assert!(!ptx.contains("rem.s64"), "{ptx}");
    assert_eq!(ptx.matches("mul.hi.s64").count(), 2, "{ptx}");
}
