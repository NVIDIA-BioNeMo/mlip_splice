// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte extents, multiplied out without lying about the answer.
//!
//! Every buffer a primitive binds is a product of counts — batch elements,
//! grid points, components per point, the width of one value — and a recipe
//! states that product as the extent a caller must allocate. A product that
//! overflows has no extent, and a wrapped one is worse than no answer: it
//! promises a buffer smaller than the kernel addresses, and the launch runs.
//! So every factor is checked, and a count that cannot be one is refused by
//! name.

/// The product of `factors`, refused rather than wrapped or negated.
///
/// No factors is a product of one — the empty product — and a zero factor
/// makes the whole product zero. Neither is an error: a scalar holds one
/// value and an empty axis holds none. Only a negative count and an overflow
/// are impossible.
pub fn product(what: &str, factors: &[i64]) -> Result<i64, String> {
    let mut total: i64 = 1;
    for &factor in factors {
        if factor < 0 {
            return Err(format!("{what}: a count cannot be {factor} ({factors:?})"));
        }
        total = total
            .checked_mul(factor)
            .ok_or_else(|| format!("{what}: {factors:?} multiply past what an i64 can count"))?;
    }
    Ok(total)
}

/// That product as a byte extent — the number a signature states and a caller
/// allocates.
pub fn nbytes(what: &str, factors: &[i64]) -> Result<usize, String> {
    let total = product(what, factors)?;
    usize::try_from(total).map_err(|_| format!("{what}: {total} bytes is not an extent"))
}

/// The same product, refused when it comes out empty.
///
/// For a buffer whose kernel cannot be emitted over nothing: a stage that
/// declares a local array of this many values, or a grid sized by it, has no
/// zero-sized form.
pub fn nonempty_nbytes(what: &str, factors: &[i64]) -> Result<usize, String> {
    let nbytes = nbytes(what, factors)?;
    if nbytes == 0 {
        return Err(format!("{what}: holds no bytes ({factors:?})"));
    }
    Ok(nbytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_factors_is_the_empty_product() {
        assert_eq!(product("x", &[]).unwrap(), 1);
        assert_eq!(nbytes("x", &[]).unwrap(), 1);
    }

    #[test]
    fn a_zero_factor_carries_the_whole_product_to_zero() {
        assert_eq!(product("x", &[7, 0, 3]).unwrap(), 0);
        assert_eq!(nbytes("x", &[0, i64::MAX]).unwrap(), 0);
    }

    #[test]
    fn a_populated_product_is_its_factors() {
        assert_eq!(product("x", &[2, 3, 5]).unwrap(), 30);
        assert_eq!(nbytes("x", &[2, 3, 5]).unwrap(), 30);
    }

    /// The defect this module exists to prevent: two large counts whose product
    /// wraps to a small positive number, which a caller would allocate and the
    /// kernel would run past.
    #[test]
    fn a_product_that_would_wrap_is_refused_by_name() {
        let error = nbytes("a grid input", &[1 << 40, 1 << 40]).unwrap_err();
        assert!(error.contains("a grid input"), "{error}");
        assert!(error.contains("past what an i64"), "{error}");
        assert!(product("x", &[i64::MAX, 2]).is_err());
        assert!(
            product("x", &[i64::MAX, 3, 0]).is_err(),
            "the zero is never reached"
        );
    }

    #[test]
    fn a_negative_count_is_refused_before_it_is_cast() {
        let error = product("an aux input", &[4, -1]).unwrap_err();
        assert!(error.contains("an aux input"), "{error}");
        assert!(error.contains("cannot be -1"), "{error}");
        // The cast a wrapped negative would survive.
        assert!(nbytes("x", &[-2, 3]).is_err());
    }

    #[test]
    fn an_extent_that_must_hold_something_refuses_an_empty_product() {
        assert_eq!(nonempty_nbytes("x", &[4, 2]).unwrap(), 8);
        let error = nonempty_nbytes("a frequency buffer", &[4, 0]).unwrap_err();
        assert!(error.contains("a frequency buffer"), "{error}");
        assert!(error.contains("holds no bytes"), "{error}");
    }
}
