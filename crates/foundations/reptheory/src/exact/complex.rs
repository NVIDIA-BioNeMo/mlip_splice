// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ops::{Add, Mul, Neg, Sub};

use algebraic_real::AlgebraicReal;

/// A Gaussian algebraic number `re + im·i`, with both parts exact.
/// The complexification carries the character basis of the real irreps.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ComplexAlgebraicReal {
    pub re: AlgebraicReal,
    pub im: AlgebraicReal,
}

impl ComplexAlgebraicReal {
    pub fn zero() -> Self {
        ComplexAlgebraicReal {
            re: AlgebraicReal::zero(),
            im: AlgebraicReal::zero(),
        }
    }

    pub fn one() -> Self {
        ComplexAlgebraicReal {
            re: AlgebraicReal::one(),
            im: AlgebraicReal::zero(),
        }
    }

    pub fn real(re: AlgebraicReal) -> Self {
        ComplexAlgebraicReal {
            re,
            im: AlgebraicReal::zero(),
        }
    }

    pub fn imag(im: AlgebraicReal) -> Self {
        ComplexAlgebraicReal {
            re: AlgebraicReal::zero(),
            im,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }

    pub fn conj(&self) -> Self {
        ComplexAlgebraicReal {
            re: self.re.clone(),
            im: -&self.im,
        }
    }

    pub fn into_real(self) -> AlgebraicReal {
        assert!(
            self.im.is_zero(),
            "into_real on a value with nonzero imaginary part"
        );
        self.re
    }
}

impl Neg for &ComplexAlgebraicReal {
    type Output = ComplexAlgebraicReal;
    fn neg(self) -> ComplexAlgebraicReal {
        ComplexAlgebraicReal {
            re: -&self.re,
            im: -&self.im,
        }
    }
}

impl Neg for ComplexAlgebraicReal {
    type Output = ComplexAlgebraicReal;
    fn neg(self) -> ComplexAlgebraicReal {
        ComplexAlgebraicReal {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl Add for &ComplexAlgebraicReal {
    type Output = ComplexAlgebraicReal;
    fn add(self, rhs: &ComplexAlgebraicReal) -> ComplexAlgebraicReal {
        ComplexAlgebraicReal {
            re: &self.re + &rhs.re,
            im: &self.im + &rhs.im,
        }
    }
}

impl Sub for &ComplexAlgebraicReal {
    type Output = ComplexAlgebraicReal;
    fn sub(self, rhs: &ComplexAlgebraicReal) -> ComplexAlgebraicReal {
        ComplexAlgebraicReal {
            re: &self.re - &rhs.re,
            im: &self.im - &rhs.im,
        }
    }
}

impl Mul for &ComplexAlgebraicReal {
    type Output = ComplexAlgebraicReal;
    fn mul(self, rhs: &ComplexAlgebraicReal) -> ComplexAlgebraicReal {
        // (a + bi)(c + di) = (ac − bd) + (ad + bc) i
        ComplexAlgebraicReal {
            re: &(&self.re * &rhs.re) - &(&self.im * &rhs.im),
            im: &(&self.re * &rhs.im) + &(&self.im * &rhs.re),
        }
    }
}

macro_rules! forward_binop {
    ($trait:ident, $method:ident) => {
        impl $trait<ComplexAlgebraicReal> for ComplexAlgebraicReal {
            type Output = ComplexAlgebraicReal;
            fn $method(self, rhs: ComplexAlgebraicReal) -> ComplexAlgebraicReal {
                $trait::$method(&self, &rhs)
            }
        }
        impl $trait<&ComplexAlgebraicReal> for ComplexAlgebraicReal {
            type Output = ComplexAlgebraicReal;
            fn $method(self, rhs: &ComplexAlgebraicReal) -> ComplexAlgebraicReal {
                $trait::$method(&self, rhs)
            }
        }
        impl $trait<ComplexAlgebraicReal> for &ComplexAlgebraicReal {
            type Output = ComplexAlgebraicReal;
            fn $method(self, rhs: ComplexAlgebraicReal) -> ComplexAlgebraicReal {
                $trait::$method(self, &rhs)
            }
        }
    };
}
forward_binop!(Add, add);
forward_binop!(Sub, sub);
forward_binop!(Mul, mul);

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::BigRational;

    fn s(n: i64) -> AlgebraicReal {
        AlgebraicReal::integer(n)
    }

    #[test]
    fn i_squared_is_minus_one() {
        let i = ComplexAlgebraicReal::imag(AlgebraicReal::one());
        assert_eq!(&i * &i, ComplexAlgebraicReal::real(s(-1)));
    }

    #[test]
    fn one_is_multiplicative_identity() {
        let z = ComplexAlgebraicReal {
            re: AlgebraicReal::sqrt(2u32),
            im: AlgebraicReal::integer(3),
        };
        assert_eq!(&z * &ComplexAlgebraicReal::one(), z);
    }

    #[test]
    fn zero_test_tracks_both_parts() {
        assert!(ComplexAlgebraicReal::zero().is_zero());
        assert!(!ComplexAlgebraicReal::real(AlgebraicReal::one()).is_zero());
        assert!(!ComplexAlgebraicReal::imag(AlgebraicReal::one()).is_zero());
    }

    #[test]
    fn conjugate_product_is_modulus_squared() {
        let z = ComplexAlgebraicReal {
            re: AlgebraicReal::sqrt(2u32),
            im: AlgebraicReal::integer(3),
        };
        let m = &z * &z.conj();
        assert_eq!(m.im, AlgebraicReal::zero());
        assert_eq!(m.re, AlgebraicReal::integer(2 + 9));
    }

    #[test]
    fn into_real_keeps_real_part() {
        let z = ComplexAlgebraicReal::real(AlgebraicReal::term(
            BigRational::new(BigInt::from(1), BigInt::from(2)),
            2u32,
        ));
        assert_eq!(z.clone().into_real(), z.re);
    }

    #[test]
    #[should_panic(expected = "nonzero imaginary part")]
    fn into_real_rejects_imaginary() {
        ComplexAlgebraicReal::imag(AlgebraicReal::one()).into_real();
    }
}
