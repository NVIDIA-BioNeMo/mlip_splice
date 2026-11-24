// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact linear algebra over the multiquadratic ring: the `Ring`/`Conjugate`
//! abstraction, the complexification `ComplexAlgebraicReal` carrying the character
//! basis of the real irreps, and a `SparseTensor` for Clebsch-Gordan tables.
//! Everything is built on `algebraic_real::AlgebraicReal`; a real coefficient is a single
//! canonical `AlgebraicReal`, so no dedicated single-surd type is needed.

mod complex;
mod ring;
mod sparse;

pub use complex::ComplexAlgebraicReal;
pub use ring::{Conjugate, Ring};
pub use sparse::SparseTensor;
