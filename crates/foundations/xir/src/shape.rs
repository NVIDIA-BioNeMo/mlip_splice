// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Logical, layout-free tensor shapes: an element [`DType`] plus dimension
//! sizes. There is deliberately no layout (minor-to-major order) here — layout
//! is a downstream concern, not part of a value's type.

use smallvec::SmallVec;

use crate::dtype::DType;
use domain_index::FiniteSetId;

pub type Dims = SmallVec<[i64; 4]>;
pub type DomainAxes = SmallVec<[FiniteSetId; 4]>;

/// The number of elements a tensor of `dims` holds, or `None` when there is no
/// such count.
///
/// No dimensions is the empty product, which is one element — a scalar needs a
/// buffer. A zero dimension makes the tensor empty, and that is a real `0`: the
/// two are different shapes, and a launch sized from the wrong one writes past
/// the end of the other.
///
/// The zero is answered first, so a shape holding nothing has the same answer
/// however its dimensions are ordered — `[i64::MAX, 2, 0]` is empty exactly as
/// `[0, i64::MAX, 2]` is, and neither is a failure. What remains has no answer:
/// a negative dimension is not a dimension, and a product past `usize` counts
/// no buffer this machine can hold.
pub fn elements_in(dims: &[i64]) -> Option<usize> {
    if dims.iter().any(|&d| d < 0) {
        return None;
    }
    if dims.contains(&0) {
        return Some(0);
    }
    dims.iter()
        .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    pub dtype: DType,
    pub domain_axes: DomainAxes,
    pub dims: Dims,
}

impl Shape {
    pub fn new(dtype: DType, dims: impl IntoIterator<Item = i64>) -> Shape {
        Shape {
            dtype,
            domain_axes: DomainAxes::new(),
            dims: dims.into_iter().collect(),
        }
    }

    /// A rank-0 (scalar) shape.
    pub fn scalar(dtype: DType) -> Shape {
        Shape {
            dtype,
            domain_axes: DomainAxes::new(),
            dims: SmallVec::new(),
        }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// The number of elements this shape holds, or `None` when there is no
    /// such count. See [`elements_in`].
    pub fn num_elements(&self) -> Option<usize> {
        elements_in(&self.dims)
    }

    /// The same count, saying why there is none. For the callers that have an
    /// error to return rather than an `Option` to thread.
    pub fn element_count(&self) -> Result<usize, String> {
        self.num_elements().ok_or_else(|| {
            format!(
                "shape {:?} of {} holds no countable elements",
                self.dims,
                self.dtype.name()
            )
        })
    }

    /// The bytes this shape occupies: its elements in its own dtype.
    pub fn byte_size(&self) -> Result<usize, String> {
        let elements = self.element_count()?;
        elements
            .checked_mul(self.dtype.size_bytes())
            .ok_or_else(|| {
                format!(
                    "shape {:?} of {} is past what a byte extent states",
                    self.dims,
                    self.dtype.name()
                )
            })
    }

    pub fn with_domain_axes(mut self, domain_axes: impl IntoIterator<Item = FiniteSetId>) -> Shape {
        self.domain_axes = domain_axes.into_iter().collect();
        self
    }

    pub fn with_dims(&self, dims: impl IntoIterator<Item = i64>) -> Shape {
        Shape {
            dtype: self.dtype,
            domain_axes: self.domain_axes.clone(),
            dims: dims.into_iter().collect(),
        }
    }

    /// Same dims, different element type.
    pub fn with_dtype(&self, dtype: DType) -> Shape {
        Shape {
            dtype,
            domain_axes: self.domain_axes.clone(),
            dims: self.dims.clone(),
        }
    }
}

/// Helper to build `[i64; N]`-style dims at call sites: `dims![128, 256]`.
#[macro_export]
macro_rules! dims {
    ($($d:expr),* $(,)?) => { $crate::shape::Dims::from_slice(&[$($d as i64),*]) };
}

/// Error returned by an op's shape inference / operand type-check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeError {
    pub op: &'static str,
    pub msg: String,
}

impl ShapeError {
    pub fn new(op: &'static str, msg: impl Into<String>) -> ShapeError {
        ShapeError {
            op,
            msg: msg.into(),
        }
    }
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.op, self.msg)
    }
}

impl std::error::Error for ShapeError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32s(dims: impl IntoIterator<Item = i64>) -> Shape {
        Shape::new(DType::F32, dims)
    }

    /// The empty product is the identity, so a rank-0 shape holds exactly one
    /// element — the thing a scalar buffer stores.
    #[test]
    fn a_scalar_holds_one_element() {
        assert_eq!(Shape::scalar(DType::F32).num_elements(), Some(1));
        assert_eq!(f32s([]).num_elements(), Some(1));
        assert!(Shape::scalar(DType::F32).is_scalar());
    }

    /// A zero dimension is not an absent dimension: the shape holds nothing.
    #[test]
    fn a_zero_dimension_holds_no_elements() {
        assert_eq!(f32s([0]).num_elements(), Some(0));
        assert_eq!(f32s([7, 0, 3]).num_elements(), Some(0));
        assert_eq!(f32s([0, 0]).num_elements(), Some(0));
    }

    /// The distinction the two rules above draw, stated as the difference it
    /// makes: `f32[0]` is empty and rank-1, a scalar is one element and rank-0.
    #[test]
    fn an_empty_shape_is_not_a_scalar() {
        let empty = f32s([0]);
        let scalar = Shape::scalar(DType::F32);
        assert_ne!(empty.num_elements(), scalar.num_elements());
        assert_eq!((empty.rank(), scalar.rank()), (1, 0));
        assert!(!empty.is_scalar());
    }

    #[test]
    fn a_populated_shape_is_the_product_of_its_dimensions() {
        assert_eq!(f32s([2, 3, 5]).num_elements(), Some(30));
        assert_eq!(f32s([1]).num_elements(), Some(1));
    }

    /// A count no buffer could hold is refused rather than wrapped into a
    /// smaller one that a caller would allocate and a launch would run past.
    /// The bound is `usize` — the count a caller allocates against — not the
    /// `i64` the dimensions happen to be written in.
    #[test]
    fn an_uncountable_shape_has_no_count() {
        assert_eq!(f32s([1 << 40, 1 << 40]).num_elements(), None);
        assert_eq!(f32s([1 << 30, 1 << 30, 1 << 30]).num_elements(), None);
        let error = f32s([1 << 40, 1 << 40]).element_count().unwrap_err();
        assert!(error.contains("holds no countable elements"), "{error}");
    }

    /// A count that fits but whose bytes do not is refused where the bytes are
    /// asked for, not where the elements are.
    #[test]
    fn a_countable_shape_can_still_have_no_byte_extent() {
        let enormous = f32s([i64::MAX, 2]);
        assert!(enormous.num_elements().is_some(), "the count fits a usize");
        assert!(enormous.byte_size().is_err(), "four times it does not");
    }

    /// A shape that holds nothing holds nothing however its dimensions are
    /// ordered. Multiplying left to right would have overflowed on the way to
    /// the zero in one of these and not the other.
    #[test]
    fn an_empty_shape_is_empty_in_every_order() {
        let dims = [0, i64::MAX, 2];
        for rotation in 0..dims.len() {
            let mut ordered = dims;
            ordered.rotate_left(rotation);
            assert_eq!(
                f32s(ordered).num_elements(),
                Some(0),
                "{ordered:?} holds nothing"
            );
        }
        assert_eq!(f32s([i64::MAX, 0]).num_elements(), Some(0));
        assert_eq!(f32s([0, i64::MAX]).num_elements(), Some(0));
    }

    /// A negative dimension is not a dimension. It is refused rather than cast,
    /// so it can never become the very large `usize` its bits spell.
    #[test]
    fn a_negative_dimension_has_no_count() {
        assert_eq!(f32s([-1]).num_elements(), None);
        assert_eq!(f32s([4, -2]).num_elements(), None);
        // Not even beside a zero, which would otherwise answer first.
        assert_eq!(f32s([0, -1]).num_elements(), None);
        assert_eq!(f32s([-1, 0]).num_elements(), None);
    }

    /// The byte extent is the elements in the shape's own dtype, and it
    /// inherits every refusal above.
    #[test]
    fn a_byte_size_is_the_count_in_the_shapes_own_dtype() {
        assert_eq!(f32s([2, 3]).byte_size().unwrap(), 24);
        assert_eq!(Shape::new(DType::F64, [2, 3]).byte_size().unwrap(), 48);
        assert_eq!(f32s([0]).byte_size().unwrap(), 0);
        assert_eq!(Shape::scalar(DType::F32).byte_size().unwrap(), 4);
        assert!(f32s([-1]).byte_size().is_err());
        assert!(f32s([i64::MAX, 2]).byte_size().is_err());
    }

    /// The free form and the method are the same rule.
    #[test]
    fn the_shape_asks_the_same_question_the_dimensions_do() {
        for dims in [vec![], vec![0], vec![2, 3], vec![-1], vec![i64::MAX, 2]] {
            assert_eq!(f32s(dims.clone()).num_elements(), elements_in(&dims));
        }
    }
}
