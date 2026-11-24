// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::marker::PhantomData;

use xla_ffi_sys::{XLA_FFI_Buffer, XLA_FFI_DataType};
#[cfg(test)]
use xla_ffi_sys::{XLA_FFI_Buffer_STRUCT_SIZE, XLA_FFI_DataType_F32};

use crate::dtype::{Dtype, dtype_size};
use crate::error::{Error, Result};

pub struct AnyBuffer<'a> {
    raw: &'a XLA_FFI_Buffer,
}

impl<'a> AnyBuffer<'a> {
    pub(crate) fn from_raw(raw: &'a XLA_FFI_Buffer) -> Self {
        Self { raw }
    }

    pub fn dtype(&self) -> XLA_FFI_DataType {
        self.raw.dtype
    }

    pub fn rank(&self) -> usize {
        self.raw.rank as usize
    }

    pub fn dims(&self) -> &'a [i64] {
        // SAFETY: XLA guarantees `dims` is valid for `rank` entries for
        // the call's lifetime. `rank == 0` would dereference a null
        // pointer, so handle that case explicitly.
        if self.raw.rank == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.raw.dims, self.raw.rank as usize) }
        }
    }

    pub fn data(&self) -> *mut std::ffi::c_void {
        self.raw.data
    }

    /// The elements XLA says this buffer holds, or why there is no such count.
    pub fn num_elements(&self) -> Result<usize> {
        elements_in(self.dims())
    }

    /// The bytes XLA says this buffer holds — the size a handler is entitled to
    /// touch, and the one it must check a recipe's port against.
    pub fn size_bytes(&self) -> Result<usize> {
        let esz = dtype_size(self.raw.dtype).ok_or_else(|| {
            Error::invalid_argument(format!("unsupported dtype {}", self.raw.dtype))
        })?;
        let elements = self.num_elements()?;
        elements.checked_mul(esz).ok_or_else(|| {
            Error::invalid_argument(format!(
                "buffer {:?} of {esz}-byte elements is past what a size states",
                self.dims()
            ))
        })
    }
}

/// The elements a buffer of `dims` holds, or why there is no such count.
///
/// No dimensions is the empty product — one element, the scalar XLA passes for
/// a rank-0 value. A zero dimension makes the buffer empty, and that is a real
/// `0`; it is answered before anything is multiplied, so an empty buffer has
/// the same count however its dimensions are ordered. A negative dimension is
/// refused before it can be cast into the enormous `usize` its bits spell, and
/// a product past `usize` is refused rather than wrapped into a smaller size
/// that a handler would then trust.
fn elements_in(dims: &[i64]) -> Result<usize> {
    if dims.iter().any(|&d| d < 0) {
        return Err(Error::invalid_argument(format!(
            "buffer has negative dimension(s) {dims:?}"
        )));
    }
    if dims.contains(&0) {
        return Ok(0);
    }
    dims.iter()
        .try_fold(1usize, |n, &d| {
            usize::try_from(d).ok().and_then(|d| n.checked_mul(d))
        })
        .ok_or_else(|| {
            Error::invalid_argument(format!(
                "buffer {dims:?} holds more elements than usize counts"
            ))
        })
}

pub struct Buffer<'a, T: Dtype> {
    raw: &'a XLA_FFI_Buffer,
    _phantom: PhantomData<T>,
}

impl<'a, T: Dtype> Buffer<'a, T> {
    pub(crate) fn from_raw(raw: &'a XLA_FFI_Buffer) -> Result<Self> {
        if raw.dtype != T::TAG {
            return Err(Error::invalid_argument(format!(
                "expected buffer of type {} (tag {}), got tag {}",
                T::NAME,
                T::TAG,
                raw.dtype
            )));
        }
        Ok(Self {
            raw,
            _phantom: PhantomData,
        })
    }

    pub fn rank(&self) -> usize {
        self.raw.rank as usize
    }

    pub fn dims(&self) -> &'a [i64] {
        if self.raw.rank == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.raw.dims, self.raw.rank as usize) }
        }
    }

    pub fn ptr(&self) -> *mut T {
        self.raw.data as *mut T
    }

    /// The elements XLA says this buffer holds, by the same rule
    /// [`AnyBuffer::num_elements`] applies.
    pub fn num_elements(&self) -> Result<usize> {
        elements_in(self.dims())
    }

    /// The bytes XLA says this buffer holds, in the element type it was typed
    /// with.
    pub fn size_bytes(&self) -> Result<usize> {
        self.num_elements()?
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "buffer {:?} of {} is past what a size states",
                    self.dims(),
                    T::NAME
                ))
            })
    }
}

pub struct Ret<T>(pub T);

impl<T> Ret<T> {}

impl<T> std::ops::Deref for Ret<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_buffer(dims: &mut [i64]) -> XLA_FFI_Buffer {
        XLA_FFI_Buffer {
            struct_size: XLA_FFI_Buffer_STRUCT_SIZE as usize,
            extension_start: std::ptr::null_mut(),
            dtype: XLA_FFI_DataType_F32,
            data: std::ptr::null_mut(),
            rank: dims.len() as i64,
            dims: if dims.is_empty() {
                std::ptr::null_mut()
            } else {
                dims.as_mut_ptr()
            },
        }
    }

    #[test]
    fn scalar_buffer_has_one_element() {
        let mut dims = [];
        let raw = raw_buffer(&mut dims);
        let b = AnyBuffer::from_raw(&raw);
        assert_eq!(b.num_elements().unwrap(), 1);
        assert_eq!(b.size_bytes().unwrap(), 4);
    }

    #[test]
    fn a_populated_buffer_is_the_product_of_its_dimensions() {
        let mut dims = [2, 3, 5];
        let raw = raw_buffer(&mut dims);
        let b = AnyBuffer::from_raw(&raw);
        assert_eq!(b.num_elements().unwrap(), 30);
        assert_eq!(b.size_bytes().unwrap(), 120);
    }

    #[test]
    fn zero_extent_buffer_has_zero_bytes() {
        let mut dims = [2, 0, 3];
        let raw = raw_buffer(&mut dims);
        let b = AnyBuffer::from_raw(&raw);
        assert_eq!(b.num_elements().unwrap(), 0);
        assert_eq!(b.size_bytes().unwrap(), 0);
    }

    /// An empty buffer is empty however its dimensions arrive. Multiplying left
    /// to right would have overflowed on the way to the zero in one of these
    /// orders and not the other, and a wrapped size is one a handler trusts.
    #[test]
    fn an_empty_buffer_is_empty_in_every_order() {
        for dims in [[0, i64::MAX, 2], [i64::MAX, 2, 0], [2, 0, i64::MAX]] {
            let mut dims = dims;
            let raw = raw_buffer(&mut dims);
            let b = AnyBuffer::from_raw(&raw);
            assert_eq!(b.num_elements().unwrap(), 0, "{dims:?}");
            assert_eq!(b.size_bytes().unwrap(), 0, "{dims:?}");
        }
    }

    /// A negative dimension is refused before the cast that would turn it into
    /// an enormous accessible size.
    #[test]
    fn a_negative_dimension_has_no_size() {
        for dims in [[-1, 4], [4, -1], [0, -1]] {
            let mut dims = dims;
            let raw = raw_buffer(&mut dims);
            let b = AnyBuffer::from_raw(&raw);
            assert!(b.num_elements().is_err(), "{dims:?}");
            assert!(b.size_bytes().is_err(), "{dims:?}");
        }
    }

    /// A size past what `usize` states is refused rather than wrapped into a
    /// smaller one. This is the whole point: the returned size is what a caller
    /// checks a recipe port against, so a wrapped one admits an overrun.
    #[test]
    fn a_size_past_the_countable_is_refused_not_wrapped() {
        let mut dims = [1 << 40, 1 << 40];
        let raw = raw_buffer(&mut dims);
        let b = AnyBuffer::from_raw(&raw);
        assert!(b.num_elements().is_err());
        assert!(b.size_bytes().is_err());

        // Countable elements whose bytes are not.
        let mut dims = [i64::MAX, 2];
        let raw = raw_buffer(&mut dims);
        let b = AnyBuffer::from_raw(&raw);
        assert!(b.num_elements().is_ok());
        assert!(b.size_bytes().is_err(), "four times it does not fit");
    }

    /// The typed view answers the same questions the untyped one does.
    #[test]
    fn a_typed_buffer_counts_what_the_untyped_one_does() {
        let mut dims = [2, 3];
        let raw = raw_buffer(&mut dims);
        let typed = Buffer::<f32>::from_raw(&raw).unwrap();
        assert_eq!(typed.num_elements().unwrap(), 6);
        assert_eq!(typed.size_bytes().unwrap(), 24);
        assert_eq!(
            typed.size_bytes().unwrap(),
            AnyBuffer::from_raw(&raw).size_bytes().unwrap()
        );
    }
}
