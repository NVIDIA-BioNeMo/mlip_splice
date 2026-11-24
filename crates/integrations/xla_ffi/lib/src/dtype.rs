// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use xla_ffi_sys::{
    XLA_FFI_DataType, XLA_FFI_DataType_BF16, XLA_FFI_DataType_C64, XLA_FFI_DataType_C128,
    XLA_FFI_DataType_F16, XLA_FFI_DataType_F32, XLA_FFI_DataType_F64, XLA_FFI_DataType_PRED,
    XLA_FFI_DataType_S8, XLA_FFI_DataType_S16, XLA_FFI_DataType_S32, XLA_FFI_DataType_S64,
    XLA_FFI_DataType_U8, XLA_FFI_DataType_U16, XLA_FFI_DataType_U32, XLA_FFI_DataType_U64,
};

pub trait Dtype: Copy + 'static {
    const TAG: XLA_FFI_DataType;
    const NAME: &'static str;
    const SIZE: usize = std::mem::size_of::<Self>();
}

macro_rules! impl_dtype {
    ($t:ty, $tag:ident, $name:literal) => {
        impl Dtype for $t {
            const TAG: XLA_FFI_DataType = $tag;
            const NAME: &'static str = $name;
        }
    };
}

impl_dtype!(bool, XLA_FFI_DataType_PRED, "bool/PRED");
impl_dtype!(i8, XLA_FFI_DataType_S8, "i8");
impl_dtype!(u8, XLA_FFI_DataType_U8, "u8");
impl_dtype!(i16, XLA_FFI_DataType_S16, "i16");
impl_dtype!(u16, XLA_FFI_DataType_U16, "u16");
impl_dtype!(i32, XLA_FFI_DataType_S32, "i32");
impl_dtype!(u32, XLA_FFI_DataType_U32, "u32");
impl_dtype!(i64, XLA_FFI_DataType_S64, "i64");
impl_dtype!(u64, XLA_FFI_DataType_U64, "u64");
impl_dtype!(f32, XLA_FFI_DataType_F32, "f32");
impl_dtype!(f64, XLA_FFI_DataType_F64, "f64");

pub fn dtype_size(dt: XLA_FFI_DataType) -> Option<usize> {
    Some(match dt {
        x if x == XLA_FFI_DataType_PRED => 1,
        x if x == XLA_FFI_DataType_S8 => 1,
        x if x == XLA_FFI_DataType_U8 => 1,
        x if x == XLA_FFI_DataType_S16 => 2,
        x if x == XLA_FFI_DataType_U16 => 2,
        x if x == XLA_FFI_DataType_F16 => 2,
        x if x == XLA_FFI_DataType_BF16 => 2,
        x if x == XLA_FFI_DataType_S32 => 4,
        x if x == XLA_FFI_DataType_U32 => 4,
        x if x == XLA_FFI_DataType_F32 => 4,
        x if x == XLA_FFI_DataType_S64 => 8,
        x if x == XLA_FFI_DataType_U64 => 8,
        x if x == XLA_FFI_DataType_F64 => 8,
        x if x == XLA_FFI_DataType_C64 => 8,
        x if x == XLA_FFI_DataType_C128 => 16,
        _ => return None,
    })
}
