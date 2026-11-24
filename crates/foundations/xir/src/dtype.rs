// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Logical element types. These carry no CUDA spelling;
//! they describe the *mathematical* element type of a tensor value.

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    Bf16,
    F32,
    F64,
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Complex64,
    Complex128,
}

impl DType {
    /// Short mnemonic used by the text printer (`f32`, `bf16`, `c64`, ...).
    pub fn name(self) -> &'static str {
        use DType::*;
        match self {
            F16 => "f16",
            Bf16 => "bf16",
            F32 => "f32",
            F64 => "f64",
            Bool => "i1",
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            U8 => "u8",
            U16 => "u16",
            U32 => "u32",
            U64 => "u64",
            Complex64 => "c64",
            Complex128 => "c128",
        }
    }

    pub fn size_bytes(self) -> usize {
        use DType::*;
        match self {
            Bool | I8 | U8 => 1,
            F16 | Bf16 | I16 | U16 => 2,
            F32 | I32 | U32 => 4,
            F64 | I64 | U64 | Complex64 => 8,
            Complex128 => 16,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::F16 | DType::Bf16 | DType::F32 | DType::F64)
    }

    pub fn is_complex(self) -> bool {
        matches!(self, DType::Complex64 | DType::Complex128)
    }

    pub fn is_integer(self) -> bool {
        use DType::*;
        matches!(self, I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64)
    }

    pub fn is_signed_integer(self) -> bool {
        matches!(self, DType::I8 | DType::I16 | DType::I32 | DType::I64)
    }

    pub fn is_bool(self) -> bool {
        self == DType::Bool
    }
}
