// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub use libloading::{Library, Symbol};

pub unsafe fn sym<'a, T>(lib: &'a Library, name: &[u8]) -> Symbol<'a, T> {
    unsafe { lib.get::<T>(name) }
        .unwrap_or_else(|_| panic!("Failed to load symbol: {}", String::from_utf8_lossy(name)))
}

#[macro_export]
macro_rules! load_sym {
    ($lib:expr, $name:literal, $ty:ty) => {{
        let name = concat!($name, "\0").as_bytes();
        // SAFETY: the caller-supplied signature must match the symbol's ABI.
        unsafe { *$crate::sym::<$ty>($lib, name) }
    }};
}
