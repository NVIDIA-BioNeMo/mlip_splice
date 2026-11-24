// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Float-constant bit patterns for PTX immediates.

/// PTX f32 immediate: `0f` + IEEE bits.
pub(crate) fn f32_bits(x: f64) -> String {
    format!("0f{:08X}", (x as f32).to_bits())
}

/// PTX f64 immediate: `0d` + IEEE bits.
pub(crate) fn f64_bits(x: f64) -> String {
    format!("0d{:016X}", x.to_bits())
}

/// f64 → IEEE binary16 bits, round-to-nearest-even (via f32).
pub(crate) fn f16_bits(x: f64) -> u16 {
    let b = (x as f32).to_bits();
    let (sign, exp, man) = (b >> 31, ((b >> 23) & 0xff) as i32, b & 0x7f_ffff);
    let sign16 = (sign as u16) << 15;
    if exp == 0xff {
        // inf/nan
        return sign16 | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign16 | 0x7c00; // overflow → inf
    }
    if e <= 0 {
        if e < -10 {
            return sign16; // underflow → signed zero
        }
        // subnormal: shift in the implicit bit, round to nearest even
        let man = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rounded = (man + half - 1 + ((man >> shift) & 1)) >> shift;
        return sign16 | rounded as u16;
    }
    let half = 0x0fff + ((man >> 13) & 1);
    let rounded = ((((e as u32) << 23) | man) + half) >> 13;
    sign16 | rounded as u16
}

/// f64 → bfloat16 bits, round-to-nearest-even (truncate f32 with rounding).
pub(crate) fn bf16_bits(x: f64) -> u16 {
    let b = (x as f32).to_bits();
    if b & 0x7f80_0000 == 0x7f80_0000 && b & 0x7f_ffff != 0 {
        return ((b >> 16) | 0x40) as u16; // quiet the nan
    }
    (((b as u64 + 0x7fff + ((b >> 16) & 1) as u64) >> 16) & 0xffff) as u16
}
