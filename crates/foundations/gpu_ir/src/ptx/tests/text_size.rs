// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte counts and reservations share one instruction-selection path.

use super::{kernel, ptr};
use crate::Scalar;
use crate::ptx::{measure_module, render_module, text_layout};

#[test]
fn an_empty_module_has_no_kernel_body_buffer() {
    for sm in [80, 90, 100, 103, 110, 120, 121] {
        let text = render_module(&[], sm).unwrap();
        let size = measure_module(&[], sm).unwrap();
        assert_eq!(text.len(), size.module_bytes);
        assert_eq!(text.capacity(), size.module_bytes);
        assert_eq!(size.module_bytes, size.peak_buffer_bytes);
    }
}

#[test]
fn body_buffers_are_reused_in_time_not_summed() {
    let mut first = kernel(|_| {});
    first.function.name = "first".into();
    let mut second = kernel(|b| {
        let out = ptr(b, "out", Scalar::F32);
        let value = b.float(Scalar::F32, 1.25);
        b.store(out, value);
    });
    second.function.name = "second".into();
    let launches = [first, second];
    let refs = launches.iter().collect::<Vec<_>>();
    let verified = crate::verified_module(&refs, 100).unwrap();
    let layout = text_layout(&verified);
    assert_eq!(layout.body_bytes.len(), 2);
    assert!(layout.body_bytes[1] > layout.body_bytes[0]);
    assert_eq!(
        layout.size().peak_buffer_bytes,
        layout.module_bytes + layout.body_bytes[1]
    );
    let text = render_module(&launches, 100).unwrap();
    assert_eq!(measure_module(&launches, 100).unwrap(), layout.size());
    assert_eq!(text.len(), layout.module_bytes);
    assert_eq!(text.capacity(), layout.module_bytes);
}

#[test]
fn constant_initializers_keep_their_exact_spelling() {
    let launch = kernel(|b| {
        let out = ptr(b, "out", Scalar::F32);
        let index = b.i32(0);
        let value = b.const_load("table", Scalar::F32, vec![0, 1, 254, 255], index);
        b.store(out, value);
    });
    for sm in [80, 90, 100, 103, 110, 120, 121] {
        let launches = std::slice::from_ref(&launch);
        let size = measure_module(launches, sm).unwrap();
        let text = render_module(launches, sm).unwrap();
        assert!(text.contains(".global .align 4 .b8 const_table[4] = {0x00, 0x01, 0xfe, 0xff};\n"));
        assert_eq!(size.module_bytes, text.len());
        assert_eq!(size.module_bytes, text.capacity());
    }
}

#[test]
fn counting_and_rendering_refuse_the_same_invalid_launch() {
    let mut launch = kernel(|_| {});
    launch.max_threads = 0;
    let launches = [launch];
    let counted = measure_module(&launches, 100).unwrap_err();
    let rendered = render_module(&launches, 100).unwrap_err();
    assert_eq!(counted.to_string(), rendered.to_string());
}
