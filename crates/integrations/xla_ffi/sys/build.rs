// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::PathBuf;

fn main() {
    // The runtime variable, not `env!`: a build script compiled once in one
    // worktree would otherwise carry that worktree's path into every later
    // build, and look for the header where it no longer is.
    let header = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("vendor/c_api.h");
    println!("cargo:rerun-if-changed={}", header.display());

    // libclang on Debian/Ubuntu doesn't ship resource headers; point it at
    // GCC's so stddef.h / stdint.h resolve. Same workaround pjrt-sys uses.
    let mut builder = bindgen::Builder::default().header(header.to_str().unwrap());
    if let Some(inc) = find_gcc_include() {
        builder = builder.clang_arg(format!("-I{}", inc.display()));
    }
    let bindings = builder
        .allowlist_type("XLA_FFI_.*")
        .allowlist_function("XLA_FFI_.*")
        .allowlist_var("XLA_FFI_.*")
        .derive_default(true)
        .prepend_enum_name(false)
        .layout_tests(false)
        .generate()
        .expect("bindgen failed");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings.write_to_file(&out).expect("write bindings");
}

fn find_gcc_include() -> Option<PathBuf> {
    let triples = ["x86_64-linux-gnu", "aarch64-linux-gnu"];
    for triple in triples {
        let base = PathBuf::from(format!("/usr/lib/gcc/{triple}"));
        let Ok(rd) = std::fs::read_dir(&base) else {
            continue;
        };
        let mut versions: Vec<PathBuf> = rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.join("include/stddef.h").exists())
            .collect();
        versions.sort();
        if let Some(latest) = versions.pop() {
            return Some(latest.join("include"));
        }
    }
    None
}
