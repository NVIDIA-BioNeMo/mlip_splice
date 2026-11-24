// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::useless_conversion, unsafe_op_in_unsafe_fn)]

use std::path::PathBuf;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyfunction]
fn add_lib_dir(path: &str) -> PyResult<()> {
    cuda_toolchain::add_lib_dir(PathBuf::from(path)).map_err(PyRuntimeError::new_err)
}

#[pyfunction]
fn lib_dirs() -> Vec<String> {
    paths_to_strings(cuda_toolchain::lib_dirs())
}

fn paths_to_strings(paths: Vec<PathBuf>) -> Vec<String> {
    paths
        .into_iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect()
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(add_lib_dir, module)?)?;
    module.add_function(wrap_pyfunction!(lib_dirs, module)?)?;
    Ok(())
}
