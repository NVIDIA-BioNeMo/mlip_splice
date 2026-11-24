// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// pyo3 0.22's `#[pyfunction]` macro expands to code that trips
// `clippy::useless_conversion` on `?`-returning bodies, and (under
// edition 2024) `unsafe_op_in_unsafe_fn` because the macro injects
// unsafe calls into the body of the user-marked function without
// wrapping them. Both lints fire on macro-generated code where
// `#[allow]` on the function doesn't reach, so suppress at module scope.
#![allow(clippy::useless_conversion, unsafe_op_in_unsafe_fn)]

use std::ffi::c_void;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

mod handler;
mod registry;
pub use handler::{BoundSide, RunnableRecipe, run_recipe, run_recipe_core};

pub unsafe fn capsule_for_fn(py: Python<'_>, fn_ptr: *mut c_void) -> PyResult<PyObject> {
    let cap = unsafe { pyo3::ffi::PyCapsule_New(fn_ptr, std::ptr::null(), None) };
    if cap.is_null() {
        return Err(PyErr::fetch(py));
    }
    Ok(unsafe { PyObject::from_owned_ptr(py, cap) })
}

#[pyfunction]
fn run_recipe_capsule(py: Python<'_>) -> PyResult<PyObject> {
    unsafe { capsule_for_fn(py, run_recipe as *mut c_void) }
}

/// The executable recipe a Python caller passed, without its Python metadata.
fn recipe_blob(data: &Bound<'_, PyAny>) -> PyResult<recipe::Blob> {
    recipe_py::executable_blob(data).map_err(|_| {
        PyRuntimeError::new_err(
            "recipe must be bytes or SerializedRecipe, or support bytes(recipe)",
        )
    })
}

/// Returns `(identity, workspace_nbytes)`.
///
/// The identity is what a compiled program carries to name this recipe, and it
/// is the recipe's content digest scoped to its tuning policy, not an address.
/// It is stable across processes. The loaded recipe itself stays here,
/// found again by that name at execution time.
///
/// `scheduling_time_limit_ms` limits opening new candidates from the start of
/// autotuning, not completion of in-flight work. `None` measures every candidate;
/// zero selects a feasible fallback without measurement. Values must be finite,
/// nonnegative, and representable as a duration, or `ValueError` is raised.
#[pyfunction]
#[pyo3(signature = (data, persist_to=None, tune_report_to=None, scheduling_time_limit_ms=None))]
fn load_recipe(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    persist_to: Option<std::path::PathBuf>,
    tune_report_to: Option<std::path::PathBuf>,
    scheduling_time_limit_ms: Option<f64>,
) -> PyResult<(Py<PyBytes>, usize)> {
    kiln::TuningPolicy {
        scheduling_time_limit_ms,
    }
    .scheduling_time_limit()
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let recipe = if let Ok(held) = data.extract::<PyRef<'_, recipe_py::Recipe>>() {
        held.shared_executable()
    } else {
        let spilled = recipe_blob(data)?;
        std::sync::Arc::new(
            recipe::deserialize_shared(spilled)
                .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?,
        )
    };
    let workspace_nbytes = recipe::algebra::workspace_upper_bound(&recipe);
    let identity = kiln::RecipeIdentity::of(&recipe)
        .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?
        .with_tuning_policy(&kiln::TuningPolicy {
            scheduling_time_limit_ms,
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    registry::load(
        identity,
        RunnableRecipe::Pending {
            source: recipe,
            persist_to,
            report_to: tune_report_to,
            scheduling_time_limit_ms,
        },
    )
    .map_err(PyRuntimeError::new_err)?;
    Ok((
        PyBytes::new_bound(py, identity.as_bytes()).unbind(),
        workspace_nbytes,
    ))
}

#[pyfunction]
fn tuned_recipe_bytes(py: Python<'_>, identity: &[u8]) -> PyResult<Option<Py<PyBytes>>> {
    let handle = registry::resolve(identity).map_err(PyRuntimeError::new_err)?;
    let bytes = handler::tuned_recipe_bytes(&handle)
        .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;
    Ok(bytes.map(|bytes| PyBytes::new_bound(py, &bytes).unbind()))
}

/// How replays of every loaded recipe were served, process-wide.
///
/// `captured` counts replays a caller recorded into a graph of its own instead
/// of running; `replayed` counts launches of our own exec; `repointed` counts
/// how many of those first had to point it at moved buffers. Read against the
/// number of steps, the first says whether a caller that records this handler
/// is replaying its recording or re-making it, and the last two say how stable
/// the addresses it binds are. None of it is observable any other way — the
/// cheap outcomes are cheap because they call nothing.
#[pyfunction]
fn replay_stats(py: Python<'_>) -> PyResult<PyObject> {
    let (captured, replayed, repointed) = kiln::stats::counts();
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("captured", captured)?;
    dict.set_item("replayed", replayed)?;
    dict.set_item("repointed", repointed)?;
    Ok(dict.unbind().into())
}

#[pyfunction]
fn reset_replay_stats() {
    kiln::stats::reset();
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_recipe_capsule, m)?)?;
    m.add_function(wrap_pyfunction!(load_recipe, m)?)?;
    m.add_function(wrap_pyfunction!(tuned_recipe_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(replay_stats, m)?)?;
    m.add_function(wrap_pyfunction!(reset_replay_stats, m)?)?;
    Ok(())
}

#[cfg(test)]
mod scheduling_time_limit_tests {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::types::PyDict;

    #[test]
    fn python_keyword_rejects_invalid_limits_before_decoding_recipe() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let load = pyo3::wrap_pyfunction_bound!(load_recipe, py).unwrap();
            for ms in [
                -1.0,
                -f64::from_bits(1),
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::MAX,
            ] {
                let kwargs = PyDict::new_bound(py);
                kwargs.set_item("scheduling_time_limit_ms", ms).unwrap();
                let error = load.call((py.None(),), Some(&kwargs)).unwrap_err();
                assert!(error.is_instance_of::<PyValueError>(py), "{error}");
                assert!(error.to_string().contains("scheduling_time_limit_ms"));
            }
        });
    }

    #[test]
    fn loaded_identity_separates_runtime_policies_without_a_device() {
        let source = recipe::Recipe::stating_edges(
            recipe::RecipeSignature::new(vec![4], vec![4]),
            recipe::CompileRequests::default(),
            recipe::RecipeBody::new(
                1,
                1,
                Vec::new(),
                Vec::new(),
                vec![recipe::Node::new(
                    recipe::Work::Memset {
                        destination: recipe::WritableBuf::Output(0).at(0),
                        value: 0,
                        nbytes: 4,
                    },
                    [],
                )],
            ),
        )
        .unwrap();
        let data = recipe::serialize(&source).unwrap();
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let load = pyo3::wrap_pyfunction_bound!(load_recipe, py).unwrap();
            let key = |limit: Option<f64>| {
                let kwargs = PyDict::new_bound(py);
                kwargs.set_item("scheduling_time_limit_ms", limit).unwrap();
                let (identity, _): (Vec<u8>, usize) = load
                    .call((PyBytes::new_bound(py, &data),), Some(&kwargs))
                    .unwrap()
                    .extract()
                    .unwrap();
                identity
            };
            assert_ne!(key(None), key(Some(0.0)));
            assert_ne!(key(Some(0.0)), key(Some(10.0)));
            assert_eq!(key(Some(10.0)), key(Some(10.0)));
        });
    }

    #[test]
    fn python_keyword_accepts_unlimited_zero_and_fractional_limits() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let load = pyo3::wrap_pyfunction_bound!(load_recipe, py).unwrap();
            for ms in [None, Some(0.0), Some(-0.0), Some(0.001), Some(300_000.0)] {
                let kwargs = PyDict::new_bound(py);
                kwargs.set_item("scheduling_time_limit_ms", ms).unwrap();
                let error = load.call((py.None(),), Some(&kwargs)).unwrap_err();
                // Reaching recipe decoding proves the keyword and limit were accepted.
                assert!(error.is_instance_of::<PyRuntimeError>(py), "{error}");
                assert!(error.to_string().contains("recipe must be bytes"));
            }
        });
    }
}
