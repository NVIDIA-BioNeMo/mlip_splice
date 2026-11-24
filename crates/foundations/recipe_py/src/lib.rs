// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::useless_conversion, unsafe_op_in_unsafe_fn)]

use std::collections::BTreeSet;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};

#[pyclass(frozen, module = "loom._native")]
#[derive(Clone)]
pub struct SerializedRecipe {
    bytes: recipe::Blob,
}

impl SerializedRecipe {
    pub fn from_blob(bytes: recipe::Blob) -> Self {
        SerializedRecipe { bytes }
    }

    pub fn blob(&self) -> recipe::Blob {
        self.bytes.clone()
    }
}

#[pymethods]
impl SerializedRecipe {
    fn __bytes__(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new_bound(py, &self.bytes).unbind()
    }

    fn __len__(&self) -> usize {
        self.bytes.len()
    }

    fn __repr__(&self) -> String {
        format!("SerializedRecipe({} bytes)", self.bytes.len())
    }
}

/// An element type: its name and how many bytes one value occupies.
#[pyclass(frozen, module = "loom._native")]
#[derive(Clone)]
pub struct Dtype {
    name: String,
    nbytes: usize,
}

impl Dtype {
    pub fn from_parts(name: impl Into<String>, nbytes: usize) -> PyResult<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(PyValueError::new_err("a dtype name is nonempty"));
        }
        if nbytes == 0 {
            return Err(PyValueError::new_err("a dtype occupies at least one byte"));
        }
        Ok(Dtype { name, nbytes })
    }
}

#[pymethods]
impl Dtype {
    #[new]
    fn new(name: String, nbytes: usize) -> PyResult<Self> {
        Dtype::from_parts(name, nbytes)
    }

    /// Return the type's name.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// Return how many bytes one value of this type occupies.
    #[getter]
    fn nbytes(&self) -> usize {
        self.nbytes
    }

    fn __eq__(&self, other: &Dtype) -> bool {
        self.name == other.name && self.nbytes == other.nbytes
    }

    /// State this type as its constructor arguments, so it survives pickling.
    fn __reduce__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let class = py.get_type_bound::<Dtype>();
        let args = PyTuple::new_bound(py, [self.name.to_object(py), self.nbytes.to_object(py)]);
        Ok(
            PyTuple::new_bound(py, [class.into_any().unbind(), args.into_any().unbind()])
                .into_any()
                .unbind(),
        )
    }

    fn __repr__(&self) -> String {
        format!("Dtype({:?}, {} bytes)", self.name, self.nbytes)
    }
}

/// One named operand of a recipe: its name, element type, and shape.
#[pyclass(frozen, module = "loom._native")]
#[derive(Clone)]
pub struct Port {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
}

impl Port {
    pub fn from_parts(name: impl Into<String>, dtype: Dtype, shape: Vec<usize>) -> PyResult<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(PyValueError::new_err("a port name is nonempty"));
        }
        let port = Port { name, dtype, shape };
        if port.byte_extent().is_none() {
            return Err(PyValueError::new_err("port byte extent overflows"));
        }
        Ok(port)
    }

    fn byte_extent(&self) -> Option<usize> {
        self.shape
            .iter()
            .try_fold(1usize, |size, &axis| size.checked_mul(axis))?
            .checked_mul(self.dtype.nbytes)
    }
}

#[pymethods]
impl Port {
    #[new]
    fn new(name: String, dtype: Dtype, shape: Vec<usize>) -> PyResult<Self> {
        Port::from_parts(name, dtype, shape)
    }

    /// Return the port's name.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// Return the port's element type.
    #[getter]
    fn dtype(&self) -> Dtype {
        self.dtype.clone()
    }

    /// Return the port's shape, axis by axis.
    #[getter]
    fn shape(&self, py: Python<'_>) -> Py<PyTuple> {
        PyTuple::new_bound(py, self.shape.iter().copied()).unbind()
    }

    fn __eq__(&self, other: &Port) -> bool {
        self.name == other.name && self.dtype.name == other.dtype.name && self.shape == other.shape
    }

    /// State this port as its constructor arguments, so it survives pickling.
    fn __reduce__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let class = py.get_type_bound::<Port>();
        let args = PyTuple::new_bound(
            py,
            [
                self.name.to_object(py),
                Py::new(py, self.dtype.clone())?.into_any(),
                PyTuple::new_bound(py, self.shape.iter().copied())
                    .into_any()
                    .unbind(),
            ],
        );
        Ok(
            PyTuple::new_bound(py, [class.into_any().unbind(), args.into_any().unbind()])
                .into_any()
                .unbind(),
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "Port({:?}, {}, {:?})",
            self.name, self.dtype.name, self.shape
        )
    }
}

/// An executable recipe: its compiled payload and the ports it binds.
#[pyclass(frozen, module = "loom._native")]
#[derive(Clone)]
pub struct Recipe {
    executable: Arc<recipe::Recipe>,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
}

impl Recipe {
    pub fn from_executable(
        executable: recipe::Recipe,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> PyResult<Self> {
        check_ports(&inputs, &outputs, executable.contract())?;
        Ok(Recipe {
            executable: Arc::new(executable),
            inputs,
            outputs,
        })
    }

    pub fn executable(&self) -> &recipe::Recipe {
        &self.executable
    }

    pub fn shared_executable(&self) -> Arc<recipe::Recipe> {
        Arc::clone(&self.executable)
    }
}

#[pymethods]
impl Recipe {
    #[new]
    #[pyo3(signature = (executable, inputs, outputs))]
    fn new(executable: &[u8], inputs: Vec<Port>, outputs: Vec<Port>) -> PyResult<Self> {
        let executable = decode_executable(executable)?;
        Recipe::from_executable(executable, inputs, outputs)
    }

    /// Return the recipe's input ports, in binding order.
    #[getter]
    fn inputs(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new_bound(
            py,
            self.inputs
                .iter()
                .cloned()
                .map(|port| Py::new(py, port))
                .collect::<PyResult<Vec<_>>>()?,
        )
        .unbind())
    }

    /// Return the recipe's output ports, in binding order.
    #[getter]
    fn outputs(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new_bound(
            py,
            self.outputs
                .iter()
                .cloned()
                .map(|port| Py::new(py, port))
                .collect::<PyResult<Vec<_>>>()?,
        )
        .unbind())
    }

    /// Return which input port feeds each output port that aliases it.
    #[getter]
    fn aliases(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let held = PyDict::new_bound(py);
        for (input, output) in self.executable.contract().aliases().iter() {
            held.set_item(input, output)?;
        }
        Ok(py
            .import_bound("types")?
            .getattr("MappingProxyType")?
            .call1((held,))?
            .unbind())
    }

    /// Return how many workspace bytes running the recipe needs.
    #[getter]
    fn workspace_nbytes(&self) -> usize {
        recipe::algebra::workspace_upper_bound(&self.executable)
    }

    fn __bytes__(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let executable = recipe::serialize(&self.executable).map_err(wire_error)?;
        let metadata = encode_ports(&self.inputs, &self.outputs)?;
        let mut out = Vec::with_capacity(16 + executable.len() + metadata.len());
        out.extend_from_slice(b"CPRP");
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        push_u32(&mut out, executable.len())?;
        push_u32(&mut out, metadata.len())?;
        out.extend_from_slice(&executable);
        out.extend_from_slice(&metadata);
        Ok(PyBytes::new_bound(py, &out).unbind())
    }

    /// State this recipe as its constructor arguments, so it survives pickling.
    ///
    /// A tuned recipe is worth keeping across processes, which is what an
    /// ahead-of-time cache written to disk is for. `__new__` takes the
    /// serialized executable and the ports separately, so this states them
    /// that way rather than through `__bytes__`, which bundles the two.
    fn __reduce__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let class = py.get_type_bound::<Recipe>();
        let executable = recipe::serialize(&self.executable).map_err(wire_error)?;
        let args = PyTuple::new_bound(
            py,
            [
                PyBytes::new_bound(py, &executable).into_any().unbind(),
                self.inputs(py)?.into_any(),
                self.outputs(py)?.into_any(),
            ],
        );
        Ok(
            PyTuple::new_bound(py, [class.into_any().unbind(), args.into_any().unbind()])
                .into_any()
                .unbind(),
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "Recipe({} input(s), {} output(s), {} workspace bytes)",
            self.inputs.len(),
            self.outputs.len(),
            recipe::algebra::workspace_upper_bound(&self.executable)
        )
    }
}

/// Deserialize a recipe from its self-describing bytes.
#[pyfunction]
pub fn decode(data: &[u8]) -> PyResult<Recipe> {
    let mut at = 0usize;
    let magic = take(data, &mut at, 4)?;
    if magic != b"CPRP" {
        return Err(PyValueError::new_err("wrong recipe envelope magic"));
    }
    let version = u16::from_le_bytes(take(data, &mut at, 2)?.try_into().unwrap());
    if version != 1 {
        return Err(PyValueError::new_err(format!(
            "unsupported recipe envelope version {version}"
        )));
    }
    take(data, &mut at, 2)?;
    let executable_len = read_u32(data, &mut at)?;
    let metadata_len = read_u32(data, &mut at)?;
    let executable = take(data, &mut at, executable_len)?;
    let metadata = take(data, &mut at, metadata_len)?;
    if at != data.len() {
        return Err(PyValueError::new_err("trailing recipe envelope bytes"));
    }
    let executable = decode_executable(executable)?;
    let (inputs, outputs) = decode_ports(metadata)?;
    Recipe::from_executable(executable, inputs, outputs)
}

fn check_ports(
    inputs: &[Port],
    outputs: &[Port],
    signature: &recipe::RecipeSignature,
) -> PyResult<()> {
    unique("input", inputs)?;
    unique("output", outputs)?;
    check_side("input", inputs, signature.inputs())?;
    check_side("output", outputs, signature.outputs())?;
    for (input, output) in signature.aliases().iter() {
        let left = &inputs[input];
        let right = &outputs[output];
        if left.dtype.name != right.dtype.name
            || left.dtype.nbytes != right.dtype.nbytes
            || left.shape != right.shape
        {
            return Err(PyValueError::new_err(format!(
                "alias input {input} and output {output} have different dtype or shape"
            )));
        }
    }
    Ok(())
}

fn unique(side: &str, ports: &[Port]) -> PyResult<()> {
    let mut names = BTreeSet::new();
    for port in ports {
        if !names.insert(port.name.as_str()) {
            return Err(PyValueError::new_err(format!(
                "duplicate {side} port name {:?}",
                port.name
            )));
        }
    }
    Ok(())
}

fn check_side(side: &str, ports: &[Port], extents: &[usize]) -> PyResult<()> {
    if ports.len() != extents.len() {
        return Err(PyValueError::new_err(format!(
            "{side} metadata has {} port(s), executable has {}",
            ports.len(),
            extents.len()
        )));
    }
    for (index, (port, &extent)) in ports.iter().zip(extents).enumerate() {
        if port.byte_extent() != Some(extent) {
            return Err(PyValueError::new_err(format!(
                "{side} {index} metadata is {} bytes, executable states {extent}",
                port.byte_extent().unwrap_or(usize::MAX)
            )));
        }
    }
    Ok(())
}

fn encode_ports(inputs: &[Port], outputs: &[Port]) -> PyResult<Vec<u8>> {
    let mut out = Vec::new();
    encode_side(&mut out, inputs)?;
    encode_side(&mut out, outputs)?;
    Ok(out)
}

fn encode_side(out: &mut Vec<u8>, ports: &[Port]) -> PyResult<()> {
    push_u32(out, ports.len())?;
    for port in ports {
        encode_string(out, &port.name)?;
        encode_string(out, &port.dtype.name)?;
        push_u32(out, port.dtype.nbytes)?;
        push_u32(out, port.shape.len())?;
        for &axis in &port.shape {
            out.extend_from_slice(
                &u64::try_from(axis)
                    .map_err(|_| PyValueError::new_err("shape axis does not fit u64"))?
                    .to_le_bytes(),
            );
        }
    }
    Ok(())
}

fn decode_ports(data: &[u8]) -> PyResult<(Vec<Port>, Vec<Port>)> {
    let mut at = 0;
    let inputs = decode_side(data, &mut at)?;
    let outputs = decode_side(data, &mut at)?;
    if at != data.len() {
        return Err(PyValueError::new_err("trailing port metadata"));
    }
    Ok((inputs, outputs))
}

fn decode_side(data: &[u8], at: &mut usize) -> PyResult<Vec<Port>> {
    let count = read_u32(data, at)?;
    let mut ports = Vec::with_capacity(count);
    for _ in 0..count {
        let name = decode_string(data, at)?;
        let dtype_name = decode_string(data, at)?;
        let nbytes = read_u32(data, at)?;
        let rank = read_u32(data, at)?;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            let axis = u64::from_le_bytes(take(data, at, 8)?.try_into().unwrap());
            shape.push(
                usize::try_from(axis)
                    .map_err(|_| PyValueError::new_err("shape axis does not fit usize"))?,
            );
        }
        ports.push(Port {
            name,
            dtype: Dtype {
                name: dtype_name,
                nbytes,
            },
            shape,
        });
    }
    Ok(ports)
}

fn encode_string(out: &mut Vec<u8>, value: &str) -> PyResult<()> {
    push_u32(out, value.len())?;
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn decode_string(data: &[u8], at: &mut usize) -> PyResult<String> {
    let len = read_u32(data, at)?;
    String::from_utf8(take(data, at, len)?.to_vec())
        .map_err(|_| PyValueError::new_err("port metadata is not UTF-8"))
}

fn push_u32(out: &mut Vec<u8>, value: usize) -> PyResult<()> {
    out.extend_from_slice(
        &u32::try_from(value)
            .map_err(|_| PyValueError::new_err("recipe envelope field exceeds u32"))?
            .to_le_bytes(),
    );
    Ok(())
}

fn read_u32(data: &[u8], at: &mut usize) -> PyResult<usize> {
    Ok(u32::from_le_bytes(take(data, at, 4)?.try_into().unwrap()) as usize)
}

fn take<'a>(data: &'a [u8], at: &mut usize, len: usize) -> PyResult<&'a [u8]> {
    let end = at
        .checked_add(len)
        .ok_or_else(|| PyValueError::new_err("recipe envelope length overflows"))?;
    let held = data
        .get(*at..end)
        .ok_or_else(|| PyValueError::new_err("truncated recipe envelope"))?;
    *at = end;
    Ok(held)
}

pub fn executable_blob(data: &Bound<'_, PyAny>) -> PyResult<recipe::Blob> {
    if let Ok(serialized) = data.extract::<PyRef<'_, SerializedRecipe>>() {
        return Ok(serialized.blob());
    }
    let owned;
    let bytes = match data.extract::<&[u8]>() {
        Ok(bytes) => bytes,
        Err(_) => {
            owned = data
                .call_method0("__bytes__")
                .and_then(|bytes| bytes.extract::<Vec<u8>>())
                .map_err(|_| {
                    PyValueError::new_err(
                        "recipe must be bytes, SerializedRecipe, or support bytes(recipe)",
                    )
                })?;
            &owned
        }
    };
    let executable = envelope_executable(bytes)?;
    recipe::Blob::spilled(executable)
        .map_err(|error| PyValueError::new_err(format!("recipe spill failed: {error}")))
}

fn envelope_executable(data: &[u8]) -> PyResult<&[u8]> {
    if !data.starts_with(b"CPRP") {
        return Ok(data);
    }
    let mut at = 4usize;
    let version = u16::from_le_bytes(take(data, &mut at, 2)?.try_into().unwrap());
    if version != 1 {
        return Err(PyValueError::new_err(format!(
            "unsupported recipe envelope version {version}"
        )));
    }
    take(data, &mut at, 2)?;
    let executable_len = read_u32(data, &mut at)?;
    let metadata_len = read_u32(data, &mut at)?;
    let executable = take(data, &mut at, executable_len)?;
    take(data, &mut at, metadata_len)?;
    if at != data.len() {
        return Err(PyValueError::new_err("trailing recipe envelope bytes"));
    }
    Ok(executable)
}

fn decode_executable(bytes: &[u8]) -> PyResult<recipe::Recipe> {
    let source = recipe::Blob::spilled(bytes)
        .map_err(|error| PyValueError::new_err(format!("recipe spill failed: {error}")))?;
    recipe::deserialize_shared(source).map_err(wire_error)
}

fn wire_error(error: recipe::WireError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SerializedRecipe>()?;
    module.add_class::<Dtype>()?;
    module.add_class::<Port>()?;
    module.add_class::<Recipe>()?;
    module.add_function(wrap_pyfunction!(decode, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_share_the_executable() {
        let executable = recipe::Recipe::stating_edges(
            recipe::RecipeSignature::new(Vec::new(), Vec::new()),
            recipe::CompileRequests::default(),
            recipe::RecipeBody::new(0, 0, Vec::new(), Vec::new(), Vec::new()),
        )
        .unwrap();
        let held = Recipe::from_executable(executable, Vec::new(), Vec::new()).unwrap();
        assert!(Arc::ptr_eq(
            &held.shared_executable(),
            &held.clone().shared_executable()
        ));
    }
}
