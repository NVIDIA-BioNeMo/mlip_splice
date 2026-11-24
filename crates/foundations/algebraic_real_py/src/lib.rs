#![allow(clippy::useless_conversion, unsafe_op_in_unsafe_fn)]
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyComplex, PyFloat, PyInt, PyTuple};

use algebraic_real::{AlgebraicReal as Exact, AlgebraicRealData};

const MAX_EXACT_POWER_EXPONENT: u64 = 1_000_000;
const MAX_DECIMAL_DIGITS: u64 = 1_000_000;

#[pyclass(frozen, name = "AlgebraicReal", module = "algebraic_real")]
#[derive(Clone)]
pub struct AlgebraicReal {
    inner: Exact,
}

impl AlgebraicReal {
    pub fn wrap(inner: Exact) -> Self {
        AlgebraicReal { inner }
    }

    pub fn exact(&self) -> &Exact {
        &self.inner
    }
}

fn rational_parts(obj: &Bound<'_, PyAny>) -> Option<PyResult<BigRational>> {
    let num = obj.getattr("numerator").ok()?;
    let den = obj.getattr("denominator").ok()?;
    Some((|| {
        let num: BigInt = num.extract()?;
        let den: BigInt = den.extract()?;
        if den.is_zero() {
            return Err(PyZeroDivisionError::new_err(
                "AlgebraicReal(): zero denominator",
            ));
        }
        Ok(BigRational::new(num, den))
    })())
}

/// A binary-operator operand after coercion: an exact value we can compute
/// with, or a type we don't understand — for which the arithmetic dunders
/// return `NotImplemented` so Python falls back to the reflected operation.
enum Operand {
    Exact(Exact),
    NotImplemented,
}

fn operand(obj: &Bound<'_, PyAny>) -> PyResult<Operand> {
    if let Ok(s) = obj.extract::<AlgebraicReal>() {
        return Ok(Operand::Exact(s.inner));
    }
    if let Some(q) = rational_parts(obj) {
        return Ok(Operand::Exact(Exact::rational(q?)));
    }
    // A float or complex would silently lose exactness (via `__float__`), so
    // reject it loudly rather than deferring to its reflected op.
    if obj.is_instance_of::<PyFloat>() || obj.is_instance_of::<PyComplex>() {
        return Err(PyTypeError::new_err(format!(
            "AlgebraicReal(): expected AlgebraicReal, int, or Fraction; got {}",
            obj.get_type().name()?,
        )));
    }
    Ok(Operand::NotImplemented)
}

pub fn coerce(obj: &Bound<'_, PyAny>) -> PyResult<Exact> {
    match operand(obj)? {
        Operand::Exact(e) => Ok(e),
        Operand::NotImplemented => Err(PyTypeError::new_err(format!(
            "AlgebraicReal(): expected AlgebraicReal, int, or Fraction; got {}",
            obj.get_type().name()?,
        ))),
    }
}

/// Run an exact binary op, returning `NotImplemented` (rather than raising)
/// when the operand is a type we don't handle, per the Python numeric protocol.
fn dispatch(
    py: Python<'_>,
    other: &Bound<'_, PyAny>,
    f: impl FnOnce(Exact) -> PyResult<Exact>,
) -> PyResult<PyObject> {
    match operand(other)? {
        Operand::Exact(other) => Ok(Bound::new(py, AlgebraicReal::wrap(f(other)?))?
            .into_any()
            .unbind()),
        Operand::NotImplemented => Ok(py.NotImplemented()),
    }
}

fn divide(numerator: &Exact, denominator: &Exact) -> PyResult<Exact> {
    numerator
        .checked_div(denominator)
        .ok_or_else(|| PyZeroDivisionError::new_err("division by zero"))
}

fn fraction<'py>(py: Python<'py>, value: &BigRational) -> PyResult<Bound<'py, PyAny>> {
    py.import_bound("fractions")?
        .getattr("Fraction")?
        .call1((value.numer().clone(), value.denom().clone()))
}

fn cmp_result(op: CompareOp, sign: i32) -> bool {
    match op {
        CompareOp::Eq => sign == 0,
        CompareOp::Ne => sign != 0,
        CompareOp::Lt => sign < 0,
        CompareOp::Le => sign <= 0,
        CompareOp::Gt => sign > 0,
        CompareOp::Ge => sign >= 0,
    }
}

fn exact_power_exponent(exponent: &BigInt) -> PyResult<()> {
    let magnitude = exponent.magnitude().to_u64().ok_or_else(|| {
        PyOverflowError::new_err(format!(
            "pow(): exponent too large (limit: {MAX_EXACT_POWER_EXPONENT})"
        ))
    })?;
    if magnitude > MAX_EXACT_POWER_EXPONENT {
        return Err(PyOverflowError::new_err(format!(
            "pow(): exponent too large (limit: {MAX_EXACT_POWER_EXPONENT})"
        )));
    }
    Ok(())
}

fn check_power_exponent(base: &Exact, exponent: &BigInt) -> PyResult<()> {
    let bounded_result = base.as_fraction().is_some_and(|value| {
        value.is_zero() || value.is_one() || value == BigRational::from_integer(BigInt::from(-1))
    });
    if !exponent.is_zero() && !bounded_result {
        exact_power_exponent(exponent)?;
    }
    Ok(())
}

/// `base**n` for integer `n`, with Python's resource cap skipped only for the
/// identities whose result size cannot grow.
fn pow_int(base: &Exact, exponent: &BigInt) -> PyResult<AlgebraicReal> {
    check_power_exponent(base, exponent)?;
    base.pow_integer(exponent)
        .map(AlgebraicReal::wrap)
        .map_err(algebraic_error)
}

fn pow10(k: u64) -> BigInt {
    BigInt::from(10).pow(k as u32)
}

/// Round to the nearest integer, ties to even (Python's `round` convention).
fn round_to_int(inner: &Exact) -> BigInt {
    let one = BigInt::from(1);
    let two = BigInt::from(2);
    let f = inner.floor();
    let half = Exact::rational(BigRational::new(&f * &two + &one, two.clone()));
    match (inner - &half).signum() {
        s if s < 0 => f,
        s if s > 0 => f + &one,
        _ if (&f % &two).is_zero() => f,
        _ => f + &one,
    }
}

#[pymethods]
impl AlgebraicReal {
    #[new]
    #[pyo3(signature = (value=None))]
    fn new(value: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        match value {
            None => Ok(AlgebraicReal::wrap(Exact::zero())),
            Some(v) => Ok(AlgebraicReal::wrap(coerce(v)?)),
        }
    }

    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        dispatch(py, other, |other| Ok(&self.inner + &other))
    }

    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        dispatch(py, other, |other| Ok(&other + &self.inner))
    }

    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        dispatch(py, other, |other| Ok(&self.inner - &other))
    }

    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        dispatch(py, other, |other| Ok(&other - &self.inner))
    }

    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        dispatch(py, other, |other| Ok(&self.inner * &other))
    }

    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        dispatch(py, other, |other| Ok(&other * &self.inner))
    }

    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        dispatch(py, other, |other| divide(&self.inner, &other))
    }

    fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        dispatch(py, other, |other| divide(&other, &self.inner))
    }

    fn __pow__(
        &self,
        py: Python<'_>,
        exponent: &Bound<'_, PyAny>,
        modulo: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        if !modulo.is_none() {
            return Err(PyTypeError::new_err("pow(): modulus not supported"));
        }
        if exponent.downcast::<PyInt>().is_ok() {
            let n: BigInt = exponent.extract()?;
            return Ok(Bound::new(py, pow_int(&self.inner, &n)?)?
                .into_any()
                .unbind());
        }

        if exponent.is_instance_of::<PyFloat>() || exponent.is_instance_of::<PyComplex>() {
            return Err(PyTypeError::new_err(
                "pow(): float and complex exponents are not exact",
            ));
        }
        if let Some(exponent) = rational_parts(exponent) {
            let exponent = exponent?;
            check_power_exponent(&self.inner, exponent.numer())?;
            let value = self
                .inner
                .pow_rational(exponent.numer(), exponent.denom().magnitude())
                .map_err(algebraic_error)?;
            return Ok(Bound::new(py, AlgebraicReal::wrap(value))?
                .into_any()
                .unbind());
        }
        Ok(py.NotImplemented())
    }

    fn __neg__(&self) -> AlgebraicReal {
        AlgebraicReal::wrap(-&self.inner)
    }

    fn __pos__(&self) -> AlgebraicReal {
        self.clone()
    }

    fn __abs__(&self) -> AlgebraicReal {
        AlgebraicReal::wrap(self.inner.abs())
    }

    fn __floor__(&self) -> BigInt {
        self.inner.floor()
    }

    fn __ceil__(&self) -> BigInt {
        self.inner.ceil()
    }

    fn __trunc__(&self) -> BigInt {
        self.inner.trunc()
    }

    #[pyo3(signature = (ndigits=None))]
    fn __round__(&self, py: Python<'_>, ndigits: Option<&Bound<'_, PyAny>>) -> PyResult<PyObject> {
        // No digits: the nearest integer (ties to even).
        let Some(ndigits) = ndigits else {
            return Ok(round_to_int(&self.inner).into_py(py));
        };
        let k: BigInt = ndigits.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "round(): integer ndigits required; got {}",
                ndigits
                    .get_type()
                    .name()
                    .map(|name| name.to_string())
                    .unwrap_or_else(|_| "unknown".to_owned()),
            ))
        })?;
        let magnitude = k.magnitude().to_u64().ok_or_else(|| {
            PyOverflowError::new_err(format!(
                "round(): ndigits too large (limit: {MAX_DECIMAL_DIGITS})"
            ))
        })?;
        if magnitude > MAX_DECIMAL_DIGITS {
            return Err(PyOverflowError::new_err(format!(
                "round(): ndigits too large (limit: {MAX_DECIMAL_DIGITS})"
            )));
        }
        let k = k
            .to_i64()
            .expect("bounded decimal digit count must fit in i64");
        // Otherwise the nearest multiple of 10**-k, as an exact AlgebraicReal.
        let shift = pow10(magnitude);
        let value = if k >= 0 {
            let scaled = &self.inner * &Exact::integer(shift.clone());
            Exact::rational(BigRational::new(round_to_int(&scaled), shift))
        } else {
            let scaled = divide(&self.inner, &Exact::integer(shift.clone()))?;
            Exact::integer(round_to_int(&scaled) * shift)
        };
        Ok(Bound::new(py, AlgebraicReal::wrap(value))?
            .into_any()
            .unbind())
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(PyObject, PyObject)> {
        let data = data_object(py, self.inner.to_data())?;
        let schema: PyObject = (1u8, data).into_py(py);
        let ctor = RECONSTRUCT
            .get(py)
            .expect("algebraic_real_from_data is registered")
            .clone_ref(py);
        Ok((ctor, (schema,).into_py(py)))
    }

    fn __bool__(&self) -> bool {
        !self.inner.is_zero()
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<PyObject> {
        if let Ok(value) = other.extract::<f64>()
            && other.is_instance_of::<PyFloat>()
        {
            if value.is_nan() {
                return Ok(matches!(op, CompareOp::Ne).into_py(py));
            }
            let sign = if value == f64::INFINITY {
                -1
            } else if value == f64::NEG_INFINITY {
                1
            } else {
                let rational = BigRational::from_float(value).expect("finite f64 is rational");
                (&self.inner - &Exact::rational(rational)).signum()
            };
            return Ok(cmp_result(op, sign).into_py(py));
        }
        match operand(other) {
            Ok(Operand::Exact(o)) => {
                let sign = match self.inner.cmp(&o) {
                    std::cmp::Ordering::Less => -1,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                };
                Ok(cmp_result(op, sign).into_py(py))
            }
            Ok(Operand::NotImplemented) => Ok(py.NotImplemented()),
            Err(error) => Err(error),
        }
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        // A rational value must hash like the equal int/Fraction/float so they
        // share a dict/set bucket; delegate to Python's numeric hash.
        if let Some(q) = self.inner.as_fraction() {
            let fraction = py.import_bound("fractions")?.getattr("Fraction")?;
            return fraction
                .call1((q.numer().clone(), q.denom().clone()))?
                .hash();
        }
        // An irrational value never compares equal to a plain number, so a
        // structural hash of the canonical form is enough.
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        let h = hasher.finish() as isize;
        Ok(if h == -1 { -2 } else { h })
    }

    fn __float__(&self) -> f64 {
        self.inner.to_f64()
    }

    fn sign(&self) -> i32 {
        self.inner.signum()
    }

    fn as_fraction<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let q = self
            .inner
            .as_fraction()
            .ok_or_else(|| PyValueError::new_err("as_fraction(): value is irrational"))?;
        fraction(py, &q)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        let (polynomial, root_index) = self.inner.minimal_polynomial_and_root_index();
        let polynomial = polynomial
            .into_iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!("AlgebraicReal.from_polynomial([{polynomial}], {root_index})")
    }

    #[classmethod]
    fn from_polynomial(
        _class: &Bound<'_, pyo3::types::PyType>,
        coefficients: &Bound<'_, PyAny>,
        root_index: usize,
    ) -> PyResult<Self> {
        let coefficients = coefficients
            .iter()
            .map_err(|_| PyTypeError::new_err("coefficients must be iterable"))?
            .map(|value| {
                let value = value?;
                rational_parts(&value).ok_or_else(|| {
                    PyTypeError::new_err("polynomial coefficients must be int or Fraction")
                })?
            })
            .collect::<PyResult<Vec<_>>>()?;
        Exact::from_polynomial(coefficients, root_index)
            .map(AlgebraicReal::wrap)
            .map_err(algebraic_error)
    }

    fn degree(&self) -> usize {
        self.inner.degree()
    }

    fn minimal_polynomial(&self) -> Vec<BigInt> {
        self.inner.minimal_polynomial()
    }

    fn trace<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        fraction(py, &self.inner.trace())
    }

    fn norm<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        fraction(py, &self.inner.norm())
    }

    fn _algebraic_real_data(&self, py: Python<'_>) -> PyResult<PyObject> {
        data_object(py, self.inner.to_data())
    }
}

fn algebraic_error(error: algebraic_real::AlgebraicError) -> PyErr {
    match error {
        algebraic_real::AlgebraicError::DivisionByZero => {
            PyZeroDivisionError::new_err(error.to_string())
        }
        _ => PyValueError::new_err(error.to_string()),
    }
}

fn data_object(py: Python<'_>, data: AlgebraicRealData) -> PyResult<PyObject> {
    let object = match data {
        AlgebraicRealData::Multiquadratic(terms) => {
            let terms = terms
                .into_iter()
                .map(|(n, d, r)| (n.to_string(), d.to_string(), r.to_string()))
                .collect::<Vec<_>>();
            ("m", terms).into_py(py)
        }
        AlgebraicRealData::Polynomial {
            coefficients,
            root_index,
        } => {
            let coefficients = coefficients
                .into_iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>();
            ("p", coefficients, root_index).into_py(py)
        }
    };
    Ok(object)
}

#[pyfunction]
fn algebraic_real_from_data(schema: &Bound<'_, PyTuple>) -> PyResult<AlgebraicReal> {
    decode_data(schema).map(AlgebraicReal::wrap)
}

fn decode_data(schema: &Bound<'_, PyTuple>) -> PyResult<Exact> {
    let version: u8 = schema.get_item(0)?.extract()?;
    if version != 1 {
        return Err(PyValueError::new_err("unsupported AlgebraicReal schema"));
    }
    let data = schema.get_item(1)?.downcast_into::<PyTuple>()?;
    let tag: String = data.get_item(0)?.extract()?;
    let encoded = match tag.as_str() {
        "m" => {
            let terms: Vec<(String, String, String)> = data.get_item(1)?.extract()?;
            AlgebraicRealData::Multiquadratic(
                terms
                    .into_iter()
                    .map(|(n, d, r)| {
                        Ok((
                            n.parse().map_err(|_| {
                                PyValueError::new_err("invalid algebraic numerator")
                            })?,
                            d.parse().map_err(|_| {
                                PyValueError::new_err("invalid algebraic denominator")
                            })?,
                            r.parse()
                                .map_err(|_| PyValueError::new_err("invalid algebraic radicand"))?,
                        ))
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            )
        }
        "p" => {
            let coefficients: Vec<String> = data.get_item(1)?.extract()?;
            let root_index = data.get_item(2)?.extract()?;
            AlgebraicRealData::Polynomial {
                coefficients: coefficients
                    .into_iter()
                    .map(|value| {
                        value
                            .parse()
                            .map_err(|_| PyValueError::new_err("invalid algebraic coefficient"))
                    })
                    .collect::<PyResult<Vec<_>>>()?,
                root_index,
            }
        }
        _ => return Err(PyValueError::new_err("unsupported AlgebraicReal data")),
    };
    Exact::from_data(encoded).map_err(algebraic_error)
}

/// Convert through the owning module rather than sharing a PyClass across dynamic libraries.
pub fn to_python(py: Python<'_>, value: Exact) -> PyResult<PyObject> {
    let data = data_object(py, value.to_data())?;
    let schema: PyObject = (1u8, data).into_py(py);
    Ok(py
        .import_bound("algebraic_real._native")?
        .getattr("algebraic_real_from_data")?
        .call1((schema,))?
        .unbind())
}

/// Read the versioned exact value representation across an extension-module boundary.
pub fn from_python(obj: &Bound<'_, PyAny>) -> PyResult<Exact> {
    let py = obj.py();
    let value = py
        .import_bound("algebraic_real")?
        .getattr("AlgebraicReal")?
        .call1((obj,))?;
    let data = value.call_method0("_algebraic_real_data")?;
    let schema = PyTuple::new_bound(py, [1u8.into_py(py), data.unbind()]);
    decode_data(&schema)
}

/// The reconstruction function object, captured at registration so
/// `__reduce__` can hand it to pickle without naming the host module.
static RECONSTRUCT: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

/// Mount `AlgebraicReal` and its pickle reconstructor into the public extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AlgebraicReal>()?;
    let reconstruct = wrap_pyfunction!(algebraic_real_from_data, m)?;
    m.add_function(reconstruct.clone())?;
    let _ = RECONSTRUCT.set(m.py(), reconstruct.into_any().unbind());
    Ok(())
}
