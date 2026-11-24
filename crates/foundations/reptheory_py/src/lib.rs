// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::useless_conversion, unsafe_op_in_unsafe_fn)]

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};

use algebraic_real::AlgebraicReal as Exact;
use algebraic_real_py::{from_python as coerce, to_python};
use reptheory::exact::SparseTensor as ExactTensor;
use reptheory::rep::{CouplingIrrep, Irrep, Mat, Rep};
use reptheory::{Irreps as CoreIrreps, SignedPermGroup, So3Irrep, Su2Irrep, parse_formula};

/// A sparse tensor of exact Clebsch-Gordan coefficients over `algebraic_real`: it holds
/// only its nonzero entries, and accumulation prunes exact cancellations to zero.
#[pyclass(frozen, name = "SparseTensor", module = "e3")]
#[derive(Clone)]
pub struct SparseTensor {
    inner: ExactTensor<Exact>,
}

impl SparseTensor {
    pub fn wrap(inner: ExactTensor<Exact>) -> Self {
        SparseTensor { inner }
    }

    pub fn exact(&self) -> &ExactTensor<Exact> {
        &self.inner
    }
}

#[pymethods]
impl SparseTensor {
    #[new]
    #[pyo3(signature = (shape, entries=None))]
    fn new(shape: Vec<usize>, entries: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let mut tensor = ExactTensor::<Exact>::zeros(shape);
        if let Some(entries) = entries {
            for item in entries.iter()? {
                let (coords, value): (Vec<usize>, Bound<'_, PyAny>) = item?.extract()?;
                tensor.add(&coords, coerce(&value)?);
            }
        }
        Ok(SparseTensor { inner: tensor })
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        PyTuple::new_bound(py, self.inner.shape())
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.rank()
    }

    fn get(&self, py: Python<'_>, coords: Vec<usize>) -> PyResult<PyObject> {
        to_python(py, self.inner.get(&coords))
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let items = self
            .inner
            .nonzeros()
            .map(|(coordinates, value)| {
                let coordinates = PyTuple::new_bound(py, coordinates).into_any().unbind();
                let value = to_python(py, value.clone())?;
                Ok(PyTuple::new_bound(py, [coordinates, value])
                    .into_any()
                    .unbind())
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new_bound(py, items)
            .into_any()
            .iter()?
            .unbind()
            .into())
    }

    fn __len__(&self) -> usize {
        self.inner.nnz()
    }

    fn __repr__(&self) -> String {
        let shape: Vec<String> = self.inner.shape().iter().map(usize::to_string).collect();
        format!(
            "SparseTensor(shape=({}), nnz={})",
            shape.join(", "),
            self.inner.nnz()
        )
    }
}

fn sparse_to_mat(tensor: &ExactTensor<Exact>, da: usize, db: usize, dc: usize) -> Mat<Exact> {
    assert_eq!(tensor.shape(), [da, db, dc], "Clebsch-Gordan tensor shape");
    let mut mat = Mat::zeros(da * db, dc);
    for (coords, value) in tensor.nonzeros() {
        mat.set(coords[0] * db + coords[1], coords[2], value.clone());
    }
    mat
}

fn mat_to_sparse(mat: &Mat<Exact>, da: usize, db: usize, dc: usize) -> SparseTensor {
    let mut entries = Vec::new();
    for i in 0..da {
        for j in 0..db {
            for k in 0..dc {
                let value = mat.get(i * db + j, k);
                if !value.is_zero() {
                    entries.push((vec![i, j, k], value.clone()));
                }
            }
        }
    }
    SparseTensor::wrap(ExactTensor::from_entries(vec![da, db, dc], entries))
}

#[pyfunction]
fn _su2_clebsch_gordan(twice_j1: u32, twice_j2: u32, twice_j3: u32) -> Vec<SparseTensor> {
    let (a, b, c) = (
        Su2Irrep::new(twice_j1),
        Su2Irrep::new(twice_j2),
        Su2Irrep::new(twice_j3),
    );
    let (da, db, dc) = (a.dim(), b.dim(), c.dim());
    <Su2Irrep as Irrep>::clebsch_gordan(&a, &b, &c)
        .iter()
        .map(|mat| mat_to_sparse(mat, da, db, dc))
        .collect()
}

#[pyfunction]
fn _so3_clebsch_gordan(l1: u32, l2: u32, l3: u32) -> Vec<SparseTensor> {
    let (a, b, c) = (So3Irrep::new(l1), So3Irrep::new(l2), So3Irrep::new(l3));
    let (da, db, dc) = (a.dim(), b.dim(), c.dim());
    <So3Irrep as Irrep>::clebsch_gordan(&a, &b, &c)
        .iter()
        .map(|mat| mat_to_sparse(mat, da, db, dc))
        .collect()
}

type ClebschGordanTable = HashMap<(usize, usize, usize), Vec<Mat<Exact>>>;

static NEXT_SNAPSHOT_SERIAL: AtomicU64 = AtomicU64::new(0);

struct Snapshot {
    serial: u64,
    py_irreps: Vec<Py<PyAny>>,
    dims: Vec<usize>,
    trivial_id: usize,
    selection: HashMap<(usize, usize), Vec<usize>>,
    cg: ClebschGordanTable,
}

#[derive(Clone)]
struct TableIrrep {
    id: usize,
    table: Arc<Snapshot>,
}

impl PartialEq for TableIrrep {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && Arc::ptr_eq(&self.table, &other.table)
    }
}

impl Eq for TableIrrep {}

impl PartialOrd for TableIrrep {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TableIrrep {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.table.serial, self.id).cmp(&(other.table.serial, other.id))
    }
}

impl CouplingIrrep for TableIrrep {
    fn dimension(&self) -> usize {
        self.table.dims[self.id]
    }

    fn trivial(&self) -> Self {
        TableIrrep {
            id: self.table.trivial_id,
            table: self.table.clone(),
        }
    }

    fn tensor_product(&self, other: &Self) -> Vec<Self> {
        self.table
            .selection
            .get(&(self.id, other.id))
            .map(|ids| {
                ids.iter()
                    .map(|&id| TableIrrep {
                        id,
                        table: self.table.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn clebsch_gordan(a: &Self, b: &Self, c: &Self) -> Vec<Mat<Exact>> {
        a.table
            .cg
            .get(&(a.id, b.id, c.id))
            .cloned()
            .unwrap_or_default()
    }
}

fn lookup_id(id_of: &Bound<'_, PyDict>, ir: &Bound<'_, PyAny>) -> PyResult<usize> {
    id_of
        .get_item(ir)?
        .ok_or_else(|| PyValueError::new_err("irrep missing from the snapshot closure"))?
        .extract()
}

fn seed_closure<'py>(
    py: Python<'py>,
    group: &Bound<'py, PyAny>,
    inputs: &[Bound<'py, PyAny>],
    keep: Option<&[Bound<'py, PyAny>]>,
) -> PyResult<Bound<'py, PySet>> {
    let closure = PySet::empty_bound(py)?;
    closure.add(group.call_method0("trivial")?)?;
    for ir in inputs {
        closure.add(ir)?;
    }
    for ir in keep.into_iter().flatten() {
        closure.add(ir)?;
    }
    Ok(closure)
}

fn grow_closure<'py>(
    group: &Bound<'py, PyAny>,
    closure: &Bound<'py, PySet>,
    inputs: &[Bound<'py, PyAny>],
    max_factors: usize,
) -> PyResult<()> {
    for _ in 1..max_factors {
        let members: Vec<Bound<'py, PyAny>> = closure.iter().collect();
        let mut grew = false;
        for a in &members {
            for b in inputs {
                for c in group.call_method1("selection_rule", (a, b))?.iter()? {
                    let c = c?;
                    if !closure.contains(&c)? {
                        closure.add(&c)?;
                        grew = true;
                    }
                }
            }
        }
        if !grew {
            break;
        }
    }
    Ok(())
}

fn sorted_members<'py>(closure: &Bound<'py, PySet>) -> Vec<Bound<'py, PyAny>> {
    let mut members: Vec<_> = closure.iter().collect();
    members.sort_by(|a, b| {
        if a.lt(b).expect("irrep __lt__ failed") {
            std::cmp::Ordering::Less
        } else if b.lt(a).expect("irrep __lt__ failed") {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    members
}

type IndexedIrreps<'py> = (Bound<'py, PyDict>, Vec<Py<PyAny>>, Vec<usize>);

fn index_members<'py>(
    py: Python<'py>,
    members: &[Bound<'py, PyAny>],
) -> PyResult<IndexedIrreps<'py>> {
    let id_of = PyDict::new_bound(py);
    let mut py_irreps = Vec::with_capacity(members.len());
    let mut dims = Vec::with_capacity(members.len());
    for (id, ir) in members.iter().enumerate() {
        id_of.set_item(ir, id)?;
        dims.push(ir.getattr("dim")?.extract()?);
        py_irreps.push(ir.clone().unbind());
    }
    Ok((id_of, py_irreps, dims))
}

fn build_selection<'py>(
    group: &Bound<'py, PyAny>,
    members: &[Bound<'py, PyAny>],
    id_of: &Bound<'py, PyDict>,
) -> PyResult<HashMap<(usize, usize), Vec<usize>>> {
    let mut selection = HashMap::new();
    for (i, a) in members.iter().enumerate() {
        for (j, b) in members.iter().enumerate() {
            let mut ids = Vec::new();
            for c in group.call_method1("selection_rule", (a, b))?.iter()? {
                let c = c?;
                if let Some(cid) = id_of.get_item(&c)?.map(|v| v.extract()).transpose()? {
                    ids.push(cid);
                }
            }
            selection.insert((i, j), ids);
        }
    }
    Ok(selection)
}

fn minimum_factor_counts(
    member_count: usize,
    input_ids: &[usize],
    selection: &HashMap<(usize, usize), Vec<usize>>,
    max_factors: usize,
) -> Vec<usize> {
    let mut min_factors = vec![usize::MAX; member_count];
    for &i in input_ids {
        min_factors[i] = 1;
    }
    for _ in 1..max_factors {
        for a in 0..member_count {
            let Some(sum) = min_factors[a].checked_add(1) else {
                continue;
            };
            for &b in input_ids {
                for &c in &selection[&(a, b)] {
                    min_factors[c] = min_factors[c].min(sum);
                }
            }
        }
    }
    min_factors
}

fn build_clebsch_gordan<'py>(
    group: &Bound<'py, PyAny>,
    members: &[Bound<'py, PyAny>],
    dims: &[usize],
    selection: &HashMap<(usize, usize), Vec<usize>>,
    min_factors: &[usize],
    max_factors: usize,
) -> PyResult<ClebschGordanTable> {
    let mut cg = HashMap::new();
    for i in 0..members.len() {
        for j in 0..members.len() {
            if !matches!(min_factors[i].checked_add(min_factors[j]), Some(sum) if sum <= max_factors)
            {
                continue;
            }
            for &cid in &selection[&(i, j)] {
                let paths = group
                    .call_method1("clebsch_gordan", (&members[i], &members[j], &members[cid]))?;
                let mut mats = Vec::new();
                for path in paths.iter()? {
                    let tensor: SparseTensor = path?.extract()?;
                    mats.push(sparse_to_mat(tensor.exact(), dims[i], dims[j], dims[cid]));
                }
                if !mats.is_empty() {
                    cg.insert((i, j, cid), mats);
                }
            }
        }
    }
    Ok(cg)
}

fn build_snapshot<'py>(
    py: Python<'py>,
    group: &Bound<'py, PyAny>,
    inputs: &[Bound<'py, PyAny>],
    keep: Option<&[Bound<'py, PyAny>]>,
    max_factors: usize,
) -> PyResult<(Arc<Snapshot>, Bound<'py, PyDict>)> {
    let closure = seed_closure(py, group, inputs, keep)?;
    // The seeds are 1-factor products, so `max_factors`-factor products need
    // `max_factors - 1` growth rounds.
    grow_closure(group, &closure, inputs, max_factors)?;

    let members = sorted_members(&closure);
    let (id_of, py_irreps, dims) = index_members(py, &members)?;
    let trivial_id = lookup_id(&id_of, &group.call_method0("trivial")?)?;
    let selection = build_selection(group, &members, &id_of)?;

    // The fusion only takes Clebsch-Gordan of two products of inputs whose
    // factor counts sum to at most `max_factors`; keep-only members (reached by
    // no input product) never fuse. `min_factors` is the fewest inputs whose
    // product contains each member.
    let mut input_ids: Vec<usize> = Vec::new();
    for ir in inputs {
        input_ids.push(lookup_id(&id_of, ir)?);
    }
    let min_factors = minimum_factor_counts(members.len(), &input_ids, &selection, max_factors);
    let cg = build_clebsch_gordan(
        group,
        &members,
        &dims,
        &selection,
        &min_factors,
        max_factors,
    )?;

    let snapshot = Snapshot {
        serial: NEXT_SNAPSHOT_SERIAL.fetch_add(1, Ordering::Relaxed),
        py_irreps,
        dims,
        trivial_id,
        selection,
        cg,
    };
    Ok((Arc::new(snapshot), id_of))
}

fn table_irrep(
    snapshot: &Arc<Snapshot>,
    id_of: &Bound<'_, PyDict>,
    ir: &Bound<'_, PyAny>,
) -> PyResult<TableIrrep> {
    Ok(TableIrrep {
        id: lookup_id(id_of, ir)?,
        table: snapshot.clone(),
    })
}

fn blocks_to_py(
    py: Python<'_>,
    snapshot: &Snapshot,
    basis: reptheory::Basis<TableIrrep>,
) -> Vec<(PyObject, SparseTensor)> {
    basis
        .irreps
        .entries
        .into_iter()
        .zip(basis.segments)
        .map(|((_, ir), segment)| {
            (
                snapshot.py_irreps[ir.id].clone_ref(py),
                SparseTensor::wrap(segment.into_sparse_tensor()),
            )
        })
        .collect()
}

fn irreps_pairs<'py>(irreps: &Bound<'py, PyAny>) -> PyResult<Vec<(usize, Bound<'py, PyAny>)>> {
    irreps.call_method0("pairs")?.extract()
}

fn index_group(n: usize, formula: Option<&str>) -> Result<SignedPermGroup, String> {
    match formula {
        Some(s) => {
            let (indices, group) = parse_formula(s).map_err(|e| format!("formula {s:?}: {e}"))?;
            if indices.len() != n {
                return Err(format!(
                    "formula {s:?} has {} indices but per_index has {n}",
                    indices.len()
                ));
            }
            Ok(group)
        }
        None => Ok(SignedPermGroup::generate(n, [])),
    }
}

#[pyfunction]
#[pyo3(name = "reduced_tensor_product", signature = (per_index, keep=None, formula=None))]
fn reduced_tensor_product_py<'py>(
    py: Python<'py>,
    per_index: Vec<Bound<'py, PyAny>>,
    keep: Option<Vec<Bound<'py, PyAny>>>,
    formula: Option<&str>,
) -> PyResult<Vec<(PyObject, SparseTensor)>> {
    if per_index.is_empty() {
        return Err(PyValueError::new_err("per_index must be non-empty"));
    }
    let n = per_index.len();
    let group = index_group(n, formula).map_err(PyValueError::new_err)?;
    let py_group = per_index[0].getattr("group")?;

    let mut per_index_pairs = Vec::with_capacity(n);
    for irreps in &per_index {
        per_index_pairs.push(irreps_pairs(irreps)?);
    }
    let inputs: Vec<Bound<'py, PyAny>> = per_index_pairs
        .iter()
        .flat_map(|pairs| pairs.iter().map(|(_, ir)| ir.clone()))
        .collect();

    let (snapshot, id_of) = build_snapshot(py, &py_group, &inputs, keep.as_deref(), n)?;

    let per_index_t: Vec<CoreIrreps<TableIrrep>> = per_index_pairs
        .iter()
        .map(|pairs| {
            let entries: PyResult<Vec<(usize, TableIrrep)>> = pairs
                .iter()
                .map(|(mul, ir)| Ok((*mul, table_irrep(&snapshot, &id_of, ir)?)))
                .collect();
            Ok(CoreIrreps::new(entries?))
        })
        .collect::<PyResult<_>>()?;
    let keep_t = keep_table_irreps(&snapshot, &id_of, keep.as_deref())?;

    let basis = reptheory::reduced_tensor_product(&per_index_t, &group, keep_t.as_deref());
    Ok(blocks_to_py(py, &snapshot, basis))
}

#[pyfunction]
#[pyo3(name = "reduced_symmetric_tensor_product", signature = (irreps, degree, keep=None))]
fn reduced_symmetric_tensor_product_py<'py>(
    py: Python<'py>,
    irreps: Bound<'py, PyAny>,
    degree: usize,
    keep: Option<Vec<Bound<'py, PyAny>>>,
) -> PyResult<Vec<(PyObject, SparseTensor)>> {
    let py_group = irreps.getattr("group")?;
    let pairs = irreps_pairs(&irreps)?;
    let inputs: Vec<Bound<'py, PyAny>> = pairs.iter().map(|(_, ir)| ir.clone()).collect();

    let (snapshot, id_of) = build_snapshot(py, &py_group, &inputs, keep.as_deref(), degree)?;

    let entries: PyResult<Vec<(usize, TableIrrep)>> = pairs
        .iter()
        .map(|(mul, ir)| Ok((*mul, table_irrep(&snapshot, &id_of, ir)?)))
        .collect();
    let irreps_t = CoreIrreps::new(entries?);
    let keep_t = keep_table_irreps(&snapshot, &id_of, keep.as_deref())?;

    let basis = reptheory::reduced_symmetric_tensor_product(&irreps_t, degree, keep_t.as_deref());
    Ok(blocks_to_py(py, &snapshot, basis))
}

fn keep_table_irreps(
    snapshot: &Arc<Snapshot>,
    id_of: &Bound<'_, PyDict>,
    keep: Option<&[Bound<'_, PyAny>]>,
) -> PyResult<Option<Vec<TableIrrep>>> {
    keep.map(|ks| {
        ks.iter()
            .map(|ir| table_irrep(snapshot, id_of, ir))
            .collect::<PyResult<Vec<_>>>()
    })
    .transpose()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SparseTensor>()?;
    m.add_function(wrap_pyfunction!(_su2_clebsch_gordan, m)?)?;
    m.add_function(wrap_pyfunction!(_so3_clebsch_gordan, m)?)?;
    m.add_function(wrap_pyfunction!(reduced_tensor_product_py, m)?)?;
    m.add_function(wrap_pyfunction!(reduced_symmetric_tensor_product_py, m)?)?;
    Ok(())
}

// PyO3 tests link libpython because only the top-level extension enables
// `extension-module`.
#[cfg(test)]
mod tests {
    use super::index_group;
    use reptheory::{Irreps as CoreIrreps, O3Irrep, reduced_tensor_product};

    #[test]
    fn qk_scalar_routes() {
        let q: CoreIrreps<O3Irrep> = CoreIrreps::parse("0e+1o+2e").unwrap();
        let sph: CoreIrreps<O3Irrep> = CoreIrreps::parse("0e+1o+2e+3o").unwrap();
        let group = index_group(3, None).unwrap();
        assert!(group.generators.is_empty(), "no formula ⇒ identity group");
        let trivial = O3Irrep::new(0, 1);
        let basis = reduced_tensor_product(&[q.clone(), q, sph], &group, Some(&[trivial]));
        assert_eq!(basis.segments.len(), 1);
        let seg = &basis.segments[0];
        assert_eq!(basis.irreps.iter().next().unwrap().1, trivial);
        assert_eq!(seg.mul, 13);
        assert_eq!(seg.input_dims, vec![9, 9, 16]);
    }

    #[test]
    fn formula_index_mismatch_errors() {
        assert!(index_group(2, Some("ijk=jik")).is_err());
        assert!(index_group(3, Some("ijk=jik")).is_ok());
    }
}
