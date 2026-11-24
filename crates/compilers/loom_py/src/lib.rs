// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::useless_conversion, unsafe_op_in_unsafe_fn)]

use domain_index::{AlignedView, Domain, IndexMap, Slot};
use loom::{CompileOptions, ElementOrder, Lanes, Operand, Program};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::num::NonZeroUsize;
use std::time::Duration;
use xir::dtype::DType;

type SlotRow = (String, String, String, String);
type IndexMapRow = (String, String, bool, bool, bool);
type OperandSpec = (Vec<i64>, i64, i32);
type ProfileRow = std::collections::BTreeMap<String, f64>;
type DiagnosticRows = (
    Vec<PyObject>,
    Vec<PyObject>,
    Vec<PyObject>,
    Vec<PyObject>,
    Vec<PyObject>,
    Vec<PyObject>,
    Vec<PyObject>,
    Vec<PyObject>,
    Vec<PyObject>,
);

fn dtype_from_id(id: i32) -> Result<DType, String> {
    Ok(match id {
        0 => DType::F32,
        1 => DType::F64,
        2 => DType::F16,
        3 => DType::Bf16,
        4 => DType::Bool,
        5 => DType::I8,
        6 => DType::I16,
        7 => DType::I32,
        8 => DType::I64,
        9 => DType::U8,
        10 => DType::U16,
        11 => DType::U32,
        12 => DType::U64,
        13 => DType::Complex64,
        14 => DType::Complex128,
        value => return Err(format!("unknown dtype id {value}")),
    })
}

#[allow(clippy::too_many_arguments)]
fn loom_recipe_bytes(
    name: &str,
    module: &xir_py::XirModule,
    u: i64,
    extents: &std::collections::HashMap<String, i64>,
    index_maps: &[IndexMapRow],
    operand_domains: &[Vec<Option<String>>],
    num_inputs: i32,
    operand_specs: &[OperandSpec],
    available_views: &[(String, String, String)],
    search_budget_ms: Option<f64>,
    shortlist_budget_ms: Option<f64>,
    max_kernel_statements: Option<usize>,
    max_algebraic_work: usize,
    max_algebraic_expression_work: usize,
    element_order: Option<String>,
    frontier_graph: Option<usize>,
    dump_ptx: bool,
    record_structure: bool,
    max_workspace_bytes: Option<usize>,
    search_memory_budget_bytes: Option<u64>,
) -> PyResult<(recipe::Blob, Vec<SlotRow>, ProfileRow, loom::CompileProfile)> {
    if num_inputs < 0 || num_inputs as usize > operand_specs.len() {
        return Err(PyRuntimeError::new_err(format!(
            "num_inputs={num_inputs} outside [0, operands={}]",
            operand_specs.len()
        )));
    }
    let num_axes = operand_domains.first().map(Vec::len).unwrap_or(0);
    if operand_domains.len() != operand_specs.len() {
        return Err(PyRuntimeError::new_err(format!(
            "operand_domains has {} rows, expected {}",
            operand_domains.len(),
            operand_specs.len()
        )));
    }
    if operand_domains.iter().any(|row| row.len() != num_axes) {
        return Err(PyRuntimeError::new_err(
            "operand_domains rows have inconsistent lengths",
        ));
    }

    let mut name_set = std::collections::BTreeSet::new();
    for row in operand_domains {
        name_set.extend(row.iter().flatten().cloned());
    }
    for (source, target, ..) in index_maps {
        name_set.insert(source.clone());
        name_set.insert(target.clone());
    }
    for (source, anchor, other) in available_views {
        name_set.insert(source.clone());
        name_set.insert(anchor.clone());
        name_set.insert(other.clone());
    }
    let names: Vec<String> = name_set.into_iter().collect();
    let ids: std::collections::HashMap<&str, i32> = names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i as i32))
        .collect();

    let mut domains = Vec::with_capacity(names.len());
    for name in &names {
        let extent = extents
            .get(name)
            .copied()
            .ok_or_else(|| PyRuntimeError::new_err(format!("extents missing domain {name:?}")))?;
        domains.push(Domain { extent });
    }

    let rust_index_maps = index_maps
        .iter()
        .map(|(source, target, injective, sorted, unmapped)| IndexMap {
            source_id: ids[source.as_str()],
            target_id: ids[target.as_str()],
            is_injective: *injective,
            is_sorted: *sorted,
            may_have_unmapped: *unmapped,
        })
        .collect();
    let rust_views = available_views
        .iter()
        .map(|(source, anchor, other)| AlignedView {
            source_id: ids[source.as_str()],
            anchor_target_id: ids[anchor.as_str()],
            other_target_id: ids[other.as_str()],
        })
        .collect();
    let lane_count = usize::try_from(u)
        .ok()
        .and_then(NonZeroUsize::new)
        .ok_or_else(|| PyRuntimeError::new_err(format!("loom u must be positive, got {u}")))?;
    let mut operands = operand_specs
        .iter()
        .zip(operand_domains)
        .enumerate()
        .map(|(index, ((element_shape, lane_extent, dtype), row))| {
            let lanes = match *lane_extent {
                1 => Lanes::Shared,
                extent if extent == u => Lanes::PerLane,
                extent => {
                    return Err(format!(
                        "loom operand {index} lane extent {extent} is neither 1 nor u={u}"
                    ));
                }
            };
            Ok(Operand {
                dtype: dtype_from_id(*dtype)?,
                element_shape: element_shape.clone(),
                lanes,
                axis_domains: row
                    .iter()
                    .map(|domain| domain.as_deref().map(|name| ids[name] as usize))
                    .collect(),
            })
        })
        .collect::<Result<Vec<_>, String>>()
        .map_err(PyRuntimeError::new_err)?;
    let outputs = operands.split_off(num_inputs as usize);

    let program = Program {
        name: name.to_string(),
        module: module.inner.clone(),
        lane_count,
        graph: domain_index::Graph {
            domains,
            index_maps: rust_index_maps,
            aligned_views: rust_views,
        },
        inputs: operands,
        outputs,
    };
    let only_element_order = element_order
        .map(|name| name.parse::<ElementOrder>())
        .transpose()
        .map_err(PyRuntimeError::new_err)?;
    let positive = |name: &str, value: usize| {
        NonZeroUsize::new(value)
            .ok_or_else(|| PyRuntimeError::new_err(format!("loom {name} must be positive")))
    };
    let milliseconds = |name: &str, value: Option<f64>| {
        value
            .map(|ms| {
                Duration::try_from_secs_f64(ms / 1000.0).map_err(|_| {
                    PyRuntimeError::new_err(format!(
                        "loom {name} must be nonnegative and finite, got {ms}"
                    ))
                })
            })
            .transpose()
    };
    let defaults = CompileOptions::default();
    let options = CompileOptions {
        search_budget: milliseconds("search_budget_ms", search_budget_ms)?,
        shortlist_budget: milliseconds("shortlist_budget_ms", shortlist_budget_ms)?,
        target: None,
        max_kernel_statements: max_kernel_statements
            .map(|cap| positive("max_kernel_statements", cap))
            .transpose()?,
        max_exact_work: positive("max_algebraic_work", max_algebraic_work)?,
        max_exact_work_per_expression: positive(
            "max_algebraic_expression_work",
            max_algebraic_expression_work,
        )?,
        keep_ptx: dump_ptx,
        only_element_order,
        only_frontier_graph: frontier_graph,
        record_dependencies_and_partitions: record_structure,
        max_workspace_bytes,
        search_size_estimate_limit: search_memory_budget_bytes,
        // The compile cache fingerprint Python computes carries neither of
        // these, so the boundary keeps the calibrated defaults: a caller that
        // varied a selection input the key cannot see would read back another
        // shortlist's recipe.
        tuning_cost_per_candidate: defaults.tuning_cost_per_candidate,
        assumed_assembly_workers: defaults.assumed_assembly_workers,
    };
    let (
        loom::CompiledProgram {
            recipe,
            input_count: _,
            output_count: _,
            index_arrays: slots,
        },
        compile,
    ) = loom::compile(program, options).map_err(PyRuntimeError::new_err)?;
    let serialization_started = std::time::Instant::now();
    recipe
        .validate()
        .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;
    let bytes =
        recipe::serialize_spilled(&recipe).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let slot_rows = slots
        .iter()
        .map(
            |Slot {
                 kind,
                 source_id,
                 target_id,
                 other_target_id,
             }| {
                (
                    kind.name().to_string(),
                    names[*source_id as usize].clone(),
                    names[*target_id as usize].clone(),
                    names[*other_target_id as usize].clone(),
                )
            },
        )
        .collect();
    let environment = cuda_toolchain::compile_environment();
    let timings = compile.timings;
    let mut profile = vec![
        ("simplify_ms", timings.simplify.as_secs_f64() * 1_000.0),
        ("plan_ms", timings.plan.as_secs_f64() * 1_000.0),
        ("render_ptx_ms", timings.render_ptx.as_secs_f64() * 1_000.0),
        (
            "state_requests_ms",
            timings.state_requests.as_secs_f64() * 1_000.0,
        ),
        (
            "build_recipe_ms",
            (timings.build_recipe + serialization_started.elapsed()).as_secs_f64() * 1_000.0,
        ),
        ("compile_workers", environment.workers as f64),
        ("exact_work", compile.exact_work as f64),
        (
            "abandoned_exact_expressions",
            compile.abandoned_exact_expressions as f64,
        ),
        (
            "available_parallelism",
            environment.available_parallelism as f64,
        ),
    ];
    if let Some(cpus) = environment.affinity_cpus {
        profile.push(("affinity_cpus", cpus as f64));
    }
    if let Some(limit) = environment.cgroup_cpu_limit {
        profile.push(("cgroup_cpu_quota", limit.quota_micros as f64));
        profile.push(("cgroup_cpu_period", limit.period_micros as f64));
    }
    if let Some(bytes) = environment.host_memory_bytes {
        profile.push(("host_memory_bytes", bytes as f64));
    }
    if let Some(bytes) = environment.cgroup_memory_bytes {
        profile.push(("cgroup_memory_bytes", bytes as f64));
    }
    let profile = profile
        .into_iter()
        .map(|(name, elapsed)| (name.to_string(), elapsed))
        .collect();
    Ok((bytes, slot_rows, profile, compile))
}

/// The compile's structured diagnostics as Python data: one dict per candidate
/// element order and one per source unit the compiler produced.
fn diagnostics_dicts(py: Python<'_>, profile: &loom::CompileProfile) -> PyResult<DiagnosticRows> {
    let arms = profile
        .arms
        .iter()
        .map(|arm| {
            let row = pyo3::types::PyDict::new_bound(py);
            row.set_item("graph", arm.graph)?;
            row.set_item("order", arm.order)?;
            row.set_item("statement_cap", arm.statement_cap)?;
            row.set_item("plan_clusters", arm.plan_clusters)?;
            Ok(row.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    let element_orders = profile
        .element_orders
        .iter()
        .map(|record| {
            let row = pyo3::types::PyDict::new_bound(py);
            row.set_item("graph", record.graph)?;
            row.set_item("order", record.order)?;
            row.set_item("decision_ms", record.decision_ms)?;
            for (name, value) in &record.summary {
                row.set_item(*name, *value)?;
            }
            Ok(row.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    let frontier_graphs = profile
        .frontier_graphs
        .iter()
        .map(|graph| {
            let row = pyo3::types::PyDict::new_bound(py);
            row.set_item("graph", graph.graph)?;
            row.set_item("frontier_size", graph.frontier_size)?;
            row.set_item("recipe_arm", graph.recipe_arm)?;
            row.set_item("before_accumulation", graph.before_accumulation)?;
            row.set_item("after_accumulation", graph.after_accumulation)?;
            row.set_item("estimated_statements", graph.estimated_statements)?;
            row.set_item("register_pressure", graph.register_pressure)?;
            row.set_item("hoistable_computed_values", graph.hoistable_computed_values)?;
            Ok(row.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    let planned_candidates = profile
        .planned_candidates
        .iter()
        .map(|candidate| {
            let row = pyo3::types::PyDict::new_bound(py);
            row.set_item("graph", candidate.graph)?;
            row.set_item("order", candidate.order)?;
            row.set_item("statement_cap", candidate.statement_cap)?;
            row.set_item("plan_clusters", candidate.plan_clusters)?;
            row.set_item("placement", &candidate.placement)?;
            row.set_item("candidate", candidate.candidate)?;
            row.set_item("role", candidate.role)?;
            row.set_item("origin", candidate.origin)?;
            row.set_item("producer_launches", candidate.producer_launches)?;
            row.set_item("consumer_launches", candidate.consumer_launches)?;
            row.set_item("total_statements", candidate.total_statements)?;
            row.set_item("max_statements", candidate.max_statements)?;
            row.set_item("materialized_bytes", candidate.materialized_bytes)?;
            row.set_item("max_closure_statements", candidate.max_closure_statements)?;
            row.set_item("max_weighted_footprint", candidate.max_weighted_footprint)?;
            row.set_item(
                "duplicated_closure_statements",
                candidate.duplicated_closure_statements,
            )?;
            row.set_item("duplicated_input_loads", candidate.duplicated_input_loads)?;
            row.set_item("distinct_input_loads", candidate.distinct_input_loads)?;
            row.set_item("described_arrangements", candidate.described_arrangements)?;
            row.set_item("geometry_arrangements", candidate.geometry_arrangements)?;
            row.set_item("zero_skip_arrangements", candidate.zero_skip_arrangements)?;
            row.set_item("lane_hoist_arrangements", candidate.lane_hoist_arrangements)?;
            row.set_item(
                "register_capped_arrangements",
                candidate.register_capped_arrangements,
            )?;
            row.set_item("min_positions_per_block", candidate.min_positions_per_block)?;
            row.set_item("max_positions_per_block", candidate.max_positions_per_block)?;
            row.set_item("max_io_lanes", candidate.max_io_lanes)?;
            row.set_item("max_warps", candidate.max_warps)?;
            row.set_item("max_shards", candidate.max_shards)?;
            row.set_item("max_passes", candidate.max_passes)?;
            row.set_item("fiber_bridge", candidate.fiber_bridge)?;
            Ok(row.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    let fold_buckets = profile
        .fold_buckets
        .iter()
        .map(|fold| {
            let row = pyo3::types::PyDict::new_bound(py);
            row.set_item("graph", fold.graph)?;
            row.set_item("placement", &fold.placement)?;
            row.set_item("arm", fold.arm)?;
            row.set_item("output", fold.output)?;
            row.set_item("key", fold.key)?;
            row.set_item("collisions", fold.collisions)?;
            row.set_item("buckets", fold.buckets)?;
            row.set_item("admitted", fold.admitted.clone())?;
            row.set_item("cap", fold.cap)?;
            row.set_item("cap_bound", fold.cap_bound)?;
            row.set_item("elements", fold.elements)?;
            row.set_item("workspace_bytes", fold.workspace_bytes)?;
            row.set_item("projection", fold.projection)?;
            row.set_item("tile_widths", fold.tile_widths.clone())?;
            row.set_item("entry", &fold.entry)?;
            Ok(row.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    let kernels = profile
        .kernels
        .iter()
        .map(|unit| {
            let row = pyo3::types::PyDict::new_bound(py);
            row.set_item("graph", unit.graph)?;
            row.set_item("order", unit.order)?;
            row.set_item("statement_cap", unit.statement_cap)?;
            row.set_item("placement", &unit.placement)?;
            row.set_item("entry", &unit.entry)?;
            row.set_item("flavor", unit.flavor)?;
            row.set_item("part", unit.part)?;
            row.set_item("slot", unit.slot)?;
            row.set_item("register_cap", unit.register_cap)?;
            row.set_item("assembly_mode", unit.assembly_mode)?;
            row.set_item("statements", unit.statements)?;
            row.set_item(
                "addition_chain_within_thread",
                unit.addition_chain_within_thread,
            )?;
            row.set_item(
                "addition_chain_folds_across_threads",
                unit.addition_chain_folds_across_threads,
            )?;
            row.set_item("ptx_bytes", unit.ptx_bytes)?;
            row.set_item("ptx_hash", &unit.ptx_hash)?;
            if let Some(ptx) = &unit.ptx {
                row.set_item("ptx", ptx)?;
            }
            row.set_item("admitted", unit.admitted)?;
            Ok(row.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    let binding_dependencies = profile
        .binding_dependencies
        .iter()
        .map(|record| {
            let row = pyo3::types::PyDict::new_bound(py);
            row.set_item("graph", record.graph)?;
            row.set_item("program", &record.program)?;
            row.set_item("binding", record.binding)?;
            row.set_item("dependencies", record.dependencies.clone())?;
            Ok(row.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    let partition_parts = profile
        .partition_parts
        .iter()
        .map(|record| {
            let row = pyo3::types::PyDict::new_bound(py);
            row.set_item("graph", record.graph)?;
            row.set_item("program", &record.program)?;
            row.set_item("order", record.order)?;
            row.set_item("statement_cap", record.statement_cap)?;
            row.set_item("plan_clusters", record.plan_clusters)?;
            row.set_item("placement", &record.placement)?;
            row.set_item("part", record.part)?;
            row.set_item("bisections", record.bisections)?;
            row.set_item("oversized", record.oversized)?;
            row.set_item(
                "slices",
                record
                    .slices
                    .iter()
                    .map(|(output, element, terms)| (*output, *element, terms.clone()))
                    .collect::<Vec<_>>(),
            )?;
            Ok(row.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    let addition_depths = profile
        .addition_depths
        .iter()
        .map(|record| {
            let row = pyo3::types::PyDict::new_bound(py);
            row.set_item("graph", record.graph)?;
            row.set_item("placement", &record.placement)?;
            row.set_item("arm", record.arm)?;
            row.set_item("candidate", record.candidate)?;
            row.set_item("output", record.output)?;
            row.set_item("within_thread", record.within_thread)?;
            row.set_item("across_blocks", record.across_blocks)?;
            row.set_item("projection", record.projection)?;
            row.set_item("total", record.total)?;
            row.set_item("blocks_per_bucket", record.blocks_per_bucket)?;
            row.set_item("collisions", record.collisions)?;
            row.set_item("buckets", record.buckets)?;
            Ok(row.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok((
        arms,
        frontier_graphs,
        planned_candidates,
        fold_buckets,
        element_orders,
        kernels,
        addition_depths,
        binding_dependencies,
        partition_parts,
    ))
}

/// Python's view of an optional duration: milliseconds, or `None`.
fn as_ms(duration: Option<Duration>) -> Option<f64> {
    duration.map(|duration| duration.as_secs_f64() * 1000.0)
}

/// Default scalar values for Python's Compile options, owned by the Rust compiler.
/// Unset limits are None.
/// This only reads defaults; it does not query a device or compile a program.
#[pyfunction]
fn compile_defaults(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let options = CompileOptions::default();
    let defaults = pyo3::types::PyDict::new_bound(py);
    defaults.set_item("search_budget_ms", as_ms(options.search_budget))?;
    defaults.set_item("shortlist_budget_ms", as_ms(options.shortlist_budget))?;
    defaults.set_item(
        "search_memory_budget_bytes",
        options.search_size_estimate_limit,
    )?;
    defaults.set_item("max_workspace_bytes", options.max_workspace_bytes)?;
    defaults.set_item(
        "max_kernel_statements",
        options.max_kernel_statements.map(NonZeroUsize::get),
    )?;
    defaults.set_item("max_algebraic_work", options.max_exact_work.get())?;
    defaults.set_item(
        "max_algebraic_expression_work",
        options.max_exact_work_per_expression.get(),
    )?;
    Ok(defaults)
}

#[pyfunction]
#[pyo3(signature = (
    name,
    module,
    fingerprint,
    u,
    extents,
    index_maps,
    operand_domains,
    num_inputs,
    operand_specs,
    available_views,
    search_budget_ms=None,
    max_kernel_statements=None,
    max_algebraic_work=CompileOptions::default().max_exact_work.get(),
    max_algebraic_expression_work=CompileOptions::default().max_exact_work_per_expression.get(),
    element_order=None,
    frontier_graph=None,
    dump_ptx=CompileOptions::default().keep_ptx,
    record_structure=CompileOptions::default().record_dependencies_and_partitions,
    max_workspace_bytes=None,
    diagnostics=None,
    shortlist_budget_ms=as_ms(CompileOptions::default().shortlist_budget),
    search_memory_budget_bytes=CompileOptions::default().search_size_estimate_limit,
))]
#[allow(clippy::too_many_arguments)]
fn compile_loom_to_bytes(
    py: Python<'_>,
    name: &str,
    module: &xir_py::XirModule,
    fingerprint: &str,
    u: i64,
    extents: std::collections::HashMap<String, i64>,
    index_maps: Vec<IndexMapRow>,
    operand_domains: Vec<Vec<Option<String>>>,
    num_inputs: i32,
    operand_specs: Vec<OperandSpec>,
    available_views: Vec<(String, String, String)>,
    search_budget_ms: Option<f64>,
    max_kernel_statements: Option<usize>,
    max_algebraic_work: usize,
    max_algebraic_expression_work: usize,
    element_order: Option<String>,
    frontier_graph: Option<usize>,
    dump_ptx: bool,
    record_structure: bool,
    max_workspace_bytes: Option<usize>,
    diagnostics: Option<Bound<'_, pyo3::types::PyDict>>,
    shortlist_budget_ms: Option<f64>,
    search_memory_budget_bytes: Option<u64>,
) -> PyResult<(PyObject, Vec<SlotRow>, ProfileRow)> {
    let _ = fingerprint;
    let (bytes, slots, profile, compile_profile) = loom_recipe_bytes(
        name,
        module,
        u,
        &extents,
        &index_maps,
        &operand_domains,
        num_inputs,
        &operand_specs,
        &available_views,
        search_budget_ms,
        shortlist_budget_ms,
        max_kernel_statements,
        max_algebraic_work,
        max_algebraic_expression_work,
        element_order,
        frontier_graph,
        dump_ptx,
        record_structure,
        max_workspace_bytes,
        search_memory_budget_bytes,
    )?;
    if let Some(diagnostics) = diagnostics {
        let (
            arms,
            frontier_graphs,
            planned_candidates,
            fold_buckets,
            element_orders,
            kernels,
            addition_depths,
            binding_dependencies,
            partition_parts,
        ) = diagnostics_dicts(py, &compile_profile)?;
        diagnostics.set_item(
            "candidate_elapsed_ms",
            compile_profile
                .candidate_elapsed
                .iter()
                .map(|elapsed| elapsed.as_secs_f64() * 1_000.0)
                .collect::<Vec<_>>(),
        )?;
        diagnostics.set_item("refused_attempts", compile_profile.refused_attempts)?;
        diagnostics.set_item(
            "stop_reason",
            compile_profile
                .stop_reason
                .map(loom::profile::StopReason::name),
        )?;
        diagnostics.set_item("search_size_estimate", compile_profile.search_size_estimate)?;
        diagnostics.set_item("search_size_estimate_limit", search_memory_budget_bytes)?;
        diagnostics.set_item(
            "shortlisted_candidates",
            &compile_profile.shortlisted_candidates,
        )?;
        diagnostics.set_item(
            "shortlist_estimated_ms",
            compile_profile.shortlist_estimated_ms,
        )?;
        diagnostics.set_item(
            "shortlist_compile_units",
            compile_profile.shortlist_compile_units,
        )?;
        diagnostics.set_item("arms", arms)?;
        diagnostics.set_item("frontier_graphs", frontier_graphs)?;
        diagnostics.set_item("planned_candidates", planned_candidates)?;
        diagnostics.set_item("fold_buckets", fold_buckets)?;
        diagnostics.set_item("element_orders", element_orders)?;
        diagnostics.set_item("kernels", kernels)?;
        diagnostics.set_item("addition_depths", addition_depths)?;
        diagnostics.set_item("binding_dependencies", binding_dependencies)?;
        diagnostics.set_item("partition_parts", partition_parts)?;
    }
    Ok((
        Py::new(py, recipe_py::SerializedRecipe::from_blob(bytes))?
            .into_bound(py)
            .into_any()
            .unbind(),
        slots,
        profile,
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<xir_py::XirBuilder>()?;
    m.add_class::<xir_py::XirModule>()?;
    m.add_function(wrap_pyfunction!(compile_loom_to_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(compile_defaults, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_compile_defaults_match_rust_options() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = PyModule::new_bound(py, "loom").unwrap();
            register(&module).unwrap();
            let defaults = module.getattr("compile_defaults").unwrap().call0().unwrap();
            let options = CompileOptions::default();
            assert_eq!(defaults.len().unwrap(), 7);
            let get = |key: &str| defaults.get_item(key).unwrap();
            assert_eq!(
                get("search_budget_ms").extract::<Option<f64>>().unwrap(),
                as_ms(options.search_budget),
            );
            assert_eq!(
                get("shortlist_budget_ms").extract::<Option<f64>>().unwrap(),
                Some(600_000.0),
            );
            assert_eq!(
                get("search_memory_budget_bytes")
                    .extract::<Option<u64>>()
                    .unwrap(),
                Some(16 << 30),
            );
            assert_eq!(
                get("max_workspace_bytes")
                    .extract::<Option<usize>>()
                    .unwrap(),
                options.max_workspace_bytes,
            );
            assert_eq!(
                get("max_kernel_statements")
                    .extract::<Option<usize>>()
                    .unwrap(),
                options.max_kernel_statements.map(NonZeroUsize::get),
            );
            assert_eq!(
                get("max_algebraic_work").extract::<usize>().unwrap(),
                options.max_exact_work.get(),
            );
            assert_eq!(
                get("max_algebraic_expression_work")
                    .extract::<usize>()
                    .unwrap(),
                options.max_exact_work_per_expression.get(),
            );
        });
    }
}
