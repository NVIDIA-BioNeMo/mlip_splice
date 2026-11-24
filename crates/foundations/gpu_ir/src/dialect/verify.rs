// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use weave::{Ir, OperationId as CoreOperationId};

use super::GpuAttribute;

pub(super) fn typed_attribute<'a>(
    operation: &'a weave::Operation,
    name: &str,
) -> Option<&'a GpuAttribute> {
    operation
        .attributes
        .get(name)?
        .downcast_ref::<GpuAttribute>()
}

pub(super) fn reject_attributes_except(
    operation: &weave::Operation,
    allowed: &[&str],
    diagnostics: &mut Vec<weave::Diagnostic>,
) {
    for name in operation.attributes.keys() {
        if !allowed.contains(&name.as_str()) {
            diagnostics.push(weave::Diagnostic::error(format!(
                "{} attribute `{name}` is not representable in gpu_ir",
                operation.name
            )));
        }
    }
}

pub(super) fn operation_attribute_is_representable(operation: &str, attribute: &str) -> bool {
    if matches!(
        attribute,
        "materialized"
            | "operand_count"
            | "source_operation"
            | "result_indices"
            | "region_indices"
            | "destinations"
    ) {
        return true;
    }
    match operation {
        "gpu.int_const" | "gpu.float_const" => matches!(attribute, "scalar" | "value"),
        "gpu.sreg" => attribute == "register",
        "gpu.load" => matches!(attribute, "lanes" | "non_coherent" | "evict"),
        "gpu.unary" | "gpu.binary" => attribute == "operator",
        "gpu.compare" => attribute == "predicate",
        "gpu.cast" => attribute == "scalar",
        "gpu.splat" => attribute == "lanes",
        "gpu.extract" => attribute == "lane",
        "gpu.const_load" => matches!(attribute, "name" | "scalar" | "data" | "constant_identity"),
        "gpu.write" => matches!(attribute, "mode" | "evict"),
        "gpu.loop" => matches!(attribute, "index" | "trip"),
        "gpu.call" => matches!(
            attribute,
            "callee" | "callee_symbol" | "parameter_count" | "result_count"
        ),
        _ => false,
    }
}

#[allow(clippy::cognitive_complexity)]
pub(super) fn verify_operation(
    ir: &Ir,
    id: CoreOperationId,
    expected_regions: usize,
    diagnostics: &mut Vec<weave::Diagnostic>,
) {
    let operation = ir.operation(id).expect("registered operation exists");
    if operation.parent.is_none() {
        diagnostics.push(weave::Diagnostic::error(
            "only gpu.func may be a root operation",
        ));
    }
    if !operation.successors.is_empty() {
        diagnostics.push(weave::Diagnostic::error(
            "GPU operations cannot have CFG successors",
        ));
    }
    let materialized = typed_attribute(operation, "materialized");
    if materialized.is_none() {
        diagnostics.push(weave::Diagnostic::error(
            "GPU operations require a typed `materialized` attribute",
        ));
    } else if !matches!(materialized, Some(GpuAttribute::Bool(_))) {
        diagnostics.push(weave::Diagnostic::error(
            "GPU `materialized` must be bool(true) or bool(false)",
        ));
    }
    let operand_count = typed_attribute(operation, "operand_count");
    if operand_count.is_none() {
        diagnostics.push(weave::Diagnostic::error(
            "GPU operations require a typed `operand_count` attribute",
        ));
    }
    let declared = match operand_count {
        Some(GpuAttribute::UInt(value)) => Some(*value),
        _ => None,
    };
    if declared.is_none_or(|count| count > operation.operands.len()) {
        diagnostics.push(weave::Diagnostic::error(
            "GPU `operand_count` exceeds the operation operand list",
        ));
    }
    let source_operation = typed_attribute(operation, "source_operation");
    if source_operation.is_none() {
        diagnostics.push(weave::Diagnostic::error(
            "GPU operations require a typed `source_operation` attribute",
        ));
    } else if !matches!(source_operation, Some(GpuAttribute::UInt(_))) {
        diagnostics.push(weave::Diagnostic::error(
            "GPU `source_operation` must be unsigned",
        ));
    }
    if !matches!(
        typed_attribute(operation, "result_indices"),
        Some(GpuAttribute::Indices(_))
    ) || !matches!(
        typed_attribute(operation, "region_indices"),
        Some(GpuAttribute::Indices(_))
    ) {
        diagnostics.push(weave::Diagnostic::error(
            "GPU operations require typed result_indices and region_indices attributes",
        ));
    }
    if let Some(GpuAttribute::Indices(values)) = typed_attribute(operation, "result_indices")
        && values
            .iter()
            .flatten()
            .any(|index| *index >= ir.values().count())
    {
        diagnostics.push(weave::Diagnostic::error(
            "GPU result_indices contains an out-of-range source value index",
        ));
    }
    let destinations = match typed_attribute(operation, "destinations") {
        None => Some(Vec::new()),
        Some(GpuAttribute::Indices(values)) => Some(values.clone()),
        Some(_) => None,
    };
    let expected_operands =
        declared
            .zip(destinations.as_ref())
            .and_then(|(source, destinations)| {
                source.checked_add(
                    destinations
                        .iter()
                        .filter(|destination| destination.is_some())
                        .count(),
                )
            });
    if expected_operands != Some(operation.operands.len()) {
        diagnostics.push(weave::Diagnostic::error(
            "GPU operand list must contain exactly its source operands and destinations",
        ));
    }
    if let (Some(source), Some(destinations)) = (declared, destinations.as_ref()) {
        let count = destinations.iter().flatten().count();
        let expected_slots = source.checked_add(count).map(|end| source..end);
        if expected_slots.is_none_or(|slots| destinations.iter().flatten().copied().ne(slots)) {
            diagnostics.push(weave::Diagnostic::error(
                "GPU destination slots must exactly follow source operands",
            ));
        }
        if !destinations.is_empty() && destinations.len() != operation.results.len() {
            diagnostics.push(weave::Diagnostic::error(
                "GPU destinations must be absent or have one entry per result",
            ));
        }
    }
    if operation.regions.len() != expected_regions {
        diagnostics.push(weave::Diagnostic::error(format!(
            "{} requires {expected_regions} regions, got {}",
            operation.name,
            operation.regions.len()
        )));
    }
    if operation.name.as_str() == "gpu.call" {
        let parameter_count = match typed_attribute(operation, "parameter_count") {
            Some(GpuAttribute::UInt(value)) => Some(*value),
            _ => None,
        };
        let result_count = match typed_attribute(operation, "result_count") {
            Some(GpuAttribute::UInt(value)) => Some(*value),
            _ => None,
        };
        if parameter_count != declared || result_count != Some(operation.results.len()) {
            diagnostics.push(weave::Diagnostic::error(
                "gpu.call arities must match parameter_count and result_count",
            ));
        }
        if !matches!(
            typed_attribute(operation, "callee"),
            Some(GpuAttribute::Text(_))
        ) || !matches!(
            typed_attribute(operation, "callee_symbol"),
            Some(GpuAttribute::UInt(_))
        ) {
            diagnostics.push(weave::Diagnostic::error(
                "gpu.call requires typed callee and callee_symbol attributes",
            ));
        }
    }
    let (operands, results) = match operation.name.as_str() {
        "gpu.int_const" | "gpu.float_const" | "gpu.sreg" | "gpu.active_mask" => (Some(0), Some(1)),
        "gpu.offset" | "gpu.unary" | "gpu.cast" | "gpu.splat" | "gpu.extract" | "gpu.shfl_down"
        | "gpu.warp_any" | "gpu.const_load" | "gpu.bind" => (Some(1), Some(1)),
        "gpu.binary" | "gpu.compare" | "gpu.load" | "gpu.fetch_add" => (Some(2), Some(1)),
        "gpu.fma" | "gpu.select" | "gpu.shfl_from" => (Some(3), Some(1)),
        "gpu.pack" => (None, Some(1)),
        "gpu.write" => (Some(2), Some(0)),
        "gpu.break" | "gpu.ret" | "gpu.barrier" | "gpu.scope" => (Some(0), Some(0)),
        _ => (None, None),
    };
    if let Some(expected) = operands
        && declared != Some(expected)
    {
        diagnostics.push(weave::Diagnostic::error(format!(
            "{} requires {expected} source operands",
            operation.name
        )));
    }
    if let Some(expected) = results
        && operation.results.len() != expected
    {
        diagnostics.push(weave::Diagnostic::error(format!(
            "{} requires {expected} results",
            operation.name
        )));
    }
    for name in operation.attributes.keys() {
        if !operation_attribute_is_representable(operation.name.as_str(), name) {
            diagnostics.push(weave::Diagnostic::error(format!(
                "{} attribute `{name}` is not representable in gpu_ir",
                operation.name
            )));
        }
    }
}
