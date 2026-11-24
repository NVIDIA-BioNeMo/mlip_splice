// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `xir` dialect for [`weave`] and lossless conversion from the legacy
//! graph model.

use std::collections::HashMap;

use weave::{
    Attribute, Diagnostic, DialectName, Ir, OperationDefinition, OperationId, OperationName,
    Registry, RegistryError, Type,
};

use crate::ir::{FuncId, Module};
use crate::op::Op;

pub use crate::dialect_types::{
    BoolAttribute, BoolsAttribute, BytesAttribute, EnumAttribute, ExactValuesAttribute,
    FiniteMapGraphAttribute, FunctionRefsAttribute, IntegerAttribute, IntegerPairsAttribute,
    IntegersAttribute, StringAttribute, SymbolNameAttribute, SymbolRefAttribute, TensorType,
};
pub use crate::import::from_ir;

pub const OPERATION_MNEMONICS: &[&str] = &[
    "module",
    "module_end",
    "func",
    "return",
    "negate",
    "abs",
    "exponential",
    "exponential_minus_one",
    "log",
    "log_plus_one",
    "sqrt",
    "rsqrt",
    "cbrt",
    "tanh",
    "sine",
    "cosine",
    "tan",
    "floor",
    "ceil",
    "round_nearest_even",
    "round_nearest_afz",
    "sign",
    "not",
    "count_leading_zeros",
    "popcnt",
    "is_finite",
    "logistic",
    "add",
    "subtract",
    "multiply",
    "divide",
    "maximum",
    "minimum",
    "power",
    "remainder",
    "and",
    "or",
    "xor",
    "shift_left",
    "shift_right_arithmetic",
    "shift_right_logical",
    "atan2",
    "select",
    "clamp",
    "compare",
    "dot_general",
    "convolution",
    "dynamic_slice",
    "dynamic_update_slice",
    "gather",
    "convert",
    "broadcast_in_dim",
    "transpose",
    "reshape",
    "slice",
    "concatenate",
    "pad",
    "reverse",
    "iota",
    "constant",
    "fft",
    "reduce",
    "reduce_window",
    "scatter",
    "sort",
    "while",
    "case",
    "call",
    "gather_along_map",
    "sum_along_map",
];

#[derive(Clone, Copy)]
pub struct OpReconstructor {
    mnemonic: &'static str,
}

impl OpReconstructor {
    pub(crate) fn reconstruct(
        self,
        attributes: &std::collections::BTreeMap<String, Attribute>,
        callees: &[FuncId],
        result_shapes: &[crate::shape::Shape],
    ) -> Result<Box<dyn Op>, ConversionError> {
        crate::op_factory::reconstruct(self.mnemonic, attributes, callees, result_shapes)
    }
}

#[derive(Debug)]
pub enum ConversionError {
    Core(weave::IrError),
    InvalidName(String),
    InvalidModule(crate::verify::VerifyError),
    InvalidCore(String),
    Print(String),
}

impl std::fmt::Display for ConversionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}
impl std::error::Error for ConversionError {}
impl From<weave::IrError> for ConversionError {
    fn from(error: weave::IrError) -> Self {
        Self::Core(error)
    }
}
impl From<crate::verify::VerifyError> for ConversionError {
    fn from(error: crate::verify::VerifyError) -> Self {
        Self::InvalidModule(error)
    }
}

pub fn register(registry: &mut Registry) -> Result<(), RegistryError> {
    registry.register_dialect(DialectName::new("xir").expect("static dialect name"))?;
    for mnemonic in OPERATION_MNEMONICS {
        let name = *mnemonic;
        let definition = OperationDefinition::new()
            .interface(OpReconstructor { mnemonic: name })
            .verifier(move |ir, operation, diagnostics| {
                verify_operation_shape(ir, operation, name, diagnostics);
            });
        registry.register_operation(
            OperationName::new(format!("xir.{mnemonic}")).expect("static operation name"),
            if matches!(*mnemonic, "return" | "module_end") {
                definition.terminator()
            } else {
                definition
            },
        )?;
    }
    registry.register_type::<TensorType>()?;
    registry.register_attribute::<StringAttribute>()?;
    registry.register_attribute::<IntegerAttribute>()?;
    registry.register_attribute::<BoolAttribute>()?;
    registry.register_attribute::<BytesAttribute>()?;
    registry.register_attribute::<IntegersAttribute>()?;
    registry.register_attribute::<EnumAttribute>()?;
    registry.register_attribute::<ExactValuesAttribute>()?;
    registry.register_attribute::<BoolsAttribute>()?;
    registry.register_attribute::<IntegerPairsAttribute>()?;
    registry.register_attribute::<SymbolNameAttribute>()?;
    registry.register_attribute::<SymbolRefAttribute>()?;
    registry.register_attribute::<FunctionRefsAttribute>()?;
    registry.register_attribute::<FiniteMapGraphAttribute>()?;
    Ok(())
}

pub fn registry() -> Registry {
    let mut registry = Registry::new();
    register(&mut registry).expect("fresh XIR registry accepts its dialect");
    registry
}

fn verify_operation_shape(
    ir: &Ir,
    operation: OperationId,
    mnemonic: &str,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let op = ir
        .operation(operation)
        .expect("registered operation exists");
    if !op.successors.is_empty() {
        diagnostics.push(Diagnostic::error(format!(
            "xir.{mnemonic} does not support CFG successors"
        )));
    }
    let expected_regions = match mnemonic {
        "module" | "func" => 1,
        _ => 0,
    };
    if op.regions.len() != expected_regions {
        diagnostics.push(Diagnostic::error(format!(
            "xir.{mnemonic} requires {expected_regions} region(s), got {}",
            op.regions.len()
        )));
    }
    if matches!(mnemonic, "module" | "func" | "module_end")
        && (!op.operands.is_empty() || !op.results.is_empty())
    {
        diagnostics.push(Diagnostic::error(format!(
            "xir.{mnemonic} cannot have operands or results"
        )));
    }
    if mnemonic == "return" && !op.results.is_empty() {
        diagnostics.push(Diagnostic::error("xir.return cannot have results"));
    }
    let required = required_attributes(mnemonic);
    let allowed = allowed_attributes(mnemonic);
    for name in required {
        if !op.attributes.contains_key(*name) {
            diagnostics.push(Diagnostic::error(format!(
                "xir.{mnemonic} requires attribute `{name}`"
            )));
        }
    }
    verify_callee_count(op, mnemonic, diagnostics);
    verify_arity(ir, op, mnemonic, diagnostics);
    for (name, attribute) in &op.attributes {
        if !allowed.contains(&name.as_str()) {
            diagnostics.push(Diagnostic::error(format!(
                "xir.{mnemonic} does not allow attribute `{name}`"
            )));
        } else if !attribute_has_expected_type(name, attribute) {
            diagnostics.push(Diagnostic::error(format!(
                "xir.{mnemonic} attribute `{name}` has the wrong type"
            )));
        }
    }
}

fn required_attributes(mnemonic: &str) -> &'static [&'static str] {
    match mnemonic {
        "module" => &["entry", "finite_map_graph"],
        "func" => &["symbol"],
        "constant" => &["data"],
        "compare" => &["comparison_direction"],
        "convert" => &["dtype"],
        "broadcast_in_dim" => &["out_dims", "broadcast_dimensions"],
        "transpose" => &["permutation"],
        "reshape" => &["out_dims"],
        "slice" => &["start_indices", "limit_indices", "strides"],
        "concatenate" => &["dimension"],
        "sort" => &["dimension", "is_stable", "callees"],
        "pad" => &["edge_padding_low", "edge_padding_high", "interior_padding"],
        "reverse" => &["dimensions"],
        "reduce" => &["dimensions", "callees"],
        "iota" => &["iota_dimension"],
        "fft" => &["fft_type", "fft_length"],
        "dynamic_slice" => &["slice_sizes"],
        "dot_general" => &[
            "lhs_contracting_dimensions",
            "rhs_contracting_dimensions",
            "lhs_batching_dimensions",
            "rhs_batching_dimensions",
            "algorithm",
        ],
        "convolution" => &[
            "input_batch_dimension",
            "input_feature_dimension",
            "input_spatial_dimensions",
            "kernel_input_feature_dimension",
            "kernel_output_feature_dimension",
            "kernel_spatial_dimensions",
            "output_batch_dimension",
            "output_feature_dimension",
            "output_spatial_dimensions",
            "window_strides",
            "padding",
            "lhs_dilation",
            "rhs_dilation",
            "feature_group_count",
            "batch_group_count",
        ],
        "gather" => &[
            "offset_dims",
            "collapsed_slice_dims",
            "operand_batching_dims",
            "start_indices_batching_dims",
            "start_index_map",
            "index_vector_dim",
            "slice_sizes",
            "indices_are_sorted",
        ],
        "reduce_window" => &[
            "window_dimensions",
            "window_strides",
            "base_dilations",
            "window_dilations",
            "padding",
            "callees",
        ],
        "scatter" => &[
            "update_window_dims",
            "inserted_window_dims",
            "input_batching_dims",
            "scatter_indices_batching_dims",
            "scatter_dims_to_operand_dims",
            "index_vector_dim",
            "indices_are_sorted",
            "unique_indices",
            "callees",
        ],
        "gather_along_map" | "sum_along_map" => &["map", "axis"],
        "call" | "case" | "while" => &["callees"],
        _ => &[],
    }
}

fn allowed_attributes(mnemonic: &str) -> &'static [&'static str] {
    match mnemonic {
        "constant" => &["data", "exact"],
        "compare" => &["comparison_direction", "compare_type"],
        "convolution" => &[
            "input_batch_dimension",
            "input_feature_dimension",
            "input_spatial_dimensions",
            "kernel_input_feature_dimension",
            "kernel_output_feature_dimension",
            "kernel_spatial_dimensions",
            "output_batch_dimension",
            "output_feature_dimension",
            "output_spatial_dimensions",
            "window_strides",
            "padding",
            "lhs_dilation",
            "rhs_dilation",
            "window_reversal",
            "feature_group_count",
            "batch_group_count",
        ],
        "module" => &["entry", "finite_map_graph"],
        "func" => &["symbol"],
        "convert" => &["dtype"],
        "broadcast_in_dim" => &["out_dims", "broadcast_dimensions"],
        "transpose" => &["permutation"],
        "reshape" => &["out_dims"],
        "slice" => &["start_indices", "limit_indices", "strides"],
        "concatenate" => &["dimension"],
        "sort" => &["dimension", "is_stable", "callees"],
        "pad" => &["edge_padding_low", "edge_padding_high", "interior_padding"],
        "reverse" => &["dimensions"],
        "reduce" => &["dimensions", "callees"],
        "iota" => &["iota_dimension"],
        "fft" => &["fft_type", "fft_length"],
        "dynamic_slice" => &["slice_sizes"],
        "dot_general" => &[
            "lhs_contracting_dimensions",
            "rhs_contracting_dimensions",
            "lhs_batching_dimensions",
            "rhs_batching_dimensions",
            "algorithm",
        ],
        "gather" => &[
            "offset_dims",
            "collapsed_slice_dims",
            "operand_batching_dims",
            "start_indices_batching_dims",
            "start_index_map",
            "index_vector_dim",
            "slice_sizes",
            "indices_are_sorted",
        ],
        "reduce_window" => &[
            "window_dimensions",
            "window_strides",
            "base_dilations",
            "window_dilations",
            "padding",
            "callees",
        ],
        "scatter" => &[
            "update_window_dims",
            "inserted_window_dims",
            "input_batching_dims",
            "scatter_indices_batching_dims",
            "scatter_dims_to_operand_dims",
            "index_vector_dim",
            "indices_are_sorted",
            "unique_indices",
            "callees",
        ],
        "gather_along_map" | "sum_along_map" => &["map", "axis"],
        "call" | "case" | "while" => &["callees"],
        _ => &[],
    }
}

fn verify_callee_count(op: &weave::Operation, mnemonic: &str, diagnostics: &mut Vec<Diagnostic>) {
    let expected = match mnemonic {
        "call" | "reduce" | "reduce_window" | "scatter" | "sort" => Some(1),
        "while" => Some(2),
        _ => None,
    };
    let actual = op
        .attributes
        .get("callees")
        .and_then(|attribute| attribute.downcast_ref::<FunctionRefsAttribute>())
        .map_or(0, |references| references.0.len());
    if let Some(expected) = expected
        && actual != expected
    {
        diagnostics.push(Diagnostic::error(format!(
            "xir.{mnemonic} requires {expected} callee(s), got {actual}"
        )));
    }
    if mnemonic == "case" && actual == 0 {
        diagnostics.push(Diagnostic::error("xir.case requires at least one callee"));
    }
}

fn verify_arity(ir: &Ir, op: &weave::Operation, mnemonic: &str, diagnostics: &mut Vec<Diagnostic>) {
    let fixed = match mnemonic {
        "negate"
        | "abs"
        | "exponential"
        | "exponential_minus_one"
        | "log"
        | "log_plus_one"
        | "sqrt"
        | "rsqrt"
        | "cbrt"
        | "tanh"
        | "sine"
        | "cosine"
        | "tan"
        | "floor"
        | "ceil"
        | "round_nearest_even"
        | "round_nearest_afz"
        | "sign"
        | "not"
        | "count_leading_zeros"
        | "popcnt"
        | "is_finite"
        | "logistic"
        | "convert"
        | "broadcast_in_dim"
        | "transpose"
        | "reshape"
        | "slice"
        | "reverse"
        | "fft"
        | "gather_along_map"
        | "sum_along_map" => Some((1, 1)),
        "add"
        | "subtract"
        | "multiply"
        | "divide"
        | "maximum"
        | "minimum"
        | "power"
        | "remainder"
        | "and"
        | "or"
        | "xor"
        | "shift_left"
        | "shift_right_arithmetic"
        | "shift_right_logical"
        | "atan2"
        | "compare"
        | "dot_general"
        | "convolution"
        | "gather"
        | "pad" => Some((2, 1)),
        "select" | "clamp" | "scatter" => Some((3, 1)),
        "iota" | "constant" => Some((0, 1)),
        "module" | "module_end" | "func" => Some((0, 0)),
        _ => None,
    };
    if let Some((operands, results)) = fixed
        && (op.operands.len(), op.results.len()) != (operands, results)
    {
        diagnostics.push(Diagnostic::error(format!(
            "xir.{mnemonic} requires {operands} operand(s) and {results} result(s)"
        )));
    }
    let valid_variable = match mnemonic {
        "concatenate" | "sort" => {
            !op.operands.is_empty()
                && op.results.len()
                    == if mnemonic == "sort" {
                        op.operands.len()
                    } else {
                        1
                    }
        }
        "reduce" | "reduce_window" => {
            !op.operands.is_empty()
                && op.operands.len().is_multiple_of(2)
                && op.results.len() * 2 == op.operands.len()
        }
        "while" => op.operands.len() == op.results.len(),
        "dynamic_slice" => {
            op.results.len() == 1
                && op
                    .operands
                    .first()
                    .and_then(|value| ir.value(*value))
                    .is_some_and(|value| {
                        value
                            .ty
                            .downcast_ref::<TensorType>()
                            .is_some_and(|ty| op.operands.len() == ty.0.rank() + 1)
                    })
        }
        "dynamic_update_slice" => {
            op.results.len() == 1
                && op
                    .operands
                    .first()
                    .and_then(|value| ir.value(*value))
                    .is_some_and(|value| {
                        value
                            .ty
                            .downcast_ref::<TensorType>()
                            .is_some_and(|ty| op.operands.len() == ty.0.rank() + 2)
                    })
        }
        "case" => !op.operands.is_empty(),
        "call" => true,
        "return" => op.results.is_empty(),
        _ => true,
    };
    if fixed.is_none() && !valid_variable {
        diagnostics.push(Diagnostic::error(format!(
            "xir.{mnemonic} has invalid arity"
        )));
    }
}

fn attribute_has_expected_type(name: &str, attribute: &Attribute) -> bool {
    match name {
        "entry" => attribute.downcast_ref::<SymbolRefAttribute>().is_some(),
        "symbol" => attribute.downcast_ref::<SymbolNameAttribute>().is_some(),
        "callees" => attribute.downcast_ref::<FunctionRefsAttribute>().is_some(),
        "finite_map_graph" => attribute
            .downcast_ref::<FiniteMapGraphAttribute>()
            .is_some(),
        "data" => attribute.downcast_ref::<BytesAttribute>().is_some(),
        "indices_are_sorted" | "unique_indices" | "is_stable" | "allow_imprecise_accum" => {
            attribute.downcast_ref::<BoolAttribute>().is_some()
        }
        "dimension"
        | "iota_dimension"
        | "index_vector_dim"
        | "input_batch_dimension"
        | "input_feature_dimension"
        | "kernel_input_feature_dimension"
        | "kernel_output_feature_dimension"
        | "output_batch_dimension"
        | "output_feature_dimension"
        | "feature_group_count"
        | "batch_group_count"
        | "axis"
        | "map" => attribute.downcast_ref::<IntegerAttribute>().is_some(),
        "dimensions"
        | "out_dims"
        | "broadcast_dimensions"
        | "permutation"
        | "start_indices"
        | "limit_indices"
        | "strides"
        | "edge_padding_low"
        | "edge_padding_high"
        | "interior_padding"
        | "slice_sizes"
        | "fft_length"
        | "window_dimensions"
        | "window_strides"
        | "base_dilations"
        | "window_dilations"
        | "lhs_dilation"
        | "rhs_dilation"
        | "lhs_contracting_dimensions"
        | "rhs_contracting_dimensions"
        | "lhs_batching_dimensions"
        | "rhs_batching_dimensions"
        | "offset_dims"
        | "collapsed_slice_dims"
        | "operand_batching_dims"
        | "start_indices_batching_dims"
        | "start_index_map"
        | "input_spatial_dimensions"
        | "kernel_spatial_dimensions"
        | "output_spatial_dimensions"
        | "update_window_dims"
        | "inserted_window_dims"
        | "input_batching_dims"
        | "scatter_indices_batching_dims"
        | "scatter_dims_to_operand_dims" => attribute.downcast_ref::<IntegersAttribute>().is_some(),
        "padding" => attribute.downcast_ref::<IntegerPairsAttribute>().is_some(),
        "window_reversal" => attribute.downcast_ref::<BoolsAttribute>().is_some(),
        "dtype" | "fft_type" | "comparison_direction" | "compare_type" | "algorithm" => {
            attribute.downcast_ref::<EnumAttribute>().is_some()
        }
        "exact" => attribute.downcast_ref::<ExactValuesAttribute>().is_some(),
        _ => false,
    }
}

pub fn to_ir(module: &Module) -> Result<Ir, ConversionError> {
    crate::verify::verify(module)?;
    let mut ir = Ir::new();
    let module_op = ir.create_operation(operation_name("module")?, [], [])?;
    ir.add_root(module_op)?;
    ir.set_attribute(
        module_op,
        "entry",
        Attribute::typed(SymbolRefAttribute(module.func(module.entry).name.clone())),
    )?;
    ir.set_attribute(
        module_op,
        "finite_map_graph",
        Attribute::typed(FiniteMapGraphAttribute(module.finite_map_graph.clone())),
    )?;
    let region = ir.append_region(module_op)?;
    let block = ir.append_block(region, [])?;

    for func in &module.funcs {
        let parameter_types = func
            .params
            .iter()
            .map(|&value| Type::typed(TensorType(func.value_shape(value).clone())));
        let func_op = ir.create_operation(operation_name("func")?, [], [])?;
        ir.set_attribute(
            func_op,
            "symbol",
            Attribute::typed(SymbolNameAttribute(func.name.clone())),
        )?;
        ir.append_operation(block, func_op)?;
        let body = ir.append_region(func_op)?;
        let body_block = ir.append_block(body, parameter_types)?;
        let arguments = ir
            .block(body_block)
            .expect("new function block")
            .arguments
            .clone();
        let mut values = HashMap::new();
        for (&legacy, core) in func.params.iter().zip(arguments) {
            values.insert(legacy, core);
        }
        for node in &func.nodes {
            let operands = node
                .inputs
                .iter()
                .map(|value| values[value])
                .collect::<Vec<_>>();
            let result_types = node
                .results
                .iter()
                .map(|&value| Type::typed(TensorType(func.value_shape(value).clone())));
            let operation =
                ir.create_operation(operation_name(node.op.mnemonic())?, operands, result_types)?;
            let has_callees = !node.op.called_funcs().is_empty();
            for (name, value) in node.op.attrs() {
                if !has_callees
                    || !matches!(
                        name,
                        "callee"
                            | "reducer"
                            | "update"
                            | "comparator"
                            | "cond"
                            | "body"
                            | "branches"
                    )
                {
                    ir.set_attribute(operation, name, typed_operation_attribute(name, value))?;
                }
            }
            if let Some(constant) = node.op.as_any().downcast_ref::<crate::ops::Constant>() {
                ir.set_attribute(
                    operation,
                    "data",
                    Attribute::typed(BytesAttribute(constant.literal.data.clone())),
                )?;
                if let Some(exact) = &constant.literal.exact {
                    ir.set_attribute(
                        operation,
                        "exact",
                        Attribute::typed(ExactValuesAttribute(exact.clone())),
                    )?;
                }
            }
            let called = node
                .op
                .called_funcs()
                .iter()
                .map(|id| module.func(*id).name.clone())
                .collect::<Vec<_>>();
            if !called.is_empty() {
                ir.set_attribute(
                    operation,
                    "callees",
                    Attribute::typed(FunctionRefsAttribute(called)),
                )?;
            }
            ir.append_operation(body_block, operation)?;
            let results = ir
                .operation(operation)
                .expect("new operation")
                .results
                .clone();
            for (&legacy, core) in node.results.iter().zip(results) {
                values.insert(legacy, core);
            }
        }
        let returned = func
            .results
            .iter()
            .map(|value| values[value])
            .collect::<Vec<_>>();
        let return_op = ir.create_operation(operation_name("return")?, returned, [])?;
        ir.append_operation(body_block, return_op)?;
    }
    let module_end = ir.create_operation(operation_name("module_end")?, [], [])?;
    ir.append_operation(block, module_end)?;
    Ok(ir)
}

fn typed_operation_attribute(name: &str, value: String) -> Attribute {
    if matches!(
        name,
        "indices_are_sorted" | "unique_indices" | "is_stable" | "allow_imprecise_accum"
    ) && let Ok(value) = value.parse()
    {
        return Attribute::typed(BoolAttribute(value));
    }
    if matches!(
        name,
        "dimension"
            | "iota_dimension"
            | "index_vector_dim"
            | "input_batch_dimension"
            | "input_feature_dimension"
            | "kernel_input_feature_dimension"
            | "kernel_output_feature_dimension"
            | "output_batch_dimension"
            | "output_feature_dimension"
            | "feature_group_count"
            | "batch_group_count"
            | "axis"
            | "map"
    ) && let Ok(value) = value.parse()
    {
        return Attribute::typed(IntegerAttribute(value));
    }
    if matches!(
        name,
        "dimensions"
            | "out_dims"
            | "broadcast_dimensions"
            | "permutation"
            | "start_indices"
            | "limit_indices"
            | "strides"
            | "edge_padding_low"
            | "edge_padding_high"
            | "interior_padding"
            | "slice_sizes"
            | "fft_length"
            | "window_dimensions"
            | "window_strides"
            | "base_dilations"
            | "window_dilations"
            | "lhs_dilation"
            | "rhs_dilation"
            | "lhs_contracting_dimensions"
            | "rhs_contracting_dimensions"
            | "lhs_batching_dimensions"
            | "rhs_batching_dimensions"
            | "offset_dims"
            | "collapsed_slice_dims"
            | "operand_batching_dims"
            | "start_indices_batching_dims"
            | "start_index_map"
            | "input_spatial_dimensions"
            | "kernel_spatial_dimensions"
            | "output_spatial_dimensions"
            | "update_window_dims"
            | "inserted_window_dims"
            | "input_batching_dims"
            | "scatter_indices_batching_dims"
            | "scatter_dims_to_operand_dims"
    ) && let Ok(values) = parse_bracketed_integers(&value)
    {
        return Attribute::typed(IntegersAttribute(values));
    }
    if name == "padding"
        && let Ok(values) = parse_bracketed_pairs(&value)
    {
        return Attribute::typed(IntegerPairsAttribute(values));
    }
    if name == "window_reversal"
        && let Ok(values) = parse_bracketed_bools(&value)
    {
        return Attribute::typed(BoolsAttribute(values));
    }
    if matches!(
        name,
        "dtype" | "fft_type" | "comparison_direction" | "compare_type" | "algorithm"
    ) {
        return Attribute::typed(EnumAttribute(value));
    }
    Attribute::typed(StringAttribute(value))
}

fn parse_bracketed_integers(value: &str) -> Result<Vec<i64>, String> {
    let body = value
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .ok_or("expected bracketed integer list")?;
    if body.trim().is_empty() {
        return Ok(Vec::new());
    }
    body.split(',')
        .map(|item| {
            item.trim()
                .parse()
                .map_err(|error: std::num::ParseIntError| error.to_string())
        })
        .collect()
}

fn parse_bracketed_pairs(value: &str) -> Result<Vec<[i64; 2]>, String> {
    let body = value
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .ok_or("expected bracketed pairs")?;
    if body.trim().is_empty() {
        return Ok(Vec::new());
    }
    body.split("], [")
        .map(|pair| {
            let pair = pair.trim_matches(['[', ']']);
            let values = pair
                .replace(' ', "")
                .split(',')
                .map(|value| value.parse().map_err(|_| "invalid integer".to_owned()))
                .collect::<Result<Vec<i64>, _>>()?;
            values
                .try_into()
                .map_err(|_| "expected integer pair".to_owned())
        })
        .collect()
}

fn parse_bracketed_bools(value: &str) -> Result<Vec<bool>, String> {
    let body = value
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .ok_or("expected bracketed booleans")?;
    if body.trim().is_empty() {
        return Ok(Vec::new());
    }
    body.split(',')
        .map(|value| {
            value
                .trim()
                .parse()
                .map_err(|_| "invalid boolean".to_owned())
        })
        .collect()
}

fn operation_name(mnemonic: &str) -> Result<OperationName, ConversionError> {
    if !OPERATION_MNEMONICS.contains(&mnemonic) {
        return Err(ConversionError::InvalidName(format!(
            "unregistered XIR operation `xir.{mnemonic}`"
        )));
    }
    OperationName::new(format!("xir.{mnemonic}"))
        .map_err(|error| ConversionError::InvalidName(error.to_string()))
}
