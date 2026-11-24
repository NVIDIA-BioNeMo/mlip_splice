// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `gpu` dialect view of a device function.

mod from_weave;
mod types;
mod verify;

pub use from_weave::from_weave;
pub use types::{DataType, GpuAttribute, PointerType};
use verify::{
    operation_attribute_is_representable, reject_attributes_except, typed_attribute,
    verify_operation,
};

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use weave::{
    Attribute, BlockId as CoreBlockId, DialectName, Ir, OperationDefinition,
    OperationId as CoreOperationId, OperationName, RegionId as CoreRegionId, Registry,
    RegistryError, Type, ValueId as CoreValueId,
};

use crate::arena::{BlockId, Declaration, Function, OpId, RegionId, ValueId};
use crate::op::OpKind;
use crate::{ParamTy, ValTy};

const OPERATIONS: &[(&str, bool)] = &[
    ("gpu.func", false),
    ("gpu.yield", true),
    ("gpu.return", true),
    ("gpu.int_const", false),
    ("gpu.float_const", false),
    ("gpu.sreg", false),
    ("gpu.load", false),
    ("gpu.offset", false),
    ("gpu.unary", false),
    ("gpu.binary", false),
    ("gpu.fma", false),
    ("gpu.compare", false),
    ("gpu.select", false),
    ("gpu.cast", false),
    ("gpu.splat", false),
    ("gpu.extract", false),
    ("gpu.pack", false),
    ("gpu.shfl_down", false),
    ("gpu.active_mask", false),
    ("gpu.shfl_from", false),
    ("gpu.warp_any", false),
    ("gpu.const_load", false),
    ("gpu.bind", false),
    ("gpu.write", false),
    ("gpu.fetch_add", false),
    ("gpu.if", false),
    ("gpu.loop", false),
    ("gpu.scope", false),
    ("gpu.call", false),
    ("gpu.break", false),
    ("gpu.ret", false),
    ("gpu.barrier", false),
];

pub fn register(registry: &mut Registry) -> Result<(), RegistryError> {
    registry.register_dialect(DialectName::new("gpu").expect("static dialect name"))?;
    registry.register_type::<DataType>()?;
    registry.register_type::<PointerType>()?;
    registry.register_attribute::<GpuAttribute>()?;
    for &(name, terminator) in OPERATIONS {
        let mut definition = if terminator {
            OperationDefinition::new().terminator()
        } else {
            OperationDefinition::new()
        };
        if name == "gpu.func" {
            definition = definition.verifier(|ir, id, diagnostics| {
                let operation = ir.operation(id).expect("registered operation exists");
                if operation.parent.is_some() || ir.roots() != [id] {
                    diagnostics.push(weave::Diagnostic::error(
                        "gpu.func must be the sole root operation",
                    ));
                }
                if !operation.operands.is_empty()
                    || !operation.results.is_empty()
                    || !operation.successors.is_empty()
                {
                    diagnostics.push(weave::Diagnostic::error(
                        "gpu.func cannot have operands, results, or successors",
                    ));
                }
                if operation.regions.len() != 1 {
                    diagnostics.push(weave::Diagnostic::error(
                        "gpu.func requires exactly one body region",
                    ));
                }
                if !matches!(
                    typed_attribute(operation, "entry"),
                    Some(GpuAttribute::UInt(_))
                ) {
                    diagnostics.push(weave::Diagnostic::error(
                        "gpu.func requires a typed unsigned `entry` attribute",
                    ));
                }
                for name in operation.attributes.keys() {
                    if name != "entry"
                        && !name.starts_with("declaration_")
                        && !name.starts_with("name_")
                    {
                        diagnostics.push(weave::Diagnostic::error(format!(
                            "gpu.func attribute `{name}` is not representable in gpu_ir"
                        )));
                    }
                }
            });
        } else if name == "gpu.yield" || name == "gpu.return" {
            definition = definition.verifier(|ir, id, diagnostics| {
                let operation = ir.operation(id).expect("registered operation exists");
                if operation.parent.is_none() {
                    diagnostics.push(weave::Diagnostic::error(
                        "only gpu.func may be a root operation",
                    ));
                }
                if !operation.successors.is_empty() {
                    diagnostics.push(weave::Diagnostic::error(
                        "GPU terminators cannot have successors",
                    ));
                }
                if !operation.results.is_empty() || !operation.regions.is_empty() {
                    diagnostics.push(weave::Diagnostic::error(
                        "GPU terminators cannot produce results or own regions",
                    ));
                }
                if !matches!(
                    typed_attribute(operation, "source_block"),
                    Some(GpuAttribute::UInt(_))
                ) || !matches!(
                    typed_attribute(operation, "parameter_indices"),
                    Some(GpuAttribute::Indices(_))
                ) {
                    diagnostics.push(weave::Diagnostic::error(
                        "GPU terminators require source_block and parameter_indices metadata",
                    ));
                }
                reject_attributes_except(
                    operation,
                    &["source_block", "parameter_indices"],
                    diagnostics,
                );
                if let Some(block) = operation.parent.and_then(|id| ir.block(id))
                    && let Some(region) = ir.region(block.parent)
                    && let Some(owner) = ir.operation(region.parent)
                {
                    let expected = if owner.name.as_str() == "gpu.func" {
                        "gpu.return"
                    } else {
                        "gpu.yield"
                    };
                    if operation.name.as_str() != expected {
                        diagnostics.push(weave::Diagnostic::error(format!(
                            "{} region must end with {expected}",
                            owner.name
                        )));
                    }
                }
            });
        } else {
            let expected_regions = match name {
                "gpu.if" => 2,
                "gpu.loop" | "gpu.scope" => 1,
                _ => 0,
            };
            definition = definition.verifier(move |ir, id, diagnostics| {
                verify_operation(ir, id, expected_regions, diagnostics);
            });
        }
        registry.register_operation(
            OperationName::new(name).expect("static operation name"),
            definition,
        )?;
    }
    Ok(())
}

pub struct Conversion {
    pub ir: Ir,
    pub function: CoreOperationId,
    pub values: Vec<CoreValueId>,
    pub operations: Vec<CoreOperationId>,
    pub blocks: Vec<CoreBlockId>,
    pub regions: Vec<CoreRegionId>,
    pub callees: Vec<Arc<crate::DeviceFunction>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConversionError {
    Ir(weave::IrError),
    UnreachableBlock(usize),
    UnreachableRegion(usize),
    UnscheduledOperation(usize),
    ValueUsedBeforeDefinition(usize),
    Malformed(String),
}

impl std::fmt::Display for ConversionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}
impl std::error::Error for ConversionError {}
impl From<weave::IrError> for ConversionError {
    fn from(error: weave::IrError) -> Self {
        Self::Ir(error)
    }
}

pub fn to_weave(source: &Function) -> Result<Conversion, ConversionError> {
    let compacted;
    let source = if source.has_deletions() {
        compacted = source.compacted();
        &compacted
    } else {
        source
    };
    Converter::new(source).convert()
}

pub fn print(function: &Function) -> Result<String, ConversionError> {
    let conversion = to_weave(function)?;
    Ok(weave::print(&conversion.ir).expect("a converted function is structurally complete"))
}

pub fn parse(text: &str) -> Result<Ir, Vec<weave::Diagnostic>> {
    let registry = registry();
    weave::parse(text, &registry)
}

pub fn validate(ir: &Ir) -> Vec<weave::Diagnostic> {
    let mut diagnostics = weave::verify(ir, &registry());
    let functions = ir
        .operations()
        .filter(|(_, operation)| operation.name.as_str() == "gpu.func")
        .collect::<Vec<_>>();
    if ir.roots().len() != 1
        || functions.len() != 1
        || ir.roots().first() != functions.first().map(|(id, _)| id)
    {
        diagnostics.push(weave::Diagnostic::error(
            "GPU IR requires exactly one root gpu.func",
        ));
    }
    if let Some((_, function)) = functions.first()
        && function
            .regions
            .first()
            .and_then(|region| ir.region(*region))
            .is_none_or(|region| region.blocks.len() != 1)
    {
        diagnostics.push(weave::Diagnostic::error(
            "gpu.func requires exactly one entry block",
        ));
    }
    let gpu_operations = ir
        .operations()
        .filter(|(_, operation)| {
            !matches!(
                operation.name.as_str(),
                "gpu.func" | "gpu.yield" | "gpu.return"
            )
        })
        .collect::<Vec<_>>();
    let count = gpu_operations.len();
    let mut seen = vec![false; count];
    for (_, operation) in gpu_operations {
        match typed_attribute(operation, "source_operation") {
            Some(GpuAttribute::UInt(index)) if *index < count && !seen[*index] => {
                seen[*index] = true;
            }
            Some(GpuAttribute::UInt(_)) => diagnostics.push(weave::Diagnostic::error(
                "GPU source operation index is out of range or duplicated",
            )),
            _ => {}
        }
    }
    if seen.iter().any(|seen| !seen) {
        diagnostics.push(weave::Diagnostic::error(
            "GPU source operation indices must form a dense permutation",
        ));
    }
    diagnostics
}

pub fn registry() -> Registry {
    let mut registry = Registry::new();
    register(&mut registry).expect("a new registry is empty");
    registry
}

struct Converter<'a> {
    source: &'a Function,
    ir: Ir,
    function: CoreOperationId,
    values: Vec<Option<CoreValueId>>,
    operations: Vec<Option<CoreOperationId>>,
    blocks: Vec<Option<CoreBlockId>>,
    regions: Vec<Option<CoreRegionId>>,
    visited_blocks: BTreeSet<usize>,
    visited_regions: BTreeSet<usize>,
    callees: Vec<Arc<crate::DeviceFunction>>,
    constants: Vec<Arc<crate::op::ConstArray>>,
}

impl<'a> Converter<'a> {
    fn new(source: &'a Function) -> Self {
        let mut ir = Ir::new();
        let function = ir
            .create_operation(op_name("gpu.func"), [], [])
            .expect("an operand-free operation is valid");
        Self {
            source,
            ir,
            function,
            values: vec![None; source.num_values()],
            operations: vec![None; source.num_ops()],
            blocks: vec![None; source.num_blocks()],
            regions: vec![None; source.num_regions()],
            visited_blocks: BTreeSet::new(),
            visited_regions: BTreeSet::new(),
            callees: Vec::new(),
            constants: Vec::new(),
        }
    }

    fn convert(mut self) -> Result<Conversion, ConversionError> {
        self.ir.add_root(self.function)?;
        self.set_function_attributes()?;
        let body = self.ir.append_region(self.function)?;
        self.convert_block(self.source.entry(), body, true)?;

        for index in 0..self.source.num_blocks() {
            if !self.visited_blocks.contains(&index) {
                return Err(ConversionError::UnreachableBlock(index));
            }
        }
        for index in 0..self.source.num_regions() {
            if !self.visited_regions.contains(&index) {
                return Err(ConversionError::UnreachableRegion(index));
            }
        }
        for index in 0..self.source.num_ops() {
            if self.operations[index].is_none() {
                return Err(ConversionError::UnscheduledOperation(index));
            }
        }
        Ok(Conversion {
            ir: self.ir,
            function: self.function,
            values: collect(self.values),
            operations: collect(self.operations),
            blocks: collect(self.blocks),
            regions: collect(self.regions),
            callees: self.callees,
        })
    }

    fn set_function_attributes(&mut self) -> Result<(), weave::IrError> {
        for (index, declaration) in self.source.declarations().iter().enumerate() {
            let value = match declaration {
                Declaration::Param(param) => match param.ty {
                    ParamTy::Ptr(elem) => format!(
                        "param({},ptr,{})",
                        hex(param.name.as_bytes()),
                        crate::arena::text::scalar_name(elem)
                    ),
                    ParamTy::Scalar(scalar) => format!(
                        "param({},scalar,{})",
                        hex(param.name.as_bytes()),
                        crate::arena::text::scalar_name(scalar)
                    ),
                },
                Declaration::Shared(array) => format!(
                    "shared({},{},{})",
                    hex(array.name.as_bytes()),
                    crate::arena::text::scalar_name(array.elem),
                    array.len
                ),
            };
            self.attribute(
                self.function,
                format!("declaration_{index:06}"),
                GpuAttribute::from_payload(value),
            )?;
        }
        for (value, name) in self.source.names() {
            self.attribute(
                self.function,
                format!("name_{:06}", value.index()),
                GpuAttribute::Text(name.to_owned()),
            )?;
        }
        self.attribute(
            self.function,
            "entry",
            GpuAttribute::UInt(self.source.entry().index()),
        )?;
        Ok(())
    }

    fn convert_region(
        &mut self,
        source: RegionId,
        owner: CoreOperationId,
    ) -> Result<(), ConversionError> {
        self.visited_regions.insert(source.index());
        let region = self.ir.append_region(owner)?;
        self.regions[source.index()] = Some(region);
        for &block in &self.source.region(source).blocks {
            self.convert_block(block, region, false)?;
        }
        Ok(())
    }

    fn convert_block(
        &mut self,
        source: BlockId,
        region: CoreRegionId,
        entry: bool,
    ) -> Result<(), ConversionError> {
        self.visited_blocks.insert(source.index());
        let argument_types = self
            .source
            .block(source)
            .params
            .iter()
            .map(|&value| gpu_type(self.source.ty(value)))
            .collect::<Vec<_>>();
        let block = self.ir.append_block(region, argument_types)?;
        self.blocks[source.index()] = Some(block);
        let arguments = self.ir.block(block).expect("new block").arguments.clone();
        for (&source_value, core_value) in self.source.block(source).params.iter().zip(arguments) {
            self.values[source_value.index()] = Some(core_value);
        }

        for &operation in &self.source.block(source).ops {
            self.create_operation(operation, block)?;
        }
        for &operation in &self.source.block(source).ops {
            self.connect_operation(operation)?;
        }
        let yields = self
            .source
            .block(source)
            .terminator
            .yields
            .iter()
            .map(|&value| self.mapped_value(value))
            .collect::<Result<Vec<_>, _>>()?;
        let terminator = self.ir.create_operation(
            op_name(if entry { "gpu.return" } else { "gpu.yield" }),
            yields,
            [],
        )?;
        self.attribute(
            terminator,
            "source_block",
            GpuAttribute::UInt(source.index()),
        )?;
        self.attribute(
            terminator,
            "parameter_indices",
            GpuAttribute::Indices(
                self.source
                    .block(source)
                    .params
                    .iter()
                    .map(|value| Some(value.index()))
                    .collect(),
            ),
        )?;
        self.ir.append_operation(block, terminator)?;
        Ok(())
    }

    fn create_operation(
        &mut self,
        source_id: OpId,
        block: CoreBlockId,
    ) -> Result<(), ConversionError> {
        let source = self.source.op(source_id);
        let result_types = source
            .results
            .iter()
            .map(|&value| gpu_type(self.source.ty(value)))
            .collect::<Vec<_>>();
        let operation = self.ir.create_operation(
            op_name(operation_name(&source.kind)),
            std::iter::empty(),
            result_types,
        )?;
        self.ir.append_operation(block, operation)?;
        self.operations[source_id.index()] = Some(operation);
        self.attribute(
            operation,
            "source_operation",
            GpuAttribute::UInt(source_id.index()),
        )?;
        self.attribute(
            operation,
            "result_indices",
            GpuAttribute::Indices(
                source
                    .results
                    .iter()
                    .map(|value| Some(value.index()))
                    .collect(),
            ),
        )?;
        self.attribute(
            operation,
            "region_indices",
            GpuAttribute::Indices(
                source
                    .regions
                    .iter()
                    .map(|region| Some(region.index()))
                    .collect(),
            ),
        )?;
        let results = self
            .ir
            .operation(operation)
            .expect("new operation")
            .results
            .clone();
        for (&source_value, core_value) in source.results.iter().zip(results) {
            self.values[source_value.index()] = Some(core_value);
        }
        Ok(())
    }

    fn connect_operation(&mut self, source_id: OpId) -> Result<(), ConversionError> {
        let source = self.source.op(source_id);
        let operation = self.operations[source_id.index()].expect("operation was created");
        let mut operands = source
            .operands
            .iter()
            .map(|&value| self.mapped_value(value))
            .collect::<Result<Vec<_>, _>>()?;
        let mut destinations = Vec::with_capacity(source.into.len());
        for destination in &source.into {
            match destination {
                None => destinations.push("n".to_owned()),
                Some(value) => {
                    destinations.push(operands.len().to_string());
                    operands.push(self.mapped_value(*value)?);
                }
            }
        }
        self.ir.set_operands(operation, operands)?;
        self.attribute(operation, "materialized", GpuAttribute::Bool(source.placed))?;
        self.attribute(
            operation,
            "operand_count",
            GpuAttribute::UInt(source.operands.len()),
        )?;
        if !source.into.is_empty() {
            self.attribute(
                operation,
                "destinations",
                GpuAttribute::from_payload(format!("indices({})", destinations.join(","))),
            )?;
        }
        for (name, value) in operation_attributes(&source.kind) {
            self.attribute(operation, name, GpuAttribute::from_payload(value))?;
        }
        if let OpKind::ConstLoad(array) = &source.kind {
            let identity = self
                .constants
                .iter()
                .position(|held| Arc::ptr_eq(held, array))
                .unwrap_or_else(|| {
                    self.constants.push(array.clone());
                    self.constants.len() - 1
                });
            self.attribute(
                operation,
                "constant_identity",
                GpuAttribute::Constant(identity),
            )?;
        }
        if let OpKind::Call(callee) = &source.kind {
            let symbol = self
                .callees
                .iter()
                .position(|held| Arc::ptr_eq(held, callee))
                .unwrap_or_else(|| {
                    self.callees.push(callee.clone());
                    self.callees.len() - 1
                });
            self.attribute(operation, "callee_symbol", GpuAttribute::UInt(symbol))?;
        }
        for &region in &source.regions {
            self.convert_region(region, operation)?;
        }
        Ok(())
    }

    fn mapped_value(&self, value: ValueId) -> Result<CoreValueId, ConversionError> {
        self.values[value.index()].ok_or(ConversionError::ValueUsedBeforeDefinition(value.index()))
    }

    fn attribute(
        &mut self,
        operation: CoreOperationId,
        name: impl Into<String>,
        value: GpuAttribute,
    ) -> Result<(), weave::IrError> {
        self.ir
            .set_attribute(operation, name, Attribute::typed(value))?;
        Ok(())
    }
}

fn collect<T: Copy>(values: Vec<Option<T>>) -> Vec<T> {
    values
        .into_iter()
        .map(|value| value.expect("conversion checked completeness"))
        .collect()
}

fn gpu_type(ty: ValTy) -> Type {
    match ty {
        ValTy::Data(data) => Type::typed(DataType(data)),
        ValTy::Ptr(pointer) => Type::typed(PointerType(pointer)),
    }
}

fn op_name(name: &str) -> OperationName {
    OperationName::new(name).expect("static GPU operation name")
}

fn operation_name(kind: &OpKind) -> &'static str {
    match kind {
        OpKind::IntConst { .. } => "gpu.int_const",
        OpKind::FloatConst { .. } => "gpu.float_const",
        OpKind::Sreg(_) => "gpu.sreg",
        OpKind::Load { .. } => "gpu.load",
        OpKind::Offset => "gpu.offset",
        OpKind::Un(_) => "gpu.unary",
        OpKind::Bin(_) => "gpu.binary",
        OpKind::Fma => "gpu.fma",
        OpKind::Cmp(_) => "gpu.compare",
        OpKind::Select => "gpu.select",
        OpKind::Cast(_) => "gpu.cast",
        OpKind::Splat(_) => "gpu.splat",
        OpKind::Extract(_) => "gpu.extract",
        OpKind::Pack => "gpu.pack",
        OpKind::ShflDown => "gpu.shfl_down",
        OpKind::ActiveMask => "gpu.active_mask",
        OpKind::ShflFrom => "gpu.shfl_from",
        OpKind::WarpAny => "gpu.warp_any",
        OpKind::ConstLoad(_) => "gpu.const_load",
        OpKind::Bind => "gpu.bind",
        OpKind::Write { .. } => "gpu.write",
        OpKind::FetchAdd => "gpu.fetch_add",
        OpKind::If => "gpu.if",
        OpKind::Loop { .. } => "gpu.loop",
        OpKind::Scope => "gpu.scope",
        OpKind::Call(_) => "gpu.call",
        OpKind::Break => "gpu.break",
        OpKind::Ret => "gpu.ret",
        OpKind::Bar => "gpu.barrier",
    }
}

fn operation_attributes(kind: &OpKind) -> BTreeMap<&'static str, String> {
    let mut attributes = BTreeMap::new();
    let scalar = |value| format!("scalar({})", crate::arena::text::scalar_name(value));
    match kind {
        OpKind::IntConst { scalar: ty, value } => {
            attributes.insert("scalar", scalar(*ty));
            attributes.insert("value", format!("sint({value})"));
        }
        OpKind::FloatConst { scalar: ty, value } => {
            attributes.insert("scalar", scalar(*ty));
            attributes.insert("value", format!("f64_bits({:016x})", value.to_bits()));
        }
        OpKind::Sreg(value) => {
            attributes.insert(
                "register",
                format!("enum({})", crate::arena::text::sreg_name(*value)),
            );
        }
        OpKind::Load { lanes, nc, evict } => {
            attributes.insert("lanes", format!("uint({lanes})"));
            attributes.insert("non_coherent", format!("bool({nc})"));
            attributes.insert(
                "evict",
                format!("enum({})", crate::arena::text::evict_name(*evict)),
            );
        }
        OpKind::Un(value) => {
            attributes.insert(
                "operator",
                format!("enum({})", crate::arena::text::unary_name(*value)),
            );
        }
        OpKind::Bin(value) => {
            attributes.insert(
                "operator",
                format!("enum({})", crate::arena::text::binary_name(*value)),
            );
        }
        OpKind::Cmp(value) => {
            attributes.insert(
                "predicate",
                format!("enum({})", crate::arena::text::comparison_name(*value)),
            );
        }
        OpKind::Cast(value) => {
            attributes.insert("scalar", scalar(*value));
        }
        OpKind::Splat(lanes) => {
            attributes.insert("lanes", format!("uint({lanes})"));
        }
        OpKind::Extract(lane) => {
            attributes.insert("lane", format!("uint({lane})"));
        }
        OpKind::ConstLoad(array) => {
            attributes.insert("name", format!("text({})", hex(array.name.as_bytes())));
            attributes.insert("scalar", scalar(array.scalar));
            attributes.insert("data", format!("bytes({})", hex(&array.data)));
        }
        OpKind::Write { mode, evict } => {
            attributes.insert(
                "mode",
                format!("enum({})", crate::arena::text::write_mode_name(*mode)),
            );
            attributes.insert(
                "evict",
                format!("enum({})", crate::arena::text::evict_name(*evict)),
            );
        }
        OpKind::Loop { index, trip } => {
            attributes.insert("index", scalar(*index));
            attributes.insert(
                "trip",
                format!("enum({})", crate::arena::text::trip_name(*trip)),
            );
        }
        OpKind::Call(callee) => {
            attributes.insert("callee", format!("text({})", hex(callee.name.as_bytes())));
            attributes.insert(
                "parameter_count",
                format!("uint({})", callee.parameter_tys().len()),
            );
            attributes.insert(
                "result_count",
                format!("uint({})", callee.result_tys().len()),
            );
        }
        OpKind::Offset
        | OpKind::Fma
        | OpKind::Select
        | OpKind::Pack
        | OpKind::ShflDown
        | OpKind::ActiveMask
        | OpKind::ShflFrom
        | OpKind::WarpAny
        | OpKind::Bind
        | OpKind::FetchAdd
        | OpKind::If
        | OpKind::Scope
        | OpKind::Break
        | OpKind::Ret
        | OpKind::Bar => {}
    }
    attributes
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests;
