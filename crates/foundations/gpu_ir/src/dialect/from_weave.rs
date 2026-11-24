// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use weave::{Attribute, Ir, Operation, Type, ValueId};

use super::{
    ConversionError, DataType, GpuAttribute, PointerType, operation_attribute_is_representable,
};
use crate::arena::text::{
    At, RawBlock, RawDef, RawFunction, RawKind, RawOp, RawRegion, RawValue, Ref,
};
use crate::op::{ConstArray, OpKind};
use crate::{DeviceFunction, Function, ValTy};

pub fn from_weave(ir: &Ir, callees: &[Arc<DeviceFunction>]) -> Result<Function, ConversionError> {
    let root = one(ir.roots(), "GPU function root")?;
    let function = ir
        .operation(root)
        .ok_or_else(|| error("missing GPU function"))?;
    if function.name.as_str() != "gpu.func" || function.regions.len() != 1 {
        return Err(error("the root must be gpu.func with one body region"));
    }
    if ir
        .operations()
        .filter(|(_, operation)| operation.name.as_str() == "gpu.func")
        .count()
        != 1
    {
        return Err(error("gpu.func must be the sole function operation"));
    }
    if !function.operands.is_empty()
        || !function.results.is_empty()
        || !function.successors.is_empty()
    {
        return Err(error(
            "gpu.func operands, results, and successors are not representable",
        ));
    }
    for name in function.attributes.keys() {
        if name != "entry" && !name.starts_with("declaration_") && !name.starts_with("name_") {
            return Err(error("gpu.func has an unrepresentable attribute"));
        }
    }

    let mut raw = RawFunction {
        line: 1,
        column: 1,
        entry: Some(reference(uint(function, "entry")?)),
        ..RawFunction::default()
    };
    raw.declarations = numbered(function, "declaration_")?
        .into_iter()
        .map(|(_, value)| match gpu_attribute(value)? {
            GpuAttribute::Declaration(value) => Ok(value.clone()),
            _ => Err(error("declaration has the wrong attribute family")),
        })
        .collect::<Result<_, _>>()?;
    raw.names = numbered(function, "name_")?
        .into_iter()
        .map(|(index, value)| match gpu_attribute(value)? {
            GpuAttribute::Text(value) => Ok((reference(index), value.clone())),
            _ => Err(error("value name has the wrong attribute family")),
        })
        .collect::<Result<_, _>>()?;

    let body = function.regions[0];
    if ir
        .region(body)
        .is_none_or(|region| region.blocks.len() != 1)
    {
        return Err(error("gpu.func requires exactly one entry block"));
    }
    let mut region_indices = HashMap::new();
    for (_, operation) in ir
        .operations()
        .filter(|(_, operation)| is_gpu_op(operation))
    {
        let indices = indices(operation, "region_indices")?;
        if indices.len() != operation.regions.len() {
            return Err(error("region index arity mismatch"));
        }
        for (&region, index) in operation.regions.iter().zip(indices) {
            region_indices.insert(region, required(index, "region")?);
        }
    }

    let mut block_indices = HashMap::new();
    let mut value_indices = HashMap::new();
    for (block_id, block) in ir.blocks() {
        let terminator = block
            .operations
            .last()
            .and_then(|id| ir.operation(*id))
            .ok_or_else(|| error("a GPU block needs a terminator"))?;
        let expected_terminator = if block.parent == body {
            "gpu.return"
        } else {
            "gpu.yield"
        };
        if terminator.name.as_str() != expected_terminator
            || !terminator.results.is_empty()
            || !terminator.regions.is_empty()
            || !terminator.successors.is_empty()
            || terminator
                .attributes
                .keys()
                .any(|name| !matches!(name.as_str(), "source_block" | "parameter_indices"))
        {
            return Err(error("GPU block has an unrepresentable terminator"));
        }
        let source_block = uint(terminator, "source_block")?;
        block_indices.insert(block_id, source_block);
        let parameters = indices(terminator, "parameter_indices")?;
        if parameters.len() != block.arguments.len() {
            return Err(error("block parameter index arity mismatch"));
        }
        for (&value, index) in block.arguments.iter().zip(parameters) {
            value_indices.insert(value, required(index, "parameter")?);
        }
    }
    for (_, operation) in ir
        .operations()
        .filter(|(_, operation)| is_gpu_op(operation))
    {
        let results = indices(operation, "result_indices")?;
        if results.len() != operation.results.len() {
            return Err(error("result index arity mismatch"));
        }
        for (&value, index) in operation.results.iter().zip(results) {
            value_indices.insert(value, required(index, "result")?);
        }
    }

    let value_count = ir.values().count();
    let mut values = (0..value_count).map(|_| None).collect::<Vec<_>>();
    for (value_id, value) in ir.values() {
        let index = *value_indices
            .get(&value_id)
            .ok_or_else(|| error("value has no source index"))?;
        let def = match value.definition {
            weave::ValueDefinition::BlockArgument { block, index } => RawDef::Param {
                block: reference(
                    *block_indices
                        .get(&block)
                        .ok_or_else(|| error("missing block index"))?,
                ),
                index,
            },
            weave::ValueDefinition::OperationResult { operation, index } => RawDef::Result {
                op: reference(uint(
                    ir.operation(operation)
                        .ok_or_else(|| error("missing defining operation"))?,
                    "source_operation",
                )?),
                index,
            },
        };
        put(
            &mut values,
            index,
            RawValue {
                ty: gpu_type(&value.ty)?,
                def,
                at: At::default(),
            },
            "value",
        )?;
    }
    raw.values = collect(values, "value")?;

    let operation_count = ir
        .operations()
        .filter(|(_, operation)| is_gpu_op(operation))
        .count();
    let mut operations = (0..operation_count).map(|_| None).collect::<Vec<_>>();
    let mut constants: HashMap<usize, Arc<ConstArray>> = HashMap::new();
    for (_, operation) in ir
        .operations()
        .filter(|(_, operation)| is_gpu_op(operation))
    {
        if !operation.successors.is_empty() {
            return Err(error("GPU operation successors are not representable"));
        }
        if operation
            .attributes
            .keys()
            .any(|name| !operation_attribute_is_representable(operation.name.as_str(), name))
        {
            return Err(error("GPU operation has an unrepresentable attribute"));
        }
        let index = uint(operation, "source_operation")?;
        let operand_count = uint(operation, "operand_count")?;
        if operand_count > operation.operands.len() {
            return Err(error("operand_count exceeds operand arity"));
        }
        let source_value = |value: &ValueId| {
            value_indices
                .get(value)
                .copied()
                .map(reference)
                .ok_or_else(|| error("operand has no source index"))
        };
        if operation.name.as_str() == "gpu.call" {
            let symbol = uint(operation, "callee_symbol")?;
            let callee = callees
                .get(symbol)
                .ok_or_else(|| error("callee symbol is unresolved"))?;
            let name = match named(operation, "callee")? {
                GpuAttribute::Text(name) => name,
                _ => return Err(error("callee name is not text")),
            };
            let parameter_count = uint(operation, "parameter_count")?;
            let result_count = uint(operation, "result_count")?;
            if name != &callee.name
                || parameter_count != callee.parameter_tys().len()
                || result_count != callee.result_tys().len()
                || operand_count != parameter_count
                || operation.results.len() != result_count
            {
                return Err(error(
                    "gpu.call attributes do not match its callee and arities",
                ));
            }
        }
        let destinations = optional_indices(operation, "destinations")?;
        let into = destinations
            .into_iter()
            .map(|slot| match slot {
                None => Ok(None),
                Some(slot) => operation
                    .operands
                    .get(slot)
                    .ok_or_else(|| error("destination slot exceeds operands"))
                    .and_then(source_value)
                    .map(Some),
            })
            .collect::<Result<_, _>>()?;
        let placed = match named(operation, "materialized")? {
            GpuAttribute::Bool(value) => *value,
            _ => return Err(error("materialized is not boolean")),
        };
        put(
            &mut operations,
            index,
            RawOp {
                kind: decode_kind(operation, &mut constants)?,
                placed,
                operands: operation.operands[..operand_count]
                    .iter()
                    .map(source_value)
                    .collect::<Result<_, _>>()?,
                results: operation
                    .results
                    .iter()
                    .map(source_value)
                    .collect::<Result<_, _>>()?,
                into,
                regions: operation
                    .regions
                    .iter()
                    .map(|region| {
                        region_indices
                            .get(region)
                            .copied()
                            .map(reference)
                            .ok_or_else(|| error("region has no source index"))
                    })
                    .collect::<Result<_, _>>()?,
                at: At::default(),
            },
            "operation",
        )?;
    }
    raw.ops = collect(operations, "operation")?;

    let mut blocks = (0..ir.blocks().count()).map(|_| None).collect::<Vec<_>>();
    for (block_id, block) in ir.blocks() {
        let source_block = *block_indices
            .get(&block_id)
            .ok_or_else(|| error("block has no source index"))?;
        let terminator_id = block
            .operations
            .last()
            .copied()
            .ok_or_else(|| error("a GPU block needs a terminator"))?;
        let terminator = ir
            .operation(terminator_id)
            .ok_or_else(|| error("missing terminator operation"))?;
        let expected_terminator = if block.parent == body {
            "gpu.return"
        } else {
            "gpu.yield"
        };
        if terminator.name.as_str() != expected_terminator || !terminator.successors.is_empty() {
            return Err(error("GPU block has an unrepresentable terminator"));
        }
        let source_op = |id| {
            uint(
                ir.operation(id).ok_or_else(|| error("missing operation"))?,
                "source_operation",
            )
            .map(reference)
        };
        put(
            &mut blocks,
            source_block,
            RawBlock {
                region: (block.parent != body)
                    .then(|| {
                        region_indices
                            .get(&block.parent)
                            .copied()
                            .map(reference)
                            .ok_or_else(|| error("block region has no source index"))
                    })
                    .transpose()?,
                params: block
                    .arguments
                    .iter()
                    .map(source_value_ref(&value_indices))
                    .collect::<Result<_, _>>()?,
                ops: block.operations[..block.operations.len() - 1]
                    .iter()
                    .copied()
                    .map(source_op)
                    .collect::<Result<_, _>>()?,
                yields: terminator
                    .operands
                    .iter()
                    .map(source_value_ref(&value_indices))
                    .collect::<Result<_, _>>()?,
                at: At::default(),
            },
            "block",
        )?;
    }
    raw.blocks = collect(blocks, "block")?;

    let mut regions = (0..region_indices.len()).map(|_| None).collect::<Vec<_>>();
    for (region_id, index) in &region_indices {
        let region = ir
            .region(*region_id)
            .ok_or_else(|| error("missing region"))?;
        let owner = ir
            .operation(region.parent)
            .ok_or_else(|| error("missing region owner"))?;
        put(
            &mut regions,
            *index,
            RawRegion {
                owner: Some(reference(uint(owner, "source_operation")?)),
                blocks: region
                    .blocks
                    .iter()
                    .map(|block| {
                        block_indices
                            .get(block)
                            .copied()
                            .map(reference)
                            .ok_or_else(|| error("region block has no source index"))
                    })
                    .collect::<Result<_, _>>()?,
                at: At::default(),
            },
            "region",
        )?;
    }
    raw.regions = collect(regions, "region")?;
    raw.build(&|symbol| callees.get(symbol).cloned())
        .map_err(|failure| error(&failure.to_string()))
}

fn decode_kind(
    operation: &Operation,
    constants: &mut HashMap<usize, Arc<ConstArray>>,
) -> Result<RawKind, ConversionError> {
    let scalar = |name| match named(operation, name)? {
        GpuAttribute::Scalar(value) => Ok(*value),
        _ => Err(error("expected scalar attribute")),
    };
    let symbol = |name| match named(operation, name)? {
        GpuAttribute::Symbol(value) => Ok(value.as_str()),
        _ => Err(error("expected enum attribute")),
    };
    let uint = |name| match named(operation, name)? {
        GpuAttribute::UInt(value) => Ok(*value),
        _ => Err(error("expected uint attribute")),
    };
    let direct = |kind| Ok(RawKind::Direct(kind));
    match operation.name.as_str() {
        "gpu.int_const" => direct(OpKind::IntConst {
            scalar: scalar("scalar")?,
            value: match named(operation, "value")? {
                GpuAttribute::SInt(value) => *value,
                _ => return Err(error("expected integer")),
            },
        }),
        "gpu.float_const" => direct(OpKind::FloatConst {
            scalar: scalar("scalar")?,
            value: f64::from_bits(match named(operation, "value")? {
                GpuAttribute::F64Bits(value) => *value,
                _ => return Err(error("expected float bits")),
            }),
        }),
        "gpu.sreg" => direct(OpKind::Sreg(
            crate::arena::text::sreg_named(symbol("register")?)
                .ok_or_else(|| error("unknown register"))?,
        )),
        "gpu.load" => direct(OpKind::Load {
            lanes: u8::try_from(uint("lanes")?).map_err(|_| error("lanes exceed u8"))?,
            nc: match named(operation, "non_coherent")? {
                GpuAttribute::Bool(value) => *value,
                _ => return Err(error("expected bool")),
            },
            evict: crate::arena::text::evict_named(symbol("evict")?)
                .ok_or_else(|| error("unknown eviction"))?,
        }),
        "gpu.offset" => direct(OpKind::Offset),
        "gpu.unary" => direct(OpKind::Un(
            crate::arena::text::unary_named(symbol("operator")?)
                .ok_or_else(|| error("unknown unary"))?,
        )),
        "gpu.binary" => direct(OpKind::Bin(
            crate::arena::text::binary_named(symbol("operator")?)
                .ok_or_else(|| error("unknown binary"))?,
        )),
        "gpu.fma" => direct(OpKind::Fma),
        "gpu.compare" => direct(OpKind::Cmp(
            crate::arena::text::comparison_named(symbol("predicate")?)
                .ok_or_else(|| error("unknown comparison"))?,
        )),
        "gpu.select" => direct(OpKind::Select),
        "gpu.cast" => direct(OpKind::Cast(scalar("scalar")?)),
        "gpu.splat" => direct(OpKind::Splat(
            u8::try_from(uint("lanes")?).map_err(|_| error("lanes exceed u8"))?,
        )),
        "gpu.extract" => direct(OpKind::Extract(
            u8::try_from(uint("lane")?).map_err(|_| error("lane exceeds u8"))?,
        )),
        "gpu.pack" => direct(OpKind::Pack),
        "gpu.shfl_down" => direct(OpKind::ShflDown),
        "gpu.active_mask" => direct(OpKind::ActiveMask),
        "gpu.shfl_from" => direct(OpKind::ShflFrom),
        "gpu.warp_any" => direct(OpKind::WarpAny),
        "gpu.const_load" => {
            let identity = match named(operation, "constant_identity")? {
                GpuAttribute::Constant(value) => *value,
                _ => return Err(error("expected constant identity")),
            };
            let candidate = ConstArray {
                name: match named(operation, "name")? {
                    GpuAttribute::Text(value) => value.clone(),
                    _ => return Err(error("expected text")),
                },
                scalar: scalar("scalar")?,
                data: match named(operation, "data")? {
                    GpuAttribute::Bytes(value) => value.clone().into(),
                    _ => return Err(error("expected bytes")),
                },
            };
            let array = match constants.get(&identity) {
                Some(array)
                    if array.name == candidate.name
                        && array.scalar == candidate.scalar
                        && array.data == candidate.data =>
                {
                    array.clone()
                }
                Some(_) => return Err(error("one constant identity has conflicting payloads")),
                None => {
                    let array = Arc::new(candidate);
                    constants.insert(identity, array.clone());
                    array
                }
            };
            direct(OpKind::ConstLoad(array))
        }
        "gpu.bind" => direct(OpKind::Bind),
        "gpu.write" => direct(OpKind::Write {
            mode: crate::arena::text::write_mode_named(symbol("mode")?)
                .ok_or_else(|| error("unknown write mode"))?,
            evict: crate::arena::text::evict_named(symbol("evict")?)
                .ok_or_else(|| error("unknown eviction"))?,
        }),
        "gpu.fetch_add" => direct(OpKind::FetchAdd),
        "gpu.if" => direct(OpKind::If),
        "gpu.loop" => direct(OpKind::Loop {
            index: scalar("index")?,
            trip: crate::arena::text::trip_named(symbol("trip")?)
                .ok_or_else(|| error("unknown trip"))?,
        }),
        "gpu.scope" => direct(OpKind::Scope),
        "gpu.call" => Ok(RawKind::Call(reference(uint("callee_symbol")?))),
        "gpu.break" => direct(OpKind::Break),
        "gpu.ret" => direct(OpKind::Ret),
        "gpu.barrier" => direct(OpKind::Bar),
        name => Err(error(&format!("unsupported GPU operation `{name}`"))),
    }
}

fn source_value_ref(
    indices: &HashMap<ValueId, usize>,
) -> impl Fn(&ValueId) -> Result<Ref, ConversionError> + '_ {
    |value| {
        indices
            .get(value)
            .copied()
            .map(reference)
            .ok_or_else(|| error("value has no source index"))
    }
}
fn is_gpu_op(operation: &Operation) -> bool {
    !matches!(
        operation.name.as_str(),
        "gpu.func" | "gpu.yield" | "gpu.return"
    )
}
fn gpu_type(ty: &Type) -> Result<ValTy, ConversionError> {
    if let Some(value) = ty.downcast_ref::<DataType>() {
        Ok(ValTy::Data(value.0))
    } else if let Some(value) = ty.downcast_ref::<PointerType>() {
        Ok(ValTy::Ptr(value.0))
    } else {
        Err(error("non-GPU value type"))
    }
}
fn gpu_attribute(value: &Attribute) -> Result<&GpuAttribute, ConversionError> {
    value
        .downcast_ref()
        .ok_or_else(|| error("attribute is not typed GPU data"))
}
fn named<'a>(operation: &'a Operation, name: &str) -> Result<&'a GpuAttribute, ConversionError> {
    gpu_attribute(
        operation
            .attributes
            .get(name)
            .ok_or_else(|| error(&format!("missing `{name}`")))?,
    )
}
fn uint(operation: &Operation, name: &str) -> Result<usize, ConversionError> {
    match named(operation, name)? {
        GpuAttribute::UInt(value) => Ok(*value),
        _ => Err(error("expected uint attribute")),
    }
}
fn indices(operation: &Operation, name: &str) -> Result<Vec<Option<usize>>, ConversionError> {
    match named(operation, name)? {
        GpuAttribute::Indices(value) => Ok(value.clone()),
        _ => Err(error("expected indices attribute")),
    }
}
fn optional_indices(
    operation: &Operation,
    name: &str,
) -> Result<Vec<Option<usize>>, ConversionError> {
    operation
        .attributes
        .get(name)
        .map(gpu_attribute)
        .transpose()?
        .map(|value| match value {
            GpuAttribute::Indices(value) => Ok(value.clone()),
            _ => Err(error("expected indices attribute")),
        })
        .transpose()
        .map(|value| value.unwrap_or_default())
}
fn numbered<'a>(
    operation: &'a Operation,
    prefix: &str,
) -> Result<Vec<(usize, &'a Attribute)>, ConversionError> {
    let mut values = operation
        .attributes
        .iter()
        .filter_map(|(name, value)| name.strip_prefix(prefix).map(|index| (index, value)))
        .map(|(index, value)| {
            index
                .parse()
                .map(|index| (index, value))
                .map_err(|_| error("invalid numbered attribute"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    values.sort_by_key(|(index, _)| *index);
    Ok(values)
}
fn one<T: Copy>(values: &[T], what: &str) -> Result<T, ConversionError> {
    if let [value] = values {
        Ok(*value)
    } else {
        Err(error(&format!("expected one {what}")))
    }
}
fn reference(index: usize) -> Ref {
    Ref {
        index,
        at: At::default(),
    }
}
fn required(value: Option<usize>, what: &str) -> Result<usize, ConversionError> {
    value.ok_or_else(|| error(&format!("missing {what} index")))
}
fn put<T>(
    values: &mut [Option<T>],
    index: usize,
    value: T,
    what: &str,
) -> Result<(), ConversionError> {
    let slot = values
        .get_mut(index)
        .ok_or_else(|| error(&format!("{what} index out of range")))?;
    if slot.is_some() {
        return Err(error(&format!("duplicate {what} index")));
    }
    *slot = Some(value);
    Ok(())
}
fn collect<T>(values: Vec<Option<T>>, what: &str) -> Result<Vec<T>, ConversionError> {
    values
        .into_iter()
        .map(|value| value.ok_or_else(|| error(&format!("missing {what} index"))))
        .collect()
}
fn error(message: &str) -> ConversionError {
    ConversionError::Malformed(message.to_owned())
}
