// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Import of canonical `weave` XIR into the compatibility graph model.

use std::collections::{BTreeMap, HashMap};

use smallvec::SmallVec;
use weave::{Attribute, AttributeData, Ir, OperationId};

use crate::dialect::{
    ConversionError, FiniteMapGraphAttribute, FunctionRefsAttribute, OpReconstructor,
    SymbolNameAttribute, SymbolRefAttribute, TensorType, registry,
};
use crate::ir::{Def, Func, FuncId, Module, Node, ValueDef, ValueId as XirValueId};
use crate::shape::Shape;

pub fn from_ir(ir: &Ir) -> Result<Module, ConversionError> {
    let dialect = registry();
    let diagnostics = weave::verify(ir, &dialect);
    if !diagnostics.is_empty() {
        return Err(ConversionError::InvalidCore(
            diagnostics
                .into_iter()
                .map(|diagnostic| diagnostic.message)
                .collect::<Vec<_>>()
                .join("; "),
        ));
    }
    if ir.roots().len() != 1 {
        return Err(invalid_core("XIR requires exactly one module root"));
    }
    let module = ir
        .operation(ir.roots()[0])
        .ok_or_else(|| invalid_core("missing module"))?;
    if module.name.as_str() != "xir.module" || module.regions.len() != 1 {
        return Err(invalid_core("root must be one xir.module"));
    }
    let graph = attribute::<FiniteMapGraphAttribute>(&module.attributes, "finite_map_graph")?
        .0
        .clone();
    let entry_name = attribute::<SymbolRefAttribute>(&module.attributes, "entry")?
        .0
        .clone();
    let region = ir
        .region(module.regions[0])
        .ok_or_else(|| invalid_core("missing module region"))?;
    if region.blocks.len() != 1 {
        return Err(invalid_core("xir.module requires one block"));
    }
    let block = ir
        .block(region.blocks[0])
        .ok_or_else(|| invalid_core("missing module block"))?;
    let function_ops = block
        .operations
        .iter()
        .filter_map(|id| {
            ir.operation(*id)
                .filter(|op| op.name.as_str() == "xir.func")
                .map(|_| *id)
        })
        .collect::<Vec<_>>();
    if block.operations.len() != function_ops.len() + 1
        || block
            .operations
            .last()
            .and_then(|id| ir.operation(*id))
            .is_none_or(|op| op.name.as_str() != "xir.module_end")
    {
        return Err(invalid_core(
            "xir.module may contain only functions followed by xir.module_end",
        ));
    }
    let names = function_ops
        .iter()
        .map(|id| {
            let op = ir.operation(*id).expect("collected operation");
            attribute::<SymbolNameAttribute>(&op.attributes, "symbol").map(|name| name.0.clone())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut ids = HashMap::new();
    for (index, name) in names.iter().enumerate() {
        if ids.insert(name.clone(), FuncId(index as u32)).is_some() {
            return Err(invalid_core(format!("duplicate function symbol `{name}`")));
        }
    }
    let entry = *ids
        .get(&entry_name)
        .ok_or_else(|| invalid_core(format!("unknown entry symbol `{entry_name}`")))?;
    let funcs = function_ops
        .into_iter()
        .zip(names)
        .map(|(id, name)| import_function(ir, id, name, &ids, &dialect))
        .collect::<Result<Vec<_>, _>>()?;
    let module = Module {
        finite_map_graph: graph,
        funcs,
        entry,
    };
    crate::verify::verify(&module)?;
    Ok(module)
}

fn import_function(
    ir: &Ir,
    id: OperationId,
    name: String,
    symbols: &HashMap<String, FuncId>,
    dialect: &weave::Registry,
) -> Result<Func, ConversionError> {
    let operation = ir
        .operation(id)
        .ok_or_else(|| invalid_core("missing function"))?;
    if operation.regions.len() != 1 {
        return Err(invalid_core(format!(
            "function `{name}` requires one region"
        )));
    }
    let region = ir
        .region(operation.regions[0])
        .ok_or_else(|| invalid_core("missing function region"))?;
    if region.blocks.len() != 1 {
        return Err(invalid_core(format!(
            "function `{name}` requires one block"
        )));
    }
    let block = ir
        .block(region.blocks[0])
        .ok_or_else(|| invalid_core("missing function block"))?;
    let mut values = Vec::new();
    let mut map = HashMap::new();
    let mut params = Vec::new();
    for (index, value) in block.arguments.iter().enumerate() {
        let shape = value_shape(ir, *value)?;
        let legacy = XirValueId(values.len() as u32);
        values.push(ValueDef {
            shape,
            def: Def::Param(index),
        });
        params.push(legacy);
        map.insert(*value, legacy);
    }
    let mut nodes = Vec::new();
    let mut results = None;
    for operation_id in &block.operations {
        let op = ir
            .operation(*operation_id)
            .ok_or_else(|| invalid_core("missing function operation"))?;
        if op.name.as_str() == "xir.return" {
            results = Some(
                op.operands
                    .iter()
                    .map(|value| {
                        map.get(value)
                            .copied()
                            .ok_or_else(|| invalid_core("return uses unknown value"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
            continue;
        }
        let mnemonic = op.name.mnemonic();
        if matches!(mnemonic, "module" | "module_end" | "func") {
            return Err(invalid_core(format!(
                "xir.{mnemonic} cannot appear in a function"
            )));
        }
        let inputs = op
            .operands
            .iter()
            .map(|value| {
                map.get(value)
                    .copied()
                    .ok_or_else(|| invalid_core("operation uses unknown value"))
            })
            .collect::<Result<SmallVec<[XirValueId; 4]>, _>>()?;
        let result_shapes = op
            .results
            .iter()
            .map(|value| value_shape(ir, *value))
            .collect::<Result<SmallVec<[Shape; 1]>, _>>()?;
        let callees = op
            .attributes
            .get("callees")
            .map(|attribute| {
                attribute
                    .downcast_ref::<FunctionRefsAttribute>()
                    .ok_or_else(|| invalid_core("`callees` must be #xir.function_refs"))
                    .and_then(|refs| {
                        refs.0
                            .iter()
                            .map(|name| {
                                symbols.get(name).copied().ok_or_else(|| {
                                    invalid_core(format!("unknown function symbol `{name}`"))
                                })
                            })
                            .collect::<Result<SmallVec<[FuncId; 2]>, _>>()
                    })
            })
            .transpose()?
            .unwrap_or_default();
        let definition = dialect
            .operation(&op.name)
            .ok_or_else(|| invalid_core(format!("unknown XIR operation `{mnemonic}`")))?;
        let reconstructor = definition
            .get_interface::<OpReconstructor>()
            .ok_or_else(|| {
                invalid_core(format!("xir.{mnemonic} has no reconstruction interface"))
            })?;
        let imported = reconstructor.reconstruct(&op.attributes, &callees, &result_shapes)?;
        let node_index = nodes.len();
        let mut node_results = SmallVec::new();
        for (slot, (&core, shape)) in op.results.iter().zip(result_shapes).enumerate() {
            let legacy = XirValueId(values.len() as u32);
            values.push(ValueDef {
                shape,
                def: Def::NodeResult {
                    node: node_index,
                    slot,
                },
            });
            node_results.push(legacy);
            map.insert(core, legacy);
        }
        nodes.push(Node {
            op: imported.into(),
            inputs,
            results: node_results,
        });
    }
    Ok(Func {
        name,
        params,
        nodes,
        results: results.ok_or_else(|| invalid_core("function has no xir.return"))?,
        values,
    })
}

fn value_shape(ir: &Ir, value: weave::ValueId) -> Result<Shape, ConversionError> {
    ir.value(value)
        .and_then(|value| value.ty.downcast_ref::<TensorType>())
        .map(|ty| ty.0.clone())
        .ok_or_else(|| invalid_core("XIR value must have !xir.tensor type"))
}

fn attribute<'a, T: AttributeData>(
    attributes: &'a BTreeMap<String, Attribute>,
    name: &str,
) -> Result<&'a T, ConversionError> {
    attributes
        .get(name)
        .and_then(Attribute::downcast_ref::<T>)
        .ok_or_else(|| invalid_core(format!("attribute `{name}` has the wrong XIR type")))
}

fn invalid_core(message: impl Into<String>) -> ConversionError {
    ConversionError::InvalidCore(message.into())
}
