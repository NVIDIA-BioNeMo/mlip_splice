// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structural and type checking for a [`Module`], independent of how it was
//! built. Re-runs each op's [`Op::infer`](crate::op::Op::infer) against the
//! recorded value shapes (the builder and verifier share one inference oracle),
//! checks SSA def-before-use, and validates called-function references.

use crate::ir::{Def, Func, Module, ValueId};
use crate::op::{FuncSig, InferCtx};
use crate::shape::{Shape, ShapeError};

#[derive(Clone, Debug, PartialEq)]
pub enum VerifyError {
    /// An op's operands failed shape inference.
    Shape(ShapeError),
    /// A structural invariant was violated (SSA, ranges, arity, references).
    Structure { func: String, msg: String },
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerifyError::Shape(e) => write!(f, "shape error: {e}"),
            VerifyError::Structure { func, msg } => write!(f, "in `{func}`: {msg}"),
        }
    }
}

impl std::error::Error for VerifyError {}

fn structure(func: &str, msg: impl Into<String>) -> VerifyError {
    VerifyError::Structure {
        func: func.to_string(),
        msg: msg.into(),
    }
}

fn shape_of(f: &Func, v: ValueId) -> Result<&Shape, VerifyError> {
    f.values
        .get(v.index())
        .map(|d| &d.shape)
        .ok_or_else(|| structure(&f.name, format!("value %{} is out of range", v.index())))
}

fn func_sig(f: &Func) -> Result<FuncSig, VerifyError> {
    let params = f
        .params
        .iter()
        .map(|&p| shape_of(f, p).cloned())
        .collect::<Result<_, _>>()?;
    let results = f
        .results
        .iter()
        .map(|&r| shape_of(f, r).cloned())
        .collect::<Result<_, _>>()?;
    Ok(FuncSig { params, results })
}

pub fn verify(m: &Module) -> Result<(), VerifyError> {
    m.finite_map_graph
        .verify()
        .map_err(|msg| structure("<module>", msg))?;
    if m.entry.index() >= m.funcs.len() {
        return Err(structure("<module>", "entry function id is out of range"));
    }
    let sigs: Vec<FuncSig> = m.funcs.iter().map(func_sig).collect::<Result<_, _>>()?;
    for f in &m.funcs {
        verify_func(f, &sigs, m.funcs.len(), &m.finite_map_graph)?;
    }
    Ok(())
}

fn verify_func(
    f: &Func,
    sigs: &[FuncSig],
    num_funcs: usize,
    finite_map_graph: &domain_index::FiniteMapGraph,
) -> Result<(), VerifyError> {
    for (slot, &value) in f.params.iter().enumerate() {
        let Some(definition) = f.values.get(value.index()) else {
            return Err(structure(
                &f.name,
                format!(
                    "parameter {slot} references unknown value %{}",
                    value.index()
                ),
            ));
        };
        if definition.def != Def::Param(slot) {
            return Err(structure(
                &f.name,
                format!(
                    "parameter %{} has an inconsistent definition",
                    value.index()
                ),
            ));
        }
    }
    for (value, definition) in f.values.iter().enumerate() {
        definition.shape.element_count().map_err(|error| {
            structure(
                &f.name,
                format!("value %{value} has an invalid shape: {error}"),
            )
        })?;
        match definition.def {
            Def::Param(slot) => {
                if f.params.get(slot).copied() != Some(ValueId(value as u32)) {
                    return Err(structure(
                        &f.name,
                        format!("value %{value} has an orphaned parameter definition"),
                    ));
                }
            }
            Def::NodeResult { node, slot } => {
                if f.nodes
                    .get(node)
                    .and_then(|node| node.results.get(slot))
                    .copied()
                    != Some(ValueId(value as u32))
                {
                    return Err(structure(
                        &f.name,
                        format!("value %{value} has an orphaned node-result definition"),
                    ));
                }
            }
        }
        for &domain in &definition.shape.domain_axes {
            finite_map_graph.set(domain).map_err(|error| {
                structure(
                    &f.name,
                    format!("value %{value} has an invalid domain axis: {error}"),
                )
            })?;
        }
    }
    for (i, node) in f.nodes.iter().enumerate() {
        for &g in node.op.called_funcs() {
            if g.index() >= num_funcs {
                return Err(structure(
                    &f.name,
                    format!("node {i} references unknown function @{}", g.index()),
                ));
            }
        }

        let mut input_shapes: Vec<Shape> = Vec::with_capacity(node.inputs.len());
        for &v in &node.inputs {
            let def = f.values.get(v.index()).map(|d| d.def).ok_or_else(|| {
                structure(
                    &f.name,
                    format!("node {i} reads unknown value %{}", v.index()),
                )
            })?;
            if let Def::NodeResult { node: j, .. } = def
                && j >= i
            {
                return Err(structure(
                    &f.name,
                    format!("node {i} uses value %{} before it is defined", v.index()),
                ));
            }
            input_shapes.push(shape_of(f, v)?.clone());
        }

        let ctx = InferCtx::new(&input_shapes, sigs, finite_map_graph);
        let inferred = node.op.infer(&ctx).map_err(VerifyError::Shape)?;

        if inferred.len() != node.results.len() {
            return Err(structure(
                &f.name,
                format!(
                    "node {i} ({}) produced {} results but inference expects {}",
                    node.op.mnemonic(),
                    node.results.len(),
                    inferred.len()
                ),
            ));
        }
        for (slot, want) in inferred.iter().enumerate() {
            let r = node.results[slot];
            match f.values.get(r.index()).map(|d| d.def) {
                Some(Def::NodeResult { node: jn, slot: js }) if jn == i && js == slot => {}
                _ => {
                    return Err(structure(
                        &f.name,
                        format!(
                            "result %{} of node {i} has an inconsistent definition",
                            r.index()
                        ),
                    ));
                }
            }
            if shape_of(f, r)? != want {
                return Err(structure(
                    &f.name,
                    format!(
                        "node {i} ({}) result %{} shape disagrees with inference",
                        node.op.mnemonic(),
                        r.index()
                    ),
                ));
            }
        }
    }

    for &r in &f.results {
        shape_of(f, r)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ModuleBuilder;
    use crate::dtype::DType;

    fn module_with_unchecked_param_shape(shape: Shape) -> Module {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("unchecked_shape");
        let value = builder.param(Shape::scalar(DType::F32));
        let entry = builder.end_func(vec![value]);
        let mut module = builder.finish(entry).unwrap();
        module.funcs[entry.index()].values[value.index()].shape = shape;
        module
    }

    #[test]
    fn negative_dimension_is_rejected_even_when_another_dimension_is_zero() {
        let module = module_with_unchecked_param_shape(Shape::new(DType::F32, [-1, 0]));
        let error = verify(&module).unwrap_err().to_string();
        assert!(error.contains("value %0 has an invalid shape"), "{error}");
        assert!(error.contains("holds no countable elements"), "{error}");
    }

    #[test]
    fn dimension_product_past_usize_is_rejected() {
        let module = module_with_unchecked_param_shape(Shape::new(DType::F32, [i64::MAX, 3]));
        let error = verify(&module).unwrap_err().to_string();
        assert!(error.contains("value %0 has an invalid shape"), "{error}");
        assert!(error.contains("holds no countable elements"), "{error}");
    }

    #[test]
    fn zero_dimension_keeps_an_otherwise_overflowing_shape_valid() {
        let module = module_with_unchecked_param_shape(Shape::new(DType::F32, [i64::MAX, 3, 0]));
        verify(&module).unwrap();
    }
}
