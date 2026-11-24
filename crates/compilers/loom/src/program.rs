// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ops::Deref;
#[cfg(test)]
use std::ops::DerefMut;
use xir::dtype::DType;
use xir::ops::{Literal, ScalarOp};

use super::{FlatOperand, FlatProgram};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct InputElement {
    pub operand: usize,
    pub element: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum OutputFactor {
    Input(InputElement),
    Constant(Literal),
    Convert {
        from: DType,
        to: DType,
        input: Box<OutputFactor>,
    },
    Elementwise {
        op: ScalarOp,
        dtype: DType,
        inputs: Vec<OutputFactor>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct LiftedProgram {
    request: FlatProgram,
    output_factors: Vec<Vec<Vec<OutputFactor>>>,
    finite_map_work: xir_optimize::FiniteMapWork,
}

impl LiftedProgram {
    /// The bindings of the operands this program writes, in output order.
    pub(crate) fn outputs(&self) -> &[FlatOperand] {
        &self.request.operands[self.request.num_inputs..]
    }

    pub(super) fn request_mut(&mut self) -> &mut FlatProgram {
        &mut self.request
    }

    pub(super) fn output_factors(&self, output: usize, element: usize) -> &[OutputFactor] {
        self.output_factors
            .get(output)
            .and_then(|factors| factors.get(element))
            .map_or(&[], Vec::as_slice)
    }

    pub(super) fn set_output_factors(&mut self, factors: Vec<Vec<Vec<OutputFactor>>>) {
        self.output_factors = factors;
    }

    pub(super) fn has_output_factors(&self) -> bool {
        self.output_factors
            .iter()
            .flatten()
            .any(|factors| !factors.is_empty())
    }

    pub(super) fn has_indexed_ops(&self) -> bool {
        self.module.entry_func().nodes.iter().any(|node| {
            node.op.as_any().is::<xir::ops::Gather>() || node.op.as_any().is::<xir::ops::Scatter>()
        })
    }

    pub(super) fn keep_outputs(&mut self, outputs: &[usize]) {
        let entry = self.request.module.entry.index();
        let results = outputs
            .iter()
            .map(|&output| self.request.module.entry_func().results[output])
            .collect();
        let operands = self.request.operands[..self.request.num_inputs]
            .iter()
            .cloned()
            .chain(outputs.iter().map(|&output| self.outputs()[output].clone()))
            .collect();
        let output_factors = if self.output_factors.is_empty() {
            Vec::new()
        } else {
            outputs
                .iter()
                .map(|&output| self.output_factors[output].clone())
                .collect()
        };
        self.request.module.funcs[entry].results = results;
        self.request.operands = operands;
        self.output_factors = output_factors;
    }

    pub(super) fn finite_map_work(&self) -> xir_optimize::FiniteMapWork {
        self.finite_map_work
    }

    pub(super) fn set_finite_map_work(&mut self, work: xir_optimize::FiniteMapWork) {
        self.finite_map_work = work;
    }

    #[cfg(test)]
    pub(super) fn from_request_unchecked(request: FlatProgram) -> Self {
        Self {
            request,
            output_factors: Vec::new(),
            finite_map_work: xir_optimize::FiniteMapWork::default(),
        }
    }
}

impl Deref for LiftedProgram {
    type Target = FlatProgram;

    fn deref(&self) -> &Self::Target {
        &self.request
    }
}

#[cfg(test)]
impl DerefMut for LiftedProgram {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.request
    }
}

impl TryFrom<FlatProgram> for LiftedProgram {
    type Error = String;

    fn try_from(mut request: FlatProgram) -> Result<Self, Self::Error> {
        let graph = domain_index::Graph {
            domains: request.domains.clone(),
            index_maps: request.index_maps.clone(),
            aligned_views: request.available_views.clone(),
        };
        request.module.finite_map_graph = graph.finite_map_graph()?;
        validate(&request)?;
        Ok(LiftedProgram {
            request,
            output_factors: Vec::new(),
            finite_map_work: xir_optimize::FiniteMapWork::default(),
        })
    }
}

fn validate(request: &FlatProgram) -> Result<(), String> {
    xir::verify::verify(&request.module).map_err(|e| format!("loom XIR verification: {e}"))?;
    super::scalarize::validate_dots(&request.module)?;
    if request.lane_count <= 0 {
        return Err(format!(
            "loom lane_count must be positive, got {}",
            request.lane_count
        ));
    }
    if request.num_inputs > request.operands.len() {
        return Err(format!(
            "loom num_inputs={} exceeds operands.len()={}",
            request.num_inputs,
            request.operands.len()
        ));
    }
    let func = request.module.entry_func();
    if func.params.len() != request.num_inputs {
        return Err(format!(
            "loom entry has {} parameters, expected {} inputs",
            func.params.len(),
            request.num_inputs
        ));
    }
    let num_outputs = request.operands.len() - request.num_inputs;
    if func.results.len() != num_outputs {
        return Err(format!(
            "loom entry has {} results, expected {num_outputs} outputs",
            func.results.len()
        ));
    }
    if request.operands.is_empty() {
        return Err("loom needs at least one input or output operand".into());
    }
    for (b, binding) in request.operands.iter().enumerate() {
        if binding.axis_domains.len() != request.num_axes {
            return Err(format!(
                "loom operand {b} has {} axis domains, expected {} like the first operand",
                binding.axis_domains.len(),
                request.num_axes
            ));
        }
        if binding.lane_extent != 1 && binding.lane_extent != request.lane_count {
            return Err(format!(
                "loom operand {b} lane extent {} is neither 1 nor lane_count={}",
                binding.lane_extent, request.lane_count
            ));
        }
        visible_elements(&binding.element_shape)
            .map_err(|e| format!("loom operand {b} element shape: {e}"))?;
    }
    for (i, (&param, binding)) in func
        .params
        .iter()
        .zip(&request.operands[..request.num_inputs])
        .enumerate()
    {
        validate_signature(func.value_shape(param), binding, "parameter", i, true)?;
    }
    for (i, (&result, binding)) in func
        .results
        .iter()
        .zip(&request.operands[request.num_inputs..])
        .enumerate()
    {
        validate_signature(func.value_shape(result), binding, "result", i, false)?;
    }
    Ok(())
}

fn validate_signature(
    shape: &xir::shape::Shape,
    binding: &FlatOperand,
    kind: &str,
    index: usize,
    require_matching_dtype: bool,
) -> Result<(), String> {
    let dtype = validate_storage_dtype(binding.dtype)?;
    if shape.dims.as_slice() != binding.element_shape
        || require_matching_dtype && shape.dtype != dtype
    {
        return Err(format!(
            "loom {kind} {index} signature is {}{:?}, physical binding exposes {}{:?}",
            shape.dtype.name(),
            shape.dims,
            dtype.name(),
            binding.element_shape
        ));
    }
    if !require_matching_dtype && (shape.dtype == DType::Bool) != (dtype == DType::Bool) {
        return Err(format!(
            "loom result {index} does not support conversion between boolean and numeric dtypes"
        ));
    }
    Ok(())
}

fn validate_storage_dtype(dtype: DType) -> Result<DType, String> {
    if dtype.is_complex() {
        Err(format!("unsupported Loom dtype {}", dtype.name()))
    } else {
        Ok(dtype)
    }
}

fn visible_elements(dims: &[i64]) -> Result<usize, String> {
    dims.iter().try_fold(1usize, |product, &dim| {
        let dim = usize::try_from(dim).map_err(|_| format!("negative dimension {dim}"))?;
        if dim == 0 {
            return Err("zero-sized visible dimensions are not supported".into());
        }
        product
            .checked_mul(dim)
            .ok_or_else(|| format!("shape {dims:?} overflows usize"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use domain_index::{Domain, IndexMap};
    use xir::builder::ModuleBuilder;
    use xir::shape::Shape;

    fn request(dtype: DType, xir_dtype: DType) -> FlatProgram {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("main");
        let x = builder.param(Shape::new(xir_dtype, [3]));
        let main = builder.end_func(vec![x]);
        let module = builder.finish(main).unwrap();
        let binding = || FlatOperand {
            dtype,
            element_shape: vec![3],
            lane_extent: 8,
            axis_domains: vec![0],
        };
        FlatProgram {
            module,
            lane_count: 8,
            domains: vec![Domain { extent: 2 }],
            index_maps: Vec::new(),
            num_axes: 1,
            operands: vec![binding(), binding()],
            num_inputs: 1,
            available_views: Vec::new(),
            name: "typed_program".into(),
        }
    }

    #[test]
    fn accepts_all_scalar_storage_types() {
        for (dtype, xir_dtype) in [
            (DType::Bool, DType::Bool),
            (DType::I8, DType::I8),
            (DType::I16, DType::I16),
            (DType::I32, DType::I32),
            (DType::I64, DType::I64),
            (DType::U8, DType::U8),
            (DType::U16, DType::U16),
            (DType::U32, DType::U32),
            (DType::U64, DType::U64),
            (DType::F16, DType::F16),
            (DType::Bf16, DType::Bf16),
            (DType::F32, DType::F32),
            (DType::F64, DType::F64),
        ] {
            LiftedProgram::try_from(request(dtype, xir_dtype)).unwrap();
        }
    }

    #[test]
    fn rejects_complex_storage() {
        let error =
            LiftedProgram::try_from(request(DType::Complex64, DType::Complex64)).unwrap_err();
        assert!(error.contains("unsupported Loom dtype"));
    }

    #[test]
    fn output_storage_may_narrow_the_result_dtype() {
        let mut request = request(DType::F16, DType::F32);
        request.operands[0].dtype = DType::F32;
        LiftedProgram::try_from(request).unwrap();
    }

    #[test]
    fn input_storage_must_match_the_parameter_dtype() {
        let error = LiftedProgram::try_from(request(DType::F16, DType::F32)).unwrap_err();
        assert!(error.contains("loom parameter 0 signature"));
    }

    #[test]
    fn carries_index_maps_into_xir() {
        let mut request = request(DType::F32, DType::F32);
        request.domains.push(Domain { extent: 5 });
        request.index_maps.push(IndexMap {
            source_id: 1,
            target_id: 0,
            is_injective: false,
            is_sorted: false,
            may_have_unmapped: false,
        });

        let program = LiftedProgram::try_from(request).unwrap();

        assert_eq!(program.module.finite_map_graph.sets.len(), 2);
        assert_eq!(program.module.finite_map_graph.maps.len(), 1);
    }

    #[test]
    fn keeping_outputs_keeps_their_factors_aligned() {
        let mut request = request(DType::F32, DType::F32);
        let result = request.module.entry_func().results[0];
        let output = request.operands[1].clone();
        request.module.funcs[request.module.entry.index()]
            .results
            .push(result);
        request.operands.push(output);
        let mut program = LiftedProgram::try_from(request).unwrap();
        let factors = |element| {
            vec![vec![OutputFactor::Input(InputElement {
                operand: 0,
                element,
            })]]
        };
        program.set_output_factors(vec![factors(0), factors(1)]);

        program.keep_outputs(&[1]);

        assert_eq!(program.outputs().len(), 1);
        assert_eq!(program.module.entry_func().results, [result]);
        assert_eq!(
            program.output_factors(0, 0),
            &[OutputFactor::Input(InputElement {
                operand: 0,
                element: 1,
            })]
        );
    }
}
