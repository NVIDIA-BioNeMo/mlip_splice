// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reconstruction of concrete XIR operations from dialect attributes.

use std::collections::BTreeMap;

use domain_index::FiniteMapId;
use smallvec::SmallVec;
use weave::Attribute;

use crate::dialect::{
    BoolAttribute, BoolsAttribute, BytesAttribute, ConversionError, EnumAttribute,
    ExactValuesAttribute, IntegerAttribute, IntegerPairsAttribute, IntegersAttribute,
};
use crate::dim_numbers::{
    ConvDimensionNumbers, DotDimensionNumbers, GatherDimensionNumbers, ScatterDimensionNumbers,
};
use crate::dtype::DType;
use crate::ir::FuncId;
use crate::op::Op;
use crate::ops::{
    BroadcastInDim, Call, Case, Clamp, Compare, ComparisonDirection, ComparisonType, Concatenate,
    Constant, Convert, Convolution, DotAlgorithm, DotGeneral, DynamicSlice, DynamicUpdateSlice,
    Fft, FftType, Gather, GatherAlongMap, Iota, Literal, Pad, Padding, PrecisionType, Reduce,
    ReduceWindow, Reshape, Reverse, Scatter, Select, Slice, Sort, SumAlongMap, Transpose, While,
};
use crate::shape::{Dims, Shape};

pub(crate) fn reconstruct(
    mnemonic: &str,
    attributes: &BTreeMap<String, Attribute>,
    callees: &[FuncId],
    result_shapes: &[Shape],
) -> Result<Box<dyn Op>, ConversionError> {
    if let Some(op) = crate::builder::unary_from_name(mnemonic)
        .or_else(|| crate::builder::binary_from_name(mnemonic))
    {
        return Ok(op);
    }
    match mnemonic {
        "select" => return Ok(Box::new(Select)),
        "clamp" => return Ok(Box::new(Clamp)),
        _ => {}
    }
    let op: Box<dyn Op> = match mnemonic {
        "dynamic_update_slice" => Box::new(DynamicUpdateSlice {}),
        "compare" => Box::new(Compare {
            comparison_direction: comparison_direction(enum_value(
                attributes,
                "comparison_direction",
            )?)?,
            compare_type: attributes
                .get("compare_type")
                .map(|_| enum_value(attributes, "compare_type").and_then(comparison_type))
                .transpose()?,
        }),
        "convert" => Box::new(Convert {
            dtype: dtype(enum_value(attributes, "dtype")?)?,
        }),
        "broadcast_in_dim" => Box::new(BroadcastInDim {
            out_dims: dims(attributes, "out_dims")?,
            broadcast_dimensions: axes(attributes, "broadcast_dimensions")?,
        }),
        "transpose" => Box::new(Transpose {
            permutation: axes(attributes, "permutation")?,
        }),
        "reshape" => Box::new(Reshape {
            out_dims: dims(attributes, "out_dims")?,
        }),
        "slice" => Box::new(Slice {
            start_indices: dims(attributes, "start_indices")?,
            limit_indices: dims(attributes, "limit_indices")?,
            strides: dims(attributes, "strides")?,
        }),
        "concatenate" => Box::new(Concatenate {
            dimension: u32_value(attributes, "dimension")?,
        }),
        "pad" => Box::new(Pad {
            edge_padding_low: dims(attributes, "edge_padding_low")?,
            edge_padding_high: dims(attributes, "edge_padding_high")?,
            interior_padding: dims(attributes, "interior_padding")?,
        }),
        "reverse" => Box::new(Reverse {
            dimensions: axes(attributes, "dimensions")?,
        }),
        "iota" => Box::new(Iota {
            shape: one_result(result_shapes, mnemonic)?,
            iota_dimension: u32_value(attributes, "iota_dimension")?,
        }),
        "constant" => {
            let literal = Literal::new(
                one_result(result_shapes, mnemonic)?,
                bytes(attributes, "data")?,
            );
            let literal = match attributes.get("exact") {
                Some(_) => literal
                    .with_exact(get::<ExactValuesAttribute>(attributes, "exact")?.0.clone())
                    .map_err(error)?,
                None => literal,
            };
            Box::new(Constant { literal })
        }
        "dynamic_slice" => Box::new(DynamicSlice {
            slice_sizes: dims(attributes, "slice_sizes")?,
        }),
        "gather" => Box::new(Gather {
            dnums: GatherDimensionNumbers {
                offset_dims: axes(attributes, "offset_dims")?,
                collapsed_slice_dims: axes(attributes, "collapsed_slice_dims")?,
                operand_batching_dims: axes(attributes, "operand_batching_dims")?,
                start_indices_batching_dims: axes(attributes, "start_indices_batching_dims")?,
                start_index_map: axes(attributes, "start_index_map")?,
                index_vector_dim: u32_value(attributes, "index_vector_dim")?,
            },
            slice_sizes: dims(attributes, "slice_sizes")?,
            indices_are_sorted: bool_value(attributes, "indices_are_sorted")?,
        }),
        "dot_general" => Box::new(DotGeneral {
            dnums: DotDimensionNumbers {
                lhs_contracting_dimensions: axes(attributes, "lhs_contracting_dimensions")?,
                rhs_contracting_dimensions: axes(attributes, "rhs_contracting_dimensions")?,
                lhs_batching_dimensions: axes(attributes, "lhs_batching_dimensions")?,
                rhs_batching_dimensions: axes(attributes, "rhs_batching_dimensions")?,
            },
            algorithm: dot_algorithm(enum_value(attributes, "algorithm")?)?,
        }),
        "convolution" => Box::new(Convolution {
            dnums: ConvDimensionNumbers {
                input_batch_dimension: u32_value(attributes, "input_batch_dimension")?,
                input_feature_dimension: u32_value(attributes, "input_feature_dimension")?,
                input_spatial_dimensions: axes(attributes, "input_spatial_dimensions")?,
                kernel_input_feature_dimension: u32_value(
                    attributes,
                    "kernel_input_feature_dimension",
                )?,
                kernel_output_feature_dimension: u32_value(
                    attributes,
                    "kernel_output_feature_dimension",
                )?,
                kernel_spatial_dimensions: axes(attributes, "kernel_spatial_dimensions")?,
                output_batch_dimension: u32_value(attributes, "output_batch_dimension")?,
                output_feature_dimension: u32_value(attributes, "output_feature_dimension")?,
                output_spatial_dimensions: axes(attributes, "output_spatial_dimensions")?,
            },
            window_strides: dims(attributes, "window_strides")?,
            padding: pairs(attributes, "padding")?,
            lhs_dilation: dims(attributes, "lhs_dilation")?,
            rhs_dilation: dims(attributes, "rhs_dilation")?,
            window_reversal: bools(attributes, "window_reversal")?.unwrap_or_default(),
            feature_group_count: integer(attributes, "feature_group_count")?,
            batch_group_count: integer(attributes, "batch_group_count")?,
        }),
        "fft" => Box::new(Fft {
            fft_type: fft_type(enum_value(attributes, "fft_type")?)?,
            fft_length: dims(attributes, "fft_length")?,
        }),
        "reduce" => Box::new(Reduce {
            dimensions: axes(attributes, "dimensions")?,
            reducer: callee(callees, 0, mnemonic)?,
        }),
        "reduce_window" => Box::new(ReduceWindow {
            window_dimensions: dims(attributes, "window_dimensions")?,
            window_strides: dims(attributes, "window_strides")?,
            base_dilations: dims(attributes, "base_dilations")?,
            window_dilations: dims(attributes, "window_dilations")?,
            padding: pairs(attributes, "padding")?,
            reducer: callee(callees, 0, mnemonic)?,
        }),
        "scatter" => Box::new(Scatter {
            dnums: ScatterDimensionNumbers {
                update_window_dims: axes(attributes, "update_window_dims")?,
                inserted_window_dims: axes(attributes, "inserted_window_dims")?,
                input_batching_dims: axes(attributes, "input_batching_dims")?,
                scatter_indices_batching_dims: axes(attributes, "scatter_indices_batching_dims")?,
                scatter_dims_to_operand_dims: axes(attributes, "scatter_dims_to_operand_dims")?,
                index_vector_dim: u32_value(attributes, "index_vector_dim")?,
            },
            indices_are_sorted: bool_value(attributes, "indices_are_sorted")?,
            unique_indices: bool_value(attributes, "unique_indices")?,
            update: callee(callees, 0, mnemonic)?,
        }),
        "sort" => Box::new(Sort {
            dimension: u32_value(attributes, "dimension")?,
            is_stable: bool_value(attributes, "is_stable")?,
            comparator: callee(callees, 0, mnemonic)?,
        }),
        "while" => Box::new(While::new(
            callee(callees, 0, mnemonic)?,
            callee(callees, 1, mnemonic)?,
        )),
        "case" => Box::new(Case {
            branches: callees.iter().copied().collect(),
        }),
        "call" => Box::new(Call {
            callee: callee(callees, 0, mnemonic)?,
        }),
        "gather_along_map" => Box::new(GatherAlongMap {
            map: FiniteMapId(u32_value(attributes, "map")?),
            axis: u32_value(attributes, "axis")?,
        }),
        "sum_along_map" => Box::new(SumAlongMap {
            map: FiniteMapId(u32_value(attributes, "map")?),
            axis: u32_value(attributes, "axis")?,
        }),
        _ => return Err(error(format!("no concrete factory for xir.{mnemonic}"))),
    };
    Ok(op)
}

fn get<'a, T: weave::AttributeData>(
    attributes: &'a BTreeMap<String, Attribute>,
    name: &str,
) -> Result<&'a T, ConversionError> {
    attributes
        .get(name)
        .and_then(Attribute::downcast_ref::<T>)
        .ok_or_else(|| error(format!("attribute `{name}` has the wrong type")))
}
fn integer(a: &BTreeMap<String, Attribute>, n: &str) -> Result<i64, ConversionError> {
    Ok(get::<IntegerAttribute>(a, n)?.0)
}
fn u32_value(a: &BTreeMap<String, Attribute>, n: &str) -> Result<u32, ConversionError> {
    u32::try_from(integer(a, n)?).map_err(|_| error(format!("attribute `{n}` is outside u32")))
}
fn bool_value(a: &BTreeMap<String, Attribute>, n: &str) -> Result<bool, ConversionError> {
    Ok(get::<BoolAttribute>(a, n)?.0)
}
fn enum_value<'a>(a: &'a BTreeMap<String, Attribute>, n: &str) -> Result<&'a str, ConversionError> {
    Ok(&get::<EnumAttribute>(a, n)?.0)
}
fn dims(a: &BTreeMap<String, Attribute>, n: &str) -> Result<Dims, ConversionError> {
    Ok(get::<IntegersAttribute>(a, n)?.0.iter().copied().collect())
}
fn axes(a: &BTreeMap<String, Attribute>, n: &str) -> Result<SmallVec<[u32; 4]>, ConversionError> {
    get::<IntegersAttribute>(a, n)?
        .0
        .iter()
        .map(|&v| {
            u32::try_from(v)
                .map_err(|_| error(format!("attribute `{n}` contains value outside u32")))
        })
        .collect()
}
fn bytes(a: &BTreeMap<String, Attribute>, n: &str) -> Result<Vec<u8>, ConversionError> {
    Ok(get::<BytesAttribute>(a, n)?.0.clone())
}
fn pairs(a: &BTreeMap<String, Attribute>, n: &str) -> Result<Padding, ConversionError> {
    Ok(get::<IntegerPairsAttribute>(a, n)?
        .0
        .iter()
        .copied()
        .collect())
}
fn bools(
    a: &BTreeMap<String, Attribute>,
    n: &str,
) -> Result<Option<SmallVec<[bool; 4]>>, ConversionError> {
    a.get(n)
        .map(|_| get::<BoolsAttribute>(a, n).map(|v| v.0.iter().copied().collect()))
        .transpose()
}
fn callee(values: &[FuncId], index: usize, mnemonic: &str) -> Result<FuncId, ConversionError> {
    values
        .get(index)
        .copied()
        .ok_or_else(|| error(format!("xir.{mnemonic} requires callee {index}")))
}
fn one_result(values: &[Shape], mnemonic: &str) -> Result<Shape, ConversionError> {
    if let [value] = values {
        Ok(value.clone())
    } else {
        Err(error(format!("xir.{mnemonic} requires one result")))
    }
}

fn dtype(value: &str) -> Result<DType, ConversionError> {
    match value {
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::Bf16),
        "f32" => Ok(DType::F32),
        "f64" => Ok(DType::F64),
        "i1" => Ok(DType::Bool),
        "i8" => Ok(DType::I8),
        "i16" => Ok(DType::I16),
        "i32" => Ok(DType::I32),
        "i64" => Ok(DType::I64),
        "u8" => Ok(DType::U8),
        "u16" => Ok(DType::U16),
        "u32" => Ok(DType::U32),
        "u64" => Ok(DType::U64),
        "c64" => Ok(DType::Complex64),
        "c128" => Ok(DType::Complex128),
        _ => Err(error("invalid dtype")),
    }
}
fn comparison_direction(value: &str) -> Result<ComparisonDirection, ConversionError> {
    match value {
        "eq" => Ok(ComparisonDirection::Eq),
        "ne" => Ok(ComparisonDirection::Ne),
        "lt" => Ok(ComparisonDirection::Lt),
        "le" => Ok(ComparisonDirection::Le),
        "gt" => Ok(ComparisonDirection::Gt),
        "ge" => Ok(ComparisonDirection::Ge),
        _ => Err(error("invalid comparison direction")),
    }
}
fn comparison_type(value: &str) -> Result<ComparisonType, ConversionError> {
    match value {
        "float" => Ok(ComparisonType::Float),
        "totalorder" => Ok(ComparisonType::TotalOrder),
        "signed" => Ok(ComparisonType::Signed),
        "unsigned" => Ok(ComparisonType::Unsigned),
        _ => Err(error("invalid comparison type")),
    }
}
fn fft_type(value: &str) -> Result<FftType, ConversionError> {
    match value {
        "fft" => Ok(FftType::Fft),
        "ifft" => Ok(FftType::Ifft),
        "rfft" => Ok(FftType::Rfft),
        "irfft" => Ok(FftType::Irfft),
        _ => Err(error("invalid FFT type")),
    }
}
fn dot_algorithm(value: &str) -> Result<DotAlgorithm, ConversionError> {
    let preset = match value {
        "f16_f16_f16" => Some(DotAlgorithm::f16_f16_f16()),
        "f16_f16_f32" => Some(DotAlgorithm::f16_f16_f32()),
        "bf16_bf16_f32" => Some(DotAlgorithm::bf16_bf16_f32()),
        "bf16_bf16_f32_x3" => Some(DotAlgorithm::bf16_bf16_f32_x3()),
        "tf32_tf32_f32" => Some(DotAlgorithm::tf32_tf32_f32()),
        "tf32_tf32_f32_x3" => Some(DotAlgorithm::tf32_tf32_f32_x3()),
        "f32_f32_f32" => Some(DotAlgorithm::f32_f32_f32()),
        "f64_f64_f64" => Some(DotAlgorithm::f64_f64_f64()),
        _ => None,
    };
    if let Some(preset) = preset {
        return Ok(preset);
    }
    let body = value
        .strip_prefix('{')
        .and_then(|value| value.strip_suffix('}'))
        .ok_or_else(|| error("invalid dot algorithm"))?;
    let fields = body
        .split(", ")
        .map(|field| {
            field
                .split_once('=')
                .ok_or_else(|| error("invalid dot algorithm field"))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let precision = |name| {
        fields
            .get(name)
            .and_then(|value| PrecisionType::parse(value))
            .ok_or_else(|| error(format!("invalid dot algorithm `{name}`")))
    };
    let number = |name| {
        fields
            .get(name)
            .ok_or_else(|| error(format!("missing dot algorithm `{name}`")))?
            .parse()
            .map_err(|_| error(format!("invalid dot algorithm `{name}`")))
    };
    Ok(DotAlgorithm {
        lhs: precision("lhs")?,
        rhs: precision("rhs")?,
        accum: precision("accum")?,
        lhs_components: number("lhs_components")?,
        rhs_components: number("rhs_components")?,
        num_primitive_ops: number("num_primitive_ops")?,
        allow_imprecise_accum: fields
            .get("allow_imprecise_accum")
            .ok_or_else(|| error("missing dot algorithm `allow_imprecise_accum`"))?
            .parse()
            .map_err(|_| error("invalid dot algorithm `allow_imprecise_accum`"))?,
    })
}
fn error(message: impl Into<String>) -> ConversionError {
    ConversionError::InvalidCore(message.into())
}
