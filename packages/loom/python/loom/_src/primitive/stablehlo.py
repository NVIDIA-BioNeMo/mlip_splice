# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re

import numpy as np
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo

_BINARY = {
    f"stablehlo.{name}": name
    for name in (
        "add",
        "subtract",
        "multiply",
        "divide",
        "remainder",
        "power",
        "maximum",
        "minimum",
        "and",
        "or",
        "xor",
        "shift_left",
        "shift_right_arithmetic",
        "shift_right_logical",
        "atan2",
    )
}
_UNARY = {
    f"stablehlo.{name}": name
    for name in (
        "negate",
        "not",
        "abs",
        "sqrt",
        "rsqrt",
        "sign",
        "cbrt",
        "exponential",
        "exponential_minus_one",
        "log",
        "log_plus_one",
        "logistic",
        "tanh",
        "sine",
        "cosine",
        "tan",
        "floor",
        "ceil",
        "round_nearest_even",
        "round_nearest_afz",
        "is_finite",
        "count_leading_zeros",
        "popcnt",
    )
}
_DT_SIZE = {
    "i1": 1,
    "i8": 1,
    "u8": 1,
    "f16": 2,
    "bf16": 2,
    "i16": 2,
    "u16": 2,
    "f32": 4,
    "i32": 4,
    "u32": 4,
    "f64": 8,
    "i64": 8,
    "u64": 8,
}
_EXACT_COMPOSITE = "algebraic_real.exact_constant"
_EXACT_ATTRIBUTE = "algebraic_real"


def _dtype(t) -> str:
    dtype = str(ir.RankedTensorType(t).element_type)
    return "u" + dtype[2:] if dtype.startswith("ui") else dtype


def _shape(t) -> list[int]:
    return list(ir.RankedTensorType(t).shape)


def _dense_i64(attr) -> list[int]:
    return [int(value) for value in ir.DenseI64ArrayAttr(attr)]


def _int_attr(attr) -> int:
    return int(ir.IntegerAttr(attr).value)


def _bool_attr(attr) -> bool:
    return bool(ir.BoolAttr(attr).value)


def _compare_direction(attr) -> str:
    value = str(attr)
    for direction in ("EQ", "NE", "LT", "LE", "GT", "GE"):
        if direction in value:
            return direction.lower()
    raise NotImplementedError(f"StableHLO comparison direction {value!r}")


def _compare_type(attrs) -> str:
    if "compare_type" not in attrs:
        return ""
    value = str(attrs["compare_type"])
    for compare_type in ("TOTALORDER", "FLOAT", "SIGNED", "UNSIGNED"):
        if compare_type in value:
            return compare_type
    raise NotImplementedError(f"StableHLO comparison type {value!r}")


def _bf16_bits(token: str) -> int:
    """One `bf16` element of a dense literal, as the sixteen bits it is.

    MLIR writes an element either as a decimal or, where a decimal would not say
    it — an infinity, a NaN — as the bit pattern in hex.
    """
    token = token.strip()
    if token.startswith(("0x", "0X")):
        return int(token, 16)
    held = np.asarray(float(token), dtype=np.dtype("bfloat16"))
    return int(held.view(np.uint16))


def _bf16_bytes(attr, count: int) -> bytes:
    """The elements of a `bf16` dense attribute, exactly as it holds them.

    Every other element type is read as an array through the buffer numpy
    understands. `bf16` has no numpy type to read it as, so it is read from what
    MLIR wrote — and MLIR writes a dense attribute one of three ways: a splat,
    a bracketed literal, or, past a size it chooses, the raw little-endian
    elements in hex. A coefficient table of any size reaches that third form,
    which is the elements themselves and decodes without passing through a
    decimal at all.
    """
    dense = ir.DenseElementsAttr(attr)
    if dense.is_splat:
        held = np.asarray(
            float(ir.FloatAttr(dense.get_splat_value()).value),
            dtype=np.dtype("bfloat16"),
        )
        return np.full(count, held, dtype=np.dtype("bfloat16")).tobytes()
    text = str(attr)
    if (raw := re.search(r'dense<"0x([0-9A-Fa-f]*)">', text)) is not None:
        return bytes.fromhex(raw.group(1))
    literal = re.search(r"dense<(.*)>\s*:\s*tensor<", text, re.DOTALL)
    if literal is None:
        raise ValueError(f"cannot decode bf16 constant {attr}")
    tokens = literal.group(1).replace("[", " ").replace("]", " ").split(",")
    bits = [_bf16_bits(token) for token in tokens if token.strip()]
    return np.asarray(bits, dtype=np.uint16).tobytes()


def _const_bytes(attr, dtype: str, count: int) -> bytes:
    numpy_dtypes = {
        "f16": np.float16,
        "f32": np.float32,
        "f64": np.float64,
        "i8": np.int8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
        "u8": np.uint8,
        "u16": np.uint16,
        "u32": np.uint32,
        "u64": np.uint64,
        "i1": np.bool_,
    }
    if dtype == "bf16":
        data = _bf16_bytes(attr, count)
        held = len(data) // _DT_SIZE["bf16"]
        if held != count:
            raise ValueError(f"bf16 constant has {held} values, expected {count}")
        return data

    numpy_dtype = numpy_dtypes.get(dtype)
    if numpy_dtype is None or dtype not in _DT_SIZE:
        raise NotImplementedError(f"StableHLO constant dtype {dtype!r}")
    values = np.asarray(ir.DenseElementsAttr(attr)).astype(numpy_dtype, copy=False)
    if values.size == 1 and count != 1:
        values = np.broadcast_to(values, (count,))
    data = np.ascontiguousarray(values).tobytes()
    if len(data) != count * _DT_SIZE[dtype]:
        raise ValueError(
            f"StableHLO constant has {len(data)} bytes, expected {count * _DT_SIZE[dtype]}"
        )
    return data


def _composite_string(attrs, name: str) -> str | None:
    if "composite_attributes" not in attrs:
        return None
    for entry in ir.DictAttr(attrs["composite_attributes"]):
        if entry.name == name:
            return ir.StringAttr(entry.attr).value
    return None


def _exact_data(attrs, count: int) -> list[list] | None:
    payload = _composite_string(attrs, _EXACT_ATTRIBUTE)
    if payload is None:
        return None
    schema = json.loads(payload)
    if not isinstance(schema, list) or len(schema) != 2 or schema[0] != 1:
        raise ValueError("unsupported AlgebraicReal payload schema")
    values = schema[1]
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"StableHLO exact constant has the wrong value count; expected {count}")
    for value in values:
        if not isinstance(value, list) or not value or value[0] not in ("m", "p"):
            raise ValueError("invalid AlgebraicReal payload value")
    return values


def _dot_algorithm(op, operand_dtype: str) -> dict:
    attrs = op.operation.attributes
    if "algorithm" in attrs:
        algorithm = stablehlo.DotAlgorithm(attrs["algorithm"])
        return {
            "lhs_precision": str(algorithm.lhs_precision_type),
            "rhs_precision": str(algorithm.rhs_precision_type),
            "accum_precision": str(algorithm.accumulation_type),
            "lhs_components": algorithm.lhs_component_count,
            "rhs_components": algorithm.rhs_component_count,
            "num_primitive_ops": algorithm.num_primitive_operations,
            "allow_imprecise_accum": algorithm.allow_imprecise_accumulation,
        }
    presets = {
        "f16": ("f16", "f16", "f32"),
        "bf16": ("bf16", "bf16", "f32"),
        "f64": ("f64", "f64", "f64"),
    }
    lhs, rhs, accum = presets.get(operand_dtype, ("tf32", "tf32", "f32"))
    return {
        "lhs_precision": lhs,
        "rhs_precision": rhs,
        "accum_precision": accum,
        "lhs_components": 1,
        "rhs_components": 1,
        "num_primitive_ops": 1,
        "allow_imprecise_accum": False,
    }


class _Walker:
    def __init__(self, builder, functions):
        self.builder = builder
        self.functions = functions
        self.next_region = 0

    def build_func(self, name, block):
        self.builder.begin_func(name)
        values = {
            arg: self.builder.param(_dtype(arg.type), _shape(arg.type)) for arg in block.arguments
        }
        return self.builder.end_func(self.walk(block, values))

    def walk(self, block, values):
        for op in block.operations:
            name = op.operation.name
            if name in ("func.return", "stablehlo.return"):
                return [values[value] for value in op.operation.operands]
            self.lower_op(op, values)
        raise NotImplementedError("StableHLO function has no return")

    def region(self, op, index=0):
        self.next_region += 1
        return self.build_func(
            f"loom_region_{self.next_region}", op.operation.regions[index].blocks[0]
        )

    def lower_op(self, op, values):
        name = op.operation.name
        attrs = op.operation.attributes
        operands = list(op.operation.operands)
        results = list(op.operation.results)
        inputs = [values[value] for value in operands]
        result_type = results[0].type if results else None

        if name in _BINARY:
            output = self.builder.binary(_BINARY[name], inputs[0], inputs[1])
        elif name in _UNARY:
            output = self.builder.unary(_UNARY[name], inputs[0])
        elif name == "stablehlo.compare":
            output = self.builder.compare(
                _compare_direction(attrs["comparison_direction"]),
                _compare_type(attrs),
                inputs[0],
                inputs[1],
            )
        elif name == "stablehlo.select":
            output = self.builder.select(inputs[0], inputs[1], inputs[2])
        elif name == "stablehlo.clamp":
            output = self.builder.clamp(inputs[0], inputs[1], inputs[2])
        elif name == "stablehlo.convert":
            output = self.builder.convert(inputs[0], _dtype(result_type))
        elif name == "stablehlo.reshape":
            output = self.builder.reshape(inputs[0], _shape(result_type))
        elif name == "stablehlo.broadcast_in_dim":
            output = self.builder.broadcast(
                inputs[0],
                _shape(result_type),
                _dense_i64(attrs["broadcast_dimensions"]),
            )
        elif name == "stablehlo.transpose":
            output = self.builder.transpose(inputs[0], _dense_i64(attrs["permutation"]))
        elif name == "stablehlo.slice":
            output = self.builder.slice(
                inputs[0],
                _dense_i64(attrs["start_indices"]),
                _dense_i64(attrs["limit_indices"]),
                _dense_i64(attrs["strides"]),
            )
        elif name == "stablehlo.concatenate":
            output = self.builder.concatenate(inputs, _int_attr(attrs["dimension"]))
        elif name == "stablehlo.pad":
            output = self.builder.pad(
                inputs[0],
                inputs[1],
                _dense_i64(attrs["edge_padding_low"]),
                _dense_i64(attrs["edge_padding_high"]),
                _dense_i64(attrs["interior_padding"]),
            )
        elif name == "stablehlo.constant":
            dtype = _dtype(result_type)
            dims = _shape(result_type)
            count = int(np.prod(dims, dtype=np.int64)) if dims else 1
            data = _const_bytes(attrs["value"], dtype, count)
            output = self.builder.constant(dtype, dims, data)
        elif name == "stablehlo.composite":
            composite = ir.StringAttr(attrs["name"]).value
            if composite != _EXACT_COMPOSITE:
                raise NotImplementedError(
                    f"Loom does not support StableHLO composite {composite!r}"
                )
            if len(operands) != 1 or len(results) != 1:
                raise ValueError(f"StableHLO {_EXACT_COMPOSITE} must have one input and one result")
            if operands[0].type != result_type:
                raise ValueError(f"StableHLO {_EXACT_COMPOSITE} input and result types differ")
            constant = operands[0].owner
            if constant.name != "stablehlo.constant":
                raise ValueError(f"StableHLO {_EXACT_COMPOSITE} input must be a constant")
            dtype = _dtype(result_type)
            dims = _shape(result_type)
            count = int(np.prod(dims, dtype=np.int64)) if dims else 1
            data = _const_bytes(constant.attributes["value"], dtype, count)
            algebraic_values = _exact_data(attrs, count)
            if algebraic_values is None:
                raise ValueError(
                    f"StableHLO {_EXACT_COMPOSITE} has no {_EXACT_ATTRIBUTE!r} attribute"
                )
            output = self.builder.exact_constant(dtype, dims, data, algebraic_values)
        elif name == "stablehlo.reduce":
            arity = len(operands) // 2
            output = self.builder.reduce(
                inputs[:arity],
                inputs[arity:],
                _dense_i64(attrs["dimensions"]),
                self.region(op),
            )
        elif name == "stablehlo.dot_general":
            dimensions = stablehlo.DotDimensionNumbers(attrs["dot_dimension_numbers"])
            output = self.builder.dot(
                inputs[0],
                inputs[1],
                list(dimensions.lhs_contracting_dimensions),
                list(dimensions.rhs_contracting_dimensions),
                list(dimensions.lhs_batching_dimensions),
                list(dimensions.rhs_batching_dimensions),
                **_dot_algorithm(op, _dtype(operands[0].type)),
            )
        elif name == "stablehlo.gather":
            dimensions = stablehlo.GatherDimensionNumbers(attrs["dimension_numbers"])
            output = self.builder.gather(
                inputs[0],
                inputs[1],
                list(dimensions.offset_dims),
                list(dimensions.collapsed_slice_dims),
                list(dimensions.operand_batching_dims),
                list(dimensions.start_indices_batching_dims),
                list(dimensions.start_index_map),
                dimensions.index_vector_dim,
                _dense_i64(attrs["slice_sizes"]),
                _bool_attr(attrs["indices_are_sorted"]) if "indices_are_sorted" in attrs else False,
            )
        elif name == "stablehlo.scatter":
            dimensions = stablehlo.ScatterDimensionNumbers(attrs["scatter_dimension_numbers"])
            output = self.builder.scatter(
                inputs[0],
                inputs[1],
                inputs[2],
                list(dimensions.update_window_dims),
                list(dimensions.inserted_window_dims),
                list(dimensions.input_batching_dims),
                list(dimensions.scatter_indices_batching_dims),
                list(dimensions.scattered_dims_to_operand_dims),
                dimensions.index_vector_dim,
                _bool_attr(attrs["indices_are_sorted"]) if "indices_are_sorted" in attrs else False,
                _bool_attr(attrs["unique_indices"]) if "unique_indices" in attrs else False,
                self.region(op),
            )
        elif name == "stablehlo.sort":
            output = self.builder.sort(
                inputs,
                _int_attr(attrs["dimension"]),
                _bool_attr(attrs["is_stable"]) if "is_stable" in attrs else False,
                self.region(op),
            )
        elif name == "stablehlo.while":
            # The condition is the first region and the body the second; both take
            # the carried values, so Loom unrolls the loop where it can read the
            # count and refuses it where it cannot.
            output = self.builder.repeat_while(inputs, self.region(op, 0), self.region(op, 1))
        elif name == "func.call":
            callee_name = ir.FlatSymbolRefAttr(attrs["callee"]).value
            callee = self.functions.get(callee_name)
            if callee is None:
                raise NotImplementedError(f"Loom StableHLO calls unknown function @{callee_name}")
            block = callee.operation.regions[0].blocks[0]
            output = self.walk(block, dict(zip(block.arguments, inputs, strict=True)))
        else:
            operand_shapes = [_shape(operand.type) for operand in operands]
            result_shapes = [_shape(result.type) for result in results]
            raise NotImplementedError(
                f"Loom does not support StableHLO op {name!r}; "
                f"operand shapes={operand_shapes}, result shapes={result_shapes}"
            )

        if isinstance(output, list):
            for result, value in zip(results, output, strict=True):
                values[result] = value
        else:
            if len(results) != 1:
                raise ValueError(
                    f"StableHLO op {name!r} produced one value for {len(results)} results"
                )
            values[results[0]] = output


def lower_module(module, builder):
    with module.context:
        functions = {
            ir.StringAttr(op.operation.attributes["sym_name"]).value: op
            for op in module.body.operations
            if op.operation.name == "func.func"
        }
        main = functions.get("main")
        if main is None:
            raise NotImplementedError("StableHLO module has no func.func @main")
        walker = _Walker(builder, functions)
        entry = walker.build_func("main", main.operation.regions[0].blocks[0])
        return builder.finish(entry)
