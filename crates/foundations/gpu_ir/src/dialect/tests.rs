// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{Builder, DeviceFunction, Launch, Param, ParamTy, Scalar, Ty};

fn branching_function() -> Function {
    let mut builder = Builder::new();
    let left = builder.parameter(Param {
        name: "left".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let right = builder.parameter(Param {
        name: "right".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let sum = builder.add(left, right);
    let held = builder.bind_value_as("sum", Ty::scalar(Scalar::F32), sum);
    let zero = builder.float(Scalar::F32, 0.0);
    let positive = builder.lt(zero, held.clone());
    let selected = builder.if_value(
        positive,
        Ty::scalar(Scalar::F32),
        |builder| builder.float(Scalar::F32, 1.0),
        |builder| builder.float(Scalar::F32, -1.0),
    );
    builder.returns(vec![selected]);
    builder.into_function()
}

#[test]
fn conversion_preserves_every_source_entity_and_ssa_edge() {
    let source = branching_function();
    let conversion = to_weave(&source).unwrap();
    assert_eq!(conversion.values.len(), source.num_values());
    assert_eq!(conversion.operations.len(), source.num_ops());
    assert_eq!(conversion.blocks.len(), source.num_blocks());
    assert_eq!(conversion.regions.len(), source.num_regions());

    for index in 0..source.num_values() {
        let source_value = source.value(source.value_at(index));
        let converted = conversion.ir.value(conversion.values[index]).unwrap();
        assert_eq!(converted.ty, gpu_type(source_value.ty));
    }
    for index in 0..source.num_ops() {
        let source_op = source.op(source.op_at(index));
        let converted = conversion
            .ir
            .operation(conversion.operations[index])
            .unwrap();
        assert_eq!(converted.name.as_str(), operation_name(&source_op.kind));
        assert_eq!(converted.results.len(), source_op.results.len());
        assert_eq!(converted.regions.len(), source_op.regions.len());
        for (operand, mapped) in source_op.operands.iter().zip(&converted.operands) {
            assert_eq!(*mapped, conversion.values[operand.index()]);
        }
    }
    for index in 0..source.num_regions() {
        let source_region = source.region(source.region_at(index));
        let converted = conversion.ir.region(conversion.regions[index]).unwrap();
        assert_eq!(converted.blocks.len(), source_region.blocks.len());
        for (&block, mapped) in source_region.blocks.iter().zip(&converted.blocks) {
            assert_eq!(*mapped, conversion.blocks[block.index()]);
        }
    }
    assert!(validate(&conversion.ir).is_empty());
}

#[test]
fn generic_gpu_text_round_trips_canonically() {
    let text = print(&branching_function()).unwrap();
    assert!(text.contains("\"gpu.func\""));
    assert!(text.contains("\"gpu.if\""));
    assert!(text.contains("\"!gpu.data<f32x1>\""));
    let parsed = parse(&text).unwrap();
    assert!(validate(&parsed).is_empty());
    assert_eq!(weave::print(&parsed).unwrap(), text);
}

#[test]
fn conversion_does_not_change_legacy_wire_or_ptx() {
    let mut builder = Builder::new();
    let destination = builder.parameter(Param {
        name: "destination".into(),
        ty: ParamTy::Ptr(Scalar::F32),
    });
    let zero = builder.i32(0);
    let at = builder.offset(destination, zero);
    let one = builder.float(Scalar::F32, 1.0);
    builder.store(at, one);
    let function = builder.into_function();
    let wire = crate::arena::wire::encode_function(&function);
    let launch = Launch::new(DeviceFunction::new("store_one", function), 32, None);
    let ptx = crate::ptx::render_module(std::slice::from_ref(&launch), 80).unwrap();

    let text = print(launch.body()).unwrap();
    let parsed = parse(&text).unwrap();
    let reconstructed = from_weave(&parsed, &[]).unwrap();
    assert_eq!(crate::arena::wire::encode_function(&reconstructed), wire);
    let reconstructed_launch =
        Launch::new(DeviceFunction::new("store_one", reconstructed), 32, None);
    assert_eq!(
        crate::ptx::render_module(std::slice::from_ref(&reconstructed_launch), 80).unwrap(),
        ptx
    );
    assert_eq!(weave::print(&parsed).unwrap(), text);
}

#[test]
fn validation_rejects_incomplete_gpu_operations() {
    let text = r##"module {
  (%0) = "gpu.int_const"() [] {} : () -> ("!gpu.data<s32x1>")
}
"##;
    let parsed = parse(text).unwrap();
    let messages = validate(&parsed)
        .into_iter()
        .map(|diagnostic| diagnostic.message)
        .collect::<Vec<_>>();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("materialized"))
    );
}

#[test]
fn validation_rejects_wrong_constant_result_arity() {
    let mut ir = Ir::new();
    let operation = ir
        .create_operation(op_name("gpu.int_const"), [], [])
        .unwrap();
    ir.set_attribute(
        operation,
        "materialized",
        Attribute::typed(GpuAttribute::Bool(false)),
    )
    .unwrap();
    ir.set_attribute(
        operation,
        "operand_count",
        Attribute::typed(GpuAttribute::UInt(0)),
    )
    .unwrap();
    ir.add_root(operation).unwrap();
    assert!(
        validate(&ir)
            .iter()
            .any(|diagnostic| diagnostic.message.contains("requires 1 results"))
    );
}

#[test]
fn registered_attributes_are_canonicalized() {
    let parsed = registry().parse_attribute("#gpu.attr<uint(0007)>").unwrap();
    assert_eq!(parsed.syntax(), "#gpu.attr<uint(7)>");
}

#[test]
fn every_gpu_operation_is_registered() {
    let registry = registry();
    for &(name, _) in OPERATIONS {
        assert!(registry.operation(&op_name(name)).is_some(), "{name}");
    }
}

#[test]
fn repeated_constant_loads_reconstruct_one_shared_array() {
    let mut source = Function::new();
    let index_op = source.add_op(
        OpKind::IntConst {
            scalar: Scalar::S32,
            value: 0,
        },
        false,
        Vec::new(),
        &[ValTy::Data(Ty::scalar(Scalar::S32))],
        Vec::new(),
        Vec::new(),
    );
    let index = source.op(index_op).results[0];
    let array = Arc::new(crate::op::ConstArray {
        name: "table".into(),
        scalar: Scalar::U8,
        data: vec![3, 5, 8].into(),
    });
    let first = source.add_op(
        OpKind::ConstLoad(array.clone()),
        false,
        vec![index],
        &[ValTy::Data(Ty::scalar(Scalar::U8))],
        Vec::new(),
        Vec::new(),
    );
    let second = source.add_op(
        OpKind::ConstLoad(array),
        false,
        vec![index],
        &[ValTy::Data(Ty::scalar(Scalar::U8))],
        Vec::new(),
        Vec::new(),
    );
    source.set_block_ops(source.entry(), vec![index_op, first, second]);
    source.set_terminator(
        source.entry(),
        vec![source.op(first).results[0], source.op(second).results[0]],
    );
    let conversion = to_weave(&source).unwrap();
    let text = weave::print(&conversion.ir).unwrap();
    let parsed = parse(&text).unwrap();
    let reconstructed = from_weave(&parsed, &conversion.callees).unwrap();
    let arrays = (0..reconstructed.num_ops())
        .filter_map(
            |index| match &reconstructed.op(reconstructed.op_at(index)).kind {
                OpKind::ConstLoad(array) => Some(array.clone()),
                _ => None,
            },
        )
        .collect::<Vec<_>>();
    assert_eq!(arrays.len(), 2);
    assert!(Arc::ptr_eq(&arrays[0], &arrays[1]));
    assert_eq!(
        crate::arena::wire::encode_function(&reconstructed),
        crate::arena::wire::encode_function(&source)
    );
}

#[test]
fn validation_rejects_non_function_roots_and_nested_functions() {
    let malformed_root = r##"module {
  (%0) = "gpu.int_const"() [] {materialized = "#gpu.attr<bool(false)>", operand_count = "#gpu.attr<uint(0)>"} : () -> ("!gpu.data<s32x1>")
}
"##;
    assert!(
        validate(&parse(malformed_root).unwrap())
            .iter()
            .any(|diagnostic| diagnostic.message.contains("only gpu.func"))
    );

    let nested = r##"module {
  () = "gpu.func"() [] {entry = "#gpu.attr<uint(0)>"} : () -> () regions (
    {
      ^bb0():
        () = "gpu.func"() [] {entry = "#gpu.attr<uint(0)>"} : () -> () regions (
          {
            ^bb0():
              () = "gpu.return"() [] {parameter_indices = "#gpu.attr<indices()>", source_block = "#gpu.attr<uint(0)>"} : () -> ()
          }
        )
        () = "gpu.return"() [] {parameter_indices = "#gpu.attr<indices()>", source_block = "#gpu.attr<uint(0)>"} : () -> ()
    }
  )
}
"##;
    assert!(
        validate(&parse(nested).unwrap())
            .iter()
            .any(|diagnostic| diagnostic.message.contains("sole root"))
    );
}

#[test]
fn maximal_source_indices_are_rejected_without_panicking() {
    let source = branching_function();
    let mut conversion = to_weave(&source).unwrap();
    let operation = conversion.operations[0];
    conversion
        .ir
        .set_attribute(
            operation,
            "result_indices",
            Attribute::typed(GpuAttribute::Indices(vec![Some(usize::MAX)])),
        )
        .unwrap();
    let messages = validate(&conversion.ir)
        .into_iter()
        .map(|diagnostic| diagnostic.message)
        .collect::<Vec<_>>();
    assert!(
        messages
            .iter()
            .any(|diagnostic| diagnostic.contains("result_indices"))
    );
    assert!(from_weave(&conversion.ir, &conversion.callees).is_err());
}

#[test]
fn validation_and_reverse_conversion_reject_unrepresentable_fields() {
    let source = branching_function();
    let mut conversion = to_weave(&source).unwrap();
    let foreign = conversion
        .ir
        .create_operation(op_name("gpu.ret"), [], [])
        .unwrap();
    conversion
        .ir
        .add_successor(conversion.function, conversion.blocks[0], [])
        .unwrap();
    conversion
        .ir
        .set_attribute(
            conversion.function,
            "foreign",
            Attribute::typed(GpuAttribute::Bool(true)),
        )
        .unwrap();
    let messages = validate(&conversion.ir)
        .into_iter()
        .map(|diagnostic| diagnostic.message)
        .collect::<Vec<_>>();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("successors"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("not representable"))
    );
    assert!(from_weave(&conversion.ir, &conversion.callees).is_err());
    assert!(conversion.ir.operation(foreign).is_some());
}

#[test]
fn validation_rejects_empty_ir_and_function_without_entry_block() {
    assert!(
        validate(&Ir::new())
            .iter()
            .any(|diagnostic| diagnostic.message.contains("exactly one root gpu.func"))
    );
    let mut ir = Ir::new();
    let function = ir.create_operation(op_name("gpu.func"), [], []).unwrap();
    ir.add_root(function).unwrap();
    ir.set_attribute(function, "entry", Attribute::typed(GpuAttribute::UInt(0)))
        .unwrap();
    ir.append_region(function).unwrap();
    assert!(
        validate(&ir)
            .iter()
            .any(|diagnostic| diagnostic.message.contains("entry block"))
    );
    assert!(from_weave(&ir, &[]).is_err());
}

#[test]
fn call_metadata_must_match_the_resolved_callee() {
    let callee = Arc::new(DeviceFunction::new(
        "callee",
        Builder::new().into_function(),
    ));
    let mut builder = Builder::new();
    builder.call(callee, Vec::new());
    let source = builder.into_function();
    let mut conversion = to_weave(&source).unwrap();
    let call = conversion
        .operations
        .iter()
        .copied()
        .find(|operation| conversion.ir.operation(*operation).unwrap().name.as_str() == "gpu.call")
        .unwrap();
    conversion
        .ir
        .set_attribute(
            call,
            "callee",
            Attribute::typed(GpuAttribute::Text("wrong".into())),
        )
        .unwrap();
    assert!(from_weave(&conversion.ir, &conversion.callees).is_err());
    conversion
        .ir
        .set_attribute(
            call,
            "callee",
            Attribute::typed(GpuAttribute::Text("callee".into())),
        )
        .unwrap();
    conversion
        .ir
        .set_attribute(
            call,
            "parameter_count",
            Attribute::typed(GpuAttribute::UInt(1)),
        )
        .unwrap();
    assert!(
        validate(&conversion.ir)
            .iter()
            .any(|diagnostic| diagnostic.message.contains("arities"))
    );
    assert!(from_weave(&conversion.ir, &conversion.callees).is_err());
}

#[test]
fn validation_rejects_trailing_operands_and_wrong_metadata_families() {
    let source = branching_function();
    let mut conversion = to_weave(&source).unwrap();
    let operation = conversion.operations[0];
    let extra = conversion.values[0];
    conversion.ir.set_operands(operation, [extra]).unwrap();
    conversion
        .ir
        .set_attribute(
            operation,
            "source_operation",
            Attribute::typed(GpuAttribute::Bool(false)),
        )
        .unwrap();
    let messages = validate(&conversion.ir)
        .into_iter()
        .map(|diagnostic| diagnostic.message)
        .collect::<Vec<_>>();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("exactly its source operands"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("source_operation"))
    );
    assert!(from_weave(&conversion.ir, &conversion.callees).is_err());
}

#[test]
fn source_operation_indices_are_a_dense_gpu_operation_permutation() {
    let source = branching_function();
    let mut conversion = to_weave(&source).unwrap();
    let operation = conversion.operations[0];
    conversion
        .ir
        .set_attribute(
            operation,
            "source_operation",
            Attribute::typed(GpuAttribute::UInt(conversion.operations.len())),
        )
        .unwrap();
    assert!(
        validate(&conversion.ir)
            .iter()
            .any(|diagnostic| diagnostic.message.contains("out of range or duplicated"))
    );
    assert!(from_weave(&conversion.ir, &conversion.callees).is_err());
}

/// Every kind of operation, in both placements, with the regions each states:
/// the MLIR form is exact when what it reads back encodes to the same bytes.
fn every_kind_of_operation() -> (Function, Arc<crate::DeviceFunction>) {
    let callee = Arc::new(crate::DeviceFunction::declared(
        "callee",
        crate::func::Signature {
            parameters: vec![
                Param {
                    name: "a".to_string(),
                    ty: ParamTy::Scalar(Scalar::F32),
                },
                Param {
                    name: "b".to_string(),
                    ty: ParamTy::Scalar(Scalar::F32),
                },
            ],
            results: vec![crate::ValTy::Data(Ty::scalar(Scalar::F32))],
            accesses: vec![crate::func::Access::default(); 2],
            sync: None,
        },
    ));
    let array = Arc::new(crate::op::ConstArray {
        name: "table".to_string(),
        scalar: Scalar::F32,
        data: vec![0u8; 4].into(),
    });
    let kinds = vec![
        crate::OpKind::IntConst {
            scalar: Scalar::S32,
            value: -7,
        },
        crate::OpKind::FloatConst {
            scalar: Scalar::F64,
            value: -0.0,
        },
        crate::OpKind::Sreg(crate::Sreg::CtaidY),
        crate::OpKind::Load {
            lanes: 4,
            nc: false,
            evict: crate::Evict::Last,
        },
        crate::OpKind::Offset,
        crate::OpKind::Un(crate::UnOp::Sqrt),
        crate::OpKind::Bin(crate::BinOp::Shr),
        crate::OpKind::Fma,
        crate::OpKind::Cmp(crate::CmpOp::Ge),
        crate::OpKind::Select,
        crate::OpKind::Cast(Scalar::BF16),
        crate::OpKind::Splat(3),
        crate::OpKind::Extract(2),
        crate::OpKind::Pack,
        crate::OpKind::ShflDown,
        crate::OpKind::ActiveMask,
        crate::OpKind::ShflFrom,
        crate::OpKind::WarpAny,
        crate::OpKind::ConstLoad(array),
        crate::OpKind::Bind,
        crate::OpKind::Write {
            mode: crate::WriteMode::Add,
            evict: crate::Evict::First,
        },
        crate::OpKind::FetchAdd,
        crate::OpKind::If,
        crate::OpKind::Loop {
            index: Scalar::U64,
            trip: crate::Trip::BlockUniform,
        },
        crate::OpKind::Scope,
        crate::OpKind::Call(callee.clone()),
        crate::OpKind::Break,
        crate::OpKind::Ret,
        crate::OpKind::Bar,
    ];
    assert_eq!(kinds.len(), 29, "one of every kind");

    let mut f = Function::new();
    let held = f
        .declare_param(Param {
            name: "p".to_string(),
            ty: ParamTy::Scalar(Scalar::F32),
        })
        .expect("one declaration");
    let data = crate::ValTy::Data(Ty::scalar(Scalar::F32));
    for kind in &kinds {
        let placements: &[bool] = match kind.may_be_either() {
            true => &[false, true],
            false => match kind.must_be_placed() {
                true => &[true],
                false => &[false],
            },
        };
        for &placed in placements {
            let regions = (0..kind.regions())
                .map(|_| {
                    let block = f.new_block();
                    f.new_region(vec![block]).expect("a fresh block")
                })
                .collect();
            let op = f
                .new_op(
                    kind.clone(),
                    placed,
                    vec![held, held],
                    &[data],
                    vec![Some(held)],
                    regions,
                )
                .expect("the shape each kind states");
            let entry = f.entry();
            let position = f.block(entry).ops.len();
            f.insert_step(entry, position, op)
                .expect("every operation has a block");
        }
    }
    (f, callee)
}

#[test]
fn every_kind_of_operation_survives_the_generic_form() {
    let (function, callee) = every_kind_of_operation();
    let text = print(&function).unwrap();
    let parsed = parse(&text).unwrap();
    let back = from_weave(&parsed, std::slice::from_ref(&callee)).unwrap();
    assert_eq!(
        crate::arena::wire::encode_function(&function),
        crate::arena::wire::encode_function(&back),
        "{text}"
    );
    assert_eq!(weave::print(&parsed).unwrap(), text);
}
