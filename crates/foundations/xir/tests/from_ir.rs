// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use xir::builder::ModuleBuilder;
use xir::dialect::{from_ir, to_ir};
use xir::dtype::DType;
use xir::generic_text::{parse, print};
use xir::ops::{Add, Call, Literal, Transpose};
use xir::shape::Shape;

#[test]
fn module_to_text_to_ir_to_module_preserves_structure() {
    let mut builder = ModuleBuilder::new();
    let callee = builder.func("twice", |function| {
        let value = function.param(Shape::new(DType::F32, [2, 3]));
        vec![function.op1(Add, &[value, value])]
    });
    let main = builder.func("main", |function| {
        let value = function.constant(Literal::new(Shape::new(DType::F32, [3, 2]), vec![0; 24]));
        let transposed = function.op1(
            Transpose {
                permutation: smallvec::smallvec![1, 0],
            },
            &[value],
        );
        function.op(Call { callee }, &[transposed]).into_vec()
    });
    let module = builder.finish(main).unwrap();
    let text = print(&module).unwrap();
    let rebuilt = from_ir(&parse(&text).unwrap()).unwrap();

    assert_eq!(rebuilt.entry_func().name, "main");
    assert_eq!(rebuilt.funcs.len(), 2);
    assert_eq!(
        rebuilt
            .entry_func()
            .nodes
            .iter()
            .map(|node| node.op.mnemonic())
            .collect::<Vec<_>>(),
        ["constant", "transpose", "call"]
    );
    assert_eq!(
        rebuilt.entry_func().nodes.last().unwrap().op.called_funcs(),
        [callee]
    );
    assert!(
        rebuilt.entry_func().nodes[0]
            .op
            .as_any()
            .downcast_ref::<xir::ops::Constant>()
            .is_some()
    );
    assert!(
        xir::analysis::constant_literal(
            rebuilt.entry_func(),
            rebuilt.entry_func().nodes[0].results[0],
        )
        .is_some()
    );
    assert_eq!(
        rebuilt.entry_func().nodes[1]
            .op
            .as_any()
            .downcast_ref::<Transpose>()
            .unwrap()
            .permutation
            .as_slice(),
        &[1, 0]
    );
    assert!(
        rebuilt.entry_func().nodes[2]
            .op
            .as_any()
            .downcast_ref::<Call>()
            .is_some()
    );
    assert_eq!(print(&rebuilt).unwrap(), text);
    assert!(to_ir(&rebuilt).is_ok());
}

#[test]
fn from_ir_rejects_semantically_inconsistent_add_types() {
    let source = r##"module {
  () = "xir.module"() [] {entry = "#xir.symbol_ref<6d61696e>", finite_map_graph = "#xir.finite_map_graph<;>"} : () -> () regions (
    {
      ^bb0():
        () = "xir.func"() [] {symbol = "#xir.symbol<6d61696e>"} : () -> () regions (
          {
            ^bb0(%0: "!xir.tensor<f32;;>", %1: "!xir.tensor<i32;;>"):
              (%2) = "xir.add"(%0, %1) [] {} : ("!xir.tensor<f32;;>", "!xir.tensor<i32;;>") -> ("!xir.tensor<f32;;>")
              () = "xir.return"(%2) [] {} : ("!xir.tensor<f32;;>") -> ()
          }
        )
        () = "xir.module_end"() [] {} : () -> ()
    }
  )
}
"##;
    let ir = parse(source).unwrap();
    assert!(from_ir(&ir).is_err());
}

#[test]
fn from_ir_rejects_non_function_module_children() {
    let source = r##"module {
  () = "xir.module"() [] {entry = "#xir.symbol_ref<6d61696e>", finite_map_graph = "#xir.finite_map_graph<;>"} : () -> () regions (
    {
      ^bb0():
        (%0) = "xir.constant"() [] {data = "#xir.bytes<>"} : () -> ("!xir.tensor<f32;;>")
        () = "xir.module_end"() [] {} : () -> ()
    }
  )
}
"##;
    let ir = parse(source).unwrap();
    assert!(from_ir(&ir).is_err());
}

#[test]
fn dynamic_slice_with_one_scalar_index_round_trips() {
    use xir::ops::DynamicSlice;

    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |function| {
        let value = function.param(Shape::new(DType::F32, [8]));
        let index = function.param(Shape::scalar(DType::I32));
        function
            .op(
                DynamicSlice {
                    slice_sizes: smallvec::smallvec![2],
                },
                &[value, index],
            )
            .into_vec()
    });
    let module = builder.finish(main).unwrap();
    let rebuilt = from_ir(&parse(&print(&module).unwrap()).unwrap()).unwrap();
    assert!(
        rebuilt.entry_func().nodes[0]
            .op
            .as_any()
            .downcast_ref::<DynamicSlice>()
            .is_some()
    );
}

#[test]
fn exact_callee_cardinality_and_required_sort_attributes_are_verified() {
    let extra_callee = r##"module {
  () = "xir.call"() [] {callees = "#xir.function_refs<61,62>"} : () -> ()
}
"##;
    let ir = parse(extra_callee).unwrap();
    assert!(
        weave::verify(&ir, &xir::dialect::registry())
            .iter()
            .any(|diagnostic| diagnostic.message.contains("requires 1 callee(s), got 2"))
    );

    let missing_sort_fields = r##"module {
  () = "xir.sort"() [] {dimension = "#xir.integer<0>"} : () -> ()
}
"##;
    let ir = parse(missing_sort_fields).unwrap();
    let messages = weave::verify(&ir, &xir::dialect::registry())
        .into_iter()
        .map(|diagnostic| diagnostic.message)
        .collect::<Vec<_>>();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("`is_stable`"))
    );
    assert!(messages.iter().any(|message| message.contains("`callees`")));
}

#[test]
fn xir_operations_reject_successors() {
    let source = r##"module {
  () = "xir.module"() [] {entry = "#xir.symbol_ref<6d61696e>", finite_map_graph = "#xir.finite_map_graph<;>"} : () -> () regions (
    {
      ^bb0():
        () = "xir.func"() [] {symbol = "#xir.symbol<6d61696e>"} : () -> () regions (
          {
            ^bb0():
              () = "xir.return"() [^bb0()] {} : () -> ()
          }
        )
        () = "xir.module_end"() [] {} : () -> ()
    }
  )
}
"##;
    let ir = parse(source).unwrap();
    assert!(
        weave::verify(&ir, &xir::dialect::registry())
            .iter()
            .any(|diagnostic| diagnostic
                .message
                .contains("does not support CFG successors"))
    );
}

#[test]
fn reduce_and_sort_validate_callback_signatures() {
    use xir::op::{FuncSig, InferCtx, Op};
    use xir::ops::{Reduce, Sort};

    let scalar = Shape::scalar(DType::F32);
    let tensor = Shape::new(DType::F32, [4]);
    let invalid = [FuncSig {
        params: vec![scalar.clone()],
        results: vec![scalar.clone()],
    }];
    let graph = domain_index::FiniteMapGraph::default();
    assert!(
        Reduce {
            dimensions: smallvec::smallvec![0],
            reducer: xir::ir::FuncId(0),
        }
        .infer(&InferCtx::new(&[tensor.clone(), scalar], &invalid, &graph))
        .is_err()
    );
    assert!(
        Sort {
            dimension: 0,
            is_stable: false,
            comparator: xir::ir::FuncId(0),
        }
        .infer(&InferCtx::new(&[tensor], &invalid, &graph))
        .is_err()
    );
}
