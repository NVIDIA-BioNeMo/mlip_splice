// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use weave::{Attribute, Type};
use xir::builder::ModuleBuilder;
use xir::dialect::{
    BytesAttribute, FunctionRefsAttribute, StringAttribute, SymbolNameAttribute,
    SymbolRefAttribute, TensorType, registry, to_ir,
};
use xir::dtype::DType;
use xir::generic_text::{parse, print};
use xir::ops::{Add, Call, Literal};
use xir::shape::Shape;

#[test]
fn registers_xir_operations_types_and_attributes() {
    let registry = registry();
    assert!(registry.contains_dialect("xir"));
    assert!(
        registry
            .operation(&weave::OperationName::new("xir.dot_general").unwrap())
            .is_some()
    );

    let ty = registry.parse_type("!xir.tensor<f32;2,5;4,8>").unwrap();
    assert_eq!(
        ty.downcast_ref::<TensorType>().unwrap().0,
        Shape::new(DType::F32, [4, 8])
            .with_domain_axes([domain_index::FiniteSetId(2), domain_index::FiniteSetId(5)])
    );
    assert_eq!(
        Type::typed(TensorType(Shape::scalar(DType::I32))).syntax(),
        "!xir.tensor<i32;;>"
    );

    let symbol = registry
        .parse_attribute("#xir.symbol_ref<6d61696e>")
        .unwrap();
    assert_eq!(
        symbol.downcast_ref::<SymbolRefAttribute>().unwrap().0,
        "main"
    );
    assert_eq!(
        Attribute::typed(BytesAttribute(vec![0, 127, 255])).syntax(),
        "#xir.bytes<007fff>"
    );
}

#[test]
fn legacy_module_converts_to_nested_ssa_and_round_trips_canonically() {
    let mut builder = ModuleBuilder::new();
    let callee = builder.func("twice", |function| {
        let value = function.param(Shape::new(DType::F32, [4]));
        vec![function.op1(Add, &[value, value])]
    });
    let main = builder.func("main", |function| {
        let value = function.constant(Literal::new(Shape::new(DType::F32, [4]), vec![0; 16]));
        function.op(Call { callee }, &[value]).into_vec()
    });
    let module = builder.finish(main).unwrap();

    let ir = to_ir(&module).unwrap();
    assert_eq!(ir.roots().len(), 1);
    let module_op = ir.operation(ir.roots()[0]).unwrap();
    assert_eq!(module_op.name.as_str(), "xir.module");
    assert_eq!(
        module_op.attributes["entry"]
            .downcast_ref::<SymbolRefAttribute>()
            .unwrap()
            .0,
        "main"
    );
    let module_region = ir.region(module_op.regions[0]).unwrap();
    let module_block = ir.block(module_region.blocks[0]).unwrap();
    assert_eq!(module_block.operations.len(), 3);
    let main_op = ir.operation(module_block.operations[1]).unwrap();
    assert_eq!(main_op.name.as_str(), "xir.func");
    assert_eq!(
        main_op.attributes["symbol"]
            .downcast_ref::<SymbolNameAttribute>()
            .unwrap()
            .0,
        "main"
    );
    let main_region = ir.region(main_op.regions[0]).unwrap();
    let main_block = ir.block(main_region.blocks[0]).unwrap();
    let call = ir.operation(main_block.operations[1]).unwrap();
    assert_eq!(call.name.as_str(), "xir.call");
    assert_eq!(
        call.attributes["callees"]
            .downcast_ref::<FunctionRefsAttribute>()
            .unwrap()
            .0,
        ["twice"]
    );

    let text = print(&module).unwrap();
    let parsed = parse(&text).unwrap();
    assert_eq!(weave::print(&parsed).unwrap(), text);
    let rebuilt = xir::dialect::from_ir(&parsed).unwrap();
    assert_eq!(print(&rebuilt).unwrap(), text);
    assert_eq!(rebuilt.funcs.len(), module.funcs.len());
    assert_eq!(rebuilt.entry_func().name, module.entry_func().name);
    assert_eq!(
        rebuilt
            .entry_func()
            .nodes
            .iter()
            .map(|node| node.op.mnemonic())
            .collect::<Vec<_>>(),
        module
            .entry_func()
            .nodes
            .iter()
            .map(|node| node.op.mnemonic())
            .collect::<Vec<_>>()
    );
    let diagnostics = weave::verify(&parsed, &registry());
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
}

#[test]
fn typed_attribute_identity_is_independent_of_input_spelling_after_parse() {
    let registry = registry();
    let first = registry.parse_attribute("#xir.string<78>").unwrap();
    let second = Attribute::typed(StringAttribute("x".into()));
    assert_eq!(
        first.downcast_ref::<StringAttribute>(),
        second.downcast_ref::<StringAttribute>()
    );
    assert_eq!(first.syntax(), second.syntax());
}

#[derive(Clone, Debug)]
struct Unregistered;

impl xir::op::Op for Unregistered {
    fn mnemonic(&self) -> &'static str {
        "unregistered"
    }

    fn infer(
        &self,
        context: &xir::op::InferCtx<'_>,
    ) -> Result<smallvec::SmallVec<[Shape; 1]>, xir::shape::ShapeError> {
        Ok(context.inputs().iter().cloned().collect())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn xir::op::Op> {
        Box::new(self.clone())
    }
}

#[test]
fn conversion_rejects_operations_outside_the_xir_dialect_registry() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |function| {
        let value = function.param(Shape::scalar(DType::F32));
        function.op(Unregistered, &[value]).into_vec()
    });
    let module = builder.finish(main).unwrap();
    let error = match to_ir(&module) {
        Ok(_) => panic!("unregistered operation unexpectedly converted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("unregistered"));
}

#[test]
fn dialect_verifiers_reject_malformed_function_structure() {
    let source = r#"module {
  () = "xir.func"() [] {} : () -> ()
}
"#;
    let ir = parse(source).unwrap();
    let messages = weave::verify(&ir, &registry())
        .into_iter()
        .map(|diagnostic| diagnostic.message)
        .collect::<Vec<_>>();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("requires 1 region"))
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("requires attribute `symbol`"))
    );
}

#[test]
fn typed_syntax_requires_canonical_numbers() {
    assert!(registry().parse_type("!xir.tensor<f32;;+01>").is_err());
}

#[test]
fn operation_attributes_are_typed_and_convolution_dimensions_are_preserved() {
    use smallvec::smallvec;
    use xir::dim_numbers::ConvDimensionNumbers;
    use xir::ops::{Convolution, Padding};

    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |function| {
        let input = function.param(Shape::new(DType::F32, [2, 3, 8]));
        let kernel = function.param(Shape::new(DType::F32, [4, 3, 3]));
        function
            .op(
                Convolution {
                    dnums: ConvDimensionNumbers {
                        input_batch_dimension: 0,
                        input_feature_dimension: 1,
                        input_spatial_dimensions: smallvec![2],
                        kernel_input_feature_dimension: 1,
                        kernel_output_feature_dimension: 0,
                        kernel_spatial_dimensions: smallvec![2],
                        output_batch_dimension: 0,
                        output_feature_dimension: 1,
                        output_spatial_dimensions: smallvec![2],
                    },
                    window_strides: smallvec![1],
                    padding: Padding::from_slice(&[[0, 0]]),
                    lhs_dilation: smallvec![1],
                    rhs_dilation: smallvec![1],
                    window_reversal: smallvec![],
                    feature_group_count: 1,
                    batch_group_count: 1,
                },
                &[input, kernel],
            )
            .into_vec()
    });
    let ir = to_ir(&builder.finish(main).unwrap()).unwrap();
    let module = ir.operation(ir.roots()[0]).unwrap();
    let module_block = ir
        .block(ir.region(module.regions[0]).unwrap().blocks[0])
        .unwrap();
    let function = ir.operation(module_block.operations[0]).unwrap();
    let function_block = ir
        .block(ir.region(function.regions[0]).unwrap().blocks[0])
        .unwrap();
    let convolution = ir.operation(function_block.operations[0]).unwrap();

    assert_eq!(
        convolution.attributes["input_batch_dimension"]
            .downcast_ref::<xir::dialect::IntegerAttribute>(),
        Some(&xir::dialect::IntegerAttribute(0))
    );
    assert_eq!(
        convolution.attributes["input_spatial_dimensions"]
            .downcast_ref::<xir::dialect::IntegersAttribute>(),
        Some(&xir::dialect::IntegersAttribute(vec![2]))
    );
    assert!(
        convolution
            .attributes
            .contains_key("kernel_spatial_dimensions")
    );
    assert!(
        convolution
            .attributes
            .contains_key("output_spatial_dimensions")
    );
}

#[test]
fn invalid_legacy_module_returns_an_error_instead_of_panicking() {
    let invalid = xir::ir::Module {
        finite_map_graph: domain_index::FiniteMapGraph::default(),
        funcs: Vec::new(),
        entry: xir::ir::FuncId(0),
    };
    assert!(matches!(
        to_ir(&invalid),
        Err(xir::dialect::ConversionError::InvalidModule(_))
    ));
}

#[test]
fn generic_parser_rejects_noncanonical_typed_attributes() {
    let source = r##"module {
  () = "xir.module"() [] {entry = "#xir.symbol_ref<6D61696E>", finite_map_graph = "#xir.finite_map_graph<;>"} : () -> () regions (
    {
      ^bb0():
        () = "xir.module_end"() [] {} : () -> ()
    }
  )
}
"##;
    assert!(parse(source).is_err());
}

#[test]
fn malformed_hex_and_zero_exact_denominator_are_diagnostics_not_panics() {
    let registry = registry();
    let unicode_hex = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        registry.parse_attribute("#xir.string<éé>")
    }));
    assert!(unicode_hex.is_ok());
    assert!(unicode_hex.unwrap().is_err());
    let zero_denominator = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        registry.parse_attribute("#xir.exact<1/0:1>")
    }));
    assert!(zero_denominator.is_ok());
    assert!(zero_denominator.unwrap().is_err());
}

#[test]
fn operation_schema_rejects_known_but_inapplicable_attributes() {
    let source = r##"module {
  () = "xir.module"() [] {entry = "#xir.symbol_ref<6d61696e>", finite_map_graph = "#xir.finite_map_graph<;>"} : () -> () regions (
    {
      ^bb0():
        () = "xir.func"() [] {symbol = "#xir.symbol<6d61696e>"} : () -> () regions (
          {
            ^bb0(%0: "!xir.tensor<f32;;>", %1: "!xir.tensor<f32;;>"):
              (%2) = "xir.add"(%0, %1) [] {dimension = "#xir.integer<0>"} : ("!xir.tensor<f32;;>", "!xir.tensor<f32;;>") -> ("!xir.tensor<f32;;>")
              () = "xir.return"(%2) [] {} : ("!xir.tensor<f32;;>") -> ()
          }
        )
        () = "xir.module_end"() [] {} : () -> ()
    }
  )
}
"##;
    let ir = parse(source).unwrap();
    let messages = weave::verify(&ir, &registry())
        .into_iter()
        .map(|diagnostic| diagnostic.message)
        .collect::<Vec<_>>();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("does not allow attribute `dimension`"))
    );
}
