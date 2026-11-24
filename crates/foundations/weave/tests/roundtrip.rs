// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use weave::{
    AttributeData, DialectName, Ir, OperationDefinition, OperationName, Registry, Type, TypeData,
    parse, print, verify,
};

fn registry() -> Registry {
    let mut registry = Registry::new();
    registry
        .register_dialect(DialectName::new("builtin").unwrap())
        .unwrap();
    registry
        .register_dialect(DialectName::new("test").unwrap())
        .unwrap();
    registry
        .register_operation(
            OperationName::new("builtin.module").unwrap(),
            OperationDefinition::new(),
        )
        .unwrap();
    registry
        .register_operation(
            OperationName::new("test.constant").unwrap(),
            OperationDefinition::new(),
        )
        .unwrap();
    registry
        .register_operation(
            OperationName::new("test.add").unwrap(),
            OperationDefinition::new().verifier(|ir, id, diagnostics| {
                let op = ir.operation(id).unwrap();
                if op.operands.len() != 2 || op.results.len() != 1 {
                    diagnostics.push(weave::Diagnostic::error(
                        "test.add requires two operands and one result",
                    ));
                }
            }),
        )
        .unwrap();
    registry
        .register_operation(
            OperationName::new("test.br").unwrap(),
            OperationDefinition::new().terminator(),
        )
        .unwrap();
    registry
        .register_operation(
            OperationName::new("test.return").unwrap(),
            OperationDefinition::new().terminator(),
        )
        .unwrap();
    registry
}

const GOLDEN: &str = r##"module {
  () = "builtin.module"() [] {future = "#future.payload<{x = [1, 2]}>"} : () -> () regions (
    {
      ^bb0():
        (%0) = "test.constant"() [] {value = "7 : i32"} : () -> ("i32")
        () = "test.br"() [^bb1(%0)] {} : () -> ()
      ^bb1(%1: "i32"):
        (%2) = "test.add"(%1, %1) [] {} : ("i32", "i32") -> ("i32")
        () = "test.return"(%2) [] {} : ("i32") -> ()
    }
  )
}
"##;

#[test]
fn generic_syntax_round_trips_canonically_and_preserves_unknown_payloads() {
    let registry = registry();
    let ir = parse(GOLDEN, &registry).unwrap();
    assert_eq!(print(&ir).unwrap(), GOLDEN);
    assert_eq!(
        print(&parse(&print(&ir).unwrap(), &registry).unwrap()).unwrap(),
        GOLDEN
    );
    assert_eq!(
        verify(&ir, &registry)
            .into_iter()
            .map(|d| d.message)
            .collect::<Vec<_>>(),
        vec!["unknown attribute `#future.payload<{x = [1, 2]}>`"]
    );
}

#[test]
fn malformed_ssa_is_a_parse_diagnostic() {
    let bad = r#"module { () = "test.return"(%missing) [] {} : ("i32") -> () }"#;
    let errors = match parse(bad, &registry()) {
        Ok(_) => panic!("invalid SSA unexpectedly parsed"),
        Err(errors) => errors,
    };
    assert!(
        errors
            .iter()
            .any(|error| error.message.contains("unknown SSA value"))
    );
}

#[test]
fn verifier_rejects_use_before_definition_and_missing_terminator() {
    let registry = registry();
    let mut ir = Ir::new();
    let module = ir
        .create_operation(OperationName::new("builtin.module").unwrap(), [], [])
        .unwrap();
    ir.add_root(module).unwrap();
    let region = ir.append_region(module).unwrap();
    let block = ir.append_block(region, []).unwrap();
    let late = ir
        .create_operation(
            OperationName::new("test.constant").unwrap(),
            [],
            [Type::opaque("i32").unwrap()],
        )
        .unwrap();
    let late_result = ir.operation(late).unwrap().results[0];
    let early = ir
        .create_operation(
            OperationName::new("test.add").unwrap(),
            [late_result, late_result],
            [Type::opaque("i32").unwrap()],
        )
        .unwrap();
    ir.append_operation(block, early).unwrap();
    ir.append_operation(block, late).unwrap();
    let messages: Vec<_> = verify(&ir, &registry)
        .into_iter()
        .map(|d| d.message)
        .collect();
    assert!(messages.iter().any(|m| m.contains("does not dominate")));
    assert!(
        messages
            .iter()
            .any(|m| m.contains("no registered terminator"))
    );
}

#[test]
fn successor_arity_and_types_are_checked() {
    let registry = registry();
    let source = GOLDEN.replace("^bb1(%0)", "^bb1()");
    let ir = parse(&source, &registry).unwrap();
    assert!(
        verify(&ir, &registry)
            .iter()
            .any(|d| d.message.contains("expects 1 arguments, got 0"))
    );
}

#[derive(Debug, Eq, PartialEq)]
struct TestType(u32);
impl TypeData for TestType {
    const NAME: &'static str = "!test.bits";
    fn parse(syntax: &str) -> Result<Self, String> {
        syntax
            .strip_prefix("!test.bits<")
            .and_then(|s| s.strip_suffix('>'))
            .ok_or_else(|| "expected !test.bits<N>".to_owned())?
            .parse()
            .map(Self)
            .map_err(|e: std::num::ParseIntError| e.to_string())
    }
    fn print(&self) -> String {
        format!("!test.bits<{}>", self.0)
    }
}

#[derive(Debug, Eq, PartialEq)]
struct TestAttribute(String);
impl AttributeData for TestAttribute {
    const NAME: &'static str = "#test.label";
    fn parse(syntax: &str) -> Result<Self, String> {
        Ok(Self(syntax.to_owned()))
    }
    fn print(&self) -> String {
        self.0.clone()
    }
}

#[test]
fn registered_extensions_are_typed_and_keep_original_syntax() {
    let mut registry = registry();
    registry.register_type::<TestType>().unwrap();
    registry.register_attribute::<TestAttribute>().unwrap();
    let ty = registry.parse_type("!test.bits<32>").unwrap();
    let attribute = registry.parse_attribute("#test.label<hello>").unwrap();
    assert_eq!(ty.downcast_ref::<TestType>(), Some(&TestType(32)));
    assert_eq!(
        attribute.downcast_ref::<TestAttribute>().unwrap().0,
        "#test.label<hello>"
    );
    assert_eq!(ty.syntax(), "!test.bits<32>");
}

#[test]
fn generated_linear_programs_round_trip() {
    let registry = registry();
    for length in 0..32 {
        let mut text = String::from(
            "module {\n  () = \"builtin.module\"() [] {} : () -> () regions (\n    {\n      ^bb0():\n",
        );
        for index in 0..length {
            text.push_str(&format!("        (%v{index}) = \"test.constant\"() [] {{value = \"{index}\"}} : () -> (\"i32\")\n"));
        }
        text.push_str("        () = \"test.return\"() [] {} : () -> ()\n    }\n  )\n}\n");
        let first = print(&parse(&text, &registry).unwrap()).unwrap();
        let second = print(&parse(&first, &registry).unwrap()).unwrap();
        assert_eq!(first, second);
    }
}

#[test]
fn duplicate_operation_registration_keeps_the_original_definition() {
    let mut registry = registry();
    let name = OperationName::new("test.unique").unwrap();
    registry
        .register_operation(
            name.clone(),
            OperationDefinition::new().terminator().interface(17_u32),
        )
        .unwrap();
    assert!(
        registry
            .register_operation(name.clone(), OperationDefinition::new())
            .is_err()
    );
    let original = registry.operation(&name).unwrap();
    assert!(original.traits.terminator);
    assert_eq!(original.get_interface::<u32>(), Some(&17));
}

#[test]
fn values_defined_in_a_region_do_not_escape_it() {
    let source = r#"module {
  () = "builtin.module"() [] {} : () -> () regions (
    {
      ^bb0():
        () = "builtin.module"() [] {} : () -> () regions (
          {
            ^bb0():
              (%leaked) = "test.constant"() [] {} : () -> ("i32")
              () = "test.return"() [] {} : () -> ()
          }
        )
        () = "test.return"(%leaked) [] {} : ("i32") -> ()
    }
  )
}
"#;
    let errors = match parse(source, &registry()) {
        Ok(_) => panic!("region-local SSA value unexpectedly escaped"),
        Err(errors) => errors,
    };
    assert!(
        errors
            .iter()
            .any(|error| error.message.contains("unknown SSA value `%leaked`"))
    );
}

#[test]
fn ownership_api_rejects_cycles() {
    let mut ir = Ir::new();
    let operation = ir
        .create_operation(OperationName::new("builtin.module").unwrap(), [], [])
        .unwrap();
    let region = ir.append_region(operation).unwrap();
    let block = ir.append_block(region, []).unwrap();
    assert_eq!(
        ir.append_operation(block, operation),
        Err(weave::IrError::OwnershipCycle)
    );
    assert!(ir.block(block).unwrap().operations.is_empty());
    assert_eq!(ir.operation(operation).unwrap().parent, None);
}

#[test]
fn parser_rejects_declared_operand_type_mismatch() {
    let source = r#"module {
  () = "builtin.module"() [] {} : () -> () regions (
    {
      ^bb0():
        (%0) = "test.constant"() [] {} : () -> ("i32")
        () = "test.return"(%0) [] {} : ("f32") -> ()
    }
  )
}
"#;
    let errors = match parse(source, &registry()) {
        Ok(_) => panic!("mismatched operand type unexpectedly parsed"),
        Err(errors) => errors,
    };
    assert!(errors.iter().any(|error| {
        error
            .message
            .contains("operand `%0` has type `i32`, but the operation declares `f32`")
    }));
}

#[test]
fn operation_results_are_not_visible_inside_their_own_regions() {
    let source = r#"module {
  () = "builtin.module"() [] {} : () -> () regions (
    {
      ^bb0():
        (%self) = "builtin.module"() [] {} : () -> ("i32") regions (
          {
            ^bb0():
              () = "test.return"(%self) [] {} : ("i32") -> ()
          }
        )
        () = "test.return"() [] {} : () -> ()
    }
  )
}
"#;
    let errors = match parse(source, &registry()) {
        Ok(_) => panic!("operation result unexpectedly visible in its own region"),
        Err(errors) => errors,
    };
    assert!(
        errors
            .iter()
            .any(|error| error.message.contains("unknown SSA value `%self`"))
    );
}

#[test]
fn verifier_accepts_lexically_captured_values() {
    let registry = registry();
    let mut ir = Ir::new();
    let module = ir
        .create_operation(OperationName::new("builtin.module").unwrap(), [], [])
        .unwrap();
    ir.add_root(module).unwrap();
    let outer_region = ir.append_region(module).unwrap();
    let outer_block = ir.append_block(outer_region, []).unwrap();
    let constant = ir
        .create_operation(
            OperationName::new("test.constant").unwrap(),
            [],
            [Type::opaque("i32").unwrap()],
        )
        .unwrap();
    let captured = ir.operation(constant).unwrap().results[0];
    ir.append_operation(outer_block, constant).unwrap();
    let nested = ir
        .create_operation(OperationName::new("builtin.module").unwrap(), [], [])
        .unwrap();
    ir.append_operation(outer_block, nested).unwrap();
    let nested_region = ir.append_region(nested).unwrap();
    let nested_block = ir.append_block(nested_region, []).unwrap();
    let nested_return = ir
        .create_operation(OperationName::new("test.return").unwrap(), [captured], [])
        .unwrap();
    ir.append_operation(nested_block, nested_return).unwrap();
    let outer_return = ir
        .create_operation(OperationName::new("test.return").unwrap(), [], [])
        .unwrap();
    ir.append_operation(outer_block, outer_return).unwrap();

    assert!(
        !verify(&ir, &registry)
            .iter()
            .any(|diagnostic| diagnostic.message.contains("does not dominate"))
    );
}

#[test]
fn verifier_rejects_values_that_escape_nested_regions() {
    let registry = registry();
    let mut ir = Ir::new();
    let module = ir
        .create_operation(OperationName::new("builtin.module").unwrap(), [], [])
        .unwrap();
    ir.add_root(module).unwrap();
    let outer_region = ir.append_region(module).unwrap();
    let outer_block = ir.append_block(outer_region, []).unwrap();
    let nested = ir
        .create_operation(OperationName::new("builtin.module").unwrap(), [], [])
        .unwrap();
    ir.append_operation(outer_block, nested).unwrap();
    let nested_region = ir.append_region(nested).unwrap();
    let nested_block = ir.append_block(nested_region, []).unwrap();
    let constant = ir
        .create_operation(
            OperationName::new("test.constant").unwrap(),
            [],
            [Type::opaque("i32").unwrap()],
        )
        .unwrap();
    let leaked = ir.operation(constant).unwrap().results[0];
    ir.append_operation(nested_block, constant).unwrap();
    let nested_return = ir
        .create_operation(OperationName::new("test.return").unwrap(), [], [])
        .unwrap();
    ir.append_operation(nested_block, nested_return).unwrap();
    let outer_return = ir
        .create_operation(OperationName::new("test.return").unwrap(), [leaked], [])
        .unwrap();
    ir.append_operation(outer_block, outer_return).unwrap();

    assert!(
        verify(&ir, &registry)
            .iter()
            .any(|diagnostic| diagnostic.message.contains("does not dominate"))
    );
}

#[test]
fn verifier_reports_detached_operations() {
    let registry = registry();
    let mut ir = Ir::new();
    let detached = ir
        .create_operation(OperationName::new("test.constant").unwrap(), [], [])
        .unwrap();
    let diagnostics = verify(&ir, &registry);
    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic.message == format!("operation {detached:?} is detached")
        })
    );
}

#[test]
fn verifier_rejects_values_crossing_sibling_regions() {
    let registry = registry();
    let mut ir = Ir::new();
    let module = ir
        .create_operation(OperationName::new("builtin.module").unwrap(), [], [])
        .unwrap();
    ir.add_root(module).unwrap();

    let defining_region = ir.append_region(module).unwrap();
    let defining_block = ir.append_block(defining_region, []).unwrap();
    let constant = ir
        .create_operation(
            OperationName::new("test.constant").unwrap(),
            [],
            [Type::opaque("i32").unwrap()],
        )
        .unwrap();
    let sibling_value = ir.operation(constant).unwrap().results[0];
    ir.append_operation(defining_block, constant).unwrap();
    let defining_return = ir
        .create_operation(OperationName::new("test.return").unwrap(), [], [])
        .unwrap();
    ir.append_operation(defining_block, defining_return)
        .unwrap();

    let using_region = ir.append_region(module).unwrap();
    let using_block = ir.append_block(using_region, []).unwrap();
    let using_return = ir
        .create_operation(
            OperationName::new("test.return").unwrap(),
            [sibling_value],
            [],
        )
        .unwrap();
    ir.append_operation(using_block, using_return).unwrap();

    assert!(
        verify(&ir, &registry)
            .iter()
            .any(|diagnostic| diagnostic.message.contains("does not dominate"))
    );
}

#[test]
fn verifier_reports_every_operation_in_a_detached_subtree() {
    let registry = registry();
    let mut ir = Ir::new();
    let detached_parent = ir
        .create_operation(OperationName::new("builtin.module").unwrap(), [], [])
        .unwrap();
    let region = ir.append_region(detached_parent).unwrap();
    let block = ir.append_block(region, []).unwrap();
    let detached_child = ir
        .create_operation(OperationName::new("test.return").unwrap(), [], [])
        .unwrap();
    ir.append_operation(block, detached_child).unwrap();

    let messages: Vec<_> = verify(&ir, &registry)
        .into_iter()
        .map(|diagnostic| diagnostic.message)
        .collect();
    assert!(messages.contains(&format!("operation {detached_parent:?} is detached")));
    assert!(messages.contains(&format!("operation {detached_child:?} is detached")));
}
