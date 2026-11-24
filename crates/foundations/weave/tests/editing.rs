// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use weave::{Ir, IrError, OperationName, Type};

fn operation(
    ir: &mut Ir,
    name: &str,
    operands: impl IntoIterator<Item = weave::ValueId>,
) -> weave::OperationId {
    ir.create_operation(OperationName::new(name).unwrap(), operands, [])
        .unwrap()
}

fn operation_with_result(ir: &mut Ir, name: &str) -> weave::OperationId {
    ir.create_operation(
        OperationName::new(name).unwrap(),
        [],
        [Type::opaque("i32").unwrap()],
    )
    .unwrap()
}

fn body(ir: &mut Ir) -> (weave::OperationId, weave::BlockId) {
    let root = operation(ir, "test.root", []);
    ir.add_root(root).unwrap();
    let region = ir.append_region(root).unwrap();
    let block = ir.append_block(region, []).unwrap();
    (root, block)
}

#[test]
fn inserting_moving_and_detaching_preserve_identity() {
    let mut ir = Ir::new();
    let (_, first_block) = body(&mut ir);
    let second_region_owner = operation(&mut ir, "test.owner", []);
    ir.append_operation(first_block, second_region_owner)
        .unwrap();
    let second_region = ir.append_region(second_region_owner).unwrap();
    let second_block = ir.append_block(second_region, []).unwrap();
    let first = operation(&mut ir, "test.first", []);
    let second = operation(&mut ir, "test.second", []);

    ir.append_operation(first_block, first).unwrap();
    ir.insert_operation(first_block, 1, second).unwrap();
    assert_eq!(
        ir.block(first_block).unwrap().operations,
        [second_region_owner, second, first]
    );

    ir.move_operation(second, second_block, 0).unwrap();
    assert_eq!(ir.operation(second).unwrap().parent, Some(second_block));
    assert_eq!(ir.block(second_block).unwrap().operations, [second]);

    ir.detach_operation(second).unwrap();
    assert_eq!(ir.operation(second).unwrap().parent, None);
    assert!(ir.block(second_block).unwrap().operations.is_empty());
}

#[test]
fn refused_edits_change_nothing() {
    let mut ir = Ir::new();
    let (root, block) = body(&mut ir);
    let child = operation(&mut ir, "test.child", []);
    ir.append_operation(block, child).unwrap();
    let before = ir.block(block).unwrap().operations.clone();

    let late = operation(&mut ir, "test.late", []);
    assert_eq!(
        ir.insert_operation(block, usize::MAX, late),
        Err(IrError::InvalidPosition)
    );
    assert_eq!(ir.block(block).unwrap().operations, before);
    assert_eq!(
        ir.move_operation(root, block, 0),
        Err(IrError::OperationNotAttached)
    );
    assert_eq!(ir.block(block).unwrap().operations, before);
}

#[test]
fn replace_all_uses_with_updates_operands_and_successor_arguments() {
    let mut ir = Ir::new();
    let (_, block) = body(&mut ir);
    let old_def = operation_with_result(&mut ir, "test.old");
    let new_def = operation_with_result(&mut ir, "test.new");
    let old = ir.operation(old_def).unwrap().results[0];
    let new = ir.operation(new_def).unwrap().results[0];
    ir.append_operation(block, old_def).unwrap();
    ir.append_operation(block, new_def).unwrap();
    let user = operation(&mut ir, "test.user", [old, old]);
    ir.append_operation(block, user).unwrap();
    ir.add_successor(user, block, [old]).unwrap();

    assert_eq!(ir.replace_all_uses_with(old, new), Ok(3));
    let user = ir.operation(user).unwrap();
    assert_eq!(user.operands, [new, new]);
    assert_eq!(user.successors[0].arguments, [new]);
}

#[test]
fn erase_refuses_live_results_and_never_reuses_an_id() {
    let mut ir = Ir::new();
    let (_, block) = body(&mut ir);
    let definition = operation_with_result(&mut ir, "test.definition");
    let value = ir.operation(definition).unwrap().results[0];
    ir.append_operation(block, definition).unwrap();
    let user = operation(&mut ir, "test.user", [value]);
    ir.append_operation(block, user).unwrap();

    assert_eq!(
        ir.erase_operation(definition),
        Err(IrError::OperationHasUses)
    );
    assert!(ir.operation(definition).is_some());
    ir.erase_operation(user).unwrap();
    ir.erase_operation(definition).unwrap();
    let replacement = operation(&mut ir, "test.replacement", []);
    assert_ne!(definition.slot(), replacement.slot());
    assert!(ir.operation(definition).is_none());
}

#[test]
fn freeze_compacts_holes_and_builds_typed_indexes() {
    let mut ir = Ir::new();
    let (root, block) = body(&mut ir);
    let deleted = operation(&mut ir, "test.deleted", []);
    let definition = operation_with_result(&mut ir, "test.definition");
    let value = ir.operation(definition).unwrap().results[0];
    ir.append_operation(block, deleted).unwrap();
    ir.append_operation(block, definition).unwrap();
    let user = operation(&mut ir, "test.user", [value]);
    ir.append_operation(block, user).unwrap();
    ir.erase_operation(deleted).unwrap();

    let (frozen, map) = ir.freeze_with_map().unwrap();
    let frozen_root = map.operation(root).unwrap();
    let frozen_definition = map.operation(definition).unwrap();
    let frozen_value = map.value(value).unwrap();
    let frozen_user = map.operation(user).unwrap();

    assert_eq!(frozen.operations().count(), 3);
    assert_eq!(frozen.roots(), [frozen_root]);
    assert_eq!(frozen.operation(frozen_user).operands, [frozen_value]);
    assert_eq!(
        frozen.uses().of_value(frozen_value.index()),
        &[frozen_user.index() as u32]
    );
    assert!(map.operation(deleted).is_none());
    assert_eq!(frozen.operation(frozen_definition).results, [frozen_value]);
}

#[test]
fn creating_with_a_foreign_operand_is_atomic() {
    let mut source = Ir::new();
    let definition = operation_with_result(&mut source, "test.definition");
    let foreign = source.operation(definition).unwrap().results[0];
    let mut target = Ir::new();

    assert_eq!(
        target.create_operation(OperationName::new("test.user").unwrap(), [foreign], []),
        Err(IrError::MissingValue)
    );
    assert_eq!(target.operations().count(), 0);
}

#[test]
fn replacing_uses_requires_equal_types() {
    let mut ir = Ir::new();
    let (_, block) = body(&mut ir);
    let i32_definition = operation_with_result(&mut ir, "test.i32");
    let f32_definition = ir
        .create_operation(
            OperationName::new("test.f32").unwrap(),
            [],
            [Type::opaque("f32").unwrap()],
        )
        .unwrap();
    let i32_value = ir.operation(i32_definition).unwrap().results[0];
    let f32_value = ir.operation(f32_definition).unwrap().results[0];
    ir.append_operation(block, i32_definition).unwrap();
    ir.append_operation(block, f32_definition).unwrap();
    let user = operation(&mut ir, "test.user", [i32_value]);
    ir.append_operation(block, user).unwrap();

    assert_eq!(
        ir.replace_all_uses_with(i32_value, f32_value),
        Err(IrError::TypeMismatch)
    );
    assert_eq!(ir.operation(user).unwrap().operands, [i32_value]);
}

#[test]
#[should_panic(expected = "operation belongs to another frozen IR")]
fn a_frozen_id_cannot_address_another_ir() {
    let mut first = Ir::new();
    let (root, _) = body(&mut first);
    let (first, map) = first.freeze_with_map().unwrap();
    let first_root = map.operation(root).unwrap();
    let mut second = Ir::new();
    body(&mut second);
    let (second, _) = second.freeze_with_map().unwrap();

    let _ = first.operation(first_root);
    let _ = second.operation(first_root);
}

#[test]
fn freeze_refuses_detached_operations() {
    let mut ir = Ir::new();
    body(&mut ir);
    operation(&mut ir, "test.detached", []);

    assert!(matches!(ir.freeze(), Err(IrError::DetachedOperation)));
}
