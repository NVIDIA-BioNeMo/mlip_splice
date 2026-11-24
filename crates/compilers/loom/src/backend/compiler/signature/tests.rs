// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use domain_index::{Domain, IndexMap, SlotKind};
use xir::builder::ModuleBuilder;
use xir::dtype::DType as Dtype;
use xir::dtype::DType;
use xir::shape::Shape;

use super::*;
use crate::FlatOperand;

/// Two operands whose every factor differs, so a signature that transposed a
/// shape for a lane, an operand for another, or a dtype for its neighbour's
/// would land on a different number.
///
/// The coarse domain holds 3 cells and the fine one 40. Input 0 rides the fine
/// domain with 5 visible elements at lane 2 in `f32`; the output rides the
/// coarse one with 7 visible elements at lane 4 in `f16`. No two factors are
/// equal and no product of a wrong subset collides with the right one.
fn asymmetric_program() -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("asymmetric");
    let value = builder.param(Shape::scalar(DType::F32));
    let entry = builder.end_func(vec![value]);
    loom_program! {
        name: "asymmetric".into(),
        module: builder.finish(entry).unwrap(),
        lane_count: 1,
        domains: vec![
            Domain { extent: 3 },
            Domain { extent: 40 },
        ],
        index_maps: vec![IndexMap {
            source_id: 1,
            target_id: 0,
            is_injective: false,
            is_sorted: false,
            may_have_unmapped: false,
        }],
        num_axes: 1,
        operands: vec![
            FlatOperand {
                dtype: Dtype::F32,
                element_shape: vec![5],
                lane_extent: 2,
                axis_domains: vec![1],
            },
            FlatOperand {
                dtype: Dtype::F16,
                element_shape: vec![7],
                lane_extent: 4,
                axis_domains: vec![0],
            },
        ],
        num_inputs: 1,
        available_views: Vec::new(),
    }
}

#[test]
fn an_operand_is_its_cells_its_shape_its_lane_and_its_width() {
    let loom = asymmetric_program();
    let program = Plan::prepare(&loom).unwrap();
    // 40 cells * 5 elements * lane 2 * 4 bytes.
    assert_eq!(operand_nbytes(&program, &loom, 0).unwrap(), 1600);
    // 3 cells * 7 elements * lane 4 * 2 bytes.
    assert_eq!(operand_nbytes(&program, &loom, 1).unwrap(), 168);
}

#[test]
fn an_operand_past_the_program_has_no_extent() {
    let loom = asymmetric_program();
    let program = Plan::prepare(&loom).unwrap();
    assert!(operand_nbytes(&program, &loom, 2).is_err());
}

/// A target-indices slot on the 40-element map: one `int32` per source element.
fn target_indices_slot() -> Slot {
    Slot {
        kind: SlotKind::TargetIndices,
        source_id: 1,
        target_id: 0,
        other_target_id: -1,
    }
}

#[test]
fn the_reference_arrays_follow_the_data_inputs() {
    let loom = asymmetric_program();
    let program = Plan::prepare(&loom).unwrap();
    let slots = [target_indices_slot()];
    let signature = program_signature(&program, &loom, &slots).unwrap();
    // The data operand first, then the target indices: 40 indices at 4 bytes.
    assert_eq!(signature.inputs(), [1600, 160]);
    assert_eq!(signature.outputs(), [168]);
}

#[test]
fn a_program_that_binds_no_slots_states_only_its_operands() {
    let loom = asymmetric_program();
    let program = Plan::prepare(&loom).unwrap();
    let signature = program_signature(&program, &loom, &[]).unwrap();
    assert_eq!(signature.inputs(), [1600]);
}

#[test]
fn a_launch_reads_the_workspace_after_the_program() {
    let program = RecipeSignature::new(vec![64, 96], vec![48]);
    let launch = launch_signature(&program, &[512, 1024], &[80]);
    assert_eq!(launch.inputs(), [64, 96, 512, 1024]);
    // A launch accumulates into the shared buffer, which the bucket fold makes
    // wider than the output the program states.
    assert_eq!(launch.outputs(), [80]);
}

/// One arm that binds nothing but states the ports it is spliced onto.
fn arm(inputs: Vec<usize>, outputs: Vec<usize>) -> Recipe {
    ::recipe::RecipeBuilder::new(inputs.len(), outputs.len())
        .finish(RecipeSignature::new(inputs, outputs))
        .unwrap()
        .into_recipe()
}

/// The same map composed rather than declared, so an arm can bind a slot the
/// other one does not.
fn composed_slot() -> Slot {
    Slot {
        kind: SlotKind::ComposedTargetIndices,
        source_id: 1,
        target_id: 0,
        other_target_id: -1,
    }
}

#[test]
fn a_shared_port_takes_the_extent_of_the_arm_that_binds_it() {
    let (declared, composed) = (target_indices_slot(), composed_slot());
    let all = [declared.clone(), composed.clone()];
    // Two data inputs, then the declared map, then the composed one. The first
    // arm binds only the declared map, the second only the composed one, so
    // each fills a different shared port and every extent is distinct.
    let first = arm(vec![64, 96, 160], vec![48]);
    let second = arm(vec![64, 96, 320], vec![48]);
    let signature = shared_signature(
        &[
            SplicedArm {
                recipe: &first,
                slots: std::slice::from_ref(&declared),
                outputs: &[0],
            },
            SplicedArm {
                recipe: &second,
                slots: std::slice::from_ref(&composed),
                outputs: &[0],
            },
        ],
        2,
        &all,
        1,
    )
    .unwrap();
    assert_eq!(signature.inputs(), [64, 96, 160, 320]);
    assert_eq!(signature.outputs(), [48]);
}

#[test]
fn arms_that_own_different_outputs_state_them_all() {
    let slots = [target_indices_slot()];
    let first = arm(vec![64, 160], vec![48]);
    let second = arm(vec![64, 160], vec![24]);
    let signature = shared_signature(
        &[
            SplicedArm {
                recipe: &first,
                slots: &slots,
                outputs: &[1],
            },
            SplicedArm {
                recipe: &second,
                slots: &slots,
                outputs: &[0],
            },
        ],
        1,
        &slots,
        2,
    )
    .unwrap();
    assert_eq!(signature.outputs(), [24, 48]);
}

#[test]
fn arms_that_disagree_about_a_shared_port_are_refused() {
    let slots = [target_indices_slot()];
    let first = arm(vec![64, 160], vec![48]);
    let second = arm(vec![128, 160], vec![48]);
    let error = shared_signature(
        &[
            SplicedArm {
                recipe: &first,
                slots: &slots,
                outputs: &[0],
            },
            SplicedArm {
                recipe: &second,
                slots: &slots,
                outputs: &[0],
            },
        ],
        1,
        &slots,
        1,
    )
    .unwrap_err();
    assert!(error.contains("64 and 128"), "{error}");
}

#[test]
fn a_shared_port_no_arm_binds_is_refused() {
    let (declared, composed) = (target_indices_slot(), composed_slot());
    let only = arm(vec![64, 160], vec![48]);
    let error = shared_signature(
        &[SplicedArm {
            recipe: &only,
            slots: std::slice::from_ref(&declared),
            outputs: &[0],
        }],
        1,
        &[declared.clone(), composed],
        1,
    )
    .unwrap_err();
    assert!(error.contains("shared port 2"), "{error}");
}

#[test]
fn an_arm_that_states_fewer_ports_than_it_binds_is_refused() {
    let slots = [target_indices_slot()];
    let short = arm(vec![64], vec![48]);
    let error = shared_signature(
        &[SplicedArm {
            recipe: &short,
            slots: &slots,
            outputs: &[0],
        }],
        1,
        &slots,
        1,
    )
    .unwrap_err();
    assert!(error.contains("without stating it"), "{error}");
}
