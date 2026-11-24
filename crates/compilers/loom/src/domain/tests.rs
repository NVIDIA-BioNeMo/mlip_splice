// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Slot semantics and order determine the GPU array parameters.

use domain_index::{AlignedView, AxisBridge, Domain, Graph, IndexMap, OperandGraph};

use super::*;

struct Index {
    graph: Graph,
    operands: usize,
}

impl DomainIndex for Index {
    fn graph(&self) -> &Graph {
        &self.graph
    }
    fn num_operands(&self) -> usize {
        self.operands
    }
    fn num_axes(&self) -> usize {
        panic!("parameter construction does not inspect axes")
    }
    fn iter_domain(&self, _axis: usize) -> i32 {
        panic!("parameter construction does not inspect iteration domains")
    }
    fn operand_domain(&self, _operand: usize, _axis: usize) -> i32 {
        panic!("parameter construction does not inspect operand domains")
    }
    fn axis_bridge(&self, _operand: usize, _axis: usize) -> AxisBridge {
        panic!("parameter construction does not resolve bridges")
    }
}

fn index_map(source_id: i32, target_id: i32, is_sorted: bool) -> IndexMap {
    IndexMap {
        source_id,
        target_id,
        is_sorted,
        is_injective: false,
        may_have_unmapped: false,
    }
}

fn slot(kind: SlotKind, source_id: i32, target_id: i32) -> Slot {
    Slot {
        kind,
        source_id,
        target_id,
        other_target_id: target_id,
    }
}

fn view(source_id: i32, anchor_target_id: i32, other_target_id: i32) -> AlignedView {
    AlignedView {
        source_id,
        anchor_target_id,
        other_target_id,
    }
}

fn view_slot(view: AlignedView) -> Slot {
    Slot {
        kind: SlotKind::AlignedTargetIndices,
        source_id: view.source_id,
        target_id: view.anchor_target_id,
        other_target_id: view.other_target_id,
    }
}

fn param_names(layout: &Layout) -> Vec<&str> {
    layout
        .params
        .iter()
        .map(|param| param.name.as_str())
        .collect()
}

fn slot_semantics(slots: &[Slot]) -> Vec<(SlotKind, i32, i32, i32)> {
    slots
        .iter()
        .map(|slot| {
            (
                slot.kind,
                slot.source_id,
                slot.target_id,
                slot.other_target_id,
            )
        })
        .collect()
}

#[test]
fn slot_kinds_and_order_determine_arrays_and_graph_bindings() {
    let first_view = view(0, 1, 2);
    let second_view = view(6, 7, 4);
    let index = Index {
        graph: Graph {
            domains: vec![Domain { extent: 8 }; 8],
            index_maps: vec![
                index_map(0, 2, false),
                index_map(6, 7, true),
                index_map(3, 5, false),
                index_map(0, 1, false),
                index_map(3, 4, true),
            ],
            aligned_views: vec![second_view, view(0, 1, 5), first_view],
        },
        operands: 3,
    };
    let slots = [
        slot(SlotKind::Preimage, 0, 1),
        view_slot(first_view),
        slot(SlotKind::TargetIndices, 3, 4),
        slot(SlotKind::ComposedTargetIndices, 3, 5),
        slot(SlotKind::PreimageSorted, 6, 7),
        view_slot(second_view),
    ];
    let layout = Layout::new(
        &index,
        &slots,
        2,
        &[Scalar::F64, Scalar::F16, Scalar::S32],
        &[],
    )
    .unwrap();
    assert_eq!(
        param_names(&layout),
        [
            "in_0",
            "in_1",
            "preimage_offsets_0",
            "preimage_source_indices_0",
            "aligned_target_indices_0",
            "target_indices_1",
            "target_indices_2",
            "preimage_offsets_3",
            "aligned_target_indices_1",
            "out_0",
        ]
    );
    assert_eq!(layout.used_index_maps, [3, 4, 2, 1]);
    assert_eq!(layout.used_views, [2, 0]);
    assert_eq!(
        layout.operand_params,
        [ParamSlot(0), ParamSlot(1), ParamSlot(9)]
    );
    assert_eq!(layout.params[0].ty, ParamTy::Ptr(Scalar::F64));
    assert_eq!(layout.params[1].ty, ParamTy::Ptr(Scalar::F16));
    assert!(
        layout.params[2..]
            .iter()
            .all(|p| p.ty == ParamTy::Ptr(Scalar::S32))
    );
    for (map_idx, target, offsets, sources) in [
        (3, None, Some(2), Some(3)),
        (4, Some(5), None, None),
        (2, Some(6), None, None),
        (1, None, Some(7), None),
    ] {
        let refs = layout.ref_params(map_idx);
        assert_eq!(refs.target_indices, target.map(ParamSlot));
        assert_eq!(refs.preimage_offsets, offsets.map(ParamSlot));
        assert_eq!(refs.preimage_source_indices, sources.map(ParamSlot));
    }
    assert_eq!(layout.view_param(2), ParamSlot(4));
    assert_eq!(layout.view_param(0), ParamSlot(8));
    assert_eq!(layout.param_name(layout.operand_params[2]), "out_0");
}

#[test]
fn built_slots_keep_views_before_later_maps_in_both_csr_modes() {
    for sorted in [false, true] {
        let domains = vec![Domain { extent: 8 }; 5];
        let maps = [
            index_map(0, 1, sorted),
            index_map(0, 2, false),
            index_map(3, 4, false),
        ];
        let views = [view(0, 1, 2)];
        let operands = [vec![2, 4], vec![0, 3], vec![1, 3]];
        let operand_graph = OperandGraph::new(&domains, &maps, &operands, 2, 2, &views).unwrap();
        let (graph, iter, slots) = operand_graph.build_index(&[1, 3], &views).unwrap();
        assert_eq!(iter, [1, 3]);
        assert_eq!(
            slot_semantics(&slots),
            [
                (
                    if sorted {
                        SlotKind::PreimageSorted
                    } else {
                        SlotKind::Preimage
                    },
                    0,
                    1,
                    1
                ),
                (SlotKind::AlignedTargetIndices, 0, 1, 2),
                (SlotKind::TargetIndices, 3, 4, 4),
            ]
        );
        let index = Index {
            graph,
            operands: operands.len(),
        };
        let layout = Layout::new(&index, &slots, 2, &[Scalar::F32; 3], &[]).unwrap();
        let mut expected = vec!["in_0", "in_1", "preimage_offsets_0"];
        if !sorted {
            expected.push("preimage_source_indices_0");
        }
        expected.extend(["aligned_target_indices_0", "target_indices_1", "out_0"]);
        assert_eq!(param_names(&layout), expected);
        assert_eq!(layout.used_index_maps, [0, 1]);
        assert_eq!(layout.used_views, [0]);
        assert_eq!(
            layout.param_name(layout.view_param(0)),
            "aligned_target_indices_0"
        );
        assert_eq!(
            layout.param_name(layout.ref_params(1).target_indices.unwrap()),
            "target_indices_1"
        );
    }
}

#[test]
fn built_composed_slot_binds_one_target_array_once() {
    let domains = vec![Domain { extent: 8 }; 3];
    let maps = [index_map(0, 1, false), index_map(1, 2, false)];
    let operands = [vec![2], vec![2], vec![0]];
    let operand_graph = OperandGraph::new(&domains, &maps, &operands, 1, 2, &[]).unwrap();
    let (graph, _, slots) = operand_graph.build_index(&[0], &[]).unwrap();
    assert_eq!(
        slot_semantics(&slots),
        [(SlotKind::ComposedTargetIndices, 0, 2, 2)]
    );
    let index = Index {
        graph,
        operands: operands.len(),
    };
    let layout = Layout::new(&index, &slots, 2, &[Scalar::F32; 3], &[]).unwrap();
    assert_eq!(
        param_names(&layout),
        ["in_0", "in_1", "target_indices_0", "out_0"]
    );
    assert_eq!(layout.used_index_maps, [0]);
    assert_eq!(layout.ref_params(0).target_indices, Some(ParamSlot(2)));
}

#[test]
fn trivial_and_zero_axis_placements_have_no_index_arrays() {
    for axes in [0, 1] {
        let operands = [vec![-1; axes], vec![-1; axes]];
        let operand_graph = OperandGraph::new(&[], &[], &operands, axes, 1, &[]).unwrap();
        let (graph, iter, slots) = operand_graph.build_index(&vec![-1; axes], &[]).unwrap();
        assert_eq!(iter, vec![0; axes]);
        assert!(slots.is_empty());
        let index = Index {
            graph,
            operands: operands.len(),
        };
        let layout = Layout::new(&index, &slots, 1, &[Scalar::F32; 2], &[]).unwrap();
        assert_eq!(param_names(&layout), ["in_0", "out_0"]);
        assert!(layout.used_index_maps.is_empty());
        assert!(layout.used_views.is_empty());
        assert!(layout.ref_params.is_empty());
        assert!(layout.view_params.is_empty());
    }
}

#[test]
fn empty_slots_bind_only_the_requested_operands() {
    let graph = Graph {
        domains: vec![Domain { extent: 8 }; 3],
        index_maps: vec![index_map(0, 1, false)],
        aligned_views: vec![view(0, 1, 2)],
    };
    for (operands, inputs, names) in [
        (0, 0, vec![]),
        (2, 0, vec!["out_0", "out_1"]),
        (2, 2, vec!["in_0", "in_1"]),
    ] {
        let index = Index {
            graph: graph.clone(),
            operands,
        };
        let layout = Layout::new(&index, &[], inputs, &vec![Scalar::F32; operands], &[]).unwrap();
        assert_eq!(param_names(&layout), names);
        assert!(layout.used_index_maps.is_empty());
        assert!(layout.used_views.is_empty());
    }
}

#[test]
fn slots_must_name_an_existing_map_or_exact_view() {
    let index = Index {
        graph: Graph {
            domains: vec![Domain { extent: 8 }; 4],
            index_maps: vec![index_map(0, 1, false)],
            aligned_views: vec![view(0, 1, 2)],
        },
        operands: 0,
    };
    for kind in [
        SlotKind::TargetIndices,
        SlotKind::ComposedTargetIndices,
        SlotKind::Preimage,
        SlotKind::PreimageSorted,
    ] {
        for (source, target) in [(1, 0), (0, 2), (2, 1)] {
            let slots = [slot(kind, source, target)];
            let error = Layout::new(&index, &slots, 0, &[], &[]).err().unwrap();
            assert!(error.contains("has no index map in the graph"), "{error}");
        }
    }
    for wrong_view in [view(3, 1, 2), view(0, 3, 2), view(0, 1, 3)] {
        let error = Layout::new(&index, &[view_slot(wrong_view)], 0, &[], &[])
            .err()
            .unwrap();
        assert!(
            error.contains("has no aligned view in the graph"),
            "{error}"
        );
    }
}

#[test]
fn operand_counts_are_checked() {
    let index = Index {
        graph: Graph {
            domains: vec![],
            index_maps: vec![],
            aligned_views: vec![],
        },
        operands: 1,
    };
    let error = Layout::new(&index, &[], 2, &[Scalar::F32], &[])
        .err()
        .unwrap();
    assert_eq!(error, "num_inputs=2 exceeds num_operands=1");
    let error = Layout::new(&index, &[], 1, &[], &[]).err().unwrap();
    assert_eq!(error, "got 0 operand scalars for 1 operands");
}

#[test]
fn a_bucket_mask_is_not_a_layout_parameter() {
    let index = Index {
        graph: Graph {
            domains: vec![Domain { extent: 8 }; 2],
            index_maps: vec![index_map(0, 1, false)],
            aligned_views: vec![],
        },
        operands: 3,
    };
    let slots = [slot(SlotKind::TargetIndices, 0, 1)];
    let layout = Layout::new(&index, &slots, 1, &[Scalar::F32; 3], &[1, 0]).unwrap();
    assert_eq!(
        param_names(&layout),
        ["in_0", "target_indices_0", "out_0", "out_1"]
    );
    assert_eq!(layout.bucket_mask_slot(1), Some(BucketMaskSlot(0)));
    assert_eq!(layout.bucket_mask_slot(0), Some(BucketMaskSlot(1)));
    assert!(layout.bucket_mask_slot(2).is_none());
    for (mask, output) in layout.bucket_masks.iter().zip([1, 0]) {
        assert_eq!(mask.output, output);
        assert_eq!(mask.param.name, format!("bmask_{output}"));
        assert_eq!(mask.param.ty, ParamTy::Scalar(Scalar::S32));
    }
}

#[test]
fn multiple_slots_for_one_graph_handle_are_rejected() {
    let index = Index {
        graph: Graph {
            domains: vec![Domain { extent: 8 }; 3],
            index_maps: vec![index_map(0, 1, false)],
            aligned_views: vec![view(0, 1, 2)],
        },
        operands: 0,
    };
    for (slots, message) in [
        (
            [
                slot(SlotKind::TargetIndices, 0, 1),
                slot(SlotKind::Preimage, 0, 1),
            ],
            "index map 0 has multiple slots",
        ),
        (
            [view_slot(view(0, 1, 2)), view_slot(view(0, 1, 2))],
            "aligned view 0 has multiple slots",
        ),
    ] {
        let error = Layout::new(&index, &slots, 0, &[], &[]).err().unwrap();
        assert_eq!(error, message);
    }
}
