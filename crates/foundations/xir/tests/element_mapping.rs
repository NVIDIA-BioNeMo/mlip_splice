// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use xir::dim_numbers::{
    Axes, DotDimensionNumbers, GatherDimensionNumbers, ScatterDimensionNumbers,
};
use xir::element_mapping::{
    DotPlan, axes, broadcast_elements, concatenate_elements, gather_map, num_elements,
    pad_elements, ravel, reduction_map, scatter_map, slice_elements, transpose_elements, unravel,
};
use xir::ir::FuncId;
use xir::ops::{DotAlgorithm, DotGeneral, Gather, Pad, Scatter, Slice};
use xir::shape::Dims;

#[test]
fn row_major_coordinates_cover_scalars_vectors_and_tensors() {
    for dims in [&[][..], &[4], &[2, 3, 4]] {
        for flat in 0..num_elements(dims).unwrap() {
            let coord = unravel(flat, dims).unwrap();
            assert_eq!(coord.len(), dims.len());
            assert_eq!(ravel(&coord, dims).unwrap(), flat);
        }
        assert!(unravel(num_elements(dims).unwrap(), dims).is_err());
    }
    assert_eq!(num_elements(&[]).unwrap(), 1);
    assert_eq!(unravel(0, &[]).unwrap(), Vec::<usize>::new());
    assert_eq!(unravel(17, &[2, 3, 4]).unwrap(), [1, 1, 1]);
    assert!(ravel(&[0], &[]).is_err());
    assert!(ravel(&[], &[2]).is_err());
    assert!(ravel(&[2, 0], &[2, 3]).is_err());
    assert!(num_elements(&[-1]).is_err());
    assert!(ravel(&[0], &[-1]).is_err());
    assert!(num_elements(&[i64::MAX; 3]).is_err());
    assert!(ravel(&[i64::MAX as usize - 1; 3], &[i64::MAX; 3]).is_err());
}

#[test]
fn zero_length_axes_remain_unsupported_by_element_expansion() {
    for dims in [&[0][..], &[2, 0, 3]] {
        assert!(num_elements(dims).unwrap_err().contains("zero-sized"));
        assert!(unravel(0, dims).is_err());
        assert!(ravel(&vec![0; dims.len()], dims).is_err());
        assert!(reduction_map(dims, &[]).is_err());
    }
    assert!(broadcast_elements(&[7], &[], &[0], &[]).is_err());
    assert!(transpose_elements::<i32>(&[], &[0, 2], &[1, 0]).is_err());
    assert!(
        slice_elements(
            &[7],
            &[1],
            &Slice {
                start_indices: Dims::from_slice(&[0]),
                limit_indices: Dims::from_slice(&[0]),
                strides: Dims::from_slice(&[1]),
            }
        )
        .is_err()
    );
}

#[test]
fn scalar_views_have_one_element_and_no_axes() {
    assert_eq!(broadcast_elements(&[7], &[], &[], &[]).unwrap(), [7]);
    assert_eq!(transpose_elements(&[7], &[], &[]).unwrap(), [7]);
    assert_eq!(
        slice_elements(
            &[7],
            &[],
            &Slice {
                start_indices: Dims::new(),
                limit_indices: Dims::new(),
                strides: Dims::new(),
            }
        )
        .unwrap(),
        [7]
    );
    assert_eq!(
        pad_elements(
            &[7],
            &[],
            &0,
            &Pad {
                edge_padding_low: Dims::new(),
                edge_padding_high: Dims::new(),
                interior_padding: Dims::new(),
            }
        )
        .unwrap(),
        [7]
    );
    assert_eq!(reduction_map(&[], &[]).unwrap(), (vec![], vec![0]));
}

#[test]
fn broadcast_maps_scalar_and_singleton_axes() {
    assert_eq!(broadcast_elements(&[7], &[], &[2, 3], &[]).unwrap(), [7; 6]);
    assert_eq!(
        broadcast_elements(&[1, 2], &[2, 1], &[3, 2, 2], &[1, 2]).unwrap(),
        [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2]
    );
    assert_eq!(
        broadcast_elements(&[1, 2, 3], &[3], &[2, 3], &[1]).unwrap(),
        [1, 2, 3, 1, 2, 3]
    );
}

#[test]
fn transpose_slice_and_concatenate_preserve_row_major_order() {
    assert_eq!(
        transpose_elements(&[0, 1, 2, 3, 4, 5], &[2, 3], &[1, 0]).unwrap(),
        [0, 3, 1, 4, 2, 5]
    );
    assert_eq!(
        slice_elements(
            &(0..12).collect::<Vec<_>>(),
            &[3, 4],
            &Slice {
                start_indices: Dims::from_slice(&[0, 1]),
                limit_indices: Dims::from_slice(&[3, 4]),
                strides: Dims::from_slice(&[2, 2]),
            }
        )
        .unwrap(),
        [1, 3, 9, 11]
    );
    let left_dims = Dims::from_slice(&[2, 1]);
    let right_dims = Dims::from_slice(&[2, 2]);
    assert_eq!(
        concatenate_elements(&[(&[0, 3], &left_dims), (&[1, 2, 4, 5], &right_dims)], 1).unwrap(),
        [0, 1, 2, 3, 4, 5]
    );
}

#[test]
fn pad_maps_edges_interiors_and_negative_edge_cropping() {
    assert_eq!(
        pad_elements(
            &[1, 2, 3, 4],
            &[2, 2],
            &0,
            &Pad {
                edge_padding_low: Dims::from_slice(&[1, 0]),
                edge_padding_high: Dims::from_slice(&[0, 1]),
                interior_padding: Dims::from_slice(&[0, 1]),
            }
        )
        .unwrap(),
        [0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0]
    );
    assert_eq!(
        pad_elements(
            &[1, 2, 3, 4],
            &[4],
            &0,
            &Pad {
                edge_padding_low: Dims::from_slice(&[-1]),
                edge_padding_high: Dims::from_slice(&[-1]),
                interior_padding: Dims::from_slice(&[0]),
            }
        )
        .unwrap(),
        [2, 3]
    );
}

#[test]
fn reduction_maps_drop_only_reduced_axes() {
    assert_eq!(axes(&[2, 0]), [2, 0]);
    let (dims, map) = reduction_map(&[2, 3, 2], &[0, 2]).unwrap();
    assert_eq!(dims, [3]);
    assert_eq!(map, [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]);
    assert_eq!(
        reduction_map(&[2, 3], &[0, 1]).unwrap(),
        (vec![], vec![0; 6])
    );
    assert_eq!(
        reduction_map(&[2, 3], &[]).unwrap(),
        (vec![2, 3], (0..6).collect())
    );
}

fn dot(dnums: DotDimensionNumbers) -> DotGeneral {
    DotGeneral {
        dnums,
        algorithm: DotAlgorithm::f32_f32_f32(),
    }
}

#[test]
fn dot_plan_maps_scalar_outer_product_and_permuted_batch_axes() {
    let scalar = DotPlan::new(&[], &[], &dot(DotDimensionNumbers::default())).unwrap();
    assert_eq!(scalar.out_dims, []);
    assert_eq!(scalar.contract_dims, []);
    assert_eq!(scalar.contraction(0).unwrap(), [(0, 0)]);
    let outer = DotPlan::new(&[2], &[3], &dot(DotDimensionNumbers::default())).unwrap();
    assert_eq!(outer.out_dims, [2, 3]);
    assert_eq!(outer.contraction(4).unwrap(), [(1, 1)]);
    // lhs[k,b,m] rhs[n,b,k] -> output[b,m,n], summing over k.
    let plan = DotPlan::new(
        &[3, 2, 2],
        &[4, 2, 3],
        &dot(DotDimensionNumbers::new(&[0], &[2], &[1], &[1])),
    )
    .unwrap();
    assert_eq!(plan.out_dims, [2, 2, 4]);
    assert_eq!(plan.contract_dims, [3]);
    for batch in 0..2 {
        for left in 0..2 {
            for right in 0..4 {
                let expected = (0..3)
                    .map(|k| ((k * 2 + batch) * 2 + left, (right * 2 + batch) * 3 + k))
                    .collect::<Vec<_>>();
                assert_eq!(
                    plan.contraction((batch * 2 + left) * 4 + right).unwrap(),
                    expected
                );
            }
        }
    }
}

fn gather(dnums: GatherDimensionNumbers, slice_sizes: &[i64]) -> Gather {
    Gather {
        dnums,
        slice_sizes: Dims::from_slice(slice_sizes),
        indices_are_sorted: false,
    }
}

#[test]
fn gather_maps_implicit_scalar_indices_and_clamps_explicit_windows() {
    let mut op = gather(
        GatherDimensionNumbers {
            offset_dims: Axes::from_slice(&[0]),
            collapsed_slice_dims: Axes::from_slice(&[0]),
            start_index_map: Axes::from_slice(&[0]),
            ..Default::default()
        },
        &[1, 3],
    );
    assert_eq!(
        gather_map(&[4, 3], &[], &[2], &[3], &op).unwrap(),
        [6, 7, 8]
    );
    op.dnums.offset_dims = Axes::from_slice(&[1]);
    op.dnums.index_vector_dim = 1;
    op.slice_sizes = Dims::from_slice(&[1, 2]);
    assert_eq!(
        gather_map(&[4, 3], &[2, 1], &[-1, 99], &[2, 2], &op).unwrap(),
        [0, 1, 9, 10]
    );
    assert!(gather_map(&[4, 3], &[2, 1], &[1], &[2, 2], &op).is_err());
    assert!(gather_map(&[4, 3], &[0, 1], &[], &[0, 2], &op).is_err());
}

#[test]
fn gather_maps_multiple_index_components_with_a_leading_vector_axis() {
    let op = gather(
        GatherDimensionNumbers {
            collapsed_slice_dims: Axes::from_slice(&[0, 1]),
            start_index_map: Axes::from_slice(&[0, 1]),
            ..Default::default()
        },
        &[1, 1],
    );
    assert_eq!(
        gather_map(&[3, 4], &[2, 2], &[0, 2, 1, 3], &[2], &op).unwrap(),
        [1, 11]
    );
}

fn scatter(dnums: ScatterDimensionNumbers) -> Scatter {
    Scatter {
        dnums,
        indices_are_sorted: false,
        unique_indices: false,
        update: FuncId(0),
    }
}

#[test]
fn scatter_maps_scalar_indices_and_drops_out_of_bounds_updates() {
    let mut op = scatter(ScatterDimensionNumbers {
        update_window_dims: Axes::from_slice(&[0]),
        inserted_window_dims: Axes::from_slice(&[0]),
        scatter_dims_to_operand_dims: Axes::from_slice(&[0]),
        ..Default::default()
    });
    assert_eq!(
        scatter_map(&[4, 3], &[], &[2], &[3], &op).unwrap(),
        [Some(6), Some(7), Some(8)]
    );
    op.dnums.update_window_dims = Axes::from_slice(&[1]);
    op.dnums.index_vector_dim = 1;
    assert_eq!(
        scatter_map(&[4, 3], &[4, 1], &[-1, 2, 2, 4], &[4, 3], &op).unwrap(),
        [
            None,
            None,
            None,
            Some(6),
            Some(7),
            Some(8),
            Some(6),
            Some(7),
            Some(8),
            None,
            None,
            None
        ]
    );
    assert!(scatter_map(&[4, 3], &[4, 1], &[2], &[4, 3], &op).is_err());
    assert!(scatter_map(&[4, 3], &[0, 1], &[], &[0, 3], &op).is_err());
}

#[test]
fn scatter_maps_batched_windows_and_partially_out_of_bounds_windows() {
    let op = scatter(ScatterDimensionNumbers {
        update_window_dims: Axes::from_slice(&[2]),
        input_batching_dims: Axes::from_slice(&[0]),
        scatter_indices_batching_dims: Axes::from_slice(&[0]),
        scatter_dims_to_operand_dims: Axes::from_slice(&[1]),
        index_vector_dim: 2,
        ..Default::default()
    });
    assert_eq!(
        scatter_map(&[2, 4], &[2, 2, 1], &[0, 3, -1, 2], &[2, 2, 2], &op).unwrap(),
        [
            Some(0),
            Some(1),
            Some(3),
            None,
            None,
            Some(4),
            Some(6),
            Some(7)
        ]
    );
}

#[test]
fn gather_and_scatter_map_rank_zero_results_and_updates() {
    let take = gather(
        GatherDimensionNumbers {
            collapsed_slice_dims: Axes::from_slice(&[0]),
            start_index_map: Axes::from_slice(&[0]),
            ..Default::default()
        },
        &[1],
    );
    assert_eq!(gather_map(&[4], &[], &[2], &[], &take).unwrap(), [2]);
    let update = scatter(ScatterDimensionNumbers {
        inserted_window_dims: Axes::from_slice(&[0]),
        scatter_dims_to_operand_dims: Axes::from_slice(&[0]),
        ..Default::default()
    });
    assert_eq!(
        scatter_map(&[4], &[], &[2], &[], &update).unwrap(),
        [Some(2)]
    );
}

#[test]
fn gather_rejects_unsupported_operand_batching() {
    let op = gather(
        GatherDimensionNumbers {
            collapsed_slice_dims: Axes::from_slice(&[1]),
            operand_batching_dims: Axes::from_slice(&[0]),
            start_indices_batching_dims: Axes::from_slice(&[0]),
            start_index_map: Axes::from_slice(&[1]),
            index_vector_dim: 1,
            ..Default::default()
        },
        &[1, 1],
    );
    assert_eq!(
        gather_map(&[2, 3], &[2], &[0, 1], &[2], &op).unwrap_err(),
        "gather window rank does not match offset_dims"
    );
}

#[test]
fn dot_plan_preserves_multiple_contraction_axis_order() {
    // lhs[m,j,k] rhs[k,j,n] -> output[m,n], with contraction order k,j.
    let op = dot(DotDimensionNumbers::new(&[2, 1], &[0, 1], &[], &[]));
    let plan = DotPlan::new(&[2, 3, 4], &[4, 3, 5], &op).unwrap();
    assert_eq!(plan.out_dims, [2, 5]);
    assert_eq!(plan.contract_dims, [4, 3]);
    let expected = (0..4)
        .flat_map(|k| (0..3).map(move |j| ((3 + j) * 4 + k, (k * 3 + j) * 5 + 2)))
        .collect::<Vec<_>>();
    assert_eq!(plan.contraction(7).unwrap(), expected);
}
