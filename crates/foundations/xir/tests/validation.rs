// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use domain_index::FiniteSetId;
use smallvec::{SmallVec, smallvec};

use xir::builder::ModuleBuilder;
use xir::dim_numbers::{DotDimensionNumbers, GatherDimensionNumbers, ScatterDimensionNumbers};
use xir::dtype::DType;
use xir::ir::{Def, FuncId, ValueDef};
use xir::op::{FuncSig, InferCtx, Op};
use xir::ops::{DotAlgorithm, DotGeneral, Gather, Scatter};
use xir::shape::Shape;
use xir::verify::verify;

fn infer<'a>(inputs: &'a [Shape], sigs: &'a [FuncSig]) -> InferCtx<'a> {
    InferCtx::new(inputs, sigs, Box::leak(Box::default()))
}

#[test]
fn dot_rejects_repeated_and_overlapping_axes() {
    let inputs = [
        Shape::new(DType::F32, [2, 2]),
        Shape::new(DType::F32, [2, 2]),
    ];
    for dnums in [
        DotDimensionNumbers::new(&[0, 0], &[0, 1], &[], &[]),
        DotDimensionNumbers::new(&[0], &[0], &[0], &[1]),
    ] {
        let op = DotGeneral {
            dnums,
            algorithm: DotAlgorithm::f32_f32_f32(),
        };
        assert!(op.infer(&infer(&inputs, &[])).is_err());
    }
}

fn vector_gather(offset_dims: &[u32]) -> Gather {
    Gather {
        dnums: GatherDimensionNumbers {
            offset_dims: SmallVec::from_slice(offset_dims),
            collapsed_slice_dims: smallvec![0],
            operand_batching_dims: SmallVec::new(),
            start_indices_batching_dims: SmallVec::new(),
            start_index_map: smallvec![0],
            index_vector_dim: 1,
        },
        slice_sizes: smallvec![1],
        indices_are_sorted: false,
    }
}

#[test]
fn gather_rejects_duplicate_offset_dims_without_panicking() {
    let inputs = [
        Shape::new(DType::F32, [8, 4]),
        Shape::new(DType::I32, [3, 1]),
    ];
    let op = Gather {
        dnums: GatherDimensionNumbers {
            offset_dims: smallvec![0, 0],
            collapsed_slice_dims: SmallVec::new(),
            operand_batching_dims: SmallVec::new(),
            start_indices_batching_dims: SmallVec::new(),
            start_index_map: smallvec![0],
            index_vector_dim: 1,
        },
        slice_sizes: smallvec![8, 4],
        indices_are_sorted: false,
    };
    assert!(op.infer(&infer(&inputs, &[])).is_err());
}

#[test]
fn gather_validates_indices_and_dimension_relations() {
    let good = [Shape::new(DType::F32, [8]), Shape::new(DType::I32, [3, 1])];
    assert_eq!(
        vector_gather(&[]).infer(&infer(&good, &[])).unwrap()[0]
            .dims
            .as_slice(),
        &[3]
    );

    let non_integer = [good[0].clone(), Shape::new(DType::F32, [3, 1])];
    assert!(vector_gather(&[]).infer(&infer(&non_integer, &[])).is_err());

    let mut duplicate_map = vector_gather(&[]);
    duplicate_map.dnums.start_index_map = smallvec![0, 0];
    let wider_indices = [good[0].clone(), Shape::new(DType::I32, [3, 2])];
    assert!(duplicate_map.infer(&infer(&wider_indices, &[])).is_err());
}

fn scatter(update: FuncId) -> Scatter {
    Scatter {
        dnums: ScatterDimensionNumbers {
            update_window_dims: SmallVec::new(),
            inserted_window_dims: smallvec![0],
            input_batching_dims: SmallVec::new(),
            scatter_indices_batching_dims: SmallVec::new(),
            scatter_dims_to_operand_dims: smallvec![0],
            index_vector_dim: 1,
        },
        indices_are_sorted: false,
        unique_indices: false,
        update,
    }
}

#[test]
fn scatter_update_signature_inherits_operand_domain_axes() {
    let domain = FiniteSetId(0);
    let operand = Shape::new(DType::F32, [8]).with_domain_axes([domain]);
    let indices = Shape::new(DType::I32, [3, 1]).with_domain_axes([domain]);
    let updates = Shape::new(DType::F32, [3]).with_domain_axes([domain]);
    let scalar = Shape::scalar(DType::F32).with_domain_axes([domain]);
    let inputs = [operand.clone(), indices, updates];
    let sigs = [FuncSig {
        params: vec![scalar.clone(), scalar.clone()],
        results: vec![scalar],
    }];

    assert_eq!(
        scatter(FuncId(0)).infer(&infer(&inputs, &sigs)).unwrap()[0],
        operand
    );
}

#[test]
fn scatter_validates_dimensions_shapes_indices_and_update_signature() {
    let inputs = [
        Shape::new(DType::F32, [8]),
        Shape::new(DType::I32, [3, 1]),
        Shape::new(DType::F32, [3]),
    ];
    let scalar = Shape::scalar(DType::F32);
    let sigs = [FuncSig {
        params: vec![scalar.clone(), scalar.clone()],
        results: vec![scalar.clone()],
    }];
    assert_eq!(
        scatter(FuncId(0)).infer(&infer(&inputs, &sigs)).unwrap()[0],
        inputs[0]
    );

    let bad_indices = [
        inputs[0].clone(),
        Shape::new(DType::F32, [3, 1]),
        inputs[2].clone(),
    ];
    assert!(
        scatter(FuncId(0))
            .infer(&infer(&bad_indices, &sigs))
            .is_err()
    );

    let bad_updates = [
        inputs[0].clone(),
        inputs[1].clone(),
        Shape::new(DType::F32, [4]),
    ];
    assert!(
        scatter(FuncId(0))
            .infer(&infer(&bad_updates, &sigs))
            .is_err()
    );

    let bad_sigs = [FuncSig {
        params: vec![scalar.clone(), scalar.clone()],
        results: vec![Shape::scalar(DType::I32)],
    }];
    assert!(
        scatter(FuncId(0))
            .infer(&infer(&inputs, &bad_sigs))
            .is_err()
    );
}

#[test]
fn verifier_rejects_non_bijective_parameter_definitions() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |function| {
        let value = function.param(Shape::scalar(DType::F32));
        vec![value]
    });
    let mut module = builder.finish(main).unwrap();

    module.funcs[0].values[0].def = Def::Param(1);
    assert!(verify(&module).is_err());

    module.funcs[0].values[0].def = Def::Param(0);
    module.funcs[0].values.push(ValueDef {
        shape: Shape::scalar(DType::F32),
        def: Def::Param(0),
    });
    assert!(verify(&module).is_err());
}
