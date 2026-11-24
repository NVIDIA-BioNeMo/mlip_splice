// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A custom op defined *outside* the crate is as first-class as a built-in: it
//! implements the same `Op` trait, so the builder, verifier, and printer handle
//! it with no special-casing. This is the extensibility the design promises for
//! future domain ops.

use std::any::Any;

use smallvec::{SmallVec, smallvec};

use xir::builder::ModuleBuilder;
use xir::dtype::DType;
use xir::op::{InferCtx, Op};
use xir::print::module_to_string;
use xir::shape::{Shape, ShapeError};

/// Reduce the last axis: `[.., d] -> [..]`.
#[derive(Clone, Debug)]
struct L2Norm;

impl Op for L2Norm {
    fn mnemonic(&self) -> &'static str {
        "custom.l2norm"
    }

    fn infer(&self, ctx: &InferCtx<'_>) -> Result<SmallVec<[Shape; 1]>, ShapeError> {
        let inputs = ctx.inputs();
        if inputs.len() != 1 {
            return Err(ShapeError::new("custom.l2norm", "expects one operand"));
        }
        let x = &inputs[0];
        if x.rank() == 0 {
            return Err(ShapeError::new(
                "custom.l2norm",
                "operand must have rank >= 1",
            ));
        }
        let dims = x.dims[..x.rank() - 1].iter().copied();
        Ok(smallvec![Shape::new(x.dtype, dims)])
    }

    fn attrs(&self) -> Vec<(&'static str, String)> {
        vec![("axis", "last".to_string())]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[test]
fn custom_op_is_first_class() {
    let mut m = ModuleBuilder::new();
    let main = m.func("main", |f| {
        let x = f.param(Shape::new(DType::F32, [32, 8]));
        let n = f.op1(L2Norm, &[x]);
        vec![n]
    });
    let module = m.finish(main).unwrap();

    // Inferred, verified, and printed exactly like a built-in op.
    let ef = module.entry_func();
    assert_eq!(ef.value_shape(ef.results[0]).dims.as_slice(), &[32]);

    let text = module_to_string(&module);
    assert!(
        text.contains("%1 = custom.l2norm %0 {axis=last} : f32[32]"),
        "{text}"
    );
}
