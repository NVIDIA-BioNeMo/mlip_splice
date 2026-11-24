// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use gpu_ir::{Builder, Value};
use xir::element_mapping::{DotPlan, num_elements};
use xir::ir::{Func, ValueId};
use xir::ops::DotGeneral;

pub(super) fn lower_dot_values(
    b: &mut Builder,
    lhs: &[Value],
    lhs_dims: &[i64],
    rhs: &[Value],
    rhs_dims: &[i64],
    op: &DotGeneral,
) -> Result<Vec<Value>, String> {
    let plan = DotPlan::new(lhs_dims, rhs_dims, op)?;
    let cells = num_elements(&plan.out_dims)?;
    let mut output = Vec::with_capacity(cells);
    for flat in 0..cells {
        let mut sum = None;
        for (left, right) in plan.contraction(flat)? {
            let product = b.mul(lhs[left].clone(), rhs[right].clone());
            sum = Some(match sum {
                None => product,
                Some(sum) => b.add(sum, product),
            });
        }
        output.push(sum.ok_or_else(|| "dot_general has an empty contraction".to_string())?);
    }
    Ok(output)
}

pub(super) fn zip_exact<'a, T, U>(
    lhs: &'a [T],
    rhs: &'a [U],
) -> Result<impl Iterator<Item = (&'a T, &'a U)>, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "elementwise sizes differ: {} vs {}",
            lhs.len(),
            rhs.len()
        ));
    }
    Ok(lhs.iter().zip(rhs))
}

pub(super) fn shapes(func: &Func, values: &[ValueId]) -> String {
    values
        .iter()
        .map(|&value| {
            let shape = func.value_shape(value);
            format!("{}{:?}", shape.dtype.name(), shape.dims)
        })
        .collect::<Vec<_>>()
        .join(", ")
}
