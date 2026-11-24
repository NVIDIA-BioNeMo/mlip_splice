// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The built-in op set. Semantics are copied from StableHLO/HLO: the
//! dimension-number algebra, higher-order ops carrying a sub-graph, and the
//! no-implicit-broadcast discipline (elementwise operands must match exactly).

mod contract_tensors;
mod convolve_tensors;
mod elementwise;
mod higher_order;
mod index_tensors;
mod infer_shapes;
mod literal;
mod map_domain_axes;
mod transform_shapes;
mod transform_spectra;

pub use contract_tensors::*;
pub use convolve_tensors::*;
pub use elementwise::*;
pub use higher_order::*;
pub use index_tensors::*;
pub use infer_shapes::Padding;
pub use literal::*;
pub use map_domain_axes::*;
pub use transform_shapes::*;
pub use transform_spectra::*;
