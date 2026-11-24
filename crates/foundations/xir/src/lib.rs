// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `xir` — a Rust-native SSA tensor IR.
//!
//! The IR is a static-shaped, **layout-free** dataflow graph. A value's
//! [`Shape`](shape::Shape) is `dtype + dims` and carries no layout; layout is a
//! later, separate concern. Operations form an open registry: every op — built
//! in or future custom — implements the single [`Op`](op::Op) trait, so custom
//! ops are first-class. Op semantics are copied from StableHLO/HLO (the
//! dimension-number algebra, higher-order ops carrying a sub-graph, and the
//! no-implicit-broadcast discipline).
//!
//! ```
//! use xir::builder::ModuleBuilder;
//! use xir::dim_numbers::DotDimensionNumbers;
//! use xir::dtype::DType;
//! use xir::ops::DotAlgorithm;
//! use xir::shape::Shape;
//!
//! let mut m = ModuleBuilder::new();
//! let main = m.func("main", |f| {
//!     let a = f.param(Shape::new(DType::F32, [128, 256]));
//!     let b = f.param(Shape::new(DType::F32, [256, 64]));
//!     let c = f.dot_general(a, b, DotDimensionNumbers::matmul(), DotAlgorithm::tf32_tf32_f32());
//!     vec![c]
//! });
//! let module = m.finish(main).unwrap();
//! assert_eq!(module.entry_func().results.len(), 1);
//! ```

pub mod analysis;
pub mod builder;
pub mod dialect;
mod dialect_types;
pub mod dim_numbers;
pub mod dtype;
pub mod element_mapping;
pub mod generic_text;
mod import;
pub mod ir;
pub mod op;
mod op_factory;
pub mod ops;
pub mod print;
pub mod scalarize;
pub mod shape;
pub mod verify;
