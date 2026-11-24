// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::rep::Mat;
use algebraic_real::AlgebraicReal;

pub mod o3;
pub mod so3;
pub mod su2;

/// Process-wide memo for an exact Clebsch-Gordan keyed on its integer label
/// triple, holding the heavy `Mat<AlgebraicReal>` behind an `Arc` so callers clone
/// cheaply. Each coefficient is a single surd (a rational multiple of one √).
type CgCache = OnceLock<Mutex<HashMap<(u32, u32, u32), Arc<Mat<AlgebraicReal>>>>>;
