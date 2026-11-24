// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source-only estimates of native assembly and linking cost.

use crate::CompileRequest;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompilationCostModel {
    /// What any compile costs before its size matters: process setup, linking.
    pub per_unit_ms: f64,
    pub per_kib_ms: f64,
    pub per_kib_squared_ms: f64,
}

impl Default for CompilationCostModel {
    fn default() -> Self {
        CompilationCostModel {
            per_unit_ms: 8.0,
            per_kib_ms: 0.96,
            per_kib_squared_ms: 2.5e-3,
        }
    }
}

impl CompilationCostModel {
    /// What one compile unit of `bytes` of PTX is priced at.
    pub fn source_cost_ms(&self, bytes: usize) -> f64 {
        let kib = bytes as f64 / 1024.0;
        self.per_unit_ms + kib * self.per_kib_ms + kib * kib * self.per_kib_squared_ms
    }

    /// What one compile request is priced at: its sources, plus one unit cost
    /// for the link that joins them.
    pub fn request_cost_ms(&self, request: &CompileRequest) -> f64 {
        let sources: f64 = request
            .sources
            .iter()
            .map(|source| self.source_cost_ms(source.blob.len()))
            .sum();
        sources + self.per_unit_ms
    }
}
