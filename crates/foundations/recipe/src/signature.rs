// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The ports a device-executable recipe is bound through.

use crate::PortAliases;

/// The external byte-buffer ports of a recipe: one extent per port and an
/// explicit input-to-output alias relation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecipeSignature {
    input_nbytes: Vec<usize>,
    output_nbytes: Vec<usize>,
    aliases: PortAliases,
}

impl RecipeSignature {
    pub fn new(input_nbytes: Vec<usize>, output_nbytes: Vec<usize>) -> Self {
        RecipeSignature {
            input_nbytes,
            output_nbytes,
            aliases: PortAliases::empty(),
        }
    }

    pub fn with_aliases(
        input_nbytes: Vec<usize>,
        output_nbytes: Vec<usize>,
        aliases: PortAliases,
    ) -> Self {
        RecipeSignature {
            input_nbytes,
            output_nbytes,
            aliases,
        }
    }

    pub fn empty() -> Self {
        RecipeSignature::default()
    }

    pub fn inputs(&self) -> &[usize] {
        &self.input_nbytes
    }

    pub fn outputs(&self) -> &[usize] {
        &self.output_nbytes
    }

    pub fn aliases(&self) -> &PortAliases {
        &self.aliases
    }

    pub fn num_inputs(&self) -> usize {
        self.input_nbytes.len()
    }

    pub fn num_outputs(&self) -> usize {
        self.output_nbytes.len()
    }

    pub fn input_nbytes(&self, port: usize) -> Option<usize> {
        self.input_nbytes.get(port).copied()
    }

    pub fn output_nbytes(&self, port: usize) -> Option<usize> {
        self.output_nbytes.get(port).copied()
    }
}
