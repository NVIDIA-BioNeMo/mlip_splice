// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PortAliases(BTreeMap<usize, usize>);

impl PortAliases {
    pub fn new(
        aliases: impl IntoIterator<Item = (usize, usize)>,
        inputs: usize,
        outputs: usize,
    ) -> Result<Self, String> {
        let mut held = BTreeMap::new();
        let mut aliased_outputs = BTreeSet::new();
        for (input, output) in aliases {
            if input >= inputs {
                return Err(format!(
                    "alias input {input} is outside {inputs} input port(s)"
                ));
            }
            if output >= outputs {
                return Err(format!(
                    "alias output {output} is outside {outputs} output port(s)"
                ));
            }
            if held.insert(input, output).is_some() {
                return Err(format!("input {input} is aliased more than once"));
            }
            if !aliased_outputs.insert(output) {
                return Err(format!("output {output} is aliased more than once"));
            }
        }
        Ok(Self(held))
    }

    pub fn empty() -> Self {
        Self::default()
    }

    pub fn as_map(&self) -> &BTreeMap<usize, usize> {
        &self.0
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.0.iter().map(|(&input, &output)| (input, output))
    }

    pub fn output_for_input(&self, input: usize) -> Option<usize> {
        self.0.get(&input).copied()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
