// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::isolated_root::IsolatedRoot;
use crate::multiquadratic::Multiquadratic;
use crate::number_field::NumberFieldElement;

#[derive(Clone, Debug)]
pub(crate) enum Representation {
    Multiquadratic(Multiquadratic),
    NumberField(NumberFieldElement),
    IsolatedRoot(IsolatedRoot),
}

impl Representation {
    pub(crate) fn same_stored_value(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Multiquadratic(left), Self::Multiquadratic(right)) => left == right,
            (Self::NumberField(left), Self::NumberField(right)) => left.same_field_element(right),
            (Self::IsolatedRoot(left), Self::IsolatedRoot(right)) => {
                left.polynomial() == right.polynomial() && left.root_index() == right.root_index()
            }
            _ => false,
        }
    }
}
