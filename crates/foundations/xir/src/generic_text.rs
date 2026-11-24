// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical generic text for XIR's [`weave`] representation.

use weave::{Diagnostic, Ir};

use crate::dialect::{ConversionError, registry, to_ir};
use crate::ir::Module;

pub fn parse(source: &str) -> Result<Ir, Vec<Diagnostic>> {
    weave::parse(source, &registry())
}

pub fn print(module: &Module) -> Result<String, ConversionError> {
    print_ir(&to_ir(module)?).map_err(|error| ConversionError::Print(error.to_string()))
}

pub fn print_ir(ir: &Ir) -> Result<String, Diagnostic> {
    weave::print(ir)
}
