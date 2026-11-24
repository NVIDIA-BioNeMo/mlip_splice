// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{BackendSettings, facts::LoomFacts};
use crate::{CompileOptions, LiftedProgram};
use std::num::NonZeroUsize;

const AMPERE_SM: u16 = 80;

pub(super) fn configure_backend_settings(
    loom: &LiftedProgram,
    options: CompileOptions,
) -> Result<BackendSettings, String> {
    let mut settings = BackendSettings::default();
    settings.partition.max_kernel_statements = statement_cap_for_target(
        &settings,
        super::render_target(options)?,
        LoomFacts::analyze(loom).register_pressure(),
        options.max_kernel_statements.map(NonZeroUsize::get),
    );
    Ok(settings)
}

fn statement_cap_for_target(
    settings: &BackendSettings,
    target_sm: u16,
    register_pressure: u64,
    explicit_statement_cap: Option<usize>,
) -> usize {
    let default_cap = settings.partition.max_kernel_statements;
    if let Some(statement_cap) = explicit_statement_cap {
        return statement_cap;
    }
    let cut_capacity = settings
        .schedule
        .default_register_cap
        .saturating_mul(settings.schedule.default_cut_limit) as u64;
    if target_sm == AMPERE_SM && register_pressure > cut_capacity {
        (default_cap / 2).max(1)
    } else {
        default_cap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_the_ampere_default_only_above_default_cut_capacity() {
        let mut settings = BackendSettings::default();
        settings.partition.max_kernel_statements = 64;
        settings.schedule.default_register_cap = 4;
        settings.schedule.default_cut_limit = 8;

        assert_eq!(statement_cap_for_target(&settings, 80, 33, None), 32);
        assert_eq!(statement_cap_for_target(&settings, 80, 32, None), 64);
        assert_eq!(statement_cap_for_target(&settings, 100, 33, None), 64);
    }

    #[test]
    fn keeps_an_explicit_statement_cap() {
        let settings = BackendSettings::default();
        assert_eq!(
            statement_cap_for_target(&settings, 80, u64::MAX, Some(123)),
            123
        );
    }
}
