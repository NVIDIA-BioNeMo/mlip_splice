// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    BackendSettings, CompileOptions, ElementOrder, LiftedProgram, LoomFacts, num_elements,
    program_order_summaries,
};

pub(super) fn compile_passes(
    loom: &LiftedProgram,
    options: CompileOptions,
    settings: &BackendSettings,
) -> Result<Vec<(usize, bool, ElementOrder)>, String> {
    let statement_cap = settings.partition.max_kernel_statements;
    let fine_cap = statement_cap / settings.compiler.fine_cap_divisor;
    let facts = LoomFacts::analyze(loom);
    let mut passes = vec![(statement_cap, true)];
    if (facts.estimated_statements() > statement_cap as u64
        || facts.register_pressure() > settings.schedule.default_register_cap as u64)
        && fine_cap >= settings.compiler.fine_cap_floor
    {
        passes.push((fine_cap, false));
    }
    let elements = loom
        .outputs()
        .iter()
        .map(|binding| num_elements(&binding.element_shape))
        .sum::<Result<usize, String>>()?;
    let orders = match options.only_element_order {
        Some(order) => vec![order],
        None if elements > 1 => vec![ElementOrder::Trace, ElementOrder::Liveness],
        None => vec![ElementOrder::Trace],
    };
    Ok(passes
        .into_iter()
        .flat_map(|(cap, clusters)| orders.iter().map(move |&order| (cap, clusters, order)))
        .collect())
}

/// Both element orders as they look before either reaches the compiler, with
/// what describing them cost. A program whose symbolic profile does not expand
/// has no summary to give and the record stays empty.
pub(super) fn describe_orders(loom: &LiftedProgram) -> Vec<crate::profile::ElementOrderSummary> {
    let started = std::time::Instant::now();
    let Ok(summaries) = program_order_summaries(loom) else {
        return Vec::new();
    };
    let decision_ms = started.elapsed().as_secs_f64() * 1_000.0 / summaries.len().max(1) as f64;
    summaries
        .into_iter()
        .map(|(order, summary)| crate::profile::ElementOrderSummary {
            graph: 0,
            order: order.name(),
            decision_ms,
            summary: summary.fields(),
        })
        .collect()
}
