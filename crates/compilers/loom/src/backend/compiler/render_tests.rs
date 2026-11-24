// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::backend::compiler::{CompileLimits, CompileProfile, ElementOrder, Placement};

fn with_render(test: impl FnOnce(Render<'_>)) {
    let settings = BackendSettings::default();
    let limits = CompileLimits {
        statement_cap: usize::MAX,
        register_cap: usize::MAX,
        cut_limit: 0,
        plan_clusters: false,
        max_workspace_bytes: None,
        element_order: ElementOrder::Trace,
        target_sm: 100,
        dump_ptx: false,
        record_structure: false,
        settings: &settings,
    };
    let planned = crate::backend::compiler::pipeline::plan_single_with_limits_profiled(
        crate::backend::compiler::tests::guarded_program(),
        limits,
        Placement::Default,
        &mut CompileProfile::default(),
    )
    .unwrap();
    let table = SlotTable::new(&settings.compiler.block_sizes, &settings.compiler);
    test(Render {
        loom: &planned.loom,
        execution: &planned.execution,
        incidence: &planned.incidence,
        settings: &settings,
        table: &table,
        sm: 100,
        statement_cap: usize::MAX,
        required: true,
    });
}

#[test]
fn only_the_selected_source_is_rendered_and_aliases_share_it() {
    with_render(|render| {
        let mut cache = RenderCache::default();
        assert!(cache.sources.is_empty());
        let baseline = render
            .slot_alias(&mut cache, 0, 0, 0, &mut || Ok(()))
            .unwrap();
        assert_eq!(cache.sources.len(), 1);
        for flavor in [CandidateFlavor::RegCap, CandidateFlavor::WholeProgram] {
            let slot = render
                .table
                .specs
                .iter()
                .position(|spec| {
                    spec.flavor == flavor && spec.block_size == render.table.specs[0].block_size
                })
                .unwrap();
            let alias = render
                .slot_alias(&mut cache, 0, 0, slot, &mut || Ok(()))
                .unwrap();
            assert!(std::sync::Arc::ptr_eq(&baseline.source, &alias.source));
            assert_eq!(cache.sources.len(), 1);
        }
        let next = render
            .slot_alias(&mut cache, 0, 0, 1, &mut || Ok(()))
            .unwrap();
        assert!(!std::sync::Arc::ptr_eq(&baseline.source, &next.source));
        assert_eq!(cache.sources.len(), 2);
    });
}

#[test]
fn cancellation_does_not_become_a_source_refusal() {
    with_render(|render| {
        for stop in [1, 2, 3] {
            let mut cache = RenderCache::default();
            let mut checkpoints = 0;
            let result = render.slot_alias(&mut cache, 0, 0, 0, &mut || {
                checkpoints += 1;
                if checkpoints == stop {
                    Err("cancelled".into())
                } else {
                    Ok(())
                }
            });
            assert!(matches!(result, Err(error) if error == "cancelled"));
            assert_eq!(cache.sources.len(), usize::from(stop == 3));
            assert!(cache.sources.values().all(Result::is_ok));
            let saved = cache
                .sources
                .values()
                .next()
                .map(|source| source.as_ref().unwrap().source.clone());
            let alias = render
                .slot_alias(&mut cache, 0, 0, 0, &mut || Ok(()))
                .unwrap();
            assert_eq!(cache.sources.len(), 1);
            if let Some(saved) = saved {
                assert!(std::sync::Arc::ptr_eq(&saved, &alias.source));
            }
        }
    });
}

#[test]
fn an_emitter_refusal_is_memoized_for_its_exact_geometry() {
    with_render(|render| {
        let mut cache = RenderCache::default();
        let slot = render
            .table
            .specs
            .iter()
            .position(|spec| spec.io_lanes == 8)
            .unwrap();
        let refusal = render
            .slot_alias(&mut cache, 0, 0, slot, &mut || Ok(()))
            .err()
            .unwrap();
        assert_eq!(cache.sources.len(), 1);
        assert!(cache.sources.values().all(Result::is_err));
        let mut checkpoints = 0;
        assert_eq!(
            render
                .slot_alias(&mut cache, 0, 0, slot, &mut || {
                    checkpoints += 1;
                    Ok(())
                })
                .err()
                .unwrap(),
            refusal
        );
        assert_eq!(checkpoints, 2);
        assert!(
            render
                .slot_alias(&mut cache, 0, 0, 0, &mut || Ok(()))
                .is_ok()
        );
        assert_eq!(cache.sources.len(), 2);
    });
}
