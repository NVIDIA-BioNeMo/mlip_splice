// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::backend::compiler::{ElementOrder, LiftedProgram};
use domain_index::{Domain, IndexMap};
use xir::{builder::ModuleBuilder, dtype::DType, shape::Shape};

fn folding_program() -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("folding_outputs");
    let x = builder.param(Shape::scalar(DType::F32));
    let y = builder.multiply(x, x);
    let z = builder.multiply(y, x);
    let main = builder.end_func(vec![x, y, z]);
    let binding = |domain| crate::FlatOperand {
        dtype: DType::F32,
        element_shape: vec![],
        lane_extent: 32,
        axis_domains: vec![domain],
    };
    crate::FlatProgram {
        name: "folding_outputs".into(),
        module: builder.finish(main).unwrap(),
        lane_count: 32,
        domains: vec![Domain { extent: 65536 }, Domain { extent: 16 }],
        index_maps: vec![IndexMap {
            source_id: 0,
            target_id: 1,
            is_injective: false,
            is_sorted: false,
            may_have_unmapped: false,
        }],
        num_axes: 1,
        operands: vec![binding(0), binding(0), binding(1), binding(1)],
        num_inputs: 1,
        available_views: vec![],
    }
    .try_into()
    .unwrap()
}

fn assert_complete(body: &recipe::RecipeBody) {
    for node in body.nodes() {
        if let recipe::Work::Choice { candidates, .. } = node.work() {
            assert_eq!(candidates.len(), 1);
            assert_complete(&candidates[0]);
        }
    }
}

#[test]
fn fold_widths_select_complete_leaves_without_new_source_or_requests() {
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
    let mut profile = CompileProfile::default();
    let planned = crate::backend::compiler::pipeline::plan_single_with_limits_profiled(
        folding_program(),
        limits,
        Placement::Default,
        &mut profile,
    )
    .unwrap();
    let mut producer =
        SingleCandidates::new(planned, limits, Placement::Default, true, &mut profile).unwrap();
    assert!(profile.kernels.is_empty());
    assert!(producer.modules.is_empty() && producer.projections.is_empty());
    let baseline = producer.planned.estimates.baseline_index.expect("baseline");
    let folds = &producer.planned.execution.domain.folds;
    assert!(
        folds.output(1).bucket_counts().len() > 1,
        "key={} collisions={} cap={:?} buckets={:?}",
        folds.output(1).key().name(),
        folds.output(1).collisions(),
        folds.output(1).bucket_cap(),
        folds.output(1).bucket_counts()
    );
    let direct_alternative = producer
        .configurations()
        .find(|configuration| {
            matches!(configuration, Configuration::Launches { candidate, .. } if *candidate != baseline)
        })
        .expect("direct alternative");
    let before = profile.fold_buckets.len();
    producer
        .materialize(direct_alternative, &mut profile, &mut || Ok(()))
        .unwrap();
    let reported = &profile.fold_buckets[before..];
    assert!(!reported.is_empty());
    assert!(
        reported
            .iter()
            .all(|record| record.buckets == 1 && record.admitted == vec![1])
    );
    let mut configurations = producer.configurations();
    let selection = configurations
        .by_ref()
        .find(|configuration| {
            matches!(configuration, Configuration::Launches { tile_widths, .. }
            if tile_widths.iter().skip(1).all(Option::is_some))
        })
        .expect("two fold projections");
    let Configuration::Launches {
        candidate,
        slots,
        buckets,
        tile_widths,
    } = selection.clone()
    else {
        unreachable!();
    };
    assert_eq!(tile_widths.len(), 3);
    assert!(tile_widths[0].is_none());
    let first = producer
        .materialize(selection, &mut profile, &mut || Ok(()))
        .unwrap();
    assert_complete(first.recipe.body());
    assert_eq!(producer.projections.len(), 2);
    assert_eq!(
        profile
            .kernels
            .iter()
            .filter(|unit| unit.flavor == "projection")
            .count(),
        2
    );
    let sources = profile.kernels.len();
    let requests = first.recipe.context().as_slice().len();
    let widths = buckets
        .iter()
        .enumerate()
        .map(|(output, &buckets)| {
            let projection = producer
                .planned
                .execution
                .domain
                .folds
                .output(output)
                .projection(buckets)?;
            let elements = reduction::output_elements(
                &producer.planned.execution.domain.index,
                &producer.planned.loom,
                output,
            )
            .unwrap();
            Some(
                projection
                    .tile_widths(elements)
                    .into_iter()
                    .find(|&width| Some(width) != tile_widths[output])
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let second = producer
        .materialize(
            Configuration::Launches {
                candidate,
                slots,
                buckets,
                tile_widths: widths.clone(),
            },
            &mut profile,
            &mut || Ok(()),
        )
        .unwrap();
    assert_complete(second.recipe.body());
    assert_eq!(profile.kernels.len(), sources);
    assert_eq!(second.recipe.context().as_slice().len(), requests);
    assert_eq!(
        profile
            .fold_buckets
            .iter()
            .rev()
            .take(3)
            .rev()
            .map(|record| record.tile_widths.clone())
            .collect::<Vec<_>>(),
        widths
            .into_iter()
            .map(|width| width.into_iter().collect::<Vec<_>>())
            .collect::<Vec<_>>()
    );
    let combined = crate::backend::compiler::combine_compiled(vec![first, second], 1).unwrap();
    assert_eq!(combined.recipe.context().as_slice().len(), requests);
    let recipe::Work::Choice { candidates, .. } = combined.recipe.body().nodes()[0].work() else {
        panic!("complete root choice");
    };
    assert_eq!(candidates.len(), 2);
    for candidate in candidates {
        assert_complete(candidate);
    }
    let continuation = configurations
        .clone()
        .take(30)
        .map(|configuration| format!("{configuration:?}"))
        .collect::<Vec<_>>();
    assert_eq!(
        continuation,
        configurations
            .take(30)
            .map(|configuration| format!("{configuration:?}"))
            .collect::<Vec<_>>()
    );
}

/// Spreading a folded output over `b` bucket copies gives each copy a `b`-th of
/// the launch's blocks, so the fan-in the figure reports falls with `b`, and pays
/// at most `b - 1` for the projection that sums the copies back. What the figure
/// must not do is fall faster than the blocks do: which positions share an
/// address is the index map's business, and this is an upper bound.
#[test]
fn bucket_copies_shorten_the_fan_in_the_figure_reports() {
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
    let mut profile = CompileProfile::default();
    let planned = crate::backend::compiler::pipeline::plan_single_with_limits_profiled(
        folding_program(),
        limits,
        Placement::Default,
        &mut profile,
    )
    .unwrap();
    let mut producer =
        SingleCandidates::new(planned, limits, Placement::Default, true, &mut profile).unwrap();
    let folded = 1;
    let counts = producer
        .planned
        .execution
        .domain
        .folds
        .output(folded)
        .bucket_counts()
        .to_vec();
    assert!(counts.len() > 1, "{counts:?} offers no split to compare");

    let mut figures = Vec::new();
    for configuration in producer.configurations() {
        let Configuration::Launches { buckets, .. } = &configuration else {
            continue;
        };
        let taken = buckets[folded];
        if figures.iter().any(|(count, _, _)| *count == taken) {
            continue;
        }
        let before = profile.addition_depths.len();
        if producer
            .materialize(configuration, &mut profile, &mut || Ok(()))
            .is_err()
        {
            continue;
        }
        let record = profile.addition_depths[before..]
            .iter()
            .find(|record| record.output == folded)
            .expect("one record per output");
        assert_eq!(record.buckets, taken);
        assert!(
            record.projection.unwrap_or(0) < u64::from(taken).max(1),
            "a fold over {taken} copies cannot add {:?} roundings",
            record.projection
        );
        // This output's addresses are picked by a non-injective map, so the
        // figure counts the positions a copy receives — at least one per block
        // the copy owns — and never the share an even spread would give it.
        assert!(
            record.across_blocks + 1 >= u64::try_from(record.blocks_per_bucket).unwrap(),
            "{} roundings under-counts {} blocks per copy",
            record.across_blocks,
            record.blocks_per_bucket
        );
        assert!(
            record
                .total
                .is_some_and(|total| total >= record.across_blocks),
            "the figure stays readable at {taken} copies"
        );
        figures.push((taken, record.across_blocks, record.total));
        if figures.len() == counts.len() {
            break;
        }
    }
    figures.sort_unstable();
    assert!(figures.len() > 1, "{figures:?} compares nothing");
    assert!(
        figures.first().unwrap().1 > figures.last().unwrap().1,
        "more copies shorten the fan-in: {figures:?}"
    );
    assert!(
        figures.windows(2).all(|pair| pair[0].1 >= pair[1].1),
        "more copies never deepen the fan-in: {figures:?}"
    );
}
