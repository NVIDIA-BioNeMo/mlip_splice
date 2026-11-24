// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the fold by key: what a ladder offers, what a copy's run is
//! bounded by, and what the projection kernels render.

use domain_index::{Domain, Graph};

use super::*;

struct Index {
    graph: Graph,
}

impl DomainIndex for Index {
    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn num_axes(&self) -> usize {
        1
    }

    fn iter_domain(&self, _axis: usize) -> i32 {
        0
    }

    fn num_operands(&self) -> usize {
        3
    }

    fn operand_domain(&self, operand: usize, _axis: usize) -> i32 {
        if operand == 2 { -1 } else { 0 }
    }
}

/// One colliding-key fold over `collisions` values per address and
/// `elements` values in one cell — the shape a broadcast output takes.
fn colliding(collisions: i64, elements: usize) -> OutputFold {
    OutputFold::new(
        Key::Colliding,
        Scalar::F32,
        Scalar::F32,
        false,
        Collisions::Mean(collisions),
        elements as i64,
        elements,
    )
    .unwrap()
}

fn entries() -> ProjectionEntries {
    ProjectionEntries {
        convert: "convert".into(),
        fold: "fold".into(),
    }
}

fn ptx(fold: &OutputFold, elements: usize) -> String {
    let kernels = fold.projection_kernels(&entries(), elements).unwrap();
    gpu_ir::ptx::render_ptx_module(&kernels, 100)
}

#[test]
fn narrow_atomic_outputs_use_word_sized_parameters() {
    let index = Index {
        graph: Graph {
            domains: vec![Domain { extent: 4 }],
            index_maps: Vec::new(),
            aligned_views: Vec::new(),
        },
    };
    let folds = Folds::new(
        &index,
        1,
        &[Scalar::S8, Scalar::S8, Scalar::U16],
        &[Scalar::S8, Scalar::U16],
        &[false, false],
        &[4, 4],
    )
    .unwrap();
    assert_eq!(
        folds.parameter_scalars(),
        &[Scalar::S8, Scalar::S8, Scalar::U32]
    );
    assert_eq!(folds.output(0).key(), Key::Unique);
    assert_eq!(folds.output(1).key(), Key::Colliding);
    assert!(folds.output(1).projection(1).is_some());
    assert_eq!(folds.output(1).accumulation_nbytes(7, 1).unwrap(), 28);
    assert_eq!(folds.output(1).accumulation_nbytes(7, 4).unwrap(), 112);
}

#[test]
fn a_unique_key_stores_and_offers_one_bucket() {
    let fold = OutputFold::new(
        Key::Unique,
        Scalar::F32,
        Scalar::F32,
        false,
        Collisions::Mean(1_000_000),
        64,
        64,
    )
    .unwrap();
    assert!(!fold.combines_atomically());
    assert_eq!(fold.bucket_counts(), &[1]);
    assert_eq!(fold.bucket_cap(), BucketCap::UniqueKey);
    assert!(!fold.selects_buckets());
    assert!(fold.projection(1).is_none());
}

/// A run short enough to round cleanly leaves the throughput rule in charge:
/// the chain a bucket would split is the read-modify-writes one L2 sector
/// serialises, and below a launch's worth there is nothing to time.
#[test]
fn a_short_run_and_a_short_chain_offer_no_second_bucket() {
    let shallow = colliding(8, 8);
    assert_eq!(shallow.bucket_counts(), &[1]);
    assert_eq!(shallow.bucket_cap(), BucketCap::Chain(64));
}

/// A run past [`TARGET_ACCUMULATED_RUN`] is an accuracy matter, so the copies
/// that shorten it are offered whether or not the sector chain is long
/// enough to be worth a launch on its own.
#[test]
fn a_long_run_buckets_even_when_its_sector_chain_is_short() {
    let fold = colliding(1_000, 1);
    assert_eq!(fold.bucket_cap(), BucketCap::Collisions(1_000));
    assert_eq!(fold.bucket_counts(), &[16, 32, 256, 512]);
}

#[test]
fn a_colliding_key_offers_the_run_floor_and_up() {
    let fold = colliding(1_000, 4_096);
    assert!(fold.combines_atomically());
    assert!(fold.selects_buckets());
    let counts = fold.bucket_counts();
    // 1000 values over 16 copies leaves a run of 63, within the bound; over
    // 8 it would leave 125, so 8 is not offered and neither is 1.
    assert_eq!(counts, &[16, 32, 64]);
    assert!(counts.iter().all(|count| count.is_power_of_two()));
    assert!(counts.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(
        counts
            .iter()
            .all(|&count| 1_000i64.div_euclid(i64::from(count)) <= TARGET_ACCUMULATED_RUN)
    );
    assert_eq!(fold.bucket_cap(), BucketCap::Collisions(1_000));
}

/// A floor the workspace cannot reach is approached, never refused: two
/// copies are all this output affords, so two is what it offers.
#[test]
fn the_workspace_budget_bounds_a_wide_output() {
    let elements = BUCKET_WORKSPACE_BYTES / 4 / 3;
    let fold = colliding(1 << 20, elements);
    assert_eq!(fold.bucket_counts(), &[2]);
    assert_eq!(
        fold.bucket_cap(),
        BucketCap::Workspace(BUCKET_WORKSPACE_BYTES)
    );
    assert!(
        fold.accumulation_nbytes(elements, fold.max_buckets())
            .unwrap()
            <= BUCKET_WORKSPACE_BYTES
    );
    assert!(
        fold.selects_buckets(),
        "a lone count of two is still two copies to address"
    );
}

/// The figure a fold reports for one copy is an upper bound, so it counts
/// what a copy can receive, not what an even spread would give it. Only a
/// uniform fold — broadcast axes alone deciding the addresses — may cut the
/// count down to the values one address receives.
#[test]
fn a_copy_bounds_its_run_by_the_positions_it_receives() {
    let mean = colliding(1_000, 4_096);
    assert_eq!(mean.longest_run(10_000), 10_000);
    assert_eq!(mean.longest_run(625), 625);
    assert_eq!(mean.longest_run(0), 1);

    let uniform = OutputFold::new(
        Key::Colliding,
        Scalar::F32,
        Scalar::F32,
        false,
        Collisions::Uniform(1_000),
        4_096,
        4_096,
    )
    .unwrap();
    assert_eq!(uniform.longest_run(10_000), 1_000);
    assert_eq!(uniform.longest_run(625), 625);

    let stored = OutputFold::new(
        Key::Unique,
        Scalar::F32,
        Scalar::F32,
        false,
        Collisions::Uniform(1_000),
        4_096,
        4_096,
    )
    .unwrap();
    assert_eq!(stored.longest_run(10_000), 1, "a store folds nothing");
}

#[test]
fn one_bucket_needs_no_projection_when_the_dtypes_agree() {
    let fold = colliding(1_000, 4_096);
    assert!(fold.projection(1).is_none());
    assert_eq!(fold.projection(2), Some(Projection::Fold { buckets: 2 }));
    assert_eq!(
        Projection::Convert.launch(64, None).unwrap().0,
        [1, 1, 1],
        "one block covers 64 elements"
    );
}

#[test]
fn a_widened_accumulator_converts_before_it_reaches_storage() {
    let fold = OutputFold::new(
        Key::Colliding,
        Scalar::F16,
        Scalar::F32,
        false,
        Collisions::Mean(1),
        64,
        64,
    )
    .unwrap();
    assert_eq!(fold.parameter_scalar(), Scalar::F32);
    assert_eq!(fold.projection(1), Some(Projection::Convert));
    let ptx = ptx(&fold, 64);
    assert!(ptx.contains(".entry convert"));
    assert!(ptx.contains("cvt.rn.f16.f32"));
    assert!(ptx.contains("st.global.b16"));
}

#[test]
fn a_deferred_conversion_accumulates_in_the_math_dtype() {
    let fold = OutputFold::new(
        Key::Unique,
        Scalar::F16,
        Scalar::F32,
        true,
        Collisions::Mean(1),
        64,
        64,
    )
    .unwrap();
    assert_eq!(fold.parameter_scalar(), Scalar::F32);
    assert!(!fold.combines_atomically());
    assert_eq!(fold.projection(1), Some(Projection::Convert));
    assert!(ptx(&fold, 64).contains("cvt.rn.f16.f32"));
}

/// A unique key stores what the kernel computed, in the output's own
/// dtype: there is nothing left to convert, whatever the math dtype was.
#[test]
fn a_unique_key_that_stores_its_own_dtype_needs_no_projection() {
    for math in [Scalar::Pred, Scalar::S8] {
        let fold = OutputFold::new(
            Key::Unique,
            Scalar::U8,
            math,
            false,
            Collisions::Mean(1),
            64,
            64,
        )
        .unwrap();
        assert_eq!(fold.parameter_scalar(), Scalar::U8);
        assert_eq!(fold.projection(1), None, "math {math:?}");
    }
}

/// A colliding key widens a narrow integer so the atomic chain cannot wrap,
/// then converts straight back to the output's own dtype.
#[test]
fn a_widened_integer_converts_to_its_storage_dtype() {
    let direct = OutputFold::new(
        Key::Colliding,
        Scalar::S32,
        Scalar::S8,
        false,
        Collisions::Mean(1),
        64,
        64,
    )
    .unwrap();
    assert_eq!(direct.parameter_scalar(), Scalar::S32);
    assert!(direct.projection(1).is_none());

    let converted = OutputFold::new(
        Key::Colliding,
        Scalar::S16,
        Scalar::S8,
        false,
        Collisions::Mean(1),
        64,
        64,
    )
    .unwrap();
    let narrowed = ptx(&converted, 64);
    assert!(!narrowed.contains("cvt.s8.s32"));
    assert!(narrowed.contains("cvt.s16.s32"));
}

#[test]
fn the_fold_entry_takes_its_bucket_count_at_launch() {
    for elements in [1_472, 8] {
        let fold = colliding(100_000, elements);
        assert!(fold.max_buckets() > 1);
        let ptx = ptx(&fold, elements);
        assert!(ptx.contains(".entry fold"));
        assert!(
            ptx.contains("fold_buckets"),
            "the bucket count is a parameter, not a constant"
        );
    }
}

/// Tile candidates straddle 256 blocks, and every concrete tile determines
/// one launch grid without changing the compiled entry.
#[test]
fn fold_tiles_span_neighboring_launch_geometries() {
    let wide = colliding(1 << 20, 4_096);
    let projection = Projection::Fold { buckets: 4 };
    assert_eq!(projection.tile_widths(4_096), [8, 16, 32]);
    assert_eq!(projection.launch(4_096, Some(16)).unwrap().0, [256, 1, 1]);

    assert_eq!(projection.tile_widths(8), [1, 2]);
    assert_eq!(projection.launch(8, Some(1)).unwrap().0, [8, 1, 1]);
    let source = ptx(&wide, 4_096);
    assert!(source.contains("shfl.sync.down"));
    assert!(source.contains("fold_tile_log2"));
}

#[test]
fn the_bucket_offset_masks_the_block_index() {
    let mut b = gpu_ir::Builder::new();
    let mask = b.parameter(gpu_ir::Param {
        name: "mask".into(),
        ty: gpu_ir::ParamTy::Scalar(Scalar::S32),
    });
    let fold = colliding(100_000, 1_472);
    let offset = fold
        .bucket_offset(&mut b, mask.clone(), 1_472)
        .expect("a ladder of more than one bucket count selects");
    b.bind_value_as("probe", gpu_ir::Ty::scalar(Scalar::S64), offset);
    let f = b.function();
    assert!(
        b.owned()
            .iter()
            .any(|&op| matches!(f.op(op).kind, gpu_ir::OpKind::Bind))
    );
    assert!(f.value_named("probe").is_some());

    let lone = colliding(1, 64);
    assert!(lone.bucket_offset(&mut b, mask, 64).is_none());
}

#[test]
fn arms_apply_each_ladder_rung_to_every_output() {
    let index = Index {
        graph: Graph {
            domains: vec![Domain { extent: 1 << 16 }],
            index_maps: Vec::new(),
            aligned_views: Vec::new(),
        },
    };
    let folds = Folds::new(
        &index,
        1,
        &[Scalar::F32, Scalar::F32, Scalar::F32],
        &[Scalar::F32, Scalar::F32],
        &[false, false],
        &[1 << 16, 4_096],
    )
    .unwrap();

    assert_eq!(folds.bucket_selectors(), vec![1]);
    let arms = folds.bucket_arms();
    assert!(arms.len() > 1);
    assert_eq!(arms[0], vec![1, folds.output(1).bucket_counts()[0]]);
    for arm in &arms {
        assert_eq!(arm[0], 1, "a unique-key output never buckets");
    }
    assert_eq!(
        arms.last().unwrap()[1],
        folds.output(1).max_buckets(),
        "the last arm reaches the highest measured count"
    );
}

#[test]
fn arms_pair_outputs_by_sector_spreading_rung() {
    let output_128 = colliding(1 << 16, 128);
    let output_640 = colliding(1 << 16, 640);
    // Both outputs receive 65536 values per address, so both need 1024
    // copies to bring a run within the bound; the sector-spreading counts
    // below that are not offered.
    assert_eq!(output_128.bucket_counts(), &[1_024, 2_048]);
    assert_eq!(output_640.bucket_counts(), &[1_024, 2_048]);

    let folds = Folds {
        num_inputs: 0,
        parameter_scalars: Vec::new(),
        outputs: vec![output_128, output_640],
    };
    assert_eq!(
        folds.bucket_arms(),
        [vec![1_024, 1_024], vec![2_048, 2_048]]
    );
}

/// The per-block finalizer, rendered whole.
///
/// One block's warps deposit their sums and thread 0 folds them: the
/// combine writes the register the warp sum already sits in and the store
/// reads it there, so the branch produces nothing and no register joins the
/// two sides of it. A result nobody reads would put one back.
#[test]
fn the_block_leader_folds_the_partials_in_place_and_stores() {
    let fold = colliding(100_000, 1_472);
    let source = ptx(&fold, 1_472);
    let leader = [
        "    mov.u32 %r31, %tid.x;",
        "    setp.eq.s32 %p7, %r31, 0;",
        "    @!%p7 bra $Lelse_9;",
        "    ld.shared.f32 %f3, [%r1];",
        "    ld.shared.f32 %f10, [%r1+4];",
        "    add.rn.f32 %f3, %f3, %f10;",
        "    ld.shared.f32 %f11, [%r1+8];",
        "    add.rn.f32 %f3, %f3, %f11;",
        "    ld.shared.f32 %f12, [%r1+12];",
        "    add.rn.f32 %f3, %f3, %f12;",
        "    ld.shared.f32 %f13, [%r1+16];",
        "    add.rn.f32 %f3, %f3, %f13;",
        "    ld.shared.f32 %f14, [%r1+20];",
        "    add.rn.f32 %f3, %f3, %f14;",
        "    ld.shared.f32 %f15, [%r1+24];",
        "    add.rn.f32 %f3, %f3, %f15;",
        "    ld.shared.f32 %f16, [%r1+28];",
        "    add.rn.f32 %f3, %f3, %f16;",
        "    mad.lo.s64 %rd20, %rd14, 4, %rd4;",
        "    st.global.f32 [%rd20], %f3;",
        "$Lelse_9:",
    ]
    .join("\n");
    assert!(
        source.contains(&leader),
        "the block finalizer is not the text it was: {source}"
    );
}
