// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What resolution promises: which requests a look-ahead may start, how far
//! ahead of the measurement it runs, and which failures are the request's own.

use std::collections::BTreeSet;

use super::*;
use crate::compile::check_bounds;
use recipe::{
    Arg, Blob, CompileRequests, Geometry, ImagePersistence, KernelLaunch, ModuleImage,
    PtxAssemblyMode, PtxCompileOptions, RecipeSignature, SourceImage, SourceKind,
};

fn request(max_registers: Option<u32>) -> CompileRequest {
    CompileRequest {
        sources: vec![SourceImage {
            kind: SourceKind::Ptx,
            target: recipe::PtxTarget {
                isa: [8, 8],
                minimum_sm: 100,
            },
            blob: Blob::from_vec(b"ptx".to_vec()),
        }],
        options: PtxCompileOptions {
            max_registers,
            assembly_mode: PtxAssemblyMode::Relocatable,
        },
        requires: Default::default(),
    }
}

/// A body of one module resolving `request`, launched as `entry`, with
/// `nested` hung off it as a choice.
/// The one word `launching` writes through output 0.
const OUTPUT_NBYTES: usize = 4;

fn launching(request: usize, entry: &str, nested: Vec<RecipeBody>) -> RecipeBody {
    let mut nodes = vec![recipe::Node::new(
        Work::Call {
            call: KernelLaunch {
                module: 0,
                entry: entry.into(),
                args: vec![Arg::output(0)],
            },
            geometry: vec![Geometry {
                grid: [1, 1, 1],
                block: [1, 1, 1],
                dynamic_shmem: 0,
            }],
        },
        [],
    )];
    if !nested.is_empty() {
        nodes.push(recipe::Node::new(
            Work::Choice {
                candidates: nested,
                input_binding: Vec::new(),
                output_binding: vec![recipe::BufRef::Output(0)],
            },
            [],
        ));
    }
    RecipeBody::new(
        0,
        1,
        vec![KernelModule::from_request(
            recipe::CompileRequestId(request),
            Vec::new(),
            ResourceBounds::default(),
        )],
        Vec::new(),
        nodes,
    )
}

fn request_over(source: &[u8]) -> CompileRequest {
    CompileRequest {
        sources: vec![SourceImage {
            blob: Blob::from_vec(source.to_vec()),
            ..request(None).sources.remove(0)
        }],
        options: PtxCompileOptions {
            max_registers: None,
            assembly_mode: PtxAssemblyMode::Relocatable,
        },
        requires: Default::default(),
    }
}

fn resolved_request() -> Arc<ResolvedRequest> {
    Arc::new(ResolvedRequest {
        image: ModuleImage::CudaLoadable {
            bytes: Blob::from_vec(b"image".to_vec()),
            persistence: ImagePersistence::Persistable,
            provenance: None,
        },
        resources: HashMap::new(),
        production_cost_ms: 7,
        compile_wall_ms: 7,
        resource_ms: 0,
        cache_source: None,
        assembler: "test".into(),
    })
}

fn resolver_over(recipe: &Recipe) -> Resolver<'_, recipe::Executable> {
    Resolver::new(
        std::ptr::null_mut(),
        100,
        RealizerOptions::default(),
        recipe,
    )
}

fn failure(kind: CompileFailureKind, message: &str) -> CompileFailure {
    CompileFailure {
        kind,
        message: message.into(),
    }
}

/// Look-ahead is speculative in *when*, never in *what*. What a body will
/// certainly resolve is its own modules plus, through each nested choice's
/// first candidate, the whole default selection the warm-up run executes.
/// A losing sibling — at any depth — is reached only when the traversal
/// gets there, which is what keeps a pruned subtree uncompiled.
#[test]
fn priming_reaches_nested_defaults_and_no_losing_sibling() {
    let body = launching(
        0,
        "root",
        vec![
            launching(
                1,
                "winner",
                vec![
                    launching(3, "deep_first", Vec::new()),
                    launching(4, "deep_other", Vec::new()),
                ],
            ),
            launching(2, "loser", Vec::new()),
        ],
    );
    let mut primed = Vec::new();
    certain_requests(&body, &mut primed);
    assert_eq!(
        primed,
        vec![0, 1, 3],
        "the body's own request, the first candidate's, and its first \
         nested candidate's — and nothing else"
    );
    assert!(!primed.contains(&2), "a losing sibling is not primed");
    assert!(
        !primed.contains(&4),
        "a losing sibling inside the default candidate is not primed either"
    );
}

/// A module that already carries a usable image needs no compile, so
/// priming asks for nothing on its behalf.
#[test]
fn priming_skips_a_module_that_already_has_an_image() {
    let (inputs, outputs, mut modules, workspace, nodes) =
        launching(0, "root", Vec::new()).into_parts();
    modules[0].code = recipe::KernelCode::PtxBacked {
        compile_request: recipe::CompileRequestId(0),
        images: vec![ModuleImage::CudaLoadable {
            bytes: Blob::from_vec(vec![1, 2, 3]),
            persistence: ImagePersistence::Persistable,
            provenance: None,
        }],
    };
    let body = RecipeBody::new(inputs, outputs, modules, workspace, nodes);
    let mut primed = Vec::new();
    certain_requests(&body, &mut primed);
    assert!(primed.is_empty());
}

#[test]
fn the_prefetch_window_walks_the_traversal_one_step_ahead() {
    let order = [0, 1, 2, 3, 4];
    assert_eq!(prefetch_at(&order, 0, 3), &[0, 1, 2]);
    assert_eq!(prefetch_at(&order, 1, 3), &[3]);
    assert_eq!(prefetch_at(&order, 2, 3), &[4]);
    assert_eq!(prefetch_at(&order, 3, 3), &[] as &[usize]);
    assert_eq!(prefetch_at(&order, 4, 3), &[] as &[usize]);

    let started: Vec<usize> = (0..order.len())
        .flat_map(|position| prefetch_at(&order, position, 3).to_vec())
        .collect();
    assert_eq!(started, order);

    for position in 0..order.len() {
        assert_eq!(
            prefetch_at(&order, position, 1),
            &order[position..position + 1]
        );
    }
    assert_eq!(prefetch_at(&order, 0, 4), &[0, 1, 2, 3]);
    assert_eq!(prefetch_at(&order, 0, 0), &[] as &[usize]);

    let resumed = [1, 3, 4];
    assert_eq!(prefetch_at(&resumed, 0, 2), &[1, 3]);
    assert_eq!(prefetch_at(&resumed, 1, 2), &[4]);
}

/// Only a failure the request itself causes is an answer about the
/// request. A machine failure is remembered as a cause and forgotten as a
/// verdict.
#[test]
fn a_machine_failure_defers_where_a_program_failure_settles() {
    let pending: Recipe = Recipe::stating_edges(
        RecipeSignature::empty(),
        CompileRequests::new(vec![request(None), request(Some(64))]),
        RecipeBody::default(),
    )
    .unwrap();
    let mut resolver = resolver_over(&pending);

    resolver.fail(
        0,
        failure(CompileFailureKind::CompilerDiagnostic, "bad ptx"),
    );
    resolver.fail(1, failure(CompileFailureKind::Interrupted, "worker died"));
    assert!(matches!(resolver.states[0], RequestState::Failed(_)));
    assert!(matches!(resolver.states[1], RequestState::Deferred(_)));

    // A deferral is still reported: the realization was left with it.
    let report = resolver.finish();
    assert_eq!(report.requests.len(), 2);
    assert_eq!(report.requests[0].outcome, RequestOutcome::Failed);
    assert_eq!(report.requests[1].outcome, RequestOutcome::Deferred);
}

/// A retried request keeps one row, holding where it ended up.
#[test]
fn a_retried_request_reports_one_row() {
    let pending: Recipe = Recipe::stating_edges(
        RecipeSignature::empty(),
        CompileRequests::new(vec![request(None)]),
        RecipeBody::default(),
    )
    .unwrap();
    let mut resolver = resolver_over(&pending);
    resolver.fail(0, failure(CompileFailureKind::Interrupted, "first"));
    resolver.fail(
        0,
        failure(CompileFailureKind::UnavailableToolchain, "second"),
    );
    let report = resolver.finish();
    assert_eq!(report.requests.len(), 1);
    assert_eq!(report.requests[0].outcome, RequestOutcome::Deferred);
    assert_eq!(
        report.requests[0]
            .failure
            .as_ref()
            .map(|f| f.message.as_str()),
        Some("second")
    );
}

/// The retry rule, from both sides. A settled failure is the answer and is
/// never re-attempted; a deferred one is re-attempted by the next demand,
/// and that demand makes exactly one attempt — it returns whatever the
/// attempt left rather than looping on it.
#[test]
fn a_deferred_request_is_re_attempted_and_a_settled_one_is_not() {
    let pending: Recipe = Recipe::stating_edges(
        RecipeSignature::empty(),
        CompileRequests::new(vec![request(None)]),
        RecipeBody::default(),
    )
    .unwrap();

    let mut resolver = resolver_over(&pending);
    resolver.states[0] = RequestState::Failed(failure(
        CompileFailureKind::CompilerDiagnostic,
        "settled-sentinel",
    ));
    let answered = resolver
        .resolve_request(0)
        .err()
        .expect("a settled failure is returned, not re-attempted");
    assert_eq!(
        answered.message, "settled-sentinel",
        "a deterministic failure is the request's answer, not a reason to compile again"
    );

    let mut resolver = resolver_over(&pending);
    resolver.states[0] = RequestState::Deferred(failure(
        CompileFailureKind::Interrupted,
        "deferred-sentinel",
    ));
    // This really re-attempts: whatever this machine has (or lacks) decides
    // the outcome, but the stale cause cannot be it.
    let outcome = resolver.resolve_request(0);
    assert!(
        outcome
            .as_ref()
            .err()
            .is_none_or(|failure| failure.message != "deferred-sentinel"),
        "the demand must attempt the request rather than repeat its last cause"
    );
    // One demand, one attempt: the request is settled either way, never
    // left mid-flight for this call to spin on.
    assert!(
        matches!(
            resolver.states[0],
            RequestState::Ready(_) | RequestState::Failed(_) | RequestState::Deferred(_)
        ),
        "a demand leaves the request with an outcome"
    );
}

/// The gate a capped request states about its base is a statement about
/// the base *image*. When no launch names that base, kiln has no entry to
/// measure — and a gate it cannot evaluate must not read as satisfied.
#[test]
fn a_base_no_launch_names_offers_nothing_to_measure() {
    let mut capped = request(Some(128));
    capped.require(
        recipe::CompileRequestId(0),
        ResourceBounds {
            max_local_bytes: Some(0),
            ..ResourceBounds::default()
        },
    );
    // Only the capped request has a module; the base exists solely as the
    // prerequisite that gates it.
    let pending: Recipe = Recipe::stating_edges(
        // The launch writes output 0 and reads nothing.
        RecipeSignature::new(Vec::new(), vec![OUTPUT_NBYTES]),
        CompileRequests::new(vec![request(None), capped]),
        launching(1, "loom_a", Vec::new()),
    )
    .unwrap();
    let resolver = resolver_over(&pending);
    assert_eq!(resolver.entries[1], BTreeSet::from(["loom_a".to_string()]));
    assert!(
        resolver.entries[0].is_empty(),
        "no launch names the base, so it exposes no entry"
    );

    let bounds = ResourceBounds {
        max_local_bytes: Some(0),
        ..ResourceBounds::default()
    };
    let measured = HashMap::new();
    assert!(
        check_bounds(bounds, measured.values()).is_ok(),
        "the old check passed this gate vacuously"
    );
    assert!(
        check_measured_bounds(bounds, &resolver.entries[0], &measured).is_err(),
        "the gate cannot be evaluated, so it is not satisfied"
    );
}

/// The no-recompile promise, at the scope where it actually holds: one
/// resolver, which is one realization.
///
/// A settled request is answered from the side table — the image it already
/// produced is handed over — and a second request naming the same compile
/// becomes an alias of the first rather than a second compile. Neither
/// survives the resolver: `Realizer::realize` builds a new one per call, so a
/// *resumed* realization re-derives every image from scratch and it is
/// `cuda_toolchain`'s content-addressed compile cache, not this table, that
/// keeps the work from being redone.
#[test]
fn one_compile_answers_every_demand_that_names_it_in_this_realization() {
    let pending: Recipe = Recipe::stating_edges(
        RecipeSignature::empty(),
        CompileRequests::new(vec![request_over(b"ptx-a"), request_over(b"ptx-a")]),
        RecipeBody::default(),
    )
    .unwrap();
    let mut resolver = resolver_over(&pending);

    // A request that already produced an image is never scheduled again: the
    // same image comes back, and nothing goes to the compile pool.
    let ready = resolved_request();
    resolver.states[0] = RequestState::Ready(ready.clone());
    for _ in 0..3 {
        let answered = resolver
            .resolve_request(0)
            .expect("a settled request answers");
        assert!(
            Arc::ptr_eq(&answered, &ready),
            "the demand was answered from the side table, not from a fresh compile"
        );
    }
    resolver.schedule_request(0, Reach::Available);
    assert!(matches!(resolver.states[0], RequestState::Ready(_)));
    assert_eq!(
        resolver.outstanding_source_bytes(),
        0,
        "a completed request puts nothing back on the compile pool"
    );

    // Two requests over one source and one set of options are one compile.
    // The second names the first rather than starting its own.
    let mut resolver = resolver_over(&pending);
    resolver.schedule_request(0, Reach::Available);
    resolver.schedule_request(1, Reach::Available);
    assert!(matches!(resolver.states[0], RequestState::Pending(_)));
    assert!(
        matches!(resolver.states[1], RequestState::Alias(0)),
        "a request whose compile identity is already owned aliases its owner"
    );
    assert_eq!(
        resolver.outstanding_source_bytes(),
        pending.context().as_slice()[0].sources[0].blob.len(),
        "one source is outstanding, not two"
    );
}

/// What the report says about outstanding compile work has to be what was
/// outstanding.
///
/// Look-ahead starts compiles the traversal may never reach — that is the
/// point of it — so the spawn and the claim are not paired, and a counter
/// incremented at one and decremented at the other keeps the bytes of every
/// unclaimed compile until the realization ends. Reading the outstanding set
/// off the request states instead makes the peak a real peak.
#[test]
fn an_unclaimed_look_ahead_does_not_inflate_the_outstanding_peak() {
    let pending: Recipe = Recipe::stating_edges(
        RecipeSignature::empty(),
        CompileRequests::new(vec![request_over(b"ptx-aaaaaaaa"), request_over(b"ptx-b")]),
        RecipeBody::default(),
    )
    .unwrap();
    let (larger, smaller) = (
        pending.context().as_slice()[0].sources[0].blob.len(),
        pending.context().as_slice()[1].sources[0].blob.len(),
    );
    assert!(smaller < larger);

    let mut resolver = resolver_over(&pending);
    resolver.schedule_request(0, Reach::Available);
    assert_eq!(resolver.outstanding_source_bytes(), larger);

    // The traversal never reaches this one; the compile it started is left
    // where it stands, exactly as a pruned subtree or an expired deadline
    // leaves it. Nothing claims it, so nothing pairs with its spawn.
    resolver.states[0] =
        RequestState::Failed(failure(CompileFailureKind::InvalidImage, "unclaimed"));

    resolver.schedule_request(1, Reach::Available);
    assert_eq!(
        resolver.outstanding_source_bytes(),
        smaller,
        "only the second compile is outstanding now"
    );
    assert_eq!(
        resolver.finish().peak_queued_source_bytes,
        larger,
        "the peak is the most that was ever outstanding at one instant, \
         not the sum of everything ever started"
    );
}

#[test]
fn request_inventory_crosses_nested_choices() {
    let module = || {
        KernelModule::from_request(
            recipe::CompileRequestId(0),
            Vec::new(),
            ResourceBounds::default(),
        )
    };
    let candidate: RecipeBody = RecipeBody::new(
        0,
        0,
        vec![module()],
        Vec::new(),
        vec![recipe::Node::new(
            Work::Call {
                call: KernelLaunch {
                    module: 0,
                    entry: "nested".into(),
                    args: vec![Arg::output(0)],
                },
                geometry: vec![Geometry {
                    grid: [1, 1, 1],
                    block: [1, 1, 1],
                    dynamic_shmem: 0,
                }],
            },
            [],
        )],
    );
    let body = RecipeBody::new(
        0,
        0,
        vec![module()],
        Vec::new(),
        vec![
            recipe::Node::new(
                Work::Call {
                    call: KernelLaunch {
                        module: 0,
                        entry: "root".into(),
                        args: vec![Arg::output(0)],
                    },
                    geometry: vec![Geometry {
                        grid: [1, 1, 1],
                        block: [1, 1, 1],
                        dynamic_shmem: 0,
                    }],
                },
                [],
            ),
            recipe::Node::new(
                Work::Choice {
                    candidates: vec![candidate],
                    input_binding: Vec::new(),
                    output_binding: vec![recipe::BufRef::Output(0)],
                },
                [],
            ),
        ],
    );
    let mut entries = vec![BTreeSet::new()];
    collect_request_entries(&body, &mut entries);
    assert_eq!(
        entries[0],
        BTreeSet::from(["nested".to_string(), "root".to_string()])
    );
}

/// An image crosses from one request to another only when their sources are
/// the same bytes.
///
/// Two requests land in one bucket by claiming the same sources, and an
/// encoded blob's claim is unchecked until it is expanded. Aliasing on that
/// claim would run one request's compiled code for another request's bytes —
/// bytes nothing had read. So the forged request is refused instead, while an
/// honest twin still aliases and still costs one compile.
#[test]
fn a_forged_request_cannot_adopt_another_s_compile() {
    use crate::compile::forgery;

    let honest = forgery::recipe_over(vec![
        vec![forgery::source(forgery::CALLER)],
        vec![forgery::source(forgery::CALLEE)],
    ]);
    let forged = forgery::claiming_another_digest(&honest, 0, 1);
    assert_eq!(
        forged.context().as_slice()[0].sources[0]
            .blob
            .claimed_digest()
            .unwrap(),
        forged.context().as_slice()[1].sources[0]
            .blob
            .claimed_digest()
            .unwrap(),
        "the second request now claims the first one's source"
    );
    assert_eq!(
        forged.context().as_slice()[1].sources[0]
            .blob
            .content_digest(),
        Err(recipe::BlobDecodeError::Digest),
        "and holds bytes that are not it"
    );

    let mut resolver = resolver_over(&forged);
    resolver.schedule_request(0, Reach::Available);
    resolver.schedule_request(1, Reach::Available);
    assert!(matches!(resolver.states[0], RequestState::Pending(_)));
    assert!(
        !matches!(resolver.states[1], RequestState::Alias(_)),
        "a claimed digest is not an image to hand over"
    );
    assert!(
        matches!(
            &resolver.states[1],
            RequestState::Failed(failure) if failure.kind == CompileFailureKind::CorruptSource
        ),
        "bytes that are not what they say are refused, not substituted"
    );

    // The same shape without the forgery: one compile, and the second request
    // names it.
    let twins = forgery::recipe_over(vec![
        vec![forgery::source(forgery::CALLER)],
        vec![forgery::source(forgery::CALLER)],
    ]);
    let mut resolver = resolver_over(&twins);
    resolver.schedule_request(0, Reach::Available);
    resolver.schedule_request(1, Reach::Available);
    assert!(matches!(resolver.states[1], RequestState::Alias(0)));
}

struct CompileCapture(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Drop for CompileCapture {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

fn one_request_recipe() -> Recipe {
    Recipe::stating_edges(
        RecipeSignature::empty(),
        CompileRequests::new(vec![request(None)]),
        RecipeBody::default(),
    )
    .unwrap()
}

#[test]
fn resolver_finish_waits_for_running_compile_cleanup() {
    let recipe = one_request_recipe();
    let capture_dropped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut resolver = resolver_over(&recipe);
    let capture = CompileCapture(std::sync::Arc::clone(&capture_dropped));
    let (entered, started) = std::sync::mpsc::sync_channel(1);
    let (release, wait) = std::sync::mpsc::sync_channel(1);
    let task = cuda_toolchain::spawn_compile(move || -> Result<CompileProduct, CompileFailure> {
        let _capture = capture;
        entered.send(()).unwrap();
        wait.recv_timeout(std::time::Duration::from_secs(3))
            .unwrap();
        Err(failure(CompileFailureKind::Interrupted, "sentinel"))
    });
    resolver.states[0] = RequestState::Pending(task);
    started
        .recv_timeout(std::time::Duration::from_secs(3))
        .unwrap();
    let controller = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(30));
        release.send(()).unwrap();
    });

    let report = resolver.finish();
    controller.join().unwrap();
    assert!(capture_dropped.load(std::sync::atomic::Ordering::SeqCst));
    assert!(report.compile_blocked_ms >= 20);
}

#[test]
fn resolver_drop_retires_compile_capture() {
    let recipe = one_request_recipe();
    let capture_dropped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let mut resolver = resolver_over(&recipe);
        let capture = CompileCapture(std::sync::Arc::clone(&capture_dropped));
        let task =
            cuda_toolchain::spawn_compile(move || -> Result<CompileProduct, CompileFailure> {
                let _capture = capture;
                Err(failure(CompileFailureKind::Interrupted, "sentinel"))
            });
        resolver.states[0] = RequestState::Pending(task);
    }
    assert!(capture_dropped.load(std::sync::atomic::Ordering::SeqCst));
}
