// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Giving every module one loadable image: which compile a request names, when
//! it is started, and whether what it produced may stand.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Instant;

use cuda_driver::{CUcontext, Context, Device, Module};
use cuda_toolchain::{CompileTask as OwnedCompileTask, KernelResources};
#[cfg(test)]
use recipe::CompileRequest;
use recipe::CompileRequests;
#[cfg(test)]
use recipe::Work;
use recipe::algebra::{Launches, ModuleTable};
use recipe::{
    Blob, CompileProvenance, FlatRecipe, KernelCode, KernelModule, ModuleImage, Recipe, RecipeBody,
    ResourceBounds,
};

use crate::Error;
use crate::compile::{
    CompileFailure, CompileFailureKind, CompileProduct, CompileTask, RequestKey,
    check_measured_bounds,
};
use crate::realize::{ArtifactPolicy, RealizerOptions};

mod structure;

#[cfg(test)]
use structure::certain_requests;
use structure::entries_by_module;
pub(crate) use structure::{
    certain_requests_of, collect_request_entries, hash_resolved_modules, prefetch_at,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestOutcome {
    Compiled,
    SourceLoaded,
    Reused,
    /// The request itself cannot produce an image, and would not on any
    /// machine. Nothing will retry it.
    Failed,
    /// The last attempt was stopped by this machine or this moment, not by the
    /// request. The next demand attempts it again; this is what the realization
    /// was left with.
    Deferred,
}

#[derive(Clone, Debug)]
pub struct RequestReport {
    pub request: usize,
    pub outcome: RequestOutcome,
    pub assembler: Option<String>,
    /// What producing this image cost, as reported by the compile cache. A
    /// cache hit reports what it cost when it was produced — historical work,
    /// not work this run did.
    pub production_cost_ms: Option<u64>,
    /// Wall clock this process spent on the request's compile task: a real
    /// assembly on a miss, a cache read on a hit.
    pub compile_wall_ms: Option<u64>,
    pub resource_ms: Option<u64>,
    pub cache_source: Option<String>,
    pub image_bytes: Option<usize>,
    pub resources: Vec<(String, KernelResources)>,
    pub failure: Option<CompileFailure>,
}

/// What resolution did, for the realization report to aggregate.
#[derive(Clone, Debug, Default)]
pub struct ResolutionReport {
    pub requests: Vec<RequestReport>,
    /// The most decoded source bytes any one instant of this realization had
    /// outstanding: compiles spawned and not yet claimed.
    ///
    /// A work measure, not a resident-memory one. An outstanding task holds a
    /// [`recipe::Blob`] handle on each of its unit's sources and copies
    /// nothing; the decoded sources exist only for the length of one `produce`
    /// call, and at most `workers` of those run at once. What this bounds is
    /// how much decoding and assembly the pool has been handed ahead of the
    /// traversal.
    pub peak_queued_source_bytes: usize,
    /// Wall clock the resolver spent waiting for required compile products or
    /// retiring already-submitted work at its ownership boundary. Against the
    /// summed `compile_wall_ms` it is what the compile pool's concurrency
    /// actually bought.
    pub compile_blocked_ms: u64,
}

/// A choice-free body whose every module carries exactly one loadable image,
/// paired with the pending compile request each module resolved from.
///
/// A body and not a recipe, because resolution does not decide what work is
/// bound through. The root's body and a candidate's are resolved the same way
/// and the candidate has no ports of its own — inventing some to hold it would
/// state that its slot `k` is the root's port `k`, which is what the binding
/// that selects it decides and not this.
///
/// The pairing travels with the value rather than living in the resolver: an
/// artifact policy re-attaches sources to *this* body's modules, so reading the
/// origins of some other resolution would silently mislabel them.
pub(crate) struct ResolvedRecipe<F: Launches> {
    pub(crate) body: RecipeBody<F>,
    pub(crate) origins: Vec<Option<usize>>,
}

impl<F: Launches> ResolvedRecipe<F> {
    pub(crate) fn detach_blobs(&mut self) {
        self.body.detach_blobs();
    }

    /// This resolution as a whole recipe, bound through `contract`.
    ///
    /// The one place a resolved body acquires ports, and it is the caller that
    /// holds them: a resolution of the root's body is bound through the root's
    /// contract, and the two are proved against each other here.
    pub(crate) fn into_flat(
        self,
        contract: F::Contract,
        authorization: &F::Authorization,
    ) -> Result<recipe::FlatRecipe<F>, Error> {
        let resolved = Recipe::stating_edges_authorized(
            contract,
            recipe::CompileRequests::default(),
            self.body,
            authorization,
        )
        .map_err(|error| Error::Decode(error.to_string()))?;
        FlatRecipe::new(resolved).map_err(|error| Error::Decode(error.to_string()))
    }
}

#[derive(Clone)]
struct ResolvedRequest {
    image: ModuleImage,
    resources: HashMap<String, KernelResources>,
    production_cost_ms: u64,
    compile_wall_ms: u64,
    resource_ms: u64,
    cache_source: Option<String>,
    assembler: String,
}

enum RequestState {
    Unscheduled,
    Resolving,
    Pending(OwnedCompileTask<Result<CompileProduct, CompileFailure>>),
    Alias(usize),
    Ready(Arc<ResolvedRequest>),
    /// This request produces no image, and would produce none anywhere. The
    /// cause is the answer for the rest of the realization.
    Failed(CompileFailure),
    /// The last attempt was stopped by something that is not this request — an
    /// absent toolchain, an interrupted worker, a failed spill. The cause is
    /// kept for the report, but it is not an answer: the next demand attempts
    /// the request again from [`RequestState::Unscheduled`].
    Deferred(CompileFailure),
}

/// How far a caller will go to settle a request it names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reach {
    /// Look-ahead. Start only what can start right now: a request whose
    /// prerequisite has not settled stays unscheduled rather than blocking the
    /// traversal that is filling the window ahead of itself.
    Available,
    /// The traversal needs this request's image before it can go on, so
    /// waiting for a prerequisite is the point.
    Required,
}

pub(crate) struct Resolver<'a, F: Launches> {
    ctx: CUcontext,
    sm: u16,
    options: RealizerOptions,
    requests: &'a CompileRequests,
    /// The ports of the recipe being realized. Every candidate body this
    /// resolver resolves is bound through the same ones, so it is borrowed once
    /// here rather than threaded through each `resolve_body`.
    signature: &'a F::Contract,
    entries: Vec<BTreeSet<String>>,
    states: Vec<RequestState>,
    keys: Vec<Option<RequestKey>>,
    /// The modules each scheduled request's compile links, in link order.
    units: Vec<Option<Vec<Blob>>>,
    /// The requests each key buckets. One key can hold several compiles: it
    /// carries what its sources *claim*, and equal claims are not equal bytes.
    owners: HashMap<RequestKey, Vec<usize>>,
    report: ResolutionReport,
    /// Where each request's row sits in `report.requests`. A retried request
    /// keeps one row and overwrites it, so the report states what the
    /// realization was finally left with rather than every attempt.
    reported: HashMap<usize, usize>,
}

impl<'a, F: Launches> Resolver<'a, F> {
    pub(crate) fn new(
        ctx: CUcontext,
        sm: u16,
        options: RealizerOptions,
        recipe: &'a Recipe<F>,
    ) -> Self {
        let mut entries = vec![BTreeSet::new(); recipe.context().as_slice().len()];
        collect_request_entries(recipe.body(), &mut entries);
        Resolver {
            ctx,
            sm,
            options,
            requests: recipe.context(),
            signature: recipe.contract(),
            entries,
            states: (0..recipe.context().as_slice().len())
                .map(|_| RequestState::Unscheduled)
                .collect(),
            keys: vec![None; recipe.context().as_slice().len()],
            units: vec![None; recipe.context().as_slice().len()],
            owners: HashMap::new(),
            report: ResolutionReport::default(),
            reported: HashMap::new(),
        }
    }

    /// The compile table every module below this recipe names.
    pub(crate) fn requests(&self) -> &CompileRequests {
        self.requests
    }

    /// How many upcoming candidates' request sets may be started at once: one
    /// per compile worker, so the pool is never idle while a candidate is being
    /// timed and never holds more outstanding sources than it has threads to
    /// assemble them with. Zero disables look-ahead entirely, leaving every
    /// request to be scheduled by the demand that needs it.
    pub(crate) fn prefetch_window(&self) -> usize {
        if !self.options.prefetch_modules {
            return 0;
        }
        cuda_toolchain::compile_environment().workers.max(1)
    }

    /// Start everything `body` will certainly need, without waiting for any of
    /// it. Speculative in *when*, never in *what*: only the requests a
    /// measurement of this body must resolve are started, so a subtree the
    /// tuning bound prunes is still never compiled.
    pub(crate) fn prefetch_body(&mut self, body: &RecipeBody<F>) {
        if !self.options.prefetch_modules {
            return;
        }
        let mut requests = Vec::new();
        certain_requests_of(body, &mut requests);
        for request in requests {
            self.schedule_request(request, Reach::Available);
        }
    }

    /// Give every module one loadable image, compiling its request when none of
    /// the images it carries fits. Takes a body, not a `Recipe`: the compile-request
    /// table stays borrowed from the pending recipe for the resolver's whole
    /// life, so tuning never copies it per candidate.
    pub(crate) fn resolve_body(&mut self, body: RecipeBody<F>) -> Result<ResolvedRecipe<F>, Error> {
        Context::from_raw(self.ctx)
            .set_current()
            .map_err(Error::Driver)?;
        let entries = entries_by_module(&body);
        let (inputs, outputs, mut modules, workspace, nodes) = body.into_parts();
        let mut origins = Vec::with_capacity(modules.modules().len());
        for (index, module) in modules.modules_mut().iter_mut().enumerate() {
            let request = module.code.compile_request();
            origins.push(request.map(|request| request.0));
            let module_entries = &entries[index];
            let bounds = module.resource_bounds;
            let mut failures = Vec::new();
            // Order is priority: the first image this machine accepts wins,
            // and every one of them is held to the module's bounds.
            let mut winner = None;
            for image in module.code.images() {
                match self.accept_image(image, bounds, module_entries) {
                    Ok(()) => {
                        winner = Some(image.clone());
                        break;
                    }
                    Err(failure) => failures.push(failure),
                }
            }
            if winner.is_none()
                && let Some(request) = request
            {
                match self.resolve_request(request.0) {
                    Ok(resolved) => {
                        match check_measured_bounds(bounds, module_entries, &resolved.resources) {
                            Err(failure) => failures.push(failure),
                            Ok(()) => winner = Some(resolved.image.clone()),
                        }
                    }
                    Err(failure) => failures.push(failure),
                }
            }
            let image = winner.ok_or(Error::ModuleResolution {
                module: index,
                failures,
            })?;
            *module = KernelModule {
                code: KernelCode::ImageOnly {
                    preferred_image: image,
                    alternative_images: Vec::new(),
                },
                resource_bounds: bounds,
            };
        }
        let body = RecipeBody::new(inputs, outputs, modules, workspace, nodes);
        // Every module now carries its own image, so the body names no compile
        // table at all — and what is left to prove about it is its shape.
        // Extents are the binding's, and this body's binding is whatever
        // selected it, which resolution does not touch.
        body.validate_shape(&recipe::CompileRequests::default())
            .map_err(|error| Error::Decode(error.to_string()))?;
        if let Some(node) = body
            .nodes()
            .iter()
            .position(|node| matches!(node.work(), recipe::Work::Choice { .. }))
        {
            return Err(Error::UnresolvedChoice { node });
        }
        Ok(ResolvedRecipe { body, origins })
    }

    /// The artifact `policy` asks this resolution to be written as, bound
    /// through the ports the recipe it resolved states.
    pub(crate) fn artifact(
        &self,
        resolved: &ResolvedRecipe<F>,
        policy: ArtifactPolicy,
        authorization: &F::Authorization,
    ) -> Result<Recipe<F>, Error> {
        crate::project::artifact(
            self.requests.as_slice(),
            self.signature.clone(),
            resolved,
            policy,
            authorization,
        )
    }

    fn settle_pending_compiles(&mut self) {
        // Cancel every queued callback before waiting for any running one. A
        // callback destructor may block or re-enter, so the first pass only
        // transfers cancelled callbacks into their owners.
        for state in &mut self.states {
            if let RequestState::Pending(task) = state {
                task.cancel_if_queued();
            }
        }
        for state in &mut self.states {
            if matches!(state, RequestState::Pending(_)) {
                *state = RequestState::Unscheduled;
            }
        }
    }

    pub(crate) fn finish(mut self) -> ResolutionReport {
        let blocked = Instant::now();
        self.settle_pending_compiles();
        self.report.compile_blocked_ms = self
            .report
            .compile_blocked_ms
            .saturating_add(blocked.elapsed().as_millis() as u64);
        std::mem::take(&mut self.report)
    }

    /// Whether this machine can use `image` for a module bounded by `bounds`.
    ///
    /// A cubin names the one architecture it was assembled for, so that is
    /// compared exactly. A CUDA-loadable image states no architecture — the
    /// loader owns that question — so it is loaded and asked. Either way the
    /// bounds are applied to what is loaded, so no image kind reaches a launch
    /// on weaker evidence than another.
    fn accept_image(
        &self,
        image: &ModuleImage,
        bounds: ResourceBounds,
        entries: &BTreeSet<String>,
    ) -> Result<(), CompileFailure> {
        if let ModuleImage::Cubin { target_sm, .. } = image
            && *target_sm != self.sm
        {
            return Err(CompileFailure {
                kind: CompileFailureKind::IncompatibleTarget,
                message: format!(
                    "cubin targets sm_{target_sm}, current target is sm_{}",
                    self.sm
                ),
            });
        }
        let module = Module::load(image.bytes()).map_err(|error| CompileFailure {
            kind: CompileFailureKind::InvalidImage,
            message: error.to_string(),
        })?;
        // A bound that states nothing needs no measurement; anything it does
        // state is applied here exactly as it is to an image this machine
        // assembled, measurement and all.
        if bounds != ResourceBounds::default() {
            check_measured_bounds(bounds, entries, &measure_loaded_entries(&module, entries)?)?;
        }
        Ok(())
    }

    /// Take one request from unscheduled to in-flight, resolved, or failed.
    ///
    /// `reach` decides only what happens at a prerequisite that has not
    /// settled: look-ahead leaves the request unscheduled for a later demand,
    /// so filling the window never blocks on the window. What is *checked* is
    /// the same either way — no request is ever started against an unsatisfied
    /// prerequisite.
    fn schedule_request(&mut self, request_id: usize, reach: Reach) {
        if !matches!(self.states.get(request_id), Some(RequestState::Unscheduled)) {
            return;
        }
        self.states[request_id] = RequestState::Resolving;
        let request = self.requests.as_slice()[request_id].clone();
        // Every module of the unit has to run here, so the first one that does
        // not is what stops the compile — a link is no more portable than its
        // least portable input.
        if let Some((source, image)) = request
            .sources
            .iter()
            .enumerate()
            .find(|(_, source)| source.target.minimum_sm > self.sm)
        {
            self.fail(
                request_id,
                CompileFailure {
                    kind: CompileFailureKind::IncompatibleTarget,
                    message: format!(
                        "PTX source {source} requires sm_{}, current target is sm_{}",
                        image.target.minimum_sm, self.sm
                    ),
                },
            );
            return;
        }
        let task = match CompileTask::for_request(&request, self.sm) {
            Ok(task) => task,
            Err(failure) => {
                self.fail(request_id, failure);
                return;
            }
        };
        let key = task.key();
        self.keys[request_id] = Some(key.clone());
        // Kept past the spawn: a key only buckets requests, so whenever an
        // image would cross to another request without being compiled, these
        // bytes are what proves the two are one compile.
        self.units[request_id] = Some(task.unit_sources().to_vec());
        for (prerequisite, bounds) in &request.requires {
            if reach == Reach::Available {
                // Pull the base forward — it is needed too — but never wait on
                // it here: this call is look-ahead, and the traversal it runs
                // ahead of is what should be spending the time.
                self.schedule_request(prerequisite.0, Reach::Available);
                if !matches!(self.states[prerequisite.0], RequestState::Ready(_)) {
                    self.states[request_id] = RequestState::Unscheduled;
                    return;
                }
            }
            let resolved = match self.resolve_request(prerequisite.0) {
                Ok(resolved) => resolved,
                Err(failure) => {
                    self.fail(request_id, failure);
                    return;
                }
            };
            // The bound speaks about the base *image*, so it is applied to
            // every entry the recipe launches from that base — and refuses to
            // pass when there is no such entry to measure. A base named only as
            // a prerequisite exposes nothing to measure, and a gate that cannot
            // be evaluated is not a gate that was satisfied.
            let measured =
                check_measured_bounds(*bounds, &self.entries[prerequisite.0], &resolved.resources);
            if let Err(failure) = measured {
                self.fail(request_id, failure);
                return;
            }
            let base = self.keys[prerequisite.0].clone();
            let reuses_base = key.max_registers.is_some_and(|cap| {
                base.is_some_and(|base| key.reuses_uncapped(&base))
                    && !resolved.resources.is_empty()
                    && resolved
                        .resources
                        .values()
                        .all(|resources| resources.registers <= cap)
            });
            // The base's image is handed over only once its unit is proved to
            // be this one's, so a capped request cannot take an image
            // assembled from bytes it does not hold.
            let reuses_base = match reuses_base {
                false => false,
                true => match self.same_unit(prerequisite.0, request_id) {
                    Ok(same) => same,
                    Err(failure) => {
                        self.fail(request_id, failure);
                        return;
                    }
                },
            };
            if reuses_base {
                let resolved = match self.adopt_shared_image(request_id, resolved) {
                    Ok(resolved) => resolved,
                    Err(failure) => {
                        self.fail(request_id, failure);
                        return;
                    }
                };
                self.states[request_id] = RequestState::Ready(resolved.clone());
                self.record_ready(request_id, RequestOutcome::Reused, &resolved);
                return;
            }
        }
        // A request that already owns this compile is re-attempting it, not
        // aliasing itself. The others in this bucket merely claim the same
        // sources; the one whose bytes are this request's, if any, owns the
        // image it may use.
        let bucket: Vec<usize> = self.owners.get(&key).cloned().unwrap_or_default();
        for owner in bucket {
            if owner == request_id {
                continue;
            }
            match self.same_unit(owner, request_id) {
                Ok(true) => {
                    self.states[request_id] = RequestState::Alias(owner);
                    return;
                }
                Ok(false) => {}
                Err(failure) => {
                    self.fail(request_id, failure);
                    return;
                }
            }
        }
        self.owners.entry(key).or_default().push(request_id);
        chronicle::mark(|| format!("kiln.assembly_queued request={request_id}"));
        let receiver = cuda_toolchain::spawn_compile(move || {
            let _assembly = chronicle::range(|| format!("kiln.assembly request={request_id}"));
            let result = task.produce();
            chronicle::mark(|| {
                format!(
                    "kiln.assembly_result request={request_id} result={}",
                    match &result {
                        Ok(product) => product.cache_source.as_deref().unwrap_or("session"),
                        Err(_) => "failed",
                    }
                )
            });
            result
        });
        self.states[request_id] = RequestState::Pending(receiver);
        self.observe_outstanding();
    }

    /// The decoded size of every source whose compile is outstanding: spawned
    /// and not yet claimed by a demand.
    ///
    /// Read off the request states rather than carried in a counter. Every
    /// spawn has a demand that claims it *only* when the traversal reaches it,
    /// and look-ahead exists precisely so that it sometimes does not — a
    /// pruned subtree, an expired deadline, a body that failed at an earlier
    /// module. A counter has nobody to decrement for those, so it would keep
    /// their bytes for the rest of the realization and report a peak that grows
    /// with the very pruning it is meant to show paying off.
    fn outstanding_source_bytes(&self) -> usize {
        self.states
            .iter()
            .enumerate()
            .filter(|(_, state)| matches!(state, RequestState::Pending(_)))
            .map(|(request, _)| {
                self.requests.as_slice()[request]
                    .sources
                    .iter()
                    .map(|source| source.blob.len())
                    .sum::<usize>()
            })
            .sum()
    }

    /// Sample the outstanding set into the report's peak. The set grows only
    /// where a compile is spawned, so sampling there sees every maximum it
    /// reaches.
    fn observe_outstanding(&mut self) {
        self.report.peak_queued_source_bytes = self
            .report
            .peak_queued_source_bytes
            .max(self.outstanding_source_bytes());
    }

    fn resolve_request(
        &mut self,
        request_id: usize,
    ) -> Result<Arc<ResolvedRequest>, CompileFailure> {
        if request_id >= self.states.len() {
            return Err(CompileFailure {
                kind: CompileFailureKind::InvalidImage,
                message: format!("compile request {request_id} is out of range"),
            });
        }
        // One demand makes one attempt. A deferred request is forgotten first,
        // so this demand re-attempts it; whatever that attempt leaves behind —
        // including another deferral — is this demand's answer, and the loop
        // that would re-attempt it again belongs to the next demand.
        if matches!(
            self.states[request_id],
            RequestState::Unscheduled | RequestState::Deferred(_)
        ) {
            self.states[request_id] = RequestState::Unscheduled;
            self.schedule_request(request_id, Reach::Required);
        }
        let state = std::mem::replace(&mut self.states[request_id], RequestState::Resolving);
        match state {
            RequestState::Ready(resolved) => {
                self.states[request_id] = RequestState::Ready(resolved.clone());
                Ok(resolved)
            }
            RequestState::Failed(failure) => {
                self.states[request_id] = RequestState::Failed(failure.clone());
                Err(failure)
            }
            RequestState::Deferred(failure) => {
                self.states[request_id] = RequestState::Deferred(failure.clone());
                Err(failure)
            }
            RequestState::Alias(owner) => {
                self.states[request_id] = RequestState::Alias(owner);
                let resolved = self.resolve_request(owner)?;
                let resolved = match self.adopt_shared_image(request_id, resolved) {
                    Ok(resolved) => resolved,
                    Err(failure) => {
                        self.fail(request_id, failure.clone());
                        return Err(failure);
                    }
                };
                self.states[request_id] = RequestState::Ready(resolved.clone());
                self.record_ready(request_id, RequestOutcome::Reused, &resolved);
                Ok(resolved)
            }
            RequestState::Pending(receiver) => {
                let blocked = Instant::now();
                let waiting =
                    chronicle::range(|| format!("kiln.compile_wait request={request_id}"));
                let product = receiver.recv();
                drop(waiting);
                self.report.compile_blocked_ms = self
                    .report
                    .compile_blocked_ms
                    .saturating_add(blocked.elapsed().as_millis() as u64);
                match product {
                    Ok(Ok(product)) => self.finish_compile(request_id, product),
                    Ok(Err(failure)) => {
                        self.fail(request_id, failure.clone());
                        Err(failure)
                    }
                    Err(error) => {
                        let failure = CompileFailure {
                            kind: CompileFailureKind::Interrupted,
                            message: error.to_string(),
                        };
                        self.fail(request_id, failure.clone());
                        Err(failure)
                    }
                }
            }
            RequestState::Resolving => {
                let failure = CompileFailure {
                    kind: CompileFailureKind::InvalidImage,
                    message: format!("compile request dependency cycle at {request_id}"),
                };
                self.fail(request_id, failure.clone());
                Err(failure)
            }
            RequestState::Unscheduled => unreachable!("request was prefetched above"),
        }
    }

    /// Whether two requests compile the same modules, in link order, byte for
    /// byte.
    ///
    /// Every place an image crosses from one request to another without being
    /// compiled asks this. The key that brought the two together carries only
    /// what their sources *claim* — an encoded blob's digest is unchecked
    /// until its bytes are expanded — so a request whose bytes are not the
    /// owner's is not that compile, and one whose bytes cannot be read at all
    /// is not anyone's.
    fn same_unit(&self, left: usize, right: usize) -> Result<bool, CompileFailure> {
        match (&self.units[left], &self.units[right]) {
            (Some(left), Some(right)) => crate::compile::same_unit(left, right),
            _ => Ok(false),
        }
    }

    /// Take over an image another request already produced, measuring whatever
    /// entries this request launches and that one never did.
    ///
    /// Two requests share an image because their *sources* and compile options
    /// agree, which says nothing about the entry points their modules launch.
    /// Without this, a producer-declared bound would be checked against an
    /// entry set that does not contain the kernel it bounds.
    fn adopt_shared_image(
        &mut self,
        request_id: usize,
        resolved: Arc<ResolvedRequest>,
    ) -> Result<Arc<ResolvedRequest>, CompileFailure> {
        let missing: Vec<String> = self.entries[request_id]
            .iter()
            .filter(|entry| !resolved.resources.contains_key(*entry))
            .cloned()
            .collect();
        if missing.is_empty() {
            return Ok(resolved);
        }
        Context::from_raw(self.ctx)
            .set_current()
            .map_err(|error| CompileFailure {
                kind: CompileFailureKind::UnavailableToolchain,
                message: error.to_string(),
            })?;
        let measured = cuda_toolchain::kernel_image_resources(resolved.image.bytes(), &missing)
            .ok_or_else(|| CompileFailure {
                kind: CompileFailureKind::InvalidImage,
                message: format!("shared image does not expose entries {missing:?}"),
            })?;
        let mut adopted = (*resolved).clone();
        adopted.resources.extend(missing.into_iter().zip(measured));
        Ok(Arc::new(adopted))
    }

    fn finish_compile(
        &mut self,
        request_id: usize,
        product: CompileProduct,
    ) -> Result<Arc<ResolvedRequest>, CompileFailure> {
        Context::from_raw(self.ctx)
            .set_current()
            .map_err(|error| CompileFailure {
                kind: CompileFailureKind::UnavailableToolchain,
                message: error.to_string(),
            })?;
        let entries: Vec<String> = self.entries[request_id].iter().cloned().collect();
        let resource_started = Instant::now();
        let measured = if entries.is_empty() {
            Vec::new()
        } else {
            cuda_toolchain::kernel_image_resources(&product.image, &entries).ok_or_else(|| {
                CompileFailure {
                    kind: CompileFailureKind::InvalidImage,
                    message: format!("compiled image does not expose entries {entries:?}"),
                }
            })?
        };
        let resources: HashMap<_, _> = entries.iter().cloned().zip(measured).collect();
        let resource_ms = resource_started.elapsed().as_millis() as u64;
        let provenance = Some(CompileProvenance {
            unit_digest: product.unit_digest,
            options: self.requests.as_slice()[request_id].options,
            target_sm: self.sm,
            assembler: product.assembler.clone(),
            assembler_version: product.assembler_version,
            driver_version: None,
            toolkit_version: None,
        });
        let resolved = Arc::new(ResolvedRequest {
            image: match product.form {
                crate::compile::ProductForm::Cubin => ModuleImage::Cubin {
                    target_sm: self.sm,
                    bytes: product.image,
                    provenance,
                },
                form => ModuleImage::CudaLoadable {
                    bytes: product.image,
                    persistence: form.persistence(),
                    provenance,
                },
            },
            resources,
            production_cost_ms: product.production_cost_ms,
            compile_wall_ms: product.wall_ms,
            resource_ms,
            cache_source: product.cache_source,
            assembler: product.assembler,
        });
        self.states[request_id] = RequestState::Ready(resolved.clone());
        let outcome = match product.form {
            crate::compile::ProductForm::Cubin => RequestOutcome::Compiled,
            crate::compile::ProductForm::SessionLoadable => RequestOutcome::SourceLoaded,
        };
        self.record_ready(request_id, outcome, &resolved);
        Ok(resolved)
    }

    /// Record what stopped a request, and decide whether that is its answer.
    ///
    /// Only a failure the request itself causes settles it. A missing
    /// toolchain, an interrupted worker, a spill that could not be written —
    /// none of those say anything about this PTX, so remembering them as the
    /// request's outcome would let one blip eliminate every candidate that
    /// shares the request for the rest of the realization. Those are deferred:
    /// the cause is reported, the attempt is forgotten, and the next demand
    /// makes a new one.
    fn fail(&mut self, request_id: usize, failure: CompileFailure) {
        let deterministic = failure.kind.is_deterministic();
        self.states[request_id] = if deterministic {
            RequestState::Failed(failure.clone())
        } else {
            RequestState::Deferred(failure.clone())
        };
        self.record(RequestReport {
            request: request_id,
            outcome: if deterministic {
                RequestOutcome::Failed
            } else {
                RequestOutcome::Deferred
            },
            assembler: None,
            production_cost_ms: None,
            compile_wall_ms: None,
            resource_ms: None,
            cache_source: None,
            image_bytes: None,
            resources: Vec::new(),
            failure: Some(failure),
        });
    }

    fn record_ready(
        &mut self,
        request_id: usize,
        outcome: RequestOutcome,
        resolved: &ResolvedRequest,
    ) {
        let mut resources: Vec<_> = resolved
            .resources
            .iter()
            .map(|(entry, resources)| (entry.clone(), *resources))
            .collect();
        resources.sort_by(|left, right| left.0.cmp(&right.0));
        self.record(RequestReport {
            request: request_id,
            outcome,
            assembler: Some(resolved.assembler.clone()),
            production_cost_ms: Some(resolved.production_cost_ms),
            compile_wall_ms: Some(resolved.compile_wall_ms),
            resource_ms: Some(resolved.resource_ms),
            cache_source: resolved.cache_source.clone(),
            image_bytes: Some(resolved.image.bytes().len()),
            resources,
            failure: None,
        });
    }

    /// One row per request, holding its latest outcome. A retried request
    /// overwrites its own row, so the report states where the realization
    /// ended up rather than every attempt it took to get there.
    fn record(&mut self, entry: RequestReport) {
        match self.reported.get(&entry.request) {
            Some(&row) => self.report.requests[row] = entry,
            None => {
                self.reported
                    .insert(entry.request, self.report.requests.len());
                self.report.requests.push(entry);
            }
        }
    }
}

impl<F: Launches> Drop for Resolver<'_, F> {
    fn drop(&mut self) {
        self.settle_pending_compiles();
    }
}

/// What each of `entries` costs in a loaded image, keyed as the compile path
/// keys what it measured — so one bound contract reads both.
fn measure_loaded_entries(
    module: &Module,
    entries: &BTreeSet<String>,
) -> Result<HashMap<String, KernelResources>, CompileFailure> {
    let device = Device::current();
    entries
        .iter()
        .map(|entry| {
            let attribute = |attribute| {
                module
                    .kernel_attribute(entry, attribute, device)
                    .map_err(|error| CompileFailure {
                        kind: CompileFailureKind::InvalidImage,
                        message: format!("cannot inspect {entry}: {error}"),
                    })
            };
            let registers = attribute(cuda_driver::ffi::CU_FUNC_ATTRIBUTE_NUM_REGS)?;
            let local_bytes = attribute(cuda_driver::ffi::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?;
            Ok((
                entry.clone(),
                KernelResources {
                    registers: registers as u32,
                    local_bytes: local_bytes as u32,
                },
            ))
        })
        .collect()
}

#[cfg(test)]
#[path = "resolve_tests.rs"]
mod tests;
