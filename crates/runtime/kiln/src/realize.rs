// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Turning a pending recipe into the best executable one for this device and
//! this workload — resolving, tuning, emitting the artifact, and reusing a
//! completed realization instead of repeating it.

use std::time::Instant;

use cuda_driver::{CUcontext, CUstream, Context, Device};
use recipe::algebra::Launches;
#[cfg(test)]
use recipe::algebra::ModuleTable;
#[cfg(test)]
use recipe::{CompileRequests, Geometry, KernelLaunch};
use recipe::{FlatRecipe, Recipe};

use crate::artifact_cache::{
    ArtifactCache, ArtifactCacheOutcome, ArtifactKey, DeviceIdentity, Storable, TargetIdentity,
    WorkloadIdentity,
};
use crate::compile::reachable_assemblers;
use crate::identity::RecipeIdentity;
use crate::progress::{TuneReport, TuningProgress};
use crate::resolve::{RequestReport, ResolutionReport, Resolver};
use crate::tune::{Deadline, TuningPolicy, TuningRun};
use crate::validate::{ValidationContract, ValidationStatus};
use crate::{Bindings, Error, ResidentRecipe};

#[derive(Clone, Copy, Debug)]
pub struct RealizerOptions {
    pub prefetch_modules: bool,
}

impl Default for RealizerOptions {
    fn default() -> Self {
        RealizerOptions {
            prefetch_modules: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ArtifactPolicy {
    #[default]
    NativeOnly,
    NativeWithSourceFallback,
    SourceBacked,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RealizationPolicy {
    pub collect_tuning_report: bool,
    pub artifact: ArtifactPolicy,
    pub tuning: TuningPolicy,
}

/// Everything one realization needs: the program, the buffers it is measured
/// against, and the policies that shape the result.
///
/// A bundle rather than a parameter list because the policy inputs keep
/// arriving — an output policy, a tuning deadline, a workload identity, a
/// cache — and each new one would otherwise become another positional argument
/// at every call site.
pub struct RealizationArgs<'a, F: Launches = recipe::Executable> {
    pub recipe: &'a Recipe<F>,
    pub bindings: &'a Bindings<'a>,
    pub stream: CUstream,
    pub policy: RealizationPolicy,
    /// The caller's name for the data in `bindings`. Without it the tuning
    /// result is this process's alone: nothing is looked up and nothing is
    /// stored.
    pub workload: Option<&'a WorkloadIdentity>,
    pub cache: Option<&'a ArtifactCache>,
    /// Where a previous realization's deadline stopped. In-process only; see
    /// [`TuningProgress`].
    pub resume: Option<&'a TuningProgress>,
    pub validation: Option<ValidationContract<'a>>,
    /// What says this realization may make recipes of the family it realizes:
    /// the artifact it emits and every body it inlines to measure one are that
    /// family's values.
    pub authorization: &'a F::Authorization,
    /// What each measured candidate's output buffers are put back to before it
    /// is timed. Without one, a program that reads what it writes measures its
    /// own previous results.
    pub starting_state: Option<&'a dyn crate::validate::StartingState>,
}

impl<'a, F: Launches> RealizationArgs<'a, F> {
    /// A realization of work whose family says who may make its values.
    pub fn authorized(
        recipe: &'a Recipe<F>,
        bindings: &'a Bindings<'a>,
        stream: CUstream,
        policy: RealizationPolicy,
        authorization: &'a F::Authorization,
    ) -> Self {
        RealizationArgs {
            recipe,
            bindings,
            stream,
            policy,
            workload: None,
            cache: None,
            resume: None,
            validation: None,
            starting_state: None,
            authorization,
        }
    }
}

impl<'a, F> RealizationArgs<'a, F>
where
    F: Launches + recipe::algebra::Computation<Authorization = ()>,
{
    /// The same, for a family whose values anyone may make.
    pub fn new(
        recipe: &'a Recipe<F>,
        bindings: &'a Bindings<'a>,
        stream: CUstream,
        policy: RealizationPolicy,
    ) -> Self {
        RealizationArgs::authorized(recipe, bindings, stream, policy, &())
    }
}

/// Where a realization's wall clock went.
#[derive(Clone, Copy, Debug, Default)]
pub struct RealizationTiming {
    /// The whole `realize` call.
    pub total_ms: u64,
    /// Tuning, which is also where candidate compilation happens: the resolver
    /// compiles what the traversal reaches, so the two overlap by design.
    pub tuning_ms: u64,
    /// Projecting the selected recipe into the output artifact.
    pub artifact_ms: u64,
    /// Looking up, loading, and storing the realized artifact.
    pub artifact_cache_ms: u64,
    /// How long resolution stood still waiting for compile products.
    pub compile_blocked_ms: u64,
    /// Summed over requests: wall clock their compile tasks ran in this
    /// process. Against `tuning_ms` this is the concurrency the compile pool
    /// actually delivered.
    pub compile_wall_ms: u64,
    /// Summed over requests: what producing their images cost according to the
    /// compile cache. A cache hit contributes the cost recorded when the image
    /// was first produced — historical work, not work this run did, so this
    /// exceeds `compile_wall_ms` on a warm compile cache.
    pub compile_production_cost_ms: u64,
    /// Summed over requests: reading back kernel resources.
    pub resource_ms: u64,
}

#[derive(Clone, Debug)]
pub struct RealizationReport {
    pub sm: u16,
    pub environment: cuda_toolchain::CompileEnvironment,
    pub requests: Vec<RequestReport>,
    pub peak_queued_source_bytes: usize,
    pub artifact_cache: ArtifactCacheOutcome,
    pub validation: ValidationStatus,
    pub timing: RealizationTiming,
}

impl RealizationReport {
    fn new(sm: u16, resolution: ResolutionReport) -> Self {
        let mut timing = RealizationTiming {
            compile_blocked_ms: resolution.compile_blocked_ms,
            ..RealizationTiming::default()
        };
        for request in &resolution.requests {
            timing.compile_wall_ms = timing
                .compile_wall_ms
                .saturating_add(request.compile_wall_ms.unwrap_or(0));
            timing.compile_production_cost_ms = timing
                .compile_production_cost_ms
                .saturating_add(request.production_cost_ms.unwrap_or(0));
            timing.resource_ms = timing
                .resource_ms
                .saturating_add(request.resource_ms.unwrap_or(0));
        }
        RealizationReport {
            sm,
            environment: cuda_toolchain::compile_environment().clone(),
            requests: resolution.requests,
            peak_queued_source_bytes: resolution.peak_queued_source_bytes,
            artifact_cache: ArtifactCacheOutcome::Disabled,
            validation: ValidationStatus::NotRequested,
            timing,
        }
    }
}

/// One arm of a covering sweep that could not be given a loadable image, and
/// why. The selection names the arm; `arm` is its position in the sweep, which
/// a caller counting only the arms it visited cannot recover.
#[derive(Clone, Debug)]
pub struct UnrealizedArm {
    pub arm: usize,
    pub selection: Vec<crate::selection::Selection>,
    pub error: Error,
}

pub struct RealizationResult<F: Launches = recipe::Executable> {
    pub resident: ResidentRecipe<F>,
    pub artifact: Recipe<F>,
    pub report: RealizationReport,
    pub tuning: Option<TuneReport>,
    pub progress: TuningProgress,
}

pub struct Realizer {
    ctx: CUcontext,
    device: DeviceIdentity,
    options: RealizerOptions,
}

impl Realizer {
    pub fn new(ctx: CUcontext, options: RealizerOptions) -> Result<Self, Error> {
        Context::from_raw(ctx)
            .set_current()
            .map_err(Error::Driver)?;
        Ok(Realizer {
            ctx,
            device: device_identity()?,
            options,
        })
    }

    pub fn target_sm(&self) -> u16 {
        self.device.sm
    }

    pub fn device(&self) -> &DeviceIdentity {
        &self.device
    }

    pub unsafe fn realize<F: Storable>(
        &mut self,
        args: RealizationArgs<'_, F>,
    ) -> Result<RealizationResult<F>, Error> {
        let _realization = chronicle::range(|| "kiln.realize".into());
        let started = Instant::now();
        // First, before anything is asked *about* this recipe. A cache hit is
        // reached without ever constructing a resolver for the pending program,
        // and the artifact key names its content rather than its validity — so
        // an invalid recipe whose key happens to have an entry would otherwise
        // be answered from the cache and returned as a realization of itself.
        // Nothing downstream can catch that: the artifact loads and runs, it is
        // simply not this program.
        args.recipe.validate().map_err(Error::InvalidRecipe)?;
        let limit = args
            .policy
            .tuning
            .scheduling_time_limit()
            .map_err(Error::InvalidTuningPolicy)?;
        // The contract is checked against the program before anything runs: a
        // contract that does not cover this recipe's outputs would otherwise
        // only be discovered once a candidate had already written somewhere.
        if let Some(contract) = &args.validation {
            contract.covers(args.recipe, args.bindings)?;
        }
        let cache_started = Instant::now();
        let (mut slot, hit) = ArtifactSlot::open(&args, &self.device);
        let mut cache_ms = cache_started.elapsed().as_millis() as u64;

        if let Some(cached) = hit {
            // A hit is only reachable when nothing was asked to be validated:
            // `ArtifactSlot::open` bypasses the cache otherwise, precisely so
            // that no realization can report a comparison it did not make.
            debug_assert!(args.validation.is_none());
            let mut result = unsafe { self.load_artifact(cached, args.authorization)? };
            result.report.artifact_cache = slot.outcome;
            result.report.validation = ValidationStatus::NotRequested;
            result.report.timing.artifact_cache_ms = cache_ms;
            result.report.timing.total_ms = started.elapsed().as_millis() as u64;
            return Ok(result);
        }

        let mut resolver = self.resolver(args.recipe)?;
        let tuning_started = Instant::now();
        let tuning_range = chronicle::range(|| "kiln.autotune".into());
        let tuned = unsafe {
            crate::tune::tune_with_resolver(
                TuningRun {
                    ctx: self.ctx,
                    authorization: args.authorization,
                    recipe: args.recipe,
                    bindings: args.bindings,
                    stream: args.stream,
                    deadline: Deadline::start(limit),
                    resume: args.resume,
                    report: args.policy.collect_tuning_report.then(TuneReport::default),
                    validation: args.validation,
                    starting_state: args.starting_state,
                },
                &mut resolver,
            )
        }?;
        drop(tuning_range);
        let tuning_ms = tuning_started.elapsed().as_millis() as u64;

        let artifact_started = Instant::now();
        let artifact =
            resolver.artifact(&tuned.resolved, args.policy.artifact, args.authorization)?;
        let artifact_ms = artifact_started.elapsed().as_millis() as u64;

        // After tuning, so the selection being stored is the one validation
        // already filtered — never a candidate set a comparison had yet to see.
        let store_started = Instant::now();
        slot.store(&artifact, &tuned.progress, &tuned.validation);
        cache_ms = cache_ms.saturating_add(store_started.elapsed().as_millis() as u64);

        let mut report = RealizationReport::new(self.device.sm, resolver.finish());
        report.artifact_cache = slot.outcome;
        report.validation = tuned.validation.clone();
        report.timing.tuning_ms = tuning_ms;
        report.timing.artifact_ms = artifact_ms;
        report.timing.artifact_cache_ms = cache_ms;
        let resident = unsafe {
            ResidentRecipe::new(
                self.ctx,
                tuned
                    .resolved
                    .into_flat(args.recipe.contract().clone(), args.authorization)?,
            )?
        };
        report.timing.total_ms = started.elapsed().as_millis() as u64;
        Ok(RealizationResult {
            resident,
            artifact,
            report,
            tuning: tuned.report,
            progress: tuned.progress,
        })
    }

    /// Load a completed artifact: no candidate is compiled and nothing is
    /// measured. A source-backed artifact still resolves the one request its
    /// winning module kept — that compile is the artifact, not a search.
    unsafe fn load_artifact<F: Launches>(
        &mut self,
        cached: Recipe<F>,
        authorization: &F::Authorization,
    ) -> Result<RealizationResult<F>, Error> {
        let mut resolver = self.resolver(&cached)?;
        let resolved = resolver.resolve_body(cached.body().clone())?;
        let report = RealizationReport::new(self.device.sm, resolver.finish());
        let resident = unsafe {
            ResidentRecipe::new(
                self.ctx,
                resolved.into_flat(cached.contract().clone(), authorization)?,
            )?
        };
        Ok(RealizationResult {
            resident,
            artifact: cached,
            report,
            tuning: None,
            progress: TuningProgress::default(),
        })
    }

    /// Visit every covering selection of `recipe` — each candidate of each
    /// choice at least once — as a recipe that can load: one image per
    /// module, with nothing left to compile.
    ///
    /// Realization, not measurement. Nothing here is timed and nothing is
    /// selected; a caller that wants each arm *run* is asking about
    /// correctness, and each arm's own result is the answer. What it shares
    /// with `realize` is the part that must not be re-derived: one resolver
    /// spans the whole sweep, so a request that several arms name is assembled
    /// once, its declared prerequisites gate it, and the module's resource
    /// bounds eliminate the same arms they would eliminate under tuning.
    ///
    /// An arm that does not realize is *skipped*, not fatal — the same thing
    /// tuning does with an infeasible candidate, and for the same reason: a
    /// register cap that spills or a cubin for another target says nothing
    /// about the arms beside it. Every remaining arm is still visited, and the
    /// ones that did not realize come back with their causes. They are returned
    /// rather than swallowed because "the sweep checked every arm" and "the
    /// sweep checked the arms this machine could build" are different claims,
    /// and only the caller knows which one it needs.
    #[must_use = "an unrealized arm was not checked; a sweep that ignores this \
                  reports coverage it does not have"]
    ///
    /// A recipe that is not a program at all fails the whole sweep rather than
    /// appearing as an unrealized arm: that is a statement about this machine's
    /// toolchain, and an invalid recipe is a statement about the recipe.
    pub fn for_each_covering_recipe<F: Launches>(
        &self,
        recipe: &Recipe<F>,
        authorization: &F::Authorization,
        mut visit: impl FnMut(FlatRecipe<F>),
    ) -> Result<Vec<UnrealizedArm>, Error> {
        let mut resolver = self.resolver(recipe)?;
        let mut unrealized = Vec::new();
        for (arm, selection) in crate::selection::covering_selections(recipe)
            .into_iter()
            .enumerate()
        {
            let covering = crate::selection::inline_body(
                recipe.context(),
                recipe.body(),
                &selection,
                authorization,
            )
            .map_err(Error::InvalidRecipe)?;
            match resolver.resolve_body(covering) {
                Ok(resolved) => {
                    visit(resolved.into_flat(recipe.contract().clone(), authorization)?)
                }
                Err(error) => unrealized.push(UnrealizedArm {
                    arm,
                    selection,
                    error,
                }),
            }
        }
        Ok(unrealized)
    }

    /// The one door into resolution, and so into every compile, tune, load, and
    /// launch kiln performs.
    ///
    /// A recipe is checked against itself here rather than at each of the
    /// callers — realization, tuning, an artifact-cache hit, a covering walk —
    /// because a caller that forgot would be the one that reached the device.
    /// Bytes decoded from the wire were already checked at `deserialize`; this
    /// covers everything that arrives some other way, including the in-memory
    /// artifact cache, which hands back a `Recipe` that never round-tripped.
    pub(crate) fn resolver<'a, F: Launches>(
        &self,
        recipe: &'a Recipe<F>,
    ) -> Result<Resolver<'a, F>, Error> {
        recipe.validate().map_err(Error::InvalidRecipe)?;
        Ok(Resolver::new(
            self.ctx,
            self.device.sm,
            self.options,
            recipe,
        ))
    }
}

fn device_identity() -> Result<DeviceIdentity, Error> {
    let device = Device::current();
    let attribute = |value: Result<i32, cuda_driver::Error>| -> Result<u32, Error> {
        u32::try_from(value.map_err(Error::Driver)?)
            .map_err(|_| Error::Decode("negative CUDA device attribute".into()))
    };
    let sm = attribute(device.sm_arch())?;
    Ok(DeviceIdentity {
        sm: u16::try_from(sm).map_err(|_| Error::Decode("CUDA target out of range".into()))?,
        sm_count: attribute(device.sm_count())?,
        max_threads_per_sm: attribute(device.max_threads_per_sm())?,
        max_smem_optin: attribute(device.max_smem_optin())?,
        driver_version: driver_version()?,
    })
}

fn driver_version() -> Result<u32, Error> {
    let mut version = 0;
    let status = unsafe { (cuda_driver::CudaDriver::get().cuDriverGetVersion)(&mut version) };
    cuda_driver::check(status, "cuDriverGetVersion").map_err(Error::Driver)?;
    u32::try_from(version).map_err(|_| Error::Decode("negative CUDA driver version".into()))
}

/// The realized-artifact cache as one realization sees it: the key this run
/// would be stored under, what was found there, and what happened in the end.
struct ArtifactSlot<'a> {
    cache: Option<&'a ArtifactCache>,
    key: Option<ArtifactKey>,
    outcome: ArtifactCacheOutcome,
}

impl<'a> ArtifactSlot<'a> {
    /// An artifact is reused only when the program *and* the workload are
    /// identified. Program identity alone would reuse a selection measured
    /// against other data, and shapes or device pointers do not supply the
    /// missing half — so an unidentified workload means no cache at all.
    ///
    /// A validated realization gets no cache at all either, in both directions.
    /// [`ArtifactKey`] names the program, the workload, the target and the
    /// output policy — not the contract, not the comparator, and not whether
    /// any comparison happened. So an entry under that key is no evidence that
    /// the selection inside it was ever checked: reusing it would let a
    /// realization that asked for validation report a comparison nothing made,
    /// and storing under it would offer an unvalidated selection to a later
    /// caller who asked for one. Naming the contract in the key is the way to
    /// get this cache back, and it needs a caller-owned validation identity —
    /// kiln cannot derive one from a `&dyn` comparator.
    fn open<F: Storable>(
        args: &RealizationArgs<'a, F>,
        device: &DeviceIdentity,
    ) -> (Self, Option<Recipe<F>>) {
        let Some(cache) = args.cache else {
            return (
                ArtifactSlot {
                    cache: None,
                    key: None,
                    outcome: ArtifactCacheOutcome::Disabled,
                },
                None,
            );
        };
        if F::as_stored(args.recipe).is_none() {
            return (
                ArtifactSlot {
                    cache: None,
                    key: None,
                    outcome: ArtifactCacheOutcome::Bypassed {
                        reason: "the cache holds recipes in the public wire format, which states \
                                 the default contract, and this one is bound through another"
                            .into(),
                    },
                },
                None,
            );
        }
        if args.validation.is_some() {
            return (
                ArtifactSlot {
                    cache: None,
                    key: None,
                    outcome: ArtifactCacheOutcome::Bypassed {
                        reason: "validation was requested, and the artifact key does not name the \
                                 validation contract: an entry under it neither proves a stored \
                                 selection was checked nor stays honest if this one is"
                            .into(),
                    },
                },
                None,
            );
        }
        let unidentified = |reason: String| ArtifactSlot {
            cache: None,
            key: None,
            outcome: ArtifactCacheOutcome::Unidentified { reason },
        };
        let Some(workload) = args.workload else {
            return (
                unidentified(
                    "no workload identity: a tuning result measured against unnamed data \
                     cannot be reused"
                        .into(),
                ),
                None,
            );
        };
        let program = match RecipeIdentity::of(args.recipe) {
            Ok(program) => program,
            Err(error) => return (unidentified(format!("pending recipe: {error}")), None),
        };
        let target = TargetIdentity {
            device: device.clone(),
            assemblers: reachable_assemblers(args.recipe),
        };
        let key = ArtifactKey::new(program, workload, &target, args.policy.artifact);
        // A miss stands as "not stored" until the realization reaches the
        // store; an unreadable entry is a miss that keeps its cause, and a
        // successful store overwrites it.
        let (outcome, hit) = match cache.load(&key) {
            // The entry is bytes in the public format; what this realization
            // wants back is a recipe of its own family. `open` already proved
            // the two are the same recipes, so the conversion is the identity
            // it looks like.
            Ok(Some(hit)) => match F::from_stored(hit.recipe) {
                Some(recipe) => (
                    ArtifactCacheOutcome::Hit { source: hit.source },
                    Some(recipe),
                ),
                None => (
                    ArtifactCacheOutcome::NotStored {
                        reason: "the cached entry is not this recipe's family, and not replaced"
                            .into(),
                    },
                    None,
                ),
            },
            Ok(None) => (
                ArtifactCacheOutcome::NotStored {
                    reason: "the realization did not reach the store".into(),
                },
                None,
            ),
            Err(reason) => (
                ArtifactCacheOutcome::NotStored {
                    reason: format!("unusable cached entry ({reason}), and not replaced"),
                },
                None,
            ),
        };
        (
            ArtifactSlot {
                cache: Some(cache),
                key: Some(key),
                outcome,
            },
            hit,
        )
    }

    fn store<F: Storable>(
        &mut self,
        artifact: &Recipe<F>,
        progress: &TuningProgress,
        validation: &ValidationStatus,
    ) {
        let (Some(cache), Some(key)) = (self.cache, self.key) else {
            return;
        };
        if let Err(reason) = publishable(progress, validation) {
            self.outcome = ArtifactCacheOutcome::NotStored { reason };
            return;
        }
        let Some(artifact) = F::as_stored(artifact) else {
            return;
        };
        self.outcome = match cache.store(&key, artifact) {
            Ok(()) => ArtifactCacheOutcome::Stored,
            Err(reason) => ArtifactCacheOutcome::StoreFailed { reason },
        };
    }
}

/// Whether a realization is the artifact its key promises.
///
/// The key names a program, a workload, a target and an output policy — and
/// says nothing about how long the tuner was given, what this machine could
/// compile, or whether anything was checked against a reference. So a selection
/// the deadline cut short is not that key's answer, neither is one made around
/// a candidate another machine would have measured, and neither is one filtered
/// by a contract the key cannot name. All three run; none is published.
///
/// `ArtifactSlot::open` already withholds the cache from a validated
/// realization, so the validation arm here is the same rule stated where the
/// write happens: a caller who gives the key a validation identity has to pass
/// this too.
fn publishable(progress: &TuningProgress, validation: &ValidationStatus) -> Result<(), String> {
    if !matches!(validation, ValidationStatus::NotRequested) {
        return Err(
            "the selection was filtered by a validation contract the artifact key does not name"
                .into(),
        );
    }
    if !progress.is_complete() {
        return Err("tuning stopped at its deadline before scheduling every candidate".into());
    }
    if progress.unrepresentative_failures > 0 {
        return Err(format!(
            "{} candidates were eliminated by this machine rather than by the program",
            progress.unrepresentative_failures
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progress::ChoiceProgress;

    fn progress(unscheduled: usize, unrepresentative_failures: usize) -> TuningProgress {
        TuningProgress {
            choices: vec![ChoiceProgress {
                selected: 0,
                candidates: 3,
                unscheduled,
                nested: Vec::new(),
            }],
            unrepresentative_failures,
        }
    }

    struct Agrees;

    impl crate::validate::OutputComparator for Agrees {
        fn agrees(&self, _: usize, _: u64, _: u64, _: usize, _: CUstream) -> bool {
            true
        }
    }

    struct Untouched;

    impl crate::validate::OutputInitializer for Untouched {
        fn initialize(&self, _: usize, _: u64, _: usize, _: CUstream) -> Result<(), String> {
            Ok(())
        }
    }

    /// A launching family that is not the default one. What it is bound
    /// through does not matter here: what matters is that it is another
    /// family, and so not what the cache's bytes state.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Elsewhere;

    impl recipe::algebra::Computation for Elsewhere {
        type Authorization = ();
        type Contract = recipe::RecipeSignature;
        type RootContext = CompileRequests;
        type BodyContext = Vec<recipe::KernelModule>;
        type Call = KernelLaunch;
        type RootRemap = recipe::algebra::RequestRemap;
        type BodyRemap = recipe::algebra::ModuleBase;
        type Refusal = recipe::algebra::ExecutableRefusal;

        fn merge_root_context(
            into: &mut CompileRequests,
            incoming: CompileRequests,
        ) -> Self::RootRemap {
            recipe::Executable::merge_root_context(into, incoming)
        }

        fn merge_body_context(
            into: &mut Vec<recipe::KernelModule>,
            incoming: Vec<recipe::KernelModule>,
        ) -> Self::BodyRemap {
            recipe::Executable::merge_body_context(into, incoming)
        }

        fn rebase_body_context(context: &mut Vec<recipe::KernelModule>, remap: &Self::RootRemap) {
            recipe::Executable::rebase_body_context(context, remap);
        }

        fn rebase_call(call: &mut KernelLaunch, remap: &Self::BodyRemap) {
            recipe::Executable::rebase_call(call, remap);
        }

        fn validate_root_context(context: &CompileRequests) -> Result<(), Self::Refusal> {
            recipe::Executable::validate_root_context(context)
        }

        fn validate_body_context(
            context: &Vec<recipe::KernelModule>,
            root: &CompileRequests,
        ) -> Result<(), Self::Refusal> {
            recipe::Executable::validate_body_context(context, root)
        }

        fn validate_call(
            call: &KernelLaunch,
            context: &Vec<recipe::KernelModule>,
        ) -> Result<(), Self::Refusal> {
            recipe::Executable::validate_call(call, context)
        }

        fn launch_arity(call: &KernelLaunch) -> usize {
            recipe::Executable::launch_arity(call)
        }

        fn validate_geometry(
            call: &KernelLaunch,
            launch: usize,
            geometry: &Geometry,
        ) -> Result<(), Self::Refusal> {
            recipe::Executable::validate_geometry(call, launch, geometry)
        }
    }

    impl Launches for Elsewhere {}

    impl Storable for Elsewhere {}

    /// One output, the single word the kernel writes — in whichever launching
    /// family the caller asks for it.
    fn recipe<F: Launches<Contract = recipe::RecipeSignature, Authorization = ()>>() -> Recipe<F> {
        let mut modules = F::BodyContext::default();
        modules
            .modules_mut()
            .push(recipe::KernelModule::from_cuda_loadable(b"image".to_vec()));
        Recipe::stating_edges_authorized(
            recipe::RecipeSignature::new(Vec::new(), vec![4]),
            CompileRequests::default(),
            recipe::RecipeBody::new(
                0,
                1,
                modules,
                Vec::new(),
                vec![recipe::Node::new(
                    recipe::Work::Call {
                        call: KernelLaunch {
                            module: 0,
                            entry: "kernel".into(),
                            args: vec![recipe::Arg::output(0)],
                        },
                        geometry: vec![Geometry {
                            grid: [1, 1, 1],
                            block: [1, 1, 1],
                            dynamic_shmem: 0,
                        }],
                    },
                    [],
                )],
            ),
            &(),
        )
        .unwrap()
    }

    fn device() -> DeviceIdentity {
        DeviceIdentity {
            sm: 100,
            sm_count: 148,
            max_threads_per_sm: 2048,
            max_smem_optin: 232448,
            driver_version: 12080,
        }
    }

    #[test]
    fn invalid_scheduling_limits_are_rejected_before_device_work() {
        let recipe = recipe::<recipe::Executable>();
        let bindings = Bindings {
            inputs: &[],
            outputs: &[0],
            workspace: 0,
        };
        // No CUDA context is needed: policy validation must precede device work.
        let mut realizer = Realizer {
            ctx: std::ptr::null_mut(),
            device: device(),
            options: RealizerOptions::default(),
        };
        for ms in [
            -1.0,
            -f64::from_bits(1),
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
        ] {
            let policy = RealizationPolicy {
                tuning: TuningPolicy {
                    scheduling_time_limit_ms: Some(ms),
                },
                ..RealizationPolicy::default()
            };
            let result = unsafe {
                realizer.realize(RealizationArgs::new(
                    &recipe,
                    &bindings,
                    std::ptr::null_mut(),
                    policy,
                ))
            };
            let error = result.err().expect("invalid limit must fail");
            assert!(matches!(error, Error::InvalidTuningPolicy(_)), "{error:?}");
            assert!(!error.is_program_property());
        }
    }

    #[test]
    fn only_a_finished_representative_tuning_is_published() {
        let unchecked = ValidationStatus::NotRequested;
        assert!(publishable(&progress(3, 0), &unchecked).is_ok());
        assert!(publishable(&TuningProgress::default(), &unchecked).is_ok());
        assert!(
            publishable(&progress(2, 0), &unchecked).is_err(),
            "a deadline-truncated selection is not the key's answer"
        );
        assert!(
            publishable(&progress(3, 1), &unchecked).is_err(),
            "a selection made around this machine's own failure is not the key's answer"
        );
    }

    /// A selection a contract filtered is not what the key promises: the key
    /// names no contract, so it cannot tell a validated artifact from an
    /// unvalidated one.
    #[test]
    fn a_validated_selection_is_never_published_under_a_key_that_omits_the_contract() {
        for checked in [
            ValidationStatus::Checked {
                outputs: 1,
                validated: 2,
                rejected: 1,
            },
            ValidationStatus::Checked {
                outputs: 1,
                validated: 0,
                rejected: 0,
            },
        ] {
            assert!(publishable(&progress(3, 0), &checked).is_err());
        }
    }

    /// The invariant the key cannot express, enforced where it is decided: a
    /// realization that asked for validation neither reads nor writes the
    /// artifact cache. Reading would let it report a comparison nothing made;
    /// writing would offer an unvalidated selection to a caller who asked for
    /// one.
    /// The cache holds recipes in the public wire format, and that format
    /// states the default contract. A realization of another family is
    /// reported as bypassing the cache — never looked up under a key whose
    /// bytes mean something else, and never stored.
    #[test]
    fn only_the_default_family_reaches_the_artifact_cache() {
        let cache = ArtifactCache::open(crate::ArtifactCacheConfig {
            dir: None,
            max_bytes: None,
        });
        let workload = WorkloadIdentity::new("workload");
        let pending = recipe::<Elsewhere>();
        let bindings = Bindings {
            inputs: &[],
            outputs: &[0],
            workspace: 0,
        };
        let args = RealizationArgs {
            authorization: &(),
            starting_state: None,
            recipe: &pending,
            bindings: &bindings,
            stream: std::ptr::null_mut(),
            policy: RealizationPolicy::default(),
            workload: Some(&workload),
            cache: Some(&cache),
            resume: None,
            validation: None,
        };
        let (mut slot, hit) = ArtifactSlot::open(&args, &device());
        assert!(hit.is_none());
        assert!(matches!(
            slot.outcome,
            ArtifactCacheOutcome::Bypassed { .. }
        ));
        slot.store(&pending, &progress(3, 0), &ValidationStatus::NotRequested);
        assert!(
            matches!(slot.outcome, ArtifactCacheOutcome::Bypassed { .. }),
            "a bypassed slot stores nothing either"
        );
    }

    #[test]
    fn requesting_validation_withholds_the_artifact_cache_in_both_directions() {
        let cache = ArtifactCache::open(crate::ArtifactCacheConfig {
            dir: None,
            max_bytes: None,
        });
        let workload = WorkloadIdentity::new("workload");
        let pending = recipe::<recipe::Executable>();
        let bindings = Bindings {
            inputs: &[],
            outputs: &[0],
            workspace: 0,
        };
        let args = |validation| RealizationArgs {
            authorization: &(),
            starting_state: None,
            recipe: &pending,
            bindings: &bindings,
            stream: std::ptr::null_mut(),
            policy: RealizationPolicy::default(),
            workload: Some(&workload),
            cache: Some(&cache),
            resume: None,
            validation,
        };

        // Without validation the cache is live: a miss, then a store.
        let (mut slot, hit) = ArtifactSlot::open(&args(None), &device());
        assert!(hit.is_none());
        assert!(matches!(
            slot.outcome,
            ArtifactCacheOutcome::NotStored { .. }
        ));
        slot.store(&pending, &progress(3, 0), &ValidationStatus::NotRequested);
        assert_eq!(slot.outcome, ArtifactCacheOutcome::Stored);

        // That stored entry was never validated. A later realization that asks
        // for validation must not be handed it, and must not overwrite it.
        let outputs = [crate::ValidationOutput {
            nbytes: 256,
            reference: 1,
            candidate: 2,
        }];
        let contract = ValidationContract {
            outputs: &outputs,
            input_extents: &[],
            comparator: &Agrees,
            initializer: &Untouched,
        };
        let (mut slot, hit) = ArtifactSlot::open(&args(Some(contract)), &device());
        assert!(
            hit.is_none(),
            "an entry stored without validation is not an answer for a caller that wants one"
        );
        assert!(matches!(
            slot.outcome,
            ArtifactCacheOutcome::Bypassed { .. }
        ));
        let checked = ValidationStatus::Checked {
            outputs: 1,
            validated: 2,
            rejected: 1,
        };
        slot.store(&pending, &progress(3, 0), &checked);
        assert!(
            matches!(slot.outcome, ArtifactCacheOutcome::Bypassed { .. }),
            "a bypassed slot stores nothing either"
        );
    }
}
