// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use cuda_driver::{CUcontext, CUdeviceptr, CUstream, Context, Device, StreamRef};
/// The ports a resident recipe is bound through.
///
/// A residency loaded here always came from a whole recipe, which states them;
/// the refusal exists because the type admits one loaded from a body alone, and
/// binding a caller's buffers to work that states no ports would be checking
/// nothing.
fn bound_ports(loaded: &kiln::ResidentRecipe) -> Option<&recipe::RecipeSignature> {
    loaded.contract()
}

const UNBOUND: &str = "resident recipe states no ports to check a binding against";

use kiln::{
    ArtifactCacheOutcome, Bindings, CandidateOutcome, ChoiceReport, CompileFailureKind,
    RealizationArgs, RealizationPolicy, RealizationReport, Realizer, RealizerOptions,
    RequestOutcome, ResidentRecipe, TuneReport, ValidationStatus, check_contract, execute,
    kernel_entries,
};
use recipe::{Recipe, serialize};
use xla_ffi::{Error, RemainingArgs, RemainingRets, Result, Stream, handler};

// A handle lives in the registry behind an `Arc<Mutex<_>>` and turns Pending
// into Resident once, so the idle bytes of a pending handle cost nothing,
// while boxing Resident would add an indirection to every execution.
#[allow(clippy::large_enum_variant)]
pub enum RunnableRecipe {
    Pending {
        source: Arc<Recipe>,
        persist_to: Option<PathBuf>,
        report_to: Option<PathBuf>,
        scheduling_time_limit_ms: Option<f64>,
    },
    Resident(ResidentRecipe),
}

fn persist_flat_recipe(path: &Path, flat: &Recipe) {
    static WRITE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = WRITE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = path.with_extension(format!("tmp.{}.{seq}", std::process::id()));
    let result = serialize(flat)
        .map_err(|e| e.to_string())
        .and_then(|bytes| {
            std::fs::write(&tmp, bytes)
                .and_then(|()| std::fs::rename(&tmp, path))
                .map_err(|e| e.to_string())
        });
    if let Err(e) = result {
        eprintln!(
            "[{}] failed to persist tuned recipe to {}: {e}",
            env!("CARGO_PKG_NAME"),
            path.display()
        );
        let _ = std::fs::remove_file(&tmp);
    }
}

fn json_escape(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{00}'..='\u{1f}' => out.push_str(&format!("\\u{:04x}", c as u32)),
            _ => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::json_escape;

    #[test]
    fn json_strings_escape_every_control_character() {
        let mut out = String::new();
        json_escape(&mut out, "\"\\\u{08}\u{0c}\n\r\t\u{00}\u{01}\u{1f}é");
        assert_eq!(out, "\"\\\"\\\\\\b\\f\\n\\r\\t\\u0000\\u0001\\u001fé\"");
    }
}

fn json_string_array(out: &mut String, items: &[String]) {
    out.push('[');
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        json_escape(out, item);
    }
    out.push(']');
}

fn json_optional_number(out: &mut String, value: Option<impl std::fmt::Display>) {
    match value {
        Some(value) => out.push_str(&value.to_string()),
        None => out.push_str("null"),
    }
}

fn compile_failure_kind(kind: CompileFailureKind) -> &'static str {
    match kind {
        CompileFailureKind::IncompatibleTarget => "incompatible_target",
        CompileFailureKind::IncompatibleAssembly => "incompatible_assembly",
        CompileFailureKind::UnavailableToolchain => "unavailable_toolchain",
        CompileFailureKind::CompilerDiagnostic => "compiler_diagnostic",
        CompileFailureKind::ResourceBound => "resource_bound",
        CompileFailureKind::InvalidImage => "invalid_image",
        CompileFailureKind::CorruptSource => "corrupt_source",
        CompileFailureKind::Interrupted => "interrupted",
    }
}

fn append_realization_report(out: &mut String, report: &RealizationReport) {
    let environment = &report.environment;
    out.push_str(&format!(
        ",\"realization\":{{\"sm\":{},\"environment\":{{\"available_parallelism\":{},\"affinity_cpus\":",
        report.sm, environment.available_parallelism
    ));
    json_optional_number(out, environment.affinity_cpus);
    out.push_str(",\"cgroup_cpu_quota\":");
    json_optional_number(
        out,
        environment.cgroup_cpu_limit.map(|limit| limit.quota_micros),
    );
    out.push_str(",\"cgroup_cpu_period\":");
    json_optional_number(
        out,
        environment
            .cgroup_cpu_limit
            .map(|limit| limit.period_micros),
    );
    out.push_str(",\"host_memory_bytes\":");
    json_optional_number(out, environment.host_memory_bytes);
    out.push_str(",\"cgroup_memory_bytes\":");
    json_optional_number(out, environment.cgroup_memory_bytes);
    out.push_str(",\"configured_worker_ceiling\":");
    json_optional_number(out, environment.configured_worker_ceiling);
    out.push_str(&format!(
        ",\"workers\":{}}},\"peak_queued_source_bytes\":{},\"requests\":[",
        environment.workers, report.peak_queued_source_bytes
    ));
    for (index, request) in report.requests.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&format!("{{\"request\":{},\"outcome\":", request.request));
        json_escape(
            out,
            match request.outcome {
                RequestOutcome::Compiled => "compiled",
                RequestOutcome::SourceLoaded => "source_loaded",
                RequestOutcome::Reused => "reused",
                RequestOutcome::Failed => "failed",
                RequestOutcome::Deferred => "deferred",
            },
        );
        out.push_str(",\"assembler\":");
        match &request.assembler {
            Some(value) => json_escape(out, value),
            None => out.push_str("null"),
        }
        for (name, value) in [
            ("production_cost_ms", request.production_cost_ms),
            ("compile_wall_ms", request.compile_wall_ms),
            ("resource_ms", request.resource_ms),
            ("image_bytes", request.image_bytes.map(|value| value as u64)),
        ] {
            out.push_str(&format!(",\"{name}\":"));
            json_optional_number(out, value);
        }
        out.push_str(",\"cache_source\":");
        match &request.cache_source {
            Some(value) => json_escape(out, value),
            None => out.push_str("null"),
        }
        out.push_str(",\"resources\":[");
        for (resource_index, (entry, resource)) in request.resources.iter().enumerate() {
            if resource_index > 0 {
                out.push(',');
            }
            out.push_str("{\"entry\":");
            json_escape(out, entry);
            out.push_str(&format!(
                ",\"registers\":{},\"local_bytes\":{}}}",
                resource.registers, resource.local_bytes
            ));
        }
        out.push_str("],\"failure\":");
        match &request.failure {
            Some(failure) => {
                out.push_str("{\"kind\":");
                json_escape(out, compile_failure_kind(failure.kind));
                out.push_str(",\"message\":");
                json_escape(out, &failure.message);
                out.push('}');
            }
            None => out.push_str("null"),
        }
        out.push('}');
    }
    let timing = report.timing;
    out.push_str(&format!(
        "],\"timing_ms\":{{\"total\":{},\"tuning\":{},\"artifact\":{},\"artifact_cache\":{},\"compile_blocked\":{},\"compile_wall\":{},\"compile_production_cost\":{},\"resource\":{}}},\"artifact_cache\":",
        timing.total_ms,
        timing.tuning_ms,
        timing.artifact_ms,
        timing.artifact_cache_ms,
        timing.compile_blocked_ms,
        timing.compile_wall_ms,
        timing.compile_production_cost_ms,
        timing.resource_ms,
    ));
    let (cache, cache_reason) = match &report.artifact_cache {
        ArtifactCacheOutcome::Disabled => ("disabled", None),
        ArtifactCacheOutcome::Unidentified { reason } => ("unidentified", Some(reason)),
        ArtifactCacheOutcome::Bypassed { reason } => ("bypassed", Some(reason)),
        ArtifactCacheOutcome::Hit { .. } => ("hit", None),
        ArtifactCacheOutcome::Stored => ("stored", None),
        ArtifactCacheOutcome::NotStored { reason } => ("not_stored", Some(reason)),
        ArtifactCacheOutcome::StoreFailed { reason } => ("store_failed", Some(reason)),
    };
    json_escape(out, cache);
    out.push_str(",\"artifact_cache_reason\":");
    match cache_reason {
        Some(reason) => json_escape(out, reason),
        None => out.push_str("null"),
    }
    out.push_str(",\"validation\":");
    match &report.validation {
        ValidationStatus::NotRequested => json_escape(out, "not_requested"),
        ValidationStatus::Checked {
            outputs,
            validated,
            rejected,
        } => out.push_str(&format!(
            "{{\"outputs\":{outputs},\"validated\":{validated},\"rejected\":{rejected}}}"
        )),
    }
    out.push('}');
}

fn write_realization_report(
    path: &Path,
    tune: Option<&TuneReport>,
    report: &RealizationReport,
    flat: &Recipe,
) {
    // Fresh measurements only: memo-cached candidates cost no GPU time.
    let choices = tune.map_or(&[][..], |report| report.choices.as_slice());
    let num_timed: usize = choices
        .iter()
        .flat_map(|c| &c.candidates)
        .filter(|c| matches!(c.outcome, CandidateOutcome::Measured { cached: false, .. }))
        .count();

    let mut out = String::new();
    out.push_str(&format!("{{\"num_timed\":{num_timed},\"choices\":["));
    for (i, choice) in choices.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let ChoiceReport {
            depth,
            winner,
            candidates,
            ..
        } = choice;
        out.push_str(&format!("{{\"depth\":{depth},\"winner\":"));
        match winner {
            Some(w) => out.push_str(&w.to_string()),
            None => out.push_str("null"),
        }
        out.push_str(",\"candidates\":[");
        for (j, cand) in candidates.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            let (ns, cached) = match &cand.outcome {
                CandidateOutcome::Measured { ns, cached } => (Some(*ns), *cached),
                _ => (None, false),
            };
            out.push_str(&format!("{{\"candidate\":{},\"ns\":", cand.candidate));
            match ns {
                Some(ns) => out.push_str(&ns.to_string()),
                None => out.push_str("null"),
            }
            out.push_str(&format!(",\"cached\":{cached}"));
            out.push_str(",\"status\":");
            json_escape(
                &mut out,
                match &cand.outcome {
                    CandidateOutcome::Measured { .. } => "measured",
                    CandidateOutcome::Unmeasured => "unmeasured",
                    CandidateOutcome::Pruned => "pruned",
                    CandidateOutcome::Infeasible(failure) if failure.of_the_program => "infeasible",
                    CandidateOutcome::Infeasible(_) => "unavailable",
                },
            );
            out.push_str(",\"entries\":");
            json_string_array(&mut out, &cand.entries);
            out.push_str(",\"failure\":");
            match &cand.outcome {
                CandidateOutcome::Infeasible(failure) => json_escape(&mut out, &failure.message),
                _ => out.push_str("null"),
            }
            out.push('}');
        }
        out.push_str("]}");
    }
    out.push_str("],\"final_entries\":");
    json_string_array(&mut out, &kernel_entries(flat));
    append_realization_report(&mut out, report);
    out.push('}');

    if let Err(e) = std::fs::write(path, out) {
        eprintln!(
            "[{}] failed to write tune report to {}: {e}",
            env!("CARGO_PKG_NAME"),
            path.display()
        );
    }
}

unsafe fn realize_pending(
    ctx: CUcontext,
    source: &Recipe,
    bindings: &Bindings<'_>,
    stream: CUstream,
    persist_to: Option<&PathBuf>,
    report_to: Option<&PathBuf>,
    scheduling_time_limit_ms: Option<f64>,
) -> std::result::Result<ResidentRecipe, String> {
    let mut realizer = Realizer::new(ctx, RealizerOptions::default())
        .map_err(|error| format!("realizer setup failed: {error:?}"))?;
    let result = unsafe {
        realizer.realize(RealizationArgs::new(
            source,
            bindings,
            stream,
            RealizationPolicy {
                collect_tuning_report: report_to.is_some(),
                tuning: kiln::TuningPolicy {
                    scheduling_time_limit_ms,
                },
                ..RealizationPolicy::default()
            },
        ))
    }
    .map_err(|error| format!("realization failed: {error:?}"))?;
    if let Some(path) = report_to {
        write_realization_report(
            path,
            result.tuning.as_ref(),
            &result.report,
            &result.artifact,
        );
    }
    if let Some(path) = persist_to {
        persist_flat_recipe(path, &result.artifact);
    }
    Ok(result.resident)
}

/// What a caller bound for one side, as pointers and the bytes reachable from
/// each. The two travel together because a pointer alone cannot be checked.
pub struct BoundSide<'a> {
    pub ptrs: &'a [CUdeviceptr],
    pub nbytes: &'a [usize],
}

/// The extents of both sides, refusing a caller that gave a different number of
/// sizes than pointers — that would leave a buffer unchecked rather than
/// checked and wrong.
fn bound_extents<'a>(
    inputs: &BoundSide<'a>,
    outputs: &BoundSide<'a>,
) -> std::result::Result<(), String> {
    for (side, bound) in [("input", inputs), ("output", outputs)] {
        if bound.ptrs.len() != bound.nbytes.len() {
            return Err(format!(
                "{} pointers and sizes disagree: {} pointers, {} sizes",
                side,
                bound.ptrs.len(),
                bound.nbytes.len()
            ));
        }
    }
    Ok(())
}

pub unsafe fn run_recipe_core(
    handle: &Mutex<RunnableRecipe>,
    stream: CUstream,
    workspace_ptr: CUdeviceptr,
    workspace_size: usize,
    inputs: BoundSide<'_>,
    outputs: BoundSide<'_>,
) -> std::result::Result<(), String> {
    bound_extents(&inputs, &outputs)?;
    let (in_ptrs, out_ptrs) = (inputs.ptrs, outputs.ptrs);
    let mut state = handle
        .lock()
        .map_err(|_| "recipe handle lock poisoned".to_string())?;

    if let RunnableRecipe::Pending {
        source,
        persist_to,
        report_to,
        scheduling_time_limit_ms,
    } = &*state
    {
        // Tuning times candidates against the real bindings, so the buffer
        // must already cover the pending recipe's worst selection.
        let required = recipe::algebra::workspace_upper_bound(source);
        if workspace_size < required {
            return Err(format!(
                "workspace is {workspace_size} bytes, recipe needs {required}"
            ));
        }
        // Realization launches these kernels to time them, so the buffers must
        // already be the ones the pending recipe states.
        check_contract(source.contract(), inputs.nbytes, outputs.nbytes)
            .map_err(|error| format!("pending recipe: {error}"))?;
        let ctx = unsafe { set_stream_context_str(stream) }?;
        let bindings = Bindings {
            inputs: in_ptrs,
            outputs: out_ptrs,
            workspace: workspace_ptr,
        };
        let resident = unsafe {
            realize_pending(
                ctx,
                source,
                &bindings,
                stream,
                persist_to.as_ref(),
                report_to.as_ref(),
                *scheduling_time_limit_ms,
            )
        }?;
        *state = RunnableRecipe::Resident(resident);
    }
    let RunnableRecipe::Resident(loaded) = &mut *state else {
        unreachable!("just transitioned to Resident");
    };

    // The resident recipe declares its full workspace (including the winning
    // candidates' formerly-private scratch); validate the buffer covers it.
    let required = loaded.workspace_upper_bound();
    if workspace_size < required {
        return Err(format!(
            "workspace is {workspace_size} bytes, recipe needs {required}"
        ));
    }
    // Every replay is checked, not just the first: a resident recipe outlives
    // the call that realized it and the next caller may bind anything.
    check_contract(
        bound_ports(loaded).ok_or_else(|| UNBOUND.to_string())?,
        inputs.nbytes,
        outputs.nbytes,
    )
    .map_err(|error| format!("resident recipe: {error}"))?;

    let bindings = Bindings {
        inputs: in_ptrs,
        outputs: out_ptrs,
        workspace: workspace_ptr,
    };
    unsafe { execute(loaded, &bindings, stream) }.map_err(|e| format!("execute failed: {e:?}"))?;
    Ok(())
}

unsafe fn set_stream_context(stream: CUstream) -> Result<CUcontext> {
    let ctx = StreamRef::from_raw(stream)
        .context()
        .map_err(|e| Error::internal(format!("cuStreamGetCtx failed: {e}")))?;
    ctx.set_current()
        .map_err(|e| Error::internal(format!("cuCtxSetCurrent failed: {e}")))?;
    Ok(ctx.raw())
}

unsafe fn set_stream_context_str(stream: CUstream) -> std::result::Result<CUcontext, String> {
    // `cuStreamGetCtx` on the default stream (NULL) returns whatever context is
    // *currently current* on this thread, not a fixed owner. Between successive
    // calls, libraries (cuFFTDx, libmathdx, nvjitlink, …) may leave a different
    // context current. Pin to the primary context of the stream's device instead
    // — that pointer is stable across the process lifetime and matches torch's
    // own primary context for its tensors and legacy-default stream.
    let owner = StreamRef::from_raw(stream)
        .context()
        .map_err(|e| format!("cuStreamGetCtx failed: {e}"))?;
    owner
        .set_current()
        .map_err(|e| format!("cuCtxSetCurrent failed: {e}"))?;

    // Retain is refcounted, so calling it on every dispatch is safe and
    // idempotent. We do *not* release on exit: resident recipes live for the
    // lifetime of the process and reference this context.
    let primary = Context::primary(Device::current())
        .map_err(|e| format!("cuDevicePrimaryCtxRetain failed: {e}"))?;
    primary
        .set_current()
        .map_err(|e| format!("cuCtxSetCurrent (primary) failed: {e}"))?;
    Ok(primary.raw())
}

#[handler]
pub fn run_recipe(
    stream: Stream,
    #[attr("recipe")] recipe: &[u8],
    inputs: RemainingArgs<'_>,
    outputs: RemainingRets<'_>,
) -> Result<()> {
    let handle = crate::registry::resolve(recipe).map_err(Error::invalid_argument)?;

    // Each buffer's own extent travels with its pointer: XLA states both, and
    // a pointer alone cannot be checked against the recipe.
    let mut in_ptrs: Vec<CUdeviceptr> = Vec::with_capacity(inputs.len());
    let mut in_nbytes: Vec<usize> = Vec::with_capacity(inputs.len());
    for b in inputs.iter() {
        in_ptrs.push(b.data() as CUdeviceptr);
        in_nbytes.push(b.size_bytes()?);
    }
    let mut raw_out_ptrs: Vec<CUdeviceptr> = Vec::with_capacity(outputs.len());
    let mut raw_out_nbytes: Vec<usize> = Vec::with_capacity(outputs.len());
    for b in outputs.iter() {
        raw_out_ptrs.push(b.data() as CUdeviceptr);
        raw_out_nbytes.push(b.size_bytes()?);
    }

    // Output 0 is the workspace this call allocates for the recipe, not a port.
    let Some((&workspace_base, real_out_ptrs)) = raw_out_ptrs.split_first() else {
        return Err(Error::invalid_argument("missing workspace output"));
    };
    let (&workspace_size, real_out_nbytes) = raw_out_nbytes
        .split_first()
        .expect("outputs are non-empty past the split");

    let stream = stream.0 as CUstream;

    let mut state = handle
        .lock()
        .map_err(|_| Error::internal("recipe handle lock poisoned"))?;

    if let RunnableRecipe::Pending {
        source,
        persist_to,
        report_to,
        scheduling_time_limit_ms,
    } = &*state
    {
        // Tuning times candidates against the real bindings, so the buffer
        // must already cover the pending recipe's worst selection.
        let required = recipe::algebra::workspace_upper_bound(source);
        if workspace_size < required {
            return Err(Error::resource_exhausted(format!(
                "workspace output is {workspace_size} bytes, recipe needs {required}"
            )));
        }
        // Realization launches these kernels to time them.
        check_contract(source.contract(), &in_nbytes, real_out_nbytes)
            .map_err(|error| Error::invalid_argument(format!("pending recipe: {error}")))?;
        // Not while the caller is recording the stream into a graph. Tuning
        // decides between candidates by running them and timing them, and a
        // captured launch neither runs nor can be timed — the events it would
        // be measured with are not allowed to be waited on. The buffers are no
        // better: XLA documents that a handler being recorded may be handed
        // uninitialized arguments, so even a timing that worked would be a
        // timing of the wrong thing, baked into every later replay.
        //
        // Nothing should reach here, because XLA runs a custom call once
        // outside any command buffer before it records one, and that first
        // execution is what realizes. Saying so anyway costs one host query
        // and turns a silently mistuned recipe into a refusal.
        if StreamRef::from_raw(stream)
            .is_capturing()
            .map_err(|error| Error::internal(format!("{error:?}")))?
        {
            return Err(Error::internal(
                "recipe is unrealized and the stream is capturing: \
                 autotuning cannot run inside a graph capture",
            ));
        }
        let ctx = unsafe { set_stream_context(stream)? };
        let bindings = Bindings {
            inputs: &in_ptrs,
            outputs: real_out_ptrs,
            workspace: workspace_base,
        };
        let resident = unsafe {
            realize_pending(
                ctx,
                source,
                &bindings,
                stream,
                persist_to.as_ref(),
                report_to.as_ref(),
                *scheduling_time_limit_ms,
            )
        }
        .map_err(Error::internal)?;
        *state = RunnableRecipe::Resident(resident);
    }
    let RunnableRecipe::Resident(loaded) = &mut *state else {
        unreachable!("just transitioned to Resident");
    };

    // The resident recipe declares its full workspace (including the winning
    // candidates' formerly-private scratch); validate the XLA buffer covers it.
    let required = loaded.workspace_upper_bound();
    if workspace_size < required {
        return Err(Error::resource_exhausted(format!(
            "workspace output is {workspace_size} bytes, recipe needs {required}"
        )));
    }
    // Every replay, not just the first: a resident recipe outlives the call
    // that realized it and the next caller may bind anything.
    check_contract(
        bound_ports(loaded).ok_or_else(|| Error::invalid_argument(UNBOUND))?,
        &in_nbytes,
        real_out_nbytes,
    )
    .map_err(|error| Error::invalid_argument(format!("resident recipe: {error}")))?;

    let bindings = Bindings {
        inputs: &in_ptrs,
        outputs: real_out_ptrs,
        workspace: workspace_base,
    };
    unsafe { execute(loaded, &bindings, stream) }
        .map_err(|e| Error::internal(format!("execute failed: {e:?}")))?;
    Ok(())
}

pub(crate) fn tuned_recipe_bytes(
    handle: &Mutex<RunnableRecipe>,
) -> std::result::Result<Option<Vec<u8>>, recipe::WireError> {
    match &*handle.lock().expect("recipe handle lock poisoned") {
        RunnableRecipe::Pending { .. } => Ok(None),
        RunnableRecipe::Resident(resident) => match resident.recipe() {
            Some(recipe) => serialize(recipe).map(Some),
            None => Ok(None),
        },
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use recipe::{Node, RecipeBody, RecipeSignature, Work, WritableBuf};

    /// A recipe that touches no device: the binding check runs before any CUDA
    /// call, so every rejection below is reached without a GPU.
    fn pending(signature: RecipeSignature) -> Mutex<RunnableRecipe> {
        Mutex::new(RunnableRecipe::Pending {
            source: recipe::Recipe::stating_edges(
                signature.clone(),
                recipe::CompileRequests::default(),
                RecipeBody::new(
                    signature.num_inputs(),
                    signature.num_outputs(),
                    Vec::new(),
                    Vec::new(),
                    vec![Node::new(
                        Work::Memset {
                            destination: WritableBuf::Output(0).at(0),
                            value: 0,
                            nbytes: 16,
                        },
                        [],
                    )],
                ),
            )
            .expect("the fixture is a recipe")
            .into(),
            persist_to: None,
            report_to: None,
            scheduling_time_limit_ms: None,
        })
    }

    /// Two inputs of different sizes and one output, so no mistake lands on
    /// another port's number.
    fn signature() -> RecipeSignature {
        RecipeSignature::new(vec![48, 96], vec![16])
    }

    fn run(in_nbytes: &[usize], out_nbytes: &[usize]) -> std::result::Result<(), String> {
        let handle = pending(signature());
        let in_ptrs = vec![0 as CUdeviceptr; in_nbytes.len()];
        let out_ptrs = vec![0 as CUdeviceptr; out_nbytes.len()];
        unsafe {
            run_recipe_core(
                &handle,
                std::ptr::null_mut(),
                0 as CUdeviceptr,
                1 << 20,
                BoundSide {
                    ptrs: &in_ptrs,
                    nbytes: in_nbytes,
                },
                BoundSide {
                    ptrs: &out_ptrs,
                    nbytes: out_nbytes,
                },
            )
        }
    }

    #[test]
    fn an_undersized_input_never_reaches_the_device() {
        let error = run(&[48, 95], &[16]).unwrap_err();
        assert_eq!(
            error,
            "pending recipe: input 1 is 95 bytes, recipe addresses 96"
        );
    }

    #[test]
    fn an_undersized_output_never_reaches_the_device() {
        let error = run(&[48, 96], &[15]).unwrap_err();
        assert_eq!(
            error,
            "pending recipe: output 0 is 15 bytes, recipe addresses 16"
        );
    }

    #[test]
    fn too_few_inputs_are_refused() {
        let error = run(&[48], &[16]).unwrap_err();
        assert_eq!(
            error,
            "pending recipe: recipe states 2 input buffer(s), 1 bound"
        );
    }

    #[test]
    fn too_many_outputs_are_refused() {
        let error = run(&[48, 96], &[16, 16]).unwrap_err();
        assert_eq!(
            error,
            "pending recipe: recipe states 1 output buffer(s), 2 bound"
        );
    }

    /// A caller that hands sizes for some pointers and not others would leave
    /// the rest unchecked, which is worse than checking them wrong.
    #[test]
    fn pointers_without_sizes_are_refused() {
        let handle = pending(signature());
        let error = unsafe {
            run_recipe_core(
                &handle,
                std::ptr::null_mut(),
                0 as CUdeviceptr,
                1 << 20,
                BoundSide {
                    ptrs: &[0 as CUdeviceptr, 0 as CUdeviceptr],
                    nbytes: &[48],
                },
                BoundSide {
                    ptrs: &[0 as CUdeviceptr],
                    nbytes: &[16],
                },
            )
        }
        .unwrap_err();
        assert_eq!(
            error,
            "input pointers and sizes disagree: 2 pointers, 1 sizes"
        );
    }

    /// The workspace is still checked, and before the ports: it is the buffer
    /// tuning needs in order to run at all.
    #[test]
    fn the_workspace_is_still_checked() {
        let handle = Mutex::new(RunnableRecipe::Pending {
            source: recipe::Recipe::stating_edges(
                signature(),
                recipe::CompileRequests::default(),
                RecipeBody::new(
                    2,
                    1,
                    Vec::new(),
                    vec![recipe::WorkspaceBuf {
                        nbytes: 4096,
                        init: None,
                    }],
                    Vec::new(),
                ),
            )
            .expect("the fixture is a recipe")
            .into(),
            persist_to: None,
            report_to: None,
            scheduling_time_limit_ms: None,
        });
        let error = unsafe {
            run_recipe_core(
                &handle,
                std::ptr::null_mut(),
                0 as CUdeviceptr,
                8,
                BoundSide {
                    ptrs: &[0 as CUdeviceptr, 0 as CUdeviceptr],
                    nbytes: &[48, 96],
                },
                BoundSide {
                    ptrs: &[0 as CUdeviceptr],
                    nbytes: &[16],
                },
            )
        }
        .unwrap_err();
        assert!(error.starts_with("workspace is 8 bytes"), "{error}");
    }

    /// Exactly the stated extents, and a larger buffer, both pass the check —
    /// so the run proceeds to the device, which is where a test without a GPU
    /// stops. Reaching a CUDA error proves the binding check let it through.
    #[test]
    fn exact_and_oversized_bindings_pass_the_check() {
        for (inputs, outputs) in [
            (vec![48usize, 96], vec![16usize]),
            (vec![1 << 20, 1 << 20], vec![1 << 20]),
        ] {
            let error = run(&inputs, &outputs).unwrap_err();
            assert!(
                !error.contains("recipe states") && !error.contains("recipe addresses"),
                "binding {inputs:?} -> {outputs:?} should pass the check, got: {error}"
            );
        }
    }
}
