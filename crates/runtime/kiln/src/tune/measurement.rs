// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::bind::{Ports, range_start};
use crate::exec::run;
use crate::resolve::hash_resolved_modules;
use crate::{Error, ResidentRecipe};
use cuda_driver::{CUstream, Event, StreamRef};
use recipe::algebra::Launches;
use recipe::{Geometry, KernelLaunch, RecipeBody, Segment, Work};
use std::hash::{DefaultHasher, Hash, Hasher};

const TARGET_BATCH_NS: u64 = 250_000;

const MAX_PACK: usize = 64;

const WARMUP: usize = 3;

const BATCHES: usize = 5;

/// A completed measurement went through every batch; an aborted one carries a
/// ceiling-truncated value that must not be reused for another choice.
pub(super) unsafe fn time_recipe<F: Launches>(
    loaded: &mut ResidentRecipe<F>,
    ports: &Ports,
    stream: CUstream,
    probe_ceil: Option<u64>,
    batch_ceil: Option<u64>,
) -> std::result::Result<(u64, bool), Error> {
    let _measurement = chronicle::range(|| "kiln.measure".into());
    let stream_ref = StreamRef::from_raw(stream);
    let start = Event::new().map_err(Error::Driver)?;
    let stop = Event::new().map_err(Error::Driver)?;

    let bracket = |loaded: &mut ResidentRecipe<F>,
                   n: usize,
                   phase: &str|
     -> std::result::Result<u64, Error> {
        let _batch = chronicle::range(|| format!("kiln.batch phase={phase} executions={n}"));
        start.record(stream_ref).map_err(Error::Driver)?;
        for _ in 0..n {
            unsafe { run(loaded, ports, stream) }?;
        }
        stop.record(stream_ref).map_err(Error::Driver)?;
        stop.synchronize().map_err(Error::Driver)?;
        let ms = stop.elapsed_since(&start).map_err(Error::Driver)?;
        chronicle::mark(|| {
            format!(
                "kiln.batch_result phase={phase} executions={n} elapsed_ns={}",
                (ms as f64 * 1.0e6) as u64
            )
        });
        Ok((ms as f64 * 1.0e6 / n as f64) as u64)
    };

    let probe = bracket(loaded, 1, "probe")?;
    if probe_ceil.is_some_and(|c| probe > c) {
        chronicle::mark(|| format!("kiln.measure_result status=probe_pruned ns={probe}"));
        return Ok((probe, false));
    }
    let pack = (TARGET_BATCH_NS / probe.max(1)).clamp(1, MAX_PACK as u64) as usize;

    let warmup = chronicle::range(|| format!("kiln.warmup executions={WARMUP}"));
    for _ in 0..WARMUP {
        unsafe { run(loaded, ports, stream) }?;
    }
    drop(warmup);
    let mut samples = [0; BATCHES];
    let mut min_ns = u64::MAX;
    for sample in &mut samples {
        *sample = bracket(loaded, pack, "sample")?;
        min_ns = min_ns.min(*sample);
        if batch_ceil.is_some_and(|ceiling| min_ns > ceiling) {
            chronicle::mark(|| format!("kiln.measure_result status=batch_pruned ns={min_ns}"));
            return Ok((min_ns, false));
        }
    }
    samples.sort_unstable();
    chronicle::mark(|| {
        format!(
            "kiln.measure_result status=complete ns={}",
            samples[BATCHES / 2]
        )
    });
    Ok((samples[BATCHES / 2], true))
}

/// Two candidates with identical execution content — modules, workspace,
/// nodes — bound to identical resolved addresses are the same program, so the
/// first complete measurement serves every recurrence. `None` when a binding
/// does not resolve (the run would fail the same way).
pub(super) fn candidate_key<F: Launches>(flat: &RecipeBody<F>, ports: &Ports) -> Option<u64> {
    let mut hasher = DefaultHasher::new();
    hash_resolved_modules(flat, &mut hasher);
    for buf in flat.workspace() {
        buf.nbytes.hash(&mut hasher);
        buf.init.is_some().hash(&mut hasher);
    }
    // Payload bytes from the one walk that says which payloads a body carries,
    // rather than a second list of the places they live: a role this misses is
    // a candidate measured under another candidate's timing.
    for payload in recipe::payloads::body_payloads(flat) {
        payload.role.tag().hash(&mut hasher);
        payload.blob.content_digest().ok()?.hash(&mut hasher);
    }
    for node in flat.nodes() {
        node.deps().hash(&mut hasher);
        match node.work() {
            Work::Call {
                call:
                    KernelLaunch {
                        module,
                        entry,
                        args,
                    },
                geometry,
            } => {
                let [
                    Geometry {
                        grid,
                        block,
                        dynamic_shmem: shmem,
                    },
                ] = geometry[..]
                else {
                    return None;
                };
                (0u8, module, entry, grid, block, shmem).hash(&mut hasher);
                for arg in args {
                    for segment in arg.segments() {
                        match segment {
                            Segment::Bytes(bytes) => (0u8, bytes).hash(&mut hasher),
                            Segment::Pointer(address) => {
                                (1u8, range_start(ports, *address, 0).ok()?).hash(&mut hasher);
                            }
                        }
                    }
                }
            }
            Work::Memset {
                destination,
                value,
                nbytes,
            } => (
                1u8,
                range_start(ports, *destination, *nbytes).ok()?,
                value,
                nbytes,
            )
                .hash(&mut hasher),
            Work::Memcpy {
                source,
                destination,
                nbytes,
            } => (
                2u8,
                range_start(ports, *source, *nbytes).ok()?,
                range_start(ports, *destination, *nbytes).ok()?,
                nbytes,
            )
                .hash(&mut hasher),
            Work::Choice { .. } => return None,
        }
    }
    Some(hasher.finish())
}
