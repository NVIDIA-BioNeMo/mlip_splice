// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use cuda_driver::{CUcontext, CUdeviceptr, CUstream, Context, DeviceBuffer, Event, Kernel, Module};

use recipe::algebra::{Launches, ModuleTable};
use recipe::{BufRef, FlatRecipe, KernelLaunch, Recipe, RecipeBody, Segment, Work};

use crate::Error;
use crate::bind::Ports;
use crate::exec::{COPY_ENTRY, COPY_PTX, ExecSlot, GRAPH_RING_DEPTH, GraphRing, build_graph};

const STATIC_SHMEM_CAP: u32 = 49152;

fn apply_dynamic_shmem(kernel: &Kernel, shmem: u32) -> Result<(), Error> {
    if shmem <= STATIC_SHMEM_CAP {
        return Ok(());
    }
    kernel
        .set_max_dynamic_shared(shmem as i32)
        .map_err(Error::Driver)
}

/// A recipe loaded into a CUDA graph, ready to be replayed.
///
/// What it holds is the work: a body whose modules are loaded and whose nodes
/// are graph nodes waiting to be repointed. The ports are held beside it only
/// when there are ports to hold — a residency loaded from a whole recipe is
/// replayed by a caller binding its own buffers, and states what those must be;
/// one loaded from a body alone is a candidate being measured inside the recipe
/// that offers it, whose slots the tuner binds itself.
pub struct ResidentRecipe<F: Launches = recipe::Executable> {
    ctx: CUcontext,
    pub(crate) held: Loaded<F>,
    // What the work replayed here declares, proved to have a layout as it
    // loaded. A body carries no such proof of its own, so this is where one
    // starts existing for the bodies a tuner hands over.
    workspace_nbytes: usize,
    // The last stream whose context was checked against `ctx`. A stream cannot
    // change context, so the check is per stream and not per replay.
    pub(crate) checked_stream: Option<CUstream>,
    // Each workspace buffer's declared initial bytes, held on the device.
    // Indexed alongside `body().workspace()`, `None` where a buffer declares
    // no initial contents. Uploaded once here so a replay seeds the workspace
    // device-to-device: the copy then names no host address that a later
    // replay, or a graph recording this one, could find stale.
    pub(crate) seeds: Vec<Option<DeviceBuffer<u8>>>,
    // Field order is the drop order: the graph (and its execs/events) must be
    // freed before the kernels and modules whose code its nodes reference.
    pub(crate) graph: GraphRing,
    pub(crate) functions: HashMap<(usize, String), Kernel>,
    // The builtin kernel realizing `Work::Memcpy` nodes (owns its module).
    pub(crate) copy: Option<Kernel>,
    // Kept alive so the kernels (and their graph nodes) stay loaded.
    _modules: Vec<Module>,
}

/// What a residency was loaded from.
///
/// A whole recipe states the ports every replay must check its buffers
/// against, and can be handed back as the bytes it is. A body alone is a
/// candidate being measured inside the recipe that offers it: its slots are
/// bound by whatever selected it, so it has no ports of its own and there is
/// no artifact to write. Holding one or the other is what keeps the ports from
/// being invented for work that has none.
pub(crate) enum Loaded<F: Launches> {
    Whole(Recipe<F>),
    Work(RecipeBody<F>),
}

impl<F: Launches> Loaded<F> {
    pub(crate) fn body(&self) -> &RecipeBody<F> {
        match self {
            Loaded::Whole(recipe) => recipe.body(),
            Loaded::Work(body) => body,
        }
    }
}

// SAFETY: everything held here is a CUDA driver handle — a context, graph
// execs, loaded modules and their kernels — and the driver API is thread-safe
// for all of them: a handle is valid in whichever thread names it, and its
// context is made current per call rather than bound to a thread. Only the raw
// `CUcontext` and the handles inside `GraphRing` lack the marker, and they are
// what the driver documents as shareable. Executing still needs `&mut`, so one
// realization is used by one thread at a time whoever owns it.
unsafe impl<F: Launches> Send for ResidentRecipe<F> {}

impl<F: Launches> ResidentRecipe<F> {
    /// A residency over a whole recipe, bound through the ports it states.
    pub unsafe fn new(ctx: CUcontext, flat: FlatRecipe<F>) -> Result<Self, Error> {
        unsafe { Self::load(ctx, Loaded::Whole(flat.into_recipe())) }
    }

    /// A residency over a body alone, for work whose slots whatever selected it
    /// binds. It states no ports, because a body has none.
    pub(crate) unsafe fn over_body(ctx: CUcontext, body: RecipeBody<F>) -> Result<Self, Error> {
        unsafe { Self::load(ctx, Loaded::Work(body)) }
    }

    unsafe fn load(ctx: CUcontext, loaded: Loaded<F>) -> Result<Self, Error> {
        let body = loaded.body();
        // Before any of it is loaded: work whose declared workspace leaves the
        // address space has no allocation that could back it, and every reply
        // to "how much does this need" below would otherwise be a lie or a
        // panic.
        let workspace_nbytes = recipe::algebra::checked_body_workspace_upper_bound(body).ok_or(
            Error::InvalidRecipe(recipe::RecipeError::WorkspaceLayoutOverflow),
        )?;
        let context = Context::from_raw(ctx);
        context.set_current().map_err(Error::Driver)?;

        let mut modules: Vec<Module> = Vec::with_capacity(body.context().modules().len());
        for (index, module) in body.context().modules().iter().enumerate() {
            let image = module
                .sole_image()
                .ok_or(Error::UnresolvedModule { module: index })?;
            modules.push(Module::load(image.bytes()).map_err(Error::Driver)?);
        }

        let mut functions: HashMap<(usize, String), Kernel> = HashMap::new();
        for node in body.nodes() {
            if let Work::Call {
                call: KernelLaunch { module, entry, .. },
                geometry,
                ..
            } = node.work()
            {
                let key = (*module, entry.clone());
                if functions.contains_key(&key) {
                    continue;
                }
                let kernel = modules[*module]
                    .kernel(entry)
                    .map_err(|_| Error::MissingSymbol(entry.clone()))?;
                for shape in geometry {
                    apply_dynamic_shmem(&kernel, shape.dynamic_shmem)?;
                }
                functions.insert(key, kernel);
            }
        }

        // The builtin copy kernel backs `Work::Memcpy` nodes; a recipe
        // without any skips the module load entirely.
        let copy = body
            .nodes()
            .iter()
            .any(|n| matches!(n.work(), Work::Memcpy { .. }))
            .then(|| {
                Module::load(COPY_PTX.as_bytes())
                    .map_err(Error::Driver)?
                    .kernel(COPY_ENTRY)
                    .map_err(|_| Error::MissingSymbol(COPY_ENTRY.to_string()))
            })
            .transpose()?;

        // The declared initial contents, moved to the device once. A replay
        // seeds from here rather than from the recipe's blob, so the copy it
        // issues carries no host address.
        let seeds = body
            .workspace()
            .iter()
            .map(|buf| {
                buf.init
                    .as_ref()
                    .map(|init| DeviceBuffer::<u8>::from_host(ctx, &init[..]))
                    .transpose()
                    .map_err(Error::Driver)
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let placeholder = PlaceholderPorts::for_body(ctx, body)?;
        let ports = placeholder.ports();
        let copy_fn = copy.as_ref().map(|k| k.function());
        let (graph, nodes) = build_graph(body, &functions, copy_fn, &ports, context)?;
        let mut execs: Vec<ExecSlot> = Vec::with_capacity(GRAPH_RING_DEPTH);
        for _ in 0..GRAPH_RING_DEPTH {
            let exec = graph.instantiate().map_err(Error::Driver)?;
            execs.push(ExecSlot {
                exec,
                done: Event::new_disable_timing().map_err(Error::Driver)?,
                launched_on: None,
                bound: None,
            });
        }
        let graph = GraphRing {
            _graph: graph,
            nodes,
            execs,
            current: 0,
        };

        Ok(Self {
            ctx,
            held: loaded,
            workspace_nbytes,
            checked_stream: None,
            seeds,
            graph,
            functions,
            copy,
            _modules: modules,
        })
    }

    pub(crate) fn ctx(&self) -> CUcontext {
        self.ctx
    }

    /// The work this residency replays.
    pub fn body(&self) -> &RecipeBody<F> {
        self.held.body()
    }

    /// How large the workspace a replay binds must be.
    ///
    /// A number rather than a maybe because loading proved it is one: the work
    /// held here declares a workspace that fits in an address, or this
    /// residency does not exist.
    pub fn workspace_upper_bound(&self) -> usize {
        self.workspace_nbytes
    }

    /// The whole recipe this was loaded from, when it was loaded from one.
    pub fn recipe(&self) -> Option<&Recipe<F>> {
        match &self.held {
            Loaded::Whole(recipe) => Some(recipe),
            Loaded::Work(_) => None,
        }
    }

    /// The ports this residency is bound through, when it was loaded from a
    /// recipe that states them.
    ///
    /// Realization chooses which candidates run; it does not change what the
    /// program is bound through, so these are the extents every replay must
    /// check its buffers against. A residency loaded from a body alone has
    /// none, and nothing replays one of those through the execution boundary.
    pub fn contract(&self) -> Option<&F::Contract> {
        self.recipe().map(Recipe::contract)
    }
}

impl<F: Launches> Drop for ResidentRecipe<F> {
    fn drop(&mut self) {
        // Wait out any in-flight launch so the events and graph execs are not
        // freed while the device still references them. The owned Event,
        // GraphExec, Graph and Module values free themselves afterward.
        for slot in &self.graph.execs {
            if slot.launched_on.is_some() {
                let _ = slot.done.synchronize();
            }
        }
    }
}

struct PlaceholderPorts {
    inputs: Vec<CUdeviceptr>,
    outputs: Vec<CUdeviceptr>,
    workspace: Vec<CUdeviceptr>,
    _dummy: DeviceBuffer<u8>,
}

/// How many ports of each kind a buffer reference implies exist.
fn note_port(buf: BufRef, n_inputs: &mut usize, n_outputs: &mut usize) {
    match buf {
        BufRef::Input(i) => *n_inputs = (*n_inputs).max(i + 1),
        BufRef::Output(i) => *n_outputs = (*n_outputs).max(i + 1),
        BufRef::Workspace(_) => {}
    }
}

impl PlaceholderPorts {
    fn for_body<F: Launches>(ctx: CUcontext, body: &RecipeBody<F>) -> Result<Self, Error> {
        let mut n_inputs = 0usize;
        let mut n_outputs = 0usize;
        let mut dummy_bytes = 256usize;
        for node in body.nodes() {
            match node.work() {
                Work::Call {
                    call: KernelLaunch { args, .. },
                    ..
                } => {
                    for arg in args {
                        for segment in arg.segments() {
                            if let Segment::Pointer(address) = segment {
                                note_port(address.buf, &mut n_inputs, &mut n_outputs);
                            }
                        }
                    }
                }
                Work::Memset {
                    destination,
                    nbytes,
                    ..
                } => {
                    note_port(destination.readable().buf, &mut n_inputs, &mut n_outputs);
                    dummy_bytes = dummy_bytes.max(destination.byte_offset.saturating_add(*nbytes));
                }
                Work::Memcpy {
                    source,
                    destination,
                    nbytes,
                } => {
                    note_port(source.buf, &mut n_inputs, &mut n_outputs);
                    note_port(destination.readable().buf, &mut n_inputs, &mut n_outputs);
                    for byte_offset in [source.byte_offset, destination.byte_offset] {
                        dummy_bytes = dummy_bytes.max(byte_offset.saturating_add(*nbytes));
                    }
                }
                Work::Choice { .. } => {}
            }
        }
        let dummy = DeviceBuffer::<u8>::alloc(ctx, dummy_bytes).map_err(Error::Driver)?;
        let p = dummy.ptr();
        Ok(Self {
            inputs: vec![p; n_inputs],
            outputs: vec![p; n_outputs],
            workspace: vec![p; body.workspace().len()],
            _dummy: dummy,
        })
    }

    fn ports(&self) -> Ports<'_> {
        Ports {
            inputs: &self.inputs,
            outputs: &self.outputs,
            workspace: &self.workspace,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use recipe::{Blob, BufRef, Node, RecipeBody, WorkspaceBuf, WritableBuf};

    /// A residency refuses work whose declared workspace has no layout, before
    /// it touches the driver at all.
    ///
    /// A whole recipe cannot be this — validation asks the same question first —
    /// but a body carries no such proof, and the tuner loads bodies. So the
    /// refusal is where a residency comes to know its own workspace is a number:
    /// nothing downstream, `carve` included, has to wonder again.
    #[test]
    fn work_whose_workspace_has_no_layout_is_not_loaded() {
        let body: RecipeBody = RecipeBody::new(
            0,
            1,
            Vec::new(),
            vec![WorkspaceBuf {
                nbytes: usize::MAX,
                init: Some(Blob::from_vec(vec![0u8; 4])),
            }],
            vec![Node::new(
                Work::Memcpy {
                    source: BufRef::Workspace(0).at(0),
                    destination: WritableBuf::Output(0).at(0),
                    nbytes: 4,
                },
                [],
            )],
        );
        // SAFETY: the context is never read. The declaration above is refused
        // before `load` makes one current, which is the whole claim here — a
        // machine with no device reaches this refusal exactly as one with a
        // device does.
        let refusal = unsafe { ResidentRecipe::over_body(std::ptr::null_mut(), body) };
        assert!(
            matches!(
                refusal,
                Err(Error::InvalidRecipe(
                    recipe::RecipeError::WorkspaceLayoutOverflow
                ))
            ),
            "expected a workspace refusal, got {:?}",
            refusal.map(|_| ())
        );
    }
}
