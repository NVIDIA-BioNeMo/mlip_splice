// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use cuda_driver::{
    CUdeviceptr, CUstream, Context, Event, Function, Graph, GraphExec, GraphNode, Kernel,
    KernelNode, LaunchConfig, MemsetNode, StreamRef,
};

use recipe::algebra::Launches;
use recipe::algebra::workspace_sizes;
use recipe::{Address, Arg, BufRef, KernelLaunch, RecipeBody, Work, WritableBuf};

use crate::bind::{KernelParams, Ports, kernel_params, range_start, seed_workspace};
use crate::device::context_of_stream;
use crate::layout::carve;
use crate::{Bindings, Error, ResidentRecipe};

/// What a launch that states any number of shapes but one is refused as. A
/// kernel launch is one launch, so a recipe kiln runs states one shape for it.
fn sole_launch(node: usize, stated: usize) -> Error {
    Error::InvalidRecipe(recipe::RecipeError::GeometryArity {
        node,
        stated,
        arity: 1,
    })
}

// `Work::Memcpy` is realized as this copy kernel, not as a driver memcpy node:
// an instantiated memcpy node rejects re-pointing across memory mapping kinds
// (cuMemAlloc'd vs cuMemMap'd arenas — XLA binds buffers from both), while
// kernel parameters re-point freely. Grid-strided bytes, so any (dst, src,
// nbytes) is valid, including nbytes = 0.
pub(crate) const COPY_PTX: &str = concat!(
    r#".version 7.0
.target sm_52
.address_size 64
.visible .entry kiln_memcpy(
    .param .u64 kiln_memcpy_param_0,
    .param .u64 kiln_memcpy_param_1,
    .param .u64 kiln_memcpy_param_2
)
{
    .reg .pred %p1;
    .reg .b16 %h1;
    .reg .b32 %r<7>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [kiln_memcpy_param_0];
    ld.param.u64 %rd2, [kiln_memcpy_param_1];
    ld.param.u64 %rd3, [kiln_memcpy_param_2];
    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r2, %r1, %r3;
    cvt.u64.u32 %rd6, %r4;
    mov.u32 %r5, %nctaid.x;
    mul.lo.s32 %r6, %r5, %r1;
    cvt.u64.u32 %rd7, %r6;
    setp.ge.u64 %p1, %rd6, %rd3;
    @%p1 bra $L_end;
$L_loop:
    add.s64 %rd8, %rd5, %rd6;
    ld.global.u8 %h1, [%rd8];
    add.s64 %rd9, %rd4, %rd6;
    st.global.u8 [%rd9], %h1;
    add.s64 %rd6, %rd6, %rd7;
    setp.lt.u64 %p1, %rd6, %rd3;
    @%p1 bra $L_loop;
$L_end:
    ret;
}
"#,
    "\0"
);
pub(crate) const COPY_ENTRY: &str = "kiln_memcpy";
const COPY_BLOCK: u32 = 256;
const COPY_GRID_CAP: u32 = 65535;

fn copy_grid(nbytes: usize) -> [u32; 3] {
    let blocks = nbytes
        .div_ceil(COPY_BLOCK as usize)
        .clamp(1, COPY_GRID_CAP as usize);
    [blocks as u32, 1, 1]
}

/// The three arguments the copy kernel is launched with, once both ends are
/// proved to hold the bytes the copy moves.
///
/// The kernel is handed two bases and a count, and a pointer argument states
/// no length — so packing the arguments proves nothing about `nbytes` and the
/// two ranges are established here instead. Both caller paths that launch a
/// copy go through this, so neither can pack a length it did not check.
fn copy_params(
    ports: &Ports,
    destination: Address<WritableBuf>,
    source: Address<BufRef>,
    nbytes: usize,
) -> Result<KernelParams, Error> {
    range_start(ports, source, nbytes)?;
    range_start(ports, destination, nbytes)?;
    kernel_params(
        &[
            Arg::pointer(destination.readable()),
            Arg::pointer(source),
            Arg::scalar((nbytes as u64).to_le_bytes().to_vec()),
        ],
        ports,
    )
}

fn function(
    functions: &HashMap<(usize, String), Kernel>,
    module: usize,
    entry: &str,
) -> Result<Function, Error> {
    functions
        .get(&(module, entry.to_string()))
        .map(|k| k.function())
        .ok_or_else(|| Error::MissingSymbol(entry.to_string()))
}

// 2 is enough to overlap one re-point with one in-flight launch.
pub(crate) const GRAPH_RING_DEPTH: usize = 2;

/// Process-wide tally of how replays were served.
///
/// The three counts are the question anyone tuning this path is actually
/// asking. `captured` counts launches queued into a caller's graph capture,
/// including the single-launch fast path (one `cuLaunchKernel`, no graph
/// ring). `replayed` counts launches that ran rather than recorded — also
/// including that fast path, so a real `model_call` is not stuck at 0/0.
/// `repointed` counts, of the replays that did launch an exec, how many first
/// had to re-point it, so against `replayed` it says how stable the bound
/// addresses are. Neither the fast path nor a ring hit is visible from the
/// outside: a hit costs nothing and calls nothing, so it leaves no trace
/// anywhere but here.
pub mod stats {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub(super) static CAPTURED: AtomicU64 = AtomicU64::new(0);
    pub(super) static REPLAYED: AtomicU64 = AtomicU64::new(0);
    pub(super) static REPOINTED: AtomicU64 = AtomicU64::new(0);

    pub(super) fn bump(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Replays queued into a caller's graph capture, exec launches, and
    /// launches that had to re-point first.
    pub fn counts() -> (u64, u64, u64) {
        (
            CAPTURED.load(Ordering::Relaxed),
            REPLAYED.load(Ordering::Relaxed),
            REPOINTED.load(Ordering::Relaxed),
        )
    }

    pub fn reset() {
        CAPTURED.store(0, Ordering::Relaxed);
        REPLAYED.store(0, Ordering::Relaxed);
        REPOINTED.store(0, Ordering::Relaxed);
    }
}

/// The port addresses an exec is currently pointed at.
///
/// Kept so a replay can tell whether re-pointing would change anything. It
/// usually would not: a caller that reuses its allocations binds the same
/// addresses every step, and re-pointing is the one thing on the replay path
/// that costs a driver call for every node in the body.
pub(crate) struct BoundPorts {
    inputs: Vec<CUdeviceptr>,
    outputs: Vec<CUdeviceptr>,
    workspace: Vec<CUdeviceptr>,
}

impl BoundPorts {
    fn of(ports: &Ports) -> Self {
        Self {
            inputs: ports.inputs.to_vec(),
            outputs: ports.outputs.to_vec(),
            workspace: ports.workspace.to_vec(),
        }
    }

    fn matches(&self, ports: &Ports) -> bool {
        self.inputs == ports.inputs
            && self.outputs == ports.outputs
            && self.workspace == ports.workspace
    }
}

pub(crate) struct ExecSlot {
    pub(crate) exec: GraphExec,
    // Reused completion event for cross-stream waits and resource destruction.
    pub(crate) done: Event,
    // The last launch stream, retained even when re-pointing fails.
    pub(crate) launched_on: Option<CUstream>,
    pub(crate) bound: Option<BoundPorts>,
}

pub(crate) struct GraphRing {
    // Field order is the drop order: tear the execs (and their events) down
    // before the graph they were instantiated from.
    pub(crate) execs: Vec<ExecSlot>,
    // The graph is kept alive so its node handles stay valid for re-pointing.
    pub(crate) _graph: Graph,
    pub(crate) nodes: Vec<GraphNode>,
    // The slot pointed at the addresses the last replay bound. Rotation
    // happens when the addresses move, not on every replay: turning away from
    // a slot that is already pointed where the caller wants it would throw
    // away the re-point that put it there.
    pub(crate) current: usize,
}

pub unsafe fn execute<F: Launches>(
    loaded: &mut ResidentRecipe<F>,
    bindings: &Bindings,
    stream: CUstream,
) -> Result<(), Error> {
    let workspace = carve(bindings.workspace, workspace_sizes(loaded.body()))?;
    let ports = Ports {
        inputs: bindings.inputs,
        outputs: bindings.outputs,
        workspace: &workspace,
    };
    unsafe { run(loaded, &ports, stream) }
}

pub(crate) unsafe fn run<F: Launches>(
    loaded: &mut ResidentRecipe<F>,
    ports: &Ports,
    stream: CUstream,
) -> Result<(), Error> {
    // The stream is owned by the caller (XLA), so only borrow it.
    let stream = StreamRef::from_raw(stream);

    let ctx = loaded.ctx();
    // Which context a stream belongs to is a property of the stream, so it is
    // asked once per stream rather than once per replay. A caller that hands
    // over the same stream every step is answering the same question every
    // step, and the answer cannot have changed: a stream cannot be moved
    // between contexts.
    if loaded.checked_stream != Some(stream.raw()) {
        if unsafe { context_of_stream(stream.raw())? } != ctx {
            return Err(Error::ContextMismatch);
        }
        loaded.checked_stream = Some(stream.raw());
    }
    Context::from_raw(ctx)
        .set_current()
        .map_err(Error::Driver)?;

    // Whether the caller is recording this stream into a graph rather than
    // running it. Replaying our own exec is not available while it is: the
    // driver refuses to launch one into a capturing stream, and waiting on an
    // event recorded into one invalidates the capture. So a captured replay
    // queues the launches itself and lets the caller's graph hold them.
    let capturing = stream.is_capturing().map_err(Error::Driver)?;

    seed_workspace(loaded.body(), &loaded.seeds, ports, stream.raw())?;

    if let [node] = loaded.body().nodes()
        && let Work::Call {
            call:
                KernelLaunch {
                    module,
                    entry,
                    args,
                },
            geometry,
        } = node.work()
    {
        let [geometry] = geometry[..] else {
            return Err(sole_launch(0, geometry.len()));
        };
        let func = function(&loaded.functions, *module, entry)?;
        let mut params = kernel_params(args, ports)?;
        stream
            .launch(
                func,
                LaunchConfig {
                    grid: geometry.grid,
                    block: geometry.block,
                    shared_mem: geometry.dynamic_shmem,
                },
                params.as_mut_slice(),
            )
            .map_err(Error::Driver)?;
        stats::bump(if capturing {
            &stats::CAPTURED
        } else {
            &stats::REPLAYED
        });
        return Ok(());
    }

    // Split the borrows: `repoint` reads the recipe and function map while
    // mutating the ring, so hand it the pieces rather than `&mut loaded`.
    let body = loaded.held.body();
    let functions = &loaded.functions;
    let copy = loaded.copy.as_ref().map(|k| k.function());

    if capturing {
        stats::bump(&stats::CAPTURED);
        return unsafe { launch_nodes(body, functions, copy, ports, stream) };
    }

    stats::bump(&stats::REPLAYED);
    let ring = &mut loaded.graph;

    // A slot already pointed at these addresses is launched as it stands.
    if !ring.execs[ring.current]
        .bound
        .as_ref()
        .is_some_and(|bound| bound.matches(ports))
    {
        // Parameter updates affect only future launches, not work already queued.
        // Keep the last launch tracked even if re-pointing fails partway through.
        let next = (ring.current + 1) % ring.execs.len();
        let slot = &mut ring.execs[next];
        // Cleared first: a re-point that fails partway leaves the exec
        // pointed at some mixture of old and new addresses, and the one thing
        // that must not then happen is a later replay deciding it is already
        // pointed where it wants.
        slot.bound = None;
        stats::bump(&stats::REPOINTED);
        repoint(&slot.exec, &ring.nodes, body, functions, copy, ports)?;
        slot.bound = Some(BoundPorts::of(ports));
        ring.current = next;
    }

    let slot = &mut ring.execs[ring.current];
    // Finish a slot's previous use before moving its reusable completion
    // event to another stream.
    if slot
        .launched_on
        .is_some_and(|previous| previous != stream.raw())
    {
        slot.done.synchronize().map_err(Error::Driver)?;
    }
    slot.exec.launch(stream).map_err(Error::Driver)?;
    slot.done.record(stream).map_err(Error::Driver)?;
    slot.launched_on = Some(stream.raw());

    Ok(())
}

/// Queue every node of the body onto the stream, one launch at a time.
///
/// This stands in for replaying the instantiated graph when the caller is
/// capturing the stream. The replay is unavailable there twice over: the
/// driver refuses `cuGraphLaunch` into a capturing stream, and re-pointing an
/// exec describes an execution that a capture is not going to perform anyway.
/// Queueing the launches instead lets the capture record the same kernels, so
/// the work ends up in the caller's graph rather than in one of ours — which
/// is the point, because the caller can then hold this recipe and everything
/// around it in a single graph.
///
/// Nodes are queued in the order the body states them. That order is
/// topological — `build_graph` reads each node's dependencies as already-built
/// handles — so one stream serializing them respects every dependency. It also
/// imposes some it need not: nodes the graph would have run side by side are
/// recorded in a chain. That lost concurrency is what the caller's graph is
/// traded for.
unsafe fn launch_nodes<F: Launches>(
    body: &RecipeBody<F>,
    functions: &HashMap<(usize, String), Kernel>,
    copy: Option<Function>,
    ports: &Ports,
    stream: StreamRef,
) -> Result<(), Error> {
    for (idx, node) in body.nodes().iter().enumerate() {
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
                let [geometry] = geometry[..] else {
                    return Err(sole_launch(idx, geometry.len()));
                };
                let func = function(functions, *module, entry)?;
                let mut params = kernel_params(args, ports)?;
                stream
                    .launch(
                        func,
                        LaunchConfig {
                            grid: geometry.grid,
                            block: geometry.block,
                            shared_mem: geometry.dynamic_shmem,
                        },
                        params.as_mut_slice(),
                    )
                    .map_err(Error::Driver)?;
            }
            Work::Memset {
                destination,
                value,
                nbytes,
            } => {
                let dst = range_start(ports, *destination, *nbytes)?;
                stream
                    .memset_d8(dst, *value, *nbytes)
                    .map_err(Error::Driver)?;
            }
            Work::Memcpy {
                source,
                destination,
                nbytes,
            } => {
                let mut params = copy_params(ports, *destination, *source, *nbytes)?;
                stream
                    .launch(
                        copy.expect("a recipe with Work::Memcpy loads the copy kernel"),
                        LaunchConfig {
                            grid: copy_grid(*nbytes),
                            block: [COPY_BLOCK, 1, 1],
                            shared_mem: 0,
                        },
                        params.as_mut_slice(),
                    )
                    .map_err(Error::Driver)?;
            }
            Work::Choice { .. } => return Err(Error::UnresolvedChoice { node: idx }),
        }
    }
    Ok(())
}

fn repoint<F: Launches>(
    exec: &GraphExec,
    nodes: &[GraphNode],
    body: &RecipeBody<F>,
    functions: &HashMap<(usize, String), Kernel>,
    copy: Option<Function>,
    ports: &Ports,
) -> Result<(), Error> {
    for (idx, (node, &handle)) in body.nodes().iter().zip(nodes).enumerate() {
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
                let [geometry] = geometry[..] else {
                    return Err(sole_launch(idx, geometry.len()));
                };
                let func = function(functions, *module, entry)?;
                // The exec copies the params at the call, so the backing
                // store in `params` only needs to outlive this call.
                let params = kernel_params(args, ports)?;
                exec.set_kernel_node_params(
                    handle,
                    &KernelNode {
                        func,
                        grid: geometry.grid,
                        block: geometry.block,
                        shared_mem: geometry.dynamic_shmem,
                        params: params.as_slice(),
                    },
                )
                .map_err(Error::Driver)?;
            }
            Work::Memset {
                destination,
                value,
                nbytes,
            } => {
                let dst = range_start(ports, *destination, *nbytes)?;
                exec.set_memset_node_params(
                    handle,
                    &MemsetNode {
                        dst,
                        value: *value,
                        nbytes: *nbytes,
                    },
                )
                .map_err(Error::Driver)?;
            }
            Work::Memcpy {
                source,
                destination,
                nbytes,
            } => {
                let params = copy_params(ports, *destination, *source, *nbytes)?;
                exec.set_kernel_node_params(
                    handle,
                    &KernelNode {
                        func: copy.expect("a recipe with Work::Memcpy loads the copy kernel"),
                        grid: copy_grid(*nbytes),
                        block: [COPY_BLOCK, 1, 1],
                        shared_mem: 0,
                        params: params.as_slice(),
                    },
                )
                .map_err(Error::Driver)?;
            }
            Work::Choice { .. } => return Err(Error::UnresolvedChoice { node: idx }),
        }
    }
    Ok(())
}

pub(crate) fn build_graph<F: Launches>(
    body: &RecipeBody<F>,
    functions: &HashMap<(usize, String), Kernel>,
    copy: Option<Function>,
    ports: &Ports,
    ctx: Context,
) -> Result<(Graph, Vec<GraphNode>), Error> {
    let mut graph = Graph::new(ctx).map_err(Error::Driver)?;
    let mut handles: Vec<GraphNode> = Vec::with_capacity(body.nodes().len());

    for (idx, node) in body.nodes().iter().enumerate() {
        let deps: Vec<GraphNode> = node.deps().iter().map(|&d| handles[d]).collect();

        let handle = match node.work() {
            Work::Call {
                call:
                    KernelLaunch {
                        module,
                        entry,
                        args,
                    },
                geometry,
            } => {
                let [geometry] = geometry[..] else {
                    return Err(sole_launch(idx, geometry.len()));
                };
                let func = function(functions, *module, entry)?;
                // The graph copies the params at add time, so the backing
                // store in `params` only needs to outlive the add call.
                let params = kernel_params(args, ports)?;
                graph
                    .add_kernel_node(
                        &deps,
                        &KernelNode {
                            func,
                            grid: geometry.grid,
                            block: geometry.block,
                            shared_mem: geometry.dynamic_shmem,
                            params: params.as_slice(),
                        },
                    )
                    .map_err(Error::Driver)?
            }
            Work::Memset {
                destination,
                value,
                nbytes,
            } => {
                let dst = range_start(ports, *destination, *nbytes)?;
                graph
                    .add_memset_node(
                        &deps,
                        &MemsetNode {
                            dst,
                            value: *value,
                            nbytes: *nbytes,
                        },
                    )
                    .map_err(Error::Driver)?
            }
            Work::Memcpy {
                source,
                destination,
                nbytes,
            } => {
                let params = copy_params(ports, *destination, *source, *nbytes)?;
                graph
                    .add_kernel_node(
                        &deps,
                        &KernelNode {
                            func: copy.expect("a recipe with Work::Memcpy loads the copy kernel"),
                            grid: copy_grid(*nbytes),
                            block: [COPY_BLOCK, 1, 1],
                            shared_mem: 0,
                            params: params.as_slice(),
                        },
                    )
                    .map_err(Error::Driver)?
            }
            Work::Choice { .. } => return Err(Error::UnresolvedChoice { node: idx }),
        };

        handles.push(handle);
    }

    Ok((graph, handles))
}

#[cfg(test)]
mod tests {
    use super::*;
    use recipe::BufRef;

    const BASE: cuda_driver::CUdeviceptr = 0x1_0000;

    fn params_for(
        source: recipe::Address<BufRef>,
        destination: recipe::Address<WritableBuf>,
        nbytes: usize,
    ) -> Result<KernelParams, Error> {
        let bound = [BASE];
        let ports = Ports {
            inputs: &bound,
            outputs: &bound,
            workspace: &bound,
        };
        copy_params(&ports, destination, source, nbytes)
    }

    /// A copy is launched as a kernel, and a pointer argument states no
    /// length — so if the launch path did not check the count itself, nothing
    /// between the recipe and the device would. Each end is checked, and each
    /// on its own.
    #[test]
    fn each_copy_endpoint_is_checked_against_the_bytes_the_copy_moves() {
        let ok = params_for(BufRef::Input(0).at(16), WritableBuf::Output(0).at(32), 64);
        assert!(ok.is_ok(), "a copy both ends can hold");

        assert!(
            matches!(
                params_for(
                    BufRef::Input(0).at(usize::MAX),
                    WritableBuf::Output(0).at(0),
                    64
                ),
                Err(Error::RangeOverflow {
                    byte_offset: usize::MAX,
                    nbytes: 64
                })
            ),
            "the source's own offset plus the count"
        );
        assert!(
            matches!(
                params_for(
                    BufRef::Input(0).at(0),
                    WritableBuf::Output(0).at(usize::MAX),
                    64
                ),
                Err(Error::RangeOverflow {
                    byte_offset: usize::MAX,
                    nbytes: 64
                })
            ),
            "and the destination's"
        );
    }

    /// The same two ends against where the caller bound them. A count that
    /// runs off the end of the address space is caught for whichever end
    /// carries it, which a check that only resolved the two addresses could
    /// not see at all.
    #[test]
    fn each_copy_endpoint_is_checked_against_where_it_is_bound() {
        let at_the_top = |source: usize, destination: usize, nbytes: usize| {
            let source_port = [u64::MAX - 63];
            let destination_port = [BASE];
            let ports = Ports {
                inputs: &source_port,
                outputs: &destination_port,
                workspace: &destination_port,
            };
            copy_params(
                &ports,
                WritableBuf::Output(0).at(destination),
                BufRef::Input(0).at(source),
                nbytes,
            )
        };
        assert!(at_the_top(0, 0, 64).is_ok(), "the last 64 bytes there are");
        assert!(
            matches!(
                at_the_top(0, 0, 65),
                Err(Error::AddressOverflow { nbytes: 65, .. })
            ),
            "one byte more than the source has left"
        );

        let destination_at_the_top = |nbytes: usize| {
            let source_port = [BASE];
            let destination_port = [u64::MAX - 63];
            let ports = Ports {
                inputs: &source_port,
                outputs: &destination_port,
                workspace: &source_port,
            };
            copy_params(
                &ports,
                WritableBuf::Output(0).at(0),
                BufRef::Input(0).at(0),
                nbytes,
            )
        };
        assert!(destination_at_the_top(64).is_ok());
        assert!(
            matches!(
                destination_at_the_top(65),
                Err(Error::AddressOverflow { nbytes: 65, .. })
            ),
            "one byte more than the destination has left"
        );
    }

    /// A copy of nothing is a legal launch — the kernel is grid-strided over
    /// zero bytes — so proving the range must not turn it into a failure.
    #[test]
    fn a_copy_of_nothing_still_resolves() {
        assert!(params_for(BufRef::Input(0).at(0), WritableBuf::Output(0).at(0), 0).is_ok());
    }
}
