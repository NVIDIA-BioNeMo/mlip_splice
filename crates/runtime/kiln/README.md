# kiln

Kiln realizes, loads and executes GPU work in the recipe format.

A [`Recipe`](../../foundations/recipe/src/lib.rs) (defined by the `recipe` crate)
is a serializable, backend-agnostic description of a unit of GPU work,
shaped as a CUDA Graph: a list of nodes (kernel launches, memsets,
device-to-device copies) plus the dependency edges between them. It states
possibilities rather than one program: a `Choice` node offers alternative
sub-recipes, and each `KernelModule` offers a PTX compile request, prebuilt
images, or both. Kiln consumes that IR, so the xla bindings and the weldx
backend share a single realizer and launcher instead of each carrying its own.

```
[a Recipe: nodes + dependency edges, workspace sizes, and per module a
 compile request and/or prebuilt images]  →  [realization: one loadable
 image per module, the measured winner per Choice]  →  [an artifact
 Recipe worth persisting + a ResidentRecipe on the device]  →  [the same
 work executed many times on an NVIDIA GPU]
```

## Allocation-free

Kiln never allocates device memory. A `Recipe` declares the byte sizes of
the scratch buffers it needs; the caller allocates them and binds
concrete `Input` / `Output` / `Workspace` device pointers through
[`Bindings`](src/lib.rs) at execution time. A
[`ResidentRecipe`](src/resident.rs) owns its modules but not the workspace, so
the same loaded recipe runs against any set of compatible pointers.

## Load once, execute many

Running a recipe splits into a one-time load and a cheap repeatable
execute:

- **`ResidentRecipe::new(ctx, flat)`** ([`resident`](src/resident.rs)) —
  takes an already-deserialized `FlatRecipe`, `cuLibraryLoadData`s every
  module, eagerly resolves every kernel-launch entry to a `CUfunction`
  (raising its dynamic-shared-memory cap once), builds the recipe's DAG
  into one `CUgraph` (one node per recipe node, dependencies → edges), and
  instantiates the `CUgraphExec` ring. The result holds its libraries
  resident and unloads them on drop.
- **`execute(&mut loaded, &bindings, stream)`**
  ([`exec`](src/exec.rs)) — seeds workspace init bytes, then
  re-points the next exec in the ring to the current bindings and launches
  it, amortizing per-launch overhead across the many replays.

## Realization

Before a recipe runs, every module needs exactly one loadable image and every
`Choice` needs a winner. Producing both for one device and one workload is
realization: resolve, compile, measure, select, emit.

`Realizer::realize` takes one [`RealizationArgs`](src/realize.rs) — the pending
recipe, the bindings, the stream, and the policies — and returns the resident
recipe, the artifact to persist, a report, and the tuning progress. A bundle
rather than a parameter list: the output policy, the tuning deadline, the
workload identity, the cache and the validation contract all arrive here, and
each new one would otherwise be another positional argument at every call site.

**Resolving a module.** Kiln first reuses an image the module already carries: a
`Cubin` assembled for this device's `sm_`, or `CudaLoadable` bytes whose
compatibility the CUDA loader decides, such as a captured vendor fatbin. Failing that it assembles the module's compile
request: nvJitLink for a relocatable request, an external `ptxas` for a
whole-program one, and the CUDA driver's own PTX JIT when the request constrains
neither assembly nor registers and no assembler is installed. Assembly runs
through [`cuda_toolchain`](../../cuda/cuda_toolchain/), which owns the content-addressed
compile cache and the bounded worker pool, so requests that share a source,
target and assembler compile once and stay compiled across restarts. The
registers and local bytes of the produced image are measured back and checked
against the module's producer-declared resource bounds — kiln applies them, it
does not own the spill policy behind them.

Kiln generates nothing: no NVRTC call, no CUDA C, no instruction selection, no
graph alternative of its own. Recipe compilers state the candidates; kiln compiles,
measures and picks — never a heuristic. A recipe whose every module already
carries an image loads with no compilation library present at all, because the toolchain is
opened only when a module actually needs its source assembled.

**Reusing a realization.** Pass an [`ArtifactCache`](src/artifact_cache.rs) and a
`WorkloadIdentity` and a completed realization is stored under the identity of
everything that produced it: the program (a streaming BLAKE3 over the pending
recipe's structure and its blobs' stored digests — never a serialization or a
decompression of it), the caller's workload name, the device and assemblers, and
the output policy. A warm hit loads that artifact and runs it: no candidate is
compiled, nothing is measured. The cache is an object the caller owns, not a
process global, and it stores only *finished* work — a selection cut short by a
deadline, or made around a candidate this machine could not compile, runs but is
never published under a key that does not mention either.

There is no data-blind default: without a workload identity nothing is looked up
or stored, because a zero-skip candidate wins or loses by the distribution in
the buffers, which shapes and pointers do not describe.

**Scheduling time limits.** Rust's `TuningPolicy::scheduling_time_limit_ms`
(`Option<f64>`) and Python's `load_recipe(..., scheduling_time_limit_ms=...)`
use the same wall-clock limit in milliseconds, measured from the start of
autotuning. `None` measures every candidate; zero is allowed and selects a
feasible fallback without measurement. Values must be finite, nonnegative, and
representable as a duration; invalid values return an error rather than panic.
The limit stops the tuner from *opening* another candidate. It cannot preempt an
assembly already in the compile pool or a kernel already launched, so a
realization overruns it by whatever was in flight, and every choice still selects a feasible candidate — unmeasured if the
deadline expired first. What it stopped is returned as `TuningProgress`, which a
later call can resume in this process; it is not persisted, because a partial
selection is a position in an interrupted run rather than a result with an
identity.

**Validation.** [`ValidationContract`](src/validate.rs) states what a comparison
needs — explicit output byte extents, caller-owned reference and candidate
buffers, an initializer for each clean candidate run, and a caller-supplied
comparator. Before timing a feasible candidate, kiln runs the whole recipe with
that candidate selected into the candidate buffers and rejects it on the first
output disagreement. Validation never writes a live output or allocates device
memory. A validated selection bypasses the artifact cache because its current
key cannot name the comparator or tolerance; reuse requires a future
caller-supplied validation identity rather than an inferred one.

## Serialized artifact

The byte format is defined by `recipe`:
`recipe::serialize` encodes a `Recipe` to its public, versioned layout —
a flat, little-endian, sectioned layout specified in
[`FORMAT.md`](../../foundations/recipe/FORMAT.md), the normative contract.
`recipe::deserialize` validates the magic and version before decoding,
so a stale or foreign artifact is rejected rather than misread. The bytes
are the save-to-disk / send-over-the-wire form of a recipe;
`ResidentRecipe::new` takes the deserialized `FlatRecipe` and does the device load.

Both a pending recipe and a realized artifact use this one format; which of the
two a given file holds is visible in its modules, not in its magic. A consumer
drives kiln's Rust API directly: `recipe::deserialize`, then
`Realizer::realize` for a recipe that still states choices or compile requests,
or `ResidentRecipe::new` straight away for one already reduced to a single image
per module. The xla bindings wrap the first path behind the `run_recipe` XLA FFI
handler — realize once on the first call, `execute` on every later one — so JAX
launches recipes without a bespoke C entry point.

External consumers that only need to read/inspect a recipe use the
header-only reader
[`recipe/include/recipe.h`](../../foundations/recipe/include/recipe.h),
which has no library or CUDA dependency.

## Grid dimensions

A kernel node's grid is a concrete `[u32; 3]` block count. The producer
(the producing compiler) bakes it at compile time — for an occupancy-saturating grid, the compiler
resolves the kernel's occupancy times the device SM count there. Kiln
launches the baked number as-is; there are no data-dependent grid
dimensions today.

## Crate DAG

```
cuda_driver        raw libcuda function table
cuda_toolchain     PTX assembly (nvJitLink / ptxas), compile cache, worker pool
blob_cache         content-addressed storage, in memory or on disk
recipe             the recipe format: Recipe IR + wire layout
   ↑
kiln               realize (resolve → compile → tune) + load + execute
   ↑
{ a compiler produces a Recipe ; bindings + weldx realize and execute it }
```
