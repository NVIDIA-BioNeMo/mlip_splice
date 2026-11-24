# cuda_driver

A safe, idiomatic Rust layer over the CUDA driver API (`libcuda`), resolved
lazily by `dlopen` so the CUDA SDK is never needed at build time. One
process-wide driver handle, shared by every consumer.

## Safe layer

The crate root exposes only the safe API; reach for raw libcuda (the `ffi`
module) only for what the safe layer doesn't cover.

- **Errors** — `Error` / `Result` / `check`: every `CUresult` becomes a
  `Result` carrying the call site and the driver's decoded message.
- **`Device` / `Context`** — explicit, never a hidden "current" global.
  `Device::get(ordinal)` / `Device::count()` (multi-GPU ready) and per-device
  queries (`sm_arch`, `sm_count`, `max_smem_optin`); `Context::create` /
  `primary` / `current` / `set_current`, a `ContextGuard` that restores the
  previous context on drop, and `ensure_context()` for the single-GPU default.
- **RAII handles**, each freed on drop and carrying its owning context:
  - `Stream` (owned) / `StreamRef` (borrowed — use this for streams handed in
    by XLA/PJRT, never an owning `Stream`): synchronize, launch, capture,
    memcpy/memset.
  - `Event`: record / synchronize / `elapsed_since`.
  - `Module` (Arc-shared) → `Kernel` → `Function`: load a CUBIN/PTX image, look
    up kernels (a `Kernel` keeps its module loaded), launch and introspect.
  - `DeviceBuffer<T>`: allocate in an explicit context, copy to/from host.
  - `Graph` / `GraphExec` with typed `KernelNode` / `MemsetNode` / `MemcpyNode`
    builders, instantiation, launch, in-place re-pointing, and introspection.

Contexts and devices are always passed explicitly — nothing reads or mutates a
hidden current-context global behind your back.

## Raw substrate (`cuda_driver::ffi`)

`ffi` is the unsafe substrate: the `CudaDriver` function table (`CudaDriver::get()`),
the `CU*` type aliases, the `CUDA_*` constants, and the `repr(C)` parameter
structs. The safe layer is built entirely on top of it.
