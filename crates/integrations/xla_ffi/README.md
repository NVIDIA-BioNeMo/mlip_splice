# xla_ffi

Reusable Rust bindings for declaring XLA custom-call handlers — the
bridge that lets native code be invoked through XLA's FFI.

Its mission: turn a Rust `fn` into an XLA custom-call handler behind a
safe, typed surface. A `#[handler]` proc-macro lowers the signature
(`Stream`, `Buffer<T>`, `Ret<...>`, `RemainingArgs`/`RemainingRets`,
`#[attr]` scalars, slices, strings) into the C-ABI shim, decoding each
parameter from the call frame and dispatching against the raw FFI types.

## One unit (three crates)

The split is the standard Rust `sys`/`macros` companion pattern; read
and ship it as one:

- `sys` — raw bindgen bindings to `xla/ffi/api/c_api.h`, verbatim, no
  safe layer.
- `macros` — the `#[xla_ffi::handler]` proc-macro.
- `lib` — the safe runtime types and dispatch logic the macro binds
  against, layered over `sys`.

## Stages and state

`#[handler]` writes an execute-only handler, which is what a stateless
custom call needs. A handler that must remember something across
executions registers a bundle instead: `dispatch_stage` names the stage
one entry point answers, and `StateType<T>` registers a Rust type with
XLA — deleter included — so an instantiate handler can hand XLA a value
that every execute of that call site borrows and that dies with the
executable. Nothing is a process-global map, and nothing survives the
program that asked for it.
