# xir_optimize

`xir_optimize` transforms verified `xir::Module` values through deterministic
whole-module passes. It owns constant and algebraic canonicalization,
exact-polynomial rewriting, finite-map alternatives, shape cleanup, CSE, and
dead-code elimination.

The crate depends on `xir`; the IR crate does not depend on this optimizer.
Construct an explicit `PassManager` for semantics-preserving cleanup. Use
`algebraic_pipeline()` only when reassociation and cancellation are allowed.
