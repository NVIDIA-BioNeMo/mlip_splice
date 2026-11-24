# xir

`xir` defines Loom's static-shaped, layout-free SSA tensor dialect. Its
canonical representation uses `ir_core` operations, regions, blocks, SSA values,
typed tensor types, and symbol references. The established `Module`/`Func` model
remains available while clients migrate and converts losslessly with
`dialect::to_ir`; `generic_text` parses and prints the canonical representation.
The crate also provides the builder, structural analyses, verification, and
scalarization.

Whole-module rewrites live in the separate `xir_optimize` crate so consumers
that only construct, inspect, verify, print, or lower XIR do not compile the
optimization engine.
