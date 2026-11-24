# weave

`weave` supplies a Rust-native, MLIR-like substrate without depending on
MLIR. It owns a nested operation/region/block tree, SSA operation results and
block arguments, typed CFG successors, extensible types and attributes, a
dialect and operation registry, diagnostics, structural and semantic
verification, and one deterministic generic syntax.

The crate is deliberately dialect-neutral. XIR can use an `xir` dialect and
`gpu_ir` can use a `gpu` dialect without either representation being embedded
here.


## Representations

`Ir` is the editable form. Its arenas are append-only: erasing leaves a hole
that no later node reuses, so a surviving handle keeps its meaning and a stale
handle never names a different node. `uses()` is a lazy CSR index — compressed
sparse row, all readers in one flat array with one start offset per value —
invalidated by edits to SSA edges and preserved by edits to layout.

`Ir::freeze` consumes the editable form into `FrozenIr`: holes are closed once
and every entity is renumbered into its own typed dense index space, so an ID
of one entity kind cannot be confused with another. `FrozenIr` exposes no
mutable access and owns its use-def index, so readers share it without
invalidation or locking. Editable is for construction and rewriting; frozen is
for verification, analysis, serialization, and parallel lowering.

## Minimal dialect

```rust
use weave::{DialectName, OperationDefinition, OperationName, Registry};

let mut registry = Registry::new();
registry.register_dialect(DialectName::new("gpu")?)?;
registry.register_operation(
    OperationName::new("gpu.return")?,
    OperationDefinition::new().terminator(),
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Types and attributes whose syntax starts with `!dialect.mnemonic` or
`#dialect.mnemonic` may be registered with `TypeData` and `AttributeData`.
Unregistered payloads remain exact opaque strings so `print(parse(text))`
canonicalizes structure without discarding future dialect data. Verification
still diagnoses unregistered operations and dialect extensions.

## Generic syntax

Operations always use quoted `dialect.mnemonic` names. Types and attribute
values are quoted opaque strings, which makes delimiter nesting and unknown
payloads lossless. Results, operands, successors, attributes, and function-like
type signatures are always present; regions are optional.

```text
module {
  () = "builtin.module"() [] {} : () -> () regions (
    {
      ^bb0(%0: "i32"):
        () = "gpu.return"(%0) [] {} : ("i32") -> ()
    }
  )
}
```

`parse` accepts arbitrary whitespace and line comments. `print` emits stable
indentation, sequential SSA/block names, sorted attributes, and escaped
strings. A parsed program is expected to be verified separately with
`verify(&ir, &registry)`.
