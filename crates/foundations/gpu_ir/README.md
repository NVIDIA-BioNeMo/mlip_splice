# gpu_ir

*A typed device-level GPU program, and the PTX it becomes.* One arena owns every
value, operation, block and region of one function; values are `Scalar × lanes`
vectors or typed pointers, every operation is elementwise over lanes, and every
conversion is an explicit `Cast`. There is no textual type, no `union`, and no
architecture dispatch in the arena.

*[`LANGUAGE.md`](LANGUAGE.md) is the normative reference* — what a program is,
what it means, and what the verifier requires of it. It is written beside the
implementation and against the executable behaviour.

*Instruction selection belongs to the backend, not to the producer.*
[`ptx`](src/ptx.rs) lowers a **verified** program's arithmetic for every target,
choosing packed pair forms where the device has them and lane-split scalars
where it does not, so no candidate is lost to the device generation over how it
computes. What is wrong with a program — including the few things a target must
have an instruction for, which are atomics — is said by
[`verify`](src/verify/mod.rs), in the program's own words, before a byte is
emitted.

*It depends on nothing.* The IR is scalars, vectors, pointers, operations,
regions and functions — enough to say what runs on a device and what it emits,
and not one thing about who asked. Element-type systems that belong to a
frontend, like a CUDA-source `Dtype`, convert to `Scalar` at that frontend.

## Generic dialect form

`gpu_ir::dialect` registers the `gpu` operations, `!gpu.data` and `!gpu.ptr`
types, and structured `#gpu.attr` attributes with `ir_core`. Its `to_ir_core`
adapter preserves every value, operation, block, region and SSA edge of a
`gpu_ir::Function`; `print`, `parse` and `validate` expose the canonical generic
syntax. The original arena remains the source for the stable wire format and
PTX lowering, so neither representation changes.
