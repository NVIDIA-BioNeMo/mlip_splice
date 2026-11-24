# loom

Loom is a compiler. It takes a small function on scalars, applied to every
element of arrays indexed by a *domain graph* (for example atoms, edges, and the
maps between them), and fuses the whole thing into GPU kernels. It does not run
anything: it produces a [`recipe::Recipe`](../../foundations/recipe) of
candidate PTX kernels, and [`kiln`](../../runtime/kiln) times them on the
device, keeps the fastest, and executes it.

Most users call Loom from Python through `loom.call` in
[`packages/loom`](../../../packages/loom), which traces a JAX function and
reaches this crate through [`loom_py`](../loom_py). Use this crate directly only
to compile from Rust.

## Terms

- **Lane program**: the function, written for one element. Each operand also
  has a hidden last axis of `lane_count` lanes, and the function runs on every
  lane. An operand with `Lanes::Shared` holds one value for all lanes: an
  input broadcast across them, or an output summed over them.
- **Domain**: a finite index set, such as "atoms" with extent 3. An **index
  map** links two domains, such as "each edge belongs to one atom".
- **Placement**: which domain each operand axis is indexed by. When an output
  lives on a coarser domain than its inputs, the values are summed over that
  map.

## Example

Compile `f(c, x) = 2c · x` in Rust, where `c` lives on 3 atoms, `x` on 8 edges
with 4 lanes, and the output is summed per atom:

```rust
use domain_index::{Domain, Graph, IndexMap};
use loom::{CompileOptions, GpuTarget, Lanes, Operand, Program, compile};
use std::num::NonZeroUsize;
use xir::{builder::ModuleBuilder, dtype::DType, shape::Shape};

let mut b = ModuleBuilder::new();
b.begin_func("scaled_sum");
let c = b.param(Shape::scalar(DType::F32));
let x = b.param(Shape::scalar(DType::F32));
let two_c = b.add(c, c);
let y = b.multiply(two_c, x);
let entry = b.end_func(vec![y]);
let module = b.finish(entry).unwrap();

let operand = |domain, lanes| Operand {
    dtype: DType::F32,
    element_shape: vec![],
    lanes,
    axis_domains: vec![Some(domain)],
};
let (compiled, _profile) = compile(
    Program {
        name: "scaled_sum".into(),
        module,
        lane_count: NonZeroUsize::new(4).unwrap(),
        graph: Graph {
            domains: vec![Domain { extent: 3 }, Domain { extent: 8 }], // atoms, edges
            index_maps: vec![IndexMap {
                source_id: 1, // edge -> atom
                target_id: 0,
                is_injective: false,
                is_sorted: true,
                may_have_unmapped: false,
            }],
            aligned_views: vec![],
        },
        inputs: vec![operand(0, Lanes::Shared), operand(1, Lanes::PerLane)], // c, x
        outputs: vec![operand(0, Lanes::Shared)], // summed over edges and lanes
    },
    CompileOptions {
        // The GPU to compile for. Leave it `None` to read the current device.
        target: Some(GpuTarget {
            sm: 90,
            sm_count: 132,
            warp_size: 32,
            max_threads_per_sm: 2048,
            max_blocks_per_sm: 32,
        }),
        ..CompileOptions::default()
    },
)
.unwrap();
assert_eq!(compiled.output_count, 1);
// compiled.recipe goes to kiln::tune, then kiln::execute.
```

This README is also the crate documentation, so `cargo test -p loom --doc`
compiles and runs the example.
[`tests/projection_formula.rs`](tests/projection_formula.rs) runs this example
end to end, including the kiln calls and binding the arrays in
`compiled.index_arrays`.

## API

- `compile(program, options) -> (CompiledProgram, CompileProfile)` compiles a
  program. `CompiledProgram` holds the `recipe`, the `input_count` and
  `output_count`, and the `index_arrays` the caller binds after its own inputs.
- `stream_candidates` hands candidates to the caller one at a time.
- `CompileOptions::default()` is right for most uses. Its fields bound compile
  time, memory, kernel size and scratch space. Each field is documented in
  [`src/lib.rs`](src/lib.rs).
- `CompileProfile` reports timings and counters, with detailed records in
  `loom::profile`.

## Arithmetic is exact, not IEEE

Loom treats the program as math over the real numbers: it reorders sums,
cancels terms and simplifies coefficients exactly, using
[`algebraic_real`](../../foundations/algebraic_real). This is what makes a whole
expanded program small enough to fit in fused kernels. The costs:

- Results can differ from a fixed IEEE-754 evaluation order in the last bit.
- `x * 0` is `0` even if `x` is infinite or NaN, and the sign of zero is not
  preserved.

Do not use Loom for code that depends on either behavior.

## How it works

1. **Lift** (`src/lift.rs`, `src/scalarize.rs`): simplify the `xir` module.
2. **Evaluate** (`src/eval/`): expand it symbolically into one graph of scalar
   operations.
3. **Partition** (`src/backend/`): split the graph into kernels and choose
   which values to store between them.
4. **Render** (`src/backend/codegen/`): emit each kernel as PTX through
   [`gpu_ir`](../../foundations/gpu_ir).
5. **Shortlist** (`src/shortlist/`, `src/search.rs`): keep the candidates
   worth timing and package them as the recipe's `Choice` nodes.

`src/domain/` turns placements and index maps into index arithmetic.

## Testing

```bash
cargo test -p loom --lib              # CPU-only unit tests
cargo test -p loom                    # also the GPU tests in tests/; needs CUDA
cargo bench -p loom --bench compile_cost -- broadcast 128 4096
```
