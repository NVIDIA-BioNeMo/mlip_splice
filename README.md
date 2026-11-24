# SPLICE

SPLine Interatomic Cluster Expansion (SPLICE) is a machine-learned interatomic
potential built on a free-knot spline radial basis fused into the channelwise
tensor product.

This workspace contains four independently installable local distributions:

| Distribution | Import | Owns | Core dependencies |
| --- | --- | --- | --- |
| `algebraic-real` | `algebraic_real` | Exact real algebraic numbers and powers | None |
| `e3` | `e3` | Exact groups, representations, tensor bases | `algebraic-real` |
| `loom` | `loom` | JAX lane-program compiler and GPU runtime | `algebraic-real[jax]`, JAX |
| `splice` | `splice` | SPLICE model, training, inference | All three above |

The math and E3 cores do not import JAX. Array adapters are explicit:
`algebraic_real.jax.exact` carries exact constants through JAX; `e3.jax`
evaluates equivariant arrays. E3 and Loom do not depend on each other.
There are no `splice.loom`, `splice.e3`, or Loom math reexports.

## Install locally

Requires Python >= 3.12 and a Rust toolchain. GPU execution needs CUDA 13.
Use the workspace sources, **not** `pip install loom` or `pip install e3`:
those names on PyPI belong to unrelated projects. These short names are for
local installation only; this workspace does not publish them.

```bash
uv sync --all-packages --extra cuda13 --extra dev --extra data
uv run python packages/splice/scripts/train.py --help
```

For just one package and its dependencies, use `uv sync --package algebraic-real`,
`uv sync --package e3`, or `uv sync --package loom --extra cuda13`.
JAX is optional for the first two: select their `jax` extra for array adapters.
Workspace sources keep every local dependency bound to this checkout.
Python edits are editable; repeat `uv sync --reinstall-package loom` after
native compiler changes (likewise for `e3` or `algebraic-real`).

Build local wheels and source archives with `uv build --all-packages`.
For an offline install, place these wheels together and use `uv pip install
--no-index --find-links /path/to/wheels algebraic-real e3 loom splice` after
providing their external dependencies. A source archive includes its transitive
Rust sources and can be built outside this workspace.

The `cuda13` extra includes the cuBLAS grouped GEMM used by the per-species
linear layer. The MD extra adds its separate CUDA and ASE dependencies.

## Rust workspace

The root `Cargo.toml` declares every dependency's source once, in
`[workspace.dependencies]`: paths for workspace crates, versions or git revisions
for external crates. Member manifests list only `name.workspace = true`, adding
`features` when a crate needs more than the shared entry, which already enables
`abi3-py311` for `pyo3`. New crates under the directories in `members` join the
workspace automatically.

The PyO3 crates need a Python >= 3.11 interpreter to build. If the `python3` on
`PATH` is older, point Cargo at one:

```bash
PYO3_PYTHON="$(uv python find 3.12)" cargo check --workspace --all-targets
```

The `cargo-sort` pre-commit hook keeps manifests ordered (`cargo install
cargo-sort`). `cargo machete` (`cargo install cargo-machete`) reports
dependencies that no crate uses.

## Molecular dynamics

The MD harness (`splice.md`) runs a trained potential under real dynamics: a
cell-list GPU neighbour list, a velocity-Verlet integrator and a Langevin
thermostat, with the state resident on the device between steps. Its CUDA
kernels are built separately from the Rust workspace:

```bash
uv sync --all-packages --extra cuda13 --extra md
make -C packages/splice/python/splice/md/_ops_src         # add ARCH=ARCH_FULL to build a shippable fat binary
```

```bash
python packages/splice/scripts/md.py --arch runs/example/arch.yaml --card runs/example/card.yaml \
  --weights runs/example/ckpt_epoch_0005.eqx --positions water.xyz \
  --steps 5000 --equil 1000 --dt 0.5 --temperature 300 --trace-out nve.dat
```

The neighbour list emits a compact, symmetric edge list directly — no dense
intermediate — and is exact against a brute-force minimum-image reference, in
both directions, with a reproducible edge order. Boxes must be orthorhombic and
periodic in all three directions, and at least twice the cutoff on every axis.

`splice.md.NeighborBackend` is the seam: anything that can `update()` a list and
hand back `(senders, receivers, mask)` at a fixed length can replace
`CellNeighborList` without the driver noticing.

## Train a model

A training run needs two files besides the data: an architecture (`--arch`, one
of `examples/configs/`) and a model card (`--card`) that binds it to a dataset: the
species map, the cutoff, the average neighbour count and the per-species energy
offsets subtracted from the energy targets.

1. Check the loop on the bundled example (200 small molecules, a tiny model):

   ```bash
   python packages/splice/scripts/train.py --arch examples/configs/smoke.yaml --card examples/card.yaml \
     --train examples/data/molecules.xyz --output-dir runs/example --n-epochs 5 \
     --search-budget-ms 250 --shortlist-budget-ms 2000
   ```

   Always pass both budgets: without them the GPU kernel search at the first step
   is unbounded (see [Compilation budgets](#compilation-budgets)).

2. Build a card for your own extended-XYZ data, once per dataset:

   ```bash
   python packages/splice/scripts/build_model_card.py --data train.xyz --cutoff 6.0 --out card.yaml
   ```

3. Train a real architecture on it, with a held-out set for evaluation:

   ```bash
   python packages/splice/scripts/train.py --arch examples/configs/fk_cw64_tr256.yaml --card card.yaml \
     --train train.xyz --eval val.xyz --output-dir runs/my_model \
     --search-budget-ms 250 --shortlist-budget-ms 2000
   ```

   Each run writes its resolved architecture (`arch.yaml`), the card and the
   checkpoints to `--output-dir`; those three are what the calculator below loads.
   `packages/splice/scripts/launch_distributed.py` runs the same training data-parallel on all GPUs
   of one node.

For large datasets, convert to the packed format with `packages/splice/scripts/pack_extxyz.py`
and pass `--data-format packed` (see [examples/data/dataset-format.md](examples/data/dataset-format.md)).

### Compilation budgets

Set compilation budgets explicitly with `--search-budget-ms` and
`--shortlist-budget-ms`, not environment variables. Both accept milliseconds as
one number or a comma-separated schedule indexed by Loom transpose depth, for
example `--search-budget-ms 250 --shortlist-budget-ms 2000,1600,200`.

`search_budget_ms` limits optional source search by a cooperative wall-clock
horizon; a started operation can overrun it. After search, `shortlist_budget_ms`
limits the source-only shortlist by predicted realization time:
`max(assembly cost / workers, serial tuning cost)`. Shared assembly work is
charged once, and the first ranked candidate and its required support are kept
even if they exceed the limit. Kiln assembles and measures the frozen shortlist;
this estimate is not a hard runtime deadline or an assembly-only budget.

In Python, share one `Compile` configuration between per-call settings and AOT:

```python
import loom

with loom.aot(loom.Compile(shortlist_budget_ms=600_000, scheduling_time_limit_ms=5000)):
    result = loom.call(fn, inputs, compile_options=loom.Compile(max_workspace_bytes=0))
```

Each omitted field inherits from the innermost AOT context, then enclosing
contexts, then defaults. Explicit `None` disables nullable limits; for
`max_kernel_statements`, it selects automatic splitting. Python budget schedules
use mappings such as `{0: 2000, 1: 1600, 2: 200}`. A missing applicable entry
inherits from the next layer; an entry of `None` explicitly removes the limit.
`Compile().resolve()` exposes the effective defaults without an AOT context.

| `Compile` field | Default when not set in any layer |
|---|---|
| `search_budget_ms` | `None` (unlimited optional source search) |
| `shortlist_budget_ms` | `600_000` ms of **predicted realization cost**, not a timeout |
| `search_memory_budget_bytes` | `17_179_869_184` (16 GiB of estimated search volume, not measured RAM) |
| `max_workspace_bytes` | `None` (unlimited private scratch) |
| `max_kernel_statements` | `None` (automatic backend limit) |
| `max_algebraic_work` | `268_435_456` (positive exact-work bound per XIR execution) |
| `max_algebraic_expression_work` | `1_048_576` (positive exact-work bound per expression) |
| `scheduling_time_limit_ms` | `None` (measure all shortlisted candidates) |

Source defaults come from Rust via `_native.compile_defaults()`. The runtime
limit only stops Kiln from opening new tuning candidates; in-flight work may
finish later. Zero selects a feasible fallback without measurement. It never
changes source search or shortlist selection. A standalone `loom.compile`
returns an executable Recipe without runtime policy; pass runtime settings to
`loom.execute(..., compile_options=...)` or its enclosing AOT context.

The cache stays separate: `loom.aot(Compile(...), cache=cache,
workload_identity="my-workload")`. Its key includes all source settings and
runtime policy. Settings affect compilation, not an already-compiled JAX
program. Wrapper defaults remain separate: `splice.aot.aot_session` uses
250/2000 ms, while the training CLI uses `None`/`None`.

## Inference

```python
from ase.io import read
from splice.aot import aot_session
from splice.ase_calculator import SpliceCalculator

atoms = read("molecule.xyz")
atoms.calc = SpliceCalculator(
    "runs/example/arch.yaml",
    "runs/example/ckpt_epoch_0005.eqx",
    card_path="runs/example/card.yaml",
)
with aot_session():  # bounded kernel search: 250 ms search, 2000 ms shortlist
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
```

Run evaluations inside `aot_session()`. Outside it, the kernel search at the first
call for each new system size is unbounded, which costs minutes on large models.

The energy is relative to the card's per-species offsets: it is the total energy
minus the sum of `offsets` over the atoms. Energy differences between structures
with the same atoms (conformers, reactions, interaction energies) need no
correction; to compare with absolute reference energies, add the offsets back.

A charge-aware model (one trained with a charge and spin embedding) reads the
net charge and the spin multiplicity 2S+1 from `atoms.info["charge"]` and
`atoms.info["multiplicity"]`. When they are missing it evaluates a neutral
singlet, and prints once which values it used.

## Layout

| path | contents |
|---|---|
| `packages/splice/python/splice/` | model, ops, data pipeline, training support, ASE calculator |
| `packages/loom/python/loom/` | the Loom compiler's Python API |
| `packages/e3/python/e3/` | the mathematics of E(3): irreps, equivariant bases, geometry |
| `packages/splice/python/splice/draupnir/` | config -> chain engine |
| `packages/splice/python/splice/md/` | molecular dynamics: state, neighbour list, integrator |
| `packages/splice/python/splice/md/_ops_src/` | CUDA/C++ behind `splice.md._ops`, built separately |
| `crates/` | Loom's Rust implementation |
| `packages/splice/scripts/` | training, inference, MD and data-conversion entry points |
| `examples/` | model architectures (`configs/`), a model card, an example dataset and its format (`data/`) |
| `packages/*/tests/`, `tests/` | each package's own tests; the contracts that span packages |

Each package tests only what it can run alone:

| directory | covers |
|---|---|
| `packages/algebraic_real/tests/` | exact real algebraic numbers |
| `packages/e3/tests/` | E(3): representation theory, equivariant bases, geometry |
| `packages/loom/tests/` | Loom: lane semantics, dtypes, domains, autodiff, StableHLO, workloads |
| `packages/splice/tests/` | one model operation at a time, the assembled model, the chain engine, data, MD |
| `tests/` | the packages together: import boundaries and the one exact type |

## License

Apache-2.0. See [LICENSE](LICENSE).

## Citation

If you use SPLICE, please cite:

```bibtex
@misc{pellegrini2026splice,
  title  = {Performance-driven design of machine-learned interatomic potentials},
  author = {Pellegrini, Franco and Geiger, Mario and Stadler, Maximilian and
            Aykent, Sarp and Peng, Yuxing and Silveira, Ana and Boothroyd, Simon and
            Lynch, Brian and Batatia, Ilyes and Cs{\'a}nyi, G{\'a}bor and
            Kucukbenli, Emine},
  year   = {2026}
}
```

## Observe and replay compilation

`loom.observe(callback)` emits `loom.compilation.SourceCompilation` before
assembly and `RecipeLoadStarted` / `RecipeLoadFinished` around native
recipe registration. Assembly and tuning are lazy; their report is available
after the first completed execution, not necessarily at registration. A callback may raise to stop at a stage; no dummy JAX
executable is installed. Use `loom.aot(debug_dump_dir=...)` for tuning reports.

```python
captures = []
with loom.observe(lambda event: captures.append(event)
                  if isinstance(event, loom.compilation.SourceCompilation) else None):
    executable = jax.jit(fn).lower(*inputs).compile()
for index, capture in enumerate(captures):
    capture.save(f"source-{index}.json")
```

In a fresh process, `with loom.replay(captures): ...` substitutes each frozen
source compilation once. It matches XIR, domain/operand contracts, and source
options, not just XIR text. Runtime tuning policy may differ. Records require
the same compiler content, JAX versions, and GPU target; incompatible, missing,
or unused records raise `loom.ReplayError`. Clear JAX caches or start a fresh
process: these contexts cannot intercept already compiled JAX executables.
Records use versioned, checksummed JSON, not pickle. Load only trusted records:
their recipes contain executable GPU code.
