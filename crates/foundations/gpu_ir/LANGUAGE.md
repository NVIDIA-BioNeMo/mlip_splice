# The `gpu_ir` device language

*Normative reference for the shared device IR: what a program is, what it
means, and what [`verify`](src/verify/mod.rs) requires of it before the PTX
backend will render it.* Every rule below is executable — it is enforced in the
module named beside it — and the executable behaviour is what governs where this
text and the code disagree.

## 1. One function kind, two ways to use it

A [`DeviceFunction`](src/func.rs) is a name, typed parameters, typed results and
one body. There is no second function type.

- A **launch** is a use of one at a block geometry: `Launch { function,
  max_threads, max_registers }`. The geometry belongs to the launch, so one
  function serves every block a program wants to try it at.
- A **call** is a use of one inside another body: `OpKind::Call(Arc<DeviceFunction>)`.
  The callee is the function itself, not an index into a table beside the
  program, so a call keeps its meaning wherever the operation is carried and
  there is no second numbering to hold consistent.

A callee existed before anything could call it, so **call graphs are acyclic by
construction** and neither verification nor inlining needs a recursion guard.

`inline_calls` (`src/inline.rs`) runs the callee's body where the call stood and
is what the renderer sees; the backend never meets a `Call`. It is the two-arena
copy of `Builder::import` with the arguments substituted for the callee's
parameters — the same primitive kernel materialization uses — so there is no
second renumbering path. **A function with no call in it is not rebuilt**: the
pass leaves the arena exactly as its builder wrote it.

## 2. The arena

One [`Function`](src/arena.rs) arena owns every value, operation, block and
region. Every operation occupies exactly one block position; expression results
may still be re-evaluated explicitly at each use rather than stored once. Ids
are dense `u32` indices carrying the tag of the arena they index, and the tag is
always checked, so an id of one arena cannot name a value of another. Cloning
re-tags: an arena is cloned in order to diverge from it, and
`clone_with_map` is the way to carry ids across.

A finished function is plain data — `Send + Sync`, no reference counting inside,
no cells — which is what lets candidates compile in parallel.

**Operation placement and evaluation.** Every operation occupies exactly one
position in exactly one block. That position defines its scope and its order
relative to stored effects. `Bind`, `Write`, `Call`, `Scope`, `Break`, `Ret` and
`Bar` run once at that position. Other operations are expressions whose results
are explicitly re-evaluated at each use, so **a result read twice is computed
twice**; this preserves the established load and warp-collective semantics
without allowing operations outside blocks. `If` and `Loop` may either store
results once or re-evaluate them at each use.

**Construction is non-normalizing.** The builder does not fold constants,
reassociate, reorder commutative operands, contract or expand an FMA, or share a
repeated subexpression. Float association is a contract in this workspace: the
rounding sequence a caller wrote is the one that reaches the device.

**Order is meaning.** Registers are numbered per class in definition order and
labels from one ascending counter, so the emitted text is a function of the
order operations are placed. Placing in a different order emits different text
for the same program.

## 3. Types

A value holds either a data vector — `lanes` elements of one `Scalar`, `lanes >
0` — or a typed pointer into `Global` or `Shared`. Every operation is
elementwise over lanes and every conversion is an explicit `Cast`. There is no
textual type, no union, no inline asm and no architecture dispatch in the arena:
instruction selection is the backend's, and it lowers every well-typed program
for every target.

`verify/typing.rs` decides, per operation, what its operands must hold and what
it produces. The rules that are not merely arity:

| operation | rule |
|---|---|
| `IntConst` | a non-float, non-predicate scalar |
| `FloatConst` | a float scalar |
| `Sreg` | produces `U32` |
| `Load { lanes }` | operand 0 is a pointer; produces `lanes` of its element |
| `ConstLoad` | index is one non-float lane of 32 or 64 bits; the array holds whole elements |
| `Offset` | a pointer and one non-float lane; produces the same pointer type |
| `Un(Neg)` | not a predicate |
| `Un(Not)` | a predicate |
| `Un(Sqrt)` | a float |
| `Bin(Shl\|Shr)` | the amount is one `S32`/`U32` lane |
| `Bin(_)` | both sides identical |
| `Fma` | three identical float operands |
| `Cmp` | both sides identical; produces `Pred` of their lane count |
| `Select` | a predicate of one lane or of the arms' lane count; arms identical |
| `Cast` | predicate on both sides or on neither |
| `Splat(lanes)` | a scalar operand |
| `Extract(lane)` | `lane` inside the operand |
| `Pack` | one scalar type throughout; at most 255 lanes |
| `ShflDown` | not a predicate — there is no bit pattern to move |
| `WarpAny` | a predicate, producing one |
| `Bind` | produces exactly what its operand holds |
| `Call` | arguments and results agree with the callee |

## 4. Named math contracts

**Every arithmetic operation names its contract, and a scalar with no such
operation has none to name.**

- `Fma` is fused: one rounding, never two. It is a float operation, because only
  a float rounds.
- `Bin(Div)` and `Un(Sqrt)` on floats are correctly rounded — `div.rn`,
  `sqrt.rn`. There is no approximate spelling of either in this language, so a
  producer cannot lose accuracy by accident.
- `Add`, `Sub`, `Mul` on floats are `.rn`. The packed pair forms the backend
  selects (`f32x2` on sm_100+, `f16x2`, `bf16x2`) compute the same `.rn` result
  as the lane-split ones, which is why selection can be the backend's and no
  candidate is lost to the device generation.
- `f16` and `bf16` operations with no native instruction at a target compute in
  `f32` and convert back, which is the same value the native form produces.
- Integer `Div`/`Rem` by a constant lower to the exact multiply-shift sequence,
  not to an approximation.
- **Predicates have no arithmetic but `And`, `Or`, `Xor`.** There is no sum of
  two predicates.
- **Floats have no bit operations and no shift.**
- **`Rem` has no float form.** A float remainder would be a library call this
  language does not contain, so it is refused rather than lowered into
  something else.

## 5. Regions and control

A region belongs to exactly one operation and holds exactly one block. A block
belongs to exactly one region, except the entry block, which belongs to the
function. `verify/structure.rs` decides all of it.

- `If` — one predicate operand, a then and an else region, one result per value
  the regions yield. Both arms yield the results' arity and types. An arm with
  no steps yielding nothing is legal only where the `if` has no results.
- `Loop { index, trip }` — `start, stop, step` then one starting value per
  carried result. Its region's block takes the index and each carried value and
  yields what the next iteration receives; the starting values, the parameters,
  the yields and the results all agree in type.
- `Scope` — one region that runs where it stands and yields nothing.
- `Break` — leaves the innermost enclosing loop; that loop must carry nothing,
  since a break would leave its results undefined.
- `Ret` — leaves the kernel; legal only where every enclosing region belongs to
  an operation with no results, for the same reason.
- A result of an `If` is undefined where an arm returns or breaks, so that is
  refused.

**Ownership.** A value is defined by the block that takes it or the operation
that produces it, at the index that thing claims, and **a step stands in exactly
one block, exactly once**. Standing twice would define its values twice, and
what they hold the second time need not be what they held the first — the same
read, either side of a write to the cell it reads, is two values. A caller that
means to compute something twice builds it twice.

**Dominance** is scope. Every place a value reads must be defined in a scope
that is still open. A region's own bindings are the region's: **what a region
computes leaves it only through what it yields**, which the operation owning the
region produces. Naming storage does not make a value escape — see §8 — because
the storage outliving the region says nothing about whether the region ran.

## 6. Effects

An operation's effect is a property of its kind (`src/effect.rs`), not a
separate walk and not a declaration beside the program:

| kind | effect |
|---|---|
| `Load` | `Read { ptr: 0 }` |
| `ConstLoad` | `ReadConst` — immutable memory it addresses itself |
| `Write { Set }` | `Write { ptr: 0 }` |
| `Write { Add }` | `ReadWrite { ptr: 0 }` |
| `Bar` | `Barrier(Block)` |
| `ShflDown`, `WarpAny` | `Barrier(Warp)` |
| `If`, `Loop`, `Scope`, `Call`, `Break`, `Ret` | `Control` |
| everything else | `Pure` |

Where an effect touches memory it names the **operand** addressing it, so
provenance is a use-def walk from that operand rather than a name to resolve.
`verify` requires that operand to hold an address.

The union over a region is the same answer folded: `Function::synchronisation`
gives the strongest synchronisation an operation performs, following its
operands as well as its regions — an operand can synchronise, because a loop
standing for the value it reduces to is evaluated where it is read and its body
may hold a barrier.

**Speculation is derived from effects**: an operation may be evaluated where it
would not otherwise run exactly when its effect is `Pure`. A memory read may not
be, because the address may only be in range where the guard holds; nor may a
warp collective, which observes which lanes arrived.

## 7. Memory

- `Load { nc }` reads non-coherently only from global memory.
- An eviction hint (`Evict::First`/`Last`) applies only to global memory.
  `Evict::Normal` renders the plain instruction, byte-identical to an unhinted
  render.
- `Write { mode: Set }` stores; `Write { mode: Add }` is the atomic — one
  `red.global.add` per lane at consecutive addresses. It is global-only, takes
  no eviction hint, and has a form only for `f16`, `bf16`, `f32`, `f64` and
  32/64-bit integers.
- A write's value scalar is the pointer's element type.
- A shared array declares a positive length and is aligned to 16 bytes, which
  covers the widest vector access it can host.

**Alignment.** An access that renders as one vector instruction lands on a
boundary as wide as the vector. That obligation splits in two:

1. the element offset the body computes must be a whole multiple of the lane
   count — decided here, over the arena, by the stride algebra in
   `src/stride.rs`;
2. the base must be aligned to it — which is what a binding owes the kernel, and
   what `wa`'s `ParameterEffect::alignment` states.

The stride lattice is divisibility: `0` is every stride (nothing was added to
the base), `1` is nothing proven, meets are `gcd`, products scale, and a product
that does not fit keeps the larger factor rather than a saturated number.
Constants prove their own value; `Add`/`Sub` meet; `Mul` and a constant `Shl`
scale; a widening `Cast` keeps what it had and a narrowing one keeps the
power-of-two part; an `If` result is what both arms prove; **a loop index is
`gcd(start, step)`**, because every value it takes is `start + n·step`; a
carried value is what its start and its yield both prove. A value defined in
terms of itself proves nothing on that path, which is the conservative answer
and the one that terminates.

Widths with no vector form split into element accesses and ask nothing beyond an
element. `ptx::vector_form` is the one table both the renderer and the verifier
read.

## 8. Taking over storage

`OpDef::into` names, per result, the storage that result is produced into. It is
a **request about allocation, never about meaning**: the value a result holds is
the same whether it runs in the storage it names or in storage of its own, and
the backend declines wherever running there would change what something else
reads — a destination of the wrong type, two results naming one storage, a
register a write in progress has reserved, a place something later still reads.

So the verifier requires only what makes the request answerable: one destination
per result, and each a place with storage to name. Two things it does refuse
outright. **A binding may not take over the index a running loop counts with**,
because the loop writes that register at the bottom, where no step shows. And a
**binding's destination must hold what the binding holds**, because a binding
runs in the storage it names with no arm to fall back to; a branch's destination
is only a request, and one it cannot honour it declines.

Naming a place is **not** how a value leaves a region. An accumulator folded
under a guard is produced into the accumulator's storage, but the value the fold
reached is the guard's own: it escapes by being yielded, and the guard produces
one result per accumulator — the arm that runs hands back what it reached, the
arm that does not hands back what the accumulator already held.

## 9. Convergence

`bar.sync` completes when every **non-exited** thread of the block has arrived.
Two things break it, and only two:

1. a thread that is still running and takes a path around the barrier — an arm
   it did not enter, or an iteration it did not run;
2. a thread that leaves the loop the barrier is in while others go round again.

Leaving the kernel is not one of them: a thread that returned has exited and the
barrier is satisfied by the rest, which is why the early-out guard every
grid-strided kernel opens with is legal above a barrier.

`verify/converge.rs` therefore refuses:

- a barrier under an `If` whose predicate is not proven **block-uniform**;
- a barrier in a loop body not stated to run the same number of times in every
  thread of the block;
- a `Break` out of a loop whose body holds a barrier.

**Block-uniformity is proven, not assumed.** A launch parameter, a shared
array's base, `%ctaid.x`, `%ctaid.y`, `%ntid.x`, a constant, and pure arithmetic
over uniform values are uniform. `%tid.x` is not. Neither is anything read from
memory, a warp collective (which is a warp's answer, not the block's), a call, or
what a loop carries. A loop's index is uniform when its bounds are.

**Trip counts are a stated fact, not a guess.** How often a loop goes round is
how its bounds come out, and bounds are values the arena cannot read. So
`OpKind::Loop` carries `Trip`, and the producer — which usually knows the
geometry as plain integers — states `Trip::BlockUniform` where the counts agree.
A barrier inside a `Trip::PerThread` loop is refused. Warp collectives need no
such statement: they reduce over the lanes that arrived, which is what makes
them sound inside a partially retired warp.

## 10. What a module must agree on

A module is one text, so some rules are neither one function's nor decidable
inside one (`verify/module.rs`):

- two launches may not be entered under one name;
- a constant array is emitted once under its name, so two launches reading
  different arrays under one name is a module that cannot be written;
- within a launch, a parameter and a shared array may not share a name;
- **every name that reaches the text — an entry, a parameter, a shared array, a
  constant — must be a PTX identifier.** A name outside that set is not escaped
  into one: it would land in the middle of a declaration saying whatever it
  says, so it is refused.

## 11. What a target must have

Instruction selection is the backend's and almost nothing depends on the
target: packed pair forms where the device has them, lane-split scalars where it
does not, an `f32` detour where a half width has no native instruction. A
candidate is never lost to the device generation *for arithmetic*.

Atomics are the exception, and they are checked against the target a module
renders for. A reduction of a float width the target has no `red` for cannot be
split into anything with the same rounding and the same atomicity, so it is
refused for that target rather than lowered into something else:

| reduction | first target |
|---|---|
| `red.global.add.f64` | sm_60 |
| `red.global.add.f16` | sm_70 |
| `red.global.add.bf16` | sm_90 |
| 32/64-bit integers, `f32` | every target |

## 12. The launch ABI

A launched function is a kernel:

- it hands back nothing;
- it takes what a host can pass — a global pointer or a scalar — and may declare
  shared arrays of its own;
- `max_threads` is in `1..=1024`;
- `max_registers`, when stated, is in `16..=255`, the range `.maxnreg` admits.

A **callee** may hand values back, and may not declare shared memory: shared
memory belongs to the kernel that is launched, and a callee is inlined into one.

## 13. Rendering

`ptx::render_ptx_module` inlines calls, verifies the whole module against the
target, and then renders. **Rendering is total over verified programs**: every rule the backend relies on is decided
before a byte is emitted, so what remains in the renderer is either unreachable
by construction or a lowering choice. A renderer that met an unverified program
would have no way to explain what is wrong with it — it is halfway through
emitting text — which is why the boundary is where it is.

`render_module` is the same thing returning the diagnostic instead of panicking,
for a caller that answers for an unverified program itself.

A diagnostic states which rule, where, and what about that place breaks it:
`Rule` ∈ {ownership, region, dominance, type, effect, abi, aliasing, memory,
alignment, convergence}, `Site` ∈ {the function, an operation, a value, a block,
a region}.

**Naming.** `gpu_ir` emits PTX mnemonics and register names, which carry no
engine IP, and depends on nothing. Names that carry meaning — a kernel entry, a
shared array, or a value's provenance — are chosen by the producer. This crate
emits no CUDA-C++ scaffolding and introduces no separate naming transformation.

## 14. What is deliberately not here

No hash-consing or CSE, no canonical reordering of independent pure operations,
no register coalescing. Each would change the emitted text for the same program,
so each is its own decision with its own benchmark — not something construction
does quietly.
