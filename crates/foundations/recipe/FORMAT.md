<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# The `RCPE` recipe wire format

This document specifies version 13 of the public byte format produced by
`recipe::serialize`. A Recipe is a graph of GPU operations plus alternative
implementations. It may carry prebuilt module images, PTX compilation requests,
or both. Reading the structure requires no CUDA context or compiler.

The reference C reader is [`include/recipe.h`](include/recipe.h).

## Scalar conventions

- Integers are little-endian.
- Fields are byte-packed with no padding or alignment.
- `u8`, `u16`, `u32`, and `u64` have their usual fixed widths.
- `blob` is a `u32` byte length followed by its bytes.
- `str` has the same layout as `blob`; its bytes must be UTF-8 and are not
  NUL-terminated.
- `vec<T>` is a `u32` count followed by that many `T` values.
- `option<T>` is tag `0` for absent or tag `1` followed by `T` for present.
- Enum tags are `u8` values stated below.

Byte counts that may exceed 4 GiB use `u64`. Counts, lengths inside `blob`, and
indices use `u32`.

## Header and section table

The 12-byte header is:

| offset | size | field | value |
|---:|---:|---|---|
| 0 | 4 | magic | ASCII `RCPE` |
| 4 | 2 | version | `13` |
| 6 | 2 | reserved | writer emits `0`; reader ignores |
| 8 | 4 | section count | `4` |

It is followed by `section_count` entries of 20 bytes:

| size | field | meaning |
|---:|---|---|
| 4 | id | `1` BLOBS, `2` REQUESTS, `3` BODY, `4` SIGNATURE |
| 8 | offset | absolute offset from the start of the artifact |
| 8 | length | section payload bytes |

All four sections are required. Readers skip unknown section IDs. Writers
emit the required sections in ID order with contiguous payloads, but readers
must use the table rather than assume that layout.

## BLOBS, section 1

The section is `vec<BlobEntry>`. One entry is:

| field | encoding |
|---|---|
| encoding | `u8`: `0` raw, `1` zstd |
| decoded length | `u64` |
| decoded digest | 32 raw bytes: BLAKE3 of decoded content |
| payload | `blob` |

Raw entries require `payload.len == decoded_length` and their digest is checked
while opening the artifact. Zstd entries are decoded and checked lazily on
first access. A decode or digest failure is a hard error.

The pool is global to the Recipe, including every nested Choice. Two payloads
share an entry when they are stored the same way and their decoded bytes are
equal — digest first, then decoded length and decoded content, because a digest
narrows the search and does not settle it. So equal content is stored once even
if referenced from several candidates.

A reader takes each entry's encoding from the entry. A writer picks it from
what the payload is: compile sources are zstd, module images and workspace
initializers are raw. Content that appears as both is therefore two entries,
one per form; a reader must not assume equal decoded content implies one entry.

Entries appear in the order the recipe's payloads are reached — the compile
table, then each body's modules (their images in priority order), its
workspace, and its nested candidates — so a module's image and a candidate's
initializer are numbered where the fields that name them are written.

The digest provides identity and integrity, not confidentiality. A Recipe that
contains a PTX request discloses that source. Encryption requires a future blob
encoding with an explicit key and nonce contract; it must not be inferred from
the digest.

## SIGNATURE, section 4

The external byte-buffer ports of the recipe:

| field | encoding |
|---|---|
| input extents | `u32` count, then that many `u64` |
| output extents | `u32` count, then that many `u64` |
| aliases | `u32` count, then `(input: u32, output: u32)` pairs |

Each alias requires the named input and output to share one allocation. Alias
pairs are ordered by strictly increasing input index, and both input and output
indices are unique. Both indices must be in range and their extents equal.

Each `u64` is one port's size in bytes, and a port's position in its vector is
its index — the index `Input(i)` and `Output(i)` references elsewhere use. A
recipe bound to nothing from outside encodes three zero counts.

The count is a `u32` and the extent a `u64`: a program has few ports, but one
port can be larger than 4 GiB, and an extent that wrapped would name a small
buffer that every later check would then accept.

Only the root carries this section. A Choice candidate is a `body`, and its
ports are the parent buffers its `input_binding` and `output_binding` select,
so nothing inside the BODY section restates them.

Element types are not encoded. A port is a byte extent here; what those bytes
mean belongs to the producer above this format.

## REQUESTS, section 2

The section is `vec<CompileRequest>`. Request indices used elsewhere address
this top-level vector. Nested Choice bodies use the same table.

### CompileRequest

Fields occur in this order:

| field | encoding | meaning |
|---|---|---|
| sources | `vec<CompileSource>` | the unit this compile links, in link order |
| options | `PtxCompileOptions` | exact compilation request |
| prerequisites | `vec<Prerequisite>` | feedback dependencies, by request |

A `CompileSource` is:

| field | encoding | meaning |
|---|---|---|
| source kind | `u8` | `0` PTX |
| PTX ISA | `u8` major, `u8` minor | declared PTX version |
| minimum SM | `u16` | lowest compatible CUDA architecture |
| source blob | `u32` | BLOBS index |

The sources are a sequence, not a set: they are handed to the linker as the
separate inputs they are, never concatenated, so a `call` in one resolves
against a `.func` in another and two orders are two compiles. The sequence is
never empty, and readers reject a request that states no source. The
architecture a unit needs is the highest `minimum SM` among its sources.

`PtxCompileOptions` is:

| field | encoding |
|---|---|
| max registers | `option<u32>` |
| assembly mode | `u8`: `0` environment default, `1` relocatable, `2` whole program |

A `Prerequisite` is a request index `u32` followed by `ResourceBounds`, and
says the referenced request must resolve within those bounds before this
request is feasible. `ResourceBounds` is `min_registers: option<u32>` followed
by `max_local_bytes: option<u32>`.

Prerequisites are one entry per referenced request, in strictly ascending
request order. Two entries for one request would be two statements about it,
and readers that took different ones would disagree; readers reject the
sequence instead.

## BODY, section 3

The section is one `body`. A nested Choice candidate is another `body` inline.
Each body is:

| field | encoding |
|---|---|
| modules | `vec<KernelModule>` |
| workspace | `vec<WorkspaceBuf>` |
| nodes | `vec<Node>` |

### KernelModule

A module is `KernelCode` followed by `ResourceBounds`, the bounds its kernels
must fit whichever image or compile answers.

`KernelCode` is a `u8` tag and its payload:

| tag | payload |
|---:|---|
| `0` PTX-backed | compile request `u32`, then `vec<ModuleImage>` |
| `1` image-only | one `ModuleImage`, then `vec<ModuleImage>` |

A module always has somewhere to get its kernels from: tag `0` names a
compile, and tag `1` carries at least the one image the encoding requires. A
module with neither cannot be written.

Images in one module are compatibility alternatives, not tuning competitors,
and their order is priority: the first a reader can load is the one it uses.
For tag `0` the images come first and the compile is the fallback, so an empty
vector means the module must be compiled. For tag `1` the single leading image
is preferred and the vector holds alternatives.

A `ModuleImage` is a `u8` tag and its payload:

| tag | payload |
|---:|---|
| `0` cubin | target SM `u16`, bytes `u32` BLOBS index, `option<CompileProvenance>` |
| `1` CUDA-loadable | persistence `u8`, bytes `u32` BLOBS index, `option<CompileProvenance>` |

A cubin is assembled for exactly one architecture and names it; a reader uses
it only on that architecture. CUDA-loadable denotes opaque bytes accepted by
`cuLibraryLoadData` — a fatbin, an unlabelled cubin, PTX for the driver's own
JIT — whose compatibility CUDA decides. Because a reader cannot tell those
apart, the image states instead whether its bytes outlive the session that
produced them: persistence `0` persistable, `1` session-only. Session-only
bytes are not written into an artifact another process will load.

`CompileProvenance` is encoded as:

1. 32 raw unit-digest bytes, over the whole compile unit;
2. `PtxCompileOptions`;
3. target SM `u16`;
4. assembler `str`;
5. assembler-version `str`;
6. driver version `option<u32>`;
7. toolkit version `option<u32>`.

Provenance is diagnostic and cache identity. Compatibility is determined by
a cubin's stated target or by the CUDA loader, never by provenance strings.

### WorkspaceBuf

| field | encoding |
|---|---|
| nbytes | `u64` |
| initializer | `option<u32>` | BLOBS index |

### Node

A node is an operation followed by `deps: vec<u32>`, the nodes it runs after.
Every dependency must name a strictly earlier node, making body order a valid
topological order.

Dependencies are a set: one entry per node, in strictly ascending order.
Depending on a node twice is depending on it, and the order two dependencies
are written in says nothing, so a second spelling of the same edges is refused
rather than accepted as a different encoding of one program.

Operation tag `0`, `KernelLaunch`:

| field | encoding |
|---|---|
| module | `u32` |
| entry | `str` |
| grid | three `u32` |
| block | three `u32` |
| dynamic shared memory | `u32` bytes |
| arguments | `vec<Arg>` |

Operation tag `1`, `Memset`:

| field | encoding |
|---|---|
| destination | `WritableAddress` |
| value | `u8` |
| nbytes | `u64` |

Operation tag `2`, `Memcpy`:

| field | encoding |
|---|---|
| source | `Address` |
| destination | `WritableAddress` |
| nbytes | `u64` |

A copy reads its source and writes its destination, so an input may be a
source and can never be a destination.

Operation tag `3`, `Choice`:

| field | encoding |
|---|---|
| input binding | `vec<BufRef>` |
| output binding | `vec<BufRef>` |
| candidates | `vec<body>` |

Candidates are interchangeable bodies under one shared call-site binding. A
candidate's `Input(i)` and `Output(i)` refer to its own formal slots. The
Choice bindings map those slots to buffers in the enclosing body. Candidate
module and initializer indices still address the top-level BLOBS and REQUESTS
sections.

### Arguments and buffers

An `Arg` is `vec<Segment>`. Segment tag `0`, `Bytes`, carries a `blob`.
Segment tag `1`, `Pointer`, carries an `Address`. The ABI argument is the
concatenation of segment payloads, with each Pointer expanded to the bound
64-bit device address plus its byte offset.

Segments are canonical: no `Bytes` segment is empty, and no two `Bytes`
segments are adjacent. The ABI sees the concatenation either way, so an empty
run or a split run is a second spelling of one argument; readers refuse it
rather than joining, which would map two byte strings onto one value.

A `BufRef` is a tag and `u32` index: `0` Input, `1` Output, `2` Workspace.

A `WritableBuf` is a tag and `u32` index: `0` Output, `1` Workspace. Inputs are
read-only and therefore cannot appear here.

An `Address` is a `BufRef` followed by a `u64` byte offset; a
`WritableAddress` is a `WritableBuf` followed by a `u64` byte offset. Every
memory reference in a body is one of the two, so where an operation starts is
stated the same way wherever it appears.

## Required validation

Readers must reject:

- wrong magic or an unsupported version;
- missing required sections;
- section ranges outside the artifact;
- truncation, unknown tags, invalid UTF-8, or lengths that overflow the host;
- BLOBS indices or REQUESTS indices outside their tables;
- compile prerequisites, port aliases, or node dependencies that repeat an
  index or do not ascend;
- an empty `Bytes` segment, or two adjacent `Bytes` segments, in one argument;
- invalid decoded lengths, decompression failures, or digest mismatches;
- recursive Choice bodies that violate the same structural invariants;
- a byte offset or byte length larger than the largest object the reader's
  host can address, which names no byte of any buffer that could be bound;
- an address whose byte offset plus the bytes it touches leaves the address
  space, which has no range to compare against a buffer at all;
- any access — a memset, either side of a memcpy, or a pointer argument's
  address — reaching past the buffer it names. A buffer's extent is the port's
  from SIGNATURE, or, inside a Choice, the extent of the parent buffer that
  candidate's binding selects. The check descends through every candidate, so a
  body nested two Choices deep is still bound by the root's ports;
- a workspace buffer seeded with more bytes than it declares;
- a store whose buffer is read-only where it runs: inside a Choice, a
  candidate writing a formal output that its binding sends to a parent input.

A reader accepts one version and no other. Persistent caches must include the
Recipe wire version in their namespace, or reject and rebuild entries whose
version is not the one this build reads.

## Reference readers

- Rust `recipe::deserialize_shared` keeps raw blobs as views into the caller's
  shared storage and lazily expands zstd entries.
- [`include/recipe.h`](include/recipe.h) is allocation-free and exposes blob
  metadata plus raw payload views. It does not link zstd; callers that need PTX
  decode use their own zstd implementation and verify decoded length/digest.
- Kiln resolves compilation requests, selects a compatible realization, tunes
  Choices, and loads the resulting images.
