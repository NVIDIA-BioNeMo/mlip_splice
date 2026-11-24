// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef RECIPE_H
#define RECIPE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef enum {
  RECIPE_OK = 0,
  RECIPE_TRUNCATED,
  RECIPE_BAD_MAGIC,
  RECIPE_UNSUPPORTED_VERSION,
  RECIPE_MISSING_SECTION,
  RECIPE_BAD_TAG,
  RECIPE_UNSUPPORTED_ENCODING,
  RECIPE_OVERSIZE,
  // A sequence this format states arrived in some other form — out of order,
  // repeated, or empty where the format states at least one element. Reading
  // it anyway would let this reader and the Rust one disagree about what one
  // artifact says.
  RECIPE_NON_CANONICAL
} recipe_status;

typedef struct {
  const uint8_t* ptr;
  uint32_t len;
} recipe_bytes;

typedef struct {
  const uint8_t* p;
  const uint8_t* end;
  recipe_status err;
} recipe_cursor;

typedef struct {
  recipe_cursor blobs;
  recipe_cursor signature;
  recipe_cursor requests;
  recipe_cursor body;
} recipe;

static inline const uint8_t* recipe_take(recipe_cursor* c, size_t n) {
  if (c->err != RECIPE_OK) {
    return NULL;
  }
  if ((size_t)(c->end - c->p) < n) {
    c->err = RECIPE_TRUNCATED;
    return NULL;
  }
  const uint8_t* at = c->p;
  c->p += n;
  return at;
}

static inline uint8_t recipe_rd_u8(recipe_cursor* c) {
  const uint8_t* p = recipe_take(c, 1);
  return p ? p[0] : 0;
}

static inline uint16_t recipe_rd_u16(recipe_cursor* c) {
  const uint8_t* p = recipe_take(c, 2);
  if (!p) {
    return 0;
  }
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static inline uint32_t recipe_rd_u32(recipe_cursor* c) {
  const uint8_t* p = recipe_take(c, 4);
  if (!p) {
    return 0;
  }
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static inline uint64_t recipe_rd_u64(recipe_cursor* c) {
  const uint8_t* p = recipe_take(c, 8);
  if (!p) {
    return 0;
  }
  return (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) |
         ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32) |
         ((uint64_t)p[5] << 40) | ((uint64_t)p[6] << 48) |
         ((uint64_t)p[7] << 56);
}

// A wire u64 that is a byte offset or a byte count. No allocation may exceed
// PTRDIFF_MAX bytes, so a larger number names no byte of any buffer a caller
// could bind; refusing it here is what keeps this reader and the Rust one
// agreeing about which artifacts are readable.
static inline uint64_t recipe_rd_nbytes(recipe_cursor* c) {
  uint64_t value = recipe_rd_u64(c);
  if (value > (uint64_t)PTRDIFF_MAX) {
    if (c->err == RECIPE_OK) {
      c->err = RECIPE_OVERSIZE;
    }
    return 0;
  }
  return value;
}

// Bytes one port extent occupies on the wire: a port can exceed 4 GiB, so
// the extent is 64-bit even though the count beside it is 32.
#define RECIPE_PORT_STRIDE 8u
#define RECIPE_ALIAS_STRIDE 8u

// A counted sequence header, refused unless the cursor still holds the
// `stride` bytes each of its elements needs.
//
// The count comes off untrusted bytes. Without this bound a malformed one
// drives a loop the section could never back — up to 2^32 iterations of a
// reader that has already failed — so it is rejected here, before it is
// believed. Returns 0 and marks the cursor truncated when it does not fit.
static inline uint32_t recipe_rd_count(recipe_cursor* c, uint32_t stride) {
  uint32_t n = recipe_rd_u32(c);
  if (c->err != RECIPE_OK) {
    return 0;
  }
  uint64_t needed = (uint64_t)n * (uint64_t)stride;
  if (needed > (uint64_t)(size_t)(c->end - c->p)) {
    c->err = RECIPE_TRUNCATED;
    return 0;
  }
  return n;
}

static inline recipe_bytes recipe_rd_blob(recipe_cursor* c) {
  recipe_bytes b = {NULL, 0};
  uint32_t len = recipe_rd_u32(c);
  const uint8_t* p = recipe_take(c, len);
  if (p) {
    b.ptr = p;
    b.len = len;
  }
  return b;
}

static inline recipe_bytes recipe_rd_str(recipe_cursor* c) {
  return recipe_rd_blob(c);
}

typedef enum {
  RECIPE_BLOB_RAW = 0,
  RECIPE_BLOB_ZSTD = 1
} recipe_blob_encoding;

typedef struct {
  recipe_blob_encoding encoding;
  uint64_t decoded_len;
  const uint8_t* digest;
  recipe_bytes payload;
} recipe_blob_entry;

static inline recipe_blob_entry recipe_rd_blob_entry(recipe_cursor* c) {
  recipe_blob_entry entry;
  memset(&entry, 0, sizeof(entry));
  uint8_t encoding = recipe_rd_u8(c);
  if (encoding > RECIPE_BLOB_ZSTD) {
    if (c->err == RECIPE_OK) {
      c->err = RECIPE_BAD_TAG;
    }
    return entry;
  }
  entry.encoding = (recipe_blob_encoding)encoding;
  entry.decoded_len = recipe_rd_u64(c);
  entry.digest = recipe_take(c, 32);
  entry.payload = recipe_rd_blob(c);
  return entry;
}

// Resolves a raw blob-pool entry. Encoded entries require a decoder and are
// exposed through recipe_blob_entry_at instead.
static inline recipe_blob_entry recipe_blob_entry_at(const recipe* r,
                                                       uint32_t index) {
  recipe_blob_entry none;
  memset(&none, 0, sizeof(none));
  recipe_cursor c = r->blobs;
  // A blob entry is at least its 45 bytes of metadata.
  uint32_t count = recipe_rd_count(&c, 45);
  if (index >= count) {
    return none;
  }
  for (uint32_t i = 0; i < index; i++) {
    (void)recipe_rd_blob_entry(&c);
  }
  recipe_blob_entry entry = recipe_rd_blob_entry(&c);
  if (c.err != RECIPE_OK) {
    return none;
  }
  return entry;
}

static inline recipe_bytes recipe_blob_at(const recipe* r, uint32_t index) {
  recipe_bytes none = {NULL, 0};
  recipe_blob_entry entry = recipe_blob_entry_at(r, index);
  if (entry.encoding != RECIPE_BLOB_RAW || entry.payload.ptr == NULL ||
      entry.decoded_len != entry.payload.len) {
    return none;
  }
  return entry.payload;
}

typedef struct {
  int present;
  uint32_t value;
} recipe_optional_u32;

static inline recipe_optional_u32 recipe_rd_optional_u32(recipe_cursor* c) {
  recipe_optional_u32 value = {0, 0};
  uint8_t tag = recipe_rd_u8(c);
  if (tag == 1) {
    value.present = 1;
    value.value = recipe_rd_u32(c);
  } else if (tag != 0 && c->err == RECIPE_OK) {
    c->err = RECIPE_BAD_TAG;
  }
  return value;
}

typedef enum {
  RECIPE_ASSEMBLY_ENVIRONMENT_DEFAULT = 0,
  RECIPE_ASSEMBLY_RELOCATABLE = 1,
  RECIPE_ASSEMBLY_WHOLE_PROGRAM = 2
} recipe_assembly_mode;

typedef struct {
  recipe_optional_u32 max_registers;
  recipe_assembly_mode assembly_mode;
} recipe_compile_options;

static inline recipe_compile_options recipe_rd_compile_options(recipe_cursor* c) {
  recipe_compile_options options;
  options.max_registers = recipe_rd_optional_u32(c);
  uint8_t tag = recipe_rd_u8(c);
  if (tag > RECIPE_ASSEMBLY_WHOLE_PROGRAM) {
    if (c->err == RECIPE_OK) {
      c->err = RECIPE_BAD_TAG;
    }
    tag = RECIPE_ASSEMBLY_ENVIRONMENT_DEFAULT;
  }
  options.assembly_mode = (recipe_assembly_mode)tag;
  return options;
}

typedef struct {
  recipe_optional_u32 min_registers;
  recipe_optional_u32 max_local_bytes;
} recipe_resource_bounds;

static inline recipe_resource_bounds recipe_rd_resource_bounds(recipe_cursor* c) {
  recipe_resource_bounds bounds;
  bounds.min_registers = recipe_rd_optional_u32(c);
  bounds.max_local_bytes = recipe_rd_optional_u32(c);
  return bounds;
}

typedef enum {
  RECIPE_SOURCE_PTX = 0
} recipe_source_kind;

// One module of a compile unit: what it is, the ISA it is written in, the
// lowest architecture it runs on, and where its text is pooled.
typedef struct {
  recipe_source_kind kind;
  uint8_t ptx_isa[2];
  uint16_t minimum_sm;
  uint32_t blob;
} recipe_compile_source;

static inline recipe_compile_source recipe_rd_compile_source(recipe_cursor* c) {
  recipe_compile_source source;
  memset(&source, 0, sizeof(source));
  uint8_t kind = recipe_rd_u8(c);
  if (kind != RECIPE_SOURCE_PTX && c->err == RECIPE_OK) {
    c->err = RECIPE_BAD_TAG;
  }
  source.kind = RECIPE_SOURCE_PTX;
  source.ptx_isa[0] = recipe_rd_u8(c);
  source.ptx_isa[1] = recipe_rd_u8(c);
  source.minimum_sm = recipe_rd_u16(c);
  source.blob = recipe_rd_u32(c);
  return source;
}

// The unit one compile links, in link order. `entries` is an independent
// window positioned at the first source, so a caller re-reads them without
// disturbing the cursor they were read from. A unit is never empty — a compile
// that links nothing is not one this format states.
typedef struct {
  recipe_cursor entries;
  uint32_t count;
} recipe_compile_sources;

typedef struct {
  uint32_t request;
  recipe_resource_bounds bounds;
} recipe_resource_prerequisite;

static inline recipe_resource_prerequisite recipe_rd_resource_prerequisite(
    recipe_cursor* c) {
  recipe_resource_prerequisite prerequisite;
  prerequisite.request = recipe_rd_u32(c);
  prerequisite.bounds = recipe_rd_resource_bounds(c);
  return prerequisite;
}

// What a compile says about the compiles it must agree with: one entry per
// request it names, in ascending order. `entries` is an independent window
// positioned at the first, so a caller re-reads them without disturbing the
// cursor it was read from.
typedef struct {
  recipe_cursor entries;
  uint32_t count;
} recipe_prerequisites;

typedef struct {
  recipe_compile_sources sources;
  recipe_compile_options options;
  recipe_prerequisites prerequisites;
} recipe_compile_request;

// One source on the wire: a kind byte, a two-byte PTX ISA, a `u16` minimum SM
// and a `u32` blob index.
#define RECIPE_COMPILE_SOURCE_STRIDE 9u

static inline recipe_compile_request recipe_rd_compile_request(recipe_cursor* c) {
  recipe_compile_request request;
  memset(&request, 0, sizeof(request));
  request.sources.count = recipe_rd_count(c, RECIPE_COMPILE_SOURCE_STRIDE);
  if (request.sources.count == 0 && c->err == RECIPE_OK) {
    c->err = RECIPE_NON_CANONICAL;
  }
  request.sources.entries = *c;
  for (uint32_t i = 0; i < request.sources.count; i++) {
    (void)recipe_rd_compile_source(c);
  }
  request.sources.entries.end = c->p;
  request.options = recipe_rd_compile_options(c);
  // A prerequisite is a u32 request plus two optional u32 (a tag byte each
  // when absent), so six bytes at least.
  request.prerequisites.count = recipe_rd_count(c, 6);
  request.prerequisites.entries = *c;
  uint32_t previous = 0;
  for (uint32_t i = 0; i < request.prerequisites.count; i++) {
    recipe_resource_prerequisite entry = recipe_rd_resource_prerequisite(c);
    if (i > 0 && entry.request <= previous && c->err == RECIPE_OK) {
      c->err = RECIPE_NON_CANONICAL;
    }
    previous = entry.request;
  }
  request.prerequisites.entries.end = c->p;
  return request;
}

typedef enum {
  RECIPE_IMAGE_CUBIN = 0,
  RECIPE_IMAGE_CUDA_LOADABLE = 1
} recipe_image_kind;

typedef enum {
  RECIPE_IMAGE_PERSISTABLE = 0,
  RECIPE_IMAGE_SESSION_ONLY = 1
} recipe_image_persistence;

// What an image was produced from: a digest over the whole compile unit, in
// link order, rather than over one of its sources.
typedef struct {
  const uint8_t* unit_digest;
  recipe_compile_options options;
  uint16_t target_sm;
  recipe_bytes assembler;
  recipe_bytes assembler_version;
  recipe_optional_u32 driver_version;
  recipe_optional_u32 toolkit_version;
} recipe_compile_provenance;

static inline recipe_compile_provenance recipe_rd_compile_provenance(
    recipe_cursor* c) {
  recipe_compile_provenance provenance;
  memset(&provenance, 0, sizeof(provenance));
  provenance.unit_digest = recipe_take(c, 32);
  provenance.options = recipe_rd_compile_options(c);
  provenance.target_sm = recipe_rd_u16(c);
  provenance.assembler = recipe_rd_str(c);
  provenance.assembler_version = recipe_rd_str(c);
  provenance.driver_version = recipe_rd_optional_u32(c);
  provenance.toolkit_version = recipe_rd_optional_u32(c);
  return provenance;
}

typedef struct {
  recipe_image_kind kind;
  uint16_t target_sm;                    // valid for Cubin
  recipe_image_persistence persistence;  // valid for CudaLoadable
  uint32_t blob;
  int has_provenance;
  recipe_compile_provenance provenance;
} recipe_module_image;

static inline recipe_module_image recipe_rd_module_image(recipe_cursor* c) {
  recipe_module_image image;
  memset(&image, 0, sizeof(image));
  uint8_t kind = recipe_rd_u8(c);
  if (kind > RECIPE_IMAGE_CUDA_LOADABLE) {
    if (c->err == RECIPE_OK) {
      c->err = RECIPE_BAD_TAG;
    }
    return image;
  }
  image.kind = (recipe_image_kind)kind;
  if (image.kind == RECIPE_IMAGE_CUBIN) {
    image.target_sm = recipe_rd_u16(c);
  } else {
    uint8_t persistence = recipe_rd_u8(c);
    if (persistence > RECIPE_IMAGE_SESSION_ONLY) {
      if (c->err == RECIPE_OK) {
        c->err = RECIPE_BAD_TAG;
      }
      return image;
    }
    image.persistence = (recipe_image_persistence)persistence;
  }
  image.blob = recipe_rd_u32(c);
  uint8_t has_provenance = recipe_rd_u8(c);
  if (has_provenance > 1) {
    if (c->err == RECIPE_OK) {
      c->err = RECIPE_BAD_TAG;
    }
    return image;
  }
  image.has_provenance = has_provenance;
  if (has_provenance) {
    image.provenance = recipe_rd_compile_provenance(c);
  }
  return image;
}

typedef enum {
  RECIPE_CODE_PTX_BACKED = 0,
  RECIPE_CODE_IMAGE_ONLY = 1
} recipe_kernel_code_kind;

// A module always has somewhere to get its kernels from, so there is no
// "neither" to check for: tag 0 names a compile, tag 1 carries a preferred
// image the encoding requires. `image_count` counts the images that follow the
// header, most preferred first; for tag 1 the preferred image has already been
// read into `preferred_image`.
typedef struct {
  recipe_kernel_code_kind kind;
  uint32_t request;  // valid for PtxBacked
  recipe_module_image preferred_image;  // valid for ImageOnly
  uint32_t image_count;
} recipe_kernel_module;

static inline recipe_kernel_module recipe_rd_kernel_module(recipe_cursor* c) {
  recipe_kernel_module module;
  memset(&module, 0, sizeof(module));
  uint8_t kind = recipe_rd_u8(c);
  if (kind > RECIPE_CODE_IMAGE_ONLY) {
    if (c->err == RECIPE_OK) {
      c->err = RECIPE_BAD_TAG;
    }
    return module;
  }
  module.kind = (recipe_kernel_code_kind)kind;
  if (module.kind == RECIPE_CODE_PTX_BACKED) {
    module.request = recipe_rd_u32(c);
  } else {
    module.preferred_image = recipe_rd_module_image(c);
  }
  // An image is at least a kind, a blob index and a provenance tag.
  module.image_count = recipe_rd_count(c, 6);
  return module;
}

typedef struct {
  uint64_t nbytes;
  int has_init;
  uint32_t init_blob;  // pool index, valid when has_init
} recipe_workspace_buf;

static inline recipe_workspace_buf recipe_rd_workspace_buf(recipe_cursor* c) {
  recipe_workspace_buf w;
  w.nbytes = recipe_rd_nbytes(c);
  w.has_init = recipe_rd_u8(c);
  w.init_blob = 0;
  if (w.has_init) {
    w.init_blob = recipe_rd_u32(c);
  }
  return w;
}

typedef enum {
  RECIPE_NODE_KERNEL_LAUNCH = 0,
  RECIPE_NODE_MEMSET = 1,
  RECIPE_NODE_MEMCPY = 2,
  RECIPE_NODE_CHOICE = 3
} recipe_node_kind;

typedef enum {
  RECIPE_SEGMENT_BYTES = 0,
  RECIPE_SEGMENT_POINTER = 1
} recipe_segment_tag;

typedef enum {
  RECIPE_BUFREF_INPUT = 0,
  RECIPE_BUFREF_OUTPUT = 1,
  RECIPE_BUFREF_WORKSPACE = 2
} recipe_bufref_tag;

typedef struct {
  recipe_bufref_tag tag;
  uint32_t index;
} recipe_bufref;

static inline recipe_bufref recipe_rd_bufref(recipe_cursor* c) {
  recipe_bufref b;
  uint8_t tag = recipe_rd_u8(c);
  if (tag > RECIPE_BUFREF_WORKSPACE) {
    if (c->err == RECIPE_OK) {
      c->err = RECIPE_BAD_TAG;
    }
    b.tag = RECIPE_BUFREF_INPUT;
    b.index = 0;
    return b;
  }
  b.tag = (recipe_bufref_tag)tag;
  b.index = recipe_rd_u32(c);
  return b;
}

typedef enum {
  RECIPE_WRITABLE_OUTPUT = 0,
  RECIPE_WRITABLE_WORKSPACE = 1
} recipe_writablebuf_tag;

typedef struct {
  recipe_writablebuf_tag tag;
  uint32_t index;
} recipe_writablebuf;

static inline recipe_writablebuf recipe_rd_writablebuf(recipe_cursor* c) {
  recipe_writablebuf b;
  uint8_t tag = recipe_rd_u8(c);
  if (tag > RECIPE_WRITABLE_WORKSPACE) {
    if (c->err == RECIPE_OK) {
      c->err = RECIPE_BAD_TAG;
    }
    b.tag = RECIPE_WRITABLE_OUTPUT;
    b.index = 0;
    return b;
  }
  b.tag = (recipe_writablebuf_tag)tag;
  b.index = recipe_rd_u32(c);
  return b;
}

// Where in a buffer an operation starts. Every memory reference in a body is
// one of these two, so the byte offset is read the same way everywhere.
typedef struct {
  recipe_bufref buf;
  uint64_t byte_offset;
} recipe_address;

typedef struct {
  recipe_writablebuf buf;
  uint64_t byte_offset;
} recipe_writable_address;

static inline recipe_address recipe_rd_address(recipe_cursor* c) {
  recipe_address a;
  a.buf = recipe_rd_bufref(c);
  a.byte_offset = recipe_rd_nbytes(c);
  return a;
}

static inline recipe_writable_address recipe_rd_writable_address(recipe_cursor* c) {
  recipe_writable_address a;
  a.buf = recipe_rd_writablebuf(c);
  a.byte_offset = recipe_rd_nbytes(c);
  return a;
}

typedef struct {
  recipe_segment_tag tag;
  recipe_bytes bytes;       // valid for Bytes
  recipe_address address;   // valid for Pointer
} recipe_segment;

static inline recipe_segment recipe_rd_segment(recipe_cursor* c) {
  recipe_segment s;
  s.bytes.ptr = NULL;
  s.bytes.len = 0;
  s.address.buf.tag = RECIPE_BUFREF_INPUT;
  s.address.buf.index = 0;
  s.address.byte_offset = 0;
  uint8_t tag = recipe_rd_u8(c);
  switch (tag) {
    case RECIPE_SEGMENT_BYTES:
      s.tag = RECIPE_SEGMENT_BYTES;
      s.bytes = recipe_rd_blob(c);
      break;
    case RECIPE_SEGMENT_POINTER:
      s.tag = RECIPE_SEGMENT_POINTER;
      s.address = recipe_rd_address(c);
      break;
    default:
      if (c->err == RECIPE_OK) {
        c->err = RECIPE_BAD_TAG;
      }
      s.tag = RECIPE_SEGMENT_BYTES;
      break;
  }
  return s;
}

// One ABI argument: a run of segments whose payloads concatenate. `segments`
// is an independent window positioned at the first.
typedef struct {
  recipe_cursor segments;
  uint32_t count;
} recipe_arg;

// Reads one argument, refusing an empty byte run or two adjacent ones.
//
// The concatenation is the same either way, so those are second spellings of
// an argument that already has one. A reader that accepted them would let two
// artifacts holding one argument disagree about how many segments it has.
static inline recipe_arg recipe_rd_arg(recipe_cursor* c) {
  recipe_arg a;
  // A segment is at least a tag byte and a u32 length or a bufref.
  a.count = recipe_rd_count(c, 5);
  a.segments = *c;
  int previous_was_bytes = 0;
  for (uint32_t i = 0; i < a.count; i++) {
    recipe_segment s = recipe_rd_segment(c);
    int is_bytes = s.tag == RECIPE_SEGMENT_BYTES;
    if (is_bytes && (s.bytes.len == 0 || previous_was_bytes) &&
        c->err == RECIPE_OK) {
      c->err = RECIPE_NON_CANONICAL;
    }
    previous_was_bytes = is_bytes;
  }
  a.segments.end = c->p;
  return a;
}

// The nodes one node runs after: a set, so one entry per node in ascending
// order. `entries` is an independent window positioned at the first.
typedef struct {
  recipe_cursor entries;
  uint32_t count;
} recipe_deps;

static inline recipe_deps recipe_rd_deps(recipe_cursor* c) {
  recipe_deps d;
  d.count = recipe_rd_count(c, 4);
  d.entries = *c;
  uint32_t previous = 0;
  for (uint32_t i = 0; i < d.count; i++) {
    uint32_t dep = recipe_rd_u32(c);
    if (i > 0 && dep <= previous && c->err == RECIPE_OK) {
      c->err = RECIPE_NON_CANONICAL;
    }
    previous = dep;
  }
  d.entries.end = c->p;
  return d;
}

// A node's operation. Its `deps` follow it on the wire, but a Choice's
// candidates come between, so the caller reads them with `recipe_rd_deps`
// once it has walked whatever tail its kind has.
typedef struct {
  recipe_node_kind kind;
  union {
    struct {
      uint32_t module;
      recipe_bytes entry;
      uint32_t grid[3];
      uint32_t block[3];
      uint32_t shmem;
      // An independent window over the argument list, positioned at the first
      // argument's segment count. Every argument in it has been read once, so
      // the canonical form is proved before a caller looks at any of them.
      recipe_cursor args;
      uint32_t arg_count;
    } launch;
    struct {
      recipe_writable_address destination;
      uint8_t value;
      uint64_t nbytes;
    } memset;
    struct {
      recipe_address source;
      recipe_writable_address destination;
      uint64_t nbytes;
    } memcpy_;
  };
} recipe_node;

static inline recipe_node recipe_rd_node(recipe_cursor* c) {
  recipe_node n;
  memset(&n, 0, sizeof(n));
  uint8_t kind = recipe_rd_u8(c);
  switch (kind) {
    case RECIPE_NODE_KERNEL_LAUNCH:
      n.kind = RECIPE_NODE_KERNEL_LAUNCH;
      n.launch.module = recipe_rd_u32(c);
      n.launch.entry = recipe_rd_str(c);
      for (int i = 0; i < 3; i++) {
        n.launch.grid[i] = recipe_rd_u32(c);
      }
      for (int i = 0; i < 3; i++) {
        n.launch.block[i] = recipe_rd_u32(c);
      }
      n.launch.shmem = recipe_rd_u32(c);
      n.launch.arg_count = recipe_rd_count(c, 4);
      n.launch.args = *c;
      for (uint32_t i = 0; i < n.launch.arg_count; i++) {
        (void)recipe_rd_arg(c);
      }
      n.launch.args.end = c->p;
      break;
    case RECIPE_NODE_MEMSET:
      n.kind = RECIPE_NODE_MEMSET;
      n.memset.destination = recipe_rd_writable_address(c);
      n.memset.value = recipe_rd_u8(c);
      n.memset.nbytes = recipe_rd_nbytes(c);
      break;
    case RECIPE_NODE_MEMCPY:
      n.kind = RECIPE_NODE_MEMCPY;
      n.memcpy_.source = recipe_rd_address(c);
      n.memcpy_.destination = recipe_rd_writable_address(c);
      n.memcpy_.nbytes = recipe_rd_nbytes(c);
      break;
    case RECIPE_NODE_CHOICE:
      n.kind = RECIPE_NODE_CHOICE;  // tail is `candidates`, read by the caller
      break;
    default:
      if (c->err == RECIPE_OK) {
        c->err = RECIPE_BAD_TAG;
      }
      n.kind = RECIPE_NODE_KERNEL_LAUNCH;
      break;
  }
  return n;
}

// A Choice node's shared interface. Each binding cursor is positioned at its
// leading `u32` count. `candidates` is positioned at the `u32`
// candidate_count; each candidate follows inline as a body (modules,
// workspace, nodes — same layout as the top-level BODY section), whose module
// and init entries index the shared top-level blob pool. The returned binding
// cursors are independent windows that re-read each vector from its count.
typedef struct {
  recipe_cursor input_binding;
  recipe_cursor output_binding;
  recipe_cursor candidates;
} recipe_choice;

// One `BufRef` on the wire: a tag byte and a `u32` index.
#define RECIPE_BUFREF_STRIDE 5u

static inline recipe_choice recipe_rd_choice(recipe_cursor* c) {
  recipe_choice ch;
  memset(&ch, 0, sizeof(ch));
  // Both counts are bounded before the `end` pointers below are formed:
  // `c->p + n * stride` past the end of the section is out-of-range pointer
  // arithmetic, undefined before anyone dereferences it, so the count is
  // refused first and the arithmetic then stays inside the object.
  ch.input_binding = *c;
  uint32_t n_in = recipe_rd_count(c, RECIPE_BUFREF_STRIDE);
  ch.input_binding.end = c->p + (size_t)n_in * RECIPE_BUFREF_STRIDE;
  for (uint32_t i = 0; i < n_in; i++) {
    (void)recipe_rd_bufref(c);
  }
  ch.output_binding = *c;
  uint32_t n_out = recipe_rd_count(c, RECIPE_BUFREF_STRIDE);
  ch.output_binding.end = c->p + (size_t)n_out * RECIPE_BUFREF_STRIDE;
  for (uint32_t i = 0; i < n_out; i++) {
    (void)recipe_rd_bufref(c);
  }
  ch.candidates = *c;
  return ch;
}

// Section IDs: 1=BLOBS, 2=REQUESTS, 3=BODY, 4=SIGNATURE. All four are
// required. Writers emit them in ID order with contiguous payloads; readers
// find each by its own offset and skip IDs they do not know.
static inline recipe_status recipe_open(const uint8_t* bytes, size_t len,
                                         recipe* out) {
  recipe_cursor c = {bytes, bytes + len, RECIPE_OK};

  const uint8_t* magic = recipe_take(&c, 4);
  if (!magic) {
    return RECIPE_TRUNCATED;
  }
  if (magic[0] != 'R' || magic[1] != 'C' || magic[2] != 'P' ||
      magic[3] != 'E') {
    return RECIPE_BAD_MAGIC;
  }
  uint16_t version = recipe_rd_u16(&c);
  if (c.err != RECIPE_OK) {
    return RECIPE_TRUNCATED;
  }
  if (version != 13) {
    return RECIPE_UNSUPPORTED_VERSION;
  }
  (void)recipe_rd_u16(&c);  // reserved
  // Bounded before the walk: one entry is 20 bytes, so a count the header
  // cannot back is truncation rather than a loop that discovers it late.
  uint32_t section_count = recipe_rd_count(&c, 20);
  if (c.err != RECIPE_OK) {
    return RECIPE_TRUNCATED;
  }

  int have_blobs = 0, have_signature = 0, have_requests = 0, have_body = 0;
  for (uint32_t i = 0; i < section_count; i++) {
    uint32_t id = recipe_rd_u32(&c);
    uint64_t offset = recipe_rd_u64(&c);
    uint64_t length = recipe_rd_u64(&c);
    if (c.err != RECIPE_OK) {
      return RECIPE_TRUNCATED;
    }
    if (offset > len || length > (uint64_t)len - offset) {
      return RECIPE_TRUNCATED;
    }
    recipe_cursor payload = {bytes + offset, bytes + offset + length, RECIPE_OK};
    switch (id) {
      case 1:
        out->blobs = payload;
        have_blobs = 1;
        break;
      case 2:
        out->requests = payload;
        have_requests = 1;
        break;
      case 3:
        out->body = payload;
        have_body = 1;
        break;
      case 4:
        out->signature = payload;
        have_signature = 1;
        break;
      default:
        break;  // forward-compat: skip unknown sections
    }
  }

  if (!have_blobs) {
    return RECIPE_MISSING_SECTION;
  }
  if (!have_signature) {
    return RECIPE_MISSING_SECTION;
  }
  if (!have_requests) {
    return RECIPE_MISSING_SECTION;
  }
  if (!have_body) {
    return RECIPE_MISSING_SECTION;
  }
  return RECIPE_OK;
}

// The external ports of a recipe: input extents then output extents, each a
// counted sequence of byte sizes in port order. Only the root carries these; a
// choice candidate's ports are the parent buffers its bindings select.
//
// `count` is bounded by the section, so iterating `0..count` is always backed
// by bytes. A malformed signature yields `count == 0` and a cursor in error.
typedef struct {
  recipe_cursor ports;
  uint32_t count;
} recipe_port_extents;

typedef struct {
  uint32_t input;
  uint32_t output;
} recipe_port_alias;

typedef struct {
  recipe_cursor entries;
  uint32_t count;
} recipe_port_aliases;

static inline recipe_port_extents recipe_inputs(const recipe* r) {
  recipe_port_extents extents;
  extents.ports = r->signature;
  extents.count = recipe_rd_count(&extents.ports, RECIPE_PORT_STRIDE);
  return extents;
}

// Skips the inputs by their bounded extent rather than by reading them one at
// a time, so a stated count never costs a loop.
static inline recipe_port_extents recipe_outputs(const recipe* r) {
  recipe_cursor c = r->signature;
  uint32_t inputs = recipe_rd_count(&c, RECIPE_PORT_STRIDE);
  (void)recipe_take(&c, (size_t)inputs * RECIPE_PORT_STRIDE);
  recipe_port_extents extents;
  extents.ports = c;
  extents.count = recipe_rd_count(&extents.ports, RECIPE_PORT_STRIDE);
  return extents;
}

static inline recipe_port_aliases recipe_aliases(const recipe* r) {
  recipe_cursor c = r->signature;
  uint32_t inputs = recipe_rd_count(&c, RECIPE_PORT_STRIDE);
  (void)recipe_take(&c, (size_t)inputs * RECIPE_PORT_STRIDE);
  uint32_t outputs = recipe_rd_count(&c, RECIPE_PORT_STRIDE);
  (void)recipe_take(&c, (size_t)outputs * RECIPE_PORT_STRIDE);
  recipe_port_aliases aliases;
  aliases.count = recipe_rd_count(&c, RECIPE_ALIAS_STRIDE);
  aliases.entries = c;
  uint32_t previous = 0;
  for (uint32_t i = 0; i < aliases.count; i++) {
    uint32_t input = recipe_rd_u32(&c);
    (void)recipe_rd_u32(&c);
    if (i > 0 && input <= previous && c.err == RECIPE_OK) {
      c.err = RECIPE_NON_CANONICAL;
    }
    previous = input;
  }
  aliases.entries.end = c.p;
  if (c.err != RECIPE_OK) {
    aliases.entries.err = c.err;
    aliases.count = 0;
  }
  return aliases;
}

static inline recipe_port_alias recipe_rd_port_alias(recipe_cursor* c) {
  recipe_port_alias alias;
  alias.input = recipe_rd_u32(c);
  alias.output = recipe_rd_u32(c);
  return alias;
}

// The byte extent of one port, or zero past the end. Ports are addressed
// directly, so reading the last one costs the same as reading the first.
static inline uint64_t recipe_port_nbytes(recipe_port_extents* extents,
                                          uint32_t port) {
  if (port >= extents->count) {
    return 0;
  }
  recipe_cursor c = extents->ports;
  (void)recipe_take(&c, (size_t)port * RECIPE_PORT_STRIDE);
  return recipe_rd_nbytes(&c);
}

#endif  // RECIPE_H
