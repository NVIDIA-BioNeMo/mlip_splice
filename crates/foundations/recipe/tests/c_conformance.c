// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "recipe.h"

#define CHECK(cond, code) \
  do {                    \
    if (!(cond)) {        \
      return (code);      \
    }                     \
  } while (0)

static int bytes_eq(recipe_bytes b, const uint8_t* want, uint32_t n) {
  return b.len == n && b.ptr != NULL && memcmp(b.ptr, want, n) == 0;
}

static uint8_t* read_file(const char* path, size_t* size) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    return NULL;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  long n = ftell(f);
  if (n < 0) {
    fclose(f);
    return NULL;
  }
  rewind(f);
  uint8_t* buf = (uint8_t*)malloc((size_t)n);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
    free(buf);
    fclose(f);
    return NULL;
  }
  fclose(f);
  *size = (size_t)n;
  return buf;
}

// Set by `check_canonical` to where the last node's dependency count sits in
// the canonical artifact — the one place it states an order between two
// dependencies, and so the one the corruption checks below can patch.
static size_t canonical_deps_at = 0;

static int check_canonical(const uint8_t* buf, size_t size) {
  recipe rcp;
  CHECK(recipe_open(buf, size, &rcp) == RECIPE_OK, 10);

  // The ports, read from the root's own section: two inputs of distinct sizes
  // and one output, so reading the wrong port or the wrong side is visible.
  recipe_port_extents inputs = recipe_inputs(&rcp);
  recipe_port_extents outputs = recipe_outputs(&rcp);
  CHECK(inputs.count == 2, 1);
  CHECK(outputs.count == 1, 2);
  CHECK(recipe_port_nbytes(&inputs, 0) == 48, 3);
  CHECK(recipe_port_nbytes(&inputs, 1) == 96, 4);
  CHECK(recipe_port_nbytes(&outputs, 0) == 16, 5);
  CHECK(recipe_port_nbytes(&inputs, 2) == 0, 6);  // past the end
  // The extent is 64-bit on both sides: a port past 4 GiB reads back whole
  // rather than wrapping into a small one the checks would then accept.
  CHECK(recipe_port_nbytes(&outputs, 0) == (uint64_t)16, 7);

  CHECK(recipe_rd_u32(&rcp.requests) == 0, 9);  // compile request count

  CHECK(recipe_rd_u32(&rcp.body) == 1, 11);  // module count
  recipe_kernel_module module = recipe_rd_kernel_module(&rcp.body);
  CHECK(module.kind == RECIPE_CODE_IMAGE_ONLY && module.image_count == 0, 110);
  recipe_module_image image = module.preferred_image;
  CHECK(image.kind == RECIPE_IMAGE_CUDA_LOADABLE &&
            image.persistence == RECIPE_IMAGE_PERSISTABLE &&
            !image.has_provenance,
        111);
  recipe_resource_bounds module_bounds =
      recipe_rd_resource_bounds(&rcp.body);
  CHECK(!module_bounds.min_registers.present, 111);
  CHECK(!module_bounds.max_local_bytes.present, 112);
  static const uint8_t want_cubin[4] = {0xDE, 0xAD, 0xBE, 0xEF};
  CHECK(bytes_eq(recipe_blob_at(&rcp, image.blob), want_cubin, 4), 12);
  CHECK(rcp.body.err == RECIPE_OK, 13);

  CHECK(recipe_rd_u32(&rcp.body) == 1, 20);  // workspace count
  recipe_workspace_buf w = recipe_rd_workspace_buf(&rcp.body);
  CHECK(w.nbytes == 4, 21);
  CHECK(w.has_init, 22);
  static const uint8_t want_init[4] = {0x0A, 0x00, 0x00, 0x00};
  CHECK(bytes_eq(recipe_blob_at(&rcp, w.init_blob), want_init, 4), 23);
  CHECK(rcp.body.err == RECIPE_OK, 24);

  CHECK(recipe_rd_u32(&rcp.body) == 3, 30);  // node count

  recipe_node n0 = recipe_rd_node(&rcp.body);
  CHECK(n0.kind == RECIPE_NODE_MEMSET, 31);
  CHECK(n0.memset.destination.buf.tag == RECIPE_WRITABLE_OUTPUT, 32);
  CHECK(n0.memset.destination.buf.index == 0, 33);
  CHECK(n0.memset.destination.byte_offset == 0, 330);
  CHECK(n0.memset.value == 0, 34);
  CHECK(n0.memset.nbytes == 16, 35);
  CHECK(recipe_rd_deps(&rcp.body).count == 0, 36);
  CHECK(rcp.body.err == RECIPE_OK, 37);

  recipe_node n1 = recipe_rd_node(&rcp.body);
  CHECK(n1.kind == RECIPE_NODE_KERNEL_LAUNCH, 40);
  CHECK(n1.launch.module == 0, 41);
  CHECK(bytes_eq(n1.launch.entry, (const uint8_t*)"fill", 4), 42);
  CHECK(n1.launch.grid[0] == 1 && n1.launch.grid[1] == 1 &&
            n1.launch.grid[2] == 1,
        43);
  CHECK(n1.launch.block[0] == 256 && n1.launch.block[1] == 1 &&
            n1.launch.block[2] == 1,
        44);
  CHECK(n1.launch.shmem == 0, 45);

  CHECK(n1.launch.arg_count == 3, 46);
  recipe_cursor args = n1.launch.args;
  recipe_arg arg0 = recipe_rd_arg(&args);
  CHECK(arg0.count == 1, 47);
  recipe_segment a0 = recipe_rd_segment(&arg0.segments);
  CHECK(a0.tag == RECIPE_SEGMENT_POINTER &&
            a0.address.buf.tag == RECIPE_BUFREF_INPUT &&
            a0.address.buf.index == 0 && a0.address.byte_offset == 0,
        470);
  recipe_arg arg1 = recipe_rd_arg(&args);
  CHECK(arg1.count == 1, 48);
  recipe_segment a1 = recipe_rd_segment(&arg1.segments);
  CHECK(a1.tag == RECIPE_SEGMENT_POINTER &&
            a1.address.buf.tag == RECIPE_BUFREF_WORKSPACE &&
            a1.address.buf.index == 0 && a1.address.byte_offset == 0,
        480);
  recipe_arg arg2 = recipe_rd_arg(&args);
  CHECK(arg2.count == 1, 49);
  recipe_segment a2 = recipe_rd_segment(&arg2.segments);
  CHECK(a2.tag == RECIPE_SEGMENT_POINTER &&
            a2.address.buf.tag == RECIPE_BUFREF_OUTPUT &&
            a2.address.buf.index == 0 && a2.address.byte_offset == 4,
        490);
  CHECK(args.p == args.end, 491);

  recipe_deps d1 = recipe_rd_deps(&rcp.body);
  CHECK(d1.count == 1, 50);
  CHECK(recipe_rd_u32(&d1.entries) == 0, 51);
  CHECK(rcp.body.err == RECIPE_OK, 52);

  recipe_node n2 = recipe_rd_node(&rcp.body);
  CHECK(n2.kind == RECIPE_NODE_MEMCPY, 53);
  CHECK(n2.memcpy_.source.buf.tag == RECIPE_BUFREF_INPUT &&
            n2.memcpy_.source.buf.index == 1 &&
            n2.memcpy_.source.byte_offset == 8,
        54);
  CHECK(n2.memcpy_.destination.buf.tag == RECIPE_WRITABLE_OUTPUT &&
            n2.memcpy_.destination.buf.index == 0 &&
            n2.memcpy_.destination.byte_offset == 4,
        55);
  CHECK(n2.memcpy_.nbytes == 8, 56);
  canonical_deps_at = (size_t)(rcp.body.p - buf);
  recipe_deps d2 = recipe_rd_deps(&rcp.body);
  CHECK(d2.count == 2, 57);
  CHECK(recipe_rd_u32(&d2.entries) == 0, 58);
  CHECK(recipe_rd_u32(&d2.entries) == 1, 580);
  CHECK(rcp.body.err == RECIPE_OK, 59);

  CHECK(rcp.body.p == rcp.body.end, 60);
  return 0;
}

// Parses one inline candidate body from the choice fixture; both candidates
// share the pooled cubin at index 0.
static int check_candidate_body(const recipe* rcp, recipe_cursor* c) {
  CHECK(recipe_rd_u32(c) == 1, 87);  // module count
  recipe_kernel_module module = recipe_rd_kernel_module(c);
  CHECK(module.kind == RECIPE_CODE_IMAGE_ONLY && module.image_count == 0, 860);
  recipe_module_image image = module.preferred_image;
  CHECK(image.kind == RECIPE_IMAGE_CUDA_LOADABLE &&
            image.persistence == RECIPE_IMAGE_PERSISTABLE &&
            !image.has_provenance,
        861);
  recipe_resource_bounds module_bounds = recipe_rd_resource_bounds(c);
  CHECK(!module_bounds.min_registers.present, 861);
  CHECK(!module_bounds.max_local_bytes.present, 862);
  CHECK(image.blob == 0, 86);  // deduplicated across candidates
  static const uint8_t want_cc[2] = {0xAA, 0xBB};
  CHECK(bytes_eq(recipe_blob_at(rcp, image.blob), want_cc, 2), 88);
  CHECK(recipe_rd_u32(c) == 1, 89);  // workspace count
  recipe_workspace_buf nw = recipe_rd_workspace_buf(c);
  CHECK(nw.nbytes == 32 && !nw.has_init, 90);
  CHECK(recipe_rd_u32(c) == 1, 91);  // node count
  recipe_node nn = recipe_rd_node(c);
  CHECK(nn.kind == RECIPE_NODE_KERNEL_LAUNCH, 92);
  CHECK(nn.launch.module == 0, 93);
  CHECK(nn.launch.block[0] == 64, 94);
  CHECK(nn.launch.arg_count == 3, 95);
  CHECK(recipe_rd_deps(c).count == 0, 96);
  CHECK(c->err == RECIPE_OK, 97);
  return 0;
}

static int check_choice(const uint8_t* buf, size_t size) {
  recipe cp;
  CHECK(recipe_open(buf, size, &cp) == RECIPE_OK, 70);
  CHECK(recipe_rd_u32(&cp.requests) == 0, 69);
  CHECK(recipe_rd_u32(&cp.body) == 0, 71);    // no top-level modules
  CHECK(recipe_rd_u32(&cp.body) == 1, 72);    // one workspace buffer
  recipe_workspace_buf cw = recipe_rd_workspace_buf(&cp.body);
  CHECK(cw.nbytes == 64 && !cw.has_init, 73);
  CHECK(recipe_rd_u32(&cp.body) == 2, 74);  // node count

  recipe_node cn0 = recipe_rd_node(&cp.body);
  CHECK(cn0.kind == RECIPE_NODE_MEMSET, 75);
  CHECK(cn0.memset.destination.buf.tag == RECIPE_WRITABLE_WORKSPACE &&
            cn0.memset.destination.buf.index == 0 &&
            cn0.memset.destination.byte_offset == 16,
        76);
  CHECK(recipe_rd_u32(&cp.body) == 0, 77);  // deps count

  recipe_node cn1 = recipe_rd_node(&cp.body);
  CHECK(cn1.kind == RECIPE_NODE_CHOICE, 78);
  recipe_choice ch = recipe_rd_choice(&cp.body);

  CHECK(recipe_rd_u32(&ch.input_binding) == 1, 80);
  recipe_bufref ib = recipe_rd_bufref(&ch.input_binding);
  CHECK(ib.tag == RECIPE_BUFREF_INPUT && ib.index == 0, 81);
  CHECK(ch.input_binding.p == ch.input_binding.end, 82);

  CHECK(recipe_rd_u32(&ch.output_binding) == 1, 83);
  recipe_bufref ob = recipe_rd_bufref(&ch.output_binding);
  CHECK(ob.tag == RECIPE_BUFREF_WORKSPACE && ob.index == 0, 84);
  CHECK(ch.output_binding.p == ch.output_binding.end, 85);

  uint32_t ncand = recipe_rd_u32(&cp.body);
  CHECK(ncand == 2, 79);
  for (uint32_t k = 0; k < ncand; k++) {
    uint32_t aliases = recipe_rd_u32(&cp.body);
    CHECK(aliases == 0, 790);
    int sub = check_candidate_body(&cp, &cp.body);
    if (sub != 0) {
      return sub;
    }
  }

  CHECK(recipe_rd_u32(&cp.body) == 1, 100);  // Choice deps count
  CHECK(recipe_rd_u32(&cp.body) == 0, 101);  // deps[0] = 0
  CHECK(cp.body.err == RECIPE_OK, 102);
  CHECK(cp.body.p == cp.body.end, 103);
  return 0;
}

static int check_source(const uint8_t* buf, size_t size) {
  recipe source;
  CHECK(recipe_open(buf, size, &source) == RECIPE_OK, 120);
  CHECK(recipe_rd_u32(&source.requests) == 3, 121);

  // One request per assembly mode, read in order.
  recipe_compile_request environment_default =
      recipe_rd_compile_request(&source.requests);
  CHECK(environment_default.sources.count == 1, 1210);
  CHECK(environment_default.options.assembly_mode ==
            RECIPE_ASSEMBLY_ENVIRONMENT_DEFAULT,
        1211);
  CHECK(!environment_default.options.max_registers.present, 1212);
  CHECK(environment_default.prerequisites.count == 0, 1213);

  // Two modules in one unit, each stating its own kind, ISA and target: the
  // unit needs the highest of them, and a reader that stopped at the first
  // would say sm_100 where this compile links something that only runs from
  // sm_90 up.
  recipe_compile_request relocatable =
      recipe_rd_compile_request(&source.requests);
  CHECK(relocatable.options.assembly_mode == RECIPE_ASSEMBLY_RELOCATABLE, 1214);
  CHECK(relocatable.options.max_registers.present &&
            relocatable.options.max_registers.value == 64,
        1215);
  CHECK(relocatable.prerequisites.count == 0, 1216);
  CHECK(relocatable.sources.count == 2, 1217);
  recipe_cursor unit = relocatable.sources.entries;
  recipe_compile_source generated = recipe_rd_compile_source(&unit);
  CHECK(generated.kind == RECIPE_SOURCE_PTX && generated.ptx_isa[0] == 8 &&
            generated.ptx_isa[1] == 8 && generated.minimum_sm == 100 &&
            generated.blob == 1,
        1218);
  recipe_compile_source callable = recipe_rd_compile_source(&unit);
  CHECK(callable.kind == RECIPE_SOURCE_PTX && callable.ptx_isa[0] == 8 &&
            callable.ptx_isa[1] == 7 && callable.minimum_sm == 90 &&
            callable.blob == 2,
        1219);
  CHECK(unit.p == unit.end, 12191);
  // Each source of the unit is its own pool entry, stored the way a compile
  // source is stored.
  recipe_blob_entry callable_blob = recipe_blob_entry_at(&source, callable.blob);
  CHECK(callable_blob.encoding == RECIPE_BLOB_ZSTD &&
            callable_blob.decoded_len == 8,
        12192);

  recipe_compile_request request = recipe_rd_compile_request(&source.requests);
  CHECK(request.sources.count == 1, 122);
  recipe_cursor guarded_unit = request.sources.entries;
  recipe_compile_source guarded = recipe_rd_compile_source(&guarded_unit);
  CHECK(guarded.kind == RECIPE_SOURCE_PTX && guarded.ptx_isa[0] == 8 &&
            guarded.ptx_isa[1] == 8 && guarded.minimum_sm == 100,
        123);
  CHECK(request.options.max_registers.present &&
            request.options.max_registers.value == 128 &&
            request.options.assembly_mode == RECIPE_ASSEMBLY_WHOLE_PROGRAM,
        125);
  // Both guards, in ascending request order, each with its own bounds.
  CHECK(request.prerequisites.count == 2, 126);
  recipe_cursor prerequisites = request.prerequisites.entries;
  recipe_resource_prerequisite first =
      recipe_rd_resource_prerequisite(&prerequisites);
  CHECK(first.request == 0, 1260);
  CHECK(first.bounds.min_registers.present &&
            first.bounds.min_registers.value == 129,
        1261);
  CHECK(!first.bounds.max_local_bytes.present, 1262);
  recipe_resource_prerequisite second =
      recipe_rd_resource_prerequisite(&prerequisites);
  CHECK(second.request == 1, 1263);
  CHECK(!second.bounds.min_registers.present, 1264);
  CHECK(second.bounds.max_local_bytes.present &&
            second.bounds.max_local_bytes.value == 0,
        1265);
  CHECK(prerequisites.p == prerequisites.end, 1266);
  CHECK(source.requests.p == source.requests.end, 127);

  // Four different sources are four pool entries, numbered as the walk reaches
  // them: request by request, and within a request in link order. The guarded
  // request is the one carrying the large PTX below.
  recipe_cursor first_unit = environment_default.sources.entries;
  CHECK(recipe_rd_compile_source(&first_unit).blob == 0, 124);

  static const char ptx_line[] =
      ".version 8.8\n.target sm_100\n.address_size 64\n";
  CHECK(guarded.blob == 3, 1240);
  recipe_blob_entry ptx = recipe_blob_entry_at(&source, guarded.blob);
  CHECK(ptx.encoding == RECIPE_BLOB_ZSTD, 128);
  CHECK(ptx.decoded_len == (sizeof(ptx_line) - 1) * 256, 129);
  CHECK(ptx.digest != NULL && ptx.payload.ptr != NULL &&
            ptx.payload.len < ptx.decoded_len,
        130);
  CHECK(recipe_blob_at(&source, guarded.blob).ptr == NULL, 131);

  CHECK(recipe_rd_u32(&source.body) == 1, 132);
  recipe_kernel_module module = recipe_rd_kernel_module(&source.body);
  CHECK(module.kind == RECIPE_CODE_PTX_BACKED && module.request == 2 &&
            module.image_count == 3,
        133);
  // Order is priority, and each image states its own compatibility: the cubin
  // by the architecture it was assembled for, the loadable one by whether its
  // bytes outlive this session.
  recipe_module_image preferred = recipe_rd_module_image(&source.body);
  CHECK(preferred.kind == RECIPE_IMAGE_CUBIN && preferred.target_sm == 100,
        1330);
  static const uint8_t want_cubin_image[2] = {0xC0, 0xDE};
  CHECK(bytes_eq(recipe_blob_at(&source, preferred.blob), want_cubin_image, 2),
        1331);
  // Provenance in full, including the mode it was assembled under.
  CHECK(preferred.has_provenance, 1334);
  CHECK(preferred.provenance.unit_digest != NULL &&
            preferred.provenance.unit_digest[0] == 7 &&
            preferred.provenance.unit_digest[31] == 7,
        1335);
  CHECK(preferred.provenance.options.max_registers.present &&
            preferred.provenance.options.max_registers.value == 96,
        1336);
  CHECK(preferred.provenance.options.assembly_mode ==
            RECIPE_ASSEMBLY_RELOCATABLE,
        1337);
  CHECK(preferred.provenance.target_sm == 100, 1338);
  CHECK(bytes_eq(preferred.provenance.assembler, (const uint8_t*)"ptxas", 5),
        1339);
  CHECK(bytes_eq(preferred.provenance.assembler_version,
                 (const uint8_t*)"13.0", 4),
        1340);
  CHECK(preferred.provenance.driver_version.present &&
            preferred.provenance.driver_version.value == 13000,
        1341);
  CHECK(preferred.provenance.toolkit_version.present &&
            preferred.provenance.toolkit_version.value == 12080,
        1342);

  // Both persistence values, each stated by the image that carries it.
  recipe_module_image session = recipe_rd_module_image(&source.body);
  CHECK(session.kind == RECIPE_IMAGE_CUDA_LOADABLE &&
            session.persistence == RECIPE_IMAGE_SESSION_ONLY &&
            !session.has_provenance,
        1332);
  static const uint8_t want_session[2] = {0x5E, 0x55};
  CHECK(bytes_eq(recipe_blob_at(&source, session.blob), want_session, 2), 1333);
  recipe_module_image keeps = recipe_rd_module_image(&source.body);
  CHECK(keeps.kind == RECIPE_IMAGE_CUDA_LOADABLE &&
            keeps.persistence == RECIPE_IMAGE_PERSISTABLE,
        1343);
  static const uint8_t want_keeps[2] = {0xCA, 0xFE};
  CHECK(bytes_eq(recipe_blob_at(&source, keeps.blob), want_keeps, 2), 1344);

  recipe_resource_bounds bounds = recipe_rd_resource_bounds(&source.body);
  CHECK(bounds.min_registers.present && bounds.min_registers.value == 32, 889);
  CHECK(bounds.max_local_bytes.present && bounds.max_local_bytes.value == 64,
        134);
  CHECK(recipe_rd_u32(&source.body) == 0, 135);
  CHECK(recipe_rd_u32(&source.body) == 1, 136);
  recipe_node node = recipe_rd_node(&source.body);
  CHECK(node.kind == RECIPE_NODE_KERNEL_LAUNCH && node.launch.module == 0, 137);
  CHECK(bytes_eq(node.launch.entry, (const uint8_t*)"source", 6), 138);
  CHECK(node.launch.arg_count == 2, 139);
  recipe_cursor source_args = node.launch.args;
  for (int i = 0; i < 2; i++) {
    CHECK(recipe_rd_arg(&source_args).count == 1, 140 + i);
  }
  CHECK(source_args.p == source_args.end, 1420);
  CHECK(recipe_rd_deps(&source.body).count == 0, 142);
  CHECK(source.body.err == RECIPE_OK && source.body.p == source.body.end, 143);
  return 0;
}

// Counts come off untrusted bytes, so one larger than its section can back
// must yield no ports rather than a walk the section cannot support. These
// buffers are hand-made: no serializer can produce them.
static int check_malformed_signature(void) {
  recipe rcp;
  memset(&rcp, 0, sizeof(rcp));

  // u32::MAX inputs stated, four bytes of section behind them.
  static const uint8_t huge_inputs[4] = {0xFF, 0xFF, 0xFF, 0xFF};
  rcp.signature.p = huge_inputs;
  rcp.signature.end = huge_inputs + sizeof(huge_inputs);
  rcp.signature.err = RECIPE_OK;
  recipe_port_extents inputs = recipe_inputs(&rcp);
  CHECK(inputs.count == 0, 60);
  CHECK(inputs.ports.err == RECIPE_TRUNCATED, 61);
  CHECK(recipe_port_nbytes(&inputs, 0) == 0, 62);
  recipe_port_extents no_outputs = recipe_outputs(&rcp);
  CHECK(no_outputs.count == 0, 63);

  // One input stated and present, then an output count nothing backs. The
  // inputs still read: the bound refuses only the sequence that overruns.
  static const uint8_t short_outputs[16] = {1, 0, 0,    0,    64,   0,    0, 0,
                                            0, 0, 0,    0,    0xFF, 0xFF, 0xFF,
                                            0xFF};
  rcp.signature.p = short_outputs;
  rcp.signature.end = short_outputs + sizeof(short_outputs);
  rcp.signature.err = RECIPE_OK;
  recipe_port_extents ok_inputs = recipe_inputs(&rcp);
  CHECK(ok_inputs.count == 1, 64);
  CHECK(recipe_port_nbytes(&ok_inputs, 0) == 64, 65);
  recipe_port_extents bad_outputs = recipe_outputs(&rcp);
  CHECK(bad_outputs.count == 0, 66);

  // A section count the header cannot back is truncation, not a walk.
  static const uint8_t bad_table[12] = {'R',  'C',  'P',  'E', 13, 0,
                                        0,    0,    0xFF, 0xFF, 0xFF, 0xFF};
  recipe parsed;
  CHECK(recipe_open(bad_table, sizeof(bad_table), &parsed) == RECIPE_TRUNCATED,
        67);
  return 0;
}

// Bytes that state a set in some order other than the one this format states.
//
// A reader that took them would be reading a second spelling of a value that
// already has one, and could then disagree with the Rust reader — which
// refuses them — about what one artifact says.
static int check_non_canonical(const uint8_t* buf, size_t size) {
  // Twelve bytes of the real artifact: a count of two and its two entries.
  CHECK(canonical_deps_at != 0 && canonical_deps_at + 12 <= size, 1500);
  uint8_t deps[12];
  memcpy(deps, buf + canonical_deps_at, sizeof(deps));

  recipe_cursor c = {deps, deps + sizeof(deps), RECIPE_OK};
  recipe_deps as_written = recipe_rd_deps(&c);
  CHECK(as_written.count == 2 && c.err == RECIPE_OK, 1501);

  static const uint8_t descending[8] = {1, 0, 0, 0, 0, 0, 0, 0};
  static const uint8_t repeated[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const uint8_t* orders[2] = {descending, repeated};
  for (int i = 0; i < 2; i++) {
    uint8_t patched[12];
    memcpy(patched, deps, sizeof(patched));
    memcpy(patched + 4, orders[i], 8);
    recipe_cursor p = {patched, patched + sizeof(patched), RECIPE_OK};
    (void)recipe_rd_deps(&p);
    CHECK(p.err == RECIPE_NON_CANONICAL, 1502 + i);
  }

  // An argument is the concatenation of its segments, so an empty byte run and
  // a split one are spellings of arguments that already have one. These are
  // hand-made: no serializer emits them.
  static const uint8_t canonical_arg[] = {
      2, 0, 0, 0,                        // two segments
      0, 4, 0, 0, 0, 'a', 'b', 'c', 'd', // a four-byte run
      1, 0, 0, 0, 0, 0,                  // a pointer into input 0...
      0, 0, 0, 0, 0, 0, 0, 0             // ...at byte 0
  };
  static const uint8_t empty_run[] = {
      1, 0, 0, 0,  // one segment
      0, 0, 0, 0, 0  // an empty byte run
  };
  static const uint8_t split_run[] = {
      2, 0, 0, 0,             // two segments
      0, 2, 0, 0, 0, 'a', 'b',  // a two-byte run...
      0, 2, 0, 0, 0, 'c', 'd'   // ...and the run adjacent to it
  };
  recipe_cursor good = {canonical_arg, canonical_arg + sizeof(canonical_arg),
                        RECIPE_OK};
  recipe_arg arg = recipe_rd_arg(&good);
  CHECK(arg.count == 2 && good.err == RECIPE_OK, 1510);
  CHECK(good.p == good.end, 1511);

  recipe_cursor empty = {empty_run, empty_run + sizeof(empty_run), RECIPE_OK};
  (void)recipe_rd_arg(&empty);
  CHECK(empty.err == RECIPE_NON_CANONICAL, 1512);

  recipe_cursor split = {split_run, split_run + sizeof(split_run), RECIPE_OK};
  (void)recipe_rd_arg(&split);
  CHECK(split.err == RECIPE_NON_CANONICAL, 1513);
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    return 1;
  }

  size_t canon_size = 0;
  uint8_t* canon = read_file(argv[1], &canon_size);
  if (!canon) {
    return 2;
  }
  size_t choice_size = 0;
  uint8_t* choice = read_file(argv[2], &choice_size);
  if (!choice) {
    free(canon);
    return 3;
  }
  size_t source_size = 0;
  uint8_t* source = read_file(argv[3], &source_size);
  if (!source) {
    free(canon);
    free(choice);
    return 4;
  }

  int rc = check_canonical(canon, canon_size);
  if (rc == 0) {
    rc = check_choice(choice, choice_size);
  }
  if (rc == 0) {
    rc = check_source(source, source_size);
  }
  if (rc == 0) {
    rc = check_non_canonical(canon, canon_size);
  }
  if (rc == 0) {
    rc = check_malformed_signature();
  }

  free(canon);
  free(choice);
  free(source);
  return rc;
}
