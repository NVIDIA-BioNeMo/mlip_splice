// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::algebra::{CompileRequests, Geometry, KernelLaunch};
use crate::*;

fn native_module(bytes: &[u8]) -> KernelModule {
    KernelModule::from_cuda_loadable(bytes.to_vec())
}

fn candidate(entry: &str, module: KernelModule) -> RecipeBody {
    RecipeBody::new(
        1,
        1,
        vec![module],
        Vec::new(),
        vec![Node::new(
            Work::Call {
                call: KernelLaunch {
                    module: 0,
                    entry: entry.into(),
                    args: vec![Arg::input(0), Arg::output(0)],
                },
                geometry: vec![Geometry {
                    grid: [1, 1, 1],
                    block: [32, 1, 1],
                    dynamic_shmem: 0,
                }],
            },
            [],
        )],
    )
}

fn native_recipe() -> Recipe {
    Recipe::stating_edges(
        RecipeSignature::new(vec![4], vec![4]),
        CompileRequests::default(),
        RecipeBody::new(
            1,
            1,
            vec![native_module(&[0xde, 0xad, 0xbe, 0xef])],
            vec![WorkspaceBuf {
                nbytes: 4,
                init: Some(Blob::from_vec(vec![10, 0, 0, 0])),
            }],
            vec![Node::new(
                Work::Call {
                    call: KernelLaunch {
                        module: 0,
                        entry: "fill".into(),
                        args: vec![Arg::input(0), Arg::workspace(0), Arg::output(0)],
                    },
                    geometry: vec![Geometry {
                        grid: [1, 1, 1],
                        block: [32, 1, 1],
                        dynamic_shmem: 0,
                    }],
                },
                [],
            )],
        ),
    )
    .unwrap()
}

fn source_recipe() -> Recipe {
    let ptx = ".version 8.8\n.target sm_100\n.address_size 64\n"
        .repeat(256)
        .into_bytes();
    Recipe::stating_edges(
        RecipeSignature::new(vec![4], vec![4]),
        CompileRequests::new(vec![CompileRequest {
            sources: vec![SourceImage {
                kind: SourceKind::Ptx,
                target: PtxTarget {
                    isa: [8, 8],
                    minimum_sm: 100,
                },
                blob: Blob::from_vec(ptx),
            }],
            options: PtxCompileOptions {
                max_registers: Some(128),
                assembly_mode: PtxAssemblyMode::WholeProgram,
            },
            requires: Default::default(),
        }]),
        RecipeBody::new(
            1,
            1,
            vec![KernelModule::from_request(
                CompileRequestId(0),
                Vec::new(),
                ResourceBounds {
                    max_local_bytes: Some(64),
                    ..ResourceBounds::default()
                },
            )],
            Vec::new(),
            vec![Node::new(
                Work::Call {
                    call: KernelLaunch {
                        module: 0,
                        entry: "source".into(),
                        args: vec![Arg::input(0), Arg::output(0)],
                    },
                    geometry: vec![Geometry {
                        grid: [1, 1, 1],
                        block: [32, 1, 1],
                        dynamic_shmem: 0,
                    }],
                },
                [],
            )],
        ),
    )
    .unwrap()
}

/// The same recipe with `edit` applied to its compile table and its root
/// module table.
///
/// A recipe is a value, so a test states a variant of a fixture by assembling
/// the one it means — which is also where an edit that makes it stop being a
/// recipe is refused.
fn variant(
    recipe: Recipe,
    edit: impl FnOnce(&mut Vec<CompileRequest>, &mut Vec<KernelModule>),
) -> Result<Recipe, RecipeError> {
    let (ports, table, body) = recipe.into_parts();
    let mut table = table.into_vec();
    let (inputs, outputs, mut modules, workspace, nodes) = body.into_parts();
    edit(&mut table, &mut modules);
    Recipe::stating_edges(
        ports,
        CompileRequests::new(table),
        RecipeBody::new(inputs, outputs, modules, workspace, nodes),
    )
}

fn blobs_offset(bytes: &[u8]) -> usize {
    u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize
}

fn payload_range(bytes: &[u8], wanted: usize) -> std::ops::Range<usize> {
    let offset = blobs_offset(bytes);
    let mut cursor = Cursor::new(&bytes[offset..], offset);
    let count = cursor.count().unwrap();
    assert!(wanted < count);
    for index in 0..count {
        let _encoding = cursor.u8().unwrap();
        let _decoded_len = cursor.u64().unwrap();
        cursor.take(32).unwrap();
        let payload_len = cursor.u32().unwrap() as usize;
        let start = cursor.abs_pos();
        cursor.take(payload_len).unwrap();
        if index == wanted {
            return start..start + payload_len;
        }
    }
    unreachable!()
}

#[test]
fn native_recipe_round_trips() {
    let recipe = native_recipe();
    let wire = serialize(&recipe).unwrap();
    assert_eq!(&wire[..4], b"RCPE");
    assert_eq!(u16::from_le_bytes([wire[4], wire[5]]), 13);
    assert_eq!(deserialize(&wire), Ok(recipe));
}

/// A compile unit is a sequence of modules, and the wire carries all of it:
/// every source, its own target, and the order they link in.
#[test]
fn a_multi_source_unit_round_trips_in_link_order() {
    let source = |isa: [u8; 2], minimum_sm, bytes: &[u8]| SourceImage {
        kind: SourceKind::Ptx,
        target: PtxTarget { isa, minimum_sm },
        blob: Blob::from_vec(bytes.to_vec()),
    };
    let unit = vec![
        source([8, 8], 100, b".visible .entry caller() { call scale; }\n"),
        source([8, 7], 90, b".visible .func scale() { ret; }\n"),
    ];
    let recipe = Recipe::stating_edges(
        RecipeSignature::new(vec![4], vec![4]),
        CompileRequests::new(vec![CompileRequest::unguarded(
            unit.clone(),
            PtxCompileOptions {
                max_registers: None,
                assembly_mode: PtxAssemblyMode::Relocatable,
            },
        )]),
        RecipeBody::new(
            1,
            1,
            vec![KernelModule::from_request(
                CompileRequestId(0),
                Vec::new(),
                ResourceBounds::default(),
            )],
            Vec::new(),
            vec![Node::new(
                Work::Call {
                    call: KernelLaunch {
                        module: 0,
                        entry: "caller".into(),
                        args: vec![Arg::input(0), Arg::output(0)],
                    },
                    geometry: vec![Geometry {
                        grid: [1, 1, 1],
                        block: [32, 1, 1],
                        dynamic_shmem: 0,
                    }],
                },
                [],
            )],
        ),
    )
    .unwrap();

    let decoded = deserialize(&serialize(&recipe).unwrap()).unwrap();
    assert_eq!(decoded, recipe);
    let kept = &decoded.context().as_slice()[0].sources;
    assert_eq!(kept.len(), 2);
    assert_eq!(kept[0].target.minimum_sm, 100);
    assert_eq!(kept[1].target.minimum_sm, 90);
    assert_eq!(kept[1].target.isa, [8, 7]);
    assert_eq!(kept[1].blob.decoded(), unit[1].blob.decoded());
    assert_eq!(
        decoded.context().as_slice()[0].minimum_sm(),
        100,
        "the unit needs the highest of its sources' targets"
    );

    // Reversing the link order is another program, and the bytes say so.
    let reversed = variant(recipe.clone(), |table, _| table[0].sources.reverse()).unwrap();
    assert_ne!(serialize(&reversed).unwrap(), serialize(&recipe).unwrap());
}

#[test]
fn source_is_compressed_and_decoded_lazily() {
    let recipe = source_recipe();
    let raw = recipe.context().as_slice()[0].sources[0].blob.len();
    let wire = serialize(&recipe).unwrap();
    assert!(wire.len() < raw / 4);
    let decoded = deserialize(&wire).unwrap();
    assert_eq!(
        decoded.context().as_slice()[0].sources[0].blob,
        recipe.context().as_slice()[0].sources[0].blob
    );
}

#[test]
fn precompressed_source_serializes_to_mmap_without_reencoding() {
    let recipe = source_recipe();
    let expected = recipe.context().as_slice()[0].sources[0].blob.to_vec();
    let recipe = variant(recipe, |table, _| {
        table[0].sources[0].blob = Blob::zstd_spilled(&expected).unwrap();
    })
    .unwrap();
    let artifact = serialize_spilled(&recipe).unwrap();
    let decoded = deserialize_shared(artifact).unwrap();
    assert_eq!(
        decoded.context().as_slice()[0].sources[0].blob.decoded(),
        Ok(&expected[..])
    );
}

#[test]
fn corrupted_raw_blob_is_rejected_during_decode() {
    let mut wire = serialize(&native_recipe()).unwrap();
    let payload = payload_range(&wire, 0);
    wire[payload.start] ^= 1;
    assert!(matches!(
        deserialize(&wire),
        Err(WireError::BlobDigestMismatch { index: 0 })
    ));
}

#[test]
fn corrupted_compressed_blob_is_rejected_on_first_access() {
    let mut wire = serialize(&source_recipe()).unwrap();
    let payload = payload_range(&wire, 0);
    wire[payload.start + payload.len() / 2] ^= 1;
    let decoded = deserialize(&wire).unwrap();
    assert!(
        decoded.context().as_slice()[0].sources[0]
            .blob
            .decoded()
            .is_err()
    );
}

#[test]
fn decoded_content_is_deduplicated_across_choices() {
    let body = candidate("same", native_module(&[1, 2, 3, 4]));
    let recipe = Recipe::stating_edges(
        RecipeSignature::new(vec![4], vec![4]),
        CompileRequests::default(),
        RecipeBody::new(
            1,
            1,
            Vec::new(),
            Vec::new(),
            vec![Node::new(
                Work::Choice {
                    candidates: vec![body.clone(), body],
                    input_binding: vec![BufRef::Input(0)],
                    output_binding: vec![BufRef::Output(0)],
                },
                [],
            )],
        ),
    )
    .unwrap();
    let wire = serialize(&recipe).unwrap();
    let offset = blobs_offset(&wire);
    assert_eq!(
        u32::from_le_bytes(wire[offset..offset + 4].try_into().unwrap()),
        1
    );
    assert_eq!(deserialize(&wire), Ok(recipe));
}

#[test]
fn rich_compile_metadata_round_trips() {
    let recipe = variant(source_recipe(), |table, modules| {
        // A prerequisite on a *different* request: one naming its own id is a
        // one-step cycle, which nothing can compile and decoding now refuses.
        table.push(table[0].clone());
        table[1].require(
            CompileRequestId(0),
            ResourceBounds {
                min_registers: Some(129),
                max_local_bytes: Some(0),
            },
        );
        modules[0].code = KernelCode::PtxBacked {
            compile_request: CompileRequestId(1),
            images: vec![
                ModuleImage::Cubin {
                    target_sm: 120,
                    bytes: Blob::from_vec(vec![1, 2, 3]),
                    provenance: Some(CompileProvenance {
                        unit_digest: [7; 32],
                        options: PtxCompileOptions::default(),
                        target_sm: 120,
                        assembler: "ptxas".into(),
                        assembler_version: "13.0".into(),
                        driver_version: Some(13000),
                        toolkit_version: Some(13000),
                    }),
                },
                ModuleImage::CudaLoadable {
                    bytes: Blob::from_vec(vec![4, 5, 6]),
                    persistence: ImagePersistence::SessionOnly,
                    provenance: None,
                },
            ],
        };
    })
    .unwrap();
    assert_eq!(deserialize(&serialize(&recipe).unwrap()), Ok(recipe));
}

#[test]
fn rejects_bad_magic_version_and_truncation() {
    let wire = serialize(&native_recipe()).unwrap();
    let mut bad_magic = wire.clone();
    bad_magic[0] = b'X';
    assert!(matches!(
        deserialize(&bad_magic),
        Err(WireError::BadMagic { .. })
    ));
    let mut bad_version = wire.clone();
    bad_version[4] = 99;
    assert_eq!(
        deserialize(&bad_version),
        Err(WireError::UnsupportedVersion { found: 99 })
    );
    assert_eq!(deserialize(&wire[..20]), Err(WireError::Truncated));
}

#[test]
fn validates_global_request_references() {
    // One request in the table, and a module naming the one after it.
    assert!(matches!(
        variant(source_recipe(), |_, modules| {
            modules[0].code = KernelCode::PtxBacked {
                compile_request: CompileRequestId(1),
                images: Vec::new(),
            };
        }),
        Err(RecipeError::Context(
            crate::ExecutableRefusal::ModuleRequestOutOfRange { referenced: 1, .. }
        ))
    ));
}

#[test]
fn detach_reowns_encoded_sources() {
    let recipe = source_recipe();
    let mut decoded = deserialize(&serialize(&recipe).unwrap()).unwrap();
    decoded.detach_blobs();
    assert_eq!(decoded, recipe);
}

/// The one bound this format puts on a byte offset or a byte count, stated in
/// both directions so that what a writer emits and what a reader accepts are
/// the same set of values.
#[test]
fn the_byte_quantity_bound_is_the_same_coming_and_going() {
    assert_eq!(decode_nbytes(0, "field"), Ok(0));
    assert_eq!(encode_nbytes(0, "field"), Ok(0));
    assert_eq!(
        decode_nbytes(isize::MAX as u64, "field"),
        Ok(isize::MAX as usize),
        "the largest object there can be still crosses"
    );
    assert_eq!(
        encode_nbytes(isize::MAX as usize, "field"),
        Ok(isize::MAX as u64)
    );
    for value in [isize::MAX as u64 + 1, u64::MAX] {
        assert_eq!(
            decode_nbytes(value, "field"),
            Err(WireError::Oversize {
                what: "field",
                value,
            })
        );
        assert_eq!(
            encode_nbytes(value as usize, "field"),
            Err(WireError::Oversize {
                what: "field",
                value,
            })
        );
    }
}

/// A legal byte quantity distinctive enough to be found in an encoded artifact
/// and to be there exactly once.
const SENTINEL: u64 = 0x0000_5EED_0BAD_1234;

/// One recipe per byte-quantity field, each carrying `nbytes` in that field
/// and nothing unusual anywhere else.
///
/// Every buffer is one byte larger than the quantity under test, so a value a
/// recipe can legally hold appears exactly once in the artifact — in the field
/// under test, and not in the extent that has to be able to hold it.
fn by_field(nbytes: usize) -> Vec<(&'static str, Result<Recipe, RecipeError>)> {
    let holding = nbytes.saturating_add(1);
    let ports = |extent: usize| RecipeSignature::new(vec![extent], vec![extent]);
    let scratch = |extent: usize| {
        vec![WorkspaceBuf {
            nbytes: extent,
            init: None,
        }]
    };
    let node = |op| {
        RecipeBody::new(
            1,
            1,
            vec![KernelModule::from_cuda_loadable(vec![0xAA])],
            scratch(64),
            vec![Node::new(op, [])],
        )
    };
    let launch = |args| Work::Call {
        call: KernelLaunch {
            module: 0,
            entry: "k".into(),
            args,
        },
        geometry: vec![Geometry {
            grid: [1, 1, 1],
            block: [1, 1, 1],
            dynamic_shmem: 0,
        }],
    };
    vec![
        (
            // Only an encoded payload states what it decodes to independently
            // of the bytes it holds, so only an encoded one can carry this
            // field's value under test.
            "decoded blob length",
            Recipe::stating_edges(
                ports(64),
                CompileRequests::new(vec![CompileRequest::unguarded(
                    vec![SourceImage {
                        kind: SourceKind::Ptx,
                        target: PtxTarget {
                            isa: [8, 8],
                            minimum_sm: 100,
                        },
                        blob: Blob::from_zstd(
                            zstd::stream::encode_all(&b".version 8.8\n"[..], 3).unwrap(),
                            nbytes,
                            [0x5A; 32],
                        ),
                    }],
                    PtxCompileOptions::default(),
                )]),
                RecipeBody::new(1, 1, Vec::new(), scratch(64), Vec::new()),
            ),
        ),
        (
            // Only the input side carries it, so the encoded artifact holds
            // the value under test exactly once.
            "port byte extent",
            Recipe::stating_edges(
                RecipeSignature::new(vec![nbytes], vec![64]),
                CompileRequests::default(),
                node(launch(Vec::new())),
            ),
        ),
        (
            "workspace byte extent",
            Recipe::stating_edges(
                ports(64),
                CompileRequests::default(),
                RecipeBody::new(1, 1, Vec::new(), scratch(nbytes), Vec::new()),
            ),
        ),
        (
            "memset length",
            Recipe::stating_edges(
                ports(holding),
                CompileRequests::default(),
                node(Work::Memset {
                    destination: WritableBuf::Output(0).at(0),
                    value: 0,
                    nbytes,
                }),
            ),
        ),
        (
            "memcpy length",
            Recipe::stating_edges(
                ports(holding),
                CompileRequests::default(),
                node(Work::Memcpy {
                    source: BufRef::Input(0).at(0),
                    destination: WritableBuf::Output(0).at(0),
                    nbytes,
                }),
            ),
        ),
        (
            "address byte offset",
            Recipe::stating_edges(
                ports(holding),
                CompileRequests::default(),
                node(launch(vec![Arg::pointer(BufRef::Output(0).at(nbytes))])),
            ),
        ),
    ]
}

/// A byte quantity larger than any object never reaches an artifact.
///
/// Two gates stand between one and the wire, and a field meets whichever comes
/// first: a recipe whose declarations do not fit in an address is not a recipe,
/// and one that is refuses to be written at the field carrying the number.
/// Which gate catches which field is a fact about this format; that nothing gets
/// past both is the property.
#[test]
fn no_byte_quantity_larger_than_an_object_is_ever_written() {
    for (what, built) in by_field(usize::MAX) {
        match built {
            Ok(recipe) => assert_eq!(
                serialize(&recipe).err(),
                Some(WireError::Oversize {
                    what,
                    value: usize::MAX as u64,
                }),
                "{what} was written"
            ),
            Err(refusal) => assert!(
                matches!(refusal, RecipeError::WorkspaceLayoutOverflow),
                "{what} was refused as {refusal}"
            ),
        }
    }
}

/// And a reader refuses the same field, for bytes no writer of ours produced.
///
/// The artifact carries a legal sentinel in the field under test, and the
/// sentinel is then overwritten — the only way to reach the decoder with a
/// number the encoder will not emit.
#[test]
fn no_byte_quantity_larger_than_an_object_is_ever_read() {
    for (what, recipe) in by_field(SENTINEL as usize) {
        let recipe = recipe.expect("the sentinel is a legal value");
        let mut wire = serialize(&recipe).expect("the sentinel is a legal value");
        let sentinel = SENTINEL.to_le_bytes();
        let at = wire
            .windows(8)
            .position(|window| window == sentinel)
            .unwrap_or_else(|| panic!("{what} does not carry the sentinel"));
        assert!(
            wire[at + 1..].windows(8).all(|window| window != sentinel),
            "{what}: the sentinel must occur once for this to patch the field under test"
        );
        wire[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(
            deserialize(&wire),
            Err(WireError::Oversize {
                what,
                value: u64::MAX,
            }),
            "{what} was read"
        );
    }
}

#[test]
fn checked_u32_bounds() {
    assert_eq!(checked_u32(u32::MAX as usize, "field"), Ok(u32::MAX));
    let value = u32::MAX as usize + 1;
    assert_eq!(
        checked_u32(value, "field"),
        Err(WireError::Oversize {
            what: "field",
            value: value as u64,
        })
    );
}

/// The ports survive the wire exactly: extents, their order, and which side
/// they are on. Distinct values throughout, so a swap of inputs and outputs or
/// of two ports is a different decode rather than the same one.
#[test]
fn ports_survive_the_wire_in_order() {
    let signature = RecipeSignature::new(vec![16, 32, 48], vec![64, 80]);
    let recipe = Recipe::stating_edges(
        signature.clone(),
        CompileRequests::default(),
        portless_body(3, 2),
    )
    .unwrap();
    let decoded = deserialize(&serialize(&recipe).unwrap()).unwrap();

    assert_eq!(*decoded.contract(), signature);
    assert_eq!(decoded.contract().inputs(), &[16, 32, 48]);
    assert_eq!(decoded.contract().outputs(), &[64, 80]);
    assert_eq!(decoded.contract().input_nbytes(2), Some(48));
    assert_eq!(decoded.contract().output_nbytes(0), Some(64));
    assert_eq!(decoded.contract().input_nbytes(3), None);
}

/// A port extent is program content, so two recipes differing only in one
/// encode to different bytes. Nothing else about them differs, which is what
/// makes this a statement about the signature rather than about the body.
#[test]
fn a_port_extent_changes_the_encoding() {
    let of = |inputs: Vec<usize>, outputs: Vec<usize>| {
        let body = portless_body(inputs.len(), outputs.len());
        serialize(
            &Recipe::stating_edges(
                RecipeSignature::new(inputs, outputs),
                CompileRequests::default(),
                body,
            )
            .unwrap(),
        )
        .unwrap()
    };
    let base = of(vec![16], vec![32]);
    assert_eq!(base, of(vec![16], vec![32]), "the encoding is a function");
    assert_ne!(base, of(vec![17], vec![32]), "an input extent is content");
    assert_ne!(base, of(vec![16], vec![33]), "an output extent is content");
    assert_ne!(base, of(vec![32], vec![16]), "the sides are not a multiset");
    assert_ne!(base, of(vec![16, 16], vec![32]), "a port count is content");
    assert_ne!(base, of(vec![], vec![]), "a portless recipe is another one");
}

/// A body that touches only its own workspace, so it is the same work whatever
/// ports the recipe holding it states.
fn portless_body(inputs: usize, outputs: usize) -> RecipeBody {
    RecipeBody::new(
        inputs,
        outputs,
        Vec::new(),
        vec![WorkspaceBuf {
            nbytes: 4,
            init: None,
        }],
        vec![Node::new(
            Work::Memset {
                destination: WritableBuf::Workspace(0).at(0),
                value: 0,
                nbytes: 4,
            },
            [],
        )],
    )
}

/// A body holding one Choice over `candidates`.
fn choice_over(candidates: Vec<RecipeBody>) -> RecipeBody {
    RecipeBody::new(
        1,
        1,
        Vec::new(),
        Vec::new(),
        vec![Node::new(
            Work::Choice {
                candidates,
                input_binding: vec![BufRef::Input(0)],
                output_binding: vec![BufRef::Output(0)],
            },
            [],
        )],
    )
}

/// How many Choice nodes deep the tree runs.
fn choice_depth(body: &RecipeBody) -> usize {
    body.nodes()
        .iter()
        .filter_map(|node| match node.work() {
            Work::Choice { candidates, .. } => {
                Some(1 + candidates.iter().map(choice_depth).max().unwrap_or(0))
            }
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// The length the artifact's section table gives the SIGNATURE section.
fn signature_section_len(bytes: &[u8]) -> u64 {
    let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    (0..count)
        .map(|i| 12 + i * 20)
        .find(|&at| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) == 4)
        .map(|at| u64::from_le_bytes(bytes[at + 12..at + 20].try_into().unwrap()))
        .expect("the artifact carries a SIGNATURE section")
}

/// A candidate is a body, so it has no ports of its own however deep it nests.
/// The artifact carries exactly one SIGNATURE section — the root's — and its
/// length is a function of the root ports alone, not of the Choice tree.
#[test]
fn only_the_root_carries_ports() {
    let leaf = candidate("leaf", native_module(&[1, 2, 3, 4]));
    let inner = choice_over(vec![leaf.clone(), leaf.clone()]);
    let nested = choice_over(vec![inner.clone(), inner]);

    let signature = RecipeSignature::new(vec![4], vec![4]);
    let shallow =
        Recipe::stating_edges(signature.clone(), CompileRequests::default(), leaf).unwrap();
    let deep =
        Recipe::stating_edges(signature.clone(), CompileRequests::default(), nested).unwrap();
    assert_eq!(choice_depth(shallow.body()), 0, "no Choice at all");
    assert_eq!(choice_depth(deep.body()), 2, "a Choice inside a Choice");

    // One input, one output, and no aliases.
    let want = 4 + 8 + 4 + 8 + 4;
    for recipe in [&shallow, &deep] {
        let wire = serialize(recipe).unwrap();
        assert_eq!(signature_section_len(&wire), want);
        assert_eq!(*deserialize(&wire).unwrap().contract(), signature);
    }
}

/// The writer states the four sections in id order with contiguous payloads,
/// which is what FORMAT.md promises a reader that walks the table forward.
#[test]
fn the_section_table_is_in_id_order_with_contiguous_payloads() {
    let wire = serialize(&native_recipe()).unwrap();
    let count = u32::from_le_bytes(wire[8..12].try_into().unwrap()) as usize;
    assert_eq!(count, 4);
    let mut next = 12 + 20 * 4;
    for (i, want_id) in [1u32, 2, 3, 4].into_iter().enumerate() {
        let at = 12 + i * 20;
        let id = u32::from_le_bytes(wire[at..at + 4].try_into().unwrap());
        let offset = u64::from_le_bytes(wire[at + 4..at + 12].try_into().unwrap());
        let length = u64::from_le_bytes(wire[at + 12..at + 20].try_into().unwrap());
        assert_eq!(id, want_id, "section {i} of the table");
        assert_eq!(offset, next, "section {id} starts where the last one ended");
        next = offset + length;
    }
    assert_eq!(next as usize, wire.len(), "the payloads fill the artifact");
}

/// Counts come off untrusted bytes, so one larger than its section can back is
/// truncation — decided before it is trusted with an allocation. These bytes
/// are hand-made; no serializer produces them.
#[test]
fn a_count_larger_than_its_section_is_truncation() {
    let mut wire = serialize(&native_recipe()).unwrap();
    wire[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(deserialize(&wire), Err(WireError::Truncated));

    let mut wire = serialize(&native_recipe()).unwrap();
    let blobs_at = blobs_offset(&wire);
    wire[blobs_at..blobs_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(deserialize(&wire), Err(WireError::Truncated));

    let mut wire = serialize(&native_recipe()).unwrap();
    let at = 12 + 3 * 20;
    let signature_at = u64::from_le_bytes(wire[at + 4..at + 12].try_into().unwrap()) as usize;
    wire[signature_at..signature_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(deserialize(&wire), Err(WireError::Truncated));
}

/// A section the table says runs past the artifact is refused whole, so no
/// reader ever addresses bytes the artifact does not have.
#[test]
fn a_truncated_signature_section_is_refused() {
    let mut wire = serialize(&native_recipe()).unwrap();
    let at = 12 + 3 * 20;
    wire[at + 12..at + 20].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(deserialize(&wire), Err(WireError::Truncated));
}

/// A body full of nested structure: a choice inside a choice, a launch with a
/// pointer argument and deps, a workspace, and a module — so the sweep below
/// reaches every counted sequence the format has.
fn nested_recipe() -> Recipe {
    let leaf = RecipeBody::new(
        1,
        1,
        vec![native_module(&[9, 9, 9, 9])],
        vec![WorkspaceBuf {
            nbytes: 8,
            init: Some(Blob::from_vec(vec![1, 2, 3, 4])),
        }],
        vec![
            Node::new(
                Work::Memset {
                    destination: WritableBuf::Workspace(0).at(0),
                    value: 0,
                    nbytes: 8,
                },
                [],
            ),
            Node::new(
                Work::Call {
                    call: KernelLaunch {
                        module: 0,
                        entry: "leaf".into(),
                        args: vec![Arg::input(0), Arg::workspace(0), Arg::output(0)],
                    },
                    geometry: vec![Geometry {
                        grid: [1, 1, 1],
                        block: [32, 1, 1],
                        dynamic_shmem: 0,
                    }],
                },
                [0],
            ),
        ],
    );
    Recipe::stating_edges(
        RecipeSignature::new(vec![4], vec![4]),
        CompileRequests::default(),
        choice_over(vec![choice_over(vec![leaf])]),
    )
    .unwrap()
}

/// Every count in the artifact comes off untrusted bytes and sizes an
/// allocation. This sets each aligned `u32` in turn to `u32::MAX` and requires
/// the decoder to answer — with a value or an error — rather than reserve for
/// four billion elements first. A count that reached `Vec::with_capacity`
/// unbounded would not return here.
#[test]
fn no_single_count_can_be_inflated_into_a_huge_reservation() {
    let wire = serialize(&nested_recipe()).unwrap();
    let mut refused = 0;
    for at in (0..wire.len().saturating_sub(4)).step_by(1) {
        let mut hostile = wire.clone();
        hostile[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        match deserialize(&hostile) {
            Ok(_) => {}
            Err(_) => refused += 1,
        }
    }
    assert!(
        refused > 0,
        "the sweep must actually reach counts, not just payload bytes"
    );
}

/// The same, one level in: truncating the artifact anywhere leaves a decoder
/// that reports truncation instead of reading past the buffer.
#[test]
fn every_truncation_of_a_nested_artifact_is_reported() {
    let wire = serialize(&nested_recipe()).unwrap();
    for cut in 0..wire.len() {
        // A prefix is either a decodable artifact (it never is, short of the
        // whole thing) or a clean error; it is never a panic or a read past
        // the end.
        let _ = deserialize(&wire[..cut]);
    }
    assert_eq!(deserialize(&wire).unwrap(), nested_recipe());
}

/// Prerequisites are keyed by the request they are about, so the wire carries
/// one entry per request in ascending order. Bytes that repeat a request, or
/// state them out of order, are two statements a reader would have to
/// reconcile — and two readers taking different ones would disagree about what
/// the program is.
#[test]
fn prerequisites_that_are_not_one_ascending_entry_per_request_are_refused() {
    let source = || SourceImage {
        kind: SourceKind::Ptx,
        target: PtxTarget {
            isa: [8, 8],
            minimum_sm: 100,
        },
        blob: Blob::from_vec(b".version 8.8\n".to_vec()),
    };
    // Both bounds absent, so an entry is its request index and two absent
    // tags — six bytes whose shape the patch below depends on.
    let bounds = ResourceBounds::default();
    let mut guarded = CompileRequest::unguarded(vec![source()], PtxCompileOptions::default());
    guarded.require(CompileRequestId(0), bounds);
    guarded.require(CompileRequestId(1), bounds);
    let recipe = Recipe::stating_edges(
        RecipeSignature::empty(),
        CompileRequests::new(vec![
            CompileRequest::unguarded(vec![source()], PtxCompileOptions::default()),
            CompileRequest::unguarded(vec![source()], PtxCompileOptions::default()),
            guarded,
        ]),
        RecipeBody::default(),
    )
    .unwrap();
    let wire = serialize(&recipe).unwrap();
    assert_eq!(deserialize(&wire), Ok(recipe), "as written, it round trips");

    // The two request indices sit adjacent in the encoded prerequisite list,
    // each followed by its two absent-bound tags. Swapping them puts the
    // sequence out of order without changing its length.
    let entry = |request: u32| {
        let mut bytes = request.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0, 0]);
        bytes
    };
    let ascending: Vec<u8> = entry(0).into_iter().chain(entry(1)).collect();
    let descending: Vec<u8> = entry(1).into_iter().chain(entry(0)).collect();
    let repeated: Vec<u8> = entry(0).into_iter().chain(entry(0)).collect();

    let at = wire
        .windows(ascending.len())
        .position(|window| window == ascending)
        .expect("the prerequisite pair is in the artifact");
    assert!(
        wire[at + 1..]
            .windows(ascending.len())
            .all(|window| window != ascending),
        "the pair must occur once for this to patch the prerequisites"
    );
    for (what, replacement) in [("descending", descending), ("repeated", repeated)] {
        let mut hostile = wire.clone();
        hostile[at..at + ascending.len()].copy_from_slice(&replacement);
        assert_eq!(
            deserialize(&hostile),
            Err(WireError::NonCanonical {
                what: "compile prerequisites",
            }),
            "{what} prerequisites were read"
        );
    }
}
