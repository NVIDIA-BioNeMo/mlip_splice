// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the reader refuses, stated once as a table.
//!
//! One artifact carries every construct the format has. Each row below names a
//! way to damage it and the error that damage must produce. Every row starts
//! from the same bytes and changes them in place, so no row can pass by
//! producing something the reader would have refused for a different reason,
//! and the untouched artifact is asserted to decode.
//!
//! The rows a reader is most likely to get wrong are the canonical ones: bytes
//! that are structurally well-formed and merely spell a value a second way.
//! Accepting one is silent — it costs nothing at the time and shows up later as
//! two artifacts with one meaning and two identities.

use std::collections::BTreeSet;

use crate::algebra::{CompileRequests, Geometry, KernelLaunch};
use crate::*;

fn source() -> SourceImage {
    SourceImage {
        kind: SourceKind::Ptx,
        target: PtxTarget {
            isa: [8, 8],
            minimum_sm: 100,
        },
        blob: Blob::from_vec(b".version 8.8\n.target sm_100\n".to_vec()),
    }
}

/// One argument spelled canonically as a byte run followed by a pointer.
///
/// The two together encode to as many bytes as two byte runs do, which is what
/// lets the rows below restate them without moving anything after.
fn mixed_arg() -> Arg {
    Arg::from_segments([
        Segment::Bytes(vec![0xA1, 0xA2, 0xA3, 0xA4]),
        Segment::Pointer(BufRef::Input(0).at(0)),
    ])
}

/// A body below a Choice, so the rows reach a reader that recursed to get
/// there.
fn candidate() -> RecipeBody {
    RecipeBody::new(
        1,
        1,
        vec![KernelModule::from_cuda_loadable(Blob::from_vec(vec![
            0xC0, 0xC1,
        ]))],
        Vec::new(),
        vec![Node::new(
            Work::Call {
                call: KernelLaunch {
                    module: 0,
                    entry: "candidate".into(),
                    args: vec![
                        Arg::from_segments([
                            Segment::Bytes(vec![0xB1, 0xB2, 0xB3, 0xB4]),
                            Segment::Pointer(BufRef::Input(0).at(0)),
                        ]),
                        Arg::output(0),
                    ],
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

/// Every construct the format has, in one recipe: a guarded compile, a native
/// module, a seeded workspace, an argument of mixed segments, a node with two
/// dependencies, and a Choice whose candidate is another body.
fn corruptible_recipe() -> Recipe {
    let mut guarded = CompileRequest::unguarded(vec![source()], PtxCompileOptions::default());
    guarded.require(
        CompileRequestId(0),
        ResourceBounds {
            min_registers: Some(0x81),
            max_local_bytes: None,
        },
    );
    guarded.require(CompileRequestId(1), ResourceBounds::default());
    Recipe::stating_edges(
        RecipeSignature::new(vec![64], vec![64]),
        CompileRequests::new(vec![
            CompileRequest::unguarded(vec![source()], PtxCompileOptions::default()),
            CompileRequest::unguarded(vec![source()], PtxCompileOptions::default()),
            guarded,
        ]),
        RecipeBody::new(
            1,
            1,
            vec![KernelModule::from_cuda_loadable(Blob::from_vec(vec![
                0xD0, 0xD1, 0xD2, 0xD3,
            ]))],
            vec![WorkspaceBuf {
                nbytes: 8,
                init: Some(Blob::from_vec(vec![0xE0, 0xE1, 0xE2, 0xE3, 0, 0, 0, 0])),
            }],
            vec![
                Node::new(
                    Work::Memset {
                        destination: WritableBuf::Output(0).at(0),
                        value: 0,
                        nbytes: 64,
                    },
                    [],
                ),
                Node::new(
                    Work::Call {
                        call: KernelLaunch {
                            module: 0,
                            entry: "fill".into(),
                            args: vec![mixed_arg()],
                        },
                        geometry: vec![Geometry {
                            grid: [1, 1, 1],
                            block: [64, 1, 1],
                            dynamic_shmem: 0,
                        }],
                    },
                    [0],
                ),
                Node::new(
                    Work::Memcpy {
                        source: BufRef::Input(0).at(0x11),
                        destination: WritableBuf::Workspace(0).at(0),
                        nbytes: 8,
                    },
                    [0, 1],
                ),
                Node::new(
                    Work::Choice {
                        candidates: vec![candidate()],
                        input_binding: vec![BufRef::Input(0)],
                        output_binding: vec![BufRef::Workspace(0)],
                    },
                    [2],
                ),
            ],
        ),
    )
    .unwrap()
}

/// Where `pattern` occurs, refusing to guess when it occurs anywhere else: a
/// row that patched the wrong occurrence would test something other than what
/// it names.
fn at(wire: &[u8], pattern: &[u8]) -> usize {
    let mut found = wire
        .windows(pattern.len())
        .enumerate()
        .filter(|(_, window)| *window == pattern)
        .map(|(index, _)| index);
    let first = found.next().expect("the pattern is in the artifact");
    assert!(found.next().is_none(), "the pattern occurs more than once");
    first
}

/// Replaces `pattern` with bytes of its own length, so everything after it
/// stays where the section table says it is.
fn restate(wire: &mut [u8], pattern: &[u8], replacement: &[u8]) {
    assert_eq!(
        pattern.len(),
        replacement.len(),
        "a restatement that moved the bytes after it would test truncation"
    );
    let at = at(wire, pattern);
    wire[at..at + replacement.len()].copy_from_slice(replacement);
}

/// The last node's dependency list, behind enough of the node it belongs to to
/// name it uniquely: the copy's length, then the count and one `u32` per entry.
fn deps(entries: &[u32]) -> Vec<u8> {
    let mut bytes = 8u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for entry in entries {
        bytes.extend_from_slice(&entry.to_le_bytes());
    }
    bytes
}

/// The guarded compile's prerequisite list, whose first entry carries a bound
/// and so names the pair uniquely. `requests` restates only which request each
/// entry is about; the bounds stay where they are.
fn prerequisites(requests: [u32; 2]) -> Vec<u8> {
    let mut bytes = requests[0].to_le_bytes().to_vec();
    bytes.extend_from_slice(&[1]);
    bytes.extend_from_slice(&0x81u32.to_le_bytes());
    bytes.extend_from_slice(&[0]);
    bytes.extend_from_slice(&requests[1].to_le_bytes());
    bytes.extend_from_slice(&[0, 0]);
    bytes
}

/// A `Bytes` segment: the tag, the length, and the payload.
fn bytes_segment(payload: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0u8];
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

/// The pointer half of [`mixed_arg`]: a tag, a bufref, and a byte offset.
fn pointer_segment() -> Vec<u8> {
    let mut bytes = vec![1u8, 0u8];
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes
}

/// An argument of a byte run then a pointer, as the wire holds it, without its
/// leading segment count. `run` names which one: the root's launch carries
/// `0xA1..`, the candidate's carries `0xB1..`.
fn mixed_arg_segments(run: [u8; 4]) -> Vec<u8> {
    let mut bytes = bytes_segment(&run);
    bytes.extend_from_slice(&pointer_segment());
    bytes
}

/// Where the first pool entry states what its payload decodes to: past the
/// section table to BLOBS, past the pool count, past the entry's encoding byte.
fn first_blob_decoded_len(wire: &[u8]) -> usize {
    let word = |at: usize| u32::from_le_bytes(wire[at..at + 4].try_into().unwrap()) as usize;
    let blobs = (0..word(8))
        .map(|entry| 12 + entry * 20)
        .find(|&at| word(at) == 1)
        .map(|at| u64::from_le_bytes(wire[at + 4..at + 12].try_into().unwrap()) as usize)
        .expect("the BLOBS section");
    blobs + 4 + 1
}

/// One way of damaging the artifact: what it is called, what it does to the
/// bytes, and the error the reader must answer with.
struct Corruption {
    what: &'static str,
    damage: fn(&mut Vec<u8>),
    expected: WireError,
}

fn corruptions() -> Vec<Corruption> {
    vec![
        Corruption {
            what: "a payload stating a length no allocation could reach",
            damage: |wire| {
                let at = first_blob_decoded_len(wire);
                wire[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
            },
            expected: WireError::Oversize {
                what: "decoded blob length",
                value: u64::MAX,
            },
        },
        Corruption {
            what: "another format's bytes",
            damage: |wire| wire[0] = b'X',
            expected: WireError::BadMagic { found: *b"XCPE" },
        },
        Corruption {
            what: "a protocol this build does not read",
            damage: |wire| wire[4..6].copy_from_slice(&(wire::VERSION + 1).to_le_bytes()),
            expected: WireError::UnsupportedVersion {
                found: wire::VERSION + 1,
            },
        },
        Corruption {
            what: "a header without its section table",
            damage: |wire| wire.truncate(8),
            expected: WireError::Truncated,
        },
        Corruption {
            what: "a segment kind that does not exist",
            damage: |wire| {
                let at = at(wire, &mixed_arg_segments([0xA1, 0xA2, 0xA3, 0xA4]));
                wire[at] = 7;
            },
            expected: WireError::BadTag {
                what: "segment",
                tag: 7,
            },
        },
        Corruption {
            what: "dependencies out of order",
            damage: |wire| restate(wire, &deps(&[0, 1]), &deps(&[1, 0])),
            expected: WireError::NonCanonical {
                what: "node dependencies",
            },
        },
        Corruption {
            what: "a dependency stated twice",
            damage: |wire| restate(wire, &deps(&[0, 1]), &deps(&[0, 0])),
            expected: WireError::NonCanonical {
                what: "node dependencies",
            },
        },
        Corruption {
            what: "prerequisites out of order",
            damage: |wire| restate(wire, &prerequisites([0, 1]), &prerequisites([1, 0])),
            expected: WireError::NonCanonical {
                what: "compile prerequisites",
            },
        },
        Corruption {
            what: "a prerequisite stated twice",
            damage: |wire| restate(wire, &prerequisites([0, 1]), &prerequisites([0, 0])),
            expected: WireError::NonCanonical {
                what: "compile prerequisites",
            },
        },
        Corruption {
            what: "one byte run split into two",
            damage: |wire| {
                // The pointer's fourteen bytes hold a nine-byte run exactly,
                // so the argument keeps its length and gains an adjacency.
                let split = [
                    bytes_segment(&[0xA1, 0xA2, 0xA3, 0xA4]),
                    bytes_segment(&[0; 9]),
                ]
                .concat();
                restate(wire, &mixed_arg_segments([0xA1, 0xA2, 0xA3, 0xA4]), &split);
            },
            expected: WireError::NonCanonical {
                what: "argument segments",
            },
        },
        Corruption {
            what: "a byte run carrying nothing",
            damage: |wire| {
                let empty = [bytes_segment(&[]), bytes_segment(&[0; 13])].concat();
                restate(wire, &mixed_arg_segments([0xA1, 0xA2, 0xA3, 0xA4]), &empty);
            },
            expected: WireError::NonCanonical {
                what: "argument segments",
            },
        },
        Corruption {
            what: "one byte run split into two, inside a Choice candidate",
            damage: |wire| {
                let split = [
                    bytes_segment(&[0xB1, 0xB2, 0xB3, 0xB4]),
                    bytes_segment(&[0; 9]),
                ]
                .concat();
                restate(wire, &mixed_arg_segments([0xB1, 0xB2, 0xB3, 0xB4]), &split);
            },
            expected: WireError::NonCanonical {
                what: "argument segments",
            },
        },
        Corruption {
            what: "a payload that is not what the pool says it is",
            damage: |wire| {
                let at = at(wire, &[0xD0, 0xD1, 0xD2, 0xD3]);
                wire[at] = 0xDF;
            },
            expected: WireError::BlobDigestMismatch { index: 1 },
        },
    ]
}

#[test]
fn the_artifact_every_row_starts_from_decodes() {
    let recipe = corruptible_recipe();
    let wire = serialize(&recipe).unwrap();
    assert_eq!(deserialize(&wire), Ok(recipe));
}

#[test]
fn each_way_of_damaging_an_artifact_is_named_by_the_error_it_produces() {
    let wire = serialize(&corruptible_recipe()).unwrap();
    for row in corruptions() {
        let mut damaged = wire.clone();
        (row.damage)(&mut damaged);
        assert_ne!(damaged, wire, "{} left the artifact alone", row.what);
        assert_eq!(
            deserialize(&damaged),
            Err(row.expected),
            "{} was read",
            row.what
        );
    }
}

/// The rows that restate a value rather than break it must leave an artifact
/// of the same length, or they would be testing truncation under another name.
#[test]
fn a_restated_value_leaves_the_artifact_the_size_it_was() {
    let wire = serialize(&corruptible_recipe()).unwrap();
    for row in corruptions() {
        if !matches!(row.expected, WireError::NonCanonical { .. }) {
            continue;
        }
        let mut damaged = wire.clone();
        (row.damage)(&mut damaged);
        assert_eq!(
            damaged.len(),
            wire.len(),
            "{} moved the bytes after it",
            row.what
        );
    }
}

/// Every canonical row states the same value the artifact already states, so
/// the reader's answer is a decision about spelling rather than about meaning.
#[test]
fn what_the_canonical_rows_restate_denotes_what_it_replaced() {
    let recipe = corruptible_recipe();
    assert_eq!(
        BTreeSet::from([1, 0]),
        *recipe.body().nodes()[2].deps(),
        "a set does not remember which order it was given"
    );
    assert_eq!(
        Arg::from_segments([
            Segment::Bytes(vec![0xA1, 0xA2]),
            Segment::Bytes(vec![0xA3, 0xA4]),
            Segment::Pointer(BufRef::Input(0).at(0)),
        ]),
        mixed_arg(),
        "a split run is the run"
    );
    assert_eq!(
        Arg::from_segments([
            Segment::Bytes(Vec::new()),
            Segment::Bytes(vec![0xA1, 0xA2, 0xA3, 0xA4]),
            Segment::Pointer(BufRef::Input(0).at(0)),
        ]),
        mixed_arg(),
        "an empty run carries nothing"
    );
}
