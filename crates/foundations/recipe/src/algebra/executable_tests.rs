// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::Blob;
use crate::code::{
    ModuleImage, PtxAssemblyMode, PtxCompileOptions, PtxTarget, ResourceBounds, SourceImage,
    SourceKind,
};

fn source(ptx: &str) -> SourceImage {
    SourceImage {
        kind: SourceKind::Ptx,
        target: PtxTarget {
            isa: [8, 7],
            minimum_sm: 90,
        },
        blob: Blob::from_vec(ptx.as_bytes().to_vec()),
    }
}

fn request(ptx: &str, max_registers: Option<u32>) -> CompileRequest {
    CompileRequest::unguarded(
        vec![source(ptx)],
        PtxCompileOptions {
            max_registers,
            assembly_mode: PtxAssemblyMode::Relocatable,
        },
    )
}

fn requested_module(request: usize) -> KernelModule {
    KernelModule::from_request(
        CompileRequestId(request),
        Vec::new(),
        ResourceBounds::default(),
    )
}

/// A key only buckets the entries worth comparing; the bytes decide. The same
/// unit is one entry, a different link order is another.
#[test]
fn interning_collapses_the_same_compile_unit() {
    let mut table = CompileRequests::default();
    let unit = |sources: Vec<SourceImage>| {
        CompileRequest::unguarded(sources, PtxCompileOptions::default())
    };
    let forward = table.intern(unit(vec![source("caller"), source("callee")]));
    let again = table.intern(unit(vec![source("caller"), source("callee")]));
    let reversed = table.intern(unit(vec![source("callee"), source("caller")]));
    let alone = table.intern(unit(vec![source("caller")]));
    assert_eq!(forward, again);
    assert_ne!(forward, reversed);
    assert_ne!(forward, alone);
    assert_eq!(table.len(), 3);
}

/// The key index is derived from the entries, so it is not part of the value:
/// a table assembled by interning equals one stated whole.
#[test]
fn a_table_is_its_entries_and_not_how_it_was_assembled() {
    let mut interned = CompileRequests::default();
    interned.intern(request("a", None));
    interned.intern(request("b", None));
    interned.intern(request("a", None));
    assert_eq!(
        interned,
        CompileRequests::new(vec![request("a", None), request("b", None)])
    );
}

/// Merging interns in dependency order, so a prerequisite is already on its
/// merged index when the request naming it arrives — and what the host already
/// held stays exactly where it was.
#[test]
fn a_prerequisite_follows_the_request_it_names() {
    let mut host = CompileRequests::default();
    host.intern(request("kernel", None));

    let mut incoming = CompileRequests::default();
    let base = incoming.intern(request("kernel", None));
    let mut guarded = request("kernel", Some(128));
    guarded.require(
        base,
        ResourceBounds {
            max_local_bytes: Some(0),
            ..ResourceBounds::default()
        },
    );
    let capped = incoming.intern(guarded);

    let remap = Executable::merge_root_context(&mut host, incoming);
    assert_eq!(
        remap.of(base.0),
        0,
        "the twin lands on the entry already there"
    );
    assert_eq!(remap.of(capped.0), 1);
    assert_eq!(host.len(), 2);
    assert_eq!(
        host.as_slice()[1]
            .requires
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        vec![CompileRequestId(0)]
    );
    Executable::validate_root_context(&host).unwrap();
}

/// A merge moves only what arrives. Everything the host already held is still
/// at the index it was at, and still says what it said — which is what lets a
/// builder take in one candidate after another without the earlier ones'
/// handles going stale.
#[test]
fn a_merge_leaves_what_was_already_there_where_it_was() {
    let mut host = CompileRequests::default();
    let kept: Vec<CompileRequestId> = ["a", "b", "c"]
        .iter()
        .map(|ptx| host.intern(request(ptx, None)))
        .collect();
    let before = host.as_slice().to_vec();

    for round in ["b", "d", "e"] {
        let mut incoming = CompileRequests::default();
        incoming.intern(request(round, None));
        incoming.intern(request("a", None));
        let remap = Executable::merge_root_context(&mut host, incoming);

        for (index, request) in before.iter().enumerate() {
            assert_eq!(&host.as_slice()[index], request, "entry {index} moved");
        }
        assert_eq!(
            remap.of(1),
            kept[0].0,
            "an incoming twin lands on the entry already there"
        );
    }
    assert_eq!(host.len(), 5);

    let mut modules = vec![requested_module(0), requested_module(1)];
    let held = modules.clone();
    for round in 0..3 {
        let base = Executable::merge_body_context(&mut modules, vec![requested_module(round)]);
        assert_eq!(base, ModuleBase(2 + round));
        assert_eq!(&modules[..2], &held[..], "a module already there moved");
    }
}

/// A table that cannot be walked in dependency order moves across whole and
/// unmerged, so validation reports what is wrong with it.
#[test]
fn a_cyclic_table_moves_across_whole() {
    let mut cyclic = request("kernel", None);
    cyclic.require(CompileRequestId(0), ResourceBounds::default());
    let mut host = CompileRequests::default();
    host.intern(request("other", None));

    let remap = Executable::merge_root_context(&mut host, CompileRequests::new(vec![cyclic]));
    assert_eq!(remap.of(0), 1);
    assert_eq!(
        Executable::validate_root_context(&host),
        Err(ExecutableRefusal::CompileRequestCycle { request: 1 })
    );
}

/// The two rebases meet where the handles live: a module's request moves with
/// the root merge, a launch's module with the body merge.
#[test]
fn rebasing_moves_each_handle_at_its_own_level() {
    let mut modules = vec![requested_module(0), requested_module(1)];
    Executable::rebase_body_context(&mut modules, &RequestRemap(vec![5, 3]));
    assert_eq!(
        modules
            .iter()
            .map(|module| module.code.compile_request())
            .collect::<Vec<_>>(),
        vec![Some(CompileRequestId(5)), Some(CompileRequestId(3))]
    );

    let mut host = vec![requested_module(0)];
    let base = Executable::merge_body_context(&mut host, modules);
    assert_eq!(base, ModuleBase(1));
    let mut call = KernelLaunch {
        module: 1,
        entry: "loom_a".to_string(),
        args: Vec::new(),
    };
    Executable::rebase_call(&mut call, &base);
    assert_eq!(call.module, 2);
}

/// A handle the remap does not cover is sent out of range rather than left to
/// name whichever entry now sits at that offset.
#[test]
fn an_uncovered_handle_is_sent_out_of_range() {
    let mut modules = vec![requested_module(4)];
    Executable::rebase_body_context(&mut modules, &RequestRemap(vec![0]));
    assert_eq!(
        modules[0].code.compile_request(),
        Some(CompileRequestId(usize::MAX))
    );
}

/// An image-only module names no compile, so a root merge moves nothing in it.
#[test]
fn an_image_only_module_names_no_compile() {
    let mut modules = vec![KernelModule::from_cuda_loadable(vec![7u8])];
    Executable::rebase_body_context(&mut modules, &RequestRemap(vec![9]));
    assert_eq!(modules[0].code.compile_request(), None);
}

#[test]
fn the_family_refuses_what_it_owns() {
    let empty = CompileRequest::unguarded(Vec::new(), PtxCompileOptions::default());
    assert_eq!(
        Executable::validate_root_context(&CompileRequests::new(vec![empty])),
        Err(ExecutableRefusal::EmptyCompileUnit { request: 0 })
    );

    let mut dangling = request("kernel", None);
    dangling.require(CompileRequestId(7), ResourceBounds::default());
    assert_eq!(
        Executable::validate_root_context(&CompileRequests::new(vec![dangling])),
        Err(ExecutableRefusal::RequestOutOfRange {
            request: 0,
            referenced: 7
        })
    );

    assert_eq!(
        Executable::validate_body_context(
            &vec![requested_module(3)],
            &CompileRequests::new(vec![request("kernel", None)])
        ),
        Err(ExecutableRefusal::ModuleRequestOutOfRange {
            module: 0,
            referenced: 3
        })
    );

    let call = KernelLaunch {
        module: 2,
        entry: "loom_a".to_string(),
        args: Vec::new(),
    };
    assert_eq!(
        Executable::validate_call(&call, &vec![requested_module(0)]),
        Err(ExecutableRefusal::ModuleOutOfRange { module: 2 })
    );
    assert_eq!(
        Executable::validate_call(&call, &vec![requested_module(0); 3]),
        Ok(())
    );
}

/// A module carrying its own image compiles nothing, so a table it sits beside
/// may be empty.
#[test]
fn an_image_only_module_needs_no_table() {
    let module = KernelModule {
        code: crate::code::KernelCode::ImageOnly {
            preferred_image: ModuleImage::CudaLoadable {
                bytes: Blob::from_vec(vec![1, 2, 3]),
                persistence: crate::code::ImagePersistence::Persistable,
                provenance: None,
            },
            alternative_images: Vec::new(),
        },
        resource_bounds: ResourceBounds::default(),
    };
    assert_eq!(
        Executable::validate_body_context(&vec![module], &CompileRequests::default()),
        Ok(())
    );
}

/// The contract states its own identity, and the walk is the one a digest is
/// framed by: input count, input widths, output count, output widths.
#[test]
fn the_contract_states_its_own_identity() {
    #[derive(Default, PartialEq, Eq, Debug)]
    struct Recorded(Vec<String>);

    impl IdentityWalk for Recorded {
        fn byte(&mut self, value: u8) {
            self.0.push(format!("byte {value}"));
        }
        fn count(&mut self, value: usize) {
            self.0.push(format!("count {value}"));
        }
        fn number(&mut self, value: u64) {
            self.0.push(format!("number {value}"));
        }
        fn bytes(&mut self, value: &[u8]) {
            self.0.push(format!("bytes {value:?}"));
        }
    }

    let mut walk = Recorded::default();
    RecipeSignature::new(vec![64, 8], vec![32]).identify(&mut walk);
    assert_eq!(
        walk.0,
        vec![
            "count 2",
            "number 64",
            "number 8",
            "count 1",
            "number 32",
            "count 0",
        ]
    );

    // Extents that agree port for port are the same contract; a port moved
    // between the two runs is not.
    let identity = |signature: &RecipeSignature| {
        let mut walk = Recorded::default();
        signature.identify(&mut walk);
        walk.0
    };
    assert_ne!(
        identity(&RecipeSignature::new(vec![64, 32], Vec::new())),
        identity(&RecipeSignature::new(vec![64], vec![32]))
    );
}
