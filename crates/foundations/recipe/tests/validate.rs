// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the executable specialization proves about itself: that its ports
//! decide what is in range, and that a compile table stays meaningful wherever
//! the work naming it lands.

use recipe::algebra::{CompileRequests, ExecutableRefusal, Geometry, KernelLaunch};
use recipe::{
    Access, Arg, Blob, Buf, BufKind, BufRef, CompileRequest, CompileRequestId, KernelModule, Node,
    PtxCompileOptions, PtxTarget, Recipe, RecipeBody, RecipeBuilder, RecipeError, RecipeFragment,
    RecipeSignature, ResourceBounds, SourceImage, SourceKind, Work, WorkspaceBuf, Writable,
    WritableBuf, deserialize, serialize,
};

fn scratch(nbytes: usize) -> WorkspaceBuf {
    WorkspaceBuf { nbytes, init: None }
}

fn launch(module: usize, entry: &str, block: [u32; 3], args: Vec<Arg>) -> Work {
    Work::Call {
        call: KernelLaunch {
            module,
            entry: entry.into(),
            args,
        },
        geometry: vec![Geometry::new([1, 1, 1], block, 0)],
    }
}

fn stating(signature: RecipeSignature, body: RecipeBody) -> Result<Recipe, RecipeError> {
    Recipe::stating_edges(signature, CompileRequests::default(), body)
}

#[test]
fn memset_only_recipe_validates() {
    let recipe = stating(
        RecipeSignature::empty(),
        RecipeBody::new(
            0,
            0,
            Vec::new(),
            vec![scratch(256)],
            vec![Node::new(
                Work::Memset {
                    destination: WritableBuf::Workspace(0).at(0),
                    value: 0,
                    nbytes: 256,
                },
                [],
            )],
        ),
    )
    .unwrap();
    assert_eq!(recipe.validate(), Ok(()));
}

#[test]
fn dep_out_of_range_is_rejected() {
    assert_eq!(
        stating(
            RecipeSignature::empty(),
            RecipeBody::new(
                0,
                0,
                Vec::new(),
                vec![scratch(16)],
                vec![Node::new(
                    Work::Memset {
                        destination: WritableBuf::Workspace(0).at(0),
                        value: 0,
                        nbytes: 16,
                    },
                    [1],
                )],
            ),
        ),
        Err(RecipeError::DepNotEarlier { node: 0, dep: 1 })
    );
}

#[test]
fn workspace_index_out_of_range_is_rejected() {
    assert_eq!(
        stating(
            RecipeSignature::empty(),
            RecipeBody::new(
                0,
                0,
                Vec::new(),
                Vec::new(),
                vec![Node::new(
                    Work::Memset {
                        destination: WritableBuf::Workspace(0).at(0),
                        value: 0,
                        nbytes: 16,
                    },
                    [],
                )],
            ),
        ),
        Err(RecipeError::BufOutOfRange {
            node: 0,
            kind: BufKind::Workspace,
            index: 0
        })
    );
}

#[test]
fn whole_recipe_choice_wraps_whole_recipe_candidates() {
    let candidate = |entry: &str| {
        let mut b = <RecipeBuilder>::new(1, 1);
        let module = b.module(vec![0xAA]);
        let ws = b.scratch(128);
        b.kernel(module, entry)
            .block([64, 1, 1])
            .read(Buf::Input(0))
            .read_write(ws)
            .write(Writable::Output(0))
            .add();
        b.finish(RecipeSignature::new(vec![256], vec![256]))
            .unwrap()
            .into_fragment()
    };

    let mut b = <RecipeBuilder>::new(1, 1);
    b.whole_recipe_choice(vec![candidate("a"), candidate("b")], 1, 1)
        .unwrap();
    let recipe = b
        .finish(RecipeSignature::new(vec![256], vec![256]))
        .unwrap()
        .into_recipe();

    assert!(recipe.body().context().is_empty());
    assert!(recipe.body().workspace().is_empty());
    let [node] = recipe.body().nodes() else {
        panic!("expected a single choice node");
    };
    let Work::Choice {
        candidates,
        input_binding,
        output_binding,
    } = node.work()
    else {
        panic!("expected a choice, got {:?}", node.work());
    };
    assert!(node.deps().is_empty());
    assert_eq!(candidates.len(), 2);
    assert_eq!(input_binding, &vec![BufRef::Input(0)]);
    assert_eq!(output_binding, &vec![BufRef::Output(0)]);
    recipe.validate().unwrap();
}

#[test]
fn recipe_builder_assembles_and_indexes() {
    let mut b = <RecipeBuilder>::new(1, 1);
    let m = b.module(vec![0xDE, 0xAD]);
    let w = b.scratch(64);
    let init = b.memset(Writable::Workspace(w).at(0), 0, 64);
    let launched = b
        .kernel(m, "k")
        .grid([1, 1, 1])
        .block([32, 1, 1])
        .read(Buf::Input(0))
        .read(w)
        .scalar_i32(7)
        .write(Writable::Output(0))
        .add();
    assert_eq!(
        (m.index(), w.index(), init.index(), launched.index()),
        (0, 0, 0, 1)
    );

    let recipe = b
        .finish(RecipeSignature::new(vec![64], vec![64]))
        .unwrap()
        .into_recipe();
    assert_eq!(recipe.body().context().len(), 1);
    assert_eq!(recipe.body().workspace().len(), 1);
    assert_eq!(recipe.body().nodes().len(), 2);
    assert_eq!(
        recipe.body().nodes()[1].deps().iter().copied().next(),
        Some(init.index()),
        "the launch reads what the memset wrote"
    );
    let Work::Call {
        call: KernelLaunch { args, .. },
        ..
    } = recipe.body().nodes()[1].work()
    else {
        unreachable!()
    };
    assert_eq!(args[2], Arg::scalar(7i32.to_le_bytes().to_vec()));
    recipe.validate().unwrap();
}

#[test]
fn compile_request_cycle_is_rejected() {
    let request = |prerequisite| {
        let mut request = CompileRequest::unguarded(
            vec![SourceImage {
                kind: SourceKind::Ptx,
                target: PtxTarget {
                    isa: [8, 8],
                    minimum_sm: 100,
                },
                blob: Blob::from_vec(Vec::new()),
            }],
            PtxCompileOptions::default(),
        );
        request
            .requires
            .insert(CompileRequestId(prerequisite), ResourceBounds::default());
        request
    };
    assert!(matches!(
        <Recipe>::stating_edges(
            RecipeSignature::empty(),
            CompileRequests::new(vec![request(1), request(0)]),
            RecipeBody::default(),
        ),
        Err(RecipeError::Context(
            ExecutableRefusal::CompileRequestCycle { .. }
        ))
    ));
}

/// The recipe's own ports decide what is in range. A reference the signature
/// does not cover is out of range however the caller was going to bind it,
/// which is what makes validation answerable without the caller present.
#[test]
fn the_signature_decides_what_is_in_range() {
    let bound = |signature: RecipeSignature| {
        let inputs = signature.num_inputs();
        let outputs = signature.num_outputs();
        stating(
            signature,
            RecipeBody::new(
                inputs,
                outputs,
                vec![KernelModule::from_cuda_loadable(vec![0xAA])],
                Vec::new(),
                vec![Node::new(
                    launch(0, "k", [1, 1, 1], vec![Arg::input(1), Arg::output(0)]),
                    [],
                )],
            ),
        )
        .map(|_| ())
    };
    assert_eq!(bound(RecipeSignature::new(vec![8, 8], vec![8])), Ok(()));
    assert_eq!(
        bound(RecipeSignature::new(vec![8], vec![8])),
        Err(RecipeError::BufOutOfRange {
            node: 0,
            kind: BufKind::Input,
            index: 1
        }),
        "one input does not cover a reference to the second"
    );
    assert_eq!(
        bound(RecipeSignature::new(vec![8, 8], vec![])),
        Err(RecipeError::BufOutOfRange {
            node: 0,
            kind: BufKind::Output,
            index: 0
        })
    );
    assert_eq!(
        bound(RecipeSignature::empty()),
        Err(RecipeError::BufOutOfRange {
            node: 0,
            kind: BufKind::Input,
            index: 1
        }),
        "a portless recipe covers no external reference at all"
    );
}

/// A port carries two facts and both are load-bearing: how many buffers there
/// are decides what a node may name, and how big each one is decides what a
/// node may touch. This pins the second — the same body against the same port
/// count validates or not by extent alone.
#[test]
fn a_port_extent_decides_what_fits_in_it() {
    let bound = |signature: RecipeSignature| {
        let outputs = signature.num_outputs();
        stating(
            signature,
            RecipeBody::new(
                0,
                outputs,
                Vec::new(),
                Vec::new(),
                vec![Node::new(
                    Work::Memset {
                        destination: WritableBuf::Output(0).at(0),
                        value: 0,
                        nbytes: 1_000_000,
                    },
                    [],
                )],
            ),
        )
        .map(|_| ())
    };
    assert_eq!(
        bound(RecipeSignature::new(vec![], vec![1])),
        Err(RecipeError::AccessPastEnd {
            node: 0,
            kind: BufKind::Output,
            index: 0,
            wanted: 1_000_000,
            capacity: 1,
        }),
        "one byte cannot back a megabyte"
    );
    assert_eq!(
        bound(RecipeSignature::new(vec![], vec![1 << 20])),
        Ok(()),
        "and a megabyte can"
    );
}

/// The root body's arity is its contract's. A body that states a different
/// number of formal slots than its ports makes every index below it mean two
/// things, so it is not a recipe.
#[test]
fn the_root_body_states_the_arity_its_ports_state() {
    assert_eq!(
        stating(
            RecipeSignature::new(vec![8], vec![8]),
            RecipeBody::new(2, 1, Vec::new(), Vec::new(), Vec::new()),
        ),
        Err(RecipeError::RootArity {
            kind: BufKind::Input,
            stated: 2,
            contract: 1
        })
    );
}

fn ptx_request(source: &str) -> CompileRequest {
    CompileRequest::unguarded(
        vec![SourceImage {
            kind: SourceKind::Ptx,
            target: PtxTarget {
                isa: [8, 0],
                minimum_sm: 80,
            },
            blob: Blob::from_vec(source.as_bytes().to_vec()),
        }],
        PtxCompileOptions::default(),
    )
}

/// A fragment that compiles one PTX source and launches it, so its single
/// module names request 0 of its own table.
fn compiled_fragment(source: &str, entry: &str) -> RecipeFragment {
    let mut b = <RecipeBuilder>::new(0, 1);
    let request = b.compile_request(ptx_request(source));
    let module = b.requested_module(request, Vec::new(), ResourceBounds::default());
    b.kernel(module, entry).write(Writable::Output(0)).add();
    b.finish_fragment()
}

/// Every module in `body`, and in every candidate below it, paired with the
/// request it names.
fn requests_named(body: &RecipeBody, out: &mut Vec<Option<usize>>) {
    for module in body.context() {
        out.push(module.code.compile_request().map(|r| r.0));
    }
    for node in body.nodes() {
        if let Work::Choice { candidates, .. } = node.work() {
            for candidate in candidates {
                requests_named(candidate, out);
            }
        }
    }
}

/// Distinct sources, so deduplication cannot hide a misbound index behind a
/// shared request.
#[test]
fn a_choice_over_compiled_fragments_remaps_every_request() {
    let a = compiled_fragment(".version 8.0\n// a\n", "ka");
    let b = compiled_fragment(".version 8.0\n// b\n", "kb");
    assert_eq!(a.context().len(), 1, "each names its own table");
    assert_eq!(b.context().len(), 1);

    let mut root = <RecipeBuilder>::new(0, 1);
    root.choice(vec![a, b])
        .outputs([Buf::Output(0)])
        .does([Buf::Output(0)], Access::Write)
        .add()
        .unwrap();
    let recipe = root
        .finish(RecipeSignature::new(vec![], vec![4]))
        .unwrap()
        .into_recipe();

    assert_eq!(
        recipe.context().len(),
        2,
        "two distinct sources, two requests in the root table"
    );
    let mut named = Vec::new();
    requests_named(recipe.body(), &mut named);
    let mut indices: Vec<usize> = named.into_iter().flatten().collect();
    indices.sort();
    assert_eq!(
        indices,
        vec![0, 1],
        "and the candidates name both, remapped"
    );
    assert_eq!(recipe.validate(), Ok(()), "no index dangles");
}

/// The same source twice is one request, and both candidates name it.
#[test]
fn identical_sources_share_one_request_after_import() {
    let a = compiled_fragment(".version 8.0\n// same\n", "k");
    let b = compiled_fragment(".version 8.0\n// same\n", "k");
    let mut root = <RecipeBuilder>::new(0, 1);
    root.choice(vec![a, b])
        .outputs([Buf::Output(0)])
        .does([Buf::Output(0)], Access::Write)
        .add()
        .unwrap();
    let recipe = root
        .finish(RecipeSignature::new(vec![], vec![4]))
        .unwrap()
        .into_recipe();

    assert_eq!(recipe.context().len(), 1);
    let mut named = Vec::new();
    requests_named(recipe.body(), &mut named);
    assert_eq!(named, vec![Some(0), Some(0)]);
    assert_eq!(recipe.validate(), Ok(()));
}

/// A fragment nested two choices deep still names the root's table.
#[test]
fn a_fragment_two_choices_deep_names_the_root_table() {
    let leaf = compiled_fragment(".version 8.0\n// leaf\n", "leaf");
    let mut middle = <RecipeBuilder>::new(0, 1);
    middle
        .choice(vec![leaf])
        .outputs([Buf::Output(0)])
        .does([Buf::Output(0)], Access::Write)
        .add()
        .unwrap();
    let middle = middle.finish_fragment();
    assert_eq!(middle.context().len(), 1, "the table came up a level");

    let outer = compiled_fragment(".version 8.0\n// outer\n", "outer");
    let mut root = <RecipeBuilder>::new(0, 1);
    root.choice(vec![middle, outer])
        .outputs([Buf::Output(0)])
        .does([Buf::Output(0)], Access::Write)
        .add()
        .unwrap();
    let recipe = root
        .finish(RecipeSignature::new(vec![], vec![4]))
        .unwrap()
        .into_recipe();

    assert_eq!(recipe.context().len(), 2);
    let mut named = Vec::new();
    requests_named(recipe.body(), &mut named);
    for index in named.into_iter().flatten() {
        assert!(
            index < recipe.context().len(),
            "a nested module names {index}, past the {} the root has",
            recipe.context().len()
        );
    }
    assert_eq!(recipe.validate(), Ok(()));
    assert_eq!(
        deserialize(&serialize(&recipe).unwrap()).unwrap(),
        recipe,
        "and it survives the wire with the table intact"
    );
}

/// Landing a fragment moves its table too, which is what keeps the module
/// index it carries meaningful in its new home.
#[test]
fn splicing_a_fragment_carries_its_compile_table() {
    let mut host = <RecipeBuilder>::new(0, 1);
    // A request the host already holds, so the spliced one cannot land on 0.
    let existing = host.compile_request(ptx_request(".version 8.0\n// host\n"));
    let existing = host.requested_module(existing, Vec::new(), ResourceBounds::default());
    host.kernel(existing, "host")
        .write(Writable::Output(0))
        .add();

    let guest = compiled_fragment(".version 8.0\n// guest\n", "guest");
    host.splice(guest, &[], &[BufRef::Output(0)], &[]).unwrap();
    let recipe = host
        .finish(RecipeSignature::new(vec![], vec![4]))
        .unwrap()
        .into_recipe();

    assert_eq!(recipe.context().len(), 2);
    let mut named = Vec::new();
    requests_named(recipe.body(), &mut named);
    let indices: Vec<usize> = named.into_iter().flatten().collect();
    assert_eq!(indices, vec![0, 1], "the guest was remapped off 0");
    assert_eq!(recipe.validate(), Ok(()));
}

/// `into_fragment` is the one way to spend a finished recipe as nested work: the
/// table travels so the module `request` indices keep meaning something, and
/// the ports stay behind because whatever selects the fragment binds it.
#[test]
fn a_recipe_becomes_a_fragment_without_its_ports() {
    let mut b = <RecipeBuilder>::new(1, 1);
    b.memset(Writable::Output(0).at(0), 0, 4);
    let recipe = b.finish(RecipeSignature::new(vec![64], vec![4])).unwrap();
    let table = recipe.recipe().context().clone();
    let body = recipe.recipe().body().clone();

    let fragment: RecipeFragment = recipe.into_fragment();
    assert_eq!(*fragment.context(), table);
    assert_eq!(*fragment.body(), body);
    assert!(
        fragment.is_proved(),
        "a built recipe keeps what it proved when it becomes a candidate"
    );
}

/// A fragment gives up its body only when it has no root data to strand. The
/// modules index that table, so dropping a non-empty one would leave their
/// `request` pointing at nothing.
#[test]
fn a_fragment_keeps_its_body_while_it_still_names_a_table() {
    let tableless: RecipeFragment = RecipeFragment::from_body(RecipeBody::new(
        0,
        0,
        vec![KernelModule::from_cuda_loadable(vec![0xAA])],
        Vec::new(),
        Vec::new(),
    ));
    assert!(tableless.into_body().is_ok());

    let compiled = compiled_fragment(".version 8.0", "k");
    assert!(
        !compiled.context().is_empty(),
        "the fixture must name a table for this to be the case under test"
    );
    assert_eq!(
        compiled.into_body(),
        Err(RecipeError::FragmentLeavesRootContext)
    );
}

/// The root is the only place a contract attaches, and a candidate keeps none:
/// two candidates built from recipes with *different* stated ports become the
/// same-shaped choice, bound through the root's.
#[test]
fn candidates_carry_no_ports_of_their_own() {
    let leaf = |nbytes: usize| {
        let mut b = <RecipeBuilder>::new(1, 1);
        b.memset(Writable::Output(0).at(0), 0, 4);
        b.finish(RecipeSignature::new(vec![nbytes], vec![4]))
            .unwrap()
            .into_fragment()
    };
    let mut root = <RecipeBuilder>::new(1, 1);
    root.choice(vec![leaf(64), leaf(4096)])
        .inputs([Buf::Input(0)])
        .does([Buf::Input(0)], Access::Read)
        .outputs([Buf::Output(0)])
        .does([Buf::Output(0)], Access::Write)
        .add()
        .unwrap();
    let recipe = root
        .finish(RecipeSignature::new(vec![64], vec![4]))
        .unwrap()
        .into_recipe();
    assert_eq!(recipe.validate(), Ok(()));

    let Work::Choice { candidates, .. } = recipe.body().nodes()[0].work() else {
        panic!("a choice")
    };
    assert_eq!(
        candidates[0], candidates[1],
        "the ports the two were built with are gone; only the work remains"
    );
}

/// Landing is where a sub-recipe's formal slots become the host's buffers, so
/// it is where a store can first turn out to have nowhere legal to land. The
/// host binds the guest's only formal output to its own input; the guest writes
/// through it, and this is refused rather than retargeted or panicked over.
#[test]
fn splicing_a_store_onto_a_host_input_is_refused() {
    let guest = {
        let mut b = <RecipeBuilder>::new(0, 1);
        b.memset(Writable::Output(0).at(0), 0, 4);
        b.finish_fragment()
    };
    let mut host = <RecipeBuilder>::new(1, 0);
    assert_eq!(
        host.splice(guest, &[], &[BufRef::Input(0)], &[]),
        Err(RecipeError::WriteToReadOnlyBuffer {
            node: 0,
            kind: BufKind::Input,
            index: 0,
        })
    );
}

/// A refused landing leaves the host exactly as it was — every table it owns,
/// not only the ones the failing node touched.
///
/// The guest carries a compile request no host entry can intern with, so a
/// table that grew would grow visibly; it also carries a module, a workspace
/// buffer, and a legal node ahead of the illegal one, so a landing that
/// committed as it went would leave all four behind. What a caller holds after
/// reporting the refusal is the builder it had before it asked.
#[test]
fn a_refused_splice_adds_nothing_to_the_host() {
    let guest = || {
        let mut b = <RecipeBuilder>::new(0, 1);
        b.scratch(64);
        let request = b.compile_request(ptx_request(".version 8.0\n// guest only\n"));
        let m = b.requested_module(request, Vec::new(), ResourceBounds::default());
        b.kernel(m, "k").write(Writable::Output(0)).add();
        b.memset(Writable::Output(0).at(0), 0, 4);
        b.finish_fragment()
    };
    assert_eq!(
        guest().context().len(),
        1,
        "the fixture must name a table for this to be the case under test"
    );

    let host = || {
        let mut b = <RecipeBuilder>::new(1, 0);
        let request = b.compile_request(ptx_request(".version 8.0\n// host\n"));
        b.requested_module(request, Vec::new(), ResourceBounds::default());
        b.scratch(32);
        b
    };

    let mut refused = host();
    assert_eq!(
        refused.splice(guest(), &[], &[BufRef::Input(0)], &[]),
        Err(RecipeError::WriteToReadOnlyBuffer {
            node: 1,
            kind: BufKind::Input,
            index: 0,
        }),
        "the second of the guest's nodes is the store the binding sends nowhere"
    );
    // Landing the same guest again must fail the same way: a first attempt that
    // had interned the guest's request would make the second one dedupe against
    // it and report a different table.
    assert!(
        refused
            .splice(guest(), &[], &[BufRef::Input(0)], &[])
            .is_err()
    );

    let ports = || RecipeSignature::new(vec![64], vec![]);
    assert_eq!(
        refused.finish(ports()).unwrap().into_recipe(),
        host().finish(ports()).unwrap().into_recipe(),
        "a refused landing is not an edit"
    );
}

/// The same binding is fine for work that only reads through the slot: the
/// output binding is an ABI binding, and a pointer is not a store.
#[test]
fn splicing_a_pointer_onto_a_host_input_is_allowed() {
    let guest = {
        let mut b = <RecipeBuilder>::new(0, 1);
        let m = b.module(vec![0xEE]);
        b.kernel(m, "k").pointer_at(Buf::Output(0), 8).add();
        b.finish_fragment()
    };
    let mut host = <RecipeBuilder>::new(1, 0);
    host.splice(guest, &[], &[BufRef::Input(0)], &[]).unwrap();
    let recipe = host
        .finish(RecipeSignature::new(vec![64], vec![]))
        .unwrap()
        .into_recipe();
    assert_eq!(recipe.validate(), Ok(()));
    let Work::Call {
        call: KernelLaunch { args, .. },
        ..
    } = recipe.body().nodes()[0].work()
    else {
        panic!("a launch")
    };
    assert_eq!(
        args[0].segments(),
        [recipe::Segment::Pointer(BufRef::Input(0).at(8))],
        "the slot became the host's input, and the offset crossed unchanged"
    );
}

/// Landing shifts workspace indices; an address into one keeps the byte offset
/// it had, because the offset is inside the buffer the slot selects.
#[test]
fn splicing_keeps_byte_offsets_while_it_moves_slots() {
    let guest = {
        let mut b = <RecipeBuilder>::new(1, 0);
        let ws = b.scratch(64);
        b.memcpy(Buf::Input(0).at(16), Writable::Workspace(ws).at(8), 32);
        b.finish_fragment()
    };
    let mut host = <RecipeBuilder>::new(1, 0);
    host.scratch(128);
    host.splice(guest, &[BufRef::Input(0)], &[], &[]).unwrap();
    let recipe = host
        .finish(RecipeSignature::new(vec![64], vec![]))
        .unwrap()
        .into_recipe();
    assert_eq!(recipe.validate(), Ok(()));
    let Work::Memcpy {
        source,
        destination,
        nbytes,
    } = recipe.body().nodes()[0].work()
    else {
        panic!("a copy")
    };
    assert_eq!(*source, BufRef::Input(0).at(16));
    assert_eq!(*destination, WritableBuf::Workspace(1).at(8));
    assert_eq!(*nbytes, 32);
}

/// Interning collapses equal compiles, so two prerequisites that named
/// different requests can land on one — and what they then say about it is
/// both of them.
///
/// The guest's table states one compile twice, at indices 0 and 1, and guards
/// a third on each: one with a register floor, the other with a spill ceiling.
/// Nothing about them is redundant, so a collect that kept whichever entry
/// arrived last would drop a real constraint and admit an image the producer
/// ruled out. The host already holds an unrelated request, so the two guests
/// are renumbered rather than landing where they started.
#[test]
fn two_prerequisites_that_intern_to_one_request_keep_both_constraints() {
    let floor = ResourceBounds {
        min_registers: Some(129),
        max_local_bytes: None,
    };
    let ceiling = ResourceBounds {
        min_registers: None,
        max_local_bytes: Some(0),
    };

    let guest = {
        let base = ptx_request(".version 8.0\n// base\n");
        let mut guarded = ptx_request(".version 8.0\n// guarded\n");
        guarded.require(CompileRequestId(0), floor);
        guarded.require(CompileRequestId(1), ceiling);
        // Two entries stating the same compile: only interning discovers that,
        // which is exactly when the two guards collide.
        RecipeFragment::unproved(
            CompileRequests::new(vec![base.clone(), base, guarded]),
            RecipeBody::new(
                0,
                1,
                vec![KernelModule::from_request(
                    CompileRequestId(2),
                    Vec::new(),
                    ResourceBounds::default(),
                )],
                Vec::new(),
                vec![Node::new(
                    launch(0, "guarded", [1, 1, 1], vec![Arg::output(0)]),
                    [],
                )],
            ),
        )
    };

    let mut host = <RecipeBuilder>::new(0, 1);
    host.compile_request(ptx_request(".version 8.0\n// host\n"));
    host.splice(guest, &[], &[BufRef::Output(0)], &[]).unwrap();
    let recipe = host
        .finish(RecipeSignature::new(vec![], vec![4]))
        .unwrap()
        .into_recipe();

    assert_eq!(
        recipe.context().len(),
        3,
        "the guest's duplicate base interned onto one entry"
    );
    let guarded = recipe
        .context()
        .as_slice()
        .last()
        .expect("the guarded request");
    let [(request, bounds)] = guarded.requires.iter().collect::<Vec<_>>()[..] else {
        panic!(
            "both guards are about one request, so there is one entry: {:?}",
            guarded.requires
        );
    };
    assert_ne!(
        *request,
        CompileRequestId(0),
        "the guest's requests were renumbered onto the host's table"
    );
    assert_eq!(
        *bounds,
        ResourceBounds {
            min_registers: Some(129),
            max_local_bytes: Some(0),
        },
        "the floor and the ceiling both survived the collision"
    );
    assert_eq!(recipe.validate(), Ok(()));
}

/// A compile that links nothing has no image to produce, and no assembler to
/// ask for one. Refusing it where a recipe comes into being is what keeps every
/// consumer below from having to decide what an empty unit would mean.
#[test]
fn a_compile_unit_states_at_least_one_source() {
    assert_eq!(
        <Recipe>::stating_edges(
            RecipeSignature::empty(),
            CompileRequests::new(vec![CompileRequest::unguarded(
                Vec::new(),
                PtxCompileOptions::default(),
            )]),
            RecipeBody::default(),
        ),
        Err(RecipeError::Context(ExecutableRefusal::EmptyCompileUnit {
            request: 0
        }))
    );
}
