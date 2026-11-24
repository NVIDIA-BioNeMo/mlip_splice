// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile requests as a shared table: what makes two of them one compile, and
//! how a recipe's requests move onto another recipe's indices.

use std::collections::BTreeMap;

use crate::code::{
    CompileRequest, CompileRequestId, PtxCompileOptions, PtxTarget, ResourceBounds, SourceKind,
    conjoined_requires,
};

/// Which compile requests are worth comparing: the whole ordered unit it links
/// — each source by kind, target and *claimed* digest — the machine and
/// options it assembles under, and the interned requests it presupposes.
///
/// A bucket, not an identity. An encoded source carries the digest its
/// artifact states, which nothing has checked until the bytes are expanded, so
/// two requests landing in one bucket are merged only once
/// [`same_unit`] proves their sources hold the same bytes.
///
/// The unit is a sequence here for the same reason it is one in the value: two
/// requests over the same modules in a different link order are two compiles.
pub(crate) type RequestKey = (
    Vec<(SourceKind, PtxTarget, [u8; 32])>,
    PtxCompileOptions,
    Vec<(usize, ResourceBounds)>,
);

/// Whether two requests link the same modules, in order — proved on the bytes.
///
/// Only a repeat this can prove is collapsed. A source that cannot be expanded
/// is not shown to be anyone's twin, so it stays its own request and the
/// consumer that assembles it reports what is wrong with it, rather than
/// vanishing into an entry built from bytes it does not hold.
pub(crate) fn same_unit(left: &CompileRequest, right: &CompileRequest) -> bool {
    left.sources.len() == right.sources.len()
        && left
            .sources
            .iter()
            .zip(&right.sources)
            .all(|(left, right)| left.blob.same_content(&right.blob).unwrap_or(false))
}

pub(crate) fn request_key(request: &CompileRequest) -> Option<RequestKey> {
    Some((
        request
            .sources
            .iter()
            .map(|source| {
                Some((
                    source.kind,
                    source.target,
                    source.blob.claimed_digest().ok()?,
                ))
            })
            .collect::<Option<Vec<_>>>()?,
        request.options,
        request
            .requires
            .iter()
            .map(|(request, bounds)| (request.0, *bounds))
            .collect(),
    ))
}

/// The requests in an order where every prerequisite precedes what needs it, or
/// `None` when the table has a cycle or names a request it does not hold.
pub(crate) fn dependency_order(requests: &[CompileRequest]) -> Option<Vec<usize>> {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        Unvisited,
        Visiting,
        Ordered,
    }
    fn visit(
        request: usize,
        requests: &[CompileRequest],
        marks: &mut [Mark],
        order: &mut Vec<usize>,
    ) -> Option<()> {
        match marks.get(request)? {
            Mark::Ordered => return Some(()),
            Mark::Visiting => return None,
            Mark::Unvisited => {}
        }
        marks[request] = Mark::Visiting;
        for prerequisite in requests[request].requires.keys() {
            visit(prerequisite.0, requests, marks, order)?;
        }
        marks[request] = Mark::Ordered;
        order.push(request);
        Some(())
    }

    let mut marks = vec![Mark::Unvisited; requests.len()];
    let mut order = Vec::with_capacity(requests.len());
    for request in 0..requests.len() {
        visit(request, requests, &mut marks, &mut order)?;
    }
    Some(order)
}

/// The same guards against renumbered requests.
///
/// Interning collapses equal compiles, so two prerequisites that named
/// different requests can land on one. What they then say about it is both of
/// them, which is why this conjoins rather than collects: a plain collect would
/// keep whichever the map saw last and drop the other constraint.
pub(crate) fn shift_requires(
    requires: &BTreeMap<CompileRequestId, ResourceBounds>,
    onto: impl Fn(usize) -> usize,
) -> BTreeMap<CompileRequestId, ResourceBounds> {
    conjoined_requires(
        requires
            .iter()
            .map(|(request, bounds)| (CompileRequestId(onto(request.0)), *bounds)),
    )
}

#[cfg(test)]
mod tests {
    use crate::{
        Blob, Buf, CompileRequest, CompileRequestId, ProvedRecipe, PtxAssemblyMode,
        PtxCompileOptions, PtxTarget, RecipeBuilder, RecipeSignature, ResourceBounds, SourceImage,
        SourceKind, Writable,
    };

    fn request(ptx: &str, max_registers: Option<u32>) -> CompileRequest {
        CompileRequest::unguarded(
            vec![SourceImage {
                kind: SourceKind::Ptx,
                target: PtxTarget {
                    isa: [8, 7],
                    minimum_sm: 90,
                },
                blob: Blob::from_vec(ptx.as_bytes().to_vec()),
            }],
            PtxCompileOptions {
                max_registers,
                assembly_mode: PtxAssemblyMode::Relocatable,
            },
        )
    }

    /// One candidate: a module built from `request`, launching `entry`.
    /// A candidate built as a whole recipe — ports and all — so the call sites
    /// below have to say `into_fragment` and leave them behind in writing.
    fn candidate(request: CompileRequest, entry: &str) -> ProvedRecipe {
        let mut builder = RecipeBuilder::new(1, 1);
        let id = builder.compile_request(request);
        let module = builder.requested_module(id, Vec::new(), ResourceBounds::default());
        builder
            .kernel(module, entry)
            .read(Buf::Input(0))
            .write(Writable::Output(0))
            .add();
        builder.finish(ports(1, 1)).unwrap()
    }

    /// Ports for a structural test: stated rather than defaulted, and distinct so a
    /// swapped port shows up as a different signature.
    fn ports(inputs: usize, outputs: usize) -> RecipeSignature {
        RecipeSignature::new(
            (0..inputs).map(|port| 64 * (port + 1)).collect(),
            (0..outputs).map(|port| 32 * (port + 1)).collect(),
        )
    }
    #[test]
    fn candidates_that_share_a_compile_share_one_request() {
        let mut builder = RecipeBuilder::new(1, 1);
        builder
            .whole_recipe_choice(
                vec![
                    candidate(request("kernel", None), "loom_a").into_fragment(),
                    candidate(request("kernel", None), "loom_a").into_fragment(),
                ],
                1,
                1,
            )
            .unwrap();
        let recipe = builder.finish(ports(1, 1)).unwrap().into_recipe();

        assert_eq!(recipe.context().len(), 1);
        recipe.validate().unwrap();
    }

    #[test]
    fn a_differing_option_stays_its_own_request() {
        let mut builder = RecipeBuilder::new(1, 1);
        builder
            .whole_recipe_choice(
                vec![
                    candidate(request("kernel", None), "loom_a").into_fragment(),
                    candidate(request("kernel", Some(128)), "loom_a").into_fragment(),
                ],
                1,
                1,
            )
            .unwrap();
        let recipe = builder.finish(ports(1, 1)).unwrap().into_recipe();
        let table = recipe.context().as_slice();

        assert_eq!(table.len(), 2);
        assert_eq!(table[0].options.max_registers, None);
        assert_eq!(table[1].options.max_registers, Some(128));
    }

    /// A compile is identified by its whole unit, in order: the same modules
    /// linked the other way round resolve different symbols and are a
    /// different compile, so interning keeps them apart.
    #[test]
    fn link_order_is_part_of_a_compile_s_identity() {
        let source = |ptx: &str| SourceImage {
            kind: SourceKind::Ptx,
            target: PtxTarget {
                isa: [8, 7],
                minimum_sm: 90,
            },
            blob: Blob::from_vec(ptx.as_bytes().to_vec()),
        };
        let unit = |sources: Vec<SourceImage>| {
            CompileRequest::unguarded(
                sources,
                PtxCompileOptions {
                    max_registers: None,
                    assembly_mode: PtxAssemblyMode::Relocatable,
                },
            )
        };
        let mut builder = <RecipeBuilder>::new(0, 0);
        let forward = builder.compile_request(unit(vec![source("caller"), source("callee")]));
        let again = builder.compile_request(unit(vec![source("caller"), source("callee")]));
        let reversed = builder.compile_request(unit(vec![source("callee"), source("caller")]));
        let alone = builder.compile_request(unit(vec![source("caller")]));
        assert_eq!(forward, again, "the same unit is one compile");
        assert_ne!(forward, reversed);
        assert_ne!(forward, alone);
    }

    /// A capped candidate carries its uncapped twin; interning must land that
    /// twin on the entry the plain candidate already made, and the capped
    /// request's prerequisite must follow it there.
    #[test]
    fn a_prerequisite_follows_the_request_it_names() {
        let capped = {
            let mut builder = RecipeBuilder::new(1, 1);
            let base = builder.compile_request(request("kernel", None));
            let mut guarded = request("kernel", Some(128));
            guarded.require(
                base,
                ResourceBounds {
                    max_local_bytes: Some(0),
                    ..ResourceBounds::default()
                },
            );
            let capped = builder.compile_request(guarded);
            let module = builder.requested_module(capped, Vec::new(), ResourceBounds::default());
            builder
                .kernel(module, "loom_a")
                .read(Buf::Input(0))
                .write(Writable::Output(0))
                .add();
            builder.finish(ports(1, 1)).unwrap()
        };
        let mut builder = RecipeBuilder::new(1, 1);
        // The plain candidate first, so the capped one's base has to intern
        // onto an entry that is already there.
        builder
            .whole_recipe_choice(
                vec![
                    candidate(request("kernel", None), "loom_a").into_fragment(),
                    capped.into_fragment(),
                ],
                1,
                1,
            )
            .unwrap();
        let recipe = builder.finish(ports(1, 1)).unwrap().into_recipe();
        let table = recipe.context().as_slice();

        assert_eq!(table.len(), 2);
        assert_eq!(
            table[1].requires.keys().copied().collect::<Vec<_>>(),
            vec![CompileRequestId(0)]
        );
        recipe.validate().unwrap();
    }
}
