// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Projecting one selected recipe into the artifact a policy asks for: which
//! compile requests survive, and which modules keep a source to be realized
//! from again.

use std::collections::{BTreeSet, HashMap};

use recipe::algebra::{BytePortContract, Launches, ModuleTable};
use recipe::{CompileRequest, KernelModule, Recipe};

use crate::Error;
use crate::realize::ArtifactPolicy;
use crate::resolve::ResolvedRecipe;

/// The recipe `policy` asks this selection to be written as.
///
/// `requests` is the pending recipe's whole table; only the closure the
/// selected modules resolved from — each request and everything it presupposes
/// — reaches the artifact, reindexed onto its own table.
pub(crate) fn artifact<F: Launches>(
    requests: &[CompileRequest],
    signature: F::Contract,
    resolved: &ResolvedRecipe<F>,
    policy: ArtifactPolicy,
    authorization: &F::Authorization,
) -> Result<Recipe<F>, Error> {
    // An artifact is a whole program, so this is where the resolved body meets
    // the ports the recipe it came from states — and where the two are proved
    // against each other.
    let of = |modules, workspace, nodes, table: Vec<CompileRequest>| {
        let (inputs, outputs) = (signature.num_inputs(), signature.num_outputs());
        Recipe::stating_edges_authorized(
            signature.clone(),
            recipe::CompileRequests::new(table),
            recipe::RecipeBody::new(inputs, outputs, modules, workspace, nodes),
            authorization,
        )
        .map_err(|error| Error::Decode(error.to_string()))
    };
    if resolved.body.context().modules().len() != resolved.origins.len() {
        return Err(Error::Decode(
            "resolved module origins do not match the selected body".into(),
        ));
    }
    let (_, _, mut modules, workspace, nodes) = resolved.body.clone().into_parts();
    if matches!(policy, ArtifactPolicy::NativeOnly) {
        let mut artifact = of(modules, workspace, nodes, Vec::new())?;
        reject_session_only(&artifact)?;
        artifact.detach_blobs();
        return Ok(artifact);
    }
    // The artifact is a different program from the one that was measured: it
    // draws from its own table, and the table below is the one the modules are
    // then rewritten onto.
    let mut required = BTreeSet::new();
    for request in resolved.origins.iter().flatten().copied() {
        collect_prerequisites(request, requests, &mut required);
    }
    let remap: HashMap<_, _> = required
        .iter()
        .enumerate()
        .map(|(new, &old)| (old, new))
        .collect();
    let table: Vec<CompileRequest> = required
        .iter()
        .map(|&old| {
            let mut request = requests[old].clone();
            // The projection's remap is injective, but a guard about one
            // request is still the conjunction of everything said about it, so
            // this collects the way every other renumbering path does.
            request.requires = recipe::conjoined_requires(request.requires.iter().map(
                |(prerequisite, bounds)| {
                    (recipe::CompileRequestId(remap[&prerequisite.0]), *bounds)
                },
            ));
            request
        })
        .collect();
    for (module, request) in modules.modules_mut().iter_mut().zip(&resolved.origins) {
        let Some(old) = request else {
            continue;
        };
        let compile_request = recipe::CompileRequestId(remap[old]);
        // A resolved module is one image. Whether the artifact keeps it is the
        // policy's question; whether it *may* be kept is the image's own.
        let images = if matches!(policy, ArtifactPolicy::SourceBacked) {
            Vec::new()
        } else {
            module
                .code
                .images()
                .filter(|image| image.is_persistable())
                .cloned()
                .collect()
        };
        module.code = recipe::KernelCode::PtxBacked {
            compile_request,
            images,
        };
    }
    // A module with a compile behind it has had its session-only images
    // dropped above; one without has nothing to drop them in favour of, so
    // what is left may still not outlive this session.
    let mut artifact = of(modules, workspace, nodes, table)?;
    reject_session_only(&artifact)?;
    artifact.detach_blobs();
    Ok(artifact)
}

/// An artifact is bytes another process will load, so nothing in it may stop
/// at this session — whichever policy asked for it, and whether or not the
/// module it sits in was compiled here.
fn reject_session_only<F: Launches>(artifact: &Recipe<F>) -> Result<(), Error> {
    match artifact
        .body()
        .context()
        .modules()
        .iter()
        .position(module_is_session_only)
    {
        Some(module) => Err(Error::ArtifactPolicy(format!(
            "module {module} carries an image that does not outlive this session, \
             and has no compile to be realized from again"
        ))),
        None => Ok(()),
    }
}

fn collect_prerequisites(
    request: usize,
    requests: &[CompileRequest],
    required: &mut BTreeSet<usize>,
) {
    if !required.insert(request) {
        return;
    }
    for prerequisite in requests[request].requires.keys() {
        collect_prerequisites(prerequisite.0, requests, required);
    }
}

/// Whether a module would carry bytes that do not outlive this session.
///
/// The image says so itself. Reading an assembler's name here would make the
/// answer depend on a string a producer is free to choose, and on this
/// build's spelling of it.
fn module_is_session_only(module: &KernelModule) -> bool {
    module.code.images().any(|image| !image.is_persistable())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realize::RealizerOptions;
    use crate::resolve::Resolver;
    use recipe::{
        Blob, CompileProvenance, ImagePersistence, KernelCode, ModuleImage, PtxAssemblyMode,
        PtxCompileOptions, RecipeBody, ResourceBounds, SourceImage, SourceKind,
    };

    fn request(max_registers: Option<u32>) -> CompileRequest {
        CompileRequest {
            sources: vec![SourceImage {
                kind: SourceKind::Ptx,
                target: recipe::PtxTarget {
                    isa: [8, 8],
                    minimum_sm: 100,
                },
                blob: Blob::from_vec(b"ptx".to_vec()),
            }],
            options: PtxCompileOptions {
                max_registers,
                assembly_mode: PtxAssemblyMode::Relocatable,
            },
            requires: Default::default(),
        }
    }

    #[test]
    fn artifact_policy_keeps_only_selected_request_closure() {
        let mut capped = request(Some(128));
        capped.require(
            recipe::CompileRequestId(0),
            ResourceBounds {
                max_local_bytes: Some(0),
                ..ResourceBounds::default()
            },
        );
        // The bodies here are empty, so nothing names a port.
        let pending = Recipe::stating_edges(
            recipe::RecipeSignature::empty(),
            recipe::CompileRequests::new(vec![request(None), capped, request(Some(64))]),
            RecipeBody::default(),
        )
        .unwrap();
        let resolver = Resolver::new(
            std::ptr::null_mut(),
            100,
            RealizerOptions::default(),
            &pending,
        );
        let resolved: ResolvedRecipe<recipe::Executable> = ResolvedRecipe {
            body: RecipeBody::new(
                0,
                0,
                vec![KernelModule::from_cuda_loadable(vec![1, 2, 3])],
                Vec::new(),
                Vec::new(),
            ),
            origins: vec![Some(1)],
        };

        let portable = resolver
            .artifact(&resolved, ArtifactPolicy::NativeWithSourceFallback, &())
            .unwrap();
        assert_eq!(portable.context().as_slice().len(), 2);
        assert_eq!(
            portable.body().context()[0].code.compile_request(),
            Some(recipe::CompileRequestId(1))
        );
        assert_eq!(portable.body().context()[0].code.images().count(), 1);
        assert_eq!(
            portable.context().as_slice()[1]
                .requires
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            vec![recipe::CompileRequestId(0)]
        );

        let source = resolver
            .artifact(&resolved, ArtifactPolicy::SourceBacked, &())
            .unwrap();
        assert_eq!(source.body().context()[0].code.images().count(), 0);
        assert!(source.validate().is_ok());

        // A session-only image: the driver's JIT of the source, which the
        // image says of itself rather than being recognised by its assembler.
        let (inputs, outputs, mut driver_modules, workspace, nodes) =
            resolved.body.clone().into_parts();
        driver_modules[0].code = KernelCode::ImageOnly {
            preferred_image: ModuleImage::CudaLoadable {
                bytes: Blob::from_vec(vec![1, 2, 3]),
                persistence: ImagePersistence::SessionOnly,
                provenance: Some(CompileProvenance {
                    unit_digest: [0; 32],
                    options: PtxCompileOptions::default(),
                    target_sm: 100,
                    assembler: "cuda-driver-jit".into(),
                    assembler_version: "driver-managed".into(),
                    driver_version: None,
                    toolkit_version: None,
                }),
            },
            alternative_images: Vec::new(),
        };
        let driver_resolved = ResolvedRecipe {
            body: RecipeBody::new(inputs, outputs, driver_modules, workspace, nodes),
            origins: vec![Some(1)],
        };
        assert!(matches!(
            resolver.artifact(&driver_resolved, ArtifactPolicy::NativeOnly, &()),
            Err(Error::ArtifactPolicy(_))
        ));
        let driver_artifact = resolver
            .artifact(
                &driver_resolved,
                ArtifactPolicy::NativeWithSourceFallback,
                &(),
            )
            .unwrap();
        assert_eq!(driver_artifact.body().context()[0].code.images().count(), 0);
    }

    /// An image that stops at this session may not be written into an artifact
    /// under any policy.
    ///
    /// A module with a compile behind it loses the image and keeps the source.
    /// A module with no compile has nothing to lose it in favour of, so the
    /// projection fails rather than handing back bytes the next process cannot
    /// use — and it fails the same way for every policy, because what makes the
    /// bytes unusable is the bytes.
    #[test]
    fn no_policy_writes_an_image_that_stops_at_this_session() {
        let pending = Recipe::stating_edges(
            recipe::RecipeSignature::empty(),
            recipe::CompileRequests::new(vec![request(None)]),
            RecipeBody::default(),
        )
        .unwrap();
        let resolver = Resolver::new(
            std::ptr::null_mut(),
            100,
            RealizerOptions::default(),
            &pending,
        );
        let session_only = |origins: Vec<Option<usize>>| ResolvedRecipe::<recipe::Executable> {
            body: RecipeBody::new(
                0,
                0,
                vec![KernelModule {
                    code: KernelCode::ImageOnly {
                        preferred_image: ModuleImage::CudaLoadable {
                            bytes: Blob::from_vec(vec![1, 2, 3]),
                            persistence: ImagePersistence::SessionOnly,
                            provenance: None,
                        },
                        alternative_images: Vec::new(),
                    },
                    resource_bounds: recipe::ResourceBounds::default(),
                }],
                Vec::new(),
                Vec::new(),
            ),
            origins,
        };

        for policy in [
            ArtifactPolicy::NativeOnly,
            ArtifactPolicy::NativeWithSourceFallback,
            ArtifactPolicy::SourceBacked,
        ] {
            // No compile behind it: there is nothing to fall back on, so every
            // policy refuses rather than storing bytes that will not load.
            assert!(
                matches!(
                    resolver.artifact(&session_only(vec![None]), policy, &()),
                    Err(Error::ArtifactPolicy(_))
                ),
                "{policy:?} kept a session-only image on a module with no compile"
            );
        }

        // The same image on a module that *was* compiled: the two portable
        // policies drop it and keep the source, so nothing session-only is
        // written and the artifact is still realizable.
        for policy in [
            ArtifactPolicy::NativeWithSourceFallback,
            ArtifactPolicy::SourceBacked,
        ] {
            let artifact = resolver
                .artifact(&session_only(vec![Some(0)]), policy, &())
                .unwrap_or_else(|error| panic!("{policy:?} refused a compiled module: {error:?}"));
            assert_eq!(
                artifact.body().context()[0].code.images().count(),
                0,
                "{policy:?} kept a session-only image"
            );
            assert_eq!(
                artifact.body().context()[0].code.compile_request(),
                Some(recipe::CompileRequestId(0)),
                "{policy:?} dropped the compile the image was replaced by"
            );
        }
        // NativeOnly may not write a source, so it has nothing left to offer.
        assert!(matches!(
            resolver.artifact(
                &session_only(vec![Some(0)]),
                ArtifactPolicy::NativeOnly,
                &()
            ),
            Err(Error::ArtifactPolicy(_))
        ));
    }
}
