// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The realized-artifact cache through its public surface, with no device: what
//! a key separates, what survives a process boundary, and what the directory
//! is allowed to grow to.

use std::path::PathBuf;

use kiln::{
    ArtifactCache, ArtifactCacheConfig, ArtifactKey, ArtifactPolicy, DeviceIdentity,
    RecipeIdentity, TargetIdentity, WorkloadIdentity,
};
use recipe::{
    Blob, CompileRequest, CompileRequests, KernelModule, PtxAssemblyMode, PtxCompileOptions,
    PtxTarget, Recipe, RecipeBody, RecipeSignature, SourceImage, SourceKind,
};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("kiln_artifact_cache_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn device() -> DeviceIdentity {
    DeviceIdentity {
        sm: 100,
        sm_count: 148,
        max_threads_per_sm: 2048,
        max_smem_optin: 232448,
        driver_version: 12080,
    }
}

fn target() -> TargetIdentity {
    TargetIdentity {
        device: device(),
        assemblers: vec!["nvjitlink@12.8.0".into()],
    }
}

fn native(image: &[u8]) -> Recipe {
    // No node names a port, so this program is bound through nothing.
    Recipe::stating_edges(
        RecipeSignature::empty(),
        CompileRequests::default(),
        RecipeBody::new(
            0,
            0,
            vec![KernelModule::from_cuda_loadable(image.to_vec())],
            Vec::new(),
            Vec::new(),
        ),
    )
    .unwrap()
}

fn source_backed(ptx: &[u8]) -> Recipe {
    Recipe::stating_edges(
        RecipeSignature::empty(),
        CompileRequests::new(vec![CompileRequest {
            sources: vec![SourceImage {
                kind: SourceKind::Ptx,
                target: PtxTarget {
                    isa: [8, 8],
                    minimum_sm: 100,
                },
                blob: Blob::zstd_spilled(ptx).unwrap(),
            }],
            options: PtxCompileOptions {
                max_registers: Some(128),
                assembly_mode: PtxAssemblyMode::Relocatable,
            },
            requires: Default::default(),
        }]),
        RecipeBody::new(
            0,
            0,
            vec![KernelModule::from_request(
                recipe::CompileRequestId(0),
                Vec::new(),
                recipe::ResourceBounds::default(),
            )],
            Vec::new(),
            Vec::new(),
        ),
    )
    .unwrap()
}

fn key(pending: &Recipe, workload: &str, policy: ArtifactPolicy) -> ArtifactKey {
    ArtifactKey::new(
        RecipeIdentity::of(pending).unwrap(),
        &WorkloadIdentity::new(workload),
        &target(),
        policy,
    )
}

#[test]
fn an_artifact_outlives_the_cache_object_that_stored_it() {
    let dir = scratch("across_objects");
    let config = ArtifactCacheConfig {
        dir: Some(dir.clone()),
        max_bytes: None,
    };
    let pending = native(b"pending");
    let artifact = native(b"selected");
    let key = key(&pending, "gto-64-particles", ArtifactPolicy::NativeOnly);

    ArtifactCache::open(config.clone())
        .store(&key, &artifact)
        .unwrap();
    let hit = ArtifactCache::open(config)
        .load(&key)
        .unwrap()
        .expect("a stored artifact is found by a later process");
    assert_eq!(hit.recipe, artifact);
    assert_eq!(hit.source, "disk");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_component_of_the_key_invalidates_on_its_own() {
    let dir = scratch("invalidation");
    let config = ArtifactCacheConfig {
        dir: Some(dir.clone()),
        max_bytes: None,
    };
    let cache = ArtifactCache::open(config);
    let pending = native(b"pending");
    let stored = key(&pending, "gto-64-particles", ArtifactPolicy::NativeOnly);
    cache.store(&stored, &native(b"selected")).unwrap();
    assert!(cache.load(&stored).unwrap().is_some());

    // Another program, the same workload.
    assert!(
        cache
            .load(&key(
                &native(b"other pending"),
                "gto-64-particles",
                ArtifactPolicy::NativeOnly
            ))
            .unwrap()
            .is_none()
    );
    // The same program, other data: a tuning result is a function of both.
    assert!(
        cache
            .load(&key(
                &pending,
                "gto-4096-particles",
                ArtifactPolicy::NativeOnly
            ))
            .unwrap()
            .is_none()
    );
    // The same realization, a different artifact asked of it.
    for policy in [
        ArtifactPolicy::NativeWithSourceFallback,
        ArtifactPolicy::SourceBacked,
    ] {
        assert!(
            cache
                .load(&key(&pending, "gto-64-particles", policy))
                .unwrap()
                .is_none(),
            "{policy:?} is a different artifact"
        );
    }
    // Another device, and another toolchain on the same device.
    for target in [
        TargetIdentity {
            device: DeviceIdentity {
                sm_count: 78,
                ..device()
            },
            ..target()
        },
        TargetIdentity {
            assemblers: vec!["nvjitlink@12.9.0".into()],
            ..target()
        },
    ] {
        let elsewhere = ArtifactKey::new(
            RecipeIdentity::of(&pending).unwrap(),
            &WorkloadIdentity::new("gto-64-particles"),
            &target,
            ArtifactPolicy::NativeOnly,
        );
        assert!(cache.load(&elsewhere).unwrap().is_none(), "{target:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_source_backed_artifact_survives_the_cache_intact() {
    let dir = scratch("source_backed");
    let cache = ArtifactCache::open(ArtifactCacheConfig {
        dir: Some(dir.clone()),
        max_bytes: None,
    });
    let artifact = source_backed(b".version 8.8\n.target sm_100\n");
    let key = key(&artifact, "wx-bwd2", ArtifactPolicy::SourceBacked);
    cache.store(&key, &artifact).unwrap();

    let hit = cache.load(&key).unwrap().expect("stored artifact");
    assert_eq!(hit.recipe.context().as_slice().len(), 1);
    assert_eq!(
        hit.recipe.context().as_slice()[0].options.max_registers,
        Some(128)
    );
    assert_eq!(
        hit.recipe.context().as_slice()[0].sources[0]
            .blob
            .content_digest()
            .unwrap(),
        artifact.context().as_slice()[0].sources[0]
            .blob
            .content_digest()
            .unwrap()
    );
    assert_eq!(hit.recipe.body().context()[0].code.images().count(), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_directory_stays_under_its_configured_ceiling() {
    let dir = scratch("bounded");
    let cache = ArtifactCache::open(ArtifactCacheConfig {
        dir: Some(dir.clone()),
        max_bytes: Some(4096),
    });
    let big = vec![0u8; 8192];
    for workload in ["a", "b", "c", "d"] {
        let pending = native(b"pending");
        cache
            .store(
                &key(&pending, workload, ArtifactPolicy::NativeOnly),
                &native(&big),
            )
            .unwrap();
    }
    let total: u64 = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .filter_map(|entry| entry.metadata().ok())
        .filter(|metadata| metadata.is_file())
        .map(|metadata| metadata.len())
        .sum();
    assert!(
        total <= 4096 + big.len() as u64,
        "an unbounded cache would hold every artifact ({total} bytes)"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_in_memory_cache_never_touches_the_filesystem() {
    let cache = ArtifactCache::open(ArtifactCacheConfig {
        dir: None,
        max_bytes: None,
    });
    let pending = native(b"pending");
    let key = key(&pending, "workload", ArtifactPolicy::NativeOnly);
    assert!(cache.load(&key).unwrap().is_none());
    cache.store(&key, &native(b"selected")).unwrap();
    assert_eq!(cache.load(&key).unwrap().unwrap().source, "memory");
}

/// A pending recipe whose source claims another program's identity is not that
/// program, and receives none of its work.
///
/// The artifact under a key is compiled code. An encoded source states a digest
/// nothing has checked until its bytes are expanded, so an identity that
/// believed the claim would hand a forged recipe the honest one's realized
/// artifact — and the forged bytes would never be read at all. Identity is over
/// what a payload holds, so bytes that are not what they say are not
/// identifiable, and a caller that cannot identify a recipe neither reads nor
/// writes the cache under it.
#[test]
fn a_forged_source_cannot_receive_an_honest_program_s_artifact() {
    // Equal decoded length, so the second source can be made to claim the
    // first one's identity by overwriting nothing but a digest.
    let honest_ptx: &[u8] = b".visible .entry caller() { call scale; }\n";
    let other_ptx: &[u8] = b".visible .func  scale()  { ret;        }\n";
    assert_eq!(honest_ptx.len(), other_ptx.len());

    // Both programs written into one artifact, then entry 1's digest replaced
    // by entry 0's: what an attacker can write, and what decoding lets through.
    let stated = Recipe::stating_edges(
        RecipeSignature::empty(),
        CompileRequests::new(vec![
            source_backed(honest_ptx)
                .into_parts()
                .1
                .into_vec()
                .remove(0),
            source_backed(other_ptx).into_parts().1.into_vec().remove(0),
        ]),
        RecipeBody::default(),
    )
    .unwrap();
    let mut artifact = recipe::serialize(&stated).unwrap();
    let word = |bytes: &[u8], at: usize| {
        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize
    };
    let blobs = (0..word(&artifact, 8))
        .map(|entry| 12 + entry * 20)
        .find(|&at| word(&artifact, at) == 1)
        .map(|at| u64::from_le_bytes(artifact[at + 4..at + 12].try_into().unwrap()) as usize)
        .expect("the BLOBS section");
    let first = blobs + 4;
    let second = first + 45 + word(&artifact, first + 41);
    let stolen = artifact[first + 9..first + 41].to_vec();
    artifact[second + 9..second + 41].copy_from_slice(&stolen);
    let both = recipe::deserialize(&artifact).unwrap();

    let request = |index: usize| -> Recipe {
        Recipe::stating_edges(
            RecipeSignature::empty(),
            CompileRequests::new(vec![both.context().as_slice()[index].clone()]),
            RecipeBody::new(
                0,
                0,
                vec![KernelModule::from_request(
                    recipe::CompileRequestId(0),
                    Vec::new(),
                    recipe::ResourceBounds::default(),
                )],
                Vec::new(),
                Vec::new(),
            ),
        )
        .unwrap()
    };
    let honest = request(0);
    let forged = request(1);
    assert_eq!(
        honest.context().as_slice()[0].sources[0]
            .blob
            .claimed_digest()
            .unwrap(),
        forged.context().as_slice()[0].sources[0]
            .blob
            .claimed_digest()
            .unwrap(),
        "the forged source claims the honest one's identity"
    );

    // The honest program is identifiable and its artifact is stored.
    let dir = scratch("forged_identity");
    let cache = ArtifactCache::open(ArtifactCacheConfig {
        dir: Some(dir.clone()),
        max_bytes: None,
    });
    let stored = native(b"honest cubin");
    let key = key(&honest, "one-workload", ArtifactPolicy::NativeOnly);
    cache.store(&key, &stored).unwrap();
    assert_eq!(cache.load(&key).unwrap().unwrap().recipe, stored);

    // The forged one is not, so there is no key under which it could be
    // handed that artifact.
    assert_eq!(
        RecipeIdentity::of(&forged),
        Err(recipe::BlobDecodeError::Digest),
        "bytes that are not what they claim name no program"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
