// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reuse of a completed realization: the artifact a previous `realize` selected,
//! kept under the identity of everything that produced it.

use std::path::PathBuf;
use std::time::Instant;

#[cfg(test)]
use recipe::CompileRequests;
use recipe::Recipe;
use recipe::algebra::Launches;

use crate::identity::{Framed, RecipeIdentity};
use crate::realize::ArtifactPolicy;

/// A family whose realized artifacts this cache can hold.
///
/// An entry is recipe bytes in the public wire format, and that format states
/// the default contract — so the default family is the one whose artifacts go
/// in and come back out as the value they were. A family bound through a
/// richer contract answers `None` to both questions, and a realization of one
/// is reported as bypassing the cache rather than quietly writing bytes that
/// state something else.
///
/// Both directions are asked of the family rather than assumed, because the
/// cache reads back what it wrote: a family that can only be written down
/// could be stored and never recovered.
pub trait Storable: Launches {
    /// This recipe as the one the cache stores, when that is what it is.
    fn as_stored(recipe: &Recipe<Self>) -> Option<&Recipe<recipe::Executable>> {
        let _ = recipe;
        None
    }

    /// A stored recipe as this family's, when that is what it is.
    fn from_stored(recipe: Recipe<recipe::Executable>) -> Option<Recipe<Self>> {
        let _ = recipe;
        None
    }
}

impl Storable for recipe::Executable {
    fn as_stored(recipe: &Recipe<Self>) -> Option<&Recipe<recipe::Executable>> {
        Some(recipe)
    }

    fn from_stored(recipe: Recipe<recipe::Executable>) -> Option<Recipe<Self>> {
        Some(recipe)
    }
}

/// The caller's name for the data a tuning result was measured against.
///
/// Kiln never derives one. Shapes and device pointers do not identify a
/// workload: a zero-skip candidate wins or loses by the *distribution* in those
/// buffers, so an identity synthesized from what kiln can see would promise a
/// reuse it cannot honour. Only the caller knows when two runs are the same
/// workload; a caller that cannot say so tunes without a cache.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WorkloadIdentity(Vec<u8>);

impl WorkloadIdentity {
    pub fn new(name: impl Into<Vec<u8>>) -> Self {
        WorkloadIdentity(name.into())
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// The strongest hardware identity kiln can reach: the device's architectural
/// profile and the driver that loads images onto it.
///
/// Compute capability alone is too coarse — two parts of one architecture
/// differ in SM count, and the fastest candidate on a 148-SM part need not be
/// the fastest on a 78-SM one. A device *name* or UUID would be stronger still,
/// but `cuda_driver` exposes neither, and a UUID would in any case be the wrong
/// granularity: it names one card, so eight identical GPUs in a node would tune
/// eight times and share nothing. The profile below names the model's
/// performance-relevant shape, which is what a tuning result is a function of.
/// Two distinct models agreeing on every one of these attributes would alias;
/// none is known, and the failure mode is a suboptimal selection, never a wrong
/// program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceIdentity {
    pub sm: u16,
    pub sm_count: u32,
    pub max_threads_per_sm: u32,
    pub max_smem_optin: u32,
    /// `cuDriverGetVersion`: the JIT and loader that turn an image into a
    /// running kernel.
    pub driver_version: u32,
}

/// What about this machine decides the artifact: the device the winner was
/// measured on and the assemblers the pending recipe's requests can reach.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetIdentity {
    pub device: DeviceIdentity,
    /// One entry per assembler the recipe's assembly modes resolve to, including
    /// the reason an unavailable one is unavailable: a machine without `ptxas`
    /// eliminates every whole-program candidate and so realizes a different
    /// artifact. Empty for a recipe with no compile request, which keeps a
    /// cubin-only artifact independent of any installed toolchain.
    pub assemblers: Vec<String>,
}

/// The full identity of a realized artifact: which program, measured against
/// which workload, on which target, emitted under which output policy.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArtifactKey([u8; 32]);

impl ArtifactKey {
    pub fn new(
        program: RecipeIdentity,
        workload: &WorkloadIdentity,
        target: &TargetIdentity,
        policy: ArtifactPolicy,
    ) -> Self {
        ArtifactKey::framed(program, workload, target, policy, recipe::wire::VERSION)
    }

    /// The key an artifact written in Recipe wire `protocol` is addressed by.
    ///
    /// A stored artifact is bytes in one protocol, so the protocol is part of
    /// what the key names. Bumping it moves every entry to an address this
    /// build never asks for: the old bytes are unreachable rather than
    /// misread, they block nothing under the new key, and the store's own
    /// eviction reclaims them. That is what keeps a format change from
    /// needing a cache that can overwrite — an entry under this key was
    /// written by a build that read these bytes the way this one does.
    fn framed(
        program: RecipeIdentity,
        workload: &WorkloadIdentity,
        target: &TargetIdentity,
        policy: ArtifactPolicy,
        protocol: u16,
    ) -> Self {
        let mut digest = Framed::new("kiln-realized-artifact-v1");
        digest.number(u64::from(protocol));
        digest.digest(program.as_bytes());
        digest.bytes(workload.as_bytes());
        digest
            .number(u64::from(target.device.sm))
            .number(u64::from(target.device.sm_count))
            .number(u64::from(target.device.max_threads_per_sm))
            .number(u64::from(target.device.max_smem_optin))
            .number(u64::from(target.device.driver_version))
            .count(target.assemblers.len());
        for assembler in &target.assemblers {
            digest.bytes(assembler.as_bytes());
        }
        digest.byte(match policy {
            ArtifactPolicy::NativeOnly => 0,
            ArtifactPolicy::NativeWithSourceFallback => 1,
            ArtifactPolicy::SourceBacked => 2,
        });
        ArtifactKey(digest.finish())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl std::fmt::Debug for ArtifactKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Where realized artifacts live. `dir: None` keeps them in this process only.
#[derive(Clone, Debug)]
pub struct ArtifactCacheConfig {
    pub dir: Option<PathBuf>,
    /// Ceiling on the bytes the directory holds; `None` never evicts.
    pub max_bytes: Option<u64>,
}

/// A cache of completed realizations, owned by whoever realizes.
///
/// Explicitly owned rather than a process global: two callers realizing for
/// different purposes (a persistent artifact store, a scratch one in a test)
/// hold different caches, and a caller that wants none passes none.
pub struct ArtifactCache {
    store: blob_cache::Cache,
}

pub struct CachedArtifact {
    pub recipe: Recipe,
    /// `"memory"` or `"disk"` — where this hit came from, measured now.
    pub source: &'static str,
    pub load_ms: u64,
}

struct Absent;

impl ArtifactCache {
    pub fn open(config: ArtifactCacheConfig) -> Self {
        ArtifactCache {
            store: blob_cache::Cache::open(config.dir, config.max_bytes),
        }
    }

    /// The artifact stored under `key`, or `None` when there is none.
    ///
    /// `Err` is a stored entry that could not be read back — a truncated write
    /// or the entry of a colliding storage hash. Every one of those is a miss
    /// to the caller; the message says which. A foreign wire protocol is not
    /// among them: the key names the protocol, so an entry written in another
    /// one is not addressed here at all.
    pub fn load(&self, key: &ArtifactKey) -> Result<Option<CachedArtifact>, String> {
        let started = Instant::now();
        let Ok(hit) = self
            .store
            .get_or_insert_reported(key.as_bytes(), EXTENSION, || Err(Absent))
        else {
            return Ok(None);
        };
        let (stored_key, wire) = hit
            .bytes
            .split_at_checked(size_of::<ArtifactKey>())
            .ok_or_else(|| "cached artifact is shorter than its key".to_string())?;
        // The store addresses entries by its own hash of the key; the key
        // travels inside the value so that identity stays BLAKE3's.
        if stored_key != key.as_bytes() {
            return Err("cached artifact is stored under a different key".into());
        }
        let recipe = recipe::deserialize(wire).map_err(|error| error.to_string())?;
        Ok(Some(CachedArtifact {
            recipe,
            source: hit.source.name(),
            load_ms: started.elapsed().as_millis() as u64,
        }))
    }

    /// Keep `artifact` under `key`, unless an entry is already there.
    ///
    /// Storing does not displace one, and does not need to: the key names the
    /// program, the machine, the policy, and the wire protocol, so anything
    /// already under it is an artifact this build would have produced again.
    pub fn store(&self, key: &ArtifactKey, artifact: &Recipe) -> Result<(), String> {
        let mut bytes = Vec::from(key.as_bytes());
        bytes.extend_from_slice(&recipe::serialize(artifact).map_err(|error| error.to_string())?);
        self.store
            .get_or_insert(key.as_bytes(), EXTENSION, || Ok::<_, String>(bytes))?;
        Ok(())
    }
}

const EXTENSION: &str = "recipe";

/// What one realization did with its artifact cache. Exactly one of these
/// holds per `realize` call, so a report never leaves the question open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArtifactCacheOutcome {
    /// The caller passed no cache.
    Disabled,
    /// A cache was passed but the realization could not be identified, so
    /// nothing was looked up or stored.
    Unidentified { reason: String },
    /// A cache was passed and deliberately not consulted: the key does not name
    /// what this realization was asked to do, so an entry under it would answer
    /// a different question. Nothing was looked up or stored.
    Bypassed { reason: String },
    /// A previous realization was reused: no compile of a losing candidate, no
    /// measurement.
    Hit { source: &'static str },
    /// Identified, absent, and kept after this realization completed.
    Stored,
    /// Identified and absent, but this realization is not the artifact the key
    /// promises.
    NotStored { reason: String },
    /// Identified and absent, and storing what was realized failed.
    StoreFailed { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use recipe::{KernelModule, RecipeBody};

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

    fn program() -> RecipeIdentity {
        RecipeIdentity::of(
            &<Recipe>::stating_edges(
                recipe::RecipeSignature::empty(),
                CompileRequests::default(),
                RecipeBody::new(
                    0,
                    0,
                    vec![KernelModule::from_cuda_loadable(b"image".to_vec())],
                    Vec::new(),
                    Vec::new(),
                ),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn key_of(workload: &str, target: &TargetIdentity, policy: ArtifactPolicy) -> ArtifactKey {
        ArtifactKey::new(program(), &WorkloadIdentity::new(workload), target, policy)
    }

    fn on(device: DeviceIdentity) -> TargetIdentity {
        TargetIdentity { device, ..target() }
    }

    #[test]
    fn key_separates_every_component() {
        let base = key_of("workload", &target(), ArtifactPolicy::NativeOnly);
        assert_eq!(
            base,
            key_of("workload", &target(), ArtifactPolicy::NativeOnly)
        );
        assert_ne!(base, key_of("other", &target(), ArtifactPolicy::NativeOnly));
        assert_ne!(
            base,
            key_of("workload", &target(), ArtifactPolicy::SourceBacked)
        );
        for change in [
            on(DeviceIdentity { sm: 90, ..device() }),
            on(DeviceIdentity {
                sm_count: 78,
                ..device()
            }),
            on(DeviceIdentity {
                max_threads_per_sm: 1536,
                ..device()
            }),
            on(DeviceIdentity {
                max_smem_optin: 101376,
                ..device()
            }),
            on(DeviceIdentity {
                driver_version: 12090,
                ..device()
            }),
            TargetIdentity {
                assemblers: vec!["ptxas@12.8.0".into()],
                ..target()
            },
            TargetIdentity {
                assemblers: Vec::new(),
                ..target()
            },
        ] {
            assert_ne!(
                base,
                key_of("workload", &change, ArtifactPolicy::NativeOnly),
                "{change:?} must not share a key with {:?}",
                target()
            );
        }
    }

    fn artifact(image: &[u8]) -> Recipe {
        // No node, so nothing names a port.
        Recipe::stating_edges(
            recipe::RecipeSignature::empty(),
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

    #[test]
    fn a_stored_artifact_comes_back_from_a_new_cache_over_the_same_directory() {
        let dir = std::env::temp_dir().join("kiln_artifact_cache_roundtrip");
        let _ = std::fs::remove_dir_all(&dir);
        let key = key_of("workload", &target(), ArtifactPolicy::NativeOnly);
        let config = ArtifactCacheConfig {
            dir: Some(dir.clone()),
            max_bytes: None,
        };
        ArtifactCache::open(config.clone())
            .store(&key, &artifact(b"image"))
            .unwrap();

        let reopened = ArtifactCache::open(config);
        let hit = reopened.load(&key).unwrap().expect("stored artifact");
        assert_eq!(hit.source, "disk");
        assert_eq!(hit.recipe, artifact(b"image"));
        assert!(
            reopened
                .load(&key_of("other", &target(), ArtifactPolicy::NativeOnly))
                .unwrap()
                .is_none(),
            "another workload is another artifact"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_in_memory_cache_keeps_artifacts_for_this_process_only() {
        let cache = ArtifactCache::open(ArtifactCacheConfig {
            dir: None,
            max_bytes: None,
        });
        let key = key_of("workload", &target(), ArtifactPolicy::NativeOnly);
        assert!(cache.load(&key).unwrap().is_none());
        cache.store(&key, &artifact(b"image")).unwrap();
        let hit = cache.load(&key).unwrap().expect("stored artifact");
        assert_eq!(hit.source, "memory");
        assert!(
            ArtifactCache::open(ArtifactCacheConfig {
                dir: None,
                max_bytes: None,
            })
            .load(&key)
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn a_value_stored_under_a_foreign_key_is_not_a_hit() {
        let dir = std::env::temp_dir().join("kiln_artifact_cache_collision");
        let _ = std::fs::remove_dir_all(&dir);
        let config = ArtifactCacheConfig {
            dir: Some(dir.clone()),
            max_bytes: None,
        };
        let cache = ArtifactCache::open(config);
        let stored = key_of("workload", &target(), ArtifactPolicy::NativeOnly);
        cache.store(&stored, &artifact(b"image")).unwrap();

        // What a storage-hash collision would deliver: the bytes of one key's
        // entry read under another key.
        let other = key_of("other", &target(), ArtifactPolicy::NativeOnly);
        let entry = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|entry| entry.path())
            .find(|path| path.extension().is_some_and(|ext| ext == EXTENSION))
            .expect("stored entry");
        let bytes = std::fs::read(&entry).unwrap();
        let colliding = ArtifactCache::open(ArtifactCacheConfig {
            dir: None,
            max_bytes: None,
        });
        colliding
            .store
            .get_or_insert(other.as_bytes(), EXTENSION, || Ok::<_, String>(bytes))
            .unwrap();
        assert!(
            colliding.load(&other).is_err(),
            "an entry that does not carry this key is not this artifact"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An artifact is bytes in one wire protocol, so the protocol is part of
    /// the address it is kept at. A build that reads a different one asks a
    /// different question and must not be answered by, or blocked by, what the
    /// previous one left behind.
    #[test]
    fn an_entry_from_another_wire_protocol_is_not_addressed_by_this_one() {
        let key = |protocol| {
            ArtifactKey::framed(
                program(),
                &WorkloadIdentity::new("workload"),
                &target(),
                ArtifactPolicy::NativeOnly,
                protocol,
            )
        };
        let previous = key(recipe::wire::VERSION - 1);
        let current = key(recipe::wire::VERSION);
        assert_ne!(previous, current, "a bump moves every entry");
        assert_eq!(
            current,
            ArtifactKey::new(
                program(),
                &WorkloadIdentity::new("workload"),
                &target(),
                ArtifactPolicy::NativeOnly,
            )
        );

        let cache = ArtifactCache::open(ArtifactCacheConfig {
            dir: None,
            max_bytes: None,
        });
        // What the previous build left: an artifact keyed the way it keyed
        // things, whose bytes this build cannot read.
        let stale = {
            let mut bytes = Vec::from(previous.as_bytes());
            let mut wire = recipe::serialize(&artifact(b"stale")).unwrap();
            wire[4..6].copy_from_slice(&(recipe::wire::VERSION - 1).to_le_bytes());
            bytes.extend_from_slice(&wire);
            bytes
        };
        cache
            .store
            .get_or_insert(previous.as_bytes(), EXTENSION, || Ok::<_, String>(stale))
            .unwrap();

        assert_eq!(
            cache.load(&current).unwrap().map(|hit| hit.recipe),
            None,
            "the stale entry is not reachable from this build's key"
        );
        let fresh = artifact(b"fresh");
        cache.store(&current, &fresh).unwrap();
        assert_eq!(
            cache.load(&current).unwrap().map(|hit| hit.recipe),
            Some(fresh),
            "and it never occupied the address this build stores at"
        );
    }
}
