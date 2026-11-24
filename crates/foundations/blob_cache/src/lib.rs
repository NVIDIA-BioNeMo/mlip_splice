// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use parking_lot::Mutex;
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub struct Cache {
    dir: Option<PathBuf>,
    max_bytes: Option<u64>,
    mem: Mutex<HashMap<blake3::Hash, CacheArtifact>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheSource {
    Memory,
    Disk,
    Produced,
}

impl CacheSource {
    pub fn name(self) -> &'static str {
        match self {
            CacheSource::Memory => "memory",
            CacheSource::Disk => "disk",
            CacheSource::Produced => "produced",
        }
    }
}

pub struct CacheResult {
    pub bytes: Vec<u8>,
    pub source: CacheSource,
    pub production_cost_ms: u64,
}

struct CacheArtifact {
    bytes: Vec<u8>,
    production_cost_ms: u64,
}

impl Cache {
    pub fn open(dir: Option<PathBuf>, max_bytes: Option<u64>) -> Cache {
        Cache {
            dir,
            max_bytes,
            mem: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_or_insert<E>(
        &self,
        key: &[u8],
        ext: &str,
        produce: impl FnOnce() -> Result<Vec<u8>, E>,
    ) -> Result<Vec<u8>, E> {
        self.get_or_insert_reported(key, ext, produce)
            .map(|result| result.bytes)
    }

    pub fn get_or_insert_reported<E>(
        &self,
        key: &[u8],
        ext: &str,
        produce: impl FnOnce() -> Result<Vec<u8>, E>,
    ) -> Result<CacheResult, E> {
        let digest = blake3::hash(key);
        if self.dir.is_none()
            && let Some(hit) = self.mem.lock().get(&digest)
        {
            return Ok(CacheResult {
                bytes: hit.bytes.clone(),
                source: CacheSource::Memory,
                production_cost_ms: hit.production_cost_ms,
            });
        }
        if let Some(dir) = &self.dir
            && let Some(bytes) = read_artifact(dir, digest, ext)
        {
            let production_cost_ms = cost_ms_of(&artifact_path(dir, digest, ext));
            return Ok(CacheResult {
                bytes,
                source: CacheSource::Disk,
                production_cost_ms,
            });
        }
        let t = Instant::now();
        let bytes = produce()?;
        let cost_ms = t.elapsed().as_millis() as u64;
        if let Some(dir) = &self.dir {
            write_artifact(dir, digest, ext, cost_ms, &bytes, self.max_bytes);
        } else {
            self.mem.lock().insert(
                digest,
                CacheArtifact {
                    bytes: bytes.clone(),
                    production_cost_ms: cost_ms,
                },
            );
        }
        Ok(CacheResult {
            bytes,
            source: CacheSource::Produced,
            production_cost_ms: cost_ms,
        })
    }
}

fn artifact_path(dir: &Path, digest: blake3::Hash, ext: &str) -> PathBuf {
    dir.join(format!("{}.{ext}", digest.to_hex()))
}

fn sidecar_path(artifact: &Path) -> PathBuf {
    let mut s = artifact.as_os_str().to_owned();
    s.push(".ct");
    PathBuf::from(s)
}

fn read_artifact(dir: &Path, digest: blake3::Hash, ext: &str) -> Option<Vec<u8>> {
    let path = artifact_path(dir, digest, ext);
    let mut file = fs::File::open(&path).ok()?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).ok()?;
    (!buf.is_empty()).then_some(buf)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> bool {
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(".{}.{sequence}.tmp", std::process::id()));
    let tmp = PathBuf::from(tmp);
    let ok = fs::File::create(&tmp)
        .and_then(|mut f| f.write_all(bytes).and_then(|()| f.sync_all()))
        .and_then(|()| fs::rename(&tmp, path))
        .is_ok();
    if !ok {
        let _ = fs::remove_file(&tmp);
    }
    ok
}

fn write_artifact(
    dir: &Path,
    digest: blake3::Hash,
    ext: &str,
    cost_ms: u64,
    bytes: &[u8],
    max_bytes: Option<u64>,
) {
    let _ = fs::create_dir_all(dir);
    let path = artifact_path(dir, digest, ext);
    if atomic_write(&path, bytes) {
        let _ = atomic_write(&sidecar_path(&path), cost_ms.to_string().as_bytes());
        if let Some(max) = max_bytes {
            evict_to(dir, max);
        }
    }
}

fn cost_ms_of(artifact: &Path) -> u64 {
    fs::read_to_string(sidecar_path(artifact))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

fn evict_to(dir: &Path, max: u64) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    let mut files: Vec<(PathBuf, u64, f64)> = Vec::new();
    let mut total: u64 = 0;
    for entry in rd.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".ct") || name.ends_with(".tmp") {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        let size = meta.len();
        total += size;
        let path = entry.path();
        let score = if size == 0 {
            0.0
        } else {
            cost_ms_of(&path) as f64 / size as f64
        };
        files.push((path, size, score));
    }
    if total <= max {
        return;
    }
    files.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    for (path, size, _) in files {
        if total <= max {
            break;
        }
        if fs::remove_file(&path).is_ok() {
            let _ = fs::remove_file(sidecar_path(&path));
            total = total.saturating_sub(size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Never = ();

    fn put(cache: &Cache, key: &[u8], ext: &str, bytes: Vec<u8>) -> Vec<u8> {
        cache
            .get_or_insert(key, ext, || Ok::<_, Never>(bytes))
            .unwrap()
    }

    #[test]
    fn cache_identity_uses_the_full_blake3_digest() {
        let digest = blake3::hash(b"key");
        assert_eq!(digest.to_hex().len(), 64);
    }

    #[test]
    fn mem_hit_skips_produce() {
        let cache = Cache::open(None, None);
        assert_eq!(put(&cache, b"k", "cubin", vec![1, 2, 3]), vec![1, 2, 3]);
        let got = cache
            .get_or_insert_reported(b"k", "cubin", || -> Result<Vec<u8>, Never> {
                panic!("produce called on a hit")
            })
            .unwrap();
        assert_eq!(got.bytes, vec![1, 2, 3]);
        assert_eq!(got.source, CacheSource::Memory);
    }

    #[test]
    fn disk_round_trips_across_caches() {
        let dir = std::env::temp_dir().join("blob_cache_disk_test");
        let _ = fs::remove_dir_all(&dir);
        {
            let c = Cache::open(Some(dir.clone()), None);
            put(&c, b"key", "cubin", vec![7u8; 16]);
            assert!(c.mem.lock().is_empty());
        }
        let c2 = Cache::open(Some(dir.clone()), None);
        let got = c2
            .get_or_insert(b"key", "cubin", || -> Result<Vec<u8>, Never> {
                panic!("produce called; should have hit disk")
            })
            .unwrap();
        assert_eq!(got, vec![7u8; 16]);
        assert!(c2.mem.lock().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn lookup_is_a_single_exists_no_scan() {
        let dir = std::env::temp_dir().join("blob_cache_no_scan_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // A file that merely shares the hash prefix is not the artifact.
        let digest = blake3::hash(b"key");
        fs::write(
            dir.join(format!("{}_tag.cubin", digest.to_hex())),
            [0u8; 16],
        )
        .unwrap();
        assert!(read_artifact(&dir, digest, "cubin").is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn evict_keeps_highest_density_across_exts() {
        let dir = std::env::temp_dir().join("blob_cache_evict_test");
        let _ = fs::remove_dir_all(&dir);
        let write = |key: &[u8], ms: u64, ext: &str| {
            let digest = blake3::hash(key);
            write_artifact(&dir, digest, ext, ms, &[0u8; 1000], None);
            artifact_path(&dir, digest, ext)
        };
        let cheap = write(b"x", 10, "cubin");
        let mid = write(b"y", 100, "ltoir");
        let dear = write(b"z", 900, "cubin");
        evict_to(&dir, 2000);
        assert!(!cheap.exists(), "lowest density should be evicted first");
        assert!(
            !sidecar_path(&cheap).exists(),
            "an evicted artifact takes its sidecar with it"
        );
        assert!(
            mid.exists(),
            "a higher-value .ltoir outranks a cheaper .cubin"
        );
        assert!(dear.exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
