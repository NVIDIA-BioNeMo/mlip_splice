// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::sync::OnceLock;

use rayon::ThreadPool;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CgroupCpuLimit {
    pub quota_micros: u64,
    pub period_micros: u64,
}

impl CgroupCpuLimit {
    fn concurrent_workers(self) -> usize {
        usize::try_from(self.quota_micros.div_ceil(self.period_micros))
            .unwrap_or(usize::MAX)
            .max(1)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileEnvironment {
    pub available_parallelism: usize,
    pub affinity_cpus: Option<usize>,
    pub cgroup_cpu_limit: Option<CgroupCpuLimit>,
    pub host_memory_bytes: Option<u64>,
    pub cgroup_memory_bytes: Option<u64>,
    pub configured_worker_ceiling: Option<usize>,
    pub workers: usize,
}

impl CompileEnvironment {
    pub fn detect() -> Self {
        let available_parallelism = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let affinity_cpus = cpu_affinity_count();
        let cgroup = cgroup_dir();
        let cgroup_cpu_limit = cgroup
            .as_ref()
            .and_then(|dir| parse_cpu_max(&std::fs::read_to_string(dir.join("cpu.max")).ok()?));
        let host_memory_bytes = host_memory_bytes();
        let cgroup_memory_bytes = cgroup.as_ref().and_then(|dir| {
            parse_memory_max(&std::fs::read_to_string(dir.join("memory.max")).ok()?)
        });
        let configured_worker_ceiling = std::env::var("LOOM_COMPILE_JOBS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&workers| workers > 0);
        let workers = [
            Some(available_parallelism),
            affinity_cpus,
            cgroup_cpu_limit.map(CgroupCpuLimit::concurrent_workers),
            configured_worker_ceiling,
        ]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(1)
        .max(1);
        CompileEnvironment {
            available_parallelism,
            affinity_cpus,
            cgroup_cpu_limit,
            host_memory_bytes,
            cgroup_memory_bytes,
            configured_worker_ceiling,
            workers,
        }
    }
}

static ENVIRONMENT: OnceLock<CompileEnvironment> = OnceLock::new();
static COMPILE_POOL: OnceLock<ThreadPool> = OnceLock::new();

pub fn compile_environment() -> &'static CompileEnvironment {
    ENVIRONMENT.get_or_init(CompileEnvironment::detect)
}

pub fn in_compile_pool<R: Send>(compile: impl FnOnce() -> R + Send) -> R {
    compile_pool().install(compile)
}

fn compile_pool() -> &'static ThreadPool {
    COMPILE_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(compile_environment().workers)
            .thread_name(|index| format!("loom-compile-{index}"))
            .build()
            .expect("compile thread pool")
    })
}

pub fn spawn_compile<R: Send + 'static>(
    compile: impl FnOnce() -> R + Send + 'static,
) -> crate::CompileTask<R> {
    assert!(
        rayon::current_thread_index().is_none(),
        "a compile task owner must be created outside every Rayon pool"
    );
    let (task, run) = crate::CompileTask::new(compile);
    compile_pool().spawn(run);
    task
}

fn cpu_affinity_count() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let list = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))?;
    parse_cpu_list(list.trim())
}

fn parse_cpu_list(list: &str) -> Option<usize> {
    list.split(',').try_fold(0usize, |count, part| {
        let mut bounds = part.split('-');
        let first = bounds.next()?.parse::<usize>().ok()?;
        let last = bounds
            .next()
            .map(str::parse::<usize>)
            .transpose()
            .ok()?
            .unwrap_or(first);
        if last < first {
            return None;
        }
        Some(count + last - first + 1)
    })
}

fn cgroup_dir() -> Option<PathBuf> {
    let cgroup = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let relative = cgroup.lines().find_map(|line| line.strip_prefix("0::"))?;
    Some(PathBuf::from("/sys/fs/cgroup").join(relative.trim_start_matches('/')))
}

fn parse_cpu_max(value: &str) -> Option<CgroupCpuLimit> {
    let mut fields = value.split_whitespace();
    let quota = fields.next()?;
    let period_micros = fields.next()?.parse().ok()?;
    if quota == "max" {
        return None;
    }
    Some(CgroupCpuLimit {
        quota_micros: quota.parse().ok()?,
        period_micros,
    })
}

fn parse_memory_max(value: &str) -> Option<u64> {
    let value = value.trim();
    (value != "max").then(|| value.parse().ok()).flatten()
}

fn host_memory_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))?
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?;
    kib.checked_mul(1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_task_ownership_from_a_rayon_worker() {
        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let callback_called = std::sync::Arc::clone(&called);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let rejected = pool.install(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                drop(spawn_compile(move || {
                    callback_called.store(true, std::sync::atomic::Ordering::SeqCst)
                }));
            }))
            .is_err()
        });
        assert!(rejected);
        assert!(!called.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn parses_sparse_cpu_ranges() {
        assert_eq!(parse_cpu_list("0-3,8,10-11"), Some(7));
        assert_eq!(parse_cpu_list("4"), Some(1));
        assert_eq!(parse_cpu_list("3-1"), None);
    }

    #[test]
    fn parses_cgroup_limits() {
        assert_eq!(parse_cpu_max("max 100000"), None);
        assert_eq!(
            parse_cpu_max("760000 100000"),
            Some(CgroupCpuLimit {
                quota_micros: 760000,
                period_micros: 100000,
            })
        );
        assert_eq!(parse_memory_max("max"), None);
        assert_eq!(parse_memory_max("4096"), Some(4096));
    }
}
