// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, mpsc};
use std::thread;
use std::time::Duration;

const LIMIT: Duration = Duration::from_secs(3);

struct DropWitness {
    count: Arc<AtomicUsize>,
    mutex_was_free: Option<Arc<SharedTask>>,
}

impl Drop for DropWitness {
    fn drop(&mut self) {
        if let Some(task) = &self.mutex_was_free {
            assert!(task.state.try_lock().is_some());
        }
        self.count.fetch_add(1, Ordering::SeqCst);
    }
}

fn pending<R: Send + 'static>(
    compile: impl FnOnce() -> R + Send + 'static,
) -> (CompileTask<R>, impl FnOnce() + Send + 'static) {
    CompileTask::new(compile)
}

#[test]
fn queued_drop_cancels_callback_and_releases_captures() {
    let calls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let capture = DropWitness {
        count: Arc::clone(&drops),
        mutex_was_free: None,
    };
    let called = Arc::clone(&calls);
    let (task, wrapper) = pending(move || {
        let _capture = capture;
        called.fetch_add(1, Ordering::SeqCst);
    });

    drop(task);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    wrapper();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn cancelled_capture_is_destroyed_outside_state_mutex() {
    let drops = Arc::new(AtomicUsize::new(0));
    let shared = Arc::new(SharedTask {
        state: Mutex::new(TaskState::Cancelled),
        finished: Condvar::new(),
    });
    let capture = DropWitness {
        count: Arc::clone(&drops),
        mutex_was_free: Some(Arc::clone(&shared)),
    };
    let (sender, receiver) = mpsc::sync_channel(1);
    let callback: Callback = Box::new(move || {
        let _capture = capture;
        let _ = sender.send(());
    });
    *shared.state.lock() = TaskState::Queued(Some(callback));
    let mut task = CompileTask {
        cancelled_callback: None,
        receiver: Some(receiver),
        task: Arc::clone(&shared),
        not_send: PhantomData,
    };

    assert!(task.cancel_if_queued());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(task);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(shared.claim().is_none());
}

#[test]
fn cancel_and_start_race_runs_at_most_once() {
    for _ in 0..100 {
        let calls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(Barrier::new(2));
        let called = Arc::clone(&calls);
        let capture = DropWitness {
            count: Arc::clone(&drops),
            mutex_was_free: None,
        };
        let (task, wrapper) = pending(move || {
            let _capture = capture;
            called.fetch_add(1, Ordering::SeqCst);
        });
        let worker_start = Arc::clone(&start);
        let worker = thread::spawn(move || {
            worker_start.wait();
            wrapper();
        });
        start.wait();
        drop(task);
        worker.join().unwrap();
        assert!(calls.load(Ordering::SeqCst) <= 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn running_drop_waits_for_callback() {
    let (entered, started) = mpsc::sync_channel(1);
    let (release, wait) = mpsc::sync_channel(1);
    let (task, wrapper) = pending(move || {
        entered.send(()).unwrap();
        wait.recv_timeout(LIMIT).unwrap();
        7
    });
    let worker = thread::spawn(wrapper);
    started.recv_timeout(LIMIT).unwrap();
    let controller = thread::spawn(move || {
        thread::sleep(Duration::from_millis(30));
        release.send(()).unwrap();
    });
    let started_waiting = std::time::Instant::now();
    drop(task);
    assert!(started_waiting.elapsed() >= Duration::from_millis(20));
    controller.join().unwrap();
    worker.join().unwrap();
}

#[test]
fn recv_waits_for_cleanup_after_result_is_sent() {
    let (sender, receiver) = mpsc::sync_channel(1);
    let (cleanup_entered, entered) = mpsc::sync_channel(1);
    let (cleanup_release, release) = mpsc::sync_channel(1);
    struct BlockingDrop {
        entered: mpsc::SyncSender<()>,
        release: mpsc::Receiver<()>,
    }
    impl Drop for BlockingDrop {
        fn drop(&mut self) {
            self.entered.send(()).unwrap();
            self.release.recv_timeout(LIMIT).unwrap();
        }
    }
    let cleanup = BlockingDrop {
        entered: cleanup_entered,
        release,
    };
    let callback: Callback = Box::new(move || {
        sender.send(11).unwrap();
        drop(cleanup);
    });
    let shared = Arc::new(SharedTask {
        state: Mutex::new(TaskState::Queued(Some(callback))),
        finished: Condvar::new(),
    });
    let task = CompileTask {
        cancelled_callback: None,
        receiver: Some(receiver),
        task: Arc::clone(&shared),
        not_send: PhantomData,
    };
    let wrapper = thread::spawn(move || shared.claim().unwrap().run());
    let controller = thread::spawn(move || {
        entered.recv_timeout(LIMIT).unwrap();
        thread::sleep(Duration::from_millis(30));
        cleanup_release.send(()).unwrap();
    });

    let started_waiting = std::time::Instant::now();
    assert_eq!(task.recv().unwrap(), 11);
    assert!(started_waiting.elapsed() >= Duration::from_millis(20));
    controller.join().unwrap();
    wrapper.join().unwrap();
}

#[test]
fn dropping_finished_task_destroys_buffered_result() {
    let drops = Arc::new(AtomicUsize::new(0));
    let result = DropWitness {
        count: Arc::clone(&drops),
        mutex_was_free: None,
    };
    let (task, wrapper) = pending(move || result);
    wrapper();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(task);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn callback_panic_disconnects_only_after_capture_cleanup() {
    let drops = Arc::new(AtomicUsize::new(0));
    let capture = DropWitness {
        count: Arc::clone(&drops),
        mutex_was_free: None,
    };
    let (task, wrapper) = pending(move || -> () {
        let _capture = capture;
        panic!("compile panic");
    });
    assert!(catch_unwind(AssertUnwindSafe(wrapper)).is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(task.recv().is_err());
}

#[test]
fn single_worker_can_cancel_its_queued_task() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let called = Arc::clone(&calls);
    pool.install(|| {
        let (task, wrapper) = pending(move || called.fetch_add(1, Ordering::SeqCst));
        pool.spawn(wrapper);
        drop(task);
    });
    pool.install(|| {});
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn cancellation_batch_precedes_callback_destruction() {
    let destroyed = Arc::new(AtomicUsize::new(0));
    let all_cancelled = Arc::new(AtomicBool::new(false));
    let mut tasks = Vec::new();
    let mut wrappers = Vec::new();
    for _ in 0..3 {
        let destroyed = Arc::clone(&destroyed);
        let all_cancelled = Arc::clone(&all_cancelled);
        struct CheckBatch {
            destroyed: Arc<AtomicUsize>,
            all_cancelled: Arc<AtomicBool>,
        }
        impl Drop for CheckBatch {
            fn drop(&mut self) {
                assert!(self.all_cancelled.load(Ordering::SeqCst));
                self.destroyed.fetch_add(1, Ordering::SeqCst);
            }
        }
        let capture = CheckBatch {
            destroyed,
            all_cancelled,
        };
        let (task, wrapper) = pending(move || drop(capture));
        tasks.push(task);
        wrappers.push(wrapper);
    }
    for task in &mut tasks {
        assert!(task.cancel_if_queued());
    }
    all_cancelled.store(true, Ordering::SeqCst);
    drop(tasks);
    assert_eq!(destroyed.load(Ordering::SeqCst), 3);
    for wrapper in wrappers {
        wrapper();
    }
}

#[test]
fn receiving_a_cancelled_task_reports_disconnection() {
    let calls = Arc::new(AtomicUsize::new(0));
    let called = Arc::clone(&calls);
    let (mut task, wrapper) = pending(move || {
        called.fetch_add(1, Ordering::SeqCst);
        3
    });
    assert!(task.cancel_if_queued());
    assert!(task.recv().is_err());
    wrapper();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
