// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Arc, mpsc};

use parking_lot::{Condvar, Mutex};

type Callback = Box<dyn FnOnce() + Send + 'static>;

enum TaskState {
    Queued(Option<Callback>),
    Running,
    Finished,
    Cancelled,
}

struct SharedTask {
    state: Mutex<TaskState>,
    finished: Condvar,
}

impl SharedTask {
    fn claim(self: &Arc<Self>) -> Option<RunningTask<'_>> {
        let callback = {
            let mut state = self.state.lock();
            let TaskState::Queued(callback) = &mut *state else {
                return None;
            };
            let callback = callback.take().expect("queued compile callback");
            *state = TaskState::Running;
            callback
        };
        Some(RunningTask {
            callback: Some(callback),
            _completion: RunningCompletion { task: self },
        })
    }

    fn wait_until_terminal(&self) {
        let mut state = self.state.lock();
        while matches!(*state, TaskState::Queued(_) | TaskState::Running) {
            self.finished.wait(&mut state);
        }
    }
}

// Fields drop in declaration order. A callback that has not yet been invoked is
// therefore destroyed before its completion is published, including on unwind.
struct RunningTask<'a> {
    callback: Option<Callback>,
    _completion: RunningCompletion<'a>,
}

impl RunningTask<'_> {
    fn run(mut self) {
        self.callback.take().expect("claimed compile callback")();
    }
}

struct RunningCompletion<'a> {
    task: &'a SharedTask,
}

impl Drop for RunningCompletion<'_> {
    fn drop(&mut self) {
        let mut state = self.task.state.lock();
        debug_assert!(matches!(*state, TaskState::Running));
        *state = TaskState::Finished;
        drop(state);
        self.task.finished.notify_all();
    }
}

/// The lifetime owner of one computation submitted to the compile pool.
///
/// Dropping it cancels work that has not started. If the work is already
/// running, drop waits until its callback and callback-owned values have been
/// destroyed. A buffered result is then destroyed with the receiver. This does
/// not promise that the inert Rayon wrapper has returned or that a worker is
/// parked, and progress still depends on a running native call returning.
#[must_use = "dropping a compile task cancels it or waits for its callback"]
pub struct CompileTask<R> {
    // A cancelled callback contains the sender, so it must be destroyed before
    // the receiver. Both happen only after the shared-state mutex is released.
    cancelled_callback: Option<Callback>,
    receiver: Option<mpsc::Receiver<R>>,
    task: Arc<SharedTask>,
    // Resolver ownership stays on its host thread. In particular, no Rayon
    // worker can move the owner onto the stack of the callback it is joining.
    not_send: PhantomData<Rc<()>>,
}

impl<R> CompileTask<R> {
    pub(crate) fn new(
        compile: impl FnOnce() -> R + Send + 'static,
    ) -> (Self, impl FnOnce() + Send + 'static)
    where
        R: Send + 'static,
    {
        let (sender, receiver) = mpsc::sync_channel(1);
        let callback: Callback = Box::new(move || {
            let _ = sender.send(compile());
        });
        let task = Arc::new(SharedTask {
            state: Mutex::new(TaskState::Queued(Some(callback))),
            finished: Condvar::new(),
        });
        let owner = CompileTask {
            cancelled_callback: None,
            receiver: Some(receiver),
            task: Arc::clone(&task),
            not_send: PhantomData,
        };
        let run = move || {
            if let Some(claimed) = task.claim() {
                claimed.run();
            }
        };
        (owner, run)
    }

    /// Cancel the callback if its pool wrapper has not claimed it.
    ///
    /// The callback is retained by this owner so callers can cancel a batch
    /// before any callback destructors run. Dropping the owner destroys it.
    pub fn cancel_if_queued(&mut self) -> bool {
        if self.cancelled_callback.is_some() {
            return true;
        }
        let callback = {
            let mut state = self.task.state.lock();
            let TaskState::Queued(callback) = &mut *state else {
                return false;
            };
            let callback = callback.take().expect("queued compile callback");
            *state = TaskState::Cancelled;
            self.task.finished.notify_all();
            callback
        };
        self.cancelled_callback = Some(callback);
        true
    }

    /// Receive the computation's result after its callback has fully retired.
    pub fn recv(mut self) -> Result<R, mpsc::RecvError> {
        // A caller may have cancelled a batch before deciding to receive one
        // member. Retire its sender first so receive reports disconnection.
        drop(self.cancelled_callback.take());
        let result = self
            .receiver
            .take()
            .expect("compile result receiver taken once")
            .recv();
        self.task.wait_until_terminal();
        result
    }
}

impl<R> Drop for CompileTask<R> {
    fn drop(&mut self) {
        if self.receiver.is_none() {
            return;
        }
        if !self.cancel_if_queued() {
            self.task.wait_until_terminal();
        }
    }
}

#[cfg(test)]
#[path = "compile_task_tests.rs"]
mod tests;
