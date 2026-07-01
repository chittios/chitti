//! A minimal cooperative async executor (`CHITTI_OS_HANDOFF.md` Phase 2
//! scope: "agent work is yield-heavy"). This is deliberately separate
//! from the stackful task scheduler in `sched::mod`: it lets one stackful
//! task multiplex many `Future`s (e.g. several pending IPC replies) over
//! a single stack instead of paying for one kernel stack per pending
//! operation. There is no timer-preemption at this layer -- that's the
//! stackful scheduler's job one level up; a future that never yields
//! (returns `Poll::Pending` and arranges a wake) simply starves its
//! siblings, same as any cooperative executor.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::sync::Arc;
use alloc::task::Wake;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};

pub type AsyncTaskId = u64;

struct AsyncTask {
    future: Pin<Box<dyn Future<Output = ()>>>,
}

impl AsyncTask {
    fn new(future: impl Future<Output = ()> + 'static) -> Self {
        Self { future: Box::pin(future) }
    }

    fn poll(&mut self, cx: &mut Context) -> Poll<()> {
        self.future.as_mut().poll(cx)
    }
}

struct TaskWaker {
    id: AsyncTaskId,
    wake_queue: Arc<crate::mm::Locked<VecDeque<AsyncTaskId>>>,
}

impl Wake for TaskWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.wake_queue.with(|q| q.push_back(self.id));
    }
}

/// A tiny run queue of `Future<Output = ()>`s, woken via a shared
/// `wake_queue` rather than a busy-poll-every-future scan.
pub struct Executor {
    tasks: BTreeMap<AsyncTaskId, AsyncTask>,
    run_queue: VecDeque<AsyncTaskId>,
    wakers: BTreeMap<AsyncTaskId, Waker>,
    next_id: AsyncTaskId,
    wake_queue: Arc<crate::mm::Locked<VecDeque<AsyncTaskId>>>,
}

impl Executor {
    pub fn new() -> Self {
        Self {
            tasks: BTreeMap::new(),
            run_queue: VecDeque::new(),
            wakers: BTreeMap::new(),
            next_id: 0,
            wake_queue: Arc::new(crate::mm::Locked::new(VecDeque::new())),
        }
    }

    pub fn spawn(&mut self, future: impl Future<Output = ()> + 'static) -> AsyncTaskId {
        let id = self.next_id;
        self.next_id += 1;
        self.tasks.insert(id, AsyncTask::new(future));
        self.run_queue.push_back(id);
        id
    }

    /// Run until every spawned future has completed. Bounded so a future
    /// that never wakes itself (or a bug in this executor) fails loudly
    /// instead of spinning the host forever.
    pub fn run(&mut self) {
        let mut idle_spins = 0u64;
        loop {
            self.wake_queue.with(|q| {
                while let Some(id) = q.pop_front() {
                    if self.tasks.contains_key(&id) {
                        self.run_queue.push_back(id);
                    }
                }
            });

            let Some(id) = self.run_queue.pop_front() else {
                if self.tasks.is_empty() {
                    return;
                }
                idle_spins += 1;
                assert!(idle_spins < 100_000_000, "executor: pending futures but none ever woke");
                continue;
            };
            idle_spins = 0;

            let Some(task) = self.tasks.get_mut(&id) else { continue };
            let wake_queue = self.wake_queue.clone();
            let waker =
                self.wakers.entry(id).or_insert_with(|| Waker::from(Arc::new(TaskWaker { id, wake_queue }))).clone();
            let mut cx = Context::from_waker(&waker);
            if task.poll(&mut cx).is_ready() {
                self.tasks.remove(&id);
                self.wakers.remove(&id);
            }
        }
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}
