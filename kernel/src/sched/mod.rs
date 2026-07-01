//! Task abstraction, context switching, and the scheduler: Chitti's
//! execution substrate (`CHITTI_OS_HANDOFF.md` Phase 2). Tasks are
//! stackful kernel-mode coroutines, each with its own heap-allocated
//! stack; the scheduler is a round-robin ready queue, entered either
//! voluntarily (`yield_now`) or by the PIT timer once the current task's
//! slice of ticks elapses (`on_timer_tick`), so the same primitive serves
//! both "cooperative" and "timer-preemptive" scheduling.
//!
//! There is no ring 3 / user-mode separation yet -- every task runs in
//! ring 0. The capability system (`cap`) is what enforces "no ambient
//! authority" here: tasks can only reach another task's resources through
//! a `Cap` they were explicitly granted, never by holding a raw pointer
//! or task ID to reach in directly.

pub mod context;
pub mod executor;

use crate::arch::x86_64::interrupts;
use crate::cap::CapTable;
use crate::mm::Locked;
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, VecDeque};
use core::sync::atomic::{AtomicBool, Ordering};

pub type TaskId = u64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TaskState {
    Ready,
    Running,
    Dead,
}

struct TaskControlBlock {
    #[allow(dead_code)] // surfaced only via ktrace today; kept for debugging
    name: &'static str,
    state: TaskState,
    /// Saved stack pointer. Meaningful only while this task is not the
    /// one currently executing; read every switch-out, written every
    /// switch-in.
    rsp: u64,
    /// Kept alive for as long as the task might still be switched to.
    /// Dead tasks' stacks are intentionally leaked rather than freed --
    /// Phase 2 does not yet reclaim task memory (there is no safe point
    /// to free a stack you might be executing on).
    _stack: Option<Box<[u8]>>,
    cap_table: CapTable,
}

struct SchedulerState {
    tasks: BTreeMap<TaskId, TaskControlBlock>,
    ready_queue: VecDeque<TaskId>,
    current: TaskId,
    next_id: TaskId,
    tick_count: u64,
}

static SCHED: Locked<Option<SchedulerState>> = Locked::new(None);
static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// How many timer ticks (1 kHz PIT, see `arch::x86_64::pit`) a task gets
/// before `on_timer_tick` forces a switch.
const TIME_SLICE_TICKS: u64 = 5;

const STACK_SIZE: usize = 64 * 1024;

/// Bring up the scheduler, wrapping whatever's currently executing
/// (`chitti_kernel::init`'s caller) as task 0 ("bootstrap"). Must run
/// after `mm::init` (task control blocks and stacks are heap-allocated).
pub fn init() {
    SCHED.with(|slot| {
        let mut tasks = BTreeMap::new();
        tasks.insert(
            0,
            TaskControlBlock {
                name: "bootstrap",
                state: TaskState::Running,
                rsp: 0,
                _stack: None,
                cap_table: CapTable::new(),
            },
        );
        *slot = Some(SchedulerState { tasks, ready_queue: VecDeque::new(), current: 0, next_id: 1, tick_count: 0 });
    });
    INITIALIZED.store(true, Ordering::SeqCst);
    crate::ktrace::log("sched", "scheduler initialized, current context is bootstrap task 0");
}

/// Spawn a new task running `entry(arg)` on a fresh 64 KiB stack. The
/// task starts `Ready` and joins the round-robin queue; it does not run
/// until some task (or the timer) yields to it.
pub fn spawn(name: &'static str, entry: extern "C" fn(u64), arg: u64) -> TaskId {
    let mut stack = alloc::vec![0u8; STACK_SIZE].into_boxed_slice();
    let stack_top = (stack.as_mut_ptr() as u64 + STACK_SIZE as u64) & !0xf;
    // SAFETY: `stack` is a freshly allocated, exclusively-owned 64 KiB
    // region; `stack_top` is 16-byte aligned and well within it.
    let rsp = unsafe { context::init_stack(stack_top, entry, arg) };

    let id = SCHED.with(|slot| {
        let s = slot.as_mut().expect("sched::spawn: scheduler not initialized");
        let id = s.next_id;
        s.next_id += 1;
        s.tasks.insert(
            id,
            TaskControlBlock { name, state: TaskState::Ready, rsp, _stack: Some(stack), cap_table: CapTable::new() },
        );
        s.ready_queue.push_back(id);
        id
    });
    crate::ktrace::log_fmt(format_args!("sched: spawned task {id} ({name})"));
    id
}

pub fn current_task_id() -> TaskId {
    SCHED.with(|slot| slot.as_ref().expect("sched not initialized").current)
}

pub(crate) fn with_cap_table_mut<R>(task: TaskId, f: impl FnOnce(&mut CapTable) -> R) -> R {
    SCHED.with(|slot| {
        let s = slot.as_mut().expect("sched not initialized");
        let tcb = s.tasks.get_mut(&task).expect("sched: unknown task id");
        f(&mut tcb.cap_table)
    })
}

/// Voluntarily give up the CPU. If another task is ready, round-robins to
/// it; if not, returns immediately.
///
/// Safe to call both from ordinary task code (interrupts enabled) and
/// from inside the timer IRQ handler (interrupts already disabled by the
/// interrupt gate): `context::switch_to` saves/restores `RFLAGS` as part
/// of each task's own context, and wrapping the whole operation in
/// `without_interrupts` means it only re-enables interrupts here if they
/// were enabled *at this call site* -- so a preemptive switch triggered
/// from `on_timer_tick` leaves interrupts off until the resumed task's
/// own pending `iretq` (further up its call stack) restores its true
/// pre-interrupt flags, exactly as if `yield_now` had never run.
pub fn yield_now() {
    interrupts::without_interrupts(|| {
        let switch = SCHED.with(|slot| {
            let s = slot.as_mut().expect("sched not initialized");
            let next_id = s.ready_queue.pop_front()?;
            let current_id = s.current;
            if let Some(cur) = s.tasks.get(&current_id) {
                if cur.state != TaskState::Dead {
                    s.ready_queue.push_back(current_id);
                }
            }
            s.current = next_id;
            let old_rsp_ptr = &mut s.tasks.get_mut(&current_id).unwrap().rsp as *mut u64;
            let new_rsp = s.tasks.get(&next_id).unwrap().rsp;
            Some((old_rsp_ptr, new_rsp))
        });
        if let Some((old_rsp_ptr, new_rsp)) = switch {
            // SAFETY: `old_rsp_ptr` points at this task's own saved-rsp
            // slot in the scheduler's task table (not concurrently
            // accessed: interrupts are off and there is only one core);
            // `new_rsp` was produced either by `context::init_stack` for
            // a never-run task or by a previous `switch_to` save for a
            // previously-descheduled one -- exactly what `switch_to`
            // requires.
            unsafe { context::switch_to(old_rsp_ptr, new_rsp) };
        }
    });
}

/// Called from `pit::timer_handler` on every tick. Forces a `yield_now`
/// once the current task's slice of ticks has elapsed -- the
/// timer-preemptive half of "cooperative first, then timer-preemptive."
pub fn on_timer_tick() {
    if !INITIALIZED.load(Ordering::SeqCst) {
        return;
    }
    let due = SCHED.with(|slot| {
        let s = slot.as_mut().expect("sched not initialized");
        s.tick_count += 1;
        s.tick_count % TIME_SLICE_TICKS == 0 && !s.ready_queue.is_empty()
    });
    if due {
        yield_now();
    }
}

/// Marks the current task `Dead` and switches away for the last time.
/// This is what every task's `context::trampoline` calls once its entry
/// function returns; it never returns to its own caller.
pub extern "C" fn exit_current_task() -> ! {
    interrupts::disable();
    let next_rsp = SCHED.with(|slot| {
        let s = slot.as_mut().expect("sched not initialized");
        let dead_id = s.current;
        s.tasks.get_mut(&dead_id).unwrap().state = TaskState::Dead;
        s.ready_queue.pop_front().map(|next_id| {
            s.current = next_id;
            s.tasks.get(&next_id).unwrap().rsp
        })
    });
    crate::ktrace::log("sched", "task exited");
    match next_rsp {
        Some(new_rsp) => {
            let mut discard: u64 = 0;
            // SAFETY: this task is now `Dead` and no longer in the ready
            // queue, so its stack (whose top the discarded old rsp would
            // have pointed into) will never be switched back to.
            unsafe { context::switch_to(&mut discard as *mut u64, new_rsp) };
            unreachable!("a dead task's stack must never be resumed");
        }
        None => loop {
            crate::arch::x86_64::hlt();
        },
    }
}
