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

use crate::arch::interrupts;
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
    /// Per-task x87/SSE save area. `Box`ed so its address is stable
    /// regardless of the `BTreeMap` moving the `TaskControlBlock` around
    /// -- `yield_now` captures a raw pointer to it and uses that pointer
    /// across the actual context switch.
    fx_area: Box<context::FxArea>,
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

// 256 KiB per task. The ONNX executor's op dispatch (`onnx::exec::exec_graph`)
// is one function with a ~55-arm match, whose *debug* stack frame alone runs
// ~64 KiB (no stack-slot coalescing); a deep model graph plus a couple levels
// of If/Loop subgraph recursion overflowed the old 64 KiB stack (a silent
// triple fault). 256 KiB gives the interpreter — and the LLM forward — comfortable
// headroom; release frames are far smaller.
const STACK_SIZE: usize = 256 * 1024;

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
                fx_area: Box::new(context::FxArea::new()),
                cap_table: CapTable::new(),
            },
        );
        *slot = Some(SchedulerState { tasks, ready_queue: VecDeque::new(), current: 0, next_id: 1, tick_count: 0 });
    });
    INITIALIZED.store(true, Ordering::SeqCst);
    crate::ktrace::log("sched", "scheduler initialized, current context is bootstrap task 0");
}

/// Idle entry for a parked task: it should never actually be scheduled (see
/// [`spawn_parked`]), but if it ever is, it just yields forever rather than
/// falling off the end of its stack.
extern "C" fn park_forever(_arg: u64) {
    loop {
        yield_now();
    }
}

/// Create a task that exists only to *own a capability table* — an agent's
/// identity holder — without ever running. Unlike [`spawn`], it is NOT pushed
/// onto the ready queue, so it never consumes a scheduler turn or interferes
/// with other tasks' cooperative hand-off. Its cap table is live immediately
/// (grants/lookups work), which is all the agent layer needs: the agentic loop
/// runs on the foreground task and only *names* this task as the caller whose
/// authority Synapse checks. Returns the new task id.
pub fn spawn_parked(name: &'static str) -> TaskId {
    let mut stack = alloc::vec![0u8; STACK_SIZE].into_boxed_slice();
    let stack_top = (stack.as_mut_ptr() as u64 + STACK_SIZE as u64) & !0xf;
    // SAFETY: freshly allocated, exclusively-owned 64 KiB region; 16-byte
    // aligned `stack_top` well within it. The task is never scheduled, so the
    // context is only a placeholder, but we set it up correctly anyway.
    let rsp = unsafe { context::init_stack(stack_top, park_forever, 0) };
    let id = SCHED.with(|slot| {
        let s = slot.as_mut().expect("sched::spawn_parked: scheduler not initialized");
        let id = s.next_id;
        s.next_id += 1;
        s.tasks.insert(
            id,
            TaskControlBlock {
                name,
                state: TaskState::Ready,
                rsp,
                _stack: Some(stack),
                fx_area: Box::new(context::FxArea::new()),
                cap_table: CapTable::new(),
            },
        );
        // Deliberately NOT enqueued — a cap-owning identity holder, never run.
        id
    });
    crate::ktrace::log_fmt(format_args!("sched: spawned parked task {id} ({name}) [cap-owner, not scheduled]"));
    id
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
            TaskControlBlock {
                name,
                state: TaskState::Ready,
                rsp,
                _stack: Some(stack),
                fx_area: Box::new(context::FxArea::new()),
                cap_table: CapTable::new(),
            },
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

/// Whether task `id` exists and is not `Dead`. Used by the service supervisor
/// to decide whether a daemon needs restarting.
pub fn is_alive(id: TaskId) -> bool {
    SCHED.with(|slot| match slot.as_ref() {
        Some(s) => s.tasks.get(&id).map(|t| t.state != TaskState::Dead).unwrap_or(false),
        None => false,
    })
}

/// Snapshot of the task table for the shell's `/agents` process list:
/// `(id, name, state)`. Parked capability-owner tasks (agents) show as
/// "parked" — Ready but never enqueued.
pub fn list() -> alloc::vec::Vec<(TaskId, &'static str, &'static str)> {
    SCHED.with(|slot| {
        let s = match slot.as_ref() {
            Some(s) => s,
            None => return alloc::vec::Vec::new(),
        };
        s.tasks
            .iter()
            .map(|(&id, tcb)| {
                let state = match tcb.state {
                    TaskState::Running => "running",
                    TaskState::Dead => "dead",
                    TaskState::Ready if s.ready_queue.contains(&id) => "ready",
                    TaskState::Ready => "parked",
                };
                (id, tcb.name, state)
            })
            .collect()
    })
}

/// Terminate task `id`: mark it Dead, drop it from the ready queue, and drop
/// its capability table (all its authority is revoked). Refuses the bootstrap
/// task and the currently running task. The stack is reclaimed by the dead-
/// task policy, same as a normal exit.
pub fn kill(id: TaskId) -> Result<(), &'static str> {
    SCHED.with(|slot| {
        let s = slot.as_mut().ok_or("scheduler not initialized")?;
        if id == 0 {
            return Err("refusing to kill the bootstrap task");
        }
        if id == s.current {
            return Err("refusing to kill the current task");
        }
        let tcb = s.tasks.get_mut(&id).ok_or("no such task")?;
        if tcb.state == TaskState::Dead {
            return Err("already dead");
        }
        tcb.state = TaskState::Dead;
        tcb.cap_table = CapTable::new(); // revoke all authority
        s.ready_queue.retain(|&t| t != id);
        Ok(())
    })?;
    crate::cap::clear_scopes(id); // drop the fine-grained scope ledger too
    crate::ktrace::log_fmt(format_args!("sched: task {id} killed"));
    Ok(())
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
            let cur = s.tasks.get_mut(&current_id).unwrap();
            let old_rsp_ptr: *mut u64 = &mut cur.rsp;
            // Stable heap address of this task's FX area (see the field
            // doc): valid across the switch and on resume.
            let fx_ptr: *mut context::FxArea = &mut *cur.fx_area;
            let new_rsp = s.tasks.get(&next_id).unwrap().rsp;
            Some((old_rsp_ptr, new_rsp, fx_ptr))
        });
        if let Some((old_rsp_ptr, new_rsp, fx_ptr)) = switch {
            // SAFETY: `fx_ptr`/`old_rsp_ptr` point at this task's own,
            // exclusively-owned FX area and saved-rsp slot (interrupts
            // off, single core -- no concurrent access); `new_rsp` was
            // produced either by `context::init_stack` for a never-run
            // task or by a previous `switch_to` save. We `fxsave` this
            // task's SSE state before switching away and `fxrstor` it on
            // resume: `fx_ptr` was captured *before* the switch, so after
            // `switch_to` returns (this same task, resumed later) it
            // still names this task's area -- the incoming task restores
            // its own state symmetrically inside its own `yield_now`.
            unsafe {
                context::save_fpu(fx_ptr);
                context::switch_to(old_rsp_ptr, new_rsp);
                context::restore_fpu(fx_ptr);
            }
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
            crate::arch::hlt();
        },
    }
}
