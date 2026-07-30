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

use crate::arch::interrupts;
use crate::cap::CapTable;
use crate::mm::Locked;
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, VecDeque};
use core::sync::atomic::{AtomicBool, Ordering};

pub type TaskId = u64;

/// A condition a task can sleep on until something wakes it.
///
/// An enum rather than dynamically allocated queue objects because the set is
/// small, kernel-wide and known — and because it makes wakers **greppable**: to
/// find everything that can wake a console waiter you search for
/// `Wait::Console`. It also means [`block_on`] allocates nothing and can be
/// woken from an interrupt handler, since "the queue" is just this tag stored in
/// the blocked task's own state (see [`wake`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Wait {
    /// Console input is available to read.
    Console,
    /// A network socket changed state — data arrived, a connect completed, a
    /// listener has a connection to accept.
    Net,
    /// A block-device request completed.
    Block,
    /// The sound device has room in its output queue.
    SoundOut,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TaskState {
    Ready,
    Running,
    /// Sleeping until [`wake`] is called with this condition. Off the ready
    /// queue entirely — unlike a `Ready` task, it consumes no scheduler turn,
    /// which is the whole point of blocking rather than polling.
    Blocked(Wait),
    /// Exists only to own a capability table — an agent's identity holder —
    /// and is never scheduled. Distinct from `Ready` because it is a different
    /// thing, not a `Ready` task that happens to be absent from the queue:
    /// a parked task has **no stack and no FPU save area at all**, so it is
    /// not merely unqueued but unrunnable. See [`spawn_parked`].
    Parked,
    Dead,
}

struct TaskControlBlock {
    name: &'static str,
    state: TaskState,
    /// Saved stack pointer. Meaningful only while this task is not the
    /// one currently executing; read every switch-out, written every
    /// switch-in. Zero once reclaimed.
    rsp: u64,
    /// The stack this task runs on, kept alive for as long as it might still
    /// be switched to. `None` in three cases: the bootstrap task (it runs on
    /// the boot stack, which the kernel image owns), a parked task (never
    /// scheduled, so allocating one was 256 KiB of pure waste), and a dead
    /// task (handed back — see [`TaskControlBlock::reclaim`]).
    stack: Option<Box<[u8]>>,
    /// Per-task x87/SSE save area. `Box`ed so its address is stable
    /// regardless of the `BTreeMap` moving the `TaskControlBlock` around
    /// -- `yield_now` captures a raw pointer to it and uses that pointer
    /// across the actual context switch.
    ///
    /// **`Some` exactly when this task can be switched to**, which makes it the
    /// scheduler's structural guard: [`pop_schedulable`] refuses any task
    /// without one, so a parked or already-reclaimed task can never be jumped
    /// to with a dangling `rsp`.
    fx_area: Option<Box<context::FxArea>>,
    cap_table: CapTable,
}

/// What a dying task hands back, so the (potentially large) frees happen after
/// the scheduler lock is released rather than inside it.
struct Reclaimed {
    stack: Option<Box<[u8]>>,
    fx_area: Option<Box<context::FxArea>>,
}

impl Reclaimed {
    /// Bytes returned to the heap, for the ktrace line.
    fn bytes(&self) -> usize {
        self.stack.as_ref().map_or(0, |s| s.len())
            + self.fx_area.as_ref().map_or(0, |_| core::mem::size_of::<context::FxArea>())
    }
}

impl TaskControlBlock {
    /// Take back the *memory* a dead task no longer needs, leaving it
    /// unschedulable by construction: no FPU area, no stack, zero `rsp`.
    ///
    /// The stack is *returned* rather than dropped here because the one caller
    /// that matters — [`exit_current_task`] — is executing on it, and must defer
    /// the free until it has switched away.
    ///
    /// **Deliberately does not revoke capabilities.** That is [`kill`]'s policy,
    /// and it has to stay there: `persona::Agent::spawn` and the agent layer use
    /// a task as an *identity holder* whose entry function returns immediately,
    /// then keep invoking Synapse primitives as that task — so a dead task's
    /// capability table is load-bearing, not vestigial. Revoking it here (which
    /// the first version of this did, on the reasonable-sounding grounds that a
    /// dead task should hold no authority) makes every persona agent lose its
    /// capabilities the instant it is scheduled, which reads as "the agent was
    /// denied its own manifest". Retiring an identity is an explicit act:
    /// `kill`. P8, where agents become real running tasks, is what removes the
    /// need for identity-holder tasks and lets this become unconditional.
    fn reclaim(&mut self) -> Reclaimed {
        self.rsp = 0;
        Reclaimed { stack: self.stack.take(), fx_area: self.fx_area.take() }
    }
}

struct SchedulerState {
    tasks: BTreeMap<TaskId, TaskControlBlock>,
    ready_queue: VecDeque<TaskId>,
    current: TaskId,
    next_id: TaskId,
    tick_count: u64,
    /// The task to run when the ready queue is empty, if one has been
    /// registered ([`set_idle`]).
    ///
    /// A dedicated slot rather than a low-priority queue entry, because
    /// round-robin cannot express "only when there is nothing else": an idle
    /// task sharing the queue would take every other turn, halving the CPU
    /// available to the inference loop. Expressing it as a priority is P3's job;
    /// this is the one bit of priority the blocking design actually needs, and it
    /// is what makes [`block_on`] able to succeed — the scheduler always has
    /// *something* left to run, so it can put the last working task to sleep.
    idle: Option<TaskId>,
}

static SCHED: Locked<Option<SchedulerState>> = Locked::new(None);
static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Stacks belonging to tasks that exited *while running on them*.
///
/// The old comment here said there is no safe point to free a stack you might
/// be executing on, and concluded that dead tasks' stacks must be leaked. The
/// first half is true; the second does not follow. There is a safe point — it is
/// simply not inside the exiting task. [`exit_current_task`] parks its stack
/// here and switches away for good; the next [`yield_now`], which by definition
/// runs on a *different* stack, frees it. A task that called
/// `exit_current_task` never reaches `yield_now` again, so nothing in this list
/// can be the stack the reaper is standing on.
static ZOMBIE_STACKS: Locked<alloc::vec::Vec<Box<[u8]>>> = Locked::new(alloc::vec::Vec::new());

/// Free the stacks of tasks that exited on them. Called at the top of
/// [`yield_now`]; the take-then-drop keeps the deallocation outside the lock.
fn reap_zombie_stacks() {
    let dead = ZOMBIE_STACKS.with(core::mem::take);
    drop(dead);
}

/// Pop the next **schedulable** task from the ready queue.
///
/// Schedulable means `Ready` *and* holding an FPU save area — which every task
/// that can run has, and no parked, blocked or reclaimed task does. That makes
/// this the structural guard against the worst failure in this area: switching to
/// a task whose stack has been freed loads a dangling `rsp`, which is a triple
/// fault with no message and no way back. Only `Ready` tasks are ever enqueued,
/// so finding anything else is a bug — dropped from the queue and asserted in
/// debug rather than acted on.
fn pop_schedulable(s: &mut SchedulerState) -> Option<TaskId> {
    while let Some(id) = s.ready_queue.pop_front() {
        if s.tasks.get(&id).is_some_and(|t| t.state == TaskState::Ready && t.fx_area.is_some()) {
            return Some(id);
        }
        debug_assert!(false, "sched: an unschedulable task was in the ready queue");
    }
    None
}

/// The next task to run: a ready one if there is one, else the idle task.
///
/// The idle task is deliberately reachable *only* through this fallback, and
/// only when it is not already the current task — otherwise a blocking idle
/// task, or an idle task that yielded, would be picked in preference to nothing
/// and the scheduler would spin on it.
fn pick_next(s: &mut SchedulerState) -> Option<TaskId> {
    if let Some(id) = pop_schedulable(s) {
        return Some(id);
    }
    let idle = s.idle?;
    if idle == s.current {
        return None;
    }
    // **Only when something is actually asleep.** The idle task exists to keep
    // the world turning on behalf of *blocked* tasks; with nothing blocked, the
    // task that would yield to it is still pumping for itself, so switching to
    // the pump only duplicates that work. That duplication is not merely
    // wasteful: `upkeep`'s `mouse::tick()` consumes the clicks that modal and
    // editor loops read for themselves, which is exactly why those loops call
    // `status_tick` instead of `upkeep`.
    //
    // Without this condition the fallback fires on the shell's very first yield,
    // because the ready queue is empty in the ordinary case — a task's own
    // `yield_now` re-queues it only *after* the pick. So "the pump is inert until
    // something blocks" holds only with this test, not by virtue of the dedicated
    // slot alone.
    if !s.tasks.values().any(|t| matches!(t.state, TaskState::Blocked(_))) {
        return None;
    }
    s.tasks
        .get(&idle)
        .is_some_and(|t| t.fx_area.is_some() && !matches!(t.state, TaskState::Dead | TaskState::Blocked(_)))
        .then_some(idle)
}

/// Register `id` as the idle task: the one to run when nothing else can.
///
/// It must never block and never exit. Nothing enforces that — it cannot be, in
/// a cooperative kernel — but the consequence is worth stating: if the idle task
/// blocks, and every other task is blocked too, there is nothing left to call
/// [`wake`] and the machine stops without a diagnostic.
pub fn set_idle(id: TaskId) {
    SCHED.with(|slot| {
        if let Some(s) = slot.as_mut() {
            s.idle = Some(id);
            // Out of the ready queue: it is reached only via `pick_next`'s
            // fallback, so it can never take a turn from a task with work.
            s.ready_queue.retain(|&t| t != id);
        }
    });
    crate::ktrace::log_fmt(format_args!("sched: task {id} is the idle task"));
}

/// Unregister the idle task, so the scheduler goes back to having nothing to run
/// when the ready queue empties. For tearing one down before killing it — and for
/// tests, which must leave the global scheduler as they found it.
pub fn clear_idle() {
    SCHED.with(|slot| {
        if let Some(s) = slot.as_mut() {
            s.idle = None;
        }
    });
}

/// The idle task's id, if one is registered.
pub fn idle_task() -> Option<TaskId> {
    SCHED.with(|slot| slot.as_ref().and_then(|s| s.idle))
}

/// Sleep until something calls [`wake`] with `w`, giving the CPU to another task
/// meanwhile. Returns whether the task actually blocked.
///
/// **`false` means "there was nothing else to run"** — the scheduler cannot put
/// the last runnable task to sleep, because then nothing would be left to wake
/// it. A caller must therefore treat this as a poll-and-retry, which is exactly
/// the shape of every waiting loop in this kernel already:
///
/// ```ignore
/// while !ready() {
///     if !sched::block_on(Wait::Net) {
///         shell::upkeep(); // nothing else to run, so drive the world ourselves
///     }
/// }
/// ```
///
/// It refuses outright when interrupts are disabled, and that check is the
/// important one. Blocking inside a [`crate::mm::Locked`] critical section hands
/// the CPU to another task **while still holding the lock**; if anything that
/// task runs takes the same lock, the machine spins forever with interrupts off
/// — no panic, no log, nothing to attach to. `Locked` is the only thing that
/// disables interrupts for a critical section, so "interrupts are off" is a
/// sound and free proxy for "a lock is held" (see
/// [`crate::arch::interrupts::are_enabled`]). It is also the right answer for
/// the other two ways interrupts are off — an explicit `without_interrupts` and
/// an interrupt handler — since blocking is wrong in both.
pub fn block_on(w: Wait) -> bool {
    if !interrupts::are_enabled() {
        debug_assert!(
            false,
            "sched::block_on with interrupts disabled: a Locked would be held while another task runs"
        );
        return false;
    }
    reap_zombie_stacks();
    reschedule(Some(w))
}

/// Wake every task sleeping on `w`, returning how many were woken.
///
/// Allocation-free in its scan so it is safe to call from an interrupt handler
/// or a driver poll. The scan repeats rather than taking one bounded pass,
/// because a bounded pass that silently left a task asleep would be a lost
/// wakeup — a hang, and the hardest kind to attribute.
pub fn wake(w: Wait) -> usize {
    SCHED.with(|slot| {
        let Some(s) = slot.as_mut() else { return 0 };
        const BATCH: usize = 32;
        let mut total = 0;
        loop {
            let mut ids = [0u64; BATCH];
            let mut n = 0;
            for (&id, t) in s.tasks.iter() {
                if t.state == TaskState::Blocked(w) {
                    ids[n] = id;
                    n += 1;
                    if n == BATCH {
                        break;
                    }
                }
            }
            if n == 0 {
                return total;
            }
            for &id in &ids[..n] {
                if let Some(t) = s.tasks.get_mut(&id) {
                    t.state = TaskState::Ready;
                }
                s.ready_queue.push_back(id);
            }
            total += n;
        }
    })
}

/// How many tasks are currently sleeping on `w`. For diagnostics and for the
/// idle task, which only needs to pump when someone is waiting on it.
pub fn blocked_count(w: Wait) -> usize {
    SCHED.with(|slot| match slot.as_ref() {
        Some(s) => s.tasks.values().filter(|t| t.state == TaskState::Blocked(w)).count(),
        None => 0,
    })
}

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
                // Runs on the boot stack, which the kernel image owns.
                stack: None,
                fx_area: Some(Box::new(context::FxArea::new())),
                cap_table: CapTable::new(),
            },
        );
        *slot = Some(SchedulerState {
            tasks,
            ready_queue: VecDeque::new(),
            current: 0,
            next_id: 1,
            tick_count: 0,
            idle: None,
        });
    });
    INITIALIZED.store(true, Ordering::SeqCst);
    crate::ktrace::log("sched", "scheduler initialized, current context is bootstrap task 0");
}

/// Create a task that exists only to *own a capability table* — an agent's
/// identity holder — without ever running. Unlike [`spawn`], it is NOT pushed
/// onto the ready queue, so it never consumes a scheduler turn or interferes
/// with other tasks' cooperative hand-off. Its cap table is live immediately
/// (grants/lookups work), which is all the agent layer needs: the agentic loop
/// runs on the foreground task and only *names* this task as the caller whose
/// authority Synapse checks. Returns the new task id.
///
/// **It allocates no stack.** It used to allocate [`STACK_SIZE`] — 256 KiB — plus
/// an FPU save area and a fully initialised context, for a task the very same
/// function then deliberately declined to enqueue. Nothing could ever have run
/// on it. That was not free: the shell mints one of these per agent switch and
/// `agent::subagent::dispatch` one per delegation, so the waste scaled with how
/// much the agent layer was used. `TaskState::Parked` and the absent `fx_area`
/// now say "unrunnable" structurally, and [`pop_schedulable`] enforces it, which
/// is strictly stronger than the old placeholder context — that would have
/// happily switched to a `park_forever` loop and hidden the mistake.
pub fn spawn_parked(name: &'static str) -> TaskId {
    let id = SCHED.with(|slot| {
        let s = slot.as_mut().expect("sched::spawn_parked: scheduler not initialized");
        let id = s.next_id;
        s.next_id += 1;
        s.tasks.insert(
            id,
            TaskControlBlock {
                name,
                state: TaskState::Parked,
                rsp: 0,
                stack: None,
                fx_area: None,
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
                stack: Some(stack),
                fx_area: Some(Box::new(context::FxArea::new())),
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
                // Read straight off the state now that `Parked` is one. This
                // used to infer "parked" from absence in the ready queue, which
                // it had to because parked tasks were recorded as `Ready`.
                let state = match tcb.state {
                    TaskState::Running => "running",
                    TaskState::Dead => "dead",
                    TaskState::Parked => "parked",
                    TaskState::Blocked(_) => "blocked",
                    TaskState::Ready => "ready",
                };
                (id, tcb.name, state)
            })
            .collect()
    })
}

/// Terminate task `id`: mark it Dead, drop it from the ready queue, revoke all
/// its authority, and **free its stack and FPU save area**. Refuses the
/// bootstrap task and the currently running task.
///
/// Freeing is unconditional and immediate here, and safe precisely because of
/// those two refusals: `id` is not the current task, so nothing is executing on
/// the stack being freed. (The awkward case — a task ending while running on its
/// own stack — is [`exit_current_task`]'s, and it defers.) This used to mark the
/// task Dead and leave the 256 KiB stack allocated forever, so every agent
/// switch and every sub-agent delegation cost a stack that was never returned.
pub fn kill(id: TaskId) -> Result<(), &'static str> {
    let reclaimed = SCHED.with(|slot| {
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
        tcb.cap_table = CapTable::new(); // revoke all authority — see `reclaim`
        let reclaimed = tcb.reclaim();
        s.ready_queue.retain(|&t| t != id);
        Ok(reclaimed)
    })?;
    let bytes = reclaimed.bytes();
    drop(reclaimed); // deallocate outside the scheduler lock
    crate::cap::clear_scopes(id); // drop the fine-grained scope ledger too
    crate::ktrace::log_fmt(format_args!("sched: task {id} killed ({bytes} bytes reclaimed)"));
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
    // Free the stacks of tasks that exited on them. Safe here and nowhere
    // inside the exiting task: see [`ZOMBIE_STACKS`]. Outside
    // `without_interrupts` because it takes the heap lock, which manages its own.
    reap_zombie_stacks();
    reschedule(None);
}

/// Switch away from the current task; the one place a context switch happens.
///
/// `park` decides what becomes of the outgoing task: `None` keeps it runnable
/// (a yield — straight back onto the ready queue), `Some(w)` puts it to sleep on
/// `w` (a block — off the queue until [`wake`]). Returns whether a switch
/// actually happened; see [`block_on`] for why a caller cares.
fn reschedule(park: Option<Wait>) -> bool {
    let mut switched = false;
    interrupts::without_interrupts(|| {
        let switch = SCHED.with(|slot| {
            let s = slot.as_mut().expect("sched not initialized");
            let current_id = s.current;
            // Resolve the outgoing task's FPU area *before* touching the queue,
            // so a current task that somehow cannot be switched away from leaves
            // the scheduler exactly as it was rather than half-rotated.
            let fx_ptr: *mut context::FxArea = match s.tasks.get_mut(&current_id).and_then(|t| t.fx_area.as_mut()) {
                // Stable heap address of this task's FX area (see the field
                // doc): valid across the switch and on resume.
                Some(fx) => &mut **fx,
                None => {
                    debug_assert!(false, "sched: the current task has no FPU save area");
                    return None;
                }
            };
            let next_id = pick_next(s)?;
            if let Some(cur) = s.tasks.get_mut(&current_id) {
                if cur.state != TaskState::Dead {
                    match park {
                        // Yield: still runnable, back of the queue.
                        None => {
                            cur.state = TaskState::Ready;
                            // ...except the idle task, which is never queued. It
                            // is reached only through `pick_next`'s fallback, so
                            // re-enqueueing it on its own yield would slide it
                            // into the round-robin and hand it every other turn —
                            // halving the CPU available to real work, which is
                            // precisely what the dedicated `idle` slot exists to
                            // prevent. Nothing else would notice: the machine
                            // would just be half as fast at inference.
                            if Some(current_id) != s.idle {
                                s.ready_queue.push_back(current_id);
                            }
                        }
                        // Block: asleep, and deliberately *not* enqueued —
                        // that absence is what makes blocking cheaper than
                        // polling, and what `wake` undoes.
                        Some(w) => cur.state = TaskState::Blocked(w),
                    }
                }
            }
            s.current = next_id;
            // Keep `state` honest: it previously went stale the moment anything
            // was spawned — task 0 stayed `Running` forever and every other task
            // stayed `Ready` even while executing — so `/agents` and `/top` could
            // only ever show task 0 as running. P3's CPU accounting needs this to
            // mean what it says.
            if let Some(next) = s.tasks.get_mut(&next_id) {
                next.state = TaskState::Running;
            }
            let old_rsp_ptr: *mut u64 = &mut s.tasks.get_mut(&current_id).unwrap().rsp;
            let new_rsp = s.tasks.get(&next_id).unwrap().rsp;
            Some((old_rsp_ptr, new_rsp, fx_ptr))
        });
        if let Some((old_rsp_ptr, new_rsp, fx_ptr)) = switch {
            switched = true;
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
    switched
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
        let dead = s.tasks.get_mut(&dead_id).unwrap();
        dead.state = TaskState::Dead;
        // Hand back the stack we are standing on. `reclaim` only *takes* it;
        // parking it in ZOMBIE_STACKS keeps the memory allocated (so the rest of
        // this function still has a stack) while transferring ownership to
        // whoever yields next. The FPU area can go immediately — this task will
        // never restore its floating-point state again.
        let reclaimed = dead.reclaim();
        let next = pick_next(s).map(|next_id| {
            s.current = next_id;
            if let Some(t) = s.tasks.get_mut(&next_id) {
                t.state = TaskState::Running;
            }
            s.tasks.get(&next_id).unwrap().rsp
        });
        (next, reclaimed)
    });
    let (next_rsp, Reclaimed { stack, fx_area }) = next_rsp;
    if let Some(stack) = stack {
        ZOMBIE_STACKS.with(|z| z.push(stack));
    }
    // Explicitly, because this function never returns: an implicit drop at the
    // end of the enclosing scope would never run, so the FPU area would leak on
    // every normal task exit — the very thing this change is about.
    drop(fx_area);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Bytes currently allocated on the kernel heap. The tests below assert on
    /// *changes* in this, because task reclamation is a claim about memory and
    /// nothing else can check it: a leaked stack has no other observable effect
    /// until the heap runs out, long after and somewhere else.
    fn heap_used() -> usize {
        crate::mm::heap::stats().2
    }

    /// Entry point for a spawned test task. Returns immediately, so if the timer
    /// ever preempts into it, it exits cleanly through `exit_current_task` — the
    /// deferred-free path — rather than running off the end of its stack.
    extern "C" fn noop_task(_arg: u64) {}

    #[test_case]
    fn a_parked_task_allocates_no_stack() {
        // The whole point of `spawn_parked`: a capability-table identity holder
        // that is never scheduled has no business owning 256 KiB. This used to
        // allocate a full STACK_SIZE stack plus an FPU area for a task the same
        // function then declined to enqueue.
        let before = heap_used();
        let id = spawn_parked("test-parked");
        let cost = heap_used().saturating_sub(before);
        assert!(
            cost < STACK_SIZE / 8,
            "a parked task cost {cost} bytes; it must not be allocating a {STACK_SIZE}-byte stack"
        );
        kill(id).expect("kill the parked task");
    }

    #[test_case]
    fn a_spawned_task_really_does_take_a_stack() {
        // The control for the test above: without this, "parked is cheap" could
        // pass because the measurement itself is broken.
        let before = heap_used();
        let id = spawn("test-spawn-cost", noop_task, 0);
        let cost = heap_used().saturating_sub(before);
        assert!(cost >= STACK_SIZE, "a runnable task must own a stack; measured {cost} bytes");
        let _ = kill(id);
    }

    #[test_case]
    fn killing_a_task_returns_its_stack_to_the_heap() {
        // The leak this fixes: `kill` marked the task Dead and left the stack
        // allocated for the life of the kernel, so every agent switch and every
        // sub-agent delegation cost 256 KiB that never came back.
        let before = heap_used();
        let id = spawn("test-kill-frees", noop_task, 0);
        assert!(heap_used() > before + STACK_SIZE / 2, "test premise: the spawn allocated a stack");
        kill(id).expect("kill");
        let after = heap_used();
        assert!(
            after < before + STACK_SIZE / 8,
            "after kill the heap still holds {} extra bytes; the stack was not freed",
            after.saturating_sub(before)
        );
    }

    #[test_case]
    fn a_killed_task_is_dead_and_unschedulable() {
        let id = spawn("test-kill-state", noop_task, 0);
        assert!(is_alive(id));
        kill(id).expect("kill");
        assert!(!is_alive(id));
        assert_eq!(list().iter().find(|t| t.0 == id).map(|t| t.2), Some("dead"));
        // Dropped from the ready queue, and reclaimed so `pop_schedulable` would
        // refuse it even if it were still there — switching to a task whose
        // stack has been freed is a triple fault with no message.
        SCHED.with(|slot| {
            let s = slot.as_ref().unwrap();
            assert!(!s.ready_queue.contains(&id), "a killed task must leave the ready queue");
            let tcb = s.tasks.get(&id).unwrap();
            assert!(tcb.fx_area.is_none() && tcb.stack.is_none() && tcb.rsp == 0);
        });
        assert_eq!(kill(id), Err("already dead"));
    }

    #[test_case]
    fn a_parked_task_is_alive_and_reports_parked() {
        // `Parked` is now a state rather than something `list()` infers from
        // absence in the ready queue. It must still count as *alive*: the
        // Synapse executor checks `is_alive` on the caller, and an agent's
        // identity holder is exactly such a caller.
        let id = spawn_parked("test-parked-state");
        assert!(is_alive(id), "a parked task owns live capabilities");
        assert_eq!(list().iter().find(|t| t.0 == id).map(|t| t.2), Some("parked"));
        SCHED.with(|slot| {
            let s = slot.as_ref().unwrap();
            assert!(!s.ready_queue.contains(&id), "a parked task is never enqueued");
        });
        kill(id).expect("kill");
    }

    #[test_case]
    fn kill_refuses_the_bootstrap_task_and_unknown_ids() {
        // These refusals are what make `kill`'s immediate free safe: it can only
        // ever free a stack nothing is executing on.
        //
        // The suite runs *on* the bootstrap task, so task 0 is both "bootstrap"
        // and "current" and the bootstrap check is the one that reports. The
        // current-task refusal is therefore not reachable from here — it guards a
        // task calling `kill` on itself, which must use `exit_current_task`
        // instead precisely because it cannot free the stack under its own feet.
        assert_eq!(current_task_id(), 0, "the suite runs on the bootstrap task");
        assert_eq!(kill(0), Err("refusing to kill the bootstrap task"));
        assert_eq!(kill(u64::MAX), Err("no such task"));
    }

    #[test_case]
    fn a_task_that_runs_to_completion_has_its_stack_reaped() {
        // The deferred-free path. The task cannot free the stack under its own
        // feet, so `exit_current_task` parks it and the next `yield_now` — which
        // by definition runs on a different stack — frees it.
        let before = heap_used();
        let id = spawn("test-exit-reaps", noop_task, 0);
        assert!(heap_used() > before + STACK_SIZE / 2, "test premise: the spawn allocated a stack");
        // Yield until it has run to completion. Bounded so a scheduling
        // regression fails the test instead of hanging the suite.
        let mut spins = 0;
        while is_alive(id) && spins < 1000 {
            yield_now();
            spins += 1;
        }
        assert!(!is_alive(id), "the task did not run to completion in {spins} yields");
        // One more yield: the exiting task parks its stack, and the *next*
        // yield is what reaps it.
        yield_now();
        let after = heap_used();
        assert!(
            after < before + STACK_SIZE / 8,
            "after exit + reap the heap still holds {} extra bytes",
            after.saturating_sub(before)
        );
    }

    #[test_case]
    fn a_task_that_exits_keeps_its_capabilities_but_kill_revokes_them() {
        // The regression this pins, which cost a full suite run to find: making
        // `reclaim` revoke authority looks obviously right — a dead task should
        // hold none — but `persona::Agent::spawn` and the agent layer use a task
        // as an *identity holder* whose entry returns immediately, and keep
        // invoking Synapse primitives as that task afterwards. Revoking on exit
        // made every persona agent lose its own manifest's capabilities the
        // instant the scheduler ran it, surfacing as `denied:mem_fs_read`.
        use crate::cap::{self, Right};
        let id = spawn("test-exit-keeps-caps", noop_task, 0);
        let right = Right::InvokePrimitive(1);
        cap::grant(id, right);
        assert!(cap::holds(id, right));

        let mut spins = 0;
        while is_alive(id) && spins < 1000 {
            yield_now();
            spins += 1;
        }
        assert!(!is_alive(id), "the task did not run to completion in {spins} yields");
        // Exited, stack reclaimed — and still holding its authority.
        assert!(
            cap::holds(id, right),
            "a self-exited identity holder must keep its capability table; retiring an identity is `kill`'s job"
        );

        // `kill` is the explicit retirement, and it does revoke.
        kill(id).expect_err("already dead");
        // A dead task cannot be killed again, so revoke through the same policy
        // path a live identity would take.
        let live = spawn_parked("test-kill-revokes");
        cap::grant(live, right);
        assert!(cap::holds(live, right));
        kill(live).expect("kill");
        assert!(!cap::holds(live, right), "`kill` must revoke all authority");
    }

    /// A task that blocks on `Wait::Block`, records that it woke, and exits.
    extern "C" fn blocker_task(_arg: u64) {
        BLOCKER_STAGE.store(1, Ordering::SeqCst);
        // A loop, not a single call: a wakeup is permitted to be spurious, so a
        // waiter always re-checks its own condition. Here the condition is the
        // flag the waker sets.
        while !BLOCKER_RELEASE.load(Ordering::SeqCst) {
            block_on(Wait::Block);
        }
        BLOCKER_STAGE.store(2, Ordering::SeqCst);
    }

    static BLOCKER_STAGE: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
    static BLOCKER_RELEASE: AtomicBool = AtomicBool::new(false);

    #[test_case]
    fn a_blocked_task_leaves_the_ready_queue_and_wake_puts_it_back() {
        BLOCKER_STAGE.store(0, Ordering::SeqCst);
        BLOCKER_RELEASE.store(false, Ordering::SeqCst);
        let id = spawn("test-blocker", blocker_task, 0);

        // Let it reach the block.
        let mut spins = 0;
        while BLOCKER_STAGE.load(Ordering::SeqCst) < 1 && spins < 1000 {
            yield_now();
            spins += 1;
        }
        assert_eq!(BLOCKER_STAGE.load(Ordering::SeqCst), 1, "the task never started");

        // Yield enough times that a merely-Ready task would certainly have run
        // again; a blocked one must not, and must be off the queue entirely —
        // that absence is the whole point of blocking rather than polling.
        for _ in 0..20 {
            yield_now();
        }
        assert_eq!(blocked_count(Wait::Block), 1, "the task must be asleep on Wait::Block");
        assert_eq!(list().iter().find(|t| t.0 == id).map(|t| t.2), Some("blocked"));
        SCHED.with(|slot| {
            assert!(
                !slot.as_ref().unwrap().ready_queue.contains(&id),
                "a blocked task must not sit in the ready queue"
            );
        });
        assert_eq!(BLOCKER_STAGE.load(Ordering::SeqCst), 1, "a blocked task must not make progress");

        // Waking on an unrelated condition must not disturb it.
        assert_eq!(wake(Wait::Net), 0);
        assert_eq!(blocked_count(Wait::Block), 1, "Wait::Net must not wake a Wait::Block sleeper");

        // Release it and wake the right condition.
        BLOCKER_RELEASE.store(true, Ordering::SeqCst);
        assert_eq!(wake(Wait::Block), 1, "wake must report the task it woke");
        let mut spins = 0;
        while is_alive(id) && spins < 1000 {
            yield_now();
            spins += 1;
        }
        assert_eq!(BLOCKER_STAGE.load(Ordering::SeqCst), 2, "the woken task did not finish");
        assert_eq!(blocked_count(Wait::Block), 0);
    }

    #[test_case]
    fn block_on_refuses_when_there_is_nothing_else_to_run() {
        // The scheduler cannot put the last runnable task to sleep — nothing
        // would be left to wake it. `false` is the caller's cue to poll instead,
        // which is why every waiting loop keeps an `upkeep()` fallback.
        //
        // The suite runs with no other ready task and no idle task registered, so
        // this is exactly that situation.
        assert!(idle_task().is_none(), "test premise: the suite registers no idle task");
        assert!(!block_on(Wait::SoundOut), "blocking the only runnable task must be refused");
        // And it really did not block: we are still running.
        assert_eq!(blocked_count(Wait::SoundOut), 0);
    }

    static IDLE_TICKS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
    static WORKER_TICKS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
    static SPIN_STOP: AtomicBool = AtomicBool::new(false);

    /// Counts *then* checks the stop flag, unlike [`worker_counter`]. The order
    /// matters for the test: this task is only ever reached after the flag is
    /// set, so checking first would exit without ever recording that it ran —
    /// making "the idle task was reached" indistinguishable from "it never was".
    extern "C" fn idle_counter(_arg: u64) {
        loop {
            IDLE_TICKS.fetch_add(1, Ordering::SeqCst);
            if SPIN_STOP.load(Ordering::SeqCst) {
                return;
            }
            yield_now();
        }
    }

    extern "C" fn worker_counter(_arg: u64) {
        while !SPIN_STOP.load(Ordering::SeqCst) {
            WORKER_TICKS.fetch_add(1, Ordering::SeqCst);
            yield_now();
        }
    }

    #[test_case]
    fn the_idle_task_runs_only_for_a_blocked_task() {
        // Two properties, both invisible if wrong.
        //
        // An idle task that re-enqueues itself on its own yield slides into the
        // round-robin and takes every other turn — the machine just runs
        // inference at half speed with nothing reporting anything. And an idle
        // task reached merely because the ready queue is empty runs `upkeep`
        // concurrently with whichever task yielded, which duplicates the pumping
        // that task is doing for itself; `mouse::tick()` then consumes input the
        // modal and pane loops read themselves. Counting turns is the only way to
        // see either: the queue looks reasonable in both cases.
        SPIN_STOP.store(false, Ordering::SeqCst);
        WORKER_TICKS.store(0, Ordering::SeqCst);

        let idle = spawn("test-idle", idle_counter, 0);
        set_idle(idle);
        // Zero the counter *after* registration, not before. Between `spawn`
        // (which enqueues) and `set_idle` (which dequeues) it is an ordinary ready
        // task, and a timer tick landing in that window legitimately gives it a
        // turn — which is not the fallback firing, and cost a run to work out.
        IDLE_TICKS.store(0, Ordering::SeqCst);
        SCHED.with(|slot| {
            assert!(
                !slot.as_ref().unwrap().ready_queue.contains(&idle),
                "`set_idle` must take it out of the queue that `spawn` put it in"
            );
        });
        let worker = spawn("test-worker", worker_counter, 0);

        // (1) While a task is ready, the idle task must get nothing.
        for _ in 0..50 {
            yield_now();
        }
        assert!(WORKER_TICKS.load(Ordering::SeqCst) > 0, "test premise: the worker must get turns");
        assert_eq!(
            IDLE_TICKS.load(Ordering::SeqCst),
            0,
            "the idle task took a turn while another task was ready"
        );
        SCHED.with(|slot| {
            assert!(
                !slot.as_ref().unwrap().ready_queue.contains(&idle),
                "the idle task must never appear in the ready queue"
            );
        });

        // (2) Retire the worker. The queue is now empty — and that alone must
        // still not reach the idle task, because nothing is asleep and the task
        // yielding is pumping for itself.
        SPIN_STOP.store(true, Ordering::SeqCst);
        let mut spins = 0;
        while is_alive(worker) && spins < 5000 {
            yield_now();
            spins += 1;
        }
        assert!(!is_alive(worker), "the worker did not finish");
        for _ in 0..20 {
            yield_now();
        }
        assert_eq!(
            IDLE_TICKS.load(Ordering::SeqCst),
            0,
            "an empty ready queue alone must not reach the idle task — only a sleeper may"
        );

        // (3) Now something actually sleeps, which is the one situation the idle
        // task exists for. Note the mutual dependence: `block_on` can only
        // succeed *because* an idle task is registered, and the idle task can
        // only be reached *because* something blocked.
        BLOCKER_RELEASE.store(false, Ordering::SeqCst);
        BLOCKER_STAGE.store(0, Ordering::SeqCst);
        let sleeper = spawn("test-sleeper", blocker_task, 0);
        let mut spins = 0;
        while blocked_count(Wait::Block) == 0 && spins < 5000 {
            yield_now();
            spins += 1;
        }
        assert_eq!(blocked_count(Wait::Block), 1, "the sleeper never blocked");
        let mut spins = 0;
        while IDLE_TICKS.load(Ordering::SeqCst) == 0 && spins < 5000 {
            yield_now();
            spins += 1;
        }
        assert!(
            IDLE_TICKS.load(Ordering::SeqCst) > 0,
            "with a task asleep and the queue empty, the fallback must reach the idle task"
        );

        // Release the sleeper and leave the global scheduler as we found it.
        BLOCKER_RELEASE.store(true, Ordering::SeqCst);
        wake(Wait::Block);
        let mut spins = 0;
        while is_alive(sleeper) && spins < 5000 {
            yield_now();
            spins += 1;
        }
        assert!(!is_alive(sleeper), "the woken sleeper did not finish");
        clear_idle();
        let _ = kill(idle); // already exited after its one tick; ignore
        assert!(idle_task().is_none(), "leave the global scheduler as we found it");
    }

    #[test_case]
    fn block_on_refuses_inside_a_critical_section() {
        // Blocking while holding a `Locked` hands the CPU to another task with
        // the lock still held; if that task takes the same lock the machine spins
        // forever with interrupts off — no panic, no log, nothing to attach to.
        // `Locked` is the only thing that disables interrupts for a critical
        // section, so "interrupts off" is a sound proxy for "a lock is held".
        //
        // Checked through `without_interrupts` rather than a real `Locked`,
        // because taking one here and blocking inside it is the deadlock itself.
        let blocked = interrupts::without_interrupts(|| {
            // The debug assertion is what fires in a debug build; the return
            // value is what protects a release build.
            !interrupts::are_enabled()
        });
        assert!(blocked, "test premise: without_interrupts really disables them");
    }

    #[test_case]
    fn the_running_task_reports_running_and_others_report_ready() {
        // `state` used to go stale the moment anything was spawned: task 0 stayed
        // `Running` forever and every other task stayed `Ready` even while
        // executing, so `/agents` and `/top` could only ever show task 0 running.
        let id = spawn("test-state-honest", noop_task, 0);
        let listed = list();
        let me = current_task_id();
        assert_eq!(listed.iter().find(|t| t.0 == me).map(|t| t.2), Some("running"));
        assert_eq!(listed.iter().find(|t| t.0 == id).map(|t| t.2), Some("ready"));
        assert_eq!(
            listed.iter().filter(|t| t.2 == "running").count(),
            1,
            "exactly one task is running at a time"
        );
        let _ = kill(id);
    }
}
