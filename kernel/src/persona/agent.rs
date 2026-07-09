//! The **Agent** (`CHITTI_OS_HANDOFF.md` Phase 5): an agent as a first-class
//! process. It owns a scheduler task (its identity + capability table), a
//! manifest, a live context (tier-1 memory), and a plan it works through by
//! calling Synapse primitives. It has a process lifecycle -- spawn, suspend,
//! resume, kill.
//!
//! The load-bearing detail is what suspend/resume checkpoints. Per the phase
//! spec, a checkpoint keeps only the *context and memory pointers* -- the
//! cheap, durable description of where the agent is -- and explicitly drops
//! the expensive derived state (the KV cache / [`LiveState`]). Resume does
//! **not** restore that state; it **recomputes** it from the checkpointed
//! context. That is why suspend is cheap regardless of how large the KV cache
//! grew, and why a resumed agent continues correctly without ever persisting
//! hundreds of megabytes.

use super::actions::Action;
use super::manifest::Manifest;
use super::memory::{self, Context, Role};
use super::planner::Planner;
use crate::cap::{self, Right};
use crate::sched::{self, TaskId};
use crate::security::{Justification, Provenance};
use crate::synapse::{self, Invocation};
use alloc::string::String;
use alloc::vec::Vec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentState {
    Ready,
    Running,
    Suspended,
    Dead,
}

/// Expensive, derived, **recomputable** agent state -- the KV cache analogue.
/// Deliberately *not* part of a checkpoint: it is dropped on suspend and
/// rebuilt on resume. For a `ModelRef::Cortex` agent this would hold the
/// `cortex::model::Cache`, rebuilt by re-prefilling the context; in the
/// model-free runtime we track just enough (`built_from`) to make the
/// recompute observable.
pub struct LiveState {
    /// How many context messages this live state was (re)built from.
    pub built_from: usize,
}

/// Everything needed to reconstruct an agent's working state -- and nothing
/// more. Cheap to hold and independent of KV-cache size.
struct Checkpoint {
    messages: Vec<memory::Message>,
    paged_keys: Vec<String>,
    cursor: usize,
    plan: Vec<Action>,
    last_result: String,
}

pub struct Agent {
    pub manifest: Manifest,
    /// The agent's own task: its identity and capability table. The plan is
    /// executed synchronously by the driver, but Synapse calls are attributed
    /// to (and capability-checked against) *this* task.
    pub task: TaskId,
    pub state: AgentState,
    ctx: Context,
    plan: Vec<Action>,
    cursor: usize,
    live: Option<LiveState>,
    checkpoint: Option<Checkpoint>,
    recompute_count: u64,
    last_result: String,
    /// When set, this agent's destructive calls carry an explicit human
    /// confirmation, letting them through the Synapse taint gate even on a
    /// tainted justification. The interactive shell sets this only after a
    /// human answers the confirmation prompt (Phase 6). Default `false`.
    confirm_destructive: bool,
}

/// Entry point for an agent's task. The plan/act loop runs synchronously in
/// the driver (shell or test), so the task itself only needs to exist to own
/// the agent's identity and capability table; it parks immediately. (Agents
/// that must run concurrently -- e.g. the IPC-coordination test -- are spawned
/// as ordinary tasks with their own entry, not through `Agent`.)
extern "C" fn agent_task_entry(_arg: u64) {}

impl Agent {
    /// Spawn an agent from its manifest: create its task, and grant that task
    /// exactly the capabilities the manifest declares (no ambient authority).
    pub fn spawn(manifest: Manifest) -> Agent {
        let task = sched::spawn("persona-agent", agent_task_entry, 0);
        for &prim in &manifest.capabilities {
            cap::grant(task, Right::InvokePrimitive(prim));
        }
        let ctx = Context::new(manifest.memory.working_set_limit);
        crate::ktrace::log_fmt(format_args!(
            "persona.spawn: agent '{}' (task {task}) with {} capabilities, planner={:?}",
            manifest.name,
            manifest.capabilities.len(),
            manifest.model
        ));
        Agent {
            manifest,
            task,
            state: AgentState::Ready,
            ctx,
            plan: Vec::new(),
            cursor: 0,
            live: Some(LiveState { built_from: 0 }),
            checkpoint: None,
            recompute_count: 0,
            last_result: String::new(),
            confirm_destructive: false,
        }
    }

    /// Frame an intent: seed context with the persona prompt (once) and the
    /// intent, then ask `planner` for a plan. Resets the step cursor.
    pub fn begin(&mut self, intent: &str, planner: &mut dyn Planner) {
        if self.ctx.is_empty() {
            self.ctx.push(Role::System, &self.manifest.persona_prompt);
        }
        self.ctx.push(Role::User, intent);
        self.plan = planner.plan(intent);
        self.cursor = 0;
        self.last_result = String::new();
        self.state = AgentState::Running;
        crate::ktrace::log_fmt(format_args!(
            "persona.begin: agent '{}' planned {} step(s) for intent",
            self.manifest.name,
            self.plan.len()
        ));
    }

    /// Frame an intent with a *pre-built* plan, skipping the planner entirely.
    /// This is how a compiled intent (`persona::compiled`) is replayed: the
    /// validated capability trace is installed directly, so no inference runs.
    pub fn begin_with_plan(&mut self, intent: &str, plan: Vec<Action>) {
        if self.ctx.is_empty() {
            self.ctx.push(Role::System, &self.manifest.persona_prompt);
        }
        self.ctx.push(Role::User, intent);
        let n = plan.len();
        self.plan = plan;
        self.cursor = 0;
        self.last_result = String::new();
        self.state = AgentState::Running;
        crate::ktrace::log_fmt(format_args!(
            "persona.begin(compiled): agent '{}' replaying a {n}-step trace (no planning)",
            self.manifest.name
        ));
    }

    /// The current plan (used by `persona::compiled` to record a trace).
    pub fn plan(&self) -> &[Action] {
        &self.plan
    }

    /// Arm/disarm human confirmation for this agent's destructive calls
    /// (Phase 6 taint gate override). The shell sets this after a human
    /// confirms at the prompt.
    pub fn set_confirm_destructive(&mut self, confirm: bool) {
        self.confirm_destructive = confirm;
    }

    pub fn finished(&self) -> bool {
        self.cursor >= self.plan.len()
    }

    pub fn result(&self) -> &str {
        &self.last_result
    }

    /// Whether the (recomputable) live state is currently resident. `false`
    /// exactly while suspended.
    pub fn live_present(&self) -> bool {
        self.live.is_some()
    }

    pub fn recompute_count(&self) -> u64 {
        self.recompute_count
    }

    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Execute the next action in the plan. Returns `false` if the plan was
    /// already complete. A suspended agent must be resumed first.
    pub fn step(&mut self) -> bool {
        assert!(self.state != AgentState::Suspended, "cannot step a suspended agent; resume first");
        if self.finished() {
            return false;
        }
        let action = self.plan[self.cursor].clone();
        let provenance = result_provenance(&action);
        let result = self.run_action(action);
        // Tag the result with its provenance so it taints later calls
        // appropriately: content read from the store / recalled facts are
        // untrusted ingested; action acknowledgements are system-trusted.
        self.ctx.push_tagged(Role::Tool, &result, provenance);
        self.last_result = result;
        self.cursor += 1;
        if self.finished() {
            self.state = AgentState::Ready;
        }
        true
    }

    /// Run the whole plan to completion and return the final result.
    pub fn run_to_completion(&mut self) -> &str {
        while self.step() {}
        &self.last_result
    }

    fn run_action(&mut self, action: Action) -> String {
        match action {
            Action::Call(raw) => {
                // Justify the call by the provenance of the context it was
                // planned from; a human confirmation (if armed) overrides the
                // taint gate for destructive primitives.
                let mut justification = Justification::from_context(self.ctx.max_taint());
                if self.confirm_destructive {
                    justification = justification.confirmed();
                }
                match synapse::execute_with_justification(self.task, &raw, justification) {
                    Invocation::Executed { result, .. } => result,
                    Invocation::Denied { primitive } => alloc::format!("denied:{primitive}"),
                    Invocation::Rejected(err) => alloc::format!("rejected:{err:?}"),
                    Invocation::RefusedTainted { primitive } => alloc::format!("refused:tainted:{primitive}"),
                    Invocation::DeniedScope { primitive } => alloc::format!("denied:scope:{primitive}"),
                }
            }
            Action::Remember(key, value) => {
                memory::remember(&self.manifest.name, &key, &value);
                alloc::format!("ok:remembered {key}")
            }
            Action::Recall(key) => match memory::recall(&self.manifest.name, &key, &mut self.ctx) {
                Some(value) => value,
                None => alloc::format!("error:unknown {key}"),
            },
        }
    }

    /// Suspend: checkpoint context + memory pointers + plan cursor, and
    /// **drop** the recomputable live state (the KV cache). Cheap and
    /// bounded regardless of how large that state was.
    pub fn suspend(&mut self) {
        self.checkpoint = Some(Checkpoint {
            messages: self.ctx.messages.clone(),
            paged_keys: self.ctx.paged_keys.clone(),
            cursor: self.cursor,
            plan: self.plan.clone(),
            last_result: self.last_result.clone(),
        });
        self.live = None; // drop the KV cache -- it is recomputed, never saved
        self.state = AgentState::Suspended;
        crate::ktrace::log_fmt(format_args!(
            "persona.suspend: agent '{}' checkpointed {} msg(s) + {} memory ptr(s); dropped KV/live state",
            self.manifest.name,
            self.ctx.messages.len(),
            self.ctx.paged_keys.len()
        ));
    }

    /// Resume: rebuild working state from the checkpoint and **recompute**
    /// (not restore) the live state. Continues from the same plan cursor.
    pub fn resume(&mut self) {
        let cp = self.checkpoint.take().expect("resume without a checkpoint");
        let mut ctx = Context::new(self.manifest.memory.working_set_limit);
        for m in cp.messages {
            ctx.push(m.role, &m.text);
        }
        ctx.paged_keys = cp.paged_keys;
        self.ctx = ctx;
        self.cursor = cp.cursor;
        self.plan = cp.plan;
        self.last_result = cp.last_result;
        // Recompute the live state from the reconstructed context. For a
        // Cortex agent this is where prefill would rerun; here we record that
        // it was rebuilt (from scratch, not restored) so resume is observable.
        self.recompute_count += 1;
        self.live = Some(LiveState { built_from: self.ctx.messages.len() });
        self.state = if self.finished() { AgentState::Ready } else { AgentState::Running };
        crate::ktrace::log_fmt(format_args!(
            "persona.resume: agent '{}' recomputed live state from {} msg(s) (recompute #{})",
            self.manifest.name,
            self.ctx.messages.len(),
            self.recompute_count
        ));
    }

    /// Terminate the agent. Marks it dead; its task's stack is reclaimed by
    /// the scheduler's existing policy (Phase 2 leaks dead-task stacks).
    pub fn kill(&mut self) {
        self.state = AgentState::Dead;
        crate::ktrace::log_fmt(format_args!("persona.kill: agent '{}' (task {}) terminated", self.manifest.name, self.task));
    }
}

/// The provenance to tag an action's result with when it enters live context.
/// Content the agent *ingests* (a file it read, a fact it recalled) is
/// untrusted; a mere acknowledgement of an effect the agent itself caused
/// (a write/console/delete "ok") is system-trusted and does not taint later
/// reasoning.
fn result_provenance(action: &Action) -> Provenance {
    match action {
        Action::Recall(_) => Provenance::UntrustedIngested,
        // Only a content read ingests untrusted data; other calls just ack.
        Action::Call(raw) if raw.starts_with(r#"{"name":"mem_fs_read""#) => Provenance::UntrustedIngested,
        _ => Provenance::SystemTrusted,
    }
}
