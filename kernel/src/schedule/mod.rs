//! Scheduled runs — a stored intent that acts later.
//!
//! ## The governing principle
//!
//! **A schedule is bounded by what its author could have done at the moment it
//! was stored — forever.** Concretely: by the recurrence it was stored with; by
//! the agent identity and toolset frozen at creation; by the taint of its
//! authoring context, which is never lowered; and by the absence of a human at
//! run time, which turns every approval-requiring call into a refusal plus a
//! notification rather than a silent grant. A schedule cannot *acquire*
//! authority by running. It can only lose it — a revoked capability, a killed
//! agent, a disabled job.
//!
//! That is the direct analogue of invariant 5 (a skill is bounded by its
//! install-time grant, forever), and it composes with invariant 3 (delegation
//! only narrows): the run's justification is built from the frozen [`Grant`], so
//! a schedule is at most as authorised as its author was.
//!
//! ## Where the work happens, and why it is split
//!
//! [`tick`] runs from `shell::upkeep` and **only decides and enqueues**. The
//! actual run happens in `shell::schedule::drain_schedule_pending`, called from
//! the interactive loop. That split is not stylistic — it is the same one
//! `msgchan` makes, and the comment on `shell::fs::drain_channel_inbound` says
//! why: *inference is too heavy for the poll tick*. Two further reasons specific
//! to this module:
//!
//! - A prompt run goes through `ChatSession::turn`, which wraps itself in a
//!   1 MiB task (`CHAT_TURN_STACK`) because turn → forward → ONNX → tool →
//!   sub-agent is the deepest chain in the kernel. Driving it from the REPL gets
//!   that for free; driving it from a `ServiceSpec` daemon would mean the 256 KiB
//!   default stack and re-implementing the wrapper.
//! - `mm::Locked` is not reentrant, and `tick` → run → `/http` → `upkeep()` →
//!   `tick` is a real path. Hence [`with_busy`], and hence `tick` never calls
//!   `notify::post` while holding [`JOBS`].
//!
//! ## No new Synapse primitive
//!
//! A schedule creates **no effect at creation time**. Its effects happen later,
//! through the existing primitives, under the existing four gates, with the
//! frozen grant. A `schedule_create` primitive would gate *writing a JSON
//! record* while the authority question is at fire time — it would be
//! `mem_fs_write` wearing a costume. What this feature does add is one genuine
//! new gate, in `tools::dispatch::effect_of`: installing a schedule is
//! classified by the effect of **the action being installed**, so an
//! injection-authored `schedule add … rm -r` hits the router's destructive check
//! at creation, while a human can still see it.

pub mod spec;

use crate::json::Json;
use crate::mm::Locked;
use crate::synapse::fs as store;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use spec::{Author, Catchup, Due, GrantFacts, NotifyOn, Recurrence};

pub const CONFIG_PATH: &str = "/configs/core/schedules.json";
pub const MAX_JOBS: usize = 32;
/// Bounded queue of committed fires waiting for the interactive loop.
pub const MAX_PENDING: usize = 16;
/// How often [`tick`] does real work. `upkeep` runs far more often than 1 Hz.
const TICK_INTERVAL_MS: u64 = 1_000;
/// Don't evaluate while the human is mid-keystroke — the `msgchan` rule, because
/// a `/command` run can reach `/http`, which pumps `upkeep`.
const TYPING_QUIET_MS: u64 = 400;
/// Monotonic re-fire floor per job. The last line of defence: a wall clock that
/// jumps *backwards* must not be able to re-fire a job in a tight loop even if
/// both the due-time and slot guards are defeated by a pathological jump.
const MIN_RERUN_MS: u64 = 30_000;

/// What a due schedule runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// A deterministic `/command` through `shell::run_tool_command`. No model,
    /// cheap, and its printed output becomes the notification body.
    Command { name: String, arg: String },
    /// One bounded agent turn with `text` as the framed user message.
    Prompt { text: String },
}

impl Action {
    pub fn kind(&self) -> &'static str {
        match self {
            Action::Command { .. } => "command",
            Action::Prompt { .. } => "prompt",
        }
    }
    /// A one-line human summary, for `/schedule list`.
    pub fn summary(&self) -> String {
        match self {
            Action::Command { name, arg } if arg.is_empty() => alloc::format!("/{name}"),
            Action::Command { name, arg } => alloc::format!("/{name} {arg}"),
            Action::Prompt { text } => alloc::format!("prompt: {text}"),
        }
    }
}

/// The frozen authority a run acts under.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grant {
    pub facts: GrantFacts,
    /// Identity the run acts as. An agent-created schedule records **its own**
    /// id, never a widened one.
    pub agent_id: u64,
    /// FNV-1a of `name|recurrence|action`, for the audit trail — so "is this the
    /// schedule that was approved" is answerable from the log.
    pub authored_hash: u64,
    pub created_unix: i64,
}

#[derive(Clone, Debug)]
pub struct Job {
    pub id: u64,
    /// Unique handle for `/schedule <verb> <name>`.
    pub name: String,
    pub spec: Recurrence,
    pub action: Action,
    pub grant: Grant,
    pub enabled: bool,
    pub catchup: Catchup,
    pub notify: NotifyOn,
    /// Wall-clock second of the last completed run (0 = never).
    pub last_run_unix: i64,
    /// Nominal-slot key of the last fire — the duplicate guard, persisted so a
    /// reboot cannot re-fire the same minute.
    pub last_slot: i64,
    /// The next fire this scheduler committed to, in UTC seconds. Persisted so a
    /// reboot cannot re-derive a *different* answer from a clock that moved.
    ///
    /// For [`Recurrence::Every`] the authority is the monotonic [`Job::due_ms`];
    /// this field is bookkeeping so `/schedule list` can print a human answer
    /// and a boot can classify the job as overdue.
    pub next_due_unix: i64,
    pub run_count: u64,
    /// `ok` | `err: …` | `cancelled` | `skipped: no planner` | `dropped: …`
    pub last_status: String,
    /// FNV-1a of the last run's output, for `NotifyOn::OnChange`. Not persisted:
    /// re-notifying once after a reboot is better than persisting a hash whose
    /// meaning depends on a command's output format.
    pub last_output_hash: u64,
    /// RAM-only monotonic due time for `Every`.
    pub due_ms: u64,
    /// RAM-only monotonic floor for [`MIN_RERUN_MS`].
    pub last_fire_ms: u64,
}

/// A committed fire, handed to the drain.
#[derive(Clone, Debug)]
pub struct Fire {
    pub id: u64,
    pub name: String,
    pub action: Action,
    pub grant: Grant,
    pub notify: NotifyOn,
    pub coalesced_missed: u32,
}

static JOBS: Locked<Vec<Job>> = Locked::new(Vec::new());
static PENDING: Locked<Vec<Fire>> = Locked::new(Vec::new());
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static BUSY: AtomicBool = AtomicBool::new(false);
static LAST_TICK_MS: AtomicU64 = AtomicU64::new(0);
/// `(last_seen_unix, last_seen_ms)` for the clock-jump detector.
static CLOCK_WATCH: Locked<(i64, u64)> = Locked::new((0, 0));
/// Whether the clock was trusted at the last tick, so the untrusted→trusted
/// transition (the `/ntp`-at-boot case) can re-anchor the held calendar jobs.
static CLOCK_WAS_TRUSTED: AtomicBool = AtomicBool::new(false);

/// Reentrancy guard. `tick` → enqueue → drain → `/http` → `upkeep()` → `tick`
/// is a real path, and `Locked` is not reentrant: a re-entry spins forever with
/// interrupts disabled, which stops the machine with no output and no Ctrl+C.
fn with_busy<R>(f: impl FnOnce() -> R) -> Option<R> {
    if BUSY.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
        return None;
    }
    let out = f();
    BUSY.store(false, Ordering::Release);
    Some(out)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn author_hash(name: &str, spec: Recurrence, action: &Action) -> u64 {
    fnv1a(alloc::format!("{name}|{}|{}", spec::render(spec), action.summary()).as_bytes())
}

// ---------------------------------------------------------------------------
// Persistence — the `msgchan` idiom: every field defaulted, so an older or
// hand-edited file loads rather than being discarded.
// ---------------------------------------------------------------------------

fn prov_str(p: crate::security::taint::Provenance) -> &'static str {
    use crate::security::taint::Provenance as P;
    match p {
        P::SystemTrusted => "system_trusted",
        P::UserTyped => "user_typed",
        P::UntrustedIngested => "untrusted_ingested",
    }
}

fn prov_parse(s: &str) -> Option<crate::security::taint::Provenance> {
    use crate::security::taint::Provenance as P;
    match s {
        "system_trusted" => Some(P::SystemTrusted),
        "user_typed" => Some(P::UserTyped),
        "untrusted_ingested" => Some(P::UntrustedIngested),
        _ => None,
    }
}

pub fn load() {
    let Some(bytes) = store::read(CONFIG_PATH) else {
        return;
    };
    let Ok(text) = core::str::from_utf8(&bytes) else {
        return;
    };
    let Some(Json::Arr(arr)) = Json::parse(text) else {
        return;
    };
    let now_unix = crate::clock::now_unix();
    let now_ms = crate::arch::now_ms();
    let mut out: Vec<Job> = Vec::new();
    for item in arr {
        let Json::Obj(pairs) = item else { continue };
        let get = |k: &str| pairs.iter().find(|(a, _)| a == k).map(|(_, v)| v);
        let name = get("name").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        if name.is_empty() || out.iter().any(|j| j.name == name) {
            continue;
        }
        let recur = get("recur").and_then(|v| v.as_str()).unwrap_or("");
        // A recurrence that no longer parses (a hand edit, a removed kind) drops
        // the job rather than defaulting it to some interval nobody asked for.
        let Ok(spec) = spec::parse(recur, now_unix) else {
            crate::ktrace::log_fmt(format_args!(
                "schedule: dropping '{name}' — unparseable recurrence '{recur}'"
            ));
            continue;
        };
        let action = match get("kind").and_then(|v| v.as_str()).unwrap_or("command") {
            "prompt" => Action::Prompt {
                text: get("prompt").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            },
            _ => Action::Command {
                name: get("command").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                arg: get("arg").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            },
        };
        let facts = GrantFacts {
            author: get("author")
                .and_then(|v| v.as_str())
                .and_then(Author::parse)
                .unwrap_or(Author::Human),
            // Defaulting to the *worst* provenance on a missing field, not the
            // best: an unreadable authority record must not read as clean.
            provenance: get("provenance")
                .and_then(|v| v.as_str())
                .and_then(prov_parse)
                .unwrap_or(crate::security::taint::Provenance::UntrustedIngested),
            human_confirmed: get("human_confirmed").and_then(|v| v.as_bool()).unwrap_or(false),
        };
        let spec_r = spec;
        let job = Job {
            id: get("id").and_then(|v| v.as_i64()).unwrap_or(0).max(0) as u64,
            grant: Grant {
                facts,
                agent_id: get("agent")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(crate::agent::manifest::ORCHESTRATOR_ID.0 as i64)
                    .max(0) as u64,
                authored_hash: author_hash(&name, spec_r, &action),
                created_unix: get("created_unix").and_then(|v| v.as_i64()).unwrap_or(0),
            },
            name,
            spec: spec_r,
            action,
            enabled: get("enabled").and_then(|v| v.as_bool()).unwrap_or(true),
            catchup: get("catchup")
                .and_then(|v| v.as_str())
                .and_then(Catchup::parse)
                .unwrap_or(Catchup::Skip),
            notify: get("notify")
                .and_then(|v| v.as_str())
                .and_then(NotifyOn::parse)
                .unwrap_or(NotifyOn::OnChange),
            last_run_unix: get("last_run_unix").and_then(|v| v.as_i64()).unwrap_or(0),
            last_slot: get("last_slot").and_then(|v| v.as_i64()).unwrap_or(0),
            next_due_unix: get("next_due_unix").and_then(|v| v.as_i64()).unwrap_or(0),
            run_count: get("run_count").and_then(|v| v.as_i64()).unwrap_or(0).max(0) as u64,
            last_status: get("last_status").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            last_output_hash: 0,
            // An `Every` job's monotonic due time cannot survive a reboot (the
            // counter restarts), so it is re-anchored to one interval from now:
            // an interval schedule means "this often", not "at these instants".
            due_ms: match spec_r {
                Recurrence::Every { secs } => now_ms + (secs.max(1) as u64) * 1000,
                _ => 0,
            },
            last_fire_ms: 0,
        };
        out.push(job);
        if out.len() >= MAX_JOBS {
            break;
        }
    }
    let max = out.iter().map(|j| j.id).max().unwrap_or(0);
    NEXT_ID.fetch_max(max + 1, Ordering::Relaxed);
    let n = out.len();
    JOBS.with(|v| *v = out);
    CLOCK_WAS_TRUSTED.store(crate::clock::source().trusted(), Ordering::Relaxed);
    crate::ktrace::log_fmt(format_args!("schedule: loaded {n} job(s) from {CONFIG_PATH}"));
}

fn job_json(j: &Job) -> Json {
    let (cmd, arg, prompt) = match &j.action {
        Action::Command { name, arg } => (name.clone(), arg.clone(), String::new()),
        Action::Prompt { text } => (String::new(), String::new(), text.clone()),
    };
    Json::Obj(alloc::vec![
        (String::from("id"), Json::Num(j.id as f64)),
        (String::from("name"), Json::Str(j.name.clone())),
        // The recurrence is ONE string, not exploded fields: `parse`/`render`
        // round-trip it, which keeps the file human-editable and means adding a
        // recurrence kind does not change the schema.
        (String::from("recur"), Json::Str(spec::render(j.spec))),
        (String::from("kind"), Json::Str(String::from(j.action.kind()))),
        (String::from("command"), Json::Str(cmd)),
        (String::from("arg"), Json::Str(arg)),
        (String::from("prompt"), Json::Str(prompt)),
        (String::from("author"), Json::Str(String::from(j.grant.facts.author.as_str()))),
        (String::from("provenance"), Json::Str(String::from(prov_str(j.grant.facts.provenance)))),
        (String::from("human_confirmed"), Json::Bool(j.grant.facts.human_confirmed)),
        (String::from("agent"), Json::Num(j.grant.agent_id as f64)),
        (String::from("authored_hash"), Json::Str(alloc::format!("{:016x}", j.grant.authored_hash))),
        (String::from("created_unix"), Json::Num(j.grant.created_unix as f64)),
        (String::from("enabled"), Json::Bool(j.enabled)),
        (String::from("catchup"), Json::Str(String::from(j.catchup.as_str()))),
        (String::from("notify"), Json::Str(String::from(j.notify.as_str()))),
        (String::from("last_run_unix"), Json::Num(j.last_run_unix as f64)),
        (String::from("last_slot"), Json::Num(j.last_slot as f64)),
        (String::from("next_due_unix"), Json::Num(j.next_due_unix as f64)),
        (String::from("run_count"), Json::Num(j.run_count as f64)),
        (String::from("last_status"), Json::Str(j.last_status.clone())),
    ])
}

/// Persist the job table. Called by every mutator — but only when state actually
/// changed, because the ext4 backend re-formats on sync and a 60-second job must
/// not rewrite this file every minute.
pub fn save() {
    let text = JOBS.with(|v| Json::Arr(v.iter().map(job_json).collect()).to_pretty());
    store::write(CONFIG_PATH, text.as_bytes());
}

// ---------------------------------------------------------------------------
// Queries + mutators
// ---------------------------------------------------------------------------

pub fn list() -> Vec<Job> {
    JOBS.with(|v| v.clone())
}

pub fn get(name: &str) -> Option<Job> {
    JOBS.with(|v| v.iter().find(|j| j.name == name).cloned())
}

pub fn count() -> usize {
    JOBS.with(|v| v.len())
}

pub fn pending_len() -> usize {
    PENDING.with(|v| v.len())
}

pub fn take_pending() -> Option<Fire> {
    PENDING.with(|v| if v.is_empty() { None } else { Some(v.remove(0)) })
}

/// How many calendar jobs are currently held because the clock is not
/// trustworthy — the answer `/schedule next` needs, because "my schedule didn't
/// run" otherwise has three indistinguishable causes.
pub fn held_count() -> usize {
    let trusted = crate::clock::source().trusted();
    JOBS.with(|v| {
        v.iter()
            .filter(|j| j.enabled && spec::held_for_clock(j.spec, trusted))
            .count()
    })
}

/// Add a job. `facts` and `agent_id` are the frozen authority — the caller
/// decides them, because they cannot be recovered later.
#[allow(clippy::too_many_arguments)]
pub fn add(
    name: &str,
    spec_r: Recurrence,
    action: Action,
    facts: GrantFacts,
    agent_id: u64,
    catchup: Catchup,
    notify: NotifyOn,
) -> Result<u64, String> {
    let name = name.trim().to_string();
    if name.is_empty() {
        return Err(String::from("a schedule needs a name"));
    }
    if name.split_whitespace().count() > 1 {
        return Err(String::from("a schedule name cannot contain whitespace"));
    }
    if JOBS.with(|v| v.iter().any(|j| j.name == name)) {
        return Err(alloc::format!("a schedule called '{name}' already exists"));
    }
    if JOBS.with(|v| v.len()) >= MAX_JOBS {
        return Err(alloc::format!("too many schedules (limit {MAX_JOBS})"));
    }
    let now_unix = crate::clock::now_unix();
    let now_ms = crate::arch::now_ms();
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let next_due_unix = spec::next_due(spec_r, now_unix, &crate::clock::offset_at).unwrap_or(0);
    let job = Job {
        id,
        grant: Grant {
            facts,
            agent_id,
            authored_hash: author_hash(&name, spec_r, &action),
            created_unix: now_unix,
        },
        name: name.clone(),
        spec: spec_r,
        action,
        enabled: true,
        catchup,
        notify,
        last_run_unix: 0,
        last_slot: 0,
        next_due_unix,
        run_count: 0,
        last_status: String::new(),
        last_output_hash: 0,
        due_ms: match spec_r {
            Recurrence::Every { secs } => now_ms + (secs.max(1) as u64) * 1000,
            _ => 0,
        },
        last_fire_ms: 0,
    };
    let hash = job.grant.authored_hash;
    JOBS.with(|v| v.push(job));
    save();
    crate::synapse::audit::record(
        crate::sched::current_task_id(),
        "schedule_create",
        hash,
        crate::synapse::audit::Outcome::Executed,
        id,
    );
    crate::ktrace::log_fmt(format_args!(
        "schedule: added '{name}' ({}) author={} confirmed={}",
        spec::render(spec_r),
        facts.author.as_str(),
        facts.human_confirmed
    ));
    Ok(id)
}

pub fn remove(name: &str) -> bool {
    let hit = JOBS.with(|v| {
        let before = v.len();
        v.retain(|j| j.name != name);
        v.len() != before
    });
    if hit {
        // Drop any committed-but-unrun fire too: removing a schedule must not
        // leave one last run in the queue.
        PENDING.with(|v| v.retain(|f| f.name != name));
        save();
    }
    hit
}

pub fn set_enabled(name: &str, on: bool) -> Option<bool> {
    let changed = JOBS.with(|v| {
        v.iter_mut().find(|j| j.name == name).map(|j| {
            let was = j.enabled;
            j.enabled = on;
            // Re-anchor an interval job on resume, or a job paused for an hour
            // fires the instant it comes back.
            if on && !was {
                if let Recurrence::Every { secs } = j.spec {
                    j.due_ms = crate::arch::now_ms() + (secs.max(1) as u64) * 1000;
                }
            }
            was != on
        })
    })?;
    if changed {
        save();
    }
    Some(changed)
}

pub fn set_catchup(name: &str, c: Catchup) -> bool {
    let hit = JOBS.with(|v| match v.iter_mut().find(|j| j.name == name) {
        Some(j) => {
            j.catchup = c;
            true
        }
        None => false,
    });
    if hit {
        save();
    }
    hit
}

pub fn set_notify(name: &str, n: NotifyOn) -> bool {
    let hit = JOBS.with(|v| match v.iter_mut().find(|j| j.name == name) {
        Some(j) => {
            j.notify = n;
            true
        }
        None => false,
    });
    if hit {
        save();
    }
    hit
}

/// Commit a fire: advance and persist the bookkeeping **before** the run.
///
/// Deliberately at-most-once. A crash mid-run loses the run rather than
/// repeating it, because a missed maintenance run is cheaper than a duplicated
/// irreversible one, and the loss is recorded in `last_status`.
pub fn commit_fire(id: u64, coalesced_missed: u32) -> Option<Fire> {
    let fire = JOBS.with(|v| {
        let j = v.iter_mut().find(|j| j.id == id)?;
        // Re-check enablement: `/schedule pause` must take effect on a job that
        // is already queued.
        if !j.enabled {
            return None;
        }
        j.last_fire_ms = crate::arch::now_ms();
        Some(Fire {
            id: j.id,
            name: j.name.clone(),
            action: j.action.clone(),
            grant: j.grant.clone(),
            notify: j.notify,
            coalesced_missed,
        })
    })?;
    save();
    Some(fire)
}

/// Record the outcome of a run. Returns whether the output differed from last
/// time, which is what [`NotifyOn::OnChange`] needs.
pub fn record_result(id: u64, status: &str, output: &str, ran_at_unix: i64) -> bool {
    let h = fnv1a(output.as_bytes());
    let changed = JOBS.with(|v| match v.iter_mut().find(|j| j.id == id) {
        Some(j) => {
            j.last_run_unix = ran_at_unix;
            j.run_count = j.run_count.saturating_add(1);
            j.last_status = status.to_string();
            let changed = j.last_output_hash != h;
            j.last_output_hash = h;
            // A `Once` schedule has done its one job.
            if matches!(j.spec, Recurrence::Once { .. }) {
                j.enabled = false;
            }
            changed
        }
        None => false,
    });
    save();
    crate::synapse::audit::record(
        crate::sched::current_task_id(),
        "schedule_fire",
        fnv1a(status.as_bytes()),
        if status == "ok" {
            crate::synapse::audit::Outcome::Executed
        } else {
            crate::synapse::audit::Outcome::RejectedMalformed
        },
        id,
    );
    changed
}

/// Queue a job to run at the next drain, out of band. `/schedule run <name>`.
///
/// Does **not** touch `next_due_unix`: running now is not the scheduled fire, so
/// it must not consume it.
pub fn run_now(name: &str) -> Result<(), String> {
    let fire = JOBS.with(|v| {
        v.iter_mut().find(|j| j.name == name).map(|j| {
            j.last_fire_ms = crate::arch::now_ms();
            Fire {
                id: j.id,
                name: j.name.clone(),
                action: j.action.clone(),
                grant: j.grant.clone(),
                notify: NotifyOn::Always, // an explicit run always reports
                coalesced_missed: 0,
            }
        })
    });
    let Some(fire) = fire else {
        return Err(alloc::format!("no schedule called '{name}'"));
    };
    let ok = PENDING.with(|v| {
        if v.len() >= MAX_PENDING {
            return false;
        }
        v.push(fire);
        true
    });
    if ok {
        Ok(())
    } else {
        Err(String::from("the pending queue is full"))
    }
}

// ---------------------------------------------------------------------------
// The tick
// ---------------------------------------------------------------------------

/// Evaluate every job and enqueue the due ones. **Never runs anything.**
///
/// Called from `shell::upkeep`, after `msgchan::tick`.
pub fn tick() {
    let now_ms = crate::arch::now_ms();
    if now_ms.saturating_sub(LAST_TICK_MS.load(Ordering::Relaxed)) < TICK_INTERVAL_MS {
        return;
    }
    // The `msgchan` rule: never fight the line editor. A due job is a
    // human-timescale event, so waiting for a quiet moment costs nothing.
    let last_key = crate::console::input_activity_ms();
    if last_key != 0 && now_ms.saturating_sub(last_key) < TYPING_QUIET_MS {
        return;
    }
    // A modal owns `console::read_byte`; a turn already in flight owns the
    // model. Enqueuing behind either is fine, running is not — and since the
    // drain is what fires, deferring here is the whole mechanism.
    if crate::modal::is_open() || crate::shell::chat_busy() {
        return;
    }
    let Some(()) = with_busy(|| {
        LAST_TICK_MS.store(now_ms, Ordering::Relaxed);
        let now_unix = crate::clock::now_unix();
        let trusted = crate::clock::source().trusted();

        // A clock jump (NTP, `/datetime set`, a resume) invalidates every
        // calendar job's committed due time, and so does the untrusted→trusted
        // transition — that second case is the `/ntp`-at-boot path, where the
        // held jobs get their first real due time.
        let jump = CLOCK_WATCH.with(|(pu, pm)| {
            let j = spec::detect_jump(*pu, *pm, now_unix, now_ms, spec::JUMP_TOLERANCE_SECS);
            *pu = now_unix;
            *pm = now_ms;
            j
        });
        let became_trusted = trusted && !CLOCK_WAS_TRUSTED.swap(trusted, Ordering::Relaxed);
        if jump.jumped || became_trusted {
            reanchor(now_unix, jump.drift_secs, became_trusted);
        }

        // Collect ids out of the lock, then act — `Locked` is not reentrant and
        // `notify::post` takes its own lock.
        let mut due: Vec<(u64, u32)> = Vec::new();
        let mut over_capacity: Vec<u64> = Vec::new();
        let room = MAX_PENDING.saturating_sub(PENDING.with(|v| v.len()));
        JOBS.with(|v| {
            for j in v.iter_mut() {
                if !j.enabled {
                    continue;
                }
                // The monotonic re-fire floor, checked before anything else.
                if j.last_fire_ms != 0 && now_ms.saturating_sub(j.last_fire_ms) < MIN_RERUN_MS {
                    continue;
                }
                // `Every` is decided on the monotonic timebase, which is the
                // whole reason it exists: it is correct on a fictional clock.
                if let Recurrence::Every { secs } = j.spec {
                    if now_ms < j.due_ms {
                        continue;
                    }
                    j.due_ms = now_ms + (secs.max(spec::MIN_EVERY_SECS) as u64) * 1000;
                    j.next_due_unix = now_unix + secs.max(spec::MIN_EVERY_SECS);
                    if due.len() < room {
                        due.push((j.id, 0));
                    } else {
                        over_capacity.push(j.id);
                    }
                    continue;
                }
                let (verdict, next, slot) = spec::evaluate(
                    j.spec,
                    j.catchup,
                    j.enabled,
                    j.next_due_unix,
                    j.last_slot,
                    now_unix,
                    trusted,
                    &crate::clock::offset_at,
                );
                j.next_due_unix = next;
                j.last_slot = slot;
                if let Due::Now { coalesced_missed } = verdict {
                    if due.len() < room {
                        due.push((j.id, coalesced_missed));
                    } else {
                        over_capacity.push(j.id);
                    }
                }
            }
        });

        for id in over_capacity {
            // Bounded queue: say so rather than dropping silently. A schedule
            // that never runs and never complains is the worst outcome here.
            JOBS.with(|v| {
                if let Some(j) = v.iter_mut().find(|j| j.id == id) {
                    j.last_status = String::from("dropped: pending queue full");
                }
            });
            crate::ktrace::log_fmt(format_args!("schedule: dropped fire #{id} — queue full"));
        }
        for (id, missed) in due {
            if let Some(fire) = commit_fire(id, missed) {
                PENDING.with(|v| v.push(fire));
            }
        }
    }) else {
        return;
    };
}

/// Recompute every calendar job's due time after the clock moved, applying the
/// missed-run policy once and persisting the table once.
fn reanchor(now_unix: i64, drift_secs: i64, became_trusted: bool) {
    let trusted = crate::clock::source().trusted();
    let mut notes: Vec<(String, u32, Catchup)> = Vec::new();
    JOBS.with(|v| {
        for j in v.iter_mut() {
            if !spec::needs_wall_clock(j.spec) || !j.enabled {
                continue;
            }
            if !trusted {
                continue;
            }
            let (verdict, next, slot) = spec::evaluate(
                j.spec,
                j.catchup,
                j.enabled,
                j.next_due_unix,
                j.last_slot,
                now_unix,
                trusted,
                &crate::clock::offset_at,
            );
            // Only the bookkeeping is applied here; the *fire* is left to the
            // ordinary tick path so there is exactly one place that enqueues.
            if let Due::No = verdict {
                if j.next_due_unix != 0 && j.next_due_unix <= now_unix {
                    // Genuinely missed while powered off / before the clock was
                    // right. Count it for one coalesced notification.
                    notes.push((j.name.clone(), 1, j.catchup));
                }
                j.next_due_unix = next;
                j.last_slot = slot;
            }
        }
    });
    save();
    crate::ktrace::log_fmt(format_args!(
        "schedule: re-anchored (drift {drift_secs}s, became_trusted={became_trusted}); {} job(s) had missed fires",
        notes.len()
    ));
    for (name, missed, catchup) in notes {
        if matches!(catchup, Catchup::Skip) {
            // Coalesced by name, so a month off is one notification per job
            // rather than one per missed fire.
            crate::notify::post_keyed(
                crate::notify::Severity::Info,
                &alloc::format!("schedule:{name}"),
                &alloc::format!("{name}: missed run(s) skipped"),
                &alloc::format!(
                    "{missed}+ fire(s) were due while the clock was wrong or the machine was off; \
                     rescheduled forward. /schedule run {name} to run it now."
                ),
                &alloc::format!("missed:{name}"),
            );
        }
    }
}

#[cfg(test)]
pub fn reset_for_test() {
    JOBS.with(|v| v.clear());
    PENDING.with(|v| v.clear());
    NEXT_ID.store(1, Ordering::Relaxed);
    LAST_TICK_MS.store(0, Ordering::Relaxed);
    CLOCK_WATCH.with(|c| *c = (0, 0));
    BUSY.store(false, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::taint::Provenance;

    fn human() -> GrantFacts {
        GrantFacts {
            author: Author::Human,
            provenance: Provenance::UserTyped,
            human_confirmed: true,
        }
    }

    fn cmd(name: &str) -> Action {
        Action::Command { name: String::from(name), arg: String::new() }
    }

    #[test_case]
    fn add_list_remove_round_trips_through_the_live_table() {
        reset_for_test();
        let r = Recurrence::Every { secs: 60 };
        let id = add("nightly", r, cmd("pwd"), human(), 1, Catchup::Skip, NotifyOn::OnChange)
            .expect("add");
        assert_eq!(count(), 1);
        let j = get("nightly").expect("get");
        assert_eq!(j.id, id);
        assert_eq!(j.spec, r);
        assert!(j.enabled);
        assert!(j.grant.facts.human_confirmed);
        assert_ne!(j.grant.authored_hash, 0);
        assert!(remove("nightly"));
        assert!(!remove("nightly"), "removing twice is reported, not silently ok");
        assert_eq!(count(), 0);
        reset_for_test();
    }

    #[test_case]
    fn names_are_unique_bounded_and_whitespace_free() {
        reset_for_test();
        let r = Recurrence::Every { secs: 60 };
        add("a", r, cmd("pwd"), human(), 1, Catchup::Skip, NotifyOn::Never).unwrap();
        assert!(add("a", r, cmd("pwd"), human(), 1, Catchup::Skip, NotifyOn::Never).is_err());
        assert!(add("", r, cmd("pwd"), human(), 1, Catchup::Skip, NotifyOn::Never).is_err());
        assert!(
            add("two words", r, cmd("pwd"), human(), 1, Catchup::Skip, NotifyOn::Never).is_err(),
            "a name with a space could never be addressed by /schedule <verb> <name>"
        );
        for i in 0..MAX_JOBS {
            let _ = add(
                &alloc::format!("j{i}"),
                r,
                cmd("pwd"),
                human(),
                1,
                Catchup::Skip,
                NotifyOn::Never,
            );
        }
        assert_eq!(count(), MAX_JOBS, "the table is bounded");
        reset_for_test();
    }

    #[test_case]
    fn adding_a_job_does_not_fire_it_and_sets_a_future_due_time() {
        reset_for_test();
        let r = Recurrence::Daily { hour: 3, min: 0, dow_mask: 0x7f };
        add("d", r, cmd("pwd"), human(), 1, Catchup::Skip, NotifyOn::Never).unwrap();
        let j = get("d").unwrap();
        assert!(j.next_due_unix > crate::clock::now_unix());
        assert_eq!(pending_len(), 0, "adding must never enqueue a fire");
        reset_for_test();
    }

    #[test_case]
    fn pause_blocks_a_committed_fire_and_resume_re_anchors_the_interval() {
        reset_for_test();
        let r = Recurrence::Every { secs: 60 };
        let id = add("p", r, cmd("pwd"), human(), 1, Catchup::Skip, NotifyOn::Never).unwrap();
        assert_eq!(set_enabled("p", false), Some(true));
        assert_eq!(set_enabled("p", false), Some(false), "already off: no change");
        assert!(commit_fire(id, 0).is_none(), "a paused job must not commit a fire");
        // Resume pushes the monotonic due time forward, so a job paused for an
        // hour does not fire the instant it comes back.
        let before = get("p").unwrap().due_ms;
        assert_eq!(set_enabled("p", true), Some(true));
        assert!(get("p").unwrap().due_ms >= before);
        assert_eq!(set_enabled("nope", true), None, "an unknown name is None, not a panic");
        reset_for_test();
    }

    #[test_case]
    fn removing_a_job_drops_its_queued_fire() {
        reset_for_test();
        let r = Recurrence::Every { secs: 60 };
        add("q", r, cmd("pwd"), human(), 1, Catchup::Skip, NotifyOn::Never).unwrap();
        run_now("q").unwrap();
        assert_eq!(pending_len(), 1);
        assert!(remove("q"));
        assert_eq!(pending_len(), 0, "a removed schedule must not have one last run");
        reset_for_test();
    }

    #[test_case]
    fn run_now_queues_without_consuming_the_scheduled_fire() {
        reset_for_test();
        let r = Recurrence::Daily { hour: 3, min: 0, dow_mask: 0x7f };
        add("m", r, cmd("pwd"), human(), 1, Catchup::Skip, NotifyOn::Never).unwrap();
        let due_before = get("m").unwrap().next_due_unix;
        run_now("m").unwrap();
        assert_eq!(pending_len(), 1);
        assert_eq!(
            get("m").unwrap().next_due_unix,
            due_before,
            "an out-of-band run must not consume the scheduled slot"
        );
        let f = take_pending().unwrap();
        assert_eq!(f.name, "m");
        assert_eq!(f.notify, NotifyOn::Always, "an explicit run always reports");
        assert!(run_now("absent").is_err());
        reset_for_test();
    }

    #[test_case]
    fn the_pending_queue_is_bounded() {
        reset_for_test();
        let r = Recurrence::Every { secs: 60 };
        add("b", r, cmd("pwd"), human(), 1, Catchup::Skip, NotifyOn::Never).unwrap();
        for _ in 0..(MAX_PENDING + 5) {
            let _ = run_now("b");
        }
        assert_eq!(pending_len(), MAX_PENDING);
        reset_for_test();
    }

    #[test_case]
    fn record_result_counts_runs_detects_change_and_retires_a_once_job() {
        reset_for_test();
        let once = Recurrence::Once { at_unix: crate::clock::now_unix() + 3600 };
        let id = add("o", once, cmd("pwd"), human(), 1, Catchup::Skip, NotifyOn::OnChange).unwrap();
        assert!(record_result(id, "ok", "output A", 1_770_000_000), "first output is a change");
        assert!(!record_result(id, "ok", "output A", 1_770_000_060), "same output: no change");
        assert!(record_result(id, "ok", "output B", 1_770_000_120));
        let j = get("o").unwrap();
        assert_eq!(j.run_count, 3);
        assert_eq!(j.last_status, "ok");
        assert!(!j.enabled, "a `once` schedule retires itself after running");
        reset_for_test();
    }

    #[test_case]
    fn json_round_trips_a_job_including_its_authority() {
        reset_for_test();
        let r = spec::parse("at 03:00 mon,thu", crate::clock::now_unix()).unwrap();
        let facts = GrantFacts {
            author: Author::Agent,
            provenance: Provenance::UntrustedIngested,
            human_confirmed: false,
        };
        add(
            "j",
            r,
            Action::Command { name: String::from("disks"), arg: String::from("-l") },
            facts,
            9042,
            Catchup::Once,
            NotifyOn::OnError,
        )
        .unwrap();
        add(
            "p",
            Recurrence::Every { secs: 300 },
            Action::Prompt { text: String::from("summarise the day") },
            human(),
            1,
            Catchup::Skip,
            NotifyOn::Always,
        )
        .unwrap();
        let text = JOBS.with(|v| Json::Arr(v.iter().map(job_json).collect()).to_pretty());
        // Reload over a cleared table, the way `load` does at boot.
        JOBS.with(|v| v.clear());
        store::write(CONFIG_PATH, text.as_bytes());
        load();
        let j = get("j").expect("job survived the round trip");
        assert_eq!(j.spec, r);
        assert_eq!(j.action, Action::Command { name: String::from("disks"), arg: String::from("-l") });
        assert_eq!(j.grant.facts.author, Author::Agent);
        assert_eq!(
            j.grant.facts.provenance,
            Provenance::UntrustedIngested,
            "a tainted authority record must survive a reboot"
        );
        assert!(!j.grant.facts.human_confirmed);
        assert_eq!(j.grant.agent_id, 9042);
        assert_eq!(j.catchup, Catchup::Once);
        assert_eq!(j.notify, NotifyOn::OnError);
        let p = get("p").expect("prompt job survived");
        assert_eq!(p.action, Action::Prompt { text: String::from("summarise the day") });
        reset_for_test();
    }

    #[test_case]
    fn a_record_with_no_provenance_loads_as_tainted_not_as_clean() {
        reset_for_test();
        // The safe default matters: an unreadable or hand-written authority
        // record must not read as a human-typed one, or editing the file by hand
        // becomes an escalation.
        let text = r#"[{"name":"x","recur":"every 60s","kind":"command","command":"pwd"}]"#;
        store::write(CONFIG_PATH, text.as_bytes());
        load();
        let j = get("x").expect("loaded");
        assert_eq!(j.grant.facts.provenance, Provenance::UntrustedIngested);
        assert!(!j.grant.facts.human_confirmed);
        assert!(spec::grant_justification(j.grant.facts).blocks_destructive());
        reset_for_test();
    }

    #[test_case]
    fn a_job_whose_recurrence_no_longer_parses_is_dropped_not_defaulted() {
        reset_for_test();
        let text = r#"[
          {"name":"good","recur":"every 60s","kind":"command","command":"pwd"},
          {"name":"bad","recur":"whenever i feel like it","kind":"command","command":"rm"}
        ]"#;
        store::write(CONFIG_PATH, text.as_bytes());
        load();
        assert!(get("good").is_some());
        assert!(
            get("bad").is_none(),
            "an unparseable recurrence must not silently become some interval"
        );
        reset_for_test();
    }

    #[test_case]
    fn duplicate_names_in_a_hand_edited_file_load_once() {
        reset_for_test();
        let text = r#"[
          {"name":"dup","recur":"every 60s","kind":"command","command":"pwd"},
          {"name":"dup","recur":"every 90s","kind":"command","command":"disks"}
        ]"#;
        store::write(CONFIG_PATH, text.as_bytes());
        load();
        assert_eq!(count(), 1, "a duplicate name would make /schedule ambiguous");
        assert_eq!(get("dup").unwrap().spec, Recurrence::Every { secs: 60 }, "first wins");
        reset_for_test();
    }

    #[test_case]
    fn the_authored_hash_changes_with_the_action_and_the_recurrence() {
        let a = author_hash("j", Recurrence::Every { secs: 60 }, &cmd("pwd"));
        let b = author_hash("j", Recurrence::Every { secs: 61 }, &cmd("pwd"));
        let c = author_hash("j", Recurrence::Every { secs: 60 }, &cmd("rm"));
        let d = author_hash("k", Recurrence::Every { secs: 60 }, &cmd("pwd"));
        assert_ne!(a, b, "the recurrence is part of what was authorised");
        assert_ne!(a, c, "so is the action");
        assert_ne!(a, d, "so is the name");
        assert_eq!(a, author_hash("j", Recurrence::Every { secs: 60 }, &cmd("pwd")));
    }

    #[test_case]
    fn a_reentrant_tick_is_refused_rather_than_deadlocking() {
        reset_for_test();
        // `with_busy` is what stops tick → run → /http → upkeep → tick from
        // re-entering a non-reentrant `Locked` and stopping the machine.
        let inner = with_busy(|| with_busy(|| 1));
        assert_eq!(inner, Some(None), "the inner entry must be refused, not block");
        assert_eq!(with_busy(|| 2), Some(2), "and the guard must be released after");
        reset_for_test();
    }
}
