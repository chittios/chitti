//! Background shell jobs + monitor ticks for the agent tool layer.
//!
//! Cooperative: work is pumped from [`pump`] (called by `shell::upkeep`). There
//! is no host OS process — a "background" job is a deferred
//! [`crate::shell::run_tool_command`] (or a periodic monitor re-run). The model
//! receives a `task_id` and later polls with `task_output` / `kill_task`.

use crate::mm::Locked;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const MAX_JOBS: usize = 8;
const MAX_OUTPUT: usize = 16 * 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Shell,
    Monitor,
}

struct Job {
    id: u64,
    kind: Kind,
    /// First token = system command name (no leading slash).
    cmd: String,
    args: String,
    output: String,
    done: bool,
    killed: bool,
    /// Monitor interval (ms); 0 for one-shot shell jobs.
    interval_ms: u64,
    last_run_ms: u64,
    /// Shell jobs start pending until the first pump.
    started: bool,
}

static JOBS: Locked<Vec<Job>> = Locked::new(Vec::new());
static NEXT_ID: Locked<u64> = Locked::new(1);

fn append_out(job: &mut Job, chunk: &str) {
    if job.output.len() >= MAX_OUTPUT {
        return;
    }
    let room = MAX_OUTPUT - job.output.len();
    let take = chunk.len().min(room);
    job.output.push_str(&chunk[..take]);
    if take < chunk.len() {
        job.output.push_str("\n…[truncated]");
    }
}

/// Spawn a one-shot background shell command. Returns `task_id`.
pub fn spawn_shell(cmd: &str, args: &str) -> u64 {
    let id = NEXT_ID.with(|n| {
        let id = *n;
        *n = n.saturating_add(1);
        id
    });
    JOBS.with(|jobs| {
        if jobs.len() >= MAX_JOBS {
            // Drop oldest finished job if full.
            if let Some(i) = jobs.iter().position(|j| j.done || j.killed) {
                jobs.remove(i);
            } else if !jobs.is_empty() {
                jobs.remove(0);
            }
        }
        jobs.push(Job {
            id,
            kind: Kind::Shell,
            cmd: cmd.to_string(),
            args: args.to_string(),
            output: String::new(),
            done: false,
            killed: false,
            interval_ms: 0,
            last_run_ms: 0,
            started: false,
        });
    });
    id
}

/// Start a monitor: re-run `cmd args` every `interval_ms` (min 1000).
pub fn spawn_monitor(cmd: &str, args: &str, interval_ms: u64) -> u64 {
    let id = spawn_shell(cmd, args);
    JOBS.with(|jobs| {
        if let Some(j) = jobs.iter_mut().find(|j| j.id == id) {
            j.kind = Kind::Monitor;
            j.interval_ms = interval_ms.max(1000);
            j.started = false;
        }
    });
    id
}

/// Poll job status. `wait_ms == 0` = non-blocking snapshot.
pub fn task_output(id: u64, wait_ms: u64) -> String {
    let deadline = crate::arch::now_ms().saturating_add(wait_ms);
    loop {
        pump();
        let snap = JOBS.with(|jobs| {
            jobs.iter().find(|j| j.id == id).map(|j| {
                (
                    j.done,
                    j.killed,
                    j.kind == Kind::Monitor,
                    j.output.clone(),
                    j.cmd.clone(),
                    j.args.clone(),
                )
            })
        });
        let Some((done, killed, mon, out, cmd, args)) = snap else {
            return format!("error: unknown task_id {id}");
        };
        if done || killed || wait_ms == 0 {
            let status = if killed {
                "killed"
            } else if done && !mon {
                "done"
            } else if mon {
                "running"
            } else {
                "pending"
            };
            return format!(
                "ok:task={id} status={status} cmd=/{cmd} {args}\n{out}"
            );
        }
        if crate::arch::now_ms() >= deadline {
            return format!(
                "ok:task={id} status=timeout cmd=/{cmd} {args}\n{out}"
            );
        }
        crate::shell::upkeep();
        crate::sched::yield_now();
    }
}

pub fn kill_task(id: u64) -> String {
    JOBS.with(|jobs| {
        if let Some(j) = jobs.iter_mut().find(|j| j.id == id) {
            j.killed = true;
            j.done = true;
            append_out(j, "\n[killed]");
            format!("ok:killed {id}")
        } else {
            format!("error: unknown task_id {id}")
        }
    })
}

pub fn list_tasks() -> String {
    JOBS.with(|jobs| {
        if jobs.is_empty() {
            return String::from("ok:(no background tasks)");
        }
        let mut out = String::from("ok:\n");
        for j in jobs.iter() {
            let st = if j.killed {
                "killed"
            } else if j.done && j.kind == Kind::Shell {
                "done"
            } else {
                "running"
            };
            out.push_str(&format!(
                "  #{id} {st} /{cmd} {args}\n",
                id = j.id,
                cmd = j.cmd,
                args = j.args
            ));
        }
        out
    })
}

/// Advance pending work (one shell start or due monitor tick per call).
pub fn pump() {
    let now = crate::arch::now_ms();
    // Pick one job to run outside the lock (run_tool_command may re-enter).
    let work: Option<(u64, String, String, Kind)> = JOBS.with(|jobs| {
        for j in jobs.iter_mut() {
            if j.killed || (j.done && j.kind == Kind::Shell) {
                continue;
            }
            match j.kind {
                Kind::Shell if !j.started => {
                    j.started = true;
                    return Some((j.id, j.cmd.clone(), j.args.clone(), Kind::Shell));
                }
                Kind::Monitor => {
                    if !j.started || now.saturating_sub(j.last_run_ms) >= j.interval_ms {
                        j.started = true;
                        j.last_run_ms = now;
                        return Some((j.id, j.cmd.clone(), j.args.clone(), Kind::Monitor));
                    }
                }
                _ => {}
            }
        }
        None
    });
    let Some((id, cmd, args, kind)) = work else {
        return;
    };
    let result = crate::shell::run_tool_command(&cmd, &args);
    JOBS.with(|jobs| {
        if let Some(j) = jobs.iter_mut().find(|j| j.id == id) {
            if j.killed {
                return;
            }
            if kind == Kind::Monitor {
                append_out(j, &format!("\n--- tick {} ---\n", crate::arch::now_ms()));
            }
            append_out(j, &result);
            if kind == Kind::Shell {
                j.done = true;
            }
        }
    });
}
