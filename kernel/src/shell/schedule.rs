//! schedule
//!
//! The `/schedule` command surface over [`crate::schedule`], and the **drain** —
//! the half that actually runs a due job.
//!
//! The drain lives here, in the shell, rather than in `schedule::tick`, for the
//! reason `shell::fs::drain_channel_inbound` records: inference is too heavy for
//! the poll tick. It is called from the interactive loop, which also gets a
//! 1 MiB stack for a model turn for free (`ChatSession::turn` wraps itself in
//! one), answers Ctrl+C in the same loop as the work, and cannot race a modal.

use super::*;
use crate::schedule::spec::{self, Author, Catchup, GrantFacts, NotifyOn, Recurrence};
use crate::schedule::{Action, Fire};

/// Fires run per pass through the interactive loop. Bounded so the prompt stays
/// responsive; `msgchan` uses 3 for the same reason.
const MAX_PER_PASS: usize = 2;

// ---------------------------------------------------------------------------
// The command
// ---------------------------------------------------------------------------

/// `/schedule [list|add|show|run|pause|resume|remove|catchup|notify|next]`
pub(super) fn run_schedule_cmd(arg: &str) {
    let a = arg.trim();
    let mut parts = a.splitn(2, char::is_whitespace);
    let verb = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("").trim();

    match verb {
        "" | "list" | "ls" => list(),
        "add" | "new" => add(rest),
        "show" | "info" => show(rest),
        "run" => run_one(rest),
        "pause" | "disable" | "off" => enable(rest, false),
        "resume" | "enable" | "on" => enable(rest, true),
        "remove" | "rm" | "delete" | "del" => {
            if crate::schedule::remove(rest) {
                serial_println!("schedule> removed '{rest}'");
            } else {
                serial_println!("schedule> no schedule called '{rest}'");
            }
        }
        "catchup" => set_catchup(rest),
        "notify" => set_notify(rest),
        "next" => next(),
        other => {
            serial_println!("schedule> unknown '{other}'");
            usage();
        }
    }
}

fn usage() {
    serial_println!("schedule> usage:");
    serial_println!("  /schedule list");
    serial_println!("  /schedule add <name> <recurrence> command <cmd> [args…]");
    serial_println!("  /schedule add <name> <recurrence> prompt <text…>");
    serial_println!("      recurrence: 'every <n>(s|m|h|d)' | 'at HH:MM [daily|weekdays|mon,wed]'");
    serial_println!("                  | 'on <1-31> HH:MM' | 'in <n><unit>' | 'once <unix|ISO>'");
    serial_println!("  /schedule show|run|pause|resume|remove <name>");
    serial_println!("  /schedule catchup <name> skip|once");
    serial_println!("  /schedule notify <name> always|on_change|on_error|never");
    serial_println!("  /schedule next");
}

fn list() {
    let jobs = crate::schedule::list();
    if jobs.is_empty() {
        serial_println!("schedule> (no schedules) — /schedule add <name> <recurrence> command <cmd>");
        return;
    }
    let now = crate::clock::now_unix();
    let trusted = crate::clock::source().trusted();
    serial_println!("schedule> {} schedule(s):", jobs.len());
    for j in &jobs {
        let state = if !j.enabled {
            String::from("paused")
        } else if spec::held_for_clock(j.spec, trusted) {
            String::from("held (clock)")
        } else if j.next_due_unix > 0 {
            let d = j.next_due_unix - now;
            if d <= 0 {
                String::from("due")
            } else if d < 3600 {
                alloc::format!("in {}m", (d + 59) / 60)
            } else if d < 86400 {
                alloc::format!("in {}h", d / 3600)
            } else {
                alloc::format!("in {}d", d / 86400)
            }
        } else {
            String::from("-")
        };
        serial_println!(
            "  {:<14} {:<24} {:<12} runs={:<4} {} [{}]",
            j.name,
            spec::render(j.spec),
            state,
            j.run_count,
            if j.last_status.is_empty() { "-" } else { j.last_status.as_str() },
            j.grant.facts.author.as_str()
        );
        serial_println!("      {}", j.action.summary());
    }
}

fn show(name: &str) {
    let Some(j) = crate::schedule::get(name) else {
        serial_println!("schedule> no schedule called '{name}'");
        return;
    };
    // The grant, printed verbatim: "who authorised this recurring effect" must be
    // answerable without reading JSON.
    serial_println!("schedule> {}", j.name);
    serial_println!("  id           {}", j.id);
    serial_println!("  recurrence   {}", spec::render(j.spec));
    serial_println!("  action       {}", j.action.summary());
    serial_println!("  enabled      {}", j.enabled);
    serial_println!("  catchup      {}", j.catchup.as_str());
    serial_println!("  notify       {}", j.notify.as_str());
    serial_println!("  --- authority (frozen at creation) ---");
    serial_println!("  author       {}", j.grant.facts.author.as_str());
    serial_println!(
        "  provenance   {}",
        match j.grant.facts.provenance {
            crate::security::taint::Provenance::SystemTrusted => "system_trusted",
            crate::security::taint::Provenance::UserTyped => "user_typed",
            crate::security::taint::Provenance::UntrustedIngested => "untrusted_ingested",
        }
    );
    serial_println!("  confirmed    {}", j.grant.facts.human_confirmed);
    serial_println!("  runs as      agent {}", j.grant.agent_id);
    serial_println!("  authored     {:016x}", j.grant.authored_hash);
    serial_println!("  created      {}", j.grant.created_unix);
    let jt = spec::grant_justification(j.grant.facts);
    serial_println!(
        "  effects      {}",
        if jt.blocks_destructive() {
            "inert only — destructive/egress calls are refused (tainted author)"
        } else {
            "permitted (subject to the usual gates)"
        }
    );
    serial_println!("  --- history ---");
    serial_println!("  runs         {}", j.run_count);
    serial_println!("  last status  {}", if j.last_status.is_empty() { "-" } else { &j.last_status });
    serial_println!("  last run     {}", j.last_run_unix);
    serial_println!("  next due     {}", j.next_due_unix);
}

/// `/schedule next` — what fires next, **and why not** when nothing can.
///
/// This row earns its place: on a machine with no readable RTC the honest answer
/// is "3 calendar jobs are held because the clock is a guess", and without a
/// command that says so, "my schedule didn't run" has three indistinguishable
/// causes on a box you may not be able to attach a debugger to.
fn next() {
    let jobs = crate::schedule::list();
    let trusted = crate::clock::source().trusted();
    let held = crate::schedule::held_count();
    if held > 0 {
        serial_println!(
            "schedule> {held} calendar job(s) HELD — the clock source is '{}' and cannot be trusted.",
            crate::clock::source().as_str()
        );
        serial_println!("schedule> set it with /ntp or /datetime; 'every …' jobs run regardless.");
    }
    let now = crate::clock::now_unix();
    let mut soonest: Option<(i64, String)> = None;
    for j in &jobs {
        if !j.enabled || spec::held_for_clock(j.spec, trusted) || j.next_due_unix <= 0 {
            continue;
        }
        if soonest.as_ref().map(|(t, _)| j.next_due_unix < *t).unwrap_or(true) {
            soonest = Some((j.next_due_unix, j.name.clone()));
        }
    }
    match soonest {
        Some((t, name)) => {
            let d = (t - now).max(0);
            serial_println!(
                "schedule> next: '{name}' in {}s ({})",
                d,
                crate::clock::format_datetime_short_at(t, crate::clock::tz_offset())
            );
        }
        None if held == 0 => serial_println!("schedule> nothing scheduled to run"),
        None => {}
    }
    if crate::schedule::pending_len() > 0 {
        serial_println!("schedule> {} fire(s) queued to run", crate::schedule::pending_len());
    }
}

fn enable(name: &str, on: bool) {
    match crate::schedule::set_enabled(name, on) {
        Some(true) => serial_println!("schedule> '{name}' {}", if on { "resumed" } else { "paused" }),
        Some(false) => {
            serial_println!("schedule> '{name}' was already {}", if on { "running" } else { "paused" })
        }
        None => serial_println!("schedule> no schedule called '{name}'"),
    }
}

fn set_catchup(rest: &str) {
    let mut it = rest.split_whitespace();
    let name = it.next().unwrap_or("");
    let val = it.next().unwrap_or("");
    let Some(c) = Catchup::parse(val) else {
        serial_println!("schedule> usage: /schedule catchup <name> skip|once");
        return;
    };
    if crate::schedule::set_catchup(name, c) {
        serial_println!("schedule> '{name}' catchup = {}", c.as_str());
    } else {
        serial_println!("schedule> no schedule called '{name}'");
    }
}

fn set_notify(rest: &str) {
    let mut it = rest.split_whitespace();
    let name = it.next().unwrap_or("");
    let val = it.next().unwrap_or("");
    let Some(n) = NotifyOn::parse(val) else {
        serial_println!("schedule> usage: /schedule notify <name> always|on_change|on_error|never");
        return;
    };
    if crate::schedule::set_notify(name, n) {
        serial_println!("schedule> '{name}' notify = {}", n.as_str());
    } else {
        serial_println!("schedule> no schedule called '{name}'");
    }
}

fn run_one(name: &str) {
    match crate::schedule::run_now(name) {
        Ok(()) => serial_println!("schedule> '{name}' queued — it runs at the next prompt"),
        Err(e) => serial_println!("schedule> {e}"),
    }
}

/// Split `<name> <recurrence…> (command|prompt) <rest…>`.
///
/// The recurrence is variable-length, so the split point is the `command` /
/// `prompt` keyword rather than a token count — which is also what lets
/// `at 09:00 mon,wed` and `every 5m` share one grammar.
fn split_add(rest: &str) -> Result<(String, String, Action), String> {
    let toks: alloc::vec::Vec<&str> = rest.split_whitespace().collect();
    if toks.len() < 4 {
        return Err(String::from("need <name> <recurrence> command|prompt <…>"));
    }
    let name = toks[0].to_string();
    let kw = toks
        .iter()
        .position(|t| *t == "command" || *t == "cmd" || *t == "prompt" || *t == "ask")
        .ok_or_else(|| String::from("missing 'command' or 'prompt' keyword"))?;
    if kw < 2 {
        return Err(String::from("the recurrence is missing"));
    }
    let recur = toks[1..kw].join(" ");
    let tail = toks[kw + 1..].join(" ");
    if tail.is_empty() {
        return Err(alloc::format!("'{}' needs something to run", toks[kw]));
    }
    let action = if toks[kw] == "prompt" || toks[kw] == "ask" {
        Action::Prompt { text: tail }
    } else {
        let mut cs = tail.splitn(2, char::is_whitespace);
        let cmd = cs.next().unwrap_or("").trim_start_matches('/').to_string();
        Action::Command { name: cmd, arg: cs.next().unwrap_or("").trim().to_string() }
    };
    Ok((name, recur, action))
}

fn add(rest: &str) {
    if rest.is_empty() {
        usage();
        return;
    }
    let (name, recur, action) = match split_add(rest) {
        Ok(v) => v,
        Err(e) => {
            serial_println!("schedule> {e}");
            usage();
            return;
        }
    };
    let spec_r = match spec::parse(&recur, crate::clock::now_unix()) {
        Ok(r) => r,
        Err(e) => {
            serial_println!("schedule> {e}");
            return;
        }
    };
    // A `Command` action naming a command that does not exist would be a
    // schedule that fails forever. Refuse at creation, where a human is looking.
    if let Action::Command { name: cmd, .. } = &action {
        if !crate::shell::catalog::is_command_name(cmd) {
            serial_println!("schedule> '/{cmd}' is not a command (see /help)");
            return;
        }
    }

    // **Human or agent is decided here, not inferred from session taint.** A
    // Telegram DM enters the session as `UserTyped`, so trusting taint alone
    // would let a DM-authored schedule act with typed-human authority forever.
    // `in_tool_call()` is true exactly when `run_tool_command` is on the stack,
    // i.e. when a model chose this call rather than a human typing it.
    let by_agent = crate::shell::in_tool_call();
    let agent_id = active_agent_id();
    let facts = if by_agent {
        GrantFacts {
            author: Author::Agent,
            provenance: crate::shell::resident_taint(),
            // Never. An agent cannot confirm on a human's behalf, and there is
            // no human present to ask at fire time.
            human_confirmed: false,
        }
    } else {
        GrantFacts {
            author: Author::Human,
            provenance: crate::security::taint::Provenance::UserTyped,
            human_confirmed: true,
        }
    };

    let notify = match &action {
        // A prompt run's output is prose that differs every time, so
        // `on_change` would mean "always" with extra steps.
        Action::Prompt { .. } => NotifyOn::Always,
        Action::Command { .. } => NotifyOn::OnChange,
    };
    let summary = action.summary();
    match crate::schedule::add(&name, spec_r, action, facts, agent_id, Catchup::Skip, notify) {
        Ok(id) => {
            serial_println!(
                "schedule> added '{name}' #{id}: {} → {summary}",
                spec::render(spec_r)
            );
            if by_agent {
                serial_println!("schedule> author=agent:{agent_id}, not human-confirmed");
                if facts.provenance.is_tainted() {
                    serial_println!(
                        "schedule> NOTE: authored while untrusted content was in context — \
                         destructive and egress calls from this job will be refused"
                    );
                }
            }
            if spec::needs_wall_clock(spec_r) && !crate::clock::source().trusted() {
                serial_println!(
                    "schedule> the clock source is '{}' — this job is HELD until /ntp or /datetime",
                    crate::clock::source().as_str()
                );
            }
        }
        Err(e) => serial_println!("schedule> {e}"),
    }
}

// ---------------------------------------------------------------------------
// The drain
// ---------------------------------------------------------------------------

/// Run the committed fires. Called from the interactive loop, once per pass.
pub(super) fn drain_schedule_pending(
    chat: &mut Option<ChatSession>,
    session: &mut crate::agent::types::Session,
) {
    for _ in 0..MAX_PER_PASS {
        // The Ctrl+C rule, in the same loop as the work.
        if crate::shell::poll_interrupt() {
            break;
        }
        let Some(fire) = crate::schedule::take_pending() else {
            break;
        };
        run_fire(fire, chat, session);
    }
}

fn run_fire(
    fire: Fire,
    chat: &mut Option<ChatSession>,
    session: &mut crate::agent::types::Session,
) {
    let started = crate::clock::now_unix();
    serial_println!("schedule> firing '{}': {}", fire.name, fire.action.summary());
    if fire.coalesced_missed > 0 {
        serial_println!(
            "schedule> (standing in for {} missed fire(s))",
            fire.coalesced_missed
        );
    }

    // Everything the fire does happens with no human at the keyboard. Set the
    // flag around the whole run so `execute_chat_tool_inner` refuses an
    // approval-requiring call and posts a notification instead of blocking on a
    // modal nobody will answer.
    crate::shell::set_unattended(true);
    let (status, output) = match &fire.action {
        Action::Command { name, arg } => {
            let out = crate::shell::run_tool_command(name, arg);
            // A command's own error text is its status; there is no exit code to
            // read, so "did it work" is decided by the same words a human would
            // read.
            let bad = out.contains("is not available as a tool")
                || out.to_ascii_lowercase().contains("error")
                || out.to_ascii_lowercase().contains("failed");
            (if bad { String::from("err") } else { String::from("ok") }, out)
        }
        Action::Prompt { text } => run_prompt(&fire, text, chat, session),
    };
    crate::shell::set_unattended(false);

    let cancelled = status == "cancelled";
    let changed = crate::schedule::record_result(fire.id, &status, &output, started);

    // The at-most-once contract, stated where it bites: the slot was consumed
    // when the fire was committed, so a cancelled run does not retry. That is
    // deliberate, and it is why the status is recorded rather than swallowed.
    if cancelled {
        serial_println!("schedule> '{}' cancelled — it will not retry", fire.name);
    }

    let want = match fire.notify {
        NotifyOn::Never => false,
        NotifyOn::Always => true,
        NotifyOn::OnError => status != "ok",
        NotifyOn::OnChange => status != "ok" || changed,
    };
    if want {
        let sev = if status == "ok" {
            crate::notify::Severity::Success
        } else if cancelled {
            crate::notify::Severity::Warn
        } else {
            crate::notify::Severity::Error
        };
        let body = if output.trim().is_empty() {
            String::from("(no output)")
        } else {
            output.trim().to_string()
        };
        crate::notify::post_action(
            sev,
            &alloc::format!("schedule:{}", fire.name),
            &alloc::format!("{}: {}", fire.name, status),
            &body,
            &alloc::format!("/schedule show {}", fire.name),
            // Coalesced per job, so a five-second job cannot fill the ring: the
            // latest result replaces the previous one and bumps a count.
            &alloc::format!("schedule:{}", fire.name),
        );
    }
}

/// One bounded agent turn for a `prompt` schedule.
fn run_prompt(
    fire: &Fire,
    text: &str,
    chat: &mut Option<ChatSession>,
    session: &mut crate::agent::types::Session,
) -> (String, String) {
    if !crate::shell::planner_available() {
        // One coalesced notification, not one per fire: a machine with no model
        // running a five-minute prompt job would otherwise post 288 a day.
        crate::notify::post_keyed(
            crate::notify::Severity::Warn,
            &alloc::format!("schedule:{}", fire.name),
            &alloc::format!("{}: no model loaded", fire.name),
            "This schedule needs a model. Load one with /model load <file.gguf> or /model remote <url>.",
            &alloc::format!("noplanner:{}", fire.name),
        );
        return (String::from("skipped: no planner"), String::new());
    }
    if chat.is_none() {
        let mut spin = Spinner::new("schedule");
        *chat = ChatSession::load(&mut spin);
        if let Some(c) = chat.as_mut() {
            c.hydrate_from_session(session);
        }
    }
    let Some(sess) = chat.as_mut() else {
        return (String::from("skipped: no planner"), String::new());
    };
    // Framed the way `drain_channel_inbound` frames a DM: a small model must
    // stay on *this* task rather than continuing the console's topic, and must
    // call tools for machine state instead of inventing it.
    let user = alloc::format!(
        "This is a scheduled task ('{}'), running unattended — no human is at the console.\n\
         Do ONLY the task below. Do not continue an earlier conversation.\n\
         If it needs machine state (disks, files, network, time), call the right tool first; never invent those facts.\n\
         Reply in one short plain-text paragraph (no markdown) suitable for a notification.\n\
         \n\
         Task:\n{}",
        fire.name, text
    );
    let reply = sess.turn(&user, session);
    let reply = crate::shell::strip_md_light(reply.trim());
    let reply = reply.trim().to_string();
    if reply.is_empty() {
        (String::from("err: empty reply"), String::new())
    } else {
        (String::from("ok"), reply)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn add_splits_on_the_keyword_not_a_token_count() {
        // A one-word recurrence…
        let (n, r, a) = split_add("nightly every 5m command pwd").unwrap();
        assert_eq!(n, "nightly");
        assert_eq!(r, "every 5m");
        assert_eq!(a, Action::Command { name: String::from("pwd"), arg: String::new() });
        // …a three-word one, which is why the split cannot be positional.
        let (_, r, a) = split_add("w at 09:00 weekdays command disks -l").unwrap();
        assert_eq!(r, "at 09:00 weekdays");
        assert_eq!(a, Action::Command { name: String::from("disks"), arg: String::from("-l") });
        // A leading slash on the command is accepted and stripped, because that
        // is how a human types a command name.
        let (_, _, a) = split_add("s every 1h command /disks").unwrap();
        assert_eq!(a, Action::Command { name: String::from("disks"), arg: String::new() });
        // A prompt keeps its whole tail, spaces and all.
        let (_, r, a) = split_add("d at 08:00 daily prompt summarise the disks and the network")
            .unwrap();
        assert_eq!(r, "at 08:00 daily");
        assert_eq!(
            a,
            Action::Prompt { text: String::from("summarise the disks and the network") }
        );
    }

    #[test_case]
    fn add_refuses_a_malformed_invocation_with_a_reason() {
        for bad in [
            "",
            "name",
            "name every 5m",
            "name every 5m command",
            "name command pwd",     // no recurrence
            "name every 5m run pwd", // no command/prompt keyword
        ] {
            let r = split_add(bad);
            assert!(r.is_err(), "'{bad}' should be refused, got {r:?}");
            assert!(!r.unwrap_err().is_empty());
        }
    }

    #[test_case]
    fn cmd_and_ask_are_accepted_as_aliases() {
        let (_, _, a) = split_add("x every 5m cmd pwd").unwrap();
        assert!(matches!(a, Action::Command { .. }));
        let (_, _, a) = split_add("x every 5m ask what time is it").unwrap();
        assert_eq!(a, Action::Prompt { text: String::from("what time is it") });
    }
}
