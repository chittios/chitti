//! `/passwd` and `/lock` — the human-facing half of [`crate::auth`].
//!
//! **Both are interactive-only commands**, matched in the REPL above the
//! `dispatch_system` fall-through rather than inside it. That is a security
//! property, not a filing decision: `run_shell_command` hands any
//! `[A-Za-z0-9_-]+` token straight to `dispatch_system` without needing a
//! registry entry, and it is in `CORE_TOOLS` and the orchestrator manifest — so a
//! `dispatch_system` arm would be callable by every agent, with no capability and
//! no manifest change. `dispatch_system` has exactly one agent-reachable caller
//! (`run_tool_command`) and every agent surface funnels through it, so keeping
//! these names out of it closes all of them at once.
//!
//! The `in_tool_call()` refusals below are therefore defence in depth, not the
//! boundary. Keep both.

use crate::serial_println;

/// `/lock` — lock the console now.
pub(super) fn run_lock(_arg: &str) {
    if crate::shell::in_tool_call() {
        serial_println!("lock> refused: only a human at the console may lock it");
        return;
    }
    if !crate::auth::enrolled() {
        serial_println!("lock> no password set — /passwd first (there would be no way back in)");
        return;
    }
    // Requested rather than performed here: the REPL performs it, so the gate is
    // never entered from under a command's stack. Same reason the power button is
    // acted on in the loop rather than in the driver poll.
    crate::auth::request_lock(crate::auth::Reason::Manual);
    serial_println!("lock> locking…");
}

/// `/passwd [status|set|clear|idle <n|off>|resume on|off]`
pub(super) fn run_passwd(arg: &str) {
    if crate::shell::in_tool_call() {
        serial_println!("passwd> refused: the login password is set by a human at the console, never by a tool");
        return;
    }
    let mut it = arg.split_whitespace();
    match it.next().unwrap_or("status") {
        "status" => status(),
        "set" | "new" | "change" => set(arg),
        "clear" | "remove" | "off" => clear(arg),
        "idle" => idle(it.next().unwrap_or("")),
        "resume" => resume(it.next().unwrap_or("")),
        other => {
            serial_println!("passwd> unknown subcommand '{other}'");
            usage();
        }
    }
}

fn usage() {
    serial_println!("passwd> usage: /passwd [status | set | clear | idle <minutes|off> | resume on|off]");
    serial_println!("passwd>        set/clear take --yes to skip the confirmation, and set takes");
    serial_println!("passwd>        --new <password> for automation (it lands in the scrollback — prefer the prompt)");
}

fn status() {
    let Some(rec) = crate::auth::store::load() else {
        if crate::auth::store::exists() {
            serial_println!("passwd> a credential record exists but does not parse — login is DISABLED");
            serial_println!("passwd> recover by booting another OS and deleting {}", crate::auth::PATH);
        } else {
            serial_println!("passwd> not set — this machine does not ask for a password");
        }
        return;
    };
    serial_println!("passwd> user:        {}", rec.user);
    serial_println!("passwd> kdf:         {} ({} iterations)", rec.kdf, rec.iterations);
    serial_println!(
        "passwd> auto-lock:   {}",
        if rec.idle_lock_minutes == 0 {
            alloc::string::String::from("disabled")
        } else {
            alloc::format!("after {} minute(s) idle", rec.idle_lock_minutes)
        }
    );
    serial_println!("passwd> on resume:   {}", if rec.lock_on_resume { "locked" } else { "not locked" });
    serial_println!("passwd> failures:    {} lifetime, {} this boot", rec.failed_total, crate::auth::failures());
    serial_println!(
        "passwd> store:       {} ({})",
        crate::synapse::fs::backend_name(),
        if crate::synapse::fs::is_durable() { "durable" } else { "NOT durable — lost on reboot" }
    );
    // The honest caveat, printed where it is relevant rather than buried in docs.
    serial_println!("passwd> NB this is a console lock, not confidentiality: on an unencrypted");
    serial_println!("passwd>    disk it can be bypassed offline. /encrypt is what protects the data.");
}

/// Pull `--new <value>` out of the argument line, if present.
fn flag_value(arg: &str, flag: &str) -> Option<alloc::string::String> {
    let mut it = arg.split_whitespace();
    while let Some(t) = it.next() {
        if t == flag {
            return it.next().map(alloc::string::String::from);
        }
    }
    None
}

fn has_flag(arg: &str, flag: &str) -> bool {
    arg.split_whitespace().any(|t| t == flag)
}

fn set(arg: &str) {
    let assume_yes = has_flag(arg, "--yes");
    // `--new` exists so the e2e harness (which drives this over serial, where a
    // framebuffer modal is invisible) can enrol a password. It is second-class on
    // purpose: the password ends up in the scrollback and the serial log.
    let supplied = flag_value(arg, "--new");

    let pw = match supplied {
        Some(p) => p,
        None => {
            let first = crate::modal::input("Set login password", "New password:", true);
            if first.is_empty() {
                serial_println!("passwd> cancelled");
                return;
            }
            let again = crate::modal::input("Set login password", "Repeat:", true);
            if first != again {
                serial_println!("passwd> passwords do not match");
                return;
            }
            first
        }
    };

    if let Err(why) = crate::auth::validate_new(&pw) {
        serial_println!("passwd> refused: {why}");
        return;
    }

    let replacing = crate::auth::enrolled();
    if replacing && !assume_yes && !crate::modal::confirm("Change login password", "Replace the existing password?") {
        serial_println!("passwd> cancelled");
        return;
    }

    // Warn *before* deriving, so the human sees it even if they walk away during
    // the KDF. Enrolling a password that silently evaporates on reboot is worse
    // than refusing to enrol one.
    if !crate::synapse::fs::is_durable() {
        serial_println!(
            "passwd> WARNING: store backend = {} — this password will NOT survive a reboot",
            crate::synapse::fs::backend_name()
        );
    }
    serial_println!("passwd> deriving ({} rounds)…", crate::auth::DEFAULT_ITERATIONS);

    let mut rec = match crate::auth::store::enrol(&pw, &mut crate::shell::status_tick) {
        Ok(r) => r,
        Err(e) => {
            serial_println!("passwd> failed: {e}");
            return;
        }
    };
    // Preserve the human's existing policy across a password change — changing a
    // password should not silently re-enable an auto-lock they turned off.
    if let Some(old) = crate::auth::store::load() {
        rec.idle_lock_minutes = old.idle_lock_minutes;
        rec.lock_on_resume = old.lock_on_resume;
        rec.failed_total = old.failed_total;
    }
    match crate::auth::store::save(&rec) {
        Ok(()) => {
            crate::auth::reset_failures();
            serial_println!("passwd> password set");
        }
        Err(e) => serial_println!("passwd> failed: {e}"),
    }
}

fn clear(arg: &str) {
    if !crate::auth::enrolled() {
        serial_println!("passwd> no password is set");
        return;
    }
    if !has_flag(arg, "--yes")
        && !crate::modal::confirm("Remove login password", "The console will stop asking for a password. Continue?")
    {
        serial_println!("passwd> cancelled");
        return;
    }
    match crate::auth::store::clear() {
        Ok(()) => serial_println!("passwd> password cleared — this machine no longer asks for one"),
        Err(e) => serial_println!("passwd> failed: {e}"),
    }
}

fn idle(val: &str) {
    let Some(mut rec) = crate::auth::store::load() else {
        serial_println!("passwd> no password set — /passwd set first (auto-lock needs something to unlock with)");
        return;
    };
    let minutes = match val {
        "" => {
            serial_println!("passwd> usage: /passwd idle <minutes|off>");
            return;
        }
        "off" | "0" | "never" => 0,
        v => match v.parse::<u32>() {
            Ok(n) => n,
            Err(_) => {
                serial_println!("passwd> '{v}' is not a number of minutes");
                return;
            }
        },
    };
    rec.idle_lock_minutes = minutes;
    match crate::auth::store::save(&rec) {
        Ok(()) if minutes == 0 => serial_println!("passwd> auto-lock disabled"),
        Ok(()) => serial_println!("passwd> auto-lock after {minutes} minute(s) idle"),
        Err(e) => serial_println!("passwd> failed: {e}"),
    }
}

fn resume(val: &str) {
    let Some(mut rec) = crate::auth::store::load() else {
        serial_println!("passwd> no password set — /passwd set first");
        return;
    };
    let on = match val {
        "on" | "yes" | "true" => true,
        "off" | "no" | "false" => false,
        _ => {
            serial_println!("passwd> usage: /passwd resume on|off");
            return;
        }
    };
    rec.lock_on_resume = on;
    match crate::auth::store::save(&rec) {
        Ok(()) => serial_println!("passwd> resume from suspend {}", if on { "locks the console" } else { "does not lock" }),
        Err(e) => serial_println!("passwd> failed: {e}"),
    }
}
