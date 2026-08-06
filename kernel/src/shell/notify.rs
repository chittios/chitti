//! notify
//!
//! The `/notify` command surface over [`crate::notify`], plus the action-pane
//! repaint bridge. All the logic is in the kernel module (which is testable);
//! this file is argument parsing, serial output, and the `#[cfg]` twin that
//! keeps the framebuffer out of the test build.

use super::*;

/// `/notify [list|post|read|clear|test|open]` — the notification queue.
pub(super) fn run_notify_cmd(arg: &str) {
    let a = arg.trim();
    let mut parts = a.splitn(2, char::is_whitespace);
    let verb = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("").trim();

    match verb {
        "" | "open" | "show" => open_pane(),
        "list" | "ls" => list(rest),
        "post" | "add" => post(rest),
        "read" | "ack" => read(rest),
        "dismiss" | "rm" => dismiss(rest),
        "clear" => {
            let n = crate::notify::clear();
            serial_println!("notify> cleared {n} notification(s)");
        }
        "test" => {
            // The e2e handle: one deterministic notification with no dependency
            // on a schedule, a daemon or a model.
            let id = crate::notify::post(
                crate::notify::Severity::Info,
                "kernel",
                "notify test",
                "posted by /notify test",
            );
            serial_println!("notify> posted #{id} (info) — /notify list");
        }
        other => {
            serial_println!("notify> unknown '{other}'");
            usage();
        }
    }
}

fn usage() {
    serial_println!("notify> usage: /notify [open|list [n]|post <severity> <title> [-- <body>]");
    serial_println!("               |read <id>|read all|dismiss <id>|clear|test]");
    serial_println!("               severity: info|ok|warn|error|action");
}

fn list(rest: &str) {
    let limit: usize = rest.split_whitespace().next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let all = crate::notify::list();
    if all.is_empty() {
        serial_println!("notify> (no notifications)");
        return;
    }
    let now = crate::clock::now_unix();
    serial_println!(
        "notify> {} notification(s), {} unread",
        all.len(),
        crate::notify::unread_count()
    );
    for n in all.iter().take(limit) {
        let mark = if n.read { ' ' } else { '*' };
        let rep = if n.count > 1 { alloc::format!(" (x{})", n.count) } else { String::new() };
        serial_println!(
            "  {}#{} [{}] {} {} — {}{}",
            mark,
            n.id,
            n.severity.as_str(),
            crate::notify::relative_age(now, n.unix),
            n.source,
            n.title,
            rep
        );
        if !n.body.is_empty() {
            serial_println!("     {}", n.body);
        }
        if let Some(act) = &n.action {
            serial_println!("     action: {act}");
        }
    }
    if all.len() > limit {
        serial_println!("  … {} more (/notify list {})", all.len() - limit, all.len());
    }
}

fn post(rest: &str) {
    if rest.is_empty() {
        usage();
        return;
    }
    let (head, body) = match rest.split_once("--") {
        Some((h, b)) => (h.trim(), b.trim()),
        None => (rest, ""),
    };
    let mut it = head.splitn(2, char::is_whitespace);
    let sev_word = it.next().unwrap_or("");
    let Some(sev) = crate::notify::Severity::parse(sev_word) else {
        serial_println!("notify> unknown severity '{sev_word}' (info|ok|warn|error|action)");
        return;
    };
    let title = it.next().unwrap_or("").trim();
    if title.is_empty() {
        serial_println!("notify> a notification needs a title");
        return;
    }
    // The source is stamped here, from the live identity — never from the
    // argument string. An agent that could name its own source could post as
    // `kernel`, which is a transfer of authority dressed as a label.
    let agent = active_agent_id();
    let source = if agent == crate::agent::manifest::ORCHESTRATOR_ID.0 {
        String::from("shell")
    } else {
        alloc::format!("agent:{agent}")
    };
    let id = crate::notify::post(sev, &source, title, body);
    serial_println!("notify> posted #{id} ({}) from {source}", sev.as_str());
}

fn read(rest: &str) {
    match rest {
        "all" | "*" => {
            let n = crate::notify::mark_all_read();
            serial_println!("notify> marked {n} read");
        }
        "" => serial_println!("notify> usage: /notify read <id>|all"),
        id => match id.parse::<u64>() {
            Ok(v) if crate::notify::mark_read(v) => serial_println!("notify> #{v} read"),
            Ok(v) => serial_println!("notify> no notification #{v}"),
            Err(_) => serial_println!("notify> '{id}' is not an id"),
        },
    }
}

fn dismiss(rest: &str) {
    match rest.parse::<u64>() {
        Ok(v) if crate::notify::dismiss(v) => serial_println!("notify> #{v} dismissed"),
        Ok(v) => serial_println!("notify> no notification #{v}"),
        Err(_) => serial_println!("notify> usage: /notify dismiss <id>"),
    }
}

#[cfg(not(test))]
fn open_pane() {
    crate::framebuffer::open_notifications();
    refresh_notifications();
    serial_println!(
        "notify> live queue in the action pane ({} unread of {})",
        crate::notify::unread_count(),
        crate::notify::len()
    );
}

#[cfg(test)]
fn open_pane() {
    // No framebuffer in the test build: fall back to the serial listing, so the
    // command still does something useful rather than silently nothing.
    list("");
}

/// Repaint the notifications action pane from the live ring.
///
/// Called from `/notify open` and from `notify::post` — but only when the tab is
/// actually on screen, so a post while it is closed cannot force a relayout.
#[cfg(not(test))]
pub(crate) fn refresh_notifications() {
    use crate::framebuffer::NotifyViewItem;
    let all = crate::notify::list();
    let now = crate::clock::now_unix();
    // The ages are owned `String`s and the view items borrow them, so they must
    // outlive the borrow — build both lists, then draw.
    let ages: alloc::vec::Vec<String> =
        all.iter().map(|n| crate::notify::relative_age(now, n.unix)).collect();
    let items: alloc::vec::Vec<NotifyViewItem<'_>> = all
        .iter()
        .zip(ages.iter())
        .map(|(n, age)| NotifyViewItem {
            severity: n.severity,
            source: n.source.as_str(),
            title: n.title.as_str(),
            when: age.as_str(),
            read: n.read,
            count: n.count,
        })
        .collect();
    let unread = crate::notify::unread_count();
    let title = if unread > 0 {
        alloc::format!("Notifications ({unread} unread of {})", all.len())
    } else {
        alloc::format!("Notifications ({})", all.len())
    };
    crate::framebuffer::draw_notifications(&items, &title);
}

#[cfg(test)]
pub(crate) fn refresh_notifications() {}
