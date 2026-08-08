//! `/ssh` — the client command surface over [`crate::net::ssh`].
//!
//! ```text
//! /ssh [user@]host[:port] [command…]   run a command, or a shell with no command
//! /ssh -i <key> …                      use a specific private key
//! /ssh -L <lport>:<rhost>:<rport> …    forward a local port through the server
//! /ssh keys                            which identities are on the store
//! /ssh known-hosts [forget <host>]     inspect / drop trusted host keys
//! ```
//!
//! Argument parsing is pure and unit-tested ([`parse`]); everything else is the
//! driver, which lives in `net::ssh::client`.

use super::*;
use crate::net::ssh::{auth, client};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Where identities are looked for, in order. The first is the ChittiOS home
/// convention; the second matches where OpenSSH would put it, so a key copied
/// from another machine lands somewhere that works.
pub const IDENTITY_PATHS: &[&str] = &[
    "/home/chitti/.ssh/id_ed25519",
    "/configs/core/id_ed25519",
    "/home/chitti/.ssh/id_ecdsa",
    "/configs/core/id_ecdsa",
];

/// A parsed `/ssh` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invocation {
    pub user: String,
    pub host: String,
    pub port: u16,
    /// `None` runs a shell.
    pub command: Option<String>,
    pub identity: Option<String>,
    /// `-L lport:rhost:rport`
    pub local_forward: Option<(u16, String, u16)>,
    pub want_pty: bool,
}

/// Parse `[user@]host[:port]` plus flags and an optional command.
///
/// Pure so the fiddly bits — a port that is really part of an IPv6 literal, a
/// command containing `@`, `-L` with a host that itself has a colon — are
/// testable without a network.
pub fn parse(arg: &str, default_user: &str) -> Result<Invocation, String> {
    let mut identity = None;
    let mut local_forward = None;
    let mut target: Option<String> = None;
    let mut command: Vec<String> = Vec::new();
    let mut want_pty = false;

    let toks: Vec<&str> = arg.split_whitespace().collect();
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i];
        // Once the target is known, everything left is the command — including
        // things that look like flags, because `ssh host ls -l` must not have
        // `-l` eaten by the client.
        if target.is_some() {
            command.push(t.to_string());
            i += 1;
            continue;
        }
        match t {
            "-i" => {
                i += 1;
                identity = Some(toks.get(i).ok_or("usage: -i <key path>")?.to_string());
            }
            "-t" => want_pty = true,
            "-L" => {
                i += 1;
                let spec = toks.get(i).ok_or("usage: -L <lport>:<host>:<rport>")?;
                local_forward = Some(parse_forward(spec)?);
            }
            _ if t.starts_with('-') => return Err(alloc::format!("unknown option {t}")),
            _ => target = Some(t.to_string()),
        }
        i += 1;
    }

    let target = target.ok_or("usage: /ssh [user@]host[:port] [command…]")?;
    let (user, hostport) = match target.split_once('@') {
        Some((u, h)) => (u.to_string(), h.to_string()),
        None => (default_user.to_string(), target),
    };
    let (host, port) = split_host_port(&hostport)?;
    if host.is_empty() {
        return Err("no host given".to_string());
    }
    Ok(Invocation {
        user,
        host,
        port,
        command: if command.is_empty() {
            None
        } else {
            Some(command.join(" "))
        },
        identity,
        local_forward,
        want_pty,
    })
}

/// `host`, `host:port`, `[v6]:port` — the colon only means a port when it is
/// unambiguous, or an IPv6 literal loses its address.
fn split_host_port(s: &str) -> Result<(String, u16), String> {
    if let Some(rest) = s.strip_prefix('[') {
        let (h, tail) = rest.split_once(']').ok_or("unterminated [ipv6] literal")?;
        let port = match tail.strip_prefix(':') {
            Some(p) => p.parse().map_err(|_| "bad port")?,
            None => 22,
        };
        return Ok((h.to_string(), port));
    }
    // More than one colon means a bare IPv6 address, which has no port here.
    if s.matches(':').count() > 1 {
        return Ok((s.to_string(), 22));
    }
    match s.split_once(':') {
        Some((h, p)) => Ok((h.to_string(), p.parse().map_err(|_| "bad port")?)),
        None => Ok((s.to_string(), 22)),
    }
}

/// `lport:rhost:rport`, where `rhost` may itself contain colons if bracketed.
fn parse_forward(spec: &str) -> Result<(u16, String, u16), String> {
    let (lport, rest) = spec.split_once(':').ok_or("expected <lport>:<host>:<rport>")?;
    let (host, rport) = rest.rsplit_once(':').ok_or("expected <lport>:<host>:<rport>")?;
    Ok((
        lport.parse().map_err(|_| "bad local port")?,
        host.trim_matches(|c| c == '[' || c == ']').to_string(),
        rport.parse().map_err(|_| "bad remote port")?,
    ))
}

/// Load the first identity that parses, or report why each failed.
///
/// The per-path reason matters: "no key found" when the key is *there* but
/// passphrase-protected sends the user looking in the wrong place entirely.
pub fn load_identity(explicit: Option<&str>) -> Result<Option<auth::PrivateKey>, String> {
    let paths: Vec<String> = match explicit {
        Some(p) => alloc::vec![super::resolve_path(p)],
        None => IDENTITY_PATHS.iter().map(|s| s.to_string()).collect(),
    };
    let mut reasons: Vec<String> = Vec::new();
    for p in &paths {
        let Some(bytes) = crate::synapse::fs::read(p) else {
            continue;
        };
        let text = String::from_utf8_lossy(&bytes).into_owned();
        match auth::parse_openssh_private(&text) {
            Ok(k) => return Ok(Some(k)),
            Err(e) => reasons.push(alloc::format!("{p}: {e}")),
        }
    }
    if explicit.is_some() && reasons.is_empty() {
        return Err(alloc::format!("no such key: {}", paths[0]));
    }
    if !reasons.is_empty() {
        return Err(reasons.join("\n       "));
    }
    Ok(None)
}

/// `/ssh …`
pub(super) fn run_ssh(arg: &str) {
    let arg = arg.trim();
    match arg.split_whitespace().next() {
        None => {
            serial_println!(
                "ssh> usage: /ssh [user@]host[:port] [command…]\n\
                 ssh>        /ssh -i <key> …          use a specific identity\n\
                 ssh>        /ssh -t …                request a terminal\n\
                 ssh>        /ssh -L <lp>:<h>:<rp> …  forward a local port\n\
                 ssh>        /ssh keys | known-hosts [forget <host>]"
            );
            return;
        }
        Some("keys") => return list_keys(),
        Some("known-hosts") => return known_hosts_cmd(arg),
        _ => {}
    }

    let inv = match parse(arg, "chitti") {
        Ok(v) => v,
        Err(e) => {
            serial_println!("ssh> {e}");
            return;
        }
    };
    let identity = match load_identity(inv.identity.as_deref()) {
        Ok(k) => k,
        Err(e) => {
            serial_println!("ssh> {e}");
            return;
        }
    };

    serial_println!("ssh> connecting to {}@{}:{}…", inv.user, inv.host, inv.port);
    let mut c = match client::Client::connect(&inv.host, inv.port) {
        Ok(c) => c,
        Err(e) => {
            serial_println!("ssh> {e}");
            return;
        }
    };
    if let Some(k) = c.host_key.as_ref() {
        serial_println!("ssh> host key {} {}", k.algorithm(), k.fingerprint());
    }

    // A password is only asked for when no key worked — and it is asked for at
    // the console, never taken from an argument, so it cannot end up in the
    // shell history or in an agent's tool call.
    let mut password: Option<String> = None;
    if identity.is_none() {
        let p = crate::modal::input("SSH password", &alloc::format!("{}@{}:", inv.user, inv.host), true);
        if !p.is_empty() {
            password = Some(p);
        }
    }
    if let Err(e) = c.authenticate(&inv.user, identity.as_ref(), password.as_deref()) {
        serial_println!("ssh> {e}");
        c.disconnect();
        return;
    }
    serial_println!("ssh> authenticated as {}", inv.user);

    if let Some((lp, rh, rp)) = inv.local_forward.clone() {
        forward_local(&mut c, lp, &rh, rp);
        c.disconnect();
        return;
    }

    let want = match inv.command.as_deref() {
        Some(cmd) => client::Session::Exec(cmd),
        None => client::Session::Shell,
    };
    let pty = if inv.want_pty || inv.command.is_none() {
        Some((80, 24))
    } else {
        None
    };
    let mut sent_eof = false;
    let status = c.session(
        want,
        pty,
        |data, is_stderr| {
            let text = alloc::string::String::from_utf8_lossy(data);
            if is_stderr {
                serial_print!("{text}");
            } else {
                serial_print!("{text}");
            }
        },
        || {
            // A command sends no stdin; a shell forwards the console. Returning
            // `None` closes the write half, which is what makes a remote command
            // that reads stdin terminate instead of hanging forever.
            if sent_eof {
                return Some(Vec::new());
            }
            sent_eof = true;
            None
        },
    );
    match status {
        Ok(Some(0)) | Ok(None) => serial_println!("\nssh> done"),
        Ok(Some(n)) => serial_println!("\nssh> remote command exited {n}"),
        Err(e) => serial_println!("\nssh> {e}"),
    }
    c.disconnect();
}

/// `-L`: accept on a local port and pump each connection through the server.
fn forward_local(c: &mut client::Client, lport: u16, rhost: &str, rport: u16) {
    let listener = match crate::net::listen(lport) {
        Ok(l) => l,
        Err(e) => {
            serial_println!("ssh> cannot listen on :{lport}: {e}");
            return;
        }
    };
    serial_println!(
        "ssh> forwarding :{lport} -> {rhost}:{rport} through the server (Ctrl+C to stop)"
    );
    let r = c.forward(listener, rhost, rport);
    crate::net::close_listener(listener);
    match r {
        Ok(n) => serial_println!("ssh> forward closed after {n} connection(s)"),
        Err(e) => serial_println!("ssh> {e}"),
    }
}

fn list_keys() {
    let mut any = false;
    for p in IDENTITY_PATHS {
        let Some(bytes) = crate::synapse::fs::read(p) else {
            continue;
        };
        any = true;
        let text = String::from_utf8_lossy(&bytes).into_owned();
        match auth::parse_openssh_private(&text) {
            Ok(k) => serial_println!("  {p}  {}  {}", k.algorithm(), k.public().fingerprint()),
            Err(e) => serial_println!("  {p}  unusable: {e}"),
        }
    }
    if !any {
        serial_println!(
            "ssh> no identity found. Looked in:\n{}",
            IDENTITY_PATHS
                .iter()
                .map(|p| alloc::format!("  {p}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

fn known_hosts_cmd(arg: &str) {
    let mut it = arg.split_whitespace();
    let _ = it.next(); // "known-hosts"
    let text = crate::synapse::fs::read(client::KNOWN_HOSTS)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    match it.next() {
        Some("forget") => {
            let Some(host) = it.next() else {
                serial_println!("ssh> usage: /ssh known-hosts forget <host>");
                return;
            };
            // Match with **or without** the `[host]:port` decoration: the file
            // writes a non-default port that way, and a user forgetting a host
            // types the host.
            let kept: Vec<&str> = text
                .lines()
                .filter(|l| {
                    !l.split_whitespace().next().is_some_and(|pat| {
                        pat.split(',').any(|x| {
                            x == host || crate::net::ssh::hostkey::pattern_host(x) == host
                        })
                    })
                })
                .collect();
            let removed = text.lines().count() - kept.len();
            let mut out = kept.join("\n");
            if !out.is_empty() {
                out.push('\n');
            }
            let _ = crate::synapse::fs::write(client::KNOWN_HOSTS, out.as_bytes());
            serial_println!("ssh> forgot {removed} entr{} for {host}", if removed == 1 { "y" } else { "ies" });
        }
        _ => {
            let entries = crate::net::ssh::hostkey::parse_known_hosts(&text);
            if entries.is_empty() {
                serial_println!("ssh> no known hosts yet ({})", client::KNOWN_HOSTS);
                return;
            }
            for e in &entries {
                serial_println!("  {}  {}", e.hosts.join(","), e.algorithm);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The target, user and port come apart the way OpenSSH does it.
    #[test_case]
    fn targets_parse() {
        let v = parse("git@github.com", "chitti").unwrap();
        assert_eq!(v.user, "git");
        assert_eq!(v.host, "github.com");
        assert_eq!(v.port, 22);
        assert_eq!(v.command, None, "no command means a shell");

        let v = parse("example.com:2222 uname -a", "chitti").unwrap();
        assert_eq!(v.user, "chitti", "no user means the local one");
        assert_eq!(v.port, 2222);
        assert_eq!(v.command.as_deref(), Some("uname -a"));
    }

    /// **Flags after the target belong to the remote command.** `ssh host ls -l`
    /// must not have `-l` eaten by the client.
    #[test_case]
    fn flags_after_the_target_are_part_of_the_command() {
        let v = parse("host ls -l -i /tmp", "chitti").unwrap();
        assert_eq!(v.command.as_deref(), Some("ls -l -i /tmp"));
        assert_eq!(v.identity, None, "-i after the target is the command's, not ours");

        // …while before the target it is ours.
        let v = parse("-i /keys/id_ed25519 host ls", "chitti").unwrap();
        assert_eq!(v.identity.as_deref(), Some("/keys/id_ed25519"));
        assert_eq!(v.command.as_deref(), Some("ls"));
    }

    /// An IPv6 literal is not a host:port — the colon only means a port when it
    /// is unambiguous.
    #[test_case]
    fn ipv6_literals_keep_their_colons() {
        let v = parse("fe80::1", "chitti").unwrap();
        assert_eq!(v.host, "fe80::1");
        assert_eq!(v.port, 22);

        let v = parse("[fe80::1]:2222", "chitti").unwrap();
        assert_eq!(v.host, "fe80::1");
        assert_eq!(v.port, 2222);
    }

    /// `-L` splits on the *last* colon, so a bracketed host keeps its own.
    #[test_case]
    fn local_forwards_parse() {
        let v = parse("-L 8080:internal.example:80 host", "chitti").unwrap();
        assert_eq!(v.local_forward, Some((8080, "internal.example".to_string(), 80)));

        let v = parse("-L 5432:[fe80::2]:5432 host", "chitti").unwrap();
        assert_eq!(v.local_forward, Some((5432, "fe80::2".to_string(), 5432)));

        assert!(parse("-L nonsense host", "chitti").is_err());
        assert!(parse("-L 8080:host host", "chitti").is_err(), "needs three parts");
    }

    /// Missing and malformed invocations are refused with a usage message.
    #[test_case]
    fn bad_invocations_are_refused() {
        assert!(parse("", "chitti").is_err());
        assert!(parse("-i", "chitti").is_err(), "-i needs a value");
        assert!(parse("--bogus host", "chitti").is_err());
        assert!(parse("host:notaport", "chitti").is_err());
    }

    /// A terminal is requested for a shell without asking, and for a command
    /// only when `-t` says so — a pty on a piped command corrupts its output
    /// with terminal echo.
    #[test_case]
    fn pty_is_requested_for_shells_not_for_commands() {
        assert!(!parse("host ls", "chitti").unwrap().want_pty);
        assert!(parse("-t host ls", "chitti").unwrap().want_pty);
        // A shell has no command, and the caller turns that into a pty request.
        assert_eq!(parse("host", "chitti").unwrap().command, None);
    }
}
