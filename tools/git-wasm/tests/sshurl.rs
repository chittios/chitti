//! `sshurl` — the two URL shapes git accepts for an SSH remote.
//!
//! In `tests/` and not beside the code: this crate mounts kernel modules that
//! carry `#[test_case]`, so `cargo test --lib` cannot compile at all and a
//! `#[cfg(test)] mod tests` in `src/` is silent dead code rather than coverage.
//! Exactly the trap CLAUDE.md records for `framebuffer/`.

use chitti_git_wasm::sshurl::{is_ssh_url, parse};


/// **The scp-like form is what people actually paste.** Its colon separates
/// the path, not a port.
#[test]
fn the_scp_form_parses() {
    let r = parse("git@github.com:torvalds/linux.git").expect("a real-world URL");
    assert_eq!(r.user, "git");
    assert_eq!(r.host, "github.com");
    assert_eq!(r.port, 22, "the scp form has no port");
    assert_eq!(r.path, "torvalds/linux.git");
    assert_eq!(r.upload_pack(), "git-upload-pack 'torvalds/linux.git'");

    // A user other than git is honoured.
    let r = parse("deploy@example.com:/srv/repo.git").unwrap();
    assert_eq!(r.user, "deploy");
    assert_eq!(r.path, "/srv/repo.git");
}

/// The URL form takes a port, and its leading `/` is not part of the path.
#[test]
fn the_url_form_parses() {
    let r = parse("ssh://git@example.com:2222/team/repo.git").unwrap();
    assert_eq!((r.user.as_str(), r.host.as_str(), r.port), ("git", "example.com", 2222));
    assert_eq!(r.path, "/team/repo.git", "the URL form is absolute on the server");

    let r = parse("ssh://example.com/repo.git").unwrap();
    assert_eq!(r.user, "git", "the default user");
    assert_eq!(r.port, 22);
    assert_eq!(r.path, "/repo.git");

    // IPv6 keeps its colons.
    let r = parse("ssh://git@[fe80::1]:2222/repo.git").unwrap();
    assert_eq!(r.host, "fe80::1");
    assert_eq!(r.port, 2222);
}

/// What is *not* an SSH remote must not be claimed, or an https clone would
/// be routed down the SSH path and fail for a mysterious reason.
#[test]
fn non_ssh_urls_are_not_claimed() {
    for u in [
        "https://github.com/a/b.git",
        "http://example.com/repo.git",
        "file:///srv/repo.git",
        "/srv/local/repo.git",
        "relative/path",
        "",
    ] {
        assert!(parse(u).is_none(), "{u} must not parse as an SSH remote");
        assert!(!is_ssh_url(u));
    }
}

/// A path reaching a remote shell is quoted, and an embedded quote cannot break
/// out of it.
///
/// Checked by **unquoting the result the way a shell would** and asserting the
/// original path comes back. A `contains("';id;'")` proxy was tried first and
/// fails on correctly-escaped output: the POSIX idiom for a quote inside single
/// quotes is `'\''` (close, escaped quote, reopen), which legitimately contains
/// a quote-semicolon-quote run. A proxy that fires on correct output is worse
/// than no test.
#[test]
fn the_remote_path_is_shell_quoted() {
    for path in [
        "my repo.git",
        "evil';id;'.git",
        "a'b",
        "'",
        "plain/repo.git",
        "spaces and 'quotes' both.git",
    ] {
        let r = parse(&format!("git@h:{path}")).expect("parses");
        let cmd = r.upload_pack();
        let arg = cmd.strip_prefix("git-upload-pack ").expect("command prefix");
        assert_eq!(
            sh_unquote(arg).as_deref(),
            Some(path),
            "a shell must see exactly the path, from {arg}"
        );
    }
}

/// Undo POSIX single-quoting: `'…'` is literal, and outside quotes `\x` is `x`.
/// Returns `None` if the shell would see more than one word — which is the
/// failure that matters, since a second word is a second argument or a command.
fn sh_unquote(s: &str) -> Option<String> {
    let mut out = String::new();
    let mut in_q = false;
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        match c {
            '\'' => in_q = !in_q,
            '\\' if !in_q => out.push(it.next()?),
            ' ' | ';' | '&' | '|' if !in_q => return None, // an unquoted metacharacter
            c => out.push(c),
        }
    }
    if in_q {
        return None; // unterminated quote
    }
    Some(out)
}

/// A malformed port is a refusal, not a silent default — otherwise
/// `ssh://h:notaport/r` quietly connects to 22.
#[test]
fn a_bad_port_is_refused() {
    assert!(parse("ssh://h:notaport/r.git").is_none());
    assert!(parse("ssh://h:99999/r.git").is_none());
}
