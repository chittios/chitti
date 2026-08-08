//! SSH remote URLs, and the `git-upload-pack` command they name.
//!
//! Pure and unit-tested, because git accepts **two** shapes for the same thing
//! and they disagree about what a colon means:
//!
//! * `ssh://[user@]host[:port]/path` — a URL, where `:` before a number is a
//!   port and the path is everything after the next `/`.
//! * `[user@]host:path` — the scp-like form, where `:` separates the *path* and
//!   there is no port at all. `git@github.com:torvalds/linux.git` is this one,
//!   and it is what a repository page tells you to copy.
//!
//! Reading the second as the first gives host `github.com`, port… `torvalds` —
//! which fails to parse and reports a bad port for a URL the user copied from
//! the website. Distinguishing them is the whole job.

use alloc::string::{String, ToString};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SshRemote {
    pub user: String,
    pub host: String,
    pub port: u16,
    /// The repository path as the server expects it.
    pub path: String,
}

impl SshRemote {
    /// The command to `exec` on the server for a fetch.
    ///
    /// Single-quoted, as OpenSSH's own git does, because the path reaches a
    /// remote **shell**: an unquoted `my repo.git` would arrive as two arguments
    /// and a path containing `;` would be a command injection into the account
    /// the key belongs to.
    pub fn upload_pack(&self) -> String {
        alloc::format!("git-upload-pack '{}'", self.path.replace('\'', r"'\''"))
    }

    pub fn receive_pack(&self) -> String {
        alloc::format!("git-receive-pack '{}'", self.path.replace('\'', r"'\''"))
    }
}

/// Is this a URL the SSH transport handles?
pub fn is_ssh_url(url: &str) -> bool {
    parse(url).is_some()
}

/// Parse either SSH form, or `None` when it is not one (http/https/a local path).
pub fn parse(url: &str) -> Option<SshRemote> {
    if let Some(rest) = url.strip_prefix("ssh://") {
        let (authority, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        let (user, hostport) = split_user(authority);
        let (host, port) = split_host_port(hostport)?;
        if host.is_empty() {
            return None;
        }
        // **The leading `/` stays.** In URL form the path is absolute on the
        // server, and git sends it that way — stripping it asks for a path
        // relative to the login home, which for `/srv/repo.git` is a repository
        // that does not exist and a server that closes with no refs. The one
        // exception git makes is `/~` -> `~`, so a home-relative URL still works.
        let path = if path.starts_with("/~") { &path[1..] } else { path };
        return Some(SshRemote {
            user,
            host,
            path: path.to_string(),
            port,
        });
    }
    // Anything with a scheme we do not handle is not ours.
    if url.contains("://") {
        return None;
    }
    // scp-like `[user@]host:path`. The colon must be followed by something, and
    // a Windows-style `C:\…` or a bare relative path is not a remote.
    let (user, rest) = split_user(url);
    let (host, path) = rest.split_once(':')?;
    if host.is_empty() || path.is_empty() || host.contains('/') {
        return None;
    }
    Some(SshRemote {
        user,
        host: host.to_string(),
        port: 22,
        path: path.to_string(),
    })
}

fn split_user(s: &str) -> (String, &str) {
    match s.split_once('@') {
        // `git` is what every hosting provider expects, and is what the URLs
        // they publish carry — so it is the default rather than the local user.
        Some((u, rest)) => (u.to_string(), rest),
        None => ("git".to_string(), s),
    }
}

fn split_host_port(s: &str) -> Option<(String, u16)> {
    if let Some(rest) = s.strip_prefix('[') {
        let (h, tail) = rest.split_once(']')?;
        let port = match tail.strip_prefix(':') {
            Some(p) => p.parse().ok()?,
            None => 22,
        };
        return Some((h.to_string(), port));
    }
    match s.split_once(':') {
        Some((h, p)) => Some((h.to_string(), p.parse().ok()?)),
        None => Some((s.to_string(), 22)),
    }
}
