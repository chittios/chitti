//! The **generic content-server runtime**: the third stage of the web pipeline,
//! and the whole point of "write a server with just a SOUL.md + assets". It is
//! NOT specific to any one agent — it serves *whichever* content agent the
//! pipeline was started for (its install folder, set by `pipeline::start`).
//!
//! For each request it asks that agent's *model* (prompted with the agent's own
//! `SOUL.md`, where the author wrote the routing policy) which file under the
//! agent's `assets/` folder to serve. The model's answer — a filename — is the
//! *plan*; the runtime then reads that file through a capability- and
//! scope-gated `mem_fs_read` tool call confined to the agent's home, and returns
//! the bytes to the HTTP agent for framing.
//!
//! So a new web server is just `agents/<name>/{SOUL.md, manifest.json, assets/…}`:
//! the SOUL carries the routing/behaviour, the assets carry the content, and
//! this runtime (plus the generic Network + HTTP agents) does the rest — no
//! per-server Rust. Determinism boundary: the model *decides* (a judgment from
//! the author's SOUL); native code reads, and protocol bytes never touch the model.

use crate::cap::Right;
use crate::service::{pipeline, ServiceSpec};
use crate::synapse::registry;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Cache of `(home, soul)` — the SOUL is the system prompt for every routing
/// decision, so read it once per served agent rather than each request.
static SOUL: crate::mm::Locked<Option<(String, String)>> = crate::mm::Locked::new(None);

/// Read a file the served agent owns via a capability- and scope-gated
/// `mem_fs_read` tool call. `None` if the path is outside the agent's `assets/`
/// folder, the gate refuses it, or it isn't found — so the runtime can only ever
/// serve files inside the agent's own assets folder, whatever the model names.
pub fn read_asset(home: &str, path: &str) -> Option<Vec<u8>> {
    let base = alloc::format!("{home}/assets/");
    if !path.starts_with(&base) || path.contains("..") {
        return None;
    }
    match read_via_tool(path) {
        Some(s) => Some(s.into_bytes()),
        None => None,
    }
}

/// Execute a scoped `mem_fs_read` for `path` on the current (server) task and
/// return the file text, or `None` on not-found / denied.
fn read_via_tool(path: &str) -> Option<String> {
    let call = alloc::format!(r#"{{"name":"mem_fs_read","arguments":{{"path":"{path}"}}}}"#);
    match crate::synapse::execute(crate::sched::current_task_id(), &call) {
        crate::synapse::Invocation::Executed { result, .. } => result.strip_prefix("ok:").map(String::from),
        _ => None,
    }
}

/// Guess a content-type from the served file's extension.
fn ctype_for(path: &str) -> &'static str {
    if path.ends_with(".svg") {
        "image/svg+xml"
    } else if path.ends_with(".html") || path.ends_with('/') {
        "text/html; charset=utf-8"
    } else if path.ends_with(".css") {
        "text/css"
    } else if path.ends_with(".js") {
        "text/javascript"
    } else if path.ends_with(".json") {
        "application/json"
    } else if path.ends_with(".txt") || path.ends_with(".md") {
        "text/plain; charset=utf-8"
    } else {
        "application/octet-stream"
    }
}

/// Build the content reply frame (`"<status>\n<content-type>\n<body>"`) the HTTP
/// agent formats into a response.
fn frame(status: &str, ctype: &str, body: &[u8]) -> Vec<u8> {
    let mut out = alloc::format!("{status}\n{ctype}\n").into_bytes();
    out.extend_from_slice(body);
    out
}

/// The served agent's SOUL.md (cached per home) — the persona + routing policy
/// its author wrote, used as the system prompt for the model's decision.
fn soul(home: &str) -> String {
    if let Some((h, s)) = SOUL.with(|c| c.clone()) {
        if h == home {
            return s;
        }
    }
    let s = read_via_tool(&alloc::format!("{home}/SOUL.md"))
        .unwrap_or_else(|| String::from("You serve files from your assets/ folder. Reply with the filename to serve, or none."));
    SOUL.with(|c| *c = Some((home.to_string(), s.clone())));
    s
}

/// Pick the asset filename the model named in its reply: the first token that
/// looks like `<name>.<ext>` for a servable web extension. `None` if it named
/// none. Path confinement to `assets/` is enforced by [`read_asset`].
pub fn chosen_asset(reply: &str) -> Option<String> {
    const EXTS: &[&str] = &[".html", ".svg", ".css", ".js", ".json", ".txt", ".md", ".png", ".ico"];
    let is_name = |c: char| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-';
    let bytes = reply.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if is_name(bytes[i] as char) {
            let start = i;
            while i < bytes.len() && is_name(bytes[i] as char) {
                i += 1;
            }
            let tok = &reply[start..i];
            if !tok.contains("..") && EXTS.iter().any(|e| tok.ends_with(e)) {
                return Some(tok.to_string());
            }
        } else {
            i += 1;
        }
    }
    None
}

/// Serve one request for the agent installed at `home`: ask its model (prompted
/// with the agent's SOUL) which file to serve, then read that file through the
/// scoped tool call. The model *decides* (routing is a judgment from the SOUL,
/// not any compiled-in table); native code reads and frames.
pub fn serve(home: &str, path: &str) -> Vec<u8> {
    let persona = soul(home);
    let user = alloc::format!(
        "A client requested the path \"{path}\". Following your instructions, which file from \
         your assets/ folder do you serve for it? Reply with ONLY the filename, or the word none \
         if no page matches. Output nothing else."
    );
    let reply = match crate::shell::plan_reply(&persona, &user) {
        Some(r) => r,
        None => {
            crate::ktrace::log("service.server", "plan: no model loaded");
            return frame(
                "503 Service Unavailable",
                "text/html; charset=utf-8",
                b"<!doctype html><title>503</title><h1>Server agent: no model loaded to plan the route</h1>",
            );
        }
    };
    match chosen_asset(&reply) {
        Some(file) => {
            crate::ktrace::log_fmt(format_args!("service.server: model chose {file}"));
            match read_asset(home, &alloc::format!("{home}/assets/{file}")) {
                Some(body) => frame("200 OK", ctype_for(&file), &body),
                None => frame("404 Not Found", "text/html; charset=utf-8", b"<!doctype html><title>404</title><h1>Not found</h1>"),
            }
        }
        None => {
            crate::ktrace::log("service.server", "model named no document (404)");
            frame("404 Not Found", "text/html; charset=utf-8", b"<!doctype html><title>404</title><h1>Not found</h1>")
        }
    }
}

extern "C" fn server_serve(_arg: u64) {
    let Some(from_http) = pipeline::http_to_server() else {
        crate::ktrace::log("service.server", "pipeline channels not wired");
        return;
    };
    let Some(to_http) = pipeline::server_to_http() else { return };
    let home = pipeline::content_home().unwrap_or_default();
    loop {
        if let Ok(Some(req)) = crate::channel::try_recv_dgram(from_http) {
            // req = "METHOD path"; content routing is on the path.
            let text = String::from_utf8_lossy(&req);
            let path = text.split(' ').nth(1).unwrap_or("/").to_string();
            crate::ktrace::log_fmt(format_args!("service.server: planning route for {path} (agent {home})"));
            let reply = serve(&home, &path);
            pipeline::send_frame(to_http, &reply, crate::arch::now_ms() + pipeline::STAGE_DEADLINE_MS);
        }
        crate::shell::upkeep();
        crate::sched::yield_now();
    }
}

/// The generic content-server stage. Holds `InvokePrimitive(mem_fs_read)`; the
/// served agent's read scope (its own install folder) is granted by
/// `pipeline::start`. Serves any content agent — routing is planned by that
/// agent's model from its SOUL, never a compiled-in table.
pub static SERVER_STAGE: ServiceSpec =
    ServiceSpec { name: "server", entry: server_serve, autostart: false, caps: &[Right::InvokePrimitive(registry::MEM_FS_READ)] };

#[cfg(test)]
pub fn reset_soul() {
    SOUL.with(|c| *c = None);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn read_asset_is_scoped_to_the_agents_assets_dir() {
        use crate::agent::types::{CapDomain, CapabilityRequest, Rights, Scope};
        let me = crate::sched::current_task_id();
        crate::cap::grant(me, Right::InvokePrimitive(registry::MEM_FS_READ));
        crate::cap::grant_scopes(me, &[CapabilityRequest::new(CapDomain::Fs, Rights::READ, Scope::Path("/agent/srvtest/**".into()))]);
        crate::synapse::fs::write("/agent/srvtest/assets/index.html", b"<h1>hi</h1>");
        crate::synapse::fs::write("/agent/srvtest/SOUL.md", b"secret");
        assert_eq!(read_asset("/agent/srvtest", "/agent/srvtest/assets/index.html").as_deref(), Some(&b"<h1>hi</h1>"[..]));
        // Never serves files outside assets/ (e.g. the SOUL), nor traversals.
        assert_eq!(read_asset("/agent/srvtest", "/agent/srvtest/SOUL.md"), None);
        assert_eq!(read_asset("/agent/srvtest", "/agent/srvtest/assets/../SOUL.md"), None);
    }

    #[test_case]
    fn chosen_asset_extracts_the_model_named_file() {
        assert_eq!(chosen_asset("index.html").as_deref(), Some("index.html"));
        assert_eq!(chosen_asset("I'd serve docs.html").as_deref(), Some("docs.html"));
        assert_eq!(chosen_asset("style.css please").as_deref(), Some("style.css"));
        assert_eq!(chosen_asset("none"), None);
        assert_eq!(chosen_asset("no match here"), None);
    }

    #[test_case]
    fn content_type_from_extension() {
        assert_eq!(ctype_for("/x/a.html"), "text/html; charset=utf-8");
        assert_eq!(ctype_for("/x/a.svg"), "image/svg+xml");
        assert_eq!(ctype_for("/x/a.json"), "application/json");
    }
}
