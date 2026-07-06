//! The **Doc agent** — the application layer of the web pipeline, and a genuine
//! agent: for each request its *model* decides what to serve. It receives a
//! parsed request (`"METHOD path"`) from the HTTP agent, and asks its own model
//! (prompted with its `SOUL.md` persona) which document to read. The model's
//! answer is a `mem_fs_read` tool call — the *plan* — which the Doc agent then
//! executes through the Synapse gate (capability- and scope-checked to read only
//! within its own install folder). The model plans; the executor acts; the
//! bytes come back to the HTTP agent, which does all the HTTP framing.
//!
//! This is the determinism boundary in miniature: routing is a *judgment* the
//! model makes from its SOUL (not a hardcoded table), while the file read and
//! all protocol handling are deterministic, gated native code. No model loaded →
//! the agent honestly can't plan, and serves 503.

use crate::cap::Right;
use crate::service::{pipeline, ServiceSpec};
use crate::synapse::registry;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// The Doc agent's SOUL text, read once from its own home via a tool call and
/// cached (it's the system prompt for every routing decision).
static SOUL: crate::mm::Locked<Option<String>> = crate::mm::Locked::new(None);

/// Read a file the agent owns via a capability- and scope-gated `mem_fs_read`
/// tool call. Returns the bytes, or `None` if the path is outside the agent's
/// `assets/` folder, the gate refuses it, or it isn't found. This is the agent
/// acting on its plan — the read only ever succeeds within its own home.
pub fn read_asset(home: &str, path: &str) -> Option<Vec<u8>> {
    // Defence in depth: only ever serve files under our own assets/ dir. The
    // executor's scope gate independently confines reads to the home, but this
    // also stops the agent serving its own SOUL.md / skills over HTTP.
    let base = alloc::format!("{home}/assets/");
    if !path.starts_with(&base) || path.contains("..") {
        return None;
    }
    let call = alloc::format!(r#"{{"name":"mem_fs_read","arguments":{{"path":"{path}"}}}}"#);
    match crate::synapse::execute(crate::sched::current_task_id(), &call) {
        crate::synapse::Invocation::Executed { result, .. } => {
            result.strip_prefix("ok:").map(|c| c.as_bytes().to_vec())
        }
        _ => None, // denied by scope/capability, or refused
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
    } else {
        "text/plain; charset=utf-8"
    }
}

/// Build the content reply frame (`"<status>\n<content-type>\n<body>"`) the HTTP
/// agent formats into a response.
fn frame(status: &str, ctype: &str, body: &[u8]) -> Vec<u8> {
    let mut out = alloc::format!("{status}\n{ctype}\n").into_bytes();
    out.extend_from_slice(body);
    out
}

/// Read the agent's own SOUL.md (cached) — the persona/policy the model routes by.
fn soul(home: &str) -> String {
    if let Some(s) = SOUL.with(|c| c.clone()) {
        return s;
    }
    let s = read_asset_raw(&alloc::format!("{home}/SOUL.md")).unwrap_or_else(|| {
        String::from("You are the Doc agent. Serve documents from your assets/ folder.")
    });
    SOUL.with(|c| *c = Some(s.clone()));
    s
}

/// Read any file in the agent's home via a scoped tool call (used for SOUL.md,
/// which lives at the home root, not under assets/).
fn read_asset_raw(path: &str) -> Option<String> {
    let call = alloc::format!(r#"{{"name":"mem_fs_read","arguments":{{"path":"{path}"}}}}"#);
    match crate::synapse::execute(crate::sched::current_task_id(), &call) {
        crate::synapse::Invocation::Executed { result, .. } => result.strip_prefix("ok:").map(String::from),
        _ => None,
    }
}

/// Pick the asset filename the model named in its reply: the first token that
/// looks like `<name>.<html|svg|css>` (rejecting any path separator or `..`).
/// This is how the Doc agent reads the model's *decision* — the model says which
/// document to serve; we turn that into a scoped read. `None` if it named none.
pub fn chosen_asset(reply: &str) -> Option<String> {
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
            if (tok.ends_with(".html") || tok.ends_with(".svg") || tok.ends_with(".css")) && !tok.contains("..") {
                return Some(tok.to_string());
            }
        } else {
            i += 1;
        }
    }
    None
}

/// Serve one request: ask the model (prompted with the agent's SOUL) which
/// document to serve for this path, then act on that decision by reading the
/// chosen file through the capability- and scope-gated tool call. The model
/// *decides* (routing is a judgment from its SOUL, not a compiled-in table);
/// native code reads and frames.
pub fn serve(home: &str, _method: &str, path: &str) -> Vec<u8> {
    let persona = soul(home);
    let user = alloc::format!(
        "You serve a small website. Choose which file to serve for the requested path:\n\
         - the site root, i.e. the path is exactly \"/\"  -> the home page, index.html\n\
         - the path \"/docs\"                             -> docs.html\n\
         - the path \"/logo.svg\"                         -> logo.svg\n\
         - any other path                                -> none\n\n\
         The requested path is \"{path}\". Reply with ONLY the filename (index.html, docs.html, \
         or logo.svg), or the word none if no page matches. Output nothing else."
    );
    let reply = match crate::shell::plan_reply(&persona, &user) {
        Some(r) => r,
        None => {
            crate::ktrace::log("service.doc", "plan: no model loaded");
            return frame(
                "503 Service Unavailable",
                "text/html; charset=utf-8",
                b"<!doctype html><title>503</title><h1>Doc agent: no model loaded to plan the route</h1>",
            );
        }
    };
    match chosen_asset(&reply) {
        Some(file) => {
            crate::ktrace::log_fmt(format_args!("service.doc: model chose {file}"));
            match read_asset(home, &alloc::format!("{home}/assets/{file}")) {
                Some(body) => frame("200 OK", ctype_for(&file), &body),
                None => frame("404 Not Found", "text/html; charset=utf-8", b"<!doctype html><title>404</title><h1>Not found</h1>"),
            }
        }
        None => {
            crate::ktrace::log("service.doc", "model named no document (404)");
            frame("404 Not Found", "text/html; charset=utf-8", b"<!doctype html><title>404</title><h1>Not found</h1>")
        }
    }
}

extern "C" fn doc_serve(_arg: u64) {
    let Some(from_http) = pipeline::http_to_doc() else {
        crate::ktrace::log("service.doc", "pipeline channels not wired");
        return;
    };
    let Some(to_http) = pipeline::doc_to_http() else { return };
    let home = pipeline::doc_home().unwrap_or_default();
    loop {
        if let Ok(Some(req)) = crate::channel::try_recv_dgram(from_http) {
            // req = "METHOD path"
            let text = String::from_utf8_lossy(&req);
            let mut it = text.split(' ');
            let method = it.next().unwrap_or("GET").to_string();
            let path = it.next().unwrap_or("/").to_string();
            crate::ktrace::log_fmt(format_args!("service.doc: planning route for {method} {path}"));
            let reply = serve(&home, &method, &path);
            pipeline::send_frame(to_http, &reply, crate::arch::now_ms() + pipeline::STAGE_DEADLINE_MS);
        }
        crate::shell::upkeep();
        crate::sched::yield_now();
    }
}

/// The Doc content service. Holds `InvokePrimitive(mem_fs_read)`; its read scope
/// (its own install folder) is granted by `pipeline::start`. Its routing is
/// planned by the model, not a compiled-in table.
pub static DOC_STAGE: ServiceSpec =
    ServiceSpec { name: "doc", entry: doc_serve, autostart: false, caps: &[Right::InvokePrimitive(registry::MEM_FS_READ)] };

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn read_asset_is_scoped_to_the_home_assets_dir() {
        use crate::agent::types::{CapDomain, CapabilityRequest, Rights, Scope};
        // Grant this (bootstrap) task the read right + home scope, drop files,
        // and confirm read_asset reads within assets/ but refuses outside it.
        let me = crate::sched::current_task_id();
        crate::cap::grant(me, Right::InvokePrimitive(registry::MEM_FS_READ));
        crate::cap::grant_scopes(me, &[CapabilityRequest::new(CapDomain::Fs, Rights::READ, Scope::Path("/agent/doctest/**".into()))]);
        crate::synapse::fs::write("/agent/doctest/assets/index.html", b"<h1>hello docs</h1>");
        crate::synapse::fs::write("/agent/doctest/SOUL.md", b"secret persona");

        // In-scope asset reads back.
        assert_eq!(read_asset("/agent/doctest", "/agent/doctest/assets/index.html").as_deref(), Some(&b"<h1>hello docs</h1>"[..]));
        // A path outside assets/ (the SOUL) is refused even though it's in the
        // home scope — the agent won't serve its own persona over HTTP.
        assert_eq!(read_asset("/agent/doctest", "/agent/doctest/SOUL.md"), None);
        // A traversal attempt is refused.
        assert_eq!(read_asset("/agent/doctest", "/agent/doctest/assets/../SOUL.md"), None);
    }

    #[test_case]
    fn chosen_asset_extracts_the_model_named_file() {
        // The model may answer tersely or in prose; we pull the filename token.
        assert_eq!(chosen_asset("index.html").as_deref(), Some("index.html"));
        assert_eq!(chosen_asset("I would serve docs.html for that.").as_deref(), Some("docs.html"));
        assert_eq!(chosen_asset("The logo is at logo.svg").as_deref(), Some("logo.svg"));
        assert_eq!(chosen_asset("none"), None);
        assert_eq!(chosen_asset("nothing matches"), None);
        // Only a bare filename is extracted (path confinement to assets/ is
        // read_asset's job — see the scope test): a slashed path yields just the
        // trailing filename token, which read_asset then serves from assets/ only.
        assert_eq!(chosen_asset("/etc/passwd"), None); // no known extension → nothing
        assert_eq!(chosen_asset("../secret.html").as_deref(), Some("secret.html"));
    }

    #[test_case]
    fn content_type_from_extension() {
        assert_eq!(ctype_for("/x/index.html"), "text/html; charset=utf-8");
        assert_eq!(ctype_for("/x/logo.svg"), "image/svg+xml");
        assert_eq!(ctype_for("/x/data.bin"), "text/plain; charset=utf-8");
    }
}
