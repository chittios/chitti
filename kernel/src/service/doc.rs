//! The native service behind the **Doc agent**: the application layer of the web
//! pipeline. It receives a parsed request (`"METHOD path"`) from the HTTP agent,
//! maps the path to a document under its own install folder, and **reads that
//! file with a capability-gated `mem_fs_read` tool call** — going through the
//! Synapse executor, so the read is capability-checked and scope-checked to the
//! agent's home (it holds only read access there). It returns the body to the
//! HTTP agent, which does all the HTTP framing.
//!
//! The Doc agent never speaks HTTP and never touches the socket; routing +
//! reading is its whole job.

use crate::cap::Right;
use crate::service::{pipeline, ServiceSpec};
use crate::synapse::registry;
use alloc::string::String;
use alloc::vec::Vec;

/// Map a request path to a `(asset filename, content-type)` under the agent's
/// `assets/` folder. `None` = 404. This is the Doc agent's router.
pub fn route(path: &str) -> Option<(&'static str, &'static str)> {
    match path {
        "/" | "/index.html" => Some(("index.html", "text/html; charset=utf-8")),
        "/docs" | "/docs.html" => Some(("docs.html", "text/html; charset=utf-8")),
        "/logo.svg" => Some(("logo.svg", "image/svg+xml")),
        _ => None,
    }
}

/// Serve one request path: route it to a file, read that file from `home` with a
/// `mem_fs_read` tool call, and build the content reply frame
/// (`"<status>\n<content-type>\n<body>"`) the HTTP agent formats. An unrouted
/// path, or a read the scope gate refuses / that isn't found, is a 404.
pub fn serve_path(home: &str, path: &str) -> Vec<u8> {
    let (status, ctype, body): (&str, &str, Vec<u8>) = match route(path) {
        Some((file, ctype)) => {
            // The file tool call — capability + scope gated to the agent's home.
            let call = alloc::format!(r#"{{"name":"mem_fs_read","arguments":{{"path":"{home}/assets/{file}"}}}}"#);
            match crate::synapse::execute(crate::sched::current_task_id(), &call) {
                crate::synapse::Invocation::Executed { result, .. } => match result.strip_prefix("ok:") {
                    Some(contents) => ("200 OK", ctype, contents.as_bytes().to_vec()),
                    None => ("404 Not Found", "text/html; charset=utf-8", b"<!doctype html><title>404</title><h1>Not found</h1>".to_vec()),
                },
                // Denied by the scope gate / no capability, or refused: don't leak why.
                _ => ("403 Forbidden", "text/html; charset=utf-8", b"<!doctype html><title>403</title><h1>Forbidden</h1>".to_vec()),
            }
        }
        None => ("404 Not Found", "text/html; charset=utf-8", b"<!doctype html><title>404</title><h1>Not found</h1>".to_vec()),
    };
    let mut out = alloc::format!("{status}\n{ctype}\n").into_bytes();
    out.extend_from_slice(&body);
    out
}

extern "C" fn doc_serve(_arg: u64) {
    let Some(from_http) = pipeline::http_to_doc() else {
        crate::ktrace::log("service.doc", "pipeline channels not wired");
        return;
    };
    let Some(to_http) = pipeline::doc_to_http() else { return };
    let home = pipeline::doc_home().unwrap_or_default();
    loop {
        if let Ok(Some(frame)) = crate::channel::try_recv_dgram(from_http) {
            // frame = "METHOD path"; the path is what we serve.
            let text = String::from_utf8_lossy(&frame);
            let path = text.split(' ').nth(1).unwrap_or("/");
            crate::ktrace::log_fmt(format_args!("service.doc: serving {path}"));
            let reply = serve_path(&home, path);
            pipeline::send_frame(to_http, &reply, crate::arch::now_ms() + pipeline::STAGE_DEADLINE_MS);
        }
        crate::shell::upkeep();
        crate::sched::yield_now();
    }
}

/// The Doc content service. Holds `InvokePrimitive(mem_fs_read)`; its read scope
/// (its own install folder) is granted by `pipeline::start`.
pub static DOC_STAGE: ServiceSpec =
    ServiceSpec { name: "doc", entry: doc_serve, autostart: false, caps: &[Right::InvokePrimitive(registry::MEM_FS_READ)] };

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn routes_paths_to_assets() {
        assert_eq!(route("/").map(|r| r.0), Some("index.html"));
        assert_eq!(route("/docs").map(|r| r.0), Some("docs.html"));
        assert_eq!(route("/logo.svg").map(|r| r.0), Some("logo.svg"));
        assert_eq!(route("/nope"), None);
        assert_eq!(route("/../SOUL.md"), None); // traversal routes to nothing
    }

    #[test_case]
    fn serves_a_file_via_a_scoped_tool_call() {
        use crate::agent::types::{CapDomain, CapabilityRequest, Rights, Scope};
        // Give this (bootstrap) task the read right + home scope, drop a file in
        // the "home", and confirm serve_path reads it back through the tool call.
        let me = crate::sched::current_task_id();
        crate::cap::grant(me, Right::InvokePrimitive(registry::MEM_FS_READ));
        crate::cap::grant_scopes(me, &[CapabilityRequest::new(CapDomain::Fs, Rights::READ, Scope::Path("/agent/doctest/**".into()))]);
        crate::synapse::fs::write("/agent/doctest/assets/index.html", b"<h1>hello docs</h1>");
        let reply = serve_path("/agent/doctest", "/");
        let s = String::from_utf8_lossy(&reply);
        assert!(s.starts_with("200 OK\n"), "reply: {s}");
        assert!(s.contains("<h1>hello docs</h1>"), "reply: {s}");
        // An unrouted path is a 404 (no read attempted).
        assert!(String::from_utf8_lossy(&serve_path("/agent/doctest", "/secret")).starts_with("404"));
    }
}
