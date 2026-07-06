//! The **generic content-server runtime**: the third stage of the web pipeline,
//! and the whole point of "write a server with just a SOUL.md + assets". It is
//! NOT specific to any one agent — it serves *whichever* content agent the
//! pipeline was started for (its install folder, set by `pipeline::start`).
//!
//! For each request it runs that agent's *model* as a bounded ReAct loop
//! (prompted with the agent's own `SOUL.md`, where the author wrote the routing
//! policy). The agent **reads the file it decides to serve itself**, via a
//! capability- and scope-gated `mem_fs_read` tool call confined to the agent's
//! own `assets/`, and then returns a **JSON response object**
//! (`{status, content_type/headers, body}`). This runtime parses that JSON and
//! builds the reply frame for the HTTP stage — the whole HTTP response is decided
//! by the SOUL agent, not any compiled-in table.
//!
//! So a new web server is just `agents/<name>/{SOUL.md, manifest.json, assets/…}`:
//! the SOUL carries the routing/behaviour, the assets carry the content, and this
//! runtime (plus the generic Network + HTTP agents) does the rest — no per-server
//! Rust. Determinism boundary: the model *decides* and *reads* (through the gated
//! tool call); native code only parses its JSON and frames bytes — the model
//! never touches the socket or the raw capability.

use crate::cap::Right;
use crate::service::{pipeline, ServiceSpec};
use crate::synapse::registry;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Cache of `(home, soul)` — the SOUL is the system prompt for every request, so
/// read it once per served agent rather than each time.
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
    read_via_tool(path).map(String::into_bytes)
}

/// Resolve the `mem_fs_read` path argument the agent gave — either a full path
/// under the agent's `assets/`, or a bare filename (resolved into `assets/`) — so
/// a small model can just name the file. Confinement + the gate are enforced by
/// [`read_asset`].
pub fn read_asset_arg(home: &str, arg: &str) -> Option<Vec<u8>> {
    let base = alloc::format!("{home}/assets/");
    let full = if arg.starts_with(&base) {
        arg.to_string()
    } else {
        // Treat anything else as a filename (drop any leading dirs) under assets/.
        let name = arg.trim().rsplit('/').next().unwrap_or(arg);
        alloc::format!("{base}{name}")
    };
    read_asset(home, &full)
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

/// Guess a content-type from a file's extension — the fallback when the agent's
/// JSON response omits one but it did read a file.
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

/// The served agent's SOUL.md (cached per home) — the persona + routing policy
/// its author wrote, used as the system prompt for the model.
fn soul(home: &str) -> String {
    if let Some((h, s)) = SOUL.with(|c| c.clone()) {
        if h == home {
            return s;
        }
    }
    let s = read_via_tool(&alloc::format!("{home}/SOUL.md"))
        .unwrap_or_else(|| String::from("You serve files from your assets/ folder for each HTTP request."));
    SOUL.with(|c| *c = Some((home.to_string(), s.clone())));
    s
}

/// The parsed content-agent response: an HTTP status line, response headers, and
/// the body — as an optional inline `body`, and/or a `file` naming an asset the
/// server reads through the gated tool (either is the agent *deciding* the
/// content; the read stays capability- and scope-checked).
struct Response {
    status: String,
    headers: Vec<(String, String)>,
    body: Option<Vec<u8>>,
    file: Option<String>,
}

/// The reason phrase for a numeric status code, so the agent can answer with a
/// bare number (`{"status": 404}`) and still produce a valid HTTP status line.
fn status_line(code: u32) -> String {
    let reason = match code {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        301 => "Moved Permanently",
        302 => "Found",
        304 => "Not Modified",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    };
    alloc::format!("{code} {reason}")
}

/// Extract the first balanced `{…}` JSON object from the model's reply (which may
/// carry surrounding prose or a code fence), tracking strings + escapes so a
/// brace inside a body string doesn't end it early. `None` if there is no object.
fn extract_json_object(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let start = text.find('{')?;
    let (mut depth, mut in_str, mut esc) = (0i32, false, false);
    let mut i = start;
    while i < bytes.len() {
        let c = bytes[i];
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
        } else {
            match c {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(text[start..=i].to_string());
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    None
}

/// Parse the agent's final JSON response object into a [`Response`]. Accepts
/// `status` as a number (`200`) or a full status line (`"200 OK"`), an optional
/// `content_type` shortcut and/or a `headers` object, and an optional string
/// `body`. `None` if no JSON object is present or it doesn't parse.
fn parse_response(text: &str) -> Option<Response> {
    let obj = extract_json_object(text)?;
    let j = crate::json::Json::parse(&obj)?;
    let status = match j.get("status") {
        Some(crate::json::Json::Num(n)) => status_line(*n as u32),
        Some(crate::json::Json::Str(s)) => s.clone(),
        _ => String::from("200 OK"),
    };
    let mut headers: Vec<(String, String)> = Vec::new();
    if let Some(ct) = j.get("content_type").and_then(|v| v.as_str()) {
        headers.push((String::from("Content-Type"), ct.to_string()));
    }
    if let Some(crate::json::Json::Obj(pairs)) = j.get("headers") {
        for (k, v) in pairs {
            if let Some(vs) = v.as_str() {
                headers.push((k.clone(), vs.to_string()));
            }
        }
    }
    let body = j.get("body").and_then(|v| v.as_str()).map(|s| s.as_bytes().to_vec());
    // `file`/`path` (either spelling) names an asset for the server to read.
    let file = j
        .get("file")
        .or_else(|| j.get("path"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(String::from);
    Some(Response { status, headers, body, file })
}

/// Ensure a `Content-Type` header is present: keep the agent's if it set one,
/// else derive it from the extension of the file it read, else default to HTML.
fn ensure_content_type(headers: &mut Vec<(String, String)>, read_path: Option<&str>) {
    if headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-type")) {
        return;
    }
    let ct = read_path.map(ctype_for).unwrap_or("text/html; charset=utf-8");
    headers.push((String::from("Content-Type"), ct.to_string()));
}

/// Build the content reply frame the HTTP agent formats into a response:
/// `"<status>\n<Header: Value>\n…\n\n<body>"` — the status line, the response
/// headers, a blank line, then the raw body. This is the whole server→http
/// contract (the HTTP stage adds `Content-Length` + `Connection: close`).
fn build_frame(status: &str, headers: &[(String, String)], body: &[u8]) -> Vec<u8> {
    let mut head = alloc::format!("{status}\n");
    for (k, v) in headers {
        head.push_str(k);
        head.push_str(": ");
        head.push_str(v);
        head.push('\n');
    }
    head.push('\n'); // blank line: end of headers, start of body
    let mut out = head.into_bytes();
    out.extend_from_slice(body);
    out
}

/// A default HTML body for a status with no content (e.g. a 404 the agent named
/// without a body).
fn default_body(status: &str) -> Vec<u8> {
    alloc::format!("<!doctype html><title>{status}</title><h1>{status}</h1>").into_bytes()
}

/// Serve one request for the agent installed at `home`: run its model as a ReAct
/// loop (prompted with the agent's SOUL) in which the agent reads the file it
/// serves through the scoped `mem_fs_read` tool and returns a JSON response
/// object; parse that JSON and frame the response. The model *decides and reads*
/// (a judgment from the SOUL, through the gated tool); native code only parses +
/// frames. The body is the agent's inline `body` if present, otherwise the file
/// it read (so a small model need not echo a whole page).
pub fn serve(home: &str, method: &str, path: &str) -> Vec<u8> {
    let persona = soul(home);
    let user = alloc::format!("Serve the HTTP request:\n{method} {path}");
    let (reply, last_read) = match crate::shell::serve_reply(&persona, &user, home) {
        Some(r) => r,
        None => {
            crate::ktrace::log("service.server", "serve: no model loaded to run the agent");
            return build_frame(
                "503 Service Unavailable",
                &[(String::from("Content-Type"), String::from("text/html; charset=utf-8"))],
                b"<!doctype html><title>503</title><h1>Server agent: no model loaded</h1>",
            );
        }
    };
    match parse_response(&reply) {
        Some(mut r) => {
            // Resolve the body, in order: the agent's inline `body`; the file it
            // read via the tool; the asset it named in `file` (read now, gated);
            // else a default page. `ct_path` remembers which asset supplied the
            // body so the content-type can be inferred if the agent omitted one.
            let mut ct_path: Option<String> = last_read.as_ref().map(|(p, _)| p.clone());
            let body = match r.body.take() {
                Some(b) if !b.is_empty() => b,
                _ => {
                    if let Some((_, b)) = last_read {
                        b
                    } else if let Some(file) = r.file.as_deref().and_then(|f| read_asset_arg(home, f).map(|b| (f, b))) {
                        ct_path = Some(file.0.to_string());
                        file.1
                    } else {
                        default_body(&r.status)
                    }
                }
            };
            ensure_content_type(&mut r.headers, ct_path.as_deref());
            crate::ktrace::log_fmt(format_args!("service.server: agent replied {} ({} bytes)", r.status, body.len()));
            build_frame(&r.status, &r.headers, &body)
        }
        None => {
            // No parseable JSON: if the agent nevertheless read a file, serve it
            // (its routing decision still lands); otherwise 404.
            match last_read {
                Some((p, bytes)) => {
                    crate::ktrace::log_fmt(format_args!("service.server: no JSON reply; serving read asset {p}"));
                    build_frame("200 OK", &[(String::from("Content-Type"), ctype_for(&p).to_string())], &bytes)
                }
                None => {
                    crate::ktrace::log("service.server", "agent returned no JSON and read nothing (404)");
                    build_frame(
                        "404 Not Found",
                        &[(String::from("Content-Type"), String::from("text/html; charset=utf-8"))],
                        b"<!doctype html><title>404</title><h1>Not found</h1>",
                    )
                }
            }
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
            // req = "METHOD path"; the agent routes on both.
            let text = String::from_utf8_lossy(&req);
            let mut it = text.split(' ');
            let method = it.next().unwrap_or("GET").to_string();
            let path = it.next().unwrap_or("/").to_string();
            crate::ktrace::log_fmt(format_args!("service.server: agent serving {method} {path} (agent {home})"));
            let reply = serve(&home, &method, &path);
            pipeline::send_frame(to_http, &reply, crate::arch::now_ms() + pipeline::STAGE_DEADLINE_MS);
        }
        crate::shell::upkeep();
        crate::sched::yield_now();
    }
}

/// The generic content-server stage. Holds `InvokePrimitive(mem_fs_read)`; the
/// served agent's read scope (its own install folder) is granted by
/// `pipeline::start`. Serves any content agent — the response is planned and read
/// by that agent's model from its SOUL, never a compiled-in table.
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
        // A bare filename resolves into assets/ (what the agent typically names).
        assert_eq!(read_asset_arg("/agent/srvtest", "index.html").as_deref(), Some(&b"<h1>hi</h1>"[..]));
        // Never serves files outside assets/ (e.g. the SOUL), nor traversals.
        assert_eq!(read_asset("/agent/srvtest", "/agent/srvtest/SOUL.md"), None);
        assert_eq!(read_asset("/agent/srvtest", "/agent/srvtest/assets/../SOUL.md"), None);
        assert_eq!(read_asset_arg("/agent/srvtest", "../SOUL.md"), None); // basename → assets/SOUL.md (absent)
    }

    #[test_case]
    fn extracts_json_object_from_noisy_reply() {
        // Leading prose + a body brace must not truncate the object early.
        let obj = extract_json_object("Sure: {\"status\":200,\"body\":\"<div>{x}</div>\"} done").unwrap();
        assert_eq!(obj, "{\"status\":200,\"body\":\"<div>{x}</div>\"}");
        assert!(extract_json_object("no object here").is_none());
    }

    #[test_case]
    fn parses_agent_json_response() {
        let r = parse_response("{\"status\": 200, \"content_type\": \"text/html\", \"body\": \"<h1>hi</h1>\"}").unwrap();
        assert_eq!(r.status, "200 OK");
        assert_eq!(r.headers, alloc::vec![(String::from("Content-Type"), String::from("text/html"))]);
        assert_eq!(r.body.as_deref(), Some(&b"<h1>hi</h1>"[..]));
        assert!(r.file.is_none());
        // The agent names an asset via `file` (server reads it): number status,
        // headers object, no inline body.
        let r2 = parse_response("{\"status\": 404, \"headers\": {\"X-A\": \"b\"}}").unwrap();
        assert_eq!(r2.status, "404 Not Found");
        assert_eq!(r2.headers, alloc::vec![(String::from("X-A"), String::from("b"))]);
        assert!(r2.body.is_none());
        let r3 = parse_response("{\"status\": 200, \"content_type\": \"text/html; charset=utf-8\", \"file\": \"index.html\"}").unwrap();
        assert_eq!(r3.file.as_deref(), Some("index.html"));
        assert!(r3.body.is_none());
    }

    #[test_case]
    fn builds_a_multi_header_frame() {
        let f = build_frame("200 OK", &[(String::from("Content-Type"), String::from("text/html"))], b"body");
        assert_eq!(f, b"200 OK\nContent-Type: text/html\n\nbody");
    }

    #[test_case]
    fn content_type_from_extension() {
        assert_eq!(ctype_for("/x/a.html"), "text/html; charset=utf-8");
        assert_eq!(ctype_for("/x/a.svg"), "image/svg+xml");
        assert_eq!(ctype_for("/x/a.json"), "application/json");
    }
}
