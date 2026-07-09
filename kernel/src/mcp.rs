//! **MCP client** — Model Context Protocol over HTTP (JSON-RPC 2.0), the
//! `/mcp connect <url>`. A connected server's tools are
//! registered into the tool registry under namespaced names
//! (`mcp__<server>__<tool>`) so the shell agent can call them exactly like a
//! built-in tool; the call is forwarded as a JSON-RPC `tools/call`.
//!
//! Transport is the modern **Streamable HTTP**: each request is an HTTP POST of
//! a JSON-RPC envelope; the response is either `application/json` or an
//! `text/event-stream` carrying the JSON in `data:` lines (both handled). A
//! server-assigned `Mcp-Session-Id` header is captured on `initialize` and
//! echoed on later requests.
//!
//! This sits **above** the determinism boundary as a tool provider: the model
//! chooses to call a tool; the actual HTTP effect goes through the deterministic
//! `net::http` client, and (for agents) the call is still capability-gated.

use crate::json::Json;
use crate::mm::Locked;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicI64, Ordering};

/// The MCP protocol revision we advertise on `initialize`.
const PROTOCOL_VERSION: &str = "2025-06-18";

/// A tool exposed by a connected server.
#[derive(Clone)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: String,
}

/// An MCP resource (URI-addressable content).
#[derive(Clone)]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    pub description: String,
    pub mime_type: String,
}

/// A live MCP server connection.
struct Server {
    name: String,
    url: String,
    bearer: Option<String>,
    session: Option<String>,
    tools: Vec<McpTool>,
    resources: Vec<McpResource>,
}

static SERVERS: Locked<Vec<Server>> = Locked::new(Vec::new());
static NEXT_ID: AtomicI64 = AtomicI64::new(1);

/// The namespaced registry name for a server's tool: `mcp__<server>__<tool>`.
pub fn tool_registry_name(server: &str, tool: &str) -> String {
    format!("mcp__{server}__{tool}")
}

/// Extract the JSON-RPC payload from an HTTP body that is either raw JSON or an
/// SSE stream (`data:` lines). For SSE, the `data:` payloads are concatenated
/// (a single JSON message may span lines); a plain body is returned as-is.
fn extract_json(body: &str) -> String {
    let t = body.trim();
    if t.starts_with('{') || t.starts_with('[') {
        return t.to_string();
    }
    // SSE: gather every `data:` line's payload.
    let mut out = String::new();
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            out.push_str(rest.trim());
        }
    }
    if out.is_empty() {
        body.to_string()
    } else {
        out
    }
}

/// Build a JSON-RPC request envelope string.
fn envelope(id: Option<i64>, method: &str, params: Json) -> String {
    let mut fields = vec![("jsonrpc".to_string(), Json::Str("2.0".to_string()))];
    if let Some(i) = id {
        fields.push(("id".to_string(), Json::Num(i as f64)));
    }
    fields.push(("method".to_string(), Json::Str(method.to_string())));
    fields.push(("params".to_string(), params));
    Json::Obj(fields).to_pretty()
}

/// One JSON-RPC round-trip. Returns `(result, new_session_id)`. A notification
/// (`id: None`) returns `Json::Null`. Maps a JSON-RPC `error` to `Err`.
fn rpc(url: &str, bearer: Option<&str>, session: Option<&str>, id: Option<i64>, method: &str, params: Json) -> Result<(Json, Option<String>), String> {
    let body = envelope(id, method, params);
    // Header value strings must outlive the borrow slice.
    let auth = bearer.map(|b| format!("Bearer {b}"));
    let mut headers: Vec<(&str, &str)> = vec![
        ("Content-Type", "application/json"),
        ("Accept", "application/json, text/event-stream"),
    ];
    if let Some(a) = auth.as_deref() {
        headers.push(("Authorization", a));
    }
    if let Some(s) = session {
        headers.push(("Mcp-Session-Id", s));
    }
    let resp = crate::net::http::request("POST", url, &headers, body.as_bytes(), 30_000)?;
    if !(200..300).contains(&resp.status) {
        return Err(format!("mcp: HTTP {} from {}", resp.status, url));
    }
    let new_session = resp.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("mcp-session-id")).map(|(_, v)| v.clone());
    if id.is_none() {
        return Ok((Json::Null, new_session)); // notification: no response body expected
    }
    let text = extract_json(&resp.text());
    let v = Json::parse(&text).ok_or_else(|| "mcp: response was not valid JSON-RPC".to_string())?;
    if let Some(err) = v.get("error") {
        let msg = err.get("message").and_then(|m| m.as_str()).unwrap_or("unknown error");
        let code = err.get("code").and_then(|c| c.as_i64()).unwrap_or(0);
        return Err(format!("mcp error {code}: {msg}"));
    }
    let result = v.get("result").cloned().ok_or_else(|| "mcp: response had no result".to_string())?;
    Ok((result, new_session))
}

/// Parse a `tools/list` result into [`McpTool`]s.
fn parse_tools(result: &Json) -> Vec<McpTool> {
    let mut out = Vec::new();
    let Some(arr) = result.get("tools").and_then(|t| t.as_array()) else { return out };
    for t in arr {
        let Some(name) = t.get("name").and_then(|n| n.as_str()) else { continue };
        let description = t.get("description").and_then(|d| d.as_str()).unwrap_or("").to_string();
        // Re-serialize the schema object (or default to an open object).
        let input_schema = t.get("inputSchema").map(|s| s.to_pretty()).unwrap_or_else(|| "{\"type\":\"object\"}".to_string());
        out.push(McpTool { name: name.to_string(), description, input_schema });
    }
    out
}

/// Parse a `resources/list` result.
fn parse_resources(result: &Json) -> Vec<McpResource> {
    let mut out = Vec::new();
    let Some(arr) = result.get("resources").and_then(|t| t.as_array()) else {
        return out;
    };
    for r in arr {
        let Some(uri) = r.get("uri").and_then(|u| u.as_str()) else { continue };
        let name = r.get("name").and_then(|n| n.as_str()).unwrap_or(uri).to_string();
        let description = r.get("description").and_then(|d| d.as_str()).unwrap_or("").to_string();
        let mime_type = r
            .get("mimeType")
            .and_then(|m| m.as_str())
            .unwrap_or("text/plain")
            .to_string();
        out.push(McpResource {
            uri: uri.to_string(),
            name,
            description,
            mime_type,
        });
    }
    out
}

/// The `initialize` params (client info + capabilities).
fn init_params() -> Json {
    Json::Obj(vec![
        ("protocolVersion".to_string(), Json::Str(PROTOCOL_VERSION.to_string())),
        ("capabilities".to_string(), Json::Obj(vec![])),
        (
            "clientInfo".to_string(),
            Json::Obj(vec![
                ("name".to_string(), Json::Str("chitti".to_string())),
                ("version".to_string(), Json::Str(crate::VERSION.to_string())),
            ]),
        ),
    ])
}

/// Connect to an MCP server at `url` (optionally bearer-authenticated), naming
/// the connection `name`. Runs `initialize` → `notifications/initialized` →
/// `tools/list` (+ best-effort `resources/list`), registers every tool into the
/// tool registry, and stores the connection. Returns the tool count on success.
/// Re-connecting the same name refreshes tools/resources (see [`reconnect`]).
pub fn connect(name: &str, url: &str, bearer: Option<&str>) -> Result<usize, String> {
    // Drop prior registration for this name so reconnect is clean.
    let _ = disconnect(name);

    let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
    let (result, session) = rpc(url, bearer, None, Some(id), "initialize", init_params())?;
    let server_name = result
        .get("serverInfo")
        .and_then(|s| s.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or(name)
        .to_string();
    // Best-effort "initialized" notification (some servers require it before
    // tools/list). Ignore transport hiccups here.
    let _ = rpc(url, bearer, session.as_deref(), None, "notifications/initialized", Json::Obj(vec![]));

    let list_id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
    let (tools_result, session2) = rpc(url, bearer, session.as_deref(), Some(list_id), "tools/list", Json::Obj(vec![]))?;
    let tools = parse_tools(&tools_result);
    let mut session = session2.or(session);

    // Best-effort resources/list — servers without resources just skip.
    let mut resources = Vec::new();
    {
        let rid = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        if let Ok((res_result, session3)) = rpc(url, bearer, session.as_deref(), Some(rid), "resources/list", Json::Obj(vec![])) {
            resources = parse_resources(&res_result);
            if let Some(s) = session3 {
                session = Some(s);
            }
        }
    }

    // Register each tool under its namespaced name (replacing on reconnect).
    for t in &tools {
        let reg_name = tool_registry_name(name, &t.name);
        let desc = format!("[mcp:{server_name}] {}", t.description);
        crate::tools::registry::register_replace(crate::tools::registry::ToolDef::mcp(&reg_name, &desc, &t.input_schema, name, &t.name));
    }

    let n_tools = tools.len();
    let n_res = resources.len();
    let server = Server {
        name: name.to_string(),
        url: url.to_string(),
        bearer: bearer.map(|b| b.to_string()),
        session,
        tools,
        resources,
    };
    SERVERS.with(|s| {
        s.retain(|old| old.name != name);
        s.push(server);
    });
    crate::ktrace::log_fmt(format_args!(
        "mcp: connected '{name}' ({url}) — {n_tools} tool(s), {n_res} resource(s)"
    ));
    Ok(n_tools)
}

/// Re-run connect for an existing server (same name/url/bearer). Fails if
/// unknown. Used by `/mcp reconnect`.
pub fn reconnect(name: &str) -> Result<usize, String> {
    let (url, bearer) = SERVERS.with(|s| {
        s.iter()
            .find(|sv| sv.name == name)
            .map(|sv| (sv.url.clone(), sv.bearer.clone()))
            .ok_or_else(|| format!("mcp: no connected server '{name}'"))
    })?;
    connect(name, &url, bearer.as_deref())
}

/// Call `tool` on connected `server` with a JSON `arguments` object (parsed
/// from `args_json`, or an empty object). Returns the text content of the
/// result, joined across content blocks.
pub fn call(server: &str, tool: &str, args_json: &str) -> Result<String, String> {
    let (url, bearer, session) = SERVERS.with(|s| {
        s.iter()
            .find(|sv| sv.name == server)
            .map(|sv| (sv.url.clone(), sv.bearer.clone(), sv.session.clone()))
            .ok_or_else(|| format!("mcp: no connected server '{server}' (try /mcp connect)"))
    })?;
    // Parse the arguments object; a bare non-object becomes {"input": <text>}.
    let arguments = match Json::parse(args_json.trim()) {
        Some(v @ Json::Obj(_)) => v,
        _ if args_json.trim().is_empty() => Json::Obj(vec![]),
        _ => Json::Obj(vec![("input".to_string(), Json::Str(args_json.trim().to_string()))]),
    };
    let params = Json::Obj(vec![
        ("name".to_string(), Json::Str(tool.to_string())),
        ("arguments".to_string(), arguments),
    ]);
    let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
    let (result, new_session) = rpc(&url, bearer.as_deref(), session.as_deref(), Some(id), "tools/call", params)?;
    if let Some(ns) = new_session {
        SERVERS.with(|s| {
            if let Some(sv) = s.iter_mut().find(|sv| sv.name == server) {
                sv.session = Some(ns);
            }
        });
    }
    Ok(render_content(&result))
}

/// Flatten an MCP tool result's `content` array into text (the shape the model
/// consumes). `isError: true` is prefixed so the agent sees the failure.
fn render_content(result: &Json) -> String {
    let mut out = String::new();
    if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
        for block in content {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                        if !out.is_empty() {
                            out.push('\n');
                        }
                        out.push_str(t);
                    }
                }
                Some(other) => {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(&format!("[{other} content]"));
                }
                None => {}
            }
        }
    }
    if out.is_empty() {
        out = result.to_pretty();
    }
    if result.get("isError").and_then(|e| e.as_bool()).unwrap_or(false) {
        format!("[tool error] {out}")
    } else {
        out
    }
}

/// Disconnect a server: drop the connection and de-register its tools. Returns
/// the number of tools removed, or `None` if no such server.
pub fn disconnect(name: &str) -> Option<usize> {
    let tools = SERVERS.with(|s| {
        let idx = s.iter().position(|sv| sv.name == name)?;
        let sv = s.remove(idx);
        Some(sv.tools)
    })?;
    for t in &tools {
        crate::tools::registry::deregister(&tool_registry_name(name, &t.name));
    }
    crate::ktrace::log_fmt(format_args!("mcp: disconnected '{name}' — {} tool(s) removed", tools.len()));
    Some(tools.len())
}

/// Connected servers as `(name, url, tool_count)` for `/mcp` status.
pub fn servers() -> Vec<(String, String, usize)> {
    SERVERS.with(|s| s.iter().map(|sv| (sv.name.clone(), sv.url.clone(), sv.tools.len())).collect())
}

/// Rich status lines for `/mcp status`: name, url, tools, resources, session.
pub fn status_lines() -> Vec<String> {
    SERVERS.with(|s| {
        s.iter()
            .map(|sv| {
                format!(
                    "{} — {} ({} tool(s), {} resource(s){})",
                    sv.name,
                    sv.url,
                    sv.tools.len(),
                    sv.resources.len(),
                    if sv.session.is_some() { ", session live" } else { "" }
                )
            })
            .collect()
    })
}

/// The tools of a connected server as `(tool_name, description)`.
pub fn server_tools(name: &str) -> Vec<(String, String)> {
    SERVERS.with(|s| {
        s.iter()
            .find(|sv| sv.name == name)
            .map(|sv| sv.tools.iter().map(|t| (t.name.clone(), t.description.clone())).collect())
            .unwrap_or_default()
    })
}

/// Full tool def for deferred discovery (`select:mcp__…`).
pub fn server_tool_schema(server: &str, tool: &str) -> Option<(String, String)> {
    SERVERS.with(|s| {
        s.iter().find(|sv| sv.name == server).and_then(|sv| {
            sv.tools
                .iter()
                .find(|t| t.name == tool)
                .map(|t| (t.description.clone(), t.input_schema.clone()))
        })
    })
}

/// Resources for one server, or all if `server` is `None`.
pub fn list_resources(server: Option<&str>) -> String {
    SERVERS.with(|s| {
        let mut out = String::new();
        for sv in s.iter() {
            if let Some(want) = server {
                if sv.name != want {
                    continue;
                }
            }
            if sv.resources.is_empty() {
                if server.is_some() {
                    out.push_str(&format!("(no resources on '{}')\n", sv.name));
                }
                continue;
            }
            for r in &sv.resources {
                out.push_str(&format!(
                    "[{}] {} — {} ({})\n",
                    sv.name, r.uri, r.name, r.mime_type
                ));
                if !r.description.is_empty() {
                    out.push_str(&format!("    {}\n", r.description));
                }
            }
        }
        if out.is_empty() {
            if server.is_some() {
                String::from("(server not connected or has no resources)")
            } else {
                String::from("(no MCP resources; connect a server with /mcp connect)")
            }
        } else {
            out
        }
    })
}

/// Read a resource URI from a connected server via `resources/read`.
pub fn read_resource(server: &str, uri: &str) -> Result<String, String> {
    let (url, bearer, session) = SERVERS.with(|s| {
        s.iter()
            .find(|sv| sv.name == server)
            .map(|sv| (sv.url.clone(), sv.bearer.clone(), sv.session.clone()))
            .ok_or_else(|| format!("mcp: no connected server '{server}'"))
    })?;
    let params = Json::Obj(vec![("uri".to_string(), Json::Str(uri.to_string()))]);
    let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
    let (result, new_session) = rpc(&url, bearer.as_deref(), session.as_deref(), Some(id), "resources/read", params)?;
    if let Some(ns) = new_session {
        SERVERS.with(|s| {
            if let Some(sv) = s.iter_mut().find(|sv| sv.name == server) {
                sv.session = Some(ns);
            }
        });
    }
    // contents[] with text or blob.
    if let Some(contents) = result.get("contents").and_then(|c| c.as_array()) {
        let mut out = String::new();
        for c in contents {
            if let Some(t) = c.get("text").and_then(|t| t.as_str()) {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(t);
            } else if let Some(b) = c.get("blob").and_then(|t| t.as_str()) {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&format!("[blob {} bytes base64]", b.len()));
            }
        }
        if !out.is_empty() {
            return Ok(out);
        }
    }
    Ok(result.to_pretty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn extracts_plain_and_sse_json() {
        assert_eq!(extract_json("  {\"a\":1}\n"), "{\"a\":1}");
        let sse = "event: message\ndata: {\"jsonrpc\":\"2.0\",\ndata: \"id\":1}\n\n";
        // data: lines concatenated → valid JSON.
        assert_eq!(extract_json(sse), "{\"jsonrpc\":\"2.0\",\"id\":1}");
    }

    #[test_case]
    fn namespaced_tool_names() {
        assert_eq!(tool_registry_name("weather", "forecast"), "mcp__weather__forecast");
    }

    #[test_case]
    fn envelope_is_valid_jsonrpc() {
        let e = envelope(Some(7), "tools/list", Json::Obj(alloc::vec![]));
        let v = Json::parse(&e).unwrap();
        assert_eq!(v.get("jsonrpc").and_then(|x| x.as_str()), Some("2.0"));
        assert_eq!(v.get("id").and_then(|x| x.as_i64()), Some(7));
        assert_eq!(v.get("method").and_then(|x| x.as_str()), Some("tools/list"));
        // A notification omits id.
        let n = envelope(None, "notifications/initialized", Json::Obj(alloc::vec![]));
        assert!(Json::parse(&n).unwrap().get("id").is_none());
    }

    #[test_case]
    fn parse_tools_and_render_content() {
        let listed = Json::parse(
            r#"{"tools":[{"name":"echo","description":"Echo text","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}}]}"#,
        )
        .unwrap();
        let tools = parse_tools(&listed);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");
        assert!(tools[0].input_schema.contains("text"));

        let ok = Json::parse(r#"{"content":[{"type":"text","text":"hello"},{"type":"text","text":"world"}]}"#).unwrap();
        assert_eq!(render_content(&ok), "hello\nworld");
        let err = Json::parse(r#"{"isError":true,"content":[{"type":"text","text":"boom"}]}"#).unwrap();
        assert!(render_content(&err).starts_with("[tool error] boom"));
    }

    #[test_case]
    fn parse_resources_list() {
        let listed = Json::parse(
            r#"{"resources":[{"uri":"file:///notes.txt","name":"notes","description":"demo","mimeType":"text/plain"}]}"#,
        )
        .unwrap();
        let res = parse_resources(&listed);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].uri, "file:///notes.txt");
        assert_eq!(res[0].name, "notes");
    }
}
