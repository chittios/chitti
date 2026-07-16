//! The Synapse **primitive registry** (`CHITTI_OS_HANDOFF.md` Phase 4):
//! the fixed, MCP-shaped catalogue of capability primitives an agent may
//! invoke. Each entry is a *name* plus a *JSON-schema-shaped* parameter
//! list (the "input schema" of an MCP tool), reduced to just the pieces the
//! deterministic executor and the constraint grammar actually need: the key
//! names, their types, and whether each is required.
//!
//! The registry is the single source of truth two other modules read:
//! `grammar` generates the constraint that forces model output into exactly
//! these shapes, and `executor` dispatches a validated call to native code.
//! Because it is `static`, the set of primitives is fixed at build time --
//! there is no runtime registration path an agent could use to smuggle in
//! new authority.

use crate::cap::PrimitiveId;

/// The JSON value type a parameter accepts. A deliberately tiny type
/// lattice -- strings and unsigned integers cover every Phase 4 primitive,
/// and keeping it small keeps the constraint grammar (`grammar.rs`)
/// decidable and prefix-closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgType {
    Str,
    Uint,
}

/// One named parameter of a primitive's input schema. Every parameter is
/// required: an MCP `required: [...]` list with no optionals, which keeps
/// both validation and the grammar unambiguous for Phase 4.
#[derive(Clone, Copy, Debug)]
pub struct Param {
    pub key: &'static str,
    pub ty: ArgType,
}

/// A registered primitive: its stable id, wire name, ordered parameter
/// schema, and a one-line description (the MCP `description` field, surfaced
/// to the model when the tool list is rendered in later phases).
#[derive(Clone, Copy, Debug)]
pub struct PrimitiveSpec {
    pub id: PrimitiveId,
    pub name: &'static str,
    /// Canonical parameter order. The grammar emits/accepts arguments in
    /// exactly this order, so a well-formed call is unambiguous to parse.
    pub params: &'static [Param],
    pub description: &'static str,
    /// Whether this primitive is destructive / irreversible. The Synapse
    /// taint gate (Phase 6) refuses a destructive call whose justification
    /// traces to untrusted, ingested content unless a human confirms it.
    pub destructive: bool,
}

// Stable primitive ids. These are also the `cap::Right::InvokePrimitive`
// discriminants a task is granted, so they must never be renumbered.
pub const CONSOLE_WRITE: PrimitiveId = 0;
pub const MEM_FS_READ: PrimitiveId = 1;
pub const MEM_FS_WRITE: PrimitiveId = 2;
pub const LIST: PrimitiveId = 3;
pub const SPAWN_AGENT: PrimitiveId = 4;
pub const SLEEP: PrimitiveId = 5;
pub const EMIT_RESULT: PrimitiveId = 6;
pub const MEM_FS_DELETE: PrimitiveId = 7;
pub const MEM_FS_EDIT: PrimitiveId = 8;
pub const MEM_FS_SEARCH: PrimitiveId = 9;
// Inter-agent channels (Phase 1). Handle args (`chan`) are the caller's own
// `Cap` slot index, resolved by the executor — never a global id.
pub const CHANNEL_CREATE: PrimitiveId = 10;
pub const CHANNEL_WRITE: PrimitiveId = 11;
pub const CHANNEL_READ: PrimitiveId = 12;
pub const CHANNEL_CLOSE: PrimitiveId = 13;
// Hand (a copy of) a channel end to another agent (Phase 2). Destructive: it
// moves authority to another principal, so injected content can't silently
// exfiltrate a stream to an attacker-controlled agent without a human confirm.
pub const CHANNEL_GRANT: PrimitiveId = 14;
// Network capability (Phase 3). `net_listen` binds a port (destructive: an
// externally-visible effect). `net_accept` hands out an inbound connection as a
// channel. `net_http_get`/`net_http_post` are scope-gated egress; POST is
// destructive (an exfiltration vector under prompt injection).
pub const NET_LISTEN: PrimitiveId = 15;
pub const NET_ACCEPT: PrimitiveId = 16;
pub const NET_HTTP_GET: PrimitiveId = 17;
pub const NET_HTTP_POST: PrimitiveId = 18;
// UI surfaces (Phase 4): a Chess/Image/Video/Browser/Doc agent owns a surface
// and paints it with a bounded draw-op DSL. Gated by surface ownership, not just
// the primitive right. None are destructive (drawing is reversible).
pub const UI_SURFACE_REQUEST: PrimitiveId = 19;
pub const UI_DRAW: PrimitiveId = 20;
pub const UI_EVENT_POLL: PrimitiveId = 21;
pub const UI_SURFACE_CLOSE: PrimitiveId = 22;
// High-level board presentation for UI agents (Chess etc.): one structured
// call paints a FEN or square marks — models emit these instead of dozens of
// raw rects. Still ownership-gated like ui_draw.
pub const BOARD_SET: PrimitiveId = 23;
pub const BOARD_MARK: PrimitiveId = 24;
/// Set a surface's HUD text (status + wrapped hints), rendered by the
/// compositor in a reserved pane-space strip. Ownership-gated like ui_draw.
pub const UI_HUD: PrimitiveId = 25;

const STR: ArgType = ArgType::Str;
const UINT: ArgType = ArgType::Uint;

/// The complete Phase 4 primitive set. Indexed by `PrimitiveId` (each
/// entry's `id` equals its position), which `by_id` relies on.
pub static REGISTRY: &[PrimitiveSpec] = &[
    PrimitiveSpec {
        id: CONSOLE_WRITE,
        name: "console_write",
        params: &[Param { key: "text", ty: STR }],
        description: "Write a line of text to the system console.",
        destructive: false,
    },
    PrimitiveSpec {
        id: MEM_FS_READ,
        name: "mem_fs_read",
        params: &[Param { key: "path", ty: STR }],
        description: "Read the contents of a file from the in-memory store.",
        destructive: false,
    },
    PrimitiveSpec {
        id: MEM_FS_WRITE,
        name: "mem_fs_write",
        params: &[Param { key: "path", ty: STR }, Param { key: "text", ty: STR }],
        description: "Write text to a file in the in-memory store, creating or replacing it.",
        destructive: false,
    },
    PrimitiveSpec {
        id: LIST,
        name: "list",
        params: &[],
        description: "List the file paths present in the in-memory store.",
        destructive: false,
    },
    PrimitiveSpec {
        id: SPAWN_AGENT,
        name: "spawn_agent",
        params: &[Param { key: "persona", ty: STR }],
        description: "Request a new agent be spawned with the given persona (lifecycle lands in Phase 5).",
        destructive: false,
    },
    PrimitiveSpec {
        id: SLEEP,
        name: "sleep",
        params: &[Param { key: "ticks", ty: UINT }],
        description: "Yield for a number of scheduler ticks.",
        destructive: false,
    },
    PrimitiveSpec {
        id: EMIT_RESULT,
        name: "emit_result",
        params: &[Param { key: "text", ty: STR }],
        description: "Report the agent's final result for this intent.",
        destructive: false,
    },
    PrimitiveSpec {
        id: MEM_FS_DELETE,
        name: "mem_fs_delete",
        params: &[Param { key: "path", ty: STR }],
        description: "Delete a file from the in-memory store. Destructive and irreversible.",
        destructive: true,
    },
    PrimitiveSpec {
        id: MEM_FS_EDIT,
        name: "mem_fs_edit",
        params: &[Param { key: "path", ty: STR }, Param { key: "old", ty: STR }, Param { key: "new", ty: STR }],
        description: "Replace the first occurrence of `old` with `new` in a file. Not destructive (reversible edit).",
        destructive: false,
    },
    PrimitiveSpec {
        id: MEM_FS_SEARCH,
        name: "mem_fs_search",
        params: &[Param { key: "query", ty: STR }],
        description: "List the paths of files whose contents contain `query`.",
        destructive: false,
    },
    PrimitiveSpec {
        id: CHANNEL_CREATE,
        name: "channel_create",
        params: &[Param { key: "kind", ty: STR }],
        description: "Create an inter-agent channel (kind: \"stream\" or \"datagram\"). Returns the read and write end cap slots granted to the caller.",
        destructive: false,
    },
    PrimitiveSpec {
        id: CHANNEL_WRITE,
        name: "channel_write",
        params: &[Param { key: "chan", ty: UINT }, Param { key: "text", ty: STR }],
        description: "Write text bytes to a channel. `chan` is the caller's write-end cap slot. Returns bytes written or blocked.",
        destructive: false,
    },
    PrimitiveSpec {
        id: CHANNEL_READ,
        name: "channel_read",
        params: &[Param { key: "chan", ty: UINT }, Param { key: "max", ty: UINT }],
        description: "Read up to `max` bytes from a channel. `chan` is the caller's read-end cap slot. Cooperatively blocks briefly; returns data or eof.",
        destructive: false,
    },
    PrimitiveSpec {
        id: CHANNEL_CLOSE,
        name: "channel_close",
        params: &[Param { key: "chan", ty: UINT }],
        description: "Close the caller's channel end named by cap slot `chan`.",
        destructive: false,
    },
    PrimitiveSpec {
        id: CHANNEL_GRANT,
        name: "channel_grant",
        params: &[Param { key: "chan", ty: UINT }, Param { key: "to_agent", ty: STR }],
        description: "Hand the channel end at cap slot `chan` to another agent (by service name or task id). Destructive: moves authority to another principal.",
        destructive: true,
    },
    PrimitiveSpec {
        id: NET_LISTEN,
        name: "net_listen",
        params: &[Param { key: "port", ty: UINT }, Param { key: "proto", ty: STR }],
        description: "Listen for inbound TCP connections on a port. Returns a listener cap slot. Destructive: binds an externally-visible port.",
        destructive: true,
    },
    PrimitiveSpec {
        id: NET_ACCEPT,
        name: "net_accept",
        params: &[Param { key: "listener", ty: UINT }],
        description: "Accept one inbound connection on a listener (cap slot). Returns the connection's read/write channel end cap slots. Blocks briefly.",
        destructive: false,
    },
    PrimitiveSpec {
        id: NET_HTTP_GET,
        name: "net_http_get",
        params: &[Param { key: "url", ty: STR }],
        description: "HTTP GET a URL (scope-gated by host/port). Returns the response body.",
        destructive: false,
    },
    PrimitiveSpec {
        id: NET_HTTP_POST,
        name: "net_http_post",
        params: &[Param { key: "url", ty: STR }, Param { key: "body", ty: STR }],
        description: "HTTP POST a body to a URL (scope-gated). Destructive: network egress that can exfiltrate.",
        destructive: true,
    },
    PrimitiveSpec {
        id: UI_SURFACE_REQUEST,
        name: "ui_surface_request",
        params: &[Param { key: "kind", ty: STR }],
        description: "Request a drawing surface (kind: canvas|board|image|video|html). Returns its surface id.",
        destructive: false,
    },
    PrimitiveSpec {
        id: UI_DRAW,
        name: "ui_draw",
        params: &[Param { key: "surface", ty: UINT }, Param { key: "ops", ty: STR }],
        description: "Paint a surface you own with draw ops: 'clear <hex>; rect x y w h <hex>; line x0 y0 x1 y1 <hex>; pixel x y <hex>'.",
        destructive: false,
    },
    PrimitiveSpec {
        id: UI_EVENT_POLL,
        name: "ui_event_poll",
        params: &[Param { key: "surface", ty: UINT }],
        description: "Poll one input event (click/key) for a surface you own. Returns the event or none.",
        destructive: false,
    },
    PrimitiveSpec {
        id: UI_SURFACE_CLOSE,
        name: "ui_surface_close",
        params: &[Param { key: "surface", ty: UINT }],
        description: "Close a surface you own.",
        destructive: false,
    },
    PrimitiveSpec {
        id: BOARD_SET,
        name: "board_set",
        params: &[Param { key: "surface", ty: UINT }, Param { key: "fen", ty: STR }],
        description: "Paint an 8x8 chess board from a FEN string onto a surface you own.",
        destructive: false,
    },
    PrimitiveSpec {
        id: BOARD_MARK,
        name: "board_mark",
        params: &[Param { key: "surface", ty: UINT }, Param { key: "squares", ty: STR }, Param { key: "color", ty: STR }],
        description: "Highlight squares (e.g. 'e2,e4') on a board surface you own.",
        destructive: false,
    },
    PrimitiveSpec {
        id: UI_HUD,
        name: "ui_hud",
        params: &[Param { key: "surface", ty: UINT }, Param { key: "text", ty: STR }],
        description: "Set a surface's HUD (status + wrapped hint lines, '\\n'-separated), shown in a reserved strip below the surface. Empty clears it.",
        destructive: false,
    },
];

/// Look up a primitive by its wire name. `None` for any name not in the
/// registry -- the grammar rejects those before this is ever reached, but
/// the executor still treats an unknown id as a hard error.
pub fn by_name(name: &str) -> Option<&'static PrimitiveSpec> {
    REGISTRY.iter().find(|p| p.name == name)
}

/// Look up a primitive by id (its index in `REGISTRY`).
pub fn by_id(id: PrimitiveId) -> Option<&'static PrimitiveSpec> {
    REGISTRY.get(id as usize)
}

/// Whether `name` is a prefix of at least one registered primitive name.
/// The constraint grammar uses this while a name is still being read, to
/// reject an impossible name as early as the first divergent byte rather
/// than only once the closing quote arrives.
pub fn is_name_prefix(name: &str) -> bool {
    REGISTRY.iter().any(|p| p.name.as_bytes().starts_with(name.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn ids_match_registry_positions() {
        for (i, spec) in REGISTRY.iter().enumerate() {
            assert_eq!(spec.id as usize, i, "primitive {} has id/index mismatch", spec.name);
            assert!(by_id(spec.id).is_some());
            assert_eq!(by_name(spec.name).unwrap().id, spec.id);
        }
    }

    #[test_case]
    fn unknown_names_are_absent() {
        assert!(by_name("rm_rf").is_none());
        assert!(!is_name_prefix("mem_fs_destroy"));
        // A genuine prefix of a real name is a viable prefix.
        assert!(is_name_prefix("mem_fs_"));
        assert!(is_name_prefix(""));
    }

    #[test_case]
    fn destructive_primitives_are_exactly_the_known_set() {
        // The taint gate keys off this flag, so guard against a careless future
        // addition: any new destructive primitive must be a deliberate edit here.
        // `mem_fs_delete` (irreversible), `channel_grant` (moves authority to
        // another principal — an exfiltration vector under prompt injection).
        let destructive: alloc::vec::Vec<_> = REGISTRY.iter().filter(|p| p.destructive).map(|p| p.name).collect();
        assert_eq!(destructive, alloc::vec!["mem_fs_delete", "channel_grant", "net_listen", "net_http_post"]);
    }
}
