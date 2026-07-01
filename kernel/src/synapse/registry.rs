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
    },
    PrimitiveSpec {
        id: MEM_FS_READ,
        name: "mem_fs_read",
        params: &[Param { key: "path", ty: STR }],
        description: "Read the contents of a file from the in-memory store.",
    },
    PrimitiveSpec {
        id: MEM_FS_WRITE,
        name: "mem_fs_write",
        params: &[Param { key: "path", ty: STR }, Param { key: "text", ty: STR }],
        description: "Write text to a file in the in-memory store, creating or replacing it.",
    },
    PrimitiveSpec {
        id: LIST,
        name: "list",
        params: &[],
        description: "List the file paths present in the in-memory store.",
    },
    PrimitiveSpec {
        id: SPAWN_AGENT,
        name: "spawn_agent",
        params: &[Param { key: "persona", ty: STR }],
        description: "Request a new agent be spawned with the given persona (lifecycle lands in Phase 5).",
    },
    PrimitiveSpec {
        id: SLEEP,
        name: "sleep",
        params: &[Param { key: "ticks", ty: UINT }],
        description: "Yield for a number of scheduler ticks.",
    },
    PrimitiveSpec {
        id: EMIT_RESULT,
        name: "emit_result",
        params: &[Param { key: "text", ty: STR }],
        description: "Report the agent's final result for this intent.",
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
        assert!(!is_name_prefix("mem_fs_delete"));
        // A genuine prefix of a real name is a viable prefix.
        assert!(is_name_prefix("mem_fs_"));
        assert!(is_name_prefix(""));
    }
}
