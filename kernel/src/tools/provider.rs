//! In-kernel **tool providers** ("MCP servers" on bare metal): kernel modules
//! that register additional toolsets into the [`registry`](super::registry).
//! There is no network here, so a provider is just code that calls
//! [`registry::register`] at init. Phase F skill-bundled tools register through
//! this path, so they become normal Synapse-backed, capability-gated, audited
//! tools — never ambient authority.

use crate::tools::registry::{self, ToolBinding, ToolDef};
use alloc::string::ToString;
use alloc::vec::Vec;

/// A provider bundles a name and the tools it offers. Kept minimal: providers
/// are registered explicitly at boot (or at skill install, Phase G).
pub struct Provider {
    pub name: &'static str,
    pub tools: Vec<ToolDef>,
}

impl Provider {
    /// Register every tool this provider offers.
    pub fn register(self) {
        crate::ktrace::log_fmt(format_args!("tools.provider: '{}' registering {} tool(s)", self.name, self.tools.len()));
        for t in self.tools {
            registry::register(t);
        }
    }
}

/// Build a Synapse-bound tool definition from a skill's bundled-tool spec
/// (Phase F/G): the tool binds to an existing Synapse primitive by name. The
/// `arg_map` is identity (tool keys == primitive keys) for bundled tools, which
/// declare their own input schema.
pub fn synapse_tool(
    name: &str,
    description: &str,
    input_schema: &str,
    primitive: &'static str,
    arg_map: &'static [(&'static str, &'static str)],
    required: &[&str],
) -> ToolDef {
    ToolDef {
        name: name.to_string(),
        description: description.to_string(),
        input_schema: input_schema.to_string(),
        required: required.iter().map(|s| s.to_string()).collect(),
        binding: ToolBinding::Synapse { primitive, arg_map },
    }
}
