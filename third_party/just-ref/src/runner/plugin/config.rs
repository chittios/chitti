//! Plugin configuration file parsing.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::path::Path;
#[cfg(feature = "std")]
use std::fs;

use super::registry::PluginError;

/// Configuration for a native plugin.
#[derive(Debug, Clone)]
pub struct NativePluginConfig {
    /// Path to the dynamic library.
    pub path: String,
    /// Whether the plugin is enabled.
    pub enabled: bool,
}

/// Configuration for a JavaScript plugin.
#[derive(Debug, Clone)]
pub struct JsPluginConfig {
    /// Path to the JavaScript file.
    pub path: String,
    /// Whether the plugin is enabled.
    pub enabled: bool,
}

/// Method override configuration.
#[derive(Debug, Clone)]
pub struct OverrideConfig {
    /// Plugin name providing the override.
    pub plugin: String,
    /// Method name in the plugin.
    pub method: String,
}

/// Complete plugin configuration.
#[derive(Debug, Clone)]
pub struct PluginConfig {
    /// Native plugins to load.
    pub native_plugins: Vec<NativePluginConfig>,
    /// JavaScript plugins to load.
    pub js_plugins: Vec<JsPluginConfig>,
    /// Method overrides (key: "Object.method").
    pub overrides: hashbrown::HashMap<String, OverrideConfig>,
}

impl PluginConfig {
    /// Create an empty configuration.
    pub fn new() -> Self {
        PluginConfig {
            native_plugins: Vec::new(),
            js_plugins: Vec::new(),
            overrides: hashbrown::HashMap::new(),
        }
    }

    /// Load configuration from a TOML file.
    ///
    /// Expected format:
    /// ```toml
    /// [plugins]
    /// native = [
    ///     { path = "./plugins/libcustom.so", enabled = true }
    /// ]
    /// javascript = [
    ///     { path = "./plugins/utils.js", enabled = true }
    /// ]
    ///
    /// [overrides]
    /// "Array.map" = { plugin = "fast-array", method = "fastMap" }
    /// ```
    #[cfg(feature = "std")]
    pub fn load(path: &Path) -> Result<Self, PluginError> {
        let content = fs::read_to_string(path)
            .map_err(|e| PluginError::ConfigError(format!("Failed to read config file: {}", e)))?;

        Self::parse(&content)
    }

    /// Parse configuration from a TOML string.
    /// Note: This is a simple parser. For production use, consider using the `toml` crate.
    pub fn parse(content: &str) -> Result<Self, PluginError> {
        let mut config = PluginConfig::new();

        // Simple line-by-line parsing
        // This is a basic implementation - full TOML parsing would use the `toml` crate
        let mut current_section = String::new();
        let mut in_native_array = false;
        let mut in_js_array = false;

        for line in content.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Section headers
            if line.starts_with('[') && line.ends_with(']') {
                current_section = line[1..line.len() - 1].to_string();
                in_native_array = false;
                in_js_array = false;
                continue;
            }

            // Handle plugins section
            if current_section == "plugins" {
                if line.starts_with("native") {
                    in_native_array = line.contains('[');
                    in_js_array = false;
                } else if line.starts_with("javascript") {
                    in_js_array = line.contains('[');
                    in_native_array = false;
                } else if line == "]" {
                    in_native_array = false;
                    in_js_array = false;
                } else if in_native_array || in_js_array {
                    // Parse plugin entry: { path = "...", enabled = true/false }
                    if let Some(entry) = Self::parse_plugin_entry(line) {
                        if in_native_array {
                            config.native_plugins.push(NativePluginConfig {
                                path: entry.0,
                                enabled: entry.1,
                            });
                        } else {
                            config.js_plugins.push(JsPluginConfig {
                                path: entry.0,
                                enabled: entry.1,
                            });
                        }
                    }
                }
            }

            // Handle overrides section
            if current_section == "overrides" {
                if let Some((key, override_config)) = Self::parse_override_entry(line) {
                    config.overrides.insert(key, override_config);
                }
            }
        }

        Ok(config)
    }

    /// Parse a plugin entry like: { path = "./plugins/lib.so", enabled = true }
    fn parse_plugin_entry(line: &str) -> Option<(String, bool)> {
        let line = line.trim().trim_start_matches('{').trim_end_matches('}').trim_end_matches(',');

        let mut path = None;
        let mut enabled = true;

        for part in line.split(',') {
            let part = part.trim();
            if part.starts_with("path") {
                if let Some(value) = part.split('=').nth(1) {
                    path = Some(
                        value
                            .trim()
                            .trim_matches('"')
                            .to_string(),
                    );
                }
            } else if part.starts_with("enabled") {
                if let Some(value) = part.split('=').nth(1) {
                    enabled = value.trim() == "true";
                }
            }
        }

        path.map(|p| (p, enabled))
    }

    /// Parse an override entry like: "Array.map" = { plugin = "fast-array", method = "fastMap" }
    fn parse_override_entry(line: &str) -> Option<(String, OverrideConfig)> {
        let parts: Vec<&str> = line.splitn(2, '=').collect();
        if parts.len() != 2 {
            return None;
        }

        let key = parts[0].trim().trim_matches('"').to_string();
        let value = parts[1].trim().trim_start_matches('{').trim_end_matches('}');

        let mut plugin = None;
        let mut method = None;

        for part in value.split(',') {
            let part = part.trim();
            if part.starts_with("plugin") {
                if let Some(v) = part.split('=').nth(1) {
                    plugin = Some(v.trim().trim_matches('"').to_string());
                }
            } else if part.starts_with("method") {
                if let Some(v) = part.split('=').nth(1) {
                    method = Some(v.trim().trim_matches('"').to_string());
                }
            }
        }

        if let (Some(plugin), Some(method)) = (plugin, method) {
            Some((key, OverrideConfig { plugin, method }))
        } else {
            None
        }
    }
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty_config() {
        let config = PluginConfig::parse("").unwrap();
        assert!(config.native_plugins.is_empty());
        assert!(config.js_plugins.is_empty());
        assert!(config.overrides.is_empty());
    }

    #[test]
    fn test_parse_plugin_entry() {
        let entry = PluginConfig::parse_plugin_entry(
            r#"{ path = "./plugins/lib.so", enabled = true }"#
        );
        assert!(entry.is_some());
        let (path, enabled) = entry.unwrap();
        assert_eq!(path, "./plugins/lib.so");
        assert!(enabled);
    }
}
