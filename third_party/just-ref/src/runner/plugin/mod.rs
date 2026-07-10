//! Plugin architecture and super-global scope.
//!
//! This module implements the **super-global scope** - a key architectural feature
//! that provides lazy, dynamic resolution of built-in and plugin-provided objects.
//!
//! ## Super-Global Scope Concept
//!
//! The super-global scope sits outside the normal lexical environment chain and
//! provides objects that are available globally but resolved on-demand:
//!
//! ```text
//! Variable Lookup Order:
//! 1. Local scope (function/block)
//! 2. Outer scopes (lexical chain)
//! 3. Global scope
//! 4. Super-global scope ← Built-ins and plugins live here
//! ```
//!
//! ### Key Components
//!
//! - **[`PluginResolver`]**: Trait for providing objects and methods dynamically
//! - **[`SuperGlobalEnvironment`]**: Container holding multiple resolvers with caching
//! - **[`CorePluginResolver`]**: Adapter wrapping [`BuiltInRegistry`] as a resolver
//! - **[`EvalContext`](types::EvalContext)**: Execution context with super-global integration
//!
//! ### Resolution Flow
//!
//! When JavaScript code references a name (e.g., `Math`):
//!
//! 1. **Check cache**: Has this name been resolved before?
//! 2. **Query resolvers**: Ask each resolver in registration order
//! 3. **Cache result**: Store the resolved value for future lookups
//! 4. **Return value**: Provide the object to JavaScript code
//!
//! ### Method Calls
//!
//! Method calls (e.g., `Math.abs(-5)`) are handled specially:
//!
//! 1. **Direct dispatch**: Resolvers can handle method calls directly without
//!    materializing the object
//! 2. **Fallback**: If no resolver handles it, fall back to property lookup
//!
//! This allows efficient built-in method calls without creating intermediate objects.
//!
//! ## Example: Custom Plugin
//!
//! ```
//! use just::runner::plugin::resolver::PluginResolver;
//! use just::runner::plugin::types::EvalContext;
//! use just::runner::ds::value::{JsValue, JsNumberType};
//! use just::runner::ds::error::JErrorType;
//!
//! struct UtilsPlugin;
//!
//! impl PluginResolver for UtilsPlugin {
//!     fn has_binding(&self, name: &str) -> bool {
//!         name == "Utils"
//!     }
//!     
//!     fn resolve(&self, _name: &str, _ctx: &mut EvalContext) -> Result<JsValue, JErrorType> {
//!         Ok(JsValue::Undefined) // Sentinel
//!     }
//!     
//!     fn call_method(&self, obj: &str, method: &str, _ctx: &mut EvalContext,
//!                    _this: JsValue, args: Vec<JsValue>) -> Option<Result<JsValue, JErrorType>> {
//!         if obj == "Utils" && method == "double" {
//!             let n = match args.first() {
//!                 Some(JsValue::Number(JsNumberType::Integer(n))) => *n,
//!                 _ => 0,
//!             };
//!             Some(Ok(JsValue::Number(JsNumberType::Integer(n * 2))))
//!         } else {
//!             None
//!         }
//!     }
//!     
//!     fn name(&self) -> &str { "utils_plugin" }
//! }
//!
//! // Register the plugin
//! let mut ctx = EvalContext::new();
//! ctx.add_resolver(Box::new(UtilsPlugin));
//! // Now Utils.double(21) returns 42
//! ```
//!
//! ## Design Rationale
//!
//! ### Why Not Preload Everything?
//!
//! Traditional approach:
//! - Load all built-ins at startup
//! - High memory usage
//! - Slow startup time
//! - Hard to extend
//!
//! Super-global approach:
//! - Load on first use (lazy)
//! - Lower memory footprint
//! - Fast startup
//! - Easy to add plugins
//!
//! ### Why Not Use Global Scope?
//!
//! Keeping built-ins in a separate super-global scope:
//! - Prevents accidental mutation from JavaScript
//! - Allows local variables to shadow built-ins
//! - Maintains clean separation of concerns
//! - Enables efficient caching and dispatch

pub mod types;
pub mod registry;
pub mod config;
pub mod resolver;
pub mod core_resolver;
pub mod super_global;

pub use types::{BuiltInFn, BuiltInObject, NativeFn, PluginInfo};
pub use registry::BuiltInRegistry;
pub use config::PluginConfig;
pub use resolver::PluginResolver;
pub use core_resolver::CorePluginResolver;
pub use super_global::SuperGlobalEnvironment;
