//! Plugin resolver trait for lazy, dynamic resolution of super-global objects.
//!
//! Plugins implement `PluginResolver` to provide objects (like `Math`, `console`)
//! that are available in the super-global scope. Objects are resolved lazily —
//! only when JS code actually references them.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::JsValue;
use crate::runner::plugin::types::EvalContext;

/// A plugin resolver that can dynamically provide named objects and their methods.
///
/// Resolvers are queried in registration order when a name lookup reaches the
/// super-global scope. The first resolver that claims a name wins.
pub trait PluginResolver {
    /// Does this resolver provide a binding with the given name?
    ///
    /// This should be a cheap check (e.g. a `HashSet::contains`).
    /// It must NOT allocate or materialize the object.
    fn has_binding(&self, name: &str) -> bool;

    /// Materialize the object for the given name.
    ///
    /// Called only after `has_binding` returns `true`.
    /// The returned `JsValue` is cached in the super-global environment
    /// so this is called at most once per name per execution.
    fn resolve(&self, name: &str, ctx: &mut EvalContext) -> Result<JsValue, JErrorType>;

    /// Resolve a method on an object this resolver provides.
    ///
    /// For example, if this resolver provides `"Math"`, then
    /// `resolve_method("Math", "abs", ctx, this, args)` should execute `Math.abs`.
    ///
    /// Returns `None` if the method is not found, allowing fallback to
    /// property lookup on the materialized object.
    fn call_method(
        &self,
        object_name: &str,
        method_name: &str,
        ctx: &mut EvalContext,
        this: JsValue,
        args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>>;

    /// Does this resolver provide `method_name` on `object_name`? A cheap check
    /// (no allocation / materialization) used to decide whether a builtin
    /// prototype method should be exposed as a first-class value on a receiver
    /// (`[].slice`, `Object.prototype.toString`) — see
    /// `expression::materialize_builtin_method`. Defaults to `false`.
    fn has_method(&self, _object_name: &str, _method_name: &str) -> bool {
        false
    }

    /// Get a constructor for the given object name, if available.
    ///
    /// Returns `None` if this resolver doesn't provide a constructor for the name.
    fn call_constructor(
        &self,
        _object_name: &str,
        _ctx: &mut EvalContext,
        _args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>> {
        None
    }

    /// Human-readable name for this resolver (for debugging/logging).
    fn name(&self) -> &str;
}
