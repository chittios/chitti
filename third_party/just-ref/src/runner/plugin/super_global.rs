//! Super-global environment — the bottom of the scope chain.
//!
//! This environment sits below the global scope and lazily resolves
//! built-in and plugin-provided objects on first access. Objects are
//! cached after first resolution so each name is materialized at most once.
//!
//! ## How It Works
//!
//! When JavaScript code references a name that isn't found in the lexical
//! environment chain or global scope, the super-global environment is consulted:
//!
//! ```text
//! JavaScript: Math.abs(-5)
//!      ↓
//! 1. Check local scope → not found
//! 2. Check outer scopes → not found  
//! 3. Check global scope → not found
//! 4. Check super-global → "Math" found!
//!      ↓
//! 5. Query resolvers: Does anyone provide "Math"?
//! 6. CorePluginResolver says "yes"
//! 7. Cache the result
//! 8. Dispatch Math.abs() via resolver
//! ```
//!
//! ## Caching Strategy
//!
//! - **First lookup**: Query all resolvers in order, cache which one owns the name
//! - **Subsequent lookups**: Use cached resolver index directly
//! - **Method calls**: Can bypass object materialization entirely
//!
//! ## Example
//!
//! ```
//! use just::runner::plugin::super_global::SuperGlobalEnvironment;
//! use just::runner::plugin::resolver::PluginResolver;
//! use just::runner::plugin::types::EvalContext;
//! use just::runner::ds::value::{JsValue, JsNumberType};
//! use just::runner::ds::error::JErrorType;
//!
//! struct MyPlugin;
//!
//! impl PluginResolver for MyPlugin {
//!     fn has_binding(&self, name: &str) -> bool {
//!         name == "MyObject"
//!     }
//!     
//!     fn resolve(&self, _name: &str, _ctx: &mut EvalContext) -> Result<JsValue, JErrorType> {
//!         Ok(JsValue::Number(JsNumberType::Integer(42)))
//!     }
//!     
//!     fn call_method(&self, _obj: &str, _method: &str, _ctx: &mut EvalContext,
//!                    _this: JsValue, _args: Vec<JsValue>) -> Option<Result<JsValue, JErrorType>> {
//!         None
//!     }
//!     
//!     fn name(&self) -> &str { "my_plugin" }
//! }
//!
//! let mut sg = SuperGlobalEnvironment::new();
//! sg.add_resolver(Box::new(MyPlugin));
//!
//! // Now "MyObject" is available in the super-global scope
//! ```

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use hashbrown::HashMap;

use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::JsValue;
use crate::runner::plugin::resolver::PluginResolver;
use crate::runner::plugin::types::EvalContext;

/// The super-global environment for lazy resolution of built-in objects.
///
/// This is the outermost scope in the resolution chain, sitting below even
/// the global scope. It provides built-in objects (Math, console, etc.) and
/// plugin-provided objects through a lazy resolution mechanism.
///
/// ## Key Features
///
/// - **Lazy Resolution**: Objects are only materialized when first accessed
/// - **Caching**: Once resolved, values are cached for fast subsequent access
/// - **Read-Only**: JavaScript code cannot create or modify super-global bindings
/// - **Plugin System**: Multiple resolvers can be registered, queried in order
/// - **Method Dispatch**: Efficient method calls without object materialization
///
/// ## Resolution Order
///
/// When a name is looked up:
/// 1. Check if it's in the cache → return cached value
/// 2. Query each resolver's `has_binding()` in registration order
/// 3. First resolver that claims the name wins
/// 4. Call resolver's `resolve()` to materialize the value
/// 5. Cache the value and resolver index
/// 6. Return the value
///
/// ## Method Call Optimization
///
/// For method calls like `Math.abs(-5)`, the super-global can dispatch directly
/// to the resolver's `call_method()` without materializing the Math object,
/// providing better performance for built-in method calls.
///
/// ## Thread Safety
///
/// This struct is typically wrapped in `Rc<RefCell<>>` (see [`SharedSuperGlobal`](super::types::SharedSuperGlobal))
/// to allow shared mutable access across the evaluation context.
pub struct SuperGlobalEnvironment {
    /// Registered plugin resolvers, queried in order.
    resolvers: Vec<Box<dyn PluginResolver>>,
    /// ChittiOS: caches use interior mutability so `resolve_binding` can take
    /// `&self` — otherwise a builtin method call (which holds an immutable
    /// borrow of this env) that runs JS resolving another identifier would
    /// need `borrow_mut()` on the same RefCell and panic (`already borrowed`).
    cache: core::cell::RefCell<HashMap<String, JsValue>>,
    /// Cache of which resolver index owns which name.
    resolver_map: core::cell::RefCell<HashMap<String, usize>>,
}

impl SuperGlobalEnvironment {
    pub fn new() -> Self {
        SuperGlobalEnvironment {
            resolvers: Vec::new(),
            cache: core::cell::RefCell::new(HashMap::new()),
            resolver_map: core::cell::RefCell::new(HashMap::new()),
        }
    }

    /// Register a plugin resolver. Resolvers are queried in registration order.
    pub fn add_resolver(&mut self, resolver: Box<dyn PluginResolver>) {
        self.resolvers.push(resolver);
    }

    /// Find which resolver (if any) provides the given name.
    fn find_resolver_index(&self, name: &str) -> Option<usize> {
        // Check the cache first
        if let Some(&idx) = self.resolver_map.borrow().get(name) {
            return Some(idx);
        }
        // Query resolvers in order
        for (i, resolver) in self.resolvers.iter().enumerate() {
            if resolver.has_binding(name) {
                return Some(i);
            }
        }
        None
    }

    /// Call a method on a super-global object, dispatching to the owning resolver.
    ///
    /// Returns `None` if no resolver owns `object_name` or the resolver
    /// doesn't provide the method (allowing fallback to property lookup).
    pub fn call_method(
        &self,
        object_name: &str,
        method_name: &str,
        ctx: &mut EvalContext,
        this: JsValue,
        args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>> {
        let idx = if let Some(&idx) = self.resolver_map.borrow().get(object_name) {
            idx
        } else {
            self.find_resolver_index(object_name)?
        };
        self.resolvers[idx].call_method(object_name, method_name, ctx, this, args)
    }

    /// Does any resolver provide `method_name` on `object_name`? Cheap check
    /// (no materialization) — used to decide whether to expose a builtin
    /// prototype method as a first-class value on a receiver.
    pub fn has_method(&self, object_name: &str, method_name: &str) -> bool {
        let idx = if let Some(&idx) = self.resolver_map.borrow().get(object_name) {
            idx
        } else {
            match self.find_resolver_index(object_name) {
                Some(i) => i,
                None => return false,
            }
        };
        self.resolvers[idx].has_method(object_name, method_name)
    }

    /// Call a constructor on a super-global object.
    pub fn call_constructor(
        &self,
        object_name: &str,
        ctx: &mut EvalContext,
        args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>> {
        let idx = if let Some(&idx) = self.resolver_map.borrow().get(object_name) {
            idx
        } else {
            self.find_resolver_index(object_name)?
        };
        self.resolvers[idx].call_constructor(object_name, ctx, args)
    }

    /// Check if any resolver provides the given name.
    pub fn has_name(&self, name: &str) -> bool {
        self.cache.borrow().contains_key(name) || self.find_resolver_index(name).is_some()
    }

    /// Resolve a name, caching the result.
    /// `ctx` is needed because some resolvers may need it during materialization.
    pub fn resolve_binding(
        &self,
        name: &str,
        ctx: &mut EvalContext,
    ) -> Result<JsValue, JErrorType> {
        // Return cached value if available (drop the borrow before resolving).
        if let Some(val) = self.cache.borrow().get(name) {
            return Ok(val.clone());
        }

        // Find the owning resolver
        if let Some(idx) = self.find_resolver_index(name) {
            // NB: `resolve` may re-enter this env (it takes `ctx`), so hold no
            // cache borrow across the call.
            let value = self.resolvers[idx].resolve(name, ctx)?;
            self.cache.borrow_mut().insert(name.to_string(), value.clone());
            self.resolver_map.borrow_mut().insert(name.to_string(), idx);
            Ok(value)
        } else {
            Err(JErrorType::ReferenceError(format!(
                "{} is not defined",
                name
            )))
        }
    }

    /// Get a reference to the resolvers (for inspection/testing).
    pub fn resolvers(&self) -> &[Box<dyn PluginResolver>] {
        &self.resolvers
    }
}
