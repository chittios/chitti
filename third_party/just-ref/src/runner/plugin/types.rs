//! Core types for the plugin architecture.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::env_record::{
    new_declarative_environment, new_global_environment, EnvironmentRecord, EnvironmentRecordType,
};
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::execution_context::ExecutionContextStack;
use crate::runner::ds::heap::{Heap, HeapConfig};
use crate::runner::ds::lex_env::JsLexEnvironmentType;
use crate::runner::ds::object::{JsObject, JsObjectType, ObjectBase, ObjectType};
use crate::runner::ds::value::JsValue;
use crate::parser::ast::FunctionData;
use super::super_global::SuperGlobalEnvironment;
use core::cell::RefCell;
use hashbrown::HashMap;
use alloc::rc::Rc;

/// Shared heap type for use across contexts.
pub type SharedHeap = Rc<RefCell<Heap>>;

/// Shared super-global type for use across contexts.
pub type SharedSuperGlobal = Rc<RefCell<SuperGlobalEnvironment>>;

/// Execution context for JavaScript evaluation.
///
/// Contains the lexical environment chain, heap, and super-global scope.
/// This is the main context object passed through the interpreter and VMs.
///
/// # Examples
///
/// ```
/// use just::parser::JsParser;
/// use just::runner::plugin::types::EvalContext;
/// use just::runner::plugin::registry::BuiltInRegistry;
/// use just::runner::eval::statement::execute_statement;
///
/// // Create context with built-ins
/// let mut ctx = EvalContext::new();
/// ctx.install_core_builtins(BuiltInRegistry::with_core());
///
/// // Parse and execute code
/// let code = "var x = Math.abs(-42);";
/// let ast = JsParser::parse_to_ast_from_str(code).unwrap();
/// for stmt in &ast.body {
///     execute_statement(stmt, &mut ctx).unwrap();
/// }
///
/// // Access variables
/// let x = ctx.get_binding("x").unwrap();
/// ```
pub struct EvalContext {
    /// The global `this` value.
    pub global_this: Option<JsValue>,
    /// Shared heap for memory allocation tracking.
    pub heap: SharedHeap,
    /// Current lexical environment (for let/const and block scoping).
    pub lex_env: JsLexEnvironmentType,
    /// Current variable environment (for var declarations).
    pub var_env: JsLexEnvironmentType,
    /// Execution context stack for tracking function calls.
    pub ctx_stack: ExecutionContextStack,
    /// Whether we're in strict mode.
    pub strict: bool,
    /// Lexical environment version for cache invalidation.
    pub lex_env_version: u64,
    /// Super-global scope for lazy resolution of built-in and plugin objects.
    pub super_global: SharedSuperGlobal,
    /// ChittiOS: the superclass constructor of the class whose constructor is
    /// currently running, so `super(...)` can invoke it. `None` outside a
    /// derived-class constructor.
    pub current_super: Option<JsValue>,
    /// ChittiOS: generic hook for objects whose properties are backed by native
    /// host state (a live DOM element view). An object carrying an
    /// `__native_node__` integer routes its property get/set through this.
    pub native_props: Option<Rc<dyn NativeProps>>,
}

/// ChittiOS: property backing for host-native objects (e.g. a live DOM view).
/// An object with an `__native_node__` integer has its property reads/writes
/// dispatched here instead of to the JS heap. Generic — not DOM-specific.
pub trait NativeProps {
    /// Read `node.prop`; `None` means "not handled — fall back to the JS object".
    fn get(&self, node: i64, prop: &str) -> Option<JsValue>;
    /// Write `node.prop = value`; return `true` if handled.
    fn set(&self, node: i64, prop: &str, value: JsValue) -> bool;
}

impl EvalContext {
    /// Create a new evaluation context with default heap configuration.
    ///
    /// Creates a fresh context with:
    /// - Empty super-global scope (call `install_core_builtins` to add built-ins)
    /// - Default heap with no memory limits
    /// - Global lexical environment
    ///
    /// # Examples
    ///
    /// ```
    /// use just::runner::plugin::types::EvalContext;
    ///
    /// let mut ctx = EvalContext::new();
    /// // Context is ready to use, but has no built-ins yet
    /// ```
    pub fn new() -> Self {
        // Create a simple global object
        let global_obj: JsObjectType = Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(
            SimpleObject::new(),
        ))));
        let global_env = new_global_environment(global_obj.clone());

        EvalContext {
            global_this: Some(JsValue::Object(global_obj)),
            heap: Rc::new(RefCell::new(Heap::default())),
            lex_env: global_env.clone(),
            var_env: global_env,
            ctx_stack: ExecutionContextStack::new(),
            strict: false,
            lex_env_version: 0,
            super_global: Rc::new(RefCell::new(SuperGlobalEnvironment::new())),
            current_super: None,
            native_props: None,
        }
    }

    /// Create a new evaluation context with a specific heap configuration.
    ///
    /// Use this to set memory limits or custom heap behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// use just::runner::plugin::types::EvalContext;
    /// use just::runner::ds::heap::HeapConfig;
    ///
    /// let config = HeapConfig {
    ///     max_bytes: Some(10 * 1024 * 1024), // 10MB limit
    ///     gc_threshold: 1024 * 1024,
    /// };
    /// let mut ctx = EvalContext::with_heap_config(config);
    /// ```
    pub fn with_heap_config(config: HeapConfig) -> Self {
        let global_obj: JsObjectType = Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(
            SimpleObject::new(),
        ))));
        let global_env = new_global_environment(global_obj.clone());

        EvalContext {
            global_this: Some(JsValue::Object(global_obj)),
            heap: Rc::new(RefCell::new(Heap::new(config))),
            lex_env: global_env.clone(),
            var_env: global_env,
            ctx_stack: ExecutionContextStack::new(),
            strict: false,
            lex_env_version: 0,
            super_global: Rc::new(RefCell::new(SuperGlobalEnvironment::new())),
            current_super: None,
            native_props: None,
        }
    }

    /// Current lexical environment version.
    pub fn current_lex_env_version(&self) -> u64 {
        self.lex_env_version
    }

    /// Install a plugin resolver into the super-global scope.
    ///
    /// Resolvers are queried in registration order. The first resolver
    /// that claims a name wins.
    ///
    /// # Examples
    ///
    /// ```
    /// use just::runner::plugin::types::EvalContext;
    /// use just::runner::plugin::resolver::PluginResolver;
    /// use just::runner::ds::value::JsValue;
    /// use just::runner::ds::error::JErrorType;
    ///
    /// struct MyPlugin;
    ///
    /// impl PluginResolver for MyPlugin {
    ///     fn has_binding(&self, name: &str) -> bool {
    ///         name == "MyObject"
    ///     }
    ///     
    ///     fn resolve(&self, _name: &str, _ctx: &mut EvalContext) -> Result<JsValue, JErrorType> {
    ///         Ok(JsValue::Undefined)
    ///     }
    ///     
    ///     fn call_method(&self, _obj: &str, _method: &str, _ctx: &mut EvalContext,
    ///                    _this: JsValue, _args: Vec<JsValue>) -> Option<Result<JsValue, JErrorType>> {
    ///         None
    ///     }
    ///     
    ///     fn name(&self) -> &str { "my_plugin" }
    /// }
    ///
    /// let mut ctx = EvalContext::new();
    /// ctx.add_resolver(Box::new(MyPlugin));
    /// ```
    pub fn add_resolver(&mut self, resolver: Box<dyn super::resolver::PluginResolver>) {
        self.super_global.borrow_mut().add_resolver(resolver);
    }

    /// Install the core built-in objects (Math, console, JSON, etc.).
    ///
    /// This is a convenience method that wraps the `BuiltInRegistry`
    /// in a `CorePluginResolver` and adds it to the super-global scope.
    ///
    /// # Examples
    ///
    /// ```
    /// use just::runner::plugin::types::EvalContext;
    /// use just::runner::plugin::registry::BuiltInRegistry;
    ///
    /// let mut ctx = EvalContext::new();
    /// ctx.install_core_builtins(BuiltInRegistry::with_core());
    ///
    /// // Now Math, console, etc. are available
    /// ```
    pub fn install_core_builtins(&mut self, registry: super::registry::BuiltInRegistry) {
        self.add_resolver(Box::new(super::core_resolver::CorePluginResolver::new(registry)));
        // ChittiOS: global values (undefined/NaN/Infinity/globalThis) + callable
        // globals (isNaN/isFinite/parseInt/parseFloat/Boolean/Symbol).
        self.add_resolver(Box::new(crate::runner::std_lib::globals::GlobalsResolver));
    }

    /// Track a heap allocation.
    pub fn allocate(&self, bytes: usize) -> Result<(), JErrorType> {
        self.heap.borrow_mut().allocate(bytes)
    }

    /// Track a heap deallocation.
    pub fn deallocate(&self, bytes: usize) {
        self.heap.borrow_mut().deallocate(bytes)
    }

    /// Get the current heap usage in bytes.
    pub fn heap_usage(&self) -> usize {
        self.heap.borrow().get_allocated()
    }

    /// Create a tracked SimpleObject that participates in heap accounting.
    pub fn new_tracked_object(&self) -> Result<SimpleObject, JErrorType> {
        SimpleObject::new_tracked(self.heap.clone())
    }

    /// Look up a binding in the environment chain.
    pub fn get_binding(&mut self, name: &str) -> Result<JsValue, JErrorType> {
        let name_string = name.to_string();
        self.resolve_binding(&name_string)
    }

    /// Look up a binding and return the environment where it was found.
    pub fn get_binding_with_env(
        &mut self,
        name: &str,
    ) -> Result<(JsValue, JsLexEnvironmentType), JErrorType> {
        let name_string = name.to_string();
        let mut current_env = Some(self.lex_env.clone());
        let mut last_env: Option<JsLexEnvironmentType> = None;

        while let Some(env) = current_env {
            let env_borrowed = env.borrow();
            if env_borrowed.inner.as_env_record().has_binding(&name_string) {
                drop(env_borrowed);
                let value = env
                    .borrow()
                    .inner
                    .as_env_record()
                    .get_binding_value(&mut self.ctx_stack, &name_string)?;
                return Ok((value, env));
            }
            last_env = Some(env.clone());
            current_env = env_borrowed.outer.clone();
        }

        // Fall back to super-global scope.
        // Return the global env as the "owning" env for cache purposes.
        let sg = self.super_global.clone();
        let result = sg.borrow().resolve_binding(&name_string, self);
        let value = result?;
        let env = last_env.unwrap_or_else(|| self.lex_env.clone());
        Ok((value, env))
    }

    /// Get a binding from a specific environment (used by inline caches).
    pub fn get_binding_in_env(
        &mut self,
        env: &JsLexEnvironmentType,
        name: &str,
    ) -> Result<JsValue, JErrorType> {
        let name_string = name.to_string();
        env.borrow()
            .inner
            .as_env_record()
            .get_binding_value(&mut self.ctx_stack, &name_string)
    }

    /// Resolve a binding by walking up the environment chain,
    /// falling back to the super-global scope if not found.
    pub fn resolve_binding(&mut self, name: &String) -> Result<JsValue, JErrorType> {
        let mut current_env = Some(self.lex_env.clone());

        while let Some(env) = current_env {
            let env_borrowed = env.borrow();
            if env_borrowed.inner.as_env_record().has_binding(name) {
                drop(env_borrowed);
                return env
                    .borrow()
                    .inner
                    .as_env_record()
                    .get_binding_value(&mut self.ctx_stack, name);
            }
            current_env = env_borrowed.outer.clone();
        }

        // Fall back to super-global scope
        let sg = self.super_global.clone();
        let result = sg.borrow().resolve_binding(name, self);
        result
    }

    /// Set a binding in the environment chain.
    pub fn set_binding(&mut self, name: &str, value: JsValue) -> Result<(), JErrorType> {
        let name_string = name.to_string();
        self.resolve_and_set_binding(&name_string, value)
    }

    /// Set a binding and return the environment where it was set.
    pub fn set_binding_with_env(
        &mut self,
        name: &str,
        value: JsValue,
    ) -> Result<JsLexEnvironmentType, JErrorType> {
        let name_string = name.to_string();
        let mut current_env = Some(self.lex_env.clone());

        while let Some(env) = current_env.clone() {
            let has_binding = env.borrow().inner.as_env_record().has_binding(&name_string);
            if has_binding {
                self.set_binding_in_env(&env, &name_string, value)?;
                return Ok(env);
            }
            current_env = env.borrow().outer.clone();
        }

        if !self.strict {
            let env = self.var_env.clone();
            let exists = env.borrow().inner.as_env_record().has_binding(&name_string);
            if exists {
                self.set_binding_in_env(&env, &name_string, value)?;
            } else {
                let mut b = env.borrow_mut();
                let rec = b.inner.as_env_record_mut();
                rec.create_mutable_binding(name_string.clone(), true)?;
                rec.initialize_binding(&mut self.ctx_stack, name_string.clone(), value)?;
            }
            return Ok(env);
        }

        Err(JErrorType::ReferenceError(format!("{} is not defined", name)))
    }

    /// Set a binding in a specific environment (inline cache fast-path).
    pub fn set_binding_in_env_cached(
        &mut self,
        env: &JsLexEnvironmentType,
        name: &str,
        value: JsValue,
    ) -> Result<(), JErrorType> {
        let name_string = name.to_string();
        self.set_binding_in_env(env, &name_string, value)
    }

    /// Resolve and set a binding by walking up the environment chain.
    fn resolve_and_set_binding(&mut self, name: &String, value: JsValue) -> Result<(), JErrorType> {
        let mut current_env = Some(self.lex_env.clone());

        while let Some(env) = current_env.clone() {
            let has_binding = env.borrow().inner.as_env_record().has_binding(name);
            if has_binding {
                return self.set_binding_in_env(&env, name, value);
            }
            current_env = env.borrow().outer.clone();
        }

        // If not found and not strict, create an implicit global (sloppy-mode
        // `y = 1` at global scope creates a global property). The binding does
        // not yet exist, so create + initialize it rather than
        // `set_mutable_binding` (which requires an existing binding).
        if !self.strict {
            let env = self.var_env.clone();
            {
                let mut b = env.borrow_mut();
                let rec = b.inner.as_env_record_mut();
                rec.create_mutable_binding(name.clone(), true)?;
                rec.initialize_binding(&mut self.ctx_stack, name.clone(), value)?;
            }
            return Ok(());
        }

        Err(JErrorType::ReferenceError(format!("{} is not defined", name)))
    }

    /// Set a binding in a specific environment.
    fn set_binding_in_env(
        &mut self,
        env: &JsLexEnvironmentType,
        name: &String,
        value: JsValue,
    ) -> Result<(), JErrorType> {
        let mut env_borrowed = env.borrow_mut();
        match env_borrowed.inner.as_mut() {
            EnvironmentRecordType::Declarative(rec) => {
                rec.set_mutable_binding(&mut self.ctx_stack, name.clone(), value)
            }
            EnvironmentRecordType::Function(rec) => {
                rec.set_mutable_binding(&mut self.ctx_stack, name.clone(), value)
            }
            EnvironmentRecordType::Global(rec) => {
                rec.set_mutable_binding(&mut self.ctx_stack, name.clone(), value)
            }
            EnvironmentRecordType::Object(rec) => {
                rec.set_mutable_binding(&mut self.ctx_stack, name.clone(), value)
            }
        }
    }

    /// Create a new binding in the current lexical environment.
    pub fn create_binding(&mut self, name: &str, is_const: bool) -> Result<(), JErrorType> {
        let mut env = self.lex_env.borrow_mut();
        let name_string = name.to_string();
        match env.inner.as_mut() {
            EnvironmentRecordType::Declarative(rec) => {
                if is_const {
                    rec.create_immutable_binding(name_string)?
                } else {
                    rec.create_mutable_binding(name_string, false)?
                }
            }
            EnvironmentRecordType::Function(rec) => {
                if is_const {
                    rec.create_immutable_binding(name_string)?
                } else {
                    rec.create_mutable_binding(name_string, false)?
                }
            }
            EnvironmentRecordType::Global(rec) => {
                if is_const {
                    rec.create_immutable_binding(name_string)?
                } else {
                    rec.create_mutable_binding(name_string, false)?
                }
            }
            EnvironmentRecordType::Object(rec) => {
                rec.create_mutable_binding(name_string, false)?
            }
        }
        self.lex_env_version = self.lex_env_version.wrapping_add(1);
        Ok(())
    }

    /// Create a var binding in the variable environment.
    pub fn create_var_binding(&mut self, name: &str) -> Result<(), JErrorType> {
        let mut env = self.var_env.borrow_mut();
        let name_string = name.to_string();
        match env.inner.as_mut() {
            EnvironmentRecordType::Declarative(rec) => {
                rec.create_mutable_binding(name_string, true)?
            }
            EnvironmentRecordType::Function(rec) => rec.create_mutable_binding(name_string, true)?,
            EnvironmentRecordType::Global(rec) => rec.create_mutable_binding(name_string, true)?,
            EnvironmentRecordType::Object(rec) => rec.create_mutable_binding(name_string, true)?,
        }
        self.lex_env_version = self.lex_env_version.wrapping_add(1);
        Ok(())
    }

    /// Initialize a binding with a value.
    pub fn initialize_binding(&mut self, name: &str, value: JsValue) -> Result<(), JErrorType> {
        let name_string = name.to_string();
        let mut env = self.lex_env.borrow_mut();
        match env.inner.as_mut() {
            EnvironmentRecordType::Declarative(rec) => {
                rec.initialize_binding(&mut self.ctx_stack, name_string, value)?;
                Ok(())
            }
            EnvironmentRecordType::Function(rec) => {
                rec.initialize_binding(&mut self.ctx_stack, name_string, value)?;
                Ok(())
            }
            EnvironmentRecordType::Global(rec) => {
                rec.initialize_binding(&mut self.ctx_stack, name_string, value)?;
                Ok(())
            }
            EnvironmentRecordType::Object(rec) => {
                rec.initialize_binding(&mut self.ctx_stack, name_string, value)?;
                Ok(())
            }
        }
    }

    /// Initialize a var binding with a value.
    pub fn initialize_var_binding(&mut self, name: &str, value: JsValue) -> Result<(), JErrorType> {
        let name_string = name.to_string();
        let mut env = self.var_env.borrow_mut();
        match env.inner.as_mut() {
            EnvironmentRecordType::Declarative(rec) => {
                rec.initialize_binding(&mut self.ctx_stack, name_string, value)?;
                Ok(())
            }
            EnvironmentRecordType::Function(rec) => {
                rec.initialize_binding(&mut self.ctx_stack, name_string, value)?;
                Ok(())
            }
            EnvironmentRecordType::Global(rec) => {
                rec.initialize_binding(&mut self.ctx_stack, name_string, value)?;
                Ok(())
            }
            EnvironmentRecordType::Object(rec) => {
                rec.initialize_binding(&mut self.ctx_stack, name_string, value)?;
                Ok(())
            }
        }
    }

    /// Check if a var binding exists.
    pub fn has_var_binding(&self, name: &str) -> bool {
        let env = self.var_env.borrow();
        env.inner.as_env_record().has_binding(&name.to_string())
    }

    /// Set a var binding's value (for re-declaration).
    pub fn set_var_binding(&mut self, name: &str, value: JsValue) -> Result<(), JErrorType> {
        let name_string = name.to_string();
        let mut env = self.var_env.borrow_mut();
        match env.inner.as_mut() {
            EnvironmentRecordType::Declarative(rec) => {
                rec.set_mutable_binding(&mut self.ctx_stack, name_string, value)
            }
            EnvironmentRecordType::Function(rec) => {
                rec.set_mutable_binding(&mut self.ctx_stack, name_string, value)
            }
            EnvironmentRecordType::Global(rec) => {
                rec.set_mutable_binding(&mut self.ctx_stack, name_string, value)
            }
            EnvironmentRecordType::Object(rec) => {
                rec.set_mutable_binding(&mut self.ctx_stack, name_string, value)
            }
        }
    }

    /// Push a new block scope (for let/const).
    pub fn push_block_scope(&mut self) {
        let new_env = new_declarative_environment(Some(self.lex_env.clone()));
        self.lex_env = new_env;
        self.lex_env_version = self.lex_env_version.wrapping_add(1);
    }

    /// Pop a block scope.
    pub fn pop_block_scope(&mut self) {
        let outer = self.lex_env.borrow().outer.clone();
        if let Some(outer_env) = outer {
            self.lex_env = outer_env;
        }
        self.lex_env_version = self.lex_env_version.wrapping_add(1);
    }

    /// Check if a binding exists in the current environment chain
    /// or in the super-global scope.
    pub fn has_binding(&self, name: &str) -> bool {
        let name_string = name.to_string();
        let mut current_env = Some(self.lex_env.clone());

        while let Some(env) = current_env {
            let env_borrowed = env.borrow();
            if env_borrowed.inner.as_env_record().has_binding(&name_string) {
                return true;
            }
            current_env = env_borrowed.outer.clone();
        }

        // Check super-global scope
        self.super_global.borrow().has_name(name)
    }
}

impl Default for EvalContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple object for use as global object.
pub struct SimpleObject {
    base: ObjectBase,
    heap: Option<SharedHeap>,
    allocated_bytes: usize,
}

impl SimpleObject {
    pub fn new() -> Self {
        SimpleObject {
            base: ObjectBase::new(),
            heap: None,
            allocated_bytes: 0,
        }
    }

    pub fn new_tracked(heap: SharedHeap) -> Result<Self, JErrorType> {
        const SIMPLE_OBJECT_ALLOCATION_BYTES: usize = 64;
        heap.borrow_mut().allocate(SIMPLE_OBJECT_ALLOCATION_BYTES)?;
        Ok(SimpleObject {
            base: ObjectBase::new(),
            heap: Some(heap),
            allocated_bytes: SIMPLE_OBJECT_ALLOCATION_BYTES,
        })
    }
}

impl Drop for SimpleObject {
    fn drop(&mut self) {
        if let Some(heap) = &self.heap {
            heap.borrow_mut().deallocate(self.allocated_bytes);
        }
    }
}

impl JsObject for SimpleObject {
    fn get_object_base_mut(&mut self) -> &mut ObjectBase {
        &mut self.base
    }

    fn get_object_base(&self) -> &ObjectBase {
        &self.base
    }

    fn as_js_object(&self) -> &dyn JsObject {
        self
    }

    fn as_js_object_mut(&mut self) -> &mut dyn JsObject {
        self
    }
}

/// Function signature for built-in methods.
/// Native functions receive the evaluation context, `this` value, and arguments.
pub type NativeFn = fn(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType>;

/// Built-in function - either compiled-in or plugin-provided.
pub enum BuiltInFn {
    /// Direct function pointer - zero overhead for compiled-in functions.
    Native(NativeFn),

    /// Plugin-provided function - small vtable indirection cost.
    Plugin(Box<dyn Fn(&mut EvalContext, JsValue, Vec<JsValue>) -> Result<JsValue, JErrorType> + Send + Sync>),

    /// JavaScript implementation - interpreted at runtime.
    Script(Rc<FunctionData>),
}

impl BuiltInFn {
    /// Execute this built-in function.
    pub fn call(
        &self,
        ctx: &mut EvalContext,
        this: JsValue,
        args: Vec<JsValue>,
    ) -> Result<JsValue, JErrorType> {
        match self {
            BuiltInFn::Native(f) => f(ctx, this, args),
            BuiltInFn::Plugin(f) => f(ctx, this, args),
            BuiltInFn::Script(_f) => {
                // TODO: Implement script execution when runtime is ready
                Err(JErrorType::TypeError("Script built-ins not yet implemented".to_string()))
            }
        }
    }
}

/// Built-in object definition.
/// Represents a JavaScript built-in object like Array, Object, String, etc.
pub struct BuiltInObject {
    /// Name of the object (e.g., "Array", "Object", "Math").
    pub name: String,

    /// Parent prototype name, if any (e.g., "Object" for most built-ins).
    pub prototype: Option<String>,

    /// Methods defined on this object or its prototype.
    pub methods: HashMap<String, BuiltInFn>,

    /// Static properties.
    pub properties: HashMap<String, JsValue>,

    /// Constructor function, if this object is constructable.
    pub constructor: Option<BuiltInFn>,
}

impl BuiltInObject {
    /// Create a new built-in object with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        BuiltInObject {
            name: name.into(),
            prototype: Some("Object".to_string()),
            methods: HashMap::new(),
            properties: HashMap::new(),
            constructor: None,
        }
    }

    /// Set the prototype chain parent.
    pub fn with_prototype(mut self, prototype: impl Into<String>) -> Self {
        self.prototype = Some(prototype.into());
        self
    }

    /// Set no prototype (for objects like Object.prototype itself).
    pub fn with_no_prototype(mut self) -> Self {
        self.prototype = None;
        self
    }

    /// Add a native method.
    pub fn add_method(mut self, name: impl Into<String>, func: NativeFn) -> Self {
        self.methods.insert(name.into(), BuiltInFn::Native(func));
        self
    }

    /// Add a property.
    pub fn add_property(mut self, name: impl Into<String>, value: JsValue) -> Self {
        self.properties.insert(name.into(), value);
        self
    }

    /// Set the constructor function.
    pub fn with_constructor(mut self, constructor: NativeFn) -> Self {
        self.constructor = Some(BuiltInFn::Native(constructor));
        self
    }
}

/// Plugin metadata.
/// Contains information about a loaded plugin.
#[derive(Debug, Clone)]
pub struct PluginInfo {
    /// Plugin name.
    pub name: String,

    /// Plugin version.
    pub version: String,

    /// List of object names this plugin provides.
    pub provides: Vec<String>,
}

impl PluginInfo {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        PluginInfo {
            name: name.into(),
            version: version.into(),
            provides: Vec::new(),
        }
    }

    pub fn with_provides(mut self, provides: Vec<String>) -> Self {
        self.provides = provides;
        self
    }
}

