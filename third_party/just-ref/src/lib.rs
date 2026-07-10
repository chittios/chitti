//! # just-engine - JavaScript Engine in Rust
//!
//! A ground-up implementation of an ES6 JavaScript engine featuring:
//! - PEG parser with ESTree-compliant AST
//! - Tree-walking interpreter
//! - Stack-based and register-based bytecode VMs
//! - Cranelift-powered native JIT compiler
//! - Plugin architecture with lazy super-global scope resolution
//!
//! ## Quick Start
//!
//! ### Parsing JavaScript
//!
//! ```
//! use just::parser::JsParser;
//!
//! let code = "var x = 5 + 3;";
//! let ast = JsParser::parse_to_ast_from_str(code).unwrap();
//! println!("Parsed {} statements", ast.body.len());
//! ```
//!
//! ### Running JavaScript with the Interpreter
//!
//! ```
//! use just_engine::parser::JsParser;
//! use just_engine::runner::plugin::types::EvalContext;
//! use just_engine::runner::plugin::registry::BuiltInRegistry;
//! use just_engine::runner::eval::statement::execute_statement;
//!
//! // Parse the code
//! let code = "var x = Math.abs(-42);";
//! let ast = JsParser::parse_to_ast_from_str(code).unwrap();
//!
//! // Create evaluation context with built-ins
//! let mut ctx = EvalContext::new();
//! ctx.install_core_builtins(BuiltInRegistry::with_core());
//!
//! // Execute statements
//! for stmt in &ast.body {
//!     execute_statement(stmt, &mut ctx).unwrap();
//! }
//!
//! // Get the result
//! let x = ctx.get_binding("x").unwrap();
//! println!("x = {:?}", x);
//! ```
//!
//! ### Using the JIT Compiler
//!
//! ```
//! use just_engine::parser::JsParser;
//! use just_engine::runner::plugin::types::EvalContext;
//! use just_engine::runner::plugin::registry::BuiltInRegistry;
//! use just_engine::runner::jit;
//!
//! let code = "var sum = 0; for (var i = 0; i < 100; i++) { sum = sum + i; } sum";
//! let ast = JsParser::parse_to_ast_from_str(code).unwrap();
//!
//! let mut ctx = EvalContext::new();
//! ctx.install_core_builtins(BuiltInRegistry::with_core());
//!
//! let result = jit::execute(&ast, &mut ctx).unwrap();
//! println!("Result: {:?}", result);
//! ```
//!
//! ## Super-Global Scope Architecture
//!
//! One of the key architectural features of this engine is the **super-global scope**,
//! which provides lazy, dynamic resolution of built-in and plugin-provided objects.
//!
//! ### How It Works
//!
//! Traditional JavaScript engines preload all built-in objects (Math, console, Array, etc.)
//! into the global scope at startup. This engine uses a different approach:
//!
//! 1. **Lazy Resolution**: Built-in objects are resolved only when JavaScript code
//!    actually references them, reducing startup time and memory usage.
//!
//! 2. **Plugin Architecture**: Objects are provided by "plugin resolvers" that implement
//!    the [`runner::plugin::resolver::PluginResolver`] trait. Multiple resolvers can be
//!    registered, and they're queried in order.
//!
//! 3. **Caching**: Once resolved, objects are cached in the super-global environment
//!    to avoid repeated lookups.
//!
//! 4. **Immutable from JS**: JavaScript code cannot mutate the super-global scope itself,
//!    though it can shadow super-global names with local variables.
//!
//! ### Example: Custom Plugin
//!
//! ```
//! use just_engine::parser::JsParser;
//! use just_engine::runner::plugin::types::EvalContext;
//! use just_engine::runner::plugin::resolver::PluginResolver;
//! use just_engine::runner::plugin::registry::BuiltInRegistry;
//! use just_engine::runner::ds::value::{JsValue, JsNumberType};
//! use just_engine::runner::ds::error::JErrorType;
//! use just_engine::runner::eval::statement::execute_statement;
//!
//! // Define a custom plugin that provides a "MyMath" object
//! struct MyMathPlugin;
//!
//! impl PluginResolver for MyMathPlugin {
//!     fn has_binding(&self, name: &str) -> bool {
//!         name == "MyMath"
//!     }
//!     
//!     fn resolve(&self, _name: &str, _ctx: &mut EvalContext) -> Result<JsValue, JErrorType> {
//!         Ok(JsValue::Undefined) // Sentinel value
//!     }
//!     
//!     fn call_method(&self, obj: &str, method: &str, _ctx: &mut EvalContext,
//!                    _this: JsValue, args: Vec<JsValue>) -> Option<Result<JsValue, JErrorType>> {
//!         if obj == "MyMath" && method == "triple" {
//!             let n = match args.first() {
//!                 Some(JsValue::Number(JsNumberType::Integer(n))) => *n,
//!                 _ => 0,
//!             };
//!             Some(Ok(JsValue::Number(JsNumberType::Integer(n * 3))))
//!         } else {
//!             None
//!         }
//!     }
//!     
//!     fn name(&self) -> &str { "my_math_plugin" }
//! }
//!
//! // Use the plugin
//! let mut ctx = EvalContext::new();
//! ctx.install_core_builtins(BuiltInRegistry::with_core()); // Core built-ins
//! ctx.add_resolver(Box::new(MyMathPlugin)); // Custom plugin
//!
//! let code = "var result = MyMath.triple(7);";
//! let ast = JsParser::parse_to_ast_from_str(code).unwrap();
//! for stmt in &ast.body {
//!     execute_statement(stmt, &mut ctx).unwrap();
//! }
//!
//! let result = ctx.get_binding("result").unwrap();
//! // result is 21
//! ```
//!
//! ### Benefits
//!
//! - **Extensibility**: Easy to add custom objects without modifying the engine core
//! - **Performance**: Only pay for what you use - unused built-ins are never materialized
//! - **Modularity**: Built-ins and plugins are cleanly separated
//! - **Testing**: Easy to mock or replace built-ins for testing
//!
//! ## Architecture
//!
//! - **[`parser`]** - PEG parser and AST types
//! - **[`runner`]** - Execution engines (interpreter, VMs, JIT)
//!   - **[`runner::plugin`]** - Plugin system and super-global scope
//!   - **[`runner::ds`]** - Data structures (values, objects, environments)
//!   - **[`runner::eval`]** - Tree-walking interpreter
//!   - **[`runner::jit`]** - Bytecode VMs and JIT compiler

// ChittiOS: no_std + alloc when built without the `std` feature (kernel).
#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
extern crate alloc;

#[macro_use]
extern crate lazy_static;

pub mod parser;
pub mod runner;
