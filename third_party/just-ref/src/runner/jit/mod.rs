//! JIT compilation module for the JavaScript engine.
//!
//! Provides a bytecode compiler and stack-based VM as an alternative
//! execution path to the tree-walking interpreter. The pipeline is:
//!
//! ```text
//! JavaScript source → Parser → AST → Compiler → Bytecode → VM → Result
//! ```
//!
//! The bytecode representation eliminates per-node dispatch overhead
//! and improves cache locality compared to recursive AST evaluation.

pub mod bytecode;
pub mod compiler;
pub mod reg_bytecode;
pub mod reg_compiler;
pub mod reg_jit;
pub mod reg_vm;
pub mod vm;

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::parser::ast::ProgramData;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::JsValue;
use crate::runner::plugin::registry::BuiltInRegistry;
use crate::runner::plugin::types::EvalContext;

use self::bytecode::Chunk;
use self::compiler::Compiler;
use self::reg_bytecode::RegChunk;
use self::reg_compiler::RegCompiler;
use self::reg_jit::RegJit;
use self::reg_vm::{RegVm, RegVmResult};
use self::vm::{Vm, VmResult};

/// Compile an AST program into stack-based bytecode.
///
/// Transforms the AST into a linear sequence of bytecode instructions
/// for execution by the stack-based VM.
///
/// # Examples
///
/// ```
/// use just::parser::JsParser;
/// use just::runner::jit;
///
/// let code = "var x = 5 + 3;";
/// let ast = JsParser::parse_to_ast_from_str(code).unwrap();
/// let chunk = jit::compile(&ast);
/// ```
pub fn compile(program: &ProgramData) -> Chunk {
    let compiler = Compiler::new();
    compiler.compile_program(program)
}

/// Execute register bytecode with Cranelift JIT compilation.
///
/// Uses the Cranelift code generator to compile numeric-heavy code paths
/// to native machine code for maximum performance.
///
/// # Examples
///
/// ```
/// use just::parser::JsParser;
/// use just::runner::plugin::types::EvalContext;
/// use just::runner::jit::reg_compiler::RegCompiler;
/// use just::runner::jit;
///
/// let code = "var sum = 0; for (var i = 0; i < 100; i++) { sum = sum + i; }";
/// let ast = JsParser::parse_to_ast_from_str(code).unwrap();
/// let compiler = RegCompiler::new();
/// let chunk = compiler.compile_program(&ast);
///
/// let ctx = EvalContext::new();
/// let (result, ctx) = jit::execute_reg_jit(&chunk, ctx).unwrap();
/// ```
pub fn execute_reg_jit(chunk: &RegChunk, mut ctx: EvalContext) -> Result<(JsValue, EvalContext), JErrorType> {
    let mut jit = RegJit::new()?;
    let (result, regs) = jit.execute(chunk)?;

    for local in &chunk.locals {
        let name = chunk.get_name(local.name_idx);
        let val = f64_to_jsvalue(regs[local.reg as usize]);
        if !ctx.has_var_binding(name) {
            let _ = ctx.create_var_binding(name);
            let _ = ctx.initialize_var_binding(name, val.clone());
        } else {
            let _ = ctx.set_var_binding(name, val.clone());
        }
    }

    Ok((f64_to_jsvalue(result), ctx))
}

/// Execute register bytecode with JIT when possible; fall back to RegVm on failure.
pub fn execute_reg_jit_or_vm(
    chunk: &RegChunk,
    mut ctx: EvalContext,
    registry: &BuiltInRegistry,
) -> Result<(JsValue, EvalContext), JErrorType> {
    if let Ok(mut jit) = RegJit::new() {
        if let Ok((result, regs)) = jit.execute(chunk) {
            for local in &chunk.locals {
                let name = chunk.get_name(local.name_idx);
                let val = f64_to_jsvalue(regs[local.reg as usize]);
                if !ctx.has_var_binding(name) {
                    let _ = ctx.create_var_binding(name);
                    let _ = ctx.initialize_var_binding(name, val.clone());
                } else {
                    let _ = ctx.set_var_binding(name, val.clone());
                }
            }
            return Ok((f64_to_jsvalue(result), ctx));
        }
    }

    let mut vm = RegVm::new(chunk, ctx, registry);
    let result = match vm.run() {
        RegVmResult::Ok(val) => Ok(val),
        RegVmResult::Error(e) => Err(e),
    };
    let ctx_out = vm.into_ctx();
    result.map(|val| (val, ctx_out))
}

/// Compile an AST program into register-based bytecode.
///
/// Register-based bytecode is more suitable for JIT compilation
/// compared to stack-based bytecode.
///
/// # Examples
///
/// ```
/// use just::parser::JsParser;
/// use just::runner::jit;
///
/// let code = "var x = 5 + 3;";
/// let ast = JsParser::parse_to_ast_from_str(code).unwrap();
/// let chunk = jit::compile_reg(&ast);
/// ```
pub fn compile_reg(program: &ProgramData) -> RegChunk {
    let compiler = RegCompiler::new();
    compiler.compile_program(program)
}

/// Execute stack-based bytecode.
///
/// Runs the bytecode through the stack-based VM.
///
/// # Examples
///
/// ```
/// use just::parser::JsParser;
/// use just::runner::plugin::types::EvalContext;
/// use just::runner::plugin::registry::BuiltInRegistry;
/// use just::runner::jit;
///
/// let code = "var x = Math.abs(-42);";
/// let ast = JsParser::parse_to_ast_from_str(code).unwrap();
/// let chunk = jit::compile(&ast);
///
/// let mut ctx = EvalContext::new();
/// ctx.install_core_builtins(BuiltInRegistry::with_core());
///
/// let result = jit::execute(&chunk, ctx).unwrap();
/// ```
pub fn execute(chunk: &Chunk, ctx: EvalContext) -> Result<JsValue, JErrorType> {
    let mut vm = Vm::new(chunk, ctx);
    match vm.run() {
        VmResult::Ok(val) => Ok(val),
        VmResult::Error(e) => Err(e),
    }
}

/// Execute a compiled register bytecode chunk.
pub fn execute_reg(
    chunk: &RegChunk,
    ctx: EvalContext,
    registry: &BuiltInRegistry,
) -> Result<JsValue, JErrorType> {
    let mut vm = RegVm::new(chunk, ctx, registry);
    match vm.run() {
        RegVmResult::Ok(val) => Ok(val),
        RegVmResult::Error(e) => Err(e),
    }
}

/// Compile and execute a program in one step.
///
/// Convenience function that combines [`compile`] and [`execute`].
///
/// # Examples
///
/// ```
/// use just::parser::JsParser;
/// use just::runner::plugin::types::EvalContext;
/// use just::runner::plugin::registry::BuiltInRegistry;
/// use just::runner::jit;
///
/// let code = "var sum = 0; for (var i = 0; i < 10; i++) { sum = sum + i; } sum";
/// let ast = JsParser::parse_to_ast_from_str(code).unwrap();
///
/// let mut ctx = EvalContext::new();
/// ctx.install_core_builtins(BuiltInRegistry::with_core());
///
/// let result = jit::compile_and_run(&ast, ctx).unwrap();
/// ```
pub fn compile_and_run(
    program: &ProgramData,
    ctx: EvalContext,
) -> Result<JsValue, JErrorType> {
    let chunk = compile(program);
    execute(&chunk, ctx)
}

/// Compile and execute a program with the register VM.
pub fn compile_and_run_reg(
    program: &ProgramData,
    ctx: EvalContext,
    registry: &BuiltInRegistry,
) -> Result<JsValue, JErrorType> {
    let chunk = compile_reg(program);
    execute_reg(&chunk, ctx, registry)
}

/// Compile and execute, returning the context for variable inspection.
///
/// Returns both the result and the evaluation context, allowing you to
/// inspect variables after execution.
///
/// # Examples
///
/// ```
/// use just::parser::JsParser;
/// use just::runner::plugin::registry::BuiltInRegistry;
/// use just::runner::jit;
///
/// let code = "var x = 42; var y = x * 2;";
/// let ast = JsParser::parse_to_ast_from_str(code).unwrap();
///
/// let (result, ctx) = jit::compile_and_run_with_ctx(&ast);
/// let y = ctx.get_binding("y").unwrap();
/// ```
pub fn compile_and_run_with_ctx(
    program: &ProgramData,
) -> (Result<JsValue, JErrorType>, EvalContext) {
    let chunk = compile(program);
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    let mut vm = Vm::new(&chunk, ctx);
    let result = match vm.run() {
        VmResult::Ok(val) => Ok(val),
        VmResult::Error(e) => Err(e),
    };
    let ctx_out = vm.into_ctx();
    (result, ctx_out)
}

/// Compile and execute with the numeric JIT prototype.
pub fn compile_and_run_reg_jit_with_ctx(
    program: &ProgramData,
) -> (Result<JsValue, JErrorType>, EvalContext) {
    let chunk = compile_reg(program);
    let ctx = EvalContext::new();
    match execute_reg_jit(&chunk, ctx) {
        Ok((value, ctx_out)) => (Ok(value), ctx_out),
        Err(err) => (Err(err), EvalContext::new()),
    }
}

fn f64_to_jsvalue(n: f64) -> JsValue {
    if n.is_nan() {
        JsValue::Number(crate::runner::ds::value::JsNumberType::NaN)
    } else if n == f64::INFINITY {
        JsValue::Number(crate::runner::ds::value::JsNumberType::PositiveInfinity)
    } else if n == f64::NEG_INFINITY {
        JsValue::Number(crate::runner::ds::value::JsNumberType::NegativeInfinity)
    } else if n.fract() == 0.0 && n >= i64::MIN as f64 && n <= i64::MAX as f64 {
        JsValue::Number(crate::runner::ds::value::JsNumberType::Integer(n as i64))
    } else {
        JsValue::Number(crate::runner::ds::value::JsNumberType::Float(n))
    }
}

/// Compile and execute with register VM, returning the EvalContext.
pub fn compile_and_run_reg_with_ctx(
    program: &ProgramData,
    registry: &BuiltInRegistry,
) -> (Result<JsValue, JErrorType>, EvalContext) {
    let chunk = compile_reg(program);
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    let mut vm = RegVm::new(&chunk, ctx, registry);
    let result = match vm.run() {
        RegVmResult::Ok(val) => Ok(val),
        RegVmResult::Error(e) => Err(e),
    };
    let ctx_out = vm.into_ctx();
    (result, ctx_out)
}
