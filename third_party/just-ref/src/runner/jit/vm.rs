//! Stack-based bytecode virtual machine.
//!
//! Executes the bytecode emitted by the compiler. Uses a flat instruction
//! dispatch loop instead of recursive AST walking, which eliminates
//! per-node function call overhead and improves cache locality.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::{JsNumberType, JsValue};
use crate::runner::plugin::types::EvalContext;
use crate::runner::ds::lex_env::JsLexEnvironmentType;
use crate::runner::ds::object::{JsObject, ObjectType};
use crate::runner::ds::object_property::{PropertyDescriptor, PropertyDescriptorData, PropertyKey};
use crate::runner::ds::value::JsValueOrSelf;
use crate::runner::ds::operations::type_conversion::to_string;
use crate::runner::eval::expression::{SimpleFunctionObject, call_function_object};

use core::cell::RefCell;
use alloc::rc::Rc;

use super::bytecode::{Chunk, OpCode};

/// Result of VM execution.
pub enum VmResult {
    /// Normal completion with a value.
    Ok(JsValue),
    /// Runtime error.
    Error(JErrorType),
}

#[derive(Clone)]
struct EnvCacheEntry {
    version: u64,
    env: JsLexEnvironmentType,
}

#[derive(Clone)]
struct PropCacheEntry {
    obj: crate::runner::ds::object::JsObjectType,
    prop: String,
}

/// The bytecode virtual machine.
pub struct Vm<'a> {
    chunk: &'a Chunk,
    /// Instruction pointer.
    ip: usize,
    /// Operand stack.
    stack: Vec<JsValue>,
    /// Local slots for fast var access.
    locals: Vec<JsValue>,
    /// Inline caches for GetVar.
    get_var_cache: Vec<Option<EnvCacheEntry>>,
    /// Inline caches for SetVar.
    set_var_cache: Vec<Option<EnvCacheEntry>>,
    /// Inline caches for GetProp.
    get_prop_cache: Vec<Option<PropCacheEntry>>,
    /// Evaluation context (variables, scoping, heap).
    ctx: EvalContext,
}

impl<'a> Vm<'a> {
    pub fn new(chunk: &'a Chunk, ctx: EvalContext) -> Self {
        Vm {
            chunk,
            ip: 0,
            stack: Vec::with_capacity(256),
            locals: vec![JsValue::Undefined; chunk.locals.len()],
            get_var_cache: vec![None; chunk.code.len()],
            set_var_cache: vec![None; chunk.code.len()],
            get_prop_cache: vec![None; chunk.code.len()],
            ctx,
        }
    }

    /// Consume the VM and return the evaluation context (for variable inspection after execution).
    pub fn into_ctx(mut self) -> EvalContext {
        self.sync_locals_into_ctx();
        self.ctx
    }

    /// Run the bytecode to completion.
    pub fn run(&mut self) -> VmResult {
        loop {
            if self.ip >= self.chunk.code.len() {
                return VmResult::Ok(self.stack.pop().unwrap_or(JsValue::Undefined));
            }

            let instr_index = self.ip;
            let instr = &self.chunk.code[instr_index];
            let op = instr.op;
            let operand = instr.operand;
            let operand2 = instr.operand2;
            let operand3 = instr.operand3;
            self.ip += 1;

            match op {
                // ── Constants & Literals ──────────────────────
                OpCode::Constant => {
                    let val = self.chunk.constants[operand as usize].clone();
                    self.stack.push(val);
                }
                OpCode::Undefined => self.stack.push(JsValue::Undefined),
                OpCode::Null => self.stack.push(JsValue::Null),
                OpCode::True => self.stack.push(JsValue::Boolean(true)),
                OpCode::False => self.stack.push(JsValue::Boolean(false)),

                // ── Arithmetic ───────────────────────────────
                OpCode::Add => {
                    let b = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let a = self.stack.pop().unwrap_or(JsValue::Undefined);
                    self.stack.push(self.js_add(a, b));
                }
                OpCode::Sub => {
                    let b = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let a = self.stack.pop().unwrap_or(JsValue::Undefined);
                    self.stack.push(self.js_sub(a, b));
                }
                OpCode::Mul => {
                    let b = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let a = self.stack.pop().unwrap_or(JsValue::Undefined);
                    self.stack.push(self.js_mul(a, b));
                }
                OpCode::Div => {
                    let b = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let a = self.stack.pop().unwrap_or(JsValue::Undefined);
                    self.stack.push(self.js_div(a, b));
                }
                OpCode::Mod => {
                    let b = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let a = self.stack.pop().unwrap_or(JsValue::Undefined);
                    self.stack.push(self.js_mod(a, b));
                }
                OpCode::Negate => {
                    let a = self.stack.pop().unwrap_or(JsValue::Undefined);
                    self.stack.push(self.js_negate(a));
                }

                // ── Bitwise ──────────────────────────────────
                OpCode::BitAnd => {
                    let bv = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let av = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let b = self.to_i32(bv);
                    let a = self.to_i32(av);
                    self.stack.push(JsValue::Number(JsNumberType::Integer((a & b) as i64)));
                }
                OpCode::BitOr => {
                    let bv = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let av = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let b = self.to_i32(bv);
                    let a = self.to_i32(av);
                    self.stack.push(JsValue::Number(JsNumberType::Integer((a | b) as i64)));
                }
                OpCode::BitXor => {
                    let bv = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let av = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let b = self.to_i32(bv);
                    let a = self.to_i32(av);
                    self.stack.push(JsValue::Number(JsNumberType::Integer((a ^ b) as i64)));
                }
                OpCode::BitNot => {
                    let av = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let a = self.to_i32(av);
                    self.stack.push(JsValue::Number(JsNumberType::Integer((!a) as i64)));
                }
                OpCode::ShiftLeft => {
                    let bv = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let av = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let b = self.to_u32(bv);
                    let a = self.to_i32(av);
                    self.stack.push(JsValue::Number(JsNumberType::Integer(
                        (a << (b & 0x1f)) as i64,
                    )));
                }
                OpCode::ShiftRight => {
                    let bv = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let av = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let b = self.to_u32(bv);
                    let a = self.to_i32(av);
                    self.stack.push(JsValue::Number(JsNumberType::Integer(
                        (a >> (b & 0x1f)) as i64,
                    )));
                }
                OpCode::UShiftRight => {
                    let bv = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let av = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let b = self.to_u32(bv);
                    let a = self.to_u32(av);
                    self.stack.push(JsValue::Number(JsNumberType::Integer(
                        (a >> (b & 0x1f)) as i64,
                    )));
                }

                // ── Comparison ───────────────────────────────
                OpCode::StrictEqual => {
                    let b = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let a = self.stack.pop().unwrap_or(JsValue::Undefined);
                    self.stack.push(JsValue::Boolean(self.js_strict_equal(&a, &b)));
                }
                OpCode::StrictNotEqual => {
                    let b = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let a = self.stack.pop().unwrap_or(JsValue::Undefined);
                    self.stack.push(JsValue::Boolean(!self.js_strict_equal(&a, &b)));
                }
                OpCode::Equal => {
                    let b = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let a = self.stack.pop().unwrap_or(JsValue::Undefined);
                    self.stack.push(JsValue::Boolean(self.js_abstract_equal(&a, &b)));
                }
                OpCode::NotEqual => {
                    let b = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let a = self.stack.pop().unwrap_or(JsValue::Undefined);
                    self.stack.push(JsValue::Boolean(!self.js_abstract_equal(&a, &b)));
                }
                OpCode::LessThan => {
                    let b = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let a = self.stack.pop().unwrap_or(JsValue::Undefined);
                    self.stack.push(JsValue::Boolean(self.js_less_than(&a, &b)));
                }
                OpCode::LessEqual => {
                    let b = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let a = self.stack.pop().unwrap_or(JsValue::Undefined);
                    // a <= b  is  !(b < a)
                    self.stack.push(JsValue::Boolean(!self.js_less_than(&b, &a)));
                }
                OpCode::GreaterThan => {
                    let b = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let a = self.stack.pop().unwrap_or(JsValue::Undefined);
                    self.stack.push(JsValue::Boolean(self.js_less_than(&b, &a)));
                }
                OpCode::GreaterEqual => {
                    let b = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let a = self.stack.pop().unwrap_or(JsValue::Undefined);
                    self.stack.push(JsValue::Boolean(!self.js_less_than(&a, &b)));
                }

                // ── Logical / Unary ──────────────────────────
                OpCode::Not => {
                    let a = self.stack.pop().unwrap_or(JsValue::Undefined);
                    self.stack.push(JsValue::Boolean(!self.is_truthy(&a)));
                }
                OpCode::TypeOf => {
                    let a = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let type_str = match &a {
                        JsValue::Undefined => "undefined",
                        JsValue::Null => "object",
                        JsValue::Boolean(_) => "boolean",
                        JsValue::Number(_) => "number",
                        JsValue::String(_) => "string",
                        JsValue::BigInt(_) => "bigint",
                        JsValue::Symbol(_) => "symbol",
                        JsValue::Object(_) => "object",
                    };
                    self.stack.push(JsValue::String(type_str.to_string()));
                }
                OpCode::Void => {
                    self.stack.pop();
                    self.stack.push(JsValue::Undefined);
                }
                OpCode::UnaryPlus => {
                    let a = self.stack.pop().unwrap_or(JsValue::Undefined);
                    self.stack.push(self.to_number_value(a));
                }

                // ── Variables ────────────────────────────────
                OpCode::GetVar => {
                    let name = self.chunk.get_name(operand);
                    match self.get_var_cached(instr_index, name) {
                        Ok(val) => self.stack.push(val),
                        Err(e) => return VmResult::Error(e),
                    }
                }
                OpCode::SetVar => {
                    let name = self.chunk.get_name(operand);
                    let val = self.stack.pop().unwrap_or(JsValue::Undefined);
                    if let Err(e) = self.set_var_cached(instr_index, name, val) {
                        return VmResult::Error(e);
                    }
                }
                OpCode::DeclareVar => {
                    let name = self.chunk.get_name(operand);
                    if !self.ctx.has_var_binding(name) {
                        if let Err(e) = self.ctx.create_var_binding(name) {
                            return VmResult::Error(e);
                        }
                    }
                }
                OpCode::DeclareLet => {
                    let name = self.chunk.get_name(operand);
                    if let Err(e) = self.ctx.create_binding(name, false) {
                        return VmResult::Error(e);
                    }
                }
                OpCode::DeclareConst => {
                    let name = self.chunk.get_name(operand);
                    if let Err(e) = self.ctx.create_binding(name, true) {
                        return VmResult::Error(e);
                    }
                }
                OpCode::InitVar => {
                    let name = self.chunk.get_name(operand);
                    let val = self.stack.pop().unwrap_or(JsValue::Undefined);
                    // First try initialize (works when binding is uninitialized).
                    // If that silently no-ops (binding already initialized),
                    // fall back to set (works when binding is already initialized).
                    if let Err(e) = self.ctx.initialize_var_binding(name, val.clone()) {
                        return VmResult::Error(e);
                    }
                    // Also set, in case initialize was a no-op (already initialized).
                    // set_var_binding will error on uninitialized, so ignore that error.
                    let _ = self.ctx.set_var_binding(name, val);
                }
                OpCode::InitBinding => {
                    let name = self.chunk.get_name(operand);
                    let val = self.stack.pop().unwrap_or(JsValue::Undefined);
                    if let Err(e) = self.ctx.initialize_binding(name, val) {
                        return VmResult::Error(e);
                    }
                }

                OpCode::GetLocal => {
                    let slot = operand as usize;
                    let val = self.locals.get(slot).cloned().unwrap_or(JsValue::Undefined);
                    self.stack.push(val);
                }
                OpCode::SetLocal | OpCode::InitLocal => {
                    let slot = operand as usize;
                    let val = self.stack.pop().unwrap_or(JsValue::Undefined);
                    if let Some(local) = self.locals.get_mut(slot) {
                        *local = val;
                    }
                }

                // ── Control Flow ─────────────────────────────
                OpCode::Jump => {
                    self.ip = operand as usize;
                }
                OpCode::JumpIfFalse => {
                    let val = self.stack.pop().unwrap_or(JsValue::Undefined);
                    if !self.is_truthy(&val) {
                        self.ip = operand as usize;
                    }
                }
                OpCode::JumpIfTrue => {
                    let val = self.stack.pop().unwrap_or(JsValue::Undefined);
                    if self.is_truthy(&val) {
                        self.ip = operand as usize;
                    }
                }

                OpCode::LoopStart => {
                    // No-op marker
                }

                // ── Stack manipulation ───────────────────────
                OpCode::Pop => {
                    self.stack.pop();
                }
                OpCode::Dup => {
                    if let Some(top) = self.stack.last() {
                        self.stack.push(top.clone());
                    }
                }
                OpCode::Dup2 => {
                    if self.stack.len() >= 2 {
                        let len = self.stack.len();
                        let first = self.stack[len - 2].clone();
                        let second = self.stack[len - 1].clone();
                        self.stack.push(first);
                        self.stack.push(second);
                    }
                }

                // ── Scope ────────────────────────────────────
                OpCode::PushScope => {
                    self.ctx.push_block_scope();
                }
                OpCode::PopScope => {
                    self.ctx.pop_block_scope();
                }

                // ── Objects & Properties ─────────────────────
                OpCode::GetProp => {
                    let obj_val = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let prop_name = self.chunk.get_name(operand);
                    match self.get_prop_cached(instr_index, &obj_val, prop_name) {
                        Ok(val) => self.stack.push(val),
                        Err(e) => return VmResult::Error(e),
                    }
                }
                OpCode::SetProp => {
                    let val = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let obj_val = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let prop_name = self.chunk.get_name(operand);
                    if let Err(e) = self.set_prop_value(&obj_val, prop_name, val.clone()) {
                        return VmResult::Error(e);
                    }
                    self.stack.push(val);
                }
                OpCode::GetElem => {
                    let key_val = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let obj_val = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let prop_name = match self.to_property_key(&key_val) {
                        Ok(key) => key,
                        Err(e) => return VmResult::Error(e),
                    };
                    match self.get_prop_cached(instr_index, &obj_val, &prop_name) {
                        Ok(val) => self.stack.push(val),
                        Err(e) => return VmResult::Error(e),
                    }
                }
                OpCode::SetElem => {
                    let val = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let key_val = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let obj_val = self.stack.pop().unwrap_or(JsValue::Undefined);
                    let prop_name = match self.to_property_key(&key_val) {
                        Ok(key) => key,
                        Err(e) => return VmResult::Error(e),
                    };
                    if let Err(e) = self.set_prop_value(&obj_val, &prop_name, val.clone()) {
                        return VmResult::Error(e);
                    }
                    self.stack.push(val);
                }

                // ── Function calls ───────────────────────────
                OpCode::Call => {
                    let argc = operand as usize;
                    let mut args = Vec::with_capacity(argc);
                    for _ in 0..argc {
                        args.push(self.stack.pop().unwrap_or(JsValue::Undefined));
                    }
                    args.reverse();
                    let callee = self.stack.pop().unwrap_or(JsValue::Undefined);
                    match &callee {
                        JsValue::Object(obj) => {
                            let this_val = JsValue::Undefined;
                            match call_function_object(obj, this_val, args, &mut self.ctx) {
                                Ok(result) => self.stack.push(result),
                                Err(e) => return VmResult::Error(e),
                            }
                        }
                        _ => return VmResult::Error(JErrorType::TypeError(
                            format!("{} is not a function", callee),
                        )),
                    }
                }
                OpCode::CallMethod => {
                    let argc = operand as usize;
                    let method_name = self.chunk.get_name(operand2);
                    let obj_var_name = self.chunk.get_name(operand3);
                    let mut args = Vec::with_capacity(argc);
                    for _ in 0..argc {
                        args.push(self.stack.pop().unwrap_or(JsValue::Undefined));
                    }
                    args.reverse();
                    let object = self.stack.pop().unwrap_or(JsValue::Undefined);

                    // 1. Try super-global dispatch using compile-time object name
                    //    (e.g. "Math", "console")
                    if !obj_var_name.is_empty() {
                        let sg = self.ctx.super_global.clone();
                        let result = sg.borrow().call_method(
                            obj_var_name, method_name,
                            &mut self.ctx, object.clone(), args.clone(),
                        );
                        if let Some(res) = result {
                            match res {
                                Ok(val) => { self.stack.push(val); continue; }
                                Err(e) => return VmResult::Error(e),
                            }
                        }
                    }

                    // 2. Type-based super-global dispatch (e.g. String, Number)
                    let type_name = match &object {
                        JsValue::String(_) => Some("String"),
                        JsValue::Number(_) => Some("Number"),
                        _ => None,
                    };
                    if let Some(type_name) = type_name {
                        let sg = self.ctx.super_global.clone();
                        let result = sg.borrow().call_method(
                            type_name, method_name,
                            &mut self.ctx, object.clone(), args.clone(),
                        );
                        if let Some(res) = result {
                            match res {
                                Ok(val) => { self.stack.push(val); continue; }
                                Err(e) => return VmResult::Error(e),
                            }
                        }
                    }

                    // 3. Fallback: look up the method as a property on the object
                    if let JsValue::Object(ref obj) = object {
                        let prop_key = PropertyKey::Str(method_name.to_string());
                        let method_val = {
                            let obj_ref = obj.borrow();
                            obj_ref
                                .as_js_object()
                                .get_own_property(&prop_key)
                                .ok()
                                .flatten()
                                .and_then(|desc| match desc {
                                    PropertyDescriptor::Data(data) => Some(data.value.clone()),
                                    _ => None,
                                })
                        };
                        let method_val = method_val.or_else(|| {
                            let obj_ref = obj.borrow();
                            let proto = obj_ref.as_js_object().get_prototype_of();
                            if let Some(proto) = proto {
                                let proto_ref = proto.borrow();
                                proto_ref
                                    .as_js_object()
                                    .get_own_property(&prop_key)
                                    .ok()
                                    .flatten()
                                    .and_then(|desc| match desc {
                                        PropertyDescriptor::Data(data) => Some(data.value.clone()),
                                        _ => None,
                                    })
                            } else {
                                None
                            }
                        });

                        if let Some(JsValue::Object(method_obj)) = method_val {
                            match call_function_object(&method_obj, object.clone(), args, &mut self.ctx) {
                                Ok(result) => {
                                    self.stack.push(result);
                                    continue;
                                }
                                Err(e) => return VmResult::Error(e),
                            }
                        }
                    }

                    // Final fallback: undefined
                    self.stack.push(JsValue::Undefined);
                }

                // ── Misc ─────────────────────────────────────
                OpCode::Return => {
                    let val = self.stack.pop().unwrap_or(JsValue::Undefined);
                    return VmResult::Ok(val);
                }
                OpCode::Halt => {
                    return VmResult::Ok(self.stack.pop().unwrap_or(JsValue::Undefined));
                }

                // ── Pre/Post increment/decrement ─────────────
                OpCode::PreIncVar => {
                    let name = self.chunk.get_name(operand);
                    match self.get_var_cached(instr_index, name) {
                        Ok(val) => {
                            let num = self.to_f64(&val) + 1.0;
                            let new_val = self.f64_to_jsvalue(num);
                            if let Err(e) = self.set_var_cached(instr_index, name, new_val.clone()) {
                                return VmResult::Error(e);
                            }
                            self.stack.push(new_val);
                        }
                        Err(e) => return VmResult::Error(e),
                    }
                }
                OpCode::PreDecVar => {
                    let name = self.chunk.get_name(operand);
                    match self.get_var_cached(instr_index, name) {
                        Ok(val) => {
                            let num = self.to_f64(&val) - 1.0;
                            let new_val = self.f64_to_jsvalue(num);
                            if let Err(e) = self.set_var_cached(instr_index, name, new_val.clone()) {
                                return VmResult::Error(e);
                            }
                            self.stack.push(new_val);
                        }
                        Err(e) => return VmResult::Error(e),
                    }
                }
                OpCode::PostIncVar => {
                    let name = self.chunk.get_name(operand);
                    match self.get_var_cached(instr_index, name) {
                        Ok(val) => {
                            let old_val = val.clone();
                            let num = self.to_f64(&val) + 1.0;
                            let new_val = self.f64_to_jsvalue(num);
                            if let Err(e) = self.set_var_cached(instr_index, name, new_val) {
                                return VmResult::Error(e);
                            }
                            self.stack.push(old_val);
                        }
                        Err(e) => return VmResult::Error(e),
                    }
                }
                OpCode::PostDecVar => {
                    let name = self.chunk.get_name(operand);
                    match self.get_var_cached(instr_index, name) {
                        Ok(val) => {
                            let old_val = val.clone();
                            let num = self.to_f64(&val) - 1.0;
                            let new_val = self.f64_to_jsvalue(num);
                            if let Err(e) = self.set_var_cached(instr_index, name, new_val) {
                                return VmResult::Error(e);
                            }
                            self.stack.push(old_val);
                        }
                        Err(e) => return VmResult::Error(e),
                    }
                }

                OpCode::GetVarForUpdate => {
                    let name = self.chunk.get_name(operand);
                    match self.get_var_cached(instr_index, name) {
                        Ok(val) => self.stack.push(val),
                        Err(e) => return VmResult::Error(e),
                    }
                }

                // ── Closures ─────────────────────────────────
                OpCode::MakeClosure => {
                    let func_template = &self.chunk.functions[operand as usize];
                    let body_ptr = func_template.body_ptr;
                    let params_ptr = func_template.params_ptr;
                    let environment = self.ctx.lex_env.clone();

                    let mut func_obj = SimpleFunctionObject::new(body_ptr, params_ptr, environment);

                    // Add marker property to identify as callable
                    func_obj.get_object_base_mut().properties.insert(
                        PropertyKey::Str("__simple_function__".to_string()),
                        PropertyDescriptor::Data(PropertyDescriptorData {
                            value: JsValue::Boolean(true),
                            writable: false,
                            enumerable: false,
                            configurable: false,
                        }),
                    );

                    let obj_ref = Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(func_obj))));
                    self.stack.push(JsValue::Object(obj_ref));
                }
            }
        }
    }

    // Helper methods
    // ════════════════════════════════════════════════════════════

    fn get_var_cached(&mut self, instr_index: usize, name: &str) -> Result<JsValue, JErrorType> {
        if let Some(Some(entry)) = self.get_var_cache.get(instr_index) {
            if entry.version == self.ctx.current_lex_env_version() {
                if let Ok(val) = self.ctx.get_binding_in_env(&entry.env, name) {
                    return Ok(val);
                }
            }
        }

        let (val, env) = self.ctx.get_binding_with_env(name)?;
        let entry = EnvCacheEntry {
            version: self.ctx.current_lex_env_version(),
            env,
        };
        if let Some(slot) = self.get_var_cache.get_mut(instr_index) {
            *slot = Some(entry);
        }
        Ok(val)
    }

    fn set_var_cached(
        &mut self,
        instr_index: usize,
        name: &str,
        value: JsValue,
    ) -> Result<(), JErrorType> {
        if let Some(Some(entry)) = self.set_var_cache.get(instr_index) {
            if entry.version == self.ctx.current_lex_env_version() {
                if self
                    .ctx
                    .set_binding_in_env_cached(&entry.env, name, value.clone())
                    .is_ok()
                {
                    return Ok(());
                }
            }
        }

        let env = self.ctx.set_binding_with_env(name, value)?;
        let entry = EnvCacheEntry {
            version: self.ctx.current_lex_env_version(),
            env,
        };
        if let Some(slot) = self.set_var_cache.get_mut(instr_index) {
            *slot = Some(entry);
        }
        Ok(())
    }

    fn get_prop_cached(
        &mut self,
        instr_index: usize,
        obj_val: &JsValue,
        prop_name: &str,
    ) -> Result<JsValue, JErrorType> {
        if let JsValue::Object(obj) = obj_val {
            if let Some(Some(entry)) = self.get_prop_cache.get(instr_index) {
                if alloc::rc::Rc::ptr_eq(obj, &entry.obj) && entry.prop == prop_name {
                    let prop_key = PropertyKey::Str(prop_name.to_string());
                    return obj
                        .borrow()
                        .as_js_object()
                        .get(&mut self.ctx.ctx_stack, &prop_key, JsValueOrSelf::SelfValue);
                }
            }

            let prop_key = PropertyKey::Str(prop_name.to_string());
            let val = obj
                .borrow()
                .as_js_object()
                .get(&mut self.ctx.ctx_stack, &prop_key, JsValueOrSelf::SelfValue)?;
            let entry = PropCacheEntry {
                obj: obj.clone(),
                prop: prop_name.to_string(),
            };
            if let Some(slot) = self.get_prop_cache.get_mut(instr_index) {
                *slot = Some(entry);
            }
            Ok(val)
        } else if matches!(obj_val, JsValue::Undefined | JsValue::Null) {
            Err(JErrorType::TypeError("Cannot read property of null/undefined".to_string()))
        } else {
            Ok(JsValue::Undefined)
        }
    }

    fn set_prop_value(
        &mut self,
        obj_val: &JsValue,
        prop_name: &str,
        value: JsValue,
    ) -> Result<(), JErrorType> {
        match obj_val {
            JsValue::Object(obj) => {
                let prop_key = PropertyKey::Str(prop_name.to_string());
                let mut obj_ref = obj.borrow_mut();
                let ok = obj_ref
                    .as_js_object_mut()
                    .set(&mut self.ctx.ctx_stack, prop_key, value, JsValueOrSelf::SelfValue)?;
                if ok {
                    Ok(())
                } else {
                    Err(JErrorType::TypeError("Failed to set property".to_string()))
                }
            }
            _ => Err(JErrorType::TypeError("Cannot set property on non-object".to_string())),
        }
    }

    fn to_property_key(&mut self, value: &JsValue) -> Result<String, JErrorType> {
        to_string(&mut self.ctx.ctx_stack, value)
    }

    fn sync_locals_into_ctx(&mut self) {
        for (slot, val) in self.locals.iter().enumerate() {
            let name = self.chunk.get_local_name(slot as u32);
            if !self.ctx.has_var_binding(name) {
                let _ = self.ctx.create_var_binding(name);
                let _ = self.ctx.initialize_var_binding(name, val.clone());
            } else {
                let _ = self.ctx.set_var_binding(name, val.clone());
            }
        }
    }

    // ── Type coercion helpers ────────────────────────────────

    fn is_truthy(&self, val: &JsValue) -> bool {
        match val {
            JsValue::Undefined | JsValue::Null => false,
            JsValue::Boolean(b) => *b,
            JsValue::Number(n) => match n {
                JsNumberType::Integer(i) => *i != 0,
                JsNumberType::Float(f) => *f != 0.0 && !f.is_nan(),
                JsNumberType::NaN => false,
                JsNumberType::PositiveInfinity | JsNumberType::NegativeInfinity => true,
            },
            JsValue::String(s) => !s.is_empty(),
            _ => true,
        }
    }

    fn to_f64(&self, val: &JsValue) -> f64 {
        match val {
            JsValue::Number(JsNumberType::Integer(i)) => *i as f64,
            JsValue::Number(JsNumberType::Float(f)) => *f,
            JsValue::Number(JsNumberType::NaN) => f64::NAN,
            JsValue::Number(JsNumberType::PositiveInfinity) => f64::INFINITY,
            JsValue::Number(JsNumberType::NegativeInfinity) => f64::NEG_INFINITY,
            JsValue::Boolean(true) => 1.0,
            JsValue::Boolean(false) => 0.0,
            JsValue::Null => 0.0,
            JsValue::Undefined => f64::NAN,
            JsValue::String(s) => s.parse::<f64>().unwrap_or(f64::NAN),
            _ => f64::NAN,
        }
    }

    fn to_i32(&self, val: JsValue) -> i32 {
        let n = self.to_f64(&val);
        if n.is_nan() || n.is_infinite() || n == 0.0 {
            0
        } else {
            (n as i64 & 0xFFFFFFFF) as i32
        }
    }

    fn to_u32(&self, val: JsValue) -> u32 {
        let n = self.to_f64(&val);
        if n.is_nan() || n.is_infinite() || n == 0.0 {
            0
        } else {
            (n as i64 & 0xFFFFFFFF) as u32
        }
    }

    fn f64_to_jsvalue(&self, n: f64) -> JsValue {
        if n.is_nan() {
            JsValue::Number(JsNumberType::NaN)
        } else if n.is_infinite() {
            if n > 0.0 {
                JsValue::Number(JsNumberType::PositiveInfinity)
            } else {
                JsValue::Number(JsNumberType::NegativeInfinity)
            }
        } else if n.fract() == 0.0 && n >= i64::MIN as f64 && n <= i64::MAX as f64 {
            JsValue::Number(JsNumberType::Integer(n as i64))
        } else {
            JsValue::Number(JsNumberType::Float(n))
        }
    }

    fn to_number_value(&self, val: JsValue) -> JsValue {
        match val {
            JsValue::Number(_) => val,
            _ => self.f64_to_jsvalue(self.to_f64(&val)),
        }
    }

    // ── Arithmetic helpers ───────────────────────────────────

    fn js_add(&self, a: JsValue, b: JsValue) -> JsValue {
        // String concatenation takes priority
        match (&a, &b) {
            (JsValue::String(sa), JsValue::String(sb)) => {
                JsValue::String(format!("{}{}", sa, sb))
            }
            (JsValue::String(sa), _) => {
                JsValue::String(format!("{}{}", sa, self.to_display_string(&b)))
            }
            (_, JsValue::String(sb)) => {
                JsValue::String(format!("{}{}", self.to_display_string(&a), sb))
            }
            _ => {
                if let Some(val) = self.fast_number_binop(&a, &b, |x, y| x + y) {
                    return val;
                }
                self.f64_to_jsvalue(self.to_f64(&a) + self.to_f64(&b))
            }
        }
    }

    fn js_sub(&self, a: JsValue, b: JsValue) -> JsValue {
        if let Some(val) = self.fast_number_binop(&a, &b, |x, y| x - y) {
            return val;
        }
        self.f64_to_jsvalue(self.to_f64(&a) - self.to_f64(&b))
    }

    fn js_mul(&self, a: JsValue, b: JsValue) -> JsValue {
        if let Some(val) = self.fast_number_binop(&a, &b, |x, y| x * y) {
            return val;
        }
        self.f64_to_jsvalue(self.to_f64(&a) * self.to_f64(&b))
    }

    fn js_div(&self, a: JsValue, b: JsValue) -> JsValue {
        if let Some(nb) = self.as_number(&b) {
            let nb = self.num_to_f64(nb);
            if nb == 0.0 {
                let na = self.to_f64(&a);
                if na == 0.0 || na.is_nan() {
                    return JsValue::Number(JsNumberType::NaN);
                } else if na > 0.0 {
                    return JsValue::Number(JsNumberType::PositiveInfinity);
                } else {
                    return JsValue::Number(JsNumberType::NegativeInfinity);
                }
            }
            if let Some(val) = self.fast_number_binop(&a, &b, |x, y| x / y) {
                return val;
            }
        }

        let nb = self.to_f64(&b);
        if nb == 0.0 {
            let na = self.to_f64(&a);
            if na == 0.0 || na.is_nan() {
                JsValue::Number(JsNumberType::NaN)
            } else if na > 0.0 {
                JsValue::Number(JsNumberType::PositiveInfinity)
            } else {
                JsValue::Number(JsNumberType::NegativeInfinity)
            }
        } else {
            self.f64_to_jsvalue(self.to_f64(&a) / nb)
        }
    }

    fn js_mod(&self, a: JsValue, b: JsValue) -> JsValue {
        if let Some(val) = self.fast_number_binop(&a, &b, |x, y| x % y) {
            return val;
        }
        let na = self.to_f64(&a);
        let nb = self.to_f64(&b);
        if nb == 0.0 {
            JsValue::Number(JsNumberType::NaN)
        } else {
            self.f64_to_jsvalue(na % nb)
        }
    }

    fn js_negate(&self, a: JsValue) -> JsValue {
        if let JsValue::Number(n) = &a {
            return self.f64_to_jsvalue(-self.num_to_f64(n));
        }
        self.f64_to_jsvalue(-self.to_f64(&a))
    }

    fn as_number<'b>(&self, v: &'b JsValue) -> Option<&'b JsNumberType> {
        match v {
            JsValue::Number(n) => Some(n),
            _ => None,
        }
    }

    fn fast_number_binop(
        &self,
        a: &JsValue,
        b: &JsValue,
        op: fn(f64, f64) -> f64,
    ) -> Option<JsValue> {
        match (self.as_number(a), self.as_number(b)) {
            (Some(na), Some(nb)) => {
                let fa = self.num_to_f64(na);
                let fb = self.num_to_f64(nb);
                Some(self.f64_to_jsvalue(op(fa, fb)))
            }
            _ => None,
        }
    }

    // ── Comparison helpers ───────────────────────────────────

    fn js_strict_equal(&self, a: &JsValue, b: &JsValue) -> bool {
        match (a, b) {
            (JsValue::Undefined, JsValue::Undefined) => true,
            (JsValue::Null, JsValue::Null) => true,
            (JsValue::Boolean(a), JsValue::Boolean(b)) => a == b,
            (JsValue::Number(a), JsValue::Number(b)) => {
                let fa = self.num_to_f64(a);
                let fb = self.num_to_f64(b);
                if fa.is_nan() || fb.is_nan() {
                    false
                } else {
                    fa == fb
                }
            }
            (JsValue::String(a), JsValue::String(b)) => a == b,
            _ => false,
        }
    }

    fn js_abstract_equal(&self, a: &JsValue, b: &JsValue) -> bool {
        // Simplified abstract equality
        match (a, b) {
            (JsValue::Undefined, JsValue::Null) | (JsValue::Null, JsValue::Undefined) => true,
            _ => {
                // Fall back to strict equality for same types
                if core::mem::discriminant(a) == core::mem::discriminant(b) {
                    self.js_strict_equal(a, b)
                } else {
                    // Numeric comparison
                    let na = self.to_f64(a);
                    let nb = self.to_f64(b);
                    if na.is_nan() || nb.is_nan() {
                        false
                    } else {
                        na == nb
                    }
                }
            }
        }
    }

    fn js_less_than(&self, a: &JsValue, b: &JsValue) -> bool {
        match (a, b) {
            (JsValue::String(sa), JsValue::String(sb)) => sa < sb,
            _ => {
                let na = self.to_f64(a);
                let nb = self.to_f64(b);
                if na.is_nan() || nb.is_nan() {
                    false
                } else {
                    na < nb
                }
            }
        }
    }

    fn num_to_f64(&self, n: &JsNumberType) -> f64 {
        match n {
            JsNumberType::Integer(i) => *i as f64,
            JsNumberType::Float(f) => *f,
            JsNumberType::NaN => f64::NAN,
            JsNumberType::PositiveInfinity => f64::INFINITY,
            JsNumberType::NegativeInfinity => f64::NEG_INFINITY,
        }
    }

    fn to_display_string(&self, val: &JsValue) -> String {
        match val {
            JsValue::Undefined => "undefined".to_string(),
            JsValue::Null => "null".to_string(),
            JsValue::Boolean(b) => b.to_string(),
            JsValue::Number(n) => match n {
                JsNumberType::Integer(i) => i.to_string(),
                JsNumberType::Float(f) => f.to_string(),
                JsNumberType::NaN => "NaN".to_string(),
                JsNumberType::PositiveInfinity => "Infinity".to_string(),
                JsNumberType::NegativeInfinity => "-Infinity".to_string(),
            },
            JsValue::String(s) => s.clone(),
            _ => "[object Object]".to_string(),
        }
    }
}
