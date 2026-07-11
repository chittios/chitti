//! Register-based bytecode virtual machine.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::lex_env::JsLexEnvironmentType;
use crate::runner::ds::object_property::PropertyKey;
use crate::runner::ds::operations::type_conversion::to_string;
use crate::runner::ds::value::{JsNumberType, JsValue, JsValueOrSelf};
use crate::runner::plugin::registry::BuiltInRegistry;
use crate::runner::plugin::types::EvalContext;

use super::reg_bytecode::{RegChunk, RegOpCode};

/// Result of VM execution.
pub enum RegVmResult {
    Ok(JsValue),
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

pub struct RegVm<'a> {
    chunk: &'a RegChunk,
    ip: usize,
    registers: Vec<JsValue>,
    get_var_cache: Vec<Option<EnvCacheEntry>>,
    set_var_cache: Vec<Option<EnvCacheEntry>>,
    get_prop_cache: Vec<Option<PropCacheEntry>>,
    ctx: EvalContext,
    registry: &'a BuiltInRegistry,
}

impl<'a> RegVm<'a> {
    pub fn new(chunk: &'a RegChunk, ctx: EvalContext, registry: &'a BuiltInRegistry) -> Self {
        RegVm {
            chunk,
            ip: 0,
            registers: vec![JsValue::Undefined; chunk.register_count as usize],
            get_var_cache: vec![None; chunk.code.len()],
            set_var_cache: vec![None; chunk.code.len()],
            get_prop_cache: vec![None; chunk.code.len()],
            ctx,
            registry,
        }
    }

    pub fn into_ctx(mut self) -> EvalContext {
        self.sync_locals_into_ctx();
        self.ctx
    }

    pub fn run(&mut self) -> RegVmResult {
        loop {
            if self.ip >= self.chunk.code.len() {
                return RegVmResult::Ok(JsValue::Undefined);
            }
            let instr_index = self.ip;
            let instr = &self.chunk.code[instr_index];
            self.ip += 1;
            match instr.op {
                RegOpCode::LoadConst => {
                    let val = self.chunk.constants[instr.imm as usize].clone();
                    self.write_reg(instr.dst, val);
                }
                RegOpCode::LoadUndefined => self.write_reg(instr.dst, JsValue::Undefined),
                RegOpCode::LoadNull => self.write_reg(instr.dst, JsValue::Null),
                RegOpCode::LoadTrue => self.write_reg(instr.dst, JsValue::Boolean(true)),
                RegOpCode::LoadFalse => self.write_reg(instr.dst, JsValue::Boolean(false)),
                RegOpCode::Move => {
                    let val = self.read_reg(instr.src1).clone();
                    self.write_reg(instr.dst, val);
                }
                RegOpCode::Add => {
                    let a = self.read_reg(instr.src1).clone();
                    let b = self.read_reg(instr.src2).clone();
                    self.write_reg(instr.dst, self.js_add(a, b));
                }
                RegOpCode::Sub => {
                    let a = self.read_reg(instr.src1).clone();
                    let b = self.read_reg(instr.src2).clone();
                    self.write_reg(instr.dst, self.js_sub(a, b));
                }
                RegOpCode::Mul => {
                    let a = self.read_reg(instr.src1).clone();
                    let b = self.read_reg(instr.src2).clone();
                    self.write_reg(instr.dst, self.js_mul(a, b));
                }
                RegOpCode::Div => {
                    let a = self.read_reg(instr.src1).clone();
                    let b = self.read_reg(instr.src2).clone();
                    self.write_reg(instr.dst, self.js_div(a, b));
                }
                RegOpCode::Mod => {
                    let a = self.read_reg(instr.src1).clone();
                    let b = self.read_reg(instr.src2).clone();
                    self.write_reg(instr.dst, self.js_mod(a, b));
                }
                RegOpCode::Negate => {
                    let val = self.read_reg(instr.src1).clone();
                    self.write_reg(instr.dst, self.js_negate(val));
                }
                RegOpCode::BitAnd => {
                    let a = self.read_reg(instr.src1).clone();
                    let b = self.read_reg(instr.src2).clone();
                    self.write_reg(instr.dst, self.js_bit_and(a, b));
                }
                RegOpCode::BitOr => {
                    let a = self.read_reg(instr.src1).clone();
                    let b = self.read_reg(instr.src2).clone();
                    self.write_reg(instr.dst, self.js_bit_or(a, b));
                }
                RegOpCode::BitXor => {
                    let a = self.read_reg(instr.src1).clone();
                    let b = self.read_reg(instr.src2).clone();
                    self.write_reg(instr.dst, self.js_bit_xor(a, b));
                }
                RegOpCode::BitNot => {
                    let val = self.read_reg(instr.src1).clone();
                    let a = self.to_i32(val);
                    self.write_reg(instr.dst, JsValue::Number(JsNumberType::Integer((!a) as i64)));
                }
                RegOpCode::ShiftLeft => {
                    let a = self.read_reg(instr.src1).clone();
                    let b = self.read_reg(instr.src2).clone();
                    self.write_reg(instr.dst, self.js_shift_left(a, b));
                }
                RegOpCode::ShiftRight => {
                    let a = self.read_reg(instr.src1).clone();
                    let b = self.read_reg(instr.src2).clone();
                    self.write_reg(instr.dst, self.js_shift_right(a, b));
                }
                RegOpCode::UShiftRight => {
                    let a = self.read_reg(instr.src1).clone();
                    let b = self.read_reg(instr.src2).clone();
                    self.write_reg(instr.dst, self.js_ushift_right(a, b));
                }
                RegOpCode::StrictEqual => {
                    let res = self.js_strict_equal(self.read_reg(instr.src1), self.read_reg(instr.src2));
                    self.write_reg(instr.dst, JsValue::Boolean(res));
                }
                RegOpCode::StrictNotEqual => {
                    let res = !self.js_strict_equal(self.read_reg(instr.src1), self.read_reg(instr.src2));
                    self.write_reg(instr.dst, JsValue::Boolean(res));
                }
                RegOpCode::Equal => {
                    let res = self.js_abstract_equal(self.read_reg(instr.src1), self.read_reg(instr.src2));
                    self.write_reg(instr.dst, JsValue::Boolean(res));
                }
                RegOpCode::NotEqual => {
                    let res = !self.js_abstract_equal(self.read_reg(instr.src1), self.read_reg(instr.src2));
                    self.write_reg(instr.dst, JsValue::Boolean(res));
                }
                RegOpCode::LessThan => {
                    let res = self.js_less_than(self.read_reg(instr.src1), self.read_reg(instr.src2));
                    self.write_reg(instr.dst, JsValue::Boolean(res));
                }
                RegOpCode::LessEqual => {
                    let res = !self.js_less_than(self.read_reg(instr.src2), self.read_reg(instr.src1));
                    self.write_reg(instr.dst, JsValue::Boolean(res));
                }
                RegOpCode::GreaterThan => {
                    let res = self.js_less_than(self.read_reg(instr.src2), self.read_reg(instr.src1));
                    self.write_reg(instr.dst, JsValue::Boolean(res));
                }
                RegOpCode::GreaterEqual => {
                    let res = !self.js_less_than(self.read_reg(instr.src1), self.read_reg(instr.src2));
                    self.write_reg(instr.dst, JsValue::Boolean(res));
                }
                RegOpCode::Not => {
                    let val = self.read_reg(instr.src1).clone();
                    self.write_reg(instr.dst, JsValue::Boolean(!self.is_truthy(&val)));
                }
                RegOpCode::TypeOf => {
                    let val = self.read_reg(instr.src1).clone();
                    let type_str = match &val {
                        JsValue::Undefined => "undefined",
                        JsValue::Null => "object",
                        JsValue::Boolean(_) => "boolean",
                        JsValue::Number(_) => "number",
                        JsValue::String(_) => "string",
                        JsValue::BigInt(_) => "bigint",
                        JsValue::Symbol(_) => "symbol",
                        JsValue::Object(_) => "object",
                    };
                    self.write_reg(instr.dst, JsValue::String(type_str.to_string()));
                }
                RegOpCode::UnaryPlus => {
                    let val = self.read_reg(instr.src1).clone();
                    self.write_reg(instr.dst, self.to_number_value(val));
                }
                RegOpCode::GetVar => {
                    let name = self.chunk.get_name(instr.imm);
                    match self.get_var_cached(instr_index, name) {
                        Ok(val) => self.write_reg(instr.dst, val),
                        Err(e) => return RegVmResult::Error(e),
                    }
                }
                RegOpCode::SetVar => {
                    let name = self.chunk.get_name(instr.imm);
                    let val = self.read_reg(instr.src1).clone();
                    if let Err(e) = self.set_var_cached(instr_index, name, val) {
                        return RegVmResult::Error(e);
                    }
                }
                RegOpCode::DeclareVar => {
                    let name = self.chunk.get_name(instr.imm);
                    if !self.ctx.has_var_binding(name) {
                        if let Err(e) = self.ctx.create_var_binding(name) {
                            return RegVmResult::Error(e);
                        }
                    }
                }
                RegOpCode::DeclareLet => {
                    let name = self.chunk.get_name(instr.imm);
                    if let Err(e) = self.ctx.create_binding(name, false) {
                        return RegVmResult::Error(e);
                    }
                }
                RegOpCode::DeclareConst => {
                    let name = self.chunk.get_name(instr.imm);
                    if let Err(e) = self.ctx.create_binding(name, true) {
                        return RegVmResult::Error(e);
                    }
                }
                RegOpCode::InitVar => {
                    let name = self.chunk.get_name(instr.imm);
                    let val = self.read_reg(instr.src1).clone();
                    if let Err(e) = self.ctx.initialize_var_binding(name, val.clone()) {
                        return RegVmResult::Error(e);
                    }
                    let _ = self.ctx.set_var_binding(name, val);
                }
                RegOpCode::InitBinding => {
                    let name = self.chunk.get_name(instr.imm);
                    let val = self.read_reg(instr.src1).clone();
                    if let Err(e) = self.ctx.initialize_binding(name, val) {
                        return RegVmResult::Error(e);
                    }
                }
                RegOpCode::GetProp => {
                    let obj_val = self.read_reg(instr.src1).clone();
                    let prop_name = self.chunk.get_name(instr.imm);
                    match self.get_prop_cached(instr_index, &obj_val, prop_name) {
                        Ok(val) => self.write_reg(instr.dst, val),
                        Err(e) => return RegVmResult::Error(e),
                    }
                }
                RegOpCode::SetProp => {
                    let obj_val = self.read_reg(instr.src1).clone();
                    let val = self.read_reg(instr.src2).clone();
                    let prop_name = self.chunk.get_name(instr.imm);
                    if let Err(e) = self.set_prop_value(&obj_val, prop_name, val.clone()) {
                        return RegVmResult::Error(e);
                    }
                    self.write_reg(instr.dst, val);
                }
                RegOpCode::GetElem => {
                    let obj_val = self.read_reg(instr.src1).clone();
                    let key_val = self.read_reg(instr.src2).clone();
                    let prop_name = match self.to_property_key(&key_val) {
                        Ok(key) => key,
                        Err(e) => return RegVmResult::Error(e),
                    };
                    match self.get_prop_cached(instr_index, &obj_val, &prop_name) {
                        Ok(val) => self.write_reg(instr.dst, val),
                        Err(e) => return RegVmResult::Error(e),
                    }
                }
                RegOpCode::SetElem => {
                    let obj_val = self.read_reg(instr.src1).clone();
                    let key_val = self.read_reg(instr.src2).clone();
                    let val = self.read_reg(instr.dst).clone();
                    let prop_name = match self.to_property_key(&key_val) {
                        Ok(key) => key,
                        Err(e) => return RegVmResult::Error(e),
                    };
                    if let Err(e) = self.set_prop_value(&obj_val, &prop_name, val.clone()) {
                        return RegVmResult::Error(e);
                    }
                }
                RegOpCode::Jump => {
                    self.ip = instr.imm as usize;
                }
                RegOpCode::JumpIfFalse => {
                    let val = self.read_reg(instr.src1).clone();
                    if !self.is_truthy(&val) {
                        self.ip = instr.imm as usize;
                    }
                }
                RegOpCode::JumpIfTrue => {
                    let val = self.read_reg(instr.src1).clone();
                    if self.is_truthy(&val) {
                        self.ip = instr.imm as usize;
                    }
                }
                RegOpCode::Call => {
                    let _callee = self.read_reg(instr.src1).clone();
                    self.write_reg(instr.dst, JsValue::Undefined);
                }
                RegOpCode::CallMethod => {
                    let object = self.read_reg(instr.src1).clone();
                    let method_name = self.chunk.get_name(instr.src2);
                    let obj_name = self.resolve_object_name(&object);
                    if let Some(ref name) = obj_name {
                        if let Some(builtin_fn) = self.registry.get_method(name, method_name) {
                            match builtin_fn.call(&mut self.ctx, object, Vec::new()) {
                                Ok(result) => {
                                    self.write_reg(instr.dst, result);
                                    continue;
                                }
                                Err(e) => return RegVmResult::Error(e),
                            }
                        }
                    }
                    self.write_reg(instr.dst, JsValue::Undefined);
                }
                RegOpCode::Return => {
                    let val = self.read_reg(instr.src1).clone();
                    return RegVmResult::Ok(val);
                }
                RegOpCode::Halt => {
                    return RegVmResult::Ok(JsValue::Undefined);
                }
            }
        }
    }

    fn read_reg(&self, reg: u32) -> &JsValue {
        self.registers.get(reg as usize).unwrap_or(&JsValue::Undefined)
    }

    fn write_reg(&mut self, reg: u32, value: JsValue) {
        if let Some(slot) = self.registers.get_mut(reg as usize) {
            *slot = value;
        }
    }

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
        for local in &self.chunk.locals {
            let name = self.chunk.get_name(local.name_idx);
            let val = self.read_reg(local.reg).clone();
            if !self.ctx.has_var_binding(name) {
                let _ = self.ctx.create_var_binding(name);
                let _ = self.ctx.initialize_var_binding(name, val.clone());
            } else {
                let _ = self.ctx.set_var_binding(name, val.clone());
            }
        }
    }

    fn resolve_object_name(&self, value: &JsValue) -> Option<String> {
        match value {
            JsValue::String(_) => Some("String".to_string()),
            JsValue::Number(_) => Some("Number".to_string()),
            _ => None,
        }
    }

    fn is_truthy(&self, val: &JsValue) -> bool {
        match val {
            JsValue::Undefined | JsValue::Null => false,
            JsValue::Boolean(b) => *b,
            JsValue::Number(n) => match n {
                JsNumberType::NaN => false,
                JsNumberType::Integer(i) => *i != 0,
                JsNumberType::Float(f) => *f != 0.0,
                JsNumberType::PositiveInfinity => true,
                JsNumberType::NegativeInfinity => true,
            },
            JsValue::String(s) => !s.is_empty(),
            JsValue::BigInt(b) => !num_traits::Zero::is_zero(b),
            JsValue::Symbol(_) => true,
            JsValue::Object(_) => true,
        }
    }

    fn to_f64(&self, val: &JsValue) -> f64 {
        match val {
            JsValue::Undefined => f64::NAN,
            JsValue::Null => 0.0,
            JsValue::Boolean(b) => if *b { 1.0 } else { 0.0 },
            JsValue::Number(n) => self.num_to_f64(n),
            JsValue::String(s) => s.parse::<f64>().unwrap_or(f64::NAN),
            JsValue::BigInt(b) => num_traits::ToPrimitive::to_f64(b).unwrap_or(f64::NAN),
            JsValue::Symbol(_) => f64::NAN,
            JsValue::Object(_) => f64::NAN,
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

    fn f64_to_jsvalue(&self, n: f64) -> JsValue {
        if n.is_nan() {
            JsValue::Number(JsNumberType::NaN)
        } else if n == f64::INFINITY {
            JsValue::Number(JsNumberType::PositiveInfinity)
        } else if n == f64::NEG_INFINITY {
            JsValue::Number(JsNumberType::NegativeInfinity)
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

    fn js_add(&self, a: JsValue, b: JsValue) -> JsValue {
        match (&a, &b) {
            (JsValue::String(sa), JsValue::String(sb)) => JsValue::String(format!("{}{}", sa, sb)),
            (JsValue::String(sa), _) => JsValue::String(format!("{}{}", sa, self.to_display_string(&b))),
            (_, JsValue::String(sb)) => JsValue::String(format!("{}{}", self.to_display_string(&a), sb)),
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

    fn js_bit_and(&self, a: JsValue, b: JsValue) -> JsValue {
        let av = self.to_i32(a);
        let bv = self.to_i32(b);
        JsValue::Number(JsNumberType::Integer((av & bv) as i64))
    }

    fn js_bit_or(&self, a: JsValue, b: JsValue) -> JsValue {
        let av = self.to_i32(a);
        let bv = self.to_i32(b);
        JsValue::Number(JsNumberType::Integer((av | bv) as i64))
    }

    fn js_bit_xor(&self, a: JsValue, b: JsValue) -> JsValue {
        let av = self.to_i32(a);
        let bv = self.to_i32(b);
        JsValue::Number(JsNumberType::Integer((av ^ bv) as i64))
    }

    fn js_shift_left(&self, a: JsValue, b: JsValue) -> JsValue {
        let av = self.to_i32(a);
        let bv = self.to_u32(b);
        JsValue::Number(JsNumberType::Integer((av << (bv & 0x1f)) as i64))
    }

    fn js_shift_right(&self, a: JsValue, b: JsValue) -> JsValue {
        let av = self.to_i32(a);
        let bv = self.to_u32(b);
        JsValue::Number(JsNumberType::Integer((av >> (bv & 0x1f)) as i64))
    }

    fn js_ushift_right(&self, a: JsValue, b: JsValue) -> JsValue {
        let av = self.to_u32(a);
        let bv = self.to_u32(b);
        JsValue::Number(JsNumberType::Integer((av >> (bv & 0x1f)) as i64))
    }

    fn to_i32(&self, val: JsValue) -> i32 {
        self.to_f64(&val) as i32
    }

    fn to_u32(&self, val: JsValue) -> u32 {
        self.to_f64(&val) as u32
    }

    fn js_strict_equal(&self, a: &JsValue, b: &JsValue) -> bool {
        match (a, b) {
            (JsValue::Undefined, JsValue::Undefined) => true,
            (JsValue::Null, JsValue::Null) => true,
            (JsValue::Boolean(a), JsValue::Boolean(b)) => a == b,
            (JsValue::Number(a), JsValue::Number(b)) => self.num_to_f64(a) == self.num_to_f64(b),
            (JsValue::String(a), JsValue::String(b)) => a == b,
            (JsValue::Symbol(a), JsValue::Symbol(b)) => a == b,
            (JsValue::Object(a), JsValue::Object(b)) => alloc::rc::Rc::ptr_eq(a, b),
            _ => false,
        }
    }

    fn js_abstract_equal(&self, a: &JsValue, b: &JsValue) -> bool {
        self.js_strict_equal(a, b)
    }

    fn js_less_than(&self, a: &JsValue, b: &JsValue) -> bool {
        match (a, b) {
            (JsValue::String(sa), JsValue::String(sb)) => sa < sb,
            _ => self.to_f64(a) < self.to_f64(b),
        }
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
            JsValue::BigInt(b) => b.to_string(),
            JsValue::Symbol(_) => "symbol".to_string(),
            JsValue::Object(_) => "[object]".to_string(),
        }
    }
}
