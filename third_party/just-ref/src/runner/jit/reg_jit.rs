//! Cranelift-based JIT for register bytecode (numeric prototype).

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::mem;

use cranelift::prelude::{types, AbiParam, FunctionBuilder, FunctionBuilderContext, InstBuilder, MemFlags, Variable, FloatCC};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};

use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::{JsNumberType, JsValue};

use super::reg_bytecode::{RegChunk, RegOpCode};

pub type RegJitFn = unsafe extern "C" fn(*mut f64) -> f64;

pub struct RegJitFunction {
    _func_id: FuncId,
    code_ptr: *const u8,
}

impl RegJitFunction {
    pub unsafe fn call(&self, registers: &mut [f64]) -> f64 {
        let func: RegJitFn = mem::transmute(self.code_ptr);
        func(registers.as_mut_ptr())
    }
}

pub struct RegJit {
    module: JITModule,
}

impl RegJit {
    pub fn new() -> Result<Self, JErrorType> {
        if !cfg!(target_arch = "x86_64") {
            return Err(JErrorType::TypeError(
                "Cranelift JIT prototype is only supported on x86_64".to_string(),
            ));
        }
        let builder = JITBuilder::new(default_libcall_names())
            .map_err(|e| JErrorType::TypeError(format!("Cranelift init failed: {}", e)))?;
        let module = JITModule::new(builder);
        Ok(RegJit { module })
    }

    /// Pre-scan: return Err if the chunk contains ops the numeric JIT cannot handle.
    fn can_jit_compile(chunk: &RegChunk) -> Result<(), JErrorType> {
        for instr in &chunk.code {
            match instr.op {
                RegOpCode::GetProp
                | RegOpCode::SetProp
                | RegOpCode::GetElem
                | RegOpCode::SetElem
                | RegOpCode::Call
                | RegOpCode::CallMethod
                | RegOpCode::TypeOf => {
                    return Err(JErrorType::TypeError(format!(
                        "JIT bail: unsupported op {:?}",
                        instr.op
                    )));
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn compile(&mut self, chunk: &RegChunk) -> Result<RegJitFunction, JErrorType> {
        Self::can_jit_compile(chunk)?;
        let ptr_ty = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_ty));
        sig.returns.push(AbiParam::new(types::F64));

        let func_id = self
            .module
            .declare_function("reg_jit_entry", Linkage::Local, &sig)
            .map_err(|e| JErrorType::TypeError(format!("declare_function failed: {}", e)))?;

        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;
        let mut builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let regs_ptr = builder.block_params(entry_block)[0];
        let reg_var = Variable::from_u32(0);
        builder.declare_var(reg_var, ptr_ty);
        builder.def_var(reg_var, regs_ptr);

        let mut local_regs = vec![None; chunk.names.len()];
        for local in &chunk.locals {
            if let Some(slot) = local_regs.get_mut(local.name_idx as usize) {
                *slot = Some(local.reg);
            }
        }
        let mut lexical_names = vec![false; chunk.names.len()];
        for instr in &chunk.code {
            if matches!(instr.op, RegOpCode::DeclareLet | RegOpCode::DeclareConst | RegOpCode::InitBinding) {
                if let Some(slot) = lexical_names.get_mut(instr.imm as usize) {
                    *slot = true;
                }
            }
        }

        let mut blocks = Vec::with_capacity(chunk.code.len());
        for _ in 0..chunk.code.len() {
            blocks.push(builder.create_block());
        }
        let exit_block = builder.create_block();

        builder.ins().jump(blocks[0], &[]);

        for (idx, instr) in chunk.code.iter().enumerate() {
            let block = blocks[idx];
            builder.switch_to_block(block);

            let next_block = if idx + 1 < blocks.len() {
                Some(blocks[idx + 1])
            } else {
                None
            };

            match instr.op {
                RegOpCode::LoadConst => {
                    let value = chunk
                        .constants
                        .get(instr.imm as usize)
                        .ok_or_else(|| JErrorType::TypeError("Const out of range".to_string()))?;
                    let num = match value {
                        JsValue::Number(n) => self.num_to_f64(n),
                        _ => {
                            return Err(JErrorType::TypeError(
                                "JIT supports only numeric constants".to_string(),
                            ))
                        }
                    };
                    let val = builder.ins().f64const(num);
                    self.store_reg(&mut builder, reg_var, instr.dst, val);
                }
                RegOpCode::LoadUndefined => {
                    let val = builder.ins().f64const(f64::NAN);
                    self.store_reg(&mut builder, reg_var, instr.dst, val);
                }
                RegOpCode::LoadNull | RegOpCode::LoadFalse => {
                    let val = builder.ins().f64const(0.0);
                    self.store_reg(&mut builder, reg_var, instr.dst, val);
                }
                RegOpCode::LoadTrue => {
                    let val = builder.ins().f64const(1.0);
                    self.store_reg(&mut builder, reg_var, instr.dst, val);
                }
                RegOpCode::Move => {
                    let val = self.load_reg(&mut builder, reg_var, instr.src1);
                    self.store_reg(&mut builder, reg_var, instr.dst, val);
                }
                RegOpCode::Add | RegOpCode::Sub | RegOpCode::Mul | RegOpCode::Div => {
                    let a = self.load_reg(&mut builder, reg_var, instr.src1);
                    let b = self.load_reg(&mut builder, reg_var, instr.src2);
                    let res = match instr.op {
                        RegOpCode::Add => builder.ins().fadd(a, b),
                        RegOpCode::Sub => builder.ins().fsub(a, b),
                        RegOpCode::Mul => builder.ins().fmul(a, b),
                        RegOpCode::Div => builder.ins().fdiv(a, b),
                        _ => unreachable!(),
                    };
                    self.store_reg(&mut builder, reg_var, instr.dst, res);
                }
                RegOpCode::GetVar => {
                    let name_idx = instr.imm as usize;
                    if name_idx >= local_regs.len() || lexical_names[name_idx] {
                        return Err(JErrorType::TypeError(
                            "JIT supports only local var GetVar".to_string(),
                        ));
                    }
                    let local_reg = local_regs[name_idx].ok_or_else(|| {
                        JErrorType::TypeError("JIT supports only local var GetVar".to_string())
                    })?;
                    if instr.dst != local_reg {
                        let val = self.load_reg(&mut builder, reg_var, local_reg);
                        self.store_reg(&mut builder, reg_var, instr.dst, val);
                    }
                }
                RegOpCode::SetVar => {
                    let name_idx = instr.imm as usize;
                    if name_idx >= local_regs.len() || lexical_names[name_idx] {
                        return Err(JErrorType::TypeError(
                            "JIT supports only local var SetVar".to_string(),
                        ));
                    }
                    let local_reg = local_regs[name_idx].ok_or_else(|| {
                        JErrorType::TypeError("JIT supports only local var SetVar".to_string())
                    })?;
                    if instr.src1 != local_reg {
                        let val = self.load_reg(&mut builder, reg_var, instr.src1);
                        self.store_reg(&mut builder, reg_var, local_reg, val);
                    }
                }
                RegOpCode::DeclareVar | RegOpCode::DeclareLet | RegOpCode::DeclareConst => {
                    let name_idx = instr.imm as usize;
                    if name_idx >= local_regs.len() {
                        return Err(JErrorType::TypeError(
                            "JIT supports only local var Declare*".to_string(),
                        ));
                    }
                    let _local_reg = local_regs[name_idx].ok_or_else(|| {
                        JErrorType::TypeError("JIT supports only local var Declare*".to_string())
                    })?;
                }
                RegOpCode::InitVar | RegOpCode::InitBinding => {
                    let name_idx = instr.imm as usize;
                    if name_idx >= local_regs.len() {
                        return Err(JErrorType::TypeError(
                            "JIT supports only local var Init*".to_string(),
                        ));
                    }
                    let local_reg = local_regs[name_idx].ok_or_else(|| {
                        JErrorType::TypeError("JIT supports only local var Init*".to_string())
                    })?;
                    if instr.src1 != local_reg {
                        let val = self.load_reg(&mut builder, reg_var, instr.src1);
                        self.store_reg(&mut builder, reg_var, local_reg, val);
                    }
                }
                RegOpCode::Mod => {
                    let a = self.load_reg(&mut builder, reg_var, instr.src1);
                    let b = self.load_reg(&mut builder, reg_var, instr.src2);
                    let zero = builder.ins().f64const(0.0);
                    let b_is_zero = builder.ins().fcmp(FloatCC::Equal, b, zero);
                    let b_is_nan = builder.ins().fcmp(FloatCC::Unordered, b, b);
                    let a_is_nan = builder.ins().fcmp(FloatCC::Unordered, a, a);
                    let b_bad = builder.ins().bor(b_is_zero, b_is_nan);
                    let is_bad = builder.ins().bor(b_bad, a_is_nan);

                    let ok_block = builder.create_block();
                    let bad_block = builder.create_block();
                    let cont_block = next_block.unwrap_or(exit_block);

                    builder.ins().brif(is_bad, bad_block, &[], ok_block, &[]);
                    builder.seal_block(block);

                    builder.switch_to_block(ok_block);
                    let div = builder.ins().fdiv(a, b);
                    let quot = builder.ins().fcvt_to_sint(types::I64, div);
                    let quot_f = builder.ins().fcvt_from_sint(types::F64, quot);
                    let prod = builder.ins().fmul(quot_f, b);
                    let res = builder.ins().fsub(a, prod);
                    self.store_reg(&mut builder, reg_var, instr.dst, res);
                    builder.ins().jump(cont_block, &[]);
                    builder.seal_block(ok_block);

                    builder.switch_to_block(bad_block);
                    let nan = builder.ins().f64const(f64::NAN);
                    self.store_reg(&mut builder, reg_var, instr.dst, nan);
                    builder.ins().jump(cont_block, &[]);
                    builder.seal_block(bad_block);
                    continue;
                }
                RegOpCode::Negate => {
                    let a = self.load_reg(&mut builder, reg_var, instr.src1);
                    let zero = builder.ins().f64const(0.0);
                    let res = builder.ins().fsub(zero, a);
                    self.store_reg(&mut builder, reg_var, instr.dst, res);
                }
                RegOpCode::UnaryPlus => {
                    let val = self.load_reg(&mut builder, reg_var, instr.src1);
                    self.store_reg(&mut builder, reg_var, instr.dst, val);
                }
                RegOpCode::Not => {
                    let val = self.load_reg(&mut builder, reg_var, instr.src1);
                    let truthy = self.emit_truthy(&mut builder, val);
                    let one = builder.ins().f64const(1.0);
                    let zero = builder.ins().f64const(0.0);
                    let res = builder.ins().select(truthy, zero, one);
                    self.store_reg(&mut builder, reg_var, instr.dst, res);
                }
                RegOpCode::BitAnd | RegOpCode::BitOr | RegOpCode::BitXor => {
                    let a = self.load_reg(&mut builder, reg_var, instr.src1);
                    let b = self.load_reg(&mut builder, reg_var, instr.src2);
                    let a_i32 = self.emit_to_i32(&mut builder, a);
                    let b_i32 = self.emit_to_i32(&mut builder, b);
                    let res_i32 = match instr.op {
                        RegOpCode::BitAnd => builder.ins().band(a_i32, b_i32),
                        RegOpCode::BitOr => builder.ins().bor(a_i32, b_i32),
                        RegOpCode::BitXor => builder.ins().bxor(a_i32, b_i32),
                        _ => unreachable!(),
                    };
                    let res = self.emit_from_i32(&mut builder, res_i32);
                    self.store_reg(&mut builder, reg_var, instr.dst, res);
                }
                RegOpCode::BitNot => {
                    let val = self.load_reg(&mut builder, reg_var, instr.src1);
                    let a_i32 = self.emit_to_i32(&mut builder, val);
                    let res_i32 = builder.ins().bnot(a_i32);
                    let res = self.emit_from_i32(&mut builder, res_i32);
                    self.store_reg(&mut builder, reg_var, instr.dst, res);
                }
                RegOpCode::ShiftLeft | RegOpCode::ShiftRight | RegOpCode::UShiftRight => {
                    let a = self.load_reg(&mut builder, reg_var, instr.src1);
                    let b = self.load_reg(&mut builder, reg_var, instr.src2);
                    let mask = builder.ins().iconst(types::I32, 0x1f);
                    let b_u32 = self.emit_to_u32(&mut builder, b);
                    let shift = builder.ins().band(b_u32, mask);
                    let res = match instr.op {
                        RegOpCode::ShiftLeft => {
                            let a_i32 = self.emit_to_i32(&mut builder, a);
                            let res_i32 = builder.ins().ishl(a_i32, shift);
                            self.emit_from_i32(&mut builder, res_i32)
                        }
                        RegOpCode::ShiftRight => {
                            let a_i32 = self.emit_to_i32(&mut builder, a);
                            let res_i32 = builder.ins().sshr(a_i32, shift);
                            self.emit_from_i32(&mut builder, res_i32)
                        }
                        RegOpCode::UShiftRight => {
                            let a_u32 = self.emit_to_u32(&mut builder, a);
                            let res_u32 = builder.ins().ushr(a_u32, shift);
                            self.emit_from_u32(&mut builder, res_u32)
                        }
                        _ => unreachable!(),
                    };
                    self.store_reg(&mut builder, reg_var, instr.dst, res);
                }
                RegOpCode::LessThan
                | RegOpCode::LessEqual
                | RegOpCode::GreaterThan
                | RegOpCode::GreaterEqual
                | RegOpCode::Equal
                | RegOpCode::NotEqual
                | RegOpCode::StrictEqual
                | RegOpCode::StrictNotEqual => {
                    let a = self.load_reg(&mut builder, reg_var, instr.src1);
                    let b = self.load_reg(&mut builder, reg_var, instr.src2);
                    let cc = match instr.op {
                        RegOpCode::LessThan => FloatCC::LessThan,
                        RegOpCode::LessEqual => FloatCC::LessThanOrEqual,
                        RegOpCode::GreaterThan => FloatCC::GreaterThan,
                        RegOpCode::GreaterEqual => FloatCC::GreaterThanOrEqual,
                        RegOpCode::Equal | RegOpCode::StrictEqual => FloatCC::Equal,
                        RegOpCode::NotEqual | RegOpCode::StrictNotEqual => FloatCC::NotEqual,
                        _ => FloatCC::Equal,
                    };
                    let cmp = builder.ins().fcmp(cc, a, b);
                    let one = builder.ins().f64const(1.0);
                    let zero = builder.ins().f64const(0.0);
                    let res = builder.ins().select(cmp, one, zero);
                    self.store_reg(&mut builder, reg_var, instr.dst, res);
                }
                RegOpCode::Jump => {
                    builder.ins().jump(blocks[instr.imm as usize], &[]);
                    builder.seal_block(block);
                    continue;
                }
                RegOpCode::JumpIfFalse | RegOpCode::JumpIfTrue => {
                    let val = self.load_reg(&mut builder, reg_var, instr.src1);
                    let truthy = self.emit_truthy(&mut builder, val);
                    let target = blocks[instr.imm as usize];
                    let fallthrough = next_block.unwrap_or(exit_block);
                    if instr.op == RegOpCode::JumpIfFalse {
                        builder.ins().brif(truthy, fallthrough, &[], target, &[]);
                    } else {
                        builder.ins().brif(truthy, target, &[], fallthrough, &[]);
                    }
                    builder.seal_block(block);
                    continue;
                }
                // These ops are rejected by can_jit_compile; unreachable here.
                RegOpCode::GetProp
                | RegOpCode::SetProp
                | RegOpCode::GetElem
                | RegOpCode::SetElem
                | RegOpCode::Call
                | RegOpCode::CallMethod
                | RegOpCode::TypeOf => {
                    unreachable!("pre-scan should have rejected {:?}", instr.op);
                }
                RegOpCode::Return | RegOpCode::Halt => {
                    let val = if instr.op == RegOpCode::Return {
                        self.load_reg(&mut builder, reg_var, instr.src1)
                    } else {
                        builder.ins().f64const(0.0)
                    };
                    builder.ins().return_(&[val]);
                    builder.seal_block(block);
                    continue;
                }
                #[allow(unreachable_patterns)]
                _ => {
                    return Err(JErrorType::TypeError(format!(
                        "Unsupported op in JIT prototype: {:?}",
                        instr.op
                    )));
                }
            }

            if let Some(next) = next_block {
                builder.ins().jump(next, &[]);
            } else {
                builder.ins().jump(exit_block, &[]);
            }
            builder.seal_block(block);
        }

        builder.switch_to_block(exit_block);
        let zero = builder.ins().f64const(0.0);
        builder.ins().return_(&[zero]);
        builder.seal_block(exit_block);

        builder.finalize();

        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| JErrorType::TypeError(format!("define_function failed: {}", e)))?;
        self.module.clear_context(&mut ctx);
        let _ = self.module.finalize_definitions();

        let code_ptr = self.module.get_finalized_function(func_id);
        Ok(RegJitFunction { _func_id: func_id, code_ptr })
    }

    pub fn execute(&mut self, chunk: &RegChunk) -> Result<(f64, Vec<f64>), JErrorType> {
        let func = self.compile(chunk)?;
        let mut registers = vec![0.0; chunk.register_count as usize];
        let result = unsafe { func.call(&mut registers) };
        Ok((result, registers))
    }

    fn load_reg(&self, builder: &mut FunctionBuilder, reg_var: Variable, reg: u32) -> cranelift::prelude::Value {
        let ptr = builder.use_var(reg_var);
        let offset = (reg as i32) * 8;
        builder.ins().load(types::F64, MemFlags::new(), ptr, offset)
    }

    fn store_reg(
        &self,
        builder: &mut FunctionBuilder,
        reg_var: Variable,
        reg: u32,
        value: cranelift::prelude::Value,
    ) {
        let ptr = builder.use_var(reg_var);
        let offset = (reg as i32) * 8;
        builder.ins().store(MemFlags::new(), value, ptr, offset);
    }

    fn emit_truthy(&self, builder: &mut FunctionBuilder, val: cranelift::prelude::Value) -> cranelift::prelude::Value {
        let zero = builder.ins().f64const(0.0);
        let is_zero = builder.ins().fcmp(FloatCC::Equal, val, zero);
        let is_nan = builder.ins().fcmp(FloatCC::Unordered, val, val);
        let is_false = builder.ins().bor(is_zero, is_nan);
        builder.ins().bnot(is_false)
    }

    fn emit_to_i32(&self, builder: &mut FunctionBuilder, val: cranelift::prelude::Value) -> cranelift::prelude::Value {
        builder.ins().fcvt_to_sint(types::I32, val)
    }

    fn emit_to_u32(&self, builder: &mut FunctionBuilder, val: cranelift::prelude::Value) -> cranelift::prelude::Value {
        builder.ins().fcvt_to_uint(types::I32, val)
    }

    fn emit_from_i32(&self, builder: &mut FunctionBuilder, val: cranelift::prelude::Value) -> cranelift::prelude::Value {
        builder.ins().fcvt_from_sint(types::F64, val)
    }

    fn emit_from_u32(&self, builder: &mut FunctionBuilder, val: cranelift::prelude::Value) -> cranelift::prelude::Value {
        builder.ins().fcvt_from_uint(types::F64, val)
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
}
