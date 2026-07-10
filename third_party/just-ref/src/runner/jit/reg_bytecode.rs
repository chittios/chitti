//! Register-based bytecode instruction set and chunk structure.
//!
//! Defines a flat, register-based bytecode IR that the register VM will execute.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::value::JsValue;

/// Bytecode opcodes for the register-based VM.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum RegOpCode {
    // ── Constants & Literals ──────────────────────────────────
    /// dst = constant pool value.
    LoadConst,
    /// dst = undefined.
    LoadUndefined,
    /// dst = null.
    LoadNull,
    /// dst = true.
    LoadTrue,
    /// dst = false.
    LoadFalse,

    // ── Register moves ───────────────────────────────────────
    /// dst = src1.
    Move,

    // ── Arithmetic ───────────────────────────────────────────
    /// dst = src1 + src2.
    Add,
    /// dst = src1 - src2.
    Sub,
    /// dst = src1 * src2.
    Mul,
    /// dst = src1 / src2.
    Div,
    /// dst = src1 % src2.
    Mod,
    /// dst = -src1.
    Negate,

    // ── Bitwise ──────────────────────────────────────────────
    /// dst = src1 & src2.
    BitAnd,
    /// dst = src1 | src2.
    BitOr,
    /// dst = src1 ^ src2.
    BitXor,
    /// dst = ~src1.
    BitNot,
    /// dst = src1 << src2.
    ShiftLeft,
    /// dst = src1 >> src2.
    ShiftRight,
    /// dst = src1 >>> src2.
    UShiftRight,

    // ── Comparison ───────────────────────────────────────────
    /// dst = (src1 === src2).
    StrictEqual,
    /// dst = (src1 !== src2).
    StrictNotEqual,
    /// dst = (src1 == src2).
    Equal,
    /// dst = (src1 != src2).
    NotEqual,
    /// dst = (src1 < src2).
    LessThan,
    /// dst = (src1 <= src2).
    LessEqual,
    /// dst = (src1 > src2).
    GreaterThan,
    /// dst = (src1 >= src2).
    GreaterEqual,

    // ── Logical / Unary ──────────────────────────────────────
    /// dst = !src1.
    Not,
    /// dst = typeof src1.
    TypeOf,
    /// dst = +src1.
    UnaryPlus,

    // ── Variables / Bindings ─────────────────────────────────
    /// dst = get var by name (imm = name index).
    GetVar,
    /// set var by name (src1 = value, imm = name index).
    SetVar,
    /// declare var (imm = name index).
    DeclareVar,
    /// declare let (imm = name index).
    DeclareLet,
    /// declare const (imm = name index).
    DeclareConst,
    /// init var (src1 = value, imm = name index).
    InitVar,
    /// init let/const (src1 = value, imm = name index).
    InitBinding,

    // ── Objects & Properties ─────────────────────────────────
    /// dst = obj.prop (src1 = obj, imm = name index).
    GetProp,
    /// obj.prop = value (src1 = obj, src2 = value, imm = name index).
    SetProp,
    /// dst = obj[key] (src1 = obj, src2 = key).
    GetElem,
    /// obj[key] = value (dst = value, src1 = obj, src2 = key).
    SetElem,

    // ── Control Flow ─────────────────────────────────────────
    /// jump to imm (absolute index).
    Jump,
    /// if !src1 jump to imm.
    JumpIfFalse,
    /// if src1 jump to imm.
    JumpIfTrue,

    // ── Function calls ───────────────────────────────────────
    /// dst = call callee (src1 = callee, imm = arg count).
    Call,
    /// dst = call method (src1 = object, imm = arg count, src2 = name index).
    CallMethod,

    // ── Misc ─────────────────────────────────────────────────
    /// return src1.
    Return,
    /// halt (optional src1 as result).
    Halt,
}

/// A single register bytecode instruction.
#[derive(Debug, Clone)]
pub struct RegInstruction {
    pub op: RegOpCode,
    /// Destination register (when applicable).
    pub dst: u32,
    /// First source register.
    pub src1: u32,
    /// Second source register.
    pub src2: u32,
    /// Immediate operand (constant index, name index, jump target, arg count).
    pub imm: u32,
}

impl RegInstruction {
    pub fn new(op: RegOpCode, dst: u32, src1: u32, src2: u32, imm: u32) -> Self {
        RegInstruction { op, dst, src1, src2, imm }
    }

    pub fn simple(op: RegOpCode) -> Self {
        RegInstruction::new(op, 0, 0, 0, 0)
    }

    pub fn with_dst(op: RegOpCode, dst: u32) -> Self {
        RegInstruction::new(op, dst, 0, 0, 0)
    }

    pub fn with_dst_src(op: RegOpCode, dst: u32, src1: u32) -> Self {
        RegInstruction::new(op, dst, src1, 0, 0)
    }

    pub fn with_dst_srcs(op: RegOpCode, dst: u32, src1: u32, src2: u32) -> Self {
        RegInstruction::new(op, dst, src1, src2, 0)
    }

    pub fn with_dst_imm(op: RegOpCode, dst: u32, imm: u32) -> Self {
        RegInstruction::new(op, dst, 0, 0, imm)
    }

    pub fn with_src_imm(op: RegOpCode, src1: u32, imm: u32) -> Self {
        RegInstruction::new(op, 0, src1, 0, imm)
    }

    pub fn with_srcs_imm(op: RegOpCode, src1: u32, src2: u32, imm: u32) -> Self {
        RegInstruction::new(op, 0, src1, src2, imm)
    }
}

/// A compiled chunk of register bytecode with its constant pool.
#[derive(Debug, Clone)]
pub struct RegChunk {
    /// The register bytecode instructions.
    pub code: Vec<RegInstruction>,
    /// Constant pool — holds literal values.
    pub constants: Vec<JsValue>,
    /// Deduplicated name table for variable/property names.
    pub names: Vec<String>,
    /// Local register table (name index -> register index).
    pub locals: Vec<RegLocal>,
    /// Register count required by this chunk.
    pub register_count: u32,
}

impl RegChunk {
    pub fn new() -> Self {
        RegChunk {
            code: Vec::new(),
            constants: Vec::new(),
            names: Vec::new(),
            locals: Vec::new(),
            register_count: 0,
        }
    }

    /// Emit an instruction and return its index.
    pub fn emit(&mut self, instr: RegInstruction) -> usize {
        let idx = self.code.len();
        self.code.push(instr);
        idx
    }

    /// Emit a simple (no-operand) instruction.
    pub fn emit_op(&mut self, op: RegOpCode) -> usize {
        self.emit(RegInstruction::simple(op))
    }

    /// Add a constant to the pool and return its index.
    pub fn add_constant(&mut self, value: JsValue) -> u32 {
        let idx = self.constants.len();
        self.constants.push(value);
        idx as u32
    }

    /// Add a name to the deduplicated name table and return its index.
    pub fn add_name(&mut self, s: &str) -> u32 {
        for (i, existing) in self.names.iter().enumerate() {
            if existing == s {
                return i as u32;
            }
        }
        let idx = self.names.len();
        self.names.push(s.to_string());
        idx as u32
    }

    /// Get a name by index (zero-copy reference).
    #[inline]
    pub fn get_name(&self, idx: u32) -> &str {
        &self.names[idx as usize]
    }

    /// Add a local register for a name index and return its local index.
    pub fn add_local(&mut self, name_idx: u32, reg: u32) -> u32 {
        let idx = self.locals.len();
        self.locals.push(RegLocal { name_idx, reg });
        idx as u32
    }

    /// Get a local register's name.
    #[inline]
    pub fn get_local_name(&self, local_idx: u32) -> &str {
        let name_idx = self.locals[local_idx as usize].name_idx;
        self.get_name(name_idx)
    }

    /// Update the register count for this chunk.
    pub fn set_register_count(&mut self, count: u32) {
        self.register_count = count;
    }

    /// Disassemble the chunk for debugging.
    pub fn disassemble(&self, name: &str) -> String {
        let mut out = format!("== {} ==\n", name);
        for (i, instr) in self.code.iter().enumerate() {
            out.push_str(&format!("{:04}  {:?} dst={} src1={} src2={} imm={}", i, instr.op, instr.dst, instr.src1, instr.src2, instr.imm));
            match instr.op {
                RegOpCode::LoadConst => {
                    if let Some(val) = self.constants.get(instr.imm as usize) {
                        out.push_str(&format!("  const={}", val));
                    }
                }
                RegOpCode::GetVar | RegOpCode::SetVar
                | RegOpCode::DeclareVar | RegOpCode::DeclareLet
                | RegOpCode::DeclareConst | RegOpCode::InitVar
                | RegOpCode::InitBinding | RegOpCode::GetProp
                | RegOpCode::SetProp => {
                    if let Some(name) = self.names.get(instr.imm as usize) {
                        out.push_str(&format!("  name=\"{}\"", name));
                    }
                }
                RegOpCode::CallMethod => {
                    if let Some(name) = self.names.get(instr.src2 as usize) {
                        out.push_str(&format!("  method=\"{}\"", name));
                    }
                }
                RegOpCode::Jump | RegOpCode::JumpIfFalse | RegOpCode::JumpIfTrue => {
                    out.push_str(&format!("  -> {:04}", instr.imm));
                }
                _ => {}
            }
            out.push('\n');
        }
        out
    }
}

/// Local register metadata for name sync.
#[derive(Debug, Clone)]
pub struct RegLocal {
    pub name_idx: u32,
    pub reg: u32,
}
