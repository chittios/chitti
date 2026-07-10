//! Bytecode instruction set and chunk structure for the JIT compiler.
//!
//! Defines a flat, stack-based bytecode IR that the compiler emits
//! and the VM executes.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::value::JsValue;

/// Bytecode opcodes for the stack-based VM.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum OpCode {
    // ── Constants & Literals ──────────────────────────────────
    /// Push a constant from the constant pool onto the stack.
    Constant,
    /// Push `undefined` onto the stack.
    Undefined,
    /// Push `null` onto the stack.
    Null,
    /// Push `true` onto the stack.
    True,
    /// Push `false` onto the stack.
    False,

    // ── Arithmetic ───────────────────────────────────────────
    /// Pop two values, push their sum.
    Add,
    /// Pop two values, push their difference.
    Sub,
    /// Pop two values, push their product.
    Mul,
    /// Pop two values, push their quotient.
    Div,
    /// Pop two values, push their remainder.
    Mod,
    /// Pop one value, push its numeric negation.
    Negate,

    // ── Bitwise ──────────────────────────────────────────────
    /// Bitwise AND.
    BitAnd,
    /// Bitwise OR.
    BitOr,
    /// Bitwise XOR.
    BitXor,
    /// Bitwise NOT.
    BitNot,
    /// Left shift.
    ShiftLeft,
    /// Signed right shift.
    ShiftRight,
    /// Unsigned right shift.
    UShiftRight,

    // ── Comparison ───────────────────────────────────────────
    /// Strict equality (`===`).
    StrictEqual,
    /// Strict inequality (`!==`).
    StrictNotEqual,
    /// Abstract equality (`==`).
    Equal,
    /// Abstract inequality (`!=`).
    NotEqual,
    /// Less than.
    LessThan,
    /// Less than or equal.
    LessEqual,
    /// Greater than.
    GreaterThan,
    /// Greater than or equal.
    GreaterEqual,

    // ── Logical / Unary ──────────────────────────────────────
    /// Logical NOT.
    Not,
    /// `typeof` operator — pops value, pushes type string.
    TypeOf,
    /// `void` operator — pops value, pushes undefined.
    Void,
    /// Unary `+` — converts to number.
    UnaryPlus,

    // ── Variables ────────────────────────────────────────────
    /// Get a global/local variable by name (operand: constant pool index of name string).
    GetVar,
    /// Set a variable by name (operand: constant pool index of name string).
    /// The value to assign is on top of the stack.
    SetVar,
    /// Declare a `var` binding (operand: constant pool index of name string).
    DeclareVar,
    /// Declare a `let` binding (operand: constant pool index of name string).
    DeclareLet,
    /// Declare a `const` binding (operand: constant pool index of name string).
    DeclareConst,
    /// Initialize a binding with the value on top of the stack.
    InitVar,
    /// Initialize a let/const binding with the value on top of the stack.
    InitBinding,

    /// Get a local slot by index (operand: local slot index).
    GetLocal,
    /// Set a local slot by index (operand: local slot index).
    SetLocal,
    /// Initialize a local slot (operand: local slot index).
    InitLocal,

    // ── Control Flow ─────────────────────────────────────────
    /// Unconditional jump (operand: absolute offset).
    Jump,
    /// Jump if top of stack is falsy (operand: absolute offset). Pops the value.
    JumpIfFalse,
    /// Jump if top of stack is truthy (operand: absolute offset). Pops the value.
    JumpIfTrue,

    // ── Loops ────────────────────────────────────────────────
    /// Marker for loop start (used by break/continue resolution).
    LoopStart,

    // ── Stack manipulation ───────────────────────────────────
    /// Pop and discard the top of the stack.
    Pop,
    /// Duplicate the top of the stack.
    Dup,
    /// Duplicate the top two stack values (a,b -> a,b,a,b).
    Dup2,

    // ── Scope ────────────────────────────────────────────────
    /// Push a new block scope.
    PushScope,
    /// Pop a block scope.
    PopScope,

    // ── Objects & Properties ─────────────────────────────────
    /// Get a property: pop object and push object.property.
    /// Operand: constant pool index of property name.
    GetProp,
    /// Set a property: stack has [value, object].
    /// Operand: constant pool index of property name.
    SetProp,
    /// Get a computed property: stack has [key, object].
    GetElem,
    /// Set a computed property: stack has [value, key, object].
    SetElem,

    // ── Function calls ───────────────────────────────────────
    /// Call a function. Operand: argument count.
    /// Stack: [arg_n, ..., arg_1, callee]
    Call,
    /// Call a method. Operand: argument count.
    /// Stack: [arg_n, ..., arg_1, object]
    /// Second operand: constant pool index of method name.
    CallMethod,

    // ── Misc ─────────────────────────────────────────────────
    /// Return from the current function/script. Pops return value from stack.
    Return,
    /// Halt execution (end of script).
    Halt,

    // ── Pre/Post increment/decrement ─────────────────────────
    /// Pre-increment a variable (++x). Operand: constant pool index of name.
    PreIncVar,
    /// Pre-decrement a variable (--x). Operand: constant pool index of name.
    PreDecVar,
    /// Post-increment a variable (x++). Operand: constant pool index of name.
    PostIncVar,
    /// Post-decrement a variable (x--). Operand: constant pool index of name.
    PostDecVar,

    // ── Compound assignment helpers ──────────────────────────
    /// Get a variable for compound update. Operand: constant pool index of name.
    GetVarForUpdate,

    // ── Closures ────────────────────────────────────────────
    /// Create a closure from a function template.
    /// Operand: index into the chunk's `functions` table.
    /// Pushes the resulting function object onto the stack.
    MakeClosure,
}

/// A single bytecode instruction with optional operands.
#[derive(Debug, Clone)]
pub struct Instruction {
    pub op: OpCode,
    /// Primary operand (constant pool index, jump offset, arg count, etc.).
    pub operand: u32,
    /// Secondary operand (used by CallMethod for method name index).
    pub operand2: u32,
    /// Tertiary operand (used by CallMethod for object name index).
    pub operand3: u32,
}

impl Instruction {
    pub fn simple(op: OpCode) -> Self {
        Instruction { op, operand: 0, operand2: 0, operand3: 0 }
    }

    pub fn with_operand(op: OpCode, operand: u32) -> Self {
        Instruction { op, operand, operand2: 0, operand3: 0 }
    }

    pub fn with_two_operands(op: OpCode, operand: u32, operand2: u32) -> Self {
        Instruction { op, operand, operand2, operand3: 0 }
    }

    pub fn with_three_operands(op: OpCode, operand: u32, operand2: u32, operand3: u32) -> Self {
        Instruction { op, operand, operand2, operand3 }
    }
}

/// A stored function template — raw pointers to AST nodes captured at compile time.
#[derive(Debug, Clone, Copy)]
pub struct FunctionTemplate {
    /// Pointer to the function body AST node.
    pub body_ptr: *const crate::parser::ast::FunctionBodyData,
    /// Pointer to the formal parameter list.
    pub params_ptr: *const Vec<crate::parser::ast::PatternType>,
}

// Safety: FunctionTemplate is only used within a single thread.
unsafe impl Send for FunctionTemplate {}
unsafe impl Sync for FunctionTemplate {}

/// A compiled chunk of bytecode with its constant pool.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// The bytecode instructions.
    pub code: Vec<Instruction>,
    /// Constant pool — holds literal values.
    pub constants: Vec<JsValue>,
    /// Deduplicated name table for variable/property names.
    /// Indexed by operand in GetVar/SetVar/etc. opcodes.
    /// Separate from constants to allow zero-copy `&str` access in the VM.
    pub names: Vec<String>,
    /// Local slots table. Each entry stores an index into `names`.
    /// Slot index is used by GetLocal/SetLocal/InitLocal opcodes.
    pub locals: Vec<u32>,
    /// Function templates for MakeClosure.
    pub functions: Vec<FunctionTemplate>,
}

impl Chunk {
    pub fn new() -> Self {
        Chunk {
            code: Vec::new(),
            constants: Vec::new(),
            names: Vec::new(),
            locals: Vec::new(),
            functions: Vec::new(),
        }
    }

    /// Add a function template and return its index.
    pub fn add_function(&mut self, template: FunctionTemplate) -> u32 {
        let idx = self.functions.len();
        self.functions.push(template);
        idx as u32
    }

    /// Emit an instruction and return its index.
    pub fn emit(&mut self, instr: Instruction) -> usize {
        let idx = self.code.len();
        self.code.push(instr);
        idx
    }

    /// Emit a simple (no-operand) instruction.
    pub fn emit_op(&mut self, op: OpCode) -> usize {
        self.emit(Instruction::simple(op))
    }

    /// Emit an instruction with one operand.
    pub fn emit_with(&mut self, op: OpCode, operand: u32) -> usize {
        self.emit(Instruction::with_operand(op, operand))
    }

    /// Add a constant to the pool and return its index.
    pub fn add_constant(&mut self, value: JsValue) -> u32 {
        let idx = self.constants.len();
        self.constants.push(value);
        idx as u32
    }

    /// Add a name to the deduplicated name table and return its index.
    /// Used for variable names, property names — allows zero-copy `&str` access in the VM.
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

    /// Add a local slot for a name index and return its slot index.
    pub fn add_local(&mut self, name_idx: u32) -> u32 {
        let slot = self.locals.len();
        self.locals.push(name_idx);
        slot as u32
    }

    /// Get a local slot's name (zero-copy reference).
    #[inline]
    pub fn get_local_name(&self, slot: u32) -> &str {
        let name_idx = self.locals[slot as usize];
        self.get_name(name_idx)
    }

    /// Get a name by index (zero-copy reference).
    #[inline]
    pub fn get_name(&self, idx: u32) -> &str {
        &self.names[idx as usize]
    }

    /// Patch a jump instruction's operand to point to the current code position.
    pub fn patch_jump(&mut self, jump_idx: usize) {
        self.code[jump_idx].operand = self.code.len() as u32;
    }

    /// Get the current code position (for jump targets).
    pub fn current_pos(&self) -> usize {
        self.code.len()
    }

    /// Disassemble the chunk for debugging.
    pub fn disassemble(&self, name: &str) -> String {
        let mut out = format!("== {} ==\n", name);
        for (i, instr) in self.code.iter().enumerate() {
            out.push_str(&format!("{:04}  {:?}", i, instr.op));
            match instr.op {
                OpCode::Constant => {
                    let val = &self.constants[instr.operand as usize];
                    out.push_str(&format!("  {} ({})", instr.operand, val));
                }
                OpCode::GetVar | OpCode::SetVar | OpCode::DeclareVar
                | OpCode::DeclareLet | OpCode::DeclareConst
                | OpCode::InitVar | OpCode::InitBinding
                | OpCode::GetProp | OpCode::SetProp
                | OpCode::PreIncVar | OpCode::PreDecVar
                | OpCode::PostIncVar | OpCode::PostDecVar
                | OpCode::GetVarForUpdate => {
                    let name_str = self.get_name(instr.operand);
                    out.push_str(&format!("  \"{}\"", name_str));
                }
                OpCode::GetLocal | OpCode::SetLocal | OpCode::InitLocal => {
                    let name_str = self.get_local_name(instr.operand);
                    out.push_str(&format!("  slot={} \"{}\"", instr.operand, name_str));
                }
                OpCode::Jump | OpCode::JumpIfFalse | OpCode::JumpIfTrue => {
                    out.push_str(&format!("  -> {:04}", instr.operand));
                }
                OpCode::Call => {
                    out.push_str(&format!("  argc={}", instr.operand));
                }
                OpCode::MakeClosure => {
                    out.push_str(&format!("  func_idx={}", instr.operand));
                }
                OpCode::CallMethod => {
                    let method = self.get_name(instr.operand2);
                    let obj_name = self.get_name(instr.operand3);
                    out.push_str(&format!("  argc={} method=\"{}\" obj=\"{}\"", instr.operand, method, obj_name));
                }
                _ => {}
            }
            out.push('\n');
        }
        out
    }
}
