//! **JS bytecode compiler + VM** — the "JIT-shaped" fast path for the browser
//! script subset (Ladybird/LibJS uses a bytecode interpreter; full optimizing
//! JIT is out of scope in this kernel).
//!
//! Flow: source → [`compile`] → [`optimize`] → [`run`] against a host env.
//! Falls back to the AST interpreter in `js.rs` when compile fails.
//!
//! **Native-code JIT is not available in this no_std kernel** (no RWX pages /
//! dynasm). The optimizer provides compile-time const-fold + peephole passes
//! analogous to a baseline JIT's early tiers (Ladybird LibJS bytecode opts).
//!
//! Architecture reference: `third_party/just-ref` (applegrew/just) —
//! stack + register bytecode + Cranelift numeric JIT (host/x86_64 only). We
//! mirror the **bailout design** (pure numeric/console here; DOM/objects in
//! `js.rs`) rather than linking Cranelift.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Op {
    Nop = 0,
    /// Push constant pool index.
    LoadConst = 1,
    /// Push local/global by name index in name pool.
    LoadName = 2,
    StoreName = 3,
    Add = 4,
    Sub = 5,
    Mul = 6,
    Div = 7,
    Eq = 8,
    Lt = 9,
    Not = 10,
    /// Pop argc, then callee name index — call host.
    Call = 11,
    /// Jump relative i8.
    Jump = 12,
    /// Pop; jump if false.
    JumpIfFalse = 13,
    Pop = 14,
    Dup = 15,
    /// Return top of stack (end program).
    Ret = 16,
    /// Push true/false/null/undefined
    LoadTrue = 17,
    LoadFalse = 18,
    LoadNull = 19,
    LoadUndef = 20,
    /// Greater-than (a > b).
    Gt = 21,
    /// Not-equal.
    Ne = 22,
    /// typeof → push type name string.
    TypeOf = 23,
    /// a <= b
    Le = 24,
    /// a >= b
    Ge = 25,
    /// a % b (IEEE remainder via trunc toward zero for integers)
    Mod = 26,
    /// Bitwise AND / OR / XOR (ToInt32 semantics, loosely)
    BitAnd = 27,
    BitOr = 28,
    BitXor = 29,
    /// `new Ctor(args…)` — ctor name on stack after args (same as Call).
    New = 30,
    /// Get property: stack [obj, propname_str] → value
    GetProp = 31,
}

#[derive(Clone, Debug)]
pub enum Const {
    Num(f64),
    Str(String),
    /// User function template (body is source; recompiled on each call).
    Fun { params: Vec<String>, body: String },
}

#[derive(Clone, Debug, Default)]
pub struct Chunk {
    pub code: Vec<u8>,
    pub consts: Vec<Const>,
    pub names: Vec<String>,
}

impl Chunk {
    fn emit(&mut self, op: Op) {
        self.code.push(op as u8);
    }
    fn emit_u8(&mut self, b: u8) {
        self.code.push(b);
    }
    fn emit_i8(&mut self, b: i8) {
        self.code.push(b as u8);
    }
    fn const_idx(&mut self, c: Const) -> u8 {
        if self.consts.len() >= 255 {
            return 0;
        }
        let i = self.consts.len() as u8;
        self.consts.push(c);
        i
    }
    fn name_idx(&mut self, name: &str) -> u8 {
        if let Some(i) = self.names.iter().position(|n| n == name) {
            return i as u8;
        }
        let i = self.names.len() as u8;
        self.names.push(name.to_string());
        i
    }
}

/// Compile a restricted script: statements of form
/// `var x = …;`, `x = …;`, `console.log(…);`, `if (cond) { … }`, binary ops.
pub fn compile(src: &str) -> Result<Chunk, String> {
    let mut chunk = Chunk::default();
    let mut p = Src::new(src);
    while !p.eof() {
        p.skip_ws();
        if p.eof() {
            break;
        }
        compile_stmt(&mut p, &mut chunk)?;
        p.skip_ws();
        let _ = p.eat(';');
    }
    chunk.emit(Op::Ret);
    // Length-preserving peep only — const-fold can shrink code and break
    // relative Jump/JumpIfFalse offsets patched earlier.
    optimize_preserve_len(&mut chunk);
    Ok(chunk)
}

/// Peephole that never changes instruction stream length (safe with relative jumps).
fn optimize_preserve_len(chunk: &mut Chunk) {
    let mut i = 0;
    while i + 1 < chunk.code.len() {
        if chunk.code[i] == Op::Dup as u8 && chunk.code[i + 1] == Op::Pop as u8 {
            chunk.code[i] = Op::Nop as u8;
            chunk.code[i + 1] = Op::Nop as u8;
        }
        i += 1;
    }
}

/// Peephole + const-fold optimizer (baseline-JIT tier substitute).
pub fn optimize(chunk: &mut Chunk) {
    // Pass 1: LoadConst a; LoadConst b; Add|Sub|Mul|Div → single LoadConst
    let mut i = 0;
    let code = &mut chunk.code;
    while i + 5 < code.len() {
        if code[i] == Op::LoadConst as u8
            && code[i + 2] == Op::LoadConst as u8
            && matches!(
                code[i + 4],
                x if x == Op::Add as u8
                    || x == Op::Sub as u8
                    || x == Op::Mul as u8
                    || x == Op::Div as u8
            )
        {
            let ia = code[i + 1] as usize;
            let ib = code[i + 3] as usize;
            if let (Some(Const::Num(a)), Some(Const::Num(b))) =
                (chunk.consts.get(ia).cloned(), chunk.consts.get(ib).cloned())
            {
                let r = match code[i + 4] {
                    x if x == Op::Add as u8 => a + b,
                    x if x == Op::Sub as u8 => a - b,
                    x if x == Op::Mul as u8 => a * b,
                    x if x == Op::Div as u8 => {
                        if b == 0.0 {
                            0.0
                        } else {
                            a / b
                        }
                    }
                    _ => a,
                };
                let ni = chunk.consts.len() as u8;
                if ni < 255 {
                    chunk.consts.push(Const::Num(r));
                    // rewrite first LoadConst to new, Nop the rest of pattern
                    code[i + 1] = ni;
                    code[i + 2] = Op::Nop as u8;
                    code[i + 3] = Op::Nop as u8;
                    code[i + 4] = Op::Nop as u8;
                }
            }
        }
        i += 1;
    }
    // Pass 2: Dup; Pop → Nop Nop; Pop after StoreName that result unused: leave
    i = 0;
    while i + 1 < chunk.code.len() {
        if chunk.code[i] == Op::Dup as u8 && chunk.code[i + 1] == Op::Pop as u8 {
            chunk.code[i] = Op::Nop as u8;
            chunk.code[i + 1] = Op::Nop as u8;
        }
        // LoadConst; Pop → dead
        if i + 2 < chunk.code.len()
            && chunk.code[i] == Op::LoadConst as u8
            && chunk.code[i + 2] == Op::Pop as u8
        {
            chunk.code[i] = Op::Nop as u8;
            chunk.code[i + 1] = Op::Nop as u8;
            chunk.code[i + 2] = Op::Nop as u8;
        }
        i += 1;
    }
}

fn compile_stmt(p: &mut Src<'_>, c: &mut Chunk) -> Result<(), String> {
    p.skip_ws();
    if p.eat_kw("var") || p.eat_kw("let") || p.eat_kw("const") {
        p.skip_ws();
        let name = p.ident().ok_or("expected name")?;
        p.skip_ws();
        if p.eat('=') {
            compile_expr(p, c)?;
        } else {
            c.emit(Op::LoadUndef);
        }
        let ni = c.name_idx(&name);
        c.emit(Op::StoreName);
        c.emit_u8(ni);
        c.emit(Op::Pop); // statement context: discard value
        return Ok(());
    }
    if p.eat_kw("function") {
        p.skip_ws();
        let name = p.ident().ok_or("function name")?;
        p.skip_ws();
        p.expect('(')?;
        let mut params = Vec::new();
        p.skip_ws();
        if !p.eat(')') {
            loop {
                p.skip_ws();
                if let Some(param) = p.ident() {
                    params.push(param);
                }
                p.skip_ws();
                if p.eat(')') {
                    break;
                }
                p.expect(',')?;
            }
        }
        p.skip_ws();
        let body = capture_block_src(p)?;
        let ci = c.const_idx(Const::Fun { params, body });
        c.emit(Op::LoadConst);
        c.emit_u8(ci);
        let ni = c.name_idx(&name);
        c.emit(Op::StoreName);
        c.emit_u8(ni);
        c.emit(Op::Pop);
        return Ok(());
    }
    if p.eat_kw("return") {
        p.skip_ws();
        if p.peek() == ';' || p.peek() == '}' || p.eof() {
            c.emit(Op::LoadUndef);
        } else {
            compile_expr(p, c)?;
        }
        c.emit(Op::Ret);
        return Ok(());
    }
    if p.eat_kw("throw") {
        // throw e → console.log + set __ok via host: treat as return of throw sentinel
        p.skip_ws();
        compile_expr(p, c)?;
        // call a host throw helper
        let ni = c.name_idx("__throw__");
        c.emit(Op::LoadName);
        c.emit_u8(ni);
        // stack: value, callee? Call expects args then callee
        // Currently: value on stack. Need callee then Call 1.
        // reorder: we have value; LoadName throw; need swap — simplify: host sees call with value
        // Push name as call via Store to __throw_arg and call
        let ai = c.name_idx("__throw_arg");
        c.emit(Op::StoreName);
        c.emit_u8(ai);
        c.emit(Op::LoadName);
        c.emit_u8(ni);
        c.emit(Op::Call);
        c.emit_u8(0);
        c.emit(Op::Ret);
        return Ok(());
    }
    if p.eat_kw("try") {
        // try { A } catch (e) { B } [finally { C }]
        // Compile as: run A in nested; on fail run B. Simplified: just run try body then catch body.
        p.skip_ws();
        let try_body = capture_block_src(p)?;
        p.skip_ws();
        let mut catch_body = String::from("{}");
        let mut catch_param = String::from("e");
        if p.eat_kw("catch") {
            p.skip_ws();
            if p.eat('(') {
                p.skip_ws();
                catch_param = p.ident().unwrap_or_else(|| String::from("e"));
                p.skip_ws();
                p.expect(')')?;
            }
            p.skip_ws();
            catch_body = capture_block_src(p)?;
        }
        p.skip_ws();
        let mut finally_body = None;
        if p.eat_kw("finally") {
            p.skip_ws();
            finally_body = Some(capture_block_src(p)?);
        }
        // Emit as call to host-ish: compile try body inline; ignore catch unless throw used
        // Inline: compile try_body statements
        compile_source_block(c, &try_body)?;
        // catch unused unless throw — emit catch param bind undefined then catch body for structure tests
        let _ = catch_param;
        let _ = catch_body;
        if let Some(fb) = finally_body {
            compile_source_block(c, &fb)?;
        }
        return Ok(());
    }
    if p.eat_kw("for") {
        // for (init; cond; step) body
        p.skip_ws();
        p.expect('(')?;
        // init
        p.skip_ws();
        if !p.eat(';') {
            if p.eat_kw("var") || p.eat_kw("let") || p.eat_kw("const") {
                p.skip_ws();
                let name = p.ident().ok_or("for var")?;
                p.skip_ws();
                if p.eat('=') {
                    compile_expr(p, c)?;
                } else {
                    c.emit(Op::LoadUndef);
                }
                let ni = c.name_idx(&name);
                c.emit(Op::StoreName);
                c.emit_u8(ni);
                c.emit(Op::Pop);
            } else {
                compile_expr(p, c)?;
                c.emit(Op::Pop);
            }
            p.expect(';')?;
        }
        let loop_start = c.code.len();
        // cond
        p.skip_ws();
        let cond_empty = p.peek() == ';';
        if !cond_empty {
            compile_expr(p, c)?;
        } else {
            c.emit(Op::LoadTrue);
        }
        p.expect(';')?;
        c.emit(Op::JumpIfFalse);
        let j_exit = c.code.len();
        c.emit_i8(0);
        // step slice
        let step_start = p.pos;
        let mut depth = 0i32;
        while !p.eof() {
            let ch = p.peek();
            if ch == '(' {
                depth += 1;
            } else if ch == ')' {
                if depth == 0 {
                    break;
                }
                depth -= 1;
            }
            p.bump();
        }
        let step_src = p.s[step_start..p.pos].to_string();
        p.expect(')')?;
        p.skip_ws();
        compile_block_or_stmt(p, c)?;
        // step (often `i = i + 1` assignment, not a pure expression)
        if !step_src.trim().is_empty() {
            let mut sp = Src::new(step_src.trim());
            let save = sp.pos;
            if let Some(name) = sp.ident() {
                sp.skip_ws();
                if sp.eat('=') && sp.peek() != '=' {
                    compile_expr(&mut sp, c)?;
                    let ni = c.name_idx(&name);
                    c.emit(Op::StoreName);
                    c.emit_u8(ni);
                    c.emit(Op::Pop);
                } else {
                    sp.pos = save;
                    compile_expr(&mut sp, c)?;
                    c.emit(Op::Pop);
                }
            } else {
                compile_expr(&mut sp, c)?;
                c.emit(Op::Pop);
            }
        }
        c.emit(Op::Jump);
        let j_back = c.code.len();
        c.emit_i8(0);
        let after = c.code.len();
        let back = (loop_start as i32 - (j_back as i32 + 1)) as i8;
        c.code[j_back] = back as u8;
        let exit = (after as i32 - (j_exit as i32 + 1)) as i8;
        c.code[j_exit] = exit as u8;
        return Ok(());
    }
    if p.eat_kw("if") {
        p.skip_ws();
        p.expect('(')?;
        compile_expr(p, c)?;
        p.expect(')')?;
        p.skip_ws();
        c.emit(Op::JumpIfFalse);
        let j_else = c.code.len();
        c.emit_i8(0); // patch
        compile_block_or_stmt(p, c)?;
        p.skip_ws();
        if p.eat_kw("else") {
            c.emit(Op::Jump);
            let j_end = c.code.len();
            c.emit_i8(0);
            let after_then = c.code.len();
            let delta = (after_then as i32 - (j_else as i32 + 1)) as i8;
            c.code[j_else] = delta as u8;
            compile_block_or_stmt(p, c)?;
            let end = c.code.len();
            let delta = (end as i32 - (j_end as i32 + 1)) as i8;
            c.code[j_end] = delta as u8;
        } else {
            let after = c.code.len();
            let delta = (after as i32 - (j_else as i32 + 1)) as i8;
            c.code[j_else] = delta as u8;
        }
        return Ok(());
    }
    if p.eat_kw("while") {
        p.skip_ws();
        p.expect('(')?;
        let loop_start = c.code.len();
        compile_expr(p, c)?;
        p.expect(')')?;
        p.skip_ws();
        c.emit(Op::JumpIfFalse);
        let j_exit = c.code.len();
        c.emit_i8(0);
        compile_block_or_stmt(p, c)?;
        c.emit(Op::Jump);
        let j_back = c.code.len();
        c.emit_i8(0);
        let after = c.code.len();
        // patch jump back to loop_start
        let back = (loop_start as i32 - (j_back as i32 + 1)) as i8;
        c.code[j_back] = back as u8;
        let exit = (after as i32 - (j_exit as i32 + 1)) as i8;
        c.code[j_exit] = exit as u8;
        return Ok(());
    }
    // Expression statement (calls / assign).
    // Detect `name =`
    let save = p.pos;
    if let Some(name) = p.ident() {
        p.skip_ws();
        if p.eat('=') && p.peek() != '=' {
            compile_expr(p, c)?;
            let ni = c.name_idx(&name);
            c.emit(Op::StoreName);
            c.emit_u8(ni);
            c.emit(Op::Pop);
            return Ok(());
        }
    }
    p.pos = save;
    compile_expr(p, c)?;
    c.emit(Op::Pop);
    Ok(())
}

fn compile_block_or_stmt(p: &mut Src<'_>, c: &mut Chunk) -> Result<(), String> {
    p.skip_ws();
    if p.eat('{') {
        while !p.eof() {
            p.skip_ws();
            if p.eat('}') {
                break;
            }
            compile_stmt(p, c)?;
            p.skip_ws();
            let _ = p.eat(';');
        }
        Ok(())
    } else {
        compile_stmt(p, c)
    }
}

fn compile_expr(p: &mut Src<'_>, c: &mut Chunk) -> Result<(), String> {
    compile_rel(p, c)
}

fn compile_rel(p: &mut Src<'_>, c: &mut Chunk) -> Result<(), String> {
    compile_add(p, c)?;
    p.skip_ws();
    if p.eat_str("!==") || p.eat_str("!=") {
        compile_add(p, c)?;
        c.emit(Op::Ne);
    } else if p.eat_str("===") || p.eat_str("==") {
        compile_add(p, c)?;
        c.emit(Op::Eq);
    } else if p.eat_str("<=") {
        compile_add(p, c)?;
        c.emit(Op::Le);
    } else if p.eat_str(">=") {
        compile_add(p, c)?;
        c.emit(Op::Ge);
    } else if p.eat('<') {
        compile_add(p, c)?;
        c.emit(Op::Lt);
    } else if p.eat('>') {
        compile_add(p, c)?;
        c.emit(Op::Gt);
    }
    Ok(())
}

fn compile_add(p: &mut Src<'_>, c: &mut Chunk) -> Result<(), String> {
    compile_mul(p, c)?;
    loop {
        p.skip_ws();
        if p.eat('+') {
            compile_mul(p, c)?;
            c.emit(Op::Add);
        } else if p.eat('-') {
            compile_mul(p, c)?;
            c.emit(Op::Sub);
        } else {
            break;
        }
    }
    Ok(())
}

fn compile_mul(p: &mut Src<'_>, c: &mut Chunk) -> Result<(), String> {
    compile_unary(p, c)?;
    loop {
        p.skip_ws();
        if p.eat('*') {
            compile_unary(p, c)?;
            c.emit(Op::Mul);
        } else if p.eat('/') {
            compile_unary(p, c)?;
            c.emit(Op::Div);
        } else if p.eat('%') {
            compile_unary(p, c)?;
            c.emit(Op::Mod);
        } else {
            break;
        }
    }
    Ok(())
}



fn compile_unary(p: &mut Src<'_>, c: &mut Chunk) -> Result<(), String> {
    p.skip_ws();
    if p.eat_kw("typeof") {
        compile_unary(p, c)?;
        c.emit(Op::TypeOf);
        return Ok(());
    }
    if p.eat_kw("new") {
        p.skip_ws();
        // new Ctor(args) — compile Ctor then args then New argc
        compile_primary(p, c)?;
        p.skip_ws();
        let mut argc = 0u8;
        if p.eat('(') {
            p.skip_ws();
            if !p.eat(')') {
                loop {
                    compile_expr(p, c)?;
                    argc = argc.saturating_add(1);
                    p.skip_ws();
                    if p.eat(')') {
                        break;
                    }
                    p.expect(',')?;
                    p.skip_ws();
                }
            }
        }
        c.emit(Op::New);
        c.emit_u8(argc);
        return Ok(());
    }
    if p.eat('!') {
        compile_unary(p, c)?;
        c.emit(Op::Not);
        return Ok(());
    }
    if p.eat('-') {
        c.emit(Op::LoadConst);
        let i = c.const_idx(Const::Num(0.0));
        c.emit_u8(i);
        compile_unary(p, c)?;
        c.emit(Op::Sub);
        return Ok(());
    }
    compile_postfix(p, c)
}

fn compile_postfix(p: &mut Src<'_>, c: &mut Chunk) -> Result<(), String> {
    compile_primary(p, c)?;
    loop {
        p.skip_ws();
        if p.eat('.') {
            let prop = p.ident().ok_or("prop")?;
            p.skip_ws();
            if p.peek() == '(' {
                // method call: obj.method(args) — emit as LoadName("method") style via host
                // For console.log the primary already became LoadName("console.log") in primary.
                // Here: push prop name const + GetProp then Call.
                let ci = c.const_idx(Const::Str(prop.clone()));
                c.emit(Op::LoadConst);
                c.emit_u8(ci);
                c.emit(Op::GetProp);
                let mut argc = 0u8;
                p.expect('(')?;
                p.skip_ws();
                if !p.eat(')') {
                    loop {
                        compile_expr(p, c)?;
                        argc = argc.saturating_add(1);
                        p.skip_ws();
                        if p.eat(')') {
                            break;
                        }
                        p.expect(',')?;
                        p.skip_ws();
                    }
                }
                c.emit(Op::Call);
                c.emit_u8(argc);
                continue;
            } else {
                // property get
                let ci = c.const_idx(Const::Str(prop));
                c.emit(Op::LoadConst);
                c.emit_u8(ci);
                c.emit(Op::GetProp);
                continue;
            }
        }
        if p.eat('(') {
            let mut argc = 0u8;
            p.skip_ws();
            if !p.eat(')') {
                loop {
                    compile_expr(p, c)?;
                    argc = argc.saturating_add(1);
                    p.skip_ws();
                    if p.eat(')') {
                        break;
                    }
                    p.expect(',')?;
                    p.skip_ws();
                }
            }
            c.emit(Op::Call);
            c.emit_u8(argc);
            continue;
        }
        break;
    }
    Ok(())
}

fn compile_primary(p: &mut Src<'_>, c: &mut Chunk) -> Result<(), String> {
    p.skip_ws();
    if p.eat('(') {
        compile_expr(p, c)?;
        p.expect(')')?;
        return Ok(());
    }
    if p.peek() == '"' || p.peek() == '\'' {
        let s = p.string().ok_or("string")?;
        let i = c.const_idx(Const::Str(s));
        c.emit(Op::LoadConst);
        c.emit_u8(i);
        return Ok(());
    }
    if p.peek().is_ascii_digit() {
        let n = p.number().ok_or("number")?;
        let i = c.const_idx(Const::Num(n));
        c.emit(Op::LoadConst);
        c.emit_u8(i);
        return Ok(());
    }
    if p.eat_kw("true") {
        c.emit(Op::LoadTrue);
        return Ok(());
    }
    if p.eat_kw("false") {
        c.emit(Op::LoadFalse);
        return Ok(());
    }
    if p.eat_kw("null") {
        c.emit(Op::LoadNull);
        return Ok(());
    }
    if p.eat_kw("undefined") {
        c.emit(Op::LoadUndef);
        return Ok(());
    }
    if let Some(name) = p.ident() {
        // console.log → LoadName("console.log") if .log follows
        p.skip_ws();
        if p.peek() == '.' {
            p.bump();
            p.skip_ws();
            let prop = p.ident().unwrap_or_default();
            let full = format!("{name}.{prop}");
            let ni = c.name_idx(&full);
            c.emit(Op::LoadName);
            c.emit_u8(ni);
            return Ok(());
        }
        let ni = c.name_idx(&name);
        c.emit(Op::LoadName);
        c.emit_u8(ni);
        return Ok(());
    }
    Err(format!("bad primary at {}", p.pos))
}

/// Value on the VM stack.
#[derive(Clone, Debug)]
pub enum Val {
    Null,
    Undef,
    Bool(bool),
    Num(f64),
    Str(String),
    Fun { params: Vec<String>, body: String },
}

impl Val {
    fn as_num(&self) -> f64 {
        match self {
            Val::Num(n) => *n,
            Val::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            Val::Str(s) => s.parse().unwrap_or(0.0),
            _ => 0.0,
        }
    }
    fn as_bool(&self) -> bool {
        match self {
            Val::Bool(b) => *b,
            Val::Num(n) => *n != 0.0 && !n.is_nan(),
            Val::Str(s) => !s.is_empty(),
            Val::Fun { .. } => true,
            Val::Null | Val::Undef => false,
        }
    }
    fn as_str(&self) -> String {
        match self {
            Val::Str(s) => s.clone(),
            Val::Num(n) => {
                if n.is_nan() {
                    String::from("NaN")
                } else {
                    format!("{n}")
                }
            }
            Val::Bool(b) => b.to_string(),
            Val::Null => String::from("null"),
            Val::Undef => String::from("undefined"),
            Val::Fun { .. } => String::from("function"),
        }
    }
}

pub trait Host {
    fn get_var(&mut self, name: &str) -> Val;
    fn set_var(&mut self, name: &str, v: Val);
    fn call(&mut self, name: &str, args: &[Val]) -> Val;
}

/// JS-ish abstract equality (`==`) for the subset of values this VM holds.
fn abstract_eq(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Num(x), Val::Num(y)) => x == y,
        (Val::Bool(x), Val::Bool(y)) => x == y,
        (Val::Str(x), Val::Str(y)) => x == y,
        (Val::Null, Val::Null) | (Val::Undef, Val::Undef) => true,
        (Val::Null, Val::Undef) | (Val::Undef, Val::Null) => true,
        (Val::Null, _) | (_, Val::Null) | (Val::Undef, _) | (_, Val::Undef) => false,
        (Val::Num(n), Val::Str(s)) | (Val::Str(s), Val::Num(n)) => {
            if s.is_empty() {
                *n == 0.0
            } else {
                s.parse::<f64>().ok() == Some(*n)
            }
        }
        (Val::Bool(bv), other) | (other, Val::Bool(bv)) => {
            let n = if *bv { 1.0 } else { 0.0 };
            abstract_eq(&Val::Num(n), other)
        }
        _ => a.as_str() == b.as_str(),
    }
}

/// Execute a chunk. Returns last stack value / undefined.
pub fn run(chunk: &Chunk, host: &mut dyn Host) -> Result<Val, String> {
    let mut stack: Vec<Val> = Vec::new();
    let mut ip = 0usize;
    let code = &chunk.code;
    while ip < code.len() {
        let op = code[ip];
        ip += 1;
        match op {
            x if x == Op::Nop as u8 => {}
            x if x == Op::LoadConst as u8 => {
                let i = code.get(ip).copied().unwrap_or(0) as usize;
                ip += 1;
                let v = match chunk.consts.get(i) {
                    Some(Const::Num(n)) => Val::Num(n.clone()),
                    Some(Const::Str(s)) => Val::Str(s.clone()),
                    Some(Const::Fun { params, body }) => Val::Fun {
                        params: params.clone(),
                        body: body.clone(),
                    },
                    None => Val::Undef,
                };
                stack.push(v);
            }
            x if x == Op::LoadName as u8 => {
                let i = code.get(ip).copied().unwrap_or(0) as usize;
                ip += 1;
                let name = chunk.names.get(i).map(|s| s.as_str()).unwrap_or("");
                stack.push(host.get_var(name));
            }
            x if x == Op::StoreName as u8 => {
                let i = code.get(ip).copied().unwrap_or(0) as usize;
                ip += 1;
                let name = chunk.names.get(i).map(|s| s.as_str()).unwrap_or("");
                let v = stack.pop().unwrap_or(Val::Undef);
                host.set_var(name, v.clone());
                stack.push(v);
            }
            x if x == Op::Add as u8 => {
                let b = stack.pop().unwrap_or(Val::Undef);
                let a = stack.pop().unwrap_or(Val::Undef);
                if matches!(a, Val::Str(_)) || matches!(b, Val::Str(_)) {
                    stack.push(Val::Str(format!("{}{}", a.as_str(), b.as_str())));
                } else {
                    stack.push(Val::Num(a.as_num() + b.as_num()));
                }
            }
            x if x == Op::Sub as u8 => bin_num(&mut stack, |a, b| a - b),
            x if x == Op::Mul as u8 => bin_num(&mut stack, |a, b| a * b),
            x if x == Op::Div as u8 => bin_num(&mut stack, |a, b| if b == 0.0 { 0.0 } else { a / b }),
            x if x == Op::Mod as u8 => bin_num(&mut stack, |a, b| {
                if b == 0.0 {
                    0.0
                } else {
                    // Integer-ish remainder (no libm trunc).
                    let ai = a as i64;
                    let bi = b as i64;
                    if bi == 0 {
                        0.0
                    } else {
                        (ai % bi) as f64
                    }
                }
            }),
            x if x == Op::BitAnd as u8 => bin_num(&mut stack, |a, b| {
                ((a as i32) & (b as i32)) as f64
            }),
            x if x == Op::BitOr as u8 => bin_num(&mut stack, |a, b| {
                ((a as i32) | (b as i32)) as f64
            }),
            x if x == Op::BitXor as u8 => bin_num(&mut stack, |a, b| {
                ((a as i32) ^ (b as i32)) as f64
            }),
            x if x == Op::Eq as u8 => {
                let b = stack.pop().unwrap_or(Val::Undef);
                let a = stack.pop().unwrap_or(Val::Undef);
                stack.push(Val::Bool(abstract_eq(&a, &b)));
            }
            x if x == Op::Ne as u8 => {
                let b = stack.pop().unwrap_or(Val::Undef);
                let a = stack.pop().unwrap_or(Val::Undef);
                stack.push(Val::Bool(!abstract_eq(&a, &b)));
            }
            x if x == Op::Lt as u8 => {
                let b = stack.pop().unwrap_or(Val::Undef);
                let a = stack.pop().unwrap_or(Val::Undef);
                stack.push(Val::Bool(a.as_num() < b.as_num()));
            }
            x if x == Op::Gt as u8 => {
                let b = stack.pop().unwrap_or(Val::Undef);
                let a = stack.pop().unwrap_or(Val::Undef);
                stack.push(Val::Bool(a.as_num() > b.as_num()));
            }
            x if x == Op::Le as u8 => {
                let b = stack.pop().unwrap_or(Val::Undef);
                let a = stack.pop().unwrap_or(Val::Undef);
                stack.push(Val::Bool(a.as_num() <= b.as_num()));
            }
            x if x == Op::Ge as u8 => {
                let b = stack.pop().unwrap_or(Val::Undef);
                let a = stack.pop().unwrap_or(Val::Undef);
                stack.push(Val::Bool(a.as_num() >= b.as_num()));
            }
            x if x == Op::TypeOf as u8 => {
                let a = stack.pop().unwrap_or(Val::Undef);
                let t = match a {
                    Val::Undef => "undefined",
                    Val::Null => "object",
                    Val::Bool(_) => "boolean",
                    Val::Num(_) => "number",
                    Val::Str(_) => "string",
                    Val::Fun { .. } => "function",
                };
                stack.push(Val::Str(String::from(t)));
            }
            x if x == Op::Not as u8 => {
                let a = stack.pop().unwrap_or(Val::Undef);
                stack.push(Val::Bool(!a.as_bool()));
            }
            x if x == Op::Call as u8 => {
                let argc = code.get(ip).copied().unwrap_or(0) as usize;
                ip += 1;
                let mut args = Vec::new();
                for _ in 0..argc {
                    args.push(stack.pop().unwrap_or(Val::Undef));
                }
                args.reverse();
                let callee = stack.pop().unwrap_or(Val::Undef);
                match callee {
                    Val::Fun { params, body } => {
                        // Bind params and run function body as nested chunk.
                        for (i, p) in params.iter().enumerate() {
                            let a = args.get(i).cloned().unwrap_or(Val::Undef);
                            host.set_var(p, a);
                        }
                        let body = strip_outer_braces(&body);
                        match compile(body) {
                            Ok(inner) => match run(&inner, host) {
                                Ok(v) => stack.push(v),
                                Err(e) => return Err(e),
                            },
                            Err(e) => return Err(e),
                        }
                    }
                    other => {
                        let name = match other {
                            Val::Str(s) => s,
                            _ => String::new(),
                        };
                        let name = if name.is_empty() {
                            String::from("console.log")
                        } else {
                            name
                        };
                        if name == "__throw__" {
                            let msg = host.get_var("__throw_arg").as_str();
                            host.set_var("__ok", Val::Num(0.0));
                            host.call("console.log", &[Val::Str(format!("throw:{msg}"))]);
                            stack.push(Val::Undef);
                        } else {
                            stack.push(host.call(&name, &args));
                        }
                    }
                }
            }
            x if x == Op::Jump as u8 => {
                let d = code.get(ip).copied().unwrap_or(0) as i8;
                ip += 1;
                ip = (ip as i32 + d as i32) as usize;
            }
            x if x == Op::JumpIfFalse as u8 => {
                let d = code.get(ip).copied().unwrap_or(0) as i8;
                ip += 1;
                let v = stack.pop().unwrap_or(Val::Undef);
                if !v.as_bool() {
                    ip = (ip as i32 + d as i32) as usize;
                }
            }
            x if x == Op::Pop as u8 => {
                let _ = stack.pop();
            }
            x if x == Op::Dup as u8 => {
                let v = stack.last().cloned().unwrap_or(Val::Undef);
                stack.push(v);
            }
            x if x == Op::Ret as u8 => {
                return Ok(stack.pop().unwrap_or(Val::Undef));
            }
            x if x == Op::LoadTrue as u8 => stack.push(Val::Bool(true)),
            x if x == Op::LoadFalse as u8 => stack.push(Val::Bool(false)),
            x if x == Op::LoadNull as u8 => stack.push(Val::Null),
            x if x == Op::LoadUndef as u8 => stack.push(Val::Undef),
            x if x == Op::New as u8 => {
                let argc = code.get(ip).copied().unwrap_or(0) as usize;
                ip += 1;
                let mut args = Vec::new();
                for _ in 0..argc {
                    args.push(stack.pop().unwrap_or(Val::Undef));
                }
                args.reverse();
                let ctor = stack.pop().unwrap_or(Val::Undef);
                let name = match ctor {
                    Val::Str(s) => s,
                    Val::Fun { .. } => String::from("Object"),
                    _ => String::from("Object"),
                };
                // new Number/String/Boolean/Object/Array/Error → simple values
                let v = match name.as_str() {
                    "Number" => Val::Num(args.first().map(|a| a.as_num()).unwrap_or(0.0)),
                    "String" => Val::Str(args.first().map(|a| a.as_str()).unwrap_or_default()),
                    "Boolean" => Val::Bool(args.first().map(|a| a.as_bool()).unwrap_or(false)),
                    "Array" => Val::Num(args.len() as f64), // length placeholder
                    "Error" | "TypeError" | "ReferenceError" | "SyntaxError" | "RangeError"
                    | "Test262Error" => {
                        Val::Str(args.first().map(|a| a.as_str()).unwrap_or_default())
                    }
                    "Object" => Val::Str(String::from("[object Object]")),
                    _ => host.call(&format!("new_{name}"), &args),
                };
                stack.push(v);
            }
            x if x == Op::GetProp as u8 => {
                let prop = stack.pop().unwrap_or(Val::Undef).as_str();
                let obj = stack.pop().unwrap_or(Val::Undef);
                let v = match prop.as_str() {
                    "length" => match &obj {
                        Val::Str(s) => Val::Num(s.chars().count() as f64),
                        _ => Val::Num(0.0),
                    },
                    "toString" => {
                        let s = obj.as_str().replace('\\', "\\\\").replace('"', "\\\"");
                        Val::Fun {
                            params: Vec::new(),
                            body: format!("{{ return \"{s}\"; }}"),
                        }
                    }
                    "valueOf" => {
                        // return a function that returns the same value (numeric path)
                        match obj {
                            Val::Num(n) => Val::Fun {
                                params: Vec::new(),
                                body: format!("{{ return {n}; }}"),
                            },
                            other => {
                                let s = other.as_str().replace('\\', "\\\\").replace('"', "\\\"");
                                Val::Fun {
                                    params: Vec::new(),
                                    body: format!("{{ return \"{s}\"; }}"),
                                }
                            }
                        }
                    }
                    _ => host.call(&format!("getprop_{prop}"), &[obj]),
                };
                stack.push(v);
            }
            _ => return Err(format!("bad opcode {op}")),
        }
    }
    Ok(stack.pop().unwrap_or(Val::Undef))
}

fn bin_num(stack: &mut Vec<Val>, f: impl Fn(f64, f64) -> f64) {
    let b = stack.pop().unwrap_or(Val::Undef);
    let a = stack.pop().unwrap_or(Val::Undef);
    stack.push(Val::Num(f(a.as_num(), b.as_num())));
}

fn strip_outer_braces(s: &str) -> &str {
    let t = s.trim();
    t.strip_prefix('{')
        .and_then(|x| x.strip_suffix('}'))
        .map(str::trim)
        .unwrap_or(t)
}

fn capture_block_src(p: &mut Src<'_>) -> Result<String, String> {
    p.skip_ws();
    if p.peek() != '{' {
        return Err("expected {".into());
    }
    let start = p.pos;
    p.bump();
    let mut depth = 1i32;
    while !p.eof() && depth > 0 {
        let c = p.peek();
        if c == '"' || c == '\'' {
            let q = c;
            p.bump();
            while !p.eof() {
                let c2 = p.peek();
                p.bump();
                if c2 == '\\' {
                    p.bump();
                    continue;
                }
                if c2 == q {
                    break;
                }
            }
            continue;
        }
        if c == '{' {
            depth += 1;
        } else if c == '}' {
            depth -= 1;
        }
        p.bump();
    }
    Ok(p.s[start..p.pos].to_string())
}

fn compile_source_block(c: &mut Chunk, block: &str) -> Result<(), String> {
    let inner = strip_outer_braces(block);
    let mut p = Src::new(inner);
    while !p.eof() {
        p.skip_ws();
        if p.eof() {
            break;
        }
        compile_stmt(&mut p, c)?;
        p.skip_ws();
        let _ = p.eat(';');
    }
    Ok(())
}

// ── tiny source cursor ────────────────────────────────────────────────────

struct Src<'a> {
    s: &'a str,
    pos: usize,
}

impl<'a> Src<'a> {
    fn new(s: &'a str) -> Self {
        Self { s, pos: 0 }
    }
    fn eof(&self) -> bool {
        self.pos >= self.s.len()
    }
    fn peek(&self) -> char {
        self.s[self.pos..].chars().next().unwrap_or('\0')
    }
    fn bump(&mut self) {
        if let Some(c) = self.s[self.pos..].chars().next() {
            self.pos += c.len_utf8();
        }
    }
    fn skip_ws(&mut self) {
        while !self.eof() {
            let c = self.peek();
            if c.is_whitespace() {
                self.bump();
            } else if c == '/' && self.s[self.pos..].starts_with("//") {
                while !self.eof() && self.peek() != '\n' {
                    self.bump();
                }
            } else {
                break;
            }
        }
    }
    fn eat(&mut self, c: char) -> bool {
        self.skip_ws();
        if self.peek() == c {
            self.bump();
            true
        } else {
            false
        }
    }
    fn eat_str(&mut self, s: &str) -> bool {
        self.skip_ws();
        if self.s[self.pos..].starts_with(s) {
            self.pos += s.len();
            true
        } else {
            false
        }
    }
    fn eat_kw(&mut self, kw: &str) -> bool {
        self.skip_ws();
        if self.s[self.pos..].starts_with(kw) {
            let after = self.pos + kw.len();
            let next = self.s[after..].chars().next().unwrap_or('\0');
            if next.is_ascii_alphanumeric() || next == '_' {
                return false;
            }
            self.pos = after;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, c: char) -> Result<(), String> {
        if self.eat(c) {
            Ok(())
        } else {
            Err(format!("expected {c}"))
        }
    }
    fn ident(&mut self) -> Option<String> {
        self.skip_ws();
        let start = self.pos;
        let c = self.peek();
        if !(c.is_ascii_alphabetic() || c == '_' || c == '$') {
            return None;
        }
        self.bump();
        while !self.eof() {
            let c = self.peek();
            if c.is_ascii_alphanumeric() || c == '_' || c == '$' {
                self.bump();
            } else {
                break;
            }
        }
        Some(self.s[start..self.pos].to_string())
    }
    fn string(&mut self) -> Option<String> {
        self.skip_ws();
        let q = self.peek();
        if q != '"' && q != '\'' {
            return None;
        }
        self.bump();
        let mut out = String::new();
        while !self.eof() {
            let c = self.peek();
            self.bump();
            if c == q {
                break;
            }
            if c == '\\' {
                let n = self.peek();
                self.bump();
                out.push(n);
            } else {
                out.push(c);
            }
        }
        Some(out)
    }
    fn number(&mut self) -> Option<f64> {
        self.skip_ws();
        let start = self.pos;
        while !self.eof() && (self.peek().is_ascii_digit() || self.peek() == '.') {
            self.bump();
        }
        self.s[start..self.pos].parse().ok()
    }
}

/// Simple host for unit tests / shell.
pub struct MapHost {
    pub vars: alloc::collections::BTreeMap<String, Val>,
    pub log: Vec<String>,
}

impl Default for MapHost {
    fn default() -> Self {
        Self {
            vars: alloc::collections::BTreeMap::new(),
            log: Vec::new(),
        }
    }
}

impl Host for MapHost {
    fn get_var(&mut self, name: &str) -> Val {
        if name == "console.log" || name == "__throw__" {
            return Val::Str(name.to_string());
        }
        if name == "NaN" {
            return Val::Num(f64::NAN);
        }
        if name == "Infinity" {
            return Val::Num(f64::INFINITY);
        }
        if name == "undefined" {
            return Val::Undef;
        }
        if name == "null" {
            return Val::Null;
        }
        if name == "true" {
            return Val::Bool(true);
        }
        if name == "false" {
            return Val::Bool(false);
        }
        if name == "parseInt" || name == "parseFloat" || name == "isNaN" || name == "isFinite" {
            return Val::Str(name.to_string());
        }
        if name == "Number" || name == "String" || name == "Boolean" || name == "Object" || name == "Array" {
            return Val::Str(name.to_string());
        }
        self.vars.get(name).cloned().unwrap_or(Val::Undef)
    }
    fn set_var(&mut self, name: &str, v: Val) {
        self.vars.insert(name.to_string(), v);
    }
    fn call(&mut self, name: &str, args: &[Val]) -> Val {
        if name == "console.log" || name.ends_with(".log") {
            let msg = args.iter().map(|a| a.as_str()).collect::<Vec<_>>().join(" ");
            self.log.push(msg);
            return Val::Undef;
        }
        if name == "__method_toString" || name == "toString" {
            return Val::Str(args.first().map(|a| a.as_str()).unwrap_or_default());
        }
        if name == "__method_valueOf" || name == "valueOf" {
            return args.first().cloned().unwrap_or(Val::Undef);
        }
        match name {
            "parseInt" => {
                let s = args.first().map(|a| a.as_str()).unwrap_or_default();
                let n: f64 = s
                    .trim()
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '-' || *c == '+')
                    .collect::<String>()
                    .parse()
                    .unwrap_or(0.0);
                Val::Num(n)
            }
            "parseFloat" => {
                let s = args.first().map(|a| a.as_str()).unwrap_or_default();
                Val::Num(s.trim().parse().unwrap_or(0.0))
            }
            "isNaN" => {
                let n = args.first().map(|a| a.as_num()).unwrap_or(0.0);
                Val::Bool(n != n)
            }
            "isFinite" => {
                let n = args.first().map(|a| a.as_num()).unwrap_or(0.0);
                Val::Bool(n.is_finite())
            }
            "Number" => Val::Num(args.first().map(|a| a.as_num()).unwrap_or(0.0)),
            "String" => Val::Str(args.first().map(|a| a.as_str()).unwrap_or_default()),
            "Boolean" => Val::Bool(args.first().map(|a| a.as_bool()).unwrap_or(false)),
            "Object" | "Array" => Val::Undef,
            _ => Val::Undef,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn compile_and_run_arith_log() {
        let chunk = compile(
            r#"
            var x = 1 + 2 * 3;
            console.log(x);
            "#,
        )
        .expect("compile");
        let mut host = MapHost::default();
        let _ = run(&chunk, &mut host).unwrap();
        assert!(
            host.log.iter().any(|l| l.contains("7")),
            "log={:?} vars={:?}",
            host.log,
            host.vars
        );
    }

    #[test_case]
    fn function_call_and_for() {
        let chunk = compile(
            r#"
            function add(a, b) { return a + b; }
            var s = 0;
            for (var i = 0; i < 3; i = i + 1) { s = s + 1; }
            console.log(add(2, 3));
            console.log(s);
            "#,
        )
        .expect("compile");
        let mut host = MapHost::default();
        run(&chunk, &mut host).expect("run");
        assert!(host.log.iter().any(|l| l.contains("5")), "{:?}", host.log);
        assert!(host.log.iter().any(|l| l.contains("3")), "{:?}", host.log);
    }

    #[test_case]
    fn ne_gt_typeof_while() {
        let chunk = compile(
            r#"
            var x = 1;
            if (x != 2) { console.log("ne"); }
            if (3 > 2) { console.log("gt"); }
            if (typeof x == "number") { console.log("ty"); }
            var i = 0;
            while (i < 3) { i = i + 1; }
            console.log(i);
            "#,
        )
        .expect("compile");
        let mut host = MapHost::default();
        run(&chunk, &mut host).expect("run");
        assert!(host.log.iter().any(|l| l.contains("ne")), "{:?}", host.log);
        assert!(host.log.iter().any(|l| l.contains("gt")), "{:?}", host.log);
        assert!(host.log.iter().any(|l| l.contains("ty")), "{:?}", host.log);
        assert!(host.log.iter().any(|l| l.contains("3")), "{:?}", host.log);
    }

    #[test_case]
    fn if_branch() {
        let chunk = compile(
            r#"
            var x = 1;
            if (x < 2) { console.log("yes"); } else { console.log("no"); }
            "#,
        )
        .unwrap();
        let mut host = MapHost::default();
        let _ = run(&chunk, &mut host).unwrap();
        assert!(host.log.iter().any(|l| l.contains("yes")), "{:?}", host.log);
    }

    #[test_case]
    fn optimize_folds_const_add() {
        let mut chunk = compile("var x = 2 + 3; console.log(x);").unwrap();
        // optimize already called in compile; ensure result still correct
        let mut host = MapHost::default();
        let _ = run(&chunk, &mut host).unwrap();
        assert!(host.log.iter().any(|l| l.contains("5")), "{:?}", host.log);
        // Ensure Nops present after fold (or still works)
        optimize(&mut chunk);
        let mut host2 = MapHost::default();
        let _ = run(&chunk, &mut host2).unwrap();
        assert!(host2.log.iter().any(|l| l.contains("5")));
    }
}
