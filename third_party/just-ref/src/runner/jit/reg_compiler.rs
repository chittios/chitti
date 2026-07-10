//! AST-to-register-bytecode compiler.
//!
//! Emits a 3-address register bytecode representation that the register VM can execute.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::parser::ast::{
    AssignmentOperator, BinaryOperator, BlockStatementData, DeclarationType,
    ExpressionOrSpreadElement, ExpressionOrSuper, ExpressionPatternType, ExpressionType,
    ForIteratorData, FunctionBodyData, LiteralData, LiteralType, LogicalOperator,
    MemberExpressionType, NumberLiteralType, PatternOrExpression, PatternType, ProgramData,
    StatementType, SwitchCaseData, UnaryOperator, UpdateOperator, VariableDeclarationData,
    VariableDeclarationKind, VariableDeclarationOrExpression,
};
use crate::runner::ds::value::{JsNumberType, JsValue};
use std::collections::{HashMap, HashSet};

use super::reg_bytecode::{RegChunk, RegInstruction, RegOpCode};

struct LoopContext {
    continue_target: usize,
    break_jumps: Vec<usize>,
    continue_jumps: Vec<usize>,
}

pub struct RegCompiler {
    chunk: RegChunk,
    loop_stack: Vec<LoopContext>,
    locals: HashMap<String, u32>,
    local_regs: HashSet<u32>,
    lexical_scopes: Vec<HashSet<String>>,
    next_reg: u32,
    free_regs: Vec<u32>,
}

impl RegCompiler {
    pub fn new() -> Self {
        RegCompiler {
            chunk: RegChunk::new(),
            loop_stack: Vec::new(),
            locals: HashMap::new(),
            local_regs: HashSet::new(),
            lexical_scopes: vec![HashSet::new()],
            next_reg: 0,
            free_regs: Vec::new(),
        }
    }

    pub fn compile_program(mut self, program: &ProgramData) -> RegChunk {
        for stmt in &program.body {
            self.compile_statement(stmt);
        }
        self.chunk.emit_op(RegOpCode::Halt);
        self.chunk.set_register_count(self.next_reg);
        self.chunk
    }

    fn alloc_reg(&mut self) -> u32 {
        if let Some(reg) = self.free_regs.pop() {
            return reg;
        }
        let reg = self.next_reg;
        self.next_reg += 1;
        reg
    }

    fn release_reg(&mut self, reg: u32) {
        if self.local_regs.contains(&reg) {
            return;
        }
        self.free_regs.push(reg);
    }

    fn get_local_reg(&self, name: &str) -> Option<u32> {
        self.locals.get(name).copied()
    }

    fn get_or_add_local_reg(&mut self, name: &str) -> u32 {
        if let Some(reg) = self.locals.get(name) {
            return *reg;
        }
        let reg = self.alloc_reg();
        let name_idx = self.chunk.add_name(name);
        self.chunk.add_local(name_idx, reg);
        self.locals.insert(name.to_string(), reg);
        self.local_regs.insert(reg);
        reg
    }

    fn push_lexical_scope(&mut self) {
        self.lexical_scopes.push(HashSet::new());
    }

    fn pop_lexical_scope(&mut self) {
        self.lexical_scopes.pop();
    }

    fn declare_lexical(&mut self, name: &str) {
        if let Some(scope) = self.lexical_scopes.last_mut() {
            scope.insert(name.to_string());
        }
    }

    fn is_lexically_shadowed(&self, name: &str) -> bool {
        self.lexical_scopes
            .iter()
            .rev()
            .any(|scope| scope.contains(name))
    }

    // ════════════════════════════════════════════════════════════
    // Statements
    // ════════════════════════════════════════════════════════════

    fn compile_statement(&mut self, stmt: &StatementType) {
        match stmt {
            StatementType::EmptyStatement { .. } => {}
            StatementType::ExpressionStatement { expression, .. } => {
                let reg = self.compile_expression(expression);
                self.release_reg(reg);
            }
            StatementType::BlockStatement(block) => {
                self.compile_block(block);
            }
            StatementType::DeclarationStatement(decl) => {
                self.compile_declaration(decl);
            }
            StatementType::IfStatement { test, consequent, alternate, .. } => {
                self.compile_if(test, consequent, alternate.as_ref().map(|a| a.as_ref()));
            }
            StatementType::WhileStatement { test, body, .. } => {
                self.compile_while(test, body);
            }
            StatementType::DoWhileStatement { test, body, .. } => {
                self.compile_do_while(body, test);
            }
            StatementType::ForStatement { init, test, update, body, .. } => {
                self.compile_for(init.as_ref(), test.as_ref().map(|t| t.as_ref()), update.as_ref().map(|u| u.as_ref()), body);
            }
            StatementType::ForInStatement(data) => {
                self.compile_for_in(data);
            }
            StatementType::ForOfStatement(data) => {
                self.compile_for_of(data);
            }
            StatementType::SwitchStatement { discriminant, cases, .. } => {
                self.compile_switch(discriminant, cases);
            }
            StatementType::BreakStatement { .. } => {
                self.compile_break();
            }
            StatementType::ContinueStatement { .. } => {
                self.compile_continue();
            }
            StatementType::ReturnStatement { argument, .. } => {
                if let Some(arg) = argument {
                    let reg = self.compile_expression(arg);
                    self.chunk.emit(RegInstruction::with_dst_src(RegOpCode::Return, 0, reg));
                } else {
                    let reg = self.alloc_reg();
                    self.chunk.emit(RegInstruction::with_dst(RegOpCode::LoadUndefined, reg));
                    self.chunk.emit(RegInstruction::with_dst_src(RegOpCode::Return, 0, reg));
                    self.release_reg(reg);
                }
            }
            StatementType::ThrowStatement { argument, .. } => {
                let reg = self.compile_expression(argument);
                self.chunk.emit(RegInstruction::with_dst_src(RegOpCode::Return, 0, reg));
            }
            StatementType::TryStatement { block, handler: _, finalizer, .. } => {
                self.compile_block(block);
                if let Some(fin) = finalizer {
                    self.compile_block(fin);
                }
            }
            StatementType::DebuggerStatement { .. } => {}
            StatementType::FunctionBody(body) => {
                self.compile_function_body(body);
            }
        }
    }

    fn compile_block(&mut self, block: &BlockStatementData) {
        self.push_lexical_scope();
        for stmt in &block.body {
            self.compile_statement(stmt);
        }
        self.pop_lexical_scope();
    }

    fn compile_declaration(&mut self, decl: &DeclarationType) {
        match decl {
            DeclarationType::VariableDeclaration(var_decl) => {
                self.compile_var_declaration(var_decl);
            }
            DeclarationType::FunctionOrGeneratorDeclaration(func_data) => {
                if let Some(ref id) = func_data.id {
                    let reg = self.get_or_add_local_reg(&id.name);
                    self.chunk.emit(RegInstruction::with_dst(RegOpCode::LoadUndefined, reg));
                }
            }
            DeclarationType::ClassDeclaration(_) => {}
        }
    }

    fn compile_var_declaration(&mut self, var_decl: &VariableDeclarationData) {
        let (declare_op, init_op) = match var_decl.kind {
            VariableDeclarationKind::Var => (RegOpCode::DeclareVar, RegOpCode::InitVar),
            VariableDeclarationKind::Let => (RegOpCode::DeclareLet, RegOpCode::InitBinding),
            VariableDeclarationKind::Const => (RegOpCode::DeclareConst, RegOpCode::InitBinding),
        };

        for declarator in &var_decl.declarations {
            if let PatternType::PatternWhichCanBeExpression(ExpressionPatternType::Identifier(ref id)) = declarator.id.as_ref() {
                if matches!(var_decl.kind, VariableDeclarationKind::Var) {
                    let reg = self.get_or_add_local_reg(&id.name);
                    if let Some(ref init_expr) = declarator.init {
                        let init_reg = self.compile_expression(init_expr);
                        if init_reg != reg {
                            self.chunk.emit(RegInstruction::with_dst_src(RegOpCode::Move, reg, init_reg));
                        }
                        self.release_reg(init_reg);
                    } else {
                        self.chunk.emit(RegInstruction::with_dst(RegOpCode::LoadUndefined, reg));
                    }
                    continue;
                }

                self.declare_lexical(&id.name);
                let name_idx = self.chunk.add_name(&id.name);
                self.chunk.emit(RegInstruction::with_dst_imm(declare_op, 0, name_idx));

                let init_reg = if let Some(ref init_expr) = declarator.init {
                    self.compile_expression(init_expr)
                } else {
                    let reg = self.alloc_reg();
                    self.chunk.emit(RegInstruction::with_dst(RegOpCode::LoadUndefined, reg));
                    reg
                };
                self.chunk.emit(RegInstruction::with_src_imm(init_op, init_reg, name_idx));
                self.release_reg(init_reg);
            }
        }
    }

    fn compile_if(&mut self, test: &ExpressionType, consequent: &StatementType, alternate: Option<&StatementType>) {
        let test_reg = self.compile_expression(test);
        let jump_to_else = self.emit_jump(RegOpCode::JumpIfFalse, test_reg, 0);
        self.compile_statement(consequent);
        if let Some(alt) = alternate {
            let jump_over_else = self.emit_jump(RegOpCode::Jump, 0, 0);
            self.patch_jump(jump_to_else);
            self.compile_statement(alt);
            self.patch_jump(jump_over_else);
        } else {
            self.patch_jump(jump_to_else);
        }
        self.release_reg(test_reg);
    }

    fn compile_while(&mut self, test: &ExpressionType, body: &StatementType) {
        let loop_start = self.chunk.code.len();
        self.loop_stack.push(LoopContext { continue_target: loop_start, break_jumps: Vec::new(), continue_jumps: Vec::new() });

        let test_reg = self.compile_expression(test);
        let exit_jump = self.emit_jump(RegOpCode::JumpIfFalse, test_reg, 0);
        self.release_reg(test_reg);

        self.compile_statement(body);
        self.chunk.emit(RegInstruction::with_dst_imm(RegOpCode::Jump, 0, loop_start as u32));

        self.patch_jump(exit_jump);
        let ctx = self.loop_stack.pop().unwrap();
        for bj in ctx.break_jumps {
            self.patch_jump(bj);
        }
    }

    fn compile_do_while(&mut self, body: &StatementType, test: &ExpressionType) {
        let loop_start = self.chunk.code.len();
        self.loop_stack.push(LoopContext { continue_target: loop_start, break_jumps: Vec::new(), continue_jumps: Vec::new() });

        self.compile_statement(body);

        let continue_target = self.chunk.code.len();
        if let Some(ctx) = self.loop_stack.last_mut() {
            ctx.continue_target = continue_target;
        }
        let test_reg = self.compile_expression(test);
        self.chunk.emit(RegInstruction::with_src_imm(RegOpCode::JumpIfTrue, test_reg, loop_start as u32));
        self.release_reg(test_reg);

        let ctx = self.loop_stack.pop().unwrap();
        for cj in &ctx.continue_jumps {
            self.chunk.code[*cj].imm = continue_target as u32;
        }
        for bj in ctx.break_jumps {
            self.patch_jump(bj);
        }
    }

    fn compile_for(
        &mut self,
        init: Option<&VariableDeclarationOrExpression>,
        test: Option<&ExpressionType>,
        update: Option<&ExpressionType>,
        body: &StatementType,
    ) {
        self.push_lexical_scope();

        if let Some(init) = init {
            match init {
                VariableDeclarationOrExpression::VariableDeclaration(var_decl) => {
                    self.compile_var_declaration(var_decl);
                }
                VariableDeclarationOrExpression::Expression(expr) => {
                    let reg = self.compile_expression(expr);
                    self.release_reg(reg);
                }
            }
        }

        let loop_start = self.chunk.code.len();
        self.loop_stack.push(LoopContext { continue_target: loop_start, break_jumps: Vec::new(), continue_jumps: Vec::new() });

        let exit_jump = if let Some(test) = test {
            let test_reg = self.compile_expression(test);
            let jump = self.emit_jump(RegOpCode::JumpIfFalse, test_reg, 0);
            self.release_reg(test_reg);
            Some(jump)
        } else {
            None
        };

        self.compile_statement(body);

        let update_pos = self.chunk.code.len();
        if let Some(ctx) = self.loop_stack.last_mut() {
            ctx.continue_target = update_pos;
        }

        if let Some(update) = update {
            let reg = self.compile_expression(update);
            self.release_reg(reg);
        }

        self.chunk.emit(RegInstruction::with_dst_imm(RegOpCode::Jump, 0, loop_start as u32));

        if let Some(jump) = exit_jump {
            self.patch_jump(jump);
        }

        let ctx = self.loop_stack.pop().unwrap();
        // Patch continue jumps to the update position
        for cj in &ctx.continue_jumps {
            self.chunk.code[*cj].imm = update_pos as u32;
        }
        for bj in ctx.break_jumps {
            self.patch_jump(bj);
        }

        self.pop_lexical_scope();
    }

    fn compile_for_in(&mut self, _data: &ForIteratorData) {}

    fn compile_for_of(&mut self, _data: &ForIteratorData) {}

    fn compile_switch(&mut self, discriminant: &ExpressionType, cases: &[SwitchCaseData]) {
        let disc_reg = self.compile_expression(discriminant);
        let mut case_body_jumps: Vec<usize> = Vec::new();
        let mut default_jump: Option<usize> = None;

        for case in cases {
            if let Some(ref test) = case.test {
                let test_reg = self.compile_expression(test);
                let res_reg = self.alloc_reg();
                self.chunk.emit(RegInstruction::with_dst_srcs(RegOpCode::StrictEqual, res_reg, disc_reg, test_reg));
                let jump = self.emit_jump(RegOpCode::JumpIfTrue, res_reg, 0);
                self.release_reg(test_reg);
                self.release_reg(res_reg);
                case_body_jumps.push(jump);
            } else {
                default_jump = Some(case_body_jumps.len());
                case_body_jumps.push(0);
            }
        }

        let jump_to_default_or_end = self.emit_jump(RegOpCode::Jump, 0, 0);

        self.loop_stack.push(LoopContext { continue_target: 0, break_jumps: Vec::new(), continue_jumps: Vec::new() });

        for (i, case) in cases.iter().enumerate() {
            let body_pos = self.chunk.code.len();
            if i < case_body_jumps.len() && case_body_jumps[i] != 0 {
                self.chunk.code[case_body_jumps[i]].imm = body_pos as u32;
            }
            if default_jump == Some(i) {
                self.chunk.code[jump_to_default_or_end].imm = body_pos as u32;
            }
            for stmt in &case.consequent {
                self.compile_statement(stmt);
            }
        }

        let end_pos = self.chunk.code.len();
        if default_jump.is_none() {
            self.chunk.code[jump_to_default_or_end].imm = end_pos as u32;
        }

        let ctx = self.loop_stack.pop().unwrap();
        for bj in ctx.break_jumps {
            self.patch_jump(bj);
        }

        self.release_reg(disc_reg);
    }

    fn compile_break(&mut self) {
        if self.loop_stack.is_empty() {
            return;
        }
        let jump = self.emit_jump(RegOpCode::Jump, 0, 0);
        if let Some(ctx) = self.loop_stack.last_mut() {
            ctx.break_jumps.push(jump);
        }
    }

    fn compile_continue(&mut self) {
        if let Some(ctx) = self.loop_stack.last() {
            let target = ctx.continue_target;
            let jump = self.chunk.emit(RegInstruction::with_dst_imm(RegOpCode::Jump, 0, target as u32));
            // Record the jump so it can be patched later (for for-loops where
            // the update position isn't known yet when continue is compiled).
            if let Some(ctx) = self.loop_stack.last_mut() {
                ctx.continue_jumps.push(jump);
            }
        }
    }

    fn emit_jump(&mut self, op: RegOpCode, src1: u32, target: u32) -> usize {
        let instr = match op {
            RegOpCode::Jump => RegInstruction::with_dst_imm(op, 0, target),
            RegOpCode::JumpIfFalse | RegOpCode::JumpIfTrue => RegInstruction::with_src_imm(op, src1, target),
            _ => RegInstruction::with_dst_imm(op, 0, target),
        };
        self.chunk.emit(instr)
    }

    fn patch_jump(&mut self, jump_idx: usize) {
        let target = self.chunk.code.len() as u32;
        self.chunk.code[jump_idx].imm = target;
    }

    fn compile_function_body(&mut self, body: &FunctionBodyData) {
        for stmt in &body.body {
            self.compile_statement(stmt);
        }
        self.chunk.emit_op(RegOpCode::Halt);
    }

    // ════════════════════════════════════════════════════════════
    // Expressions
    // ════════════════════════════════════════════════════════════

    fn compile_expression(&mut self, expr: &ExpressionType) -> u32 {
        match expr {
            ExpressionType::Literal(lit) => self.compile_literal(lit),
            ExpressionType::ExpressionWhichCanBePattern(pattern) => self.compile_expression_pattern(pattern),
            ExpressionType::BinaryExpression { operator, left, right, .. } => self.compile_binary(operator, left, right),
            ExpressionType::LogicalExpression { operator, left, right, .. } => self.compile_logical(operator, left, right),
            ExpressionType::AssignmentExpression { operator, left, right, .. } => self.compile_assignment(operator, left, right),
            ExpressionType::UnaryExpression { operator, argument, .. } => self.compile_unary(operator, argument),
            ExpressionType::MemberExpression(member) => self.compile_member_expression(member),
            ExpressionType::CallExpression { callee, arguments, .. } => self.compile_call_expression(callee, arguments),
            ExpressionType::UpdateExpression { operator, argument, prefix, .. } => self.compile_update(operator, argument, *prefix),
            ExpressionType::SequenceExpression { expressions, .. } => {
                let mut last_reg = self.alloc_reg();
                self.chunk.emit(RegInstruction::with_dst(RegOpCode::LoadUndefined, last_reg));
                for (i, expr) in expressions.iter().enumerate() {
                    if i > 0 {
                        self.release_reg(last_reg);
                    }
                    last_reg = self.compile_expression(expr);
                }
                last_reg
            }
            ExpressionType::ConditionalExpression { test, consequent, alternate, .. } => {
                let test_reg = self.compile_expression(test);
                let result_reg = self.alloc_reg();
                let jump_to_alt = self.emit_jump(RegOpCode::JumpIfFalse, test_reg, 0);
                self.release_reg(test_reg);
                let cons_reg = self.compile_expression(consequent);
                self.chunk.emit(RegInstruction::with_dst_src(RegOpCode::Move, result_reg, cons_reg));
                self.release_reg(cons_reg);
                let jump_to_end = self.emit_jump(RegOpCode::Jump, 0, 0);
                self.patch_jump(jump_to_alt);
                let alt_reg = self.compile_expression(alternate);
                self.chunk.emit(RegInstruction::with_dst_src(RegOpCode::Move, result_reg, alt_reg));
                self.release_reg(alt_reg);
                self.patch_jump(jump_to_end);
                result_reg
            }
            _ => {
                let reg = self.alloc_reg();
                self.chunk.emit(RegInstruction::with_dst(RegOpCode::LoadUndefined, reg));
                reg
            }
        }
    }

    fn compile_literal(&mut self, lit: &LiteralData) -> u32 {
        let reg = self.alloc_reg();
        match &lit.value {
            LiteralType::NullLiteral => {
                self.chunk.emit(RegInstruction::with_dst(RegOpCode::LoadNull, reg));
            }
            LiteralType::BooleanLiteral(true) => {
                self.chunk.emit(RegInstruction::with_dst(RegOpCode::LoadTrue, reg));
            }
            LiteralType::BooleanLiteral(false) => {
                self.chunk.emit(RegInstruction::with_dst(RegOpCode::LoadFalse, reg));
            }
            LiteralType::StringLiteral(s) => {
                let idx = self.chunk.add_constant(JsValue::String(s.clone()));
                self.chunk.emit(RegInstruction::with_dst_imm(RegOpCode::LoadConst, reg, idx));
            }
            LiteralType::NumberLiteral(n) => {
                let value = match n {
                    NumberLiteralType::IntegerLiteral(i) => JsValue::Number(JsNumberType::Integer(*i)),
                    NumberLiteralType::FloatLiteral(f) => JsValue::Number(JsNumberType::Float(*f)),
                };
                let idx = self.chunk.add_constant(value);
                self.chunk.emit(RegInstruction::with_dst_imm(RegOpCode::LoadConst, reg, idx));
            }
            LiteralType::RegExpLiteral(_) => {
                self.chunk.emit(RegInstruction::with_dst(RegOpCode::LoadUndefined, reg));
            }
        }
        reg
    }

    fn compile_expression_pattern(&mut self, pattern: &ExpressionPatternType) -> u32 {
        match pattern {
            ExpressionPatternType::Identifier(id) => {
                if !self.is_lexically_shadowed(&id.name) {
                    if let Some(reg) = self.get_local_reg(&id.name) {
                        return reg;
                    }
                }
                let name_idx = self.chunk.add_name(&id.name);
                let reg = self.alloc_reg();
                self.chunk.emit(RegInstruction::with_dst_imm(RegOpCode::GetVar, reg, name_idx));
                reg
            }
        }
    }

    fn compile_binary(&mut self, operator: &BinaryOperator, left: &ExpressionType, right: &ExpressionType) -> u32 {
        let left_reg = self.compile_expression(left);
        let right_reg = self.compile_expression(right);
        let dst = self.alloc_reg();
        let op = match operator {
            BinaryOperator::Add => RegOpCode::Add,
            BinaryOperator::Subtract => RegOpCode::Sub,
            BinaryOperator::Multiply => RegOpCode::Mul,
            BinaryOperator::Divide => RegOpCode::Div,
            BinaryOperator::Modulo => RegOpCode::Mod,
            BinaryOperator::BitwiseAnd => RegOpCode::BitAnd,
            BinaryOperator::BitwiseOr => RegOpCode::BitOr,
            BinaryOperator::BitwiseXor => RegOpCode::BitXor,
            BinaryOperator::BitwiseLeftShift => RegOpCode::ShiftLeft,
            BinaryOperator::BitwiseRightShift => RegOpCode::ShiftRight,
            BinaryOperator::BitwiseUnsignedRightShift => RegOpCode::UShiftRight,
            BinaryOperator::LessThan => RegOpCode::LessThan,
            BinaryOperator::LessThanEqual => RegOpCode::LessEqual,
            BinaryOperator::GreaterThan => RegOpCode::GreaterThan,
            BinaryOperator::GreaterThanEqual => RegOpCode::GreaterEqual,
            BinaryOperator::LooselyEqual => RegOpCode::Equal,
            BinaryOperator::LooselyUnequal => RegOpCode::NotEqual,
            BinaryOperator::StrictlyEqual => RegOpCode::StrictEqual,
            BinaryOperator::StrictlyUnequal => RegOpCode::StrictNotEqual,
            _ => RegOpCode::Add,
        };
        self.chunk.emit(RegInstruction::with_dst_srcs(op, dst, left_reg, right_reg));
        self.release_reg(left_reg);
        self.release_reg(right_reg);
        dst
    }

    fn compile_logical(&mut self, operator: &LogicalOperator, left: &ExpressionType, right: &ExpressionType) -> u32 {
        let left_reg = self.compile_expression(left);
        let result_reg = self.alloc_reg();
        let jump_to_else = match operator {
            LogicalOperator::And => self.emit_jump(RegOpCode::JumpIfFalse, left_reg, 0),
            LogicalOperator::Or => self.emit_jump(RegOpCode::JumpIfTrue, left_reg, 0),
        };

        let right_reg = self.compile_expression(right);
        self.chunk.emit(RegInstruction::with_dst_src(RegOpCode::Move, result_reg, right_reg));
        self.release_reg(right_reg);
        let jump_to_end = self.emit_jump(RegOpCode::Jump, 0, 0);

        self.patch_jump(jump_to_else);
        self.chunk.emit(RegInstruction::with_dst_src(RegOpCode::Move, result_reg, left_reg));
        self.patch_jump(jump_to_end);

        self.release_reg(left_reg);
        result_reg
    }

    fn compile_assignment(&mut self, op: &AssignmentOperator, left: &PatternOrExpression, right: &ExpressionType) -> u32 {
        let name = self.extract_assignment_target_name(left);
        if let Some(name) = name {
            if !self.is_lexically_shadowed(&name) {
                if let Some(local_reg) = self.get_local_reg(&name) {
                    match op {
                        AssignmentOperator::Equals => {
                            let rhs = self.compile_expression(right);
                            if rhs != local_reg {
                                self.chunk.emit(RegInstruction::with_dst_src(RegOpCode::Move, local_reg, rhs));
                            }
                            return rhs;
                        }
                        _ => {
                            let rhs = self.compile_expression(right);
                            let op_code = self.assignment_op_to_reg(op);
                            self.chunk.emit(RegInstruction::with_dst_srcs(op_code, local_reg, local_reg, rhs));
                            self.release_reg(rhs);
                            return local_reg;
                        }
                    }
                }
            }

            let name_idx = self.chunk.add_name(&name);
            match op {
                AssignmentOperator::Equals => {
                    let rhs = self.compile_expression(right);
                    self.chunk.emit(RegInstruction::with_src_imm(RegOpCode::SetVar, rhs, name_idx));
                    rhs
                }
                _ => {
                    let cur = self.alloc_reg();
                    self.chunk.emit(RegInstruction::with_dst_imm(RegOpCode::GetVar, cur, name_idx));
                    let rhs = self.compile_expression(right);
                    let op_code = self.assignment_op_to_reg(op);
                    self.chunk.emit(RegInstruction::with_dst_srcs(op_code, cur, cur, rhs));
                    self.chunk.emit(RegInstruction::with_src_imm(RegOpCode::SetVar, cur, name_idx));
                    self.release_reg(rhs);
                    cur
                }
            }
        } else {
            // Member expression assignment (fallback)
            let rhs = self.compile_expression(right);
            rhs
        }
    }

    fn compile_unary(&mut self, operator: &UnaryOperator, argument: &ExpressionType) -> u32 {
        let arg = self.compile_expression(argument);
        let dst = self.alloc_reg();
        let op = match operator {
            UnaryOperator::Minus => RegOpCode::Negate,
            UnaryOperator::Plus => RegOpCode::UnaryPlus,
            UnaryOperator::LogicalNot => RegOpCode::Not,
            UnaryOperator::TypeOf => RegOpCode::TypeOf,
            _ => RegOpCode::Move,
        };
        match op {
            RegOpCode::Move => {
                self.chunk.emit(RegInstruction::with_dst_src(RegOpCode::Move, dst, arg));
            }
            _ => {
                self.chunk.emit(RegInstruction::with_dst_src(op, dst, arg));
            }
        }
        self.release_reg(arg);
        dst
    }

    fn compile_member_expression(&mut self, member: &MemberExpressionType) -> u32 {
        match member {
            MemberExpressionType::SimpleMemberExpression { object, property, .. } => {
                let obj_reg = self.compile_expression_or_super(object);
                let dst = self.alloc_reg();
                let name_idx = self.chunk.add_name(&property.name);
                self.chunk.emit(RegInstruction::new(RegOpCode::GetProp, dst, obj_reg, 0, name_idx));
                self.release_reg(obj_reg);
                dst
            }
            MemberExpressionType::ComputedMemberExpression { object, property, .. } => {
                let obj_reg = self.compile_expression_or_super(object);
                let key_reg = self.compile_expression(property);
                let dst = self.alloc_reg();
                self.chunk.emit(RegInstruction::with_dst_srcs(RegOpCode::GetElem, dst, obj_reg, key_reg));
                self.release_reg(obj_reg);
                self.release_reg(key_reg);
                dst
            }
        }
    }

    fn compile_call_expression(&mut self, callee: &ExpressionOrSuper, arguments: &[ExpressionOrSpreadElement]) -> u32 {
        let dst = self.alloc_reg();
        let callee_reg = self.compile_expression_or_super(callee);
        let mut arg_regs: Vec<u32> = Vec::with_capacity(arguments.len());
        for arg in arguments {
            match arg {
                ExpressionOrSpreadElement::Expression(e) => arg_regs.push(self.compile_expression(e)),
                ExpressionOrSpreadElement::SpreadElement(e) => arg_regs.push(self.compile_expression(e)),
            }
        }
        // For now, ignore arguments in register bytecode (future work)
        self.chunk.emit(RegInstruction::new(
            RegOpCode::Call,
            dst,
            callee_reg,
            0,
            arg_regs.len() as u32,
        ));
        self.release_reg(callee_reg);
        for reg in arg_regs {
            self.release_reg(reg);
        }
        dst
    }

    fn compile_expression_or_super(&mut self, expr_or_super: &ExpressionOrSuper) -> u32 {
        match expr_or_super {
            ExpressionOrSuper::Expression(expr) => self.compile_expression(expr),
            ExpressionOrSuper::Super => {
                let reg = self.alloc_reg();
                self.chunk.emit(RegInstruction::with_dst(RegOpCode::LoadUndefined, reg));
                reg
            }
        }
    }

    fn compile_update(&mut self, op: &UpdateOperator, argument: &ExpressionType, prefix: bool) -> u32 {
        if let ExpressionType::ExpressionWhichCanBePattern(ExpressionPatternType::Identifier(ref id)) = argument {
            if !self.is_lexically_shadowed(&id.name) {
                if let Some(local_reg) = self.get_local_reg(&id.name) {
                    let one = self.alloc_reg();
                    let one_idx = self.chunk.add_constant(JsValue::Number(JsNumberType::Integer(1)));
                    self.chunk.emit(RegInstruction::with_dst_imm(RegOpCode::LoadConst, one, one_idx));
                    let op_code = match op {
                        UpdateOperator::PlusPlus => RegOpCode::Add,
                        UpdateOperator::MinusMinus => RegOpCode::Sub,
                    };
                    if prefix {
                        self.chunk.emit(RegInstruction::with_dst_srcs(op_code, local_reg, local_reg, one));
                        self.release_reg(one);
                        return local_reg;
                    }

                    let old = self.alloc_reg();
                    self.chunk.emit(RegInstruction::with_dst_src(RegOpCode::Move, old, local_reg));
                    self.chunk.emit(RegInstruction::with_dst_srcs(op_code, local_reg, local_reg, one));
                    self.release_reg(one);
                    return old;
                }
            }
            let name_idx = self.chunk.add_name(&id.name);
            let cur = self.alloc_reg();
            self.chunk.emit(RegInstruction::with_dst_imm(RegOpCode::GetVar, cur, name_idx));
            let one = self.alloc_reg();
            let one_idx = self.chunk.add_constant(JsValue::Number(JsNumberType::Integer(1)));
            self.chunk.emit(RegInstruction::with_dst_imm(RegOpCode::LoadConst, one, one_idx));
            let op_code = match op {
                UpdateOperator::PlusPlus => RegOpCode::Add,
                UpdateOperator::MinusMinus => RegOpCode::Sub,
            };
            if prefix {
                self.chunk.emit(RegInstruction::with_dst_srcs(op_code, cur, cur, one));
                self.chunk.emit(RegInstruction::with_src_imm(RegOpCode::SetVar, cur, name_idx));
                self.release_reg(one);
                cur
            } else {
                let old = self.alloc_reg();
                self.chunk.emit(RegInstruction::with_dst_src(RegOpCode::Move, old, cur));
                self.chunk.emit(RegInstruction::with_dst_srcs(op_code, cur, cur, one));
                self.chunk.emit(RegInstruction::with_src_imm(RegOpCode::SetVar, cur, name_idx));
                self.release_reg(one);
                old
            }
        } else {
            let reg = self.alloc_reg();
            self.chunk.emit(RegInstruction::with_dst(RegOpCode::LoadUndefined, reg));
            reg
        }
    }

    fn assignment_op_to_reg(&self, op: &AssignmentOperator) -> RegOpCode {
        match op {
            AssignmentOperator::AddEquals => RegOpCode::Add,
            AssignmentOperator::SubtractEquals => RegOpCode::Sub,
            AssignmentOperator::MultiplyEquals => RegOpCode::Mul,
            AssignmentOperator::DivideEquals => RegOpCode::Div,
            AssignmentOperator::ModuloEquals => RegOpCode::Mod,
            AssignmentOperator::BitwiseAndEquals => RegOpCode::BitAnd,
            AssignmentOperator::BitwiseOrEquals => RegOpCode::BitOr,
            AssignmentOperator::BitwiseXorEquals => RegOpCode::BitXor,
            AssignmentOperator::BitwiseLeftShiftEquals => RegOpCode::ShiftLeft,
            AssignmentOperator::BitwiseRightShiftEquals => RegOpCode::ShiftRight,
            AssignmentOperator::BitwiseUnsignedRightShiftEquals => RegOpCode::UShiftRight,
            AssignmentOperator::Equals => RegOpCode::Move,
        }
    }

    fn extract_assignment_target_name(&self, target: &PatternOrExpression) -> Option<String> {
        match target {
            PatternOrExpression::Pattern(pat) => {
                if let PatternType::PatternWhichCanBeExpression(ExpressionPatternType::Identifier(ref id)) = pat.as_ref() {
                    Some(id.name.clone())
                } else {
                    None
                }
            }
            PatternOrExpression::Expression(expr) => {
                if let ExpressionType::ExpressionWhichCanBePattern(ExpressionPatternType::Identifier(ref id)) = expr.as_ref() {
                    Some(id.name.clone())
                } else {
                    None
                }
            }
        }
    }
}
