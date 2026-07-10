//! AST-to-bytecode compiler.
//!
//! Walks the AST once and emits a flat bytecode representation
//! that the VM can execute without tree-walking overhead.

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

use super::bytecode::{Chunk, FunctionTemplate, Instruction, OpCode};

/// Tracks loop context for break/continue resolution.
struct LoopContext {
    /// Instruction index of the loop condition (continue target).
    continue_target: usize,
    /// Indices of break jumps that need patching when the loop ends.
    break_jumps: Vec<usize>,
    /// Indices of continue jumps that need patching (for for-loops where
    /// the update position isn't known when continue is compiled).
    continue_jumps: Vec<usize>,
}

/// The bytecode compiler.
pub struct Compiler {
    chunk: Chunk,
    /// Stack of active loop contexts for break/continue.
    loop_stack: Vec<LoopContext>,
    /// Local slot indices for fast var access.
    locals: HashMap<String, u32>,
    /// Lexical scopes for let/const shadowing.
    lexical_scopes: Vec<HashSet<String>>,
}

impl Compiler {
    pub fn new() -> Self {
        Compiler {
            chunk: Chunk::new(),
            loop_stack: Vec::new(),
            locals: HashMap::new(),
            lexical_scopes: vec![HashSet::new()],
        }
    }

    fn get_local_slot(&self, name: &str) -> Option<u32> {
        self.locals.get(name).copied()
    }

    fn get_or_add_local_slot(&mut self, name: &str) -> u32 {
        if let Some(slot) = self.locals.get(name) {
            return *slot;
        }
        let name_idx = self.chunk.add_name(name);
        let slot = self.chunk.add_local(name_idx);
        self.locals.insert(name.to_string(), slot);
        slot
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

    /// Compile a full program AST into bytecode.
    pub fn compile_program(mut self, program: &ProgramData) -> Chunk {
        for stmt in &program.body {
            self.compile_statement(stmt);
        }
        self.chunk.emit_op(OpCode::Halt);
        self.chunk
    }

    // ════════════════════════════════════════════════════════════
    // Statements
    // ════════════════════════════════════════════════════════════

    fn compile_statement(&mut self, stmt: &StatementType) {
        match stmt {
            StatementType::EmptyStatement { .. } => {}

            StatementType::ExpressionStatement { expression, .. } => {
                self.compile_expression(expression);
                self.chunk.emit_op(OpCode::Pop);
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
                self.compile_for(
                    init.as_ref(),
                    test.as_ref().map(|t| t.as_ref()),
                    update.as_ref().map(|u| u.as_ref()),
                    body,
                );
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
                    self.compile_expression(arg);
                } else {
                    self.chunk.emit_op(OpCode::Undefined);
                }
                self.chunk.emit_op(OpCode::Return);
            }

            StatementType::ThrowStatement { argument, .. } => {
                self.compile_expression(argument);
                // For now, throw is treated as a return-like halt.
                // Full exception handling would need exception tables.
                self.chunk.emit_op(OpCode::Return);
            }

            StatementType::TryStatement { block, handler, finalizer, .. } => {
                // Simplified: just execute the block, ignore try/catch semantics for now.
                // Full implementation would need exception tables in the bytecode.
                self.compile_block(block);
                if let Some(_handler) = handler {
                    // TODO: exception table support
                }
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
        self.chunk.emit_op(OpCode::PushScope);
        for stmt in block.body.iter() {
            self.compile_statement(stmt);
        }
        self.chunk.emit_op(OpCode::PopScope);
        self.pop_lexical_scope();
    }

    fn compile_function_body(&mut self, body: &FunctionBodyData) {
        self.push_lexical_scope();
        for stmt in body.body.iter() {
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
                    let slot = self.get_or_add_local_slot(&id.name);
                    let name_idx = self.chunk.add_name(&id.name);
                    let func_idx = self.chunk.add_function(FunctionTemplate {
                        body_ptr: func_data.body.as_ref() as *const _,
                        params_ptr: &func_data.params.list as *const _,
                    });
                    // Declare in the EvalContext so the function is visible
                    // to its own body (recursion) and to closures.
                    self.chunk.emit_with(OpCode::DeclareVar, name_idx);
                    self.chunk.emit_with(OpCode::MakeClosure, func_idx);
                    self.chunk.emit_op(OpCode::Dup);
                    self.chunk.emit_with(OpCode::InitVar, name_idx);
                    self.chunk.emit_with(OpCode::InitLocal, slot);
                }
            }
            DeclarationType::ClassDeclaration(_class_data) => {
                // TODO: class compilation
            }
        }
    }

    fn compile_var_declaration(&mut self, var_decl: &VariableDeclarationData) {
        let (declare_op, init_op) = match var_decl.kind {
            VariableDeclarationKind::Var => (OpCode::DeclareVar, OpCode::InitVar),
            VariableDeclarationKind::Let => (OpCode::DeclareLet, OpCode::InitBinding),
            VariableDeclarationKind::Const => (OpCode::DeclareConst, OpCode::InitBinding),
        };

        for declarator in &var_decl.declarations {
            if let PatternType::PatternWhichCanBeExpression(
                ExpressionPatternType::Identifier(ref id),
            ) = declarator.id.as_ref()
            {
                if matches!(var_decl.kind, VariableDeclarationKind::Var) {
                    let slot = self.get_or_add_local_slot(&id.name);
                    if let Some(ref init_expr) = declarator.init {
                        self.compile_expression(init_expr);
                    } else {
                        self.chunk.emit_op(OpCode::Undefined);
                    }
                    self.chunk.emit_with(OpCode::InitLocal, slot);
                    continue;
                }

                // Track lexical declarations to prevent var slot usage in nested scopes.
                self.declare_lexical(&id.name);

                let name_idx = self.chunk.add_name(&id.name);
                self.chunk.emit_with(declare_op, name_idx);

                if let Some(ref init_expr) = declarator.init {
                    self.compile_expression(init_expr);
                } else {
                    self.chunk.emit_op(OpCode::Undefined);
                }
                self.chunk.emit_with(init_op, name_idx);
            }
        }
    }

    // ── Control flow ─────────────────────────────────────────

    fn compile_if(
        &mut self,
        test: &ExpressionType,
        consequent: &StatementType,
        alternate: Option<&StatementType>,
    ) {
        self.compile_expression(test);
        let jump_to_else = self.chunk.emit_with(OpCode::JumpIfFalse, 0);

        self.compile_statement(consequent);

        if let Some(alt) = alternate {
            let jump_over_else = self.chunk.emit_with(OpCode::Jump, 0);
            self.chunk.patch_jump(jump_to_else);
            self.compile_statement(alt);
            self.chunk.patch_jump(jump_over_else);
        } else {
            self.chunk.patch_jump(jump_to_else);
        }
    }

    fn compile_while(&mut self, test: &ExpressionType, body: &StatementType) {
        let loop_start = self.chunk.current_pos();

        self.loop_stack.push(LoopContext {
            continue_target: loop_start,
            break_jumps: Vec::new(),
            continue_jumps: Vec::new(),
        });

        self.compile_expression(test);
        let exit_jump = self.chunk.emit_with(OpCode::JumpIfFalse, 0);

        self.compile_statement(body);
        self.chunk.emit_with(OpCode::Jump, loop_start as u32);

        self.chunk.patch_jump(exit_jump);

        let ctx = self.loop_stack.pop().unwrap();
        for bj in ctx.break_jumps {
            self.chunk.patch_jump(bj);
        }
    }

    fn compile_do_while(&mut self, body: &StatementType, test: &ExpressionType) {
        let loop_start = self.chunk.current_pos();

        self.loop_stack.push(LoopContext {
            continue_target: loop_start,
            break_jumps: Vec::new(),
            continue_jumps: Vec::new(),
        });

        self.compile_statement(body);

        // Continue target is the test
        let continue_target = self.chunk.current_pos();
        if let Some(ctx) = self.loop_stack.last_mut() {
            ctx.continue_target = continue_target;
        }

        self.compile_expression(test);
        self.chunk.emit_with(OpCode::JumpIfTrue, loop_start as u32);

        let ctx = self.loop_stack.pop().unwrap();
        for cj in &ctx.continue_jumps {
            self.chunk.code[*cj].operand = continue_target as u32;
        }
        for bj in ctx.break_jumps {
            self.chunk.patch_jump(bj);
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
        self.chunk.emit_op(OpCode::PushScope);

        // Init
        if let Some(init) = init {
            match init {
                VariableDeclarationOrExpression::VariableDeclaration(var_decl) => {
                    self.compile_var_declaration(var_decl);
                }
                VariableDeclarationOrExpression::Expression(expr) => {
                    self.compile_expression(expr);
                    self.chunk.emit_op(OpCode::Pop);
                }
            }
        }

        let loop_start = self.chunk.current_pos();

        // We need to know where the update is for continue
        // So we use a placeholder continue_target and fix it later
        self.loop_stack.push(LoopContext {
            continue_target: loop_start, // will be updated
            break_jumps: Vec::new(),
            continue_jumps: Vec::new(),
        });

        // Test
        let exit_jump = if let Some(test) = test {
            self.compile_expression(test);
            Some(self.chunk.emit_with(OpCode::JumpIfFalse, 0))
        } else {
            None
        };

        // Body
        self.compile_statement(body);

        // Update (this is the continue target)
        let update_pos = self.chunk.current_pos();
        if let Some(ctx) = self.loop_stack.last_mut() {
            ctx.continue_target = update_pos;
        }

        if let Some(update) = update {
            self.compile_expression(update);
            self.chunk.emit_op(OpCode::Pop);
        }

        self.chunk.emit_with(OpCode::Jump, loop_start as u32);

        if let Some(ej) = exit_jump {
            self.chunk.patch_jump(ej);
        }

        let ctx = self.loop_stack.pop().unwrap();
        // Patch continue jumps to the update position
        for cj in &ctx.continue_jumps {
            self.chunk.code[*cj].operand = update_pos as u32;
        }
        for bj in ctx.break_jumps {
            self.chunk.patch_jump(bj);
        }

        self.chunk.emit_op(OpCode::PopScope);
        self.pop_lexical_scope();
    }

    fn compile_for_in(&mut self, _data: &ForIteratorData) {
        // TODO: for-in requires runtime object key iteration
        // Emit a no-op for now
    }

    fn compile_for_of(&mut self, _data: &ForIteratorData) {
        // TODO: for-of requires runtime iterator protocol
        // Emit a no-op for now
    }

    fn compile_switch(
        &mut self,
        discriminant: &ExpressionType,
        cases: &[SwitchCaseData],
    ) {
        self.compile_expression(discriminant);

        // For each case, we emit: Dup, <test_expr>, StrictEqual, JumpIfFalse
        // For the default case, we just jump.
        let mut case_body_jumps: Vec<usize> = Vec::new();
        let mut default_jump: Option<usize> = None;

        // Phase 1: emit all the tests
        for case in cases {
            if let Some(ref test) = case.test {
                self.chunk.emit_op(OpCode::Dup);
                self.compile_expression(test);
                self.chunk.emit_op(OpCode::StrictEqual);
                let jump = self.chunk.emit_with(OpCode::JumpIfTrue, 0);
                case_body_jumps.push(jump);
            } else {
                // default case
                default_jump = Some(case_body_jumps.len());
                case_body_jumps.push(0); // placeholder
            }
        }

        // Jump to default or end
        let jump_to_default_or_end = self.chunk.emit_with(OpCode::Jump, 0);

        // Set up break context
        self.loop_stack.push(LoopContext {
            continue_target: 0, // not used for switch
            break_jumps: Vec::new(),
            continue_jumps: Vec::new(),
        });

        // Phase 2: emit all the bodies
        for (i, case) in cases.iter().enumerate() {
            let body_pos = self.chunk.current_pos();
            if i < case_body_jumps.len() && case_body_jumps[i] != 0 {
                self.chunk.code[case_body_jumps[i]].operand = body_pos as u32;
            }
            if default_jump == Some(i) {
                self.chunk.code[jump_to_default_or_end].operand = body_pos as u32;
            }
            for stmt in &case.consequent {
                self.compile_statement(stmt);
            }
        }

        // If no default, patch the jump-to-end
        if default_jump.is_none() {
            self.chunk.patch_jump(jump_to_default_or_end);
        }

        // Pop the discriminant
        self.chunk.emit_op(OpCode::Pop);

        let ctx = self.loop_stack.pop().unwrap();
        for bj in ctx.break_jumps {
            self.chunk.patch_jump(bj);
        }
    }

    fn compile_break(&mut self) {
        if let Some(ctx) = self.loop_stack.last_mut() {
            let jump = self.chunk.emit_with(OpCode::Jump, 0);
            ctx.break_jumps.push(jump);
        }
    }

    fn compile_continue(&mut self) {
        if let Some(ctx) = self.loop_stack.last() {
            let target = ctx.continue_target;
            let jump = self.chunk.emit_with(OpCode::Jump, target as u32);
            // Record the jump so it can be patched later (for for-loops where
            // the update position isn't known yet when continue is compiled).
            if let Some(ctx) = self.loop_stack.last_mut() {
                ctx.continue_jumps.push(jump);
            }
        }
    }

    // ════════════════════════════════════════════════════════════
    // Expressions
    // ════════════════════════════════════════════════════════════

    fn compile_expression(&mut self, expr: &ExpressionType) {
        match expr {
            ExpressionType::Literal(lit) => {
                self.compile_literal(lit);
            }

            ExpressionType::ExpressionWhichCanBePattern(pattern) => {
                self.compile_expression_pattern(pattern);
            }

            ExpressionType::ThisExpression { .. } => {
                // Push `this` — for now, treat as undefined at top level
                self.chunk.emit_op(OpCode::Undefined);
            }

            ExpressionType::BinaryExpression { operator, left, right, .. } => {
                self.compile_expression(left);
                self.compile_expression(right);
                self.compile_binary_op(operator);
            }

            ExpressionType::LogicalExpression { operator, left, right, .. } => {
                self.compile_logical(operator, left, right);
            }

            ExpressionType::UnaryExpression { operator, argument, .. } => {
                self.compile_expression(argument);
                self.compile_unary_op(operator);
            }

            ExpressionType::UpdateExpression { operator, argument, prefix, .. } => {
                self.compile_update(operator, argument, *prefix);
            }

            ExpressionType::AssignmentExpression { operator, left, right, .. } => {
                self.compile_assignment(operator, left, right);
            }

            ExpressionType::ConditionalExpression { test, consequent, alternate, .. } => {
                self.compile_expression(test);
                let jump_to_alt = self.chunk.emit_with(OpCode::JumpIfFalse, 0);
                self.compile_expression(consequent);
                let jump_over = self.chunk.emit_with(OpCode::Jump, 0);
                self.chunk.patch_jump(jump_to_alt);
                self.compile_expression(alternate);
                self.chunk.patch_jump(jump_over);
            }

            ExpressionType::SequenceExpression { expressions, .. } => {
                for (i, expr) in expressions.iter().enumerate() {
                    self.compile_expression(expr);
                    if i < expressions.len() - 1 {
                        self.chunk.emit_op(OpCode::Pop);
                    }
                }
            }

            ExpressionType::CallExpression { callee, arguments, .. } => {
                self.compile_call(callee, arguments);
            }

            ExpressionType::MemberExpression(member_expr) => {
                self.compile_member_expression(member_expr);
            }

            ExpressionType::ArrayExpression { .. } => {
                // Fallback: push undefined for now
                // Full array compilation would need CreateArray + SetElem opcodes
                self.chunk.emit_op(OpCode::Undefined);
            }

            ExpressionType::ObjectExpression { .. } => {
                // Fallback: push undefined for now
                self.chunk.emit_op(OpCode::Undefined);
            }

            ExpressionType::FunctionOrGeneratorExpression(func_data) => {
                let func_idx = self.chunk.add_function(FunctionTemplate {
                    body_ptr: func_data.body.as_ref() as *const _,
                    params_ptr: &func_data.params.list as *const _,
                });
                self.chunk.emit_with(OpCode::MakeClosure, func_idx);
            }

            ExpressionType::NewExpression { .. } => {
                self.chunk.emit_op(OpCode::Undefined);
            }

            ExpressionType::ArrowFunctionExpression { params, body, .. } => {
                match body.as_ref() {
                    crate::parser::ast::FunctionBodyOrExpression::FunctionBody(func_body) => {
                        let func_idx = self.chunk.add_function(FunctionTemplate {
                            body_ptr: func_body as *const _,
                            params_ptr: params as *const _,
                        });
                        self.chunk.emit_with(OpCode::MakeClosure, func_idx);
                    }
                    crate::parser::ast::FunctionBodyOrExpression::Expression(_) => {
                        // Concise arrow bodies need wrapping; fall back for now
                        self.chunk.emit_op(OpCode::Undefined);
                    }
                }
            }

            ExpressionType::YieldExpression { .. } => {
                self.chunk.emit_op(OpCode::Undefined);
            }

            ExpressionType::TemplateLiteral(_) => {
                self.chunk.emit_op(OpCode::Undefined);
            }

            ExpressionType::TaggedTemplateExpression { .. } => {
                self.chunk.emit_op(OpCode::Undefined);
            }

            ExpressionType::ClassExpression(_) => {
                self.chunk.emit_op(OpCode::Undefined);
            }

            ExpressionType::MetaProperty { .. } => {
                self.chunk.emit_op(OpCode::Undefined);
            }
        }
    }

    fn compile_literal(&mut self, lit: &LiteralData) {
        match &lit.value {
            LiteralType::NullLiteral => {
                self.chunk.emit_op(OpCode::Null);
            }
            LiteralType::BooleanLiteral(true) => {
                self.chunk.emit_op(OpCode::True);
            }
            LiteralType::BooleanLiteral(false) => {
                self.chunk.emit_op(OpCode::False);
            }
            LiteralType::StringLiteral(s) => {
                let idx = self.chunk.add_constant(JsValue::String(s.clone()));
                self.chunk.emit_with(OpCode::Constant, idx);
            }
            LiteralType::NumberLiteral(n) => {
                let value = match n {
                    NumberLiteralType::IntegerLiteral(i) => {
                        JsValue::Number(JsNumberType::Integer(*i))
                    }
                    NumberLiteralType::FloatLiteral(f) => {
                        JsValue::Number(JsNumberType::Float(*f))
                    }
                };
                let idx = self.chunk.add_constant(value);
                self.chunk.emit_with(OpCode::Constant, idx);
            }
            LiteralType::RegExpLiteral(_) => {
                self.chunk.emit_op(OpCode::Undefined);
            }
        }
    }

    fn compile_expression_pattern(&mut self, pattern: &ExpressionPatternType) {
        match pattern {
            ExpressionPatternType::Identifier(id) => {
                if !self.is_lexically_shadowed(&id.name) {
                    if let Some(slot) = self.get_local_slot(&id.name) {
                        self.chunk.emit_with(OpCode::GetLocal, slot);
                        return;
                    }
                }
                let name_idx = self.chunk.add_name(&id.name);
                self.chunk.emit_with(OpCode::GetVar, name_idx);
            }
        }
    }

    fn compile_binary_op(&mut self, op: &BinaryOperator) {
        match op {
            BinaryOperator::Add => self.chunk.emit_op(OpCode::Add),
            BinaryOperator::Subtract => self.chunk.emit_op(OpCode::Sub),
            BinaryOperator::Multiply => self.chunk.emit_op(OpCode::Mul),
            BinaryOperator::Divide => self.chunk.emit_op(OpCode::Div),
            BinaryOperator::Modulo => self.chunk.emit_op(OpCode::Mod),
            BinaryOperator::StrictlyEqual => self.chunk.emit_op(OpCode::StrictEqual),
            BinaryOperator::StrictlyUnequal => self.chunk.emit_op(OpCode::StrictNotEqual),
            BinaryOperator::LooselyEqual => self.chunk.emit_op(OpCode::Equal),
            BinaryOperator::LooselyUnequal => self.chunk.emit_op(OpCode::NotEqual),
            BinaryOperator::LessThan => self.chunk.emit_op(OpCode::LessThan),
            BinaryOperator::LessThanEqual => self.chunk.emit_op(OpCode::LessEqual),
            BinaryOperator::GreaterThan => self.chunk.emit_op(OpCode::GreaterThan),
            BinaryOperator::GreaterThanEqual => self.chunk.emit_op(OpCode::GreaterEqual),
            BinaryOperator::BitwiseAnd => self.chunk.emit_op(OpCode::BitAnd),
            BinaryOperator::BitwiseOr => self.chunk.emit_op(OpCode::BitOr),
            BinaryOperator::BitwiseXor => self.chunk.emit_op(OpCode::BitXor),
            BinaryOperator::BitwiseLeftShift => self.chunk.emit_op(OpCode::ShiftLeft),
            BinaryOperator::BitwiseRightShift => self.chunk.emit_op(OpCode::ShiftRight),
            BinaryOperator::BitwiseUnsignedRightShift => self.chunk.emit_op(OpCode::UShiftRight),
            BinaryOperator::In | BinaryOperator::InstanceOf => {
                // Not yet compiled
                self.chunk.emit_op(OpCode::Undefined);
                0
            }
        };
    }

    fn compile_unary_op(&mut self, op: &UnaryOperator) {
        match op {
            UnaryOperator::Minus => { self.chunk.emit_op(OpCode::Negate); }
            UnaryOperator::Plus => { self.chunk.emit_op(OpCode::UnaryPlus); }
            UnaryOperator::LogicalNot => { self.chunk.emit_op(OpCode::Not); }
            UnaryOperator::BitwiseNot => { self.chunk.emit_op(OpCode::BitNot); }
            UnaryOperator::TypeOf => { self.chunk.emit_op(OpCode::TypeOf); }
            UnaryOperator::Void => { self.chunk.emit_op(OpCode::Void); }
            _ => { self.chunk.emit_op(OpCode::Undefined); }
        };
    }

    fn compile_logical(
        &mut self,
        op: &LogicalOperator,
        left: &ExpressionType,
        right: &ExpressionType,
    ) {
        self.compile_expression(left);
        match op {
            LogicalOperator::And => {
                // Short-circuit: if left is falsy, skip right
                self.chunk.emit_op(OpCode::Dup);
                let jump = self.chunk.emit_with(OpCode::JumpIfFalse, 0);
                self.chunk.emit_op(OpCode::Pop);
                self.compile_expression(right);
                self.chunk.patch_jump(jump);
            }
            LogicalOperator::Or => {
                // Short-circuit: if left is truthy, skip right
                self.chunk.emit_op(OpCode::Dup);
                let jump = self.chunk.emit_with(OpCode::JumpIfTrue, 0);
                self.chunk.emit_op(OpCode::Pop);
                self.compile_expression(right);
                self.chunk.patch_jump(jump);
            }
        }
    }

    fn compile_update(
        &mut self,
        op: &UpdateOperator,
        argument: &ExpressionType,
        prefix: bool,
    ) {
        // Only handle simple identifier updates
        if let ExpressionType::ExpressionWhichCanBePattern(
            ExpressionPatternType::Identifier(ref id),
        ) = argument
        {
            if !self.is_lexically_shadowed(&id.name) {
                if let Some(slot) = self.get_local_slot(&id.name) {
                    let one_idx = self
                        .chunk
                        .add_constant(JsValue::Number(JsNumberType::Integer(1)));
                    match (op, prefix) {
                        (UpdateOperator::PlusPlus, true) => {
                            self.chunk.emit_with(OpCode::GetLocal, slot);
                            self.chunk.emit_with(OpCode::Constant, one_idx);
                            self.chunk.emit_op(OpCode::Add);
                            self.chunk.emit_op(OpCode::Dup);
                            self.chunk.emit_with(OpCode::SetLocal, slot);
                        }
                        (UpdateOperator::MinusMinus, true) => {
                            self.chunk.emit_with(OpCode::GetLocal, slot);
                            self.chunk.emit_with(OpCode::Constant, one_idx);
                            self.chunk.emit_op(OpCode::Sub);
                            self.chunk.emit_op(OpCode::Dup);
                            self.chunk.emit_with(OpCode::SetLocal, slot);
                        }
                        (UpdateOperator::PlusPlus, false) => {
                            self.chunk.emit_with(OpCode::GetLocal, slot);
                            self.chunk.emit_op(OpCode::Dup);
                            self.chunk.emit_with(OpCode::Constant, one_idx);
                            self.chunk.emit_op(OpCode::Add);
                            self.chunk.emit_with(OpCode::SetLocal, slot);
                        }
                        (UpdateOperator::MinusMinus, false) => {
                            self.chunk.emit_with(OpCode::GetLocal, slot);
                            self.chunk.emit_op(OpCode::Dup);
                            self.chunk.emit_with(OpCode::Constant, one_idx);
                            self.chunk.emit_op(OpCode::Sub);
                            self.chunk.emit_with(OpCode::SetLocal, slot);
                        }
                    }
                    return;
                }
            }
            let name_idx = self.chunk.add_name(&id.name);
            match (op, prefix) {
                (UpdateOperator::PlusPlus, true) => {
                    self.chunk.emit_with(OpCode::PreIncVar, name_idx);
                }
                (UpdateOperator::MinusMinus, true) => {
                    self.chunk.emit_with(OpCode::PreDecVar, name_idx);
                }
                (UpdateOperator::PlusPlus, false) => {
                    self.chunk.emit_with(OpCode::PostIncVar, name_idx);
                }
                (UpdateOperator::MinusMinus, false) => {
                    self.chunk.emit_with(OpCode::PostDecVar, name_idx);
                }
            }
        } else {
            // Fallback for member expression updates
            self.chunk.emit_op(OpCode::Undefined);
        }
    }

    fn compile_assignment(
        &mut self,
        op: &AssignmentOperator,
        left: &PatternOrExpression,
        right: &ExpressionType,
    ) {
        // Extract the variable name from the left-hand side
        let name = self.extract_assignment_target_name(left);

        if let Some(name) = name {
            if !self.is_lexically_shadowed(&name) {
                if let Some(slot) = self.get_local_slot(&name) {
                    match op {
                        AssignmentOperator::Equals => {
                            self.compile_expression(right);
                            self.chunk.emit_op(OpCode::Dup);
                            self.chunk.emit_with(OpCode::SetLocal, slot);
                        }
                        _ => {
                            self.chunk.emit_with(OpCode::GetLocal, slot);
                            self.compile_expression(right);
                            self.compile_compound_op(op);
                            self.chunk.emit_op(OpCode::Dup);
                            self.chunk.emit_with(OpCode::SetLocal, slot);
                        }
                    }
                    return;
                }
            }
            let name_idx = self.chunk.add_name(&name);
            match op {
                AssignmentOperator::Equals => {
                    self.compile_expression(right);
                    self.chunk.emit_op(OpCode::Dup); // keep value on stack as result
                    self.chunk.emit_with(OpCode::SetVar, name_idx);
                }
                _ => {
                    // Compound assignment: get current value, compute, set
                    self.chunk.emit_with(OpCode::GetVar, name_idx);
                    self.compile_expression(right);
                    self.compile_compound_op(op);
                    self.chunk.emit_op(OpCode::Dup);
                    self.chunk.emit_with(OpCode::SetVar, name_idx);
                }
            }
        } else {
            // Member expression assignment
            self.compile_member_assignment(op, left, right);
        }
    }

    fn extract_assignment_target_name(&self, target: &PatternOrExpression) -> Option<String> {
        match target {
            PatternOrExpression::Pattern(pat) => {
                if let PatternType::PatternWhichCanBeExpression(
                    ExpressionPatternType::Identifier(ref id),
                ) = pat.as_ref()
                {
                    Some(id.name.clone())
                } else {
                    None
                }
            }
            PatternOrExpression::Expression(expr) => {
                if let ExpressionType::ExpressionWhichCanBePattern(
                    ExpressionPatternType::Identifier(ref id),
                ) = expr.as_ref()
                {
                    Some(id.name.clone())
                } else {
                    None
                }
            }
        }
    }

    fn compile_member_assignment(
        &mut self,
        op: &AssignmentOperator,
        left: &PatternOrExpression,
        right: &ExpressionType,
    ) {
        if let PatternOrExpression::Expression(expr) = left {
            if let ExpressionType::MemberExpression(member) = expr.as_ref() {
                match member {
                    MemberExpressionType::SimpleMemberExpression { object, property, .. } => {
                        match object {
                            ExpressionOrSuper::Expression(obj_expr) => {
                                self.compile_expression(obj_expr);
                            }
                            ExpressionOrSuper::Super => {
                                self.chunk.emit_op(OpCode::Undefined);
                            }
                        }

                        let prop_idx = self.chunk.add_name(&property.name);
                        if matches!(op, AssignmentOperator::Equals) {
                            self.compile_expression(right);
                            self.chunk.emit_with(OpCode::SetProp, prop_idx);
                        } else {
                            self.chunk.emit_op(OpCode::Dup);
                            self.chunk.emit_with(OpCode::GetProp, prop_idx);
                            self.compile_expression(right);
                            self.compile_compound_op(op);
                            self.chunk.emit_with(OpCode::SetProp, prop_idx);
                        }
                        return;
                    }
                    MemberExpressionType::ComputedMemberExpression { object, property, .. } => {
                        match object {
                            ExpressionOrSuper::Expression(obj_expr) => {
                                self.compile_expression(obj_expr);
                            }
                            ExpressionOrSuper::Super => {
                                self.chunk.emit_op(OpCode::Undefined);
                            }
                        }
                        self.compile_expression(property);

                        if matches!(op, AssignmentOperator::Equals) {
                            self.compile_expression(right);
                            self.chunk.emit_op(OpCode::SetElem);
                        } else {
                            self.chunk.emit_op(OpCode::Dup2);
                            self.chunk.emit_op(OpCode::GetElem);
                            self.compile_expression(right);
                            self.compile_compound_op(op);
                            self.chunk.emit_op(OpCode::SetElem);
                        }
                        return;
                    }
                }
            }
        }

        // Fallback: evaluate right side only.
        self.compile_expression(right);
    }

    fn compile_compound_op(&mut self, op: &AssignmentOperator) {
        match op {
            AssignmentOperator::AddEquals => { self.chunk.emit_op(OpCode::Add); }
            AssignmentOperator::SubtractEquals => { self.chunk.emit_op(OpCode::Sub); }
            AssignmentOperator::MultiplyEquals => { self.chunk.emit_op(OpCode::Mul); }
            AssignmentOperator::DivideEquals => { self.chunk.emit_op(OpCode::Div); }
            AssignmentOperator::ModuloEquals => { self.chunk.emit_op(OpCode::Mod); }
            AssignmentOperator::BitwiseAndEquals => { self.chunk.emit_op(OpCode::BitAnd); }
            AssignmentOperator::BitwiseOrEquals => { self.chunk.emit_op(OpCode::BitOr); }
            AssignmentOperator::BitwiseXorEquals => { self.chunk.emit_op(OpCode::BitXor); }
            AssignmentOperator::BitwiseLeftShiftEquals => { self.chunk.emit_op(OpCode::ShiftLeft); }
            AssignmentOperator::BitwiseRightShiftEquals => { self.chunk.emit_op(OpCode::ShiftRight); }
            AssignmentOperator::BitwiseUnsignedRightShiftEquals => { self.chunk.emit_op(OpCode::UShiftRight); }
            AssignmentOperator::Equals => {}
        };
    }

    fn compile_call(
        &mut self,
        callee: &ExpressionOrSuper,
        arguments: &[ExpressionOrSpreadElement],
    ) {
        // Check if this is a method call (e.g., console.log)
        match callee {
            ExpressionOrSuper::Expression(expr) => {
                if let ExpressionType::MemberExpression(
                    MemberExpressionType::SimpleMemberExpression { object, property, .. },
                ) = expr.as_ref()
                {
                    // Method call: push object, then args, then CallMethod
                    match object {
                        ExpressionOrSuper::Expression(obj_expr) => {
                            self.compile_expression(obj_expr);
                        }
                        ExpressionOrSuper::Super => {
                            self.chunk.emit_op(OpCode::Undefined);
                        }
                    }

                    let argc = arguments.len();
                    for arg in arguments {
                        match arg {
                            ExpressionOrSpreadElement::Expression(e) => {
                                self.compile_expression(e);
                            }
                            ExpressionOrSpreadElement::SpreadElement(e) => {
                                self.compile_expression(e);
                            }
                        }
                    }

                    let method_idx = self.chunk.add_name(&property.name);
                    // Extract the object variable name for built-in registry lookup
                    let obj_name_idx = match object {
                        ExpressionOrSuper::Expression(obj_expr) => {
                            if let ExpressionType::ExpressionWhichCanBePattern(
                                ExpressionPatternType::Identifier(id)
                            ) = obj_expr.as_ref() {
                                self.chunk.add_name(&id.name)
                            } else {
                                self.chunk.add_name("")
                            }
                        }
                        _ => self.chunk.add_name(""),
                    };
                    self.chunk.emit(Instruction::with_three_operands(
                        OpCode::CallMethod,
                        argc as u32,
                        method_idx,
                        obj_name_idx,
                    ));
                } else {
                    // Regular function call
                    self.compile_expression(expr);
                    let argc = arguments.len();
                    for arg in arguments {
                        match arg {
                            ExpressionOrSpreadElement::Expression(e) => {
                                self.compile_expression(e);
                            }
                            ExpressionOrSpreadElement::SpreadElement(e) => {
                                self.compile_expression(e);
                            }
                        }
                    }
                    self.chunk.emit_with(OpCode::Call, argc as u32);
                }
            }
            ExpressionOrSuper::Super => {
                self.chunk.emit_op(OpCode::Undefined);
            }
        }
    }

    fn compile_member_expression(&mut self, member: &MemberExpressionType) {
        match member {
            MemberExpressionType::SimpleMemberExpression { object, property, .. } => {
                match object {
                    ExpressionOrSuper::Expression(expr) => {
                        self.compile_expression(expr);
                    }
                    ExpressionOrSuper::Super => {
                        self.chunk.emit_op(OpCode::Undefined);
                    }
                }
                let prop_idx = self.chunk.add_name(&property.name);
                self.chunk.emit_with(OpCode::GetProp, prop_idx);
            }
            MemberExpressionType::ComputedMemberExpression { object, property, .. } => {
                match object {
                    ExpressionOrSuper::Expression(expr) => {
                        self.compile_expression(expr);
                    }
                    ExpressionOrSuper::Super => {
                        self.chunk.emit_op(OpCode::Undefined);
                    }
                }
                self.compile_expression(property);
                self.chunk.emit_op(OpCode::GetElem);
            }
        }
    }
}
