//! Statement execution.
//!
//! This module provides statement execution logic for the JavaScript interpreter.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::parser::ast::{
    ClassData, ExpressionPatternType, FunctionBodyData, FunctionData, LiteralType, NumberLiteralType,
    PatternType, StatementType, DeclarationType, VariableDeclarationData, VariableDeclarationKind,
    BlockStatementData, ExpressionType, SwitchCaseData, CatchClauseData, ForIteratorData,
    VariableDeclarationOrPattern,
};
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::JsValue;
use crate::runner::plugin::types::EvalContext;

use super::types::{Completion, CompletionType, EvalResult};
use super::expression::{evaluate_expression, to_boolean, create_function_object, evaluate_class};

/// Execute a statement and return its completion.
pub fn execute_statement(
    stmt: &StatementType,
    ctx: &mut EvalContext,
) -> EvalResult {
    match stmt {
        StatementType::EmptyStatement { .. } => {
            Ok(Completion::normal())
        }

        StatementType::ExpressionStatement { expression, .. } => {
            let value = evaluate_expression(expression, ctx)?;
            Ok(Completion::normal_with_value(value))
        }

        StatementType::BlockStatement(block) => {
            execute_block_statement(block, ctx)
        }

        StatementType::DeclarationStatement(decl) => {
            execute_declaration(decl, ctx)
        }

        StatementType::IfStatement { test, consequent, alternate, .. } => {
            execute_if_statement(test, consequent, alternate.as_ref().map(|a| a.as_ref()), ctx)
        }

        StatementType::WhileStatement { test, body, .. } => {
            execute_while_statement(test, body, ctx)
        }

        StatementType::DoWhileStatement { test, body, .. } => {
            execute_do_while_statement(body, test, ctx)
        }

        StatementType::ForStatement { init, test, update, body, .. } => {
            execute_for_statement(init.as_ref(), test.as_ref().map(|t| t.as_ref()), update.as_ref().map(|u| u.as_ref()), body, ctx)
        }

        StatementType::ForInStatement(data) => {
            execute_for_in_statement(data, ctx)
        }

        StatementType::ForOfStatement(data) => {
            execute_for_of_statement(data, ctx)
        }

        StatementType::SwitchStatement { discriminant, cases, .. } => {
            execute_switch_statement(discriminant, cases, ctx)
        }

        StatementType::BreakStatement { label, .. } => {
            Ok(Completion::break_completion(sanitize_label(label.clone())))
        }

        StatementType::ContinueStatement { label, .. } => {
            Ok(Completion::continue_completion(sanitize_label(label.clone())))
        }

        StatementType::LabelledStatement { label, body, .. } => {
            execute_labelled_statement(label, body, ctx)
        }

        StatementType::ReturnStatement { argument, .. } => {
            let value = if let Some(arg) = argument {
                evaluate_expression(arg, ctx)?
            } else {
                JsValue::Undefined
            };
            Ok(Completion::return_value(value))
        }

        StatementType::ThrowStatement { argument, .. } => {
            let value = evaluate_expression(argument, ctx)?;
            Ok(Completion {
                completion_type: CompletionType::Throw,
                value: Some(value),
                target: None,
            })
        }

        StatementType::TryStatement { block, handler, finalizer, .. } => {
            execute_try_statement(block, handler.as_ref(), finalizer.as_ref(), ctx)
        }

        StatementType::DebuggerStatement { .. } => {
            Ok(Completion::normal())
        }

        StatementType::FunctionBody(body) => {
            // Execute each statement in the function body
            execute_function_body(body, ctx)
        }
    }
}

/// Normalize a `break`/`continue` target label.
///
/// A JS label is always an *Identifier*. The vendored parser, however, builds a
/// break/continue statement's label from the rule's first child unconditionally
/// (`api.rs`), so a bare `break;` / `continue;` captures its terminating `;` as
/// the label — yielding a bogus target `Some(";")` that makes a loop fail to
/// recognise (and consume) its own unlabelled break/continue, corrupting the
/// completion value. Since a captured token that does not begin like an
/// identifier can never be a real label, treat it as "no label" here, at the
/// point the label is consumed.
fn sanitize_label(label: Option<String>) -> Option<String> {
    label.filter(|l| {
        l.chars()
            .next()
            .map_or(false, |c| c.is_alphabetic() || c == '_' || c == '$')
    })
}

/// The spec's `UpdateEmpty(completionRecord, value)` (ECMA-262 §6.2.4.4): if the
/// completion carries no value (an *empty* completion), fill it with `value` —
/// preserving the completion's type and target — otherwise leave it unchanged.
///
/// This is how statement lists, `if`, and loops thread completion values so that
/// an empty trailing statement, an empty block, or an empty branch does not
/// clobber the running value, while `if`/loops still surface `undefined` when a
/// taken branch produced no value (`UpdateEmpty(stmt, undefined)`). These values
/// are observable through `eval`.
fn update_empty(mut completion: Completion, value: Option<JsValue>) -> Completion {
    if completion.value.is_none() {
        completion.value = value;
    }
    completion
}

/// Build a normal completion carrying an optional (possibly *empty*) value.
fn normal_with_optional(value: Option<JsValue>) -> Completion {
    Completion {
        completion_type: CompletionType::Normal,
        value,
        target: None,
    }
}

/// Execute a block statement.
///
/// Statement-list evaluation threads completion values with `UpdateEmpty`: the
/// block's value is the value of the last *value-producing* statement, and an
/// empty statement/declaration/nested-empty-block does not overwrite it. An
/// abrupt completion inherits the running value too (so `{ 3; break; }` breaks
/// with value `3`).
fn execute_block_statement(
    block: &BlockStatementData,
    ctx: &mut EvalContext,
) -> EvalResult {
    // Create a new block scope for let/const bindings
    ctx.push_block_scope();

    let mut running: Option<JsValue> = None;

    for stmt in block.body.iter() {
        let completion = update_empty(execute_statement(stmt, ctx)?, running.clone());

        if completion.is_abrupt() {
            ctx.pop_block_scope();
            return Ok(completion);
        }

        running = completion.value;
    }

    // Pop the block scope
    ctx.pop_block_scope();

    Ok(normal_with_optional(running))
}

/// Execute a declaration.
fn execute_declaration(
    decl: &DeclarationType,
    ctx: &mut EvalContext,
) -> EvalResult {
    match decl {
        DeclarationType::VariableDeclaration(var_decl) => {
            execute_variable_declaration(var_decl, ctx)
        }

        DeclarationType::FunctionOrGeneratorDeclaration(func_data) => {
            execute_function_declaration(func_data, ctx)
        }

        DeclarationType::ClassDeclaration(class_data) => {
            execute_class_declaration(class_data, ctx)
        }
    }
}

/// Execute a variable declaration.
fn execute_variable_declaration(
    var_decl: &VariableDeclarationData,
    ctx: &mut EvalContext,
) -> EvalResult {
    let is_const = matches!(var_decl.kind, VariableDeclarationKind::Const);
    let is_var = matches!(var_decl.kind, VariableDeclarationKind::Var);

    for declarator in &var_decl.declarations {
        // Evaluate initializer first (before potentially creating the binding)
        let value = if let Some(init) = &declarator.init {
            evaluate_expression(init, ctx)?
        } else {
            // const must have an initializer (checked by parser), var/let default to undefined
            JsValue::Undefined
        };

        // Anonymous-function name inference: `var f = function(){}` names the
        // function `f`.
        if let (
            Some(init),
            PatternType::PatternWhichCanBeExpression(ExpressionPatternType::Identifier(id)),
        ) = (&declarator.init, &*declarator.id)
        {
            if crate::runner::eval::expression::is_anonymous_function_expr(init) {
                crate::runner::eval::expression::infer_function_name(&value, &id.name);
            }
        }

        // Bind the pattern to the value
        bind_pattern(&declarator.id, value, ctx, is_const, is_var)?;
    }

    Ok(Completion::normal())
}

/// JS var + function-declaration hoisting: pre-create every `var` binding
/// reachable in `body` (initialized to `undefined`) and bind every
/// `function` declaration to its function object, without descending into
/// nested function bodies. Called at the top of every function body and the
/// program so a name is defined rather than a `ReferenceError` before its
/// declaration statement runs (React's production bundle leans on this —
/// `getModifierState:iu` appears lexically before `function iu(){…}`).
pub fn hoist_var_declarations(body: &[StatementType], ctx: &mut EvalContext) {
    for stmt in body {
        hoist_stmt(stmt, ctx);
    }
}

fn declare_hoisted_var(name: &str, ctx: &mut EvalContext) {
    if !ctx.has_var_binding(name) {
        let _ = ctx.create_var_binding(name);
        let _ = ctx.initialize_var_binding(name, JsValue::Undefined);
    }
}

fn hoist_function_declaration(func_data: &FunctionData, ctx: &mut EvalContext) {
    // Same binding as execute-time — initialize now so earlier statements can
    // reference the name. Re-running the declaration later is a no-op overwrite.
    let _ = execute_function_declaration(func_data, ctx);
}

fn collect_pattern_names(pattern: &PatternType, out: &mut Vec<String>) {
    match pattern {
        PatternType::PatternWhichCanBeExpression(ExpressionPatternType::Identifier(id)) => {
            out.push(id.name.clone());
        }
        PatternType::ObjectPattern { properties, rest, .. } => {
            for prop in properties {
                collect_pattern_names(&prop.0.value, out);
            }
            if let Some(r) = rest {
                collect_pattern_names(r, out);
            }
        }
        PatternType::ArrayPattern { elements, .. } => {
            for element in elements.iter().flatten() {
                collect_pattern_names(element, out);
            }
        }
        PatternType::AssignmentPattern { left, .. } => collect_pattern_names(left, out),
        PatternType::RestElement { argument, .. } => collect_pattern_names(argument, out),
        _ => {}
    }
}

fn hoist_stmt(stmt: &StatementType, ctx: &mut EvalContext) {
    match stmt {
        StatementType::DeclarationStatement(DeclarationType::VariableDeclaration(var_decl)) => {
            if matches!(var_decl.kind, VariableDeclarationKind::Var) {
                let mut names = Vec::new();
                for d in &var_decl.declarations {
                    collect_pattern_names(&d.id, &mut names);
                }
                for n in names {
                    declare_hoisted_var(&n, ctx);
                }
            }
        }
        StatementType::DeclarationStatement(
            DeclarationType::FunctionOrGeneratorDeclaration(func_data),
        ) => {
            hoist_function_declaration(func_data, ctx);
        }
        StatementType::BlockStatement(b) => hoist_var_declarations(&b.body, ctx),
        StatementType::IfStatement { consequent, alternate, .. } => {
            hoist_stmt(consequent, ctx);
            if let Some(a) = alternate {
                hoist_stmt(a, ctx);
            }
        }
        StatementType::WhileStatement { body, .. }
        | StatementType::DoWhileStatement { body, .. } => hoist_stmt(body, ctx),
        StatementType::ForStatement { init, body, .. } => {
            if let Some(crate::parser::ast::VariableDeclarationOrExpression::VariableDeclaration(
                var_decl,
            )) = init
            {
                if matches!(var_decl.kind, VariableDeclarationKind::Var) {
                    let mut names = Vec::new();
                    for d in &var_decl.declarations {
                        collect_pattern_names(&d.id, &mut names);
                    }
                    for n in names {
                        declare_hoisted_var(&n, ctx);
                    }
                }
            }
            hoist_stmt(body, ctx);
        }
        StatementType::ForInStatement(data) | StatementType::ForOfStatement(data) => {
            if let VariableDeclarationOrPattern::VariableDeclaration(var_decl) = &data.left {
                if matches!(var_decl.kind, VariableDeclarationKind::Var) {
                    let mut names = Vec::new();
                    for d in &var_decl.declarations {
                        collect_pattern_names(&d.id, &mut names);
                    }
                    for n in names {
                        declare_hoisted_var(&n, ctx);
                    }
                }
            }
            hoist_stmt(&data.body, ctx);
        }
        StatementType::SwitchStatement { cases, .. } => {
            for case in cases {
                hoist_var_declarations(&case.consequent, ctx);
            }
        }
        StatementType::TryStatement { block, handler, finalizer, .. } => {
            hoist_var_declarations(&block.body, ctx);
            if let Some(h) = handler {
                hoist_var_declarations(&h.body.body, ctx);
            }
            if let Some(f) = finalizer {
                hoist_var_declarations(&f.body, ctx);
            }
        }
        // A labelled statement is transparent to var hoisting — descend into
        // its body (`foo: while (…) { var x }` must still hoist `x`).
        StatementType::LabelledStatement { body, .. } => hoist_stmt(body, ctx),
        // Nested functions establish their own var scope — do not descend.
        _ => {}
    }
}

/// Bind a pattern to a value, creating bindings for all identifiers in the pattern.
///
/// Also drives *parameter* binding (`expression::bind_parameters`) — a formal
/// parameter list is the same pattern grammar as a `var`/`let` target, and the
/// two must not drift.
pub(crate) fn bind_pattern(
    pattern: &PatternType,
    value: JsValue,
    ctx: &mut EvalContext,
    is_const: bool,
    is_var: bool,
) -> Result<(), JErrorType> {
    match pattern {
        PatternType::PatternWhichCanBeExpression(ExpressionPatternType::Identifier(id)) => {
            // Simple identifier binding
            let name = &id.name;
            if is_var {
                if ctx.has_var_binding(name) {
                    ctx.set_var_binding(name, value)?;
                } else {
                    ctx.create_var_binding(name)?;
                    ctx.initialize_var_binding(name, value)?;
                }
            } else {
                ctx.create_binding(name, is_const)?;
                ctx.initialize_binding(name, value)?;
            }
            Ok(())
        }

        PatternType::ObjectPattern { properties, rest, .. } => {
            // Destructuring `null`/`undefined` is a TypeError (there is no object
            // to read properties from).
            if matches!(value, JsValue::Null | JsValue::Undefined) {
                return Err(JErrorType::TypeError(format!(
                    "Cannot destructure '{}' as it is {}.",
                    "value",
                    if matches!(value, JsValue::Null) { "null" } else { "undefined" }
                )));
            }
            // Object destructuring: { x, y } = obj or { x: renamed } = obj
            let mut consumed: alloc::vec::Vec<alloc::string::String> = alloc::vec::Vec::new();
            for prop in properties {
                let prop_data = &prop.0;

                // Get the property key name
                let key_name = get_property_key_name(&prop_data.key)?;
                consumed.push(key_name.clone());

                // Get the value from the object (getter-aware, so a poisoned
                // accessor propagates its throw and side effects run once).
                let prop_value =
                    crate::runner::eval::expression::get_property_with_ctx(&value, &key_name, ctx)?;

                // Bind the value to the pattern
                // For shorthand like { x }, value is the same pattern as key
                // For renamed like { x: renamed }, value is the renamed pattern
                bind_pattern(&prop_data.value, prop_value, ctx, is_const, is_var)?;
            }
            // `...rest` = a fresh object with the own **enumerable** props NOT
            // already destructured above; getters are invoked (their value is
            // copied as a plain data property).
            if let Some(rest_pat) = rest {
                let rest_obj = crate::runner::eval::expression::make_object(alloc::vec::Vec::new());
                for key in crate::runner::eval::expression::own_enumerable_string_keys(&value) {
                    if !consumed.contains(&key) {
                        let v = crate::runner::eval::expression::get_property_with_ctx(
                            &value, &key, ctx,
                        )?;
                        crate::runner::eval::expression::set_own_prop(&rest_obj, &key, v, true);
                    }
                }
                bind_pattern(rest_pat, rest_obj, ctx, is_const, is_var)?;
            }
            Ok(())
        }

        PatternType::ArrayPattern { elements, .. } => {
            use crate::runner::eval::expression::{
                drive_array_pattern, get_iterator, has_custom_iterator, is_array,
            };
            // Array destructuring requires an iterable; `null`/`undefined` throw.
            if matches!(value, JsValue::Null | JsValue::Undefined) {
                return Err(JErrorType::TypeError(
                    "value is not iterable".to_string(),
                ));
            }
            // Fast path: a genuine array using the default iteration protocol —
            // index-based, no iterator object (keeps the common case cheap).
            if is_array(&value) && !has_custom_iterator(&value, ctx) {
                for (index, element) in elements.iter().enumerate() {
                    if let Some(elem_pattern) = element {
                        if let PatternType::RestElement { argument, .. } = elem_pattern.as_ref() {
                            let rest_value = get_rest_elements(&value, index)?;
                            bind_pattern(argument, rest_value, ctx, is_const, is_var)?;
                        } else {
                            let elem_value = get_array_element(&value, index)?;
                            bind_pattern(elem_pattern, elem_value, ctx, is_const, is_var)?;
                        }
                    }
                    // None means hole/skip - do nothing
                }
                return Ok(());
            }
            // General path: drive the iterator protocol (generators, custom
            // iterables, strings, arrays with an overridden Symbol.iterator).
            let iter = get_iterator(&value, ctx)?;
            drive_array_pattern(&iter, elements, ctx, |pat, v, ctx| {
                bind_pattern(pat, v, ctx, is_const, is_var)
            })
        }

        PatternType::AssignmentPattern { left, right, .. } => {
            // Default value pattern: x = default or { x = default } = obj
            // Use default if value is undefined
            let used_default = matches!(value, JsValue::Undefined);
            let actual_value = if used_default {
                evaluate_expression(right, ctx)?
            } else {
                value
            };
            // Infer an anonymous default's name from a simple identifier target
            // (`[fn = function(){}] = []` names the function `fn`).
            if used_default {
                if let PatternType::PatternWhichCanBeExpression(
                    ExpressionPatternType::Identifier(id),
                ) = &**left
                {
                    if crate::runner::eval::expression::is_anonymous_function_expr(right) {
                        crate::runner::eval::expression::infer_function_name(&actual_value, &id.name);
                    }
                }
            }
            bind_pattern(left, actual_value, ctx, is_const, is_var)
        }

        PatternType::RestElement { argument, .. } => {
            // Rest element should be handled in array context
            // If we get here directly, just bind the value as-is
            bind_pattern(argument, value, ctx, is_const, is_var)
        }
    }
}

/// Get the property key name from an expression (for destructuring).
fn get_property_key_name(key_expr: &ExpressionType) -> Result<String, JErrorType> {
    match key_expr {
        ExpressionType::ExpressionWhichCanBePattern(ExpressionPatternType::Identifier(id)) => {
            Ok(id.name.clone())
        }
        ExpressionType::Literal(lit_data) => {
            match &lit_data.value {
                LiteralType::StringLiteral(s) => Ok(s.clone()),
                LiteralType::NumberLiteral(num) => {
                    match num {
                        NumberLiteralType::IntegerLiteral(n) => Ok(n.to_string()),
                        NumberLiteralType::FloatLiteral(n) => Ok(n.to_string()),
                    }
                }
                _ => Err(JErrorType::TypeError("Invalid property key in destructuring".to_string())),
            }
        }
        _ => Err(JErrorType::TypeError("Computed property keys not yet supported in destructuring".to_string())),
    }
}

/// Get an element from an array by index (for destructuring).
fn get_array_element(arr: &JsValue, index: usize) -> Result<JsValue, JErrorType> {
    use crate::runner::ds::object_property::{PropertyDescriptor, PropertyKey};

    match arr {
        JsValue::Object(obj_ref) => {
            let borrowed = obj_ref.borrow();
            let base = borrowed.as_js_object().get_object_base();
            let prop_key = PropertyKey::Str(index.to_string());

            if let Some(PropertyDescriptor::Data(data)) = base.properties.get(&prop_key) {
                Ok(data.value.clone())
            } else {
                Ok(JsValue::Undefined)
            }
        }
        _ => Err(JErrorType::TypeError("Cannot destructure non-array".to_string())),
    }
}

/// Get remaining elements from an array starting at index (for rest patterns).
/// Returns a genuine JS Array (carrying the `__array__` marker) so that
/// `Array.isArray(rest)` holds and a nested `[...[...x]]` rest keeps working.
fn get_rest_elements(arr: &JsValue, start_index: usize) -> Result<JsValue, JErrorType> {
    match arr {
        JsValue::Object(_) => {
            let len = crate::runner::eval::expression::array_len(arr);
            let mut rest: Vec<JsValue> = Vec::new();
            for i in start_index..len {
                if let Some(v) = crate::runner::eval::expression::get_own_prop_value(arr, &i.to_string()) {
                    rest.push(v);
                } else {
                    rest.push(JsValue::Undefined);
                }
            }
            Ok(crate::runner::eval::expression::make_array(rest))
        }
        _ => Err(JErrorType::TypeError("Cannot use rest with non-array".to_string())),
    }
}

/// Execute a function declaration.
fn execute_function_declaration(
    func_data: &FunctionData,
    ctx: &mut EvalContext,
) -> EvalResult {
    // Get the function name (id is mandatory for declarations)
    let name = func_data.id.as_ref()
        .ok_or_else(|| JErrorType::TypeError("Function declaration must have a name".to_string()))?
        .name.clone();

    // Create the function object
    let func_value = create_function_object(func_data, ctx)?;

    // Bind the function name in the variable environment (functions are hoisted like var)
    if ctx.has_var_binding(&name) {
        // Update existing binding
        ctx.set_var_binding(&name, func_value)?;
    } else {
        // Create and initialize new binding
        ctx.create_var_binding(&name)?;
        ctx.initialize_var_binding(&name, func_value)?;
    }

    Ok(Completion::normal())
}

/// Execute a class declaration.
fn execute_class_declaration(
    class_data: &ClassData,
    ctx: &mut EvalContext,
) -> EvalResult {
    // Get the class name (id is mandatory for declarations)
    let name = class_data.id.as_ref()
        .ok_or_else(|| JErrorType::TypeError("Class declaration must have a name".to_string()))?
        .name.clone();

    // Evaluate the class to get the constructor function
    let class_value = evaluate_class(class_data, ctx)?;

    // Bind the class name (classes are like let bindings - not hoisted to var scope)
    ctx.create_binding(&name, false)?;
    ctx.initialize_binding(&name, class_value)?;

    Ok(Completion::normal())
}

/// Get the binding name from a pattern.
fn get_binding_name(pattern: &PatternType) -> Result<String, JErrorType> {
    match pattern {
        PatternType::PatternWhichCanBeExpression(ExpressionPatternType::Identifier(id)) => {
            Ok(id.name.clone())
        }
        PatternType::ObjectPattern { .. } => {
            Err(JErrorType::TypeError("Object destructuring not yet supported".to_string()))
        }
        PatternType::ArrayPattern { .. } => {
            Err(JErrorType::TypeError("Array destructuring not yet supported".to_string()))
        }
        PatternType::RestElement { .. } => {
            Err(JErrorType::TypeError("Rest element not yet supported".to_string()))
        }
        PatternType::AssignmentPattern { .. } => {
            Err(JErrorType::TypeError("Default value patterns not yet supported".to_string()))
        }
    }
}

/// Execute an if statement.
fn execute_if_statement(
    test: &ExpressionType,
    consequent: &StatementType,
    alternate: Option<&StatementType>,
    ctx: &mut EvalContext,
) -> EvalResult {
    let test_value = evaluate_expression(test, ctx)?;

    // Per spec, an `if` whose taken branch produced an *empty* completion still
    // yields `undefined` (`UpdateEmpty(stmtCompletion, undefined)`), and an `if`
    // with no `else` whose test is false yields `NormalCompletion(undefined)`.
    let completion = if to_boolean(&test_value) {
        execute_statement(consequent, ctx)?
    } else if let Some(alt) = alternate {
        execute_statement(alt, ctx)?
    } else {
        return Ok(Completion::normal_with_value(JsValue::Undefined));
    };

    Ok(update_empty(completion, Some(JsValue::Undefined)))
}

/// Execute `label: body`. Loops take the label at entry (via
/// `ctx.pending_labels`) so `break label`/`continue label` target them; a
/// labelled non-loop (e.g. a block `n:{…}`) handles `break label` here by
/// converting the escaping Break completion to Normal.
fn execute_labelled_statement(
    label: &str,
    body: &StatementType,
    ctx: &mut EvalContext,
) -> EvalResult {
    // A label is handed to the statement it labels — and a loop *claims* every
    // pending label at entry (`loop_labels`) so `continue label` can target it.
    // So the label may only be left pending when the labelled statement is
    // itself a loop. On a block, an `if` or a `switch`, leaving it pending lets
    // a loop nested anywhere inside claim it: `e: { for (…) { break e; } throw
    // … }` then breaks only the *for* and runs the throw. That is exactly
    // ReactDOM's commitPlacement — the "expected to find a host parent"
    // invariant fired on a tree that had one, and a React app never mounted.
    let is_loop = matches!(
        body,
        StatementType::WhileStatement { .. }
            | StatementType::DoWhileStatement { .. }
            | StatementType::ForStatement { .. }
            | StatementType::ForInStatement(_)
            | StatementType::ForOfStatement(_)
            // `a: b: for (…)` — the inner labelled statement passes both on.
            | StatementType::LabelledStatement { .. }
    );
    let saved_labels = if is_loop {
        ctx.pending_labels.push(label.to_string());
        None
    } else {
        Some(core::mem::take(&mut ctx.pending_labels))
    };
    let result = execute_statement(body, ctx);
    match saved_labels {
        // A loop drains the labels at entry; for a non-loop body they linger —
        // drop ours so it can't leak onto a sibling statement.
        None => ctx.pending_labels.retain(|l| l != label),
        Some(saved) => ctx.pending_labels = saved,
    }
    let completion = result?;
    match completion.completion_type {
        // `break label` that reached here (labelled block, or a loop that
        // propagated it) completes the labelled statement normally.
        CompletionType::Break if completion.target.as_deref() == Some(label) => {
            Ok(Completion::normal_with_value(completion.get_value()))
        }
        _ => Ok(completion),
    }
}

/// A loop takes the labels applied to it, and decides how to treat a
/// break/continue completion: `matched` = no target (innermost) or a target in
/// this loop's label set (it's ours); otherwise the completion belongs to an
/// enclosing labelled statement and must propagate out.
fn loop_labels(ctx: &mut EvalContext) -> alloc::vec::Vec<alloc::string::String> {
    core::mem::take(&mut ctx.pending_labels)
}

fn targets_this_loop(target: &Option<alloc::string::String>, labels: &[alloc::string::String]) -> bool {
    match target {
        None => true,
        Some(t) => labels.iter().any(|l| l == t),
    }
}

/// Execute a while statement.
fn execute_while_statement(
    test: &ExpressionType,
    body: &StatementType,
    ctx: &mut EvalContext,
) -> EvalResult {
    let labels = loop_labels(ctx);
    // `V` in the spec: the last non-empty iteration value, threaded with
    // UpdateEmpty. `None` = the *empty* accumulator (surfaces as `undefined`).
    let mut v: Option<JsValue> = None;

    loop {
        if crate::runner::host::host_tick() {
            return Err(crate::runner::host::interrupt_error());
        }
        let test_value = evaluate_expression(test, ctx)?;

        if !to_boolean(&test_value) {
            break;
        }

        let completion = execute_statement(body, ctx)?;

        match completion.completion_type {
            CompletionType::Break => {
                if targets_this_loop(&completion.target, &labels) {
                    return Ok(Completion::normal_with_value(
                        update_empty(completion, v).get_value(),
                    ));
                }
                // labelled break for an outer statement
                return Ok(update_empty(completion, v));
            }
            CompletionType::Continue => {
                if targets_this_loop(&completion.target, &labels) {
                    if completion.value.is_some() {
                        v = completion.value;
                    }
                    continue;
                }
                return Ok(update_empty(completion, v));
            }
            CompletionType::Return | CompletionType::Throw | CompletionType::Yield => {
                return Ok(update_empty(completion, v));
            }
            CompletionType::Normal => {
                if completion.value.is_some() {
                    v = completion.value;
                }
            }
        }
    }

    Ok(Completion::normal_with_value(v.unwrap_or(JsValue::Undefined)))
}

/// Execute a do-while statement.
fn execute_do_while_statement(
    body: &StatementType,
    test: &ExpressionType,
    ctx: &mut EvalContext,
) -> EvalResult {
    let labels = loop_labels(ctx);
    let mut v: Option<JsValue> = None;
    loop {
        if crate::runner::host::host_tick() {
            return Err(crate::runner::host::interrupt_error());
        }
        let completion = execute_statement(body, ctx)?;

        match completion.completion_type {
            CompletionType::Break => {
                if targets_this_loop(&completion.target, &labels) {
                    return Ok(Completion::normal_with_value(
                        update_empty(completion, v).get_value(),
                    ));
                }
                return Ok(update_empty(completion, v));
            }
            CompletionType::Continue => {
                if !targets_this_loop(&completion.target, &labels) {
                    return Ok(update_empty(completion, v));
                }
                if completion.value.is_some() {
                    v = completion.value;
                }
            }
            CompletionType::Return | CompletionType::Throw | CompletionType::Yield => {
                return Ok(update_empty(completion, v));
            }
            CompletionType::Normal => {
                if completion.value.is_some() {
                    v = completion.value;
                }
            }
        }

        let test_value = evaluate_expression(test, ctx)?;
        if !to_boolean(&test_value) {
            break;
        }
    }

    Ok(Completion::normal_with_value(v.unwrap_or(JsValue::Undefined)))
}

/// Execute a for statement.
fn execute_for_statement(
    init: Option<&crate::parser::ast::VariableDeclarationOrExpression>,
    test: Option<&ExpressionType>,
    update: Option<&ExpressionType>,
    body: &StatementType,
    ctx: &mut EvalContext,
) -> EvalResult {
    use crate::parser::ast::VariableDeclarationOrExpression;

    if let Some(init) = init {
        match init {
            VariableDeclarationOrExpression::VariableDeclaration(decl) => {
                execute_variable_declaration(decl, ctx)?;
            }
            VariableDeclarationOrExpression::Expression(expr) => {
                evaluate_expression(expr, ctx)?;
            }
        }
    }

    let labels = loop_labels(ctx);
    let mut v: Option<JsValue> = None;

    loop {
        if crate::runner::host::host_tick() {
            return Err(crate::runner::host::interrupt_error());
        }
        if let Some(test) = test {
            let test_value = evaluate_expression(test, ctx)?;
            if !to_boolean(&test_value) {
                break;
            }
        }

        let completion = execute_statement(body, ctx)?;

        match completion.completion_type {
            CompletionType::Break => {
                if targets_this_loop(&completion.target, &labels) {
                    return Ok(Completion::normal_with_value(
                        update_empty(completion, v).get_value(),
                    ));
                }
                return Ok(update_empty(completion, v));
            }
            CompletionType::Continue => {
                if !targets_this_loop(&completion.target, &labels) {
                    return Ok(update_empty(completion, v));
                }
                if completion.value.is_some() {
                    v = completion.value;
                }
            }
            CompletionType::Return | CompletionType::Throw | CompletionType::Yield => {
                return Ok(update_empty(completion, v));
            }
            CompletionType::Normal => {
                if completion.value.is_some() {
                    v = completion.value;
                }
            }
        }

        if let Some(update) = update {
            evaluate_expression(update, ctx)?;
        }
    }

    Ok(Completion::normal_with_value(v.unwrap_or(JsValue::Undefined)))
}

/// Execute a function body.
fn execute_function_body(
    body: &FunctionBodyData,
    ctx: &mut EvalContext,
) -> EvalResult {
    let mut completion = Completion::normal();

    for stmt in body.body.iter() {
        completion = execute_statement(stmt, ctx)?;

        // Handle abrupt completions
        match completion.completion_type {
            CompletionType::Return | CompletionType::Throw | CompletionType::Yield => {
                return Ok(completion);
            }
            CompletionType::Break | CompletionType::Continue => {
                // Break/continue inside a function body without a loop is an error
                // but for now we just return the completion
                return Ok(completion);
            }
            CompletionType::Normal => {}
        }
    }

    Ok(completion)
}

/// Execute a switch statement.
fn execute_switch_statement(
    discriminant: &ExpressionType,
    cases: &[SwitchCaseData],
    ctx: &mut EvalContext,
) -> EvalResult {
    use super::expression::strict_equality;

    // Evaluate the discriminant
    let switch_value = evaluate_expression(discriminant, ctx)?;

    // Find the matching case (or default)
    let mut found_match = false;
    let mut default_index: Option<usize> = None;
    let mut start_index: Option<usize> = None;

    // First pass: find the matching case
    for (i, case) in cases.iter().enumerate() {
        if let Some(test) = &case.test {
            // This is a case clause
            let case_value = evaluate_expression(test, ctx)?;
            if strict_equality(&switch_value, &case_value) {
                start_index = Some(i);
                found_match = true;
                break;
            }
        } else {
            // This is the default clause
            default_index = Some(i);
        }
    }

    // If no match found, use default if available
    if !found_match {
        start_index = default_index;
    }

    // Execute statements starting from the matched case (fall-through behavior)
    let mut completion = Completion::normal();

    if let Some(start) = start_index {
        for case in cases.iter().skip(start) {
            for stmt in &case.consequent {
                completion = execute_statement(stmt, ctx)?;

                match completion.completion_type {
                    CompletionType::Break => {
                        // An unlabelled `break` exits the switch; a labelled
                        // `break L` belongs to an enclosing statement L and
                        // must propagate out of the switch unchanged.
                        if completion.target.is_none() {
                            return Ok(Completion::normal_with_value(completion.get_value()));
                        }
                        return Ok(completion);
                    }
                    CompletionType::Return | CompletionType::Throw | CompletionType::Continue | CompletionType::Yield => {
                        // These propagate up
                        return Ok(completion);
                    }
                    CompletionType::Normal => {}
                }
            }
        }
    }

    Ok(completion)
}

/// Execute a try statement.
fn execute_try_statement(
    block: &BlockStatementData,
    handler: Option<&CatchClauseData>,
    finalizer: Option<&BlockStatementData>,
    ctx: &mut EvalContext,
) -> EvalResult {
    // Execute the try block
    let try_result = execute_block_statement(block, ctx);

    let mut completion = match try_result {
        Ok(comp) => {
            // Check if it's a throw completion
            if comp.completion_type == CompletionType::Throw {
                // Handle the throw in catch if available
                if let Some(catch_clause) = handler {
                    handle_catch(comp, catch_clause, ctx)?
                } else {
                    comp
                }
            } else {
                comp
            }
        }
        Err(err) => {
            // A host interrupt (Ctrl+C) is uncatchable — propagate past `catch`
            // so a script's `try/catch` can't swallow the cancellation.
            if crate::runner::host::is_interrupt(&err) {
                return Err(err);
            }
            // Runtime error - treat as throw
            if let Some(catch_clause) = handler {
                let throw_completion = Completion {
                    completion_type: CompletionType::Throw,
                    value: Some(error_to_js_value(&err, ctx)),
                    target: None,
                };
                handle_catch(throw_completion, catch_clause, ctx)?
            } else {
                return Err(err);
            }
        }
    };

    // Execute the finally block if present
    if let Some(finally_block) = finalizer {
        let finally_result = execute_block_statement(finally_block, ctx)?;

        // If finally has an abrupt completion, it overrides the try/catch result
        if finally_result.is_abrupt() {
            completion = finally_result;
        }
    }

    Ok(completion)
}

/// Handle a catch clause.
fn handle_catch(
    thrown: Completion,
    catch_clause: &CatchClauseData,
    ctx: &mut EvalContext,
) -> EvalResult {
    // Create a new block scope for the catch clause
    ctx.push_block_scope();

    // Bind the error to the catch parameter — unless this is the optional
    // catch binding `catch { … }` (ES2019), which introduces no binding.
    if let Some(param) = &catch_clause.param {
        let param_name = get_binding_name(param)?;
        let error_value = thrown.value.unwrap_or(JsValue::Undefined);
        ctx.create_binding(&param_name, false)?;
        ctx.initialize_binding(&param_name, error_value)?;
    }

    // Execute the catch block
    let result = execute_block_statement(&catch_clause.body, ctx);

    // Pop the catch scope
    ctx.pop_block_scope();

    result
}

/// Convert a JErrorType to a JsValue for catching.
///
/// Must produce a **real** Error object (with the correct `[[Prototype]]`) so
/// `catch (e) { e instanceof ReferenceError }` works — test262 and real pages
/// all key off that. A string `"ReferenceError: …"` makes every such check fail
/// and the test then `throw new Test262Error(...)` at top-level.
fn error_to_js_value(err: &JErrorType, ctx: &mut EvalContext) -> JsValue {
    use crate::runner::eval::expression::{make_object, set_own_prop};
    let (name, msg): (&str, String) = match err {
        JErrorType::TypeError(m) => ("TypeError", m.clone()),
        JErrorType::ReferenceError(m) => ("ReferenceError", m.clone()),
        JErrorType::SyntaxError(m) => ("SyntaxError", m.clone()),
        JErrorType::RangeError(m) => ("RangeError", m.clone()),
        JErrorType::YieldValue(v) => return v.clone(),
        // A user `throw <value>` — hand the ORIGINAL value to `catch (e)`.
        JErrorType::Thrown(v) => return v.clone(),
    };
    // Prefer the registered constructor so [[Prototype]] + instanceof work.
    let sg = ctx.super_global.clone();
    if let Some(result) = sg.borrow().call_constructor(
        name,
        ctx,
        JsValue::Undefined,
        alloc::vec![JsValue::String(msg.clone())],
    ) {
        if let Ok(v @ JsValue::Object(_)) = result {
            return v;
        }
    }
    // Fallback: plain tagged object (still better than a bare string).
    let obj = make_object(alloc::vec![
        ("name".to_string(), JsValue::String(name.to_string())),
        ("message".to_string(), JsValue::String(msg)),
    ]);
    set_own_prop(
        &obj,
        "__builtin_name__",
        JsValue::String(name.to_string()),
        false,
    );
    obj
}

/// Execute a for-in statement (iterate over enumerable property keys).
fn execute_for_in_statement(
    data: &ForIteratorData,
    ctx: &mut EvalContext,
) -> EvalResult {
    // Evaluate the right-hand side to get the object
    let obj_value = evaluate_expression(&data.right, ctx)?;

    // Get the property keys to iterate over
    let keys: Vec<String> = match &obj_value {
        JsValue::Object(obj) => {
            let obj_ref = obj.borrow();
            obj_ref.as_js_object().get_object_base().properties
                .keys()
                .filter_map(|k| match k {
                    crate::runner::ds::object_property::PropertyKey::Str(s) => Some(s.clone()),
                    _ => None,
                })
                .collect()
        }
        JsValue::String(s) => {
            // For strings, iterate over indices
            (0..s.len()).map(|i| i.to_string()).collect()
        }
        JsValue::Null | JsValue::Undefined => {
            // for-in over null/undefined produces no iterations
            return Ok(Completion::normal());
        }
        _ => Vec::new(),
    };

    // Execute the loop body for each key
    let labels = loop_labels(ctx);
    let mut v: Option<JsValue> = None;

    let fresh_scope = head_is_lexical(&data.left);
    for key in keys {
        if crate::runner::host::host_tick() {
            return Err(crate::runner::host::interrupt_error());
        }
        if fresh_scope {
            ctx.push_block_scope();
        }
        // Bind the key to the loop variable
        let bound = bind_for_iterator_variable(&data.left, JsValue::String(key), ctx);
        if let Err(e) = bound {
            if fresh_scope {
                ctx.pop_block_scope();
            }
            return Err(e);
        }

        // Execute the body
        let body = execute_statement(&data.body, ctx);
        if fresh_scope {
            ctx.pop_block_scope();
        }
        let completion = body?;

        match completion.completion_type {
            CompletionType::Break => {
                if targets_this_loop(&completion.target, &labels) {
                    return Ok(Completion::normal_with_value(
                        update_empty(completion, v).get_value(),
                    ));
                }
                return Ok(update_empty(completion, v));
            }
            CompletionType::Continue => {
                if targets_this_loop(&completion.target, &labels) {
                    if completion.value.is_some() {
                        v = completion.value;
                    }
                    continue;
                }
                return Ok(update_empty(completion, v));
            }
            CompletionType::Return | CompletionType::Throw | CompletionType::Yield => {
                return Ok(update_empty(completion, v));
            }
            CompletionType::Normal => {
                if completion.value.is_some() {
                    v = completion.value;
                }
            }
        }
    }

    Ok(Completion::normal_with_value(v.unwrap_or(JsValue::Undefined)))
}

/// Execute a for-of statement (iterate over iterable values).
fn execute_for_of_statement(
    data: &ForIteratorData,
    ctx: &mut EvalContext,
) -> EvalResult {
    use crate::runner::ds::value::JsNumberType;
    use crate::runner::ds::object_property::PropertyKey;

    // Evaluate the right-hand side to get the iterable
    let iterable_value = evaluate_expression(&data.right, ctx)?;

    // Get the values to iterate over
    let values: Vec<JsValue> = match &iterable_value {
        JsValue::Object(obj) => {
            let obj_ref = obj.borrow();
            let base = obj_ref.as_js_object().get_object_base();

            // Check for length property (array-like)
            if let Some(prop) = base.properties.get(&PropertyKey::Str("length".to_string())) {
                if let crate::runner::ds::object_property::PropertyDescriptor::Data(length_data) = prop {
                    if let JsValue::Number(JsNumberType::Integer(len)) = &length_data.value {
                        let len = *len as usize;
                        let mut vals = Vec::with_capacity(len);
                        for i in 0..len {
                            let key = PropertyKey::Str(i.to_string());
                            if let Some(prop) = base.properties.get(&key) {
                                if let crate::runner::ds::object_property::PropertyDescriptor::Data(d) = prop {
                                    vals.push(d.value.clone());
                                } else {
                                    vals.push(JsValue::Undefined);
                                }
                            } else {
                                vals.push(JsValue::Undefined);
                            }
                        }
                        drop(obj_ref);
                        return execute_for_of_with_values(data, vals, ctx);
                    }
                }
            }

            // Otherwise, iterate over enumerable properties' values
            base.properties
                .values()
                .filter_map(|prop| {
                    if let crate::runner::ds::object_property::PropertyDescriptor::Data(d) = prop {
                        if d.enumerable {
                            Some(d.value.clone())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect()
        }
        JsValue::String(s) => {
            // Iterate over characters
            s.chars().map(|c| JsValue::String(c.to_string())).collect()
        }
        JsValue::Null | JsValue::Undefined => {
            return Err(JErrorType::TypeError(
                "Cannot iterate over null or undefined".to_string(),
            ));
        }
        _ => {
            return Err(JErrorType::TypeError(
                "Object is not iterable".to_string(),
            ));
        }
    };

    execute_for_of_with_values(data, values, ctx)
}

/// Helper to execute for-of with a list of values.
fn execute_for_of_with_values(
    data: &ForIteratorData,
    values: Vec<JsValue>,
    ctx: &mut EvalContext,
) -> EvalResult {
    let labels = loop_labels(ctx);
    let mut v: Option<JsValue> = None;

    // `for (const x of …)` gets a FRESH binding per iteration. Reusing one
    // binding and assigning to it throws `'x' is set and immutable` on the
    // second element — which is how a minified bundle died at
    // `for (const o of …)` with an error naming a variable that was never
    // reassigned in the source.
    let fresh_scope = head_is_lexical(&data.left);
    for value in values {
        // `while`/`for`/`do-while` tick; these two did not, so a long or endless
        // `for…of` / `for…in` ran with the host unable to pump the UI, answer
        // Ctrl+C, or apply the script budget.
        if crate::runner::host::host_tick() {
            return Err(crate::runner::host::interrupt_error());
        }
        if fresh_scope {
            ctx.push_block_scope();
        }
        // Bind the value to the loop variable
        let bound = bind_for_iterator_variable(&data.left, value, ctx);
        if let Err(e) = bound {
            if fresh_scope {
                ctx.pop_block_scope();
            }
            return Err(e);
        }

        // Execute the body
        let body = execute_statement(&data.body, ctx);
        if fresh_scope {
            ctx.pop_block_scope();
        }
        let completion = body?;

        match completion.completion_type {
            CompletionType::Break => {
                if targets_this_loop(&completion.target, &labels) {
                    return Ok(Completion::normal_with_value(
                        update_empty(completion, v).get_value(),
                    ));
                }
                return Ok(update_empty(completion, v));
            }
            CompletionType::Continue => {
                if targets_this_loop(&completion.target, &labels) {
                    if completion.value.is_some() {
                        v = completion.value;
                    }
                    continue;
                }
                return Ok(update_empty(completion, v));
            }
            CompletionType::Return | CompletionType::Throw | CompletionType::Yield => {
                return Ok(update_empty(completion, v));
            }
            CompletionType::Normal => {
                if completion.value.is_some() {
                    v = completion.value;
                }
            }
        }
    }

    Ok(Completion::normal_with_value(v.unwrap_or(JsValue::Undefined)))
}

/// True when a `for (… of/in …)` head declares a **lexical** loop variable
/// (`let`/`const`), which per spec gets a *fresh binding per iteration*.
fn head_is_lexical(left: &VariableDeclarationOrPattern) -> bool {
    match left {
        VariableDeclarationOrPattern::VariableDeclaration(d) => {
            !matches!(d.kind, VariableDeclarationKind::Var)
        }
        VariableDeclarationOrPattern::Pattern(_) => false,
    }
}

/// Bind a value to a for-in/for-of loop variable.
fn bind_for_iterator_variable(
    left: &VariableDeclarationOrPattern,
    value: JsValue,
    ctx: &mut EvalContext,
) -> Result<(), JErrorType> {
    match left {
        VariableDeclarationOrPattern::VariableDeclaration(var_decl) => {
            // Create and initialize the variable
            for declarator in &var_decl.declarations {
                let name = get_binding_name(&declarator.id)?;
                let is_var = matches!(var_decl.kind, VariableDeclarationKind::Var);

                if is_var {
                    // A `var` loop variable is hoisted and pre-initialized to
                    // `undefined`, so `initialize_var_binding` (TDZ-style, a
                    // no-op once initialized) would leave it `undefined` every
                    // iteration — `for (var k in obj)` then binds nothing. Create
                    // + initialize only if genuinely absent; otherwise *assign*
                    // the key like the `let`/pattern paths do.
                    if !ctx.has_binding(&name) {
                        ctx.create_var_binding(&name)?;
                        ctx.initialize_var_binding(&name, value.clone())?;
                    } else {
                        ctx.set_binding(&name, value.clone())?;
                    }
                } else {
                    // let/const - create new binding each iteration
                    if !ctx.has_binding(&name) {
                        ctx.create_binding(&name, matches!(var_decl.kind, VariableDeclarationKind::Const))?;
                        ctx.initialize_binding(&name, value.clone())?;
                    } else {
                        ctx.set_binding(&name, value.clone())?;
                    }
                }
            }
        }
        VariableDeclarationOrPattern::Pattern(pattern) => {
            // Simple pattern assignment
            let name = get_binding_name(pattern)?;
            ctx.set_binding(&name, value)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod completion_tests {
    //! Tests for ES statement **completion values** (the `UpdateEmpty`
    //! semantics). `eval()` returns the completion value of the last
    //! value-producing statement; statements thread completion values with
    //! `UpdateEmpty` so an empty block/statement/declaration is an *empty*
    //! completion that does not clobber the running value, while an `if`/loop
    //! whose taken branch is empty surfaces `undefined`
    //! (`UpdateEmpty(stmt, undefined)`).
    use super::*;
    use crate::parser::JsParser;
    use crate::runner::ds::value::JsNumberType;
    use crate::runner::plugin::registry::BuiltInRegistry;
    use crate::runner::plugin::types::EvalContext;

    /// Evaluate a program and return its completion value — the same threading
    /// `eval` performs (keep the last *non-empty* completion value), in the same
    /// environment (core built-ins installed, `var` hoisting).
    fn completion_value(code: &str) -> JsValue {
        let ast = JsParser::parse_to_ast_from_str(code)
            .unwrap_or_else(|e| panic!("parse error for {:?}: {:?}", code, e));
        let mut ctx = EvalContext::new();
        ctx.install_core_builtins(BuiltInRegistry::with_core());
        hoist_var_declarations(&ast.body, &mut ctx);
        let mut last = JsValue::Undefined;
        for stmt in &ast.body {
            let c = execute_statement(stmt, &mut ctx)
                .unwrap_or_else(|e| panic!("runtime error for {:?}: {:?}", code, e));
            if let Some(v) = c.value.clone() {
                last = v;
            }
        }
        last
    }

    fn int(n: i64) -> JsValue {
        JsValue::Number(JsNumberType::Integer(n))
    }

    // --- `if` (UpdateEmpty(stmt, undefined)) ---

    #[test]
    fn if_taken_empty_branch_is_undefined() {
        assert_eq!(completion_value("1; if (false) { } else { }"), JsValue::Undefined);
        assert_eq!(completion_value("1; if (true) { } else { }"), JsValue::Undefined);
        // Empty statement body: `if (1) ;`
        assert_eq!(completion_value("if (1) ;"), JsValue::Undefined);
    }

    #[test]
    fn if_nonempty_branch_keeps_value() {
        assert_eq!(completion_value("2; if (false) { } else { 3; }"), int(3));
        assert_eq!(completion_value("2; if (true) { 3; } else { }"), int(3));
        assert_eq!(completion_value("6; if (false) { 7; } else { 8; }"), int(8));
        assert_eq!(completion_value("6; if (true) { 7; } else { 8; }"), int(7));
    }

    #[test]
    fn if_no_else_false_is_undefined() {
        assert_eq!(completion_value("1; if (false) { }"), JsValue::Undefined);
        assert_eq!(completion_value("2; if (false) { 3; }"), JsValue::Undefined);
    }

    #[test]
    fn if_no_else_true_updates_empty() {
        assert_eq!(completion_value("1; if (true) { }"), JsValue::Undefined);
        assert_eq!(completion_value("2; if (true) { 3; }"), int(3));
    }

    // --- block / statement-list threading ---

    #[test]
    fn empty_block_preserves_prior_value() {
        assert_eq!(completion_value("1; { }"), int(1));
    }

    #[test]
    fn block_threads_last_nonempty_value() {
        assert_eq!(completion_value("{ 1; 2; }"), int(2));
        // A trailing declaration is empty and preserves the prior value.
        assert_eq!(completion_value("{ 1; var x = 2; }"), int(1));
        // A trailing empty nested block preserves it too.
        assert_eq!(completion_value("{ 5; { } }"), int(5));
    }

    #[test]
    fn abrupt_block_inherits_running_value() {
        // `{ 3; break; }` breaks carrying the running value 3.
        assert_eq!(
            completion_value("do { 2; if (true) { 3; break; } 4; } while (false)"),
            int(3),
        );
    }

    // --- while / do-while ---

    #[test]
    fn while_no_iteration_is_undefined() {
        assert_eq!(completion_value("1; while (false) { }"), JsValue::Undefined);
        assert_eq!(completion_value("2; while (false) { 3; }"), JsValue::Undefined);
    }

    #[test]
    fn while_iterates_threads_value() {
        assert_eq!(completion_value("var c = 2; 1; while (c -= 1) { }"), JsValue::Undefined);
        assert_eq!(completion_value("var c = 2; 2; while (c -= 1) { 3; }"), int(3));
    }

    #[test]
    fn while_break_is_update_empty() {
        assert_eq!(completion_value("1; while (true) { break; }"), JsValue::Undefined);
        assert_eq!(completion_value("2; while (true) { 3; break; }"), int(3));
    }

    #[test]
    fn do_while_if_empty_branch_break() {
        assert_eq!(
            completion_value("1; do { if (false) { } else { break; } } while (false)"),
            JsValue::Undefined,
        );
        assert_eq!(
            completion_value("6; do { 7; if (false) { 8; } else { break; } } while (false)"),
            JsValue::Undefined,
        );
    }

    #[test]
    fn do_while_continue_threads_value() {
        assert_eq!(
            completion_value("8; do { 9; if (true) { 10; continue; } 11; } while (false)"),
            int(10),
        );
        assert_eq!(
            completion_value("12; do { 13; if (true) { continue; } 14; } while (false)"),
            JsValue::Undefined,
        );
    }

    // --- for / for-of ---

    #[test]
    fn for_completion_values() {
        assert_eq!(completion_value("1; for (var i = 0; i < 0; i++) { }"), JsValue::Undefined);
        assert_eq!(completion_value("for (var i = 0; i < 3; i++) { i; }"), int(2));
        assert_eq!(completion_value("9; for (var i = 0; i < 2; i++) { }"), JsValue::Undefined);
    }

    #[test]
    fn for_of_empty_body_is_undefined() {
        // An empty-body for-of yields undefined (via the loop's final
        // `UpdateEmpty`), not the prior value — regardless of iteration count.
        // (Body-value threading is covered by `for_completion_values`; for-of's
        // value extraction has unrelated pre-existing quirks.)
        assert_eq!(completion_value("7; for (var x of [1, 2]) { }"), JsValue::Undefined);
        assert_eq!(completion_value("8; for (var x of []) { }"), JsValue::Undefined);
    }

    // --- break/continue label normalization (defends against the parser
    //     capturing a bare `break;`/`continue;` terminator as a label) ---

    #[test]
    fn sanitize_label_rejects_non_identifier() {
        assert_eq!(sanitize_label(Some(";".to_string())), None);
        assert_eq!(sanitize_label(Some(String::new())), None);
        assert_eq!(sanitize_label(None), None);
        assert_eq!(sanitize_label(Some("outer".to_string())), Some("outer".to_string()));
        assert_eq!(sanitize_label(Some("_x".to_string())), Some("_x".to_string()));
        assert_eq!(sanitize_label(Some("$y".to_string())), Some("$y".to_string()));
    }

    #[test]
    fn labelled_break_and_continue_still_work() {
        // `break outer` exits both loops; the labelled construct completes normally.
        assert_eq!(
            completion_value("2; outer: while (true) { while (true) { break outer; } }"),
            JsValue::Undefined,
        );
        // A labelled block whose `break L` carries the running value.
        assert_eq!(completion_value("5; L: { 6; break L; 7; }"), int(6));
    }
}

/// Runtime-behavior tests for the real-world-JS engine fixes (found via the
/// browser-sim reproducing css3test.com's bliss.js hang): `for (var k in obj)`
/// binding, the `arguments` object, and builtin prototype methods as
/// first-class values (`Object.prototype.toString.call`, `[].slice.call`).
#[cfg(test)]
mod realworld_engine_tests {
    use crate::parser::JsParser;
    use crate::runner::ds::value::JsValue;
    use crate::runner::eval::statement::{execute_statement, hoist_var_declarations};
    use crate::runner::plugin::registry::BuiltInRegistry;
    use crate::runner::plugin::types::EvalContext;

    /// Run `code`, return the last statement's completion value as a display
    /// string (mirrors how a script's result surfaces).
    fn run(code: &str) -> String {
        let ast = JsParser::parse_to_ast_from_str(code)
            .unwrap_or_else(|e| panic!("parse error for {:?}: {:?}", code, e));
        let mut ctx = EvalContext::new();
        ctx.install_core_builtins(BuiltInRegistry::with_core());
        hoist_var_declarations(&ast.body, &mut ctx);
        let mut last = JsValue::Undefined;
        for stmt in &ast.body {
            let c = execute_statement(stmt, &mut ctx)
                .unwrap_or_else(|e| panic!("runtime error for {:?}: {:?}", code, e));
            if let Some(v) = c.value {
                last = v;
            }
        }
        match last {
            JsValue::String(s) => s,
            JsValue::Boolean(b) => b.to_string(),
            JsValue::BigInt(b) => b.to_string(),
            other => other.to_string(),
        }
    }

    /// Sort the chars of a string — for order-independent key-set assertions
    /// (property enumeration order is unspecified in this engine).
    fn sorted(s: &str) -> alloc::string::String {
        let mut v: alloc::vec::Vec<char> = s.chars().collect();
        v.sort_unstable();
        v.into_iter().collect()
    }

    #[test]
    fn for_in_var_binds_keys() {
        // The bliss hang: `for (var k in obj)` left `k` at its hoisted
        // `undefined` (an `extend`-style property copy then infinite-recursed).
        // Enumeration order is unspecified, so compare the key *set*.
        assert_eq!(sorted(&run("var s=''; for (var k in {a:1,b:2,c:3}) s+=k; s;")), "abc");
        // Passing the loop var into a call must carry the key, not undefined.
        assert_eq!(
            sorted(&run("var out=[]; function note(x){out.push(x);} for (var p in {x:1,y:2}) note(p); out.join('');")),
            "xy"
        );
        // A nested `for (var k in …)` gets its own key sequence (not undefined).
        assert_eq!(
            sorted(&run("var r=''; for (var i in {a:1}) { for (var j in {b:2,c:3}) r+=i+j; } r;")),
            "aabc"
        );
    }

    #[test]
    fn arguments_object() {
        assert_eq!(run("(function(){ return arguments.length; })(1,2,3);"), "3");
        assert_eq!(run("(function(){ return arguments[1]; })('a','b','c');"), "b");
        // `apply` forwards the arguments object.
        assert_eq!(
            run("function inner(){return arguments.length;} function f(){return inner.apply(null, arguments);} f(1,2,3,4);"),
            "4"
        );
        // A nested (non-arrow) function has its OWN arguments, not the outer's.
        assert_eq!(
            run("function outer(){ return (function(){ return arguments.length; })(9,8); }; outer(1);"),
            "2"
        );
        // An arrow inherits the enclosing function's arguments.
        assert_eq!(
            run("function f(){ var g = () => arguments.length; return g(); } f(1,2,3);"),
            "3"
        );
    }

    #[test]
    fn builtin_methods_are_first_class_values() {
        // The `Object.prototype.toString.call(x)` type-detection idiom.
        assert_eq!(run("Object.prototype.toString.call({});"), "[object Object]");
        assert_eq!(run("Object.prototype.toString.call([]);"), "[object Array]");
        assert_eq!(run("Object.prototype.toString.call(function(){});"), "[object Function]");
        assert_eq!(run("Object.prototype.toString.call(/x/);"), "[object RegExp]");
        // `Array.prototype.slice.call(arguments)` / `[].slice.call(...)`.
        assert_eq!(run("JSON.stringify(Array.prototype.slice.call([1,2,3,4], 1, 3));"), "[2,3]");
        assert_eq!(run("JSON.stringify([].slice.call([9,8,7], 1));"), "[8,7]");
        // A builtin method pulled off as a value is a function; `.call` works.
        assert_eq!(run("typeof [].push;"), "function");
        assert_eq!(run("typeof function(){};"), "function");
        assert_eq!(run("Object.prototype.hasOwnProperty.call({a:1}, 'a');"), "true");
        assert_eq!(run("'hello'.charAt.call('world', 0);"), "w");
    }

    #[test]
    fn boxed_primitive_still_unwraps() {
        // Regression guard: making builtin `valueOf`/`toString` first-class must
        // not mask a boxed primitive's ToPrimitive (`Object(2n) + 1n === 3n`).
        assert_eq!(run("Object(2n) + 1n;"), "3");
        assert_eq!(run("Object(2n) + 1n === 3n;"), "true");
        // A plain object still stringifies via ToPrimitive → toString.
        assert_eq!(run("'' + {};"), "[object Object]");
        assert_eq!(run("({} + {});"), "[object Object][object Object]");
        assert_eq!(run("1e-1 === 0.1;"), "true");
        assert_eq!(run("1 / (-0 + -0) === Number.NEGATIVE_INFINITY;"), "true");
        // GetMethod: non-callable @@toPrimitive must TypeError (not fall through).
        assert_eq!(
            run("try { ({[Symbol.toPrimitive]: 1}) + 0; 'no' } catch (e) { e.name }"),
            "TypeError"
        );
        // Accessor-defined @@toPrimitive must run the getter (defineProperty path).
        assert_eq!(
            run(
                r#"
                var log = '';
                var o = {};
                Object.defineProperty(o, Symbol.toPrimitive, {
                  get: function () { log += 'g'; return function () { return 7; }; }
                });
                var v = o + 1;
                log + ':' + v;
                "#
            ),
            "g:8"
        );
        // Getter that throws must surface that throw (not a scope ReferenceError).
        assert_eq!(
            run(
                r#"
                var hit = false;
                var o = {};
                Object.defineProperty(o, Symbol.toPrimitive, {
                  get: function () {
                    hit = true;
                    return function () { throw new Error('boom'); };
                  }
                });
                try { o + 1; 'no'; } catch (e) { (hit ? 'hit:' : 'miss:') + e.message }
                "#
            ),
            "hit:boom"
        );
        // Keyword/identifier boundary: `thrower` is an identifier, not `throw er`.
        assert_eq!(run("var thrower = 3; thrower + 1;"), "4");
        assert_eq!(run("var typeofx = 9; typeofx;"), "9");
    }
}
