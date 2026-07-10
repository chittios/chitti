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

        StatementType::BreakStatement { .. } => {
            Ok(Completion::break_completion(None))
        }

        StatementType::ContinueStatement { .. } => {
            Ok(Completion::continue_completion(None))
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

/// Execute a block statement.
fn execute_block_statement(
    block: &BlockStatementData,
    ctx: &mut EvalContext,
) -> EvalResult {
    // Create a new block scope for let/const bindings
    ctx.push_block_scope();

    let mut completion = Completion::normal();

    for stmt in block.body.iter() {
        completion = execute_statement(stmt, ctx)?;

        if completion.is_abrupt() {
            ctx.pop_block_scope();
            return Ok(completion);
        }
    }

    // Pop the block scope
    ctx.pop_block_scope();

    Ok(completion)
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

        // Bind the pattern to the value
        bind_pattern(&declarator.id, value, ctx, is_const, is_var)?;
    }

    Ok(Completion::normal())
}

/// Bind a pattern to a value, creating bindings for all identifiers in the pattern.
fn bind_pattern(
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

        PatternType::ObjectPattern { properties, .. } => {
            // Object destructuring: { x, y } = obj or { x: renamed } = obj
            for prop in properties {
                let prop_data = &prop.0;

                // Get the property key name
                let key_name = get_property_key_name(&prop_data.key)?;

                // Get the value from the object
                let prop_value = get_property_value(&value, &key_name)?;

                // Bind the value to the pattern
                // For shorthand like { x }, value is the same pattern as key
                // For renamed like { x: renamed }, value is the renamed pattern
                bind_pattern(&prop_data.value, prop_value, ctx, is_const, is_var)?;
            }
            Ok(())
        }

        PatternType::ArrayPattern { elements, .. } => {
            // Array destructuring: [a, b] = arr
            for (index, element) in elements.iter().enumerate() {
                if let Some(elem_pattern) = element {
                    // Check if it's a rest element
                    if let PatternType::RestElement { argument, .. } = elem_pattern.as_ref() {
                        // Rest element collects remaining elements into an array
                        let rest_value = get_rest_elements(&value, index)?;
                        bind_pattern(argument, rest_value, ctx, is_const, is_var)?;
                    } else {
                        // Regular element
                        let elem_value = get_array_element(&value, index)?;
                        bind_pattern(elem_pattern, elem_value, ctx, is_const, is_var)?;
                    }
                }
                // None means hole/skip - do nothing
            }
            Ok(())
        }

        PatternType::AssignmentPattern { left, right, .. } => {
            // Default value pattern: x = default or { x = default } = obj
            // Use default if value is undefined
            let actual_value = if matches!(value, JsValue::Undefined) {
                evaluate_expression(right, ctx)?
            } else {
                value
            };
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

/// Get a property value from an object (for destructuring).
fn get_property_value(obj: &JsValue, key: &str) -> Result<JsValue, JErrorType> {
    use crate::runner::ds::object_property::{PropertyDescriptor, PropertyKey};

    match obj {
        JsValue::Object(obj_ref) => {
            let borrowed = obj_ref.borrow();
            let base = borrowed.as_js_object().get_object_base();
            let prop_key = PropertyKey::Str(key.to_string());

            if let Some(PropertyDescriptor::Data(data)) = base.properties.get(&prop_key) {
                Ok(data.value.clone())
            } else {
                Ok(JsValue::Undefined)
            }
        }
        _ => Err(JErrorType::TypeError("Cannot destructure non-object".to_string())),
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
fn get_rest_elements(arr: &JsValue, start_index: usize) -> Result<JsValue, JErrorType> {
    use crate::runner::ds::object::{JsObject, JsObjectType, ObjectType};
    use crate::runner::ds::object_property::{PropertyDescriptor, PropertyDescriptorData, PropertyKey};
    use crate::runner::ds::value::JsNumberType;
    use crate::runner::plugin::types::SimpleObject;
    use core::cell::RefCell;
    use alloc::rc::Rc;

    match arr {
        JsValue::Object(obj_ref) => {
            let borrowed = obj_ref.borrow();
            let base = borrowed.as_js_object().get_object_base();

            // Get original array length
            let length = if let Some(PropertyDescriptor::Data(data)) =
                base.properties.get(&PropertyKey::Str("length".to_string()))
            {
                match &data.value {
                    JsValue::Number(JsNumberType::Integer(n)) => *n as usize,
                    JsValue::Number(JsNumberType::Float(n)) => *n as usize,
                    _ => 0,
                }
            } else {
                0
            };

            // Create a new array with remaining elements
            let mut rest_obj = SimpleObject::new();
            let mut rest_index = 0;

            for i in start_index..length {
                let prop_key = PropertyKey::Str(i.to_string());
                if let Some(PropertyDescriptor::Data(data)) = base.properties.get(&prop_key) {
                    rest_obj.get_object_base_mut().properties.insert(
                        PropertyKey::Str(rest_index.to_string()),
                        PropertyDescriptor::Data(PropertyDescriptorData {
                            value: data.value.clone(),
                            writable: true,
                            enumerable: true,
                            configurable: true,
                        }),
                    );
                    rest_index += 1;
                }
            }

            // Set length
            rest_obj.get_object_base_mut().properties.insert(
                PropertyKey::Str("length".to_string()),
                PropertyDescriptor::Data(PropertyDescriptorData {
                    value: JsValue::Number(JsNumberType::Integer(rest_index as i64)),
                    writable: true,
                    enumerable: false,
                    configurable: false,
                }),
            );

            let obj: JsObjectType = Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(rest_obj))));
            Ok(JsValue::Object(obj))
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

    if to_boolean(&test_value) {
        execute_statement(consequent, ctx)
    } else if let Some(alt) = alternate {
        execute_statement(alt, ctx)
    } else {
        Ok(Completion::normal())
    }
}

/// Execute a while statement.
fn execute_while_statement(
    test: &ExpressionType,
    body: &StatementType,
    ctx: &mut EvalContext,
) -> EvalResult {
    let mut completion = Completion::normal();

    loop {
        let test_value = evaluate_expression(test, ctx)?;

        if !to_boolean(&test_value) {
            break;
        }

        completion = execute_statement(body, ctx)?;

        match completion.completion_type {
            CompletionType::Break => {
                return Ok(Completion::normal_with_value(completion.get_value()));
            }
            CompletionType::Continue => {
                continue;
            }
            CompletionType::Return | CompletionType::Throw | CompletionType::Yield => {
                return Ok(completion);
            }
            CompletionType::Normal => {}
        }
    }

    Ok(completion)
}

/// Execute a do-while statement.
fn execute_do_while_statement(
    body: &StatementType,
    test: &ExpressionType,
    ctx: &mut EvalContext,
) -> EvalResult {
    loop {
        let completion = execute_statement(body, ctx)?;

        match completion.completion_type {
            CompletionType::Break => {
                return Ok(Completion::normal_with_value(completion.get_value()));
            }
            CompletionType::Continue => {}
            CompletionType::Return | CompletionType::Throw | CompletionType::Yield => {
                return Ok(completion);
            }
            CompletionType::Normal => {}
        }

        let test_value = evaluate_expression(test, ctx)?;
        if !to_boolean(&test_value) {
            break;
        }
    }

    Ok(Completion::normal())
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

    let mut completion = Completion::normal();

    loop {
        if let Some(test) = test {
            let test_value = evaluate_expression(test, ctx)?;
            if !to_boolean(&test_value) {
                break;
            }
        }

        completion = execute_statement(body, ctx)?;

        match completion.completion_type {
            CompletionType::Break => {
                return Ok(Completion::normal_with_value(completion.get_value()));
            }
            CompletionType::Continue => {}
            CompletionType::Return | CompletionType::Throw | CompletionType::Yield => {
                return Ok(completion);
            }
            CompletionType::Normal => {}
        }

        if let Some(update) = update {
            evaluate_expression(update, ctx)?;
        }
    }

    Ok(completion)
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
                        // Break exits the switch
                        return Ok(Completion::normal_with_value(completion.get_value()));
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
            // Runtime error - treat as throw
            if let Some(catch_clause) = handler {
                let throw_completion = Completion {
                    completion_type: CompletionType::Throw,
                    value: Some(error_to_js_value(&err)),
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

    // Bind the error to the catch parameter
    let param_name = get_binding_name(&catch_clause.param)?;
    let error_value = thrown.value.unwrap_or(JsValue::Undefined);

    ctx.create_binding(&param_name, false)?;
    ctx.initialize_binding(&param_name, error_value)?;

    // Execute the catch block
    let result = execute_block_statement(&catch_clause.body, ctx);

    // Pop the catch scope
    ctx.pop_block_scope();

    result
}

/// Convert a JErrorType to a JsValue for catching.
fn error_to_js_value(err: &JErrorType) -> JsValue {
    match err {
        JErrorType::TypeError(msg) => JsValue::String(format!("TypeError: {}", msg)),
        JErrorType::ReferenceError(msg) => JsValue::String(format!("ReferenceError: {}", msg)),
        JErrorType::SyntaxError(msg) => JsValue::String(format!("SyntaxError: {}", msg)),
        JErrorType::RangeError(msg) => JsValue::String(format!("RangeError: {}", msg)),
        JErrorType::YieldValue(v) => v.clone(),
    }
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
    let mut completion = Completion::normal();

    for key in keys {
        // Bind the key to the loop variable
        bind_for_iterator_variable(&data.left, JsValue::String(key), ctx)?;

        // Execute the body
        completion = execute_statement(&data.body, ctx)?;

        match completion.completion_type {
            CompletionType::Break => {
                return Ok(Completion::normal_with_value(completion.get_value()));
            }
            CompletionType::Continue => {
                continue;
            }
            CompletionType::Return | CompletionType::Throw | CompletionType::Yield => {
                return Ok(completion);
            }
            CompletionType::Normal => {}
        }
    }

    Ok(completion)
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
    let mut completion = Completion::normal();

    for value in values {
        // Bind the value to the loop variable
        bind_for_iterator_variable(&data.left, value, ctx)?;

        // Execute the body
        completion = execute_statement(&data.body, ctx)?;

        match completion.completion_type {
            CompletionType::Break => {
                return Ok(Completion::normal_with_value(completion.get_value()));
            }
            CompletionType::Continue => {
                continue;
            }
            CompletionType::Return | CompletionType::Throw | CompletionType::Yield => {
                return Ok(completion);
            }
            CompletionType::Normal => {}
        }
    }

    Ok(completion)
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
                    // Check if already exists
                    if !ctx.has_binding(&name) {
                        ctx.create_var_binding(&name)?;
                    }
                    ctx.initialize_var_binding(&name, value.clone())?;
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
