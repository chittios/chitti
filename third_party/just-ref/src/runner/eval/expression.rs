//! Expression evaluation.
//!
//! This module provides the core expression evaluation logic for the JavaScript interpreter.
//! It handles all expression types defined in the AST.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::parser::ast::{
    AssignmentOperator, BinaryOperator, ClassData, ExpressionOrSpreadElement, ExpressionOrSuper,
    ExpressionType, ExpressionPatternType, FunctionData, LiteralData, LiteralType, MemberExpressionType,
    MethodDefinitionKind, PatternOrExpression, PatternType, PropertyData, PropertyKind, UnaryOperator,
    UpdateOperator, LogicalOperator,
};
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::object::{JsObject, JsObjectType, ObjectBase, ObjectType};
use crate::runner::ds::object_property::{PropertyDescriptor, PropertyDescriptorAccessor, PropertyDescriptorData, PropertyKey};
use crate::runner::ds::value::{JsValue, JsNumberType};
use crate::runner::plugin::types::{EvalContext, SimpleObject};

use core::cell::RefCell;
use alloc::rc::Rc;

use super::types::ValueResult;

/// Evaluate an expression and return its value.
pub fn evaluate_expression(
    expr: &ExpressionType,
    ctx: &mut EvalContext,
) -> ValueResult {
    match expr {
        ExpressionType::Literal(lit) => evaluate_literal(lit),

        ExpressionType::ExpressionWhichCanBePattern(pattern) => {
            evaluate_expression_pattern(pattern, ctx)
        }

        ExpressionType::ThisExpression { .. } => {
            Ok(ctx.global_this.clone().unwrap_or(JsValue::Undefined))
        }

        ExpressionType::ArrayExpression { elements, .. } => {
            evaluate_array_expression(elements, ctx)
        }

        ExpressionType::ObjectExpression { properties, .. } => {
            evaluate_object_expression(properties, ctx)
        }

        ExpressionType::FunctionOrGeneratorExpression(func_data) => {
            create_function_object(func_data, ctx)
        }

        ExpressionType::UnaryExpression { operator, argument, .. } => {
            evaluate_unary_expression(operator, argument, ctx)
        }

        ExpressionType::BinaryExpression { operator, left, right, .. } => {
            evaluate_binary_expression(operator, left, right, ctx)
        }

        ExpressionType::LogicalExpression { operator, left, right, .. } => {
            evaluate_logical_expression(operator, left, right, ctx)
        }

        ExpressionType::UpdateExpression { operator, argument, prefix, .. } => {
            evaluate_update_expression(operator, argument, *prefix, ctx)
        }

        ExpressionType::AssignmentExpression { operator, left, right, .. } => {
            evaluate_assignment_expression(operator, left, right, ctx)
        }

        ExpressionType::ConditionalExpression { test, consequent, alternate, .. } => {
            evaluate_conditional_expression(test, consequent, alternate, ctx)
        }

        ExpressionType::CallExpression { callee, arguments, .. } => {
            evaluate_call_expression(callee, arguments, ctx)
        }

        ExpressionType::NewExpression { callee, arguments, .. } => {
            evaluate_new_expression(callee, arguments, ctx)
        }

        ExpressionType::SequenceExpression { expressions, .. } => {
            evaluate_sequence_expression(expressions, ctx)
        }

        ExpressionType::TemplateLiteral(_) => {
            Err(JErrorType::TypeError("Template literal not yet implemented".to_string()))
        }

        ExpressionType::TaggedTemplateExpression { .. } => {
            Err(JErrorType::TypeError("Tagged template expression not yet implemented".to_string()))
        }

        ExpressionType::ClassExpression(class_data) => {
            evaluate_class_expression(class_data, ctx)
        }

        ExpressionType::YieldExpression { argument, delegate, .. } => {
            if *delegate {
                // yield* not yet supported
                return Err(JErrorType::TypeError("yield* not yet implemented".to_string()));
            }
            // Evaluate the argument
            let value = if let Some(arg) = argument {
                evaluate_expression(arg, ctx)?
            } else {
                JsValue::Undefined
            };
            // Return a special error that signals a yield
            Err(JErrorType::YieldValue(value))
        }

        ExpressionType::MetaProperty { .. } => {
            Err(JErrorType::TypeError("Meta property not yet implemented".to_string()))
        }

        ExpressionType::ArrowFunctionExpression { params, body, is_async, .. } => {
            create_arrow_function_object(params, body, *is_async, ctx)
        }

        ExpressionType::MemberExpression(member_expr) => {
            evaluate_member_expression(member_expr, ctx)
        }

        ExpressionType::OptionalChain { object, access, .. } => {
            evaluate_optional_chain(object, access, ctx)
        }
    }
}

/// Evaluate a literal and return its value.
fn evaluate_literal(lit: &LiteralData) -> ValueResult {
    Ok(match &lit.value {
        LiteralType::NullLiteral => JsValue::Null,
        LiteralType::BooleanLiteral(b) => JsValue::Boolean(*b),
        LiteralType::StringLiteral(s) => JsValue::String(s.clone()),
        LiteralType::NumberLiteral(n) => {
            use crate::parser::ast::NumberLiteralType;
            match n {
                NumberLiteralType::IntegerLiteral(i) => JsValue::Number(JsNumberType::Integer(*i)),
                NumberLiteralType::FloatLiteral(f) => JsValue::Number(JsNumberType::Float(*f)),
            }
        }
        LiteralType::RegExpLiteral(data) => {
            crate::runner::std_lib::regexp::make_regexp(&data.pattern, &data.flags)
        }
    })
}

/// Evaluate an array expression and return an array object.
fn evaluate_array_expression(
    elements: &[Option<ExpressionOrSpreadElement>],
    ctx: &mut EvalContext,
) -> ValueResult {
    use crate::runner::ds::object::JsObject;

    // Create a new array object (tracked for heap accounting)
    let mut array_obj = ctx.new_tracked_object()?;

    let mut index = 0;
    for element in elements {
        if let Some(elem) = element {
            match elem {
                ExpressionOrSpreadElement::Expression(expr) => {
                    let value = evaluate_expression(expr, ctx)?;
                    // Define the element as a property
                    let key = PropertyKey::Str(index.to_string());
                    array_obj.get_object_base_mut().properties.insert(
                        key,
                        PropertyDescriptor::Data(PropertyDescriptorData {
                            value,
                            writable: true,
                            enumerable: true,
                            configurable: true,
                        }),
                    );
                    index += 1;
                }
                ExpressionOrSpreadElement::SpreadElement(spread_expr) => {
                    // Evaluate the spread expression
                    let spread_value = evaluate_expression(spread_expr, ctx)?;

                    // If it's an array, spread its elements
                    if let JsValue::Object(spread_obj) = spread_value {
                        let borrowed = spread_obj.borrow();
                        let base = borrowed.as_js_object().get_object_base();

                        // Get length property if it exists
                        let length = if let Some(PropertyDescriptor::Data(prop)) =
                            base.properties.get(&PropertyKey::Str("length".to_string()))
                        {
                            match &prop.value {
                                JsValue::Number(JsNumberType::Integer(n)) => *n as usize,
                                JsValue::Number(JsNumberType::Float(n)) => *n as usize,
                                _ => 0,
                            }
                        } else {
                            0
                        };

                        // Iterate over array indices
                        for i in 0..length {
                            let elem_key = PropertyKey::Str(i.to_string());
                            let elem_value = if let Some(PropertyDescriptor::Data(prop)) =
                                base.properties.get(&elem_key)
                            {
                                prop.value.clone()
                            } else {
                                JsValue::Undefined
                            };

                            // Add to result array
                            let key = PropertyKey::Str(index.to_string());
                            array_obj.get_object_base_mut().properties.insert(
                                key,
                                PropertyDescriptor::Data(PropertyDescriptorData {
                                    value: elem_value,
                                    writable: true,
                                    enumerable: true,
                                    configurable: true,
                                }),
                            );
                            index += 1;
                        }
                    } else if let JsValue::String(s) = spread_value {
                        // Spread string characters
                        for ch in s.chars() {
                            let key = PropertyKey::Str(index.to_string());
                            array_obj.get_object_base_mut().properties.insert(
                                key,
                                PropertyDescriptor::Data(PropertyDescriptorData {
                                    value: JsValue::String(ch.to_string()),
                                    writable: true,
                                    enumerable: true,
                                    configurable: true,
                                }),
                            );
                            index += 1;
                        }
                    } else {
                        return Err(JErrorType::TypeError(
                            "Spread requires an iterable".to_string(),
                        ));
                    }
                }
            }
        } else {
            // Holes (None elements) are left as undefined/missing
            index += 1;
        }
    }

    // Set the length property
    array_obj.get_object_base_mut().properties.insert(
        PropertyKey::Str("length".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Number(JsNumberType::Integer(index as i64)),
            writable: true,
            enumerable: false,
            configurable: false,
        }),
    );
    // ChittiOS: mark as an Array so instance-method dispatch + Array.isArray can
    // distinguish arrays from plain objects (they share the Ordinary rep).
    array_obj.get_object_base_mut().properties.insert(
        PropertyKey::Str("__array__".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Boolean(true),
            writable: false,
            enumerable: false,
            configurable: false,
        }),
    );

    // Wrap in JsObjectType
    let obj: JsObjectType = Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(array_obj))));
    Ok(JsValue::Object(obj))
}

// ============================================================================
// ChittiOS: Array/object value helpers shared with std_lib (arrays are Ordinary
// objects with indexed string keys + `length` + an `__array__` marker).
// ============================================================================

/// Build a JS Array value from a Rust Vec of elements.
pub fn make_array(elems: Vec<JsValue>) -> JsValue {
    use crate::runner::ds::object::JsObject;
    let mut obj = SimpleObject::new();
    for (i, v) in elems.iter().enumerate() {
        obj.get_object_base_mut().properties.insert(
            PropertyKey::Str(i.to_string()),
            PropertyDescriptor::Data(PropertyDescriptorData {
                value: v.clone(),
                writable: true,
                enumerable: true,
                configurable: true,
            }),
        );
    }
    obj.get_object_base_mut().properties.insert(
        PropertyKey::Str("length".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Number(JsNumberType::Integer(elems.len() as i64)),
            writable: true,
            enumerable: false,
            configurable: false,
        }),
    );
    obj.get_object_base_mut().properties.insert(
        PropertyKey::Str("__array__".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Boolean(true),
            writable: false,
            enumerable: false,
            configurable: false,
        }),
    );
    JsValue::Object(Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(obj)))))
}

/// Build a JS object from (key, value) pairs.
pub fn make_object(pairs: Vec<(String, JsValue)>) -> JsValue {
    use crate::runner::ds::object::JsObject;
    let mut obj = SimpleObject::new();
    for (k, v) in pairs {
        obj.get_object_base_mut().properties.insert(
            PropertyKey::Str(k),
            PropertyDescriptor::Data(PropertyDescriptorData {
                value: v,
                writable: true,
                enumerable: true,
                configurable: true,
            }),
        );
    }
    JsValue::Object(Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(obj)))))
}

/// Is this value a JS Array (has the `__array__` marker)?
pub fn is_array(v: &JsValue) -> bool {
    if let JsValue::Object(o) = v {
        let b = o.borrow();
        b.as_js_object()
            .get_object_base()
            .properties
            .contains_key(&PropertyKey::Str("__array__".to_string()))
    } else {
        false
    }
}

/// Read an array's `length`.
pub fn array_len(v: &JsValue) -> usize {
    if let JsValue::Object(o) = v {
        let b = o.borrow();
        if let Some(PropertyDescriptor::Data(d)) = b
            .as_js_object()
            .get_object_base()
            .properties
            .get(&PropertyKey::Str("length".to_string()))
        {
            return match &d.value {
                JsValue::Number(JsNumberType::Integer(n)) => (*n).max(0) as usize,
                JsValue::Number(JsNumberType::Float(n)) => (*n).max(0.0) as usize,
                _ => 0,
            };
        }
    }
    0
}

/// Read every element of an array (0..length) into a Vec.
pub fn array_elements(v: &JsValue) -> Vec<JsValue> {
    let n = array_len(v);
    let mut out = Vec::with_capacity(n);
    if let JsValue::Object(o) = v {
        let b = o.borrow();
        let base = b.as_js_object().get_object_base();
        for i in 0..n {
            let val = match base.properties.get(&PropertyKey::Str(i.to_string())) {
                Some(PropertyDescriptor::Data(d)) => d.value.clone(),
                _ => JsValue::Undefined,
            };
            out.push(val);
        }
    }
    out
}

/// Overwrite an array's elements + length in place (used by in-place methods).
pub fn array_set_elements(v: &JsValue, elems: &[JsValue]) {
    if let JsValue::Object(o) = v {
        let mut b = o.borrow_mut();
        let base = b.as_js_object_mut().get_object_base_mut();
        // Remove stale higher indices.
        let old = match base.properties.get(&PropertyKey::Str("length".to_string())) {
            Some(PropertyDescriptor::Data(d)) => match &d.value {
                JsValue::Number(JsNumberType::Integer(n)) => (*n).max(0) as usize,
                _ => 0,
            },
            _ => 0,
        };
        for i in elems.len()..old {
            base.properties.remove(&PropertyKey::Str(i.to_string()));
        }
        for (i, val) in elems.iter().enumerate() {
            base.properties.insert(
                PropertyKey::Str(i.to_string()),
                PropertyDescriptor::Data(PropertyDescriptorData {
                    value: val.clone(),
                    writable: true,
                    enumerable: true,
                    configurable: true,
                }),
            );
        }
        base.properties.insert(
            PropertyKey::Str("length".to_string()),
            PropertyDescriptor::Data(PropertyDescriptorData {
                value: JsValue::Number(JsNumberType::Integer(elems.len() as i64)),
                writable: true,
                enumerable: false,
                configurable: false,
            }),
        );
    }
}

/// Evaluate an object expression and return an object.
fn evaluate_object_expression(
    properties: &[PropertyData<Box<ExpressionType>>],
    ctx: &mut EvalContext,
) -> ValueResult {
    use hashbrown::HashMap;

    // Create a new object (tracked for heap accounting)
    let mut obj = ctx.new_tracked_object()?;

    // Track accessor properties to merge getter/setter pairs
    let mut accessors: HashMap<String, (Option<JsObjectType>, Option<JsObjectType>)> = HashMap::new();

    for prop in properties {
        // `{ ...expr }` — merge the spread source's own enumerable props first
        // (before any key evaluation; the Spread key is a placeholder).
        if matches!(prop.kind, PropertyKind::Spread) {
            let src = evaluate_expression(&prop.value, ctx)?;
            match &src {
                JsValue::Null | JsValue::Undefined => {}
                JsValue::String(sv) => {
                    // Spreading a string yields index→char entries.
                    for (i, ch) in sv.chars().enumerate() {
                        obj.get_object_base_mut().properties.insert(
                            PropertyKey::Str(i.to_string()),
                            PropertyDescriptor::Data(PropertyDescriptorData {
                                value: JsValue::String(ch.to_string()),
                                writable: true,
                                enumerable: true,
                                configurable: true,
                            }),
                        );
                    }
                }
                _ => {
                    for k in own_string_keys(&src) {
                        if let Some(v) = get_own_prop_value(&src, &k) {
                            obj.get_object_base_mut().properties.insert(
                                PropertyKey::Str(k),
                                PropertyDescriptor::Data(PropertyDescriptorData {
                                    value: v,
                                    writable: true,
                                    enumerable: true,
                                    configurable: true,
                                }),
                            );
                        }
                    }
                }
            }
            continue;
        }

        // Get the property key
        let key = get_object_property_key(&prop.key, prop.computed, ctx)?;

        match prop.kind {
            PropertyKind::Init => {
                // Evaluate the value
                let value = evaluate_expression(&prop.value, ctx)?;

                // Define the property
                obj.get_object_base_mut().properties.insert(
                    PropertyKey::Str(key),
                    PropertyDescriptor::Data(PropertyDescriptorData {
                        value,
                        writable: true,
                        enumerable: true,
                        configurable: true,
                    }),
                );
            }
            PropertyKind::Get => {
                // The value should be a function expression
                if let ExpressionType::FunctionOrGeneratorExpression(func_data) = prop.value.as_ref() {
                    let getter_fn = create_function_object(func_data, ctx)?;
                    if let JsValue::Object(getter_obj) = getter_fn {
                        let entry = accessors.entry(key).or_insert((None, None));
                        entry.0 = Some(getter_obj);
                    }
                } else {
                    return Err(JErrorType::TypeError("Getter must be a function".to_string()));
                }
            }
            PropertyKind::Set => {
                // The value should be a function expression
                if let ExpressionType::FunctionOrGeneratorExpression(func_data) = prop.value.as_ref() {
                    let setter_fn = create_function_object(func_data, ctx)?;
                    if let JsValue::Object(setter_obj) = setter_fn {
                        let entry = accessors.entry(key).or_insert((None, None));
                        entry.1 = Some(setter_obj);
                    }
                } else {
                    return Err(JErrorType::TypeError("Setter must be a function".to_string()));
                }
            }
            // Handled by the early `continue` above.
            PropertyKind::Spread => unreachable!("spread handled before key eval"),
        }
    }

    // Add accessor properties to the object
    for (key, (getter, setter)) in accessors {
        obj.get_object_base_mut().properties.insert(
            PropertyKey::Str(key),
            PropertyDescriptor::Accessor(PropertyDescriptorAccessor {
                get: getter,
                set: setter,
                enumerable: true,
                configurable: true,
            }),
        );
    }

    // Wrap in JsObjectType
    let obj_ref: JsObjectType = Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(obj))));
    Ok(JsValue::Object(obj_ref))
}

/// Get the property key from an object literal property.
fn get_object_property_key(
    key_expr: &ExpressionType,
    computed: bool,
    ctx: &mut EvalContext,
) -> Result<String, JErrorType> {
    if computed {
        // Computed property: [expr]
        let key_value = evaluate_expression(key_expr, ctx)?;
        Ok(to_string(&key_value))
    } else {
        // Static property key
        match key_expr {
            ExpressionType::ExpressionWhichCanBePattern(ExpressionPatternType::Identifier(id)) => {
                Ok(id.name.clone())
            }
            ExpressionType::Literal(lit) => match &lit.value {
                LiteralType::StringLiteral(s) => Ok(s.clone()),
                LiteralType::NumberLiteral(n) => {
                    use crate::parser::ast::NumberLiteralType;
                    match n {
                        NumberLiteralType::IntegerLiteral(i) => Ok(i.to_string()),
                        NumberLiteralType::FloatLiteral(f) => Ok(f.to_string()),
                    }
                }
                _ => Err(JErrorType::TypeError("Invalid property key".to_string())),
            },
            _ => Err(JErrorType::TypeError("Invalid property key".to_string())),
        }
    }
}

/// Evaluate an expression pattern (identifier).
fn evaluate_expression_pattern(
    pattern: &ExpressionPatternType,
    ctx: &mut EvalContext,
) -> ValueResult {
    match pattern {
        ExpressionPatternType::Identifier(id) => {
            // Look up the identifier in the environment chain
            ctx.get_binding(&id.name)
        }
    }
}

/// Evaluate an optional-chaining access `obj?.x` / `obj?.[e]` / `obj?.(args)`.
/// If `obj` is `null`/`undefined` the access short-circuits to `undefined`
/// (per-step guard — the common `a?.b?.c` form is fully covered).
fn evaluate_optional_chain(
    object: &ExpressionType,
    access: &crate::parser::ast::OptionalAccess,
    ctx: &mut EvalContext,
) -> ValueResult {
    use crate::parser::ast::OptionalAccess;
    let base = evaluate_expression(object, ctx)?;
    if matches!(base, JsValue::Null | JsValue::Undefined) {
        return Ok(JsValue::Undefined);
    }
    match access {
        OptionalAccess::Member(name) => get_property_with_ctx(&base, name, ctx),
        OptionalAccess::Computed(expr) => {
            let key = evaluate_expression(expr, ctx)?;
            get_property_with_ctx(&base, &to_property_key(&key), ctx)
        }
        OptionalAccess::Call(args) => {
            let argv = evaluate_arguments(args, ctx)?;
            // `f?.(args)` — call `base` with `this = undefined` (a plain call).
            if !value_is_callable(&base) {
                return Err(JErrorType::TypeError("optional callee is not a function".to_string()));
            }
            call_value(&base, JsValue::Undefined, argv, ctx)
        }
    }
}

/// Evaluate a member expression (property access).
fn evaluate_member_expression(
    member_expr: &MemberExpressionType,
    ctx: &mut EvalContext,
) -> ValueResult {
    match member_expr {
        MemberExpressionType::SimpleMemberExpression { object, property, .. } => {
            // Evaluate the object
            let obj_value = evaluate_expression_or_super(object, ctx)?;
            let prop_name = &property.name;

            // Get the property (with getter support)
            get_property_with_ctx(&obj_value, prop_name, ctx)
        }
        MemberExpressionType::ComputedMemberExpression { object, property, .. } => {
            // Evaluate the object
            let obj_value = evaluate_expression_or_super(object, ctx)?;

            // Evaluate the property expression to get the key
            let prop_value = evaluate_expression(property, ctx)?;
            let prop_name = to_property_key(&prop_value);

            // Get the property (with getter support)
            get_property_with_ctx(&obj_value, &prop_name, ctx)
        }
    }
}

/// Evaluate an ExpressionOrSuper.
fn evaluate_expression_or_super(
    expr_or_super: &ExpressionOrSuper,
    ctx: &mut EvalContext,
) -> ValueResult {
    match expr_or_super {
        ExpressionOrSuper::Expression(expr) => evaluate_expression(expr, ctx),
        ExpressionOrSuper::Super => {
            Err(JErrorType::TypeError("super not yet supported".to_string()))
        }
    }
}

/// Evaluate a call expression (function call).
fn evaluate_call_expression(
    callee: &ExpressionOrSuper,
    arguments: &[ExpressionOrSpreadElement],
    ctx: &mut EvalContext,
) -> ValueResult {
    // ChittiOS: `super(...)` inside a derived-class constructor — invoke the
    // superclass constructor against the current `this`.
    if let ExpressionOrSuper::Super = callee {
        let args = evaluate_arguments(arguments, ctx)?;
        let this_value = ctx.global_this.clone().unwrap_or(JsValue::Undefined);
        let parent = ctx.current_super.clone().ok_or_else(|| {
            JErrorType::SyntaxError("'super' keyword unexpected here".to_string())
        })?;
        return invoke_constructor(&parent, this_value, args, ctx);
    }

    // ChittiOS: unified method-call dispatch for `obj.method(args)` (and
    // `obj[expr](args)`). One path serves both builtin statics (Math.abs,
    // console.log, Object.keys, JSON.parse — the receiver is a `__builtin_name__`
    // sentinel) AND instance methods on arrays/strings/numbers/objects
    // (`[…].push`, `"s".toUpperCase()`, `this.foo()`), which the old
    // identifier-only fast path could never reach.
    if let ExpressionOrSuper::Expression(expr) = callee {
        if let ExpressionType::MemberExpression(member) = expr.as_ref() {
            // Extract (object expression, method name). Super receivers fall
            // through to the normal path (handled by the class/super logic).
            let parts: Option<(&ExpressionOrSuper, String)> = match member {
                MemberExpressionType::SimpleMemberExpression { object, property, .. } => {
                    Some((object, property.name.clone()))
                }
                MemberExpressionType::ComputedMemberExpression { object, property, .. } => {
                    let key = evaluate_expression(property, ctx)?;
                    Some((object, value_to_property_key(&key)))
                }
            };
            if let Some((object, method_name)) = parts {
                if let ExpressionOrSuper::Expression(_) = object {
                    let receiver = evaluate_expression_or_super(object, ctx)?;
                    let args = evaluate_arguments(arguments, ctx)?;

                    // (1) A callable own/prototype property wins (user methods,
                    //     class methods, function-valued fields).
                    if let Ok(prop) = get_property_with_receiver(&receiver, &receiver, &method_name, ctx) {
                        if value_is_callable(&prop) {
                            return call_value(&prop, receiver, args, ctx);
                        }
                    }

                    // (1b) `Function.prototype.{call,apply,bind}` on any
                    //      callable receiver — foundational for real libraries
                    //      (jQuery/lodash) that invoke helpers via `fn.call`.
                    if value_is_callable(&receiver) {
                        match method_name.as_str() {
                            "call" => {
                                let mut it = args.into_iter();
                                let this_arg = it.next().unwrap_or(JsValue::Undefined);
                                return call_value(&receiver, this_arg, it.collect(), ctx);
                            }
                            "apply" => {
                                let mut it = args.into_iter();
                                let this_arg = it.next().unwrap_or(JsValue::Undefined);
                                let arr = it.next().unwrap_or(JsValue::Undefined);
                                let call_args = match &arr {
                                    JsValue::Null | JsValue::Undefined => Vec::new(),
                                    _ => array_elements(&arr),
                                };
                                return call_value(&receiver, this_arg, call_args, ctx);
                            }
                            "bind" => {
                                // Best-effort: a bound function that fixes `this`
                                // and prepends partial args (a native closure).
                                let mut it = args.into_iter();
                                let bound_this = it.next().unwrap_or(JsValue::Undefined);
                                let bound_args: Vec<JsValue> = it.collect();
                                return Ok(make_bound_function(
                                    receiver.clone(),
                                    bound_this,
                                    bound_args,
                                ));
                            }
                            _ => {}
                        }
                    }

                    // (2) Otherwise route to a registry method keyed by the
                    //     receiver's builtin type (Array/String/Number/Object/…
                    //     or the sentinel's own name for Math/console/JSON/…).
                    let type_name = builtin_type_name(&receiver);
                    let sg = ctx.super_global.clone();
                    let sg_result = sg.borrow().call_method(
                        &type_name,
                        &method_name,
                        ctx,
                        receiver.clone(),
                        args,
                    );
                    if let Some(result) = sg_result {
                        return result;
                    }

                    // (3) Nothing matched.
                    return Err(JErrorType::TypeError(format!(
                        "{} is not a function",
                        method_name
                    )));
                }
            }
        }
    }

    // Normal call path: evaluate the callee to get the function
    let callee_value = evaluate_expression_or_super(callee, ctx)?;

    // Get the 'this' value for the call
    let this_value = get_call_this_value(callee, ctx);

    // Evaluate the arguments
    let args = evaluate_arguments(arguments, ctx)?;

    // Call the function
    call_value(&callee_value, this_value, args, ctx)
}

/// Evaluate a new expression (constructor call).
fn evaluate_new_expression(
    callee: &ExpressionType,
    arguments: &[ExpressionOrSpreadElement],
    ctx: &mut EvalContext,
) -> ValueResult {
    // Check if this is a simple identifier constructor (e.g., new String(), new Number())
    // If so, try super-global constructor dispatch first
    if let ExpressionType::ExpressionWhichCanBePattern(
        ExpressionPatternType::Identifier(id)
    ) = callee {
        let ctor_name = &id.name;
        
        // Evaluate arguments first
        let args = evaluate_arguments(arguments, ctx)?;
        
        // Try super-global constructor dispatch
        let sg = ctx.super_global.clone();
        let sg_result = sg.borrow().call_constructor(ctor_name, ctx, args.clone());
        
        if let Some(result) = sg_result {
            return result;
        }
        
        // Fall through to normal evaluation if super-global doesn't handle it
    }
    
    // Normal constructor path: evaluate the callee to get the constructor function
    let constructor = evaluate_expression(callee, ctx)?;

    // Verify it's callable
    let ctor_obj = match &constructor {
        JsValue::Object(obj) => {
            if !obj.borrow().is_callable() {
                return Err(JErrorType::TypeError(format!(
                    "{} is not a constructor",
                    constructor
                )));
            }
            obj.clone()
        }
        _ => {
            return Err(JErrorType::TypeError(format!(
                "{} is not a constructor",
                constructor
            )));
        }
    };

    // Create a new object for 'this'
    let new_obj = create_new_object_for_constructor(&ctor_obj)?;

    // Evaluate the arguments
    let args = evaluate_arguments(arguments, ctx)?;

    // Call the constructor with the new object as 'this' (threading the
    // superclass chain so `extends`/`super` set inherited instance fields).
    let result = invoke_constructor(&constructor, JsValue::Object(new_obj.clone()), args, ctx)?;

    // If constructor returns an object, use that; otherwise use the new object
    match result {
        JsValue::Object(_) => Ok(result),
        _ => Ok(JsValue::Object(new_obj)),
    }
}

/// Create a new object for use in a constructor call.
/// Sets up the prototype chain from the constructor's prototype property.
fn create_new_object_for_constructor(
    constructor: &JsObjectType,
) -> Result<JsObjectType, JErrorType> {
    use crate::runner::ds::object::{JsObject, ObjectType};
    use crate::runner::ds::object_property::{PropertyDescriptor, PropertyKey};

    // Create a new empty object
    let mut new_obj = SimpleObject::new();

    // Get the prototype from the constructor's 'prototype' property
    let ctor_borrowed = constructor.borrow();
    let prototype_key = PropertyKey::Str("prototype".to_string());

    if let Some(PropertyDescriptor::Data(data)) =
        ctor_borrowed.as_js_object().get_object_base().properties.get(&prototype_key)
    {
        if let JsValue::Object(proto_obj) = &data.value {
            // Set the prototype field on ObjectBase (used by get_prototype_of)
            new_obj.get_object_base_mut().prototype = Some(proto_obj.clone());
        }
    }

    drop(ctor_borrowed);

    let obj: JsObjectType = Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(new_obj))));
    Ok(obj)
}

/// Get the 'this' value for a call expression.
fn get_call_this_value(callee: &ExpressionOrSuper, ctx: &mut EvalContext) -> JsValue {
    match callee {
        ExpressionOrSuper::Expression(expr) => {
            match expr.as_ref() {
                // For member expressions, 'this' is the object
                ExpressionType::MemberExpression(member) => {
                    match member {
                        MemberExpressionType::SimpleMemberExpression { object, .. } |
                        MemberExpressionType::ComputedMemberExpression { object, .. } => {
                            match object {
                                ExpressionOrSuper::Expression(obj_expr) => {
                                    evaluate_expression(obj_expr, ctx).unwrap_or(JsValue::Undefined)
                                }
                                ExpressionOrSuper::Super => JsValue::Undefined,
                            }
                        }
                    }
                }
                // For other expressions, 'this' is undefined (or global in non-strict)
                _ => ctx.global_this.clone().unwrap_or(JsValue::Undefined),
            }
        }
        ExpressionOrSuper::Super => JsValue::Undefined,
    }
}

/// Evaluate call arguments.
fn evaluate_arguments(
    arguments: &[ExpressionOrSpreadElement],
    ctx: &mut EvalContext,
) -> Result<Vec<JsValue>, JErrorType> {
    let mut args = Vec::with_capacity(arguments.len());
    for arg in arguments {
        match arg {
            ExpressionOrSpreadElement::Expression(expr) => {
                args.push(evaluate_expression(expr, ctx)?);
            }
            ExpressionOrSpreadElement::SpreadElement(spread_expr) => {
                // Evaluate the spread expression
                let spread_value = evaluate_expression(spread_expr, ctx)?;

                // If it's an array, spread its elements into args
                if let JsValue::Object(spread_obj) = spread_value {
                    let borrowed = spread_obj.borrow();
                    let base = borrowed.as_js_object().get_object_base();

                    // Get length property if it exists
                    let length = if let Some(PropertyDescriptor::Data(prop)) =
                        base.properties.get(&PropertyKey::Str("length".to_string()))
                    {
                        match &prop.value {
                            JsValue::Number(JsNumberType::Integer(n)) => *n as usize,
                            JsValue::Number(JsNumberType::Float(n)) => *n as usize,
                            _ => 0,
                        }
                    } else {
                        0
                    };

                    // Iterate over array indices and add each element to args
                    for i in 0..length {
                        let elem_key = PropertyKey::Str(i.to_string());
                        let elem_value = if let Some(PropertyDescriptor::Data(prop)) =
                            base.properties.get(&elem_key)
                        {
                            prop.value.clone()
                        } else {
                            JsValue::Undefined
                        };
                        args.push(elem_value);
                    }
                } else if let JsValue::String(s) = spread_value {
                    // Spread string characters as separate args
                    for ch in s.chars() {
                        args.push(JsValue::String(ch.to_string()));
                    }
                } else {
                    return Err(JErrorType::TypeError(
                        "Spread requires an iterable".to_string(),
                    ));
                }
            }
        }
    }
    Ok(args)
}

/// Call a value as a function.
pub fn call_value(
    callee: &JsValue,
    this_value: JsValue,
    args: Vec<JsValue>,
    ctx: &mut EvalContext,
) -> ValueResult {
    match callee {
        JsValue::Object(obj) => {
            let obj_ref = obj.borrow();
            if obj_ref.is_callable() {
                drop(obj_ref);
                // For now, we'll call through our execution context stack
                // This is a simplified implementation that doesn't fully support
                // user-defined functions yet, but will work for native functions
                // stored in NativeFunctionObject
                call_function_object(obj, this_value, args, ctx)
            } else {
                // ChittiOS: a builtin sentinel (String/Number/Boolean/Array/…)
                // called directly as a function — route to its registry
                // constructor (`String(42)`, `Number("7")`, `Array(1,2)`).
                let builtin = {
                    let base = obj_ref.as_js_object().get_object_base();
                    match base.properties.get(&PropertyKey::Str("__builtin_name__".to_string())) {
                        Some(PropertyDescriptor::Data(d)) => match &d.value {
                            JsValue::String(name) => Some(name.clone()),
                            _ => None,
                        },
                        _ => None,
                    }
                };
                drop(obj_ref);
                if let Some(name) = builtin {
                    let sg = ctx.super_global.clone();
                    let result = sg.borrow().call_constructor(&name, ctx, args);
                    if let Some(r) = result {
                        return r;
                    }
                }
                Err(JErrorType::TypeError(format!(
                    "{} is not a function",
                    callee
                )))
            }
        }
        _ => Err(JErrorType::TypeError(format!(
            "{} is not a function",
            callee
        ))),
    }
}

/// Call a function object.
pub fn call_function_object(
    func: &crate::runner::ds::object::JsObjectType,
    this_value: JsValue,
    args: Vec<JsValue>,
    ctx: &mut EvalContext,
) -> ValueResult {
    // A bound function (`fn.bind(this, …)`): fix `this`, prepend the partial
    // args, and invoke the original target.
    {
        let fv = JsValue::Object(func.clone());
        if let Some(target) = get_own_prop_value(&fv, "__bound_target__") {
            let bound_this = get_own_prop_value(&fv, "__bound_this__").unwrap_or(JsValue::Undefined);
            let mut call_args = get_own_prop_value(&fv, "__bound_args__")
                .map(|a| array_elements(&a))
                .unwrap_or_default();
            call_args.extend(args);
            return call_value(&target, bound_this, call_args, ctx);
        }
    }

    let func_ref = func.borrow();

    match &*func_ref {
        ObjectType::Function(func_obj) => {
            // Get function metadata (these are Rc so cloning is cheap)
            let body = func_obj.get_function_object_base().body_code.clone();
            let params = func_obj.get_function_object_base().formal_parameters.clone();
            let func_env = func_obj.get_function_object_base().environment.clone();

            drop(func_ref);

            // Create a new function scope
            let saved_lex_env = ctx.lex_env.clone();
            let saved_var_env = ctx.var_env.clone();

            // Create new environment for the function, with the function's closure as outer
            use crate::runner::ds::env_record::new_declarative_environment;
            let func_scope = new_declarative_environment(Some(func_env));
            ctx.lex_env = func_scope.clone();
            ctx.var_env = func_scope;

            // Bind parameters to arguments
            for (i, param) in params.iter().enumerate() {
                if let crate::parser::ast::PatternType::PatternWhichCanBeExpression(
                    ExpressionPatternType::Identifier(id)
                ) = param {
                    let arg_value = args.get(i).cloned().unwrap_or(JsValue::Undefined);
                    ctx.create_binding(&id.name, false)?;
                    ctx.initialize_binding(&id.name, arg_value)?;
                }
            }

            // Execute each statement in the function body
            use super::statement::execute_statement;
            use super::types::CompletionType;

            let mut result_completion = super::types::Completion::normal();

            for stmt in body.body.iter() {
                let completion = execute_statement(stmt, ctx)?;

                match completion.completion_type {
                    CompletionType::Return => {
                        result_completion = completion;
                        break;
                    }
                    CompletionType::Throw => {
                        // Restore environment before returning error
                        ctx.lex_env = saved_lex_env;
                        ctx.var_env = saved_var_env;
                        return Err(JErrorType::Thrown(completion.value.clone().unwrap_or(JsValue::Undefined)));
                    }
                    CompletionType::Break | CompletionType::Continue | CompletionType::Yield => {
                        // These shouldn't escape function body
                        result_completion = completion;
                        break;
                    }
                    CompletionType::Normal => {
                        result_completion = completion;
                    }
                }
            }

            // Restore the previous environment
            ctx.lex_env = saved_lex_env;
            ctx.var_env = saved_var_env;

            // Return the result
            match result_completion.completion_type {
                CompletionType::Return => Ok(result_completion.get_value()),
                _ => Ok(JsValue::Undefined),
            }
        }
        ObjectType::Ordinary(obj) => {
            // Check if it's a SimpleFunctionObject (has marker property)
            let marker = PropertyKey::Str("__simple_function__".to_string());
            let default_ctor_marker = PropertyKey::Str("__default_constructor__".to_string());

            let generator_marker = PropertyKey::Str("__generator__".to_string());
            let generator_next_marker = PropertyKey::Str("__generator_next__".to_string());

            // ChittiOS: a Promise resolve/reject settler function.
            if let Some(PropertyDescriptor::Data(op)) =
                obj.get_object_base().properties.get(&PropertyKey::Str("__promise_op__".to_string()))
            {
                let is_reject = matches!(&op.value, JsValue::String(s) if s == "reject");
                let target = obj
                    .get_object_base()
                    .properties
                    .get(&PropertyKey::Str("__promise_target__".to_string()))
                    .and_then(|d| match d {
                        PropertyDescriptor::Data(dd) => Some(dd.value.clone()),
                        _ => None,
                    });
                drop(func_ref);
                if let Some(t) = target {
                    let v = args.into_iter().next().unwrap_or(JsValue::Undefined);
                    crate::runner::std_lib::promise::settle(&t, is_reject, v, ctx);
                }
                return Ok(JsValue::Undefined);
            }

            if obj.get_object_base().properties.contains_key(&default_ctor_marker) {
                // It's a default constructor (no-op) - just return undefined
                drop(func_ref);
                Ok(JsValue::Undefined)
            } else if let Some(PropertyDescriptor::Data(data)) = obj.get_object_base().properties.get(&generator_next_marker) {
                // It's a generator next method - call the generator's .next()
                let gen_obj = match &data.value {
                    JsValue::Object(o) => o.clone(),
                    _ => return Err(JErrorType::TypeError("Invalid generator reference".to_string())),
                };
                drop(func_ref);

                // Call the generator's next method
                let mut gen_borrowed = gen_obj.borrow_mut();

                // We need to cast to GeneratorObject
                match &mut *gen_borrowed {
                    ObjectType::Ordinary(inner_obj) => {
                        let gen_marker = PropertyKey::Str("__generator_object__".to_string());
                        if inner_obj.get_object_base().properties.contains_key(&gen_marker) {
                            // Cast to GeneratorObject
                            let gen_ptr = inner_obj.as_mut() as *mut dyn JsObject as *mut GeneratorObject;
                            drop(gen_borrowed);
                            // SAFETY: We know this is a GeneratorObject
                            unsafe {
                                let gen = &mut *gen_ptr;
                                gen.next(ctx)
                            }
                        } else {
                            Err(JErrorType::TypeError("Not a generator object".to_string()))
                        }
                    }
                    _ => Err(JErrorType::TypeError("Not a generator object".to_string())),
                }
            } else if obj.get_object_base().properties.contains_key(&generator_marker) {
                // It's a generator function - create a GeneratorObject instead of executing
                let obj_ptr = obj.as_ref() as *const dyn JsObject;
                drop(func_ref);

                // SAFETY: We know this is a SimpleFunctionObject and we've dropped func_ref
                unsafe {
                    let simple_func = &*(obj_ptr as *const SimpleFunctionObject);
                    // Create generator object from function data
                    create_generator_object(
                        simple_func.body_ptr,
                        simple_func.params_ptr,
                        simple_func.environment.clone(),
                        args,
                        ctx,
                    )
                }
            } else if obj.get_object_base().properties.contains_key(&marker) {
                // It's a SimpleFunctionObject - use the call_with_this method
                // Get a raw pointer to call call_with_this
                let obj_ptr = obj.as_ref() as *const dyn JsObject;
                drop(func_ref);

                // SAFETY: We know this is a SimpleFunctionObject and we've dropped func_ref
                unsafe {
                    // Cast to SimpleFunctionObject
                    let simple_func = &*(obj_ptr as *const SimpleFunctionObject);
                    simple_func.call_with_this(this_value, args, ctx)
                }
            } else {
                Err(JErrorType::TypeError("Object is not callable".to_string()))
            }
        }
    }
}


/// Get a property from a value, calling getters if necessary.
fn get_property_with_ctx(value: &JsValue, prop_name: &str, ctx: &mut EvalContext) -> ValueResult {
    // Use the receiver version with value as both receiver and lookup target
    get_property_with_receiver(value, value, prop_name, ctx)
}

/// Get a property, with separate receiver (for 'this') and lookup target.
/// This handles prototype chain lookups where 'this' should be the original receiver.
fn get_property_with_receiver(
    receiver: &JsValue,
    lookup_target: &JsValue,
    prop_name: &str,
    ctx: &mut EvalContext,
) -> ValueResult {
    // ChittiOS: native-backed property (live DOM element view).
    if let Some(node) = native_node(lookup_target) {
        if let Some(np) = ctx.native_props.clone() {
            if let Some(val) = np.get(node, prop_name) {
                return Ok(val);
            }
        }
    }
    // ChittiOS: Proxy `get` trap.
    if let Some((target, handler)) = proxy_parts(lookup_target) {
        let trap = get_own_prop_value(&handler, "get");
        if let Some(g) = trap.filter(value_is_callable) {
            return call_value(
                &g,
                handler,
                alloc::vec![target, JsValue::String(prop_name.to_string()), receiver.clone()],
                ctx,
            );
        }
        return get_property_with_receiver(&target, &target, prop_name, ctx);
    }
    match lookup_target {
        JsValue::Object(obj) => {
            // Check if this is a generator object and we're accessing 'next'
            let is_gen_obj = {
                let obj_ref = obj.borrow();
                let marker = PropertyKey::Str("__generator_object__".to_string());
                obj_ref.as_js_object().get_object_base().properties.contains_key(&marker)
            };

            if is_gen_obj && prop_name == "next" {
                // Return a special "next" method that calls the generator's next
                // We create a wrapper function that holds a reference to the generator
                return create_generator_next_method(obj.clone());
            }

            let prop_key = PropertyKey::Str(prop_name.to_string());

            // Check own property
            let desc = {
                let obj_ref = obj.borrow();
                obj_ref.as_js_object().get_own_property(&prop_key)?.cloned()
            };

            if let Some(desc) = desc {
                match desc {
                    PropertyDescriptor::Data(data) => Ok(data.value.clone()),
                    PropertyDescriptor::Accessor(accessor) => {
                        // Call the getter if it exists, with RECEIVER as 'this'
                        if let Some(getter) = accessor.get {
                            call_accessor_function(&getter, receiver.clone(), vec![], ctx)
                        } else {
                            Ok(JsValue::Undefined)
                        }
                    }
                }
            } else {
                // Check prototype chain - receiver stays the same
                let proto = obj.borrow().as_js_object().get_prototype_of();
                if let Some(proto) = proto {
                    get_property_with_receiver(receiver, &JsValue::Object(proto), prop_name, ctx)
                } else {
                    Ok(JsValue::Undefined)
                }
            }
        }
        JsValue::String(s) => {
            // String primitive property access
            if prop_name == "length" {
                Ok(JsValue::Number(JsNumberType::Integer(s.len() as i64)))
            } else if let Ok(index) = prop_name.parse::<usize>() {
                // Access character at index
                if index < s.len() {
                    Ok(JsValue::String(s.chars().nth(index).unwrap().to_string()))
                } else {
                    Ok(JsValue::Undefined)
                }
            } else {
                // Other string methods not yet supported
                Ok(JsValue::Undefined)
            }
        }
        JsValue::Undefined => Err(JErrorType::TypeError(format!(
            "Cannot read property '{}' of undefined",
            prop_name
        ))),
        JsValue::Null => Err(JErrorType::TypeError(format!(
            "Cannot read property '{}' of null",
            prop_name
        ))),
        _ => {
            // Primitive types - return undefined for now
            Ok(JsValue::Undefined)
        }
    }
}

/// Call an accessor function (getter or setter).
fn call_accessor_function(
    func: &JsObjectType,
    this_value: JsValue,
    args: Vec<JsValue>,
    ctx: &mut EvalContext,
) -> ValueResult {
    let func_ref = func.borrow();

    // Check if it's our SimpleFunctionObject (stored as Ordinary with marker)
    if let ObjectType::Ordinary(obj) = &*func_ref {
        if is_simple_function_object(obj.as_ref()) {
            // This is a SimpleFunctionObject - we need to call its method
            // Since we can't downcast easily, we use unsafe to access it
            // The object was created by create_function_object so we know it's a SimpleFunctionObject
            let simple_func = unsafe {
                // Get the raw pointer to the inner object and cast it
                let ptr = obj.as_ref() as *const dyn JsObject as *const SimpleFunctionObject;
                &*ptr
            };
            drop(func_ref);
            return simple_func.call_with_this(this_value, args, ctx);
        }

        // Regular object - try to call as function (likely will fail)
        drop(func_ref);
        call_function_object(func, this_value, args, ctx)
    } else if let ObjectType::Function(func_obj) = &*func_ref {
        // It's a full function object
        let body = func_obj.get_function_object_base().body_code.clone();
        let params = func_obj.get_function_object_base().formal_parameters.clone();
        let func_env = func_obj.get_function_object_base().environment.clone();

        drop(func_ref);

        call_function_with_body(body, params, func_env, this_value, args, ctx)
    } else {
        Err(JErrorType::TypeError("Not a function".to_string()))
    }
}

/// Call a function given its body, parameters, and environment.
fn call_function_with_body(
    body: Rc<crate::parser::ast::FunctionBodyData>,
    params: Rc<Vec<PatternType>>,
    func_env: crate::runner::ds::lex_env::JsLexEnvironmentType,
    this_value: JsValue,
    args: Vec<JsValue>,
    ctx: &mut EvalContext,
) -> ValueResult {
    use crate::runner::ds::env_record::new_declarative_environment;
    use super::statement::execute_statement;
    use super::types::CompletionType;

    // Save current environment
    let saved_lex_env = ctx.lex_env.clone();
    let saved_var_env = ctx.var_env.clone();
    let saved_this = ctx.global_this.clone();

    // Create new environment for the function, with the function's closure as outer
    let func_scope = new_declarative_environment(Some(func_env));
    ctx.lex_env = func_scope.clone();
    ctx.var_env = func_scope;
    ctx.global_this = Some(this_value);

    // Bind parameters to arguments
    for (i, param) in params.iter().enumerate() {
        if let PatternType::PatternWhichCanBeExpression(
            ExpressionPatternType::Identifier(id)
        ) = param {
            let arg_value = args.get(i).cloned().unwrap_or(JsValue::Undefined);
            ctx.create_binding(&id.name, false)?;
            ctx.initialize_binding(&id.name, arg_value)?;
        }
    }

    // Hoist `var` declarations to the top of the function scope.
    super::statement::hoist_var_declarations(&body.body, ctx);

    // Execute each statement in the function body. On ANY exit path — a normal
    // return, an `Ok(Throw)` completion, or an `Err(..)` propagating up from a
    // nested call — the function's environment must be restored, otherwise the
    // caller keeps running against the callee's (popped) scope and later
    // variable lookups fail spuriously (e.g. a `catch` block can't see the
    // enclosing function's `var`). We compute the outcome, always restore, then
    // return.
    let mut result_completion = super::types::Completion::normal();
    let mut pending_err: Option<JErrorType> = None;

    for stmt in body.body.iter() {
        match execute_statement(stmt, ctx) {
            Err(e) => {
                pending_err = Some(e);
                break;
            }
            Ok(completion) => match completion.completion_type {
                CompletionType::Return => {
                    result_completion = completion;
                    break;
                }
                CompletionType::Throw => {
                    pending_err = Some(JErrorType::Thrown(
                        completion.value.clone().unwrap_or(JsValue::Undefined),
                    ));
                    break;
                }
                CompletionType::Break | CompletionType::Continue | CompletionType::Yield => {
                    result_completion = completion;
                    break;
                }
                CompletionType::Normal => {
                    result_completion = completion;
                }
            },
        }
    }

    // Restore the previous environment on every path.
    ctx.lex_env = saved_lex_env;
    ctx.var_env = saved_var_env;
    ctx.global_this = saved_this;

    if let Some(e) = pending_err {
        return Err(e);
    }

    // Return the result
    match result_completion.completion_type {
        CompletionType::Return => Ok(result_completion.get_value()),
        _ => Ok(JsValue::Undefined),
    }
}

/// Convert a value to a property key string.
fn to_property_key(value: &JsValue) -> String {
    to_string(value)
}

/// Evaluate an assignment expression.
fn evaluate_assignment_expression(
    operator: &AssignmentOperator,
    left: &PatternOrExpression,
    right: &ExpressionType,
    ctx: &mut EvalContext,
) -> ValueResult {
    // Check if this is a member expression assignment
    if let PatternOrExpression::Expression(expr) = left {
        if let ExpressionType::MemberExpression(member) = expr.as_ref() {
            return evaluate_member_assignment(operator, member, right, ctx);
        }
    }

    // Handle destructuring assignment patterns
    if let PatternOrExpression::Pattern(pattern) = left {
        if !matches!(
            &**pattern,
            PatternType::PatternWhichCanBeExpression(ExpressionPatternType::Identifier(_))
        ) {
            if !matches!(operator, AssignmentOperator::Equals) {
                return Err(JErrorType::TypeError(
                    "Compound assignment not supported for destructuring patterns".to_string(),
                ));
            }
            let rhs_value = evaluate_expression(right, ctx)?;
            assign_pattern(pattern, rhs_value.clone(), ctx)?;
            return Ok(rhs_value);
        }
    }

    // Get the name to assign to (simple variable assignment)
    let name = match left {
        PatternOrExpression::Pattern(pattern) => get_pattern_name(pattern)?,
        PatternOrExpression::Expression(expr) => get_expression_name(expr)?,
    };

    // Evaluate the right-hand side
    let rhs_value = evaluate_expression(right, ctx)?;

    // Compute the final value based on the operator
    let final_value = match operator {
        AssignmentOperator::Equals => rhs_value,
        AssignmentOperator::AddEquals => {
            let current = ctx.get_binding(&name)?;
            add_values(&current, &rhs_value)?
        }
        AssignmentOperator::SubtractEquals => {
            let current = ctx.get_binding(&name)?;
            subtract_values(&current, &rhs_value)?
        }
        AssignmentOperator::MultiplyEquals => {
            let current = ctx.get_binding(&name)?;
            multiply_values(&current, &rhs_value)?
        }
        AssignmentOperator::DivideEquals => {
            let current = ctx.get_binding(&name)?;
            divide_values(&current, &rhs_value)?
        }
        AssignmentOperator::ModuloEquals => {
            let current = ctx.get_binding(&name)?;
            modulo_values(&current, &rhs_value)?
        }
        AssignmentOperator::BitwiseLeftShiftEquals => {
            let current = ctx.get_binding(&name)?;
            left_shift(&current, &rhs_value)?
        }
        AssignmentOperator::BitwiseRightShiftEquals => {
            let current = ctx.get_binding(&name)?;
            right_shift(&current, &rhs_value)?
        }
        AssignmentOperator::BitwiseUnsignedRightShiftEquals => {
            let current = ctx.get_binding(&name)?;
            unsigned_right_shift(&current, &rhs_value)?
        }
        AssignmentOperator::BitwiseOrEquals => {
            let current = ctx.get_binding(&name)?;
            bitwise_or(&current, &rhs_value)?
        }
        AssignmentOperator::BitwiseAndEquals => {
            let current = ctx.get_binding(&name)?;
            bitwise_and(&current, &rhs_value)?
        }
        AssignmentOperator::BitwiseXorEquals => {
            let current = ctx.get_binding(&name)?;
            bitwise_xor(&current, &rhs_value)?
        }
        AssignmentOperator::ExponentEquals => {
            let current = ctx.get_binding(&name)?;
            exponent_values(&current, &rhs_value)?
        }
        // Logical assignment: keep the current value unless the guard passes.
        AssignmentOperator::LogicalAndEquals => {
            let current = ctx.get_binding(&name)?;
            if to_boolean(&current) { rhs_value } else { current }
        }
        AssignmentOperator::LogicalOrEquals => {
            let current = ctx.get_binding(&name)?;
            if to_boolean(&current) { current } else { rhs_value }
        }
        AssignmentOperator::NullishEquals => {
            let current = ctx.get_binding(&name)?;
            if matches!(current, JsValue::Null | JsValue::Undefined) { rhs_value } else { current }
        }
    };

    // Anonymous-function name inference for `f = function(){}` / `f = () => {}`.
    if matches!(operator, AssignmentOperator::Equals) && is_anonymous_function_expr(right) {
        infer_function_name(&final_value, &name);
    }

    // Set the binding and return the value
    ctx.set_binding(&name, final_value.clone())?;
    Ok(final_value)
}

/// Assign a value to a binding pattern (destructuring assignment).
fn assign_pattern(
    pattern: &PatternType,
    value: JsValue,
    ctx: &mut EvalContext,
) -> Result<(), JErrorType> {
    match pattern {
        PatternType::PatternWhichCanBeExpression(ExpressionPatternType::Identifier(id)) => {
            ctx.set_binding(&id.name, value)?;
            Ok(())
        }
        PatternType::ObjectPattern { properties, .. } => {
            for prop in properties {
                let prop_data = &prop.0;
                let key_name = get_assignment_property_key(prop_data, ctx)?;
                let prop_value = get_object_property_value(&value, &key_name, ctx)?;
                assign_pattern(&prop_data.value, prop_value, ctx)?;
            }
            Ok(())
        }
        PatternType::ArrayPattern { elements, .. } => {
            for (index, element) in elements.iter().enumerate() {
                if let Some(elem_pattern) = element {
                    if let PatternType::RestElement { argument, .. } = elem_pattern.as_ref() {
                        let rest_value = get_rest_elements_for_assignment(&value, index, ctx)?;
                        assign_pattern(argument, rest_value, ctx)?;
                    } else {
                        let elem_value = get_array_element_for_assignment(&value, index, ctx)?;
                        assign_pattern(elem_pattern, elem_value, ctx)?;
                    }
                }
            }
            Ok(())
        }
        PatternType::AssignmentPattern { left, right, .. } => {
            let actual_value = if matches!(value, JsValue::Undefined) {
                evaluate_expression(right, ctx)?
            } else {
                value
            };
            assign_pattern(left, actual_value, ctx)
        }
        PatternType::RestElement { argument, .. } => assign_pattern(argument, value, ctx),
    }
}

fn get_assignment_property_key(
    prop: &PropertyData<Box<PatternType>>,
    ctx: &mut EvalContext,
) -> Result<String, JErrorType> {
    if prop.computed {
        let key_value = evaluate_expression(prop.key.as_ref(), ctx)?;
        Ok(to_string(&key_value))
    } else {
        match prop.key.as_ref() {
            ExpressionType::ExpressionWhichCanBePattern(ExpressionPatternType::Identifier(id)) => {
                Ok(id.name.clone())
            }
            ExpressionType::Literal(lit_data) => match &lit_data.value {
                LiteralType::StringLiteral(s) => Ok(s.clone()),
                LiteralType::NumberLiteral(num) => {
                    use crate::parser::ast::NumberLiteralType;
                    match num {
                        NumberLiteralType::IntegerLiteral(n) => Ok(n.to_string()),
                        NumberLiteralType::FloatLiteral(n) => Ok(n.to_string()),
                    }
                }
                _ => Err(JErrorType::TypeError(
                    "Invalid property key in destructuring assignment".to_string(),
                )),
            },
            _ => Err(JErrorType::TypeError(
                "Invalid property key in destructuring assignment".to_string(),
            )),
        }
    }
}

fn get_object_property_value(
    obj: &JsValue,
    key: &str,
    ctx: &mut EvalContext,
) -> Result<JsValue, JErrorType> {
    match obj {
        JsValue::Object(_) => get_property_with_ctx(obj, key, ctx),
        _ => Err(JErrorType::TypeError(
            "Cannot destructure non-object".to_string(),
        )),
    }
}

fn get_array_element_for_assignment(
    arr: &JsValue,
    index: usize,
    ctx: &mut EvalContext,
) -> Result<JsValue, JErrorType> {
    match arr {
        JsValue::Object(_) => {
            let key = index.to_string();
            get_property_with_ctx(arr, &key, ctx)
        }
        _ => Err(JErrorType::TypeError(
            "Cannot destructure non-array".to_string(),
        )),
    }
}

fn get_rest_elements_for_assignment(
    arr: &JsValue,
    start_index: usize,
    ctx: &mut EvalContext,
) -> Result<JsValue, JErrorType> {
    use crate::runner::ds::object::{JsObject, JsObjectType, ObjectType};
    use crate::runner::ds::object_property::{PropertyDescriptor, PropertyDescriptorData, PropertyKey};
    use core::cell::RefCell;
    use alloc::rc::Rc;

    let length = match arr {
        JsValue::Object(_) => {
            let length_value = get_property_with_ctx(arr, "length", ctx)?;
            match length_value {
                JsValue::Number(JsNumberType::Integer(n)) => n.max(0) as usize,
                JsValue::Number(JsNumberType::Float(n)) => {
                    if n.is_nan() || n < 0.0 {
                        0
                    } else {
                        n as usize
                    }
                }
                _ => 0,
            }
        }
        _ => {
            return Err(JErrorType::TypeError(
                "Cannot use rest with non-array".to_string(),
            ))
        }
    };

    let mut rest_obj = ctx.new_tracked_object()?;
    let mut rest_index = 0;

    for i in start_index..length {
        let key = i.to_string();
        let value = get_property_with_ctx(arr, &key, ctx)?;
        rest_obj.get_object_base_mut().properties.insert(
            PropertyKey::Str(rest_index.to_string()),
            PropertyDescriptor::Data(PropertyDescriptorData {
                value,
                writable: true,
                enumerable: true,
                configurable: true,
            }),
        );
        rest_index += 1;
    }

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

/// Evaluate assignment to a member expression (obj.prop = value or obj[prop] = value).
fn evaluate_member_assignment(
    operator: &AssignmentOperator,
    member: &MemberExpressionType,
    right: &ExpressionType,
    ctx: &mut EvalContext,
) -> ValueResult {
    // Get the object and property key
    let (obj_value, prop_name) = match member {
        MemberExpressionType::SimpleMemberExpression { object, property, .. } => {
            let obj = evaluate_expression_or_super(object, ctx)?;
            (obj, property.name.clone())
        }
        MemberExpressionType::ComputedMemberExpression { object, property, .. } => {
            let obj = evaluate_expression_or_super(object, ctx)?;
            let prop_val = evaluate_expression(property.as_ref(), ctx)?;
            (obj, to_property_key(&prop_val))
        }
    };

    // Evaluate the right-hand side
    let rhs_value = evaluate_expression(right, ctx)?;

    // Compute the final value based on the operator
    let final_value = if matches!(operator, AssignmentOperator::Equals) {
        rhs_value
    } else {
        let current = get_property_with_ctx(&obj_value, &prop_name, ctx)?;
        match operator {
            AssignmentOperator::Equals => unreachable!(),
            AssignmentOperator::AddEquals => add_values(&current, &rhs_value)?,
            AssignmentOperator::SubtractEquals => subtract_values(&current, &rhs_value)?,
            AssignmentOperator::MultiplyEquals => multiply_values(&current, &rhs_value)?,
            AssignmentOperator::DivideEquals => divide_values(&current, &rhs_value)?,
            AssignmentOperator::ModuloEquals => modulo_values(&current, &rhs_value)?,
            AssignmentOperator::BitwiseLeftShiftEquals => left_shift(&current, &rhs_value)?,
            AssignmentOperator::BitwiseRightShiftEquals => right_shift(&current, &rhs_value)?,
            AssignmentOperator::BitwiseUnsignedRightShiftEquals => unsigned_right_shift(&current, &rhs_value)?,
            AssignmentOperator::BitwiseOrEquals => bitwise_or(&current, &rhs_value)?,
            AssignmentOperator::BitwiseAndEquals => bitwise_and(&current, &rhs_value)?,
            AssignmentOperator::BitwiseXorEquals => bitwise_xor(&current, &rhs_value)?,
            AssignmentOperator::ExponentEquals => exponent_values(&current, &rhs_value)?,
            AssignmentOperator::LogicalAndEquals => {
                if to_boolean(&current) { rhs_value } else { current }
            }
            AssignmentOperator::LogicalOrEquals => {
                if to_boolean(&current) { current } else { rhs_value }
            }
            AssignmentOperator::NullishEquals => {
                if matches!(current, JsValue::Null | JsValue::Undefined) { rhs_value } else { current }
            }
        }
    };

    // Set the property (with setter support)
    set_property_with_ctx(&obj_value, &prop_name, final_value.clone(), ctx)?;
    Ok(final_value)
}

/// Set a property on an object, calling setters if necessary.
fn set_property_with_ctx(
    value: &JsValue,
    prop_name: &str,
    new_value: JsValue,
    ctx: &mut EvalContext,
) -> Result<(), JErrorType> {
    // ChittiOS: native-backed property (live DOM element view).
    if let Some(node) = native_node(value) {
        if let Some(np) = ctx.native_props.clone() {
            if np.set(node, prop_name, new_value.clone()) {
                return Ok(());
            }
        }
    }
    // ChittiOS: Proxy `set` trap.
    if let Some((target, handler)) = proxy_parts(value) {
        let trap = get_own_prop_value(&handler, "set");
        if let Some(s) = trap.filter(value_is_callable) {
            call_value(
                &s,
                handler,
                alloc::vec![target, JsValue::String(prop_name.to_string()), new_value, value.clone()],
                ctx,
            )?;
            return Ok(());
        }
        return set_property_with_ctx(&target, prop_name, new_value, ctx);
    }
    match value {
        JsValue::Object(obj) => {
            let prop_key = PropertyKey::Str(prop_name.to_string());

            // Check for accessor property (setter)
            let desc = {
                let obj_ref = obj.borrow();
                obj_ref.as_js_object().get_own_property(&prop_key)?.cloned()
            };

            if let Some(PropertyDescriptor::Accessor(accessor)) = desc {
                // Call the setter if it exists
                if let Some(setter) = accessor.set {
                    call_accessor_function(&setter, value.clone(), vec![new_value], ctx)?;
                    return Ok(());
                }
                // No setter - fall through to data property behavior
            }

            // Set as data property
            let mut obj_mut = obj.borrow_mut();
            obj_mut.as_js_object_mut().get_object_base_mut().properties.insert(
                prop_key,
                PropertyDescriptor::Data(PropertyDescriptorData {
                    value: new_value,
                    writable: true,
                    enumerable: true,
                    configurable: true,
                }),
            );
            Ok(())
        }
        _ => Err(JErrorType::TypeError("Cannot set property on non-object".to_string())),
    }
}

/// Get the name from a pattern (for simple identifier patterns).
fn get_pattern_name(pattern: &PatternType) -> Result<String, JErrorType> {
    match pattern {
        PatternType::PatternWhichCanBeExpression(ExpressionPatternType::Identifier(id)) => {
            Ok(id.name.clone())
        }
        _ => Err(JErrorType::TypeError(
            "Complex patterns in assignment not yet supported".to_string(),
        )),
    }
}

/// Get the name from an expression (for simple identifier expressions).
fn get_expression_name(expr: &ExpressionType) -> Result<String, JErrorType> {
    match expr {
        ExpressionType::ExpressionWhichCanBePattern(ExpressionPatternType::Identifier(id)) => {
            Ok(id.name.clone())
        }
        _ => Err(JErrorType::TypeError(
            "Assignment to non-identifier expressions not yet supported".to_string(),
        )),
    }
}

/// Evaluate a unary expression.
fn evaluate_unary_expression(
    operator: &UnaryOperator,
    argument: &ExpressionType,
    ctx: &mut EvalContext,
) -> ValueResult {
    match operator {
        UnaryOperator::Await => {
            // ChittiOS: promises settle synchronously here, so `await` extracts a
            // settled promise's value immediately (throwing a rejection), or
            // returns a non-promise value unchanged.
            let value = evaluate_expression(argument, ctx)?;
            crate::runner::std_lib::promise::await_value(value)
        }
        UnaryOperator::TypeOf => {
            let value = evaluate_expression(argument, ctx).unwrap_or(JsValue::Undefined);
            Ok(JsValue::String(get_typeof_string(&value)))
        }
        UnaryOperator::Void => {
            let _ = evaluate_expression(argument, ctx)?;
            Ok(JsValue::Undefined)
        }
        UnaryOperator::LogicalNot => {
            let value = evaluate_expression(argument, ctx)?;
            Ok(JsValue::Boolean(!to_boolean(&value)))
        }
        UnaryOperator::Minus => {
            let value = evaluate_expression(argument, ctx)?;
            negate_number(&value)
        }
        UnaryOperator::Plus => {
            let value = evaluate_expression(argument, ctx)?;
            to_number(&value)
        }
        UnaryOperator::BitwiseNot => {
            let value = evaluate_expression(argument, ctx)?;
            bitwise_not(&value)
        }
        UnaryOperator::Delete => {
            match argument {
                ExpressionType::MemberExpression(member) => {
                    match member {
                        MemberExpressionType::SimpleMemberExpression { object, property, .. } => {
                            let obj_value = evaluate_expression_or_super(object, ctx)?;
                            match obj_value {
                                JsValue::Object(obj) => {
                                    let prop_key = PropertyKey::Str(property.name.clone());
                                    let result = obj.borrow_mut().as_js_object_mut().delete(&prop_key)?;
                                    Ok(JsValue::Boolean(result))
                                }
                                _ => Ok(JsValue::Boolean(true))
                            }
                        }
                        MemberExpressionType::ComputedMemberExpression { object, property, .. } => {
                            let obj_value = evaluate_expression_or_super(object, ctx)?;
                            let prop_value = evaluate_expression(property.as_ref(), ctx)?;
                            let prop_name = to_property_key(&prop_value);
                            match obj_value {
                                JsValue::Object(obj) => {
                                    let prop_key = PropertyKey::Str(prop_name);
                                    let result = obj.borrow_mut().as_js_object_mut().delete(&prop_key)?;
                                    Ok(JsValue::Boolean(result))
                                }
                                _ => Ok(JsValue::Boolean(true))
                            }
                        }
                    }
                }
                _ => {
                    // Non-member: evaluate for side effects, return true
                    let _ = evaluate_expression(argument, ctx)?;
                    Ok(JsValue::Boolean(true))
                }
            }
        }
    }
}

/// Evaluate a binary expression.
fn evaluate_binary_expression(
    operator: &BinaryOperator,
    left: &ExpressionType,
    right: &ExpressionType,
    ctx: &mut EvalContext,
) -> ValueResult {
    let left_val = evaluate_expression(left, ctx)?;
    let right_val = evaluate_expression(right, ctx)?;

    match operator {
        // Arithmetic
        BinaryOperator::Add => add_values(&left_val, &right_val),
        BinaryOperator::Subtract => subtract_values(&left_val, &right_val),
        BinaryOperator::Multiply => multiply_values(&left_val, &right_val),
        BinaryOperator::Divide => divide_values(&left_val, &right_val),
        BinaryOperator::Modulo => modulo_values(&left_val, &right_val),
        BinaryOperator::Exponent => exponent_values(&left_val, &right_val),

        // Comparison
        BinaryOperator::LessThan => compare_values(&left_val, &right_val, |a, b| a < b),
        BinaryOperator::GreaterThan => compare_values(&left_val, &right_val, |a, b| a > b),
        BinaryOperator::LessThanEqual => compare_values(&left_val, &right_val, |a, b| a <= b),
        BinaryOperator::GreaterThanEqual => compare_values(&left_val, &right_val, |a, b| a >= b),

        // Equality
        BinaryOperator::StrictlyEqual => Ok(JsValue::Boolean(strict_equality(&left_val, &right_val))),
        BinaryOperator::StrictlyUnequal => Ok(JsValue::Boolean(!strict_equality(&left_val, &right_val))),
        BinaryOperator::LooselyEqual => Ok(JsValue::Boolean(loose_equality(&left_val, &right_val))),
        BinaryOperator::LooselyUnequal => Ok(JsValue::Boolean(!loose_equality(&left_val, &right_val))),

        // Bitwise
        BinaryOperator::BitwiseAnd => bitwise_and(&left_val, &right_val),
        BinaryOperator::BitwiseOr => bitwise_or(&left_val, &right_val),
        BinaryOperator::BitwiseXor => bitwise_xor(&left_val, &right_val),
        BinaryOperator::BitwiseLeftShift => left_shift(&left_val, &right_val),
        BinaryOperator::BitwiseRightShift => right_shift(&left_val, &right_val),
        BinaryOperator::BitwiseUnsignedRightShift => unsigned_right_shift(&left_val, &right_val),

        // Other
        BinaryOperator::InstanceOf => {
            // Left must be an object for instanceof to potentially return true
            let left_obj = match &left_val {
                JsValue::Object(obj) => obj.clone(),
                _ => return Ok(JsValue::Boolean(false)),
            };

            // Right must be an object (constructor function) with a prototype property
            let right_obj = match &right_val {
                JsValue::Object(obj) => obj.clone(),
                _ => return Err(JErrorType::TypeError("Right-hand side of 'instanceof' is not an object".to_string())),
            };

            // Get the prototype property from the right-hand side (constructor.prototype)
            let proto_key = PropertyKey::Str("prototype".to_string());
            let right_prototype = {
                let right_borrowed = right_obj.borrow();
                if let Some(desc) = right_borrowed.as_js_object().get_own_property(&proto_key)? {
                    match desc {
                        PropertyDescriptor::Data(data) => {
                            match &data.value {
                                JsValue::Object(p) => Some(p.clone()),
                                _ => None,
                            }
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            };

            // If right.prototype is not an object, return false
            let target_proto = match right_prototype {
                Some(p) => p,
                None => return Ok(JsValue::Boolean(false)),
            };

            // Walk the prototype chain of left
            let mut current_proto = left_obj.borrow().as_js_object().get_prototype_of();

            while let Some(proto) = current_proto {
                // Compare by reference
                if Rc::ptr_eq(&proto, &target_proto) {
                    return Ok(JsValue::Boolean(true));
                }
                current_proto = proto.borrow().as_js_object().get_prototype_of();
            }

            Ok(JsValue::Boolean(false))
        }
        BinaryOperator::In => {
            let prop_key = PropertyKey::Str(to_property_key(&left_val));
            match &right_val {
                JsValue::Object(obj) => {
                    let has = obj.borrow().as_js_object().has_property(&prop_key);
                    Ok(JsValue::Boolean(has))
                }
                _ => Err(JErrorType::TypeError("Cannot use 'in' operator with non-object".to_string()))
            }
        }
    }
}

/// Evaluate a logical expression with short-circuit evaluation.
fn evaluate_logical_expression(
    operator: &LogicalOperator,
    left: &ExpressionType,
    right: &ExpressionType,
    ctx: &mut EvalContext,
) -> ValueResult {
    let left_val = evaluate_expression(left, ctx)?;

    match operator {
        LogicalOperator::And => {
            if !to_boolean(&left_val) {
                Ok(left_val)
            } else {
                evaluate_expression(right, ctx)
            }
        }
        LogicalOperator::Or => {
            if to_boolean(&left_val) {
                Ok(left_val)
            } else {
                evaluate_expression(right, ctx)
            }
        }
        // `a ?? b` — right side only when `a` is null/undefined.
        LogicalOperator::Coalesce => {
            if matches!(left_val, JsValue::Null | JsValue::Undefined) {
                evaluate_expression(right, ctx)
            } else {
                Ok(left_val)
            }
        }
    }
}

/// `a ** b` — exponentiation over JS numbers (via f64 `powf`). Integer bases/
/// exponents fold back to an integer when the result is exact.
fn exponent_values(left: &JsValue, right: &JsValue) -> ValueResult {
    use num_traits::Float;
    let a = to_number(left)?;
    let b = to_number(right)?;
    apply_numeric_op(
        &a,
        &b,
        |x, y| {
            if y >= 0 {
                (x as f64).powf(y as f64) as i64
            } else {
                0
            }
        },
        |x, y| x.powf(y),
    )
}

/// Evaluate a conditional (ternary) expression.
fn evaluate_conditional_expression(
    test: &ExpressionType,
    consequent: &ExpressionType,
    alternate: &ExpressionType,
    ctx: &mut EvalContext,
) -> ValueResult {
    let test_val = evaluate_expression(test, ctx)?;

    if to_boolean(&test_val) {
        evaluate_expression(consequent, ctx)
    } else {
        evaluate_expression(alternate, ctx)
    }
}

/// Evaluate a sequence expression.
fn evaluate_sequence_expression(
    expressions: &[Box<ExpressionType>],
    ctx: &mut EvalContext,
) -> ValueResult {
    let mut result = JsValue::Undefined;
    for expr in expressions {
        result = evaluate_expression(expr.as_ref(), ctx)?;
    }
    Ok(result)
}

/// Evaluate an update expression (++x, x++, --x, x--).
fn evaluate_update_expression(
    operator: &UpdateOperator,
    argument: &ExpressionType,
    prefix: bool,
    ctx: &mut EvalContext,
) -> ValueResult {
    // The update target is either a plain identifier or a member expression
    // (`obj.x++`, `arr[i]--`). Read the current value from whichever it is.
    let member_target: Option<(JsValue, String)> = match argument {
        ExpressionType::MemberExpression(member) => Some(match member {
            MemberExpressionType::SimpleMemberExpression { object, property, .. } => {
                let obj = evaluate_expression_or_super(object, ctx)?;
                (obj, property.name.clone())
            }
            MemberExpressionType::ComputedMemberExpression { object, property, .. } => {
                let obj = evaluate_expression_or_super(object, ctx)?;
                let prop_val = evaluate_expression(property.as_ref(), ctx)?;
                (obj, to_property_key(&prop_val))
            }
        }),
        _ => None,
    };

    let current_value = match &member_target {
        Some((obj, prop)) => get_property_with_ctx(obj, prop, ctx)?,
        None => {
            let name = get_expression_name(argument)?;
            ctx.get_binding(&name)?
        }
    };

    // Convert to number
    let old_number = to_number(&current_value)?;
    let old_f64 = match &old_number {
        JsValue::Number(n) => number_to_f64(n),
        _ => f64::NAN,
    };

    // Compute the new value based on operator
    let new_f64 = match operator {
        UpdateOperator::PlusPlus => old_f64 + 1.0,
        UpdateOperator::MinusMinus => old_f64 - 1.0,
    };

    // Create the new JsValue
    let new_value = if new_f64.fract() == 0.0 && new_f64.abs() < i64::MAX as f64 {
        JsValue::Number(JsNumberType::Integer(new_f64 as i64))
    } else {
        JsValue::Number(JsNumberType::Float(new_f64))
    };

    // Store the new value back into the target.
    match &member_target {
        Some((obj, prop)) => set_property_with_ctx(obj, prop, new_value.clone(), ctx)?,
        None => {
            let name = get_expression_name(argument)?;
            ctx.set_binding(&name, new_value.clone())?;
        }
    }

    // Return old value (coerced to number) for postfix, new value for prefix
    if prefix {
        Ok(new_value)
    } else {
        Ok(old_number)
    }
}

// ============================================================================
// Type conversion helpers
// ============================================================================

/// Convert a value to boolean.
pub fn to_boolean(value: &JsValue) -> bool {
    match value {
        JsValue::Undefined => false,
        JsValue::Null => false,
        JsValue::Boolean(b) => *b,
        JsValue::Number(n) => match n {
            JsNumberType::Integer(0) => false,
            JsNumberType::Float(f) if *f == 0.0 || f.is_nan() => false,
            JsNumberType::NaN => false,
            _ => true,
        },
        JsValue::String(s) => !s.is_empty(),
        JsValue::Symbol(_) => true,
        JsValue::Object(_) => true,
    }
}

/// Get the typeof string for a value.
fn get_typeof_string(value: &JsValue) -> String {
    match value {
        JsValue::Undefined => "undefined".to_string(),
        JsValue::Null => "object".to_string(),
        JsValue::Boolean(_) => "boolean".to_string(),
        JsValue::Number(_) => "number".to_string(),
        JsValue::String(_) => "string".to_string(),
        JsValue::Symbol(_) => "symbol".to_string(),
        JsValue::Object(_) => "object".to_string(),
    }
}

/// Convert a value to a number.
fn to_number(value: &JsValue) -> ValueResult {
    Ok(match value {
        JsValue::Undefined => JsValue::Number(JsNumberType::NaN),
        JsValue::Null => JsValue::Number(JsNumberType::Integer(0)),
        JsValue::Boolean(true) => JsValue::Number(JsNumberType::Integer(1)),
        JsValue::Boolean(false) => JsValue::Number(JsNumberType::Integer(0)),
        JsValue::Number(n) => JsValue::Number(n.clone()),
        JsValue::String(s) => {
            if s.is_empty() {
                JsValue::Number(JsNumberType::Integer(0))
            } else if let Ok(i) = s.trim().parse::<i64>() {
                JsValue::Number(JsNumberType::Integer(i))
            } else if let Ok(f) = s.trim().parse::<f64>() {
                JsValue::Number(JsNumberType::Float(f))
            } else {
                JsValue::Number(JsNumberType::NaN)
            }
        }
        JsValue::Symbol(_) => {
            return Err(JErrorType::TypeError("Cannot convert Symbol to number".to_string()));
        }
        JsValue::Object(_) => JsValue::Number(JsNumberType::NaN),
    })
}

/// Negate a number value.
fn negate_number(value: &JsValue) -> ValueResult {
    let num_value = to_number(value)?;
    Ok(match num_value {
        JsValue::Number(JsNumberType::Integer(i)) => JsValue::Number(JsNumberType::Integer(-i)),
        JsValue::Number(JsNumberType::Float(f)) => JsValue::Number(JsNumberType::Float(-f)),
        JsValue::Number(JsNumberType::PositiveInfinity) => JsValue::Number(JsNumberType::NegativeInfinity),
        JsValue::Number(JsNumberType::NegativeInfinity) => JsValue::Number(JsNumberType::PositiveInfinity),
        JsValue::Number(JsNumberType::NaN) => JsValue::Number(JsNumberType::NaN),
        _ => JsValue::Number(JsNumberType::NaN),
    })
}

/// Bitwise NOT operation.
fn bitwise_not(value: &JsValue) -> ValueResult {
    let num = to_i32(value)?;
    Ok(JsValue::Number(JsNumberType::Integer(!num as i64)))
}

/// Convert to i32 for bitwise operations.
fn to_i32(value: &JsValue) -> Result<i32, JErrorType> {
    match to_number(value)? {
        JsValue::Number(JsNumberType::Integer(i)) => Ok(i as i32),
        JsValue::Number(JsNumberType::Float(f)) => Ok(f as i32),
        JsValue::Number(JsNumberType::NaN) => Ok(0),
        JsValue::Number(JsNumberType::PositiveInfinity) => Ok(0),
        JsValue::Number(JsNumberType::NegativeInfinity) => Ok(0),
        _ => Ok(0),
    }
}

/// Convert to u32 for unsigned bitwise operations.
fn to_u32(value: &JsValue) -> Result<u32, JErrorType> {
    Ok(to_i32(value)? as u32)
}

// ============================================================================
// Arithmetic operations
// ============================================================================

fn add_values(left: &JsValue, right: &JsValue) -> ValueResult {
    // A Symbol operand always throws: after ToPrimitive it stays a Symbol, then
    // either ToString (string concat) or ToNumber (numeric add) rejects it.
    if matches!(left, JsValue::Symbol(_)) || matches!(right, JsValue::Symbol(_)) {
        return Err(JErrorType::TypeError(
            "Cannot convert a Symbol value".to_string(),
        ));
    }
    if matches!(left, JsValue::String(_)) || matches!(right, JsValue::String(_)) {
        let left_str = to_string(left);
        let right_str = to_string(right);
        return Ok(JsValue::String(format!("{}{}", left_str, right_str)));
    }

    let left_num = to_number(left)?;
    let right_num = to_number(right)?;
    apply_numeric_op(&left_num, &right_num, |a, b| a + b, |a, b| a + b)
}

fn subtract_values(left: &JsValue, right: &JsValue) -> ValueResult {
    let left_num = to_number(left)?;
    let right_num = to_number(right)?;
    apply_numeric_op(&left_num, &right_num, |a, b| a - b, |a, b| a - b)
}

fn multiply_values(left: &JsValue, right: &JsValue) -> ValueResult {
    let left_num = to_number(left)?;
    let right_num = to_number(right)?;
    apply_numeric_op(&left_num, &right_num, |a, b| a * b, |a, b| a * b)
}

fn divide_values(left: &JsValue, right: &JsValue) -> ValueResult {
    let left_num = to_number(left)?;
    let right_num = to_number(right)?;

    if matches!(right_num, JsValue::Number(JsNumberType::Integer(0)))
        || matches!(right_num, JsValue::Number(JsNumberType::Float(f)) if f == 0.0)
    {
        let left_f = match &left_num {
            JsValue::Number(n) => number_to_f64(n),
            _ => f64::NAN,
        };
        return Ok(if left_f.is_nan() || left_f == 0.0 {
            JsValue::Number(JsNumberType::NaN)
        } else if left_f > 0.0 {
            JsValue::Number(JsNumberType::PositiveInfinity)
        } else {
            JsValue::Number(JsNumberType::NegativeInfinity)
        });
    }

    apply_numeric_op(&left_num, &right_num, |a, b| a / b, |a, b| a / b)
}

fn modulo_values(left: &JsValue, right: &JsValue) -> ValueResult {
    let left_num = to_number(left)?;
    let right_num = to_number(right)?;
    apply_numeric_op(&left_num, &right_num, |a, b| a % b, |a, b| a % b)
}

fn apply_numeric_op<F, G>(left: &JsValue, right: &JsValue, int_op: F, float_op: G) -> ValueResult
where
    F: Fn(i64, i64) -> i64,
    G: Fn(f64, f64) -> f64,
{
    match (left, right) {
        (JsValue::Number(JsNumberType::NaN), _) | (_, JsValue::Number(JsNumberType::NaN)) => {
            Ok(JsValue::Number(JsNumberType::NaN))
        }
        (JsValue::Number(JsNumberType::Integer(a)), JsValue::Number(JsNumberType::Integer(b))) => {
            Ok(JsValue::Number(JsNumberType::Integer(int_op(*a, *b))))
        }
        (JsValue::Number(a), JsValue::Number(b)) => {
            let a_f64 = number_to_f64(a);
            let b_f64 = number_to_f64(b);
            Ok(JsValue::Number(JsNumberType::Float(float_op(a_f64, b_f64))))
        }
        _ => Ok(JsValue::Number(JsNumberType::NaN)),
    }
}

fn number_to_f64(n: &JsNumberType) -> f64 {
    match n {
        JsNumberType::Integer(i) => *i as f64,
        JsNumberType::Float(f) => *f,
        JsNumberType::NaN => f64::NAN,
        JsNumberType::PositiveInfinity => f64::INFINITY,
        JsNumberType::NegativeInfinity => f64::NEG_INFINITY,
    }
}

// ============================================================================
// Comparison operations
// ============================================================================

fn compare_values<F>(left: &JsValue, right: &JsValue, cmp: F) -> ValueResult
where
    F: Fn(f64, f64) -> bool,
{
    let left_num = to_number(left)?;
    let right_num = to_number(right)?;

    let result = match (&left_num, &right_num) {
        (JsValue::Number(JsNumberType::NaN), _) | (_, JsValue::Number(JsNumberType::NaN)) => false,
        (JsValue::Number(a), JsValue::Number(b)) => {
            cmp(number_to_f64(a), number_to_f64(b))
        }
        _ => false,
    };

    Ok(JsValue::Boolean(result))
}

pub fn strict_equality(left: &JsValue, right: &JsValue) -> bool {
    match (left, right) {
        (JsValue::Undefined, JsValue::Undefined) => true,
        (JsValue::Null, JsValue::Null) => true,
        (JsValue::Boolean(a), JsValue::Boolean(b)) => a == b,
        (JsValue::String(a), JsValue::String(b)) => a == b,
        (JsValue::Number(JsNumberType::NaN), _) | (_, JsValue::Number(JsNumberType::NaN)) => false,
        (JsValue::Number(a), JsValue::Number(b)) => {
            number_to_f64(a) == number_to_f64(b)
        }
        (JsValue::Object(a), JsValue::Object(b)) => alloc::rc::Rc::ptr_eq(a, b),
        _ => false,
    }
}

fn loose_equality(left: &JsValue, right: &JsValue) -> bool {
    if core::mem::discriminant(left) == core::mem::discriminant(right) {
        return strict_equality(left, right);
    }

    match (left, right) {
        (JsValue::Null, JsValue::Undefined) | (JsValue::Undefined, JsValue::Null) => true,
        (JsValue::Number(_), JsValue::String(_)) => {
            if let Ok(r) = to_number(right) {
                strict_equality(left, &r)
            } else {
                false
            }
        }
        (JsValue::String(_), JsValue::Number(_)) => {
            if let Ok(l) = to_number(left) {
                strict_equality(&l, right)
            } else {
                false
            }
        }
        (JsValue::Boolean(_), _) => {
            if let Ok(l) = to_number(left) {
                loose_equality(&l, right)
            } else {
                false
            }
        }
        (_, JsValue::Boolean(_)) => {
            if let Ok(r) = to_number(right) {
                loose_equality(left, &r)
            } else {
                false
            }
        }
        _ => false,
    }
}

// ============================================================================
// Bitwise operations
// ============================================================================

fn bitwise_and(left: &JsValue, right: &JsValue) -> ValueResult {
    let l = to_i32(left)?;
    let r = to_i32(right)?;
    Ok(JsValue::Number(JsNumberType::Integer((l & r) as i64)))
}

fn bitwise_or(left: &JsValue, right: &JsValue) -> ValueResult {
    let l = to_i32(left)?;
    let r = to_i32(right)?;
    Ok(JsValue::Number(JsNumberType::Integer((l | r) as i64)))
}

fn bitwise_xor(left: &JsValue, right: &JsValue) -> ValueResult {
    let l = to_i32(left)?;
    let r = to_i32(right)?;
    Ok(JsValue::Number(JsNumberType::Integer((l ^ r) as i64)))
}

fn left_shift(left: &JsValue, right: &JsValue) -> ValueResult {
    let l = to_i32(left)?;
    let r = to_u32(right)? & 0x1f;
    Ok(JsValue::Number(JsNumberType::Integer((l << r) as i64)))
}

fn right_shift(left: &JsValue, right: &JsValue) -> ValueResult {
    let l = to_i32(left)?;
    let r = to_u32(right)? & 0x1f;
    Ok(JsValue::Number(JsNumberType::Integer((l >> r) as i64)))
}

fn unsigned_right_shift(left: &JsValue, right: &JsValue) -> ValueResult {
    let l = to_u32(left)?;
    let r = to_u32(right)? & 0x1f;
    Ok(JsValue::Number(JsNumberType::Integer((l >> r) as i64)))
}

// ============================================================================
// String conversion
// ============================================================================

fn to_string(value: &JsValue) -> String {
    match value {
        JsValue::Undefined => "undefined".to_string(),
        JsValue::Null => "null".to_string(),
        JsValue::Boolean(true) => "true".to_string(),
        JsValue::Boolean(false) => "false".to_string(),
        JsValue::Number(n) => n.to_string(),
        JsValue::String(s) => s.clone(),
        JsValue::Symbol(_) => "[Symbol]".to_string(),
        JsValue::Object(_) => "[object Object]".to_string(),
    }
}

// ============================================================================
// Function object creation
// ============================================================================

/// Simple function object that stores function metadata for evaluation.
/// Instead of storing AST data directly, we store references to the source AST.
pub struct SimpleFunctionObject {
    base: ObjectBase,
    /// Pointer to the function body (using raw pointer for simplicity). Null for
    /// an expression-bodied arrow (`x => expr`), where `expr_body_ptr` is set.
    body_ptr: *const crate::parser::ast::FunctionBodyData,
    /// ChittiOS: for a concise arrow `x => expr`, the expression to evaluate and
    /// return. Null for ordinary functions and block-bodied arrows.
    expr_body_ptr: *const crate::parser::ast::ExpressionType,
    /// Pointer to the formal parameters
    params_ptr: *const Vec<PatternType>,
    /// The lexical environment at the time the function was created
    environment: crate::runner::ds::lex_env::JsLexEnvironmentType,
    /// ChittiOS: true for arrow functions — `this` is lexical (not rebound to
    /// the call receiver).
    is_arrow: bool,
    /// ChittiOS: true for `async function` — the return value is wrapped in a
    /// resolved Promise.
    is_async: bool,
}

// Safety: SimpleFunctionObject is only used within a single thread
unsafe impl Send for SimpleFunctionObject {}
unsafe impl Sync for SimpleFunctionObject {}

impl SimpleFunctionObject {
    pub fn new(
        body_ptr: *const crate::parser::ast::FunctionBodyData,
        params_ptr: *const Vec<PatternType>,
        environment: crate::runner::ds::lex_env::JsLexEnvironmentType,
    ) -> Self {
        SimpleFunctionObject {
            base: ObjectBase::new(),
            body_ptr,
            expr_body_ptr: core::ptr::null(),
            params_ptr,
            environment,
            is_arrow: false,
            is_async: false,
        }
    }

    /// ChittiOS: build an arrow-function object. `body_ptr` OR `expr_body_ptr`
    /// is set (exactly one non-null); `this` is lexical.
    pub fn new_arrow(
        body_ptr: *const crate::parser::ast::FunctionBodyData,
        expr_body_ptr: *const crate::parser::ast::ExpressionType,
        params_ptr: *const Vec<PatternType>,
        environment: crate::runner::ds::lex_env::JsLexEnvironmentType,
    ) -> Self {
        SimpleFunctionObject {
            base: ObjectBase::new(),
            body_ptr,
            expr_body_ptr,
            params_ptr,
            environment,
            is_arrow: true,
            is_async: false,
        }
    }

    /// Call this function with the given this value and arguments.
    /// Safety: The AST pointers must still be valid.
    pub fn call_with_this(
        &self,
        this_value: JsValue,
        args: Vec<JsValue>,
        ctx: &mut EvalContext,
    ) -> ValueResult {
        use crate::runner::ds::env_record::new_declarative_environment;
        use super::statement::execute_statement;
        use super::types::CompletionType;

        // Params always valid; body may be a statement block or (arrow) an expr.
        let params = unsafe { &*self.params_ptr };

        // Save current environment
        let saved_lex_env = ctx.lex_env.clone();
        let saved_var_env = ctx.var_env.clone();
        let saved_this = ctx.global_this.clone();

        // Create new environment for the function, with the function's closure as outer
        let func_scope = new_declarative_environment(Some(self.environment.clone()));
        ctx.lex_env = func_scope.clone();
        ctx.var_env = func_scope;
        // Arrow functions capture `this` lexically — leave the caller's `this`
        // untouched. Ordinary functions rebind `this` to the call receiver.
        if !self.is_arrow {
            ctx.global_this = Some(this_value);
        }

        // Bind parameters to arguments
        for (i, param) in params.iter().enumerate() {
            if let PatternType::PatternWhichCanBeExpression(
                ExpressionPatternType::Identifier(id)
            ) = param {
                let arg_value = args.get(i).cloned().unwrap_or(JsValue::Undefined);
                ctx.create_binding(&id.name, false)?;
                ctx.initialize_binding(&id.name, arg_value)?;
            }
        }

        // ChittiOS: concise arrow body `x => expr` — evaluate and return the expr
        // (an `async` concise arrow wraps the result in a resolved Promise).
        if !self.expr_body_ptr.is_null() {
            let expr = unsafe { &*self.expr_body_ptr };
            let out = evaluate_expression(expr, ctx);
            ctx.lex_env = saved_lex_env;
            ctx.var_env = saved_var_env;
            ctx.global_this = saved_this;
            return match out {
                Ok(v) if self.is_async => Ok(crate::runner::std_lib::promise::resolve_value(v)),
                other => other,
            };
        }

        let body = unsafe { &*self.body_ptr };

        // Hoist `var` declarations to the top of the function scope.
        super::statement::hoist_var_declarations(&body.body, ctx);

        // Execute each statement in the function body. As in
        // `call_function_with_body`, the environment must be restored on EVERY
        // exit path — including an `Err(..)` propagating from a nested call —
        // so the caller never continues against the callee's popped scope.
        let mut result_completion = super::types::Completion::normal();
        let mut pending_err: Option<JErrorType> = None;

        for stmt in body.body.iter() {
            match execute_statement(stmt, ctx) {
                Err(e) => {
                    pending_err = Some(e);
                    break;
                }
                Ok(completion) => match completion.completion_type {
                    CompletionType::Return => {
                        result_completion = completion;
                        break;
                    }
                    CompletionType::Throw => {
                        pending_err = Some(JErrorType::Thrown(
                            completion.value.clone().unwrap_or(JsValue::Undefined),
                        ));
                        break;
                    }
                    CompletionType::Break | CompletionType::Continue | CompletionType::Yield => {
                        result_completion = completion;
                        break;
                    }
                    CompletionType::Normal => {
                        result_completion = completion;
                    }
                },
            }
        }

        // Restore the previous environment on every path.
        ctx.lex_env = saved_lex_env;
        ctx.var_env = saved_var_env;
        ctx.global_this = saved_this;

        if let Some(e) = pending_err {
            // An `async function` turns a thrown error into a rejected Promise.
            if self.is_async {
                let v = match &e {
                    JErrorType::Thrown(v) => v.clone(),
                    other => JsValue::String(other.to_string()),
                };
                return Ok(crate::runner::std_lib::promise::reject_value(v));
            }
            return Err(e);
        }

        // Return the result. An `async function` resolves to a Promise of the
        // returned value (ChittiOS synchronous-settlement model).
        let ret = match result_completion.completion_type {
            CompletionType::Return => result_completion.get_value(),
            _ => JsValue::Undefined,
        };
        if self.is_async {
            Ok(crate::runner::std_lib::promise::resolve_value(ret))
        } else {
            Ok(ret)
        }
    }
}

impl JsObject for SimpleFunctionObject {
    fn get_object_base_mut(&mut self) -> &mut ObjectBase {
        &mut self.base
    }

    fn get_object_base(&self) -> &ObjectBase {
        &self.base
    }

    fn as_js_object(&self) -> &dyn JsObject {
        self
    }

    fn as_js_object_mut(&mut self) -> &mut dyn JsObject {
        self
    }
}

/// Type ID for SimpleFunctionObject - used for downcasting
pub fn is_simple_function_object(obj: &dyn JsObject) -> bool {
    // Check if the object has a special marker property
    let marker = PropertyKey::Str("__simple_function__".to_string());
    obj.get_object_base().properties.contains_key(&marker)
}

/// Create a function object from FunctionData.
/// Note: This creates a closure that references the AST. The AST must remain valid
/// for the lifetime of the function object.
/// Is this value callable (a function object / builtin)?
pub fn value_is_callable(v: &JsValue) -> bool {
    match v {
        JsValue::Object(o) => o.borrow().is_callable(),
        _ => false,
    }
}

/// Render a computed member key (`obj[expr]`) as a property-name string.
pub fn value_to_property_key(v: &JsValue) -> String {
    match v {
        JsValue::String(s) => s.clone(),
        JsValue::Number(JsNumberType::Integer(n)) => n.to_string(),
        JsValue::Number(JsNumberType::Float(f)) => {
            if f.fract() == 0.0 {
                (*f as i64).to_string()
            } else {
                f.to_string()
            }
        }
        other => other.to_string(),
    }
}

/// The registry object-name to use for method dispatch on a receiver: a builtin
/// sentinel's own `__builtin_name__` (Math/console/JSON/Object/Array/…), else
/// the primitive/host type (`Array`/`String`/`Number`/`Boolean`/`Object`).
pub fn builtin_type_name(v: &JsValue) -> String {
    match v {
        JsValue::String(_) => "String".to_string(),
        JsValue::Number(_) => "Number".to_string(),
        JsValue::Boolean(_) => "Boolean".to_string(),
        JsValue::Object(o) => {
            let b = o.borrow();
            let base = b.as_js_object().get_object_base();
            if let Some(PropertyDescriptor::Data(d)) =
                base.properties.get(&PropertyKey::Str("__builtin_name__".to_string()))
            {
                if let JsValue::String(name) = &d.value {
                    return name.clone();
                }
            }
            if base
                .properties
                .contains_key(&PropertyKey::Str("__array__".to_string()))
            {
                "Array".to_string()
            } else {
                "Object".to_string()
            }
        }
        _ => "Object".to_string(),
    }
}

/// If `v` carries an `__native_node__` integer (a host-backed object such as a
/// live DOM element view), return the node id.
pub fn native_node(v: &JsValue) -> Option<i64> {
    match get_own_prop_value(v, "__native_node__") {
        Some(JsValue::Number(JsNumberType::Integer(n))) => Some(n),
        Some(JsValue::Number(JsNumberType::Float(f))) => Some(f as i64),
        _ => None,
    }
}

/// If `v` is a `Proxy`, return its `(target, handler)`.
pub fn proxy_parts(v: &JsValue) -> Option<(JsValue, JsValue)> {
    let target = get_own_prop_value(v, "__proxy_target__")?;
    let handler = get_own_prop_value(v, "__proxy_handler__")?;
    Some((target, handler))
}

/// Public property get (walks the prototype chain, runs getters/Proxy traps) —
/// used by `Reflect.get` and DOM bindings.
pub fn get_property(value: &JsValue, prop: &str, ctx: &mut EvalContext) -> ValueResult {
    get_property_with_ctx(value, prop, ctx)
}

/// Public property set (runs setters/Proxy traps) — used by `Reflect.set`.
pub fn set_property(value: &JsValue, prop: &str, new_value: JsValue, ctx: &mut EvalContext) -> Result<(), JErrorType> {
    set_property_with_ctx(value, prop, new_value, ctx)
}

/// Delete an own property; returns whether the object had it.
pub fn delete_own_prop(v: &JsValue, name: &str) -> bool {
    if let JsValue::Object(o) = v {
        return o
            .borrow_mut()
            .as_js_object_mut()
            .get_object_base_mut()
            .properties
            .remove(&PropertyKey::Str(name.to_string()))
            .is_some();
    }
    false
}

/// All own string property keys (for `Reflect.ownKeys`), excluding internal
/// `__…__` markers.
pub fn own_string_keys(v: &JsValue) -> Vec<alloc::string::String> {
    let mut out = Vec::new();
    if let JsValue::Object(o) = v {
        let b = o.borrow();
        for (k, _) in b.as_js_object().get_object_base().properties.iter() {
            if let PropertyKey::Str(name) = k {
                if !(name.starts_with("__") && name.ends_with("__")) {
                    out.push(name.clone());
                }
            }
        }
    }
    out
}

/// Set an own data property on an object value (no-op if `v` isn't an object).
pub fn set_own_prop(v: &JsValue, name: &str, value: JsValue, enumerable: bool) {
    if let JsValue::Object(o) = v {
        let mut b = o.borrow_mut();
        b.as_js_object_mut().get_object_base_mut().properties.insert(
            PropertyKey::Str(name.to_string()),
            PropertyDescriptor::Data(PropertyDescriptorData {
                value,
                writable: true,
                enumerable,
                configurable: true,
            }),
        );
    }
}

/// Read an own property value off an object value (`None` if absent/non-object).
pub fn get_own_prop_value(v: &JsValue, name: &str) -> Option<JsValue> {
    if let JsValue::Object(o) = v {
        let b = o.borrow();
        if let Some(PropertyDescriptor::Data(d)) = b
            .as_js_object()
            .get_object_base()
            .properties
            .get(&PropertyKey::Str(name.to_string()))
        {
            return Some(d.value.clone());
        }
    }
    None
}

/// Read an own **data** property's full descriptor as
/// `(value, writable, enumerable, configurable)`. `None` if absent, an
/// accessor, or `v` isn't an object. Used by `Object.getOwnPropertyDescriptor`
/// so it reports real attributes instead of assuming everything is
/// writable/enumerable/configurable.
pub fn get_own_prop_descriptor(v: &JsValue, name: &str) -> Option<(JsValue, bool, bool, bool)> {
    if let JsValue::Object(o) = v {
        let b = o.borrow();
        if let Some(PropertyDescriptor::Data(d)) = b
            .as_js_object()
            .get_object_base()
            .properties
            .get(&PropertyKey::Str(name.to_string()))
        {
            return Some((d.value.clone(), d.writable, d.enumerable, d.configurable));
        }
    }
    None
}

pub fn has_own_prop(v: &JsValue, name: &str) -> bool {
    if let JsValue::Object(o) = v {
        return o
            .borrow()
            .as_js_object()
            .get_object_base()
            .properties
            .contains_key(&PropertyKey::Str(name.to_string()));
    }
    false
}

/// ChittiOS: invoke a class constructor against `this_value`, threading the
/// superclass chain so `super(...)` works and a derived class with no explicit
/// constructor implicitly calls `super(...args)`. Depth-recursive.
pub fn invoke_constructor(
    ctor: &JsValue,
    this_value: JsValue,
    args: Vec<JsValue>,
    ctx: &mut EvalContext,
) -> ValueResult {
    let parent = get_own_prop_value(ctor, "__parent_constructor__");
    let is_default = has_own_prop(ctor, "__default_constructor__");

    // Implicit `super(...args)` for a derived class with no explicit constructor.
    if is_default {
        if let Some(p) = &parent {
            invoke_constructor(p, this_value.clone(), args.clone(), ctx)?;
        }
    }

    let saved = ctx.current_super.take();
    ctx.current_super = parent;
    let result = call_value(ctor, this_value, args, ctx);
    ctx.current_super = saved;
    result
}

/// ChittiOS: create an arrow-function object. Reuses `SimpleFunctionObject`
/// (with lexical `this`); a concise `x => expr` body is stored as an expression
/// pointer, a block body `(x) => { … }` as a statement-body pointer. The AST
/// outlives the object exactly as for ordinary functions.
pub fn create_arrow_function_object(
    params: &Vec<PatternType>,
    body: &crate::parser::ast::FunctionBodyOrExpression,
    is_async: bool,
    ctx: &EvalContext,
) -> ValueResult {
    use crate::parser::ast::FunctionBodyOrExpression;
    let params_ptr = params as *const _;
    let (body_ptr, expr_ptr): (
        *const crate::parser::ast::FunctionBodyData,
        *const crate::parser::ast::ExpressionType,
    ) = match body {
        FunctionBodyOrExpression::FunctionBody(fbd) => (fbd as *const _, core::ptr::null()),
        FunctionBodyOrExpression::Expression(expr) => (core::ptr::null(), expr as *const _),
    };
    let mut func_obj =
        SimpleFunctionObject::new_arrow(body_ptr, expr_ptr, params_ptr, ctx.lex_env.clone());
    func_obj.is_async = is_async;
    // Callable marker (same as ordinary functions).
    func_obj.base.properties.insert(
        PropertyKey::Str("__simple_function__".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Boolean(true),
            writable: false,
            enumerable: false,
            configurable: false,
        }),
    );
    // Arrows start with the empty name (may be inferred at a binding site) and
    // a `length` of their leading simple parameters.
    func_obj.base.properties.insert(
        PropertyKey::Str("name".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::String(String::new()),
            writable: false,
            enumerable: false,
            configurable: true,
        }),
    );
    func_obj.base.properties.insert(
        PropertyKey::Str("length".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Number(JsNumberType::Integer(function_length(params))),
            writable: false,
            enumerable: false,
            configurable: true,
        }),
    );
    let obj_ref: JsObjectType = Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(func_obj))));
    Ok(JsValue::Object(obj_ref))
}

/// Build a bound function (`fn.bind(thisArg, …partial)`): a callable object
/// carrying the target, bound `this`, and partial args. `call_function_object`
/// detects the `__bound_target__` marker and forwards the call.
pub fn make_bound_function(target: JsValue, this: JsValue, partial: Vec<JsValue>) -> JsValue {
    let f = make_object(vec![]);
    // `__simple_function__` makes `is_callable()` / `value_is_callable()` true.
    set_own_prop(&f, "__simple_function__", JsValue::Boolean(true), false);
    set_own_prop(&f, "__bound_target__", target, false);
    set_own_prop(&f, "__bound_this__", this, false);
    set_own_prop(&f, "__bound_args__", make_array(partial), false);
    f
}

/// The `length` of a function: the number of leading formal parameters that are
/// plain identifiers, stopping at the first one with a default, a rest element,
/// or a destructuring pattern (per the spec's ExpectedArgumentCount).
fn function_length(params: &[PatternType]) -> i64 {
    let mut n = 0i64;
    for p in params {
        match p {
            PatternType::PatternWhichCanBeExpression(ExpressionPatternType::Identifier(_)) => {
                n += 1
            }
            _ => break,
        }
    }
    n
}

/// True when `expr` is a syntactically anonymous function definition (an
/// unnamed function/generator expression, any arrow, or an unnamed class
/// expression) — the only RHS forms that trigger name inference.
pub fn is_anonymous_function_expr(expr: &ExpressionType) -> bool {
    match expr {
        ExpressionType::FunctionOrGeneratorExpression(d) => d.id.is_none(),
        ExpressionType::ArrowFunctionExpression { .. } => true,
        ExpressionType::ClassExpression(c) => c.id.is_none(),
        _ => false,
    }
}

/// If `value` is a function whose `name` is still the empty string, set it to
/// `name` (anonymous-function name inference for `var f = function(){}`,
/// `f = () => {}`, `[f = function(){}] = []`, etc.).
pub fn infer_function_name(value: &JsValue, name: &str) {
    if name.is_empty() {
        return;
    }
    if let JsValue::Object(o) = value {
        let mut b = o.borrow_mut();
        let base = b.as_js_object_mut().get_object_base_mut();
        // Only for callables that carry the simple-function marker.
        let is_fn = matches!(
            base.properties.get(&PropertyKey::Str("__simple_function__".to_string())),
            Some(PropertyDescriptor::Data(_))
        );
        if !is_fn {
            return;
        }
        let empty = matches!(
            base.properties.get(&PropertyKey::Str("name".to_string())),
            Some(PropertyDescriptor::Data(d)) if matches!(&d.value, JsValue::String(s) if s.is_empty())
        );
        if empty {
            base.properties.insert(
                PropertyKey::Str("name".to_string()),
                PropertyDescriptor::Data(PropertyDescriptorData {
                    value: JsValue::String(name.to_string()),
                    writable: false,
                    enumerable: false,
                    configurable: true,
                }),
            );
        }
    }
}

pub fn create_function_object(func_data: &FunctionData, ctx: &EvalContext) -> ValueResult {
    let body_ptr = func_data.body.as_ref() as *const _;
    let params_ptr = &func_data.params.list as *const _;
    let environment = ctx.lex_env.clone();

    let mut func_obj = SimpleFunctionObject::new(body_ptr, params_ptr, environment);
    func_obj.is_async = func_data.is_async;

    // Create a prototype object for this function
    let prototype = SimpleObject::new();
    let prototype_ref: JsObjectType = Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(prototype))));

    // Store prototype on the function object
    func_obj.base.properties.insert(
        PropertyKey::Str("prototype".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Object(prototype_ref),
            writable: true,
            enumerable: false,
            configurable: false,
        }),
    );

    // Add marker property to identify this as a SimpleFunctionObject
    func_obj.base.properties.insert(
        PropertyKey::Str("__simple_function__".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Boolean(true),
            writable: false,
            enumerable: false,
            configurable: false,
        }),
    );

    // `name` (the declared identifier, else empty) and `length` (count of
    // leading simple parameters, i.e. before the first default/rest/pattern) —
    // both non-enumerable, configurable, per spec.
    let name = func_data.id.as_ref().map(|i| i.name.clone()).unwrap_or_default();
    func_obj.base.properties.insert(
        PropertyKey::Str("name".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::String(name),
            writable: false,
            enumerable: false,
            configurable: true,
        }),
    );
    func_obj.base.properties.insert(
        PropertyKey::Str("length".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Number(JsNumberType::Integer(function_length(&func_data.params.list))),
            writable: false,
            enumerable: false,
            configurable: true,
        }),
    );

    // Mark as generator if applicable
    if func_data.generator {
        func_obj.base.properties.insert(
            PropertyKey::Str("__generator__".to_string()),
            PropertyDescriptor::Data(PropertyDescriptorData {
                value: JsValue::Boolean(true),
                writable: false,
                enumerable: false,
                configurable: false,
            }),
        );
    }

    let obj_ref: JsObjectType = Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(func_obj))));
    Ok(JsValue::Object(obj_ref))
}

// ============================================================================
// Generator object
// ============================================================================

/// Generator state.
#[derive(Debug, Clone, PartialEq)]
pub enum GeneratorState {
    SuspendedStart,
    SuspendedYield,
    Executing,
    Completed,
}

/// Generator object that stores the suspended state of a generator function.
pub struct GeneratorObject {
    base: ObjectBase,
    /// Pointer to the function body
    body_ptr: *const crate::parser::ast::FunctionBodyData,
    /// Pointer to the formal parameters
    _params_ptr: *const Vec<PatternType>,
    /// The lexical environment at the time the generator was created
    environment: crate::runner::ds::lex_env::JsLexEnvironmentType,
    /// Current state
    state: GeneratorState,
    /// Current statement index (for resuming)
    current_index: usize,
    /// Stored local bindings (for resuming)
    local_bindings: hashbrown::HashMap<String, JsValue>,
}

// Safety: GeneratorObject is only used within a single thread
unsafe impl Send for GeneratorObject {}
unsafe impl Sync for GeneratorObject {}

impl GeneratorObject {
    pub fn new(
        body_ptr: *const crate::parser::ast::FunctionBodyData,
        params_ptr: *const Vec<PatternType>,
        environment: crate::runner::ds::lex_env::JsLexEnvironmentType,
    ) -> Self {
        GeneratorObject {
            base: ObjectBase::new(),
            body_ptr,
            _params_ptr: params_ptr,
            environment,
            state: GeneratorState::SuspendedStart,
            current_index: 0,
            local_bindings: hashbrown::HashMap::new(),
        }
    }

    /// Call .next() on this generator, executing until yield or completion.
    pub fn next(&mut self, ctx: &mut EvalContext) -> ValueResult {
        use crate::runner::ds::env_record::new_declarative_environment;
        use super::statement::execute_statement;
        use super::types::CompletionType;

        match self.state {
            GeneratorState::Completed => {
                // Return { value: undefined, done: true }
                return create_iterator_result(JsValue::Undefined, true);
            }
            GeneratorState::Executing => {
                return Err(JErrorType::TypeError("Generator is already executing".to_string()));
            }
            GeneratorState::SuspendedStart | GeneratorState::SuspendedYield => {
                // Continue or start execution
            }
        }

        self.state = GeneratorState::Executing;

        // Get the body and params (unsafe dereference)
        let body = unsafe { &*self.body_ptr };

        // Save current environment
        let saved_lex_env = ctx.lex_env.clone();
        let saved_var_env = ctx.var_env.clone();

        // Create new environment for the generator (or restore previous)
        let func_scope = new_declarative_environment(Some(self.environment.clone()));
        ctx.lex_env = func_scope.clone();
        ctx.var_env = func_scope;

        // Restore local bindings if resuming
        for (name, value) in &self.local_bindings {
            let _ = ctx.create_binding(name, false);
            let _ = ctx.initialize_binding(name, value.clone());
        }

        // Execute statements starting from current_index
        let mut result_value = JsValue::Undefined;
        let mut yielded = false;

        for (idx, stmt) in body.body.iter().enumerate() {
            if idx < self.current_index {
                continue; // Skip already executed statements
            }

            match execute_statement(stmt, ctx) {
                Ok(completion) => {
                    match completion.completion_type {
                        CompletionType::Return => {
                            result_value = completion.get_value();
                            self.state = GeneratorState::Completed;
                            break;
                        }
                        CompletionType::Yield => {
                            // Yield the value and suspend
                            result_value = completion.get_value();
                            self.current_index = idx + 1;
                            self.state = GeneratorState::SuspendedYield;

                            // Save current bindings
                            self.save_bindings(ctx);

                            yielded = true;
                            break;
                        }
                        CompletionType::Throw => {
                            ctx.lex_env = saved_lex_env;
                            ctx.var_env = saved_var_env;
                            self.state = GeneratorState::Completed;
                            return Err(JErrorType::Thrown(completion.value.clone().unwrap_or(JsValue::Undefined)));
                        }
                        _ => {
                            result_value = completion.get_value();
                        }
                    }
                }
                Err(JErrorType::YieldValue(value)) => {
                    // Yield expression hit
                    result_value = value;
                    self.current_index = idx + 1;
                    self.state = GeneratorState::SuspendedYield;

                    // Save current bindings
                    self.save_bindings(ctx);

                    yielded = true;
                    break;
                }
                Err(e) => {
                    ctx.lex_env = saved_lex_env;
                    ctx.var_env = saved_var_env;
                    self.state = GeneratorState::Completed;
                    return Err(e);
                }
            }
        }

        // If we didn't yield, we're done
        if !yielded && self.state == GeneratorState::Executing {
            self.state = GeneratorState::Completed;
        }

        // Restore the previous environment
        ctx.lex_env = saved_lex_env;
        ctx.var_env = saved_var_env;

        // Return { value, done }
        let done = self.state == GeneratorState::Completed;
        create_iterator_result(result_value, done)
    }

    fn save_bindings(&mut self, ctx: &EvalContext) {
        // This is a simplified version - in a full implementation we'd need
        // to properly capture all bindings from the current scope
        self.local_bindings.clear();

        // Get bindings from current environment
        let env = ctx.lex_env.borrow();
        if let Some(bindings) = env.inner.as_env_record().get_all_bindings() {
            for (name, value) in bindings {
                self.local_bindings.insert(name, value);
            }
        }
    }
}

impl JsObject for GeneratorObject {
    fn get_object_base_mut(&mut self) -> &mut ObjectBase {
        &mut self.base
    }

    fn get_object_base(&self) -> &ObjectBase {
        &self.base
    }

    fn as_js_object(&self) -> &dyn JsObject {
        self
    }

    fn as_js_object_mut(&mut self) -> &mut dyn JsObject {
        self
    }
}

/// Create an iterator result object { value, done }.
fn create_iterator_result(value: JsValue, done: bool) -> ValueResult {
    let mut obj = SimpleObject::new();

    obj.get_object_base_mut().properties.insert(
        PropertyKey::Str("value".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value,
            writable: true,
            enumerable: true,
            configurable: true,
        }),
    );

    obj.get_object_base_mut().properties.insert(
        PropertyKey::Str("done".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Boolean(done),
            writable: true,
            enumerable: true,
            configurable: true,
        }),
    );

    let obj_ref: JsObjectType = Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(obj))));
    Ok(JsValue::Object(obj_ref))
}

/// Create a generator object from a generator function.
fn create_generator_object(
    body_ptr: *const crate::parser::ast::FunctionBodyData,
    params_ptr: *const Vec<PatternType>,
    environment: crate::runner::ds::lex_env::JsLexEnvironmentType,
    _args: Vec<JsValue>,
    _ctx: &mut EvalContext,
) -> ValueResult {
    let mut gen_obj = GeneratorObject::new(body_ptr, params_ptr, environment);

    // Add marker property
    gen_obj.base.properties.insert(
        PropertyKey::Str("__generator_object__".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Boolean(true),
            writable: false,
            enumerable: false,
            configurable: false,
        }),
    );

    // Store parameter values in local_bindings for when .next() is first called
    let params = unsafe { &*params_ptr };
    for (i, param) in params.iter().enumerate() {
        if let PatternType::PatternWhichCanBeExpression(
            ExpressionPatternType::Identifier(id)
        ) = param {
            let arg_value = _args.get(i).cloned().unwrap_or(JsValue::Undefined);
            gen_obj.local_bindings.insert(id.name.clone(), arg_value);
        }
    }

    // Create a 'next' method
    // We store a reference to the generator in a closure
    let gen_ref: JsObjectType = Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(gen_obj))));

    // Add next method - this is a bit hacky, we'll store the generator ref
    // Actually, we can't easily create a closure here. Let's use a different approach:
    // We'll mark the generator and handle .next() specially in property access.

    // For simplicity, we mark it and handle next() calls specially
    Ok(JsValue::Object(gen_ref))
}

/// Create a callable "next" method for a generator object.
fn create_generator_next_method(gen_obj: JsObjectType) -> ValueResult {
    // Create a simple object that holds the generator reference
    // and is marked as a generator-next-method
    let mut next_obj = SimpleObject::new();

    next_obj.get_object_base_mut().properties.insert(
        PropertyKey::Str("__generator_next__".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Object(gen_obj),
            writable: false,
            enumerable: false,
            configurable: false,
        }),
    );

    // Mark it as callable
    next_obj.get_object_base_mut().properties.insert(
        PropertyKey::Str("__simple_function__".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Boolean(true),
            writable: false,
            enumerable: false,
            configurable: false,
        }),
    );

    let obj_ref: JsObjectType = Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(next_obj))));
    Ok(JsValue::Object(obj_ref))
}

// ============================================================================
// Class evaluation
// ============================================================================

/// Evaluate a class expression and return the constructor function.
pub fn evaluate_class_expression(class_data: &ClassData, ctx: &mut EvalContext) -> ValueResult {
    evaluate_class(class_data, ctx)
}

/// Evaluate a class definition (used by both ClassExpression and ClassDeclaration).
/// Returns a constructor function object with methods on its prototype.
pub fn evaluate_class(class_data: &ClassData, ctx: &mut EvalContext) -> ValueResult {
    // 1. Evaluate the super class if present
    let parent_proto = if let Some(super_class) = &class_data.super_class {
        let parent = evaluate_expression(super_class, ctx)?;
        match &parent {
            JsValue::Object(parent_obj) => {
                // Get parent.prototype
                let borrowed = parent_obj.borrow();
                let proto_key = PropertyKey::Str("prototype".to_string());
                if let Some(PropertyDescriptor::Data(data)) = borrowed.as_js_object().get_object_base().properties.get(&proto_key) {
                    if let JsValue::Object(proto) = &data.value {
                        Some((parent_obj.clone(), proto.clone()))
                    } else {
                        return Err(JErrorType::TypeError("Parent class prototype is not an object".to_string()));
                    }
                } else {
                    return Err(JErrorType::TypeError("Parent class has no prototype".to_string()));
                }
            }
            _ => return Err(JErrorType::TypeError("Class extends value is not a constructor".to_string())),
        }
    } else {
        None
    };

    // 2. Find the constructor method
    let constructor_method = class_data.body.body.iter()
        .find(|m| matches!(m.kind, MethodDefinitionKind::Constructor));

    // 3. Create the constructor function
    let constructor_obj = if let Some(ctor_method) = constructor_method {
        // Use the defined constructor
        create_class_constructor(&ctor_method.value, parent_proto.as_ref().map(|(p, _)| p.clone()), ctx)?
    } else {
        // Create a default constructor
        create_default_constructor(parent_proto.as_ref().map(|(p, _)| p.clone()), ctx)?
    };

    // 4. Create the prototype object
    let prototype = if let Some((_, parent_proto)) = &parent_proto {
        // Derived class: prototype inherits from parent prototype
        let mut proto_obj = SimpleObject::new();
        proto_obj.get_object_base_mut().prototype = Some(parent_proto.clone());
        Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(proto_obj))))
    } else {
        // Base class: regular prototype object
        Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(SimpleObject::new()))))
    };

    // 5. Add methods to prototype (and static methods to constructor)
    for method in &class_data.body.body {
        if matches!(method.kind, MethodDefinitionKind::Constructor) {
            continue; // Skip constructor, already handled
        }

        // Get the method key
        let key = get_object_property_key(&method.key, method.computed, ctx)?;

        // Create the method function
        let method_fn = create_function_object(&method.value, ctx)?;

        // Determine target: prototype for instance methods, constructor for static
        let target = if method.static_flag {
            &constructor_obj
        } else {
            &JsValue::Object(prototype.clone())
        };

        match method.kind {
            MethodDefinitionKind::Method => {
                // Regular method
                if let JsValue::Object(target_obj) = target {
                    target_obj.borrow_mut().as_js_object_mut().get_object_base_mut().properties.insert(
                        PropertyKey::Str(key),
                        PropertyDescriptor::Data(PropertyDescriptorData {
                            value: method_fn,
                            writable: true,
                            enumerable: false,
                            configurable: true,
                        }),
                    );
                }
            }
            MethodDefinitionKind::Get => {
                // Getter
                if let JsValue::Object(getter_fn) = method_fn {
                    if let JsValue::Object(target_obj) = target {
                        let mut borrowed = target_obj.borrow_mut();
                        let prop_key = PropertyKey::Str(key.clone());
                        let existing = borrowed.as_js_object().get_object_base().properties.get(&prop_key).cloned();

                        let setter = if let Some(PropertyDescriptor::Accessor(acc)) = existing {
                            acc.set
                        } else {
                            None
                        };

                        borrowed.as_js_object_mut().get_object_base_mut().properties.insert(
                            prop_key,
                            PropertyDescriptor::Accessor(PropertyDescriptorAccessor {
                                get: Some(getter_fn),
                                set: setter,
                                enumerable: false,
                                configurable: true,
                            }),
                        );
                    }
                }
            }
            MethodDefinitionKind::Set => {
                // Setter
                if let JsValue::Object(setter_fn) = method_fn {
                    if let JsValue::Object(target_obj) = target {
                        let mut borrowed = target_obj.borrow_mut();
                        let prop_key = PropertyKey::Str(key.clone());
                        let existing = borrowed.as_js_object().get_object_base().properties.get(&prop_key).cloned();

                        let getter = if let Some(PropertyDescriptor::Accessor(acc)) = existing {
                            acc.get
                        } else {
                            None
                        };

                        borrowed.as_js_object_mut().get_object_base_mut().properties.insert(
                            prop_key,
                            PropertyDescriptor::Accessor(PropertyDescriptorAccessor {
                                get: getter,
                                set: Some(setter_fn),
                                enumerable: false,
                                configurable: true,
                            }),
                        );
                    }
                }
            }
            MethodDefinitionKind::Constructor => unreachable!(),
        }
    }

    // 6. Wire up constructor.prototype
    if let JsValue::Object(ctor_obj) = &constructor_obj {
        ctor_obj.borrow_mut().as_js_object_mut().get_object_base_mut().properties.insert(
            PropertyKey::Str("prototype".to_string()),
            PropertyDescriptor::Data(PropertyDescriptorData {
                value: JsValue::Object(prototype.clone()),
                writable: false,
                enumerable: false,
                configurable: false,
            }),
        );

        // Set prototype.constructor to point back to constructor
        prototype.borrow_mut().as_js_object_mut().get_object_base_mut().properties.insert(
            PropertyKey::Str("constructor".to_string()),
            PropertyDescriptor::Data(PropertyDescriptorData {
                value: constructor_obj.clone(),
                writable: true,
                enumerable: false,
                configurable: true,
            }),
        );
    }

    Ok(constructor_obj)
}

/// Create a constructor function from a method definition.
fn create_class_constructor(
    func_data: &FunctionData,
    parent_ctor: Option<JsObjectType>,
    ctx: &EvalContext,
) -> ValueResult {
    let body_ptr = func_data.body.as_ref() as *const _;
    let params_ptr = &func_data.params.list as *const _;
    let environment = ctx.lex_env.clone();

    let mut func_obj = SimpleFunctionObject::new(body_ptr, params_ptr, environment);

    // Add marker property to identify this as a SimpleFunctionObject (callable)
    func_obj.base.properties.insert(
        PropertyKey::Str("__simple_function__".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Boolean(true),
            writable: false,
            enumerable: false,
            configurable: false,
        }),
    );

    // Mark as class constructor
    func_obj.base.properties.insert(
        PropertyKey::Str("__class_constructor__".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Boolean(true),
            writable: false,
            enumerable: false,
            configurable: false,
        }),
    );

    // ChittiOS: remember the superclass constructor so `super(...)` and derived
    // instantiation can invoke it (see `invoke_constructor`).
    if let Some(parent) = parent_ctor {
        func_obj.base.properties.insert(
            PropertyKey::Str("__parent_constructor__".to_string()),
            PropertyDescriptor::Data(PropertyDescriptorData {
                value: JsValue::Object(parent),
                writable: false,
                enumerable: false,
                configurable: false,
            }),
        );
    }

    let obj_ref: JsObjectType = Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(func_obj))));
    Ok(JsValue::Object(obj_ref))
}

/// Create a default constructor for a class.
fn create_default_constructor(
    parent_ctor: Option<JsObjectType>,
    _ctx: &EvalContext,
) -> ValueResult {
    // A no-op body; for a derived class, `invoke_constructor` calls the parent
    // constructor with the same args (implicit `super(...args)`).
    let mut func_obj = SimpleObject::new();
    if let Some(parent) = parent_ctor {
        func_obj.get_object_base_mut().properties.insert(
            PropertyKey::Str("__parent_constructor__".to_string()),
            PropertyDescriptor::Data(PropertyDescriptorData {
                value: JsValue::Object(parent),
                writable: false,
                enumerable: false,
                configurable: false,
            }),
        );
    }

    // Add marker property to identify this as callable
    func_obj.get_object_base_mut().properties.insert(
        PropertyKey::Str("__simple_function__".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Boolean(true),
            writable: false,
            enumerable: false,
            configurable: false,
        }),
    );

    // Mark as class constructor
    func_obj.get_object_base_mut().properties.insert(
        PropertyKey::Str("__class_constructor__".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Boolean(true),
            writable: false,
            enumerable: false,
            configurable: false,
        }),
    );

    // Mark as default constructor (no-op)
    func_obj.get_object_base_mut().properties.insert(
        PropertyKey::Str("__default_constructor__".to_string()),
        PropertyDescriptor::Data(PropertyDescriptorData {
            value: JsValue::Boolean(true),
            writable: false,
            enumerable: false,
            configurable: false,
        }),
    );

    let obj_ref: JsObjectType = Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(func_obj))));
    Ok(JsValue::Object(obj_ref))
}

