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

        // ChittiOS: `import(spec)` — the interpreter has no module loader, so we
        // evaluate the specifier (for its side effects/validity) and resolve to
        // an empty namespace object. A page's `import(x).then(m => …)` then runs
        // with `m = {}` rather than crashing. (The browser tier statically
        // pre-fetches + flattens module graphs; true dynamic loading is bounded.)
        ExpressionType::ImportCall { argument, .. } => {
            let _ = evaluate_expression(argument, ctx)?;
            Ok(crate::runner::std_lib::promise::resolve_value(make_object(alloc::vec![])))
        }

        // ChittiOS: `import.meta` — a stub meta-object (`{ url: "" }`); the
        // engine doesn't track the module URL.
        ExpressionType::ImportMeta { .. } => {
            Ok(make_object(alloc::vec![(
                "url".to_string(),
                JsValue::String(String::new()),
            )]))
        }

        ExpressionType::ArrayExpression { elements, .. } => {
            evaluate_array_expression(elements, ctx)
        }

        ExpressionType::ObjectExpression { properties, .. } => {
            evaluate_object_expression(properties, ctx)
        }

        ExpressionType::FunctionOrGeneratorExpression(func_data) => {
            // A *named* function expression binds its own name (immutably) in a
            // fresh scope visible only inside its body, so it can recurse even
            // when the outer binding is reassigned — `var f = function fact(n){
            // … fact(n-1) … }`. (Function *declarations* bind their name in the
            // enclosing scope via hoisting and take the plain path below.)
            if let Some(id) = &func_data.id {
                use crate::runner::ds::env_record::new_declarative_environment;
                let self_scope = new_declarative_environment(Some(ctx.lex_env.clone()));
                let saved = core::mem::replace(&mut ctx.lex_env, self_scope);
                let f = create_function_object(func_data, ctx)?;
                // Bind the name in the self-scope (now `ctx.lex_env`) to the fn.
                let _ = ctx.create_binding(&id.name, true);
                let _ = ctx.initialize_binding(&id.name, f.clone());
                ctx.lex_env = saved;
                Ok(f)
            } else {
                create_function_object(func_data, ctx)
            }
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

        ExpressionType::TemplateLiteral(data) => {
            // Interleave cooked quasis with the string-coerced substitution
            // values: quasi[0] expr[0] quasi[1] expr[1] … quasi[n].
            let mut out = String::new();
            for (i, quasi) in data.quasis.iter().enumerate() {
                out.push_str(&quasi.cooked_value);
                if let Some(expr) = data.expressions.get(i) {
                    let v = evaluate_expression(expr, ctx)?;
                    // A template substitution coerces via ToString, which for an
                    // object first runs ToPrimitive(string).
                    let v = to_primitive(&v, "string", ctx)?;
                    out.push_str(&to_string(&v));
                }
            }
            Ok(JsValue::String(out))
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
        LiteralType::BigIntLiteral(s) => {
            // Stored as the canonical decimal string at parse time.
            JsValue::BigInt(s.parse().unwrap_or_else(|_| num_bigint::BigInt::from(0)))
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
/// Most elements any single array operation will materialize.
///
/// A JS `length` is attacker-controlled (`Array.prototype.filter.call({length:
/// 4294967295}, f)` is one expression), and every array generic turns it
/// straight into `Vec::with_capacity(n)` plus an n-iteration loop. At 2^32 that
/// is a ~137 GB allocation and four billion property lookups in **native**
/// code, where the interpreter's tick never runs — so the OS stops responding
/// entirely: no UI, no Ctrl+C, no script budget. A browser tab may hang itself
/// on that; a kernel may not. 16 Mi elements is far past any real DOM
/// collection and still bounded work — and the walk below ticks, so even that
/// much is interruptible.
pub const MAX_ARRAY_ELEMENTS: usize = 1 << 20;

/// The `length` an array/array-like reports, clamped to [`MAX_ARRAY_ELEMENTS`].
/// Callers that materialize elements use this; see the constant for why.
pub fn array_len(v: &JsValue) -> usize {
    array_len_raw(v).min(MAX_ARRAY_ELEMENTS)
}

/// The `length` exactly as reported, unclamped — for reads that do not
/// materialize (`arr.length`).
pub fn array_len_raw(v: &JsValue) -> usize {
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
    // `with_capacity` on a clamped length, and a tick in the walk: this is the
    // one place a single JS expression turns into an unbounded native loop.
    let mut out = Vec::with_capacity(n.min(4096));
    if let JsValue::Object(o) = v {
        let b = o.borrow();
        let base = b.as_js_object().get_object_base();
        for i in 0..n {
            if crate::runner::host::host_tick() {
                break; // host asked to stop (Ctrl+C / script budget)
            }
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

    // Ordinary object literals inherit from `Object.prototype` (ES OrdinaryObjectCreate).
    // Without this, `Object.prototype.isPrototypeOf({})` and similar chain walks fail.
    if let Some(proto) = object_prototype_object(ctx) {
        obj.get_object_base_mut().prototype = Some(proto);
    }

    // Wrap in JsObjectType
    let obj_ref: JsObjectType = Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(obj))));
    Ok(JsValue::Object(obj_ref))
}

/// Resolve the live `Object.prototype` carrier (materialising it if needed).
fn object_prototype_object(ctx: &mut EvalContext) -> Option<JsObjectType> {
    let sg = ctx.super_global.clone();
    let object_ctor = sg.borrow().resolve_binding("Object", ctx).ok()?;
    drop(sg);
    match get_property_with_ctx(&object_ctor, "prototype", ctx) {
        Ok(JsValue::Object(p)) => Some(p),
        _ => None,
    }
}

/// True when `this` looks like a `new X(...)` instance: its `[[Prototype]]` is
/// the live `X.prototype` carrier. Used by `Number`/`String`/`Boolean`
/// constructors so a bare `String(x)` call (whose `this` is `globalThis` or
/// `undefined`) returns a **primitive**, while `new String(x)` boxes.
///
/// Without this check, bare `String("hi")` stamped onto `globalThis` and
/// returned the global object — breaking every `String(msg)` in the harness
/// and `typeof String(1) === "string"` tests.
pub fn is_new_this_for(type_name: &str, this: &JsValue, ctx: &mut EvalContext) -> bool {
    let JsValue::Object(o) = this else {
        return false;
    };
    let Some(proto) = o.borrow().as_js_object().get_prototype_of() else {
        return false;
    };
    let sg = ctx.super_global.clone();
    let ctor = match sg.borrow().resolve_binding(type_name, ctx) {
        Ok(c) => c,
        Err(_) => return false,
    };
    drop(sg);
    match get_property_with_ctx(&ctor, "prototype", ctx) {
        Ok(JsValue::Object(expected)) => Rc::ptr_eq(&proto, &expected),
        _ => false,
    }
}

/// Get the property key from an object literal property.
fn get_object_property_key(
    key_expr: &ExpressionType,
    computed: bool,
    ctx: &mut EvalContext,
) -> Result<String, JErrorType> {
    if computed {
        // Computed property: [expr]. Use `value_to_property_key` (not `to_string`)
        // so a symbol key keeps its unique identity — `to_string` collapses every
        // symbol to "[Symbol]", colliding e.g. `[Symbol.toPrimitive]` with any
        // other symbol and breaking `ToPrimitive` lookup.
        let key_value = evaluate_expression(key_expr, ctx)?;
        Ok(value_to_property_key(&key_value))
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
                LiteralType::BigIntLiteral(s) => Ok(s.clone()),
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

        // `BigInt` and `Symbol` are callable but NOT constructors — `new` throws.
        if ctor_name == "BigInt" || ctor_name == "Symbol" {
            return Err(JErrorType::TypeError(format!(
                "{} is not a constructor",
                ctor_name
            )));
        }

        // Evaluate arguments first
        let args = evaluate_arguments(arguments, ctx)?;

        // Resolve the constructor so we can wire `[[Prototype]]` on the instance
        // before calling it (built-in sentinels aren't `is_callable` but still
        // construct via the registry).
        if let Ok(constructor) = evaluate_expression(callee, ctx) {
            if let JsValue::Object(ctor_obj) = &constructor {
                let new_obj = create_new_object_for_constructor(ctor_obj, ctx)?;
                let this_val = JsValue::Object(new_obj.clone());
                let sg = ctx.super_global.clone();
                let sg_result =
                    sg.borrow()
                        .call_constructor(ctor_name, ctx, this_val, args.clone());
                if let Some(result) = sg_result {
                    // Prefer an object return; else the pre-created instance.
                    return match result {
                        Ok(JsValue::Object(o)) => Ok(JsValue::Object(o)),
                        Ok(_) => Ok(JsValue::Object(new_obj)),
                        Err(e) => Err(e),
                    };
                }
            }
        }

        // Fall through to normal evaluation if super-global doesn't handle it
    }

    // Normal constructor path: evaluate the callee to get the constructor function
    let constructor = evaluate_expression(callee, ctx)?;

    // Verify it's callable
    let ctor_obj = match &constructor {
        JsValue::Object(obj) => {
            if !obj.borrow().is_callable() {
                // Built-in constructor sentinel (Error, Array, …): still construct.
                let name = builtin_type_name(&constructor);
                let new_obj = create_new_object_for_constructor(obj, ctx)?;
                let this_val = JsValue::Object(new_obj.clone());
                let args = evaluate_arguments(arguments, ctx)?;
                let sg = ctx.super_global.clone();
                let sg_result =
                    sg.borrow()
                        .call_constructor(&name, ctx, this_val, args);
                if let Some(result) = sg_result {
                    return match result {
                        Ok(JsValue::Object(o)) => Ok(JsValue::Object(o)),
                        Ok(_) => Ok(JsValue::Object(new_obj)),
                        Err(e) => Err(e),
                    };
                }
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
    let new_obj = create_new_object_for_constructor(&ctor_obj, ctx)?;

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
    ctx: &mut EvalContext,
) -> Result<JsObjectType, JErrorType> {
    use crate::runner::ds::object::ObjectType;

    // Create a new empty object
    let mut new_obj = SimpleObject::new();

    // Resolve `ctor.prototype` via the normal property path (synthetic carriers
    // for built-in constructors; real own props for user functions/classes).
    let ctor_val = JsValue::Object(constructor.clone());
    if let Ok(JsValue::Object(proto_obj)) = get_property_with_ctx(&ctor_val, "prototype", ctx) {
        new_obj.get_object_base_mut().prototype = Some(proto_obj);
    }

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
    // A native-method value (`[].slice`, `Object.prototype.toString.call`, a
    // builtin method pulled off as a value): dispatch to the registry with the
    // caller-supplied `this` (from `.call`/`.apply` or the member receiver),
    // falling back to the method's default receiver when called bare.
    if let Some((objname, method)) = native_method_parts(callee) {
        let this = if matches!(this_value, JsValue::Undefined) {
            get_own_prop_value(callee, "__method_this__").unwrap_or(JsValue::Undefined)
        } else {
            this_value
        };
        let sg = ctx.super_global.clone();
        return sg
            .borrow()
            .call_method(&objname, &method, ctx, this, args)
            .unwrap_or_else(|| {
                Err(JErrorType::TypeError(format!("{method} is not a function")))
            });
    }
    match callee {
        JsValue::Object(obj) => {
            let obj_ref = obj.borrow();
            // Builtin / host function sentinels (`setTimeout`, `String`, …) —
            // try the registry first even when marked callable for `typeof`.
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
            if let Some(name) = builtin {
                drop(obj_ref);
                let sg = ctx.super_global.clone();
                let result =
                    sg.borrow()
                        .call_constructor(&name, ctx, this_value.clone(), args.clone());
                drop(sg);
                if let Some(r) = result {
                    return r;
                }
                // Fall through: user may have shadowed a builtin name with a
                // real function object that also carries the marker.
                let obj_ref = obj.borrow();
                if obj_ref.is_callable() {
                    drop(obj_ref);
                    return call_function_object(obj, this_value, args, ctx);
                }
                return Err(JErrorType::TypeError(format!(
                    "{} is not a function",
                    callee
                )));
            }
            if obj_ref.is_callable() {
                drop(obj_ref);
                call_function_object(obj, this_value, args, ctx)
            } else {
                drop(obj_ref);
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
    // Cooperative yield: pump the host UI / honor Ctrl+C between calls so a
    // heavy script can't freeze the (cooperatively-scheduled) kernel thread.
    if crate::runner::host::host_tick() {
        return Err(crate::runner::host::interrupt_error());
    }
    // A native builtin that was interrupted mid-loop cannot return an error
    // (the regex matcher returns `Option`), so it raises a flag; surface it at
    // the next call boundary as the real interrupt rather than letting the
    // script continue on a truncated result.
    if crate::runner::std_lib::regexp::take_interrupt() {
        return Err(crate::runner::host::interrupt_error());
    }
    // Call-depth guard: the interpreter recurses on the host (kernel) stack, so
    // unbounded or non-tail JS recursion would fault the kernel. Throw a
    // spec-shaped RangeError past the limit and always restore the counter.
    if ctx.call_depth >= crate::runner::plugin::types::MAX_CALL_DEPTH {
        return Err(JErrorType::RangeError(
            "Maximum call stack size exceeded".to_string(),
        ));
    }
    ctx.call_depth += 1;
    let result = call_function_object_inner(func, this_value, args, ctx);
    ctx.call_depth -= 1;
    result
}

fn call_function_object_inner(
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
            bind_parameters(&params[..], &args, ctx)?;

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

            let field_ctor_marker = PropertyKey::Str("__has_field_init__".to_string());
            if obj.get_object_base().properties.contains_key(&default_ctor_marker)
                && !obj.get_object_base().properties.contains_key(&field_ctor_marker)
            {
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
pub fn get_property_with_ctx(value: &JsValue, prop_name: &str, ctx: &mut EvalContext) -> ValueResult {
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

            // Well-known symbols are exposed as static members of the `Symbol`
            // builtin (`Symbol.iterator`, `Symbol.toPrimitive`, …) so that
            // `obj[Symbol.iterator]` resolves to the canonical symbol value.
            {
                let is_symbol_builtin = {
                    let obj_ref = obj.borrow();
                    match obj_ref
                        .as_js_object()
                        .get_object_base()
                        .properties
                        .get(&PropertyKey::Str("__builtin_name__".to_string()))
                    {
                        Some(PropertyDescriptor::Data(d)) => {
                            matches!(&d.value, JsValue::String(s) if s == "Symbol")
                        }
                        _ => false,
                    }
                };
                if is_symbol_builtin {
                    if let Some(sym) = well_known_symbol(prop_name) {
                        return Ok(sym);
                    }
                }
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
                // `Constructor.prototype` (Object/Array/String/Error/…): a
                // synthetic carrier whose method reads dispatch to the
                // constructor's registry methods — so
                // `Object.prototype.toString.call(x)` and
                // `Array.prototype.slice.call(args)` resolve to real callables.
                //
                // The carrier is installed as an own property on the constructor
                // the first time it is read, so `Error.prototype === Error.prototype`
                // and `class X extends Error` / `instanceof Error` share one identity.
                if prop_name == "prototype" {
                    let builtin_name = {
                        let obj_ref = obj.borrow();
                        match obj_ref
                            .as_js_object()
                            .get_object_base()
                            .properties
                            .get(&PropertyKey::Str("__builtin_name__".to_string()))
                        {
                            Some(PropertyDescriptor::Data(d)) => match &d.value {
                                JsValue::String(name) => Some(name.clone()),
                                _ => None,
                            },
                            _ => None,
                        }
                    };
                    if let Some(name) = builtin_name {
                        let carrier = make_object(Vec::new());
                        set_own_prop(
                            &carrier,
                            "__proto_of__",
                            JsValue::String(name.clone()),
                            false,
                        );
                        // Cache on the constructor *before* wiring the parent
                        // chain so a re-entrant read of `X.prototype` is stable.
                        set_own_prop(
                            lookup_target,
                            "prototype",
                            carrier.clone(),
                            false,
                        );
                        // Wire `X.prototype.[[Prototype]] = Parent.prototype`
                        // from the registry (TypeError→Error→Object→null).
                        // Without this, `instanceof Error` / class-extends
                        // Error and `Object.getPrototypeOf(TypeError.prototype)`
                        // all fail once the carrier is materialised.
                        let parent_name = {
                            let sg = ctx.super_global.borrow();
                            sg.builtin_parent(&name)
                        };
                        if let Some(parent_name) = parent_name {
                            let sg = ctx.super_global.clone();
                            let parent_ctor = sg.borrow().resolve_binding(&parent_name, ctx).ok();
                            // Drop the super-global borrow before further ctx use.
                            drop(sg);
                            if let Some(parent_ctor) = parent_ctor {
                                if let Ok(JsValue::Object(parent_proto)) =
                                    get_property_with_ctx(&parent_ctor, "prototype", ctx)
                                {
                                    if let JsValue::Object(c) = &carrier {
                                        c.borrow_mut()
                                            .as_js_object_mut()
                                            .get_object_base_mut()
                                            .prototype = Some(parent_proto);
                                    }
                                }
                            }
                        }
                        return Ok(carrier);
                    }
                }
                // Check prototype chain - receiver stays the same
                let proto = obj.borrow().as_js_object().get_prototype_of();
                if let Some(proto) = proto {
                    get_property_with_receiver(receiver, &JsValue::Object(proto), prop_name, ctx)
                } else if let Some(m) = materialize_builtin_method(receiver, prop_name, ctx) {
                    // A builtin prototype method pulled off as a value
                    // (`obj.hasOwnProperty`, `[].slice`, `Object.prototype.toString`).
                    Ok(m)
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
            } else if let Some(m) = materialize_builtin_method(receiver, prop_name, ctx) {
                // String prototype method as a value (`"s".slice`, `.charAt`).
                Ok(m)
            } else {
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
            // Number/Boolean/BigInt primitive: expose a prototype method pulled
            // off as a value (`(5).toFixed`, `n.toString`) as a callable.
            if let Some(m) = materialize_builtin_method(receiver, prop_name, ctx) {
                Ok(m)
            } else {
                Ok(JsValue::Undefined)
            }
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
    bind_parameters(&params[..], &args[..], ctx)?;

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
    match value {
        // Symbols must key distinctly (by their canonical `Symbol(desc)` form)
        // rather than collapsing to one `[Symbol]` string, so that e.g.
        // `obj[Symbol.iterator]` and `obj[Symbol.toPrimitive]` are separate
        // properties. Matches `value_to_property_key` (used by call dispatch).
        JsValue::Symbol(_) => value_to_property_key(value),
        _ => to_string(value),
    }
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
            // Array destructuring requires an iterable; `null`/`undefined` throw.
            if matches!(value, JsValue::Null | JsValue::Undefined) {
                return Err(JErrorType::TypeError("value is not iterable".to_string()));
            }
            // Fast path: a genuine array using the default iteration protocol.
            if is_array(&value) && !has_custom_iterator(&value, ctx) {
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
                return Ok(());
            }
            // General path: drive the iterator protocol.
            let iter = get_iterator(&value, ctx)?;
            drive_array_pattern(&iter, elements, ctx, |pat, v, ctx| {
                assign_pattern(pat, v, ctx)
            })
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
                LiteralType::BigIntLiteral(s) => Ok(s.clone()),
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

    // Produce a genuine JS Array (with the `__array__` marker) so
    // `Array.isArray(rest)` holds and nested rest patterns keep iterating.
    let mut rest: Vec<JsValue> = Vec::with_capacity(length.saturating_sub(start_index));
    for i in start_index..length {
        rest.push(get_property_with_ctx(arr, &i.to_string(), ctx)?);
    }
    Ok(make_array(rest))
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
            let value = to_primitive(&value, "number", ctx)?;
            negate_number(&value)
        }
        UnaryOperator::Plus => {
            let value = evaluate_expression(argument, ctx)?;
            let value = to_primitive(&value, "number", ctx)?;
            // Unary `+` explicitly rejects BigInt (would require ToNumber).
            if matches!(value, JsValue::BigInt(_)) {
                return Err(JErrorType::TypeError(
                    "Cannot convert a BigInt to a number".to_string(),
                ));
            }
            to_number(&value)
        }
        UnaryOperator::BitwiseNot => {
            let value = evaluate_expression(argument, ctx)?;
            let value = to_primitive(&value, "number", ctx)?;
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

    // Coerce object operands to primitives via ToPrimitive for the operators
    // that require it (arithmetic/relational/bitwise). `+` and `<`-family use no
    // specific hint ("default"/"number"); the rest use "number". Equality
    // (`===`/`==`), `instanceof` and `in` are left untouched — they either never
    // coerce or handle it themselves. Non-objects and objects without user
    // coercion pass through unchanged (see `to_primitive`).
    let hint = match operator {
        BinaryOperator::Add => Some("default"),
        BinaryOperator::Subtract
        | BinaryOperator::Multiply
        | BinaryOperator::Divide
        | BinaryOperator::Modulo
        | BinaryOperator::Exponent
        | BinaryOperator::LessThan
        | BinaryOperator::GreaterThan
        | BinaryOperator::LessThanEqual
        | BinaryOperator::GreaterThanEqual
        | BinaryOperator::BitwiseAnd
        | BinaryOperator::BitwiseOr
        | BinaryOperator::BitwiseXor
        | BinaryOperator::BitwiseLeftShift
        | BinaryOperator::BitwiseRightShift
        | BinaryOperator::BitwiseUnsignedRightShift => Some("number"),
        _ => None,
    };
    let (left_val, right_val) = match hint {
        Some(h) => (
            to_primitive(&left_val, h, ctx)?,
            to_primitive(&right_val, h, ctx)?,
        ),
        None => (left_val, right_val),
    };

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

            // Resolve `constructor.prototype` via the normal property path so
            // synthetic built-in carriers (Error/TypeError/…) are materialised
            // and cached — own-property-only lookup left `instanceof Error`
            // false until something else had forced the carrier into place.
            let ctor_val = JsValue::Object(right_obj);
            let target_proto = match get_property_with_ctx(&ctor_val, "prototype", ctx)? {
                JsValue::Object(p) => p,
                _ => return Ok(JsValue::Boolean(false)),
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
            let prop_name = to_property_key(&left_val);
            let prop_key = PropertyKey::Str(prop_name.clone());
            match &right_val {
                JsValue::Object(obj) => {
                    let has_own = obj.borrow().as_js_object().has_property(&prop_key);
                    if has_own {
                        return Ok(JsValue::Boolean(true));
                    }
                    // Built-in constructor statics (`Object.create`, `Array.from`)
                    // and prototype methods live in the registry, not as real
                    // own properties on the sentinel — still visible to `in`.
                    let type_name = builtin_type_name(&right_val);
                    if ctx.super_global.borrow().has_method(&type_name, &prop_name) {
                        // Statics only on the constructor; prototype methods on
                        // any instance/carrier of that type.
                        if is_static_method(&type_name, &prop_name) {
                            return Ok(JsValue::Boolean(is_constructor_sentinel(&right_val)));
                        }
                        return Ok(JsValue::Boolean(true));
                    }
                    Ok(JsValue::Boolean(false))
                }
                _ => Err(JErrorType::TypeError(
                    "Cannot use 'in' operator with non-object".to_string(),
                )),
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
    if let Some(r) = bigint_binary_op(left, right, |base, exp| {
        use num_traits::Signed;
        if exp.is_negative() {
            return Err(JErrorType::RangeError(
                "Exponent must be non-negative".to_string(),
            ));
        }
        // BigInt exponent fits a u32 for any realistic test; guard the range.
        match num_traits::ToPrimitive::to_u32(exp) {
            Some(e) => Ok(JsValue::BigInt(base.pow(e))),
            None => Err(JErrorType::RangeError("Exponent too large".to_string())),
        }
    }) {
        return r;
    }
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

    // BigInt `++`/`--` stays a BigInt (no ToNumber, ±1n).
    if let JsValue::BigInt(b) = &current_value {
        let one = num_bigint::BigInt::from(1);
        let new_big = match operator {
            UpdateOperator::PlusPlus => b + &one,
            UpdateOperator::MinusMinus => b - &one,
        };
        let old_value = JsValue::BigInt(b.clone());
        let new_value = JsValue::BigInt(new_big);
        match &member_target {
            Some((obj, prop)) => set_property_with_ctx(obj, prop, new_value.clone(), ctx)?,
            None => {
                let name = get_expression_name(argument)?;
                ctx.set_binding(&name, new_value.clone())?;
            }
        }
        return Ok(if prefix { new_value } else { old_value });
    }

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
        // A BigInt is falsy only when it is `0n`.
        JsValue::BigInt(b) => !num_traits::Zero::is_zero(b),
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
        JsValue::BigInt(_) => "bigint".to_string(),
        JsValue::Symbol(_) => "symbol".to_string(),
        // A callable object (user function, bound function, or a native-method
        // value like `[].slice`) is `"function"`; everything else is `"object"`.
        JsValue::Object(_) => {
            if value_is_callable(value) {
                "function".to_string()
            } else {
                "object".to_string()
            }
        }
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
        JsValue::BigInt(_) => {
            return Err(JErrorType::TypeError("Cannot convert a BigInt to a number".to_string()));
        }
        JsValue::Object(_) => JsValue::Number(JsNumberType::NaN),
    })
}

/// Negate a number value.
fn negate_number(value: &JsValue) -> ValueResult {
    if let JsValue::BigInt(b) = value {
        return Ok(JsValue::BigInt(-b));
    }
    let num_value = to_number(value)?;
    Ok(match num_value {
        // `i64` has no negative zero — `-0` must be a float so `1/(-0)` is -∞
        // and `-0 + -0` stays -0 (IEEE).
        JsValue::Number(JsNumberType::Integer(0)) => {
            JsValue::Number(JsNumberType::Float(-0.0))
        }
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
    if let JsValue::BigInt(b) = value {
        // ~b == -(b + 1) for arbitrary-precision two's-complement semantics.
        return Ok(JsValue::BigInt(-(b + num_bigint::BigInt::from(1))));
    }
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

/// Error thrown when a BigInt is mixed with a Number in an arithmetic/bitwise op.
fn bigint_number_mix_err() -> JErrorType {
    JErrorType::TypeError("Cannot mix BigInt and other types, use explicit conversions".to_string())
}

/// If either operand is a BigInt, apply `op` when BOTH are BigInt, else throw
/// the mixed-type TypeError. Returns `None` when neither operand is a BigInt so
/// the caller falls through to its numeric path.
fn bigint_binary_op<F>(left: &JsValue, right: &JsValue, op: F) -> Option<ValueResult>
where
    F: Fn(&num_bigint::BigInt, &num_bigint::BigInt) -> ValueResult,
{
    match (left, right) {
        (JsValue::BigInt(a), JsValue::BigInt(b)) => Some(op(a, b)),
        (JsValue::BigInt(_), _) | (_, JsValue::BigInt(_)) => Some(Err(bigint_number_mix_err())),
        _ => None,
    }
}

fn add_values(left: &JsValue, right: &JsValue) -> ValueResult {
    // A Symbol operand always throws: after ToPrimitive it stays a Symbol, then
    // either ToString (string concat) or ToNumber (numeric add) rejects it.
    if matches!(left, JsValue::Symbol(_)) || matches!(right, JsValue::Symbol(_)) {
        return Err(JErrorType::TypeError(
            "Cannot convert a Symbol value".to_string(),
        ));
    }
    // String concatenation takes precedence (BigInt + String is a string).
    if matches!(left, JsValue::String(_)) || matches!(right, JsValue::String(_)) {
        let left_str = to_string(left);
        let right_str = to_string(right);
        return Ok(JsValue::String(format!("{}{}", left_str, right_str)));
    }
    if let Some(r) = bigint_binary_op(left, right, |a, b| Ok(JsValue::BigInt(a + b))) {
        return r;
    }

    let left_num = to_number(left)?;
    let right_num = to_number(right)?;
    apply_numeric_op(&left_num, &right_num, |a, b| a + b, |a, b| a + b)
}

fn subtract_values(left: &JsValue, right: &JsValue) -> ValueResult {
    if let Some(r) = bigint_binary_op(left, right, |a, b| Ok(JsValue::BigInt(a - b))) {
        return r;
    }
    let left_num = to_number(left)?;
    let right_num = to_number(right)?;
    apply_numeric_op(&left_num, &right_num, |a, b| a - b, |a, b| a - b)
}

fn multiply_values(left: &JsValue, right: &JsValue) -> ValueResult {
    if let Some(r) = bigint_binary_op(left, right, |a, b| Ok(JsValue::BigInt(a * b))) {
        return r;
    }
    let left_num = to_number(left)?;
    let right_num = to_number(right)?;
    apply_numeric_op(&left_num, &right_num, |a, b| a * b, |a, b| a * b)
}

fn divide_values(left: &JsValue, right: &JsValue) -> ValueResult {
    if let Some(r) = bigint_binary_op(left, right, |a, b| {
        if num_traits::Zero::is_zero(b) {
            Err(JErrorType::RangeError("Division by zero".to_string()))
        } else {
            Ok(JsValue::BigInt(a / b))
        }
    }) {
        return r;
    }
    let left_num = to_number(left)?;
    let right_num = to_number(right)?;

    // IEEE division so signed zero is preserved (`1 / -0 === -∞`). The old
    // `f == 0.0` zero-check treated -0 like +0 and always returned +∞ for a
    // positive dividend.
    let left_f = match &left_num {
        JsValue::Number(n) => number_to_f64(n),
        _ => f64::NAN,
    };
    let right_f = match &right_num {
        JsValue::Number(n) => number_to_f64(n),
        _ => f64::NAN,
    };
    let q = left_f / right_f;
    Ok(if q.is_nan() {
        JsValue::Number(JsNumberType::NaN)
    } else if q.is_infinite() {
        if q.is_sign_negative() {
            JsValue::Number(JsNumberType::NegativeInfinity)
        } else {
            JsValue::Number(JsNumberType::PositiveInfinity)
        }
    } else {
        // Prefer Integer when both operands were integers and the quotient is
        // an exact non-negative integer (negative zero cannot arise here).
        if matches!(
            (&left_num, &right_num),
            (
                JsValue::Number(JsNumberType::Integer(_)),
                JsValue::Number(JsNumberType::Integer(_))
            )
        ) && q.fract() == 0.0
            && !q.is_sign_negative()
            && q >= i64::MIN as f64
            && q <= i64::MAX as f64
        {
            JsValue::Number(JsNumberType::Integer(q as i64))
        } else {
            JsValue::Number(JsNumberType::Float(q))
        }
    })
}

fn modulo_values(left: &JsValue, right: &JsValue) -> ValueResult {
    if let Some(r) = bigint_binary_op(left, right, |a, b| {
        if num_traits::Zero::is_zero(b) {
            Err(JErrorType::RangeError("Division by zero".to_string()))
        } else {
            Ok(JsValue::BigInt(a % b))
        }
    }) {
        return r;
    }
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

/// Best-effort numeric value of `v` for a relational comparison, `None` when it
/// is NaN / not comparable. BigInt and String participate cross-type.
fn compare_operand_f64(v: &JsValue) -> Option<f64> {
    match v {
        JsValue::BigInt(b) => num_traits::ToPrimitive::to_f64(b),
        JsValue::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                Some(0.0)
            } else {
                t.parse::<f64>().ok()
            }
        }
        JsValue::Number(n) => {
            let f = number_to_f64(n);
            if f.is_nan() {
                None
            } else {
                Some(f)
            }
        }
        JsValue::Boolean(true) => Some(1.0),
        JsValue::Boolean(false) | JsValue::Null => Some(0.0),
        JsValue::Undefined => None,
        _ => None,
    }
}

/// Order two strings the way JS does: by UTF-16 code unit.
///
/// Rust's own `str` ordering is by code *point*, which disagrees for a
/// supplementary character (U+10000+, whose surrogates are 0xD800..0xDBFF)
/// compared against U+E000..U+FFFF. ASCII — every real comparison a page makes
/// — takes the byte fast path.
fn code_unit_cmp(a: &str, b: &str) -> core::cmp::Ordering {
    if a.is_ascii() && b.is_ascii() {
        return a.as_bytes().cmp(b.as_bytes());
    }
    a.encode_utf16().cmp(b.encode_utf16())
}

fn compare_values<F>(left: &JsValue, right: &JsValue, cmp: F) -> ValueResult
where
    F: Fn(f64, f64) -> bool,
{
    // Abstract Relational Comparison: when *both* operands are Strings they are
    // ordered by UTF-16 code unit, NOT through ToNumber — which would make both
    // sides NaN and answer `false` for every string pair. React's production
    // bundle guards every DOM access with `typeof document < "u"`, so a numeric
    // `"object" < "u"` reports the document as absent on a page that has one.
    if let (JsValue::String(a), JsValue::String(b)) = (left, right) {
        let result = match code_unit_cmp(a, b) {
            core::cmp::Ordering::Less => cmp(-1.0, 0.0),
            core::cmp::Ordering::Greater => cmp(1.0, 0.0),
            core::cmp::Ordering::Equal => cmp(0.0, 0.0),
        };
        return Ok(JsValue::Boolean(result));
    }

    // A BigInt operand compares mathematically against the other side (Number,
    // String, or BigInt) without the ToNumber TypeError.
    if matches!(left, JsValue::BigInt(_)) || matches!(right, JsValue::BigInt(_)) {
        // Exact BigInt-vs-BigInt comparison avoids f64 rounding.
        if let (JsValue::BigInt(a), JsValue::BigInt(b)) = (left, right) {
            let af = num_traits::ToPrimitive::to_f64(a).unwrap_or(f64::NAN);
            let bf = num_traits::ToPrimitive::to_f64(b).unwrap_or(f64::NAN);
            // For exactness when magnitudes exceed f64, fall back to Ord.
            if a == b {
                return Ok(JsValue::Boolean(cmp(0.0, 0.0)));
            }
            let result = match a.cmp(b) {
                core::cmp::Ordering::Less => cmp(-1.0, 0.0),
                core::cmp::Ordering::Greater => cmp(1.0, 0.0),
                core::cmp::Ordering::Equal => cmp(af, bf),
            };
            return Ok(JsValue::Boolean(result));
        }
        let result = match (compare_operand_f64(left), compare_operand_f64(right)) {
            (Some(a), Some(b)) => cmp(a, b),
            _ => false,
        };
        return Ok(JsValue::Boolean(result));
    }

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
        // BigInt === BigInt by mathematical value; BigInt === Number is always
        // false (handled by the fall-through `_ => false`).
        (JsValue::BigInt(a), JsValue::BigInt(b)) => a == b,
        (JsValue::Object(a), JsValue::Object(b)) => alloc::rc::Rc::ptr_eq(a, b),
        // Symbols compare by description identity (well-knowns + Symbol.for
        // registry keys share a description; Symbol()/new_empty get unique ones).
        (JsValue::Symbol(a), JsValue::Symbol(b)) => a == b,
        _ => false,
    }
}

/// Mathematical equality between a BigInt and a Number/String (the `==` operator
/// crossing BigInt and non-BigInt). Compares exact integer values.
fn bigint_loose_eq_number(b: &num_bigint::BigInt, n: f64) -> bool {
    if n.is_nan() || n.is_infinite() || n.fract() != 0.0 {
        return false;
    }
    match <num_bigint::BigInt as num_traits::FromPrimitive>::from_f64(n) {
        Some(bn) => *b == bn,
        None => false,
    }
}

/// Parse a decimal/hex/octal/binary integer string into a BigInt (used by `==`
/// with a String operand). Returns `None` on any non-integer string.
fn bigint_from_str(s: &str) -> Option<num_bigint::BigInt> {
    let t = s.trim();
    if t.is_empty() {
        return Some(num_bigint::BigInt::from(0));
    }
    let (neg, rest) = match t.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    let (radix, digits): (u32, &str) = if let Some(r) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
        (16, r)
    } else if let Some(r) = rest.strip_prefix("0o").or_else(|| rest.strip_prefix("0O")) {
        (8, r)
    } else if let Some(r) = rest.strip_prefix("0b").or_else(|| rest.strip_prefix("0B")) {
        (2, r)
    } else {
        (10, rest)
    };
    let v = num_bigint::BigInt::parse_bytes(digits.as_bytes(), radix)?;
    Some(if neg { -v } else { v })
}

fn loose_equality(left: &JsValue, right: &JsValue) -> bool {
    if core::mem::discriminant(left) == core::mem::discriminant(right) {
        return strict_equality(left, right);
    }

    // BigInt == Number / String compares by mathematical value.
    match (left, right) {
        (JsValue::BigInt(a), JsValue::Number(n)) | (JsValue::Number(n), JsValue::BigInt(a)) => {
            return bigint_loose_eq_number(a, number_to_f64(n));
        }
        (JsValue::BigInt(a), JsValue::String(s)) | (JsValue::String(s), JsValue::BigInt(a)) => {
            return match bigint_from_str(s) {
                Some(b) => *a == b,
                None => false,
            };
        }
        (JsValue::BigInt(a), JsValue::Boolean(b)) | (JsValue::Boolean(b), JsValue::BigInt(a)) => {
            return bigint_loose_eq_number(a, if *b { 1.0 } else { 0.0 });
        }
        _ => {}
    }

    // Boxed primitives (`new String("x")`, `Object(1)`) store the primitive
    // under `__primitive_value__` — unwrap before the rest of the algorithm so
    // `"ABC" == new String("ABC")` holds.
    if let Some(p) = get_own_prop_value(left, "__primitive_value__") {
        return loose_equality(&p, right);
    }
    if let Some(p) = get_own_prop_value(right, "__primitive_value__") {
        return loose_equality(left, &p);
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
    if let Some(r) = bigint_binary_op(left, right, |a, b| Ok(JsValue::BigInt(a & b))) {
        return r;
    }
    let l = to_i32(left)?;
    let r = to_i32(right)?;
    Ok(JsValue::Number(JsNumberType::Integer((l & r) as i64)))
}

fn bitwise_or(left: &JsValue, right: &JsValue) -> ValueResult {
    if let Some(r) = bigint_binary_op(left, right, |a, b| Ok(JsValue::BigInt(a | b))) {
        return r;
    }
    let l = to_i32(left)?;
    let r = to_i32(right)?;
    Ok(JsValue::Number(JsNumberType::Integer((l | r) as i64)))
}

fn bitwise_xor(left: &JsValue, right: &JsValue) -> ValueResult {
    if let Some(r) = bigint_binary_op(left, right, |a, b| Ok(JsValue::BigInt(a ^ b))) {
        return r;
    }
    let l = to_i32(left)?;
    let r = to_i32(right)?;
    Ok(JsValue::Number(JsNumberType::Integer((l ^ r) as i64)))
}

/// BigInt shift by a (possibly negative) BigInt amount: a negative left-shift is
/// a right-shift and vice versa.
fn bigint_shift(a: &num_bigint::BigInt, amount: &num_bigint::BigInt, left: bool) -> ValueResult {
    use num_traits::ToPrimitive;
    let amt_i64 = amount.to_i64().ok_or_else(|| {
        JErrorType::RangeError("BigInt shift amount out of range".to_string())
    })?;
    let effective = if left { amt_i64 } else { -amt_i64 };
    let out = if effective >= 0 {
        a << (effective as usize)
    } else {
        a >> ((-effective) as usize)
    };
    Ok(JsValue::BigInt(out))
}

fn left_shift(left: &JsValue, right: &JsValue) -> ValueResult {
    if let Some(r) = bigint_binary_op(left, right, |a, b| bigint_shift(a, b, true)) {
        return r;
    }
    let l = to_i32(left)?;
    let r = to_u32(right)? & 0x1f;
    Ok(JsValue::Number(JsNumberType::Integer((l << r) as i64)))
}

fn right_shift(left: &JsValue, right: &JsValue) -> ValueResult {
    if let Some(r) = bigint_binary_op(left, right, |a, b| bigint_shift(a, b, false)) {
        return r;
    }
    let l = to_i32(left)?;
    let r = to_u32(right)? & 0x1f;
    Ok(JsValue::Number(JsNumberType::Integer((l >> r) as i64)))
}

fn unsigned_right_shift(left: &JsValue, right: &JsValue) -> ValueResult {
    // `>>>` is not defined for BigInt — always a TypeError if either is BigInt.
    if matches!(left, JsValue::BigInt(_)) || matches!(right, JsValue::BigInt(_)) {
        return Err(JErrorType::TypeError(
            "BigInts have no unsigned right shift, use >> instead".to_string(),
        ));
    }
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
        JsValue::BigInt(b) => b.to_string(),
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
    /// ChittiOS: ES2022 instance-field initializers for a class constructor —
    /// `(resolved-key, init-expr-ptr)` pairs applied to the freshly-created
    /// `this` at the start of the constructor call. A null pointer means the
    /// field has no initializer (→ `undefined`). Empty for every non-constructor
    /// function. The pointers target initializer expressions inside the class's
    /// AST, which outlives the object exactly as `body_ptr` does.
    instance_fields: Vec<(String, *const crate::parser::ast::ExpressionType)>,
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
            instance_fields: Vec::new(),
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
            instance_fields: Vec::new(),
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
        // A synthesized field-only default constructor has a null params_ptr and
        // null body_ptr (no source function) — treat both as empty.
        let params: &[PatternType] = if self.params_ptr.is_null() {
            &[]
        } else {
            unsafe { &*self.params_ptr }
        };

        // ChittiOS: keep a handle to `this` for applying instance-field
        // initializers (the original is moved into `ctx.global_this` below).
        let this_for_fields = this_value.clone();

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
        bind_parameters(&params[..], &args[..], ctx)?;

        // ChittiOS: the `arguments` object — an array-like of the actual call
        // arguments, exposed in every *non-arrow* function (arrows inherit it
        // lexically). Ubiquitous in pre-ES6 code (`arguments.length`,
        // `fn.apply(this, arguments)`, `Array.from(arguments)`). A same-named
        // parameter (`function f(arguments)`) wins — checked against the param
        // list, NOT `has_binding` (which would find an *enclosing* function's
        // `arguments` and wrongly skip a nested function's own). Modeled as a
        // real array — array-like enough for `.length`/indexing/`.apply`/`Array.from`.
        if !self.is_arrow {
            let param_named_arguments = params.iter().any(|p| {
                matches!(p, PatternType::PatternWhichCanBeExpression(
                    ExpressionPatternType::Identifier(id)) if id.name == "arguments")
            });
            if !param_named_arguments {
                let arguments_obj = make_array(args.clone());
                ctx.create_binding("arguments", false)?;
                ctx.initialize_binding("arguments", arguments_obj)?;
            }
        }

        // ChittiOS: ES2022 instance-field initializers run at the top of the
        // constructor, against the freshly-created `this`, before the body.
        // (Non-constructor functions and arrows carry none.) NB: for a *derived*
        // class with an explicit constructor these run before the body's
        // `super(...)` rather than strictly after it — `this` already exists in
        // this engine, so it works, but a parent constructor could overwrite a
        // same-named field. Rare in practice; documented.
        if !self.instance_fields.is_empty() {
            for (key, init_ptr) in &self.instance_fields {
                let v = if init_ptr.is_null() {
                    JsValue::Undefined
                } else {
                    // SAFETY: the pointer targets an initializer expression in
                    // the class AST, alive for the object's lifetime (as body_ptr).
                    match evaluate_expression(unsafe { &**init_ptr }, ctx) {
                        Ok(v) => v,
                        Err(e) => {
                            ctx.lex_env = saved_lex_env;
                            ctx.var_env = saved_var_env;
                            ctx.global_this = saved_this;
                            return Err(e);
                        }
                    }
                };
                // Private fields (`#x`) are non-enumerable; public fields are
                // enumerable own data properties.
                set_own_prop(&this_for_fields, key, v, !key.starts_with('#'));
            }
        }

        // A field-only default constructor has no body — done after fields.
        // (A concise arrow has a null body_ptr but a non-null expr_body_ptr.)
        if self.body_ptr.is_null() && self.expr_body_ptr.is_null() {
            ctx.lex_env = saved_lex_env;
            ctx.var_env = saved_var_env;
            ctx.global_this = saved_this;
            return Ok(if self.is_async {
                crate::runner::std_lib::promise::resolve_value(JsValue::Undefined)
            } else {
                JsValue::Undefined
            });
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
        JsValue::Object(o) => {
            if o.borrow().is_callable() {
                return true;
            }
            // A native-method value (`[].slice`, `Object.prototype.toString`) is
            // an ordinary object tagged callable — dispatched by `call_value`.
            if get_own_prop_value(v, "__native_method__").is_some() {
                return true;
            }
            // Host/builtin function sentinels (`setTimeout`, `fetch`, `Symbol`)
            // carry `__builtin_name__` *and* `__host_fn__` so `typeof` is
            // `"function"` without treating `document`/`window` as callables.
            get_own_prop_value(v, "__host_fn__").is_some()
        }
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

/// AbstractOperation ToPrimitive(input, hint) for the object case.
///
/// A non-object is returned unchanged. For an object: if `Symbol.toPrimitive`
/// is callable it is invoked with the hint string (`"number"`/`"string"`/
/// `"default"`) and its result is required to be primitive (else a TypeError);
/// otherwise `valueOf`/`toString` are tried in hint order and the first
/// primitive result wins. A boxed primitive wrapper (`Object(2n)`, `Object(42)`
/// — carrying `__primitive_value__`) falls back to its stored primitive. An
/// object with none of these (a plain `{}` with no user coercion and no real
/// prototype) is returned unchanged, so the interpreter's legacy stringify/NaN
/// coercion still applies and the passing corpus is unaffected.
pub fn to_primitive(value: &JsValue, hint: &str, ctx: &mut EvalContext) -> ValueResult {
    if !matches!(value, JsValue::Object(_)) {
        return Ok(value.clone());
    }
    // 1. GetMethod(input, @@toPrimitive): undefined/null → OrdinaryToPrimitive;
    // anything non-callable → TypeError (spec). Keyed by the symbol's
    // stringified form, like every other symbol-valued property here.
    let sym_key = value_to_property_key(&JsValue::Symbol(
        crate::runner::ds::symbol::SYMBOL_TO_PRIMITIVE.clone(),
    ));
    let exotic = get_property_with_ctx(value, &sym_key, ctx)?;
    match &exotic {
        JsValue::Undefined | JsValue::Null => {}
        _ if value_is_callable(&exotic) => {
            let h = if hint.is_empty() { "default" } else { hint };
            let res = call_value(
                &exotic,
                value.clone(),
                alloc::vec![JsValue::String(h.to_string())],
                ctx,
            )?;
            if matches!(res, JsValue::Object(_)) {
                return Err(JErrorType::TypeError(
                    "Cannot convert object to primitive value".to_string(),
                ));
            }
            return Ok(res);
        }
        _ => {
            return Err(JErrorType::TypeError(
                "Symbol.toPrimitive is not a function".to_string(),
            ));
        }
    }
    // 2. Boxed primitive wrapper (`Object(prim)`, `Object(2n)`): unwrap to the
    // stored primitive. This must precede the valueOf/toString loop — the
    // wrapper's spec `valueOf` returns the primitive, but this engine's generic
    // `Object.prototype.valueOf` returns the wrapper itself (and `toString`
    // returns "[object Object]"), which would otherwise mask the primitive now
    // that builtin prototype methods are first-class values.
    if let Some(p) = get_own_prop_value(value, "__primitive_value__") {
        return Ok(p);
    }
    // 3. OrdinaryToPrimitive: valueOf/toString in hint order.
    let order: [&str; 2] = if hint == "string" {
        ["toString", "valueOf"]
    } else {
        ["valueOf", "toString"]
    };
    // Date's @@toPrimitive treats hint "default" like "string" (prefer
    // toString over valueOf) so `date + date` string-concats. Without a real
    // Symbol.toPrimitive install, flip the order for Date receivers.
    let order: [&str; 2] = if hint == "default" && builtin_type_name(value) == "Date" {
        ["toString", "valueOf"]
    } else {
        order
    };

    let mut tried_coercer = false;
    for m in order {
        let method = get_property_with_ctx(value, m, ctx)?;
        if !value_is_callable(&method) {
            continue;
        }
        // Skip Object.prototype.valueOf only — it returns the receiver (still
        // an Object), so calling it is a no-op that would flip `tried_coercer`
        // and turn a missing toString into a TypeError. Do **not** skip
        // Object.prototype.toString: `({} + {})` must be string concat of
        // "[object Object]" twice (skipping it left objects for ToNumber → NaN).
        if let Some((objname, mname)) = native_method_parts(&method) {
            if objname == "Object" && mname == "valueOf" {
                continue;
            }
        }
        tried_coercer = true;
        let res = call_value(&method, value.clone(), alloc::vec![], ctx)?;
        if !matches!(res, JsValue::Object(_)) {
            return Ok(res);
        }
    }
    // Spec OrdinaryToPrimitive: if every callable returned an Object, TypeError.
    // Only when we actually tried a coercer (user or type-specific native) —
    // plain `{}` (Object natives skipped) still falls through for legacy paths.
    if tried_coercer {
        return Err(JErrorType::TypeError(
            "Cannot convert object to primitive value".to_string(),
        ));
    }
    // Functions have no user valueOf/toString and Object natives are skipped —
    // still coerce to a function source string so `fn + 1` / `"" + fn` work
    // (Function.prototype.toString semantics, source not preserved).
    if value_is_callable(value) {
        let name = match get_own_prop_value(value, "name") {
            Some(JsValue::String(s)) if !s.is_empty() => s,
            _ => String::new(),
        };
        return Ok(JsValue::String(alloc::format!(
            "function {}() {{ [native code] }}",
            name
        )));
    }
    // 4. No coercer available: leave the object for the legacy path.
    Ok(value.clone())
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
            // A synthetic `X.prototype` carrier dispatches methods to X's
            // registry (so `Object.prototype.toString`, `Array.prototype.slice`
            // resolve to real callables). Checked before `__builtin_name__` so
            // the prototype object isn't mistaken for its constructor sentinel.
            if let Some(PropertyDescriptor::Data(d)) =
                base.properties.get(&PropertyKey::Str("__proto_of__".to_string()))
            {
                if let JsValue::String(name) = &d.value {
                    return name.clone();
                }
            }
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
                return "Array".to_string();
            }
            // Boxed primitives (`new Number(1)`, `new String("x")`): the
            // instance itself has no tag, but its [[Prototype]] is the
            // synthetic `Number.prototype` / `String.prototype` carrier.
            // Walk one step (and a few more for Error subclasses) so
            // `(new Number(1)).toString` resolves to Number's method, not
            // Object.prototype.toString → "[object Object]".
            let mut cur = base.prototype.clone();
            let mut steps = 0usize;
            while let Some(p) = cur {
                if steps > 8 {
                    break;
                }
                steps += 1;
                let pb = p.borrow();
                let pbase = pb.as_js_object().get_object_base();
                if let Some(PropertyDescriptor::Data(d)) = pbase
                    .properties
                    .get(&PropertyKey::Str("__proto_of__".to_string()))
                {
                    if let JsValue::String(name) = &d.value {
                        // Object.prototype is the default chain root — keep
                        // looking only if we haven't found a more specific type.
                        if name != "Object" {
                            return name.clone();
                        }
                    }
                }
                cur = pbase.prototype.clone();
            }
            "Object".to_string()
        }
        _ => "Object".to_string(),
    }
}

/// Build a first-class callable value for a builtin prototype method
/// (`[].slice`, `Object.prototype.toString`, `"s".charAt`). It carries the
/// registry coordinates `__native_method__ = "ObjName:method"` and a
/// `__method_this__` default receiver (used when the value is later called
/// without an explicit `this`, e.g. `var f = [].slice; f()`); `.call`/`.apply`
/// override that. `call_value` dispatches it to the registry. This is what makes
/// the `Array.prototype.slice.call(arguments)` / `fn.bind` idioms work.
pub fn make_native_method(objname: &str, method: &str, default_this: JsValue) -> JsValue {
    let obj = make_object(Vec::new());
    set_own_prop(
        &obj,
        "__native_method__",
        JsValue::String(format!("{objname}:{method}")),
        false,
    );
    set_own_prop(&obj, "__method_this__", default_this, false);
    set_own_prop(&obj, "name", JsValue::String(method.to_string()), false);
    // Mark callable so `value_is_callable` / `typeof` report a function.
    set_own_prop(&obj, "__callable__", JsValue::Boolean(true), false);
    obj
}

/// True when `v` is a built-in **constructor sentinel** (`Object`, `Array`, …)
/// as materialized by `CorePluginResolver` — not an instance and not a
/// synthetic `X.prototype` carrier (`__proto_of__`).
fn is_constructor_sentinel(v: &JsValue) -> bool {
    let JsValue::Object(o) = v else {
        return false;
    };
    let b = o.borrow();
    let base = b.as_js_object().get_object_base();
    if base
        .properties
        .contains_key(&PropertyKey::Str("__proto_of__".to_string()))
    {
        return false;
    }
    matches!(
        base.properties
            .get(&PropertyKey::Str("__builtin_name__".to_string())),
        Some(PropertyDescriptor::Data(d)) if matches!(&d.value, JsValue::String(_))
    )
}

/// If `receiver`'s builtin type provides `prop_name` as a registry method,
/// return a first-class callable bound (by default) to `receiver`. Used as the
/// fallback when a property isn't found as a real own/prototype property, so a
/// builtin prototype method read (`[].map`, `obj.hasOwnProperty`) yields a
/// function value instead of `undefined`.
///
/// Constructor **statics** (`Object.create`, `Object.keys`, `Array.from`) are
/// only materialised when the receiver *is* that constructor sentinel — so
/// `Object.create` is a real function value (for extract/pass/feature-detect)
/// while `({}).create` stays undefined.
pub fn materialize_builtin_method(
    receiver: &JsValue,
    prop_name: &str,
    ctx: &EvalContext,
) -> Option<JsValue> {
    let mut type_name = builtin_type_name(receiver);
    // Callables that aren't a more specific builtin (Array/Date/…) still need
    // Function.prototype methods (`toString`, `call`, `apply`, `bind`).
    if type_name == "Object" && value_is_callable(receiver) {
        type_name = "Function".to_string();
    }
    if is_static_method(&type_name, prop_name) {
        if !is_constructor_sentinel(receiver) {
            // Statics never live on instances / prototype carriers.
            return None;
        }
        // Constructor static — materialise if the registry has it.
        if ctx.super_global.borrow().has_method(&type_name, prop_name) {
            return Some(make_native_method(
                &type_name,
                prop_name,
                receiver.clone(),
            ));
        }
        return None;
    }
    if ctx.super_global.borrow().has_method(&type_name, prop_name) {
        Some(make_native_method(&type_name, prop_name, receiver.clone()))
    } else {
        None
    }
}

/// True if `method` on builtin `type_name` is a constructor *static* (belongs on
/// the constructor, not on instances/prototype). Used to keep statics off the
/// prototype-method materialization path.
fn is_static_method(type_name: &str, method: &str) -> bool {
    match type_name {
        "Object" => matches!(
            method,
            "keys"
                | "values"
                | "entries"
                | "assign"
                | "defineProperty"
                | "defineProperties"
                | "getOwnPropertyDescriptor"
                | "getOwnPropertyDescriptors"
                | "getOwnPropertyNames"
                | "getOwnPropertySymbols"
                | "hasOwn"
                | "create"
                | "freeze"
                | "seal"
                | "preventExtensions"
                | "isFrozen"
                | "isSealed"
                | "isExtensible"
                | "getPrototypeOf"
                | "setPrototypeOf"
                | "is"
                | "fromEntries"
        ),
        "Array" => matches!(method, "from" | "isArray" | "of"),
        "String" => matches!(method, "fromCharCode" | "fromCodePoint" | "raw"),
        "Number" => matches!(
            method,
            "isInteger" | "isFinite" | "isNaN" | "isSafeInteger" | "parseFloat" | "parseInt"
        ),
        "Symbol" => matches!(method, "for" | "keyFor"),
        _ => false,
    }
}

/// The `"ObjName:method"` registry coordinates of a native-method value, if `v`
/// is one (see [`make_native_method`]).
fn native_method_parts(v: &JsValue) -> Option<(String, String)> {
    if let Some(JsValue::String(s)) = get_own_prop_value(v, "__native_method__") {
        let mut it = s.splitn(2, ':');
        let obj = it.next()?.to_string();
        let m = it.next()?.to_string();
        return Some((obj, m));
    }
    None
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

/// Own string keys whose property is enumerable (data **or** accessor),
/// excluding internal `__…__` markers. Used by object-rest destructuring, which
/// copies only enumerable own properties.
pub fn own_enumerable_string_keys(v: &JsValue) -> Vec<alloc::string::String> {
    let mut out = Vec::new();
    if let JsValue::Object(o) = v {
        let b = o.borrow();
        for (k, desc) in b.as_js_object().get_object_base().properties.iter() {
            if let PropertyKey::Str(name) = k {
                if name.starts_with("__") && name.ends_with("__") {
                    continue;
                }
                let enumerable = match desc {
                    PropertyDescriptor::Data(d) => d.enumerable,
                    PropertyDescriptor::Accessor(a) => a.enumerable,
                };
                if enumerable {
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

/// Bind a call's arguments to a function's formal parameters.
///
/// A parameter list is the full pattern grammar, not a list of names: an object
/// or array pattern (`function ({label, ok})`), a default (`a = 1`) and a rest
/// (`...args`) are all legal there. Binding only the plain identifiers left
/// every other form *silently unbound*, so the first read of one raised
/// `ReferenceError: label is not defined` from inside a function that was
/// called correctly — which is how a React component written as
/// `function Check({ label, ok })` failed with the parameter's name and no hint
/// that the parameter was the problem. `bind_pattern` already implements the
/// grammar for `var`/`let`, so parameters go through it rather than a second
/// copy that can drift.
pub(crate) fn bind_parameters(
    params: &[PatternType],
    args: &[JsValue],
    ctx: &mut EvalContext,
) -> Result<(), JErrorType> {
    for (i, param) in params.iter().enumerate() {
        match param {
            // Fast path: the overwhelmingly common plain `function (a, b)`.
            PatternType::PatternWhichCanBeExpression(ExpressionPatternType::Identifier(id)) => {
                let arg_value = args.get(i).cloned().unwrap_or(JsValue::Undefined);
                ctx.create_binding(&id.name, false)?;
                ctx.initialize_binding(&id.name, arg_value)?;
            }
            // `...rest` takes every remaining argument, so it is the one form
            // whose value is not `args[i]`.
            PatternType::RestElement { argument, .. } => {
                let rest: alloc::vec::Vec<JsValue> = args.iter().skip(i).cloned().collect();
                super::statement::bind_pattern(argument, make_array(rest), ctx, false, false)?;
            }
            other => {
                let arg_value = args.get(i).cloned().unwrap_or(JsValue::Undefined);
                super::statement::bind_pattern(other, arg_value, ctx, false, false)?;
            }
        }
    }
    Ok(())
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
        // A parenthesized anonymous function is still an anonymous function
        // definition for name inference (`f = (function(){})`). The parser
        // models `( expr )` as a single-element `SequenceExpression`; a multi-
        // element sequence (`(0, function(){})`) is a comma expression and does
        // NOT qualify.
        ExpressionType::SequenceExpression { expressions, .. } => {
            expressions.len() == 1 && is_anonymous_function_expr(&expressions[0])
        }
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
        // Only for callables: a simple function, or a class constructor (whether
        // user-defined or the synthesized default constructor).
        let is_fn = ["__simple_function__", "__class_constructor__", "__default_constructor__"]
            .iter()
            .any(|m| {
                matches!(
                    base.properties.get(&PropertyKey::Str(m.to_string())),
                    Some(PropertyDescriptor::Data(_))
                )
            });
        if !is_fn {
            return;
        }
        // Infer only when `name` is absent or still the empty string (a named
        // function/class keeps its own name).
        let empty = match base.properties.get(&PropertyKey::Str("name".to_string())) {
            Some(PropertyDescriptor::Data(d)) => {
                matches!(&d.value, JsValue::String(s) if s.is_empty())
            }
            None => true,
            _ => false,
        };
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

/// Map a `Symbol.<name>` static-member name to its canonical well-known symbol
/// value. Returns `None` for names that are not well-known symbols.
pub fn well_known_symbol(name: &str) -> Option<JsValue> {
    use crate::runner::ds::symbol::{
        SYMBOL_HAS_INSTANCE, SYMBOL_IS_CONCAT_SPREADABLE, SYMBOL_ITERATOR, SYMBOL_MATCH,
        SYMBOL_REPLACE, SYMBOL_SEARCH, SYMBOL_SPECIES, SYMBOL_SPLIT, SYMBOL_TO_PRIMITIVE,
        SYMBOL_TO_STRING_TAG,
    };
    let sym = match name {
        "iterator" => SYMBOL_ITERATOR.clone(),
        "hasInstance" => SYMBOL_HAS_INSTANCE.clone(),
        "isConcatSpreadable" => SYMBOL_IS_CONCAT_SPREADABLE.clone(),
        "match" => SYMBOL_MATCH.clone(),
        "replace" => SYMBOL_REPLACE.clone(),
        "search" => SYMBOL_SEARCH.clone(),
        "species" => SYMBOL_SPECIES.clone(),
        "split" => SYMBOL_SPLIT.clone(),
        "toPrimitive" => SYMBOL_TO_PRIMITIVE.clone(),
        "toStringTag" => SYMBOL_TO_STRING_TAG.clone(),
        _ => return None,
    };
    Some(JsValue::Symbol(sym))
}

/// The string property key under which a value's `Symbol.iterator` method is
/// stored. The interpreter keys symbol-valued properties by their stringified
/// form (`value_to_property_key`), so a lookup must use the same string.
fn symbol_iterator_key() -> String {
    value_to_property_key(&JsValue::Symbol(
        crate::runner::ds::symbol::SYMBOL_ITERATOR.clone(),
    ))
}

/// Does `value` carry a callable `[Symbol.iterator]` method (own or inherited)?
/// Used by array destructuring to decide between the index-based fast path
/// (plain arrays) and the generic iterator protocol.
pub fn has_custom_iterator(value: &JsValue, ctx: &mut EvalContext) -> bool {
    let key = symbol_iterator_key();
    match get_property_with_receiver(value, value, &key, ctx) {
        Ok(m) => value_is_callable(&m),
        Err(_) => false,
    }
}

/// Is `value` already an iterator — a generator object, or an object exposing a
/// callable `next` method?
fn is_iterator_like(value: &JsValue, ctx: &mut EvalContext) -> bool {
    if let JsValue::Object(o) = value {
        let marker = PropertyKey::Str("__generator_object__".to_string());
        if o.borrow()
            .as_js_object()
            .get_object_base()
            .properties
            .contains_key(&marker)
        {
            return true;
        }
    }
    match get_property_with_receiver(value, value, "next", ctx) {
        Ok(m) => value_is_callable(&m),
        Err(_) => false,
    }
}

/// Build a native default iterator over a genuine array or a string primitive.
/// Stepped directly by [`iterator_step`] (it carries no user-level `next`).
fn make_native_iterator(target: JsValue) -> JsValue {
    make_object(vec![
        ("__native_iter__".to_string(), JsValue::Boolean(true)),
        (
            "__native_iter_index__".to_string(),
            JsValue::Number(JsNumberType::Integer(0)),
        ),
        ("__native_iter_target__".to_string(), target),
    ])
}

/// ES `GetIterator(obj)` (sync form). Returns the iterator object that
/// [`iterator_step`]/[`iterator_close`] then drive.
pub fn get_iterator(value: &JsValue, ctx: &mut EvalContext) -> ValueResult {
    if matches!(value, JsValue::Null | JsValue::Undefined) {
        return Err(JErrorType::TypeError("value is not iterable".to_string()));
    }
    // 1. An explicit `[Symbol.iterator]` method wins (user iterables and any
    //    array/object that overrides the default).
    let key = symbol_iterator_key();
    let method = get_property_with_receiver(value, value, &key, ctx)?;
    if value_is_callable(&method) {
        let iter = call_value(&method, value.clone(), Vec::new(), ctx)?;
        if !matches!(iter, JsValue::Object(_)) {
            return Err(JErrorType::TypeError(
                "Result of Symbol.iterator method is not an object".to_string(),
            ));
        }
        return Ok(iter);
    }
    // 2. Genuine arrays and strings get the native default iterator.
    if is_array(value) || matches!(value, JsValue::String(_)) {
        return Ok(make_native_iterator(value.clone()));
    }
    // 3. A value that is already an iterator is its own iterator.
    if is_iterator_like(value, ctx) {
        return Ok(value.clone());
    }
    Err(JErrorType::TypeError("value is not iterable".to_string()))
}

/// ES `IteratorStep` + value read. `Ok(Some(v))` yields the next value,
/// `Ok(None)` signals the iterator is exhausted (`done: true`).
pub fn iterator_step(
    iter: &JsValue,
    ctx: &mut EvalContext,
) -> Result<Option<JsValue>, JErrorType> {
    // Native default iterator (array / string) — stepped directly.
    if matches!(
        get_own_prop_value(iter, "__native_iter__"),
        Some(JsValue::Boolean(true))
    ) {
        let idx = match get_own_prop_value(iter, "__native_iter_index__") {
            Some(JsValue::Number(JsNumberType::Integer(n))) => n.max(0) as usize,
            _ => 0,
        };
        let target = get_own_prop_value(iter, "__native_iter_target__").unwrap_or(JsValue::Undefined);
        let value = match &target {
            JsValue::String(s) => {
                let chars: Vec<char> = s.chars().collect();
                if idx < chars.len() {
                    Some(JsValue::String(chars[idx].to_string()))
                } else {
                    None
                }
            }
            _ => {
                if idx < array_len(&target) {
                    Some(get_own_prop_value(&target, &idx.to_string()).unwrap_or(JsValue::Undefined))
                } else {
                    None
                }
            }
        };
        return match value {
            Some(v) => {
                set_own_prop(
                    iter,
                    "__native_iter_index__",
                    JsValue::Number(JsNumberType::Integer((idx + 1) as i64)),
                    false,
                );
                Ok(Some(v))
            }
            None => Ok(None),
        };
    }
    // Generic path: call `iter.next()` and read `done`/`value` off the result.
    let next_method = get_property_with_receiver(iter, iter, "next", ctx)?;
    if !value_is_callable(&next_method) {
        return Err(JErrorType::TypeError(
            "iterator.next is not a function".to_string(),
        ));
    }
    let result = call_value(&next_method, iter.clone(), Vec::new(), ctx)?;
    if !matches!(result, JsValue::Object(_)) {
        return Err(JErrorType::TypeError(
            "iterator result is not an object".to_string(),
        ));
    }
    let done = get_property_with_receiver(&result, &result, "done", ctx)?;
    if to_boolean(&done) {
        Ok(None)
    } else {
        let value = get_property_with_receiver(&result, &result, "value", ctx)?;
        Ok(Some(value))
    }
}

/// ES `IteratorClose` (best-effort). Calls `iter.return()` if present, ignoring
/// its result/errors — used on an abrupt or early completion of destructuring.
pub fn iterator_close(iter: &JsValue, ctx: &mut EvalContext) {
    if matches!(
        get_own_prop_value(iter, "__native_iter__"),
        Some(JsValue::Boolean(true))
    ) {
        return;
    }
    if let Ok(ret) = get_property_with_receiver(iter, iter, "return", ctx) {
        if value_is_callable(&ret) {
            let _ = call_value(&ret, iter.clone(), Vec::new(), ctx);
        }
    }
}

/// Drive an iterator through an array binding/assignment pattern's elements,
/// invoking `bind` for each obtained value. Shared by `let/var/const`
/// destructuring (`bind_pattern`) and destructuring assignment
/// (`assign_pattern`); `bind` receives `(element_pattern, value)` and does the
/// actual binding. Handles elisions, a trailing rest element, and closing the
/// iterator on early/abrupt completion per the spec.
pub fn drive_array_pattern<F>(
    iter: &JsValue,
    elements: &[Option<Box<PatternType>>],
    ctx: &mut EvalContext,
    mut bind: F,
) -> Result<(), JErrorType>
where
    F: FnMut(&PatternType, JsValue, &mut EvalContext) -> Result<(), JErrorType>,
{
    let mut done = false;
    let mut outcome: Result<(), JErrorType> = Ok(());

    'outer: for element in elements.iter() {
        match element {
            // Elision (`[, x]`) — consume one step, bind nothing.
            None => {
                if !done {
                    match iterator_step(iter, ctx) {
                        Ok(Some(_)) => {}
                        Ok(None) => done = true,
                        Err(e) => {
                            done = true;
                            outcome = Err(e);
                            break 'outer;
                        }
                    }
                }
            }
            Some(elem_pattern) => {
                if let PatternType::RestElement { argument, .. } = elem_pattern.as_ref() {
                    // Rest element drains every remaining step into a new array.
                    // Bounded so a misbehaving/endless iterator throws instead of
                    // exhausting the kernel heap (same posture as the call-depth
                    // guard) — no legitimate `[...rest]` reaches 2^24 elements.
                    const REST_CAP: usize = 1 << 24;
                    let mut rest: Vec<JsValue> = Vec::new();
                    while !done {
                        if rest.len() >= REST_CAP {
                            done = true;
                            outcome = Err(JErrorType::RangeError(
                                "iterator produced too many values for a rest binding".to_string(),
                            ));
                            break 'outer;
                        }
                        match iterator_step(iter, ctx) {
                            Ok(Some(v)) => rest.push(v),
                            Ok(None) => done = true,
                            Err(e) => {
                                done = true;
                                outcome = Err(e);
                                break 'outer;
                            }
                        }
                    }
                    let rest_arr = make_array(rest);
                    if let Err(e) = bind(argument, rest_arr, ctx) {
                        outcome = Err(e);
                        break 'outer;
                    }
                } else {
                    let v = if done {
                        JsValue::Undefined
                    } else {
                        match iterator_step(iter, ctx) {
                            Ok(Some(v)) => v,
                            Ok(None) => {
                                done = true;
                                JsValue::Undefined
                            }
                            Err(e) => {
                                done = true;
                                outcome = Err(e);
                                break 'outer;
                            }
                        }
                    };
                    if let Err(e) = bind(elem_pattern, v, ctx) {
                        outcome = Err(e);
                        break 'outer;
                    }
                }
            }
        }
    }

    // Close the iterator whenever it wasn't exhausted (both on normal early
    // completion and on an abrupt completion that left `done` false).
    if !done {
        iterator_close(iter, ctx);
    }
    outcome
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
                // Resolve `Parent.prototype` through the normal property path so
                // built-in constructors (Error, Array, …) whose `prototype` is a
                // synthetic carrier still work with `class X extends Error`.
                let proto_val = get_property_with_ctx(&parent, "prototype", ctx)?;
                match proto_val {
                    JsValue::Object(proto) => Some((parent_obj.clone(), proto)),
                    JsValue::Null => {
                        return Err(JErrorType::TypeError(
                            "Class extends null is not supported".to_string(),
                        ));
                    }
                    _ => {
                        return Err(JErrorType::TypeError(
                            "Parent class prototype is not an object".to_string(),
                        ));
                    }
                }
            }
            _ => {
                return Err(JErrorType::TypeError(
                    "Class extends value is not a constructor".to_string(),
                ))
            }
        }
    } else {
        None
    };

    // 2. Find the constructor method
    let constructor_method = class_data.body.body.iter()
        .find(|m| matches!(m.kind, MethodDefinitionKind::Constructor));

    // 2b. Resolve instance-field initializers (ES2022 class fields). Computed
    // keys are evaluated once, now, at class-definition time (per spec); the
    // initializer expression pointer is applied to each `this` at construction.
    let mut instance_fields: Vec<(String, *const ExpressionType)> = Vec::new();
    for field in &class_data.body.fields {
        if field.is_static {
            continue;
        }
        let key = get_object_property_key(&field.key, field.computed, ctx)?;
        let init_ptr = field
            .value
            .as_deref()
            .map(|e| e as *const ExpressionType)
            .unwrap_or(core::ptr::null());
        instance_fields.push((key, init_ptr));
    }

    // 3. Create the constructor function
    let constructor_obj = if let Some(ctor_method) = constructor_method {
        // Use the defined constructor
        create_class_constructor(&ctor_method.value, parent_proto.as_ref().map(|(p, _)| p.clone()), instance_fields, ctx)?
    } else {
        // Create a default constructor
        create_default_constructor(parent_proto.as_ref().map(|(p, _)| p.clone()), instance_fields, ctx)?
    };

    // 3b. A class's `name` is its binding identifier (`class X {}` → "X"), or the
    // empty string for an anonymous class — which later name inference
    // (`let C = class {}`, `[C = class {}] = []`) fills in. Stored non-enumerable
    // like `Function.prototype.name`.
    {
        let class_name = class_data
            .id
            .as_ref()
            .map(|id| id.name.clone())
            .unwrap_or_default();
        if let JsValue::Object(o) = &constructor_obj {
            o.borrow_mut()
                .as_js_object_mut()
                .get_object_base_mut()
                .properties
                .insert(
                    PropertyKey::Str("name".to_string()),
                    PropertyDescriptor::Data(PropertyDescriptorData {
                        value: JsValue::String(class_name),
                        writable: false,
                        enumerable: false,
                        configurable: true,
                    }),
                );
        }
    }

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

    // 7. Static fields and static initialization blocks run at class-definition
    // time, in source order, with `this` bound to the constructor. (Static
    // fields are interleaved with blocks in the source; we run all fields then
    // blocks — a minor ordering simplification that real code rarely observes.)
    if !class_data.body.fields.iter().all(|f| !f.is_static)
        || !class_data.body.static_blocks.is_empty()
    {
        let saved_this = ctx.global_this.take();
        ctx.global_this = Some(constructor_obj.clone());
        let mut static_result: Result<(), JErrorType> = Ok(());
        'statics: for field in &class_data.body.fields {
            if !field.is_static {
                continue;
            }
            let key = match get_object_property_key(&field.key, field.computed, ctx) {
                Ok(k) => k,
                Err(e) => {
                    static_result = Err(e);
                    break 'statics;
                }
            };
            let value = match &field.value {
                Some(e) => match evaluate_expression(e, ctx) {
                    Ok(v) => v,
                    Err(e) => {
                        static_result = Err(e);
                        break 'statics;
                    }
                },
                None => JsValue::Undefined,
            };
            set_own_prop(&constructor_obj, &key, value, !key.starts_with('#'));
        }
        if static_result.is_ok() {
            'blocks: for block in &class_data.body.static_blocks {
                for stmt in &block.body {
                    match super::statement::execute_statement(stmt, ctx) {
                        Ok(completion) => {
                            if let super::types::CompletionType::Throw = completion.completion_type {
                                static_result = Err(JErrorType::Thrown(
                                    completion.value.clone().unwrap_or(JsValue::Undefined),
                                ));
                                break 'blocks;
                            }
                        }
                        Err(e) => {
                            static_result = Err(e);
                            break 'blocks;
                        }
                    }
                }
            }
        }
        ctx.global_this = saved_this;
        static_result?;
    }

    Ok(constructor_obj)
}

/// Create a constructor function from a method definition.
fn create_class_constructor(
    func_data: &FunctionData,
    parent_ctor: Option<JsObjectType>,
    instance_fields: Vec<(String, *const ExpressionType)>,
    ctx: &EvalContext,
) -> ValueResult {
    let body_ptr = func_data.body.as_ref() as *const _;
    let params_ptr = &func_data.params.list as *const _;
    let environment = ctx.lex_env.clone();

    let mut func_obj = SimpleFunctionObject::new(body_ptr, params_ptr, environment);
    func_obj.instance_fields = instance_fields;

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
    instance_fields: Vec<(String, *const ExpressionType)>,
    ctx: &EvalContext,
) -> ValueResult {
    // ChittiOS: when the class declares instance fields, the (otherwise no-op)
    // default constructor must still initialize them. Build a real
    // `SimpleFunctionObject` with a null body carrying the field inits; it keeps
    // the `__default_constructor__` marker so `invoke_constructor` still performs
    // the implicit `super(...args)` for a derived class *before* `call_with_this`
    // applies the fields, and a `__has_field_init__` marker so `call_function_object`
    // routes it through `call_with_this` instead of the no-op short-circuit.
    if !instance_fields.is_empty() {
        let mut func_obj = SimpleFunctionObject::new(
            core::ptr::null(),
            core::ptr::null(),
            ctx.lex_env.clone(),
        );
        func_obj.instance_fields = instance_fields;
        for marker in [
            "__simple_function__",
            "__class_constructor__",
            "__default_constructor__",
            "__has_field_init__",
        ] {
            func_obj.base.properties.insert(
                PropertyKey::Str(marker.to_string()),
                PropertyDescriptor::Data(PropertyDescriptorData {
                    value: JsValue::Boolean(true),
                    writable: false,
                    enumerable: false,
                    configurable: false,
                }),
            );
        }
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
        let obj_ref: JsObjectType =
            Rc::new(RefCell::new(ObjectType::Ordinary(Box::new(func_obj))));
        return Ok(JsValue::Object(obj_ref));
    }
    let _ = ctx;
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


#[cfg(test)]
mod to_primitive_tests {
    use super::*;
    use crate::runner::plugin::types::EvalContext;

    /// A non-object is returned unchanged by ToPrimitive.
    #[test]
    fn to_primitive_passthrough_non_object() {
        let mut ctx = EvalContext::new();
        let v = JsValue::Number(JsNumberType::Integer(5));
        assert_eq!(to_primitive(&v, "number", &mut ctx).unwrap(), v);
        let s = JsValue::String("hi".to_string());
        assert_eq!(to_primitive(&s, "string", &mut ctx).unwrap(), s);
    }

    /// A boxed primitive wrapper (`Object(prim)`) unwraps to its stored primitive.
    #[test]
    fn to_primitive_unwraps_boxed_primitive() {
        let mut ctx = EvalContext::new();
        let wrapper = make_object(alloc::vec![]);
        set_own_prop(
            &wrapper,
            "__primitive_value__",
            JsValue::BigInt(num_bigint::BigInt::from(2)),
            false,
        );
        match to_primitive(&wrapper, "default", &mut ctx).unwrap() {
            JsValue::BigInt(b) => assert_eq!(b, num_bigint::BigInt::from(2)),
            other => panic!("expected 2n, got {:?}", other),
        }
    }

    /// A plain object with no user coercion and no wrapper is left unchanged, so
    /// the legacy stringify/NaN path still applies (no corpus regression).
    #[test]
    fn to_primitive_leaves_plain_object() {
        let mut ctx = EvalContext::new();
        let obj = make_object(alloc::vec![("a".to_string(), JsValue::Boolean(true))]);
        assert!(matches!(
            to_primitive(&obj, "number", &mut ctx).unwrap(),
            JsValue::Object(_)
        ));
    }

    /// A computed object-literal / member key for a symbol keeps its unique
    /// stringified identity (not the lossy "[Symbol]" collapse), so distinct
    /// well-known symbols map to distinct property keys.
    #[test]
    fn symbol_keys_are_distinct() {
        let to_prim = value_to_property_key(&JsValue::Symbol(
            crate::runner::ds::symbol::SYMBOL_TO_PRIMITIVE.clone(),
        ));
        let iter = value_to_property_key(&JsValue::Symbol(
            crate::runner::ds::symbol::SYMBOL_ITERATOR.clone(),
        ));
        assert_ne!(to_prim, iter);
        assert!(to_prim.contains("toPrimitive"));
    }
}
