// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use core::fmt::{Display, Formatter};
use alloc::rc::Rc;

use crate::runner::ds::object::JsObjectType;
use crate::runner::ds::operations::type_conversion::{TYPE_STR_NULL, TYPE_STR_UNDEFINED};
use crate::runner::ds::symbol::SymbolData;

/// JavaScript value representation.
///
/// Represents all possible JavaScript values including primitives and objects.
/// This is the primary value type used throughout the engine.
///
/// # Examples
///
/// ```
/// use just::runner::ds::value::{JsValue, JsNumberType};
///
/// let undefined = JsValue::Undefined;
/// let null = JsValue::Null;
/// let boolean = JsValue::Boolean(true);
/// let string = JsValue::String("hello".to_string());
/// let number = JsValue::Number(JsNumberType::Integer(42));
/// ```
pub enum JsValue {
    /// The `undefined` value.
    Undefined,
    /// The `null` value.
    Null,
    /// A boolean value (`true` or `false`).
    Boolean(bool),
    /// A string value.
    String(String),
    /// A symbol value (ES6 symbols).
    Symbol(SymbolData),
    /// A numeric value (integer or float).
    Number(JsNumberType),
    /// An object value (including arrays, functions, etc.).
    Object(JsObjectType),
}
impl Clone for JsValue {
    fn clone(&self) -> Self {
        match self {
            JsValue::Undefined => JsValue::Undefined,
            JsValue::String(d) => JsValue::String(d.to_string()),
            JsValue::Boolean(d) => JsValue::Boolean(*d),
            JsValue::Null => JsValue::Null,
            JsValue::Number(d) => JsValue::Number(d.clone()),
            JsValue::Object(o) => JsValue::Object(o.clone()),
            JsValue::Symbol(d) => JsValue::Symbol(d.clone()),
        }
    }
}
impl Display for JsValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                JsValue::Undefined => TYPE_STR_UNDEFINED.to_string(),
                JsValue::Null => TYPE_STR_NULL.to_string(),
                JsValue::Boolean(b) => format!("bool({})", b),
                JsValue::String(s) => format!("\"{}\"", s),
                JsValue::Symbol(s) => s.to_string(),
                JsValue::Number(n) => n.to_string(),
                JsValue::Object(o) => (**o).borrow().as_js_object().to_string(),
            }
        )
    }
}

impl fmt::Debug for JsValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            JsValue::Undefined => write!(f, "JsValue::Undefined"),
            JsValue::Null => write!(f, "JsValue::Null"),
            JsValue::Boolean(b) => write!(f, "JsValue::Boolean({})", b),
            JsValue::String(s) => write!(f, "JsValue::String({:?})", s),
            JsValue::Symbol(s) => write!(f, "JsValue::Symbol({})", s),
            JsValue::Number(n) => write!(f, "JsValue::Number({:?})", n),
            JsValue::Object(_) => write!(f, "JsValue::Object(...)"),
        }
    }
}

impl PartialEq for JsValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (JsValue::Undefined, JsValue::Undefined) => true,
            (JsValue::Null, JsValue::Null) => true,
            (JsValue::Boolean(a), JsValue::Boolean(b)) => a == b,
            (JsValue::String(a), JsValue::String(b)) => a == b,
            (JsValue::Number(a), JsValue::Number(b)) => a == b,
            (JsValue::Object(a), JsValue::Object(b)) => Rc::ptr_eq(a, b),
            (JsValue::Symbol(a), JsValue::Symbol(b)) => a == b,
            _ => false,
        }
    }
}

/// JavaScript number type.
///
/// JavaScript numbers can be represented as either integers or floats.
/// The engine uses integers when possible for better performance.
///
/// # Examples
///
/// ```
/// use just::runner::ds::value::JsNumberType;
///
/// let int = JsNumberType::Integer(42);
/// let float = JsNumberType::Float(3.14);
/// let infinity = JsNumberType::Float(f64::INFINITY);
/// ```
#[derive(Debug, PartialEq)]
pub enum JsNumberType {
    /// An integer value (64-bit signed).
    Integer(i64),
    Float(f64),
    NaN,
    PositiveInfinity,
    NegativeInfinity,
}
impl Display for JsNumberType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            JsNumberType::Integer(i) => write!(f, "{}", i),
            JsNumberType::Float(nf) => write!(f, "{}", nf),
            JsNumberType::NaN => write!(f, "NaN"),
            JsNumberType::PositiveInfinity => write!(f, "+Infinity"),
            JsNumberType::NegativeInfinity => write!(f, "-Infinity"),
        }
    }
}
impl Clone for JsNumberType {
    fn clone(&self) -> Self {
        match self {
            JsNumberType::Integer(i) => JsNumberType::Integer(*i),
            JsNumberType::Float(nf) => JsNumberType::Float(*nf),
            JsNumberType::NaN => JsNumberType::NaN,
            JsNumberType::PositiveInfinity => JsNumberType::PositiveInfinity,
            JsNumberType::NegativeInfinity => JsNumberType::NegativeInfinity,
        }
    }
}

pub enum JsValueOrSelf<'a> {
    ValueRef(&'a JsValue),
    SelfValue,
}
