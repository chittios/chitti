// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::value::JsValue;

#[derive(Debug, Clone)]
pub enum JErrorType {
    ReferenceError(String),
    TypeError(String),
    RangeError(String),
    SyntaxError(String),
    /// Special "error" type for generator yield (not a real error)
    YieldValue(JsValue),
}
impl JErrorType {
    pub fn new_copy(other: &Self) -> Self {
        match other {
            JErrorType::ReferenceError(m) => JErrorType::ReferenceError(m.to_string()),
            JErrorType::TypeError(m) => JErrorType::TypeError(m.to_string()),
            JErrorType::RangeError(m) => JErrorType::RangeError(m.to_string()),
            JErrorType::SyntaxError(m) => JErrorType::SyntaxError(m.to_string()),
            JErrorType::YieldValue(v) => JErrorType::YieldValue(v.clone()),
        }
    }
    pub fn to_string(&self) -> String {
        match self {
            JErrorType::ReferenceError(m) => format!("Uncaught reference error: {}.", m),
            JErrorType::TypeError(m) => format!("Uncaught type error: {}.", m),
            JErrorType::RangeError(m) => format!("Uncaught range error: {}.", m),
            JErrorType::SyntaxError(m) => format!("Uncaught syntax error: {}.", m),
            JErrorType::YieldValue(_) => "Yield outside of generator".to_string(),
        }
    }
}
