//! Core types for the evaluation engine.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::JsValue;

/// Completion record type.
/// Represents the result of evaluating a statement or expression.
#[derive(Debug, Clone, PartialEq)]
pub enum CompletionType {
    /// Normal completion - execution continues.
    Normal,
    /// Return completion - function returns.
    Return,
    /// Throw completion - exception thrown.
    Throw,
    /// Break completion - break from loop/switch.
    Break,
    /// Continue completion - continue loop iteration.
    Continue,
    /// Yield completion - generator yields a value.
    Yield,
}

/// Completion record.
/// Every statement evaluation returns a completion record.
pub struct Completion {
    /// The type of completion.
    pub completion_type: CompletionType,
    /// The value, if any.
    pub value: Option<JsValue>,
    /// Target label for break/continue.
    pub target: Option<String>,
}

impl Completion {
    /// Create a normal completion with no value.
    pub fn normal() -> Self {
        Completion {
            completion_type: CompletionType::Normal,
            value: None,
            target: None,
        }
    }

    /// Create a normal completion with a value.
    pub fn normal_with_value(value: JsValue) -> Self {
        Completion {
            completion_type: CompletionType::Normal,
            value: Some(value),
            target: None,
        }
    }

    /// Create a return completion.
    pub fn return_value(value: JsValue) -> Self {
        Completion {
            completion_type: CompletionType::Return,
            value: Some(value),
            target: None,
        }
    }

    /// Create a return completion with undefined.
    pub fn return_undefined() -> Self {
        Completion {
            completion_type: CompletionType::Return,
            value: Some(JsValue::Undefined),
            target: None,
        }
    }

    /// Create a throw completion (exception).
    pub fn throw(error: JErrorType) -> Self {
        Completion {
            completion_type: CompletionType::Throw,
            value: Some(JsValue::String(error.to_string())),
            target: None,
        }
    }

    /// Create a break completion.
    pub fn break_completion(target: Option<String>) -> Self {
        Completion {
            completion_type: CompletionType::Break,
            value: None,
            target,
        }
    }

    /// Create a continue completion.
    pub fn continue_completion(target: Option<String>) -> Self {
        Completion {
            completion_type: CompletionType::Continue,
            value: None,
            target,
        }
    }

    /// Create a yield completion with a value.
    pub fn yield_value(value: JsValue) -> Self {
        Completion {
            completion_type: CompletionType::Yield,
            value: Some(value),
            target: None,
        }
    }

    /// Check if this is a normal completion.
    pub fn is_normal(&self) -> bool {
        matches!(self.completion_type, CompletionType::Normal)
    }

    /// Check if this is an abrupt completion (not normal).
    pub fn is_abrupt(&self) -> bool {
        !self.is_normal()
    }

    /// Get the value, or undefined if none.
    pub fn get_value(&self) -> JsValue {
        self.value.clone().unwrap_or(JsValue::Undefined)
    }

    /// Update the value of a normal completion.
    pub fn update_empty(self, value: JsValue) -> Self {
        if self.is_normal() && self.value.is_none() {
            Completion {
                value: Some(value),
                ..self
            }
        } else {
            self
        }
    }
}

/// Reference base type.
/// A reference can be based on an object, environment record, or be unresolvable.
#[derive(Clone)]
pub enum ReferenceBase {
    /// Reference to a property on an object.
    Object(JsValue),
    /// Reference to a binding in an environment record.
    Environment(EnvironmentRef),
    /// Unresolvable reference (identifier not found).
    Unresolvable,
}

/// Reference to an environment record.
#[derive(Clone)]
pub struct EnvironmentRef {
    // Will be expanded to actual environment record reference
    pub name: String,
}

/// Reference type.
/// Used for identifier resolution and property access.
#[derive(Clone)]
pub struct Reference {
    /// The base value (object or environment).
    pub base: ReferenceBase,
    /// The referenced name (property name or identifier).
    pub referenced_name: String,
    /// Whether this is a strict mode reference.
    pub strict: bool,
    /// The `this` value for super references.
    pub this_value: Option<JsValue>,
}

impl Reference {
    /// Create a new reference to a property on an object.
    pub fn property(base: JsValue, name: impl Into<String>, strict: bool) -> Self {
        Reference {
            base: ReferenceBase::Object(base),
            referenced_name: name.into(),
            strict,
            this_value: None,
        }
    }

    /// Create a new reference to an environment binding.
    pub fn environment(env_name: impl Into<String>, binding_name: impl Into<String>, strict: bool) -> Self {
        Reference {
            base: ReferenceBase::Environment(EnvironmentRef { name: env_name.into() }),
            referenced_name: binding_name.into(),
            strict,
            this_value: None,
        }
    }

    /// Create an unresolvable reference.
    pub fn unresolvable(name: impl Into<String>, strict: bool) -> Self {
        Reference {
            base: ReferenceBase::Unresolvable,
            referenced_name: name.into(),
            strict,
            this_value: None,
        }
    }

    /// Check if this reference has a primitive base.
    pub fn has_primitive_base(&self) -> bool {
        match &self.base {
            ReferenceBase::Object(v) => matches!(v, JsValue::Boolean(_) | JsValue::String(_) | JsValue::Number(_)),
            _ => false,
        }
    }

    /// Check if this is a property reference.
    pub fn is_property_reference(&self) -> bool {
        matches!(self.base, ReferenceBase::Object(_)) || self.has_primitive_base()
    }

    /// Check if this reference is unresolvable.
    pub fn is_unresolvable(&self) -> bool {
        matches!(self.base, ReferenceBase::Unresolvable)
    }

    /// Check if this is a super reference.
    pub fn is_super_reference(&self) -> bool {
        self.this_value.is_some()
    }

    /// Get the base object value.
    pub fn get_base(&self) -> Option<&JsValue> {
        match &self.base {
            ReferenceBase::Object(v) => Some(v),
            _ => None,
        }
    }

    /// Get the this value for method calls.
    pub fn get_this_value(&self) -> JsValue {
        if let Some(ref this) = self.this_value {
            this.clone()
        } else if let ReferenceBase::Object(ref base) = self.base {
            base.clone()
        } else {
            JsValue::Undefined
        }
    }
}

/// Result type for evaluation operations.
pub type EvalResult = Result<Completion, JErrorType>;

/// Result type for value-returning operations.
pub type ValueResult = Result<JsValue, JErrorType>;

/// Result type for reference-returning operations.
pub type ReferenceResult = Result<Reference, JErrorType>;
