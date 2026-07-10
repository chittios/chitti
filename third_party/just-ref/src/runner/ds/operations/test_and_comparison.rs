// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::function_object::ConstructorKind;
use crate::runner::ds::object::{JsObject, ObjectType};
use crate::runner::ds::operations::type_conversion::{
    get_type, TYPE_STR_BOOLEAN, TYPE_STR_FUNCTION, TYPE_STR_NULL,
    TYPE_STR_NUMBER, TYPE_STR_OBJECT, TYPE_STR_STRING, TYPE_STR_SYMBOL, TYPE_STR_UNDEFINED,
};
use crate::runner::ds::value::{JsNumberType, JsValue};
use core::ops::Deref;
use core::ptr;

fn is_same_value<'a>(a: &'a JsValue, b: &'a JsValue, strict_mode: bool) -> bool {
    let type_a = get_type(a);
    let type_b = get_type(b);
    if type_a == type_b {
        if type_a == TYPE_STR_UNDEFINED || type_a == TYPE_STR_NULL {
            true
        } else {
            if type_a == TYPE_STR_NUMBER {
                if let JsValue::Number(na) = a {
                    if let JsValue::Number(nb) = b {
                        match na {
                            JsNumberType::NaN => {
                                if strict_mode {
                                    false
                                } else {
                                    if let JsNumberType::NaN = nb {
                                        true
                                    } else {
                                        false
                                    }
                                }
                            }
                            JsNumberType::PositiveInfinity => {
                                if let JsNumberType::PositiveInfinity = nb {
                                    true
                                } else {
                                    false
                                }
                            }
                            JsNumberType::NegativeInfinity => {
                                if let JsNumberType::NegativeInfinity = nb {
                                    true
                                } else {
                                    false
                                }
                            }
                            _ => {
                                let na_value = match na {
                                    JsNumberType::Float(f) => *f,
                                    JsNumberType::Integer(i) => *i as f64,
                                    _ => 0 as f64, // Not really possible
                                };
                                let nb_value = match nb {
                                    JsNumberType::Float(f) => *f,
                                    JsNumberType::Integer(i) => *i as f64,
                                    _ => return false,
                                };
                                na_value == nb_value
                            }
                        }
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else if type_a == TYPE_STR_STRING {
                if let JsValue::String(a_value) = a {
                    if let JsValue::String(b_value) = b {
                        if a_value == b_value {
                            return true;
                        }
                    }
                }
                false
            } else if type_a == TYPE_STR_BOOLEAN {
                if let JsValue::Boolean(a_value) = a {
                    if let JsValue::Boolean(b_value) = b {
                        if a_value == b_value {
                            return true;
                        }
                    }
                }
                false
            } else if type_a == TYPE_STR_SYMBOL {
                if let JsValue::Symbol(a_value) = a {
                    if let JsValue::Symbol(b_value) = b {
                        if a_value == b_value {
                            return true;
                        }
                    }
                }
                false
            } else if type_a == TYPE_STR_OBJECT || type_a == TYPE_STR_FUNCTION {
                if let JsValue::Object(a_value) = a {
                    if let JsValue::Object(b_value) = b {
                        if same_object(
                            a_value.deref().borrow().deref(),
                            b_value.deref().borrow().deref(),
                        ) {
                            return true;
                        }
                    }
                }
                false
            } else {
                false
            }
        }
    } else {
        false
    }
}

pub fn same_object<'a>(a: &'a ObjectType, b: &'a ObjectType) -> bool {
    a == b
}

pub fn same_js_object(a: &dyn JsObject, b: &dyn JsObject) -> bool {
    ptr::eq(a.get_object_base(), b.get_object_base())
}

pub fn same_value<'a>(a: &'a JsValue, b: &'a JsValue) -> bool {
    is_same_value(a, b, false)
}

pub fn is_constructor(obj: &ObjectType) -> bool {
    if let ObjectType::Function(f) = obj {
        if let ConstructorKind::None = f.get_function_object_base().constructor_kind {
            false
        } else {
            true
        }
    } else {
        false
    }
}
