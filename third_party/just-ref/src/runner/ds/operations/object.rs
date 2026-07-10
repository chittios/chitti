// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;
use alloc::rc::Rc;

use crate::runner::ds::error::JErrorType;
use crate::runner::ds::execution_context::ExecutionContextStack;
use crate::runner::ds::object::{JsObject, JsObjectType, ObjectType};
use crate::runner::ds::object_property::{
    PropertyDescriptorSetter, PropertyKey,
};
use crate::runner::ds::operations::type_conversion::to_object;
use crate::runner::ds::realm::CodeRealm;
use crate::runner::ds::value::{JsValue, JsValueOrSelf};

pub fn get(
    ctx_stack: &mut ExecutionContextStack,
    o: &JsObjectType,
    p: &PropertyKey,
) -> Result<JsValue, JErrorType> {
    let o1 = (**o).borrow();
    get_from_js_object(ctx_stack, o1.as_js_object(), p)
}

pub fn get_from_js_object(
    ctx_stack: &mut ExecutionContextStack,
    o: &dyn JsObject,
    p: &PropertyKey,
) -> Result<JsValue, JErrorType> {
    o.get(ctx_stack, p, JsValueOrSelf::SelfValue)
}

pub fn get_v(
    ctx_stack: &mut ExecutionContextStack,
    v: &JsValue,
    p: &PropertyKey,
) -> Result<JsValue, JErrorType> {
    let o = to_object(v)?;
    if let JsValue::Object(o) = o {
        let o = (*o).borrow();
        o.as_js_object()
            .get(ctx_stack, p, JsValueOrSelf::ValueRef(v))
    } else {
        Ok(v.clone())
    }
}

pub fn set(
    ctx_stack: &mut ExecutionContextStack,
    o: &JsObjectType,
    p: PropertyKey,
    v: JsValue,
) -> Result<bool, JErrorType> {
    let mut o1 = (**o).borrow_mut();
    set_from_js_object(ctx_stack, o1.as_js_object_mut(), p, v)
}

pub fn set_from_js_object(
    ctx_stack: &mut ExecutionContextStack,
    o: &mut dyn JsObject,
    p: PropertyKey,
    v: JsValue,
) -> Result<bool, JErrorType> {
    o.set(ctx_stack, p, v, JsValueOrSelf::SelfValue)
}

pub fn get_method(
    ctx_stack: &mut ExecutionContextStack,
    v: &JsValue,
    p: &PropertyKey,
) -> Result<JsValue, JErrorType> {
    let f = get_v(ctx_stack, v, p)?;
    match &f {
        JsValue::Undefined => Ok(JsValue::Undefined),
        JsValue::Null => Ok(JsValue::Undefined),
        JsValue::Object(o) => {
            if (*o).borrow().is_callable() {
                Ok(f)
            } else {
                Err(JErrorType::TypeError(format!("'{}' is not a function", p)))
            }
        }
        _ => Err(JErrorType::TypeError(format!("'{}' is not a function", p))),
    }
}

pub fn get_function_realm(obj: &ObjectType) -> Result<Rc<RefCell<CodeRealm>>, JErrorType> {
    if let ObjectType::Function(f) = obj {
        Ok(f.get_function_object_base().realm.clone())
    } else {
        Err(JErrorType::TypeError(format!(
            "The object \"{}\" is not callable",
            obj
        )))
    }
}

pub fn has_own_property(obj: &JsObjectType, property: &PropertyKey) -> Result<bool, JErrorType> {
    has_own_property_from_js_object((**obj).borrow().as_js_object(), property)
}

pub fn has_own_property_from_js_object(
    obj: &dyn JsObject,
    property: &PropertyKey,
) -> Result<bool, JErrorType> {
    let desc = obj.get_own_property(property)?;
    Ok(if let Some(_) = desc { true } else { false })
}

pub fn define_property_or_throw(
    o: &mut dyn JsObject,
    p: PropertyKey,
    desc: PropertyDescriptorSetter,
) -> Result<(), JErrorType> {
    let err = format!("Defining property \"{}\" failed", &p);
    let r = o.define_own_property(p, desc)?;
    if r {
        Ok(())
    } else {
        Err(JErrorType::TypeError(err))
    }
}

