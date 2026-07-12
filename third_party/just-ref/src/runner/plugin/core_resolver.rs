//! Core plugin resolver — wraps the existing `BuiltInRegistry` as a `PluginResolver`.
//!
//! This makes all built-in objects (Math, console, String, etc.) available
//! through the super-global scope's lazy resolution mechanism.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::JsValue;
use crate::runner::plugin::registry::BuiltInRegistry;
use crate::runner::plugin::resolver::PluginResolver;
use crate::runner::plugin::types::EvalContext;

/// Wraps a `BuiltInRegistry` as a `PluginResolver`.
///
/// When the super-global scope queries for a name like `"Math"`,
/// this resolver checks the registry and materializes a proxy object
/// whose method calls are dispatched back to the registry.
pub struct CorePluginResolver {
    registry: BuiltInRegistry,
}

impl CorePluginResolver {
    pub fn new(registry: BuiltInRegistry) -> Self {
        CorePluginResolver { registry }
    }

    pub fn registry(&self) -> &BuiltInRegistry {
        &self.registry
    }
}

impl PluginResolver for CorePluginResolver {
    fn has_binding(&self, name: &str) -> bool {
        self.registry.has_object(name)
    }

    fn resolve(&self, name: &str, _ctx: &mut EvalContext) -> Result<JsValue, JErrorType> {
        if self.registry.has_object(name) {
            // Return a sentinel object that identifies this built-in.
            // The actual method dispatch happens via `call_method`.
            use crate::runner::ds::object::{ObjectType};
            use crate::runner::ds::object_property::{
                PropertyDescriptor, PropertyDescriptorData, PropertyKey,
            };
            use crate::runner::ds::object::JsObject;
            use crate::runner::plugin::types::SimpleObject;
            use core::cell::RefCell;
            use alloc::rc::Rc;

            let mut obj = SimpleObject::new();

            // Tag the object with its built-in name so CallMethod can identify it
            obj.get_object_base_mut().properties.insert(
                PropertyKey::Str("__builtin_name__".to_string()),
                PropertyDescriptor::Data(PropertyDescriptorData {
                    value: JsValue::String(name.to_string()),
                    writable: false,
                    enumerable: false,
                    configurable: false,
                }),
            );

            // Also add any static properties from the registry
            if let Some(builtin_obj) = self.registry.get_object(name) {
                for (prop_name, prop_value) in &builtin_obj.properties {
                    obj.get_object_base_mut().properties.insert(
                        PropertyKey::Str(prop_name.clone()),
                        PropertyDescriptor::Data(PropertyDescriptorData {
                            value: prop_value.clone(),
                            writable: false,
                            enumerable: true,
                            configurable: false,
                        }),
                    );
                }
            }

            Ok(JsValue::Object(Rc::new(RefCell::new(
                ObjectType::Ordinary(Box::new(obj)),
            ))))
        } else {
            Err(JErrorType::ReferenceError(format!(
                "{} is not defined",
                name
            )))
        }
    }

    fn call_method(
        &self,
        object_name: &str,
        method_name: &str,
        ctx: &mut EvalContext,
        this: JsValue,
        args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>> {
        self.registry
            .get_method(object_name, method_name)
            .map(|builtin_fn| builtin_fn.call(ctx, this, args))
    }

    fn has_method(&self, object_name: &str, method_name: &str) -> bool {
        self.registry.has_method(object_name, method_name)
    }

    fn call_constructor(
        &self,
        object_name: &str,
        ctx: &mut EvalContext,
        args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>> {
        self.registry
            .get_constructor(object_name)
            .map(|ctor_fn| ctor_fn.call(ctx, JsValue::Undefined, args))
    }

    fn name(&self) -> &str {
        "core"
    }
}
