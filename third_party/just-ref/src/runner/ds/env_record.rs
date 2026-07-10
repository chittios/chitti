// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use hashbrown::HashMap;

use crate::runner::ds::error::JErrorType;
use crate::runner::ds::execution_context::ExecutionContextStack;
use crate::runner::ds::lex_env::{JsLexEnvironmentType, LexEnvironment};
use crate::runner::ds::object::JsObjectType;
use crate::runner::ds::object_property::{
    PropertyDescriptor, PropertyDescriptorData, PropertyDescriptorSetter, PropertyKey,
};
use crate::runner::ds::operations::object::{define_property_or_throw, get, has_own_property, set};
use crate::runner::ds::value::JsValue;
use core::cell::RefCell;
use alloc::rc::Rc;

pub trait EnvironmentRecord {
    fn has_binding(&self, name: &String) -> bool;
    fn create_mutable_binding(&mut self, name: String, can_delete: bool) -> Result<(), JErrorType>;
    fn create_immutable_binding(&mut self, name: String) -> Result<(), JErrorType>;
    fn initialize_binding(
        &mut self,
        ctx_stack: &mut ExecutionContextStack,
        name: String,
        value: JsValue,
    ) -> Result<bool, JErrorType>;
    fn set_mutable_binding(
        &mut self,
        ctx_stack: &mut ExecutionContextStack,
        name: String,
        value: JsValue,
    ) -> Result<(), JErrorType>;
    fn get_binding_value(
        &self,
        ctx_stack: &mut ExecutionContextStack,
        name: &String,
    ) -> Result<JsValue, JErrorType>;
    fn delete_binding(&mut self, name: &String) -> Result<bool, JErrorType>;
    fn has_this_binding(&self) -> bool;
    fn has_super_binding(&self) -> bool;
    /// Get all bindings as a vector of (name, value) pairs. Used for generators.
    fn get_all_bindings(&self) -> Option<Vec<(String, JsValue)>> {
        None // Default implementation returns None
    }
}

pub enum EnvironmentRecordType {
    Declarative(DeclarativeEnvironmentRecord),
    Object(ObjectEnvironmentRecord),
    Function(FunctionEnvironmentRecord),
    Global(GlobalEnvironmentRecord),
}
impl EnvironmentRecordType {
    pub fn as_env_record(&self) -> &dyn EnvironmentRecord {
        match self {
            EnvironmentRecordType::Declarative(d) => d,
            EnvironmentRecordType::Object(d) => d,
            EnvironmentRecordType::Function(d) => d,
            EnvironmentRecordType::Global(d) => d,
        }
    }
}

#[derive(PartialEq)]
pub enum BindingFlag {
    NoDelete,
    IsImmutable,
}

pub struct DeclarativeEnvironmentRecord {
    bindings: HashMap<String, Option<JsValue>>,
    binding_flags: HashMap<String, Vec<BindingFlag>>,
}
impl DeclarativeEnvironmentRecord {
    pub fn new() -> Self {
        DeclarativeEnvironmentRecord {
            bindings: HashMap::new(),
            binding_flags: HashMap::new(),
        }
    }
}
impl EnvironmentRecord for DeclarativeEnvironmentRecord {
    fn has_binding(&self, id: &String) -> bool {
        self.bindings.contains_key(id)
    }

    fn create_mutable_binding(&mut self, id: String, can_delete: bool) -> Result<(), JErrorType> {
        if !self.has_binding(&id) {
            self.bindings.insert(id.to_string(), None);
            if can_delete {
                self.binding_flags.insert(id, vec![]);
            } else {
                self.binding_flags.insert(id, vec![BindingFlag::NoDelete]);
            }
        }
        Ok(())
    }

    fn create_immutable_binding(&mut self, id: String) -> Result<(), JErrorType> {
        if !self.has_binding(&id) {
            self.bindings.insert(id.to_string(), None);
            self.binding_flags
                .insert(id, vec![BindingFlag::IsImmutable]);
        }
        Ok(())
    }

    fn initialize_binding(
        &mut self,
        _ctx_stack: &mut ExecutionContextStack,
        id: String,
        value: JsValue,
    ) -> Result<bool, JErrorType> {
        if let Some(v) = self.bindings.get(&id) {
            if v.is_none() {
                self.bindings.insert(id, Some(value));
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    fn set_mutable_binding(
        &mut self,
        _ctx_stack: &mut ExecutionContextStack,
        id: String,
        value: JsValue,
    ) -> Result<(), JErrorType> {
        if let Some(v) = self.bindings.get(&id) {
            if !v.is_none() {
                if !self
                    .binding_flags
                    .get(&id)
                    .unwrap()
                    .contains(&BindingFlag::IsImmutable)
                {
                    self.bindings.insert(id, Some(value));
                    Ok(())
                } else {
                    Err(JErrorType::TypeError(format!(
                        "'{}' is set and immutable",
                        id
                    )))
                }
            } else {
                Err(JErrorType::ReferenceError(format!(
                    "'{}' is not initialized",
                    id
                )))
            }
        } else {
            Err(JErrorType::ReferenceError(format!(
                "'{}' is not defined",
                id
            )))
        }
    }

    fn get_binding_value(
        &self,
        _ctx_stack: &mut ExecutionContextStack,
        id: &String,
    ) -> Result<JsValue, JErrorType> {
        match self.bindings.get(id) {
            None => Err(JErrorType::ReferenceError(format!(
                "'{}' is not defined",
                id
            ))),
            Some(v) => match v {
                None => Err(JErrorType::ReferenceError(format!(
                    "'{}' is not initialized",
                    id
                ))),
                Some(v) => Ok(v.clone()),
            },
        }
    }

    fn delete_binding(&mut self, id: &String) -> Result<bool, JErrorType> {
        Ok(if let Some(flags) = self.binding_flags.get(id) {
            if flags.contains(&BindingFlag::NoDelete) {
                false
            } else {
                self.bindings.remove(id);
                self.binding_flags.remove(id);
                true
            }
        } else {
            false
        })
    }

    fn has_this_binding(&self) -> bool {
        false
    }

    fn has_super_binding(&self) -> bool {
        false
    }

    fn get_all_bindings(&self) -> Option<Vec<(String, JsValue)>> {
        let mut result = Vec::new();
        for (name, value) in &self.bindings {
            if let Some(v) = value {
                result.push((name.clone(), v.clone()));
            }
        }
        Some(result)
    }
}

pub struct ObjectEnvironmentRecord {
    binding_object: JsObjectType,
}
impl ObjectEnvironmentRecord {
    pub fn new(o: JsObjectType) -> Self {
        ObjectEnvironmentRecord { binding_object: o }
    }
}
impl EnvironmentRecord for ObjectEnvironmentRecord {
    fn has_binding(&self, name: &String) -> bool {
        self.binding_object
            .borrow()
            .as_js_object()
            .has_property(&PropertyKey::Str(name.to_string()))
    }

    fn create_mutable_binding(&mut self, name: String, can_delete: bool) -> Result<(), JErrorType> {
        define_property_or_throw(
            (*self.binding_object).borrow_mut().as_js_object_mut(),
            PropertyKey::Str(name),
            PropertyDescriptorSetter::new_from_property_descriptor(PropertyDescriptor::Data(
                PropertyDescriptorData {
                    value: JsValue::Undefined,
                    writable: true,
                    enumerable: true,
                    configurable: can_delete,
                },
            )),
        )
    }

    fn create_immutable_binding(&mut self, _name: String) -> Result<(), JErrorType> {
        panic!("Not supported")
    }

    fn initialize_binding(
        &mut self,
        ctx_stack: &mut ExecutionContextStack,
        name: String,
        value: JsValue,
    ) -> Result<bool, JErrorType> {
        self.set_mutable_binding(ctx_stack, name, value)?;
        Ok(true)
    }

    fn set_mutable_binding(
        &mut self,
        ctx_stack: &mut ExecutionContextStack,
        name: String,
        value: JsValue,
    ) -> Result<(), JErrorType> {
        set(
            ctx_stack,
            &mut self.binding_object,
            PropertyKey::Str(name),
            value,
        )?;
        Ok(())
    }

    fn get_binding_value(
        &self,
        ctx_stack: &mut ExecutionContextStack,
        name: &String,
    ) -> Result<JsValue, JErrorType> {
        let p = PropertyKey::Str(name.to_string());
        if self.binding_object.borrow().as_js_object().has_property(&p) {
            get(ctx_stack, &self.binding_object, &p)
        } else {
            Err(JErrorType::ReferenceError(format!(
                "'{}' reference is undefined",
                name
            )))
        }
    }

    fn delete_binding(&mut self, name: &String) -> Result<bool, JErrorType> {
        (*self.binding_object)
            .borrow_mut()
            .as_js_object_mut()
            .delete(&PropertyKey::Str(name.to_string()))
    }

    fn has_this_binding(&self) -> bool {
        false
    }

    fn has_super_binding(&self) -> bool {
        false
    }
}

pub struct FunctionEnvironmentRecord {
    base_env: DeclarativeEnvironmentRecord,
    this_value: Option<JsValue>,
    is_lexical_binding: bool,
    _function_object: JsObjectType,
    home_object: Option<JsObjectType>,
    _new_target: Option<JsObjectType>,
}
impl FunctionEnvironmentRecord {
    pub fn new(f: JsObjectType, new_target: Option<JsObjectType>) -> Self {
        // Extract values from borrow before moving f
        let (is_lexical, home_object) = {
            let func = (*f).borrow();
            let func_obj = func.as_js_function_object();
            let base = func_obj.get_function_object_base();
            (
                base.is_lexical,
                base.home_object.as_ref().map(|ho| ho.clone()),
            )
        };
        FunctionEnvironmentRecord {
            base_env: DeclarativeEnvironmentRecord::new(),
            this_value: None,
            is_lexical_binding: is_lexical,
            _function_object: f,
            home_object,
            _new_target: new_target,
        }
    }

    pub fn bind_this_value(&mut self, this: JsValue) -> Result<bool, JErrorType> {
        if !self.is_lexical_binding {
            if let Some(_) = &self.this_value {
                Err(JErrorType::ReferenceError(
                    "'this' is already initialized".to_string(),
                ))
            } else {
                self.this_value = Some(this);
                Ok(true)
            }
        } else {
            Err(JErrorType::TypeError(
                "Cannot set 'this' of Arrow Function".to_string(),
            ))
        }
    }

    pub fn get_this_binding(&self) -> Result<&JsValue, JErrorType> {
        if self.is_lexical_binding {
            Err(JErrorType::TypeError(
                "Cannot get 'this' of Arrow Function".to_string(),
            ))
        } else {
            if let Some(this) = &self.this_value {
                Ok(this)
            } else {
                Err(JErrorType::ReferenceError(
                    "'this' is not initialized".to_string(),
                ))
            }
        }
    }

    pub fn get_super_base(&self) -> Option<JsObjectType> {
        if let Some(ho) = &self.home_object {
            ho.borrow().as_js_object().get_prototype_of()
        } else {
            None
        }
    }
}
impl EnvironmentRecord for FunctionEnvironmentRecord {
    fn has_binding(&self, name: &String) -> bool {
        self.base_env.has_binding(name)
    }

    fn create_mutable_binding(&mut self, name: String, can_delete: bool) -> Result<(), JErrorType> {
        self.base_env.create_mutable_binding(name, can_delete)
    }

    fn create_immutable_binding(&mut self, name: String) -> Result<(), JErrorType> {
        self.base_env.create_immutable_binding(name)
    }

    fn initialize_binding(
        &mut self,
        ctx_stack: &mut ExecutionContextStack,
        name: String,
        value: JsValue,
    ) -> Result<bool, JErrorType> {
        self.base_env.initialize_binding(ctx_stack, name, value)
    }

    fn set_mutable_binding(
        &mut self,
        ctx_stack: &mut ExecutionContextStack,
        name: String,
        value: JsValue,
    ) -> Result<(), JErrorType> {
        self.base_env.set_mutable_binding(ctx_stack, name, value)
    }

    fn get_binding_value(
        &self,
        ctx_stack: &mut ExecutionContextStack,
        name: &String,
    ) -> Result<JsValue, JErrorType> {
        self.base_env.get_binding_value(ctx_stack, name)
    }

    fn delete_binding(&mut self, name: &String) -> Result<bool, JErrorType> {
        self.base_env.delete_binding(name)
    }

    fn has_this_binding(&self) -> bool {
        !self.is_lexical_binding
    }

    fn has_super_binding(&self) -> bool {
        if self.is_lexical_binding {
            false
        } else {
            self.home_object.is_some()
        }
    }
}

pub struct GlobalEnvironmentRecord {
    object_record: ObjectEnvironmentRecord,
    declarative_record: DeclarativeEnvironmentRecord,
    var_names: Vec<String>,
}
impl GlobalEnvironmentRecord {
    pub fn new(global_object: JsObjectType) -> Self {
        GlobalEnvironmentRecord {
            object_record: ObjectEnvironmentRecord::new(global_object),
            declarative_record: DeclarativeEnvironmentRecord::new(),
            var_names: Vec::new(),
        }
    }

    pub fn get_this_binding(&self) -> &JsObjectType {
        &self.object_record.binding_object
    }

    pub fn has_var_declaration(&self, name: &String) -> bool {
        self.var_names.contains(name)
    }

    pub fn has_lexical_declaration(&self, name: &String) -> bool {
        self.declarative_record.has_binding(name)
    }

    pub fn has_restricted_global_property(&self, name: &String) -> Result<bool, JErrorType> {
        Ok(
            if let Some(desc) = self
                .object_record
                .binding_object
                .borrow()
                .as_js_object()
                .get_own_property(&PropertyKey::Str(name.to_string()))?
            {
                !desc.is_configurable()
            } else {
                false
            },
        )
    }

    pub fn can_declare_global_var(&self, name: &String) -> Result<bool, JErrorType> {
        Ok(
            if has_own_property(
                &self.object_record.binding_object,
                &PropertyKey::Str(name.to_string()),
            )? {
                true
            } else {
                self.object_record
                    .binding_object
                    .borrow()
                    .as_js_object()
                    .is_extensible()
            },
        )
    }

    pub fn can_declare_global_function(&self, name: &String) -> Result<bool, JErrorType> {
        Ok(
            if let Some(desc) = self
                .object_record
                .binding_object
                .borrow()
                .as_js_object()
                .get_own_property(&PropertyKey::Str(name.to_string()))?
            {
                if desc.is_configurable() {
                    true
                } else if desc.is_data_descriptor() && desc.is_enumerable() {
                    if let PropertyDescriptor::Data(PropertyDescriptorData { writable, .. }) = desc
                    {
                        if *writable {
                            return Ok(true);
                        }
                    }
                    false
                } else {
                    false
                }
            } else {
                self.object_record
                    .binding_object
                    .borrow()
                    .as_js_object()
                    .is_extensible()
            },
        )
    }

    pub fn create_global_var_binding(
        &mut self,
        ctx_stack: &mut ExecutionContextStack,
        name: String,
        can_delete: bool,
    ) -> Result<(), JErrorType> {
        let has_property = has_own_property(
            &self.object_record.binding_object,
            &PropertyKey::Str(name.to_string()),
        )?;
        let is_extensible = self
            .object_record
            .binding_object
            .borrow()
            .as_js_object()
            .is_extensible();
        if !has_property && is_extensible {
            self.object_record
                .create_mutable_binding(name.to_string(), can_delete)?;
            self.object_record.initialize_binding(
                ctx_stack,
                name.to_string(),
                JsValue::Undefined,
            )?;
        }
        if !self.var_names.contains(&name) {
            self.var_names.push(name);
        }
        Ok(())
    }

    pub fn create_global_function_binding(
        &mut self,
        name: String,
        f: JsValue,
        can_delete: bool,
    ) -> Result<(), JErrorType> {
        let set_new_desc = if let Some(desc) = self
            .object_record
            .binding_object
            .borrow()
            .as_js_object()
            .get_own_property(&PropertyKey::Str(name.to_string()))?
        {
            desc.is_configurable()
        } else {
            true
        };
        // let f_clone = f.clone();
        let new_desc = if set_new_desc {
            PropertyDescriptorSetter::new_from_property_descriptor(PropertyDescriptor::Data(
                PropertyDescriptorData {
                    value: f,
                    writable: true,
                    enumerable: true,
                    configurable: can_delete,
                },
            ))
        } else {
            PropertyDescriptorSetter {
                honour_value: true,
                honour_writable: false,
                honour_set: false,
                honour_get: false,
                honour_enumerable: false,
                honour_configurable: false,
                descriptor: PropertyDescriptor::Data(PropertyDescriptorData {
                    value: f,
                    writable: false,
                    enumerable: false,
                    configurable: false,
                }),
            }
        };
        define_property_or_throw(
            (*self.object_record.binding_object)
                .borrow_mut()
                .as_js_object_mut(),
            PropertyKey::Str(name.to_string()),
            new_desc,
        )?;
        // Commenting out this code. The specs (on Pg. 89) requires this but not sure why, seems redundant.
        // set(
        //     &mut self.object_record.binding_object,
        //     PropertyKey::Str(name.to_string()),
        //     f_clone,
        // )?;
        if !self.var_names.contains(&name) {
            self.var_names.push(name);
        }
        Ok(())
    }
}
impl EnvironmentRecord for GlobalEnvironmentRecord {
    fn has_binding(&self, name: &String) -> bool {
        if self.declarative_record.has_binding(name) {
            true
        } else {
            self.object_record.has_binding(name)
        }
    }

    fn create_mutable_binding(&mut self, name: String, can_delete: bool) -> Result<(), JErrorType> {
        if self.declarative_record.has_binding(&name) {
            Err(JErrorType::TypeError(format!(
                "'{}' binding is already present",
                name
            )))
        } else {
            self.declarative_record
                .create_mutable_binding(name, can_delete)
        }
    }

    fn create_immutable_binding(&mut self, name: String) -> Result<(), JErrorType> {
        if self.declarative_record.has_binding(&name) {
            Err(JErrorType::TypeError(format!(
                "'{}' binding is already present",
                name
            )))
        } else {
            self.declarative_record.create_immutable_binding(name)
        }
    }

    fn initialize_binding(
        &mut self,
        ctx_stack: &mut ExecutionContextStack,
        name: String,
        value: JsValue,
    ) -> Result<bool, JErrorType> {
        if self.declarative_record.has_binding(&name) {
            self.declarative_record
                .initialize_binding(ctx_stack, name, value)
        } else if self.object_record.has_binding(&name) {
            self.object_record
                .initialize_binding(ctx_stack, name, value)
        } else {
            Ok(false)
        }
    }

    fn set_mutable_binding(
        &mut self,
        ctx_stack: &mut ExecutionContextStack,
        name: String,
        value: JsValue,
    ) -> Result<(), JErrorType> {
        if self.declarative_record.has_binding(&name) {
            self.declarative_record
                .set_mutable_binding(ctx_stack, name, value)
        } else if self.object_record.has_binding(&name) {
            self.object_record
                .set_mutable_binding(ctx_stack, name, value)
        } else {
            Ok(())
        }
    }

    fn get_binding_value(
        &self,
        ctx_stack: &mut ExecutionContextStack,
        name: &String,
    ) -> Result<JsValue, JErrorType> {
        if self.declarative_record.has_binding(name) {
            self.declarative_record.get_binding_value(ctx_stack, name)
        } else {
            self.object_record.get_binding_value(ctx_stack, name)
        }
    }

    fn delete_binding(&mut self, name: &String) -> Result<bool, JErrorType> {
        if self.declarative_record.has_binding(name) {
            self.declarative_record.delete_binding(name)
        } else {
            Ok(
                if has_own_property(
                    &self.object_record.binding_object,
                    &PropertyKey::Str(name.to_string()),
                )? {
                    if self.object_record.delete_binding(name)? {
                        self.var_names.retain(|n| n != name);
                        true
                    } else {
                        false
                    }
                } else {
                    true
                },
            )
        }
    }

    fn has_this_binding(&self) -> bool {
        true
    }

    fn has_super_binding(&self) -> bool {
        false
    }
}

pub fn new_declarative_environment(
    outer_lex: Option<JsLexEnvironmentType>,
) -> JsLexEnvironmentType {
    Rc::new(RefCell::new(LexEnvironment {
        inner: Box::new(EnvironmentRecordType::Declarative(
            DeclarativeEnvironmentRecord::new(),
        )),
        outer: match outer_lex {
            None => None,
            Some(o) => Some(o.clone()),
        },
    }))
}

pub fn new_object_environment(
    o: JsObjectType,
    outer_lex: Option<JsLexEnvironmentType>,
) -> JsLexEnvironmentType {
    Rc::new(RefCell::new(LexEnvironment {
        inner: Box::new(EnvironmentRecordType::Object(ObjectEnvironmentRecord::new(
            o,
        ))),
        outer: match outer_lex {
            None => None,
            Some(o) => Some(o.clone()),
        },
    }))
}

pub fn new_function_environment(
    f: JsObjectType,
    new_target: Option<JsObjectType>,
) -> JsLexEnvironmentType {
    assert!((*f).borrow().is_callable(), "f needs to be callable");
    if let Some(new_target) = &new_target {
        assert!(
            (**new_target).borrow().is_callable(),
            "new_target needs to be callable"
        );
    }
    let outer_lex = (*f)
        .borrow()
        .as_js_function_object()
        .get_function_object_base()
        .environment
        .clone();
    Rc::new(RefCell::new(LexEnvironment {
        inner: Box::new(EnvironmentRecordType::Function(
            FunctionEnvironmentRecord::new(f, new_target),
        )),
        outer: Some(outer_lex),
    }))
}

pub fn new_global_environment(global_object: JsObjectType) -> JsLexEnvironmentType {
    Rc::new(RefCell::new(LexEnvironment {
        inner: Box::new(EnvironmentRecordType::Global(GlobalEnvironmentRecord::new(
            global_object,
        ))),
        outer: None,
    }))
}
