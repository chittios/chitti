// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::execution_context::ExecutionContextStack;
use crate::runner::ds::function_object::JsFunctionObject;
use crate::runner::ds::iterator_object::JsIteratorObject;
use crate::runner::ds::object_property::{
    PropertyDescriptor, PropertyDescriptorAccessor, PropertyDescriptorData,
    PropertyDescriptorSetter, PropertyKey,
};
use crate::runner::ds::operations::object::{get, get_function_realm};
use crate::runner::ds::operations::test_and_comparison::{
    is_constructor, same_js_object, same_object, same_value,
};
use crate::runner::ds::operations::type_conversion::{
    canonical_numeric_index_string, to_string_int,
};
use crate::runner::ds::realm::WellKnownIntrinsics;
use crate::runner::ds::value::{JsValue, JsValueOrSelf};
use core::cell::RefCell;
use hashbrown::HashMap;
use core::fmt;
use core::fmt::{Display, Formatter};
use core::ops::Deref;
use alloc::rc::Rc;

lazy_static! {
    pub static ref OBJECT_PROTOTYPE_PROP: PropertyKey = PropertyKey::Str("prototype".to_string());
}

pub type JsObjectType = Rc<RefCell<ObjectType>>;

pub enum ObjectType {
    Ordinary(Box<dyn JsObject>),
    Function(Box<dyn JsFunctionObject>),
}
impl ObjectType {
    pub fn is_callable(&self) -> bool {
        match self {
            ObjectType::Function(_) => true,
            ObjectType::Ordinary(obj) => {
                // Check for SimpleFunctionObject marker property
                let marker = PropertyKey::Str("__simple_function__".to_string());
                obj.get_object_base().properties.contains_key(&marker)
            }
        }
    }

    pub fn is_constructor(&self) -> bool {
        is_constructor(self)
    }

    pub fn as_js_object(&self) -> &dyn JsObject {
        match self {
            ObjectType::Ordinary(o) => o.as_js_object(),
            ObjectType::Function(o) => o.as_js_object(),
        }
    }

    pub fn as_js_object_mut(&mut self) -> &mut dyn JsObject {
        match self {
            ObjectType::Ordinary(o) => o.as_js_object_mut(),
            ObjectType::Function(o) => o.as_js_object_mut(),
        }
    }

    pub fn as_js_function_object(&self) -> &dyn JsFunctionObject {
        match self {
            ObjectType::Ordinary(_o) => panic!("Not callable hence cannot get as JsFunctionObject"),
            ObjectType::Function(o) => o.as_js_function_object(),
        }
    }

    pub fn as_js_function_object_mut(&mut self) -> &mut dyn JsFunctionObject {
        match self {
            ObjectType::Ordinary(_o) => panic!("Not callable hence cannot get as JsFunctionObject"),
            ObjectType::Function(o) => o.as_js_function_object_mut(),
        }
    }
}
impl PartialEq for ObjectType {
    fn eq(&self, other: &Self) -> bool {
        match self {
            ObjectType::Ordinary(o) => {
                if let ObjectType::Ordinary(o1) = other {
                    same_js_object(o.deref(), o1.deref())
                } else {
                    false
                }
            }
            ObjectType::Function(o) => {
                if let ObjectType::Function(o1) = other {
                    same_js_object(o.deref().as_js_object(), o1.deref().as_js_object())
                } else {
                    false
                }
            }
        }
    }
}
impl Display for ObjectType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_js_object().to_string())
    }
}

pub struct ObjectBase {
    pub properties: HashMap<PropertyKey, PropertyDescriptor>,
    pub is_extensible: bool,
    pub prototype: Option<JsObjectType>,
}
impl ObjectBase {
    pub fn new() -> Self {
        ObjectBase {
            properties: HashMap::new(),
            is_extensible: true,
            prototype: None,
        }
    }
}

pub trait JsObject {
    fn get_object_base_mut(&mut self) -> &mut ObjectBase;

    fn get_object_base(&self) -> &ObjectBase;

    fn as_js_object(&self) -> &dyn JsObject;

    fn as_js_object_mut(&mut self) -> &mut dyn JsObject;

    fn get_prototype_of(&self) -> Option<JsObjectType> {
        match &self.get_object_base().prototype {
            None => None,
            Some(j) => Some(j.clone()),
        }
    }

    fn set_prototype_of(&mut self, prototype: Option<JsObjectType>) -> bool {
        let current_value = &self.get_object_base().prototype;
        let new_value: Option<JsObjectType> = match prototype {
            None => {
                if current_value.is_none() {
                    return true;
                } else {
                    None
                }
            }
            Some(p) => {
                if let Some(current_p) = current_value {
                    if same_object(
                        p.deref().borrow().deref(),
                        current_p.deref().borrow().deref(),
                    ) {
                        return true;
                    } else {
                        Some(p.clone())
                    }
                } else {
                    None
                }
            }
        };
        if self.is_extensible() {
            if let Some(new_value) = new_value {
                let mut p = Some(new_value.clone());
                loop {
                    if let Some(some_p) = p {
                        if same_js_object(self.as_js_object(), (*(*some_p).borrow()).as_js_object())
                        {
                            // To prevent circular chain
                            return false;
                        } else {
                            let t1 = (*some_p).borrow();
                            let t2 = (*t1).as_js_object().get_prototype_of();
                            p = t2;
                        }
                    } else {
                        break;
                    }
                }
                self.get_object_base_mut().prototype = Some(new_value);
            } else {
                self.get_object_base_mut().prototype = None;
            }
            true
        } else {
            false
        }
    }

    fn is_extensible(&self) -> bool {
        self.get_object_base().is_extensible
    }

    fn prevent_extensions(&mut self) -> bool {
        self.get_object_base_mut().is_extensible = false;
        true
    }

    fn get_own_property(
        &self,
        property: &PropertyKey,
    ) -> Result<Option<&PropertyDescriptor>, JErrorType> {
        Ok(self.get_object_base().properties.get(property))
    }

    fn define_own_property(
        &mut self,
        property: PropertyKey,
        descriptor_setter: PropertyDescriptorSetter,
    ) -> Result<bool, JErrorType> {
        ordinary_define_own_property(self, property, descriptor_setter)
    }

    fn has_property(&self, property: &PropertyKey) -> bool {
        if self.get_object_base().properties.contains_key(property) {
            true
        } else {
            match &self.get_object_base().prototype {
                None => false,
                Some(o) => (**o).borrow().as_js_object().has_property(property),
            }
        }
    }

    fn get(
        &self,
        ctx_stack: &mut ExecutionContextStack,
        property: &PropertyKey,
        receiver: JsValueOrSelf,
    ) -> Result<JsValue, JErrorType> {
        match self.get_own_property(property)? {
            None => match self.get_prototype_of() {
                None => Ok(JsValue::Undefined),
                Some(p) => (*(*p).borrow())
                    .as_js_object()
                    .get(ctx_stack, property, receiver),
            },
            Some(pd) => match pd {
                PropertyDescriptor::Data(PropertyDescriptorData { value, .. }) => Ok(value.clone()),
                PropertyDescriptor::Accessor(PropertyDescriptorAccessor { get, .. }) => match get {
                    None => Ok(JsValue::Undefined),
                    Some(getter) => (**getter).borrow().as_js_function_object().call(
                        getter,
                        ctx_stack,
                        receiver,
                        Vec::new(),
                    ),
                },
            },
        }
    }

    fn set(
        &mut self,
        ctx_stack: &mut ExecutionContextStack,
        property: PropertyKey,
        value: JsValue,
        receiver: JsValueOrSelf,
    ) -> Result<bool, JErrorType> {
        // Get own property first (immutable borrow)
        let own_prop = self.get_own_property(&property)?;

        match own_prop {
            None => match self.get_prototype_of() {
                None => {
                    self.get_object_base_mut().properties.insert(
                        property,
                        PropertyDescriptor::Data(PropertyDescriptorData {
                            value,
                            writable: true,
                            enumerable: true,
                            configurable: true,
                        }),
                    );
                    Ok(true)
                }
                Some(p) => (*p)
                    .borrow_mut()
                    .as_js_object_mut()
                    .set(ctx_stack, property, value, receiver),
            },
            Some(pd) => match pd {
                PropertyDescriptor::Data(PropertyDescriptorData {
                    value: _, writable, ..
                }) => {
                    if *writable {
                        match receiver {
                            JsValueOrSelf::ValueRef(value_ref) => match value_ref {
                                JsValue::Object(o) => {
                                    let mut ot_obj = (**o).borrow_mut();
                                    let obj = (*ot_obj).as_js_object_mut();
                                    set_or_update_data_descriptor(obj, property, &value)
                                }
                                _ => Ok(false),
                            },
                            JsValueOrSelf::SelfValue => {
                                set_or_update_data_descriptor(self.as_js_object_mut(), property, &value)
                            }
                        }
                    } else {
                        Ok(false)
                    }
                }
                PropertyDescriptor::Accessor(PropertyDescriptorAccessor { set, .. }) => match set {
                    None => Ok(false),
                    Some(setter) => {
                        (**setter).borrow_mut().as_js_function_object_mut().call(
                            setter,
                            ctx_stack,
                            receiver,
                            vec![value],
                        )?;
                        Ok(true)
                    }
                },
            },
        }
    }

    fn delete(&mut self, property: &PropertyKey) -> Result<bool, JErrorType> {
        Ok(match self.get_own_property(property)? {
            None => true,
            Some(pd) => {
                if pd.is_configurable() {
                    self.get_object_base_mut().properties.remove(property);
                    true
                } else {
                    false
                }
            }
        })
    }

    fn enumerate(&self) -> JsIteratorObject {
        todo!()
    }

    fn own_property_keys(&self, ctx_stack: &mut ExecutionContextStack) -> Vec<PropertyKey> {
        let mut int_keys = vec![];
        let mut str_keys = vec![];
        let mut sym_keys = vec![];
        for (key, _) in &self.get_object_base().properties {
            match key {
                PropertyKey::Str(d) => {
                    if let Some(idx) = canonical_numeric_index_string(ctx_stack, d) {
                        int_keys.push(idx);
                    } else {
                        str_keys.push(d.to_string());
                    }
                }
                PropertyKey::Sym(d) => {
                    sym_keys.push(d.clone());
                }
            }
        }
        int_keys.sort();

        let mut result = vec![];
        result.append(
            &mut int_keys
                .into_iter()
                .map(|v| PropertyKey::Str(to_string_int(v as i64)))
                .collect::<Vec<PropertyKey>>(),
        );
        result.append(
            &mut str_keys
                .into_iter()
                .map(|v| PropertyKey::Str(v))
                .collect::<Vec<PropertyKey>>(),
        );
        result.append(
            &mut sym_keys
                .into_iter()
                .map(|v| PropertyKey::Sym(v))
                .collect::<Vec<PropertyKey>>(),
        );
        result
    }

    fn to_string(&self) -> String {
        format!("object")
    }
}

pub struct CoreObject {
    base: ObjectBase,
}
impl CoreObject {
    fn new(proto: Option<JsObjectType>) -> Self {
        let mut o = CoreObject {
            base: ObjectBase::new(),
        };
        finish_object_create(&mut o, proto);
        o
    }

    fn new_from_constructor(
        ctx_stack: &mut ExecutionContextStack,
        constructor: JsObjectType,
        intrinsic_default_proto: &WellKnownIntrinsics,
    ) -> Result<Self, JErrorType> {
        let mut o = CoreObject {
            base: ObjectBase::new(),
        };
        finish_object_create_from_constructor(
            ctx_stack,
            &mut o,
            constructor,
            intrinsic_default_proto,
        )?;
        Ok(o)
    }
}
impl JsObject for CoreObject {
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

pub fn ordinary_define_own_property<J: JsObject + ?Sized>(
    o: &mut J,
    property: PropertyKey,
    mut descriptor_setter: PropertyDescriptorSetter,
) -> Result<bool, JErrorType> {
    let current_descriptor = o.get_own_property(&property)?;
    Ok(if let Some(current_descriptor) = current_descriptor {
        if descriptor_setter.is_empty() {
            true
        } else if descriptor_setter.are_all_fields_set()
            && current_descriptor == &descriptor_setter.descriptor
        {
            true
        } else {
            let descriptor = &descriptor_setter.descriptor;
            if !current_descriptor.is_configurable() {
                if descriptor.is_configurable() {
                    return Ok(false);
                } else {
                    if (current_descriptor.is_enumerable() && !descriptor.is_enumerable())
                        || (!current_descriptor.is_enumerable() && descriptor.is_enumerable())
                    {
                        return Ok(false);
                    }
                }
            }
            if !descriptor_setter.is_generic_descriptor() {
                if (current_descriptor.is_data_descriptor() && !descriptor.is_data_descriptor())
                    || (!current_descriptor.is_data_descriptor() && descriptor.is_data_descriptor())
                {
                    if !current_descriptor.is_configurable() {
                        return Ok(false);
                    }
                    descriptor_setter.descriptor = match descriptor_setter.descriptor {
                        PropertyDescriptor::Data(PropertyDescriptorData {
                            value,
                            writable,
                            ..
                        }) => PropertyDescriptor::Data(PropertyDescriptorData {
                            value,
                            writable,
                            enumerable: current_descriptor.is_enumerable(),
                            configurable: current_descriptor.is_configurable(),
                        }),
                        PropertyDescriptor::Accessor(PropertyDescriptorAccessor {
                            set: setter,
                            get: getter,
                            ..
                        }) => PropertyDescriptor::Accessor(PropertyDescriptorAccessor {
                            set: setter,
                            get: getter,
                            enumerable: current_descriptor.is_enumerable(),
                            configurable: current_descriptor.is_configurable(),
                        }),
                    };
                } else if current_descriptor.is_data_descriptor() && descriptor.is_data_descriptor()
                {
                    if !current_descriptor.is_configurable() {
                        if let PropertyDescriptor::Data(PropertyDescriptorData {
                            writable: current_writable,
                            value: current_value,
                            ..
                        }) = current_descriptor
                        {
                            if let PropertyDescriptor::Data(PropertyDescriptorData {
                                writable: desc_writable,
                                value: desc_value,
                                ..
                            }) = &descriptor
                            {
                                if !*current_writable && *desc_writable {
                                    return Ok(false);
                                }
                                if !*current_writable {
                                    if descriptor_setter.honour_value
                                        && !same_value(current_value, desc_value)
                                    {
                                        return Ok(false);
                                    }
                                }
                            }
                        }
                    }
                } else if current_descriptor.is_accessor_descriptor()
                    && descriptor.is_accessor_descriptor()
                {
                    if !current_descriptor.is_configurable() {
                        if let PropertyDescriptor::Accessor(PropertyDescriptorAccessor {
                            set: current_set,
                            get: current_get,
                            ..
                        }) = current_descriptor
                        {
                            if let PropertyDescriptor::Accessor(PropertyDescriptorAccessor {
                                set: desc_set,
                                get: desc_get,
                                ..
                            }) = &descriptor
                            {
                                if !(current_set.is_none() && desc_set.is_none())
                                    && descriptor_setter.honour_set
                                    && ((!current_set.is_none() && desc_set.is_none())
                                        || (current_set.is_none() && !desc_set.is_none())
                                        || !same_js_object(
                                            current_set.as_ref().unwrap().borrow().as_js_object(),
                                            desc_set.as_ref().unwrap().borrow().as_js_object(),
                                        ))
                                {
                                    return Ok(false);
                                }

                                if !(current_get.is_none() && desc_get.is_none())
                                    && descriptor_setter.honour_get
                                    && ((!current_get.is_none() && desc_get.is_none())
                                        || (current_get.is_none() && !desc_get.is_none())
                                        || !same_object(
                                            &(*current_get.as_ref().unwrap()).borrow(),
                                            &(*desc_get.as_ref().unwrap()).borrow(),
                                        ))
                                {
                                    return Ok(false);
                                }
                            }
                        }
                    }
                }
            }
            o.get_object_base_mut().properties.insert(
                property,
                PropertyDescriptor::new_from_property_descriptor_setter(descriptor_setter),
            );
            true
        }
    } else {
        if o.is_extensible() {
            o.get_object_base_mut().properties.insert(
                property,
                PropertyDescriptor::new_from_property_descriptor_setter(descriptor_setter),
            );
            true
        } else {
            false
        }
    })
}

fn set_or_update_data_descriptor(
    obj: &mut dyn JsObject,
    property: PropertyKey,
    value: &JsValue,
) -> Result<bool, JErrorType> {
    match obj.get_own_property(&property)? {
        None => {
            let desc_setter = PropertyDescriptorSetter::new_from_property_descriptor(
                PropertyDescriptor::Data(PropertyDescriptorData {
                    value: value.clone(),
                    writable: true,
                    enumerable: true,
                    configurable: true,
                }),
            );
            obj.define_own_property(property, desc_setter)?;
            Ok(true)
        }
        Some(pd) => match pd {
            PropertyDescriptor::Data(PropertyDescriptorData { writable, .. }) => {
                if *writable {
                    obj.define_own_property(
                        property,
                        PropertyDescriptorSetter {
                            honour_value: true,
                            honour_writable: false,
                            honour_set: false,
                            honour_get: false,
                            honour_enumerable: false,
                            honour_configurable: false,
                            descriptor: PropertyDescriptor::Data(PropertyDescriptorData {
                                value: value.clone(),
                                writable: false,
                                enumerable: false,
                                configurable: false,
                            }),
                        },
                    )
                } else {
                    Ok(false)
                }
            }
            PropertyDescriptor::Accessor { .. } => Ok(false),
        },
    }
}

pub fn object_create(proto: Option<JsObjectType>) -> impl JsObject {
    CoreObject::new(proto)
}

pub fn object_create_from_constructor(
    ctx_stack: &mut ExecutionContextStack,
    constructor: JsObjectType,
    intrinsic_default_proto: &WellKnownIntrinsics,
) -> Result<impl JsObject, JErrorType> {
    CoreObject::new_from_constructor(ctx_stack, constructor, intrinsic_default_proto)
}

pub fn finish_object_create(obj: &mut dyn JsObject, proto: Option<JsObjectType>) {
    obj.get_object_base_mut().prototype = proto;
    obj.get_object_base_mut().is_extensible = true;
}

pub fn finish_object_create_from_constructor(
    ctx_stack: &mut ExecutionContextStack,
    obj: &mut dyn JsObject,
    constructor: JsObjectType,
    intrinsic_default_proto: &WellKnownIntrinsics,
) -> Result<(), JErrorType> {
    let proto = get_prototype_from_constructor(ctx_stack, constructor, intrinsic_default_proto)?;
    finish_object_create(obj, Some(proto));
    Ok(())
}

pub fn get_prototype_from_constructor(
    ctx_stack: &mut ExecutionContextStack,
    constructor: JsObjectType,
    intrinsic_default_proto: &WellKnownIntrinsics,
) -> Result<JsObjectType, JErrorType> {
    let proto = get(ctx_stack, &constructor, &*OBJECT_PROTOTYPE_PROP)?;
    if let JsValue::Object(o) = proto {
        Ok(o)
    } else {
        let realm = get_function_realm(&*(*constructor).borrow())?;
        let proto = (*realm)
            .borrow()
            .get_intrinsics_value(intrinsic_default_proto)
            .clone();
        Ok(proto)
    }
}
