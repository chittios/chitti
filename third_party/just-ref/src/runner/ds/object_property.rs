// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::object::JsObjectType;
use crate::runner::ds::operations::test_and_comparison::{same_object, same_value};
use crate::runner::ds::symbol::SymbolData;
use crate::runner::ds::value::JsValue;
use core::fmt;
use core::fmt::{Display, Formatter};
use core::hash::{Hash, Hasher};

pub enum PropertyKey {
    Str(String),
    Sym(SymbolData),
}
impl Display for PropertyKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            PropertyKey::Str(s) => write!(f, "\"{}\"", s),
            PropertyKey::Sym(s) => write!(f, "{}", s),
        }
    }
}
impl PartialEq for PropertyKey {
    fn eq(&self, other: &Self) -> bool {
        match self {
            PropertyKey::Str(s) => {
                if let PropertyKey::Str(s2) = other {
                    s == s2
                } else {
                    false
                }
            }
            PropertyKey::Sym(s) => {
                if let PropertyKey::Sym(s2) = other {
                    s == s2
                } else {
                    false
                }
            }
        }
    }
}
impl Hash for PropertyKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            PropertyKey::Str(s) => s.hash(state),
            PropertyKey::Sym(s) => s.hash(state),
        }
    }
}
impl Eq for PropertyKey {}
impl Clone for PropertyKey {
    fn clone(&self) -> Self {
        match self {
            PropertyKey::Str(s) => PropertyKey::Str(s.clone()),
            PropertyKey::Sym(s) => PropertyKey::Sym(s.clone()),
        }
    }
}

pub struct PropertyDescriptorSetter {
    pub honour_value: bool,
    pub honour_writable: bool,
    pub honour_set: bool,
    pub honour_get: bool,
    pub honour_enumerable: bool,
    pub honour_configurable: bool,
    pub descriptor: PropertyDescriptor,
}
impl PropertyDescriptorSetter {
    pub fn new_from_property_descriptor(desc: PropertyDescriptor) -> Self {
        match desc {
            PropertyDescriptor::Data { .. } => PropertyDescriptorSetter {
                honour_value: true,
                honour_writable: true,
                honour_configurable: true,
                honour_enumerable: true,
                descriptor: desc,
                honour_set: false,
                honour_get: false,
            },
            PropertyDescriptor::Accessor { .. } => PropertyDescriptorSetter {
                honour_set: true,
                honour_get: true,
                honour_configurable: true,
                honour_enumerable: true,
                descriptor: desc,
                honour_value: false,
                honour_writable: false,
            },
        }
    }

    pub fn is_empty(&self) -> bool {
        !self.honour_configurable
            && !self.honour_enumerable
            && !self.honour_get
            && !self.honour_set
            && !self.honour_value
            && !self.honour_writable
    }

    pub fn are_all_fields_set(&self) -> bool {
        if self.descriptor.is_data_descriptor() {
            self.honour_configurable
                && self.honour_enumerable
                && self.honour_value
                && self.honour_writable
        } else {
            self.honour_configurable && self.honour_enumerable && self.honour_get && self.honour_set
        }
    }

    pub fn is_generic_descriptor(&self) -> bool {
        !self.honour_get && !self.honour_set && !self.honour_value && !self.honour_writable
    }
}
impl Clone for PropertyDescriptorSetter {
    fn clone(&self) -> Self {
        PropertyDescriptorSetter {
            honour_set: self.honour_set,
            honour_get: self.honour_get,
            honour_configurable: self.honour_configurable,
            honour_enumerable: self.honour_enumerable,
            honour_value: self.honour_value,
            honour_writable: self.honour_writable,
            descriptor: self.descriptor.clone(),
        }
    }
}

pub struct PropertyDescriptorData {
    pub value: JsValue,
    pub writable: bool,
    pub enumerable: bool,
    pub configurable: bool,
}

pub struct PropertyDescriptorAccessor {
    pub set: Option<JsObjectType>,
    pub get: Option<JsObjectType>,
    pub enumerable: bool,
    pub configurable: bool,
}

pub enum PropertyDescriptor {
    Data(PropertyDescriptorData),
    Accessor(PropertyDescriptorAccessor),
}
impl PropertyDescriptor {
    pub fn new_from_property_descriptor_setter(desc_setter: PropertyDescriptorSetter) -> Self {
        if desc_setter.is_generic_descriptor() {
            if desc_setter.descriptor.is_data_descriptor() {
                PropertyDescriptor::Data(PropertyDescriptorData {
                    value: JsValue::Undefined,
                    writable: false,
                    enumerable: if desc_setter.honour_enumerable {
                        desc_setter.descriptor.is_enumerable()
                    } else {
                        false
                    },
                    configurable: if desc_setter.honour_configurable {
                        desc_setter.descriptor.is_configurable()
                    } else {
                        false
                    },
                })
            } else {
                PropertyDescriptor::Accessor(PropertyDescriptorAccessor {
                    set: None,
                    get: None,
                    enumerable: if desc_setter.honour_enumerable {
                        desc_setter.descriptor.is_enumerable()
                    } else {
                        false
                    },
                    configurable: if desc_setter.honour_configurable {
                        desc_setter.descriptor.is_configurable()
                    } else {
                        false
                    },
                })
            }
        } else if desc_setter.descriptor.is_data_descriptor() {
            if let PropertyDescriptor::Data(PropertyDescriptorData {
                writable, value, ..
            }) = &desc_setter.descriptor
            {
                PropertyDescriptor::Data(PropertyDescriptorData {
                    value: if desc_setter.honour_value {
                        value.clone()
                    } else {
                        JsValue::Undefined
                    },
                    writable: if desc_setter.honour_writable {
                        *writable
                    } else {
                        false
                    },
                    enumerable: if desc_setter.honour_enumerable {
                        desc_setter.descriptor.is_enumerable()
                    } else {
                        false
                    },
                    configurable: if desc_setter.honour_configurable {
                        desc_setter.descriptor.is_configurable()
                    } else {
                        false
                    },
                })
            } else {
                unreachable!()
            }
        } else if desc_setter.descriptor.is_accessor_descriptor() {
            let is_enumerable = desc_setter.descriptor.is_enumerable();
            let is_configurable = desc_setter.descriptor.is_configurable();
            if let PropertyDescriptor::Accessor(PropertyDescriptorAccessor { set, get, .. }) =
                desc_setter.descriptor
            {
                PropertyDescriptor::Accessor(PropertyDescriptorAccessor {
                    set: if desc_setter.honour_set { set } else { None },
                    get: if desc_setter.honour_get { get } else { None },
                    enumerable: if desc_setter.honour_enumerable {
                        is_enumerable
                    } else {
                        false
                    },
                    configurable: if desc_setter.honour_configurable {
                        is_configurable
                    } else {
                        false
                    },
                })
            } else {
                unreachable!()
            }
        } else {
            panic!("Unexpected code path reached in PropertyDescriptor::new_from_property_descriptor_setter")
        }
    }

    pub fn is_enumerable(&self) -> bool {
        match self {
            PropertyDescriptor::Data(PropertyDescriptorData { enumerable, .. }) => *enumerable,
            PropertyDescriptor::Accessor(PropertyDescriptorAccessor { enumerable, .. }) => {
                *enumerable
            }
        }
    }

    pub fn is_configurable(&self) -> bool {
        match self {
            PropertyDescriptor::Data(PropertyDescriptorData { configurable, .. }) => *configurable,
            PropertyDescriptor::Accessor(PropertyDescriptorAccessor { configurable, .. }) => {
                *configurable
            }
        }
    }

    pub fn is_data_descriptor(&self) -> bool {
        match self {
            PropertyDescriptor::Data { .. } => true,
            PropertyDescriptor::Accessor { .. } => false,
        }
    }

    pub fn is_accessor_descriptor(&self) -> bool {
        match self {
            PropertyDescriptor::Data { .. } => false,
            PropertyDescriptor::Accessor { .. } => true,
        }
    }
}
impl PartialEq for PropertyDescriptor {
    fn eq(&self, other: &Self) -> bool {
        match self {
            PropertyDescriptor::Data(PropertyDescriptorData {
                value,
                writable,
                enumerable,
                configurable,
            }) => {
                if let PropertyDescriptor::Data(PropertyDescriptorData {
                    value: other_value,
                    writable: other_writable,
                    enumerable: other_enumerable,
                    configurable: other_configurable,
                }) = other
                {
                    same_value(value, other_value)
                        && writable == other_writable
                        && enumerable == other_enumerable
                        && configurable == other_configurable
                } else {
                    false
                }
            }
            PropertyDescriptor::Accessor(PropertyDescriptorAccessor {
                set: setter,
                get: _getter,
                enumerable,
                configurable,
            }) => {
                if let PropertyDescriptor::Accessor(PropertyDescriptorAccessor {
                    set: other_setter,
                    get: _other_getter,
                    enumerable: other_enumerable,
                    configurable: other_configurable,
                }) = other
                {
                    if enumerable != other_enumerable {
                        false
                    } else if configurable != other_configurable {
                        false
                    } else if setter.is_none() && !other_setter.is_none() {
                        false
                    } else if !setter.is_none() && other_setter.is_none() {
                        false
                    } else if setter.is_none() && other_setter.is_none() {
                        true
                    } else {
                        if let Some(s) = setter {
                            if let Some(os) = other_setter {
                                return same_object(&(**s).borrow(), &(**os).borrow());
                            }
                        }
                        false
                    }
                } else {
                    false
                }
            }
        }
    }
}
impl Clone for PropertyDescriptor {
    fn clone(&self) -> Self {
        match self {
            PropertyDescriptor::Data(d) => PropertyDescriptor::Data(PropertyDescriptorData {
                value: d.value.clone(),
                writable: d.writable,
                enumerable: d.enumerable,
                configurable: d.configurable,
            }),
            PropertyDescriptor::Accessor(d) => {
                PropertyDescriptor::Accessor(PropertyDescriptorAccessor {
                    set: match &d.set {
                        None => None,
                        Some(s) => Some(s.clone()),
                    },
                    get: match &d.get {
                        None => None,
                        Some(g) => Some(g.clone()),
                    },
                    enumerable: d.enumerable,
                    configurable: d.configurable,
                })
            }
        }
    }
}
