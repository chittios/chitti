// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::env_record::new_global_environment;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::lex_env::JsLexEnvironmentType;
use crate::runner::ds::object::{object_create, JsObjectType, ObjectType};
use crate::runner::ds::object_property::{
    PropertyDescriptor, PropertyDescriptorData, PropertyDescriptorSetter, PropertyKey,
};
use crate::runner::ds::operations::object::define_property_or_throw;
use crate::runner::ds::value::{JsNumberType, JsValue};
use core::cell::RefCell;
use hashbrown::HashMap;
use core::hash::{Hash, Hasher};
use alloc::rc::Rc;

pub enum WellKnownIntrinsics {
    Array,
    ArrayBuffer,
    ArrayBufferPrototype,
    ArrayIteratorPrototype,
    ArrayPrototype,
    ArrayProtoValues,
    Boolean,
    BooleanPrototype,
    Date,
    DatePrototype,
    DecodeURI,
    DecodeURIComponent,
    EncodeURI,
    EncodeURIComponent,
    Error,
    ErrorPrototype,
    Function,
    FunctionPrototype,
    Generator,
    GeneratorFunction,
    GeneratorPrototype,
    IsFinite,
    IsNaN,
    IteratorPrototype,
    JSON,
    Map,
    MapIteratorPrototype,
    MapPrototype,
    Math,
    Number,
    NumberPrototype,
    Object,
    ObjectPrototype,
    ObjProtoToString,
    ParseFloat,
    ParseInt,
    Promise,
    PromisePrototype,
    RangeError,
    RangeErrorPrototype,
    ReferenceError,
    ReferenceErrorPrototype,
    RegExp,
    RegExpPrototype,
    Set,
    SetIteratorPrototype,
    SetPrototype,
    String,
    StringIteratorPrototype,
    StringPrototype,
    Symbol,
    SymbolPrototype,
    SyntaxError,
    SyntaxErrorPrototype,
    ThrowTypeError,
    TypeError,
    TypeErrorPrototype,
    URIError,
    URIErrorPrototype,
}
impl WellKnownIntrinsics {
    pub fn value(&self) -> JsObjectType {
        match self {
            WellKnownIntrinsics::ObjectPrototype => Rc::new(RefCell::new(ObjectType::Ordinary(
                Box::new(object_create(None)),
            ))),
            _ => todo!(), // WellKnownIntrinsics::ThrowTypeError => {}
                          // WellKnownIntrinsics::Array => {}
                          // WellKnownIntrinsics::ArrayBuffer => {}
                          // WellKnownIntrinsics::ArrayBufferPrototype => {}
                          // WellKnownIntrinsics::ArrayIteratorPrototype => {}
                          // WellKnownIntrinsics::ArrayPrototype => {}
                          // WellKnownIntrinsics::ArrayProtoValues => {}
                          // WellKnownIntrinsics::Boolean => {}
                          // WellKnownIntrinsics::BooleanPrototype => {}
                          // WellKnownIntrinsics::Date => {}
                          // WellKnownIntrinsics::DatePrototype => {}
                          // WellKnownIntrinsics::DecodeURI => {}
                          // WellKnownIntrinsics::DecodeURIComponent => {}
                          // WellKnownIntrinsics::EncodeURI => {}
                          // WellKnownIntrinsics::EncodeURIComponent => {}
                          // WellKnownIntrinsics::Error => {}
                          // WellKnownIntrinsics::ErrorPrototype => {}
                          // WellKnownIntrinsics::Function => {}
                          // WellKnownIntrinsics::FunctionPrototype => {}
                          // WellKnownIntrinsics::Generator => {}
                          // WellKnownIntrinsics::GeneratorFunction => {}
                          // WellKnownIntrinsics::GeneratorPrototype => {}
                          // WellKnownIntrinsics::IsFinite => {}
                          // WellKnownIntrinsics::IsNaN => {}
                          // WellKnownIntrinsics::IteratorPrototype => {}
                          // WellKnownIntrinsics::JSON => {}
                          // WellKnownIntrinsics::Map => {}
                          // WellKnownIntrinsics::MapIteratorPrototype => {}
                          // WellKnownIntrinsics::MapPrototype => {}
                          // WellKnownIntrinsics::Math => {}
                          // WellKnownIntrinsics::Number => {}
                          // WellKnownIntrinsics::NumberPrototype => {}
                          // WellKnownIntrinsics::Object => {}
                          // WellKnownIntrinsics::ObjProtoToString => {}
                          // WellKnownIntrinsics::ParseFloat => {}
                          // WellKnownIntrinsics::ParseInt => {}
                          // WellKnownIntrinsics::Promise => {}
                          // WellKnownIntrinsics::PromisePrototype => {}
                          // WellKnownIntrinsics::RangeError => {}
                          // WellKnownIntrinsics::RangeErrorPrototype => {}
                          // WellKnownIntrinsics::ReferenceError => {}
                          // WellKnownIntrinsics::ReferenceErrorPrototype => {}
                          // WellKnownIntrinsics::RegExp => {}
                          // WellKnownIntrinsics::RegExpPrototype => {}
                          // WellKnownIntrinsics::Set => {}
                          // WellKnownIntrinsics::SetIteratorPrototype => {}
                          // WellKnownIntrinsics::SetPrototype => {}
                          // WellKnownIntrinsics::String => {}
                          // WellKnownIntrinsics::StringIteratorPrototype => {}
                          // WellKnownIntrinsics::StringPrototype => {}
                          // WellKnownIntrinsics::Symbol => {}
                          // WellKnownIntrinsics::SymbolPrototype => {}
                          // WellKnownIntrinsics::SyntaxError => {}
                          // WellKnownIntrinsics::SyntaxErrorPrototype => {}
                          // WellKnownIntrinsics::TypeError => {}
                          // WellKnownIntrinsics::TypeErrorPrototype => {}
                          // WellKnownIntrinsics::URIError => {}
                          // WellKnownIntrinsics::URIErrorPrototype => {}
        }
    }
}
impl PartialEq for WellKnownIntrinsics {
    fn eq(&self, _other: &Self) -> bool {
        todo!()
    }
}
impl Eq for WellKnownIntrinsics {}
impl Hash for WellKnownIntrinsics {
    fn hash<H: Hasher>(&self, _state: &mut H) {
        todo!()
    }
}

pub type JsCodeRealmType = Rc<RefCell<CodeRealm>>;

pub struct CodeRealm {
    pub intrinsics: HashMap<WellKnownIntrinsics, JsObjectType>,
    pub global_this: Option<JsObjectType>,
    pub global_env: Option<JsLexEnvironmentType>,
    // template_map
}
impl CodeRealm {
    pub fn new() -> Self {
        CodeRealm {
            intrinsics: create_intrinsics(),
            global_this: None,
            global_env: None,
        }
    }

    pub fn get_intrinsics_value(&self, int_name: &WellKnownIntrinsics) -> &JsObjectType {
        self.intrinsics.get(int_name).unwrap()
    }
}

fn insert_into_realm(
    map: &mut HashMap<WellKnownIntrinsics, JsObjectType>,
    int: WellKnownIntrinsics,
) {
    let v = int.value();
    map.insert(int, v);
}

pub fn create_intrinsics() -> HashMap<WellKnownIntrinsics, JsObjectType> {
    let mut intrinsics = HashMap::new();
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::ObjectPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::Array);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::ArrayBuffer);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::ArrayBufferPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::ArrayIteratorPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::ArrayPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::ArrayProtoValues);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::Boolean);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::BooleanPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::Date);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::DatePrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::DecodeURI);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::DecodeURIComponent);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::EncodeURI);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::EncodeURIComponent);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::Error);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::ErrorPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::Function);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::FunctionPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::Generator);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::GeneratorFunction);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::GeneratorPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::IsFinite);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::IsNaN);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::IteratorPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::JSON);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::Map);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::MapIteratorPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::MapPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::Math);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::Number);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::NumberPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::Object);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::ObjProtoToString);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::ParseFloat);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::ParseInt);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::Promise);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::PromisePrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::RangeError);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::RangeErrorPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::ReferenceError);
    insert_into_realm(
        &mut intrinsics,
        WellKnownIntrinsics::ReferenceErrorPrototype,
    );
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::RegExp);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::RegExpPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::Set);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::SetIteratorPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::SetPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::String);
    insert_into_realm(
        &mut intrinsics,
        WellKnownIntrinsics::StringIteratorPrototype,
    );
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::StringPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::Symbol);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::SymbolPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::SyntaxError);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::SyntaxErrorPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::ThrowTypeError);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::TypeError);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::TypeErrorPrototype);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::URIError);
    insert_into_realm(&mut intrinsics, WellKnownIntrinsics::URIErrorPrototype);

    intrinsics
}

pub fn set_realm_global_object(r: &mut CodeRealm, global_obj: Option<JsObjectType>) {
    let (final_global_obj, inner_global_obj) = if let Some(g) = global_obj {
        let inner = g.clone();
        (Some(g), inner)
    } else {
        let intrinsic = r.get_intrinsics_value(&WellKnownIntrinsics::ObjectPrototype).clone();
        (Some(intrinsic.clone()), intrinsic)
    };
    r.global_this = final_global_obj;
    r.global_env = Some(new_global_environment(inner_global_obj));
}

pub fn set_default_global_bindings(r: &mut CodeRealm) -> Result<(), JErrorType> {
    // Get intrinsic values first to avoid borrow conflicts
    let is_finite_intrinsic = r.get_intrinsics_value(&WellKnownIntrinsics::IsFinite).clone();

    if let Some(global_obj) = &mut r.global_this {
        define_property_or_throw(
            (**global_obj).borrow_mut().as_js_object_mut(),
            PropertyKey::Str("Infinity".to_string()),
            PropertyDescriptorSetter::new_from_property_descriptor(PropertyDescriptor::Data(
                PropertyDescriptorData {
                    value: JsValue::Number(JsNumberType::PositiveInfinity),
                    writable: false,
                    enumerable: false,
                    configurable: false,
                },
            )),
        )?;
        define_property_or_throw(
            (**global_obj).borrow_mut().as_js_object_mut(),
            PropertyKey::Str("NaN".to_string()),
            PropertyDescriptorSetter::new_from_property_descriptor(PropertyDescriptor::Data(
                PropertyDescriptorData {
                    value: JsValue::Number(JsNumberType::NaN),
                    writable: false,
                    enumerable: false,
                    configurable: false,
                },
            )),
        )?;
        define_property_or_throw(
            (**global_obj).borrow_mut().as_js_object_mut(),
            PropertyKey::Str("undefined".to_string()),
            PropertyDescriptorSetter::new_from_property_descriptor(PropertyDescriptor::Data(
                PropertyDescriptorData {
                    value: JsValue::Undefined,
                    writable: false,
                    enumerable: false,
                    configurable: false,
                },
            )),
        )?;
        define_property_or_throw(
            (**global_obj).borrow_mut().as_js_object_mut(),
            PropertyKey::Str("isFinite".to_string()),
            PropertyDescriptorSetter::new_from_property_descriptor(PropertyDescriptor::Data(
                PropertyDescriptorData {
                    value: JsValue::Object(is_finite_intrinsic),
                    writable: false,
                    enumerable: false,
                    configurable: false,
                },
            )),
        )?;
        todo!()
    }
    Ok(())
}
