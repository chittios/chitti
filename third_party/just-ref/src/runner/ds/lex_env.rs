// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::env_record::EnvironmentRecordType;
use core::cell::RefCell;
use alloc::rc::Rc;

pub type JsLexEnvironmentType = Rc<RefCell<LexEnvironment>>;

pub struct LexEnvironment {
    pub inner: Box<EnvironmentRecordType>,
    pub outer: Option<JsLexEnvironmentType>,
}
