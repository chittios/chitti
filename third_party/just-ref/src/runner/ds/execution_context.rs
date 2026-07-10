// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::lex_env::JsLexEnvironmentType;
use crate::runner::ds::object::JsObjectType;
use crate::runner::ds::realm::JsCodeRealmType;

pub struct ExecutionContext {
    pub function: Option<JsObjectType>,
    pub realm: JsCodeRealmType,
    pub lex_env: JsLexEnvironmentType,
    pub var_env: JsLexEnvironmentType,
}

pub struct ExecutionContextStack {
    stack: Vec<ExecutionContext>,
}
impl ExecutionContextStack {
    pub fn new() -> Self {
        ExecutionContextStack { stack: Vec::new() }
    }

    pub fn get_running_execution_ctx(&self) -> Option<&ExecutionContext> {
        self.stack.get(self.stack.len() - 1)
    }

    pub fn get_running_execution_ctx_mut(&mut self) -> Option<&mut ExecutionContext> {
        let stack = &mut self.stack;
        let stack_len = stack.len();
        stack.get_mut(stack_len - 1)
    }

    pub fn pop_running_execution_ctx(&mut self) -> Option<ExecutionContext> {
        self.stack.pop()
    }

    pub fn push_execution_ctx(&mut self, ctx: ExecutionContext) {
        self.stack.push(ctx)
    }
}
