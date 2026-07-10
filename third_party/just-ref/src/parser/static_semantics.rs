// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::parser::ast::IdentifierData;

pub struct TriStateBool {
    flag: Option<bool>,
}
impl TriStateBool {
    pub(crate) fn new_unset() -> Self {
        TriStateBool { flag: None }
    }

    pub(crate) fn is_true(&self) -> bool {
        self.flag.is_some() && self.flag.unwrap()
    }

    pub(crate) fn is_false(&self) -> bool {
        self.flag.is_some() && !self.flag.unwrap()
    }

    pub(crate) fn make_true(&mut self) {
        self.flag = Some(true);
    }

    pub(crate) fn merge<F>(&mut self, other: TriStateBool, conflict_resolver: F)
    where
        F: Fn(bool, bool) -> bool,
    {
        self.flag = if let Some(f) = self.flag {
            if let Some(f2) = other.flag {
                if f == f2 {
                    Some(f)
                } else {
                    Some(conflict_resolver(f, f2))
                }
            } else {
                Some(false)
            }
        } else {
            other.flag
        };
    }
}
impl PartialEq for TriStateBool {
    fn eq(&self, other: &Self) -> bool {
        if let Some(f) = &self.flag {
            if let Some(f2) = &other.flag {
                *f == *f2
            } else {
                false
            }
        } else {
            if let Some(_) = &other.flag {
                false
            } else {
                true
            }
        }
    }
}

pub struct Semantics {
    pub(crate) bound_names: Vec<IdentifierData>,
    pub(crate) lexically_declared_names: Vec<IdentifierData>,
    pub(crate) var_declared_names: Vec<IdentifierData>,
    pub(crate) top_level_lexically_declared_names: Vec<IdentifierData>,
    pub(crate) top_level_var_declared_names: Vec<IdentifierData>,
    pub(crate) has_direct_super: TriStateBool,
    pub(crate) is_valid_simple_assignment_target: TriStateBool,
    pub(crate) contains_yield_expression: TriStateBool,
    pub(crate) contains_unpaired_continue: TriStateBool,
    pub(crate) contains_unpaired_break: TriStateBool,
    pub(crate) contains_super_property: TriStateBool,
    pub(crate) contains_super_call: TriStateBool,
}
impl Semantics {
    pub(crate) fn new_empty() -> Self {
        Semantics {
            bound_names: vec![],
            lexically_declared_names: vec![],
            var_declared_names: vec![],
            top_level_lexically_declared_names: vec![],
            top_level_var_declared_names: vec![],
            has_direct_super: TriStateBool::new_unset(),
            is_valid_simple_assignment_target: TriStateBool::new_unset(),
            contains_yield_expression: TriStateBool::new_unset(),
            contains_unpaired_continue: TriStateBool::new_unset(),
            contains_unpaired_break: TriStateBool::new_unset(),
            contains_super_property: TriStateBool::new_unset(),
            contains_super_call: TriStateBool::new_unset(),
        }
    }

    pub(crate) fn merge(&mut self, mut other: Self) -> &mut Self {
        let panic_on_resolve = |f, f2| {
            panic!(
                "Not sure how to resolve this conflict: (f, f2) = ({},{})",
                f, f2
            );
        };
        let true_wins = |f, f2| f || f2;

        self.bound_names.append(&mut other.bound_names);
        self.lexically_declared_names
            .append(&mut other.lexically_declared_names);
        self.has_direct_super
            .merge(other.has_direct_super, panic_on_resolve);
        self.is_valid_simple_assignment_target
            .merge(other.is_valid_simple_assignment_target, panic_on_resolve);
        self.contains_yield_expression
            .merge(other.contains_yield_expression, true_wins);
        self.contains_unpaired_continue
            .merge(other.contains_unpaired_continue, true_wins);
        self.contains_unpaired_break
            .merge(other.contains_unpaired_break, true_wins);
        self.contains_super_property
            .merge(other.contains_super_property, true_wins);
        self.contains_super_call
            .merge(other.contains_super_call, true_wins);
        self
    }
}

