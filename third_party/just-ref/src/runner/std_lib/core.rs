//! Core built-ins registration.
//!
//! This module provides the function to register all core built-in objects
//! with the BuiltInRegistry.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::plugin::registry::BuiltInRegistry;

use super::console;
use super::object;
use super::array;
use super::string;
use super::number;
use super::math;
use super::json;
use super::error;
use super::collections;
use super::date;
use super::regexp;
use super::promise;
use super::proxy;

/// Register all core built-in objects with the registry.
pub fn register_core_builtins(registry: &mut BuiltInRegistry) {
    // Register in order (some may depend on Object)
    object::register(registry);
    array::register(registry);
    string::register(registry);
    number::register(registry);
    math::register(registry);
    json::register(registry);
    error::register(registry);
    collections::register(registry);
    date::register(registry);
    regexp::register(registry);
    promise::register(registry);
    proxy::register(registry);
    console::register(registry);
}
