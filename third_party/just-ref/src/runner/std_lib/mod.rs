//! Standard library built-in objects.
//!
//! This module contains implementations of JavaScript built-in objects
//! like console, Object, Array, String, Number, Math, JSON, and Error types.

pub mod core;
pub mod console;
pub mod object;
pub mod array;
pub mod string;
pub mod number;
pub mod math;
pub mod json;
pub mod error;
pub mod collections;
pub mod date;
pub mod regexp;
pub mod promise;
pub mod proxy;
pub mod globals;

pub use core::register_core_builtins;
