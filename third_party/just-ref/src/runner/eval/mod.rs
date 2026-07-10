//! Evaluation module for executing JavaScript AST.
//!
//! This module contains the core evaluation logic for the JavaScript interpreter.

pub mod types;
pub mod expression;
pub mod statement;
pub mod function;

pub use types::{Completion, CompletionType, Reference, ReferenceBase};
