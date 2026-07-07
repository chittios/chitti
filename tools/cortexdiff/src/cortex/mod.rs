//! The kernel's cortex modules, mounted verbatim from `kernel/src/cortex/`.
//! This directory exists only so the `#[path]` mounts resolve relative to a
//! real `src/cortex/`, keeping the kernel modules' absolute `crate::cortex::*`
//! self-references valid inside the harness crate.

#[path = "../../../../kernel/src/cortex/gguf.rs"]
pub mod gguf;
#[path = "../../../../kernel/src/cortex/iq_tables.rs"]
pub mod iq_tables;
#[path = "../../../../kernel/src/cortex/tensor.rs"]
pub mod tensor;
#[path = "../../../../kernel/src/cortex/tokenizer.rs"]
pub mod tokenizer;
#[path = "../../../../kernel/src/cortex/model.rs"]
pub mod model;
