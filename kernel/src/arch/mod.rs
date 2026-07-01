//! Architecture-specific code lives entirely under `arch/<name>/`, so a
//! future port (e.g. RISC-V) only has to add a sibling module here.

#[cfg(target_arch = "x86_64")]
pub mod x86_64;
