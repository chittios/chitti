//! The kernel's own HTML parser, mounted by path — so the "ours" column is the
//! code that ships, not a stand-in.

#[path = "../../../../kernel/src/browser/html.rs"]
pub mod html;
#[path = "../../../../kernel/src/browser/elements.rs"]
pub mod elements;
