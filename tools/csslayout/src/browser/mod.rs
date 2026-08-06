//! The kernel's own browser modules, mounted by path.
//!
//! Nothing here is a copy: these ARE `kernel/src/browser/*.rs`. That is the
//! whole point — a layout fix verified in this harness is the fix that ships,
//! and there is no second engine to drift.

#[path = "../../../../kernel/src/browser/html.rs"]
pub mod html;
#[path = "../../../../kernel/src/browser/css.rs"]
pub mod css;
#[path = "../../../../kernel/src/browser/elements.rs"]
pub mod elements;
#[path = "../../../../kernel/src/browser/flex.rs"]
pub mod flex;
#[path = "../../../../kernel/src/browser/layout.rs"]
pub mod layout;
// Reached from layout: `<canvas>` sizing and `form_fields`.
#[path = "../../../../kernel/src/browser/canvas.rs"]
pub mod canvas;
#[path = "../../../../kernel/src/browser/form.rs"]
pub mod form;
#[path = "../../../../kernel/src/browser/url.rs"]
pub mod url;

/// `layout` stamps DOM indices through `crate::browser::js`; the harness runs
/// no JS, and layout only needs the stamp to exist.
pub mod js {
    pub fn stamp_elem_indices(_root: &mut super::html::Node) {}
}
