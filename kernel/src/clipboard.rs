//! A single global clipboard shared by the editor (yank/paste) and the shell
//! (paste into the input line). `linewise` mirrors vim's register kind: a
//! line-wise yank (`yy`/`dd`) pastes as whole lines, a char-wise yank (visual
//! `y`) pastes inline.

use crate::mm::Locked;
use alloc::string::String;

struct Clip {
    text: String,
    linewise: bool,
}

static CLIP: Locked<Option<Clip>> = Locked::new(None);

/// Replace the clipboard contents.
pub fn set(text: String, linewise: bool) {
    CLIP.with(|c| *c = Some(Clip { text, linewise }));
}

/// The clipboard `(text, linewise)`, or `None` if empty.
pub fn get() -> Option<(String, bool)> {
    CLIP.with(|c| c.as_ref().map(|x| (x.text.clone(), x.linewise)))
}
