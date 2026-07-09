//! Package tools for notes, paint, slides, minesweeper, snake, synth.
//! All app logic lives here; host only provides chitti.* imports.

#![no_std]
#![no_main]

extern crate alloc;

mod guest;
mod mines;
mod notes;
mod paint;
mod slides;
mod snake;
mod synth;

use guest::{json_str, result_string, unpack_args};

#[no_mangle]
pub extern "C" fn chitti_alloc(size: i32) -> i32 {
    guest::chitti_alloc(size)
}

macro_rules! tool {
    ($name:ident, $body:expr) => {
        #[no_mangle]
        pub extern "C" fn $name(args_ptr: i32, args_len: i32) -> i64 {
            let raw = match core::str::from_utf8(unpack_args(args_ptr, args_len)) {
                Ok(s) => s,
                Err(_) => return result_string("error:bad utf-8"),
            };
            result_string(&$body(raw))
        }
    };
}

tool!(notes_list, notes::list);
tool!(notes_get, notes::get);
tool!(notes_set, notes::set);
tool!(notes_remove, notes::remove);

tool!(paint_start, paint::start);
tool!(paint_clear, paint::clear);
tool!(paint_rect, paint::rect);
tool!(paint_line, paint::line);
tool!(paint_pixel, paint::pixel);
tool!(paint_draw, paint::draw_ops);
tool!(paint_status, paint::status);

tool!(slides_start, slides::start);
tool!(slides_next, slides::next);
tool!(slides_prev, slides::prev);
tool!(slides_goto, slides::goto);
tool!(slides_status, slides::status);

tool!(mines_start, mines::start);
tool!(mines_click, mines::click);
tool!(mines_flag, mines::flag);
tool!(mines_status, mines::status);

tool!(snake_start, snake::start);
tool!(snake_dir, snake::dir);
tool!(snake_tick, snake::tick_once);
tool!(snake_status, snake::status);

tool!(synth_tone, synth::tone);
tool!(synth_beep, synth::beep);
tool!(synth_stop, synth::stop);
tool!(synth_status, synth::status);

/// Auto-tick from package_ui (snake).
#[no_mangle]
pub extern "C" fn tick(args_ptr: i32, args_len: i32) -> i64 {
    let raw = match core::str::from_utf8(unpack_args(args_ptr, args_len)) {
        Ok(s) => s,
        Err(_) => return result_string("error:bad utf-8"),
    };
    let app = json_str(raw, "app").unwrap_or_default();
    if app == "snake" || app.is_empty() {
        result_string(&snake::tick_once(raw))
    } else {
        result_string("ok")
    }
}
