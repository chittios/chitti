//! Shared package tools for notes/paint/slides/games + default-OS UI apps.
//! All app logic lives here; host only provides chitti.* imports.
//! Each agent package ships a copy of this `tools.wasm` and declares its
//! toolset; package_ui routes start/click/key/tick by app name.

#![no_std]
#![no_main]

extern crate alloc;

mod activity;
mod archive;
mod breakout;
mod calc;
mod calendar;
mod clock;
mod console;
mod contacts;
mod dict;
mod diff;
mod endscreen;
mod fa;
mod files;
mod gallery;
mod game2048;
mod guest;
mod hex;
mod maps;
mod mines;
mod notes;
mod paint;
mod radio;
mod sandbox;
mod settings;
mod sheets;
mod slides;
mod snake;
mod synth;
mod tetris;
mod weather;
mod writer;

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

// --- existing apps ----------------------------------------------------------
tool!(notes_start, notes::start);
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

tool!(synth_start, synth::start);
tool!(synth_tone, synth::tone);
tool!(synth_beep, synth::beep);
tool!(synth_stop, synth::stop);
tool!(synth_status, synth::status);

// --- default-OS UI apps -----------------------------------------------------
tool!(calc_start, calc::start);
tool!(calc_eval, calc::eval);
tool!(calc_status, calc::status);

tool!(clock_start, clock::start);
tool!(clock_set_timer, clock::set_timer);
tool!(clock_status, clock::status);

tool!(files_start, files::start);
tool!(files_list, files::list);
tool!(files_get, files::get);
tool!(files_set, files::set);
tool!(files_remove, files::remove);
tool!(files_status, files::status);

tool!(gallery_start, gallery::start);
tool!(gallery_list, gallery::list);
tool!(gallery_get, gallery::get);
tool!(gallery_set, gallery::set);
tool!(gallery_status, gallery::status);

tool!(sheets_start, sheets::start);
tool!(sheets_get, sheets::get);
tool!(sheets_set, sheets::set);
tool!(sheets_status, sheets::status);

tool!(calendar_start, calendar::start);
tool!(calendar_add, calendar::add);
tool!(calendar_list, calendar::list);
tool!(calendar_status, calendar::status);

tool!(contacts_start, contacts::start);
tool!(contacts_list, contacts::list);
tool!(contacts_get, contacts::get);
tool!(contacts_set, contacts::set);
tool!(contacts_remove, contacts::remove);
tool!(contacts_status, contacts::status);

tool!(writer_start, writer::start);
tool!(writer_get, writer::get);
tool!(writer_set, writer::set);
tool!(writer_status, writer::status);

tool!(archive_start, archive::start);
tool!(archive_pack, archive::pack);
tool!(archive_unpack, archive::unpack);
tool!(archive_list, archive::list);
tool!(archive_status, archive::status);

tool!(hex_start, hex::start);
tool!(hex_open, hex::open);
tool!(hex_dump, hex::dump);
tool!(hex_status, hex::status);

tool!(game2048_start, game2048::start);
tool!(game2048_status, game2048::status);

tool!(activity_start, activity::start);
tool!(activity_set, activity::set);
tool!(activity_status, activity::status);

tool!(weather_start, weather::start);
tool!(weather_set, weather::set);
tool!(weather_status, weather::status);

tool!(settings_start, settings::start);
tool!(settings_get, settings::get);
tool!(settings_set, settings::set);
tool!(settings_status, settings::status);

tool!(dict_start, dict::start);
tool!(dict_lookup, dict::lookup_tool);
tool!(dict_set, dict::define);
tool!(dict_status, dict::status);

tool!(diff_start, diff::start);
tool!(diff_set, diff::set);
tool!(diff_status, diff::status);

tool!(breakout_start, breakout::start);
tool!(breakout_status, breakout::status);

tool!(tetris_start, tetris::start);
tool!(tetris_status, tetris::status);

tool!(console_start, console::start);
tool!(console_log, console::log);
tool!(console_list, console::list);
tool!(console_status, console::status);

tool!(maps_start, maps::start);
tool!(maps_set, maps::set);
tool!(maps_list, maps::list);
tool!(maps_status, maps::status);

tool!(radio_start, radio::start);
tool!(radio_tune, radio::tune);
tool!(radio_status, radio::status);

// Agent package name is sandbox-lab; wasm exports use sandbox_ prefix.
tool!(sandbox_start, sandbox::start);
tool!(sandbox_home_write, sandbox::home_write);
tool!(sandbox_try_escape, sandbox::try_escape);
tool!(sandbox_child, sandbox::child_toggle);
tool!(sandbox_list, sandbox::list_home);
tool!(sandbox_get, sandbox::get);
tool!(sandbox_status, sandbox::status);

/// Auto-tick from package_ui.
#[no_mangle]
pub extern "C" fn tick(args_ptr: i32, args_len: i32) -> i64 {
    let raw = match core::str::from_utf8(unpack_args(args_ptr, args_len)) {
        Ok(s) => s,
        Err(_) => return result_string("error:bad utf-8"),
    };
    let app = json_str(raw, "app").unwrap_or_default();
    let out = match app.as_str() {
        "snake" | "" => snake::tick_once(raw),
        "clock" => clock::tick(raw),
        "activity" => activity::tick(raw),
        "breakout" => breakout::tick(raw),
        "tetris" => tetris::tick(raw),
        "radio" => radio::tick(raw),
        // Keep confetti animating on end screens even when the game is idle.
        "minesweeper" | "mines" => mines::tick_anim(raw),
        "game2048" => game2048::tick_anim(raw),
        _ => alloc::string::String::from("ok"),
    };
    result_string(&out)
}

/// Surface click from package_ui: `{"app":..,"x":..,"y":..}`.
#[no_mangle]
pub extern "C" fn on_click(args_ptr: i32, args_len: i32) -> i64 {
    let raw = match core::str::from_utf8(unpack_args(args_ptr, args_len)) {
        Ok(s) => s,
        Err(_) => return result_string("error:bad utf-8"),
    };
    let app = json_str(raw, "app").unwrap_or_default();
    let x = guest::json_i32(raw, "x", -1);
    let y = guest::json_i32(raw, "y", -1);
    let out = match app.as_str() {
        "minesweeper" => mines::on_click(x, y),
        "paint" => paint::on_click(x, y),
        "slides" => slides::on_click(x, y),
        "synth" => synth::on_click(x, y),
        "calc" => calc::on_click(x, y),
        "clock" => clock::on_click(x, y),
        "files" => files::on_click(x, y),
        "gallery" => gallery::on_click(x, y),
        "sheets" => sheets::on_click(x, y),
        "calendar" => calendar::on_click(x, y),
        "contacts" => contacts::on_click(x, y),
        "writer" => writer::on_click(x, y),
        "archive" => archive::on_click(x, y),
        "hex" => hex::on_click(x, y),
        "game2048" => game2048::on_click(x, y),
        "activity" => activity::on_click(x, y),
        "weather" => weather::on_click(x, y),
        "settings" => settings::on_click(x, y),
        "dict" => dict::on_click(x, y),
        "diff" => diff::on_click(x, y),
        "breakout" => breakout::on_click(x, y),
        "tetris" => tetris::on_click(x, y),
        "snake" => snake::on_click(x, y),
        "console" => console::on_click(x, y),
        "maps" => maps::on_click(x, y),
        "radio" => radio::on_click(x, y),
        "notes" => notes::on_click(x, y),
        "sandbox-lab" | "sandbox" => sandbox::on_click(x, y),
        _ => alloc::string::String::from("ok"),
    };
    result_string(&out)
}

/// Key from package_ui: `{"app":..,"key":"…"}`.
#[no_mangle]
pub extern "C" fn on_key(args_ptr: i32, args_len: i32) -> i64 {
    let raw = match core::str::from_utf8(unpack_args(args_ptr, args_len)) {
        Ok(s) => s,
        Err(_) => return result_string("error:bad utf-8"),
    };
    let app = json_str(raw, "app").unwrap_or_default();
    let key = json_str(raw, "key").unwrap_or_default();
    let out = match app.as_str() {
        "minesweeper" => mines::on_key(&key),
        "snake" => snake::on_key(&key),
        // note: snake also has on_click for endscreen restart
        "paint" => paint::on_key(&key),
        "slides" => slides::on_key(&key),
        "synth" => synth::on_key(&key),
        "calc" => calc::on_key(&key),
        "clock" => clock::on_key(&key),
        "files" => files::on_key(&key),
        "gallery" => gallery::on_key(&key),
        "sheets" => sheets::on_key(&key),
        "calendar" => calendar::on_key(&key),
        "contacts" => contacts::on_key(&key),
        "writer" => writer::on_key(&key),
        "archive" => archive::on_key(&key),
        "hex" => hex::on_key(&key),
        "game2048" => game2048::on_key(&key),
        "activity" => activity::on_key(&key),
        "weather" => weather::on_key(&key),
        "settings" => settings::on_key(&key),
        "dict" => dict::on_key(&key),
        "diff" => diff::on_key(&key),
        "breakout" => breakout::on_key(&key),
        "tetris" => tetris::on_key(&key),
        "console" => console::on_key(&key),
        "maps" => maps::on_key(&key),
        "radio" => radio::on_key(&key),
        "notes" => notes::on_key(&key),
        "sandbox-lab" | "sandbox" => sandbox::on_key(&key),
        _ => alloc::string::String::from("ok"),
    };
    result_string(&out)
}
