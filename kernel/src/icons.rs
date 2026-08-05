//! **System icons** — Font Awesome 7 Free Solid codepoints.
//!
//! The face is bundled as `assets/fonts/FontAwesome7Free-Solid-900.otf` and
//! registered first in the TTF fallback chain ([`crate::font_ttf`]), so UI text
//! that uses these Private-Use-Area scalars paints FA glyphs rather than tofu.
//!
//! Font Awesome Free fonts are SIL OFL 1.1; see THIRDPARTY-LICENSES.md.
//! Only **Free Solid** is vendored (no brands pack — brand icons need a separate
//! face and have stricter trademark rules).
//!
//! Codepoints match Font Awesome 7 Free Solid (desktop OTF). Names follow the
//! FA slug in SCREAMING_SNAKE. Apps that cannot import this module (wasm guests)
//! mirror the same literals in `tools/apps-wasm/src/fa.rs`.

use alloc::string::ToString;

/// Font Awesome 7 Free Solid Private Use Area icons used by the shell / apps.
pub mod fa {
    // --- chrome / status bar ---
    /// keyboard
    pub const KEYBOARD: char = '\u{f11c}';
    /// computer-mouse
    pub const MOUSE: char = '\u{f8cc}';
    /// wifi
    pub const WIFI: char = '\u{f1eb}';
    /// network-wired
    pub const NETWORK: char = '\u{f6ff}';
    /// globe
    pub const GLOBE: char = '\u{f0ac}';
    /// circle (active pulse / solid dot)
    pub const CIRCLE: char = '\u{f111}';
    /// circle-dot
    pub const CIRCLE_DOT: char = '\u{f192}';
    /// grip-lines-vertical — the connector down a tool call's left edge
    pub const GRIP_LINES_VERTICAL: char = '\u{f7a5}';
    /// microchip — CPU / activity
    pub const MICROCHIP: char = '\u{f2db}';
    /// memory
    pub const MEMORY: char = '\u{f538}';
    /// battery-full
    pub const BATTERY: char = '\u{f240}';
    /// clock
    pub const CLOCK: char = '\u{f017}';
    /// hard-drive
    pub const HARD_DRIVE: char = '\u{f0a0}';
    /// server
    pub const SERVER: char = '\u{f233}';
    /// bolt
    pub const BOLT: char = '\u{f0e7}';
    /// power-off
    pub const POWER_OFF: char = '\u{f011}';

    // --- files / documents ---
    /// folder
    pub const FOLDER: char = '\u{f07b}';
    /// folder-open
    pub const FOLDER_OPEN: char = '\u{f07c}';
    /// file
    pub const FILE: char = '\u{f15b}';
    /// file-lines
    pub const FILE_LINES: char = '\u{f15c}';
    /// book
    pub const BOOK: char = '\u{f02d}';
    /// book-open
    pub const BOOK_OPEN: char = '\u{f518}';
    /// box-archive
    pub const BOX_ARCHIVE: char = '\u{f187}';
    /// download
    pub const DOWNLOAD: char = '\u{f019}';
    /// pen
    pub const PEN: char = '\u{f304}';
    /// newspaper
    pub const NEWSPAPER: char = '\u{f1ea}';

    // --- settings / tools ---
    /// gear
    pub const GEAR: char = '\u{f013}';
    /// sliders
    pub const SLIDERS: char = '\u{f1de}';
    /// palette
    pub const PALETTE: char = '\u{f53f}';
    /// shield-halved
    pub const SHIELD: char = '\u{f3ed}';
    /// screwdriver-wrench
    pub const WRENCH: char = '\u{f7d9}';
    /// key
    pub const KEY: char = '\u{f084}';
    /// lock
    pub const LOCK: char = '\u{f023}';
    /// user-secret
    pub const USER_SECRET: char = '\u{f21b}';
    /// user
    pub const USER: char = '\u{f007}';
    /// users
    pub const USERS: char = '\u{f0c0}';
    /// address-book
    pub const ADDRESS_BOOK: char = '\u{f2b9}';
    /// envelope
    pub const ENVELOPE: char = '\u{f0e0}';

    // --- apps / media ---
    /// robot — model / agent default
    pub const ROBOT: char = '\u{f544}';
    /// microphone
    pub const MICROPHONE: char = '\u{f130}';
    /// headphones
    pub const HEADPHONES: char = '\u{f025}';
    /// music
    pub const MUSIC: char = '\u{f001}';
    /// volume-high (status bar / media)
    pub const VOLUME_HIGH: char = '\u{f028}';
    /// volume-low
    pub const VOLUME_LOW: char = '\u{f027}';
    /// volume-off (0% but not muted)
    pub const VOLUME_OFF: char = '\u{f026}';
    /// volume-xmark (muted)
    pub const VOLUME_XMARK: char = '\u{f6a9}';
    /// image
    pub const IMAGE: char = '\u{f03e}';
    /// video
    pub const VIDEO: char = '\u{f03d}';
    /// camera
    pub const CAMERA: char = '\u{f030}';
    /// play
    pub const PLAY: char = '\u{f04b}';
    /// circle-play
    pub const CIRCLE_PLAY: char = '\u{f144}';

    // --- shell / UI class ---
    /// comments — shell agent
    pub const COMMENTS: char = '\u{f086}';
    /// terminal
    pub const TERMINAL: char = '\u{f120}';
    /// display / monitor — package UI canvas
    pub const DISPLAY: char = '\u{f390}';
    /// window-maximize
    pub const WINDOW: char = '\u{f2d0}';
    /// house
    pub const HOUSE: char = '\u{f015}';
    /// magnifying-glass
    pub const SEARCH: char = '\u{f002}';
    /// list-check — todos
    pub const LIST_CHECK: char = '\u{f0ae}';
    /// list
    pub const LIST: char = '\u{f03a}';
    /// clipboard-list
    pub const CLIPBOARD_LIST: char = '\u{f46d}';
    /// layer-group
    pub const LAYER_GROUP: char = '\u{f5fd}';
    /// bars — menu
    pub const BARS: char = '\u{f0c9}';
    /// check
    pub const CHECK: char = '\u{f00c}';
    /// square-check
    pub const SQUARE_CHECK: char = '\u{f14a}';
    /// square (empty checkbox)
    pub const SQUARE: char = '\u{f0c8}';
    /// minus
    pub const MINUS: char = '\u{f068}';
    /// plus
    pub const PLUS: char = '\u{f067}';
    /// chevron-right
    pub const CHEVRON_RIGHT: char = '\u{f054}';
    /// angle-right
    pub const ANGLE_RIGHT: char = '\u{f105}';
    /// pen-to-square — editor
    pub const PEN_TO_SQUARE: char = '\u{f044}';
    /// scroll / log stream
    pub const SCROLL: char = '\u{f70e}';
    /// bug — ktrace
    pub const BUG: char = '\u{f188}';
    /// gauge-high — /top
    pub const GAUGE: char = '\u{f624}';
    /// wave-square — audio
    pub const WAVE_SQUARE: char = '\u{f83e}';
    /// film — video
    pub const FILM: char = '\u{f008}';
    /// circle-info
    pub const CIRCLE_INFO: char = '\u{f05a}';
    /// triangle-exclamation
    pub const TRIANGLE_EXCLAMATION: char = '\u{f071}';
    /// ban
    pub const BAN: char = '\u{f05e}';
    /// circle-notch — spinner-ish
    pub const CIRCLE_NOTCH: char = '\u{f1ce}';
    /// star
    pub const STAR: char = '\u{f005}';
    /// ellipsis
    pub const ELLIPSIS: char = '\u{f141}';
    /// table-columns
    pub const TABLE_COLUMNS: char = '\u{f0db}';

    // --- chrome controls ---
    /// xmark — pane / modal close
    pub const XMARK: char = '\u{f00d}';
    /// circle-xmark
    pub const CIRCLE_XMARK: char = '\u{f057}';

    // --- cursor shapes (OS mouse pointer) ---
    /// arrow-pointer — default pointer
    pub const ARROW_POINTER: char = '\u{f245}';
    /// hand-pointer — link / clickable
    pub const HAND_POINTER: char = '\u{f25a}';
    /// i-cursor — text input
    pub const I_CURSOR: char = '\u{f246}';
    /// hourglass — wait / busy
    pub const HOURGLASS: char = '\u{f254}';

    // --- games / specialised ---
    /// chess-knight (also agent badge)
    pub const CHESS: char = '\u{f441}';
    /// chess-pawn
    pub const CHESS_PAWN: char = '\u{f443}';
    /// chess-rook
    pub const CHESS_ROOK: char = '\u{f447}';
    /// chess-knight
    pub const CHESS_KNIGHT: char = '\u{f441}';
    /// chess-bishop
    pub const CHESS_BISHOP: char = '\u{f43a}';
    /// chess-queen
    pub const CHESS_QUEEN: char = '\u{f445}';
    /// chess-king
    pub const CHESS_KING: char = '\u{f43f}';
    /// chess-board
    pub const CHESS_BOARD: char = '\u{f43c}';
    /// gamepad
    pub const GAMEPAD: char = '\u{f11b}';
    /// bomb (minesweeper)
    pub const BOMB: char = '\u{f1e2}';
    /// border-all / grid
    pub const GRID: char = '\u{f84c}';
    /// table-cells
    pub const TABLE_CELLS: char = '\u{f00a}';
    /// calculator
    pub const CALCULATOR: char = '\u{f1ec}';
    /// table
    pub const TABLE: char = '\u{f0ce}';
    /// calendar-days
    pub const CALENDAR: char = '\u{f073}';
    /// map
    pub const MAP: char = '\u{f279}';
    /// map-marker
    pub const MAP_MARKER: char = '\u{f3c5}';
    /// cloud
    pub const CLOUD: char = '\u{f0c2}';
    /// cloud-sun
    pub const CLOUD_SUN: char = '\u{f6c4}';
    /// sun
    pub const SUN: char = '\u{f185}';
    /// flask
    pub const FLASK: char = '\u{f0c3}';
    /// cart-shopping
    pub const CART: char = '\u{f07a}';
    /// code
    pub const CODE: char = '\u{f121}';
    /// code-compare (FA6, private-use extension)
    pub const CODE_COMPARE: char = '\u{e13a}';
    /// database
    pub const DATABASE: char = '\u{f1c0}';
    /// puzzle-piece
    pub const PUZZLE: char = '\u{f12e}';
}

/// True if `ch` is a Font Awesome Private-Use scalar we treat as a UI icon
/// (drawn larger in the status bar).
///
/// Free Solid lives mainly in `U+F000`–`U+F8FF`; FA6 also places newer glyphs
/// in `U+E000`–`U+EFFF` (e.g. `code-compare`).
pub fn is_icon(ch: char) -> bool {
    let u = ch as u32;
    (0xf000..=0xf8ff).contains(&u) || (0xe000..=0xefff).contains(&u)
}

/// Body-sized active pulse (not Font Awesome — FA solid circle was scaled to
/// full icon size and read as a giant blob next to the mouse/keyboard).
const ACTIVE_MARK: char = '\u{00B7}'; // middle dot ·

/// Status-bar keyboard indicator (active = icon + small body-size mark).
pub fn status_kbd(active: bool) -> alloc::string::String {
    if active {
        alloc::format!("{}{}", fa::KEYBOARD, ACTIVE_MARK)
    } else {
        fa::KEYBOARD.to_string()
    }
}

/// Status-bar mouse indicator.
pub fn status_mouse(active: bool) -> alloc::string::String {
    if active {
        alloc::format!("{}{}", fa::MOUSE, ACTIVE_MARK)
    } else {
        fa::MOUSE.to_string()
    }
}

/// Status-bar network indicator.
pub fn status_net(up: bool) -> alloc::string::String {
    if up {
        fa::WIFI.to_string()
    } else {
        // network-wired when down still reads as "net", dimmer via theme fg.
        fa::NETWORK.to_string()
    }
}

/// Status-bar **volume** chip: FA speaker glyph that reflects mute / level.
/// Compact like macOS (icon only — percent lives in the dropdown).
pub fn status_volume(muted: bool, pct: u32) -> alloc::string::String {
    let icon = if muted {
        fa::VOLUME_XMARK
    } else if pct == 0 {
        fa::VOLUME_OFF
    } else if pct < 50 {
        fa::VOLUME_LOW
    } else {
        fa::VOLUME_HIGH
    };
    icon.to_string()
}

/// FA speaker glyph for a given mute/level (menus, about panels).
pub fn volume_icon(muted: bool, pct: u32) -> char {
    if muted {
        fa::VOLUME_XMARK
    } else if pct == 0 {
        fa::VOLUME_OFF
    } else if pct < 50 {
        fa::VOLUME_LOW
    } else {
        fa::VOLUME_HIGH
    }
}

/// Icon for a package-UI vs shell agent class badge.
pub fn ui_class_icon(canvas: bool) -> char {
    if canvas {
        fa::DISPLAY
    } else {
        fa::COMMENTS
    }
}

/// Font Awesome chess piece for a FEN letter (`KQRBNP` / `kqrbnp`).
/// Unknown → knight (generic chess mark).
pub fn chess_piece(fen: char) -> char {
    match fen.to_ascii_lowercase() {
        'p' => fa::CHESS_PAWN,
        'r' => fa::CHESS_ROOK,
        'n' => fa::CHESS_KNIGHT,
        'b' => fa::CHESS_BISHOP,
        'q' => fa::CHESS_QUEEN,
        'k' => fa::CHESS_KING,
        _ => fa::CHESS_KNIGHT,
    }
}

/// Font Awesome close mark used by pane tabs and list-browser modals.
#[inline]
pub fn close_mark() -> char {
    fa::XMARK
}

/// Icon for a `/help` command category header.
pub fn for_command_category(cat: &str) -> char {
    match cat {
        "Session" => fa::COMMENTS,
        "Context" => fa::CLIPBOARD_LIST,
        "Model & Agent" => fa::ROBOT,
        "Files" => fa::FOLDER,
        "Storage" => fa::HARD_DRIVE,
        "Network" => fa::WIFI,
        "System & UI" => fa::GEAR,
        "Media" => fa::MUSIC,
        _ => fa::LIST,
    }
}

/// Best-effort icon for a shell slash-command name (help browser + suggest).
pub fn for_command(name: &str) -> char {
    match name {
        "session" | "clear" | "compact" => fa::COMMENTS,
        "agents" => fa::ROBOT,
        "exit" | "restart" => fa::POWER_OFF,
        "memory" | "skills" | "skill" | "plan" | "permissions" => fa::BOOK,
        "todos" => fa::LIST_CHECK,
        "model" | "infer" | "perf" | "bench" | "think" | "mode" => fa::ROBOT,
        "redteam" | "audit" => fa::SHIELD,
        "ls" | "cat" | "open" | "mkdir" | "cp" | "mv" | "rm" | "touch" | "glob" | "grep"
        | "pwd" | "decoder" | "js" => fa::FOLDER,
        "disks" | "mounts" | "mount" | "umount" | "install" | "mkext4" => fa::HARD_DRIVE,
        "network" | "ping" | "wifi" | "http" | "ws" | "mcp" | "channel" | "browse" => fa::WIFI,
        "info" | "about" | "ui" | "shortcuts" | "help" => fa::CIRCLE_INFO,
        "top" => fa::GAUGE,
        "ktrace" => fa::BUG,
        "datetime" => fa::CLOCK,
        "battery" | "power" | "suspend" => fa::BATTERY,
        "theme" | "statusbar" | "pane" | "display" => fa::PALETTE,
        "clip" => fa::FILE_LINES,
        "close" => fa::XMARK,
        "voice" | "onnx" => fa::MICROPHONE,
        "bluetooth" | "camera" | "touchscreen" | "lspci" => fa::MICROCHIP,
        _ => fa::TERMINAL,
    }
}

/// Font Awesome glyph for an OS cursor shape (CSS-ish set).
///
/// - Arrow → `arrow-pointer`
/// - Hand → `hand-pointer`
/// - IBeam → `i-cursor`
/// - Wait → `hourglass`
pub fn cursor_glyph(shape: u8) -> char {
    match shape {
        1 => fa::HAND_POINTER,
        2 => fa::I_CURSOR,
        3 => fa::HOURGLASS,
        _ => fa::ARROW_POINTER,
    }
}

/// Best-effort Font Awesome glyph for a system agent name (fallback: robot).
pub fn for_agent(name: &str) -> char {
    match name {
        "files" | "disk" => fa::FOLDER,
        "notes" | "todo" => fa::FILE_LINES,
        "writer" | "reader" => fa::PEN,
        "doc" | "pdf" | "librarian" => fa::BOOK,
        "settings" => fa::GEAR,
        "chess" => fa::CHESS,
        "paint" => fa::PALETTE,
        "gallery" => fa::IMAGE,
        "media" => fa::VIDEO,
        "synth" | "radio" => fa::MUSIC,
        "recorder" => fa::MICROPHONE,
        "clock" => fa::CLOCK,
        "calc" => fa::CALCULATOR,
        "weather" => fa::CLOUD_SUN,
        "maps" => fa::MAP,
        "download" => fa::DOWNLOAD,
        "browser" => fa::GLOBE,
        "pass" => fa::KEY,
        "mail" => fa::ENVELOPE,
        "activity" | "ops" => fa::MICROCHIP,
        "ssh" | "console" => fa::TERMINAL,
        "contacts" => fa::ADDRESS_BOOK,
        "calendar" => fa::CALENDAR,
        "dict" => fa::BOOK_OPEN,
        "diff" => fa::CODE_COMPARE,
        "hex" => fa::CODE,
        "sheets" => fa::TABLE,
        "slides" => fa::LAYER_GROUP,
        "archive" => fa::BOX_ARCHIVE,
        "minesweeper" => fa::BOMB,
        "snake" | "game2048" | "breakout" | "tetris" => fa::GAMEPAD,
        "sandbox-lab" => fa::FLASK,
        "onboard" => fa::HOUSE,
        "store" => fa::CART,
        "researcher" => fa::SEARCH,
        "newspaper" => fa::NEWSPAPER,
        _ => fa::ROBOT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn fa_codepoints_are_private_use() {
        // Classic block
        assert!(is_icon(fa::KEYBOARD));
        assert!(is_icon(fa::WIFI));
        assert!(is_icon(fa::GEAR));
        assert!(is_icon(fa::FOLDER));
        // FA6 extension block
        assert!(is_icon(fa::CODE_COMPARE));
        // Not icons
        assert!(!is_icon('A'));
        assert!(!is_icon('⌨'));
        assert!(!is_icon(' '));
    }

    #[test_case]
    fn status_helpers_compose_active_mark() {
        let k = status_kbd(true);
        assert!(k.starts_with(fa::KEYBOARD));
        // Active mark is a body-size middle-dot, never a full FA circle.
        assert!(k.contains(ACTIVE_MARK));
        assert!(!k.contains(fa::CIRCLE));
        assert_eq!(status_kbd(false), fa::KEYBOARD.to_string());
        assert_eq!(status_mouse(true).chars().count(), 2);
        assert_eq!(status_net(true), fa::WIFI.to_string());
        assert_eq!(status_net(false), fa::NETWORK.to_string());
    }

    #[test_case]
    fn status_volume_icon_tracks_mute_and_level() {
        assert_eq!(volume_icon(true, 80), fa::VOLUME_XMARK);
        assert_eq!(volume_icon(false, 0), fa::VOLUME_OFF);
        assert_eq!(volume_icon(false, 25), fa::VOLUME_LOW);
        assert_eq!(volume_icon(false, 50), fa::VOLUME_HIGH);
        assert_eq!(volume_icon(false, 100), fa::VOLUME_HIGH);
        assert!(is_icon(fa::VOLUME_HIGH));
        assert!(is_icon(fa::VOLUME_XMARK));
        assert_eq!(status_volume(false, 100), fa::VOLUME_HIGH.to_string());
        assert_eq!(status_volume(true, 100), fa::VOLUME_XMARK.to_string());
    }

    #[test_case]
    fn agent_icons_are_stable_and_iconic() {
        assert_eq!(for_agent("files"), fa::FOLDER);
        assert_eq!(for_agent("settings"), fa::GEAR);
        assert_eq!(for_agent("chess"), fa::CHESS);
        assert_eq!(for_agent("notes"), fa::FILE_LINES);
        assert_eq!(for_agent("unknown-agent-xyz"), fa::ROBOT);
        assert!(is_icon(for_agent("weather")));
        assert!(is_icon(for_agent("diff"))); // extension PUA
        assert_eq!(ui_class_icon(true), fa::DISPLAY);
        assert_eq!(ui_class_icon(false), fa::COMMENTS);
    }

    #[test_case]
    fn chess_and_cursor_glyphs_map_fen_and_shapes() {
        assert_eq!(chess_piece('K'), fa::CHESS_KING);
        assert_eq!(chess_piece('q'), fa::CHESS_QUEEN);
        assert_eq!(chess_piece('N'), fa::CHESS_KNIGHT);
        assert_eq!(chess_piece('p'), fa::CHESS_PAWN);
        assert!(is_icon(chess_piece('b')));
        assert_eq!(cursor_glyph(0), fa::ARROW_POINTER);
        assert_eq!(cursor_glyph(1), fa::HAND_POINTER);
        assert_eq!(cursor_glyph(2), fa::I_CURSOR);
        assert_eq!(cursor_glyph(3), fa::HOURGLASS);
        assert!(is_icon(cursor_glyph(0)));
        assert_eq!(close_mark(), fa::XMARK);
        assert!(is_icon(close_mark()));
    }
}
