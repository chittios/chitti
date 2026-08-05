//! Framebuffer compositor (`CHITTI_OS_HANDOFF.md` Phase 7 stretch: "framebuffer
//! text UI beyond serial"). A tmux-style split-pane terminal drawn directly on
//! the framebuffer: two bordered panes side by side -- **chat** (left, the
//! interactive REPL) and **logs** (right, the live ktrace stream) -- an
//! active-pane highlight, and a bottom status bar. Text is rendered with the
//! Geist Mono glyph atlas ([`crate::font_geist`]) alpha-blended per pixel, so
//! the panes show antialiased type rather than a bare bitmap grid.
//!
//! The framebuffer geometry and pixel format are always taken from the boot
//! source -- the Limine framebuffer (x86), the UEFI GOP handed over by the stub
//! (aarch64 real hardware / VirtualBox / UTM), or QEMU ramfb -- never hardcoded.
//! On a high-resolution panel (a 4K HDMI monitor) the 10x22 atlas cell would be
//! microscopic, so the console picks an integer font `scale` from the panel
//! height: text stays legible while the panes still fill the whole screen.
//!
//! It is a global singleton ([`SCREEN`]) rather than a transient writer because
//! the two log channels mirror here automatically: `serial::Serial` (every
//! `serial_print!`/`serial_println!`, i.e. the shell + chat) draws into the
//! chat pane via [`console_print`], while `ktrace` draws into the logs pane via
//! [`log_print`]. Keyboard input (`arch::keyboard`) plus this output is what
//! makes the framebuffer a real console, not just a log mirror.

//! ## How this module is laid out
//!
//! `mod.rs` owns the **data model** — every type, constant and static — and the
//! submodules own the **code**. That division is not taste: a child module can
//! see its parent's private items, so a `Screen` field declared here is reachable
//! from every painter without a single visibility annotation, while the reverse
//! (a struct in a submodule) would force `pub(super)` on all ~100 of its fields.
//! Items a submodule owns outright are `pub(super)`, which is visible across
//! `framebuffer::*` and nowhere else, and each submodule is re-exported below so
//! **every `crate::framebuffer::…` path is unchanged** by the split.
//!
//! | module | what it draws / owns |
//! |---|---|
//! | [`colors`] | the [`Theme`] palette, hex parsing, ANSI colour tables |
//! | [`paint`] | pixel/rect/disc fills, wallpaper + surface compositing, AA coverage |
//! | [`text`] | glyph rasterisation, string drawing, wrap/truncate |
//! | [`pane`] | a pane's grid + scrollback, the ANSI/UTF-8 feed, its rows and caret |
//! | [`layout`] | building the pane tree, font scale, logical desktop, dividers |
//! | [`tabs`] | which view lives in which action pane, and drag-and-drop between them |
//! | [`focus`] | which pane has keyboard focus, and scrolling it |
//! | [`status`] | the status bar on any edge, its chips and hit rects |
//! | [`menu`] · [`clock`] | the status-chip dropdowns, and the analog clock face |
//! | [`modal`] | confirm/choose/input prompts, About, the list browsers |
//! | [`composer`] | the chat composer box, caret, hint line, autosuggest popup |
//! | [`views`] | `/top`, the audio/video/browser HUDs, todos, the editor |
//! | [`surface`] | presenting an app's own RGB buffer into a pane |
//! | [`select`] | mouse text selection in the chat pane |
//! | [`cursor`] | the mouse cursor sprites and overlay |
//! | [`console`] | bringing the console up from a boot framebuffer; the log channels |

use crate::font_geist::{CELL_H as CH, CELL_W as CW, FIRST, GLYPHS, LAST};
use crate::limine_protocol::Framebuffer;
use crate::mm::Locked;
use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

mod clock;
mod colors;
mod composer;
mod console;
mod cursor;
mod focus;
mod layout;
mod menu;
mod modal;
mod paint;
mod pane;
mod select;
mod status;
mod surface;
mod tabs;
mod text;
mod views;

use clock::*;
pub use colors::*;
pub use composer::*;
pub use console::*;
pub use cursor::*;
pub use focus::*;
pub use layout::*;
pub use menu::*;
pub use modal::*;
use paint::*;
use pane::*;
pub use select::*;
pub use status::*;
pub use surface::*;
pub use tabs::*;
use text::*;
pub use views::*;

const CELL_W: u64 = CW as u64;
const CELL_H: u64 = CH as u64;

type Rgb = (u8, u8, u8);

/// One character cell of a pane's text grid: the byte drawn (0 = empty) and its
/// colour at draw time. The grid (plus the scrollback ring behind it) is the
/// source of truth for a pane's text, so content survives redraws, relayouts,
/// and modal dismissals, and can be scrolled back through.
type Cell = (char, Rgb);

/// An elevated background band over a run of chat lines.
///
/// The chat pane has no per-cell background (`Cell` is `(char, fg)`), so a block tint is
/// carried as line metadata the compositor owns rather than as colour in the text stream.
/// That also keeps the serial console byte-identical: the tint and the icon chrome are
/// rendering, and a host terminal reading the same stream is unaffected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Band {
    None,
    /// A line the human typed (`theme.composer_bg`).
    User,
    /// An agent tool call and its output (`theme.tool_bg`).
    Tool,
}

/// Scrollback depth per pane, in lines. At 200 cols a full ring is ~3 MB —
/// noise next to the model heap. Cleared only by `/clear`.
const HIST_MAX: usize = 2000;

const OUTER: u64 = 8; // margin around the whole content region
const GAP: u64 = 10; // between the two panes
const BORDER: u64 = 2; // pane border thickness
const PAD: u64 = 10; // interior padding inside a pane
const CHAT_PCT: u64 = 56; // chat pane width as a % of the content region

/// One bordered text pane: an outer box plus the interior character grid it
/// scrolls text within. Colours, cursor, and the (scaled) cell size live here;
/// the pixel plumbing lives on [`Screen`], which owns the framebuffer.
struct Pane {
    // Outer box (border-inclusive), pixels.
    x: u64,
    y: u64,
    w: u64,
    h: u64,
    // Interior text origin (top-left of cell 0,0), pixels.
    ix: u64,
    iy: u64,
    // Scaled cell size, pixels.
    cw: u64,
    ch: u64,
    // Interior size, cells.
    cols: u64,
    rows: u64,
    // Cursor, cells.
    col: u64,
    row: u64,
    // `fg` is the *current* text colour (mutated by ANSI SGR codes in the byte
    // stream); `default_fg` is what a reset (`\x1b[0m`/`\x1b[39m`) restores.
    fg: Rgb,
    default_fg: Rgb,
    bg: Rgb,
    // ANSI escape-sequence parser state (see `pane_putc`).
    esc: EscState,
    csi: [u8; 32],
    csi_len: usize,
    bold: bool,
    title: String,
    show_caret: bool,
    /// The live character grid (`cols * rows` cells) mirroring what is drawn.
    grid: Vec<Cell>,
    /// Scrollback: lines evicted off the top of the grid, oldest first.
    hist: VecDeque<Vec<Cell>>,
    /// Scrollback view offset in lines back from live (0 = live). While > 0,
    /// incoming bytes still update the grid/hist but pixels are frozen on the
    /// scrolled view; the offset auto-advances so the view stays anchored.
    view: usize,
    /// Mouse text selection `(anchor, head)`, both inclusive `(line, col)` in
    /// **absolute** coordinates over `hist` + grid (see `crate::textsel`), so
    /// it stays glued to its text while the pane scrolls. `None` = no selection.
    sel: Option<((usize, usize), (usize, usize))>,
    /// When true, the pane reserves its bottom for a bordered input composer
    /// (bordered box + hint row); the scrollback grid sits above it.
    has_composer: bool,
    /// Expandable folds: `(gi, hidden)` where `gi` is the absolute line index
    /// (same coords as `sel`) of a clickable "▸ N more…" line and `hidden` is
    /// the collapsed text revealed on click. Evicted with the scrollback.
    folds: Vec<(usize, String)>,
    /// Absolute line indices (`hist`+grid, same as `sel`) painted with the
    /// elevated user-prompt band background (`theme.composer_bg`).
    user_band: Vec<usize>,
    /// Lines evicted from `hist` over this pane's life.
    ///
    /// Absolute `gi` coordinates are *not* monotonic: once `hist` saturates at `HIST_MAX`
    /// every new line evicts one, so `hist.len() + row` stops advancing and a
    /// "mark before, measure after" span comes out as **zero** — the tool tint would
    /// simply stop appearing once a session got long enough, which is the kind of bug
    /// that shows up only after an hour of use. `evicted + hist.len() + row` always
    /// advances, so a span measured across printing is right whether or not the ring wrapped.
    evicted: usize,
    /// The same, for agent **tool-call blocks** (`theme.tool_bg`).
    ///
    /// A second sorted vec rather than one list of `(gi, kind)`: both are searched per
    /// cell on every repaint, and `binary_search` over a plain `usize` is what makes that
    /// free. They are disjoint by construction — a line is either the human's prompt or a
    /// tool block — and [`Pane::band`] resolves user-first if that ever stops holding.
    tool_band: Vec<usize>,
    /// Incremental UTF-8 decode buffer: the incoming byte stream is decoded one
    /// `char` at a time (a multi-byte glyph spans several `pane_putc` calls).
    utf8: [u8; 4],
    utf8_len: u8,
}

/// Minimal ANSI escape-sequence parser state for a pane's byte stream: we
/// recognise `ESC [ … <final>` (CSI) and honour SGR (`… m`) colour/emphasis so
/// the shell agent can format replies with [ANSI codes]. Other CSI sequences
/// (cursor moves, erase) are consumed and ignored — enough to render coloured
/// text without a full terminal emulator.
#[derive(Clone, Copy, PartialEq)]
enum EscState {
    Ground,
    Esc,
    Csi,
}

/// The compositor: framebuffer geometry + the two panes it draws.
pub struct Screen {
    addr: usize,
    width: u64,
    height: u64,
    pitch: u64,
    bpp_bytes: u64,
    r_shift: u32,
    g_shift: u32,
    b_shift: u32,
    scale: u64,
    chat: Pane,
    /// Action columns (1..=8). Index 0 is the primary; focused column is
    /// [`Self::focused_action`].
    actions: Vec<ActionSlot>,
    /// Which action column has keyboard/mouse focus for tabs and open targets.
    focused_action: usize,
    /// Status-bar text (left = brand, right = datetime); set by the shell from
    /// the UI-config templates + clock, so it stays configurable.
    status_left: String,
    status_right: String,
    /// Blinking-caret state for the chat pane / composer.
    caret_on: bool,
    caret_last_ms: u64,
    /// bordered input composer (bottom of chat pane): when `composer_active`
    /// the caret lives in the bordered box, not the scrollback grid.
    composer_active: bool,
    composer_line: String,
    composer_cur: usize,
    /// Hint bar under the composer (left / right halves).
    composer_hint_l: String,
    composer_hint_r: String,
    /// The composer prompt prefix — `~/path (branch) > `, set by the shell so
    /// it reflects the live cwd + git branch (falls back to `"> "`).
    composer_prompt: String,
    /// Per-character colours for the **leading** cells of `composer_hint_l`
    /// (empty = the whole hint takes `theme.composer_hint`). This is what lets
    /// the shell paint its gradient progress bar into the hint bar without the
    /// framebuffer knowing anything about the animation.
    composer_hint_l_lead: alloc::vec::Vec<Rgb>,
    /// Slash-command / @file / path-argument suggestion popup above the composer.
    suggest_open: bool,
    suggest_items: alloc::vec::Vec<(String, String)>, // (label, detail)
    suggest_sel: usize,
    /// Last painted popup rect `(x, y, w, h)` — used to erase the dirty region
    /// cleanly (includes the gap above the composer that the chat grid does
    /// not cover).
    suggest_rect: Option<(u64, u64, u64, u64)>,
    /// Fallback blink cadence when the monotonic clock is frozen (some VBox
    /// configs): the last `now_ms` seen, and a call counter. `clock_alive`
    /// latches once `now_ms` is ever seen advancing — after that the fallback
    /// is never used (on a fast host thousands of calls land in the same
    /// millisecond, which used to trip the counter and blink far too fast).
    blink_seen_ms: u64,
    blink_calls: u32,
    clock_alive: bool,
    /// Physical framebuffer size, as the firmware reported it. `width`/`height`
    /// are the **logical** desktop, which may be smaller (a letterboxed viewport).
    fb_w: u64,
    fb_h: u64,
    /// Top-left of the logical desktop inside the physical framebuffer. Both zero
    /// when the desktop is native.
    origin_x: u64,
    origin_y: u64,
    /// The requested logical desktop, carried across rebuilds. `None` = native.
    logical_pref: Option<(u64, u64)>,
    /// The status bar's rect and the content rect left over, from
    /// `panes_layout::status_split`. Every pane-layout calculation works inside
    /// `content_*` rather than `0..width`/`0..height`, so the bar can sit on any
    /// edge without a second set of layout paths. At `Top` (the default) the
    /// content origin is `(0, bar_h)`; at `Bottom` it is `(0, 0)`.
    status_rect: crate::panes_layout::Rect,
    content_x: u64,
    content_y: u64,
    content_w: u64,
    content_h: u64,
    /// Whether keyboard focus is on an action column (vs the shell/chat).
    focus_action: bool,
    /// Action column currently highlighted as a tab drag's drop target.
    drop_target: Option<usize>,
    /// The last-applied layout config, reused when opening/closing the action
    /// pane so the split ratio / titles / scale are preserved.
    layout: LayoutCfg,
    /// The active colour palette (from `layout.theme`).
    theme: Theme,
    /// Mouse cursor sprite state: position + the framebuffer patch saved beneath
    /// it (restored before each move so the cursor leaves no trail).
    cur_x: u64,
    cur_y: u64,
    cur_vis: bool,
    /// True once the mouse has moved, so content redraws keep re-drawing the
    /// cursor on top instead of leaving it erased.
    cur_active: bool,
    /// Background patch saved beneath the cursor sprite; sized to the last-drawn
    /// sprite (`cur_sw`×`cur_sh`) so a theme's custom (variable-size) cursor can
    /// be restored with the exact dims it was drawn at.
    cur_saved: Vec<Rgb>,
    cur_sw: u64,
    cur_sh: u64,
    /// Decoded wallpaper scaled to the full screen (`0x00RRGGBB`, width×height),
    /// or `None` for the solid-colour desktop. Windows blend over it at
    /// [`Self::opacity`]; the gutters show it directly.
    wallpaper: Option<Vec<u32>>,
    /// Window opacity over the wallpaper (255 = opaque; only used when
    /// `wallpaper` is `Some`).
    opacity: u8,
}

// Mouse cursor sprites: 0 = transparent, 1 = fill, 2 = outline.
// Shapes: Arrow (default), Hand (link pointer), IBeam (text input).

/// What the right ("action") pane shows.
#[derive(Clone, Copy, PartialEq)]
pub enum RightMode {
    /// Closed: chat pane is full-width (the default).
    Closed,
    /// The live ktrace log stream.
    Ktrace,
    /// The `/open` editor.
    Editor,
    /// The live `/top` system dashboard (CPU + memory).
    Top,
    /// Live session todo list (`/todos open`).
    Todos,
    /// The `/open <file>.wav|.mp3` background audio player.
    Audio,
    /// An agent-owned drawing surface (`synapse::ui`), by surface id.
    Surface(u32),
}

/// A snapshot of one action pane's interior geometry, copied out so a painter
/// can keep using it while it mutates the screen (no borrow held on `actions`).
#[derive(Clone, Copy)]
struct PaneDims {
    /// Outer box origin (frame, not interior).
    x: u64,
    /// Interior origin.
    ix: u64,
    iy: u64,
    /// Interior size in pixels.
    w: u64,
    iw: u64,
    ih: u64,
    cw: u64,
    ch: u64,
    cols: u64,
    rows: u64,
    bg: Rgb,
}

impl PaneDims {
    fn of(p: &Pane) -> PaneDims {
        PaneDims {
            x: p.x,
            ix: p.ix,
            iy: p.iy,
            w: p.w,
            iw: p.cols * p.cw,
            ih: p.rows * p.ch,
            cw: p.cw,
            ch: p.ch,
            cols: p.cols,
            rows: p.rows,
            bg: p.bg,
        }
    }
}

/// One action pane in the grid: its geometry + tmux-style tab list.
struct ActionSlot {
    pane: Pane,
    tabs: Vec<RightMode>,
    active: usize,
}

impl ActionSlot {
    fn right(&self) -> RightMode {
        self.tabs.get(self.active).copied().unwrap_or(RightMode::Closed)
    }

    fn is_empty(&self) -> bool {
        self.tabs.is_empty()
    }
}

/// Config knobs the UI config (`/configs/core/ui.json`) can set for the layout.
#[derive(Clone)]
pub struct LayoutCfg {
    /// Chat pane width as a % of the content region (10..90).
    pub chat_pct: u64,
    /// Total panes including the shell (2..=9). Action panes = max_panes - 1.
    pub max_panes: u8,
    /// The action band's grid shape and per-track weights. `cols * rows` is the
    /// action-pane count, so it and `max_panes` are kept consistent by
    /// [`set_max_panes`] / [`set_grid`].
    pub grid: crate::panes_layout::GridSpec,
    /// Font scale; 0 = auto from panel height.
    pub scale: u64,
    /// Put the chat pane on the right instead of the left.
    pub swap: bool,
    /// Which desktop edge the OS status bar occupies. Everything else lays out
    /// inside the leftover content rect, so this shifts the whole UI.
    pub status_pos: crate::panes_layout::StatusPos,
    pub chat_title: String,
    pub logs_title: String,
    /// Colour palette (from `ui.json` `theme`; default = brand dark).
    pub theme: Theme,
    /// Show the boot splash (logo + name). Default true.
    pub splash: bool,
    /// Fullscreen state: 0 = normal split, 1 = chat fills the screen, 2 = the
    /// action pane fills the screen. Toggled at runtime (F11 / `/fullscreen`).
    pub fullscreen: u8,
    /// Wallpaper spec: `""` = solid `screen_bg`; `"gradient:#rrggbb,#rrggbb"` =
    /// generated vertical gradient; otherwise a path to an image in the store.
    pub wallpaper: String,
    /// Window opacity over the wallpaper (0..=255; 255 = opaque, the default —
    /// identical to the no-wallpaper look). Only meaningful with a wallpaper.
    pub opacity: u8,
}

impl Default for LayoutCfg {
    fn default() -> Self {
        LayoutCfg {
            chat_pct: CHAT_PCT,
            max_panes: crate::panes_layout::MAX_PANES_DEFAULT,
            grid: crate::panes_layout::GridSpec::even(1, 1),
            scale: 0,
            swap: false,
            status_pos: crate::panes_layout::StatusPos::default(),
            chat_title: String::from("Shell Agent"),
            logs_title: String::from("ktrace"),
            theme: Theme::default(),
            splash: true,
            fullscreen: 0,
            wallpaper: String::new(),
            opacity: 255,
        }
    }
}

static SCREEN: Locked<Option<Screen>> = Locked::new(None);

// --- modal overlay (approval / input dialogs) ---------------------------

/// Which modal control the mouse hit, for [`modal_hit`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModalHit {
    None,
    Yes,
    No,
    Ok,
    /// Commands/agents-browser close (FA xmark; slot 0 reused when that modal is up).
    Close,
    /// Absolute row index in a list browser (`scroll + visible row`). Headers
    /// and items share the index space; callers skip non-selectable rows.
    ListRow(usize),
    /// Option index in a [`draw_choose`] multi-choice modal.
    Choose(usize),
}

/// Pixel rects of the modal's clickable controls: `[yes, no, ok]`. Set when a
/// modal is drawn, read by [`modal_hit`] for mouse routing. Zero-size = absent.
static MODAL_RECTS: Locked<[(u64, u64, u64, u64); 3]> = Locked::new([(0, 0, 0, 0); 3]);

/// Dedicated close-mark rect (FA xmark). Checked **first** in [`modal_hit`] so
/// About / status menus can put Close in slot 0 *and* use slots 1–2 for other
/// chrome without Close being mis-classified as Yes.
static MODAL_CLOSE_RECT: Locked<(u64, u64, u64, u64)> = Locked::new((0, 0, 0, 0));

/// Geometry of the scrollable list in `/help` and `/agents` browsers, for
/// mouse row hit-testing. Cleared on dismiss.
#[derive(Clone, Copy)]
struct ListBrowserGeom {
    list_x: u64,
    list_y: u64,
    list_w: u64,
    row_h: u64,
    /// Visible row count (≤ 12).
    n_rows: usize,
    /// Absolute index of the first visible row.
    scroll: usize,
}

static LIST_BROWSER_GEOM: Locked<Option<ListBrowserGeom>> = Locked::new(None);

/// Option row rects for [`draw_choose`] (up to 9 numbered choices).
static CHOOSE_RECTS: Locked<[(u64, u64, u64, u64); 9]> =
    Locked::new([(0, 0, 0, 0); 9]);

static CHOOSE_COUNT: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

/// True while a modal overlays the panes: upkeep ticks running under it (long
/// compute pumps `shell::upkeep`) must not blink the pane caret into the box.
static MODAL_ON: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Clickable status-bar chips (macOS menu-bar style). Brand opens About; the
/// rest open a dropdown popover with live details.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StatusChip {
    Brand = 0,
    Kbd = 1,
    Mouse = 2,
    Net = 3,
    Mem = 4,
    Cpu = 5,
    Battery = 6,
    /// Software output volume (`sound::volume` / mute).
    Volume = 7,
    Clock = 8,
}

const STATUS_CHIP_N: usize = 9;

/// Hit rects for [`StatusChip`] (index = `chip as usize`). Zero-size = absent.
static STATUS_CHIP_RECTS: Locked<[(u64, u64, u64, u64); STATUS_CHIP_N]> =
    Locked::new([(0, 0, 0, 0); STATUS_CHIP_N]);

/// Full status-bar rect (for anchoring menus above/below/beside).
static STATUS_BAR_RECT: Locked<(u64, u64, u64, u64)> = Locked::new((0, 0, 0, 0));

fn in_rect(x: u64, y: u64, r: (u64, u64, u64, u64)) -> bool {
    r.2 != 0 && x >= r.0 && x < r.0 + r.2 && y >= r.1 && y < r.1 + r.3
}

/// One visible row for [`draw_commands_browser`].
pub enum CommandsRow<'a> {
    Header(&'a str),
    Item {
        title: &'a str,
        slash: &'a str,
        shortcut: &'a str,
        selected: bool,
    },
}

/// Open `mode` on the focused action column (or select it if already open
/// anywhere — focuses that column). First tab on a collapsed band opens the split.
/// Set when the action band's geometry or tab set changed, so the pump knows the
/// panes' *interiors* need repainting.
///
/// The compositor redraws frames on a relayout, but each view owns its interior — a
/// browser page, a chess board, a paint canvas are RGB buffers the app holds, and
/// `Screen::redraw` cannot reproduce them. So an unfocused surface pane goes blank until
/// it happens to tick, which for a browser or a finished game is never.
///
/// `shell::repaint_visible_tabs` has always existed for this and documented that "a
/// divider drag, `/pane grid|max|split`, a tab move" must call it — and then had exactly
/// **one** caller (the theme path), so every other band change blanked its neighbours.
/// A flag drained by the pump fixes that class rather than that instance: a band mutation
/// added later gets the repaint without anyone remembering to ask.
static TABS_DIRTY: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Which pane a click at `(x, y)` landed in: `Some(true)` = action pane,
/// `Some(false)` = chat pane, `None` = neither (status bar / margins).
/// Last presented surface dimensions (logical sw×sh) so hit-testing matches
/// the aspect-fit used by [`present_surface_reserve`] (browser is 640×400,
/// chess/ui is 256×192, etc.).
static LAST_SURF_DIM: crate::mm::Locked<alloc::collections::BTreeMap<u32, (usize, usize)>> =
    crate::mm::Locked::new(alloc::collections::BTreeMap::new());

/// Bottom HUD reserve (px) last used with [`present_surface_reserve`] per surface.
static LAST_SURF_RESERVE: crate::mm::Locked<alloc::collections::BTreeMap<u32, u64>> =
    crate::mm::Locked::new(alloc::collections::BTreeMap::new());
