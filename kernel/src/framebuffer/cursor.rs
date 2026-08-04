//! The mouse cursor: built-in sprites, the themable fill/outline and Font
//! Awesome shapes, and the save/restore overlay that keeps it above the text.

use super::*;

pub(super) const CUR_W: u64 = 12;

pub(super) const CUR_H: u64 = 19;

/// OS cursor shape (CSS `cursor` subset — Ladybird pointer/text/default).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CursorShape {
    Arrow = 0,
    Hand = 1,
    IBeam = 2,
    Wait = 3,
}

static CURSOR_SHAPE: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);

/// Set the mouse cursor glyph (restored on next move/draw).
pub fn set_cursor_shape(shape: CursorShape) {
    CURSOR_SHAPE.store(shape as u8, core::sync::atomic::Ordering::Relaxed);
}

pub fn cursor_shape() -> CursorShape {
    match CURSOR_SHAPE.load(core::sync::atomic::Ordering::Relaxed) {
        1 => CursorShape::Hand,
        2 => CursorShape::IBeam,
        3 => CursorShape::Wait,
        _ => CursorShape::Arrow,
    }
}

#[rustfmt::skip]
const CURSOR_ARROW: [u8; (CUR_W * CUR_H) as usize] = [
    2,0,0,0,0,0,0,0,0,0,0,0,
    2,2,0,0,0,0,0,0,0,0,0,0,
    2,1,2,0,0,0,0,0,0,0,0,0,
    2,1,1,2,0,0,0,0,0,0,0,0,
    2,1,1,1,2,0,0,0,0,0,0,0,
    2,1,1,1,1,2,0,0,0,0,0,0,
    2,1,1,1,1,1,2,0,0,0,0,0,
    2,1,1,1,1,1,1,2,0,0,0,0,
    2,1,1,1,1,1,1,1,2,0,0,0,
    2,1,1,1,1,1,1,1,1,2,0,0,
    2,1,1,1,1,1,1,1,1,1,2,0,
    2,1,1,1,1,1,1,1,1,1,1,2,
    2,1,1,1,1,1,2,2,2,2,2,2,
    2,1,1,2,1,1,2,0,0,0,0,0,
    2,1,2,0,2,1,1,2,0,0,0,0,
    2,2,0,0,2,1,1,2,0,0,0,0,
    2,0,0,0,0,2,1,1,2,0,0,0,
    0,0,0,0,0,2,1,1,2,0,0,0,
    0,0,0,0,0,0,2,2,0,0,0,0,
];

// Pointing hand (hotspot near tip of index finger, top-leftish).

#[rustfmt::skip]
const CURSOR_HAND: [u8; (CUR_W * CUR_H) as usize] = [
    0,0,0,2,2,0,0,0,0,0,0,0,
    0,0,2,1,1,2,0,0,0,0,0,0,
    0,0,2,1,1,2,0,0,0,0,0,0,
    0,0,2,1,1,2,2,2,0,0,0,0,
    0,0,2,1,1,2,1,1,2,0,0,0,
    2,2,2,1,1,2,1,1,2,2,0,0,
    2,1,1,1,1,1,1,1,1,1,2,0,
    2,1,1,1,1,1,1,1,1,1,2,0,
    2,1,1,1,1,1,1,1,1,1,2,0,
    0,2,1,1,1,1,1,1,1,2,0,0,
    0,2,1,1,1,1,1,1,1,2,0,0,
    0,0,2,1,1,1,1,1,2,0,0,0,
    0,0,2,1,1,1,1,1,2,0,0,0,
    0,0,0,2,1,1,1,2,0,0,0,0,
    0,0,0,2,1,1,1,2,0,0,0,0,
    0,0,0,0,2,1,2,0,0,0,0,0,
    0,0,0,0,2,1,2,0,0,0,0,0,
    0,0,0,0,0,2,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
];

// I-beam for text fields.

#[rustfmt::skip]
const CURSOR_IBEAM: [u8; (CUR_W * CUR_H) as usize] = [
    2,2,2,0,0,2,2,2,0,0,0,0,
    0,0,0,2,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,2,2,2,0,0,0,0,0,0,
    2,2,2,0,0,2,2,2,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
];

// Hourglass-ish wait (loading).

#[rustfmt::skip]
const CURSOR_WAIT: [u8; (CUR_W * CUR_H) as usize] = [
    2,2,2,2,2,2,2,2,0,0,0,0,
    2,1,1,1,1,1,1,2,0,0,0,0,
    0,2,1,1,1,1,2,0,0,0,0,0,
    0,0,2,1,1,2,0,0,0,0,0,0,
    0,0,0,2,2,0,0,0,0,0,0,0,
    0,0,0,2,2,0,0,0,0,0,0,0,
    0,0,2,1,1,2,0,0,0,0,0,0,
    0,2,1,1,1,1,2,0,0,0,0,0,
    2,1,1,1,1,1,1,2,0,0,0,0,
    2,2,2,2,2,2,2,2,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
];

/// Built-in fixed 12×19 sprite for a shape (index bitmap: 0/1/2).
fn cursor_builtin(shape: CursorShape) -> &'static [u8; (CUR_W * CUR_H) as usize] {
    match shape {
        CursorShape::Hand => &CURSOR_HAND,
        CursorShape::IBeam => &CURSOR_IBEAM,
        CursorShape::Wait => &CURSOR_WAIT,
        CursorShape::Arrow => &CURSOR_ARROW,
    }
}

/// A theme-supplied cursor bitmap: `w`×`h` index values (0 transparent / 1 fill
/// / 2 outline), same encoding as the built-ins. Dimensions are capped at
/// [`CUR_MAX`] per side so the save buffer stays bounded.
#[derive(Clone, Default)]
pub struct CursorSprite {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

/// Max cursor sprite dimension per side (bounds the RMW save patch).
pub const CUR_MAX: usize = 32;

/// Live cursor theme: fill/outline colours + optional per-shape custom sprites
/// (order Arrow, Hand, IBeam, Wait). `None` ⇒ brand defaults + built-in sprites.
struct CursorTheme {
    fill: Rgb,
    outline: Rgb,
    sprites: Option<[CursorSprite; 4]>,
}

/// Default cursor colours (brand): near-black fill, cream outline.
const CURSOR_FILL_DEFAULT: Rgb = (15, 15, 17);

const CURSOR_OUTLINE_DEFAULT: Rgb = (245, 245, 248);

static CURSOR_THEME: Locked<CursorTheme> = Locked::new(CursorTheme {
    fill: CURSOR_FILL_DEFAULT,
    outline: CURSOR_OUTLINE_DEFAULT,
    sprites: None,
});

/// Set the cursor fill/outline colours (from `ui.json`), preserving any custom
/// sprites already installed by a theme.
pub fn set_cursor_colors(fill: Rgb, outline: Rgb) {
    CURSOR_THEME.with(|t| {
        t.fill = fill;
        t.outline = outline;
    });
}

/// Install (or clear, with `None`) the per-shape custom cursor sprites (from a
/// theme file), preserving the current colours. A sprite with zero dims falls
/// back to the built-in bitmap for that shape.
pub fn set_cursor_sprites(sprites: Option<[CursorSprite; 4]>) {
    CURSOR_THEME.with(|t| t.sprites = sprites);
}

/// Reset the cursor to brand colours + built-in sprites.
pub fn reset_cursor_theme() {
    CURSOR_THEME.with(|t| {
        t.fill = CURSOR_FILL_DEFAULT;
        t.outline = CURSOR_OUTLINE_DEFAULT;
        t.sprites = None;
    });
}

/// Cached Font Awesome cursor sprites (Arrow/Hand/IBeam/Wait), built once the
/// first time each shape is drawn. Falls back to the hand-drawn bitmaps if FA
/// is not yet registered or rasterization fails.
static FA_CURSOR_CACHE: Locked<[Option<(u64, u64, Vec<u8>)>; 4]> =
    Locked::new([None, None, None, None]);

fn cursor_fa_sprite(shape: CursorShape) -> Option<(u64, u64, Vec<u8>)> {
    let i = shape as usize;
    FA_CURSOR_CACHE.with(|cache| {
        if let Some(hit) = &cache[i] {
            return Some(hit.clone());
        }
        let ch = crate::icons::cursor_glyph(shape as u8);
        // ~18 px matches the classic 12×19 hand-drawn arrow optical weight.
        let Some((w, h, data)) = crate::font_ttf::raster_cursor_sprite(ch, 18.0) else {
            return None;
        };
        if w == 0 || h == 0 || data.is_empty() {
            return None;
        }
        let entry = (w as u64, h as u64, data);
        cache[i] = Some(entry.clone());
        Some(entry)
    })
}

/// Resolve the active cursor for the current shape: `(w, h, index-data,
/// fill, outline)`. Order: theme custom sprite → Font Awesome Free Solid
/// (`arrow-pointer` / `hand-pointer` / `i-cursor` / `hourglass`) → hand-drawn
/// built-in bitmap.
fn cursor_active() -> (u64, u64, alloc::borrow::Cow<'static, [u8]>, Rgb, Rgb) {
    let shape = cursor_shape();
    CURSOR_THEME.with(|t| {
        let (fill, outline) = (t.fill, t.outline);
        if let Some(sprites) = &t.sprites {
            let sp = &sprites[shape as usize];
            if sp.w > 0 && sp.h > 0 && sp.data.len() >= sp.w * sp.h {
                return (
                    sp.w as u64,
                    sp.h as u64,
                    alloc::borrow::Cow::Owned(sp.data.clone()),
                    fill,
                    outline,
                );
            }
        }
        if let Some((w, h, data)) = cursor_fa_sprite(shape) {
            return (w, h, alloc::borrow::Cow::Owned(data), fill, outline);
        }
        (
            CUR_W,
            CUR_H,
            alloc::borrow::Cow::Borrowed(&cursor_builtin(shape)[..]),
            fill,
            outline,
        )
    })
}

/// Move the mouse cursor sprite to `(x, y)` (erasing it from the old spot).
pub fn cursor_move(x: u64, y: u64) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_active = true;
            sc.cur_x = x.min(sc.width.saturating_sub(1));
            sc.cur_y = y.min(sc.height.saturating_sub(1));
            sc.cursor_draw();
        }
    });
}

/// Redraw the cursor sprite at its current position (after a content redraw
/// erased it). Safe no-op if the console isn't up.
pub fn cursor_move_here() {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            if sc.cur_active {
                sc.cursor_draw();
            }
        }
    });
}

impl Screen {
    /// Restore the patch saved beneath the cursor (erasing the sprite). Uses the
    /// dims the patch was *saved* at (`cur_sw`×`cur_sh`), which may differ from
    /// the current shape's sprite after a theme/shape change.
    pub(super) fn cursor_restore(&self) {
        if !self.cur_vis {
            return;
        }
        let (w, h) = (self.cur_sw, self.cur_sh);
        for dy in 0..h {
            for dx in 0..w {
                let i = (dy * w + dx) as usize;
                if i < self.cur_saved.len() {
                    self.put_pixel(self.cur_x + dx, self.cur_y + dy, self.cur_saved[i]);
                }
            }
        }
    }

    /// Save the framebuffer under the cursor and draw the active shape sprite
    /// (theme-driven colours + optional custom, variable-size bitmap).
    fn cursor_draw(&mut self) {
        let (w, h, data, fill, outline) = cursor_active();
        let n = (w * h) as usize;
        self.cur_saved.clear();
        self.cur_saved.reserve(n);
        for dy in 0..h {
            for dx in 0..w {
                self.cur_saved.push(self.get_pixel(self.cur_x + dx, self.cur_y + dy));
            }
        }
        self.cur_sw = w;
        self.cur_sh = h;
        for dy in 0..h {
            for dx in 0..w {
                match data[(dy * w + dx) as usize] {
                    1 => self.put_pixel(self.cur_x + dx, self.cur_y + dy, fill),
                    2 => self.put_pixel(self.cur_x + dx, self.cur_y + dy, outline),
                    _ => {}
                }
            }
        }
        self.cur_vis = true;
    }

    /// Redraw the cursor on top after a content change (if the mouse is active).
    /// The caller must have already erased the old cursor (`cursor_restore`) so
    /// the freshly-saved background is clean.
    pub(super) fn cursor_overlay(&mut self) {
        if self.cur_active {
            self.cursor_draw();
        }
    }
}
