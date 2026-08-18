//! Pure layout math for the multi-pane UI: the shell (chat) pane on one side and
//! a resizable **grid** of action panes on the other. Unit-tested off hardware.
//!
//! The geometry is two independent steps so each is simple enough to pin with
//! tests: [`split_band`] divides the content region into `chat | band` (one gap,
//! driven by `chat_pct`), then [`layout_grid`] tiles the band into
//! `cols × rows` cells using permille track weights. Dragging any divider is
//! [`resize_tracks`] (grid) or [`band_divider_pct`] (shell|band), and both are
//! the exact inverses of the forward layout — a property the tests assert by
//! round-tripping.

use alloc::vec::Vec;

/// Total panes including the shell agent: 2 (default) ..= 9.
pub const MAX_PANES_MIN: u8 = 2;
pub const MAX_PANES_MAX: u8 = 9;
pub const MAX_PANES_DEFAULT: u8 = 2;

/// Clamp total pane count (shell + action columns).
pub fn clamp_max_panes(n: u64) -> u8 {
    n.clamp(MAX_PANES_MIN as u64, MAX_PANES_MAX as u64) as u8
}

/// Number of action panes for a total pane budget (`max_panes - 1`).
pub fn action_column_count(max_panes: u8) -> usize {
    let m = clamp_max_panes(max_panes as u64);
    (m as usize).saturating_sub(1).max(1)
}

/// Whether the action band should be visible (not parked).
///
/// * `max_panes == 2`: only when at least one action tab is open (today's collapse).
/// * `max_panes > 2`: always show the grid so empty cells stay drop targets.
pub fn action_band_visible(max_panes: u8, any_tab_open: bool) -> bool {
    if clamp_max_panes(max_panes as u64) > 2 {
        true
    } else {
        any_tab_open
    }
}

/// Pure focus-cycle math for Ctrl+Tab: slots are **shell chat** then each
/// visible action column (row-major). Returns `(to_action, action_index)`.
///
/// The shell is part of the ring so focus can always return to the composer —
/// without this, cycling only among action panes traps the keyboard. Unit-
/// tested here because `framebuffer` is gated out of the test binary.
pub fn cycle_focus_target(
    visible: &[usize],
    at_action: bool,
    focused_action: usize,
    forward: bool,
) -> (bool, usize) {
    if visible.is_empty() {
        return (false, focused_action);
    }
    // Virtual list: index 0 = chat, 1.. = visible[i-1].
    let n = visible.len() + 1;
    let cur = if !at_action {
        0
    } else {
        visible
            .iter()
            .position(|&i| i == focused_action)
            .map(|p| p + 1)
            .unwrap_or(1)
    };
    let next = if forward {
        (cur + 1) % n
    } else {
        (cur + n - 1) % n
    };
    if next == 0 {
        (false, focused_action)
    } else {
        (true, visible[next - 1])
    }
}

/// Smallest track size in pixels a resize drag may leave a pane at, so a
/// divider can never be dragged far enough to make a pane vanish (a zero-width
/// pane would reflow its whole scrollback to one column and be unreachable).
pub const MIN_TRACK_PX: u64 = 64;

/// Weights are permille of the band, so all the split math is exact integer
/// arithmetic and a saved layout restores byte-identically.
pub const WEIGHT_TOTAL: u64 = 1000;

/// A screen rectangle as `(x, y, w, h)`, matching what [`layout_grid`] yields.
pub type Rect = (u64, u64, u64, u64);

/// Which edge of the desktop the OS status bar occupies.
///
/// Everything else — the shell pane, the action grid, the gutters — lays out
/// inside the [`status_split`] content rect, so moving the bar moves the whole UI
/// rather than overlapping it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum StatusPos {
    /// Default: bar along the top edge (keyboard-first layout keeps chrome above content).
    #[default]
    Top,
    Bottom,
    Left,
    Right,
}

impl StatusPos {
    /// True for the edges that make the bar a **column**. This is the one
    /// distinction the renderer needs: a column cannot run text across it, so its
    /// content stacks as rows instead (see [`status_segments`]).
    pub fn vertical(self) -> bool {
        matches!(self, StatusPos::Left | StatusPos::Right)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            StatusPos::Top => "top",
            StatusPos::Bottom => "bottom",
            StatusPos::Left => "left",
            StatusPos::Right => "right",
        }
    }

    /// Parse a config or command value.
    ///
    /// `None` for anything unrecognised, so a typo in `ui.json` leaves the bar
    /// where it is instead of silently moving it somewhere the user didn't ask
    /// for. Case- and whitespace-insensitive.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            s if s.eq_ignore_ascii_case("top") => Some(StatusPos::Top),
            s if s.eq_ignore_ascii_case("bottom") => Some(StatusPos::Bottom),
            s if s.eq_ignore_ascii_case("left") => Some(StatusPos::Left),
            s if s.eq_ignore_ascii_case("right") => Some(StatusPos::Right),
            _ => None,
        }
    }
}

/// Width of a **vertical** status bar, in text columns.
///
/// Deliberately fixed rather than fitted to the longest segment: the segments
/// hold live values (a clock, an IP, a battery percentage), so a fitted width
/// would relayout every pane — reflowing scrollback — each time one of them
/// changed a digit. 18 columns holds icon + label rows and `255.255.255.255`.
pub const STATUS_V_COLS: u64 = 18;

/// For a **vertical** bar: one field per row (split on every space), then wrap
/// any field that is still wider than `cols`. Top→bottom reading order — not
/// the horizontal bar's left/right half layout.
pub fn status_lines_vertical(s: &str, cols: usize) -> Vec<alloc::string::String> {
    let mut out = alloc::vec::Vec::new();
    for word in s.split_whitespace() {
        if word.is_empty() {
            continue;
        }
        for row in wrap_segment(word, cols) {
            out.push(crate::textsel::ellipsize(row, cols));
        }
    }
    out
}

/// Content rows a **centred modal** can show on a `height`-px screen with
/// `ch`-px cells, given the box's total vertical `frame` (border + padding, top
/// and bottom). The title and separator rows are deducted, and a cell of margin
/// is kept top and bottom so the box never touches the screen edges.
///
/// Lives here rather than in the compositor because the compositor is
/// `#[cfg(not(test))]`, and this is the arithmetic that decides whether a dialog
/// **fits**: an over-tall box made `framebuffer`'s centring subtraction wrap, so
/// every draw landed off-screen and an approval modal painted *nothing* while
/// still waiting for a keypress — a consent prompt the human cannot read, and
/// indistinguishable from a frozen shell. Nothing visual would have caught it;
/// a test on this function does.
pub fn modal_max_rows(height: u64, ch: u64, frame: u64) -> u64 {
    if ch == 0 {
        return 1;
    }
    let usable = height.saturating_sub(frame).saturating_sub(2 * ch);
    (usable / ch).saturating_sub(2).max(1)
}

/// Trim `lines` to `budget` rows, replacing the tail with a count of what was
/// dropped. Truncation is **stated**, never silent: a dialog that quietly hides
/// half of what it is asking about is worse than one that admits the payload is
/// too long to show.
pub fn clamp_modal_lines(
    mut lines: Vec<alloc::string::String>,
    budget: usize,
) -> Vec<alloc::string::String> {
    if budget == 0 {
        return Vec::new();
    }
    if lines.len() <= budget {
        return lines;
    }
    let dropped = lines.len() - (budget - 1);
    lines.truncate(budget - 1);
    lines.push(alloc::format!("... {dropped} more line(s) not shown"));
    lines
}

/// Carve the status bar off one edge of a `w × h` desktop, returning
/// `(bar, content)`.
///
/// `thickness` is the bar's height for [`StatusPos::Top`]/[`StatusPos::Bottom`]
/// and its **width** for [`StatusPos::Left`]/[`StatusPos::Right`]. It is clamped
/// to half the span it consumes: a vertical bar is many times thicker than a
/// horizontal one, and on a small desktop an unclamped one would leave the panes
/// no room at all (a zero-width pane reflows its whole scrollback to one column,
/// the same failure `MIN_TRACK_PX` guards against).
pub fn status_split(w: u64, h: u64, pos: StatusPos, thickness: u64) -> (Rect, Rect) {
    match pos {
        StatusPos::Top => {
            let t = thickness.min(h / 2);
            ((0, 0, w, t), (0, t, w, h - t))
        }
        StatusPos::Bottom => {
            let t = thickness.min(h / 2);
            ((0, h - t, w, t), (0, 0, w, h - t))
        }
        StatusPos::Left => {
            let t = thickness.min(w / 2);
            ((0, 0, t, h), (t, 0, w - t, h))
        }
        StatusPos::Right => {
            let t = thickness.min(w / 2);
            ((w - t, 0, t, h), (0, 0, w - t, h))
        }
    }
}

/// Split a resolved status template into the groups it was written as.
///
/// The templates join *related* fields with one space and separate *groups* with
/// two or more (`"${kbd} ${mouse}  ${net}  ${mem}"`), so runs of 2+ spaces are
/// already the author's own grouping — exactly the unit a vertical bar needs to
/// stack. That means a re-themed template stacks sensibly with no extra syntax,
/// and a template with no double spaces yields one segment rather than nothing.
pub fn status_segments(s: &str) -> Vec<&str> {
    s.split("  ").map(str::trim).filter(|t| !t.is_empty()).collect()
}

/// Wrap one status segment onto `cols`-wide rows, breaking at spaces.
///
/// A vertical bar is narrow enough that the most-read field does not fit:
/// `${datetime} ${tz}` resolves to `Mon 2026-07-27 05:59:12 UTC`, and ellipsizing
/// it to 16 columns cuts off the **time** — the part anyone actually looks at. So a
/// long segment takes as many rows as it needs instead of losing its tail.
///
/// A single word wider than `cols` is emitted whole for the caller to ellipsize:
/// breaking mid-token would turn one unreadable value into two.
pub fn wrap_segment(seg: &str, cols: usize) -> Vec<&str> {
    let mut out = Vec::new();
    if cols == 0 {
        return out;
    }
    let (mut start, mut end) = (0usize, 0usize); // byte range of the current row
    for word in seg.split_whitespace() {
        let w_start = word.as_ptr() as usize - seg.as_ptr() as usize;
        let w_end = w_start + word.len();
        if end == start {
            (start, end) = (w_start, w_end); // first word of a row always fits
            continue;
        }
        // Would appending "<space><word>" overflow the row?
        if seg[start..w_end].chars().count() <= cols {
            end = w_end;
        } else {
            out.push(&seg[start..end]);
            (start, end) = (w_start, w_end);
        }
    }
    if end > start {
        out.push(&seg[start..end]);
    }
    out
}

/// Which divider a pointer grabbed. There is one shell|band divider plus a
/// divider between each pair of adjacent grid tracks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Divider {
    /// The shell | action-band split (moves `chat_pct`).
    Band,
    /// Between action grid column `i` and `i + 1`.
    Col(usize),
    /// Between action grid row `i` and `i + 1`.
    Row(usize),
}

/// The action band's grid shape and track weights.
///
/// Panes are addressed **row-major**: `index = row * cols + col`, so column 0 of
/// row 0 is pane 0 and the pane order matches reading order.
#[derive(Clone, PartialEq, Debug)]
pub struct GridSpec {
    pub cols: usize,
    pub rows: usize,
    /// Per-column widths in permille (len == `cols`).
    pub col_w: Vec<u64>,
    /// Per-row heights in permille (len == `rows`).
    pub row_h: Vec<u64>,
}

impl GridSpec {
    /// An evenly-divided `cols × rows` grid.
    pub fn even(cols: usize, rows: usize) -> GridSpec {
        let (cols, rows) = clamp_grid(cols, rows);
        GridSpec {
            cols,
            rows,
            col_w: even_weights(cols),
            row_h: even_weights(rows),
        }
    }

    /// Number of action panes in the grid.
    pub fn len(&self) -> usize {
        self.cols * self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Repair a spec loaded from disk: clamp the shape and make the weight
    /// vectors the right length and non-degenerate. A hand-edited or
    /// truncated `panes.json` must not be able to produce a zero-width pane.
    pub fn sanitized(&self) -> GridSpec {
        let (cols, rows) = clamp_grid(self.cols, self.rows);
        GridSpec {
            cols,
            rows,
            col_w: sanitize_weights(&self.col_w, cols),
            row_h: sanitize_weights(&self.row_h, rows),
        }
    }
}

/// Clamp a grid shape so it has at least one cell and at most
/// [`MAX_PANES_MAX`] − 1 action panes.
pub fn clamp_grid(cols: usize, rows: usize) -> (usize, usize) {
    let max_cells = (MAX_PANES_MAX - 1) as usize;
    let mut c = cols.clamp(1, max_cells);
    let r = rows.clamp(1, max_cells);
    // Shrink the columns (not the rows the user just asked for) until the cell
    // count fits the budget — dropping rows instead would silently discard the
    // dimension being set.
    while c * r > max_cells && c > 1 {
        c -= 1;
    }
    (c, r.min(max_cells / c.max(1)).max(1))
}

/// Even permille weights for `n` tracks, with the remainder on the last so they
/// sum to exactly [`WEIGHT_TOTAL`].
pub fn even_weights(n: usize) -> Vec<u64> {
    let n = n.max(1);
    let each = WEIGHT_TOTAL / n as u64;
    let mut v = alloc::vec![each; n];
    let used = each * n as u64;
    if let Some(last) = v.last_mut() {
        *last += WEIGHT_TOTAL - used;
    }
    v
}

/// Coerce `w` to exactly `n` positive weights summing to [`WEIGHT_TOTAL`].
fn sanitize_weights(w: &[u64], n: usize) -> Vec<u64> {
    let n = n.max(1);
    if w.len() != n || w.iter().any(|&x| x == 0) || w.iter().sum::<u64>() == 0 {
        return even_weights(n);
    }
    let sum: u64 = w.iter().sum();
    if sum == WEIGHT_TOTAL {
        return w.to_vec();
    }
    // Rescale to the canonical total.
    let mut out: Vec<u64> = w.iter().map(|&x| (x * WEIGHT_TOTAL / sum).max(1)).collect();
    let got: u64 = out.iter().sum();
    if let Some(last) = out.last_mut() {
        *last = last.saturating_add(WEIGHT_TOTAL).saturating_sub(got).max(1);
    }
    out
}

/// How many tracks of at least `min_px` fit in `total_px` (with a `gap` between
/// each adjacent pair), capped at `requested`.
///
/// A grid asked for more rows than the band is tall would give each pane less
/// height than its own title header, and its interior — which `Pane::new` floors
/// at one text row — would then extend past the cell and paint over the pane
/// below. Clamping at the point the shape is *set* keeps the reported shape and
/// the drawn shape identical.
pub fn fit_tracks(total_px: u64, gap: u64, requested: usize, min_px: u64) -> usize {
    let requested = requested.max(1);
    let min_px = min_px.max(1);
    if total_px == 0 {
        return requested; // geometry not known yet — don't clamp on no information
    }
    let mut n = requested;
    while n > 1 && n as u64 * min_px + (n as u64 - 1) * gap > total_px {
        n -= 1;
    }
    n
}

/// The most balanced `cols × rows` grid holding exactly `n` action panes.
///
/// Prefers wider than tall (screens are wide, and a pane's tab bar + frame eat
/// vertical space), and keeps rows ≤ cols. A prime `n` has no balanced
/// factorisation, so it degrades to a single row — which is the honest answer
/// rather than silently changing the pane count the user asked for.
pub fn grid_for_count(n: usize) -> (usize, usize) {
    let n = n.clamp(1, (MAX_PANES_MAX - 1) as usize);
    let mut best = (n, 1);
    for rows in 1..=n {
        if n % rows != 0 {
            continue;
        }
        let cols = n / rows;
        if rows <= cols && rows > best.1 {
            best = (cols, rows);
        }
    }
    best
}

/// Split the content region into the shell box and the action band.
///
/// Layout is `[outer][chat][gap][band][outer]` (or the two swapped), so there is
/// exactly **one** gap here regardless of the grid shape — the grid's internal
/// gaps are [`layout_grid`]'s business.
///
/// Returns `(chat_x, chat_w, band_x, band_w)`. With `band_open == false` the
/// chat takes the full width and the band is returned zero-width.
pub fn split_band(
    width: u64,
    outer: u64,
    gap: u64,
    chat_pct: u64,
    band_open: bool,
    swap: bool,
) -> (u64, u64, u64, u64) {
    let full_w = width.saturating_sub(2 * outer);
    if !band_open || full_w == 0 {
        return (outer, full_w, width, 0);
    }
    let avail = width.saturating_sub(2 * outer + gap);
    if avail == 0 {
        return (outer, full_w, width, 0);
    }
    let chat_w = (avail * chat_pct.clamp(10, 90) / 100).max(1);
    let band_w = avail.saturating_sub(chat_w);
    if swap {
        (outer + band_w + gap, chat_w, outer, band_w)
    } else {
        (outer, chat_w, outer + chat_w + gap, band_w)
    }
}

/// Chat width percentage for the shell|band divider dragged to pixel `x`.
/// The inverse of [`split_band`].
pub fn band_divider_pct(width: u64, outer: u64, gap: u64, swap: bool, x: u64) -> u64 {
    let avail = width.saturating_sub(2 * outer + gap).max(1);
    let chat_w = if swap {
        // Swapped, the chat box begins one gap right of the divider.
        width.saturating_sub(outer + gap).saturating_sub(x)
    } else {
        x.saturating_sub(outer)
    }
    .min(avail);
    (chat_w * 100 / avail).clamp(10, 90)
}

/// Pixel sizes of `weights.len()` tracks filling `total_px`, minus one `gap`
/// between each adjacent pair.
///
/// The last track absorbs the division remainder so the tracks plus gaps fill
/// `total_px` exactly and no gutter is left at the far edge.
pub fn track_sizes(total_px: u64, gap: u64, weights: &[u64]) -> Vec<u64> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }
    let gaps = gap * (n as u64 - 1);
    let content = total_px.saturating_sub(gaps);
    let sum: u64 = weights.iter().sum::<u64>().max(1);
    let mut out: Vec<u64> = weights.iter().map(|&w| content * w / sum).collect();
    let used: u64 = out.iter().sum();
    if let Some(last) = out.last_mut() {
        *last = last.saturating_add(content.saturating_sub(used));
    }
    out
}

/// Track start offsets (from the band origin) for the sizes of `weights`.
pub fn track_offsets(total_px: u64, gap: u64, weights: &[u64]) -> Vec<u64> {
    let sizes = track_sizes(total_px, gap, weights);
    let mut out = Vec::with_capacity(sizes.len());
    let mut at = 0u64;
    for s in sizes {
        out.push(at);
        at += s + gap;
    }
    out
}

/// Grid cell rectangles `(x, y, w, h)` in **row-major** order for a band at
/// `(band_x, band_y, band_w, band_h)`.
pub fn layout_grid(
    band_x: u64,
    band_y: u64,
    band_w: u64,
    band_h: u64,
    gap: u64,
    spec: &GridSpec,
) -> Vec<(u64, u64, u64, u64)> {
    let spec = spec.sanitized();
    let cw = track_sizes(band_w, gap, &spec.col_w);
    let cx = track_offsets(band_w, gap, &spec.col_w);
    let rh = track_sizes(band_h, gap, &spec.row_h);
    let ry = track_offsets(band_h, gap, &spec.row_h);
    let mut out = Vec::with_capacity(spec.len());
    for r in 0..spec.rows {
        for c in 0..spec.cols {
            out.push((band_x + cx[c], band_y + ry[r], cw[c], rh[r]));
        }
    }
    out
}

/// Move the boundary between track `i` and `i + 1` to `pos_px` (measured from
/// the band origin), adjusting **only that pair**: every other track keeps its
/// exact pixel size, which is what makes a drag feel local rather than
/// rebalancing the whole band.
///
/// Returns false when the move is impossible (bad index, or the pair has no room
/// for two panes at [`MIN_TRACK_PX`]).
pub fn resize_tracks(
    weights: &mut [u64],
    i: usize,
    total_px: u64,
    gap: u64,
    pos_px: u64,
) -> bool {
    if i + 1 >= weights.len() {
        return false;
    }
    let pair_w = weights[i] + weights[i + 1];
    let sum: u64 = weights.iter().sum();
    if pair_w == 0 || sum == 0 {
        return false;
    }
    let content = total_px.saturating_sub(gap * (weights.len() as u64 - 1));
    if content == 0 {
        return false;
    }
    // Work in **weights**, not pixels. `track_sizes` derives a track's pixels from
    // the *global* weight sum (`content * w / sum`), so converting a pixel target
    // through the dragged pair's own px:weight ratio is biased — the two ratios
    // differ, and the bias is enough to land a track one pixel under the minimum.
    //
    // Expressing the minimum as a weight and rounding **up** makes it exact:
    // `min_w = ceil(MIN * sum / content)` implies `content * min_w / sum >= MIN`,
    // so the flooring inside `track_sizes` cannot cross the minimum. (The last
    // track also absorbs the division remainder, which only ever makes it
    // larger.)
    let min_w = (MIN_TRACK_PX * sum).div_ceil(content).max(1);
    if pair_w < 2 * min_w {
        return false; // no room for two panes either side of the divider
    }
    let offs = track_offsets(total_px, gap, weights);
    let want_px = pos_px.saturating_sub(offs[i]);
    let want_w = (want_px * sum).div_ceil(content);
    // The pair's combined weight is conserved, so every other track keeps its
    // exact size and the weights stay normalised across repeated drags.
    let wi = want_w.clamp(min_w, pair_w - min_w);
    weights[i] = wi;
    weights[i + 1] = pair_w - wi;
    true
}

/// Where a tab pulled from `(from_pane, from_idx)` lands when dropped at
/// `to_idx` in a destination list that is `to_len` long **after** the source tab
/// was removed. `to_idx` is an index into the list the user saw — i.e. before
/// the removal.
///
/// The fiddly part of a tab move: within one pane the removal shifts every later
/// index down by one, so a drop *after* the original slot must be decremented.
/// The shift has to happen **before** the clamp — clamping first collapses a
/// drop-at-the-end onto the last index, which the decrement then pulls one
/// further, and the tab lands second-to-last instead of last.
pub fn insert_index(
    from_pane: usize,
    from_idx: usize,
    to_pane: usize,
    to_len: usize,
    to_idx: usize,
) -> usize {
    let mut insert = to_idx;
    if from_pane == to_pane && from_idx < insert {
        insert = insert.saturating_sub(1);
    }
    insert.min(to_len)
}

/// Total pane budget from a `panes.json` pair, or `None` when neither key is
/// present (leave the live layout alone).
///
/// `max_panes` is the current key and wins. Legacy `num_action_panes` counted
/// **action panes**, so it maps to `n + 1` totals — that back-compat reading is
/// why an existing config's `1` still means today's 2-pane layout.
pub fn max_panes_from_cfg(max_panes: Option<i64>, num_action_panes: Option<i64>) -> Option<u8> {
    if let Some(m) = max_panes {
        return Some(clamp_max_panes(m.max(0) as u64));
    }
    let n = num_action_panes?;
    let n = n.clamp(1, (MAX_PANES_MAX - 1) as i64) as u64;
    Some(clamp_max_panes(n + 1))
}

/// Move a tab from `(from_pane, from_idx)` to `to_pane` at insert index
/// `to_idx` (clamped). Pure tab-list surgery for unit tests + compositor.
///
/// Returns false if indices are invalid or `from_pane == to_pane` with a no-op
/// identity move that still reorders when indices differ.
pub fn move_tab<T>(
    panes: &mut [Vec<T>],
    from_pane: usize,
    from_idx: usize,
    to_pane: usize,
    to_idx: usize,
) -> bool {
    if from_pane >= panes.len() || to_pane >= panes.len() {
        return false;
    }
    if from_idx >= panes[from_pane].len() {
        return false;
    }
    let mode = panes[from_pane].remove(from_idx);
    let insert = insert_index(from_pane, from_idx, to_pane, panes[to_pane].len(), to_idx);
    panes[to_pane].insert(insert, mode);
    true
}

/// Drop-shadow geometry for an elevated box, in pixels: how far the blur
/// reaches past the box, how far the shadow is pushed down (the light is above,
/// so the bottom edge carries more of it than the sides), and the darkening at
/// the shadow's own edge as a 0..=255 alpha.
///
/// Derived from the font scale so a shadow is the same *visual* weight on a
/// 1024x768 panel and a 2560x1440 one.
pub struct ShadowGeom {
    pub blur: u64,
    pub offset: u64,
    pub peak: u32,
}

/// The house drop shadow at font scale `scale`.
pub fn shadow_geom(scale: u64) -> ShadowGeom {
    let s = scale.max(1);
    ShadowGeom { blur: 6 * s, offset: 2 * s, peak: 104 }
}

/// Distance from `p` to the span `lo..hi` (0 when inside) — one axis of the
/// distance from a point to a rectangle.
pub fn span_dist(p: u64, lo: u64, hi: u64) -> u64 {
    if p < lo {
        lo - p
    } else if p >= hi {
        p - hi + 1
    } else {
        0
    }
}

/// The 1-D shadow falloff: full weight (255) on the shadow rect, easing to 0
/// `blur` px outside it.
///
/// Quadratic rather than linear because a linear ramp still reads as a *band*
/// — the eye finds the constant-slope edge — which is exactly what the two
/// hard-stepped rectangles this replaced looked like. Integer-only, like
/// `aa_coverage`, so it is FPU-independent.
pub fn shadow_falloff(d: u64, blur: u64) -> u32 {
    if blur == 0 || d >= blur {
        return 0;
    }
    let t = 255 - (d * 255 / blur) as u32; // 255 at the edge, 0 at `blur`
    t * t / 255
}

/// Shadow alpha at a point `(dx, dy)` px from the shadow rect, 0..=255.
///
/// Separable — the product of the two axes' falloffs — which is how a gaussian
/// box-shadow actually behaves, and it rounds the corners for free.
pub fn shadow_alpha(dx: u64, dy: u64, blur: u64, peak: u32) -> u32 {
    let (fx, fy) = (shadow_falloff(dx, blur), shadow_falloff(dy, blur));
    peak * fx / 255 * fy / 255
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A modal's box must fit the screen it is drawn on, at every font scale.
    /// `framebuffer::modal_box` centres with `height - bh`, so a box taller than
    /// the screen wrapped that subtraction and put every draw off-screen: the
    /// approval dialog painted **nothing** while still waiting for a key — a
    /// consent prompt the human cannot read, looking exactly like a hang. The
    /// trigger was ordinary: an `Edit` whose args were a whole config file.
    #[test_case]
    fn modal_row_budget_keeps_the_box_on_screen() {
        // The compositor's frame: 2 * (BORDER + PAD) = 2 * (2 + 10).
        const FRAME: u64 = 24;
        for height in [480u64, 600, 720, 900, 1080, 1440, 2160] {
            for ch in [16u64, 32, 48, 64] {
                let rows = modal_max_rows(height, ch, FRAME);
                assert!(rows >= 1, "always room for one row ({height}, {ch})");
                // The box `modal_box` builds for that many rows (`rows + 2` for
                // the title + separator).
                let bh = (rows + 2) * ch + FRAME;
                assert!(bh <= height, "box {bh} > screen {height} at ch={ch}");
                // …and is not needlessly small: one more row would encroach on
                // the cell of margin reserved at each edge (2 * ch).
                let bh_next = (rows + 3) * ch + FRAME;
                assert!(
                    bh_next + 2 * ch > height,
                    "budget {rows} too conservative for {height}/{ch}"
                );
            }
        }
        // A degenerate cell height must not divide by zero.
        assert_eq!(modal_max_rows(900, 0, 24), 1);
        // A screen too small for any box still reports a usable row.
        assert_eq!(modal_max_rows(8, 16, 24), 1);
    }

    /// Over-long content is truncated *and says so*.
    #[test_case]
    fn clamp_modal_lines_states_what_it_dropped() {
        let mk = |n: usize| -> Vec<alloc::string::String> {
            (0..n).map(|i| alloc::format!("line {i}")).collect()
        };
        // Fits: untouched.
        let out = clamp_modal_lines(mk(5), 10);
        assert_eq!(out.len(), 5);
        assert_eq!(out[4], "line 4");
        // Exactly at budget: untouched (no spurious marker).
        let out = clamp_modal_lines(mk(10), 10);
        assert_eq!(out.len(), 10);
        assert!(!out[9].contains("more line"), "got {:?}", out[9]);
        // Over budget: clamped to the budget, last row is the count, and the
        // number accounts for the row the marker itself took.
        let out = clamp_modal_lines(mk(100), 10);
        assert_eq!(out.len(), 10, "never exceeds the budget");
        assert_eq!(out[8], "line 8");
        assert!(out[9].contains("91 more line(s)"), "got {:?}", out[9]);
        // A zero budget draws nothing rather than panicking.
        assert!(clamp_modal_lines(mk(3), 0).is_empty());
    }

    #[test_case]
    fn clamp_max_panes_range() {
        assert_eq!(clamp_max_panes(0), 2);
        assert_eq!(clamp_max_panes(1), 2);
        assert_eq!(clamp_max_panes(2), 2);
        assert_eq!(clamp_max_panes(5), 5);
        assert_eq!(clamp_max_panes(9), 9);
        assert_eq!(clamp_max_panes(99), 9);
        assert_eq!(action_column_count(2), 1);
        assert_eq!(action_column_count(9), 8);
    }

    #[test_case]
    fn action_band_visible_rules() {
        assert!(!action_band_visible(2, false));
        assert!(action_band_visible(2, true));
        assert!(action_band_visible(3, false));
        assert!(action_band_visible(9, false));
    }


    // --- band + grid geometry -------------------------------------------

    #[test_case]
    fn split_band_matches_the_classic_two_pane_split() {
        // width 1000, outer 8, gap 10, 56% chat → avail = 1000-16-10 = 974
        let (cx, cw, bx, bw) = split_band(1000, 8, 10, 56, true, false);
        assert_eq!(cx, 8);
        assert_eq!(cw, 974 * 56 / 100);
        assert_eq!(bx, cx + cw + 10);
        assert_eq!(bx + bw + 8, 1000, "band fills to the right margin");
        assert!(cw > bw, "default chat is the wider band");
    }

    #[test_case]
    fn split_band_swapped_puts_the_band_first() {
        let (cx, cw, bx, bw) = split_band(1000, 8, 10, 56, true, true);
        assert_eq!(bx, 8, "band starts at the left margin");
        assert_eq!(cx, bx + bw + 10);
        assert_eq!(cx + cw + 8, 1000);
    }

    #[test_case]
    fn split_band_closed_gives_chat_the_full_width() {
        let (cx, cw, _bx, bw) = split_band(800, 8, 10, 56, false, false);
        assert_eq!((cx, cw), (8, 800 - 16));
        assert_eq!(bw, 0, "band parked");
    }

    #[test_case]
    fn band_divider_pct_round_trips_split_band() {
        for &width in &[640u64, 1000, 1600, 3840] {
            for pct in [10u64, 33, 56, 90] {
                for swap in [false, true] {
                    let (cx, cw, _bx, _bw) = split_band(width, 8, 10, pct, true, swap);
                    // The divider's left edge: right of the chat box unswapped,
                    // one gap left of it when swapped.
                    let x = if swap { cx.saturating_sub(10) } else { cx + cw };
                    let got = band_divider_pct(width, 8, 10, swap, x);
                    assert!(
                        got.abs_diff(pct) <= 1,
                        "w={} pct={} swap={} got={}",
                        width,
                        pct,
                        swap,
                        got
                    );
                }
            }
        }
    }

    #[test_case]
    fn track_sizes_fill_exactly_with_gaps() {
        for n in 1..=8usize {
            for &total in &[200u64, 511, 1024, 2000] {
                let w = even_weights(n);
                let sizes = track_sizes(total, 10, &w);
                assert_eq!(sizes.len(), n);
                let sum: u64 = sizes.iter().sum();
                assert_eq!(
                    sum + 10 * (n as u64 - 1),
                    total,
                    "n={} total={} left a gutter",
                    n,
                    total
                );
            }
        }
    }

    #[test_case]
    fn track_offsets_are_contiguous() {
        let w = even_weights(4);
        let sizes = track_sizes(1000, 10, &w);
        let offs = track_offsets(1000, 10, &w);
        assert_eq!(offs[0], 0);
        for i in 1..4 {
            assert_eq!(offs[i], offs[i - 1] + sizes[i - 1] + 10);
        }
    }

    #[test_case]
    fn layout_grid_is_row_major_and_disjoint() {
        let spec = GridSpec::even(3, 2);
        let cells = layout_grid(100, 50, 900, 600, 10, &spec);
        assert_eq!(cells.len(), 6);
        // Row-major: index = row * cols + col.
        assert_eq!(cells[0].1, cells[1].1, "row 0 shares a y");
        assert_eq!(cells[0].0, cells[3].0, "column 0 shares an x");
        assert!(cells[3].1 > cells[0].1, "row 1 is below row 0");
        // Every cell inside the band, no overlaps within a row or column.
        for &(x, y, w, h) in &cells {
            assert!(x >= 100 && x + w <= 1000, "x out of band");
            assert!(y >= 50 && y + h <= 650, "y out of band");
        }
        for r in 0..2 {
            for c in 1..3 {
                let prev = cells[r * 3 + c - 1];
                let cur = cells[r * 3 + c];
                assert!(cur.0 >= prev.0 + prev.2, "columns overlap");
            }
        }
    }

    #[test_case]
    fn layout_grid_single_cell_fills_the_band() {
        let cells = layout_grid(10, 20, 300, 400, 10, &GridSpec::even(1, 1));
        assert_eq!(cells, alloc::vec![(10, 20, 300, 400)]);
    }

    #[test_case]
    fn clamp_grid_caps_cells_at_the_pane_budget() {
        assert_eq!(clamp_grid(1, 1), (1, 1));
        assert_eq!(clamp_grid(4, 2), (4, 2)); // exactly 8 action panes
        assert_eq!(clamp_grid(0, 0), (1, 1));
        // Over budget: columns shrink so the requested row count survives.
        let (c, r) = clamp_grid(8, 3);
        assert!(c * r <= 8, "got {}x{}", c, r);
        assert_eq!(r, 3, "the dimension being set is kept");
        let (c, r) = clamp_grid(99, 99);
        assert!(c * r <= 8 && c >= 1 && r >= 1, "got {}x{}", c, r);
    }

    #[test_case]
    fn fit_tracks_clamps_to_what_the_band_can_host() {
        // 1000 px, 10 px gaps, 100 px minimum → at most 9 tracks; ask for 4 → 4.
        assert_eq!(fit_tracks(1000, 10, 4, 100), 4);
        // Ask for more than fits → clamped, never below one.
        assert_eq!(fit_tracks(1000, 10, 20, 100), 9);
        assert_eq!(fit_tracks(100, 10, 8, 100), 1);
        assert_eq!(fit_tracks(10, 10, 8, 100), 1);
        // Unknown geometry must not clamp (the screen isn't up yet).
        assert_eq!(fit_tracks(0, 10, 6, 100), 6);
        // Whatever it returns actually fits.
        for total in [200u64, 640, 1080, 1440, 2160] {
            for req in 1..=8usize {
                let n = fit_tracks(total, 10, req, 96);
                assert!(n >= 1 && n <= req);
                if n > 1 {
                    assert!(
                        n as u64 * 96 + (n as u64 - 1) * 10 <= total,
                        "total={} req={} n={} does not fit",
                        total,
                        req,
                        n
                    );
                }
            }
        }
    }

    #[test_case]
    fn grid_for_count_prefers_wide_balanced_shapes() {
        assert_eq!(grid_for_count(1), (1, 1));
        assert_eq!(grid_for_count(2), (2, 1));
        assert_eq!(grid_for_count(3), (3, 1)); // prime → one row
        assert_eq!(grid_for_count(4), (2, 2));
        assert_eq!(grid_for_count(6), (3, 2));
        assert_eq!(grid_for_count(8), (4, 2));
        // Every shape holds exactly the requested pane count.
        for n in 1..=8usize {
            let (c, r) = grid_for_count(n);
            assert_eq!(c * r, n, "n={} → {}x{}", n, c, r);
            assert!(r <= c, "n={} taller than wide", n);
        }
    }

    #[test_case]
    fn even_weights_sum_to_the_canonical_total() {
        for n in 1..=8usize {
            let w = even_weights(n);
            assert_eq!(w.len(), n);
            assert_eq!(w.iter().sum::<u64>(), WEIGHT_TOTAL, "n={}", n);
            assert!(w.iter().all(|&x| x > 0));
        }
    }

    #[test_case]
    fn sanitized_repairs_a_hand_edited_config() {
        // Wrong weight-vector length → evened out.
        let s = GridSpec { cols: 3, rows: 1, col_w: alloc::vec![500, 500], row_h: alloc::vec![] }
            .sanitized();
        assert_eq!(s.col_w, even_weights(3));
        assert_eq!(s.row_h, even_weights(1));
        // A zero weight would be a zero-width pane → evened out.
        let s = GridSpec { cols: 2, rows: 1, col_w: alloc::vec![1000, 0], row_h: alloc::vec![1000] }
            .sanitized();
        assert_eq!(s.col_w, even_weights(2));
        // A valid but unnormalised pair is rescaled, keeping the ratio.
        let s = GridSpec { cols: 2, rows: 1, col_w: alloc::vec![3, 1], row_h: alloc::vec![1000] }
            .sanitized();
        assert_eq!(s.col_w.iter().sum::<u64>(), WEIGHT_TOTAL);
        assert!(s.col_w[0] > s.col_w[1] * 2, "ratio preserved: {:?}", s.col_w);
    }

    // --- resizing --------------------------------------------------------

    #[test_case]
    fn resize_tracks_moves_the_boundary_to_the_cursor() {
        for n in 2..=4usize {
            for i in 0..n - 1 {
                let mut w = even_weights(n);
                let total = 1200u64;
                let offs = track_offsets(total, 10, &w);
                // Drag the boundary 40 px right of where it sits.
                let target = offs[i + 1].saturating_sub(10) + 40;
                assert!(resize_tracks(&mut w, i, total, 10, target));
                let sizes = track_sizes(total, 10, &w);
                let new_offs = track_offsets(total, 10, &w);
                let landed = new_offs[i] + sizes[i];
                assert!(
                    landed.abs_diff(target) <= 2,
                    "n={} i={} target={} landed={}",
                    n,
                    i,
                    target,
                    landed
                );
            }
        }
    }

    #[test_case]
    fn resize_tracks_leaves_every_other_track_alone() {
        // The whole point of per-gap resize: panes you did not touch must not move.
        let mut w = even_weights(4);
        let total = 1600u64;
        let before = track_sizes(total, 10, &w);
        let offs = track_offsets(total, 10, &w);
        assert!(resize_tracks(&mut w, 1, total, 10, offs[1] + 100));
        let after = track_sizes(total, 10, &w);
        assert_eq!(before[0], after[0], "track 0 moved");
        assert_eq!(before[3], after[3], "track 3 moved");
        assert_eq!(
            before[1] + before[2],
            after[1] + after[2],
            "the dragged pair changed total size"
        );
        assert!(after[1] < before[1], "dragged left, track 1 should shrink");
    }

    #[test_case]
    fn resize_tracks_never_collapses_a_pane() {
        // Dragging a divider past its neighbour must stop at the minimum, in
        // PIXELS — and the pixel target is converted to a weight by a truncating
        // division, so a clamp of exactly MIN lands just under it. Sweep sizes and
        // track counts, because whether the truncation crosses the minimum
        // depends on the pixels-per-weight-unit ratio.
        for &total in &[300u64, 512, 900, 1440, 2560, 3840] {
            for n in 2..=8usize {
                for i in 0..n - 1 {
                    for &to in &[0u64, total, 1] {
                        let mut w = even_weights(n);
                        if !resize_tracks(&mut w, i, total, 10, to) {
                            continue; // legitimately too small to split
                        }
                        let sizes = track_sizes(total, 10, &w);
                        assert!(
                            sizes[i] >= MIN_TRACK_PX,
                            "total={} n={} i={} to={} collapsed to {}",
                            total, n, i, to, sizes[i]
                        );
                        assert!(
                            sizes[i + 1] >= MIN_TRACK_PX,
                            "total={} n={} i={} to={} neighbour collapsed to {}",
                            total, n, i, to, sizes[i + 1]
                        );
                        assert_eq!(
                            w.iter().sum::<u64>(),
                            WEIGHT_TOTAL,
                            "weights left unnormalised (total={} n={})",
                            total, n
                        );
                        assert!(w.iter().all(|&x| x > 0), "a zero weight survived");
                    }
                }
            }
        }
    }

    #[test_case]
    fn repeated_resizes_stay_stable() {
        // A drag fires on every pointer report, so `resize_tracks` runs dozens of
        // times per gesture on its own output. It must not drift the total or
        // ratchet a track to zero.
        let mut w = even_weights(4);
        let total = 1600u64;
        for k in 0..200u64 {
            let target = (k * 37) % total; // wander across the whole band
            resize_tracks(&mut w, k as usize % 3, total, 10, target);
            assert_eq!(w.iter().sum::<u64>(), WEIGHT_TOTAL, "drifted at step {}", k);
            let sizes = track_sizes(total, 10, &w);
            for (t, s) in sizes.iter().enumerate() {
                assert!(*s >= MIN_TRACK_PX, "track {} fell to {} at step {}", t, s, k);
            }
        }
    }

    #[test_case]
    fn resize_tracks_rejects_impossible_moves() {
        let mut w = even_weights(2);
        assert!(!resize_tracks(&mut w, 1, 1000, 10, 500), "no divider after the last track");
        assert!(!resize_tracks(&mut w, 9, 1000, 10, 500), "index past the end");
        // A band too small for two minimum panes cannot be split.
        let mut w = even_weights(2);
        assert!(!resize_tracks(&mut w, 0, MIN_TRACK_PX, 10, 10));
    }

    #[test_case]
    fn move_tab_between_panes() {
        let mut panes = alloc::vec![alloc::vec![1u32, 2], alloc::vec![3u32], alloc::vec![]];
        assert!(move_tab(&mut panes, 0, 0, 2, 0));
        assert_eq!(panes[0], alloc::vec![2]);
        assert_eq!(panes[2], alloc::vec![1]);
        assert!(move_tab(&mut panes, 2, 0, 1, 1));
        assert_eq!(panes[1], alloc::vec![3, 1]);
        assert!(panes[2].is_empty());
    }

    #[test_case]
    fn move_tab_reorder_same_pane() {
        let mut panes = alloc::vec![alloc::vec![10u32, 20, 30]];
        assert!(move_tab(&mut panes, 0, 0, 0, 3)); // 10 → end
        assert_eq!(panes[0], alloc::vec![20, 30, 10]);
    }

    #[test_case]
    fn move_tab_rejects_bad_indices() {
        let mut panes = alloc::vec![alloc::vec![1u32], alloc::vec![]];
        assert!(!move_tab(&mut panes, 0, 5, 1, 0), "from_idx past end");
        assert!(!move_tab(&mut panes, 9, 0, 1, 0), "from_pane past end");
        assert!(!move_tab(&mut panes, 0, 0, 9, 0), "to_pane past end (shell/outside)");
        assert!(!move_tab(&mut panes, 1, 0, 0, 0), "empty source pane");
        assert_eq!(panes[0], alloc::vec![1], "no move mutated nothing");
    }

    #[test_case]
    fn insert_index_same_pane_shifts_after_removal() {
        // Cross-pane: no shift, the source removal doesn't touch the dest list.
        assert_eq!(insert_index(0, 0, 1, 2, 2), 2);
        assert_eq!(insert_index(0, 3, 1, 2, 0), 0);
        // Same pane, drop after the source slot: decremented.
        assert_eq!(insert_index(0, 0, 0, 2, 2), 1);
        // Same pane, drop before the source slot: unchanged.
        assert_eq!(insert_index(0, 2, 0, 2, 0), 0);
        // Clamped to the destination length.
        assert_eq!(insert_index(0, 0, 1, 1, 99), 1);
    }

    #[test_case]
    fn move_tab_reorder_covers_every_slot() {
        // Dragging each tab of a 3-tab bar to each drop slot must land exactly
        // where the cursor was — the off-by-one that clamping-before-shifting
        // introduced only showed up on the last slot.
        let cases: [(usize, usize, [u32; 3]); 6] = [
            (0, 3, [20, 30, 10]), // first → end
            (0, 2, [20, 10, 30]), // first → middle
            (2, 0, [30, 10, 20]), // last → front
            (2, 1, [10, 30, 20]), // last → middle
            (1, 0, [20, 10, 30]), // middle → front
            (1, 3, [10, 30, 20]), // middle → end
        ];
        for (from, to, want) in cases {
            let mut panes = alloc::vec![alloc::vec![10u32, 20, 30]];
            assert!(move_tab(&mut panes, 0, from, 0, to));
            assert_eq!(panes[0], want.to_vec(), "from={} to={}", from, to);
        }
    }

    #[test_case]
    fn move_tab_onto_itself_is_identity() {
        for i in 0..3usize {
            let mut panes = alloc::vec![alloc::vec![10u32, 20, 30]];
            assert!(move_tab(&mut panes, 0, i, 0, i));
            assert_eq!(panes[0], alloc::vec![10, 20, 30], "slot {}", i);
        }
    }

    #[test_case]
    fn max_panes_from_cfg_prefers_new_key() {
        assert_eq!(max_panes_from_cfg(Some(4), Some(1)), Some(4));
        assert_eq!(max_panes_from_cfg(Some(99), None), Some(9));
        assert_eq!(max_panes_from_cfg(Some(0), None), Some(2));
        assert_eq!(max_panes_from_cfg(Some(-3), None), Some(2));
        // Legacy key counted ACTION columns, so 1 → today's 2-pane layout.
        assert_eq!(max_panes_from_cfg(None, Some(1)), Some(2));
        assert_eq!(max_panes_from_cfg(None, Some(3)), Some(4));
        assert_eq!(max_panes_from_cfg(None, Some(50)), Some(9));
        // Neither key: leave the live layout alone.
        assert_eq!(max_panes_from_cfg(None, None), None);
    }

    #[test_case]
    fn status_split_carves_the_named_edge_and_leaves_the_rest() {
        // Bottom is the historical layout, so it must be exactly what it was:
        // content starts at the origin and simply stops short of the bar.
        let (bar, content) = status_split(1920, 1080, StatusPos::Bottom, 24);
        assert_eq!(bar, (0, 1056, 1920, 24));
        assert_eq!(content, (0, 0, 1920, 1056));
        // Top: same sizes, but the content is pushed down by the bar.
        let (bar, content) = status_split(1920, 1080, StatusPos::Top, 24);
        assert_eq!(bar, (0, 0, 1920, 24));
        assert_eq!(content, (0, 24, 1920, 1056));
        // Left/right take width instead of height, full height either way.
        let (bar, content) = status_split(1920, 1080, StatusPos::Left, 200);
        assert_eq!(bar, (0, 0, 200, 1080));
        assert_eq!(content, (200, 0, 1720, 1080));
        let (bar, content) = status_split(1920, 1080, StatusPos::Right, 200);
        assert_eq!(bar, (1720, 0, 200, 1080));
        assert_eq!(content, (0, 0, 1720, 1080));
    }

    #[test_case]
    fn status_split_is_exact_and_never_starves_the_content() {
        // Bar + content must tile the desktop with no overlap and no gap, on every
        // edge and at any thickness — including ones past the clamp.
        for &pos in &[StatusPos::Top, StatusPos::Bottom, StatusPos::Left, StatusPos::Right] {
            for &t in &[0u64, 1, 24, 200, 5000] {
                let ((bx, by, bw, bh), (cx, cy, cw, ch)) = status_split(800, 600, pos, t);
                if pos.vertical() {
                    assert_eq!(bw + cw, 800, "{pos:?} t={t} widths must tile");
                    assert_eq!((bh, ch), (600, 600), "{pos:?} full height");
                    assert!(bw <= 400, "{pos:?} t={t}: bar took more than half");
                    assert_eq!(by, 0);
                    assert_eq!(cy, 0);
                    // Adjacent, in the order the edge implies.
                    if pos == StatusPos::Left {
                        assert_eq!((bx, cx), (0, bw));
                    } else {
                        assert_eq!((cx, bx), (0, cw));
                    }
                } else {
                    assert_eq!(bh + ch, 600, "{pos:?} t={t} heights must tile");
                    assert_eq!((bw, cw), (800, 800), "{pos:?} full width");
                    assert!(bh <= 300, "{pos:?} t={t}: bar took more than half");
                    assert_eq!(bx, 0);
                    assert_eq!(cx, 0);
                    if pos == StatusPos::Top {
                        assert_eq!((by, cy), (0, bh));
                    } else {
                        assert_eq!((cy, by), (0, ch));
                    }
                }
            }
        }
    }

    #[test_case]
    fn status_pos_round_trips_and_rejects_typos() {
        for &pos in &[StatusPos::Top, StatusPos::Bottom, StatusPos::Left, StatusPos::Right] {
            assert_eq!(StatusPos::parse(pos.as_str()), Some(pos));
        }
        assert_eq!(StatusPos::parse(" Bottom "), Some(StatusPos::Bottom));
        assert_eq!(StatusPos::parse("LEFT"), Some(StatusPos::Left));
        // A typo must not move the bar — the caller keeps the current position.
        assert_eq!(StatusPos::parse("botom"), None);
        assert_eq!(StatusPos::parse("centre"), None);
        assert_eq!(StatusPos::parse(""), None);
        // Top is the default (chrome above content).
        assert_eq!(StatusPos::default(), StatusPos::Top);
        assert!(StatusPos::Left.vertical() && StatusPos::Right.vertical());
        assert!(!StatusPos::Top.vertical() && !StatusPos::Bottom.vertical());
    }

    #[test_case]
    fn status_lines_vertical_is_one_field_per_row_top_to_bottom() {
        // Font Awesome PUA icons (same shape the live status bar emits).
        let kbd = crate::icons::fa::KEYBOARD;
        let mouse = crate::icons::fa::MOUSE;
        let sample = alloc::format!("{kbd} {mouse}  net  mem 88M  cpu 3%");
        let lines = status_lines_vertical(&sample, 18);
        assert!(lines.len() >= 5, "each token becomes its own row, got {lines:?}");
        assert_eq!(lines[0].chars().next(), Some(kbd));
        // Long token wraps rather than vanishing.
        let clock = status_lines_vertical("Mon 2026-07-27 05:59:12 UTC", 10);
        assert!(clock.len() >= 2);
        assert!(clock.iter().any(|l| l.contains("05:59") || l.contains("UTC") || l.contains("2026")));
    }

    #[test_case]
    fn status_segments_splits_on_group_gaps_not_field_gaps() {
        // The real default template: single spaces bind a group, doubles separate.
        let t = "usb usb  10.0.2.15  12%/8G  3%  8 cores  84%=  00:12:34 UTC";
        assert_eq!(
            status_segments(t),
            alloc::vec!["usb usb", "10.0.2.15", "12%/8G", "3%", "8 cores", "84%=", "00:12:34 UTC"]
        );
        // Three-plus spaces are one separator, not an empty segment.
        assert_eq!(status_segments("a    b"), alloc::vec!["a", "b"]);
        // No double space at all is one segment, never zero.
        assert_eq!(status_segments("ChittiOS v0.1"), alloc::vec!["ChittiOS v0.1"]);
        // Nothing to show yields nothing to draw.
        assert!(status_segments("").is_empty());
        assert!(status_segments("   ").is_empty());
    }

    #[test_case]
    fn wrap_segment_keeps_the_clock_whole() {
        // The case this exists for: at 16 columns, ellipsizing loses the time.
        assert_eq!(
            wrap_segment("Mon 2026-07-27 05:59:12 UTC", 16),
            alloc::vec!["Mon 2026-07-27", "05:59:12 UTC"]
        );
        // Something that already fits stays on one row.
        assert_eq!(wrap_segment("mem 88M/6.0G", 16), alloc::vec!["mem 88M/6.0G"]);
        assert_eq!(wrap_segment("kbd", 16), alloc::vec!["kbd"]);
        // Exactly-at-the-limit fits; one over wraps.
        assert_eq!(wrap_segment("abcd efgh", 9), alloc::vec!["abcd efgh"]);
        assert_eq!(wrap_segment("abcd efgh", 8), alloc::vec!["abcd", "efgh"]);
        // A word longer than the row is handed over whole rather than split
        // mid-token — the painter ellipsizes it.
        assert_eq!(wrap_segment("supercalifragilistic", 8), alloc::vec!["supercalifragilistic"]);
        assert_eq!(wrap_segment("ab supercalifragilistic", 8), alloc::vec!["ab", "supercalifragilistic"]);
        // Degenerate inputs draw nothing rather than panicking.
        assert!(wrap_segment("", 16).is_empty());
        assert!(wrap_segment("   ", 16).is_empty());
        assert!(wrap_segment("anything", 0).is_empty());
        // Multi-byte content must not slice a char boundary (byte ranges, char counts).
        assert_eq!(wrap_segment("84%≡ 12°C", 5), alloc::vec!["84%≡", "12°C"]);
    }

    #[test_case]
    fn cycle_focus_includes_shell_and_wraps() {
        // One action pane: chat ↔ action0.
        let vis = [0usize];
        assert_eq!(cycle_focus_target(&vis, false, 0, true), (true, 0));
        assert_eq!(cycle_focus_target(&vis, true, 0, true), (false, 0));
        assert_eq!(cycle_focus_target(&vis, true, 0, false), (false, 0));
        assert_eq!(cycle_focus_target(&vis, false, 0, false), (true, 0));
    }

    #[test_case]
    fn cycle_focus_grid_walks_all_panes_then_shell() {
        // Grid of 3 visible action columns.
        let vis = [0usize, 1, 2];
        // chat → 0 → 1 → 2 → chat
        assert_eq!(cycle_focus_target(&vis, false, 0, true), (true, 0));
        assert_eq!(cycle_focus_target(&vis, true, 0, true), (true, 1));
        assert_eq!(cycle_focus_target(&vis, true, 1, true), (true, 2));
        assert_eq!(cycle_focus_target(&vis, true, 2, true), (false, 2));
        // reverse: chat → 2 → 1 → 0 → chat
        assert_eq!(cycle_focus_target(&vis, false, 0, false), (true, 2));
        assert_eq!(cycle_focus_target(&vis, true, 2, false), (true, 1));
        assert_eq!(cycle_focus_target(&vis, true, 1, false), (true, 0));
        assert_eq!(cycle_focus_target(&vis, true, 0, false), (false, 0));
    }

    #[test_case]
    fn cycle_focus_empty_band_stays_on_shell() {
        let vis: [usize; 0] = [];
        assert_eq!(cycle_focus_target(&vis, false, 0, true), (false, 0));
        assert_eq!(cycle_focus_target(&vis, true, 3, true), (false, 3));
    }

    #[test_case]
    fn wrap_segment_never_loses_or_reorders_words() {
        // Whatever the width, the words come back in order and all of them come back.
        let s = "Mon 2026-07-27 05:59:12 UTC and some more fields here";
        for cols in 1..40usize {
            let rows = wrap_segment(s, cols);
            let flat: alloc::vec::Vec<&str> = rows.iter().flat_map(|r| r.split_whitespace()).collect();
            let want: alloc::vec::Vec<&str> = s.split_whitespace().collect();
            assert_eq!(flat, want, "cols={cols} dropped or reordered words");
            // No row is blank, and no row overflows unless it is a single long word.
            for r in &rows {
                assert!(!r.is_empty(), "cols={cols} produced an empty row");
                assert!(
                    r.chars().count() <= cols || !r.contains(' '),
                    "cols={cols} row {r:?} overflows and could have been broken"
                );
            }
        }
    }

    /// The falloff must actually *fall off*: full at the shadow's own edge,
    /// nothing past the blur, and strictly decreasing in between. A shadow that
    /// steps between two constants — which is what this replaced — reads as a
    /// black border, not as elevation.
    #[test_case]
    fn shadow_falloff_is_monotone_and_bounded() {
        for blur in [1u64, 6, 12, 24] {
            assert_eq!(shadow_falloff(0, blur), 255, "blur={blur} edge is full");
            assert_eq!(shadow_falloff(blur, blur), 0, "blur={blur} ends at the blur");
            assert_eq!(shadow_falloff(blur + 99, blur), 0);
            let mut prev = 256u32;
            for d in 0..blur {
                let a = shadow_falloff(d, blur);
                assert!(a < prev, "blur={blur} d={d} did not decrease ({a} >= {prev})");
                prev = a;
            }
        }
        // A degenerate geometry casts no shadow rather than dividing by zero.
        assert_eq!(shadow_falloff(0, 0), 0);
    }

    /// The corner is the product of both axes, so it is lighter than either
    /// straight edge — that is what rounds it. And nothing anywhere exceeds the
    /// peak, or the shadow would be darker than its own darkest point.
    #[test_case]
    fn shadow_alpha_rounds_the_corner_and_never_exceeds_peak() {
        let g = shadow_geom(2);
        let (blur, peak) = (g.blur, g.peak);
        let edge = shadow_alpha(1, 0, blur, peak); // straight side
        let corner = shadow_alpha(1, 1, blur, peak); // diagonally out
        assert!(corner < edge, "corner {corner} should be lighter than edge {edge}");
        assert_eq!(shadow_alpha(0, 0, blur, peak), peak);
        for dx in 0..blur + 4 {
            for dy in 0..blur + 4 {
                assert!(shadow_alpha(dx, dy, blur, peak) <= peak);
            }
        }
        // Outside the blur on either axis there is no shadow at all.
        assert_eq!(shadow_alpha(blur, 0, blur, peak), 0);
        assert_eq!(shadow_alpha(0, blur, blur, peak), 0);
    }

    /// One axis of the point-to-rectangle distance. `hi` is exclusive, so the
    /// last pixel *inside* is at distance 0 and the first one outside is at 1 —
    /// an off-by-one here puts the whole shadow one pixel under the box, where
    /// the box paints over it and the shadow looks a pixel thin on two sides.
    #[test_case]
    fn span_dist_is_zero_inside_and_one_at_the_first_pixel_out() {
        assert_eq!(span_dist(10, 10, 20), 0);
        assert_eq!(span_dist(19, 10, 20), 0);
        assert_eq!(span_dist(20, 10, 20), 1);
        assert_eq!(span_dist(9, 10, 20), 1);
        assert_eq!(span_dist(25, 10, 20), 6);
        assert_eq!(span_dist(0, 10, 20), 10);
    }

    /// Shadow weight is a ratio of the font scale, so the same box looks the
    /// same on a 768p panel and a 1440p one.
    #[test_case]
    fn shadow_geom_scales_with_the_font() {
        let (a, b) = (shadow_geom(1), shadow_geom(2));
        assert_eq!(b.blur, 2 * a.blur);
        assert_eq!(b.offset, 2 * a.offset);
        assert_eq!(a.peak, b.peak, "opacity is not a function of pixel density");
        assert!(a.offset < a.blur, "the offset must stay inside the blur");
        // A scale of 0 would be a shadowless (and division-prone) geometry.
        assert_eq!(shadow_geom(0).blur, shadow_geom(1).blur);
    }
}

/// [`present_fit`], with the integer-upscale rule made optional.
///
/// Integer upscaling exists to keep **package-UI text** crisp: a canvas carries
/// deferred labels re-rasterized at presentation scale, and a fractional factor
/// makes those shimmer. A surface that was filled by `ui_present` has no labels
/// at all — presenting a frame clears them — so the rule is protecting nothing
/// there and only costs screen area. At 320x200 in a ~1080x1000 pane the integer
/// factor is 2, using 640x400 of it and leaving most of the pane empty; the free
/// fit is 3.4x.
///
/// The trade is honest: a fractional nearest-neighbour upscale gives pixel
/// columns of uneven width. For a photo or a video frame that is invisible, and
/// for a game it is the same thing every source port does in a non-integer
/// window. Filling the pane is worth more than uniform pixels here.
pub fn present_fit_mode(sw: u64, sh: u64, pw: u64, ph: u64, integer: bool) -> (u64, u64) {
    if sw == 0 || sh == 0 || pw == 0 || ph == 0 {
        return (0, 0);
    }
    // Free aspect-fit ("contain").
    let fit_w = pw;
    let fit_h = sh.saturating_mul(pw).saturating_div(sw).max(1);
    let (free_w, free_h) = if fit_h <= ph {
        (fit_w, fit_h)
    } else {
        let fit_h = ph;
        let fit_w = sw.saturating_mul(ph).saturating_div(sh).max(1);
        (fit_w.min(pw), fit_h)
    };
    // Integer upscale when the free fit would grow the image.
    if integer && free_w >= sw && free_h >= sh {
        let s = (pw / sw).min(ph / sh).max(1);
        let iw = sw.saturating_mul(s);
        let ih = sh.saturating_mul(s);
        if iw <= pw && ih <= ph && s >= 1 {
            return (iw, ih);
        }
    }
    (free_w, free_h)
}
