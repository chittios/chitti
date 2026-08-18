//! **Flexbox + CSS Grid** geometry (pure).
//!
//! Reference: CSS Flexible Box / Grid Layout; Ladybird
//! `FlexFormattingContext` / `GridFormattingContext`.
//!
//! Supports: main-axis placement, cross-axis align, **flex-wrap** multi-line,
//! flex-grow distribution, **dense grid** packing, line fragmentation (height
//! budget clip — last line dropped when over max height).

use super::css::{AlignItems, FlexDirection, GridTrack, Justify};
use alloc::vec::Vec;

/// Resolve `grid-template-columns` tracks to pixel widths within `container`.
/// Fixed `px` tracks take their length; the remaining free space is shared
/// among `fr` tracks by weight, with `auto` tracks counted as `1fr`. Empty
/// `tracks` falls back to `cols` equal columns.
pub fn grid_track_widths(tracks: &[GridTrack], container: i32, gap: i32, cols: usize) -> Vec<i32> {
    if tracks.is_empty() {
        let w = grid_col_width(container, cols.max(1) as u8, gap);
        return alloc::vec![w; cols.max(1)];
    }
    let n = tracks.len();
    let total_gap = gap * (n as i32 - 1).max(0);
    let mut fixed = 0i32;
    let mut fr_total = 0.0f32;
    for t in tracks {
        match t {
            GridTrack::Px(p) => fixed += (*p).max(0),
            GridTrack::Fr(f) => fr_total += f.max(0.0),
            GridTrack::Auto => fr_total += 1.0,
        }
    }
    let free = (container - total_gap - fixed).max(0) as f32;
    tracks
        .iter()
        .map(|t| {
            let w = match t {
                GridTrack::Px(p) => (*p).max(0) as f32,
                GridTrack::Fr(f) if fr_total > 0.0 => free * f.max(0.0) / fr_total,
                GridTrack::Auto if fr_total > 0.0 => free / fr_total,
                _ => 0.0,
            };
            (w as i32).max(1)
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FlexWrap {
    #[default]
    NoWrap,
    Wrap,
    WrapReverse,
}

/// One placed item after flex/grid algorithm.
#[derive(Clone, Copy, Debug)]
pub struct Placed {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub index: usize,
}

/// Place flex items along the main axis. Returns x (or y if column) offsets.
pub fn flex_main_offsets(
    item_sizes: &[i32],
    container: i32,
    gap: i32,
    justify: Justify,
    direction: FlexDirection,
) -> Vec<i32> {
    let _ = direction;
    let n = item_sizes.len();
    if n == 0 {
        return Vec::new();
    }
    let total: i32 = item_sizes.iter().sum::<i32>() + gap * (n as i32 - 1).max(0);
    let free = (container - total).max(0);
    let mut offsets = Vec::with_capacity(n);
    let mut cursor = match justify {
        Justify::Start => 0,
        Justify::Center => free / 2,
        Justify::End => free,
        Justify::SpaceBetween => 0,
    };
    let between = if matches!(justify, Justify::SpaceBetween) && n > 1 {
        free / (n as i32 - 1)
    } else {
        0
    };
    for (i, &sz) in item_sizes.iter().enumerate() {
        offsets.push(cursor);
        cursor += sz + gap;
        if matches!(justify, Justify::SpaceBetween) && i + 1 < n {
            cursor += between;
        }
    }
    offsets
}

/// Distribute free space by flex-grow factors (integer, proportional).
pub fn apply_flex_grow(bases: &[i32], grows: &[u32], free: i32) -> Vec<i32> {
    let n = bases.len();
    let mut out = bases.to_vec();
    if free <= 0 || n == 0 {
        return out;
    }
    let total_grow: u32 = grows.iter().sum();
    if total_grow == 0 {
        return out;
    }
    let mut used = 0i32;
    for i in 0..n {
        let g = *grows.get(i).unwrap_or(&0);
        if g == 0 {
            continue;
        }
        let add = (free as i64 * g as i64 / total_grow as i64) as i32;
        out[i] = out[i].saturating_add(add);
        used += add;
    }
    // Remainder to last growing item
    if used < free {
        for i in (0..n).rev() {
            if *grows.get(i).unwrap_or(&0) > 0 {
                out[i] += free - used;
                break;
            }
        }
    }
    out
}

/// Flex wrap: pack items into lines of max `container` main size.
pub fn flex_wrap_lines(
    main_sizes: &[i32],
    cross_sizes: &[i32],
    container_main: i32,
    gap: i32,
    wrap: FlexWrap,
) -> Vec<Vec<usize>> {
    if matches!(wrap, FlexWrap::NoWrap) || main_sizes.is_empty() {
        return alloc::vec![(0..main_sizes.len()).collect()];
    }
    let mut lines: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut used = 0i32;
    for i in 0..main_sizes.len() {
        let need = main_sizes[i]
            + if cur.is_empty() {
                0
            } else {
                gap
            };
        if !cur.is_empty() && used + need > container_main && container_main > 0 {
            lines.push(core::mem::take(&mut cur));
            used = 0;
        }
        used += if cur.is_empty() {
            main_sizes[i]
        } else {
            main_sizes[i] + gap
        };
        cur.push(i);
        let _ = cross_sizes;
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    if matches!(wrap, FlexWrap::WrapReverse) {
        lines.reverse();
    }
    lines
}

/// Full flex layout → absolute positions (row or column).
pub fn flex_place(
    widths: &[i32],
    heights: &[i32],
    grows: &[u32],
    container_w: i32,
    container_h: i32,
    gap: i32,
    direction: FlexDirection,
    justify: Justify,
    align: AlignItems,
    wrap: FlexWrap,
    // If `Some`, drop lines that would exceed this content height (fragmentation).
    max_content_h: Option<i32>,
) -> Vec<Placed> {
    let n = widths.len();
    if n == 0 {
        return Vec::new();
    }
    let (main, cross) = match direction {
        FlexDirection::Row => (widths, heights),
        FlexDirection::Column => (heights, widths),
    };
    let container_main = match direction {
        FlexDirection::Row => container_w,
        FlexDirection::Column => container_h.max(
            main.iter().sum::<i32>() + gap * (n as i32 - 1).max(0),
        ),
    };
    let lines = flex_wrap_lines(main, cross, container_main, gap, wrap);
    let mut placed = Vec::new();
    let mut cross_cursor = 0i32;
    for line in &lines {
        let line_main: Vec<i32> = line.iter().map(|&i| main[i]).collect();
        let line_grow: Vec<u32> = line.iter().map(|&i| *grows.get(i).unwrap_or(&0)).collect();
        let free = (container_main
            - line_main.iter().sum::<i32>()
            - gap * (line.len() as i32 - 1).max(0))
        .max(0);
        let grown = apply_flex_grow(&line_main, &line_grow, free);
        let offs = flex_main_offsets(&grown, container_main, gap, justify, direction);
        let line_cross = line
            .iter()
            .map(|&i| cross[i])
            .max()
            .unwrap_or(0)
            .max(1);
        // A single-line flex container with a definite cross size (`h-9` +
        // `items-center`) sizes the line to that inner height so items
        // center in the painted box, not in their own ink. `container_h == 0`
        // (row, height:auto) keeps the item-sized line.
        let line_cross = if lines.len() == 1 {
            let definite = match direction {
                FlexDirection::Row => container_h,
                FlexDirection::Column => container_w,
            };
            if definite > 0 {
                line_cross.max(definite)
            } else {
                line_cross
            }
        } else {
            line_cross
        };
        if let Some(max_h) = max_content_h {
            if cross_cursor + line_cross > max_h && !placed.is_empty() {
                break; // fragmentation: stop adding lines
            }
        }
        for (j, &idx) in line.iter().enumerate() {
            let main_off = offs.get(j).copied().unwrap_or(0);
            let csz = cross[idx];
            let cross_off = flex_cross_offset(csz, line_cross, align);
            let (x, y, w, h) = match direction {
                FlexDirection::Row => (
                    main_off,
                    cross_cursor + cross_off,
                    grown.get(j).copied().unwrap_or(widths[idx]),
                    heights[idx],
                ),
                FlexDirection::Column => (
                    cross_off,
                    main_off + cross_cursor,
                    widths[idx],
                    grown.get(j).copied().unwrap_or(heights[idx]),
                ),
            };
            placed.push(Placed {
                x,
                y,
                w,
                h,
                index: idx,
            });
        }
        cross_cursor += line_cross + gap;
    }
    placed
}

pub fn flex_cross_offset(item: i32, line: i32, align: AlignItems) -> i32 {
    match align {
        AlignItems::Start | AlignItems::Stretch => 0,
        AlignItems::Center => ((line - item) / 2).max(0),
        AlignItems::End => (line - item).max(0),
    }
}

pub fn grid_cell(i: usize, cols: u8, cell_w: i32, cell_h: i32, gap: i32) -> (i32, i32) {
    let cols = cols.max(1) as usize;
    let col = i % cols;
    let row = i / cols;
    let x = col as i32 * (cell_w + gap);
    let y = row as i32 * (cell_h + gap);
    (x, y)
}

pub fn grid_col_width(container: i32, cols: u8, gap: i32) -> i32 {
    let cols = cols.max(1) as i32;
    let gaps = gap * (cols - 1).max(0);
    ((container - gaps) / cols).max(1)
}

/// Dense grid packing: place each item in the first free cell that fits
/// (row-major scan). `span_cols`/`span_rows` default 1.
pub fn grid_dense_place(
    item_cols: &[u8],
    item_rows: &[u8],
    cols: u8,
    cell_w: i32,
    cell_h: i32,
    gap: i32,
) -> Vec<Placed> {
    let cols = cols.max(1) as usize;
    let n = item_cols.len();
    // Occupancy: rows grow dynamically
    let mut occ: Vec<Vec<bool>> = Vec::new();
    let mut placed = Vec::new();
    for i in 0..n {
        let sc = (*item_cols.get(i).unwrap_or(&1)).max(1) as usize;
        let sr = (*item_rows.get(i).unwrap_or(&1)).max(1) as usize;
        let mut found = None;
        let mut r = 0usize;
        'search: loop {
            while occ.len() < r + sr {
                occ.push(alloc::vec![false; cols]);
            }
            for c in 0..=cols.saturating_sub(sc) {
                let mut free = true;
                for rr in r..r + sr {
                    for cc in c..c + sc {
                        if occ[rr][cc] {
                            free = false;
                            break;
                        }
                    }
                    if !free {
                        break;
                    }
                }
                if free {
                    for rr in r..r + sr {
                        for cc in c..c + sc {
                            occ[rr][cc] = true;
                        }
                    }
                    found = Some((c, r));
                    break 'search;
                }
            }
            r += 1;
            if r > n + cols {
                found = Some((0, r));
                break;
            }
        }
        let (c, r) = found.unwrap_or((0, 0));
        let x = c as i32 * (cell_w + gap);
        let y = r as i32 * (cell_h + gap);
        placed.push(Placed {
            x,
            y,
            w: sc as i32 * cell_w + gap * (sc as i32 - 1).max(0),
            h: sr as i32 * cell_h + gap * (sr as i32 - 1).max(0),
            index: i,
        });
    }
    placed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn flex_space_between_and_center() {
        let sizes = [10i32, 10, 10];
        let o = flex_main_offsets(&sizes, 100, 0, Justify::SpaceBetween, FlexDirection::Row);
        assert_eq!(o[0], 0);
        assert_eq!(o[1], 45);
        assert_eq!(o[2], 90);
        let c = flex_main_offsets(&sizes, 100, 0, Justify::Center, FlexDirection::Row);
        assert_eq!(c[0], 35);
    }

    #[test_case]
    fn grid_cells() {
        assert_eq!(grid_cell(0, 2, 50, 40, 10), (0, 0));
        assert_eq!(grid_cell(1, 2, 50, 40, 10), (60, 0));
        assert_eq!(grid_cell(2, 2, 50, 40, 10), (0, 50));
        assert_eq!(grid_col_width(100, 2, 10), 45);
    }

    #[test_case]
    fn flex_wrap_two_lines() {
        // 60+60 fits in 130; third wraps.
        let mains = [60i32, 60, 60];
        let cross = [10i32, 10, 10];
        let lines = flex_wrap_lines(&mains, &cross, 130, 0, FlexWrap::Wrap);
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert_eq!(lines[0], alloc::vec![0, 1]);
        assert_eq!(lines[1], alloc::vec![2]);
    }

    #[test_case]
    fn flex_grow_distributes() {
        let g = apply_flex_grow(&[10, 10], &[1, 3], 40);
        assert_eq!(g[0], 20);
        assert_eq!(g[1], 40);
    }

    #[test_case]
    fn dense_grid_places() {
        let cols = [1u8, 2, 1];
        let rows = [1u8, 1, 1];
        let p = grid_dense_place(&cols, &rows, 2, 50, 40, 0);
        assert_eq!(p.len(), 3);
        assert_eq!(p[0].x, 0);
        assert_eq!(p[1].w, 100); // span 2
    }

    #[test_case]
    fn flex_place_wrap() {
        let w = [40i32, 40, 40];
        let h = [20i32, 20, 20];
        let g = [0u32, 0, 0];
        let p = flex_place(
            &w,
            &h,
            &g,
            70,
            200,
            0,
            FlexDirection::Row,
            Justify::Start,
            AlignItems::Start,
            FlexWrap::Wrap,
            None,
        );
        assert!(p.iter().any(|x| x.y > 0), "second line y>0: {p:?}");
    }

    #[test_case]
    fn flex_row_centers_in_the_container_cross_size() {
        // `h-9` + `items-center`: a 16px label sits in the middle of 36px,
        // not at y=0 of its own ink.
        let w = [40i32];
        let h = [16i32];
        let g = [0u32];
        let p = flex_place(
            &w,
            &h,
            &g,
            80,
            36,
            0,
            FlexDirection::Row,
            Justify::Start,
            AlignItems::Center,
            FlexWrap::NoWrap,
            None,
        );
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].y, 10, "((36-16)/2) = 10, got {}", p[0].y);
        // Auto height (0) must not invent a cross size.
        let auto = flex_place(
            &w,
            &h,
            &g,
            80,
            0,
            0,
            FlexDirection::Row,
            Justify::Start,
            AlignItems::Center,
            FlexWrap::NoWrap,
            None,
        );
        assert_eq!(auto[0].y, 0, "auto-height line is the item itself");
    }
}
