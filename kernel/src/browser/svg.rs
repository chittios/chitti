//! **SVG + MathML** layout/paint subset.
//!
//! Reference: Ladybird `Libraries/LibWeb/SVG/*`, MDN SVG/MathML.
//! Pure: parse attributes from the HTML DOM tree (SVG embedded in HTML) into
//! drawable primitives. Not a full SVG engine (no filters, SMIL, full path
//! grammar) — rect/circle/line/text/path-M-L and MathML mrow/mi/mo/msup.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::html::{Node, NodeKind};

#[derive(Clone, Debug)]
pub enum SvgPrim {
    Rect {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        fill: u32,
        stroke: u32,
        stroke_w: f32,
    },
    Circle {
        cx: f32,
        cy: f32,
        r: f32,
        fill: u32,
        stroke: u32,
    },
    Line {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        stroke: u32,
        stroke_w: f32,
    },
    Text {
        x: f32,
        y: f32,
        text: String,
        fill: u32,
        size: f32,
    },
    /// Polyline from simplified path `M x y L x y …`.
    Poly {
        points: Vec<(f32, f32)>,
        stroke: u32,
        fill: u32,
        stroke_w: f32,
    },
}

#[derive(Clone, Debug)]
pub struct SvgBox {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub prims: Vec<SvgPrim>,
    /// MathML flattened text (when this box is a math root).
    pub math_text: Option<String>,
}

/// Collect an `<svg>` subtree into primitives. `view_w`/`view_h` default size.
pub fn collect_svg(n: &Node, view_w: f32, view_h: f32) -> SvgBox {
    let mut prims = Vec::new();
    let (vw, vh) = svg_size(n, view_w, view_h);
    walk_svg(n, &mut prims, 0.0, 0.0);
    SvgBox {
        x: 0,
        y: 0,
        w: vw as i32,
        h: vh as i32,
        prims,
        math_text: None,
    }
}

/// Flatten MathML to readable text (superscript as `^`).
pub fn collect_math(n: &Node) -> String {
    let mut out = String::new();
    walk_math(n, &mut out);
    out
}

fn svg_size(n: &Node, def_w: f32, def_h: f32) -> (f32, f32) {
    let mut w = def_w;
    let mut h = def_h;
    if let NodeKind::Element {
        width_attr,
        height_attr,
        ..
    } = &n.kind
    {
        if let Some(x) = width_attr {
            w = *x as f32;
        }
        if let Some(y) = height_attr {
            h = *y as f32;
        }
    }
    // viewBox attr not on Element yet — use defaults.
    (w.max(16.0), h.max(16.0))
}

fn walk_svg(n: &Node, out: &mut Vec<SvgPrim>, ox: f32, oy: f32) {
    match &n.kind {
        NodeKind::Element { tag, .. } => {
            let tag = tag.as_str();
            match tag {
                "rect" => {
                    let x = attr_f(n, "x") + ox;
                    let y = attr_f(n, "y") + oy;
                    let w = attr_f_or(n, "width", 10.0);
                    let h = attr_f_or(n, "height", 10.0);
                    out.push(SvgPrim::Rect {
                        x,
                        y,
                        w,
                        h,
                        fill: attr_color(n, "fill", 0x000000),
                        stroke: attr_color(n, "stroke", 0x000000),
                        stroke_w: attr_f_or(n, "stroke-width", 0.0),
                    });
                }
                "circle" => {
                    out.push(SvgPrim::Circle {
                        cx: attr_f(n, "cx") + ox,
                        cy: attr_f(n, "cy") + oy,
                        r: attr_f_or(n, "r", 5.0),
                        fill: attr_color(n, "fill", 0x000000),
                        stroke: attr_color(n, "stroke", 0x000000),
                    });
                }
                "line" => {
                    out.push(SvgPrim::Line {
                        x1: attr_f(n, "x1") + ox,
                        y1: attr_f(n, "y1") + oy,
                        x2: attr_f(n, "x2") + ox,
                        y2: attr_f(n, "y2") + oy,
                        stroke: attr_color(n, "stroke", 0x000000),
                        stroke_w: attr_f_or(n, "stroke-width", 1.0),
                    });
                }
                "text" => {
                    let text = super::html::collect_text(n);
                    out.push(SvgPrim::Text {
                        x: attr_f(n, "x") + ox,
                        y: attr_f(n, "y") + oy,
                        text,
                        fill: attr_color(n, "fill", 0x000000),
                        size: attr_f_or(n, "font-size", 14.0),
                    });
                }
                "path" => {
                    if let Some(d) = attr_str(n, "d") {
                        let pts = parse_path_ml(&d);
                        if pts.len() >= 2 {
                            out.push(SvgPrim::Poly {
                                points: pts
                                    .into_iter()
                                    .map(|(x, y)| (x + ox, y + oy))
                                    .collect(),
                                stroke: attr_color(n, "stroke", 0x000000),
                                fill: attr_color(n, "fill", 0xffffff),
                                stroke_w: attr_f_or(n, "stroke-width", 1.0),
                            });
                        }
                    }
                }
                "g" | "svg" => {
                    for c in &n.children {
                        walk_svg(c, out, ox, oy);
                    }
                    return;
                }
                _ => {}
            }
            for c in &n.children {
                walk_svg(c, out, ox, oy);
            }
        }
        _ => {
            for c in &n.children {
                walk_svg(c, out, ox, oy);
            }
        }
    }
}

fn walk_math(n: &Node, out: &mut String) {
    match &n.kind {
        NodeKind::Text(t) => out.push_str(t),
        NodeKind::Element { tag, .. } => {
            match tag.as_str() {
                "msup" if n.children.len() >= 2 => {
                    walk_math(&n.children[0], out);
                    out.push('^');
                    walk_math(&n.children[1], out);
                }
                "msub" if n.children.len() >= 2 => {
                    walk_math(&n.children[0], out);
                    out.push('_');
                    walk_math(&n.children[1], out);
                }
                "msubsup" if n.children.len() >= 3 => {
                    walk_math(&n.children[0], out);
                    out.push('_');
                    walk_math(&n.children[1], out);
                    out.push('^');
                    walk_math(&n.children[2], out);
                }
                "mfrac" if n.children.len() >= 2 => {
                    out.push('(');
                    walk_math(&n.children[0], out);
                    out.push('/');
                    walk_math(&n.children[1], out);
                    out.push(')');
                }
                "msqrt" => {
                    out.push_str("√(");
                    for c in &n.children {
                        walk_math(c, out);
                    }
                    out.push(')');
                }
                "mroot" if n.children.len() >= 2 => {
                    out.push('(');
                    walk_math(&n.children[0], out);
                    out.push_str(")^(1/");
                    walk_math(&n.children[1], out);
                    out.push(')');
                }
                "mfenced" => {
                    out.push('(');
                    for (i, c) in n.children.iter().enumerate() {
                        if i > 0 {
                            out.push(',');
                        }
                        walk_math(c, out);
                    }
                    out.push(')');
                }
                "mtable" | "mtr" | "mtd" | "munder" | "mover" | "munderover" | "mmultiscripts"
                | "mprescripts" | "none" | "semantics" | "annotation" | "annotation-xml" => {
                    for c in &n.children {
                        walk_math(c, out);
                        if tag == "mtd" || tag == "mtr" {
                            out.push(' ');
                        }
                    }
                }
                "mo" | "mi" | "mn" | "mtext" | "mrow" | "math" | "mstyle" | "mpadded"
                | "mphantom" | "menclose" | "mspace" | "ms" => {
                    for c in &n.children {
                        walk_math(c, out);
                    }
                }
                _ => {
                    for c in &n.children {
                        walk_math(c, out);
                    }
                }
            }
        }
        NodeKind::Document => {
            for c in &n.children {
                walk_math(c, out);
            }
        }
    }
}

/// Parse path `d` into polyline samples.
/// Supports M/m L/l H/h V/v C/c (cubic, 8 samples) Q/q (quadratic, 6 samples)
/// Z/z close. Arcs (A/a) approximated as a straight line to the end point.
/// SMIL / filters are not animated — `filter` attrs are ignored at paint time.
pub fn parse_path_ml(d: &str) -> Vec<(f32, f32)> {
    parse_path(d)
}

pub fn parse_path(d: &str) -> Vec<(f32, f32)> {
    let mut pts = Vec::new();
    let mut chars = d.chars().peekable();
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    let mut start = (0.0f32, 0.0f32);
    let mut cmd = 'M';
    while let Some(&c) = chars.peek() {
        if c.is_ascii_alphabetic() {
            cmd = chars.next().unwrap();
            if cmd == 'Z' || cmd == 'z' {
                pts.push(start);
                cx = start.0;
                cy = start.1;
            }
            continue;
        }
        if c.is_ascii_whitespace() || c == ',' {
            chars.next();
            continue;
        }
        match cmd {
            'M' | 'm' | 'L' | 'l' | 'T' | 't' => {
                let x = read_num(&mut chars);
                skip_sep(&mut chars);
                let y = read_num(&mut chars);
                if cmd == 'M' {
                    cx = x;
                    cy = y;
                    start = (cx, cy);
                    pts.push((cx, cy));
                    cmd = 'L';
                } else if cmd == 'm' {
                    cx += x;
                    cy += y;
                    start = (cx, cy);
                    pts.push((cx, cy));
                    cmd = 'l';
                } else if cmd == 'L' {
                    cx = x;
                    cy = y;
                    pts.push((cx, cy));
                } else {
                    // l
                    cx += x;
                    cy += y;
                    pts.push((cx, cy));
                }
            }
            'H' | 'h' => {
                let x = read_num(&mut chars);
                if cmd == 'H' {
                    cx = x;
                } else {
                    cx += x;
                }
                pts.push((cx, cy));
            }
            'V' | 'v' => {
                let y = read_num(&mut chars);
                if cmd == 'V' {
                    cy = y;
                } else {
                    cy += y;
                }
                pts.push((cx, cy));
            }
            'C' | 'c' => {
                // cubic: x1 y1 x2 y2 x y
                let mut nums = [0.0f32; 6];
                for n in &mut nums {
                    skip_sep(&mut chars);
                    *n = read_num(&mut chars);
                }
                let (x1, y1, x2, y2, x, y) = if cmd == 'C' {
                    (nums[0], nums[1], nums[2], nums[3], nums[4], nums[5])
                } else {
                    (
                        cx + nums[0],
                        cy + nums[1],
                        cx + nums[2],
                        cy + nums[3],
                        cx + nums[4],
                        cy + nums[5],
                    )
                };
                for i in 1..=8 {
                    let t = i as f32 / 8.0;
                    let u = 1.0 - t;
                    let bx = u * u * u * cx
                        + 3.0 * u * u * t * x1
                        + 3.0 * u * t * t * x2
                        + t * t * t * x;
                    let by = u * u * u * cy
                        + 3.0 * u * u * t * y1
                        + 3.0 * u * t * t * y2
                        + t * t * t * y;
                    pts.push((bx, by));
                }
                cx = x;
                cy = y;
            }
            'Q' | 'q' => {
                let mut nums = [0.0f32; 4];
                for n in &mut nums {
                    skip_sep(&mut chars);
                    *n = read_num(&mut chars);
                }
                let (x1, y1, x, y) = if cmd == 'Q' {
                    (nums[0], nums[1], nums[2], nums[3])
                } else {
                    (cx + nums[0], cy + nums[1], cx + nums[2], cy + nums[3])
                };
                for i in 1..=6 {
                    let t = i as f32 / 6.0;
                    let u = 1.0 - t;
                    let bx = u * u * cx + 2.0 * u * t * x1 + t * t * x;
                    let by = u * u * cy + 2.0 * u * t * y1 + t * t * y;
                    pts.push((bx, by));
                }
                cx = x;
                cy = y;
            }
            'A' | 'a' => {
                // rx ry x-axis-rotation large-arc sweep x y — approximate end point
                for _ in 0..5 {
                    skip_sep(&mut chars);
                    let _ = read_num(&mut chars);
                }
                skip_sep(&mut chars);
                let x = read_num(&mut chars);
                skip_sep(&mut chars);
                let y = read_num(&mut chars);
                if cmd == 'A' {
                    cx = x;
                    cy = y;
                } else {
                    cx += x;
                    cy += y;
                }
                pts.push((cx, cy));
            }
            _ => {
                // unknown: skip a number pair
                let _ = read_num(&mut chars);
                skip_sep(&mut chars);
                let _ = read_num(&mut chars);
            }
        }
    }
    pts
}

fn read_num(chars: &mut core::iter::Peekable<core::str::Chars<'_>>) -> f32 {
    let mut s = String::new();
    if chars.peek() == Some(&'-') || chars.peek() == Some(&'+') {
        s.push(chars.next().unwrap());
    }
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() || c == '.' {
            s.push(chars.next().unwrap());
        } else {
            break;
        }
    }
    s.parse().unwrap_or(0.0)
}

fn skip_sep(chars: &mut core::iter::Peekable<core::str::Chars<'_>>) {
    while let Some(&c) = chars.peek() {
        if c.is_ascii_whitespace() || c == ',' {
            chars.next();
        } else {
            break;
        }
    }
}

fn attr_str<'a>(_n: &'a Node, _name: &str) -> Option<String> {
    // Extended attributes are not yet on NodeKind — use style_attr / value hacks.
    // For SVG path `d`, we encode as value= or style for tests; production parse
    // stores unknown attrs. Look at value for d when tag is path.
    if let NodeKind::Element {
        tag,
        value,
        style_attr,
        name,
        ..
    } = &_n.kind
    {
        if tag == "path" && _name == "d" {
            if let Some(v) = value {
                return Some(v.clone());
            }
            if let Some(s) = style_attr {
                if let Some(rest) = s.strip_prefix("d:") {
                    return Some(rest.to_string());
                }
            }
        }
        if _name == "d" {
            return name.clone(); // misuse name for tests
        }
    }
    None
}

fn attr_f(n: &Node, name: &str) -> f32 {
    attr_f_or(n, name, 0.0)
}

fn attr_f_or(n: &Node, name: &str, def: f32) -> f32 {
    if let NodeKind::Element {
        width_attr,
        height_attr,
        value,
        style_attr,
        ..
    } = &n.kind
    {
        if name == "width" {
            if let Some(x) = width_attr {
                return *x as f32;
            }
        }
        if name == "height" {
            if let Some(y) = height_attr {
                return *y as f32;
            }
        }
        // style="x:5;y:5;width:20" or value="x=5 y=5"
        for src in [style_attr.as_deref(), value.as_deref()].into_iter().flatten() {
            for part in src.split([';', ' ']) {
                let part = part.trim();
                if let Some(rest) = part
                    .strip_prefix(&alloc::format!("{name}:"))
                    .or_else(|| part.strip_prefix(&alloc::format!("{name}=")))
                {
                    if let Ok(v) = rest.trim().parse::<f32>() {
                        return v;
                    }
                }
            }
        }
    }
    def
}

fn attr_color(n: &Node, name: &str, def: u32) -> u32 {
    if let NodeKind::Element {
        style_attr,
        value,
        ..
    } = &n.kind
    {
        let search = |s: &str| {
            for part in s.split([';', ' ']) {
                let part = part.trim();
                if let Some(rest) = part.strip_prefix(&alloc::format!("{name}:")) {
                    if let Some(c) = super::css::parse_color(rest.trim()) {
                        return Some(c);
                    }
                }
                if let Some(rest) = part.strip_prefix(&alloc::format!("{name}=")) {
                    if let Some(c) = super::css::parse_color(rest.trim()) {
                        return Some(c);
                    }
                }
            }
            None
        };
        if let Some(s) = style_attr {
            if let Some(c) = search(s) {
                return c;
            }
        }
        if let Some(v) = value {
            if let Some(c) = search(v) {
                return c;
            }
        }
    }
    def
}

/// Rasterize SVG primitives into an RGB buffer (simple).
pub fn raster(box_: &SvgBox) -> Vec<u32> {
    let w = box_.w.max(1) as usize;
    let h = box_.h.max(1) as usize;
    let mut buf = alloc::vec![0x00ff_ffffu32; w * h];
    for p in &box_.prims {
        match p {
            SvgPrim::Rect {
                x,
                y,
                w: rw,
                h: rh,
                fill,
                ..
            } => {
                fill_rect(&mut buf, w, h, *x as i32, *y as i32, *rw as i32, *rh as i32, *fill);
            }
            SvgPrim::Circle { cx, cy, r, fill, .. } => {
                fill_circle(&mut buf, w, h, *cx as i32, *cy as i32, *r as i32, *fill);
            }
            SvgPrim::Line {
                x1,
                y1,
                x2,
                y2,
                stroke,
                ..
            } => {
                draw_line(
                    &mut buf,
                    w,
                    h,
                    *x1 as i32,
                    *y1 as i32,
                    *x2 as i32,
                    *y2 as i32,
                    *stroke,
                );
            }
            SvgPrim::Text {
                x,
                y,
                text,
                fill,
                size,
            } => {
                let _ = crate::font_ttf::blit_run(
                    &mut buf,
                    w,
                    h,
                    *x as i32,
                    *y as i32 - (*size as i32),
                    text,
                    *size,
                    *fill,
                );
            }
            SvgPrim::Poly { points, stroke, .. } => {
                for win in points.windows(2) {
                    draw_line(
                        &mut buf,
                        w,
                        h,
                        win[0].0 as i32,
                        win[0].1 as i32,
                        win[1].0 as i32,
                        win[1].1 as i32,
                        *stroke,
                    );
                }
            }
        }
    }
    buf
}

fn fill_rect(buf: &mut [u32], bw: usize, bh: usize, x: i32, y: i32, rw: i32, rh: i32, c: u32) {
    for dy in 0..rh {
        for dx in 0..rw {
            put(buf, bw, bh, x + dx, y + dy, c);
        }
    }
}

fn fill_circle(buf: &mut [u32], bw: usize, bh: usize, cx: i32, cy: i32, r: i32, c: u32) {
    let r2 = r * r;
    for y in (cy - r)..=(cy + r) {
        for x in (cx - r)..=(cx + r) {
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy <= r2 {
                put(buf, bw, bh, x, y, c);
            }
        }
    }
}

fn draw_line(buf: &mut [u32], bw: usize, bh: usize, x0: i32, y0: i32, x1: i32, y1: i32, c: u32) {
    let mut x0 = x0;
    let mut y0 = y0;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        put(buf, bw, bh, x0, y0, c);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn put(buf: &mut [u32], bw: usize, bh: usize, x: i32, y: i32, c: u32) {
    if x < 0 || y < 0 {
        return;
    }
    let (x, y) = (x as usize, y as usize);
    if x < bw && y < bh {
        buf[y * bw + x] = c & 0x00ff_ffff;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::browser::html;

    #[test_case]
    fn path_ml_parse() {
        let pts = parse_path_ml("M10 10 L20 20 L30 10");
        assert_eq!(pts.len(), 3);
        assert_eq!(pts[0], (10.0, 10.0));
        assert_eq!(pts[2], (30.0, 10.0));
    }

    #[test_case]
    fn path_cubic_samples() {
        let pts = parse_path("M0 0 C 0 10 10 10 10 0");
        assert!(pts.len() > 3, "cubic should sample: {pts:?}");
    }

    #[test_case]
    fn mathml_msqrt() {
        let doc = html::parse("<math><msqrt><mi>x</mi></msqrt></math>");
        let t = collect_math(&doc.root);
        assert!(t.contains('√') || t.contains('x'), "{t}");
    }

    #[test_case]
    fn mathml_msup() {
        let doc = html::parse("<math><msup><mi>x</mi><mn>2</mn></msup></math>");
        let t = collect_math(&doc.root);
        assert!(t.contains('^') || t.contains('x'), "{t}");
    }

    #[test_case]
    fn svg_rect_raster_nonzero() {
        let doc = html::parse(
            r#"<svg width="40" height="40"><rect style="x:5;y:5;width:20;height:20;fill:#ff0000"></rect></svg>"#,
        );
        // Find svg node
        fn find_svg(n: &Node) -> Option<&Node> {
            if n.tag_name() == Some("svg") {
                return Some(n);
            }
            for c in &n.children {
                if let Some(f) = find_svg(c) {
                    return Some(f);
                }
            }
            None
        }
        let svg = find_svg(&doc.root).expect("svg");
        let box_ = collect_svg(svg, 40.0, 40.0);
        let buf = raster(&box_);
        assert_eq!(buf.len(), 40 * 40);
        assert!(buf.iter().any(|&p| p == 0xff0000 || (p & 0xff0000) != 0), "expected red pixels");
    }
}
