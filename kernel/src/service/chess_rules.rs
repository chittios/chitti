//! **Chess rules helpers** for the UI agent — pure functions, no I/O.
//!
//! Used by the Chess SOUL runtime for legal-move hints and to refuse
//! obviously illegal `from→to` updates. Not a full tournament engine
//! (no en-passant capture generation beyond the FEN ep field, limited
//! castling validation, no underpromotion choices). Good enough for an
//! on-device SOUL-driven board and unit-testable offline.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// Standard start position (placement + side + castling + ep + half + full).
pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Parse `"e4"` → (file 0..7, rank 0..7) with rank 0 = white's first rank.
pub fn parse_square(sq: &str) -> Option<(u8, u8)> {
    let b = sq.as_bytes();
    if b.len() != 2 {
        return None;
    }
    let file = b[0].wrapping_sub(b'a');
    let rank = b[1].wrapping_sub(b'1');
    if file < 8 && rank < 8 {
        Some((file, rank))
    } else {
        None
    }
}

pub fn square_name(file: u8, rank: u8) -> String {
    let mut s = String::new();
    s.push((b'a' + file) as char);
    s.push((b'1' + rank) as char);
    s
}

/// Board with rank 0 = white's first rank (a1..h1), file 0 = a.
#[derive(Clone, Copy)]
pub struct Board {
    /// `sq[rank][file]`
    pub sq: [[char; 8]; 8],
    pub white_to_move: bool,
    pub castling: u8, // bit0 K, bit1 Q, bit2 k, bit3 q
    pub ep_file: Option<u8>,
}

impl Board {
    pub fn from_fen(fen: &str) -> Option<Self> {
        let mut parts = fen.split_whitespace();
        let placement = parts.next()?;
        let side = parts.next().unwrap_or("w");
        let castle = parts.next().unwrap_or("-");
        let ep = parts.next().unwrap_or("-");
        // FEN ranks are top-down (black's back rank first) → convert to rank0=white.
        let mut top = [[' '; 8]; 8];
        let mut r = 0usize;
        let mut f = 0usize;
        for c in placement.chars() {
            if r >= 8 {
                break;
            }
            match c {
                '/' => {
                    r += 1;
                    f = 0;
                }
                '1'..='8' => f = (f + (c as u8 - b'0') as usize).min(8),
                p if p.is_ascii_alphabetic() => {
                    if f < 8 {
                        top[r][f] = p;
                        f += 1;
                    }
                }
                _ => {}
            }
        }
        let mut sq = [[' '; 8]; 8];
        for rank in 0..8 {
            sq[rank] = top[7 - rank];
        }
        let mut castling = 0u8;
        for c in castle.chars() {
            match c {
                'K' => castling |= 1,
                'Q' => castling |= 2,
                'k' => castling |= 4,
                'q' => castling |= 8,
                _ => {}
            }
        }
        let ep_file = if ep != "-" && ep.len() >= 1 {
            let f = ep.as_bytes()[0].wrapping_sub(b'a');
            if f < 8 {
                Some(f)
            } else {
                None
            }
        } else {
            None
        };
        Some(Self {
            sq,
            white_to_move: side != "b",
            castling,
            ep_file,
        })
    }

    pub fn to_fen_placement_and_side(&self) -> String {
        let mut out = String::new();
        for top_r in 0..8 {
            if top_r > 0 {
                out.push('/');
            }
            let rank = 7 - top_r;
            let mut empty = 0u8;
            for file in 0..8 {
                let p = self.sq[rank][file];
                if p == ' ' {
                    empty += 1;
                } else {
                    if empty > 0 {
                        out.push((b'0' + empty) as char);
                        empty = 0;
                    }
                    out.push(p);
                }
            }
            if empty > 0 {
                out.push((b'0' + empty) as char);
            }
        }
        out.push(' ');
        out.push(if self.white_to_move { 'w' } else { 'b' });
        out.push(' ');
        let mut c = String::new();
        if self.castling & 1 != 0 {
            c.push('K');
        }
        if self.castling & 2 != 0 {
            c.push('Q');
        }
        if self.castling & 4 != 0 {
            c.push('k');
        }
        if self.castling & 8 != 0 {
            c.push('q');
        }
        if c.is_empty() {
            c.push('-');
        }
        out.push_str(&c);
        out.push_str(" - 0 1"); // ep/half/full simplified
        out
    }

    fn at(&self, file: u8, rank: u8) -> char {
        self.sq[rank as usize][file as usize]
    }

    fn set(&mut self, file: u8, rank: u8, p: char) {
        self.sq[rank as usize][file as usize] = p;
    }

    fn is_white(p: char) -> bool {
        p.is_ascii_uppercase()
    }

    fn is_enemy(&self, p: char) -> bool {
        if p == ' ' {
            return false;
        }
        Self::is_white(p) != self.white_to_move
    }

    fn is_friend(&self, p: char) -> bool {
        if p == ' ' {
            return false;
        }
        Self::is_white(p) == self.white_to_move
    }

    fn king_sq(&self, white: bool) -> Option<(u8, u8)> {
        let k = if white { 'K' } else { 'k' };
        for r in 0..8u8 {
            for f in 0..8u8 {
                if self.at(f, r) == k {
                    return Some((f, r));
                }
            }
        }
        None
    }

    /// Whether `white`'s king is attacked.
    pub fn in_check(&self, white: bool) -> bool {
        let Some((kf, kr)) = self.king_sq(white) else {
            return false;
        };
        self.square_attacked(kf, kr, !white)
    }

    fn square_attacked(&self, file: u8, rank: u8, by_white: bool) -> bool {
        // Pawn attacks
        let dir: i8 = if by_white { 1 } else { -1 };
        for df in [-1i8, 1] {
            let f = file as i8 + df;
            let r = rank as i8 - dir; // pawn attacks from behind relative to move dir
            if (0..8).contains(&f) && (0..8).contains(&r) {
                let p = self.at(f as u8, r as u8);
                if by_white && p == 'P' || !by_white && p == 'p' {
                    return true;
                }
            }
        }
        // Knight
        for (df, dr) in [
            (1i8, 2),
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
            (-2, 1),
            (-1, 2),
        ] {
            let f = file as i8 + df;
            let r = rank as i8 + dr;
            if (0..8).contains(&f) && (0..8).contains(&r) {
                let p = self.at(f as u8, r as u8);
                if by_white && p == 'N' || !by_white && p == 'n' {
                    return true;
                }
            }
        }
        // King
        for df in -1i8..=1 {
            for dr in -1i8..=1 {
                if df == 0 && dr == 0 {
                    continue;
                }
                let f = file as i8 + df;
                let r = rank as i8 + dr;
                if (0..8).contains(&f) && (0..8).contains(&r) {
                    let p = self.at(f as u8, r as u8);
                    if by_white && p == 'K' || !by_white && p == 'k' {
                        return true;
                    }
                }
            }
        }
        // Sliding: bishop/queen diagonals, rook/queen ranks
        let rays = [
            (1i8, 0, true),
            (-1, 0, true),
            (0, 1, true),
            (0, -1, true),
            (1, 1, false),
            (1, -1, false),
            (-1, 1, false),
            (-1, -1, false),
        ];
        for (df, dr, ortho) in rays {
            let mut f = file as i8 + df;
            let mut r = rank as i8 + dr;
            while (0..8).contains(&f) && (0..8).contains(&r) {
                let p = self.at(f as u8, r as u8);
                if p != ' ' {
                    let hits = if ortho {
                        by_white && (p == 'R' || p == 'Q') || !by_white && (p == 'r' || p == 'q')
                    } else {
                        by_white && (p == 'B' || p == 'Q') || !by_white && (p == 'b' || p == 'q')
                    };
                    if hits {
                        return true;
                    }
                    break;
                }
                f += df;
                r += dr;
            }
        }
        false
    }

    /// Pseudo-legal destinations for a piece on `from`, then filter checks.
    pub fn legal_to_squares(&self, from: &str) -> Vec<String> {
        let Some((ff, fr)) = parse_square(from) else {
            return Vec::new();
        };
        let p = self.at(ff, fr);
        if p == ' ' || Self::is_white(p) != self.white_to_move {
            return Vec::new();
        }
        let mut dests = Vec::new();
        self.gen_pseudo(ff, fr, p, &mut dests);
        dests
            .into_iter()
            .filter(|&(tf, tr)| {
                let mut b = *self;
                b.apply_raw(ff, fr, tf, tr, p);
                !b.in_check(self.white_to_move)
            })
            .map(|(tf, tr)| square_name(tf, tr))
            .collect()
    }

    fn gen_pseudo(&self, ff: u8, fr: u8, p: char, out: &mut Vec<(u8, u8)>) {
        let white = Self::is_white(p);
        let add = |out: &mut Vec<(u8, u8)>, f: i8, r: i8| {
            if (0..8).contains(&f) && (0..8).contains(&r) {
                let t = self.at(f as u8, r as u8);
                if t == ' ' || self.is_enemy(t) {
                    out.push((f as u8, r as u8));
                }
            }
        };
        match p.to_ascii_lowercase() {
            'p' => {
                let dir: i8 = if white { 1 } else { -1 };
                let r1 = fr as i8 + dir;
                if (0..8).contains(&r1) && self.at(ff, r1 as u8) == ' ' {
                    out.push((ff, r1 as u8));
                    let start = if white { 1u8 } else { 6u8 };
                    let r2 = fr as i8 + 2 * dir;
                    if fr == start && (0..8).contains(&r2) && self.at(ff, r2 as u8) == ' ' {
                        out.push((ff, r2 as u8));
                    }
                }
                for df in [-1i8, 1] {
                    let f = ff as i8 + df;
                    let r = fr as i8 + dir;
                    if (0..8).contains(&f) && (0..8).contains(&r) {
                        let t = self.at(f as u8, r as u8);
                        if self.is_enemy(t) {
                            out.push((f as u8, r as u8));
                        }
                        // en passant
                        if t == ' ' && self.ep_file == Some(f as u8) {
                            let ep_rank = if white { 5u8 } else { 2u8 };
                            if r as u8 == ep_rank {
                                out.push((f as u8, r as u8));
                            }
                        }
                    }
                }
            }
            'n' => {
                for (df, dr) in [
                    (1i8, 2),
                    (2, 1),
                    (2, -1),
                    (1, -2),
                    (-1, -2),
                    (-2, -1),
                    (-2, 1),
                    (-1, 2),
                ] {
                    add(out, ff as i8 + df, fr as i8 + dr);
                }
            }
            'b' => self.slide(ff, fr, &[(1, 1), (1, -1), (-1, 1), (-1, -1)], out),
            'r' => self.slide(ff, fr, &[(1, 0), (-1, 0), (0, 1), (0, -1)], out),
            'q' => self.slide(
                ff,
                fr,
                &[(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)],
                out,
            ),
            'k' => {
                for df in -1i8..=1 {
                    for dr in -1i8..=1 {
                        if df == 0 && dr == 0 {
                            continue;
                        }
                        add(out, ff as i8 + df, fr as i8 + dr);
                    }
                }
                // Castling (simplified: empty between, rights set; path-not-check checked later).
                if white && fr == 0 && ff == 4 {
                    if self.castling & 1 != 0
                        && self.at(5, 0) == ' '
                        && self.at(6, 0) == ' '
                        && !self.square_attacked(4, 0, false)
                        && !self.square_attacked(5, 0, false)
                    {
                        out.push((6, 0));
                    }
                    if self.castling & 2 != 0
                        && self.at(3, 0) == ' '
                        && self.at(2, 0) == ' '
                        && self.at(1, 0) == ' '
                        && !self.square_attacked(4, 0, false)
                        && !self.square_attacked(3, 0, false)
                    {
                        out.push((2, 0));
                    }
                }
                if !white && fr == 7 && ff == 4 {
                    if self.castling & 4 != 0
                        && self.at(5, 7) == ' '
                        && self.at(6, 7) == ' '
                        && !self.square_attacked(4, 7, true)
                        && !self.square_attacked(5, 7, true)
                    {
                        out.push((6, 7));
                    }
                    if self.castling & 8 != 0
                        && self.at(3, 7) == ' '
                        && self.at(2, 7) == ' '
                        && self.at(1, 7) == ' '
                        && !self.square_attacked(4, 7, true)
                        && !self.square_attacked(3, 7, true)
                    {
                        out.push((2, 7));
                    }
                }
            }
            _ => {}
        }
    }

    fn slide(&self, ff: u8, fr: u8, dirs: &[(i8, i8)], out: &mut Vec<(u8, u8)>) {
        for &(df, dr) in dirs {
            let mut f = ff as i8 + df;
            let mut r = fr as i8 + dr;
            while (0..8).contains(&f) && (0..8).contains(&r) {
                let t = self.at(f as u8, r as u8);
                if t == ' ' {
                    out.push((f as u8, r as u8));
                } else {
                    if self.is_enemy(t) {
                        out.push((f as u8, r as u8));
                    }
                    break;
                }
                f += df;
                r += dr;
            }
        }
    }

    fn apply_raw(&mut self, ff: u8, fr: u8, tf: u8, tr: u8, p: char) {
        self.set(ff, fr, ' ');
        // Castling rook move
        if p == 'K' && ff == 4 && fr == 0 && tr == 0 {
            if tf == 6 {
                self.set(7, 0, ' ');
                self.set(5, 0, 'R');
            }
            if tf == 2 {
                self.set(0, 0, ' ');
                self.set(3, 0, 'R');
            }
        }
        if p == 'k' && ff == 4 && fr == 7 && tr == 7 {
            if tf == 6 {
                self.set(7, 7, ' ');
                self.set(5, 7, 'r');
            }
            if tf == 2 {
                self.set(0, 7, ' ');
                self.set(3, 7, 'r');
            }
        }
        // En passant capture
        if p.to_ascii_lowercase() == 'p' && tf != ff && self.at(tf, tr) == ' ' {
            self.set(tf, fr, ' ');
        }
        // Promotion → queen
        let mut piece = p;
        if p == 'P' && tr == 7 {
            piece = 'Q';
        }
        if p == 'p' && tr == 0 {
            piece = 'q';
        }
        self.set(tf, tr, piece);
        self.white_to_move = !self.white_to_move;
        // Clear castling rights crudely
        if p == 'K' {
            self.castling &= !3;
        }
        if p == 'k' {
            self.castling &= !12;
        }
        if p == 'R' && ff == 0 && fr == 0 {
            self.castling &= !2;
        }
        if p == 'R' && ff == 7 && fr == 0 {
            self.castling &= !1;
        }
        if p == 'r' && ff == 0 && fr == 7 {
            self.castling &= !8;
        }
        if p == 'r' && ff == 7 && fr == 7 {
            self.castling &= !4;
        }
        self.ep_file = None;
        if p.to_ascii_lowercase() == 'p' && (tr as i8 - fr as i8).abs() == 2 {
            self.ep_file = Some(ff);
        }
    }
}

/// Legal destination squares from `from` given `fen` (comma-separated, or
/// `"none"` / empty).
pub fn legal_moves(fen: &str, from: &str) -> String {
    let Some(b) = Board::from_fen(fen) else {
        return String::from("error:bad_fen");
    };
    let dests = b.legal_to_squares(from);
    if dests.is_empty() {
        String::from("none")
    } else {
        dests.join(",")
    }
}

/// Apply a legal move; returns new FEN or an error string starting with `error:`.
pub fn try_move(fen: &str, from: &str, to: &str) -> Result<String, String> {
    let b = Board::from_fen(fen).ok_or_else(|| String::from("error:bad_fen"))?;
    let legal = b.legal_to_squares(from);
    if !legal.iter().any(|s| s == to) {
        return Err(format!("error:illegal {from}->{to} (legal: {})", legal.join(",")));
    }
    let (ff, fr) = parse_square(from).ok_or_else(|| String::from("error:bad_from"))?;
    let (tf, tr) = parse_square(to).ok_or_else(|| String::from("error:bad_to"))?;
    let p = b.at(ff, fr);
    let mut nb = b;
    nb.apply_raw(ff, fr, tf, tr, p);
    Ok(nb.to_fen_placement_and_side())
}

/// Whether the side to move is in check.
pub fn in_check(fen: &str) -> bool {
    Board::from_fen(fen).map(|b| b.in_check(b.white_to_move)).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn start_position_pawn_and_knight_moves() {
        let b = Board::from_fen(START_FEN).unwrap();
        let e2 = b.legal_to_squares("e2");
        assert!(e2.iter().any(|s| s == "e3"));
        assert!(e2.iter().any(|s| s == "e4"));
        assert!(!e2.iter().any(|s| s == "e5"));
        let nb1 = b.legal_to_squares("b1");
        assert!(nb1.iter().any(|s| s == "a3") || nb1.iter().any(|s| s == "c3"));
        // Wrong side to move piece
        assert!(b.legal_to_squares("e7").is_empty());
    }

    #[test_case]
    fn try_move_updates_fen_and_rejects_illegal() {
        let fen = START_FEN;
        let next = try_move(fen, "e2", "e4").unwrap();
        assert!(next.starts_with("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b"));
        assert!(try_move(fen, "e2", "e5").is_err());
        assert!(try_move(fen, "e7", "e5").is_err()); // black to move? white's turn
    }

    #[test_case]
    fn legal_moves_string_format() {
        let s = legal_moves(START_FEN, "g1");
        assert!(s.contains("f3") || s.contains("h3"));
        assert_eq!(legal_moves(START_FEN, "a3"), "none");
    }
}
