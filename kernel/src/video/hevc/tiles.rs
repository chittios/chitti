//! Tile geometry for HEVC (H.265 §6.5): column/row boundaries and the
//! raster-scan ↔ tile-scan address maps.
//!
//! The CTU walk is in **tile-scan order** (`ctb_addr_ts`), not picture raster
//! order. Getting the maps wrong reads the right bits against the wrong CTB
//! coordinates — a picture that looks shuffled rather than broken.

use alloc::vec::Vec;

/// Resolved tile layout for one PPS against one SPS geometry.
#[derive(Clone, Debug)]
pub struct TileMap {
    pub num_columns: usize,
    pub num_rows: usize,
    /// Cumulative CTB column boundaries; length `num_columns + 1`, `col_bd[0]=0`.
    pub col_bd: Vec<usize>,
    /// Cumulative CTB row boundaries; length `num_rows + 1`.
    pub row_bd: Vec<usize>,
    /// Column width in CTBs.
    pub column_width: Vec<usize>,
    /// Row height in CTBs.
    pub row_height: Vec<usize>,
    /// `ctb_addr_rs_to_ts[rs] = ts`
    pub rs_to_ts: Vec<usize>,
    /// `ctb_addr_ts_to_rs[ts] = rs`
    pub ts_to_rs: Vec<usize>,
    /// Tile id of each CTB in tile-scan order.
    pub tile_id: Vec<usize>,
}

impl TileMap {
    /// Build a single-tile map covering the whole picture (the common case).
    pub fn single(ctb_w: usize, ctb_h: usize) -> TileMap {
        let n = ctb_w * ctb_h;
        TileMap {
            num_columns: 1,
            num_rows: 1,
            col_bd: alloc::vec![0, ctb_w],
            row_bd: alloc::vec![0, ctb_h],
            column_width: alloc::vec![ctb_w],
            row_height: alloc::vec![ctb_h],
            rs_to_ts: (0..n).collect(),
            ts_to_rs: (0..n).collect(),
            tile_id: alloc::vec![0; n],
        }
    }

    /// Build from a PPS tile description and the SPS CTB grid.
    ///
    /// `column_width` / `row_height` are in CTBs; when `uniform` they may be
    /// empty and are filled with the uniform-spacing rule (H.265 §6.5.1).
    pub fn from_pps(
        ctb_w: usize,
        ctb_h: usize,
        num_columns: usize,
        num_rows: usize,
        uniform: bool,
        column_width: &[u32],
        row_height: &[u32],
    ) -> Result<TileMap, &'static str> {
        if num_columns == 0 || num_rows == 0 || num_columns > ctb_w || num_rows > ctb_h {
            return Err("hevc tiles: implausible tile grid");
        }
        let mut col_w = Vec::with_capacity(num_columns);
        let mut row_h = Vec::with_capacity(num_rows);
        if uniform {
            for i in 0..num_columns {
                col_w.push(((i + 1) * ctb_w) / num_columns - (i * ctb_w) / num_columns);
            }
            for i in 0..num_rows {
                row_h.push(((i + 1) * ctb_h) / num_rows - (i * ctb_h) / num_rows);
            }
        } else {
            if column_width.len() != num_columns.saturating_sub(1)
                || row_height.len() != num_rows.saturating_sub(1)
            {
                return Err("hevc tiles: non-uniform width list length");
            }
            let mut sum = 0usize;
            for &w in column_width {
                col_w.push(w as usize);
                sum += w as usize;
            }
            if sum >= ctb_w {
                return Err("hevc tiles: column widths exceed picture");
            }
            col_w.push(ctb_w - sum);
            sum = 0;
            for &h in row_height {
                row_h.push(h as usize);
                sum += h as usize;
            }
            if sum >= ctb_h {
                return Err("hevc tiles: row heights exceed picture");
            }
            row_h.push(ctb_h - sum);
        }

        let mut col_bd = alloc::vec![0usize; num_columns + 1];
        for i in 0..num_columns {
            col_bd[i + 1] = col_bd[i] + col_w[i];
        }
        let mut row_bd = alloc::vec![0usize; num_rows + 1];
        for i in 0..num_rows {
            row_bd[i + 1] = row_bd[i] + row_h[i];
        }

        let n = ctb_w * ctb_h;
        let mut rs_to_ts = alloc::vec![0usize; n];
        let mut ts_to_rs = alloc::vec![0usize; n];
        for rs in 0..n {
            let tb_x = rs % ctb_w;
            let tb_y = rs / ctb_w;
            let mut tile_x = 0usize;
            let mut tile_y = 0usize;
            for i in 0..num_columns {
                if tb_x < col_bd[i + 1] {
                    tile_x = i;
                    break;
                }
            }
            for i in 0..num_rows {
                if tb_y < row_bd[i + 1] {
                    tile_y = i;
                    break;
                }
            }
            let mut val = 0usize;
            for i in 0..tile_x {
                val += row_h[tile_y] * col_w[i];
            }
            for i in 0..tile_y {
                val += ctb_w * row_h[i];
            }
            val += (tb_y - row_bd[tile_y]) * col_w[tile_x] + (tb_x - col_bd[tile_x]);
            rs_to_ts[rs] = val;
            ts_to_rs[val] = rs;
        }

        let mut tile_id = alloc::vec![0usize; n];
        let mut tid = 0usize;
        for j in 0..num_rows {
            for i in 0..num_columns {
                for y in row_bd[j]..row_bd[j + 1] {
                    for x in col_bd[i]..col_bd[i + 1] {
                        tile_id[rs_to_ts[y * ctb_w + x]] = tid;
                    }
                }
                tid += 1;
            }
        }

        Ok(TileMap {
            num_columns,
            num_rows,
            col_bd,
            row_bd,
            column_width: col_w,
            row_height: row_h,
            rs_to_ts,
            ts_to_rs,
            tile_id,
        })
    }

    /// Tile id of a raster-scan CTB address.
    #[inline]
    pub fn tile_id_rs(&self, rs: usize) -> usize {
        self.tile_id[self.rs_to_ts[rs]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn single_tile_is_identity() {
        let m = TileMap::single(4, 3);
        assert_eq!(m.num_columns, 1);
        assert_eq!(m.num_rows, 1);
        for i in 0..12 {
            assert_eq!(m.rs_to_ts[i], i);
            assert_eq!(m.ts_to_rs[i], i);
            assert_eq!(m.tile_id[i], 0);
        }
    }

    #[test_case]
    fn uniform_2x2_on_4x4_grid() {
        // FFmpeg's uniform rule: col widths [2,2], row heights [2,2].
        let m = TileMap::from_pps(4, 4, 2, 2, true, &[], &[]).unwrap();
        assert_eq!(m.column_width, &[2, 2]);
        assert_eq!(m.row_height, &[2, 2]);
        // Tile 0 is the top-left 2x2 of CTBs in raster; tile-scan walks it first.
        // rs 0,1,4,5 → tile 0 → ts 0,1,2,3
        assert_eq!(m.rs_to_ts[0], 0);
        assert_eq!(m.rs_to_ts[1], 1);
        assert_eq!(m.rs_to_ts[4], 2);
        assert_eq!(m.rs_to_ts[5], 3);
        // Tile 1 top-right
        assert_eq!(m.rs_to_ts[2], 4);
        assert_eq!(m.rs_to_ts[3], 5);
        assert_eq!(m.rs_to_ts[6], 6);
        assert_eq!(m.rs_to_ts[7], 7);
        // Bottom-left tile 2
        assert_eq!(m.rs_to_ts[8], 8);
        // Bottom-right tile 3
        assert_eq!(m.rs_to_ts[15], 15);
        // Inverse
        for rs in 0..16 {
            assert_eq!(m.ts_to_rs[m.rs_to_ts[rs]], rs);
        }
    }

    #[test_case]
    fn non_uniform_columns() {
        // 5 CTB cols, widths 1 + 2 + (rest 2)
        let m = TileMap::from_pps(5, 2, 3, 1, false, &[1, 2], &[]).unwrap();
        assert_eq!(m.column_width, &[1, 2, 2]);
        assert_eq!(m.col_bd, &[0, 1, 3, 5]);
    }
}
