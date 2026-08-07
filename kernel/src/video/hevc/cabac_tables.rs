//! HEVC CABAC context tables — **generated, do not edit**.
//!
//! Produced by `tools/gen_hevc_tables.py` from FFmpeg's `libavcodec/hevc/cabac.c`
//! (the values are ITU-T H.265's own tables; see THIRDPARTY-LICENSES.md).
//! Regenerate with:
//!
//! ```sh
//! python3 tools/gen_hevc_tables.py --fetch
//! ```
//!
//! 199 contexts across 49 syntax elements. The **offsets** matter as much as the
//! values: a wrong one decodes an element against another element's
//! probabilities, which does not fail — it produces a picture that is wrong in a
//! way that looks like a different bug entirely.
//!
//! The arithmetic engine is *not* here. HEVC shares H.264's `rangeTabLPS` and
//! state-transition tables, which are already in-tree as
//! [`crate::video::h264::cabac_tables`].

#![allow(clippy::all)]

pub const HEVC_CONTEXTS: usize = 199;
/// How many of those the base profile actually initialises; the rest
/// belong to the range extensions and are zero.
pub const HEVC_CONTEXTS_USED: usize = 179;

/// Per-syntax-element base context index (FFmpeg's `CABAC_ELEMS` order).
/// 1 context.
pub const SAO_MERGE_FLAG: usize = 0;
/// 1 context.
pub const SAO_TYPE_IDX: usize = 1;
/// 0 contexts.
pub const SAO_EO_CLASS: usize = 2;
/// 0 contexts.
pub const SAO_BAND_POSITION: usize = 2;
/// 0 contexts.
pub const SAO_OFFSET_ABS: usize = 2;
/// 0 contexts.
pub const SAO_OFFSET_SIGN: usize = 2;
/// 0 contexts.
pub const END_OF_SLICE_FLAG: usize = 2;
/// 3 contexts.
pub const SPLIT_CODING_UNIT_FLAG: usize = 2;
/// 1 context.
pub const CU_TRANSQUANT_BYPASS_FLAG: usize = 5;
/// 3 contexts.
pub const SKIP_FLAG: usize = 6;
/// 3 contexts.
pub const CU_QP_DELTA: usize = 9;
/// 1 context.
pub const PRED_MODE_FLAG: usize = 12;
/// 4 contexts.
pub const PART_MODE: usize = 13;
/// 0 contexts.
pub const PCM_FLAG: usize = 17;
/// 1 context.
pub const PREV_INTRA_LUMA_PRED_FLAG: usize = 17;
/// 0 contexts.
pub const MPM_IDX: usize = 18;
/// 0 contexts.
pub const REM_INTRA_LUMA_PRED_MODE: usize = 18;
/// 2 contexts.
pub const INTRA_CHROMA_PRED_MODE: usize = 18;
/// 1 context.
pub const MERGE_FLAG: usize = 20;
/// 1 context.
pub const MERGE_IDX: usize = 21;
/// 5 contexts.
pub const INTER_PRED_IDC: usize = 22;
/// 2 contexts.
pub const REF_IDX_L0: usize = 27;
/// 2 contexts.
pub const REF_IDX_L1: usize = 29;
/// 2 contexts.
pub const ABS_MVD_GREATER0_FLAG: usize = 31;
/// 2 contexts.
pub const ABS_MVD_GREATER1_FLAG: usize = 33;
/// 0 contexts.
pub const ABS_MVD_MINUS2: usize = 35;
/// 0 contexts.
pub const MVD_SIGN_FLAG: usize = 35;
/// 1 context.
pub const MVP_LX_FLAG: usize = 35;
/// 1 context.
pub const NO_RESIDUAL_DATA_FLAG: usize = 36;
/// 3 contexts.
pub const SPLIT_TRANSFORM_FLAG: usize = 37;
/// 2 contexts.
pub const CBF_LUMA: usize = 40;
/// 5 contexts.
pub const CBF_CB_CR: usize = 42;
/// 2 contexts.
pub const TRANSFORM_SKIP_FLAG: usize = 47;
/// 2 contexts.
pub const EXPLICIT_RDPCM_FLAG: usize = 49;
/// 2 contexts.
pub const EXPLICIT_RDPCM_DIR_FLAG: usize = 51;
/// 18 contexts.
pub const LAST_SIGNIFICANT_COEFF_X_PREFIX: usize = 53;
/// 18 contexts.
pub const LAST_SIGNIFICANT_COEFF_Y_PREFIX: usize = 71;
/// 0 contexts.
pub const LAST_SIGNIFICANT_COEFF_X_SUFFIX: usize = 89;
/// 0 contexts.
pub const LAST_SIGNIFICANT_COEFF_Y_SUFFIX: usize = 89;
/// 4 contexts.
pub const SIGNIFICANT_COEFF_GROUP_FLAG: usize = 89;
/// 44 contexts.
pub const SIGNIFICANT_COEFF_FLAG: usize = 93;
/// 24 contexts.
pub const COEFF_ABS_LEVEL_GREATER1_FLAG: usize = 137;
/// 6 contexts.
pub const COEFF_ABS_LEVEL_GREATER2_FLAG: usize = 161;
/// 0 contexts.
pub const COEFF_ABS_LEVEL_REMAINING: usize = 167;
/// 0 contexts.
pub const COEFF_SIGN_FLAG: usize = 167;
/// 8 contexts.
pub const LOG2_RES_SCALE_ABS: usize = 167;
/// 2 contexts.
pub const RES_SCALE_SIGN_FLAG: usize = 175;
/// 1 context.
pub const CU_CHROMA_QP_OFFSET_FLAG: usize = 177;
/// 1 context.
pub const CU_CHROMA_QP_OFFSET_IDX: usize = 178;

/// `init_values[init_type][ctx]` — the specification's initialisation
/// bytes. `init_type` is `2 - slice_type`, flipped by `cabac_init_flag`
/// on a non-I slice.
pub const INIT_VALUES: [[u8; 199]; 3] = [
[
    153, 200, 139, 141, 157, 154, 154, 154, 154, 154, 154, 154, 154, 184, 154, 154,
    154, 184, 63, 139, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
    154, 154, 154, 154, 154, 153, 138, 138, 111, 141, 94, 138, 182, 154, 154, 139,
    139, 139, 139, 139, 139, 110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111,
    143, 127, 111, 79, 108, 123, 63, 110, 110, 124, 125, 140, 153, 125, 127, 140,
    109, 111, 143, 127, 111, 79, 108, 123, 63, 91, 171, 134, 141, 111, 111, 125,
    110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107, 125, 141, 179,
    153, 125, 107, 125, 141, 179, 153, 125, 140, 139, 182, 182, 152, 136, 152, 136,
    153, 136, 139, 111, 136, 139, 111, 141, 111, 140, 92, 137, 138, 140, 152, 138,
    139, 153, 74, 149, 92, 139, 107, 122, 152, 140, 179, 166, 182, 140, 227, 122,
    197, 138, 153, 136, 167, 152, 152, 154, 154, 154, 154, 154, 154, 154, 154, 154,
    154, 154, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
],
[
    153, 185, 107, 139, 126, 154, 197, 185, 201, 154, 154, 154, 149, 154, 139, 154,
    154, 154, 152, 139, 110, 122, 95, 79, 63, 31, 31, 153, 153, 153, 153, 140,
    198, 140, 198, 168, 79, 124, 138, 94, 153, 111, 149, 107, 167, 154, 154, 139,
    139, 139, 139, 139, 139, 125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110,
    111, 111, 95, 94, 108, 123, 108, 125, 110, 94, 110, 95, 79, 125, 111, 110,
    78, 110, 111, 111, 95, 94, 108, 123, 108, 121, 140, 61, 154, 155, 154, 139,
    153, 139, 123, 123, 63, 153, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136,
    153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 123, 123, 107, 121, 107, 121,
    167, 151, 183, 140, 151, 183, 140, 140, 140, 154, 196, 196, 167, 154, 152, 167,
    182, 182, 134, 149, 136, 153, 121, 136, 137, 169, 194, 166, 167, 154, 167, 137,
    182, 107, 167, 91, 122, 107, 167, 154, 154, 154, 154, 154, 154, 154, 154, 154,
    154, 154, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
],
[
    153, 160, 107, 139, 126, 154, 197, 185, 201, 154, 154, 154, 134, 154, 139, 154,
    154, 183, 152, 139, 154, 137, 95, 79, 63, 31, 31, 153, 153, 153, 153, 169,
    198, 169, 198, 168, 79, 224, 167, 122, 153, 111, 149, 92, 167, 154, 154, 139,
    139, 139, 139, 139, 139, 125, 110, 124, 110, 95, 94, 125, 111, 111, 79, 125,
    126, 111, 111, 79, 108, 123, 93, 125, 110, 124, 110, 95, 94, 125, 111, 111,
    79, 125, 126, 111, 111, 79, 108, 123, 93, 121, 140, 61, 154, 170, 154, 139,
    153, 139, 123, 123, 63, 124, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136,
    153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 138, 138, 122, 121, 122, 121,
    167, 151, 183, 140, 151, 183, 140, 140, 140, 154, 196, 167, 167, 154, 152, 167,
    182, 182, 134, 149, 136, 153, 121, 136, 122, 169, 208, 166, 167, 154, 152, 167,
    182, 107, 167, 91, 107, 107, 167, 154, 154, 154, 154, 154, 154, 154, 154, 154,
    154, 154, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
],
];
