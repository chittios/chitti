//! **Chunked stateful tenant ABI** — the ring-3 shape for long-running decoders
//! (H.264, AAC, ONNX) that cannot finish in one crossing.
//!
//! [`crate::synapse::tenant::ImageTenant`] resets its bump arena on every entry
//! and decodes a whole file in one shot. That is wrong for a multi-second AAC
//! decode or an H.264 stream: Ctrl+C and `shell::upkeep()` are standing rules,
//! and a tenant has no device access by construction. The working shape is:
//!
//! 1. A **command word** in the startup block (`Init` / `Continue` / `Cancel`).
//! 2. **Decoder state that survives an entry** (the opposite of the image
//!    tenant's one-line arena reset).
//! 3. The kernel pumps `upkeep()` / `poll_interrupt()` **between** entries and
//!    re-enters until `Done` or `Cancelled`.
//!
//! This module pins the shared layout and status vocabulary. Concrete tenants
//! (H.264 bitstream→YUV, AAC frames, ONNX ops) mount it the way `imgdec`
//! mounts the image decoder.

/// Commands the kernel writes before each tenant entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum ChunkCmd {
    /// First entry: parse headers, size the arena, leave state warm.
    Init = 1,
    /// Decode up to `max_units` more units (frames / samples / ops).
    Continue = 2,
    /// Abandon state; tenant may free its own scratch.
    Cancel = 3,
    /// Video: decode display frame `max_units` (rewinds the engine if needed).
    Seek = 4,
    /// JS: dispatch a DOM event (`meta0` type, `meta1` target, `meta2/3` x/y).
    Event = 5,
}

impl ChunkCmd {
    pub fn from_u64(v: u64) -> Option<Self> {
        match v {
            1 => Some(Self::Init),
            2 => Some(Self::Continue),
            3 => Some(Self::Cancel),
            4 => Some(Self::Seek),
            5 => Some(Self::Event),
            _ => None,
        }
    }
}

/// Status the tenant writes before returning to the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum ChunkStatus {
    /// More units remain; kernel should re-enter with [`ChunkCmd::Continue`].
    NeedMore = 1,
    /// Stream complete; output is ready at the reported offsets.
    Done = 2,
    /// [`ChunkCmd::Cancel`] honoured (or Ctrl+C observed by the kernel).
    Cancelled = 3,
    /// Input was corrupt / unsupported.
    Failed = 4,
    /// Arena too small; tenant reports `HEAP_WANT` and the loader grows.
    OutOfMemory = 5,
}

impl ChunkStatus {
    pub fn from_u64(v: u64) -> Option<Self> {
        match v {
            1 => Some(Self::NeedMore),
            2 => Some(Self::Done),
            3 => Some(Self::Cancelled),
            4 => Some(Self::Failed),
            5 => Some(Self::OutOfMemory),
            _ => None,
        }
    }
}

/// What the tenant is decoding. Written by the tenant on [`ChunkCmd::Init`].
pub const KIND_AUDIO: u64 = 1;
pub const KIND_VIDEO: u64 = 2;
pub const KIND_JS: u64 = 3;

/// Packed-DOM page envelope (`"JDOM"` little-endian) for the jsint tenant.
pub const JS_DOM_MAGIC: u32 = 0x4D4F_444A;
/// Raw-HTML page envelope (`"HTML"` little-endian): the tenant tokenizes.
pub const JS_HTML_MAGIC: u32 = 0x4C4D_5448;
/// Mutation log magic (`"JMUT"` little-endian).
pub const JS_MUT_MAGIC: u32 = 0x5455_4D4A;
/// `JS_HTML_MAGIC` flags: run extracted `<script>` bodies after parse.
pub const JS_HTML_RUN_SCRIPTS: u32 = 1;

pub const JS_OP_TITLE: u8 = 1;
pub const JS_OP_NAV: u8 = 2;
pub const JS_OP_TEXT: u8 = 3;
pub const JS_OP_VALUE: u8 = 4;
pub const JS_OP_STYLE: u8 = 5;
pub const JS_OP_HIDDEN: u8 = 6;
pub const JS_OP_LOG: u8 = 7;
pub const JS_OP_INTERACTIVE: u8 = 8;
pub const JS_OP_SCROLL: u8 = 9;
/// Full element-table snapshot (createElement/appendChild trees).
pub const JS_OP_SNAPSHOT: u8 = 10;
/// Concatenated `<style>` text extracted while parsing HTML in the tenant.
pub const JS_OP_CSS: u8 = 11;

pub const JS_EV_CLICK: u64 = 1;
pub const JS_EV_INPUT: u64 = 2;
pub const JS_EV_KEYDOWN: u64 = 3;

/// Startup / result block shared between kernel and tenant.
///
/// Layout is fixed so H.264/AAC/JS guests can `#[path]`-mount the same
/// constants. All multi-byte fields are little-endian `u64` slots.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ChunkBlock {
    /// [`ChunkCmd`] on entry; [`ChunkStatus`] on exit.
    pub cmd_or_status: u64,
    /// Input buffer guest-virtual address.
    pub input_ptr: u64,
    /// Input length in bytes.
    pub input_len: u64,
    /// Max units to process this entry (frames / AAC frames / JS scripts).
    pub max_units: u64,
    /// Units completed across all entries so far (tenant updates).
    pub units_done: u64,
    /// Output buffer guest-VA (Y plane, PCM, JS log) — tenant-reported.
    pub output_ptr: u64,
    /// Output length in bytes.
    pub output_len: u64,
    /// Arena bytes the tenant wants on an OOM retry.
    pub heap_want: u64,
    /// Opaque decoder-state cookie the tenant may stash (must survive Continue).
    pub state_cookie: u64,
    /// [`KIND_AUDIO`] / [`KIND_VIDEO`] / [`KIND_JS`].
    pub kind: u64,
    /// Audio: sample rate. Video: width. JS: host-op code.
    pub meta0: u64,
    /// Audio: channels. Video: height. JS: host-op arg.
    pub meta1: u64,
    /// Video: Cb plane guest-VA.
    pub meta2: u64,
    /// Video: Cr plane guest-VA.
    pub meta3: u64,
    /// Video: Y plane length.
    pub meta4: u64,
    /// Video: Cb plane length.
    pub meta5: u64,
    /// Video: Cr plane length.
    pub meta6: u64,
    /// Heap the loader mapped (guest-VA).
    pub heap_ptr: u64,
    /// Heap length in bytes.
    pub heap_len: u64,
}

impl ChunkBlock {
    pub const BYTES: usize = core::mem::size_of::<Self>();

    pub fn new_init(input_ptr: u64, input_len: u64, max_units: u64) -> Self {
        Self {
            cmd_or_status: ChunkCmd::Init as u64,
            input_ptr,
            input_len,
            max_units,
            units_done: 0,
            output_ptr: 0,
            output_len: 0,
            heap_want: 0,
            state_cookie: 0,
            kind: 0,
            meta0: 0,
            meta1: 0,
            meta2: 0,
            meta3: 0,
            meta4: 0,
            meta5: 0,
            meta6: 0,
            heap_ptr: 0,
            heap_len: 0,
        }
    }

    pub fn as_continue(&mut self, max_units: u64) {
        self.cmd_or_status = ChunkCmd::Continue as u64;
        self.max_units = max_units;
    }

    pub fn as_cancel(&mut self) {
        self.cmd_or_status = ChunkCmd::Cancel as u64;
    }

    /// Encode into a little-endian byte buffer (for the startup page).
    pub fn pack(&self, out: &mut [u8]) -> Option<()> {
        if out.len() < Self::BYTES {
            return None;
        }
        let words = [
            self.cmd_or_status,
            self.input_ptr,
            self.input_len,
            self.max_units,
            self.units_done,
            self.output_ptr,
            self.output_len,
            self.heap_want,
            self.state_cookie,
            self.kind,
            self.meta0,
            self.meta1,
            self.meta2,
            self.meta3,
            self.meta4,
            self.meta5,
            self.meta6,
            self.heap_ptr,
            self.heap_len,
        ];
        for (i, w) in words.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        Some(())
    }

    pub fn unpack(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::BYTES {
            return None;
        }
        let word = |i: usize| -> Option<u64> {
            let bytes: [u8; 8] = buf[i * 8..i * 8 + 8].try_into().ok()?;
            Some(u64::from_le_bytes(bytes))
        };
        Some(Self {
            cmd_or_status: word(0)?,
            input_ptr: word(1)?,
            input_len: word(2)?,
            max_units: word(3)?,
            units_done: word(4)?,
            output_ptr: word(5)?,
            output_len: word(6)?,
            heap_want: word(7)?,
            state_cookie: word(8)?,
            kind: word(9)?,
            meta0: word(10)?,
            meta1: word(11)?,
            meta2: word(12)?,
            meta3: word(13)?,
            meta4: word(14)?,
            meta5: word(15)?,
            meta6: word(16)?,
            heap_ptr: word(17)?,
            heap_len: word(18)?,
        })
    }

    pub fn status(&self) -> Option<ChunkStatus> {
        ChunkStatus::from_u64(self.cmd_or_status)
    }

    pub fn cmd(&self) -> Option<ChunkCmd> {
        ChunkCmd::from_u64(self.cmd_or_status)
    }
}

/// Decide whether the kernel should re-enter the tenant after this status.
pub fn should_continue(status: ChunkStatus) -> bool {
    matches!(status, ChunkStatus::NeedMore | ChunkStatus::OutOfMemory)
}

pub fn put_u16(out: &mut alloc::vec::Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}
pub fn put_u32(out: &mut alloc::vec::Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
pub fn put_bytes(out: &mut alloc::vec::Vec<u8>, b: &[u8]) {
    put_u32(out, b.len() as u32);
    out.extend_from_slice(b);
}
pub fn put_str(out: &mut alloc::vec::Vec<u8>, s: &str) {
    put_bytes(out, s.as_bytes());
}
pub fn get_u16(src: &[u8], off: &mut usize) -> Option<u16> {
    let s = *off;
    if s + 2 > src.len() {
        return None;
    }
    *off = s + 2;
    Some(u16::from_le_bytes(src[s..s + 2].try_into().ok()?))
}
pub fn get_u32(src: &[u8], off: &mut usize) -> Option<u32> {
    let s = *off;
    if s + 4 > src.len() {
        return None;
    }
    *off = s + 4;
    Some(u32::from_le_bytes(src[s..s + 4].try_into().ok()?))
}
pub fn get_bytes<'a>(src: &'a [u8], off: &mut usize) -> Option<&'a [u8]> {
    let n = get_u32(src, off)? as usize;
    let s = *off;
    let e = s.checked_add(n)?;
    if e > src.len() {
        return None;
    }
    *off = e;
    Some(&src[s..e])
}
pub fn get_str<'a>(src: &'a [u8], off: &mut usize) -> Option<&'a str> {
    core::str::from_utf8(get_bytes(src, off)?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn chunk_block_roundtrips_and_commands_parse() {
        let mut b = ChunkBlock::new_init(0x1000, 4096, 32);
        assert_eq!(b.cmd(), Some(ChunkCmd::Init));
        b.as_continue(16);
        assert_eq!(b.cmd(), Some(ChunkCmd::Continue));
        assert_eq!(b.max_units, 16);
        b.state_cookie = 0xdead_beef;
        b.units_done = 8;
        b.kind = KIND_AUDIO;
        b.meta0 = 44100;
        b.meta1 = 2;
        let mut buf = [0u8; ChunkBlock::BYTES];
        b.pack(&mut buf).unwrap();
        let back = ChunkBlock::unpack(&buf).unwrap();
        assert_eq!(back.state_cookie, 0xdead_beef);
        assert_eq!(back.units_done, 8);
        assert_eq!(back.kind, KIND_AUDIO);
        assert_eq!(back.meta0, 44100);
        assert_eq!(back.cmd(), Some(ChunkCmd::Continue));
        b.kind = KIND_VIDEO;
        b.meta0 = 1920;
        b.meta1 = 1080;
        b.meta2 = 0x2000;
        b.meta3 = 0x3000;
        b.meta4 = 1920 * 1080;
        b.meta5 = 1920 * 1080 / 4;
        b.meta6 = 1920 * 1080 / 4;
        b.heap_ptr = 0x4000_0000;
        b.heap_len = 8 << 20;
        b.pack(&mut buf).unwrap();
        let back = ChunkBlock::unpack(&buf).unwrap();
        assert_eq!(back.kind, KIND_VIDEO);
        assert_eq!(back.meta0, 1920);
        assert_eq!(back.meta1, 1080);
        assert_eq!(back.meta2, 0x2000);
        assert_eq!(back.meta3, 0x3000);
        assert_eq!(back.meta4, 1920 * 1080);
        assert_eq!(back.heap_ptr, 0x4000_0000);
        assert_eq!(back.heap_len, 8 << 20);
        b.as_cancel();
        assert_eq!(b.cmd(), Some(ChunkCmd::Cancel));
    }

    #[test_case]
    fn audio_and_video_kind_slots_are_distinct() {
        assert_eq!(KIND_AUDIO, 1);
        assert_eq!(KIND_VIDEO, 2);
        let mut b = ChunkBlock::new_init(0, 0, 32);
        b.kind = KIND_AUDIO;
        b.meta0 = 44100;
        b.meta1 = 2;
        let mut buf = [0u8; ChunkBlock::BYTES];
        b.pack(&mut buf).unwrap();
        let back = ChunkBlock::unpack(&buf).unwrap();
        assert_eq!(back.kind, KIND_AUDIO);
        assert_eq!(back.meta0, 44100);
        assert_eq!(back.meta1, 2);
        assert_eq!(back.meta2, 0);
    }

    #[test_case]
    fn should_continue_only_for_need_more_and_oom() {
        assert!(should_continue(ChunkStatus::NeedMore));
        assert!(should_continue(ChunkStatus::OutOfMemory));
        assert!(!should_continue(ChunkStatus::Done));
        assert!(!should_continue(ChunkStatus::Failed));
        assert!(!should_continue(ChunkStatus::Cancelled));
    }

    #[test_case]
    fn unknown_cmd_and_status_are_refused() {
        assert!(ChunkCmd::from_u64(0).is_none());
        assert!(ChunkCmd::from_u64(99).is_none());
        assert!(ChunkStatus::from_u64(0).is_none());
        assert!(ChunkStatus::from_u64(99).is_none());
    }
}
