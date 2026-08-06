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
}

impl ChunkCmd {
    pub fn from_u64(v: u64) -> Option<Self> {
        match v {
            1 => Some(Self::Init),
            2 => Some(Self::Continue),
            3 => Some(Self::Cancel),
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

/// Startup / result block shared between kernel and tenant.
///
/// Layout is fixed so a future H.264/AAC/ONNX guest can `#[path]`-mount the
/// same constants. All multi-byte fields are little-endian `u64` slots.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ChunkBlock {
    /// [`ChunkCmd`] on entry; [`ChunkStatus`] on exit.
    pub cmd_or_status: u64,
    /// Input buffer guest-virtual address.
    pub input_ptr: u64,
    /// Input length in bytes.
    pub input_len: u64,
    /// Max units to process this entry (frames / AAC frames / ONNX nodes).
    pub max_units: u64,
    /// Units completed across all entries so far (tenant updates).
    pub units_done: u64,
    /// Output buffer guest-VA (YUV plane, PCM, etc.) — tenant-reported.
    pub output_ptr: u64,
    /// Output length in bytes.
    pub output_len: u64,
    /// Arena bytes the tenant wants on an OOM retry.
    pub heap_want: u64,
    /// Opaque decoder-state cookie the tenant may stash (must survive Continue).
    pub state_cookie: u64,
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
        let mut buf = [0u8; ChunkBlock::BYTES];
        b.pack(&mut buf).unwrap();
        let back = ChunkBlock::unpack(&buf).unwrap();
        assert_eq!(back.state_cookie, 0xdead_beef);
        assert_eq!(back.units_done, 8);
        assert_eq!(back.cmd(), Some(ChunkCmd::Continue));
        b.as_cancel();
        assert_eq!(b.cmd(), Some(ChunkCmd::Cancel));
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
