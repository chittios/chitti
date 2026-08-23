//! Kernel driver for a **chunked stateful tenant**.
//!
//! [`super::tenant::ImageTenant`] is the one-shot anti-pattern: whole file, bump
//! cursor reset every entry, kernel waits. H.264 / AAC / JS cannot finish in one
//! crossing — Ctrl+C and [`crate::shell::upkeep`] are standing rules, and a
//! tenant has no device access. This loader:
//!
//! 1. Writes a [`crate::synapse::chunked::ChunkCmd`] before each entry.
//! 2. Leaves the guest arena and `.bss` **alone** across [`ChunkCmd::Continue`].
//! 3. Pumps `upkeep` / `poll_interrupt` **between** entries.
//! 4. Bounds-checks every VA the tenant reports against the heap frames it owns.
//! 5. Has **no in-kernel fallback**.

use super::chunked::{self, ChunkBlock, ChunkCmd, ChunkStatus};
use super::tenant::{load_image, Blob, LoadError};
use alloc::string::{String, ToString};
use crate::arch::enter_tenant;
use crate::mm::space::{self, AddressSpace, UserPerms};
use alloc::vec::Vec;

const PAGE: usize = 0x1000;
const INPUT_WINDOW: u64 = 4 << 30;
const DEFAULT_HEAP: usize = 8 << 20;
const MAX_HEAP: usize = 512 << 20;
const KEEP_HEAP: usize = 32 << 20;

/// Reusable chunked tenant: one address space, one parked cap-less task.
pub struct ChunkedTenant {
    space: Option<AddressSpace>,
    entry: u64,
    stack: u64,
    task: crate::sched::TaskId,
    args_va: u64,
    args_kernel: u64,
    in_va: u64,
    in_frames: Vec<u64>,
    in_len: usize,
    heap_va: u64,
    heap_frames: Vec<u64>,
    want: usize,
}

impl ChunkedTenant {
    pub fn new(blob: Blob, heap_bytes: usize) -> Result<Self, LoadError> {
        let mut loaded = load_image(blob)?;
        let args_va = loaded.stack_page + PAGE as u64;
        let args_phys = loaded
            .space
            .map_new_page(args_va, UserPerms::RW)
            .map_err(|_| LoadError::Map)?;
        let args_kernel = space::phys_to_kernel(args_phys);
        let in_va = args_va + PAGE as u64;
        let heap_va = args_va + INPUT_WINDOW;
        let mut me = Self {
            space: Some(loaded.space),
            entry: loaded.entry,
            stack: loaded.stack,
            task: crate::sched::spawn_parked("chunked"),
            args_va,
            args_kernel,
            in_va,
            in_frames: Vec::new(),
            in_len: 0,
            heap_va,
            heap_frames: Vec::new(),
            want: 0,
        };
        if !me.grow_heap(heap_bytes) {
            return Err(LoadError::NoAddressSpace);
        }
        Ok(me)
    }

    pub fn heap_bytes(&self) -> usize {
        self.heap_frames.len() * PAGE
    }

    pub fn wanted_heap(&self) -> usize {
        self.want
    }

    fn extend(
        space: &mut AddressSpace,
        frames: &mut Vec<u64>,
        va: u64,
        want: usize,
        perms: UserPerms,
    ) -> bool {
        while frames.len() < want {
            let Some(f) = space::take_shared_frame() else {
                return false;
            };
            let at = va + (frames.len() as u64) * PAGE as u64;
            if space.map_frame(at, f, perms).is_err() {
                space::give_frame(f);
                return false;
            }
            frames.push(f);
        }
        true
    }

    pub fn grow_heap(&mut self, bytes: usize) -> bool {
        let want = bytes.div_ceil(PAGE);
        let heap_va = self.heap_va;
        let Some(space) = self.space.as_mut() else {
            return false;
        };
        Self::extend(space, &mut self.heap_frames, heap_va, want, UserPerms::RW)
    }

    /// Copy `bytes` into the input window (RO shared frames). Zeroes the last-page tail.
    pub fn copy_input(&mut self, bytes: &[u8]) -> Result<(), LoadError> {
        let want = bytes.len().div_ceil(PAGE).max(1);
        {
            let Some(space) = self.space.as_mut() else {
                return Err(LoadError::NoAddressSpace);
            };
            if !Self::extend(space, &mut self.in_frames, self.in_va, want, UserPerms::RO) {
                return Err(LoadError::NoAddressSpace);
            }
        }
        for (i, &f) in self.in_frames.iter().take(want).enumerate() {
            let off = i * PAGE;
            let n = bytes.len().saturating_sub(off).min(PAGE);
            // SAFETY: kernel-owned shared frame; `n` bounded by page and remaining input.
            unsafe {
                let dst = space::phys_to_kernel(f) as *mut u8;
                if n > 0 {
                    core::ptr::copy_nonoverlapping(bytes.as_ptr().add(off), dst, n);
                }
                if n < PAGE {
                    core::ptr::write_bytes(dst.add(n), 0, PAGE - n);
                }
            }
        }
        self.in_len = bytes.len();
        Ok(())
    }

    /// Copy `len` bytes from a guest heap VA into `dst`. Every number is a claim.
    pub fn copy_out(&self, guest_ptr: u64, len: usize, dst: &mut [u8]) -> Result<(), LoadError> {
        if len > dst.len() {
            return Err(LoadError::Faulted);
        }
        let off = guest_ptr
            .checked_sub(self.heap_va)
            .map(|o| o as usize)
            .unwrap_or(usize::MAX);
        if off.saturating_add(len) > self.heap_bytes() {
            crate::ktrace::log_fmt(format_args!(
                "tenant: chunked tenant reported an impossible output: ptr={guest_ptr:#x} len={len}"
            ));
            return Err(LoadError::Faulted);
        }
        let mut done = 0usize;
        while done < len {
            let at = off + done;
            let frame = self.heap_frames[at / PAGE];
            let within = at % PAGE;
            let n = (PAGE - within).min(len - done);
            // SAFETY: range proven to lie inside heap frames this loader owns.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    (space::phys_to_kernel(frame) as *const u8).add(within),
                    dst.as_mut_ptr().add(done),
                    n,
                );
            }
            done += n;
        }
        Ok(())
    }

    /// One entry. Does **not** reset the guest heap. The guest resets itself on Init.
    pub fn enter(&mut self, cmd: ChunkCmd, max_units: u64) -> Result<ChunkBlock, LoadError> {
        self.enter_meta(cmd, max_units, 0, 0, 0, 0)
    }

    pub fn enter_meta(
        &mut self,
        cmd: ChunkCmd,
        max_units: u64,
        meta0: u64,
        meta1: u64,
        meta2: u64,
        meta3: u64,
    ) -> Result<ChunkBlock, LoadError> {
        let mut blk = ChunkBlock::new_init(self.in_va, self.in_len as u64, max_units);
        blk.cmd_or_status = cmd as u64;
        blk.meta0 = meta0;
        blk.meta1 = meta1;
        blk.meta2 = meta2;
        blk.meta3 = meta3;
        blk.heap_ptr = self.heap_va;
        blk.heap_len = self.heap_bytes() as u64;
        // SAFETY: this tenant's block page, mapped RW and kernel-aliased.
        unsafe {
            core::ptr::write_bytes(self.args_kernel as *mut u8, 0, PAGE);
            let mut packed = [0u8; ChunkBlock::BYTES];
            blk.pack(&mut packed).ok_or(LoadError::Faulted)?;
            core::ptr::copy_nonoverlapping(
                packed.as_ptr(),
                self.args_kernel as *mut u8,
                ChunkBlock::BYTES,
            );
        }
        let Some(space) = self.space.as_ref() else {
            return Err(LoadError::NoAddressSpace);
        };
        // SAFETY: mappings are this space's; no Synapse call so no justification.
        let exit = unsafe { enter_tenant(self.task, space, self.entry, self.stack, self.args_va) };
        if !exit.is_deliberate_exit() {
            crate::ktrace::log_fmt(format_args!(
                "tenant: chunked tenant did not exit cleanly: {exit:?}"
            ));
            #[cfg(target_arch = "aarch64")]
            if let crate::arch::TenantExit::Faulted { syndrome, address } = exit {
                crate::ktrace::log_fmt(format_args!(
                    "tenant: {} / {} FAR={address:#x} (user+{:#x})",
                    crate::mm::armv8::esr_class(syndrome),
                    crate::mm::armv8::esr_fault_status(syndrome),
                    address.saturating_sub(crate::mm::space::USER_BASE)
                ));
            }
            return Err(LoadError::Faulted);
        }
        let mut packed = [0u8; ChunkBlock::BYTES];
        // SAFETY: tenant wrote the block page, or left it as we filled it.
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.args_kernel as *const u8,
                packed.as_mut_ptr(),
                ChunkBlock::BYTES,
            );
        }
        let out = ChunkBlock::unpack(&packed).ok_or(LoadError::Faulted)?;
        self.want = (out.heap_want as usize).min(MAX_HEAP);
        Ok(out)
    }
}

impl Drop for ChunkedTenant {
    fn drop(&mut self) {
        drop(self.space.take());
        for &f in self.in_frames.iter().chain(self.heap_frames.iter()) {
            space::give_frame(f);
        }
        let _ = crate::sched::kill(self.task);
    }
}

/// Drive `blob` over `input` until Done/Failed/Cancelled. `on_chunk` is called
/// with the block after every successful entry that produced output (and after
/// Done). Returns the last block.
pub fn run_to_idle(
    tenant: &mut ChunkedTenant,
    input: &[u8],
    max_units: u64,
    mut on_chunk: impl FnMut(&ChunkedTenant, &ChunkBlock) -> Result<(), LoadError>,
) -> Result<ChunkBlock, LoadError> {
    tenant.copy_input(input)?;
    let mut cmd = ChunkCmd::Init;
    loop {
        let blk = tenant.enter(cmd, max_units)?;
        let Some(st) = blk.status() else {
            return Err(LoadError::Declined { status: blk.cmd_or_status });
        };
        match st {
            ChunkStatus::OutOfMemory => {
                let want = tenant.wanted_heap().max(tenant.heap_bytes() * 2).min(MAX_HEAP);
                let need = (want.saturating_sub(tenant.heap_bytes())) / PAGE;
                if want <= tenant.heap_bytes() || space::free_frames() < (need as u64) * 2 {
                    return Err(LoadError::Declined {
                        status: ChunkStatus::OutOfMemory as u64,
                    });
                }
                if !tenant.grow_heap(want) {
                    return Err(LoadError::NoAddressSpace);
                }
                // Re-enter the same command; the guest must tolerate Init retry after OOM.
            }
            ChunkStatus::NeedMore => {
                on_chunk(tenant, &blk)?;
                crate::shell::upkeep();
                if crate::shell::poll_interrupt() {
                    let _ = tenant.enter(ChunkCmd::Cancel, 0);
                    return Err(LoadError::Declined {
                        status: ChunkStatus::Cancelled as u64,
                    });
                }
                cmd = ChunkCmd::Continue;
            }
            ChunkStatus::Done => {
                on_chunk(tenant, &blk)?;
                return Ok(blk);
            }
            ChunkStatus::Failed | ChunkStatus::Cancelled => {
                return Err(LoadError::Declined {
                    status: blk.cmd_or_status,
                });
            }
        }
    }
}

/// Slot + stats for one kind of chunked tenant (audio, video, js).
pub struct Slot {
    inner: crate::mm::Locked<Option<ChunkedTenant>>,
    blob: fn() -> Blob,
}

impl Slot {
    pub const fn new(blob: fn() -> Blob) -> Self {
        Self {
            inner: crate::mm::Locked::new(None),
            blob,
        }
    }

    pub fn take_or_new(&self) -> Result<ChunkedTenant, LoadError> {
        match self.inner.with(|s| s.take()) {
            Some(t) => Ok(t),
            None => ChunkedTenant::new((self.blob)(), DEFAULT_HEAP),
        }
    }

    pub fn put(&self, tenant: ChunkedTenant) {
        if tenant.heap_bytes() <= KEEP_HEAP {
            self.inner.with(|s| *s = Some(tenant));
        }
    }
}

// --- media (audio PCM + video YUV) tenant ------------------------------------

#[cfg(target_arch = "x86_64")]
static MEDIADEC_BLOB: &[u8] = include_bytes!("../../../userspace/mediadec/mediadec-x86_64.bin");
#[cfg(target_arch = "aarch64")]
static MEDIADEC_BLOB: &[u8] = include_bytes!("../../../userspace/mediadec/mediadec-aarch64.bin");
#[cfg(target_arch = "x86_64")]
const MEDIADEC_LAYOUT: (u64, u64, u64) = include!("../../../userspace/mediadec/entry-x86_64.in");
#[cfg(target_arch = "aarch64")]
const MEDIADEC_LAYOUT: (u64, u64, u64) = include!("../../../userspace/mediadec/entry-aarch64.in");

fn mediadec_image() -> Blob {
    let (entry, rx, rw) = MEDIADEC_LAYOUT;
    Blob { bytes: MEDIADEC_BLOB, entry, rx, rw }
}

static MEDIADEC: Slot = Slot::new(mediadec_image);

/// Off in the unit suite so existing codec fixtures keep exercising the
/// in-kernel engines; on at boot so `/open` uses the tenant. Tests that want
/// the sandbox call [`set_sandboxed_media`].
static SANDBOXED_MEDIA: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(!cfg!(test));

pub fn sandboxed_media() -> bool {
    SANDBOXED_MEDIA.load(core::sync::atomic::Ordering::Relaxed)
}

pub fn set_sandboxed_media(on: bool) {
    SANDBOXED_MEDIA.store(on, core::sync::atomic::Ordering::Relaxed);
}

/// Decode WAV/MP3/AAC in ring 3. PCM is gathered across Continue entries.
pub fn decode_audio(bytes: &[u8]) -> Result<crate::audio::Audio, LoadError> {
    let mut tenant = MEDIADEC.take_or_new()?;
    let mut pcm: Vec<i16> = Vec::new();
    let mut rate = 0u32;
    let mut channels = 1u8;
    let result = run_to_idle(&mut tenant, bytes, 32, |t, blk| {
        if blk.kind != chunked::KIND_AUDIO && blk.output_len == 0 {
            return Ok(());
        }
        if blk.meta0 != 0 {
            rate = blk.meta0 as u32;
        }
        if blk.meta1 != 0 {
            channels = blk.meta1 as u8;
        }
        if blk.output_len > 0 {
            let n = blk.output_len as usize;
            if n % 2 != 0 {
                return Err(LoadError::Faulted);
            }
            let mut raw = alloc::vec![0u8; n];
            t.copy_out(blk.output_ptr, n, &mut raw)?;
            let nsamp = n / 2;
            let start = pcm.len();
            pcm.resize(start + nsamp, 0);
            for i in 0..nsamp {
                pcm[start + i] = i16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
            }
        }
        Ok(())
    });
    MEDIADEC.put(tenant);
    result?;
    if pcm.is_empty() || rate == 0 {
        return Err(LoadError::Declined {
            status: ChunkStatus::Failed as u64,
        });
    }
    Ok(crate::audio::Audio { rate, pcm, channels: channels.max(1) })
}

/// One 4:2:0 8-bit YUV frame from the media tenant.
pub struct YuvFrame {
    pub w: usize,
    pub h: usize,
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
}

impl YuvFrame {
    pub fn to_rgb(&self, pts_ms: u64) -> crate::video::Frame {
        crate::video::rgb_from_i420(&self.y, &self.cb, &self.cr, self.w, self.h, pts_ms)
    }
}

/// Sequential YUV decode of a container in ring 3. One display frame per Continue.
pub struct VideoPipe {
    tenant: ChunkedTenant,
    opened: bool,
    pub w: u32,
    pub h: u32,
    pub frame_count: usize,
    pub duration_ms: u64,
    pub pts: Vec<u64>,
    /// Last display index handed to the kernel. Sequential `idx == delivered+1`
    /// uses Continue (no rewind); jumps use Seek.
    delivered: Option<usize>,
}

impl VideoPipe {
    pub fn open(bytes: &[u8]) -> Result<Self, LoadError> {
        let mut tenant = ChunkedTenant::new(mediadec_image(), DEFAULT_HEAP)?;
        tenant.copy_input(bytes)?;
        let blk = Self::init_with_oom(&mut tenant)?;
        if blk.kind != chunked::KIND_VIDEO {
            return Err(LoadError::Declined { status: blk.cmd_or_status });
        }
        let w = blk.meta0 as u32;
        let h = blk.meta1 as u32;
        let frame_count = blk.meta2 as usize;
        let mut duration_ms = blk.meta3;
        let mut pts = alloc::vec![0u64; frame_count];
        if blk.output_len > 0 && frame_count > 0 {
            let n = (frame_count * 8).min(blk.output_len as usize);
            let mut raw = alloc::vec![0u8; n];
            tenant.copy_out(blk.output_ptr, n, &mut raw)?;
            for i in 0..n / 8 {
                pts[i] = u64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().unwrap_or([0; 8]));
            }
        }
        if duration_ms == 0 {
            duration_ms = pts.last().copied().unwrap_or(0);
        }
        if duration_ms == 0 && frame_count > 1 {
            duration_ms = (frame_count as u64).saturating_mul(33);
        }
        if frame_count > 1 && pts.iter().all(|&p| p == 0) && duration_ms > 0 {
            for i in 0..frame_count {
                pts[i] = duration_ms * i as u64 / (frame_count as u64);
            }
        }
        Ok(Self {
            tenant,
            opened: true,
            w,
            h,
            frame_count,
            duration_ms,
            pts,
            delivered: None,
        })
    }

    fn init_with_oom(tenant: &mut ChunkedTenant) -> Result<ChunkBlock, LoadError> {
        let blk = tenant.enter(ChunkCmd::Init, 1)?;
        let st = blk.status().ok_or(LoadError::Declined { status: blk.cmd_or_status })?;
        if matches!(st, ChunkStatus::Failed | ChunkStatus::Cancelled) {
            return Err(LoadError::Declined { status: blk.cmd_or_status });
        }
        if st != ChunkStatus::OutOfMemory {
            return Ok(blk);
        }
        let want = tenant.wanted_heap().max(tenant.heap_bytes() * 2).min(MAX_HEAP);
        if !tenant.grow_heap(want) {
            return Err(LoadError::NoAddressSpace);
        }
        let blk = tenant.enter(ChunkCmd::Init, 1)?;
        let st = blk.status().ok_or(LoadError::Declined { status: blk.cmd_or_status })?;
        if matches!(st, ChunkStatus::Failed | ChunkStatus::Cancelled) {
            return Err(LoadError::Declined { status: blk.cmd_or_status });
        }
        Ok(blk)
    }

    pub fn pts_ms(&self, idx: usize) -> u64 {
        self.pts.get(idx).copied().unwrap_or(0)
    }

    /// Display frame `idx`. Sequential `delivered+1` uses Continue (one picture,
    /// no rewind). Jumps and the first frame use Seek. Never calls `upkeep`:
    /// the player invokes this from the pump, and `upkeep` → `pump_video` is a
    /// `Locked` deadlock (interrupts off, machine stopped).
    pub fn decode_display(&mut self, idx: usize) -> Result<Option<YuvFrame>, LoadError> {
        if !self.opened {
            return Ok(None);
        }
        if idx >= self.frame_count {
            return Ok(None);
        }
        match self.delivered {
            // After Init the tenant's display cursor is 0. Continue is the
            // sequential command; Seek(0) used `max_units = 0` as the index
            // and, on rust_h264 High-profile failure, returned no planes.
            None if idx == 0 => self.continue_one(0),
            Some(d) if idx == d + 1 => self.continue_one(idx),
            Some(d) if idx > d + 1 => {
                // Pump hurry is capped at +2. Continue each picture rather than
                // Seek: rust_h264 looks one sample ahead, and a Seek that
                // compared against that cursor rewound every frame, re-decoded
                // the prefix with IRQs masked, and froze `/open flower.mp4`.
                let mut last = None;
                for i in (d + 1)..=idx {
                    match self.continue_one(i)? {
                        Some(f) => last = Some(f),
                        None => return Ok(last),
                    }
                }
                Ok(last)
            }
            _ => self.seek_yuv(idx),
        }
    }

    pub fn seek_yuv(&mut self, idx: usize) -> Result<Option<YuvFrame>, LoadError> {
        if !self.opened {
            return Ok(None);
        }
        match self.enter_frame(ChunkCmd::Seek, idx as u64)? {
            Some(frame) => {
                self.delivered = Some(idx);
                Ok(Some(frame))
            }
            None => Ok(None),
        }
    }

    pub fn next_yuv(&mut self) -> Result<Option<YuvFrame>, LoadError> {
        let idx = match self.delivered {
            Some(d) => d + 1,
            None => 0,
        };
        if idx >= self.frame_count {
            self.opened = false;
            return Ok(None);
        }
        self.continue_one(idx)
    }

    fn continue_one(&mut self, idx: usize) -> Result<Option<YuvFrame>, LoadError> {
        match self.enter_frame(ChunkCmd::Continue, 1)? {
            Some(frame) => {
                self.delivered = Some(idx);
                Ok(Some(frame))
            }
            None => Ok(None),
        }
    }

    fn enter_frame(&mut self, cmd: ChunkCmd, max_units: u64) -> Result<Option<YuvFrame>, LoadError> {
        loop {
            let blk = self.tenant.enter(cmd, max_units)?;
            let Some(st) = blk.status() else {
                return Err(LoadError::Declined { status: blk.cmd_or_status });
            };
            match st {
                ChunkStatus::OutOfMemory => {
                    let want = self.tenant.wanted_heap().max(self.tenant.heap_bytes() * 2).min(MAX_HEAP);
                    if !self.tenant.grow_heap(want) {
                        return Err(LoadError::NoAddressSpace);
                    }
                }
                ChunkStatus::Failed | ChunkStatus::Cancelled => {
                    return Err(LoadError::Declined { status: blk.cmd_or_status });
                }
                ChunkStatus::NeedMore | ChunkStatus::Done => {
                    if blk.meta4 == 0 {
                        if st == ChunkStatus::Done {
                            self.opened = false;
                            return Ok(None);
                        }
                        return Err(LoadError::Declined { status: blk.cmd_or_status });
                    }
                    let frame = self.read_yuv(&blk)?;
                    if st == ChunkStatus::Done {
                        self.opened = false;
                    }
                    return Ok(Some(frame));
                }
            }
        }
    }

    fn read_yuv(&self, blk: &ChunkBlock) -> Result<YuvFrame, LoadError> {
        let (w, h) = (blk.meta0 as usize, blk.meta1 as usize);
        let mut y = alloc::vec![0u8; blk.meta4 as usize];
        let mut cb = alloc::vec![0u8; blk.meta5 as usize];
        let mut cr = alloc::vec![0u8; blk.meta6 as usize];
        self.tenant.copy_out(blk.output_ptr, y.len(), &mut y)?;
        self.tenant.copy_out(blk.meta2, cb.len(), &mut cb)?;
        self.tenant.copy_out(blk.meta3, cr.len(), &mut cr)?;
        Ok(YuvFrame { w, h, y, cb, cr })
    }
}

// --- JS interpreter tenant ---------------------------------------------------

#[cfg(target_arch = "x86_64")]
static JSINT_BLOB: &[u8] = include_bytes!("../../../userspace/jsint/jsint-x86_64.bin");
#[cfg(target_arch = "aarch64")]
static JSINT_BLOB: &[u8] = include_bytes!("../../../userspace/jsint/jsint-aarch64.bin");
#[cfg(target_arch = "x86_64")]
const JSINT_LAYOUT: (u64, u64, u64) = include!("../../../userspace/jsint/entry-x86_64.in");
#[cfg(target_arch = "aarch64")]
const JSINT_LAYOUT: (u64, u64, u64) = include!("../../../userspace/jsint/entry-aarch64.in");

fn jsint_image() -> Blob {
    let (entry, rx, rw) = JSINT_LAYOUT;
    Blob { bytes: JSINT_BLOB, entry, rx, rw }
}

static JSINT: Slot = Slot::new(jsint_image);

static SANDBOXED_JS: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(!cfg!(test));

pub fn sandboxed_js() -> bool {
    SANDBOXED_JS.load(core::sync::atomic::Ordering::Relaxed)
}

pub fn set_sandboxed_js(on: bool) {
    SANDBOXED_JS.store(on, core::sync::atomic::Ordering::Relaxed);
}

/// Length-prefixed envelope the jsint tenant accepts: `u32le n` then n × (`u32le len` + bytes).
pub fn pack_scripts(scripts: &[alloc::string::String]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(scripts.len() as u32).to_le_bytes());
    for s in scripts {
        let b = s.as_bytes();
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(b);
    }
    out
}

/// Run DOM-free page scripts in ring 3. Returns the tenant's UTF-8 log (title=/value=/console).
pub fn eval_scripts(scripts: &[alloc::string::String]) -> Result<String, LoadError> {
    let packed = pack_scripts(scripts);
    let mut tenant = JSINT.take_or_new()?;
    let mut log = String::new();
    let result = run_to_idle(&mut tenant, &packed, 1, |t, blk| {
        if blk.output_len > 0 {
            let n = blk.output_len as usize;
            let mut buf = alloc::vec![0u8; n];
            t.copy_out(blk.output_ptr, n, &mut buf)?;
            if let Ok(s) = core::str::from_utf8(&buf) {
                log.push_str(s);
            }
        }
        Ok(())
    });
    JSINT.put(tenant);
    result?;
    Ok(log)
}

static JS_PAGE: crate::mm::Locked<Option<ChunkedTenant>> = crate::mm::Locked::new(None);

/// Pack a live `JsDom` + page scripts as the jsint `JDOM` envelope.
pub fn pack_js_page(dom: &crate::browser::js::JsDom, scripts: &[String]) -> Vec<u8> {
    use super::chunked::{put_str, put_u16, put_u32, JS_DOM_MAGIC};
    let mut out = Vec::new();
    put_u32(&mut out, JS_DOM_MAGIC);
    put_u32(&mut out, dom.inner_width as u32);
    put_u32(&mut out, dom.inner_height as u32);
    put_str(&mut out, &dom.location_href);
    put_u32(&mut out, dom.elements.len() as u32);
    for e in &dom.elements {
        put_str(&mut out, &e.tag);
        put_str(&mut out, e.id.as_deref().unwrap_or(""));
        put_str(&mut out, e.class.as_deref().unwrap_or(""));
        put_str(&mut out, &e.text);
        put_str(&mut out, &e.value);
        put_str(&mut out, &e.style);
        put_u32(
            &mut out,
            e.parent.map(|p| p as u32).unwrap_or(u32::MAX),
        );
        let mut flags = 0u8;
        if e.hidden {
            flags |= 1;
        }
        out.push(flags);
        put_u16(&mut out, e.children.len() as u16);
        for &c in &e.children {
            put_u32(&mut out, c as u32);
        }
        put_u16(&mut out, e.attrs.len() as u16);
        for (k, v) in &e.attrs {
            put_str(&mut out, k);
            put_str(&mut out, v);
        }
    }
    put_u32(&mut out, scripts.len() as u32);
    for s in scripts {
        put_str(&mut out, s);
    }
    out
}

/// Apply a `JMUT` log onto `dom`. Returns the interactive-element index list.
pub fn apply_js_mutations(dom: &mut crate::browser::js::JsDom, bytes: &[u8]) -> Vec<usize> {
    use super::chunked::{
        get_bytes, get_u32, JS_MUT_MAGIC, JS_OP_HIDDEN, JS_OP_INTERACTIVE, JS_OP_LOG, JS_OP_NAV,
        JS_OP_SCROLL, JS_OP_SNAPSHOT, JS_OP_STYLE, JS_OP_TEXT, JS_OP_TITLE, JS_OP_VALUE,
    };
    let mut interactive = Vec::new();
    let mut off = 0usize;
    let Some(mag) = get_u32(bytes, &mut off) else { return interactive };
    if mag != JS_MUT_MAGIC {
        // DOM-free UTF-8 log (title=/value=/console).
        if let Ok(s) = core::str::from_utf8(bytes) {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("title=") {
                    dom.title = rest.to_string();
                } else if line.starts_with("value=") || line.is_empty() {
                    continue;
                } else {
                    dom.log.push(line.to_string());
                }
            }
        }
        return interactive;
    }
    let Some(n) = get_u32(bytes, &mut off) else { return interactive };
    for _ in 0..n {
        let Some(op) = bytes.get(off).copied() else { break };
        off += 1;
        let Some(idx) = get_u32(bytes, &mut off) else { break };
        let Some(payload) = get_bytes(bytes, &mut off) else { break };
        let text = core::str::from_utf8(payload).unwrap_or("");
        match op {
            JS_OP_TITLE => dom.title = text.to_string(),
            JS_OP_NAV => dom.navigate = Some(text.to_string()),
            JS_OP_TEXT => {
                if let Some(e) = dom.elements.get_mut(idx as usize) {
                    e.text = text.to_string();
                }
            }
            JS_OP_VALUE => {
                if let Some(e) = dom.elements.get_mut(idx as usize) {
                    e.value = text.to_string();
                }
            }
            JS_OP_STYLE => {
                if let Some(e) = dom.elements.get_mut(idx as usize) {
                    e.style = text.to_string();
                }
            }
            JS_OP_HIDDEN => {
                if let Some(e) = dom.elements.get_mut(idx as usize) {
                    e.hidden = payload.first().copied().unwrap_or(0) != 0;
                }
            }
            JS_OP_LOG => {
                for line in text.lines() {
                    if !line.is_empty() {
                        dom.log.push(line.to_string());
                    }
                }
            }
            JS_OP_SCROLL => {
                if payload.len() >= 4 {
                    let v = i32::from_le_bytes(payload[0..4].try_into().unwrap_or([0; 4]));
                    dom.scroll_to = Some(v);
                }
            }
            JS_OP_INTERACTIVE => {
                let mut o = 0usize;
                if let Some(c) = get_u32(payload, &mut o) {
                    for _ in 0..c {
                        if let Some(i) = get_u32(payload, &mut o) {
                            interactive.push(i as usize);
                        }
                    }
                }
            }
            JS_OP_SNAPSHOT => {
                if let Some(elems) = unpack_elem_snapshot(payload) {
                    dom.elements = elems;
                }
            }
            _ => {}
        }
    }
    interactive
}

fn unpack_elem_snapshot(payload: &[u8]) -> Option<Vec<crate::browser::js::ElemRef>> {
    use super::chunked::{get_str, get_u16, get_u32};
    use crate::browser::js::empty_elem;
    let mut off = 0usize;
    let n = get_u32(payload, &mut off)? as usize;
    let mut elems = Vec::with_capacity(n);
    for _ in 0..n {
        let tag = get_str(payload, &mut off)?.to_string();
        let id = get_str(payload, &mut off)?.to_string();
        let class = get_str(payload, &mut off)?.to_string();
        let text = get_str(payload, &mut off)?.to_string();
        let value = get_str(payload, &mut off)?.to_string();
        let style = get_str(payload, &mut off)?.to_string();
        let parent = get_u32(payload, &mut off)?;
        let flags = *payload.get(off)?;
        off += 1;
        let nc = get_u16(payload, &mut off)? as usize;
        let mut children = Vec::with_capacity(nc);
        for _ in 0..nc {
            children.push(get_u32(payload, &mut off)? as usize);
        }
        let na = get_u16(payload, &mut off)? as usize;
        let mut e = empty_elem(&tag);
        e.id = if id.is_empty() { None } else { Some(id) };
        e.class = if class.is_empty() { None } else { Some(class) };
        e.text = text;
        e.value = value;
        e.style = style;
        e.parent = if parent == u32::MAX { None } else { Some(parent as usize) };
        e.hidden = flags & 1 != 0;
        e.children = children;
        for _ in 0..na {
            let k = get_str(payload, &mut off)?.to_string();
            let v = get_str(payload, &mut off)?.to_string();
            match k.as_str() {
                "id" => e.id = Some(v.clone()),
                "class" => e.class = Some(v.clone()),
                "href" => e.href = v.clone(),
                "src" => e.src = v.clone(),
                "type" => e.type_attr = v.clone(),
                "name" => e.name_attr = v.clone(),
                _ => {}
            }
            e.attrs.insert(k, v);
        }
        elems.push(e);
    }
    Some(elems)
}

fn copy_out_vec(t: &ChunkedTenant, blk: &ChunkBlock) -> Result<Vec<u8>, LoadError> {
    if blk.output_len == 0 {
        return Ok(Vec::new());
    }
    let n = blk.output_len as usize;
    let mut buf = alloc::vec![0u8; n];
    t.copy_out(blk.output_ptr, n, &mut buf)?;
    Ok(buf)
}

/// One-shot DOM script run (does not keep the tenant for events).
fn js_heap_for(packed_len: usize) -> usize {
    packed_len
        .saturating_mul(32)
        .saturating_add(16 << 20)
        .clamp(DEFAULT_HEAP, MAX_HEAP)
}

pub fn js_eval_dom(packed: &[u8], max_scripts: u64) -> Result<Vec<u8>, LoadError> {
    let mut tenant = ChunkedTenant::new(jsint_image(), js_heap_for(packed.len()))?;
    tenant.copy_input(packed)?;
    let mut last = Vec::new();
    let mut cmd = ChunkCmd::Init;
    loop {
        crate::shell::upkeep();
        if crate::shell::poll_interrupt() {
            let _ = tenant.enter(ChunkCmd::Cancel, 0);
            return Err(LoadError::Declined {
                status: ChunkStatus::Cancelled as u64,
            });
        }
        let blk = tenant.enter(cmd, max_scripts.max(1))?;
        let st = blk.status().ok_or(LoadError::Declined { status: blk.cmd_or_status })?;
        match st {
            ChunkStatus::OutOfMemory => {
                let want = tenant.wanted_heap().max(tenant.heap_bytes() * 2).min(MAX_HEAP);
                if !tenant.grow_heap(want) {
                    return Err(LoadError::NoAddressSpace);
                }
            }
            ChunkStatus::Failed | ChunkStatus::Cancelled => {
                return Err(LoadError::Declined { status: blk.cmd_or_status });
            }
            ChunkStatus::NeedMore => {
                last = copy_out_vec(&tenant, &blk)?;
                cmd = ChunkCmd::Continue;
            }
            ChunkStatus::Done => {
                last = copy_out_vec(&tenant, &blk)?;
                let _ = tenant.enter(ChunkCmd::Cancel, 0);
                return Ok(last);
            }
        }
    }
}

/// Boot page JS in ring 3. Keeps the tenant for later [`js_page_event`].
pub fn js_page_open(packed: &[u8], max_scripts: u64) -> Result<Vec<u8>, LoadError> {
    js_page_close();
    let mut tenant = ChunkedTenant::new(jsint_image(), js_heap_for(packed.len()))?;
    tenant.copy_input(packed)?;
    let mut last = Vec::new();
    let mut cmd = ChunkCmd::Init;
    loop {
        crate::shell::upkeep();
        if crate::shell::poll_interrupt() {
            let _ = tenant.enter(ChunkCmd::Cancel, 0);
            return Err(LoadError::Declined {
                status: ChunkStatus::Cancelled as u64,
            });
        }
        let blk = tenant.enter(cmd, max_scripts.max(1))?;
        let st = blk.status().ok_or(LoadError::Declined { status: blk.cmd_or_status })?;
        match st {
            ChunkStatus::OutOfMemory => {
                let want = tenant.wanted_heap().max(tenant.heap_bytes() * 2).min(MAX_HEAP);
                if !tenant.grow_heap(want) {
                    return Err(LoadError::NoAddressSpace);
                }
            }
            ChunkStatus::Failed | ChunkStatus::Cancelled => {
                return Err(LoadError::Declined { status: blk.cmd_or_status });
            }
            ChunkStatus::NeedMore => {
                last = copy_out_vec(&tenant, &blk)?;
                cmd = ChunkCmd::Continue;
            }
            ChunkStatus::Done => {
                last = copy_out_vec(&tenant, &blk)?;
                JS_PAGE.with(|s| *s = Some(tenant));
                return Ok(last);
            }
        }
    }
}

pub fn js_page_event(ty: u64, target: usize, x: i32, y: i32) -> Result<Vec<u8>, LoadError> {
    let mut tenant = JS_PAGE.with(|s| s.take()).ok_or(LoadError::Declined {
        status: ChunkStatus::Failed as u64,
    })?;
    let blk = tenant.enter_meta(ChunkCmd::Event, 1, ty, target as u64, x as u32 as u64, y as u32 as u64)?;
    let out = match blk.status() {
        Some(ChunkStatus::NeedMore | ChunkStatus::Done) => copy_out_vec(&tenant, &blk),
        Some(st) => Err(LoadError::Declined { status: st as u64 }),
        None => Err(LoadError::Declined { status: blk.cmd_or_status }),
    };
    JS_PAGE.with(|s| *s = Some(tenant));
    out
}

pub fn js_page_close() {
    if let Some(mut t) = JS_PAGE.with(|s| s.take()) {
        let _ = t.enter(ChunkCmd::Cancel, 0);
    }
}

pub fn js_page_live() -> bool {
    JS_PAGE.with(|s| s.is_some())
}

#[cfg(test)]
mod tests {
    use super::super::chunked::{ChunkBlock, ChunkCmd, ChunkStatus};
    use super::*;

    #[test_case]
    fn chunked_block_is_smaller_than_the_startup_page() {
        assert!(ChunkBlock::BYTES <= 0x1000);
        assert!(ChunkCmd::from_u64(1) == Some(ChunkCmd::Init));
        assert!(ChunkStatus::from_u64(5) == Some(ChunkStatus::OutOfMemory));
    }

    #[test_case]
    fn pack_scripts_roundtrips_the_envelope_header() {
        let scripts = alloc::vec![alloc::string::String::from("1+2")];
        let p = pack_scripts(&scripts);
        assert_eq!(&p[0..4], &1u32.to_le_bytes());
        assert_eq!(&p[4..8], &3u32.to_le_bytes());
        assert_eq!(&p[8..], b"1+2");
    }

    #[test_case]
    fn mediadec_tenant_rejects_empty_input() {
        set_sandboxed_media(true);
        let err = decode_audio(b"not-a-media-file");
        assert!(err.is_err(), "garbage must not decode as audio");
        assert!(VideoPipe::open(b"not-a-container").is_err(), "garbage video must not open");
        set_sandboxed_media(false);
    }

    fn tiny_wav_s16_mono() -> Vec<u8> {
        let pcm: [i16; 8] = [0, 1000, 2000, 1000, 0, -1000, -2000, -1000];
        let data_bytes = pcm.len() * 2;
        let mut v = Vec::new();
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36u32 + data_bytes as u32).to_le_bytes());
        v.extend_from_slice(b"WAVE");
        v.extend_from_slice(b"fmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&8000u32.to_le_bytes());
        v.extend_from_slice(&16000u32.to_le_bytes());
        v.extend_from_slice(&2u16.to_le_bytes());
        v.extend_from_slice(&16u16.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&(data_bytes as u32).to_le_bytes());
        for s in pcm {
            v.extend_from_slice(&s.to_le_bytes());
        }
        v
    }

    fn adts_frame_44100_stereo(payload: &[u8]) -> Vec<u8> {
        let frame_len = 7 + payload.len();
        let mut h = [0u8; 7];
        h[0] = 0xff;
        h[1] = 0xf1;
        h[2] = 0x50;
        h[3] = 0x80 | (((frame_len >> 11) as u8) & 0x03);
        h[4] = (frame_len >> 3) as u8;
        h[5] = ((frame_len & 0x07) << 5) as u8;
        let mut out = Vec::from(h.as_slice());
        out.extend_from_slice(payload);
        out
    }

    /// Real AUs from `testdata_echo_head.bin`, wrapped as ADTS so Init takes the
    /// AAC path (`Decoder::new` / `Dsp::new`) rather than the empty-input reject.
    fn echo_adts() -> Vec<u8> {
        static FIX: &[u8] = include_bytes!("../audio/aac/testdata_echo_head.bin");
        let asc_len = u32::from_le_bytes(FIX[0..4].try_into().unwrap()) as usize;
        let n = u32::from_le_bytes(FIX[4 + asc_len..8 + asc_len].try_into().unwrap()) as usize;
        let mut off = 8 + asc_len;
        let mut out = Vec::new();
        for _ in 0..n {
            if off + 4 > FIX.len() {
                break;
            }
            let sz = u32::from_le_bytes(FIX[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            if off + sz > FIX.len() {
                break;
            }
            out.extend_from_slice(&adts_frame_44100_stereo(&FIX[off..off + sz]));
            off += sz;
        }
        out
    }

    #[test_case]
    fn mediadec_tenant_decodes_a_real_wav() {
        set_sandboxed_media(true);
        let a = decode_audio(&tiny_wav_s16_mono()).expect("wav tenant must not fault");
        assert_eq!(a.rate, 8000);
        assert_eq!(a.channels, 1);
        assert_eq!(a.pcm.len(), 8);
        set_sandboxed_media(false);
    }

    #[test_case]
    fn mediadec_tenant_decodes_real_adts_without_faulting() {
        set_sandboxed_media(true);
        let adts = echo_adts();
        assert!(adts.len() > 7, "fixture produced no ADTS frames");
        match decode_audio(&adts) {
            Ok(a) => {
                assert!(a.rate > 0, "decoded ADTS must report a sample rate");
                assert!(!a.pcm.is_empty(), "decoded ADTS must produce PCM");
            }
            Err(LoadError::Faulted) => {
                panic!("AAC Init must not overflow the tenant stack (data abort in .rodata)")
            }
            Err(_) => panic!("echo ADTS must decode in the mediadec tenant"),
        }
        set_sandboxed_media(false);
    }

    #[test_case]
    fn mediadec_tenant_decodes_an_h264_clip_to_yuv() {
        set_sandboxed_media(true);
        let clip = crate::video::sandbox_test_clip();
        let mut pipe = VideoPipe::open(clip).expect("h264 tenant must open");
        assert!(pipe.w > 0 && pipe.h > 0, "Init must report geometry");
        assert!(pipe.frame_count > 0, "Init must report a frame count");
        let yuv = pipe
            .decode_display(0)
            .expect("first frame must not fault")
            .expect("first frame must exist");
        assert_eq!(yuv.w, pipe.w as usize);
        assert_eq!(yuv.h, pipe.h as usize);
        assert!(!yuv.y.is_empty());
        let rgb = yuv.to_rgb(0);
        assert!(rgb.pixels.len() > 0);
        // Sequential Continue (not Seek): the playback pump takes this path.
        // Seeking every frame used to rewind rust_h264's one-sample lookahead
        // and freeze `/open flower.mp4` with IRQs masked.
        for i in 1..pipe.frame_count {
            let yuv_i = pipe
                .decode_display(i)
                .unwrap_or_else(|_| panic!("display {i} must not fault"))
                .unwrap_or_else(|| panic!("display {i} must exist"));
            assert_eq!(yuv_i.w, pipe.w as usize);
        }
        set_sandboxed_media(false);
    }

    #[test_case]
    fn jsint_tenant_evals_a_dom_free_script() {
        let scripts = alloc::vec![alloc::string::String::from("1 + 2")];
        let log = eval_scripts(&scripts).expect("jsint tenant must run");
        assert!(
            log.contains("3") || log.contains("value="),
            "expected the completion in the tenant log, got {log:?}"
        );
    }

    #[test_case]
    fn jsint_tenant_runs_document_title_and_inner_text() {
        set_sandboxed_js(true);
        let mut dom = crate::browser::js::JsDom::from_document(&crate::browser::html::parse(
            "<html><body><div id=\"t\">x</div></body></html>",
        ));
        let scripts = alloc::vec![alloc::string::String::from(
            "document.title = 'hi'; document.getElementById('t').innerText = 'y';"
        )];
        let packed = pack_js_page(&dom, &scripts);
        let muts = js_eval_dom(&packed, 1).expect("dom tenant must not fault");
        let _ = apply_js_mutations(&mut dom, &muts);
        assert_eq!(dom.title, "hi", "title must come back from the tenant");
        let text = dom
            .elements
            .iter()
            .find(|e| e.id.as_deref() == Some("t"))
            .map(|e| e.text.as_str())
            .unwrap_or("");
        assert_eq!(text, "y", "innerText write must round-trip");
        set_sandboxed_js(false);
    }

    #[test_case]
    fn jsint_create_element_replaces_loading_placeholder() {
        // React createRoot does `root.textContent = ""` then appendChild. Without
        // those APIs (and a tree snapshot back to the kernel) shadcn.html stays
        // on "loading shadcn/ui bundle…".
        set_sandboxed_js(true);
        let mut dom = crate::browser::js::JsDom::from_document(&crate::browser::html::parse(
            r#"<html><body><div id="root"><p>loading shadcn/ui bundle…</p></div></body></html>"#,
        ));
        let scripts = alloc::vec![alloc::string::String::from(
            r#"var root = document.getElementById("root");
               root.textContent = "";
               var h1 = document.createElement("h1");
               h1.textContent = "shadcn/ui gallery";
               root.appendChild(h1);"#
        )];
        let packed = pack_js_page(&dom, &scripts);
        let muts = js_eval_dom(&packed, 1).expect("createElement tenant must not fault");
        let _ = apply_js_mutations(&mut dom, &muts);
        let root = dom
            .elements
            .iter()
            .find(|e| e.id.as_deref() == Some("root"))
            .expect("root survives");
        assert_eq!(root.children.len(), 1, "placeholder must be detached, h1 appended");
        let kid = &dom.elements[root.children[0]];
        assert_eq!(kid.tag, "h1");
        assert_eq!(kid.text, "shadcn/ui gallery");
        set_sandboxed_js(false);
    }

    #[test_case]
    fn jsint_settimeout_is_a_function_and_runs_now() {
        // React's scheduler: `F = typeof setTimeout == "function" ? setTimeout : null`.
        // Without typeof "function" it stores null and later throws
        // `null is not a function` — which is how `/browse shadcn.html` died
        // after the stack-overflow fault was fixed.
        set_sandboxed_js(true);
        let mut dom = crate::browser::js::JsDom::from_document(&crate::browser::html::parse(
            "<html><body></body></html>",
        ));
        let scripts = alloc::vec![alloc::string::String::from(
            "console.log(typeof setTimeout);\
             var n=0; var F=typeof setTimeout=='function'?setTimeout:null;\
             F(function(){ n=1; }, 0);\
             console.log('n='+n);"
        )];
        let packed = pack_js_page(&dom, &scripts);
        let muts = js_eval_dom(&packed, 1).expect("setTimeout tenant must not fault");
        let _ = apply_js_mutations(&mut dom, &muts);
        assert!(
            dom.log.iter().any(|l| l == "function"),
            "typeof setTimeout: {:?}",
            dom.log
        );
        assert!(
            dom.log.iter().any(|l| l == "n=1"),
            "F(cb) must run: {:?}",
            dom.log
        );
        set_sandboxed_js(false);
    }
}
