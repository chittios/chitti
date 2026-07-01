//! Hand-rolled bindings for the subset of the Limine boot protocol Chitti
//! needs: the base revision handshake, the framebuffer request, the
//! memory-map request, and (from Phase 1) the higher-half direct map (HHDM)
//! request used to reach arbitrary physical frames (page tables, heap
//! backing memory) by virtual address.
//!
//! Struct layouts and magic numbers are taken from the Limine boot protocol
//! specification (stable since Limine 5.x; the base-revision mechanism
//! exists precisely so these wire values don't change across bootloader
//! versions). Hand-rolling this tiny subset — rather than pulling in a
//! third-party `limine` crate — keeps every unsafe field access here fully
//! auditable, per the project's determinism/guardrail rules.

use core::cell::UnsafeCell;

const COMMON_MAGIC: [u64; 2] = [0xc7b1dd30df4c8b88, 0x0a82e883a194f07b];

/// Declares which revision of the Limine protocol this kernel was written
/// against. Must be placed in the `.requests` section between the start and
/// end markers; Limine overwrites the third field with 0 if it accepted the
/// requested revision.
#[repr(C)]
pub struct BaseRevision {
    magic: UnsafeCell<[u64; 3]>,
}

// SAFETY: Limine only ever writes this struct once, before jumping to
// `_start`, and the kernel only reads it afterwards from a single core with
// interrupts disabled — there is no concurrent access to race against.
unsafe impl Sync for BaseRevision {}

impl BaseRevision {
    pub const fn new(revision: u64) -> Self {
        Self {
            magic: UnsafeCell::new([0xf9562b2d5c95a6c8, 0x6a7b384944536bdc, revision]),
        }
    }

    /// Whether Limine accepted the requested revision.
    pub fn is_supported(&self) -> bool {
        // SAFETY: `magic` is a 3-element array; Limine writes element 2
        // in place before handoff. `read_volatile` matches the bootloader's
        // own write so the compiler cannot fold this into a constant.
        unsafe { (self.magic.get() as *const u64).add(2).read_volatile() == 0 }
    }
}

/// Marks the start of the `.requests` section for Limine's scanner.
#[repr(C)]
pub struct RequestsStartMarker([u64; 4]);

impl RequestsStartMarker {
    pub const fn new() -> Self {
        Self([
            0xf6b8f4b39de7d1ae,
            0xfab91a6940fcb9cf,
            0x785c6ed015d3e316,
            0x181e920a7852b9d9,
        ])
    }
}

/// Marks the end of the `.requests` section for Limine's scanner.
#[repr(C)]
pub struct RequestsEndMarker([u64; 2]);

impl RequestsEndMarker {
    pub const fn new() -> Self {
        Self([0xadc0e0531bb10d03, 0x9572709f31764c62])
    }
}

/// Request that Limine hand back a framebuffer.
#[repr(C)]
pub struct FramebufferRequest {
    magic: [u64; 2],
    id: [u64; 2],
    revision: u64,
    response: UnsafeCell<*const FramebufferResponse>,
}

// SAFETY: see `BaseRevision`'s impl above — single write by the bootloader
// before handoff, single-threaded read afterwards.
unsafe impl Sync for FramebufferRequest {}

impl FramebufferRequest {
    pub const fn new() -> Self {
        Self {
            magic: COMMON_MAGIC,
            id: [0x9d5827dcd881dd75, 0xa3148604f6fab11b],
            revision: 0,
            response: UnsafeCell::new(core::ptr::null()),
        }
    }

    pub fn response(&self) -> Option<&'static FramebufferResponse> {
        // SAFETY: Limine either leaves this null (request refused) or
        // points it at a `'static` response it allocated before handoff.
        let ptr = unsafe { self.response.get().read_volatile() };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { &*ptr })
        }
    }
}

#[repr(C)]
pub struct FramebufferResponse {
    revision: u64,
    framebuffer_count: u64,
    framebuffers: *const *const Framebuffer,
}

impl FramebufferResponse {
    pub fn framebuffers(&self) -> &'static [&'static Framebuffer] {
        // SAFETY: Limine provides `framebuffer_count` valid pointers at
        // `framebuffers` for the lifetime of the boot session.
        unsafe {
            core::slice::from_raw_parts(
                self.framebuffers as *const &Framebuffer,
                self.framebuffer_count as usize,
            )
        }
    }
}

/// Revision-0 framebuffer descriptor (sufficient for Phase 0; revision-1
/// alternate video modes are not read here).
#[repr(C)]
pub struct Framebuffer {
    pub address: *mut u8,
    pub width: u64,
    pub height: u64,
    pub pitch: u64,
    pub bpp: u16,
    pub memory_model: u8,
    pub red_mask_size: u8,
    pub red_mask_shift: u8,
    pub green_mask_size: u8,
    pub green_mask_shift: u8,
    pub blue_mask_size: u8,
    pub blue_mask_shift: u8,
    _unused: [u8; 7],
    pub edid_size: u64,
    pub edid: *const u8,
}

/// Request the firmware/bootloader-provided memory map.
#[repr(C)]
pub struct MemmapRequest {
    magic: [u64; 2],
    id: [u64; 2],
    revision: u64,
    response: UnsafeCell<*const MemmapResponse>,
}

// SAFETY: see `BaseRevision`'s impl above.
unsafe impl Sync for MemmapRequest {}

impl MemmapRequest {
    pub const fn new() -> Self {
        Self {
            magic: COMMON_MAGIC,
            id: [0x67cf3d9d378a806f, 0xe304acdfc50c3c62],
            revision: 0,
            response: UnsafeCell::new(core::ptr::null()),
        }
    }

    pub fn response(&self) -> Option<&'static MemmapResponse> {
        // SAFETY: see `FramebufferRequest::response`.
        let ptr = unsafe { self.response.get().read_volatile() };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { &*ptr })
        }
    }
}

#[repr(C)]
pub struct MemmapResponse {
    revision: u64,
    entry_count: u64,
    entries: *const *const MemmapEntry,
}

impl MemmapResponse {
    pub fn entries(&self) -> &'static [&'static MemmapEntry] {
        // SAFETY: Limine provides `entry_count` valid pointers at `entries`
        // for the lifetime of the boot session.
        unsafe {
            core::slice::from_raw_parts(
                self.entries as *const &MemmapEntry,
                self.entry_count as usize,
            )
        }
    }
}

#[repr(C)]
pub struct MemmapEntry {
    pub base: u64,
    pub length: u64,
    pub entry_type: u64,
}

pub const MEMMAP_USABLE: u64 = 0;

/// Request the offset of the bootloader's higher-half direct map (HHDM):
/// physical address `p` is reachable at virtual address `p + offset`.
#[repr(C)]
pub struct HhdmRequest {
    magic: [u64; 2],
    id: [u64; 2],
    revision: u64,
    response: UnsafeCell<*const HhdmResponse>,
}

// SAFETY: see `BaseRevision`'s impl above.
unsafe impl Sync for HhdmRequest {}

impl HhdmRequest {
    pub const fn new() -> Self {
        Self {
            magic: COMMON_MAGIC,
            id: [0x48dcf1cb8ad2b852, 0x63984e959a98244b],
            revision: 0,
            response: UnsafeCell::new(core::ptr::null()),
        }
    }

    pub fn response(&self) -> Option<&'static HhdmResponse> {
        // SAFETY: see `FramebufferRequest::response`.
        let ptr = unsafe { self.response.get().read_volatile() };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { &*ptr })
        }
    }
}

#[repr(C)]
pub struct HhdmResponse {
    revision: u64,
    pub offset: u64,
}

/// Request that the bootloader hand back the executable's load addresses,
/// so we can compute an inclusive physical range for the kernel image
/// (used to sanity-check the frame allocator excludes it).
#[repr(C)]
pub struct ExecutableAddressRequest {
    magic: [u64; 2],
    id: [u64; 2],
    revision: u64,
    response: UnsafeCell<*const ExecutableAddressResponse>,
}

// SAFETY: see `BaseRevision`'s impl above.
unsafe impl Sync for ExecutableAddressRequest {}

impl ExecutableAddressRequest {
    pub const fn new() -> Self {
        Self {
            magic: COMMON_MAGIC,
            id: [0x71ba76863cc55f63, 0xb2644a48c516a487],
            revision: 0,
            response: UnsafeCell::new(core::ptr::null()),
        }
    }

    pub fn response(&self) -> Option<&'static ExecutableAddressResponse> {
        // SAFETY: see `FramebufferRequest::response`.
        let ptr = unsafe { self.response.get().read_volatile() };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { &*ptr })
        }
    }
}

#[repr(C)]
pub struct ExecutableAddressResponse {
    revision: u64,
    pub physical_base: u64,
    pub virtual_base: u64,
}
