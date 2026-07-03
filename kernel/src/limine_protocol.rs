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

/// Request the list of boot modules Limine loaded (per `limine.conf`'s
/// `module_path` entries). Phase 3 uses this to reach `model.gguf`, loaded
/// as a module rather than compiled in (`CHITTI_OS_HANDOFF.md` Part 1).
#[repr(C)]
pub struct ModuleRequest {
    magic: [u64; 2],
    id: [u64; 2],
    revision: u64,
    response: UnsafeCell<*const ModuleResponse>,
}

// SAFETY: see `BaseRevision`'s impl above.
unsafe impl Sync for ModuleRequest {}

impl ModuleRequest {
    pub const fn new() -> Self {
        Self {
            magic: COMMON_MAGIC,
            id: [0x3e7e279702be32af, 0xca1c4f3bd1280cee],
            revision: 0,
            response: UnsafeCell::new(core::ptr::null()),
        }
    }

    pub fn response(&self) -> Option<&'static ModuleResponse> {
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
pub struct ModuleResponse {
    revision: u64,
    module_count: u64,
    modules: *const *const File,
}

impl ModuleResponse {
    pub fn modules(&self) -> &'static [&'static File] {
        // SAFETY: Limine provides `module_count` valid `File` pointers at
        // `modules` for the lifetime of the boot session.
        unsafe { core::slice::from_raw_parts(self.modules as *const &File, self.module_count as usize) }
    }
}

/// A loaded module. Only the leading fields Cortex needs are declared;
/// modules are referenced by pointer, so the omitted trailing fields
/// (media type, disk/partition UUIDs, ...) don't affect field offsets.
#[repr(C)]
pub struct File {
    pub revision: u64,
    /// Higher-half (HHDM) virtual address of the module's contents --
    /// directly usable as a pointer since we keep Limine's page tables.
    pub address: *const u8,
    pub size: u64,
    /// Null-terminated absolute path of the module.
    pub path: *const u8,
    pub cmdline: *const u8,
}

/// Request that Limine start the other CPUs (the SMP / multiprocessor
/// request). Phase 7 uses this to bring up application processors: Limine
/// discovers them via ACPI/MADT and parks each spinning on its `goto_address`,
/// which we then write to launch it. The trailing `flags` field can request
/// x2APIC; we leave it 0 (xAPIC), which is all the local-APIC code needs.
#[repr(C)]
pub struct SmpRequest {
    magic: [u64; 2],
    id: [u64; 2],
    revision: u64,
    response: UnsafeCell<*const SmpResponse>,
    flags: u64,
}

// SAFETY: see `BaseRevision`'s impl above -- Limine writes `response` once
// before handoff; we only read it afterwards.
unsafe impl Sync for SmpRequest {}

impl SmpRequest {
    pub const fn new() -> Self {
        Self {
            magic: COMMON_MAGIC,
            id: [0x95a67b819a1b857e, 0xa0b61b723b6a73e0],
            revision: 0,
            response: UnsafeCell::new(core::ptr::null()),
            flags: 0,
        }
    }

    pub fn response(&self) -> Option<&'static SmpResponse> {
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
pub struct SmpResponse {
    revision: u64,
    flags: u32,
    bsp_lapic_id: u32,
    cpu_count: u64,
    cpus: *const *const SmpInfo,
}

impl SmpResponse {
    /// The local-APIC id of the bootstrap processor (the core `_start` runs
    /// on). Used to skip the BSP when launching APs.
    pub fn bsp_lapic_id(&self) -> u32 {
        self.bsp_lapic_id
    }

    /// One `SmpInfo` per CPU (including the BSP).
    pub fn cpus(&self) -> &'static [&'static SmpInfo] {
        // SAFETY: Limine provides `cpu_count` valid `SmpInfo` pointers at
        // `cpus` for the lifetime of the boot session.
        unsafe { core::slice::from_raw_parts(self.cpus as *const &SmpInfo, self.cpu_count as usize) }
    }
}

/// Per-CPU control block Limine hands us. Writing a function pointer to
/// `goto_address` (atomically) launches that AP: it jumps there on a
/// Limine-provided stack with a pointer to its own `SmpInfo` in `rdi`.
/// `goto_address`/`extra_argument` are modelled as atomics because we write
/// them while the AP core is concurrently reading `goto_address`.
#[repr(C)]
pub struct SmpInfo {
    pub processor_id: u32,
    pub lapic_id: u32,
    reserved: u64,
    pub goto_address: core::sync::atomic::AtomicU64,
    pub extra_argument: core::sync::atomic::AtomicU64,
}

// SAFETY: `SmpInfo` lives in Limine-owned memory shared across cores; all our
// access is through atomics (`goto_address`/`extra_argument`) or read-only
// fields set by Limine before handoff.
unsafe impl Sync for SmpInfo {}

impl File {
    /// The module contents as a byte slice.
    pub fn data(&self) -> &'static [u8] {
        // SAFETY: Limine guarantees `address` points at `size` valid,
        // mapped bytes for the boot session.
        unsafe { core::slice::from_raw_parts(self.address, self.size as usize) }
    }

    /// The module's path as a string (e.g. `/boot/model.gguf.001`). Used to
    /// sort multi-part model modules into order.
    pub fn path_str(&self) -> &'static str {
        // SAFETY: `path` is a valid null-terminated C string from Limine.
        let mut len = 0usize;
        unsafe {
            while *self.path.add(len) != 0 {
                len += 1;
            }
            core::str::from_utf8(core::slice::from_raw_parts(self.path, len)).unwrap_or("")
        }
    }

    /// Whether the module's path contains `needle`.
    pub fn path_contains(&self, needle: &str) -> bool {
        self.path_str().contains(needle)
    }

    /// Whether the module's path ends with `suffix` (e.g. ".gguf").
    pub fn path_ends_with(&self, suffix: &str) -> bool {
        // SAFETY: `path` is a valid null-terminated C string from Limine;
        // we scan to the terminator (bounded) to recover its length.
        let mut len = 0usize;
        unsafe {
            while *self.path.add(len) != 0 {
                len += 1;
            }
            let bytes = core::slice::from_raw_parts(self.path, len);
            bytes.ends_with(suffix.as_bytes())
        }
    }
}
