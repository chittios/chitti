//! **Shared virtio plumbing** — one split-virtqueue implementation and one
//! transport trait, used by the devices that carry host integration.
//!
//! The tree already had seven virtio drivers before this, each with its own
//! copy of the ring arithmetic and its own bring-up sequence, split by arch:
//! the mmio ones under [`crate::arch::aarch64`], the PCI ones beside their
//! subsystem ([`crate::net`], [`crate::sound`], [`crate::kms`]). That was
//! workable while every such device existed on exactly one transport.
//!
//! It stops working for [`virtio_9p`](crate::drivers::virtio_9p) and
//! [`virtio_serial`](crate::drivers::virtio_serial), which must bind on **both**
//! — x86 has only PCI, the aarch64 `-kernel` dev loop has only mmio, and the
//! dual-architecture rule says a capability that exists on one arch exists on
//! the other. Writing each driver twice would be two chances to diverge, so the
//! transport is abstracted once, here, and the ring math is pulled out into a
//! module the test build can actually compile.
//!
//! Nothing existing was migrated onto this. Rewriting seven working, verified
//! drivers to prove a point would risk them all for no functional gain; the
//! shared layer earns its place on new devices, and an old one can move if it
//! is being touched anyway.

pub mod layout;
pub mod queue;
pub mod transport;

pub use queue::{Buf, Completion, Virtq};
pub use transport::{find_any, Transport, ID_9P, ID_CONSOLE, F_VERSION_1};

/// Order device-visible memory writes against the ring index that publishes
/// them.
///
/// On aarch64 this must be a real `dsb sy`: the queue lives in Normal cacheable
/// memory shared with a device that is not on the CPU's coherency domain from
/// the compiler's point of view, and every other virtio driver in the tree uses
/// exactly this barrier. On x86 the DMA region is coherent, so ordering the
/// compiler and the store buffer is enough.
#[inline]
pub fn barrier() {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: a barrier instruction; touches no memory.
    unsafe {
        core::arch::asm!("dsb sy", options(nomem, nostack, preserves_flags))
    };
    #[cfg(not(target_arch = "aarch64"))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
}
