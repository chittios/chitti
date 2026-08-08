//! **virtio-9p** — the device that carries a host shared folder.
//!
//! The guest half of `-virtfs local,path=…,mount_tag=…` (QEMU) — a single
//! request virtqueue over which whole 9P2000.L messages are exchanged, and a
//! device config holding the **mount tag** the host chose. All the protocol
//! lives in [`crate::fs::ninep`]; this file is the wire underneath it.
//!
//! One request is in flight at a time. 9P allows many (that is what tags are
//! for), but the callers here are synchronous filesystem operations on a
//! cooperative scheduler — there is no second thread to issue the second
//! request, so a pipeline would add a completion-matching path that nothing
//! could exercise.
//!
//! **The session is taken out of the lock, not held across I/O.** A whole-file
//! read is many round trips, and `Locked::with` runs with interrupts disabled;
//! holding it for the duration would stop the timer, the mouse and the net
//! stack for as long as the copy took. So an operation removes the session,
//! runs, and puts it back — and a re-entrant call finds it absent and reports
//! the device busy rather than deadlocking on a non-reentrant lock.

use crate::drivers::virtio::transport::{cfg_read, cfg_read16};
use crate::drivers::virtio::{find_any, Buf, Transport, Virtq, F_VERSION_1, ID_9P};
use crate::fs::ninep::client::{Rpc, Session, MSIZE_WANT};
use crate::fs::ninep::wire::P9Error;
use crate::mm::Locked;
use alloc::boxed::Box;
use alloc::string::String;

/// `VIRTIO_9P_MOUNT_TAG` — the device publishes a mount tag in its config.
const F_MOUNT_TAG: u64 = 1 << 0;

/// The request queue. virtio-9p has exactly one.
const Q_REQUEST: u16 = 0;

/// Device config: `tag_len[2]` then the tag bytes.
const CFG_TAG_LEN: usize = 0;
const CFG_TAG: usize = 2;

/// A tag longer than this is refused rather than truncated — a truncated tag
/// would silently name a different export.
const MAX_TAG: usize = 256;

/// How long to wait for the host to answer one message before giving up.
///
/// Bounded, per the standing rule: a 9P round trip is host-backed and takes
/// microseconds, so anything approaching this bound means the device is wedged
/// and spinning forever would leave no way out but killing the VM.
const SPIN_LIMIT: u64 = 2_000_000_000;

/// The device: a transport, its request queue and the two DMA buffers every
/// message travels through.
pub struct Virtio9pDev {
    t: Box<dyn Transport>,
    q: Virtq,
    tx_phys: u64,
    tx_virt: u64,
    rx_phys: u64,
    rx_virt: u64,
    /// Capacity of each DMA buffer.
    cap: usize,
}

impl Rpc for Virtio9pDev {
    fn rpc(&mut self, req: &[u8], reply: &mut [u8]) -> Result<usize, P9Error> {
        if req.len() > self.cap {
            return Err(P9Error::TooLarge);
        }
        // SAFETY: `tx_virt` is a `cap`-byte DMA region owned by this device and
        // `req` is at most `cap` bytes.
        unsafe { core::ptr::copy_nonoverlapping(req.as_ptr(), self.tx_virt as *mut u8, req.len()) };

        let out = [Buf { phys: self.tx_phys, len: req.len() as u32 }];
        let inb = [Buf { phys: self.rx_phys, len: self.cap as u32 }];
        if self.q.add(&out, &inb).is_none() {
            // Only ever one request in flight, so a full queue means the
            // previous one was never completed.
            return Err(P9Error::Transport);
        }
        self.q.kick(&*self.t, Q_REQUEST);

        let mut spins = 0u64;
        let done = loop {
            if let Some(c) = self.q.take_used() {
                break c;
            }
            spins += 1;
            if spins > SPIN_LIMIT {
                crate::ktrace::log("virtio-9p", "host did not answer; giving up on this request");
                return Err(P9Error::Transport);
            }
            // Answer Ctrl+C. Only `poll_interrupt` (a non-blocking console-ring
            // read that pushes back anything that is not Ctrl+C), never
            // `upkeep()`: this runs from inside filesystem calls that are
            // themselves reachable from the UI pump, and pumping here is the
            // re-entrancy hang the block drivers document.
            if spins % 0x10_0000 == 0 && crate::shell::poll_interrupt() {
                crate::ktrace::log("virtio-9p", "request cancelled by Ctrl+C");
                return Err(P9Error::Transport);
            }
            core::hint::spin_loop();
        };
        crate::drivers::virtio::barrier();

        // `len` is what the device wrote across the chain's writable buffers —
        // here exactly the reply. Clamp to both buffers: a device reporting
        // more than it could have written must not make us read past either.
        let n = (done.len as usize).min(self.cap).min(reply.len());
        // SAFETY: `rx_virt` is a `cap`-byte DMA region and `n <= cap`, `n <=
        // reply.len()`.
        unsafe { core::ptr::copy_nonoverlapping(self.rx_virt as *const u8, reply.as_mut_ptr(), n) };
        Ok(n)
    }
}

/// The live session, plus the mount tag the host published.
struct Mounted {
    session: Session<Virtio9pDev>,
    tag: String,
}

static DEV: Locked<Option<Mounted>> = Locked::new(None);

/// Whether a host shared folder is attached.
pub fn present() -> bool {
    DEV.with(|d| d.is_some())
}

/// The host's mount tag (`-virtfs …,mount_tag=X`), if attached.
pub fn tag() -> Option<String> {
    DEV.with(|d| d.as_ref().map(|m| m.tag.clone()))
}

/// Run `f` against the live session.
///
/// The session is **taken** for the duration so the lock is not held across
/// I/O. A re-entrant call therefore sees `None` — reported as a busy device,
/// which is the honest answer and not a deadlock.
pub fn with_session<R>(f: impl FnOnce(&mut Session<Virtio9pDev>) -> R) -> Option<R> {
    let mut m = DEV.with(|d| d.take())?;
    let r = f(&mut m.session);
    DEV.with(|d| *d = Some(m));
    Some(r)
}

/// Probe for a virtio-9p device, negotiate 9P2000.L and attach to the export.
///
/// Returns the mount tag on success. Absent hardware is not an error — most
/// boots have no shared folder — so this is quiet unless a device is found.
pub fn init() -> Option<String> {
    let mut t = find_any(ID_9P, 0, &[])?;
    t.begin();

    let offered = t.device_features();
    // The mount tag is the only feature this driver needs, and a device without
    // it cannot tell us what it exported.
    if offered & F_MOUNT_TAG == 0 {
        crate::ktrace::log("virtio-9p", "device publishes no mount tag; not claiming it");
        return None;
    }
    // Accept VERSION_1 only where it is offered: the legacy transport does not
    // have it and writes only the low feature word.
    let want = F_MOUNT_TAG | (offered & F_VERSION_1);
    if !t.accept_features(want) {
        crate::ktrace::log("virtio-9p", "device rejected our feature set");
        return None;
    }

    let tag_len = cfg_read16(&*t, CFG_TAG_LEN) as usize;
    if tag_len == 0 || tag_len > MAX_TAG {
        crate::ktrace::log_fmt(format_args!("virtio-9p: implausible mount tag length {tag_len}"));
        return None;
    }
    let mut raw = alloc::vec![0u8; tag_len];
    cfg_read(&*t, CFG_TAG, &mut raw);
    let tag = String::from_utf8_lossy(&raw).into_owned();

    // The queue must be at least two descriptors deep: every message is a
    // request buffer plus a reply buffer, and a one-deep queue could not carry
    // even one exchange.
    let q = Virtq::setup(&mut *t, Q_REQUEST, 8)?;
    if q.available() < 2 {
        crate::ktrace::log("virtio-9p", "request queue too shallow for a request/reply pair");
        return None;
    }
    let cap = MSIZE_WANT as usize;
    let (tx_phys, tx_virt) = crate::mm::alloc_dma(cap)?;
    let (rx_phys, rx_virt) = crate::mm::alloc_dma(cap)?;
    t.ready();

    let dev = Virtio9pDev { t, q, tx_phys, tx_virt, rx_phys, rx_virt, cap };
    // `aname` is empty: QEMU's 9p export is the tree itself, and the tag names
    // which export rather than a subtree within it.
    let session = match Session::attach(dev, "chitti", "") {
        Ok(s) => s,
        Err(e) => {
            crate::ktrace::log_fmt(format_args!(
                "virtio-9p: attach to '{tag}' failed: {}",
                crate::fs::ninep::describe(e)
            ));
            return None;
        }
    };
    crate::ktrace::log_fmt(format_args!(
        "virtio-9p: host folder '{tag}' attached (msize {})",
        session.msize()
    ));
    DEV.with(|d| *d = Some(Mounted { session, tag: tag.clone() }));
    Some(tag)
}
