//! **virtio-serial (multiport virtio-console)** — the byte pipe the clipboard
//! agent rides on.
//!
//! QEMU exposes `-device virtserialport,name=com.redhat.spice.0` on this
//! device, and the clipboard channel is whatever is written to and read from
//! that named port ([`crate::clipboard::vdagent`] is the protocol).
//!
//! Three things about this device shape the driver:
//!
//! * **Queue numbering is not sequential per port.** Port 0 owns queues 0 and
//!   1, the *control* pair is 2 and 3, and port `i >= 1` owns `2i+2` and
//!   `2i+3`. So port 1 is queues 4 and 5 — not 2 and 3, which is the obvious
//!   guess and lands on the control channel, where a clipboard message reads
//!   as a malformed control packet.
//! * **A port is not usable until a four-step handshake completes**, and each
//!   step is a control packet in one direction. Writing to a port before its
//!   `PORT_OPEN` is silently dropped by the host — nothing errors, the
//!   clipboard simply never arrives.
//! * **The port is identified by name, not by number.** QEMU picks the port
//!   number, so the driver waits for the `PORT_NAME` control packet naming
//!   `com.redhat.spice.0` rather than assuming port 1.
//!
//! Queues must be configured before `DRIVER_OK`, and the port number is only
//! learned afterwards, so queues are set up for the first few ports up front
//! and the unused ones simply stay idle.

use crate::drivers::virtio::transport::cfg_read;
use crate::drivers::virtio::{find_any, Buf, Transport, Virtq, F_VERSION_1, ID_CONSOLE};
use crate::mm::Locked;
use alloc::boxed::Box;
use alloc::vec::Vec;

/// `VIRTIO_CONSOLE_F_MULTIPORT` — without it there is no control channel and
/// no named ports, so there is nothing here to drive.
const F_MULTIPORT: u64 = 1 << 1;

// Control queue pair.
const Q_CTRL_RX: u16 = 2;
const Q_CTRL_TX: u16 = 3;

/// Ports this driver prepares queues for. Port 0 plus two more covers every
/// layout QEMU produces for a single `virtserialport`, within the eight queues
/// the shared transport tracks notify addresses for.
const MAX_PORTS: u32 = 3;

/// Receive queue for port `i`: 0 for port 0, else `2i + 2`.
fn rx_queue(port: u32) -> u16 {
    if port == 0 {
        0
    } else {
        (2 * port + 2) as u16
    }
}
/// Transmit queue for port `i`: 1 for port 0, else `2i + 3`.
fn tx_queue(port: u32) -> u16 {
    if port == 0 {
        1
    } else {
        (2 * port + 3) as u16
    }
}

// --- virtio_console_control events ---
const EV_DEVICE_READY: u16 = 0;
const EV_PORT_ADD: u16 = 1;
const EV_PORT_REMOVE: u16 = 2;
const EV_PORT_READY: u16 = 3;
const EV_CONSOLE_PORT: u16 = 4;
const EV_RESIZE: u16 = 5;
const EV_PORT_OPEN: u16 = 6;
const EV_PORT_NAME: u16 = 7;

/// `struct virtio_console_control { le32 id; le16 event; le16 value; }`.
const CTRL_LEN: usize = 8;

/// Size of each parked receive buffer. The vdagent chunk cap is 1024 bytes plus
/// its header, so this holds any single chunk with room to spare.
const RX_BUF: usize = 2048;
/// How many receive buffers to park per queue.
const RX_SLOTS: usize = 8;
/// Transmit staging buffer.
const TX_BUF: usize = 4096;

/// The port name the SPICE clipboard agent lives on.
pub const SPICE_PORT_NAME: &str = "com.redhat.spice.0";

/// A DMA buffer pool parked on a receive queue.
struct RxPool {
    phys: [u64; RX_SLOTS],
    virt: [u64; RX_SLOTS],
    /// Descriptor head → slot, so a completion knows which buffer it filled.
    head_of: [u16; RX_SLOTS],
}

impl RxPool {
    fn new(q: &mut Virtq) -> Option<RxPool> {
        let mut p = RxPool { phys: [0; RX_SLOTS], virt: [0; RX_SLOTS], head_of: [0; RX_SLOTS] };
        for i in 0..RX_SLOTS {
            let (phys, virt) = crate::mm::alloc_dma(RX_BUF)?;
            p.phys[i] = phys;
            p.virt[i] = virt;
            let head = q.add(&[], &[Buf { phys, len: RX_BUF as u32 }])?;
            p.head_of[i] = head;
        }
        Some(p)
    }

    /// Slot index for a completed descriptor head.
    fn slot_of(&self, head: u16) -> Option<usize> {
        self.head_of.iter().position(|h| *h == head)
    }

    /// Re-park slot `i`'s buffer on the queue.
    fn repost(&mut self, q: &mut Virtq, i: usize) {
        if let Some(head) = q.add(&[], &[Buf { phys: self.phys[i], len: RX_BUF as u32 }]) {
            self.head_of[i] = head;
        }
    }
}

struct Serial {
    t: Box<dyn Transport>,
    ctrl_rx: Virtq,
    ctrl_tx: Virtq,
    ctrl_pool: RxPool,
    ctrl_tx_phys: u64,
    ctrl_tx_virt: u64,
    /// Per-port data queues, indexed by port number.
    ports: Vec<Option<PortQueues>>,
    /// The port that named itself [`SPICE_PORT_NAME`], once known.
    spice: Option<u32>,
    /// Whether that port has completed its handshake and may be written to.
    spice_open: bool,
}

struct PortQueues {
    rx: Virtq,
    tx: Virtq,
    pool: RxPool,
    tx_phys: u64,
    tx_virt: u64,
}

static DEV: Locked<Option<Serial>> = Locked::new(None);

/// Whether a SPICE agent port is present and open.
pub fn spice_ready() -> bool {
    DEV.with(|d| d.as_ref().map(|s| s.spice_open).unwrap_or(false))
}

/// Whether the device exists at all (even if no agent port opened).
pub fn present() -> bool {
    DEV.with(|d| d.is_some())
}

impl Serial {
    /// Send one control packet.
    fn ctrl_send(&mut self, id: u32, event: u16, value: u16) {
        // SAFETY: `ctrl_tx_virt` is a TX_BUF-byte DMA region owned here.
        unsafe {
            let p = self.ctrl_tx_virt as *mut u8;
            core::ptr::copy_nonoverlapping(id.to_le_bytes().as_ptr(), p, 4);
            core::ptr::copy_nonoverlapping(event.to_le_bytes().as_ptr(), p.add(4), 2);
            core::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), p.add(6), 2);
        }
        if self
            .ctrl_tx
            .add(&[Buf { phys: self.ctrl_tx_phys, len: CTRL_LEN as u32 }], &[])
            .is_some()
        {
            self.ctrl_tx.kick(&*self.t, Q_CTRL_TX);
            // Reclaim synchronously: the control channel is low rate, and a
            // bounded wait here keeps the descriptor pool from draining. A host
            // that never completes it costs one descriptor, not a hang.
            let mut spins = 0u32;
            while self.ctrl_tx.take_used().is_none() && spins < 1_000_000 {
                core::hint::spin_loop();
                spins += 1;
            }
        }
    }

    /// Drain the control queue, advancing the port handshake.
    fn pump_ctrl(&mut self) {
        while let Some(c) = self.ctrl_rx.take_used() {
            let Some(slot) = self.ctrl_pool.slot_of(c.head) else {
                continue;
            };
            let len = (c.len as usize).min(RX_BUF);
            // SAFETY: the device wrote `len` bytes into this slot's buffer.
            let bytes = unsafe {
                core::slice::from_raw_parts(self.ctrl_pool.virt[slot] as *const u8, len)
            };
            if len >= CTRL_LEN {
                let id = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                let event = u16::from_le_bytes([bytes[4], bytes[5]]);
                let value = u16::from_le_bytes([bytes[6], bytes[7]]);
                // A PORT_NAME carries its name after the fixed fields, and it
                // is NOT NUL-terminated by contract — QEMU appends one, but
                // trusting that would truncate or overrun on a host that does
                // not.
                let name = core::str::from_utf8(&bytes[CTRL_LEN..])
                    .unwrap_or("")
                    .trim_end_matches('\0');
                self.on_ctrl(id, event, value, name);
            }
            self.ctrl_pool.repost(&mut self.ctrl_rx, slot);
            self.ctrl_rx.kick(&*self.t, Q_CTRL_RX);
        }
    }

    fn on_ctrl(&mut self, id: u32, event: u16, value: u16, name: &str) {
        match event {
            EV_PORT_ADD => {
                // Acknowledge, which is what makes the host send PORT_NAME and
                // PORT_OPEN for it.
                if (id as usize) < self.ports.len() && self.ports[id as usize].is_some() {
                    self.ctrl_send(id, EV_PORT_READY, 1);
                } else {
                    // A port beyond the queues we prepared: say so rather than
                    // acknowledging one we cannot service, which would leave
                    // the host expecting reads that never come.
                    crate::ktrace::log_fmt(format_args!(
                        "virtio-serial: port {id} has no queues (prepared {MAX_PORTS}); ignoring"
                    ));
                }
            }
            EV_PORT_NAME => {
                if name == SPICE_PORT_NAME {
                    crate::ktrace::log_fmt(format_args!(
                        "virtio-serial: SPICE agent port is port {id}"
                    ));
                    self.spice = Some(id);
                }
            }
            EV_PORT_OPEN => {
                // The host opened its end. Open ours, and only then is the
                // port writable — a write before this is dropped in silence.
                if Some(id) == self.spice {
                    self.ctrl_send(id, EV_PORT_OPEN, 1);
                    self.spice_open = value != 0;
                    crate::ktrace::log_fmt(format_args!(
                        "virtio-serial: SPICE port {id} {}",
                        if self.spice_open { "open" } else { "closed by host" }
                    ));
                }
            }
            EV_PORT_REMOVE => {
                if Some(id) == self.spice {
                    self.spice = None;
                    self.spice_open = false;
                }
            }
            EV_CONSOLE_PORT | EV_RESIZE | EV_DEVICE_READY | EV_PORT_READY => {}
            other => crate::ktrace::log_fmt(format_args!(
                "virtio-serial: unhandled control event {other} on port {id}"
            )),
        }
    }

    /// Drain received bytes from the SPICE port.
    fn read_spice(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        let Some(port) = self.spice else {
            return out;
        };
        let Some(Some(pq)) = self.ports.get_mut(port as usize) else {
            return out;
        };
        while let Some(c) = pq.rx.take_used() {
            let Some(slot) = pq.pool.slot_of(c.head) else {
                continue;
            };
            let len = (c.len as usize).min(RX_BUF);
            // SAFETY: the device wrote `len` bytes into this slot's buffer.
            let bytes =
                unsafe { core::slice::from_raw_parts(pq.pool.virt[slot] as *const u8, len) };
            out.extend_from_slice(bytes);
            pq.pool.repost(&mut pq.rx, slot);
        }
        if !out.is_empty() {
            pq.rx.kick(&*self.t, rx_queue(port));
        }
        out
    }

    /// Write bytes to the SPICE port.
    fn write_spice(&mut self, data: &[u8]) -> bool {
        if !self.spice_open {
            return false;
        }
        let Some(port) = self.spice else {
            return false;
        };
        let txq = tx_queue(port);
        let Some(Some(pq)) = self.ports.get_mut(port as usize) else {
            return false;
        };
        let mut off = 0;
        while off < data.len() {
            let take = (data.len() - off).min(TX_BUF);
            // SAFETY: `tx_virt` is a TX_BUF-byte DMA region and `take <= TX_BUF`.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    data[off..].as_ptr(),
                    pq.tx_virt as *mut u8,
                    take,
                )
            };
            if pq.tx.add(&[Buf { phys: pq.tx_phys, len: take as u32 }], &[]).is_none() {
                return false;
            }
            pq.tx.kick(&*self.t, txq);
            // The staging buffer is reused for the next slice, so this send
            // must complete before it is overwritten.
            let mut spins = 0u32;
            while pq.tx.take_used().is_none() && spins < 50_000_000 {
                core::hint::spin_loop();
                spins += 1;
            }
            if spins >= 50_000_000 {
                crate::ktrace::log("virtio-serial", "port write never completed; giving up");
                return false;
            }
            off += take;
        }
        true
    }
}

/// Probe for a multiport virtio-serial device and start the port handshake.
///
/// Returns whether a device was claimed. Absent hardware is the common case and
/// is quiet.
pub fn init() -> bool {
    let Some(mut t) = find_any(ID_CONSOLE, 0, &[]) else {
        return false;
    };
    t.begin();
    let offered = t.device_features();
    if offered & F_MULTIPORT == 0 {
        // A plain console has no control channel and no named ports, so there
        // is no agent port to find. Say so rather than half-initialising it.
        crate::ktrace::log("virtio-serial", "device is not multiport; no agent port possible");
        return false;
    }
    if !t.accept_features(F_MULTIPORT | (offered & F_VERSION_1)) {
        crate::ktrace::log("virtio-serial", "device rejected our feature set");
        return false;
    }

    // config: cols[2] rows[2] max_nr_ports[4]
    let mut cfg = [0u8; 8];
    cfg_read(&*t, 0, &mut cfg);
    let max_ports = u32::from_le_bytes([cfg[4], cfg[5], cfg[6], cfg[7]]);
    let prepare = max_ports.clamp(1, MAX_PORTS);

    // Every queue must be configured before DRIVER_OK, and the agent's port
    // number is only learned after it — hence preparing several.
    let mut ports: Vec<Option<PortQueues>> = Vec::new();
    for p in 0..prepare {
        let made = (|| {
            let mut rx = Virtq::setup(&mut *t, rx_queue(p), 16)?;
            let tx = Virtq::setup(&mut *t, tx_queue(p), 16)?;
            let pool = RxPool::new(&mut rx)?;
            let (tx_phys, tx_virt) = crate::mm::alloc_dma(TX_BUF)?;
            Some(PortQueues { rx, tx, pool, tx_phys, tx_virt })
        })();
        ports.push(made);
    }
    let Some(mut ctrl_rx) = Virtq::setup(&mut *t, Q_CTRL_RX, 16) else {
        return false;
    };
    let Some(ctrl_tx) = Virtq::setup(&mut *t, Q_CTRL_TX, 16) else {
        return false;
    };
    let Some(ctrl_pool) = RxPool::new(&mut ctrl_rx) else {
        return false;
    };
    let Some((ctrl_tx_phys, ctrl_tx_virt)) = crate::mm::alloc_dma(TX_BUF) else {
        return false;
    };
    t.ready();

    let mut s = Serial {
        t,
        ctrl_rx,
        ctrl_tx,
        ctrl_pool,
        ctrl_tx_phys,
        ctrl_tx_virt,
        ports,
        spice: None,
        spice_open: false,
    };
    // Park the receive buffers, then announce readiness — which is what makes
    // the host start describing its ports.
    s.ctrl_rx.kick(&*s.t, Q_CTRL_RX);
    for p in 0..prepare {
        if let Some(Some(pq)) = s.ports.get_mut(p as usize) {
            let q = rx_queue(p);
            pq.rx.kick(&*s.t, q);
        }
    }
    s.ctrl_send(0, EV_DEVICE_READY, 1);
    crate::ktrace::log_fmt(format_args!(
        "virtio-serial: up, {max_ports} port(s) advertised, {prepare} prepared"
    ));
    DEV.with(|d| *d = Some(s));
    // The host answers the readiness announcement with its port descriptions;
    // pumping now means the agent port is usually open before the first paste.
    pump();
    true
}

/// Advance the control handshake and return any bytes received on the SPICE
/// port. Called from the UI pump.
pub fn pump() -> Vec<u8> {
    DEV.with(|d| {
        let Some(s) = d.as_mut() else {
            return Vec::new();
        };
        s.pump_ctrl();
        s.read_spice()
    })
}

/// Write `data` to the SPICE agent port. `false` if there is no open port.
pub fn write(data: &[u8]) -> bool {
    DEV.with(|d| d.as_mut().map(|s| s.write_spice(data)).unwrap_or(false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn port_queue_numbering_is_not_sequential() {
        // Port 0 owns 0/1, the CONTROL pair is 2/3, and port i>=1 owns
        // 2i+2 / 2i+3. The obvious guess — port 1 on queues 2 and 3 — lands on
        // the control channel, where a clipboard message reads as a malformed
        // control packet rather than as an error.
        assert_eq!((rx_queue(0), tx_queue(0)), (0, 1));
        assert_eq!((Q_CTRL_RX, Q_CTRL_TX), (2, 3));
        assert_eq!((rx_queue(1), tx_queue(1)), (4, 5));
        assert_eq!((rx_queue(2), tx_queue(2)), (6, 7));
        // No port's queues ever collide with the control pair.
        for p in 0..MAX_PORTS {
            assert_ne!(rx_queue(p), Q_CTRL_RX);
            assert_ne!(rx_queue(p), Q_CTRL_TX);
            assert_ne!(tx_queue(p), Q_CTRL_RX);
            assert_ne!(tx_queue(p), Q_CTRL_TX);
        }
        // And the queues we prepare fit what the shared transport tracks.
        let highest = tx_queue(MAX_PORTS - 1);
        assert!(
            (highest as usize) < crate::drivers::virtio::transport::MAX_QUEUES,
            "port {} needs queue {highest}, beyond MAX_QUEUES",
            MAX_PORTS - 1
        );
    }

    #[test_case]
    fn the_control_packet_is_eight_bytes_id_event_value() {
        // le32 id, le16 event, le16 value — reading `event` as a u32 makes
        // every event look like PORT_ADD on a little-endian host.
        assert_eq!(CTRL_LEN, 8);
        assert_eq!(EV_DEVICE_READY, 0);
        assert_eq!(EV_PORT_ADD, 1);
        assert_eq!(EV_PORT_READY, 3);
        assert_eq!(EV_PORT_OPEN, 6);
        assert_eq!(EV_PORT_NAME, 7);
    }
}
