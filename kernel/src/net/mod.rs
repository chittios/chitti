//! **Net** — the TCP/IP subsystem: a [smoltcp](https://crates.io/crates/smoltcp)
//! stack over a pluggable NIC. A driver implements [`NetDevice`] (raw Ethernet
//! frames in/out + a MAC); [`ChittiPhy`] adapts that to smoltcp's `phy::Device`,
//! and [`NetState`] owns the `Interface` + sockets (DHCPv4, DNS, ICMP). The shell
//! commands (`/network`, `/ping`, `/wifi`) drive it; [`poll`] is pumped from the
//! shell idle loop so DHCP/ARP/ICMP make progress cooperatively.
//!
//! One NIC at a time (the first discovered: virtio-net, else e1000). Static or
//! DHCP addressing, DNS resolution, and ICMP echo (ping) are supported.

use crate::mm::Locked;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use smoltcp::iface::{Config, Interface, SocketHandle, SocketSet};
use smoltcp::socket::{dhcpv4, dns, icmp};
use smoltcp::time::Instant;
use smoltcp::wire::{DnsQueryType, EthernetAddress, HardwareAddress, IpAddress, IpCidr, Ipv4Address, Ipv4Cidr};

pub mod e1000;
pub mod pci;
pub mod virtio_net_pci;

/// A raw-Ethernet NIC the stack sits on. Frames exclude the FCS. Implemented by
/// the virtio-net and e1000 drivers.
pub trait NetDevice {
    /// The NIC's MAC address.
    fn mac(&self) -> [u8; 6];
    /// Poll one received frame into `out`; returns its length, or `None` if the
    /// RX ring is empty.
    fn receive(&mut self, out: &mut [u8]) -> Option<usize>;
    /// Transmit one Ethernet frame.
    fn transmit(&mut self, frame: &[u8]);
    /// Link MTU (payload); 1500 for Ethernet.
    fn mtu(&self) -> usize {
        1500
    }
}

const MTU: usize = 1514;

// --- smoltcp phy adapter -------------------------------------------------

/// Adapts a [`NetDevice`] to smoltcp's `phy::Device`. `receive` copies the frame
/// into an owned token (so the returned TX token can borrow the device);
/// `transmit` fills a scratch buffer the closure writes, then hands it to the NIC.
pub mod http;
pub mod tls;
pub mod ws;

pub struct ChittiPhy {
    dev: Box<dyn NetDevice>,
}

pub struct RxToken(Vec<u8>);
pub struct TxToken<'a> {
    dev: &'a mut dyn NetDevice,
}

impl smoltcp::phy::RxToken for RxToken {
    // smoltcp 0.13 hands the received frame to the closure by shared ref.
    fn consume<R, F: FnOnce(&[u8]) -> R>(self, f: F) -> R {
        f(self.0.as_slice())
    }
}
impl smoltcp::phy::TxToken for TxToken<'_> {
    fn consume<R, F: FnOnce(&mut [u8]) -> R>(self, len: usize, f: F) -> R {
        let mut buf = vec![0u8; len];
        let r = f(&mut buf);
        self.dev.transmit(&buf);
        r
    }
}

impl smoltcp::phy::Device for ChittiPhy {
    type RxToken<'a> = RxToken;
    type TxToken<'a> = TxToken<'a>;

    fn receive(&mut self, _t: Instant) -> Option<(RxToken, TxToken<'_>)> {
        let mut buf = [0u8; MTU];
        let n = self.dev.receive(&mut buf)?;
        Some((RxToken(buf[..n].to_vec()), TxToken { dev: &mut *self.dev }))
    }
    fn transmit(&mut self, _t: Instant) -> Option<TxToken<'_>> {
        Some(TxToken { dev: &mut *self.dev })
    }
    fn capabilities(&self) -> smoltcp::phy::DeviceCapabilities {
        let mut c = smoltcp::phy::DeviceCapabilities::default();
        c.medium = smoltcp::phy::Medium::Ethernet;
        c.max_transmission_unit = MTU;
        c
    }
}

// --- global stack state --------------------------------------------------

struct NetState {
    iface: Interface,
    sockets: SocketSet<'static>,
    phy: ChittiPhy,
    dhcp: SocketHandle,
    dns: SocketHandle,
    dhcp_on: bool,
    mac: [u8; 6],
    ip: Option<Ipv4Cidr>,
    gateway: Option<Ipv4Address>,
    dns_servers: Vec<Ipv4Address>,
    /// A friendly name for the interface (e.g. "wlan0" once `/wifi` "connects").
    ifname: String,
}

static NET: Locked<Option<NetState>> = Locked::new(None);

fn now() -> Instant {
    Instant::from_millis(crate::arch::now_ms() as i64)
}

/// Bring the stack up on `dev` (link only; no address until DHCP or a static
/// config is set). Idempotent-ish: replaces any existing stack.
pub fn init(dev: Box<dyn NetDevice>, ifname: &str) {
    let mac = dev.mac();
    let mut phy = ChittiPhy { dev };
    let mut config = Config::new(HardwareAddress::Ethernet(EthernetAddress(mac)));
    // Seed smoltcp's PRNG (TCP ISN / DHCP xid) from the boot clock.
    config.random_seed = crate::arch::now_ms().wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let iface = Interface::new(config, &mut phy, now());
    let mut sockets = SocketSet::new(Vec::new());
    let dhcp = sockets.add(dhcpv4::Socket::new());
    // DNS socket: no servers until configured; four concurrent query slots.
    let queries: Vec<Option<dns::DnsQuery>> = vec![None, None, None, None];
    let dns = sockets.add(dns::Socket::new(&[], queries));
    NET.with(|n| {
        *n = Some(NetState {
            iface,
            sockets,
            phy,
            dhcp,
            dns,
            dhcp_on: false,
            mac,
            ip: None,
            gateway: None,
            dns_servers: Vec::new(),
            ifname: String::from(ifname),
        });
    });
    crate::ktrace::log_fmt(format_args!("net: {ifname} up, MAC {}", fmt_mac(&mac)));
}

/// True once a NIC has been brought up.
pub fn is_up() -> bool {
    NET.with(|n| n.is_some())
}

/// Discover and bring up the first available NIC, then **auto-start DHCP** so the
/// link comes up with an address on boot — the way a desktop OS does — without
/// the user running `/network dhcp`. Tries virtio-net, then a PCI NIC, over each
/// transport the platform exposes. No-op if none is found. Called once at boot.
pub fn autodetect() {
    if is_up() {
        return;
    }
    let mut brought_up = false;
    #[cfg(target_arch = "aarch64")]
    {
        if let Some(nic) = crate::arch::aarch64::virtio_net::VirtioNetMmio::probe() {
            init(Box::new(nic), "eth0");
            brought_up = true;
        }
    }
    // PCI NICs (virtio-net-pci, e1000) — VBox + real Intel hardware.
    if !brought_up {
        if let Some(nic) = crate::net::pci::probe() {
            init(nic, "eth0");
            brought_up = true;
        }
    }
    // Autoconnect: kick off DHCP immediately; the shell idle loop pumps `poll`
    // and the lease lands a moment later (or the user sets a static IP, which
    // supersedes it).
    if brought_up {
        let _ = dhcp_start();
        crate::ktrace::log("net", "autoconnect: DHCP started on eth0");
    }
}

/// Pump the stack: poll the interface (ARP/DHCP/ICMP/…) and apply any freshly
/// acquired DHCP lease. Called from the shell idle loop and around blocking ops.
pub fn poll() {
    NET.with(|n| {
        let Some(s) = n.as_mut() else { return };
        let t = now();
        s.iface.poll(t, &mut s.phy, &mut s.sockets);
        if s.dhcp_on {
            // The DHCP `Event` borrows the socket (hence s.sockets), so copy the
            // lease out to owned values before touching `s` again.
            enum Act {
                Cfg(Ipv4Cidr, Option<Ipv4Address>, Vec<Ipv4Address>),
                Deconf,
                None,
            }
            let act = match s.sockets.get_mut::<dhcpv4::Socket>(s.dhcp).poll() {
                Some(dhcpv4::Event::Configured(cfg)) => Act::Cfg(cfg.address, cfg.router, cfg.dns_servers.iter().copied().collect()),
                Some(dhcpv4::Event::Deconfigured) => Act::Deconf,
                None => Act::None,
            };
            match act {
                Act::Cfg(a, r, d) => apply_ipv4(s, a, r, &d),
                Act::Deconf => clear_addrs(s),
                Act::None => {}
            }
        }
    });
}

fn apply_ipv4(s: &mut NetState, cidr: Ipv4Cidr, router: Option<Ipv4Address>, dns: &[Ipv4Address]) {
    s.iface.update_ip_addrs(|a| {
        a.clear();
        let _ = a.push(IpCidr::Ipv4(cidr));
    });
    s.iface.routes_mut().remove_default_ipv4_route();
    if let Some(gw) = router {
        let _ = s.iface.routes_mut().add_default_ipv4_route(gw);
    }
    s.ip = Some(cidr);
    s.gateway = router;
    s.dns_servers = dns.to_vec();
    let servers: Vec<IpAddress> = dns.iter().map(|a| IpAddress::Ipv4(*a)).collect();
    s.sockets.get_mut::<dns::Socket>(s.dns).update_servers(&servers);
    crate::ktrace::log_fmt(format_args!("net: configured {} gw {:?}", cidr, router));
}

fn clear_addrs(s: &mut NetState) {
    s.iface.update_ip_addrs(|a| a.clear());
    s.iface.routes_mut().remove_default_ipv4_route();
    s.ip = None;
    s.gateway = None;
}

/// Start DHCPv4 (dynamic addressing). Resets the socket so it re-DISCOVERs.
pub fn dhcp_start() -> Result<(), &'static str> {
    NET.with(|n| {
        let s = n.as_mut().ok_or("no network interface")?;
        s.sockets.get_mut::<dhcpv4::Socket>(s.dhcp).reset();
        clear_addrs(s);
        s.dhcp_on = true;
        Ok(())
    })
}

/// Assign a static IPv4 config (disables DHCP). `cidr` like 192.168.1.50/24.
pub fn set_static(ip: Ipv4Address, prefix: u8, gw: Option<Ipv4Address>) -> Result<(), &'static str> {
    NET.with(|n| {
        let s = n.as_mut().ok_or("no network interface")?;
        s.dhcp_on = false;
        let cidr = Ipv4Cidr::new(ip, prefix);
        apply_ipv4(s, cidr, gw, &[]);
        Ok(())
    })
}

/// Set the DNS resolver list (does not touch addressing).
pub fn set_dns(servers: &[Ipv4Address]) -> Result<(), &'static str> {
    NET.with(|n| {
        let s = n.as_mut().ok_or("no network interface")?;
        s.dns_servers = servers.to_vec();
        let list: Vec<IpAddress> = servers.iter().map(|a| IpAddress::Ipv4(*a)).collect();
        s.sockets.get_mut::<dns::Socket>(s.dns).update_servers(&list);
        Ok(())
    })
}

/// A snapshot of the current network configuration for `/network info`.
pub struct Info {
    pub ifname: String,
    pub mac: [u8; 6],
    pub ip: Option<Ipv4Cidr>,
    pub gateway: Option<Ipv4Address>,
    pub dns: Vec<Ipv4Address>,
    pub dhcp: bool,
}

pub fn info() -> Option<Info> {
    NET.with(|n| {
        n.as_ref().map(|s| Info {
            ifname: s.ifname.clone(),
            mac: s.mac,
            ip: s.ip,
            gateway: s.gateway,
            dns: s.dns_servers.clone(),
            dhcp: s.dhcp_on,
        })
    })
}

/// Rename the interface (used by the `/wifi` facade to present "wlan0").
pub fn set_ifname(name: &str) {
    NET.with(|n| {
        if let Some(s) = n.as_mut() {
            s.ifname = String::from(name);
        }
    });
}

pub fn fmt_mac(m: &[u8; 6]) -> String {
    alloc::format!("{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}", m[0], m[1], m[2], m[3], m[4], m[5])
}

// --- DNS resolution ------------------------------------------------------

/// Resolve `name` to an IPv4 address, pumping the stack until the query
/// completes or times out (~`timeout_ms`).
pub fn resolve(name: &str, timeout_ms: u64) -> Result<Ipv4Address, &'static str> {
    let query = NET.with(|n| {
        let s = n.as_mut().ok_or("no network interface")?;
        if s.dns_servers.is_empty() {
            return Err("no DNS server configured");
        }
        // Disjoint field borrows: the DNS socket (s.sockets) + the iface context
        // (s.iface) are both needed mutably by start_query.
        let dns_h = s.dns;
        let cx = s.iface.context();
        s.sockets
            .get_mut::<dns::Socket>(dns_h)
            .start_query(cx, name, DnsQueryType::A)
            .map_err(|_| "DNS query failed to start")
    })?;
    let deadline = crate::arch::now_ms() + timeout_ms;
    loop {
        poll();
        let done = NET.with(|n| {
            let s = n.as_mut().ok_or("no network interface")?;
            match s.sockets.get_mut::<dns::Socket>(s.dns).get_query_result(query) {
                Ok(addrs) => {
                    let v4 = addrs.iter().find_map(|a| match a {
                        IpAddress::Ipv4(v) => Some(*v),
                        #[allow(unreachable_patterns)]
                        _ => None,
                    });
                    v4.map(Some).ok_or("no A record")
                }
                Err(dns::GetQueryResultError::Pending) => Ok(None),
                Err(_) => Err("DNS query failed"),
            }
        })?;
        if let Some(a) = done {
            return Ok(a);
        }
        if crate::arch::now_ms() >= deadline {
            return Err("DNS timeout");
        }
        crate::sched::yield_now();
    }
}

// --- ICMP echo (ping) ----------------------------------------------------

/// Ping `addr` once, returning the round-trip time in milliseconds.
pub fn ping(addr: Ipv4Address, timeout_ms: u64) -> Result<u64, &'static str> {
    use smoltcp::wire::{Icmpv4Packet, Icmpv4Repr};
    // A dedicated ICMP socket for this ping.
    let handle = NET.with(|n| {
        let s = n.as_mut().ok_or("no network interface")?;
        if s.ip.is_none() {
            return Err("interface has no IP address");
        }
        let rx = icmp::PacketBuffer::new(vec![icmp::PacketMetadata::EMPTY; 4], vec![0u8; 1024]);
        let tx = icmp::PacketBuffer::new(vec![icmp::PacketMetadata::EMPTY; 4], vec![0u8; 1024]);
        Ok(s.sockets.add(icmp::Socket::new(rx, tx)))
    })?;
    let ident = 0x22b_u16;
    // Bind + send the echo request.
    let start = crate::arch::now_ms();
    let seq = (start & 0xffff) as u16;
    NET.with(|n| {
        let s = n.as_mut().ok_or("no network interface")?;
        let sock = s.sockets.get_mut::<icmp::Socket>(handle);
        sock.bind(icmp::Endpoint::Ident(ident)).map_err(|_| "icmp bind failed")?;
        let repr = Icmpv4Repr::EchoRequest { ident, seq_no: seq, data: b"chitti-ping-0123" };
        let payload = sock.send(repr.buffer_len(), IpAddress::Ipv4(addr)).map_err(|_| "icmp send failed")?;
        let mut pkt = Icmpv4Packet::new_unchecked(payload);
        repr.emit(&mut pkt, &smoltcp::phy::ChecksumCapabilities::default());
        Ok::<(), &'static str>(())
    })?;
    let deadline = start + timeout_ms;
    let result = loop {
        poll();
        let got = NET.with(|n| {
            let s = n.as_mut()?;
            let sock = s.sockets.get_mut::<icmp::Socket>(handle);
            if sock.can_recv() {
                if let Ok((payload, _src)) = sock.recv() {
                    let pkt = Icmpv4Packet::new_unchecked(payload);
                    if let Ok(Icmpv4Repr::EchoReply { seq_no, .. }) = Icmpv4Repr::parse(&pkt, &smoltcp::phy::ChecksumCapabilities::default()) {
                        if seq_no == seq {
                            return Some(crate::arch::now_ms().saturating_sub(start));
                        }
                    }
                }
            }
            None
        });
        if let Some(rtt) = got {
            break Ok(rtt);
        }
        if crate::arch::now_ms() >= deadline {
            break Err("ping timeout");
        }
        crate::sched::yield_now();
    };
    // Drop the temporary ICMP socket.
    NET.with(|n| {
        if let Some(s) = n.as_mut() {
            s.sockets.remove(handle);
        }
    });
    result
}
