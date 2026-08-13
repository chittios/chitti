//! **Net** — the TCP/IP subsystem: a [smoltcp](https://crates.io/crates/smoltcp)
//! stack over a pluggable NIC. A driver implements [`NetDevice`] (raw Ethernet
//! frames in/out + a MAC); [`ChittiPhy`] adapts that to smoltcp's `phy::Device`,
//! and [`NetState`] owns the `Interface` + sockets (DHCPv4, DNS, ICMP). The shell
//! commands (`/network`, `/ping`, `/wifi`) drive it; [`poll`] is pumped from the
//! shell idle loop so DHCP/ARP/ICMP make progress cooperatively.
//!
//! One NIC at a time (the first discovered: virtio-net, else e1000). Static or
//! DHCP addressing, DNS resolution, and ICMP echo (ping) are supported.

use crate::cap::ListenerId;
use crate::mm::Locked;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use smoltcp::iface::{Config, Interface, SocketHandle, SocketSet};
use smoltcp::phy::{Loopback, Medium};
use smoltcp::socket::{dhcpv4, dns, icmp, tcp};
use smoltcp::time::Instant;
use smoltcp::wire::{
    DnsQueryType, EthernetAddress, HardwareAddress, IpAddress, IpCidr, Ipv4Address, Ipv4Cidr,
    Ipv6Address, Ipv6Cidr,
};

pub mod e1000;
pub mod igb;
pub mod nic_ids;
pub mod r8169;
pub mod rndis;
pub mod usb_eth;
pub mod pci;
pub mod virtio_net_pci;

/// Wi-Fi driver facade — lives under [`crate::drivers::wifi`] (brcm FullMAC).
/// Re-exported here so existing `crate::net::wifi` call sites keep working.
pub use crate::drivers::wifi;

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

/// How many times to poll the loopback interface per `poll()` call before
/// yielding. A TCP handshake plus a small reply completes in ~6 rounds; the
/// bound stops a busy loopback transfer from starving the rest of the poll.
const LOOPBACK_POLL_ROUNDS: usize = 32;

// --- smoltcp phy adapter -------------------------------------------------

/// Adapts a [`NetDevice`] to smoltcp's `phy::Device`. `receive` copies the frame
/// into an owned token (so the returned TX token can borrow the device);
/// `transmit` fills a scratch buffer the closure writes, then hands it to the NIC.
pub mod ca_roots;
pub mod hashes;
pub mod http;
pub mod rsa;
pub mod sha1;
pub mod sntp;
pub mod ssh;
pub mod tls;
pub mod ws;
pub mod x509;

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
    /// The **loopback interface** — an Ethernet-medium interface over an
    /// in-memory `Loopback` device, addressed `127.0.0.1/8`. It is polled against
    /// its *own* `lo_sockets` set (never the NIC's `sockets`), so its egress can't
    /// interfere with NIC sockets and vice-versa. A client connecting to
    /// `127.0.0.1`/`localhost` opens its socket in `lo_sockets` with this
    /// interface's context (source `127.0.0.1`); its segments loop through the
    /// device queue back to a loopback listen socket in the same set — never
    /// touching the NIC. This is what makes in-OS `localhost` connections work.
    lo_iface: Interface,
    lo_phy: Loopback,
    lo_sockets: SocketSet<'static>,
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
    /// Active TCP listeners (a Network service agent's `net_listen`), keyed by id.
    listeners: BTreeMap<ListenerId, Listener>,
}

/// A TCP listener: a pool of sockets in `Listen` state on `port`. Accepting one
/// hands out an established socket and refills the pool with a fresh listener, so
/// the backlog stays open (the classic accept pattern). It keeps *two* pools —
/// one in the NIC socket set (external/hostfwd clients arrive at the NIC address)
/// and one in the loopback set (`127.0.0.1` clients arrive via the lo interface)
/// — so a single `listen(port)` serves both without either interface touching
/// the other's sockets.
struct Listener {
    port: u16,
    backlog: Vec<SocketHandle>,
    lo_backlog: Vec<SocketHandle>,
}

/// A TCP socket handle tagged with which interface's socket set it lives in, so
/// the raw `tcp_*` helpers (and a `Tcp`-backed channel) operate on the right set.
/// Loopback and NIC handles are distinct principals even if the underlying
/// `SocketHandle` index collides across the two sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TcpHandle {
    pub(crate) handle: SocketHandle,
    pub(crate) loopback: bool,
}

static NEXT_LISTENER: AtomicU64 = AtomicU64::new(0);

static NET: Locked<Option<NetState>> = Locked::new(None);

impl NetState {
    /// The socket set a [`TcpHandle`] belongs to: the loopback set for a loopback
    /// handle, the NIC set otherwise.
    fn tcp_set(&mut self, h: TcpHandle) -> &mut SocketSet<'static> {
        if h.loopback {
            &mut self.lo_sockets
        } else {
            &mut self.sockets
        }
    }
}

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
    let seed = crate::arch::now_ms().wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    config.random_seed = seed;
    let iface = Interface::new(config, &mut phy, now());
    // The loopback interface: an Ethernet-medium interface over an in-memory
    // queue, addressed 127.0.0.1/8, no routes (loopback is always on-link). It is
    // polled against its own `lo_sockets` set (below), never the NIC's. Ethernet
    // (not Ip) medium so it doesn't have to special-case socket types; loopback
    // ARP self-resolves because the device loops the request back to this
    // interface, which owns 127.0.0.1 and answers it. Its MAC is locally
    // administered (02:…) and irrelevant — frames never leave the queue.
    let mut lo_phy = Loopback::new(Medium::Ethernet);
    let mut lo_config = Config::new(HardwareAddress::Ethernet(EthernetAddress([0x02, 0, 0, 0, 0, 0x01])));
    lo_config.random_seed = seed ^ 0x5151_5151_5151_5151;
    let mut lo_iface = Interface::new(lo_config, &mut lo_phy, now());
    lo_iface.update_ip_addrs(|a| {
        let _ = a.push(IpCidr::Ipv4(Ipv4Cidr::new(Ipv4Address::new(127, 0, 0, 1), 8)));
        let _ = a.push(IpCidr::Ipv6(Ipv6Cidr::new(Ipv6Address::LOCALHOST, 128)));
    });
    // NIC: link-local IPv6 from MAC (EUI-64) so dual-stack is live even before
    // DHCPv4 / SLAAC global. Global addresses may arrive later via SLAAC.
    let mut iface = iface;
    let ll = ipv6_link_local_eui64(mac);
    iface.update_ip_addrs(|a| {
        let _ = a.push(IpCidr::Ipv6(Ipv6Cidr::new(ll, 64)));
    });
    let lo_sockets = SocketSet::new(Vec::new());
    let mut sockets = SocketSet::new(Vec::new());
    let dhcp = sockets.add(dhcpv4::Socket::new());
    // DNS socket: no servers until configured; four concurrent query slots.
    let queries: Vec<Option<dns::DnsQuery>> = vec![None, None, None, None];
    let dns = sockets.add(dns::Socket::new(&[], queries));
    NET.with(|n| {
        *n = Some(NetState {
            iface,
            lo_iface,
            lo_phy,
            lo_sockets,
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
            listeners: BTreeMap::new(),
        });
    });
    crate::ktrace::log_fmt(format_args!(
        "net: {ifname} up, MAC {}, IPv6 LL {}",
        fmt_mac(&mac),
        ll
    ));
}

/// Link-local `fe80::/64` address from a MAC via modified EUI-64 (RFC 4291).
fn ipv6_link_local_eui64(mac: [u8; 6]) -> Ipv6Address {
    let mut o = [0u8; 16];
    o[0] = 0xfe;
    o[1] = 0x80;
    // interface id
    o[8] = mac[0] ^ 0x02;
    o[9] = mac[1];
    o[10] = mac[2];
    o[11] = 0xff;
    o[12] = 0xfe;
    o[13] = mac[3];
    o[14] = mac[4];
    o[15] = mac[5];
    Ipv6Address::from(o)
}

/// True once a NIC has been brought up.
/// Re-establish the network interface after a suspend.
///
/// The NIC loses its ring base addresses, its receive descriptors and its link
/// state across S3, and every driver here is polled — so the retained interface
/// keeps being polled, never receives a frame, and reports no error at all.
///
/// **This drops the IP configuration with it**, because the interface is rebuilt
/// from scratch: a DHCP lease taken before the suspend is not re-asserted, and
/// the caller has to re-acquire it. That is stated rather than papered over —
/// silently keeping an address the network may have reassigned is worse than
/// coming back unconfigured and saying so.
pub fn resume() {
    let had = is_up();
    NET.with(|n| *n = None);
    autodetect();
    if is_up() {
        crate::ktrace::log("net", "resume re-probe ok -- address must be re-acquired (/network dhcp)");
    } else {
        crate::ktrace::log_fmt(format_args!(
            "net: resume re-probe found no NIC (was {})",
            if had { "up" } else { "down" }
        ));
    }
}

pub fn is_up() -> bool {
    NET.with(|n| n.is_some())
}

/// Set once boot (or `/network`) has finished looking for a NIC.
static DISCOVERY_DONE: AtomicBool = AtomicBool::new(false);

/// Record that NIC discovery has run, whether or not a device was found.
pub fn mark_probed() {
    DISCOVERY_DONE.store(true, Ordering::Relaxed);
}

/// Status-bar chip: Ready with a NIC, Pending until the first probe, Disabled
/// after a probe that found nothing.
pub fn device_status() -> crate::icons::DeviceStatus {
    crate::icons::device_status(is_up(), DISCOVERY_DONE.load(Ordering::Relaxed))
}

/// Discover and bring up the first available NIC, then **auto-start DHCP** so the
/// link comes up with an address on boot — the way a desktop OS does — without
/// the user running `/network dhcp`. Tries virtio-net, then a PCI NIC, over each
/// transport the platform exposes. No-op if none is found. Called once at boot.
pub fn autodetect() {
    if is_up() {
        mark_probed();
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
    // PCI NICs (virtio-net-pci, e1000/e1000e, igb/igc, r8169) — VBox + real hardware.
    if !brought_up {
        if let Some(nic) = crate::net::pci::probe() {
            init(nic, "eth0");
            brought_up = true;
        }
    }
    // USB Ethernet, last: it is the fallback for machines with no Ethernet port at
    // all (most laptops) and no WiFi driver, so a built-in NIC should win if there
    // is one. Only succeeds if enumeration already configured an adapter's bulk
    // endpoints.
    if !brought_up {
        if let Some(nic) = crate::net::usb_eth::probe() {
            init(Box::new(nic), "eth0");
            brought_up = true;
        }
    }
    // Autoconnect: kick off DHCP immediately; the shell idle loop pumps `poll`
    // and the lease lands a moment later (or the user sets a static IP, which
    // supersedes it). Best-effort SNTP runs once the first lease is applied
    // ([`try_boot_ntp`]).
    if brought_up {
        let _ = dhcp_start();
        crate::ktrace::log("net", "autoconnect: DHCP started on eth0");
    }
    mark_probed();
}

/// One-shot flag: best-effort SNTP after first IPv4 config (DHCP or static).
static NTP_BOOT_TRIED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// After DHCP/static brings up IPv4 + DNS, try SNTP once (non-blocking timeout).
/// No-op if already tried, no interface, or the clock is already NTP/manual.
pub fn try_boot_ntp() {
    use core::sync::atomic::Ordering;
    if NTP_BOOT_TRIED.swap(true, Ordering::Relaxed) {
        return;
    }
    // Don't stomp a human-set or already-NTP clock.
    if matches!(
        crate::clock::source(),
        crate::clock::ClockSource::Ntp | crate::clock::ClockSource::Manual
    ) {
        return;
    }
    let has_dns = NET.with(|n| n.as_ref().map(|s| !s.dns_servers.is_empty()).unwrap_or(false));
    if !has_dns {
        // Retry allowed if DNS wasn't ready yet.
        NTP_BOOT_TRIED.store(false, Ordering::Relaxed);
        return;
    }
    match ntp_sync_host("pool.ntp.org", 3_000) {
        Ok(u) => crate::ktrace::log_fmt(format_args!("net: boot SNTP ok (unix {u})")),
        Err(e) => {
            crate::ktrace::log_fmt(format_args!("net: boot SNTP skipped: {e}"));
            // Leave tried=true so a dead NTP server doesn't retry every poll.
        }
    }
}

/// Pump the stack: poll the interface (ARP/DHCP/ICMP/…) and apply any freshly
/// acquired DHCP lease. Called from the shell idle loop and around blocking ops.
pub fn poll() {
    NET.with(|n| {
        let Some(s) = n.as_mut() else { return };
        let t = now();
        s.iface.poll(t, &mut s.phy, &mut s.sockets);
        // Pump the loopback interface against its own socket set. Each poll drains
        // its queue and emits queued segments; a handshake or a small transfer
        // takes a handful of rounds, so loop until it makes no further progress
        // (bounded — a large transfer just continues on the next call).
        use smoltcp::iface::PollResult;
        for _ in 0..LOOPBACK_POLL_ROUNDS {
            if let PollResult::None = s.lo_iface.poll(t, &mut s.lo_phy, &mut s.lo_sockets) {
                break;
            }
        }
        if s.dhcp_on {
            // The DHCP `Event` borrows the socket (hence s.sockets), so copy the
            // lease out to owned values before touching `s` again.
            enum Act {
                Cfg(Ipv4Cidr, Option<Ipv4Address>, Vec<Ipv4Address>),
                Deconf,
                None,
            }
            let act = match s.sockets.get_mut::<dhcpv4::Socket>(s.dhcp).poll() {
                Some(dhcpv4::Event::Configured(cfg)) => Act::Cfg(
                    cfg.address,
                    cfg.router,
                    cfg.dns_servers.iter().copied().collect(),
                ),
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
    // Outside the lock: one-shot SNTP when we have IPv4 + DNS (after DHCP).
    if NET.with(|n| {
        n.as_ref()
            .map(|s| s.ip.is_some() && !s.dns_servers.is_empty())
            .unwrap_or(false)
    }) {
        try_boot_ntp();
    }
}

fn apply_ipv4(s: &mut NetState, cidr: Ipv4Cidr, router: Option<Ipv4Address>, dns: &[Ipv4Address]) {
    // Preserve IPv6 addresses (link-local / SLAAC) when DHCPv4 reconfigures v4.
    s.iface.update_ip_addrs(|a| {
        let v6: Vec<IpCidr> = a
            .iter()
            .filter(|c| matches!(c, IpCidr::Ipv6(_)))
            .copied()
            .collect();
        a.clear();
        let _ = a.push(IpCidr::Ipv4(cidr));
        for c in v6 {
            let _ = a.push(c);
        }
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
    // Drop IPv4 only; keep IPv6 link-local.
    s.iface.update_ip_addrs(|a| {
        let v6: Vec<IpCidr> = a
            .iter()
            .filter(|c| matches!(c, IpCidr::Ipv6(_)))
            .copied()
            .collect();
        a.clear();
        for c in v6 {
            let _ = a.push(c);
        }
    });
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
    /// IPv6 addresses currently on the NIC (link-local and any SLAAC globals).
    pub ipv6: Vec<Ipv6Cidr>,
}

pub fn info() -> Option<Info> {
    NET.with(|n| {
        n.as_ref().map(|s| {
            let ipv6: Vec<Ipv6Cidr> = s
                .iface
                .ip_addrs()
                .iter()
                .filter_map(|c| match c {
                    IpCidr::Ipv6(v) => Some(*v),
                    _ => None,
                })
                .collect();
            Info {
                ifname: s.ifname.clone(),
                mac: s.mac,
                ip: s.ip,
                gateway: s.gateway,
                dns: s.dns_servers.clone(),
                dhcp: s.dhcp_on,
                ipv6,
            }
        })
    })
}

/// Resolve `name` to any IP (A then AAAA). Prefer IPv4 when both exist so
/// existing DHCP-only networks keep working; IPv6-only hosts get AAAA.
pub fn resolve_any(name: &str, timeout_ms: u64) -> Result<IpAddress, &'static str> {
    match resolve(name, timeout_ms) {
        Ok(v4) => Ok(IpAddress::Ipv4(v4)),
        Err(_) => resolve_aaaa(name, timeout_ms).map(IpAddress::Ipv6),
    }
}

/// DNS AAAA query (uncached path).
pub fn resolve_aaaa(name: &str, timeout_ms: u64) -> Result<Ipv6Address, &'static str> {
    let query = NET.with(|n| {
        let s = n.as_mut().ok_or("no network interface")?;
        if s.dns_servers.is_empty() {
            return Err("no DNS server configured");
        }
        let dns_h = s.dns;
        let cx = s.iface.context();
        s.sockets
            .get_mut::<dns::Socket>(dns_h)
            .start_query(cx, name, DnsQueryType::Aaaa)
            .map_err(|_| "DNS AAAA query failed to start")
    })?;
    let deadline = crate::arch::now_ms() + timeout_ms;
    loop {
        poll();
        let done = NET.with(|n| {
            let s = n.as_mut().ok_or("no network interface")?;
            match s.sockets.get_mut::<dns::Socket>(s.dns).get_query_result(query) {
                Ok(addrs) => {
                    let v6 = addrs.iter().find_map(|a| match a {
                        IpAddress::Ipv6(v) => Some(*v),
                        _ => None,
                    });
                    v6.map(Some).ok_or("no AAAA record")
                }
                Err(dns::GetQueryResultError::Pending) => Ok(None),
                Err(_) => Err("DNS AAAA query failed"),
            }
        })?;
        if let Some(a) = done {
            return Ok(a);
        }
        if crate::arch::now_ms() >= deadline {
            return Err("DNS AAAA timeout");
        }
        if crate::shell::poll_interrupt() {
            return Err("cancelled");
        }
        crate::sched::yield_now();
    }
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

// --- DNS resolution + cache ---------------------------------------------

/// Resolved-host cache: `name` → `(addr, expiry_ms)`. Avoids re-querying DNS for
/// every subresource fetch to the same host (a page pulls dozens of assets off a
/// handful of CDN hosts — without this each one paid a full DNS round trip).
/// Bounded; entries expire after [`DNS_TTL_MS`].
static DNS_CACHE: Locked<BTreeMap<String, (Ipv4Address, u64)>> = Locked::new(BTreeMap::new());
static DNS_HITS: AtomicU64 = AtomicU64::new(0);
static DNS_MISSES: AtomicU64 = AtomicU64::new(0);

/// How long a cached resolution stays fresh (smoltcp doesn't surface the record
/// TTL, so use a conservative fixed lifetime).
const DNS_TTL_MS: u64 = 300_000; // 5 min
/// Cap on cached hosts (LRU-ish: cleared wholesale when exceeded — tiny + rare).
const DNS_CACHE_CAP: usize = 128;

/// Look up `name` in the DNS cache if still fresh.
fn dns_cache_get(name: &str, now: u64) -> Option<Ipv4Address> {
    DNS_CACHE.with(|c| {
        c.get(name)
            .and_then(|(a, exp)| if *exp > now { Some(*a) } else { None })
    })
}

/// Insert `name → addr` with a fresh expiry.
fn dns_cache_put(name: &str, addr: Ipv4Address, now: u64) {
    DNS_CACHE.with(|c| {
        if c.len() >= DNS_CACHE_CAP && !c.contains_key(name) {
            c.clear();
        }
        c.insert(String::from(name), (addr, now + DNS_TTL_MS));
    });
}

/// `(cached_entries, hits, misses)` — for `/network` / diagnostics.
pub fn dns_cache_stats() -> (usize, u64, u64) {
    (
        DNS_CACHE.with(|c| c.len()),
        DNS_HITS.load(Ordering::Relaxed),
        DNS_MISSES.load(Ordering::Relaxed),
    )
}

/// Best-effort **DNS prefetch**: warm the cache for `host` (e.g. a page's
/// subresource domains / `<link rel="dns-prefetch">`) so the real fetch skips
/// the DNS round trip. No-op for empties/literals/`localhost`/already-cached;
/// a short timeout keeps a dead host from stalling the caller.
pub fn prefetch_dns(host: &str) {
    if host.is_empty() || host.eq_ignore_ascii_case("localhost") {
        return;
    }
    // Skip dotted-quad IPv4 literals (no DNS needed).
    if host.split('.').count() == 4 && host.split('.').all(|p| p.parse::<u8>().is_ok()) {
        return;
    }
    if dns_cache_get(host, crate::arch::now_ms()).is_some() {
        return;
    }
    let _ = resolve(host, 2_000);
}

/// Resolve `name` to an IPv4 address, pumping the stack until the query
/// completes or times out (~`timeout_ms`). Results are cached (see
/// [`DNS_CACHE`]); a fresh cache hit returns immediately with no round trip.
pub fn resolve(name: &str, timeout_ms: u64) -> Result<Ipv4Address, &'static str> {
    let now = crate::arch::now_ms();
    if let Some(addr) = dns_cache_get(name, now) {
        DNS_HITS.fetch_add(1, Ordering::Relaxed);
        return Ok(addr);
    }
    DNS_MISSES.fetch_add(1, Ordering::Relaxed);
    let addr = resolve_uncached(name, timeout_ms)?;
    dns_cache_put(name, addr, now);
    Ok(addr)
}

/// The actual DNS query (uncached). Wrapped by [`resolve`].
fn resolve_uncached(name: &str, timeout_ms: u64) -> Result<Ipv4Address, &'static str> {
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
        if crate::shell::poll_interrupt() {
            return Err("cancelled");
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
        if crate::shell::poll_interrupt() {
            break Err("cancelled");
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

// --- SNTP (UDP/123) -------------------------------------------------------

/// Synchronise the wall clock from an SNTP server at `addr` (typically after
/// DNS of `pool.ntp.org`). Returns the new Unix time on success.
///
/// Human-driven (`/ntp`); not an agent tool. Pumps `poll` + Ctrl+C.
pub fn ntp_sync(addr: Ipv4Address, timeout_ms: u64) -> Result<u64, &'static str> {
    use smoltcp::socket::udp;
    use smoltcp::wire::IpEndpoint;

    let handle = NET.with(|n| {
        let s = n.as_mut().ok_or("no network interface")?;
        if s.ip.is_none() {
            return Err("interface has no IP address");
        }
        let rx = udp::PacketBuffer::new(vec![udp::PacketMetadata::EMPTY; 4], vec![0u8; 512]);
        let tx = udp::PacketBuffer::new(vec![udp::PacketMetadata::EMPTY; 4], vec![0u8; 512]);
        let mut sock = udp::Socket::new(rx, tx);
        sock.bind(12300u16).map_err(|_| "udp bind failed")?;
        Ok(s.sockets.add(sock))
    })?;

    let now = crate::clock::now_unix().max(0) as u64;
    let req = sntp::build_request(now);
    let result = (|| {
        NET.with(|n| {
            let s = n.as_mut().ok_or("no network interface")?;
            let sock = s.sockets.get_mut::<udp::Socket>(handle);
            let endpoint = IpEndpoint::new(IpAddress::Ipv4(addr), 123);
            sock.send_slice(&req, endpoint).map_err(|_| "udp send failed")?;
            Ok(())
        })?;
        let deadline = crate::arch::now_ms() + timeout_ms;
        loop {
            poll();
            let got = NET.with(|n| {
                let s = n.as_mut().ok_or("no network interface")?;
                let sock = s.sockets.get_mut::<udp::Socket>(handle);
                match sock.recv() {
                    Ok((data, _meta)) => {
                        let mut buf = [0u8; 48];
                        let n = data.len().min(48);
                        buf[..n].copy_from_slice(&data[..n]);
                        Ok(Some(buf))
                    }
                    Err(udp::RecvError::Exhausted) => Ok(None),
                    Err(_) => Err("udp recv failed"),
                }
            })?;
            if let Some(pkt) = got {
                let unix = sntp::parse_reply(&pkt).map_err(|e| match e {
                    sntp::SntpError::KissOfDeath => "SNTP kiss-o'-death (server unsync)",
                    sntp::SntpError::Unsync => "SNTP server unsynchronised",
                    sntp::SntpError::BadMode => "SNTP bad mode",
                    sntp::SntpError::Implausible => "SNTP implausible timestamp",
                    _ => "SNTP parse failed",
                })?;
                let trusted = crate::clock::source().trusted();
                if !sntp::plausible(crate::clock::now_unix(), unix, trusted) {
                    return Err("SNTP time outside plausible window");
                }
                crate::clock::set_unix_with_source(unix as i64, crate::clock::ClockSource::Ntp);
                crate::ktrace::log_fmt(format_args!("net: SNTP synced from {addr} → unix {unix}"));
                return Ok(unix);
            }
            if crate::arch::now_ms() >= deadline {
                return Err("SNTP timeout");
            }
            if crate::shell::poll_interrupt() {
                return Err("cancelled");
            }
            crate::shell::upkeep();
            crate::sched::yield_now();
        }
    })();

    NET.with(|n| {
        if let Some(s) = n.as_mut() {
            s.sockets.remove(handle);
        }
    });
    result
}

/// Resolve `pool.ntp.org` (or `host`) and run [`ntp_sync`].
pub fn ntp_sync_host(host: &str, timeout_ms: u64) -> Result<u64, &'static str> {
    let host = if host.is_empty() { "pool.ntp.org" } else { host };
    // Dotted-quad literal.
    if let Ok(a) = host.parse::<Ipv4Address>() {
        return ntp_sync(a, timeout_ms);
    }
    // smoltcp Ipv4Address doesn't implement FromStr the same way — try manual.
    if let Some(a) = parse_ipv4(host) {
        return ntp_sync(a, timeout_ms);
    }
    let addr = resolve(host, timeout_ms.min(5_000).max(2_000))?;
    ntp_sync(addr, timeout_ms)
}

fn parse_ipv4(s: &str) -> Option<Ipv4Address> {
    let mut o = [0u8; 4];
    let mut i = 0;
    for part in s.split('.') {
        if i >= 4 {
            return None;
        }
        o[i] = part.parse().ok()?;
        i += 1;
    }
    if i != 4 {
        return None;
    }
    Some(Ipv4Address::new(o[0], o[1], o[2], o[3]))
}

// --- TCP listeners + raw socket I/O (inter-agent stream handoff) ------------
//
// A Network service agent uses these: `listen` opens a backlog of Listen-state
// sockets on a port; `try_accept` hands out an established one (a live inbound
// connection) as a bare `SocketHandle`, which `channel::adopt_tcp` wraps as a
// Tcp-backed channel it can then `channel_grant` to another agent. The raw
// `tcp_*` helpers are how the channel's Tcp backend moves bytes.

/// Per-listener backlog depth: how many sockets sit in `Listen` on the port so
/// several inbound connections can queue before the serve loop accepts them.
const LISTEN_BACKLOG: usize = 4;

fn new_listen_socket(port: u16) -> tcp::Socket<'static> {
    let mut sock = tcp::Socket::new(
        tcp::SocketBuffer::new(vec![0u8; 64 * 1024]),
        tcp::SocketBuffer::new(vec![0u8; 16 * 1024]),
    );
    // Not fatal if it fails; the socket just won't accept (checked at accept).
    let _ = sock.listen(port);
    sock
}

/// Open a TCP listener on `port` (a pool of Listen-state sockets). Returns a
/// `ListenerId` the caller resolves via a `NetListen` capability. Requires the
/// interface to be up.
pub fn listen(port: u16) -> Result<ListenerId, &'static str> {
    NET.with(|n| {
        let s = n.as_mut().ok_or("no network interface (try /network dhcp)")?;
        // A pool in the NIC set (external/hostfwd clients) and a pool in the
        // loopback set (127.0.0.1 clients), so the one listener serves both.
        let mut backlog = Vec::with_capacity(LISTEN_BACKLOG);
        let mut lo_backlog = Vec::with_capacity(LISTEN_BACKLOG);
        for _ in 0..LISTEN_BACKLOG {
            backlog.push(s.sockets.add(new_listen_socket(port)));
            lo_backlog.push(s.lo_sockets.add(new_listen_socket(port)));
        }
        let id = NEXT_LISTENER.fetch_add(1, Ordering::SeqCst);
        s.listeners.insert(id, Listener { port, backlog, lo_backlog });
        crate::ktrace::log_fmt(format_args!("net: listening on TCP :{port} (listener {id}, NIC + loopback)"));
        Ok(id)
    })
}

/// Non-blocking accept: if any backlog socket has left `Listen` state (an
/// inbound connection arrived), hand it out as a `SocketHandle` and refill the
/// backlog slot with a fresh listener so the port keeps accepting. `net::poll()`
/// (pumped from `shell::upkeep`) drives the Listen→Established transition.
pub fn try_accept(id: ListenerId) -> Option<TcpHandle> {
    NET.with(|n| {
        let s = n.as_mut()?;
        let port = s.listeners.get(&id)?.port;
        // Find a backlog slot whose socket is fully established, in either pool.
        // Requiring Established (not merely "left Listen") avoids handing out a
        // half-open SynReceived socket whose `may_recv` is transiently false —
        // which an echo loop would misread as an immediate EOF. The NIC pool is
        // checked first, then the loopback pool.
        let nic_idx = {
            let lis = s.listeners.get(&id)?;
            lis.backlog.iter().position(|&h| s.sockets.get::<tcp::Socket>(h).state() == tcp::State::Established)
        };
        if let Some(idx) = nic_idx {
            let established = s.listeners.get(&id)?.backlog[idx];
            let fresh = s.sockets.add(new_listen_socket(port));
            if let Some(lis) = s.listeners.get_mut(&id) {
                lis.backlog[idx] = fresh;
            }
            return Some(TcpHandle { handle: established, loopback: false });
        }
        let lo_idx = {
            let lis = s.listeners.get(&id)?;
            lis.lo_backlog.iter().position(|&h| s.lo_sockets.get::<tcp::Socket>(h).state() == tcp::State::Established)?
        };
        let established = s.listeners.get(&id)?.lo_backlog[lo_idx];
        let fresh = s.lo_sockets.add(new_listen_socket(port));
        if let Some(lis) = s.listeners.get_mut(&id) {
            lis.lo_backlog[lo_idx] = fresh;
        }
        Some(TcpHandle { handle: established, loopback: true })
    })
}

/// Close a listener: drop its backlog sockets (both pools) and forget it.
pub fn close_listener(id: ListenerId) {
    NET.with(|n| {
        if let Some(s) = n.as_mut() {
            if let Some(lis) = s.listeners.remove(&id) {
                for h in lis.backlog {
                    s.sockets.remove(h);
                }
                for h in lis.lo_backlog {
                    s.lo_sockets.remove(h);
                }
            }
        }
    });
}

/// Non-blocking read from a connected TCP socket into `buf`. `Some(0)` = no data
/// buffered right now; `None` = the socket is gone or can no longer receive.
/// Used by `channel`'s Tcp backend.
pub fn tcp_recv(handle: TcpHandle, buf: &mut [u8]) -> Option<usize> {
    poll();
    NET.with(|n| {
        let s = n.as_mut()?;
        let sock = s.tcp_set(handle).get_mut::<tcp::Socket>(handle.handle);
        if sock.can_recv() {
            sock.recv_slice(buf).ok()
        } else if sock.may_recv() {
            Some(0) // still open, just nothing buffered
        } else {
            None // peer closed / socket done
        }
    })
}

/// Non-blocking write of `data` to a connected TCP socket. Returns bytes queued
/// (may be fewer than `data.len()`), or `None` if the socket can't send.
pub fn tcp_send(handle: TcpHandle, data: &[u8]) -> Option<usize> {
    let r = NET.with(|n| {
        let s = n.as_mut()?;
        let sock = s.tcp_set(handle).get_mut::<tcp::Socket>(handle.handle);
        if sock.can_send() {
            sock.send_slice(data).ok()
        } else if sock.may_send() {
            Some(0)
        } else {
            None
        }
    });
    poll(); // push the queued segment out
    r
}

/// Whether the socket may still deliver more received data (open for reading).
pub fn tcp_may_recv(handle: TcpHandle) -> bool {
    NET.with(|n| n.as_mut().map(|s| s.tcp_set(handle).get_mut::<tcp::Socket>(handle.handle).may_recv()).unwrap_or(false))
}

/// Bytes still queued in the socket's transmit buffer (not yet sent *and* acked
/// by the peer — smoltcp keeps them buffered until acknowledged).
pub fn tcp_send_queue(handle: TcpHandle) -> usize {
    NET.with(|n| n.as_mut().map(|s| s.tcp_set(handle).get_mut::<tcp::Socket>(handle.handle).send_queue()).unwrap_or(0))
}

/// Gracefully close (and remove) a TCP socket adopted by a channel. Crucially,
/// this first **drains the transmit buffer** — a `tcp_send`/`try_write` only
/// queues bytes into the socket's TX buffer; removing the socket before smoltcp
/// has actually transmitted (and had acked) those bytes would truncate the
/// response and give an HTTP client a Content-Length mismatch. So: initiate the
/// close (queues a FIN after the pending data), poll until the TX buffer drains
/// or a deadline, then remove.
pub fn tcp_close(handle: TcpHandle) {
    NET.with(|n| {
        if let Some(s) = n.as_mut() {
            s.tcp_set(handle).get_mut::<tcp::Socket>(handle.handle).close();
        }
    });
    let deadline = crate::arch::now_ms() + 5_000;
    loop {
        poll();
        if tcp_send_queue(handle) == 0 || crate::arch::now_ms() >= deadline {
            break;
        }
        crate::shell::upkeep();
        crate::sched::yield_now();
    }
    NET.with(|n| {
        if let Some(s) = n.as_mut() {
            s.tcp_set(handle).remove(handle.handle);
        }
    });
    poll();
}

#[cfg(test)]
mod dns_tests {
    use super::*;

    #[test_case]
    fn dns_cache_freshness_and_expiry() {
        let a = Ipv4Address::new(93, 184, 216, 34);
        DNS_CACHE.with(|c| c.clear());
        dns_cache_put("example.com", a, 1_000);
        // Fresh within the TTL window.
        assert_eq!(dns_cache_get("example.com", 1_000), Some(a));
        assert_eq!(dns_cache_get("example.com", 1_000 + DNS_TTL_MS - 1), Some(a));
        // Expired at/after the deadline.
        assert_eq!(dns_cache_get("example.com", 1_000 + DNS_TTL_MS), None);
        assert_eq!(dns_cache_get("example.com", 1_000 + DNS_TTL_MS + 1), None);
        // Unknown host misses.
        assert_eq!(dns_cache_get("other.com", 1_000), None);
        DNS_CACHE.with(|c| c.clear());
    }

    #[test_case]
    fn dns_cache_cap_evicts() {
        DNS_CACHE.with(|c| c.clear());
        for i in 0..(DNS_CACHE_CAP + 5) {
            dns_cache_put(&alloc::format!("h{i}.example"), Ipv4Address::new(10, 0, 0, 1), 0);
        }
        // Wholesale clear on overflow keeps the map bounded.
        assert!(DNS_CACHE.with(|c| c.len()) <= DNS_CACHE_CAP);
        DNS_CACHE.with(|c| c.clear());
    }
}
