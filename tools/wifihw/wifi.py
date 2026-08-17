#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Live BCM4387/4388 WiFi bring-up harness over the m1n1 proxyclient.

Lets us drive the WiFi endpoint on a real M2 (t8112/j473) directly from Python —
power (SMC gP0d), PERST#, refclk, LTSSM, BAR mapping, and the BAR0 backplane
window — so the dongle bring-up sequence (and specifically what powers the
SYS_MEM/RAM domain) can be found EMPIRICALLY instead of guessed, then ported to
the ChittiOS driver (kernel/src/arch/aarch64/apple_pcie.rs + drivers/wifi/brcm).

Usage:
    M1N1DEVICE=/dev/cu.usbmodemW945XQL26D1 \
      third_party/m1n1/.venv/bin/python tools/wifihw/wifi.py <cmd>

Requires the M2 booted into m1n1 **proxy mode** (not ChittiOS). p.read32/write32
recover from external aborts (return 0xabad1dea), so pokes can't crash m1n1.
"""
import os, sys, time, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "third_party/m1n1/proxyclient"))
os.environ.setdefault("M1N1DEVICE", "/dev/cu.usbmodemW945XQL26D1")

from m1n1.setup import p, u          # noqa: E402
from m1n1.fw.smc import SMCClient    # noqa: E402

ABORT = 0xABAD1DEA

# ── Apple PCIe port 0 (WiFi) ────────────────────────────────────────────────
ECAM  = 0x690000000
PORT0 = 0x681000000
PHY0  = 0x680084000
P_APPCLK, P_REFCLK, P_PERST, P_STATUS, P_LINKSTS, P_LTSSM = 0x800, 0x810, 0x814, 0x804, 0x208, 0x080
APPCLK_EN, APPCLK_CGDIS = 1 << 0, 1 << 8
REFCLK_EN, REFCLK_CGDIS = 1 << 0, 1 << 8
PERST_OFF = 1 << 0                 # set = PERST# deasserted
STATUS_READY, LINKSTS_UP, LTSSM_START = 1 << 0, 1 << 0, 1 << 0
PHY_CFG, PHY_CTL = 0x00, 0x04
CFG_REFCLK0REQ, CFG_REFCLK1REQ = 1 << 0, 1 << 1
CFG_REFCLK0ACK, CFG_REFCLK1ACK = 1 << 2, 1 << 3
CFG_REFCLKEN = (1 << 9) | (1 << 10)
CTL_CFGACC = 1 << 15

def pr(a):  return p.read32(a)
def pw(a, v): p.write32(a, v)
def rmw_set(a, bits):   pw(a, pr(a) | bits)
def rmw_clr(a, bits):   pw(a, pr(a) & ~bits)

# ── SMC power (gP0d = GPIO13) ───────────────────────────────────────────────
_smc = None
_smc_clean = False
def smc():
    global _smc, _smc_clean
    if _smc is None:
        addr = u.adt["arm-io/smc"].get_reg(0)[0]
        _smc = SMCClient(u, addr, None)
        try:
            _smc.start()
            _smc_clean = True
        except Exception as e:
            # SMC RTKit stays booted across proxyclient runs — a second start()
            # times out, and a RE-ATTACHED client CANNOT drive gP0d (the power
            # write silently no-ops → chip never powers → link won't train).
            _smc_clean = False
            print("  !! smc.start() TIMED OUT — SMC already running from a prior run.")
            print("  !! gP0d power writes will NOT work. REBOOT the M2 into fresh")
            print("  !! m1n1 proxy mode and run this ONCE (single process).")
        _smc.start_ep(0x20)
    return _smc

def smc_stop():
    # DO NOT call _smc.stop() — stopping the SMC RTKit shuts down power
    # management on the M2 and DROPS the USB/proxy connection (learned the hard
    # way). Leave the SMC running; the chip stays powered for the next run.
    pass

def gp0d(val):
    smc().smcep.write32("gP0d", val)
def gp1a(val):
    try: smc().smcep.write32("gP1a", val)
    except Exception as e: print("  gP1a:", e)
def gp0d_read():
    try: return smc().smcep.read32("gP0d")
    except Exception as e: return f"err:{e}"

def power_on():
    gp0d(0x800001); gp1a(1)
def power_off():
    gp0d(0x800000); gp1a(0)

# ── PCIe config space (ECAM), abort-safe ────────────────────────────────────
def cfg(b, d, f, o):
    return p.read32(ECAM + (b << 20) + (d << 15) + (f << 12) + o)
def cfg_w(b, d, f, o, v):
    p.write32(ECAM + (b << 20) + (d << 15) + (f << 12) + o, v)

def bar_to_cpu(bus):
    if bus >= 0x600000000: return bus
    if bus >= 0xc0000000:  return 0x600000000 + bus
    return bus

# ── PERST# GPIO 166 (pinctrl_ap, ACTIVE_LOW) — the physical reset line ───────
PINCTRL_AP = 0x23C100000
PERST_PIN = 166
G_DATA, G_MODE, G_MODE_OUT, G_PERIPH = 1 << 0, 0x7 << 1, 1 << 1, 0x3 << 5

def gpio_out(pin, level_high):
    a = PINCTRL_AP + 4 * pin
    v = pr(a) & ~(G_MODE | G_PERIPH | G_DATA)
    v |= G_MODE_OUT
    if level_high:
        v |= G_DATA
    pw(a, v)

# ── Port bring-up (apple_pcie_setup_link order) ───────────────────────
def perst_assert():
    gpio_out(PERST_PIN, False)          # ACTIVE_LOW: assert = drive 0
    rmw_clr(PORT0 + P_PERST, PERST_OFF)
def perst_deassert():
    rmw_set(PORT0 + P_PERST, PERST_OFF)
    gpio_out(PERST_PIN, True)           # ACTIVE_LOW: deassert = drive 1

def refclk_on():
    ctl = pr(PHY0 + PHY_CTL); pw(PHY0 + PHY_CTL, ctl | CTL_CFGACC)
    rmw_set(PHY0 + PHY_CFG, CFG_REFCLK0REQ)
    for _ in range(500):
        if pr(PHY0 + PHY_CFG) & CFG_REFCLK0ACK: break
        time.sleep(1e-4)
    rmw_set(PHY0 + PHY_CFG, CFG_REFCLK1REQ)
    for _ in range(500):
        if pr(PHY0 + PHY_CFG) & CFG_REFCLK1ACK: break
        time.sleep(1e-4)
    rmw_clr(PHY0 + PHY_CTL, CTL_CFGACC)
    rmw_set(PHY0 + PHY_CFG, CFG_REFCLKEN)
    rmw_set(PORT0 + P_REFCLK, REFCLK_EN)

# ── Root-complex / bridge config (ChittiOS configure_wifi_root_only) ────────
# In m1n1 proxy mode the RC/bridge is NOT configured (bus0 config reads
# 0xffffffff, bus1 aborts) — m1n1 defers PCIe to the OS. Program the WiFi root
# port (bus 0, dev 0) like ChittiOS does so bus 1 config is forwarded and the
# non-pref MEM window (PCI 0xc000_0000 → CPU 0x6c…) reaches the endpoint BARs.
PORT_MEM_BASE, PORT_MEM_LIMIT = 0x10, 0x14
PORT_PREF_BASE, PORT_PREF_LIMIT, PORT_PREF_BU, PORT_PREF_LU = 0x18, 0x20, 0x24, 0x28

def configure_root_port(dev=0, sec=1):
    # primary=0 secondary=sec subordinate=sec
    cfg_w(0, dev, 0, 0x18, (sec << 8) | (sec << 16))
    cfg_w(0, dev, 0, 0x20, 0xFFF0_C000)   # non-pref MEM: base 0xc000_0000 limit 0xfff0_0000
    cfg_w(0, dev, 0, 0x24, 0x0000_FFF0)   # pref disabled (base>limit)
    cfg_w(0, dev, 0, 0x28, 0); cfg_w(0, dev, 0, 0x2C, 0)
    cfg_w(0, dev, 0, 0x1C, 0x0000_00F0)   # IO disabled
    cmd = cfg(0, dev, 0, 4)
    cfg_w(0, dev, 0, 4, cmd | 0b111)      # IO|MEM|MASTER
    # mirror non-pref window into port MMIO (pref mirrors stay 0)
    pw(PORT0 + PORT_MEM_BASE, 0xC000); pw(PORT0 + PORT_MEM_LIMIT, 0xFFF0)
    for o in (PORT_PREF_BASE, PORT_PREF_LIMIT, PORT_PREF_BU, PORT_PREF_LU):
        pw(PORT0 + o, 0)

def _st(tag):
    print("  [%s] PERST=%#x STATUS=%#x LINKSTS=%#x APPCLK=%#x REFCLK=%#x lane_cfg=%#x gP0d=%s" % (
        tag, pr(PORT0 + P_PERST), pr(PORT0 + P_STATUS), pr(PORT0 + P_LINKSTS),
        pr(PORT0 + P_APPCLK), pr(PORT0 + P_REFCLK), pr(PHY0 + PHY_CFG), gp0d_read()))

# ── ADT PCIe tunables (m1n1 proxy mode doesn't run pcie_init, so PHY/RC config
#    the link needs is unset). Apply via m1n1's OWN tunable applier — same data
#    ChittiOS hardcodes (reapply_host_tunables), applied by m1n1 itself (safe).
RC_BASE = 0x680000000
AXI2AF_BASE = 0x68C000000
APCIE = "/arm-io/apcie"
BRIDGE = "/arm-io/apcie/pci-bridge0"

def apply_tunables():
    for path, prop, base in [
        (APCIE, "apcie-axi2af-tunables", AXI2AF_BASE),
        (APCIE, "apcie-common-tunables", RC_BASE),
        (BRIDGE, "apcie-config-tunables", PORT0),
        (BRIDGE, "pcie-rc-tunables", ECAM),
    ]:
        try:
            p.tunables_apply_local_addr(path, prop, base)
            print("  tunable %-26s @ %#x ok" % (prop, base))
        except Exception as e:
            print("  tunable %-26s ERR %s %s" % (prop, type(e).__name__, e))

# t8112 pcie_pins: CLKREQ on pinctrl_ap 162/163/164 → periph1.
# PERIPH = GENMASK(6,5), INPUT_ENABLE = BIT(9). See pinctrl-apple-gpio.c.
CLKREQ_PINS = (162, 163, 164)
PIN_PERIPH, PIN_INPUT_EN = 0x3 << 5, 1 << 9

def pinmux_clkreq():
    for pin in CLKREQ_PINS:
        a = PINCTRL_AP + 4 * pin
        cur = pr(a)
        pw(a, (cur & ~PIN_PERIPH) | (1 << 5) | PIN_INPUT_EN)

def bringup(power_during_perst=True, off_first=True, off_hold=0.5, settle=0.1, verbose=True):
    """Full port bring-up (apple_pcie_setup_link order). Returns (link_up, status, linksts)."""
    apply_tunables()
    configure_root_port()
    pinmux_clkreq()
    if verbose: _st("start")
    rmw_set(PORT0 + P_APPCLK, APPCLK_EN)
    perst_assert(); time.sleep(0.001)
    # Rail off ONLY while PERST# is held — never with PERST deasserted.
    if off_first:
        power_off(); time.sleep(off_hold)
    if power_during_perst:
        power_on()
    if verbose: _st("perst+pwr")
    refclk_on()
    if verbose: _st("refclk")
    time.sleep(settle)
    perst_deassert()
    time.sleep(settle)
    for _ in range(2500):
        if pr(PORT0 + P_STATUS) & STATUS_READY: break
        time.sleep(1e-4)
    if verbose: _st("perst-off")
    rmw_clr(PORT0 + P_REFCLK, REFCLK_CGDIS)
    rmw_clr(PORT0 + P_APPCLK, APPCLK_CGDIS)
    pw(PORT0 + P_LTSSM, 0); time.sleep(0.001)
    pw(PORT0 + P_LTSSM, LTSSM_START)
    up = False
    for _ in range(10000):
        ls = pr(PORT0 + P_LINKSTS)
        if ls & LINKSTS_UP: up = True; break
        if not (ls & LINKSTS_BUSY) and _ > 100: break   # training stopped, not up
        time.sleep(1e-4)
    if verbose: _st("ltssm")
    return up, pr(PORT0 + P_STATUS), pr(PORT0 + P_LINKSTS)

LINKSTS_BUSY = 1 << 2

# ── WiFi endpoint + backplane (BAR0 sliding window) ─────────────────────────
CFG_BAR0_WINDOW = 0x80
SI_ENUM = 0x18000000
CC_CHIPID, CC_EROMPTR = 0x00, 0xfc
SYSMEM_BASE, SYSMEM_COREINFO = 0x18024000, 0x00

def wifi_present():
    return (cfg(1, 0, 0, 0) & 0xffff) == 0x14e4

def map_bars(bar0_pci=0xc1000000, bar2_pci=0xc0000000):
    """Program BAR0/BAR2 and enable MEM; return (bar0_cpu, bar2_cpu)."""
    cmd = cfg(1, 0, 0, 4)
    cfg_w(1, 0, 0, 4, cmd & ~0b10)
    # BAR0 (64-bit): lo carries type bits (0x4 = 64-bit non-pref)
    cfg_w(1, 0, 0, 0x10, bar0_pci | 0x4); cfg_w(1, 0, 0, 0x14, 0)
    cfg_w(1, 0, 0, 0x18, bar2_pci | 0x4); cfg_w(1, 0, 0, 0x1c, 0)
    cfg_w(1, 0, 0, 4, cmd | 0b110)
    cfg(1, 0, 0, 4)
    return bar_to_cpu(bar0_pci), bar_to_cpu(bar2_pci)

def bp_read(bar0_cpu, addr):
    cfg_w(1, 0, 0, CFG_BAR0_WINDOW, addr & ~0xfff)
    cfg(1, 0, 0, 4)
    return pr(bar0_cpu + (addr & 0xfff))

def sysmem_coreinfo(bar0_cpu):
    return bp_read(bar0_cpu, SYSMEM_BASE + SYSMEM_COREINFO)


# ── Commands ────────────────────────────────────────────────────────────────
def cmd_state():
    print("port PERST=%#x STATUS=%#x LINKSTS=%#x APPCLK=%#x REFCLK=%#x LTSSM=%#x" % (
        pr(PORT0 + P_PERST), pr(PORT0 + P_STATUS), pr(PORT0 + P_LINKSTS),
        pr(PORT0 + P_APPCLK), pr(PORT0 + P_REFCLK), pr(PORT0 + P_LTSSM)))
    print("gP0d readback:", gp0d_read())
    print("wlan 1:0.0 id:", hex(cfg(1, 0, 0, 0)), "(present)" if wifi_present() else "(absent/reset)")

def cmd_up():
    up, st, ls = bringup()
    print("bringup: link_up=%s STATUS=%#x LINKSTS=%#x" % (up, st, ls))
    print("wlan id:", hex(cfg(1, 0, 0, 0)), "present" if wifi_present() else "ABSENT")
    if wifi_present():
        b0, b2 = map_bars()
        chipid = bp_read(b0, SI_ENUM + CC_CHIPID)
        erom = bp_read(b0, SI_ENUM + CC_EROMPTR)
        ci = sysmem_coreinfo(b0)
        print("  BAR0_cpu=%#x BAR2_cpu=%#x" % (b0, b2))
        print("  chipcommon chipid=%#x eromptr=%#x" % (chipid, erom))
        print("  ==> SYS_MEM coreinfo=%#x  (0xffffffff/abad=unpowered; a real value=POWERED)" % ci)
        print("  BAR2@0=%#x BAR2@rambase(0x740000)=%#x" % (pr(b2), pr(b2 + 0x740000)))

if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "state"
    try:
        {"state": cmd_state, "up": cmd_up}.get(c, cmd_state)()
    finally:
        # Cleanly shut the SMC RTKit down so the NEXT run's start() succeeds and
        # gP0d writes work. Without this each run poisons the next.
        smc_stop()
