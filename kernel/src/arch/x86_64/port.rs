//! Raw x86 port I/O (`in`/`out`). Every caller is responsible for knowing
//! the target port is a valid, mapped I/O port for the given access width;
//! these instructions cannot fault but can have arbitrary hardware side
//! effects, which is why every wrapper here stays `unsafe`.

use core::arch::asm;

/// Write a byte to `port`.
///
/// # Safety
/// `port` must refer to an I/O port where a single-byte write is the
/// documented, intended access for that device.
#[inline]
pub unsafe fn outb(port: u16, value: u8) {
    unsafe {
        asm!("out dx, al", in("dx") port, in("al") value, options(nomem, nostack, preserves_flags));
    }
}

/// Read a byte from `port`.
///
/// # Safety
/// `port` must refer to an I/O port where a single-byte read is the
/// documented, intended access for that device.
#[inline]
pub unsafe fn inb(port: u16) -> u8 {
    let value: u8;
    unsafe {
        asm!("in al, dx", out("al") value, in("dx") port, options(nomem, nostack, preserves_flags));
    }
    value
}

/// Write a 32-bit dword to `port`.
///
/// # Safety
/// `port` must refer to an I/O port where a 4-byte write is the documented,
/// intended access for that device.
#[inline]
pub unsafe fn outl(port: u16, value: u32) {
    unsafe {
        asm!("out dx, eax", in("dx") port, in("eax") value, options(nomem, nostack, preserves_flags));
    }
}

/// Read a 32-bit dword from `port`.
///
/// # Safety
/// As `outl`, for a 4-byte read.
#[inline]
pub unsafe fn inl(port: u16) -> u32 {
    let value: u32;
    unsafe {
        asm!("in eax, dx", out("eax") value, in("dx") port, options(nomem, nostack, preserves_flags));
    }
    value
}

/// Write a 16-bit word to `port`.
///
/// # Safety
/// As `outb`, for a 2-byte write.
#[inline]
pub unsafe fn outw(port: u16, value: u16) {
    unsafe {
        asm!("out dx, ax", in("dx") port, in("ax") value, options(nomem, nostack, preserves_flags));
    }
}

/// Read a 16-bit word from `port`.
///
/// # Safety
/// As `inb`, for a 2-byte read.
#[inline]
pub unsafe fn inw(port: u16) -> u16 {
    let value: u16;
    unsafe {
        asm!("in ax, dx", out("ax") value, in("dx") port, options(nomem, nostack, preserves_flags));
    }
    value
}
