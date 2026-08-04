//! Byte-level mutations for the fuzzer (libFuzzer-flavoured, dependency-free).
//!
//! Every mutation is a pure function of `(seed, data)` so a crash is fully
//! reproducible: the same `--seed` replays the same input stream.

use crate::rng::Rng;

/// The mutation budget per input: one primitive chosen at random and applied.
pub fn mutate(data: &mut Vec<u8>, rng: &mut Rng) {
    if data.is_empty() {
        // A one-byte input is enough to mutate.
        data.push(rng.byte());
        return;
    }
    match rng.range(0, 11) {
        0 => flip_bit(data, rng),
        1 => flip_byte(data, rng),
        2 => overwrite_byte(data, rng),
        3 => insert_bytes(data, rng),
        4 => delete_chunk(data, rng),
        5 => duplicate_chunk(data, rng),
        6 => splice_chunk(data, rng),
        7 => change_byte_order(data, rng),
        8 => insert_zero_block(data, rng),
        9 => set_interesting(data, rng),
        _ => grow_block(data, rng),
    }
}

/// Randomly flip a single bit in a single byte.
fn flip_bit(data: &mut Vec<u8>, rng: &mut Rng) {
    let i = rng.range(0, data.len());
    data[i] ^= 1 << rng.range(0, 8);
}

/// Overwrite one byte with a random value.
fn flip_byte(data: &mut Vec<u8>, rng: &mut Rng) {
    let i = rng.range(0, data.len());
    data[i] ^= rng.byte();
}

fn overwrite_byte(data: &mut Vec<u8>, rng: &mut Rng) {
    let i = rng.range(0, data.len());
    data[i] = rng.byte();
}

/// Insert a random byte at a random position.
fn insert_bytes(data: &mut Vec<u8>, rng: &mut Rng) {
    let at = rng.range(0, data.len() + 1);
    let n = 1 + rng.range(0, 4);
    let mut buf = Vec::with_capacity(n);
    for _ in 0..n {
        buf.push(rng.byte());
    }
    data.splice(at..at, buf);
}

/// Delete a random contiguous chunk.
fn delete_chunk(data: &mut Vec<u8>, rng: &mut Rng) {
    let start = rng.range(0, data.len());
    let end = (start + 1 + rng.range(0, 16)).min(data.len());
    data.drain(start..end);
}

/// Duplicate a random chunk (inserts a copy right after the original).
fn duplicate_chunk(data: &mut Vec<u8>, rng: &mut Rng) {
    if data.len() >= 4096 {
        return; // never let growth run away on a single step
    }
    let start = rng.range(0, data.len());
    let end = (start + 1 + rng.range(0, 16)).min(data.len());
    let chunk: Vec<u8> = data[start..end].to_vec();
    data.splice(end..end, chunk);
}

/// Splice: concatenate two random chunks from the input.
fn splice_chunk(data: &mut Vec<u8>, rng: &mut Rng) {
    if data.len() >= 4096 {
        return;
    }
    let a = rng.range(0, data.len());
    let b = rng.range(0, data.len());
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let chunk: Vec<u8> = data[lo..(hi + 1).min(data.len())].to_vec();
    let at = rng.range(0, data.len() + 1);
    data.splice(at..at, chunk);
}

/// Reverse a random 2-8 byte window (catches endianness/alignment assumptions).
fn change_byte_order(data: &mut Vec<u8>, rng: &mut Rng) {
    let start = rng.range(0, data.len());
    let n = 2 + rng.range(0, 7);
    let end = (start + n).min(data.len());
    if end - start < 2 {
        return;
    }
    data[start..end].reverse();
}

/// Insert a run of zeroes (parser length-field / padding traps).
fn insert_zero_block(data: &mut Vec<u8>, rng: &mut Rng) {
    if data.len() >= 4096 {
        return;
    }
    let at = rng.range(0, data.len() + 1);
    let n = 1 + rng.range(0, 32);
    data.splice(at..at, core::iter::repeat(0).take(n));
}

/// Overwrite a byte with a "magic" value (0x00, 0xff, 0x7f, 0x80, 0x10, 0x1a).
fn set_interesting(data: &mut Vec<u8>, rng: &mut Rng) {
    const INTERESTING: [u8; 6] = [0x00, 0xff, 0x7f, 0x80, 0x10, 0x1a];
    let i = rng.range(0, data.len());
    data[i] = INTERESTING[rng.range(0, INTERESTING.len())];
}

/// Grow the input by copying a random prefix (helps length-driven parsers reach
/// deeper code paths, the libFuzzer "increase" stage).
fn grow_block(data: &mut Vec<u8>, rng: &mut Rng) {
    if data.len() >= 4096 {
        return;
    }
    let n = (data.len() * 2 + 1).min(8192).min(rng.range(1, 4096));
    let copy = data.to_vec();
    while data.len() < n {
        data.push(copy[rng.range(0, copy.len())]);
    }
}
