//! A persistent, ext4-backed key/value store for `synapse::fs` — so agent
//! runtime writes survive reboots on the installed system. It keeps the small
//! set of agent files in an in-memory cache and **persists on every mutation by
//! rewriting the (small, dedicated) ext4 data partition** with the verified
//! [`Ext4Writer`] (mkfs + write-all), reading it back with [`Ext4Reader`] on
//! mount.
//!
//! This reuses the two verified drivers, so every persisted image is
//! e2fsck-clean by construction. It is *rewrite-on-sync*, not an incremental
//! read-write ext4 driver — correct + durable for the KB-scale agent state
//! (notes, facts, sessions), but O(total) per write; a true incremental RW
//! layer (allocate/free from live bitmaps, in-place directory edits) is a
//! documented follow-on.

use crate::block::ext4::{Ext4Writer, FileSpec};
use crate::block::ext4_read::Ext4Reader;
use crate::block::{virtio::VirtioBlk, Partition};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// synapse::fs keys are flat strings that routinely contain `/`
/// (e.g. `sess/5/cmp/26`, `skills/1/body.md`), but `/` is the path separator
/// and is illegal inside an ext4 directory-entry name. Percent-encode `/` (and
/// `%` itself, so the mapping is reversible) into a legal single-component
/// filename; [`key_decode`] inverts it on mount.
fn key_encode(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    for c in key.chars() {
        match c {
            '%' => out.push_str("%25"),
            '/' => out.push_str("%2F"),
            _ => out.push(c),
        }
    }
    out
}

fn key_decode(name: &str) -> String {
    let b = name.as_bytes();
    let mut out = String::with_capacity(name.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() + 1 && i + 3 <= b.len() {
            match &name[i..i + 3] {
                "%2F" | "%2f" => {
                    out.push('/');
                    i += 3;
                    continue;
                }
                "%25" => {
                    out.push('%');
                    i += 3;
                    continue;
                }
                _ => {}
            }
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

pub struct Ext4Store {
    dev: VirtioBlk,
    start: u64,
    count: u64,
    cache: BTreeMap<String, Vec<u8>>,
}

impl Ext4Store {
    /// Mount the ext4 data partition at `[start, start+count)` of `dev`, reading
    /// its existing root files into the cache. Returns `None` if the partition
    /// is not a readable ext4.
    pub fn mount(mut dev: VirtioBlk, start: u64, count: u64) -> Option<Ext4Store> {
        let mut cache = BTreeMap::new();
        {
            let mut part = Partition::new(&mut dev, start, count);
            let mut r = Ext4Reader::open(&mut part)?;
            for (name, _ino, is_dir) in r.list_root() {
                if is_dir {
                    continue;
                }
                if let Some(sz) = r.file_size(&name) {
                    let mut buf = vec![0u8; sz as usize];
                    let n = r.read_root_file(&name, &mut buf).unwrap_or(0);
                    buf.truncate(n);
                    cache.insert(key_decode(&name), buf);
                }
            }
        }
        crate::ktrace::log_fmt(format_args!("ext4_store: mounted ext4 data partition, {} file(s) recovered", cache.len()));
        Some(Ext4Store { dev, start, count, cache })
    }

    /// Rewrite the partition from the current cache (verified mkfs + write-all).
    fn sync(&mut self) {
        // Encode keys to legal ext4 filenames; hold the owned strings so the
        // borrowed `FileSpec::name`s outlive the format call.
        let names: Vec<String> = self.cache.keys().map(|k| key_encode(k)).collect();
        let files: Vec<FileSpec> = names.iter().zip(self.cache.values()).map(|(n, d)| FileSpec { name: n.as_str(), data: d.as_slice() }).collect();
        let mut part = Partition::new(&mut self.dev, self.start, self.count);
        if let Err(e) = Ext4Writer::format(&mut part, &files) {
            crate::ktrace::log_fmt(format_args!("ext4_store: sync failed: {:?}", e));
        }
    }

    pub fn write(&mut self, name: &str, data: &[u8]) {
        self.cache.insert(String::from(name), data.to_vec());
        self.sync();
    }
    pub fn read(&self, name: &str) -> Option<Vec<u8>> {
        self.cache.get(name).cloned()
    }
    pub fn exists(&self, name: &str) -> bool {
        self.cache.contains_key(name)
    }
    pub fn list(&self) -> Vec<String> {
        self.cache.keys().cloned().collect()
    }
    pub fn delete(&mut self, name: &str) -> bool {
        let removed = self.cache.remove(name).is_some();
        if removed {
            self.sync();
        }
        removed
    }
}
