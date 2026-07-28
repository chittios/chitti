//! Prefilled-KV prefix cache: reuse the model context for a system prompt that
//! has already been prefilled, instead of prefilling it again.
//!
//! Several callers start a *fresh* model context and then prefill a system
//! prompt that never changes between calls — a content agent's `serve_loop`
//! does it on **every HTTP request**, a wasm app's model-ask loop on every ask,
//! and a sub-agent dispatch on every delegation. On the 0.8B at ~105 prefill
//! tok/s a ~1.5k-token SOUL is ~14 s, so that identical prefix was the dominant
//! cost of a served request. Restoring it is a clone of the cache instead: tens
//! of milliseconds, and independent of how long the prompt is.
//!
//! Two deliberate scoping choices:
//!
//! - **The store is owned by the chat session, not global.** A KV prefix is only
//!   valid for the exact model that produced it, and `/model load` drops the
//!   `ChatSession` — so tying the cache's lifetime to the session makes a
//!   stale-model hit unrepresentable, with no generation counter and no lock. A
//!   global cache would have to prove the model had not changed underneath it.
//! - **Keys are the full system-prompt text, compared exactly.** A hash would
//!   make a collision serve one agent's KV to another, and the failure mode
//!   there is a confident wrong answer rather than an error. Comparing a few KiB
//!   of text is nothing against the prefill it saves.
//!
//! [`PrefixStore`] is generic over the payload so its eviction policy is pure
//! and testable under `cargo xtask test` (x86, no model); [`Snapshot`] is the
//! cortex-specific payload.

use alloc::string::String;
use alloc::vec::Vec;

/// A prefilled model context: the cache as it stood after the prefix, plus the
/// position that cache is valid up to.
pub struct Snapshot {
    pub cache: super::model::Cache,
    pub pos: usize,
}

struct Entry<T> {
    key: String,
    value: T,
    /// Payload cost, supplied by the caller (the KV size is not something a
    /// generic container can compute).
    bytes: usize,
    /// Clock reading of the last hit — LRU victim selection.
    used: u64,
}

/// A byte-budgeted LRU map from prefix text to a prefilled context.
///
/// The budget is a *byte* budget rather than an entry count because a prefix's
/// cost is its token length, which varies by orders of magnitude between a
/// one-line routing SOUL and a full agent system prompt. Insertions that cannot
/// fit even in an empty store are declined rather than evicting everything to
/// make room for something that will be evicted next.
pub struct PrefixStore<T> {
    entries: Vec<Entry<T>>,
    bytes: usize,
    budget: usize,
    clock: u64,
    hits: u64,
    misses: u64,
}

impl<T> PrefixStore<T> {
    pub fn new(budget: usize) -> Self {
        Self { entries: Vec::new(), bytes: 0, budget, clock: 0, hits: 0, misses: 0 }
    }

    /// Look up `key`, marking it most-recently-used. Counts a hit or a miss.
    pub fn get(&mut self, key: &str) -> Option<&T> {
        self.clock += 1;
        let clock = self.clock;
        match self.entries.iter().position(|e| e.key == key) {
            Some(i) => {
                self.hits += 1;
                self.entries[i].used = clock;
                Some(&self.entries[i].value)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Whether a payload of `bytes` could be stored at all.
    ///
    /// Callers should ask *before* building the payload: a KV snapshot is a
    /// multi-hundred-megabyte clone on a large model, and cloning it only to
    /// have `insert` decline it is pure allocator churn. (The 27B's 1546-token
    /// prefix is ~353 MiB — well over the default budget, so that clone was
    /// always going to be thrown away.)
    pub fn accepts(&self, bytes: usize) -> bool {
        bytes <= self.budget
    }

    /// Insert (or replace) `key`, evicting least-recently-used entries until
    /// `bytes` fits. Returns `false` (leaving the store untouched) when the
    /// payload cannot fit the budget even empty — so a caller can report
    /// "declined" rather than claiming it cached something it did not.
    pub fn insert(&mut self, key: String, value: T, bytes: usize) -> bool {
        if !self.accepts(bytes) {
            return false;
        }
        // Replacing an existing key must release its bytes first, or the same
        // prefix re-inserted would inflate the accounting until the store
        // evicted everything including itself.
        if let Some(i) = self.entries.iter().position(|e| e.key == key) {
            self.bytes -= self.entries[i].bytes;
            self.entries.swap_remove(i);
        }
        while self.bytes + bytes > self.budget {
            // The store is non-empty here: `bytes <= budget`, so an empty store
            // always has room.
            let victim = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.used)
                .map(|(i, _)| i)
                .expect("a non-empty store has a least-recently-used entry");
            self.bytes -= self.entries[victim].bytes;
            self.entries.swap_remove(victim);
        }
        self.clock += 1;
        self.bytes += bytes;
        self.entries.push(Entry { key, value, bytes, used: self.clock });
        true
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Bytes of payload currently held.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// Lookups served from cache, and lookups that had to prefill.
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test_case]
    fn a_stored_prefix_is_returned_and_a_missing_one_is_not() {
        let mut s: PrefixStore<u32> = PrefixStore::new(1000);
        assert!(s.get("sys-a").is_none());
        s.insert("sys-a".to_string(), 7, 100);
        assert_eq!(s.get("sys-a").copied(), Some(7));
        assert!(s.get("sys-b").is_none());
        assert_eq!(s.stats(), (1, 2), "one hit, two misses");
        assert_eq!(s.bytes(), 100);
    }

    /// Keys are compared in full: two prompts that differ anywhere must not
    /// share a context. (A hashed key would make this a silent wrong answer.)
    #[test_case]
    fn prefixes_differing_by_one_byte_are_distinct() {
        let mut s: PrefixStore<u32> = PrefixStore::new(1000);
        s.insert("You are a helpful agent.".to_string(), 1, 10);
        s.insert("You are a helpful agent!".to_string(), 2, 10);
        assert_eq!(s.get("You are a helpful agent.").copied(), Some(1));
        assert_eq!(s.get("You are a helpful agent!").copied(), Some(2));
        assert_eq!(s.len(), 2);
    }

    /// Over budget, the least-recently-*used* entry goes — not the oldest
    /// inserted. `a` is refreshed by a lookup, so `b` is the victim.
    #[test_case]
    fn eviction_takes_the_least_recently_used_entry() {
        let mut s: PrefixStore<u32> = PrefixStore::new(250);
        s.insert("a".to_string(), 1, 100);
        s.insert("b".to_string(), 2, 100);
        assert_eq!(s.get("a").copied(), Some(1)); // refresh a
        s.insert("c".to_string(), 3, 100); // must evict one
        assert_eq!(s.len(), 2);
        assert_eq!(s.get("a").copied(), Some(1), "a was used most recently");
        assert_eq!(s.get("c").copied(), Some(3));
        assert!(s.get("b").is_none(), "b was least recently used");
        assert!(s.bytes() <= 250, "budget exceeded: {}", s.bytes());
    }

    /// A payload bigger than the whole budget is declined rather than evicting
    /// everything to hold something that cannot be kept.
    #[test_case]
    fn an_oversized_prefix_is_declined_and_leaves_the_store_intact() {
        let mut s: PrefixStore<u32> = PrefixStore::new(200);
        s.insert("keep".to_string(), 1, 150);
        s.insert("huge".to_string(), 2, 5000);
        assert_eq!(s.len(), 1);
        assert_eq!(s.get("keep").copied(), Some(1));
        assert!(s.get("huge").is_none());
        assert_eq!(s.bytes(), 150);
    }

    /// Re-inserting the same key must release the old payload's bytes. Without
    /// that, a `serve_loop` re-storing one prefix would inflate the accounting
    /// every request until the store evicted itself empty.
    #[test_case]
    fn reinserting_a_key_replaces_it_without_leaking_budget() {
        let mut s: PrefixStore<u32> = PrefixStore::new(300);
        for i in 0..20u32 {
            s.insert("sys".to_string(), i, 100);
        }
        assert_eq!(s.len(), 1);
        assert_eq!(s.bytes(), 100, "byte accounting drifted across re-inserts");
        assert_eq!(s.get("sys").copied(), Some(19));
    }

    /// `accepts` must agree with what `insert` actually does, so a caller can
    /// skip building a payload it cannot store. Getting this wrong meant a 27B
    /// cloned a 353 MiB KV snapshot and then threw it away.
    #[test_case]
    fn accepts_predicts_insert_and_insert_reports_the_truth() {
        let mut s: PrefixStore<u32> = PrefixStore::new(200);
        assert!(s.accepts(200), "a payload exactly at budget fits");
        assert!(!s.accepts(201), "one byte over budget does not");
        assert!(s.insert("fits".to_string(), 1, 200), "insert must report success");
        // Over budget: declined, reported as such, store untouched.
        assert!(!s.accepts(5000));
        assert!(!s.insert("huge".to_string(), 2, 5000), "insert must report the decline");
        assert_eq!(s.len(), 1);
        assert_eq!(s.get("fits").copied(), Some(1));
    }

    #[test_case]
    fn a_zero_budget_store_caches_nothing_and_still_answers() {
        let mut s: PrefixStore<u32> = PrefixStore::new(0);
        s.insert("sys".to_string(), 1, 1);
        assert!(s.is_empty());
        assert!(s.get("sys").is_none());
    }
}
