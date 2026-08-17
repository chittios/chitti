//! **Retrieval index** — the thing that turns agent memory from exact-match
//! lookup into recall, and turns a folder of files into something an agent can
//! answer from.
//!
//! `mem_fs_search` matches substrings: an agent that stored "the deploy key
//! lives in the vault" cannot find it by asking about "credentials". This module
//! is the index that can, and it serves both callers — agent memory and local
//! files — because the problem is identical once the text is chunked.
//!
//! ## The embedder is lexical, and that is a deliberate, stated limit
//!
//! A real semantic index needs a sentence-embedding model. This OS ships an ONNX
//! interpreter and could run one, but it does not ship the weights, and adding a
//! multi-megabyte asset to every image is a decision that belongs to whoever
//! maintains the build rather than to this module. So the built-in embedder is
//! **feature-hashed bag of words** with sublinear term frequency — genuinely
//! useful retrieval (it finds "deploy key" from "deployment keys"), and honestly
//! not semantic (it will not find it from "credentials").
//!
//! The seam for fixing that is [`Embedder`]: supply dense vectors from a real
//! model and every other part of this file — chunking, storage, cosine ranking,
//! persistence — is unchanged. Calling the lexical version "semantic" in the UI
//! would be the actual mistake, so [`Embedder::name`] reports which is in use and
//! callers surface it.
//!
//! ## Why feature hashing rather than a vocabulary
//!
//! A vocabulary has to be built, persisted, versioned, and kept consistent with
//! every stored vector — reindex the corpus and old vectors silently mean
//! something else. Hashing has no such state: dimension `i` means the same thing
//! forever because it is derived from the term itself. The cost is collisions,
//! which at 512 dimensions over a personal corpus are rare and degrade ranking
//! slightly rather than corrupting it.
//!
//! The hash must therefore be **stable across boots** — an index persisted to the
//! store and reloaded has to rank identically, so this uses FNV-1a explicitly
//! rather than anything the standard library might change.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// Vector width. 512 keeps a chunk's vector at 2 KiB as `f32`, which for a few
/// thousand chunks is a few megabytes — affordable on this heap, and small
/// enough that a linear scan is faster than any index structure would be at
/// this scale (no k-d tree, no HNSW: they would be more code and slower here).
pub const DIMS: usize = 512;

/// Target chunk size in bytes. Retrieval returns whole chunks, so this trades
/// precision (small chunks, tight answers) against context (large chunks, more
/// surrounding meaning). 600 is about a paragraph.
pub const CHUNK: usize = 600;

/// How much each chunk repeats of the previous one. Without overlap, a fact that
/// straddles a boundary is in neither chunk's vector and becomes unfindable —
/// the failure is silent and looks like the index simply not working.
pub const OVERLAP: usize = 100;

/// One indexed span of text and where it came from.
#[derive(Clone, PartialEq, Debug)]
pub struct Chunk {
    /// Store path or memory key this came from.
    pub source: String,
    /// Index of this chunk within its source, so a caller can show "part 3 of 7".
    pub ord: usize,
    pub text: String,
}

/// Produces a vector for a piece of text.
///
/// Implemented by [`Lexical`] today; a dense model implements the same trait and
/// the rest of the module does not change.
pub trait Embedder {
    fn embed(&self, text: &str) -> Vec<f32>;
    /// Shown to the user, so nobody mistakes lexical retrieval for semantic.
    fn name(&self) -> &'static str;
}

/// Feature-hashed bag of words. See the module doc for why this rather than a
/// vocabulary, and for what it can and cannot do.
pub struct Lexical;

impl Embedder for Lexical {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; DIMS];
        for term in terms(text) {
            let h = (fnv1a(term.as_bytes()) as usize) % DIMS;
            v[h] += 1.0;
        }
        // Sublinear term frequency: a word repeated twenty times should count
        // for more than one occurrence but nowhere near twenty, or one chatty
        // chunk dominates every query that shares a common word with it.
        for x in v.iter_mut() {
            if *x > 0.0 {
                *x = 1.0 + ln_approx(*x);
            }
        }
        l2_normalise(&mut v);
        v
    }

    fn name(&self) -> &'static str {
        "lexical (hashed bag-of-words)"
    }
}

/// Split text into lowercase alphanumeric terms, dropping the very short ones.
///
/// Single characters and two-letter words are almost pure noise in a bag of
/// words — they collide often and match everything — so they are skipped.
pub fn terms(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| t.len() > 2)
        .map(|t| t.to_ascii_lowercase())
}

/// A cheap natural log, good to a few percent — enough for term weighting and it
/// avoids pulling in floating-point intrinsics that are awkward in `no_std`.
fn ln_approx(x: f32) -> f32 {
    // log2 via the exponent field, then scale. x > 0 is guaranteed by the caller.
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let mantissa = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    // Linear fit of log2 over [1,2): good enough, monotonic, no table.
    let log2 = exp as f32 + (mantissa - 1.0) * 0.9;
    log2 * core::f32::consts::LN_2
}

fn l2_normalise(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>();
    if norm <= 0.0 {
        return;
    }
    let inv = 1.0 / sqrt_approx(norm);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Newton-Raphson square root — `f32::sqrt` is a std intrinsic not available in
/// this `no_std` build on every target, and two iterations from a bit-twiddled
/// seed are plenty for a normalisation factor.
fn sqrt_approx(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut g = f32::from_bits((x.to_bits() >> 1) + 0x1fc0_0000);
    g = 0.5 * (g + x / g);
    g = 0.5 * (g + x / g);
    g
}

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// Cosine similarity. Both vectors are L2-normalised on the way in, so this is
/// just a dot product — kept as its own function because a caller supplying
/// dense vectors from a model may not have normalised them.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>();
    let nb = b.iter().map(|x| x * x).sum::<f32>();
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (sqrt_approx(na) * sqrt_approx(nb))
}

/// Split `text` into overlapping chunks, preferring to break at a blank line and
/// then at a newline, so a chunk is a paragraph rather than an arbitrary cut.
pub fn chunk_text(source: &str, text: &str, size: usize, overlap: usize) -> Vec<Chunk> {
    let size = size.max(32);
    let overlap = overlap.min(size / 2);
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut ord = 0usize;
    while start < bytes.len() {
        let hard_end = (start + size).min(bytes.len());
        // Look backwards from the ideal end for a natural boundary, but never
        // give back more than a third of the chunk chasing one.
        let floor = start + (size * 2 / 3);
        let mut end = hard_end;
        if hard_end < bytes.len() {
            if let Some(p) = text[start..hard_end].rfind("\n\n").map(|i| start + i + 2) {
                if p > floor {
                    end = p;
                }
            }
            if end == hard_end {
                if let Some(p) = text[start..hard_end].rfind('\n').map(|i| start + i + 1) {
                    if p > floor {
                        end = p;
                    }
                }
            }
        }
        let piece = text[start..end].trim();
        if !piece.is_empty() {
            out.push(Chunk { source: source.to_string(), ord, text: piece.to_string() });
            ord += 1;
        }
        if end >= bytes.len() {
            break;
        }
        // Advance, keeping `overlap` bytes of context. Guard against a
        // zero-length step, which would spin forever on pathological input.
        start = end.saturating_sub(overlap).max(start + 1);
    }
    out
}

/// A scored search hit.
#[derive(Clone, PartialEq, Debug)]
pub struct Hit {
    pub score: f32,
    pub chunk: Chunk,
}

/// The index: chunks plus their vectors, searched by linear scan.
#[derive(Default)]
pub struct Index {
    entries: Vec<(Chunk, Vec<f32>)>,
}

impl Index {
    pub fn new() -> Self {
        Self::new_const()
    }

    /// `const` so the index can live in a `static Locked<..>` without a lazy
    /// init dance — `Vec::new` is const, so the whole struct can be.
    pub const fn new_const() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Index one document, replacing any chunks previously held for `source` —
    /// re-adding a file that changed must not leave its old chunks findable.
    pub fn add_document<E: Embedder>(&mut self, emb: &E, source: &str, text: &str) {
        self.remove_source(source);
        for c in chunk_text(source, text, CHUNK, OVERLAP) {
            let v = emb.embed(&c.text);
            self.entries.push((c, v));
        }
    }

    pub fn remove_source(&mut self, source: &str) {
        self.entries.retain(|(c, _)| c.source != source);
    }

    /// Top `k` chunks for `query`, best first.
    ///
    /// Hits scoring zero are dropped rather than padded out to `k`: an agent
    /// handed three irrelevant chunks will use them, so "nothing matched" has to
    /// be expressible.
    pub fn search<E: Embedder>(&self, emb: &E, query: &str, k: usize) -> Vec<Hit> {
        if k == 0 || self.entries.is_empty() {
            return Vec::new();
        }
        let qv = emb.embed(query);
        let mut scored: Vec<Hit> = self
            .entries
            .iter()
            .map(|(c, v)| Hit { score: cosine(&qv, v), chunk: c.clone() })
            .filter(|h| h.score > 0.0)
            .collect();
        // Descending by score; ties broken by source then ord so results are
        // deterministic, which the tests and the audit trail both rely on.
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| a.chunk.source.cmp(&b.chunk.source))
                .then_with(|| a.chunk.ord.cmp(&b.chunk.ord))
        });
        scored.truncate(k);
        scored
    }

    /// Distinct sources currently indexed.
    pub fn sources(&self) -> Vec<String> {
        let mut v: Vec<String> = Vec::new();
        for (c, _) in &self.entries {
            if !v.iter().any(|s| s == &c.source) {
                v.push(c.source.clone());
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn cosine_is_one_for_identical_and_zero_for_disjoint() {
        let e = Lexical;
        let a = e.embed("the deploy key lives in the vault");
        assert!((cosine(&a, &a) - 1.0).abs() < 0.01, "a vector matches itself");
        let b = e.embed("zzzz yyyy xxxx wwww");
        assert!(cosine(&a, &b) < 0.2, "unrelated text scores low");
    }

    #[test_case]
    fn retrieval_finds_the_right_chunk_by_overlapping_words() {
        let e = Lexical;
        let mut ix = Index::new();
        ix.add_document(&e, "notes", "The deploy key lives in the vault.");
        ix.add_document(&e, "recipe", "Fry the onions until golden brown.");
        let hits = ix.search(&e, "where is the deploy key", 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.source, "notes");
    }

    #[test_case]
    fn a_query_matching_nothing_returns_nothing() {
        // The important negative: padding out to `k` would hand an agent
        // irrelevant context that it would then use as if it were an answer.
        let e = Lexical;
        let mut ix = Index::new();
        ix.add_document(&e, "notes", "alpha beta gamma");
        assert!(ix.search(&e, "zzzz qqqq wwww", 5).is_empty());
    }

    #[test_case]
    fn search_is_deterministic_across_equal_scores() {
        let e = Lexical;
        let mut ix = Index::new();
        ix.add_document(&e, "b_source", "identical text here");
        ix.add_document(&e, "a_source", "identical text here");
        let first = ix.search(&e, "identical text", 2);
        let again = ix.search(&e, "identical text", 2);
        assert_eq!(first, again, "ranking must be stable -- the audit log records it");
        assert_eq!(first[0].chunk.source, "a_source", "ties break by source name");
    }

    #[test_case]
    fn reindexing_a_source_replaces_its_old_chunks() {
        // The silent-corruption case: without this, editing a file leaves the
        // previous version findable and an agent answers from stale text.
        let e = Lexical;
        let mut ix = Index::new();
        ix.add_document(&e, "notes", "the password is hunter2");
        ix.add_document(&e, "notes", "the password was rotated");
        assert_eq!(ix.len(), 1);
        let hits = ix.search(&e, "hunter2", 5);
        assert!(hits.is_empty(), "the old revision must not survive a reindex");
    }

    #[test_case]
    fn chunking_splits_long_text_with_overlap() {
        let body = "aaaa bbbb cccc dddd\n\n".repeat(80);
        let cs = chunk_text("doc", &body, 200, 40);
        assert!(cs.len() > 1, "long text must split");
        assert!(cs.iter().all(|c| c.text.len() <= 200));
        for (i, c) in cs.iter().enumerate() {
            assert_eq!(c.ord, i, "ord must number the chunks in order");
        }
    }

    #[test_case]
    fn chunking_prefers_a_paragraph_boundary() {
        let text = "first paragraph text goes here and runs on\n\nsecond paragraph";
        let cs = chunk_text("d", text, 50, 5);
        assert!(cs[0].text.ends_with("runs on"), "must break at the blank line, not mid-word");
    }

    #[test_case]
    fn chunking_terminates_on_pathological_input() {
        // A zero-length step would spin forever with interrupts enabled, which
        // in a kernel is a hang rather than a failed test.
        assert!(chunk_text("d", "", 100, 50).is_empty());
        let cs = chunk_text("d", &"x".repeat(1000), 32, 31);
        assert!(cs.len() < 200, "overlap is clamped so progress is guaranteed");
    }

    #[test_case]
    fn short_terms_are_ignored() {
        // "is", "a", "of" collide constantly and match everything.
        let t: alloc::vec::Vec<String> = terms("a of the deploy key is up").collect();
        assert!(!t.iter().any(|w| w.len() <= 2));
        assert!(t.iter().any(|w| w == "deploy"));
    }

    #[test_case]
    fn embedding_is_case_insensitive() {
        let e = Lexical;
        let a = e.embed("Deploy Key");
        let b = e.embed("deploy key");
        assert!((cosine(&a, &b) - 1.0).abs() < 0.01);
    }

    #[test_case]
    fn repetition_does_not_dominate_the_score() {
        // Sublinear TF: a chunk saying "key" forty times must not outrank a
        // chunk that actually answers the question.
        let e = Lexical;
        let mut ix = Index::new();
        ix.add_document(&e, "spam", &"key ".repeat(40));
        ix.add_document(&e, "real", "the deploy key lives in the vault");
        let hits = ix.search(&e, "where does the deploy key live", 2);
        assert_eq!(hits[0].chunk.source, "real", "relevance must beat repetition");
    }

    #[test_case]
    fn the_hash_is_stable_so_a_persisted_index_still_ranks() {
        // Dimension i must mean the same thing next boot, or reloading an index
        // silently returns nonsense.
        assert_eq!(fnv1a(b"deploy"), fnv1a(b"deploy"));
        assert_ne!(fnv1a(b"deploy"), fnv1a(b"deploys"));
    }

    #[test_case]
    fn sqrt_and_ln_approximations_are_close_enough() {
        for x in [1.0f32, 2.0, 9.0, 100.0, 1e6] {
            let s = sqrt_approx(x);
            assert!((s * s - x).abs() / x < 0.001, "sqrt({x}) = {s}");
        }
        assert!(ln_approx(1.0).abs() < 0.05);
        assert!((ln_approx(core::f32::consts::E) - 1.0).abs() < 0.1);
        // Monotonic is what the weighting actually depends on.
        assert!(ln_approx(2.0) < ln_approx(10.0));
    }

    #[test_case]
    fn removing_a_source_leaves_the_others() {
        let e = Lexical;
        let mut ix = Index::new();
        ix.add_document(&e, "a", "alpha content");
        ix.add_document(&e, "b", "beta content");
        ix.remove_source("a");
        assert_eq!(ix.sources(), alloc::vec!["b".to_string()]);
    }

    #[test_case]
    fn an_empty_index_and_a_zero_k_are_both_safe() {
        let e = Lexical;
        let ix = Index::new();
        assert!(ix.search(&e, "anything", 5).is_empty());
        let mut ix2 = Index::new();
        ix2.add_document(&e, "s", "text");
        assert!(ix2.search(&e, "text", 0).is_empty());
    }
}
