//! **Public Suffix List** subset for cookie domain checks.
//!
//! Reference: Ladybird `LibURL/PublicSuffixData`, publicsuffix.org.
//! Full PSL is multi-MB; we ship a compact list of common multi-part suffixes
//! so `eTLD+1` cookie domain rules work for typical sites (`.co.uk`, `.com.au`, …).

/// Returns true if `label` (no leading dot) is a known public suffix.
pub fn is_public_suffix(label: &str) -> bool {
    let l = label.trim_start_matches('.').to_ascii_lowercase();
    if l.is_empty() {
        return false;
    }
    // Single-label ICANN-like TLDs always treated as public suffixes.
    if !l.contains('.') {
        return true;
    }
    PUBLIC_SUFFIXES.binary_search(&l.as_str()).is_ok()
}

/// eTLD+1 registrable domain for cookie Domain attribute validation.
/// Returns `None` if the host is an IP or cannot form a registrable domain.
pub fn registrable_domain(host: &str) -> Option<alloc::string::String> {
    let host = host.trim_start_matches('.').to_ascii_lowercase();
    if host.is_empty() || host.parse::<u8>().is_ok() {
        // crude IP reject: digits-only first label handled below
    }
    if host.chars().all(|c| c.is_ascii_digit() || c == '.') && host.contains('.') {
        // IPv4-ish: no cookie domain attribute matching
        if host.split('.').all(|p| p.parse::<u8>().is_ok()) {
            return None;
        }
    }
    let labels: alloc::vec::Vec<&str> = host.split('.').filter(|s| !s.is_empty()).collect();
    if labels.is_empty() {
        return None;
    }
    // Longest matching public suffix from the right (prefer co.uk over uk).
    let mut best_n = 0usize;
    for n in 1..=labels.len() {
        let suffix = labels[labels.len() - n..].join(".");
        if is_public_suffix(&suffix) {
            best_n = n;
        }
    }
    if best_n == 0 {
        if labels.len() >= 2 {
            return Some(labels[labels.len() - 2..].join("."));
        }
        return Some(host);
    }
    if labels.len() <= best_n {
        return None; // host is itself a public suffix
    }
    let start = labels.len() - best_n - 1;
    Some(labels[start..].join("."))
}

/// True if `cookie_domain` is an allowed Domain= for requests to `host`
/// (must be host or superdomain, and not a public suffix alone).
pub fn cookie_domain_ok(cookie_domain: &str, host: &str) -> bool {
    let cd = cookie_domain.trim_start_matches('.').to_ascii_lowercase();
    let h = host.trim_start_matches('.').to_ascii_lowercase();
    if cd.is_empty() || is_public_suffix(&cd) {
        return false;
    }
    if let Some(reg) = registrable_domain(&h) {
        // Cookie domain must be reg domain or subdomain of host, and cover reg.
        if !h.ends_with(&cd) && h != cd {
            return false;
        }
        // Domain attribute cannot be shorter than eTLD+1 in a way that escapes reg.
        if !cd.ends_with(&reg) && cd != reg {
            // allow exact host-only style
            return h == cd;
        }
        return h == cd || h.ends_with(&alloc::format!(".{cd}"));
    }
    h == cd || h.ends_with(&alloc::format!(".{cd}"))
}

/// Multi-label public suffixes (sorted for binary search). Not exhaustive.
const PUBLIC_SUFFIXES: &[&str] = &[
    "ac.uk",
    "blogspot.com",
    "co.jp",
    "co.kr",
    "co.nz",
    "co.uk",
    "co.za",
    "com.au",
    "com.br",
    "com.cn",
    "com.hk",
    "com.mx",
    "com.sg",
    "com.tw",
    "edu.au",
    "github.io",
    "gov.uk",
    "ne.jp",
    "net.au",
    "net.uk",
    "or.jp",
    "org.au",
    "org.uk",
];

// Keep sorted at compile time — verify in tests.

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn psl_sorted_and_etld() {
        for w in PUBLIC_SUFFIXES.windows(2) {
            assert!(w[0] < w[1], "PSL not sorted: {} >= {}", w[0], w[1]);
        }
        assert!(is_public_suffix("co.uk"));
        assert!(is_public_suffix("com"));
        assert!(!is_public_suffix("example.co.uk"));
        assert_eq!(
            registrable_domain("www.foo.co.uk").as_deref(),
            Some("foo.co.uk")
        );
        assert_eq!(
            registrable_domain("bar.example.com").as_deref(),
            Some("example.com")
        );
        assert!(!cookie_domain_ok("co.uk", "www.foo.co.uk"));
        assert!(cookie_domain_ok("foo.co.uk", "www.foo.co.uk"));
        assert!(cookie_domain_ok("example.com", "www.example.com"));
    }
}
