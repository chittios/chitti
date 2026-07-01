//! The Synapse **constraint grammar** (`CHITTI_OS_HANDOFF.md` Phase 4): the
//! GBNF-style grammar, *generated from the registry*, that forces model
//! output into exactly the set of registered, well-formed tool calls. This
//! is the enforcement point of the determinism boundary (Part 2): model
//! output is an untrusted plan, and nothing below this line will act on a
//! call that this grammar has not accepted.
//!
//! The tool-call wire shape is canonical MCP-flavoured JSON, with no
//! insignificant whitespace so the language is unambiguous:
//!
//! ```text
//! {"name":"<registered name>","arguments":{"<k1>":<v1>,"<k2>":<v2>}}
//! ```
//!
//! arguments appear in the registry's declared order; string values are
//! `"..."` (with `\` escapes) and unsigned integers are bare digits. A
//! primitive with no parameters takes `"arguments":{}`.
//!
//! The grammar is realised as a hand-written recursive-descent parser that
//! is **prefix-closed**: it distinguishes a *complete* valid call, a valid
//! but *incomplete* prefix of some completable call, and an *invalid* input
//! no continuation could rescue. That three-way answer is what lets the same
//! grammar serve both roles the phase needs:
//!
//! * `parse` (accept only *complete*) is the executor's front door -- a
//!   malformed call is rejected here and never reaches a primitive.
//! * the [`ConstrainedDecoder`] adapter (accept *complete* or *prefix*) plugs
//!   into `sampler::Grammar`, masking any next token whose bytes would leave
//!   the viable-prefix set -- so a constrained model can only ever emit a
//!   well-formed call in the first place.

use super::registry::{self, ArgType, PrimitiveSpec};
use crate::cap::PrimitiveId;
use crate::cortex::sampler::Grammar;
use alloc::string::String;
use alloc::vec::Vec;

/// A value parsed for one primitive argument, already typed per the schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArgValue {
    Str(String),
    Uint(u64),
}

/// A fully validated tool call: a registered primitive id plus its argument
/// values in the registry's declared order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Call {
    pub id: PrimitiveId,
    pub args: Vec<ArgValue>,
}

/// Why a candidate string is not a complete, valid tool call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrammarError {
    /// The input violates the grammar and no continuation could fix it.
    Malformed,
    /// The input is a valid *prefix* but stops short of a complete call.
    Incomplete,
}

/// Internal control-flow signal while parsing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fail {
    /// Ran out of input on an otherwise-valid path (a viable prefix).
    Partial,
    /// A byte violated the grammar; unrecoverable.
    Invalid,
}

type PResult<T> = Result<T, Fail>;

/// Byte cursor over the candidate input. Every "need a byte but none left"
/// turns into `Fail::Partial`, which is exactly what makes the parser
/// prefix-closed.
struct Cur<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cur<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, i: 0 }
    }

    fn eof(&self) -> bool {
        self.i >= self.b.len()
    }

    fn peek(&self) -> PResult<u8> {
        self.b.get(self.i).copied().ok_or(Fail::Partial)
    }

    fn bump(&mut self) -> PResult<u8> {
        let c = self.peek()?;
        self.i += 1;
        Ok(c)
    }

    /// Match an exact literal. Truncation mid-literal is `Partial` (it could
    /// still be completed); a mismatched byte is `Invalid`.
    fn lit(&mut self, s: &[u8]) -> PResult<()> {
        for &want in s {
            if self.peek()? != want {
                return Err(Fail::Invalid);
            }
            self.i += 1;
        }
        Ok(())
    }

    /// Parse `"<registered name>"`, returning the primitive id. While the
    /// name is still being read, every accepted byte must keep the
    /// accumulated string a prefix of some registered name, so an impossible
    /// name is rejected at its first divergent byte.
    fn name(&mut self) -> PResult<PrimitiveId> {
        self.lit(b"\"")?;
        let start = self.i;
        loop {
            let c = self.peek()?;
            if c == b'"' {
                let name = core::str::from_utf8(&self.b[start..self.i]).map_err(|_| Fail::Invalid)?;
                let spec = registry::by_name(name).ok_or(Fail::Invalid)?;
                self.i += 1; // closing quote
                return Ok(spec.id);
            }
            // Validate the prefix *including* this byte before accepting it.
            let so_far = core::str::from_utf8(&self.b[start..=self.i]).map_err(|_| Fail::Invalid)?;
            if !registry::is_name_prefix(so_far) {
                return Err(Fail::Invalid);
            }
            self.i += 1;
        }
    }

    /// Parse a JSON-ish string value: `"..."` with `\` escaping the next
    /// byte. Value strings are unconstrained (unlike primitive names).
    fn string(&mut self) -> PResult<String> {
        self.lit(b"\"")?;
        let mut buf: Vec<u8> = Vec::new();
        loop {
            match self.bump()? {
                b'"' => return String::from_utf8(buf).map_err(|_| Fail::Invalid),
                b'\\' => {
                    let esc = self.bump()?;
                    let byte = match esc {
                        b'"' => b'"',
                        b'\\' => b'\\',
                        b'/' => b'/',
                        b'n' => b'\n',
                        b't' => b'\t',
                        b'r' => b'\r',
                        _ => return Err(Fail::Invalid),
                    };
                    buf.push(byte);
                }
                // Unescaped control characters are not allowed in a string.
                c if c < 0x20 => return Err(Fail::Invalid),
                c => buf.push(c),
            }
        }
    }

    /// Parse a bare unsigned integer (one or more digits). If input ends
    /// mid-number this returns the digits seen so far as `Ok`; the enclosing
    /// grammar then demands a `,`/`}` terminator and reports `Partial`, so a
    /// truncated number is still treated as a viable prefix.
    fn uint(&mut self) -> PResult<u64> {
        if !self.peek()?.is_ascii_digit() {
            return Err(Fail::Invalid);
        }
        let mut val: u64 = 0;
        while let Ok(d) = self.peek() {
            if !d.is_ascii_digit() {
                break;
            }
            val = val.saturating_mul(10).saturating_add((d - b'0') as u64);
            self.i += 1;
        }
        Ok(val)
    }

    /// Parse the `{...}` argument object for `spec`, enforcing the exact key
    /// names, canonical order, and value types the schema declares.
    fn args(&mut self, spec: &PrimitiveSpec) -> PResult<Vec<ArgValue>> {
        self.lit(b"{")?;
        let mut out = Vec::with_capacity(spec.params.len());
        for (idx, p) in spec.params.iter().enumerate() {
            if idx > 0 {
                self.lit(b",")?;
            }
            self.lit(b"\"")?;
            self.lit(p.key.as_bytes())?;
            self.lit(b"\":")?;
            let v = match p.ty {
                ArgType::Str => ArgValue::Str(self.string()?),
                ArgType::Uint => ArgValue::Uint(self.uint()?),
            };
            out.push(v);
        }
        self.lit(b"}")?;
        Ok(out)
    }
}

/// Drive the full grammar over `bytes`. `Ok` only for a *complete* valid
/// call with no trailing bytes; `Err(Partial)` for a viable prefix;
/// `Err(Invalid)` otherwise.
fn run(bytes: &[u8]) -> PResult<Call> {
    let mut c = Cur::new(bytes);
    c.lit(b"{\"name\":")?;
    let id = c.name()?;
    c.lit(b",\"arguments\":")?;
    // `name()` only returns ids that resolve, but be defensive.
    let spec = registry::by_id(id).ok_or(Fail::Invalid)?;
    let args = c.args(spec)?;
    c.lit(b"}")?;
    if !c.eof() {
        return Err(Fail::Invalid); // trailing garbage after a complete call
    }
    Ok(Call { id, args })
}

/// Validate a *complete* tool call. This is the executor's front door: a
/// malformed (or merely truncated) call is rejected here and never runs.
pub fn parse(input: &str) -> Result<Call, GrammarError> {
    match run(input.as_bytes()) {
        Ok(call) => Ok(call),
        Err(Fail::Partial) => Err(GrammarError::Incomplete),
        Err(Fail::Invalid) => Err(GrammarError::Malformed),
    }
}

/// Whether `bytes` is a *viable prefix* of some completable tool call
/// (either already complete, or extendable into one). This is the predicate
/// the constrained decoder masks tokens against.
pub fn accepts_prefix(bytes: &[u8]) -> bool {
    !matches!(run(bytes), Err(Fail::Invalid))
}

/// A grammar-constrained decoding adapter over `sampler::Grammar`. It holds
/// the bytes accepted so far and, for each candidate token, asks whether
/// appending that token's bytes would keep the output a viable prefix of a
/// tool call. `detok(token, &mut buf)` appends a token's bytes to `buf` --
/// supplied by the caller so this stays independent of any specific
/// tokenizer (tests use a byte-per-token vocabulary).
pub struct ConstrainedDecoder<F: Fn(usize, &mut Vec<u8>)> {
    accepted: Vec<u8>,
    detok: F,
}

impl<F: Fn(usize, &mut Vec<u8>)> ConstrainedDecoder<F> {
    pub fn new(detok: F) -> Self {
        Self { accepted: Vec::new(), detok }
    }

    /// The bytes accepted so far.
    pub fn bytes(&self) -> &[u8] {
        &self.accepted
    }

    /// Parse whatever has been accepted so far as a complete call.
    pub fn parse_accepted(&self) -> Result<Call, GrammarError> {
        match run(&self.accepted) {
            Ok(call) => Ok(call),
            Err(Fail::Partial) => Err(GrammarError::Incomplete),
            Err(Fail::Invalid) => Err(GrammarError::Malformed),
        }
    }

    /// Whether the accepted bytes already form a complete, valid call (a
    /// point at which decoding may stop).
    pub fn is_complete(&self) -> bool {
        matches!(run(&self.accepted), Ok(_))
    }
}

impl<F: Fn(usize, &mut Vec<u8>)> Grammar for ConstrainedDecoder<F> {
    fn allows(&self, token: usize) -> bool {
        let mut candidate = self.accepted.clone();
        (self.detok)(token, &mut candidate);
        accepts_prefix(&candidate)
    }

    fn accept(&mut self, token: usize) {
        (self.detok)(token, &mut self.accepted);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synapse::registry;

    #[test_case]
    fn parses_a_well_formed_call_with_two_string_args() {
        let call = parse(r#"{"name":"mem_fs_write","arguments":{"path":"notes","text":"hello"}}"#).unwrap();
        assert_eq!(call.id, registry::MEM_FS_WRITE);
        assert_eq!(call.args, alloc::vec![ArgValue::Str("notes".into()), ArgValue::Str("hello".into())]);
    }

    #[test_case]
    fn parses_a_no_arg_call() {
        let call = parse(r#"{"name":"list","arguments":{}}"#).unwrap();
        assert_eq!(call.id, registry::LIST);
        assert!(call.args.is_empty());
    }

    #[test_case]
    fn parses_a_uint_arg() {
        let call = parse(r#"{"name":"sleep","arguments":{"ticks":42}}"#).unwrap();
        assert_eq!(call.args, alloc::vec![ArgValue::Uint(42)]);
    }

    #[test_case]
    fn honours_escapes_in_strings() {
        let call = parse(r#"{"name":"console_write","arguments":{"text":"a\"b\\c\nd"}}"#).unwrap();
        assert_eq!(call.args, alloc::vec![ArgValue::Str("a\"b\\c\nd".into())]);
    }

    #[test_case]
    fn rejects_unregistered_primitive() {
        // A name that diverges from every registered name is Malformed, not
        // merely Incomplete.
        assert_eq!(parse(r#"{"name":"rm_rf","arguments":{}}"#), Err(GrammarError::Malformed));
    }

    #[test_case]
    fn rejects_wrong_key_wrong_type_and_extra_data() {
        // Wrong key.
        assert_eq!(
            parse(r#"{"name":"mem_fs_read","arguments":{"file":"x"}}"#),
            Err(GrammarError::Malformed)
        );
        // Wrong value type (string where a uint is required).
        assert_eq!(
            parse(r#"{"name":"sleep","arguments":{"ticks":"soon"}}"#),
            Err(GrammarError::Malformed)
        );
        // Trailing garbage after an otherwise-complete call.
        assert_eq!(
            parse(r#"{"name":"list","arguments":{}}JUNK"#),
            Err(GrammarError::Malformed)
        );
        // Missing a required argument.
        assert_eq!(
            parse(r#"{"name":"mem_fs_write","arguments":{"path":"a"}}"#),
            Err(GrammarError::Malformed)
        );
    }

    #[test_case]
    fn truncated_call_is_incomplete_not_malformed() {
        for prefix in [
            r#"{"na"#,
            r#"{"name":"mem_fs_wr"#,
            r#"{"name":"mem_fs_write","arguments":{"path":"no"#,
            r#"{"name":"sleep","arguments":{"ticks":4"#,
            r#"{"name":"list","arguments":{"#,
        ] {
            assert_eq!(parse(prefix), Err(GrammarError::Incomplete), "prefix {prefix:?} should be incomplete");
            assert!(accepts_prefix(prefix.as_bytes()), "prefix {prefix:?} should be a viable prefix");
        }
    }

    #[test_case]
    fn an_impossible_prefix_is_not_accepted() {
        // '(' can never begin a tool call; a bad name byte diverges early.
        assert!(!accepts_prefix(b"("));
        assert!(!accepts_prefix(br#"{"name":"zzz"#));
    }

    /// The generative tie to `sampler.rs`: constrained decoding over a
    /// byte-per-token vocabulary can only ever walk to a well-formed call.
    /// At each step the sampler is handed logits that favour a *disallowed*
    /// byte, yet the grammar mask forces a valid continuation.
    #[test_case]
    fn constrained_decoding_only_emits_well_formed_calls() {
        use crate::cortex::sampler::{self, Rng};

        // Every position of this call is structurally constrained (no
        // free-form string value), so the biasing byte `~` is never a valid
        // continuation -- letting us prove the grammar *mask*, not the
        // logits, drives the shape.
        let target = r#"{"name":"sleep","arguments":{"ticks":42}}"#;
        let mut dec = ConstrainedDecoder::new(|tok: usize, buf: &mut Vec<u8>| buf.push(tok as u8));
        let mut rng = Rng::new(0xABCDEF);

        for step in 0..target.len() {
            let mut logits = alloc::vec![0.0f32; 256];
            // Peak on a byte the grammar must reject at every position here;
            // the intended byte gets a far lower logit yet is still forced
            // once `~` is masked to -inf.
            logits[b'~' as usize] = 100.0;
            logits[target.as_bytes()[step] as usize] = 1.0;
            let tok = sampler::sample(&mut logits, 0.0, &mut rng, Some(&dec));
            assert_ne!(tok, b'~' as usize, "grammar failed to mask the invalid byte");
            assert!(dec.allows(tok), "sampler returned a grammar-disallowed token");
            dec.accept(tok);
            // Every intermediate prefix stays viable.
            assert!(accepts_prefix(dec.bytes()));
        }
        assert!(dec.is_complete(), "decoding did not reach a complete call");
        let call = dec.parse_accepted().unwrap();
        assert_eq!(call.id, registry::SLEEP);
        assert_eq!(call.args, alloc::vec![ArgValue::Uint(42)]);
    }
}
