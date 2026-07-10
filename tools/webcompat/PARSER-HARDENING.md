# ChittiOS `just` parser hardening log

Hardening the vendored `just` ES6 engine (`third_party/just-ref`) so real-world
libraries parse. Acceptance corpus: jQuery 3.7.1 slim minified
(`fixtures/jquery-3.7.1.slim.min.js`), lodash-core 4.17.21
(`fixtures/lodash.core.min.js`). Regression gate: the test262 subset under
`tools/webcompat/test262/test` via `just_runner`.

**Baseline (before any change):** files=1104 pass=880 fail=124 skip=100 panics=0.

## Fix 1 — multi-line comments never parsed (PEG greediness)

- **Repro:** `/* hello */ var x = 1;` → `expected regular_expression_body`.
  Every form failed: leading, trailing (`var x = 1; /* t */`), and
  mid-expression (`var x = /* m */ 1;`). Line comments worked.
- **Root cause:** the grammar transcribed the spec's *CFG* for
  `MultiLineCommentChars` into PEG. The branch
  `"*" ~ post_asterisk_comment_chars?` greedily consumes the comment's final
  `*` (the trailing `?` matches empty and PEG never backtracks into a
  committed option), so the closing `*/` literal could never match. No
  `/* */` comment ever parsed — jQuery dies at byte 0 (its license header).
- **Fix (`js_grammar.pest`):** standard PEG comment idiom
  `multi_line_comment = @{ "/*" ~ (!"*/" ~ source_character)* ~ "*/" }`.
- **Corpus delta:** exposed 84 spurious passes (ES2025 regexp-modifiers
  negative parse tests that only "passed" because their trailing block
  comment broke the parse) → fixed legitimately in Fix 2. Net after fixes
  1+2+3: zero regressions.

## Fix 2 — ES2025 regexp-modifiers early errors

- **Repro:** `/(?i-i:a)/` (and `(?-Q:a)`, `(?ii-:a)`, `(?-:a)`, …) must be a
  parse-time SyntaxError; after Fix 1 these 84 negative test262 tests parsed
  without error.
- **Root cause:** the engine never validated `(?…` groups other than
  `(?<name>`; anything after `(?` that is not `:`/`=`/`!`/`<` fell through
  as pattern text.
- **Fix (`api.rs`):** `validate_regexp_modifiers` — invoked from the existing
  `validate_regexp_named_groups` scan when `(?` is followed by anything
  other than `:`/`=`/`!`/`<` (outside char classes, not escaped). Enforces:
  flags ∈ {i,m,s}, no duplicate within a list, no flag in both lists, dash
  form with both lists empty is an error. Ordinary regexes (`(?:`, `(?=`,
  `(?!`, `(?<…`) are untouched.
- **Corpus delta:** the 84 tests pass again, legitimately.

## Fix 3 — U+2028/U+2029 are line terminators

- **Repro:** `/<U+2028>/` must be a parse error (LineTerminator can't appear
  in a regex literal); 4 test262 tests
  (`regexp-{first,source}-char-no-{line,paragraph}-separator`) regressed
  after Fix 1 (their explanatory block comment previously broke the parse).
- **Root cause:** `line_terminator` only listed `\n` and `\r`; the spec's
  LineTerminator also includes U+2028 LINE SEPARATOR and U+2029 PARAGRAPH
  SEPARATOR.
- **Fix (`js_grammar.pest`):** added `"\u{2028}" | "\u{2029}"` to
  `line_terminator` and `line_terminator_sequence`.
- **Corpus after fixes 1–3:** files=1104 pass=888 fail=122 skip=94 panics=0 —
  zero regressions vs baseline, +8 net new passes.
