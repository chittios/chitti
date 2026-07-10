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

## Fix 4 — labelled statements + labelled break/continue

- **Repro:** lodash `else n:{switch(f){…break n…}}` → `expected …` parse error
  at the `switch`. Labelled statements were stubbed out (commented in the
  grammar; AST comment "Label Statement not supported").
- **Fix:** grammar — enabled `labelled_statement` (+`__yield`/`__return`/
  `__yield_return`) *before* `expression_statement`, and the labelled
  `break`/`continue` alternatives. AST — new `LabelledStatement { label, body }`,
  `label: Option<String>` on Break/Continue. Interpreter (`statement.rs`) —
  `EvalContext.pending_labels`; `execute_labelled_statement` pushes the label,
  each loop takes its labels at entry and only swallows a `break`/`continue`
  whose target is `None` or one of its labels (else propagates outward); the
  switch propagates labelled breaks; a labelled block converts an escaping
  `break L` to a normal completion. Host-only JIT compilers got a body-compiling
  arm.
- **Corpus delta:** +1 net (test262 labelled-statement tests); **lodash now
  parses**.

## Fix 5 — `Function.prototype.call` / `apply` / `bind`

- **Repro:** `fn.call(o, …)` → `TypeError("call is not a function")` (jQuery and
  lodash use these pervasively for internal dispatch).
- **Fix (`expression.rs`):** the unified member-call dispatch intercepts
  `call`/`apply`/`bind` on any callable receiver. `call` forwards args after the
  `this` arg; `apply` spreads the array arg; `bind` returns a `__bound_target__`
  marker object (callable via the `__simple_function__` marker) that
  `call_function_object` forwards with the fixed `this` + prepended partial
  args.
- **Corpus delta:** 0 regressions; both libraries now execute deep into init.

## Fix 6 — modern-JS syntax pass (ES2018–2022), driven by the vercel.com corpus

Downloaded all 41 `vercel.com` Next.js/turbopack chunks to
`fixtures/vercel/` and ran them through `--raw`. Each fix below unblocked the
next; fully-parsing chunks went **4 → 25 / 41** (96% of runnable), test262
**889 → 906**, zero panics/regressions:

- **Object binding-pattern renaming** `{ x: a } = o` — `binding_property` tried
  `single_name_binding` first, greedily matching `x` (PEG no-backtrack), so the
  `:` rename form was unreachable. Reordered. **(+16 test262.)**
- **Exponentiation `**`** (right-assoc, above `*`) — grammar
  `exponentiation_expression` + `BinaryOperator::Exponent` + interpreter.
- **Nullish coalescing `??`** — `coalesce_expression` between `||` and `?:`;
  `LogicalOperator::Coalesce`.
- **Logical/exponent assignment `**= &&= ||= ??=`** — added to
  `assignment_operator` + `AssignmentOperator` + both assign paths (logical
  forms short-circuit on the current value).
- **Optional chaining `?.` / `?.[]` / `?.()`** — new `ExpressionType::
  OptionalChain { object, access }` (a single variant, not a flag on every
  member/call node) + `optional_member`/`optional_index`/`optional_call`
  grammar; `?.` before a digit stays a ternary (`a ? .5 : b`). Per-step guard:
  nullish base → `undefined`.
- **Optional catch binding** `catch { … }` — `catch_parameter` made optional
  in grammar; `CatchClauseData.param: Option<…>`; `handle_catch` skips the bind.
- **do-while ASI** — `do … while(c)` no longer requires a trailing `;`.
- **Object rest** `{ a, ...rest } = o` — `binding_rest_property` grammar;
  `ObjectPattern.rest`; interpreter collects the remaining own enumerable keys.
- **Object spread** `{ ...a, k: v }` — `spread_property` grammar +
  `PropertyKind::Spread`; the object-literal evaluator merges the spread
  source's own enumerable props (string spread → index entries). **This was the
  biggest single unlock: vercel 8 → 25.**

Host-only JIT compilers got best-effort/fallback arms for every new node
(the tree-walking interpreter is authoritative; the kernel never uses the JIT).

## Fix 7 — ES2022 class fields/private/static-blocks, async methods, template eval, `for await`, rest patterns

Closed the entire remaining vercel.com parse gap — **all 41 chunks now parse
(skip=0)**; test262 **906 → 909**, zero regressions.

- **Class fields** (instance + static) — grammar `field_definition`,
  `class_element_name`, `class_static_block`; AST `ClassFieldData` /
  `StaticBlockData` on `ClassBodyData`. Instance-field initializers are carried
  on the constructor `SimpleFunctionObject` (`instance_fields`) and applied to
  `this` at the top of the constructor call (a synthesized field-only default
  constructor has a null body + `__has_field_init__` so `call_function_object`
  routes it through `call_with_this` while `invoke_constructor` still performs
  the implicit `super(...)`). Static fields/blocks run at class-definition time
  with `this` = constructor. **Caveat:** for a *derived* class with an explicit
  constructor, fields init before the body's `super(...)` (this engine creates
  `this` up front) — a parent ctor could overwrite a same-named field; rare,
  documented.
- **Private names** `#x` — a pragmatic non-enumerable string key `"#x"`
  (`private_name` grammar; reachable via `literal_property_name` for methods and
  `class_element_name` for fields; `.#x` member access). No per-instance brand.
- **Static init blocks** `static { … }` — run in source order at definition.
- **Async / async-generator methods** `async m(){}`, `async *g(){}` (class and
  object literals) — atomic `async_kw` keeps `asyncFoo(){}` a plain method;
  `is_async`/`generator` flow to `FunctionData` (Promise-wrapped return).
- **Template-literal substitution eval** was a runtime stub — now interleaves
  cooked quasis with string-coerced expression values. Also made
  `template_literal` a *normal* (non-atomic) rule so `${ 1 }`, `${a + b}`, and
  nested templates `` `a${ `b` }c` `` parse (`s` alone matches only line
  terminators, so the old compound-atomic form dropped every space).
- **`for await (x of y)`** — optional `await` on every for-of variant (maps to a
  normal for-of; ChittiOS async is synchronous-settlement).
- **Rest-element binding patterns** `(...[a,b])` / `(...{x})` — `binding_rest_element`
  accepts a `binding_pattern` (the AST `RestElement.argument` was already a
  pattern; the interpreter already recurses via `bind_pattern`).
- **`globalThis` in the browser** now aliases the stable `window`
  (`js_just.rs`) — the engine default returns a fresh empty object, breaking the
  `globalThis.X = globalThis.X || []` turbopack pattern.

## Final state

files=1104 **pass=909** skip=64 panics=0. jQuery 3.7.1 slim + lodash-core
parse+run; **41/41 vercel.com chunks parse (skip=0).** The only two `--raw`
runtime errors left are bare-harness environment gaps (`self is not defined`,
webpack module-registry `.call`) — both chunks parse and run in the browser tier
(which provides `window`/`self`/`globalThis`/`document`).
