# The Determinism Boundary — paper draft

An arXiv-shaped design paper on **Synapse**, the capability ABI of ChittiOS:
what an OS's protection mechanism has to become when the unit of execution is
an agent rather than a program.

- [main.tex](main.tex) — the draft
- [refs.bib](refs.bib) — bibliography (**metadata needs a verification pass**;
  entries written from working knowledge, some flagged `VERIFY`)
- `make` — build `main.pdf` (`make manual` if you have no `latexmk`)

Intended categories: **cs.OS** (primary), cross-list **cs.CR**, **cs.AI**.

## Status

Sections 1–4 (introduction, execution model + threat model, mechanism,
implementation) are drafted against the code and are meant to be accurate as
written. Section 5 (evaluation) is **deliberately stubbed**: it states each
experiment's question, method, and hypothesis, and marks the numbers `TBD`.
Sections 6–8 (limitations, related work, conclusion) are drafted.

Every factual claim about the mechanism traces to source:

| Claim in the paper | Source |
| --- | --- |
| Five ordered gates, one audit record per path | [executor.rs:69-130](../kernel/src/synapse/executor.rs#L69-L130) |
| 26 primitives, 5 destructive, ids fixed at build time | [registry.rs](../kernel/src/synapse/registry.rs) |
| Prefix-closed three-valued grammar; decoder + front-door parse | [grammar.rs](../kernel/src/synapse/grammar.rs) |
| `Cap` is an index into the caller's own table | [cap/mod.rs:63-102](../kernel/src/cap/mod.rs#L63-L102) |
| Provenance lattice, join, `blocks_destructive` | [taint.rs](../kernel/src/security/taint.rs) |
| Justification computed from resident context taint | [dispatch.rs:67-91](../kernel/src/tools/dispatch.rs#L67-L91), [session.rs:113-122](../kernel/src/session/session.rs#L113-L122) |
| Identity-file writes treated as destructive | [executor.rs:99-107](../kernel/src/synapse/executor.rs#L99-L107) |
| Scope gate, deny-only-when-recorded | [cap/mod.rs:187-205](../kernel/src/cap/mod.rs#L187-L205) |
| Per-agent home sandbox floor | [skills/install.rs:145-165](../kernel/src/skills/install.rs#L145-L165) |
| Append-only audit, `Copy` entries, ktrace coalescing | [audit.rs](../kernel/src/synapse/audit.rs) |

## E1 (gate cost) — the harness

Built: [`kernel/src/synapse/bench.rs`](../kernel/src/synapse/bench.rs) +
`executor::gate_prefix` ([executor.rs](../kernel/src/synapse/executor.rs)),
driven by **`/bench synapse`** on the running kernel. E2E scenario
`synapse_bench` (in the always-run `os` group) asserts both the cost and, more
usefully, the *attribution* — each refusal row must name the gate that refused
it.

```sh
python3 tests/e2e/run.py --only synapse_bench -v     # ~2 min, no model needed
```

It measures through a gate-prefix path that runs the real predicates in the
real order but executes no primitive and writes no audit entry, against a
synthetic parked task with an explicitly granted table and scope ledger (so a
figure never depends on what the live session holds, and pricing a *denied*
call never means granting an agent a right).

**Take the numbers on an idle machine.** The first run was taken with a QEMU
guest and a TCG test suite competing for the host, which inflates everything.

Four methodology bugs the measurement runs exposed — all now fixed, all worth
remembering because each printed a plausible wrong number:

1. **Dead-code elimination.** The FNV row read `0 ns` over 16.7M iterations:
   the loop was gone. Everything timed now goes through
   `core::hint::black_box`.
2. **Non-monotonic prefixes.** Cumulative gate costs *decreased*
   (1190 → 954 → 809 ns), so gates 2 and 3 differenced to a meaningless
   `+0 ns`. Cause: the gate path allocates and the first batch paid to grow the
   heap. Fixed with a warm-up per batch and one shared batch size across the
   four prefixes. A non-positive delta now prints "below noise floor", not zero.
3. **"cycles" that were not cycles.** `CNTVCT_EL0` is a fixed ~24 MHz counter;
   the column is now `tick/call` with the rate printed beside it.

4. **Sub-nanosecond is also a lie.** After the `black_box` fix the FNV row still
   read 0.36 ns/call: `black_box` on the *return value* does not stop a pure
   function of a `const` input being hoisted out of the loop. Both ends need the
   barrier. `Row::is_suspect` now flags any row whose per-call figure rounds to
   zero, not just a zero-millisecond batch.

### Results (filled in §5.1, Table `tab:e1`)

Medians of 4 runs, aarch64/HVF, release, idle host. Full authorization decision
**1358 ns** (range 1144–1644, ±18%) against 43 ms per decoded token — a ratio of
**3×10⁻⁵**, i.e. ~32,000 gate crossings per token. Two findings that were not
expected:

- **The fine-grained gate dominates.** Scope is ~921 ns (68% of the decision);
  capability and taint are both *below the noise floor*. Against a task with no
  ledger entry the same call is 812 ns, so ~550 ns is the ledger walk + glob for
  a **single-entry** ledger and ~375 ns is building the normalized target. Both
  are heap allocations in the one gate that runs on every path — fixable
  (borrow the normalized path; compare a one-entry ledger without constructing a
  `Scope`), and written up as a limitation rather than hidden.
- **Recording the decision costs as much as making it.** One audit append is
  ~986 ns, 73% of the whole gate chain: a `Vec` push under a lock plus the
  field-by-field comparison the ktrace coalescer does per record.

Not measured on **x86** — the only x86 target here is QEMU TCG, where the figure
would be meaningless. That row needs real hardware or KVM.

## Before submitting

1. **Verify every bib entry** against DOI/DBLP. Several are flagged.
2. **Fill §5.** E1's harness exists (above) — run it on an idle machine and fill
   Table `tab:e1`. E2–E4 still need the adversarial corpus and the baselines.
3. Re-count the headline figures at submission time — LOC, primitive count,
   test count, and the `/perf` throughput numbers all drift:

   ```sh
   # unit-test count (paper says 1,196)
   grep -rc '#\[test_case\]' $(grep -rl '#\[test_case\]' ../kernel/src) | awk -F: '{s+=$2} END {print s}'

   # mechanism LOC *excluding* each file's own #[cfg(test)] module (paper says 2,383).
   # Deliberately excludes synapse/{ui,bench}.rs: ui is a surface registry, bench is
   # measurement. Counting whole files instead gives ~4,300 and is the wrong number.
   for f in ../kernel/src/synapse/{registry,grammar,executor,audit,vpath,fs}.rs \
            ../kernel/src/cap/mod.rs ../kernel/src/security/{taint,mod}.rs; do
     t=$(grep -n '^#\[cfg(test)\]' $f | head -1 | cut -d: -f1)
     echo ${t:-$(wc -l < $f)}
   done | paste -sd+ - | bc
   ```

4. Decide authorship/affiliation and add an acknowledgements section.
5. Consider whether the artifact should be cited with a commit hash and a
   reproduction recipe (`cargo xtask test`, `make e2e`).

## The argument, in one page

The classical OS contract assumes an untrusted but **deterministic** program
with an enumerable request set and ambient authority. An agent is untrusted,
**stochastic**, its request set is bounded only by expressibility, and its
intent is derived from data it read — so an adversary who controls ingested
content controls the requests. That is an *authorization* defect, and current
practice answers it above the OS with prompt guardrails and per-call approval.

Synapse moves it into the kernel: model output is an untrusted **plan** that
must pass grammar → capability → taint → scope before deterministic native
code executes it. The two claims worth defending are (a) enforcement applied
in the model's own token space *and* revalidated at the door, where only the
second is in the TCB, and (b) **provenance as a syscall argument** — a
Biba-style integrity policy, low-water-marked over the live context, that
refuses destructive primitives justified by untrusted content without ever
reading the injected text.

The number that decides whether it is more than a sound idea is the
**false-refusal rate** (E3), not the attack-success rate.
