# The Determinism Boundary — paper draft

An arXiv-shaped design paper on **Synapse**, the capability ABI of ChittiOS:
what an OS's protection mechanism has to become when the unit of execution is
an agent rather than a program.

- [main.tex](main.tex) — the draft
- [refs.bib](refs.bib) — bibliography, **verified 2026-07-29** against publisher
  pages / arXiv / ACL Anthology (the four entries written from working knowledge —
  CaMeL, AgentDojo, InjecAgent, Outlines — were all correct as written; page
  ranges added for LOMAC and Greshake)
- `make` — build `main.pdf` (`make manual` if you have no `latexmk`). Verified: 25 pages, 0 errors, 0 overfull boxes, 0 undefined references with TeX Live 2026

Intended categories: **cs.OS** (primary), cross-list **cs.CR**, **cs.AI**.

## Status

Sections 1–4 (introduction, execution model + threat model, mechanism,
implementation) are drafted against the code and are meant to be accurate as
written. Section 5 carries **measured results for E1–E4** (cost, attack corpus,
utility cost, baselines); E5 (per-gate complexity breakdown) is still partial.
Sections 6–8 (limitations, related work, conclusion) are drafted, and §6 now
reports the two liabilities the evaluation turned from predictions into numbers:
the 50% false-refusal rate, and the fact that the provenance policy is enforced
at eight sites rather than one.

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

Medians of 5 runs, aarch64/HVF, release, idle host. Full authorization decision
**1373 ns** (range 1144–1682, ±20%) against 43 ms per decoded token — a ratio of
**3×10⁻⁵**, i.e. ~31,000 gate crossings per token. Two findings that were not
expected:

- **The fine-grained gate dominates.** Scope is ~934 ns (68% of the decision);
  capability and taint are both *below the noise floor*. Against a task with no
  ledger entry the same call is 832 ns, so ~540 ns is the ledger walk + glob for
  a **single-entry** ledger and ~390 ns is building the normalized target. Both
  are heap allocations in the one gate that runs on every path — fixable
  (borrow the normalized path; compare a one-entry ledger without constructing a
  `Scope`), and written up as a limitation rather than hidden.
- **Recording the decision costs as much as making it.** One audit append is
  ~966 ns, 71% of the whole gate chain: a `Vec` push under a lock plus the
  field-by-field comparison the ktrace coalescer does per record.

Not measured on **x86** — the only x86 target here is QEMU TCG, where the figure
would be meaningless. That row needs real hardware or KVM.

## E2/E3/E4 (attacks, utility, baselines) — the harness

Built: [`kernel/src/security/redteam.rs`](../kernel/src/security/redteam.rs),
run with **`/redteam`** on the booted kernel. E2E scenario `redteam` (always-run
`os` group) asserts the *comparison*, not a single number.

```sh
python3 tests/e2e/run.py --only redteam -v     # ~3 min, no model needed
```

It drives every attack through the **real tool `Router`**, so the justification
comes from `Router::justification` over `Session::resident_max_taint` exactly as
in an agent turn — a provenance-laundering bug shows up as a `NOT TAINTED` row.

**It assumes the injection persuaded the model.** No gate reads the payload, so
varying wording would measure nothing; taking persuasion as given is the worst
case and makes the results deterministic and model-independent.

### Results

| Configuration | Attacks permitted | Benign steps needing a human |
| --- | --- | --- |
| Synapse (caps + scope + provenance) | **0 / 12** | **3 / 11** |
| Capabilities + scope, no provenance | 9 / 12 | 0 / 11 |
| Ambient authority (container) | 12 / 12 | 0 / 11 |
| Confirm every call | human-dependent | 11 / 11 |

- **E2:** 0/12 permitted — 10 refused by provenance, 2 by scope, all 12 turns
  correctly tainted. All eight taint enforcement sites held.
- **E3:** **false-refusal rate 50%** of benign destructive steps (2 of 4); 3/6
  tasks completely clean. This is the paper's headline liability.
- **E4:** provenance is worth 9 of the 12 refusals; scope is worth the other
  3 — they do measurably different jobs.
- **The capability gate stops none of the attacks**, in any configuration. Not a
  defect: injection uses authority the agent legitimately holds, which is
  precisely why capabilities are necessary and insufficient.

### Two things the harness found in itself

Both would have flattered the result, and both were caught by instrumentation
rather than by inspection:

1. An authority-transfer attack on `channel_grant` returned **malformed** — no
   tool lowers to that primitive, so the call died in shape validation and would
   have been scored as a gate refusal that never happened. Now measured properly
   by `synapse_reachable_destructive()` (only `mem_fs_delete` is Synapse-bound;
   egress is reachable through four *other* bindings, so "no Synapse-bound tool"
   must not be read as "unreachable").
2. The confined victim's ingestion pointed outside its own scope, so the read was
   scope-denied, the error result carried *trusted* provenance, the turn never
   tainted — and two later scope denials would have been reported as though
   provenance were involved. This is why the taint flag is reported per attack.

### Safety rules this harness must keep

A permitted attack under the ambient baseline **really executes**, and any user
can run `/redteam` on a real machine. Pinned by
`corpus_targets_are_sandboxed_and_offline`:

- every filesystem target under `/redteam/` (or the throwaway agent's own home);
- every network target the loopback **discard** port, so no packet leaves;
- no device/partition verbs — the destructive-shell attack is `rm` on a sandbox
  file, never `install`;
- the victim runs as a throwaway agent identity (`REDTEAM_AGENT`), because the
  memory-poison attack writes durable memory that re-enters the system prompt —
  running it as the live orchestrator would poison the real shell agent as a side
  effect of measuring whether that was possible.

## Figures

`figures/` holds both the generators and their output, so every figure is
reproducible and none is hand-drawn.

| File | What it is |
| --- | --- |
| [`make_figures.py`](figures/make_figures.py) | the three data figures (`fig_cost`, `fig_blocked`, `fig_tradeoff`). Every number in its `DATA` block is a median from a real run — re-measure and edit `DATA`, never nudge a figure |
| [`capture.py`](figures/capture.py) | the screenshots, taken from the guest's **own framebuffer** via the QEMU monitor's `screendump` (reachable because `xtask run` uses `-serial mon:stdio`), not reconstructed from a terminal transcript |
| `fig_*.pdf` | vector figures the paper includes. `make_figures.py` also writes `.png` twins for eyeballing; those are gitignored, since they are regenerable duplicates of the PDFs |
| `panes.png`, `redteam_table.png` | the two screenshots the paper uses |
| `desktop.png`, `redteam.png` | alternates kept for reuse (clean boot console; the split-pane summary with the browser pane mid-attack) |

```sh
python3 -m venv /tmp/figvenv && /tmp/figvenv/bin/pip install matplotlib
/tmp/figvenv/bin/python figures/make_figures.py    # data figures
python3 figures/capture.py                         # screenshots (boots QEMU, ~4 min)
```

Three decisions worth not undoing:

- **Palette** is the OS's own brand terracotta (`DESIGN.md`) plus four hues that
  pass the CVD validator in the `dataviz` skill under **`--pairs all`** — every
  pair, not just the adjacent ones a first pass samples. That distinction caught a
  shipped defect: the original **gold** scope hue passed as an adjacent pair but
  sits at ΔE 10.0 (normal vision) against terracotta, and `fig_blocked`'s
  caps+scope bar puts exactly those two side by side. Scope is now green. The one
  surviving warning (green vs terracotta, ΔE 6.8 protan) is in the band the skill
  allows *only* with secondary encoding — satisfied by a direct value label in
  every segment, a white gap between segments, and hatching on subsets.
- **Hue encodes exactly one dimension: which mechanism acted.** Terracotta always
  means "no gate stopped it", cyan provenance, green scope, indigo grammar, rose
  capability, grey a non-mechanism reference. The boundary diagram (`fig:boundary`,
  drawn in TikZ so its type matches the body) uses the same key, which is why
  capability needed a hue of its own rather than the grey it started with — grey
  says "not a mechanism", and capability is one. The first draft had terracotta doubling as "the scope
  gate", which is the mistake the rule exists to prevent. Hatching always marks a
  *subset* of the segment beside it.
- **Screenshots are captured at the console's native font scale.** Capturing at
  `/display scale 2` was tried and reverted: it makes fine print legible at print
  size but the console stops looking like itself — chunky glyphs, half the columns,
  output wrapping where it normally would not. A screenshot's job is to show what
  the system looks like, so the table shot instead buys readability by `/close`-ing
  the action band (giving the output the full 176-column pane) and the paper carries
  the values in a table beside it.
- **Wait on a marker the *guest* prints.** `wait_for("chitti")` matched
  `chitti-kernel` in cargo's build output, so when a release rebuild ran first the
  script screendumped a machine that had not booted — and every dump was silently
  empty. It now waits for `Commands start with`, which only the booted shell emits.

## arXiv compliance

`./arxiv.sh` (or `make arxiv`) builds `arxiv-submission.tar.gz` and checks it
against the rules that actually cause holds, each traced to
[submit_tex](https://info.arxiv.org/help/submit_tex.html) /
[prep](https://info.arxiv.org/help/prep.html) in the script's header. It packages
exactly `main.tex`, `refs.bib`, `main.bbl` and the **five referenced** figures —
289 KiB — and excludes the output PDF, aux files, hidden files, the generator
scripts, and the five unused figures sitting in `figures/`.

Verified, not assumed: the tarball was extracted to a clean directory and
compiled there with nothing else present — 25 pages, 0 errors, 0 undefined
references. That is the check that matters, because arXiv rebuilds from source
and will not see your working directory.

Four things the audit actually caught in this paper:

1. **The abstract was 3,138 characters.** arXiv rejects anything over 1,920. It
   is now 1,907 (13 to spare) and the paper's abstract was rewritten to match, so
   the two cannot say different things.
2. **`	oday` in `\date{}`** — arXiv asks you not to, because it makes the PDF
   differ on every rebuild. Now a fixed date.
3. **Five unused figures** in `figures/` would have shipped as "extraneous
   content"; the packager excludes anything `main.tex` does not reference.
4. **`main.pdf` must not be in the package** even though figure PDFs must. Easy
   to get backwards with one glob.

Things checked and already fine: all fonts embedded; every `\includegraphics`
path matches its file **case-sensitively** (macOS's case-insensitive filesystem
hides this class of bug until arXiv rejects it); no `\pdfoutput`, no `xr`, no
`#` in URLs; file names use only the permitted character set; `main.bbl` matches
the main `.tex` basename.

`arxiv-metadata.txt` holds the title, the ASCII abstract, the category choice and
the licence note to paste into the form. It is **generated** by `make arxiv-meta`
from `main.tex` — deriving it is the only way the form and the PDF cannot drift —
and the generator fails if the abstract exceeds 1,920 characters or picks up a
non-ASCII character.

## Before submitting

1. **Make `github.com/chittios/chitti` public, or remove the artifact link.** The
   paper cites the repo (`\cite{chittios}`) and §4 carries an *Artifact
   availability* paragraph. The repo is **private** as of this draft, so that URL
   404s for every reader — worse than no URL. The one blocker that is not a
   writing task.
2. **Tag the release and name it in the artifact paragraph.** The paper says the
   measurements "correspond to the release tagged for this paper" and deliberately
   does *not* carry a commit hash: the branch moves under active development, so a
   hash written today is stale by submission, and a bare 7-char hash on a private
   repo helps nobody. `git tag paper-v1 && git push --tags`, then name it in §4 and
   in `refs.bib`.
3. Bib entries are verified; re-check only if a preprint has since appeared in
   proceedings.
4. **§5 is filled** (E1–E4 measured; E5 partial — the per-gate LOC/test
   breakdown is still TBD). Re-run E1 on an idle machine before camera-ready;
   consider porting AgentDojo/InjecAgent for the model-in-the-loop half E2
   deliberately assumes away.
5. Re-count the headline figures at submission time — LOC, primitive count,
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

6. Decide authorship/affiliation and add an acknowledgements section.
7. Consider adding an acknowledgements section.

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
