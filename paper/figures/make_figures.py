#!/usr/bin/env python3
"""Generate the paper's figures from the measured E1-E4 numbers.

Every value in DATA below is a median from a real run on the booted kernel
(`/bench synapse` and `/redteam`); nothing here is illustrative. Re-measure and
edit DATA — do not hand-adjust a figure.

Vector PDF out, sized to the paper's text width, so the figures are crisp at any
zoom and the type matches the body at 8pt.

Design decisions, deliberate and worth keeping:

* **Palette** is the OS's own brand terracotta plus four hues validated for
  colour-vision deficiency with `dataviz/scripts/validate_palette.js`, light mode,
  **`--pairs all`** — every pair, not the adjacent ones a first pass happens to
  sample. That distinction caught a real defect: the original gold scope hue
  passed as an adjacent pair but sits at dE 10.0 (normal vision) against
  terracotta, and `fig_blocked`'s caps+scope bar puts exactly those two segments
  side by side. Gold is gone; scope is green. The surviving warning is
  green-vs-terracotta at dE 6.8 protan, which the skill permits only with
  secondary encoding — satisfied here by a direct value label in every segment, a
  white gap between segments, and hatching on the subsets.
* **Hue encodes exactly one dimension: which mechanism acted.** Terracotta is
  always "no gate stopped it", cyan always provenance, gold always scope, indigo
  always grammar, grey always a non-mechanism reference. Whether "nothing stopped
  it" is good or bad is what the panel title says — which is precisely why the
  colour must not also be reused for a gate, as it was in the first draft of
  fig_cost. Configurations get no hue of their own (fig_tradeoff is single-hue);
  position carries that.
* **No dual axes anywhere.** Where two measures differ by five orders of
  magnitude (a gate versus a token) they get their own panel with their own axis,
  and the ratio is stated in the annotation instead of implied by geometry.
* **Texture, not a fifth hue,** separates over-broad refusals from warranted
  ones: they are the same mechanism doing the same thing, and only one of them is
  a mistake, so a hatch marks the subset without pretending it is a new category.
* Bars carry direct value labels, so the numbers survive greyscale printing and
  the grid can stay almost invisible.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# --- the design system ------------------------------------------------------
# ChittiOS brand (DESIGN.md) + CVD-validated companions.
# Hue encodes ONE dimension throughout: which mechanism acted. That is what lets
# the same key be read across figures -- and it is why terracotta cannot also be
# "the scope gate" in one panel while meaning "nothing stopped it" in another.
THROUGH = "#cc785c"  # terracotta: no gate stopped it (permitted / proceeded)
PROV = "#0891b2"  # cyan: provenance (the taint gate)
SCOPE = "#2e7d32"  # green: scope
GRAMMAR = "#6349b8"  # indigo: grammar
CAPABILITY = "#b0447a"  # rose: capability
# Grey is the non-mechanism role: reference magnitudes and whole-policy baselines.
INK = "#141413"
BODY = "#3d3d3a"
MUTED = "#6c6a64"
HAIR = "#e6dfd8"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": BODY,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": BODY,
    "axes.linewidth": 0.6,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# --- measured data ----------------------------------------------------------
DATA = {
    # E1, medians of 5 runs, aarch64/HVF release. Marginal cost per gate (ns).
    "gates": [("grammar", 435), ("capability", 4), ("taint", 0), ("scope", 934)],
    "decision_ns": 1373,
    "audit_ns": 966,
    "hash_ns": 69,
    "noledger_ns": 832,
    "decode_token_ns": 43_478_261,  # 23 tok/s
    "prefill_token_ns": 9_523_810,  # 105 tok/s
    # E2/E4: attacks (n=12) by outcome, per configuration.
    # `eff` = permitted AND the effect then happened. The gap is attacks the policy
    # allowed that failed for a reason which is not a defence (a loopback connect
    # refused; `/http` not being exposed to agents at all). Both numbers are
    # reported because either alone misleads.
    "attacks": {
        "Synapse\n(caps+scope+provenance)": {"eff": 0, "failed": 0, "prov": 10, "scope": 2},
        "Capabilities + scope\n(no provenance)": {"eff": 6, "failed": 3, "prov": 0, "scope": 3},
        "Ambient authority\n(container)": {"eff": 9, "failed": 3, "prov": 0, "scope": 0},
    },
    # E3: benign steps (n=11) under the full policy.
    "benign": {"proceeded": 8, "warranted": 1, "over_broad": 2},
    "destructive_steps": 4,
}


def _clean(ax, grid_axis="x"):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis=grid_axis, color=HAIR, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def _label_on(colour):
    """Ink or white, whichever the segment can actually carry.

    The cut is at 0.22 rather than the intuitive 0.5 because it is set by the
    marginal cases, not the obvious ones: terracotta and cyan are light enough that
    a 7pt number reads better in ink than in white, while green, rose, indigo and
    grey are not. Picking per segment keeps every in-bar number legible instead of
    legible on average.
    """
    r, g, b = (int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return INK if lum > 0.22 else "white"


def fig_cost(path):
    """E1: what the decision is made of, and whether any of it matters.

    Two panels because the two questions live five orders of magnitude apart. The
    left is a composition (which gate spends the time); the right is a ratio, on a
    log axis with dots rather than bars, since bar *length* on a log scale encodes
    nothing.
    """
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.1), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(wspace=0.5)

    seg_colours = {"grammar": GRAMMAR, "capability": CAPABILITY, "taint": PROV, "scope": SCOPE}
    left = 0.0
    for name, ns in DATA["gates"]:
        if ns <= 0:
            continue
        a.barh(1, ns, left=left, height=0.46, color=seg_colours[name],
               edgecolor="white", linewidth=1.2, zorder=3)
        if ns > 200:
            a.text(left + ns / 2, 1, f"{name}\n{ns} ns", ha="center", va="center",
                   fontsize=7, color=_label_on(seg_colours[name]), zorder=4)
        left += ns
    a.barh(0, DATA["audit_ns"], height=0.46, color=MUTED, edgecolor="white", linewidth=1.2, zorder=3)
    a.text(DATA["audit_ns"] / 2, 0, f"{DATA['audit_ns']} ns", ha="center", va="center",
           fontsize=7, color=_label_on(MUTED), zorder=4)
    a.text(DATA["decision_ns"] + 35, 1, f"{DATA['decision_ns']} ns", va="center",
           fontsize=7.5, color=BODY)
    a.set_yticks([1, 0], ["authorization\ndecision", "audit append\n(one record)"])
    a.set_ylim(-0.5, 1.62)
    a.set_xlim(0, 1780)
    a.set_xlabel("nanoseconds per call (median of 5 runs)")
    a.set_title("Where the decision goes", loc="left", color=INK, pad=6)
    _clean(a)
    # Two real gates measured below the noise floor. Said out loud, in the empty
    # band between the bars, rather than left as invisible slivers that would
    # imply they were free.
    a.text(20, 0.52, "capability +4 ns; taint below\nthe method's noise floor",
           fontsize=6.3, color=MUTED, va="center", linespacing=1.35)

    pts = [("authorization decision", DATA["decision_ns"], THROUGH),
           ("one prefilled token", DATA["prefill_token_ns"], MUTED),
           ("one decoded token", DATA["decode_token_ns"], MUTED)]
    for i, (label, ns, c) in enumerate(pts):
        b.plot([1, ns], [i, i], color=HAIR, linewidth=1.0, zorder=2, solid_capstyle="butt")
        b.plot(ns, i, "o", markersize=7, color=c, zorder=3)
        txt = f"{ns / 1000:.1f} $\\mu$s" if ns < 1e6 else f"{ns / 1e6:.0f} ms"
        b.text(ns * 1.9, i, txt, va="center", fontsize=7.5, color=BODY)
    b.set_yticks(range(len(pts)), [p[0] for p in pts])
    b.set_xscale("log")
    b.set_xlim(500, 1.2e9)
    b.set_ylim(-0.55, 2.5)
    b.set_xlabel("nanoseconds (log scale)")
    b.set_title("…and whether it matters", loc="left", color=INK, pad=6)
    _clean(b)
    b.text(1.1e4, 1.45, r"the decision is $3\times10^{-5}$" "\n" "of one decoded token",
           fontsize=7, color=INK, ha="center", va="center")

    fig.savefig(path)
    plt.close(fig)


def fig_blocked(path):
    """E2/E4 beside E3: what stops the attacks, and what that costs.

    One shared key for both panels, phrased by *mechanism* ("no gate stopped it")
    rather than by valence, because the same outcome is the goal on the right and
    the failure on the left — which is the comparison the figure exists to make.
    """
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.55), gridspec_kw={"width_ratios": [1.55, 1]})
    fig.subplots_adjust(wspace=0.32, bottom=0.30, top=0.80)

    labels = list(DATA["attacks"].keys())
    ys = list(range(len(labels)))[::-1]
    for y, name in zip(ys, labels):
        d = DATA["attacks"][name]
        left = 0
        for key, colour, hatch in (("eff", THROUGH, None), ("failed", THROUGH, "///"),
                                   ("prov", PROV, None), ("scope", SCOPE, None)):
            v = d[key]
            if v == 0:
                continue
            a.barh(y, v, left=left, height=0.5, color=colour, edgecolor="white",
                   linewidth=1.2, hatch=hatch, zorder=3)
            a.text(left + v / 2, y, str(v), ha="center", va="center", fontsize=7.5,
                   color=_label_on(colour), zorder=4)
            left += v
    a.set_yticks(ys, labels)
    a.set_xlim(0, 12.4)
    a.set_xticks([0, 3, 6, 9, 12])
    a.set_ylim(-0.6, 2.6)
    a.set_xlabel("injected attacks (n = 12)")
    a.set_title("Attacks: provenance is what stops them", loc="left", color=INK, pad=6)
    _clean(a)

    left = 0
    for v, colour, hatch in ((DATA["benign"]["proceeded"], THROUGH, None),
                             (DATA["benign"]["warranted"], PROV, None),
                             (DATA["benign"]["over_broad"], PROV, "///")):
        b.barh(0, v, left=left, height=0.5, color=colour, edgecolor="white",
               linewidth=1.2, hatch=hatch, zorder=3)
        # A one- or two-unit segment is about 0.2in wide at this figure size, which
        # cannot hold a 7.5pt number: those labels go above the bar instead of
        # being technically present and practically unreadable.
        if v >= 3:
            b.text(left + v / 2, 0, str(v), ha="center", va="center", fontsize=7.5,
                   color=_label_on(colour), zorder=4)
        else:
            b.text(left + v / 2, 0.30, str(v), ha="center", va="bottom", fontsize=7,
                   color=BODY, zorder=4)
        left += v
    b.set_ylim(-0.6, 0.6)
    b.set_yticks([])
    b.set_xlim(0, 11.3)
    b.set_xticks([0, 4, 8, 11])
    b.set_xlabel("benign steps, full policy (n = 11)")
    b.set_title("The cost: 2 of 4 destructive\nsteps refused over-broadly", loc="left", color=INK, pad=6)
    _clean(b)

    # One key for both panels. A hatch always means "a subset of the segment beside
    # it": permitted-but-the-effect-failed on the left, refused-over-broadly on the
    # right. Hue still means only which mechanism acted.
    handles = [Patch(facecolor=THROUGH, label="no gate stopped it"),
               Patch(facecolor=THROUGH, hatch="///", edgecolor="white", label="…effect failed anyway"),
               Patch(facecolor=PROV, label="refused: provenance"),
               Patch(facecolor=PROV, hatch="///", edgecolor="white", label="…over-broadly"),
               Patch(facecolor=SCOPE, label="denied: scope")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               handlelength=1.2, columnspacing=1.1, bbox_to_anchor=(0.5, -0.03))

    fig.savefig(path)
    plt.close(fig)


def fig_tradeoff(path):
    """E4 as a position, not a table: a defence must be good on both axes.

    A scatter, because the claim is about a *pair* of measures and bars invite
    reading one column and stopping. Single hue: the entities are configurations,
    and inventing a colour dimension for them would collide with the mechanism key
    the other figures teach. Nothing is shaded as a "good" region — Synapse's 27%
    interruption rate is a real cost and a green box around it would launder that.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    ax.plot([0, 100], [100, 100], color=MUTED, linewidth=2, zorder=3, solid_capstyle="butt")
    ax.plot([0, 100], [100, 100], "|", color=MUTED, markersize=7, zorder=4)
    ax.text(50, 91, "confirm every call — attack rate is\nwhatever the human notices",
            fontsize=6.5, color=MUTED, ha="center", va="top")

    # Labels are staggered rather than centred over their dots: at 75% and 100% the
    # two baselines sit close enough that centred captions overlap.
    for label, x, y, dx, dy, ha in (("Synapse", 0.0, 27.3, 5, 6, "left"),
                                    ("caps + scope,\nno provenance", 75.0, 0.0, -7, 1, "right"),
                                    ("ambient\nauthority", 100.0, 0.0, 0, 9, "center")):
        ax.plot(x, y, "o", markersize=9, color=THROUGH, zorder=5,
                markeredgecolor="white", markeredgewidth=1.4)
        ax.text(x + dx, y + dy, label, fontsize=7, color=BODY, ha=ha, va="bottom")

    ax.annotate("better", xy=(1, 2), xytext=(20, 14), fontsize=6.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.7))
    ax.set_xlim(-8, 114)
    ax.set_ylim(-11, 116)
    ax.set_xlabel("attacks permitted (%)")
    ax.set_ylabel("benign steps needing a human (%)")
    ax.set_title("Good on both axes, or not a defence", loc="left", color=INK, pad=6)
    _clean(ax, grid_axis="both")
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    out = []
    for name, fn in (("fig_cost", fig_cost), ("fig_blocked", fig_blocked), ("fig_tradeoff", fig_tradeoff)):
        pdf = os.path.join(HERE, f"{name}.pdf")
        png = os.path.join(HERE, f"{name}.png")  # for eyeballing only
        fn(pdf)
        fn(png)
        out.append(name)
    print("wrote:", ", ".join(out))
