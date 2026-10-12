"""Figures of the manuscript other than Fig 4 and Fig 5, drawn from the per-fly tables of the battery designs, the tables in results/stats and the engine validation files (text of 8 to 12 pt).

  python code/scripts/analyze_battery.py       # writes results/stats/ (already present)
  python code/scripts/facts_*.py               # writes results/facts/facts_*.json (already present)
  python code/scripts/make_figures.py [fig1 fig2 fig3 fig6 fig7 s1 s2 s3 s4]

Writes figures/Fig1, Fig2, Fig3, Fig6, Fig7, S1_Fig, S2_Fig, S3_Fig and S4_Fig as PNG (300 dpi), PDF
and LZW-compressed TIFF, sized for a 7.5-inch page width. Fig4 (burden-matched maps) and Fig5 (within-fly
rescue) are made by their own scripts and are not touched here.

Numbering. Each function is named after the figure it writes. The learning index and the
joint cohort are supporting figures:
  fig6  Fig6    discrete phenotypes, panels A to F
  fig7  Fig7    separation from healthy, panels A, B, C
  s3    S3_Fig  learning index and MBON rate, panels A to C
  s4    S4_Fig  joint cohort, one panel

The other figures: Fig1, model and protocol (panel C is a schematic of the three maps); Fig2, the ten
readouts against severity; Fig3, rows A rule ablation, B Kenyon-cell-only, C maps at equal neuron loss;
S1_Fig, engine validation; S2_Fig, the independent groups of the default model. Fig6 counts giant fibre
output below 100 Hz and high-activity flies (two criteria); Fig7 B shows the observed first crossings
with those of the bootstrap.

Order. The functions stand in this file, and are run, in the order in which they take numbers from the one
random generator RNG: fig1, fig2, fig3, s3, fig6, fig7, s4, s1, s2 (s3 is the fourth; fig7, s4 and s1
take none). The bootstrap intervals and the jitter of a figure depend on that order,
so it must not follow the figure numbers: see the dispatcher at the end.

Computation: bootstrap intervals drawn here use 4,000 resamples, the number behind the intervals
quoted in the text. Every count printed inside a figure is computed from
the per-fly tables and checked against results/facts/facts_*.json (assertions below).
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())  # code/scripts locally, scripts/ on the server
BAT = ROOT / "results/battery"
STATS_DIR = ROOT / "results/stats"
FACTS_DIR = ROOT / "results/facts"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


STATS = _json(STATS_DIR / "battery_stats.json")
FDET = _json(FACTS_DIR / "facts_detection.json")
FBAT = _json(FACTS_DIR / "facts_battery.json")
FMOD = _json(FACTS_DIR / "facts_model.json")
RNG = np.random.default_rng(7)
NBOOT = 4000

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "figure.dpi": 150, "pdf.fonttype": 42, "axes.titlepad": 3,
    "axes.labelpad": 2, "legend.frameon": False, "legend.handlelength": 1.2,
})
INK, GREY, LIGHT, SUB = "#1d2329", "#8a9096", "#d5d9dc", "#4a5057"
COL = {"brain_spikes_A": "#1f4e79", "LH_hz_A": "#d9711c", "PN_hz_A": "#7d8b99", "KC_hz_A": "#4f8f3a",
       "KC_frac_A": "#86b86f", "MBON_hz_A": "#2a8c8c", "V_pre_A": "#6f6f8f", "LI": "#b5367a",
       "LI_norm": "#e08bb8", "MN9_sugar_hz": "#8c564b", "GF_loom_hz": "#5e3c99",
       "behaviour_feed_s": "#a07d1c", "behaviour_esc": "#b0413e", "behaviour_near_A_s": "#9aa5ad",
       "behaviour_PI": "#c3c9ce"}
NAME = {"brain_spikes_A": "Whole-brain spikes", "LH_hz_A": "Lateral horn rate", "PN_hz_A": "Projection neuron rate",
        "KC_hz_A": "Kenyon cell rate", "KC_frac_A": "Kenyon cells recruited", "MBON_hz_A": "MBON rate",
        "V_pre_A": "Naive MB valence", "LI": "Learning index", "LI_norm": "Learning index / MBON rate",
        "MN9_sugar_hz": "MN9 feeding drive", "GF_loom_hz": "Giant fibre rate",
        "behaviour_feed_s": "Time feeding", "behaviour_esc": "Jumps per threat",
        "behaviour_near_A_s": "Time near trained odour", "behaviour_PI": "Odour preference index"}
UNIT = {"brain_spikes_A": "spikes in 1 s", "LH_hz_A": "Hz", "PN_hz_A": "Hz", "KC_hz_A": "Hz",
        "KC_frac_A": "% of Kenyon cells", "MBON_hz_A": "Hz", "V_pre_A": "signed sum, Hz", "LI": "signed sum, Hz",
        "MN9_sugar_hz": "Hz", "GF_loom_hz": "Hz"}
MAPC = {"hub": "#1f4e79", "shuffled": "#8a9096", "antihub": "#e08a2e"}
RULEC = {"all": INK, "no_hyper": "#6b7178", "hyper_only": "#c0392b", "synloss_only": "#2e86c1",
         "death_only": "#27ae60"}
RULEN = {"all": "All three", "no_hyper": "No threshold\nshift", "hyper_only": "Threshold\nshift only",
         "synloss_only": "Synapse loss\nonly", "death_only": "Neuron loss\nonly"}
ANCHOR = FMOD["fig_anchor_hardcoded"]   # 0.55: severity at which Kenyon cell loss matches the ~17% at 40 days
FAIL_RED = "#c0392b"


def load(name):
    d = pd.read_csv(BAT / name)
    d["LI_norm"] = d["LI"] / d["MBON_hz_A"].where(d["MBON_hz_A"] >= 0.5)
    return d


SC = load("stage_curve_x4.csv")
CO = load("components_x4.csv")
VU = load("vulnerability_x4.csv")
SP = load("scope_x4.csv")
H = SC[SC.stage == 0]
STAGES = sorted(SC.stage.unique())
AFF = SC[SC.stage > 0]


def panel(ax, letter, dx=-0.32, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left",
            color=INK)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04)
    im = Image.open(OUT / f"{name}.png").convert("RGB")
    im.save(OUT / f"{name}.tif", compression="tiff_lzw", dpi=(300, 300))
    plt.close(fig)
    print("wrote", name)


def pct_ci(x, h, n=NBOOT):
    x, h = np.asarray(pd.Series(x).dropna(), float), np.asarray(pd.Series(h).dropna(), float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    est = np.median(x) / np.median(h) * 100
    r = (np.median(RNG.choice(x, (n, len(x))), 1) / np.median(RNG.choice(h, (n, len(h))), 1) * 100)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return est, lo, hi


def jitter(n, w=0.018):
    return RNG.uniform(-w, w, n)


# ============================================================ Fig 1: model and protocol
def fig1():
    ev = _json(ROOT / "results/validation/engine_validation.json")["comparisons"]
    r_b2 = ev["gate=True vs local_brian2_2.10.1"]["pearson_r"]
    assert abs(r_b2 - 0.9997) < 1e-9, r_b2
    n_flies = FMOD["cohort_total_flies"]
    assert n_flies == 1424 and FDET["cohorts_reported_total"][2] == 1424
    assert "v630" in FMOD["engine_validation_network"]

    # Layout in inches from the bottom-left corner. Lower row: three axes of height hP whose x axes line up,
    # with a strip above them that holds the legends (yleg) and the panel letters (yrow). Upper row: panel A.
    W, y0, hP, hA = 7.5, 0.52, 1.65, 1.60
    yleg, yrow = y0 + hP + 0.12, y0 + hP + 0.37
    yA = yrow + 0.11 + 0.18
    H = yA + hA + 0.07 + 0.11 + 0.04
    xB, wB, xC, wC, xD, wD = 0.10, 2.80, 3.51, 1.64, 5.71, 1.72
    fig = plt.figure(figsize=(W, H))

    def axes_in(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    def letter(x, y, s):
        fig.text(x / W, y / H, s, fontsize=11, fontweight="bold", va="baseline", ha="left", color=INK)

    # A: pipeline schematic (axes coordinates in inches)
    ax = axes_in(0.05, yA, W - 0.1, hA)
    ax.set_xlim(0.05, W - 0.05); ax.set_ylim(0, hA); ax.axis("off")
    letter(0.05, yA + hA + 0.07, "A")
    bw, gap, bx0 = 1.715, 0.17, 0.065
    xs = [bx0 + i * (bw + gap) for i in range(4)]
    cx = [x + bw / 2 for x in xs]
    for x, title in zip(xs, ("Connectome", "Whole-CNS LIF engine", "Pathology in each fly", "Readouts")):
        ax.add_patch(FancyBboxPatch((x, 0.01), bw, hA - 0.02, boxstyle="round,pad=0,rounding_size=0.07",
                                    fc="#f3f5f7", ec=GREY, lw=0.8, clip_on=False))
        ax.text(x + bw / 2, hA - 0.20, title, ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
    body = dict(ha="center", va="center", fontsize=8, color=INK, linespacing=1.45)
    ymid = (hA - 0.34) / 2 + 0.02
    ax.text(cx[0], ymid, "MaleCNS v1.0\n166,700 neurons\n24.5 million signed edges\n120 million synaptic contacts",
            **body)
    # engine box: validated base model on top, MaleCNS extensions listed separately below a divider
    ax.text(cx[1], hA - 0.55, f"Shiu et al. parameters\nbase model on FlyWire v630\n"
                              f"matches Brian2 (r = {r_b2:.4f})", **body)
    ax.plot([xs[1] + 0.12, xs[1] + bw - 0.12], [hA - 0.83, hA - 0.83], color=GREY, lw=0.6, ls=(0, (2, 2)))
    ax.text(cx[1], hA - 1.17, "MaleCNS extensions,\nnot in that comparison:\n+ feedforward antennal lobe\n"
                              "+ spike-frequency adaptation", **dict(body, color=SUB))
    ax.text(cx[2], ymid, "load l = s × u\n(u: connectivity rank)\nthreshold lowered by 2 mV × l\n"
                         "synapse loss, p = 0.6 × l\nneuron loss, p = 0.5 × l", **body)
    ax.text(cx[3], ymid, f"12 circuit readouts\n4 free-behaviour readouts\n{n_flies:,} simulated flies\n"
                         "each seeded and reproducible", **body)
    for x in xs[:-1]:
        ax.add_patch(FancyArrowPatch((x + bw + 0.035, hA / 2), (x + bw + gap - 0.03, hA / 2), arrowstyle="-|>",
                                     mutation_scale=9, color=GREY, lw=1.0, shrinkA=0, shrinkB=0))

    # B: protocol timeline (x in seconds, y in inches above the time axis)
    ax = axes_in(xB, y0, wB, hP)
    letter(0.05, yrow, "B")
    segs = [("A", 0, 1, COL["LH_hz_A"]), ("B", 2, 1, COL["LH_hz_A"]),
            ("S", 4, 0.5, COL["MN9_sugar_hz"]), ("L", 5.5, 0.3, COL["GF_loom_hz"]),
            ("A+D", 6.8, 1, COL["LI"]), ("B", 9.8, 1, COL["LI"]),
            ("A", 12.8, 1, COL["LI"]), ("B", 14.8, 1, COL["LI"])]
    ylab, bar_h = 0.58, 0.36                     # top of the bracket labels; height of the stimulus bars
    ybar = ylab + 0.15
    for lab, t0, dur, c in segs:
        ax.add_patch(Rectangle((t0, ybar), dur, bar_h, fc=c, ec="none", alpha=0.9))
        ax.text(t0 + dur / 2, ybar + bar_h + 0.035, lab, ha="center", va="bottom", fontsize=8, color=INK)
    # label centres: the three left-hand labels are nudged apart by at most 0.2 s so that they stay separate
    brackets = [(0, 3, "naive odour\nresponses", 1.35), (4, 4.5, "MN9", 4.2), (5.5, 5.8, "GF", 5.8),
                (6.8, 10.8, "training", 8.8), (12.8, 15.8, "memory test\n(learning index)", 14.3)]
    for a, b_, lab, xc in brackets:
        ax.plot([a, a, b_, b_], [ylab + 0.10, ylab + 0.04, ylab + 0.04, ylab + 0.10], color=GREY, lw=0.7)
        ax.text(xc, ylab, lab, ha="center", va="top", fontsize=8, color=SUB, linespacing=1.2)
    for k, key in enumerate(("A, B: odours   S: sugar   L: looming (left)", "D: dopaminergic PAM drive")):
        fig.text((xB + 0.2 * wB / 16.4) / W, (yleg + 0.025 - 0.155 * k) / H, key, fontsize=8, color=SUB,
                 va="baseline", ha="left")
    ax.set_xlim(-0.2, 16.2); ax.set_ylim(0, hP)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 4, 8, 12, 16]); ax.set_xlabel("Time in protocol (s)")

    # C: load against connectivity rank under the three maps (idealised schematic, s = 1)
    ax = axes_in(xC, y0, wC, hP)
    letter(xC - 0.37, yrow, "C")
    u = np.linspace(0, 1, 400)
    ax.plot(u, u, color=MAPC["hub"], lw=1.6, label="Hub (default)")
    ax.scatter(RNG.uniform(0, 1, 200), RNG.uniform(0, 1, 200), s=4, color=MAPC["shuffled"], lw=0,
               alpha=0.8, label="Shuffled")
    ax.plot(u, 1 - u, color=MAPC["antihub"], lw=1.6, label="Antihub")
    fig.text(xC / W, yrow / H, "Schematic of the three maps", fontsize=8, color=SUB, va="baseline", ha="left")
    ax.set_xlabel("Connectivity rank of neuron\n(percentile of synapse count)")
    ax.set_ylabel("Load at s = 1")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"])
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, (yleg - y0) / hP), ncol=3, columnspacing=0.8,
              handletextpad=0.3, handlelength=1.0, markerscale=2.2, borderaxespad=0, borderpad=0)

    # D: burden against severity
    ax = axes_in(xD, y0, wD, hP)
    letter(xD - 0.35, yrow, "D")
    b = SC.groupby("stage").agg(dead=("dead_frac", "mean"), kc=("kc_dead_frac", "mean")).reset_index()
    ax.axvspan(ANCHOR, 1.03, color="#eef0f2", lw=0, zorder=0)
    ax.text(0.79, 35.3, "extrapolation", ha="center", va="top", fontsize=8, color=GREY)
    ax.axhspan(16.0, 17.0, color="#f2c9b4", lw=0)
    ax.text(0.015, 18.2, "Kenyon cell loss\nat 40 days,\nAβ42 flies", fontsize=8, color="#9a4a24", va="bottom",
            linespacing=1.15)
    ax.plot(b.stage, 100 * b.kc, "-o", ms=3.2, lw=1.2, color=COL["KC_hz_A"], label="Kenyon cells", clip_on=False)
    ax.plot(b.stage, 100 * b.dead, "-o", ms=3.2, lw=1.2, color=INK, label="All neurons", clip_on=False)
    ax.set_xlabel("Severity s"); ax.set_ylabel("Neurons removed (%)")
    ax.set_xlim(-0.02, 1.03); ax.set_ylim(0, 36)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks([0, 10, 20, 30]); ax.set_yticks([5, 15, 25, 35], minor=True)
    ax.tick_params(axis="y", which="minor", length=1.5, width=0.6)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, (yleg - y0) / hP), ncol=2, columnspacing=1.0,
              handletextpad=0.4, handlelength=1.6, borderaxespad=0, borderpad=0)
    save(fig, "Fig1")


# ============================================================ Fig 2: dose response
def _fig2_head(fig, ax, letter, title, left_pt, up_pt=7.0):
    """Panel letter and title on one line above the axes; the line starts left_pt points left of the y axis."""
    tr = mpl.transforms.offset_copy(ax.transAxes, fig=fig, x=-left_pt, y=up_pt, units="points")
    ax.text(0, 1, letter, transform=tr, fontsize=11, fontweight="bold", va="baseline", ha="left", color=INK)
    tr = mpl.transforms.offset_copy(ax.transAxes, fig=fig, x=-left_pt + 13.5, y=up_pt, units="points")
    ax.text(0, 1, title, transform=tr, fontsize=8, va="baseline", ha="left", color=INK)


def fig2():
    reads = ["brain_spikes_A", "LH_hz_A", "PN_hz_A", "KC_hz_A", "KC_frac_A",
             "MBON_hz_A", "V_pre_A", "LI", "MN9_sugar_hz", "GF_loom_hz"]
    yticks = {"brain_spikes_A": [0, 5000, 10000, 15000, 20000], "LH_hz_A": [0, 0.5, 1.0, 1.5],
              "PN_hz_A": [0.5, 1.0, 1.5], "KC_hz_A": [0, 0.5, 1.0, 1.5], "KC_frac_A": [0, 10, 20, 30],
              "MBON_hz_A": [0, 2, 4, 6], "V_pre_A": [0, 100, 200], "LI": [0, 50, 100],
              "MN9_sugar_hz": [0, 20, 40], "GF_loom_hz": [0, 50, 100, 150, 200]}
    # layout in inches: five columns of 0.96 in axes; the left margin holds the five-digit ticks of panel A,
    # and the right margin keeps the cropped file under 7.5 in
    W, Hh = 7.5, 4.12
    left, right, top, bottom, aw, ah, hgap = 0.555, 0.15, 0.25, 0.40, 0.96, 1.50, 0.45
    wgap = (W - left - right - 5 * aw) / 4
    fig, axs = plt.subplots(2, 5, figsize=(W, Hh))
    fig.subplots_adjust(left=left / W, right=1 - right / W, top=1 - top / Hh, bottom=bottom / Hh,
                        wspace=wgap / aw, hspace=hgap / ah)
    trend = STATS["series"]["trend"]
    for k, (ax, r) in enumerate(zip(axs.flat, reads)):
        y = SC[r] * (100 if r == "KC_frac_A" else 1)
        ax.axvspan(ANCHOR, 1.04, color="#eef0f2", lw=0, zorder=0)
        ax.scatter(SC.stage + jitter(len(SC)), y, s=3.6, color=COL[r], alpha=0.45, lw=0, zorder=2)
        med = SC.assign(y=y).groupby("stage").y.median()
        ax.plot(med.index, med.values, color=INK, lw=1.2, zorder=3)
        ax.set_ylabel(UNIT[r], fontsize=8, labelpad=3)
        ax.set_yticks(yticks[r])
        # the correlation sits in the top right corner: raise the upper limit where flies at s >= 0.4 reach it
        lo, hi = ax.get_ylim()
        reach = y[SC.stage >= 0.4].max()
        if (reach - lo) / (hi - lo) > 0.82:
            ax.set_ylim(lo, lo + (reach - lo) / 0.82)
        ax.text(0.97, 0.97, f"ρ = {trend[r]['rho']:+.2f}".replace("-", "\u2212"), transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=SUB)
        ax.set_xticks([0, 0.5, 1]); ax.set_xlim(-0.04, 1.04)
        if k >= 5:
            ax.set_xlabel("Severity s", labelpad=3)
        else:
            ax.set_xticklabels([])
        _fig2_head(fig, ax, "ABCDEFGHIJ"[k], NAME[r], left_pt=29.5)
    save(fig, "Fig2")


# ============================================================ Fig 3: rules, scope, placement
def _cis(xs, groups, ref, r, offs=(-0.14, 0.14)):
    out = []
    for gi, (x, (g05, g10)) in enumerate(zip(xs, groups)):
        for g, off, fill in ((g05, offs[0], True), (g10, offs[1], False)):
            e, lo, hi = pct_ci(g[r], ref[r])
            if not np.isnan(e):
                out.append((gi, x + off, e, lo, hi, fill))
    return out


def fig3():
    reads = ["brain_spikes_A", "MBON_hz_A", "LI", "MN9_sugar_hz", "GF_loom_hz"]
    conds = ["all", "no_hyper", "hyper_only", "synloss_only", "death_only"]
    maps = ["hub", "shuffled", "antihub"]
    # same Kenyon cell loss in the KC-only and pan-neuronal groups at each severity (checked against facts)
    for s, key in ((0.5, "kconly_s0.5_kc_dead_pct"), (1.0, "kconly_s1_kc_dead_pct")):
        kc_only = 100 * SP[SP.stage == s].kc_dead_frac.mean()
        pan = 100 * SC[SC.stage == s].kc_dead_frac.mean()
        assert abs(kc_only - FBAT[key][0]) < 0.01 and abs(pan - FBAT[key][1]) < 0.01, (kc_only, pan)
        assert abs(kc_only - pan) < 0.5
    rows = [
        ("A", "Each rule alone and in combination", list(range(5)),
         [(CO[CO.condition == f"{c}_s0.5"], CO[CO.condition == f"{c}_s1"]) for c in conds],
         [RULEC[c] for c in conds], [RULEN[c].replace("\n", " ") for c in conds], (-8, 265), [0, 100, 200]),
        ("B", "Pathology confined to Kenyon cells (same Kenyon cell loss as pan-neuronal)", [0, 1],
         [(SP[SP.stage == 0.5], SP[SP.stage == 1.0]), (SC[SC.stage == 0.5], SC[SC.stage == 1.0])],
         ["#4f8f3a", INK], ["Kenyon cells only", "Pan-neuronal"], (-5, 125), [0, 50, 100]),
        ("C", "Same number of neurons removed, different neurons", list(range(3)),
         [(VU[VU.condition == f"{m}_s0.5"], VU[VU.condition == f"{m}_s1"]) for m in maps],
         [MAPC[m] for m in maps], ["Hub (load rises with connectivity)", "Shuffled",
                                   "Antihub (load falls with connectivity)"],
         (-5, 112), [0, 50, 100]),
    ]
    # Layout in inches. Each row is a heading line (letter and title), a colour key line and five panels.
    # Text is 8 pt (row titles 9 pt, panel letters 11 pt), so the key does not fit beside the title.
    # The intervals come from the shared generator: the order of the _cis calls (row, readout, group,
    # s = 0.5 then 1.0) must stay as it is.
    FW = 7.5
    X0, X1, GAP = 0.53, 7.41, 0.37            # left spine of the first panel, right edge of the last, gap
    AX_W, AX_H = (X1 - X0 - 4 * GAP) / 5, 1.18
    TOP, HEAD_KEY, KEY_AX, ROW_GAP, BOT = 0.20, 0.235, 0.375, 0.40, 0.06
    ROW = HEAD_KEY + KEY_AX + AX_H
    FH = TOP + 3 * ROW + 2 * ROW_GAP + BOT
    fig = plt.figure(figsize=(FW, FH))
    key = dict(fontsize=8, columnspacing=1.3, handletextpad=0.3, handlelength=0.9, borderpad=0,
               borderaxespad=0, frameon=False)
    for i, (letter, title, xs, groups, colors, labels, ylim, yticks) in enumerate(rows):
        y_head = FH - TOP - i * (ROW + ROW_GAP)      # baseline of the heading line
        y_key = y_head - HEAD_KEY                    # centre of the colour key line
        y_ax = y_key - KEY_AX - AX_H                 # bottom of the panels
        pts = {r: _cis(xs, groups, H, r, offs=(-0.19, 0.19)) for r in reads}
        top = max(ylim[1], max(p[4] for r in reads for p in pts[r]) * 1.04)
        for j, r in enumerate(reads):
            ax = fig.add_axes([(X0 + j * (AX_W + GAP)) / FW, y_ax / FH, AX_W / FW, AX_H / FH])
            ax.axhline(100, color=GREY, lw=0.7, ls="--", zorder=0)
            for gi, x, e, lo, hi, fill in pts[r]:
                c = colors[gi]
                ax.plot([x] * 2, [lo, hi], color=c, lw=1.1)
                ax.plot(x, e, "o", ms=4.0, mfc=c if fill else "white", mec=c, mew=1.0)
            ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
            ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7); ax.set_ylim(ylim[0], top)
            ax.set_yticks(yticks)
            ax.set_title(NAME[r], fontsize=8, loc="left", pad=4)
            if j == 0:
                ax.set_ylabel("Median, % of healthy", labelpad=3)
        fig.text(0.05 / FW, y_head / FH, letter, fontsize=11, fontweight="bold", color=INK, va="baseline")
        fig.text(X0 / FW, y_head / FH, title, fontsize=9, color=INK, va="baseline")
        handles = [plt.Line2D([], [], marker="o", ls="", ms=4.2, color=c, label=l) for c, l in zip(colors, labels)]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=((X0 - 0.02) / FW, y_key / FH),
                   ncol=len(handles), **key)
        if i == 0:      # filled and open symbols mean the same in all three rows
            fill = [plt.Line2D([], [], marker="o", ls="", ms=4.2, color=GREY, label="filled: s = 0.5"),
                    plt.Line2D([], [], marker="o", ls="", ms=4.2, mfc="white", mec=GREY, mew=1.0,
                               label="open: s = 1.0")]
            fig.legend(handles=fill, loc="center right", bbox_to_anchor=(X1 / FW, (y_head + 0.029) / FH),
                       ncol=2, **key)
    save(fig, "Fig3")


# ============================================================ S3 Fig: learning index and MBON drive
# Stands here, between Fig 3 and Fig 6, because its bootstrap resamples are the fourth block of draws from RNG.
def s3():
    # Layout in inches (7.5 in page width; save() crops to the drawn content). Text is 8 pt, panel letters 11 pt.
    W, HT = 7.5, 3.42
    Y0, AH = 0.86, 2.24                                   # bottom and height of the three axes
    XA, WA, XB, WB, XC, WC = 0.47, 1.60, 3.27, 1.84, 5.86, 1.57
    LET = {"A": 0.44, "B": 0.44, "C": 0.56}               # panel letter: distance left of the axes

    def _s3_axes(x, w, y=Y0, h=AH):
        return fig.add_axes([x / W, y / HT, w / W, h / HT])

    fig = plt.figure(figsize=(W, HT))
    ax = _s3_axes(XA, WA)
    cmap = mpl.colormaps["viridis_r"]
    sc_ = ax.scatter(SC.MBON_hz_A, SC.LI, c=SC.stage, cmap=cmap, vmin=0, vmax=1, s=6, lw=0, alpha=0.85)
    ax.set_xlabel("MBON rate during odour A (Hz)", labelpad=3); ax.set_ylabel("Learning index (signed sum, Hz)")
    ax.set_xticks([0, 2, 4, 6]); ax.set_yticks(range(0, 141, 20))
    cax = _s3_axes(XA + WA + 0.09, 0.085, y=Y0 + 0.14 * AH, h=0.72 * AH)
    cb = fig.colorbar(sc_, cax=cax)
    cb.set_label("Severity s", fontsize=8, labelpad=3); cb.ax.tick_params(labelsize=8, length=2.5)
    cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    rho = STATS["series"]["learning"]["spearman_LI_MBON_all"][0]
    ax.text(0.98, 0.025, f"ρ = {rho:.2f}\n{len(SC)} flies", transform=ax.transAxes, va="bottom", ha="right",
            fontsize=8, linespacing=1.15)
    panel(ax, "A", dx=-LET["A"] / WA, dy=1 + 0.10 / AH)
    ax = _s3_axes(XB, WB)
    ax.axhline(100, color=GREY, lw=0.7, ls="--")
    ax.axvspan(ANCHOR, 1.04, color="#eef0f2", lw=0, zorder=0)
    for r, lab in (("MBON_hz_A", "MBON rate"), ("LI", "Learning index"), ("LI_norm", "Learning index / MBON rate")):
        xs, es, los, his = [], [], [], []
        for s in STAGES:
            g = SC[SC.stage == s][r].dropna()
            if len(g) < 8:
                continue
            e, lo, hi = pct_ci(g, H[r])
            xs.append(s); es.append(e); los.append(lo); his.append(hi)
        ax.fill_between(xs, los, his, color=COL[r], alpha=0.16, lw=0)
        ax.plot(xs, es, "-o", ms=3.0, lw=1.2, color=COL[r], label=lab)
    ax.set_xlabel("Severity s", labelpad=3); ax.set_ylabel("Median, % of healthy")
    ax.set_ylim(-5, 190); ax.set_yticks([0, 25, 50, 75, 100, 125, 150]); ax.set_xlim(-0.03, 1.04)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    ax.legend(loc="upper left", fontsize=8, borderaxespad=0.15, labelspacing=0.3, handlelength=1.5,
              handletextpad=0.5)
    panel(ax, "B", dx=-LET["B"] / WB, dy=1 + 0.10 / AH)
    ax = _s3_axes(XC, WC)
    ax.axhline(100, color=GREY, lw=0.7, ls="--")
    conds = ["all", "no_hyper", "hyper_only", "synloss_only", "death_only"]
    for i, c in enumerate(conds):
        for s, off, fill in ((0.5, -0.15, True), (1.0, 0.15, False)):
            g = CO[CO.condition == f"{c}_s{s:g}"]["LI_norm"].dropna()
            if len(g) < 8:
                continue
            e, lo, hi = pct_ci(g, H["LI_norm"])
            ax.plot([i + off] * 2, [lo, hi], color=RULEC[c], lw=1.1)
            ax.plot(i + off, e, "o", ms=4.2, mfc=RULEC[c] if fill else "white", mec=RULEC[c], mew=1.0)
    ax.set_xlim(-0.5, 4.5)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([RULEN[c] for c in conds], rotation=90, fontsize=8,
                                                         linespacing=1.0, multialignment="right")
    ax.set_ylabel("Learning index / MBON rate\n(median, % of healthy)"); ax.set_ylim(0, 180)
    ax.set_yticks(range(0, 181, 20))
    ax.plot([], [], "o", ms=4.2, color=INK, label="s = 0.5")
    ax.plot([], [], "o", ms=4.2, mfc="white", mec=INK, mew=1.0, label="s = 1.0")
    ax.legend(loc="upper left", fontsize=8, borderaxespad=0.15, labelspacing=0.3, handletextpad=0.3)
    panel(ax, "C", dx=-LET["C"] / WC, dy=1 + 0.10 / AH)
    save(fig, "S3_Fig")


# ============================================================ Fig 6: discrete phenotypes
def _fig6_letter(ax, letter, left_in=0.46):
    """Panel letter a fixed distance (inches) left of the axes, on the baseline of the panel title."""
    w_in = ax.get_position().width * ax.figure.get_figwidth()
    panel(ax, letter, dx=-left_in / w_in, dy=1.005)


def fig6():
    onl = STATS["output_neuron_loss"]
    # counts printed in the figure, checked against facts_detection.json
    t = onl["gf_fail_by_left_removed"]
    assert [t["True|True"], t["True|True"] + t["True|False"]] == FDET["gf_recon_left_removed_fail"]
    assert [t["False|True"], t["False|True"] + t["False|False"]] == FDET["gf_recon_left_kept_fail"]
    n_below = int((AFF.GF_loom_hz < 100).sum()); n_silent = int((AFF.GF_loom_hz == 0).sum())
    assert (n_below, n_silent) == (FDET["gf_n_below100"], FDET["gf_n_below100_silent"])
    hmax_kc, hmax_mb = H.KC_hz_A.max(), H.MBON_hz_A.max()

    # Layout for 8 pt text: three columns of equal width across 7.5 in, two rows. The printed counts sit in
    # the empty band above the data (A, B, E), the legends in empty corners (C, F), and the colour scale of D
    # inside its empty upper-left quadrant so that all six axes keep the same size.
    sev_ticks = [0, 0.25, 0.5, 0.75, 1.0]
    fig = plt.figure(figsize=(7.5, 5.5))
    gs = fig.add_gridspec(2, 3, left=0.066, right=0.984, top=0.964, bottom=0.109, hspace=0.40, wspace=0.31)
    # A: giant fibre
    axA = ax = fig.add_subplot(gs[0, 0])
    y = SC.GF_loom_hz
    low = y < 100
    ax.scatter(SC.stage[~low] + jitter((~low).sum()), y[~low], s=3.6, color=COL["GF_loom_hz"], alpha=0.45, lw=0)
    ax.scatter(SC.stage[low] + jitter(low.sum()), y[low], s=7, color=FAIL_RED, lw=0, alpha=0.8)
    ax.axhline(100, color=GREY, lw=0.7, ls="--")
    ax.set_xlabel("Severity s"); ax.set_ylabel("Giant fibre rate (Hz)"); ax.set_ylim(-8, 300)
    ax.set_xticks(sev_ticks); ax.set_yticks([0, 50, 100, 150, 200, 250])
    ax.set_title("Looming: giant fibre", loc="left")
    ax.text(0.03, 0.975, f"left GF lost: {t['True|True']} of {t['True|True'] + t['True|False']} below 100 Hz\n"
            f"left GF kept: {t['False|True']} of {t['False|True'] + t['False|False']} below 100 Hz\n"
            f"below 100 Hz: {n_below} flies, {n_silent} silent",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=SUB, linespacing=1.15)
    _fig6_letter(ax, "A")
    # B: MN9
    axB = ax = fig.add_subplot(gs[0, 1])
    y = SC.MN9_sugar_hz
    sil = y == 0
    ax.scatter(SC.stage[~sil] + jitter((~sil).sum()), y[~sil], s=3.6, color=COL["MN9_sugar_hz"], alpha=0.45, lw=0)
    ax.scatter(SC.stage[sil] + jitter(sil.sum()), y[sil] + RNG.uniform(-0.6, 0.6, sil.sum()), s=7,
               color=FAIL_RED, lw=0, alpha=0.8)
    ax.set_xlabel("Severity s"); ax.set_ylabel("MN9 feeding drive (Hz)"); ax.set_ylim(-2.2, 66)
    ax.set_xticks(sev_ticks); ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_title("Sugar: MN9", loc="left")
    m = onl["mn9_silent_by_n_removed"]
    n_sil = m["0|True"] + m["1|True"] + m["2|True"]
    one = m["1|True"] + m["1|False"]
    assert n_sil == FDET["mn9_n_silent_affected"] == int((AFF.MN9_sugar_hz == 0).sum())
    ax.text(0.98, 0.975, f"silent: {n_sil} flies, {m['1|True'] + m['2|True']} of them\nlacking ≥1 MN9 neuron\n"
            f"one MN9 lost, still active: {m['1|False']} of {one}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=SUB, linespacing=1.15)
    _fig6_letter(ax, "B")
    # C: counts per severity
    axC = ax = fig.add_subplot(gs[0, 2])
    mf = STATS["series"]["motor_failures"]
    xs = [s for s in STAGES if s > 0]
    gf_n = [mf["GF_loom_hz"][str(s)]["n_fail"] for s in xs]
    assert gf_n == [b[0] for b in FDET["gf_bands_by_severity"]]
    ax.plot(xs, gf_n, "-o", ms=3.4, lw=1.2, color=COL["GF_loom_hz"], label="Giant fibre < 100 Hz", clip_on=False)
    ax.plot(xs, [mf["MN9_sugar_hz"][str(s)]["n_fail"] for s in xs], "-o", ms=3.4, lw=1.2, color=COL["MN9_sugar_hz"],
            label="MN9 silent (0 Hz)", clip_on=False)
    ax.set_xlabel("Severity s"); ax.set_ylabel("Flies (of 32)")
    ax.set_xticks(sev_ticks); ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0, 22); ax.set_yticks([0, 5, 10, 15, 20])
    ax.legend(loc="upper left", fontsize=8, borderaxespad=0.2, handlelength=1.6, labelspacing=0.35)
    _fig6_letter(ax, "C")
    # D: KC vs MBON at low severity, with the healthy maxima
    axD = ax = fig.add_subplot(gs[1, 0])
    low = SC[SC.stage <= 0.3]
    cmap = mpl.colormaps["viridis_r"]
    sc_ = ax.scatter(low.KC_hz_A, low.MBON_hz_A, c=low.stage, cmap=cmap, vmin=0, vmax=1, s=7, lw=0, alpha=0.9)
    ax.axvline(hmax_kc, color=GREY, lw=0.7, ls=":", zorder=0)
    ax.axhline(hmax_mb, color=GREY, lw=0.7, ls=":", zorder=0)
    cax = ax.inset_axes([0.065, 0.44, 0.042, 0.41])
    cb = fig.colorbar(sc_, cax=cax)
    cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); cb.ax.tick_params(labelsize=8, length=2.5, pad=2)
    cb.outline.set_linewidth(0.6)
    cax.set_title("Severity s", fontsize=8, loc="left", pad=7.5)
    ax.set_xlabel("Kenyon cell rate (Hz)"); ax.set_ylabel("MBON rate (Hz)")
    ax.set_xticks([0.5, 1.0, 1.5]); ax.set_yticks([2, 4, 6])
    ax.set_title("Severity 0 to 0.3", loc="left")
    ax.annotate("high-activity flies", xy=(1.22, 4.5), xytext=(1.08, 3.3), ha="left", va="center", fontsize=8,
                color=SUB, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7, shrinkA=1))
    ax.text(hmax_kc + 0.035, 0.72, "healthy\nmaxima", fontsize=8, color=GREY, va="bottom", ha="left",
            linespacing=1.1)
    _fig6_letter(ax, "D")
    # E: KC rate by rule at s = 0.5, with the number of flies above the healthy maximum
    axE = ax = fig.add_subplot(gs[1, 1])
    groups = [("Healthy", H, GREY, None),
              ("Threshold\nshift only", CO[CO.condition == "hyper_only_s0.5"], RULEC["hyper_only"], "kchigh_s0.5_hyper_only"),
              ("Synapse\nloss only", CO[CO.condition == "synloss_only_s0.5"], RULEC["synloss_only"], "kchigh_s0.5_synloss_only"),
              ("Neuron\nloss only", CO[CO.condition == "death_only_s0.5"], RULEC["death_only"], "kchigh_s0.5_death_only"),
              ("All three", CO[CO.condition == "all_s0.5"], RULEC["all"], "kchigh_s0.5_all")]
    for i, (lab, g, c, key) in enumerate(groups):
        ax.scatter(i + RNG.uniform(-0.18, 0.18, len(g)), g.KC_hz_A, s=4.5, color=c, lw=0, alpha=0.7)
        ax.plot([i - 0.25, i + 0.25], [g.KC_hz_A.median()] * 2, color=INK, lw=1.2)
        n_hi = int((g.KC_hz_A > hmax_kc).sum())
        if key is not None:
            assert n_hi == FBAT[key][0], (key, n_hi)
        ax.text(i, 0.895, f"{n_hi}/{len(g)}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color=SUB)
    ax.text(0.03, 0.975, "flies above the healthy maximum:", transform=ax.transAxes, fontsize=8, color=SUB,
            va="top", ha="left")
    ax.axhline(hmax_kc, color=GREY, lw=0.7, ls=":")
    ax.set_ylim(-0.04, 1.86); ax.set_yticks([0, 0.5, 1.0, 1.5]); ax.set_xlim(-0.55, 4.55)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], rotation=90, fontsize=8,
                                                           linespacing=1.1)
    ax.set_ylabel("Kenyon cell rate (Hz)"); ax.set_title("Severity 0.5", loc="left")
    _fig6_letter(ax, "E")
    # F: high-activity flies per severity under the healthy-referenced and the group-median criteria
    axF = ax = fig.add_subplot(gs[1, 2])
    xs = [s for s in STAGES if s > 0]
    n_healthy_ref, n_group = [], []
    for s in xs:
        g = SC[SC.stage == s]
        n_healthy_ref.append(int(((g.KC_hz_A > hmax_kc) & (g.MBON_hz_A > hmax_mb)).sum()))
        n_group.append(int(((g.KC_hz_A > 1.5 * g.KC_hz_A.median()) & (g.MBON_hz_A > 1.5 * g.MBON_hz_A.median())).sum()))
    assert n_healthy_ref == FDET["hiact_healthy_referenced_KCmax_and_MBONmax_by_severity"], n_healthy_ref
    assert n_group == FDET["hiact_groupmedian_criterion_by_severity"], n_group
    w = 0.019
    ax.bar(np.array(xs) - w / 2, n_healthy_ref, width=w, color=COL["KC_hz_A"], lw=0,
           label="KC and MBON rate both above\nthe healthy maximum")
    ax.bar(np.array(xs) + w / 2, n_group, width=w, color="white", ec=COL["KC_hz_A"], lw=0.8,
           label="KC and MBON rate both > 1.5 ×\nthe median of the fly's own\nseverity group")
    ax.set_xlabel("Severity s"); ax.set_ylabel("High-activity flies (of 32)")
    ax.set_ylim(0, 14.5); ax.set_yticks([0, 2, 4, 6, 8]); ax.set_xlim(0, 1.04); ax.set_xticks(sev_ticks)
    ax.legend(loc="upper left", fontsize=8, borderaxespad=0.15, handlelength=1.0, handletextpad=0.6,
              labelspacing=0.55)
    _fig6_letter(ax, "F")
    fig.align_ylabels([axA, axD]); fig.align_ylabels([axB, axE]); fig.align_ylabels([axC, axF])
    save(fig, "Fig6")


# ============================================================ Fig 7: which readouts separate first (panels A to C)
def _axes_in(fig, x0, y0, w, h):
    """Axes placed in inches, measured from the lower-left corner of the figure."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x0 / fw, y0 / fh, w / fw, h / fh])


def _letter_in(ax, letter, x_in, up_in=0.11):
    """Panel letter with its left edge x_in inches from the figure's left edge, up_in above the axes."""
    fw, fh = ax.figure.get_size_inches()
    p = ax.get_position()
    panel(ax, letter, dx=(x_in / fw - p.x0) / p.width, dy=1 + up_in / (p.height * fh))


def fig7():
    # Row 1: A and B side by side. Row 2: C, whose three columns share the readout names at the left and
    # span the page width. Inches; every text item is 8 pt except the panel letters (11 pt). No random numbers.
    FW, FH = 7.5, 6.00
    H1, H2 = 2.15, 2.00               # axes heights of the two rows
    Y1 = FH - 0.34 - H1               # bottom of row 1 (A, B)
    Y2 = Y1 - 0.86 - H2               # bottom of row 2 (C)
    XR, WR = 4.80, 2.60               # left edge and width of the axes of B
    fig = plt.figure(figsize=(FW, FH))
    au = pd.read_csv(STATS_DIR / "series_auc.csv")
    order = ["brain_spikes_A", "LH_hz_A", "MBON_hz_A", "KC_hz_A", "KC_frac_A", "V_pre_A", "MN9_sugar_hz",
             "GF_loom_hz", "LI", "PN_hz_A"]
    hi_ = ["brain_spikes_A", "LH_hz_A", "MN9_sugar_hz", "GF_loom_hz", "LI"]
    ax = _axes_in(fig, 0.52, Y1, 2.50, H1)
    ax.axhline(0.8, color=GREY, lw=0.7, ls="--")
    for r in order:
        if r in hi_:
            continue
        g = au[au.readout == r].sort_values("stage")
        ax.plot(g.stage, g.auc, "-", color=LIGHT, lw=1.0, zorder=1)
    for r in hi_:
        g = au[au.readout == r].sort_values("stage")
        ax.plot(g.stage, g.auc, "-o", ms=3.0, color=COL[r], lw=1.4, label=NAME[r], zorder=3)
    ax.plot([], [], "-", color=LIGHT, lw=1.0, label="other circuit readouts")
    ax.set_xlabel("Severity s"); ax.set_ylabel("Separation from healthy (AUC)")
    ax.set_ylim(0.48, 1.02); ax.set_xlim(0, 1.03)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.legend(loc="lower right", fontsize=8, handlelength=1.6, labelspacing=0.32, borderaxespad=0.2,
              handletextpad=0.6)
    _letter_in(ax, "A", 0.05)
    # B: first severity with AUC >= 0.80: observed crossing and bootstrap distribution
    ax = _axes_in(fig, XR, Y1, WR, H1)
    fd = STATS["series"]["first_detection"]
    fkey = {"brain_spikes_A": "brain", "LH_hz_A": "LH", "PN_hz_A": "PN", "KC_hz_A": "KCrate",
            "KC_frac_A": "KCrecruited", "MBON_hz_A": "MBON", "V_pre_A": "valence", "LI": "LI", "LI_norm": "LInorm",
            "MN9_sugar_hz": "MN9", "GF_loom_hz": "GF"}
    for r, k in fkey.items():
        assert fd[r]["observed"] == FDET[f"fd_series_observed_{k}"], r
        assert fd[r]["boot_median"] == FDET[f"fd_series_bootmedian_copied_{k}"], r
    rows = [r for r in order + ["LI_norm"] if fd[r]["boot_median"] is not None]
    rows = sorted(rows, key=lambda r: (fd[r]["observed"] if fd[r]["observed"] is not None else 9,
                                       fd[r]["boot_median"], fd[r]["boot_ci95"][1]))
    for i, r in enumerate(rows):
        lo, hi = fd[r]["boot_ci95"]
        c = COL.get(r, INK)
        ax.plot([lo, hi], [i, i], color=c, lw=3.0, alpha=0.45, solid_capstyle="round")
        ax.plot(fd[r]["boot_median"], i, "o", ms=4.4, color=c, zorder=3)
        if fd[r]["observed"] is not None:
            ax.plot(fd[r]["observed"], i, "D", ms=6.4, mfc="none", mec=INK, mew=0.9, zorder=4)
        else:
            ax.text(hi + 0.04, i, f"never reached (observed);\nnot reached in {100 * fd[r]['frac_never']:.0f}%\n"
                    "of resamples", va="center", fontsize=8, color=SUB, linespacing=1.15)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([NAME[r] for r in rows], fontsize=8)
    # first row on top; the x range runs past 1.0 so that the three-line note fits to the right of the last dot
    ax.set_ylim(len(rows) + 0.1, -0.6); ax.set_xlim(0, 1.35); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("First severity with AUC ≥ 0.80")
    hd = [plt.Line2D([], [], marker="D", ls="", ms=6.4, mfc="none", mec=INK, mew=0.9, label="observed"),
          plt.Line2D([], [], marker="o", ls="", ms=4.4, color=GREY, label="bootstrap median"),
          plt.Line2D([], [], ls="-", lw=3.0, color=GREY, alpha=0.45, label="bootstrap 95% range")]
    ax.legend(handles=hd, loc="upper right", fontsize=8, borderaxespad=0.1, handlelength=1.6, labelspacing=0.4)
    _letter_in(ax, "B", 3.22)
    # C: change, trial-to-trial spread and AUC at s = 0.10 for each readout
    pr = STATS["series"]["precision"]
    rkey = {"brain_spikes_A": "brain", "LH_hz_A": "LH", "PN_hz_A": "PN", "KC_hz_A": "KCrate",
            "KC_frac_A": "KCrecruited", "MBON_hz_A": "MBON", "V_pre_A": "valence", "LI": "LI",
            "MN9_sugar_hz": "MN9", "GF_loom_hz": "GF"}
    for r, k in rkey.items():
        assert abs(100 * pr[r]["healthy_rcv"] - FDET[f"series_healthy_rcv_pct_{k}"]) < 0.01, r
    rr = sorted(pr, key=lambda r: -pr[r]["auc_0.1"])
    yy = np.arange(len(rr))
    cols = [("Median change at s = 0.10 (%)", lambda v: abs(v["median_shift_pct_0.1"]), (0, 28), [0, 10, 20]),
            ("Trial-to-trial spread, healthy\n(robust CV, %)", lambda v: 100 * v["healthy_rcv"], (0, 36),
             [0, 15, 30]),
            ("AUC at s = 0.10", lambda v: v["auc_0.1"], (0.45, 0.9), [0.5, 0.7, 0.9])]
    # left edge, width and gap of the three columns of C; the last column ends 0.10 in left of the axes of B,
    # so that its last tick label stays inside the 7.5 in page width
    XC, WC, GC = 1.32, 1.80, 0.29
    for k, (lab, fn, xl, xt) in enumerate(cols):
        ax = _axes_in(fig, XC + k * (WC + GC), Y2, WC, H2)
        for i, r in enumerate(rr):
            v = fn(pr[r])
            motor = r in ("MN9_sugar_hz", "GF_loom_hz")
            ax.plot([xl[0], v], [i, i], color=LIGHT, lw=1.0, zorder=1)
            ax.plot(v, i, "s" if motor else "o", ms=4.6, color=COL[r], zorder=2)
        ax.set_ylim(len(rr) - 0.5, -0.5); ax.set_xlim(*xl); ax.set_xticks(xt)
        ax.set_xlabel(lab, fontsize=8, linespacing=1.15)
        ax.tick_params(axis="x", labelsize=8)
        if k == 0:
            ax.set_yticks(yy); ax.set_yticklabels([NAME[r] for r in rr], fontsize=8)
            _letter_in(ax, "C", 0.05)
        else:
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
    save(fig, "Fig7")


# ============================================================ S4 Fig: joint cohort, circuit and arena readouts
# (one panel, no panel letter)
def s4():
    ja = pd.read_csv(STATS_DIR / "joint_auc.csv")
    # readout, line style: circuit readouts solid, arena readouts dashed (feeding, jumps) or dotted (odour)
    lines = (("brain_spikes_A", "-"), ("LH_hz_A", "-"), ("LI", "-"), ("behaviour_feed_s", "--"),
             ("behaviour_esc", "--"), ("behaviour_near_A_s", ":"), ("behaviour_PI", ":"))
    # every plotted AUC, the severities and the group sizes against facts_detection.json; the number of flies
    # of the cohort (given in the caption) against battery_stats.json and facts_model.json
    fkey = {"brain_spikes_A": "brain", "LH_hz_A": "LH", "LI": "LI", "behaviour_feed_s": "feedTime",
            "behaviour_esc": "jumpsPerThreat", "behaviour_near_A_s": "nearOdourA", "behaviour_PI": "PI"}
    assert [r for r, _ in lines] == list(fkey)
    for r, k in fkey.items():
        g = ja[ja.readout == r].sort_values("stage")
        assert [round(float(v), 4) for v in g.stage] == FDET["joint_severities"], r
        assert [round(float(v), 4) for v in g.auc] == FDET[f"joint_auc_{k}"], r
        assert g.n_healthy.max() <= FDET["n_joint_healthy"] and g.n_group.max() <= FDET["n_joint_per_severity"], r
    b = ja[ja.readout == "brain_spikes_A"]
    assert (b.n_healthy == FDET["n_joint_healthy"]).all() and (b.n_group == FDET["n_joint_per_severity"]).all()
    assert ((b.lo <= b.auc) & (b.auc <= b.hi)).all()
    n_flies = FDET["n_joint_healthy"] + FDET["n_joint_per_severity"] * len(FDET["joint_severities"])
    assert n_flies == 176 == STATS["joint"]["n"] == FMOD["joint_seconds_seed_nflies"][2]

    # Layout in inches: one axes with its key below. Text is 8 pt. No random numbers.
    FW, FH = 4.6, 4.2
    X0, WA, Y0, HA = 0.55, 3.90, 1.22, 2.70
    fig = plt.figure(figsize=(FW, FH))
    ax = _axes_in(fig, X0, Y0, WA, HA)
    ax.axhline(0.8, color=GREY, lw=0.7, ls="--")
    g = ja[ja.readout == "brain_spikes_A"].sort_values("stage")
    ax.fill_between(g.stage, g.lo, g.hi, color=COL["brain_spikes_A"], alpha=0.12, lw=0)
    for r, ls in lines:
        g = ja[ja.readout == r].sort_values("stage")
        ax.plot(g.stage, g.auc, ls, marker="o", ms=3.0, color=COL[r], lw=1.3, label=NAME[r])
    ax.set_xlabel("Severity s"); ax.set_ylabel("Separation from healthy (AUC)")
    ax.set_ylim(0.45, 1.02); ax.set_xlim(0.03, 0.52)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5]); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # no heading inside the figure: it has one panel, and its caption gives the title and the cohort
    # key below the axes: circuit readouts (solid) in the left column, arena readouts in the right one
    hs, ls_ = ax.get_legend_handles_labels()
    ax.legend(hs[:3] + [plt.Line2D([], [], ls="none")] + hs[3:], ls_[:3] + [""] + ls_[3:], loc="upper center",
              bbox_to_anchor=((X0 + WA / 2) / FW, (Y0 - 0.45) / FH), bbox_transform=fig.transFigure,
              fontsize=8, ncol=2, columnspacing=2.0, handlelength=2.3, labelspacing=0.32, borderaxespad=0,
              handletextpad=0.6)
    save(fig, "S4_Fig")


# ============================================================ S1 Fig: engine validation (B, D like for like)
def s1():
    # Each column compares the engine with a reference at the input rate of that reference (seed 7).
    # A, C: Brian2 2.10.1 run locally at 150 Hz, the default of the released code (validate_engine_final.py).
    # B, D: the example output released with the code, which was generated at 200 Hz, against the engine
    #       driven at 200 Hz (validate_engine_200hz.py), so that both runs have the same input rate.
    z150 = np.load(ROOT / "results/validation/engine_validation_rates.npz")
    c150 = _json(ROOT / "results/validation/engine_validation.json")["comparisons"]
    z200 = np.load(ROOT / "results/validation/engine_validation_200hz_rates.npz")
    c200 = _json(ROOT / "results/validation/engine_validation_200hz.json")["rates"]["200"]["comparisons"]
    cols = (("150-Hz input", "Brian2 2.10.1, run locally",
             lambda g: (z150["ref_local_brian2_2.10.1"], z150[f"ours_gate{g}"],
                        c150[f"gate={bool(g)} vs local_brian2_2.10.1"])),
            ("200-Hz input", "Released example output",
             lambda g: (z200["ref_200hz_distributed"], z200[f"ours_200hz_gate{g}_seed7"],
                        c200[f"gate={bool(g)} vs distributed_example_output"]["seed_7"])))
    # 8 pt text throughout; decade ticks are written as plain numbers (mathtext exponents render below 8 pt)
    fig, axs = plt.subplots(2, 2, figsize=(5.7, 5.54))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.09, top=0.95, wspace=0.34, hspace=0.36)
    floor = 0.25
    decades = [1, 10, 100]
    for i, (g, gname) in enumerate(((1, "Refractory input discarded"), (0, "Refractory input accumulated"))):
        for j, (title, rname, pick) in enumerate(cols):
            ax = axs[i, j]
            x, y, c = pick(g)
            m = (x > 0) | (y > 0)
            assert int(m.sum()) == c["n_compared"], (title, gname, int(m.sum()), c["n_compared"])
            assert max(x.max(), y.max()) < 400, "rate outside the axis range"
            ax.scatter(np.maximum(x[m], floor), np.maximum(y[m], floor), s=4.5, lw=0, alpha=0.6,
                       color="#1f4e79" if i == 0 else "#d9711c")
            ax.plot([floor, 400], [floor, 400], color=LIGHT, lw=0.9, zorder=0)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.2, 400); ax.set_ylim(0.2, 400)
            ax.set_box_aspect(1)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(mpl.ticker.FixedLocator(decades))
                axis.set_major_formatter(mpl.ticker.FixedFormatter([f"{d:g}" for d in decades]))
                axis.set_minor_formatter(mpl.ticker.NullFormatter())
            ax.tick_params(axis="both", which="major", labelsize=8, length=3.5)
            ax.tick_params(axis="both", which="minor", length=2.0, width=0.5)
            ax.text(0.05, 0.96, f"r = {c['pearson_r']:.4f}\nslope = {c['slope']:.3f}\nn = {c['n_compared']}",
                    transform=ax.transAxes, va="top", fontsize=8, linespacing=1.25)
            ax.set_title(title, fontsize=8, pad=4)
            ax.set_xlabel(f"{rname} (Hz)", fontsize=8, labelpad=3)
            if j == 0:
                ax.set_ylabel(f"This engine (Hz)\n{gname}", fontsize=8, labelpad=3, linespacing=1.25)
            panel(ax, "ABCD"[2 * i + j], dx=-0.3, dy=1.04)
    save(fig, "S1_Fig")


# ============================================================ S2 Fig: independent groups of the default model
def s2():
    reads = ["brain_spikes_A", "LH_hz_A", "KC_hz_A", "MBON_hz_A", "LI", "MN9_sugar_hz", "GF_loom_hz"]
    short = {"brain_spikes_A": "Whole-brain\nspikes", "LH_hz_A": "Lateral horn\nrate", "KC_hz_A": "Kenyon cell\nrate",
             "MBON_hz_A": "MBON\nrate", "LI": "Learning\nindex", "MN9_sugar_hz": "MN9 feeding\ndrive",
             "GF_loom_hz": "Giant fibre\nrate"}
    # Geometry in inches: seven columns of independent y axes at 8 pt need about as much room for
    # tick labels and units as for the data, so margins, panel width and gaps are fixed here. The
    # content must stay inside 7.42 in: save() adds 0.04 in on each side.
    W, HT = 7.5, 4.86
    left, right, top, bottom, row_gap = 0.80, 0.07, 0.55, 1.27, 0.48
    aw, ah = 0.54, (HT - top - bottom - row_gap) / 2
    gap = (W - left - right - 7 * aw) / 6
    fig, axs = plt.subplots(2, 7, figsize=(W, HT))
    fig.subplots_adjust(left=left / W, right=1 - right / W, bottom=bottom / HT, top=1 - top / HT,
                        wspace=gap / aw, hspace=row_gap / ah)
    for i, s in enumerate((0.5, 1.0)):
        groups = [("Severity series", SC[SC.stage == s]), ("Rule ablation", CO[CO.condition == f"all_s{s:g}"]),
                  ("Map comparison (hub)", VU[VU.condition == f"hub_s{s:g}"])]
        for j, r in enumerate(reads):
            ax = axs[i, j]
            for k, (lab, g) in enumerate(groups):
                ax.scatter(k + RNG.uniform(-0.18, 0.18, len(g)), g[r], s=4.5, color=COL[r], lw=0, alpha=0.6)
                ax.plot([k - 0.3, k + 0.3], [g[r].median()] * 2, color=INK, lw=1.4, solid_capstyle="butt")
            ax.set_xticks(range(3)); ax.set_xlim(-0.55, 2.55)
            ax.set_xticklabels([g[0] for g in groups] if i == 1 else [], rotation=90, fontsize=8)
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
            ax.tick_params(axis="y", labelsize=8, pad=2)
            ax.tick_params(axis="x", pad=3)
            p = STATS["replicates"][str(s)][r]["kruskal_p"]
            head = short[r] + "\n" if i == 0 else ""
            ax.set_title(head + f"p = {p:.2f}", fontsize=8, loc="left", pad=8, linespacing=1.15)
            ax.set_ylabel(UNIT[r], fontsize=8, labelpad=3)
        # row label: severity, set apart from the unit of the first column
        fig.text(0.115 / W, (bottom + (1 - i) * (ah + row_gap) + ah / 2) / HT, f"s = {s:.1f}", rotation=90,
                 fontsize=9, fontweight="bold", color=INK, ha="center", va="center")
    for j in range(7):                               # one x position for the unit label of both rows
        fig.align_ylabels(list(axs[:, j]))
    save(fig, "S2_Fig")


if __name__ == "__main__":
    import sys
    # The order of ALL is the order of the draws from RNG and must stay as it is: s3 comes fourth (its
    # bootstrap resamples are the fourth block of draws) and s2 last.
    ALL = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "s3": s3, "fig6": fig6, "fig7": fig7, "s4": s4,
           "s1": s1, "s2": s2}
    want = sys.argv[1:] or list(ALL)              # no argument: every figure
    unknown = [n for n in want if n not in ALL]
    if unknown:
        sys.exit(f"unknown figure: {', '.join(unknown)} (choose from {', '.join(sorted(ALL))})")
    # All figures take their bootstrap resamples and jitter from the one generator RNG, so the intervals a
    # figure shows depend on the draws made by the figures before it (in the order of ALL when all are
    # drawn). To write the same file whether one figure or all are requested, every figure up to the last
    # requested one is drawn in the order of ALL; those not requested are drawn and closed, not written.
    _write = save
    order = list(ALL)
    for name in order[:max(order.index(n) for n in want) + 1]:
        save = _write if name in want else (lambda fig, _name: plt.close(fig))
        ALL[name]()
    save = _write
