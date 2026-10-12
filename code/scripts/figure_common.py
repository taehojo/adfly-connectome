"""Style, colours, readout names and helpers shared by figure scripts, and a small-print set of the battery figures.

  python code/scripts/analyze_battery.py      # first: writes results/stats/
  python code/scripts/figure_common.py

make_fig4_burden.py, make_fig5_rescue.py, exp1_stats.py and exp2_stats.py import this module, which creates figures/scratch/.
Run as a script it writes eight figures of its own there (PNG at 300 dpi, PDF, TIFF); the manuscript's come from make_figures.py.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from scipy import stats as st

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())  # code/scripts locally, scripts/ on the server
BAT = ROOT / "results/battery"
STATS_DIR = ROOT / "results/stats"
OUT = ROOT / "figures/scratch"
OUT.mkdir(parents=True, exist_ok=True)
STATS = json.loads((STATS_DIR / "battery_stats.json").read_text())
RNG = np.random.default_rng(7)

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "figure.dpi": 150, "pdf.fonttype": 42, "axes.titlepad": 3,
    "axes.labelpad": 2, "legend.frameon": False, "legend.handlelength": 1.2,
})
INK, GREY, LIGHT = "#1d2329", "#8a9096", "#d5d9dc"
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
ANCHOR = 0.55     # severity at which Kenyon cell loss matches the ~17% reported at 40 days


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


def panel(ax, letter, dx=-0.32, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left",
            color=INK)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04)
    im = Image.open(OUT / f"{name}.png").convert("RGB")
    im.save(OUT / f"{name}.tif", compression="tiff_lzw", dpi=(300, 300))
    plt.close(fig)
    print("wrote", name)


def pct_ci(x, h, n=3000):
    x, h = np.asarray(pd.Series(x).dropna(), float), np.asarray(pd.Series(h).dropna(), float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    est = np.median(x) / np.median(h) * 100
    r = (np.median(RNG.choice(x, (n, len(x))), 1) / np.median(RNG.choice(h, (n, len(h))), 1) * 100)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return est, lo, hi


def jitter(n, w=0.018):
    return RNG.uniform(-w, w, n)


# ============================================================ model and protocol
def draw_model():
    fig = plt.figure(figsize=(7.5, 4.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.62, 1.0], width_ratios=[1.35, 1.0, 1.0],
                          hspace=0.38, wspace=0.42)
    # A: pipeline schematic
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 100); ax.set_ylim(0, 21); ax.axis("off")
    panel(ax, "A", dx=-0.005, dy=0.93)
    boxes = [
        (1, "Connectome", "MaleCNS v1.0\n166,700 neurons\n24.5 million signed edges\n120 million synaptic contacts"),
        (26, "Whole-CNS LIF engine", "Shiu et al. parameters\nmatches Brian2 (r = 0.9997)\n"
                                     "+ feedforward antennal lobe\n+ spike-frequency adaptation"),
        (51, "Pathology in each fly", "load l = s × u (u: connectivity rank)\nthreshold lowered by 2 mV × l\n"
                                      "synapse loss, p = 0.6 × l\nneuron loss, p = 0.5 × l"),
        (76, "Readouts", "12 circuit readouts\n4 free-behaviour readouts\n1,136 simulated flies\n"
                         "each seeded and reproducible"),
    ]
    for x, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.2), 22.6, 17.2, boxstyle="round,pad=0.25,rounding_size=1.0",
                                    fc="#f3f5f7", ec=GREY, lw=0.7))
        ax.text(x + 11.3, 16.0, title, ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK)
        ax.text(x + 11.3, 8.2, body, ha="center", va="center", fontsize=6.3, color=INK, linespacing=1.5)
    for x in (23.9, 48.9, 73.9):
        ax.add_patch(FancyArrowPatch((x, 9.8), (x + 1.9, 9.8), arrowstyle="-|>", mutation_scale=8, color=GREY,
                                     lw=0.9))

    # B: protocol timeline
    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "B", dx=-0.08)
    segs = [("A", 0, 1, COL["LH_hz_A"]), ("B", 2, 1, COL["LH_hz_A"]),
            ("S", 4, 0.5, COL["MN9_sugar_hz"]), ("L", 5.5, 0.3, COL["GF_loom_hz"]),
            ("A+D", 6.8, 1, COL["LI"]), ("B", 9.8, 1, COL["LI"]),
            ("A", 12.8, 1, COL["LI"]), ("B", 14.8, 1, COL["LI"])]
    for lab, t0, dur, c in segs:
        ax.add_patch(Rectangle((t0, 0.6), dur, 0.42, fc=c, ec="none", alpha=0.9))
        ax.text(t0 + dur / 2, 1.1, lab, ha="center", va="bottom", fontsize=5.8, color=INK)
    brackets = [(0, 3, "naive odour\nresponses"), (4, 4.5, "MN9"), (5.5, 5.8, "GF"),
                (6.8, 10.8, "training"), (12.8, 15.8, "memory test\n(learning index)")]
    for a, b_, lab in brackets:
        ax.plot([a, a, b_, b_], [0.5, 0.42, 0.42, 0.5], color=GREY, lw=0.6)
        ax.text((a + b_) / 2, 0.32, lab, ha="center", va="top", fontsize=5.6, color="#4a5057", linespacing=1.2)
    ax.text(0, 1.62, "A, B: odours   S: sugar   L: looming (left)   D: dopaminergic PAM drive",
            fontsize=5.4, color="#4a5057", va="bottom")
    ax.set_xlim(-0.2, 16.2); ax.set_ylim(-0.45, 1.95)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 4, 8, 12, 16]); ax.set_xlabel("Time in protocol (s)")

    # C: load against connectivity rank under the three maps (rule illustration, s = 1)
    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "C")
    u = np.linspace(0, 1, 400)
    ax.plot(u, u, color=MAPC["hub"], lw=1.4, label="Hub (default)")
    ax.scatter(RNG.uniform(0, 1, 260), RNG.uniform(0, 1, 260), s=2.5, color=MAPC["shuffled"], lw=0,
               alpha=0.8, label="Shuffled")
    ax.plot(u, 1 - u, color=MAPC["antihub"], lw=1.4, label="Antihub")
    ax.set_xlabel("Connectivity rank of neuron\n(percentile of synapse count)")
    ax.set_ylabel("Load at s = 1")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, columnspacing=0.8, handletextpad=0.3,
              markerscale=3)

    # D: burden against severity
    ax = fig.add_subplot(gs[1, 2])
    panel(ax, "D")
    b = SC.groupby("stage").agg(dead=("dead_frac", "mean"), kc=("kc_dead_frac", "mean")).reset_index()
    ax.axvspan(ANCHOR, 1.03, color="#eef0f2", lw=0, zorder=0)
    ax.text(0.79, 30.6, "extrapolation", ha="center", fontsize=5.8, color=GREY)
    ax.axhspan(16.0, 17.0, color="#f2c9b4", lw=0)
    ax.text(0.02, 18.2, "Kenyon cell loss\nat 40 days, Aβ42 flies", fontsize=5.6, color="#9a4a24", va="bottom")
    ax.plot(b.stage, 100 * b.kc, "-o", ms=2.6, color=COL["KC_hz_A"], label="Kenyon cells")
    ax.plot(b.stage, 100 * b.dead, "-o", ms=2.6, color=INK, label="All neurons")
    ax.set_xlabel("Severity s"); ax.set_ylabel("Neurons removed (%)")
    ax.set_xlim(-0.02, 1.03); ax.set_ylim(0, 33)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.18))
    save(fig, "model")


# ============================================================ dose response
def draw_dose_response():
    reads = ["brain_spikes_A", "LH_hz_A", "PN_hz_A", "KC_hz_A", "KC_frac_A",
             "MBON_hz_A", "V_pre_A", "LI", "MN9_sugar_hz", "GF_loom_hz"]
    fig, axs = plt.subplots(2, 5, figsize=(7.5, 3.6))
    fig.subplots_adjust(wspace=0.55, hspace=0.62)
    trend = STATS["series"]["trend"]
    for k, (ax, r) in enumerate(zip(axs.flat, reads)):
        y = SC[r] * (100 if r == "KC_frac_A" else 1)
        ax.axvspan(ANCHOR, 1.04, color="#eef0f2", lw=0, zorder=0)
        ax.scatter(SC.stage + jitter(len(SC)), y, s=2.2, color=COL[r], alpha=0.45, lw=0, zorder=2)
        med = SC.assign(y=y).groupby("stage").y.median()
        ax.plot(med.index, med.values, color=INK, lw=1.0, zorder=3)
        ax.set_title(NAME[r], loc="left", fontsize=7)
        ax.set_ylabel(UNIT[r], fontsize=6.3)
        ax.text(0.97, 0.97, f"ρ = {trend[r]['rho']:+.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=6.2, color="#4a5057")
        ax.set_xticks([0, 0.5, 1]); ax.set_xlim(-0.04, 1.04)
        if k >= 5:
            ax.set_xlabel("Severity s")
        else:
            ax.set_xticklabels([])
        panel(ax, "ABCDEFGHIJ"[k], dx=-0.42, dy=1.08)
    save(fig, "dose_response")


# ============================================================ rules, placement, scope
def _dots(ax, xs, groups, ref, r, colors, offs=(-0.14, 0.14)):
    for x, (g05, g10), c in zip(xs, groups, colors):
        for g, off, fill in ((g05, offs[0], True), (g10, offs[1], False)):
            e, lo, hi = pct_ci(g[r], ref[r])
            if np.isnan(e):
                continue
            ax.plot([x + off] * 2, [lo, hi], color=c, lw=0.9)
            ax.plot(x + off, e, "o", ms=3.4, mfc=c if fill else "white", mec=c, mew=0.9)


def draw_rules_maps_scope():
    reads = ["brain_spikes_A", "MBON_hz_A", "LI", "MN9_sugar_hz", "GF_loom_hz"]
    fig, axs = plt.subplots(3, 5, figsize=(7.5, 5.6))
    fig.subplots_adjust(wspace=0.38, hspace=0.62, top=0.9)
    conds = ["all", "no_hyper", "hyper_only", "synloss_only", "death_only"]
    maps = ["hub", "shuffled", "antihub"]
    rows = [
        ("A", "Each rule alone and in combination", list(range(5)),
         [(CO[CO.condition == f"{c}_s0.5"], CO[CO.condition == f"{c}_s1"]) for c in conds],
         [RULEC[c] for c in conds], [RULEN[c].replace("\n", " ") for c in conds], (-8, 265)),
        ("B", "Same number of neurons removed, different neurons", list(range(3)),
         [(VU[VU.condition == f"{m}_s0.5"], VU[VU.condition == f"{m}_s1"]) for m in maps],
         [MAPC[m] for m in maps], ["Hub (load rises with connectivity)", "Shuffled", "Antihub (load falls with connectivity)"],
         (-5, 112)),
        ("C", "Pathology confined to Kenyon cells", [0, 1],
         [(SP[SP.stage == 0.5], SP[SP.stage == 1.0]), (SC[SC.stage == 0.5], SC[SC.stage == 1.0])],
         ["#4f8f3a", INK], ["Kenyon cells only", "Pan-neuronal"], (-5, 125)),
    ]
    for i, (letter, title, xs, groups, colors, labels, ylim) in enumerate(rows):
        for j, r in enumerate(reads):
            ax = axs[i, j]
            ax.axhline(100, color=GREY, lw=0.6, ls="--", zorder=0)
            _dots(ax, xs, groups, H, r, colors)
            ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
            ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6); ax.set_ylim(*ylim)
            ax.set_title(NAME[r], fontsize=6.8, loc="left")
            if j == 0:
                ax.set_ylabel("Median, % of healthy")
        bb = axs[i, 0].get_position()
        fig.text(0.035, bb.y1 + 0.045, letter, fontsize=9, fontweight="bold", color=INK, va="bottom")
        fig.text(0.065, bb.y1 + 0.047, title, fontsize=7.2, color=INK, va="bottom")
        handles = [plt.Line2D([], [], marker="o", ls="", ms=3.6, color=c, label=l) for c, l in zip(colors, labels)]
        if i == 0:
            handles += [plt.Line2D([], [], marker="o", ls="", ms=3.6, color=GREY, label="filled: s = 0.5"),
                        plt.Line2D([], [], marker="o", ls="", ms=3.6, mfc="white", mec=GREY, label="open: s = 1.0")]
        fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.985, bb.y1 + 0.038), ncol=len(handles),
                   fontsize=6, columnspacing=0.9, handletextpad=0.25, frameon=False)
    save(fig, "rules_maps_scope")


# ============================================================ learning index and MBON drive
def draw_learning():
    fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.6), gridspec_kw={"width_ratios": [1.2, 1.1, 1.0]})
    fig.subplots_adjust(wspace=0.62)
    ax = axs[0]
    cmap = mpl.colormaps["viridis_r"]
    sc_ = ax.scatter(SC.MBON_hz_A, SC.LI, c=SC.stage, cmap=cmap, vmin=0, vmax=1, s=4, lw=0, alpha=0.85)
    ax.set_xlabel("MBON rate during odour A (Hz)"); ax.set_ylabel("Learning index (signed sum, Hz)")
    cb = fig.colorbar(sc_, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Severity s", fontsize=6.3, labelpad=4); cb.ax.tick_params(labelsize=6)
    rho = STATS["series"]["learning"]["spearman_LI_MBON_all"][0]
    ax.text(0.97, 0.03, f"ρ = {rho:.2f}\n384 flies", transform=ax.transAxes, va="bottom", ha="right",
            fontsize=6.2)
    panel(ax, "A", dx=-0.3)
    ax = axs[1]
    ax.axhline(100, color=GREY, lw=0.6, ls="--")
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
        ax.plot(xs, es, "-o", ms=2.4, color=COL[r], label=lab)
    ax.set_xlabel("Severity s"); ax.set_ylabel("Median, % of healthy")
    ax.set_ylim(-5, 140); ax.set_xlim(-0.03, 1.04)
    ax.legend(loc="lower left", fontsize=6)
    panel(ax, "B", dx=-0.3)
    ax = axs[2]
    ax.axhline(100, color=GREY, lw=0.6, ls="--")
    conds = ["all", "no_hyper", "hyper_only", "synloss_only", "death_only"]
    for i, c in enumerate(conds):
        for s, off, fill in ((0.5, -0.15, True), (1.0, 0.15, False)):
            g = CO[CO.condition == f"{c}_s{s:g}"]["LI_norm"].dropna()
            if len(g) < 8:
                continue
            e, lo, hi = pct_ci(g, H["LI_norm"])
            ax.plot([i + off] * 2, [lo, hi], color=RULEC[c], lw=0.9)
            ax.plot(i + off, e, "o", ms=3.4, mfc=RULEC[c] if fill else "white", mec=RULEC[c], mew=0.9)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([RULEN[c] for c in conds], rotation=90, fontsize=5.8)
    ax.set_ylabel("Learning index / MBON rate\n(median, % of healthy)"); ax.set_ylim(0, 180)
    ax.plot([], [], "o", ms=3.4, color=INK, label="s = 0.5")
    ax.plot([], [], "o", ms=3.4, mfc="white", mec=INK, label="s = 1.0")
    ax.legend(loc="upper left", fontsize=6)
    panel(ax, "C", dx=-0.42)
    save(fig, "learning")


# ============================================================ discrete phenotypes from single-neuron loss
def draw_discrete():
    fig = plt.figure(figsize=(7.5, 4.7))
    gs = fig.add_gridspec(2, 3, hspace=0.66, wspace=0.5)
    onl = STATS["output_neuron_loss"]
    ax = fig.add_subplot(gs[0, 0])
    y = SC.GF_loom_hz
    fail = y < 100
    ax.scatter(SC.stage[~fail] + jitter((~fail).sum()), y[~fail], s=2.4, color=COL["GF_loom_hz"], alpha=0.45, lw=0)
    ax.scatter(SC.stage[fail] + jitter(fail.sum()), y[fail], s=5, color="#c0392b", lw=0, alpha=0.8)
    ax.axhline(100, color=GREY, lw=0.6, ls="--")
    ax.set_xlabel("Severity s"); ax.set_ylabel("Giant fibre rate (Hz)"); ax.set_ylim(-8, 262)
    ax.set_title("Escape: giant fibre", loc="left", fontsize=7)
    t = onl["gf_fail_by_left_removed"]
    ax.text(0.02, 0.99, f"left GF removed: {t['True|True']} of {t['True|True'] + t['True|False']} fail\n"
            f"left GF kept: {t['False|True']} of {t['False|True'] + t['False|False']} fail",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.8, color="#4a5057")
    panel(ax, "A", dx=-0.32)
    ax = fig.add_subplot(gs[0, 1])
    y = SC.MN9_sugar_hz
    fail = y == 0
    ax.scatter(SC.stage[~fail] + jitter((~fail).sum()), y[~fail], s=2.4, color=COL["MN9_sugar_hz"], alpha=0.45, lw=0)
    ax.scatter(SC.stage[fail] + jitter(fail.sum()), y[fail] + RNG.uniform(-0.6, 0.6, fail.sum()), s=5,
               color="#c0392b", lw=0, alpha=0.8)
    ax.set_xlabel("Severity s"); ax.set_ylabel("MN9 feeding drive (Hz)"); ax.set_ylim(-2, 56)
    ax.set_title("Feeding: MN9", loc="left", fontsize=7)
    m = onl["mn9_silent_by_n_removed"]
    sil = m["0|True"] + m["1|True"] + m["2|True"]
    one = m["1|True"] + m["1|False"]
    ax.text(0.98, 0.99, f"silent: {sil} flies, {m['1|True'] + m['2|True']} of them\nlacking ≥1 MN9 neuron\n"
            f"one MN9 lost, still active: {m['1|False']} of {one}",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color="#4a5057")
    panel(ax, "B", dx=-0.32)
    ax = fig.add_subplot(gs[0, 2])
    mf = STATS["series"]["motor_failures"]
    xs = [s for s in STAGES if s > 0]
    ax.plot(xs, [mf["GF_loom_hz"][str(s)]["n_fail"] for s in xs], "-o", ms=2.6, color=COL["GF_loom_hz"],
            label="GF below 100 Hz")
    ax.plot(xs, [mf["MN9_sugar_hz"][str(s)]["n_fail"] for s in xs], "-o", ms=2.6, color=COL["MN9_sugar_hz"],
            label="MN9 silent")
    ax.set_xlabel("Severity s"); ax.set_ylabel("Flies failing (of 32)")
    ax.set_ylim(0, 21); ax.legend(loc="upper left")
    panel(ax, "C", dx=-0.32)
    ax = fig.add_subplot(gs[1, 0])
    low = SC[SC.stage <= 0.3]
    cmap = mpl.colormaps["viridis_r"]
    sc_ = ax.scatter(low.KC_hz_A, low.MBON_hz_A, c=low.stage, cmap=cmap, vmin=0, vmax=1, s=5, lw=0, alpha=0.9)
    cb = fig.colorbar(sc_, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Severity s", fontsize=6.0, labelpad=4); cb.ax.tick_params(labelsize=5.8)
    ax.set_xlabel("Kenyon cell rate (Hz)"); ax.set_ylabel("MBON rate (Hz)")
    ax.set_title("Severity 0 to 0.3", loc="left", fontsize=7)
    ax.annotate("high-activity flies", xy=(1.42, 5.6), xytext=(0.32, 6.6), fontsize=5.8, color="#4a5057",
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    panel(ax, "D", dx=-0.32)
    ax = fig.add_subplot(gs[1, 1])
    groups = [("Healthy", H, GREY), ("Threshold\nshift only", CO[CO.condition == "hyper_only_s0.5"], RULEC["hyper_only"]),
              ("Synapse\nloss only", CO[CO.condition == "synloss_only_s0.5"], RULEC["synloss_only"]),
              ("Neuron\nloss only", CO[CO.condition == "death_only_s0.5"], RULEC["death_only"]),
              ("All three", CO[CO.condition == "all_s0.5"], RULEC["all"])]
    for i, (lab, g, c) in enumerate(groups):
        ax.scatter(i + RNG.uniform(-0.18, 0.18, len(g)), g.KC_hz_A, s=3, color=c, lw=0, alpha=0.7)
        ax.plot([i - 0.25, i + 0.25], [g.KC_hz_A.median()] * 2, color=INK, lw=1.0)
    ax.axhline(H.KC_hz_A.max(), color=GREY, lw=0.6, ls=":")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], rotation=90, fontsize=5.8)
    ax.set_ylabel("Kenyon cell rate (Hz)"); ax.set_title("Severity 0.5", loc="left", fontsize=7)
    panel(ax, "E", dx=-0.32)
    ax = fig.add_subplot(gs[1, 2])
    ko = pd.DataFrame(STATS["series"]["kc_outliers"])
    ax.bar(ko.stage, ko.n_both, width=0.035, color=COL["KC_hz_A"])
    ax.set_xlabel("Severity s"); ax.set_ylabel("High-activity flies (of 32)")
    ax.set_ylim(0, 10); ax.set_xlim(0, 1.04)
    panel(ax, "F", dx=-0.32)
    save(fig, "discrete")


# ============================================================ which readouts separate first
def draw_separation():
    fig = plt.figure(figsize=(7.5, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.5, width_ratios=[1, 1.1])
    au = pd.read_csv(STATS_DIR / "series_auc.csv")
    order = ["brain_spikes_A", "LH_hz_A", "MBON_hz_A", "KC_hz_A", "KC_frac_A", "V_pre_A", "MN9_sugar_hz",
             "GF_loom_hz", "LI", "PN_hz_A"]
    hi_ = ["brain_spikes_A", "LH_hz_A", "MN9_sugar_hz", "GF_loom_hz", "LI"]
    ax = fig.add_subplot(gs[0, 0])
    ax.axhline(0.8, color=GREY, lw=0.6, ls="--")
    for r in order:
        if r in hi_:
            continue
        g = au[au.readout == r].sort_values("stage")
        ax.plot(g.stage, g.auc, "-", color=LIGHT, lw=0.9, zorder=1)
    for r in hi_:
        g = au[au.readout == r].sort_values("stage")
        ax.plot(g.stage, g.auc, "-o", ms=2.2, color=COL[r], lw=1.2, label=NAME[r], zorder=3)
    ax.plot([], [], "-", color=LIGHT, label="other circuit readouts")
    ax.set_xlabel("Severity s"); ax.set_ylabel("Separation from healthy (AUC)")
    ax.set_ylim(0.48, 1.02); ax.set_xlim(0, 1.03)
    ax.legend(loc="lower right", fontsize=6)
    panel(ax, "A", dx=-0.22)
    ax = fig.add_subplot(gs[0, 1])
    fd = STATS["series"]["first_detection"]
    rows = [r for r in order + ["LI_norm"] if fd[r]["boot_median"] is not None]
    rows = sorted(rows, key=lambda r: (fd[r]["boot_median"], fd[r]["boot_ci95"][1]))
    for i, r in enumerate(rows):
        lo, hi = fd[r]["boot_ci95"]
        c = COL.get(r, INK)
        ax.plot([lo, hi], [i, i], color=c, lw=2.2, alpha=0.45, solid_capstyle="round")
        ax.plot(fd[r]["boot_median"], i, "o", ms=4, color=c)
        if fd[r]["frac_never"] > 0.05:
            ax.text(hi + 0.03, i, f"not reached in {100 * fd[r]['frac_never']:.0f}%", va="center", fontsize=5.6,
                    color="#4a5057")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([NAME[r] for r in rows], fontsize=6.2)
    ax.invert_yaxis(); ax.set_xlim(0, 1.2); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("First severity with AUC ≥ 0.80")
    panel(ax, "B", dx=-0.62)
    # C: change, spread and AUC at s = 0.10 for each readout
    sub = gs[1, 0].subgridspec(1, 3, wspace=0.12, width_ratios=[1, 1, 1])
    pr = STATS["series"]["precision"]
    rr = sorted(pr, key=lambda r: -pr[r]["auc_0.1"])
    yy = np.arange(len(rr))
    cols = [("Median change\nat s = 0.10 (%)", lambda v: abs(v["median_shift_pct_0.1"]), (0, 28)),
            ("Spread in healthy\nflies (robust CV, %)", lambda v: 100 * v["healthy_rcv"], (0, 36)),
            ("AUC at\ns = 0.10", lambda v: v["auc_0.1"], (0.45, 0.9))]
    for k, (lab, fn, xl) in enumerate(cols):
        ax = fig.add_subplot(sub[0, k])
        for i, r in enumerate(rr):
            v = fn(pr[r])
            motor = r in ("MN9_sugar_hz", "GF_loom_hz")
            ax.plot([xl[0], v], [i, i], color=LIGHT, lw=0.8, zorder=1)
            ax.plot(v, i, "s" if motor else "o", ms=3.6, color=COL[r], zorder=2)
        ax.set_ylim(len(rr) - 0.5, -0.5); ax.set_xlim(*xl)
        ax.set_xlabel(lab, fontsize=6.0)
        ax.tick_params(axis="x", labelsize=5.8)
        if k == 0:
            ax.set_yticks(yy); ax.set_yticklabels([NAME[r] for r in rr], fontsize=6.0)
            panel(ax, "C", dx=-1.55)
        else:
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
    ax = fig.add_subplot(gs[1, 1])
    ja = pd.read_csv(STATS_DIR / "joint_auc.csv")
    ax.axhline(0.8, color=GREY, lw=0.6, ls="--")
    g = ja[ja.readout == "brain_spikes_A"].sort_values("stage")
    ax.fill_between(g.stage, g.lo, g.hi, color=COL["brain_spikes_A"], alpha=0.12, lw=0)
    for r, ls in (("brain_spikes_A", "-"), ("LH_hz_A", "-"), ("LI", "-"), ("behaviour_feed_s", "--"),
                  ("behaviour_esc", "--"), ("behaviour_near_A_s", ":"), ("behaviour_PI", ":")):
        g = ja[ja.readout == r].sort_values("stage")
        ax.plot(g.stage, g.auc, ls, marker="o", ms=2.2, color=COL[r], lw=1.1, label=NAME[r])
    ax.set_xlabel("Severity s"); ax.set_ylabel("Separation from healthy (AUC)")
    ax.set_ylim(0.45, 1.02); ax.set_xlim(0.03, 0.52)
    ax.set_title("Same 176 flies: circuit and arena readouts", loc="left", fontsize=7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=5.8, ncol=2, columnspacing=1.0)
    panel(ax, "D", dx=-0.22)
    save(fig, "separation")


# ============================================================ engine validation
def draw_engine_validation():
    z = np.load(ROOT / "results/validation/engine_validation_rates.npz")
    ev = json.loads((ROOT / "results/validation/engine_validation.json").read_text())["comparisons"]
    fig, axs = plt.subplots(2, 2, figsize=(5.2, 5.0))
    fig.subplots_adjust(wspace=0.35, hspace=0.42)
    floor = 0.25
    for i, (gate, gname) in enumerate((("ours_gate1", "Refractory input discarded"),
                                       ("ours_gate0", "Refractory input accumulated"))):
        for j, (ref, rname, key) in enumerate((("ref_local_brian2_2.10.1", "Brian2 2.10.1, run locally", "local_brian2_2.10.1"),
                                               ("ref_released_example_output", "Released example output", "released_example_output"))):
            ax = axs[i, j]
            x, y = z[ref], z[gate]
            m = (x > 0) | (y > 0)
            ax.scatter(np.maximum(x[m], floor), np.maximum(y[m], floor), s=3, lw=0, alpha=0.6,
                       color="#1f4e79" if i == 0 else "#d9711c")
            ax.plot([floor, 400], [floor, 400], color=LIGHT, lw=0.8, zorder=0)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.2, 400); ax.set_ylim(0.2, 400)
            c = ev[f"gate={'True' if i == 0 else 'False'} vs {key}"]
            ax.text(0.05, 0.95, f"r = {c['pearson_r']:.4f}\nslope = {c['slope']:.3f}\nn = {c['n_compared']}",
                    transform=ax.transAxes, va="top", fontsize=6.2)
            ax.set_xlabel(f"{rname} (Hz)", fontsize=6.4)
            if j == 0:
                ax.set_ylabel(f"This engine (Hz)\n{gname}", fontsize=6.4)
            panel(ax, "ABCD"[2 * i + j], dx=-0.3)
    save(fig, "engine_validation")


# ============================================================ independent groups of the default model
def draw_replicates():
    reads = ["brain_spikes_A", "LH_hz_A", "MBON_hz_A", "LI", "MN9_sugar_hz", "GF_loom_hz"]
    fig, axs = plt.subplots(2, 6, figsize=(7.5, 3.3))
    fig.subplots_adjust(wspace=0.6, hspace=0.55)
    for i, s in enumerate((0.5, 1.0)):
        groups = [("Severity\nseries", SC[SC.stage == s]), ("Rule\nablation", CO[CO.condition == f"all_s{s:g}"]),
                  ("Map\ncomparison", VU[VU.condition == f"hub_s{s:g}"])]
        for j, r in enumerate(reads):
            ax = axs[i, j]
            for k, (lab, g) in enumerate(groups):
                ax.scatter(k + RNG.uniform(-0.18, 0.18, len(g)), g[r], s=2.6, color=COL[r], lw=0, alpha=0.6)
                ax.plot([k - 0.25, k + 0.25], [g[r].median()] * 2, color=INK, lw=1.0)
            ax.set_xticks(range(3))
            ax.set_xticklabels([g[0] for g in groups] if i == 1 else [], rotation=90, fontsize=5.6)
            p = STATS["replicates"][str(s)][r]["kruskal_p"]
            head = NAME[r] + "\n" if i == 0 else ""
            ax.set_title(head + f"p = {p:.2f}", fontsize=6.2, loc="left")
            if j == 0:
                ax.set_ylabel(f"s = {s:.1f}\n{UNIT[r]}", fontsize=6.3)
            else:
                ax.set_ylabel(UNIT[r], fontsize=6.0)
    save(fig, "replicates")


if __name__ == "__main__":
    draw_model(); draw_dose_response(); draw_rules_maps_scope(); draw_learning(); draw_discrete(); draw_separation(); draw_engine_validation(); draw_replicates()
