"""Experiment 2 (vulnerability maps at matched synaptic burden): summary statistics and a summary figure.

  python code/scripts/exp2_stats.py

Reads results/exp2/ (calibration.csv, matched.json, battery.csv, summary.json)
and, for the neuron-matched comparison of the paper, results/battery/
(vulnerability_x4.csv with the healthy flies of stage_curve_x4.csv). Writes
results/stats/exp2_stats.json, which facts_exp2.py reads as a cross-check, and draws
figures/scratch/exp2_summary.{png,pdf,tif}. Readouts are medians as % of the healthy median of the
same design, with 95% intervals from 4,000 bootstrap resamples of flies within groups.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())  # code/scripts locally, scripts/ on the server
sys.path.insert(0, str(ROOT / "code/scripts"))
import figure_common as mf  # noqa: E402  (style, colours, save helper)

E2 = ROOT / "results/exp2"
OUT_STATS = ROOT / "results/stats"
OUT_STATS.mkdir(parents=True, exist_ok=True)
mf.OUT = ROOT / "figures/scratch"
mf.OUT.mkdir(parents=True, exist_ok=True)

MAPS = ["hub", "shuffled", "antihub"]
MAPN = {"hub": "Hub", "shuffled": "Shuffled", "antihub": "Antihub"}
READ = ["brain_spikes_A", "LH_hz_A", "KC_hz_A", "MBON_hz_A", "LI", "MN9_sugar_hz", "GF_loom_hz"]
RNG = np.random.default_rng(9)
NB = 4000

d = pd.read_csv(E2 / "battery.csv")
d["reference"] = d["reference"].map(lambda x: "" if pd.isna(x) else f"{float(x):g}")
H = d[d["map"] == "healthy"]
cal = pd.read_csv(E2 / "calibration.csv").groupby(["map", "s"]).mean(numeric_only=True).reset_index()
matched = json.loads((E2 / "matched.json").read_text())
summ = json.loads((E2 / "summary.json").read_text())
REFS = [k for k in matched["maps"]]
VU = pd.read_csv(ROOT / "results/battery/vulnerability_x4.csv")
SC = pd.read_csv(ROOT / "results/battery/stage_curve_x4.csv")
HS = SC[SC.stage == 0]


def pct(x, h):
    x, h = np.asarray(pd.Series(x).dropna(), float), np.asarray(pd.Series(h).dropna(), float)
    b = 100 * np.median(RNG.choice(x, (NB, len(x))), 1) / np.median(RNG.choice(h, (NB, len(h))), 1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"pct": round(float(100 * np.median(x) / np.median(h)), 1), "lo": round(float(lo), 1),
            "hi": round(float(hi), 1)}


def burden_at(m, s):
    return round(100 * float(cal[(cal["map"] == m) & np.isclose(cal["s"], s)].syn_lost_frac.iloc[0]), 1)


stats = {"n_flies": int(len(d)), "n_healthy": int(len(H)), "n_per_condition": int(d[d["map"] != "healthy"].groupby(
    ["reference", "map"]).size().min()),
    "neuron_matched_burden_pct": {f"s{s:g}": {m: burden_at(m, s) for m in MAPS} for s in (0.5, 1.0)},
    "neuron_matched_brain_pct_s0.5": {m: pct(VU[VU.condition == f"{m}_s0.5"].brain_spikes_A, HS.brain_spikes_A)
                                      for m in MAPS},
    "antihub_max_burden_pct": burden_at("antihub", 1.0),
    "levels": {}}
for ref in REFS:
    g = d[d.reference == ref]
    lv = {"target_pct": round(100 * matched["maps"][ref]["target_syn_lost_frac"], 1), "maps": {},
          "kruskal_p": {}, "mw_p": {}, "primary": summ["references"][ref].get("primary")}
    here = [m for m in MAPS if (g["map"] == m).any()]
    for m in here:
        x = g[g["map"] == m]
        lv["maps"][m] = {"s": matched["maps"][ref][m]["s"], "n": int(len(x)),
                         "syn_lost_pct": round(100 * float(x.syn_lost_frac.median()), 1),
                         "dead_pct": round(100 * float(x.dead_frac.median()), 1),
                         "kc_dead_pct": round(100 * float(x.kc_dead_frac.median()), 1),
                         "mean_load": round(float(x.mean_load.median()), 3),
                         **{r: pct(x[r], H[r]) for r in READ}}
    for r in READ:
        lv["kruskal_p"][r] = float(st.kruskal(*[g[g["map"] == m][r].dropna() for m in here]).pvalue)
        for m in here[1:]:
            lv["mw_p"][f"hub_vs_{m}|{r}"] = float(st.mannwhitneyu(g[g["map"] == "hub"][r].dropna(),
                                                                  g[g["map"] == m][r].dropna()).pvalue)
    lv["unreachable"] = [m for m in MAPS if m not in here]
    stats["levels"][ref] = lv
(OUT_STATS / "exp2_stats.json").write_text(json.dumps(stats, indent=1))
print("wrote results/stats/exp2_stats.json")


# ------------------------------------------------------------------ summary figure (not a figure of the manuscript)
def summary_figure():
    fig = plt.figure(figsize=(7.5, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1], hspace=0.62, wspace=0.62, left=0.08, right=0.985,
                          top=0.95, bottom=0.09)
    # A: synaptic burden against severity for each map
    ax = fig.add_subplot(gs[0, :2])
    for m in MAPS:
        c = cal[cal["map"] == m].sort_values("s")
        ax.plot(c.s, 100 * c.syn_lost_frac, color=mf.MAPC[m], label=MAPN[m])
    ax.axvline(0.5, color=mf.GREY, lw=0.6, ls="--")
    for m in MAPS:
        y = burden_at(m, 0.5)
        ax.plot([0.5], [y], "o", ms=3.5, color=mf.MAPC[m])
        ax.text(0.47, y, f"{y:.0f}%", ha="right", va="center", fontsize=6, color=mf.MAPC[m])
    for ref in REFS:
        t = 100 * matched["maps"][ref]["target_syn_lost_frac"]
        ax.axhline(t, color=mf.LIGHT, lw=0.6, ls=":", zorder=0)
        for m in MAPS:
            s = matched["maps"][ref][m]["s"]
            if s is not None:
                ax.plot([s], [t], "s", ms=3, mfc="white", mec=mf.MAPC[m], mew=0.9)
    ax.text(0.515, 3, "equal neurons\nremoved (s = 0.5)", fontsize=5.8, color=mf.GREY, va="bottom")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 82)
    ax.set_xlabel("Severity s")
    ax.set_ylabel("Synaptic contacts lost (%)")
    ax.legend(loc="upper left", fontsize=6)
    mf.panel(ax, "A", dx=-0.15, dy=1.04)
    # B: whole-brain activity, equal neurons against equal synapses
    ax = fig.add_subplot(gs[0, 2:])
    groups = [("Equal neurons\n11.9%", None)] + [(f"Equal synapses\n{100 * matched['maps'][r]['target_syn_lost_frac']:.1f}%",
                                            r) for r in REFS]
    for i, (lab, ref) in enumerate(groups):
        for j, m in enumerate(MAPS):
            if ref is None:
                y, h = VU[VU.condition == f"{m}_s0.5"].brain_spikes_A, HS.brain_spikes_A
            else:
                g = d[(d.reference == ref) & (d["map"] == m)]
                if g.empty:
                    continue
                y, h = g.brain_spikes_A, H.brain_spikes_A
            y = 100 * y / np.median(h)
            x = i + (j - 1) * 0.25 + RNG.uniform(-0.05, 0.05, len(y))
            ax.scatter(x, y, s=2.5, color=mf.MAPC[m], alpha=0.35, lw=0)
            ax.plot([i + (j - 1) * 0.25 - 0.09, i + (j - 1) * 0.25 + 0.09], [np.median(y)] * 2, color=mf.MAPC[m],
                    lw=1.6)
    ax.axhline(100, color=mf.GREY, lw=0.5, ls=":")
    ax.axvline(0.5, color=mf.LIGHT, lw=0.6)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=5.8)
    ax.set_ylabel("Whole-brain activity (% of healthy)")
    ax.set_ylim(0, 125)
    mf.panel(ax, "B", dx=-0.17, dy=1.04)
    # C: lateral horn, MBON rate and learning index across matched burdens
    levels = [100 * matched["maps"][r]["target_syn_lost_frac"] for r in REFS]
    for k, r in enumerate(["LH_hz_A", "MBON_hz_A", "LI"]):
        ax = fig.add_subplot(gs[1, k])
        for j, m in enumerate(MAPS):
            xs, ys, lo, hi = [], [], [], []
            for lv, ref in zip(levels, REFS):
                v = stats["levels"][ref]["maps"].get(m)
                if v is None:
                    continue
                xs.append(lv + (j - 1) * 1.2); ys.append(v[r]["pct"]); lo.append(v[r]["lo"]); hi.append(v[r]["hi"])
            ys, lo, hi = np.array(ys), np.array(lo), np.array(hi)
            ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], color=mf.MAPC[m], marker="o", ms=2.5, lw=0.9,
                        elinewidth=0.6, capsize=1.2)
        ps = [stats["levels"][ref]["kruskal_p"][r] for ref in REFS]
        ax.set_title(mf.NAME[r] + "\n" + "Kruskal-Wallis p: " + ", ".join(
            "<0.001" if p < 0.001 else f"{p:.2f}" for p in ps), fontsize=6, loc="left")
        ax.set_xticks([round(l, 1) for l in levels], [f"{l:.1f}" for l in levels])
        ax.set_xlabel("Contacts lost (%)", fontsize=6.3)
        ax.set_ylim(0, 115)
        if k == 0:
            ax.set_ylabel("% of healthy")
            mf.panel(ax, "C", dx=-0.42, dy=1.12)
    # D: neurons removed at matched synaptic burden
    ax = fig.add_subplot(gs[1, 3])
    for j, m in enumerate(MAPS):
        xs, ys = [], []
        for lv, ref in zip(levels, REFS):
            v = stats["levels"][ref]["maps"].get(m)
            if v is not None:
                xs.append(lv); ys.append(v["dead_pct"])
        ax.plot(xs, ys, "-o", ms=2.5, color=mf.MAPC[m], label=MAPN[m])
    ax.set_xticks([round(l, 1) for l in levels], [f"{l:.1f}" for l in levels])
    ax.set_xlabel("Contacts lost (%)", fontsize=6.3)
    ax.set_ylabel("Neurons removed (%)")
    ax.set_title("Neurons removed at\nmatched synaptic loss", fontsize=6, loc="left")
    mf.panel(ax, "D", dx=-0.42, dy=1.12)
    mf.save(fig, "exp2_summary")


summary_figure()
