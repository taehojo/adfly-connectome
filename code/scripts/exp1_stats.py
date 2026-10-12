"""Experiment 1 (counterfactual rescue): summary statistics and a summary figure.

  python code/scripts/exp1_stats.py

Reads results/exp1/ (screen, robust, confirm and the summary tables written by
analyze_exp1.py), writes results/stats/exp1_stats.json, which facts_battery.py and
facts_exp1.py read, and draws figures/scratch/exp1_summary.{png,pdf,tif}. Restored fraction is
(y_variant - y_damaged) / (y_healthy - y_damaged) within one fly under identical input;
readouts whose deficit is under 5% of the healthy value in a fly are left out for it.
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())  # code/scripts locally, scripts/ on the server
sys.path.insert(0, str(ROOT / "code/scripts"))
import figure_common as mf  # noqa: E402  (style, colours, save helper)

E1 = ROOT / "results/exp1"
OUT_STATS = ROOT / "results/stats"
OUT_STATS.mkdir(parents=True, exist_ok=True)
mf.OUT = ROOT / "figures/scratch"
mf.OUT.mkdir(parents=True, exist_ok=True)

SEL = ["brain_spikes_A", "LH_hz_A", "KC_hz_A", "MBON_hz_A", "MN9_sugar_hz", "GF_loom_hz"]
SHORT = {"brain_spikes_A": "Whole\nbrain", "LH_hz_A": "Lateral\nhorn", "KC_hz_A": "Kenyon\ncells",
         "MBON_hz_A": "MBONs", "MN9_sugar_hz": "MN9", "GF_loom_hz": "Giant\nfibre", "LI": "Learning\nindex"}
MODELS = [("w0.8", "Weights ×0.8"), ("w1.2", "Weights ×1.2"), ("glu0.5", "Glutamate ×0.5"),
          ("da0.5", "Adaptation ×0.5"), ("noise1", "Input noise 1"), ("noise2", "Input noise 2")]
MIN_DEF = 0.05


def fractions(runs):
    out = []
    for seed, g in runs.groupby("seed"):
        b = g[g.level == "baseline"].set_index("unit")
        yd, yh = b.loc["damaged"], b.loc["healthy"]
        f = g.copy()
        for r in SEL:
            d = yh[r] - yd[r]
            ok = np.isfinite(d) and abs(d) > MIN_DEF * max(abs(yh[r]), 1e-9)
            f[r] = (g[r] - yd[r]) / d if ok else np.nan
        out.append(f)
    return pd.concat(out, ignore_index=True)


def q(s):
    s = pd.Series(s).dropna()
    return {"median": round(float(s.median()), 3), "q25": round(float(s.quantile(0.25)), 3),
            "q75": round(float(s.quantile(0.75)), 3), "n": int(len(s))}


runs = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(E1 / "screen/*_runs.csv")))], ignore_index=True)
F = fractions(runs)
B = F[F.level == "baseline"]
stats = {"n_flies": int(F.seed.nunique()), "n_runs": int(len(runs)),
         "flies_by_stage": {str(k): int(v) for k, v in F.groupby("stage").seed.nunique().items()}}

# deficit of the damaged fly, % of its own healthy copy
raw = runs[runs.level == "baseline"].pivot_table(index=["seed", "stage"], columns="unit", values=SEL)
stats["damaged_pct_of_healthy"] = {r: {str(s): round(float(100 * v), 1) for s, v in
                                       (raw[(r, "damaged")] / raw[(r, "healthy")]).groupby(level=1).median().items()}
                                   for r in SEL}

# 1. decomposition
stats["decomposition"] = {}
for st_ in ("all", 0.3, 0.5):
    bb = B if st_ == "all" else B[B.stage == st_]
    piv = bb.pivot_table(index="seed", columns="unit", values=SEL)
    stats["decomposition"][str(st_)] = {
        u: {r: q(piv[(r, u)]) for r in SEL} for u in ("restore_dead_all", "restore_syn_all", "no_shift")}
    stats["decomposition"][str(st_)]["sum_dead_syn"] = {
        r: q(piv[(r, "restore_dead_all")] + piv[(r, "restore_syn_all")]) for r in SEL}

# 2. single types and their concentration
T = F[F.level == "type"]
n_typ = runs[runs.level == "type"].groupby("seed").size()
meta = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(E1 / "screen/*.done.json")))]
stats["types"] = {"tested_per_fly": int(n_typ.median()), "damaged_types_per_fly": q([m["n_types_damaged"] for m in meta]),
                  "classes_per_fly": q([m["n_classes"] for m in meta]),
                  "removed_per_fly": q([m["n_removed"] for m in meta]),
                  "best_single_type_per_fly": {r: q(T.groupby("seed")[r].max()) for r in SEL},
                  "share_types_above_0.05": {r: round(float((T[r] > 0.05).mean()), 4) for r in SEL},
                  "share_types_above_0.10": {r: round(float((T[r] > 0.10).mean()), 4) for r in SEL}}
gf = T[T.unit == "DNp01"]
stats["types"]["giant_fibre_restored"] = {"n_flies": int(len(gf)), "values": [round(float(v), 3) for v in gf.GF_loom_hz]}
C = F[F.level == "cumulative"].copy()
C["r"] = C.unit.str.split(":").str[0]
C["k"] = C.unit.str.split("top").str[-1].astype(int)
C["frac"] = [row[row.r] for _, row in C.iterrows()]
dead = B[B.unit == "restore_dead_all"].set_index("seed")
stats["concentration"] = {}
for r in SEL:
    stats["concentration"][r] = {}
    for k in (1, 3, 10, 30):
        c = C[(C.r == r) & (C.k == k)].set_index("seed").frac
        d = dead[r].reindex(c.index)
        stats["concentration"][r][str(k)] = {"restored": q(c), "share_of_all_removed": q((c / d)[d > 0.1])}

# 3. classes (Banzhaf) and wiring
bz = pd.read_csv(E1 / "summary/class_banzhaf.csv")
stats["class_banzhaf_top"] = {r: bz[bz.readout == r].sort_values("median", ascending=False).head(3)[
    ["class", "median"]].round(3).values.tolist() for r in SEL}
stats["class_additivity_r2"] = {r: round(float(bz[bz.readout == r].additive_r2.median()), 3) for r in SEL}
gv = pd.read_csv(E1 / "summary/graph_vs_counterfactual.csv")
stats["wiring"] = {r: {"r2_loo": q(gv[gv.readout == r].r2_wiring_loo),
                       **{p: round(float(gv[gv.readout == r][f"rho_{p}"].median()), 3)
                          for p in ("n_lost", "syn_lost", "mean_rank", "direct", "twostep")}} for r in SEL}
do = pd.read_csv(E1 / "summary/dynamic_outliers.csv")
stats["wiring_outliers"] = {r: do[(do.readout == r) & (do.z_median >= 2) & (do.n_flies >= 10)][
    ["unit", "n_flies", "z_median", "frac_median"]].round(3).values.tolist() for r in SEL}

# 4. confirm (full protocol) and robust
cs = pd.read_csv(E1 / "summary/confirm_summary.csv")
nec = cs[cs.test.str.startswith("necessity")].set_index("unit")
suf = cs[cs.test.str.startswith("sufficiency")].set_index("unit")
stats["confirm"] = {"n_units": int(len(nec)), "flies_per_unit": [int(nec.n_flies.min()), int(nec.n_flies.max())],
                    "necessity_max": {r: [nec[r].idxmax(), round(float(nec[r].max()), 3)] for r in SEL + ["LI"]},
                    "necessity_n_above_0.05": {r: int((nec[r] > 0.05).sum()) for r in SEL + ["LI"]},
                    "sufficiency_max": {r: [suf[r].idxmax(), round(float(suf[r].max()), 3)] for r in SEL + ["LI"]},
                    "sufficiency_n_above_0.05": {r: int((suf[r] > 0.05).sum()) for r in SEL + ["LI"]}}
rb = pd.read_csv(E1 / "summary/robust_summary.csv")
rb["pass"] = (rb.spearman_vs_base >= 0.7) & (rb.top10_overlap >= 0.6)
stats["robust"] = {"cells": int(len(rb)), "pass_both": int(rb["pass"].sum()),
                   "overlap_ge_0.6": int((rb.top10_overlap >= 0.6).sum()),
                   "rho_ge_0.7": int((rb.spearman_vs_base >= 0.7).sum()),
                   "pass_by_readout": {r: int(rb[rb.readout == r]["pass"].sum()) for r in SEL},
                   "pass_by_model": {m: int(rb[rb.model == m]["pass"].sum()) for m, _ in MODELS},
                   "rho_range": [round(float(rb.spearman_vs_base.min()), 2), round(float(rb.spearman_vs_base.max()), 2)]}
(OUT_STATS / "exp1_stats.json").write_text(json.dumps(stats, indent=1))
print("wrote results/stats/exp1_stats.json")


# ------------------------------------------------------------------ summary figure (not a figure of the manuscript)
def summary_figure():
    fig = plt.figure(figsize=(7.5, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.5, wspace=0.34, left=0.08, right=0.98,
                          top=0.95, bottom=0.1)
    # A: what kind of damage carries the deficit
    ax = fig.add_subplot(gs[0, 0])
    piv = B.pivot_table(index="seed", columns="unit", values=SEL)
    kinds = [("restore_dead_all", "Removed neurons restored", mf.RULEC["death_only"]),
             ("restore_syn_all", "Thinned synapses restored", mf.RULEC["synloss_only"]),
             ("no_shift", "Threshold shift removed", mf.RULEC["hyper_only"])]
    for i, r in enumerate(SEL):
        for j, (u, lab, col) in enumerate(kinds):
            y = piv[(r, u)].dropna().clip(-0.6, 1.6)
            x = i + (j - 1) * 0.26 + mf.RNG.uniform(-0.06, 0.06, len(y))
            ax.scatter(x, y, s=3, color=col, alpha=0.35, lw=0)
            ax.plot([i + (j - 1) * 0.26 - 0.1, i + (j - 1) * 0.26 + 0.1], [y.median()] * 2, color=col, lw=1.6,
                    label=lab if i == 0 else None)
    ax.axhline(0, color=mf.GREY, lw=0.5)
    ax.axhline(1, color=mf.GREY, lw=0.5, ls=":")
    ax.set_xticks(range(len(SEL)), [SHORT[r] for r in SEL])
    ax.set_ylim(-0.65, 1.65)
    ax.set_ylabel("Fraction of the fly's deficit restored")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.13), ncol=3, fontsize=5.6, columnspacing=0.6, handlelength=0.9,
              handletextpad=0.3)
    mf.panel(ax, "A", dx=-0.13, dy=1.08)
    # B: restoring the top cell types together, against all removed neurons
    ax = fig.add_subplot(gs[0, 1])
    ks = [1, 3, 10, 30]
    for r in SEL:
        m = [C[(C.r == r) & (C.k == k)].frac.median() for k in ks]
        ax.plot(range(4), m, "-o", ms=2.5, color=mf.COL[r], label=SHORT[r].replace("\n", " "))
        ax.plot([4.6], [dead[r].median()], "s", ms=3.5, color=mf.COL[r])
    ax.set_xticks([0, 1, 2, 3, 4.6], ["1", "3", "10", "30", "All\nremoved"])
    ax.set_xlabel("Cell types restored (most effective first)")
    ax.set_ylabel("Fraction restored (median)")
    ax.set_ylim(0, 0.92)
    ax.axvline(3.8, color=mf.LIGHT, lw=0.6)
    ax.legend(loc="upper left", ncol=2, fontsize=6)
    mf.panel(ax, "B", dx=-0.2, dy=1.08)
    # D: necessity and sufficiency of single types with the full protocol
    ax = fig.add_subplot(gs[1, 1])
    reads = ["brain_spikes_A", "KC_hz_A", "MBON_hz_A", "LI", "MN9_sugar_hz", "GF_loom_hz"]
    for i, r in enumerate(reads):
        col = mf.COL[r]
        for j, (tab, mk) in enumerate([(nec, "o"), (suf, "^")]):
            y = tab[r].dropna().clip(-0.35, 0.5)
            x = i + (j - 0.5) * 0.36 + mf.RNG.uniform(-0.07, 0.07, len(y))
            ax.scatter(x, y, s=6, marker=mk, facecolor=col if j == 0 else "none", edgecolor=col, lw=0.6,
                       alpha=0.8)
    ax.axhline(0, color=mf.GREY, lw=0.5)
    ax.set_xticks(range(len(reads)), [SHORT[r] for r in reads])
    ax.set_ylim(-0.38, 0.52)
    ax.set_ylabel("Fraction of the full deficit")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3, color=mf.GREY, label="Restored alone in the damaged fly"),
                       Line2D([], [], marker="^", ls="", ms=3, mfc="none", color=mf.GREY,
                              label="Removed alone from a healthy fly")],
              loc="upper left", fontsize=6, handletextpad=0.3)
    mf.panel(ax, "D", dx=-0.2, dy=1.04)
    # C: stability of the type ranking under model variants
    ax = fig.add_subplot(gs[1, 0])
    M = rb.pivot(index="model", columns="readout", values="spearman_vs_base").reindex([m for m, _ in MODELS])[SEL]
    O = rb.pivot(index="model", columns="readout", values="top10_overlap").reindex([m for m, _ in MODELS])[SEL]
    im = ax.imshow(M.values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            v, o = M.values[a, b], O.values[a, b]
            ok = v >= 0.7 and o >= 0.6
            ax.text(b, a, f"{v:.2f}\n{o:.1f}", ha="center", va="center", fontsize=5.2,
                    color="white" if v > 0.6 else mf.INK, fontweight="bold" if ok else "normal")
            if ok:
                ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec="#c0392b", lw=1.0))
    ax.set_xticks(range(len(SEL)), [SHORT[r] for r in SEL], fontsize=5.8)
    ax.set_yticks(range(len(MODELS)), [n for _, n in MODELS])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Rank correlation with base model", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    mf.panel(ax, "C", dx=-0.32, dy=1.04)
    mf.save(fig, "exp1_summary")


summary_figure()
