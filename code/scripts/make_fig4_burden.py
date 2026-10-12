"""Fig 4 (text of 8 to 12 pt): vulnerability maps at matched synaptic burden (experiment 2).

  PYTHONIOENCODING=utf-8 python code/scripts/make_fig4_burden.py

Reads the per-fly tables of results/exp2/ and, for the comparison at equal neuron loss, of
results/battery/, recomputes every plotted value, checks each one against
results/facts/facts_exp2.json and writes

  figures/Fig4.{png,pdf,tif}
  figures/Fig4_values.json   every plotted median / interval / p value and the check result

Panels
  A  contacts lost against severity for each map (calibration.csv, mean of 3 draws per s);
     filled circles = equal neurons removed (s = 0.5), open squares = matched severities.
  B  whole-brain activity per fly: equal neurons removed (s = 0.5, map-comparison flies,
     % of the 64 series healthy flies) and the three matched levels (% of the 32 exp2
     healthy flies). Bars: medians with 95% bootstrap intervals.
  C  lateral horn rate, MBON rate, learning index and giant fibre rate at the matched levels:
     medians (% of exp2 healthy) with 95% bootstrap intervals; Kruskal-Wallis p per level.
  D  neurons removed and Kenyon cells removed at the matched levels (medians of per-fly %).
  E  projection neuron rate (whole ALPN class; delivered odour input) at the matched levels.

Layout: 7.5 in wide, about 8.2 in high, axes placed in inches. All text is 8 pt (panel
titles 8.5 pt, panel letters 11 pt). The legend of A and the two reference headers of B sit above
their axes; the four C panels share one y axis, as do the two D panels; the first p value of
each row carries the "p" prefix and is kept clear of the second. The random draws are made
in a fixed order: the resamples of every bootstrap interval, then the jitter of the points in B.

Conventions (as in code/scripts/facts_exp2.py): medians as % of the healthy median of
the same design; 95% percentile bootstrap intervals from 4,000 resamples of flies, group and
healthy group resampled independently (numpy default_rng(20261010)). Point estimates must
equal the facts file; interval bounds use a different random stream than the facts script
and are reported with their largest difference from the facts values.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory
from scipy import stats as st

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())  # code/scripts locally, scripts/ on the server
sys.path.insert(0, str(ROOT / "code/scripts"))
import figure_common as mf  # noqa: E402  (style, colours, save helper)

mf.OUT = ROOT / "figures"
mf.OUT.mkdir(parents=True, exist_ok=True)

# text of 8 to 12 pt at the printed size; overrides the style of the module imported above
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})
FACTS = json.loads((ROOT / "results/facts/facts_exp2.json").read_text(encoding="utf-8"))

SEED, NB = 20261010, 4000
RNG = np.random.default_rng(SEED)
MAPS = ["hub", "shuffled", "antihub"]
MAPN = {"hub": "Hub", "shuffled": "Shuffled", "antihub": "Antihub"}
# readout key in the facts file -> column
RCOL = {"brain": "brain_spikes_A", "LH": "LH_hz_A", "MBON": "MBON_hz_A", "LI": "LI", "GF": "GF_loom_hz",
        "PN": "PN_hz_A"}
RTITLE = {"LH": "Lateral horn rate", "MBON": "MBON rate", "LI": "Learning index", "GF": "Giant fibre rate",
          "PN": "Projection neuron rate\n(delivered odour input)"}

# ------------------------------------------------------------------ data
E2 = pd.read_csv(ROOT / "results/exp2/battery.csv")
E2["ref"] = E2["reference"].map(lambda x: "" if pd.isna(x) else f"{float(x):g}")
H2 = E2[E2["map"] == "healthy"]                                   # 32 exp2 healthy flies
SC = pd.read_csv(ROOT / "results/battery/stage_curve_x4.csv")
HS = SC[(SC.scope == "pan") & (SC.stage == 0)]                   # 64 series healthy flies
VU = pd.read_csv(ROOT / "results/battery/vulnerability_x4.csv")  # equal-neuron map comparison
CAL = pd.read_csv(ROOT / "results/exp2/calibration.csv").groupby(["map", "s"]).mean(numeric_only=True).reset_index()
MATCH = json.loads((ROOT / "results/exp2/matched.json").read_text())
REFS = list(MATCH["maps"].keys())                                 # "0.1", "0.25", "0.5"
TGT = {r: 100 * MATCH["maps"][r]["target_syn_lost_frac"] for r in REFS}
LV = {r: f"syn{TGT[r]:.1f}" for r in REFS}                        # facts level names
assert len(HS) == FACTS["series_n_healthy"] and len(H2) == FACTS["exp2_n_healthy"]


def grp(ref, m):
    return E2[(E2.ref == ref) & (E2["map"] == m)]


def present(ref):
    return [m for m in MAPS if len(grp(ref, m))]


def pct_ci(x, h):
    """[median % of healthy, 95% lo, hi]: percentile bootstrap, group and healthy resampled independently."""
    x, h = np.asarray(pd.Series(x).dropna(), float), np.asarray(pd.Series(h).dropna(), float)
    b = 100 * np.median(RNG.choice(x, (NB, len(x))), 1) / np.median(RNG.choice(h, (NB, len(h))), 1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return [round(float(100 * np.median(x) / np.median(h)), 1), round(float(lo), 1), round(float(hi), 1)]


def burden_at(m, s):
    return 100 * float(CAL[(CAL["map"] == m) & np.isclose(CAL["s"], s)].syn_lost_frac.iloc[0])


def ptxt(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


# ------------------------------------------------------------------ values
V = {"settings": {"bootstrap_resamples": NB, "seed": SEED, "interval": "95% percentile bootstrap of the ratio of "
                  "group median to healthy median, flies resampled within the group and the healthy group"},
     "n_series_healthy": int(len(HS)), "n_exp2_healthy": int(len(H2)),
     "A": {"contacts_lost_pct_s0.5_equal_neurons": {m: round(burden_at(m, 0.5), 1) for m in MAPS},
           "max_contacts_lost_pct_s1": {m: round(burden_at(m, 1.0), 1) for m in MAPS},
           "targets_pct": {LV[r]: round(TGT[r], 2) for r in REFS},
           "matched_s": {LV[r]: {m: MATCH["maps"][r][m]["s"] for m in MAPS} for r in REFS}},
     "B_equal_neurons": {}, "levels": {}}
for m in MAPS:
    x = VU[VU.condition == f"{m}_s0.5"]
    V["B_equal_neurons"][m] = {"n": int(len(x)), "neurons_removed_pct": round(100 * float(x.dead_frac.median()), 2),
                               "brain": pct_ci(x.brain_spikes_A, HS.brain_spikes_A)}
for r in REFS:
    lv = {"target_pct": round(TGT[r], 2), "maps": {}, "kruskal_p": {}}
    for m in present(r):
        x = grp(r, m)
        lv["maps"][m] = {"s": MATCH["maps"][r][m]["s"], "n": int(len(x)),
                         "contacts_lost_pct": round(100 * float(x.syn_lost_frac.median()), 1),
                         "neurons_removed_pct": round(100 * float(x.dead_frac.median()), 2),
                         "KC_removed_pct": round(100 * float(x.kc_dead_frac.median()), 2),
                         "neurons_removed_iqr_pp": round(100 * float(x.dead_frac.quantile(.75) - x.dead_frac.quantile(.25)), 2),
                         "KC_removed_iqr_pp": round(100 * float(x.kc_dead_frac.quantile(.75) - x.kc_dead_frac.quantile(.25)), 2),
                         **{k: pct_ci(x[c], H2[c]) for k, c in RCOL.items()}}
    for k, c in RCOL.items():
        lv["kruskal_p"][k] = float(st.kruskal(*[grp(r, m)[c].dropna() for m in present(r)]).pvalue)
    V["levels"][LV[r]] = lv

# ------------------------------------------------------------------ check against facts_exp2.json
point_bad, ci_diff = [], 0.0
for m in MAPS:
    f = FACTS[f"neu_s0.5_{m}_brain_pct"]
    v = V["B_equal_neurons"][m]["brain"]
    if v[0] != f[0]:
        point_bad.append((f"neu_s0.5_{m}_brain_pct", v[0], f[0]))
    ci_diff = max(ci_diff, abs(v[1] - f[1]), abs(v[2] - f[2]))
    if round(V["A"]["contacts_lost_pct_s0.5_equal_neurons"][m], 1) != FACTS[f"neu_s0.5_{m}_contacts_lost_pct_calibration"]:
        point_bad.append((f"neu_s0.5_{m}_contacts_lost_pct_calibration",))
    if V["B_equal_neurons"][m]["neurons_removed_pct"] != FACTS[f"neu_s0.5_{m}_neurons_removed_pct"]:
        point_bad.append((f"neu_s0.5_{m}_neurons_removed_pct",))
for m in MAPS:
    if round(V["A"]["max_contacts_lost_pct_s1"][m], 1) != FACTS[f"max_reachable_contacts_lost_pct_{m}"]:
        point_bad.append((f"max_reachable_contacts_lost_pct_{m}",))
for r in REFS:
    L = LV[r]
    for m, v in V["levels"][L]["maps"].items():
        if v["s"] != FACTS[f"{L}_{m}_severity_s"]:
            point_bad.append((f"{L}_{m}_severity_s",))
        for k in ("contacts_lost_pct", "neurons_removed_pct", "KC_removed_pct"):
            if v[k] != FACTS[f"{L}_{m}_{k}"]:
                point_bad.append((f"{L}_{m}_{k}", v[k], FACTS[f"{L}_{m}_{k}"]))
        for k in RCOL:
            f = FACTS[f"{L}_{m}_{k}_pct"]
            if v[k][0] != f[0]:
                point_bad.append((f"{L}_{m}_{k}_pct", v[k][0], f[0]))
            ci_diff = max(ci_diff, abs(v[k][1] - f[1]), abs(v[k][2] - f[2]))
    for k in RCOL:
        f = FACTS[f"{L}_{k}_kruskal_p"]
        if float(f"{V['levels'][L]['kruskal_p'][k]:.3g}") != f:      # facts keep 3 significant figures
            point_bad.append((f"{L}_{k}_kruskal_p", V["levels"][L]["kruskal_p"][k], f))
assert not point_bad, f"values differ from facts_exp2.json: {point_bad}"
V["check_vs_facts_exp2"] = {"point_estimates_identical": True,
                            "max_abs_ci_bound_diff_pct_points": round(ci_diff, 1),
                            "note": "Interval bounds are recomputed with seed 20261010 but a different resampling "
                                    "order than facts_exp2.py, so they differ from the facts by Monte Carlo "
                                    "error only; quote the facts file in the text."}
(mf.OUT / "Fig4_values.json").write_text(json.dumps(V, indent=1))
print(f"checked against facts_exp2.json: points identical, largest CI-bound difference {ci_diff:.1f} points")

# ------------------------------------------------------------------ figure
# layout: every length below is in inches at the printed size (7.5 in wide); all ordinary text is 8 pt,
# panel titles 8.5 pt, panel letters 11 pt.
FS, SUB = 8.0, "#4a5057"
W_IN, H_IN = 7.5, 8.2
X0, X1 = 0.54, 7.44                                # left edge of the first column, right edge of the last
GAP = 0.18                                         # between the four columns of rows 2 and 3
WC = (X1 - X0 - 3 * GAP) / 4                       # column width
COLX = [X0 + i * (WC + GAP) for i in range(4)]
YLAB = 0.36                                        # right edge of a y-axis label, inches left of its axes
LEVX = [round(TGT[r], 1) for r in REFS]            # 11.5, 27.1, 48.3
OFF = {"hub": -1.6, "shuffled": 0.0, "antihub": 1.6}


def axes_in(fig, left, top, width, height, **kw):
    """Axes placed in inches; left and top are measured from the left and top edges of the figure."""
    return fig.add_axes([left / W_IN, 1 - (top + height) / H_IN, width / W_IN, height / H_IN], **kw)


def letter(fig, s, x, ref):
    """Panel letter at x inches from the left, on the baseline of the first line of the text artist ref."""
    fig.draw_without_rendering()
    r = fig.canvas.get_renderer()
    _, h, d = r.get_text_width_height_descent("lp", ref.get_fontproperties(), ismath=False)
    y = (ref.get_window_extent(r).y1 - (h - d)) / fig.dpi / H_IN
    return fig.text(x / W_IN, y, s, fontsize=11, fontweight="bold", va="baseline", ha="left", color=mf.INK)


def ylabel_at(ax, text):
    ax.set_ylabel(text)
    ax.yaxis.set_label_coords(-YLAB / (ax.get_position().width * W_IN), 0.5)


def wrap_in(fig, text, width, **kw):
    """Greedy word wrap of one paragraph to a width in inches, measured with the figure's own renderer."""
    r = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "", **kw)
    lines, cur = [], ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        probe.set_text(trial)
        if cur and probe.get_window_extent(r).width / fig.dpi > width:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    probe.remove()
    return "\n".join(lines + [cur])


def level_axis(ax):
    ax.set_xticks(LEVX, [f"{v:.1f}" for v in LEVX])
    ax.set_xlim(-2, 55.5)
    ax.set_xlabel("Contacts lost (%)")


def ci_series(fig, ax, key, ylim, pvals=True, title=None):
    """Medians with 95% intervals at the matched levels; returns the title artist (None without a title)."""
    for m in MAPS:
        xs, ys, lo, hi = [], [], [], []
        for r, lx in zip(REFS, LEVX):
            v = V["levels"][LV[r]]["maps"].get(m)
            if v is None:
                continue
            xs.append(lx + OFF[m]); ys.append(v[key][0]); lo.append(v[key][1]); hi.append(v[key][2])
        ys, lo, hi = np.array(ys), np.array(lo), np.array(hi)
        ax.plot(xs, ys, color=mf.MAPC[m], lw=1.0, zorder=2)
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt="o", color=mf.MAPC[m], ms=3.8, mec="white", mew=0.5,
                    elinewidth=1.0, capsize=0, zorder=3)
    ax.axhline(100, color=mf.GREY, lw=0.6, ls=":", zorder=0)
    ax.set_ylim(*ylim)
    level_axis(ax)
    ttl = ax.set_title(title, loc="left", linespacing=1.25) if title else None
    if pvals:
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        pt = []
        for r, lx in zip(REFS, LEVX):
            p = V["levels"][LV[r]]["kruskal_p"][key]
            pt.append(ax.text(lx, 0.03, ("p " if p < 0.001 else "p = ") + ptxt(p) if lx == LEVX[0] else ptxt(p),
                              transform=tr, ha="center", va="bottom", fontsize=FS, color=SUB))
        # the first label carries the "p" prefix: keep it at least 0.11 in clear of the second
        rd = fig.canvas.get_renderer()
        b0, b1 = pt[0].get_window_extent(rd), pt[1].get_window_extent(rd)
        short = 0.11 * fig.dpi - (b1.x0 - b0.x1)
        if short > 0:
            xd = ax.transData.transform((LEVX[0], 0))[0] - short
            pt[0].set_x(ax.transData.inverted().transform((xd, 0))[0])
    return ttl


def fig4():
    fig = plt.figure(figsize=(W_IN, H_IN))
    HAB, HC, HDE = 2.2, 1.65, 1.7                   # axes heights of the three rows
    TAB = 0.57                                      # top of the A and B axes (legend and headers sit above)
    TC = TAB + HAB + 0.83                           # top of the C axes
    TDE = TC + HC + 0.91                            # top of the D and E axes
    WA = 2.28
    XB = COLX[2]                                    # B and E start at the third column

    # A: contacts lost against severity
    ax = axes_in(fig, X0, TAB, WA, HAB)
    for m in MAPS:
        c = CAL[CAL["map"] == m].sort_values("s")
        ax.plot(c.s, 100 * c.syn_lost_frac, color=mf.MAPC[m], lw=1.4, label=MAPN[m], zorder=2)
    ax.axvline(0.5, color=mf.GREY, lw=0.7, ls="--", zorder=0)
    for m in MAPS:
        y = burden_at(m, 0.5)
        ax.plot([0.5], [y], "o", ms=4.6, color=mf.MAPC[m], zorder=4)
        up = m == "hub"                             # hub label above left, the others below right (clear of the curves)
        ax.annotate(f"{y:.0f}%", (0.5, y), xytext=(-6, 4) if up else (6, -3.5), textcoords="offset points",
                    ha="right" if up else "left", va="bottom" if up else "top", fontsize=FS,
                    color=mf.MAPC[m], fontweight="bold")
    trA = blended_transform_factory(ax.transAxes, ax.transData)
    for r in REFS:
        t = TGT[r]
        ax.axhline(t, color=mf.LIGHT, lw=0.8, ls=":", zorder=0)
        ax.text(1.03, t, f"{t:.1f}%", transform=trA, ha="left", va="center", fontsize=FS, color=SUB,
                clip_on=False)
        for m in MAPS:
            s = MATCH["maps"][r][m]["s"]
            if s is not None:
                ax.plot([s], [t], "s", ms=4.2, mfc="white", mec=mf.MAPC[m], mew=1.1, zorder=5)
    amax = burden_at("antihub", 1.0)
    ax.text(1.03, amax, f"max {amax:.1f}%", transform=trA, ha="left", va="center", fontsize=FS,
            color=mf.MAPC["antihub"], clip_on=False)
    ax.annotate("equal neurons\nremoved (s = 0.5)", (0.5, 81), xytext=(-5, 0), textcoords="offset points",
                ha="right", va="top", fontsize=FS, color=SUB, linespacing=1.15)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 82)
    ax.set_xlabel("Severity s")
    ylabel_at(ax, "Synaptic contacts lost (%)")
    h, lab = ax.get_legend_handles_labels()
    h += [plt.Line2D([], [], ls="", marker="o", ms=4.6, color=mf.GREY),
          plt.Line2D([], [], ls="", marker="s", ms=4.2, mfc="white", mec=mf.GREY, mew=1.1)]
    lab += ["equal neurons (s = 0.5)", "matched contacts lost"]
    leg = ax.legend(h, lab, loc="lower left", bbox_to_anchor=(0, 1 + 0.07 / HAB), ncol=2, fontsize=FS,
                    handlelength=1.5, borderaxespad=0, borderpad=0, labelspacing=0.3, columnspacing=2.2)
    la = letter(fig, "A", 0.04, leg.get_texts()[0])

    # B: whole-brain activity per fly
    ax = axes_in(fig, XB, TAB, X1 - XB, HAB)
    groups = [("Equal neurons\n(s = 0.5)", None)] + [(f"{TGT[r]:.1f}%", r) for r in REFS]
    xpos = [0, 1.35, 2.35, 3.35]
    for i, (lab, ref) in enumerate(groups):
        for j, m in enumerate(MAPS):
            xc = xpos[i] + (j - 1) * 0.27
            if ref is None:
                y, hh = VU[VU.condition == f"{m}_s0.5"].brain_spikes_A, HS.brain_spikes_A
                med = V["B_equal_neurons"][m]["brain"]
            else:
                g = grp(ref, m)
                if g.empty:
                    ax.text(xc + 0.02, 3, "not\nreachable", ha="center", va="bottom", fontsize=FS,
                            color=mf.MAPC[m], linespacing=1.1)
                    continue
                y, hh = g.brain_spikes_A, H2.brain_spikes_A
                med = V["levels"][LV[ref]]["maps"][m]["brain"]
            y = 100 * np.asarray(y, float) / np.median(hh)
            ax.scatter(xc + RNG.uniform(-0.07, 0.07, len(y)), y, s=4.5, color=mf.MAPC[m], alpha=0.4, lw=0,
                       zorder=1)
            ax.plot([xc, xc], [med[1], med[2]], color=mf.MAPC[m], lw=1.1, zorder=3)
            ax.plot([xc - 0.1, xc + 0.1], [med[0]] * 2, color=mf.MAPC[m], lw=2.1, zorder=4,
                    solid_capstyle="butt")
    ax.axhline(100, color=mf.GREY, lw=0.6, ls=":", zorder=0)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot([0.675, 0.675], [0, 1 + 0.48 / HAB], transform=tr, color=mf.LIGHT, lw=0.8, clip_on=False)
    ax.set_xticks(xpos, [g[0] for g in groups])
    ax.set_xlim(-0.48, 4.0)
    ax.set_ylim(0, 123)
    ax.set_yticks(range(0, 121, 20))
    ylabel_at(ax, "Whole-brain activity (% of healthy)")
    ax.plot([1.05, 3.65], [-0.25 / HAB] * 2, transform=tr, color=mf.GREY, lw=0.7, clip_on=False)
    ax.text(2.35, -0.30 / HAB, "Matched contacts lost", transform=tr, ha="center", va="top", fontsize=FS)
    ax.text(0, 1 + 0.07 / HAB, f"% of series\nhealthy\n(n = {len(HS)})", transform=tr, ha="center", va="bottom",
            fontsize=FS, color=SUB, linespacing=1.15)
    ax.text(2.35, 1 + 0.07 / HAB, f"% of burden-matched design healthy\n(n = {len(H2)})", transform=tr,
            ha="center", va="bottom", fontsize=FS, color=SUB, linespacing=1.15)
    fig.text((XB - 0.52) / W_IN, la.get_position()[1], "B", fontsize=11, fontweight="bold", va="baseline",
             ha="left", color=mf.INK)

    # C: four readouts at matched levels (shared y axis)
    for k, key in enumerate(["LH", "MBON", "LI", "GF"]):
        ax = axes_in(fig, COLX[k], TC, WC, HC)
        ttl = ci_series(fig, ax, key, (0, 118), title=RTITLE[key])
        ax.set_yticks(range(0, 101, 20))
        if k == 0:
            ylabel_at(ax, "% of healthy")
            letter(fig, "C", 0.04, ttl)
        else:
            ax.tick_params(labelleft=False)

    # D: neurons and Kenyon cells removed at matched levels (shared y axis)
    WD = (COLX[2] - 0.62 - X0 - GAP) / 2
    for k, (key, lab) in enumerate([("neurons_removed_pct", "All neurons"), ("KC_removed_pct", "Kenyon cells")]):
        ax = axes_in(fig, X0 + k * (WD + GAP), TDE, WD, HDE)
        for m in MAPS:
            xs, ys = [], []
            for r, lx in zip(REFS, LEVX):
                v = V["levels"][LV[r]]["maps"].get(m)
                if v is not None:
                    xs.append(lx); ys.append(v[key])
            ax.plot(xs, ys, "-o", ms=3.8, color=mf.MAPC[m], mec="white", mew=0.5, lw=1.0, label=MAPN[m])
        ax.set_ylim(0, 21)
        ax.set_yticks(range(0, 21, 5))
        level_axis(ax)
        ax.set_title(lab + " removed", loc="left")
        if k == 0:
            ylabel_at(ax, "Removed (%)")
        else:
            ax.tick_params(labelleft=False)

    # E: projection neuron rate (delivered odour input)
    ax = axes_in(fig, COLX[2], TDE, WC, HDE)
    ttl = ci_series(fig, ax, "PN", (55, 108), title=RTITLE["PN"])
    ax.set_yticks(range(60, 101, 10))
    ylabel_at(ax, "% of healthy")
    le = letter(fig, "E", COLX[2] - 0.52, ttl)
    fig.text(0.04 / W_IN, le.get_position()[1], "D", fontsize=11, fontweight="bold", va="baseline", ha="left",
             color=mf.INK)                         # D and E letters on one line (first line of the E title)

    # map key and notes in the last slot
    ax = axes_in(fig, COLX[3], TDE - 0.25, WC, HDE + 0.55)
    ax.axis("off")
    hs = [plt.Line2D([], [], color=mf.MAPC[m], marker="o", ms=3.8, mec="white", mew=0.5, lw=1.0) for m in MAPS]
    ax.legend(hs, [f"{MAPN[m]} map" for m in MAPS], loc="upper left", bbox_to_anchor=(0, 1.0), fontsize=FS,
              title="Vulnerability map", title_fontsize=FS, alignment="left", borderaxespad=0, borderpad=0,
              labelspacing=0.3, handlelength=1.6)
    notes = [wrap_in(fig, s, WC, fontsize=FS) for s in (
        "C, E: medians with 95% bootstrap intervals. Numbers along the bottom: Kruskal-Wallis p across maps "
        "at each level (hub vs shuffled only at 48.3%).",
        f"The antihub map cannot reach 48.3% (at most {burden_at('antihub', 1.0):.1f}% at s = 1).")]
    ax.text(0.0, 1 - 0.80 / (HDE + 0.55), "\n\n".join(notes), transform=ax.transAxes, ha="left", va="top",
            fontsize=FS, color=SUB, linespacing=1.25)
    mf.save(fig, "Fig4")


fig4()
