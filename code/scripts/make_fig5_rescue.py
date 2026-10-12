"""Fig 5 (text of 8 to 12 pt): within-fly rescue (experiment 1).

  python code/scripts/make_fig5_rescue.py

Reads only result files in results/exp1/ (screen, robust, confirm), recomputes every
plotted value, checks it against results/facts/facts_exp1.json (the script stops on a
mismatch) and writes figures/Fig5.{png,pdf,tif} (300 dpi, 7.44 in wide) plus
figures/Fig5_values.json with the plotted numbers and their n.

Restored fraction f = (y_variant - y_damaged) / (y_healthy - y_damaged), within one fly under
identical input (analyze_exp1.frac_table). A readout is left out for a fly when
|y_healthy - y_damaged| <= 5% of |y_healthy|. Fly-readout pairs whose damaged value is ABOVE
healthy (screen: KC 2 flies, MBON 4, MN9 4) are kept with a negative denominator, exactly as
in facts_exp1.json (definitions.deficit): they enter every median, and their per-fly points
are drawn with a separate marker; per-fly values outside the plotted range are drawn at the
axis edge as triangles (they are overshoots or small-denominator artefacts, not rescues).

Panels
  A  decomposition (removed neurons restored, thinned synapses restored, threshold shift
     removed), per readout, s = 0.3 (open) and s = 0.5 (filled), median per severity
  B  single-class rescues of five classes next to restoring all removed neurons (median, 95% CI)
  C  flies with giant fibre output below 100 Hz: all removed, descending-neuron class,
     best tested type, giant fibre (DNp01) type where tested
  D  top-k tested types restored together vs all removed neurons (medians)
  E  robustness of the type ranking (pre-specified criteria) for parameter variants and
     input-noise realisations, shown as separate blocks
  F  full-protocol necessity (restored alone) and sufficiency (removed alone from the healthy copy)

Layout: all text is 8 pt, panel
letters 11 pt; the figure is 7.44 x 8.54 in with the axes placed in inches (fig5). The legend of A
is two rows above the axes, the legend of B four rows above y = 1, the legend of D one column, the legend of F
above the axes under its title, and the colour bar of E is horizontal below the grid.
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())  # code/scripts locally, scripts/ on the server
sys.path.insert(0, str(ROOT / "code/scripts"))
import analyze_exp1 as ax_  # noqa: E402  (frac_table, rho: the definitions used by facts_exp1.json)
import figure_common as mf  # noqa: E402  (style, colours, panel and save helpers)

E1 = ROOT / "results/exp1"
mf.OUT = ROOT / "figures"
mf.OUT.mkdir(parents=True, exist_ok=True)

# text of 8 to 12 pt at the printed size; overrides the style of the module imported above
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})
FACTS = json.loads((ROOT / "results/facts/facts_exp1.json").read_text(encoding="utf-8"))
RNG = np.random.default_rng(5)
SEED, NBOOT, MIN_DEF = 20261009, 4000, 0.05          # bootstrap as in facts_exp1.py

SEL = ["brain_spikes_A", "LH_hz_A", "KC_hz_A", "MBON_hz_A", "MN9_sugar_hz", "GF_loom_hz"]
R = {"brain_spikes_A": "WB", "LH_hz_A": "LH", "KC_hz_A": "KC", "MBON_hz_A": "MBON",
     "MN9_sugar_hz": "MN9", "GF_loom_hz": "GF", "LI": "LI"}
SHORT = {"brain_spikes_A": "Whole\nbrain", "LH_hz_A": "Lateral\nhorn", "KC_hz_A": "Kenyon\ncells",
         "MBON_hz_A": "MBONs", "MN9_sugar_hz": "MN9", "GF_loom_hz": "Giant\nfibre", "LI": "Learning\nindex"}
CLASSES = [("ALPN", "Antennal lobe PNs", "#e08a2e"),
           ("sc:descending_neuron", "Descending neurons", "#5e3c99"),
           ("MBON", "MBONs", "#2a8c8c"),
           ("Kenyon_Cell", "Kenyon cells", "#4f8f3a"),
           ("sc:cb_intrinsic", "CB intrinsic (incl. APL)", "#8a6d3b")]
PARAM = [("w0.8", "Weights ×0.8"), ("w1.2", "Weights ×1.2"), ("glu0.5", "Glutamate ×0.5"),
         ("da0.5", "Adaptation ×0.5")]
NOISE = [("noise1", "Input noise 1"), ("noise2", "Input noise 2")]
KINDS = [("restore_dead_all", "Removed neurons restored", mf.RULEC["death_only"]),
         ("restore_syn_all", "Thinned synapses restored", mf.RULEC["synloss_only"]),
         ("no_shift", "Threshold shift removed", mf.RULEC["hyper_only"])]
YLO, YHI = -0.6, 1.6                                    # panel A plotted range


# ------------------------------------------------------------------ helpers
def mci(v):
    """[median, 95% bootstrap CI low, high, n] exactly as facts_exp1.mci (same seed and resamples)."""
    v = np.asarray(pd.Series(v, dtype=float).dropna())
    if len(v) == 0:
        return [np.nan, np.nan, np.nan, 0]
    if len(v) < 2:
        return [float(np.median(v)), np.nan, np.nan, 1]
    rng = np.random.default_rng(SEED)
    b = np.median(v[rng.integers(0, len(v), (NBOOT, len(v)))], axis=1)
    return [float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(len(v))]


N_CHECKS = [0]


def check(key, got, idx=None, tol=6e-4):
    """Stop if a recomputed value differs from facts_exp1.json beyond rounding."""
    want = FACTS[key] if idx is None else FACTS[key][idx]
    g, w = np.atleast_1d(np.asarray(got, dtype=float)), np.atleast_1d(np.asarray(want, dtype=float))
    if g.shape != w.shape or not np.allclose(g, w, atol=tol, equal_nan=True):
        raise SystemExit(f"MISMATCH {key}{'' if idx is None else f'[{idx}]'}: computed {got}, facts {want}")
    N_CHECKS[0] += 1


def draw_strip(ax, x, y, excess, col, filled, w=0.035, s=5.0, lo=YLO, hi=YHI):
    """Per-fly points with jitter; values outside [lo, hi] as edge triangles; excess flies as crosses."""
    y = pd.Series(y).dropna()
    if y.empty:
        return
    jx = x + RNG.uniform(-w, w, len(y))
    ex = y.index.isin(excess)
    inr = (y >= lo) & (y <= hi)
    norm = inr & ~ex
    kw = dict(facecolor=col if filled else "white", edgecolor=col, lw=0.45, alpha=0.75 if filled else 0.9, zorder=2)
    ax.scatter(jx[norm], y[norm], s=s, marker="o", **kw)
    ab, be = (y > hi) & ~ex, (y < lo) & ~ex
    ax.scatter(jx[ab], np.full(ab.sum(), hi), s=s + 3, marker="^", **kw)
    ax.scatter(jx[be], np.full(be.sum(), lo), s=s + 3, marker="v", **kw)
    if ex.any():
        ax.scatter(jx[ex], y[ex].clip(lo, hi), s=s + 5, marker="x", color=mf.INK, lw=0.7, zorder=3)


# ------------------------------------------------------------------ data
runs = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(E1 / "screen/*_runs.csv")))], ignore_index=True)
F = ax_.frac_table(runs, SEL, MIN_DEF)
stage_of = runs.groupby("seed").stage.first().to_dict()
seeds = sorted(stage_of)
raw = runs[runs.level == "baseline"].pivot_table(index="seed", columns="unit", values=SEL)
EXCESS = {}
for r in SEL:
    d = raw[(r, "healthy")] - raw[(r, "damaged")]
    kept = d.abs() > MIN_DEF * raw[(r, "healthy")].abs()
    EXCESS[r] = sorted(d[kept & (d < 0)].index)
    check(f"k.n_flies_with_deficit.{R[r]}", [int(kept.sum()), int(kept[[stage_of[s] == 0.3 for s in kept.index]].sum()),
                                            int(kept[[stage_of[s] == 0.5 for s in kept.index]].sum())])
    check(f"k.n_flies_damaged_above_healthy.{R[r]}", len(EXCESS[r]), idx=0)
B = F[F.level == "baseline"].pivot_table(index="seed", columns="unit", values=[f"f_{r}" for r in SEL])
CL = F[F.level == "class"]
T = F[F.level == "type"]
CU = F[F.level == "cumulative"].copy()
CU["r"] = CU.unit.str.split(":").str[0]
CU["k"] = CU.unit.str.split("top").str[-1].astype(int)
CU["frac"] = [row[f"f_{row.r}"] for _, row in CU.iterrows()]
VAL = {"_meaning": "Values plotted in figures/Fig5 (make_fig5_rescue.py); restored fractions of each "
                   "fly's deficit, medians over flies; [median, 95% bootstrap CI low, high, n] where 4 numbers."}

# A: decomposition, per stage
VAL["A"] = {}
for r in SEL:
    for u, _, _ in KINDS:
        v = B[(f"f_{r}", u)]
        check(f"f.{u}.{R[r]}.all", mci(v)[0], idx=0)
        for s_ in (0.3, 0.5):
            vs = v[[stage_of[s] == s_ for s in v.index]]
            m = mci(vs)
            check(f"f.{u}.{R[r]}.s{s_}", m)
            VAL["A"][f"{u}|{R[r]}|s{s_}"] = [round(x, 4) for x in m[:3]] + [m[3]]

# B: single-class rescues
VAL["B"] = {}
for r in SEL:
    for cls, _, _ in CLASSES:
        m = mci(CL[CL.unit == cls][f"f_{r}"])
        check(f"a.class_single.{cls}.{R[r]}.all", m, tol=6e-4)
        VAL["B"][f"{cls}|{R[r]}"] = [round(x, 4) for x in m[:3]] + [m[3]]
    m = mci(B[(f"f_{r}", "restore_dead_all")])
    check(f"f.restore_dead_all.{R[r]}.all", m)
    VAL["B"][f"restore_dead_all|{R[r]}"] = [round(x, 4) for x in m[:3]] + [m[3]]
alpn = CL[CL.unit == "ALPN"].set_index("seed")
nz_seeds = alpn.index[alpn.f_brain_spikes_A.fillna(0) != 0]
check("a.ALPN.n_flies_nonzero_WB", len(nz_seeds), idx=0)
for r in SEL[:4]:
    m = mci(alpn.loc[nz_seeds, f"f_{r}"])
    check(f"a.ALPN.nonzero_flies.{R[r]}", m)
    VAL["B"][f"ALPN_nonzero_WB_flies|{R[r]}"] = [round(x, 4) for x in m[:3]] + [m[3]]

# C: flies with giant fibre output below 100 Hz
gfd = raw[("GF_loom_hz", "damaged")]
fail = sorted(gfd[gfd < 100].index)
check("c.escape_fail_seeds", fail)
dn = CL[CL.unit == "sc:descending_neuron"].set_index("seed").f_GF_loom_hz
dnp = T[T.unit == "DNp01"].set_index("seed").f_GF_loom_hz
best_gf = T.dropna(subset=["f_GF_loom_hz"]).groupby("seed").f_GF_loom_hz.max()
best_gf_unit = T.dropna(subset=["f_GF_loom_hz"]).loc[T.dropna(subset=["f_GF_loom_hz"]).groupby("seed")
                                                     .f_GF_loom_hz.idxmax()].set_index("seed").unit
dead_gf = B[("f_GF_loom_hz", "restore_dead_all")]
CTAB = pd.DataFrame({"all_removed": dead_gf.reindex(fail), "DN_class": dn.reindex(fail),
                     "best_type": best_gf.reindex(fail), "DNp01": dnp.reindex(fail)})
tab = {row[0]: row for row in FACTS["c.escape_fail_table"]}
for s in fail:
    got = [CTAB.DN_class[s], CTAB.all_removed[s], CTAB.best_type[s], CTAB.DNp01[s]]
    want = [tab[s][1], tab[s][2], tab[s][3], np.nan if tab[s][5] is None else tab[s][5]]
    if not np.allclose(got, want, atol=6e-4, equal_nan=True):
        raise SystemExit(f"MISMATCH c.escape_fail_table fly {s}: {got} vs {want}")
    N_CHECKS[0] += 1
VAL["C"] = {"seeds": fail, "damaged_GF_hz": [round(float(gfd[s]), 1) for s in fail],
            "stage": [stage_of[s] for s in fail],
            **{c: [None if pd.isna(x) else round(float(x), 3) for x in CTAB[c]] for c in CTAB},
            "best_type_unit": [best_gf_unit[s] for s in fail]}

# D: cumulative top-k
VAL["D"] = {}
for r in SEL:
    for k in (1, 3, 10, 30):
        m = mci(CU[(CU.r == r) & (CU.k == k)].frac)
        check(f"e.cumulative_top{k}.{R[r]}", m)
        VAL["D"][f"top{k}|{R[r]}"] = [round(x, 4) for x in m[:3]] + [m[3]]
    VAL["D"][f"all_removed|{R[r]}"] = VAL["B"][f"restore_dead_all|{R[r]}"]

# E: robustness (pre-specified rho >= 0.7 and top-10 overlap >= 0.6 against the base model)
Rb = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(E1 / "robust/*.csv")))], ignore_index=True)
FR = {m: ax_.frac_table(g, SEL, MIN_DEF) for m, g in Rb.groupby("model")}
MED = pd.concat([FR[m][FR[m].level == "type"].groupby("unit")[f"f_{r}"].median().rename(f"{m}|{r}")
                 for m in FR for r in SEL], axis=1)
cells = {}
for r in SEL:
    for m, _ in PARAM + NOISE:
        ok = MED[[f"{m}|{r}", f"base|{r}"]].dropna()
        x, y = ok[f"{m}|{r}"], ok[f"base|{r}"]
        rho = ax_.rho(x, y)
        ov = len(set(y.nlargest(10).index) & set(x.nlargest(10).index)) / 10
        n_nz = int(((x != 0) | (y != 0)).sum())
        cells[(m, r)] = {"rho": rho, "ov": ov, "pass": bool(rho >= 0.7 and ov >= 0.6), "n_nonzero": n_nz,
                         "degenerate": bool(rho >= 0.7 and ov >= 0.6 and n_nz <= 5)}
CE = pd.DataFrame([{"m": m, "r": r, **v} for (m, r), v in cells.items()])
rbs = pd.read_csv(E1 / "summary/robust_summary.csv")
chk = CE.merge(rbs, left_on=["m", "r"], right_on=["model", "readout"])
if not (len(chk) == 36 and np.allclose(chk.rho, chk.spearman_vs_base) and np.allclose(chk.ov, chk.top10_overlap)):
    raise SystemExit("MISMATCH robust_summary.csv")
grp = CE.m.isin([m for m, _ in NOISE])
check("g.prespecified_pass", [int(CE["pass"].sum()), len(CE), int((CE.ov >= 0.6).sum()), int((CE.rho >= 0.7).sum())])
check("g.prespecified_pass_by_group", [int(CE[grp]["pass"].sum()), int(grp.sum()), int(CE[~grp]["pass"].sum()),
                                       int((~grp).sum())])
check("g.degenerate_passes", int(CE.degenerate.sum()))
VAL["E"] = {f"{m}|{R[r]}": [round(v["rho"], 3), round(v["ov"], 2), v["pass"], v["n_nonzero"], v["degenerate"]]
            for (m, r), v in cells.items()}
VAL["E_meaning"] = "[Spearman rho of type medians vs base, top-10 overlap, passes both, n types non-zero in either, " \
                   "pass resting on <= 5 non-zero types]"

# F: full protocol (confirm phase), per-type medians over the flies that lost the type
Cf = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(E1 / "confirm/*.csv")))], ignore_index=True)
FC = ax_.frac_table(Cf, SEL + ["LI"], MIN_DEF)
READ_F = SEL + ["LI"]
NEC = FC[FC.level == "type"].groupby("unit")[[f"f_{r}" for r in READ_F]].median()
SUF = (1 - FC[FC.level == "lesion"][[f"f_{r}" for r in READ_F]]).groupby(FC[FC.level == "lesion"].unit).median()
NEC.columns = SUF.columns = READ_F
cs = pd.read_csv(E1 / "summary/confirm_summary.csv")
for tab_, key in ((NEC, "necessity"), (SUF, "sufficiency")):
    ref = cs[cs.test.str.startswith(key)].set_index("unit")[READ_F].reindex(tab_.index)
    if not np.allclose(tab_.values, ref.values, atol=1e-9, equal_nan=True):
        raise SystemExit(f"MISMATCH confirm_summary.csv ({key})")
for r in READ_F:
    if FACTS[f"i.necessity_max.{R[r]}"][0] != NEC[r].idxmax() or abs(FACTS[f"i.necessity_max.{R[r]}"][1] - NEC[r].max()) > 6e-4:
        raise SystemExit(f"MISMATCH i.necessity_max.{R[r]}")
    if FACTS[f"i.sufficiency_max.{R[r]}"][0] != SUF[r].idxmax() or abs(FACTS[f"i.sufficiency_max.{R[r]}"][1] - SUF[r].max()) > 6e-4:
        raise SystemExit(f"MISMATCH i.sufficiency_max.{R[r]}")
    N_CHECKS[0] += 2
VAL["F"] = {"n_types": int(len(NEC)), "flies_per_type": [int(x) for x in FC[FC.level == "type"].groupby("unit").seed
                                                            .nunique().agg(["min", "median", "max"])],
            **{f"necessity|{R[r]}": [round(float(NEC[r].min()), 3), round(float(NEC[r].max()), 3),
                                     int((NEC[r] > 0.05).sum())] for r in READ_F},
            **{f"sufficiency|{R[r]}": [round(float(SUF[r].min()), 3), round(float(SUF[r].max()), 3),
                                       int((SUF[r] > 0.05).sum()), int((SUF[r] < -0.05).sum())] for r in READ_F}}
VAL["F_meaning"] = "necessity|X = [min, max, n types > 0.05] of the 54 type medians; sufficiency|X adds n < -0.05"

# sensitivity: the plotted medians without the fly-readout pairs whose damaged value is above healthy
SENS = {}
for r in SEL:
    if not EXCESS[r]:
        continue
    for u, _, _ in KINDS:
        v = B[(f"f_{r}", u)].drop(EXCESS[r], errors="ignore")
        check(f"f.{u}.{R[r]}.all_deficit_only", mci(v))
        for s_ in (0.3, 0.5):
            vs = v[[stage_of[s] == s_ for s in v.index]].dropna()
            SENS[f"A|{u}|{R[r]}|s{s_}"] = [round(float(vs.median()), 4), int(len(vs))]
    for cls, _, _ in CLASSES:
        v = CL[CL.unit == cls].set_index("seed")[f"f_{r}"].drop(EXCESS[r], errors="ignore").dropna()
        SENS[f"B|{cls}|{R[r]}"] = [round(float(v.median()), 4), int(len(v))]
    for k in (1, 3, 10, 30):
        v = CU[(CU.r == r) & (CU.k == k)].set_index("seed").frac.drop(EXCESS[r], errors="ignore").dropna()
        SENS[f"D|top{k}|{R[r]}"] = [round(float(v.median()), 4), int(len(v))]
VAL["sensitivity_without_damaged_above_healthy"] = SENS
VAL["sensitivity_meaning"] = ("Same medians as plotted, leaving out the flies whose damaged value is above healthy "
                              f"({ {R[r]: EXCESS[r] for r in SEL if EXCESS[r]} }): [median, n]. Computed here, not in "
                              "facts_exp1.json except the pooled decomposition (f.*.all_deficit_only, checked).")
print(f"{N_CHECKS[0]} checks against facts_exp1.json / summary tables passed")


# ------------------------------------------------------------------ figure
# layout: every text item is 8 pt (panel letters 11 pt) at the printed size. Axes are placed in inches
# (x from the left edge, y from the top edge) so that tick labels, legends and titles have the room 8 pt needs.
FS = 8.0
W_IN, H_IN = 7.5, 8.5


def place(fig, x, y, w, h):
    """Axes with its top-left corner x, y inches from the top-left of the figure, w by h inches."""
    return fig.add_axes([x / W_IN, 1 - (y + h) / H_IN, w / W_IN, h / H_IN])


def letter(fig, s, x, y):
    """Panel letter with its baseline y inches from the top (11 pt bold, as in make_figures.panel)."""
    fig.text(x / W_IN, 1 - y / H_IN, s, fontsize=11, fontweight="bold", va="baseline", ha="left", color=mf.INK)


def head(fig, s, x, y):
    """Panel title on the baseline of the panel letter."""
    fig.text(x / W_IN, 1 - y / H_IN, s, fontsize=FS, va="baseline", ha="left", color=mf.INK)


def panel_a(ax):
    xs_off = {"restore_dead_all": -0.28, "restore_syn_all": 0.0, "no_shift": 0.28}
    for i, r in enumerate(SEL):
        for u, lab, col in KINDS:
            v = B[(f"f_{r}", u)]
            for s_, dx, filled in ((0.3, -0.065, False), (0.5, 0.065, True)):
                vs = v[[stage_of[s] == s_ for s in v.index]].dropna()
                x = i + xs_off[u] + dx
                draw_strip(ax, x, vs, EXCESS[r], col, filled, w=0.028, s=6.0)
                # median bar: solid for s = 0.5, hollow (white core) for s = 0.3
                ax.plot([x - 0.05, x + 0.05], [vs.median()] * 2, color=col, lw=2.1, solid_capstyle="butt", zorder=4)
                if not filled:
                    ax.plot([x - 0.042, x + 0.042], [vs.median()] * 2, color="white", lw=0.8,
                            solid_capstyle="butt", zorder=5)
        v = B[(f"f_{r}", "restore_dead_all")].dropna()
        n3 = int(sum(stage_of[s] == 0.3 for s in v.index))
        ax.text(i, YHI + 0.085, f"n = {len(v)} ({n3} + {len(v) - n3})", ha="center", va="bottom", fontsize=FS,
                color=mf.GREY)
    for i in range(1, len(SEL)):
        ax.axvline(i - 0.5, color=mf.LIGHT, lw=0.5, zorder=0)
    ax.axhline(0, color=mf.GREY, lw=0.5, zorder=1)
    ax.axhline(1, color=mf.GREY, lw=0.5, ls=":", zorder=1)
    ax.set_xticks(range(len(SEL)), [SHORT[r].replace("\n", " ") for r in SEL])
    ax.tick_params(axis="x", pad=2.5)
    ax.set_xlim(-0.5, len(SEL) - 0.5)
    ax.set_ylim(YLO - 0.07, YHI + 0.31)
    ax.set_yticks([-0.5, 0, 0.5, 1, 1.5])
    ax.set_ylabel("Fraction of the fly's\ndeficit restored")
    # two legend rows above the axes: what was undone (colour), then severity and special markers
    kw = dict(loc="lower left", fontsize=FS, handletextpad=0.25, columnspacing=1.5, borderaxespad=0.0, borderpad=0.0)
    h1 = [Line2D([], [], color=c, lw=0, marker="s", ms=5, label=lab) for _, lab, c in KINDS]
    h2 = [Line2D([], [], color=mf.GREY, lw=0, marker="o", mfc="white", ms=4, label="s = 0.3"),
          Line2D([], [], color=mf.GREY, lw=0, marker="o", ms=4, label="s = 0.5"),
          Line2D([], [], color=mf.INK, lw=0, marker="x", ms=4.5, label="Damaged above healthy"),
          Line2D([], [], color=mf.GREY, lw=0, marker="^", ms=4.2, label=f"Outside {YLO:g} to {YHI:g}".replace("-", "−"))]
    ax.add_artist(ax.legend(handles=h1, bbox_to_anchor=(0.0, 1.125), ncol=3, **kw))
    ax.legend(handles=h2, bbox_to_anchor=(0.0, 1.025), ncol=4, **kw)


def panel_b(ax):
    n_c = len(CLASSES) + 1
    width = 0.8 / n_c
    for i, r in enumerate(SEL):
        for j, (cls, lab, col) in enumerate(CLASSES):
            m, lo, hi, n = VAL["B"][f"{cls}|{R[r]}"]
            x = i - 0.4 + width * (j + 0.5)
            ax.plot([x, x], [lo, hi], color=col, lw=1.0)
            ax.plot(x, m, "o", ms=3.5, color=col, label=lab if i == 0 else None, zorder=3)
            if cls == "ALPN" and f"ALPN_nonzero_WB_flies|{R[r]}" in VAL["B"]:
                m2 = VAL["B"][f"ALPN_nonzero_WB_flies|{R[r]}"][0]
                ax.plot(x, m2, "o", ms=3.9, mfc="white", mec=col, mew=0.9, zorder=4,
                        label="PNs, 36 flies with a whole-brain effect" if i == 0 else None)
        m, lo, hi, n = VAL["B"][f"restore_dead_all|{R[r]}"]
        x = i - 0.4 + width * (n_c - 0.5)
        ax.plot([x, x], [lo, hi], color=mf.INK, lw=1.0)
        ax.plot(x, m, "s", ms=3.6, color=mf.INK, label="All removed" if i == 0 else None, zorder=3)
    y_top = 1.415                              # room above y = 1 for the four legend rows
    for i in range(1, len(SEL)):
        ax.axvline(i - 0.5, color=mf.LIGHT, lw=0.5, zorder=0, ymax=(1.0 + 0.06) / (y_top + 0.06))
    ax.axhline(0, color=mf.GREY, lw=0.5)
    ax.set_xticks(range(len(SEL)), [SHORT[r] for r in SEL])
    ax.tick_params(axis="x", pad=2.5)
    ax.set_xlim(-0.5, len(SEL) - 0.5)
    ax.set_ylim(-0.06, y_top)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["left"].set_bounds(-0.06, 1.0)
    ax.set_ylabel("Fraction of the deficit restored\n(median, 95% CI)", y=(0.5 + 0.06) / (y_top + 0.06))
    hs, ls_ = ax.get_legend_handles_labels()
    # three rows of two (ALPN, DN, MBON | KC, cb_intrinsic, all removed), then the open-circle entry on its own row
    kw = dict(loc="upper left", fontsize=FS, handlelength=0.9, handletextpad=0.3, columnspacing=1.6, borderaxespad=0.0,
              borderpad=0.0, labelspacing=0.3)
    order = [0, 2, 3, 4, 5, 6]
    ax.add_artist(ax.legend([hs[k] for k in order], [ls_[k] for k in order], bbox_to_anchor=(0.0, 1.0), ncol=2, **kw))
    # one legend row is the Arial line height (0.9225 em) plus labelspacing; as a fraction of the axes height
    row = (0.9225 + kw["labelspacing"]) * FS / 72 / (ax.get_position().height * ax.figure.get_figheight())
    ax.legend([hs[1]], [ls_[1]], bbox_to_anchor=(0.0, 1.0 - 3 * row), **kw)


def panel_c(ax):
    cols = ["all_removed", "DN_class", "best_type", "DNp01"]
    labs = ["All\nremoved", "DN\nclass", "Best\ntested\ntype", "GF\n(DNp01)"]
    gfc = mf.COL["GF_loom_hz"]
    for s in fail:
        y = CTAB.loc[s, cols].to_numpy(float)
        tested = s in dnp.index
        col = gfc if tested else mf.GREY
        ok = np.isfinite(y)
        ax.plot(np.arange(4)[ok], y[ok], "-", color=col, lw=0.8, alpha=0.9 if tested else 0.55, zorder=2)
        ax.scatter(np.arange(4)[ok], y[ok], s=12, color=col, edgecolor="white", lw=0.35, zorder=3)
    ax.axhline(0, color=mf.GREY, lw=0.5)
    ax.axhline(1, color=mf.GREY, lw=0.5, ls=":")
    ax.set_xticks(range(4), labs)
    ax.tick_params(axis="x", pad=2.5)
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(-0.05, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Fraction of the giant fibre\ndeficit restored")
    # direct labels instead of a legend
    ax.text(2.14, 0.80, f"DNp01\ntested ({len(dnp)})", ha="left", va="top", fontsize=FS, color=gfc, linespacing=1.1)
    ax.text(2.14, 0.1, f"DNp01 not\ntested ({len(fail) - len(dnp)})", ha="left", va="bottom", fontsize=FS,
            color=mf.GREY, linespacing=1.1)


def panel_d(ax):
    ks = [1, 3, 10, 30]
    for r in SEL:
        m = [VAL["D"][f"top{k}|{R[r]}"][0] for k in ks]
        ax.plot(range(4), m, "-o", ms=3.0, lw=1.1, color=mf.COL[r], label=SHORT[r].replace("\n", " "),
                clip_on=False, zorder=3)        # the MN9 median at k = 30 is 0: draw its marker whole, on the axis
        ax.plot([4.7], [VAL["D"][f"all_removed|{R[r]}"][0]], "s", ms=4.0, color=mf.COL[r])
    ax.axvline(3.85, color=mf.LIGHT, lw=0.6)
    ax.set_xticks([0, 1, 2, 3, 4.7], ["1", "3", "10", "30", "All\nremoved"])
    ax.tick_params(axis="x", pad=2.5)
    ax.set_xlim(-0.3, 5.1)
    ax.set_xlabel("Top-k tested types,\nrestored together", labelpad=2, x=0.36)
    ax.set_ylabel("Fraction restored (median)")
    ax.set_ylim(0, 0.95)
    ax.legend(loc="upper left", bbox_to_anchor=(0.03, 1.0), ncol=1, fontsize=FS, handlelength=1.3, handletextpad=0.4,
              borderaxespad=0.0, borderpad=0.0, labelspacing=0.3)


E_CELL, E_HEAD, E_GAP = 0.30, 0.065, 0.9        # panel E: row height (in), heading baseline above its block (in),
E_TOP = 0.5 + (0.13 + E_HEAD) / E_CELL          # space between the blocks (rows); rows above the centre of row 0, so
#                                                 that the first heading sits on the panel letter's baseline (0.13 in)


def panel_e(ax, cax):
    cmap, norm = plt.get_cmap("Greys"), Normalize(0, 1)
    rows = [(m, lab, 0) for m, lab in PARAM] + [(m, lab, 1) for m, lab in NOISE]
    gap = E_GAP
    ypos = [k + (gap if g else 0) for k, (_, _, g) in enumerate(rows)]
    for k, (m, lab, g) in enumerate(rows):
        y = ypos[k]
        for j, r in enumerate(SEL):
            c = cells[(m, r)]
            v = c["rho"]
            ax.add_patch(Rectangle((j - 0.5, y - 0.5), 1, 1, fc=cmap(norm(v)), ec="white", lw=0.6))
            txt = f"{v:.2f}\n{c['ov']:.1f}" + ("†" if c["degenerate"] else "")
            ax.text(j, y, txt, ha="center", va="center", fontsize=FS, linespacing=1.0,
                    color="white" if v > 0.6 else mf.INK, fontweight="bold" if c["pass"] else "normal")
            if c["pass"]:
                ax.add_patch(Rectangle((j - 0.46, y - 0.46), 0.92, 0.92, fill=False, ec="#c0392b", lw=1.1, zorder=5))
    ax.set_xlim(-0.5, len(SEL) - 0.5)
    ax.set_ylim(ypos[-1] + 0.5, -E_TOP)
    ax.set_xticks(range(len(SEL)), [SHORT[r] for r in SEL])
    ax.set_yticks(ypos, [lab for _, lab, _ in rows])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0, pad=3)
    n_par = sum(cells[(m, r)]["pass"] for m, _ in PARAM for r in SEL)
    n_noi = sum(cells[(m, r)]["pass"] for m, _ in NOISE for r in SEL)
    up = E_HEAD / E_CELL                         # block headings: baseline E_HEAD inches above the block
    ax.text(-0.5, -0.5 - up, f"Parameter variants ({n_par} of {len(PARAM) * len(SEL)} cells pass)", ha="left",
            va="baseline", fontsize=FS, color=mf.INK, fontweight="bold")
    ax.text(-0.5, ypos[len(PARAM)] - 0.5 - up, f"Input-noise realisations ({n_noi} of {len(NOISE) * len(SEL)} pass)",
            ha="left", va="baseline", fontsize=FS, color=mf.INK, fontweight="bold")
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1], labels=["0.0", "0.5", "1.0"])
    cb.ax.tick_params(labelsize=FS, length=2, pad=2)
    cb.outline.set_linewidth(0.4)
    cax.text(-0.05, 0.5, "Rank correlation with base model", transform=cax.transAxes, ha="right", va="center",
             fontsize=FS, color=mf.INK)


def panel_f(ax):
    for i, r in enumerate(READ_F):
        col = mf.COL[r]
        for j, (tab_, mk, filled) in enumerate(((NEC, "o", True), (SUF, "^", False))):
            y = tab_[r].dropna()
            x = i + (j - 0.5) * 0.36 + RNG.uniform(-0.07, 0.07, len(y))
            ax.scatter(x, y, s=9, marker=mk, facecolor=col if filled else "none", edgecolor=col, lw=0.6,
                       alpha=0.8, zorder=3)
    ax.axvspan(-0.5, 3.5, color="#eef1f3", zorder=0, lw=0)
    ax.text(1.5, -0.262, "restored alone: identical\nto the short protocol", ha="center", va="bottom", fontsize=FS,
            color=mf.GREY, linespacing=1.1)
    ax.axhline(0, color=mf.GREY, lw=0.5, zorder=1)
    ax.set_xticks(range(len(READ_F)), [SHORT[r] for r in READ_F])
    ax.tick_params(axis="x", pad=2.5)
    ax.set_xlim(-0.5, len(READ_F) - 0.5)
    ax.set_ylim(-0.27, 0.32)
    ax.set_ylabel("Fraction of the fly's deficit\n(type median over flies)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=4, color=mf.GREY, label="Restored alone in the damaged fly"),
                       Line2D([], [], marker="^", ls="", ms=4, mfc="none", color=mf.GREY,
                              label="Removed alone from its healthy copy")],
              loc="lower left", bbox_to_anchor=(0.0, 1.025), ncol=1, fontsize=FS, handlelength=0.9, handletextpad=0.3,
              borderaxespad=0.0, borderpad=0.0, labelspacing=0.3)


def fig5():
    fig = plt.figure(figsize=(W_IN, H_IN))
    x0, x1 = 0.60, 7.36                         # left edge of the first-column axes, right edge of the content
    # row 1: A
    letter(fig, "A", 0.0, 0.13)
    panel_a(place(fig, x0, 0.42, x1 - x0, 1.62))
    # row 2: B, C, D (the same axes height; B keeps its legend inside, above the data)
    y2, h2 = 2.42, 2.15
    letter(fig, "B", 0.0, y2 + 0.13)
    panel_b(place(fig, x0, y2 + 0.24, 2.62, h2))
    letter(fig, "C", 3.33, y2 + 0.13)
    head(fig, f"GF output < 100 Hz ({len(fail)} flies)", 3.83, y2 + 0.13)
    panel_c(place(fig, 3.83, y2 + 0.24, 1.60, h2))
    letter(fig, "D", 5.62, y2 + 0.13)
    panel_d(place(fig, 5.99, y2 + 0.24, 1.25, h2))
    # row 3: E (colour bar below the grid), F
    y3 = 5.56
    letter(fig, "E", 0.0, y3 + 0.13)
    xe, we, he = 0.84, 2.64, (len(PARAM) + len(NOISE) - 0.5 + E_GAP + E_TOP) * E_CELL
    cax = place(fig, xe + we - 1.50, y3 + he + 0.41, 1.50, 0.09)
    panel_e(place(fig, xe, y3, we, he), cax)
    xf = 4.17
    letter(fig, "F", 3.62, y3 + 0.13)
    head(fig, f"Full protocol, {len(NEC)} types, 12 flies at s = 0.5", xf, y3 + 0.13)
    panel_f(place(fig, xf, y3 + 0.52, x1 - xf, he + 0.36 - 0.52))
    mf.save(fig, "Fig5")


fig5()
(mf.OUT / "Fig5_values.json").write_text(json.dumps(VAL, indent=1, default=lambda o: o.item() if hasattr(o, "item")
                                                    else str(o)))
print("wrote", mf.OUT / "Fig5_values.json")
