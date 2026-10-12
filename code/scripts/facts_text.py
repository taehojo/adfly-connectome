"""Facts table "text": numbers quoted in the text
that no other facts table holds.

  PYTHONIOENCODING=utf-8 python code/scripts/facts_text.py

(a) Projection neuron spikes in the whole-brain count (results/battery/stage_curve_x4.csv): share of the
    count that is projection neuron spikes, the whole-brain count without them at s = 1.0, and the
    correlation of projection neuron rate with the count without them at s = 0.5.
(b) Superclasses exempt from load (results/validation/baseline_nostim.parquet).
(c) Learning index / MBON rate pooled over s = 0.05 to 0.3, and the 40,000-resample interval and seed
    range of the pooled value over s = 0.10 to 0.3 (functions and seeds of facts_battery.py).
(d) Healthy flies: naive valence against MBON rate and against valence after training; learning index
    against MBON rate with naive valence controlled.
(e) Exact permutation p of the rank correlation between AUC and change relative to spread (readouts,
    AUC, change and spread as in facts_detection.py; every pairing of the two rank vectors).
(f) Projection neuron rate between the two runs of the 16 healthy pilot pairs: p of the Spearman rho.
(g) High-activity flies by the own-group criterion from s = 0.5: rates against the healthy maxima.
(h) Values on a rounding edge, with more digits than the other facts tables keep (others are under their
    themes: the keys ending in _3dp, _4dp or _5dp).
(i) Shuffled map at equal neurons removed against the shuffled map at matched burden, the shuffled/hub
    ratio against each of the four hub groups at s = 0.5, their projection neuron rates, and the matched
    burden levels with three decimals (results/battery/*_x4.csv, results/exp2/battery.csv, matched.json).
(j) Within-fly analysis (results/exp1/screen, robust, confirm): most effective tested type for giant fibre
    rate counted with ties, descending-neuron class rescue by group of flies, neurons removed per fly,
    lesion sizes, readouts above healthy in the robustness and full-protocol runs, series giant fibre
    rates of the screen flies below 100 Hz.
(k) Joint cohort (results/joint/cohort_joint.csv): trend of the arena readouts with Holm correction over
    the sixteen readouts, giant fibre rate against jumps per threat within severity groups, flies with a
    defined arena readout.
(l) Battery designs: flies that sat in the batch position of a healthy series fly; replicate-group
    Kruskal-Wallis tests below 0.05.
(m) Design of the exploratory odour intensity sweep (results/calibration/odor_intensity_sweep.csv) and the
    number of simulations of the within-fly analysis.
(n) Figures: the hub-map MBON interval of Fig 3C as drawn by make_figures.py against the text value;
    the largest interquartile range not drawn in Fig 4D against the symbol size of make_fig4_burden.py.

Conventions as in the other facts scripts: one flat dict of named facts, plus "_meaning" (what each key
is: population, n, criterion, statistic, interval type and resamples, test) and "_source" (file and
column per key). Medians are of per-fly values. No simulation of the model is run and no existing file
is changed.

Medians as % of healthy, bootstrap intervals with their seeds and resample counts, Mann-Whitney,
Kruskal-Wallis, Spearman, partial rank correlation, Holm correction, AUC, robust CV and batch positions
are computed by the functions of facts_battery.py, facts_detection.py, facts_exp2.py and
analyze_joint.py themselves: their definitions are compiled from the source of those scripts without
running them (defs_from), so a value recorded here cannot drift from the way the same statistic is
computed elsewhere. Where a value recorded here repeats one of another facts table, the two are compared
and the script stops on a mismatch. Deterministic: bootstrap generators are seeded as in
facts_battery.py (SEED and the label of the interval); (n) uses the generator of the figure script.

Writes results/facts/facts_text.json.
"""
import ast
import glob
import itertools
import json
import math
import re
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())  # code/scripts locally, scripts/ on the server
SCRIPTS = ROOT / "code/scripts"
sys.path.insert(0, str(SCRIPTS))
import analyze_exp1 as ax  # noqa: E402  (frac_table: restored fraction with the 5% deficit rule)

OUT = ROOT / "results/facts"
OUT.mkdir(parents=True, exist_ok=True)
BAT = ROOT / "results/battery"
FACTS_DIR = ROOT / "results/facts"
P_SERIES, P_PILOT = BAT / "stage_curve_x4.csv", BAT / "stage_curve.csv"
P_COMP, P_VULN, P_SCOPE = BAT / "components_x4.csv", BAT / "vulnerability_x4.csv", BAT / "scope_x4.csv"
P_EXP2, P_MATCH = ROOT / "results/exp2/battery.csv", ROOT / "results/exp2/matched.json"
P_JOINT, P_JSTATS = ROOT / "results/joint/cohort_joint.csv", ROOT / "results/joint/stats_joint.json"
P_NEU = ROOT / "results/validation/baseline_nostim.parquet"
E1 = ROOT / "results/exp1"
S_BAT, S_DET, S_EXP2 = SCRIPTS / "facts_battery.py", SCRIPTS / "facts_detection.py", SCRIPTS / "facts_exp2.py"
S_JOINT, S_FIG, S_FIG4 = SCRIPTS / "analyze_joint.py", SCRIPTS / "make_figures.py", SCRIPTS / "make_fig4_burden.py"
P_INT = ROOT / "results/calibration/odor_intensity_sweep.csv"
SEL = ["brain_spikes_A", "LH_hz_A", "KC_hz_A", "MBON_hz_A", "MN9_sugar_hz", "GF_loom_hz"]
MIN_DEF = 0.05                       # exp1_stats.py MIN_DEF / facts_exp1.json
GF_FAIL_HZ = 100.0                   # giant fibre output below 100 Hz (mean of the two giant fibres)

FACTS, MEANING, SOURCE = {}, {}, {}


def put(key, value, meaning, source):
    if key in FACTS:
        raise KeyError(f"duplicate key {key}")
    FACTS[key] = value
    MEANING[key] = meaning
    SOURCE[key] = source


def rnd(x, k=4):
    if isinstance(x, (list, tuple)):
        return [rnd(v, k) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    x = float(x)
    return None if not np.isfinite(x) else round(x, k)


def rel(p):
    return str(Path(p).relative_to(ROOT)).replace("\\", "/")


def must(pattern, text, what):
    m = re.search(pattern, text)
    if not m:
        raise AssertionError(f"code changed: {what} ({pattern})")
    return m


def defs_from(script, names, env=None):
    """Compile the named top-level functions and constants of another script without running that script."""
    tree = ast.parse(Path(script).read_text(encoding="utf-8"))
    body, found = [], set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            body.append(node)
            found.add(node.name)
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in names):
            body.append(node)
            found.add(node.targets[0].id)
    missing = set(names) - found
    if missing:
        raise AssertionError(f"code changed: {sorted(missing)} not defined in {rel(script)}")
    ns = {"np": np, "pd": pd, "st": st, "zlib": zlib, **(env or {})}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(script), "exec"), ns)
    return ns


def same(a, b, what, tol=0.0):
    """Stop if a value recomputed here differs from the one another facts table holds."""
    fa, fb = np.ravel(np.asarray(a, float)), np.ravel(np.asarray(b, float))
    if fa.shape != fb.shape or not np.allclose(fa, fb, rtol=0, atol=tol + 1e-12, equal_nan=True):
        raise AssertionError(f"{what}: recomputed {a} against recorded {b}")


def facts(name):
    return json.loads((FACTS_DIR / f"facts_{name}.json").read_text(encoding="utf-8"))


B = defs_from(S_BAT, ["SEED", "N_BOOT", "MBON_FLOOR", "BOOT_ALT_SEEDS", "BATTERY_BATCH", "R", "REP_READS", "sk", "r1", "r3",
                      "pg", "rng_for", "clean", "_boot_bounds", "mw", "kw", "spear", "partial_spearman", "prep", "_slots"])
D = defs_from(S_DET, ["MBON_FLOOR", "PREC", "GRADED", "RT", "auc", "rcv", "shift_pct", "rank_corrs", "sig", "load"])
X = defs_from(S_EXP2, ["med", "mw", "pv"])
J = defs_from(S_JOINT, ["CIRCUIT", "BEHAV", "MBON_FLOOR", "add_normalised_li", "holm"])
r1, r3, pg = B["r1"], B["r3"], B["pg"]
FB, FD, FE1, FE2 = facts("battery"), facts("detection"), facts("exp1"), facts("exp2")
SEED, N_BOOT = B["SEED"], B["N_BOOT"]
BOOT = (f"95% percentile bootstrap of median(group) / median(healthy) x 100, flies resampled with replacement within the group "
        f"and within the healthy flies, numpy default_rng([{SEED}, crc32(label)]) as in facts_battery.py")


def boot_pct(x, h, label, n_boot=None):
    """[lo, hi] of median(x) / median(h) x 100 with the generator of facts_battery.py for that label."""
    lo, hi = B["_boot_bounds"](B["clean"](x), B["clean"](h), B["rng_for"](label), n_boot or N_BOOT, 100.0)
    return float(lo), float(hi)


# ===================================================================== data
sc0 = pd.read_csv(P_SERIES)                              # all rows, in the order in which they were run
sc = B["prep"](sc0)
sc = sc[sc.scope == "pan"].copy()
assert len(sc) == len(sc0) == 384
H = sc[sc.stage == 0]
STAGES = sorted(s for s in sc.stage.unique() if s > 0)
co, vu, sp = (B["prep"](pd.read_csv(p)) for p in (P_COMP, P_VULN, P_SCOPE))
e2 = pd.read_csv(P_EXP2)
assert len(H) == 64 and all((sc.stage == s).sum() == 32 for s in STAGES) and len(STAGES) == 10


def grp(s):
    g = sc[np.isclose(sc.stage, s)]
    assert len(g) == (64 if s == 0 else 32), (s, len(g))
    return g


# ===================================================================== (a) projection neuron spikes in the whole-brain count
N_PN = FB["pn_n_ALPN"]
for d_ in (sc, co, vu, sp, e2):
    k_ = d_.PN_hz_A.to_numpy() * N_PN                    # PN_hz_A is the mean spike count of the ALPN class in the 1-s odour A window
    assert np.allclose(k_, np.round(k_), atol=1e-6), "PN_hz_A x 686 is not a spike count"
ab_src = (SCRIPTS / "assay_battery.py").read_text(encoding="utf-8")
assert '"PN_hz_A": float(preA[b, pn].mean())' in ab_src and '"brain_spikes_A": float(preA[b].sum())' in ab_src, "code changed: readouts"
assert 'pn = np.flatnonzero((cx.cls == "ALPN").to_numpy())' in ab_src, "code changed: PN population"
DEF_PN = (f"Projection neuron spikes = PN_hz_A x {N_PN} (PN_hz_A is the mean spike count over all {N_PN} antennal lobe projection "
          f"neurons, class ALPN, in the 1-s presentation of odour A before training, and brain_spikes_A is the spike count of all "
          f"neurons in the same window: assay_battery.py)")
SRC_A = rel(P_SERIES) + " columns PN_hz_A, brain_spikes_A (scope pan)"
put("pn_spikes_n_projection_neurons", int(N_PN),
    f"Antennal lobe projection neurons averaged in PN_hz_A (facts_battery pn_n_ALPN); PN_hz_A x {N_PN} is a whole number of spikes "
    "in every fly of the four battery designs and of the burden-matched comparison (checked here).", "results/facts/facts_battery.json pn_n_ALPN")
share = {}
for lab, s in (("healthy", 0.0), ("s0.5", 0.5), ("s1", 1.0)):
    g = grp(s)
    share[lab] = r1(100 * np.median(N_PN * g.PN_hz_A / g.brain_spikes_A))
put("series_pn_share_of_brain_count_pct", [share["healthy"], share["s0.5"], share["s1"]],
    f"Severity series, share of the whole-brain spike count during odour A that is projection neuron spikes, %: median over flies "
    f"of the per-fly share [64 healthy flies, 32 flies at s = 0.5, 32 flies at s = 1.0]. {DEF_PN}.", SRC_A)
g = grp(0.5)
excl = g.brain_spikes_A - N_PN * g.PN_hz_A
rho_in, rho_ex = B["spear"](g.PN_hz_A, g.brain_spikes_A), B["spear"](g.PN_hz_A, excl)
same(r3(rho_in[0]), FB["within_s0.5_rho_PN_vs"]["brain"], "rho PN vs whole-brain at s = 0.5")
put("series_s0.5_rho_PN_vs_brain_excl_PN", [r3(rho_ex[0]), pg(rho_ex[1]), rho_ex[2]],
    f"Severity series, the 32 flies at s = 0.5: Spearman rho, p (t approximation) and n between the projection neuron rate "
    f"(PN_hz_A) and the whole-brain spike count with projection neuron spikes excluded (brain_spikes_A - PN_hz_A x {N_PN}). "
    f"With the spikes included the correlation is series_s0.5_rho_PN_vs_brain_incl_PN. {DEF_PN}.", SRC_A)
put("series_s0.5_rho_PN_vs_brain_incl_PN", [r3(rho_in[0]), pg(rho_in[1]), rho_in[2]],
    "Same flies: Spearman rho, p and n between the projection neuron rate and the whole-brain spike count as measured (projection "
    "neuron spikes included); the rho equals facts_battery within_s0.5_rho_PN_vs['brain'] (checked here).", SRC_A)
g1 = grp(1.0)
ex1, exh = g1.brain_spikes_A - N_PN * g1.PN_hz_A, H.brain_spikes_A - N_PN * H.PN_hz_A
lo, hi = boot_pct(ex1, exh, "text:series1.0brain_exclPN")
put("series_s1_brain_excl_PN_pct", [r1(100 * np.median(ex1) / np.median(exh)), r1(lo), r1(hi)],
    f"Severity series, s = 1.0 (32 flies against 64 healthy): median whole-brain spike count with projection neuron spikes "
    f"excluded in both groups, as % of the healthy median [estimate, lo, hi]. Interval: {BOOT}, label "
    f"'text:series1.0brain_exclPN', {N_BOOT:,} resamples (the text quotes the estimate only). {DEF_PN}.", SRC_A)
same(r1(100 * g1.brain_spikes_A.median() / H.brain_spikes_A.median()), FB["series_s1_brain_pct"][0], "whole-brain at s = 1.0")
put("series_s1_brain_incl_PN_pct", r1(100 * g1.brain_spikes_A.median() / H.brain_spikes_A.median()),
    "Same comparison with the count as measured (projection neuron spikes included): estimate only; equals "
    "facts_battery series_s1_brain_pct[0] (checked here), which holds its interval.", SRC_A)

# ===================================================================== (b) superclasses exempt from load
neu = pd.read_parquet(P_NEU)
assert len(neu) == 166700
assert 'self.sensory = n.superclass.str.contains("sensory").to_numpy()' in (ROOT / "code/adfly/model.py").read_text(encoding="utf-8"), "code changed: sensory"
sens = neu.superclass.fillna("").str.contains("sensory")
names = sorted(neu.superclass[sens].unique())
same(int(sens.sum()), FB["n_neurons_sensory_exempt"], "sensory neurons")
put("sensory_exempt_superclasses", [len(names), names, int(sens.sum()), r1(100 * sens.mean())],
    "Neurons exempt from pathology load, those whose superclass name contains 'sensory' (model.py, Connectome.sensory): [number of such "
    "superclasses, their names, neurons, % of the 166,700 neurons of the network]. The neuron count equals facts_battery "
    "n_neurons_sensory_exempt (checked here).", rel(P_NEU) + " column superclass (MaleCNS neuron table in simulation order)")

# ===================================================================== (c) learning index / MBON rate, pooled
assert B["MBON_FLOOR"] == D["MBON_FLOOR"] == J["MBON_FLOOR"] == 0.5
DEF_LIN = (f"Learning index / MBON rate (LI / MBON_hz_A, defined where MBON_hz_A >= {B['MBON_FLOOR']} Hz) pooled over the flies "
           "of the named severities, median as % of the median of the 64 healthy flies")
SRC_C = rel(P_SERIES) + " columns LI, MBON_hz_A, stage (scope pan)"


def pooled(lo_s, hi_s, name):
    d = sc[(sc.stage >= lo_s - 1e-9) & (sc.stage <= hi_s + 1e-9)].LI_norm
    lab = f"lin{name}"
    est = 100 * np.median(B["clean"](d)) / np.median(B["clean"](H.LI_norm))
    b4, b40 = boot_pct(d, H.LI_norm, lab), boot_pct(d, H.LI_norm, f"{lab}#big", 10 * N_BOOT)
    alt = np.array([b4] + [boot_pct(d, H.LI_norm, f"{lab}#alt{i}") for i in range(B["BOOT_ALT_SEEDS"])])
    return d, lab, est, b4, b40, alt


d, lab, est, b4, b40, alt = pooled(0.1, 0.3, "s01_03")
n_def = int(d.notna().sum())
same([r1(est), r1(b4[0]), r1(b4[1]), B["mw"](d, H.LI_norm), n_def], FB["LInorm_pooled_s01_03_vs_healthy"], "pooled ratio s = 0.10 to 0.3")
put("LInorm_pooled_s01_03_ci_40000", [r1(b40[0]), r1(b40[1])],
    f"{DEF_LIN}, s = 0.10 to 0.3 ({n_def} flies, all with a defined ratio): 95% interval [lo, hi] from {10 * N_BOOT:,} resamples, "
    f"the bounds quoted in the text (67 to 99.8%). {BOOT}, label '{lab}#big'. The estimate, the {N_BOOT:,}-resample interval, the "
    "Mann-Whitney p and n are facts_battery LInorm_pooled_s01_03_vs_healthy, which this script reproduces exactly (checked here).", SRC_C)
put("LInorm_pooled_s01_03_ci_4000_seed_range", [[r1(alt[:, 0].min()), r1(alt[:, 0].max())], [r1(alt[:, 1].min()), r1(alt[:, 1].max())], len(alt)],
    f"Same interval at {N_BOOT:,} resamples under the reported seed (label '{lab}') and {B['BOOT_ALT_SEEDS']} further seeds (labels "
    f"'{lab}#alt0' to '#alt{B['BOOT_ALT_SEEDS'] - 1}'): [[lowest, highest lower bound], [lowest, highest upper bound], number of "
    "seeds]. The upper bound crosses 100% between seeds, which is why the text quotes the 40,000-resample value.", SRC_C)
d, lab, est, b4, b40, alt = pooled(0.05, 0.3, "s005_03")
n_def = int(d.notna().sum())
assert n_def == len(d) == 160
put("LInorm_pooled_s005_03_vs_healthy", [r1(est), r1(b40[0]), r1(b40[1]), B["mw"](d, H.LI_norm), n_def],
    f"{DEF_LIN}, s = 0.05 to 0.3 ({n_def} of {len(d)} flies with a defined ratio): [estimate, lo, hi, two-sided Mann-Whitney p "
    f"against the healthy flies, n defined]. Interval from {10 * N_BOOT:,} resamples, the one quoted in the text (73 to 110%): "
    f"{BOOT}, label '{lab}#big'.", SRC_C)
put("LInorm_pooled_s005_03_ci_4000", [r1(b4[0]), r1(b4[1])],
    f"Same interval at {N_BOOT:,} resamples (label '{lab}', the label scheme of LInorm_pooled_* in facts_battery.py): [lo, hi]. "
    "Not quoted; it differs from the 40,000-resample interval by Monte Carlo error.", SRC_C)
put("LInorm_pooled_s005_03_ci_4000_seed_range", [[r1(alt[:, 0].min()), r1(alt[:, 0].max())], [r1(alt[:, 1].min()), r1(alt[:, 1].max())], len(alt)],
    f"Same interval at {N_BOOT:,} resamples under the reported seed and {B['BOOT_ALT_SEEDS']} further seeds (labels '{lab}#alt0' to "
    f"'#alt{B['BOOT_ALT_SEEDS'] - 1}'): [[lowest, highest lower bound], [lowest, highest upper bound], number of seeds].", SRC_C)
put("LInorm_pooled_s005_03_unrounded", [rnd(est, 2), rnd(b40[0], 2), rnd(b40[1], 2)],
    f"The estimate and the {10 * N_BOOT:,}-resample bounds of LInorm_pooled_s005_03_vs_healthy with two decimals, to show how far "
    "each is from a rounding edge.", SRC_C)

# ===================================================================== (d) healthy flies
SRC_D = rel(P_SERIES) + " rows stage = 0 (64 healthy flies); columns MBON_hz_A, V_pre_A, V_post_A, LI"
r_ = B["spear"](H.MBON_hz_A, H.V_pre_A)
put("healthy_rho_MBON_vs_naive_valence", [r3(r_[0]), pg(r_[1]), r_[2]],
    "The 64 healthy flies of the severity series (runs of one intact network under different input noise): Spearman rho, p "
    "(t approximation) and n between MBON rate (MBON_hz_A) and the naive valence of odour A (V_pre_A), both measured in the same "
    "1-s presentation of odour A before training.", SRC_D)
r_ = B["spear"](H.V_pre_A, H.V_post_A)
put("healthy_rho_naive_valence_vs_valence_after_training", [r3(r_[0]), pg(r_[1]), r_[2]],
    "Same flies: Spearman rho, p and n between the valence of odour A before training (V_pre_A) and after training (V_post_A).", SRC_D)
r_ = B["partial_spearman"](H.LI, H.MBON_hz_A, H.V_pre_A)
put("healthy_partial_LI_MBON_given_naive_valence", [r3(r_[0]), pg(r_[1]), int(len(H))],
    "Same flies: partial rank correlation of the learning index with MBON rate controlling for the naive valence of odour A, its p "
    "and n (partial_spearman of facts_battery.py: Pearson correlation of the residuals of rank(LI) and rank(MBON rate) after a "
    "linear fit on rank(naive valence)).", SRC_D)
r_ = B["spear"](H.LI, H.MBON_hz_A)
put("healthy_rho_LI_vs_MBON", [r3(r_[0]), pg(r_[1]), r_[2]],
    "Same flies: Spearman rho, p and n of the learning index with MBON rate (the uncontrolled correlation, for comparison with "
    "healthy_partial_LI_MBON_given_naive_valence).", SRC_D)

# ===================================================================== (e) exact permutation p of the rank correlation
dsc = D["load"](P_SERIES)
dsc = dsc[dsc.scope == "pan"].copy()
D.update(sc=dsc, H=dsc[dsc.stage == 0], cv={r: D["rcv"](dsc[dsc.stage == 0][r]) for r in D["PREC"]})


def perm_p(x, y):
    """Exact two-sided permutation p of Spearman's rho: share of all n! pairings of the two rank vectors with |rho| >= |rho observed|."""
    rx, ry = st.rankdata(x), st.rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    n, obs = len(rx), float((rx * ry).sum()) / den
    idx = np.array(list(itertools.permutations(range(n - 1))), dtype=np.int8)       # pairings of the first n - 1 values
    hit = total = 0
    for j in range(n):                                                              # the value paired with the last one
        rest = np.delete(ry, j)
        rho = (rest[idx] @ rx[:n - 1] + rx[n - 1] * ry[j]) / den
        hit += int((np.abs(rho) >= abs(obs) - 1e-9).sum())
        total += len(rho)
    assert total == math.factorial(n)
    return obs, hit / total, total


def change_over_spread(reads, s):
    """|median change| / healthy robust CV and AUC per readout, as rank_corrs of facts_detection.py computes them."""
    g = D["sc"][D["sc"].stage == s]
    z = [abs(D["shift_pct"](g[r], D["H"][r])) / 100 / D["cv"][r] for r in reads]
    a = [D["auc"](D["H"][r], g[r]) for r in reads]
    return z, a


DEF_E = ("AUC = direction-free Mann-Whitney AUC of the 32 flies of that severity against the 64 healthy flies; change = |group median "
         "- healthy median| / |healthy median|; spread = robust CV of the healthy flies, interquartile range / 1.349 / |median| "
         "(auc, shift_pct, rcv of facts_detection.py). The exact p is the share of all n! pairings of the two rank vectors whose "
         "|rho| is at least the observed |rho| (two-sided, ties kept as mid-ranks); the t-approximation p of scipy is in the "
         "facts_detection key named")
SRC_E = rel(P_SERIES) + " (scope pan), stage 0 against the named severity; readouts PREC / GRADED of facts_detection.py"
for s, tag in ((0.05, "0.05"), (0.1, "0.1"), (0.2, "0.2")):
    z, a = change_over_spread(D["GRADED"], s)
    rho, p, n_perm = perm_p(z, a)
    t_rho, t_p = D["rank_corrs"](D["GRADED"], s)[0]
    same([round(rho, 3), D["sig"](t_p)], FD[f"series_rankcorr_auc_vs_shift_over_spread_s{tag}"], f"rank correlation s = {tag}")
    assert abs(rho - t_rho) < 1e-12
    put(f"series_rankcorr_auc_vs_shift_over_spread_s{tag}_exact_p", [round(rho, 3), float(f"{p:.3g}"), len(z), n_perm],
        f"Severity series, the {len(z)} graded readouts (whole-brain spike count, lateral horn, projection neuron and Kenyon cell "
        f"rates, Kenyon cells recruited, MBON rate, naive valence, learning index) at s = {s:g}: [Spearman rho between AUC and "
        f"change / spread, exact two-sided permutation p, n readouts, pairings enumerated]. {DEF_E} "
        f"series_rankcorr_auc_vs_shift_over_spread_s{tag} (rho reproduced exactly, checked here).", SRC_E)
inv = D["rank_corrs"](D["GRADED"], 0.05)[2]
same([round(float(inv[0]), 3), D["sig"](inv[1])], FD["series_rankcorr_auc_vs_inverse_spread_s0.05"], "rank correlation with inverse spread")
put("series_rankcorr_auc_vs_inverse_spread_s0.05_p_4dp", [round(float(inv[0]), 3), round(float(inv[1]), 4)],
    "Same eight readouts at s = 0.05: Spearman rho between AUC and the inverse of the healthy spread alone, and its p (t "
    "approximation) with four decimals. facts_detection series_rankcorr_auc_vs_inverse_spread_s0.05 keeps three significant "
    "figures (0.955), which must not be rounded again: the value is 0.96 to two decimals.", SRC_E)
z, a = change_over_spread(D["PREC"], 0.1)
rho, p, n_perm = perm_p(z, a)
same([round(rho, 3), D["sig"](D["rank_corrs"](D["PREC"], 0.1)[0][1])], FD["series_rankcorr_auc_vs_shift_over_spread_s0.1_all10"], "rank correlation, ten readouts")
put("series_rankcorr_auc_vs_shift_over_spread_s0.1_all10_exact_p", [round(rho, 3), float(f"{p:.3g}"), len(z), n_perm],
    f"Same at s = 0.1 across all {len(z)} series readouts of the precision analysis (the eight graded readouts with MN9 drive and "
    f"giant fibre rate): [rho, exact two-sided permutation p, n readouts, pairings enumerated]. {DEF_E} "
    "series_rankcorr_auc_vs_shift_over_spread_s0.1_all10.", SRC_E)

# ===================================================================== (f) healthy reruns, projection neuron rate
pilot = pd.read_csv(P_PILOT).merge(sc0, on=["seed", "condition", "stage", "scope"], suffixes=("_p", "_x"))
hp = pilot[pilot.condition == "healthy"]
assert len(hp) == 16 and (hp.dead_frac_p == 0).all() and (hp.dead_frac_x == 0).all()
rho, p = st.spearmanr(hp.PN_hz_A_p.astype(float), hp.PN_hz_A_x.astype(float))
same(round(float(rho), 2), FD["retest_healthy_PN"][0], "healthy rerun rho of PN rate")
n_rt = len(D["RT"])
put("retest_healthy_PN_p", [r3(rho), pg(p), int(len(hp)), n_rt, pg(min(1.0, n_rt * p))],
    f"Projection neuron rate (PN_hz_A) in the {len(hp)} healthy pilot flies that were simulated again in the severity series with "
    f"the same seed (two runs of the intact network under different input noise): [Spearman rho between the two runs, its p "
    f"(t approximation, uncorrected), n pairs, number of readouts in the test-retest table, p after Bonferroni correction over "
    f"those {n_rt} readouts]. The rho is facts_detection retest_healthy_PN[0] (checked here).",
    rel(P_PILOT) + " merged with " + rel(P_SERIES) + " on seed, condition, stage, scope; rows condition = healthy; column PN_hz_A")

# ===================================================================== (g) own-group high-activity flies from s = 0.5
kmax, mmax = float(H.KC_hz_A.max()), float(H.MBON_hz_A.max())


def own_group(s):
    g = grp(s)
    return g[(g.KC_hz_A > 1.5 * g.KC_hz_A.median()) & (g.MBON_hz_A > 1.5 * g.MBON_hz_A.median())]


same([len(own_group(s)) for s in STAGES], FD["hiact_groupmedian_criterion_by_severity"], "own-group counts")
rows, n_flag = [], 0
for s in [x for x in STAGES if x >= 0.5]:
    fl = own_group(s)
    n_flag += len(fl)
    rows += [[float(s), int(x.seed), rnd(x.MBON_hz_A, 3), rnd(x.KC_hz_A, 3)] for _, x in fl.iterrows()
             if x.MBON_hz_A > mmax or x.KC_hz_A > kmax]
put("hiact_owngroup_from_s0.5_above_a_healthy_maximum", [n_flag, rows, rnd(mmax, 3), rnd(kmax, 3)],
    "Severity series, flies meeting the own-group high-activity criterion (Kenyon cell rate and MBON rate both above 1.5 times the "
    "median of the fly's own severity group) at s = 0.5, 0.6, 0.75 and 1.0: [number of such flies, those with a rate above a "
    "healthy maximum as [s, seed, MBON rate Hz, Kenyon cell rate Hz], healthy maximum MBON rate Hz, healthy maximum Kenyon cell "
    "rate Hz]. A listed fly is above the healthy maximum in the rate that exceeds it only.",
    rel(P_SERIES) + " columns KC_hz_A, MBON_hz_A by stage")

# ===================================================================== (h) values on a rounding edge
put("series_healthy_rcv_pct_LI_3dp", rnd(100 * D["rcv"](D["H"].LI), 3),
    "Healthy robust CV of the learning index (interquartile range / 1.349 / |median|, %, 64 healthy flies of the severity series) "
    "with three decimals. facts_detection series_healthy_rcv_pct_LI keeps two (32.85), which must not be rounded again: the value "
    "is 32.8% to one decimal.", rel(P_SERIES) + " column LI, stage 0")
g = grp(0.5)
r_ = B["spear"](g.PN_hz_A, g.MBON_hz_A)
same(r3(r_[0]), FB["within_s0.5_rho_PN_vs"]["MBON"], "rho PN vs MBON at s = 0.5")
put("series_s0.5_rho_PN_vs_MBON_5dp", [rnd(r_[0], 5), pg(r_[1]), r_[2]],
    "Severity series, the 32 flies at s = 0.5: Spearman rho of projection neuron rate with MBON rate with five decimals, its p and "
    "n. facts_battery within_s0.5_rho_PN_vs['MBON'] keeps three (0.525), which must not be rounded again: the value is 0.52 to two "
    "decimals.", rel(P_SERIES) + " columns PN_hz_A, MBON_hz_A, stage 0.5")

# ===================================================================== (i) maps: equal neurons against matched burden
e2["ref"] = e2["reference"].map(lambda x: "" if pd.isna(x) else f"{float(x):g}")
match = json.loads(P_MATCH.read_text())
s_match = match["maps"]["0.5"]["shuffled"]["s"]
sh_eq = vu[vu.condition == "shuffled_s0.5"]
sh_mt = e2[(e2["map"] == "shuffled") & (e2.ref == "0.5")]
assert len(sh_eq) == len(sh_mt) == 32 and np.allclose(sh_mt.stage, s_match) and np.allclose(sh_eq.stage, 0.5)
m_eq, m_mt = X["med"](sh_eq.brain_spikes_A), X["med"](sh_mt.brain_spikes_A)
same([m_eq, m_mt], [FE2["neu_s0.5_shuffled_brain_median_raw"], FE2["syn48.3_shuffled_brain_median_raw"]], "shuffled medians")
SRC_I = (rel(P_VULN) + " (condition shuffled_s0.5, hub_s0.5), " + rel(P_EXP2) + " (map shuffled and hub, reference 0.5), "
         + rel(P_SERIES) + " (stage 0.5), " + rel(P_COMP) + " (condition all_s0.5); column brain_spikes_A")
put("shuffled_s0.5_equal_neurons_vs_matched_burden_brain",
    [m_eq, m_mt, r3(m_mt / m_eq), r1(100 * (1 - m_mt / m_eq)), X["pv"](X["mw"](sh_eq.brain_spikes_A, sh_mt.brain_spikes_A)), 32, 32],
    f"Whole-brain spike count during odour A of the shuffled map in the two designs: [median with equal neurons removed (map "
    f"comparison, s = 0.5), median at matched burden (burden-matched comparison, 48.3% of contacts lost, s = {s_match}), ratio "
    "matched / equal-neuron, % by which the matched median is lower, two-sided Mann-Whitney p between the two groups of raw spike "
    "counts, n, n]. Because the hub condition (s = 0.5) is the same in both designs, the shuffled/hub ratio against any one hub "
    "group falls by this ratio from equal neurons to matched burden.", SRC_I)
hubs = {"mapcomp": vu[vu.condition == "hub_s0.5"], "series": grp(0.5), "ablation": co[co.condition == "all_s0.5"],
        "exp2": e2[(e2["map"] == "hub") & (e2.ref == "0.5")]}
assert all(len(g_) == 32 for g_ in hubs.values())
eq = {k: r3(m_eq / X["med"](g_.brain_spikes_A)) for k, g_ in hubs.items()}
mt = {k: r3(m_mt / X["med"](g_.brain_spikes_A)) for k, g_ in hubs.items()}
for k in hubs:
    same(eq[k], FE2[f"l4l_shuffled_s05_over_hub_{k}_brain"], f"equal-neuron ratio against the {k} hub group")
same(mt["exp2"], FE2["l4l_hubs05_eqsyn_shuffled_over_hub_brain"][0], "matched ratio within the burden-matched design")
put("shuffled_matched_over_hub_s0.5_groups_brain", mt,
    "Median whole-brain spike count of the shuffled flies at matched burden (48.3%) divided by the median of each of the four "
    "independent hub groups at s = 0.5 (32 flies each): map comparison, severity series, rule ablation ('all three rules') and the "
    "hub group of the burden-matched comparison. The ratio within the burden-matched design ('exp2') equals facts_exp2 "
    "l4l_hubs05_eqsyn_shuffled_over_hub_brain[0] (checked here).", SRC_I)
put("shuffled_equal_neurons_over_hub_s0.5_groups_brain", eq,
    "The same four ratios with the shuffled flies at equal neurons removed (s = 0.5) in the numerator; they equal facts_exp2 "
    "l4l_shuffled_s05_over_hub_*_brain (checked here).", SRC_I)
put("shuffled_over_hub_s0.5_groups_brain_ranges", [[min(eq.values()), max(eq.values())], [min(mt.values()), max(mt.values())]],
    "[lowest, highest] of the four ratios: [equal neurons removed, matched burden].", SRC_I)
pn = {"mapcomp": 100 * hubs["mapcomp"].PN_hz_A.median() / H.PN_hz_A.median(), "series": 100 * hubs["series"].PN_hz_A.median() / H.PN_hz_A.median(),
      "ablation": 100 * hubs["ablation"].PN_hz_A.median() / H.PN_hz_A.median(),
      "exp2": 100 * hubs["exp2"].PN_hz_A.median() / e2[e2["map"] == "healthy"].PN_hz_A.median()}
same([r1(pn["series"]), r1(pn["ablation"]), r1(pn["mapcomp"])], FB["rep_s0.5_PN_pct"], "PN rate of the three default-model groups")
same(r1(pn["exp2"]), FE2["syn48.3_hub_PN_pct"][0], "PN rate of the burden-matched hub group")
others = [pn[k] for k in ("series", "ablation", "exp2")]
put("hub_s0.5_groups_PN_pct", {k: r1(v) for k, v in pn.items()},
    "Median projection neuron rate (PN_hz_A) of each of the four hub groups at s = 0.5 as % of the median of its healthy "
    "reference (the 64 series healthy flies; for 'exp2' the 32 healthy flies of the burden-matched comparison). Equal to "
    "facts_battery rep_s0.5_PN_pct and facts_exp2 syn48.3_hub_PN_pct[0] (checked here).",
    SRC_I.replace("brain_spikes_A", "PN_hz_A"))
put("hub_s0.5_groups_PN_pct_other_than_mapcomp_range", [r1(min(others)), r1(max(others))],
    "[lowest, highest] of the three hub groups other than the map-comparison group (series, rule ablation, burden-matched), % "
    "of healthy.", SRC_I.replace("brain_spikes_A", "PN_hz_A"))
h2 = e2[e2["map"] == "healthy"]
g_hub, g_shu = (e2[(e2["map"] == m_) & (e2.ref == "0.25")].GF_loom_hz for m_ in ("hub", "shuffled"))
p_gf = X["mw"](g_hub, g_shu)
same(X["pv"](p_gf), FE2["syn27.1_GF_mw_p_hub_vs_shuffled"], "giant fibre hub vs shuffled at 27.1%")
put("syn27.1_GF_mw_p_hub_vs_shuffled_4dp", [round(p_gf, 4), int(len(g_hub)), int(len(g_shu))],
    "Burden-matched comparison at 27.1% of contacts lost: two-sided Mann-Whitney p of giant fibre rate, hub against shuffled map, "
    "with four decimals, and the two group sizes. facts_exp2 syn27.1_GF_mw_p_hub_vs_shuffled keeps three significant figures "
    "(0.485), which must not be rounded again: the value is 0.48 to two decimals.",
    rel(P_EXP2) + " rows reference = 0.25, map hub and shuffled; column GF_loom_hz")
kc_hub = 100 * X["med"](hubs["exp2"].KC_hz_A) / X["med"](h2.KC_hz_A)
same(r1(kc_hub), FE2["syn48.3_hub_KCrate_pct"][0], "Kenyon cell rate of the hub map at 48.3%")
put("syn48.3_hub_KCrate_pct_3dp", [round(kc_hub, 3), int(len(hubs["exp2"])), int(len(h2))],
    "Burden-matched comparison at 48.3% of contacts lost: median Kenyon cell rate of the hub map (s = 0.5) as % of the median of "
    "the healthy flies of that design, with three decimals, and the two group sizes. facts_exp2 syn48.3_hub_KCrate_pct keeps one "
    "(37.5), which must not be rounded again: the value is 37% to the nearest whole number.",
    rel(P_EXP2) + " rows map hub, reference 0.5 and map healthy; column KC_hz_A")
tg = [100 * match["maps"][k]["target_syn_lost_frac"] for k in ("0.1", "0.25", "0.5")]
same([round(v, 2) for v in tg], [FE2[f"syn{v:.1f}_target_contacts_lost_pct"] for v in tg], "burden targets")
put("matched_target_contacts_lost_pct_3dp", [round(v, 3) for v in tg],
    "Contacts lost by the hub map at s = 0.10, 0.25 and 0.5, the three burden levels to which the shuffled and antihub maps were "
    "matched, % of all contacts with three decimals (mean of the three unsimulated calibration flies; facts_exp2 "
    "syn*_target_contacts_lost_pct keeps two).", rel(P_MATCH) + " maps[0.1 | 0.25 | 0.5].target_syn_lost_frac")

# ===================================================================== (j) within-fly analysis
runs = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(E1 / "screen/*_runs.csv")))], ignore_index=True)
F = ax.frac_table(runs, SEL, MIN_DEF)
T = F[F.level == "type"]
dmg = runs[(runs.level == "baseline") & (runs.unit == "damaged")].set_index("seed")
assert len(dmg) == 48 and len(T) == 48 * 450
SRC_J = rel(E1 / "screen") + "/*_runs.csv"
DEF_J = ("Restored fraction = (readout with the unit restored - damaged) / (healthy - damaged) per fly, kept where |healthy - "
         "damaged| > 5% of |healthy| (analyze_exp1.frac_table, as in facts_exp1.json)")
G = T.dropna(subset=["f_GF_loom_hz"])
first = G.loc[G.groupby("seed").f_GF_loom_hz.idxmax()].unit.value_counts()
same([int(first["LPLC2"]), int(first["LC4"])], [dict(FE1["c.best_GF_type_identity"])[k] for k in ("LPLC2", "LC4")], "first-row best type")
best = {int(seed): set(g_.unit[g_.f_GF_loom_hz == g_.f_GF_loom_hz.max()]) for seed, g_ in G.groupby("seed")}
has = {k: [k in b for b in best.values()] for k in ("LPLC2", "LC4")}
tied = [len(b) > 1 for b in best.values()]
either = [a or b for a, b in zip(has["LPLC2"], has["LC4"])]
put("screen_best_GF_type_with_ties",
    {"n_flies_with_GF_deficit": len(best), "n_flies_with_ties": int(sum(tied)), "LPLC2_alone_or_tied": int(sum(has["LPLC2"])),
     "LC4_alone_or_tied": int(sum(has["LC4"])), "LPLC2_or_LC4_alone_or_tied": int(sum(either)),
     "both_tied": int(sum(a and b for a, b in zip(has["LPLC2"], has["LC4"]))),
     "LPLC2_alone": int(sum(a and not t for a, t in zip(has["LPLC2"], tied))),
     "LC4_alone": int(sum(a and not t for a, t in zip(has["LC4"], tied))),
     "n_flies_with_no_type_above_0": int(sum(G.groupby("seed").f_GF_loom_hz.max() <= 0))},
    f"Within-fly screen, the {len(best)} of 48 flies with a giant fibre deficit: the most effective tested type for giant fibre "
    "rate, counting every type whose restored fraction equals the fly's maximum over its 450 tested types (ties arise because "
    "giant fibre rate moves in steps of one spike). Flies with ties; flies in which LPLC2, LC4, either of them, or both are in "
    "the best set; flies in which each is the single best type; flies in which no tested type restored anything (the maximum is "
    f"0, so all its types tie; counted among the flies with ties). {DEF_J}. facts_exp1 c.best_GF_type_identity counts only the "
    "first type in row order (LPLC2 11, LC4 7; reproduced and checked here).", SRC_J + " level type, column GF_loom_hz")
CL = F[F.level == "class"]
dn = CL[CL.unit == "sc:descending_neuron"].set_index("seed").f_GF_loom_hz.dropna()
fail = [int(s) for s in dmg.index[dmg.GF_loom_hz < GF_FAIL_HZ]]
same(fail, FE1["c.escape_fail_seeds"], "screen flies below 100 Hz")
same(rnd(dn.median()), FE1["c.DN_class_GF_all"][0], "descending-neuron class, all flies")
dn_fail, dn_other = dn.reindex(fail), dn.drop(fail)
assert dn_fail.notna().all()
put("screen_DN_class_GF_by_group",
    [[rnd(dn_fail.median()), rnd(dn_fail.min()), rnd(dn_fail.max()), int(len(dn_fail))], [rnd(dn_other.median()), int(len(dn_other))]],
    f"Within-fly screen, fraction of the giant fibre deficit restored by restoring the descending-neuron class: [[median, min, "
    f"max, n] over the flies with giant fibre output below {GF_FAIL_HZ:g} Hz, [median, n] over the other flies with a giant fibre "
    f"deficit]. {DEF_J}. The median over all {len(dn)} flies with a deficit is facts_exp1 c.DN_class_GF_all (checked here).",
    SRC_J + " level class, unit sc:descending_neuron; baseline rows")
rd = runs[(runs.level == "baseline") & (runs.unit == "restore_dead_all")].set_index("seed")
n5 = rd[np.isclose(rd.stage, 0.5)].n_restored
same(float(n5.median()), FE1["k.removed_per_fly_s0.5"][0], "neurons removed at s = 0.5")
put("screen_s0.5_neurons_removed_range", [int(n5.min()), int(n5.max()), round(100 * (n5.max() - n5.min()) / n5.min(), 2), int(len(n5))],
    "Within-fly screen, neurons removed per fly in the 24 flies at s = 0.5 (neurons restored by the variant that restores all "
    "removed neurons): [min, max, (max - min) / min in %, n flies]. The median is facts_exp1 k.removed_per_fly_s0.5 (checked "
    "here).", SRC_J + " baseline row restore_dead_all, column n_restored")
sc_by_seed = sc.set_index("seed")
fired = [s for s in fail if dmg.GF_loom_hz[s] > 0]
ser = sc_by_seed.loc[fired]
assert np.allclose(ser.stage, dmg.stage[fired])
put("screen_GF_below100_still_firing_series_GF_hz",
    [int(len(fired)), fired, [r1(v) for v in ser.GF_loom_hz], r1(ser.GF_loom_hz.min()), r1(ser.GF_loom_hz.max()), int((ser.GF_loom_hz < GF_FAIL_HZ).sum())],
    f"The screen flies with giant fibre output below {GF_FAIL_HZ:g} Hz in the screen run that still fired (damaged giant fibre "
    "rate above 0): [n flies, their seeds, giant fibre rate (Hz) of the same fly in the severity-series run (same pathology, "
    f"other input noise and protocol), lowest, highest, number below {GF_FAIL_HZ:g} Hz in the series run].",
    SRC_J + " baseline row damaged; " + rel(P_SERIES) + " column GF_loom_hz by seed")
conf = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(E1 / "confirm/*.csv")))], ignore_index=True)
les = conf[conf.level == "lesion"]
assert (les.n_lesioned > 0).all() and (conf[conf.level != "lesion"].n_lesioned == 0).all()
per_type = les.groupby("unit").n_lesioned.median()
same([float(per_type.median()), len(per_type)], [FE1["i.lesion_n_neurons"][0], FE1["i.lesion_n_neurons"][3]], "lesion size per type")
put("confirm_lesion_n_neurons", [float(les.n_lesioned.median()), int(len(les)), float(per_type.median()), int(len(per_type))],
    "Full-protocol sufficiency arm (the fly's own lost neurons of a type removed from its healthy counterpart): [median number of "
    "neurons removed over all lesions, n lesions, median over types of the per-type median, n types]. The last two are "
    "facts_exp1 i.lesion_n_neurons (checked here).", rel(E1 / "confirm") + "/*.csv level lesion, column n_lesioned")
rob = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(E1 / "robust/*.csv")))], ignore_index=True)
MODEL = {"base": "base model", "w0.8": "weights x0.8", "w1.2": "weights x1.2", "glu0.5": "glutamatergic weights x0.5",
         "da0.5": "adaptation increment x0.5", "noise1": "input noise 1", "noise2": "input noise 2"}
assert set(rob.model) == set(MODEL) and rob.seed.nunique() == 12 and conf.seed.nunique() == 12
same(len(runs), FE1["design.n_screen_runs"], "screen simulations")
hl = runs[(runs.level == "baseline") & (runs.unit == "healthy")].set_index("seed")
screen_above = {(int(s), r) for r in SEL for s in dmg.index if dmg[r][s] - hl[r][s] > MIN_DEF * abs(hl[r][s])}
same(sorted(s for s, r in screen_above if r == "MBON_hz_A"), FE1["k.n_flies_damaged_above_healthy.MBON"][3], "screen MBON above healthy")
above = []
for model, g_ in list(rob.groupby("model")) + [("full protocol", conf)]:
    b_ = g_[g_.level == "baseline"].pivot_table(index="seed", columns="unit", values=SEL + (["LI"] if "LI" in g_ else []))
    for r in b_.columns.get_level_values(0).unique():
        for s in b_.index:
            d_, h_ = b_[(r, "damaged")][s], b_[(r, "healthy")][s]
            if d_ - h_ > MIN_DEF * abs(h_):
                above.append([int(s), r, MODEL.get(model, model), rnd(d_, 3), rnd(h_, 3),
                              model in ("base", "full protocol") and (int(s), r) in screen_above])
above.sort(key=lambda x: (x[0], x[1], x[2]))
put("robust_confirm_readouts_damaged_above_healthy", above,
    "Robustness runs (12 flies at s = 0.5 under seven models: the base model, four parameter variants and two further input-noise "
    "realisations) and full-protocol runs (the same 12 flies): every fly-readout pair kept with a negative denominator because the "
    "damaged value exceeds the healthy one by more than 5% of healthy: [seed, readout, model or 'full protocol', damaged value, "
    "healthy value, whether the pair repeats one already present in the screen (same fly, readout, model and input: facts_exp1 "
    "k.n_flies_damaged_above_healthy.*)] (rates in Hz; LI in learning index points; the six screen readouts in the robustness "
    "runs, and the learning index as well in the full protocol). A model that is not listed for a fly and readout had no such pair.",
    rel(E1 / "robust") + "/*.csv and " + rel(E1 / "confirm") + "/*.csv, level baseline")
bl = T.groupby("seed").f_LH_hz_A.max().dropna()
same(rnd(bl.median()), FE1["a.best_type_per_fly.LH.all"][0], "best type, lateral horn")
put("screen_best_type_per_fly_LH_median_5dp", [rnd(bl.median(), 5), int(len(bl))],
    "Within-fly screen: median over flies of the largest fraction of the lateral horn deficit restored by a single tested type, "
    "with five decimals, and n flies. facts_exp1 a.best_type_per_fly.LH.all keeps four (0.0385), which must not be rounded "
    f"again: the value is 3.8%. {DEF_J}.", SRC_J + " level type, column LH_hz_A")

# ===================================================================== (k) joint cohort
jd = J["add_normalised_li"](pd.read_csv(P_JOINT))
sj = json.loads(P_JSTATS.read_text())
reads = J["CIRCUIT"] + J["BEHAV"]
assert len(jd) == 176 and len(reads) == 16 and all(r in jd.columns for r in reads)
raw, tr = {}, {}
for r in reads:
    v = jd[["stage", r]].dropna()
    rho, p = st.spearmanr(v.stage, v[r])
    raw[r], tr[r] = float(p), [round(float(rho), 3), float(p), int(len(v))]
hol = J["holm"](raw)
for r in reads:
    t_ = sj["trend"][r]
    same([tr[r][0], tr[r][2]], [t_["spearman_rho"], t_["n"]], f"joint trend {r}")
    assert math.isclose(tr[r][1], t_["p"], rel_tol=1e-9) and math.isclose(hol[r], t_["p_holm"], rel_tol=1e-9), r
same([tr["behaviour_esc"][0], pg(tr["behaviour_esc"][1]), tr["behaviour_esc"][2]], FD["joint_jpt_vs_severity"], "jumps per threat trend")
SRC_K = rel(P_JOINT)
for r, nm, what in (("behaviour_feed_s", "feedTime", "time spent feeding in the arena"),
                    ("behaviour_esc", "jumpsPerThreat", "jumps per threat met in the arena (flies that met a threat)")):
    put(f"joint_trend_{nm}", [tr[r][0], pg(tr[r][1]), tr[r][2], pg(hol[r])],
        f"Joint cohort (176 flies: 32 healthy, 24 at each of six severities), {what}: [Spearman rho with severity, p (t "
        f"approximation), n flies with a defined value, Holm-corrected p over the {len(reads)} readouts tested in the joint cohort "
        "(the twelve circuit readouts of the series and the four arena readouts; analyze_joint.py)]. Recomputed from the per-fly "
        "table and equal to results/joint/stats_joint.json trend (checked here for all sixteen readouts).",
        SRC_K + f" columns stage, {r}")
jst = sorted(s for s in jd.stage.unique() if s > 0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")                      # a constant column (every fly exactly 1.0) gives an undefined rho
    by_gf = [B["spear"](jd[jd.stage == s].GF_loom_hz, jd[jd.stage == s].behaviour_esc) for s in [0.0] + jst]
    by_mn9 = [B["spear"](jd[jd.stage == s].MN9_sugar_hz, jd[jd.stage == s].behaviour_feed_s) for s in [0.0] + jst]
same([round(x[0], 2) + 0.0 for x in by_mn9], FD["joint_withinfly_MN9_vs_feedTime_by_severity"], "MN9 vs feeding time by severity")
same(r3(by_gf[0][0]), FD["joint_withinfly_GF_vs_jumpsPerThreat_healthy"][0], "GF vs jumps per threat, healthy")
put("joint_withinfly_GF_vs_jumpsPerThreat_by_severity",
    [[float(s), None if not np.isfinite(x[0]) else round(x[0], 2) + 0.0, x[2]] for s, x in zip([0.0] + jst, by_gf)],
    "Joint cohort: within-severity Spearman rho of giant fibre rate (circuit protocol) with jumps per threat met (arena) at s = 0 "
    "and 0.05 to 0.5, flies that met a threat: [s, rho, n]. null = undefined because every fly of the group scored exactly 1 jump "
    "per threat. The counterpart for MN9 drive and feeding time is facts_detection joint_withinfly_MN9_vs_feedTime_by_severity "
    "(reproduced and checked here).", SRC_K + " columns GF_loom_hz, behaviour_esc by stage")
cnt = {r: [int(jd[jd.stage == s][r].notna().sum()) for s in [0.0] + jst] for r in ("behaviour_PI", "behaviour_esc")}
same(cnt["behaviour_esc"], [x[1] for x in FD["joint_jpt_decomposition_by_severity"]], "flies that met a threat")
put("joint_n_defined_by_severity", {"severities": [0.0] + [float(s) for s in jst], "preference_index": cnt["behaviour_PI"],
                                    "jumps_per_threat": cnt["behaviour_esc"]},
    "Joint cohort: flies with a defined arena preference index (flies that entered an odour plume) and with a defined jumps per "
    "threat (flies that met a threat) at s = 0 (32 healthy flies) and at each severity (24 flies each).",
    SRC_K + " columns behaviour_PI, behaviour_esc by stage")

# ===================================================================== (l) battery designs
h_pos = set(zip(*divmod(np.flatnonzero((sc0.stage == 0).to_numpy()), B["BATTERY_BATCH"])))
assert len(sc0) % B["BATTERY_BATCH"] == 0 and len(h_pos) == 64
pos = {}
for nm, d_ in (("rule_ablation", co), ("map_comparison", vu), ("kenyon_cell_only", sp)):
    for c in sorted(d_.condition.unique()):
        full, part = B["_slots"](d_, c)
        assert len(full) + len(part) == 32
        pos.setdefault(nm, {})[c] = [len(full & h_pos) + len(part & h_pos), len(part & h_pos)]
a_f, _ = B["_slots"](sc0, "s0.5")
same(len(a_f & B["_slots"](co, "all_s0.5")[0]) + len(a_f & B["_slots"](vu, "hub_s0.5")[0]) + len(B["_slots"](co, "all_s0.5")[0] & B["_slots"](vu, "hub_s0.5")[0]),
     sum(FB["rep_shared_stimulus_slots_s0.5"][:3]), "shared positions among the default-model groups")
tot = [v[0] for d_ in pos.values() for v in d_.values()]
put("battery_flies_in_position_of_a_healthy_series_fly", pos,
    f"Flies of each 32-fly group of the rule ablation, the map comparison and the Kenyon-cell-only scenario that sat in the same "
    f"position of the same batch (batches of {B['BATTERY_BATCH']} in run order) as one of the 64 healthy flies of the severity "
    "series, and so received the same input trains as that healthy fly (all four designs used stimulus seed 7): per condition "
    "[n flies, of which in a last batch shorter than a full batch, where only the first odour presentation is identical]. Rows "
    "of each table are in run order; positions as _slots of facts_battery.py.",
    ", ".join(rel(p) for p in (P_COMP, P_VULN, P_SCOPE, P_SERIES)) + " row order and column condition")
put("battery_flies_in_position_of_a_healthy_series_fly_range",
    {"all_groups": [min(tot), max(tot)], **{nm: [min(v[0] for v in d_.values()), max(v[0] for v in d_.values())] for nm, d_ in pos.items()},
     "total": int(sum(tot)), "total_in_a_short_last_batch": int(sum(v[1] for d_ in pos.values() for v in d_.values()))},
    "[lowest, highest] number per 32-fly group over all eighteen groups and within each design, the total over the groups and the "
    "number of them in a short last batch.", "battery_flies_in_position_of_a_healthy_series_fly")
kwp = {}
for s in (0.5, 1.0):
    for k in B["REP_READS"]:
        c = B["R"][k]
        p = B["kw"](grp(s)[c], co[co.condition == f"all_{B['sk'](s)}"][c], vu[vu.condition == f"hub_{B['sk'](s)}"][c])
        same(p, FB[f"rep_{B['sk'](s)}_{k}_kw_p"], f"replicate Kruskal-Wallis {k} s = {s:g}")
        kwp[(k, s)] = p
low = sorted((p, k, s) for (k, s), p in kwp.items() if p < 0.05)
put("rep_22tests_n_below_005", [int(len(low)), int(len(kwp)), [[k, s, p] for p, k, s in low]],
    f"Kruskal-Wallis tests across the three default-model replicate groups (severity series, rule ablation 'all three rules', map "
    f"comparison hub; 32 flies each) for {len(B['REP_READS'])} readouts at s = 0.5 and 1.0: [number with p < 0.05 (uncorrected), "
    "number of tests, those below 0.05 as [readout, s, p]]. Each p equals facts_battery rep_s*_*_kw_p (checked here); "
    "facts_battery rep_14tests_n_below_005 counts the 14 tests of the seven readouts shown in S2 Fig.",
    ", ".join(rel(p) for p in (P_SERIES, P_COMP, P_VULN)))

# ===================================================================== (m) design counts
sw = pd.read_csv(P_INT)
sets = sw.groupby(["glom", "side"]).n_orn.agg(["min", "max", "size"]).reset_index().sort_values("min")
assert (sets["min"] == sets["max"]).all() and len(sw) == 24 and not sw.duplicated(["glom", "side", "rate", "da"]).any()
rates, das = sorted(float(v) for v in sw.rate.unique()), sorted(float(v) for v in sw.da.unique())
assert len(sw) == len(sets) * len(rates) * len(das)
same([int(sw.n_orn.min()), int(sw.n_orn.max())], facts("model")["intensity_sweep_n_orn_range"], "ORNs in the intensity sweep")
same([int(sw.PN_active.min()), int(sw.PN_active.max())], facts("model")["intensity_sweep_pn_active_range"], "active PNs in the intensity sweep")
put("intensity_sweep_design", {"n_conditions": int(len(sw)), "orn_sets": [[x.glom, x.side, int(x["min"])] for _, x in sets.iterrows()],
                               "rates_hz": rates, "adaptation_increment_mv": das},
    "Exploratory odour intensity sweep (ORN drive, recurrent antennal lobe; its script was not kept, so the protocol is inferred "
    "from the result file, as facts_model says of intensity_sweep_*): number of conditions, the stimulated ORN sets as "
    "[glomeruli, side(s), number of ORNs], the input rates (Hz) and the adaptation increments (mV; 0 = none). Every set was run at "
    "every rate with each increment. The ranges of ORNs and of active projection neurons equal facts_model "
    "intensity_sweep_n_orn_range and intensity_sweep_pn_active_range (checked here).", rel(P_INT) + " columns glom, side, n_orn, rate, da")
n_sim = {"screen": int(len(runs)), "robustness": int(len(rob)), "full_protocol": int(len(conf))}
n_sim["total"] = sum(n_sim.values())
rb = rob.groupby("model").size()
assert (rb == rb.iloc[0]).all() and rb.iloc[0] * len(MODEL) == len(rob)
put("within_fly_n_simulations", n_sim,
    f"Simulations of the within-fly analysis = rows of the result tables: screen (48 flies; facts_exp1 design.n_screen_runs, "
    f"checked here), robustness test ({len(MODEL)} models x {int(rb.iloc[0])} rows: the single-type restorations and the damaged "
    f"and healthy baselines of 12 flies), full protocol (necessity and sufficiency runs and baselines of the same 12 flies; "
    "facts_exp1 i.confirm_rows), and their sum.",
    SRC_J + ", " + rel(E1 / "robust") + "/*.csv, " + rel(E1 / "confirm") + "/*.csv (row counts)")

# ===================================================================== (n) figures: drawn values against the text
f4 = S_FIG4.read_text(encoding="utf-8")
h_de = float(must(r"HAB, HC, HDE = [\d.]+, [\d.]+, ([\d.]+)", f4, "axes heights of Fig 4").group(1))
blk = f4[f4.index("# D: neurons and Kenyon cells removed at matched levels"):f4.index("level_axis(ax)", f4.index("# D: neurons and Kenyon cells removed"))]
must(r"axes_in\(fig, X0 \+ k \* \(WD \+ GAP\), TDE, WD, HDE\)", blk, "Fig 4D axes height")
ms_d = float(must(r'ax\.plot\(xs, ys, "-o", ms=([\d.]+),', blk, "Fig 4D marker size").group(1))
y0, y1 = (float(v) for v in must(r"ax\.set_ylim\(([\d.]+), ([\d.]+)\)", blk, "Fig 4D y limits").groups())
iq = {}
for (m_, r_), g_ in e2[e2["map"] != "healthy"].groupby(["map", "ref"]):
    assert len(g_) == 32
    iq[(m_, r_)] = [100 * float(g_[c].quantile(.75) - g_[c].quantile(.25)) for c in ("dead_frac", "kc_dead_frac")]
pt_per_pp = h_de * 72 / (y1 - y0)
big = max(v[1] for v in iq.values())
put("fig4D_largest_iqr_in_symbol_diameters",
    [round(max(v[0] for v in iq.values()), 2), round(big, 2), round(big * pt_per_pp / ms_d, 2), h_de, [y0, y1], ms_d, int(len(iq))],
    "Fig 4D (neurons and Kenyon cells removed at matched burden; medians drawn without a spread): [largest interquartile range "
    "of neurons removed per fly over the groups drawn, percentage points; largest interquartile range of Kenyon cells removed; "
    "the latter in symbol diameters on the printed axis; axes height in inches; y-axis limits in %; symbol diameter in points; "
    "number of groups drawn (32 flies each)]. One percentage point is axes height x 72 / (y range) points "
    "(make_fig4_burden.py, panel D).", rel(P_EXP2) + " columns dead_frac, kc_dead_frac by map and reference; " + rel(S_FIG4) + " (HDE, set_ylim, ms)")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import make_figures as mf  # noqa: E402  (draws nothing to disk here: save is replaced below)

fig_src = S_FIG.read_text(encoding="utf-8")
assert 'ALL = {"fig1": fig1, "fig2": fig2, "fig3": fig3' in fig_src and "RNG = np.random.default_rng(7)" in fig_src, "code changed: figure order"
mf.save = lambda fig, name: plt.close(fig)               # the figures are drawn in memory only
drawn, _cis = [], mf._cis


def _spy(xs, groups, ref, r, offs=(-0.14, 0.14)):
    out = _cis(xs, groups, ref, r, offs)
    drawn.append((r, out))
    return out


mf._cis = _spy
mf.fig1()
mf.fig2()
assert not drawn
mf.fig3()                                                # rows A, B, C; five readouts each; groups; s = 0.5 then 1.0
assert len(drawn) == 15 and drawn[11][0] == "MBON_hz_A"
gi, _, e_, lo_, hi_, fill = drawn[11][1][0]              # row C (maps), MBON rate, hub map, s = 0.5 (filled symbol)
assert gi == 0 and fill
txt = FB["map_s0.5_hub_MBON"]
same(r1(e_), txt[0], "hub-map MBON estimate in Fig 3C")
put("fig3C_hub_s0.5_MBON_drawn_vs_text", [[r1(e_), r1(lo_), r1(hi_)], txt[:3], r1(hi_ - txt[2]), r1(lo_ - txt[1])],
    "Fig 3C, hub map at s = 0.5, MBON rate as % of healthy: [[estimate, lo, hi] as drawn by make_figures.py, [estimate, lo, "
    "hi] quoted in the text (facts_battery map_s0.5_hub_MBON), drawn upper bound minus text upper bound, drawn lower bound minus "
    f"text lower bound] in percentage points. The figure takes its {mf.NBOOT:,} resamples from one generator (numpy "
    "default_rng(7)) shared by all figures in the order Fig 1, Fig 2, Fig 3; the same calls are repeated here in memory, without "
    "writing a figure. Both intervals are 95% percentile bootstraps of the same ratio of medians and differ by Monte Carlo error.",
    rel(S_FIG) + " fig1(), fig2(), fig3() (_cis, row C); " + rel(P_VULN) + " condition hub_s0.5, column MBON_hz_A")
hubm = vu[vu.condition == "hub_s0.5"].MBON_hz_A
b4 = boot_pct(hubm, H.MBON_hz_A, "map0.5hubMBON")
same([r1(b4[0]), r1(b4[1])], txt[1:3], "hub-map MBON interval of facts_battery")
alt = np.array([b4] + [boot_pct(hubm, H.MBON_hz_A, f"map0.5hubMBON#alt{i}") for i in range(B["BOOT_ALT_SEEDS"])])
b40 = boot_pct(hubm, H.MBON_hz_A, "map0.5hubMBON#big", 10 * N_BOOT)
put("map_s0.5_hub_MBON_ci_seed_range",
    [[r1(alt[:, 0].min()), r1(alt[:, 0].max())], [r1(alt[:, 1].min()), r1(alt[:, 1].max())], len(alt), [r1(b40[0]), r1(b40[1])]],
    f"The text interval of the same value (facts_battery map_s0.5_hub_MBON, label 'map0.5hubMBON', reproduced here) under the "
    f"reported seed and {B['BOOT_ALT_SEEDS']} further seeds of {N_BOOT:,} resamples (labels '#alt0' to '#alt{B['BOOT_ALT_SEEDS'] - 1}'), "
    f"the stability check that facts_battery.py applies when a bound moves at {10 * N_BOOT:,} resamples: [[lowest, highest "
    f"lower bound], [lowest, highest upper bound], number of seeds, [lo, hi] at {10 * N_BOOT:,} resamples (label '#big')]. "
    "Shows how far the upper bound moves between seeds, for comparison with the bound drawn in Fig 3C.",
    rel(P_VULN) + " condition hub_s0.5 and " + rel(P_SERIES) + " stage 0; column MBON_hz_A")

# ===================================================================== anchor of the severity axis
kc = sc0[sc0.scope == "pan"].groupby("stage").kc_dead_frac.mean() * 100
datum = (132.1 - 110.3) / 132.1 * 100
s_anchor = 0.5 + 0.1 * (datum - kc.loc[0.5]) / (kc.loc[0.6] - kc.loc[0.5])
put("anchor_severity_interpolated", [round(float(datum), 3), round(float(kc.loc[0.5]), 3), round(float(kc.loc[0.6]), 3),
                                     round(float(s_anchor), 4)],
    "Severity at which the model's mean Kenyon cell loss equals the anchoring datum: [Kenyon cell loss of the datum in % "
    "(132.1 against 110.3 labelled mushroom body neurons per section, Ping et al. 2015, flies expressing amyloid-beta42 in the "
    "mushroom body), mean % of Kenyon cells removed in the severity series at s = 0.5, the same at s = 0.6, s by linear "
    "interpolation between the two]. Methods quotes s of about 0.56; figures shade severities from 0.55 upward.",
    rel(P_SERIES) + " column kc_dead_frac (scope pan), stages 0.5 and 0.6")

# ===================================================================== write
FACTS["_meaning"] = MEANING
FACTS["_source"] = SOURCE
missing = [k for k in FACTS if not k.startswith("_") and (k not in MEANING or k not in SOURCE)]
assert not missing, missing
(OUT / "facts_text.json").write_text(json.dumps(FACTS, indent=1, ensure_ascii=False,
                                                default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                     encoding="utf-8")
print(f"wrote {rel(OUT / 'facts_text.json')}: {len(MEANING)} facts")
