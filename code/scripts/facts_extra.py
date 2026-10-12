"""Facts table "extra": four numbers quoted in the text that the other facts tables do not hold.

  PYTHONIOENCODING=utf-8 python code/scripts/facts_extra.py

(a) Split of lost contacts in the burden-matched comparison (results/exp2/battery.csv): hub map at
    reference s = 0.5 (32 flies), contacts removed by thinning (thin_frac) and contacts lost through the
    connections of removed neurons (syn_lost_frac - thin_frac), as defined in exp2_burden_matched.py burden().
(b) Chance level of the direction-free AUC max(AUC, 1 - AUC) for 32 vs 64 flies (severity series) and
    24 vs 32 flies (joint cohort): mean and 95th percentile under the null of identical distributions,
    by simulation (20,000 pairs of standard normal samples, numpy default_rng(SEED)) and, as a check,
    exactly from the null distribution of the Mann-Whitney U statistic without ties.
(c) Within-fly screen (results/exp1/screen/*_units.csv, *_runs.csv): tested cell types whose lost neurons
    are highly connected (mean hub vulnerability rank >= 0.996), how many changed no readout and how many
    restored less than 5% of every deficit. Restored fraction and 5% deficit rule as in exp1_stats.py
    and results/facts/facts_exp1.json (analyze_exp1.frac_table).
(d) Odour persistence in the MaleCNS weight calibration (results/calibration/calibration.json): range of
    neurons active per 100-ms bin after stimulus offset at w_syn = 0.157 and 0.275 mV.

Conventions as in the other facts_*.py scripts: one flat dict of named facts, plus "_meaning" (what
each key is) and "_source" (file and field or column per key). Medians are of per-fly values. No
simulation of the model is run and no existing file is changed. Deterministic: the only random numbers
are the AUC null draws, from a fixed seed.

Writes results/facts/facts_extra.json.
"""
import glob
import json
import re
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())  # code/scripts locally, scripts/ on the server
sys.path.insert(0, str(ROOT / "code/scripts"))
import analyze_exp1 as ax  # noqa: E402  (frac_table: restored fraction with the 5% deficit rule)

OUT = ROOT / "results/facts"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261012
N_DRAWS = 20_000
P_EXP2 = ROOT / "results/exp2/battery.csv"
S_EXP2 = ROOT / "code/scripts/exp2_burden_matched.py"
SCR = ROOT / "results/exp1/screen"
P_CAL = ROOT / "results/calibration/calibration.json"
S_CAL = ROOT / "code/scripts/calibrate_wsyn.py"
SEL = ["brain_spikes_A", "LH_hz_A", "KC_hz_A", "MBON_hz_A", "MN9_sugar_hz", "GF_loom_hz"]
MIN_DEF = 0.05                       # exp1_stats.py MIN_DEF / facts_exp1.json
HUB_RANK = 0.996                     # rank of the second giant fibre (DNp01) in the hub map

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


# ===================================================================== (a) contacts lost, hub s = 0.5
src2 = S_EXP2.read_text(encoding="utf-8")
must(r'"syn_lost_frac": float\(lost / total\.sum\(\)\)', src2, "syn_lost_frac definition")
must(r'"thin_frac": float\(\(total - kept\)\.sum\(\) / total\.sum\(\)\)', src2, "thin_frac definition")
must(r"lost = \(total - kept\)\.sum\(\) \+ kept\[gone_edge\]\.sum\(\)", src2, "lost = thinned + kept contacts of removed neurons")
must(r"use = ~cx\.al_edge", src2, "antennal-lobe edges excluded")
bat = pd.read_csv(P_EXP2)
g = bat[(bat["map"] == "hub") & np.isclose(bat["reference"], 0.5)]
assert len(g) == 32 and np.allclose(g.stage, 0.5) and (g.condition == "hub_ref0.5").all(), g.condition.unique()
thin = 100 * g.thin_frac.to_numpy(float)
total = 100 * g.syn_lost_frac.to_numpy(float)
extra = total - thin
assert (extra >= 0).all()
DEF_A = ("percent of all synaptic contacts of the model (antennal-lobe connections excluded; exp2_burden_matched.py "
         "burden(), recomputed from each fly's seed)")
SRC_A = rel(P_EXP2) + " rows map=hub, reference=0.5 (condition hub_ref0.5)"
put("burden_hub_s0.5_n_flies", int(len(g)),
    "Flies of the burden-matched comparison with the hub map at reference s = 0.5 (stage 0.5, 48.3% of contacts lost "
    "by design).", SRC_A)
put("burden_hub_s0.5_thinning_pct", rnd(np.median(thin), 2),
    f"Median over the 32 hub flies at s = 0.5 of the contacts removed by synapse thinning (thin_frac), {DEF_A}.",
    SRC_A + " column thin_frac")
put("burden_hub_s0.5_removed_neuron_connections_pct", rnd(np.median(extra), 2),
    "Median over the same flies of the additional contacts lost through the connections of removed neurons: the "
    "contacts that survived thinning on every connection whose presynaptic or postsynaptic neuron was removed "
    f"(syn_lost_frac - thin_frac, per fly), {DEF_A}.", SRC_A + " columns syn_lost_frac - thin_frac")
put("burden_hub_s0.5_total_pct", rnd(np.median(total), 2),
    f"Median over the same flies of all contacts lost (syn_lost_frac = thinning + connections of removed neurons), {DEF_A}.",
    SRC_A + " column syn_lost_frac")
put("burden_hub_s0.5_sum_of_two_medians_pct", rnd(np.median(thin) + np.median(extra), 2),
    "Sum of burden_hub_s0.5_thinning_pct and burden_hub_s0.5_removed_neuron_connections_pct (medians are not "
    "additive; compare burden_hub_s0.5_total_pct).", SRC_A)
put("burden_hub_s0.5_ranges_pct", [rnd([thin.min(), thin.max()], 2), rnd([extra.min(), extra.max()], 2),
                                   rnd([total.min(), total.max()], 2)],
    "[min, max] over the 32 flies of [thinning, connections of removed neurons, total], percent of all contacts.", SRC_A)
put("burden_hub_s0.5_flies_removed_connections_ge_thinning", [int((extra >= thin).sum()), int(len(g))],
    "Flies in which the contacts lost through the connections of removed neurons (after thinning) were at least those "
    "removed by thinning [n, of n flies]. A neuron-loss-only fly loses these connections without thinning, so its "
    "loss is at least this large.", SRC_A)

# ===================================================================== (b) chance level of max(AUC, 1 - AUC)


def auc_null_sim(n1, n2):
    """Direction-free AUC of two independent N(0, 1) samples, N_DRAWS times (no ties)."""
    rng = np.random.default_rng([SEED, n1, n2])
    z = rng.standard_normal((N_DRAWS, n1 + n2))
    ranks = z.argsort(axis=1).argsort(axis=1) + 1
    u = ranks[:, :n1].sum(axis=1) - n1 * (n1 + 1) / 2
    auc = u / (n1 * n2)
    return np.maximum(auc, 1 - auc)


def u_null_counts(m, n):
    """Number of orderings giving each Mann-Whitney U = 0..m*n (no ties): Gaussian binomial [m+n, m]_q."""
    p = [1]
    for i in range(1, m + 1):
        a = n + i
        q = p + [0] * a                     # multiply by (1 - q^a)
        for k, c in enumerate(p):
            q[k + a] -= c
        for k in range(i, len(q)):          # divide by (1 - q^i), exact
            q[k] += q[k - i]
        assert all(c == 0 for c in q[n * i + 1:])
        p = q[: n * i + 1]
    assert sum(p) == comb(m + n, m) and all(c >= 0 for c in p)
    return p


def auc_null_exact(m, n):
    c = u_null_counts(m, n)
    tot, mn = sum(c), m * n
    d = {}
    for k, ck in enumerate(c):
        key = max(k, mn - k)
        d[key] = d.get(key, 0) + ck
    mean = sum(k * ck for k, ck in d.items()) / (tot * mn)
    cum = 0
    for k in sorted(d):
        cum += d[k]
        if 100 * cum >= 95 * tot:
            return mean, k / mn


for (n1, n2), lab, design in [((32, 64), "n32_vs_64", "severity series (32 flies per severity against 64 healthy)"),
                              ((24, 32), "n24_vs_32", "joint cohort (24 flies per severity against 32 healthy)")]:
    dfa = auc_null_sim(n1, n2)
    put(f"auc_chance_{lab}_sim", [rnd(dfa.mean()), rnd(np.percentile(dfa, 95)), rnd(dfa.std(ddof=1) / np.sqrt(N_DRAWS), 5)],
        f"Chance level of the direction-free AUC max(AUC, 1 - AUC) for {n1} against {n2} flies, the {design}: [mean, "
        f"95th percentile (numpy linear interpolation), Monte Carlo standard error of the mean] over {N_DRAWS:,} pairs of "
        f"independent standard normal samples (identical distributions, no ties), numpy default_rng([{SEED}, {n1}, {n2}]). "
        f"AUC = Mann-Whitney U / (n1 n2) from ranks.", "computed here (simulation)")
    em, ep = auc_null_exact(n1, n2)
    put(f"auc_chance_{lab}_exact", [rnd(em), rnd(ep)],
        f"Same quantity for {n1} against {n2} flies computed exactly from the null distribution of U without ties "
        f"(Gaussian binomial coefficients, all {comb(n1 + n2, n1):.3e} orderings equally likely): [mean, 95th percentile "
        f"= smallest value v with P(max(AUC, 1 - AUC) <= v) >= 0.95]. Check of the simulation.", "computed here (exact)")

# ===================================================================== (c) highly connected types in the screen
runs = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(SCR / "*_runs.csv")))], ignore_index=True)
units = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(SCR / "*_units.csv")))], ignore_index=True)
F = ax.frac_table(runs, SEL, MIN_DEF)
T = F[F.level == "type"]
UT = units[units.level == "type"]
assert len(T) == len(UT) == 48 * 450 and not T.duplicated(["seed", "unit"]).any()
T = T.merge(UT[["seed", "unit", "mean_rank", "n_lost"]], on=["seed", "unit"], how="inner", validate="1:1")
assert len(T) == len(UT) and T.mean_rank.notna().all()
dmg = runs[(runs.level == "baseline") & (runs.unit == "damaged")].set_index("seed")[SEL]
assert dmg.index.is_unique and len(dmg) == 48
assert T[SEL].notna().all().all() and dmg.notna().all().all()
T["unchanged"] = (T[SEL].to_numpy() == dmg.loc[T.seed, SEL].to_numpy()).all(axis=1)
fc = [f"f_{r}" for r in SEL]
T["fmax"] = T[fc].max(axis=1)                     # NaN readouts (deficit <= 5% of healthy) are skipped
T["fabsmax"] = T[fc].abs().max(axis=1)
T["fzero"] = (T[fc].fillna(0) == 0).all(axis=1)
assert T.fmax.notna().all()
G = T.groupby("unit").agg(rank_mean=("mean_rank", "mean"), rank_max=("mean_rank", "max"), n_flies=("seed", "nunique"),
                          unchanged=("unchanged", "all"), fzero=("fzero", "all"), fmax=("fmax", "max"),
                          fabsmax=("fabsmax", "max"))
SRC_C = rel(SCR) + "/*_units.csv (level=type: mean_rank) and *_runs.csv (level=type and baseline)"
DEF_C = ("mean_rank = mean hub vulnerability rank u of the type's neurons removed in that fly (default hub map; all screen "
         "flies use it)")
put("screen_types_tested_distinct", int(len(G)),
    f"Distinct cell types tested singly in the within-fly screen (48 flies x 450 types; {len(T)} type rescues).", SRC_C)
for lab, sel, how in [("mean", G.rank_mean >= HUB_RANK, "averaged over the flies in which the type was tested"),
                      ("anyfly", G.rank_max >= HUB_RANK, "reached in at least one fly in which the type was tested")]:
    h = G[sel]
    put(f"screen_hub_types_rank_{lab}_ge_0.996", [int(len(h)), int(h.unchanged.sum()), int((h.fmax < 0.05).sum())],
        f"Tested types whose removed neurons are highly connected, mean_rank >= {HUB_RANK} {how} ({DEF_C}): [n types, n "
        f"that changed none of the six readouts (WB, LH, KC, MBON, MN9, GF) in any fly, i.e. every raw readout "
        f"bit-identical to the damaged fly in every tested fly, n whose maximum restored fraction over all tested flies "
        f"and readouts was below 0.05, i.e. restored less than 5% of every kept deficit]. Restored fraction = "
        f"(y_type - y_damaged) / (y_healthy - y_damaged) per fly; a readout is left out for a fly when |healthy - "
        f"damaged| <= 5% of |healthy| (either sign kept), as in exp1_stats.py and facts_exp1.json. The unchanged "
        f"types are a subset of those below 0.05 ({int((h.unchanged & (h.fmax < 0.05)).sum())} of "
        f"{int(h.unchanged.sum())}).", SRC_C)
    put(f"screen_hub_types_rank_{lab}_ge_0.996_variants",
        [int(h.fzero.sum()), int((h.fabsmax < 0.05).sum()), int((h.n_flies == 1).sum()), rnd(h.n_flies.median(), 1)],
        f"Variants for the same {len(h)} types (rank criterion: {how}): [n whose restored fraction was exactly 0 for every "
        f"kept fly-readout pair (a type may still change a readout without a kept deficit), n with |restored fraction| < "
        f"0.05 for every kept pair (negative values bounded too), n tested in only one fly, median flies tested].", SRC_C)
hm = G[G.rank_mean >= HUB_RANK].sort_values("rank_mean", ascending=False)
put("screen_hub_types_rank_mean_ge_0.996_list",
    [[t, int(x.n_flies), rnd(x.rank_mean), bool(x.unchanged), rnd(x.fmax, 3)] for t, x in hm.iterrows()],
    "The types of screen_hub_types_rank_mean_ge_0.996: [type, n flies tested, mean_rank averaged over those flies, "
    "changed no readout in any fly, maximum restored fraction over flies and kept readouts], by decreasing rank.", SRC_C)

# ===================================================================== (d) odour persistence, MaleCNS calibration
csrc = S_CAL.read_text(encoding="utf-8")
must(r"on = br\.run\(3000, si, poisson_events\(rng, np\.full\(si\.shape, 150\.0\), 3000, 0\.1\), bin_steps=1000\)",
     csrc, "300 ms on at 0.1 ms steps, 100-ms bins")
must(r"off = br\.run\(7000, si, np\.zeros\(\(B, 7000, stim\.size\), bool\), bin_steps=1000\)", csrc, "700 ms off")
must(r'res\[f"malecns_odor_w\{w:\.3f\}_active_per_100ms"\] = \[int\(x\) for x in \(r > 0\)\.sum\(1\)\]', csrc,
     "active = replicate-mean rate > 0")
cal = json.loads(P_CAL.read_text())
N_ON = 3
for w in ("0.157", "0.275"):
    b = cal[f"malecns_odor_w{w}_active_per_100ms"]
    assert len(b) == 10
    post = b[N_ON:]
    put(f"calib_malecns_odour_w{w}_active_post_offset_range", [int(min(post)), int(max(post))],
        f"MaleCNS base model (no adaptation, recurrent antennal lobe), w_syn = {w} mV, left DM1/DM4/VA2 ORNs at 150 Hz "
        f"for 300 ms (4 replicate brains): [min, max] neurons active per 100-ms bin after stimulus offset, over bins 3 to "
        f"9 of the 1-s run (300-1000 ms from onset = 0-700 ms after offset; 7 bins). Active = replicate-mean rate > 0 in "
        f"the bin.", rel(P_CAL) + f" malecns_odor_w{w}_active_per_100ms[3:]")
    put(f"calib_malecns_odour_w{w}_active_per_100ms_post_offset", [int(v) for v in post],
        f"The 7 post-offset bins of the w_syn = {w} mV odour run, in order (0-100 ms ... 600-700 ms after offset).",
        rel(P_CAL) + f" malecns_odor_w{w}_active_per_100ms[3:]")
put("calib_malecns_odour_bins_used", [N_ON, 9],
    "Bin indices (0-based, 100-ms bins of the 1-s calibration run) used for the post-offset ranges: bins 0-2 are the "
    "300-ms stimulus, bins 3-9 cover 0-700 ms after offset (calibrate_wsyn.py persistence(): 3000 + 7000 steps of "
    "0.1 ms, bin_steps=1000).", rel(S_CAL) + " persistence()")

# ===================================================================== write
FACTS["_meaning"] = MEANING
FACTS["_source"] = SOURCE
missing = [k for k in FACTS if not k.startswith("_") and (k not in MEANING or k not in SOURCE)]
assert not missing, missing
(OUT / "facts_extra.json").write_text(json.dumps(FACTS, indent=1, ensure_ascii=False,
                                                 default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                      encoding="utf-8")
print(f"wrote {rel(OUT / 'facts_extra.json')}: {len(MEANING)} facts")
