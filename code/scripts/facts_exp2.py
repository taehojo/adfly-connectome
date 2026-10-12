"""Facts table of experiment 2 (vulnerability maps at matched synaptic burden).

  PYTHONIOENCODING=utf-8 python code/scripts/facts_exp2.py

Recomputes, from the per-fly result tables, the exp2 numbers that the text may need
and writes results/facts/facts_exp2.json: a flat dict of named facts (numbers or short
lists) plus "_meaning", one sentence per key. No simulation is run and no existing
file is changed.

Inputs
  results/exp2/battery.csv       288 flies: 32 healthy + 8 map x level groups of 32
  results/exp2/calibration.csv   burden against severity, 3 pathology draws per map and s
  results/exp2/matched.json      matched severities;  results/exp2/summary.json  prespecified verdicts
  results/battery/stage_curve_x4.csv   severity series; stage 0 = the 64 series healthy flies
  results/battery/vulnerability_x4.csv equal-neuron map comparison (s = 0.5 and 1.0)
  results/battery/components_x4.csv    rule ablation (threshold-shift ablation, replicate group)
  results/joint/cohort_joint.csv       joint cohort (seed overlap; third healthy group, different protocol)
  results/stats/exp2_stats.json, results/stats/battery_stats.json  (cross-checks only)

Conventions
  * Medians as % of the healthy median of the same design: exp2 groups against the 32
    exp2 healthy flies; equal-neuron (map comparison) groups and the other battery
    designs against the 64 series healthy flies (stage 0 of stage_curve_x4.csv).
  * 95% intervals: percentile bootstrap, 4,000 resamples of flies within each group,
    groups (and the healthy group) resampled independently, numpy default_rng(SEED).
    Values given as [estimate, lo, hi].
  * Deficit = 100 - % of healthy. Deficit ratio map/hub = (healthy - map)/(healthy - hub)
    on medians; reported only when the hub deficit is at least 5% of healthy.
  * Ratio of highest to lowest map median: point = ratio of the observed medians; the
    "maxmin" interval is the analyze_exp2.py statistic (per resample, largest map median
    over smallest; never below 1, so it tests the 1.27 criterion, not "no difference");
    the "pair" interval keeps the two maps fixed (identified by the observed medians).
    Tied medians are reported as "hub=shuffled"; the pair is then oriented over hub by
    map order, and "placement" requires hub to be the only lowest map.
  * Ratios: a zero observed denominator median gives "inf" (point not quotable), 0/0
    gives null; resamples with a zero denominator count as +-inf, 0/0 resamples are
    dropped, and each such case is described in the key's meaning.
  * Monte Carlo check: the whole file is recomputed with 20 alternative bootstrap seeds;
    bounds whose second significant figure is not reproducible are listed in
    _mc_sensitive_keys and their ranges are added to the meanings. The prespecified
    whole-brain verdicts are rechecked over 300 seeds. Run time about one minute.
  * Mann-Whitney U two-sided; Kruskal-Wallis across the maps present; Fisher exact
    two-sided for counts; Spearman via scipy.
  * Levels are named by contacts lost: syn11.5 (hub s = 0.1), syn27.1 (hub s = 0.25),
    syn48.3 (hub s = 0.5; antihub unreachable). Equal-neuron design: neu_s0.5, neu_s1.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())  # code/scripts locally, scripts/ on the server
OUT = ROOT / "results/facts"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261010
RNG = np.random.default_rng(SEED)
NB = 4000
REPLICATE_SPREAD = 1.27          # prespecified whole-brain criterion (analyze_exp2.py)
DTH_MAX = 2.0                    # mV threshold shift at full load (adfly/pathology.py)
MIN_DEF = 5.0                    # % of healthy; smaller hub deficits give no deficit ratio
MAPS = ["hub", "shuffled", "antihub"]
READ = {"brain": "brain_spikes_A", "LH": "LH_hz_A", "PN": "PN_hz_A", "KCrate": "KC_hz_A",
        "KCsurv": "KC_surv_hz", "KCrecruit": "KC_frac_A", "MBON": "MBON_hz_A", "LI": "LI",
        "MN9": "MN9_sugar_hz", "GF": "GF_loom_hz"}
RDESC = {"brain": "whole-brain spike count during odour A (brain_spikes_A)",
         "LH": "mean lateral horn neuron rate during odour A (LH_hz_A)",
         "PN": "mean ALPN-class rate during odour A (PN_hz_A: all antennal-lobe projection neurons, the "
               "odour-A-driven DM1/DM4/VA2 PNs and the unstimulated PNs together)",
         "KCrate": "mean Kenyon cell rate during odour A over all Kenyon cells, removed ones included (KC_hz_A)",
         "KCsurv": "Kenyon cell rate per surviving Kenyon cell, KC_hz_A / (1 - kc_dead_frac)",
         "KCrecruit": "fraction of Kenyon cells firing at least one spike to odour A (KC_frac_A)",
         "MBON": "mean MBON rate during odour A (MBON_hz_A)",
         "LI": "learning index (LI)",
         "MN9": "MN9 feeding drive to sugar (MN9_sugar_hz)",
         "GF": "giant fibre rate to looming (GF_loom_hz)"}
PN_CAVEAT = ("PN_hz_A averages the whole ALPN class (assay_battery.py), but odour A is injected only into the "
             "DM1/DM4/VA2 PNs (cx.upn), so this criterion mixes stimulated and unstimulated PNs and is not a direct "
             "measure of the stimulated input delivered.")
LOAD_POP = ("The per-fly mean (mean_load = f.load.mean()) is over ALL neurons of the fly, including sensory neurons "
            "(ineligible for pathology, load 0) and removed neurons (counted with their load although their "
            "threshold is set to infinity).")
JOINT_PROTOCOL = ("The joint cohort (cohort_joint.py) is a third healthy group run under a DIFFERENT protocol, not a "
                  "replicate of the battery healthy condition: flies built and trained through HouseNB (same "
                  "build_brains), a no-refractory set that adds the right-side LPLC2/LC4 looming neurons, and, after "
                  "the naive assays and before training, br.reset_state() and a naive-valence probe (odours A and B "
                  "at three concentrations), with training by HouseNB.train() (same pairing and learn() rule, "
                  "separate random stream). The learning index is the most affected; whole-brain spikes come from "
                  "the first odour A segment, before the reset, probe and training.")

MC_SEEDS = [SEED + i for i in range(1, 21)]   # alternative bootstrap seeds for the Monte Carlo check
MC_REL_BOUND, MC_MIN_STEPS = 0.05, 2           # flag a bound whose range over seeds exceeds 5% of the bound
                                               # and 2 steps of the reported rounding (0.1 for %, 0.001 for ratios)
MC_VERDICT_SEEDS = 300                         # seeds for the stability of the prespecified verdicts

F, M = {}, {}
BOOT_KEYS = set()        # keys whose value is a bootstrap [estimate, lo, hi]


class Tri(list):
    """[estimate, lo, hi] plus a note on infinite, undefined or dropped resamples; put() appends
    the note to the key's meaning."""
    note = ""


def put(key, val, meaning):
    if key in F:
        raise KeyError(f"duplicate key {key}")
    if isinstance(val, Tri):
        BOOT_KEYS.add(key)
        if val.note:
            meaning += " " + val.note
        val = list(val)
    F[key] = val
    M[key] = meaning


def r1(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 1)


def r3(x):
    if x is None:
        return None
    if np.isinf(x):
        return "inf" if x > 0 else "-inf"
    return None if np.isnan(x) else round(float(x), 3)


def pv(p):
    return None if p is None or not np.isfinite(p) else float(f"{p:.3g}")


def zero_text(x, name):
    """'the observed <name> median is 0 (k of n flies at 0)'."""
    x = pd.Series(x).dropna()
    return f"the observed {name} median is 0 ({int((x == 0).sum())} of {len(x)} flies at 0)"


def tri(est, b, nd=1, zero_den=None):
    """[estimate, lo, hi] from a bootstrap array, as a Tri carrying a note.

    +-inf resamples (zero denominator) are kept and ordered at the ends; 0/0 resamples are
    dropped and counted. zero_den names the denominator when the point estimate is not finite."""
    b = np.asarray(b, float)
    n_nan = int(np.isnan(b).sum())
    b = b[~np.isnan(b)]
    f = r1 if nd == 1 else r3

    def fmt(x):
        return ("inf" if x > 0 else "-inf") if np.isinf(x) else f(x)

    notes = []
    if est is not None and np.isnan(est):
        notes.append(f"The point estimate is undefined (0/0, reported as null): {zero_den or 'both observed medians are 0'}.")
    elif est is not None and np.isinf(est):
        notes.append(f"The point estimate is infinite because {zero_den or 'the observed denominator median is 0'}; "
                     f"it is not a usable ratio and should not be quoted.")
    if len(b) == 0:
        out = Tri([None if est is None or np.isnan(est) else fmt(est), None, None])
        notes.append("Every resample was 0/0, so there is no interval.")
        out.note = " ".join(notes)
        return out
    lo, hi = np.percentile(np.clip(b, -1e300, 1e300), [2.5, 97.5])
    lo = np.inf if lo >= 1e299 else -np.inf if lo <= -1e299 else lo
    hi = np.inf if hi >= 1e299 else -np.inf if hi <= -1e299 else hi
    n_pos, n_neg = int(np.isposinf(b).sum()), int(np.isneginf(b).sum())
    if n_pos or n_neg:
        sign = " and ".join(t for t in (f"{n_pos} as +inf" if n_pos else "", f"{n_neg} as -inf" if n_neg else "") if t)
        notes.append(f"{n_pos + n_neg} of the {NB} resamples have a denominator of 0 with a non-zero numerator and "
                     f"enter the percentiles as infinite ({sign})"
                     + ("; this is why a bound is infinite." if np.isinf(lo) or np.isinf(hi) else "."))
    if n_nan:
        notes.append(f"{n_nan} of the {NB} resamples were 0/0 (numerator and denominator both 0) and were dropped "
                     f"before taking percentiles, so the interval describes the remaining {NB - n_nan}.")
    out = Tri([None if est is None or np.isnan(est) else fmt(est), fmt(lo), fmt(hi)])
    out.note = " ".join(notes)
    return out


def ratio_value(num, den):
    """num/den with +-inf for a zero denominator and nan for 0/0."""
    if den != 0:
        return num / den
    return np.nan if num == 0 else np.inf if num > 0 else -np.inf


def spread(meds):
    """Max/min of group medians: None if no median is positive (0/0) or one is negative;
    inf if only the smallest is 0."""
    lo, hi = min(meds), max(meds)
    if hi <= 0 or lo < 0:
        return None
    return hi / lo if lo > 0 else np.inf


def spread_note(meds, groups, col):
    """Meaning text for a spread that is not a finite ratio."""
    sil = ", ".join(f"{n} {int((g[col] == 0).sum())}/{len(g)}" for n, g in groups.items())
    if max(meds) == 0:
        return (f" All medians are 0 (flies at exactly 0 per group: {sil}), so there is no spread to report: "
                f"the ratio is 0/0 and is given as null.")
    if min(meds) < 0:
        return " A median is negative, so max/min is not a meaningful spread and is given as null."
    if min(meds) == 0:
        return (f" The smallest median is 0 (flies at exactly 0 per group: {sil}), so the point ratio is infinite; "
                f"do not quote it.")
    return ""


# --------------------------------------------------------------------------- data
def prep(d):
    d = d.copy()
    d["KC_surv_hz"] = d["KC_hz_A"] / (1.0 - d["kc_dead_frac"])
    return d


E2 = prep(pd.read_csv(ROOT / "results/exp2/battery.csv"))
E2["ref"] = E2["reference"].map(lambda x: "" if pd.isna(x) else f"{float(x):g}")
H2 = E2[E2["map"] == "healthy"]
SC = prep(pd.read_csv(ROOT / "results/battery/stage_curve_x4.csv"))
SC = SC[SC.scope == "pan"]
HS = SC[SC.stage == 0]
VU = prep(pd.read_csv(ROOT / "results/battery/vulnerability_x4.csv"))
CO = prep(pd.read_csv(ROOT / "results/battery/components_x4.csv"))
JO = prep(pd.read_csv(ROOT / "results/joint/cohort_joint.csv"))
JH = JO[JO.stage == 0]
CALRAW = pd.read_csv(ROOT / "results/exp2/calibration.csv")
CAL = CALRAW.groupby(["map", "s"]).mean(numeric_only=True).reset_index()
MATCH = json.loads((ROOT / "results/exp2/matched.json").read_text())
SUMM = json.loads((ROOT / "results/exp2/summary.json").read_text())
E2STATS = json.loads((ROOT / "results/stats/exp2_stats.json").read_text())
BSTATS = json.loads((ROOT / "results/stats/battery_stats.json").read_text())

REFS = list(MATCH["maps"].keys())                        # "0.1", "0.25", "0.5"
TGT = {r: 100 * MATCH["maps"][r]["target_syn_lost_frac"] for r in REFS}
LV = {r: f"syn{TGT[r]:.1f}" for r in REFS}               # syn11.5, syn27.1, syn48.3


def grp(ref, m):
    return E2[(E2.ref == ref) & (E2["map"] == m)]


def present(ref):
    return [m for m in MAPS if len(grp(ref, m))]


def cal_burden(m, s):
    return 100 * float(CAL[(CAL["map"] == m) & np.isclose(CAL["s"], s)].syn_lost_frac.iloc[0])


# bootstrap medians, cached per (group id, column) so that every statistic of a group
# uses the same resamples
_BC = {}


def bmed(gid, x):
    k = gid
    if k not in _BC:
        x = np.asarray(pd.Series(x).dropna(), float)
        _BC[k] = np.median(RNG.choice(x, (NB, len(x))), 1)
    return _BC[k]


def med(x):
    return float(np.median(pd.Series(x).dropna()))


def pct_tri(gid, x, hid, h):
    est = 100 * med(x) / med(h)
    b = 100 * bmed((gid, x.name), x) / bmed((hid, h.name), h)
    return tri(est, b)


def ratio_pair(gid_a, a, gid_b, b, den="denominator"):
    """Ratio of medians a/b, [point, 95% CI]; den names group b in the notes."""
    est = ratio_value(med(a), med(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        bb = bmed((gid_a, a.name), a) / bmed((gid_b, b.name), b)
    return tri(est, bb, 3, zero_text(b, den))


def deficit_ratio(gid_a, a, gid_ref, ref, hid, h):
    hm = med(h)
    if 100 * (hm - med(ref)) / hm < MIN_DEF:
        return Tri([None, None, None])
    est = (hm - med(a)) / (hm - med(ref))
    hb = bmed((hid, h.name), h)
    den = hb - bmed((gid_ref, ref.name), ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        bb = (hb - bmed((gid_a, a.name), a)) / den
    out = tri(est, bb, 3, "the reference deficit is 0")
    n_neg = int((den < 0).sum())
    if n_neg:
        out.note = (out.note + f" In {n_neg} of the {NB} resamples the reference (hub) median exceeds the healthy "
                    f"median, so the denominator is negative there.").strip()
    return out


def maxmin(groups):
    """groups: {map: (gid, series)} -> (maxmin CI, pair CI, info).

    info: hi, lo = maps used as highest / lowest (on a tie the earlier map in MAPS order counts
    as lowest and the later as highest, so a tie involving hub is oriented over hub); top, bot =
    all maps sharing the highest / lowest median; meds = the medians."""
    meds = {m: med(x) for m, (g, x) in groups.items()}
    order = sorted(meds, key=meds.get)            # stable sort
    hi_m, lo_m = order[-1], order[0]
    top = [m for m in meds if meds[m] == meds[hi_m]]
    bot = [m for m in meds if meds[m] == meds[lo_m]]
    est = ratio_value(meds[hi_m], meds[lo_m])
    zd = zero_text(groups[lo_m][1], f"{lo_m} (lowest map)")
    B = np.column_stack([bmed((g, x.name), x) for g, x in groups.values()])
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = B.max(1) / B.min(1)
        pb = bmed((groups[hi_m][0], groups[hi_m][1].name), groups[hi_m][1]) / \
            bmed((groups[lo_m][0], groups[lo_m][1].name), groups[lo_m][1])
    return tri(est, rb, 3, zd), tri(est, pb, 3, zd), dict(hi=hi_m, lo=lo_m, top=top, bot=bot, meds=meds)


def verdict(mm, info, crit):
    """Prespecified rule (analyze_exp2.py) with criterion crit: burden if the max/min ratio is at most
    crit; placement if the maxmin lower bound exceeds crit and hub is the only lowest map; else
    unresolved. Undefined when the ratio or the criterion is not a finite number."""
    ratio, lo = mm[0], mm[1]
    if crit is None or not np.isfinite(crit) or ratio is None or isinstance(ratio, str):
        return "undefined"
    if ratio <= crit:
        return "burden"
    if lo is not None and (lo == "inf" or (not isinstance(lo, str) and lo > crit)) and info["bot"] == ["hub"]:
        return "placement"
    return "unresolved"


def rank_labels(info):
    """[highest, lowest]; tied maps are joined with '='."""
    return ["=".join(info["top"]), "=".join(info["bot"])]


def tie_text(info, href=None, href_name=""):
    """'' or a sentence describing tied medians (href: healthy median for a % of healthy)."""
    def val(m):
        v = info["meds"][m]
        return f"{v:.4g}" + (f", {100 * v / href:.1f}% of {href_name}" if href else "")
    if len(info["top"]) == 1 and len(info["bot"]) == 1:
        return ""
    if info["top"] == info["bot"]:
        return (f" Tie: the {' and '.join(info['top'])} medians are identical ({val(info['top'][0])}), so no map is "
                f"highest or lowest; the '='-joined names mark the tie.")
    parts = [f"{' and '.join(g)} ({val(g[0])})" for g in (info["top"], info["bot"]) if len(g) > 1]
    return " Tie: identical medians for " + "; ".join(parts) + "; '='-joined names mark tied maps."


def maxmin_caveat(maps, pair_key, over_hub_key):
    """Meaning text: the folded max/min interval cannot go below 1."""
    txt = (" Max/min is at least 1 in every resample, so the lower bound is at least 1 by construction: this "
           "interval tests the prespecified 1.27 criterion, not 'no difference' between maps.")
    if len(maps) == 2:
        return txt + f" With only two maps ({', '.join(maps)}), quote {over_hub_key} for a two-sided comparison."
    return txt + f" For a two-sided comparison quote {pair_key} or the over-hub ratios."


def pair_text(info, ratio_key):
    """Meaning text for the pair interval when the ordering comes from a tie."""
    if len(info["top"]) == 1 and len(info["bot"]) == 1:
        return ""
    txt = (f" The medians are tied, so {info['hi']} (numerator) and {info['lo']} (denominator) are fixed by the "
           f"map order hub, shuffled, antihub, not by an observed ordering")
    return txt + (f"; this interval equals {ratio_key}." if ratio_key else ".")


def mw(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 3 or len(b) < 3 or (a.nunique() == 1 and b.nunique() == 1 and a.iloc[0] == b.iloc[0]):
        return None
    return float(st.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def kw(series_list):
    s = [pd.Series(x).dropna() for x in series_list]
    allv = pd.concat(s)
    if allv.nunique() <= 1:
        return None
    return float(st.kruskal(*s).pvalue)


def auc(x, y):
    """P(x > y) + 0.5 P(x = y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(((x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y)))


def fisher(a, na, b, nb):
    return float(st.fisher_exact([[a, na - a], [b, nb - b]])[1])


def holm(p):
    o = sorted(p, key=lambda k: p[k])
    m, prev, out = len(o), 0.0, {}
    for i, k in enumerate(o):
        prev = min(1.0, max(prev, (m - i) * p[k]))
        out[k] = prev
    return out


def build(seed):
    """Compute every fact with bootstrap seed `seed`; fills the module-level F and M."""
    global RNG
    RNG = np.random.default_rng(seed)
    _BC.clear()
    F.clear()
    M.clear()
    BOOT_KEYS.clear()

    # =========================================================================== design
    put("exp2_n_flies", int(len(E2)), "Number of flies in the burden-matched experiment (results/exp2/battery.csv).")
    put("exp2_n_healthy", int(len(H2)), "Number of exp2 healthy flies (map == healthy, s = 0), the reference for all exp2 percentages.")
    put("exp2_n_per_group", int(E2[E2["map"] != "healthy"].groupby(["ref", "map"]).size().min()),
        "Smallest number of flies in any exp2 map x level group (all groups have this n).")
    put("series_n_healthy", int(len(HS)), "Number of series healthy flies (stage 0 of stage_curve_x4.csv, scope pan), the reference for equal-neuron percentages.")
    put("exp2_seed_range", [int(E2.seed.min()), int(E2.seed.max())], "Smallest and largest pathology seed of the 288 exp2 flies.")
    for r in REFS:
        L = LV[r]
        put(f"{L}_target_contacts_lost_pct", round(TGT[r], 2),
            f"Target fraction of functional synaptic contacts lost (%), the calibration mean of the hub map at s = {r} (matched.json), defining level {L}.")
        for m in MAPS:
            s = MATCH["maps"][r][m]["s"]
            put(f"{L}_{m}_severity_s", s,
                f"Severity s at which the {m} map removes the target contacts of level {L} (matched.json; null = unreachable).")
    dev = max(abs(100 * grp(r, m).syn_lost_frac.median() - TGT[r]) for r in REFS for m in present(r))
    put("exp2_max_abs_dev_contacts_lost_pp", round(dev, 3),
        "Largest absolute difference (percentage points) between a group's median measured contacts lost and its level target, over the 8 exp2 map x level groups.")
    for m in MAPS:
        put(f"max_reachable_contacts_lost_pct_{m}", round(100 * MATCH["maps"]["0.1"][m]["max_syn_lost_frac"], 1),
            f"Largest fraction of contacts lost (%) the {m} map reaches within s <= 1 (calibration mean of 3 draws at s = 1; matched.json).")

    # =========================================================================== per level, map
    for r in REFS:
        L = LV[r]
        here = present(r)
        for m in here:
            g = grp(r, m)
            gid = ("e2", r, m)
            gd = f"the 32 exp2 flies of the {m} map at level {L} (s = {MATCH['maps'][r][m]['s']})"
            put(f"{L}_{m}_contacts_lost_pct", r1(100 * g.syn_lost_frac.median()),
                f"Median per-fly fraction of functional synaptic contacts lost (%) in {gd}.")
            put(f"{L}_{m}_neurons_removed_pct", round(100 * g.dead_frac.median(), 2),
                f"Median per-fly fraction of all neurons removed (%) in {gd}.")
            put(f"{L}_{m}_KC_removed_pct", round(100 * g.kc_dead_frac.median(), 2),
                f"Median per-fly fraction of Kenyon cells removed (%) in {gd}.")
            put(f"{L}_{m}_mean_load", round(float(g.mean_load.median()), 4),
                f"Median per-fly mean pathology load in {gd}. {LOAD_POP} Each neuron's firing threshold is lowered by its load x {DTH_MAX} mV.")
            put(f"{L}_{m}_mean_threshold_shift_mV", round(float(g.mean_load.median()) * DTH_MAX, 4),
                f"Median per-fly mean threshold DECREASE (mV) = mean load x dth_max ({DTH_MAX} mV) in {gd}: pathology lowers the firing threshold (v_th - load x dth_max, adfly/pathology.py). {LOAD_POP}")
            for k, c in READ.items():
                x = g[c]
                put(f"{L}_{m}_{k}_pct", pct_tri(gid, x, ("e2h",), H2[c]),
                    f"Median {RDESC[k]} as % of the exp2 healthy median, [estimate, 95% bootstrap lo, hi], in {gd}.")
                put(f"{L}_{m}_{k}_median_raw", round(med(x), 4),
                    f"Median {RDESC[k]} in raw units (spikes, Hz, fraction or LI points) in {gd}.")
                put(f"{L}_{m}_{k}_deficit_pct", r1(100 - 100 * med(x) / med(H2[c])),
                    f"Deficit of {RDESC[k]} = 100 - median as % of the exp2 healthy median, in {gd}.")
            put(f"{L}_{m}_KCsurv_vs_healthy_mw_p", pv(mw(g.KC_surv_hz, H2.KC_surv_hz)),
                f"Two-sided Mann-Whitney p, per-surviving-Kenyon-cell rate of {gd} vs the 32 exp2 healthy flies.")
            put(f"{L}_{m}_n_GF_below100Hz", int((g.GF_loom_hz < 100).sum()),
                f"Number of flies (of 32) with giant fibre rate below 100 Hz (escape-failure criterion) in {gd}.")
            put(f"{L}_{m}_n_MN9_silent", int((g.MN9_sugar_hz == 0).sum()),
                f"Number of flies (of 32) with MN9 feeding drive exactly 0 Hz in {gd}.")
            put(f"{L}_{m}_frac_PN_below95pct", round(float((g.PN_hz_A < 0.95 * med(H2.PN_hz_A)).mean()), 3),
                f"Fraction of flies whose PN rate (whole ALPN class, PN_hz_A) is below 95% of the exp2 healthy median, in {gd}.")
            for k in ("brain", "LH", "MBON", "LI"):
                c = READ[k]
                put(f"{L}_{m}_{k}_pct_vs_series_healthy", r1(100 * med(g[c]) / med(HS[c])),
                    f"Median {RDESC[k]} of {gd} as % of the SERIES healthy median (64 flies), for comparison with the figures of the battery designs.")
        # tests across maps
        for k, c in READ.items():
            put(f"{L}_{k}_kruskal_p", pv(kw([grp(r, m)[c] for m in here])),
                f"Kruskal-Wallis p for {RDESC[k]} across the maps present at level {L} ({', '.join(here)}; 32 flies each).")
            for i, a in enumerate(here):
                for b in here[i + 1:]:
                    put(f"{L}_{k}_mw_p_{a}_vs_{b}", pv(mw(grp(r, a)[c], grp(r, b)[c])),
                        f"Two-sided Mann-Whitney p for {RDESC[k]}, {a} vs {b} map, level {L} (32 vs 32 flies).")
            groups = {m: (("e2", r, m), grp(r, m)[c]) for m in here}
            mm, pr, info = maxmin(groups)
            tie = tie_text(info, med(H2[c]), "exp2 healthy")
            ohk = f"{L}_{info['hi']}_over_hub_{k}_ratio" if info["lo"] == "hub" else None
            put(f"{L}_{k}_ratio_high_low_maxmin", mm,
                f"Ratio of the highest to the lowest map median of {RDESC[k]} at level {L}, [point, 95% CI of the per-resample max/min over maps (analyze_exp2.py statistic)]."
                + maxmin_caveat(here, f"{L}_{k}_ratio_high_low_pair", f"{L}_shuffled_over_hub_{k}_ratio") + tie)
            put(f"{L}_{k}_ratio_high_low_pair", pr,
                (f"Ratio of the highest to the lowest map median of {RDESC[k]} at level {L}, [point, 95% CI with the two maps fixed at the observed highest ({info['hi']}) and lowest ({info['lo']})]."
                 if not tie else
                 f"Ratio of the {info['hi']} to the {info['lo']} map median of {RDESC[k]} at level {L}, [point, 95% CI with the two maps fixed].")
                + tie + pair_text(info, ohk))
            put(f"{L}_{k}_highest_lowest_map", rank_labels(info),
                f"Maps with the highest and the lowest median {RDESC[k]} at level {L}." + tie)
            for m in here[1:]:
                put(f"{L}_{m}_over_hub_{k}_ratio", ratio_pair(("e2", r, m), grp(r, m)[c], ("e2", r, "hub"), grp(r, "hub")[c], f"hub ({L})"),
                    f"Ratio of the {m} median to the hub median of {RDESC[k]} at level {L}, [point, 95% bootstrap CI].")
                put(f"{L}_{m}_over_hub_{k}_deficit_ratio",
                    deficit_ratio(("e2", r, m), grp(r, m)[c], ("e2", r, "hub"), grp(r, "hub")[c], ("e2h",), H2[c]),
                    f"Deficit ratio {m}/hub for {RDESC[k]} at level {L}: (healthy - {m})/(healthy - hub) on medians, exp2 healthy, [point, 95% CI]; null if the hub deficit is below {MIN_DEF}% of healthy.")
        for m in here[1:]:
            for col, nm in (("dead_frac", "neurons_removed"), ("kc_dead_frac", "KC_removed"), ("mean_load", "mean_load")):
                put(f"{L}_{m}_over_hub_{nm}_ratio", r3(grp(r, m)[col].median() / grp(r, "hub")[col].median()),
                    f"Ratio of the {m} to the hub median per-fly {nm.replace('_', ' ')} at level {L}.")
            put(f"{L}_{m}_vs_hub_brain_auc", round(auc(grp(r, m).brain_spikes_A, grp(r, "hub").brain_spikes_A), 3),
                f"Probability that a {m} fly has more whole-brain spikes than a hub fly at level {L} (Mann-Whitney AUC, ties 0.5).")
            for cnt, crit, nm in (("GF_loom_hz", lambda x: x < 100, "GF_below100Hz"),
                                  ("MN9_sugar_hz", lambda x: x == 0, "MN9_silent")):
                a, b = int(crit(grp(r, m)[cnt]).sum()), int(crit(grp(r, "hub")[cnt]).sum())
                put(f"{L}_{m}_vs_hub_{nm}_fisher_p", pv(fisher(a, 32, b, 32)),
                    f"Two-sided Fisher exact p comparing the number of flies with {nm.replace('_', ' ')} in the {m} vs hub map at level {L} (of 32 each).")
        for col, nm in (("dead_frac", "neurons_removed"), ("kc_dead_frac", "KC_removed")):
            put(f"{L}_{nm}_kruskal_p", pv(kw([grp(r, m)[col] for m in here])),
                f"Kruskal-Wallis p for per-fly {nm.replace('_', ' ')} across the maps at level {L}.")
        # PN-conditioned comparison (critic U3): flies whose PN rate is at least 95% of healthy
        thr = 0.95 * med(H2.PN_hz_A)
        for m in here:
            g = grp(r, m)
            gk = g[g.PN_hz_A >= thr]
            put(f"{L}_{m}_n_PN_ge95pct", int(len(gk)),
                f"Number of {m} flies at level {L} whose PN rate is at least 95% of the exp2 healthy median. {PN_CAVEAT}")
            put(f"{L}_{m}_brain_pct_among_PN_ge95pct", r1(100 * med(gk.brain_spikes_A) / med(H2.brain_spikes_A)) if len(gk) else None,
                f"Median whole-brain spikes as % of exp2 healthy among {m} flies at level {L} with PN rate >= 95% of healthy (PN-rate-preserved subset). {PN_CAVEAT}")
            put(f"{L}_{m}_LH_pct_among_PN_ge95pct", r1(100 * med(gk.LH_hz_A) / med(H2.LH_hz_A)) if len(gk) else None,
                f"Median lateral horn rate as % of exp2 healthy among {m} flies at level {L} with PN rate >= 95% of healthy (PN-rate-preserved subset). {PN_CAVEAT}")
        for m in here[1:]:
            a = grp(r, m)
            a = a[a.PN_hz_A >= thr]
            h = grp(r, "hub")
            h = h[h.PN_hz_A >= thr]
            put(f"{L}_{m}_vs_hub_brain_mw_p_among_PN_ge95pct", pv(mw(a.brain_spikes_A, h.brain_spikes_A)),
                f"Two-sided Mann-Whitney p for whole-brain spikes, {m} vs hub at level {L}, restricted to flies with PN rate >= 95% of exp2 healthy. {PN_CAVEAT}")
        pooled = pd.concat([grp(r, m) for m in here])
        for k in ("brain", "LH", "KCrate", "MBON"):
            rho, p = st.spearmanr(pooled.PN_hz_A, pooled[READ[k]])
            put(f"{L}_spearman_PN_vs_{k}", [r3(rho), pv(p)],
                f"Spearman [rho, p] between the {RDESC['PN']} and {RDESC[k]} over all {len(pooled)} map flies at level {L} (maps pooled).")

    # healthy baselines for the counts and PN criterion
    put("exp2_healthy_n_GF_below100Hz", int((H2.GF_loom_hz < 100).sum()), "Number of the 32 exp2 healthy flies with giant fibre rate below 100 Hz.")
    put("exp2_healthy_n_MN9_silent", int((H2.MN9_sugar_hz == 0).sum()), "Number of the 32 exp2 healthy flies with MN9 feeding drive 0 Hz.")
    put("exp2_healthy_frac_PN_below95pct", round(float((H2.PN_hz_A < 0.95 * med(H2.PN_hz_A)).mean()), 3),
        "Fraction of the 32 exp2 healthy flies whose PN rate is below 95% of their own median (baseline for the PN criterion).")
    put("series_healthy_frac_PN_below95pct", round(float((HS.PN_hz_A < 0.95 * med(HS.PN_hz_A)).mean()), 3),
        "Fraction of the 64 series healthy flies whose PN rate is below 95% of their own median.")

    # =========================================================================== prespecified criterion
    for r in REFS:
        L = LV[r]
        p = SUMM["references"][r].get("primary")
        if p:
            put(f"{L}_primary_ratio_summaryjson", [p["ratio"]] + p["ratio_ci"],
                f"Prespecified whole-brain ratio highest/lowest map median at level {L} with its 95% CI, copied from results/exp2/summary.json (analyze_exp2.py, rng seed 2).")
            put(f"{L}_primary_verdict_summaryjson", p["verdict"],
                f"Prespecified verdict at level {L} copied from results/exp2/summary.json: burden if ratio <= 1.27; placement if CI lower bound > 1.27 and hub lowest; else unresolved.")
            put(f"{L}_primary_highest_lowest_summaryjson", [p["highest"], p["lowest"]],
                f"Maps with the highest and lowest whole-brain median at level {L} per summary.json.")

    # post hoc: each readout against its own replicate spread (three default groups at s = 0.5)
    def rep_groups(s):
        return {"series": SC[np.isclose(SC.stage, s)], "ablation_all": CO[CO.condition == f"all_s{s:g}"],
                "maps_hub": VU[VU.condition == f"hub_s{s:g}"]}


    SPREAD = {}
    for s in (0.5, 1.0):
        gs = rep_groups(s)
        for k, c in READ.items():
            meds = [med(g[c]) for g in gs.values()]
            sp = spread(meds)
            if s == 0.5:
                SPREAD[k] = sp
            put(f"replicate_spread_s{s:g}_{k}", r3(sp),
                f"Max/min of the medians of {RDESC[k]} across the three independent default-model groups at s = {s:g} (series, rule-ablation 'all', map-comparison hub; 32 flies each)."
                + spread_note(meds, gs, c))
            put(f"replicate_kruskal_p_s{s:g}_{k}", pv(kw([g[c] for g in gs.values()])),
                f"Kruskal-Wallis p for {RDESC[k]} across the three independent default-model groups at s = {s:g}.")
        if s == 0.5:
            hub5 = grp("0.5", "hub")
            for k, c in READ.items():
                meds = [med(g[c]) for g in gs.values()] + [med(hub5[c])]
                put(f"replicate_spread_s0.5_with_exp2hub_{k}", r3(spread(meds)),
                    f"Max/min of the medians of {RDESC[k]} across four independent hub s = 0.5 groups (the three default groups plus exp2 hub at 48.3%)."
                    + spread_note(meds, {**gs, "exp2_hub": hub5}, c))
            put("replicate_kruskal_p_s0.5_with_exp2hub_brain",
                pv(kw([g.brain_spikes_A for g in gs.values()] + [hub5.brain_spikes_A])),
                "Kruskal-Wallis p for whole-brain spikes across the four independent hub s = 0.5 groups (series, ablation, map comparison, exp2).")

    for r in REFS:
        L = LV[r]
        here = present(r)
        for k, c in READ.items():
            if k in ("KCsurv",):
                continue
            groups = {m: (("e2", r, m), grp(r, m)[c]) for m in here}
            mm, _, info = maxmin(groups)
            sp = SPREAD[k]
            v = verdict(mm, info, sp)
            put(f"{L}_{k}_posthoc_verdict_own_spread", v,
                f"Post hoc (not prespecified) verdict for {RDESC[k]} at level {L}: the prespecified rule applied with this readout's own s = 0.5 replicate spread "
                f"({'undefined' if sp is None else 'inf' if np.isinf(sp) else f'{sp:.3f}'}) instead of 1.27, using the maxmin ratio and CI; 'placement' requires hub to be the only lowest map."
                + tie_text(info, med(H2[c]), "exp2 healthy"))
        # the prespecified rule recomputed with this script's bootstrap
        groups = {m: (("e2", r, m), grp(r, m).brain_spikes_A) for m in here}
        mm, _, info = maxmin(groups)
        put(f"{L}_primary_verdict_recomputed", verdict(mm, info, REPLICATE_SPREAD),
            f"Prespecified whole-brain verdict at level {L} recomputed with this script's bootstrap (seed {seed}); should equal the summary.json verdict.")

    # =========================================================================== ceiling
    for r in ("0.1", "0.25"):
        L = LV[r]
        hubm = med(grp(r, "hub").brain_spikes_A)
        put(f"{L}_ceiling_ratio_healthy_over_hub_brain", r3(med(H2.brain_spikes_A) / hubm),
            f"Largest antihub/hub whole-brain ratio possible at level {L} unless antihub flies exceed the exp2 healthy median (healthy median / hub median).")
        hp = 100 * hubm / med(H2.brain_spikes_A)
        put(f"{L}_burden_verdict_max_antihub_pct", r1(REPLICATE_SPREAD * hp),
            f"Highest antihub whole-brain median (% of exp2 healthy) that still yields the 'burden' verdict at level {L} (1.27 x hub %).")
        put(f"{L}_burden_verdict_min_deficit_ratio", r3((100 - REPLICATE_SPREAD * hp) / (100 - hp)),
            f"Smallest antihub/hub whole-brain deficit ratio that still yields the 'burden' verdict at level {L}.")
        # the prespecified bootstrap with the 32 exp2 healthy flies standing in for the antihub group
        groups = {"hub": (("e2", r, "hub"), grp(r, "hub").brain_spikes_A),
                  "shuffled": (("e2", r, "shuffled"), grp(r, "shuffled").brain_spikes_A),
                  "antihub": (("e2h",), H2.brain_spikes_A)}
        mm, _, info = maxmin(groups)
        put(f"{L}_ceiling_ratio_healthy_as_antihub_maxmin", mm,
            f"Prespecified max/min whole-brain ratio at level {L} with the 32 exp2 healthy flies in place of the antihub group, [point, 95% maxmin CI]; a lower bound <= 1.27 means 'placement' was unreachable.")

    # =========================================================================== equal-neuron design
    for s in (0.5, 1.0):
        N = f"neu_s{s:g}"
        for m in MAPS:
            g = VU[VU.condition == f"{m}_s{s:g}"]
            gid = ("vu", s, m)
            gd = f"the 32 map-comparison flies (vulnerability_x4.csv) of the {m} map at s = {s:g} (equal neurons removed)"
            put(f"{N}_{m}_contacts_lost_pct_calibration", r1(cal_burden(m, s)),
                f"Fraction of functional contacts lost (%) by the {m} map at s = {s:g}: mean of the 3 calibration draws in results/exp2/calibration.csv (not measured in the map-comparison flies).")
            put(f"{N}_{m}_neurons_removed_pct", round(100 * g.dead_frac.median(), 2), f"Median per-fly fraction of neurons removed (%) in {gd}.")
            put(f"{N}_{m}_KC_removed_pct", round(100 * g.kc_dead_frac.median(), 2), f"Median per-fly fraction of Kenyon cells removed (%) in {gd}.")
            put(f"{N}_{m}_mean_load", round(float(g.mean_load.median()), 4), f"Median per-fly mean pathology load in {gd}. {LOAD_POP}")
            for k, c in READ.items():
                put(f"{N}_{m}_{k}_pct", pct_tri(gid, g[c], ("hs",), HS[c]),
                    f"Median {RDESC[k]} as % of the series healthy median, [estimate, 95% bootstrap lo, hi], in {gd}.")
                put(f"{N}_{m}_{k}_median_raw", round(med(g[c]), 4), f"Median {RDESC[k]} in raw units in {gd}.")
                put(f"{N}_{m}_{k}_deficit_pct", r1(100 - 100 * med(g[c]) / med(HS[c])),
                    f"Deficit of {RDESC[k]} = 100 - median as % of the series healthy median, in {gd}.")
            put(f"{N}_{m}_n_GF_below100Hz", int((g.GF_loom_hz < 100).sum()), f"Number of flies (of 32) with giant fibre rate below 100 Hz in {gd}.")
            put(f"{N}_{m}_n_MN9_silent", int((g.MN9_sugar_hz == 0).sum()), f"Number of flies (of 32) with MN9 feeding drive 0 Hz in {gd}.")
            put(f"{N}_{m}_frac_PN_below95pct", round(float((g.PN_hz_A < 0.95 * med(HS.PN_hz_A)).mean()), 3),
                f"Fraction of flies whose PN rate (whole ALPN class, PN_hz_A) is below 95% of the series healthy median, in {gd}.")
        gs = {m: VU[VU.condition == f"{m}_s{s:g}"] for m in MAPS}
        for k, c in READ.items():
            put(f"{N}_{k}_kruskal_p", pv(kw([gs[m][c] for m in MAPS])),
                f"Kruskal-Wallis p for {RDESC[k]} across the three maps at s = {s:g} with equal neurons removed (32 flies each).")
            for i, a in enumerate(MAPS):
                for b in MAPS[i + 1:]:
                    put(f"{N}_{k}_mw_p_{a}_vs_{b}", pv(mw(gs[a][c], gs[b][c])),
                        f"Two-sided Mann-Whitney p for {RDESC[k]}, {a} vs {b}, equal neurons removed at s = {s:g}.")
            mm, pr, info = maxmin({m: (("vu", s, m), gs[m][c]) for m in MAPS})
            tie = tie_text(info, med(HS[c]), "series healthy")
            put(f"{N}_{k}_ratio_high_low_maxmin", mm,
                f"Ratio of the highest ({info['hi']}) to the lowest ({info['lo']}) map median of {RDESC[k]} with equal neurons removed at s = {s:g}, [point, 95% maxmin CI]."
                + " Max/min is at least 1 in every resample, so the lower bound is at least 1 by construction and does not test 'no difference'; "
                  f"for a two-sided comparison quote {N}_shuffled_over_hub_{k}_ratio or {N}_antihub_over_hub_{k}_ratio." + tie)
            put(f"{N}_{k}_highest_lowest_map", rank_labels(info), f"Maps with the highest and lowest median {RDESC[k]} with equal neurons removed at s = {s:g}." + tie)
            for m in ("shuffled", "antihub"):
                put(f"{N}_{m}_over_hub_{k}_ratio", ratio_pair(("vu", s, m), gs[m][c], ("vu", s, "hub"), gs["hub"][c], f"hub (s = {s:g})"),
                    f"Ratio of the {m} to the hub median of {RDESC[k]} with equal neurons removed at s = {s:g}, [point, 95% CI].")
                put(f"{N}_{m}_over_hub_{k}_deficit_ratio",
                    deficit_ratio(("vu", s, m), gs[m][c], ("vu", s, "hub"), gs["hub"][c], ("hs",), HS[c]),
                    f"Deficit ratio {m}/hub for {RDESC[k]} with equal neurons removed at s = {s:g} (series healthy), [point, 95% CI]; null if hub deficit < {MIN_DEF}%.")
        for m in ("shuffled", "antihub"):
            for cnt, crit, nm in (("GF_loom_hz", lambda x: x < 100, "GF_below100Hz"),
                                  ("MN9_sugar_hz", lambda x: x == 0, "MN9_silent")):
                a, b = int(crit(gs[m][cnt]).sum()), int(crit(gs["hub"][cnt]).sum())
                put(f"{N}_{m}_vs_hub_{nm}_fisher_p", pv(fisher(a, 32, b, 32)),
                    f"Two-sided Fisher exact p for the number of flies with {nm.replace('_', ' ')}, {m} vs hub, equal neurons removed at s = {s:g}.")
    ah1 = VU[VU.condition == "antihub_s1"]
    put("neu_s1_antihub_brain_pct_vs_exp2_healthy", r1(100 * med(ah1.brain_spikes_A) / med(H2.brain_spikes_A)),
        "Median whole-brain spikes of the 32 antihub s = 1.0 map-comparison flies as % of the EXP2 healthy median (cross-design reference).")

    # =========================================================================== like-for-like
    # (1) hub vs shuffled at hub s = 0.5: equal neurons vs equal contacts
    vh, vs = VU[VU.condition == "hub_s0.5"], VU[VU.condition == "shuffled_s0.5"]
    eh, es = grp("0.5", "hub"), grp("0.5", "shuffled")
    for k, c in READ.items():
        put(f"l4l_hubs05_eqneu_shuffled_over_hub_{k}", ratio_pair(("vu", 0.5, "shuffled"), vs[c], ("vu", 0.5, "hub"), vh[c], "hub (s = 0.5)"),
            f"Shuffled/hub ratio of median {RDESC[k]} at hub s = 0.5 with equal neurons removed (vulnerability_x4, shuffled s = 0.5), [point, 95% CI].")
        put(f"l4l_hubs05_eqsyn_shuffled_over_hub_{k}", ratio_pair(("e2", "0.5", "shuffled"), es[c], ("e2", "0.5", "hub"), eh[c], "hub (48.3%)"),
            f"Shuffled/hub ratio of median {RDESC[k]} at hub s = 0.5 with equal contacts lost (exp2 48.3%, shuffled s = 0.7348), [point, 95% CI].")
        put(f"l4l_hubs05_eqneu_mw_p_{k}", pv(mw(vs[c], vh[c])), f"Two-sided Mann-Whitney p, shuffled vs hub, {RDESC[k]}, equal neurons at s = 0.5.")
        put(f"l4l_hubs05_eqsyn_mw_p_{k}", pv(mw(es[c], eh[c])), f"Two-sided Mann-Whitney p, shuffled vs hub, {RDESC[k]}, equal contacts at 48.3%.")
        put(f"hubs05_exp2_vs_mapcomp_mw_p_{k}", pv(mw(eh[c], vh[c])),
            f"Two-sided Mann-Whitney p comparing raw {RDESC[k]} between the two independent hub s = 0.5 groups (exp2 48.3% vs map comparison).")
    put("l4l_hubs05_eqneu_neurons_removed_pct", [round(100 * vh.dead_frac.median(), 2), round(100 * vs.dead_frac.median(), 2)],
        "Median neurons removed (%) [hub, shuffled] in the equal-neuron pair at s = 0.5.")
    put("l4l_hubs05_eqneu_contacts_lost_pct", [r1(cal_burden("hub", 0.5)), r1(cal_burden("shuffled", 0.5))],
        "Contacts lost (%) [hub, shuffled] at s = 0.5 (calibration means), the equal-neuron pair.")
    put("l4l_hubs05_eqsyn_neurons_removed_pct", [round(100 * eh.dead_frac.median(), 2), round(100 * es.dead_frac.median(), 2)],
        "Median neurons removed (%) [hub, shuffled] in the equal-contact pair at 48.3%.")
    put("l4l_hubs05_eqneu_brain_deficit_ratio_shuffled_over_hub",
        deficit_ratio(("vu", 0.5, "shuffled"), vs.brain_spikes_A, ("vu", 0.5, "hub"), vh.brain_spikes_A, ("hs",), HS.brain_spikes_A),
        "Whole-brain deficit ratio shuffled/hub at s = 0.5 with equal neurons (series healthy), [point, 95% CI].")
    put("l4l_hubs05_eqsyn_brain_deficit_ratio_shuffled_over_hub",
        deficit_ratio(("e2", "0.5", "shuffled"), es.brain_spikes_A, ("e2", "0.5", "hub"), eh.brain_spikes_A, ("e2h",), H2.brain_spikes_A),
        "Whole-brain deficit ratio shuffled/hub at 48.3% with equal contacts (exp2 healthy), [point, 95% CI].")
    hubgroups = {"series": SC[np.isclose(SC.stage, 0.5)], "ablation": CO[CO.condition == "all_s0.5"],
                 "mapcomp": vh, "exp2": eh}
    for nm, g in hubgroups.items():
        put(f"hubs05_group_{nm}_brain_median", float(med(g.brain_spikes_A)),
            f"Median whole-brain spikes of the independent hub s = 0.5 group '{nm}' (32 flies).")
        put(f"l4l_shuffled_s05_over_hub_{nm}_brain", r3(med(vs.brain_spikes_A) / med(g.brain_spikes_A)),
            f"Equal-neuron shuffled s = 0.5 median whole-brain spikes divided by the '{nm}' hub s = 0.5 group median (alternative denominators).")
    pool = pd.concat(hubgroups.values())
    put("hubs05_pooled4_brain_median", float(med(pool.brain_spikes_A)), "Median whole-brain spikes of the four hub s = 0.5 groups pooled (128 flies).")
    put("l4l_shuffled_s05_over_hub_pooled4_brain", r3(med(vs.brain_spikes_A) / med(pool.brain_spikes_A)),
        "Equal-neuron shuffled s = 0.5 median whole-brain spikes divided by the pooled four-group hub s = 0.5 median.")
    put("hubs05_exp2_vs_mapcomp_n_GF_below100Hz", [int((eh.GF_loom_hz < 100).sum()), int((vh.GF_loom_hz < 100).sum())],
        "Flies with giant fibre rate below 100 Hz [exp2 hub 48.3%, map-comparison hub s = 0.5], 32 each.")
    put("hubs05_exp2_vs_mapcomp_GF_fail_fisher_p", pv(fisher(int((eh.GF_loom_hz < 100).sum()), 32, int((vh.GF_loom_hz < 100).sum()), 32)),
        "Two-sided Fisher exact p for the giant fibre failure counts of the two independent hub s = 0.5 groups.")

    # (2) within-exp2 neuron-matched pairs
    PAIRS = {"pairA_antihub_s0.281_vs_hub_s0.25": (("0.1", "antihub"), ("0.25", "hub"), ("0.25", "antihub")),
             "pairB_antihub_s0.7175_vs_shuffled_s0.7348": (("0.25", "antihub"), ("0.5", "shuffled"), ("0.25", "antihub"))}
    for pn, ((ra, ma), (rb, mb), _) in PAIRS.items():
        A, B = grp(ra, ma), grp(rb, mb)
        put(f"{pn}_neurons_removed_pct", [round(100 * A.dead_frac.median(), 2), round(100 * B.dead_frac.median(), 2)],
            f"Median neurons removed (%) [{ma} {LV[ra]}, {mb} {LV[rb]}] in the within-exp2 near-neuron-matched pair.")
        put(f"{pn}_contacts_lost_pct", [r1(100 * A.syn_lost_frac.median()), r1(100 * B.syn_lost_frac.median())],
            f"Median contacts lost (%) [{ma} {LV[ra]}, {mb} {LV[rb]}] in the within-exp2 near-neuron-matched pair.")
        put(f"{pn}_mean_load", [round(float(A.mean_load.median()), 3), round(float(B.mean_load.median()), 3)],
            f"Median per-fly mean load [{ma} {LV[ra]}, {mb} {LV[rb]}] in the within-exp2 near-neuron-matched pair. {LOAD_POP}")
        for k, c in READ.items():
            put(f"{pn}_{k}_ratio", ratio_pair(("e2", ra, ma), A[c], ("e2", rb, mb), B[c], f"{mb} ({LV[rb]})"),
                f"Ratio of medians {ma} ({LV[ra]}) / {mb} ({LV[rb]}) of {RDESC[k]}, near-equal neurons removed within exp2, [point, 95% CI].")
            put(f"{pn}_{k}_mw_p", pv(mw(A[c], B[c])),
                f"Two-sided Mann-Whitney p of {RDESC[k]}, {ma} ({LV[ra]}) vs {mb} ({LV[rb]}), near-equal neurons removed within exp2.")
    # equal-contact comparators for the pairs
    A, B = grp("0.25", "antihub"), grp("0.25", "shuffled")
    for k, c in READ.items():
        put(f"syn27.1_antihub_over_shuffled_{k}_ratio", ratio_pair(("e2", "0.25", "antihub"), A[c], ("e2", "0.25", "shuffled"), B[c], "shuffled (27.1%)"),
            f"Ratio of medians antihub/shuffled of {RDESC[k]} at equal contacts lost (27.1%), [point, 95% CI]; comparator for pair B.")
    # deficit ratios: equal neurons vs matched (antihub/hub, whole brain) are in neu_s0.5_antihub_over_hub_brain_deficit_ratio
    # and syn11.5/syn27.1_antihub_over_hub_brain_deficit_ratio.

    # (3) the other anchor (cross-design, linear interpolation over the measured exp2 levels)
    xs = [TGT[r] for r in REFS]
    hub_pct = [100 * med(grp(r, "hub").brain_spikes_A) / med(H2.brain_spikes_A) for r in REFS]
    ah_x = [TGT[r] for r in ("0.1", "0.25")]
    ah_pct = [100 * med(grp(r, "antihub").brain_spikes_A) / med(H2.brain_spikes_A) for r in ("0.1", "0.25")]
    b_ah, b_sh = cal_burden("antihub", 0.5), cal_burden("shuffled", 0.5)
    hub_at_ah = float(np.interp(b_ah, xs, hub_pct))
    hub_at_sh = float(np.interp(b_sh, xs, hub_pct))
    ah_at_ah = float(np.interp(b_ah, ah_x, ah_pct))
    vu_ah5 = 100 * med(VU[VU.condition == "antihub_s0.5"].brain_spikes_A) / med(HS.brain_spikes_A)
    vu_sh5 = 100 * med(vs.brain_spikes_A) / med(HS.brain_spikes_A)
    put("interp_hub_brain_pct_at_antihub_s05_burden", r1(hub_at_ah),
        f"Exp2 hub whole-brain % of exp2 healthy, linearly interpolated to the antihub s = 0.5 burden ({b_ah:.1f}% contacts; calibration).")
    put("interp_antihub_brain_pct_at_antihub_s05_burden", r1(ah_at_ah),
        f"Exp2 antihub whole-brain % of exp2 healthy, linearly interpolated between 11.5% and 27.1% to {b_ah:.1f}% contacts.")
    put("interp_ratio_antihub_s05_eqneu_over_hub_eqsyn", r3(vu_ah5 / hub_at_ah),
        "Antihub s = 0.5 equal-neuron whole-brain % (series healthy) over interpolated exp2 hub % at the same contacts lost (antihub-anchored, cross-design approximation).")
    put("interp_ratio_antihub_over_hub_at_antihub_s05_burden_exp2", r3(ah_at_ah / hub_at_ah),
        "Interpolated exp2 antihub over interpolated exp2 hub whole-brain % at the antihub s = 0.5 burden (within-exp2 interpolation).")
    put("interp_hub_brain_pct_at_shuffled_s05_burden", r1(hub_at_sh),
        f"Exp2 hub whole-brain % of exp2 healthy, linearly interpolated to the shuffled s = 0.5 burden ({b_sh:.1f}% contacts).")
    put("interp_ratio_shuffled_s05_eqneu_over_hub_eqsyn", r3(vu_sh5 / hub_at_sh),
        "Shuffled s = 0.5 equal-neuron whole-brain % (series healthy) over interpolated exp2 hub % at the same contacts lost (shuffled-anchored, cross-design approximation).")
    ser = SC.groupby("stage").brain_spikes_A.median()
    ser_pct = 100 * ser / med(HS.brain_spikes_A)
    for r in ("0.1", "0.25"):
        s_ah = MATCH["maps"][r]["antihub"]["s"]
        hp = float(np.interp(s_ah, ser_pct.index.values, ser_pct.values))
        ap = 100 * med(grp(r, "antihub").brain_spikes_A) / med(H2.brain_spikes_A)
        put(f"interp_series_hub_brain_pct_at_s{s_ah}", r1(hp),
            f"Severity-series (hub map) whole-brain median % of series healthy, linearly interpolated to the antihub severity s = {s_ah}.")
        put(f"interp_ratio_antihub_s{s_ah}_over_series_hub_same_s", r3(ap / hp),
            f"Exp2 antihub ({LV[r]}, s = {s_ah}) whole-brain % over the interpolated series hub % at the same s (antihub-anchored equal-severity ratio, cross-design).")
        put(f"interp_deficit_ratio_antihub_s{s_ah}_over_series_hub_same_s", r3((100 - ap) / (100 - hp)),
            f"Antihub/hub whole-brain deficit ratio at the same s = {s_ah} (exp2 antihub vs interpolated series hub; cross-design).")

    # =========================================================================== healthy groups
    praw = {}
    for k, c in READ.items():
        if k == "KCsurv":
            continue
        a, b = H2[c], HS[c]
        put(f"healthy_exp2_vs_series_{k}_medians", [round(med(a), 4), round(med(b), 4)],
            f"Median {RDESC[k]} of the exp2 healthy (32) and the series healthy (64) flies.")
        put(f"healthy_exp2_over_series_{k}_ratio", r3(med(a) / med(b)), f"Exp2 healthy median / series healthy median of {RDESC[k]}.")
        praw[k] = mw(a, b)
        put(f"healthy_exp2_vs_series_{k}_mw_p", pv(praw[k]), f"Two-sided Mann-Whitney p, exp2 healthy (32) vs series healthy (64), {RDESC[k]}.")
    for k, q in holm(praw).items():
        put(f"healthy_exp2_vs_series_{k}_mw_p_holm", pv(q),
            f"Holm-adjusted p (over the {len(praw)} readouts listed) for exp2 vs series healthy, {RDESC[k]}.")
    for k in ("brain", "LI"):
        c = READ[k]
        put(f"healthy_joint_{k}_median", round(med(JH[c]), 4),
            f"Median {RDESC[k]} of the {len(JH)} joint-cohort healthy flies (third healthy group, different protocol). {JOINT_PROTOCOL}")
        put(f"healthy_joint_vs_series_{k}_mw_p", pv(mw(JH[c], HS[c])),
            f"Two-sided Mann-Whitney p, joint healthy vs series healthy, {RDESC[k]}. {JOINT_PROTOCOL}")
        put(f"healthy_joint_vs_exp2_{k}_mw_p", pv(mw(JH[c], H2[c])),
            f"Two-sided Mann-Whitney p, joint healthy vs exp2 healthy, {RDESC[k]}. {JOINT_PROTOCOL}")

    # =========================================================================== threshold-shift ablation (confound)
    for s in (0.5, 1.0):
        a, b = CO[CO.condition == f"all_s{s:g}"], CO[CO.condition == f"no_hyper_s{s:g}"]
        for k in ("brain", "LH", "KCrate", "MBON", "LI", "MN9", "GF"):
            c = READ[k]
            put(f"ablation_s{s:g}_all_vs_nohyper_{k}_pct", [r1(100 * med(a[c]) / med(HS[c])), r1(100 * med(b[c]) / med(HS[c]))],
                f"Median {RDESC[k]} as % of series healthy [full model, model without the threshold shift] at s = {s:g} (components_x4.csv, 32 flies each).")
            put(f"ablation_s{s:g}_all_vs_nohyper_{k}_mw_p", pv(mw(a[c], b[c])),
                f"Two-sided Mann-Whitney p, full model vs no threshold shift, {RDESC[k]}, s = {s:g}.")

    # =========================================================================== seeds
    jseeds = set(JO.seed.astype(int))
    eseeds = set(E2.seed.astype(int))
    ov = sorted(jseeds & eseeds)
    put("seeds_exp2_joint_overlap_n", len(ov), "Number of pathology seeds shared by the 288 exp2 flies and the 176 joint-cohort flies.")
    put("seeds_exp2_joint_overlap_range_step", [ov[0], ov[-1], int(np.unique(np.diff(ov))[0]) if len(ov) > 1 else None],
        "First and last shared seed and the step between shared seeds.")
    emap = E2.set_index("seed")
    jmap = JO.set_index("seed")
    for m in ("healthy", "hub", "shuffled", "antihub"):
        put(f"seeds_overlap_n_exp2_{m}", int(sum(emap.loc[s_, "map"] == m for s_ in ov)),
            f"Number of the shared exp2/joint seeds that belong to exp2 {m} flies.")
    both_damaged = [s_ for s_ in ov if emap.loc[s_, "stage"] > 0 and jmap.loc[s_, "stage"] > 0]
    put("seeds_overlap_n_both_damaged", len(both_damaged), "Number of shared seeds with pathology (s > 0) in both the exp2 and the joint-cohort fly.")
    dup = [s_ for s_ in both_damaged if emap.loc[s_, "map"] == "hub" and np.isclose(emap.loc[s_, "stage"], jmap.loc[s_, "stage"])]
    put("seeds_overlap_identical_condition", [int(s_) for s_ in dup],
        "Shared seeds where both flies are hub map at the same s (identical pathology draw; joint cohort uses the hub map).")
    put("seeds_overlap_identical_dead_frac_equal",
        [bool(np.isclose(emap.loc[s_, "dead_frac"], jmap.loc[s_, "dead_frac"])) for s_ in dup],
        "Whether neurons-removed fractions are identical in the exp2 and joint flies of each identical-condition shared seed.")
    other = set()
    for f in ("stage_curve_x4", "components_x4", "vulnerability_x4", "scope_x4"):
        other |= set(pd.read_csv(ROOT / f"results/battery/{f}.csv").seed.astype(int))
    for p in (ROOT / "results/exp1/screen").glob("*_runs.csv"):
        mt = re.search(r"seed(\d+)", p.name)
        if mt:
            other.add(int(mt.group(1)))
    put("seeds_exp2_overlap_battery_exp1_n", len(eseeds & other),
        "Number of exp2 seeds shared with the four battery x4 designs (series, ablation, map comparison, KC-only) and the exp1 screen flies.")

    # =========================================================================== cross-checks against exp2_stats.json and battery_stats.json
    diffs = []
    for r in REFS:
        for m in present(r):
            for k in ("brain", "LH", "KCrate", "MBON", "LI", "MN9", "GF"):
                ref_pct = E2STATS["levels"][r]["maps"][m][READ[k]]["pct"]
                diffs.append(abs(F[f"{LV[r]}_{m}_{k}_pct"][0] - ref_pct))
    put("check_max_abs_diff_pct_vs_exp2_stats", round(max(diffs), 3),
        "Largest absolute difference between this script's point % of healthy and results/stats/exp2_stats.json over its 7 readouts and 8 groups (0 = identical).")
    dbat = []
    for m in MAPS:
        for k, c in (("brain", "brain_spikes_A"), ("MBON", "MBON_hz_A"), ("LI", "LI"), ("GF", "GF_loom_hz")):
            dbat.append(abs(F[f"neu_s0.5_{m}_{k}_pct"][0] - BSTATS["vulnerability"]["0.5"][m][c]["pct"]))
    put("check_max_abs_diff_pct_vs_battery_stats_vulnerability", round(max(dbat), 3),
        "Largest absolute difference between this script's equal-neuron s = 0.5 point % of healthy and results/stats/battery_stats.json vulnerability (4 readouts x 3 maps).")
    put("_bootstrap_resamples_seed", [NB, seed],
        "Bootstrap settings used for every interval in this file: [number of resamples, numpy default_rng seed].")

# =========================================================================== Monte Carlo check
def num(x):
    """JSON value -> float (None -> nan, 'inf' -> inf, '-inf' -> -inf)."""
    if x is None:
        return np.nan
    if isinstance(x, str):
        return {"inf": np.inf, "-inf": -np.inf}.get(x, np.nan)
    return float(x)


def g2(x):
    """A bound for the meaning text, 3 significant figures."""
    return f"{x:.3g}" if np.isfinite(x) else ("inf" if x > 0 else "-inf")


def bound_unstable(xs, step):
    """True if the bound values over seeds range wider than MC_REL_BOUND of the bound and MC_MIN_STEPS
    rounding steps (i.e. its second significant figure is not reproducible), or are infinite in some
    seeds only."""
    fin = np.isfinite(xs)
    if not fin.all():
        return bool(fin.any())
    rng = xs.max() - xs.min()
    return bool(rng > max(MC_REL_BOUND * abs(np.median(xs)), MC_MIN_STEPS * step - 1e-9))


def primary_mc(r):
    """Prespecified whole-brain maxmin lower bound and verdict at level r over MC_VERDICT_SEEDS seeds."""
    here = present(r)
    xs = {m: np.asarray(grp(r, m).brain_spikes_A, float) for m in here}
    meds = {m: float(np.median(x)) for m, x in xs.items()}
    ratio = max(meds.values()) / min(meds.values())
    bot = [m for m in here if meds[m] == min(meds.values())]
    los, vs = [], []
    for sd in range(MC_VERDICT_SEEDS):
        rng = np.random.default_rng(SEED + 1000 + sd)
        B = np.column_stack([np.median(rng.choice(x, (NB, len(x))), 1) for x in xs.values()])
        lo = float(np.percentile(B.max(1) / B.min(1), 2.5))
        los.append(lo)
        vs.append(verdict([ratio, lo], {"bot": bot}, REPLICATE_SPREAD))
    return np.array(los), vs


build(SEED)
F0, M0, K0 = dict(F), dict(M), set(BOOT_KEYS)
ALT = []
for sd in MC_SEEDS:
    build(sd)
    ALT.append(dict(F))
NS = 1 + len(MC_SEEDS)
flagged = []
for k in sorted(K0, key=list(F0).index):
    v0 = F0[k]
    if v0[1] is None:
        continue
    los = np.array([num(v0[1])] + [num(a[k][1]) for a in ALT])
    his = np.array([num(v0[2])] + [num(a[k][2]) for a in ALT])
    step = 0.1 if k.endswith("_pct") else 0.001        # r1 for % of healthy, r3 for ratios
    parts = []
    for nm, xs in (("lower", los), ("upper", his)):
        if bound_unstable(xs, step):
            parts.append(f"the {nm} bound ranged {g2(np.nanmin(xs) if np.isfinite(xs).any() else xs.min())} to "
                         f"{g2(xs.max())}" + ("" if np.isfinite(xs).all() else f" (infinite in {int((~np.isfinite(xs)).sum())} of {NS} seeds)"))
    if parts:
        flagged.append(k)
        M0[k] += (f" Monte Carlo sensitive: over {NS} bootstrap seeds (this one and {len(MC_SEEDS)} alternatives) "
                  + " and ".join(parts) + f"; {'these bounds are' if len(parts) > 1 else 'this bound is'} not "
                  "reproducible to the reported precision, so quote the interval to at most 2 significant figures.")
for k in [k for k in F0 if k.endswith("_posthoc_verdict_own_spread")]:
    vs = [F0[k]] + [a[k] for a in ALT]
    n_same = sum(v == F0[k] for v in vs)
    if n_same < NS:
        flagged.append(k)
        others = sorted(set(vs) - {F0[k]})
        M0[k] += (f" Monte Carlo sensitive: the verdict is '{F0[k]}' in {n_same} of {NS} bootstrap seeds "
                  f"(otherwise {', '.join(others)}).")
for r in REFS:
    L = LV[r]
    los, vs = primary_mc(r)
    v0 = F0[f"{L}_primary_verdict_recomputed"]
    agree = float(np.mean([v == v0 for v in vs]))
    F0[f"{L}_primary_maxmin_lo_mc"] = [round(float(los.mean()), 3), round(float(los.min()), 3), round(float(los.max()), 3),
                                       round(float((los > REPLICATE_SPREAD).mean()), 3)]
    M0[f"{L}_primary_maxmin_lo_mc"] = (
        f"Monte Carlo spread of the prespecified whole-brain maxmin lower bound at level {L} over {MC_VERDICT_SEEDS} "
        f"bootstrap seeds of {NB} resamples: [mean, min, max, fraction of seeds with lower bound > {REPLICATE_SPREAD}]. "
        f"Compare {L}_brain_ratio_high_low_maxmin and the summary.json interval.")
    F0[f"{L}_primary_verdict_mc_agreement"] = round(agree, 3)
    M0[f"{L}_primary_verdict_mc_agreement"] = (
        f"Fraction of {MC_VERDICT_SEEDS} bootstrap seeds whose prespecified whole-brain verdict at level {L} equals "
        f"{L}_primary_verdict_recomputed ('{v0}').")
    if agree < 1:
        M0[f"{L}_primary_verdict_recomputed"] += (
            f" Monte Carlo sensitive: the same verdict results in {100 * agree:.0f}% of {MC_VERDICT_SEEDS} bootstrap seeds "
            f"(lower bound mean {los.mean():.3f}, range {los.min():.3f} to {los.max():.3f}, against {REPLICATE_SPREAD}); "
            f"see {L}_primary_maxmin_lo_mc.")
        if f"{L}_brain_ratio_high_low_maxmin" in M0:
            M0[f"{L}_brain_ratio_high_low_maxmin"] += (
                f" The lower bound lies close to the {REPLICATE_SPREAD} criterion: over {MC_VERDICT_SEEDS} bootstrap "
                f"seeds it averages {los.mean():.3f} (range {los.min():.3f} to {los.max():.3f}) and exceeds "
                f"{REPLICATE_SPREAD} in {100 * (los > REPLICATE_SPREAD).mean():.0f}% of seeds.")
        flagged += [f"{L}_primary_verdict_recomputed", f"{L}_brain_ratio_high_low_maxmin"]
F0["_mc_check"] = [len(MC_SEEDS), MC_SEEDS[0], MC_SEEDS[-1], MC_REL_BOUND, MC_MIN_STEPS, MC_VERDICT_SEEDS]
M0["_mc_check"] = ("Monte Carlo check of every bootstrap interval: [number of alternative seeds, first, last, flag "
                   "threshold as a fraction of the bound, minimum range in rounding steps, seeds for the "
                   "prespecified-verdict check]. The whole file is recomputed with each alternative seed; a bound is "
                   "flagged when its range over all seeds exceeds both thresholds (its second significant figure is "
                   "not reproducible) or it is infinite in only some seeds, and the range is appended to the key's "
                   "meaning. Post hoc verdicts that change across these seeds are flagged as well.")
F0["_mc_sensitive_keys"] = flagged
M0["_mc_sensitive_keys"] = ("Keys whose interval bounds or verdict changed beyond the _mc_check thresholds across "
                            "bootstrap seeds; quote their bounds to at most 2 significant figures.")

out = dict(F0)
out["_meaning"] = M0
(OUT / "facts_exp2.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(f"wrote {OUT / 'facts_exp2.json'} with {len(F0)} facts; {len(flagged)} Monte Carlo sensitive")
