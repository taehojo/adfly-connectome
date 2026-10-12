"""Facts table of the battery designs (core simulation results).

    PYTHONIOENCODING=utf-8 python code/scripts/facts_battery.py

No simulation is run. Everything is computed from the per-fly tables of the
assay battery (results/battery/*_x4.csv), with a few values copied, with their
provenance, from results/stats/exp1_stats.json, results/exp1/screen/*_units.csv,
results/exp1/screen/*_runs.csv (baseline rows only),
results/exp1/summary/baseline_decomposition.csv, results/exp2/battery.csv and
results/validation/baseline_nostim.parquet (the MaleCNS neuron table with type,
side and superclass, row order = the simulation's neuron order). The connectome
pack itself is not needed: the hub vulnerability rank u of every neuron
(pathology.hub_vulnerability, i.e. Connectome.vuln) is read from
results/validation/hub_vulnerability_u.npz (keys u, ids = bodyId in neuron-table
order; computed from the MaleCNS pack on the cluster). Before it is used, that
table is checked here against every exp1 screen type row in which all neurons of
a type were lost (mean_rank must equal the type's mean u), every row with one lost
neuron (mean_rank must equal one of the type's u values), and the DNp01 ranks in
results/paper/output_neuron_loss.json.

Writes results/facts/facts_battery.json: a flat dict of facts plus "_meaning",
which maps every key to one sentence saying what it is.

Conventions (same as code/scripts/analyze_battery.py, which produced battery_stats.json):
  * healthy reference = the 64 stage-0, pan-scope flies of stage_curve_x4.csv. The
    components, vulnerability and scope designs have no healthy group of their own
    and are referred to this same group.
  * "% of healthy" = group median / healthy median x 100. Intervals: 95% percentile
    bootstrap of that ratio, resampling flies within the group and within the healthy
    group (4,000 resamples, numpy default_rng seeded from a fixed seed and the key).
  * Mann-Whitney U tests are two-sided; Spearman and Kruskal-Wallis from scipy.
  * LI_norm = LI / MBON_hz_A, undefined where MBON_hz_A < 0.5 Hz.
  * GF failure = GF_loom_hz < 100 Hz; MN9 failure (silent) = MN9_sugar_hz == 0.
  * AUC = direction-free Mann-Whitney AUC against healthy, max(U, 1 - U).
  * Every rate readout averages over all neurons of its population, removed ones
    included as silent (assay_battery.py), so it mixes loss of neurons with changes
    in the firing of survivors.
  * Readouts that can be undefined (LI_norm, GF latency) are summarised over the flies
    where they are defined; meanings give that n.
  * Bootstrap bounds are checked for Monte Carlo stability: each interval is also
    computed with 40,000 resamples, and any reported bound that differs from it by
    more than 1 percentage point (0.01 for ratios) is listed in boot_unstable_bounds,
    with its range over 8 further seeds of 4,000 resamples. The reported value stays
    the 4,000-resample one.
The *_x4 tables are the analysed data. The files without _x4 (a pilot that
re-simulated 240 flies with the same seeds) are not used here.
"""
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())  # code/scripts locally, scripts/ on the server
BAT = ROOT / "results/battery"
OUT = ROOT / "results/facts"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261009
N_BOOT = 4000
MBON_FLOOR = 0.5
FLOOR_PCT = 5.0            # a group median below 5% of healthy is called near-floor
GF_FAIL_HZ = 100.0
STIM_HZ = {"odour": 150.0, "PAM": 100.0, "loom": 150.0, "sugar": 150.0}   # assay_battery.py
DEATH_MAX = 0.5            # pathology.Stage default: P(removal) = load x death_max, load = s x u
EXP1_MIN_DEFICIT = 0.05    # exp1_counterfactual.py --min-deficit: |healthy - damaged| > 0.05 x |healthy| (strict)
BATTERY_BATCH = 48         # assay_battery.py --batch default (queue_x4.sh passes none); stimulus seed = 7 x 100 + first row of batch
BOOT_ALT_SEEDS = 8         # extra bootstrap seeds for the stability check of interval bounds
BOOT_TOL = {"pct": 1.0, "ratio": 0.01}

R = {"brain": "brain_spikes_A", "LH": "LH_hz_A", "PN": "PN_hz_A", "KC": "KC_hz_A",
     "KCrec": "KC_frac_A", "MBON": "MBON_hz_A", "valence": "V_pre_A", "LI": "LI",
     "LInorm": "LI_norm", "MN9": "MN9_sugar_hz", "GF": "GF_loom_hz", "GFlat": "GF_latency_ms",
     "PAM": "PAM_pair_hz"}
DESC = {"brain": "whole-brain spike count during odour A (1 s; removed neurons contribute none)",
        "LH": "mean lateral horn neuron rate during odour A (Hz; all LH neurons, removed ones counted as silent)",
        "PN": "mean rate of all 686 antennal-lobe projection neurons (class ALPN, removed ones counted as silent) during odour A (Hz); "
              "the 8 odour-A uPNs are themselves driven at 150 Hz, so this mostly tracks how many driven uPNs survive (see pn_* keys)",
        "KC": "mean Kenyon cell rate during odour A (Hz; all Kenyon cells, removed ones counted as silent)",
        "KCrec": "fraction of all Kenyon cells (removed ones counted as silent) firing >= 1 spike during odour A",
        "MBON": "mean MBON rate during odour A (Hz; all MBONs, removed ones counted as silent)",
        "valence": "naive MB valence V_pre_A (approach minus avoidance MBON rates, Hz)",
        "LI": "learning index",
        "LInorm": "learning index / MBON rate (defined only where MBON rate >= 0.5 Hz)",
        "MN9": "MN9 feeding drive to sugar (Hz; mean of the two MN9 neurons, a removed one counted as silent)",
        "GF": "giant fibre (DNp01) rate to looming (Hz; mean of the two DNp01, a removed one counted as silent)",
        "GFlat": "giant fibre first-spike latency to looming (ms; undefined in flies without a GF spike)",
        "PAM": "PAM rate during odour-PAM pairing (mean over the 15 PAM types of each type's mean rate, removed PAMs counted as silent, Hz); "
               "the PAMs are themselves driven by 100 Hz Poisson input during pairing, so this mostly measures the type-averaged "
               "fraction of driven PAMs that survive (delivered teaching input), not a circuit response (see stim_PAM_* keys)"}
UNDEF = {"LInorm": "MBON rate >= 0.5 Hz, a survivor-selected subset", "GFlat": "at least one GF spike"}
SERIES_READS = ["brain", "LH", "PN", "KC", "KCrec", "MBON", "valence", "LI", "LInorm", "MN9", "GF", "GFlat", "PAM"]
HOLM_FAMILY = ["brain", "LH", "PN", "KC", "KCrec", "MBON", "valence", "LI", "LInorm", "MN9", "GF", "GFlat"]
REP_READS = ["brain", "LH", "PN", "KC", "KCrec", "MBON", "valence", "LI", "MN9", "GF", "PAM"]
REP_SEVEN = ["brain", "LH", "MBON", "KC", "LI", "MN9", "GF"]       # the seven readouts of the 14 replicate tests in battery_stats.json
ABL_READS = ["brain", "LH", "PN", "KC", "KCrec", "MBON", "LI", "LInorm", "MN9", "GF", "PAM"]
CONDS = ["all", "hyper_only", "synloss_only", "death_only", "no_hyper"]
COND_DESC = {"all": "full model (threshold shift + synapse loss + neuron loss)",
             "hyper_only": "threshold shift alone", "synloss_only": "synapse loss alone",
             "death_only": "neuron loss alone", "no_hyper": "synapse loss + neuron loss, no threshold shift"}
MAPS = ["hub", "shuffled", "antihub"]
MAP_READS = ["brain", "LH", "PN", "KC", "KCrec", "MBON", "LI", "MN9", "GF", "PAM"]

F, M = {}, {}
UNSTABLE, _pending = {}, []      # bootstrap bounds that move with the seed, attached to the key being written


def put(key, value, meaning):
    assert key not in F, key
    F[key] = value
    M[key] = meaning
    if _pending:                 # intervals computed while evaluating this put's arguments
        UNSTABLE[key] = list(_pending)
        _pending.clear()


def sk(s):
    return f"s{s:g}"


def r1(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 1)


def r3(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


def pg(p):
    return None if p is None or not np.isfinite(p) else float(f"{p:.3g}")


def rng_for(label):
    return np.random.default_rng([SEED, zlib.crc32(label.encode())])


def clean(x):
    return np.asarray(pd.Series(x).dropna(), float)


def _boot_bounds(a, b, rng, n_boot, scale):
    """2.5 and 97.5 percentiles of median(a*)/median(b*) x scale, flies resampled within each group."""
    ba = np.median(rng.choice(a, (n_boot, len(a)), replace=True), 1)
    bb = np.median(rng.choice(b, (n_boot, len(b)), replace=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = ba / bb * scale
    return np.percentile(r[np.isfinite(r)], [2.5, 97.5])


def _stability(a, b, label, scale, lo, hi, kind):
    """Queue the interval for boot_unstable_bounds if a reported bound is off its 10x-resample value by more than the tolerance."""
    rnd = r1 if kind == "pct" else r3
    big = _boot_bounds(a, b, rng_for(f"{label}#big"), 10 * N_BOOT, scale)
    if max(abs(lo - big[0]), abs(hi - big[1])) > BOOT_TOL[kind]:
        alt = np.array([_boot_bounds(a, b, rng_for(f"{label}#alt{i}"), N_BOOT, scale) for i in range(BOOT_ALT_SEEDS)])
        los, his = np.r_[lo, alt[:, 0]], np.r_[hi, alt[:, 1]]
        _pending.append({"label": label, "reported": [rnd(lo), rnd(hi)],
                         f"lo_range_{BOOT_ALT_SEEDS + 1}_seeds": [rnd(los.min()), rnd(los.max())],
                         f"hi_range_{BOOT_ALT_SEEDS + 1}_seeds": [rnd(his.min()), rnd(his.max())],
                         f"at_{10 * N_BOOT}_resamples": [rnd(big[0]), rnd(big[1])]})


def pct_ci(x, h, label, check=True):
    """[median(x)/median(h)*100, lo, hi] with a two-group bootstrap of flies."""
    x, h = clean(x), clean(h)
    if len(x) == 0 or len(h) == 0 or np.median(h) == 0:
        return [None, None, None]
    est = np.median(x) / np.median(h) * 100
    lo, hi = _boot_bounds(x, h, rng_for(label), N_BOOT, 100.0)
    if check:
        _stability(x, h, label, 100.0, lo, hi, "pct")
    return [r1(est), r1(lo), r1(hi)]


def ratio_ci(a, b, label):
    """[median(a)/median(b), lo, hi]: ratio of two group medians with a bootstrap interval."""
    a, b = clean(a), clean(b)
    if len(a) == 0 or len(b) == 0 or np.median(b) == 0:
        return [None, None, None]
    lo, hi = _boot_bounds(a, b, rng_for(label), N_BOOT, 1.0)
    _stability(a, b, label, 1.0, lo, hi, "ratio")
    return [r3(np.median(a) / np.median(b)), r3(lo), r3(hi)]


def med_ratio(a, b):
    """Unrounded median(a)/median(b), None if undefined."""
    a, b = clean(a), clean(b)
    if len(a) == 0 or len(b) == 0 or np.median(b) == 0:
        return None
    return float(np.median(a) / np.median(b))


def n_txt(x, k=None):
    """'32 flies', or 'n of 32 flies with <criterion>' for a readout that is undefined in some flies."""
    n, N = int(pd.Series(x).notna().sum()), len(x)
    if n == N:
        return f"{N} flies"
    return f"{n} of {N} flies, those with {UNDEF.get(k, 'a defined value')}"


def mw(a, b):
    a, b = clean(a), clean(b)
    if len(a) < 3 or len(b) < 3:
        return None
    if np.all(a == a[0]) and np.all(b == a[0]):
        return 1.0
    return pg(st.mannwhitneyu(a, b, alternative="two-sided")[1])


def kw(*groups):
    gs = [clean(g) for g in groups]
    if min(map(len, gs)) < 3:
        return None
    try:
        return pg(st.kruskal(*gs).pvalue)
    except ValueError:            # all values identical
        return 1.0


def auc(a, b):
    a, b = clean(a), clean(b)
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    u = (gt + 0.5 * eq) / (len(a) * len(b))
    return max(u, 1.0 - u)


def spear(x, y):
    d = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}).dropna()
    rho, p = st.spearmanr(d.x, d.y)
    return float(rho), float(p), int(len(d))


def holm(p):
    o = sorted(p, key=lambda k: p[k])
    m, prev, out = len(o), 0.0, {}
    for i, k in enumerate(o):
        prev = min(1.0, max(prev, (m - i) * p[k]))
        out[k] = prev
    return out


def partial_spearman(x, y, z):
    rx, ry, rz = (st.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r, p = st.pearsonr(ex, ey)
    return float(r), float(p)


def prep(d):
    d = d.copy()
    d["LI_norm"] = d["LI"] / d["MBON_hz_A"].where(d["MBON_hz_A"] >= MBON_FLOOR)
    return d


# ================================================================ data
sc = prep(pd.read_csv(BAT / "stage_curve_x4.csv"))
sc = sc[sc.scope == "pan"].copy()
H = sc[sc.stage == 0]
STAGES = sorted(s for s in sc.stage.unique() if s > 0)
ALLST = [0.0] + STAGES
co = prep(pd.read_csv(BAT / "components_x4.csv"))
vu = prep(pd.read_csv(BAT / "vulnerability_x4.csv"))
sp = prep(pd.read_csv(BAT / "scope_x4.csv"))
neu = pd.read_parquet(ROOT / "results/validation/baseline_nostim.parquet")
bstats = json.loads((ROOT / "results/stats/battery_stats.json").read_text())


def grp_series(s):
    return sc[sc.stage == s]


def grp_cond(c, s):
    return co[co.condition == f"{c}_{sk(s)}"]


def grp_map(m, s):
    return vu[vu.condition == f"{m}_{sk(s)}"]


def grp_mb(s):
    return sp[sp.stage == s]


hmed = {k: float(H[c].median()) for k, c in R.items()}

# ================================================================ 0. design and healthy reference
put("n_flies_series", int(len(sc)), "Flies in the severity series (stage_curve_x4.csv, pan-neuronal hub map): 64 healthy + 32 at each of 10 severities.")
put("n_flies_components", int(len(co)), "Flies in the rule ablation (components_x4.csv): 5 conditions x 2 severities (0.5, 1.0) x 32 flies.")
put("n_flies_maps", int(len(vu)), "Flies in the vulnerability-map comparison (vulnerability_x4.csv): hub, shuffled, antihub x 2 severities x 32 flies.")
put("n_flies_kc_only", int(len(sp)), "Flies with pathology confined to Kenyon cells (scope_x4.csv): 2 severities x 32 flies.")
put("n_healthy", int(len(H)), "Healthy reference flies (stage 0, pan scope, stage_curve_x4.csv) used as the denominator for every % of healthy in this file.")
put("series_stages", [float(s) for s in STAGES], "Severities s of the series (stage 0 = healthy is not listed); s = 1.0 is the end point.")
for k in SERIES_READS:
    put(f"healthy_median_{k}", round(hmed[k], 4), f"Median over the healthy flies ({n_txt(H[R[k]], k)}) of the {DESC[k]}.")
put("healthy_KC_max_hz", round(float(H.KC_hz_A.max()), 4), "Highest Kenyon cell rate (Hz, odour A) among the 64 healthy flies; the dotted line of Fig 6E.")
put("series_dead_pct_by_stage", [r3(100 * grp_series(s).dead_frac.mean()) for s in STAGES], "Mean % of all neurons removed per fly at each severity of the series (order of series_stages).")
put("series_kc_dead_pct_by_stage", [r3(100 * grp_series(s).kc_dead_frac.mean()) for s in STAGES], "Mean % of Kenyon cells removed per fly at each severity of the series (order of series_stages).")

# ================================================================ 1. severity series medians
for k in SERIES_READS:
    c = R[k]
    put(f"series_{k}_pct_by_stage",
        [r1(grp_series(s)[c].median() / hmed[k] * 100) if grp_series(s)[c].notna().any() else None for s in ALLST],
        f"Median {DESC[k]} at each stage (0, then series_stages) as % of the healthy median (point estimates; Fig 2 lines)"
        + (f"; medians over the flies with {UNDEF[k]} only ({k}_n_defined_by_stage)." if k in UNDEF else "."))
    for s in (0.5, 1.0):
        put(f"series_{sk(s)}_{k}_pct", pct_ci(grp_series(s)[c], H[c], f"series{s}{k}"),
            f"Severity series at s = {s:g} ({n_txt(grp_series(s)[c], k)}): median {DESC[k]} as % of the healthy median ({n_txt(H[c], k)}), [estimate, 95% bootstrap lo, hi].")
for s in (0.5, 1.0):
    g = grp_series(s)
    surv = g.KC_frac_A / (1 - g.kc_dead_frac)
    hs = H.KC_frac_A / (1 - H.kc_dead_frac)
    put(f"series_{sk(s)}_KCrec_survivors_pct", pct_ci(surv, hs, f"surv{s}"),
        f"Series at s = {s:g}: Kenyon cell recruitment counted among surviving Kenyon cells only (KC_frac_A / (1 - kc_dead_frac)), median as % of healthy, [est, lo, hi].")
g1 = grp_series(1.0)
put("series_s1_n_zero", {"MBON": int((g1.MBON_hz_A == 0).sum()), "LI_le0": int((g1.LI <= 0).sum()),
                         "valence_le0": int((g1.V_pre_A <= 0).sum()), "valence_eq0": int((g1.V_pre_A == 0).sum()),
                         "MN9_eq0": int((g1.MN9_sugar_hz == 0).sum()), "GF_eq0": int((g1.GF_loom_hz == 0).sum())},
    "Series at s = 1.0 (32 flies): number of flies with MBON rate exactly 0, learning index <= 0, naive valence <= 0 and == 0, MN9 drive 0, GF rate 0.")

# ================================================================ 2. trend tests, Holm family of twelve
raw, tr = {}, {}
for k in HOLM_FAMILY:
    rho, p, n = spear(sc.stage, sc[R[k]])
    raw[k] = p
    tr[k] = (rho, p, n)
hp = holm(raw)
for k in HOLM_FAMILY:
    rho, p, n = tr[k]
    put(f"trend_{k}", [r3(rho), pg(p), pg(hp[k]), n],
        f"Spearman trend of the {DESC[k]} against severity over all defined series flies (healthy included): [rho, raw p, Holm p within the 12-test family, n].")
others = [k for k in HOLM_FAMILY if k != "LInorm"]
worst = max(others, key=lambda k: hp[k])
put("trend_max_holm_p_eleven_readouts", [worst, pg(hp[worst])], f"Largest Holm-corrected p among the eleven circuit readouts (family of twelve incl. LI/MBON): [readout, Holm p]; every one of the eleven is at or below this ({worst} equals it).")
put("trend_abs_rho_range_eleven", [r3(min(abs(tr[k][0]) for k in others)), r3(max(abs(tr[k][0]) for k in others))], "Range of |Spearman rho| with severity over the eleven circuit readouts (series, all flies).")
put("trend_n_falling_of_eleven", int(sum(tr[k][0] < 0 for k in others)), "Number of the eleven circuit readouts whose Spearman rho with severity is negative (GF latency rises instead).")
rho, p, n = spear(sc.stage, sc.PAM_pair_hz)
put("trend_PAM", [r3(rho), pg(p), n], "Spearman trend of the PAM pairing rate against severity (series, all flies; not part of the 12-test Holm family): [rho, p, n]. "
    "The PAMs are driven at 100 Hz during pairing and their rate barely changes without neuron loss (threshold shift alone, synapse loss alone and "
    "KC-only pathology all ~100% of healthy, stim_PAM_pct_by_rule_*), so this trend mainly reflects the removal of driven PAMs, not a downstream circuit effect.")
twelfth = tr["LInorm"]
put("twelfth_test_LInorm", [r3(twelfth[0]), pg(twelfth[1]), pg(hp["LInorm"]), twelfth[2]], f"The twelfth Holm-family test: learning index / MBON rate vs severity over all {twelfth[2]} series flies where it is defined (MBON rate >= 0.5 Hz; healthy included): [rho, raw p, Holm p, n].")


def li_trend(lo, hi, incl_healthy=False):
    d = sc[((sc.stage >= lo) & (sc.stage <= hi)) | ((sc.stage == 0) & incl_healthy)]
    return spear(d.stage, d.LI_norm)


for name, (lo, hi, ih), txt in [
        ("le03_incl_healthy", (0.0, 0.3, True), "s = 0 to 0.3 with healthy flies"),
        ("s005_03", (0.05, 0.3, False), "affected flies s = 0.05 to 0.3"),
        ("s01_03", (0.1, 0.3, False), "affected flies s = 0.1 to 0.3, where the ratio is defined in every fly"),
        ("s01_06", (0.1, 0.6, False), "affected flies s = 0.1 to 0.6"),
        ("s01_10", (0.1, 1.0, False), "all defined affected flies at s >= 0.1")]:
    rho, p, n = li_trend(lo, hi, ih)
    put(f"LInorm_trend_{name}", [r3(rho), pg(p), n], f"Spearman of learning index / MBON rate against severity over {txt}: [rho, p, n defined].")
for name, (lo, hi) in [("s01_03", (0.1, 0.3)), ("s01_06", (0.1, 0.6))]:
    d = sc[(sc.stage >= lo) & (sc.stage <= hi)].LI_norm
    put(f"LInorm_pooled_{name}_vs_healthy", pct_ci(d, H.LI_norm, f"lin{name}") + [mw(d, H.LI_norm), int(d.notna().sum())],
        f"Learning index / MBON rate pooled over s = {lo:g} to {hi:g} as % of healthy: [est, lo, hi, Mann-Whitney p vs healthy, n defined].")
put("LInorm_kw_s01_06", kw(*[grp_series(s).LI_norm for s in STAGES if 0.1 <= s <= 0.6]),
    "Kruskal-Wallis p of learning index / MBON rate across the seven severity groups s = 0.1 to 0.6 (defined flies only): a test for any difference "
    "among the groups, not a trend test (the trend is LInorm_trend_s01_06).")
put("LInorm_pct_by_stage", [pct_ci(grp_series(s).LI_norm, H.LI_norm, f"linst{s}", check=False)[0] if grp_series(s).LI_norm.notna().any() else None for s in STAGES],
    "Learning index / MBON rate median at each severity (series_stages order) as % of healthy; s = 1.0 undefined.")
put("LInorm_ci_by_stage", [pct_ci(grp_series(s).LI_norm, H.LI_norm, f"linst{s}")[1:] if grp_series(s).LI_norm.notna().any() else None for s in STAGES],
    "95% bootstrap [lo, hi] of learning index / MBON rate as % of healthy at each severity (series_stages order).")
put("LInorm_n_defined_by_stage", [int(grp_series(s).LI_norm.notna().sum()) for s in STAGES], "Flies (of 32) with MBON rate >= 0.5 Hz, where the ratio is defined, at each severity (series_stages order).")
put("LInorm_mw_p_vs_healthy_by_stage", [mw(grp_series(s).LI_norm, H.LI_norm) for s in STAGES], "Uncorrected Mann-Whitney p of learning index / MBON rate against healthy at each severity (series_stages order).")
for s in (0.6, 0.75):
    g = grp_series(s)
    dfd = g.LI_norm.notna()
    put(f"LInorm_selection_{sk(s)}", [r3(g.MBON_hz_A[dfd].median()), r3(g.MBON_hz_A.median()), r1(g.LI[dfd].median()), r1(g.LI.median())],
        f"Survivor selection of the ratio at s = {s:g}: [median MBON rate (Hz) of flies with a defined ratio, of all 32 flies, median LI of defined flies, of all flies].")

# ================================================================ 3. motor failure split per stage
for k, fail in (("GF", lambda x: x < GF_FAIL_HZ), ("MN9", lambda x: x == 0)):
    c = R[k]
    nf, med, pct, au = [], [], [], []
    for s in ALLST:
        g = grp_series(s)[c]
        ok = g[~fail(g)]
        nf.append(int(fail(g).sum()))
        med.append(r1(ok.median()) if len(ok) else None)
        pct.append(r1(ok.median() / hmed[k] * 100) if len(ok) else None)
        au.append(r3(auc(H[c][~fail(H[c])], ok)) if s > 0 and len(ok) > 2 else None)
    crit = "GF rate < 100 Hz" if k == "GF" else "MN9 drive = 0 Hz"
    put(f"{k}_n_fail_by_stage", nf, f"Series flies counted as {k} failures ({crit}) at each stage (0, then series_stages); n = 64 healthy, 32 otherwise.")
    put(f"{k}_nonfail_median_hz_by_stage", med, f"Median {k} rate (Hz) among series flies that did not fail ({crit} excluded), at each stage (0, then series_stages).")
    put(f"{k}_nonfail_pct_by_stage", pct, f"Median {k} rate among non-failing flies as % of the healthy median (no healthy fly fails), at each stage (0, then series_stages).")
    put(f"{k}_nonfail_auc_by_stage", au, f"Direction-free AUC of non-failing flies against healthy for {k} rate at each stage (0 = None, then series_stages): graded decline without failures.")
put("GF_healthy_min_hz", r1(H.GF_loom_hz.min()), "Lowest GF looming rate (Hz) among the 64 healthy flies.")
g5 = grp_series(0.5).GF_loom_hz
put("GF_s0.5_nonfail_max_hz", r1(g5[g5 >= GF_FAIL_HZ].max()), "Highest GF rate (Hz) among the non-failing flies at s = 0.5; below every healthy fly if smaller than GF_healthy_min_hz.")
put("GF_s1_median_hz", r1(grp_series(1.0).GF_loom_hz.median()), "Median GF rate (Hz) of all 32 flies at s = 1.0, which sits between the failing and responding modes when half the flies fail.")

# ================================================================ 4. non-monotone medians
for k in SERIES_READS:
    c = R[k]
    meds = []
    for s in ALLST:
        g = grp_series(s)[c]
        meds.append(g.median() / hmed[k] * 100 if g.notna().any() else np.nan)
    rises, ties = [], []
    for i in range(1, len(ALLST)):
        a, b = meds[i - 1], meds[i]
        if np.isfinite(a) and np.isfinite(b):
            if b > a + 1e-9:
                rises.append([ALLST[i], r1(a), r1(b)])
            elif abs(b - a) <= 1e-9:
                ties.append(ALLST[i])
    put(f"nonmono_{k}_rises", rises, f"Steps of the series where the median {DESC[k]} (% of healthy) went UP from the previous stage: list of [stage, previous %, this %]; for GF latency an increase is a worsening.")
    if ties:
        put(f"nonmono_{k}_ties", ties, f"Stages at which the median {DESC[k]} (% of healthy) equalled the previous stage's.")
put("nonmono_readouts_with_rises", [k for k in HOLM_FAMILY + ["PAM"] if k != "GFlat" and F[f"nonmono_{k}_rises"]], "Readouts (expected to fall) whose series medians rose at least once between adjacent stages.")
lat = sc.dropna(subset=["GF_latency_ms"])
put("GFlat_median_ms_by_stage", [r1(lat[lat.stage == s].GF_latency_ms.median()) for s in ALLST], "Median GF first-spike latency (ms) at each stage (0, then series_stages).")
put("GFlat_frac_above3ms_by_stage", [r3((lat[lat.stage == s].GF_latency_ms > 3).mean()) for s in ALLST], "Share of flies with a defined GF latency above 3 ms at each stage (0, then series_stages).")
gl_n = [int((lat.stage == s).sum()) for s in ALLST]
gl_short = [s for s, n_ in zip(ALLST, gl_n) if n_ < len(grp_series(s))]
put("GFlat_n_defined_by_stage", gl_n, "Series flies with a defined GF latency (at least one GF spike) at each stage (0, then series_stages), of 64 healthy and 32 per severity; the denominators of GFlat_* keys.")
put("GFlat_n_2ms_by_stage", [int((lat[lat.stage == s].GF_latency_ms == 2).sum()) for s in ALLST],
    f"Flies with a 2-ms GF latency at each stage (0, then series_stages), out of the flies with a defined latency (GFlat_n_defined_by_stage; "
    f"fewer than all flies at s = {', '.join(f'{s:g}' for s in gl_short) or 'none'}); the loss of 2-ms responses drives the latency trend.")
put("GFlat_n_undefined", int(sc.GF_latency_ms.isna().sum()), "Series flies without a GF spike (latency undefined), excluded from the latency trend.")

# ================================================================ 5. replicate spread of the default model
rep_near_floor = {}
for s in (0.5, 1.0):
    groups = {"series": grp_series(s), "ablation_all": grp_cond("all", s), "maps_hub": grp_map("hub", s)}
    floor = []
    for k in REP_READS:
        c = R[k]
        meds = [float(g[c].median()) for g in groups.values()]
        pcts = [r1(m / hmed[k] * 100) for m in meds]
        put(f"rep_{sk(s)}_{k}_medians", [round(m, 4) for m in meds], f"Default model (hub, pan) at s = {s:g}: raw median {DESC[k]} in the three 32-fly groups [series, ablation 'all', map 'hub'], which have independent pathology draws but share stimulus-noise realisations by batch slot (see rep_shared_stimulus_slots).")
        put(f"rep_{sk(s)}_{k}_pct", pcts, f"Same three group medians as % of the healthy median [series, ablation 'all', map 'hub'], s = {s:g}.")
        mx, mn = max(meds), min(meds)
        put(f"rep_{sk(s)}_{k}_maxmin", r3(mx / mn) if mn > 0 else None, f"Replicate spread: max/min of the three default-model group medians of the {DESC[k]} at s = {s:g} (None if a median is 0).")
        put(f"rep_{sk(s)}_{k}_kw_p", kw(*[g[c] for g in groups.values()]), f"Kruskal-Wallis p across the three default-model groups for the {DESC[k]} at s = {s:g}.")
        if mn <= 0 or min(pcts) < FLOOR_PCT:
            floor.append(k)
    put(f"rep_{sk(s)}_near_floor_readouts", floor, f"Readouts whose lowest default-model group median at s = {s:g} is below {FLOOR_PCT:g}% of healthy (or 0); their max/min ratios are not precision estimates.")
    rep_near_floor[s] = floor
    put(f"rep_{sk(s)}_GF_fail_counts", [int((g.GF_loom_hz < GF_FAIL_HZ).sum()) for g in groups.values()],
        f"GF failures (< 100 Hz) in the three default-model groups at s = {s:g} [series, ablation 'all', map 'hub']; near 16/32 the median falls between modes.")
rv = {k: F[f"rep_s0.5_{k}_maxmin"] for k in REP_SEVEN}
put("rep_s0.5_max_ratio_seven_readouts", [max(rv, key=rv.get), max(rv.values())], "Largest replicate spread at s = 0.5 among the seven readouts tested in battery_stats.json (brain, LH, MBON, KC rate, LI, MN9, GF): [readout, max/min].")
ok_v = {k: F[f"rep_s0.5_{k}_maxmin"] for k in REP_READS if F[f"rep_s0.5_{k}_maxmin"] is not None}
put("rep_s0.5_max_ratio_all_readouts", [max(ok_v, key=ok_v.get), max(ok_v.values())], "Largest replicate spread at s = 0.5 over all eleven readouts examined here (incl. PN, KC recruitment, valence, PAM): [readout, max/min].")
pv = sorted((F[f"rep_{sk(s)}_{k}_kw_p"], k, s) for s in (0.5, 1.0) for k in REP_SEVEN)
put("rep_14tests_n_below_005", int(sum(p < 0.05 for p, _, _ in pv)), "Of the 14 replicate Kruskal-Wallis tests of battery_stats.json (7 readouts x 2 severities), the number with p < 0.05.")
put("rep_14tests_smallest", [[k, s, p] for p, k, s in pv[:2]], "The two smallest replicate Kruskal-Wallis p values of the 14: [readout, s, p].")
put("rep_wholebrain_criterion_1p27", r3(max(F["rep_s0.5_brain_maxmin"], F["rep_s1_brain_maxmin"])), "Largest whole-brain replicate spread over both severities (the '1.27' criterion; it is whole-brain specific).")
# Stimulus noise is shared across designs: every design ran assay_battery.py with the default --seed 7, so batch k
# (rows 48k..48k+47 of each *_x4.csv, written in batch order) uses stimulus seed 700 + 48k, and the fly in slot b of a
# full batch receives the same Poisson trains in every design. Check the row order against the design's permutation.
sc0 = pd.read_csv(BAT / "stage_curve_x4.csv")                       # unfiltered, in batch order
for nm_, d_, base_ in (("stage_curve", sc0, 1000), ("components", co, 2000), ("vulnerability", vu, 3000)):
    assert np.array_equal(d_.seed.to_numpy(), base_ + np.random.default_rng(7).permutation(len(d_))), nm_


def _slots(d, cond):
    """(batch, slot) of the group's flies: (in full-size batches, in a partial last batch)."""
    i = np.flatnonzero((d.condition == cond).to_numpy())
    full = (i // BATTERY_BATCH + 1) * BATTERY_BATCH <= len(d)
    return set(zip(i[full] // BATTERY_BATCH, i[full] % BATTERY_BATCH)), set(zip(i[~full] // BATTERY_BATCH, i[~full] % BATTERY_BATCH))


for s in (0.5, 1.0):
    (a_f, a_p), (b_f, b_p), (c_f, c_p) = _slots(sc0, f"s{s:g}"), _slots(co, f"all_{sk(s)}"), _slots(vu, f"hub_{sk(s)}")
    n_full = [len(a_f & b_f), len(a_f & c_f), len(b_f & c_f)]
    n_any = len((a_f | a_p) & (b_f | b_p)) + len((a_f | a_p) & (c_f | c_p)) + len((b_f | b_p) & (c_f | c_p))
    put(f"rep_shared_stimulus_slots_{sk(s)}", n_full + [n_any - sum(n_full)],
        f"s = {s:g}: pairs of flies in the three default-model groups that received identical stimulus trains because they sit in the same slot of the "
        f"same batch (all designs ran with stimulus seed 7): [series & ablation 'all', series & map 'hub', ablation 'all' & map 'hub', pairs where one "
        f"fly is in a partial batch (identical odour-A naive segment only)]. Conditions are permuted into slots, so this adds no bias, but the groups "
        f"are not fully independent in stimulus noise.")
# map contrasts against the replicate spread (direction-free: max(r, 1/r) vs max/min of the three default groups)
for s in (0.5, 1.0):
    for k in REP_READS:
        c = R[k]
        hub, shu, anti = (grp_map(m, s)[c] for m in MAPS)
        put(f"mapratio_{sk(s)}_{k}", [r3(med_ratio(shu, hub)), r3(med_ratio(anti, hub)), F[f"rep_{sk(s)}_{k}_maxmin"]],
            f"At s = {s:g}, {DESC[k]}: [shuffled/hub median ratio, antihub/hub median ratio, replicate spread max/min] (equal-neuron maps vs precision).")
    put(f"mapratio_{sk(s)}_brain_shuffled_over_hub_ci", ratio_ci(grp_map("shuffled", s).brain_spikes_A, grp_map("hub", s).brain_spikes_A, f"mrs{s}brain"),
        f"Whole-brain activity at s = {s:g}: shuffled/hub median ratio with 95% bootstrap interval [est, lo, hi].")
    put(f"mapratio_{sk(s)}_brain_antihub_over_hub_ci", ratio_ci(grp_map("antihub", s).brain_spikes_A, grp_map("hub", s).brain_spikes_A, f"mra{s}brain"),
        f"Whole-brain activity at s = {s:g}: antihub/hub median ratio with 95% bootstrap interval [est, lo, hi].")
fold, spread = {}, {}
for k in REP_READS:
    meds = [float(g[R[k]].median()) for g in (grp_series(0.5), grp_cond("all", 0.5), grp_map("hub", 0.5))]
    r_ = med_ratio(grp_map("shuffled", 0.5)[R[k]], grp_map("hub", 0.5)[R[k]])
    if r_ is not None and r_ > 0 and min(meds) > 0:
        fold[k], spread[k] = max(r_, 1 / r_), max(meds) / min(meds)
put("mapratio_s0.5_shuffled_hub_fold_vs_spread", {k: [round(fold[k], 4), round(spread[k], 4)] for k in fold},
    "At s = 0.5, per readout: [direction-free shuffled-vs-hub fold difference max(r, 1/r) with r = shuffled/hub median ratio, replicate spread max/min "
    "of the three default-model groups], unrounded to 4 decimals.")
put("mapratio_s0.5_readouts_shuffled_hub_within_spread", [k for k in fold if fold[k] <= spread[k]],
    "Readouts whose shuffled-vs-hub fold difference at s = 0.5, in either direction (max(r, 1/r)), does not exceed their own replicate spread "
    "(compared unrounded); see mapratio_s0.5_readouts_shuffled_hub_near_spread for the ties.")
put("mapratio_s0.5_readouts_shuffled_hub_near_spread", [[k, round(fold[k] / spread[k], 4)] for k in fold if abs(fold[k] / spread[k] - 1) <= 0.01],
    "Readouts whose shuffled-vs-hub fold difference at s = 0.5 lies within 1% of their replicate spread, [readout, fold / spread]: "
    "on the boundary, so neither 'within' nor 'beyond' the spread is a clear case.")
# second healthy group (exp2) against the battery healthy group
e2 = pd.read_csv(ROOT / "results/exp2/battery.csv")
e2h = e2[e2.condition == "healthy"]
put("healthy_groups_LI", [r1(H.LI.median()), r1(e2h.LI.median()), mw(H.LI, e2h.LI)], "Learning index median of the battery healthy group (n = 64) and the exp2 healthy group (results/exp2/battery.csv, n = 32), and Mann-Whitney p between them.")
put("healthy_groups_brain", [r1(H.brain_spikes_A.median()), r1(e2h.brain_spikes_A.median()), mw(H.brain_spikes_A, e2h.brain_spikes_A)], "Whole-brain spike median of the battery healthy group and the exp2 healthy group, and Mann-Whitney p between them.")

# ================================================================ 6. rule ablation
for s in (0.5, 1.0):
    for cnd in CONDS:
        g = grp_cond(cnd, s)
        put(f"abl_{sk(s)}_{cnd}_dead_pct", r3(100 * g.dead_frac.mean()), f"Rule ablation, {COND_DESC[cnd]}, s = {s:g}: mean % of neurons removed per fly.")
        for k in ABL_READS:
            c = R[k]
            put(f"abl_{sk(s)}_{cnd}_{k}", pct_ci(g[c], H[c], f"abl{s}{cnd}{k}") + [mw(g[c], H[c])],
                f"Rule ablation, {COND_DESC[cnd]}, s = {s:g} ({n_txt(g[c], k)}): median {DESC[k]} as % of healthy ({n_txt(H[c], k)}) [est, 95% lo, hi, Mann-Whitney p vs healthy].")
    for k in ABL_READS:
        c = R[k]
        put(f"abl_{sk(s)}_{k}_p_nohyper_vs_all", mw(grp_cond("no_hyper", s)[c], grp_cond("all", s)[c]),
            f"Mann-Whitney p, full model vs no threshold shift (different flies), {DESC[k]}, s = {s:g}.")
        put(f"abl_{sk(s)}_{k}_p_pairs", [mw(grp_cond("death_only", s)[c], grp_cond("synloss_only", s)[c]),
                                       mw(grp_cond("no_hyper", s)[c], grp_cond("synloss_only", s)[c]),
                                       mw(grp_cond("no_hyper", s)[c], grp_cond("death_only", s)[c])],
            f"Mann-Whitney p for {DESC[k]} at s = {s:g}: [neuron loss alone vs synapse loss alone, both losses vs synapse loss alone, both losses vs neuron loss alone].")
put("abl_LInorm_n_defined", {f"{cnd}_{sk(s)}": int(grp_cond(cnd, s).LI_norm.notna().sum()) for s in (0.5, 1.0) for cnd in CONDS},
    "Rule ablation: flies (of 32 per condition) with MBON rate >= 0.5 Hz, i.e. with a defined learning index / MBON rate; the abl_*_LInorm "
    "values are medians over these survivor-selected flies only.")
for k in ("MBON", "KC"):
    g = grp_cond("death_only", 0.5)[R[k]]
    put(f"abl_s0.5_death_only_{k}_outside_healthy_range", [int((g < H[R[k]].min()).sum()), int((g > H[R[k]].max()).sum())],
        f"Neuron loss alone, s = 0.5: flies (of 32) with {DESC[k]} below the lowest and above the highest healthy value.")
g = grp_cond("synloss_only", 0.5).MBON_hz_A
put("abl_s0.5_synloss_only_MBON_below_healthy_min", int((g < H.MBON_hz_A.min()).sum()), "Synapse loss alone, s = 0.5: flies (of 32) with MBON rate below the lowest healthy value.")
# between-group resolution of all vs no_hyper at s = 0.5
mdd = {}
for k in ("brain", "LH", "KC", "MBON", "MN9"):
    c = R[k]
    a = grp_cond("all", 0.5)[c] / hmed[k] * 100
    b = grp_cond("no_hyper", 0.5)[c] / hmed[k] * 100
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    mdd[k] = [r1(sd), r1((1.959964 + 0.841621) * sd * np.sqrt(2 / 32)), r1(a.median() - b.median())]
    put(f"abl_s0.5_resolution_{k}", mdd[k], f"Full model vs no threshold shift at s = 0.5, {DESC[k]} in % of healthy: [pooled per-fly SD, normal-approximation difference detectable with 80% power at alpha 0.05 with 32 vs 32 flies, observed difference of medians (all minus no_hyper)].")
# within-fly counterpart (exp1, copied with provenance)
e1 = json.loads((ROOT / "results/stats/exp1_stats.json").read_text())
bd = pd.read_csv(ROOT / "results/exp1/summary/baseline_decomposition.csv")
bd.columns = ["seed", "readout"] + list(bd.columns[2:])
e1b = pd.concat([pd.read_csv(f) for f in sorted((ROOT / "results/exp1/screen").glob("*_runs.csv"))])
e1b = e1b[e1b.level == "baseline"]
e1_stage = e1b.groupby("seed").stage.first()
for k in ("brain", "LH", "KC", "MBON", "MN9", "GF"):
    c = R[k]
    d = e1["decomposition"]["all"]["no_shift"][c]
    x = bd[bd.readout == f"f_{c}"].set_index("seed").no_shift.dropna().sort_index()
    p = e1b.pivot_table(index="seed", columns="unit", values=c)
    dfc = p.healthy - p.damaged
    ok = dfc.abs() > EXP1_MIN_DEFICIT * np.maximum(p.healthy.abs(), 1e-9)            # exp1 criterion, strict
    fr = ((p.no_shift - p.damaged) / dfc)[ok].sort_index()
    assert np.array_equal(fr.index, x.index) and np.allclose(fr, x), c                 # same flies and fractions as analyze_exp1.py
    assert d["n"] == int(ok.sum())
    put(f"exp1_noshift_{k}", [d["median"], d["q25"], d["q75"], int((p.no_shift < p.damaged)[ok].sum()), int(ok.sum()), int((fr < 0).sum()),
                              int((dfc < 0)[ok].sum())],
        f"Within-fly (exp1, {len(p)} flies: {int((e1_stage == 0.3).sum())} at s = 0.3, {int((e1_stage == 0.5).sum())} at s = 0.5): removing the threshold "
        f"shift, change of the {DESC[k]} as a fraction of the fly's own deficit, (y_no_shift - y_damaged) / (y_healthy - y_damaged) "
        f"[median, q25, q75 from results/stats/exp1_stats.json decomposition.all.no_shift; flies whose readout was lowered (y_no_shift < y_damaged), "
        f"flies with |deficit| > 5% of |healthy| (the flies summarised), flies moved further from healthy (fraction < 0), flies whose damaged value "
        f"is above healthy (negative deficit, where a lowered readout gives a fraction > 0); counts from the baseline rows of results/exp1/screen/*_runs.csv].")

# ================================================================ 7. Kenyon cell rates above the healthy range
hmax, hkm = float(H.KC_hz_A.max()), float(H.KC_hz_A.median())
for s in (0.5, 1.0):
    for cnd in CONDS:
        g = grp_cond(cnd, s)
        kc = g.KC_hz_A
        put(f"kchigh_{sk(s)}_{cnd}", [int((kc > hmax).sum()), int((kc > 1.2 * hmax).sum()), int((kc > 1.5 * hkm).sum())],
            f"{COND_DESC[cnd]}, s = {s:g} (32 flies): flies with Kenyon cell rate [above the healthy maximum, above 1.2 x the healthy maximum, above 1.5 x the healthy median].")
        put(f"kcmax_{sk(s)}_{cnd}", [r3(kc.max()), r3(kc.max() / hmax), r1(kc.max() / hkm * 100), r1(kc.median() / hkm * 100)],
            f"{COND_DESC[cnd]}, s = {s:g}: [highest Kenyon cell rate (Hz), as multiple of the healthy maximum, as % of the healthy median, group median as % of the healthy median].")
        hi = (g.KC_hz_A > 1.5 * g.KC_hz_A.median()) & (g.MBON_hz_A > 1.5 * g.MBON_hz_A.median())
        put(f"kchighact_owngroup_{sk(s)}_{cnd}", int(hi.sum()),
            f"{COND_DESC[cnd]}, s = {s:g}: flies meeting the Methods high-activity criterion (Kenyon cell AND MBON rate both > 1.5 x the median of the fly's own group).")
g = grp_cond("hyper_only", 0.5).KC_hz_A
put("kc_hyper_only_s0.5_below_healthy_median", [int((g < hkm).sum()), r1(g.min() / hkm * 100)], "Threshold shift alone, s = 0.5: flies (of 32) with Kenyon cell rate below the healthy median, and the lowest fly as % of the healthy median.")
d9 = grp_cond("death_only", 0.5)
a_ = set(d9.seed[d9.KC_hz_A > hmax])
b_ = set(d9.seed[(d9.KC_hz_A > 1.5 * d9.KC_hz_A.median()) & (d9.MBON_hz_A > 1.5 * d9.MBON_hz_A.median())])
put("kc_death_only_s0.5_overlap_criteria", [len(a_), len(b_), len(a_ & b_)], "Neuron loss alone, s = 0.5: [flies above the healthy KC maximum, flies meeting the own-group high-activity criterion, flies in both] (same seeds if all equal).")

# ================================================================ 8. Kenyon-cell-only vs pan-neuronal at matched Kenyon cell loss
for s in (0.5, 1.0):
    mb, pan = grp_mb(s), grp_series(s)
    put(f"kconly_{sk(s)}_kc_dead_pct", [r3(100 * mb.kc_dead_frac.mean()), r3(100 * pan.kc_dead_frac.mean())], f"s = {s:g}: mean % of Kenyon cells removed [KC-only, pan-neuronal series].")
    put(f"kconly_{sk(s)}_dead_pct", [r3(100 * mb.dead_frac.mean()), r3(100 * pan.dead_frac.mean())], f"s = {s:g}: mean % of all neurons removed [KC-only, pan-neuronal series].")
    for k in SERIES_READS:
        c = R[k]
        put(f"kconly_{sk(s)}_{k}", pct_ci(mb[c], H[c], f"mb{s}{k}") + [mw(mb[c], H[c]), mw(mb[c], pan[c])],
            f"Pathology confined to Kenyon cells, s = {s:g} ({n_txt(mb[c], k)}): median {DESC[k]} as % of healthy ({n_txt(H[c], k)}) [est, 95% lo, hi, Mann-Whitney p vs healthy, p vs pan-neuronal series at the same s ({n_txt(pan[c], k)})]; the pan value is series_{sk(s)}_{k}_pct.")
    for k in ("LI", "MBON", "KCrec", "KC"):
        a = F[f"kconly_{sk(s)}_{k}"][0]
        b = F[f"series_{sk(s)}_{k}_pct"][0]
        put(f"kconly_{sk(s)}_{k}_deficit_share", [r1(100 - a), r1(100 - b), r3((100 - a) / (100 - b))],
            f"s = {s:g}, {DESC[k]}: [points of healthy lost under KC-only, under pan-neuronal, ratio] (share of the pan deficit reproduced by KC-only damage, assuming additivity).")
    surv = mb.KC_frac_A / (1 - mb.kc_dead_frac)
    put(f"kconly_{sk(s)}_KCrec_survivors_pct", pct_ci(surv, H.KC_frac_A / (1 - H.kc_dead_frac), f"mbsurv{s}"),
        f"KC-only, s = {s:g}: Kenyon cell recruitment among surviving Kenyon cells, median as % of healthy [est, lo, hi].")

# ================================================================ 9. vulnerability maps
for s in (0.5, 1.0):
    gs = {m: grp_map(m, s) for m in MAPS}
    put(f"map_{sk(s)}_dead_pct", [r3(100 * gs[m].dead_frac.mean()) for m in MAPS], f"Mean % of neurons removed per fly at s = {s:g} [hub, shuffled, antihub].")
    put(f"map_{sk(s)}_kc_dead_pct", [r3(100 * gs[m].kc_dead_frac.mean()) for m in MAPS] + [kw(*[gs[m].kc_dead_frac for m in MAPS])],
        f"Mean % of Kenyon cells removed per fly at s = {s:g} [hub, shuffled, antihub, Kruskal-Wallis p].")
    for k in MAP_READS:
        c = R[k]
        for m in MAPS:
            put(f"map_{sk(s)}_{m}_{k}", pct_ci(gs[m][c], H[c], f"map{s}{m}{k}"),
                f"{m} map, s = {s:g} ({n_txt(gs[m][c], k)}): median {DESC[k]} as % of healthy [est, 95% lo, hi].")
        put(f"map_{sk(s)}_{k}_p", [kw(*[gs[m][c] for m in MAPS]), mw(gs["hub"][c], gs["shuffled"][c]), mw(gs["hub"][c], gs["antihub"][c])],
            f"Map comparison at s = {s:g}, {DESC[k]}: [Kruskal-Wallis p across maps, Mann-Whitney hub vs shuffled, hub vs antihub].")
        put(f"map_{sk(s)}_antihub_{k}_p_vs_healthy", mw(gs["antihub"][c], H[c]), f"Mann-Whitney p of the antihub map against healthy at s = {s:g}, {DESC[k]}.")
    put(f"map_{sk(s)}_GF_fail", [int((gs[m].GF_loom_hz < GF_FAIL_HZ).sum()) for m in MAPS], f"GF failures (< 100 Hz) of 32 flies at s = {s:g} [hub, shuffled, antihub]; healthy 0 of 64.")
    put(f"map_{sk(s)}_MN9_silent", [int((gs[m].MN9_sugar_hz == 0).sum()) for m in MAPS], f"Flies with MN9 drive 0 (of 32) at s = {s:g} [hub, shuffled, antihub]; healthy 0 of 64.")
    gfok = [gs[m].GF_loom_hz[gs[m].GF_loom_hz >= GF_FAIL_HZ] for m in MAPS]
    mnok = [gs[m].MN9_sugar_hz[gs[m].MN9_sugar_hz > 0] for m in MAPS]
    put(f"map_{sk(s)}_GF_nonfail_pct", [r1(x.median() / hmed["GF"] * 100) for x in gfok] + [kw(*gfok)],
        f"Median GF rate among non-failing flies as % of healthy at s = {s:g} [hub, shuffled, antihub, Kruskal-Wallis p]: the map effect without the failures.")
    put(f"map_{sk(s)}_MN9_nonsilent_pct", [r1(x.median() / hmed["MN9"] * 100) for x in mnok] + [kw(*mnok)],
        f"Median MN9 drive among non-silent flies as % of healthy at s = {s:g} [hub, shuffled, antihub, Kruskal-Wallis p].")

# ================================================================ 10. stimulated non-sensory populations
t = neu.type.fillna("")
sens = neu.superclass.fillna("").str.contains("sensory").to_numpy()     # model.py: eligible = ~sensory
pops = {"odourA_uPN": t.str.match(r"^(DM1|DM4|VA2)_").to_numpy(),     # model.ODORS["A"]; class ALPN
        "odourB_uPN": t.str.match(r"^(DL5|DM2|VM2)_").to_numpy(),
        "PAM": t.str.startswith("PAM").to_numpy(),
        "loom_left": (t.isin(["LPLC2", "LC4"]) & (neu.side == "L")).to_numpy(),
        "sugar_GRN": t.isin(["LB3c", "LB3b"]).to_numpy()}
pdesc = {"odourA_uPN": "uniglomerular PNs driven by odour A (types DM1_*, DM4_*, VA2_*; 150 Hz Poisson)",
         "odourB_uPN": "uniglomerular PNs driven by odour B (DL5_*, DM2_*, VM2_*; 150 Hz)",
         "PAM": "PAM dopaminergic neurons driven during pairing (100 Hz)",
         "loom_left": "left LPLC2 + LC4 neurons driven by looming (150 Hz)",
         "sugar_GRN": "sugar GRNs LB3c + LB3b, both sides (150 Hz)"}
for p_, m in pops.items():
    put(f"stim_{p_}_n", int(m.sum()), f"Number of {pdesc[p_]} in the MaleCNS neuron table (results/validation/baseline_nostim.parquet).")
    put(f"stim_{p_}_superclasses", sorted(neu.superclass[m].unique().tolist()), f"Superclass(es) of the {pdesc[p_]}.")
    put(f"stim_{p_}_n_eligible_for_load", int((m & ~sens).sum()), f"Of the {pdesc[p_]}, how many can carry pathology load and be removed (model.py: only superclasses containing 'sensory' are exempt).")
put("stim_odourA_uPN_types", sorted(t[pops["odourA_uPN"]].unique().tolist()), "Cell types of the odour-A driven uniglomerular PNs.")
put("stim_PAM_n_types", int(t[pops["PAM"]].nunique()), "Number of PAM cell types; PAM_pair_hz is the mean over these types of each type's mean pairing rate.")
put("n_neurons_sensory_exempt", int(sens.sum()), "Neurons exempt from pathology (superclass contains 'sensory'), of 166,700.")
put("n_neurons_eligible", int((~sens).sum()), "Neurons that can carry pathology load (all non-sensory superclasses).")
# hub vulnerability rank u of every neuron, checked against the exp1 screen before use
units = pd.concat([pd.read_csv(f) for f in sorted((ROOT / "results/exp1/screen").glob("*_units.csv"))])
ut = units[units.level == "type"]
uz = np.load(ROOT / "results/validation/hub_vulnerability_u.npz")
assert np.array_equal(uz["ids"], neu.bodyId.to_numpy()), "u table is not in neuron-table order"
U = np.asarray(uz["u"], float)
assert np.allclose(np.sort(U), np.arange(1, U.size + 1) / U.size), "u is not a percentile rank"
tsize = t[t != ""].value_counts()
u_by_type = {k: np.asarray(v, float) for k, v in pd.Series(U[(t != "").to_numpy()]).groupby(t[t != ""].to_numpy())}
typed = ut[ut.unit.isin(tsize.index)]
full_rows = typed[typed.n_lost == typed.unit.map(tsize)]
err_full = float((full_rows.mean_rank - full_rows.unit.map(lambda k: u_by_type[k].mean())).abs().max())
one_rows = typed[typed.n_lost == 1]
err_one = float(max(np.abs(u_by_type[r.unit] - r.mean_rank).min() for r in one_rows.itertuples()))
onl = json.loads((ROOT / "results/paper/output_neuron_loss.json").read_text())["vulnerability_percentile"]
err_gf = max(abs(float(U[((t == "DNp01") & (neu.side == sd)).to_numpy()][0]) - onl[f"DNp01_{sd}"]) for sd in ("L", "R"))
assert max(err_full, err_one, err_gf) < 1e-9, (err_full, err_one, err_gf)
put("u_table_check", [int(len(full_rows)), int(full_rows.unit.nunique()), int(len(one_rows)), float(f"{max(err_full, err_one, err_gf):.1e}")],
    "Check of results/validation/hub_vulnerability_u.npz (hub rank u of every neuron, as pathology.hub_vulnerability) against independent outputs: "
    "[exp1 screen type rows in which every neuron of the type was lost (mean_rank must equal the type's mean u), distinct types among them, rows with "
    "one lost neuron (mean_rank must equal one of the type's u), largest absolute mismatch incl. the DNp01 ranks of results/paper/output_neuron_loss.json].")

# vulnerability ranks of the driven populations from the full u table
upop = {"odourA_uPN": pops["odourA_uPN"], "odourB_uPN": pops["odourB_uPN"], "PAM": pops["PAM"],
        "LPLC2": ((t == "LPLC2") & (neu.side == "L")).to_numpy(), "LC4": ((t == "LC4") & (neu.side == "L")).to_numpy()}
utype_avg = {}
for p_, m in upop.items():
    um = U[m]
    utype_avg[p_] = float(pd.Series(um).groupby(t[m].to_numpy()).mean().mean())
    put(f"stim_u_{p_}", [r3(um.mean()), [r3(um.min()), r3(um.max())], int(m.sum()), r3(utype_avg[p_]), r3((um ** 2).mean() / um.mean())],
        f"Hub vulnerability rank u (percentile rank of total synapse count, in (0, 1]) of the driven {p_}{' (left side, the looming-driven ones)' if p_ in ('LPLC2', 'LC4') else ''}, "
        f"from the full u table: [mean over neurons, [min, max], n neurons, mean over cell types of the type-mean u, mean u of the neurons expected to be "
        f"lost (E[u^2]/E[u], what a mean over lost neurons estimates)].")
put("stim_u_odourA_uPN_by_neuron", {f"{ty}_{sd}": round(float(uu), 4) for ty, sd, uu in
                                    sorted(zip(t[pops["odourA_uPN"]], neu.side[pops["odourA_uPN"]], U[pops["odourA_uPN"]]))},
    "Hub rank u of each odour-A driven uPN (type_side). DM4_vPN, absent from the exp1 screen, is far less vulnerable than the other three types.")
uA = float(U[pops["odourA_uPN"]].mean())

# the same from the exp1 screen, which is a capped, selected subset of each fly's lost types
def rank_of(types):
    r = ut[ut.unit.isin(types)]
    if r.empty:
        return [None, None, 0, 0]
    w = np.average(r.mean_rank, weights=r.n_lost)
    return [r3(w), [r3(r.mean_rank.min()), r3(r.mean_rank.max())], int(r.seed.nunique()), int(r.n_lost.sum())]


n_type_rows = units[units.level == "type"].groupby("seed").size()
n_chosen, n_ctrl = ut.groupby("seed").chosen.sum(), ut.groupby("seed").control.sum()
alpn_cls, alpn_typ = int(units[(units.level == "class") & (units.unit == "ALPN")].n_lost.sum()), int(ut[ut["class"] == "ALPN"].n_lost.sum())
for p_, types in {"odourA_uPN": sorted(t[pops["odourA_uPN"]].unique()), "odourB_uPN": sorted(t[pops["odourB_uPN"]].unique()),
                  "PAM": sorted(t[pops["PAM"]].unique()), "LPLC2": ["LPLC2"], "LC4": ["LC4"]}.items():
    put(f"stim_rank_{p_}", rank_of(types),
        f"Screen-based hub rank of {p_} from results/exp1/screen/*_units.csv type rows (mean_rank = mean u of the type's lost neurons in that fly): "
        f"[n_lost-weighted mean, [min, max] over fly x type rows, exp1 flies with at least one screen type row of these types, neurons lost within those rows]. "
        f"Elements [2] and [3] are NOT counts of flies with a loss or of neurons lost: the screen keeps only {int(n_type_rows.min())} type rows per fly "
        f"({int(n_chosen.max())} types inside the fly's chosen classes, largest n_lost first, plus {int(n_ctrl.max())} random control types), e.g. the ALPN class lost {alpn_cls} neurons over "
        f"the {units.seed.nunique()} exp1 flies but only {alpn_typ} of them appear in type rows. A mean over lost neurons overweights high u, and the "
        f"rows are a selected subset, so element [0] is biased upward for heterogeneous types (PAM); LPLC2/LC4 rows pool both sides. Use stim_u_* instead.")
put("stim_odourA_types_screened", sorted(ut[ut.unit.isin(t[pops["odourA_uPN"]].unique())].unit.unique().tolist()),
    "Odour-A uPN types that appear in the exp1 screen type rows (DM4_vPN never does; its u is in stim_u_odourA_uPN_by_neuron).")
e1s = units.groupby("seed").stage.first()
pA = DEATH_MAX * np.outer(e1s.to_numpy(), U[pops["odourA_uPN"]])                 # per exp1 fly x driven uPN
put("stim_odourA_uPN_expected_loss_exp1", [r1(pA.sum()), r1((1 - np.prod(1 - pA, 1)).sum()), int(len(e1s)), F["stim_rank_odourA_uPN"][2], F["stim_rank_odourA_uPN"][3]],
    "Odour-A driven uPNs in the exp1 flies, expected from their u (P(removal) = 0.5 s u): [expected neurons lost summed over flies, expected flies "
    "with at least one lost, exp1 flies, flies with an odour-A uPN type row in the screen, neurons in those rows (stim_rank_odourA_uPN)], "
    "showing how few of the losses the capped screen lists.")
for s in (0.5, 1.0):
    put(f"stim_premoval_{sk(s)}", {**{p_: r3(DEATH_MAX * s * U[m].mean()) for p_, m in upop.items()}, "PAM_type_avg": r3(DEATH_MAX * s * utype_avg["PAM"])},
        f"Expected fraction of each driven population removed at s = {s:g} under the hub map, 0.5 x s x mean u (pathology.apply, death_max 0.5; "
        f"u from stim_u_*; LPLC2 and LC4 = left side). PAM_type_avg averages over the 15 PAM types first, as PAM_pair_hz does.")
# PAM pairing rate under neuron loss alone against the expected surviving fraction of driven PAMs
for k_ in ("hyper_only", "synloss_only"):
    for s in (0.5, 1.0):
        assert abs(F[f"abl_{sk(s)}_{k_}_PAM"][0] - 100) <= 1.5, (k_, s)               # trend_PAM meaning relies on this
for s in (0.5, 1.0):
    assert abs(F[f"kconly_{sk(s)}_PAM"][0] - 100) <= 1.5
put("stim_PAM_removal_vs_expected", [[s, r3(DEATH_MAX * s * utype_avg["PAM"]), r3(1 - F[f"abl_{sk(s)}_death_only_PAM"][0] / 100),
                                      r3(1 - grp_cond("death_only", s).PAM_pair_hz.mean() / H.PAM_pair_hz.mean())] for s in (0.5, 1.0)],
    "Per s: [s, expected type-averaged fraction of driven PAMs removed (0.5 s x mean type-mean u), 1 - median PAM rate under neuron loss alone as a "
    "fraction of healthy, 1 - mean ratio]. Surviving PAMs fire at about the healthy rate (threshold shift alone and synapse loss alone ~100%), "
    "so the PAM readout drop is close to the share of driven PAMs removed.")
# PN and PAM per rule
order = [("series", lambda s: grp_series(s))] + [(c, (lambda c: lambda s: grp_cond(c, s))(c)) for c in CONDS] + \
        [("kc_only", grp_mb)] + [(m, (lambda m: lambda s: grp_map(m, s))(m)) for m in MAPS]
for k in ("PN", "PAM"):
    for s in (0.5, 1.0):
        put(f"stim_{k}_pct_by_rule_{sk(s)}", [F[f"series_{sk(s)}_{k}_pct"][0]] + [F[f"abl_{sk(s)}_{c}_{k}"][0] for c in CONDS] +
            [F[f"kconly_{sk(s)}_{k}"][0]] + [F[f"map_{sk(s)}_{m}_{k}"][0] for m in MAPS],
            f"{DESC[k]} at s = {s:g}, median % of healthy, in order [series, all, threshold only, synapse loss only, neuron loss only, no threshold shift, KC-only, hub map, shuffled map, antihub map].")
# PN readout spikes vs Poisson events delivered to the driven PNs
allb = pd.concat([sc, co, vu, sp])
n_alpn = next(n for n in range(1, 5000) if np.all(np.abs(allb.PN_hz_A * n - np.round(allb.PN_hz_A * n)) < 1e-6))
nA = int(pops["odourA_uPN"].sum())
ev = nA * STIM_HZ["odour"] * 1.0
tot = H.PN_hz_A * n_alpn
put("pn_n_ALPN", int(n_alpn), "Number of class-ALPN neurons averaged in PN_hz_A, recovered as the smallest N for which PN_hz_A x N is an integer spike count in all 960 battery flies.")
put("pn_expected_events_odourA", ev, "Expected Poisson input events delivered to the odour-A driven uPNs in the 1-s odour-A window: n driven uPNs x 150 Hz x 1 s.")
put("pn_healthy_total_spikes", [r1(tot.median()), r1(tot.min()), r1(tot.max()), r1(tot.mean()), r1(tot.std(ddof=1))], "Total ALPN spikes during odour A in healthy flies (PN_hz_A x 686): [median, min, max, mean, SD].")
put("pn_healthy_spikes_over_events", [r3(tot.median() / ev), r3(tot.max() / ev)], "Healthy total ALPN spikes / expected events to the 8 driven uPNs: [median fly, highest fly]; at or below 1 means the readout has no room beyond the driven PNs' own spikes.")
put("pn_healthy_n_above_events", int((tot > ev).sum()), "Healthy flies whose total ALPN spike count exceeds the expected number of input events to the driven uPNs.")
put("pn_poisson_sd_events", r1(np.sqrt(ev)), "Poisson SD of the number of input events delivered to the driven uPNs in 1 s (compare the healthy spike-count SD).")
per_pn = tot.mean() / nA
put("pn_spikes_per_driven_uPN", [r1(per_pn), r1(per_pn / STIM_HZ["odour"] * 100)], "Healthy mean total ALPN spikes divided by the 8 driven uPNs (spikes per driven uPN in 1 s if the undriven ALPNs were silent), and as % of the 150 input events each receives.")
for nm, grp in (("series", grp_series), ("death_only", lambda s: grp_cond("death_only", s)), ("synloss_only", lambda s: grp_cond("synloss_only", s)),
                ("hyper_only", lambda s: grp_cond("hyper_only", s))):
    put(f"pn_mean_ratio_vs_expected_{nm}", [[s, r3(grp(s).PN_hz_A.mean() / H.PN_hz_A.mean()), r3(1 - DEATH_MAX * s * uA if nm in ("series", "death_only") else 1.0)] for s in (0.5, 1.0)],
        f"{nm}: [s, mean PN rate / healthy mean, expectation if PN rate = expected surviving fraction of the 8 driven uPNs (1 - 0.5 s mean u, "
        f"mean u = {uA:.4f} over the 8 neurons incl. the two DM4_vPN, stim_u_odourA_uPN) or 1 if no neurons are removed].")
# share of the readout not proportional to surviving driven uPNs: fit mean total = A x f + b over healthy,
# neuron-loss-only s = 0.5 and s = 1.0, with f = expected surviving fraction of driven uPNs (1 - 0.5 s u)
fit_groups = [(H, 1.0), (grp_cond("death_only", 0.5), 1 - DEATH_MAX * 0.5 * uA), (grp_cond("death_only", 1.0), 1 - DEATH_MAX * 1.0 * uA)]


def _fit(groups):
    f = np.array([fx for _, fx in groups])
    y = np.array([g for g, _ in groups])
    A, b = np.polyfit(f, y, 1)
    return b / y[0]


tots = [(clean(g.PN_hz_A) * n_alpn, fx) for g, fx in fit_groups]
est = _fit([(t_.mean(), fx) for t_, fx in tots])
rng = rng_for("pn_nondriven_fit")
bs = [_fit([(rng.choice(t_, len(t_)).mean(), fx) for t_, fx in tots]) for _ in range(N_BOOT)]
put("pn_nondriven_share_fit", [r3(est), r3(np.percentile(bs, 2.5)), r3(np.percentile(bs, 97.5))],
    "Share of the healthy ALPN spike total not proportional to surviving driven uPNs: intercept / healthy mean of a line fitted to mean total ALPN spikes against the expected surviving fraction of the 8 driven uPNs, 1 - 0.5 s mean u with their actual u (healthy, neuron loss alone s = 0.5 and 1.0) [est, 95% bootstrap over flies]; ignores sampling error in which uPNs were removed.")
# within-severity correlations: driven input vs downstream readouts
wc = {"PN_KC": [], "PN_MBON": [], "PAM_LI": []}
for s in STAGES:
    g = grp_series(s)
    wc["PN_KC"].append(spear(g.PN_hz_A, g.KC_hz_A)[0])
    wc["PN_MBON"].append(spear(g.PN_hz_A, g.MBON_hz_A)[0])
    wc["PAM_LI"].append(spear(g.PAM_pair_hz, g.LI)[0])
for nm, v in wc.items():
    a, b = nm.split("_")
    put(f"within_stage_rho_{nm}", [r3(np.median(v)), r3(min(v)), r3(max(v))], f"Spearman rho between {a} and {b} within each of the 10 severity groups of the series: [median, min, max] over groups.")
g = grp_series(0.5)
put("within_s0.5_rho_PN_vs", {k: r3(spear(g.PN_hz_A, g[R[k]])[0]) for k in ("brain", "LH", "KC", "MBON", "LI", "MN9", "GF")},
    "Within the 32 series flies at s = 0.5, Spearman rho of PN rate (delivered odour input) with each readout.")

# ================================================================ 11. learning index vs MBON rate within severity groups
put("LI_MBON_rho_pooled", r3(spear(sc.LI, sc.MBON_hz_A)[0]), "Spearman rho of learning index with MBON rate over all 384 series flies (mostly the shared decline with severity).")
put("LI_MBON_rho_by_stage", [r3(spear(grp_series(s).LI, grp_series(s).MBON_hz_A)[0]) for s in ALLST], "Spearman rho of learning index with MBON rate within each stage (0, then series_stages).")
put("LI_MBON_p_by_stage", [pg(spear(grp_series(s).LI, grp_series(s).MBON_hz_A)[1]) for s in ALLST], "p of the within-stage Spearman rho of learning index with MBON rate (0, then series_stages).")
lo3, hi4 = sc[sc.stage <= 0.3], sc[sc.stage >= 0.4]
put("LI_MBON_partial_given_stage", [r3(partial_spearman(sc.LI, sc.MBON_hz_A, sc.stage)[0]), r3(partial_spearman(lo3.LI, lo3.MBON_hz_A, lo3.stage)[0]),
                                    pg(partial_spearman(lo3.LI, lo3.MBON_hz_A, lo3.stage)[1]), r3(partial_spearman(hi4.LI, hi4.MBON_hz_A, hi4.stage)[0])],
    f"Partial rank correlation of learning index with MBON rate controlling for severity rank: [all series flies (n = {len(sc)}), s <= 0.3 including "
    f"the {len(H)} healthy flies (n = {len(lo3)}), its p, s >= 0.4 (n = {len(hi4)})].")
_r, _p = partial_spearman(sc.stage, sc.LI, sc.MBON_hz_A)
put("LI_stage_partial_given_MBON", [r3(_r), pg(_p)],"Partial rank correlation of severity with learning index controlling for MBON rate over all series flies: [r, p].")

# ================================================================ 12. cross-checks against battery_stats.json
chk = []
for s in ("0.5", "1.0"):
    for c in bstats["replicates"][s]:
        if c == "n_pooled":
            continue
        k = next(k for k, v in R.items() if v == c)
        chk.append(abs(bstats["replicates"][s][c]["kruskal_p"] - F[f"rep_{sk(float(s))}_{k}_kw_p"]) / max(bstats["replicates"][s][c]["kruskal_p"], 1e-300))
chk.append(abs(bstats["series"]["trend"]["LI_norm"]["p_holm"] - F["twelfth_test_LInorm"][2]) / bstats["series"]["trend"]["LI_norm"]["p_holm"])
nk = [bstats["components"][f"{c}_{sk(s)}"]["n_kc_high"] == F[f"kchigh_{sk(s)}_{c}"][0] for c in CONDS for s in (0.5, 1.0)]
pc = [abs(bstats["vulnerability"][str(s)][m][R[k]]["pct"] - F[f"map_{sk(s)}_{m}_{k}"][0]) for s in (0.5, 1.0) for m in MAPS
      for k in ("brain", "LH", "MBON", "KCrec", "LI", "MN9", "GF")]
put("check_vs_battery_stats", [r3(max(chk)), bool(all(nk)), r3(max(pc))], "Agreement with results/stats/battery_stats.json: [max relative difference of replicate KW p and twelfth-test Holm p, all n_kc_high counts equal, max abs difference of map % of healthy point estimates].")

# ================================================================ Monte Carlo stability of the bootstrap bounds
assert not _pending, "interval computed outside a put()"
put("boot_unstable_bounds", dict(UNSTABLE),
    f"Keys with a reported {N_BOOT}-resample bootstrap bound that differs by more than {BOOT_TOL['pct']:g} percentage point "
    f"({BOOT_TOL['ratio']:g} for ratios) from the same interval at {10 * N_BOOT} resamples: Monte Carlo instability, not a coding error, "
    f"usually where the per-fly values have a gap that the 2.5th or 97.5th percentile jumps across. Per interval: label, reported lo/hi, range "
    f"of lo and of hi over the reported seed and {BOOT_ALT_SEEDS} further seeds of {N_BOOT} resamples, interval at {10 * N_BOOT} resamples. "
    f"Quote such a bound from the {10 * N_BOOT}-resample value or flag it as unstable.")

# ================================================================ write
missing = set(F) - set(M)
assert not missing, missing
F["_meaning"] = M
F["_provenance"] = {"script": "code/scripts/facts_battery.py",
                    "inputs": ["results/battery/stage_curve_x4.csv", "results/battery/components_x4.csv",
                               "results/battery/vulnerability_x4.csv", "results/battery/scope_x4.csv",
                               "results/validation/baseline_nostim.parquet",
                               "results/validation/hub_vulnerability_u.npz (checked against exp1 units and output_neuron_loss.json)",
                               "results/exp1/screen/*_units.csv", "results/exp1/screen/*_runs.csv (baseline rows)",
                               "results/exp1/summary/baseline_decomposition.csv", "results/stats/exp1_stats.json",
                               "results/paper/output_neuron_loss.json (u check)",
                               "results/exp2/battery.csv", "results/stats/battery_stats.json (cross-check only)"],
                    "bootstrap": f"{N_BOOT} resamples, numpy default_rng([{SEED}, crc32(key)]), ratio of medians, flies resampled within each group; "
                                 f"bounds checked with {BOOT_ALT_SEEDS} further seeds (boot_unstable_bounds)",
                    "healthy_reference": "stage_curve_x4.csv stage 0, scope pan, n = 64 (shared by all battery designs)"}
(OUT / "facts_battery.json").write_text(json.dumps(F, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print(f"wrote {OUT / 'facts_battery.json'}: {len(M)} facts")
