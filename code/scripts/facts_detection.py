"""Facts table 'detection': learning, discrete phenotypes,
early detection and the virtual arena.

No simulation is run. Every number is computed from the per-fly result tables or
copied, with its provenance, from result JSON files and code:

  results/battery/*_x4.csv           severity series (stage_curve, pan scope), rule
                                     ablation (components), maps (vulnerability),
                                     Kenyon-cell-only scope
  results/battery/*.csv (no _x4)     earlier 240-fly run (the pilot); used only for
                                     test-retest reliability (same seed, condition)
  results/joint/cohort_joint.csv     176-fly joint cohort (circuit + arena)
  results/joint/stats_joint.json     analyze_joint.py output (first-detection bootstrap)
  results/stats/*                    analyze_battery.py output (battery_stats.json and csv tables)
  results/paper/output_neuron_loss.json   output_neuron_loss.py reconstruction
  results/exp1/screen/*              counterfactual screen (APL restoration in one fly)
  results/joint/smoke*.{csv,json}  4-fly joint smoke test (listed among unreported cohorts)
  code/scripts/cohort_joint.py, code/adfly/house*.py, code/adfly/lif.py
                                     joint-cohort protocol and arena constants

Conventions (same as analyze_battery.py unless stated):
  * AUC is the direction-free Mann-Whitney AUC, max(U, 1 - U), of a group against
    the healthy flies (stage 0) of the same design.
  * Robust CV = IQR / 1.349 / |median| of healthy flies (pandas linear quantiles).
  * Change = (group median - healthy median) / |healthy median| x 100.
  * Medians as % of healthy use the healthy median of the same design; intervals are
    95% bootstrap percentile intervals over flies within groups (4,000 resamples,
    numpy default_rng seeded from SEED and a fixed per-analysis tag).
  * Mann-Whitney tests are two-sided; Spearman via scipy.stats.spearmanr.
  * Paired AUC differences are reported as point differences (full-sample AUCs) with
    a paired bootstrap interval (healthy and group flies resampled with the same
    indices for both readouts).
  * First detection: a bootstrap median of the first-crossing severity lands on the
    severity grid and flips with the RNG seed when about half of the resamples cross by
    a given severity. Whenever a cumulative crossing share is within 0.10 of 50%, the
    bootstrap is repeated with 10 further seeds and the medians are tallied
    (fd_*_median_stability); seed-dependent medians must not be quoted.
  * Healthy flies (s = 0) carry no pathology, so all of them are the same network; their
    spread is input noise of one network, not variability between individuals.

  python code/scripts/facts_detection.py

Writes results/facts/facts_detection.json.
"""
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())  # code/scripts locally, scripts/ on the server
RES = ROOT / "results"
CODE = ROOT / "code"
OUT = RES / "facts"
OUT.mkdir(parents=True, exist_ok=True)
N_BOOT = 4000
SEED = 20261009
MBON_FLOOR = 0.5          # Hz, as analyze_battery.py / analyze_joint.py
GF_FAIL_HZ = 100.0        # escape-failure criterion of the manuscript (two-GF mean rate)

F, M = {}, {}


# ----------------------------------------------------------------------------- helpers
def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if np.isnan(v):
            return None
        return v
    return v


def put(key, value, meaning):
    assert key not in F, f"duplicate key {key}"
    F[key] = _clean(value)
    M[key] = meaning


def _rnd(x, k):
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return None
    return round(float(x), k) + 0.0          # + 0.0 turns -0.0 into 0.0


def r2(x):
    return _rnd(x, 2)


def r3(x):
    return _rnd(x, 3)


def r4(x):
    return _rnd(x, 4)


def sig(p):
    return None if p is None or np.isnan(p) else float(f"{float(p):.3g}")


def rng(tag):
    return np.random.default_rng([SEED, zlib.crc32(tag.encode())])


def auc(a, b):
    """Direction-free AUC of group b against healthy a (Mann-Whitney statistic)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    u = (gt + 0.5 * eq) / (len(a) * len(b))
    return max(u, 1.0 - u)


def rcv(x):
    x = pd.Series(x).dropna()
    return float((x.quantile(0.75) - x.quantile(0.25)) / 1.349 / abs(x.median()))


def shift_pct(g, h):
    g, h = pd.Series(g).dropna(), pd.Series(h).dropna()
    return float((g.median() - h.median()) / abs(h.median()) * 100)


def pct_ci(x, h, tag):
    """Median of x as % of the median of h, with a 95% bootstrap interval."""
    x = np.asarray(pd.Series(x).dropna(), float)
    h = np.asarray(pd.Series(h).dropna(), float)
    est = np.median(x) / np.median(h) * 100
    g = rng(tag)
    bx = g.choice(x, (N_BOOT, len(x)), replace=True)
    bh = g.choice(h, (N_BOOT, len(h)), replace=True)
    r = np.median(bx, 1) / np.median(bh, 1) * 100
    lo, hi = np.percentile(r, [2.5, 97.5])
    return [round(float(est), 1), round(float(lo), 1), round(float(hi), 1)]


def mw(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    return float(st.mannwhitneyu(a, b, alternative="two-sided")[1])


def spear(x, y):
    m = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}).dropna()
    r, p = st.spearmanr(m.x, m.y)
    return float(r), float(p), int(len(m))


def partial_spearman(x, y, z):
    """Rank correlation of x and y after removing the linear effect of rank(z) from both
    (as analyze_battery.partial_spearman)."""
    m = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float), "z": np.asarray(z, float)}).dropna()
    rx, ry, rz = (st.rankdata(m[c]) for c in ("x", "y", "z"))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r, p = st.pearsonr(ex, ey)
    return float(r), float(p), int(len(m))


def _boot_first(h, per, stages, g):
    """First-crossing severity per bootstrap resample (NaN = never), as analyze_battery.py:
    healthy resampled once per resample, each group resampled in turn until crossing."""
    boot = np.full(N_BOOT, np.nan)
    for i in range(N_BOOT):
        hb = g.choice(h, len(h), replace=True)
        for s in stages:
            if len(per[s]) and auc(hb, g.choice(per[s], len(per[s]), replace=True)) >= 0.80:
                boot[i] = s
                break
    return boot


def _medians(boot, stages):
    ok = ~np.isnan(boot)
    cond_med = float(np.median(boot[ok])) if ok.any() else None
    # unconditional median: smallest severity reached by >= 50% of all resamples
    uncond = next((s for s in stages if np.mean(boot <= s) >= 0.5), None)
    return cond_med, uncond


N_REP = 10            # replicate bootstrap seeds for medians that sit near a 50% boundary
NEAR = 0.10           # a cumulative fraction within NEAR of 0.5 triggers the replicate check


def _cum_at(boot, med, stages, reached):
    """Cumulative fraction of resamples crossing by severity med and by the step below
    (reached=True: over resamples that ever crossed, as for the copied medians; else over all)."""
    b = boot[~np.isnan(boot)] if reached else boot
    if med is None or len(b) == 0 or med not in stages:
        return None, None
    i = stages.index(med)
    below = float(np.mean(b <= stages[i - 1])) if i > 0 else 0.0
    return float(np.mean(b <= med)), below


def first_detection_boot(h, per, stages, tag, copied_median=None):
    """Observed first severity with AUC >= 0.80, the bootstrap of the first crossing, and a
    replicate-seed check of the bootstrap medians.

    A bootstrap median lands on a grid severity; when the share of resamples crossing by that
    severity (or by the step below) is close to 50%, the median flips with the RNG seed. If any
    cumulative fraction of the main run is within NEAR of 0.5, the bootstrap is repeated with
    N_REP further seeds (4,000 resamples each) and the medians of all runs are tallied."""
    obs = next((s for s in stages if len(per[s]) and auc(h, per[s]) >= 0.80), None)
    boot = _boot_first(h, per, stages, rng(tag))
    ok = ~np.isnan(boot)
    cond_med, uncond = _medians(boot, stages)
    cum_all = [float(np.mean(boot <= s)) for s in stages]
    cum_reached = [float(np.mean(boot[ok] <= s)) for s in stages] if ok.any() else []
    near = any(abs(c - 0.5) < NEAR for c in cum_all + cum_reached)
    runs = [boot]
    if near:
        runs += [_boot_first(h, per, stages, rng(f"{tag}_rep{j}")) for j in range(N_REP)]
    meds = [_medians(b, stages) for b in runs]
    pooled = np.concatenate(runs)
    cmed_counts = pd.Series([m[0] for m in meds]).value_counts().sort_index()
    umed_counts = pd.Series(["never" if m[1] is None else m[1] for m in meds], dtype=object).value_counts()
    c_by, c_below = _cum_at(pooled, cond_med, stages, True)
    u_by, u_below = _cum_at(pooled, uncond, stages, False)
    seed_dep = (len(cmed_counts) > 1 or len(umed_counts) > 1
                or (copied_median is not None and cond_med is not None and abs(copied_median - cond_med) > 1e-9))
    # weak: the median held, but barely more than half of the resamples had crossed by it
    near_even = any(x is not None and 0.5 <= x < 0.5 + NEAR for x in (c_by, u_by))
    verdict = "seed-dependent" if seed_dep else ("near-even" if near_even else "stable")
    stab = {"verdict": verdict, "n_runs": len(runs), "n_resamples_pooled": int(len(pooled)),
            "cond_median_counts": {stg(k): int(v) for k, v in cmed_counts.items()},
            "cond_frac_by_median": r3(c_by), "cond_frac_by_step_below": r3(c_below),
            "uncond_median_counts": {(stg(k) if k != "never" else k): int(v) for k, v in umed_counts.items()},
            "uncond_frac_by_median": r3(u_by), "uncond_frac_by_step_below": r3(u_below)}
    return {"observed": obs, "frac_never": float(1 - ok.mean()), "cond_median": cond_med,
            "uncond_median": uncond,
            "frac_by": {s: float(np.mean(boot <= s)) for s in stages},
            "stability": stab}


def stability_note(b, copied_median, cohort, which):
    """Plain-language warning appended to the meaning of a bootstrap-median fact.
    which = 'cond' (median over resamples that reached AUC 0.80, as for the copied medians) or
    'uncond' (median over all resamples, 'never' counted as beyond the last severity)."""
    st_ = b["stability"]
    counts_d = st_[f"{which}_median_counts"]
    by, below = st_[f"{which}_frac_by_median"], st_[f"{which}_frac_by_step_below"]
    med = b["cond_median"] if which == "cond" else b["uncond_median"]
    dep = len(counts_d) > 1 or (which == "cond" and copied_median is not None and med is not None and abs(copied_median - med) > 1e-9)
    weak = by is not None and 0.5 <= by < 0.5 + NEAR
    if not (dep or weak):
        return ""
    obs = "never" if b["observed"] is None else f"s = {stg(b['observed'])}"
    over = "reached resamples" if which == "cond" else "all resamples"
    counts = ", ".join(f"{k} in {v}" for k, v in counts_d.items())
    runs = f"{st_['n_runs']} bootstrap runs of 4,000 resamples (seed {SEED} plus {st_['n_runs'] - 1} replicate seeds)"
    pooled = ("" if by is None else f" Pooled over all runs, {by} of {over} had crossed by s = {stg(med)} and {below} by the step below.")
    if dep:
        copy_note = "" if (copied_median is None or which != "cond") else f"; the copied median is {stg(copied_median)}"
        return (f" CAUTION, seed-dependent ({cohort}): over {runs} the median over {over} came out {counts}{copy_note}.{pooled} The median is not "
                f"reproducible and must not be quoted as a separation severity; quote the observed crossing ({obs}) and the share of resamples "
                f"that crossed by each severity (fd_*_fracby_recomputed_*).")
    return (f" CAUTION, weak ({cohort}): the median over {over} was s = {stg(med)} in all {runs}, but only {by} of {over} had crossed by "
            f"that severity (pooled). Quote it as 'about half of resamples crossed by s = {stg(med)}', alongside the observed crossing ({obs}).")


def stg(x):
    return f"{x:g}"


SHORT = {"brain_spikes_A": "brain", "LH_hz_A": "LH", "PN_hz_A": "PN", "KC_hz_A": "KCrate",
         "KC_frac_A": "KCrecruited", "MBON_hz_A": "MBON", "V_pre_A": "valence", "LI": "LI",
         "LI_norm": "LInorm", "MN9_sugar_hz": "MN9", "GF_loom_hz": "GF", "GF_latency_ms": "GFlatency",
         "behaviour_feed_s": "feedTime", "behaviour_esc": "jumpsPerThreat",
         "behaviour_near_A_s": "nearOdourA", "behaviour_PI": "PI"}
LABEL = {"brain_spikes_A": "whole-brain spike count (odour A)", "LH_hz_A": "lateral horn rate (odour A)",
         "PN_hz_A": "projection neuron rate (odour A)", "KC_hz_A": "Kenyon cell rate (odour A)",
         "KC_frac_A": "fraction of Kenyon cells recruited (odour A)", "MBON_hz_A": "MBON rate (odour A)",
         "V_pre_A": "naive MB valence (odour A)", "LI": "learning index",
         "LI_norm": "learning index / MBON rate (MBON >= 0.5 Hz)", "MN9_sugar_hz": "MN9 sugar response rate",
         "GF_loom_hz": "giant fibre looming rate (mean of the two DNp01)", "GF_latency_ms": "giant fibre first-spike latency",
         "behaviour_feed_s": "arena feeding time", "behaviour_esc": "arena jumps per threat encountered",
         "behaviour_near_A_s": "arena time near odour A", "behaviour_PI": "arena preference index"}

# ----------------------------------------------------------------------------- data
BAT = RES / "battery"


def load(p):
    d = pd.read_csv(p)
    d["stage"] = d["stage"].round(3)
    d["LI_norm"] = d["LI"] / d["MBON_hz_A"].where(d["MBON_hz_A"] >= MBON_FLOOR)
    return d


sc = load(BAT / "stage_curve_x4.csv")
sc = sc[sc.scope == "pan"].copy()
H = sc[sc.stage == 0]
SER_ST = sorted(s for s in sc.stage.unique() if s > 0)
co = load(BAT / "components_x4.csv")
vu = load(BAT / "vulnerability_x4.csv")
sp = load(BAT / "scope_x4.csv")
jd = load(RES / "joint/cohort_joint.csv")
JH = jd[jd.stage == 0]
JST = sorted(s for s in jd.stage.unique() if s > 0)
BSTATS = json.loads((RES / "stats/battery_stats.json").read_text())
SJ = json.loads((RES / "joint/stats_joint.json").read_text())
ONL = json.loads((RES / "paper/output_neuron_loss.json").read_text())
JMETA = json.loads((RES / "joint/cohort_joint_meta.json").read_text())

# Readouts that are undefined in some flies. AUCs, first detections and medians of these use
# only the flies with a defined value; the note says which flies drop out and how many remain.
WHY_NA = {"GF_latency_ms": "flies whose giant fibre never spiked (latency undefined), which are the most affected flies, are "
                           "excluded, so the value describes only flies with at least one GF spike",
          "behaviour_esc": "flies that met no threat (jumps per threat undefined) are excluded",
          "behaviour_PI": "flies that spent no time near either odour (preference index undefined) are excluded",
          "LI_norm": "flies with MBON rate below 0.5 Hz (ratio undefined) are excluded"}


def na_note(d, r, stages):
    if r not in WHY_NA or r not in d:
        return ""
    rows = []
    for s in [0.0] + list(stages):
        g = d[d.stage == s][r]
        if g.isna().any():
            rows.append(f"{int(g.notna().sum())} of {len(g)} at s = {stg(s)}")
    if not rows:
        return ""
    return f" Note: {WHY_NA[r]}: n = {', '.join(rows)} (all flies defined at the other severities)."


NOTE_SER = {r: na_note(sc, r, SER_ST) for r in WHY_NA}
NOTE_J = {r: na_note(jd, r, JST) for r in WHY_NA}

put("n_series_healthy", len(H), "Number of healthy (stage 0, pan scope) flies in the severity series (results/battery/stage_curve_x4.csv).")
put("n_series_per_severity", int(sc[sc.stage == 0.1].shape[0]), "Number of flies per non-zero severity group in the severity series.")
put("series_severities", SER_ST, "Non-zero severities s of the severity series; per-severity lists of series facts follow this order unless their meaning names other severities.")
put("n_joint_healthy", len(JH), "Number of healthy (stage 0) flies in the 176-fly joint cohort (results/joint/cohort_joint.csv).")
put("n_joint_per_severity", int(jd[jd.stage == 0.1].shape[0]), "Number of flies per non-zero severity group in the joint cohort.")
put("joint_severities", JST, "Non-zero severities of the joint cohort; per-severity lists of joint facts follow this order unless their meaning names other severities.")

# Healthy flies carry no pathology: pathology.apply with s = 0 keeps every synapse (binomial with
# p = 1), kills no neuron and shifts no threshold, so every healthy fly is the same network whatever
# its seed. Healthy flies differ only in their slice of the Poisson stimulus stream that the flies of
# a batch share (assay_battery.py, cohort_joint.py). Their spread is trial-to-trial input noise of one
# network, not variability between individuals.
PATH_COLS = ["dead_frac", "kc_dead_frac", "mean_load"]
put("healthy_pathology_all_zero", [int((H[PATH_COLS] == 0).all(axis=1).sum()), len(H), int((JH[PATH_COLS] == 0).all(axis=1).sum()), len(JH)],
    "Healthy flies with dead_frac = kc_dead_frac = mean_load = 0: [severity series n, series healthy total, joint cohort n, joint healthy total]. "
    "Every healthy fly is therefore the identical intact network (pathology.apply at s = 0 removes nothing); healthy flies differ only in the "
    "Poisson input noise they receive.")
NOISE = (" The healthy flies are all the identical intact network (healthy_pathology_all_zero), so this is the trial-to-trial spread of one "
         "network under Poisson input noise, not variability between individual flies.")

# ============================================================================= A. precision
PREC = ["brain_spikes_A", "LH_hz_A", "PN_hz_A", "KC_hz_A", "KC_frac_A", "MBON_hz_A", "V_pre_A",
        "LI", "MN9_sugar_hz", "GF_loom_hz"]
GRADED = PREC[:8]
cv = {r: rcv(H[r]) for r in PREC}
rank10 = {r: i + 1 for i, r in enumerate(sorted(PREC, key=lambda r: cv[r]))}
rank8 = {r: i + 1 for i, r in enumerate(sorted(GRADED, key=lambda r: cv[r]))}
for r in PREC:
    k = SHORT[r]
    put(f"series_healthy_rcv_pct_{k}", r2(100 * cv[r]),
        f"Healthy robust CV (IQR/1.349/|median|, %) of {LABEL[r]} over the 64 healthy flies of the severity series.{NOISE}")
    put(f"series_healthy_rcv_rank_of10_{k}", rank10[r],
        f"Rank of the healthy robust CV (input-noise spread of the one healthy network) of {LABEL[r]} among the 10 series readouts of the precision analysis (1 = least variable).")
    if r in GRADED:
        put(f"series_healthy_rcv_rank_of8graded_{k}", rank8[r],
            f"Rank of the healthy robust CV (input-noise spread of the one healthy network) of {LABEL[r]} among the 8 graded readouts (MN9 and GF excluded; 1 = least variable).")
    put(f"series_change_pct_{k}", [r2(shift_pct(sc[sc.stage == s][r], H[r])) for s in (0.05, 0.1, 0.2)],
        f"Change of the group median of {LABEL[r]} relative to the healthy median (%), at s = 0.05, 0.10, 0.20 (32 flies each vs 64 healthy).")
    put(f"series_auc_{k}", [r4(auc(H[r], sc[sc.stage == s][r])) for s in (0.05, 0.1, 0.2)],
        f"Direction-free AUC of {LABEL[r]} against healthy at s = 0.05, 0.10, 0.20 (severity series).")


def rank_corrs(reads, s):
    h = {r: H[r] for r in reads}
    shift = [abs(shift_pct(sc[sc.stage == s][r], h[r])) for r in reads]
    spread = [cv[r] for r in reads]
    a = [auc(h[r], sc[sc.stage == s][r]) for r in reads]
    z = [x / 100 / y for x, y in zip(shift, spread)]
    return (st.spearmanr(z, a), st.spearmanr(shift, a), st.spearmanr([1 / y for y in spread], a))


for s in (0.05, 0.1, 0.2):
    zr, sr, pr = rank_corrs(GRADED, s)
    put(f"series_rankcorr_auc_vs_shift_over_spread_s{stg(s)}", [r3(zr[0]), sig(zr[1])],
        f"Spearman rho and p across the 8 graded series readouts between AUC at s = {stg(s)} and |median change|/healthy robust CV (n = 8 readouts).")
    put(f"series_rankcorr_auc_vs_shift_alone_s{stg(s)}", [r3(sr[0]), sig(sr[1])],
        f"Spearman rho and p across the 8 graded series readouts between AUC at s = {stg(s)} and |median change| alone (n = 8).")
    put(f"series_rankcorr_auc_vs_inverse_spread_s{stg(s)}", [r3(pr[0]), sig(pr[1])],
        f"Spearman rho and p across the 8 graded series readouts between AUC at s = {stg(s)} and 1/healthy robust CV alone (n = 8).")
zr, _, _ = rank_corrs(PREC, 0.1)
put("series_rankcorr_auc_vs_shift_over_spread_s0.1_all10", [r3(zr[0]), sig(zr[1])],
    "Spearman rho and p between AUC at s = 0.10 and |median change|/healthy robust CV across all 10 series readouts including MN9 and GF.")
put("series_rankcorr_s0.1_copied", [BSTATS["series"]["precision_rank_corr"]["rho"], sig(BSTATS["series"]["precision_rank_corr"]["p"])],
    "Copied from results/stats/battery_stats.json series.precision_rank_corr: rho and p (8 graded readouts, s = 0.10).")

# full AUC curves (observed) for every series readout, the curves of Fig 7A
for r in ["brain_spikes_A", "LH_hz_A", "PN_hz_A", "KC_hz_A", "KC_frac_A", "MBON_hz_A", "V_pre_A", "LI",
          "LI_norm", "MN9_sugar_hz", "GF_loom_hz", "GF_latency_ms"]:
    put(f"series_auc_curve_{SHORT[r]}", [r3(auc(H[r].dropna(), sc[sc.stage == s][r].dropna())) if sc[sc.stage == s][r].notna().any() else None
                                         for s in SER_ST],
        f"Observed AUC of {LABEL[r]} against healthy at every series severity s = 0.05-1.0 (null = undefined in every fly).{NOTE_SER.get(r, '')}")
# joint-cohort precision, circuit and arena
JPREC = ["brain_spikes_A", "LH_hz_A", "KC_hz_A", "MBON_hz_A", "LI", "MN9_sugar_hz", "GF_loom_hz",
         "behaviour_feed_s", "behaviour_esc", "behaviour_near_A_s", "behaviour_PI"]
SIGNED = {"behaviour_PI": (" NOT INTERPRETABLE, do not quote: the preference index is a signed index in [-1, 1] centred near 0 "
                          "(healthy median {hm}, bimodal at -1 and +1), so a percentage of its median has no meaning. Use "
                          "joint_PI_absolute_s0.1 or the AUC.")}
for r in JPREC:
    k = SHORT[r]
    nh = int(JH[r].notna().sum())
    bad = SIGNED.get(r, "").format(hm=r3(JH[r].median()))
    put(f"joint_healthy_rcv_pct_{k}", r2(100 * rcv(JH[r])),
        f"Healthy robust CV (%) of {LABEL[r]} in the joint cohort (n = {nh} healthy flies with a defined value).{NOISE}{bad}")
    put(f"joint_change_pct_s0.1_{k}", r2(shift_pct(jd[jd.stage == 0.1][r], JH[r])),
        f"Change of the s = 0.10 median of {LABEL[r]} relative to the joint-cohort healthy median (%).{bad}")
    put(f"joint_auc_{k}", [r4(auc(JH[r].dropna(), jd[jd.stage == s][r].dropna())) for s in JST],
        f"Direction-free AUC of {LABEL[r]} against joint-cohort healthy flies at s = 0.05, 0.10, 0.15, 0.20, 0.30, 0.50.{NOTE_J.get(r, '')}")
pi_h, pi_1 = JH.behaviour_PI.dropna(), jd[jd.stage == 0.1].behaviour_PI.dropna()
put("joint_PI_absolute_s0.1", [r3(pi_h.median()), r3(pi_1.median()), r3(pi_1.median() - pi_h.median()),
                               r3((pi_h.quantile(0.75) - pi_h.quantile(0.25)) / 1.349)],
    "Joint cohort arena preference index on its own scale (PI in [-1, 1]): [healthy median, s = 0.10 median, difference of medians (PI units), "
    "healthy IQR/1.349 (PI units)]. Replaces the percentage change and robust CV, which are undefined in meaning for a signed index near 0.")

# ============================================================================= B. first detection
SER_READS = ["brain_spikes_A", "LH_hz_A", "PN_hz_A", "KC_hz_A", "KC_frac_A", "MBON_hz_A",
             "V_pre_A", "LI", "LI_norm", "MN9_sugar_hz", "GF_loom_hz", "GF_latency_ms"]
fd_ser = {}
for r in SER_READS:
    k = SHORT[r]
    h = H[r].dropna().values
    per = {s: sc[sc.stage == s][r].dropna().values for s in SER_ST}
    js = BSTATS["series"]["first_detection"][r]
    b = first_detection_boot(h, per, SER_ST, f"fd_series_{r}", js["boot_median"])
    fd_ser[r] = b
    assert js["observed"] == b["observed"], (r, js["observed"], b["observed"])
    nd = NOTE_SER.get(r, "")
    warn_c = stability_note(b, js["boot_median"], "severity series", "cond")
    warn_u = stability_note(b, js["boot_median"], "severity series", "uncond")
    put(f"fd_series_observed_{k}", b["observed"],
        f"Observed (full-sample) first severity at which {LABEL[r]} reaches AUC >= 0.80 against healthy in the severity series (null = never).{nd}")
    put(f"fd_series_bootmedian_copied_{k}", js["boot_median"],
        f"Bootstrap median first severity for {LABEL[r]}, taken only over resamples that reached AUC 0.80 (copied from battery_stats.json series.first_detection, the bootstrap of analyze_battery.py, 4,000 resamples).{nd}{warn_c}")
    put(f"fd_series_bootci_copied_{k}", js["boot_ci95"],
        f"2.5-97.5 percentile range of the first-crossing severity for {LABEL[r]} over resamples that reached the criterion (battery_stats.json).{nd}")
    put(f"fd_series_fracnever_copied_{k}", js["frac_never"],
        f"Fraction of the 4,000 bootstrap resamples of analyze_battery.py in which {LABEL[r]} never reached AUC 0.80 (battery_stats.json).{nd}")
    put(f"fd_series_uncondmedian_recomputed_{k}", b["uncond_median"],
        f"Median first severity for {LABEL[r]} over ALL 4,000 recomputed resamples with 'never' counted as beyond s = 1.0 (null = median is never); seed {SEED}.{nd}{warn_u}")
    put(f"fd_series_fracby_recomputed_{k}", [r3(b["frac_by"][s]) for s in SER_ST],
        f"Share of ALL 4,000 recomputed resamples (seed {SEED}) in which {LABEL[r]} first reached AUC 0.80 at or below each series severity s = 0.05-1.0 (cumulative; 1 minus the last value never reached).{nd}")
    if b["observed"] is not None:
        i = SER_ST.index(b["observed"])
        if i > 0:
            put(f"fd_series_auc_step_before_observed_{k}", r4(auc(h, per[SER_ST[i - 1]])),
                f"AUC of {LABEL[r]} at the severity one step below its observed first crossing (s = {stg(SER_ST[i - 1])}).{nd}")
put("fd_series_recomputed_condmedian_check", {SHORT[r]: fd_ser[r]["cond_median"] for r in SER_READS},
    f"Recomputed (seed {SEED}, not the seed of analyze_battery.py) bootstrap median over reached resamples, per series readout, as a check of the copied values. "
    f"A median that agrees with the copied one is not thereby reproducible: see fd_series_median_stability (seed-dependent: "
    f"{', '.join(SHORT[r] for r in SER_READS if fd_ser[r]['stability']['verdict'] == 'seed-dependent') or 'none'}).")
put("fd_series_median_stability", {SHORT[r]: fd_ser[r]["stability"] for r in SER_READS},
    f"Severity series, per readout: replicate-seed check of the first-detection bootstrap medians. When any cumulative crossing share of the main run "
    f"(seed {SEED}) is within {NEAR} of 50%, the 4,000-resample bootstrap is repeated with {N_REP} further seeds. verdict: 'seed-dependent' = the median "
    f"(over reached resamples, or over all resamples) differs between runs or from the copied median; 'near-even' = same median in every run but the pooled share "
    f"of resamples crossing by the median severity is below {0.5 + NEAR:g} (barely half); 'stable' = neither. cond_* = over resamples that reached "
    f"AUC 0.80 (as for the copied medians); uncond_* = over all resamples ('never' beyond the last severity); *_median_counts tallies the median of each run; "
    f"*_frac_by_median and *_frac_by_step_below are pooled over all runs and taken at the seed-{SEED} median severity.")
put("fd_series_median_seed_dependent", [SHORT[r] for r in SER_READS if fd_ser[r]["stability"]["verdict"] == "seed-dependent"],
    "Severity-series readouts whose bootstrap first-detection median changes with the RNG seed (see fd_series_median_stability); quote their observed crossing instead.")
put("fd_series_median_near_even", [SHORT[r] for r in SER_READS if fd_ser[r]["stability"]["verdict"] == "near-even"],
    f"Severity-series readouts whose bootstrap median held in every run, but only {0.5:g}-{0.5 + NEAR:g} of the resamples had crossed by the median "
    f"severity: weak, quote with that share (see fd_series_median_stability).")

JREADS = ["KC_frac_A", "KC_hz_A", "PN_hz_A", "LH_hz_A", "MBON_hz_A", "brain_spikes_A", "V_pre_A",
          "MN9_sugar_hz", "GF_loom_hz", "GF_latency_ms", "LI", "LI_norm",
          "behaviour_feed_s", "behaviour_esc", "behaviour_near_A_s", "behaviour_PI"]
fd_j = {}
for r in JREADS:
    k = SHORT[r]
    h = JH[r].dropna().values
    per = {s: jd[jd.stage == s][r].dropna().values for s in JST}
    js = SJ["cascade"][r]
    b = first_detection_boot(h, per, JST, f"fd_joint_{r}", js["boot_median"])
    fd_j[r] = b
    assert js["observed"] == b["observed"], (r, js["observed"], b["observed"])
    nd = NOTE_J.get(r, "")
    warn_c = stability_note(b, js["boot_median"], "joint cohort", "cond")
    warn_u = stability_note(b, js["boot_median"], "joint cohort", "uncond")
    put(f"fd_joint_observed_{k}", b["observed"],
        f"Observed first severity at which {LABEL[r]} reaches AUC >= 0.80 against healthy in the joint cohort (null = never up to s = 0.5).{nd}")
    put(f"fd_joint_bootmedian_copied_{k}", js["boot_median"],
        f"Joint-cohort bootstrap median first severity for {LABEL[r]} over resamples that reached AUC 0.80 (copied from results/joint/stats_joint.json cascade, analyze_joint.py, 4,000 resamples).{nd}{warn_c}")
    put(f"fd_joint_bootci_copied_{k}", js["boot_ci95"],
        f"2.5-97.5 percentile range of the joint-cohort first-crossing severity for {LABEL[r]} over reached resamples (stats_joint.json).{nd}")
    put(f"fd_joint_fracnever_copied_{k}", round(js["frac_never"], 4),
        f"Fraction of 4,000 joint-cohort resamples in which {LABEL[r]} never reached AUC 0.80 by s = 0.5 (stats_joint.json).{nd}")
    put(f"fd_joint_uncondmedian_recomputed_{k}", b["uncond_median"],
        f"Median first severity for {LABEL[r]} in the joint cohort over ALL 4,000 recomputed resamples, 'never' counted as beyond s = 0.5 (null = never); seed {SEED}.{nd}{warn_u}")
    put(f"fd_joint_fracby_recomputed_{k}", [r3(b["frac_by"][s]) for s in JST],
        f"Share of ALL 4,000 recomputed joint-cohort resamples (seed {SEED}) in which {LABEL[r]} first reached AUC 0.80 at or below each severity s = 0.05-0.50 (cumulative; 1 minus the last value never reached).{nd}")
put("fd_joint_recomputed_condmedian_check", {SHORT[r]: fd_j[r]["cond_median"] for r in JREADS},
    f"Joint cohort: recomputed (seed {SEED}, not the seed of analyze_joint.py) bootstrap median over reached resamples, per readout, as a check of the values "
    f"in stats_joint.json. See fd_joint_median_stability for which medians change with the seed (seed-dependent: "
    f"{', '.join(SHORT[r] for r in JREADS if fd_j[r]['stability']['verdict'] == 'seed-dependent') or 'none'}).")
put("fd_joint_median_stability", {SHORT[r]: fd_j[r]["stability"] for r in JREADS},
    f"Joint cohort, per readout: replicate-seed check of the first-detection bootstrap medians, defined as fd_series_median_stability "
    f"(seed {SEED} plus {N_REP} replicate seeds when a cumulative crossing share is within {NEAR} of 50%).")
put("fd_joint_median_seed_dependent", [SHORT[r] for r in JREADS if fd_j[r]["stability"]["verdict"] == "seed-dependent"],
    "Joint-cohort readouts whose bootstrap first-detection median changes with the RNG seed or differs from the copied median (see fd_joint_median_stability); "
    "quote their observed crossing instead.")
put("fd_joint_median_near_even", [SHORT[r] for r in JREADS if fd_j[r]["stability"]["verdict"] == "near-even"],
    f"Joint-cohort readouts whose bootstrap median held in every run, but only {0.5:g}-{0.5 + NEAR:g} of the resamples had crossed by the median "
    f"severity: weak, quote with that share (see fd_joint_median_stability).")
b = fd_j["behaviour_esc"]
put("fd_joint_jumpsPerThreat_recomputed", [round(b["frac_never"], 3), round(b["frac_by"][0.2], 3), round(b["frac_by"][0.5], 3)],
    "Recomputed joint-cohort bootstrap for jumps per threat: fraction of all 4,000 resamples never reaching AUC 0.80, fraction reaching it by s = 0.20, and by s = 0.50.")

# ============================================================================= C. paired comparisons (point differences)
pv = pd.read_csv(RES / "stats/series_paired_vs_LI.csv")
pv["stage"] = pv.stage.round(3)
for r in ("brain_spikes_A", "LH_hz_A"):
    k = SHORT[r]
    rows = pv[pv.readout == r].sort_values("stage")
    pts = []
    for _, x in rows.iterrows():
        S = sc[sc.stage == x.stage]
        pts.append(r4(auc(H[r], S[r]) - auc(H["LI"], S["LI"])))
    put(f"series_pair_{k}_minus_LI_point", pts,
        f"Point AUC difference ({LABEL[r]} minus learning index, full-sample AUCs, same flies) at s = 0.05, 0.10, 0.15, 0.20, 0.30 in the severity series.")
    put(f"series_pair_{k}_minus_LI_bootmean_copied", rows.mean_diff.round(4).tolist(),
        f"Bootstrap mean of the paired AUC difference ({k} minus LI) at s = 0.05-0.30 as written by analyze_battery.py (results/stats/series_paired_vs_LI.csv).")
    put(f"series_pair_{k}_minus_LI_ci", [[r4(a), r4(c)] for a, c in zip(rows.ci_lo, rows.ci_hi)],
        f"95% paired-bootstrap interval (4,000 resamples) of the AUC difference ({k} minus LI) at s = 0.05-0.30 (series_paired_vs_LI.csv).")
x = pv[(pv.stage == 0.05) & (pv.readout == "brain_spikes_A")].iloc[0]
put("series_pair_brain_minus_LI_s0.05_aucs", [x.auc_readout, x.auc_LI],
    "Full-sample AUCs at s = 0.05 of whole-brain spike count and of the learning index (series_paired_vs_LI.csv); their difference is the point estimate to quote.")


def paired_boot(d0, d1, ra, rb, tag):
    """Point difference AUC(ra) - AUC(rb) and paired bootstrap 95% interval."""
    pt = auc(d0[ra].dropna(), d1[ra].dropna()) - auc(d0[rb].dropna(), d1[rb].dropna())
    g = rng(tag)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        hi = g.integers(0, len(d0), len(d0))
        si = g.integers(0, len(d1), len(d1))
        hh, ss = d0.iloc[hi], d1.iloc[si]
        diffs[i] = auc(hh[ra].dropna(), ss[ra].dropna()) - auc(hh[rb].dropna(), ss[rb].dropna())
    lo, hi_ = np.percentile(diffs, [2.5, 97.5])
    return r4(pt), [r4(lo), r4(hi_)]


for ra in ("brain_spikes_A", "LH_hz_A"):
    pts, cis = [], []
    for s in JST:
        p_, c_ = paired_boot(JH, jd[jd.stage == s], ra, "LI", f"joint_pair_{ra}_LI_{s}")
        pts.append(p_)
        cis.append(c_)
    put(f"joint_pair_{SHORT[ra]}_minus_LI_point", pts,
        f"Joint cohort: point AUC difference ({LABEL[ra]} minus learning index, same flies) at s = 0.05, 0.10, 0.15, 0.20, 0.30, 0.50.")
    put(f"joint_pair_{SHORT[ra]}_minus_LI_ci", cis,
        f"Joint cohort: 95% paired-bootstrap interval (4,000 resamples, seed {SEED}) of the AUC difference ({SHORT[ra]} minus LI) at s = 0.05-0.50.")

jva = pd.read_csv(RES / "stats/joint_vs_arena.csv")
jva["stage"] = jva.stage.round(3)
BEH = ["behaviour_feed_s", "behaviour_esc", "behaviour_near_A_s", "behaviour_PI"]
for c in ("brain_spikes_A", "LH_hz_A", "LI"):
    rows = jva[jva.circuit == c].sort_values("stage")
    pts = []
    for _, x in rows.iterrows():
        S = jd[jd.stage == x.stage]
        pts.append(r4(auc(JH[c].dropna(), S[c].dropna()) - auc(JH[x.closest_arena].dropna(), S[x.closest_arena].dropna())))
    put(f"joint_{SHORT[c]}_vs_closest_arena_point", pts,
        f"Joint cohort: point AUC difference of {LABEL[c]} minus the closest arena readout (lowest bootstrap lower bound, results/stats/joint_vs_arena.csv) at s = 0.05-0.50.")
    put(f"joint_{SHORT[c]}_vs_closest_arena_name", [SHORT[a] for a in rows.closest_arena],
        f"Closest arena readout (lowest 2.5th percentile of the paired difference) for {SHORT[c]} at s = 0.05-0.50 (joint_vs_arena.csv).")
    put(f"joint_{SHORT[c]}_vs_closest_arena_ci", [[a, b_] for a, b_ in zip(rows.lo, rows.hi)],
        f"95% paired-bootstrap interval (4,000 resamples, analyze_battery.py) of {SHORT[c]} minus the closest arena readout at s = 0.05-0.50 (joint_vs_arena.csv).")
    put(f"joint_{SHORT[c]}_beats_all_arena", rows.beats_all.astype(bool).tolist(),
        f"Whether the 95% interval of {SHORT[c]} minus every arena readout excluded zero (lowest lower bound > 0), at s = 0.05-0.50 (joint_vs_arena.csv).")
li_best = []
for s in JST:
    S = jd[jd.stage == s]
    ab = {b_: auc(JH[b_].dropna(), S[b_].dropna()) for b_ in BEH}
    best = max(ab, key=ab.get)
    li_best.append([r3(auc(JH.LI, S.LI)), r3(ab[best]), SHORT[best]])
put("joint_LI_vs_best_arena_point_auc", li_best,
    "Joint cohort, per severity s = 0.05-0.50: [learning-index AUC, highest arena-readout AUC, which arena readout] (full-sample point AUCs).")
li_rows = jva[jva.circuit == "LI"].sort_values("stage")
li_beats = int(li_rows.beats_all.astype(bool).sum())
li_ci0 = int(((li_rows.lo <= 0) & (li_rows.hi >= 0)).sum())
put("joint_LI_point_auc_above_all_arena_n_of6", int(sum(a > b_ for a, b_, _ in li_best)),
    "Number of the 6 joint-cohort severities at which the learning-index point AUC exceeds the point AUC of every arena readout. "
    f"Point estimates only: never quote without joint_LI_vs_closest_arena_ci_includes0_n_of6. The paired 95% interval of LI minus the closest "
    f"arena readout includes 0 at {li_ci0} of 6 severities, and LI is resolved better than every arena readout (joint_LI_beats_all_arena) "
    f"at {li_beats} of 6, so the learning index is not shown to beat the arena at any severity.")
put("joint_LI_vs_closest_arena_ci_includes0_n_of6", [li_ci0, li_beats],
    "Joint cohort: [number of the 6 severities at which the 95% paired-bootstrap interval of LI AUC minus the closest arena readout AUC includes 0, "
    "number at which LI is resolved better than every arena readout] (results/stats/joint_vs_arena.csv).")

# ============================================================================= D. joint-cohort learning index
put("joint_LI_healthy_rcv_pct_vs_series", [r2(100 * rcv(JH.LI)), r2(100 * rcv(H.LI))],
    "Healthy robust CV (%) of the learning index: [joint cohort (n = 32), severity series (n = 64)]." + NOISE)
pairs_li = []
for s in [0.0] + JST:
    a, b_ = jd[jd.stage == s].LI, sc[sc.stage == s].LI
    pairs_li.append([s, r2(a.median()), r2(b_.median()), sig(mw(a, b_))])
put("LI_median_joint_vs_series_by_severity", pairs_li,
    "Per severity (0 and the joint-cohort severities): [s, joint-cohort LI median, series LI median, two-sided Mann-Whitney p joint vs series].")
put("joint_LI_pct_healthy", [pct_ci(jd[jd.stage == s].LI, JH.LI, f"jli_{s}") for s in JST],
    "Joint cohort learning index median as % of joint healthy median, with 95% bootstrap interval, at s = 0.05-0.50 ([pct, lo, hi] per severity).")
put("series_LI_pct_healthy_s0.05", pct_ci(sc[sc.stage == 0.05].LI, H.LI, "sli_0.05"),
    "Severity-series learning index median at s = 0.05 as % of series healthy median [pct, lo, hi].")
rj = spear(jd.stage, jd.LI)
rs = spear(sc[sc.stage <= 0.5].stage, sc[sc.stage <= 0.5].LI)
put("LI_vs_severity_rho_joint_series_le0.5", [r3(rj[0]), r3(rs[0])],
    "Spearman rho of severity with learning index: [joint cohort, all 176 flies; severity series restricted to s <= 0.5]. The series subset "
    "includes s = 0.4, which the joint cohort lacks; LI_vs_severity_rho_series_joint_severities is the like-for-like comparison.")
JSEV = [0.0] + JST
sj = sc[sc.stage.isin(JSEV)]
rsj = spear(sj.stage, sj.LI)
put("LI_vs_severity_rho_series_joint_severities", [r3(rsj[0]), sig(rsj[1]), rsj[2]],
    "Severity series restricted to the joint-cohort severities (0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50; s = 0.4 dropped): Spearman rho, p, n of "
    "severity with learning index; compare with the joint-cohort rho in LI_vs_severity_rho_joint_series_le0.5.")
jn = jd.dropna(subset=["LI_norm"])
sn = sc[(sc.stage <= 0.5)].dropna(subset=["LI_norm"])
put("LInorm_vs_severity_joint", [r3(spear(jn.stage, jn.LI_norm)[0]), sig(spear(jn.stage, jn.LI_norm)[1]), len(jn)],
    "Joint cohort: Spearman rho, p and n of severity with LI/MBON ratio (flies with MBON >= 0.5 Hz).")
put("LInorm_vs_severity_series_le0.5", [r3(spear(sn.stage, sn.LI_norm)[0]), sig(spear(sn.stage, sn.LI_norm)[1]), len(sn)],
    "Severity series, s <= 0.5 (including healthy and s = 0.4): Spearman rho, p and n of severity with LI/MBON ratio (MBON >= 0.5 Hz).")
snj = sj.dropna(subset=["LI_norm"])
rr = spear(snj.stage, snj.LI_norm)
put("LInorm_vs_severity_series_joint_severities", [r3(rr[0]), sig(rr[1]), rr[2]],
    "Severity series restricted to the joint-cohort severities (s = 0.4 dropped, healthy included): Spearman rho, p and n of severity with "
    "LI/MBON ratio (MBON >= 0.5 Hz); like-for-like with LInorm_vs_severity_joint.")
put("LInorm_healthy_median_joint_vs_series", [r2(JH.LI_norm.median()), r2(H.LI_norm.median())],
    "Healthy median LI/MBON ratio: [joint cohort, severity series].")

# ============================================================================= E. LI/MBON ratio
put("series_LInorm_pct_healthy", [pct_ci(sc[sc.stage == s].LI_norm, H.LI_norm, f"ratio_{s}")
                                  if sc[sc.stage == s].LI_norm.notna().sum() else None for s in SER_ST],
    "Severity series: median LI/MBON ratio (defined where MBON >= 0.5 Hz) as % of healthy median, 95% bootstrap interval, per severity s = 0.05-1.0 ([pct, lo, hi]; null = undefined in every fly).")
put("series_LInorm_n_defined", [int(sc[sc.stage == s].LI_norm.notna().sum()) for s in SER_ST],
    "Severity series: number of flies (of 32) with a defined LI/MBON ratio (MBON >= 0.5 Hz) at s = 0.05-1.0.")
put("series_LInorm_mw_p_vs_healthy", [sig(mw(sc[sc.stage == s].LI_norm, H.LI_norm))
                                      if sc[sc.stage == s].LI_norm.notna().sum() >= 3 else None for s in SER_ST],
    "Severity series: uncorrected two-sided Mann-Whitney p of LI/MBON ratio, each severity vs healthy, s = 0.05-1.0.")


def trend(lo, hi, incl0=False):
    d = sc[((sc.stage >= lo) & (sc.stage <= hi)) | ((sc.stage == 0) & incl0)].dropna(subset=["LI_norm"])
    r, p, n = spear(d.stage, d.LI_norm)
    return [r3(r), sig(p), n]


put("series_LInorm_trend_0_to_0.3_incl_healthy", trend(0.05, 0.3, True),
    "Spearman rho, p, n of severity with LI/MBON ratio over s = 0-0.3 including healthy flies.")
put("series_LInorm_trend_0.05_to_0.3", trend(0.05, 0.3), "Spearman rho, p, n of severity with LI/MBON ratio over affected flies s = 0.05-0.3.")
put("series_LInorm_trend_0.1_to_0.3", trend(0.1, 0.3), "Spearman rho, p, n of severity with LI/MBON ratio within affected flies s = 0.1-0.3.")
put("series_LInorm_trend_0.1_to_0.6", trend(0.1, 0.6), "Spearman rho, p, n of severity with LI/MBON ratio within affected flies s = 0.1-0.6 (defined ratios only).")
put("series_LInorm_trend_ge0.1_all", trend(0.1, 1.0), "Spearman rho, p, n of severity with LI/MBON ratio over all affected flies with a defined ratio, s >= 0.1.")
grp = [sc[sc.stage == s].LI_norm.dropna() for s in SER_ST if 0.1 <= s <= 0.6]
put("series_LInorm_kruskal_0.1_to_0.6", [r2(st.kruskal(*grp).statistic), sig(st.kruskal(*grp).pvalue)],
    "Kruskal-Wallis H and p of LI/MBON ratio across the 7 severity groups s = 0.1-0.6.")
for lo, hi in ((0.1, 0.3), (0.1, 0.6)):
    pool = sc[(sc.stage >= lo) & (sc.stage <= hi)].LI_norm.dropna()
    put(f"series_LInorm_pooled_{stg(lo)}_to_{stg(hi)}", pct_ci(pool, H.LI_norm, f"pool_{lo}_{hi}") + [sig(mw(pool, H.LI_norm)), int(len(pool))],
        f"Severity series, flies at s = {stg(lo)}-{stg(hi)} pooled: median LI/MBON ratio as % of healthy [pct, lo, hi], Mann-Whitney p vs healthy, n.")
d = sc[(sc.stage >= 0.1) & (sc.stage <= 0.6)]
pr = partial_spearman(d.stage, d.LI, d.MBON_hz_A)
put("series_partial_stage_LI_given_MBON_0.1_to_0.6", [r3(pr[0]), sig(pr[1]), pr[2]],
    "Partial rank correlation of severity with raw learning index controlling for MBON rate, affected flies s = 0.1-0.6 (rho, p, n).")
surv = []
for s in (0.4, 0.5, 0.6, 0.75):
    g = sc[sc.stage == s]
    dd = g.dropna(subset=["LI_norm"])
    surv.append([s, len(dd), r2(g.MBON_hz_A.median()), r2(dd.MBON_hz_A.median()), r2(g.LI.median()), r2(dd.LI.median())])
put("series_LInorm_survivor_selection", surv,
    "Severity series, s = 0.4, 0.5, 0.6, 0.75: [s, n with defined ratio (of 32), MBON median all flies (Hz), MBON median defined flies, LI median all flies, LI median defined flies].")
put("series_LInorm_n_undefined_total", int(sc.LI_norm.isna().sum()), "Severity series: number of flies with undefined LI/MBON ratio (MBON < 0.5 Hz).")
put("components_all_s0.5_LInorm_pct_copied", [BSTATS["components"]["all_s0.5"]["LI_norm"][k] for k in ("pct", "lo", "hi", "n")],
    "Rule-ablation cohort, all three rules at s = 0.5: LI/MBON ratio as % of series healthy [pct, lo, hi, n] (copied from battery_stats.json components.all_s0.5.LI_norm).")

# ============================================================================= F. LI vs MBON within severity
wr = []
for s in [0.0] + SER_ST:
    g = sc[sc.stage == s]
    r, p, n = spear(g.LI, g.MBON_hz_A)
    wr.append([s, r3(r), sig(p)])
put("series_LI_vs_MBON_within_severity", wr,
    "Severity series, within each group (healthy n = 64, others n = 32): [s, Spearman rho of LI vs MBON rate, p].")
put("series_LI_vs_MBON_pooled_all", [r3(spear(sc.LI, sc.MBON_hz_A)[0]), len(sc)],
    "Severity series: pooled Spearman rho of LI vs MBON rate over all 384 flies (the rho printed in S3 Fig, panel A), and n.")
put("series_LI_vs_MBON_pooled_le0.3", [r3(spear(sc[sc.stage <= 0.3].LI, sc[sc.stage <= 0.3].MBON_hz_A)[0]), int((sc.stage <= 0.3).sum())],
    "Severity series: pooled Spearman rho of LI vs MBON rate over flies at s <= 0.3 including healthy, and n.")
put("series_LI_vs_MBON_pooled_ge0.4", [r3(spear(sc[sc.stage >= 0.4].LI, sc[sc.stage >= 0.4].MBON_hz_A)[0]), int((sc.stage >= 0.4).sum())],
    "Severity series: pooled Spearman rho of LI vs MBON rate over flies at s >= 0.4, and n.")
for nm, d, desc in (("all", sc, "all 384 flies, healthy included"), ("le0.3", sc[sc.stage <= 0.3], "s <= 0.3 including the 64 healthy flies"),
                    ("ge0.4", sc[sc.stage >= 0.4], "s >= 0.4, affected flies only")):
    pr = partial_spearman(d.LI, d.MBON_hz_A, d.stage)
    put(f"series_partial_LI_MBON_given_severity_{nm}", [r3(pr[0]), sig(pr[1]), pr[2]],
        f"Severity series ({desc}): partial rank correlation of LI with MBON rate controlling for severity rank (rho, p, n).")
pr = partial_spearman(sc.stage, sc.LI, sc.MBON_hz_A)
put("series_partial_severity_LI_given_MBON_all", [r3(pr[0]), sig(pr[1]), pr[2]],
    "Severity series, all 384 flies: partial rank correlation of severity with LI controlling for MBON rate.")

# ============================================================================= G. giant fibre and MN9 failures
A = sc[sc.stage > 0]
gfH = H.GF_loom_hz
put("gf_healthy_range_hz", [r2(gfH.min()), r2(gfH.max())], "Severity series: range of giant fibre rate (two-DNp01 mean, Hz) in the 64 healthy flies.")
put("gf_healthy_latency_ms_counts", {str(int(k)): int(v) for k, v in H.GF_latency_ms.value_counts().sort_index().items()},
    "Severity series healthy flies: number of flies at each giant fibre first-spike latency (1-ms bins).")
fail = A[A.GF_loom_hz < GF_FAIL_HZ]
silent = fail[fail.GF_loom_hz == 0]
firing = fail[fail.GF_loom_hz > 0]
put("gf_n_affected", len(A), "Severity series: number of affected flies (s > 0).")
put("gf_n_below100", len(fail), "Severity series: affected flies with giant fibre rate below 100 Hz (the criterion of Fig 6A and 6C).")
put("gf_n_below100_silent", len(silent), "Of the affected flies below 100 Hz, number with giant fibre rate exactly 0 (no spike).")
put("gf_n_below100_firing", len(firing), "Of the affected flies below 100 Hz, number that still fire (rate > 0).")
put("gf_below100_firing_rate_hz", [r2(firing.GF_loom_hz.median()), r2(firing.GF_loom_hz.min()), r2(firing.GF_loom_hz.max())],
    "Firing flies below 100 Hz: [median, min, max] giant fibre rate (two-DNp01 mean, Hz).")
lat = firing.GF_latency_ms
put("gf_below100_firing_latency_ms", [r2(lat.median()), r2(lat.quantile(0.25)), r2(lat.quantile(0.75)), r2(lat.min()), r2(lat.max())],
    "Firing flies below 100 Hz: first-spike latency [median, Q1, Q3, min, max] in ms (healthy flies are 2-3 ms).")
put("gf_nonfail_affected_latency_median_ms", r2(A[A.GF_loom_hz >= GF_FAIL_HZ].GF_latency_ms.median()),
    "Median first-spike latency (ms) of affected flies at or above 100 Hz.")
mid = A[(A.GF_loom_hz >= 60) & (A.GF_loom_hz < 100)].sort_values("GF_loom_hz")
put("gf_affected_between_60_and_100hz", [[r2(x.GF_loom_hz), x.stage, int(x.seed), r2(x.GF_latency_ms)] for _, x in mid.iterrows()],
    "Affected flies with giant fibre rate in [60, 100) Hz: [rate Hz, s, seed, latency ms].")
bands = []
hmin = gfH.min()
for s in SER_ST:
    g = sc[sc.stage == s].GF_loom_hz
    bands.append([int((g < 100).sum()), int(((g >= 100) & (g < hmin)).sum()), int((g >= hmin).sum())])
put("gf_bands_by_severity", bands,
    "Severity series, per s = 0.05-1.0: [n below 100 Hz, n from 100 Hz to below the lowest healthy value, n at or above the lowest healthy value] (of 32).")
put("gf_bands_total", [int((A.GF_loom_hz < 100).sum()), int(((A.GF_loom_hz >= 100) & (A.GF_loom_hz < hmin)).sum()), int((A.GF_loom_hz >= hmin).sum())],
    "All 320 affected flies: [below 100 Hz, 100 Hz to below healthy minimum, within healthy range].")
put("gf_failing_median_by_severity", [r2(sc[(sc.stage == s) & (sc.GF_loom_hz < 100)].GF_loom_hz.median())
                                       if ((sc.stage == s) & (sc.GF_loom_hz < 100)).any() else None for s in SER_ST],
    "Median giant fibre rate (Hz) of the flies below 100 Hz, per severity s = 0.05-1.0 (null = no such fly).")
h_ok = gfH[gfH >= 100]
put("gf_nonfail_median_by_severity", [r2(sc[(sc.stage == s) & (sc.GF_loom_hz >= 100)].GF_loom_hz.median()) for s in SER_ST],
    "Median giant fibre rate (Hz) of flies at or above 100 Hz, per severity s = 0.05-1.0 (healthy median 195).")
put("gf_nonfail_auc_by_severity", [r4(auc(h_ok, sc[(sc.stage == s) & (sc.GF_loom_hz >= 100)].GF_loom_hz))
                                   if ((sc.stage == s) & (sc.GF_loom_hz >= 100)).sum() > 2 else None for s in SER_ST],
    "AUC of giant fibre rate against healthy using only flies at or above 100 Hz, per severity s = 0.05-1.0.")
g10 = sc[sc.stage == 0.1].GF_loom_hz
put("gf_s0.1_auc_all_vs_nonfail", [r4(auc(gfH, g10)), r4(auc(h_ok, g10[g10 >= 100]))],
    "Giant fibre AUC at s = 0.10: [all 32 flies, only flies at or above 100 Hz].")
# output neuron reconstruction (copied)
lr = ONL["gf_fail_by_left_right"]
put("gf_recon_left_removed_fail", [ONL["gf_fail_by_left_removed"]["True|True"],
                                   ONL["gf_fail_by_left_removed"]["True|True"] + ONL["gf_fail_by_left_removed"]["True|False"]],
    "Reconstruction (results/paper/output_neuron_loss.json): [flies below 100 Hz, flies] among affected flies that lost the left (stimulated-side) giant fibre.")
put("gf_recon_left_kept_fail", [ONL["gf_fail_by_left_removed"]["False|True"],
                                ONL["gf_fail_by_left_removed"]["False|True"] + ONL["gf_fail_by_left_removed"]["False|False"]],
    "Reconstruction: [flies below 100 Hz, flies] among affected flies that kept the left giant fibre.")
put("gf_recon_by_loss", {"left_only": [lr["True/False|True"], lr["True/False|True"] + lr["True/False|False"]],
                         "both": [lr["True/True|True"], lr["True/True|True"] + lr["True/True|False"]],
                         "right_only": [lr["False/True|True"], lr["False/True|True"] + lr["False/True|False"]],
                         "neither": [lr["False/False|True"], lr["False/False|True"] + lr["False/False|False"]]},
    "Reconstruction: [flies below 100 Hz, total flies] by which giant fibres were removed (left only, both, right only, neither); output_neuron_loss.json gf_fail_by_left_right.")
put("gf_recon_right_removed_total", lr["False/True|True"] + lr["False/True|False"] + lr["True/True|True"] + lr["True/True|False"],
    "Reconstruction: affected flies that lost the right (unstimulated-side) giant fibre (right only plus both).")
put("gf_vulnerability_percentile_L_R", [r4(ONL["vulnerability_percentile"]["DNp01_L"]), r4(ONL["vulnerability_percentile"]["DNp01_R"])],
    "Vulnerability (load) percentile of the left and right giant fibre (DNp01) among neurons (output_neuron_loss.json).")
put("mn9_recon_silent_by_n_removed", {k2: [ONL["mn9_silent_by_n_removed"].get(f"{k2}|True", 0),
                                           ONL["mn9_silent_by_n_removed"].get(f"{k2}|True", 0) + ONL["mn9_silent_by_n_removed"].get(f"{k2}|False", 0)]
                                      for k2 in ("0", "1", "2")},
    "Reconstruction: [silent MN9 flies, total flies] by number of MN9 neurons removed (0, 1, 2) among the 320 affected flies.")
put("mn9_n_silent_affected", int((A.MN9_sugar_hz == 0).sum()), "Severity series: affected flies with silent MN9 (rate 0).")
put("mn9_nonsilent_median_by_severity", [r2(sc[(sc.stage == s) & (sc.MN9_sugar_hz > 0)].MN9_sugar_hz.median()) for s in [0.0] + SER_ST],
    "Median MN9 rate (Hz) among non-silent flies at s = 0 and 0.05-1.0.")
mn_ok = sc[sc.MN9_sugar_hz > 0]
gf_ok = sc[sc.GF_loom_hz >= 100]
put("motor_nonfail_trend_rho_incl_healthy", [r3(spear(gf_ok.stage, gf_ok.GF_loom_hz)[0]), r3(spear(mn_ok.stage, mn_ok.MN9_sugar_hz)[0])],
    "Spearman rho of severity with [giant fibre rate among flies >= 100 Hz, MN9 rate among non-silent flies], severity series, healthy flies included.")
gf_a, mn_a = gf_ok[gf_ok.stage > 0], mn_ok[mn_ok.stage > 0]
put("motor_nonfail_trend_affected_only", [[r3(spear(gf_a.stage, gf_a.GF_loom_hz)[0]), sig(spear(gf_a.stage, gf_a.GF_loom_hz)[1]), len(gf_a)],
                                          [r3(spear(mn_a.stage, mn_a.MN9_sugar_hz)[0]), sig(spear(mn_a.stage, mn_a.MN9_sugar_hz)[1]), len(mn_a)]],
    "Severity series, affected flies only: [[rho, p, n] of severity with GF rate among flies >= 100 Hz], [[rho, p, n] with MN9 rate among non-silent flies].")
mh_ok = H.MN9_sugar_hz[H.MN9_sugar_hz > 0]
put("mn9_nonsilent_auc_by_severity", [r4(auc(mh_ok, sc[(sc.stage == s) & (sc.MN9_sugar_hz > 0)].MN9_sugar_hz)) for s in SER_ST],
    "AUC of MN9 rate against healthy using only non-silent flies, per severity s = 0.05-1.0.")

# ============================================================================= H. joint cohort: arena escape and feeding
# Arena take-off rule, parsed from code so the caveats below follow the code.
hn = (CODE / "adfly/house_nb.py").read_text(encoding="utf-8")
hs = (CODE / "adfly/house.py").read_text(encoding="utf-8")
lif_src = (CODE / "adfly/lif.py").read_text(encoding="utf-8")
shadow = float(re.search(r"shadow_dur: float = ([\d.]+)", hs).group(1))
gfsp = int(re.search(r"gf_spikes: int = (\d+)", hs).group(1))
gfwin = int(re.search(r"gf_spikes: int = \d+\s+#[^\n]*?in (\d+) ms", hs).group(1))
feedhz = float(re.search(r"feed_hz: float = ([\d.]+)", hs).group(1))
jump_line = re.search(r"if grounded and gf_sp >= gf_need and tsec - last_jump > ([\d.]+):", hn)
refr = float(jump_line.group(1))
assert "shadow" not in jump_line.group(0)                       # the take-off test does not look at threats
assert 'self.out_id[cx.idx(["DNp01"])] = 1' in hn               # both giant fibres feed one spike counter
assert re.search(r"gf_sp = ring\[b, w, 1\] \+ \(ring\[b, \(k - 1\) % RW, 1\][^\n]*\(ring\[b, \(k - 2\) % RW, 1\]", hn)
body_ms = int(re.search(r"\n\s+spb = (\d+)", hn).group(1)) * float(re.search(r"dt: float = ([\d.]+)\s+# ms", lif_src).group(1))
assert abs(3 * body_ms - gfwin) < 1e-9, (body_ms, gfwin)
thr_on = sorted(t[0] for t in JMETA["threats"])
thr_gap = float(np.min(np.diff(thr_on)))
assert thr_gap > shadow and int(np.floor(shadow / refr)) + 1 == 1
JUMP_NOTE = (f" Caveat: a jump is counted at any take-off, not only during a threat. house_nb.py takes off whenever the fly is grounded, "
             f"the giant fibre spikes summed over BOTH DNp01 neurons in the last {gfwin:g} ms (three {body_ms:g}-ms body steps) reach {gfsp}, and "
             f"more than {refr:g} s has passed since the last jump. A threat window lasts {shadow:g} s and threat onsets are at least "
             f"{thr_gap:.1f} s apart, so at most one jump fits in the window of a threat met; every jump beyond the number of threats met is a "
             f"take-off outside those windows, and a jump inside a window need not be a response to it. Jumps per threat is therefore not "
             f"escape success.")
jt = jd[jd.behaviour_threats_seen > 0].copy()
jt["jpt"] = jt.behaviour_jumps / jt.behaviour_threats_seen
jf = jd[jd.GF_loom_hz < GF_FAIL_HZ]
jft = jf[jf.behaviour_threats_seen > 0]
put("joint_gf_below100_n", len(jf), "Joint cohort: flies with circuit giant fibre rate below 100 Hz.")
put("joint_gf_below100_met_threat_n", len(jft), "Joint cohort: flies below 100 Hz that met at least one arena threat.")
put("joint_gf_below100_jpt_eq1_n", int((jft.behaviour_esc == 1.0).sum()),
    "Joint cohort: flies below 100 Hz that met a threat and jumped exactly once per threat (jumps per threat = 1.0). The 100 Hz criterion is "
    f"the circuit mean rate of the two giant fibres, but arena take-off needs only {gfsp} spikes summed over both giant fibres in {gfwin:g} ms, "
    "so one surviving giant fibre can still trigger take-off." + JUMP_NOTE)
put("joint_gf_below100_jpt_eq1_gf_range_hz", [r2(jft[jft.behaviour_esc == 1.0].GF_loom_hz.min()), r2(jft[jft.behaviour_esc == 1.0].GF_loom_hz.max())],
    "Joint cohort: range of circuit giant fibre rate (Hz) among the below-100-Hz flies that jumped once per threat.")
z = jft[jft.behaviour_esc < 1]
put("joint_gf_below100_never_jumped", [[x.stage, r2(x.GF_loom_hz), int(x.behaviour_jumps), int(x.behaviour_threats_seen)] for _, x in z.iterrows()],
    "Joint cohort: below-100-Hz flies with fewer jumps than threats: [s, GF rate Hz, jumps, threats seen].")
put("joint_n_met_threat", len(jt), "Joint cohort: flies that met at least one threat (jumps per threat defined).")
put("joint_n_fewer_jumps_than_threats", int((jt.behaviour_jumps < jt.behaviour_threats_seen).sum()),
    "Joint cohort: flies (of those that met a threat) with fewer jumps than threats met." + JUMP_NOTE)
put("joint_fewer_jumps_flies_gf_hz", [r2(v) for v in jt[jt.behaviour_jumps < jt.behaviour_threats_seen].GF_loom_hz],
    "Circuit giant fibre rate (Hz) of the joint-cohort flies with fewer jumps than threats.")
nothreat = jd[~(jd.behaviour_threats_seen > 0)]
put("joint_no_threat_flies", [[x.stage, int(x.behaviour_jumps), r2(x.GF_loom_hz)] for _, x in nothreat.iterrows()],
    "Joint-cohort flies that met no threat: [s, total jumps, circuit giant fibre rate Hz].")
dec = []
for s in [0.0] + JST:
    g = jt[jt.stage == s].jpt
    dec.append([s, len(g), int((g > 1).sum()), int((g == 1).sum()), int((g < 1).sum()), r3(g.median()), r3(g.mean())])
put("joint_jpt_decomposition_by_severity", dec,
    "Joint cohort, per s (0 and 0.05-0.50), flies that met a threat: [s, n, n with >1 jump per threat, n with exactly 1, n with <1, median, mean jumps per threat]."
    + JUMP_NOTE)
r_ = spear(jt.stage, jt.jpt)
put("joint_jpt_vs_severity", [r3(r_[0]), sig(r_[1]), r_[2]], "Joint cohort: Spearman rho, p, n of severity with jumps per threat." + JUMP_NOTE)
r_ = spear(jt.stage, jt.jpt.clip(upper=1))
put("joint_jpt_capped1_vs_severity", [r3(r_[0]), sig(r_[1]), r_[2]], "Joint cohort: Spearman rho, p, n of severity with jumps per threat capped at 1 (extra jumps removed)." + JUMP_NOTE)
r_ = spear(jt.stage, (jt.jpt > 1).astype(float))
put("joint_jpt_gt1_indicator_vs_severity", [r3(r_[0]), sig(r_[1]), r_[2]],
    "Joint cohort: Spearman rho, p, n of severity with the indicator 'more than one jump per threat', i.e. at least one take-off outside the windows of the threats met." + JUMP_NOTE)
r_ = spear(jd.stage, jd.behaviour_threats_seen)
put("joint_threats_seen_vs_severity", [r3(r_[0]), sig(r_[1]), r_[2]], "Joint cohort: Spearman rho, p, n of severity with number of threats met.")
put("joint_extra_jumps_mean_by_severity", [r2((jt[jt.stage == s].behaviour_jumps - jt[jt.stage == s].behaviour_threats_seen).mean()) for s in [0.0] + JST],
    "Joint cohort: mean (jumps minus threats met) per fly, at s = 0 and 0.05-0.50, flies that met a threat." + JUMP_NOTE)
beyond = []
for s in [0.0] + JST:
    g = jd[jd.stage == s]
    ex = (g.behaviour_jumps - g.behaviour_threats_seen).clip(lower=0)
    beyond.append([s, len(g), int((ex > 0).sum()), int(ex.sum()), int(g.behaviour_jumps.sum())])
put("joint_jumps_beyond_threats_met_by_severity", beyond,
    "Joint cohort, all flies (including those that met no threat), per s = 0 and 0.05-0.50: [s, n flies, n flies with more jumps than threats met, "
    "total jumps beyond threats met (sum of max(jumps - threats met, 0)), total jumps]. Jumps beyond threats met are a lower bound on take-offs "
    "outside the windows of the threats met." + JUMP_NOTE)
put("joint_threat_min_gap_s", r2(thr_gap),
    f"Joint-cohort arena: smallest interval (s) between threat onsets in results/joint/cohort_joint_meta.json ({len(thr_on)} threats); "
    f"larger than the {shadow:g} s threat window and the {refr:g} s jump refractory period.")
m9 = jd[jd.MN9_sugar_hz == 0]
put("joint_mn9_silent_feed", [len(m9), int((m9.behaviour_feed_s == 0).sum())],
    "Joint cohort: [flies with silent MN9, of these flies with zero arena feeding time].")
hf = JH[JH.behaviour_feed_s == 0]
put("joint_healthy_never_fed", [len(hf), len(JH)], "Joint cohort: [healthy flies with zero feeding time, healthy flies].")
put("joint_healthy_never_fed_mn9_hz", [r2(v) for v in hf.MN9_sugar_hz], "MN9 rates (Hz) of the healthy joint-cohort flies that never fed (arena feeding threshold 10 Hz).")
ht = JH.behaviour_threats_seen
put("joint_healthy_threats_seen", [r2(ht.median()), int(ht.min()), int(ht.max())], "Joint-cohort healthy flies: threats met [median, min, max].")
put("joint_healthy_jpt_exactly1", [int((JH.behaviour_esc == 1.0).sum()), int(JH.behaviour_esc.notna().sum())],
    "Joint-cohort healthy flies: [n with jumps per threat exactly 1.0, n with a defined value]." + JUMP_NOTE)
for c, b_, nm in (("MN9_sugar_hz", "behaviour_feed_s", "MN9_vs_feedTime"), ("GF_loom_hz", "behaviour_esc", "GF_vs_jumpsPerThreat")):
    r_ = spear(jd[c], jd[b_])
    pr = partial_spearman(jd[c], jd[b_], jd.stage)
    rh = spear(JH[c], JH[b_])
    put(f"joint_withinfly_{nm}_raw", [r3(r_[0]), sig(r_[1]), r_[2]], f"Joint cohort, all flies: Spearman rho, p, n of {LABEL[c]} with {LABEL[b_]} in the same fly.")
    put(f"joint_withinfly_{nm}_partial_severity", [r3(pr[0]), sig(pr[1]), pr[2]],
        f"Joint cohort: partial rank correlation of {LABEL[c]} with {LABEL[b_]} controlling for severity rank (rho, p, n).")
    put(f"joint_withinfly_{nm}_healthy", [r3(rh[0]), sig(rh[1]), rh[2]], f"Joint cohort healthy flies only: Spearman rho, p, n of {LABEL[c]} with {LABEL[b_]}.")
put("joint_withinfly_MN9_vs_feedTime_by_severity", [r2(spear(jd[jd.stage == s].MN9_sugar_hz, jd[jd.stage == s].behaviour_feed_s)[0]) for s in [0.0] + JST],
    "Joint cohort: within-severity Spearman rho of MN9 rate with feeding time at s = 0 and 0.05-0.50.")
pa = pd.read_csv(RES / "joint/paired_auc.csv")
put("joint_pairs_circuit_better_copied", [SJ["n_pairs_circuit_better"], SJ["n_pairs_tested"], int((pa.ci_hi < 0).sum())],
    "analyze_joint.py all-pairs design (12 circuit x 4 arena readouts x 6 severities): [pairs where circuit AUC is resolved better, pairs tested, pairs where arena is resolved better] (stats_joint.json, paired_auc.csv).")
# web demo and other cohorts
demo = json.loads((RES / "house_20v20_s05_full.json").read_text())
cfg = demo["config"]
put("webdemo_config", [cfg["healthy"], cfg["ad"], cfg["stage"], cfg["seconds"], len(demo["threats"]), cfg["seed"], len(demo["frames"]), demo["dt_frame"]],
    "Web-demo arena run (results/house_20v20_s05_full.json): [healthy flies, s>0 flies, severity, seconds, threats, seed, frames, frame interval s].")
dfl = pd.read_csv(RES / "house_20v20_s05_flies.csv")
put("webdemo_feed_s_median_healthy_vs_s0.5", [r2(dfl[dfl.group == "healthy"].feed_s.median()), r2(dfl[dfl.group == "AD"].feed_s.median()),
                                               sig(mw(dfl[dfl.group == "healthy"].feed_s, dfl[dfl.group == "AD"].feed_s))],
    "Web-demo run: median feeding time (s) of healthy and s = 0.5 flies, and Mann-Whitney p (results/house_20v20_s05_flies.csv).")
put("joint_arena_config", [JMETA["seconds"], len(JMETA["threats"]), JMETA["seed"], JMETA["stage_list"]],
    "Joint-cohort arena (results/joint/cohort_joint_meta.json): [seconds, threats scheduled, seed, stage list].")
pilot_n = sum(len(pd.read_csv(BAT / f"{n}.csv")) for n in ("stage_curve", "components", "vulnerability", "scope"))
put("cohorts_reported_total", [BSTATS["totals"]["all"], 288, BSTATS["totals"]["all"] + 288],
    "Flies in the reported designs: [series+ablation+maps+KC-only+joint (battery_stats.json totals.all), burden-matched exp2 maps (288, results/exp2), total].")
smoke = pd.read_csv(RES / "joint/smoke.csv")
smoke_meta = json.loads((RES / "joint/smoke_meta.json").read_text())
assert len(smoke) == smoke_meta["n_flies"]
put("cohorts_unreported", {"pilot_battery": pilot_n, "learning_cohort": len(pd.read_csv(RES / "learning/learning_by_stage.csv")),
                           "early_arena": len(pd.read_csv(RES / "paper/house_early_flies.csv")), "web_demo": len(dfl),
                           "joint_smoke_test": len(smoke)},
    "Flies in simulated cohorts not counted in the reported total: the earlier run of the four battery designs, called the pilot "
    "(results/battery/*.csv without _x4), the learning cohort, the early arena cohort, the web demo, and the technical test of the "
    f"joint-cohort script (results/joint/smoke.csv and smoke_meta.json: {len(smoke)} flies, stage list {smoke_meta['stage_list']}, "
    f"{smoke_meta['seconds']:g}-s arena).")

# ============================================================================= I. high-activity flies
kmax, mmax = H.KC_hz_A.max(), H.MBON_hz_A.max()
kmed, mmed = H.KC_hz_A.median(), H.MBON_hz_A.median()
put("healthy_KC_max_median_hz", [r4(kmax), r4(kmed)], "Severity series healthy flies: [maximum, median] Kenyon cell rate (Hz).")
put("healthy_MBON_max_median_hz", [r4(mmax), r4(mmed)], "Severity series healthy flies: [maximum, median] MBON rate (Hz).")
grp_c, ref_c, ref15, fl_kmax, exp_, binp, ratio_med = [], [], [], [], [], [], []
flagged_all = []
for s in SER_ST:
    g = sc[sc.stage == s]
    fl = (g.KC_hz_A > 1.5 * g.KC_hz_A.median()) & (g.MBON_hz_A > 1.5 * g.MBON_hz_A.median())
    flagged_all.append(g[fl])
    grp_c.append(int(fl.sum()))
    ref_c.append(int(((g.KC_hz_A > kmax) & (g.MBON_hz_A > mmax)).sum()))
    ref15.append(int(((g.KC_hz_A > 1.5 * kmed) & (g.MBON_hz_A > 1.5 * mmed)).sum()))
    fl_kmax.append(int((g[fl].KC_hz_A > kmax).sum()))
    p = 1 - (1 - 0.5 * s) ** 2          # analyze_battery.py: P(at least one of two maximally loaded neurons removed)
    exp_.append(round(p * len(g), 2))
    binp.append(sig(st.binom.cdf(int(fl.sum()), len(g), p)))
    ratio_med.append(r2((g[fl].KC_hz_A / kmed).median()) if fl.any() else None)
put("hiact_groupmedian_criterion_by_severity", grp_c,
    "Severity series, s = 0.05-1.0: flies (of 32) with KC and MBON rates both above 1.5 x their own severity group's median (own-group criterion, open bars of Fig 6F).")
put("hiact_healthy_referenced_KCmax_and_MBONmax_by_severity", ref_c,
    "Severity series, s = 0.05-1.0: flies (of 32) with KC rate above the healthy maximum AND MBON rate above the healthy maximum.")
put("hiact_healthy_referenced_1.5x_healthy_median_by_severity", ref15,
    "Severity series, s = 0.05-1.0: flies (of 32) with KC and MBON rates both above 1.5 x the healthy median.")
put("hiact_groupmedian_flagged_with_KC_above_healthy_max", fl_kmax,
    "Severity series, s = 0.05-1.0: of the group-median-flagged flies, number with KC rate above the healthy maximum.")
put("hiact_groupmedian_flagged_KC_over_healthy_median", ratio_med,
    "Severity series, s = 0.05-1.0: median KC rate of the group-median-flagged flies as a multiple of the healthy median KC rate.")
put("hiact_expected_one_of_two_top_hubs", exp_,
    "Expected flies (of 32) losing at least one of two maximally loaded neurons, 32 x [1 - (1 - 0.5 s)^2] (analyze_battery.py expected_if_one_of_two_top_hubs), s = 0.05-1.0.")
put("hiact_binomial_P_le_observed", binp,
    "Binomial P(X <= observed group-median count) with n = 32 and p = 1 - (1 - 0.5 s)^2, s = 0.05-1.0.")
for nm, lo, hi in (("le0.15", 0, 0.15), ("0.2_to_0.4", 0.2, 0.4), ("ge0.5", 0.5, 1.0)):
    idx = [i for i, s in enumerate(SER_ST) if lo <= s <= hi]
    sv = [SER_ST[i] for i in idx]
    put(f"hiact_observed_vs_expected_{nm}", [sum(grp_c[i] for i in idx), round(sum(exp_[i] for i in idx), 2)],
        f"Severity series, affected flies at s = {', '.join(stg(x) for x in sv)} ({len(sv)} x 32 flies, no healthy flies): "
        f"[observed group-median-flagged flies, expected under the two-top-hub formula].")
kco = pd.read_csv(RES / "stats/kc_outliers_series.csv")
assert kco.n_both.tolist() == grp_c and np.allclose(kco.expected_if_one_of_two_top_hubs, exp_)
put("hiact_healthy_flagged", [int(((H.KC_hz_A > 1.5 * kmed) & (H.MBON_hz_A > 1.5 * mmed)).sum()), len(H)],
    "Healthy flies meeting the 1.5 x (healthy) median criterion for both KC and MBON: [n, total].")
fl_df = pd.concat(flagged_all)
exp1_seeds = sorted(int(p.name.split("seed")[1].split(".")[0]) for p in (RES / "exp1/screen").glob("*.done.json"))
put("hiact_flagged_seeds_in_exp1", [[x.stage, int(x.seed)] for _, x in fl_df[fl_df.seed.isin(exp1_seeds)].iterrows()],
    "Group-median-flagged high-activity flies that were also re-simulated in the exp1 counterfactual screen: [s, seed].")
ru = pd.read_csv(RES / "exp1/screen/s0.5_seed1266_runs.csv")
un = pd.read_csv(RES / "exp1/screen/s0.5_seed1266_units.csv")
apl = un[(un.level == "type") & (un.unit == "APL")].iloc[0]
put("exp1_seed1266_APL_units", [int(apl.n_lost), round(float(apl.mean_rank), 6), int(apl.syn_lost)],
    "Exp1 screen, fly s = 0.5 seed 1266 (a group-median-flagged fly), APL type row of s0.5_seed1266_units.csv: [number of APL neurons that died "
    "in this fly, mean vulnerability percentile rank of those neurons (cx.vuln = percentile of total synapse count among all neurons, 1 = most "
    "connected; column mean_rank), total input + output synapse count of those neurons in the intact connectome (column syn_lost = "
    "GraphPredictors.syn = out_syn + in_syn summed over the dead neurons; it counts the synapses of the removed neurons, not synapses removed "
    "by synapse thinning)].")


def run_row(unit):
    x = ru[ru.unit == unit].iloc[0]
    return [r3(x.KC_hz_A), r3(x.MBON_hz_A)]


put("exp1_seed1266_KC_MBON_hz", {"damaged": run_row("damaged"), "restore_APL_both": run_row("APL"),
                                 "restore_APL#10540": run_row("APL#10540"), "restore_APL#10977": run_row("APL#10977")},
    "Exp1 screen, seed 1266 at s = 0.5 (shortened screen protocol): [KC rate, MBON rate] in Hz for the damaged fly and after restoring both or one APL neuron (s0.5_seed1266_runs.csv).")
s1266 = sc[(sc.seed == 1266)].iloc[0]
g05 = sc[sc.stage == 0.5]
put("series_seed1266_KC_MBON_vs_group_median", [r3(s1266.KC_hz_A), r3(s1266.MBON_hz_A), r3(g05.KC_hz_A.median()), r3(g05.MBON_hz_A.median())],
    "Severity series fly seed 1266 (s = 0.5): [KC rate, MBON rate, s = 0.5 group median KC, group median MBON] (Hz).")
# rule ablation, KC excursions by criterion
abl = []
for c in ["all_s0.5", "hyper_only_s0.5", "death_only_s0.5", "synloss_only_s0.5", "no_hyper_s0.5",
          "all_s1", "hyper_only_s1", "death_only_s1", "synloss_only_s1", "no_hyper_s1"]:
    g = co[co.condition == c]
    flc = (g.KC_hz_A > 1.5 * g.KC_hz_A.median()) & (g.MBON_hz_A > 1.5 * g.MBON_hz_A.median())
    abl.append([c, int((g.KC_hz_A > kmax).sum()), int((g.KC_hz_A > 1.2 * kmax).sum()),
                int((g.KC_hz_A > 1.5 * kmed).sum()), int(flc.sum()), r3(g.KC_hz_A.max())])
put("ablation_KC_excursions", abl,
    "Rule ablation (components_x4.csv, 32 flies each): [condition, n KC > healthy max, n KC > 1.2 x healthy max, n KC > 1.5 x healthy median, n flagged by the group-median KC&MBON criterion, max KC Hz].")

# ============================================================================= J. AUC dip of mushroom body readouts
MB = ["KC_hz_A", "KC_frac_A", "MBON_hz_A"]
dip_st = [0.15, 0.2, 0.3, 0.4, 0.5]
for r in MB:
    a_all, a_wo = [], []
    for s in dip_st:
        g = sc[sc.stage == s]
        fl = (g.KC_hz_A > 1.5 * g.KC_hz_A.median()) & (g.MBON_hz_A > 1.5 * g.MBON_hz_A.median())
        a_all.append(r4(auc(H[r], g[r])))
        a_wo.append(r4(auc(H[r], g[~fl][r])))
    put(f"dip_auc_all_{SHORT[r]}", a_all, f"Severity series AUC of {LABEL[r]} at s = 0.15, 0.2, 0.3, 0.4, 0.5 with all 32 flies.")
    put(f"dip_auc_without_flagged_{SHORT[r]}", a_wo,
        f"Severity series AUC of {LABEL[r]} at s = 0.15, 0.2, 0.3, 0.4, 0.5 after removing the group-median-flagged high-activity flies.")
put("dip_MBON_pct_healthy_s0.3_s0.4", [pct_ci(sc[sc.stage == 0.3].MBON_hz_A, H.MBON_hz_A, "mbon_0.3"),
                                       pct_ci(sc[sc.stage == 0.4].MBON_hz_A, H.MBON_hz_A, "mbon_0.4")],
    "MBON rate median as % of healthy with 95% bootstrap interval at s = 0.3 and s = 0.4 ([pct, lo, hi] each).")
put("dip_n_above_healthy_max_s0.4", [int((sc[sc.stage == 0.4].KC_hz_A > kmax).sum()), int((sc[sc.stage == 0.4].MBON_hz_A > mmax).sum())],
    "Severity series s = 0.4: [flies with KC rate above healthy max, flies with MBON rate above healthy max].")

# ============================================================================= K. test-retest (pilot vs x4)
RT = ["brain_spikes_A", "LH_hz_A", "PN_hz_A", "KC_hz_A", "KC_frac_A", "MBON_hz_A", "V_pre_A", "LI",
      "MN9_sugar_hz", "GF_loom_hz"]
pairs = []
for n in ("stage_curve", "components", "vulnerability", "scope"):
    a = pd.read_csv(BAT / f"{n}.csv")
    b_ = pd.read_csv(BAT / f"{n}_x4.csv")
    m = a.merge(b_, on=["seed", "condition", "stage", "scope"], suffixes=("_p", "_x"))
    pairs.append(m)
pm = pd.concat(pairs, ignore_index=True)
put("retest_n_pairs_by_condition", pm.condition.value_counts().sort_index().to_dict(),
    "Pilot battery flies re-simulated in the *_x4 tables with the same seed and condition, by condition. Damaged pairs are the same pathology "
    "under new input noise. Healthy pairs carry no pathology (healthy_pathology_all_zero; pilot and x4 healthy dead_frac = mean_load = 0), so "
    "every healthy run is the identical intact network and a healthy 'pair' is two runs of that one network under different input noise; "
    "the seed gives a healthy fly no individual identity.")
put("retest_healthy_pathology_zero", [int(((pm.condition == "healthy") & (pm.dead_frac_p == 0) & (pm.dead_frac_x == 0)
                                            & (pm.mean_load_p == 0) & (pm.mean_load_x == 0)).sum()), int((pm.condition == "healthy").sum())],
    "Test-retest healthy pairs with zero pathology in both the pilot and the x4 run (dead_frac and mean_load = 0): [n, healthy pairs].")
put("retest_max_abs_dead_frac_difference", float((pm.dead_frac_p - pm.dead_frac_x).abs().max()),
    "Largest absolute difference in fraction of neurons removed between pilot and x4 runs of the same fly (0 = identical pathology).")


def retest(m):
    out = {}
    for r in RT:
        x, y = m[f"{r}_p"].astype(float), m[f"{r}_x"].astype(float)
        rho = st.spearmanr(x, y)[0]
        within = np.sqrt(np.mean((x - y) ** 2) / 2)
        total = np.concatenate([x, y]).std(ddof=1)
        out[r] = [r2(rho), r2(within / total) if total > 0 else None]
    return out


groups = {"healthy": pm[pm.condition == "healthy"],
          "s0.5_pan": pm[pm.condition.isin(["all_s0.5", "hub_s0.5"])],
          "s0.5_KConly": pm[pm.condition == "mb_s0.5"]}
RT_DESC = {"healthy": lambda n: f"{n} pairs of runs of the identical healthy network under different input noise (pilot vs x4 run)",
           "s0.5_pan": lambda n: f"{n} damaged flies at s = 0.5, pan scope (all_s0.5 and hub_s0.5; pilot vs x4 run, same seed and pathology, new input noise)",
           "s0.5_KConly": lambda n: f"{n} damaged flies at s = 0.5, Kenyon-cell-only scope (mb_s0.5; pilot vs x4 run, same seed and pathology, new input noise)"}
RT_CAVEAT = {"healthy": (" Healthy flies have no pathology (retest_healthy_pathology_zero), so there is no individual to retest: rho near 0 and a "
                         "within/total SD ratio near 1 follow by construction and are NOT a measured low reliability of healthy individuals.")}
for gname, m in groups.items():
    rt = retest(m)
    for r in RT:
        put(f"retest_{gname}_{SHORT[r]}", rt[r],
            f"Test-retest of {LABEL[r]} in {RT_DESC[gname](len(m))}: [Spearman rho, within-pair SD / total SD] (within-pair SD = sqrt(mean diff^2 / 2); "
            f"total SD over both runs).{RT_CAVEAT.get(gname, '')}")

# ============================================================================= L. joint-cohort protocol (from code)
cj = (CODE / "scripts/cohort_joint.py").read_text(encoding="utf-8")
ab = (CODE / "scripts/assay_battery.py").read_text(encoding="utf-8")
markers = ["preA = seg(", "preB = seg(", "sug = seg(", "gfb = seg(", "br.reset_state()", "house.probe_naive()",
           "house.train()", "postA = seg(", "postB = seg(", "house.run("]
pos = [cj.find(mk) for mk in markers]
assert all(p_ >= 0 for p_ in pos) and pos == sorted(pos), "joint protocol order changed"
assert cj.count("house.train(") == 1 and "learn(" not in cj.replace("house.train", "")
segs = re.findall(r"(preA|preB|sug|gfb|postA|postB) = seg\([^,]+,[^,]+(?:\([^)]*\))?,\s*(\d+)", cj)
dur = {k: int(v) for k, v in segs}
gaps = [int(v) for v in re.findall(r"gap\((\d+)\)", cj)]
pn_sig = re.search(r"def probe_naive\(self, levels=\(([^)]*)\), ms=([\d.]+), gap_ms=([\d.]+)", hn)
tr_sig = re.search(r"def train\(self, eta=([\d.]+)", hn)
tr_doc = re.search(r'def train[^\n]*\n\s*"""([^"]+)"""', hn).group(1).strip()
run_reset = "br.reset_state()" in hn[hn.find("def run(self"):]
seq = [f"odour A {dur['preA']} ms (pre-training valence V_pre)", f"gap {gaps[0]} ms",
       f"odour B {dur['preB']} ms", f"gap {gaps[1]} ms", f"sugar {dur['sug']} ms (MN9)", f"gap {gaps[2]} ms",
       f"left looming {dur['gfb']} ms (giant fibre)", f"gap {gaps[3]} ms",
       "br.reset_state(): v, g, adaptation, refractory and delay buffers reset; weights kept",
       f"probe_naive: odour A then B at {pn_sig.group(1)} x 150 Hz, {float(pn_sig.group(2)):g} ms each, {float(pn_sig.group(3)):g} ms gaps (arena baseline only)",
       f"train (once, eta {tr_sig.group(1)}): {tr_doc}",
       f"memory test: odour A {dur['postA']} ms, gap {gaps[4]} ms, odour B {dur['postB']} ms (V_post)",
       f"arena {JMETA['seconds']:g} s with {len(JMETA['threats'])} threats (reset_state at arena start: {run_reset})"]
put("protocol_joint_sequence", seq,
    "Joint-cohort per-fly protocol in execution order, parsed from code/scripts/cohort_joint.py, code/adfly/house_nb.py (probe_naive, train, run) and results/joint/cohort_joint_meta.json.")
put("protocol_joint_LI_uses_prereset_valence", bool(re.search(r"vpa, vpb, vqa, vqb = \(valence\(cx, r\) for r in \(preA, preB, postA, postB\)\)", cj)),
    "True if the joint-cohort learning index pairs pre-training valences from step 1 (before the state reset) with post-training valences (cohort_joint.py).")
put("protocol_series_has_reset_or_probe", ("reset_state" in ab) or ("probe_naive" in ab),
    "Whether the severity-series protocol (code/scripts/assay_battery.py) contains a state reset or naive-valence probe (False = one continuous 5-step run).")
put("arena_constants", [shadow, refr, gfsp, gfwin, feedhz],
    "Arena constants from code: [shadow (threat) duration s (house.py Arena.shadow_dur), jump refractory s (house_nb.py), giant fibre spikes "
    "needed for take-off, summed over BOTH giant fibres (DNp01 left and right share out_id 1 in house_nb.py) (house.py Body.gf_spikes), "
    f"take-off window ms = the last three {body_ms:g}-ms body steps (house_nb.py gf_sp), MN9 feeding threshold Hz (Body.feed_hz)]. "
    "The take-off test does not depend on a threat being present." + JUMP_NOTE)
put("arena_max_jumps_per_threat_window", int(np.floor(shadow / refr)) + 1,
    f"Maximum number of take-offs possible during one threat window, given the {shadow:g} s shadow duration and the {refr:g} s jump refractory period. "
    "Jumps are counted at any grounded giant fibre burst, so jumps beyond the number of threats met are take-offs outside threat windows." + JUMP_NOTE)

# ----------------------------------------------------------------------------- write
assert set(F) == set(M)
F["_meaning"] = M
F["_provenance"] = {"script": "code/scripts/facts_detection.py", "seed": SEED, "n_boot": N_BOOT,
                    "inputs": ["results/battery/stage_curve_x4.csv", "results/battery/components_x4.csv",
                               "results/battery/vulnerability_x4.csv", "results/battery/scope_x4.csv",
                               "results/battery/{stage_curve,components,vulnerability,scope}.csv (pilot)",
                               "results/joint/cohort_joint.csv", "results/joint/cohort_joint_meta.json",
                               "results/joint/stats_joint.json", "results/joint/paired_auc.csv",
                               "results/stats/battery_stats.json", "results/stats/series_paired_vs_LI.csv",
                               "results/stats/joint_vs_arena.csv", "results/stats/kc_outliers_series.csv",
                               "results/paper/output_neuron_loss.json", "results/exp1/screen/s0.5_seed1266_*.csv",
                               "results/house_20v20_s05_full.json", "results/house_20v20_s05_flies.csv",
                               "results/learning/learning_by_stage.csv", "results/paper/house_early_flies.csv",
                               "results/joint/smoke.csv", "results/joint/smoke_meta.json",
                               "code/scripts/cohort_joint.py", "code/scripts/assay_battery.py",
                               "code/adfly/house_nb.py", "code/adfly/house.py", "code/adfly/lif.py"],
                    "first_detection_replicates": {"n_rep": N_REP, "near": NEAR,
                                                   "rule": "replicate seeds rng([SEED, crc32(tag + '_rep' + j)]) run when a cumulative crossing share is within NEAR of 0.5"}}
(OUT / "facts_detection.json").write_text(json.dumps(F, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT / 'facts_detection.json'} with {len(M)} facts")
