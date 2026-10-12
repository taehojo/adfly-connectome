"""Statistics of the four battery designs and the joint cohort, from their per-fly result tables.

No simulation is run here. Every number comes from the per-fly tables that the
simulations wrote (results/battery/*_x4.csv, results/joint/cohort_joint.csv)
and from results/paper/output_neuron_loss.json.

  python code/scripts/analyze_battery.py

Writes results/stats/battery_stats.json and the csv tables that the figure scripts read.
What it computes:
  * medians and ratios to healthy, with bootstrap 95% intervals;
  * the three independent hub-map groups at s = 0.5 and s = 1.0, compared
    with each other and pooled where the design allows;
  * the precision argument for every readout, with the two motor
    readouts split into a failure fraction and a graded part;
  * Kenyon cell / MBON rate outliers at low severity, counted per group and set
    against the rule ablations;
  * the learning index against MBON drive in all flies (partial rank
    correlation), not only through the ratio that is undefined at high severity.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())  # code/scripts locally, scripts/ on the server
BAT = ROOT / "results/battery"
OUT = ROOT / "results/stats"
OUT.mkdir(parents=True, exist_ok=True)
N_BOOT = 4000
MBON_FLOOR = 0.5
RNG = np.random.default_rng(20261005)

READS = ["brain_spikes_A", "LH_hz_A", "PN_hz_A", "KC_hz_A", "KC_frac_A", "MBON_hz_A",
         "V_pre_A", "LI", "LI_norm", "MN9_sugar_hz", "GF_loom_hz", "GF_latency_ms"]
LABEL = {"brain_spikes_A": "Whole-brain spikes", "LH_hz_A": "Lateral horn rate",
         "PN_hz_A": "Projection neuron rate", "KC_hz_A": "Kenyon cell rate",
         "KC_frac_A": "Kenyon cells recruited", "MBON_hz_A": "MBON rate",
         "V_pre_A": "Naive MB valence", "LI": "Learning index",
         "LI_norm": "Learning index / MBON rate", "MN9_sugar_hz": "MN9 feeding drive",
         "GF_loom_hz": "Giant fibre rate", "GF_latency_ms": "Giant fibre latency"}


def prep(d):
    d = d.copy()
    d["LI_norm"] = d["LI"] / d["MBON_hz_A"].where(d["MBON_hz_A"] >= MBON_FLOOR)
    return d


def auc(a, b):
    """Direction-free AUC of group b against healthy a (Mann-Whitney statistic)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    u = (gt + 0.5 * eq) / (len(a) * len(b))
    return max(u, 1.0 - u)


def holm(p):
    o = sorted(p, key=lambda k: p[k])
    m, prev, out = len(o), 0.0, {}
    for i, k in enumerate(o):
        prev = min(1.0, max(prev, (m - i) * p[k]))
        out[k] = prev
    return out


def boot_ratio_of_medians(x, h, n=N_BOOT):
    """Median of x as a percentage of the median of h, with a bootstrap 95% interval."""
    x, h = np.asarray(pd.Series(x).dropna(), float), np.asarray(pd.Series(h).dropna(), float)
    if len(x) == 0 or len(h) == 0:
        return None
    est = np.median(x) / np.median(h) * 100
    bx = RNG.choice(x, (n, len(x)), replace=True)
    bh = RNG.choice(h, (n, len(h)), replace=True)
    r = np.median(bx, 1) / np.median(bh, 1) * 100
    lo, hi = np.percentile(r, [2.5, 97.5])
    return {"pct": round(float(est), 1), "lo": round(float(lo), 1), "hi": round(float(hi), 1),
            "median": float(np.median(x)), "n": int(len(x))}


def boot_median(x, n=N_BOOT):
    x = np.asarray(pd.Series(x).dropna(), float)
    b = np.median(RNG.choice(x, (n, len(x)), replace=True), 1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"median": float(np.median(x)), "lo": float(lo), "hi": float(hi), "n": int(len(x))}


def mw(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 3 or len(b) < 3:
        return None
    return float(st.mannwhitneyu(a, b, alternative="two-sided")[1])


def auc_boot(h, g, n=2000):
    h, g = np.asarray(pd.Series(h).dropna(), float), np.asarray(pd.Series(g).dropna(), float)
    v = np.array([auc(RNG.choice(h, len(h)), RNG.choice(g, len(g))) for _ in range(n)])
    return float(auc(h, g)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def partial_spearman(x, y, z):
    """Rank correlation of x and y after removing the linear effect of rank(z) from both."""
    rx, ry, rz = (st.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r, p = st.pearsonr(ex, ey)
    return float(r), float(p)


res = {}

# ---------------------------------------------------------------- severity series
sc = prep(pd.read_csv(BAT / "stage_curve_x4.csv"))
sc = sc[sc.scope == "pan"].copy()
H = sc[sc.stage == 0]
stages = sorted(s for s in sc.stage.unique() if s > 0)
res["series"] = {"n": int(len(sc)), "n_healthy": int(len(H)),
                 "n_by_stage": {str(k): int(v) for k, v in sc.groupby("stage").size().items()}}

# burden
res["series"]["burden"] = {str(s): {"dead_pct": round(100 * g.dead_frac.mean(), 2),
                                    "kc_dead_pct": round(100 * g.kc_dead_frac.mean(), 2),
                                    "dead_pct_range": [round(100 * g.dead_frac.min(), 2),
                                                       round(100 * g.dead_frac.max(), 2)]}
                           for s, g in sc.groupby("stage")}

# trend
raw, trend = {}, {}
for r in READS:
    v = sc[["stage", r]].dropna()
    rho, p = st.spearmanr(v.stage, v[r])
    raw[r] = float(p)
    trend[r] = {"rho": round(float(rho), 3), "p": float(p), "n": int(len(v))}
for r, q in holm(raw).items():
    trend[r]["p_holm"] = float(q)
res["series"]["trend"] = trend

# medians as % of healthy, per severity
tab = []
for r in READS:
    for s in [0.0] + stages:
        g = sc[sc.stage == s][r]
        b = boot_ratio_of_medians(g, H[r])
        if b:
            tab.append({"readout": r, "stage": s, **b})
pd.DataFrame(tab).to_csv(OUT / "series_pct_of_healthy.csv", index=False)

# AUC per severity, with bootstrap interval
rows = []
for r in READS:
    for s in stages:
        g = sc[sc.stage == s][r].dropna()
        if len(g) == 0:
            continue
        a, lo, hi = auc_boot(H[r], g)
        rows.append({"readout": r, "stage": s, "auc": round(a, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                     "mw_p": mw(H[r], g)})
pd.DataFrame(rows).to_csv(OUT / "series_auc.csv", index=False)

# first severity reaching AUC >= 0.80 (bootstrap over flies)
casc = {}
for r in READS:
    h = H[r].dropna().values
    per = {s: sc[sc.stage == s][r].dropna().values for s in stages}
    obs = next((s for s in stages if len(per[s]) and auc(h, per[s]) >= 0.80), None)
    boot = np.full(N_BOOT, np.nan)
    for i in range(N_BOOT):
        hb = RNG.choice(h, len(h))
        for s in stages:
            if len(per[s]) and auc(hb, RNG.choice(per[s], len(per[s]))) >= 0.80:
                boot[i] = s
                break
    ok = ~np.isnan(boot)
    casc[r] = {"observed": obs, "frac_never": round(float(1 - ok.mean()), 3),
               "boot_median": float(np.median(boot[ok])) if ok.any() else None,
               "boot_ci95": [float(np.percentile(boot[ok], 2.5)), float(np.percentile(boot[ok], 97.5))]
               if ok.any() else None}
res["series"]["first_detection"] = casc

# paired AUC difference against the learning index, same flies
pairs = []
for s in [x for x in stages if x <= 0.3]:
    S = sc[sc.stage == s]
    for a_ in ("brain_spikes_A", "LH_hz_A", "MBON_hz_A", "KC_hz_A", "MN9_sugar_hz", "GF_loom_hz"):
        diff = np.empty(N_BOOT)
        for i in range(N_BOOT):
            hi = RNG.integers(0, len(H), len(H))
            si = RNG.integers(0, len(S), len(S))
            diff[i] = auc(H[a_].values[hi], S[a_].values[si]) - auc(H["LI"].values[hi], S["LI"].values[si])
        lo_, hi_ = np.percentile(diff, [2.5, 97.5])
        pairs.append({"stage": s, "readout": a_, "auc_readout": round(auc(H[a_], S[a_]), 4),
                      "auc_LI": round(auc(H["LI"], S["LI"]), 4), "mean_diff": round(float(diff.mean()), 4),
                      "ci_lo": round(float(lo_), 4), "ci_hi": round(float(hi_), 4), "beats_LI": bool(lo_ > 0)})
pd.DataFrame(pairs).to_csv(OUT / "series_paired_vs_LI.csv", index=False)

# precision decomposition at s = 0.10 (and 0.05, 0.20)
prec = {}
for r in ["brain_spikes_A", "LH_hz_A", "PN_hz_A", "KC_hz_A", "KC_frac_A", "MBON_hz_A", "V_pre_A",
          "LI", "MN9_sugar_hz", "GF_loom_hz"]:
    h = H[r]
    e = {"healthy_cv": round(float(h.std() / abs(h.mean())), 4),
         "healthy_rcv": round(float((h.quantile(0.75) - h.quantile(0.25)) / 1.349 / abs(h.median())), 4)}
    for s in (0.05, 0.1, 0.2):
        g = sc[sc.stage == s][r]
        e[f"mean_shift_pct_{s:g}"] = round(float((g.mean() - h.mean()) / abs(h.mean()) * 100), 2)
        e[f"median_shift_pct_{s:g}"] = round(float((g.median() - h.median()) / abs(h.median()) * 100), 2)
        e[f"auc_{s:g}"] = round(float(auc(h, g)), 4)
    prec[r] = e
# the two motor readouts: failure fraction and the graded part among flies that did not fail
fail_def = {"MN9_sugar_hz": lambda x: x == 0, "GF_loom_hz": lambda x: x < 100}
motor = {}
for r, f in fail_def.items():
    m = {}
    for s in [0.0] + stages:
        g = sc[sc.stage == s][r]
        ok = g[~f(g)]
        m[str(s)] = {"n": int(len(g)), "n_fail": int(f(g).sum()),
                     "median_nonfail": float(ok.median()) if len(ok) else None,
                     "auc_nonfail": round(float(auc(H[r][~f(H[r])], ok)), 4) if s > 0 and len(ok) > 2 else None}
    motor[r] = m
res["series"]["precision"] = prec
res["series"]["motor_failures"] = motor
# graded readouts: does AUC at s = 0.10 follow |shift| / cv ?
graded = ["brain_spikes_A", "LH_hz_A", "PN_hz_A", "KC_hz_A", "KC_frac_A", "MBON_hz_A", "V_pre_A", "LI"]
z = [abs(prec[r]["median_shift_pct_0.1"]) / 100 / prec[r]["healthy_rcv"] for r in graded]
a10 = [prec[r]["auc_0.1"] for r in graded]
rho, p = st.spearmanr(z, a10)
res["series"]["precision_rank_corr"] = {"readouts": graded, "rho": round(float(rho), 3), "p": float(p),
                                        "standardised_median_shift": [round(x, 3) for x in z], "auc_0.1": a10}

# Kenyon cell / MBON outliers: flies far above their own severity group's median
out_rows = []
for s, g in sc.groupby("stage"):
    if s == 0:
        continue
    k = g.KC_hz_A > 1.5 * g.KC_hz_A.median()
    m = g.MBON_hz_A > 1.5 * g.MBON_hz_A.median()
    p_apl = 1 - (1 - 0.5 * s) ** 2           # P(at least one of two maximally loaded neurons removed)
    out_rows.append({"stage": s, "n": int(len(g)), "n_kc_high": int(k.sum()), "n_mbon_high": int(m.sum()),
                     "n_both": int((k & m).sum()), "expected_if_one_of_two_top_hubs": round(p_apl * len(g), 2),
                     "kc_median": round(float(g.KC_hz_A.median()), 3), "kc_mean": round(float(g.KC_hz_A.mean()), 3),
                     "kc_median_without_high": round(float(g.KC_hz_A[~k].median()), 3),
                     "kc_mean_without_high": round(float(g.KC_hz_A[~k].mean()), 3),
                     "mbon_mean": round(float(g.MBON_hz_A.mean()), 3),
                     "mbon_mean_without_high": round(float(g.MBON_hz_A[~k].mean()), 3),
                     "kc_ratio_high_to_median": round(float((g.KC_hz_A[k] / g.KC_hz_A.median()).median()), 2)
                     if k.any() else None})
pd.DataFrame(out_rows).to_csv(OUT / "kc_outliers_series.csv", index=False)
res["series"]["kc_outliers"] = out_rows
# healthy reference for the same criterion
res["series"]["kc_outliers_healthy"] = int((H.KC_hz_A > 1.5 * H.KC_hz_A.median()).sum())

# learning index against MBON drive
lo = sc[sc.stage <= 0.3]
res["series"]["learning"] = {
    "spearman_LI_MBON_all": [round(float(x), 4) for x in st.spearmanr(sc.LI, sc.MBON_hz_A)],
    "spearman_LI_MBON_healthy": [round(float(x), 4) for x in st.spearmanr(H.LI, H.MBON_hz_A)],
    "spearman_LI_Vpre_healthy": [round(float(x), 4) for x in st.spearmanr(H.LI, H.V_pre_A)],
    "partial_spearman_stage_LI_given_MBON_all": partial_spearman(sc.stage, sc.LI, sc.MBON_hz_A),
    "partial_spearman_stage_LI_given_MBON_le03": partial_spearman(lo.stage, lo.LI, lo.MBON_hz_A),
    "trend_le03": {r: dict(zip(("rho", "p"), [float(x) for x in st.spearmanr(lo.stage, lo[r])]))
                   for r in ("LI", "LI_norm", "MBON_hz_A")},
    "n_undefined_norm": int(sc.LI_norm.isna().sum()),
    "min_stage_undefined_norm": float(sc[sc.LI_norm.isna()].stage.min()),
    "by_stage": {str(s): {"LI_median": float(g.LI.median()), "MBON_median": float(g.MBON_hz_A.median()),
                          "LI_norm_median": float(g.LI_norm.median()) if g.LI_norm.notna().any() else None,
                          "LI_norm_n": int(g.LI_norm.notna().sum())}
                 for s, g in sc.groupby("stage")},
}

# ---------------------------------------------------------------- rule ablation
co = prep(pd.read_csv(BAT / "components_x4.csv"))
comp = {}
CR = ["brain_spikes_A", "LH_hz_A", "MBON_hz_A", "KC_hz_A", "LI", "LI_norm", "MN9_sugar_hz", "GF_loom_hz"]
for c, g in co.groupby("condition"):
    comp[c] = {r: {**(boot_ratio_of_medians(g[r], H[r]) or {}), "p_vs_healthy": mw(g[r], H[r])} for r in CR}
    comp[c]["dead_pct"] = round(100 * g.dead_frac.mean(), 2)
    comp[c]["n_kc_high"] = int((g.KC_hz_A > H.KC_hz_A.max()).sum())
    comp[c]["n_kc_above_1p5x_healthy_median"] = int((g.KC_hz_A > 1.5 * H.KC_hz_A.median()).sum())
res["components"] = comp
# full model against all-but-threshold, directly
for s in (0.5, 1.0):
    a = co[co.condition == f"all_s{s:g}"]
    b = co[co.condition == f"no_hyper_s{s:g}"]
    res["components"][f"all_vs_no_hyper_s{s:g}"] = {r: mw(a[r], b[r]) for r in CR}

# ---------------------------------------------------------------- vulnerability maps
vu = prep(pd.read_csv(BAT / "vulnerability_x4.csv"))
vul = {}
VR = ["brain_spikes_A", "LH_hz_A", "MBON_hz_A", "KC_frac_A", "LI", "LI_norm", "MN9_sugar_hz", "GF_loom_hz"]
for s, g in vu.groupby("stage"):
    m = {k: g[g.condition.str.startswith(k + "_")] for k in ("hub", "shuffled", "antihub")}
    out = {k: {r: boot_ratio_of_medians(v[r], H[r]) for r in VR} for k, v in m.items()}
    for k, v in m.items():
        out[k]["dead_pct"] = round(100 * v.dead_frac.mean(), 3)
        out[k]["kc_dead_pct"] = round(100 * v.kc_dead_frac.mean(), 2)
        out[k]["n_gf_fail"] = int((v.GF_loom_hz < 100).sum())
        out[k]["n_mn9_silent"] = int((v.MN9_sugar_hz == 0).sum())
    out["tests"] = {}
    for r in VR + ["dead_frac", "kc_dead_frac"]:
        gr = [v[r].dropna() for v in m.values()]
        out["tests"][r] = {"kruskal_p": float(st.kruskal(*gr).pvalue) if min(map(len, gr)) >= 3 else None,
                           "hub_vs_shuffled_p": mw(m["hub"][r], m["shuffled"][r]),
                           "hub_vs_antihub_p": mw(m["hub"][r], m["antihub"][r]),
                           "shuffled_vs_antihub_p": mw(m["shuffled"][r], m["antihub"][r])}
    vul[str(s)] = out
res["vulnerability"] = vul

# three independent groups of the default (hub, pan-neuronal) model at s = 0.5 and 1.0
rep = {}
for s in (0.5, 1.0):
    groups = {"series": sc[sc.stage == s], "ablation_all": co[co.condition == f"all_s{s:g}"],
              "maps_hub": vu[vu.condition == f"hub_s{s:g}"]}
    pooled = pd.concat(groups.values())
    e = {"n_pooled": int(len(pooled))}
    for r in ["brain_spikes_A", "LH_hz_A", "MBON_hz_A", "KC_hz_A", "LI", "MN9_sugar_hz", "GF_loom_hz"]:
        e[r] = {"medians": {k: float(v[r].median()) for k, v in groups.items()},
                "kruskal_p": float(st.kruskal(*[v[r] for v in groups.values()]).pvalue),
                "pooled_pct": boot_ratio_of_medians(pooled[r], H[r])}
    rep[str(s)] = e
res["replicates"] = rep

# ---------------------------------------------------------------- Kenyon-cell-only pathology
sp = prep(pd.read_csv(BAT / "scope_x4.csv"))
scope = {}
SR = ["brain_spikes_A", "LH_hz_A", "MBON_hz_A", "KC_frac_A", "LI", "LI_norm", "MN9_sugar_hz", "GF_loom_hz"]
for s, g in sp.groupby("stage"):
    pan = sc[sc.stage == s]
    scope[str(s)] = {"mb_only": {r: {**(boot_ratio_of_medians(g[r], H[r]) or {}), "p_vs_healthy": mw(g[r], H[r])}
                                 for r in SR},
                     "pan": {r: boot_ratio_of_medians(pan[r], H[r]) for r in SR},
                     "dead_pct_mb": round(100 * g.dead_frac.mean(), 3),
                     "dead_pct_pan": round(100 * pan.dead_frac.mean(), 3),
                     "kc_dead_pct_mb": round(100 * g.kc_dead_frac.mean(), 2)}
res["scope"] = scope

# ---------------------------------------------------------------- output neuron removal (from output_neuron_loss.py)
res["output_neuron_loss"] = json.loads((ROOT / "results/paper/output_neuron_loss.json").read_text())

# ---------------------------------------------------------------- joint cohort
jd = prep(pd.read_csv(ROOT / "results/joint/cohort_joint.csv"))
JH = jd[jd.stage == 0]
jst = sorted(s for s in jd.stage.unique() if s > 0)
JR = ["brain_spikes_A", "LH_hz_A", "KC_hz_A", "MBON_hz_A", "LI", "MN9_sugar_hz", "GF_loom_hz",
      "behaviour_feed_s", "behaviour_esc", "behaviour_near_A_s", "behaviour_PI"]
jrows = []
for r in JR:
    for s in jst:
        g = jd[jd.stage == s][r].dropna()
        a, lo_, hi_ = auc_boot(JH[r], g)
        jrows.append({"readout": r, "stage": s, "auc": round(a, 4), "lo": round(lo_, 4), "hi": round(hi_, 4),
                      "n_healthy": int(JH[r].notna().sum()), "n_group": int(len(g))})
pd.DataFrame(jrows).to_csv(OUT / "joint_auc.csv", index=False)
# whole-brain spikes (and LH rate) against the best arena readout, same flies
beh = ["behaviour_feed_s", "behaviour_esc", "behaviour_near_A_s", "behaviour_PI"]
jp = []
for c in ("brain_spikes_A", "LH_hz_A", "LI", "MN9_sugar_hz", "GF_loom_hz"):
    for s in jst:
        sub = jd[jd.stage == s]
        diffs = {b: np.empty(N_BOOT) for b in beh}
        for i in range(N_BOOT):
            hi = RNG.integers(0, len(JH), len(JH))
            si = RNG.integers(0, len(sub), len(sub))
            hh, ss = JH.iloc[hi], sub.iloc[si]
            ac = auc(hh[c].dropna(), ss[c].dropna())
            for b in beh:
                diffs[b][i] = ac - auc(hh[b].dropna(), ss[b].dropna())
        worst = min(beh, key=lambda b: np.percentile(diffs[b], 2.5))
        lo_, hi_ = np.percentile(diffs[worst], [2.5, 97.5])
        jp.append({"circuit": c, "stage": s, "closest_arena": worst, "mean_diff": round(float(diffs[worst].mean()), 4),
                   "lo": round(float(lo_), 4), "hi": round(float(hi_), 4), "beats_all": bool(lo_ > 0)})
pd.DataFrame(jp).to_csv(OUT / "joint_vs_arena.csv", index=False)
jtrend, jraw = {}, {}
for r in JR + ["KC_frac_A", "PN_hz_A", "V_pre_A", "GF_latency_ms", "LI_norm"]:
    v = jd[["stage", r]].dropna()
    rho, p = st.spearmanr(v.stage, v[r])
    jraw[r] = float(p)
    jtrend[r] = {"rho": round(float(rho), 3), "p": float(p), "n": int(len(v))}
for r, q in holm(jraw).items():
    jtrend[r]["p_holm"] = float(q)
pi_h = JH.behaviour_PI.dropna()
res["joint"] = {"n": int(len(jd)), "n_by_stage": {str(k): int(v) for k, v in jd.groupby("stage").size().items()},
                "trend": jtrend,
                "healthy_PI_mean": round(float(pi_h.mean()), 3), "healthy_PI_n": int(len(pi_h)),
                "healthy_PI_wilcoxon_p": float(st.wilcoxon(pi_h).pvalue),
                "healthy_PI_vs_LI_spearman": [float(x) for x in st.spearmanr(JH.dropna(subset=["behaviour_PI"]).behaviour_PI,
                                                                             JH.dropna(subset=["behaviour_PI"]).LI)],
                "n_no_threat_seen": int(jd.behaviour_esc.isna().sum()),
                "n_PI_undefined": int(jd.behaviour_PI.isna().sum())}

# ---------------------------------------------------------------- totals
res["totals"] = {"series": len(sc), "ablation": len(co), "maps": len(vu), "kc_only": len(sp), "joint": len(jd),
                 "all": len(sc) + len(co) + len(vu) + len(sp) + len(jd)}

(OUT / "battery_stats.json").write_text(json.dumps(res, indent=1, default=float))
print("wrote", OUT / "battery_stats.json")
