"""Paper statistics for the joint cohort (circuit and arena readouts in the same flies).

  python scripts/analyze_joint.py --csv results/joint/cohort_joint.csv

Writes results/joint/stats_joint.json and three csv tables.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
CIRCUIT = ["KC_frac_A", "KC_hz_A", "PN_hz_A", "LH_hz_A", "MBON_hz_A", "brain_spikes_A",
           "V_pre_A", "MN9_sugar_hz", "GF_loom_hz", "GF_latency_ms", "LI", "LI_norm"]
BEHAV = ["behaviour_feed_s", "behaviour_esc", "behaviour_near_A_s", "behaviour_PI"]
N_BOOT = 4000
MBON_FLOOR = 0.5   # Hz; below this the normalised learning index is not defined


def add_normalised_li(d):
    """Learning index scaled by the fly's own MBON drive.

    The raw index is a difference of signed MBON rate sums, so a uniform change
    in excitability rescales it even when learning is unchanged. Dividing by the
    fly's naive MBON rate removes that gain term. The ratio is unstable as the
    denominator approaches zero, so it is left undefined below MBON_FLOOR.
    """
    den = d["MBON_hz_A"].where(d["MBON_hz_A"] >= MBON_FLOOR)
    d["LI_norm"] = d["LI"] / den
    return d


def auc(a, b):
    """Direction-free AUC of healthy a against group b."""
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    u = (gt + 0.5 * eq) / (len(a) * len(b))
    return max(u, 1.0 - u)


def holm(d):
    order = sorted(d, key=lambda k: d[k])
    m, prev, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        prev = min(1.0, max(prev, (m - i) * d[k]))
        out[k] = prev
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/joint/cohort_joint.csv")
    ap.add_argument("--out", default="results/joint")
    a = ap.parse_args()
    d = add_normalised_li(pd.read_csv(ROOT / a.csv))
    out = Path(ROOT / a.out)
    reads = [r for r in CIRCUIT + BEHAV if r in d.columns]
    stages = sorted(s for s in d.stage.unique() if s > 0)
    rng = np.random.default_rng(0)
    res = {"n_flies": int(len(d)), "n_by_stage": {str(k): int(v) for k, v in d.groupby("stage").size().items()},
           "stages": stages, "n_boot": N_BOOT}

    # 1. monotone dose-response across every fly
    raw, trend = {}, {}
    for r in reads:
        v = d[["stage", r]].dropna()
        rho, p = st.spearmanr(v.stage, v[r])
        raw[r] = float(p)
        trend[r] = {"spearman_rho": round(float(rho), 3), "p": float(p), "n": int(len(v))}
    for r, q in holm(raw).items():
        trend[r]["p_holm"] = float(q)
    res["trend"] = trend
    pd.DataFrame(trend).T.to_csv(out / "trend.csv")

    # 2. AUC against healthy at each severity, with permutation p
    H = {r: d[(d.stage == 0)][r].dropna().values for r in reads}
    rows = []
    for s in stages:
        for r in reads:
            g = d[(d.stage == s)][r].dropna().values
            if len(g) == 0 or len(H[r]) == 0:
                continue
            obs = auc(H[r], g)
            pool = np.concatenate([H[r], g])
            null = np.empty(2000)
            for i in range(2000):
                p = rng.permutation(pool)
                null[i] = auc(p[:len(H[r])], p[len(H[r]):])
            rows.append({"stage": s, "readout": r, "n_healthy": len(H[r]), "n_group": len(g),
                         "auc": round(float(obs), 4),
                         "mw_p": float(st.mannwhitneyu(H[r], g, alternative="two-sided")[1]),
                         "perm_p": float(((null >= obs).sum() + 1) / (len(null) + 1))})
    auc_tab = pd.DataFrame(rows)
    auc_tab.to_csv(out / "auc_by_stage.csv", index=False)

    # 3. first severity reaching AUC >= 0.80, with a bootstrap interval over flies
    casc = {}
    for r in reads:
        per = {s: d[(d.stage == s)][r].dropna().values for s in stages}
        obs = next((s for s in stages if len(per[s]) and auc(H[r], per[s]) >= 0.80), None)
        boot = np.full(N_BOOT, np.nan)
        for i in range(N_BOOT):
            hb = rng.choice(H[r], len(H[r]), replace=True)
            for s in stages:
                if len(per[s]) == 0:
                    continue
                if auc(hb, rng.choice(per[s], len(per[s]), replace=True)) >= 0.80:
                    boot[i] = s
                    break
        ok = ~np.isnan(boot)
        casc[r] = {"observed": obs, "frac_never": float(1 - ok.mean()),
                   "boot_median": float(np.median(boot[ok])) if ok.any() else None,
                   "boot_ci95": [float(np.percentile(boot[ok], 2.5)),
                                 float(np.percentile(boot[ok], 97.5))] if ok.any() else None}
    res["cascade"] = casc

    # 4. paired AUC difference between readout classes, same flies, bootstrapped over flies
    pairs = []
    cl = [r for r in CIRCUIT if r in d.columns]
    bh = [r for r in BEHAV if r in d.columns]
    for s in stages:
        sub = d[d.stage == s]
        hs = d[d.stage == 0]
        for c in cl:
            for b in bh:
                m = sub[[c, b]].dropna()
                mh = hs[[c, b]].dropna()
                if len(m) < 4 or len(mh) < 4:
                    continue
                diffs = np.empty(N_BOOT)
                for i in range(N_BOOT):
                    hi = rng.integers(0, len(mh), len(mh))
                    si = rng.integers(0, len(m), len(m))
                    diffs[i] = (auc(mh[c].values[hi], m[c].values[si])
                                - auc(mh[b].values[hi], m[b].values[si]))
                lo, hi_ = np.percentile(diffs, [2.5, 97.5])
                pairs.append({"stage": s, "circuit": c, "behaviour": b,
                              "auc_circuit": round(float(auc(mh[c].values, m[c].values)), 4),
                              "auc_behaviour": round(float(auc(mh[b].values, m[b].values)), 4),
                              "mean_diff": round(float(diffs.mean()), 4),
                              "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi_), 4),
                              "n_pairs": int(len(m)),
                              "circuit_better": bool(lo > 0)})
    pd.DataFrame(pairs).to_csv(out / "paired_auc.csv", index=False)
    res["n_pairs_circuit_better"] = int(sum(p["circuit_better"] for p in pairs))
    res["n_pairs_tested"] = len(pairs)

    (out / "stats_joint.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "trend"}, indent=1))
    print("\ntrend:")
    for r, v in res["trend"].items():
        print(f"  {r:<22} rho={v['spearman_rho']:+.3f} p_holm={v['p_holm']:.2e} n={v['n']}")
    print("\ncascade (first severity with AUC>=0.80):")
    for r, v in res["cascade"].items():
        print(f"  {r:<22} obs={str(v['observed']):<6} median={v['boot_median']} CI={v['boot_ci95']} never={v['frac_never']:.2f}")


if __name__ == "__main__":
    main()
