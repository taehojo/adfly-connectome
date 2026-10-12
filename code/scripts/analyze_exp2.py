"""Analysis of experiment 2 (maps at matched synaptic burden).

  python scripts/analyze_exp2.py

Reads results/exp2/battery.csv and matched.json, writes results/exp2/summary.json and
summary.csv. Readouts are medians as % of the healthy median with 95% bootstrap
intervals (4,000 resamples of flies within groups).

Decision rule, fixed before the run phase (primary readout: whole-brain spike count,
for each reference burden at which at least two maps are reachable):
  ratio = highest map median / lowest map median, with a bootstrap interval.
  * burden explains the map effect  if the ratio is at most 1.27, the largest spread
    between three independent groups of the same condition in the paper (S2 Fig);
  * placement matters beyond burden if the ratio exceeds 1.27 and its 95% interval
    lies entirely above 1.27, with the hub map lowest;
  * otherwise unresolved at this sample size.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
E2 = ROOT / "results/exp2"
READ = ["brain_spikes_A", "LH_hz_A", "KC_hz_A", "MBON_hz_A", "LI", "MN9_sugar_hz", "GF_loom_hz"]
REPLICATE_SPREAD = 1.27
RNG = np.random.default_rng(2)
NB = 4000


def boot_med(x):
    x = np.asarray(x, float)
    return np.median(RNG.choice(x, (NB, len(x))), axis=1)


def main():
    d = pd.read_csv(E2 / "battery.csv")
    m = json.loads((E2 / "matched.json").read_text())
    h = d[d["map"] == "healthy"]
    out = {"n_healthy": int(len(h)), "references": {}}
    rows = []
    for ref, info in m["maps"].items():
        g = d[d.reference.astype(str) == str(ref)]
        maps = [v for v in ("hub", "shuffled", "antihub") if (g["map"] == v).any()]
        o = {"target_syn_lost_frac": info["target_syn_lost_frac"], "maps": {}}
        for v in maps:
            x = g[g["map"] == v]
            o["maps"][v] = {"s": info[v]["s"], "n": int(len(x)),
                            "syn_lost_pct": round(100 * float(x.syn_lost_frac.median()), 2),
                            "dead_pct": round(100 * float(x.dead_frac.median()), 2),
                            "kc_dead_pct": round(100 * float(x.kc_dead_frac.median()), 2)}
            for r in READ:
                hb = boot_med(h[r]); xb = boot_med(x[r])
                hm = np.median(h[r])
                pct = 100 * np.median(x[r]) / hm
                lo, hi = np.percentile(100 * xb / hb, [2.5, 97.5])
                o["maps"][v][r] = {"pct": round(float(pct), 1), "lo": round(float(lo), 1), "hi": round(float(hi), 1)}
                rows.append({"reference": ref, "map": v, "readout": r, "pct": pct, "lo": lo, "hi": hi,
                             "s": info[v]["s"], "n": len(x)})
        o["unreachable"] = [v for v in ("hub", "shuffled", "antihub") if not info[v]["reachable"]]
        o["kruskal_p"] = {r: float(st.kruskal(*[g[g["map"] == v][r].dropna() for v in maps]).pvalue)
                          for r in READ} if len(maps) > 1 else {}
        if len(maps) > 1:
            meds = {v: np.median(g[g["map"] == v].brain_spikes_A) for v in maps}
            boots = {v: boot_med(g[g["map"] == v].brain_spikes_A) for v in maps}
            hi_v, lo_v = max(meds, key=meds.get), min(meds, key=meds.get)
            ratio = meds[hi_v] / meds[lo_v]
            rb = np.max(np.column_stack([boots[v] for v in maps]), 1) / np.min(np.column_stack([boots[v] for v in maps]), 1)
            ci = np.percentile(rb, [2.5, 97.5])
            if ratio <= REPLICATE_SPREAD:
                verdict = "burden explains the map effect"
            elif ci[0] > REPLICATE_SPREAD and lo_v == "hub":
                verdict = "placement matters beyond burden"
            else:
                verdict = "unresolved"
            o["primary"] = {"highest": hi_v, "lowest": lo_v, "ratio": round(float(ratio), 3),
                            "ratio_ci": [round(float(c), 3) for c in ci], "verdict": verdict}
        out["references"][ref] = o
    (E2 / "summary.json").write_text(json.dumps(out, indent=1))
    pd.DataFrame(rows).to_csv(E2 / "summary.csv", index=False)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
