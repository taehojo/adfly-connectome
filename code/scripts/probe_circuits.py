"""Probe sensorimotor circuits needed for the fly house.

1. persistence: does activity return to rest after a stimulus ends?
2. looming: LPLC2 / LC4 stimulation -> giant fiber (DNp01) and other DNs
3. olfaction: unilateral ORN stimulation -> steering DNs (DNa01/DNa02)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adfly.lif import PACK, Brains, poisson_events  # noqa: E402

OUT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir()) / "results/probe"
OUT.mkdir(parents=True, exist_ok=True)
neu = pd.read_parquet(PACK / "neurons.parquet")
neu["side"] = neu.somaSide.where(neu.somaSide.isin(["L", "R"]), neu.rootSide)
DT = 0.1


def idx(types, sides=("L", "R")):
    return np.flatnonzero((neu.type.isin(types) & neu.side.isin(sides)).to_numpy())


def run(stim, rate, t_on, t_off, n_trials=10, bin_ms=50.0, seed=0):
    br = Brains(n_trials)
    br.set_groups(np.arange(br.N), br.N)
    br.no_refractory(stim)
    rng = np.random.default_rng(seed)
    si = np.tile(stim.astype(np.int32), (n_trials, 1))
    out = []
    for dur, r in ((t_on, rate), (t_off, 0.0)):
        n = int(round(dur / DT))
        if n == 0:
            continue
        ev = poisson_events(rng, np.full(si.shape, r), n, DT)
        out.append(br.run(n, si, ev, bin_steps=int(bin_ms / DT)))
    return np.concatenate(out, axis=1) / (bin_ms / 1000.0)  # (B, bins, N) Hz


report = {}

# 1. persistence after sugar stimulation (LB3c+LB3b, right)
r = run(idx(["LB3c", "LB3b"], ("R",)), 150.0, 300.0, 700.0)
tot = r.mean(0)  # (bins, N)
n_active = (tot > 0).sum(1)
report["persistence_sugar"] = {"bin_ms": 50, "n_active_per_bin": n_active.tolist(),
                               "total_rate_per_bin": tot.sum(1).round(1).tolist()}
print("persistence (sugar 0-300ms on): active neurons per 50ms bin:", n_active.tolist())

# 2. looming
dn_types = ["DNp01", "DNp02", "DNp04", "DNp11", "DNa01", "DNa02", "DNp09", "MDN", "MN9"]
for name, stim in {"LPLC2_R": idx(["LPLC2"], ("R",)), "LPLC2_LR": idx(["LPLC2"]),
                   "LC4_R": idx(["LC4"], ("R",)), "LC4_LPLC2_R": idx(["LC4", "LPLC2"], ("R",))}.items():
    r = run(stim, 150.0, 300.0, 200.0)
    m = r[:, :6, :].mean((0, 1))  # during stimulus
    rows = []
    for t in dn_types:
        for s in ("L", "R"):
            ii = idx([t], (s,))
            if ii.size:
                rows.append((t, s, round(float(m[ii].mean()), 2)))
    report[f"looming_{name}"] = {"n_stim": int(stim.size), "dn_rates": rows,
                                 "n_active": int((m > 0).sum())}
    print(f"looming {name} (n={stim.size}):", rows)

# 3. olfaction, unilateral (ORN axons project bilaterally; use rootSide)
for orn in (["ORN_DM1", "ORN_DM4", "ORN_VA2"], ["ORN_DA2"], ["ORN_V"]):
    for s in ("L", "R"):
        stim = idx(orn, (s,))
        r = run(stim, 150.0, 300.0, 0.0)
        m = r.mean((0, 1))
        rows = []
        for t in ["DNa01", "DNa02", "DNp09", "MDN", "DNp01"]:
            for ss in ("L", "R"):
                ii = idx([t], (ss,))
                rows.append((t, ss, round(float(m[ii].mean()), 2)))
        key = f"odor_{'+'.join(orn)}_{s}"
        report[key] = {"n_stim": int(stim.size), "dn_rates": rows, "n_active": int((m > 0).sum())}
        print(key, f"(n={stim.size}, active={int((m > 0).sum())}):", rows)

(OUT / "probe_circuits.json").write_text(json.dumps(report, indent=1))
