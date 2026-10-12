"""Sweep spike-frequency adaptation (tau_a, da) on MaleCNS v1.0.

Criteria fixed before the sweep:
  C1 quiescence: active neurons in the last 200 ms of a 700 ms post-stimulus
     window should be ~0 for odor, sugar and looming stimuli.
  C2 validated responses kept: sugar GRNs -> MN9 and LPLC2 -> DNp01 (giant
     fiber) remain strongly driven.
  C3 KC sparseness: fraction of KCs firing during a single-odor stimulus in
     the experimentally reported range (roughly 5-20 %).
Adaptation applies to all non-sensory neurons (sensory neurons are driven by
the external Poisson input).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adfly.lif import PACK, Brains, Params, poisson_events  # noqa: E402

OUT = ROOT / "results/calibration"
neu = pd.read_parquet(PACK / "neurons.parquet")
neu["side"] = neu.somaSide.where(neu.somaSide.isin(["L", "R"]), neu.rootSide)
t = neu.type.fillna("")
sensory = neu.superclass.str.contains("sensory").to_numpy()
kc = np.flatnonzero(t.str.startswith("KC").to_numpy())


def idx(types, sides=("L", "R")):
    return np.flatnonzero((t.isin(types) & neu.side.isin(sides)).to_numpy())


STIM = {"odor": idx(["ORN_DM1", "ORN_DM4", "ORN_VA2"], ("L",)),
        "sugar": idx(["LB3c", "LB3b"], ("R",)),
        "looming": idx(["LPLC2"], ("R",))}
MN9, GF = idx(["MN9"]), idx(["DNp01"])


def run(stim, tau_a, da, B=4, seed=0):
    br = Brains(B, params=Params(tau_a=tau_a))
    br.da[:, ~sensory] = da
    br.set_groups(np.arange(br.N), br.N)
    br.no_refractory(stim)
    si = np.tile(stim.astype(np.int32), (B, 1))
    rng = np.random.default_rng(seed)
    on = br.run(3000, si, poisson_events(rng, np.full(si.shape, 150.0), 3000, 0.1), bin_steps=1000)
    off = br.run(7000, si, np.zeros((B, 7000, stim.size), bool), bin_steps=1000)
    return np.concatenate([on, off], 1).mean(0) * 10  # (10 bins of 100 ms, N) Hz


rows = []
for tau_a, da in [(1000.0, 0.0)] + list(itertools.product((300.0, 1000.0), (0.5, 1.0, 2.0, 4.0))):
    r = {k: run(s, tau_a, da) for k, s in STIM.items()}
    row = {"tau_a": tau_a, "da": da,
           "odor_active_on": int((r["odor"][:3] > 0).any(0).sum()),
           "odor_active_last200": int((r["odor"][-2:] > 0).any(0).sum()),
           "KC_frac_on": round(float((r["odor"][:3][:, kc] > 0).any(0).mean()), 3),
           "KC_mean_hz_on": round(float(r["odor"][:3][:, kc].mean()), 2),
           "sugar_MN9_hz": r["sugar"][:3][:, MN9].mean(0).round(1).tolist(),
           "sugar_active_last200": int((r["sugar"][-2:] > 0).any(0).sum()),
           "loom_GF_hz": r["looming"][:3][:, GF].mean(0).round(1).tolist(),
           "loom_active_last200": int((r["looming"][-2:] > 0).any(0).sum())}
    rows.append(row)
    print(row, flush=True)
pd.DataFrame(rows).to_csv(OUT / "adaptation_sweep.csv", index=False)
(OUT / "adaptation_sweep.json").write_text(json.dumps(rows, indent=1))
