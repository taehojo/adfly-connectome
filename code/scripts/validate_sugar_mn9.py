"""Replicate the Shiu et al. (2024) feeding result on MaleCNS v1.0.

Poisson stimulation (150 Hz, 1 s, 30 trials) of right-side sugar GRNs should
activate the proboscis motor neuron MN9. Writes per-neuron mean rates.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adfly.lif import PACK, Brains, poisson_events  # noqa: E402

OUT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir()) / "results/validation"
OUT.mkdir(parents=True, exist_ok=True)

neu = pd.read_parquet(PACK / "neurons.parquet")
side = neu.somaSide.where(neu.somaSide.isin(["L", "R"]), neu.rootSide)
neu["side"] = side


def idx(types, sides=("L", "R")):
    m = neu.type.isin(types) & neu.side.isin(sides)
    return np.flatnonzero(m.to_numpy())


def experiment(name, stim, rate=150.0, n_trials=30, t_ms=1000.0, seed=0):
    br = Brains(n_trials)
    br.set_groups(np.arange(br.N), br.N)
    br.no_refractory(stim)
    n_steps = int(round(t_ms / br.p.dt))
    rng = np.random.default_rng(seed)
    stim_idx = np.tile(stim.astype(np.int32), (n_trials, 1))
    ev = poisson_events(rng, np.full(stim_idx.shape, rate), n_steps, br.p.dt)
    t0 = time.time()
    c = br.run(n_steps, stim_idx, ev)[:, 0, :]
    dt = time.time() - t0
    rates = c / (t_ms / 1000.0)
    res = pd.DataFrame({"bodyId": neu.bodyId, "type": neu.type, "side": neu.side,
                        "superclass": neu.superclass,
                        "rate_mean": rates.mean(0), "rate_sd": rates.std(0)})
    res.to_parquet(OUT / f"{name}.parquet")
    active = res[res.rate_mean > 0].sort_values("rate_mean", ascending=False)
    mn9 = res[res.type == "MN9"]
    print(f"== {name}: stim n={stim.size}, wall {dt:.1f}s for {n_trials}x{t_ms:.0f}ms")
    print(f"   active neurons (mean rate > 0): {len(active)}")
    print("   MN9:", mn9[["bodyId", "side", "rate_mean", "rate_sd"]].to_string(index=False))
    print(active.head(25)[["type", "side", "superclass", "rate_mean"]].to_string(index=False))
    return {"name": name, "n_stim": int(stim.size), "wall_s": dt,
            "n_active": int(len(active)),
            "mn9": mn9[["bodyId", "side", "rate_mean", "rate_sd"]].to_dict("records")}


summary = [
    experiment("baseline_nostim", np.array([], np.int64)),
    experiment("sugar_LB3c_R", idx(["LB3c"], ("R",))),
    experiment("sugar_LB3c_LB3b_R", idx(["LB3c", "LB3b"], ("R",))),
    experiment("bitter_LB1abcd_R", idx(["LB1a", "LB1b", "LB1c", "LB1d"], ("R",))),
]
(OUT / "summary_sugar_mn9.json").write_text(json.dumps(summary, indent=2, default=float))
