"""Engine validation for the paper: our numba LIF against Brian2 on the same machine.

Two reference outputs, two engine settings, all four comparisons reported.

references
  local_brian2_2.10.1      Brian2 2.10.1 run here with the unmodified published code.
  released_example_output  the output shipped with the published code, produced
                           under an older Brian2.

engine settings
  gate=True   synaptic arrivals during the refractory period are discarded, which
              is what `dg/dt` under `(unless refractory)` does in Brian2 2.10.1.
  gate=False  arrivals accumulate during the refractory period.

Writes results/validation/engine_validation.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adfly.lif import Brains, poisson_events  # noqa: E402

PACK = ROOT / "data/pack/flywire_630"
OUT = ROOT / "results/validation"
OUT.mkdir(parents=True, exist_ok=True)
SUGAR = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
         720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
         720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
         720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
         720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
         720575940611875570]
MN9 = 720575940660219265
N_TRIALS, T_STEPS = 30, 10000

REFS = [("local_brian2_2.10.1", OUT / "brian2_rerun/sugarR.parquet"),
        ("released_example_output", ROOT / "data/shiu2024/example/sugarR.parquet")]


def ref_rates(path, pos, n):
    df = pd.read_parquet(path)
    v = np.zeros(n)
    for f, c in (df.groupby("flywire_id").size() / N_TRIALS).items():
        if f in pos:
            v[pos[f]] = c
    return v, len(df) / N_TRIALS


ids = np.load(PACK / "neuron_ids.npy")
pos = {int(f): i for i, f in enumerate(ids)}
stim = np.array([pos[f] for f in SUGAR], np.int32)

res = {"model": "Shiu et al. 2024 leaky integrate-and-fire on FlyWire v630",
       "neurons": int(len(ids)),
       "stimulus": "21 right sugar GRNs, 150 Hz Poisson, 30 trials of 1 s",
       "engine": "numba, exact exponential integration at dt = 0.1 ms, Brian2 update order",
       "references": {}, "engine_runs": {}, "comparisons": {}}
rate_vectors = {}

refs = {}
for name, path in REFS:
    if not Path(path).exists():
        continue
    r, sp = ref_rates(path, pos, len(ids))
    refs[name] = r
    res["references"][name] = {"spikes_per_trial": round(sp, 1),
                               "active_neurons": int((r > 0).sum()),
                               "mn9_hz": round(float(r[pos[MN9]]), 2)}

for gate in (True, False):
    key = f"gate={gate}"
    per_seed = []
    for seed in (7, 11, 12):
        br = Brains(N_TRIALS, pack=PACK, gate_refractory_input=gate)
        br.set_groups(np.arange(br.N), br.N)
        br.no_refractory(stim)
        ev = poisson_events(np.random.default_rng(seed),
                            np.full((N_TRIALS, stim.size), 150.0), T_STEPS, 0.1)
        t0 = time.time()
        counts = br.run(T_STEPS, np.tile(stim, (N_TRIALS, 1)), ev)[:, 0, :]
        wall = time.time() - t0
        rate = counts.mean(0)
        per_seed.append(rate)
        res["engine_runs"].setdefault(key, []).append(
            {"seed": seed, "spikes_per_trial": round(float(counts.sum(1).mean()), 1),
             "active_neurons": int((rate > 0).sum()),
             "mn9_hz": round(float(rate[pos[MN9]]), 2), "wall_s": round(wall, 1)})
    ours = per_seed[0]
    rate_vectors[f"ours_gate{int(gate)}"] = ours
    for name, ref in refs.items():
        a = (ours > 0) | (ref > 0)
        res["comparisons"][f"{key} vs {name}"] = {
            "pearson_r": round(float(stats.pearsonr(ours[a], ref[a])[0]), 4),
            "spearman_r": round(float(stats.spearmanr(ours[a], ref[a])[0]), 4),
            "slope": round(float(np.polyfit(ref[a], ours[a], 1)[0]), 3),
            "n_compared": int(a.sum())}

(OUT / "engine_validation.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(OUT / "engine_validation_rates.npz", **rate_vectors,
                    **{f"ref_{k}": v for k, v in refs.items()})
print(json.dumps(res, indent=1))
