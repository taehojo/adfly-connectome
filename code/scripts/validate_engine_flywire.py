"""Engine check: reproduce the published Brian2 sugarR example on FlyWire v630.

Builds a CSR pack from the Shiu et al. connectivity table, runs the same
experiment (21 right sugar GRNs, 150 Hz Poisson, 30 x 1 s), and compares
per-neuron firing rates with the published Brian2 output
(results/example/sugarR.parquet in philshiu/Drosophila_brain_model).
Also tests whether odor-like broad input produces self-sustained activity.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adfly.lif import Brains, poisson_events  # noqa: E402

SRC = ROOT / "data/shiu2024"
PACK = ROOT / "data/pack/flywire_630"
OUT = ROOT / "results/validation"
SUGAR = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
         720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
         720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
         720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
         720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
         720575940611875570]
MN9 = 720575940660219265


def build():
    PACK.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(SRC / "2023_03_23_completeness_630_final.csv", index_col=0)
    con = pd.read_parquet(SRC / "2023_03_23_connectivity_630_final.parquet")
    n = len(comp)
    pre = con.Presynaptic_Index.to_numpy(np.int64)
    post = con.Postsynaptic_Index.to_numpy(np.int32)
    w = con["Excitatory x Connectivity"].to_numpy(np.int32)
    o = np.lexsort((post, pre))
    pre, post, w = pre[o], post[o], w[o]
    row_ptr = np.zeros(n + 1, np.int64)
    np.add.at(row_ptr, pre + 1, 1)
    np.save(PACK / "row_ptr.npy", np.cumsum(row_ptr))
    np.save(PACK / "dst.npy", post)
    np.save(PACK / "signed_w.npy", w)
    np.save(PACK / "neuron_ids.npy", comp.index.to_numpy(np.int64))
    return comp.index.to_numpy(np.int64), int(np.abs(w).sum())


ids, n_syn = build()
pos = {f: i for i, f in enumerate(ids)}
stim = np.array([pos[f] for f in SUGAR], np.int32)
B, T = 30, 10000
br = Brains(B, pack=PACK)
br.set_groups(np.arange(br.N), br.N)
br.no_refractory(stim)
rng = np.random.default_rng(1)
si = np.tile(stim, (B, 1))
c = br.run(T, si, poisson_events(rng, np.full(si.shape, 150.0), T, 0.1))[:, 0, :]
ours = c.mean(0)  # Hz (1 s trials)

ref = pd.read_parquet(SRC / "example/sugarR.parquet")
ref_rate = ref.groupby("flywire_id").size() / 30.0
ref_vec = np.zeros(len(ids))
for f, r in ref_rate.items():
    ref_vec[pos[f]] = r
act = (ours > 0) | (ref_vec > 0)
pr = stats.pearsonr(ours[act], ref_vec[act])
sr = stats.spearmanr(ours[act], ref_vec[act])
res = {
    "neurons": int(len(ids)), "synapses": n_syn,
    "n_active_ours": int((ours > 0).sum()), "n_active_brian2": int((ref_vec > 0).sum()),
    "pearson_r": float(pr[0]), "spearman_r": float(sr[0]),
    "mn9_ours_hz": float(ours[pos[MN9]]), "mn9_brian2_hz": float(ref_vec[pos[MN9]]),
}

# persistence on FlyWire: sugar on 300 ms then 700 ms off
br2 = Brains(8, pack=PACK)
br2.set_groups(np.arange(br2.N), br2.N)
br2.no_refractory(stim)
si2 = np.tile(stim, (8, 1))
on = br2.run(3000, si2, poisson_events(rng, np.full(si2.shape, 150.0), 3000, 0.1), bin_steps=1000)
off = br2.run(7000, si2, np.zeros((8, 7000, stim.size), bool), bin_steps=1000)
res["flywire_sugar_active_per_100ms"] = [int(x) for x in
                                         ((np.concatenate([on, off], 1).mean(0)) > 0).sum(1)]
print(json.dumps(res, indent=1))
(OUT / "engine_vs_brian2_flywire630.json").write_text(json.dumps(res, indent=1))
