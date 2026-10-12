"""Appetitive olfactory conditioning in healthy and AD-stage in-silico flies.

Protocol (all flies simulated in one batch, each with its own pathology draw):
  pre-test   odor A 1 s, 2 s gap, odor B 1 s, 2 s gap
  training   odor A + PAM dopamine neurons 1 s (paired), 2 s gap, odor B 1 s, 2 s gap
  post-test  same as pre-test
Odors drive uniglomerular PNs at 150 Hz; reward is direct PAM drive (100 Hz),
as in optogenetic reward substitution (Liu et al. 2012). Learning index
LI = [V_post(A) - V_pre(A)] - [V_post(B) - V_pre(B)], V = MBON ensemble valence.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adfly.lif import poisson_events  # noqa: E402
from adfly.model import ODORS, Connectome, Fly, build_brains, learn, valence  # noqa: E402

STAGES = [0.0, 0.25, 0.5, 0.75, 1.0]
N_PER = 6
DT = 0.1
OUT = ROOT / "results/learning"
OUT.mkdir(parents=True, exist_ok=True)

cx = Connectome()
flies = [Fly(fly_id=i, stage=s, seed=i) for i, s in enumerate(np.repeat(STAGES, N_PER))]
t0 = time.time()
br = build_brains(cx, flies)
print(f"built {len(flies)} brains in {time.time() - t0:.0f}s", flush=True)
br.set_groups(np.arange(cx.N), cx.N)
pnA, pnB, pam = cx.upn(ODORS["A"]), cx.upn(ODORS["B"]), cx.pam
br.no_refractory(np.concatenate([pnA, pnB, pam]))
rng = np.random.default_rng(1)
B = br.B


def seg(stim, rate, ms):
    n = int(round(ms / DT))
    si = np.tile(stim.astype(np.int32), (B, 1))
    rates = np.zeros(si.shape) if stim.size == 0 else np.broadcast_to(rate, si.shape)
    ev = poisson_events(rng, rates, n, DT) if stim.size else np.zeros((B, n, 0), bool)
    if stim.size == 0:
        si = np.zeros((B, 0), np.int32)
    return br.run(n, si, ev)[:, 0, :] / (ms / 1000.0)  # Hz


none = np.array([], np.int64)


def test():
    rA = seg(pnA, 150.0, 1000); seg(none, 0, 2000)
    rB = seg(pnB, 150.0, 1000); seg(none, 0, 2000)
    return rA, rB


preA, preB = test()
pair_stim = np.concatenate([pnA, pam])
pair_rate = np.concatenate([np.full(pnA.size, 150.0), np.full(pam.size, 100.0)])
rPair = seg(pair_stim, pair_rate, 1000); seg(none, 0, 2000)
seg(pnB, 150.0, 1000); seg(none, 0, 2000)
dan_types = sorted(set(cx.t.to_numpy()[cx.pam]))
dan_rate = [{d: float(rPair[b, (cx.t == d).to_numpy()].mean()) for d in dan_types} for b in range(B)]
kc_sp = rPair  # Hz over 1 s == spike counts
learn(cx, br, kc_sp, dan_rate)
postA, postB = test()

vpa, vpb, vqa, vqb = (valence(cx, r) for r in (preA, preB, postA, postB))
kc = np.flatnonzero(cx.is_kc)
df = pd.DataFrame({
    "fly": [f.fly_id for f in flies], "stage": [f.stage for f in flies],
    "dead_frac": [f.dead.mean() for f in flies],
    "kc_dead_frac": [f.dead[kc].mean() for f in flies],
    "V_pre_A": vpa, "V_post_A": vqa, "V_pre_B": vpb, "V_post_B": vqb,
    "KC_frac_A_pre": (preA[:, kc] > 0).mean(1), "KC_frac_B_pre": (preB[:, kc] > 0).mean(1),
    "PAM_rate_pair": [np.mean(list(d.values())) for d in dan_rate],
})
df["LI"] = (df.V_post_A - df.V_pre_A) - (df.V_post_B - df.V_pre_B)
df.to_csv(OUT / "learning_by_stage.csv", index=False)
print(df.round(3).to_string(index=False))
print(df.groupby("stage")[["LI", "V_pre_A", "V_post_A", "KC_frac_A_pre", "dead_frac"]].agg(["mean", "std"]).round(3).to_string())
print(f"total wall {time.time() - t0:.0f}s")
