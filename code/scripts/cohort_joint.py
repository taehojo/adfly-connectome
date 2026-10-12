"""One cohort, both readout classes: circuit assays and free behaviour in the same flies.

When the stage-curve battery and the arena are run on separate cohorts,
comparing a circuit readout against an arena readout is a between-cohort
comparison. This script scores both in the same simulated individuals.

Order (brains are naive until the training step):
  1. odour A, odour B                     naive circuit responses, MB valence
  2. sugar                                feeding drive at MN9
  3. looming                              escape rate and latency at the giant fibre
  4. naive-valence probe                  per-fly baseline the arena subtracts
  5. odour A + PAM, odour B               training, KC->MBON depression
  6. odour A, odour B                     memory test, learning index
  7. arena                                free behaviour

  python scripts/cohort_joint.py --stage-list "0:32,0.05:16,0.1:16,0.15:16,0.2:16,0.3:16,0.5:16" --seconds 90
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adfly.house import Arena, Body  # noqa: E402
from adfly.house_nb import HouseNB  # noqa: E402
from adfly.lif import poisson_events  # noqa: E402
from adfly.model import Connectome, Fly, valence  # noqa: E402

DT = 0.1
OUT = ROOT / "results/joint"
OUT.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--stage-list", default="0:32,0.05:16,0.1:16,0.15:16,0.2:16,0.3:16,0.5:16")
ap.add_argument("--seconds", type=float, default=90.0)
ap.add_argument("--seed", type=int, default=5150)
ap.add_argument("--out", default="cohort_joint.csv")
args = ap.parse_args()

t0 = time.time()
cx = Connectome()
stages = [float(a) for a, c in (x.split(":") for x in args.stage_list.split(",")) for _ in range(int(c))]
n = len(stages)
# interleave severities so that any position effect in the arena cannot align with group
order = np.random.default_rng(args.seed).permutation(n)
flies = [Fly(fly_id=i, stage=stages[order[i]], seed=args.seed + 13 * i) for i in range(n)]
house = HouseNB(cx, flies, Arena(), Body(), seed=args.seed)
br, B = house.br, len(flies)
print(f"[{time.time()-t0:.0f}s] brains built, B={B}", flush=True)

rng = np.random.default_rng(args.seed + 1)
allg = np.arange(cx.N)
none = np.zeros((B, 0), np.int32)
grn = cx.idx(["LB3c", "LB3b"])
loom = cx.idx(["LPLC2", "LC4"], ("L",))
# the arena already exempts its own inputs; sugar and looming sets are the same ones
br.no_refractory(np.concatenate([house.in_idx, house.pam]))


def seg(idx, rates, ms, grp=None, n_grp=None, bin_ms=None):
    nstep = int(round(ms / DT))
    br.set_groups(allg if grp is None else grp, cx.N if grp is None else n_grp)
    bs = None if bin_ms is None else int(round(bin_ms / DT))
    if idx is None or len(idx) == 0:
        return br.run(nstep, none, np.zeros((B, nstep, 0), bool), bin_steps=bs)
    si = np.tile(np.asarray(idx, np.int32), (B, 1))
    return br.run(nstep, si, poisson_events(rng, np.tile(rates, (B, 1)), nstep, DT), bin_steps=bs)


def gap(ms):
    return seg(None, None, ms, grp=np.full(cx.N, -1, np.int32), n_grp=1)


hz = lambda idx: np.full(np.asarray(idx).size, 150.0)  # noqa: E731

preA = seg(house.pnA, hz(house.pnA), 1000)[:, 0, :]; gap(1000)
preB = seg(house.pnB, hz(house.pnB), 1000)[:, 0, :]; gap(1000)
g_mn9 = np.full(cx.N, -1, np.int32); g_mn9[cx.idx(["MN9"])] = 0
sug = seg(grn, hz(grn), 500, grp=g_mn9, n_grp=1)[:, 0, 0] / 0.5 / 2.0; gap(1000)
g_gf = np.full(cx.N, -1, np.int32); g_gf[cx.idx(["DNp01"])] = 0
gfb = seg(loom, hz(loom), 300, grp=g_gf, n_grp=1, bin_ms=1.0)[:, :, 0]; gap(1000)
gf_rate = gfb.sum(1) / 0.3 / 2.0
gf_lat = np.array([float(np.flatnonzero(r)[0]) if r.any() else np.nan for r in gfb])
print(f"[{time.time()-t0:.0f}s] naive circuit assays done", flush=True)

br.reset_state()   # the circuit assays leave adaptation charged; probe from rest
v0 = house.probe_naive()
print(f"[{time.time()-t0:.0f}s] naive valence probed", flush=True)
house.train()
print(f"[{time.time()-t0:.0f}s] trained", flush=True)

postA = seg(house.pnA, hz(house.pnA), 1000)[:, 0, :]; gap(1000)
postB = seg(house.pnB, hz(house.pnB), 1000)[:, 0, :]
print(f"[{time.time()-t0:.0f}s] memory test done", flush=True)

threats, tt = [], 6.0
trng = np.random.default_rng(args.seed + 2)
while tt < args.seconds - 2:
    ang, rad = trng.random() * 2 * np.pi, 18.0 * np.sqrt(trng.random())
    threats.append((round(tt, 2), round(rad * np.cos(ang), 1), round(rad * np.sin(ang), 1)))
    tt += 10.0 + trng.random() * 4.0
_, st = house.run(args.seconds, threats=threats, frame_ms=200.0,
                  progress=lambda s: print(f"  arena {s:.0f}/{args.seconds:.0f}s ({time.time()-t0:.0f}s)", flush=True))
print(f"[{time.time()-t0:.0f}s] arena done", flush=True)

kc = np.flatnonzero(cx.is_kc)
pn = np.flatnonzero((cx.cls == "ALPN").to_numpy())
lh = np.flatnonzero(cx.t.str.startswith("LH").to_numpy())
vpa, vpb, vqa, vqb = (valence(cx, r) for r in (preA, preB, postA, postB))
feed, jumps = np.array(st["feed_s"]), np.array(st["jumps"])
nA, nB = np.array(st["near_A_s"]), np.array(st["near_B_s"])
seen = np.array(st["threats_seen"], float)

rows = []
for b, f in enumerate(flies):
    tot = nA[b] + nB[b]
    rows.append({
        "fly": f.fly_id, "stage": f.stage, "seed": f.seed,
        "dead_frac": float(f.dead.mean()), "kc_dead_frac": float(f.dead[kc].mean()),
        "mean_load": float(f.load.mean()),
        # circuit readouts
        "KC_frac_A": float((preA[b, kc] > 0).mean()), "KC_hz_A": float(preA[b, kc].mean()),
        "PN_hz_A": float(preA[b, pn].mean()), "LH_hz_A": float(preA[b, lh].mean()),
        "MBON_hz_A": float(preA[b, cx.mbon].mean()), "brain_spikes_A": float(preA[b].sum()),
        "V_pre_A": float(vpa[b]), "V_pre_B": float(vpb[b]),
        "V_post_A": float(vqa[b]), "V_post_B": float(vqb[b]),
        "LI": float((vqa[b] - vpa[b]) - (vqb[b] - vpb[b])),
        "MN9_sugar_hz": float(sug[b]), "GF_loom_hz": float(gf_rate[b]),
        "GF_latency_ms": float(gf_lat[b]),
        # arena readouts, same fly
        "behaviour_feed_s": float(feed[b]), "behaviour_jumps": float(jumps[b]),
        "behaviour_threats_seen": float(seen[b]),
        "behaviour_esc": float(jumps[b] / seen[b]) if seen[b] > 0 else np.nan,
        "behaviour_near_A_s": float(nA[b]), "behaviour_near_B_s": float(nB[b]),
        "behaviour_PI": float((nA[b] - nB[b]) / tot) if tot > 0 else np.nan,
    })
pd.DataFrame(rows).to_csv(OUT / args.out, index=False)
json.dump({"stage_list": args.stage_list, "seconds": args.seconds, "seed": args.seed,
           "n_flies": B, "threats": threats, "wall_s": round(time.time() - t0, 1)},
          open(OUT / (Path(args.out).stem + "_meta.json"), "w"), indent=1)
print(f"[{time.time()-t0:.0f}s] wrote {OUT/args.out}", flush=True)
