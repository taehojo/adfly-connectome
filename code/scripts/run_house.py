"""Run the fly house and export playback data for the web page.

  python scripts/run_house.py --healthy 8 --ad 8 --stage 0.75 --seconds 60

Writes web/adfly/data/house.json (frames, per-fly stats, fly metadata) and
web/adfly/data/brain.json (soma sample with per-fly pathology for display).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adfly.house import Arena, Body  # noqa: E402
from adfly.house_nb import HouseNB  # noqa: E402
from adfly.model import Connectome, Fly  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--healthy", type=int, default=8)
ap.add_argument("--ad", type=int, default=8)
ap.add_argument("--stage", type=float, default=0.75)
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--out", default=str(ROOT / "web/adfly/data"))
ap.add_argument("--n-soma", type=int, default=7000)
ap.add_argument("--stage-list", default=None,
                help='e.g. "0:9,0.1:9,0.2:9" (stage:count); overrides --healthy/--ad/--stage')
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

t0 = time.time()
cx = Connectome()
if args.stage_list:
    stages = [float(a) for a, c in (x.split(":") for x in args.stage_list.split(",")) for _ in range(int(c))]
else:
    stages = [0.0] * args.healthy + [args.stage] * args.ad
n = len(stages)
order = np.random.default_rng(args.seed).permutation(n)       # interleave groups in the arena
flies = [Fly(fly_id=i, stage=stages[order[i]], seed=args.seed + i) for i in range(n)]
house = HouseNB(cx, flies, Arena(), Body(), seed=args.seed)
print(f"brains built: {time.time() - t0:.0f}s", flush=True)
v0 = house.probe_naive()
print(f"naive valence probed: {time.time() - t0:.0f}s", flush=True)
dan = house.train()
print(f"trained: {time.time() - t0:.0f}s", flush=True)

rng = np.random.default_rng(args.seed + 1)
threats = []
tt = 6.0
while tt < args.seconds - 2:
    ang, rad = rng.random() * 2 * np.pi, 18.0 * np.sqrt(rng.random())
    threats.append((round(tt, 2), round(rad * np.cos(ang), 1), round(rad * np.sin(ang), 1)))
    tt += 10.0 + rng.random() * 4.0
frames, stats = house.run(args.seconds, threats=threats,
                          progress=lambda s: print(f"  simulated {s:.0f}s ({time.time() - t0:.0f}s wall)", flush=True))
print(f"house run: {time.time() - t0:.0f}s", flush=True)

kc = np.flatnonzero(cx.is_kc)
meta = [{"id": f.fly_id, "group": "AD" if f.stage > 0 else "healthy", "stage": f.stage,
         "dead_frac": round(float(f.dead.mean()), 4),
         "kc_dead_frac": round(float(f.dead[kc].mean()), 4),
         "mean_load": round(float(f.load.mean()), 4)} for f in flies]
arena = Arena()
house_json = {
    "version": 1,
    "dt_frame": 0.05,
    "arena": {"R": arena.R, "food": arena.food, "odor": arena.odor_src, "sigma": arena.sigma,
              "shadow_dur": arena.shadow_dur, "shadow_rmax": arena.shadow_rmax},
    "threats": threats,
    "flies": meta,
    "naive_valence": {"levels": house.v0_lv.tolist(), "A": np.round(v0[:, 0], 1).tolist(), "B": np.round(v0[:, 1], 1).tolist()},
    "disp_groups": house.disp_names,
    "frames": frames,
    "stats": stats,
    "config": vars(args),
}
(out / "house.json").write_text(json.dumps(house_json, separators=(",", ":")))

# soma sample for the brain panel (deterministic: every k-th neuron with a soma)
loc = cx.neu.somaLocation
has = np.flatnonzero(loc.notna().to_numpy())
pick = has[np.linspace(0, has.size - 1, min(args.n_soma, has.size)).astype(int)]
xyz = np.array([loc.iloc[i] for i in pick], float)
brain_json = {
    "xyz": np.round(xyz / 1000.0, 1).tolist(),             # 8 nm voxels / 1000
    "disp": house.disp_id[pick].tolist(),
    "load": [np.round(f.load[pick], 2).tolist() for f in flies],
    "dead": [np.flatnonzero(f.dead[pick]).tolist() for f in flies],
}
(out / "brain.json").write_text(json.dumps(brain_json, separators=(",", ":")))
print("stats:", json.dumps(stats))
print(f"done {time.time() - t0:.0f}s; frames {len(frames)}")
