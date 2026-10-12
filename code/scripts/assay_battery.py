"""Neural assay battery for the paper: one row of readouts per simulated fly.

  python scripts/assay_battery.py --design stage_curve
  python scripts/assay_battery.py --design components
  python scripts/assay_battery.py --design vulnerability
  python scripts/assay_battery.py --design scope

Protocol per fly (all flies of a batch run in parallel, 0.1 ms steps):
  odor A 1 s | gap 1 s | odor B 1 s | gap 1 s          naive odor responses
  sugar (LB3c+LB3b, both sides) 0.5 s | gap 1 s          MN9 feeding drive
  looming (LPLC2+LC4, left) 0.3 s | gap 1 s              giant-fiber rate and latency
  odor A + PAM 1 s | gap 2 s | odor B 1 s | gap 2 s      training, then plasticity
  odor A 1 s | gap 1 s | odor B 1 s                      memory test
Odors drive uniglomerular PNs at 150 Hz, sensory and PAM neurons are Poisson
driven (Shiu et al. convention). All flies share the stimulus RNG stream of
their batch; pathology draws are seeded per fly.
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
from adfly.lif import poisson_events  # noqa: E402
from adfly.model import ODORS, Connectome, Fly, build_brains, learn, valence  # noqa: E402

OUT = ROOT / "results/battery"
OUT.mkdir(parents=True, exist_ok=True)
DT = 0.1


def design(name):
    """List of (condition, stage, scope, kw, n)."""
    if name == "stage_curve":
        d = [("healthy", 0.0, "pan", {}, 16)]
        d += [(f"s{s:g}", s, "pan", {}, 8) for s in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0)]
        return d
    if name == "components":
        d = []
        for s in (0.5, 1.0):
            d += [(f"all_s{s:g}", s, "pan", {}, 8),
                  (f"hyper_only_s{s:g}", s, "pan", {"syn_loss_max": 0.0, "death_max": 0.0}, 8),
                  (f"synloss_only_s{s:g}", s, "pan", {"dth_max": 0.0, "death_max": 0.0}, 8),
                  (f"death_only_s{s:g}", s, "pan", {"dth_max": 0.0, "syn_loss_max": 0.0}, 8),
                  (f"no_hyper_s{s:g}", s, "pan", {"dth_max": 0.0}, 8)]
        return d
    if name == "vulnerability":
        d = []
        for s in (0.5, 1.0):
            d += [(f"{v}_s{s:g}", s, "pan", {"vuln": v}, 8) for v in ("hub", "shuffled", "antihub")]
        return d
    if name == "scope":
        return [(f"mb_s{s:g}", s, "mb", {}, 8) for s in (0.5, 1.0)]
    raise ValueError(name)


def run_batch(cx, specs, seed):
    flies = [Fly(fly_id=i, stage=s, scope=sc, seed=fs, kw=kw) for i, (_, s, sc, kw, fs) in enumerate(specs)]
    br = build_brains(cx, flies)
    B = br.B
    rng = np.random.default_rng(seed)
    t = cx.t
    pnA, pnB, pam = cx.upn(ODORS["A"]), cx.upn(ODORS["B"]), cx.pam
    grn = cx.idx(["LB3c", "LB3b"])
    loom = cx.idx(["LPLC2", "LC4"], ("L",))
    br.no_refractory(np.concatenate([pnA, pnB, pam, grn, loom]))
    allg = np.arange(cx.N)
    none = np.zeros((B, 0), np.int32)

    def seg(idx, rates, ms, grp=None, n_grp=None, bin_ms=None):
        n = int(round(ms / DT))
        if grp is None:
            br.set_groups(allg, cx.N)
        else:
            br.set_groups(grp, n_grp)
        bs = None if bin_ms is None else int(round(bin_ms / DT))
        if idx is None or len(idx) == 0:
            return br.run(n, none, np.zeros((B, n, 0), bool), bin_steps=bs)
        si = np.tile(np.asarray(idx, np.int32), (B, 1))
        return br.run(n, si, poisson_events(rng, np.tile(rates, (B, 1)), n, DT), bin_steps=bs)

    gap = lambda ms: seg(None, None, ms, grp=np.full(cx.N, -1, np.int32), n_grp=1)  # noqa: E731
    preA = seg(pnA, np.full(pnA.size, 150.0), 1000)[:, 0, :]; gap(1000)
    preB = seg(pnB, np.full(pnB.size, 150.0), 1000)[:, 0, :]; gap(1000)
    g_mn9 = np.full(cx.N, -1, np.int32); g_mn9[cx.idx(["MN9"])] = 0
    sug = seg(grn, np.full(grn.size, 150.0), 500, grp=g_mn9, n_grp=1)[:, 0, 0] / 0.5 / 2.0; gap(1000)
    g_gf = np.full(cx.N, -1, np.int32); g_gf[cx.idx(["DNp01"])] = 0
    gfb = seg(loom, np.full(loom.size, 150.0), 300, grp=g_gf, n_grp=1, bin_ms=1.0)[:, :, 0]; gap(1000)
    gf_rate = gfb.sum(1) / 0.3 / 2.0
    gf_lat = np.array([float(np.flatnonzero(r)[0]) if r.any() else np.nan for r in gfb])
    pair_idx = np.concatenate([pnA, pam])
    pair = seg(pair_idx, np.concatenate([np.full(pnA.size, 150.0), np.full(pam.size, 100.0)]), 1000)[:, 0, :]
    gap(2000); seg(pnB, np.full(pnB.size, 150.0), 1000); gap(2000)
    dan_types = sorted(set(t.to_numpy()[pam]))
    dan_rate = [{d: float(pair[b, (t == d).to_numpy()].mean()) for d in dan_types} for b in range(B)]
    learn(cx, br, pair, dan_rate)
    postA = seg(pnA, np.full(pnA.size, 150.0), 1000)[:, 0, :]; gap(1000)
    postB = seg(pnB, np.full(pnB.size, 150.0), 1000)[:, 0, :]

    kc = np.flatnonzero(cx.is_kc)
    pn = np.flatnonzero((cx.cls == "ALPN").to_numpy())
    lh = np.flatnonzero(t.str.startswith("LH").to_numpy())
    vpa, vpb, vqa, vqb = (valence(cx, r) for r in (preA, preB, postA, postB))
    rows = []
    for b, (cond, s, sc, kw, fs) in enumerate(specs):
        f = flies[b]
        rows.append({
            "condition": cond, "stage": s, "scope": sc, "kw": json.dumps(kw), "seed": fs,
            "dead_frac": float(f.dead.mean()), "kc_dead_frac": float(f.dead[kc].mean()),
            "mean_load": float(f.load.mean()),
            "KC_frac_A": float((preA[b, kc] > 0).mean()), "KC_frac_B": float((preB[b, kc] > 0).mean()),
            "KC_overlap_AB": float(((preA[b, kc] > 0) & (preB[b, kc] > 0)).sum() / max(((preA[b, kc] > 0) | (preB[b, kc] > 0)).sum(), 1)),
            "KC_hz_A": float(preA[b, kc].mean()), "PN_hz_A": float(preA[b, pn].mean()),
            "LH_hz_A": float(preA[b, lh].mean()), "MBON_hz_A": float(preA[b, cx.mbon].mean()),
            "brain_spikes_A": float(preA[b].sum()),
            "V_pre_A": float(vpa[b]), "V_pre_B": float(vpb[b]), "V_post_A": float(vqa[b]), "V_post_B": float(vqb[b]),
            "LI": float((vqa[b] - vpa[b]) - (vqb[b] - vpb[b])),
            "MN9_sugar_hz": float(sug[b]), "GF_loom_hz": float(gf_rate[b]), "GF_latency_ms": float(gf_lat[b]),
            "PAM_pair_hz": float(np.mean(list(dan_rate[b].values()))),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scale", type=int, default=1, help="multiply every group size by this")
    ap.add_argument("--out", default=None, help="output csv name (default: <design>.csv)")
    args = ap.parse_args()
    t0 = time.time()
    cx = Connectome()
    specs = []
    base = {"stage_curve": 1000, "components": 2000, "vulnerability": 3000, "scope": 4000}[args.design]
    k = 0
    for cond, s, sc, kw, n in design(args.design):
        for _ in range(n * args.scale):
            specs.append((cond, s, sc, kw, base + k))
            k += 1
    # interleave conditions across batches so batch effects cannot align with condition
    order = np.random.default_rng(args.seed).permutation(len(specs))
    specs = [specs[i] for i in order]
    name = args.out or f"{args.design}.csv"
    rows = []
    for bi in range(0, len(specs), args.batch):
        batch = specs[bi:bi + args.batch]
        rows += run_batch(cx, batch, seed=args.seed * 100 + bi)
        pd.DataFrame(rows).to_csv(OUT / name, index=False)
        print(f"{args.design}: {len(rows)}/{len(specs)} flies done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
