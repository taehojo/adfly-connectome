"""Experiment 2: vulnerability maps compared at matched synaptic burden.

The paper's map comparison held the number of removed neurons fixed, so the hub map,
which loads the neurons with the most synapses, also removed the most synapses. Here
each map is run at the severity at which it removes the same number of functional
synaptic contacts as the hub map at a reference severity. If the hub map still lowers
activity more than the other maps, where damage falls matters beyond how much
connectivity it removes.

Functional contacts lost in one fly = contacts removed by thinning, plus the surviving
contacts of every connection whose presynaptic or postsynaptic neuron was removed,
as a fraction of all contacts (antennal-lobe connections, absent from the model in
every fly, are left out).

Phases (outputs under results/exp2/):

  calibrate  burden against severity for each map (pathology draws only, no
             simulation), and the severity per map that matches the hub map at each
             reference severity; writes calibration.csv and matched.json
  run        the full assay battery (as in assay_battery.py) for healthy flies and for
             each map at its matched severity, conditions interleaved across batches;
             resumable, writes battery.csv with each fly's burden
  (analysis: scripts/analyze_exp2.py)

  python scripts/exp2_burden_matched.py calibrate
  python scripts/exp2_burden_matched.py run --n 32
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adfly import pathology  # noqa: E402
from adfly.model import W_SYN, Connectome  # noqa: E402

OUT = ROOT / "results/exp2"
MAPS = ("hub", "shuffled", "antihub")
SEED_BASE = 6000          # pathology seeds of the run phase (the paper used 1000-4999)
S_MAX = 1.0               # above 1 a neuron's load would exceed the calibrated rule range


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def burden(cx, vuln, s, seed):
    """Functional contacts lost, neurons removed and Kenyon cells removed for one fly."""
    rng = np.random.default_rng(10_000 + int(seed))
    st = pathology.Stage(s=float(s), vuln=vuln)
    w, _, dead, _ = pathology.apply(st, rng, cx.row_ptr, cx.dst, cx.signed, W_SYN, cx.vuln, cx.is_kc, ~cx.sensory)
    use = ~cx.al_edge                      # antennal-lobe synapses are absent from the model in every fly
    total = np.abs(cx.signed).astype(np.int64)[use]
    kept = np.rint(np.abs(w) / W_SYN).astype(np.int64)[use]
    gone_edge = (dead[cx.src] | dead[cx.dst])[use]
    lost = (total - kept).sum() + kept[gone_edge].sum()
    return {"syn_lost_frac": float(lost / total.sum()),
            "thin_frac": float((total - kept).sum() / total.sum()),
            "dead_frac": float(dead.mean()), "kc_dead_frac": float(dead[cx.is_kc].mean())}


def calibrate(args):
    cx = Connectome()
    OUT.mkdir(parents=True, exist_ok=True)
    grid = np.round(np.arange(0.05, S_MAX + 1e-9, 0.05), 3)
    rows = []
    if args.reuse and (OUT / "calibration.csv").exists():
        rows = pd.read_csv(OUT / "calibration.csv").to_dict("records")
        log(f"reusing {len(rows)} calibration draws")
    for v in ([] if rows else MAPS):
        for s in grid:
            for k in range(args.calib_seeds):
                rows.append({"map": v, "s": float(s), "seed": 90_000 + k, **burden(cx, v, s, 90_000 + k)})
            log(v, s, f"{np.mean([r['syn_lost_frac'] for r in rows[-args.calib_seeds:]]):.4f}")
        pd.DataFrame(rows).to_csv(OUT / "calibration.csv", index=False)
    cal = pd.DataFrame(rows).groupby(["map", "s"]).mean(numeric_only=True).reset_index()
    out = {"reference_stages": args.ref, "maps": {}}
    for r in args.ref:
        target = float(cal[(cal["map"] == "hub") & np.isclose(cal["s"], r)].syn_lost_frac.iloc[0])
        out["maps"][str(r)] = {"target_syn_lost_frac": target}
        for v in MAPS:
            c = cal[cal["map"] == v].sort_values("s")
            if target > c.syn_lost_frac.max():
                s_v, ok = None, False                  # not reachable within the calibrated load range
            else:
                s_v, ok = float(np.interp(target, c.syn_lost_frac, c.s)), True
            out["maps"][str(r)][v] = {"s": None if s_v is None else round(s_v, 4), "reachable": ok,
                                      "max_syn_lost_frac": float(c.syn_lost_frac.max())}
    (OUT / "matched.json").write_text(json.dumps(out, indent=1))
    log("matched:", json.dumps(out))


def run(args):
    import assay_battery as ab                    # same protocol and readouts as the paper's battery
    cx = Connectome()
    m = json.loads((OUT / "matched.json").read_text())
    specs, k = [], 0
    for _ in range(args.n):
        specs.append(("healthy", 0.0, "pan", {}, SEED_BASE + k)); k += 1
    for r, d in m["maps"].items():
        for v in MAPS:
            if not d[v]["reachable"]:
                log(f"skip {v} at reference {r}: burden {d['target_syn_lost_frac']:.4f} not reachable")
                continue
            for _ in range(args.n):
                specs.append((f"{v}_ref{r}", d[v]["s"], "pan", {"vuln": v}, SEED_BASE + k)); k += 1
    order = np.random.default_rng(args.seed).permutation(len(specs))
    specs = [specs[i] for i in order]
    path = OUT / "battery.csv"
    done = set(pd.read_csv(path).seed) if path.exists() else set()
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    todo = [sp for sp in specs if sp[4] not in done]
    log(f"{len(specs)} flies in the design, {len(todo)} to run")
    t0 = time.time()
    for bi in range(0, len(todo), args.batch):
        batch = todo[bi:bi + args.batch]
        res = ab.run_batch(cx, batch, seed=args.seed * 100 + bi)
        for row, (cond, s, sc, kw, fs) in zip(res, batch):
            v = kw.get("vuln", "hub")
            row.update({"map": v if s > 0 else "healthy", "reference": cond.split("_ref")[-1] if s > 0 else "",
                        **({"syn_lost_frac": 0.0, "thin_frac": 0.0} if s == 0 else
                           {k2: b for k2, b in burden(cx, v, s, fs).items() if k2 in ("syn_lost_frac", "thin_frac")})})
        rows += res
        pd.DataFrame(rows).to_csv(path, index=False)
        log(f"{len(rows)}/{len(specs)} flies done, {time.time() - t0:.0f} s")
    log("run done")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["calibrate", "run"])
    ap.add_argument("--ref", type=float, nargs="+", default=[0.1, 0.25, 0.5], help="hub severities to match")
    ap.add_argument("--reuse", action="store_true", help="calibrate: reuse calibration.csv instead of drawing again")
    ap.add_argument("--calib-seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=32, help="flies per condition")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    a.ref = [float(r) for r in a.ref]
    {"calibrate": calibrate, "run": run}[a.phase](a)


if __name__ == "__main__":
    main()
