"""Experiment 1: counterfactual rescue of single simulated flies.

Question: when flies of the same severity differ in how impaired they are, is the
difference carried by the loss of a few particular neurons? Each fly of the 384-fly
severity series is rebuilt from its seed and simulated again with part of its own damage
undone. All variants of a fly receive identical input spikes, so every difference between
them is caused by the counterfactual (see adfly/counterfactual.py).

Phases, in order (outputs under results/exp1/):

  bench    checks that each fly's removed set is rebuilt exactly (against dead_frac in
           the series table), that a variant gives identical readouts wherever it sits in a
           batch, the speed of the dense and sparse kernels and how far they agree, and
           projects the run time of the screen.
  screen   per fly: five baselines (damaged; all removed neurons restored; all synapses
           restored; threshold shift removed; healthy), single rescue of every damaged
           class, random half rescues of classes (for Banzhaf values), single rescue of
           the cell types inside the most important classes plus random control types,
           single rescue of the neurons inside the most important types, and cumulative
           rescue of the top 1, 3, 10 and 30 types.
  robust   single rescue of the units listed in top_units.csv (written by
           analyze_exp1.py) under model variants: all weights x0.8 and x1.2, glutamatergic
           inhibition x0.5, adaptation x0.5, and two further input-noise realisations.
  confirm  the full protocol, which adds the learning index: rescue of each listed unit
           alone (necessity), and loss of that unit alone in an otherwise healthy fly
           (sufficiency).

  python scripts/exp1_counterfactual.py bench  --stage 0.5
  python scripts/exp1_counterfactual.py screen --stage 0.5 --n-flies 24 --batch 64
  python scripts/analyze_exp1.py
  python scripts/exp1_counterfactual.py robust  --stage 0.5 --n-flies 12 --batch 64
  python scripts/exp1_counterfactual.py confirm --stage 0.5 --n-flies 12 --batch 16

Every phase skips flies whose output is already complete, so an interrupted run can be
restarted with the same command. --pilot shrinks the screen for a first timing run.
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
from adfly import counterfactual as cf  # noqa: E402
from adfly.lif import PACK  # noqa: E402
from adfly.model import Connectome  # noqa: E402

PROTOCOL_S = {"short": 3.8, "full": 15.8}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- shared setup
def choose_flies(args, cx):
    d = pd.read_csv(args.flies_csv)
    d = d[(d.scope == "pan") & np.isclose(d.stage, args.stage)].sort_values("seed")
    if args.seeds:
        want = {int(s) for s in args.seeds.split(",")}
        d = d[d.seed.isin(want)]
    else:
        d = d.head(args.n_flies)
    if d.empty:
        sys.exit(f"no flies at severity {args.stage} in {args.flies_csv}")
    out = []
    for _, r in d.iterrows():
        dmg = cf.fly_damage(cx, int(r.seed), float(r.stage))
        if abs(dmg.dead.mean() - r.dead_frac) > 1e-9:
            sys.exit(f"seed {r.seed}: rebuilt removed fraction {dmg.dead.mean():.10f} differs from "
                     f"the series table {r.dead_frac:.10f}; the connectome pack or the code differs from "
                     f"the one that produced the series")
        out.append(dmg)
    log(f"{len(out)} flies at s = {args.stage}, removed sets rebuilt exactly")
    return out


def batched(runner, sets, variants, dmg, input_seed):
    res = []
    for i in range(0, len(variants), runner.B):
        res += runner.run(sets, variants[i:i + runner.B], dmg, input_seed)
    return res


def rows_of(variants, results, dmg, **extra):
    return [{"seed": dmg.seed, "stage": dmg.stage, "level": v.level, "unit": v.unit, "label": v.label,
             "n_restored": 0 if v.restore is None else int(len(v.restore)),
             "n_lesioned": 0 if v.lesion is None else int(len(v.lesion)), **extra, **r}
            for v, r in zip(variants, results)]


def fractions(df, y_dmg, y_h, readouts, min_deficit):
    """Fraction of the fly's own deficit restored, per readout (NaN where there is no deficit)."""
    out = pd.DataFrame(index=df.index)
    for r in readouts:
        d = y_h[r] - y_dmg[r]
        ok = np.isfinite(d) and abs(d) > min_deficit * max(abs(y_h[r]), 1e-9)
        out[r] = (df[r] - y_dmg[r]) / d if ok else np.nan
    return out


def top_by(fr, labels, k):
    """Union of the k units with the largest restored fraction for each readout."""
    chosen = []
    for r in fr.columns:
        s = fr[r].dropna()
        if s.empty:
            continue
        for i in s.sort_values(ascending=False).index[:k]:
            if labels[i] not in chosen:
                chosen.append(labels[i])
    return chosen


# ---------------------------------------------------------------- bench
def bench(args):
    cx = Connectome(Path(args.pack))
    flies = choose_flies(args, cx)[:2]
    w_h = cf.healthy_weights(cx)
    res = {"stage": args.stage, "batch": args.batch, "N": int(cx.N), "E": int(cx.signed.size)}
    dmg = flies[0]
    sets = [dmg.w, w_h]
    base = cf.baseline_variants(dmg)
    for kernel in (["dense", "sparse"] if args.compare_sparse else ["sparse" if args.sparse else "dense"]):
        runner = cf.Runner(cx, args.batch, "short", sparse=(kernel == "sparse"))
        runner.run(sets, base[:1], dmg, 1)                       # compile
        t0 = time.time()
        out = runner.run(sets, [base[0]] * args.batch, dmg, 1)
        wall = time.time() - t0
        res[f"{kernel}_wall_s_per_batch"] = round(wall, 2)
        res[f"{kernel}_wall_s_per_brain_second"] = round(wall / (args.batch * PROTOCOL_S["short"]), 4)
        a = runner.run(sets, base, dmg, 7)
        b = runner.run(sets, base[::-1], dmg, 7)[::-1]
        same = all(abs(x[k] - y[k]) < 1e-9 or (np.isnan(x[k]) and np.isnan(y[k]))
                   for x, y in zip(a, b) for k in runner.readouts)
        res[f"{kernel}_deterministic_across_batch_positions"] = bool(same)
        res[f"{kernel}_baselines"] = {v.label: r for v, r in zip(base, a)}
        log(kernel, f"{wall:.1f} s per batch of {args.batch}; deterministic: {same}")
    if args.compare_sparse:
        diff = {v.label: {k: (res["sparse_baselines"][v.label][k] - res["dense_baselines"][v.label][k])
                          / max(abs(res["dense_baselines"][v.label][k]), 1e-9)
                          for k in cf.READOUTS_SHORT} for v in base}
        res["sparse_minus_dense_relative"] = diff
    # projection of the screen
    labels = cf.unit_labels(cx)
    dead_idx = np.flatnonzero(dmg.dead)
    n_cls = len(cf.group(dead_idx, labels["class"]))
    runs = 5 + n_cls + args.n_subsets + args.max_types + args.random_types + \
        len(cf.SELECT) * args.top_types * args.max_neurons_per_type // 2 + len(cf.SELECT) * len(args.cumulative_k)
    key = "sparse" if (args.sparse and not args.compare_sparse) else "dense"
    if args.compare_sparse:
        key = "sparse" if res["sparse_wall_s_per_batch"] < res["dense_wall_s_per_batch"] else "dense"
    per_fly_h = runs / args.batch * res[f"{key}_wall_s_per_batch"] / 3600
    res["screen_projection"] = {"damaged_classes_in_first_fly": n_cls, "runs_per_fly_approx": int(runs),
                                "kernel": key, "hours_per_fly": round(per_fly_h, 2),
                                "hours_for_n_flies": round(per_fly_h * args.n_flies, 1)}
    log("projected screen:", res["screen_projection"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"bench_s{args.stage:g}.json").write_text(json.dumps(res, indent=1, default=float))


# ---------------------------------------------------------------- screen
def screen(args):
    cx = Connectome(Path(args.pack))
    flies = choose_flies(args, cx)
    labels = cf.unit_labels(cx)
    w_h = cf.healthy_weights(cx)
    runner = cf.Runner(cx, args.batch, "short", sparse=args.sparse)
    gp = cf.GraphPredictors(cx, runner)
    out = Path(args.out) / "screen"
    out.mkdir(parents=True, exist_ok=True)
    body = cx.neu.bodyId.to_numpy()
    for dmg in flies:
        tag = f"s{dmg.stage:g}_seed{dmg.seed}"
        if (out / f"{tag}.done.json").exists() and not args.redo:
            log(tag, "already done")
            continue
        t0 = time.time()
        sets, in_seed = [dmg.w, w_h], args.input_seed_base + dmg.seed
        rng = np.random.default_rng(args.subset_seed + dmg.seed)
        rows = []

        def go(vs, step):
            r = batched(runner, sets, vs, dmg, in_seed)
            rows.extend(rows_of(vs, r, dmg, step=step))
            return pd.DataFrame(r)

        # 1. baselines
        base = cf.baseline_variants(dmg)
        b = go(base, "baseline")
        y_dmg, y_h = b.iloc[0].to_dict(), b.iloc[4].to_dict()
        # 2. single rescue of each damaged class
        dead_idx = np.flatnonzero(dmg.dead)
        cls = cf.group(dead_idx, labels["class"])
        cnames = sorted(cls)
        vs = [cf.Variant(f"restore {c}", "class", c, restore=cls[c]) for c in cnames]
        fr_c = fractions(go(vs, "class_single"), y_dmg, y_h, cf.SELECT, args.min_deficit)
        # 3. random half rescues of classes (Banzhaf values)
        if args.n_subsets and len(cnames) > 1:
            X = rng.random((args.n_subsets, len(cnames))) < 0.5
            vs = [cf.Variant(f"subset {i}", "subset", f"subset{i}",
                             restore=np.concatenate([cls[c] for c, x in zip(cnames, X[i]) if x] or [np.zeros(0, np.int64)]))
                  for i in range(args.n_subsets)]
            go(vs, "class_subset")
            np.savez_compressed(out / f"{tag}_subsets.npz", X=X, classes=np.array(cnames, dtype=object))
        # 4. single rescue of the types inside the chosen classes, plus random control types
        chosen_cls = top_by(fr_c, cnames, args.top_classes)
        typ = cf.group(dead_idx, labels["type"])
        type_cls = {t: labels["class"][idx[0]] for t, idx in typ.items()}
        inside = sorted([t for t in typ if type_cls[t] in chosen_cls], key=lambda t: -len(typ[t]))[:args.max_types]
        outside = [t for t in sorted(typ) if type_cls[t] not in chosen_cls]
        controls = list(rng.choice(outside, min(args.random_types, len(outside)), replace=False)) if outside else []
        tnames = inside + controls
        vs = [cf.Variant(f"restore {t}", "type", t, restore=typ[t]) for t in tnames]
        fr_t = fractions(go(vs, "type_single"), y_dmg, y_h, cf.SELECT, args.min_deficit)
        # 5. single rescue of neurons inside the most important types
        top_types = top_by(fr_t, tnames, args.top_types)
        vs = []
        for t in top_types:
            idx = typ[t][np.argsort(-gp.syn[typ[t]])][:args.max_neurons_per_type]
            vs += [cf.Variant(f"restore {t} #{body[i]}", "neuron", f"{t}#{body[i]}", restore=np.array([i]))
                   for i in idx]
        if vs:
            go(vs, "neuron_single")
        # 6. cumulative rescue of the top types, per readout
        vs = []
        for r in cf.SELECT:
            s = fr_t[r].dropna()
            if s.empty:
                continue
            order = [tnames[i] for i in s.sort_values(ascending=False).index]
            for k in args.cumulative_k:
                if k <= len(order):
                    vs.append(cf.Variant(f"top{k} for {r}", "cumulative", f"{r}:top{k}",
                                         restore=np.concatenate([typ[t] for t in order[:k]])))
        if vs:
            go(vs, "cumulative")
        # units table: structure-only predictors of every damaged class and every tested type
        urows = [{"seed": dmg.seed, "stage": dmg.stage, "level": "class", "unit": c, "chosen": c in chosen_cls,
                  **gp.unit_row(cls[c])} for c in cnames]
        urows += [{"seed": dmg.seed, "stage": dmg.stage, "level": "type", "unit": t, "class": type_cls[t],
                   "chosen": t in inside, "control": t in controls, **gp.unit_row(typ[t])} for t in tnames]
        pd.DataFrame(rows).to_csv(out / f"{tag}_runs.csv", index=False)
        pd.DataFrame(urows).to_csv(out / f"{tag}_units.csv", index=False)
        meta = {"seed": dmg.seed, "stage": dmg.stage, "input_seed": in_seed, "n_runs": len(rows),
                "n_removed": int(dmg.dead.sum()), "n_classes": len(cnames), "n_types_damaged": len(typ),
                "chosen_classes": chosen_cls, "top_types": top_types, "minutes": round((time.time() - t0) / 60, 1),
                "y_damaged": y_dmg, "y_healthy": y_h}
        (out / f"{tag}.done.json").write_text(json.dumps(meta, indent=1, default=float))
        log(tag, f"{len(rows)} runs in {meta['minutes']} min; classes {len(cnames)}, types tested {len(tnames)}")


# ---------------------------------------------------------------- robust and confirm
def read_units(args):
    u = pd.read_csv(args.units)
    if "level" in u:
        u = u[u.level == "type"]
    return list(dict.fromkeys(u.unit.astype(str)))


MODELS = [("base", 1.0, 1.0, 1.0, 0), ("w0.8", 0.8, 1.0, 1.0, 0), ("w1.2", 1.2, 1.0, 1.0, 0),
          ("glu0.5", 1.0, 0.5, 1.0, 0), ("da0.5", 1.0, 1.0, 0.5, 0), ("noise1", 1.0, 1.0, 1.0, 1),
          ("noise2", 1.0, 1.0, 1.0, 2)]


def robust(args):
    cx = Connectome(Path(args.pack))
    flies = choose_flies(args, cx)
    labels = cf.unit_labels(cx)
    units = read_units(args)
    w_h0 = cf.healthy_weights(cx)
    out = Path(args.out) / "robust"
    out.mkdir(parents=True, exist_ok=True)
    runners = {}
    for dmg in flies:
        tag = f"s{dmg.stage:g}_seed{dmg.seed}"
        if (out / f"{tag}.csv").exists() and not args.redo:
            log(tag, "already done")
            continue
        typ = cf.group(np.flatnonzero(dmg.dead), labels["type"])
        here = [u for u in units if u in typ]
        rows = []
        for name, ws, gs, das, rep in MODELS:
            if das not in runners:
                runners[das] = cf.Runner(cx, args.batch, "short", sparse=args.sparse, da_scale=das)
            sets = [cf.scale_weights(cx, dmg.w, ws, gs), cf.scale_weights(cx, w_h0, ws, gs)]
            vs = [cf.Variant("damaged", "baseline", "damaged"),
                  cf.Variant("healthy", "baseline", "healthy", wset=1, shift=False, dead=False)]
            vs += [cf.Variant(f"restore {u}", "type", u, restore=typ[u]) for u in here]
            r = batched(runners[das], sets, vs, dmg, args.input_seed_base + dmg.seed + 1_000_000 * rep)
            rows += rows_of(vs, r, dmg, model=name)
            log(tag, name, f"{len(vs)} runs")
        pd.DataFrame(rows).to_csv(out / f"{tag}.csv", index=False)


def confirm(args):
    cx = Connectome(Path(args.pack))
    flies = choose_flies(args, cx)
    labels = cf.unit_labels(cx)
    units = read_units(args)
    w_h = cf.healthy_weights(cx)
    runner = cf.Runner(cx, args.batch, "full", sparse=args.sparse)
    out = Path(args.out) / "confirm"
    out.mkdir(parents=True, exist_ok=True)
    for dmg in flies:
        tag = f"s{dmg.stage:g}_seed{dmg.seed}"
        if (out / f"{tag}.csv").exists() and not args.redo:
            log(tag, "already done")
            continue
        typ = cf.group(np.flatnonzero(dmg.dead), labels["type"])
        here = [u for u in units if u in typ]
        vs = [cf.Variant("damaged", "baseline", "damaged"),
              cf.Variant("healthy", "baseline", "healthy", wset=1, shift=False, dead=False)]
        vs += [cf.Variant(f"restore {u}", "type", u, restore=typ[u]) for u in here]
        vs += [cf.Variant(f"lesion {u}", "lesion", u, wset=1, shift=False, dead=False, lesion=typ[u]) for u in here]
        r = batched(runner, [dmg.w, w_h], vs, dmg, args.input_seed_base + dmg.seed)
        pd.DataFrame(rows_of(vs, r, dmg)).to_csv(out / f"{tag}.csv", index=False)
        log(tag, f"{len(vs)} runs, full protocol")


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["bench", "screen", "robust", "confirm"])
    ap.add_argument("--pack", default=str(PACK), help="connectome pack directory")
    ap.add_argument("--flies-csv", default=str(ROOT / "results/battery/stage_curve_x4.csv"))
    ap.add_argument("--stage", type=float, default=0.5)
    ap.add_argument("--n-flies", type=int, default=24)
    ap.add_argument("--seeds", default="", help="comma-separated fly seeds (overrides --n-flies)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sparse", action="store_true", help="use the sparse kernel")
    ap.add_argument("--compare-sparse", action="store_true", help="bench: time and compare both kernels")
    ap.add_argument("--out", default=str(ROOT / "results/exp1"))
    ap.add_argument("--units", default=str(ROOT / "results/exp1/top_units.csv"))
    ap.add_argument("--input-seed-base", type=int, default=777_000)
    ap.add_argument("--subset-seed", type=int, default=31_000)
    ap.add_argument("--n-subsets", type=int, default=256)
    ap.add_argument("--top-classes", type=int, default=3)
    ap.add_argument("--max-types", type=int, default=400)
    ap.add_argument("--random-types", type=int, default=50)
    ap.add_argument("--top-types", type=int, default=5)
    ap.add_argument("--max-neurons-per-type", type=int, default=20)
    ap.add_argument("--cumulative-k", default="1,3,10,30")
    ap.add_argument("--min-deficit", type=float, default=0.05,
                    help="a readout counts as impaired when |healthy - damaged| exceeds this fraction of healthy")
    ap.add_argument("--pilot", action="store_true", help="small screen for a first timing run")
    ap.add_argument("--redo", action="store_true", help="recompute flies that are already done")
    args = ap.parse_args()
    args.cumulative_k = [int(k) for k in args.cumulative_k.split(",") if k]
    if args.pilot:
        args.n_flies, args.n_subsets, args.max_types = min(args.n_flies, 2), 32, 40
        args.random_types, args.top_types, args.max_neurons_per_type = 10, 2, 5
    log(args.phase, vars(args))
    {"bench": bench, "screen": screen, "robust": robust, "confirm": confirm}[args.phase](args)
    log(args.phase, "done")


if __name__ == "__main__":
    main()
