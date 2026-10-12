"""Analysis of experiment 1 (counterfactual rescue). Needs only the result files.

  python scripts/analyze_exp1.py [--exp1 results/exp1]

For every fly and readout, the restored fraction of a rescue is
(y_rescued - y_damaged) / (y_healthy - y_damaged), computed under identical input spikes.
Writes to results/exp1/summary/:

  baseline_decomposition.csv  how much of each fly's deficit removed neurons, thinned
                              synapses and the threshold shift account for
  class_single.csv            restored fraction for every damaged class, per fly
  class_banzhaf.csv           average marginal effect of restoring a class over random
                              contexts (Banzhaf value), from the random half rescues
  types_ranked.csv            cell types ranked by restored fraction across flies
  neurons_ranked.csv          single neurons inside the top types
  concentration.csv           restored fraction when the top 1, 3, 10, 30 types are restored together
  graph_vs_counterfactual.csv per fly and readout: how well wiring-only predictors rank
                              the counterfactual importance of the tested types
  dynamic_outliers.csv        types more important than the wiring predicts, across flies
  top_units.csv               input to the robust and confirm phases
  summary.json                the headline numbers
and, when present, robust_summary.csv and confirm_summary.csv.
The two conditions under which the screen counts as a finding are tested here:
importance not explained by wiring alone, and rankings stable across model variants.
"""
import argparse
import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

SELECT = ["brain_spikes_A", "LH_hz_A", "KC_hz_A", "MBON_hz_A", "MN9_sugar_hz", "GF_loom_hz"]
ALL_R = SELECT + ["PN_hz_A", "KC_frac_A", "V_pre_A", "GF_latency_ms", "LI"]
PRED = ["n_lost", "syn_lost", "mean_rank", "direct", "twostep"]


def rho(x, y):
    """Spearman correlation, NaN when either input is constant (instead of a warning)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return st.spearmanr(x, y).statistic


def frac_table(runs, readouts, min_deficit):
    """Restored fraction for every run, using that fly's damaged and healthy baselines."""
    out = []
    for seed, g in runs.groupby("seed"):
        b = g[g.level == "baseline"].set_index("unit")
        if not {"damaged", "healthy"} <= set(b.index):
            continue
        yd, yh = b.loc["damaged"], b.loc["healthy"]
        f = g.copy()
        for r in readouts:
            if r not in g:
                continue
            d = yh[r] - yd[r]
            ok = np.isfinite(d) and abs(d) > min_deficit * max(abs(yh[r]), 1e-9)
            f[f"f_{r}"] = (g[r] - yd[r]) / d if ok else np.nan
            f[f"deficit_{r}"] = d
        out.append(f)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def banzhaf(runs_f, subsets_dir):
    """Per fly and readout: least-squares effect of each class from the random half rescues."""
    rows = []
    for npz in sorted(glob.glob(str(subsets_dir / "*_subsets.npz"))):
        z = np.load(npz, allow_pickle=True)
        X, classes = z["X"].astype(float), list(z["classes"])
        seed = int(Path(npz).name.split("_seed")[1].split("_")[0])
        g = runs_f[(runs_f.seed == seed) & (runs_f.level == "subset")].copy()
        g["i"] = g.unit.str.replace("subset", "").astype(int)
        g = g.sort_values("i")
        if len(g) != len(X):
            continue
        D = np.column_stack([np.ones(len(X)), X])
        lam = 1e-3 * len(X)                      # light ridge: stable when classes approach subsets
        for r in SELECT:
            y = g[f"f_{r}"].to_numpy()
            if not np.isfinite(y).all():
                continue
            A = D.T @ D + lam * np.eye(D.shape[1])
            A[0, 0] -= lam                            # no penalty on the intercept
            beta = np.linalg.solve(A, D.T @ y)
            pred = D @ beta
            r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
            for c, b in zip(classes, beta[1:]):
                rows.append({"seed": seed, "readout": r, "class": c, "banzhaf": b, "additive_r2": r2})
    return pd.DataFrame(rows)


def rank_units(f, level, min_flies):
    rows = []
    g = f[f.level == level]
    for r in SELECT:
        col = f"f_{r}"
        if col not in g:
            continue
        s = g.dropna(subset=[col]).groupby("unit")[col]
        agg = pd.DataFrame({"n_flies": s.size(), "median": s.median(), "mean": s.mean(),
                            "q25": s.quantile(0.25), "q75": s.quantile(0.75)})
        agg = agg[agg.n_flies >= min_flies].sort_values("median", ascending=False)
        agg["readout"], agg["rank"] = r, np.arange(1, len(agg) + 1)
        rows.append(agg.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def graph_vs(f, units):
    """Spearman of counterfactual importance with each wiring predictor, and the R^2 of all together."""
    rows, resid = [], []
    t = f[f.level == "type"].merge(units[units.level == "type"].drop(columns=["stage"]), on=["seed", "unit"],
                                   how="left")
    for seed, g0 in t.groupby("seed"):
        for r in SELECT:
            col = f"f_{r}"
            g = g0.dropna(subset=[col])
            if len(g) < 10:
                continue
            row = {"seed": seed, "readout": r, "n_types": len(g)}
            P = {}
            for p in PRED:
                c = p if p in ("n_lost", "syn_lost", "mean_rank") else f"{p}_{r}"
                if c in g:
                    P[p] = np.log1p(np.abs(g[c].to_numpy())) if p != "mean_rank" else g[c].to_numpy()
                    row[f"rho_{p}"] = rho(g[col], g[c])
            M = np.column_stack([np.ones(len(g))] + [P[p] for p in P])
            y = g[col].to_numpy()
            # leave-one-out predictions: an in-sample R^2 would credit the wiring with overfitting
            pred = np.empty(len(y))
            for i in range(len(y)):
                keep = np.arange(len(y)) != i
                beta, *_ = np.linalg.lstsq(M[keep], y[keep], rcond=None)
                pred[i] = M[i] @ beta
            row["r2_wiring_loo"] = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
            rows.append(row)
            sd = np.std(y - pred) or 1.0
            resid += [{"seed": seed, "readout": r, "unit": u, "frac": yy, "predicted": pp, "z_resid": (yy - pp) / sd}
                      for u, yy, pp in zip(g.unit, y, pred)]
    return pd.DataFrame(rows), pd.DataFrame(resid)


def main():
    warnings.filterwarnings("ignore", message="Mean of empty slice")  # readouts without a deficit
    ap = argparse.ArgumentParser()
    root = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
    ap.add_argument("--exp1", default=str(root / "results/exp1"))
    ap.add_argument("--min-deficit", type=float, default=0.05)
    ap.add_argument("--min-flies", type=int, default=3, help="a type is ranked when damaged in at least this many flies")
    ap.add_argument("--top-n", type=int, default=10, help="top types per readout passed to robust/confirm")
    ap.add_argument("--n-controls", type=int, default=10)
    a = ap.parse_args()
    exp1 = Path(a.exp1)
    scr, out = exp1 / "screen", exp1 / "summary"
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(scr / "*_runs.csv")))], ignore_index=True)
    units = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(scr / "*_units.csv")))], ignore_index=True)
    f = frac_table(runs, ALL_R, a.min_deficit)
    summ = {"n_flies": int(f.seed.nunique()), "stages": sorted(f.stage.unique().tolist()), "n_runs": int(len(f))}

    # 1. what kind of damage carries each fly's deficit
    b = f[f.level == "baseline"]
    dec = b.pivot_table(index="seed", columns="unit", values=[f"f_{r}" for r in SELECT]).stack(0, future_stack=True)
    dec.to_csv(out / "baseline_decomposition.csv")
    summ["restored_by"] = {u: {r: float(b[b.unit == u][f"f_{r}"].median()) for r in SELECT}
                           for u in ("restore_dead_all", "restore_syn_all", "no_shift")}

    # 2. classes: single rescue and Banzhaf values
    cs = f[f.level == "class"][["seed", "stage", "unit"] + [f"f_{r}" for r in SELECT]]
    cs.to_csv(out / "class_single.csv", index=False)
    bz = banzhaf(f, scr)
    if not bz.empty:
        agg = bz.groupby(["readout", "class"]).agg(n_flies=("banzhaf", "size"), median=("banzhaf", "median"),
                                                     additive_r2=("additive_r2", "median")).reset_index()
        agg.sort_values(["readout", "median"], ascending=[True, False]).to_csv(out / "class_banzhaf.csv", index=False)
        summ["class_additivity_r2"] = {r: float(bz[bz.readout == r].additive_r2.median()) for r in SELECT
                                       if (bz.readout == r).any()}

    # 3. types and neurons
    rt = rank_units(f, "type", a.min_flies)
    rt.to_csv(out / "types_ranked.csv", index=False)
    rank_units(f, "neuron", 1).to_csv(out / "neurons_ranked.csv", index=False)

    # 4. concentration: restored fraction when the top-k types are restored together
    cu = f[f.level == "cumulative"].copy()
    if not cu.empty:
        cu["readout"] = cu.unit.str.split(":").str[0]
        cu["k"] = cu.unit.str.split("top").str[-1].astype(int)
        cu["frac"] = [row[f"f_{row.readout}"] for _, row in cu.iterrows()]
        conc = cu.groupby(["readout", "k"]).frac.agg(["size", "median", lambda s: s.quantile(0.25),
                                                       lambda s: s.quantile(0.75)]).reset_index()
        conc.columns = ["readout", "k", "n_flies", "median", "q25", "q75"]
        conc.to_csv(out / "concentration.csv", index=False)
        summ["concentration_median"] = {f"{r}|top{k}": float(m) for r, k, m in conc[["readout", "k", "median"]].values}

    # 5. is counterfactual importance more than wiring? (condition 1)
    gv, resid = graph_vs(f, units)
    gv.to_csv(out / "graph_vs_counterfactual.csv", index=False)
    if not gv.empty:
        summ["wiring_r2_loo_median"] = {r: float(gv[gv.readout == r].r2_wiring_loo.median()) for r in SELECT
                                        if (gv.readout == r).any()}
        outl = resid.groupby(["readout", "unit"]).agg(n_flies=("z_resid", "size"), z_median=("z_resid", "median"),
                                                      frac_median=("frac", "median")).reset_index()
        outl = outl[outl.n_flies >= a.min_flies].sort_values(["readout", "z_median"], ascending=[True, False])
        outl.to_csv(out / "dynamic_outliers.csv", index=False)

    # 6. units for the robust and confirm phases: top types per readout, and controls
    if not rt.empty:
        top = rt[rt["rank"] <= a.top_n][["readout", "unit", "median", "n_flies"]].assign(role="top")
        tested = set(f[(f.level == "type")].unit)
        rest = sorted(tested - set(top.unit))
        rng = np.random.default_rng(0)
        ctrl = pd.DataFrame({"unit": rng.choice(rest, min(a.n_controls, len(rest)), replace=False)}) if rest else \
            pd.DataFrame({"unit": []})
        ctrl = ctrl.assign(readout="control", median=np.nan, n_flies=np.nan, role="control")
        tu = pd.concat([top, ctrl], ignore_index=True).assign(level="type")
        tu.to_csv(exp1 / "top_units.csv", index=False)
        summ["n_top_units"] = int(tu.unit.nunique())

    # 7. robust: are rankings stable across model variants? (condition 2)
    rob = sorted(glob.glob(str(exp1 / "robust" / "*.csv")))
    if rob:
        R = pd.concat([pd.read_csv(p) for p in rob], ignore_index=True)
        rows = []
        for m, g in R.groupby("model"):
            fm = frac_table(g, SELECT, a.min_deficit)
            fm = fm[fm.level == "type"]
            for r in SELECT:
                if f"f_{r}" in fm:
                    rows.append(fm.groupby("unit")[f"f_{r}"].median().rename(f"{m}|{r}"))
        M = pd.concat(rows, axis=1)
        rs = []
        for c in M.columns:
            m, r = c.split("|")
            base = M.get(f"base|{r}")
            if m == "base" or base is None:
                continue
            ok = M[[c, f"base|{r}"]].dropna()
            top_b = set(ok[f"base|{r}"].nlargest(10).index)
            top_m = set(ok[c].nlargest(10).index)
            rs.append({"model": m, "readout": r, "n_units": len(ok),
                       "spearman_vs_base": rho(ok[c], ok[f"base|{r}"]),
                       "top10_overlap": len(top_b & top_m) / max(len(top_b), 1)})
        rsd = pd.DataFrame(rs)
        rsd.to_csv(out / "robust_summary.csv", index=False)
        summ["robust_spearman_median"] = {m: float(g.spearman_vs_base.median()) for m, g in rsd.groupby("model")}

    # 8. confirm: necessity and sufficiency with the full protocol
    con = sorted(glob.glob(str(exp1 / "confirm" / "*.csv")))
    if con:
        C = frac_table(pd.concat([pd.read_csv(p) for p in con], ignore_index=True), ALL_R, a.min_deficit)
        rows = []
        for (lvl, u), g in C[C.level.isin(["type", "lesion"])].groupby(["level", "unit"]):
            row = {"unit": u, "test": "necessity: restored fraction" if lvl == "type"
                   else "sufficiency: deficit produced, as fraction of the full deficit", "n_flies": len(g)}
            for r in SELECT + ["LI"]:
                col = f"f_{r}"
                if col in g:
                    v = g[col] if lvl == "type" else 1 - g[col]     # lesion alone: share of the deficit reproduced
                    row[r] = float(v.median())
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / "confirm_summary.csv", index=False)

    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
