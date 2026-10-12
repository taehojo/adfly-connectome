"""Facts table of experiment 1 (counterfactual rescue). Needs only result files.

  python code/scripts/facts_exp1.py

Reads results/exp1/ (screen, robust, confirm, summary), results/battery/stage_curve_x4.csv,
results/paper/output_neuron_loss.json, results/stats/exp1_stats.json,
results/validation/baseline_nostim.parquet + hub_vulnerability_u.npz, results/calibration/
(ff_al_test.json, matched_type_input_synapses.csv), and the code that defines the design
(code/scripts/exp1_counterfactual.py, code/adfly/counterfactual.py, code/adfly/model.py,
code/adfly/pathology.py). Writes results/facts/facts_exp1.json: one flat dict of named
facts plus a "_meaning" dict that says what each key is.

Definitions (as in exp1_stats.py / analyze_exp1.py):
  restored fraction f = (y_variant - y_damaged) / (y_healthy - y_damaged), within one fly
  under identical input spikes; a readout is left out for a fly (NaN) when
  |y_healthy - y_damaged| <= 5% of |y_healthy|. "Deficit" in every meaning means this
  kept difference, in EITHER direction: fly-readout pairs whose damaged value is above
  healthy are kept with a negative denominator, as in analyze_exp1.frac_table (screen: KC 2
  flies, MBON 4, MN9 4; see definitions.deficit and k.n_flies_damaged_above_healthy.*).
  Values > 1 mean the variant moved the readout past healthy (overshoot). Sensitivity
  medians without the above-healthy flies are in the *.all_deficit_only keys.
  "share of all removed" = per-fly ratio f_unit / f_restore_dead_all over flies with
  f_restore_dead_all > 0.1, then the median (exp1_stats.py share_of_all_removed).
Summary lists: a median of a per-fly quantity with an interval is given under its plain key
as [median, 95% bootstrap CI low, high, n flies], resampling flies within the group (4,000
resamples, numpy default_rng(SEED)).
Interquartile ranges [median, q25, q75, n] appear only under keys ending in ".iqr", the
j.r2_loo_iqr_* keys (whose names say so) and b.coverage_neurons_450 (quartiles labelled).
Other lists ([median, min, max, n], [min, max], counts) say in their meaning what each element is.
Readout short names: WB whole-brain spike count during odour A (brain_spikes_A),
LH lateral horn mean rate (LH_hz_A), KC Kenyon cell mean rate (KC_hz_A), MBON mean rate
(MBON_hz_A), MN9 rate during sucrose (MN9_sugar_hz), GF giant fibre rate during looming
(GF_loom_hz).
Hub vulnerability ranks of single neurons come from results/validation/hub_vulnerability_u.npz
(u of every neuron in the order of the MaleCNS neuron table results/validation/baseline_nostim.parquet;
checked here against the screen's mean_rank of the odour-A PN type rows and the DNp01 ranks
of results/paper/output_neuron_loss.json).
"""
import ast
import glob
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())  # code/scripts locally, scripts/ on the server
sys.path.insert(0, str(ROOT / "code/scripts"))
import analyze_exp1 as ax  # noqa: E402  (frac_table, banzhaf, graph_vs, rho; definitions only)

warnings.filterwarnings("ignore")
E1 = ROOT / "results/exp1"
SCR = E1 / "screen"
OUT = ROOT / "results/facts"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261009
NBOOT = 4000
MIN_DEF = 0.05
SEL = ["brain_spikes_A", "LH_hz_A", "KC_hz_A", "MBON_hz_A", "MN9_sugar_hz", "GF_loom_hz"]
R = {"brain_spikes_A": "WB", "LH_hz_A": "LH", "KC_hz_A": "KC", "MBON_hz_A": "MBON",
     "MN9_sugar_hz": "MN9", "GF_loom_hz": "GF"}
RLONG = {"WB": "whole-brain spike count (odour A)", "LH": "lateral horn rate (odour A)",
         "KC": "Kenyon cell rate (odour A)", "MBON": "MBON rate (odour A)", "MN9": "MN9 rate (sucrose)",
         "GF": "giant fibre rate (looming)"}
ODOUR = SEL[:4]
HL = ["ALPN", "sc:descending_neuron", "MBON", "sc:cb_intrinsic", "Kenyon_Cell"]

F_OUT, MEANING = {}, {}


def put(key, value, meaning):
    F_OUT[key] = value
    MEANING[key] = meaning


def rnd(x, k=4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    return None if not np.isfinite(x) else round(x, k)


def sig(p, k=3):
    """p-value to k significant digits."""
    return None if p is None or not np.isfinite(p) else float(f"{float(p):.{k}g}")


def boot_ci(v, fn=np.median):
    v = np.asarray(pd.Series(v, dtype=float).dropna())
    if len(v) < 2:
        return [None, None]
    rng = np.random.default_rng(SEED)
    b = fn(v[rng.integers(0, len(v), (NBOOT, len(v)))], axis=1)
    return [rnd(np.percentile(b, 2.5)), rnd(np.percentile(b, 97.5))]


def mci(v):
    """[median, ci95_lo, ci95_hi, n] over flies."""
    v = pd.Series(v, dtype=float).dropna()
    if v.empty:
        return [None, None, None, 0]
    return [rnd(v.median())] + boot_ci(v) + [int(len(v))]


def miqr(v):
    """[median, q25, q75, n]."""
    v = pd.Series(v, dtype=float).dropna()
    if v.empty:
        return [None, None, None, 0]
    return [rnd(v.median()), rnd(v.quantile(0.25)), rnd(v.quantile(0.75)), int(len(v))]


def rng_(v):
    v = pd.Series(v, dtype=float).dropna()
    return [rnd(v.min()), rnd(v.max())] if len(v) else [None, None]


MCI = "[median, 95% bootstrap CI low, high, n flies]"
MIQR = "[median, q25, q75, n]"


def pm(key, v, stem):
    """Median of a per-fly quantity: 95% bootstrap CI under key, interquartile range under key + '.iqr'."""
    put(key, mci(v), f"{stem} {MCI}; interquartile range in {key}.iqr.")
    put(key + ".iqr", miqr(v), f"{stem} {MIQR} (interquartile range, not a CI; CI in {key}).")

# ================================================================== load
runs = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(SCR / "*_runs.csv")))], ignore_index=True)
units = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(SCR / "*_units.csv")))], ignore_index=True)
meta = pd.DataFrame([json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(SCR / "*.done.json")))])
meta = meta.set_index("seed")
F = ax.frac_table(runs, SEL, MIN_DEF)
stage_of = meta.stage.to_dict()
seeds = sorted(meta.index)
raw_b = runs[runs.level == "baseline"].pivot_table(index="seed", columns="unit",
                                                   values=SEL + ["PN_hz_A", "GF_latency_ms"])
B = F[F.level == "baseline"].pivot_table(index="seed", columns="unit", values=[f"f_{r}" for r in SEL])
dead_f = {r: B[(f"f_{r}", "restore_dead_all")] for r in SEL}
CL = F[F.level == "class"]
T = F[F.level == "type"]
NEU = F[F.level == "neuron"]
CU = F[F.level == "cumulative"].copy()
CU["r"] = CU.unit.str.split(":").str[0]
CU["k"] = CU.unit.str.split("top").str[-1].astype(int)
CU["frac"] = [row[f"f_{row.r}"] for _, row in CU.iterrows()]
UC = units[units.level == "class"]
UT = units[units.level == "type"]
cur = json.loads((ROOT / "results/stats/exp1_stats.json").read_text())

put("design.n_flies", int(len(seeds)), "Number of screen flies in experiment 1 (results/exp1/screen/*.done.json).")
put("design.n_flies_by_stage", [int((meta.stage == 0.3).sum()), int((meta.stage == 0.5).sum())],
    "Screen flies at s = 0.3 and s = 0.5.")
put("design.seeds_s0.3", [min(s for s in seeds if stage_of[s] == 0.3), max(s for s in seeds if stage_of[s] == 0.3)],
    "First and last fly seed of the s = 0.3 screen flies (consecutive seeds).")
put("design.seeds_s0.5", [min(s for s in seeds if stage_of[s] == 0.5), max(s for s in seeds if stage_of[s] == 0.5)],
    "First and last fly seed of the s = 0.5 screen flies (consecutive seeds).")
put("design.n_screen_runs", int(len(runs)), "Total simulations in the screen phase (rows of all *_runs.csv).")
put("design.screen_runs_by_level", [[k, int(v)] for k, v in runs.level.value_counts().items()],
    "Screen simulations by level (type, subset, neuron, class, cumulative, baseline).")
sc = pd.read_csv(ROOT / "results/battery/stage_curve_x4.csv")
scs = sc[sc.seed.isin(seeds)]
put("design.screen_flies_are_default_hub_map", bool(len(scs) == len(seeds) and (scs.kw == "{}").all()
                                                    and (scs.scope == "pan").all()),
    "All 48 screen seeds are severity-series rows in stage_curve_x4.csv with scope=pan and kw='{}' (default hub map), "
    "so experiment 1 cannot contrast vulnerability maps.")
put("design.input_seed_is_777000_plus_fly_seed", bool((meta.input_seed == 777000 + meta.index).all()),
    "Screen input seed equals 777,000 + fly seed in every done.json (same seed used by the robust base model and confirm).")

# ---------------------------------------------------------------- code-derived design constants
src = (ROOT / "code/scripts/exp1_counterfactual.py").read_text(encoding="utf-8")


def argdef(name):
    m = re.search(r'add_argument\("--%s",\s*(?:type=\w+,\s*)?default=("[^"]*"|[^,\)]+)' % re.escape(name), src)
    return ast.literal_eval(m.group(1).replace("_", "")) if m else None


for name in ["top-classes", "max-types", "random-types", "top-types", "max-neurons-per-type", "n-subsets",
             "min-deficit", "input-seed-base", "cumulative-k"]:
    put(f"design.code.{name}", argdef(name),
        f"Default of --{name} in code/scripts/exp1_counterfactual.py (used for the screen).")
put("design.code.inside_type_sort_key", "-len(typ[t]) (number of removed neurons of the type in that fly)"
    if "key=lambda t: -len(typ[t])" in src else "not found",
    "Sort key for types inside the chosen classes in exp1_counterfactual.py screen() step 4 (then [:max_types]).")
put("design.selection_rule",
    "Per fly: rescue every damaged class singly; per readout take the 3 classes with the largest single-class "
    "restored fraction and take the union (chosen classes); pool all damaged types of the chosen classes, sort by "
    "number of removed neurons in that fly (descending) and keep the first 400; add 50 random control types drawn "
    "from types outside the chosen classes (rng default_rng(31000+seed)); single neurons: union over readouts of "
    "the 5 most effective tested types, up to 20 neurons per type with the most synapses (in+out); cumulative: top "
    "1, 3, 10, 30 types per readout in order of their single-type fraction.",
    "Exact selection rule of the type screen, read from exp1_counterfactual.py screen().")
an_src = (ROOT / "code/scripts/analyze_exp1.py").read_text(encoding="utf-8")
m = re.search(r'"--min-flies", type=int, default=(\d+)', an_src)
put("design.code.rank_min_flies", int(m.group(1)) if m else None,
    "analyze_exp1.py --min-flies: a type is ranked (and can be carried forward) only if tested with a deficit in at "
    "least this many of the 48 screen flies (both stages pooled).")
msrc = (ROOT / "code/adfly/model.py").read_text(encoding="utf-8")
odors = ast.literal_eval(re.search(r"ODORS = (\{.*\})", msrc).group(1))
put("design.code.odour_A_glomeruli", odors["A"],
    "code/adfly/model.py ODORS['A']: glomeruli whose uniglomerular ALPNs receive the odour-A Poisson input "
    "(Connectome.upn regex ^(DM1|DM4|VA2)_ on class ALPN; 150 Hz per neuron).")
csrc = (ROOT / "code/adfly/counterfactual.py").read_text(encoding="utf-8")
put("design.code.loom_input_types", "LPLC2, LC4 (left side)" if 'cx.idx(["LPLC2", "LC4"], ("L",))' in csrc else "?",
    "counterfactual.py Runner: the looming stimulus is injected into LPLC2 and LC4 neurons on the left side.")
vw = {}
for tree_ in (ast.parse(csrc), ast.parse(src)):
    for nd in ast.walk(tree_):
        if isinstance(nd, ast.Call) and getattr(nd.func, "attr", getattr(nd.func, "id", None)) == "Variant":
            lev_ = nd.args[1].value
            lab_ = nd.args[0].value if isinstance(nd.args[0], ast.Constant) else None
            w_ = [k.value.value for k in nd.keywords if k.arg == "wset"]
            vw.setdefault(lab_ if lev_ == "baseline" else lev_, set()).add(int(w_[0]) if w_ else 0)
assert all(len(v) == 1 for v in vw.values()), vw
assert re.search(r"wset: int = 0\b", csrc), "Variant.wset default is no longer 0"
vw = {k: v.pop() for k, v in vw.items()}
NEURON_RESTORE = ["restore_dead_all", "class", "type", "neuron", "subset", "cumulative"]
assert all(vw[k] == 0 for k in NEURON_RESTORE) and vw["restore_syn_all"] == 1 and vw["lesion"] == 1, vw
put("design.code.variant_weight_set", [[k, int(vw[k])] for k in ["damaged", "restore_dead_all", "restore_syn_all", "no_shift",
                                                                  "healthy", "class", "type", "neuron", "subset",
                                                                  "cumulative", "lesion"]],
    "Weight set of each variant, parsed from the Variant(...) calls in counterfactual.py and exp1_counterfactual.py: "
    "[variant or level, wset] with 0 = the fly's damaged (thinned) weights, 1 = healthy weights.")
psrc = (ROOT / "code/adfly/pathology.py").read_text(encoding="utf-8")
thin_all = bool(re.search(r"keep_p = 1\.0 - load\[src\] \* stage\.syn_loss_max", psrc))
put("design.code.restored_neurons_keep_threshold_shift",
    bool(re.search(r"shift: bool = True", csrc) and re.search(r"if v\.shift:\s*\n\s*vth -= dmg\.dth", csrc)) and thin_all,
    "counterfactual.py: Variant.shift defaults to True and vth_vector subtracts the fly's dth from restored neurons, so "
    "restored neurons return at the shifted threshold (v_th - 2 mV x load), not the healthy value. The neuron-restoring "
    "variants (restore_dead_all and the class, type, neuron, subset and cumulative levels) run on the damaged weight set "
    "(wset=0), and pathology.apply thins the outgoing contacts of every loaded neuron, removed ones included "
    "(keep_p = 1 - load[src] x syn_loss_max), so a restored neuron also comes back with its own thinned outgoing "
    "contacts. restore_syn_all is the exception: it uses the healthy weights (wset=1) and keeps the neuron removal and "
    "threshold shift (design.code.variant_weight_set).")
pp = {k: float(re.search(r"%s: float = ([\d.]+)" % k, psrc).group(1)) for k in ("dth_max", "syn_loss_max", "death_max")}
put("design.code.pathology_max", [pp["dth_max"], pp["syn_loss_max"], pp["death_max"]],
    "code/adfly/pathology.py Stage defaults [dth_max mV, syn_loss_max, death_max]: load l = s x hub rank; "
    "P(removed) = 0.5 l; output contacts kept with p = 1 - 0.6 l; threshold lowered by 2 mV x l.")

# ================================================================== (k) removed neurons, per-readout n
N_total = int(round(np.median(sc[sc.seed.isin(seeds)].set_index("seed").reindex(seeds).dead_frac.rdiv(
    meta.n_removed.reindex(seeds)).dropna())))
put("k.n_neurons_model", N_total, "Neurons in the model, from n_removed / dead_frac of the screen flies.")
pm("k.removed_per_fly_all", meta.n_removed, "Neurons removed per screen fly, pooled over both stages")
for s_ in (0.3, 0.5):
    v = meta[meta.stage == s_].n_removed
    pm(f"k.removed_per_fly_s{s_}", v, f"Neurons removed per screen fly at s = {s_}")
    put(f"k.removed_pct_of_neurons_s{s_}", rnd(100 * v.median() / N_total, 2),
        f"Median neurons removed per fly at s = {s_} as % of all model neurons.")
pm("k.damaged_types_per_fly", meta.n_types_damaged, "Cell types with at least one removed neuron per fly")
pm("k.damaged_classes_per_fly", meta.n_classes, "Classes with at least one removed neuron per fly")
is3 = lambda idx: np.array([stage_of[s] == 0.3 for s in idx])  # noqa: E731
is5 = lambda idx: np.array([stage_of[s] == 0.5 for s in idx])  # noqa: E731
EXCESS = {}
for r in SEL:
    v = dead_f[r]
    dh = raw_b[(r, "healthy")] - raw_b[(r, "damaged")]
    exc = sorted(int(s) for s in v[v.notna()].index if dh[s] < 0)
    EXCESS[r] = exc
    put(f"k.n_flies_with_deficit.{R[r]}", [int(v.notna().sum()), int(v[is3(v.index)].notna().sum()), int(v[is5(v.index)].notna().sum())],
        f"Flies in which the {RLONG[R[r]]} readout is kept, i.e. |healthy - damaged| > 5% of healthy in EITHER direction "
        f"(deficit or excess), [all, s=0.3, s=0.5]; {len(exc)} of them have damaged ABOVE healthy "
        f"(k.n_flies_damaged_above_healthy.{R[r]}), so the count is not a count of flies with a deficit only "
        f"(those are in k.n_flies_damaged_below_healthy.{R[r]}).")
    put(f"k.n_flies_damaged_above_healthy.{R[r]}", [len(exc), int(sum(stage_of[s] == 0.3 for s in exc)),
                                                   int(sum(stage_of[s] == 0.5 for s in exc)), exc],
        f"Kept flies whose damaged {RLONG[R[r]]} is ABOVE healthy by > 5% (negative denominator in every restored fraction): "
        f"[n all, n s=0.3, n s=0.5, seeds]. They enter every pooled restored-fraction median of this readout.")
    below = v.notna() & (dh > 0)
    put(f"k.n_flies_damaged_below_healthy.{R[r]}", [int(below.sum()), int(below[is3(below.index)].sum()), int(below[is5(below.index)].sum())],
        f"Flies whose damaged {RLONG[R[r]]} is BELOW healthy by > 5% (a true deficit), [all, s=0.3, s=0.5].")
put("definitions.deficit",
    "Throughout this file 'deficit' means the fly's healthy-minus-damaged difference of a readout when |healthy - damaged| > "
    "5% of |healthy|, in EITHER direction (analyze_exp1.frac_table, as in exp1_stats.py). Fly-readout pairs whose damaged "
    "value is ABOVE healthy are kept with a negative denominator: screen " +
    ", ".join(f"{R[r]} {len(EXCESS[r])}" for r in SEL) + " flies (seeds in k.n_flies_damaged_above_healthy.*). For them a "
    "positive restored fraction means the variant moved the readout down toward healthy, and a small denominator can give "
    "values far outside 0-1 (fly 1266 MBON). A fraction > 1 means the readout moved past healthy (overshoot). Pooled "
    "medians include these flies; *.all_deficit_only keys repeat the main medians without them. The robust (g., h.) and "
    "confirm (i.) tables use the same rule on their own baselines.",
    "Definition of 'deficit' used by every restored-fraction key.")
dmg_pct = (raw_b.xs("damaged", axis=1, level=1) / raw_b.xs("healthy", axis=1, level=1) * 100)
for r in SEL:
    put(f"k.damaged_pct_of_healthy.{R[r]}", [rnd(dmg_pct[r][is3(dmg_pct.index)].median(), 1),
                                             rnd(dmg_pct[r][is5(dmg_pct.index)].median(), 1)],
        f"Median of per-fly ratios damaged/healthy x 100 for {RLONG[R[r]]} in the screen flies, [s=0.3, s=0.5] (each fly "
        f"against its own healthy copy). NOT the project's '% of healthy' convention (group median / healthy median), which "
        f"is k.damaged_pct_of_healthy_median.{R[r]}.")
    md, mdci = [], []
    for s_ in (0.3, 0.5):
        sel_ = np.array([stage_of[s] == s_ for s in raw_b.index])
        dv = raw_b[(r, "damaged")][sel_].to_numpy(float)
        hv = raw_b[(r, "healthy")][sel_].to_numpy(float)
        pt = 100 * np.median(dv) / np.median(hv)
        ii = np.random.default_rng(SEED).integers(0, len(dv), (NBOOT, len(dv)))
        bt_ = 100 * np.median(dv[ii], axis=1) / np.median(hv[ii], axis=1)
        md.append(rnd(pt, 1))
        mdci.append([rnd(pt, 1), rnd(np.percentile(bt_, 2.5), 1), rnd(np.percentile(bt_, 97.5), 1), int(len(dv))])
    put(f"k.damaged_pct_of_healthy_median.{R[r]}", md,
        f"Median damaged {RLONG[R[r]]} as % of the median of the healthy copies of the same 24 screen flies, [s=0.3, s=0.5] "
        f"(the project's '% of healthy' convention; CI in k.damaged_pct_of_healthy_median_ci.{R[r]}).")
    put(f"k.damaged_pct_of_healthy_median_ci.{R[r]}", mdci,
        f"k.damaged_pct_of_healthy_median.{R[r]} with a 95% bootstrap CI: per stage [value, CI low, CI high, n flies], "
        f"[s=0.3, s=0.5]; flies resampled with their own healthy copies (paired), {NBOOT} resamples.")
d_mn9 = raw_b[("MN9_sugar_hz", "healthy")] - raw_b[("MN9_sugar_hz", "damaged")]
mn9_all = runs.MN9_sugar_hz.dropna().to_numpy()
mn9_int = bool(np.allclose(mn9_all, np.round(mn9_all)))
mn9_odd = bool((np.round(mn9_all) % 2 == 1).any())
assert mn9_int and mn9_odd and 'sug = self._seg(' in csrc and "/ 0.5 / max(len(self.mn9), 1)" in csrc
put("design.code.MN9_rate_resolution_hz", 1.0,
    "MN9_sugar_hz = summed spikes of the MN9 neurons in the 0.5-s sucrose window / 0.5 s / number of MN9 neurons "
    "(counterfactual.py _protocol). Every screen value is an integer and odd values occur, which fixes the number of MN9 "
    "neurons at 2, so the readout moves in steps of 1 Hz (one spike of one MN9 neuron = 2 Hz for that neuron, 1 Hz for "
    "the mean).")
pm("k.MN9_deficit_hz", d_mn9, "MN9 healthy minus damaged rate per screen fly, all 48 flies, signed (negative = damaged above "
   "healthy) (Hz; 1-Hz resolution, design.code.MN9_rate_resolution_hz)")
gf_rel = (raw_b[("GF_loom_hz", "healthy")] - raw_b[("GF_loom_hz", "damaged")]) / raw_b[("GF_loom_hz", "healthy")]
put("k.GF_relative_deficit_counts", [int((gf_rel.abs() <= 0.05).sum()), int(((gf_rel > 0.05) & (gf_rel <= 0.2)).sum()),
                                     int((gf_rel > 0.2).sum())],
    "Screen flies with GF relative deficit <=5% (excluded), 5-20%, >20% of healthy.")

# ================================================================== (f) decomposition by stage
UNITS3 = ["restore_dead_all", "restore_syn_all", "no_shift"]
for r in SEL:
    for u in UNITS3 + ["sum_dead_syn"]:
        v = (B[(f"f_{r}", "restore_dead_all")] + B[(f"f_{r}", "restore_syn_all")]) if u == "sum_dead_syn" else B[(f"f_{r}", u)]
        a3 = v[[stage_of[s] == 0.3 for s in v.index]]
        a5 = v[[stage_of[s] == 0.5 for s in v.index]]
        pm(f"f.{u}.{R[r]}.all", v, f"{u} restored fraction of the {RLONG[R[r]]} deficit, both stages")
        put(f"f.{u}.{R[r]}.all_deficit_only", mci(v.drop(EXCESS[r], errors="ignore")),
            f"As f.{u}.{R[r]}.all but without the {len(EXCESS[r])} flies whose damaged value is above healthy "
            f"(k.n_flies_damaged_above_healthy.{R[r]}), sensitivity check " + MCI + ".")
        put(f"f.{u}.{R[r]}.s0.3", mci(a3), f"{u} restored fraction of the {RLONG[R[r]]} deficit at s = 0.3 " + MCI + ".")
        put(f"f.{u}.{R[r]}.s0.5", mci(a5), f"{u} restored fraction of the {RLONG[R[r]]} deficit at s = 0.5 " + MCI + ".")
        if a3.notna().sum() > 2 and a5.notna().sum() > 2:
            put(f"f.{u}.{R[r]}.mw_p_stage", sig(st.mannwhitneyu(a3.dropna(), a5.dropna(), alternative="two-sided").pvalue),
                f"Two-sided Mann-Whitney p, s = 0.3 vs s = 0.5, for {u} on {RLONG[R[r]]}.")
    # threshold-shift removal, within fly, raw readout
    yd, yns, yh = raw_b[(r, "damaged")], raw_b[(r, "no_shift")], raw_b[(r, "healthy")]
    ok = dead_f[r].notna()
    put(f"f.no_shift_direction.{R[r]}", [int((yns[ok] < yd[ok]).sum()), int((yns[ok] == yd[ok]).sum()),
                                         int((yns[ok] > yd[ok]).sum()), int(ok.sum())],
        f"Removing the threshold shift (all else damaged) in flies with a {R[r]} deficit: [n flies where the raw readout "
        f"fell, unchanged, rose, n flies].")
    pm(f"f.no_shift_change_pct_of_healthy.{R[r]}", ((yns - yd) / yh * 100)[ok],
       f"Within-fly change of {RLONG[R[r]]} when the threshold shift is removed, as % of the fly's healthy value, flies "
       f"with the readout kept")
nr = meta.n_removed
share05 = dead_f["brain_spikes_A"][[stage_of[s] == 0.5 for s in dead_f["brain_spikes_A"].index]]
rr = st.spearmanr(share05, nr.reindex(share05.index))
put("f.WB_neuron_share_vs_n_removed_s0.5", [rnd(rr.statistic, 3), sig(rr.pvalue), int(len(share05))],
    "Spearman [rho, p, n] between restore_dead_all WB fraction and neurons removed, within s = 0.5 flies.")
put("f.exp1_stats_check_dead_all_WB", cur["decomposition"]["all"]["restore_dead_all"]["brain_spikes_A"]["median"],
    "Provenance check: restore_dead_all WB median in results/stats/exp1_stats.json.")

# ================================================================== (a) classes
fstage = lambda df, s_: df[df.stage == s_]  # noqa: E731
for r in SEL:
    c = f"f_{r}"
    for grp, sub in [("all", CL), ("s0.3", fstage(CL, 0.3)), ("s0.5", fstage(CL, 0.5))]:
        agg = sub.dropna(subset=[c]).groupby("unit")[c].agg(["median", "size"])
        agg = agg[agg["size"] >= 5].sort_values("median", ascending=False).head(5)
        put(f"a.class_single_top5.{R[r]}.{grp}", [[u, rnd(m), int(n)] for u, (m, n) in agg.iterrows()],
            f"Top 5 classes by median single-class restored fraction of the {RLONG[R[r]]} deficit ({grp}; only classes with "
            f"a defined fraction in >= 5 flies, i.e. the class was damaged AND the fly's {R[r]} readout was kept "
            f"(|healthy - damaged| > 5% of healthy)): [class, median, n such flies].")
    for cls in HL:
        g = CL[CL.unit == cls]
        for grp, sub in [("all", g), ("s0.3", fstage(g, 0.3)), ("s0.5", fstage(g, 0.5))]:
            put(f"a.class_single.{cls}.{R[r]}.{grp}", mci(sub[c]),
                f"Restoring all removed neurons of class {cls} alone: restored fraction of the {RLONG[R[r]]} deficit, "
                f"{grp} " + MCI + ".")
    # best class per fly
    best = CL.dropna(subset=[c]).groupby("seed")[c].max()
    put(f"a.best_class_per_fly.{R[r]}.all", mci(best), f"Best single class per fly (max over classes), {RLONG[R[r]]} " + MCI + ".")
    put(f"a.best_class_per_fly.{R[r]}.all_deficit_only", mci(best.drop(EXCESS[r], errors="ignore")),
        f"As a.best_class_per_fly.{R[r]}.all but without the {len(EXCESS[r])} flies whose damaged value is above healthy, "
        f"sensitivity check " + MCI + ".")
    for s_ in (0.3, 0.5):
        put(f"a.best_class_per_fly.{R[r]}.s{s_}", mci(best[[stage_of[s] == s_ for s in best.index]]),
            f"Best single class per fly at s = {s_}, {RLONG[R[r]]} " + MCI + ".")
    arg = CL.dropna(subset=[c]).loc[CL.dropna(subset=[c]).groupby("seed")[c].idxmax()]
    put(f"a.best_class_identity.{R[r]}", [[k, int(v)] for k, v in arg.unit.value_counts().head(4).items()],
        f"Which class was the best single class per fly for {RLONG[R[r]]}: [class, n flies] (top 4).")
    d = dead_f[r].reindex(best.index)
    put(f"a.best_class_share_of_all_removed.{R[r]}", mci((best / d)[d > 0.1]),
        f"Best single class / restore_dead_all per fly (flies with restore_dead_all > 0.1), {RLONG[R[r]]} " + MCI + ".")
    # best class vs best type
    bt = T.dropna(subset=[c]).groupby("seed")[c].max()
    put(f"a.best_type_per_fly.{R[r]}.all", mci(bt), f"Best single tested type per fly (max over the 450), {RLONG[R[r]]} " + MCI + ".")
    put(f"a.best_type_per_fly.{R[r]}.all_deficit_only", mci(bt.drop(EXCESS[r], errors="ignore")),
        f"As a.best_type_per_fly.{R[r]}.all but without the {len(EXCESS[r])} flies whose damaged value is above healthy, "
        f"sensitivity check " + MCI + ".")
    for s_ in (0.3, 0.5):
        pm(f"a.best_type_per_fly.{R[r]}.s{s_}", bt[[stage_of[s] == s_ for s in bt.index]],
           f"Best single tested type per fly at s = {s_}, {RLONG[R[r]]}")
    if R[r] in ("WB", "MN9"):
        put(f"a.best_type_per_fly.{R[r]}.mw_p_stage",
            sig(st.mannwhitneyu(bt[[stage_of[s] == 0.3 for s in bt.index]], bt[[stage_of[s] == 0.5 for s in bt.index]],
                                alternative="two-sided").pvalue),
            f"Two-sided Mann-Whitney p, best tested type per fly s = 0.3 vs 0.5, {RLONG[R[r]]}.")
    put(f"a.best_type_share_over_0.1.{R[r]}", rnd((bt > 0.1).mean(), 3),
        f"Share of flies whose best tested type restored >10% of the {RLONG[R[r]]} deficit.")
    # highlighted class share of all removed
    for cls in HL:
        g = CL[CL.unit == cls].set_index("seed")[c]
        dd = dead_f[r].reindex(g.index)
        put(f"a.class_share_of_all_removed.{cls}.{R[r]}", mci((g / dd)[dd > 0.1]),
            f"Class {cls} single rescue / restore_dead_all per fly (flies with restore_dead_all > 0.1), {RLONG[R[r]]} " + MCI + ".")
    # sum of single-class rescues vs restoring all removed
    sc_sum = CL.groupby("seed")[c].sum(min_count=1)
    pm(f"a.sum_single_classes.{R[r]}", sc_sum, f"Sum over classes of single-class restored fractions per fly, {RLONG[R[r]]}")
    dd = dead_f[r].reindex(sc_sum.index)
    pm(f"a.sum_single_classes_over_all_removed.{R[r]}", (sc_sum / dd)[dd > 0.1],
       f"Per-fly ratio sum of single-class fractions / restore_dead_all (restore_dead_all > 0.1), {RLONG[R[r]]}")

# n lost per class
for cls in HL + ["sc:cb_motor"]:
    g = UC[UC.unit == cls].set_index("seed")
    pm(f"a.n_lost.{cls}", g.n_lost, f"Removed neurons of class {cls} per fly (flies with the class damaged)")
    put(f"a.n_lost_pct_of_removed.{cls}", rnd(100 * (g.n_lost / meta.n_removed.reindex(g.index)).median(), 3),
        f"Median per-fly % of removed neurons that belong to class {cls}.")
put("a.check_class_nlost_sums_to_removed", bool((UC.groupby("seed").n_lost.sum() == meta.n_removed).all()),
    "Check: class n_lost in units.csv sums to n_removed in done.json for every fly.")
pm("a.chosen_classes_per_fly", [len(x) for x in meta.chosen_classes],
   "Number of chosen classes per fly (union of top 3 per readout)")
put("a.chosen_classes_range", rng_([len(x) for x in meta.chosen_classes]), "Range of chosen classes per fly.")
for cls in HL:
    put(f"a.n_flies_class_chosen.{cls}", int(sum(cls in x for x in meta.chosen_classes)),
        f"Flies (of 48) in which class {cls} was a chosen class, so its types could be tested inside the 400.")

# Banzhaf, recomputed
bz = ax.banzhaf(F, SCR)
bza = bz.groupby(["readout", "class"]).agg(n_flies=("banzhaf", "size"), median=("banzhaf", "median")).reset_index()
old = pd.read_csv(E1 / "summary/class_banzhaf.csv")
chk = bza.merge(old, on=["readout", "class"], suffixes=("", "_old"))
put("a.banzhaf_check_max_abs_diff_vs_summary_csv", rnd((chk["median"] - chk["median_old"]).abs().max(), 8),
    "Max |recomputed - results/exp1/summary/class_banzhaf.csv| over class Banzhaf medians (provenance check).")
for r in SEL:
    g = bza[(bza.readout == r) & (bza.n_flies >= 10)].sort_values("median", ascending=False).head(3)
    put(f"a.banzhaf_top3.{R[r]}", [[cl, rnd(m, 3), int(n)] for cl, m, n in g[["class", "median", "n_flies"]].values],
        f"Top 3 classes by median Banzhaf value (ridge fit on 256 random half-class rescues per fly) for {RLONG[R[r]]}, "
        f"among classes damaged in >= 10 flies: [class, median, n flies].")
    put(f"a.additive_r2.{R[r]}", rnd(bz[bz.readout == r].groupby("seed").additive_r2.first().median(), 3),
        f"Median per-fly R^2 of the additive class model on the 256 random half rescues, {RLONG[R[r]]}.")
    for cls in ["ALPN", "MBON", "sc:descending_neuron", "sc:cb_intrinsic", "Kenyon_Cell"]:
        g = bz[(bz.readout == r) & (bz["class"] == cls)]
        if len(g):
            put(f"a.banzhaf.{cls}.{R[r]}", mci(g.banzhaf), f"Banzhaf value of class {cls} on {RLONG[R[r]]} " + MCI + ".")

# ---------------------------------------------------------------- ALPN and the stimulated PNs
alpn = CL[CL.unit == "ALPN"].set_index("seed")
alpn_raw = runs[(runs.level == "class") & (runs.unit == "ALPN")].set_index("seed")
nz = (alpn_raw.brain_spikes_A != raw_b[("brain_spikes_A", "damaged")].reindex(alpn_raw.index))
all0 = pd.concat([(alpn_raw[r] == raw_b[(r, "damaged")].reindex(alpn_raw.index)) for r in SEL], axis=1).all(axis=1)
od0 = pd.concat([(alpn_raw[r] == raw_b[(r, "damaged")].reindex(alpn_raw.index)) for r in ODOUR + ["PN_hz_A"]],
                axis=1).all(axis=1)
pnr = raw_b[("PN_hz_A", "damaged")] / raw_b[("PN_hz_A", "healthy")]
nzs = nz[nz].index
put("a.ALPN.n_flies_damaged", int(len(alpn)), "Screen flies in which at least one ALPN was removed.")
put("a.ALPN.n_flies_nonzero_WB", [int(nz.sum()), int(sum(stage_of[s] == 0.3 for s in nzs)), int(sum(stage_of[s] == 0.5 for s in nzs))],
    "Flies where restoring the ALPN class changed the WB readout at all [all, s=0.3, s=0.5]; inferred to be the flies "
    "that lost at least one odour-A (stimulated) PN, since every tested non-odour-A ALPN type changed no odour readout "
    "(see a.ALPN_type_tests).")
put("a.ALPN.n_flies_exactly_zero_odour_readouts", [int(od0.sum()), int(all0.sum())],
    "Flies where restoring the ALPN class left the odour-A readouts (WB, LH, PN, KC, MBON) bit-identical to the damaged "
    "fly [n flies, n of them also unchanged on MN9 and GF].")
put("a.ALPN.zero_flies_seeds", [int(s) for s in od0[od0].index],
    "Seeds of the flies where the ALPN class rescue changed no odour-A readout.")
put("a.ALPN.PN_ratio_zero_flies_range", rng_(pnr.reindex(od0[od0].index)),
    "Range of damaged/healthy ALPN mean rate in the flies where the ALPN rescue changed no odour readout.")
# every odour-A driven PN with its own hub rank u, from the full MaleCNS neuron table and u table
pat = r"^(%s)_" % "|".join(odors["A"])
neu = pd.read_parquet(ROOT / "results/validation/baseline_nostim.parquet")
uz = np.load(ROOT / "results/validation/hub_vulnerability_u.npz")
assert np.array_equal(uz["ids"], neu.bodyId.to_numpy()), "u table is not in neuron-table order"
U = np.asarray(uz["u"], float)
tN = neu.type.fillna("")
mA = tN.str.match(pat).to_numpy()
ffal = json.loads((ROOT / "results/calibration/ff_al_test.json").read_text())
assert {int(x["n_PN_A"]) for x in ffal} == {int(mA.sum())}, "class-ALPN filter (cx.upn) and type regex disagree"
uA = pd.DataFrame({"type": tN[mA].to_numpy(), "side": neu.side[mA].to_numpy(), "u": U[mA]}).sort_values(["type", "side"])
n_chk = 0
for x in UT[UT.unit.str.match(pat)].itertuples():
    uu = uA[uA.type == x.unit].u.to_numpy()
    if x.n_lost == len(uu):
        assert abs(x.mean_rank - uu.mean()) < 1e-9, x
    else:
        assert x.n_lost == 1 and np.abs(uu - x.mean_rank).min() < 1e-9, x
    n_chk += 1
onl_u = json.loads((ROOT / "results/paper/output_neuron_loss.json").read_text())["vulnerability_percentile"]
for sd in ("L", "R"):
    assert abs(U[((tN == "DNp01") & (neu.side == sd)).to_numpy()][0] - onl_u[f"DNp01_{sd}"]) < 1e-9
put("a.ALPN.stimulated_PN_u", [[a, b, rnd(c, 4)] for a, b, c in uA[["type", "side", "u"]].values],
    "Every odour-A driven PN (class ALPN, type ^(DM1|DM4|VA2)_) with its hub vulnerability rank u: [type, side, u], from "
    "results/validation/hub_vulnerability_u.npz in the order of the MaleCNS neuron table "
    f"(results/validation/baseline_nostim.parquet). u checked against the screen mean_rank of all {n_chk} odour-A PN type "
    "rows and the DNp01 ranks of results/paper/output_neuron_loss.json.")
pexp = {s_: 1 - np.prod(1 - pp["death_max"] * s_ * uA.u.to_numpy()) for s_ in (0.3, 0.5)}
put("a.ALPN.expected_flies_losing_stimulated_PN", [rnd(24 * pexp[0.3], 1), rnd(24 * pexp[0.5], 1)],
    f"Expected number of the 24 screen flies per stage [s=0.3, s=0.5] that lose at least one of the {int(mA.sum())} "
    "odour-A driven PNs (a.ALPN.stimulated_PN_u, DM4_vPN included), P(removed) = 0.5 x s x u per neuron, independent; "
    "compare a.ALPN.n_flies_nonzero_WB.")
mA3 = uA.type != "DM4_vPN"
pexp3 = {s_: 1 - np.prod(1 - pp["death_max"] * s_ * uA.u[mA3].to_numpy()) for s_ in (0.3, 0.5)}
put("a.ALPN.expected_flies_losing_stimulated_PN_without_DM4_vPN", [rnd(24 * pexp3[0.3], 1), rnd(24 * pexp3[0.5], 1)],
    "Same expectation leaving out the two DM4_vPN neurons (an approximation that uses only DM1_lPN, DM4_adPN and "
    "VA2_adPN); shown for comparison.")
for r in ODOUR:
    put(f"a.ALPN.n_flies_nonzero.{R[r]}", int((alpn[f"f_{r}"].fillna(0) != 0).sum()),
        f"Flies where the ALPN class rescue gave a non-zero restored fraction of the {RLONG[R[r]]} deficit.")
put("a.ALPN.PN_ratio_nonzero_vs_zero", [rnd(pnr.reindex(nzs).median(), 3), rnd(pnr.reindex(nz[~nz].index).median(), 3)],
    "Median damaged/healthy ALPN mean rate (PN_hz_A) in flies where the ALPN rescue changed WB vs flies where it did not.")
put("a.ALPN.PN_ratio_lt_0.95", [int(((pnr < 0.95) & pd.Series([stage_of[s] == 0.3 for s in pnr.index], pnr.index)).sum()),
                                int(((pnr < 0.95) & pd.Series([stage_of[s] == 0.5 for s in pnr.index], pnr.index)).sum())],
    "Flies with ALPN rate reduced by more than 5% (damaged/healthy < 0.95), [s=0.3, s=0.5] of 24 each.")
for r in ODOUR:
    c = f"f_{r}"
    put(f"a.ALPN.nonzero_flies.{R[r]}", mci(alpn.loc[nzs, c]),
        f"ALPN class restored fraction of {RLONG[R[r]]} deficit in the flies where it changed WB " + MCI + ".")
    rat = alpn[c] / dead_f[r].reindex(alpn.index)
    put(f"a.ALPN.share_of_all_removed_nonzero_flies.{R[r]}", [rnd(rat.reindex(nzs).median(), 3),
                                                              rnd(rat.reindex(nzs)[dead_f[r].reindex(nzs) > 0.1].median(), 3)],
        f"ALPN class / restore_dead_all per fly, median over WB-nonzero flies [no filter, restore_dead_all > 0.1], {RLONG[R[r]]}.")
    for s_ in (0.3, 0.5):
        sub = rat[[stage_of[s] == s_ for s in rat.index] & (dead_f[r].reindex(rat.index) > 0.1).to_numpy()]
        put(f"a.ALPN.share_of_all_removed.s{s_}.{R[r]}", mci(sub),
            f"ALPN class / restore_dead_all per fly at s = {s_} (flies with restore_dead_all > 0.1), {RLONG[R[r]]} " + MCI + ".")
    sp = st.spearmanr(pnr.reindex(alpn.index), alpn[c], nan_policy="omit")
    put(f"a.ALPN.spearman_PNratio_vs_rescue.{R[r]}", [rnd(sp.statistic, 3), sig(sp.pvalue)],
        f"Spearman [rho, p] across flies between damaged/healthy ALPN rate and the ALPN class rescue of {RLONG[R[r]]}.")
# stimulated PN types in the type screen
pat = r"^(%s)_" % "|".join(odors["A"])
oa = T[T.unit.str.match(pat) & T.seed.isin(UT[UT["class"] == "ALPN"].seed)]
oa = oa.merge(UT[["seed", "unit", "n_lost", "chosen", "mean_rank"]], on=["seed", "unit"])
put("a.odourA_PN_type_tests", [[int(x.seed), x.unit, int(x.n_lost)] + [rnd(x[f"f_{r}"], 3) for r in ODOUR] for _, x in oa.iterrows()],
    "Every single-type test of an odour-A PN type (ALPN types of DM1/DM4/VA2): [seed, type, neurons restored, "
    "restored fraction WB, LH, KC, MBON].")
put("a.odourA_PN_types_tested_n", [int(len(oa)), int(oa.seed.nunique())],
    "Odour-A PN type tests in the screen [n tests, n distinct flies] out of 48 flies.")
put("a.odourA_PN_type_ALPN_class_same_fly", [[int(s), rnd(alpn.loc[s, "f_brain_spikes_A"], 3), rnd(alpn.loc[s, "f_LH_hz_A"], 3),
                                              rnd(alpn.loc[s, "f_KC_hz_A"], 3), rnd(alpn.loc[s, "f_MBON_hz_A"], 3)]
                                             for s in sorted(oa.seed.unique())],
    "ALPN class rescue in the flies where an odour-A PN type was tested: [seed, WB, LH, KC, MBON].")
ranks = []
for _, x in oa.iterrows():
    g = T[T.seed == x.seed]
    ranks.append([int(x.seed), x.unit] + [int((g[f"f_{r}"] > x[f"f_{r}"]).sum() + 1) for r in ODOUR])
put("a.odourA_PN_type_rank_in_fly", ranks, "Rank of each tested odour-A PN type among the 450 tested types of its fly "
    "[seed, type, rank WB, LH, KC, MBON] (1 = largest restored fraction).")
mc = pd.read_csv(ROOT / "results/calibration/matched_type_input_synapses.csv")
oat = mc[mc.cell_type.str.match(pat)]
full_list = uA.groupby("type").size()
assert sorted(oat.cell_type) == sorted(full_list.index) and all(full_list[t] == n for t, n in oat[["cell_type", "size_mc"]].values)
put("a.odourA_PN_types_in_MaleCNS", [[t, int(n)] for t, n in full_list.items()],
    "All odour-A driven PN types and their neuron counts in MaleCNS: every neuron of the full MaleCNS neuron table "
    "(results/validation/baseline_nostim.parquet, 166,700 neurons) whose type matches ^(DM1|DM4|VA2)_. All are class ALPN: "
    "results/calibration/ff_al_test.json n_PN_A, computed with the model's class-ALPN filter, equals the same count. "
    "Identical to the FW-MC matched types of results/calibration/matched_type_input_synapses.csv (size_mc).")
put("a.odourA_PN_mean_rank", rng_(oa.mean_rank), "Range of the hub vulnerability rank (mean over lost neurons) of the tested odour-A PN types.")
alpn_t = T.merge(UT[["seed", "unit", "class"]], on=["seed", "unit"])
alpn_t = alpn_t[alpn_t["class"] == "ALPN"]
alnz = alpn_t[[f"f_{r}" for r in ODOUR]].fillna(0).ne(0).any(axis=1)
put("a.ALPN_type_tests", [int(len(alpn_t)), int(alnz.sum()), int(alpn_t[alnz].unit.str.match(pat).sum())],
    "ALPN single-type tests in the screen [n tests, n with any non-zero odour readout, n of those that are odour-A PN types].")
put("a.ALPN_types_tested", [int(alpn_t.unit.nunique()), int((alpn_t.groupby("unit").seed.nunique() >= 3).sum())],
    "Distinct ALPN types tested [n types, n tested in >=3 flies].")
oan = NEU[NEU.unit.str.match(pat)]
put("a.odourA_PN_neuron_tests", [[int(x.seed), x.unit] + [rnd(x[f"f_{r}"], 3) for r in ODOUR] for _, x in oan.iterrows()],
    "Single-neuron rescues of odour-A PNs: [seed, neuron, restored fraction WB, LH, KC, MBON].")
def rescue_row(x, r):
    s, u = int(x.seed), x.unit
    yd, yh = raw_b.loc[s, (r, "damaged")], raw_b.loc[s, (r, "healthy")]
    return [s, u, rnd(x[f"f_{r}"], 3), rnd(yd, 3), rnd(yh, 3), rnd(x[r], 3), "excess" if yd > yh else "deficit"]


RESC = ("[seed, type, restored fraction, damaged, healthy, with the type restored (raw readout), 'deficit' if damaged < "
        "healthy or 'excess' if damaged > healthy (negative denominator)]")
for r in ODOUR:
    c = f"f_{r}"
    Tv = T.dropna(subset=[c]).sort_values(c, ascending=False)
    ex1266 = [x for x in Tv.head(4).itertuples() if int(x.seed) in EXCESS[r]]
    warn = (" Fractions > 1 are overshoots past healthy, not rescues of more than the deficit. In 'excess' flies the "
            "denominator is negative and small, so the fraction is not a rescue size" +
            ("".join(f": fly {int(x.seed)} {x.unit} moved {R[r]} from {raw_b.loc[int(x.seed), (r, 'damaged')]:.3f} to "
                     f"{getattr(x, r):.3f} Hz against healthy {raw_b.loc[int(x.seed), (r, 'healthy')]:.3f} Hz "
                     f"({100 * getattr(x, r) / raw_b.loc[int(x.seed), (r, 'healthy')]:.0f}% of healthy), fraction "
                     f"{getattr(x, c):.2f}" for x in ex1266[:1])) +
            f". Do not quote these as % of the deficit restored; the largest among true-deficit flies is in "
            f"a.top4_type_rescues_true_deficit.{R[r]}.")
    put(f"a.largest_type_rescue.{R[r]}", rescue_row(Tv.iloc[0], r),
        f"Largest single-type restored fraction of {RLONG[R[r]]} in the screen, all kept fly-readout pairs: " + RESC + "." + warn)
    put(f"a.top4_type_rescues.{R[r]}", [rescue_row(x, r) for _, x in Tv.head(4).iterrows()],
        f"Four largest single-type restored fractions of {RLONG[R[r]]} in the screen, all kept fly-readout pairs: "
        f"list of " + RESC + "." + warn)
    Tt = Tv[~Tv.seed.isin(EXCESS[r])]
    put(f"a.top4_type_rescues_true_deficit.{R[r]}", [rescue_row(x, r) for _, x in Tt.head(4).iterrows()],
        f"Four largest single-type restored fractions of {RLONG[R[r]]} among flies whose damaged value is below healthy "
        f"(true deficit; positive denominator): list of " + RESC + ". Values > 1 are overshoots past healthy.")

# ================================================================== (b) coverage of the type screen
tsum = UT.groupby("seed").agg(nl=("n_lost", "sum"), sl=("syn_lost", "sum"))
ins = UT[UT.chosen == True].groupby("seed").agg(nl=("n_lost", "sum"), sl=("syn_lost", "sum"))  # noqa: E712
csum = UC.groupby("seed").agg(nl=("n_lost", "sum"), sl=("syn_lost", "sum"))
cov_n, cov_s = tsum.nl / meta.n_removed, tsum.sl / csum.sl
put("b.tested_types_per_fly", int(UT.groupby("seed").size().median()), "Types tested per fly (400 inside chosen classes + 50 controls).")
put("b.coverage_neurons_450", [rnd(cov_n.median(), 3)] + rng_(cov_n) + [rnd(cov_n.quantile(0.25), 3), rnd(cov_n.quantile(0.75), 3)],
    "Share of each fly's removed neurons that lie in the 450 tested types: [median, min, max, q25, q75].")
put("b.coverage_synapses_450", [rnd(cov_s.median(), 3)] + rng_(cov_s),
    "Share of the removed neurons' synapses (in+out contacts) that lie in the 450 tested types: [median, min, max].")
cov_ni, cov_si = ins.nl / meta.n_removed, ins.sl / csum.sl
put("b.coverage_neurons_400_inside", [rnd(cov_ni.median(), 3)] + rng_(cov_ni),
    "Share of each fly's removed neurons that lie in the 400 inside (non-control) tested types: [median, min, max].")
put("b.coverage_synapses_400_inside", [rnd(cov_si.median(), 3)] + rng_(cov_si),
    "Synapse share in the 400 inside types only: [median, min, max].")
mins = UT[UT.chosen == True].groupby("seed").n_lost.min()  # noqa: E712
put("b.smallest_inside_type_nlost_counts", [[int(k), int(v)] for k, v in mins.value_counts().sort_index().items()],
    "Distribution over flies of the smallest number of removed neurons among the 400 inside types: [n_lost, n flies].")
put("b.flies_min_inside_nlost_ge2", int((mins >= 2).sum()),
    "Flies in which every inside (non-control) tested type had lost >= 2 neurons (single-loss types untested).")
ctrl1 = UT[(UT.control == True) & (UT.n_lost == 1)].groupby("seed").size()  # noqa: E712
put("b.controls_with_one_lost_neuron", [int(ctrl1.sum())] + rng_(ctrl1) + [rnd(ctrl1.median(), 1)],
    "Random control types with exactly 1 lost neuron [total tests, min per fly, max per fly, median per fly].")
for cls in HL + ["sc:visual_projection"]:
    t_ = UT[UT["class"] == cls].groupby("seed").n_lost.sum()
    c_ = UC[UC.unit == cls].set_index("seed").n_lost
    cv = (t_.reindex(c_.index).fillna(0) / c_)
    chosen = pd.Series({s: cls in meta.loc[s, "chosen_classes"] for s in c_.index})
    put(f"b.class_coverage.{cls}", [rnd(cv.median(), 3), rnd(cv[chosen].median(), 3), rnd(t_.sum() / c_.sum(), 3)],
        f"Share of removed {cls} neurons inside tested types: [median over flies with the class damaged, median over "
        f"flies where it was a chosen class, pooled over flies].")
pairs = [("MBON", "MBON_hz_A"), ("sc:descending_neuron", "GF_loom_hz"), ("ALPN", "LH_hz_A"),
         ("sc:cb_intrinsic", "LH_hz_A"), ("Kenyon_Cell", "KC_hz_A"), ("Kenyon_Cell", "MBON_hz_A"),
         ("ALPN", "KC_hz_A"), ("ALPN", "brain_spikes_A")]
TT = T.merge(UT[["seed", "unit", "class"]], on=["seed", "unit"])
for cls, r in pairs:
    c = f"f_{r}"
    cv = CL[CL.unit == cls].set_index("seed")[c].dropna()
    ts = TT[TT["class"] == cls].groupby("seed")[c].sum().reindex(cv.index).fillna(0)
    chosen = pd.Series({s: cls in meta.loc[s, "chosen_classes"] for s in cv.index})
    put(f"b.class_vs_sum_tested_types.{cls}.{R[r]}", [rnd(cv.median(), 3), rnd(ts.median(), 3),
                                                       rnd(cv[chosen].median(), 3), rnd(ts[chosen].median(), 3), int(len(cv))],
        f"Class {cls} single rescue vs summed single rescues of its tested types, {RLONG[R[r]]}: [class median, type-sum "
        f"median (all flies with deficit), class median, type-sum median (flies where the class was chosen), n flies].")
for t_ in ["DNp01", "APL", "MN9", "LPLC2", "LC4", "PVLP151", "GNG120"] + sorted(oat.cell_type):
    g = UT[UT.unit == t_]
    put(f"b.tested.{t_}", [int(g.seed.nunique()), [int(s) for s in g.seed][:6], rnd(g.n_lost.median(), 1),
                           int(g.chosen.sum()), int(g.control.sum())],
        f"Type {t_} in the screen: [n flies tested (of 48), first seeds, median neurons lost when tested, n as inside "
        f"type, n as random control].")
put("b.LPLC2_LC4_mean_rank", [rnd(UT[UT.unit == "LPLC2"].mean_rank.median(), 3), rnd(UT[UT.unit == "LC4"].mean_rank.median(), 3)],
    "Median hub rank of lost LPLC2 and LC4 neurons when tested (looming stimulus input types, which carry load).")
top_units = pd.read_csv(E1 / "top_units.csv")
put("b.carried_forward", [int(top_units[top_units.role == "top"].unit.nunique()), int((top_units.role == "control").sum()),
                          int(top_units.unit.nunique())],
    "Types carried to the robust/confirm phases [unique top-10-per-readout types, random controls, total].")
oa_rank = [t_ for t_ in sorted(oat.cell_type) if t_ in set(top_units.unit)]
put("b.odourA_PN_or_hubs_carried_forward", oa_rank + [t_ for t_ in ["DNp01", "APL"] if t_ in set(top_units.unit)],
    "Odour-A PN types, DNp01 or APL among the 54 carried-forward types (empty list = none).")
pm("b.top_types_per_fly_neuron_step", [len(x) for x in meta.top_types],
   "Types entering the single-neuron step per fly (union of the top 5 per readout)")
put("b.top_types_per_fly_range", rng_([len(x) for x in meta.top_types]), "Range of types entering the single-neuron step per fly.")
nn = runs[runs.level == "neuron"].groupby("seed").size()
put("b.neuron_runs", [int(nn.sum()), rnd(nn.sum() / len(runs), 3), rnd(nn.median(), 1)] + rng_(nn),
    "Single-neuron rescues [total, share of screen simulations, median per fly, min, max].")

# ================================================================== (c) escape-failing flies
gfd = raw_b[("GF_loom_hz", "damaged")]
fail = sorted(gfd[gfd < 100].index)
put("c.escape_fail_n", [int(len(fail)), int(sum(stage_of[s] == 0.3 for s in fail)), int(sum(stage_of[s] == 0.5 for s in fail))],
    "Screen flies with damaged giant fibre rate < 100 Hz (escape failure) [all, s=0.3, s=0.5].")
put("c.escape_fail_seeds", [int(s) for s in fail], "Seeds of the escape-failing screen flies.")
put("c.escape_fail_damaged_GF_hz", [rnd(gfd[s], 1) for s in fail], "Damaged GF rate (Hz) of each escape-failing fly, same order.")
dn = CL[CL.unit == "sc:descending_neuron"].set_index("seed").f_GF_loom_hz
dnp = T[T.unit == "DNp01"].set_index("seed").f_GF_loom_hz
cum10 = CU[(CU.r == "GF_loom_hz") & (CU.k == 10)].set_index("seed").frac
best_gf = T.dropna(subset=["f_GF_loom_hz"]).groupby("seed").f_GF_loom_hz.max()
rows = []
for s in fail:
    g = CL[(CL.seed == s) & (CL.unit != "sc:descending_neuron")].f_GF_loom_hz
    rows.append([int(s), rnd(dn.get(s), 3), rnd(dead_f["GF_loom_hz"].get(s), 3), rnd(best_gf.get(s), 3),
                 rnd(cum10.get(s), 3), rnd(dnp.get(s), 3), rnd(g.max(), 3)])
put("c.escape_fail_table", rows, "Per escape-failing fly: [seed, DN class rescue, restore_dead_all, best tested type, top-10 "
    "types together, DNp01 type rescue (null = not tested), best other class] as fractions of the GF deficit.")
untested = [s for s in fail if s not in dnp.index]
put("c.DNp01_tested_seeds", [int(s) for s in dnp.index], "Seeds where DNp01 (giant fibre) was a tested type.")
put("c.DNp01_type_rescue", [rnd(dnp[s], 3) for s in dnp.index], "DNp01 type rescue (both giant fibres restored) of the GF deficit, same order.")
put("c.DNp01_damaged_GF_hz", [rnd(gfd[s], 1) for s in dnp.index], "Damaged GF rate in the DNp01-tested flies (Hz).")
dnn = NEU[NEU.unit.str.startswith("DNp01#")]
put("c.DNp01_single_neuron", [[int(s), u, rnd(v, 3)] for s, u, v in dnn[["seed", "unit", "f_GF_loom_hz"]].values],
    "Single giant fibre neuron rescues [seed, neuron, fraction of the GF deficit].")
put("c.untested_fail.DN_class", [rnd(dn.reindex(untested).median(), 3)] + rng_(dn.reindex(untested)),
    "DN class (sc:descending_neuron) rescue in the escape-failing flies where DNp01 was not tested [median, min, max].")
put("c.untested_fail.all_removed", rng_(dead_f["GF_loom_hz"].reindex(untested)),
    "restore_dead_all in the escape-failing flies where DNp01 was not tested, fraction of the GF deficit [min, max].")
put("c.untested_fail.best_type_max", rnd(best_gf.reindex(untested).max(), 3), "Largest best-tested-type rescue of the GF deficit in the escape-failing flies where DNp01 was not tested.")
put("c.untested_fail.top10", [rnd(cum10.reindex(untested).median(), 3)] + rng_(cum10.reindex(untested)),
    "Top-10 types restored together, fraction of the GF deficit, in the escape-failing flies where DNp01 was not tested [median, min, max].")
nxt = [CL[(CL.seed == s) & (CL.unit != "sc:descending_neuron")].f_GF_loom_hz.max() for s in untested]
put("c.untested_fail.next_best_class_median", rnd(np.median(nxt), 3), "Median over the escape-failing flies where DNp01 was not tested of the best non-DN class rescue of the GF deficit.")
put("c.DNp01_flies.DN_class", [rnd(dn.get(s), 3) for s in dnp.index], "DN class rescue in the two DNp01-tested flies.")
nonfail = [s for s in cum10.index if s not in fail]
pm("c.top10_GF_pooled", cum10, "Top-10 types together, fraction of the GF deficit, all flies with a GF deficit")
pm("c.top10_GF_nonfailing", cum10.reindex(nonfail), "Top-10 types together, fraction of the GF deficit, flies with a GF deficit that did not fail escape")
put("c.DN_class_GF_all", mci(dn), "DN class single rescue of the GF deficit, all flies with a GF deficit " + MCI + ".")
argGF = T.dropna(subset=["f_GF_loom_hz"]).loc[T.dropna(subset=["f_GF_loom_hz"]).groupby("seed").f_GF_loom_hz.idxmax()]
UNTYPED = ("A unit named 'untyped:<class>' is not a cell type: it pools all of that fly's removed neurons that have no "
           "cell-type label in that class ('sc:<superclass>' when the class is empty too; counterfactual.unit_labels), "
           "restored together.")
put("c.best_GF_type_identity", [[k, int(v)] for k, v in argGF.unit.value_counts().head(5).items()],
    "Best tested type per fly for GF [type, n flies] (top 5). " + UNTYPED)
inside_min_fail = UT[(UT.chosen == True) & UT.seed.isin(fail)].groupby("seed").n_lost.min()  # noqa: E712
put("c.fail_min_inside_nlost", rng_(inside_min_fail), "Smallest n_lost among inside tested types in escape-failing flies [min, max].")
onl = json.loads((ROOT / "results/paper/output_neuron_loss.json").read_text())
uL, uR = onl["vulnerability_percentile"]["DNp01_L"], onl["vulnerability_percentile"]["DNp01_R"]
exp_gf = sum(1 - (1 - pp["death_max"] * stage_of[s] * uL) * (1 - pp["death_max"] * stage_of[s] * uR) for s in seeds)
put("c.expected_flies_losing_a_GF", rnd(exp_gf, 1),
    "Expected number of the 48 screen flies losing at least one giant fibre, from DNp01 hub ranks "
    "(output_neuron_loss.json) and P(removed) = 0.5 x s x rank.")
put("c.series_gf_fail_by_left_removed", [[k, int(v)] for k, v in onl["gf_fail_by_left_removed"].items()],
    "Provenance (output_neuron_loss.json, 320-fly series): ['left GF removed|escape failed', n flies] counts.")

# ================================================================== (d) high-activity flies
scp = sc[(sc.scope == "pan") & (sc.stage > 0)]
hi = []
ratios = {}
for s_, g in scp.groupby("stage"):
    k = g.KC_hz_A > 1.5 * g.KC_hz_A.median()
    mm = g.MBON_hz_A > 1.5 * g.MBON_hz_A.median()
    for _, x in g.iterrows():
        ratios[int(x.seed)] = (x.KC_hz_A / g.KC_hz_A.median(), x.MBON_hz_A / g.MBON_hz_A.median())
    hi += [int(x) for x in g[k & mm].seed]
hs = [s for s in hi if s in seeds]
put("d.high_activity_screen_flies", hs,
    "Screen flies meeting the paper's criterion (KC and MBON rate both > 1.5x their stage-group median in "
    "stage_curve_x4.csv, scope pan).")
put("d.high_activity_s0.3_s0.5_all", [s for s in hi if sc.set_index("seed").stage.get(s) in (0.3, 0.5)],
    "All severity-series flies at s = 0.3 or 0.5 meeting the criterion (screen covers only the first 24 seeds per stage).")
cbi = CL[CL.unit == "sc:cb_intrinsic"].set_index("seed")
cbi_raw = runs[(runs.level == "class") & (runs.unit == "sc:cb_intrinsic")].set_index("seed")
rows = []
for s in hs:
    rows.append([s, rnd(cbi_raw.loc[s, "KC_hz_A"] - raw_b.loc[s, ("KC_hz_A", "damaged")], 3),
                 rnd(cbi_raw.loc[s, "MBON_hz_A"] - raw_b.loc[s, ("MBON_hz_A", "damaged")], 3), int(cbi_raw.loc[s, "n_restored"])])
put("d.cb_intrinsic_effect_high_flies", rows,
    "Restoring the sc:cb_intrinsic class (contains APL) in the high-activity screen flies: [seed, change in KC rate (Hz), "
    "change in MBON rate (Hz), neurons restored].")
put("d.cb_intrinsic_raw_high_flies",
    [[s] + [rnd(raw_b.loc[s, (r, u)], 3) for r in ("KC_hz_A", "MBON_hz_A") for u in ("damaged", "healthy")]
     + [rnd(cbi_raw.loc[s, "KC_hz_A"], 3), rnd(cbi_raw.loc[s, "MBON_hz_A"], 3)] for s in hs],
    "High-activity screen flies, raw rates in Hz: [seed, KC damaged, KC healthy, MBON damaged, MBON healthy, KC with "
    "sc:cb_intrinsic restored, MBON with sc:cb_intrinsic restored].")
oth = runs[(runs.level == "class") & runs.seed.isin(hs) & (runs.unit != "sc:cb_intrinsic")].copy()
oth["dKC"] = oth.KC_hz_A.values - raw_b.loc[oth.seed, ("KC_hz_A", "damaged")].values
oth["dMB"] = oth.MBON_hz_A.values - raw_b.loc[oth.seed, ("MBON_hz_A", "damaged")].values
put("d.other_classes_high_flies", [rnd(oth.dKC.min(), 4), rnd(oth.dMB.min(), 4), rnd(oth.dKC.max(), 3), rnd(oth.dMB.max(), 3)],
    "All other classes in the five high-activity screen flies: [most negative KC change, most negative MBON change, largest KC rise, "
    "largest MBON rise] (Hz).")
allc = runs[(runs.level == "class") & (runs.unit == "sc:cb_intrinsic")].copy()
allc["dKC"] = allc.KC_hz_A.values - raw_b.loc[allc.seed, ("KC_hz_A", "damaged")].values
allc["dMB"] = allc.MBON_hz_A.values - raw_b.loc[allc.seed, ("MBON_hz_A", "damaged")].values
grad = allc[(~allc.seed.isin(hs)) & (allc.dKC < -0.1)].sort_values("dKC")
put("d.cb_intrinsic_effect_other_flies", [[int(s), rnd(a, 3), rnd(b, 3), rnd(ratios[int(s)][0], 2), rnd(ratios[int(s)][1], 2)]
                                          for s, a, b in grad[["seed", "dKC", "dMB"]].values],
    "Non-criterion screen flies where restoring sc:cb_intrinsic lowered KC by > 0.1 Hz: [seed, dKC, dMBON (Hz), KC and "
    "MBON ratio to stage median in stage_curve_x4].")
apl_seed = UT[UT.unit == "APL"].seed.unique().tolist()
put("d.APL_tested_seeds", [int(s) for s in apl_seed], "Screen flies where the APL type was tested (APL lost and among the 400).")
for s in apl_seed:
    s = int(s)
    rr_ = runs[runs.seed == s].set_index("unit")
    gl = lambda u, r: rnd(rr_.loc[u, r], 3)  # noqa: E731
    put(f"d.APL_fly{s}.KC_hz", [gl("damaged", "KC_hz_A"), gl("healthy", "KC_hz_A"), gl("APL", "KC_hz_A"),
                                gl("sc:cb_intrinsic", "KC_hz_A")],
        f"Fly {s}: KC rate (Hz) [damaged, healthy, APL type restored, sc:cb_intrinsic class restored].")
    put(f"d.APL_fly{s}.MBON_hz", [gl("damaged", "MBON_hz_A"), gl("healthy", "MBON_hz_A"), gl("APL", "MBON_hz_A"),
                                  gl("sc:cb_intrinsic", "MBON_hz_A")],
        f"Fly {s}: MBON rate (Hz) [damaged, healthy, APL type restored, sc:cb_intrinsic class restored].")
    put(f"d.APL_fly{s}.WB_spikes", [int(rr_.loc["damaged", "brain_spikes_A"]), int(rr_.loc["healthy", "brain_spikes_A"]),
                                    int(rr_.loc["APL", "brain_spikes_A"])],
        f"Fly {s}: whole-brain spikes during odour A [damaged, healthy, APL restored].")
    an = rr_[rr_.index.str.startswith("APL#")]
    put(f"d.APL_fly{s}.single_neurons", [[u, rnd(a, 3), rnd(b, 3)] for u, a, b in zip(an.index, an.KC_hz_A, an.MBON_hz_A)],
        f"Fly {s}: single APL neuron restored [neuron, KC Hz, MBON Hz].")
    put(f"d.APL_fly{s}.n_lost", int(UT[(UT.seed == s) & (UT.unit == "APL")].n_lost.iloc[0]), f"APL neurons lost in fly {s}.")
    fa = F[(F.seed == s) & (F.unit == "APL")].iloc[0]
    put(f"d.APL_fly{s}.fractions", [rnd(fa.f_brain_spikes_A, 3), rnd(fa.f_KC_hz_A, 3), rnd(fa.f_MBON_hz_A, 3)],
        f"Fly {s}: APL type restored fractions [WB, KC, MBON] on the paper's deficit metric (MBON denominator is "
        f"negative because damaged MBON exceeded healthy).")
    others = runs[(runs.seed == s) & (runs.level == "type") & (runs.unit != "APL")]
    put(f"d.APL_fly{s}.next_largest_MBON_drop", rnd((others.MBON_hz_A - rr_.loc["damaged", "MBON_hz_A"]).min(), 3),
        f"Fly {s}: most negative MBON change (Hz) from any other tested type.")
    put(f"d.APL_fly{s}.stage_curve_ratio", [rnd(ratios[s][0], 2), rnd(ratios[s][1], 2)],
        f"Fly {s}: KC and MBON rate / stage-group median in stage_curve_x4.csv (the paper's criterion table).")
dm5 = raw_b[[stage_of[s] == 0.5 for s in raw_b.index]]
put("d.screen_s0.5_damaged_median_hz", [rnd(dm5[("KC_hz_A", "damaged")].median(), 3), rnd(dm5[("MBON_hz_A", "damaged")].median(), 3)],
    "Median damaged KC and MBON rate (Hz) over the 24 s = 0.5 screen flies.")
g5 = scp[scp.stage == 0.5]
put("d.stage_curve_s0.5_median_hz", [rnd(g5.KC_hz_A.median(), 3), rnd(g5.MBON_hz_A.median(), 3)],
    "Median KC and MBON rate (Hz) of the 32 s = 0.5 flies in stage_curve_x4.csv.")
put("d.high_flies_min_inside_nlost", [[s, int(UT[(UT.seed == s) & (UT.chosen == True)].n_lost.min())] for s in hs],  # noqa: E712
    "Smallest n_lost among inside tested types in each high-activity screen fly [seed, n_lost] (a single lost APL "
    "could not be tested where this is 2 or more).")
dmg_gt = []
for r in SEL:
    d_ = raw_b[(r, "healthy")] - raw_b[(r, "damaged")]
    ok = (d_ < 0) & (d_.abs() > MIN_DEF * raw_b[(r, "healthy")].abs())
    dmg_gt.append([R[r], [int(s) for s in d_[ok].index]])
put("d.readouts_damaged_above_healthy", dmg_gt,
    "Fly-readout pairs kept with damaged > healthy (negative denominator) in the screen: [readout, seeds].")
put("d.fly1266_MBON_extreme_fractions", [rnd(F[(F.seed == 1266) & (F.unit == u)].f_MBON_hz_A.iloc[0], 2)
                                         for u in ["APL", "ALPN", "restore_dead_all", "restore_syn_all"]],
    "Fly 1266 MBON restored fractions [APL type, ALPN class, restore_dead_all, restore_syn_all] (small negative "
    "denominator: damaged 2.48 Hz vs healthy 2.33 Hz).")

# ================================================================== (e) additivity and concentration
for r in SEL:
    c = f"f_{r}"
    tr = runs[runs.level == "type"].copy()
    tr["same"] = tr[r].values == raw_b.loc[tr.seed, (r, "damaged")].values
    okf = dead_f[r].dropna().index
    tro = tr[tr.seed.isin(okf)]
    put(f"e.single_type_bit_identical.{R[r]}", [rnd(tro.same.mean(), 4), rnd(tr.same.mean(), 4)],
        f"Share of single-type rescues whose {RLONG[R[r]]} equals the damaged value exactly [flies with a deficit, all 48].")
    chg = (~tro.same).groupby(tro.seed).sum()
    pos = T[T.seed.isin(okf)].groupby("seed")[c].apply(lambda s: int((s > 0).sum()))
    pm(f"e.types_changing_readout_per_fly.{R[r]}", chg, f"Tested types (of 450) per fly that changed {RLONG[R[r]]} at all")
    pm(f"e.types_positive_per_fly.{R[r]}", pos, f"Tested types per fly with a positive restored fraction of {RLONG[R[r]]}")
    tv = T[c].dropna()
    put(f"e.single_type_share_above.{R[r]}", [rnd((tv > 0.01).mean(), 4), rnd((tv > 0.05).mean(), 4), rnd((tv > 0.10).mean(), 4)],
        f"Share of single-type rescues restoring >1%, >5%, >10% of the {RLONG[R[r]]} deficit.")
    if R[r] == "WB":
        for s_ in (0.3, 0.5):
            tvs = T[T.stage == s_][c].dropna()
            put(f"e.single_type_share_above_0.05.WB.s{s_}", rnd((tvs > 0.05).mean(), 4),
                f"Share of single-type rescues restoring >5% of the WB deficit at s = {s_}.")
    put(f"e.median_single_type_fraction.{R[r]}", rnd(tv.median(), 4), f"Median single-type restored fraction of {RLONG[R[r]]}.")
    # cumulative
    cr = CU[CU.r == r].pivot_table(index="seed", columns="k", values="frac")
    for k in (1, 3, 10, 30):
        pm(f"e.cumulative_top{k}.{R[r]}", cr[k], f"Top {k} types restored together, fraction of {RLONG[R[r]]} deficit")
        dd = dead_f[r].reindex(cr.index)
        pm(f"e.cumulative_top{k}_share_of_all_removed.{R[r]}", (cr[k] / dd)[dd > 0.1],
           f"Top {k} together / restore_dead_all per fly (restore_dead_all > 0.1), {RLONG[R[r]]}")
    for s_ in (0.3, 0.5):
        put(f"e.cumulative_top10.{R[r]}.s{s_}", rnd(cr[10][[stage_of[s] == s_ for s in cr.index]].median(), 3),
            f"Median top-10 together at s = {s_}, {RLONG[R[r]]}.")
    s10 = T.dropna(subset=[c]).groupby("seed")[c].apply(lambda s: s.nlargest(10).sum())
    ratio = (cr[10] / s10.reindex(cr.index))[s10.reindex(cr.index) > 0]
    pm(f"e.top10_over_sum_of_singles.{R[r]}", ratio,
       f"Cumulative top-10 fraction / sum of the same ten single-type fractions per fly (flies with sum > 0), {RLONG[R[r]]}")
    put(f"e.top3_below_top1.{R[r]}", [int((cr[3] < cr[1]).sum()), int(cr[1].notna().sum())],
        f"Flies where top-3 together restored less than the best single type [n, of n flies], {RLONG[R[r]]}.")
    put(f"e.top10_below_top1.{R[r]}", [int((cr[10] < cr[1]).sum()), int(cr[1].notna().sum())],
        f"Flies where top-10 together restored less than the best single type [n, of n], {RLONG[R[r]]}.")
    cw = CU[CU.r == r].pivot_table(index="seed", columns="k", values=r)
    put(f"e.top30_equals_top10_raw.{R[r]}", [int((cw[30] == cw[10]).sum()), int(cw[10].notna().sum())],
        f"Flies where the raw {R[r]} readout with the top 30 restored equals that with the top 10 [n, of n].")
    s450 = T.groupby("seed")[c].sum(min_count=1)
    put(f"e.sum_all_single_types.{R[r]}", [rnd(s450.median(), 3), rnd(dead_f[r].median(), 3)],
        f"[Median per-fly sum of all 450 single-type fractions, median restore_dead_all] for {RLONG[R[r]]}.")
    bn = NEU.dropna(subset=[c]).groupby("seed")[c].max()
    pm(f"e.best_single_neuron_per_fly.{R[r]}", bn, f"Best single-neuron rescue per fly, {RLONG[R[r]]}")
argMN9 = T.dropna(subset=["f_MN9_sugar_hz"]).loc[T.dropna(subset=["f_MN9_sugar_hz"]).groupby("seed").f_MN9_sugar_hz.idxmax()]
put("e.best_MN9_type_identity", [[k, int(v)] for k, v in argMN9.unit.value_counts().head(4).items()],
    "Best tested unit per fly for MN9 [unit, n flies] (top 4). " + UNTYPED + " 'untyped:sc:cb_intrinsic' is the pool of all the "
    "fly's removed central-brain intrinsic neurons without a type or class label, a large heterogeneous group "
    "(size in e.best_MN9_untyped_pool_n_lost), not one cell type.")
upool = argMN9[argMN9.unit.str.startswith("untyped:")].merge(UT[["seed", "unit", "n_lost"]], on=["seed", "unit"])
put("e.best_MN9_untyped_pool_n_lost", [[u, int(s), int(n)] for u, s, n in upool[["unit", "seed", "n_lost"]].values],
    "Flies where an untyped pool was the best tested unit for MN9: [unit, seed, neurons restored together].")
k1201 = F[(F.seed == 1201) & (F.unit.isin(["KCab-m"]))].f_brain_spikes_A
put("e.fly1201_KCabm_type_WB", rnd(k1201.iloc[0], 3) if len(k1201) else None,
    "Fly 1201: KCab-m type (all its removed neurons) restored fraction of the WB deficit.")
k1201n = NEU[(NEU.seed == 1201) & NEU.unit.str.startswith("KCab-m#")].f_brain_spikes_A
put("e.fly1201_KCabm_best_neuron_WB", rnd(k1201n.max(), 3), "Fly 1201: best single KCab-m neuron, fraction of the WB deficit.")

# ================================================================== (g) robustness
Rb = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(E1 / "robust/*.csv")))], ignore_index=True)
FR = {m: ax.frac_table(g, SEL, MIN_DEF) for m, g in Rb.groupby("model")}
MED = pd.concat([FR[m][FR[m].level == "type"].groupby("unit")[f"f_{r}"].median().rename(f"{m}|{r}")
                 for m in FR for r in SEL], axis=1)
VAR_NOISE, VAR_PARAM = ["noise1", "noise2"], ["w0.8", "w1.2", "glu0.5", "da0.5"]
rng = np.random.default_rng(SEED)
cells = []
for r in SEL:
    bcol = f"base|{r}"
    put(f"g.nonzero_base_medians.{R[r]}", int((MED[bcol].dropna() != 0).sum()),
        f"Of the 54 carried-forward types, number with a non-zero median restored fraction of {RLONG[R[r]]} in the base model.")
    allz = MED[[f"{m}|{r}" for m in FR]].fillna(0).eq(0).all(axis=1)
    put(f"g.types_zero_under_all_7_models.{R[r]}", int(allz.sum()), f"Types with a zero median under all 7 models, {RLONG[R[r]]}.")
    for m in VAR_NOISE + VAR_PARAM:
        ok = MED[[f"{m}|{r}", bcol]].dropna()
        x, y = ok[f"{m}|{r}"], ok[bcol]
        rho0 = ax.rho(x, y)
        ov0 = len(set(y.nlargest(10).index) & set(x.nlargest(10).index)) / 10
        nzm = (x != 0) | (y != 0)
        rho_nz = ax.rho(x[nzm], y[nzm])
        pb = set(y.nlargest(10).index) & set(y[y > 0].index)
        pm = set(x.nlargest(10).index) & set(x[x > 0].index)
        ov_pos = len(pb & pm) / len(pb) if pb else np.nan
        ovr = []
        for _ in range(1000):
            jx = x + rng.random(len(x)) * 1e-12
            jy = y + rng.random(len(y)) * 1e-12
            ovr.append(len(set(jy.nlargest(10).index) & set(jx.nlargest(10).index)) / 10)
        cells.append({"r": R[r], "m": m, "group": "noise" if m in VAR_NOISE else "param", "rho": rho0, "ov": ov0,
                      "rho_nz": rho_nz, "n_nz": int(nzm.sum()), "ov_pos": ov_pos, "ov_rand": float(np.mean(ovr)),
                      "pass": (rho0 >= 0.7) and (ov0 >= 0.6)})
CE = pd.DataFrame(cells)
rbs = pd.read_csv(E1 / "summary/robust_summary.csv")
chk = CE.merge(rbs.assign(r=rbs.readout.map(R)), left_on=["r", "m"], right_on=["r", "model"])
put("g.check_reproduces_robust_summary", bool(np.allclose(chk.rho, chk.spearman_vs_base, equal_nan=True)
                                              and np.allclose(chk.ov, chk.top10_overlap)),
    "Check: recomputed rho and top-10 overlap equal results/exp1/summary/robust_summary.csv.")
put("g.prespecified_pass", [int(CE["pass"].sum()), int(len(CE)), int((CE.ov >= 0.6).sum()), int((CE.rho >= 0.7).sum())],
    "Prespecified criterion (rho >= 0.7 and top-10 overlap >= 0.6 vs base): [cells passing, cells, cells with overlap "
    ">= 0.6, cells with rho >= 0.7].")
put("g.prespecified_pass_by_group", [int(CE[CE.group == "noise"]["pass"].sum()), int((CE.group == "noise").sum()),
                                     int(CE[CE.group == "param"]["pass"].sum()), int((CE.group == "param").sum())],
    "Passing cells [input-noise variants passing, of cells; parameter variants passing, of cells].")
put("g.prespecified_pass_by_variant", [[m, int(CE[CE.m == m]["pass"].sum())] for m in VAR_NOISE + VAR_PARAM],
    "Passing readouts (of 6) per variant.")
put("g.degenerate_passes", int(CE[CE["pass"] & (CE.n_nz <= 5)].shape[0]),
    "Passing cells that rest on <= 5 types with a non-zero median in either model (tied-zero artefacts; MN9 and GF).")
put("g.degenerate_passes_cells", [[a, b, int(c)] for a, b, c in CE[CE["pass"] & (CE.n_nz <= 5)][["r", "m", "n_nz"]].values],
    "The degenerate passing cells [readout, variant, n non-zero types].")
for r in SEL:
    g = CE[CE.r == R[r]]
    put(f"g.rho_nonzero.{R[r]}", [[m, rnd(v, 3), int(n)] for m, v, n in g[["m", "rho_nz", "n_nz"]].values],
        f"Spearman rho vs base restricted to types with a non-zero median in either model, {RLONG[R[r]]}: [variant, rho, "
        f"n types] (post hoc).")
    for grp in ("noise", "param"):
        gg = g[g.group == grp]
        put(f"g.rho_nonzero_range.{R[r]}.{grp}", rng_(gg.rho_nz),
            f"Range of non-zero-restricted rho over the {'input-noise' if grp == 'noise' else 'parameter'} variants, {RLONG[R[r]]}.")
        put(f"g.rho_prespecified_range.{R[r]}.{grp}", rng_(gg.rho),
            f"Range of prespecified rho over the {'input-noise' if grp == 'noise' else 'parameter'} variants, {RLONG[R[r]]}.")
    put(f"g.top10_overlap.{R[r]}", [[m, rnd(a, 2), rnd(b, 2), rnd(c_, 2)] for m, a, b, c_ in g[["m", "ov", "ov_pos", "ov_rand"]].values],
        f"Top-10 overlap with base, {RLONG[R[r]]}: [variant, prespecified (ties broken in fixed type order), restricted to "
        f"positive-median types, mean under random tie-breaking (1,000 draws)].")
put("g.cells_overlap_ge_0.6_random_ties", int((CE.ov_rand >= 0.6).sum()),
    "Cells (of 36) whose mean top-10 overlap under random tie-breaking reaches 0.6.")
put("g.LH_prespecified", [[m, rnd(a, 2), rnd(b, 1)] for m, a, b in CE[CE.r == "LH"][["m", "rho", "ov"]].values],
    "Lateral horn under the prespecified metrics [variant, rho, top-10 overlap].")
put("g.WB_prespecified", [[m, rnd(a, 2), rnd(b, 1)] for m, a, b in CE[CE.r == "WB"][["m", "rho", "ov"]].values],
    "Whole brain under the prespecified metrics [variant, rho, top-10 overlap].")
put("g.KC_noise1", [rnd(CE[(CE.r == "KC") & (CE.m == "noise1")].rho.iloc[0], 3), rnd(CE[(CE.r == "KC") & (CE.m == "noise1")].ov.iloc[0], 2)],
    "KC under input noise 1 alone [rho, top-10 overlap].")
put("g.MN9_nonzero_base_types", [u for u in MED.index if MED.loc[u, "base|MN9_sugar_hz"] not in (0,) and pd.notna(MED.loc[u, "base|MN9_sugar_hz"])],
    "Types with a non-zero base median for MN9 (all MN9 cells rest on these).")
put("g.GF_nonzero_base_types", [u for u in MED.index if MED.loc[u, "base|GF_loom_hz"] != 0 and pd.notna(MED.loc[u, "base|GF_loom_hz"])],
    "Types with a non-zero base median for GF.")

# ================================================================== (h) input-noise variability
Cf = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(E1 / "confirm/*.csv")))], ignore_index=True)
FC = ax.frac_table(Cf, SEL + ["LI"], MIN_DEF)
FCt = FC[FC.level == "type"]


def per_fly(m, unit, r):
    tab = FCt if m == "full" else FR[m][FR[m].level == "type"]
    return tab[tab.unit == unit].set_index("seed")[f"f_{r}"]


REAL = ["base", "noise1", "noise2", "full"]
for unit, r, flies in [("LPLC2", "GF_loom_hz", [1258]), ("GNG120", "MN9_sugar_hz", None),
                       ("PVLP151", "GF_loom_hz", None), ("LC4", "GF_loom_hz", None)]:
    pf = {m: per_fly(m, unit, r) for m in REAL}
    put(f"h.type_median.{unit}.{R[r]}", [rnd(pf[m].median(), 3) for m in REAL],
        f"{unit} median restored fraction of the {RLONG[R[r]]} deficit [base, noise1, noise2, full protocol] (s = 0.5 flies "
        f"where it was lost).")
    put(f"h.type_rank.{unit}.{R[r]}", [int((MED[f"{m}|{r}"].dropna() > MED.loc[unit, f"{m}|{r}"]).sum() + 1) for m in REAL[:3]]
        + [int((FCt.groupby("unit")[f"f_{r}"].median() > pf["full"].median()).sum() + 1)],
        f"Rank of {unit} among the 54 types by median, {R[r]} [base, noise1, noise2, full].")
    fl = flies or sorted(set(pf["base"].index) | set(pf["full"].index))
    put(f"h.per_fly.{unit}.{R[r]}", [[int(s)] + [rnd(pf[m].get(s), 3) for m in REAL] for s in fl],
        f"{unit} per-fly restored fraction of {RLONG[R[r]]} [seed, base, noise1, noise2, full].")
for r in SEL:
    a = FR["base"][FR["base"].level == "type"].set_index(["seed", "unit"])[f"f_{r}"]
    vals = []
    for m in ("noise1", "noise2"):
        b = FR[m][FR[m].level == "type"].set_index(["seed", "unit"])[f"f_{r}"]
        j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        nzj = j[(j.a != 0) | (j.b != 0)]
        vals += [rnd(ax.rho(j.a, j.b), 3), rnd(ax.rho(nzj.a, nzj.b), 3), int(len(nzj))]
    put(f"h.spearman_base_vs_noise_per_fly_type.{R[r]}", vals,
        f"Spearman of per-fly x type restored fractions of {RLONG[R[r]]}, base vs noise1 [all pairs, non-zero-in-either "
        f"pairs, n non-zero pairs] then base vs noise2 [same].")
mn9d = {m: (lambda bb: bb.loc["healthy", "MN9_sugar_hz"] - bb.loc["damaged", "MN9_sugar_hz"])(
    (Rb[(Rb.model == m) & (Rb.seed == 1261) & (Rb.level == "baseline")] if m != "full" else
     Cf[(Cf.seed == 1261) & (Cf.level == "baseline")]).set_index("unit")) for m in REAL}
put("h.MN9_deficit_fly1261", [rnd(mn9d[m], 1) for m in REAL], "Fly 1261 MN9 deficit (Hz) [base, noise1, noise2, full]: the denominator itself varies.")
r1198 = runs[runs.seed == 1198].set_index("unit")
d1198 = r1198.loc["healthy", "GF_loom_hz"] - r1198.loc["damaged", "GF_loom_hz"]
n1198 = NEU[NEU.seed == 1198].f_GF_loom_hz.dropna()
put("h.fly1198_GF_quantisation", [rnd(d1198, 2), rnd(100 * d1198 / r1198.loc["healthy", "GF_loom_hz"], 1), int(len(n1198)),
                                  int((n1198 >= 0.75 - 1e-9).sum()), int((n1198 >= 0.625 - 1e-9).sum())],
    "Fly 1198: [GF deficit Hz, deficit % of healthy, n single-neuron rescues, n restoring >= 75%, n restoring >= 62.5%] "
    "(GF rate moves in 1.67 Hz steps, so small deficits give coarse fractions).")

# ================================================================== (i) confirm phase
bse = Rb[Rb.model == "base"]
mg = Cf.merge(bse, on=["seed", "level", "unit"], suffixes=("_c", "_r"))
eq = {r: int(((mg[f"{r}_c"] == mg[f"{r}_r"]) | (mg[f"{r}_c"].isna() & mg[f"{r}_r"].isna())).sum()) for r in
      ["brain_spikes_A", "LH_hz_A", "PN_hz_A", "KC_hz_A", "KC_frac_A", "MBON_hz_A", "V_pre_A", "MN9_sugar_hz",
       "GF_loom_hz", "GF_latency_ms"]}
put("i.confirm_rows", [[k, int(v)] for k, v in Cf.level.value_counts().items()], "Confirm-phase rows by level (baseline, type = necessity, lesion = sufficiency).")
put("i.matched_rows_with_robust_base", int(len(mg)), "Confirm rows matched to the robust base-model run on (seed, level, unit).")
put("i.bit_identical_counts", [[k, v] for k, v in eq.items()],
    "Of the matched rows, number with an identical readout in confirm (full protocol) and robust base (short protocol); "
    "NaN in both (no GF spike, latency undefined) counts as identical.")
put("i.identical_readouts", [k for k, v in eq.items() if v == len(mg)],
    "Readouts identical in every matched row (odour A is the first segment of both protocols with the same input seed).")
put("i.new_readouts", [k for k, v in eq.items() if v < len(mg)] + ["LI"],
    "Readouts that differ from the short protocol (measured after odour B and a gap) plus the learning index, which only "
    "the full protocol has; the 483 lesion (sufficiency) rows are new for every readout.")
ms = Cf.merge(runs[runs.stage == 0.5], on=["seed", "level", "unit"], suffixes=("_c", "_s"))
put("i.matched_rows_with_screen", [int(len(ms)), int((ms.level == "type").sum()),
                                   int(all((ms[f"{r}_c"] == ms[f"{r}_s"]).all() for r in ODOUR))],
    "Confirm rows that also exist in the screen [n rows, n type rows, odour readouts all identical (1 = yes)].")
for r in ["MN9_sugar_hz", "GF_loom_hz"]:
    rhos = []
    for s, g in FCt.groupby("seed"):
        b = FR["base"][(FR["base"].level == "type") & (FR["base"].seed == s)].set_index("unit")[f"f_{r}"]
        j = pd.concat([g.set_index("unit")[f"f_{r}"], b], axis=1, keys=["c", "b"]).dropna()
        rv = ax.rho(j.c, j.b)
        if np.isfinite(rv):
            rhos.append(rv)
    put(f"i.per_fly_spearman_full_vs_short.{R[r]}", [rnd(np.median(rhos), 2), int(len(rhos))] + rng_(rhos),
        f"Per-fly Spearman of type restored fractions, full vs short (robust base) protocol, {RLONG[R[r]]}: [median, n "
        f"flies, min, max].")
    put(f"i.max_type_median_full_vs_short.{R[r]}", [rnd(FCt.groupby("unit")[f"f_{r}"].median().max(), 3),
                                                    rnd(MED[f"base|{r}"].max(), 3)],
        f"Largest type median restored fraction, {RLONG[R[r]]} [full protocol, short base].")
cs = pd.read_csv(E1 / "summary/confirm_summary.csv")
nec = cs[cs.test.str.startswith("necessity")].set_index("unit")
suf = cs[cs.test.str.startswith("sufficiency")].set_index("unit")
for r in SEL + ["LI"]:
    lab = R.get(r, "LI")
    put(f"i.necessity_max.{lab}", [nec[r].idxmax(), rnd(nec[r].max(), 3), int((nec[r] > 0.05).sum())],
        f"Full protocol, the fly's removed neurons of a type restored alone in the damaged fly: [type with largest median, "
        f"its median restored fraction of the {lab} deficit, n types > 0.05].")
    put(f"i.sufficiency_max.{lab}", [suf[r].idxmax(), rnd(suf[r].max(), 3), int((suf[r] > 0.05).sum()), int((suf[r] < -0.05).sum())],
        f"Full protocol, sufficiency lesion: only the neurons of type X that this fly lost (typ[u] from its dead set, "
        f"exp1_counterfactual.py confirm()) removed alone from its healthy copy (healthy weights, no threshold shift), not "
        f"the whole type (size in i.lesion_n_neurons): [type with largest median share of the {lab} deficit reproduced, "
        f"value, n types > +0.05, n types < -0.05].")
nf = nec.n_flies
put("i.flies_per_type", [int(nf.min()), rnd(nf.median(), 1), int(nf.max()), int((nf == 12).sum())],
    "Flies (of 12 at s = 0.5) in which each of the 54 types was tested = flies that lost it [min, median, max, n types "
    "tested in all 12].")
ctrl = top_units[top_units.role == "control"].unit
put("i.controls_LI_sufficiency_zero", [int((suf.loc[ctrl, "LI"] == 0).sum()), int(len(ctrl))],
    "Random control types with a zero median learning-index sufficiency [n, of n controls].")
les = Cf[Cf.level == "lesion"].copy()
hb = Cf[Cf.level == "baseline"].set_index(["seed", "unit"])
for col in ["MBON_hz_A", "LI"] + ODOUR + ["LH_hz_A", "PN_hz_A", "KC_frac_A", "V_pre_A", "MN9_sugar_hz", "GF_loom_hz"]:
    les[f"h_{col}"] = [hb.loc[(s, "healthy"), col] for s in les.seed]
mb_same = les.MBON_hz_A == les.h_MBON_hz_A
li_chg = les.LI != les.h_LI
pre = ["brain_spikes_A", "LH_hz_A", "PN_hz_A", "KC_hz_A", "KC_frac_A", "MBON_hz_A", "V_pre_A", "MN9_sugar_hz", "GF_loom_hz"]
anypre = pd.concat([les[c] != les[f"h_{c}"] for c in pre], axis=1).any(axis=1)
FL = FC[FC.level == "lesion"]
for u_ in ["CRE080_c", "DNge016", "GNG120", "DNge106"]:
    for r_ in (["LI"] if u_ in ("CRE080_c", "DNge016") else ["MN9_sugar_hz", "LI"]):
        v_ = (1 - FL[FL.unit == u_].set_index("seed")[f"f_{r_}"]).dropna()
        put(f"i.sufficiency_per_fly.{u_}.{R.get(r_, 'LI')}", [[int(s), rnd(x, 3)] for s, x in v_.items()],
            f"The fly's removed {u_} neurons (its own lost neurons of that type, not the whole type) removed alone from its "
            f"healthy copy (full protocol): per-fly share of the {R.get(r_, 'LI')} deficit reproduced, 1 - restored "
            f"fraction [seed, value] (flies with |healthy - damaged| > 5% of healthy only).")
    nl_ = les[les.unit == u_].n_lesioned
    put(f"i.lesion_n_neurons.{u_}", [rnd(nl_.median(), 1), int(nl_.min()), int(nl_.max()), int(len(nl_))],
        f"Neurons removed from the healthy copy in the {u_} sufficiency lesions = that fly's lost {u_} neurons "
        f"[median, min, max, n flies].")
nl_ = les.groupby("unit").n_lesioned.median()
put("i.lesion_n_neurons", [rnd(nl_.median(), 1), rnd(nl_.min(), 1), rnd(nl_.max(), 1), int(len(nl_))],
    "Per type, median neurons removed in its sufficiency lesions (the fly's own lost neurons of the type); then over the "
    "54 types [median, min, max, n types].")
put("i.lesion_LI_carryover", [int(mb_same.sum()), int((mb_same & li_chg).sum()), int(anypre.sum()), int((anypre & li_chg).sum()),
                              int((~anypre).sum()), int((~anypre & li_chg).sum())],
    "Sufficiency lesions: [n with MBON unchanged, of those LI changed; n with any pre-training readout changed, of those "
    "LI changed; n with no pre-training readout changed, of those LI changed] (no state reset between segments).")

# ================================================================== (j) wiring model
gv_old = pd.read_csv(E1 / "summary/graph_vs_counterfactual.csv")
gv, resid = ax.graph_vs(F, units)
put("j.check_graph_vs_reproduced", bool(np.allclose(gv.sort_values(["seed", "readout"]).r2_wiring_loo.values,
                                                    gv_old.sort_values(["seed", "readout"]).r2_wiring_loo.values)),
    "Check: recomputed per-fly LOO R^2 equals results/exp1/summary/graph_vs_counterfactual.csv.")
const = set()
for s in seeds:
    g = T[T.seed == s]
    for r in SEL:
        v = g[f"f_{r}"].dropna()
        if len(v) >= 10 and np.ptp(v) == 0:
            const.add((s, r))
gv["const"] = [(s, r) in const for s, r in zip(gv.seed, gv.readout)]
for r in SEL:
    g = gv[gv.readout == r]
    put(f"j.r2_loo.{R[r]}", [rnd(g.r2_wiring_loo.median(), 3), int(len(g)), rnd(g[~g.const].r2_wiring_loo.median(), 3),
                             int((~g.const).sum()), int(g.const.sum()), int((g[g.const].r2_wiring_loo == 1).sum())],
        f"Per-fly leave-one-out R^2 of the wiring model for {RLONG[R[r]]}: [median all flies, n, median excluding flies where "
        f"all tested types gave the same fraction, n, n constant flies, n of those with R^2 = 1 (artefact of "
        f"analyze_exp1.graph_vs: 1 - 0/1e-12)].")
    put(f"j.r2_loo_iqr_all.{R[r]}", miqr(g.r2_wiring_loo), f"LOO R^2 of the wiring model, all flies, {RLONG[R[r]]} " + MIQR + " (interquartile range, not a CI).")
    put(f"j.r2_loo_iqr_excl_const.{R[r]}", miqr(g[~g.const].r2_wiring_loo), f"LOO R^2 of the wiring model, {RLONG[R[r]]}, excluding flies where all tested types gave the same fraction " + MIQR + " (interquartile range, not a CI).")
excl = resid[[not ((s, r) in const) for s, r in zip(resid.seed, resid.readout)]]


def outliers(res):
    o = res.groupby(["readout", "unit"]).agg(n_flies=("z_resid", "size"), z_median=("z_resid", "median"),
                                             frac_median=("frac", "median")).reset_index()
    return o


O_all, O_ex = outliers(resid), outliers(excl)
do = pd.read_csv(E1 / "summary/dynamic_outliers.csv")
for r in SEL:
    for tag, O in [("paper_criterion", O_all), ("excl_constant_flies", O_ex)]:
        o = O[(O.readout == r) & (O.z_median >= 2) & (O.n_flies >= 10)].sort_values("z_median", ascending=False)
        put(f"j.outliers.{tag}.{R[r]}", [[u, int(n), rnd(z, 2), rnd(fm, 3)] for u, n, z, fm in o[["unit", "n_flies", "z_median", "frac_median"]].values],
            f"Wiring outliers for {RLONG[R[r]]} (median standardised LOO residual >= 2 in >= 10 flies), "
            f"{'all flies' if tag == 'paper_criterion' else 'excluding flies with constant fractions'}: "
            f"[type, n flies, median z, median restored fraction].")
    o = do[(do.readout == r) & (do.z_median >= 2)].sort_values("z_median", ascending=False).head(4)
    if R[r] == "WB":
        put("j.WB_outliers_without_10fly_floor", [[u, int(n), rnd(z, 2)] for u, n, z in o[["unit", "n_flies", "z_median"]].values],
            "WB top residual types without the >= 10-fly floor (dynamic_outliers.csv keeps >= 3 flies) [type, n flies, z].")
kab = do[(do.readout == "brain_spikes_A") & (do.unit == "KCa'b'-m")]
put("j.KCapbpm_WB_z", [rnd(kab.z_median.iloc[0], 3), int(kab.n_flies.iloc[0])] if len(kab) else None,
    "KCa'b'-m median standardised LOO residual for the whole-brain spike count (WB) [z, n flies]: below the outlier "
    "criterion (median z >= 2 in >= 10 flies).")

# ================================================================== write
# every meaning that speaks of a deficit or a restored fraction points to the either-sign definition
RAW = {"definitions.deficit", "k.MN9_deficit_hz", "k.MN9_deficit_hz.iqr", "k.GF_relative_deficit_counts", "h.MN9_deficit_fly1261",
       "design.code.min-deficit"}
SHORT2LONG = {v: k for k, v in R.items()}
for k in list(MEANING):
    if k in RAW or k.startswith("k.n_flies_") or not re.search(r"deficit|restored fraction", MEANING[k], re.I):
        continue
    comp = [c_ for c_ in k.split(".") if c_ in SHORT2LONG]
    rd = comp[-1] if comp else ("GF" if k.startswith("c.") else None)
    if rd and k[:2] in ("a.", "b.", "c.", "d.", "e.", "f.", "k."):        # screen keys: say how many flies it concerns
        n_ = len(EXCESS[SHORT2LONG[rd]])
        tail = (f"{n_} screen fl{'y has' if n_ == 1 else 'ies have'} {rd} damaged above healthy and "
                f"{'is' if n_ == 1 else 'are'} included with a negative denominator (k.n_flies_damaged_above_healthy.{rd})"
                if n_ else f"no screen fly has {rd} damaged above healthy")
    else:
        tail = "flies with damaged above healthy are included with a negative denominator"
    MEANING[k] += f" ['Deficit' = |healthy - damaged| > 5% of healthy, either sign; {tail}; see definitions.deficit.]"
for k in F_OUT:
    assert k in MEANING
out = {**F_OUT, "_meaning": MEANING}
(OUT / "facts_exp1.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=lambda o: o.item()
                                                if hasattr(o, "item") else str(o)), encoding="utf-8")
print(f"wrote {OUT / 'facts_exp1.json'}: {len(F_OUT)} facts")
