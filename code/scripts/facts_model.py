"""Facts table "model": connectome build, engine validation,
synaptic-weight calibration, persistence/adaptation calibration, feedforward
antennal lobe, odour A vs B, MaleCNS gustatory check, steering/looming probe,
pathology rules and vulnerability rank, circuit protocol and arena parameters.

  python code/scripts/facts_model.py

Most model facts are not computed here but read: values come from the JSON/CSV
result files written by the calibration/validation scripts, and parameters are
parsed from the code (dataclass defaults, function defaults and literal
expressions, via `ast` and anchored regular expressions that fail loudly if the
code changes). A few quantities are computed from the per-fly tables
(results/battery/*_x4.csv, results/exp1, results/joint) with the conventions of
the paper: medians as % of healthy use the healthy median of the same design,
intervals are 95% percentile bootstrap over flies within groups (4,000
resamples, numpy default_rng(SEED)), Mann-Whitney U two-sided, Spearman via
scipy; within-fly A-vs-B comparisons use the Wilcoxon signed-rank test.
One small simulation is run: the engine of code/adfly/lif.py on a one-neuron
pack without edges (temporary directory), to measure how many Poisson events
an isolated driven neuron turns into spikes with and without adaptation.

Writes results/facts/facts_model.json: a flat dict of facts, plus "_meaning"
(one sentence per key) and "_source" (file, and field or code line, per key).
No manuscript file and no existing script is modified.
"""
import ast
import glob
import json
import re
import sys
from pathlib import Path


import numpy as np
import pandas as pd
from scipy import stats

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())  # code/scripts locally, scripts/ on the server
RES = ROOT / "results"
CODE = ROOT / "code"
OUT = RES / "facts"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261009
NBOOT = 4000

FACTS, MEANING, SOURCE = {}, {}, {}


def put(key, value, meaning, source):
    if key in FACTS:
        raise KeyError(f"duplicate key {key}")
    FACTS[key] = value
    MEANING[key] = meaning
    SOURCE[key] = source


def rnd(x, k=4):
    """Round floats (and lists of floats) for the JSON; keep ints exact."""
    if isinstance(x, (list, tuple)):
        return [rnd(v, k) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return None
        return float(round(float(x), k))
    return x


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def src(p):
    return Path(p).read_text(encoding="utf-8")


def rel(p):
    return str(Path(p).relative_to(ROOT)).replace("\\", "/")


def must(pattern, text, what, flags=0):
    m = re.search(pattern, text, flags)
    if not m:
        raise RuntimeError(f"code pattern not found ({what}): {pattern}")
    return m


def line_of(text, pattern):
    for i, ln in enumerate(text.splitlines(), 1):
        if re.search(pattern, ln):
            return i
    raise RuntimeError(f"no line matches {pattern}")


def _literal(node):
    """literal_eval with support for field(default_factory=lambda: {...})."""
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "field":
        for kw in node.keywords:
            if kw.arg == "default_factory" and isinstance(kw.value, ast.Lambda):
                return ast.literal_eval(kw.value.body)
            if kw.arg == "default":
                return ast.literal_eval(kw.value)
        return None
    try:
        return ast.literal_eval(node)
    except ValueError:
        return None


def dataclass_defaults(path, cls):
    tree = ast.parse(src(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            out = {}
            for st in node.body:
                if isinstance(st, ast.AnnAssign) and st.value is not None:
                    out[st.target.id] = _literal(st.value)
            return out
    raise RuntimeError(f"class {cls} not in {path}")


def func_defaults(path, func, cls=None):
    tree = ast.parse(src(path))
    scope = tree
    if cls is not None:
        scope = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    for node in ast.walk(scope):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            a = node.args
            names = [x.arg for x in a.args]
            d = a.defaults
            out = {n: _literal(v) for n, v in zip(names[len(names) - len(d):], d)}
            for k, v in zip(a.kwonlyargs, a.kw_defaults):
                if v is not None:
                    out[k.arg] = _literal(v)
            return out
    raise RuntimeError(f"function {func} not in {path}")


def module_assign(path, name):
    tree = ast.parse(src(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return ast.literal_eval(node.value)
                if isinstance(tgt, ast.Tuple):
                    names = [e.id for e in tgt.elts]
                    if name in names:
                        return ast.literal_eval(node.value)[names.index(name)]
    raise RuntimeError(f"{name} not assigned at module level in {path}")


def boot_median_ci(x, seed):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    m = np.median(x[rng.integers(0, x.size, (NBOOT, x.size))], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def boot_ratio_ci(x, ref, seed):
    """CI of median(x)/median(ref), flies resampled independently within each group."""
    x = np.asarray(x, float); ref = np.asarray(ref, float)
    rng = np.random.default_rng(seed)
    mx = np.median(x[rng.integers(0, x.size, (NBOOT, x.size))], axis=1)
    mr = np.median(ref[rng.integers(0, ref.size, (NBOOT, ref.size))], axis=1)
    r = mx / mr
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def boot_paired_ratio_ci(a, b, seed):
    """CI of median(b)/median(a) with flies (pairs) resampled together."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, (NBOOT, a.size))
    r = np.median(b[idx], axis=1) / np.median(a[idx], axis=1)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


# ---------------------------------------------------------------- file handles
P_MANIFEST = ROOT / "connectome/manifest.json"
P_CAL = RES / "calibration/calibration.json"
P_MATCH = RES / "calibration/matched_type_input_synapses.csv"
P_SWEEP = RES / "calibration/adaptation_sweep.json"
P_FFAL = RES / "calibration/ff_al_test.json"
P_KCKC = RES / "calibration/kckc_test.json"
P_SENS = RES / "calibration/sensory_input_test.json"
P_INT = RES / "calibration/odor_intensity_sweep.csv"
P_PROBE = RES / "probe/probe_circuits.json"
P_ENG = RES / "validation/engine_validation.json"
P_ENGNPZ = RES / "validation/engine_validation_rates.npz"
P_MN9 = RES / "validation/summary_sugar_mn9.json"
P_PARQ = RES / "validation/baseline_nostim.parquet"
P_STAGE = RES / "battery/stage_curve_x4.csv"
P_COMP = RES / "battery/components_x4.csv"
P_VULN = RES / "battery/vulnerability_x4.csv"
P_SCOPE = RES / "battery/scope_x4.csv"
P_BSTATS = RES / "stats/battery_stats.json"
P_ONL = RES / "paper/output_neuron_loss.json"
P_JOINT = RES / "joint/cohort_joint.csv"
P_JMETA = RES / "joint/cohort_joint_meta.json"
P_EXP2 = RES / "exp2/battery.csv"

S_LIF = CODE / "adfly/lif.py"
S_MODEL = CODE / "adfly/model.py"
S_PATH = CODE / "adfly/pathology.py"
S_BUILD = CODE / "adfly/build_pack.py"
S_HOUSE = CODE / "adfly/house.py"
S_HNB = CODE / "adfly/house_nb.py"
S_CAL = CODE / "scripts/calibrate_wsyn.py"
S_SWEEP = CODE / "scripts/sweep_adaptation.py"
S_FFAL = CODE / "scripts/test_ff_al.py"
S_PROBE = CODE / "scripts/probe_circuits.py"
S_ENG = CODE / "scripts/validate_engine_final.py"
S_ENGFW = CODE / "scripts/validate_engine_flywire.py"
S_MN9 = CODE / "scripts/validate_sugar_mn9.py"
S_BAT = CODE / "scripts/assay_battery.py"
S_JOINT = CODE / "scripts/cohort_joint.py"
S_FIG = CODE / "scripts/figure_common.py"

put("_bootstrap_seed", SEED, "Seed of numpy default_rng used for every bootstrap interval in this file (4,000 resamples, percentile 2.5/97.5).", "this script")

# =============================================================== 1. connectome build
man = jload(P_MANIFEST)
put("connectome_manifest_found", True, "Whether connectome/manifest.json (written by build_pack.py) exists; it is the build record of the network (the network itself is not kept with the result tables).", rel(P_MANIFEST))
put("connectome_dataset", man["dataset"], "Name of the MaleCNS v1.0 materialisation the pack follows (superclass non-null, known transmitter).", rel(P_MANIFEST) + " dataset")
put("connectome_neurons", man["neurons"], "Number of MaleCNS v1.0 bodies retained (non-null superclass), i.e. neurons in the simulated network.", rel(P_MANIFEST) + " neurons")
put("connectome_edges_signed", man["edges"], "Number of signed directed edges (neuron pairs) kept in the CSR pack.", rel(P_MANIFEST) + " edges")
put("connectome_synaptic_contacts", man["synapses_abs"], "Total synaptic contacts carried by the kept edges (sum of |signed weight|).", rel(P_MANIFEST) + " synapses_abs")
put("connectome_raw_weight_rows", man["raw_rows"], "Rows in the raw MaleCNS connectome-weights table (minconf 0.5) before selection.", rel(P_MANIFEST) + " raw_rows")
put("connectome_rows_dropped_endpoint_unannotated", man["dropped_endpoint_not_selected"], "Raw weight rows dropped because an endpoint body had no superclass annotation (rows, not synapses).", rel(P_MANIFEST) + " dropped_endpoint_not_selected")
put("connectome_rows_dropped_endpoint_unannotated_frac", rnd(man["dropped_endpoint_not_selected"] / man["raw_rows"]), "Fraction of raw weight rows dropped for an unannotated endpoint.", rel(P_MANIFEST))
put("connectome_rows_dropped_unsigned_presyn", man["dropped_unsigned_presynaptic"], "Weight rows (edges) between retained bodies dropped because the presynaptic neuron's consensus transmitter is not ACh, GABA or Glu.", rel(P_MANIFEST) + " dropped_unsigned_presynaptic")
put("connectome_row_accounting_closes", bool(man["raw_rows"] - man["dropped_endpoint_not_selected"] - man["dropped_unsigned_presynaptic"] == man["edges"]), "True if raw rows minus both dropped counts equals the kept edge count exactly (no further rows lost).", rel(P_MANIFEST))
sc = man["sign_counts"]
put("connectome_neurons_excitatory", sc["1"], "Neurons whose outgoing edges are signed +1 (acetylcholine).", rel(P_MANIFEST) + " sign_counts[1]")
put("connectome_neurons_inhibitory", sc["-1"], "Neurons whose outgoing edges are signed -1 (GABA or glutamate).", rel(P_MANIFEST) + " sign_counts[-1]")
put("connectome_neurons_unsigned_no_output", sc["0"], "Neurons with another or no transmitter prediction: they receive input but send no edges.", rel(P_MANIFEST) + " sign_counts[0]")
bsrc = src(S_BUILD)
must(r'SIGN = \{"acetylcholine": 1, "gaba": -1, "glutamate": -1\}', bsrc, "sign rule")
put("connectome_sign_rule", "ACh +1, GABA -1, Glu -1, others no edges", "Signing rule of the MaleCNS pack (presynaptic consensus_nt).", rel(S_BUILD) + f" line {line_of(bsrc, 'SIGN = ')}")
put("connectome_minconf", 0.5, "Minimum synapse confidence of the MaleCNS source tables.", rel(S_BUILD) + " F_W/F_ANN filenames (minconf-0.5)")

pq = pd.read_parquet(P_PARQ)
assert (np.diff(pq.bodyId.to_numpy()) > 0).all(), "parquet not in pack (bodyId) order"
pt = pq.type.fillna("")
sens = pq.superclass.str.contains("sensory").to_numpy()
put("n_sensory_superclass_neurons", int(sens.sum()), "Neurons whose superclass contains 'sensory' (cb_, vnc_, ol_sensory, sensory_ascending/descending and _tbc): the only neurons exempt from pathology load and from adaptation.", rel(P_PARQ) + " superclass (pack order)")
put("n_sensory_superclass_frac", rnd(sens.mean()), "Fraction of the 166,700 neurons that are sensory and therefore carry zero load under every vulnerability map.", rel(P_PARQ))
cal_src = src(S_CAL)
mpre = must(r'brain = ~neu\.superclass\.str\.startswith\(\(([^)]*)\)\)', cal_src, "brain mask")
prefixes = tuple(ast.literal_eval("(" + mpre.group(1) + ")"))
brain = ~pq.superclass.str.startswith(prefixes)
vis = pq.superclass.isin(["ol_intrinsic", "ol_sensory", "visual_projection", "visual_centrifugal", "visual_projection_tbc"])
put("calib_brain_mask_excluded_prefixes", list(prefixes), "Superclass prefixes excluded from the MaleCNS 'brain' used for the synapse-count comparison (VNC and ascending only; optic lobes are kept).", rel(S_CAL) + f" line {line_of(cal_src, 'brain = ~neu')}")
put("calib_brain_mask_neurons", int(brain.sum()), "MaleCNS neurons inside that 'brain' mask (not a central-brain mask).", rel(P_PARQ) + " superclass with the calibrate_wsyn.py mask")
put("calib_brain_mask_optic_visual_neurons", int((brain & vis).sum()), "Of the brain-mask neurons, those in optic-lobe or visual superclasses (ol_intrinsic, ol_sensory, visual_projection, visual_centrifugal, visual_projection_tbc).", rel(P_PARQ))

# =============================================================== 2. neuron model and engine
par = dataclass_defaults(S_LIF, "Params")
for k, mean in [("v_0", "Resting potential (mV)."), ("v_rst", "Reset potential (mV)."), ("v_th", "Spike threshold (mV)."),
                ("t_mbr", "Membrane time constant (ms)."), ("tau", "Synaptic time constant (ms)."),
                ("t_rfc", "Refractory period (ms)."), ("t_dly", "Synaptic delay (ms)."),
                ("w_syn", "Unitary synaptic weight used throughout (mV per contact), the published Shiu et al. value."),
                ("f_poi", "Poisson-input scaling: each input event adds f_poi * w_syn to v."),
                ("dt", "Integration step (ms)."), ("tau_a", "Default adaptation time constant in Params (ms); adaptation is off unless da > 0.")]:
    put(f"lif_{k}", par[k], "LIF parameter: " + mean, rel(S_LIF) + " Params default")
put("lif_poisson_kick_mV", rnd(par["w_syn"] * par["f_poi"]),
    "Membrane jump per Poisson input event (w_syn * f_poi, mV), about ten times the 7-mV rest-to-threshold gap. "
    "The project's scripts give their Poisson targets no refractory period (Brains.no_refractory), so a driven neuron near rest, "
    "in the absence of adaptation and synaptic input, spikes on the step after almost every event; an event that arrives "
    "in the step a spike is emitted is overwritten by the reset (lif_isolated_driven_spikes_per_event). This need not hold "
    "for the paper model's driven PNs, PAM and LPLC2/LC4, which are non-sensory, carry adaptation (da 1 mV, tau_a 1 s) "
    "and receive synaptic input: healthy driven PNs miss more events than an isolated neuron "
    "(pn_spikes_healthy_over_isolated_expectation), while healthy PAM neurons fire like one (pam_healthy_median_over_isolated_expectation).", rel(S_LIF))
lsrc = src(S_LIF)
must(r'def no_refractory\(self, idx\):', lsrc, "no_refractory")
must(r'v\[b, j\] \+= stim_w', lsrc, "Poisson kick added to v")


def isolated_driven_spike_per_event(rate_hz, da, B=256, ms=1000.0):
    """Run the engine of code/adfly/lif.py on a one-neuron pack without edges.

    The neuron starts at rest, has no refractory period (as every Poisson
    target in the project), the paper model's tau_a and w_syn, and adaptation
    increment da; it is driven at rate_hz for ms. Returns spikes / events over
    B replicate brains (default_rng(SEED)) and the mean number of events.
    """
    import tempfile
    sys.path.insert(0, str(CODE))
    from adfly.lif import Brains, Params, poisson_events
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        np.save(d / "row_ptr.npy", np.zeros(2, np.int64))
        np.save(d / "dst.npy", np.zeros(0, np.int32))
        np.save(d / "signed_w.npy", np.zeros(0, np.int32))
        br = Brains(B, params=Params(tau_a=module_assign(S_MODEL, "TAU_A"), w_syn=module_assign(S_MODEL, "W_SYN")), pack=d)
    br.da[:, 0] = da
    br.no_refractory([0])
    br.set_groups(np.zeros(1, np.int32), 1)
    n = int(round(ms / br.p.dt))
    ev = poisson_events(np.random.default_rng(SEED), np.full((B, 1), rate_hz), n, br.p.dt)
    c = br.run(n, np.zeros((B, 1), np.int32), ev)
    return float(c.sum() / ev.sum()), float(ev.sum() / B)


ISO = {f"{r:g}Hz_da{d:g}": isolated_driven_spike_per_event(r, d) for r in (150.0, 100.0) for d in (0.0, module_assign(S_MODEL, "DA"))}
put("lif_isolated_driven_spikes_per_event", {k: rnd(v[0]) for k, v in ISO.items()},
    "Engine test (lif.py on a one-neuron pack without edges, 256 replicates, 1 s from rest, no refractory period, tau_a 1 s): "
    "spikes per Poisson event at 150 Hz (odour/sugar/looming drive) and 100 Hz (PAM drive), without (da 0) and with the "
    "paper model's adaptation (da 1 mV). Losses without adaptation are events overwritten by the reset in a spike step; "
    f"adaptation from rest adds {100 * (ISO['150Hz_da0'][0] - ISO['150Hz_da1'][0]):.1f} points of loss at 150 Hz and "
    f"{100 * (ISO['100Hz_da0'][0] - ISO['100Hz_da1'][0]):.1f} at 100 Hz over 1 s.", rel(S_LIF) + " Brains.run; this script")
must(r'self\.da = np\.zeros\(\(B, self\.N\), np\.float32\)', lsrc, "da off by default")
put("lif_adaptation_off_by_default", True, "Brains initialises da to zeros, so every script that builds Brains without setting da (calibrate_wsyn, probe_circuits, validate_sugar_mn9, validate_engine_*) ran the unmodified base model without adaptation.", rel(S_LIF) + f" line {line_of(lsrc, 'self.da = np.zeros')}")
gate_def = func_defaults(S_LIF, "__init__", "Brains")["gate_refractory_input"]
put("lif_gate_refractory_input_default", gate_def, "Default engine setting: synaptic input arriving during the refractory period is discarded (the setting that matched Brian2 2.10.1); used by all paper simulations.", rel(S_LIF) + " Brains.__init__ default")
put("model_tau_a_ms", module_assign(S_MODEL, "TAU_A"), "Adaptation time constant of the paper model (ms).", rel(S_MODEL) + " TAU_A")
put("model_da_mV", module_assign(S_MODEL, "DA"), "Adaptation increment per spike in the paper model (mV), applied to non-sensory neurons only.", rel(S_MODEL) + " DA")
put("model_w_syn_mV", module_assign(S_MODEL, "W_SYN"), "Unitary weight of the paper model (mV).", rel(S_MODEL) + " W_SYN")
msrc = src(S_MODEL)
must(r'br\.da\[:, ~cx\.sensory\] = DA', msrc, "da on non-sensory")
must(r'w\[cx\.al_edge\] = 0\.0', msrc, "AL edges zeroed")
odors = module_assign(S_MODEL, "ODORS")
put("odour_A_glomeruli", odors["A"], "Glomeruli whose uniglomerular PNs are driven for odour A (always the paired odour).", rel(S_MODEL) + " ODORS")
put("odour_B_glomeruli", odors["B"], "Glomeruli whose uniglomerular PNs are driven for odour B (never paired).", rel(S_MODEL) + " ODORS")

# =============================================================== 3. engine validation (FlyWire v630)
eng = jload(P_ENG)
esrc = src(S_ENG)
seeds = ast.literal_eval(must(r'for seed in (\([^)]*\)):', esrc, "seeds").group(1))
must(r'ours = per_seed\[0\]', esrc, "seed 7 used")
must(r'np\.polyfit\(ref\[a\], ours\[a\], 1\)', esrc, "regression direction")
put("engine_validation_network", eng["model"], "Network and model of the engine validation (base model, no MaleCNS extensions).", rel(P_ENG) + " model")
put("engine_validation_neurons", eng["neurons"], "Neurons in the FlyWire v630 network used for engine validation.", rel(P_ENG) + " neurons")
put("engine_validation_stimulus", eng["stimulus"], "Stimulus of the engine validation run.", rel(P_ENG) + " stimulus")
put("engine_validation_seeds", list(seeds), "Poisson input seeds run for each engine setting (30 trials of 1 s each).", rel(S_ENG) + f" line {line_of(esrc, 'for seed in')}")
put("engine_validation_seed_used_for_comparisons", seeds[0], "Only the first seed's 30-trial realisation enters the per-neuron correlations, slopes and S1 Fig.", rel(S_ENG) + f" line {line_of(esrc, 'ours = per_seed')}")
put("engine_validation_slope_direction", "engine rate regressed on reference rate (np.polyfit(ref, ours)), linear rates over neurons active in either run", "Direction and scale of the reported regression slopes.", rel(S_ENG) + f" line {line_of(esrc, 'np.polyfit')}")
for gate, lab, done in (("gate=True", "discard", "discarded"), ("gate=False", "accumulate", "accumulated")):
    runs = eng["engine_runs"][gate]
    put(f"engine_{lab}_spikes_per_trial_by_seed", [r["spikes_per_trial"] for r in runs], f"Total spikes per 1-s trial, engine with refractory input {done}, seeds {list(seeds)}.", rel(P_ENG) + f" engine_runs[{gate}]")
    put(f"engine_{lab}_mn9_hz_by_seed", [r["mn9_hz"] for r in runs], f"MN9 rate (Hz), engine with refractory input {done}, seeds {list(seeds)}.", rel(P_ENG) + f" engine_runs[{gate}]")
    put(f"engine_{lab}_active_by_seed", [r["active_neurons"] for r in runs], f"Neurons with nonzero mean rate, engine with refractory input {done}, seeds {list(seeds)}.", rel(P_ENG) + f" engine_runs[{gate}]")
for rk, lab in (("local_brian2_2.10.1", "brian2_local"), ("released_example_output", "released_example")):
    r = eng["references"][rk]
    put(f"engine_ref_{lab}", [r["spikes_per_trial"], r["active_neurons"], r["mn9_hz"]], f"Reference '{rk}': [spikes per trial, active neurons, MN9 Hz] (single realisation).", rel(P_ENG) + f" references[{rk}]")
for ck, lab in (("gate=True vs local_brian2_2.10.1", "discard_vs_local"), ("gate=True vs released_example_output", "discard_vs_released"),
                ("gate=False vs local_brian2_2.10.1", "accumulate_vs_local"), ("gate=False vs released_example_output", "accumulate_vs_released")):
    c = eng["comparisons"][ck]
    put(f"engine_cmp_{lab}", [c["pearson_r"], c["spearman_r"], c["slope"], c["n_compared"]], f"Comparison '{ck}' (seed {seeds[0]}): [Pearson r, Spearman r, slope engine-on-reference, n neurons active in either run].", rel(P_ENG) + f" comparisons[{ck}]")
z = np.load(P_ENGNPZ)
for ours_k, ref_k, lab in (("ours_gate1", "ref_local_brian2_2.10.1", "discard_vs_local"),
                           ("ours_gate1", "ref_released_example_output", "discard_vs_released"),
                           ("ours_gate0", "ref_local_brian2_2.10.1", "accumulate_vs_local"),
                           ("ours_gate0", "ref_released_example_output", "accumulate_vs_released")):
    o, rf = z[ours_k], z[ref_k]
    a = (o > 0) | (rf > 0)
    put(f"engine_cmp_{lab}_recomputed", [rnd(stats.pearsonr(o[a], rf[a])[0]), rnd(np.polyfit(rf[a], o[a], 1)[0], 3), rnd(np.polyfit(o[a], rf[a], 1)[0], 3), int(a.sum())],
        f"Recomputed from the saved seed-{seeds[0]} rate vectors: [Pearson r, slope engine-on-reference, reverse slope reference-on-engine, n]; the reverse slope shows how much the reported slope depends on regression direction.", rel(P_ENGNPZ))
dis = eng["engine_runs"]["gate=True"]
put("engine_discard_seed_spread_spikes", rnd(max(r["spikes_per_trial"] for r in dis) - min(r["spikes_per_trial"] for r in dis), 1), "Range (max-min) of spikes per trial across the three seeds, discard setting.", rel(P_ENG))
put("engine_discard_seed_spread_mn9_hz", rnd(max(r["mn9_hz"] for r in dis) - min(r["mn9_hz"] for r in dis), 2), "Range of MN9 rate across the three seeds, discard setting (Hz).", rel(P_ENG))
put("engine_discard_mn9_minus_brian2_by_seed", rnd([r["mn9_hz"] - eng["references"]["local_brian2_2.10.1"]["mn9_hz"] for r in dis], 2), "Engine minus local Brian2 MN9 rate (Hz) for each seed; all positive, so the reference lies below every engine seed.", rel(P_ENG))
fsrc = src(S_ENGFW)
must(r'Excitatory x Connectivity', fsrc, "v630 sign column")
must(r'2023_03_23_connectivity_630_final', fsrc, "v630 file")
put("flywire_sign_convention", "Shiu et al. 'Excitatory x Connectivity' column (GABA/Glu negative, all else positive) for both v630 and v783 packs", "FlyWire packs are not signed by the MaleCNS rule (which drops non-ACh/GABA/Glu outputs).", rel(S_ENGFW) + f" line {line_of(fsrc, 'Excitatory x Connectivity')}; " + rel(S_CAL) + f" line {line_of(cal_src, 'Excitatory x Connectivity')}")

# =============================================================== 4. MaleCNS gustatory check (base model)
msum = {e["name"]: e for e in jload(P_MN9)}
mn9src = src(S_MN9)
dflt = func_defaults(S_MN9, "experiment")
put("malecns_gust_check_protocol", f"base model (w_syn {par['w_syn']}, no adaptation, recurrent AL), {dflt['rate']:g} Hz Poisson, {dflt['n_trials']} trials of {dflt['t_ms']:g} ms, seed {dflt['seed']}, right-side GRNs", "Protocol of validate_sugar_mn9.py on MaleCNS.", rel(S_MN9) + " experiment() defaults")
for name, lab in (("baseline_nostim", "nostim"), ("sugar_LB3c_R", "LB3c_R"), ("sugar_LB3c_LB3b_R", "LB3cb_R"), ("bitter_LB1abcd_R", "bitter_R")):
    e = msum[name]
    mL = next(m for m in e["mn9"] if m["side"] == "L"); mR = next(m for m in e["mn9"] if m["side"] == "R")
    put(f"malecns_gust_{lab}_n_stim", e["n_stim"], f"Stimulated neurons in condition {name}.", rel(P_MN9))
    put(f"malecns_gust_{lab}_n_active", e["n_active"], f"Neurons with nonzero mean rate (30 trials) in condition {name}.", rel(P_MN9))
    put(f"malecns_gust_{lab}_mn9_L_R_hz", [rnd(mL["rate_mean"], 1), rnd(mR["rate_mean"], 1)], f"Mean rate (Hz) of the left-annotated MN9 (bodyId {mL['bodyId']}) and right-annotated MN9 (bodyId {mR['bodyId']}) in condition {name}.", rel(P_MN9))
    put(f"malecns_gust_{lab}_mn9_L_sd_hz", rnd(mL["rate_sd"], 2), f"SD across 30 trials of the left-annotated MN9 rate in condition {name}.", rel(P_MN9))
    pp = pd.read_parquet(RES / f"validation/{name}.parquet")
    put(f"malecns_gust_{lab}_spikes_per_trial", rnd(pp.rate_mean.sum(), 0), f"Total spikes per 1-s trial (sum of per-neuron mean rates) in condition {name}.", rel(RES / f'validation/{name}.parquet'))
pb = pd.read_parquet(RES / "validation/bitter_LB1abcd_R.parquet")
put("malecns_gust_bitter_R_kc_active", int(((pb.rate_mean > 0) & pb.type.fillna("").str.startswith("KC")).sum()), "Kenyon cells with nonzero rate under right bitter (LB1a-d) stimulation, base model.", rel(RES / "validation/bitter_LB1abcd_R.parquet"))

# =============================================================== 5. synaptic weight calibration
cal = jload(P_CAL)
mt = pd.read_csv(P_MATCH)
must(r'Connectivity_783\.parquet', cal_src, "v783")
must(r'Completeness_783\.csv', cal_src, "v783")
flt = must(r'm\[\(m\.mean_fw >= (\d+)\) & \(m\.mean_mc >= (\d+)\)\]', cal_src, "filter")
must(r'groupby\("flywireType"\)', cal_src, "mc key")
must(r'groupby\("cell_type"\)', cal_src, "fw key")
put("calib_flywire_version", "v783", "FlyWire release used for the synapse-count calibration and for the FlyWire odour-persistence test (Connectivity_783 / Completeness_783).", rel(S_CAL) + f" line {line_of(cal_src, 'Connectivity_783')}")
_l1, _l2 = line_of(cal_src, r'groupby\("flywireType"\)'), line_of(cal_src, r'groupby\("cell_type"\)')
put("calib_type_matching_key", "MaleCNS flywireType == FlyWire cell_type", "Key used to match cell types between the two datasets.", rel(S_CAL) + f" lines {_l1}, {_l2}")
put("calib_min_mean_input_synapses", int(flt.group(1)), "Inclusion filter: a matched type enters only if its mean input synapses per neuron are >= this value in BOTH datasets.", rel(S_CAL) + f" line {line_of(cal_src, 'mean_fw >= ')}")
put("calib_n_matched_types", cal["n_matched_types"], "Matched cell types passing the filter.", rel(P_CAL) + " n_matched_types")
put("calib_n_matched_types_recomputed", int(len(mt)), "Rows of matched_type_input_synapses.csv (should equal n_matched_types).", rel(P_MATCH))
put("calib_min_mean_fw_mc_in_table", [rnd(mt.mean_fw.min(), 2), rnd(mt.mean_mc.min(), 2)], "Smallest FlyWire and MaleCNS type-mean input synapses among matched types (confirms the >= 50 filter).", rel(P_MATCH))
put("calib_ratio_median", rnd(cal["ratio_median"]), "Median over matched types of MaleCNS/FlyWire mean input synapses per neuron.", rel(P_CAL) + " ratio_median")
put("calib_ratio_median_recomputed", rnd(float((mt.mean_mc / mt.mean_fw).median())), "Same median recomputed from the matched-type table.", rel(P_MATCH))
put("calib_ratio_iqr", rnd(cal["ratio_iqr"]), "Interquartile range (25th, 75th percentile) of the per-type ratio.", rel(P_CAL) + " ratio_iqr")
put("calib_w_syn_scaled_mV", rnd(cal["w_syn_calibrated_mV"]), "0.275 mV divided by the median ratio: the weight a face-value rescaling would give (reported as 0.157 mV).", rel(P_CAL) + " w_syn_calibrated_mV")
put("calib_w_syn_scaled_over_published", rnd(cal["w_syn_calibrated_mV"] / 0.275, 3), "Rescaled weight as a fraction of 0.275 mV (outside the 0.8-1.2x robustness range of the counterfactual experiment).", rel(P_CAL))
put("calib_fw_mean_in_syn_per_neuron", rnd(cal["fw_mean_in_syn_per_neuron"], 1), "Mean input synapses per neuron over all FlyWire v783 neurons (optic lobes included).", rel(P_CAL) + " fw_mean_in_syn_per_neuron")
put("calib_mc_brain_mean_in_syn_per_neuron", rnd(cal["mc_brain_mean_in_syn_per_neuron"], 1), "Mean input synapses per neuron over the MaleCNS brain mask (excluding VNC/ascending only, optic lobes included; partners counted only among the 166,700 retained bodies).", rel(P_CAL) + " mc_brain_mean_in_syn_per_neuron")
pers_def = func_defaults(S_CAL, "persistence")
must(r'on = br\.run\(3000,', cal_src, "300 ms on"); must(r'off = br\.run\(7000,', cal_src, "700 ms off")
put("calib_persistence_protocol", f"base model (no adaptation, recurrent AL), Poisson 150 Hz for 300 ms then 700 ms off, 10 bins of 100 ms, rates averaged over B replicates (odour B={pers_def['B']}, sugar B=16), seed 0", "Protocol of calibrate_wsyn.py persistence(): 'active' = neurons whose replicate-mean rate is > 0 in a bin.", rel(S_CAL) + f" lines {line_of(cal_src, 'def persistence')}-{line_of(cal_src, 'return r')}")
must(r'neu\.type\.isin\(\["LB3c", "LB3b"\]\) & \(neu\.side == "R"\)', cal_src, "sugar right")
mn9_order = pq.loc[pt == "MN9", ["bodyId", "side"]]
put("mn9_pack_order_sides", mn9_order.side.tolist(), "Soma sides of the two MN9s in pack (bodyId) order; lists of MN9 rates in calibration/sweep/kckc files follow this order.", rel(P_PARQ))
gf_order = pq.loc[pt == "DNp01", ["bodyId", "side"]]
put("dnp01_pack_order_sides", gf_order.side.tolist(), "Soma sides of the two giant fibres (DNp01) in pack order; GF rate lists in the sweep/kckc files follow this order.", rel(P_PARQ))
put("calib_w0157_sugar_mn9_hz", cal["malecns_sugar_wcal_MN9_on_hz"], "At 0.157 mV, right LB3c+LB3b sugar: rate (Hz) of each MN9 (pack order L, R) averaged over the whole 300-ms stimulus (B=16), not the first 100 ms.", rel(P_CAL) + " malecns_sugar_wcal_MN9_on_hz")
put("calib_w0157_sugar_mn9_mean_of_two_hz", rnd(np.mean(cal["malecns_sugar_wcal_MN9_on_hz"]), 1), "Mean of the two MN9 rates in that run (Hz).", rel(P_CAL))
sug_bins = cal["malecns_sugar_wcal_active_per_100ms"]
put("calib_w0157_sugar_active_per_100ms", sug_bins, "At 0.157 mV, sugar run: active neurons in each 100-ms bin (bins 0-2 stimulus, 3-9 after offset).", rel(P_CAL) + " malecns_sugar_wcal_active_per_100ms")
first0 = next(i for i, v in enumerate(sug_bins) if v == 0)
put("calib_w0157_sugar_silent_from_ms", first0 * 100, "Start (ms from stimulus onset) of the first bin with no active neuron; = 100 ms after the 300-ms stimulus ended.", rel(P_CAL))
for w in ("0.275", "0.157"):
    b = cal[f"malecns_odor_w{w}_active_per_100ms"]; kc3 = cal[f"malecns_odor_w{w}_KC_active_on_vs_last"]
    put(f"calib_malecns_odour_w{w}_active_range_all_bins", [min(b), max(b)], f"MaleCNS, w_syn {w} mV, left DM1/DM4/VA2 ORNs: min and max active neurons per 100-ms bin over the whole 1-s run.", rel(P_CAL) + f" malecns_odor_w{w}_active_per_100ms")
    put(f"calib_malecns_odour_w{w}_active_last_bin", b[-1], f"MaleCNS, w_syn {w} mV odour run: active neurons in the last bin (600-700 ms after offset).", rel(P_CAL))
    put(f"calib_malecns_odour_w{w}_kc_on_last_total", kc3, f"MaleCNS, w_syn {w} mV odour run: [KCs active during stimulus, KCs active in last 200 ms (500-700 ms after offset), total KCs] (union over B=4 replicates).", rel(P_CAL) + f" malecns_odor_w{w}_KC_active_on_vs_last")
fwb = cal["flywire_odor_active_per_100ms"]; fwk = cal["flywire_odor_KC_active_on_vs_last"]
put("calib_flywire783_odour_active_range_all_bins", [min(fwb), max(fwb)], "FlyWire v783 (w 0.275, no adaptation), left ORN DM1/DM4/VA2 at 150 Hz for 300 ms: min and max active neurons per 100-ms bin over 1 s.", rel(P_CAL) + " flywire_odor_active_per_100ms")
put("calib_flywire783_odour_active_range_post_offset", [min(fwb[3:]), max(fwb[3:])], "Same run, bins after stimulus offset only (300-1000 ms).", rel(P_CAL))
put("calib_flywire783_odour_kc_on_last_total", fwk, "FlyWire v783 odour run: [KCs active during stimulus, KCs active 500-700 ms after offset, total KCs].", rel(P_CAL) + " flywire_odor_KC_active_on_vs_last")
put("calib_flywire_odour_stimulus_sets_tested", 1, "Number of distinct odour (ORN) stimulus sets ever run on FlyWire: one (left DM1/DM4/VA2), so 'independent of odour identity on FlyWire' is untested.", rel(S_CAL) + f" line {line_of(cal_src, 'ORN_DM1')}")
# kckc_test.json, sensory_input_test.json and odor_intensity_sweep.csv: is the writing script in code/?
gen = [p for p in glob.glob(str(CODE / "**/*.py"), recursive=True)
       if not Path(p).name.startswith("facts_") and re.search(r"kckc_test|sensory_input_test|odor_intensity_sweep", src(p))]
kk = jload(P_KCKC)
sw = jload(P_SWEEP)
_sw0 = next(r for r in sw if r["da"] == 0.0)
_kk0 = kk["w0.275_dropKCKCFalse"]
_pairs = (("odor_active_on", "odor_active_on"), ("odor_active_last200", "odor_active_last200"), ("KC_frac_active_on", "KC_frac_on"),
          ("MN9_sugar_hz", "sugar_MN9_hz"), ("sugar_active_last200", "sugar_active_last200"), ("GF_LPLC2_hz", "loom_GF_hz"),
          ("lplc2_active_last200", "loom_active_last200"))
kk_eq_sw0 = all(_kk0[a] == _sw0[b] for a, b in _pairs)
put("kckc_w0275_keep_equals_sweep_da0_row", kk_eq_sw0, f"Whether the kckc_test keep-KC-KC w0.275 row equals the adaptation-sweep no-adaptation row in all {len(_pairs)} shared fields (odour active on and last200, KC fraction, MN9 Hz, sugar last200, GF Hz, looming last200); if so the two files share stimuli, model and window.", rel(P_KCKC) + "; " + rel(P_SWEEP))
NOSCRIPT = "" if gen else " Protocol inferred, not verified: no script in code/ writes this file (alt_tests_generating_scripts_found = 0)"
NOSCRIPT_KK = NOSCRIPT and (NOSCRIPT + ("; its keep-KC-KC w0.275 row equals the sweep_adaptation.py no-adaptation row exactly (kckc_w0275_keep_equals_sweep_da0_row), which supports left ORN DM1/DM4/VA2 odour, right sugar, right LPLC2, recurrent AL and the 500-700-ms window." if kk_eq_sw0 else "."))
NOSCRIPT_SI = NOSCRIPT and (NOSCRIPT + "; stimuli, recurrent AL and the 500-700-ms window are assumed to follow sweep_adaptation.py, as for kckc_test.")
NOSCRIPT_INT = NOSCRIPT and (NOSCRIPT + "; ORN drive is read from the n_orn column, and recurrent AL and the 500-700-ms window are assumed to follow sweep_adaptation.py.")
for kname, lab in (("w0.275_dropKCKCFalse", "w0275"), ("w0.157_dropKCKCFalse", "w0157")):
    r = kk[kname]
    put(f"kckc_{lab}_mn9_sugar_hz", r["MN9_sugar_hz"], f"kckc_test {kname}: MN9 rates (pack order L, R; Hz) under right sugar (like-for-like run at both weights)." + NOSCRIPT_KK, rel(P_KCKC) + f" {kname}")
    put(f"kckc_{lab}_gf_lplc2_hz", r["GF_LPLC2_hz"], f"kckc_test {kname}: DNp01 rates (pack order R, L; Hz) under right LPLC2 looming." + NOSCRIPT_KK, rel(P_KCKC) + f" {kname}")
    put(f"kckc_{lab}_odour_sugar_loom_last200", [r["odor_active_last200"], r["sugar_active_last200"], r["lplc2_active_last200"]], f"kckc_test {kname}: neurons active 500-700 ms after offset for [odour, sugar, looming]." + NOSCRIPT_KK, rel(P_KCKC) + f" {kname}")
    put(f"kckc_{lab}_kc_frac_on", r["KC_frac_active_on"], f"kckc_test {kname}: fraction of KCs active during the odour (union over replicates)." + NOSCRIPT_KK, rel(P_KCKC) + f" {kname}")
put("calib_mn9_ratio_w0157_vs_w0275", rnd(kk["w0.157_dropKCKCFalse"]["MN9_sugar_hz"][0] / kk["w0.275_dropKCKCFalse"]["MN9_sugar_hz"][0], 3), "Driven-MN9 rate at 0.157 mV divided by that at 0.275 mV (kckc_test, same protocol): about a fourfold loss." + NOSCRIPT_KK, rel(P_KCKC))
put("calib_gf_ratio_w0157_vs_w0275", rnd(kk["w0.157_dropKCKCFalse"]["GF_LPLC2_hz"][0] / kk["w0.275_dropKCKCFalse"]["GF_LPLC2_hz"][0], 3), "Ipsilateral giant-fibre rate at 0.157 mV divided by that at 0.275 mV (kckc_test)." + NOSCRIPT_KK, rel(P_KCKC))

# =============================================================== 6. adaptation sweep (persistence)
ssrc = src(S_SWEEP)
must(r'on = br\.run\(3000,', ssrc, "on"); must(r'off = br\.run\(7000,', ssrc, "off")
must(r'"odor_active_last200": int\(\(r\["odor"\]\[-2:\] > 0\)\.any\(0\)\.sum\(\)\)', ssrc, "last200 def")
c3 = must(r'roughly (\d+)-(\d+) %', ssrc, "C3 range")
swdef = func_defaults(S_SWEEP, "run")
put("sweep_on_window_ms", [0, 300], "Stimulus window (ms from onset) over which sweep 'on' quantities (KC fraction/rate, MN9, GF) are averaged.", rel(S_SWEEP) + f" line {line_of(ssrc, 'on = br.run')}")
put("sweep_persistence_window_after_offset_ms", [500, 700], "Window of every '_last200' persistence count: the last 200 ms of a 700-ms post-stimulus window, i.e. 500-700 ms after offset.", rel(S_SWEEP) + f" line {line_of(ssrc, 'odor_active_last200')}")
put("sweep_replicates_and_seed", [swdef["B"], swdef["seed"]], "Replicate brains per run and Poisson seed; 'active' = replicate-mean rate > 0, i.e. a union over replicates.", rel(S_SWEEP) + " run() defaults")
put("sweep_stimuli", "odour: left ORN DM1/DM4/VA2; sugar: right LB3c+LB3b; looming: right LPLC2; all 150 Hz, 300 ms; base MaleCNS model with recurrent AL, w 0.275", "Stimuli of the adaptation sweep (differ from the battery: bilateral sugar, left LPLC2+LC4, PN-driven odour).", rel(S_SWEEP) + f" line {line_of(ssrc, 'STIM = ')}")
put("criterion_C3_kc_sparse_range_pct", [int(c3.group(1)), int(c3.group(2))], "Pre-set KC sparseness criterion (percent of KCs firing during a single odour), written in the sweep docstring.", rel(S_SWEEP) + f" line {line_of(ssrc, 'roughly')}")
put("criterion_C1_quiescence", "~0 active neurons in the last 200 ms of a 700-ms post-stimulus window for odour, sugar and looming", "Pre-set quiescence criterion (sweep docstring).", rel(S_SWEEP) + f" line {line_of(ssrc, 'C1 quiescence')}")
put("criterion_C2_responses_kept", "sugar GRNs -> MN9 and LPLC2 -> DNp01 remain strongly driven", "Pre-set response criterion (sweep docstring); no numeric threshold.", rel(S_SWEEP) + f" line {line_of(ssrc, 'C2 validated')}")
put("sweep_settings_tau_da", [[r["tau_a"], r["da"]] for r in sw], "The 9 sweep settings [tau_a ms, da mV] in file order (first = no adaptation).", rel(P_SWEEP))
put("sweep_odour_last200_by_setting", [r["odor_active_last200"] for r in sw], "Neurons active 500-700 ms after odour offset, per setting (same order).", rel(P_SWEEP) + " odor_active_last200")
put("sweep_sugar_last200_by_setting", [r["sugar_active_last200"] for r in sw], "Neurons active 500-700 ms after sugar offset, per setting.", rel(P_SWEEP) + " sugar_active_last200")
put("sweep_loom_last200_by_setting", [r["loom_active_last200"] for r in sw], "Neurons active 500-700 ms after looming offset, per setting.", rel(P_SWEEP) + " loom_active_last200")
put("sweep_kc_frac_on_by_setting", [r["KC_frac_on"] for r in sw], "Fraction of KCs active during the odour (union over 4 replicates), per setting.", rel(P_SWEEP) + " KC_frac_on")
put("sweep_kc_mean_hz_on_by_setting", [r["KC_mean_hz_on"] for r in sw], "Mean KC rate during the odour (Hz), per setting.", rel(P_SWEEP) + " KC_mean_hz_on")
put("sweep_mn9_driven_hz_by_setting", [r["sugar_MN9_hz"][0] for r in sw], "Rate of the driven (left-annotated) MN9 under right sugar, 0-300 ms (Hz), per setting.", rel(P_SWEEP) + " sugar_MN9_hz[0]")
put("sweep_gf_ipsi_hz_by_setting", [r["loom_GF_hz"][0] for r in sw], "Rate of the right (ipsilateral) DNp01 under right LPLC2, 0-300 ms (Hz), per setting.", rel(P_SWEEP) + " loom_GF_hz[0]")
put("sweep_gf_contra_hz_by_setting", [r["loom_GF_hz"][1] for r in sw], "Rate of the left (contralateral) DNp01 under right LPLC2 (Hz), per setting.", rel(P_SWEEP) + " loom_GF_hz[1]")
base = next(r for r in sw if r["da"] == 0.0)
chosen = next(r for r in sw if r["tau_a"] == module_assign(S_MODEL, "TAU_A") and r["da"] == module_assign(S_MODEL, "DA"))
put("persist_noadapt_odour_sugar_loom_last200", [base["odor_active_last200"], base["sugar_active_last200"], base["loom_active_last200"]], "Base MaleCNS model without adaptation: neurons still active 500-700 ms after offset of [odour, sugar, looming]; persistence is not only olfactory.", rel(P_SWEEP) + " row da=0")
put("persist_chosen_odour_sugar_loom_last200", [chosen["odor_active_last200"], chosen["sugar_active_last200"], chosen["loom_active_last200"]], "Chosen setting (tau_a 1000 ms, da 1 mV), recurrent AL: neurons active 500-700 ms after offset of [odour, sugar, looming].", rel(P_SWEEP) + " row tau_a=1000, da=1")
put("persist_noadapt_odour_active_on", base["odor_active_on"], "Neurons active during the 300-ms odour without adaptation.", rel(P_SWEEP))
put("adapt_kc_mean_hz_noadapt_to_chosen", [base["KC_mean_hz_on"], chosen["KC_mean_hz_on"]], "Mean KC rate during odour (Hz) without adaptation and at the chosen setting (recurrent AL, ORN drive).", rel(P_SWEEP))
put("adapt_kc_frac_noadapt_to_chosen", [base["KC_frac_on"], chosen["KC_frac_on"]], "KC fraction active during odour without adaptation and at the chosen setting (union over 4 replicates).", rel(P_SWEEP))
put("adapt_cost_mn9_hz", [base["sugar_MN9_hz"][0], chosen["sugar_MN9_hz"][0]], "Driven MN9 rate (Hz) without adaptation and at the chosen setting.", rel(P_SWEEP))
put("adapt_cost_mn9_pct_change", rnd(100 * (chosen["sugar_MN9_hz"][0] / base["sugar_MN9_hz"][0] - 1), 1), "Percent change of driven MN9 rate caused by the chosen adaptation.", rel(P_SWEEP))
put("adapt_cost_gf_ipsi_hz", [base["loom_GF_hz"][0], chosen["loom_GF_hz"][0]], "Ipsilateral (right) DNp01 rate (Hz) without adaptation and at the chosen setting.", rel(P_SWEEP))
put("adapt_cost_gf_ipsi_pct_change", rnd(100 * (chosen["loom_GF_hz"][0] / base["loom_GF_hz"][0] - 1), 1), "Percent change of ipsilateral DNp01 rate caused by the chosen adaptation.", rel(P_SWEEP))
put("adapt_cost_gf_contra_pct_change", rnd(100 * (chosen["loom_GF_hz"][1] / base["loom_GF_hz"][1] - 1), 1), "Percent change of contralateral DNp01 rate caused by the chosen adaptation.", rel(P_SWEEP))
for da in (2.0, 4.0):
    r = next(x for x in sw if x["tau_a"] == 1000.0 and x["da"] == da)
    put(f"adapt_tau1000_da{int(da)}_odour_last200_kcfrac_mn9_gf", [r["odor_active_last200"], r["KC_frac_on"], r["sugar_MN9_hz"][0], r["loom_GF_hz"][0]],
        f"Stronger adaptation (tau_a 1000 ms, da {da:g} mV): [odour neurons active 500-700 ms after offset, KC fraction on, driven MN9 Hz, ipsilateral GF Hz].", rel(P_SWEEP))
r4 = next(x for x in sw if x["tau_a"] == 1000.0 and x["da"] == 4.0)
put("adapt_da4_mn9_pct_change", rnd(100 * (r4["sugar_MN9_hz"][0] / base["sugar_MN9_hz"][0] - 1), 1), "Percent change of driven MN9 at tau_a 1000 ms, da 4 mV (the only setting with zero odour persistence).", rel(P_SWEEP))
adap = [r for r in sw if r["da"] > 0]
put("sweep_kc_frac_range_adaptation_settings", [min(r["KC_frac_on"] for r in adap), max(r["KC_frac_on"] for r in adap)], "Range of KC fraction across the 8 adaptation settings: none reaches the 5-20% criterion, so adaptation alone cannot make KCs sparse.", rel(P_SWEEP))
put("sweep_settings_all_three_quiescent", [[r["tau_a"], r["da"]] for r in sw if r["odor_active_last200"] == 0 and r["sugar_active_last200"] == 0 and r["loom_active_last200"] == 0], "Settings with zero persistence for odour, sugar and looming at once.", rel(P_SWEEP))
put("sweep_settings_sugar_and_loom_quiescent", [[r["tau_a"], r["da"]] for r in sw if r["sugar_active_last200"] == 0 and r["loom_active_last200"] == 0], "Settings with zero persistence after sugar and after looming.", rel(P_SWEEP))

# probe: sugar persistence in 50-ms bins (base model)
pr = jload(P_PROBE)
psrc = src(S_PROBE)
prdef = func_defaults(S_PROBE, "run")
ps = pr["persistence_sugar"]
on_bins = int(300 / ps["bin_ms"])
put("probe_protocol", f"base model (default Params: w 0.275, da 0, recurrent AL), {prdef['n_trials']} trials, {prdef['bin_ms']:g}-ms bins, seed {prdef['seed']}", "Protocol of probe_circuits.py (not the paper model).", rel(S_PROBE) + " run() defaults")
put("probe_sugar_active_per_bin_stimulus_range", [min(ps["n_active_per_bin"][:on_bins]), max(ps["n_active_per_bin"][:on_bins])], "Right LB3c+LB3b sugar, 150 Hz, 0-300 ms: min/max active neurons per 50-ms bin during the stimulus.", rel(P_PROBE) + " persistence_sugar")
post = ps["n_active_per_bin"][on_bins:]
assert post[0] == max(post) and min(post[1:]) < post[0], "probe sugar post-offset shape changed; revise the meanings"
put("probe_sugar_active_per_bin_post_offset_range", [min(post), max(post)], "Same run, all 50-ms bins from offset to 700 ms after: min and max active neurons. The maximum is the first post-offset bin; after an initial drop the count plateaus with no decay toward rest (see the two keys below).", rel(P_PROBE))
put("probe_sugar_active_first_post_offset_bin_and_drop_frac", [post[0], rnd(1 - post[1] / post[0], 3)], "Active neurons in the first 50-ms bin after sugar offset (300-350 ms), and the fractional drop to the next bin (350-400 ms).", rel(P_PROBE) + " persistence_sugar")
put("probe_sugar_active_per_bin_post_offset_after_first_bin_range", [min(post[1:]), max(post[1:])], "Active neurons per 50-ms bin from 350 ms to 1000 ms (50-700 ms after offset): a plateau with no decay toward rest.", rel(P_PROBE) + " persistence_sugar")
put("probe_sugar_total_rate_stimulus_range_hz", [min(ps["total_rate_per_bin"][:on_bins]), max(ps["total_rate_per_bin"][:on_bins])], "Summed rate over all neurons (Hz) per 50-ms bin during the sugar stimulus (first bin included).", rel(P_PROBE))
put("probe_sugar_total_rate_post_offset_range_hz", [min(ps["total_rate_per_bin"][on_bins:]), max(ps["total_rate_per_bin"][on_bins:])], "Summed rate per 50-ms bin after sugar offset: flat, no decay.", rel(P_PROBE))

# =============================================================== 7. failed alternatives
put("alt_drop_kckc_w0275_odour_last200", [kk["w0.275_dropKCKCFalse"]["odor_active_last200"], kk["w0.275_dropKCKCTrue"]["odor_active_last200"]], "Odour neurons active 500-700 ms after offset with KC-KC synapses kept vs removed (w 0.275, no adaptation): removal does not end persistence." + NOSCRIPT_KK, rel(P_KCKC))
put("alt_drop_kckc_w0275_kc_last200", [kk["w0.275_dropKCKCFalse"]["KC_active_last200"], kk["w0.275_dropKCKCTrue"]["KC_active_last200"]], "KCs active 500-700 ms after offset with KC-KC kept vs removed." + NOSCRIPT_KK, rel(P_KCKC))
put("alt_drop_kckc_w0275_kc_frac_on", kk["w0.275_dropKCKCTrue"]["KC_frac_active_on"], "KC fraction during odour after KC-KC removal (still above 20%)." + NOSCRIPT_KK, rel(P_KCKC))
si = jload(P_SENS)
s1 = si["w0.275_dropSensIn"]
put("alt_drop_sensory_input_w0275_odour_last200", s1["odor_active_last200"], "Odour neurons active 500-700 ms after offset when input to sensory neurons is removed (w 0.275): persistence remains." + NOSCRIPT_SI, rel(P_SENS) + " w0.275_dropSensIn")
put("alt_drop_sensory_input_w0275_kc_last200_and_share", [s1["KC_active_last200"], rnd(s1["KC_active_last200"] / s1["odor_active_last200"], 3)], "KCs among those persistent neurons, and their share." + NOSCRIPT_SI, rel(P_SENS))
put("alt_drop_sensory_input_top_type", [list(s1["persist_top"].keys())[0], list(s1["persist_top"].values())[0]], "Largest persistent type after sensory-input removal and its count." + NOSCRIPT_SI, rel(P_SENS) + " persist_top")
put("alt_drop_sensory_and_kckc_w0275_odour_last200", si["w0.275_dropSensIn+KCKC"]["odor_active_last200"], "Persistent odour neurons with both sensory input and KC-KC removed." + NOSCRIPT_SI, rel(P_SENS))
put("alt_lower_weight_w0157_odour_last200_kcfrac", [kk["w0.157_dropKCKCFalse"]["odor_active_last200"], kk["w0.157_dropKCKCFalse"]["KC_frac_active_on"]], "Lower weight (0.157 mV): odour persistence remains (neurons active 500-700 ms after offset) while KC recruitment becomes sparse." + NOSCRIPT_KK, rel(P_KCKC))
put("alt_adaptation_only_da4", [r4["odor_active_last200"], r4["KC_frac_on"], r4["sugar_MN9_hz"][0]], "Adaptation alone at da 4 mV ends odour persistence but leaves KCs non-sparse and MN9 weak: [odour persistent neurons, KC fraction, MN9 Hz].", rel(P_SWEEP))
put("alt_tests_generating_scripts_found", len(gen), "Number of scripts under code/ that write kckc_test.json, sensory_input_test.json or odor_intensity_sweep.csv (0 = not in the code release).", "grep over code/**/*.py (excluding the facts_* scripts)")
it = pd.read_csv(P_INT)
for da in (0.0, 1.0):
    q = it[it.da == da]
    put(f"intensity_sweep_da{int(da)}_last200_range", [int(q.active_last200.min()), int(q.active_last200.max())], f"odor_intensity_sweep.csv, da {da:g}: range of the active_last200 column (neurons active in the last 200 ms after offset) over the {len(q)} conditions." + NOSCRIPT_INT, rel(P_INT))
    put(f"intensity_sweep_da{int(da)}_kc_frac_range", [float(q.KC_frac.min()), float(q.KC_frac.max())], f"odor_intensity_sweep.csv, da {da:g}: range of KC fraction." + NOSCRIPT_INT, rel(P_INT))
put("intensity_sweep_pn_active_range", [int(it.PN_active.min()), int(it.PN_active.max())], f"Number of active PNs (PN_active column) over all {len(it)} intensity-sweep conditions (35 DM1 ORNs at 20 Hz up to 179 ORNs of 3 glomeruli at 100 Hz): a similar number (~550) of PNs regardless of input. The file holds counts only, not PN identities, so it does not show that the same PNs fire." + NOSCRIPT_INT, rel(P_INT))
put("intensity_sweep_n_orn_range", [int(it.n_orn.min()), int(it.n_orn.max())], "Range of stimulated ORNs in the intensity sweep (n_orn column).", rel(P_INT))
put("intensity_sweep_all_rows_include_DM1", bool(it.glom.str.contains("DM1").all()), "Whether every intensity-sweep condition contains DM1 (glomerulus sets nested: a size/intensity test, not an odour-identity test).", rel(P_INT))
odor_keys = [k for k in pr if k.startswith("odor_")]
put("probe_odour_conditions", len(odor_keys), "Unilateral ORN conditions in the probe (DM1+DM4+VA2, DA2, V on each side).", rel(P_PROBE))
put("probe_odour_n_active_range", [min(pr[k]["n_active"] for k in odor_keys), max(pr[k]["n_active"] for k in odor_keys)], "Neurons active during the 300-ms ORN stimulus (t_off = 0, not persistence) across the six probe odours, base model.", rel(P_PROBE) + " odor_*.n_active")

# =============================================================== 8. feedforward AL and odour A vs B
ff = jload(P_FFAL)
fsrc2 = src(S_FFAL)
ffdef = func_defaults(S_FFAL, "run")
must(r'cls\.isin\(\["ALPN", "ALLN", "ALIN", "ALON"\]\)', fsrc2, "AL classes")
must(r'\^\(ORN_\|HRN_\|TRN_\)', fsrc2, "AL sensory")
put("ffal_al_edge_definition", "edges with both endpoints in classes ALPN/ALLN/ALIN/ALON or types ORN_/HRN_/TRN_ set to 0", "Definition of the removed antennal-lobe-internal chemical synapses (same mask in model.py).", rel(S_FFAL) + f" line {line_of(fsrc2, 'al = cls.isin')}")
put("ffal_calibration_protocol", f"PN drive {ffdef['rate']:g} Hz for 300 ms then 700 ms off, B={ffdef['B']} replicates, seed {ffdef['seed']}, tau_a 1000 ms, w 0.275", "Protocol of test_ff_al.py (the feedforward calibration run); counts are per-replicate means truncated to int.", rel(S_FFAL) + " run() defaults")
f0 = next(r for r in ff if r["variant"] == "ffAL" and r["da"] == 0.0)
f1 = next(r for r in ff if r["variant"] == "ffAL" and r["da"] == 1.0)
put("n_stimulated_PN_odour_A", f1["n_PN_A"], "Uniglomerular PNs driven for odour A (DM1, DM4, VA2; both sides).", rel(P_FFAL) + " n_PN_A")
put("n_stimulated_PN_odour_B", f1["n_PN_B"], "Uniglomerular PNs driven for odour B (DL5, DM2, VM2; both sides).", rel(P_FFAL) + " n_PN_B")
for g, lab in ((odors["A"], "A"), (odors["B"], "B")):
    m = pt.str.match(r"^(%s)_" % "|".join(g))
    put(f"stim_PN_odour_{lab}_by_side", [int((m & (pq.side == "L")).sum()), int((m & (pq.side == "R")).sum())], f"Odour {lab} PNs (types matching the glomeruli) on the left and right (count check against n_PN_{lab}).", rel(P_PARQ))
put("ffal_kc_frac_A_B_da1", [f1["KC_frac_A"], f1["KC_frac_B"]], "Calibration run, feedforward AL with adaptation: mean per-replicate KC fraction for odour A and odour B (B is below the 5% criterion).", rel(P_FFAL) + " ffAL da=1")
put("ffal_kc_frac_A_B_da0", [f0["KC_frac_A"], f0["KC_frac_B"]], "Same without adaptation: adaptation raised, not lowered, KC recruitment in the feedforward AL.", rel(P_FFAL) + " ffAL da=0")
put("ffal_jaccard_da1_da0", [f1["KC_jaccard_AB"], f0["KC_jaccard_AB"]], "KC-set Jaccard overlap between odours A and B (mean over replicates), with and without adaptation.", rel(P_FFAL))
put("ffal_last200_A_da1_da0", [f1["active_last200_A"], f0["active_last200_A"]], "Neurons active 500-700 ms after odour-A offset (mean per replicate, int), with and without adaptation.", rel(P_FFAL) + " active_last200_A")
put("ffal_last200_B_da1_da0", [f1["active_last200_B"], f0["active_last200_B"]], "Same for odour B.", rel(P_FFAL) + " active_last200_B")
put("ffal_mbon_hz_A_da1_da0", [f1["MBON_hz_A"], f0["MBON_hz_A"]], "Mean MBON rate during odour A (Hz) with and without adaptation (feedforward AL).", rel(P_FFAL) + " MBON_hz_A")
put("ffal_active_on_A_da1_da0", [f1["active_on_A"], f0["active_on_A"]], "Neurons active during odour A with and without adaptation (feedforward AL).", rel(P_FFAL))

st = pd.read_csv(P_STAGE)
h = st[(st.stage == 0) & (st.scope == "pan")]
put("healthy_n", int(len(h)), "Healthy (stage 0, pan) flies of the severity series, the healthy reference for the four battery x4 designs (stage_curve, components, vulnerability, scope). The burden-matched comparison (exp2 battery.csv, map 'healthy'), the joint cohort (stage 0) and exp1 (per-fly healthy twins) each have their own healthy flies.", rel(P_STAGE))
_e2 = pd.read_csv(P_EXP2)
_jc = pd.read_csv(P_JOINT)
put("healthy_n_other_designs", {"exp2_burden_matched": int((_e2["map"] == "healthy").sum()), "joint_cohort": int((_jc.stage == 0).sum())}, "Healthy flies of the designs with their own reference: exp2 battery.csv rows with map 'healthy' (analyze_exp2.py) and joint-cohort stage-0 flies (analyze_joint.py).", rel(P_EXP2) + " map; " + rel(P_JOINT) + " stage")
bsrc2 = src(S_BAT)
must(r'pair_idx = np\.concatenate\(\[pnA, pam\]\)', bsrc2, "A paired")
put("protocol_paired_odour", "A", "Odour paired with PAM activation in every fly (no reciprocal or counterbalanced pairing in code/).", rel(S_BAT) + f" line {line_of(bsrc2, 'pair_idx = ')}")
for col, lab, mean in (("KC_frac_A", "kc_frac_A", "fraction of KCs firing >= 1 spike during 1-s odour A at 150 Hz"),
                       ("KC_frac_B", "kc_frac_B", "fraction of KCs firing >= 1 spike during 1-s odour B"),
                       ("V_pre_A", "valence_naive_A", "naive MBON valence for odour A (signed sum of MBON rates, Hz)"),
                       ("V_pre_B", "valence_naive_B", "naive MBON valence for odour B (Hz)"),
                       ("KC_overlap_AB", "kc_jaccard_AB", "Jaccard overlap of the naive KC sets of odours A and B"),
                       ("MN9_sugar_hz", "mn9_sugar_hz", "MN9 rate (mean of the two MN9s) during 0.5-s bilateral sugar (Hz)"),
                       ("GF_loom_hz", "gf_loom_hz", "giant-fibre rate (mean of both DNp01) during 0.3-s left looming (Hz)"),
                       ("PAM_pair_hz", "pam_pair_hz", "mean over PAM types of the type-mean rate during pairing (Hz)")):
    x = h[col].to_numpy(float)
    put(f"healthy_{lab}_median", rnd(np.median(x)), f"Healthy flies (n={len(x)}): median {mean}.", rel(P_STAGE) + f" {col}, stage 0")
    put(f"healthy_{lab}_ci95", rnd(boot_median_ci(x, SEED)), f"95% bootstrap CI of that median.", rel(P_STAGE))
    put(f"healthy_{lab}_range", rnd([np.min(x), np.max(x)]), f"Min and max over healthy flies of the {mean}.", rel(P_STAGE))
put("healthy_kc_frac_B_over_A_ratio_of_medians", rnd(np.median(h.KC_frac_B) / np.median(h.KC_frac_A), 3), "Median KC fraction for odour B divided by that for odour A, healthy flies.", rel(P_STAGE))
put("healthy_kc_frac_B_over_A_ci95", rnd(boot_paired_ratio_ci(h.KC_frac_A, h.KC_frac_B, SEED), 3), "95% bootstrap CI (flies resampled as pairs) of that ratio.", rel(P_STAGE))
put("healthy_n_flies_B_exceeds_A_kc", int((h.KC_frac_B > h.KC_frac_A).sum()), "Healthy flies in which odour B recruited more KCs than odour A.", rel(P_STAGE))
put("healthy_kc_frac_A_vs_B_wilcoxon_p", float(f"{stats.wilcoxon(h.KC_frac_A, h.KC_frac_B).pvalue:.3g}"), "Two-sided Wilcoxon signed-rank p, KC fraction A vs B within healthy flies.", rel(P_STAGE))
put("healthy_valence_A_over_B_ratio_of_medians", rnd(np.median(h.V_pre_A) / np.median(h.V_pre_B), 2), "Median naive valence A divided by median naive valence B, healthy flies.", rel(P_STAGE))
put("healthy_valence_A_vs_B_wilcoxon_p", float(f"{stats.wilcoxon(h.V_pre_A, h.V_pre_B).pvalue:.3g}"), "Two-sided Wilcoxon signed-rank p, naive valence A vs B within healthy flies.", rel(P_STAGE))
pA, pB = h.KC_frac_A.to_numpy(), h.KC_frac_B.to_numpy()
chance = pA * pB / (pA + pB - pA * pB)
put("healthy_kc_jaccard_chance_median", rnd(np.median(chance)), "Median per-fly Jaccard expected for independent KC sets of the observed sizes, pq/(p+q-pq).", rel(P_STAGE))
jr = h.KC_overlap_AB.to_numpy() / chance
put("healthy_kc_jaccard_obs_over_chance_median", rnd(np.median(jr), 3), f"Median per-fly ratio of observed to chance Jaccard: overlap is consistently about {100 * (np.median(jr) - 1):.0f}% above the independent-set expectation (see the keys below), not at chance; the low absolute overlap mostly reflects sparseness.", rel(P_STAGE))
put("healthy_kc_jaccard_obs_over_chance_iqr_range", rnd([np.percentile(jr, 25), np.percentile(jr, 75), jr.min(), jr.max()], 3), "Per-fly observed/chance Jaccard ratio, healthy flies: [25th percentile, 75th percentile, min, max].", rel(P_STAGE))
put("healthy_kc_jaccard_n_flies_above_chance", [int((jr > 1).sum()), int(jr.size)], "Healthy flies whose observed Jaccard exceeds the independent-set expectation, of all healthy flies.", rel(P_STAGE))
put("healthy_kc_jaccard_obs_vs_chance_wilcoxon_p", float(f"{stats.wilcoxon(h.KC_overlap_AB.to_numpy(), chance).pvalue:.3g}"), "Two-sided Wilcoxon signed-rank p, observed vs chance Jaccard within healthy flies.", rel(P_STAGE))
dA = (h.V_post_A - h.V_pre_A).to_numpy(); dB = (h.V_post_B - h.V_pre_B).to_numpy()
put("healthy_dvalence_A_B_LI_medians", rnd([np.median(dA), np.median(dB), np.median(h.LI)], 2), "Healthy medians of training change in valence for odour A, for odour B, and of the learning index (LI is almost entirely the paired-odour change).", rel(P_STAGE))
put("protocol_odour_rate_and_duration_battery_vs_calibration", [[150, 1000], [ffdef["rate"], 300]], "[Hz, ms] of PN drive in the full protocol and in the feedforward calibration run; the code records no reason for the difference.", rel(S_BAT) + f" line {line_of(bsrc2, 'preA = seg')}; " + rel(S_FFAL))

# =============================================================== 9. pathology and vulnerability rank
stg = dataclass_defaults(S_PATH, "Stage")
put("path_dth_max_mV", stg["dth_max"], "Threshold drop at full load (mV); v_th,i = -45 - dth_max * l_i.", rel(S_PATH) + " Stage default")
put("path_syn_loss_max", stg["syn_loss_max"], "Fraction of output contacts lost at full load (binomial thinning, keep p = 1 - 0.6 l_i).", rel(S_PATH) + " Stage default")
put("path_death_max", stg["death_max"], "Removal probability at full load (P = 0.5 l_i); a fixed default, fitted by no script.", rel(S_PATH) + " Stage default")
dm_any = [p for p in glob.glob(str(CODE / "**/*.py"), recursive=True)
          if not Path(p).name.startswith("facts_") and Path(p).name != "pathology.py" and "death_max" in src(p)]
dm_nonzero = [p for p in dm_any if re.search(r"death_max[\"']?\s*[:=]\s*(?!0\.0\b)[0-9]", src(p))]
put("path_death_max_other_scripts", [Path(p).name for p in dm_any], "Scripts other than pathology.py that mention death_max (only to set it to 0.0 for the rule ablations).", "grep death_max over code/**/*.py (excluding the facts_* analysis scripts)")
put("path_death_max_fitting_scripts", len(dm_nonzero), "Scripts other than pathology.py that set death_max to any nonzero value (0 = the 0.5 default was never fitted or varied).", "grep death_max over code/**/*.py (excluding the facts_* analysis scripts)")
psrc = src(S_PATH)
must(r'order = tot\.argsort\(kind="stable"\)', psrc, "stable ties")
must(r'rank\[order\] = np\.arange\(1, n \+ 1\) / n', psrc, "rank def")
must(r'syn = np\.abs\(signed\)', psrc, "signed network")
must(r'self\.vuln = pathology\.hub_vulnerability\(self\.row_ptr, self\.dst, self\.signed\)', msrc, "vuln on signed pack")
put("vuln_rank_definition", "rank/N in (0,1] of in+out |signed| contacts in the signed MaleCNS pack; ties broken by pack index (ascending bodyId, stable argsort), not averaged", "How u_i is computed.", rel(S_PATH) + f" lines {line_of(psrc, 'def hub_vulnerability')}-{line_of(psrc, 'return rank')}")
put("vuln_rank_counts_al_internal_edges", True, "The rank is computed in Connectome.__init__ on the full signed pack, before build_brains zeroes antennal-lobe-internal edges, so those edges count.", rel(S_MODEL) + f" lines {line_of(msrc, 'self.vuln = ')}, {line_of(msrc, 'w.cx.al_edge. = 0.0')}")
put("vuln_rank_excludes_unsigned_outputs", [sc["0"], man["dropped_unsigned_presynaptic"]], "Outputs of these neurons (count) and these edges are absent from the signed pack, so they count neither as the sender's outputs nor as the targets' inputs.", rel(P_MANIFEST))
onl = jload(P_ONL)
vp = onl["vulnerability_percentile"]
put("vuln_rank_dnp01_L_R", [rnd(vp["DNp01_L"]), rnd(vp["DNp01_R"])], "Vulnerability percentile u of the left and right giant fibre (DNp01).", rel(P_ONL) + " vulnerability_percentile")
put("vuln_rank_mn9_L_R", rnd(vp["MN9"]), "Vulnerability percentile u of the two MN9s in pack order (left-annotated, right-annotated).", rel(P_ONL) + " vulnerability_percentile.MN9")
must(r'eligible = ~cx\.sensory', msrc, "eligible")
put("path_eligible_rule", "all neurons whose superclass does not contain 'sensory'", "Eligibility for load: stimulated non-sensory neurons (odour PNs, LPLC2/LC4, PAM) carry load and can be removed; a removed neuron has v_th = inf and cannot be driven by Poisson input.", rel(S_MODEL) + f" line {line_of(msrc, 'eligible = ~cx.sensory')}")
bstats = jload(P_BSTATS)
bur = bstats["series"]["burden"]
kc5, kc6, kc10 = bur["0.5"]["kc_dead_pct"], bur["0.6"]["kc_dead_pct"], bur["1.0"]["kc_dead_pct"]
put("kc_dead_pct_s05_s06_s1", [kc5, kc6, kc10], "Mean percent of KCs removed at s = 0.5, 0.6 and 1.0 (severity series).", rel(P_BSTATS) + " series.burden.kc_dead_pct")
put("kc_loss_17pct_crossing_s_interp", rnd(0.5 + (17.0 - kc5) / (kc6 - kc5) * 0.1, 3), "Severity at which mean KC removal reaches 17% by linear interpolation between s = 0.5 and 0.6 (the figures hard-code 0.55).", rel(P_BSTATS))
put("fig_anchor_hardcoded", float(must(r'ANCHOR = ([0-9.]+)', src(S_FIG), "anchor").group(1)), "Severity anchor hard-coded in the figure script (read off the curve, not fitted).", rel(S_FIG) + f" line {line_of(src(S_FIG), 'ANCHOR = ')}")
put("min_death_max_to_reach_17pct_kc_loss", rnd(0.17 / (kc10 / 100 / stg["death_max"]), 3), "Smallest death_max for which mean KC loss reaches 17% within s <= 1 (17% divided by the mean KC rank implied by 29.41% at s = 1 with death_max 0.5).", rel(P_BSTATS))
put("dead_frac_s1_mean", rnd(float(st[st.stage == 1.0].dead_frac.mean())), "Mean fraction of all neurons removed at s = 1 (below the 0.25 expected if every neuron carried load = rank, because sensory neurons carry none).", rel(P_STAGE))

# =============================================================== 10. stimulated non-sensory populations (M2/F7)
grn_both = pt.isin(["LB3c", "LB3b"])
loomL = pt.isin(["LPLC2", "LC4"]) & (pq.side == "L")
pam = pt.str.startswith("PAM")
put("stim_set_sizes_grn_loomL_pam", [int(grn_both.sum()), int(loomL.sum()), int(pam.sum())], "Battery stimulus sets: sugar GRNs LB3c+LB3b both sides, left LPLC2+LC4, all PAM neurons.", rel(P_PARQ) + " type/side")
put("stim_sets_sensory_flags", {"GRN": bool(sens[grn_both.to_numpy()].all()), "PN": bool(sens[pt.str.match(r'^(DM1|DM4|VA2|DL5|DM2|VM2)_').to_numpy()].any()),
                                "LPLC2_LC4": bool(sens[loomL.to_numpy()].any()), "PAM": bool(sens[pam.to_numpy()].any())},
    "Whether each stimulated set is sensory (exempt from load): only the gustatory neurons are.", rel(P_PARQ) + " superclass")
put("pam_types_n_and_size_range", [int(pt[pam].nunique()), int(pt[pam].value_counts().min()), int(pt[pam].value_counts().max())], "Number of PAM types and smallest/largest type size (neurons).", rel(P_PARQ))
put("ppl1_neurons_and_types", [int(pt.str.startswith("PPL1").sum()), int(pt[pt.str.startswith("PPL1")].nunique())], "PPL1 neurons and types: they define compartments but are never tested for activity.", rel(P_PARQ))
put("n_kc_mbon", [int(pt.str.startswith("KC").sum()), int(pt.str.startswith("MBON").sum())], "Kenyon cells and MBONs in the network.", rel(P_PARQ))
tabs = {"stage_curve": st, "components": pd.read_csv(P_COMP), "vulnerability": pd.read_csv(P_VULN), "scope": pd.read_csv(P_SCOPE)}
allx4 = pd.concat(tabs.values(), ignore_index=True)
spk = allx4.PN_hz_A * 686
put("pn_readout_denominator_686_all_flies", bool((np.abs(spk - spk.round()) < 1e-6).all()), "PN_hz_A x 686 is an integer in every fly of the four x4 tables, so the PN readout averages over 686 ALPN-class neurons of which only 8 are driven.", "results/battery/*_x4.csv PN_hz_A")
pn_tot = (h.PN_hz_A * 686).to_numpy()
n_pnA = f1["n_PN_A"]
put("pn_spikes_healthy_median", rnd(np.median(pn_tot), 1), f"Median total ALPN spikes (all 686 ALPN-class neurons) per healthy fly during 1-s odour A, against {n_pnA} x 150 = {n_pnA * 150:,} expected Poisson events to the driven PNs.", rel(P_STAGE))
put("pn_spikes_healthy_range", rnd([pn_tot.min(), pn_tot.max()], 1), "Min and max total ALPN spikes per healthy fly during 1-s odour A.", rel(P_STAGE))
_segs = re.findall(r"(\w+) = seg\(", bsrc2)
assert _segs and _segs[0] == "preA", "odour A is no longer the first protocol segment"
iso_pn = n_pnA * 150.0 * ISO["150Hz_da1"][0]
put("pn_spikes_healthy_over_isolated_expectation", rnd(np.median(pn_tot) / iso_pn, 3),
    f"Healthy median total ALPN spikes during odour A divided by the spikes {n_pnA} isolated driven neurons with the paper model's adaptation would fire "
    f"({n_pnA} x 150 events x lif_isolated_driven_spikes_per_event['150Hz_da1'] = {iso_pn:.0f}). The numerator also contains non-driven ALPN spikes, so this "
    "is an upper bound for the driven PNs (odour A is the first protocol segment, so their adaptation starts from rest). A value below 1 means "
    "the driven PNs miss more events than adaptation alone explains; the extra loss comes from their synaptic input in the network, which the "
    "saved tables cannot separate.", rel(P_STAGE) + "; this script")
put("pam_healthy_median_over_isolated_expectation", rnd(np.median(h.PAM_pair_hz) / (100.0 * ISO["100Hz_da1"][0]), 3),
    "Healthy median PAM_pair_hz divided by the rate of an isolated neuron driven at 100 Hz with the paper model's adaptation (100 x lif_isolated_driven_spikes_per_event['100Hz_da1']): healthy PAM neurons fire on about every event.", rel(P_STAGE) + "; this script")
s1d = st[st.stage == 1.0]; s05 = st[st.stage == 0.5]
MC_SEEDS = [SEED + i for i in range(1, 21)]
for col, lab in (("PN_hz_A", "pn"), ("PAM_pair_hz", "pam")):
    for g, sl in ((s05, "s05"), (s1d, "s1")):
        put(f"{lab}_rate_{sl}_pct_healthy", rnd(100 * np.median(g[col]) / np.median(h[col]), 1), f"Median {col} at s = {g.stage.iloc[0]:g} as % of the healthy median (severity series).", rel(P_STAGE))
        mc = np.array([[100 * v for v in boot_ratio_ci(g[col], h[col], sd)] for sd in MC_SEEDS])
        spread = float(max(mc[:, 0].max() - mc[:, 0].min(), mc[:, 1].max() - mc[:, 1].min()))
        note = (f" Monte-Carlo unstable: over 20 other bootstrap seeds a bound varies over a range of up to {spread:.1f} points (see the _mc_range key), so quote it to integers."
                if spread >= 1.0 else " Stable to within 1 point over 20 other bootstrap seeds (see the _mc_range key).")
        put(f"{lab}_rate_{sl}_pct_healthy_ci95", rnd([100 * v for v in boot_ratio_ci(g[col], h[col], SEED)], 1), "95% bootstrap CI of that percentage (flies resampled within groups, 4,000 resamples, seed _bootstrap_seed)." + note, rel(P_STAGE))
        put(f"{lab}_rate_{sl}_pct_healthy_ci95_mc_range", rnd([[mc[:, 0].min(), mc[:, 0].max()], [mc[:, 1].min(), mc[:, 1].max()]], 1),
            "Same CI recomputed with seeds _bootstrap_seed + 1 to + 20 (4,000 resamples each): [[lowest, highest lower bound], [lowest, highest upper bound]] (% of healthy).", rel(P_STAGE) + "; this script")
put("pam_healthy_vs_s1_mannwhitney_p", float(f"{stats.mannwhitneyu(h.PAM_pair_hz, s1d.PAM_pair_hz, alternative='two-sided').pvalue:.3g}"), "Two-sided Mann-Whitney p, PAM rate during pairing, healthy vs s = 1.", rel(P_STAGE))
allx4_lab = pd.concat([d.assign(design=k) for k, d in tabs.items()], ignore_index=True)
_im = allx4_lab.PAM_pair_hz.idxmin()
pam_min = float(allx4_lab.PAM_pair_hz.min())
n_pam_types = int(pt[pam].nunique())
put("pam_pair_hz_min_all_x4_flies", rnd(pam_min, 1),
    f"Lowest fly-level PAM_pair_hz (mean over the {n_pam_types} PAM types of each type's mean rate during pairing, removed neurons counting 0) in any fly of the four x4 designs. "
    f"The 5-Hz gate of learn() acts on each type separately, and per-type rates are not saved, so these tables cannot exclude that a type's gate closed "
    f"(e.g. {n_pam_types - 1} types at {pam_min * n_pam_types / (n_pam_types - 1):.1f} Hz and one at 0 give the same mean).", "results/battery/*_x4.csv PAM_pair_hz")
put("pam_pair_hz_min_all_x4_flies_fly", [allx4_lab.loc[_im, "design"], allx4_lab.loc[_im, "condition"], int(allx4_lab.loc[_im, "seed"])], "Fly with that lowest PAM_pair_hz: [design table, condition, fly seed].", "results/battery/*_x4.csv")
cp = tabs["components"]
c1 = cp[cp.stage == 1.0]
put("components_s1_pn_spikes_median_by_condition", {c: rnd(float(np.median(g.PN_hz_A * 686)), 1) for c, g in c1.groupby("condition")}, "Median PN spikes per fly at s = 1 by rule-ablation condition: the PN decline comes from neuron removal (death_only ~ all; synloss_only and hyper_only ~ healthy).", rel(P_COMP))
put("components_s1_pam_hz_median_by_condition", {c: rnd(float(np.median(g.PAM_pair_hz)), 1) for c, g in c1.groupby("condition")}, "Median PAM rate during pairing at s = 1 by rule-ablation condition.", rel(P_COMP))
runs = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(RES / "exp1/screen/*_runs.csv")))], ignore_index=True)
bl = runs[runs.level == "baseline"]
pnfr = {}
for s_, g in bl.groupby("stage"):
    out = {"restore_dead_all": [], "restore_syn_all": []}
    for seed, gg in g.groupby("seed"):
        b = gg.set_index("unit")["PN_hz_A"]
        d = b["healthy"] - b["damaged"]
        if abs(d) > 0.05 * abs(b["healthy"]):
            for u in out:
                out[u].append((b[u] - b["damaged"]) / d)
    pnfr[s_] = {u: [rnd(float(np.median(v)), 3), len(v)] for u, v in out.items()}
put("exp1_pn_rate_restored_fraction_by_stage", {f"s{k:g}": v for k, v in pnfr.items()}, "Within-fly fraction of the PN-rate deficit restored by restoring all removed neurons vs all synapses ([median, n flies with deficit > 5% of healthy]); ~1 vs ~0 = lost stimulus delivery.", "results/exp1/screen/*_runs.csv level=baseline")

# =============================================================== 11. final-model adaptation robustness (exp1 robust, healthy)
rb = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(RES / "exp1/robust/*.csv")))], ignore_index=True)
hb = rb[(rb.level == "baseline") & (rb.unit == "healthy")]
pv = hb.pivot_table(index="seed", columns="model", values=["KC_hz_A", "KC_frac_A", "MN9_sugar_hz", "brain_spikes_A", "GF_loom_hz"])
ratios = {}
for col in ["KC_hz_A", "KC_frac_A", "MN9_sugar_hz", "brain_spikes_A", "GF_loom_hz"]:
    ratios[col] = rnd(float(np.median(pv[(col, "da0.5")] / pv[(col, "base")])), 3)
put("final_model_da_half_over_base_healthy", ratios, "Final model, healthy twins of 12 flies (shortened protocol): median per-fly ratio of readouts with adaptation da halved (0.5 mV) to da 1 mV; adaptation sets KC recruitment, MN9 and whole-brain spikes more than KC rate.", "results/exp1/robust/*.csv level=baseline unit=healthy")
put("final_model_da_half_n_flies", int(len(pv)), "Flies in that comparison.", "results/exp1/robust/*.csv")

# =============================================================== 12. circuit protocol
put("protocol_state_reset_calls_in_battery", len(re.findall(r"reset_state", bsrc2)), "Calls to Brains.reset_state() in assay_battery.py: 0, so v, g, adaptation current and delay buffer carry over between all protocol segments (reset happens once, at construction).", rel(S_BAT))
jsrc = src(S_JOINT)
put("protocol_joint_resets_after_circuit_assays", bool(re.search(r"br\.reset_state\(\)\s+# the circuit assays leave adaptation charged", jsrc)), "The joint cohort resets state after its circuit assays and before the naive-valence probe (comment: 'the circuit assays leave adaptation charged').", rel(S_JOINT) + f" line {line_of(jsrc, 'br.reset_state')}")
hsrc = src(S_HNB)
put("protocol_arena_starts_from_reset_state", bool(re.search(r"br\.reset_state\(\)\n", hsrc)), "HouseNB.run() resets brain state (not weights) before the arena, i.e. after training and the memory test.", rel(S_HNB) + f" line {line_of(hsrc, 'br.reset_state')}")
gaps = [1.0, 2.0]
put("protocol_residual_adaptation_after_gap", rnd([float(np.exp(-g * 1000.0 / module_assign(S_MODEL, 'TAU_A'))) for g in gaps], 3), "Fraction of the adaptation current left after a 1-s and a 2-s gap (exp(-gap/tau_a)), carried into the next segment.", rel(S_MODEL) + " TAU_A; " + rel(S_BAT))
i_pairB = bsrc2.find("gap(2000); seg(pnB, np.full(pnB.size, 150.0), 1000); gap(2000)")
i_learn = bsrc2.find("learn(cx, br, pair, dan_rate)")
assert 0 < i_pairB < i_learn
put("protocol_plasticity_applied_after_training_block", True, "learn() is called once, after pairing (1 s), 2-s gap, odour B (1 s) and 2-s gap; KC activity during pairing only decides which synapses change.", rel(S_BAT) + f" lines {line_of(bsrc2, 'gap.2000.; seg.pnB')}-{line_of(bsrc2, 'learn.cx, br, pair')}")
must(r'np\.full\(pam\.size, 100\.0\)', bsrc2, "PAM 100 Hz")
put("protocol_pam_drive_hz", 100.0, "Poisson drive to every PAM neuron during pairing (Hz).", rel(S_BAT) + f" line {line_of(bsrc2, 'pam.size, 100.0')}")
put("protocol_stim_rates_hz", 150.0, "Poisson drive to odour PNs, sugar GRNs and looming LPLC2/LC4 (Hz).", rel(S_BAT))
ld = func_defaults(S_MODEL, "learn")
put("learn_dan_on_hz", ld["dan_on_hz"], "A PAM type counts as active if its type-mean rate (removed neurons count as 0) during pairing reaches this rate (Hz).", rel(S_MODEL) + " learn() default")
put("learn_eta_and_factor", [ld["eta"], rnd(1 - ld["eta"], 2)], "Depression eta and resulting multiplicative factor on KC->MBON weights (binary: applied in full to every active compartment).", rel(S_MODEL) + " learn() default")
put("learn_fraction_of_type_removed_to_close_gate", rnd(1 - ld["dan_on_hz"] / 99.0, 3), "Fraction of a PAM type's neurons that must be removed before its mean (~99 Hz per surviving neuron) drops below the gate.", rel(S_MODEL) + "; healthy PAM_pair_hz")
must(r'dm = e\[e\.pre_t\.str\.match\(r"\^\(PAM\|PPL1\)"\)', msrc, "comp map")
must(r'tab\.frac >= 0\.10', msrc, "10%")
must(r'self\.pam = np\.flatnonzero\(t\.str\.startswith\("PAM"\)', msrc, "pam only")
put("learn_compartment_rule", "MBON type in a DAN type's compartment if it receives >= 10% of that type's synapses onto MBONs; map built from PAM and PPL1, but only PAM types are tested for activity", "Compartment definition and which DAN types gate plasticity.", rel(S_MODEL) + f" lines {line_of(msrc, 'dm = e')}-{line_of(msrc, 'self.comp = ')}; " + rel(S_BAT) + f" line {line_of(bsrc2, 'dan_types = ')}")
must(r'/ 0\.5 / 2\.0', bsrc2, "MN9 readout"); must(r'gfb\.sum\(1\) / 0\.3 / 2\.0', bsrc2, "GF readout")
put("readout_mn9_definition", "MN9 spikes over the 0.5-s sugar segment / 0.5 s / 2 neurons", "MN9 readout is the mean of both MN9s (right sugar drives only the left-annotated MN9 in the base model).", rel(S_BAT) + f" line {line_of(bsrc2, '/ 0.5 / 2.0')}")
put("readout_gf_definition", "DNp01 spikes over the 0.3-s left looming segment / 0.3 s / 2 neurons; latency = first 1-ms bin with a spike", "Giant-fibre readouts.", rel(S_BAT) + f" lines {line_of(bsrc2, 'gf_rate = ')}-{line_of(bsrc2, 'gf_lat = ')}")

# =============================================================== 13. steering / looming probe
def dn(key, t, s):
    return next(r[2] for r in pr[key]["dn_rates"] if r[0] == t and r[1] == s)

steer = [r[2] for k in odor_keys for r in pr[k]["dn_rates"] if r[0] in ("DNa01", "DNa02")]
put("probe_steering_dn_max_hz", max(steer), "Highest DNa01/DNa02 rate (Hz) over six unilateral ORN stimuli (150 Hz, 300 ms) in the base model with ORN input and recurrent AL; no consistent lateralisation.", rel(P_PROBE) + " odor_*.dn_rates")
put("probe_steering_model", "unmodified base model (da 0, recurrent AL, ORN drive) - not the paper model; no steering-DN test exists for the final model", "Which model the steering probe ran on.", rel(S_PROBE) + " run(): Brains(n_trials) defaults")
for key, lab in (("looming_LPLC2_R", "LPLC2_R"), ("looming_LC4_R", "LC4_R"), ("looming_LC4_LPLC2_R", "LC4_LPLC2_R"), ("looming_LPLC2_LR", "LPLC2_LR")):
    put(f"probe_loom_{lab}_dnp01_L_R_hz", [dn(key, "DNp01", "L"), dn(key, "DNp01", "R")], f"Base model, {lab} at 150 Hz for 300 ms ({pr[key]['n_stim']} neurons): left and right DNp01 rate during the stimulus (Hz).", rel(P_PROBE) + f" {key}")
hsrc_h = src(S_HOUSE)
must(r'the\s+odor gradient direction is given to the body controller', hsrc_h, "gradient to body", flags=re.S)
put("arena_odour_drive_bilateral", True, "Arena odour drive goes to the uniglomerular PNs of both sides at one concentration (cx.upn has no side filter), so the brain receives no left-right odour difference; the gradient direction is given to the body controller.", rel(S_HOUSE) + f" line {line_of(hsrc_h, 'odor gradient direction')}; " + rel(S_MODEL) + f" line {line_of(msrc, 'def upn')}")

# =============================================================== 14. arena parameters
ar = dataclass_defaults(S_HOUSE, "Arena")
bd = dataclass_defaults(S_HOUSE, "Body")
put("arena_radius_mm", ar["R"], "Arena radius (mm).", rel(S_HOUSE) + " Arena.R")
put("arena_food_patches_xyr_mm", [list(f) for f in ar["food"]], "Food patches [x, y, radius] (mm).", rel(S_HOUSE) + " Arena.food")
put("arena_odour_sources_xy_mm", ar["odor_src"], "Odour source positions (mm); 42 mm apart.", rel(S_HOUSE) + " Arena.odor_src")
put("arena_plume_sigma_mm", ar["sigma"], "Plume concentration c = exp(-d^2 / (2 sigma^2)): sigma is a Gaussian SD (mm), not a width.", rel(S_HOUSE) + " Arena.sigma; " + rel(S_HNB) + f" line {line_of(hsrc, 'cA = math.exp')}")
must(r'if cA > math\.exp\(-0\.5\):', hsrc, "plume threshold")
put("arena_inside_plume_threshold", "c > exp(-0.5) = 0.607, i.e. within 1 sigma (14 mm) of the source", "Definition of 'time near trained odour' (time inside the plume of odour A).", rel(S_HNB) + f" line {line_of(hsrc, 'if cA > math.exp')}")
must(r'rate\[K_A\] = 150\.0 \* cA', hsrc, "PN drive")
put("arena_pn_drive", "150 Hz x concentration", "Poisson drive to the odour's uniglomerular PNs (both sides).", rel(S_HNB) + f" line {line_of(hsrc, 'rate.K_A. = 150.0')}")
must(r'rate\[K_GRN\] = 150\.0 \* hunger if \(on_food and grounded\)', hsrc, "sugar drive")
put("arena_sugar_drive", "150 Hz x hunger, only on a food patch and grounded", "Drive to LB3c/LB3b GRNs (both sides).", rel(S_HNB) + f" line {line_of(hsrc, 'rate.K_GRN.')}")
put("arena_hunger_feed_rate_per_s", bd["hunger_feed_rate"], "Hunger decrease per second of feeding (hunger starts at 1, clipped to [0, 1]).", rel(S_HOUSE) + " Body")
put("arena_hunger_recover_per_s", bd["hunger_recover"], "Hunger recovery per second when not feeding.", rel(S_HOUSE) + " Body")
put("arena_feed_mn9_threshold_hz", bd["feed_hz"], "MN9 rate (mean of two, last 100 ms) above which a grounded fly on food feeds.", rel(S_HOUSE) + " Body.feed_hz")
put("arena_body_step_ms", module_assign(S_HOUSE, "DT_BODY"), "Body step (ms); the brain runs 100 steps of 0.1 ms per body step.", rel(S_HOUSE) + " DT_BODY")
must(r'n10 = min\(k \+ 1, 10\)', hsrc, "MN9 10 steps"); must(r'ring = np\.zeros\(\(B, 30,', hsrc, "MBON 30 steps")
put("arena_readout_windows_ms", {"MN9": 100, "GF": 30, "MBON_valence": 300}, "Sliding windows of the arena readouts (MN9 rate, giant-fibre spike count for take-off, MBON valence for steering).", rel(S_HNB) + f" lines {line_of(hsrc, 'n10 = min')}, {line_of(hsrc, 'gf_sp = ')}, {line_of(hsrc, 'ring = np.zeros')}")
put("arena_gf_spikes_for_takeoff", bd["gf_spikes"], "Giant-fibre spikes (both DNp01 pooled) within 30 ms that trigger a take-off.", rel(S_HOUSE) + " Body.gf_spikes")
put("arena_steering_gain_rad_s", bd["odor_gain"], "Steering gain: steer = gain x tanh(learned valence / v_scale) x sin(phi - heading), phi = direction of sum_o c_o (source_o - x); only while walking.", rel(S_HOUSE) + " Body.odor_gain; " + rel(S_HNB) + f" line {line_of(hsrc, 'steer = odor_gain')}")
put("arena_valence_scale_hz", bd["v_scale"], "Valence saturation scale v_scale (Hz) in the steering tanh.", rel(S_HOUSE) + " Body.v_scale")
put("arena_learned_valence", "MBON valence (300 ms) minus the fly's naive valence at the current c_A and c_B (linear interpolation over probe levels 0, 0.25, 0.5, 1.0; held at the 1.0 value above)", "Valence that drives steering.", rel(S_HNB) + f" lines {line_of(hsrc, 'learned valence')}-{line_of(hsrc, 'lval = val - base')}")
put("arena_walk_speed_mm_s", bd["walk_speed"], "Walking speed (mm/s).", rel(S_HOUSE) + " Body.walk_speed")
put("arena_heading_diffusion_rad_sqrt_s", bd["turn_sd"], "Heading diffusion of the correlated random walk (rad/sqrt(s)).", rel(S_HOUSE) + " Body.turn_sd")
put("arena_jump_mm_and_air_s", [bd["jump_mm"], bd["air_s"]], "Jump distance (mm) and airborne time (s).", rel(S_HOUSE) + " Body")
must(r'tsec - last_jump > 1\.0', hsrc, "jump refractory"); must(r'away \+= 0\.5 \* _gauss\(rs\)', hsrc, "jump noise")
put("arena_jump_refractory_s", 1.0, "Minimum time between jumps (s); jumps include spontaneous GF-triggered ones (heading kept when no shadow).", rel(S_HNB) + f" line {line_of(hsrc, 'tsec - last_jump > 1.0')}")
put("arena_jump_direction_noise_rad", 0.5, "SD of Gaussian noise added to the away-from-shadow jump direction (rad).", rel(S_HNB) + f" line {line_of(hsrc, 'away .= 0.5')}")
put("arena_shadow_radius_mm_and_duration_s", [2.0, ar["shadow_rmax"], ar["shadow_dur"]], "Looming shadow grows linearly from 2 mm to shadow_rmax (mm) over shadow_dur (s).", rel(S_HOUSE) + " Arena; " + rel(S_HNB) + f" line {line_of(hsrc, 'rad = 2.0')}")
must(r'if math\.hypot\(dx, dy\) < rad \+ 6\.0:', hsrc, "seen radius")
put("arena_threat_seen_margin_mm", 6.0, "A threat is 'encountered' (denominator of jumps per threat) if the fly is within shadow radius + 6 mm at any step, even if airborne; looming drive is given only when grounded.", rel(S_HNB) + f" line {line_of(hsrc, 'rad . 6.0')}")
put("arena_looming_drive", "150 Hz per threat to LPLC2+LC4 on the side facing the shadow (sin(bearing - heading) > 0 -> left)", "Lateralised looming drive.", rel(S_HNB) + f" lines {line_of(hsrc, 'rate.K_LL. .= 150.0')}-{line_of(hsrc, 'rate.K_LR. .= 150.0')}")
must(r'bd\.hunger_recover, 300\.0\]', hsrc, "rmax")
put("arena_input_rate_cap_hz", 300.0, "Cap on every input rate (Hz).", rel(S_HNB) + f" line {line_of(hsrc, 'bd.hunger_recover, 300.0')}")
must(r'r0 = a\.R \* 0\.8 \* np\.sqrt\(init\.random\(B\)\)', hsrc, "start")
put("arena_start_disc_radius_frac", 0.8, "Flies start uniformly within a disc of 0.8 R with random heading and hunger 1.", rel(S_HNB) + f" line {line_of(hsrc, 'r0 = a.R')}")
put("arena_wall_mm", rnd(ar["R"] - 1.5, 1), "Wall position (R - 1.5 mm); a fly crossing it is put back and turned toward the centre with N(0, 0.6 rad) noise.", rel(S_HNB) + f" line {line_of(hsrc, 'if rr > R - 1.5')}")
pn = func_defaults(S_HNB, "probe_naive", "HouseNB")
put("arena_naive_probe", [list(pn["levels"]), pn["ms"], pn["gap_ms"], pn["rng_seed"]], "Naive-valence probe: [concentration levels, ms per level, gap ms, rng seed]; PN drive 150 Hz x level, run after a state reset and before training.", rel(S_HNB) + " probe_naive() defaults")
put("arena_pam_driven", False, "PAM neurons receive no drive in the arena (only GRN, odour PN and LPLC2/LC4 inputs).", rel(S_HNB) + f" line {line_of(hsrc, 'self.in_idx = ')}")
jm = jload(P_JMETA)
thr = np.array(jm["threats"], float)
must(r'tt = \[\], 6\.0|threats, tt = \[\], 6\.0', jsrc, "first threat")
must(r'tt \+= 10\.0 \+ trng\.random\(\) \* 4\.0', jsrc, "gap rule")
must(r'18\.0 \* np\.sqrt\(trng\.random\(\)\)', jsrc, "radius rule")
put("arena_threat_rule", "first at 6 s, then every 10 + 4U s (U uniform) until 2 s before the end; centres uniform within 18 mm of the arena centre; drawn once from default_rng(seed + 2), shared by all flies", "Threat schedule rule in cohort_joint.py.", rel(S_JOINT) + f" lines {line_of(jsrc, 'threats, tt = ')}-{line_of(jsrc, 'tt .= 10.0')}")
put("arena_threats_n", int(len(thr)), "Threats in the 120-s joint-cohort arena run.", rel(P_JMETA) + " threats")
put("arena_threat_times_s", thr[:, 0].tolist(), "Threat onset times (s).", rel(P_JMETA) + " threats")
put("arena_threat_gap_range_s", rnd([float(np.diff(thr[:, 0]).min()), float(np.diff(thr[:, 0]).max())], 2), "Range of inter-threat intervals (s).", rel(P_JMETA))
put("arena_threat_max_centre_dist_mm", rnd(float(np.hypot(thr[:, 1], thr[:, 2]).max()), 1), "Largest distance of a threat centre from the arena centre (mm).", rel(P_JMETA))
put("joint_seconds_seed_nflies", [jm["seconds"], jm["seed"], jm["n_flies"]], "Joint cohort: arena duration (s), master seed, number of flies.", rel(P_JMETA))
put("joint_stage_list", jm["stage_list"], "Severity:count list of the joint cohort.", rel(P_JMETA) + " stage_list")
jc = pd.read_csv(P_JOINT)
put("joint_threats_seen_mean_median_min_max", rnd([jc.behaviour_threats_seen.mean(), jc.behaviour_threats_seen.median(), jc.behaviour_threats_seen.min(), jc.behaviour_threats_seen.max()], 2), "Threats encountered per fly (of 10): mean, median, min, max.", rel(P_JOINT) + " behaviour_threats_seen")
put("joint_flies_no_threat_seen", int((jc.behaviour_threats_seen == 0).sum()), "Flies that encountered no threat, so jumps per threat is undefined for them.", rel(P_JOINT))
put("joint_flies_pi_undefined", int(jc.behaviour_PI.isna().sum()), "Flies that entered neither plume, so the preference index is undefined.", rel(P_JOINT))
must(r'seed=args\.seed \+ 13 \* i', jsrc, "fly seeds")
put("joint_seeds_rule", "fly seed = 5150 + 13 i; circuit stimulus rng 5151; threat rng 5152; probe rng 3; training rng 1; arena body rng per-fly xorshift from seed 5150", "Seeds of the joint cohort.", rel(S_JOINT) + "; " + rel(S_HNB))

# =============================================================== 15. cohort totals, model schematic of figure_common.py
tot = bstats["totals"]
n_exp2 = int(len(pd.read_csv(P_EXP2)))
put("cohort_totals_battery_stats", tot, "Flies per design in battery_stats.json (the four battery designs and the joint cohort, without the burden-matched comparison).", rel(P_BSTATS) + " totals")
put("cohort_exp2_flies", n_exp2, "Flies of the burden-matched map comparison (exp2 battery rows).", rel(P_EXP2))
put("cohort_total_flies", tot["all"] + n_exp2, "Total simulated flies (total of battery_stats.json + exp2); exp1 re-simulates existing flies and adds none.", rel(P_BSTATS) + "; " + rel(P_EXP2))
figsrc = src(S_FIG)
put("schematic_hardcoded_fly_count", must(r'\\n([0-9,]+) simulated flies', figsrc, "fig count").group(1), "Fly count hard-coded in the 'Readouts' box of the model schematic of figure_common.py (not the published Fig 1A, which make_figures.py draws with the total above).", rel(S_FIG) + f" line {line_of(figsrc, 'simulated flies')}")
# The figure files that figure_common.py writes are not kept, so nothing is recorded about them here;
# the entries below describe its source.
put("figure_common_output_dir", must(r'OUT = ROOT / "(figures/[a-z_]+)"', figsrc, "OUT").group(1), "Output directory set in figure_common.py (the scripts that import the module set their own).", rel(S_FIG) + f" line {line_of(figsrc, 'OUT = ROOT')}")
must(r'RNG\.uniform\(0, 1, 260\)', figsrc, "schematic")
put("schematic_panel_c_is_illustration", True, "Panel C of the model schematic of figure_common.py draws the line u, the line 1-u and 260 uniform random points ('rule illustration'), not model loads; real loads put sensory neurons at 0 and the antihub map is not 1-u.", rel(S_FIG) + f" lines {line_of(figsrc, 'rule illustration')}-{line_of(figsrc, 'RNG.uniform.0, 1, 260')}")
_eb = must(r'"Whole-CNS LIF engine",\s*((?:"[^"]*"\s*)+)\)', figsrc, "engine box").group(1)
eng_box = "".join(re.findall(r'"([^"]*)"', _eb)).replace("\\n", " / ").strip(" /")
put("schematic_engine_box_text", eng_box,"Text of the engine box in the model schematic of figure_common.py: r = 0.9997 sits above the two MaleCNS extensions although it was obtained on FlyWire v630 with the base model.", rel(S_FIG) + f" line {line_of(figsrc, 'matches Brian2')}")
# The schematic of figure_common.py is not a figure of the manuscript. Fig 1 is drawn by make_figures.py,
# which takes its fly count from cohort_total_flies of this table; the count written in the schematic
# leaves out the burden-matched comparison.
#
# No entry of this table is read from the text of the manuscript, which is not part of the
# repository.

# =============================================================== write
FACTS["_meaning"] = MEANING
FACTS["_source"] = SOURCE
missing = [k for k in FACTS if not k.startswith("_") and k not in MEANING]
assert not missing, missing
(OUT / "facts_model.json").write_text(json.dumps(FACTS, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
print(f"wrote {rel(OUT / 'facts_model.json')}: {len(MEANING)} facts")
