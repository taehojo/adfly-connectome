"""Facts table "validation": the engine against the published base model at the
input rate of each reference (FlyWire v630, 21 sugar gustatory receptor neurons, 30 trials of 1 s).

  python code/scripts/facts_validation.py

Nothing is simulated here. The values are read from

  results/validation/engine_validation.json, engine_validation_rates.npz          150 Hz (validate_engine_final.py)
  results/validation/engine_validation_200hz.json, engine_validation_200hz_rates.npz
                                                                                  200 and 100 Hz (validate_engine_200hz.py)
  results/validation/brian2_rerun_<rate>hz/run_info.json                          local Brian2 reruns (rerun_brian2_sugarR_rate.py)
  results/validation/shiu_example_provenance.json                                 history of the published repository (recorded by hand)

and a few are recomputed from the saved per-neuron rate vectors (rates of the 21 stimulated neurons,
identified by FlyWire ID through the index saved by validate_engine_200hz.py; ranges over seeds).

Writes results/facts/facts_validation.json: a flat dict of facts, plus "_meaning" (one sentence per key)
and "_source" (file and field per key), the convention of results/facts/facts_model.json.
No manuscript file and no existing result file is modified.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())  # code/scripts locally, scripts/ on the server
VAL = ROOT / "results/validation"
OUT = ROOT / "results/facts"
OUT.mkdir(parents=True, exist_ok=True)

P150, Z150 = VAL / "engine_validation.json", VAL / "engine_validation_rates.npz"
P200, Z200 = VAL / "engine_validation_200hz.json", VAL / "engine_validation_200hz_rates.npz"

FACTS, MEANING, SOURCE = {}, {}, {}


def put(key, value, meaning, source):
    if key in FACTS:
        raise KeyError(f"duplicate key {key}")
    FACTS[key] = value
    MEANING[key] = meaning
    SOURCE[key] = source


def rel(p):
    return str(Path(p).relative_to(ROOT)).replace("\\", "/")


def rng(xs, k=4):
    return [round(float(min(xs)), k), round(float(max(xs)), k)]


def se_diff(sd_a, sd_b, n=30):
    """Standard error of the difference between two independent means of n trials each."""
    return float(np.sqrt((sd_a ** 2 + sd_b ** 2) / n))


def stim3(v):
    r = np.asarray(v)[STIM]
    return [round(float(r.min()), 1), round(float(r.max()), 1), round(float(r.mean()), 1)]


e150 = json.loads(P150.read_text(encoding="utf-8"))
e200 = json.loads(P200.read_text(encoding="utf-8"))
z150, z200 = np.load(Z150), np.load(Z200)
STIM = z200["stim_index"]
SEEDS = e200["seeds"]
SET = {"gate=True": ("discard", "discarded"), "gate=False": ("accumulate", "accumulated")}
REF = {"distributed_example_output": ("distributed", "the example output distributed with the published code"),
       "local_brian2_2.10.1": ("local", "the local Brian2 2.10.1 run of the unmodified published code")}

# ------------------------------------------------------------------ design
put("val_network", e200["model"], "Network and model of the engine validation (base model, no MaleCNS extensions).", rel(P200) + " model")
put("val_neurons", e200["neurons"], "Neurons in the FlyWire v630 network.", rel(P200) + " neurons")
put("val_n_stimulated", int(STIM.size), "Stimulated sugar gustatory receptor neurons (the 21 FlyWire IDs of the published example notebook).", rel(Z200) + " stim_index")
put("val_trials_and_duration", [30, 1.0], "[trials, seconds per trial] of every run and of every reference.", rel(P200) + " stimulus; references[*].trials")
put("val_seeds", SEEDS, "Poisson input seeds of the engine runs at each rate and refractory setting; references are single unseeded realisations.", rel(P200) + " seeds")
put("val_seed_for_figure", SEEDS[0], "Seed whose 30-trial realisation is quoted first and drawn in S1 Fig; ranges are over the three seeds.", rel(P200) + " comparisons[*].seed_7")
put("val_slope_definition", e200["slope"], "Direction and scale of every regression slope.", rel(P200) + " slope")
put("val_engine_versions", e200["versions"], "Software versions of the engine runs at 200 and 100 Hz.", rel(P200) + " versions")
P_INFO200 = VAL / "brian2_rerun_200hz/run_info.json"
put("val_released_default_rate_hz", json.loads(P_INFO200.read_text(encoding="utf-8"))["r_poi_released_default_hz"],
    "Poisson rate (Hz) set by default in the released model.py, read from its default_params['r_poi'] when the local Brian2 rerun was started.",
    rel(P_INFO200) + " r_poi_released_default_hz")
P_PROV = VAL / "shiu_example_provenance.json"
prov = json.loads(P_PROV.read_text(encoding="utf-8"))
assert prov["released_model_py"]["r_poi_hz"] == FACTS["val_released_default_rate_hz"]
put("val_prov_repository_commit", [prov["server_copy"]["commit"][:7], prov["server_copy"]["commit_date"]],
    "Commit and date of the copy of the published code used here (unmodified).", rel(P_PROV) + " server_copy")
put("val_prov_notebook_statement", prov["example_notebook"]["statement_in_current_version"],
    "Sentence of the example notebook distributed with the code about the input rate of its first example (the call that writes sugarR.parquet has no rate argument).",
    rel(P_PROV) + " example_notebook")
put("val_prov_notebook_statement_commit", [prov["example_notebook"]["commit_that_introduced_the_statement"]["date"], prov["example_notebook"]["commit_that_introduced_the_statement"]["message"]],
    "Date and message of the commit that first stated the 200-Hz default in the notebook.", rel(P_PROV) + " example_notebook.commit_that_introduced_the_statement")
put("val_prov_model_py_never_200", all(h.get("r_poi_hz", 150) == 150 and not h.get("r_poi_line_changed", False) for h in prov["released_model_py"]["history"]),
    "Whether every committed version of model.py sets r_poi = 150 Hz (two commits, 2023-04-30 and 2023-05-09).", rel(P_PROV) + " released_model_py.history")
put("val_prov_distributed_output_commit_date", prov["distributed_output_history"][-1]["date"],
    "Date of the last commit that changed the distributed sugarR.parquet.", rel(P_PROV) + " distributed_output_history")
put("val_prov_notebook_output_is_distributed_file", prov["notebook_with_stored_outputs"]["matches"],
    "Whether the output stored in the notebook of that commit for the call without a rate argument (511,566 spikes; five listed rates of stimulated neurons) equals the distributed sugarR.parquet.",
    rel(P_PROV) + " notebook_with_stored_outputs")
put("val_prov_environment_brian2", "2.5.1", "Brian2 version in the environment files distributed with the code (" + prov["environment_file"]["environment.yml"] + ").", rel(P_PROV) + " environment_file")

# ------------------------------------------------------------------ which rate produced each output
put("val_stim_hz_distributed_sugarR", stim3(z200["ref_200hz_distributed"]),
    "[min, max, mean] firing rate (Hz) of the 21 stimulated neurons, identified by FlyWire ID, in the distributed sugarR.parquet: the input was 200 Hz, not the 150-Hz default of the released code.",
    rel(Z200) + " ref_200hz_distributed[stim_index]")
put("val_stim_are_21_highest_distributed_sugarR", e200["rates"]["200"]["references"]["distributed_example_output"]["stim_are_the_21_highest_rates"],
    "Whether the 21 stimulated neurons are the 21 highest-rate neurons of the distributed sugarR.parquet.", rel(P200) + " rates.200.references")
if "ref_100hz_distributed" in z200:
    put("val_stim_hz_distributed_sugarR_100Hz", stim3(z200["ref_100hz_distributed"]),
        "[min, max, mean] firing rate (Hz) of the 21 stimulated neurons in the distributed sugarR_100Hz.parquet (input 100 Hz, as its name says).",
        rel(Z200) + " ref_100hz_distributed[stim_index]")
put("val_stim_hz_local_brian2_150", stim3(z150["ref_local_brian2_2.10.1"]),
    "[min, max, mean] rate (Hz) of the 21 stimulated neurons in the local Brian2 2.10.1 run at the released default (150 Hz).",
    rel(Z150) + " ref_local_brian2_2.10.1[stim_index]")
put("val_stim_hz_engine_150_discard_seed7", stim3(z150["ours_gate1"]),
    "[min, max, mean] rate (Hz) of the 21 stimulated neurons in the engine run at 150 Hz (seed 7, refractory input discarded).",
    rel(Z150) + " ours_gate1[stim_index]")
put("val_stim_spikes_per_event", {str(hz): round(float(np.asarray(z200[f"ref_{hz}hz_distributed"])[STIM].mean()) / hz, 3)
                                  for hz in (200, 100) if f"ref_{hz}hz_distributed" in z200},
    "Mean rate of the 21 stimulated neurons in the distributed outputs divided by the nominal input rate (a driven neuron fires slightly less than once per Poisson event).",
    rel(Z200))

# ------------------------------------------------------------------ the comparison at unequal input rates (kept for the record)
c = e150["comparisons"]["gate=True vs released_example_output"]
put("val_mismatched_engine150_vs_distributed200", [c["pearson_r"], c["spearman_r"], c["slope"], c["n_compared"]],
    "Comparison at unequal input rates, NOT like for like: engine at 150 Hz (seed 7, discarded) against the distributed sugarR.parquet, which was generated at 200 Hz: [Pearson r, Spearman r, slope, n].",
    rel(P150) + " comparisons[gate=True vs released_example_output]")
c = e150["comparisons"]["gate=False vs released_example_output"]
put("val_mismatched_engine150_accumulate_vs_distributed200", [c["pearson_r"], c["spearman_r"], c["slope"], c["n_compared"]],
    "Comparison at unequal input rates, NOT like for like: engine at 150 Hz (seed 7, accumulated) against the distributed sugarR.parquet (200 Hz): [Pearson r, Spearman r, slope, n].",
    rel(P150) + " comparisons[gate=False vs released_example_output]")
r = e150["references"]
put("val_ref_local_brian2_150", [r["local_brian2_2.10.1"]["spikes_per_trial"], r["local_brian2_2.10.1"]["active_neurons"], r["local_brian2_2.10.1"]["mn9_hz"]],
    "Local Brian2 2.10.1 at 150 Hz: [spikes per trial, active neurons, MN9 Hz].", rel(P150) + " references")
if "reference_150hz_local_brian2_2.10.1" in e200 and "check_150hz_seed7_discard" in e200:
    a, k = e200["reference_150hz_local_brian2_2.10.1"], e200["check_150hz_seed7_discard"]
    put("val_ref_local_brian2_150_trial_sd", [a["spikes_per_trial_sd"], a["mn9_hz_sd_trials"]],
        "Local Brian2 2.10.1 at 150 Hz: standard deviation over the 30 trials of [spikes per trial, MN9 rate in Hz].", rel(P200) + " reference_150hz_local_brian2_2.10.1")
    put("val_engine_150_discard_seed7_trial_sd", [k["spikes_per_trial_sd"], k["mn9_hz_sd_trials"]],
        "Engine at 150 Hz (seed 7, discarded): standard deviation over the 30 trials of [spikes per trial, MN9 rate in Hz].", rel(P200) + " check_150hz_seed7_discard")
    put("val_engine_150_discard_seed7_diff_in_se_vs_local", [round((k["spikes_per_trial"] - a["spikes_per_trial"]) / se_diff(k["spikes_per_trial_sd"], a["spikes_per_trial_sd"]), 2),
                                                              round((k["mn9_hz"] - a["mn9_hz"]) / se_diff(k["mn9_hz_sd_trials"], a["mn9_hz_sd_trials"]), 2)],
        "Engine (seed 7, discarded) minus local Brian2 at 150 Hz, in standard errors of the difference between two 30-trial means: [spikes per trial, MN9 rate].", rel(P200))
if "check_150hz_seed7_discard" in e200:
    k = e200["check_150hz_seed7_discard"]
    put("val_check_150hz_rerun_identical", [k["identical_to_saved_run"], k["max_abs_diff_from_saved_run_hz"]],
        "validate_engine_200hz.py repeated the 150-Hz run of validate_engine_final.py (seed 7, discarded): [per-neuron rates identical to the saved run, largest absolute difference in Hz].",
        rel(P200) + " check_150hz_seed7_discard")

# ------------------------------------------------------------------ like-for-like comparisons at each rate
for hz in sorted(e200["rates"], key=int, reverse=True):
    blk = e200["rates"][hz]
    for rk, info in blk["references"].items():
        lab, what = REF[rk]
        put(f"val_ref_{lab}_{hz}", [info["spikes_per_trial"], info["active_neurons"], info["mn9_hz"]],
            f"Reference at {hz} Hz, {what}: [spikes per trial, active neurons, MN9 Hz] (single realisation of 30 trials).",
            rel(P200) + f" rates.{hz}.references[{rk}]")
        put(f"val_ref_{lab}_{hz}_trial_sd", [info["spikes_per_trial_sd"], info["mn9_hz_sd_trials"]],
            f"Same reference: standard deviation over its 30 trials of [spikes per trial, MN9 rate in Hz]; the standard error of the 30-trial mean is this divided by 5.48.",
            rel(P200) + f" rates.{hz}.references[{rk}]")
        if lab == "distributed" and hz == "200":
            assert info["total_spikes"] == prov["notebook_with_stored_outputs"]["stored_output_of_the_default_call"]["df_spike_rows"]
            assert info["active_neurons"] == prov["notebook_with_stored_outputs"]["stored_output_of_the_default_call"]["df_rate_rows"]
        put(f"val_ref_{lab}_{hz}_total_spikes", info["total_spikes"], f"Total spikes in the 30 trials of that reference ({info['file']}).",
            rel(P200) + f" rates.{hz}.references[{rk}].total_spikes")
        if lab == "local":
            put(f"val_stim_hz_local_brian2_{hz}", [info["stim_hz_min"], info["stim_hz_max"], info["stim_hz_mean"]],
                f"[min, max, mean] rate (Hz) of the 21 stimulated neurons in the local Brian2 2.10.1 run at {hz} Hz.",
                rel(P200) + f" rates.{hz}.references[{rk}]")
    for ck, cc in blk["reference_vs_reference"].items():
        put(f"val_refcmp_local_vs_distributed_{hz}", [cc["pearson_r"], cc["spearman_r"], cc["slope"], cc["slope_reverse"], cc["n_compared"]],
            f"Local Brian2 2.10.1 at {hz} Hz against the distributed output at {hz} Hz, two independent realisations: [Pearson r, Spearman r, slope local-on-distributed, reverse slope, n neurons active in either].",
            rel(P200) + f" rates.{hz}.reference_vs_reference[{ck}]")
        a, b = blk["references"]["local_brian2_2.10.1"], blk["references"]["distributed_example_output"]
        put(f"val_refdiff_local_minus_distributed_{hz}", [round(a["spikes_per_trial"] - b["spikes_per_trial"], 1), a["active_neurons"] - b["active_neurons"], round(a["mn9_hz"] - b["mn9_hz"], 2)],
            f"Local Brian2 2.10.1 minus distributed output at {hz} Hz: [spikes per trial, active neurons, MN9 Hz].", rel(P200) + f" rates.{hz}.references")
        put(f"val_refdiff_local_minus_distributed_{hz}_in_se", [round((a["spikes_per_trial"] - b["spikes_per_trial"]) / se_diff(a["spikes_per_trial_sd"], b["spikes_per_trial_sd"]), 2),
                                                                 round((a["mn9_hz"] - b["mn9_hz"]) / se_diff(a["mn9_hz_sd_trials"], b["mn9_hz_sd_trials"]), 2)],
            f"The same differences at {hz} Hz in standard errors of the difference between two 30-trial means: [spikes per trial, MN9 rate].", rel(P200) + f" rates.{hz}.references")
    for gk, runs in blk["engine_runs"].items():
        lab, done = SET[gk]
        src = rel(P200) + f" rates.{hz}.engine_runs[{gk}]"
        put(f"val_engine_{hz}_{lab}_spikes_per_trial_by_seed", [x["spikes_per_trial"] for x in runs], f"Total spikes per 1-s trial, engine at {hz} Hz with refractory input {done}, seeds {SEEDS}.", src)
        put(f"val_engine_{hz}_{lab}_mn9_hz_by_seed", [x["mn9_hz"] for x in runs], f"MN9 rate (Hz), engine at {hz} Hz with refractory input {done}, seeds {SEEDS}.", src)
        put(f"val_engine_{hz}_{lab}_active_by_seed", [x["active_neurons"] for x in runs], f"Neurons with nonzero mean rate, engine at {hz} Hz with refractory input {done}, seeds {SEEDS}.", src)
        put(f"val_engine_{hz}_{lab}_stim_hz_seed{SEEDS[0]}", [runs[0]["stim_hz_min"], runs[0]["stim_hz_max"], runs[0]["stim_hz_mean"]],
            f"[min, max, mean] rate (Hz) of the 21 stimulated neurons, engine at {hz} Hz, seed {SEEDS[0]}, refractory input {done}.", src)
        put(f"val_engine_{hz}_{lab}_trial_sd_by_seed", [[x["spikes_per_trial_sd"] for x in runs], [x["mn9_hz_sd_trials"] for x in runs]],
            f"Standard deviation over the 30 trials, engine at {hz} Hz with refractory input {done}: [spikes per trial by seed, MN9 rate (Hz) by seed].", src)
        put(f"val_engine_{hz}_{lab}_seed_spread", [round(max(x["spikes_per_trial"] for x in runs) - min(x["spikes_per_trial"] for x in runs), 1),
                                                  round(max(x["mn9_hz"] for x in runs) - min(x["mn9_hz"] for x in runs), 2),
                                                  max(x["active_neurons"] for x in runs) - min(x["active_neurons"] for x in runs)],
            f"Range (max minus min) over the three seeds, engine at {hz} Hz with refractory input {done}: [spikes per trial, MN9 Hz, active neurons].", src)
        put(f"val_engine_{hz}_{lab}_wall_s_by_seed", [x["wall_s"] for x in runs], f"Wall time (s) of each 30-trial engine run at {hz} Hz, refractory input {done} (the first run of a session includes compilation).", src)
        for rk, info in blk["references"].items():
            rlab, what = REF[rk]
            put(f"val_engine_{hz}_{lab}_mn9_minus_{rlab}_by_seed", [round(x["mn9_hz"] - info["mn9_hz"], 2) for x in runs],
                f"Engine minus reference MN9 rate (Hz) for each seed: engine at {hz} Hz, refractory input {done}, against {what}.", src)
            put(f"val_engine_{hz}_{lab}_spikes_minus_{rlab}_by_seed", [round(x["spikes_per_trial"] - info["spikes_per_trial"], 1) for x in runs],
                f"Engine minus reference spikes per trial for each seed: engine at {hz} Hz, refractory input {done}, against {what}.", src)
            put(f"val_engine_{hz}_{lab}_diff_in_se_vs_{rlab}_by_seed",
                [[round((x["spikes_per_trial"] - info["spikes_per_trial"]) / se_diff(x["spikes_per_trial_sd"], info["spikes_per_trial_sd"]), 2) for x in runs],
                 [round((x["mn9_hz"] - info["mn9_hz"]) / se_diff(x["mn9_hz_sd_trials"], info["mn9_hz_sd_trials"]), 2) for x in runs]],
                f"Those differences in standard errors of the difference between two 30-trial means (trial SDs of both runs): [spikes per trial by seed, MN9 rate by seed]; engine at {hz} Hz, refractory input {done}, against {what}.", src)
            put(f"val_engine_{hz}_{lab}_spikes_ratio_to_{rlab}_by_seed", [round(x["spikes_per_trial"] / info["spikes_per_trial"], 4) for x in runs],
                f"Engine spikes per trial divided by the reference's, for each seed: engine at {hz} Hz, refractory input {done}, against {what}.", src)
        sv = blk["engine_seed_vs_seed"][gk]
        put(f"val_engine_{hz}_{lab}_seed_vs_seed_pearson_range", rng([x["pearson_r"] for x in sv]),
            f"[min, max] Pearson r between the per-neuron rates of two engine seeds at {hz} Hz (refractory input {done}; three pairs): the agreement expected between two input realisations of the same simulator.",
            rel(P200) + f" rates.{hz}.engine_seed_vs_seed[{gk}]")
        put(f"val_engine_{hz}_{lab}_seed_vs_seed_slope_range", rng([x["slope"] for x in sv] + [x["slope_reverse"] for x in sv], 3),
            f"[min, max] slope between two engine seeds at {hz} Hz (both directions, three pairs), refractory input {done}.",
            rel(P200) + f" rates.{hz}.engine_seed_vs_seed[{gk}]")
    for ck, cc in blk["comparisons"].items():
        gk, rk = ck.split(" vs ")
        lab, done = SET[gk]
        rlab, what = REF[rk]
        s7, by = cc["seed_7"], cc["by_seed"]
        src = rel(P200) + f" rates.{hz}.comparisons[{ck}]"
        put(f"val_cmp_engine_{hz}_{lab}_vs_{rlab}_seed{SEEDS[0]}", [s7["pearson_r"], s7["spearman_r"], s7["slope"], s7["slope_reverse"], s7["n_compared"]],
            f"Engine at {hz} Hz (seed {SEEDS[0]}, refractory input {done}) against {what} at {hz} Hz: [Pearson r, Spearman r, slope engine-on-reference, reverse slope, n neurons active in either run].", src)
        put(f"val_cmp_engine_{hz}_{lab}_vs_{rlab}_pearson_range", rng([x["pearson_r"] for x in by]), f"[min, max] of that Pearson r over seeds {SEEDS}.", src + ".by_seed")
        put(f"val_cmp_engine_{hz}_{lab}_vs_{rlab}_slope_range", rng([x["slope"] for x in by], 3), f"[min, max] of that slope (engine on reference) over seeds {SEEDS}.", src + ".by_seed")
        put(f"val_cmp_engine_{hz}_{lab}_vs_{rlab}_n_range", [min(x["n_compared"] for x in by), max(x["n_compared"] for x in by)], f"[min, max] number of neurons compared over seeds {SEEDS}.", src + ".by_seed")
    # recomputed from the saved vectors (seed 7), as an independent check of the JSON
    for gk in blk["engine_runs"]:
        lab, done = SET[gk]
        o = z200[f"ours_{hz}hz_gate{int(gk == 'gate=True')}_seed{SEEDS[0]}"]
        for rk in blk["references"]:
            rlab, what = REF[rk]
            rf = z200[f"ref_{hz}hz_{'distributed' if rlab == 'distributed' else 'local_brian2'}"]
            m = (o > 0) | (rf > 0)
            got = [round(float(stats.pearsonr(o[m], rf[m])[0]), 4), round(float(np.polyfit(rf[m], o[m], 1)[0]), 3), int(m.sum())]
            want = blk["comparisons"][f"{gk} vs {rk}"]["seed_7"]
            assert got == [want["pearson_r"], want["slope"], want["n_compared"]], (hz, gk, rk, got, want)
            put(f"val_cmp_engine_{hz}_{lab}_vs_{rlab}_seed{SEEDS[0]}_log_rmsd",
                [round(float(np.sqrt(np.mean(np.log10(np.maximum(o[m], 1 / 30) / np.maximum(rf[m], 1 / 30)) ** 2))), 3), int(((o > 0) != (rf > 0))[m].sum()),
                 round(float(np.maximum(o, rf)[(o > 0) != (rf > 0)].max()), 2)],
                f"Agreement on the logarithmic scale of S1 Fig, engine at {hz} Hz (seed {SEEDS[0]}, {done}) against {what}: [root-mean-square difference of log10 rate over the neurons active in either run, with silence set to one spike in 30 trials; number of neurons active in only one of the two; highest rate (Hz) among those neurons].",
                rel(Z200))

# ------------------------------------------------------------------ local Brian2 reruns
for hz in (200, 100):
    p = VAL / f"brian2_rerun_{hz}hz/run_info.json"
    if p.exists():
        info = json.loads(p.read_text(encoding="utf-8"))
        put(f"val_brian2_rerun_{hz}_info", info, f"How the local Brian2 reference at {hz} Hz was produced (unmodified published code, r_poi passed through its parameter dictionary).", rel(p))

FACTS["_meaning"] = MEANING
FACTS["_source"] = SOURCE
(OUT / "facts_validation.json").write_text(json.dumps(FACTS, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
print(f"wrote {rel(OUT / 'facts_validation.json')}: {len(FACTS) - 2} facts")
