"""Engine validation at the input rates of the example output distributed with the published code.

  python scripts/validate_engine_200hz.py

validate_engine_final.py drives the 21 sugar neurons at 150 Hz, the default of the
released model.py, and compares the engine with a local Brian2 run at 150 Hz. The
example output distributed with the published code (results/example/sugarR.parquet,
copied to data/shiu2024/example/) was not produced at 150 Hz: its 21 stimulated
neurons fire at about 197 Hz, and the published notebook states a default of 200 Hz.
This script makes that comparison like for like. For 200 Hz, and for 100 Hz (the
second distributed example, sugarR_100Hz.parquet), it runs the engine with three
input seeds and both refractory settings and compares the per-neuron rates with

  distributed_example_output   the output shipped with the published code
                               (older Brian2; environment file: 2.5.1)
  local_brian2_2.10.1          Brian2 2.10.1 run here with the unmodified published
                               code at the same rate (rerun_brian2_sugarR_rate.py)

engine settings
  gate=True   synaptic arrivals during the refractory period are discarded
              (the setting used throughout the paper)
  gate=False  arrivals accumulate during the refractory period

It also compares the two references with each other, the engine seeds with each
other (the size of the differences expected from the Poisson input alone), and
repeats the 150-Hz run of validate_engine_final.py (seed 7, discarded) to confirm
that this script calls the engine in the same way.

Every run and reference is described by its spikes per trial, active neurons, MN9
rate, the rates of the 21 stimulated neurons (identified by FlyWire ID) and the
standard deviation over the 30 trials of the total and of the MN9 rate.

Writes results/validation/engine_validation_200hz.json and
engine_validation_200hz_rates.npz. It reads, and never writes, the outputs of
validate_engine_final.py (engine_validation_rates.npz) and the 150-Hz Brian2 run
(brian2_rerun/sugarR.parquet).
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CODE = Path(__file__).resolve().parents[1]      # folder that holds adfly/ (as in validate_engine_final.py)
sys.path.insert(0, str(CODE))
from adfly.lif import Brains, poisson_events  # noqa: E402

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir())
PACK = ROOT / "data/pack/flywire_630"
OUT = ROOT / "results/validation"
SUGAR = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
         720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
         720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
         720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
         720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
         720575940611875570]
MN9 = 720575940660219265
N_TRIALS, T_STEPS = 30, 10000
SEEDS = (7, 11, 12)
DIST, LOCAL = "distributed_example_output", "local_brian2_2.10.1"

# input rate (Hz) -> references
RATES = {
    200: [(DIST, ROOT / "data/shiu2024/example/sugarR.parquet"),
          (LOCAL, OUT / "brian2_rerun_200hz/sugarR.parquet")],
    100: [(DIST, ROOT / "data/shiu2024/example/sugarR_100Hz.parquet"),
          (LOCAL, OUT / "brian2_rerun_100hz/sugarR.parquet")],
}

ids = np.load(PACK / "neuron_ids.npy")
pos = {int(f): i for i, f in enumerate(ids)}
stim = np.array([pos[f] for f in SUGAR], np.int32)
mn9 = pos[MN9]


def stim_stats(rate):
    r = rate[stim]
    return {"stim_hz_min": round(float(r.min()), 2), "stim_hz_max": round(float(r.max()), 2),
            "stim_hz_mean": round(float(r.mean()), 2)}


def summary(rate, total_by_trial, mn9_by_trial):
    """rate: per-neuron mean over trials; the two per-trial vectors give the trial-to-trial spread."""
    top21 = set(np.argsort(rate)[::-1][:stim.size].tolist())
    assert abs(float(np.mean(mn9_by_trial)) - float(rate[mn9])) < 1e-6
    return {"spikes_per_trial": round(float(np.mean(total_by_trial)), 1),
            "spikes_per_trial_sd": round(float(np.std(total_by_trial, ddof=1)), 1),
            "active_neurons": int((rate > 0).sum()),
            "mn9_hz": round(float(rate[mn9]), 2),
            "mn9_hz_sd_trials": round(float(np.std(mn9_by_trial, ddof=1)), 2), **stim_stats(rate),
            "stim_are_the_21_highest_rates": top21 == set(stim.tolist())}


def ref_rates(path):
    df = pd.read_parquet(path)
    trials = int(df["trial"].nunique())
    assert trials == N_TRIALS, f"{path}: {trials} trials"
    assert set(df["flywire_id"].unique()) <= set(pos), f"{path}: neuron outside the pack"
    v = np.zeros(len(ids))
    for f, c in (df.groupby("flywire_id").size() / N_TRIALS).items():
        v[pos[f]] = c
    by_trial = df.groupby("trial").size().reindex(range(N_TRIALS), fill_value=0).to_numpy()
    mn9_trial = (df[df["flywire_id"] == MN9].groupby("trial").size()
                 .reindex(range(N_TRIALS), fill_value=0).to_numpy())
    info = summary(v, by_trial, mn9_trial)
    info.update(file=str(Path(path).relative_to(ROOT)), trials=trials, total_spikes=int(len(df)),
                t_max_s=round(float(df["t"].max()), 4))
    return v, info


def compare(y, x):
    """y regressed on x over the neurons active in either (as validate_engine_final.py)."""
    a = (y > 0) | (x > 0)
    return {"pearson_r": round(float(stats.pearsonr(y[a], x[a])[0]), 4),
            "spearman_r": round(float(stats.spearmanr(y[a], x[a])[0]), 4),
            "slope": round(float(np.polyfit(x[a], y[a], 1)[0]), 3),
            "slope_reverse": round(float(np.polyfit(y[a], x[a], 1)[0]), 3),
            "n_compared": int(a.sum())}


def engine(rate_hz, gate, seed):
    br = Brains(N_TRIALS, pack=PACK, gate_refractory_input=gate)
    br.set_groups(np.arange(br.N), br.N)
    br.no_refractory(stim)
    ev = poisson_events(np.random.default_rng(seed),
                        np.full((N_TRIALS, stim.size), float(rate_hz)), T_STEPS, 0.1)
    t0 = time.time()
    counts = br.run(T_STEPS, np.tile(stim, (N_TRIALS, 1)), ev)[:, 0, :]
    wall = time.time() - t0
    rate = counts.mean(0)
    return rate, {"seed": seed, **summary(rate, counts.sum(1), counts[:, mn9]), "wall_s": round(wall, 1)}


import numba  # noqa: E402
import scipy  # noqa: E402

res = {"model": "Shiu et al. 2024 leaky integrate-and-fire on FlyWire v630",
       "neurons": int(len(ids)),
       "stimulus": "21 right sugar GRNs, Poisson at the rate of each block, 30 trials of 1 s",
       "engine": "numba, exact exponential integration at dt = 0.1 ms, Brian2 update order, dense kernel",
       "seeds": list(SEEDS),
       "slope": "engine rate regressed on reference rate (np.polyfit(ref, ours)), linear rates, neurons active in "
                "either run; slope_reverse is the reference regressed on the engine",
       "spread": "spikes_per_trial_sd and mn9_hz_sd_trials are standard deviations over the 30 trials of a run "
                 "(ddof = 1); the standard error of a 30-trial mean is the SD divided by sqrt(30)",
       "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "numba": numba.__version__,
                    "scipy": scipy.__version__, "pandas": pd.__version__},
       "rates": {}}
vec = {"stim_index": stim, "stim_flywire_id": np.array(SUGAR, np.int64),
       "mn9_index": np.array(mn9), "mn9_flywire_id": np.array(MN9, np.int64)}

# ---- the 150-Hz run of validate_engine_final.py, repeated (read-only check)
old = OUT / "engine_validation_rates.npz"
if old.exists():
    saved = np.load(old)["ours_gate1"]
    rate, info = engine(150, True, SEEDS[0])
    res["check_150hz_seed7_discard"] = {
        **info, "max_abs_diff_from_saved_run_hz": float(np.abs(rate - saved).max()),
        "identical_to_saved_run": bool(np.array_equal(rate, saved)),
        "saved_run": "results/validation/engine_validation_rates.npz ours_gate1"}
    print("150-Hz check", res["check_150hz_seed7_discard"], flush=True)

p150 = OUT / "brian2_rerun/sugarR.parquet"
if p150.exists():       # the 150-Hz local Brian2 reference of validate_engine_final.py, with its trial spread
    res["reference_150hz_local_brian2_2.10.1"] = ref_rates(p150)[1]

for hz, ref_files in RATES.items():
    refs, blk = {}, {"references": {}, "reference_vs_reference": {}, "engine_runs": {},
                     "comparisons": {}, "engine_seed_vs_seed": {}}
    for name, path in ref_files:
        if Path(path).exists():
            refs[name], blk["references"][name] = ref_rates(path)
            vec[f"ref_{hz}hz_{'distributed' if name == DIST else 'local_brian2'}"] = refs[name]
    if DIST not in refs:
        continue
    if LOCAL in refs:       # local Brian2 2.10.1 regressed on the distributed output
        blk["reference_vs_reference"][f"{LOCAL} vs {DIST}"] = compare(refs[LOCAL], refs[DIST])
    for gate in (True, False):
        key = f"gate={gate}"
        per_seed = {}
        for seed in SEEDS:
            per_seed[seed], info = engine(hz, gate, seed)
            blk["engine_runs"].setdefault(key, []).append(info)
            vec[f"ours_{hz}hz_gate{int(gate)}_seed{seed}"] = per_seed[seed]
            print(hz, key, info, flush=True)
        for name, ref in refs.items():
            by_seed = [{"seed": s, **compare(per_seed[s], ref)} for s in SEEDS]
            blk["comparisons"][f"{key} vs {name}"] = {
                "seed_7": {k: v for k, v in by_seed[0].items() if k != "seed"}, "by_seed": by_seed}
        blk["engine_seed_vs_seed"][key] = [{"seeds": [a, b], **compare(per_seed[a], per_seed[b])}
                                           for a, b in itertools.combinations(SEEDS, 2)]
    res["rates"][str(hz)] = blk

(OUT / "engine_validation_200hz.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(OUT / "engine_validation_200hz_rates.npz", **vec)
print(json.dumps(res, indent=1))
